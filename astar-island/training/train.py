"""
Astar Island — XGBoost training pipeline for terrain prediction.

Trains 6 separate XGBoost models (one per terrain class) to predict
probability distributions from spatial features. Uses leave-one-round-out
cross-validation to estimate generalization. Exports model as JSON lookup
table compatible with the Cloud Run predictor.

Usage:
  # Local (with cache/ dir):
  python train.py --cache-dir ../cache --output model_lookup.json

  # On GCP VM (download from GCS first):
  python train.py --gcs-bucket gs://ainm26osl-710-astar/cache/ --output model_lookup.json
"""

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor

NC = 6  # number of terrain classes
TTC = {10: 0, 11: 0, 0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5}
PROB_FLOOR = 0.001

FEATURE_NAMES = [
    "ic_0", "ic_1", "ic_2", "ic_3", "ic_4", "ic_5",  # one-hot init class
    "food", "coastal", "sett_dist", "sett_dist_inv",
    "n_sett", "total_sett_norm",
    "local_density",  # settlement density in radius 3
    "forest_density",  # forest density in radius 3
    "ocean_adj",  # number of ocean neighbors
    "row_norm", "col_norm",  # normalized position
    # interaction features
    "food_x_coastal", "food_x_sett_dist_inv",
    "coastal_x_sett_near", "ic1_x_food",
    "edge_dist",  # min distance to map edge
]


def cc(c):
    return TTC.get(c, 0)


def extract_features(grid, H, W):
    """Extract feature matrix (H*W, n_features) from a grid."""
    g = np.array(grid, dtype=np.int32)

    # Init class map
    ic_map = np.zeros((H, W), dtype=np.int32)
    for code, cls in TTC.items():
        ic_map[g == code] = cls

    # One-hot init class
    ic_onehot = np.zeros((H, W, NC), dtype=np.float32)
    for c in range(NC):
        ic_onehot[:, :, c] = (ic_map == c).astype(np.float32)

    # Ocean mask
    ocean = (g == 10)

    # Coastal: land cell adjacent to ocean (4-connected)
    padded = np.pad(ocean, 1, constant_values=False)
    coastal = np.zeros((H, W), dtype=bool)
    for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        coastal |= padded[1 + dy:H + 1 + dy, 1 + dx:W + 1 + dx]
    coastal = coastal & ~ocean & (g != 5)
    coastal_float = coastal.astype(np.float32)

    # Ocean neighbor count (8-connected)
    ocean_float = ocean.astype(np.float32)
    ocean_adj = np.zeros((H, W), dtype=np.float32)
    for dy in range(-1, 2):
        for dx in range(-1, 2):
            if dy == 0 and dx == 0:
                continue
            shifted = np.zeros_like(ocean_float)
            sy = slice(max(0, -dy), min(H, H - dy))
            sx = slice(max(0, -dx), min(W, W - dx))
            ty = slice(max(0, dy), min(H, H + dy))
            tx = slice(max(0, dx), min(W, W + dx))
            shifted[ty, tx] = ocean_float[sy, sx]
            ocean_adj += shifted

    # Food potential: count forest neighbors (8-connected)
    forest = (g == 4).astype(np.float32)
    food = np.zeros((H, W), dtype=np.float32)
    for dy in range(-1, 2):
        for dx in range(-1, 2):
            if dy == 0 and dx == 0:
                continue
            shifted = np.zeros_like(forest)
            sy = slice(max(0, -dy), min(H, H - dy))
            sx = slice(max(0, -dx), min(W, W - dx))
            ty = slice(max(0, dy), min(H, H + dy))
            tx = slice(max(0, dx), min(W, W + dx))
            shifted[ty, tx] = forest[sy, sx]
            food += shifted
    food = np.minimum(food, 8)

    # Forest density in radius 3
    forest_density = np.zeros((H, W), dtype=np.float32)
    for dy in range(-3, 4):
        for dx in range(-3, 4):
            if dy == 0 and dx == 0:
                continue
            shifted = np.zeros_like(forest)
            sy = slice(max(0, -dy), min(H, H - dy))
            sx = slice(max(0, -dx), min(W, W - dx))
            ty = slice(max(0, dy), min(H, H + dy))
            tx = slice(max(0, dx), min(W, W + dx))
            shifted[ty, tx] = forest[sy, sx]
            forest_density += shifted
    forest_density /= 48.0  # normalize by max possible neighbors

    # Settlement distance: BFS from all settlements
    sett_mask = (g == 1) | (g == 2)
    sd = np.full((H, W), 999, dtype=np.float32)
    if sett_mask.any():
        ys, xs = np.mgrid[0:H, 0:W]
        sett_ys, sett_xs = np.where(sett_mask)
        for sy, sx in zip(sett_ys, sett_xs):
            dist = np.abs(ys - sy) + np.abs(xs - sx)
            sd = np.minimum(sd, dist.astype(np.float32))

    sd_inv = 1.0 / (1.0 + sd)  # inverse distance, more useful for models

    # Neighbor settlements (radius 2)
    sett_float = sett_mask.astype(np.float32)
    nsett = np.zeros((H, W), dtype=np.float32)
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            if dy == 0 and dx == 0:
                continue
            shifted = np.zeros_like(sett_float)
            sy = slice(max(0, -dy), min(H, H - dy))
            sx = slice(max(0, -dx), min(W, W - dx))
            ty = slice(max(0, dy), min(H, H + dy))
            tx = slice(max(0, dx), min(W, W + dx))
            shifted[ty, tx] = sett_float[sy, sx]
            nsett += shifted

    # Total settlements normalized
    total_sett = float(sett_mask.sum())
    total_sett_norm = total_sett / (H * W)

    # Local density: settlements in radius 3
    local_density = np.zeros((H, W), dtype=np.float32)
    for dy in range(-3, 4):
        for dx in range(-3, 4):
            if dy == 0 and dx == 0:
                continue
            shifted = np.zeros_like(sett_float)
            sy = slice(max(0, -dy), min(H, H - dy))
            sx = slice(max(0, -dx), min(W, W - dx))
            ty = slice(max(0, dy), min(H, H + dy))
            tx = slice(max(0, dx), min(W, W + dx))
            shifted[ty, tx] = sett_float[sy, sx]
            local_density += shifted
    local_density /= 48.0

    # Position features
    row_norm = np.repeat(np.linspace(0, 1, H).reshape(H, 1), W, axis=1).astype(np.float32)
    col_norm = np.repeat(np.linspace(0, 1, W).reshape(1, W), H, axis=0).astype(np.float32)

    # Edge distance
    ys, xs = np.mgrid[0:H, 0:W]
    edge_dist = np.minimum(
        np.minimum(ys, H - 1 - ys),
        np.minimum(xs, W - 1 - xs)
    ).astype(np.float32) / max(H, W)

    # Interaction features
    food_x_coastal = food * coastal_float
    food_x_sett_dist_inv = food * sd_inv
    coastal_x_sett_near = coastal_float * (sd <= 3).astype(np.float32)
    ic1_x_food = ic_onehot[:, :, 1] * food  # settlement x food

    # Stack all features: (H, W, n_features)
    features = np.stack([
        ic_onehot[:, :, 0], ic_onehot[:, :, 1], ic_onehot[:, :, 2],
        ic_onehot[:, :, 3], ic_onehot[:, :, 4], ic_onehot[:, :, 5],
        food, coastal_float, sd, sd_inv,
        nsett, np.full((H, W), total_sett_norm, dtype=np.float32),
        local_density, forest_density, ocean_adj,
        row_norm, col_norm,
        food_x_coastal, food_x_sett_dist_inv,
        coastal_x_sett_near, ic1_x_food,
        edge_dist,
    ], axis=-1)

    return features.reshape(H * W, -1), ic_map.reshape(H * W)


def load_data(cache_dir):
    """Load all GT files and extract features + targets."""
    cache = Path(cache_dir)
    rounds = set()
    for f in cache.glob("r*_gt_s*.json"):
        r = f.name.split("_gt_")[0]
        rounds.add(r)
    rounds = sorted(rounds)
    print(f"Found {len(rounds)} rounds: {rounds}")

    all_data = []  # list of (round_id, features, targets, ic_map)

    for r in rounds:
        # Load init to get grid for each seed
        for s in range(5):
            gt_path = cache / f"{r}_gt_s{s}.json"
            if not gt_path.exists():
                print(f"  Skipping {gt_path.name} (not found)")
                continue

            with open(gt_path) as f:
                gt_data = json.load(f)

            grid = gt_data["initial_grid"]
            H, W = gt_data["height"], gt_data["width"]
            ground_truth = np.array(gt_data["ground_truth"], dtype=np.float64)

            features, ic_map = extract_features(grid, H, W)
            targets = ground_truth.reshape(H * W, NC)

            all_data.append({
                "round": r,
                "seed": s,
                "features": features,
                "targets": targets,
                "ic_map": ic_map,
                "H": H, "W": W,
                "grid": grid,
            })
            print(f"  Loaded {gt_path.name}: {H}x{W}, score={gt_data.get('score', 'N/A')}")

    return all_data, rounds


def score_kl(pred, gt, H, W):
    """Competition scoring: 100 * exp(-3 * weighted_kl)."""
    pred = np.clip(pred, 1e-12, 1.0)
    gt = gt.reshape(H, W, NC)
    pred = pred.reshape(H, W, NC)

    # Entropy per cell
    ent = -np.sum(gt * np.log(np.where(gt > 0, gt, 1.0)), axis=-1)
    # KL per cell
    safe_gt = np.where(gt > 0, gt, 1.0)
    kl = np.sum(gt * np.log(safe_gt / pred), axis=-1)
    # Weighted
    total_wkl = np.sum(ent * kl)
    total_ent = np.sum(ent)
    if total_ent == 0:
        return 100.0
    return 100.0 * math.exp(-3.0 * total_wkl / total_ent)


def train_xgboost(X_train, y_train, X_val, y_val):
    """Train XGBoost multi-output model for probability prediction."""
    if not HAS_XGB:
        print("XGBoost not available, falling back to GradientBoosting")
        return train_sklearn_gb(X_train, y_train, X_val, y_val)

    models = []
    for c in range(NC):
        y_c = y_train[:, c]
        params = {
            "objective": "reg:squarederror",
            "max_depth": 6,
            "learning_rate": 0.05,
            "n_estimators": 500,
            "min_child_weight": 10,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "n_jobs": -1,
            "verbosity": 0,
        }
        params["early_stopping_rounds"] = 30
        model = xgb.XGBRegressor(**params)
        model.fit(
            X_train, y_c,
            eval_set=[(X_val, y_val[:, c])],
            verbose=False,
        )
        models.append(model)
        try:
            best_iter = model.best_iteration
        except AttributeError:
            best_iter = params["n_estimators"]
        print(f"    Class {c}: best_iteration={best_iter}")

    return models


def train_sklearn_gb(X_train, y_train, X_val, y_val):
    """Fallback: sklearn GradientBoosting."""
    models = []
    for c in range(NC):
        model = GradientBoostingRegressor(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            min_samples_leaf=20,
            subsample=0.8,
        )
        model.fit(X_train, y_train[:, c])
        models.append(model)
        print(f"    Class {c}: trained")
    return models


def predict_with_models(models, X):
    """Predict probabilities and normalize."""
    preds = np.zeros((X.shape[0], NC), dtype=np.float64)
    for c in range(NC):
        preds[:, c] = models[c].predict(X)

    # Floor and normalize
    preds = np.maximum(preds, PROB_FLOOR)
    preds /= preds.sum(axis=1, keepdims=True)
    return preds


def cross_validate(all_data, rounds):
    """Leave-one-round-out cross-validation."""
    scores_by_round = {}

    for val_round in rounds:
        print(f"\n--- CV fold: val={val_round} ---")

        # Split
        train_X, train_y = [], []
        val_items = []
        for item in all_data:
            if item["round"] == val_round:
                val_items.append(item)
            else:
                train_X.append(item["features"])
                train_y.append(item["targets"])

        if not val_items:
            continue

        X_train = np.concatenate(train_X, axis=0)
        y_train = np.concatenate(train_y, axis=0)

        # Use first val seed for early stopping
        X_val = val_items[0]["features"]
        y_val = val_items[0]["targets"]

        print(f"  Train: {X_train.shape[0]} samples, Val: {sum(v['features'].shape[0] for v in val_items)} samples")

        models = train_xgboost(X_train, y_train, X_val, y_val)

        # Score each val seed
        fold_scores = []
        for item in val_items:
            pred = predict_with_models(models, item["features"])

            # Apply domain constraints
            pred = apply_domain_constraints(pred, item["ic_map"], item["features"])

            score = score_kl(pred, item["targets"], item["H"], item["W"])
            fold_scores.append(score)
            print(f"  {item['round']}_s{item['seed']}: score={score:.2f}")

        avg = np.mean(fold_scores)
        scores_by_round[val_round] = avg
        print(f"  Fold avg: {avg:.2f}")

    overall = np.mean(list(scores_by_round.values()))
    print(f"\n=== Overall CV score: {overall:.2f} ===")
    for r, s in sorted(scores_by_round.items()):
        print(f"  {r}: {s:.2f}")

    return overall


def apply_domain_constraints(pred, ic_map, features):
    """Apply domain knowledge constraints to predictions."""
    pred = pred.copy()

    # Mountain cells (ic=5) should stay mountain
    mountain_mask = ic_map == 5
    if mountain_mask.any():
        pred[mountain_mask] = PROB_FLOOR
        pred[mountain_mask, 5] = 1.0 - 5 * PROB_FLOOR

    # Ocean cells (ic=0 with specific grid values) — check one-hot
    # ic_0=1 means ocean/shallow. Need to check if it's deep ocean
    # We use the coastal feature: if ic_0=1 and coastal=0 and ocean_adj=0,
    # likely deep ocean
    ocean_mask = (ic_map == 0) & (features[:, 7] == 0)  # ic=0, not coastal
    # For cells far from everything, suppress non-ocean
    far_mask = ocean_mask & (features[:, 9] < 0.05)  # sett_dist_inv < 0.05 means far
    if far_mask.any():
        # Boost ocean probability
        pred[far_mask, 0] = np.maximum(pred[far_mask, 0], 0.95)
        pred[far_mask] /= pred[far_mask].sum(axis=1, keepdims=True)

    # Port suppression for non-coastal
    non_coastal = features[:, 7] == 0
    pred[non_coastal, 2] = PROB_FLOOR

    # Re-normalize
    pred = np.maximum(pred, PROB_FLOOR)
    pred /= pred.sum(axis=1, keepdims=True)

    return pred


def train_final_model(all_data):
    """Train on ALL data for final model export."""
    print("\n=== Training final model on all data ===")
    all_X = np.concatenate([d["features"] for d in all_data], axis=0)
    all_y = np.concatenate([d["targets"] for d in all_data], axis=0)

    print(f"Total samples: {all_X.shape[0]}, features: {all_X.shape[1]}")

    models = train_xgboost(all_X, all_y, all_X[:1600], all_y[:1600])
    return models


def export_lookup_table(models, all_data, output_path):
    """Export model predictions as a JSON lookup table.

    Strategy: discretize features into the same buckets used by the
    existing predictor (ic, food, coastal, dist_bucket, n_sett) and
    average the model's predictions for each bucket. This makes the
    lookup table compatible with the existing Cloud Run predictor.
    """
    print("\n=== Exporting lookup table ===")

    # Collect all unique feature combinations and their model predictions
    bucket_preds = defaultdict(list)

    for item in all_data:
        X = item["features"]
        ic_map = item["ic_map"]
        pred = predict_with_models(models, X)
        pred = apply_domain_constraints(pred, ic_map, X)

        for i in range(X.shape[0]):
            ic = int(ic_map[i])
            food = int(min(X[i, 6], 4))  # cap at 4 like original
            coastal = int(X[i, 7])
            sd = X[i, 8]  # settlement distance

            # Distance bucket (matching predictor.py)
            if sd <= 3:
                db = "near"
            elif sd <= 7:
                db = "mid"
            elif sd <= 12:
                db = "far"
            else:
                db = "remote"

            ns = int(min(X[i, 10], 3))  # neighbor settlements, cap at 3

            key = f"{ic}_{food}_{coastal}_{db}_{ns}"
            bucket_preds[key].append(pred[i])

    # Average predictions per bucket
    lookup = {}
    for key, preds_list in bucket_preds.items():
        avg = np.mean(preds_list, axis=0)
        avg = np.maximum(avg, PROB_FLOOR)
        avg /= avg.sum()
        lookup[key] = [round(float(v), 6) for v in avg]

    print(f"Lookup table: {len(lookup)} entries")

    with open(output_path, "w") as f:
        json.dump(lookup, f)

    print(f"Saved to {output_path}")
    return lookup


def export_raw_model(models, output_path):
    """Export XGBoost models as JSON for direct inference."""
    if not HAS_XGB:
        print("Skipping raw model export (XGBoost not available)")
        return

    model_dir = Path(output_path).parent / "xgb_models"
    model_dir.mkdir(exist_ok=True)

    for c in range(NC):
        model_path = model_dir / f"class_{c}.json"
        models[c].save_model(str(model_path))
        print(f"  Saved {model_path}")

    print(f"Raw models saved to {model_dir}/")


def download_from_gcs(gcs_path, local_dir):
    """Download GT files from GCS."""
    print(f"Downloading from {gcs_path} to {local_dir}")
    os.makedirs(local_dir, exist_ok=True)
    subprocess.run(
        ["gsutil", "-m", "cp", f"{gcs_path}*.json", local_dir],
        check=True,
    )


def main():
    parser = argparse.ArgumentParser(description="Train Astar Island prediction model")
    parser.add_argument("--cache-dir", default="../cache",
                        help="Local cache directory with GT files")
    parser.add_argument("--gcs-bucket", default=None,
                        help="GCS path to download cache from (e.g. gs://bucket/cache/)")
    parser.add_argument("--output", default="model_lookup.json",
                        help="Output path for lookup table JSON")
    parser.add_argument("--skip-cv", action="store_true",
                        help="Skip cross-validation, just train final model")
    parser.add_argument("--export-raw", action="store_true",
                        help="Also export raw XGBoost model files")
    args = parser.parse_args()

    # Download from GCS if specified
    cache_dir = args.cache_dir
    if args.gcs_bucket:
        cache_dir = tempfile.mkdtemp(prefix="astar_cache_")
        download_from_gcs(args.gcs_bucket, cache_dir)

    # Load data
    all_data, rounds = load_data(cache_dir)
    if not all_data:
        print("ERROR: No GT data found!")
        sys.exit(1)

    print(f"\nLoaded {len(all_data)} GT files across {len(rounds)} rounds")
    total_samples = sum(d["features"].shape[0] for d in all_data)
    print(f"Total samples: {total_samples}, Features: {all_data[0]['features'].shape[1]}")

    # Cross-validation
    if not args.skip_cv:
        cv_score = cross_validate(all_data, rounds)

    # Train final model on all data
    models = train_final_model(all_data)

    # Export lookup table
    export_lookup_table(models, all_data, args.output)

    # Optionally export raw models
    if args.export_raw:
        export_raw_model(models, args.output)

    # Final sanity check: score on training data
    print("\n=== Sanity check: score on training data ===")
    for item in all_data[:5]:
        pred = predict_with_models(models, item["features"])
        pred = apply_domain_constraints(pred, item["ic_map"], item["features"])
        score = score_kl(pred, item["targets"], item["H"], item["W"])
        print(f"  {item['round']}_s{item['seed']}: {score:.2f}")


if __name__ == "__main__":
    main()
