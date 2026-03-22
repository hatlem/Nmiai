#!/usr/bin/env python3
"""
Train XGBoost models to predict per-cell probability distributions.
Uses ALL historical GT data + regime features (transition rates per round).
Outputs a JSON model that agent_v7.js can load directly.
"""
import json, glob, os, sys
import numpy as np
from pathlib import Path

CACHE = Path(__file__).parent / "cache"
NC = 6
TTC = {10:0, 11:0, 0:0, 1:1, 2:2, 3:3, 4:4, 5:5}

def cc(v): return TTC.get(v, 0)

def extract_features(ig, y, x, H, W, transitions=None):
    """Extract features for one cell."""
    raw = ig[y][x]
    ic = cc(raw)

    # Food neighbors
    food = 0
    for dy in range(-1, 2):
        for dx in range(-1, 2):
            if dy == 0 and dx == 0: continue
            ny, nx = y+dy, x+dx
            if 0 <= ny < H and 0 <= nx < W and ig[ny][nx] == 4: food += 1
    food = min(food, 4)

    # Coastal
    co = 0
    if ig[y][x] not in (10, 5):
        for dy, dx in [(-1,0),(1,0),(0,-1),(0,1)]:
            ny, nx = y+dy, x+dx
            if 0 <= ny < H and 0 <= nx < W and ig[ny][nx] == 10:
                co = 1; break

    # Settlement distance
    sd = 999
    for sy in range(H):
        for sx in range(W):
            if ig[sy][sx] in (1, 2):
                sd = min(sd, abs(y-sy) + abs(x-sx))

    # Neighbor settlements (r=2)
    ns = 0
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            if dy == 0 and dx == 0: continue
            ny, nx = y+dy, x+dx
            if 0 <= ny < H and 0 <= nx < W and ig[ny][nx] in (1, 2): ns += 1
    ns = min(ns, 5)

    # Mountain neighbors
    mn = 0
    for dy, dx in [(-1,0),(1,0),(0,-1),(0,1)]:
        ny, nx = y+dy, x+dx
        if 0 <= ny < H and 0 <= nx < W and ig[ny][nx] == 5: mn += 1

    # Forest density in 5x5
    fd = 0
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            ny, nx = y+dy, x+dx
            if 0 <= ny < H and 0 <= nx < W and ig[ny][nx] == 4: fd += 1

    # Edge distance
    edge = min(y, x, H-1-y, W-1-x)

    features = [ic, food, co, min(sd, 20), ns, mn, fd, min(edge, 10)]

    # Regime features (transition rates from this round)
    if transitions:
        # Empty→Settlement, Settlement→Settlement, Settlement→Ruin, Forest→Forest
        e2s = transitions.get('0', [0]*6)[1] if '0' in transitions else 0
        s2s = transitions.get('1', [0]*6)[1] if '1' in transitions else 0
        s2r = transitions.get('1', [0]*6)[3] if '1' in transitions else 0
        f2f = transitions.get('4', [0]*6)[4] if '4' in transitions else 0
        s2e = transitions.get('1', [0]*6)[0] if '1' in transitions else 0
        features.extend([e2s, s2s, s2r, f2f, s2e])
    else:
        features.extend([0, 0, 0, 0, 0])

    return features

def load_all_data():
    """Load all GT data with regime features."""
    init_files = sorted(CACHE.glob("r[0-9]*_init.json"))

    # Load transition profiles
    trans = {}
    for f in CACHE.glob("transitions_r*.json"):
        rnum = int(f.stem.split("_r")[1])
        trans[rnum] = json.loads(f.read_text())

    X, Y = [], []
    rounds_loaded = 0

    for init_file in init_files:
        rnum_str = init_file.stem.split("_")[0]  # e.g. 'r1'
        try:
            rnum = int(rnum_str[1:])
        except:
            continue

        init_data = json.loads(init_file.read_text())
        H = init_data.get('map_height', 40)
        W = init_data.get('map_width', 40)
        seeds_count = init_data.get('seeds_count', 5)

        round_trans = trans.get(rnum, None)

        for si in range(seeds_count):
            gt_file = CACHE / f"r{rnum}_gt_s{si}.json"
            if not gt_file.exists(): continue

            gt_data = json.loads(gt_file.read_text())
            ig = gt_data.get('initial_grid')
            gt = gt_data.get('ground_truth')
            if ig is None or gt is None: continue

            for y in range(H):
                for x in range(W):
                    raw = ig[y][x]
                    if raw == 5 or raw == 10: continue  # skip static

                    feat = extract_features(ig, y, x, H, W, round_trans)
                    target = gt[y][x]

                    X.append(feat)
                    Y.append(target)

        rounds_loaded += 1
        print(f"  R{rnum}: loaded ({seeds_count} seeds)", flush=True)

    print(f"\nTotal: {len(X)} cells from {rounds_loaded} rounds")
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)

def train_models(X, Y):
    """Train one XGBoost model per output class."""
    try:
        import xgboost as xgb
    except ImportError:
        print("Installing xgboost...")
        os.system(f"{sys.executable} -m pip install xgboost --break-system-packages -q")
        import xgboost as xgb

    feature_names = ['init_class', 'food', 'coastal', 'sett_dist', 'n_sett',
                     'mountain_nb', 'forest_density', 'edge_dist',
                     'e2s', 's2s', 's2r', 'f2f', 's2e']

    models = []
    for c in range(NC):
        print(f"\nTraining class {c}...", flush=True)

        model = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=10,
            reg_alpha=0.1,
            reg_lambda=1.0,
            n_jobs=-1,
            random_state=42,
        )
        model.fit(X, Y[:, c])

        # Feature importance
        imp = model.feature_importances_
        top = sorted(zip(feature_names, imp), key=lambda x: -x[1])[:5]
        print(f"  Top features: {', '.join(f'{n}={v:.3f}' for n,v in top)}")

        models.append(model)

    return models

def export_lookup(models, X_unique_contexts):
    """Export model predictions as a lookup table (JSON) for agent_v7.js."""
    # Generate predictions for all unique feature combinations
    # This is a "compiled" model — pre-compute all predictions
    predictions = {}

    for i, x in enumerate(X_unique_contexts):
        ic = int(x[0])
        food = int(x[1])
        co = int(x[2])
        sd = int(x[3])
        ns = int(x[4])
        db = 'near' if sd <= 3 else 'mid' if sd <= 7 else 'far' if sd <= 12 else 'remote'

        key = f"{ic}_{food}_{co}_{db}_{ns}"

        probs = np.array([m.predict(x.reshape(1, -1))[0] for m in models])
        probs = np.maximum(probs, 0.0005)
        probs = probs / probs.sum()

        if key not in predictions:
            predictions[key] = probs.tolist()

    return predictions

def evaluate_loocv(X, Y):
    """Leave-one-round-out cross-validation."""
    try:
        import xgboost as xgb
    except:
        return

    # Group by round (using regime features as proxy — same round has same regime)
    regime_keys = [tuple(x[8:13]) for x in X]
    unique_regimes = list(set(regime_keys))

    print(f"\n=== Leave-One-Round-Out CV ({len(unique_regimes)} rounds) ===")

    all_scores = []
    for hold_regime in unique_regimes:
        train_mask = np.array([tuple(x[8:13]) != hold_regime for x in X])
        test_mask = ~train_mask

        if test_mask.sum() == 0 or train_mask.sum() == 0: continue

        X_train, Y_train = X[train_mask], Y[train_mask]
        X_test, Y_test = X[test_mask], Y[test_mask]

        # Quick train
        preds = np.zeros_like(Y_test)
        for c in range(NC):
            m = xgb.XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1,
                                  subsample=0.8, min_child_weight=10, n_jobs=-1, verbosity=0)
            m.fit(X_train, Y_train[:, c])
            preds[:, c] = m.predict(X_test)

        # Floor + normalize
        preds = np.maximum(preds, 0.0005)
        preds = preds / preds.sum(axis=1, keepdims=True)

        # Score: entropy-weighted KL
        eps = 1e-10
        entropy = -np.sum(Y_test * np.log(Y_test + eps), axis=1)
        kl = np.sum(Y_test * np.log((Y_test + eps) / (preds + eps)), axis=1)

        mask = entropy > 0.01
        if mask.sum() == 0: continue

        weighted_kl = np.sum(entropy[mask] * kl[mask]) / np.sum(entropy[mask])
        score = max(0, min(100, 100 * np.exp(-3 * weighted_kl)))
        all_scores.append(score)
        print(f"  Round (n={test_mask.sum()}): {score:.1f}")

    print(f"\n  Mean CV: {np.mean(all_scores):.1f} (lookup CV floor was 72.2)")

def main():
    print("Loading data...", flush=True)
    X, Y = load_all_data()

    print("\n=== Cross-Validation ===", flush=True)
    evaluate_loocv(X, Y)

    print("\n=== Training Final Models ===", flush=True)
    models = train_models(X, Y)

    # Export as regime-conditioned lookup
    # For each unique (regime, cell-features) combo, pre-compute prediction
    print("\n=== Exporting ===", flush=True)

    # Save models for Python use
    for c, m in enumerate(models):
        m.save_model(f"astar-island/xgb_class{c}.json")
    print(f"Saved {NC} XGBoost models")

    # Also export a mega-lookup keyed by regime bucket + cell features
    # Group regime into buckets: harsh (e2s<0.05), moderate (0.05-0.15), mild (>0.15)
    unique_X = np.unique(X, axis=0)
    lookup = export_lookup(models, unique_X)

    out_path = Path(__file__).parent / "xgb_lookup.json"
    with open(out_path, 'w') as f:
        json.dump(lookup, f)
    print(f"Exported {len(lookup)} bins to xgb_lookup.json")

if __name__ == '__main__':
    main()
