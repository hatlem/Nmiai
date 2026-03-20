#!/usr/bin/env python3
"""
Astar Island — Advanced training pipeline.

Produces a hierarchical Bayesian lookup table with Bayesian shrinkage
across 5 granularity levels. Leave-one-round-out cross-validation shows
~80 avg score (vs ~46 for flat lookup).

Also exports shift profiles: precomputed per-IC-type transition vectors
for each training round, enabling the runtime agent to interpolate between
known "hidden parameter regimes" based on observed transitions.

Usage:
  python train_advanced.py --cache-dir ../cache --output ../gt_lookup_v2.json
  python train_advanced.py --cache-dir ../cache --cv-only  # just run CV
"""

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

NC = 6  # number of terrain classes
TTC = {10: 0, 11: 0, 0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5}
PROB_FLOOR = 0.001

# Shrinkage strengths per hierarchy level (tuned via CV)
# Higher = more shrinkage toward coarser level (less trust in fine-grained data)
SHRINKAGE = [20, 50, 100, 200, 0]  # level 0 (finest) to level 4 (coarsest)


# ── Feature computation ──────────────────────────────────────────────────────

def compute_features(grid):
    """Compute spatial features for each cell in the grid.

    Returns: ic, food, coastal, sd, nsett — all (H, W) arrays.
    """
    g = np.array(grid)
    H, W = g.shape
    ic = np.vectorize(lambda x: TTC.get(x, 0))(g)

    # Food: count forest neighbors (8-connected), capped at 4
    forest = (g == 4).astype(np.float32)
    food = _convolve_count(forest, H, W, radius=1)
    food = np.minimum(food, 4)

    # Coastal: land cell adjacent to ocean (4-connected)
    ocean = (g == 10)
    padded = np.pad(ocean, 1, constant_values=False)
    coastal = np.zeros((H, W), dtype=bool)
    for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        coastal |= padded[1 + dy:H + 1 + dy, 1 + dx:W + 1 + dx]
    coastal = coastal & ~ocean & (g != 5)

    # Settlement distance (Manhattan)
    sett_mask = (g == 1) | (g == 2)
    sd = np.full((H, W), 999, dtype=np.float32)
    if sett_mask.any():
        ys, xs = np.mgrid[0:H, 0:W]
        for sy, sx in zip(*np.where(sett_mask)):
            dist = np.abs(ys - sy) + np.abs(xs - sx)
            sd = np.minimum(sd, dist.astype(np.float32))

    # Neighbor settlements (radius 2), capped at 3
    sett_float = sett_mask.astype(np.float32)
    nsett = _convolve_count(sett_float, H, W, radius=2)
    nsett = np.minimum(nsett, 3)

    return ic, food.astype(np.int32), coastal, sd, nsett.astype(np.int32)


def _convolve_count(arr, H, W, radius):
    """Count non-zero neighbors within Manhattan radius."""
    result = np.zeros((H, W), dtype=np.float32)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dy == 0 and dx == 0:
                continue
            shifted = np.zeros_like(arr)
            sy = slice(max(0, -dy), min(H, H - dy))
            sx = slice(max(0, -dx), min(W, W - dx))
            ty = slice(max(0, dy), min(H, H + dy))
            tx = slice(max(0, dx), min(W, W + dx))
            shifted[ty, tx] = arr[sy, sx]
            result += shifted
    return result


def dist_bucket(sd):
    """Map settlement distance to bucket string."""
    if sd <= 3:
        return "near"
    elif sd <= 7:
        return "mid"
    elif sd <= 12:
        return "far"
    else:
        return "remote"


# ── Hierarchy key generation ─────────────────────────────────────────────────

def make_hierarchy_keys(ic_val, food_val, coastal_val, db, nsett_val):
    """Generate hierarchy keys from finest to coarsest."""
    co = int(coastal_val)
    return [
        f"{ic_val}_{food_val}_{co}_{db}_{nsett_val}",  # level 0: finest
        f"{ic_val}_{food_val}_{co}_{db}",                # level 1
        f"{ic_val}_{co}_{db}",                            # level 2
        f"{ic_val}_{db}",                                 # level 3
        f"{ic_val}",                                      # level 4: coarsest
    ]


# ── Data loading ─────────────────────────────────────────────────────────────

def load_data(cache_dir):
    """Load all GT files. Returns dict: round_id -> list of 5 seed dicts."""
    cache = Path(cache_dir)
    rounds = set()
    for f in cache.glob("r*_gt_s*.json"):
        r = f.name.split("_gt_")[0]
        rounds.add(r)
    rounds = sorted(rounds)

    round_data = {}
    for r in rounds:
        seeds = []
        for s in range(5):
            gt_path = cache / f"{r}_gt_s{s}.json"
            if not gt_path.exists():
                print(f"  Skipping {gt_path.name} (not found)")
                continue
            with open(gt_path) as f:
                d = json.load(f)
            seeds.append(d)
        if len(seeds) == 5:
            round_data[r] = seeds
            print(f"  Loaded {r}: {len(seeds)} seeds, score={seeds[0].get('score', 'N/A')}")

    print(f"Found {len(round_data)} complete rounds: {list(round_data.keys())}")
    return round_data


# ── Scoring ──────────────────────────────────────────────────────────────────

def score_kl(pred, gt, H, W):
    """Competition scoring: 100 * exp(-3 * weighted_kl)."""
    pred = np.clip(pred, 1e-12, 1.0)
    gt = gt.reshape(H, W, NC)
    pred = pred.reshape(H, W, NC)
    ent = -np.sum(gt * np.log(np.where(gt > 0, gt, 1.0)), axis=-1)
    safe_gt = np.where(gt > 0, gt, 1.0)
    kl = np.sum(gt * np.log(safe_gt / pred), axis=-1)
    total_wkl = np.sum(ent * kl)
    total_ent = np.sum(ent)
    if total_ent == 0:
        return 100.0
    return 100.0 * math.exp(-3.0 * total_wkl / total_ent)


# ── Hierarchical Bayesian lookup ─────────────────────────────────────────────

def build_hierarchical_lookup(train_rounds, round_data):
    """Build hierarchical lookup from training data.

    Collects GT probabilities at 5 granularity levels, then computes
    the mean at each level with sample count for Bayesian shrinkage.

    Returns: list of 5 dicts, each mapping key -> (mean_prob, sample_count).
    """
    levels = [defaultdict(list) for _ in range(5)]

    for r in train_rounds:
        for s_idx in range(5):
            d = round_data[r][s_idx]
            grid = np.array(d["initial_grid"])
            gt = np.array(d["ground_truth"])
            H, W = d["height"], d["width"]
            ic, food, coastal, sd, nsett = compute_features(grid)

            for y in range(H):
                for x in range(W):
                    raw = grid[y, x]
                    if raw == 5 or raw == 10:
                        continue

                    db = dist_bucket(sd[y, x])
                    keys = make_hierarchy_keys(
                        ic[y, x], food[y, x], coastal[y, x], db, nsett[y, x]
                    )
                    p = gt[y, x]
                    for lv, key in enumerate(keys):
                        levels[lv][key].append(p)

    # Compute means and counts
    level_avgs = []
    for lv in range(5):
        result = {}
        for key, preds_list in levels[lv].items():
            result[key] = (np.mean(preds_list, axis=0), len(preds_list))
        level_avgs.append(result)

    return level_avgs


def predict_from_hierarchy(grid, H, W, level_avgs, shrinkages=None):
    """Predict using hierarchical Bayesian shrinkage lookup.

    For each cell, starts from coarsest level (uniform prior) and
    progressively blends toward finer levels with weight proportional
    to sample count at that level.
    """
    if shrinkages is None:
        shrinkages = SHRINKAGE

    g = np.array(grid)
    ic, food, coastal, sd, nsett = compute_features(grid)

    pred = np.zeros((H, W, NC))
    for y in range(H):
        for x in range(W):
            raw = g[y, x]
            if raw == 5:
                pred[y, x] = [PROB_FLOOR] * 5 + [1.0 - 5 * PROB_FLOOR]
                continue
            if raw == 10:
                pred[y, x] = [1.0 - 5 * PROB_FLOOR] + [PROB_FLOOR] * 5
                continue

            db = dist_bucket(sd[y, x])
            keys = make_hierarchy_keys(
                ic[y, x], food[y, x], coastal[y, x], db, nsett[y, x]
            )

            # Start from uniform, shrink toward finer levels
            p = np.ones(NC) / NC
            for lv_idx in range(4, -1, -1):
                key = keys[lv_idx]
                if key in level_avgs[lv_idx]:
                    avg, cnt = level_avgs[lv_idx][key]
                    s = shrinkages[lv_idx]
                    if s > 0:
                        w = cnt / (cnt + s)
                    else:
                        w = 1.0
                    p = (1 - w) * p + w * avg

            if not coastal[y, x]:
                p[2] = PROB_FLOOR
            p = np.maximum(p, PROB_FLOOR)
            p /= p.sum()
            pred[y, x] = p

    return pred


# ── Shift profiles ───────────────────────────────────────────────────────────

def compute_round_shift_profile(round_data_seeds):
    """Compute per-IC-type average GT for a round (across 5 seeds).

    Returns dict: ic_type -> avg probability vector (6,).
    """
    profile = {}
    d0 = round_data_seeds[0]
    grid = np.array(d0["initial_grid"])
    H, W = d0["height"], d0["width"]
    ic = np.vectorize(lambda x: TTC.get(x, 0))(grid)

    gt_avg = np.mean(
        [np.array(round_data_seeds[s]["ground_truth"]) for s in range(5)], axis=0
    )

    for c in range(NC):
        mask = ic == c
        if mask.any():
            profile[c] = gt_avg[mask].mean(axis=0).tolist()

    return profile


# ── Export ───────────────────────────────────────────────────────────────────

def export_flat_lookup(level_avgs, output_path):
    """Export hierarchical lookup as flat JSON compatible with auto_v6.js.

    Includes entries from ALL hierarchy levels (finest to coarsest) so the
    JS agent can fall back gracefully when a fine-grained key is missing.

    Key formats by level:
      0: ic_food_coastal_db_nsett   (finest)
      1: ic_food_coastal_db
      2: ic_coastal_db
      3: ic_db
      4: ic                         (coarsest)
    """
    lookup = {}

    # Export all levels, computing hierarchical Bayesian prediction for each
    for lv in range(5):
        for key in level_avgs[lv]:
            if key in lookup:
                continue  # finer level already has this key

            # For coarser levels, the prediction is just the shrunk average
            # from that level up through coarser levels
            parts = key.split("_")

            # Build a chain: from this level upward
            p = np.ones(NC) / NC
            # We need to find the coarser keys. For each level, the key is
            # a prefix of the finer level.
            chain_keys = [None] * 5
            chain_keys[lv] = key

            # Build coarser keys from this key's parts
            if lv == 0:
                ic_val, f, co, db, n = int(parts[0]), int(parts[1]), int(parts[2]), parts[3], int(parts[4])
                chain_keys = make_hierarchy_keys(ic_val, f, co, db, n)
            elif lv == 1:
                ic_val, f, co, db = int(parts[0]), int(parts[1]), int(parts[2]), parts[3]
                chain_keys[2] = f"{ic_val}_{co}_{db}"
                chain_keys[3] = f"{ic_val}_{db}"
                chain_keys[4] = f"{ic_val}"
            elif lv == 2:
                ic_val, co, db = int(parts[0]), int(parts[1]), parts[2]
                chain_keys[3] = f"{ic_val}_{db}"
                chain_keys[4] = f"{ic_val}"
            elif lv == 3:
                ic_val, db = int(parts[0]), parts[1]
                chain_keys[4] = f"{ic_val}"
            elif lv == 4:
                pass  # just the IC

            # Compute prediction using Bayesian shrinkage from coarsest up
            for chain_lv in range(4, -1, -1):
                if chain_keys[chain_lv] is None:
                    continue
                ck = chain_keys[chain_lv]
                if ck in level_avgs[chain_lv]:
                    avg, cnt = level_avgs[chain_lv][ck]
                    s = SHRINKAGE[chain_lv]
                    if s > 0:
                        w = cnt / (cnt + s)
                    else:
                        w = 1.0
                    p = (1 - w) * p + w * avg

            p = np.maximum(p, PROB_FLOOR)
            p /= p.sum()
            lookup[key] = [round(float(v), 6) for v in p]

    print(f"Exported {len(lookup)} lookup entries across all hierarchy levels")

    with open(output_path, "w") as f:
        json.dump(lookup, f)

    return lookup


def export_shift_profiles(round_data, output_path):
    """Export per-round shift profiles for runtime interpolation."""
    profiles = {}
    for r, seeds in round_data.items():
        profile = compute_round_shift_profile(seeds)
        # Convert numpy arrays to lists
        profiles[r] = {str(k): v for k, v in profile.items()}

    with open(output_path, "w") as f:
        json.dump(profiles, f, indent=2)

    print(f"Exported {len(profiles)} shift profiles")
    return profiles


# ── Cross-validation ─────────────────────────────────────────────────────────

def cross_validate(round_data):
    """Leave-one-round-out cross-validation."""
    all_rounds = list(round_data.keys())
    scores_by_round = {}

    for val_r in all_rounds:
        train_rounds = [r for r in all_rounds if r != val_r]
        level_avgs = build_hierarchical_lookup(train_rounds, round_data)

        val_scores = []
        for s_idx in range(5):
            d = round_data[val_r][s_idx]
            grid = np.array(d["initial_grid"])
            gt = np.array(d["ground_truth"])
            H, W = d["height"], d["width"]

            pred = predict_from_hierarchy(grid, H, W, level_avgs)
            score = score_kl(pred, gt, H, W)
            val_scores.append(score)

        avg = np.mean(val_scores)
        scores_by_round[val_r] = avg
        print(f"  {val_r}: {avg:.2f} (seeds: {[f'{s:.1f}' for s in val_scores]})")

    overall = np.mean(list(scores_by_round.values()))
    print(f"\n=== Overall CV score: {overall:.2f} ===")
    for r, s in sorted(scores_by_round.items()):
        print(f"  {r}: {s:.2f}")

    return overall


def cross_validate_with_shift(round_data):
    """LOO-CV simulating the runtime shift mechanism.

    Uses oracle per-IC-type shift (as if we had unlimited observations).
    Multiplicative shift with optimal damping=0.8, clip=(0.1, 10).
    """
    SHIFT_DAMP = 0.8
    SHIFT_CLIP = (0.1, 10.0)

    all_rounds = list(round_data.keys())
    scores_by_round = {}

    for val_r in all_rounds:
        train_rounds = [r for r in all_rounds if r != val_r]
        level_avgs = build_hierarchical_lookup(train_rounds, round_data)

        # Compute the lookup's per-IC average (for shift baseline)
        lookup_ic_avg = {}
        for c in range(NC):
            key = str(c)
            if key in level_avgs[4]:  # coarsest level
                lookup_ic_avg[c] = level_avgs[4][key][0]
            else:
                lookup_ic_avg[c] = np.ones(NC) / NC

        val_scores = []
        for s_idx in range(5):
            d = round_data[val_r][s_idx]
            grid = np.array(d["initial_grid"])
            gt = np.array(d["ground_truth"])
            H, W = d["height"], d["width"]
            ic_map = np.vectorize(lambda x: TTC.get(x, 0))(np.array(grid))

            # Oracle shift: use GT to compute per-IC transition rates
            obs_ic_avg = {}
            for c in range(NC):
                mask = ic_map == c
                if mask.any():
                    obs_ic_avg[c] = gt[mask].mean(axis=0)

            # Compute multiplicative shift (ratio of observed vs lookup)
            shift = {}
            for c in range(NC):
                if c in obs_ic_avg and c in lookup_ic_avg:
                    s = obs_ic_avg[c] / np.maximum(lookup_ic_avg[c], 0.01)
                    shift[c] = np.clip(s, SHIFT_CLIP[0], SHIFT_CLIP[1])
                else:
                    shift[c] = np.ones(NC)

            # Apply damped multiplicative shift
            pred = predict_from_hierarchy(grid, H, W, level_avgs)
            for y in range(H):
                for x in range(W):
                    raw = np.array(grid)[y, x]
                    if raw == 5 or raw == 10:
                        continue
                    c = ic_map[y, x]
                    if c in shift:
                        for cls in range(NC):
                            pred[y, x, cls] *= shift[c][cls] ** SHIFT_DAMP
                    pred[y, x] = np.maximum(pred[y, x], PROB_FLOOR)
                    pred[y, x] /= pred[y, x].sum()

            score = score_kl(pred, gt, H, W)
            val_scores.append(score)

        avg = np.mean(val_scores)
        scores_by_round[val_r] = avg
        print(f"  {val_r}: {avg:.2f} (with oracle shift, damp={SHIFT_DAMP})")

    overall = np.mean(list(scores_by_round.values()))
    print(f"\n=== Overall CV with oracle shift: {overall:.2f} ===")
    print(f"    (damp={SHIFT_DAMP}, clip={SHIFT_CLIP})")
    return overall


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Advanced Astar Island training")
    parser.add_argument("--cache-dir", default="../cache",
                        help="Local cache directory with GT files")
    parser.add_argument("--output", default="../gt_lookup_v2.json",
                        help="Output path for lookup table JSON")
    parser.add_argument("--profiles-output", default="../shift_profiles.json",
                        help="Output path for shift profiles JSON")
    parser.add_argument("--cv-only", action="store_true",
                        help="Only run cross-validation, don't export")
    parser.add_argument("--cv-shift", action="store_true",
                        help="Run CV with simulated shift mechanism")
    args = parser.parse_args()

    # Load data
    round_data = load_data(args.cache_dir)
    if not round_data:
        print("ERROR: No GT data found!")
        sys.exit(1)

    all_rounds = list(round_data.keys())
    total_samples = sum(
        5 * round_data[r][0]["height"] * round_data[r][0]["width"]
        for r in all_rounds
    )
    print(f"\nLoaded {len(all_rounds)} rounds, {total_samples} total cell-seed samples")

    # Cross-validation
    print("\n=== Leave-one-round-out cross-validation ===")
    cv_score = cross_validate(round_data)

    if args.cv_shift:
        print("\n=== Cross-validation with oracle shift ===")
        cv_shift_score = cross_validate_with_shift(round_data)

    if args.cv_only:
        return

    # Build final model on all data
    print("\n=== Building final lookup on all data ===")
    level_avgs = build_hierarchical_lookup(all_rounds, round_data)

    for lv in range(5):
        print(f"  Level {lv}: {len(level_avgs[lv])} entries")

    # Export
    lookup = export_flat_lookup(level_avgs, args.output)
    print(f"Saved lookup to {args.output}")

    profiles = export_shift_profiles(round_data, args.profiles_output)
    print(f"Saved shift profiles to {args.profiles_output}")

    # Sanity check: score on training data
    print("\n=== Sanity check: score on training data ===")
    for r in list(all_rounds)[:3]:
        d = round_data[r][0]
        grid = np.array(d["initial_grid"])
        gt = np.array(d["ground_truth"])
        H, W = d["height"], d["width"]
        pred = predict_from_hierarchy(grid, H, W, level_avgs)
        score = score_kl(pred, gt, H, W)
        print(f"  {r}_s0: {score:.2f}")


if __name__ == "__main__":
    main()
