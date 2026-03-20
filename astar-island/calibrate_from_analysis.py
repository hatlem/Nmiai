#!/usr/bin/env python3
"""
Fetch ground truth from completed rounds via analysis endpoint,
then recalibrate priors and simulator parameters.

Usage:
    export ASTAR_ISLAND_TOKEN=your_jwt_token
    python3 calibrate_from_analysis.py
"""

import json
import os
import sys
from collections import defaultdict

import numpy as np
import requests

BASE_URL = "https://api.ainm.no"
NUM_CLASSES = 6
TERRAIN_TO_CLASS = {10: 0, 11: 0, 0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5}
CLASS_NAMES = ["Empty", "Settlement", "Port", "Ruin", "Forest", "Mountain"]


def get_session():
    token = os.getenv("ASTAR_ISLAND_TOKEN")
    if not token:
        print("Set ASTAR_ISLAND_TOKEN env var first")
        sys.exit(1)
    s = requests.Session()
    s.headers["Authorization"] = f"Bearer {token}"
    return s


def fetch_all_analysis(session):
    """Fetch analysis data from all completed rounds."""
    rounds = session.get(f"{BASE_URL}/astar-island/rounds").json()
    if isinstance(rounds, dict):
        rounds = rounds.get("rounds", rounds.get("data", []))

    completed = [r for r in rounds if r.get("status") == "completed"]
    print(f"Found {len(completed)} completed rounds")

    all_data = []
    for r in completed:
        round_id = r["id"]
        round_num = r.get("round_number", "?")
        detail = session.get(f"{BASE_URL}/astar-island/rounds/{round_id}").json()

        for seed_idx in range(detail.get("seeds_count", 5)):
            try:
                analysis = session.get(
                    f"{BASE_URL}/astar-island/analysis/{round_id}/{seed_idx}"
                ).json()

                if "ground_truth" not in analysis:
                    print(f"  Round {round_num} seed {seed_idx}: no ground_truth")
                    continue

                gt = np.array(analysis["ground_truth"])
                init_grid = np.array(analysis.get("initial_grid",
                                                   detail["initial_states"][seed_idx]["grid"]))
                pred = np.array(analysis["prediction"]) if "prediction" in analysis else None
                score = analysis.get("score")

                all_data.append({
                    "round_num": round_num,
                    "round_id": round_id,
                    "seed_idx": seed_idx,
                    "gt": gt,
                    "init_grid": init_grid,
                    "pred": pred,
                    "score": score,
                    "settlements": detail["initial_states"][seed_idx].get("settlements", []),
                })
                print(f"  Round {round_num} seed {seed_idx}: "
                      f"score={score}, gt shape={gt.shape}")

            except Exception as e:
                print(f"  Round {round_num} seed {seed_idx}: error {e}")

    return all_data


def classify_grid(grid):
    out = np.zeros_like(grid, dtype=np.int32)
    for code, cls in TERRAIN_TO_CLASS.items():
        out[grid == code] = cls
    return out


def settlement_distance(grid, settlements, W, H):
    dist = np.full((H, W), 999, dtype=np.float64)
    for s in settlements:
        sx, sy = s.get("x", -1), s.get("y", -1)
        if 0 <= sx < W and 0 <= sy < H:
            for y in range(H):
                for x in range(W):
                    d = abs(x - sx) + abs(y - sy)
                    if d < dist[y, x]:
                        dist[y, x] = d
    return dist


def calibrate_priors(all_data):
    """Compute average GT distribution per initial terrain class."""
    print("\n" + "=" * 70)
    print("PRIOR CALIBRATION")
    print("=" * 70)

    # Accumulate GT distributions by initial class
    class_gt_sum = defaultdict(lambda: np.zeros(NUM_CLASSES))
    class_gt_count = defaultdict(int)

    # Also by distance bucket
    dist_gt_sum = defaultdict(lambda: np.zeros(NUM_CLASSES))
    dist_gt_count = defaultdict(int)

    for d in all_data:
        gt = d["gt"]
        init_grid = d["init_grid"]
        init_cls = classify_grid(init_grid)
        H, W = init_grid.shape
        sett_dist = settlement_distance(
            init_grid, d["settlements"], W, H
        )

        for y in range(H):
            for x in range(W):
                ic = int(init_cls[y, x])
                gt_dist = gt[y, x]

                # Skip ocean/mountain (always same)
                if init_grid[y, x] in (10, 5):
                    continue

                class_gt_sum[ic] += gt_dist
                class_gt_count[ic] += 1

                # Distance buckets
                sd = sett_dist[y, x]
                if sd <= 3:
                    bucket = "near"
                elif sd <= 7:
                    bucket = "mid"
                else:
                    bucket = "far"

                key = (ic, bucket)
                dist_gt_sum[key] += gt_dist
                dist_gt_count[key] += 1

    # Print per-class priors
    print("\nAverage GT distribution per initial class (across ALL rounds):")
    calibrated = {}
    for ic in sorted(class_gt_count.keys()):
        avg = class_gt_sum[ic] / class_gt_count[ic]
        calibrated[ic] = avg.tolist()
        labels = " ".join(f"{CLASS_NAMES[j]:>10}:{avg[j]:.4f}" for j in range(NUM_CLASSES))
        print(f"  Class {ic} ({CLASS_NAMES[ic]:>10}, n={class_gt_count[ic]:>5}): {labels}")

    # Print per-distance priors
    print("\nBy initial class + distance to nearest settlement:")
    dist_calibrated = {}
    for (ic, bucket) in sorted(dist_gt_count.keys()):
        n = dist_gt_count[(ic, bucket)]
        if n < 10:
            continue
        avg = dist_gt_sum[(ic, bucket)] / n
        dist_calibrated[f"{ic}_{bucket}"] = avg.tolist()
        labels = " ".join(f"{avg[j]:.4f}" for j in range(NUM_CLASSES))
        print(f"  {CLASS_NAMES[ic]:>10} {bucket:>4} (n={n:>5}): [{labels}]")

    return calibrated, dist_calibrated


def analyze_prediction_errors(all_data):
    """Analyze where our predictions are worst."""
    print("\n" + "=" * 70)
    print("PREDICTION ERROR ANALYSIS")
    print("=" * 70)

    for d in all_data:
        if d["pred"] is None:
            continue

        gt = d["gt"]
        pred = d["pred"]
        init_grid = d["init_grid"]
        init_cls = classify_grid(init_grid)
        H, W = init_grid.shape

        # Per-class average error
        print(f"\nRound {d['round_num']} seed {d['seed_idx']} (score={d['score']}):")

        for ic in range(NUM_CLASSES):
            mask = init_cls == ic
            if not mask.any():
                continue
            n = mask.sum()
            gt_avg = gt[mask].mean(axis=0)
            pred_avg = pred[mask].mean(axis=0)
            diff = pred_avg - gt_avg

            labels = " ".join(f"{CLASS_NAMES[j]}:{diff[j]:+.4f}" for j in range(NUM_CLASSES))
            print(f"  {CLASS_NAMES[ic]:>10} (n={n:>4}): {labels}")


def analyze_transition_patterns(all_data):
    """Analyze what terrain types transition to what."""
    print("\n" + "=" * 70)
    print("TRANSITION ANALYSIS (init → GT mode)")
    print("=" * 70)

    trans_count = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.float64)

    for d in all_data:
        gt = d["gt"]
        init_grid = d["init_grid"]
        init_cls = classify_grid(init_grid)
        H, W = init_grid.shape

        for ic in range(NUM_CLASSES):
            mask = init_cls == ic
            if mask.any():
                # Use GT probabilities as soft transitions
                trans_count[ic] += gt[mask].sum(axis=0)

    # Normalize
    row_sums = trans_count.sum(axis=1, keepdims=True)
    row_sums = np.maximum(row_sums, 1e-10)
    trans_prob = trans_count / row_sums

    print("\nTransition matrix (rows=initial, cols=final GT probability):")
    header = "           " + " ".join(f"{CLASS_NAMES[j]:>10}" for j in range(NUM_CLASSES))
    print(header)
    for ic in range(NUM_CLASSES):
        row = " ".join(f"{trans_prob[ic, j]:>10.4f}" for j in range(NUM_CLASSES))
        print(f"  {CLASS_NAMES[ic]:>10} {row}")


def save_calibration(calibrated, dist_calibrated):
    """Save new calibration data."""
    # Update calibration.json
    cal_data = {}
    for k, v in calibrated.items():
        cal_data[str(k)] = v
    cal_data["_source"] = "multi_round_ground_truth"
    cal_data["_note"] = "Average GT distribution per initial class across all completed rounds"

    with open("calibration.json", "w") as f:
        json.dump(cal_data, f, indent=2)
    print("\nSaved calibration.json")

    # Save distance-based calibration
    with open("calibration_by_distance.json", "w") as f:
        json.dump(dist_calibrated, f, indent=2)
    print("Saved calibration_by_distance.json")


def main():
    session = get_session()

    # Fetch all analysis data
    all_data = fetch_all_analysis(session)

    if not all_data:
        print("No analysis data found!")
        return

    # Calibrate priors
    calibrated, dist_calibrated = calibrate_priors(all_data)

    # Analyze our prediction errors
    analyze_prediction_errors(all_data)

    # Analyze transitions
    analyze_transition_patterns(all_data)

    # Save
    save_calibration(calibrated, dist_calibrated)

    print("\n" + "=" * 70)
    print("DONE — calibration.json and calibration_by_distance.json updated")
    print("Re-run validation to see improved scores.")
    print("=" * 70)


if __name__ == "__main__":
    main()
