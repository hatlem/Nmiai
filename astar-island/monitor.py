#!/usr/bin/env python3
"""
Monitor round status, fetch scores, and download analysis data when available.
Also calibrates priors from ground truth for future rounds.

Usage:
    python monitor.py --token <JWT>
    python monitor.py --token <JWT> --poll   # Poll every 60s until scored
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import requests

BASE = "https://api.ainm.no"


def create_session(token):
    s = requests.Session()
    s.headers["Authorization"] = f"Bearer {token}"
    return s


def check_status(session):
    my_rounds = session.get(f"{BASE}/astar-island/my-rounds").json()
    rounds = session.get(f"{BASE}/astar-island/rounds").json()
    lb = session.get(f"{BASE}/astar-island/leaderboard").json()

    print("\n=== Rounds ===")
    for r in (rounds if isinstance(rounds, list) else []):
        print(f"  Round {r.get('round_number')}: {r.get('status')} (closes: {r.get('closes_at')})")

    print("\n=== My Results ===")
    for r in (my_rounds if isinstance(my_rounds, list) else []):
        print(f"  Round {r.get('round_number')}: {r.get('status')}")
        print(f"    Submitted: {r.get('seeds_submitted')}/5, Queries: {r.get('queries_used')}/{r.get('queries_max')}")
        if r.get("round_score") is not None:
            print(f"    Score: {r['round_score']}")
            print(f"    Seeds: {r.get('seed_scores')}")
            print(f"    Rank: {r.get('rank')}/{r.get('total_teams')}")
        else:
            print(f"    Score: pending")

    print("\n=== Leaderboard (top 10) ===")
    for entry in (lb if isinstance(lb, list) else [])[:10]:
        verified = "✓" if entry.get("is_verified") else " "
        print(f"  {entry.get('rank', '?')}. [{verified}] {entry.get('team_name', '?')} — "
              f"score: {entry.get('weighted_score', '?')}")

    return my_rounds, rounds, lb


def fetch_analysis(session, round_id, seeds_count=5):
    """Fetch ground truth analysis for a completed round."""
    print(f"\n=== Fetching analysis for round {round_id} ===")
    analyses = {}

    for seed_idx in range(seeds_count):
        try:
            data = session.get(
                f"{BASE}/astar-island/analysis/{round_id}/{seed_idx}"
            ).json()

            if "ground_truth" in data:
                gt = np.array(data["ground_truth"])
                pred = np.array(data["prediction"])
                score = data.get("score")
                init = np.array(data.get("initial_grid", []))

                analyses[seed_idx] = {
                    "ground_truth": gt,
                    "prediction": pred,
                    "score": score,
                    "initial_grid": init,
                }

                print(f"  Seed {seed_idx}: score={score}")
                print(f"    GT shape: {gt.shape}, Pred shape: {pred.shape}")

                # Analyze where we were wrong
                if gt.shape == pred.shape:
                    gt_argmax = gt.argmax(axis=-1)
                    pred_argmax = pred.argmax(axis=-1)
                    match_pct = (gt_argmax == pred_argmax).mean() * 100
                    print(f"    Argmax match: {match_pct:.1f}%")

                    # Per-class accuracy
                    for cls in range(6):
                        gt_mask = gt_argmax == cls
                        if gt_mask.any():
                            correct = (pred_argmax[gt_mask] == cls).mean() * 100
                            print(f"    Class {cls}: {correct:.1f}% correct ({gt_mask.sum()} cells)")

            else:
                print(f"  Seed {seed_idx}: analysis not available yet")

        except Exception as e:
            print(f"  Seed {seed_idx}: error fetching analysis: {e}")

    return analyses


def calibrate_from_analysis(analyses):
    """Extract transition rates from ground truth to calibrate future predictions."""
    if not analyses:
        print("\nNo analysis data to calibrate from.")
        return

    print("\n=== Calibration from Ground Truth ===")

    # Aggregate transition statistics
    # For each initial terrain type, what's the ground truth distribution?
    transitions = {}  # init_cls -> list of gt distributions

    for seed_idx, data in analyses.items():
        gt = data["ground_truth"]  # (H, W, 6)
        init = data["initial_grid"]  # (H, W) with terrain codes

        if init.size == 0:
            continue

        H, W = init.shape
        for y in range(H):
            for x in range(W):
                init_code = int(init[y, x])
                # Map to class
                cls_map = {10: 0, 11: 0, 0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5}
                init_cls = cls_map.get(init_code, 0)

                gt_dist = gt[y, x]  # 6-element probability vector
                key = init_cls
                if key not in transitions:
                    transitions[key] = []
                transitions[key].append(gt_dist)

    print("\nAverage ground truth distributions by initial terrain type:")
    class_names = ["Empty/Ocean", "Settlement", "Port", "Ruin", "Forest", "Mountain"]

    calibration = {}
    for init_cls in sorted(transitions.keys()):
        dists = np.array(transitions[init_cls])
        avg = dists.mean(axis=0)
        calibration[init_cls] = avg.tolist()
        print(f"\n  Initial class {init_cls} ({class_names[init_cls]}):")
        print(f"    N={len(dists)} cells")
        for c in range(6):
            bar = "█" * int(avg[c] * 50)
            print(f"    → {class_names[c]:15s}: {avg[c]:.4f} {bar}")

    # Save calibration
    Path("calibration.json").write_text(
        json.dumps(calibration, indent=2), encoding="utf-8"
    )
    print(f"\nCalibration saved to calibration.json")

    return calibration


def main():
    parser = argparse.ArgumentParser(description="Astar Island Monitor")
    parser.add_argument("--token", default=os.getenv("ASTAR_ISLAND_TOKEN"))
    parser.add_argument("--poll", action="store_true", help="Poll every 60s")
    args = parser.parse_args()

    if not args.token:
        parser.error("Token required")

    session = create_session(args.token)

    while True:
        my_rounds, rounds, lb = check_status(session)

        # Check for completed rounds with analysis available
        for r in (my_rounds if isinstance(my_rounds, list) else []):
            if r.get("status") == "completed" and r.get("round_score") is not None:
                round_id = r["id"]
                seeds = r.get("seeds_count", 5)

                # Check if we already have analysis
                if not Path(f"analysis_round_{r.get('round_number', round_id)}.json").exists():
                    analyses = fetch_analysis(session, round_id, seeds)
                    if analyses:
                        # Save raw analysis
                        save_data = {}
                        for si, a in analyses.items():
                            save_data[str(si)] = {
                                "score": a["score"],
                                "ground_truth": a["ground_truth"].tolist(),
                                "prediction": a["prediction"].tolist(),
                            }
                        Path(f"analysis_round_{r.get('round_number', round_id)}.json").write_text(
                            json.dumps(save_data, separators=(',', ':')), encoding="utf-8"
                        )
                        print(f"Analysis saved for round {r.get('round_number')}")

                        # Calibrate
                        calibrate_from_analysis(analyses)

        if not args.poll:
            break

        print(f"\nPolling again in 60s...")
        time.sleep(60)


if __name__ == "__main__":
    main()
