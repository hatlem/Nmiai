#!/usr/bin/env python3
"""
Calibrate simulator against real competition ground truth.

For each completed round where we have GT, run our simulator with
many parameter combinations and measure how well the output matches GT.
This tells us both:
1. The true hidden parameters for that round
2. How accurate our simulator is (systematic biases)

Usage:
    source ../.env.ainm && ASTAR_ISLAND_TOKEN=$AINM_TOKEN python3 calibrate_simulator.py
"""

import json
import os
import sys
import time

import numpy as np
import requests

from abc_inference import (
    PARAM_BOUNDS,
    NUM_CLASSES,
    TERRAIN_TO_CLASS,
    _classify_grid,
    generate_param_grid,
)
from simulator import NorseSimulator

BASE_URL = "https://api.ainm.no"
CLASS_NAMES = ["Empty", "Settlement", "Port", "Ruin", "Forest", "Mountain"]


def get_session():
    token = os.getenv("ASTAR_ISLAND_TOKEN")
    if not token:
        print("Set ASTAR_ISLAND_TOKEN env var")
        sys.exit(1)
    s = requests.Session()
    s.headers["Authorization"] = f"Bearer {token}"
    return s


def compute_gt_fit(sim_probs, gt, init_cls):
    """Compare simulated distribution against actual ground truth."""
    H, W = gt.shape[:2]

    # Skip ocean and mountain (trivial)
    dynamic = ~np.isin(init_cls, [5])  # Keep ocean class 0 since it maps to Empty
    ocean_mask = np.zeros_like(init_cls, dtype=bool)
    for y in range(H):
        for x in range(W):
            if init_cls[y, x] == 0:
                # Check if it's actually ocean (GT should be ~1.0 for class 0)
                if gt[y, x, 0] > 0.99:
                    ocean_mask[y, x] = True
    dynamic = dynamic & ~ocean_mask

    if not dynamic.any():
        return float("inf"), {}

    gt_d = gt[dynamic]
    sim_d = sim_probs[dynamic]
    sim_d = np.maximum(sim_d, 1e-8)

    # KL divergence
    kl = np.sum(gt_d * np.log(gt_d / sim_d + 1e-12), axis=1)

    # Entropy weighting
    entropy = -np.sum(gt_d * np.log(gt_d + 1e-12), axis=1)
    entropy = np.maximum(entropy, 0.01)

    weighted_kl = np.sum(entropy * kl) / np.sum(entropy)

    # Competition score
    score = max(0, min(100, 100 * np.exp(-3 * weighted_kl)))

    # Per-class bias
    bias = {}
    for cls in range(NUM_CLASSES):
        for ic in range(NUM_CLASSES):
            mask = dynamic & (init_cls == ic)
            if mask.sum() > 10:
                key = f"{CLASS_NAMES[ic]}->{CLASS_NAMES[cls]}"
                sim_avg = sim_probs[mask, cls].mean()
                gt_avg = gt[mask, cls].mean()
                bias[key] = float(sim_avg - gt_avg)

    return score, bias


def calibrate_on_round(session, round_info):
    """Run parameter search against one round's GT."""
    round_id = round_info["id"]
    round_num = round_info.get("round_number", "?")

    detail = session.get(f"{BASE_URL}/astar-island/rounds/{round_id}").json()
    seeds_count = detail.get("seeds_count", 5)

    print(f"\n{'='*70}")
    print(f"Round {round_num}")
    print(f"{'='*70}")

    # Fetch GT for all seeds
    seed_data = []
    for seed_idx in range(seeds_count):
        try:
            analysis = session.get(
                f"{BASE_URL}/astar-island/analysis/{round_id}/{seed_idx}"
            ).json()
            if "ground_truth" not in analysis:
                continue

            gt = np.array(analysis["ground_truth"])
            init_grid = np.array(analysis.get(
                "initial_grid",
                detail["initial_states"][seed_idx]["grid"]
            ))
            init_cls = _classify_grid(init_grid)

            seed_data.append({
                "grid": init_grid,
                "settlements": detail["initial_states"][seed_idx].get("settlements", []),
                "gt": gt,
                "init_cls": init_cls,
            })
        except Exception as e:
            print(f"  Seed {seed_idx}: error {e}")

    if not seed_data:
        return None

    # Use first 2 seeds for calibration (faster)
    cal_seeds = seed_data[:2]

    # Generate parameter candidates
    n_candidates = 80
    mc_per = 30  # More MC for comparing against full GT
    candidates = generate_param_grid(n_candidates, seed=round_num * 7)

    print(f"  Testing {n_candidates} parameter sets × {mc_per} MC × {len(cal_seeds)} seeds...")
    t0 = time.time()

    results = []
    for ci, params in enumerate(candidates):
        total_score = 0.0

        for sd in cal_seeds:
            probs = NorseSimulator.run_monte_carlo(
                sd["grid"], sd["settlements"], params,
                n_runs=mc_per,
                seeds=list(range(ci * 100, ci * 100 + mc_per)),
            )
            score, _ = compute_gt_fit(probs, sd["gt"], sd["init_cls"])
            total_score += score

        avg_score = total_score / len(cal_seeds)
        results.append((avg_score, params))

        if (ci + 1) % 20 == 0:
            best = max(r[0] for r in results)
            elapsed = time.time() - t0
            print(f"    {ci+1}/{n_candidates}: best score={best:.1f}, "
                  f"elapsed={elapsed:.0f}s")

    results.sort(key=lambda x: x[0], reverse=True)  # Higher score = better
    best_score, best_params = results[0]

    print(f"\n  Best simulator score against GT: {best_score:.1f}")
    print(f"  Best params: {json.dumps({k: round(v, 3) for k, v in best_params.items()}, indent=2)}")

    # Analyze systematic bias with best params
    print(f"\n  Systematic bias (sim - GT) with best params:")
    for sd in cal_seeds[:1]:
        probs = NorseSimulator.run_monte_carlo(
            sd["grid"], sd["settlements"], best_params,
            n_runs=200,  # More runs for accurate bias measurement
        )
        _, biases = compute_gt_fit(probs, sd["gt"], sd["init_cls"])
        for key in sorted(biases.keys()):
            if abs(biases[key]) > 0.01:
                print(f"    {key:>25}: {biases[key]:+.4f}")

    return {
        "round_num": round_num,
        "best_score": best_score,
        "best_params": best_params,
        "biases": biases,
    }


def main():
    session = get_session()

    rounds = session.get(f"{BASE_URL}/astar-island/rounds").json()
    if isinstance(rounds, dict):
        rounds = rounds.get("rounds", rounds.get("data", []))

    completed = [r for r in rounds if r.get("status") == "completed"]
    completed.sort(key=lambda r: r.get("round_number", 0))
    print(f"Found {len(completed)} completed rounds")

    all_results = []
    for r in completed[-3:]:  # Last 3 rounds (most relevant)
        result = calibrate_on_round(session, r)
        if result:
            all_results.append(result)

    # Summary
    print(f"\n{'='*70}")
    print("CALIBRATION SUMMARY")
    print(f"{'='*70}")

    if not all_results:
        print("No results!")
        return

    print(f"\nBest simulator scores against GT:")
    for r in all_results:
        print(f"  Round {r['round_num']}: {r['best_score']:.1f}")

    # Average best score tells us simulator ceiling
    avg_best = np.mean([r["best_score"] for r in all_results])
    print(f"\n  Average best possible with our simulator: {avg_best:.1f}")
    print(f"  This is our CEILING — even with perfect params, we can't score higher.")

    # Save results
    with open("simulator_calibration.json", "w") as f:
        json.dump([{
            "round_num": r["round_num"],
            "best_score": r["best_score"],
            "best_params": {k: round(v, 4) for k, v in r["best_params"].items()},
        } for r in all_results], f, indent=2)
    print("\nSaved simulator_calibration.json")


if __name__ == "__main__":
    main()
