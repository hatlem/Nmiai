#!/usr/bin/env python3
"""
Calibration tool: Run simulator and compare transition rates against ground truth.

Usage:
    python calibrate.py                    # Run with defaults, compare to ground truth
    python calibrate.py --rounds 500       # More MC runs for accuracy
    python calibrate.py --sweep            # Parameter sweep to find best fit
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from simulator import NorseSimulator, TERRAIN_TO_CLASS, NUM_CLASSES

CALIBRATION_PATH = Path(__file__).parent / "calibration.json"


def load_ground_truth() -> dict[int, np.ndarray]:
    """Load ground truth transition rates from calibration.json."""
    with open(CALIBRATION_PATH) as f:
        raw = json.load(f)
    gt = {}
    for key, values in raw.items():
        if key.startswith("_"):
            continue
        gt[int(key)] = np.array(values, dtype=np.float64)
    return gt


def build_test_grid():
    """Build a representative 40x40 grid for calibration testing."""
    rng = np.random.default_rng(42)
    size = 40
    grid = np.full((size, size), 11, dtype=np.int32)  # Plains

    # Ocean border
    grid[0, :] = 10; grid[-1, :] = 10; grid[:, 0] = 10; grid[:, -1] = 10
    # Fjords
    for _ in range(3):
        x = rng.integers(5, 35)
        for y in range(rng.integers(3, 8)):
            grid[y + 1, x] = 10
    for _ in range(2):
        y = rng.integers(5, 35)
        for x in range(rng.integers(3, 6)):
            grid[y, x + 1] = 10

    # Mountain chains
    for _ in range(3):
        y, x = rng.integers(5, 35), rng.integers(5, 35)
        for step in range(rng.integers(4, 10)):
            if 1 <= y < size - 1 and 1 <= x < size - 1:
                grid[y, x] = 5
                dy, dx = rng.choice([(-1, 0), (1, 0), (0, -1), (0, 1)])
                y, x = y + dy, x + dx

    # Forest patches
    for _ in range(8):
        cy, cx = rng.integers(3, 37), rng.integers(3, 37)
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                ny, nx = cy + dy, cx + dx
                if 1 <= ny < size - 1 and 1 <= nx < size - 1:
                    if grid[ny, nx] == 11 and rng.random() < 0.6:
                        grid[ny, nx] = 4

    # Place settlements: mix of coastal (ports) and inland
    settlements = []
    placed = []

    # First place 4 coastal settlements (ports) near ocean
    ocean_adj = []
    for y in range(2, size - 2):
        for x in range(2, size - 2):
            if grid[y, x] in (11, 0) and np.any(grid[max(0,y-1):y+2, max(0,x-1):x+2] == 10):
                ocean_adj.append((y, x))
    rng.shuffle(ocean_adj)
    for y, x in ocean_adj:
        if len([s for s in settlements if s.get("has_port")]) >= 4:
            break
        if any(abs(y - py) + abs(x - px) < 5 for py, px in placed):
            continue
        placed.append((y, x))
        settlements.append({"x": int(x), "y": int(y), "has_port": True, "alive": True})
        grid[y, x] = 2  # Port

    # Then place 8 inland settlements
    for _ in range(8):
        for attempt in range(100):
            y, x = rng.integers(3, 37), rng.integers(3, 37)
            if grid[y, x] not in (11, 0, 4):
                continue
            if any(abs(y - py) + abs(x - px) < 5 for py, px in placed):
                continue
            placed.append((y, x))
            settlements.append({"x": int(x), "y": int(y), "has_port": False, "alive": True})
            grid[y, x] = 1
            break

    # Add some initial ruins
    for _ in range(5):
        for attempt in range(50):
            y, x = rng.integers(3, 37), rng.integers(3, 37)
            if grid[y, x] in (11, 0) and not any(abs(y-py)+abs(x-px) < 3 for py, px in placed):
                grid[y, x] = 3
                placed.append((y, x))
                break

    return grid, settlements


def compute_transition_rates(
    initial_grid: np.ndarray,
    params: dict,
    settlements: list,
    n_runs: int = 200,
) -> np.ndarray:
    """Run MC simulations and compute empirical transition matrix (6x6)."""
    sim = NorseSimulator(initial_grid, settlements, params)
    H, W = sim.H, sim.W

    # Map initial grid to classes
    init_cls = np.zeros_like(initial_grid, dtype=np.int32)
    for code, cls in TERRAIN_TO_CLASS.items():
        init_cls[initial_grid == code] = cls

    # Accumulate transitions
    trans_counts = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.float64)

    for seed in range(n_runs):
        final_grid = sim.run(seed)
        final_cls = np.zeros_like(final_grid, dtype=np.int32)
        for code, cls in TERRAIN_TO_CLASS.items():
            final_cls[final_grid == code] = cls

        for from_cls in range(NUM_CLASSES):
            mask = init_cls == from_cls
            if mask.any():
                for to_cls in range(NUM_CLASSES):
                    trans_counts[from_cls, to_cls] += (final_cls[mask] == to_cls).sum()

    # Normalize rows
    row_sums = trans_counts.sum(axis=1, keepdims=True)
    row_sums = np.maximum(row_sums, 1.0)
    return trans_counts / row_sums


def compare_to_ground_truth(
    sim_trans: np.ndarray,
    gt: dict[int, np.ndarray],
) -> float:
    """Compare simulator transition matrix to ground truth. Returns total KL divergence."""
    class_names = ["Empty", "Settlement", "Port", "Ruin", "Forest", "Mountain"]
    total_kl = 0.0
    total_l1 = 0.0

    for from_cls in range(NUM_CLASSES):
        if from_cls not in gt:
            continue
        gt_row = gt[from_cls]
        sim_row = sim_trans[from_cls]

        # Skip mountain (class 5) — always static
        if from_cls == 5:
            continue

        # Per-class comparison
        print(f"\n  {class_names[from_cls]} (class {from_cls}):")
        row_kl = 0.0
        for to_cls in range(NUM_CLASSES):
            p = gt_row[to_cls]
            q = max(sim_row[to_cls], 1e-6)
            diff = sim_row[to_cls] - p
            sign = "+" if diff > 0 else ""
            kl_contrib = p * np.log(p / q) if p > 0.001 else 0.0
            row_kl += kl_contrib

            if abs(diff) > 0.005:
                print(f"    → {class_names[to_cls]:12s}: "
                      f"GT={p:.3f}  SIM={sim_row[to_cls]:.3f}  "
                      f"({sign}{diff:.3f})  KL={kl_contrib:.4f}")

        l1 = np.abs(gt_row - sim_row).sum()
        total_kl += row_kl
        total_l1 += l1
        print(f"    Row KL: {row_kl:.4f}  L1: {l1:.3f}")

    print(f"\n  TOTAL KL: {total_kl:.4f}  TOTAL L1: {total_l1:.3f}")
    return total_kl


def parameter_sweep(grid, settlements, gt, n_runs=100):
    """Sweep key parameters to find best fit to ground truth."""
    best_kl = float("inf")
    best_params = None

    # Define sweep ranges
    winter_range = [0.20, 0.25, 0.28, 0.32, 0.36]
    aggression_range = [0.12, 0.16, 0.20, 0.25]
    expansion_range = [0.30, 0.38, 0.45, 0.55]
    forest_rate_range = [0.08, 0.12, 0.16, 0.20]

    total = len(winter_range) * len(aggression_range) * len(expansion_range) * len(forest_rate_range)
    print(f"Sweeping {total} parameter combinations ({n_runs} MC runs each)...")
    i = 0

    for ws in winter_range:
        for fa in aggression_range:
            for er in expansion_range:
                for fg in forest_rate_range:
                    i += 1
                    params = {
                        "winter_severity": ws,
                        "faction_aggression": fa,
                        "trade_activity": 0.5,
                        "forest_growth_rate": fg,
                        "expansion_rate": er,
                        "raid_range": 5.0,
                        "food_per_forest": 0.45,
                        "port_development_threshold": 0.5,
                        "ruin_reclaim_rate": 0.09,
                    }

                    sim_trans = compute_transition_rates(grid, params, settlements, n_runs)

                    # Compute KL against ground truth
                    kl = 0.0
                    for from_cls in range(5):  # Skip mountain
                        if from_cls not in gt:
                            continue
                        gt_row = gt[from_cls]
                        sim_row = sim_trans[from_cls]
                        for to_cls in range(NUM_CLASSES):
                            p = gt_row[to_cls]
                            q = max(sim_row[to_cls], 1e-6)
                            if p > 0.001:
                                kl += p * np.log(p / q)

                    if kl < best_kl:
                        best_kl = kl
                        best_params = params.copy()
                        print(f"  [{i}/{total}] NEW BEST KL={kl:.4f}: "
                              f"ws={ws} fa={fa} er={er} fg={fg}")

    print(f"\nBest params (KL={best_kl:.4f}):")
    for k, v in best_params.items():
        print(f"  {k}: {v}")
    return best_params


def main():
    parser = argparse.ArgumentParser(description="Calibrate simulator against ground truth")
    parser.add_argument("--rounds", type=int, default=200, help="MC runs per evaluation")
    parser.add_argument("--sweep", action="store_true", help="Run parameter sweep")
    args = parser.parse_args()

    gt = load_ground_truth()
    grid, settlements = build_test_grid()

    print(f"Test grid: {grid.shape}, {len(settlements)} settlements")
    print(f"Ground truth loaded: {len(gt)} classes")

    if args.sweep:
        best_params = parameter_sweep(grid, settlements, gt, n_runs=args.rounds)
        print("\nValidating best params with more runs...")
        sim_trans = compute_transition_rates(grid, best_params, settlements, n_runs=args.rounds * 3)
        compare_to_ground_truth(sim_trans, gt)
    else:
        from simulator import DEFAULT_PARAMS
        print(f"\nRunning {args.rounds} MC simulations with DEFAULT_PARAMS...")
        sim_trans = compute_transition_rates(grid, DEFAULT_PARAMS, settlements, n_runs=args.rounds)
        compare_to_ground_truth(sim_trans, gt)


if __name__ == "__main__":
    main()
