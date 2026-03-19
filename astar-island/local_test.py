#!/usr/bin/env python3
"""
Local test harness for Astar Island.

Simulates a full round locally:
  1. Generates a random map with hidden parameters
  2. Computes ground truth (500 MC runs per seed)
  3. Lets our pipeline query, predict, and score
  4. Reports entropy-weighted KL divergence score (same as competition)

Usage:
    python3 local_test.py                    # Quick test (100 GT runs)
    python3 local_test.py --gt-runs 500      # More accurate ground truth
    python3 local_test.py --no-query         # Test without queries (prior only)
"""

import argparse
import time
from typing import Dict, List, Tuple

import numpy as np

from simulator import (
    NorseSimulator, OCEAN, PLAINS, EMPTY, SETTLEMENT, PORT, RUIN, FOREST, MOUNTAIN,
    NUM_CLASSES, TERRAIN_TO_CLASS,
)
from query_optimizer import QueryOptimizer
from inference import ParameterInference
from swarm import SwarmCoordinator
from priors import PROB_FLOOR


# ── Map Generation ──────────────────────────────────────────────────────────


def generate_map(
    W: int = 40, H: int = 40, n_settlements: int = 15, seed: int = 42
) -> Tuple[np.ndarray, List[Dict]]:
    """Generate a random map similar to the competition."""
    rng = np.random.default_rng(seed)
    grid = np.full((H, W), PLAINS, dtype=np.int32)

    # Ocean border
    grid[0, :] = OCEAN
    grid[-1, :] = OCEAN
    grid[:, 0] = OCEAN
    grid[:, -1] = OCEAN

    # Fjords: random ocean intrusions from edges
    for _ in range(rng.integers(2, 5)):
        edge = rng.integers(4)
        if edge == 0:  # top
            x, y = rng.integers(2, W - 2), 0
            dy, dx = 1, 0
        elif edge == 1:  # bottom
            x, y = rng.integers(2, W - 2), H - 1
            dy, dx = -1, 0
        elif edge == 2:  # left
            x, y = 0, rng.integers(2, H - 2)
            dy, dx = 0, 1
        else:  # right
            x, y = W - 1, rng.integers(2, H - 2)
            dy, dx = 0, -1

        length = rng.integers(3, 12)
        for step in range(length):
            ny, nx = y + dy * step, x + dx * step
            # Wiggle
            if rng.random() < 0.3:
                ny += rng.choice([-1, 0, 1])
                nx += rng.choice([-1, 0, 1])
            if 0 <= ny < H and 0 <= nx < W:
                grid[ny, nx] = OCEAN

    # Mountain chains (random walks)
    for _ in range(rng.integers(1, 4)):
        y, x = rng.integers(3, H - 3), rng.integers(3, W - 3)
        length = rng.integers(4, 10)
        for _ in range(length):
            if 1 <= y < H - 1 and 1 <= x < W - 1 and grid[y, x] != OCEAN:
                grid[y, x] = MOUNTAIN
            dy = rng.choice([-1, 0, 1])
            dx = rng.choice([-1, 0, 1])
            y, x = y + dy, x + dx

    # Forest patches
    for _ in range(rng.integers(5, 12)):
        cy, cx = rng.integers(2, H - 2), rng.integers(2, W - 2)
        radius = rng.integers(1, 4)
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                ny, nx = cy + dy, cx + dx
                if (0 <= ny < H and 0 <= nx < W
                        and grid[ny, nx] == PLAINS
                        and abs(dy) + abs(dx) <= radius + rng.integers(0, 2)):
                    grid[ny, nx] = FOREST

    # Place settlements on valid land, spaced apart
    settlements = []
    land_cells = [(y, x) for y in range(H) for x in range(W)
                  if grid[y, x] in (PLAINS, EMPTY)]
    rng.shuffle(land_cells)

    min_dist = 4
    for y, x in land_cells:
        if len(settlements) >= n_settlements:
            break
        # Check distance to existing settlements
        too_close = False
        for s in settlements:
            if abs(y - s["y"]) + abs(x - s["x"]) < min_dist:
                too_close = True
                break
        if too_close:
            continue

        # Check if coastal for port
        is_coastal = False
        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ny, nx = y + dy, x + dx
            if 0 <= ny < H and 0 <= nx < W and grid[ny, nx] == OCEAN:
                is_coastal = True
                break

        has_port = is_coastal and rng.random() < 0.3
        settlements.append({
            "x": x, "y": y,
            "has_port": has_port,
            "alive": True,
            "owner_id": len(settlements),
        })

        grid[y, x] = PORT if has_port else SETTLEMENT

    return grid, settlements


def generate_hidden_params(seed: int = 123) -> Dict[str, float]:
    """Generate random hidden parameters."""
    rng = np.random.default_rng(seed)
    return {
        "winter_severity": float(rng.uniform(0.1, 0.8)),
        "faction_aggression": float(rng.uniform(0.1, 0.7)),
        "trade_activity": float(rng.uniform(0.1, 0.8)),
        "forest_growth_rate": float(rng.uniform(0.02, 0.3)),
        "expansion_rate": float(rng.uniform(0.1, 0.4)),
        "raid_range": float(rng.uniform(3.0, 8.0)),
        "food_per_forest": float(rng.uniform(0.1, 0.6)),
        "port_development_threshold": float(rng.uniform(0.2, 0.8)),
        "ruin_reclaim_rate": float(rng.uniform(0.05, 0.4)),
    }


# ── Ground Truth ────────────────────────────────────────────────────────────


def compute_ground_truth(
    grid: np.ndarray, settlements: List[Dict], params: Dict[str, float],
    n_runs: int = 500,
) -> np.ndarray:
    """Compute ground truth probability distribution via MC simulation."""
    return NorseSimulator.run_monte_carlo(
        grid, settlements, params, n_runs=n_runs,
    )


# ── Scoring ─────────────────────────────────────────────────────────────────


def score_prediction(
    prediction: np.ndarray, ground_truth: np.ndarray,
) -> float:
    """
    Score using entropy-weighted KL divergence (same as competition).

    score = max(0, min(100, 100 * exp(-3 * weighted_kl)))
    """
    eps = 1e-10
    p = np.clip(ground_truth, eps, 1.0)
    q = np.clip(prediction, eps, 1.0)

    # Normalize
    p = p / p.sum(axis=-1, keepdims=True)
    q = q / q.sum(axis=-1, keepdims=True)

    # Per-cell entropy
    entropy = -np.sum(p * np.log(p + eps), axis=-1)

    # Per-cell KL divergence
    kl = np.sum(p * np.log((p + eps) / (q + eps)), axis=-1)

    # Weighted average
    total_entropy = entropy.sum()
    if total_entropy < eps:
        return 100.0

    weighted_kl = (entropy * kl).sum() / total_entropy

    score = max(0.0, min(100.0, 100.0 * np.exp(-3.0 * weighted_kl)))
    return score


# ── Simulate Query ──────────────────────────────────────────────────────────


def simulate_query(
    grid: np.ndarray, settlements: List[Dict], params: Dict[str, float],
    seed_index: int, vx: int, vy: int, vw: int, vh: int,
) -> Dict:
    """Simulate one query (like the API /simulate endpoint)."""
    sim = NorseSimulator(grid, settlements, params)
    # Use a random seed for each query (stochastic)
    rng_seed = np.random.randint(0, 2**31)
    final_grid = sim.run(seed=rng_seed)

    # Extract viewport
    viewport_grid = final_grid[vy:vy+vh, vx:vx+vw].tolist()

    # Extract settlements in viewport
    sim_settlements = sim._init_settlements()
    # Re-run to get settlement states (quick hack: just return grid data)
    viewport_settlements = []
    for s in settlements:
        if vx <= s["x"] < vx + vw and vy <= s["y"] < vy + vh:
            viewport_settlements.append({
                "x": s["x"], "y": s["y"],
                "has_port": s.get("has_port", False),
                "alive": True,
            })

    return {
        "grid": viewport_grid,
        "settlements": viewport_settlements,
        "viewport": {"x": vx, "y": vy, "w": vw, "h": vh},
    }


# ── Main Test ───────────────────────────────────────────────────────────────


def run_test(
    n_seeds: int = 5,
    n_gt_runs: int = 200,
    budget: int = 50,
    use_queries: bool = True,
    map_seed: int = 42,
    param_seed: int = 123,
):
    W, H = 40, 40

    print("=" * 60)
    print("ASTAR ISLAND LOCAL TEST")
    print("=" * 60)

    # Generate hidden parameters (same for all seeds, like the real game)
    true_params = generate_hidden_params(param_seed)
    print(f"\nTrue hidden parameters:")
    for k, v in true_params.items():
        print(f"  {k}: {v:.3f}")

    # Generate maps for each seed (different map layout per seed)
    initial_states = []
    ground_truths = []

    print(f"\nGenerating {n_seeds} seeds with ground truth ({n_gt_runs} MC runs each)...")
    for seed_idx in range(n_seeds):
        seed = map_seed * 100 + seed_idx
        n_sett = np.random.RandomState(seed).randint(20, 60)
        grid, settlements = generate_map(W, H, n_settlements=n_sett, seed=seed)

        initial_states.append({
            "grid": grid.tolist(),
            "settlements": settlements,
        })

        t0 = time.time()
        gt = compute_ground_truth(grid, settlements, true_params, n_runs=n_gt_runs)
        elapsed = time.time() - t0

        ground_truths.append(gt)
        n_sett_actual = len(settlements)
        n_port = sum(1 for s in settlements if s.get("has_port"))
        print(f"  Seed {seed_idx}: {n_sett_actual} settlements ({n_port} ports), "
              f"GT computed in {elapsed:.1f}s")

    # ── Query phase ──────────────────────────────────────────────────────

    observations = {s: [[None] * W for _ in range(H)] for s in range(n_seeds)}
    counts = {s: np.zeros((H, W, NUM_CLASSES), dtype=np.int32) for s in range(n_seeds)}
    settlements_data = {s: [] for s in range(n_seeds)}

    if use_queries and budget > 0:
        print(f"\nPlanning {budget} queries...")
        optimizer = QueryOptimizer(
            W=W, H=H, seeds_count=n_seeds,
            budget=budget, initial_states=initial_states,
        )
        plan = optimizer.plan_queries()
        print(optimizer.summary())

        print(f"\nExecuting {len(plan)} queries...")
        for qi, (seed_idx, vx, vy, vw, vh) in enumerate(plan):
            grid = np.asarray(initial_states[seed_idx]["grid"], dtype=np.int32)
            setts = initial_states[seed_idx]["settlements"]

            result = simulate_query(grid, setts, true_params, seed_idx, vx, vy, vw, vh)

            # Store observations
            for row_idx, row in enumerate(result["grid"]):
                gy = vy + row_idx
                if gy >= H:
                    break
                for col_idx, cell in enumerate(row):
                    gx = vx + col_idx
                    if gx >= W:
                        break
                    observations[seed_idx][gy][gx] = cell
                    cls = TERRAIN_TO_CLASS.get(cell, 0)
                    counts[seed_idx][gy, gx, cls] += 1

            if result.get("settlements"):
                settlements_data[seed_idx].append({"settlements": result["settlements"]})

        # Coverage stats
        for s in range(n_seeds):
            n_obs = int(counts[s].sum())
            n_cells = int((counts[s].sum(axis=2) > 0).sum())
            print(f"  Seed {s}: {n_obs} observations, {n_cells}/{W*H} cells covered")
    else:
        print("\nSkipping queries (prior-only mode)")

    # ── Inference ────────────────────────────────────────────────────────

    print("\nInferring parameters...")
    inferrer = ParameterInference(initial_states, observations, counts,
                                   settlements_data=settlements_data)
    inferred_params = inferrer.infer()
    print(f"  Inferred: {inferred_params}")
    print(f"\n  Parameter comparison:")
    for k in true_params:
        true_v = true_params[k]
        inf_v = inferred_params.get(k, "?")
        if isinstance(inf_v, (int, float)):
            err = abs(true_v - inf_v)
            print(f"    {k:30s}: true={true_v:.3f}  inferred={inf_v:.3f}  err={err:.3f}")

    try:
        posterior_samples = inferrer.infer_posterior(n_samples=20)
        print(f"  Posterior samples: {len(posterior_samples)}")
    except Exception as e:
        print(f"  Posterior sampling failed ({e})")
        posterior_samples = [inferred_params]

    # ── Prediction ───────────────────────────────────────────────────────

    print("\nRunning swarm predictions...")
    swarm = SwarmCoordinator(
        initial_states=initial_states,
        W=W, H=H,
        seeds_count=n_seeds,
        inferred_params=inferred_params,
        posterior_samples=posterior_samples,
    )
    predictions = swarm.predict_all(
        counts=counts,
        observations=observations,
        settlements_data=settlements_data,
    )

    # ── Scoring ──────────────────────────────────────────────────────────

    print("\n" + "=" * 60)
    print("SCORES")
    print("=" * 60)

    scores = []
    for seed_idx in range(n_seeds):
        pred = predictions[seed_idx]
        pred = np.maximum(pred, PROB_FLOOR)
        pred /= pred.sum(axis=-1, keepdims=True)

        gt = ground_truths[seed_idx]
        s = score_prediction(pred, gt)
        scores.append(s)
        print(f"  Seed {seed_idx}: {s:.2f}")

    avg = np.mean(scores)
    print(f"\n  AVERAGE SCORE: {avg:.2f}")
    print(f"  (Competition top teams: ~85)")

    # ── Baseline comparison ──────────────────────────────────────────────

    print("\n--- Baseline comparisons ---")

    # Uniform baseline
    uniform_scores = []
    for seed_idx in range(n_seeds):
        uniform = np.full((H, W, NUM_CLASSES), 1.0 / NUM_CLASSES)
        s = score_prediction(uniform, ground_truths[seed_idx])
        uniform_scores.append(s)
    print(f"  Uniform prior:    {np.mean(uniform_scores):.2f}")

    # Perfect params (cheating baseline)
    print("\n  Running with TRUE parameters (oracle)...")
    swarm_oracle = SwarmCoordinator(
        initial_states=initial_states,
        W=W, H=H,
        seeds_count=n_seeds,
        inferred_params=true_params,
        posterior_samples=[true_params],
    )
    oracle_preds = swarm_oracle.predict_all(
        counts=counts,
        observations=observations,
        settlements_data=settlements_data,
    )
    oracle_scores = []
    for seed_idx in range(n_seeds):
        pred = oracle_preds[seed_idx]
        pred = np.maximum(pred, PROB_FLOOR)
        pred /= pred.sum(axis=-1, keepdims=True)
        s = score_prediction(pred, ground_truths[seed_idx])
        oracle_scores.append(s)
    print(f"  Oracle params:    {np.mean(oracle_scores):.2f}")

    print(f"\n  Our improvement over uniform: +{avg - np.mean(uniform_scores):.2f}")
    print(f"  Gap to oracle:               -{np.mean(oracle_scores) - avg:.2f}")

    return avg


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt-runs", type=int, default=200,
                        help="MC runs for ground truth (default 200)")
    parser.add_argument("--budget", type=int, default=50,
                        help="Query budget (default 50)")
    parser.add_argument("--no-query", action="store_true",
                        help="Skip queries (prior-only)")
    parser.add_argument("--seeds", type=int, default=5,
                        help="Number of map seeds (default 5)")
    parser.add_argument("--map-seed", type=int, default=42)
    parser.add_argument("--param-seed", type=int, default=123)
    args = parser.parse_args()

    run_test(
        n_seeds=args.seeds,
        n_gt_runs=args.gt_runs,
        budget=args.budget,
        use_queries=not args.no_query,
        map_seed=args.map_seed,
        param_seed=args.param_seed,
    )
