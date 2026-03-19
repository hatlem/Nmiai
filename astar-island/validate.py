#!/usr/bin/env python3
"""
Validation script for the Astar Island prediction pipeline.

Tests the full pipeline (observation -> inference -> MC simulation -> prediction)
against known ground truth, scoring with the competition's entropy-weighted
KL divergence metric.

Usage:
    /opt/homebrew/bin/python3 validate.py
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from simulator import (
    DEFAULT_PARAMS,
    EMPTY,
    FOREST,
    MOUNTAIN,
    NUM_CLASSES,
    OCEAN,
    PLAINS,
    PORT,
    RUIN,
    SETTLEMENT,
    TERRAIN_TO_CLASS,
    NorseSimulator,
)
from inference import ParameterInference
from prediction import PredictionEngine
from query_optimizer import QueryOptimizer
from swarm import SwarmCoordinator


# ── Scoring (exact competition formula) ──────────────────────────────────────

def compute_entropy(probs: np.ndarray) -> np.ndarray:
    """Per-cell entropy of ground truth distribution. Shape (H, W)."""
    p = np.clip(probs, 1e-12, 1.0)
    return -(p * np.log(p)).sum(axis=2)


def compute_kl_divergence(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Per-cell KL(gt || pred). Shape (H, W)."""
    gt_safe = np.clip(gt, 1e-12, 1.0)
    pred_safe = np.clip(pred, 1e-12, 1.0)
    return (gt_safe * np.log(gt_safe / pred_safe)).sum(axis=2)


def score_prediction(gt: np.ndarray, pred: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Compute competition score using entropy-weighted KL divergence.

    Returns:
        (score, per_cell_kl, per_cell_entropy)
    """
    entropy = compute_entropy(gt)
    kl = compute_kl_divergence(gt, pred)

    total_entropy = entropy.sum()
    if total_entropy < 1e-10:
        return 100.0, kl, entropy

    weighted_kl = (entropy * kl).sum() / total_entropy
    score = max(0.0, min(100.0, 100.0 * np.exp(-3.0 * weighted_kl)))
    return score, kl, entropy


# ── Test map generation ──────────────────────────────────────────────────────

def create_test_map(n_settlements: int = 40, seed: int = 42) -> Tuple[np.ndarray, List[dict]]:
    """Create a realistic 40x40 test map with ocean borders, mountains, forests."""
    rng = np.random.default_rng(seed)
    size = 40
    grid = np.full((size, size), PLAINS, dtype=np.int32)

    # Ocean border (2 cells wide on edges)
    grid[0:2, :] = OCEAN
    grid[-2:, :] = OCEAN
    grid[:, 0:2] = OCEAN
    grid[:, -2:] = OCEAN

    # Additional ocean: a bay/inlet
    grid[0:6, 15:20] = OCEAN
    grid[35:40, 25:30] = OCEAN

    # Mountain ranges
    for _ in range(3):
        my, mx = rng.integers(5, 35), rng.integers(5, 35)
        length = rng.integers(3, 8)
        direction = rng.choice([(0, 1), (1, 0), (1, 1)])
        for i in range(length):
            ny, nx = my + i * direction[0], mx + i * direction[1]
            if 2 <= ny < 38 and 2 <= nx < 38:
                grid[ny, nx] = MOUNTAIN

    # Forest patches
    for _ in range(8):
        fy, fx = rng.integers(3, 37), rng.integers(3, 37)
        fsize = rng.integers(2, 5)
        for dy in range(-fsize, fsize + 1):
            for dx in range(-fsize, fsize + 1):
                ny, nx = fy + dy, fx + dx
                if (2 <= ny < 38 and 2 <= nx < 38
                        and grid[ny, nx] == PLAINS
                        and rng.random() < 0.6):
                    grid[ny, nx] = FOREST

    # Place settlements on valid land cells
    land_mask = (grid == PLAINS) | (grid == EMPTY)
    land_coords = list(zip(*np.where(land_mask)))
    rng.shuffle(land_coords)

    settlements = []
    placed = set()
    for y, x in land_coords:
        if len(settlements) >= n_settlements:
            break
        # Ensure minimum distance between settlements
        too_close = False
        for py, px in placed:
            if abs(y - py) + abs(x - px) < 3:
                too_close = True
                break
        if too_close:
            continue

        # Check if coastal for port
        is_coastal = False
        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ny, nx = y + dy, x + dx
            if 0 <= ny < size and 0 <= nx < size and grid[ny, nx] == OCEAN:
                is_coastal = True
                break

        has_port = is_coastal and rng.random() < 0.3
        owner_id = rng.integers(0, 6)  # 6 factions

        settlements.append({
            "x": int(x),
            "y": int(y),
            "population": float(0.8 + rng.random() * 1.5),
            "food": float(0.5 + rng.random() * 1.0),
            "wealth": float(rng.random() * 0.5),
            "defense": float(0.3 + rng.random() * 0.5),
            "has_port": has_port,
            "alive": True,
            "owner_id": owner_id,
            "tech_level": float(rng.random() * 0.3),
        })
        placed.add((y, x))
        grid[y, x] = PORT if has_port else SETTLEMENT

    return grid, settlements


# ── Ground truth generation ──────────────────────────────────────────────────

def generate_ground_truth(
    grid: np.ndarray,
    settlements: List[dict],
    params: dict,
    n_runs: int = 500,
) -> np.ndarray:
    """Run many simulations to get ground truth probability distribution."""
    print(f"  Generating ground truth ({n_runs} MC runs)...", end=" ", flush=True)
    t0 = time.time()
    gt = NorseSimulator.run_monte_carlo(grid, settlements, params, n_runs=n_runs)
    print(f"done in {time.time() - t0:.1f}s")
    return gt


# ── Simulate observation phase ───────────────────────────────────────────────

def simulate_observations(
    grid: np.ndarray,
    settlements: List[dict],
    params: dict,
    n_queries: int = 50,
    n_seeds: int = 5,
    map_seed: int = 0,
) -> Tuple[List[dict], dict, dict, dict]:
    """
    Simulate the observation phase: query viewports and collect observations.

    For validation, we simulate the "real" server by running the simulator
    with different seeds and returning the results as observations.

    Returns:
        initial_states, observations, counts, settlements_data
    """
    H, W = grid.shape
    rng = np.random.default_rng(map_seed + 1000)

    # Create per-seed initial states (same map, different simulation seeds)
    initial_states = []
    for seed_idx in range(n_seeds):
        initial_states.append({
            "grid": grid.copy(),
            "settlements": [s.copy() for s in settlements],
        })

    # Initialize observation storage
    observations: Dict[int, list] = {}
    counts: Dict[int, np.ndarray] = {}
    settlements_data: Dict[int, list] = {}

    for seed_idx in range(n_seeds):
        observations[seed_idx] = [[None] * W for _ in range(H)]
        counts[seed_idx] = np.zeros((H, W, NUM_CLASSES), dtype=np.float64)
        settlements_data[seed_idx] = []

    # Use QueryOptimizer to plan queries
    optimizer = QueryOptimizer(
        W=W, H=H, seeds_count=n_seeds,
        budget=n_queries, initial_states=initial_states,
    )
    planned = optimizer.plan_queries()

    # Execute queries: simulate the server response
    sim_cache: Dict[Tuple[int, int], np.ndarray] = {}

    for q_idx, (seed_idx, x, y, w, h) in enumerate(planned[:n_queries]):
        # Each query gets a fresh simulation run (simulating the server)
        sim_seed = rng.integers(0, 2**31)
        cache_key = (seed_idx, sim_seed)

        if cache_key not in sim_cache:
            sim = NorseSimulator(grid, settlements, params)
            final_grid = sim.run(seed=sim_seed)
            sim_cache[cache_key] = final_grid

        final_grid = sim_cache[cache_key]

        # Record observations for the viewport
        for dy in range(h):
            for dx in range(w):
                cy, cx = y + dy, x + dx
                if cy >= H or cx >= W:
                    continue
                terrain_code = int(final_grid[cy, cx])
                cls = TERRAIN_TO_CLASS.get(terrain_code, 0)

                observations[seed_idx][cy][cx] = terrain_code
                counts[seed_idx][cy, cx, cls] += 1.0

    return initial_states, observations, counts, settlements_data


# ── Full pipeline ────────────────────────────────────────────────────────────

def run_pipeline(
    initial_states: list,
    observations: dict,
    counts: dict,
    settlements_data: dict,
    n_seeds: int = 5,
    mc_runs: int = 200,
) -> dict:
    """
    Run the full inference + MC simulation + prediction pipeline.

    Returns dict seed_idx -> (H, W, 6) predictions.
    """
    H, W = 40, 40

    # Step 1: Infer hidden parameters
    print("  Inferring parameters...", end=" ", flush=True)
    t0 = time.time()
    inference = ParameterInference(initial_states, observations, counts, settlements_data)
    inferred_params = inference.infer()
    print(f"done in {time.time() - t0:.1f}s")
    print(f"    Inferred: { {k: f'{v:.3f}' for k, v in inferred_params.items()} }")

    # Step 2: Run MC simulation with inferred params for each seed
    print(f"  Running MC simulations ({mc_runs} runs per seed)...", end=" ", flush=True)
    t0 = time.time()
    simulator_predictions: Dict[int, np.ndarray] = {}
    for seed_idx in range(n_seeds):
        state = initial_states[seed_idx]
        grid = np.asarray(state["grid"], dtype=np.int32)
        settlements = state.get("settlements", [])
        sim_pred = NorseSimulator.run_monte_carlo(
            grid, settlements, inferred_params, n_runs=mc_runs,
        )
        simulator_predictions[seed_idx] = sim_pred
    print(f"done in {time.time() - t0:.1f}s")

    # Step 3: Build layered predictions
    print("  Building layered predictions...", flush=True)
    t0 = time.time()
    engine = PredictionEngine(initial_states, W, H, n_seeds)
    predictions = engine.build_predictions(
        counts=counts,
        settlements_data=settlements_data,
        observations=observations,
        simulator_predictions=simulator_predictions,
        inferred_params=inferred_params,
    )
    print(f"  Prediction engine done in {time.time() - t0:.1f}s")

    return predictions


def run_swarm_pipeline(
    initial_states: list,
    observations: dict,
    counts: dict,
    settlements_data: dict,
    n_seeds: int = 5,
    mc_runs: int = 30,
    mc_agents: int = 5,
) -> dict:
    """
    Run the swarm prediction pipeline (v6).

    Returns dict seed_idx -> (H, W, 6) predictions.
    """
    H, W = 40, 40

    # Step 1: Infer hidden parameters
    print("  Inferring parameters...", end=" ", flush=True)
    t0 = time.time()
    inference = ParameterInference(initial_states, observations, counts, settlements_data)
    inferred_params = inference.infer()
    print(f"done in {time.time() - t0:.1f}s")
    print(f"    Inferred: { {k: f'{v:.3f}' for k, v in inferred_params.items()} }")

    # Step 2: Sample posterior
    try:
        posterior_samples = inference.infer_posterior(n_samples=mc_agents)
    except Exception as e:
        print(f"    Posterior sampling failed ({e}), using MAP only")
        posterior_samples = [inferred_params]

    # Step 3: Run swarm
    print(f"  Running swarm ({mc_agents} MC agents × {mc_runs} runs + 5 statistical)...")
    t0 = time.time()
    swarm = SwarmCoordinator(
        initial_states=initial_states,
        W=W, H=H,
        seeds_count=n_seeds,
        inferred_params=inferred_params,
        posterior_samples=posterior_samples,
        mc_runs_per_agent=mc_runs,
        n_mc_agents=mc_agents,
    )
    predictions = swarm.predict_all(
        counts=counts,
        observations=observations,
        settlements_data=settlements_data,
    )
    print(f"  Swarm done in {time.time() - t0:.1f}s")

    return predictions


# ── Analysis ─────────────────────────────────────────────────────────────────

def analyze_weaknesses(
    gt: np.ndarray,
    pred: np.ndarray,
    kl: np.ndarray,
    entropy: np.ndarray,
    initial_grid: np.ndarray,
    top_n: int = 10,
) -> None:
    """Print analysis of where predictions are weakest."""
    H, W = gt.shape[:2]

    # Weighted KL contribution per cell
    total_entropy = entropy.sum()
    if total_entropy < 1e-10:
        print("  No entropy in ground truth (all deterministic).")
        return

    contribution = entropy * kl / total_entropy

    # Top-N worst cells
    flat_idx = np.argsort(contribution.ravel())[::-1][:top_n]
    class_names = ["Empty", "Settlement", "Port", "Ruin", "Forest", "Mountain"]

    print(f"\n  Top {top_n} worst-predicted cells (by scoring contribution):")
    print(f"  {'(y,x)':>8} {'Init':>8} {'Entropy':>8} {'KL':>8} {'Contrib':>8}  GT dist vs Pred dist")
    for idx in flat_idx:
        y, x = divmod(idx, W)
        init_code = int(initial_grid[y, x])
        init_cls = TERRAIN_TO_CLASS.get(init_code, 0)
        init_name = class_names[init_cls]

        gt_str = " ".join(f"{v:.2f}" for v in gt[y, x])
        pr_str = " ".join(f"{v:.2f}" for v in pred[y, x])

        print(f"  ({y:2d},{x:2d}) {init_name:>8} {entropy[y,x]:>8.3f} {kl[y,x]:>8.3f} "
              f"{contribution[y,x]:>8.5f}  [{gt_str}] vs [{pr_str}]")

    # Breakdown by initial terrain type
    print(f"\n  KL breakdown by initial terrain type:")
    print(f"  {'Type':>10} {'Cells':>6} {'AvgEntropy':>10} {'AvgKL':>8} {'TotalContrib':>12}")
    for cls in range(NUM_CLASSES):
        mask = np.zeros((H, W), dtype=bool)
        for code, c in TERRAIN_TO_CLASS.items():
            if c == cls:
                mask |= (initial_grid == code)
        if not mask.any():
            continue
        avg_ent = entropy[mask].mean()
        avg_kl = kl[mask].mean()
        total_contrib = contribution[mask].sum()
        n_cells = mask.sum()
        print(f"  {class_names[cls]:>10} {n_cells:>6} {avg_ent:>10.4f} {avg_kl:>8.4f} {total_contrib:>12.5f}")


# ── Scenario definitions ─────────────────────────────────────────────────────

SCENARIOS = {
    "baseline": {
        "description": "Balanced parameters (default-like)",
        "params": {
            "winter_severity": 0.35,
            "faction_aggression": 0.30,
            "trade_activity": 0.50,
            "forest_growth_rate": 0.15,
            "expansion_rate": 0.20,
            "raid_range": 5.0,
            "food_per_forest": 0.30,
            "port_development_threshold": 0.50,
            "ruin_reclaim_rate": 0.15,
        },
    },
    "high_aggression": {
        "description": "Aggressive factions, lots of raiding",
        "params": {
            "winter_severity": 0.30,
            "faction_aggression": 0.80,
            "trade_activity": 0.20,
            "forest_growth_rate": 0.15,
            "expansion_rate": 0.25,
            "raid_range": 7.0,
            "food_per_forest": 0.30,
            "port_development_threshold": 0.50,
            "ruin_reclaim_rate": 0.10,
        },
    },
    "high_trade": {
        "description": "Peaceful, trade-focused civilization",
        "params": {
            "winter_severity": 0.20,
            "faction_aggression": 0.10,
            "trade_activity": 0.85,
            "forest_growth_rate": 0.10,
            "expansion_rate": 0.15,
            "raid_range": 3.0,
            "food_per_forest": 0.35,
            "port_development_threshold": 0.30,
            "ruin_reclaim_rate": 0.20,
        },
    },
    "harsh_winter": {
        "description": "Brutal winters, high mortality",
        "params": {
            "winter_severity": 0.85,
            "faction_aggression": 0.30,
            "trade_activity": 0.40,
            "forest_growth_rate": 0.20,
            "expansion_rate": 0.10,
            "raid_range": 5.0,
            "food_per_forest": 0.40,
            "port_development_threshold": 0.60,
            "ruin_reclaim_rate": 0.05,
        },
    },
    "rapid_expansion": {
        "description": "Fast-growing, expansionist civilization",
        "params": {
            "winter_severity": 0.20,
            "faction_aggression": 0.20,
            "trade_activity": 0.40,
            "forest_growth_rate": 0.25,
            "expansion_rate": 0.50,
            "raid_range": 4.0,
            "food_per_forest": 0.35,
            "port_development_threshold": 0.40,
            "ruin_reclaim_rate": 0.25,
        },
    },
}

QUERY_CONFIGS = {
    "sparse_30": {"n_queries": 30, "description": "Sparse observations (30 queries)"},
    "standard_50": {"n_queries": 50, "description": "Standard observations (50 queries)"},
}


# ── Main ─────────────────────────────────────────────────────────────────────

def run_scenario(
    name: str,
    scenario: dict,
    query_config: dict,
    map_seed: int = 42,
    gt_runs: int = 500,
    mc_runs: int = 200,
    n_seeds: int = 5,
) -> float:
    """Run a single scenario and return the score."""
    params = scenario["params"]
    n_queries = query_config["n_queries"]

    print(f"\n{'='*70}")
    print(f"Scenario: {name} ({scenario['description']})")
    print(f"Queries: {n_queries} | GT runs: {gt_runs} | MC runs: {mc_runs}")
    print(f"True params: { {k: f'{v:.2f}' for k, v in params.items()} }")
    print(f"{'='*70}")

    # Create test map
    grid, settlements = create_test_map(n_settlements=40, seed=map_seed)
    print(f"  Map: {grid.shape}, {len(settlements)} settlements")

    # Generate ground truth
    gt = generate_ground_truth(grid, settlements, params, n_runs=gt_runs)

    # Simulate observation phase
    print(f"  Simulating observation phase ({n_queries} queries)...", end=" ", flush=True)
    t0 = time.time()
    initial_states, observations, counts, settlements_data = simulate_observations(
        grid, settlements, params,
        n_queries=n_queries, n_seeds=n_seeds, map_seed=map_seed,
    )
    print(f"done in {time.time() - t0:.1f}s")

    # Run BOTH pipelines for comparison
    print("\n  --- Old pipeline (PredictionEngine) ---")
    predictions_old = run_pipeline(
        initial_states, observations, counts, settlements_data,
        n_seeds=n_seeds, mc_runs=mc_runs,
    )

    print("\n  --- New pipeline (Swarm v6) ---")
    predictions_swarm = run_swarm_pipeline(
        initial_states, observations, counts, settlements_data,
        n_seeds=n_seeds, mc_runs=30, mc_agents=5,
    )

    # Score both pipelines
    for label, predictions in [("OLD", predictions_old), ("SWARM", predictions_swarm)]:
        scores = []
        for seed_idx in range(n_seeds):
            pred = predictions[seed_idx]
            score, kl, entropy = score_prediction(gt, pred)
            scores.append(score)
        avg = np.mean(scores)
        per_seed = " ".join(f"{s:.1f}" for s in scores)
        print(f"  {label:>5}: avg={avg:.2f}  seeds=[{per_seed}]")

    # Use swarm for detailed analysis (it's our production pipeline)
    predictions = predictions_swarm
    scores = []
    for seed_idx in range(n_seeds):
        pred = predictions[seed_idx]
        score, kl, entropy = score_prediction(gt, pred)
        scores.append(score)

    avg_score = np.mean(scores)
    print(f"\n  SWARM AVERAGE SCORE: {avg_score:.2f}")

    # Detailed analysis for the first seed
    print(f"\n  Detailed analysis (seed 0):")
    pred_0 = predictions[0]
    score_0, kl_0, entropy_0 = score_prediction(gt, pred_0)

    # Observation coverage
    obs_0 = counts[0][:, :, :NUM_CLASSES].sum(axis=2)
    n_observed = (obs_0 > 0).sum()
    total_cells = grid.shape[0] * grid.shape[1]
    print(f"  Observation coverage: {n_observed}/{total_cells} "
          f"({100 * n_observed / total_cells:.1f}%)")

    # Entropy statistics
    print(f"  Ground truth entropy: mean={entropy_0.mean():.4f}, "
          f"max={entropy_0.max():.4f}, total={entropy_0.sum():.1f}")

    # KL statistics
    print(f"  KL divergence: mean={kl_0.mean():.4f}, "
          f"max={kl_0.max():.4f}, median={np.median(kl_0):.4f}")

    analyze_weaknesses(gt, pred_0, kl_0, entropy_0, grid)

    # Compare inferred vs true params
    inference = ParameterInference(initial_states, observations, counts, settlements_data)
    inferred = inference.infer()
    print(f"\n  Parameter inference accuracy:")
    print(f"  {'Param':>28} {'True':>6} {'Inferred':>8} {'Error':>7}")
    total_error = 0.0
    for param in sorted(params.keys()):
        true_val = params[param]
        inf_val = inferred.get(param, 0.5)
        error = abs(true_val - inf_val)
        total_error += error
        marker = " ***" if error > 0.2 else ""
        print(f"  {param:>28} {true_val:>6.3f} {inf_val:>8.3f} {error:>7.3f}{marker}")
    print(f"  {'TOTAL ABS ERROR':>28} {'':>6} {'':>8} {total_error:>7.3f}")

    return avg_score


def main():
    print("=" * 70)
    print("ASTAR ISLAND PREDICTION PIPELINE VALIDATION")
    print("=" * 70)
    print(f"Scoring: score = max(0, min(100, 100 * exp(-3 * weighted_kl)))")
    print(f"  where weighted_kl = sum(entropy * KL) / sum(entropy)")
    print()

    results: Dict[str, Dict[str, float]] = {}

    for scenario_name, scenario in SCENARIOS.items():
        results[scenario_name] = {}
        for config_name, config in QUERY_CONFIGS.items():
            key = f"{scenario_name}/{config_name}"
            score = run_scenario(
                name=key,
                scenario=scenario,
                query_config=config,
                gt_runs=500,
                mc_runs=200,
                n_seeds=5,
            )
            results[scenario_name][config_name] = score

    # Final summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"\n{'Scenario':<25} {'Sparse (30q)':>12} {'Standard (50q)':>14}")
    print("-" * 55)
    all_scores = []
    for scenario_name in SCENARIOS:
        sparse = results[scenario_name].get("sparse_30", 0)
        standard = results[scenario_name].get("standard_50", 0)
        all_scores.extend([sparse, standard])
        print(f"{scenario_name:<25} {sparse:>12.2f} {standard:>14.2f}")
    print("-" * 55)
    print(f"{'OVERALL AVERAGE':<25} {np.mean(all_scores):>12.2f}")
    print()

    # Interpretation
    avg = np.mean(all_scores)
    if avg >= 80:
        print("Excellent: Predictions are strong across scenarios.")
    elif avg >= 60:
        print("Good: Predictions are reasonable but can be improved.")
    elif avg >= 40:
        print("Fair: Significant room for improvement in inference/prediction.")
    else:
        print("Poor: Pipeline needs major improvements.")


if __name__ == "__main__":
    main()
