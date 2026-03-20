#!/usr/bin/env python3
"""
ABC (Approximate Bayesian Computation) parameter inference for Astar Island.

Instead of heuristic formulas, systematically searches parameter space
by running our simulator with many parameter combinations and comparing
the output distribution against actual observations.

The parameter set whose simulated distribution best matches observations
(measured by KL divergence on observed cells) is selected.

This replaces the heuristic inference in inference.py which had ~2.75 total error.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

import numpy as np

from simulator import NorseSimulator

NUM_CLASSES = 6
TERRAIN_TO_CLASS = {10: 0, 11: 0, 0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5}

# Parameter bounds: (min, max) for each parameter
PARAM_BOUNDS = {
    "winter_severity": (0.05, 0.70),
    "faction_aggression": (0.05, 0.60),
    "trade_activity": (0.10, 0.80),
    "forest_growth_rate": (0.02, 0.30),
    "expansion_rate": (0.05, 0.40),
    "raid_range": (3.0, 8.0),
    "food_per_forest": (0.10, 0.60),
    "port_development_threshold": (0.20, 0.80),
    "ruin_reclaim_rate": (0.03, 0.30),
}

# Parameters that most affect predictions (optimize these first)
PRIMARY_PARAMS = [
    "winter_severity",
    "faction_aggression",
    "expansion_rate",
    "forest_growth_rate",
    "food_per_forest",
    "ruin_reclaim_rate",
]

SECONDARY_PARAMS = [
    "trade_activity",
    "raid_range",
    "port_development_threshold",
]


def _classify_grid(grid: np.ndarray) -> np.ndarray:
    out = np.zeros_like(grid, dtype=np.int32)
    for code, cls in TERRAIN_TO_CLASS.items():
        out[grid == code] = cls
    return out


def _simulate_worker(args):
    """Worker function for parallel simulation."""
    grid, settlements, params, n_runs, seed_offset = args
    probs = NorseSimulator.run_monte_carlo(
        grid, settlements, params, n_runs=n_runs,
        seeds=list(range(seed_offset, seed_offset + n_runs)),
    )
    return probs


def compute_fit(
    sim_probs: np.ndarray,
    obs_counts: np.ndarray,
    obs_total: np.ndarray,
    init_cls: np.ndarray,
    observed_mask: np.ndarray,
) -> float:
    """
    Compute fit between simulated probabilities and observed data.

    Uses KL divergence on observed cells, weighted by observation count
    and entropy (matching the competition scoring).

    Lower = better fit.
    """
    if not observed_mask.any():
        return float("inf")

    # Empirical distribution from observations (with smoothing)
    alpha = 0.5
    emp = (obs_counts[observed_mask] + alpha) / (
        obs_total[observed_mask, np.newaxis] + NUM_CLASSES * alpha
    )

    # Simulated distribution
    sim = sim_probs[observed_mask]
    sim = np.maximum(sim, 1e-8)

    # KL divergence: sum p * log(p/q)
    kl = np.sum(emp * np.log(emp / sim + 1e-12), axis=1)

    # Weight by entropy of empirical distribution (like competition scoring)
    entropy = -np.sum(emp * np.log(emp + 1e-12), axis=1)
    entropy = np.maximum(entropy, 0.01)  # floor for near-static cells

    # Weighted mean KL
    weighted_kl = np.sum(entropy * kl) / np.sum(entropy)
    return float(weighted_kl)


def generate_param_grid(
    n_samples: int = 50,
    seed: int = 42,
    heuristic_center: Optional[Dict[str, float]] = None,
) -> List[Dict[str, float]]:
    """
    Generate parameter combinations using Latin Hypercube Sampling.

    If heuristic_center is provided, concentrates 40% of samples near it
    (within ±30% of heuristic values) and spreads 60% across full range.
    """
    rng = np.random.default_rng(seed)
    param_names = list(PARAM_BOUNDS.keys())
    n_params = len(param_names)

    samples = []

    # Phase 1: Full-range LHS (60% of budget)
    n_full = int(n_samples * 0.6)
    intervals = np.linspace(0, 1, n_full + 1)
    for i in range(n_full):
        sample = {}
        for j, name in enumerate(param_names):
            lo, hi = PARAM_BOUNDS[name]
            # Stratified random within interval
            u = intervals[i] + rng.random() * (intervals[i + 1] - intervals[i])
            # Shuffle assignment across dimensions
            u = rng.random()  # Simple random for now, LHS later
            sample[name] = lo + u * (hi - lo)
        samples.append(sample)

    # Phase 2: Concentrated near heuristic center (40% of budget)
    if heuristic_center:
        n_local = n_samples - n_full
        for _ in range(n_local):
            sample = {}
            for name in param_names:
                lo, hi = PARAM_BOUNDS[name]
                center = heuristic_center.get(name, (lo + hi) / 2)
                center = np.clip(center, lo, hi)
                spread = (hi - lo) * 0.3  # ±30% of range
                val = center + rng.normal(0, spread * 0.5)
                sample[name] = float(np.clip(val, lo, hi))
            samples.append(sample)
    else:
        # Fill with more random samples
        for _ in range(n_samples - n_full):
            sample = {}
            for name in param_names:
                lo, hi = PARAM_BOUNDS[name]
                sample[name] = lo + rng.random() * (hi - lo)
            samples.append(sample)

    return samples


def abc_infer(
    initial_states: List[dict],
    counts: Dict[int, np.ndarray],
    n_candidates: int = 60,
    mc_per_candidate: int = 15,
    heuristic_params: Optional[Dict[str, float]] = None,
    n_refine: int = 30,
    mc_refine: int = 20,
    max_workers: int = 4,
    verbose: bool = True,
) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
    """
    ABC parameter inference.

    Phase 1: Coarse search — evaluate n_candidates parameter sets
    Phase 2: Refinement — sample near the best, evaluate with more MC runs

    Args:
        initial_states: List of {grid, settlements} per seed
        counts: Dict seed_idx -> (H, W, 6) observation counts
        n_candidates: Number of initial parameter candidates
        mc_per_candidate: MC runs per candidate in phase 1
        heuristic_params: Optional center from heuristic inference (for warm start)
        n_refine: Number of refinement candidates
        mc_refine: MC runs per candidate in refinement
        max_workers: Parallel workers

    Returns:
        (best_params, top_5_params)
    """
    n_seeds = len(initial_states)

    # Precompute observation masks per seed
    seed_data = []
    for seed_idx in range(n_seeds):
        if seed_idx not in counts:
            continue
        state = initial_states[seed_idx]
        grid = np.asarray(state["grid"], dtype=np.int32)
        init_cls = _classify_grid(grid)
        obs_counts = counts[seed_idx][:, :, :NUM_CLASSES].astype(np.float64)
        obs_total = obs_counts.sum(axis=2)
        observed = obs_total > 0

        # Only evaluate on cells with enough observations (n >= 2)
        good_obs = obs_total >= 2
        if good_obs.sum() < 20:
            good_obs = observed  # fallback if few multi-observed cells

        seed_data.append({
            "grid": grid,
            "settlements": state.get("settlements", []),
            "init_cls": init_cls,
            "obs_counts": obs_counts,
            "obs_total": obs_total,
            "observed": good_obs,
        })

    if not seed_data:
        if verbose:
            print("  ABC: No observation data, using heuristic params")
        return heuristic_params or {}, []

    # ── Phase 1: Coarse search ──────────────────────────────────────────
    if verbose:
        print(f"  ABC Phase 1: {n_candidates} candidates × {mc_per_candidate} MC "
              f"across {len(seed_data)} seeds...")

    candidates = generate_param_grid(n_candidates, seed=42,
                                      heuristic_center=heuristic_params)

    results = []
    for ci, params in enumerate(candidates):
        total_fit = 0.0
        n_evaluated = 0

        for sd in seed_data:
            # Run MC simulation with these params
            probs = NorseSimulator.run_monte_carlo(
                sd["grid"], sd["settlements"], params,
                n_runs=mc_per_candidate,
                seeds=list(range(ci * 100, ci * 100 + mc_per_candidate)),
            )

            fit = compute_fit(
                probs, sd["obs_counts"], sd["obs_total"],
                sd["init_cls"], sd["observed"],
            )
            total_fit += fit
            n_evaluated += 1

        avg_fit = total_fit / max(n_evaluated, 1)
        results.append((avg_fit, params))

        if verbose and (ci + 1) % 20 == 0:
            best_so_far = min(r[0] for r in results)
            print(f"    {ci+1}/{n_candidates} evaluated, best fit: {best_so_far:.4f}")

    results.sort(key=lambda x: x[0])
    best_fit, best_params = results[0]

    if verbose:
        print(f"  Phase 1 best fit: {best_fit:.4f}")
        print(f"  Best params: { {k: f'{v:.3f}' for k, v in best_params.items()} }")

    # ── Phase 2: Refinement ─────────────────────────────────────────────
    if verbose:
        print(f"\n  ABC Phase 2: {n_refine} refinements × {mc_refine} MC...")

    # Sample near the top-5 candidates
    top_5 = [r[1] for r in results[:5]]
    rng = np.random.default_rng(123)

    refine_candidates = []
    for _ in range(n_refine):
        # Pick a random top-5 candidate as center
        center = top_5[rng.integers(len(top_5))]
        sample = {}
        for name in PARAM_BOUNDS:
            lo, hi = PARAM_BOUNDS[name]
            spread = (hi - lo) * 0.15  # ±15% of range
            val = center[name] + rng.normal(0, spread)
            sample[name] = float(np.clip(val, lo, hi))
        refine_candidates.append(sample)

    # Also include top-5 from phase 1
    refine_candidates.extend(top_5)

    refine_results = []
    for ci, params in enumerate(refine_candidates):
        total_fit = 0.0
        n_evaluated = 0

        for sd in seed_data:
            probs = NorseSimulator.run_monte_carlo(
                sd["grid"], sd["settlements"], params,
                n_runs=mc_refine,
                seeds=list(range(1000 + ci * 100, 1000 + ci * 100 + mc_refine)),
            )

            fit = compute_fit(
                probs, sd["obs_counts"], sd["obs_total"],
                sd["init_cls"], sd["observed"],
            )
            total_fit += fit
            n_evaluated += 1

        avg_fit = total_fit / max(n_evaluated, 1)
        refine_results.append((avg_fit, params))

    refine_results.sort(key=lambda x: x[0])
    final_fit, final_params = refine_results[0]

    if verbose:
        improvement = best_fit - final_fit
        print(f"  Phase 2 best fit: {final_fit:.4f} "
              f"(improvement: {improvement:+.4f})")
        print(f"  Final params: { {k: f'{v:.3f}' for k, v in final_params.items()} }")

    top_5_final = [r[1] for r in refine_results[:5]]
    return final_params, top_5_final


def validate_abc_on_gt(
    initial_states: List[dict],
    gt: Dict[int, np.ndarray],
    true_params: Dict[str, float],
    counts: Dict[int, np.ndarray],
):
    """
    Validate ABC inference by comparing inferred params against known true params.
    Uses only the observation data that would be available in a real round.
    """
    from inference import ParameterInference

    n_seeds = len(initial_states)
    observations = {s: [[None] * 40 for _ in range(40)] for s in range(n_seeds)}

    print("\n  Heuristic inference (current):")
    inferrer = ParameterInference(initial_states, observations, counts)
    heuristic = inferrer.infer()

    heuristic_error = sum(
        abs(heuristic.get(k, 0.5) - true_params[k])
        for k in true_params if k != "raid_range"
    )
    raid_error = abs(heuristic.get("raid_range", 5.0) - true_params["raid_range"])
    print(f"    Total error (excl raid_range): {heuristic_error:.3f}")
    print(f"    Raid range error: {raid_error:.3f}")

    print("\n  ABC inference:")
    abc_params, top5 = abc_infer(
        initial_states, counts,
        n_candidates=60, mc_per_candidate=15,
        heuristic_params=heuristic,
        n_refine=30, mc_refine=20,
    )

    abc_error = sum(
        abs(abc_params.get(k, 0.5) - true_params[k])
        for k in true_params if k != "raid_range"
    )
    raid_error_abc = abs(abc_params.get("raid_range", 5.0) - true_params["raid_range"])
    print(f"    Total error (excl raid_range): {abc_error:.3f}")
    print(f"    Raid range error: {raid_error_abc:.3f}")

    print("\n  Per-parameter comparison:")
    print(f"    {'Parameter':<30} {'True':>8} {'Heur':>8} {'ABC':>8} {'HErr':>8} {'AErr':>8}")
    for k in sorted(true_params.keys()):
        true_v = true_params[k]
        heur_v = heuristic.get(k, 0.5)
        abc_v = abc_params.get(k, 0.5)
        print(f"    {k:<30} {true_v:>8.3f} {heur_v:>8.3f} {abc_v:>8.3f} "
              f"{abs(heur_v-true_v):>8.3f} {abs(abc_v-true_v):>8.3f}")


if __name__ == "__main__":
    # Test with validation scenario
    from validate import create_test_map, SCENARIOS, QUERY_CONFIGS
    from query_optimizer import QueryOptimizer

    scenario = SCENARIOS["baseline"]
    true_params = scenario["params"]
    print(f"True params: {true_params}")

    grid, settlements = create_test_map(n_settlements=40, seed=42)
    H, W = grid.shape
    n_seeds = 5

    initial_states = [
        {"grid": grid.tolist(), "settlements": settlements}
        for _ in range(n_seeds)
    ]

    # Simulate observations
    optimizer = QueryOptimizer(W, H, n_seeds, 50, initial_states)
    plan = optimizer.plan_queries()

    counts = {s: np.zeros((H, W, NUM_CLASSES), dtype=np.int32) for s in range(n_seeds)}

    # Run actual simulations to get observations
    from simulator import NorseSimulator
    sim = NorseSimulator(grid, settlements, true_params)
    for seed_idx, x, y, w, h in plan:
        result_grid = sim.run(seed=seed_idx * 1000 + len(plan))
        result_cls = _classify_grid(result_grid)
        for dy in range(h):
            for dx in range(w):
                yy, xx = y + dy, x + dx
                if yy < H and xx < W:
                    counts[seed_idx][yy, xx, result_cls[yy, xx]] += 1

    validate_abc_on_gt(initial_states, {}, true_params, counts)
