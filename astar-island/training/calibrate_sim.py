"""
Massive grid search calibration for NorseSimulator parameters.

For each cached round (r1..r8), runs the simulator with many parameter
combinations (50 MC runs each), computes KL divergence against ground truth,
and finds the optimal parameter set.

Grid: 9×8×5×6×5×5 = 54,000 combos × 50 MC runs × 7 rounds = ~18.9M sim runs.
Uses multiprocessing.Pool for parallelism.

Usage:
    python calibrate_sim.py [--workers 8] [--mc-runs 50] [--output results.json]
"""

from __future__ import annotations

import argparse
import json
import itertools
import os
import sys
import time
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Add parent dir to path so we can import simulator
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from simulator import NorseSimulator, TERRAIN_TO_CLASS, NUM_CLASSES


# ── Parameter grid ──────────────────────────────────────────────────────────

PARAM_GRID = {
    "winter_severity": np.linspace(0.1, 0.9, 9).tolist(),
    "faction_aggression": np.linspace(0.05, 0.6, 8).tolist(),
    "trade_activity": np.linspace(0.1, 0.9, 5).tolist(),
    "expansion_rate": np.linspace(0.05, 0.5, 6).tolist(),
    "forest_growth_rate": np.linspace(0.01, 0.2, 5).tolist(),
    "food_per_forest": np.linspace(0.1, 0.5, 5).tolist(),
}

# Fixed params not in the grid
FIXED_PARAMS = {
    "raid_range": 5.0,
    "port_development_threshold": 0.5,
    "ruin_reclaim_rate": 0.20,
}

PARAM_NAMES = list(PARAM_GRID.keys())
PARAM_VALUES = [PARAM_GRID[k] for k in PARAM_NAMES]
TOTAL_COMBOS = 1
for v in PARAM_VALUES:
    TOTAL_COMBOS *= len(v)

print(f"Parameter grid: {' x '.join(str(len(v)) for v in PARAM_VALUES)} = {TOTAL_COMBOS:,} combos")


# ── Ground truth loading ────────────────────────────────────────────────────

def load_round_data(cache_dir: Path, round_num: int) -> Optional[Dict[str, Any]]:
    """Load init and GT data for a round. Returns None if files missing."""
    init_file = cache_dir / f"r{round_num}_init.json"
    if not init_file.exists():
        return None

    with open(init_file) as f:
        init_data = json.load(f)

    # Load all GT seeds
    gt_seeds = {}
    for seed_idx in range(5):
        gt_file = cache_dir / f"r{round_num}_gt_s{seed_idx}.json"
        if gt_file.exists():
            with open(gt_file) as f:
                gt_data = json.load(f)
            gt_seeds[seed_idx] = np.array(gt_data["ground_truth"], dtype=np.float64)

    if not gt_seeds:
        return None

    return {
        "round_num": round_num,
        "init_data": init_data,
        "gt_seeds": gt_seeds,
    }


# ── KL divergence scoring ──────────────────────────────────────────────────

def kl_divergence(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-6) -> float:
    """
    Compute mean KL divergence D_KL(gt || pred) per cell.
    Both inputs are (H, W, 6) probability arrays.
    """
    pred_safe = np.clip(pred, eps, 1.0)
    gt_safe = np.clip(gt, eps, 1.0)

    # Normalize
    pred_safe = pred_safe / pred_safe.sum(axis=2, keepdims=True)
    gt_safe = gt_safe / gt_safe.sum(axis=2, keepdims=True)

    # KL per cell
    kl = np.sum(gt_safe * np.log(gt_safe / pred_safe), axis=2)
    return float(np.mean(kl))


def score_from_kl(kl: float) -> float:
    """Convert KL divergence to a 0-100 score (higher=better)."""
    # Score = 100 * exp(-kl * scale_factor)
    # This mimics the competition scoring where lower KL = higher score
    return 100.0 * np.exp(-kl * 5.0)


# ── Worker function ─────────────────────────────────────────────────────────

# Global state set in worker init
_worker_round_data = None
_worker_mc_runs = None


def _init_worker(round_data_serialized: bytes, mc_runs: int):
    """Initialize worker process with round data."""
    global _worker_round_data, _worker_mc_runs
    _worker_round_data = json.loads(round_data_serialized)
    # Reconstruct numpy arrays
    for seed_idx in list(_worker_round_data["gt_seeds"].keys()):
        _worker_round_data["gt_seeds"][seed_idx] = np.array(
            _worker_round_data["gt_seeds"][seed_idx]
        )
    _worker_mc_runs = mc_runs


def _evaluate_combo(combo_idx_and_values: Tuple[int, Tuple[float, ...]]) -> Tuple[int, float, Dict[str, float]]:
    """Evaluate one parameter combination against all GT seeds."""
    combo_idx, values = combo_idx_and_values

    params = dict(zip(PARAM_NAMES, values))
    params.update(FIXED_PARAMS)

    round_data = _worker_round_data
    init_data = round_data["init_data"]
    gt_seeds = round_data["gt_seeds"]
    mc_runs = _worker_mc_runs

    total_kl = 0.0
    n_seeds = 0

    for seed_idx_str, gt_probs in gt_seeds.items():
        seed_idx = int(seed_idx_str)

        # Get initial state for this seed
        initial_state = init_data["initial_states"][seed_idx]
        grid = np.array(initial_state["grid"], dtype=np.int32)
        settlements = initial_state["settlements"]

        # Run MC simulations
        try:
            pred_probs = NorseSimulator.run_monte_carlo(
                initial_grid=grid,
                initial_settlements=settlements,
                params=params,
                n_runs=mc_runs,
            )
        except Exception:
            return (combo_idx, float("inf"), params)

        kl = kl_divergence(pred_probs, gt_probs)
        total_kl += kl
        n_seeds += 1

    avg_kl = total_kl / max(n_seeds, 1)
    return (combo_idx, avg_kl, params)


# ── Main calibration ────────────────────────────────────────────────────────

def generate_combos() -> List[Tuple[int, Tuple[float, ...]]]:
    """Generate all parameter combinations with indices."""
    combos = []
    for idx, values in enumerate(itertools.product(*PARAM_VALUES)):
        combos.append((idx, values))
    return combos


def calibrate_round(
    round_data: Dict[str, Any],
    mc_runs: int = 50,
    workers: int = 8,
    batch_size: int = 500,
) -> Dict[str, Any]:
    """Run calibration for one round."""
    round_num = round_data["round_num"]
    print(f"\n{'='*70}")
    print(f"CALIBRATING ROUND {round_num}")
    print(f"GT seeds available: {list(round_data['gt_seeds'].keys())}")
    print(f"MC runs per combo: {mc_runs}")
    print(f"Workers: {workers}")
    print(f"Total combos: {TOTAL_COMBOS:,}")
    print(f"{'='*70}")

    # Serialize round data for workers (convert numpy to lists)
    rd_serializable = {
        "init_data": round_data["init_data"],
        "gt_seeds": {
            str(k): v.tolist() for k, v in round_data["gt_seeds"].items()
        },
    }
    rd_bytes = json.dumps(rd_serializable).encode()

    combos = generate_combos()

    best_kl = float("inf")
    best_params = None
    best_idx = -1

    # Top-10 tracker
    top_results = []

    t0 = time.time()
    processed = 0

    with Pool(
        processes=workers,
        initializer=_init_worker,
        initargs=(rd_bytes, mc_runs),
    ) as pool:
        # Process in batches for progress reporting
        for batch_start in range(0, len(combos), batch_size):
            batch = combos[batch_start : batch_start + batch_size]
            results = pool.map(_evaluate_combo, batch)

            for combo_idx, kl, params in results:
                processed += 1

                # Track top results
                top_results.append((kl, params.copy()))
                if len(top_results) > 20:
                    top_results.sort(key=lambda x: x[0])
                    top_results = top_results[:20]

                if kl < best_kl:
                    best_kl = kl
                    best_params = params.copy()
                    best_idx = combo_idx
                    score = score_from_kl(kl)
                    elapsed = time.time() - t0
                    print(
                        f"  [R{round_num}] New best @ combo {combo_idx:,}/{TOTAL_COMBOS:,}: "
                        f"KL={kl:.6f} score={score:.2f} "
                        f"| ws={params['winter_severity']:.2f} fa={params['faction_aggression']:.2f} "
                        f"ta={params['trade_activity']:.2f} er={params['expansion_rate']:.2f} "
                        f"fg={params['forest_growth_rate']:.3f} fpf={params['food_per_forest']:.2f} "
                        f"| {elapsed:.0f}s"
                    )

            elapsed = time.time() - t0
            pct = processed / TOTAL_COMBOS * 100
            rate = processed / max(elapsed, 1)
            eta = (TOTAL_COMBOS - processed) / max(rate, 1)
            print(
                f"  [R{round_num}] Progress: {processed:,}/{TOTAL_COMBOS:,} ({pct:.1f}%) "
                f"| {rate:.0f} combos/s | ETA {eta:.0f}s"
            )

    total_time = time.time() - t0

    top_results.sort(key=lambda x: x[0])

    result = {
        "round_num": round_num,
        "best_kl": best_kl,
        "best_score": score_from_kl(best_kl),
        "best_params": {k: round(v, 6) for k, v in best_params.items()} if best_params else None,
        "best_combo_idx": best_idx,
        "total_combos": TOTAL_COMBOS,
        "mc_runs": mc_runs,
        "total_time_s": round(total_time, 1),
        "combos_per_sec": round(TOTAL_COMBOS / max(total_time, 1), 1),
        "gt_seeds_used": list(round_data["gt_seeds"].keys()),
        "top_10": [
            {
                "kl": round(kl, 6),
                "score": round(score_from_kl(kl), 2),
                "params": {k: round(v, 6) for k, v in p.items()},
            }
            for kl, p in top_results[:10]
        ],
    }

    print(f"\n  [R{round_num}] DONE in {total_time:.0f}s")
    print(f"  [R{round_num}] Best KL={best_kl:.6f}, Score={score_from_kl(best_kl):.2f}")
    print(f"  [R{round_num}] Best params: {best_params}")

    return result


def main():
    parser = argparse.ArgumentParser(description="Calibrate NorseSimulator parameters via grid search")
    parser.add_argument("--workers", type=int, default=0, help="Number of worker processes (0=auto)")
    parser.add_argument("--mc-runs", type=int, default=50, help="MC simulations per combo")
    parser.add_argument("--output", type=str, default="calibration_results.json", help="Output file")
    parser.add_argument("--rounds", type=str, default="1,2,4,5,6,7,8", help="Comma-separated round numbers")
    parser.add_argument("--batch-size", type=int, default=500, help="Batch size for progress reporting")
    parser.add_argument("--quick", action="store_true", help="Quick test: only 1 GT seed, 10 MC runs")
    args = parser.parse_args()

    workers = args.workers if args.workers > 0 else cpu_count()
    mc_runs = 10 if args.quick else args.mc_runs
    round_nums = [int(r.strip()) for r in args.rounds.split(",")]

    # Look for cache dir — try multiple locations
    for candidate in [
        Path(__file__).resolve().parent / "cache",
        Path(__file__).resolve().parent.parent / "cache",
        Path.home() / "astar-island" / "cache",
    ]:
        if candidate.exists() and any(candidate.glob("r*_gt_*.json")):
            cache_dir = candidate
            break
    else:
        cache_dir = Path(__file__).resolve().parent / "cache"
    output_path = Path(__file__).resolve().parent / args.output

    print(f"Calibration settings:")
    print(f"  Workers: {workers}")
    print(f"  MC runs per combo: {mc_runs}")
    print(f"  Rounds: {round_nums}")
    print(f"  Total combos per round: {TOTAL_COMBOS:,}")
    print(f"  Cache dir: {cache_dir}")
    print(f"  Output: {output_path}")

    all_results = {}
    grand_t0 = time.time()

    for rn in round_nums:
        rd = load_round_data(cache_dir, rn)
        if rd is None:
            print(f"\nSkipping round {rn}: no data in cache")
            continue

        # In quick mode, only use seed 0
        if args.quick:
            first_key = list(rd["gt_seeds"].keys())[0]
            rd["gt_seeds"] = {first_key: rd["gt_seeds"][first_key]}

        result = calibrate_round(rd, mc_runs=mc_runs, workers=workers, batch_size=args.batch_size)
        all_results[f"round_{rn}"] = result

        # Save incrementally
        with open(output_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"  Saved intermediate results to {output_path}")

    grand_total = time.time() - grand_t0
    print(f"\n{'='*70}")
    print(f"ALL ROUNDS COMPLETE in {grand_total:.0f}s ({grand_total/3600:.1f}h)")
    print(f"{'='*70}")

    # Summary
    print("\nSUMMARY:")
    print(f"{'Round':<8} {'Best KL':<12} {'Score':<8} {'Key Params'}")
    print("-" * 70)
    for rn in round_nums:
        key = f"round_{rn}"
        if key in all_results:
            r = all_results[key]
            p = r["best_params"]
            if p:
                print(
                    f"R{rn:<6} {r['best_kl']:<12.6f} {r['best_score']:<8.2f} "
                    f"ws={p['winter_severity']:.2f} fa={p['faction_aggression']:.2f} "
                    f"er={p['expansion_rate']:.2f} fg={p['forest_growth_rate']:.3f}"
                )

    # Also save a simplified best-params file for easy import
    best_params_file = Path(__file__).resolve().parent.parent / "cache" / "calibrated_params_v2.json"
    best_params = {}
    for rn in round_nums:
        key = f"round_{rn}"
        if key in all_results and all_results[key]["best_params"]:
            best_params[key] = {
                "best_score": all_results[key]["best_score"],
                "best_kl": all_results[key]["best_kl"],
                "best_params": all_results[key]["best_params"],
                "gt_seeds_used": all_results[key]["gt_seeds_used"],
            }
    with open(best_params_file, "w") as f:
        json.dump(best_params, f, indent=2)
    print(f"\nBest params saved to {best_params_file}")

    print(f"\nFull results saved to {output_path}")


if __name__ == "__main__":
    main()
