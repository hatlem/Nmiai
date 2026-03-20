#!/usr/bin/env python3
"""
Run swarm predictions locally using saved round data.
Saves predictions as JSON files for browser-based submission.

Usage:
    python3 run_swarm_local.py
"""

import json
import numpy as np
from pathlib import Path

from inference import ParameterInference
from priors import NUM_CLASSES, PROB_FLOOR, STATIC_FLOOR
from swarm import SwarmCoordinator

ROUND_DATA_FILE = "round_data.json"


def main():
    detail = json.loads(Path(ROUND_DATA_FILE).read_text())
    W = detail["map_width"]
    H = detail["map_height"]
    seeds_count = detail["seeds_count"]
    initial_states = detail["initial_states"]
    round_id = detail["id"]

    print(f"Round {detail.get('round_number', '?')}: {W}x{H}, {seeds_count} seeds")
    print(f"Round ID: {round_id}")

    for i, st in enumerate(initial_states):
        n_sett = len(st.get("settlements", []))
        n_port = sum(1 for s in st.get("settlements", []) if s.get("has_port"))
        print(f"  Seed {i}: {n_sett} settlements ({n_port} ports)")

    seeds = list(range(seeds_count))
    observations = {s: [[None] * W for _ in range(H)] for s in seeds}
    counts = {s: np.zeros((H, W, NUM_CLASSES), dtype=np.int32) for s in seeds}
    settlements_data = {s: [] for s in seeds}

    print("\nInferring parameters (lightweight, for heuristic agent)...")
    inferrer = ParameterInference(initial_states, observations, counts,
                                   settlements_data=settlements_data)
    inferred_params = inferrer.infer()
    print(f"  Estimates: { {k: f'{v:.2f}' for k, v in inferred_params.items()} }")

    print("\nRunning empirical predictions (no MC)...")
    swarm = SwarmCoordinator(
        initial_states=initial_states,
        W=W, H=H,
        seeds_count=seeds_count,
        inferred_params=inferred_params,
        posterior_samples=[inferred_params],
        mc_runs_per_agent=0,
        n_mc_agents=0,
    )

    predictions = swarm.predict_all(
        counts=counts,
        observations=observations,
        settlements_data=settlements_data,
    )

    print("\nSaving predictions...")
    for seed_idx in range(seeds_count):
        pred = predictions[seed_idx]
        pred = np.maximum(pred, STATIC_FLOOR)
        pred /= pred.sum(axis=-1, keepdims=True)

        np.save(f"predictions_seed_{seed_idx}.npy", pred)

        payload = {
            "round_id": round_id,
            "seed_index": seed_idx,
            "prediction": pred.tolist(),
        }
        outpath = Path(f"prediction_{seed_idx}.json")
        outpath.write_text(json.dumps(payload, separators=(",", ":")))
        print(f"  Seed {seed_idx}: saved ({pred.shape}, "
              f"sum={pred.sum(axis=-1).mean():.4f})")

    print("\nDone! Submit via browser.")


if __name__ == "__main__":
    main()
