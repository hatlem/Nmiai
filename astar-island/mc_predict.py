#!/usr/bin/env python3
"""
Monte Carlo prediction script for Astar Island.

Takes a round_id and seed_index, fetches initial state from API (or cache),
runs MC simulation, outputs (H, W, 6) probability tensor as JSON to stdout.

Usage:
    python3 mc_predict.py --round-id <round_id> --seed <seed_index> [--runs 200]
    python3 mc_predict.py --cache-file <path_to_init.json> --seed <seed_index> [--runs 200]

Environment:
    AINM_TOKEN — API bearer token (required for API mode)
"""

from __future__ import annotations

import argparse
import json
import sys
import os
import urllib.request
import urllib.error

import numpy as np

# Import simulator from same directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from simulator import NorseSimulator, DEFAULT_PARAMS


def fetch_round_detail(round_id: str, token: str) -> dict:
    """Fetch round detail from API."""
    url = f"https://api.ainm.no/astar-island/rounds/{round_id}"
    req = urllib.request.Request(url, headers={
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    })
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.loads(resp.read().decode())


def load_from_cache(cache_file: str) -> dict:
    """Load round detail from cached JSON file."""
    with open(cache_file) as f:
        return json.load(f)


def run_mc_for_seed(detail: dict, seed_index: int, n_runs: int = 200) -> list:
    """
    Run Monte Carlo simulation for a single seed.

    Returns (H, W, 6) probability tensor as nested Python lists.
    """
    H = detail["map_height"]
    W = detail["map_width"]
    state = detail["initial_states"][seed_index]
    grid = state["grid"]
    settlements = state.get("settlements", [])

    # Load calibrated params if available
    params = dict(DEFAULT_PARAMS)
    cal_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache", "calibrated_params.json")
    if os.path.exists(cal_path):
        try:
            cal = json.load(open(cal_path))
            # Average calibrated params across all rounds for a general estimate
            all_params = [v["best_params"] for v in cal.values() if "best_params" in v]
            if all_params:
                avg_params = {}
                for key in all_params[0]:
                    avg_params[key] = sum(p[key] for p in all_params) / len(all_params)
                params.update(avg_params)
        except Exception:
            pass

    # Use diverse seeds for MC runs
    mc_seeds = list(range(seed_index * 10000, seed_index * 10000 + n_runs))

    probs = NorseSimulator.run_monte_carlo(
        initial_grid=grid,
        initial_settlements=settlements,
        params=params,
        n_runs=n_runs,
        seeds=mc_seeds,
    )

    # probs is (H, W, 6) numpy array — convert to nested lists
    return probs.tolist()


def main():
    parser = argparse.ArgumentParser(description="MC prediction for Astar Island")
    parser.add_argument("--round-id", type=str, help="Round ID to fetch from API")
    parser.add_argument("--cache-file", type=str, help="Path to cached init JSON (alternative to --round-id)")
    parser.add_argument("--seed", type=int, required=True, help="Seed index (0-4)")
    parser.add_argument("--runs", type=int, default=200, help="Number of MC runs (default: 200)")
    args = parser.parse_args()

    # Load round detail
    if args.cache_file:
        detail = load_from_cache(args.cache_file)
    elif args.round_id:
        token = os.environ.get("AINM_TOKEN") or os.environ.get("TOKEN")
        if not token:
            print(json.dumps({"error": "AINM_TOKEN env var required"}), file=sys.stderr)
            sys.exit(1)
        detail = fetch_round_detail(args.round_id, token)
    else:
        print(json.dumps({"error": "Must provide --round-id or --cache-file"}), file=sys.stderr)
        sys.exit(1)

    # Validate seed index
    seeds_count = detail.get("seeds_count", 5)
    if args.seed < 0 or args.seed >= seeds_count:
        print(json.dumps({"error": f"seed must be 0-{seeds_count-1}"}), file=sys.stderr)
        sys.exit(1)

    # Run MC simulation
    probs = run_mc_for_seed(detail, args.seed, args.runs)

    # Output as JSON to stdout
    json.dump(probs, sys.stdout)


if __name__ == "__main__":
    main()
