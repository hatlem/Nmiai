#!/usr/bin/env python3
"""
Astar Island — Viking Civilisation Prediction Agent v6 (Swarm)

Architecture:
- QueryOptimizer: Settlement-focused repeated-query strategy (3-5x per viewport)
- ParameterInference: Bayesian inference of hidden simulator parameters
- SwarmCoordinator: Multiple independent prediction agents ensembled via
  geometric mean (KL-optimal). Agents include:
  - N MonteCarloAgents with diverse posterior parameter samples
  - StatisticalAgent (KT estimator + contextual transitions)
  - TransitionAgent (global transition matrix)
  - SpatialAgent (belief propagation)
  - HeuristicAgent (domain knowledge + inferred params)
  - SettlementTrajectoryAgent (settlement metadata trajectories)

Usage:
    python agent.py --token <JWT_TOKEN>
    python agent.py --resume          # Resume from observations.json
    python agent.py --submit-only     # Submit saved predictions
    python agent.py --no-query        # Prior-only predictions
    python agent.py --mc-agents 8     # MC swarm agents (default 8)
    python agent.py --mc-runs 30      # MC runs per agent (default 30)
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import requests

from inference import ParameterInference
from query_optimizer import QueryOptimizer
from swarm import SwarmCoordinator

BASE_URL = "https://api.ainm.no"
NUM_CLASSES = 6
PROB_FLOOR = 0.01
TERRAIN_TO_CLASS = {10: 0, 11: 0, 0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5}


# ── API ──────────────────────────────────────────────────────────────────────

def api_request(session: requests.Session, method: str, path: str,
                payload: dict | None = None) -> Any:
    url = f"{BASE_URL}{path}"
    for attempt in range(3):
        try:
            resp = session.request(method, url, json=payload, timeout=30)
        except requests.RequestException as e:
            if attempt < 2:
                time.sleep(2)
                continue
            raise RuntimeError(f"Network error: {e}")

        if resp.status_code >= 500:
            if attempt < 2:
                time.sleep(2)
                continue
            raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:200]}")

        if resp.status_code == 429:
            if attempt < 2:
                time.sleep(1.5 * (attempt + 1))
                continue
            raise RuntimeError(f"Rate limited: {resp.text[:200]}")

        if resp.status_code >= 400:
            raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:200]}")

        return resp.json()

    raise RuntimeError(f"Failed after 3 attempts: {method} {path}")


def create_session(token: str) -> requests.Session:
    session = requests.Session()
    session.headers.update({
        "Authorization": f"Bearer {token}",
        "Accept": "application/json",
        "Content-Type": "application/json",
    })
    return session


# ── Crash recovery ───────────────────────────────────────────────────────────

def save_state(path, round_id, queries_used, observations, counts,
               settlements, query_log):
    seeds = sorted(observations.keys())
    Path(path).write_text(json.dumps({
        "round_id": round_id,
        "queries_used": queries_used,
        "latest": {str(s): observations[s] for s in seeds},
        "counts": {str(s): counts[s].tolist() for s in seeds},
        "settlements": {str(s): settlements[s] for s in seeds},
        "queries": query_log,
    }, separators=(',', ':')), encoding="utf-8")


def load_state(path):
    data = json.loads(Path(path).read_text())
    observations = {int(k): v for k, v in data["latest"].items()}
    counts = {int(k): np.array(v, dtype=np.int32)
              for k, v in data["counts"].items()}
    settlements_data = {int(k): v for k, v in data["settlements"].items()}
    queries_used = data["queries_used"]
    query_log = data.get("queries", [])
    return observations, counts, settlements_data, queries_used, query_log


# ── Observation storage ─────────────────────────────────────────────────────

def store_observation(result, seed_idx, x, y, H, W,
                      observations, counts, settlements_data, queries_used):
    """Store a simulation result into our data structures.

    CRITICAL: Maps raw terrain codes to 6 prediction classes before counting.
    Ocean (10), Plains (11), Empty (0) → class 0
    Settlement (1) → class 1, Port (2) → class 2, etc.
    """
    grid = result["grid"]
    new_cells = 0
    for dy, row in enumerate(grid):
        for dx, val in enumerate(row):
            yy, xx = y + dy, x + dx
            if yy < H and xx < W:
                if observations[seed_idx][yy][xx] is None:
                    new_cells += 1
                observations[seed_idx][yy][xx] = val
                cls = TERRAIN_TO_CLASS.get(val, 0)
                counts[seed_idx][yy, xx, cls] += 1

    sett_snap = result.get("settlements", [])
    settlements_data[seed_idx].append({
        "query_index": queries_used,
        "settlements": sett_snap,
    })
    return new_cells


# ── Monte Carlo simulation ──────────────────────────────────────────────────

# ── Main ─────────────────────────────────────────────────────────────────────

def run(token: str, dry_run=False, submit_only=False, no_query=False,
        resume=False, mc_runs=30, no_mc=False, mc_agents=8):
    session = create_session(token)

    # Load round
    print("Loading active round...")
    rounds = api_request(session, "GET", "/astar-island/rounds")
    if isinstance(rounds, dict):
        rounds = rounds.get("rounds", rounds.get("data", []))
    active = next((r for r in rounds if r.get("status") == "active"), None)
    if not active:
        print("No active round.")
        for r in rounds:
            print(f"  Round {r.get('round_number')}: {r.get('status')}")
        return

    round_id = active["id"]
    detail = api_request(session, "GET", f"/astar-island/rounds/{round_id}")
    W, H = detail["map_width"], detail["map_height"]
    seeds_count = detail["seeds_count"]
    initial_states = detail["initial_states"]
    print(f"Round {active.get('round_number')}: {W}x{H}, {seeds_count} seeds")
    print(f"Closes: {active.get('closes_at')}")

    for i, st in enumerate(initial_states):
        n_sett = len(st.get("settlements", []))
        n_port = sum(1 for s in st.get("settlements", []) if s.get("has_port"))
        print(f"  Seed {i}: {n_sett} settlements ({n_port} ports)")

    # Submit-only mode: load saved predictions and submit
    if submit_only:
        for i in range(seeds_count):
            pred = np.load(f"predictions_seed_{i}.npy")
            resp = api_request(session, "POST", "/astar-island/submit", {
                "round_id": round_id, "seed_index": i,
                "prediction": pred.tolist(),
            })
            print(f"  Seed {i}: {resp}")
        return

    # Initialize data structures
    seeds = list(range(seeds_count))
    observations = {s: [[None] * W for _ in range(H)] for s in seeds}
    counts = {s: np.zeros((H, W, NUM_CLASSES), dtype=np.int32)
              for s in seeds}
    settlements_data = {s: [] for s in seeds}
    query_log = []
    queries_used = 0

    # Resume from saved state
    if resume:
        observations, counts, settlements_data, queries_used, query_log = \
            load_state("observations.json")
        print(f"Resumed: {queries_used} queries loaded")

    # ── Phase 1: Execute queries ─────────────────────────────────────────
    if not no_query and not resume:
        budget_info = api_request(session, "GET", "/astar-island/budget")
        remaining = budget_info["queries_max"] - budget_info["queries_used"]
        queries_used = budget_info["queries_used"]
        budget_max = budget_info["queries_max"]
        print(f"Budget: {queries_used}/{budget_max} ({remaining} left)")

        if remaining > 0:
            # Settlement-focused repeated-query strategy
            optimizer = QueryOptimizer(
                W, H, seeds_count, remaining, initial_states,
            )
            plan = optimizer.plan_queries()
            print(optimizer.summary())

            for qi, (seed_idx, x, y, w, h) in enumerate(plan):
                if queries_used >= budget_max:
                    break

                try:
                    result = api_request(session, "POST",
                                         "/astar-island/simulate", {
                        "round_id": round_id,
                        "seed_index": seed_idx,
                        "viewport_x": x, "viewport_y": y,
                        "viewport_w": w, "viewport_h": h,
                    })
                except RuntimeError as e:
                    print(f"  Query failed: {e}")
                    time.sleep(1)
                    continue

                new_cells = store_observation(
                    result, seed_idx, x, y, H, W,
                    observations, counts, settlements_data, queries_used,
                )

                queries_used += 1
                n_obs = int(counts[seed_idx][y:y+h, x:x+w].sum(axis=2).max())
                query_log.append({
                    "seed": seed_idx, "x": x, "y": y, "w": w, "h": h,
                })
                print(f"  Q{queries_used}: seed={seed_idx} "
                      f"({x},{y},{w},{h}) +{new_cells} new, max_obs={n_obs}")

                save_state("observations.json", round_id, queries_used,
                           observations, counts, settlements_data, query_log)
                time.sleep(0.22)  # Stay under 5 req/s rate limit

    # ── Phase 2: Infer hidden parameters ─────────────────────────────────
    print("\nInferring hidden parameters...")
    inferrer = ParameterInference(initial_states, observations, counts,
                                   settlements_data=settlements_data)

    inferred_params = inferrer.infer()
    print(f"  MAP estimates: {inferred_params}")

    # Sample from posterior for ensemble predictions
    try:
        posterior_samples = inferrer.infer_posterior(n_samples=5)
        print(f"  Posterior samples: {len(posterior_samples)}")
    except Exception as e:
        print(f"  Posterior sampling failed ({e}), using MAP only")
        posterior_samples = [inferred_params]

    # ── Phase 3+4: Swarm prediction ─────────────────────────────────────
    print("\nBuilding swarm predictions...")
    swarm = SwarmCoordinator(
        initial_states=initial_states,
        W=W, H=H,
        seeds_count=seeds_count,
        inferred_params=inferred_params,
        posterior_samples=posterior_samples,
        mc_runs_per_agent=mc_runs,
        n_mc_agents=mc_agents if not no_mc else 0,
    )
    predictions = swarm.predict_all(
        counts=counts,
        observations=observations,
        settlements_data=settlements_data,
    )

    # ── Phase 5: Submit ──────────────────────────────────────────────────
    print("\nSubmitting predictions...")
    for seed_idx in range(seeds_count):
        pred = predictions[seed_idx]

        # Final safety checks
        pred = np.maximum(pred, PROB_FLOOR)
        pred /= pred.sum(axis=-1, keepdims=True)

        np.save(f"predictions_seed_{seed_idx}.npy", pred)

        if not dry_run:
            resp = api_request(session, "POST", "/astar-island/submit", {
                "round_id": round_id,
                "seed_index": seed_idx,
                "prediction": pred.tolist(),
            })
            print(f"  Seed {seed_idx}: {resp.get('status', resp)}")
        else:
            print(f"  Seed {seed_idx}: DRY RUN "
                  f"(shape={pred.shape}, "
                  f"sum_check={pred.sum(axis=-1).mean():.4f})")

    # ── Check scores ─────────────────────────────────────────────────────
    print("\nChecking scores...")
    my_rounds = api_request(session, "GET", "/astar-island/my-rounds")
    for r in (my_rounds if isinstance(my_rounds, list) else []):
        if r.get("id") == round_id:
            print(f"  Status: {r['status']}, "
                  f"submitted: {r.get('seeds_submitted')}/5")
            if r.get("round_score") is not None:
                print(f"  Score: {r['round_score']}")
                print(f"  Seeds: {r.get('seed_scores')}")
                print(f"  Rank: {r.get('rank')}/{r.get('total_teams')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Astar Island Agent v6 — Swarm Prediction"
    )
    parser.add_argument("--token", default=os.getenv("ASTAR_ISLAND_TOKEN"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--submit-only", action="store_true")
    parser.add_argument("--no-query", action="store_true")
    parser.add_argument("--no-mc", action="store_true",
                        help="Skip Monte Carlo agents in swarm")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--mc-runs", type=int, default=30,
                        help="Monte Carlo runs per swarm agent")
    parser.add_argument("--mc-agents", type=int, default=8,
                        help="Number of MC agents in swarm")
    args = parser.parse_args()

    if not args.token:
        parser.error("Token required: --token or ASTAR_ISLAND_TOKEN env var")

    run(args.token, args.dry_run, args.submit_only, args.no_query,
        args.resume, args.mc_runs, args.no_mc, args.mc_agents)
