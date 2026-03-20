#!/usr/bin/env python3
"""
Astar Island Auto-Pilot — watches for new rounds, queries, predicts, submits.

Usage:
    export AINM_TOKEN="eyJ..."
    python3 autopilot.py

Runs continuously, checking for new active rounds every 30 seconds.
When a round is found:
  1. Plans queries using QueryOptimizer
  2. Executes all 50 queries via /simulate API
  3. Runs swarm predictions
  4. Submits all 5 seeds
"""

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import requests

from query_optimizer import QueryOptimizer, TERRAIN_TO_CLASS, NUM_CLASSES
from predictor import predict_all
from priors import STATIC_FLOOR

API_BASE = "https://api.ainm.no/astar-island"
TOKEN = os.environ.get("AINM_TOKEN", "")
POLL_INTERVAL = 30  # seconds between round checks


def headers():
    return {"Authorization": f"Bearer {TOKEN}"}


def get_rounds():
    r = requests.get(f"{API_BASE}/rounds", headers=headers())
    r.raise_for_status()
    return r.json()


def get_budget():
    r = requests.get(f"{API_BASE}/budget", headers=headers())
    r.raise_for_status()
    return r.json()


def get_round_detail(round_id: str):
    r = requests.get(f"{API_BASE}/rounds/{round_id}", headers=headers())
    r.raise_for_status()
    return r.json()


def simulate(round_id: str, seed_index: int, x: int, y: int, w: int, h: int):
    payload = {
        "round_id": round_id,
        "seed_index": seed_index,
        "viewport_x": x,
        "viewport_y": y,
        "viewport_w": w,
        "viewport_h": h,
    }
    r = requests.post(f"{API_BASE}/simulate", headers=headers(),
                      json=payload)
    r.raise_for_status()
    return r.json()


def submit(round_id: str, seed_index: int, prediction):
    payload = {
        "round_id": round_id,
        "seed_index": seed_index,
        "prediction": prediction,
    }
    r = requests.post(f"{API_BASE}/submit", headers=headers(),
                      json=payload)
    r.raise_for_status()
    return r.json()


def process_round(round_info: dict):
    round_id = round_info["id"]
    round_num = round_info["round_number"]
    closes_at = round_info["closes_at"]

    print(f"\n{'='*60}")
    print(f"ROUND {round_num} — {round_id}")
    print(f"Closes at: {closes_at}")
    print(f"{'='*60}")

    # Get round detail
    detail = get_round_detail(round_id)
    W = detail["map_width"]
    H = detail["map_height"]
    seeds_count = detail["seeds_count"]
    initial_states = detail["initial_states"]

    print(f"Map: {W}x{H}, {seeds_count} seeds")
    for i, st in enumerate(initial_states):
        n_sett = len(st.get("settlements", []))
        n_port = sum(1 for s in st.get("settlements", []) if s.get("has_port"))
        print(f"  Seed {i}: {n_sett} settlements ({n_port} ports)")

    # Check budget
    budget = get_budget()
    queries_used = budget["queries_used"]
    queries_max = budget["queries_max"]
    queries_left = queries_max - queries_used
    print(f"\nQuery budget: {queries_used}/{queries_max} used, {queries_left} remaining")

    # Initialize observation storage
    observations = {s: [[None] * W for _ in range(H)] for s in range(seeds_count)}
    counts = {s: np.zeros((H, W, NUM_CLASSES), dtype=np.int32) for s in range(seeds_count)}
    settlements_data = {s: [] for s in range(seeds_count)}

    # ── STAGE 1: Instant prior-only submission ────────────────────────
    # Submit immediately with just lookup + priors (no observations needed).
    # This ensures we have a baseline score even if queries fail.
    # Prior-only scores 73-88 on typical rounds.
    print("\n[STAGE 1] Submitting prior-only predictions (instant baseline)...")
    try:
        prior_preds = predict_all(
            initial_states=initial_states,
            counts=counts,  # Empty — no observations yet
            observations=observations,
            verbose=False,
        )
        for seed_idx in range(seeds_count):
            pred = prior_preds[seed_idx]
            pred = np.maximum(pred, STATIC_FLOOR)
            pred /= pred.sum(axis=-1, keepdims=True)
            resp = submit(round_id, seed_idx, pred.tolist())
            print(f"  Seed {seed_idx}: {resp.get('status', 'ok')}")
            time.sleep(0.55)
        print("  Stage 1 complete — baseline score secured.")
    except Exception as e:
        print(f"  Stage 1 failed: {e} — continuing to queries...")

    # ── STAGE 2: Query + resubmit ────────────────────────────────────
    if queries_left > 0:
        # Plan and execute queries
        print(f"\nPlanning {queries_left} queries...")
        optimizer = QueryOptimizer(
            W=W, H=H, seeds_count=seeds_count,
            budget=queries_left, initial_states=initial_states,
        )
        plan = optimizer.plan_queries()
        print(optimizer.summary())

        from datetime import datetime, timezone
        try:
            closes_dt = datetime.fromisoformat(closes_at.replace("Z", "+00:00"))
        except Exception:
            closes_dt = None

        print(f"\nExecuting {len(plan)} queries...")
        budget_exhausted = False
        for qi, (seed_idx, x, y, w, h) in enumerate(plan):
            if budget_exhausted:
                break
            # Stop 5 min before close to ensure time for predict+submit
            if closes_dt:
                remaining_min = (closes_dt - datetime.now(timezone.utc)).total_seconds() / 60
                if remaining_min < 5:
                    print(f"  Stopping queries — {remaining_min:.0f}min left")
                    break

            for attempt in range(5):
                try:
                    result = simulate(round_id, seed_idx, x, y, w, h)

                    grid_data = result.get("grid", [])
                    sett_list = result.get("settlements", [])

                    for row_idx, row in enumerate(grid_data):
                        gy = y + row_idx
                        if gy >= H:
                            break
                        for col_idx, cell in enumerate(row):
                            gx = x + col_idx
                            if gx >= W:
                                break
                            observations[seed_idx][gy][gx] = cell
                            cls = TERRAIN_TO_CLASS.get(cell, 0)
                            counts[seed_idx][gy, gx, cls] += 1

                    if sett_list:
                        settlements_data[seed_idx].append({"settlements": sett_list})

                    if (qi + 1) % 10 == 0 or qi == len(plan) - 1:
                        print(f"  Query {qi+1}/{len(plan)} done "
                              f"(seed={seed_idx}, viewport=({x},{y},{w},{h}))")

                    time.sleep(0.3)
                    break

                except requests.exceptions.HTTPError as e:
                    err_text = str(e).lower()
                    if "budget" in err_text or "exhausted" in err_text:
                        print(f"  Budget exhausted at query {qi+1}")
                        budget_exhausted = True
                        break
                    status = getattr(getattr(e, 'response', None), 'status_code', 0)
                    if status == 429:
                        wait = 1.5 * (attempt + 1)
                        print(f"  Query {qi+1} rate limited ({attempt+1}/5), {wait:.0f}s...")
                        time.sleep(wait)
                        continue
                    if status == 400:
                        print(f"  Query {qi+1}: round closed or invalid ({e})")
                        budget_exhausted = True
                        break
                    print(f"  Query {qi+1} failed: {e}")
                    time.sleep(1)
                    break
                except Exception as e:
                    print(f"  Query {qi+1} error: {e}")
                    time.sleep(1)
                    break

        # Save observations for crash recovery / re-submission
        obs_file = Path(f"observations_r{round_num}.json")
        obs_file.write_text(json.dumps({
            "round_id": round_id,
            "round_number": round_num,
            "observations": {str(s): observations[s] for s in range(seeds_count)},
            "counts": {str(s): counts[s].tolist() for s in range(seeds_count)},
            "settlements_data": {str(s): settlements_data[s] for s in range(seeds_count)},
        }, separators=(',', ':')), encoding="utf-8")
        print(f"\n  Observations saved to {obs_file}")

        # Print observation coverage
        for s in range(seeds_count):
            n_obs = int(counts[s].sum())
            n_cells = int((counts[s].sum(axis=2) > 0).sum())
            print(f"  Seed {s}: {n_obs} observations, {n_cells}/{W*H} cells covered")
    else:
        # Try to load saved observations from a previous run
        obs_file = Path(f"observations_r{round_num}.json")
        if obs_file.exists():
            print(f"Loading saved observations from {obs_file}...")
            saved = json.loads(obs_file.read_text())
            observations = {int(k): v for k, v in saved["observations"].items()}
            counts = {int(k): np.array(v, dtype=np.int32)
                      for k, v in saved["counts"].items()}
            settlements_data = {int(k): v for k, v in saved["settlements_data"].items()}
            for s in range(seeds_count):
                n_obs = int(counts[s].sum())
                n_cells = int((counts[s].sum(axis=2) > 0).sum())
                print(f"  Seed {s}: {n_obs} observations, {n_cells}/{W*H} cells covered")
        else:
            print("No queries remaining and no saved observations — skipping round")
            return

    # ── STAGE 3: Resubmit with observations ────────────────────────────
    print("\n[STAGE 3] Generating observation-enhanced predictions...")
    predictions = predict_all(
        initial_states=initial_states,
        counts=counts,
        observations=observations,
        settlements_data=settlements_data,
    )

    print("\n[STAGE 3] Resubmitting (overrides Stage 1 baseline)...")
    for seed_idx in range(seeds_count):
        pred = predictions[seed_idx]
        pred = np.maximum(pred, STATIC_FLOOR)
        pred /= pred.sum(axis=-1, keepdims=True)

        # Save backup
        np.save(f"predictions_r{round_num}_seed_{seed_idx}.npy", pred)

        try:
            result = submit(round_id, seed_idx, pred.tolist())
            status = result.get("status", "unknown")
            print(f"  Seed {seed_idx}: {status} "
                  f"(sum={pred.sum(axis=-1).mean():.4f})")
        except Exception as e:
            print(f"  Seed {seed_idx}: FAILED — {e}")

        # Rate limit: 2 req/s for submit
        time.sleep(0.55)

    print(f"\nRound {round_num} complete!")


def main():
    if not TOKEN:
        print("ERROR: Set AINM_TOKEN environment variable")
        sys.exit(1)

    print("Astar Island Auto-Pilot started")

    # Persist completed rounds to disk so restarts don't re-process
    completed_file = Path("completed_rounds.json")
    if completed_file.exists():
        completed_rounds = set(json.loads(completed_file.read_text()))
    else:
        completed_rounds = set()

    while True:
        try:
            rounds = get_rounds()
            active = [r for r in rounds if r["status"] == "active"
                      and r["id"] not in completed_rounds]

            if active:
                for round_info in active:
                    process_round(round_info)
                    completed_rounds.add(round_info["id"])
                    completed_file.write_text(json.dumps(list(completed_rounds)))
            else:
                now = time.strftime("%H:%M:%S UTC", time.gmtime())
                print(f"[{now}] No new active rounds. Waiting {POLL_INTERVAL}s...",
                      end="\r")

        except KeyboardInterrupt:
            print("\nStopped.")
            break
        except Exception as e:
            print(f"\nError: {e}")

        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    main()
