#!/usr/bin/env python3
"""
Astar Island Cloud Run prediction service.

FastAPI app with polling loop that:
1. Polls for active rounds every 30s
2. On new round: submit lookup predictions immediately
3. Explore with 5 queries (1 per seed)
4. Compute shift from observations
5. Exploit with remaining queries, resubmitting every 10
6. Never crashes — all errors caught and logged
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import numpy as np
from fastapi import FastAPI

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("astar-cloud")

# ── Stub imports (built by other agents) ──────────────────────────
# These modules are expected to exist at runtime in the same directory.
try:
    from api_client import AstarClient
except ImportError:
    log.warning("api_client not found — using stub")

    class AstarClient:  # type: ignore[no-redef]
        def __init__(self, token: str):
            self.token = token
            log.warning("STUB AstarClient initialized — no real API calls")

        def get_rounds(self) -> list:
            return []

        def get_round(self, rid: str) -> Optional[dict]:
            return None

        def get_budget(self) -> Optional[dict]:
            return None

        def get_my_rounds(self) -> list:
            return []

        def simulate(self, rid, si, x, y, w=15, h=15) -> Optional[dict]:
            return None

        def submit(self, rid, si, pred) -> Optional[dict]:
            return None

        def get_analysis(self, rid, si) -> Optional[dict]:
            return None


try:
    from predictor import Predictor
except ImportError:
    log.warning("predictor not found — using stub")

    class Predictor:  # type: ignore[no-redef]
        def __init__(self, lookup: dict):
            self.lookup = lookup
            log.warning("STUB Predictor initialized")

        def predict(self, grid, H, W, shift=None) -> list:
            """Return uniform predictions as fallback."""
            pred = []
            for y in range(H):
                row = []
                for x in range(W):
                    row.append([1.0 / 6] * 6)
                pred.append(row)
            return pred

        def compute_shift(self, obs_transitions: dict) -> dict:
            return {}


try:
    from metadata_analyzer import extract_params
except ImportError:
    log.warning("metadata_analyzer not found — using stub")

    def extract_params(settlements: list) -> dict:  # type: ignore[misc]
        alive = [s for s in settlements if s.get("alive")]
        dead = [s for s in settlements if not s.get("alive")]
        total = len(alive) + len(dead)
        return {
            "survival_rate": len(alive) / max(total, 1),
            "avg_food": 0,
            "avg_population": 0,
            "avg_wealth": 0,
            "port_rate": 0,
            "faction_count": 0,
            "death_rate": 1 - len(alive) / max(total, 1),
        }


from query_strategy import QueryPlanner

# ── Constants ─────────────────────────────────────────────────────
TERRAIN_TO_CLASS = {10: 0, 11: 0, 0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5}
NUM_CLASSES = 6
POLL_INTERVAL = 30  # seconds

# ── App ───────────────────────────────────────────────────────────
app = FastAPI(title="Astar Island Agent")

# State
TOKEN = os.environ.get("AINM_TOKEN", "")
client: Optional[AstarClient] = None
predictor: Optional[Predictor] = None
completed: Set[str] = set()
startup_time = time.time()


def _load_lookup() -> dict:
    """Load gt_lookup.json from container or parent directory."""
    paths = [
        Path(__file__).parent / "gt_lookup.json",
        Path(__file__).parent.parent / "gt_lookup.json",
    ]
    for p in paths:
        if p.exists():
            log.info(f"Loaded lookup from {p}")
            return json.loads(p.read_text())
    log.warning("No gt_lookup.json found — predictor will use empty lookup")
    return {}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "completed_rounds": len(completed),
        "uptime_seconds": int(time.time() - startup_time),
        "has_token": bool(TOKEN),
        "has_predictor": predictor is not None,
    }


@app.on_event("startup")
async def startup():
    global client, predictor
    if not TOKEN:
        log.error("AINM_TOKEN not set — polling disabled")
        return
    client = AstarClient(TOKEN)
    lookup = _load_lookup()
    predictor = Predictor(lookup)
    log.info(f"Startup complete. Lookup has {len(lookup)} entries.")
    asyncio.create_task(poll_loop())


async def poll_loop():
    """Poll for rounds every POLL_INTERVAL seconds. Never crashes."""
    while True:
        try:
            await process_rounds()
        except Exception as e:
            log.error(f"Poll loop error: {e}", exc_info=True)
        await asyncio.sleep(POLL_INTERVAL)


async def process_rounds():
    """Check for active rounds and process any new ones."""
    if not client:
        return

    rounds = client.get_rounds()
    if not rounds:
        return

    for r in rounds:
        rid = r.get("id", "")
        status = r.get("status", "")
        if status != "active" or rid in completed:
            continue

        completed.add(rid)
        rnum = r.get("round_number", "?")
        log.info(f"=== Round {rnum} ({rid[:8]}) ACTIVE ===")
        try:
            await process_round(r)
        except Exception as e:
            log.error(f"Round {rnum} error: {e}", exc_info=True)


async def process_round(round_info: Dict[str, Any]):
    """Process a single round through all phases."""
    if not client or not predictor:
        return

    rid = round_info["id"]
    rnum = round_info.get("round_number", "?")

    # Get round details
    detail = client.get_round(rid)
    if not detail:
        log.error(f"R{rnum}: Failed to get round details")
        return

    H = detail.get("map_height", 40)
    W = detail.get("map_width", 40)
    seeds_count = detail.get("seeds_count", 5)
    initial_states = detail.get("initial_states", [])

    if not initial_states:
        log.error(f"R{rnum}: No initial states")
        return

    # ── PHASE 1: Submit lookup predictions immediately ────────────
    log.info(f"R{rnum} Phase 1: Submitting lookup predictions for {seeds_count} seeds")
    for si in range(seeds_count):
        try:
            grid = initial_states[si].get("grid", [])
            pred = predictor.predict(grid, H, W)
            result = client.submit(rid, si, pred)
            if result:
                log.info(f"R{rnum} Phase 1: Seed {si} submitted")
            else:
                log.warning(f"R{rnum} Phase 1: Seed {si} submit failed")
        except Exception as e:
            log.error(f"R{rnum} Phase 1 seed {si}: {e}")
        await asyncio.sleep(0.1)  # Respect rate limits

    # ── Start MC simulation in background ────────────────────────
    # Runs in parallel with query phases. Results collected before final submit.
    mc_futures: Dict[int, asyncio.Task] = {}
    loop = asyncio.get_event_loop()
    for si in range(seeds_count):
        grid = initial_states[si].get("grid", [])
        # Extract settlement positions from the initial grid
        grid_settlements = _extract_settlements_from_grid(grid, H, W)
        mc_futures[si] = loop.run_in_executor(
            None,  # default ThreadPoolExecutor — MC internally uses ProcessPoolExecutor
            _run_mc_for_seed,
            predictor,
            grid,
            grid_settlements,
            None,  # params — defaults; will be refined in later rounds
            100,   # n_runs
        )
    log.info(f"R{rnum} MC: Background simulation started for {seeds_count} seeds")

    # ── Check budget ──────────────────────────────────────────────
    budget = client.get_budget()
    if not budget:
        log.warning(f"R{rnum}: Cannot get budget, stopping after Phase 1")
        return

    queries_max = budget.get("queries_max", 50)
    queries_used = budget.get("queries_used", 0)
    remaining = queries_max - queries_used
    log.info(f"R{rnum}: Budget {queries_used}/{queries_max}, remaining={remaining}")

    if remaining <= 0:
        log.warning(f"R{rnum}: No queries remaining")
        return

    # ── PHASE 2: Explore (5 queries, 1 per seed) ─────────────────
    planner = QueryPlanner(initial_states, H, W, remaining)
    explore_qs = planner.explore_phase()

    all_obs_transitions: Dict[int, List[int]] = {}
    all_settlements: List[Dict[str, Any]] = []

    log.info(f"R{rnum} Phase 2: Exploring with {len(explore_qs)} queries")
    for q in explore_qs:
        if remaining <= 0:
            break
        try:
            result = client.simulate(rid, q["seed"], q["x"], q["y"])
            if result:
                _accumulate_transitions(
                    result, initial_states[q["seed"]].get("grid", []),
                    q["x"], q["y"], H, W, all_obs_transitions,
                )
                all_settlements.extend(result.get("settlements", []))
                remaining = result.get("queries_max", queries_max) - result.get("queries_used", queries_used + 1)
                log.info(
                    f"R{rnum} Phase 2: seed={q['seed']} vp=({q['x']},{q['y']}) "
                    f"setts={len(result.get('settlements', []))} remaining={remaining}"
                )
            else:
                log.warning(f"R{rnum} Phase 2: simulate failed seed={q['seed']}")
                remaining -= 1
        except Exception as e:
            log.error(f"R{rnum} Phase 2 query error: {e}")
            remaining -= 1
        await asyncio.sleep(0.25)  # ~4 req/s, under 5/s limit

    # ── PHASE 3: Analyze metadata + compute shift + resubmit ─────
    params = extract_params(all_settlements)
    shift = predictor.compute_shift(all_obs_transitions)
    log.info(
        f"R{rnum} Phase 3: survival={params.get('survival_rate', '?'):.2f}, "
        f"transitions={len(all_obs_transitions)} classes, "
        f"shift keys={len(shift)}"
    )

    # Resubmit with shift-corrected predictions
    for si in range(seeds_count):
        try:
            grid = initial_states[si].get("grid", [])
            pred = predictor.predict(grid, H, W, shift=shift)
            client.submit(rid, si, pred)
        except Exception as e:
            log.error(f"R{rnum} Phase 3 submit seed {si}: {e}")
        await asyncio.sleep(0.1)
    log.info(f"R{rnum} Phase 3: Shifted predictions submitted")

    if remaining <= 0:
        log.info(f"R{rnum}: Budget exhausted after explore phase")
        return

    # ── PHASE 4: Exploit (remaining queries) ──────────────────────
    planner.update_params(params)
    exploit_qs = planner.exploit_phase(remaining, params)
    log.info(f"R{rnum} Phase 4: Exploiting with {len(exploit_qs)} queries")

    for i, q in enumerate(exploit_qs):
        if remaining <= 0:
            log.info(f"R{rnum} Phase 4: Budget exhausted at query {i}")
            break

        try:
            result = client.simulate(rid, q["seed"], q["x"], q["y"])
            if result:
                _accumulate_transitions(
                    result, initial_states[q["seed"]].get("grid", []),
                    q["x"], q["y"], H, W, all_obs_transitions,
                )
                all_settlements.extend(result.get("settlements", []))
                remaining = result.get("queries_max", queries_max) - result.get("queries_used", queries_used)
                log.debug(
                    f"R{rnum} Phase 4 [{i+1}/{len(exploit_qs)}]: "
                    f"seed={q['seed']} vp=({q['x']},{q['y']}) remaining={remaining}"
                )
            else:
                log.warning(f"R{rnum} Phase 4: simulate failed")
                remaining -= 1
        except Exception as e:
            log.error(f"R{rnum} Phase 4 query {i}: {e}")
            remaining -= 1

        await asyncio.sleep(0.25)

        # Every 10 queries: recompute shift and resubmit
        if (i + 1) % 10 == 0:
            try:
                shift = predictor.compute_shift(all_obs_transitions)
                for si in range(seeds_count):
                    grid = initial_states[si].get("grid", [])
                    pred = predictor.predict(grid, H, W, shift=shift)
                    client.submit(rid, si, pred)
                log.info(
                    f"R{rnum} Phase 4: Resubmitted after {i+1} exploit queries, "
                    f"transitions={len(all_obs_transitions)} classes"
                )
            except Exception as e:
                log.error(f"R{rnum} Phase 4 resubmit at {i+1}: {e}")
            await asyncio.sleep(0.2)

    # ── Final resubmit: blend lookup+shift with MC ─────────────────
    try:
        shift = predictor.compute_shift(all_obs_transitions)

        # Collect MC results (should be done by now after all queries)
        mc_preds: Dict[int, Optional[List]] = {}
        for si in range(seeds_count):
            if si in mc_futures:
                try:
                    mc_preds[si] = await mc_futures[si]
                    log.info(f"R{rnum} MC: Seed {si} completed")
                except Exception as e:
                    log.error(f"R{rnum} MC seed {si} failed: {e}")
                    mc_preds[si] = None
            else:
                mc_preds[si] = None

        for si in range(seeds_count):
            grid = initial_states[si].get("grid", [])
            lookup_pred = predictor.predict(grid, H, W, shift=shift)

            mc_pred = mc_preds.get(si)
            if mc_pred is not None:
                # Blend: 60% lookup+shift, 40% MC
                blended = _blend_predictions(lookup_pred, mc_pred, H, W, alpha=0.6)
                client.submit(rid, si, blended)
                log.info(f"R{rnum} Final: Seed {si} submitted (blended 60/40)")
            else:
                client.submit(rid, si, lookup_pred)
                log.info(f"R{rnum} Final: Seed {si} submitted (lookup only, MC failed)")

        log.info(f"R{rnum} Final: Submitted with {sum(len(v) if isinstance(v, list) else v for v in all_obs_transitions.values())} total transition observations")
    except Exception as e:
        log.error(f"R{rnum} Final submit: {e}")

    log.info(f"=== Round {rnum} COMPLETE ===")


def _extract_settlements_from_grid(grid: List[List[int]], H: int, W: int) -> List[Dict[str, Any]]:
    """Extract settlement/port positions from initial grid for MC simulator."""
    settlements = []
    owner_id = 0
    for y in range(H):
        for x in range(W):
            cell = grid[y][x] if y < len(grid) and x < len(grid[y]) else 0
            if cell == 1:  # SETTLEMENT
                settlements.append({
                    "x": x, "y": y, "alive": True, "has_port": False,
                    "owner_id": owner_id, "population": 1.0, "food": 1.0,
                })
                owner_id += 1
            elif cell == 2:  # PORT
                settlements.append({
                    "x": x, "y": y, "alive": True, "has_port": True,
                    "owner_id": owner_id, "population": 1.0, "food": 1.0,
                })
                owner_id += 1
    return settlements


def _run_mc_for_seed(pred, grid, settlements, params, n_runs):
    """Run MC simulation for a single seed's initial grid. Called in executor."""
    try:
        return pred.predict_mc(grid, settlements, params, n_runs=n_runs, n_workers=4)
    except Exception as e:
        log.error(f"MC worker error: {e}")
        return None


def _blend_predictions(
    lookup_pred: List[List[List[float]]],
    mc_pred: List[List[List[float]]],
    H: int,
    W: int,
    alpha: float = 0.6,
) -> List[List[List[float]]]:
    """Blend lookup+shift predictions with MC predictions.

    alpha: weight for lookup_pred (1-alpha for mc_pred).
    Returns normalized (H, W, 6) nested list.
    """
    FLOOR = 0.001
    blended = []
    for y in range(H):
        row = []
        for x in range(W):
            p = []
            total = 0.0
            for c in range(NUM_CLASSES):
                v = alpha * lookup_pred[y][x][c] + (1 - alpha) * mc_pred[y][x][c]
                v = max(v, FLOOR)
                p.append(v)
                total += v
            # Normalize
            for c in range(NUM_CLASSES):
                p[c] /= total
            row.append(p)
        blended.append(row)
    return blended


def _accumulate_transitions(
    sim_result: Dict[str, Any],
    initial_grid: List[List[int]],
    vx: int,
    vy: int,
    H: int,
    W: int,
    obs_transitions: Dict[int, List[int]],
):
    """Accumulate terrain transitions from a simulate response.

    Compares the observed (post-simulation) grid with the initial grid
    to track how each initial class transitions to final classes.

    obs_transitions: {initial_class: [list of observed final classes]}
    """
    sim_grid = sim_result.get("grid", [])
    viewport = sim_result.get("viewport", {})
    vp_x = viewport.get("x", vx)
    vp_y = viewport.get("y", vy)
    vp_w = viewport.get("w", 15)
    vp_h = viewport.get("h", 15)

    for row_idx in range(len(sim_grid)):
        for col_idx in range(len(sim_grid[row_idx]) if sim_grid[row_idx] else 0):
            map_y = vp_y + row_idx
            map_x = vp_x + col_idx

            if map_y >= H or map_x >= W:
                continue
            if map_y >= len(initial_grid) or map_x >= len(initial_grid[map_y]):
                continue

            initial_cell = initial_grid[map_y][map_x]
            final_cell = sim_grid[row_idx][col_idx]

            ic = TERRAIN_TO_CLASS.get(initial_cell, 0)
            fc = TERRAIN_TO_CLASS.get(final_cell, 0)

            if ic not in obs_transitions:
                obs_transitions[ic] = []
            obs_transitions[ic].append(fc)
