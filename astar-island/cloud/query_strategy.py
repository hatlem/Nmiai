#!/usr/bin/env python3
"""
Explore-then-exploit query planning for Astar Island.

Strategy:
- Phase 1 (explore): 5 queries, 1 per seed, centered on densest settlement cluster
- Phase 2 (exploit): remaining queries on high-value viewports scored by
  settlement_count*10 + coastal*5 + expansion*3
- Target: n>=5 observations per dynamic cell
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np

log = logging.getLogger(__name__)

TERRAIN_TO_CLASS = {10: 0, 11: 0, 0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5}
VIEWPORT_SIZE = 15


class QueryPlanner:
    """Plans explore-then-exploit queries across 5 seeds."""

    def __init__(
        self,
        initial_states: List[Dict[str, Any]],
        H: int = 40,
        W: int = 40,
        budget: int = 50,
    ):
        self.initial_states = initial_states
        self.H = H
        self.W = W
        self.budget = budget
        self.params: Optional[Dict[str, float]] = None
        # Track which viewports we've queried per seed: list of (x, y)
        self.queried: Dict[int, List[tuple]] = {i: [] for i in range(len(initial_states))}

    def update_params(self, params: Dict[str, float]):
        """Update round parameters from metadata analysis."""
        self.params = params

    def explore_phase(self) -> List[Dict[str, Any]]:
        """5 queries: 1 viewport per seed, centered on densest settlement cluster."""
        queries = []
        for si, state in enumerate(self.initial_states):
            setts = state.get("settlements", [])
            if setts:
                cx = int(np.mean([s["x"] for s in setts]))
                cy = int(np.mean([s["y"] for s in setts]))
            else:
                cx, cy = self.W // 2, self.H // 2

            vx = max(0, min(self.W - VIEWPORT_SIZE, cx - VIEWPORT_SIZE // 2))
            vy = max(0, min(self.H - VIEWPORT_SIZE, cy - VIEWPORT_SIZE // 2))

            queries.append({
                "seed": si,
                "x": vx,
                "y": vy,
                "reason": "explore",
            })
            self.queried[si].append((vx, vy))

        return queries

    def exploit_phase(self, remaining: int, params: Optional[Dict[str, float]] = None) -> List[Dict[str, Any]]:
        """Generate exploit queries for remaining budget.

        Scores candidate viewports by:
        - settlement_count * 10 (most dynamic cells)
        - coastal_count * 5 (ports form here)
        - expansion_zone * 3 (new settlements appear near existing)

        Distributes queries across seeds, repeating high-value viewports
        to build n>=5 observations per dynamic cell.
        """
        if params:
            self.params = params

        if remaining <= 0:
            return []

        queries = []
        num_seeds = len(self.initial_states)

        # Score candidate viewports for each seed
        seed_candidates: Dict[int, List[Dict[str, Any]]] = {}
        for si, state in enumerate(self.initial_states):
            candidates = self._score_viewports(si, state)
            seed_candidates[si] = candidates

        # Round-robin across seeds, picking best viewport each time
        # Allow repeats to build observation depth
        qi = 0
        while qi < remaining:
            for si in range(num_seeds):
                if qi >= remaining:
                    break
                candidates = seed_candidates.get(si, [])
                if not candidates:
                    # Fallback: center of map
                    vx = max(0, (self.W - VIEWPORT_SIZE) // 2)
                    vy = max(0, (self.H - VIEWPORT_SIZE) // 2)
                    queries.append({"seed": si, "x": vx, "y": vy, "reason": "fallback"})
                    qi += 1
                    continue

                # Pick best candidate (may repeat for observation depth)
                # Cycle through top candidates
                idx = len(self.queried[si]) % len(candidates)
                best = candidates[min(idx, len(candidates) - 1)]
                queries.append({
                    "seed": si,
                    "x": best["x"],
                    "y": best["y"],
                    "reason": best["reason"],
                })
                self.queried[si].append((best["x"], best["y"]))
                qi += 1

        return queries

    def _score_viewports(self, seed_idx: int, state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Score all possible viewport positions for a seed.

        Returns sorted list of {x, y, score, reason} dicts, best first.
        """
        grid = state.get("grid", [])
        setts = state.get("settlements", [])

        if not grid:
            return []

        H = len(grid)
        W = len(grid[0]) if grid else 0

        # Build maps for scoring
        sett_map = np.zeros((H, W), dtype=np.float32)
        coastal_map = np.zeros((H, W), dtype=np.float32)
        expansion_map = np.zeros((H, W), dtype=np.float32)

        # Mark settlement positions
        for s in setts:
            sx, sy = s.get("x", 0), s.get("y", 0)
            if 0 <= sy < H and 0 <= sx < W:
                sett_map[sy, sx] = 1.0
                # Mark coastal cells near settlements
                if s.get("has_port", False):
                    coastal_map[sy, sx] = 1.0

        # Expansion zones: land cells within 3 of settlements
        for s in setts:
            sx, sy = s.get("x", 0), s.get("y", 0)
            for dy in range(-3, 4):
                for dx in range(-3, 4):
                    ny, nx = sy + dy, sx + dx
                    if 0 <= ny < H and 0 <= nx < W:
                        cell = grid[ny][nx]
                        ic = TERRAIN_TO_CLASS.get(cell, 0)
                        # Land cells near settlements are expansion candidates
                        if ic == 0 and cell != 10:  # not ocean
                            expansion_map[ny, nx] = 1.0

        # Mark coastal cells (land adjacent to ocean)
        for y in range(H):
            for x in range(W):
                if grid[y][x] == 10:
                    continue
                for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < H and 0 <= nx < W and grid[ny][nx] == 10:
                        coastal_map[y, x] = max(coastal_map[y, x], 0.5)

        # Score each possible viewport position
        candidates = []
        step = max(1, VIEWPORT_SIZE // 2)  # stride for candidate generation

        for vy in range(0, max(1, H - VIEWPORT_SIZE + 1), step):
            for vx in range(0, max(1, W - VIEWPORT_SIZE + 1), step):
                vy_end = min(vy + VIEWPORT_SIZE, H)
                vx_end = min(vx + VIEWPORT_SIZE, W)

                sett_count = float(sett_map[vy:vy_end, vx:vx_end].sum())
                coast_count = float(coastal_map[vy:vy_end, vx:vx_end].sum())
                expand_count = float(expansion_map[vy:vy_end, vx:vx_end].sum())

                score = sett_count * 10.0 + coast_count * 5.0 + expand_count * 3.0

                # Adjust for params if available
                if self.params:
                    survival = self.params.get("survival_rate", 0.5)
                    if survival < 0.3:
                        # High death rate: settlements are most informative
                        score += sett_count * 5.0
                    if self.params.get("port_rate", 0) > 0.3:
                        # Lots of ports: coastal areas more important
                        score += coast_count * 3.0

                reason = "settlement" if sett_count >= coast_count else "coastal"
                if expand_count > sett_count + coast_count:
                    reason = "expansion"

                candidates.append({
                    "x": vx,
                    "y": vy,
                    "score": score,
                    "reason": reason,
                })

        # Sort by score descending
        candidates.sort(key=lambda c: c["score"], reverse=True)

        # Keep top candidates (enough for repeated querying)
        return candidates[:10]
