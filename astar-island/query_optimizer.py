"""
Astar Island — Settlement-Focused Repeated-Query Optimizer

Key insight: Scoring uses entropy-weighted KL divergence. Only dynamic cells
(near settlements) contribute to score. Static terrain (ocean, mountain,
isolated forest/plains) has zero entropy → zero score weight.

Strategy:
  1. From initial_state, identify all settlement positions per seed
  2. Find minimal viewports covering all settlements (greedy set cover)
  3. Allocate budget equally across seeds (~10 per seed)
  4. Cycle viewports: repeat same viewport 3-5x for empirical probability estimates
  5. Each repeated query = independent stochastic simulation → direct frequency counting

Why repeated queries beat broad coverage:
  - 5 observations of same cell → KT estimate with ~0.06 std error
  - 1 observation of 5 cells → binary observation, no probability info
  - Scoring rewards accurate probabilities, not map coverage
"""

from __future__ import annotations

from typing import Optional

import numpy as np

NUM_CLASSES = 6
TERRAIN_TO_CLASS = {10: 0, 11: 0, 0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5}


class QueryOptimizer:
    def __init__(
        self,
        W: int,
        H: int,
        seeds_count: int,
        budget: int,
        initial_states: list,
        viewport_max: int = 15,
    ):
        self.W = W
        self.H = H
        self.seeds_count = seeds_count
        self.budget = budget
        self.initial_states = initial_states
        self.viewport_max = viewport_max

        self.settlement_coords: list[list[tuple[int, int]]] = []
        self.grids: list[np.ndarray] = []

        for seed_idx in range(seeds_count):
            state = initial_states[seed_idx]
            grid = np.asarray(state["grid"], dtype=np.int64)
            self.grids.append(grid)

            settlements = state.get("settlements", [])
            coords = []
            for s in settlements:
                sx, sy = s.get("x", -1), s.get("y", -1)
                if 0 <= sx < W and 0 <= sy < H:
                    coords.append((sx, sy))
            self.settlement_coords.append(coords)

        # Precompute settlement viewports per seed
        self._seed_viewports: list[list[tuple[int, int, int, int]]] = [
            self._find_settlement_viewports(i) for i in range(seeds_count)
        ]

    def _find_settlement_viewports(
        self, seed_idx: int
    ) -> list[tuple[int, int, int, int]]:
        """
        Find minimal set of viewports covering all settlements (greedy set cover).
        Viewports centered on settlement clusters to maximize dynamic cell coverage.
        """
        coords = self.settlement_coords[seed_idx]
        if not coords:
            cx = max(0, self.W // 2 - self.viewport_max // 2)
            cy = max(0, self.H // 2 - self.viewport_max // 2)
            w = min(self.viewport_max, self.W - cx)
            h = min(self.viewport_max, self.H - cy)
            return [(cx, cy, w, h)]

        uncovered = set(range(len(coords)))
        viewports = []

        while uncovered:
            best_vp = None
            best_covered: set = set()
            best_score = -1.0

            # Try centering viewport on each uncovered settlement
            for i in uncovered:
                sx, sy = coords[i]
                vx = max(0, min(sx - self.viewport_max // 2,
                                self.W - self.viewport_max))
                vy = max(0, min(sy - self.viewport_max // 2,
                                self.H - self.viewport_max))
                vw = min(self.viewport_max, self.W - vx)
                vh = min(self.viewport_max, self.H - vy)

                # Count uncovered settlements in this viewport
                contained = set()
                for j in uncovered:
                    cx, cy = coords[j]
                    if vx <= cx < vx + vw and vy <= cy < vy + vh:
                        contained.add(j)

                # Score: settlements covered × 100 + dynamic cell count
                dynamic_score = self._viewport_dynamic_score(
                    seed_idx, vx, vy, vw, vh
                )
                score = len(contained) * 100 + dynamic_score

                if score > best_score:
                    best_vp = (vx, vy, vw, vh)
                    best_covered = contained
                    best_score = score

            if best_vp and best_covered:
                viewports.append(best_vp)
                uncovered -= best_covered
            else:
                break

        return viewports

    def _viewport_dynamic_score(
        self, seed_idx: int, x: int, y: int, w: int, h: int
    ) -> float:
        """Score a viewport by dynamic cell content."""
        grid = self.grids[seed_idx]
        region = grid[y : y + h, x : x + w]

        # Non-static cells (not ocean, not mountain)
        dynamic = float(np.sum((region != 10) & (region != 5)))

        # Settlement/port/ruin cells are most valuable
        sett_cells = float(np.sum(np.isin(region, [1, 2, 3])))

        # Forest near settlements can be cleared — semi-dynamic
        forest_cells = float(np.sum(region == 4))

        return dynamic + sett_cells * 5 + forest_cells * 0.5

    def plan_queries(self) -> list[tuple[int, int, int, int, int]]:
        """
        Settlement-focused repeated-query plan.

        Equal budget per seed, cycling through minimal settlement viewports.
        With 50 queries / 5 seeds = 10 per seed.
        With ~2-3 viewports per seed = 3-5 repeats per viewport.

        Returns: list of (seed_idx, x, y, w, h)
        """
        queries_per_seed = self.budget // self.seeds_count
        extra = self.budget % self.seeds_count

        plan: list[tuple[int, int, int, int, int]] = []

        for seed_idx in range(self.seeds_count):
            n_queries = queries_per_seed + (1 if seed_idx < extra else 0)
            viewports = self._seed_viewports[seed_idx]

            # Cycle through viewports, repeating for empirical estimates
            for i in range(n_queries):
                vp = viewports[i % len(viewports)]
                plan.append((seed_idx, *vp))

        return plan

    def next_query(
        self,
        observations: dict,
        counts: dict,
        queries_remaining: int,
    ) -> tuple[int, int, int, int, int]:
        """
        Adaptive query: re-observe settlement viewports with fewest observations.
        Targets cells where more samples improve probability estimates most.
        """
        best_gain = -1.0
        best_query: Optional[tuple[int, int, int, int, int]] = None

        for seed_idx in range(self.seeds_count):
            seed_counts = counts.get(seed_idx)

            for vp in self._seed_viewports[seed_idx]:
                x, y, w, h = vp

                if seed_counts is None:
                    # Completely unobserved — highest priority
                    return (seed_idx, x, y, w, h)

                region_counts = seed_counts[y : y + h, x : x + w]
                n_obs = region_counts.sum(axis=2).astype(float)
                grid_region = self.grids[seed_idx][y : y + h, x : x + w]

                # Importance: dynamic cells matter, static don't
                importance = np.ones_like(n_obs)
                importance[grid_region == 10] = 0.0  # Ocean
                importance[grid_region == 5] = 0.0  # Mountain
                importance[np.isin(grid_region, [1, 2, 3])] = 5.0

                # Gain = importance / (1 + existing observations)
                # Few observations on important cells = high value
                gain = float((importance / (1.0 + n_obs)).sum())

                if gain > best_gain:
                    best_gain = gain
                    best_query = (seed_idx, x, y, w, h)

        if best_query:
            return best_query

        # Absolute fallback
        return (0, 0, 0, self.viewport_max, self.viewport_max)

    def summary(self) -> str:
        """Human-readable summary of query allocation."""
        plan = self.plan_queries()
        seed_info: dict[int, dict] = {}

        for seed_idx, x, y, w, h in plan:
            if seed_idx not in seed_info:
                seed_info[seed_idx] = {"count": 0, "viewports": set()}
            seed_info[seed_idx]["count"] += 1
            seed_info[seed_idx]["viewports"].add((x, y, w, h))

        lines = [
            f"Query plan: {len(plan)} queries, "
            f"settlement-focused repeated-query strategy"
        ]
        for seed_idx in range(self.seeds_count):
            info = seed_info.get(seed_idx, {"count": 0, "viewports": set()})
            n = info["count"]
            n_vp = len(info["viewports"])
            n_sett = len(self.settlement_coords[seed_idx])
            repeats = n / max(n_vp, 1)
            lines.append(
                f"  Seed {seed_idx}: {n} queries across {n_vp} viewports "
                f"(~{repeats:.1f}x repeats, {n_sett} settlements)"
            )
        return "\n".join(lines)
