"""
Astar Island — 3-Phase Settlement-Focused Query Optimizer

Scoring uses entropy-weighted KL divergence. Only dynamic cells (near settlements)
contribute meaningfully to score. Static terrain has ~0 entropy → ~0 score weight.

Strategy (50 queries, 5 seeds):
  Phase 1 (20 queries): Full 9-tile coverage of primary seed (most settlements).
    Then 11 queries on 2 other seeds targeting settlement-dense viewports.
  Phase 2 (20 queries): Fill coverage for remaining 2 seeds (~7 tiles each),
    prioritizing viewports with more settlements.
  Phase 3 (10 queries): Re-observe the highest-value viewports across ALL seeds
    to get 2+ observations per dynamic cell for better Jeffreys estimates.

Why re-observation beats broader coverage:
  - 2 observations → Jeffreys estimate with ~0.15 std error
  - 1 observation → binary, no probability info beyond prior
  - Dynamic cells dominate the score; accuracy there matters most
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

        # Full coverage viewports (9-tile grid for 40x40 with 15x15 viewport)
        self._full_coverage_tiles = self._compute_full_coverage_tiles()

        # Rank seeds by settlement count (descending)
        self._seed_rank = sorted(
            range(seeds_count),
            key=lambda i: len(self.settlement_coords[i]),
            reverse=True,
        )

    def _compute_full_coverage_tiles(self) -> list[tuple[int, int, int, int]]:
        """Compute minimal set of tiles to cover entire W×H map."""
        tiles = []
        y = 0
        while y < self.H:
            h = min(self.viewport_max, self.H - y)
            x = 0
            while x < self.W:
                w = min(self.viewport_max, self.W - x)
                tiles.append((x, y, w, h))
                x += self.viewport_max
            y += self.viewport_max
        return tiles

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

            for i in uncovered:
                sx, sy = coords[i]
                vx = max(0, min(sx - self.viewport_max // 2,
                                self.W - self.viewport_max))
                vy = max(0, min(sy - self.viewport_max // 2,
                                self.H - self.viewport_max))
                vw = min(self.viewport_max, self.W - vx)
                vh = min(self.viewport_max, self.H - vy)

                contained = set()
                for j in uncovered:
                    cx, cy = coords[j]
                    if vx <= cx < vx + vw and vy <= cy < vy + vh:
                        contained.add(j)

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

        dynamic = float(np.sum((region != 10) & (region != 5)))
        sett_cells = float(np.sum(np.isin(region, [1, 2, 3])))
        forest_cells = float(np.sum(region == 4))

        return dynamic + sett_cells * 5 + forest_cells * 0.5

    def _rank_viewports_by_value(
        self, seed_idx: int, viewports: list[tuple[int, int, int, int]]
    ) -> list[tuple[float, tuple[int, int, int, int]]]:
        """Rank viewports by dynamic cell value (descending)."""
        scored = []
        for vp in viewports:
            x, y, w, h = vp
            score = self._viewport_dynamic_score(seed_idx, x, y, w, h)
            scored.append((score, vp))
        scored.sort(reverse=True)
        return scored

    def plan_queries(self) -> list[tuple[int, int, int, int, int]]:
        """
        3-phase query plan optimized for entropy-weighted KL divergence scoring.

        Phase 1 (40% budget): Full coverage of primary seed + settlement viewports
            on 2 secondary seeds.
        Phase 2 (40% budget): Fill coverage gaps on remaining seeds, prioritizing
            settlement-dense viewports.
        Phase 3 (20% budget): Re-observe highest-value viewports across all seeds
            for 2+ observations on dynamic cells.

        Returns: list of (seed_idx, x, y, w, h)
        """
        phase1_budget = max(1, int(self.budget * 0.40))
        phase3_budget = max(1, int(self.budget * 0.20))
        phase2_budget = self.budget - phase1_budget - phase3_budget

        plan: list[tuple[int, int, int, int, int]] = []

        # Track which tiles have been covered per seed
        covered_tiles: dict[int, set[tuple[int, int, int, int]]] = {
            i: set() for i in range(self.seeds_count)
        }

        # ── Phase 1: Primary seed full coverage + secondary settlement viewports
        primary = self._seed_rank[0]
        secondary_seeds = self._seed_rank[1:3] if len(self._seed_rank) > 1 else []

        # Full 9-tile coverage of primary seed
        primary_tiles = self._full_coverage_tiles[:]
        # Sort by dynamic score so we get the best tiles first if we run out
        primary_ranked = self._rank_viewports_by_value(primary, primary_tiles)

        phase1_used = 0
        for _score, vp in primary_ranked:
            if phase1_used >= phase1_budget:
                break
            plan.append((primary, *vp))
            covered_tiles[primary].add(vp)
            phase1_used += 1

        # Remaining phase 1 budget on secondary seeds' settlement viewports
        remaining_p1 = phase1_budget - phase1_used
        if remaining_p1 > 0 and secondary_seeds:
            per_secondary = max(1, remaining_p1 // len(secondary_seeds))
            for sec_seed in secondary_seeds:
                sec_vps = self._rank_viewports_by_value(
                    sec_seed, self._seed_viewports[sec_seed]
                )
                count = 0
                for _score, vp in sec_vps:
                    if count >= per_secondary or len(plan) >= phase1_budget:
                        break
                    plan.append((sec_seed, *vp))
                    covered_tiles[sec_seed].add(vp)
                    count += 1

        # ── Phase 2: Fill coverage gaps for all seeds
        # Determine uncovered full-coverage tiles per seed, ranked by value
        phase2_candidates: list[tuple[float, int, tuple[int, int, int, int]]] = []

        for seed_idx in range(self.seeds_count):
            uncovered = [
                t for t in self._full_coverage_tiles
                if t not in covered_tiles[seed_idx]
            ]
            for score, vp in self._rank_viewports_by_value(seed_idx, uncovered):
                phase2_candidates.append((score, seed_idx, vp))

        # Also add settlement viewports not yet covered
        for seed_idx in range(self.seeds_count):
            for vp in self._seed_viewports[seed_idx]:
                if vp not in covered_tiles[seed_idx]:
                    score = self._viewport_dynamic_score(seed_idx, *vp)
                    # Boost settlement viewports
                    phase2_candidates.append((score * 2, seed_idx, vp))

        # Sort by value descending
        phase2_candidates.sort(reverse=True, key=lambda x: x[0])

        phase2_used = 0
        for _score, seed_idx, vp in phase2_candidates:
            if phase2_used >= phase2_budget:
                break
            # Skip if already covered (dedup)
            if vp in covered_tiles[seed_idx]:
                continue
            plan.append((seed_idx, *vp))
            covered_tiles[seed_idx].add(vp)
            phase2_used += 1

        # If phase2 has leftover budget, start re-observing high-value tiles
        remaining_p2 = phase2_budget - phase2_used
        if remaining_p2 > 0:
            phase3_budget += remaining_p2

        # ── Phase 3: Re-observe highest-value viewports for 2+ observations
        reobserve_candidates: list[tuple[float, int, tuple[int, int, int, int]]] = []

        for seed_idx in range(self.seeds_count):
            for vp in covered_tiles[seed_idx]:
                score = self._viewport_dynamic_score(seed_idx, *vp)
                reobserve_candidates.append((score, seed_idx, vp))

        # Sort by dynamic value descending — re-observe the most valuable first
        reobserve_candidates.sort(reverse=True, key=lambda x: x[0])

        phase3_used = 0
        for _score, seed_idx, vp in reobserve_candidates:
            if phase3_used >= phase3_budget:
                break
            plan.append((seed_idx, *vp))
            phase3_used += 1

        return plan

    def next_query(
        self,
        observations: dict,
        counts: dict,
        queries_remaining: int,
    ) -> tuple[int, int, int, int, int]:
        """
        Adaptive query selection based on current observation state.

        Logic:
        1. If any seed is completely unobserved, observe its best settlement viewport.
        2. If there are uncovered settlement viewports, fill those first.
        3. Otherwise, re-observe viewports with the most terrain transitions
           (settlement/ruin/port changes) — these have highest entropy.
        """
        # Priority 1: Completely unobserved seeds
        for seed_idx in self._seed_rank:
            if counts.get(seed_idx) is None:
                vps = self._seed_viewports[seed_idx]
                return (seed_idx, *vps[0])

        # Priority 2: Uncovered settlement viewports
        best_uncovered_score = -1.0
        best_uncovered: Optional[tuple[int, int, int, int, int]] = None

        for seed_idx in range(self.seeds_count):
            seed_counts = counts.get(seed_idx)
            if seed_counts is None:
                continue
            for vp in self._seed_viewports[seed_idx]:
                x, y, w, h = vp
                region_obs = seed_counts[y : y + h, x : x + w].sum(axis=2)
                if region_obs.min() == 0:
                    # Has unobserved cells
                    score = self._viewport_dynamic_score(seed_idx, x, y, w, h)
                    if score > best_uncovered_score:
                        best_uncovered_score = score
                        best_uncovered = (seed_idx, x, y, w, h)

        if best_uncovered is not None:
            return best_uncovered

        # Priority 3: Re-observe viewports with most terrain transitions
        # (indicating high entropy / dynamic cells)
        best_gain = -1.0
        best_query: Optional[tuple[int, int, int, int, int]] = None

        for seed_idx in range(self.seeds_count):
            seed_counts = counts.get(seed_idx)
            if seed_counts is None:
                continue

            # Consider both settlement viewports and full coverage tiles
            candidates = list(self._seed_viewports[seed_idx])
            for tile in self._full_coverage_tiles:
                if tile not in candidates:
                    candidates.append(tile)

            for vp in candidates:
                x, y, w, h = vp
                region_counts = seed_counts[y : y + h, x : x + w]
                n_obs = region_counts.sum(axis=2).astype(float)

                # Count cells with observed transitions (multiple classes seen)
                classes_seen = (region_counts > 0).sum(axis=2)
                transition_cells = float(np.sum(classes_seen > 1))

                # Also consider cells with settlement/ruin/port in initial grid
                grid_region = self.grids[seed_idx][y : y + h, x : x + w]
                sett_cells = float(np.sum(np.isin(grid_region, [1, 2, 3])))

                # Dynamic importance weighting
                importance = np.ones_like(n_obs)
                importance[grid_region == 10] = 0.0  # Ocean — static
                importance[grid_region == 5] = 0.0   # Mountain — static
                importance[np.isin(grid_region, [1, 2, 3])] = 5.0  # Settlements
                importance[classes_seen > 1] *= 3.0  # Observed transitions

                # Gain: high importance, low observations
                gain = float((importance / (1.0 + n_obs)).sum())

                # Bonus for viewports with actual observed transitions
                gain += transition_cells * 10.0 + sett_cells * 2.0

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
                seed_info[seed_idx] = {
                    "count": 0, "viewports": set(), "reobserves": 0
                }
            vp = (x, y, w, h)
            if vp in seed_info[seed_idx]["viewports"]:
                seed_info[seed_idx]["reobserves"] += 1
            else:
                seed_info[seed_idx]["viewports"].add(vp)
            seed_info[seed_idx]["count"] += 1

        lines = [
            f"Query plan: {len(plan)} queries, "
            f"3-phase coverage+reobserve strategy"
        ]

        # Phase breakdown
        phase1_budget = max(1, int(self.budget * 0.40))
        phase3_budget = max(1, int(self.budget * 0.20))
        phase2_budget = self.budget - phase1_budget - phase3_budget
        lines.append(
            f"  Phases: {phase1_budget} coverage-primary + "
            f"{phase2_budget} fill-gaps + {phase3_budget} re-observe"
        )
        lines.append(
            f"  Primary seed: {self._seed_rank[0]} "
            f"({len(self.settlement_coords[self._seed_rank[0]])} settlements)"
        )

        for seed_idx in range(self.seeds_count):
            info = seed_info.get(seed_idx, {
                "count": 0, "viewports": set(), "reobserves": 0
            })
            n = info["count"]
            n_vp = len(info["viewports"])
            n_reobs = info["reobserves"]
            n_sett = len(self.settlement_coords[seed_idx])
            lines.append(
                f"  Seed {seed_idx}: {n} queries, {n_vp} unique viewports, "
                f"{n_reobs} re-observations ({n_sett} settlements)"
            )
        return "\n".join(lines)
