"""
Astar Island — Information-Theoretic Query Optimizer

Scoring uses entropy-weighted KL divergence. Only dynamic cells (near settlements)
contribute meaningfully to score. Static terrain has ~0 entropy → ~0 score weight.

Strategy (50 queries, 5 seeds):
  Phase 1 (65% focus / 35% other): Every seed gets minimum coverage
    (all settlement viewports + 1 repeat). Remaining budget distributed
    proportional to settlement count.
  Phase 2: Adaptive overflow — pick highest info-gain viewport across
    all seeds, with coverage bonus for under-observed seeds.

Key improvements:
  - Minimum coverage guarantee per seed (no seed gets starved)
  - 65/35 focus/other split (was 80/20) for better average scores
  - Coverage bonus in adaptive phase for under-observed seeds
  - Information-theoretic gain weighting per cell
  - Expansion zone viewports (Empty→Settlement is 13%!)
  - Unequal seed allocation weighted by settlement count
"""

from __future__ import annotations

from typing import Optional

import numpy as np

NUM_CLASSES = 6
TERRAIN_TO_CLASS = {10: 0, 11: 0, 0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5}

# Entropy priors by terrain type and proximity to settlements
_ENTROPY_SETTLEMENT = 1.5   # Settlement/Port: most dynamic
_ENTROPY_RUIN = 1.5         # Ruin: also very dynamic
_ENTROPY_EMPTY_NEAR = 1.0   # Empty near settlements (expansion zone)
_ENTROPY_FOREST_NEAR = 0.5  # Forest near settlements
_ENTROPY_EMPTY_FAR = 0.3    # Empty far from settlements
_ENTROPY_FOREST_FAR = 0.2   # Forest far from settlements
_ENTROPY_STATIC = 0.0       # Mountain/Ocean: never change


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

        # Precompute entropy prior maps per seed
        self._entropy_prior: list[np.ndarray] = [
            self._compute_entropy_prior(i) for i in range(seeds_count)
        ]

        # Precompute settlement viewports per seed
        self._seed_viewports: list[list[tuple[int, int, int, int]]] = [
            self._find_settlement_viewports(i) for i in range(seeds_count)
        ]

        # Precompute expansion zone viewports per seed
        self._expansion_viewports: list[list[tuple[int, int, int, int]]] = [
            self._find_expansion_viewports(i) for i in range(seeds_count)
        ]

        # Full coverage viewports (9-tile grid for 40x40 with 15x15 viewport)
        self._full_coverage_tiles = self._compute_full_coverage_tiles()

        # Rank seeds by settlement count (descending)
        self._seed_rank = sorted(
            range(seeds_count),
            key=lambda i: len(self.settlement_coords[i]),
            reverse=True,
        )

        # Settlement count weights for unequal allocation
        total_sett = sum(max(len(c), 1) for c in self.settlement_coords)
        self._seed_weights = [
            max(len(self.settlement_coords[i]), 1) / total_sett
            for i in range(seeds_count)
        ]

    def _compute_entropy_prior(self, seed_idx: int) -> np.ndarray:
        """
        Compute per-cell entropy prior based on terrain type and proximity
        to settlements. Higher entropy = more dynamic = more score contribution.
        """
        grid = self.grids[seed_idx]
        H, W = grid.shape
        entropy_map = np.zeros((H, W), dtype=np.float64)
        coords = self.settlement_coords[seed_idx]

        # Compute manhattan distance to nearest settlement for each cell
        if coords:
            dist_map = np.full((H, W), 999, dtype=np.int32)
            for sx, sy in coords:
                for y in range(H):
                    for x in range(W):
                        d = abs(x - sx) + abs(y - sy)
                        if d < dist_map[y, x]:
                            dist_map[y, x] = d
        else:
            dist_map = np.full((H, W), 999, dtype=np.int32)

        near_threshold = 6  # cells within 6 manhattan distance are "near"

        for y in range(H):
            for x in range(W):
                cell = grid[y, x]
                near = dist_map[y, x] <= near_threshold

                if cell == 10:  # Ocean
                    entropy_map[y, x] = _ENTROPY_STATIC
                elif cell == 11:  # Plains (class 0, but can become settlement)
                    entropy_map[y, x] = _ENTROPY_EMPTY_NEAR if near else _ENTROPY_EMPTY_FAR
                elif cell == 5:  # Mountain
                    entropy_map[y, x] = _ENTROPY_STATIC
                elif cell in (1, 2):  # Settlement, Port
                    entropy_map[y, x] = _ENTROPY_SETTLEMENT
                elif cell == 3:  # Ruin
                    entropy_map[y, x] = _ENTROPY_RUIN
                elif cell == 4:  # Forest
                    entropy_map[y, x] = _ENTROPY_FOREST_NEAR if near else _ENTROPY_FOREST_FAR
                elif cell == 0:  # Empty
                    entropy_map[y, x] = _ENTROPY_EMPTY_NEAR if near else _ENTROPY_EMPTY_FAR
                else:
                    entropy_map[y, x] = _ENTROPY_EMPTY_FAR

        return entropy_map

    def _compute_full_coverage_tiles(self) -> list[tuple[int, int, int, int]]:
        """Compute minimal set of tiles to cover entire W x H map."""
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

    def _find_expansion_viewports(
        self, seed_idx: int
    ) -> list[tuple[int, int, int, int]]:
        """
        Find viewports covering expansion zones: cells 3-6 manhattan distance
        from settlements. These are where new settlements appear (Empty->Settlement
        happens ~13% of the time in calibration data).
        """
        coords = self.settlement_coords[seed_idx]
        if not coords:
            return []

        grid = self.grids[seed_idx]
        H, W = grid.shape

        # Find expansion zone cells
        expansion_cells: list[tuple[int, int]] = []
        for y in range(H):
            for x in range(W):
                cell = grid[y, x]
                # Only empty cells can become settlements
                if cell not in (0, 4):  # Empty or Forest
                    continue
                # Skip static terrain
                if cell in (10, 11, 5):
                    continue

                min_dist = 999
                for sx, sy in coords:
                    d = abs(x - sx) + abs(y - sy)
                    if d < min_dist:
                        min_dist = d

                if 3 <= min_dist <= 6:
                    expansion_cells.append((x, y))

        if not expansion_cells:
            return []

        # Greedy set cover of expansion cells with viewports
        uncovered = set(range(len(expansion_cells)))
        viewports = []

        # Limit to 3 expansion viewports per seed to avoid over-allocation
        max_expansion_vps = 3

        while uncovered and len(viewports) < max_expansion_vps:
            best_vp = None
            best_covered: set = set()
            best_count = 0

            # Try centering on each uncovered expansion cell
            # Sample to avoid O(n^2)
            sample_indices = list(uncovered)
            if len(sample_indices) > 20:
                sample_indices = list(np.random.choice(
                    list(uncovered), size=20, replace=False
                ))

            for i in sample_indices:
                ex, ey = expansion_cells[i]
                vx = max(0, min(ex - self.viewport_max // 2,
                                self.W - self.viewport_max))
                vy = max(0, min(ey - self.viewport_max // 2,
                                self.H - self.viewport_max))
                vw = min(self.viewport_max, self.W - vx)
                vh = min(self.viewport_max, self.H - vy)

                contained = set()
                for j in uncovered:
                    cx, cy = expansion_cells[j]
                    if vx <= cx < vx + vw and vy <= cy < vy + vh:
                        contained.add(j)

                # Also count how many settlement viewport cells this overlaps
                # (re-using settlement viewports is efficient)
                entropy_score = float(
                    self._entropy_prior[seed_idx][vy:vy+vh, vx:vx+vw].sum()
                )

                count = len(contained) + entropy_score * 0.1
                if count > best_count:
                    best_vp = (vx, vy, vw, vh)
                    best_covered = contained
                    best_count = count

            if best_vp and best_covered:
                # Skip if this viewport is a duplicate of a settlement viewport
                is_dup = False
                for svp in self._seed_viewports[seed_idx]:
                    if svp == best_vp:
                        is_dup = True
                        break
                if not is_dup:
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

    def _viewport_info_gain(
        self,
        seed_idx: int,
        x: int, y: int, w: int, h: int,
        counts: Optional[dict] = None,
    ) -> float:
        """
        Compute expected information gain for a viewport using the formula:
            expected_info_gain per cell = entropy_prior(cell) * (K-1) / (2*(n+1)*(n+2))
        where K=6 classes and n=number of existing observations for that cell.

        Cells with higher entropy prior and fewer observations contribute more.
        """
        entropy_region = self._entropy_prior[seed_idx][y:y+h, x:x+w]

        if counts is not None and seed_idx in counts:
            seed_counts = counts[seed_idx]
            n_obs = seed_counts[y:y+h, x:x+w].sum(axis=2).astype(np.float64)
        else:
            n_obs = np.zeros((h, w), dtype=np.float64)

        # Information gain formula: entropy_prior * (K-1) / (2*(n+1)*(n+2))
        K = NUM_CLASSES
        gain_per_cell = entropy_region * (K - 1) / (2.0 * (n_obs + 1) * (n_obs + 2))

        return float(gain_per_cell.sum())

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
        Concentrated query plan: deep observation beats thin coverage.

        Key insights:
        1. Each query is an independent MC sample. Re-querying the same viewport
           gives a DIFFERENT stochastic outcome → more samples → better KT estimates.
        2. Hidden params are shared across all 5 seeds. Deep observation on 2-3 seeds
           gives excellent parameter inference; remaining seeds use cross-seed transfer.
        3. Entropy-weighted KL means only dynamic cells (settlements) matter.
           Observing static terrain is wasted budget.

        Strategy:
        - Pick top 3 seeds by settlement count ("focus seeds")
        - Allocate 60% of budget to focus seeds, 40% to others
        - Every seed gets at least len(settlement_viewports)+1 queries minimum
        - After minimums, remaining budget distributed proportional to settlement count
        - Within each seed: re-query settlement viewports multiple times
        - Expansion viewports get 1-2x (lower priority but still valuable)
        - NO full-coverage tiles (static terrain = wasted queries)

        Returns: list of (seed_idx, x, y, w, h)
        """
        plan: list[tuple[int, int, int, int, int]] = []

        # Track observation counts
        sim_counts: dict[int, np.ndarray] = {
            i: np.zeros((self.H, self.W), dtype=np.float64)
            for i in range(self.seeds_count)
        }

        def record_query(seed_idx: int, x: int, y: int, w: int, h: int):
            sim_counts[seed_idx][y:y+h, x:x+w] += 1.0

        # ── Allocate budget with minimum coverage guarantees ─────────────
        #
        # Every seed gets at least len(settlement_viewports) + 1 queries
        # so all settlement viewports are covered once plus one repeat.
        # After minimum allocations, remaining budget is distributed
        # proportionally by settlement count.

        n_focus = min(3, self.seeds_count)
        focus_seeds = self._seed_rank[:n_focus]
        other_seeds = self._seed_rank[n_focus:]

        # Step 1: Compute minimum allocation per seed
        min_per_seed: dict[int, int] = {}
        for s in range(self.seeds_count):
            n_vps = len(self._seed_viewports[s])
            min_per_seed[s] = max(2, n_vps)  # cover all viewports at least once

        total_min = sum(min_per_seed.values())

        # If total minimums exceed budget, scale them down proportionally
        if total_min > self.budget:
            scale = self.budget / total_min
            for s in min_per_seed:
                min_per_seed[s] = max(1, int(min_per_seed[s] * scale))
            total_min = sum(min_per_seed.values())

        # Step 2: 60/40 split between focus and other seeds
        focus_budget = int(self.budget * 0.60)
        other_budget = self.budget - focus_budget

        # Ensure other_budget can cover minimums for other seeds
        other_min = sum(min_per_seed[s] for s in other_seeds)
        if other_budget < other_min:
            other_budget = other_min
            focus_budget = self.budget - other_budget

        # Ensure focus_budget can cover minimums for focus seeds
        focus_min = sum(min_per_seed[s] for s in focus_seeds)
        if focus_budget < focus_min:
            focus_budget = focus_min
            other_budget = self.budget - focus_budget

        # Safety: clamp to budget
        focus_budget = min(focus_budget, self.budget)
        other_budget = min(other_budget, self.budget - focus_budget)

        # Step 3: Distribute focus budget — minimums first, then proportional
        focus_sett_counts = [max(len(self.settlement_coords[s]), 1) for s in focus_seeds]
        focus_total = sum(focus_sett_counts)
        focus_remaining = focus_budget - sum(min_per_seed[s] for s in focus_seeds)
        focus_per_seed: dict[int, int] = {}
        for i, s in enumerate(focus_seeds):
            extra = int(max(0, focus_remaining) * focus_sett_counts[i] / focus_total)
            focus_per_seed[s] = min_per_seed[s] + extra

        # Step 4: Distribute other budget — minimums first, then proportional
        other_sett_counts = [max(len(self.settlement_coords[s]), 1) for s in other_seeds]
        other_total = max(sum(other_sett_counts), 1)
        other_remaining = other_budget - sum(min_per_seed[s] for s in other_seeds)
        other_per_seed: dict[int, int] = {}
        for i, s in enumerate(other_seeds):
            extra = int(max(0, other_remaining) * other_sett_counts[i] / other_total)
            other_per_seed[s] = min_per_seed[s] + extra

        # ── Focus seeds: deep observation ─────────────────────────────────

        for seed_idx in focus_seeds:
            seed_budget = focus_per_seed[seed_idx]
            seed_used = 0

            sett_vps = self._seed_viewports[seed_idx]
            exp_vps = self._expansion_viewports[seed_idx]

            # Round-robin settlement viewports: each gets multiple observations
            # Target: 4-5 observations per settlement viewport
            target_per_vp = max(3, seed_budget // max(len(sett_vps), 1))
            target_per_vp = min(target_per_vp, 6)  # Cap to avoid over-concentration

            for repeat in range(target_per_vp):
                for vp in sett_vps:
                    if seed_used >= seed_budget:
                        break
                    plan.append((seed_idx, *vp))
                    record_query(seed_idx, *vp)
                    seed_used += 1

            # Expansion viewports: 1-2x each with remaining budget
            for vp in exp_vps:
                if seed_used >= seed_budget:
                    break
                plan.append((seed_idx, *vp))
                record_query(seed_idx, *vp)
                seed_used += 1

            # If still have budget, re-query highest-entropy settlement viewports
            if seed_used < seed_budget and sett_vps:
                ranked = self._rank_viewports_by_value(seed_idx, sett_vps)
                while seed_used < seed_budget:
                    for _score, vp in ranked:
                        if seed_used >= seed_budget:
                            break
                        plan.append((seed_idx, *vp))
                        record_query(seed_idx, *vp)
                        seed_used += 1

        # ── Other seeds: minimal coverage ─────────────────────────────────

        for seed_idx in other_seeds:
            seed_budget = other_per_seed.get(seed_idx, 2)
            seed_used = 0

            sett_vps = self._seed_viewports[seed_idx]

            # At least 2x per settlement viewport
            for vp in sett_vps:
                for _ in range(2):
                    if seed_used >= seed_budget:
                        break
                    plan.append((seed_idx, *vp))
                    record_query(seed_idx, *vp)
                    seed_used += 1

            # Fill remaining with expansion viewports
            for vp in self._expansion_viewports[seed_idx]:
                if seed_used >= seed_budget:
                    break
                plan.append((seed_idx, *vp))
                record_query(seed_idx, *vp)
                seed_used += 1

        # ── Overflow: use any remaining budget on info-gain ───────────────

        used = len(plan)
        remaining = self.budget - used

        if remaining > 0:
            # Adaptive: pick highest info-gain viewport across ALL seeds
            all_candidates = []
            for seed_idx in range(self.seeds_count):
                for vp in self._seed_viewports[seed_idx]:
                    all_candidates.append((seed_idx, vp))
                for vp in self._expansion_viewports[seed_idx]:
                    all_candidates.append((seed_idx, vp))

            # Deduplicate
            seen = set()
            deduped = []
            for seed_idx, vp in all_candidates:
                key = (seed_idx, *vp)
                if key not in seen:
                    seen.add(key)
                    deduped.append((seed_idx, vp))
            all_candidates = deduped

            def make_fake_counts():
                fake = {}
                for i in range(self.seeds_count):
                    c = np.zeros((self.H, self.W, NUM_CLASSES), dtype=np.float64)
                    c[:, :, 0] = sim_counts[i]
                    fake[i] = c
                return fake

            for _ in range(remaining):
                fake_counts = make_fake_counts()
                best_gain = -1.0
                best_query = None

                for seed_idx, vp in all_candidates:
                    x, y, w, h = vp
                    gain = self._viewport_info_gain(
                        seed_idx, x, y, w, h, counts=fake_counts
                    )
                    if gain > best_gain:
                        best_gain = gain
                        best_query = (seed_idx, *vp)

                if best_query is None:
                    break

                plan.append(best_query)
                record_query(best_query[0], best_query[1], best_query[2],
                             best_query[3], best_query[4])

        return plan

    def next_query(
        self,
        observations: dict,
        counts: dict,
        queries_remaining: int,
    ) -> tuple[int, int, int, int, int]:
        """
        Adaptive query selection based on information-theoretic gain.

        Uses the formula:
            expected_info_gain per cell = entropy_prior(cell) * (K-1) / (2*(n+1)*(n+2))
        where K=6 classes and n=number of existing observations.

        Logic:
        1. If any seed is completely unobserved, observe its best settlement viewport.
        2. Otherwise, pick the viewport with highest expected information gain
           across all seeds, considering settlement viewports, expansion zones,
           and full coverage tiles.
        """
        # Priority 1: Completely unobserved seeds (no observations at all)
        for seed_idx in self._seed_rank:
            seed_counts = counts.get(seed_idx)
            if seed_counts is None or seed_counts.sum() == 0:
                vps = self._seed_viewports[seed_idx]
                return (seed_idx, *vps[0])

        # Priority 2: Information-theoretic gain across all candidates
        # Add a coverage bonus for seeds with the fewest observations relative
        # to their dynamic cell count, so under-observed seeds catch up.
        best_gain = -1.0
        best_query: Optional[tuple[int, int, int, int, int]] = None

        # Compute per-seed observation ratio for coverage bonus
        seed_obs_ratios: dict[int, float] = {}
        for seed_idx in range(self.seeds_count):
            seed_counts = counts.get(seed_idx)
            dynamic_cells = float(np.sum(self._entropy_prior[seed_idx] > 0))
            dynamic_cells = max(dynamic_cells, 1.0)
            if seed_counts is not None:
                total_obs = float(seed_counts.sum())
            else:
                total_obs = 0.0
            seed_obs_ratios[seed_idx] = total_obs / dynamic_cells

        # The seed with the lowest ratio gets the highest bonus
        max_ratio = max(seed_obs_ratios.values()) if seed_obs_ratios else 1.0
        max_ratio = max(max_ratio, 1e-6)

        for seed_idx in range(self.seeds_count):
            # Coverage bonus: higher when this seed has fewer observations
            # relative to its dynamic cells compared to the best-observed seed.
            # Ranges from 1.0 (least observed) to ~0.3 (most observed).
            ratio = seed_obs_ratios[seed_idx]
            coverage_bonus = 1.0 - 0.7 * (ratio / max_ratio)

            # Build candidate list: settlement + expansion + coverage tiles
            candidates = list(self._seed_viewports[seed_idx])
            for vp in self._expansion_viewports[seed_idx]:
                if vp not in candidates:
                    candidates.append(vp)
            for tile in self._full_coverage_tiles:
                if tile not in candidates:
                    candidates.append(tile)

            for vp in candidates:
                x, y, w, h = vp
                gain = self._viewport_info_gain(
                    seed_idx, x, y, w, h, counts=counts
                )
                adjusted_gain = gain * coverage_bonus

                if adjusted_gain > best_gain:
                    best_gain = adjusted_gain
                    best_query = (seed_idx, x, y, w, h)

        if best_query:
            return best_query

        # Absolute fallback
        return (0, 0, 0, self.viewport_max, self.viewport_max)

    def summary(self) -> str:
        """Human-readable summary of query allocation."""
        return self.summary_from_plan(self.plan_queries())

    def summary_from_plan(self, plan: list) -> str:
        """Human-readable summary from a pre-built plan."""
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

        n_focus = min(3, self.seeds_count)
        focus_seeds = self._seed_rank[:n_focus]

        lines = [
            f"Query plan: {len(plan)} queries, "
            f"concentrated deep-observation strategy",
            f"  Focus seeds: {focus_seeds} (65% budget, min coverage guaranteed)",
        ]

        for seed_idx in range(self.seeds_count):
            info = seed_info.get(seed_idx, {
                "count": 0, "viewports": set(), "reobserves": 0
            })
            n = info["count"]
            n_vp = len(info["viewports"])
            n_reobs = info["reobserves"]
            n_sett = len(self.settlement_coords[seed_idx])
            avg_obs = n / max(n_vp, 1)
            focus = "*" if seed_idx in focus_seeds else " "
            lines.append(
                f" {focus}Seed {seed_idx}: {n} queries, {n_vp} viewports, "
                f"{avg_obs:.1f}x avg obs/vp ({n_sett} settlements)"
            )
        return "\n".join(lines)
