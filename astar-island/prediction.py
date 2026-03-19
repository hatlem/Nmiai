#!/usr/bin/env python3
"""
Prediction engine for Astar Island — layered probabilistic terrain prediction.

Layers (best-to-worst confidence):
1. Direct observation with KT estimator
2. Monte Carlo simulator predictions
3. Contextual transition model (cross-seed)
4. Global transition matrix
5. Domain priors (hardcoded fallback)

Spatial smoothing via loopy belief propagation.
Final calibration to minimize expected KL divergence.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter

from priors import (
    CALIBRATED_PRIORS as DOMAIN_PRIORS,
    MOUNTAIN_PRIOR,
    OCEAN_PRIOR,
    get_domain_prior,
    NUM_CLASSES,
    PROB_FLOOR,
)

# Additional floor constants used locally in prediction layers
STATIC_FLOOR = 0.002  # Tighter floor for near-impossible transitions (mountain/ocean)
REMOTE_FLOOR = 0.003  # Floor for unlikely transitions on remote cells

TERRAIN_TO_CLASS: dict[int, int] = {10: 0, 11: 0, 0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5}


def _get_cell_floors(init_grid: np.ndarray, sett_dist: np.ndarray) -> np.ndarray:
    """Compute per-cell, per-class probability floors based on initial terrain and context.

    Returns (H, W, 6) array of minimum probability floors per class.
    Key insight: some transitions are near-impossible, so we can use tighter floors
    to recover wasted probability mass for the likely classes.
    """
    H, W = init_grid.shape
    floors = np.full((H, W, NUM_CLASSES), PROB_FLOOR, dtype=np.float64)

    # Mountain cells: classes 0-4 get tight floor (mountains NEVER change)
    mountain_mask = (init_grid == 5)
    if mountain_mask.any():
        floors[mountain_mask, :5] = STATIC_FLOOR

    # Ocean cells: classes 1-5 get tight floor (ocean never changes)
    ocean_mask = (init_grid == 10)
    if ocean_mask.any():
        floors[ocean_mask, 1:] = STATIC_FLOOR

    # Empty cells far from settlements: settlement/port/ruin are very unlikely
    init_cls = np.zeros_like(init_grid, dtype=np.int32)
    for code, cls in TERRAIN_TO_CLASS.items():
        init_cls[init_grid == code] = cls

    remote_empty = (init_cls == 0) & (sett_dist > 8) & ~ocean_mask
    if remote_empty.any():
        floors[remote_empty, 1] = REMOTE_FLOOR  # settlement
        floors[remote_empty, 2] = REMOTE_FLOOR  # port
        floors[remote_empty, 3] = REMOTE_FLOOR  # ruin

    # Forest cells far from settlements: settlement/port/ruin are very unlikely
    remote_forest = (init_cls == 4) & (sett_dist > 8)
    if remote_forest.any():
        floors[remote_forest, 1] = REMOTE_FLOOR  # settlement
        floors[remote_forest, 2] = REMOTE_FLOOR  # port
        floors[remote_forest, 3] = REMOTE_FLOOR  # ruin

    return floors


def _compute_coastal_map(grid: np.ndarray) -> np.ndarray:
    """True for land cells adjacent to ocean (4-connected)."""
    H, W = grid.shape
    ocean = (grid == 10)
    coastal = np.zeros((H, W), dtype=bool)
    padded = np.pad(ocean, 1, constant_values=True)
    for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        coastal |= padded[1 + dy:H + 1 + dy, 1 + dx:W + 1 + dx]
    coastal &= ~ocean
    coastal &= (grid != 5)
    return coastal


def _compute_food_map(grid: np.ndarray) -> np.ndarray:
    """Count adjacent forest cells per cell (8-connected)."""
    H, W = grid.shape
    forest = (grid == 4).astype(np.float32)
    food = np.zeros((H, W), dtype=np.float32)
    for dy in range(-1, 2):
        for dx in range(-1, 2):
            if dy == 0 and dx == 0:
                continue
            shifted = np.zeros_like(forest)
            sy = slice(max(0, -dy), min(H, H - dy))
            sx = slice(max(0, -dx), min(W, W - dx))
            ty = slice(max(0, dy), min(H, H + dy))
            tx = slice(max(0, dx), min(W, W + dx))
            shifted[ty, tx] = forest[sy, sx]
            food += shifted
    return food


def _compute_settlement_distance(grid: np.ndarray, settlements: list,
                                  W: int, H: int) -> np.ndarray:
    """Manhattan distance to nearest settlement."""
    dist = np.full((H, W), 999.0)
    yy, xx = np.mgrid[0:H, 0:W]
    for s in settlements:
        sx, sy = s.get("x", -1), s.get("y", -1)
        if 0 <= sx < W and 0 <= sy < H:
            d = np.abs(xx - sx) + np.abs(yy - sy)
            dist = np.minimum(dist, d)
    return dist


def _compute_neighbor_settlements(grid: np.ndarray, H: int, W: int) -> np.ndarray:
    """Count settlement/port neighbors within radius 2."""
    sett_mask = np.isin(grid, [1, 2]).astype(np.int32)
    neighbor_count = np.zeros((H, W), dtype=np.int32)
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            if dy == 0 and dx == 0:
                continue
            shifted = np.zeros_like(sett_mask)
            sy = slice(max(0, -dy), min(H, H - dy))
            sx = slice(max(0, -dx), min(W, W - dx))
            ty = slice(max(0, dy), min(H, H + dy))
            tx = slice(max(0, dx), min(W, W + dx))
            shifted[ty, tx] = sett_mask[sy, sx]
            neighbor_count += shifted
    return neighbor_count


def _init_class_grid(grid: np.ndarray) -> np.ndarray:
    """Map raw terrain codes to 0-5 class indices."""
    H, W = grid.shape
    cls_grid = np.zeros((H, W), dtype=np.int32)
    for code, cls in TERRAIN_TO_CLASS.items():
        cls_grid[grid == code] = cls
    return cls_grid


class PredictionEngine:
    """Layered prediction engine for Astar Island terrain forecasting."""

    def __init__(self, initial_states: list, W: int, H: int, seeds_count: int):
        self.initial_states = initial_states
        self.W = W
        self.H = H
        self.seeds_count = seeds_count
        self._DOMAIN_PRIORS = DOMAIN_PRIORS

        # Precompute per-seed spatial features
        self._init_grids: list[np.ndarray] = []
        self._init_cls_grids: list[np.ndarray] = []
        self._coastal_maps: list[np.ndarray] = []
        self._food_maps: list[np.ndarray] = []
        self._sett_dists: list[np.ndarray] = []
        self._neighbor_setts: list[np.ndarray] = []

        for state in initial_states:
            grid = np.asarray(state["grid"], dtype=np.int64)
            self._init_grids.append(grid)
            self._init_cls_grids.append(_init_class_grid(grid))
            self._coastal_maps.append(_compute_coastal_map(grid))
            self._food_maps.append(_compute_food_map(grid))
            settlements = state.get("settlements", [])
            self._sett_dists.append(
                _compute_settlement_distance(grid, settlements, W, H))
            self._neighbor_setts.append(
                _compute_neighbor_settlements(grid, H, W))

    def _get_cell_prior(self, init_cls: int, sett_dist: float, food: float,
                        coastal: bool, neighbor_sett: int) -> np.ndarray:
        """Return informative Dirichlet prior (6,) for a cell based on context."""
        base = self._DOMAIN_PRIORS.get(init_cls, self._DOMAIN_PRIORS[0]).copy()

        # Coastal settlement/port: boost port probability
        if coastal and init_cls in (1, 2):
            base[2] += 0.08  # Port more likely on coast
            base[0] -= 0.04  # Less likely to become empty

        # High food + settlement: boost settlement survival
        if food >= 2 and init_cls == 1:
            base[1] += 0.10  # Settlement survives better with food
            base[0] -= 0.05  # Less likely to vanish
            base[3] -= 0.03  # Less likely to become ruin

        # Far from settlements: empty cells stay empty
        if sett_dist > 6 and init_cls == 0:
            base[0] = 0.92
            base[1] = 0.01
            base[2] = 0.01
            base[3] = 0.01
            base[4] = 0.04
            base[5] = 0.01

        # Far from settlements: forest stays forest
        if sett_dist > 6 and init_cls == 4:
            base[4] = 0.90
            base[0] = 0.04
            base[1] = 0.01
            base[2] = 0.01
            base[3] = 0.01
            base[5] = 0.01

        # Near settlements + empty: boost settlement/ruin probability
        if sett_dist <= 3 and init_cls == 0:
            base[1] += 0.06  # More likely to become settlement
            base[3] += 0.03  # Slightly more likely to become ruin
            base[0] -= 0.06  # Less likely to stay empty

        # Near settlements with many neighbors: even stronger settlement boost
        if neighbor_sett >= 2 and init_cls == 0 and sett_dist <= 4:
            base[1] += 0.04
            base[0] -= 0.03

        # Ensure non-negative and normalized
        base = np.maximum(base, PROB_FLOOR)
        base /= base.sum()
        return base

    def _get_prior_strength(self, init_cls: int, sett_dist: float,
                            n_obs: float) -> float:
        """Return prior strength that decays with observations.

        Static terrain gets strong prior, dynamic cells get weaker prior.
        Strength decays as 1/(1 + n_obs/base_strength) to trust data more.
        """
        # Base strength by terrain type
        if init_cls == 5:  # Mountain (static)
            base = 4.0
        elif init_cls == 10 or (init_cls == 0 and sett_dist > 6):
            base = 3.0  # Far from settlements, unlikely to change
        elif init_cls in (1, 2):  # Settlement/Port (most dynamic)
            base = 1.5
        elif sett_dist <= 3:  # Near settlements (dynamic area)
            base = 2.0
        else:
            base = 2.5  # Default moderate prior

        # Decay with observations: effective_strength = base / (1 + n_obs / base)
        # This means at n_obs == base, strength is halved
        if n_obs > 0:
            base = base / (1.0 + n_obs / base)

        return base

    def build_predictions(
        self,
        counts: dict[int, np.ndarray],
        settlements_data: dict[int, list],
        observations: dict[int, list],
        simulator_predictions: Optional[dict[int, np.ndarray]] = None,
        inferred_params: Optional[dict[str, float]] = None,
    ) -> dict[int, np.ndarray]:
        """
        Build final predictions for all seeds.

        Args:
            counts: seed_idx -> (H, W, 16) observation count arrays
            settlements_data: seed_idx -> list of settlement snapshots
            observations: seed_idx -> 2D list of observed values (None=unobserved)
            simulator_predictions: seed_idx -> (H, W, 6) from Monte Carlo sim
            inferred_params: dict of hidden parameter estimates

        Returns:
            dict seed_idx -> (H, W, 6) numpy array of probability distributions.
        """
        # Build cross-seed models
        global_trans = self._compute_global_transition(counts)
        ctx_trans = self._compute_contextual_transition(counts)

        if inferred_params is None:
            inferred_params = self._estimate_hidden_params(counts)

        predictions: dict[int, np.ndarray] = {}
        for seed_idx in range(self.seeds_count):
            pred = self._predict_seed(
                seed_idx=seed_idx,
                counts=counts[seed_idx],
                settlements_data=settlements_data.get(seed_idx, []),
                observations=observations[seed_idx],
                global_trans=global_trans,
                ctx_trans=ctx_trans,
                simulator_pred=simulator_predictions.get(seed_idx) if simulator_predictions else None,
                inferred_params=inferred_params,
            )
            predictions[seed_idx] = pred

        return predictions

    def _predict_seed(
        self,
        seed_idx: int,
        counts: np.ndarray,
        settlements_data: list,
        observations: list,
        global_trans: np.ndarray,
        ctx_trans: dict,
        simulator_pred: Optional[np.ndarray],
        inferred_params: dict[str, float],
    ) -> np.ndarray:
        """Predict terrain probabilities for a single seed."""
        H, W = self.H, self.W
        init_grid = self._init_grids[seed_idx]
        init_cls = self._init_cls_grids[seed_idx]
        coastal = self._coastal_maps[seed_idx]
        food_map = self._food_maps[seed_idx]
        sett_dist = self._sett_dists[seed_idx]
        neighbor_sett = self._neighbor_setts[seed_idx]

        # Remap terrain code bins to class bins
        # Terrain codes 10 (Ocean) and 11 (Plains) map to class 0
        cell_counts = counts[:, :, :NUM_CLASSES].astype(np.float64)
        if counts.shape[2] > 10:
            cell_counts[:, :, 0] += counts[:, :, 10].astype(np.float64)  # Ocean → class 0
        if counts.shape[2] > 11:
            cell_counts[:, :, 0] += counts[:, :, 11].astype(np.float64)  # Plains → class 0
        n_obs = cell_counts.sum(axis=2)
        observed_mask = n_obs > 0

        pred = np.zeros((H, W, NUM_CLASSES), dtype=np.float64)

        # === Layer 1: Direct observation (KT estimator) ===
        # KT: p_i = (count_i + 0.5) / (total + K * 0.5)
        kt_alpha = 0.5
        kt_denom = n_obs + NUM_CLASSES * kt_alpha
        kt_pred = (cell_counts + kt_alpha) / kt_denom[:, :, np.newaxis]

        # For cells with observations, start from KT estimate
        obs_cells = observed_mask
        pred[obs_cells] = kt_pred[obs_cells]

        # === Layer 2: Monte Carlo simulator ===
        if simulator_pred is not None:
            n_sim_equiv = 1.5  # Very low weight: simulator overestimates forest
            # For observed cells: geometric mean blend (optimal for KL divergence)
            weight_obs = n_obs / (n_obs + n_sim_equiv)
            weight_sim = 1.0 - weight_obs
            blended = self._ensemble_blend(
                [kt_pred, simulator_pred],
                [weight_obs[:, :, np.newaxis], weight_sim[:, :, np.newaxis]],
            )
            pred[obs_cells] = blended[obs_cells]
            # For unobserved cells: blend simulator with domain prior (NOT use directly)
            # Simulator has systematic biases (too much forest growth)
            unobs = ~observed_mask
            if unobs.any():
                # Build per-cell domain priors for unobserved cells
                domain_prior_grid = np.zeros((H, W, NUM_CLASSES), dtype=np.float64)
                for ic_val in range(NUM_CLASSES):
                    ic_mask = (init_cls == ic_val)
                    domain_prior_grid[ic_mask] = DOMAIN_PRIORS.get(ic_val, DOMAIN_PRIORS[0])
                # Blend: 30% simulator, 70% domain prior for unobserved
                sim_weight_unobs = 0.30
                pred[unobs] = (
                    sim_weight_unobs * simulator_pred[unobs] +
                    (1.0 - sim_weight_unobs) * domain_prior_grid[unobs]
                )
            # Mark simulator-covered cells so we skip lower layers
            has_prediction = np.ones((H, W), dtype=bool)
        else:
            has_prediction = observed_mask.copy()

        # === Layers 3-5: For cells without prediction yet ===
        needs_prediction = ~has_prediction

        if needs_prediction.any():
            # Vectorized: build context keys for all unobserved cells
            food_binned = np.clip(food_map.astype(int), 0, 3)
            n_sett_binned = np.clip(neighbor_sett, 0, 4)
            dist_bin = np.where(sett_dist <= 3, 0,
                       np.where(sett_dist <= 7, 1, 2))
            dist_labels = {0: "near", 1: "mid", 2: "far"}

            ys, xs = np.where(needs_prediction)
            for y, x in zip(ys, xs):
                ic = int(init_cls[y, x])
                raw_code = int(init_grid[y, x])

                # Hard constraints for static terrain
                if raw_code == 5:  # Mountain
                    pred[y, x] = MOUNTAIN_PRIOR.copy()
                    continue
                if raw_code == 10:  # Ocean
                    pred[y, x] = OCEAN_PRIOR.copy()
                    continue

                # Near-static: forest/plains far from all settlements
                # These cells almost never change in 50 years
                if sett_dist[y, x] > 6:
                    if ic == 4:  # Forest far from settlements
                        pred[y, x] = np.array([0.06, 0.005, 0.005, 0.005, 0.92, 0.005])
                        continue
                    if ic == 0 and food_map[y, x] == 0:  # Empty/plains, no forest
                        pred[y, x] = np.array([0.95, 0.005, 0.005, 0.005, 0.03, 0.005])
                        continue

                # Layer 3: Contextual transition
                ctx_key = (ic, int(food_binned[y, x]),
                           bool(coastal[y, x]),
                           int(n_sett_binned[y, x]),
                           dist_labels[int(dist_bin[y, x])])

                if ctx_key in ctx_trans:
                    dist = ctx_trans[ctx_key].copy()
                elif ic < NUM_CLASSES:
                    # Layer 4: Global transition
                    dist = global_trans[ic].copy()
                else:
                    # Layer 5: Domain prior
                    dist = DOMAIN_PRIORS.get(ic, DOMAIN_PRIORS[0]).copy()

                # Heuristic adjustments based on inferred params
                dist = self._apply_heuristics(
                    dist, ic, raw_code, y, x,
                    coastal, food_map, sett_dist, neighbor_sett,
                    inferred_params,
                )
                pred[y, x] = dist

        # Incorporate settlement metadata snapshots
        pred = self._apply_settlement_snapshots(pred, settlements_data)

        # Belief propagation spatial smoothing
        pred = self._belief_propagation(pred, observed_mask, init_grid)

        # Hard constraints
        pred = self._apply_hard_constraints(pred, init_grid, coastal)

        # Calibrate for KL minimization
        pred = self._calibrate_predictions(pred, observed_mask, n_obs)

        # Final floor and normalize
        pred = np.maximum(pred, PROB_FLOOR)
        pred /= pred.sum(axis=2, keepdims=True)

        obs_pct = 100 * observed_mask.sum() / (W * H)
        print(f"  Seed {seed_idx}: {obs_pct:.1f}% observed")

        return pred

    def _apply_heuristics(
        self,
        dist: np.ndarray,
        init_cls: int,
        raw_code: int,
        y: int, x: int,
        coastal: np.ndarray,
        food_map: np.ndarray,
        sett_dist: np.ndarray,
        neighbor_sett: np.ndarray,
        params: dict[str, float],
    ) -> np.ndarray:
        """Apply domain heuristics and inferred parameter adjustments."""
        aggression = params.get("faction_aggression", 0.0)
        winter = params.get("winter_severity", 0.0)
        trade = params.get("trade_activity", 0.0)
        forest_growth = params.get("forest_growth_rate", 0.0)

        food = food_map[y, x]
        is_coast = coastal[y, x]
        sd = sett_dist[y, x]

        # Settlement/Port dynamics — moderate adjustments
        if init_cls in (1, 2):
            if food >= 2:
                dist[1] += 0.10
                dist[3] += 0.04
            elif food >= 1:
                dist[1] += 0.06
                dist[3] += 0.08
            else:
                dist[1] += 0.02
                dist[3] += 0.12

            if init_cls == 2 or is_coast:
                dist[2] += 0.05

            # Hidden param adjustments (reduced magnitude)
            dist[3] += 0.08 * aggression + 0.06 * winter
            dist[1] -= 0.04 * (aggression + winter)
            if is_coast:
                dist[2] += 0.06 * trade

        # Near-settlement boost for non-settlement cells (reduced)
        if init_cls == 0 and sd <= 3:
            dist[1] += 0.03
            dist[3] += 0.02

        # Forest dynamics — forest is very stable, barely grows into other cells
        if init_cls == 4:
            forest_pref = DOMAIN_PRIORS[4].copy()
            dist = 0.4 * dist + 0.6 * forest_pref
            dist[4] += 0.03 * forest_growth

        # Empty near forest -> very mild forest growth (was massively overestimated)
        if init_cls == 0 and food >= 2:
            dist[4] += 0.02 + 0.03 * forest_growth

        # Ruin reclamation — forest growth into ruins is also overestimated
        if init_cls == 3:
            if sd <= 4:
                dist[1] += 0.06
            dist[4] += 0.03 + 0.04 * forest_growth
            dist[0] += 0.08

        # Suppress impossible transitions
        if raw_code != 5:
            dist[5] *= 0.2

        dist = np.maximum(dist, 0.0)
        return dist

    def _apply_settlement_snapshots(
        self,
        pred: np.ndarray,
        settlements_data: list,
    ) -> np.ndarray:
        """Adjust predictions using settlement metadata from simulation snapshots."""
        H, W = self.H, self.W
        sett_history: dict[tuple[int, int], list[dict]] = defaultdict(list)

        for snapshot in settlements_data:
            for s in snapshot.get("settlements", []):
                sx, sy = s.get("x", -1), s.get("y", -1)
                if 0 <= sx < W and 0 <= sy < H:
                    sett_history[(sx, sy)].append({
                        "alive": s.get("alive", True),
                        "has_port": s.get("has_port", False),
                        "population": s.get("population", 1.0),
                        "food": s.get("food", 0.5),
                    })

        for (sx, sy), snaps in sett_history.items():
            n = len(snaps)
            if n == 0:
                continue

            alive_rate = sum(1 for s in snaps if s["alive"]) / n
            port_rate = sum(1 for s in snaps if s["has_port"]) / n
            avg_food = np.mean([s["food"] for s in snaps])

            w = min(n * 0.25, 1.2)
            pred[sy, sx, 1] += w * alive_rate * (1 - port_rate)
            pred[sy, sx, 2] += w * port_rate
            pred[sy, sx, 3] += w * (1 - alive_rate)

            if avg_food < 0.2:
                pred[sy, sx, 3] += 0.12 * w
                pred[sy, sx, 1] -= 0.06 * w

            pred[sy, sx] = np.maximum(pred[sy, sx], 0.0)

        return pred

    def _belief_propagation(
        self,
        pred: np.ndarray,
        observed_mask: np.ndarray,
        initial_grid: np.ndarray,
        n_iterations: int = 5,
    ) -> np.ndarray:
        """
        Propagate beliefs from observed to unobserved cells.

        Respects terrain boundaries (ocean/mountain are hard constraints).
        Spatial prior: neighboring cells tend to have similar terrain.
        """
        H, W = self.H, self.W
        result = pred.copy()

        # Compatibility matrix: terrain-type aware to prevent forest bleeding
        # Start with strong self-compatibility
        compat = np.eye(NUM_CLASSES) * 0.75 + 0.25 / NUM_CLASSES
        # Boost settlement-related compatibility (they cluster)
        for i in [1, 2, 3]:
            for j in [1, 2, 3]:
                if i != j:
                    compat[i, j] = 0.12
        # Reduce forest -> non-forest propagation (forest is spatially stable)
        # Forest should NOT bleed into empty/settlement/port/ruin cells
        for j in [0, 1, 2, 3]:
            compat[4, j] = 0.02  # Forest belief barely propagates to non-forest
            compat[j, 4] = 0.02  # Non-forest belief barely propagates to forest
        # Empty <-> empty is fine (keep default)
        # Mountain is static anyway but reduce its propagation too
        for j in range(5):
            compat[5, j] = 0.01
            compat[j, 5] = 0.01
        compat[5, 5] = 0.95

        # Static cells: mountains and ocean are anchored
        static_mask = (initial_grid == 5) | (initial_grid == 10)
        anchor_mask = observed_mask | static_mask

        # Damping factor — low to prevent over-smoothing toward neighbors
        damping = 0.15

        # Precompute the update mask (cells that belief propagation may modify)
        update_mask = ~anchor_mask

        for iteration in range(n_iterations):
            # Start from anchored values; only accumulate messages for unanchored cells
            new_result = result.copy()

            for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                # Shifted neighbor beliefs
                neighbor = np.zeros_like(result)
                src_y = slice(max(0, -dy), min(H, H - dy))
                src_x = slice(max(0, -dx), min(W, W - dx))
                dst_y = slice(max(0, dy), min(H, H + dy))
                dst_x = slice(max(0, dx), min(W, W + dx))
                neighbor[dst_y, dst_x] = result[src_y, src_x]

                # Message: compatibility @ neighbor belief
                # message[y,x,c] = sum_c' compat[c,c'] * neighbor[y,x,c']
                message = np.einsum('ij,hwj->hwi', compat, neighbor)

                # Only update unanchored cells — observed/static cells are never touched
                blend = damping
                new_result[update_mask] = (
                    (1 - blend) * new_result[update_mask] +
                    blend * message[update_mask]
                )

            # Re-normalize only unanchored cells
            unanchored = new_result[update_mask]
            unanchored = np.maximum(unanchored, 1e-12)
            unanchored /= unanchored.sum(axis=1, keepdims=True)
            new_result[update_mask] = unanchored

            result = new_result

        return result

    def _apply_hard_constraints(
        self,
        pred: np.ndarray,
        init_grid: np.ndarray,
        coastal: np.ndarray,
    ) -> np.ndarray:
        """Apply hard physical constraints after soft predictions."""
        H, W = self.H, self.W

        # Mountain cells
        mountain_mask = (init_grid == 5)
        if mountain_mask.any():
            pred[mountain_mask] = MOUNTAIN_PRIOR[np.newaxis, :]

        # Ocean cells
        ocean_mask = (init_grid == 10)
        if ocean_mask.any():
            pred[ocean_mask] = OCEAN_PRIOR[np.newaxis, :]

        # Non-coastal cells: port probability = floor
        non_coastal = ~coastal
        pred[non_coastal, 2] = PROB_FLOOR

        # Normalize
        pred = np.maximum(pred, PROB_FLOOR)
        pred /= pred.sum(axis=2, keepdims=True)

        return pred

    def _calibrate_predictions(
        self,
        pred: np.ndarray,
        observed_mask: np.ndarray,
        n_observations: np.ndarray,
    ) -> np.ndarray:
        """
        Calibrate predictions to minimize expected KL divergence.

        IMPORTANT: For unobserved cells we do NOT shrink toward uniform.
        The domain priors are already well-calibrated from validation data.
        Shrinking toward uniform destroys the strong Empty-stays-Empty signal
        and causes catastrophic KL loss.

        For observed cells with few observations, use very mild shrinkage.
        """
        H, W = self.H, self.W
        uniform = np.full(NUM_CLASSES, 1.0 / NUM_CLASSES)

        calibrated = pred.copy()

        # Unobserved cells: NO shrinkage toward uniform.
        # The domain priors and heuristics already encode our best guess.
        # Shrinking toward uniform makes Empty cells ~0.11 per class which is terrible.

        # Few observations (1-2): very mild shrinkage
        alpha = 2.0
        few_obs_mask = (n_observations > 0) & (n_observations <= 2)
        if few_obs_mask.any():
            n_few = n_observations[few_obs_mask]
            weight_obs = (n_few / (n_few + alpha))[:, np.newaxis]
            weight_prior = 1.0 - weight_obs
            calibrated[few_obs_mask] = (
                weight_obs * pred[few_obs_mask] + weight_prior * uniform[np.newaxis, :]
            )

        # n > 2: trust empirical (KT already handles this well)

        return calibrated

    def _ensemble_blend(
        self,
        predictions_list: list[np.ndarray],
        weights: list,
    ) -> np.ndarray:
        """
        Weighted geometric mean of predictions (optimal for KL divergence).

        Geometric mean in log space avoids underflow and is the
        information-theoretic optimal ensemble method for KL loss.

        Weights can be scalars or broadcastable arrays (e.g., per-cell weights
        with shape (H, W, 1)).
        """
        log_preds = [w * np.log(p + 1e-12) for p, w in zip(predictions_list, weights)]
        blended = np.exp(sum(log_preds))
        blended = np.maximum(blended, PROB_FLOOR)
        blended /= blended.sum(axis=-1, keepdims=True)
        return blended

    # ── Cross-seed model builders ──────────────────────────────────────────

    def _compute_global_transition(
        self,
        counts: dict[int, np.ndarray],
    ) -> np.ndarray:
        """Cross-seed transition matrix with Jeffreys smoothing."""
        trans = np.full((NUM_CLASSES, NUM_CLASSES), 0.5)
        np.fill_diagonal(trans, 5.0)

        for seed_idx, state in enumerate(self.initial_states):
            init_grid = self._init_grids[seed_idx]
            seed_counts = counts[seed_idx]

            for ic in range(NUM_CLASSES):
                mask = (self._init_cls_grids[seed_idx] == ic)
                if mask.any():
                    trans[ic] += seed_counts[mask][:, :NUM_CLASSES].sum(axis=0)

        row_sums = trans.sum(axis=1, keepdims=True)
        return trans / row_sums

    def _compute_contextual_transition(
        self,
        counts: dict[int, np.ndarray],
    ) -> dict[tuple, np.ndarray]:
        """
        Build context-dependent transition model.
        Key: (init_class, food_bin, is_coastal, neighbor_settlements_bin, dist_bin)
        Value: probability distribution over 6 classes (Jeffreys smoothed).
        """
        ctx_counts: dict[tuple, np.ndarray] = defaultdict(
            lambda: np.zeros(NUM_CLASSES))

        for seed_idx in range(self.seeds_count):
            seed_counts = counts[seed_idx]
            init_cls = self._init_cls_grids[seed_idx]
            food_map = self._food_maps[seed_idx]
            coastal = self._coastal_maps[seed_idx]
            sett_dist = self._sett_dists[seed_idx]
            neighbor_sett = self._neighbor_setts[seed_idx]

            H, W = self.H, self.W
            cell_obs_all = seed_counts[:, :, :NUM_CLASSES].astype(np.float64)
            obs_sum = cell_obs_all.sum(axis=2)
            has_obs = obs_sum > 0

            if not has_obs.any():
                continue

            # Build integer context arrays for all cells
            ic_flat = init_cls[has_obs]
            food_flat = np.clip(food_map[has_obs].astype(np.int32), 0, 3)
            coast_flat = coastal[has_obs].astype(np.int32)
            nsett_flat = np.clip(neighbor_sett[has_obs], 0, 4)
            sd_flat = sett_dist[has_obs]
            dist_flat = np.where(sd_flat <= 3, 0, np.where(sd_flat <= 7, 1, 2))

            # Encode context as a single integer for grouping:
            # ic in [0,5], food in [0,3], coast in [0,1], nsett in [0,4], dist in [0,2]
            encoded = (ic_flat * (4 * 2 * 5 * 3) +
                       food_flat * (2 * 5 * 3) +
                       coast_flat * (5 * 3) +
                       nsett_flat * 3 +
                       dist_flat)

            obs_flat = cell_obs_all[has_obs]  # (N, 6)

            unique_keys, inverse = np.unique(encoded, return_inverse=True)
            # Sum observations per unique context group
            for idx, key_enc in enumerate(unique_keys):
                group_mask = inverse == idx
                group_sum = obs_flat[group_mask].sum(axis=0)

                # Decode the key back to tuple
                remainder = int(key_enc)
                ic = remainder // (4 * 2 * 5 * 3)
                remainder %= (4 * 2 * 5 * 3)
                food = remainder // (2 * 5 * 3)
                remainder %= (2 * 5 * 3)
                coast = remainder // (5 * 3)
                remainder %= (5 * 3)
                nsett = remainder // 3
                dist_idx = remainder % 3
                dist_labels = {0: "near", 1: "mid", 2: "far"}

                key = (int(ic), int(food), bool(coast), int(nsett), dist_labels[dist_idx])
                ctx_counts[key] += group_sum

        ctx_probs: dict[tuple, np.ndarray] = {}
        for key, c in ctx_counts.items():
            total = c.sum()
            if total > 0:
                ctx_probs[key] = (c + 0.5) / (total + NUM_CLASSES * 0.5)

        return ctx_probs

    def _estimate_hidden_params(
        self,
        counts: dict[int, np.ndarray],
    ) -> dict[str, float]:
        """
        Estimate hidden simulator parameters from observed transitions.

        Returns dict with estimated parameter levels (0.0 - 1.0):
        - faction_aggression: Settlement -> Ruin rate
        - winter_severity: Settlement -> Ruin rate (correlated)
        - trade_activity: Settlement -> Port / Port survival rate
        - forest_growth_rate: Empty -> Forest rate
        - expansion_rate: Empty -> Settlement rate
        """
        trans_counts = np.zeros((NUM_CLASSES, NUM_CLASSES))
        for seed_idx in range(self.seeds_count):
            seed_counts = counts[seed_idx]
            cls_grid = self._init_cls_grids[seed_idx]
            for ic in range(NUM_CLASSES):
                mask = (cls_grid == ic)
                if mask.any():
                    trans_counts[ic] += seed_counts[mask][:, :NUM_CLASSES].sum(axis=0)

        row_sums = np.maximum(trans_counts.sum(axis=1, keepdims=True), 1.0)
        trans_probs = trans_counts / row_sums

        total_obs = trans_counts.sum()
        if total_obs < 50:
            return {}

        params: dict[str, float] = {}

        sett_to_ruin = trans_probs[1, 3] if trans_counts[1].sum() > 10 else 0.3
        params["faction_aggression"] = float(np.clip(sett_to_ruin * 2.0, 0.0, 1.0))
        params["winter_severity"] = float(np.clip(sett_to_ruin * 1.5, 0.0, 1.0))

        sett_to_port = trans_probs[1, 2] if trans_counts[1].sum() > 10 else 0.1
        port_survival = trans_probs[2, 2] if trans_counts[2].sum() > 5 else 0.3
        params["trade_activity"] = float(
            np.clip((sett_to_port + port_survival) * 1.5, 0.0, 1.0))

        empty_to_forest = trans_probs[0, 4] if trans_counts[0].sum() > 20 else 0.05
        params["forest_growth_rate"] = float(
            np.clip(empty_to_forest * 5.0, 0.0, 1.0))

        empty_to_sett = trans_probs[0, 1] if trans_counts[0].sum() > 20 else 0.05
        params["expansion_rate"] = float(
            np.clip(empty_to_sett * 5.0, 0.0, 1.0))

        return params
