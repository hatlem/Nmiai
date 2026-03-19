#!/usr/bin/env python3
"""
Astar Island — Swarm Prediction Engine

Runs multiple independent prediction agents with diverse strategies and
parameter hypotheses, then ensembles via geometric mean (KL-optimal).

Swarm agents:
1. MonteCarloAgent (N instances) — MC sims with diverse posterior parameter samples
2. StatisticalAgent — Pure observation-based KT estimator + contextual transitions
3. TransitionAgent — Global transition matrix applied to initial state
4. PriorAgent — Domain-knowledge priors calibrated by inferred params
5. SpatialAgent — Belief propagation from observed to unobserved cells

Each agent produces a (H, W, 6) probability tensor. The SwarmCoordinator
combines them using weighted geometric mean, where weights reflect agent
confidence and diversity contribution.
"""

from __future__ import annotations

import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter

from priors import (
    CALIBRATED_PRIORS,
    MOUNTAIN_PRIOR,
    OCEAN_PRIOR,
    get_domain_prior,
    NUM_CLASSES,
    PROB_FLOOR,
    STATIC_FLOOR,
    REMOTE_FLOOR,
)

TERRAIN_TO_CLASS = {10: 0, 11: 0, 0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5}

# ── Base Agent ───────────────────────────────────────────────────────────────


@dataclass
class SwarmAgent:
    """Base class for swarm prediction agents."""
    name: str
    weight: float = 1.0  # Relative weight in ensemble

    def predict(
        self,
        seed_idx: int,
        initial_state: dict,
        W: int, H: int,
        counts: np.ndarray,
        observations: list,
        settlements_data: list,
        inferred_params: dict,
    ) -> np.ndarray:
        """Return (H, W, 6) probability array."""
        raise NotImplementedError


# ── Monte Carlo Agent ────────────────────────────────────────────────────────


class MonteCarloAgent(SwarmAgent):
    """Runs Monte Carlo simulations with a specific parameter set."""

    def __init__(self, name: str, params: dict, n_runs: int = 50, weight: float = 1.0):
        super().__init__(name=name, weight=weight)
        self.params = params
        self.n_runs = n_runs

    def predict(self, seed_idx, initial_state, W, H, counts, observations,
                settlements_data, inferred_params) -> np.ndarray:
        from simulator import NorseSimulator
        grid = initial_state["grid"]
        settlements = initial_state.get("settlements", [])
        return NorseSimulator.run_monte_carlo(
            grid, settlements, self.params, n_runs=self.n_runs,
        )


# ── Statistical Agent ────────────────────────────────────────────────────────


class StatisticalAgent(SwarmAgent):
    """Pure observation-based prediction using KT estimator + contextual model."""

    def __init__(self, weight: float = 2.0):
        super().__init__(name="statistical", weight=weight)

    def predict(self, seed_idx, initial_state, W, H, counts, observations,
                settlements_data, inferred_params) -> np.ndarray:
        init_grid = np.asarray(initial_state["grid"], dtype=np.int64)
        settlements = initial_state.get("settlements", [])
        cell_counts = counts[:, :, :NUM_CLASSES].astype(np.float64)
        n_obs = cell_counts.sum(axis=2)

        # KT estimator with informative Dirichlet priors
        pred = _build_kt_with_priors(init_grid, cell_counts, n_obs, settlements, H, W)

        # Hard constraints for static terrain
        pred[init_grid == 5] = MOUNTAIN_PRIOR
        pred[init_grid == 10] = OCEAN_PRIOR

        # Domain priors for unobserved non-static cells
        observed = n_obs > 0
        unobs = ~observed & (init_grid != 5) & (init_grid != 10)
        if unobs.any():
            for y, x in zip(*np.where(unobs)):
                ic = _classify(int(init_grid[y, x]))
                pred[y, x] = get_domain_prior(ic)

        pred = np.maximum(pred, PROB_FLOOR)
        pred /= pred.sum(axis=2, keepdims=True)
        return pred


# ── Transition Matrix Agent ──────────────────────────────────────────────────


class TransitionAgent(SwarmAgent):
    """Applies learned transition matrix to initial state."""

    def __init__(self, weight: float = 1.0):
        super().__init__(name="transition", weight=weight)

    def predict(self, seed_idx, initial_state, W, H, counts, observations,
                settlements_data, inferred_params) -> np.ndarray:
        init_grid = np.asarray(initial_state["grid"], dtype=np.int64)

        # Build global transition matrix from all counts (cross-seed data passed in)
        trans = np.full((NUM_CLASSES, NUM_CLASSES), 0.5)
        np.fill_diagonal(trans, 5.0)

        init_cls = _classify_grid(init_grid)
        for ic in range(NUM_CLASSES):
            mask = init_cls == ic
            if mask.any():
                trans[ic] += counts[mask][:, :NUM_CLASSES].sum(axis=0)

        row_sums = trans.sum(axis=1, keepdims=True)
        trans_probs = trans / row_sums

        # Apply transition to each cell based on initial class
        pred = np.zeros((H, W, NUM_CLASSES), dtype=np.float64)
        for ic in range(NUM_CLASSES):
            mask = init_cls == ic
            pred[mask] = trans_probs[ic]

        # Hard constraints
        pred[init_grid == 5] = MOUNTAIN_PRIOR
        pred[init_grid == 10] = OCEAN_PRIOR

        pred = np.maximum(pred, PROB_FLOOR)
        pred /= pred.sum(axis=2, keepdims=True)
        return pred


# ── Spatial Propagation Agent ────────────────────────────────────────────────


class SpatialAgent(SwarmAgent):
    """Propagates observed beliefs to unobserved cells via spatial smoothing."""

    def __init__(self, weight: float = 0.8):
        super().__init__(name="spatial", weight=weight)

    def predict(self, seed_idx, initial_state, W, H, counts, observations,
                settlements_data, inferred_params) -> np.ndarray:
        init_grid = np.asarray(initial_state["grid"], dtype=np.int64)
        settlements = initial_state.get("settlements", [])
        cell_counts = counts[:, :, :NUM_CLASSES].astype(np.float64)
        n_obs = cell_counts.sum(axis=2)
        observed = n_obs > 0

        # KT estimator with informative Dirichlet priors
        pred = _build_kt_with_priors(init_grid, cell_counts, n_obs, settlements, H, W)

        init_cls = _classify_grid(init_grid)
        unobs = ~observed & (init_grid != 5) & (init_grid != 10)
        for y, x in zip(*np.where(unobs)):
            pred[y, x] = get_domain_prior(int(init_cls[y, x]))

        # Hard constraints
        pred[init_grid == 5] = MOUNTAIN_PRIOR
        pred[init_grid == 10] = OCEAN_PRIOR

        # Gaussian spatial smoothing per class channel
        sigma = 1.5
        static_mask = (init_grid == 5) | (init_grid == 10)
        anchor_mask = observed | static_mask

        smoothed = np.zeros_like(pred)
        for c in range(NUM_CLASSES):
            smoothed[:, :, c] = gaussian_filter(pred[:, :, c], sigma=sigma)

        # Blend: keep anchored cells, use smoothed for unanchored
        result = pred.copy()
        update = ~anchor_mask
        if update.any():
            blend = 0.6
            result[update] = (1 - blend) * pred[update] + blend * smoothed[update]

        # Re-apply hard constraints
        result[init_grid == 5] = MOUNTAIN_PRIOR
        result[init_grid == 10] = OCEAN_PRIOR

        result = np.maximum(result, PROB_FLOOR)
        result /= result.sum(axis=2, keepdims=True)
        return result


# ── Contextual Heuristic Agent ───────────────────────────────────────────────


class HeuristicAgent(SwarmAgent):
    """Uses domain knowledge + inferred params for context-aware predictions."""

    def __init__(self, weight: float = 1.0):
        super().__init__(name="heuristic", weight=weight)

    def predict(self, seed_idx, initial_state, W, H, counts, observations,
                settlements_data, inferred_params) -> np.ndarray:
        init_grid = np.asarray(initial_state["grid"], dtype=np.int64)
        init_cls = _classify_grid(init_grid)
        settlements = initial_state.get("settlements", [])

        pred = np.zeros((H, W, NUM_CLASSES), dtype=np.float64)

        # Precompute spatial features
        coastal = _coastal_mask(init_grid)
        food_map = _food_map(init_grid)
        sett_dist = _settlement_distance(init_grid, settlements, W, H)

        aggression = inferred_params.get("faction_aggression", 0.3)
        winter = inferred_params.get("winter_severity", 0.4)
        trade = inferred_params.get("trade_activity", 0.5)
        forest_growth = inferred_params.get("forest_growth_rate", 0.15)

        for y in range(H):
            for x in range(W):
                raw = int(init_grid[y, x])
                ic = int(init_cls[y, x])

                if raw == 5:
                    pred[y, x] = MOUNTAIN_PRIOR
                    continue
                if raw == 10:
                    pred[y, x] = OCEAN_PRIOR
                    continue

                dist = get_domain_prior(ic).copy()
                food = food_map[y, x]
                sd = sett_dist[y, x]
                is_coast = coastal[y, x]

                # Settlement/Port dynamics
                if ic in (1, 2):
                    death_rate = 0.15 * aggression + 0.12 * winter
                    survival = max(0.2, 1.0 - death_rate)

                    if food >= 2:
                        survival += 0.15
                    elif food < 1:
                        survival -= 0.10

                    if ic == 2 or is_coast:
                        port_prob = 0.15 + 0.10 * trade
                    else:
                        port_prob = 0.03

                    ruin_prob = max(0.05, 1.0 - survival - port_prob)
                    dist[1] = survival * (1.0 - port_prob) if ic == 1 else survival * 0.3
                    dist[2] = port_prob if is_coast else PROB_FLOOR
                    dist[3] = ruin_prob
                    dist[0] = 0.03
                    dist[4] = 0.03
                    dist[5] = PROB_FLOOR

                # Empty near settlements -> expansion
                elif ic == 0 and sd <= 4:
                    exp_prob = 0.05 + 0.03 * (4 - sd)
                    dist[1] = exp_prob
                    dist[3] = exp_prob * 0.3
                    dist[0] = 1.0 - exp_prob * 1.5 - dist[4]

                # Forest near settlements -> potential clearing
                elif ic == 4 and sd <= 3:
                    dist[1] = 0.05
                    dist[0] = 0.08
                    dist[4] = 0.82

                # Empty near forest -> forest growth
                elif ic == 0 and food >= 2:
                    dist[4] = 0.10 + 0.10 * forest_growth
                    dist[0] = 1.0 - dist[4] - 0.02

                # Ruin dynamics
                elif ic == 3:
                    if sd <= 3:
                        dist[1] = 0.15
                    dist[4] = 0.10 + 0.08 * forest_growth
                    dist[3] = max(0.3, 1.0 - dist[1] - dist[4] - 0.1)
                    dist[0] = 0.08

                pred[y, x] = dist

        pred = np.maximum(pred, PROB_FLOOR)
        pred /= pred.sum(axis=2, keepdims=True)
        return pred


# ── Settlement Trajectory Agent ──────────────────────────────────────────────


class SettlementTrajectoryAgent(SwarmAgent):
    """Uses settlement metadata trajectories across queries for fine-grained
    prediction of settlement cells."""

    def __init__(self, weight: float = 1.5):
        super().__init__(name="settlement_trajectory", weight=weight)

    def predict(self, seed_idx, initial_state, W, H, counts, observations,
                settlements_data, inferred_params) -> np.ndarray:
        init_grid = np.asarray(initial_state["grid"], dtype=np.int64)
        settlements = initial_state.get("settlements", [])
        cell_counts = counts[:, :, :NUM_CLASSES].astype(np.float64)
        n_obs = cell_counts.sum(axis=2)

        # KT estimator with informative Dirichlet priors
        pred = _build_kt_with_priors(init_grid, cell_counts, n_obs, settlements, H, W)

        # Domain priors for unobserved non-static cells
        observed = n_obs > 0
        init_cls = _classify_grid(init_grid)
        unobs = ~observed & (init_grid != 5) & (init_grid != 10)
        for y, x in zip(*np.where(unobs)):
            pred[y, x] = get_domain_prior(int(init_cls[y, x]))

        # Hard constraints
        pred[init_grid == 5] = MOUNTAIN_PRIOR
        pred[init_grid == 10] = OCEAN_PRIOR

        # Overlay settlement trajectory data
        from collections import defaultdict
        sett_history = defaultdict(list)
        for snapshot in settlements_data:
            for s in snapshot.get("settlements", []):
                sx, sy = s.get("x", -1), s.get("y", -1)
                if 0 <= sx < W and 0 <= sy < H:
                    sett_history[(sx, sy)].append(s)

        for (sx, sy), snaps in sett_history.items():
            n = len(snaps)
            if n == 0:
                continue

            alive_count = sum(1 for s in snaps if s.get("alive", True))
            port_count = sum(1 for s in snaps if s.get("has_port", False))
            alive_rate = alive_count / n
            port_rate = port_count / n

            # High-confidence adjustment from settlement metadata
            w = min(n * 0.4, 2.0)
            p = pred[sy, sx].copy()
            p[1] = max(p[1], w * alive_rate * (1 - port_rate))
            p[2] = max(p[2], w * port_rate)
            p[3] = max(p[3], w * (1 - alive_rate) * 0.5)
            pred[sy, sx] = p

        pred = np.maximum(pred, PROB_FLOOR)
        pred /= pred.sum(axis=2, keepdims=True)
        return pred


class ContextualPoolingAgent(SwarmAgent):
    """
    Pools observations across ALL seeds by context key for much better
    distribution estimates than per-cell Jeffreys with n=1.

    Key insight: hidden parameters are shared across seeds, so a settlement
    with food=2, coastal=True on seed 0 behaves identically on seed 4.
    """

    def __init__(self, all_counts: dict, all_initial_states: list, W: int, H: int, weight: float = 2.0):
        super().__init__(name="contextual_pooling", weight=weight)
        self.ctx_distributions = self._build_context_distributions(all_counts, all_initial_states, W, H)

    def _build_context_distributions(self, all_counts, all_initial_states, W, H):
        """Build per-context probability distributions from pooled cross-seed data."""
        from collections import defaultdict
        ctx_counts = defaultdict(lambda: np.zeros(NUM_CLASSES))

        for seed_idx, state in enumerate(all_initial_states):
            if seed_idx not in all_counts:
                continue
            init_grid = np.asarray(state["grid"], dtype=np.int64)
            seed_counts = all_counts[seed_idx]

            food_map = _food_map(init_grid)
            coastal = _coastal_mask(init_grid)
            settlements = state.get("settlements", [])
            sett_dist = _settlement_distance(init_grid, settlements, W, H)

            # Count neighbor settlements
            sett_mask = np.isin(init_grid, [1, 2]).astype(np.int32)
            n_sett = np.zeros((H, W), dtype=np.int32)
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
                    n_sett += shifted

            init_cls = _classify_grid(init_grid)

            for y in range(H):
                for x in range(W):
                    cell_obs = seed_counts[y, x, :NUM_CLASSES].astype(np.float64)
                    if cell_obs.sum() == 0:
                        continue

                    ic = int(init_cls[y, x])
                    food = int(min(food_map[y, x], 3))
                    is_coast = bool(coastal[y, x])
                    ns = min(int(n_sett[y, x]), 4)
                    db = "near" if sett_dist[y, x] <= 3 else "mid" if sett_dist[y, x] <= 7 else "far"

                    key = (ic, food, is_coast, ns, db)
                    ctx_counts[key] += cell_obs

        # Normalize with Jeffreys smoothing
        ctx_probs = {}
        for key, counts_arr in ctx_counts.items():
            total = counts_arr.sum()
            if total > 0:
                ctx_probs[key] = (counts_arr + 0.5) / (total + NUM_CLASSES * 0.5)

        return ctx_probs

    def predict(self, seed_idx, initial_state, W, H, counts, observations,
                settlements_data, inferred_params) -> np.ndarray:
        init_grid = np.asarray(initial_state["grid"], dtype=np.int64)
        init_cls = _classify_grid(init_grid)
        food_map = _food_map(init_grid)
        coastal = _coastal_mask(init_grid)
        settlements = initial_state.get("settlements", [])
        sett_dist = _settlement_distance(init_grid, settlements, W, H)

        sett_mask = np.isin(init_grid, [1, 2]).astype(np.int32)
        n_sett = np.zeros((H, W), dtype=np.int32)
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
                n_sett += shifted

        pred = np.full((H, W, NUM_CLASSES), 1.0 / NUM_CLASSES)

        for y in range(H):
            for x in range(W):
                raw = int(init_grid[y, x])
                if raw == 5:
                    pred[y, x] = MOUNTAIN_PRIOR
                    continue
                if raw == 10:
                    pred[y, x] = OCEAN_PRIOR
                    continue

                ic = int(init_cls[y, x])
                food = int(min(food_map[y, x], 3))
                is_coast = bool(coastal[y, x])
                ns = min(int(n_sett[y, x]), 4)
                db = "near" if sett_dist[y, x] <= 3 else "mid" if sett_dist[y, x] <= 7 else "far"

                key = (ic, food, is_coast, ns, db)
                if key in self.ctx_distributions:
                    pred[y, x] = self.ctx_distributions[key]
                else:
                    # Try less specific key (drop neighbor count)
                    for ns2 in range(5):
                        key2 = (ic, food, is_coast, ns2, db)
                        if key2 in self.ctx_distributions:
                            pred[y, x] = self.ctx_distributions[key2]
                            break
                    else:
                        pred[y, x] = get_domain_prior(ic)

        pred = np.maximum(pred, PROB_FLOOR)
        pred /= pred.sum(axis=2, keepdims=True)
        return pred


# ── Swarm Coordinator ────────────────────────────────────────────────────────


class SwarmCoordinator:
    """
    Manages a swarm of prediction agents and combines their outputs.

    Ensemble method: weighted geometric mean (information-theoretically optimal
    for KL divergence loss).
    """

    def __init__(
        self,
        initial_states: list,
        W: int, H: int,
        seeds_count: int,
        inferred_params: dict,
        posterior_samples: list[dict],
        mc_runs_per_agent: int = 30,
        n_mc_agents: int = 8,
    ):
        self.initial_states = initial_states
        self.W = W
        self.H = H
        self.seeds_count = seeds_count
        self.inferred_params = inferred_params
        self.agents: list[SwarmAgent] = []

        # ── Create diverse MC agents from posterior samples ───────────────
        n_mc = min(n_mc_agents, len(posterior_samples))
        # Pick diverse samples: spread across posterior
        if len(posterior_samples) >= n_mc:
            step = len(posterior_samples) // n_mc
            selected = [posterior_samples[i * step] for i in range(n_mc)]
        else:
            selected = posterior_samples

        for i, params in enumerate(selected):
            self.agents.append(MonteCarloAgent(
                name=f"mc_{i}",
                params=params,
                n_runs=mc_runs_per_agent,
                weight=1.0,
            ))

        # ── Add non-MC agents ────────────────────────────────────────────
        # Weights reflect reliability. Statistical + contextual pooling are most
        # reliable as they're based on calibrated empirical data.
        self.agents.append(StatisticalAgent(weight=3.0))
        self.agents.append(SettlementTrajectoryAgent(weight=2.0))
        self.agents.append(TransitionAgent(weight=0.8))
        self.agents.append(HeuristicAgent(weight=0.8))
        self.agents.append(SpatialAgent(weight=0.5))

        # Contextual pooling: cross-seed empirical data is extremely valuable
        self._contextual_agent_weight = 3.0

        print(f"Swarm initialized: {len(self.agents)} agents "
              f"({n_mc} MC + 5 statistical/heuristic + contextual pooling pending)")

    def predict_all(
        self,
        counts: dict[int, np.ndarray],
        observations: dict[int, list],
        settlements_data: dict[int, list],
    ) -> dict[int, np.ndarray]:
        """Run all agents and combine predictions for all seeds."""
        predictions = {}

        # Create contextual pooling agent with actual observation data
        ctx_agent = ContextualPoolingAgent(
            all_counts=counts, all_initial_states=self.initial_states,
            W=self.W, H=self.H, weight=self._contextual_agent_weight,
        )
        all_agents = self.agents + [ctx_agent]

        for seed_idx in range(self.seeds_count):
            print(f"\n  Seed {seed_idx}: running {len(all_agents)} agents...")
            t0 = time.time()

            agent_preds = []
            agent_weights = []

            for agent in all_agents:
                try:
                    pred = agent.predict(
                        seed_idx=seed_idx,
                        initial_state=self.initial_states[seed_idx],
                        W=self.W, H=self.H,
                        counts=counts[seed_idx],
                        observations=observations[seed_idx],
                        settlements_data=settlements_data.get(seed_idx, []),
                        inferred_params=self.inferred_params,
                    )
                    # Validate
                    pred = np.maximum(pred, PROB_FLOOR)
                    pred /= pred.sum(axis=-1, keepdims=True)
                    agent_preds.append(pred)
                    agent_weights.append(agent.weight)
                except Exception as e:
                    print(f"    Agent {agent.name} failed: {e}")

            # Ensemble via weighted geometric mean
            combined = self._geometric_ensemble(agent_preds, agent_weights)

            cell_counts = counts[seed_idx][:, :, :NUM_CLASSES].astype(np.float64)
            n_obs = cell_counts.sum(axis=2)

            # ── KT blending with informative priors: trust data over ensemble ──
            # Use informative Dirichlet priors instead of uniform Jeffreys
            init_grid_kt = np.asarray(self.initial_states[seed_idx]["grid"], dtype=np.int64)
            settlements_kt = self.initial_states[seed_idx].get("settlements", [])
            kt_pred = _build_kt_with_priors(init_grid_kt, cell_counts, n_obs, settlements_kt, self.H, self.W)

            # More observations → trust KT more
            # n=1: 33%, n=3: 60%, n=5: 71%, n=10: 83%
            observed = n_obs > 0
            if observed.any():
                kt_weight = (n_obs[observed] / (n_obs[observed] + 2.0))[:, np.newaxis]
                combined[observed] = (
                    kt_weight * kt_pred[observed] +
                    (1.0 - kt_weight) * combined[observed]
                )

            # ── Unobserved cells: blend domain prior with ensemble ──
            # Pure ensemble → too uniform. Pure prior → ignores inferred params.
            # 50/50 blend: prior anchors, ensemble adds param-specific info.
            init_cls = _classify_grid(
                np.asarray(self.initial_states[seed_idx]["grid"], dtype=np.int64)
            )
            unobserved = ~observed
            if unobserved.any():
                # For unobserved cells, domain priors are much more reliable
                # than the ensemble geometric mean (which averages toward uniform).
                # Use 80% domain prior / 20% ensemble to retain some ensemble signal
                # from cross-seed contextual pooling agent.
                for cls_id in range(NUM_CLASSES):
                    cls_mask = unobserved & (init_cls == cls_id)
                    if cls_mask.any():
                        prior = get_domain_prior(cls_id)
                        combined[cls_mask] = 0.80 * prior + 0.20 * combined[cls_mask]

            # Final safety with class-conditional floors
            init_grid = np.asarray(self.initial_states[seed_idx]["grid"], dtype=np.int64)
            combined[init_grid == 5] = MOUNTAIN_PRIOR
            combined[init_grid == 10] = OCEAN_PRIOR
            settlements = self.initial_states[seed_idx].get("settlements", [])
            sett_dist = _settlement_distance(init_grid, settlements, self.W, self.H)
            cell_floors = _get_cell_floors_swarm(init_grid, sett_dist)
            combined = np.maximum(combined, cell_floors)
            combined /= combined.sum(axis=-1, keepdims=True)

            # Temperature scaling: slightly soften predictions (T > 1) for
            # unobserved/low-observation cells to reduce KL risk.
            # KL(p||q) punishes underestimation exponentially more than
            # overestimation, so being slightly too uncertain is safer.
            # For well-observed cells (n>=3), keep sharp. For others, soften.
            combined = self._temperature_scale(combined, n_obs, init_grid)

            # Re-apply floors after temperature scaling (sharpening can push
            # values below class-conditional floors, e.g. T=0.95: 0.01 → 0.0086)
            combined[init_grid == 5] = MOUNTAIN_PRIOR
            combined[init_grid == 10] = OCEAN_PRIOR
            combined = np.maximum(combined, cell_floors)
            combined /= combined.sum(axis=-1, keepdims=True)

            predictions[seed_idx] = combined
            elapsed = time.time() - t0
            obs_pct = 100 * (n_obs > 0).sum() / (self.W * self.H)
            print(f"    Done in {elapsed:.1f}s ({obs_pct:.0f}% observed, "
                  f"{len(agent_preds)} agents contributed)")

        return predictions

    def _geometric_ensemble(
        self,
        preds: list[np.ndarray],
        weights: list[float],
    ) -> np.ndarray:
        """
        Weighted geometric mean — optimal ensemble for KL divergence.

        P_ensemble = exp( Σ wᵢ log(Pᵢ) / Σ wᵢ )
        """
        if not preds:
            return np.full((self.H, self.W, NUM_CLASSES), 1.0 / NUM_CLASSES)

        total_weight = sum(weights)
        log_sum = np.zeros_like(preds[0])

        for pred, w in zip(preds, weights):
            log_sum += (w / total_weight) * np.log(pred + 1e-12)

        result = np.exp(log_sum)
        result = np.maximum(result, PROB_FLOOR)
        result /= result.sum(axis=-1, keepdims=True)
        return result

    def _temperature_scale(
        self,
        pred: np.ndarray,
        n_obs: np.ndarray,
        init_grid: np.ndarray,
    ) -> np.ndarray:
        """
        Apply cell-adaptive temperature scaling to minimize expected KL divergence.

        KL(p||q) is asymmetric: underestimating p_i (q_i << p_i) is catastrophic.
        Temperature T > 1 softens distributions (safer), T < 1 sharpens (riskier).

        Strategy:
        - Static terrain (mountain/ocean): T=1.0 (already near-certain, don't touch)
        - Well-observed cells (n >= 4): T=0.95 (slightly sharpen — we have good data)
        - Moderately observed (n = 2-3): T=1.0 (neutral)
        - Barely observed (n = 1): T=1.05 (slightly soften — uncertain)
        - Unobserved dynamic cells: T=1.10 (soften — domain priors are imperfect)
        """
        result = pred.copy()

        static_mask = (init_grid == 5) | (init_grid == 10)

        # Build per-cell temperature map
        T = np.ones_like(n_obs, dtype=np.float64)
        T[n_obs >= 4] = 0.95
        T[(n_obs >= 2) & (n_obs < 4)] = 1.0
        T[(n_obs == 1)] = 1.05
        T[n_obs == 0] = 1.10
        T[static_mask] = 1.0  # Don't touch static terrain

        # Apply temperature: q_scaled = softmax(log(q) / T)
        # = q^(1/T) / sum(q^(1/T))
        dynamic_mask = ~static_mask
        if dynamic_mask.any():
            log_pred = np.log(result[dynamic_mask] + 1e-12)
            T_cells = T[dynamic_mask, np.newaxis]
            scaled = np.exp(log_pred / T_cells)
            scaled = np.maximum(scaled, 1e-12)
            scaled /= scaled.sum(axis=1, keepdims=True)
            result[dynamic_mask] = scaled

        return result

    def _calibrate(self, pred: np.ndarray, n_obs: np.ndarray) -> np.ndarray:
        """No calibration needed — ensemble + informative priors handle uncertainty."""
        return pred


# ── Helper functions ─────────────────────────────────────────────────────────


def _classify(code: int) -> int:
    return TERRAIN_TO_CLASS.get(code, 0)


def _classify_grid(grid: np.ndarray) -> np.ndarray:
    out = np.zeros_like(grid, dtype=np.int32)
    for code, cls in TERRAIN_TO_CLASS.items():
        out[grid == code] = cls
    return out



# _mountain_prior, _ocean_prior, _domain_prior removed — now imported from priors.py



def _coastal_mask(grid: np.ndarray) -> np.ndarray:
    H, W = grid.shape
    ocean = grid == 10
    coastal = np.zeros((H, W), dtype=bool)
    padded = np.pad(ocean, 1, constant_values=True)
    for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        coastal |= padded[1 + dy:H + 1 + dy, 1 + dx:W + 1 + dx]
    coastal &= ~ocean & (grid != 5)
    return coastal


def _food_map(grid: np.ndarray) -> np.ndarray:
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


def _settlement_distance(grid: np.ndarray, settlements: list,
                          W: int, H: int) -> np.ndarray:
    dist = np.full((H, W), 999.0)
    yy, xx = np.mgrid[0:H, 0:W]
    for s in settlements:
        sx, sy = s.get("x", -1), s.get("y", -1)
        if 0 <= sx < W and 0 <= sy < H:
            d = np.abs(xx - sx) + np.abs(yy - sy)
            dist = np.minimum(dist, d)
    return dist


def _get_cell_floors_swarm(init_grid: np.ndarray, sett_dist: np.ndarray) -> np.ndarray:
    """Compute per-cell, per-class probability floors for swarm predictions."""
    H, W = init_grid.shape
    floors = np.full((H, W, NUM_CLASSES), PROB_FLOOR, dtype=np.float64)

    mountain_mask = (init_grid == 5)
    if mountain_mask.any():
        floors[mountain_mask, :5] = STATIC_FLOOR

    ocean_mask = (init_grid == 10)
    if ocean_mask.any():
        floors[ocean_mask, 1:] = STATIC_FLOOR

    init_cls = _classify_grid(init_grid)

    remote_empty = (init_cls == 0) & (sett_dist > 8) & ~ocean_mask
    if remote_empty.any():
        floors[remote_empty, 1] = REMOTE_FLOOR
        floors[remote_empty, 2] = REMOTE_FLOOR
        floors[remote_empty, 3] = REMOTE_FLOOR

    remote_forest = (init_cls == 4) & (sett_dist > 8)
    if remote_forest.any():
        floors[remote_forest, 1] = REMOTE_FLOOR
        floors[remote_forest, 2] = REMOTE_FLOOR
        floors[remote_forest, 3] = REMOTE_FLOOR

    return floors


def _neighbor_settlements(grid: np.ndarray, H: int, W: int) -> np.ndarray:
    """Count settlement/port neighbors within radius 2."""
    sett_mask = np.isin(grid, [1, 2]).astype(np.int32)
    n_sett = np.zeros((H, W), dtype=np.int32)
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
            n_sett += shifted
    return n_sett


def _get_cell_prior(init_cls: int, sett_dist: float, food: float,
                    coastal: bool, neighbor_sett: int) -> np.ndarray:
    """Return informative Dirichlet prior (6,) for a cell based on context."""
    base = get_domain_prior(init_cls).copy()

    if coastal and init_cls in (1, 2):
        base[2] += 0.08
        base[0] -= 0.04

    if food >= 2 and init_cls == 1:
        base[1] += 0.10
        base[0] -= 0.05
        base[3] -= 0.03

    if sett_dist > 6 and init_cls == 0:
        base = np.array([0.92, 0.01, 0.01, 0.01, 0.04, 0.01])

    if sett_dist > 6 and init_cls == 4:
        base = np.array([0.04, 0.01, 0.01, 0.01, 0.90, 0.01])

    if sett_dist <= 3 and init_cls == 0:
        base[1] += 0.06
        base[3] += 0.03
        base[0] -= 0.06

    if neighbor_sett >= 2 and init_cls == 0 and sett_dist <= 4:
        base[1] += 0.04
        base[0] -= 0.03

    base = np.maximum(base, PROB_FLOOR)
    base /= base.sum()
    return base


def _get_prior_strength(init_cls: int, sett_dist: float, n_obs: float) -> float:
    """Return prior strength that decays with observations."""
    if init_cls == 5:
        base = 4.0
    elif init_cls == 0 and sett_dist > 6:
        base = 3.0
    elif init_cls in (1, 2):
        base = 1.5
    elif sett_dist <= 3:
        base = 2.0
    else:
        base = 2.5

    if n_obs > 0:
        base = base / (1.0 + n_obs / base)

    return base


def _build_kt_with_priors(init_grid: np.ndarray, cell_counts: np.ndarray,
                           n_obs: np.ndarray, settlements: list,
                           H: int, W: int) -> np.ndarray:
    """Build KT estimate with informative Dirichlet priors for all cells."""
    init_cls = _classify_grid(init_grid)
    coastal = _coastal_mask(init_grid)
    food_map = _food_map(init_grid)
    sett_dist = _settlement_distance(init_grid, settlements, W, H)
    nsett = _neighbor_settlements(init_grid, H, W)

    prior_grid = np.zeros((H, W, NUM_CLASSES), dtype=np.float64)
    strength_grid = np.zeros((H, W), dtype=np.float64)
    for y in range(H):
        for x in range(W):
            ic = int(init_cls[y, x])
            sd = float(sett_dist[y, x])
            fd = float(food_map[y, x])
            cs = bool(coastal[y, x])
            ns = int(nsett[y, x])
            prior_grid[y, x] = _get_cell_prior(ic, sd, fd, cs, ns)
            strength_grid[y, x] = _get_prior_strength(ic, sd, float(n_obs[y, x]))

    strength_3d = strength_grid[:, :, np.newaxis]
    kt_pred = (cell_counts + prior_grid * strength_3d) / (n_obs[:, :, np.newaxis] + strength_3d)
    return kt_pred
