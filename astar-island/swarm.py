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

NUM_CLASSES = 6
PROB_FLOOR = 0.01
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
        pred = np.full((H, W, NUM_CLASSES), 1.0 / NUM_CLASSES)
        init_grid = np.asarray(initial_state["grid"], dtype=np.int64)
        cell_counts = counts[:, :, :NUM_CLASSES].astype(np.float64)
        n_obs = cell_counts.sum(axis=2)

        # KT estimator for observed cells
        kt_alpha = 0.5
        observed = n_obs > 0
        kt_denom = n_obs[observed] + NUM_CLASSES * kt_alpha
        pred[observed] = (cell_counts[observed] + kt_alpha) / kt_denom[:, np.newaxis]

        # Hard constraints for static terrain
        pred[init_grid == 5] = _mountain_prior()
        pred[init_grid == 10] = _ocean_prior()

        # Domain priors for unobserved non-static cells
        unobs = ~observed & (init_grid != 5) & (init_grid != 10)
        if unobs.any():
            for y, x in zip(*np.where(unobs)):
                ic = _classify(int(init_grid[y, x]))
                pred[y, x] = _domain_prior(ic)

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
        pred[init_grid == 5] = _mountain_prior()
        pred[init_grid == 10] = _ocean_prior()

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
        cell_counts = counts[:, :, :NUM_CLASSES].astype(np.float64)
        n_obs = cell_counts.sum(axis=2)
        observed = n_obs > 0

        # Start from KT for observed, domain prior for unobserved
        pred = np.full((H, W, NUM_CLASSES), 1.0 / NUM_CLASSES)
        kt_alpha = 0.5
        kt_denom = n_obs[observed] + NUM_CLASSES * kt_alpha
        pred[observed] = (cell_counts[observed] + kt_alpha) / kt_denom[:, np.newaxis]

        init_cls = _classify_grid(init_grid)
        unobs = ~observed & (init_grid != 5) & (init_grid != 10)
        for y, x in zip(*np.where(unobs)):
            pred[y, x] = _domain_prior(int(init_cls[y, x]))

        # Hard constraints
        pred[init_grid == 5] = _mountain_prior()
        pred[init_grid == 10] = _ocean_prior()

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
        result[init_grid == 5] = _mountain_prior()
        result[init_grid == 10] = _ocean_prior()

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
                    pred[y, x] = _mountain_prior()
                    continue
                if raw == 10:
                    pred[y, x] = _ocean_prior()
                    continue

                dist = _domain_prior(ic).copy()
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
        cell_counts = counts[:, :, :NUM_CLASSES].astype(np.float64)
        n_obs = cell_counts.sum(axis=2)

        # Start from KT for observed, domain prior for unobserved
        pred = np.full((H, W, NUM_CLASSES), 1.0 / NUM_CLASSES)
        observed = n_obs > 0
        kt_alpha = 0.5
        kt_denom = n_obs[observed] + NUM_CLASSES * kt_alpha
        pred[observed] = (cell_counts[observed] + kt_alpha) / kt_denom[:, np.newaxis]

        init_cls = _classify_grid(init_grid)
        unobs = ~observed & (init_grid != 5) & (init_grid != 10)
        for y, x in zip(*np.where(unobs)):
            pred[y, x] = _domain_prior(int(init_cls[y, x]))

        # Hard constraints
        pred[init_grid == 5] = _mountain_prior()
        pred[init_grid == 10] = _ocean_prior()

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
                    pred[y, x] = _mountain_prior()
                    continue
                if raw == 10:
                    pred[y, x] = _ocean_prior()
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
                        pred[y, x] = _domain_prior(ic)

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
        self.agents.append(StatisticalAgent(weight=2.5))
        self.agents.append(SettlementTrajectoryAgent(weight=1.5))
        self.agents.append(TransitionAgent(weight=0.8))
        self.agents.append(HeuristicAgent(weight=1.0))
        self.agents.append(SpatialAgent(weight=0.6))

        # Add contextual pooling agent (needs cross-seed data, added later in predict_all)
        self._contextual_agent_weight = 2.0

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

            # Light calibration for few-observation cells only
            cell_counts = counts[seed_idx][:, :, :NUM_CLASSES].astype(np.float64)
            n_obs = cell_counts.sum(axis=2)
            combined = self._calibrate(combined, n_obs)

            # Final safety
            init_grid = np.asarray(self.initial_states[seed_idx]["grid"], dtype=np.int64)
            combined[init_grid == 5] = _mountain_prior()
            combined[init_grid == 10] = _ocean_prior()
            combined = np.maximum(combined, PROB_FLOOR)
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

    def _calibrate(
        self,
        pred: np.ndarray,
        n_obs: np.ndarray,
    ) -> np.ndarray:
        """Light calibration — do NOT shrink unobserved toward uniform.

        The domain priors are already well-calibrated. Shrinking toward uniform
        destroys the strong Empty-stays-Empty signal and causes catastrophic KL.
        Only apply mild shrinkage for cells with very few (1-2) observations.
        """
        uniform = np.full(NUM_CLASSES, 1.0 / NUM_CLASSES)
        result = pred.copy()

        # Unobserved: NO shrinkage. Domain priors from ensemble are our best guess.

        # Few observations (1-2): very mild shrinkage
        few = (n_obs > 0) & (n_obs <= 2)
        if few.any():
            n_few = n_obs[few]
            alpha = 2.0
            w_obs = (n_few / (n_few + alpha))[:, np.newaxis]
            result[few] = w_obs * pred[few] + (1 - w_obs) * uniform

        return result


# ── Helper functions ─────────────────────────────────────────────────────────


def _classify(code: int) -> int:
    return TERRAIN_TO_CLASS.get(code, 0)


def _classify_grid(grid: np.ndarray) -> np.ndarray:
    out = np.zeros_like(grid, dtype=np.int32)
    for code, cls in TERRAIN_TO_CLASS.items():
        out[grid == code] = cls
    return out


def _mountain_prior() -> np.ndarray:
    p = np.full(NUM_CLASSES, PROB_FLOOR)
    p[5] = 1.0 - 5 * PROB_FLOOR
    return p


def _ocean_prior() -> np.ndarray:
    p = np.full(NUM_CLASSES, PROB_FLOOR)
    p[0] = 1.0 - 5 * PROB_FLOOR
    return p


def _domain_prior(init_cls: int) -> np.ndarray:
    # Calibrated from Round 1 ground truth analysis
    priors = {
        0: np.array([0.82, 0.13, 0.012, 0.010, 0.028, 0.01]),  # Empty: 13% become settlement!
        1: np.array([0.37, 0.41, 0.008, 0.031, 0.181, 0.01]),  # Settlement: only 41% survive, 37% vanish
        2: np.array([0.36, 0.12, 0.319, 0.021, 0.176, 0.01]),  # Port: 32% survive, 36% vanish
        3: np.array([0.17, 0.17, 0.17, 0.17, 0.17, 0.15]),     # Ruin: near-uniform (rare terrain)
        4: np.array([0.07, 0.16, 0.014, 0.012, 0.744, 0.01]),  # Forest: 74% stable, 16% become settlement
        5: np.array([0.005, 0.005, 0.005, 0.005, 0.005, 0.975]), # Mountain: never changes
    }
    p = priors.get(init_cls, priors[0]).copy()
    p = np.maximum(p, PROB_FLOOR)
    p /= p.sum()
    return p


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
