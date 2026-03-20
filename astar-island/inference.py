#!/usr/bin/env python3
"""
Bayesian parameter inference engine for the Astar Island Norse civilization simulator.

Infers hidden simulation parameters from observed terrain transitions and settlement
statistics, then samples from an approximate posterior for ensemble predictions.

Dependencies: numpy, scipy
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import beta as beta_dist

NUM_CLASSES = 6
TERRAIN_TO_CLASS = {10: 0, 11: 0, 0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5}

# Parameter names and their semantics
PARAM_NAMES = [
    "winter_severity",
    "faction_aggression",
    "trade_activity",
    "forest_growth_rate",
    "expansion_rate",
    "raid_range",
    "food_per_forest",
    "port_development_threshold",
    "ruin_reclaim_rate",
]

# Default priors: (alpha, beta) for Beta distribution — weakly informative
DEFAULT_PRIORS = {
    "winter_severity": (2.0, 3.0),         # Slight bias toward mild
    "faction_aggression": (2.0, 3.0),       # Slight bias toward peaceful
    "trade_activity": (2.0, 2.0),           # Uniform-ish
    "forest_growth_rate": (2.0, 3.0),       # Moderate growth expected
    "expansion_rate": (2.0, 3.0),           # Moderate expansion
    "raid_range": (2.0, 2.0),               # Uniform
    "food_per_forest": (3.0, 2.0),          # Forests usually give food
    "port_development_threshold": (2.0, 2.0),  # Uniform
    "ruin_reclaim_rate": (2.0, 3.0),        # Moderate reclaim
}


def _classify_grid(grid: np.ndarray) -> np.ndarray:
    """Map raw terrain codes to 0-5 classes."""
    out = np.zeros_like(grid, dtype=np.int32)
    for code, cls in TERRAIN_TO_CLASS.items():
        out[grid == code] = cls
    return out


def _compute_food_map(grid: np.ndarray) -> np.ndarray:
    """Count adjacent forest cells per cell."""
    H, W = grid.shape
    forest = (grid == 4).astype(np.float32)
    food = np.zeros((H, W), dtype=np.float32)
    for dy in range(-1, 2):
        for dx in range(-1, 2):
            if dy == 0 and dx == 0:
                continue
            sy = slice(max(0, -dy), min(H, H - dy))
            sx = slice(max(0, -dx), min(W, W - dx))
            ty = slice(max(0, dy), min(H, H + dy))
            tx = slice(max(0, dx), min(W, W + dx))
            shifted = np.zeros_like(forest)
            shifted[ty, tx] = forest[sy, sx]
            food += shifted
    return food


def _compute_coastal(grid: np.ndarray) -> np.ndarray:
    """True for land cells adjacent to ocean."""
    H, W = grid.shape
    ocean = (grid == 10)
    coastal = np.zeros((H, W), dtype=bool)
    padded = np.pad(ocean, 1, constant_values=True)
    for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        coastal |= padded[1 + dy:H + 1 + dy, 1 + dx:W + 1 + dx]
    coastal &= ~ocean
    coastal &= (grid != 5)
    return coastal


class ParameterInference:
    """
    Infer hidden simulation parameters from initial states and observations.

    Uses three complementary methods:
    1. Terrain transition statistics (initial -> observed)
    2. Settlement metadata statistics (population, food, wealth, defense)
    3. Spatial pattern analysis (clustering, distances)

    All methods pool data across seeds (they share the same hidden params).
    """

    def __init__(
        self,
        initial_states: list,
        observations: dict,
        counts: dict,
        settlements_data: Optional[dict] = None,
    ):
        """
        Args:
            initial_states: list of {grid: 2D, settlements: [...]} per seed
            observations: dict seed_idx -> 2D list (None for unobserved cells)
            counts: dict seed_idx -> np.array (H, W, 16) of observation counts
            settlements_data: optional dict seed_idx -> list of settlement snapshots
        """
        self.initial_states = initial_states
        self.observations = observations
        self.counts = counts
        self.settlements_data = settlements_data or {}
        self.n_seeds = len(initial_states)

        # Compute transition statistics once
        self._trans_counts: Optional[np.ndarray] = None
        self._trans_probs: Optional[np.ndarray] = None
        self._total_obs: float = 0.0
        self._settlement_stats: Optional[dict] = None
        self._spatial_stats: Optional[dict] = None

        self._compute_all_statistics()

    # ── Statistics computation ────────────────────────────────────────────

    def _compute_all_statistics(self) -> None:
        """Compute all statistics needed for inference."""
        self._compute_transition_stats()
        self._compute_settlement_stats()
        self._compute_spatial_stats()

    def _compute_transition_stats(self) -> None:
        """Aggregate terrain transition counts across all seeds."""
        trans = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.float64)

        for seed_idx, state in enumerate(self.initial_states):
            if seed_idx not in self.counts:
                continue
            init_grid = np.asarray(state["grid"], dtype=np.int64)
            seed_counts = self.counts[seed_idx]

            for init_cls in range(NUM_CLASSES):
                mask = np.zeros_like(init_grid, dtype=bool)
                for code, cls in TERRAIN_TO_CLASS.items():
                    if cls == init_cls:
                        mask |= (init_grid == code)
                if mask.any():
                    trans[init_cls] += seed_counts[mask][:, :NUM_CLASSES].sum(axis=0)

        self._trans_counts = trans
        self._total_obs = trans.sum()

        # Normalize (with small smoothing to avoid division by zero)
        row_sums = trans.sum(axis=1, keepdims=True)
        row_sums = np.maximum(row_sums, 1e-10)
        self._trans_probs = trans / row_sums

    def _compute_settlement_stats(self) -> None:
        """Extract statistics from settlement snapshots."""
        stats: Dict[str, Any] = {
            "total_settlements": 0,
            "total_alive": 0,
            "total_dead": 0,
            "total_ports": 0,
            "avg_food": [],
            "avg_wealth": [],
            "avg_population": [],
            "avg_defense": [],
            "food_values": [],
            "wealth_values": [],
        }

        # From initial states
        for state in self.initial_states:
            for s in state.get("settlements", []):
                stats["total_settlements"] += 1

        # From observation snapshots
        for seed_idx, snapshots in self.settlements_data.items():
            for snapshot in snapshots:
                for s in snapshot.get("settlements", []):
                    alive = s.get("alive", True)
                    has_port = s.get("has_port", False)
                    food = s.get("food", 0.5)
                    wealth = s.get("wealth", 0.0)
                    pop = s.get("population", 1.0)
                    defense = s.get("defense", 0.5)

                    if alive:
                        stats["total_alive"] += 1
                    else:
                        stats["total_dead"] += 1
                    if has_port:
                        stats["total_ports"] += 1

                    stats["food_values"].append(food)
                    stats["wealth_values"].append(wealth)
                    stats["avg_food"].append(food)
                    stats["avg_wealth"].append(wealth)
                    stats["avg_population"].append(pop)
                    stats["avg_defense"].append(defense)

        # Compute averages
        for key in ["avg_food", "avg_wealth", "avg_population", "avg_defense"]:
            vals = stats[key]
            stats[key] = float(np.mean(vals)) if vals else 0.5

        self._settlement_stats = stats

    def _compute_spatial_stats(self) -> None:
        """Analyze spatial patterns: clustering, distances, borders."""
        stats: Dict[str, Any] = {
            "settlement_cluster_density": [],
            "ruin_settlement_distances": [],
            "forest_border_growth": [],
            "settlement_spread": [],
        }

        for seed_idx, state in enumerate(self.initial_states):
            if seed_idx not in self.counts:
                continue

            init_grid = np.asarray(state["grid"], dtype=np.int64)
            seed_counts = self.counts[seed_idx]
            init_cls_grid = _classify_grid(init_grid)
            H, W = init_grid.shape

            # Observed final class (mode of observations)
            obs_totals = seed_counts[:, :, :NUM_CLASSES].sum(axis=2)
            has_obs = obs_totals > 0

            if not has_obs.any():
                continue

            # Get most likely final class for observed cells
            final_cls = np.argmax(seed_counts[:, :, :NUM_CLASSES], axis=2)

            # Settlement positions in final state
            sett_positions = list(zip(*np.where(
                has_obs & ((final_cls == 1) | (final_cls == 2))
            )))
            ruin_positions = list(zip(*np.where(
                has_obs & (final_cls == 3)
            )))

            # Settlement clustering: average nearest-neighbor distance
            if len(sett_positions) >= 2:
                sett_arr = np.array(sett_positions)
                min_dists = []
                for i, pos in enumerate(sett_arr):
                    dists = np.abs(sett_arr - pos).sum(axis=1)
                    dists[i] = 999
                    min_dists.append(dists.min())
                stats["settlement_cluster_density"].append(float(np.mean(min_dists)))

            # Settlement spread: max distance between any two settlements
            if len(sett_positions) >= 2:
                sett_arr = np.array(sett_positions)
                max_dist = 0
                for pos in sett_arr[:20]:  # Sample to avoid O(n^2)
                    d = np.abs(sett_arr - pos).sum(axis=1).max()
                    max_dist = max(max_dist, d)
                stats["settlement_spread"].append(float(max_dist))

            # Ruin-to-nearest-settlement distances
            if ruin_positions and sett_positions:
                sett_arr = np.array(sett_positions)
                for ry, rx in ruin_positions[:50]:
                    d = np.abs(sett_arr - [ry, rx]).sum(axis=1).min()
                    stats["ruin_settlement_distances"].append(float(d))

            # Forest border growth: cells that were empty near forest, now forest
            food_map = _compute_food_map(init_grid)
            empty_mask = init_cls_grid == 0
            near_forest = food_map > 0
            grew_forest = (final_cls == 4) & has_obs & empty_mask & near_forest
            could_grow = empty_mask & near_forest & has_obs
            if could_grow.sum() > 0:
                rate = grew_forest.sum() / could_grow.sum()
                stats["forest_border_growth"].append(float(rate))

        self._spatial_stats = stats

    # ── Transition rate extractors ────────────────────────────────────────

    def _safe_rate(self, from_cls: int, to_cls: int) -> Tuple[float, float]:
        """
        Get transition rate and confidence.
        Returns (rate, n_observations).
        """
        if self._trans_counts is None:
            return 0.0, 0.0
        n_from = self._trans_counts[from_cls].sum()
        if n_from < 1:
            return 0.0, 0.0
        rate = self._trans_counts[from_cls, to_cls] / n_from
        return float(rate), float(n_from)

    def _survival_rate(self, cls: int) -> Tuple[float, float]:
        """Get self-transition (survival) rate for a class."""
        return self._safe_rate(cls, cls)

    # ── Method 1: Transition-based inference ──────────────────────────────

    def _infer_from_transitions(self) -> Dict[str, Tuple[float, float]]:
        """
        Infer parameters from terrain transition rates.
        Returns dict of param_name -> (estimate, confidence).
        Confidence is 0-1 based on observation count.
        """
        results: Dict[str, Tuple[float, float]] = {}

        if self._total_obs < 20:
            return results

        # Settlement -> Ruin rate: winter_severity + faction_aggression
        sett_to_ruin, n_sett = self._safe_rate(1, 3)
        sett_survival, _ = self._survival_rate(1)
        conf_sett = float(np.clip(n_sett / 200, 0, 1))

        # Decompose: high ruin rate = both winter and aggression contribute
        # Use port survival as discriminator: trade protects from winter but not raids
        port_to_ruin, n_port = self._safe_rate(2, 3)
        port_survival, _ = self._survival_rate(2)

        if n_sett > 10:
            # Death rate of settlements
            death_rate = sett_to_ruin
            # If ports die less than settlements, raids target settlements more
            # If ports die equally, it's more winter (affects everyone)
            if n_port > 5 and port_to_ruin > 0:
                raid_ratio = death_rate / max(port_to_ruin, 0.001)
                # raid_ratio > 1 means settlements die more than ports -> more raids
                aggression_signal = np.clip((raid_ratio - 0.5) * 0.5, 0, 1)
                winter_signal = np.clip(port_to_ruin * 3.0, 0, 1)
            else:
                # Can't distinguish, split evenly
                aggression_signal = np.clip(death_rate * 1.5, 0, 1)
                winter_signal = np.clip(death_rate * 1.2, 0, 1)

            results["faction_aggression"] = (float(aggression_signal), conf_sett)
            results["winter_severity"] = (float(winter_signal), conf_sett)

            # raid_range correlates with how far apart ruins are from surviving settlements
            results["raid_range"] = (float(np.clip(death_rate * 2.0, 0, 1)), conf_sett * 0.5)

        # Settlement -> Port rate: trade_activity + port_development_threshold
        sett_to_port, _ = self._safe_rate(1, 2)
        conf_port = float(np.clip(n_sett / 200, 0, 1))

        if n_sett > 10:
            trade_signal = np.clip(sett_to_port * 5.0 + port_survival * 0.3, 0, 1)
            results["trade_activity"] = (float(trade_signal), conf_port)

            # Port development threshold (inverted: 0=hard, 1=easy)
            # More ports = easier threshold
            port_freq = sett_to_port + (self._trans_counts[2, 2] / max(n_sett, 1))
            results["port_development_threshold"] = (
                float(np.clip(port_freq * 4.0, 0, 1)), conf_port * 0.7
            )

        # Empty -> Forest: forest_growth_rate
        empty_to_forest, n_empty = self._safe_rate(0, 4)
        conf_forest = float(np.clip(n_empty / 300, 0, 1))

        if n_empty > 20:
            # Also check Ruin -> Forest as confirmation
            ruin_to_forest, n_ruin = self._safe_rate(3, 4)
            forest_signal = empty_to_forest  # Use raw transition rate directly
            if n_ruin > 5:
                # Average with ruin→forest, weight toward higher-confidence source
                if n_empty > n_ruin:
                    forest_signal = forest_signal * 0.7 + ruin_to_forest * 0.3
                else:
                    forest_signal = forest_signal * 0.3 + ruin_to_forest * 0.7
            results["forest_growth_rate"] = (float(np.clip(forest_signal, 0, 1)), conf_forest)

        # Empty -> Settlement: expansion_rate
        empty_to_sett, _ = self._safe_rate(0, 1)
        if n_empty > 20:
            results["expansion_rate"] = (
                float(np.clip(empty_to_sett * 1.0, 0, 1)), conf_forest
            )

        # Ruin -> Settlement or Empty: ruin_reclaim_rate
        ruin_to_sett, n_ruin = self._safe_rate(3, 1)
        ruin_to_empty, _ = self._safe_rate(3, 0)
        ruin_to_forest_r, _ = self._safe_rate(3, 4)
        conf_ruin = float(np.clip(n_ruin / 100, 0, 1))

        if n_ruin > 5:
            reclaim = ruin_to_sett + ruin_to_empty + ruin_to_forest_r
            results["ruin_reclaim_rate"] = (float(np.clip(reclaim * 2.0, 0, 1)), conf_ruin)

        # food_per_forest: infer from settlement survival near forests
        # Settlements with high food adjacency that survive -> high food_per_forest
        # This is hard to get from transitions alone, use settlement stats instead
        # Provide a weak estimate based on overall settlement survival
        if n_sett > 10:
            results["food_per_forest"] = (
                float(np.clip(sett_survival * 0.8, 0, 1)), conf_sett * 0.3
            )

        return results

    # ── Method 2: Settlement statistics inference ─────────────────────────

    def _infer_from_settlements(self) -> Dict[str, Tuple[float, float]]:
        """
        Infer parameters from settlement metadata (food, wealth, population, defense).
        Returns dict of param_name -> (estimate, confidence).
        """
        results: Dict[str, Tuple[float, float]] = {}
        stats = self._settlement_stats
        if stats is None:
            return results

        n_total = stats["total_alive"] + stats["total_dead"]
        if n_total < 5:
            return results

        conf = float(np.clip(n_total / 100, 0, 1))

        # Average food: inversely correlates with winter_severity
        avg_food = stats["avg_food"]
        # Food typically in range [0, 2+]. Low food = harsh winters
        winter_from_food = np.clip(1.0 - avg_food * 0.5, 0, 1)
        results["winter_severity"] = (float(winter_from_food), conf * 0.6)

        # food_per_forest: food level correlates directly
        results["food_per_forest"] = (float(np.clip(avg_food * 0.25, 0, 1)), conf * 0.5)

        # Average wealth: correlates with trade_activity
        avg_wealth = stats["avg_wealth"]
        results["trade_activity"] = (float(np.clip(avg_wealth * 0.3, 0, 1)), conf * 0.5)

        # Death ratio: faction_aggression + winter_severity
        if n_total > 0:
            death_ratio = stats["total_dead"] / n_total
            results["faction_aggression"] = (float(np.clip(death_ratio * 1.5, 0, 1)), conf * 0.5)

        # Port frequency
        if stats["total_alive"] > 0:
            port_ratio = stats["total_ports"] / max(stats["total_alive"], 1)
            results["port_development_threshold"] = (
                float(np.clip(port_ratio * 3.0, 0, 1)), conf * 0.4
            )

        return results

    # ── Method 3: Spatial pattern inference ───────────────────────────────

    def _infer_from_spatial(self) -> Dict[str, Tuple[float, float]]:
        """
        Infer parameters from spatial patterns.
        Returns dict of param_name -> (estimate, confidence).
        """
        results: Dict[str, Tuple[float, float]] = {}
        stats = self._spatial_stats
        if stats is None:
            return results

        # Settlement clustering -> expansion_rate
        # Dense clusters (low avg NND) = high expansion
        if stats["settlement_cluster_density"]:
            avg_nnd = np.mean(stats["settlement_cluster_density"])
            # NND typically 1-15. Low = clustered = high expansion
            expansion = np.clip(1.0 - avg_nnd / 12.0, 0, 1)
            conf = min(len(stats["settlement_cluster_density"]) / 3, 1.0)
            results["expansion_rate"] = (float(expansion), float(conf) * 0.1)

        # Settlement spread -> expansion_rate (confirmation)
        if stats["settlement_spread"]:
            avg_spread = np.mean(stats["settlement_spread"])
            # High spread = high expansion
            expansion_spread = np.clip(avg_spread / 30.0, 0, 1)
            conf = min(len(stats["settlement_spread"]) / 3, 1.0)
            if "expansion_rate" in results:
                # Average with transition-based estimate
                old_val, old_conf = results["expansion_rate"]
                new_val = (old_val * old_conf + expansion_spread * conf * 0.1) / (old_conf + conf * 0.1)
                results["expansion_rate"] = (float(new_val), max(old_conf, float(conf) * 0.1))

        # Ruin-settlement distances -> raid_range
        if stats["ruin_settlement_distances"]:
            avg_dist = np.mean(stats["ruin_settlement_distances"])
            # High distance between ruins and settlements = long raid range
            raid_signal = np.clip(avg_dist / 10.0, 0, 1)
            conf = min(len(stats["ruin_settlement_distances"]) / 20, 1.0)
            results["raid_range"] = (float(raid_signal), float(conf) * 0.5)

        # Forest border growth rate
        if stats["forest_border_growth"]:
            avg_growth = np.mean(stats["forest_border_growth"])
            conf = min(len(stats["forest_border_growth"]) / 3, 1.0)
            results["forest_growth_rate"] = (float(np.clip(avg_growth * 1.0, 0, 1)), float(conf) * 0.6)

        return results

    # ── Combine estimates ─────────────────────────────────────────────────

    def _combine_estimates(
        self,
        *estimate_dicts: Dict[str, Tuple[float, float]],
    ) -> Dict[str, Tuple[float, float, float]]:
        """
        Combine multiple (value, confidence) estimates per parameter using
        confidence-weighted averaging.

        Returns dict of param_name -> (combined_value, combined_confidence, uncertainty).
        """
        combined: Dict[str, Tuple[float, float, float]] = {}

        for param in PARAM_NAMES:
            values = []
            weights = []

            for estimates in estimate_dicts:
                if param in estimates:
                    val, conf = estimates[param]
                    values.append(val)
                    weights.append(conf)

            if not values:
                # Use prior mean
                a, b = DEFAULT_PRIORS[param]
                prior_mean = a / (a + b)
                combined[param] = (prior_mean, 0.0, 0.3)
                continue

            values = np.array(values)
            weights = np.array(weights)
            total_weight = weights.sum()

            if total_weight < 1e-10:
                a, b = DEFAULT_PRIORS[param]
                combined[param] = (a / (a + b), 0.0, 0.3)
                continue

            # Weighted average
            estimate = float(np.average(values, weights=weights))
            confidence = float(np.clip(total_weight / len(estimate_dicts), 0, 1))

            # Uncertainty: weighted std + base uncertainty from low confidence
            if len(values) > 1:
                weighted_var = np.average((values - estimate) ** 2, weights=weights)
                uncertainty = float(np.sqrt(weighted_var) + 0.05 * (1 - confidence))
            else:
                uncertainty = float(0.15 + 0.15 * (1 - confidence))

            # Blend with prior based on confidence
            a, b = DEFAULT_PRIORS[param]
            prior_mean = a / (a + b)
            blended = confidence * estimate + (1 - confidence) * prior_mean
            blended = float(np.clip(blended, 0.01, 0.99))

            combined[param] = (blended, confidence, uncertainty)

        return combined

    # ── Rescaling to simulator scale ─────────────────────────────────────

    def _rescale_to_simulator(self, params: dict) -> dict:
        """
        Map inferred 0-1 values to the scales expected by simulator.py's DEFAULT_PARAMS.

        Most parameters are already 0-1, but raid_range expects 3-10 (manhattan distance).
        """
        rescaled = dict(params)

        # raid_range: inference outputs 0-1, simulator expects 3-10
        rescaled["raid_range"] = 3.0 + rescaled["raid_range"] * 7.0

        return rescaled

    # ── Public API ────────────────────────────────────────────────────────

    def infer(self) -> dict:
        """
        Infer MAP (maximum a posteriori) estimates of hidden parameters.

        Returns dict of parameter values on the same scale as simulator.py DEFAULT_PARAMS.
        """
        trans_est = self._infer_from_transitions()
        sett_est = self._infer_from_settlements()
        spatial_est = self._infer_from_spatial()

        combined = self._combine_estimates(trans_est, sett_est, spatial_est)

        raw = {param: val for param, (val, _, _) in combined.items()}
        return self._rescale_to_simulator(raw)

    def infer_with_uncertainty(self) -> Dict[str, Tuple[float, float, float]]:
        """
        Infer parameters with uncertainty estimates.

        Returns dict of param_name -> (value, confidence, uncertainty).
        """
        trans_est = self._infer_from_transitions()
        sett_est = self._infer_from_settlements()
        spatial_est = self._infer_from_spatial()

        return self._combine_estimates(trans_est, sett_est, spatial_est)

    def infer_posterior(self, n_samples: int = 100) -> list[dict]:
        """
        Sample from approximate posterior distribution over parameters.

        Uses a Beta-distribution posterior centered on MAP estimates with
        variance determined by observation uncertainty. Samples are filtered
        by approximate likelihood (ABC-style rejection).

        Returns list of parameter dicts for ensemble prediction.
        """
        combined = self.infer_with_uncertainty()
        map_estimate = {p: v for p, (v, _, _) in combined.items()}

        # Compute observed transition summary for ABC rejection
        obs_summary = self._compute_summary_stats()

        # Generate candidate samples from posterior
        candidates: list[dict] = []
        n_generate = n_samples * 5  # Oversample for rejection

        for _ in range(n_generate):
            sample = {}
            for param in PARAM_NAMES:
                val, conf, unc = combined[param]

                # Shape the Beta distribution:
                # High confidence -> tight distribution around MAP
                # Low confidence -> broad distribution
                concentration = 2.0 + conf * 20.0  # 2 (vague) to 22 (moderately tight)

                # Ensure val is in (0, 1) for Beta parametrization
                val_safe = np.clip(val, 0.02, 0.98)
                alpha = val_safe * concentration
                beta_param = (1 - val_safe) * concentration

                # Add extra spread from uncertainty
                spread_factor = max(1.0 - unc, 0.3)
                alpha *= spread_factor
                beta_param *= spread_factor

                # Ensure valid Beta params
                alpha = max(alpha, 0.5)
                beta_param = max(beta_param, 0.5)

                sample[param] = float(np.clip(
                    beta_dist.rvs(alpha, beta_param), 0.01, 0.99
                ))

            candidates.append(sample)

        # ABC rejection: score each sample by how well it explains observations
        if obs_summary is not None and self._total_obs > 50:
            scored = []
            for sample in candidates:
                distance = self._abc_distance(sample, obs_summary)
                scored.append((distance, sample))

            scored.sort(key=lambda x: x[0])
            # Accept top n_samples
            accepted = [s for _, s in scored[:n_samples]]
        else:
            # Not enough data for rejection, use all candidates
            accepted = candidates[:n_samples]

        # Rescale all accepted samples to simulator scale
        return [self._rescale_to_simulator(s) for s in accepted]

    def _compute_summary_stats(self) -> Optional[np.ndarray]:
        """
        Compute summary statistics from observations for ABC comparison.
        Returns a flat array of key statistics.
        """
        if self._trans_probs is None or self._total_obs < 50:
            return None

        # Key summary: flatten the transition matrix (6x6 = 36 values)
        # Plus settlement survival rates
        stats = self._trans_probs.flatten().copy()

        # Add settlement stats if available
        sett_stats = self._settlement_stats
        if sett_stats and (sett_stats["total_alive"] + sett_stats["total_dead"]) > 0:
            n_total = sett_stats["total_alive"] + sett_stats["total_dead"]
            extra = np.array([
                sett_stats["total_alive"] / max(n_total, 1),
                sett_stats["total_ports"] / max(sett_stats["total_alive"], 1),
                sett_stats["avg_food"],
                sett_stats["avg_wealth"],
            ])
            stats = np.concatenate([stats, extra])

        return stats

    def _abc_distance(self, params: dict, obs_summary: np.ndarray) -> float:
        """
        Compute approximate distance between a parameter sample and observations.

        Uses a simplified forward model: predict transition rates from parameters
        and compare with observed transition rates.
        """
        # Predict expected transition rates from parameters
        pred_trans = np.zeros((NUM_CLASSES, NUM_CLASSES))

        ws = params["winter_severity"]
        fa = params["faction_aggression"]
        ta = params["trade_activity"]
        fg = params["forest_growth_rate"]
        er = params["expansion_rate"]
        rr = params["ruin_reclaim_rate"]
        fpf = params["food_per_forest"]
        pdt = params["port_development_threshold"]

        # Empty (0) transitions
        pred_trans[0, 0] = 1.0 - fg * 0.08 - er * 0.05
        pred_trans[0, 4] = fg * 0.08
        pred_trans[0, 1] = er * 0.05

        # Settlement (1) transitions
        death_rate = ws * 0.15 + fa * 0.20
        port_rate = ta * 0.10 * pdt
        pred_trans[1, 1] = max(1.0 - death_rate - port_rate, 0.1)
        pred_trans[1, 3] = death_rate
        pred_trans[1, 2] = port_rate

        # Port (2) transitions
        port_death = ws * 0.08 + fa * 0.10
        pred_trans[2, 2] = max(1.0 - port_death, 0.3)
        pred_trans[2, 3] = port_death
        pred_trans[2, 1] = 0.02

        # Ruin (3) transitions
        reclaim_rate = rr * 0.25
        pred_trans[3, 3] = max(1.0 - reclaim_rate - fg * 0.10, 0.2)
        pred_trans[3, 1] = reclaim_rate * 0.4
        pred_trans[3, 0] = reclaim_rate * 0.3
        pred_trans[3, 4] = fg * 0.10 + reclaim_rate * 0.3

        # Forest (4) transitions
        pred_trans[4, 4] = 0.85
        pred_trans[4, 0] = 0.05
        pred_trans[4, 1] = er * 0.03

        # Mountain (5) - static
        pred_trans[5, 5] = 0.99
        pred_trans[5, 0] = 0.01

        # Normalize rows
        for i in range(NUM_CLASSES):
            row_sum = pred_trans[i].sum()
            if row_sum > 0:
                pred_trans[i] /= row_sum

        # Build predicted summary
        pred_summary = pred_trans.flatten()

        # Add settlement predictions if obs_summary has them
        if len(obs_summary) > 36:
            survival = 1.0 - (ws * 0.15 + fa * 0.20)
            port_frac = ta * 0.3 * pdt
            food_pred = fpf * 2.0 * (1.0 - ws * 0.5)
            wealth_pred = ta * 3.0
            extra = np.array([
                np.clip(survival, 0, 1),
                np.clip(port_frac, 0, 1),
                np.clip(food_pred, 0, 3),
                np.clip(wealth_pred, 0, 5),
            ])
            pred_summary = np.concatenate([pred_summary, extra])

        # Truncate to same length
        min_len = min(len(pred_summary), len(obs_summary))
        pred_summary = pred_summary[:min_len]
        obs_trunc = obs_summary[:min_len]

        # Weighted L2 distance (weight transition matrix entries by observation count)
        weights = np.ones(min_len)
        if self._trans_counts is not None:
            row_totals = self._trans_counts.sum(axis=1)
            for i in range(NUM_CLASSES):
                w = np.log1p(row_totals[i])
                weights[i * NUM_CLASSES:(i + 1) * NUM_CLASSES] = w

        diff = pred_summary - obs_trunc
        distance = float(np.sqrt((weights * diff ** 2).sum()))

        return distance


# ── Convenience functions ─────────────────────────────────────────────────

def infer_parameters(
    initial_states: list,
    observations: dict,
    counts: dict,
    settlements_data: Optional[dict] = None,
) -> dict:
    """Quick helper: infer MAP parameters."""
    engine = ParameterInference(initial_states, observations, counts, settlements_data)
    return engine.infer()


def sample_posterior(
    initial_states: list,
    observations: dict,
    counts: dict,
    settlements_data: Optional[dict] = None,
    n_samples: int = 100,
) -> list[dict]:
    """Quick helper: sample from posterior."""
    engine = ParameterInference(initial_states, observations, counts, settlements_data)
    return engine.infer_posterior(n_samples)
