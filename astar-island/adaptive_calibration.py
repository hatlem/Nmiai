#!/usr/bin/env python3
"""
Adaptive Calibration — Round-specific transition priors from observations.

Instead of using static calibration.json (averaged from past rounds), this module
computes transition probabilities from the current round's 50 observations.
Hidden parameters change between rounds, so observed transitions from THIS round
are far more informative than historical averages.

Flow:
1. compute_observed_transitions() — aggregate initial->observed terrain transitions
2. blend_with_calibration() — blend observed with static priors (confidence-weighted)
3. create_adaptive_prior_fn() — return a callable like get_domain_prior but adaptive
"""

from __future__ import annotations

from typing import Callable, Dict, Optional

import numpy as np

from priors import CALIBRATED_PRIORS, NUM_CLASSES, get_domain_prior

TERRAIN_TO_CLASS = {10: 0, 11: 0, 0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5}


def _classify_value(v: int) -> int:
    """Classify a single terrain value to class index."""
    return TERRAIN_TO_CLASS.get(v, 0)


def compute_observed_transitions(
    initial_states: dict,
    counts: dict,
    observations: dict,
) -> tuple[dict[int, np.ndarray], dict[int, int]]:
    """
    Compute observed transition distributions from current round's observations.

    For each seed, for each observed cell, compare initial terrain class to
    observed terrain class. Aggregate across ALL seeds (hidden params are shared).

    Args:
        initial_states: {seed_idx: {"grid": 40x40 list, ...}}
        counts: {seed_idx: (H, W, C) array of observation counts per class}
        observations: {seed_idx: list of observation dicts} (not used directly,
                      counts already aggregate this)

    Returns:
        (transition_dists, observation_counts) where:
        - transition_dists: {init_cls: np.array(6,)} normalized distributions
        - observation_counts: {init_cls: int} number of cell-observations per class
    """
    # Raw counts: for each initial class, how many times did we observe each outcome?
    raw_counts = {cls: np.zeros(NUM_CLASSES, dtype=np.float64) for cls in range(NUM_CLASSES)}
    obs_totals = {cls: 0 for cls in range(NUM_CLASSES)}

    for seed_idx, state in enumerate(initial_states):
        if seed_idx not in counts:
            continue

        grid = np.asarray(state["grid"], dtype=np.int64)
        H, W = grid.shape
        cell_counts = counts[seed_idx][:, :, :NUM_CLASSES].astype(np.float64)
        n_obs = cell_counts.sum(axis=2)

        for y in range(H):
            for x in range(W):
                if n_obs[y, x] < 1:
                    continue
                init_cls = _classify_value(int(grid[y, x]))
                # Add the observed class counts for this cell
                raw_counts[init_cls] += cell_counts[y, x]
                obs_totals[init_cls] += int(n_obs[y, x])

    # Apply Jeffreys smoothing (add 0.5 to each count) and normalize
    transition_dists = {}
    for cls in range(NUM_CLASSES):
        smoothed = raw_counts[cls] + 0.5
        transition_dists[cls] = smoothed / smoothed.sum()

    return transition_dists, obs_totals


def blend_with_calibration(
    observed_transitions: dict[int, np.ndarray],
    observation_counts: dict[int, int],
    calibrated_priors: Optional[dict[int, np.ndarray]] = None,
) -> dict[int, np.ndarray]:
    """
    Blend observed transitions with static calibration priors.

    More observations -> trust observed more.
    Formula: weight = n_obs / (n_obs + 50)
      - 50 obs:  50%  observed
      - 100 obs: 67%  observed
      - 200 obs: 80%  observed
      - 500 obs: 91%  observed

    Args:
        observed_transitions: {init_cls: np.array(6,)} from compute_observed_transitions
        observation_counts: {init_cls: int} total observations per initial class
        calibrated_priors: Static priors to blend with (default: CALIBRATED_PRIORS)

    Returns:
        {init_cls: np.array(6,)} blended distributions
    """
    if calibrated_priors is None:
        calibrated_priors = CALIBRATED_PRIORS

    blended = {}
    for cls in range(NUM_CLASSES):
        n_obs = observation_counts.get(cls, 0)
        obs_dist = observed_transitions.get(cls, None)
        cal_prior = get_domain_prior(cls)  # Already floored and normalized

        if obs_dist is None or n_obs == 0:
            blended[cls] = cal_prior
            continue

        # Skip blending for static terrain (mountain=5, ocean=0 with high self-transition)
        if cls == 5:
            blended[cls] = cal_prior
            continue

        # Prior strength n₀ derived from empirical Bayes on between-round variance:
        # n₀ ≈ p(1-p)/σ² - 1 ≈ 4-5 for transitions with σ≈0.20
        # With n₀=5 and 300 obs: data gets 98.4% weight (was 85.7% with n₀=50)
        obs_weight = n_obs / (n_obs + 5.0)
        result = obs_weight * obs_dist + (1.0 - obs_weight) * cal_prior
        result = np.maximum(result, 1e-6)
        result /= result.sum()
        blended[cls] = result

    return blended


def compute_distance_band_transitions(
    initial_states: list,
    counts: dict,
    observations: dict,
) -> dict[tuple[int, str], np.ndarray]:
    """
    Compute transition distributions per (init_class, distance_band).

    This is the KEY improvement: near-settlement cells behave very differently
    from far cells, and this varies enormously between rounds.
    Global per-class calibration misses this — a round with high expansion
    needs near-cells scaled up 2-3x but far-cells unchanged.

    Returns:
        {(init_cls, dist_band): np.array(6,)} normalized distributions
    """
    from collections import defaultdict

    band_counts = defaultdict(lambda: np.zeros(NUM_CLASSES, dtype=np.float64))
    band_totals = defaultdict(int)

    for seed_idx, state in enumerate(initial_states):
        if seed_idx not in counts:
            continue
        grid = np.asarray(state["grid"], dtype=np.int64)
        H, W = grid.shape
        settlements = state.get("settlements", [])
        cell_counts = counts[seed_idx][:, :, :NUM_CLASSES].astype(np.float64)
        n_obs = cell_counts.sum(axis=2)

        # Compute settlement distance
        dist = np.full((H, W), 999.0)
        yy, xx = np.mgrid[0:H, 0:W]
        for s in settlements:
            sx, sy = s.get("x", -1), s.get("y", -1)
            if 0 <= sx < W and 0 <= sy < H:
                dist = np.minimum(dist, np.abs(xx - sx).astype(float) + np.abs(yy - sy).astype(float))

        for y in range(H):
            for x in range(W):
                if n_obs[y, x] < 1:
                    continue
                ic = _classify_value(int(grid[y, x]))
                d = dist[y, x]
                if d <= 3:
                    band = "near"
                elif d <= 7:
                    band = "mid"
                elif d <= 12:
                    band = "far"
                else:
                    band = "remote"
                key = (ic, band)
                band_counts[key] += cell_counts[y, x]
                band_totals[key] += int(n_obs[y, x])

    # Normalize with smoothing
    result = {}
    for key, raw in band_counts.items():
        if band_totals[key] >= 10:  # Need minimum observations
            smoothed = raw + 0.5
            result[key] = smoothed / smoothed.sum()

    return result


def create_adaptive_prior_fn(
    blended_priors: dict[int, np.ndarray],
) -> Callable[[int], np.ndarray]:
    """
    Create a callable that returns adaptive priors for a given initial class.

    Works as a drop-in replacement for get_domain_prior().
    """
    def adaptive_prior(init_cls: int) -> np.ndarray:
        return blended_priors.get(init_cls, blended_priors.get(0, get_domain_prior(init_cls))).copy()

    return adaptive_prior
