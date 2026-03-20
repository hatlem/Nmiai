#!/usr/bin/env python3
"""
Best-in-class Astar Island predictor.

Combines three signal sources, ranked by reliability:
1. KT estimator (observed cells) — direct empirical data, most reliable
2. GT lookup (197 context bins) — fine-grained historical GT, excellent for typical rounds
3. Adaptive distance priors — scaled by this round's transition rates, robust for unusual rounds

No Monte Carlo simulator (our sim scores 30-54/100 against real GT).
No ensemble voting. Just clean, well-calibrated probability estimates.

Pipeline:
    observations → adaptive_calibration → per-cell prediction → submit
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Dict, Optional

import numpy as np

from priors import (
    NUM_CLASSES, PROB_FLOOR, STATIC_FLOOR, REMOTE_FLOOR,
    MOUNTAIN_PRIOR, OCEAN_PRIOR, get_domain_prior,
)
from adaptive_calibration import (
    compute_observed_transitions,
    blend_with_calibration,
    create_adaptive_prior_fn,
)

TERRAIN_TO_CLASS = {10: 0, 11: 0, 0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5}

# ── Load GT lookup ───────────────────────────────────────────────────────────

_LOOKUP_PATH = Path(__file__).parent / "gt_lookup.json"

def _load_lookup() -> dict[str, np.ndarray]:
    if not _LOOKUP_PATH.exists():
        return {}
    raw = json.loads(_LOOKUP_PATH.read_text())
    out = {}
    for k, v in raw.items():
        if k.startswith("_"):
            continue
        out[k] = np.array(v, dtype=np.float64)
    return out

GT_LOOKUP = _load_lookup()


# ── Context feature extraction ───────────────────────────────────────────────

def _classify_grid(grid: np.ndarray) -> np.ndarray:
    out = np.zeros_like(grid, dtype=np.int32)
    for code, cls in TERRAIN_TO_CLASS.items():
        out[grid == code] = cls
    return out


def _settlement_distance(grid: np.ndarray, settlements: list, W: int, H: int) -> np.ndarray:
    dist = np.full((H, W), 999.0)
    yy, xx = np.mgrid[0:H, 0:W]
    for s in settlements:
        sx, sy = s.get("x", -1), s.get("y", -1)
        if 0 <= sx < W and 0 <= sy < H:
            dist = np.minimum(dist, np.abs(xx - sx).astype(float) + np.abs(yy - sy).astype(float))
    return dist


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


def _coastal_mask(grid: np.ndarray) -> np.ndarray:
    H, W = grid.shape
    ocean = grid == 10
    coastal = np.zeros((H, W), dtype=bool)
    padded = np.pad(ocean, 1, constant_values=False)
    for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        coastal |= padded[1 + dy:H + 1 + dy, 1 + dx:W + 1 + dx]
    coastal &= ~ocean & (grid != 5)
    return coastal


def _neighbor_settlements(grid: np.ndarray, H: int, W: int) -> np.ndarray:
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


def _dist_bin(d: float) -> str:
    if d <= 3:
        return "near"
    if d <= 7:
        return "mid"
    if d <= 12:
        return "far"
    return "remote"


# ── GT lookup prediction ────────────────────────────────────────────────────

def _lookup_predict(
    init_cls: int, food: int, coastal: int, dist_b: str, n_sett: int,
) -> Optional[np.ndarray]:
    """Look up GT-calibrated prediction from context bins. Returns None if no match."""
    # Try exact key
    key = f"{init_cls}_{food}_{coastal}_{dist_b}_{n_sett}"
    if key in GT_LOOKUP:
        return GT_LOOKUP[key].copy()

    # Fallback 1: drop neighbor count
    for ns in range(4):
        key = f"{init_cls}_{food}_{coastal}_{dist_b}_{ns}"
        if key in GT_LOOKUP:
            return GT_LOOKUP[key].copy()

    # Fallback 2: drop food
    for f in range(5):
        key = f"{init_cls}_{f}_{coastal}_{dist_b}_0"
        if key in GT_LOOKUP:
            return GT_LOOKUP[key].copy()

    return None


# ── Distance-based prior (fallback when lookup misses) ──────────────────────

_DIST_SCALES = {
    # Empty: scales relative to global prior
    (0, "near"): np.array([0.94, 1.29, 1.06, 1.27, 1.33, 1.0]),
    (0, "mid"): np.array([1.06, 0.69, 1.00, 0.73, 0.58, 1.0]),
    (0, "far"): np.array([1.17, 0.13, 0.33, 0.09, 0.05, 1.0]),
    (0, "remote"): np.array([1.17, 0.05, 0.10, 0.03, 0.02, 1.0]),
    # Forest
    (4, "near"): np.array([1.34, 1.26, 1.05, 1.27, 0.93, 1.0]),
    (4, "mid"): np.array([0.59, 0.70, 0.99, 0.69, 1.08, 1.0]),
    (4, "far"): np.array([0.07, 0.15, 0.23, 0.15, 1.23, 1.0]),
    (4, "remote"): np.array([0.03, 0.05, 0.10, 0.05, 1.23, 1.0]),
}


def _distance_prior(
    init_cls: int, dist_b: str,
    adaptive_fn: Optional[Callable] = None,
) -> np.ndarray:
    """Compute distance-scaled prior for a cell."""
    if adaptive_fn is not None:
        global_prior = adaptive_fn(init_cls).copy()
    else:
        global_prior = get_domain_prior(init_cls).copy()

    key = (init_cls, dist_b)
    if key in _DIST_SCALES:
        scaled = global_prior * _DIST_SCALES[key]
        scaled = np.maximum(scaled, 0.001)
        scaled /= scaled.sum()
        return scaled
    else:
        return global_prior


# ── Main predictor ──────────────────────────────────────────────────────────

def predict_all(
    initial_states: list,
    counts: Dict[int, np.ndarray],
    observations: Dict[int, list],
    settlements_data: Optional[Dict[int, list]] = None,
    verbose: bool = True,
) -> Dict[int, np.ndarray]:
    """
    Generate predictions for all seeds.

    Three-layer prediction:
    1. Observed cells (n >= 1): KT estimator with informative prior
    2. Unobserved cells with lookup hit: GT lookup prediction (with adaptive scaling)
    3. Unobserved cells without lookup: Adaptive distance-scaled prior

    Returns:
        {seed_idx: (H, W, 6) probability tensor}
    """
    n_seeds = len(initial_states)

    # ── Adaptive calibration from this round's observations ──
    obs_trans, obs_counts = compute_observed_transitions(
        initial_states, counts, observations or {},
    )
    blended = blend_with_calibration(obs_trans, obs_counts)
    adaptive_fn = create_adaptive_prior_fn(blended)

    total_obs = sum(obs_counts.values())
    if verbose:
        print(f"  Adaptive calibration: {total_obs} cell-observations")
        for cls in range(5):
            if obs_counts.get(cls, 0) > 0:
                cls_names = ["Empty", "Settlement", "Port", "Ruin", "Forest"]
                print(f"    {cls_names[cls]}: {obs_counts[cls]} obs → {blended[cls].round(3)}")

    # ── Compute adaptive scaling factor for lookup ──
    # If adaptive calibration shows different rates than the global average,
    # we scale the lookup predictions proportionally.
    # Scale = adaptive_global / calibration_global
    lookup_scales = {}
    for cls in range(5):
        if obs_counts.get(cls, 0) > 50:  # Need enough data
            cal_global = get_domain_prior(cls)
            adp_global = adaptive_fn(cls)
            scale = adp_global / (cal_global + 1e-8)
            # Clip extreme scales (0.3x to 3.0x)
            scale = np.clip(scale, 0.3, 3.0)
            lookup_scales[cls] = scale

    predictions = {}

    for seed_idx in range(n_seeds):
        state = initial_states[seed_idx]
        grid = np.asarray(state["grid"], dtype=np.int64)
        H, W = grid.shape
        settlements = state.get("settlements", [])

        init_cls = _classify_grid(grid)
        sett_dist = _settlement_distance(grid, settlements, W, H)
        food = _food_map(grid).astype(np.int32)
        food = np.minimum(food, 4)
        coastal = _coastal_mask(grid).astype(np.int32)
        n_sett_map = _neighbor_settlements(grid, H, W)
        n_sett_map = np.minimum(n_sett_map, 3)

        cell_counts = counts[seed_idx][:, :, :NUM_CLASSES].astype(np.float64)
        n_obs = cell_counts.sum(axis=2)

        pred = np.zeros((H, W, NUM_CLASSES), dtype=np.float64)

        # Stats
        n_kt = 0
        n_lookup = 0
        n_dist = 0

        for y in range(H):
            for x in range(W):
                raw = int(grid[y, x])
                ic = int(init_cls[y, x])

                # Static terrain: hardcoded
                if raw == 5:
                    pred[y, x] = MOUNTAIN_PRIOR
                    continue
                if raw == 10:
                    pred[y, x] = OCEAN_PRIOR
                    continue

                sd = float(sett_dist[y, x])
                db = _dist_bin(sd)
                fd = int(food[y, x])
                cs = int(coastal[y, x])
                ns = int(n_sett_map[y, x])

                # ── Layer 1: KT estimator for observed cells ──
                if n_obs[y, x] >= 1:
                    # KT with informative prior from lookup or distance
                    lk = _lookup_predict(ic, fd, cs, db, ns)
                    if lk is not None:
                        # Scale lookup by adaptive factor
                        if ic in lookup_scales:
                            lk = lk * lookup_scales[ic]
                            lk = np.maximum(lk, 0.001)
                            lk /= lk.sum()
                        prior = lk
                    else:
                        prior = _distance_prior(ic, db, adaptive_fn)

                    # Prior strength: 1.5 for settlements (volatile), 2.5 for stable terrain
                    strength = 1.5 if ic in (1, 2) else 2.5
                    if sd > 6:
                        strength = 3.0  # Far cells are more predictable

                    # Dirichlet posterior: (counts + prior * strength) / (n + strength)
                    kt_pred = (cell_counts[y, x] + prior * strength) / (n_obs[y, x] + strength)
                    pred[y, x] = kt_pred
                    n_kt += 1

                else:
                    # ── Layer 2: GT lookup for unobserved cells ──
                    lk = _lookup_predict(ic, fd, cs, db, ns)
                    if lk is not None:
                        if ic in lookup_scales:
                            lk = lk * lookup_scales[ic]
                            lk = np.maximum(lk, 0.001)
                            lk /= lk.sum()
                        pred[y, x] = lk
                        n_lookup += 1
                    else:
                        # ── Layer 3: Adaptive distance prior ──
                        pred[y, x] = _distance_prior(ic, db, adaptive_fn)
                        n_dist += 1

        # ── Final safety floors ──
        # Static terrain
        pred[grid == 5] = MOUNTAIN_PRIOR
        pred[grid == 10] = OCEAN_PRIOR

        # Ensure no zeros
        pred = np.maximum(pred, STATIC_FLOOR)
        pred /= pred.sum(axis=-1, keepdims=True)

        predictions[seed_idx] = pred

        if verbose:
            obs_pct = 100 * (n_obs > 0).sum() / (H * W)
            print(f"  Seed {seed_idx}: {obs_pct:.0f}% observed, "
                  f"KT={n_kt}, lookup={n_lookup}, dist={n_dist}")

    return predictions
