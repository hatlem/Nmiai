#!/usr/bin/env python3
"""
Shared domain priors for Astar Island prediction.

Single source of truth for calibrated terrain transition priors,
loaded from calibration.json (Round 1 ground truth analysis).

All zero values are floored to MIN_FLOOR (0.003) to prevent
log(0) in KL divergence calculations while preserving probability mass.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

NUM_CLASSES = 6
PROB_FLOOR = 0.005  # Default floor for dynamic terrain
STATIC_FLOOR = 0.001  # Tighter floor for near-impossible transitions (mountain/ocean)
REMOTE_FLOOR = 0.002  # Floor for unlikely transitions on remote cells
MIN_FLOOR = 0.003  # Floor for zeros in calibration data

# ── Load calibration data ─────────────────────────────────────────────────────

_CALIBRATION_PATH = Path(__file__).parent / "calibration.json"

def _load_calibration() -> dict[int, np.ndarray]:
    """Load calibration.json and apply flooring + normalization."""
    with open(_CALIBRATION_PATH) as f:
        raw = json.load(f)

    priors = {}
    for key, values in raw.items():
        if key.startswith("_"):
            continue
        cls_idx = int(key)
        arr = np.array(values, dtype=np.float64)
        # Apply minimum floor to all zeros
        arr = np.maximum(arr, MIN_FLOOR)
        # Normalize
        arr /= arr.sum()
        priors[cls_idx] = arr

    # Override Ruin (class 3) — calibration shows uniform due to insufficient data.
    # Domain-informed prior: ruins tend to stay as ruins or become forest,
    # with modest reclamation by nearby settlements.
    priors[3] = np.array([0.15, 0.08, 0.02, 0.35, 0.35, 0.05])
    priors[3] /= priors[3].sum()

    return priors


CALIBRATED_PRIORS: dict[int, np.ndarray] = _load_calibration()

# ── Hard constraint priors ────────────────────────────────────────────────────

# Use tight floors for static terrain — mountains/ocean never change
MOUNTAIN_PRIOR = np.full(NUM_CLASSES, STATIC_FLOOR)
MOUNTAIN_PRIOR[5] = 1.0 - 5 * STATIC_FLOOR

OCEAN_PRIOR = np.full(NUM_CLASSES, STATIC_FLOOR)
OCEAN_PRIOR[0] = 1.0 - 5 * STATIC_FLOOR


def get_class_conditional_floor(init_cls: int, is_ocean: bool = False) -> np.ndarray:
    """
    Return per-class probability floors based on initial terrain class.

    Avoids wasting probability mass on near-impossible transitions.
    Class indices: 0=ocean/empty, 1=settlement, 2=port, 3=ruin, 4=forest, 5=mountain

    Note: class 0 covers both Ocean and Empty. Use is_ocean=True for ocean cells.
    """
    floors = np.full(NUM_CLASSES, PROB_FLOOR, dtype=np.float64)

    if init_cls == 5:
        # Mountain: stays mountain, all other transitions near-impossible
        floors[:] = STATIC_FLOOR
    elif init_cls == 0 and is_ocean:
        # Ocean: stays ocean, all other transitions near-impossible
        floors[:] = STATIC_FLOOR
    elif init_cls == 0:
        # Empty: mountain never appears, rest uses default floor
        floors[5] = STATIC_FLOOR
    elif init_cls == 4:
        # Forest: mountain never appears, settlement/port/ruin unlikely
        floors[1] = REMOTE_FLOOR
        floors[2] = REMOTE_FLOOR
        floors[3] = REMOTE_FLOOR
        floors[5] = STATIC_FLOOR
    elif init_cls in (1, 2):
        # Settlement/Port: mountains never appear
        floors[5] = STATIC_FLOOR
    elif init_cls == 3:
        # Ruin: mountains never appear
        floors[5] = STATIC_FLOOR
    else:
        floors[5] = STATIC_FLOOR

    return floors


def get_domain_prior(init_cls: int) -> np.ndarray:
    """
    Return calibrated prior for a given initial terrain class.

    Returns a copy with class-conditional floors applied and normalized.
    Always returns a valid probability distribution (sums to 1).
    """
    p = CALIBRATED_PRIORS.get(init_cls, CALIBRATED_PRIORS[0]).copy()
    floors = get_class_conditional_floor(init_cls)
    p = np.maximum(p, floors)
    p /= p.sum()
    return p
