#!/usr/bin/env python3
"""
Shared domain priors for Astar Island prediction.

Single source of truth for calibrated terrain transition priors,
loaded from calibration.json (Round 1 ground truth analysis).

All zero values are floored to MIN_FLOOR (0.005) to prevent
log(0) in KL divergence calculations while preserving probability mass.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

NUM_CLASSES = 6
PROB_FLOOR = 0.01
MIN_FLOOR = 0.005  # Floor for zeros in calibration data (lower than PROB_FLOOR to save mass)

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
    # with some reclamation by nearby settlements.
    priors[3] = np.array([0.15, 0.12, 0.03, 0.35, 0.30, 0.05])
    priors[3] /= priors[3].sum()

    return priors


CALIBRATED_PRIORS: dict[int, np.ndarray] = _load_calibration()

# ── Hard constraint priors ────────────────────────────────────────────────────

MOUNTAIN_PRIOR = np.full(NUM_CLASSES, PROB_FLOOR)
MOUNTAIN_PRIOR[5] = 1.0 - 5 * PROB_FLOOR

OCEAN_PRIOR = np.full(NUM_CLASSES, PROB_FLOOR)
OCEAN_PRIOR[0] = 1.0 - 5 * PROB_FLOOR


def get_domain_prior(init_cls: int) -> np.ndarray:
    """
    Return calibrated prior for a given initial terrain class.

    Returns a copy with PROB_FLOOR applied and normalized.
    Always returns a valid probability distribution (sums to 1).
    """
    p = CALIBRATED_PRIORS.get(init_cls, CALIBRATED_PRIORS[0]).copy()
    p = np.maximum(p, PROB_FLOOR)
    p /= p.sum()
    return p
