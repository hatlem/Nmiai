#!/usr/bin/env python3
"""
Comprehensive local test suite for Astar Island prediction pipeline.

Tests the full prediction pipeline against cached ground truth data:
  1. Loads gt_lookup.json (context-based prior distributions)
  2. For each GT file in cache/ (r{N}_gt_s{S}.json):
     - Loads initial_grid and ground_truth
     - Simulates 10 queries (samples from GT as fake observations)
     - Computes transition shift from observations
     - Generates shifted predictions
     - Scores against ground truth using exact competition formula
  3. Reports per-round averages and overall average
  4. Asserts average > 70 (lookup-only; real system with MC simulation targets 80+)
  5. Runs 10 trials per seed to account for observation randomness

Self-contained: requires only numpy (no fastapi/httpx).

Usage:
    python test_full_pipeline.py
    python test_full_pipeline.py --trials 20
    python test_full_pipeline.py --min-score 75
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ── Constants ──────────────────────────────────────────────────────────────

NUM_CLASSES = 6
PROB_FLOOR = 0.005
STATIC_FLOOR = 0.001
REMOTE_FLOOR = 0.002
VIEWPORT_SIZE = 15

TERRAIN_TO_CLASS = {10: 0, 11: 0, 0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5}

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
CACHE_DIR = BASE_DIR / "cache"
GT_LOOKUP_PATH = BASE_DIR / "gt_lookup.json"
CALIBRATION_PATH = BASE_DIR / "calibration.json"


# ── Scoring (exact competition formula) ────────────────────────────────────


def score_prediction(prediction: np.ndarray, ground_truth: np.ndarray) -> float:
    """
    Score using entropy-weighted KL divergence (exact competition formula).

    score = max(0, min(100, 100 * exp(-3 * weighted_kl)))
    where weighted_kl = sum(entropy(cell) * KL(gt[cell] || pred[cell])) / sum(entropy(cell))
    """
    eps = 1e-10
    p = np.clip(ground_truth, eps, 1.0)
    q = np.clip(prediction, eps, 1.0)

    # Normalize
    p = p / p.sum(axis=-1, keepdims=True)
    q = q / q.sum(axis=-1, keepdims=True)

    # Per-cell entropy of ground truth
    entropy = -np.sum(p * np.log(p + eps), axis=-1)

    # Per-cell KL divergence: KL(p || q)
    kl = np.sum(p * np.log((p + eps) / (q + eps)), axis=-1)

    # Weighted average
    total_entropy = entropy.sum()
    if total_entropy < eps:
        return 100.0

    weighted_kl = (entropy * kl).sum() / total_entropy
    score = max(0.0, min(100.0, 100.0 * float(np.exp(-3.0 * weighted_kl))))
    return score


# ── Spatial feature computation ────────────────────────────────────────────


def compute_settlement_distance(grid: np.ndarray, settlements: List[dict]) -> np.ndarray:
    """Manhattan distance to nearest settlement. Shape (H, W)."""
    H, W = grid.shape
    dist = np.full((H, W), 999.0)
    yy, xx = np.mgrid[0:H, 0:W]
    for s in settlements:
        sx, sy = s.get("x", -1), s.get("y", -1)
        if 0 <= sx < W and 0 <= sy < H:
            d = np.abs(xx - sx).astype(float) + np.abs(yy - sy).astype(float)
            dist = np.minimum(dist, d)
    return dist


def compute_neighbor_settlements(grid: np.ndarray) -> np.ndarray:
    """Count settlement/port neighbors within radius 2. Shape (H, W)."""
    H, W = grid.shape
    cls_grid = np.zeros((H, W), dtype=np.int32)
    for code, cls in TERRAIN_TO_CLASS.items():
        cls_grid[grid == code] = cls
    sett_mask = np.isin(cls_grid, [1, 2]).astype(np.int32)
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


def compute_food_map(grid: np.ndarray) -> np.ndarray:
    """Count adjacent forest cells per cell (8-connected). Shape (H, W)."""
    H, W = grid.shape
    cls_grid = np.zeros((H, W), dtype=np.int32)
    for code, cls in TERRAIN_TO_CLASS.items():
        cls_grid[grid == code] = cls
    forest = (cls_grid == 4).astype(np.float32)
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


def compute_coastal_map(grid: np.ndarray) -> np.ndarray:
    """True for land cells adjacent to ocean (4-connected). Shape (H, W)."""
    H, W = grid.shape
    ocean = (grid == 10)
    coastal = np.zeros((H, W), dtype=bool)
    padded = np.pad(ocean, 1, constant_values=True)
    for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        coastal |= padded[1 + dy:H + 1 + dy, 1 + dx:W + 1 + dx]
    coastal &= ~ocean
    coastal &= (grid != 5)  # Not mountain
    return coastal


def classify_grid(grid: np.ndarray) -> np.ndarray:
    """Map raw terrain codes to 0-5 class indices."""
    H, W = grid.shape
    cls_grid = np.zeros((H, W), dtype=np.int32)
    for code, cls in TERRAIN_TO_CLASS.items():
        cls_grid[grid == code] = cls
    return cls_grid


# ── Context key builder (matches gt_lookup.json format) ───────────────────


def build_context_key(
    init_cls: int,
    neighbor_setts: int,
    has_port: int,
    dist_bin: str,
    food_bin: int,
) -> str:
    """Build lookup key: {init_cls}_{neighbor_setts}_{has_port}_{dist_bin}_{food_bin}"""
    return f"{init_cls}_{neighbor_setts}_{has_port}_{dist_bin}_{food_bin}"


def get_dist_bin(dist: float) -> str:
    """Bin settlement distance into categories."""
    if dist <= 3:
        return "near"
    elif dist <= 7:
        return "mid"
    elif dist <= 15:
        return "far"
    else:
        return "remote"


# ── Load calibration priors ───────────────────────────────────────────────


def load_calibration_priors() -> Dict[int, np.ndarray]:
    """Load calibration.json and apply flooring + normalization."""
    with open(CALIBRATION_PATH) as f:
        raw = json.load(f)

    priors = {}
    for key, values in raw.items():
        if key.startswith("_"):
            continue
        cls_idx = int(key)
        arr = np.array(values, dtype=np.float64)
        arr = np.maximum(arr, 0.003)
        arr /= arr.sum()
        priors[cls_idx] = arr

    # Override Ruin (class 3)
    priors[3] = np.array([0.15, 0.08, 0.02, 0.35, 0.35, 0.05])
    priors[3] /= priors[3].sum()

    return priors


# ── Prediction engine (self-contained, mirrors prediction.py logic) ───────


class PredictionEngine:
    """
    Layered prediction engine:
      1. Cross-seed transition model (built from observations across seeds)
      2. gt_lookup.json as context-based prior
      3. Calibration priors as fallback
      4. Bayesian KT update with observations

    Mirrors the real pipeline's approach of combining cross-seed information.
    """

    def __init__(
        self,
        gt_lookup: Dict[str, List[float]],
        calibration_priors: Dict[int, np.ndarray],
    ):
        self.gt_lookup = gt_lookup
        self.calibration_priors = calibration_priors

    def build_cross_seed_transition(
        self,
        all_init_grids: List[np.ndarray],
        all_counts: List[np.ndarray],
        all_settlements: List[List[dict]],
    ) -> Dict[str, np.ndarray]:
        """
        Build context-aware transition model from observations across all seeds.
        Groups counts by (init_cls, dist_bin) to capture spatial context.
        Also builds a simpler per-class fallback.
        """
        # Context-aware model: key is "{init_cls}_{dist_bin}"
        ctx_counts: Dict[str, np.ndarray] = {}
        # Simple per-class fallback
        cls_counts = {c: np.zeros(NUM_CLASSES, dtype=np.float64) for c in range(NUM_CLASSES)}

        for init_grid, counts, settlements in zip(all_init_grids, all_counts, all_settlements):
            cls_grid = classify_grid(init_grid)
            sett_dist = compute_settlement_distance(init_grid, settlements)

            for y in range(init_grid.shape[0]):
                for x in range(init_grid.shape[1]):
                    cell_total = counts[y, x].sum()
                    if cell_total == 0:
                        continue
                    ic = int(cls_grid[y, x])
                    db = get_dist_bin(float(sett_dist[y, x]))
                    ctx_key = f"{ic}_{db}"

                    if ctx_key not in ctx_counts:
                        ctx_counts[ctx_key] = np.zeros(NUM_CLASSES, dtype=np.float64)
                    ctx_counts[ctx_key] += counts[y, x].astype(np.float64)
                    cls_counts[ic] += counts[y, x].astype(np.float64)

        # Normalize all entries
        result = {}
        for key, arr in ctx_counts.items():
            total = arr.sum()
            if total > 5:  # Need minimum observations
                result[key] = (arr + 0.5) / (total + 0.5 * NUM_CLASSES)
            # else: will fall through to fallback

        # Add per-class fallbacks
        for ic in range(NUM_CLASSES):
            total = cls_counts[ic].sum()
            if total > 0:
                result[f"cls_{ic}"] = (cls_counts[ic] + 0.5) / (total + 0.5 * NUM_CLASSES)
            else:
                result[f"cls_{ic}"] = self.calibration_priors.get(
                    ic, np.ones(NUM_CLASSES) / NUM_CLASSES
                ).copy()

        return result

    def predict(
        self,
        init_grid: np.ndarray,
        settlements: List[dict],
        counts: np.ndarray,
        cross_seed_trans: Optional[Dict[int, np.ndarray]] = None,
    ) -> np.ndarray:
        """
        Build predictions for a single seed using layered approach.

        Args:
            init_grid: (H, W) raw terrain codes
            settlements: list of settlement dicts with x, y, has_port
            counts: (H, W, NUM_CLASSES) observation count arrays
            cross_seed_trans: optional transition model from other seeds

        Returns:
            (H, W, NUM_CLASSES) prediction array
        """
        H, W = init_grid.shape

        # Compute spatial features
        cls_grid = classify_grid(init_grid)
        sett_dist = compute_settlement_distance(init_grid, settlements)
        neighbor_setts = compute_neighbor_settlements(init_grid)
        food_map = compute_food_map(init_grid)
        coastal_map = compute_coastal_map(init_grid)

        n_obs = counts.sum(axis=2)
        pred = np.zeros((H, W, NUM_CLASSES), dtype=np.float64)

        for y in range(H):
            for x in range(W):
                ic = int(cls_grid[y, x])
                sd = float(sett_dist[y, x])
                ns = int(min(neighbor_setts[y, x], 4))
                fb = int(min(food_map[y, x], 3))
                is_coastal = bool(coastal_map[y, x])
                has_port = 1 if is_coastal and ic in (1, 2) else 0
                dist_bin = get_dist_bin(sd)

                # Layer 1: Context-based lookup prior
                key = build_context_key(ic, ns, has_port, dist_bin, fb)
                lookup_prior = self._get_lookup_prior(key, ic)

                # Layer 2: Cross-seed transition (round-specific signal)
                if cross_seed_trans:
                    # Try context-aware key first
                    cs_key = f"{ic}_{dist_bin}"
                    cs_prior = None
                    if cs_key in cross_seed_trans:
                        cs_prior = cross_seed_trans[cs_key].copy()
                    elif f"cls_{ic}" in cross_seed_trans:
                        cs_prior = cross_seed_trans[f"cls_{ic}"].copy()

                    if cs_prior is not None:
                        cs_prior = np.maximum(cs_prior, PROB_FLOOR)
                        cs_prior /= cs_prior.sum()
                        # Blend: 60% cross-seed (round-specific), 40% lookup (general)
                        blended_prior = 0.6 * cs_prior + 0.4 * lookup_prior
                        blended_prior /= blended_prior.sum()
                    else:
                        blended_prior = lookup_prior
                else:
                    blended_prior = lookup_prior

                # Layer 3: Bayesian update with observations
                cell_counts = counts[y, x].astype(np.float64)
                cell_n = float(n_obs[y, x])

                if cell_n > 0:
                    strength = self._get_prior_strength(ic, sd, cell_n)
                    pred[y, x] = (cell_counts + blended_prior * strength) / (cell_n + strength)
                else:
                    pred[y, x] = blended_prior

        # Apply floors and normalize
        pred = self._apply_floors(pred, cls_grid, init_grid, sett_dist)

        return pred

    def _get_lookup_prior(self, key: str, init_cls: int) -> np.ndarray:
        """Get prior from gt_lookup, falling back to calibration."""
        if key in self.gt_lookup:
            arr = np.array(self.gt_lookup[key], dtype=np.float64)
            arr = np.maximum(arr, PROB_FLOOR)
            arr /= arr.sum()
            return arr

        # Try partial matches: relax food_bin, then neighbor_setts
        parts = key.split("_")
        # Try with food_bin=0
        fallback_key = f"{parts[0]}_{parts[1]}_{parts[2]}_{parts[3]}_0"
        if fallback_key in self.gt_lookup:
            arr = np.array(self.gt_lookup[fallback_key], dtype=np.float64)
            arr = np.maximum(arr, PROB_FLOOR)
            arr /= arr.sum()
            return arr

        # Try with neighbor_setts=0, food_bin=0
        fallback_key2 = f"{parts[0]}_0_{parts[2]}_{parts[3]}_0"
        if fallback_key2 in self.gt_lookup:
            arr = np.array(self.gt_lookup[fallback_key2], dtype=np.float64)
            arr = np.maximum(arr, PROB_FLOOR)
            arr /= arr.sum()
            return arr

        # Ultimate fallback: calibration prior
        return self._get_calibration_prior(init_cls)

    def _get_calibration_prior(self, init_cls: int) -> np.ndarray:
        """Get calibration prior for a given initial terrain class."""
        p = self.calibration_priors.get(init_cls, self.calibration_priors[0]).copy()
        p = np.maximum(p, PROB_FLOOR)
        p /= p.sum()
        return p

    def _get_prior_strength(self, init_cls: int, sett_dist: float, n_obs: float) -> float:
        """Prior strength — gt_lookup priors are well-calibrated, keep them strong.

        With only ~1-2 observations per cell from 10 queries, the prior should dominate.
        Observations only slightly nudge the prediction.
        """
        if init_cls == 5:  # Mountain (near-certain)
            base = 50.0
        elif init_cls == 0 and sett_dist > 10:  # Remote ocean/empty
            base = 30.0
        elif init_cls == 0 and sett_dist > 6:
            base = 15.0
        elif init_cls in (1, 2):  # Settlement/Port (most dynamic, trust obs more)
            base = 8.0
        elif sett_dist <= 3:  # Near settlements
            base = 10.0
        else:
            base = 12.0

        return base

    def _apply_floors(
        self,
        pred: np.ndarray,
        cls_grid: np.ndarray,
        init_grid: np.ndarray,
        sett_dist: np.ndarray,
    ) -> np.ndarray:
        """Apply per-cell probability floors and normalize."""
        H, W = pred.shape[:2]

        # Mountain cells: classes 0-4 get tight floor
        mountain_mask = (init_grid == 5)
        if mountain_mask.any():
            pred[mountain_mask, :5] = np.maximum(pred[mountain_mask, :5], STATIC_FLOOR)
            pred[mountain_mask, 5] = np.maximum(pred[mountain_mask, 5], 1.0 - 5 * STATIC_FLOOR)

        # Ocean cells: classes 1-5 get tight floor
        ocean_mask = (init_grid == 10)
        if ocean_mask.any():
            pred[ocean_mask, 1:] = np.maximum(pred[ocean_mask, 1:], STATIC_FLOOR)
            pred[ocean_mask, 0] = np.maximum(pred[ocean_mask, 0], 1.0 - 5 * STATIC_FLOOR)

        # General floor
        pred = np.maximum(pred, PROB_FLOOR)

        # Remote empty: tighter floors for settlement/port/ruin
        remote_empty = (cls_grid == 0) & (sett_dist > 8) & ~ocean_mask
        if remote_empty.any():
            for c in [1, 2, 3]:
                pred[remote_empty, c] = np.maximum(pred[remote_empty, c], REMOTE_FLOOR)

        # Normalize
        pred /= pred.sum(axis=-1, keepdims=True)

        return pred


# ── Observation simulator ─────────────────────────────────────────────────


def simulate_observations(
    ground_truth: np.ndarray,
    init_grid: np.ndarray,
    settlements: List[dict],
    n_queries: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Simulate observations by sampling from GT distribution.

    Picks viewports centered on settlement-dense areas and samples
    terrain class from the GT probability distribution for each cell.

    Args:
        ground_truth: (H, W, 6) probability distributions
        init_grid: (H, W) initial terrain codes
        settlements: list of settlement dicts
        n_queries: number of 15x15 viewport queries to simulate
        rng: numpy random generator

    Returns:
        (H, W, NUM_CLASSES) observation count array
    """
    H, W = init_grid.shape[:2]
    counts = np.zeros((H, W, NUM_CLASSES), dtype=np.int32)

    # Score each possible viewport position by settlement density
    viewport_scores = np.zeros((H - VIEWPORT_SIZE + 1, W - VIEWPORT_SIZE + 1))
    cls_grid = classify_grid(init_grid)

    for s in settlements:
        sx, sy = s.get("x", -1), s.get("y", -1)
        if 0 <= sx < W and 0 <= sy < H:
            # Each settlement boosts nearby viewports
            for vy in range(max(0, sy - VIEWPORT_SIZE + 1), min(H - VIEWPORT_SIZE + 1, sy + 1)):
                for vx in range(max(0, sx - VIEWPORT_SIZE + 1), min(W - VIEWPORT_SIZE + 1, sx + 1)):
                    viewport_scores[vy, vx] += 10.0
                    if s.get("has_port"):
                        viewport_scores[vy, vx] += 5.0

    # Add small random noise to break ties
    viewport_scores += rng.uniform(0, 0.1, viewport_scores.shape)

    # Select top viewports (greedy, with some diversity)
    selected = []
    used = viewport_scores.copy()
    for _ in range(n_queries):
        vy, vx = np.unravel_index(used.argmax(), used.shape)
        selected.append((int(vy), int(vx)))
        # Suppress nearby viewports for diversity
        for dy in range(-5, 6):
            for dx in range(-5, 6):
                ny, nx = vy + dy, vx + dx
                if 0 <= ny < used.shape[0] and 0 <= nx < used.shape[1]:
                    used[ny, nx] *= 0.3

    # Sample observations from GT
    for vy, vx in selected:
        for dy in range(VIEWPORT_SIZE):
            for dx in range(VIEWPORT_SIZE):
                gy, gx = vy + dy, vx + dx
                if gy >= H or gx >= W:
                    continue
                # Sample class from GT distribution
                gt_dist = ground_truth[gy, gx]
                gt_dist_safe = np.maximum(gt_dist, 1e-12)
                gt_dist_safe /= gt_dist_safe.sum()
                sampled_cls = rng.choice(NUM_CLASSES, p=gt_dist_safe)
                counts[gy, gx, sampled_cls] += 1

    return counts


# ── Data loading ──────────────────────────────────────────────────────────


def load_gt_lookup() -> Dict[str, List[float]]:
    """Load gt_lookup.json."""
    with open(GT_LOOKUP_PATH) as f:
        return json.load(f)


def find_gt_files() -> List[Tuple[int, int, Path]]:
    """Find all r{N}_gt_s{S}.json files in cache/. Returns [(round, seed, path)]."""
    files = []
    for p in sorted(CACHE_DIR.glob("r*_gt_s*.json")):
        name = p.stem  # e.g. "r1_gt_s0"
        parts = name.split("_")
        round_num = int(parts[0][1:])
        seed_num = int(parts[2][1:])
        files.append((round_num, seed_num, p))
    return files


def load_gt_file(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load a GT file and return (initial_grid, ground_truth) as numpy arrays."""
    with open(path) as f:
        data = json.load(f)
    init_grid = np.array(data["initial_grid"], dtype=np.int64)
    ground_truth = np.array(data["ground_truth"], dtype=np.float64)
    return init_grid, ground_truth


def load_init_file(round_num: int) -> Optional[dict]:
    """Load r{N}_init.json to get settlements."""
    init_path = CACHE_DIR / f"r{round_num}_init.json"
    if not init_path.exists():
        return None
    with open(init_path) as f:
        return json.load(f)


# ── Main test runner ──────────────────────────────────────────────────────


def run_full_pipeline_test(
    n_queries: int = 10,
    n_trials: int = 10,
    min_score: float = 70.0,
    verbose: bool = True,
) -> float:
    """
    Run the full pipeline test against cached ground truth.

    Returns overall average score.
    """
    # Load shared data
    gt_lookup = load_gt_lookup()
    calibration_priors = load_calibration_priors()
    engine = PredictionEngine(gt_lookup, calibration_priors)

    gt_files = find_gt_files()
    if not gt_files:
        print("ERROR: No GT files found in cache/")
        sys.exit(1)

    if verbose:
        print("=" * 70)
        print("ASTAR ISLAND - FULL PIPELINE TEST")
        print("=" * 70)
        print(f"GT files found: {len(gt_files)}")
        print(f"Queries per trial: {n_queries}")
        print(f"Trials per seed: {n_trials}")
        print(f"Minimum passing score: {min_score}")
        print()

    # Group by round
    rounds: Dict[int, List[Tuple[int, Path]]] = {}
    for round_num, seed_num, path in gt_files:
        rounds.setdefault(round_num, []).append((seed_num, path))

    all_scores = []
    round_averages = {}

    for round_num in sorted(rounds.keys()):
        seeds = rounds[round_num]
        init_data = load_init_file(round_num)

        if verbose:
            print(f"--- Round {round_num} ({len(seeds)} seeds) ---")

        # Pre-load all seed data for this round (for cross-seed transfer)
        seed_data = {}
        for seed_num, gt_path in sorted(seeds, key=lambda x: x[0]):
            init_grid, ground_truth = load_gt_file(gt_path)
            H, W = init_grid.shape

            settlements = []
            if init_data and "initial_states" in init_data:
                if seed_num < len(init_data["initial_states"]):
                    settlements = init_data["initial_states"][seed_num].get("settlements", [])
            if not settlements:
                cls_grid = classify_grid(init_grid)
                for y in range(H):
                    for x in range(W):
                        if cls_grid[y, x] in (1, 2):
                            settlements.append({
                                "x": x, "y": y,
                                "has_port": cls_grid[y, x] == 2,
                                "alive": True,
                            })

            seed_data[seed_num] = {
                "init_grid": init_grid,
                "ground_truth": ground_truth,
                "settlements": settlements,
                "gt_path": gt_path,
            }

        round_scores = []

        for seed_num in sorted(seed_data.keys()):
            sd = seed_data[seed_num]
            init_grid = sd["init_grid"]
            ground_truth = sd["ground_truth"]
            settlements = sd["settlements"]
            H, W = init_grid.shape

            # Run multiple trials
            seed_trial_scores = []
            for trial in range(n_trials):
                rng = np.random.default_rng(seed=round_num * 1000 + seed_num * 100 + trial)

                # Simulate observations for ALL seeds (needed for cross-seed transfer)
                all_init_grids = []
                all_counts = []
                all_setts = []
                this_counts = None

                for sn in sorted(seed_data.keys()):
                    other = seed_data[sn]
                    rng_other = np.random.default_rng(
                        seed=round_num * 1000 + sn * 100 + trial
                    )
                    counts_sn = simulate_observations(
                        other["ground_truth"],
                        other["init_grid"],
                        other["settlements"],
                        n_queries,
                        rng_other,
                    )
                    all_init_grids.append(other["init_grid"])
                    all_counts.append(counts_sn)
                    all_setts.append(other["settlements"])
                    if sn == seed_num:
                        this_counts = counts_sn

                # Build cross-seed transition model
                cross_seed_trans = engine.build_cross_seed_transition(
                    all_init_grids, all_counts, all_setts,
                )

                # Generate predictions with cross-seed info
                prediction = engine.predict(
                    init_grid, settlements, this_counts, cross_seed_trans,
                )

                # Score
                s = score_prediction(prediction, ground_truth)
                seed_trial_scores.append(s)

            avg_seed = float(np.mean(seed_trial_scores))
            std_seed = float(np.std(seed_trial_scores))
            round_scores.append(avg_seed)
            all_scores.append(avg_seed)

            if verbose:
                rng_check = np.random.default_rng(seed=round_num * 1000 + seed_num)
                counts_check = simulate_observations(
                    ground_truth, init_grid, settlements, n_queries, rng_check,
                )
                n_obs_cells = int((counts_check.sum(axis=2) > 0).sum())
                total_obs = int(counts_check.sum())
                n_sett = len(settlements)
                print(
                    f"  Seed {seed_num}: avg={avg_seed:.1f} +/- {std_seed:.1f}  "
                    f"(sett={n_sett}, obs_cells={n_obs_cells}/{H*W}, total_obs={total_obs})"
                )

        round_avg = float(np.mean(round_scores))
        round_averages[round_num] = round_avg

        if verbose:
            print(f"  Round {round_num} average: {round_avg:.1f}")
            print()

    overall_avg = float(np.mean(all_scores))

    if verbose:
        print("=" * 70)
        print("SUMMARY")
        print("=" * 70)
        for rn in sorted(round_averages.keys()):
            print(f"  Round {rn}: {round_averages[rn]:.1f}")
        print(f"\n  OVERALL AVERAGE: {overall_avg:.1f}")
        print(f"  Target: >= {min_score:.1f}")
        print()

    # Also test with NO observations (prior-only baseline)
    if verbose:
        print("--- Prior-only baseline (no queries) ---")
        prior_scores = []
        for round_num, seed_num, gt_path in gt_files:
            init_grid, ground_truth = load_gt_file(gt_path)
            H, W = init_grid.shape
            settlements = []
            init_data_bl = load_init_file(round_num)
            if init_data_bl and "initial_states" in init_data_bl:
                if seed_num < len(init_data_bl["initial_states"]):
                    settlements = init_data_bl["initial_states"][seed_num].get("settlements", [])
            if not settlements:
                cls_grid = classify_grid(init_grid)
                for y in range(H):
                    for x in range(W):
                        if cls_grid[y, x] in (1, 2):
                            settlements.append({"x": x, "y": y, "has_port": cls_grid[y, x] == 2})

            counts_zero = np.zeros((H, W, NUM_CLASSES), dtype=np.int32)
            prediction = engine.predict(init_grid, settlements, counts_zero)
            s = score_prediction(prediction, ground_truth)
            prior_scores.append(s)

        prior_avg = float(np.mean(prior_scores))
        print(f"  Prior-only average: {prior_avg:.1f}")
        print(f"  Improvement from queries: +{overall_avg - prior_avg:.1f}")
        print()

        # Uniform baseline
        uniform_scores = []
        for _, _, gt_path in gt_files:
            _, ground_truth = load_gt_file(gt_path)
            H, W = ground_truth.shape[:2]
            uniform = np.full((H, W, NUM_CLASSES), 1.0 / NUM_CLASSES)
            s = score_prediction(uniform, ground_truth)
            uniform_scores.append(s)
        print(f"  Uniform baseline: {float(np.mean(uniform_scores)):.1f}")
        print()

    return overall_avg


# ── Entry point ───────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Full pipeline test for Astar Island")
    parser.add_argument("--queries", type=int, default=10,
                        help="Number of simulated queries per trial (default: 10)")
    parser.add_argument("--trials", type=int, default=10,
                        help="Trials per seed for randomness averaging (default: 10)")
    parser.add_argument("--min-score", type=float, default=70.0,
                        help="Minimum passing score (default: 70.0, real system with MC sim targets 80+)")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress detailed output")
    args = parser.parse_args()

    overall_avg = run_full_pipeline_test(
        n_queries=args.queries,
        n_trials=args.trials,
        min_score=args.min_score,
        verbose=not args.quiet,
    )

    if overall_avg >= args.min_score:
        print(f"PASS: Overall average {overall_avg:.1f} >= {args.min_score:.1f}")
        sys.exit(0)
    else:
        print(f"FAIL: Overall average {overall_avg:.1f} < {args.min_score:.1f}")
        sys.exit(1)


if __name__ == "__main__":
    main()
