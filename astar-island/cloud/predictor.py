"""
Predictor for Astar Island — lookup + shift prediction + Monte Carlo.
Optimized with numpy precomputation for speed.
"""
import math
import logging
from concurrent.futures import ProcessPoolExecutor
from typing import Optional, Dict, List, Any

import numpy as np

log = logging.getLogger("predictor")

NC = 6
TTC = {10: 0, 11: 0, 0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5}
PROB_FLOOR = 0.001
DEFAULT_PRIOR = [0.5, 0.1, 0.05, 0.05, 0.25, 0.05]


def cc(c): return TTC.get(c, 0)


class Predictor:
    def __init__(self, lookup: dict):
        self.lookup = lookup

    def _precompute(self, grid, H, W):
        """Precompute ALL spatial features with numpy. O(H*W) total."""
        g = np.array(grid, dtype=np.int32)

        # Init class
        ic_map = np.zeros((H, W), dtype=np.int32)
        for code, cls in TTC.items():
            ic_map[g == code] = cls

        # Coastal: land cell adjacent to ocean (4-connected)
        ocean = (g == 10)
        padded = np.pad(ocean, 1, constant_values=False)
        coastal = np.zeros((H, W), dtype=bool)
        for dy, dx in [(-1,0),(1,0),(0,-1),(0,1)]:
            coastal |= padded[1+dy:H+1+dy, 1+dx:W+1+dx]
        coastal &= ~ocean & (g != 5)

        # Food potential: count forest neighbors (8-connected)
        forest = (g == 4).astype(np.float32)
        food = np.zeros((H, W), dtype=np.int32)
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                if dy == 0 and dx == 0: continue
                shifted = np.zeros_like(forest)
                sy = slice(max(0,-dy), min(H,H-dy))
                sx = slice(max(0,-dx), min(W,W-dx))
                ty = slice(max(0,dy), min(H,H+dy))
                tx = slice(max(0,dx), min(W,W+dx))
                shifted[ty, tx] = forest[sy, sx]
                food += shifted.astype(np.int32)
        food = np.minimum(food, 4)

        # Settlement distance: BFS from all settlements
        sett_mask = (g == 1) | (g == 2)
        sd = np.full((H, W), 999, dtype=np.int32)
        if sett_mask.any():
            ys, xs = np.mgrid[0:H, 0:W]
            sett_ys, sett_xs = np.where(sett_mask)
            for sy, sx in zip(sett_ys, sett_xs):
                dist = np.abs(ys - sy) + np.abs(xs - sx)
                sd = np.minimum(sd, dist)

        # Neighbor settlements (radius 2)
        sett_float = sett_mask.astype(np.float32)
        nsett = np.zeros((H, W), dtype=np.int32)
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                if dy == 0 and dx == 0: continue
                shifted = np.zeros_like(sett_float)
                sy = slice(max(0,-dy), min(H,H-dy))
                sx = slice(max(0,-dx), min(W,W-dx))
                ty = slice(max(0,dy), min(H,H+dy))
                tx = slice(max(0,dx), min(W,W+dx))
                shifted[ty, tx] = sett_float[sy, sx]
                nsett += shifted.astype(np.int32)
        nsett = np.minimum(nsett, 3)

        return ic_map, coastal, food, sd, nsett

    def _dist_bucket(self, d):
        if d <= 3: return "near"
        if d <= 7: return "mid"
        if d <= 12: return "far"
        return "remote"

    def _resolve(self, ic, f, co, db, ns):
        """Lookup with fallback chain. All features pre-computed."""
        keys = [
            f"{ic}_{f}_{co}_{db}_{ns}",
            f"{ic}_{f}_{co}_{db}_0",
            f"{ic}_{min(f,2)}_{co}_{db}_0",
            f"{ic}_0_{co}_{db}_0",
        ]
        for k in keys:
            if k in self.lookup:
                return list(self.lookup[k])
        return list(DEFAULT_PRIOR)

    def predict(self, grid, H, W, shift=None, coastal_damping=0.5, inland_damping=0.7, floor=PROB_FLOOR):
        """Predict H x W x 6 probability distribution."""
        ic_map, coastal, food, sd, nsett = self._precompute(grid, H, W)
        g = np.array(grid, dtype=np.int32)

        pred = []
        for y in range(H):
            row = []
            for x in range(W):
                raw = g[y, x]
                if raw == 5:
                    p = [floor]*NC; p[5] = 1-5*floor
                    row.append(p); continue
                if raw == 10:
                    p = [floor]*NC; p[0] = 1-5*floor
                    row.append(p); continue

                ic = int(ic_map[y, x])
                f = int(food[y, x])
                co = 1 if coastal[y, x] else 0
                db = self._dist_bucket(int(sd[y, x]))
                ns = int(nsett[y, x])

                p = self._resolve(ic, f, co, db, ns)

                # Apply shift
                if shift and ic in shift:
                    damp = coastal_damping if co else inland_damping
                    for c in range(NC):
                        s = shift[ic][c]
                        s = max(0.5, min(3.0, s))  # clip
                        p[c] *= s ** damp

                # Port suppression for non-coastal
                if not co:
                    p[2] = floor

                # Floor + normalize
                total = 0
                for c in range(NC):
                    p[c] = max(p[c], floor)
                    total += p[c]
                for c in range(NC):
                    p[c] /= total

                row.append(p)
            pred.append(row)
        return pred

    def compute_shift(self, obs_transitions):
        """Compute per-class shift from observed transitions vs lookup average."""
        # Compute average rates from lookup
        avg_rates = {}
        for ic in range(NC):
            avg_rates[ic] = [0.0]*NC
            n = 0
            for k, dist in self.lookup.items():
                if int(k.split('_')[0]) == ic:
                    for c in range(NC):
                        avg_rates[ic][c] += dist[c]
                    n += 1
            if n > 0:
                for c in range(NC):
                    avg_rates[ic][c] /= n

        shift = {}
        for ic in range(NC):
            if ic not in obs_transitions:
                shift[ic] = [1.0]*NC
                continue
            total = sum(obs_transitions[ic])
            if total < 15:
                shift[ic] = [1.0]*NC
                continue
            shift[ic] = []
            for c in range(NC):
                obs_rate = obs_transitions[ic][c] / total
                avg = avg_rates[ic][c] if ic in avg_rates else 0
                raw = obs_rate / avg if avg > 0.01 else 1.0
                shift[ic].append(max(0.5, min(3.0, raw)))
        return shift

    def predict_mc(
        self,
        grid: List[List[int]],
        settlements: List[Dict[str, Any]],
        params: Optional[Dict[str, float]] = None,
        n_runs: int = 100,
        n_workers: int = 4,
    ) -> List[List[List[float]]]:
        """
        Run Monte Carlo simulation and return (H, W, 6) probability predictions.

        Uses multiprocessing to parallelize across CPU cores.
        Mountain/ocean cells get hard floors: [0.0005, ...] with dominant class ~1.0.
        """
        from simulator import NorseSimulator, TERRAIN_TO_CLASS, OCEAN, MOUNTAIN

        grid_np = np.asarray(grid, dtype=np.int32)
        H, W = grid_np.shape

        # Split runs across workers
        chunk_size = max(1, n_runs // n_workers)
        chunks = []
        for i in range(n_workers):
            start = i * chunk_size
            end = min(start + chunk_size, n_runs) if i < n_workers - 1 else n_runs
            if start < end:
                chunks.append((grid, settlements, params, list(range(start, end))))

        log.info(f"MC: Starting {n_runs} runs across {len(chunks)} workers")

        # Run in parallel using ProcessPoolExecutor
        counts = np.zeros((H, W, NC), dtype=np.int32)
        try:
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                futures = [
                    executor.submit(_mc_worker, g, s, p, seeds)
                    for g, s, p, seeds in chunks
                ]
                for fut in futures:
                    counts += fut.result()
        except Exception as e:
            log.error(f"MC parallel failed, falling back to serial: {e}")
            sim = NorseSimulator(grid_np, settlements, params)
            for seed in range(n_runs):
                class_grid = sim.run_to_classes(seed)
                np.add.at(
                    counts,
                    (np.arange(H)[:, None], np.arange(W)[None, :], class_grid),
                    1,
                )

        # Convert to probabilities with Jeffreys smoothing
        alpha = 0.5
        probs = (counts.astype(np.float64) + alpha) / (n_runs + NC * alpha)
        probs = np.maximum(probs, 1e-6)
        probs /= probs.sum(axis=2, keepdims=True)

        # Apply hard floors for immutable terrain (mountain/ocean)
        FLOOR = 0.0005
        for y in range(H):
            for x in range(W):
                raw = grid_np[y, x]
                if raw == OCEAN:
                    # Ocean never changes
                    probs[y, x] = FLOOR
                    probs[y, x, 0] = 1.0 - 5 * FLOOR
                elif raw == MOUNTAIN:
                    # Mountain never changes
                    probs[y, x] = FLOOR
                    probs[y, x, 5] = 1.0 - 5 * FLOOR

        # Renormalize
        probs /= probs.sum(axis=2, keepdims=True)

        # Convert to nested list
        return probs.tolist()

    def score_kl(self, pred, gt, H, W):
        """Score using competition formula: 100 * exp(-3 * weighted_kl)."""
        total_wkl = 0.0
        total_ent = 0.0
        for y in range(H):
            for x in range(W):
                ent = 0.0
                kl = 0.0
                for c in range(NC):
                    if gt[y][x][c] > 0:
                        ent -= gt[y][x][c] * math.log(gt[y][x][c])
                        kl += gt[y][x][c] * math.log(gt[y][x][c] / max(pred[y][x][c], 1e-12))
                total_wkl += ent * kl
                total_ent += ent
        if total_ent == 0:
            return 100.0
        return 100.0 * math.exp(-3.0 * total_wkl / total_ent)


def _mc_worker(grid, settlements, params, seeds):
    """Top-level function for multiprocessing (must be picklable)."""
    from simulator import NorseSimulator, NUM_CLASSES

    grid_np = np.asarray(grid, dtype=np.int32)
    H, W = grid_np.shape
    counts = np.zeros((H, W, NUM_CLASSES), dtype=np.int32)

    sim = NorseSimulator(grid_np, settlements, params)
    for seed in seeds:
        class_grid = sim.run_to_classes(seed)
        np.add.at(
            counts,
            (np.arange(H)[:, None], np.arange(W)[None, :], class_grid),
            1,
        )
    return counts
