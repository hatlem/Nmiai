"""
Test simulator_v3 against R9 ground truth (seed 0).
Computes entropy-weighted KL divergence score.
"""
import json
import sys
import time
import numpy as np

# Import v3 simulator
from simulator_v3 import NorseSimulator, TERRAIN_TO_CLASS, NUM_CLASSES

def load_json(path):
    with open(path) as f:
        return json.load(f)

def compute_score(pred_probs, gt_grid, initial_grid):
    """
    Compute entropy-weighted KL divergence score matching competition scoring.

    pred_probs: (H, W, 6) probability distribution
    gt_grid: (H, W) ground truth terrain codes
    initial_grid: (H, W) initial terrain codes
    """
    H, W = gt_grid.shape

    # Convert GT to one-hot class distribution
    gt_classes = np.zeros_like(gt_grid)
    for code, cls in TERRAIN_TO_CLASS.items():
        gt_classes[gt_grid == code] = cls

    gt_onehot = np.zeros((H, W, NUM_CLASSES), dtype=np.float64)
    for y in range(H):
        for x in range(W):
            gt_onehot[y, x, gt_classes[y, x]] = 1.0

    # Floor predictions
    pred = np.maximum(pred_probs, 1e-10)
    pred /= pred.sum(axis=2, keepdims=True)

    # KL divergence per cell: sum_c gt[c] * log(gt[c] / pred[c])
    # Since gt is one-hot, this simplifies to -log(pred[gt_class])
    kl_per_cell = np.zeros((H, W))
    for y in range(H):
        for x in range(W):
            gt_cls = gt_classes[y, x]
            kl_per_cell[y, x] = -np.log(pred[y, x, gt_cls])

    # Entropy weight: cells that changed from initial get higher weight
    initial_classes = np.zeros_like(initial_grid)
    for code, cls in TERRAIN_TO_CLASS.items():
        initial_classes[initial_grid == code] = cls

    changed = (initial_classes != gt_classes).astype(np.float64)

    # Simple scoring: average negative log likelihood (lower is better)
    # Then convert to a 0-100 score
    avg_kl = kl_per_cell.mean()
    avg_kl_changed = kl_per_cell[changed > 0].mean() if changed.sum() > 0 else avg_kl
    avg_kl_unchanged = kl_per_cell[changed == 0].mean() if (1 - changed).sum() > 0 else avg_kl

    # Score: 100 * exp(-avg_kl) gives a 0-100 score where lower KL = higher score
    score = 100.0 * np.exp(-avg_kl)
    score_changed = 100.0 * np.exp(-avg_kl_changed)
    score_unchanged = 100.0 * np.exp(-avg_kl_unchanged)

    return {
        "score": score,
        "avg_kl": avg_kl,
        "avg_kl_changed": avg_kl_changed,
        "avg_kl_unchanged": avg_kl_unchanged,
        "score_changed": score_changed,
        "score_unchanged": score_unchanged,
        "n_changed": int(changed.sum()),
        "n_total": H * W,
    }

def main():
    n_runs = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    round_id = sys.argv[2] if len(sys.argv) > 2 else "r9"

    print(f"Loading {round_id} data...")
    init_data = load_json(f"cache/{round_id}_init.json")
    gt_data = load_json(f"cache/{round_id}_gt_s0.json")

    # Parse initial grid
    initial_grid = np.array(init_data["grid"], dtype=np.int32)
    H, W = initial_grid.shape
    print(f"Grid size: {H}x{W}")

    # Parse initial settlements
    settlements = []
    for y in range(H):
        for x in range(W):
            code = initial_grid[y, x]
            if code == 1:  # SETTLEMENT
                settlements.append({"x": x, "y": y, "has_port": False, "alive": True})
            elif code == 2:  # PORT
                settlements.append({"x": x, "y": y, "has_port": True, "alive": True})
    print(f"Found {len(settlements)} initial settlements/ports")

    # Parse GT grid
    gt_grid = np.array(gt_data["grid"], dtype=np.int32)

    # Count GT settlement survival
    gt_sett_count = 0
    for s in settlements:
        gt_code = gt_grid[s["y"], s["x"]]
        if gt_code in (1, 2):
            gt_sett_count += 1
    print(f"GT settlement survival: {gt_sett_count}/{len(settlements)} = {gt_sett_count/max(len(settlements),1)*100:.1f}%")

    # Run Monte Carlo
    print(f"\nRunning {n_runs} Monte Carlo simulations...")
    t0 = time.time()
    probs = NorseSimulator.run_monte_carlo(initial_grid, settlements, n_runs=n_runs)
    elapsed = time.time() - t0
    print(f"Completed in {elapsed:.1f}s ({elapsed/n_runs:.2f}s per run)")

    # Check simulated settlement survival
    sim = NorseSimulator(initial_grid, settlements)
    sim_survival = 0
    for seed in range(min(n_runs, 50)):
        final = sim.run(seed)
        for s in settlements:
            if final[s["y"], s["x"]] in (1, 2):
                sim_survival += 1
    n_check = min(n_runs, 50) * len(settlements)
    print(f"Sim settlement survival: {sim_survival}/{n_check} = {sim_survival/max(n_check,1)*100:.1f}%")

    # Compute score
    result = compute_score(probs, gt_grid, initial_grid)

    print(f"\n{'='*50}")
    print(f"RESULTS for {round_id} (seed 0), {n_runs} MC runs:")
    print(f"{'='*50}")
    print(f"Score (100*exp(-KL)):     {result['score']:.2f}")
    print(f"Avg KL divergence:        {result['avg_kl']:.4f}")
    print(f"KL on changed cells:      {result['avg_kl_changed']:.4f} (score: {result['score_changed']:.2f})")
    print(f"KL on unchanged cells:    {result['avg_kl_unchanged']:.4f} (score: {result['score_unchanged']:.2f})")
    print(f"Changed cells:            {result['n_changed']}/{result['n_total']}")

    # Per-class analysis
    print(f"\n--- Per-class prediction quality ---")
    gt_classes = np.zeros_like(gt_grid)
    for code, cls in TERRAIN_TO_CLASS.items():
        gt_classes[gt_grid == code] = cls

    class_names = ["Empty", "Settlement", "Port", "Ruin", "Forest", "Mountain"]
    for cls in range(NUM_CLASSES):
        mask = gt_classes == cls
        n = mask.sum()
        if n == 0:
            continue
        avg_prob = probs[mask, cls].mean()
        avg_kl_cls = -np.log(np.maximum(probs[mask, cls], 1e-10)).mean()
        print(f"  {class_names[cls]:12s}: n={n:4d}, avg_pred_prob={avg_prob:.4f}, avg_kl={avg_kl_cls:.4f}")

    # Transition matrix analysis
    print(f"\n--- Transition analysis (initial→GT) ---")
    initial_classes = np.zeros_like(initial_grid)
    for code, cls in TERRAIN_TO_CLASS.items():
        initial_classes[initial_grid == code] = cls

    for from_cls in range(NUM_CLASSES):
        from_mask = initial_classes == from_cls
        n_from = from_mask.sum()
        if n_from == 0:
            continue
        dist = []
        for to_cls in range(NUM_CLASSES):
            to_mask = gt_classes == to_cls
            both = from_mask & to_mask
            pct = both.sum() / n_from * 100
            dist.append(f"{class_names[to_cls][:4]}={pct:.1f}%")
        print(f"  {class_names[from_cls]:12s} (n={n_from:4d}) → {', '.join(dist)}")

if __name__ == "__main__":
    main()
