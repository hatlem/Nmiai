"""
Test simulator_v3 against R9 ground truth (seed 0).
GT is a probability distribution (H, W, 6), not a single grid.
Score = 100 * exp(-mean KL divergence).
"""
import json
import sys
import time
import numpy as np

from simulator_v3 import NorseSimulator, TERRAIN_TO_CLASS, NUM_CLASSES

def load_json(path):
    with open(path) as f:
        return json.load(f)

def compute_kl_score(pred_probs, gt_probs):
    """
    Compute entropy-weighted KL divergence: D_KL(gt || pred).
    Score = 100 * exp(-mean_KL).
    """
    H, W, C = gt_probs.shape

    # Floor both to avoid log(0)
    eps = 1e-10
    pred = np.maximum(pred_probs, eps)
    gt = np.maximum(gt_probs, eps)

    # Normalize
    pred /= pred.sum(axis=2, keepdims=True)
    gt /= gt.sum(axis=2, keepdims=True)

    # KL(gt || pred) = sum_c gt[c] * log(gt[c] / pred[c])
    kl = np.sum(gt * np.log(gt / pred), axis=2)

    # Entropy of gt: H(gt) = -sum_c gt[c] * log(gt[c])
    entropy = -np.sum(gt * np.log(gt), axis=2)

    avg_kl = kl.mean()
    score = 100.0 * np.exp(-avg_kl)

    # Entropy-weighted KL (focus on uncertain cells)
    max_entropy = np.log(C)
    weights = entropy / max_entropy  # 0 for deterministic, 1 for uniform
    # Cells with 0 entropy (deterministic) still matter
    weighted_kl = kl  # Use unweighted for now, competition may use different weighting

    return {
        "score": score,
        "avg_kl": avg_kl,
        "median_kl": float(np.median(kl)),
        "max_kl": float(kl.max()),
        "avg_entropy": float(entropy.mean()),
        "kl_grid": kl,
        "entropy_grid": entropy,
    }

def main():
    n_runs = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    round_id = sys.argv[2] if len(sys.argv) > 2 else "r9"

    print(f"Loading {round_id} data...")
    init_data = load_json(f"cache/{round_id}_init.json")
    gt_data = load_json(f"cache/{round_id}_gt_s0.json")

    # Parse from initial_states (seed 0)
    seed_state = init_data["initial_states"][0]
    initial_grid = np.array(seed_state["grid"], dtype=np.int32)
    settlements = seed_state["settlements"]
    H, W = initial_grid.shape
    print(f"Grid size: {H}x{W}, {len(settlements)} settlements")

    # Parse GT probability distribution
    gt_probs = np.array(gt_data["ground_truth"], dtype=np.float64)
    print(f"GT shape: {gt_probs.shape}")

    # GT initial grid (should match)
    gt_init = np.array(gt_data["initial_grid"], dtype=np.int32)
    match = (initial_grid == gt_init).all()
    print(f"Init grids match: {match}")

    # Analyze GT settlement survival
    n_sett = len(settlements)
    sett_survival = 0
    sett_to_ruin = 0
    sett_to_forest = 0
    sett_to_empty = 0
    for s in settlements:
        y, x = s["y"], s["x"]
        gt_dist = gt_probs[y, x]
        # class 1 = settlement, class 2 = port
        survival_prob = gt_dist[1] + gt_dist[2]
        sett_survival += survival_prob
        sett_to_ruin += gt_dist[3]
        sett_to_forest += gt_dist[4]
        sett_to_empty += gt_dist[0]
    print(f"\nGT Settlement analysis ({n_sett} settlements):")
    print(f"  Avg survival (sett+port): {sett_survival/n_sett*100:.1f}%")
    print(f"  Avg → Empty:              {sett_to_empty/n_sett*100:.1f}%")
    print(f"  Avg → Ruin:               {sett_to_ruin/n_sett*100:.1f}%")
    print(f"  Avg → Forest:             {sett_to_forest/n_sett*100:.1f}%")

    # Run Monte Carlo
    print(f"\nRunning {n_runs} Monte Carlo simulations (v3)...")
    t0 = time.time()
    pred_probs = NorseSimulator.run_monte_carlo(initial_grid, settlements, n_runs=n_runs)
    elapsed = time.time() - t0
    print(f"Completed in {elapsed:.1f}s ({elapsed/n_runs:.2f}s per run)")

    # Check sim settlement survival
    print(f"\nSim Settlement analysis ({n_sett} settlements):")
    sim_survival = 0
    sim_to_ruin = 0
    sim_to_forest = 0
    sim_to_empty = 0
    for s in settlements:
        y, x = s["y"], s["x"]
        sim_dist = pred_probs[y, x]
        sim_survival += sim_dist[1] + sim_dist[2]
        sim_to_ruin += sim_dist[3]
        sim_to_forest += sim_dist[4]
        sim_to_empty += sim_dist[0]
    print(f"  Avg survival (sett+port): {sim_survival/n_sett*100:.1f}%")
    print(f"  Avg → Empty:              {sim_to_empty/n_sett*100:.1f}%")
    print(f"  Avg → Ruin:               {sim_to_ruin/n_sett*100:.1f}%")
    print(f"  Avg → Forest:             {sim_to_forest/n_sett*100:.1f}%")

    # Compute score
    result = compute_kl_score(pred_probs, gt_probs)

    print(f"\n{'='*50}")
    print(f"RESULTS for {round_id} (seed 0), {n_runs} MC runs:")
    print(f"{'='*50}")
    print(f"Score (100*exp(-KL)):     {result['score']:.2f}")
    print(f"Avg KL divergence:        {result['avg_kl']:.4f}")
    print(f"Median KL:                {result['median_kl']:.4f}")
    print(f"Max KL:                   {result['max_kl']:.4f}")
    print(f"Avg GT entropy:           {result['avg_entropy']:.4f}")

    # Per-class analysis: compare GT vs sim marginals
    print(f"\n--- Per-class marginal comparison ---")
    class_names = ["Empty", "Settlement", "Port", "Ruin", "Forest", "Mountain"]
    for cls in range(NUM_CLASSES):
        gt_avg = gt_probs[:, :, cls].mean()
        sim_avg = pred_probs[:, :, cls].mean()
        print(f"  {class_names[cls]:12s}: GT={gt_avg:.4f}, Sim={sim_avg:.4f}, diff={sim_avg-gt_avg:+.4f}")

    # Worst cells
    kl_grid = result["kl_grid"]
    worst_idx = np.unravel_index(np.argsort(kl_grid.ravel())[-10:], kl_grid.shape)
    print(f"\n--- 10 worst cells (highest KL) ---")
    for y, x in zip(worst_idx[0], worst_idx[1]):
        gt_d = gt_probs[y, x]
        sim_d = pred_probs[y, x]
        init_code = initial_grid[y, x]
        init_cls = TERRAIN_TO_CLASS.get(init_code, 0)
        print(f"  ({y:2d},{x:2d}) init={class_names[init_cls]:8s} KL={kl_grid[y,x]:.3f}")
        print(f"    GT:  {' '.join(f'{v:.3f}' for v in gt_d)}")
        print(f"    Sim: {' '.join(f'{v:.3f}' for v in sim_d)}")

    # Test multiple rounds if available
    for rid in ["r1", "r2", "r4", "r5", "r6", "r7", "r8"]:
        try:
            rd_init = load_json(f"cache/{rid}_init.json")
            rd_gt = load_json(f"cache/{rid}_gt_s0.json")
            rd_state = rd_init["initial_states"][0]
            rd_grid = np.array(rd_state["grid"], dtype=np.int32)
            rd_setts = rd_state["settlements"]
            rd_gt_probs = np.array(rd_gt["ground_truth"], dtype=np.float64)

            rd_pred = NorseSimulator.run_monte_carlo(rd_grid, rd_setts, n_runs=n_runs)
            rd_result = compute_kl_score(rd_pred, rd_gt_probs)
            print(f"\n{rid}: score={rd_result['score']:.2f}, avg_kl={rd_result['avg_kl']:.4f}")
        except Exception as e:
            pass

if __name__ == "__main__":
    main()
