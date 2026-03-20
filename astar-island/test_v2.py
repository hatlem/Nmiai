"""Test simulator_v2 against R9 ground truth."""
import json, sys, os, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from simulator_v2 import run_monte_carlo, score_against_gt, NUM_CLASSES, TERRAIN_TO_CLASS

cache = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cache')

with open(f'{cache}/r9_init.json') as f:
    init = json.load(f)
with open(f'{cache}/r9_gt_s0.json') as f:
    gt_data = json.load(f)

s0 = init['initial_states'][0]
grid = np.array(s0['grid'], dtype=np.int32)
settlements = s0['settlements']
gt_probs = np.array(gt_data['ground_truth'], dtype=np.float64)

print(f"Grid: {grid.shape}, Settlements: {len(settlements)}")

# Run 100 MC simulations
print("\nRunning 100 MC simulations with simulator_v2...")
t0 = time.time()
probs = run_monte_carlo(s0['grid'], settlements, n_runs=100)
t1 = time.time()
print(f"Time: {t1-t0:.2f}s, Output shape: {probs.shape}")

# Score
score = score_against_gt(probs, gt_probs)
print(f"Score (v2 scoring): {score:.2f}")

# Settlement survival
alive_setts = [s for s in settlements if s.get('alive', True)]
sim_survival = [probs[s['y'], s['x'], 1] + probs[s['y'], s['x'], 2] for s in alive_setts]
gt_survival = [gt_probs[s['y'], s['x'], 1] + gt_probs[s['y'], s['x'], 2] for s in alive_setts]
print(f"\nSettlement survival rates:")
print(f"  GT mean:  {np.mean(gt_survival):.3f}")
print(f"  Sim mean: {np.mean(sim_survival):.3f}")
print(f"  GT range: [{np.min(gt_survival):.3f}, {np.max(gt_survival):.3f}]")
print(f"  Sim range: [{np.min(sim_survival):.3f}, {np.max(sim_survival):.3f}]")

# Also compute the 1/(1+wKL) score for comparison
eps = 1e-10
gt_c = np.clip(gt_probs, eps, 1.0)
pred_c = np.clip(probs, eps, 1.0)
kl = np.sum(gt_c * np.log(gt_c / pred_c), axis=2)
entropy = -np.sum(gt_c * np.log(gt_c), axis=2)
weights = entropy / np.log(NUM_CLASSES)
mean_wkl = np.mean(kl * weights)
print(f"\nScoring (1/(1+wKL)): {1.0/(1.0+mean_wkl):.4f}")

# Per-class
class_names = ["Empty/Ocean/Plains", "Settlement", "Port", "Ruin", "Forest", "Mountain"]
initial_class_grid = np.zeros_like(grid)
for code, cls in TERRAIN_TO_CLASS.items():
    initial_class_grid[grid == code] = cls
for cls in range(NUM_CLASSES):
    mask = initial_class_grid == cls
    if mask.any():
        gt_cls = gt_probs[mask]
        sim_cls = probs[mask]
        print(f"\n  Initial class {cls} ({class_names[cls]}): {mask.sum()} cells")
        for out_cls in range(NUM_CLASSES):
            gt_mean = gt_cls[:, out_cls].mean()
            sim_mean = sim_cls[:, out_cls].mean()
            if gt_mean > 0.01 or sim_mean > 0.01:
                print(f"    -> {class_names[out_cls]:20s}: GT={gt_mean:.3f}  Sim={sim_mean:.3f}  diff={sim_mean-gt_mean:+.3f}")
