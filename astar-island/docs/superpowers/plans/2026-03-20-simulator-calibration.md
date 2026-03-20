# Astar Island: Calibrated Simulator for #1

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a calibrated Monte Carlo simulator that matches the real server simulator, enabling 90+ scores by generating near-perfect predictions for ALL cells (observed and unobserved).

**Architecture:** 3-layer system: (1) Fetch all GT data from completed rounds, (2) Calibrate simulator params via grid search against GT, (3) During rounds: infer params from 50 observations + run calibrated MC → submit predictions. Everything runs in-browser (JS) with GT-calibrated lookup as fallback.

**Tech Stack:** Node.js (calibration scripts), JavaScript (browser agent), Astar Island API

---

### Task 1: Fetch and Cache ALL Ground Truth Data

**Files:**
- Modify: `astar-island/calibrate_gt.js`

- [ ] **Step 1:** Run calibrate_gt.js with token to fetch GT for all completed rounds (R1, R2, R4+)
- [ ] **Step 2:** Verify cache has: `r{N}_init.json` (initial states) + `r{N}_gt_s{0-4}.json` (ground truth) for each round
- [ ] **Step 3:** Log total datasets fetched

Run: `node calibrate_gt.js --token TOKEN --fetch`

### Task 2: Analyze Simulator Accuracy Against GT

**Files:**
- Create: `astar-island/analyze_sim.js`

This script runs our NorseSimulator against GT to find systematic biases:
- For each cached round + seed: run MC sim 200x with DEFAULT_PARAMS
- Compare MC output vs GT per-cell
- Report: per-class bias (sim overstimates forest? understimates settlement?)
- Report: per-context-bucket bias
- Output: `cache/sim_bias_report.json`

- [ ] **Step 1:** Port NorseSimulator to JS (or use the one already in browser_agent_v7.js)
- [ ] **Step 2:** Run against each cached GT dataset
- [ ] **Step 3:** Compute per-class average KL: `KL(GT || MC_sim)` for each initial terrain type
- [ ] **Step 4:** Identify top 10 worst cell contexts (where sim diverges most from GT)
- [ ] **Step 5:** Report findings

### Task 3: Grid Search Parameter Calibration

**Files:**
- Create: `astar-island/calibrate_params.js`

For each completed round, find the parameter set that minimizes KL(GT || MC_sim):

- [ ] **Step 1:** Define parameter grid (coarse first):
  ```
  winter_severity: [0.2, 0.3, 0.4, 0.5, 0.6]
  faction_aggression: [0.1, 0.2, 0.3, 0.4, 0.5]
  trade_activity: [0.2, 0.4, 0.6, 0.8]
  expansion_rate: [0.1, 0.2, 0.3, 0.4]
  forest_growth_rate: [0.02, 0.05, 0.1, 0.15]
  ```
- [ ] **Step 2:** For each param combo: run 50 MC sims per seed, compute weighted KL vs GT
- [ ] **Step 3:** Find best params per round. Report: `{round: N, best_params: {...}, score: X}`
- [ ] **Step 4:** Fine-grid search around best params (±20%, 3 values each)
- [ ] **Step 5:** Save calibrated params: `cache/calibrated_params.json`

### Task 4: Build GT-Calibrated Lookup Table (with fine context)

**Files:**
- Modify: `astar-island/calibrate_gt.js`

- [ ] **Step 1:** Add finer context keys: `(init_class, food_bin[0-4], coastal, sett_dist_bin[0-3/4-7/8-12/13+], neighbor_sett_bin[0/1/2/3+])`
- [ ] **Step 2:** Pool all GT data across all rounds to build lookup
- [ ] **Step 3:** Run leave-one-round-out cross-validation: for each round, build lookup from OTHER rounds, score against held-out
- [ ] **Step 4:** Report: lookup-only score per round (this is our FLOOR without any queries)
- [ ] **Step 5:** Export to `gt_lookup.json` for browser agent

### Task 5: Build the Production Browser Agent v9

**Files:**
- Create: `astar-island/browser_agent_v9.js`

The v9 agent uses a 3-layer prediction system:

**Layer 1: Direct Observation (observed cells)**
- KT estimator with GT-calibrated informative priors
- Prior strength = 0.5 for settlements (data dominates with n≥3)

**Layer 2: Calibrated MC Simulator (all cells)**
- Port NorseSimulator to JS (reuse v7's NorseSim class)
- Infer hidden params from observed transitions using ABC:
  - Run sim 50x with candidate params
  - Compare sim output for OBSERVED cells vs actual observations
  - Pick params that minimize discrepancy
  - Then run 200x with best params for full predictions
- Weight: high for unobserved dynamic cells, low for observed

**Layer 3: GT Lookup (fallback for unobserved)**
- Baked-in GT_CTX_PRIORS from gt_lookup.json
- Used where MC sim and observations are unavailable

**Blending:**
- Observed cells (n≥3): 80% KT + 20% MC_sim
- Observed cells (n=1-2): 40% KT + 30% MC_sim + 30% GT_lookup
- Unobserved dynamic cells: 60% MC_sim + 40% GT_lookup
- Static cells: hard constraints

**Query strategy:**
- Phase 1 (10 queries): 2 viewports per seed covering all settlements
- Phase 2 (40 queries): Adaptive — repeat highest-entropy viewports
- Target: n≥8 per settlement cell, n≥4 per expansion zone cell

- [ ] **Step 1:** Copy v8 structure (API, poller, helpers)
- [ ] **Step 2:** Embed GT_CTX_PRIORS from gt_lookup.json
- [ ] **Step 3:** Port NorseSim class from v7 (with fixes from calibration)
- [ ] **Step 4:** Implement ABC parameter inference from observations
- [ ] **Step 5:** Implement 3-layer blending
- [ ] **Step 6:** Implement focused query strategy
- [ ] **Step 7:** Add auto-calibration from completed rounds
- [ ] **Step 8:** Verify syntax: `node -c browser_agent_v9.js`

### Task 6: Local Validation

**Files:**
- Modify: `astar-island/test_local.js`

- [ ] **Step 1:** Test v9 predictions against R1 GT (synthetic observations from GT)
- [ ] **Step 2:** Test against R2 GT
- [ ] **Step 3:** Test against R4 GT
- [ ] **Step 4:** Compare: v8 score vs v9 score vs lookup-only score
- [ ] **Step 5:** Iterate on blending weights until v9 consistently beats 85+

### Task 7: Deploy and Monitor

- [ ] **Step 1:** Start local HTTP server: `npx serve -l 9999 --cors`
- [ ] **Step 2:** Load v9 in browser via fetch+eval
- [ ] **Step 3:** Verify poller is active
- [ ] **Step 4:** Monitor next round execution
