# Astar Island v7 — Improvement Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the gap from 55→85+ by implementing cross-seed contextual pooling, optimized query allocation, temperature calibration, and a production browser-agent that runs the full swarm pipeline.

**Architecture:** Fix domain priors in swarm.py to match R1 ground truth. Add cross-seed contextual pooling as a new swarm agent. Optimize query strategy for depth over breadth. Port full swarm to browser JS. Validate locally before deploying.

**Tech Stack:** Python (numpy, scipy), JavaScript (browser runtime), Astar Island API

---

## Critical Findings from Round 1

Ground truth transition matrix shows our priors were WRONG:
- Settlement survival: 41% (we used ~55%)
- Settlement→Empty: 37% (we used ~8%)
- Forest→Settlement: 16% (we used ~1%)
- Port survival: 32% (we used ~45%)
- Ruin distribution: near-uniform (we assumed ruin-heavy)

## File Structure

- Modify: `astar-island/swarm.py` — Fix domain priors, add ContextualPoolingAgent
- Modify: `astar-island/query_optimizer.py` — Deep-query strategy (2-3x per viewport)
- Modify: `astar-island/prediction.py` — Already updated priors, add temperature calibration
- Create: `astar-island/browser_agent_v7.js` — Full swarm in JS for browser execution
- Modify: `astar-island/calibration.json` — Already updated with R1 ground truth

---

### Task 1: Fix domain priors in swarm.py

**Files:**
- Modify: `astar-island/swarm.py:589-602` — `_domain_prior()` function

- [ ] **Step 1:** Update `_domain_prior()` to match R1 ground truth calibration
- [ ] **Step 2:** Run validate.py to confirm improvement
- [ ] **Step 3:** Commit

### Task 2: Add ContextualPoolingAgent to swarm

**Files:**
- Modify: `astar-island/swarm.py` — Add new agent class

- [ ] **Step 1:** Create ContextualPoolingAgent that pools observations across ALL seeds by context key (init_class, food, coastal, neighbors, distance)
- [ ] **Step 2:** Register it in SwarmCoordinator with weight 2.0
- [ ] **Step 3:** Test with validate.py
- [ ] **Step 4:** Commit

### Task 3: Optimize query allocation for depth

**Files:**
- Modify: `astar-island/query_optimizer.py`

- [ ] **Step 1:** Change Phase 3 to repeat-observe settlement-dense viewports 2-3x instead of covering new ground
- [ ] **Step 2:** Consider concentrating 25 queries on 2 seeds (full coverage + repeats) and 25 on remaining 3 seeds (full coverage only)
- [ ] **Step 3:** Test with validate.py
- [ ] **Step 4:** Commit

### Task 4: Temperature calibration

**Files:**
- Modify: `astar-island/swarm.py` — Add temperature scaling in _calibrate()

- [ ] **Step 1:** Add temperature parameter T that can be calibrated from R1 analysis data
- [ ] **Step 2:** Apply: p_calibrated = softmax(log(p) / T) after ensemble
- [ ] **Step 3:** Find optimal T by minimizing KL against R1 ground truth
- [ ] **Step 4:** Commit

### Task 5: Port full swarm to browser JS

**Files:**
- Create: `astar-island/browser_agent_v7.js`

- [ ] **Step 1:** Port SwarmCoordinator geometric ensemble to JS
- [ ] **Step 2:** Port all 6 agent types (MC, Statistical, Transition, Spatial, Heuristic, SettlementTrajectory) + new ContextualPooling
- [ ] **Step 3:** Include calibrated priors and temperature scaling
- [ ] **Step 4:** Wire up auto-poller
- [ ] **Step 5:** Test in browser console

### Task 6: Validate end-to-end

- [ ] **Step 1:** Run validate.py with all improvements
- [ ] **Step 2:** Compare v6 vs v7 scores
- [ ] **Step 3:** Deploy browser agent
