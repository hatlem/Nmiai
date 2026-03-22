# Settlement Metadata Inference — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Use settlement metadata (population, food, wealth, defense, owner_id) from simulate responses to infer hidden parameters and improve round-matching, achieving 90+ scores consistently.

**Architecture:** Extract 5 summary statistics from settlement metadata after first 5 queries. Use these to compute a "regime vector" that matches against historical rounds more precisely than transition rates alone. Feed regime into weighted round ensemble for better priors.

**Tech Stack:** Node.js (agent_v7.js), no new dependencies.

---

## Context

The `/simulate` response returns per-settlement metadata:
```json
{
  "x": 12, "y": 7,
  "population": 2.8,
  "food": 0.4,
  "wealth": 0.7,
  "defense": 0.6,
  "has_port": true,
  "alive": true,
  "owner_id": 3
}
```

Currently agent_v7.js **ignores all of this**. Competitors (sth1712, Meine1964) use it to detect expansion rate, survival rate, and winter severity within 5-6 queries.

## What the metadata reveals about hidden parameters

| Metadata | Low value means | High value means |
|---|---|---|
| avg food | Harsh winters, low food_per_forest | Mild winters, abundant food |
| avg population | High mortality, aggressive raiding | Peaceful expansion |
| avg wealth | Little trade | Active trade routes |
| avg defense | Low conflict | Heavy raiding environment |
| alive ratio | Harsh world (high winter/aggression) | Mild world |
| unique owner_ids | Faction consolidation (aggressive) | Many factions survive |
| port ratio | Low trade/coastal development | Active port development |

---

### Task 1: Collect settlement metadata from simulate responses

**Files:**
- Modify: `astar-island/agent_v7.js:478-510` (query loop)

- [ ] **Step 1: Add metadata collection array**

After line 478 (`let qCount = 0;`), add:

```js
const allSettlements = []; // collect settlement metadata from all queries
```

- [ ] **Step 2: Collect settlements from each simulate response**

After line 506 (`survivalRates.push(surv);`), add:

```js
// Collect settlement metadata for regime detection
if (result.settlements) {
  for (const s of result.settlements) {
    allSettlements.push(s);
  }
}
```

- [ ] **Step 3: Verify syntax**

Run: `node -c astar-island/agent_v7.js`

---

### Task 2: Compute regime vector from settlement metadata

**Files:**
- Modify: `astar-island/agent_v7.js` (add new function before processRound)

- [ ] **Step 1: Add computeRegime function**

Add before the `submitAll` function (~line 382):

```js
// ── Regime detection from settlement metadata ────────────────────────────────
function computeRegime(settlements) {
  if (!settlements.length) return null;
  const alive = settlements.filter(s => s.alive);
  const dead = settlements.filter(s => !s.alive);
  const aliveRatio = alive.length / settlements.length;

  // Average stats from alive settlements
  const avgFood = alive.length ? alive.reduce((s,a) => s + (a.food||0), 0) / alive.length : 0;
  const avgPop = alive.length ? alive.reduce((s,a) => s + (a.population||0), 0) / alive.length : 0;
  const avgWealth = alive.length ? alive.reduce((s,a) => s + (a.wealth||0), 0) / alive.length : 0;
  const avgDefense = alive.length ? alive.reduce((s,a) => s + (a.defense||0), 0) / alive.length : 0;

  // Faction diversity
  const factions = new Set(alive.map(s => s.owner_id).filter(Boolean));
  const factionCount = factions.size;

  // Port ratio
  const portRatio = alive.length ? alive.filter(s => s.has_port).length / alive.length : 0;

  return { aliveRatio, avgFood, avgPop, avgWealth, avgDefense, factionCount, portRatio };
}
```

- [ ] **Step 2: Verify syntax**

Run: `node -c astar-island/agent_v7.js`

---

### Task 3: Use regime in weighted round ensemble

**Files:**
- Modify: `astar-island/agent_v7.js:336-380` (weightedRoundLookup function)

- [ ] **Step 1: Update weightedRoundLookup to accept regime**

Change the function signature and add regime-based distance:

```js
function weightedRoundLookup(obsTrans, survivalRates, regime) {
  if (Object.keys(ROUND_PROFILES).length === 0) return null;

  // Extract key transition rates from observations
  const obsE2S = obsTrans[0] ? obsTrans[0][1] / Math.max(obsTrans[0].reduce((a,b)=>a+b,0), 1) : 0;
  const obsS2S = obsTrans[1] ? obsTrans[1][1] / Math.max(obsTrans[1].reduce((a,b)=>a+b,0), 1) : 0;
  const obsF2F = obsTrans[4] ? obsTrans[4][4] / Math.max(obsTrans[4].reduce((a,b)=>a+b,0), 1) : 0;

  // L2 distance to each historical round's transition rates
  const weights = {};
  let totalWeight = 0;
  for (const [r, profile] of Object.entries(ROUND_PROFILES)) {
    if (!ROUND_LOOKUPS[parseInt(r)]) continue;
    const refE2S = profile['0'] ? profile['0'][1] : 0;
    const refS2S = profile['1'] ? profile['1'][1] : 0;
    const refF2F = profile['4'] ? profile['4'][4] : 0;

    let d = Math.sqrt(
      ((obsE2S - refE2S) / 0.10) ** 2 +
      ((obsS2S - refS2S) / 0.20) ** 2 +
      ((obsF2F - refF2F) / 0.15) ** 2
    );

    // Boost matching with regime data if available
    if (regime) {
      // Use survival rate from regime (more accurate than grid-based)
      const refSurv = profile['1'] ? (profile['1'][1] + (profile['1'][2]||0)) : 0.5;
      d += Math.abs(regime.aliveRatio - refSurv) * 2;
    }

    const w = 1.0 / (d + 0.01);
    weights[r] = w;
    totalWeight += w;
  }

  if (totalWeight === 0) return null;

  const blended = {};
  for (const [r, w] of Object.entries(weights)) {
    const rl = ROUND_LOOKUPS[parseInt(r)];
    const nw = w / totalWeight;
    for (const [k, dist] of Object.entries(rl)) {
      if (!blended[k]) blended[k] = new Array(NC).fill(0);
      for (let c = 0; c < NC; c++) blended[k][c] += dist[c] * nw;
    }
  }

  const matchedRounds = Object.keys(weights)
    .sort((a,b) => weights[b] - weights[a])
    .slice(0, 5)
    .map(r => `R${r}(${(weights[r]/totalWeight*100).toFixed(0)}%)`)
    .join('+');
  return { lookup: blended, desc: matchedRounds };
}
```

- [ ] **Step 2: Update all weightedRoundLookup call sites to pass regime**

In processRound, at each place `weightedRoundLookup` is called (~lines 516, 541), change:

```js
// Before:
const wrl = weightedRoundLookup(obsTrans, survivalRates);
// After:
const regime = computeRegime(allSettlements);
const wrl = weightedRoundLookup(obsTrans, survivalRates, regime);
```

- [ ] **Step 3: Log regime info**

At the Phase 5q milestone log line, add regime info:

```js
if (regime) {
  log(`  Regime: alive=${(regime.aliveRatio*100).toFixed(0)}% food=${regime.avgFood.toFixed(2)} pop=${regime.avgPop.toFixed(1)} factions=${regime.factionCount} ports=${(regime.portRatio*100).toFixed(0)}%`);
}
```

- [ ] **Step 4: Verify syntax and restart**

Run: `node -c astar-island/agent_v7.js`
Then restart: `pkill -f agent_v7; TOKEN=$AINM_TOKEN nohup node agent_v7.js >> /tmp/v7.log 2>&1 &`

---

### Task 4: Store historical regime data for future matching

**Files:**
- Modify: `astar-island/watch_and_calibrate.js` (optional, lower priority)

This is a future improvement: after each round completes, compute and store the regime vector from the analysis endpoint's settlement data. This would enable regime-to-regime matching instead of relying only on transition profiles.

- [ ] **Step 1: Skip for now** — transition profiles + live regime detection is sufficient for R16+

---

## Expected Impact

- **Regime detection from 5 queries** → better round matching → better priors for ALL 1600 cells
- **Most impactful on "bad" rounds** (R10=59, R12=50) where hidden params are extreme and lookup alone fails
- **Zero cost** — we already get settlement metadata in every simulate response, just need to use it
- **Combined with existing improvements** (adaptive floors, temp scaling, spatial propagation) → target 85+ even on hard rounds
