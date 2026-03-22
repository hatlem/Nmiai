# Astar Island -- Viking Civilisation Prediction

## The Task

Astar Island is a probabilistic prediction challenge from [NM i AI 2026](https://app.ainm.no). A black-box Norse civilisation simulator runs on a 40x40 grid for 50 in-game years. Settlements grow, factions raid each other, trade routes form, winters kill off weak settlements, and forests reclaim ruins.

The goal: predict the probability distribution of 6 terrain classes (Empty, Settlement, Port, Ruin, Forest, Mountain) for every cell on the map. The simulator is stochastic -- the same seed produces different outcomes each run. Ground truth is computed by the organizers over hundreds of runs.

**Constraints:**
- 5 map seeds per round, each requiring a W x H x 6 probability tensor
- Only 50 simulation queries per round, shared across all 5 seeds
- Each query reveals at most a 15x15 viewport of the 40x40 map
- Scoring uses entropy-weighted KL divergence (score = 100 * exp(-3 * weighted_kl))

## Architecture

The agent (`agent_v7.js`) uses a **lookup-first** strategy. Empirical data showed that raw simulation queries hurt scores in 9 out of 16 rounds -- the lookup table alone scored 91.3 (our best round).

### Pipeline

1. **Phase 1 -- Instant lookup submission.** On round start, immediately submit predictions using `gt_lookup.json` -- a precomputed table mapping spatial features to probability distributions, built from ground truth of all completed rounds.

2. **Phase 2 -- Regime detection (5 queries).** Run a small number of queries to observe settlement survival rates, food levels, and transition rates. This identifies the current round's "regime" (harsh winter vs prosperous growth).

3. **Phase 3 -- Weighted round ensemble.** Compare observed transition rates against historical round profiles using L2 distance. Blend round-specific lookup tables weighted by similarity, then resubmit.

4. **Phase 4 -- XGBoost refinement.** Pass spatial features and observed transition rates to an XGBoost model for regime-conditioned cell-level predictions. Resubmit with refined predictions after additional queries.

### Cell Key System

Each non-static cell is characterized by a feature key: `{initialClass}_{foodCount}_{coastal}_{distanceBand}_{nearbySettlements}`

- `initialClass`: terrain type at simulation start (0-5)
- `foodCount`: adjacent forest cells (0-4)
- `coastal`: whether the cell borders ocean (0/1)
- `distanceBand`: Manhattan distance to nearest settlement (near/mid/far/remote)
- `nearbySettlements`: settlement count within radius 2 (0-3)

The lookup table maps each key to a 6-class probability distribution averaged across all historical ground truth data.

## Key Innovations

- **Ground truth accumulation.** After each round completes, ground truth is fetched via the analysis API and folded into `gt_lookup.json`. The lookup grew from ~50 bins in round 1 to 228+ bins by round 13, steadily improving predictions.

- **Lookup-dominant blending.** Observations are blended with a prior strength of 30, meaning even 10 observations only shift the prediction 25% from the lookup. This prevents noisy queries from corrupting strong priors.

- **Port suppression.** Non-coastal cells have their Port probability forced to the floor value before any blending. Ports can only exist adjacent to ocean.

- **Weighted round ensemble.** Instead of picking the single most similar historical round, all rounds are blended proportionally to their similarity (inverse L2 distance on transition rates).

- **Adaptive survival computation.** Settlement survival is computed from the observed grid (comparing initial vs simulated terrain) rather than metadata, which always reported 100%.

## How to Run

```bash
# Set auth token
export TOKEN="your-jwt-token"
# Or source from .env.ainm
source ../.env.ainm && export TOKEN=$AINM_TOKEN

# Run the agent (polls every 30s for active rounds)
node agent_v7.js
```

The agent runs continuously, polling for active rounds. When a round is detected, it processes it automatically and marks it as completed in `v7_completed.json`.

### Dependencies

- Node.js (no npm packages required -- uses only built-in modules)
- Python 3 with XGBoost (optional, for Phase 4 refinement)

### Key Files

| File | Purpose |
|------|---------|
| `agent_v7.js` | Main agent -- polling, querying, prediction, submission |
| `gt_lookup.json` | Precomputed probability distributions per cell feature key |
| `cache/transitions_r*.json` | Per-round transition rate profiles for regime matching |
| `cache/r*_gt_s*.json` | Cached ground truth from completed rounds |
| `xgb_predict.py` | XGBoost prediction script (called from Node.js) |
| `prediction.py` | Shared prediction utilities |
| `v7_completed.json` | Tracks which rounds have been processed |

## Results

Best round score: **91.3 points** (Round 13, rank #22).

| Round | Score | Rank | Notes |
|-------|-------|------|-------|
| R9 | 90.4 | #29 | First strong round (agent_final) |
| R11 | 88.4 | #18 | Consistent performance |
| R13 | 91.3 | #22 | Best score -- lookup-only with 228 bins |

### What Worked

- Accumulating ground truth across rounds (more data = better lookup)
- Lookup-dominant strategy (queries often add noise)
- Port suppression before blending
- Weighted round ensemble over winner-take-all

### What Did Not Work

- Heavy query usage (hurt scores 9/16 rounds)
- Temperature scaling (softened deterministic cells)
- Adaptive probability floors (added noise to harsh rounds)
- Spatial propagation of observations (spread noise)
