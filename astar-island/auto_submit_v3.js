#!/usr/bin/env node
/**
 * Auto-submit v3: Production Astar Island agent targeting score 99+.
 *
 * 5-stage pipeline:
 *   1. Instant lookup submission (baseline)
 *   2. Focused observation queries (settlements + expansion zone)
 *   3. Parameter inference (round-matching via transition rates)
 *   4. Bayesian prediction (Dirichlet posterior + round-matched lookup)
 *   5. Spatial smoothing + resubmit
 *
 * Usage: TOKEN=$(cat ../.env.ainm | cut -d= -f2) node auto_submit_v3.js
 */
const fs = require('fs');
const path = require('path');
const https = require('https');

// ── Config ──────────────────────────────────────────────────────────────────
const API = 'https://api.ainm.no/astar-island';
const TOKEN = process.env.TOKEN || process.env.AINM_TOKEN;
const POLL_INTERVAL = 30000;
const API_DELAY = 250; // 4 req/s (limit is 5)
const SUBMIT_DELAY = 520; // 2 req/s for submit
const NC = 6;
const TTC = {10:0, 11:0, 0:0, 1:1, 2:2, 3:3, 4:4, 5:5};
const CACHE_DIR = path.join(__dirname, 'cache');
const PROB_FLOOR = 0.002;

if (!TOKEN) { console.error('Set TOKEN env var'); process.exit(1); }

// ── Load data ───────────────────────────────────────────────────────────────
const LOOKUP_PATH = path.join(__dirname, 'gt_lookup.json');
let LOOKUP = JSON.parse(fs.readFileSync(LOOKUP_PATH, 'utf8'));
console.log(`[init] Loaded ${Object.keys(LOOKUP).length} context bins from gt_lookup.json`);

// Per-round transition matrices
let TRANSITIONS = {};
try {
  const files = fs.readdirSync(CACHE_DIR).filter(f => f.match(/^transitions_r\d+\.json$/));
  for (const f of files) {
    const r = parseInt(f.match(/r(\d+)/)[1]);
    TRANSITIONS[r] = JSON.parse(fs.readFileSync(path.join(CACHE_DIR, f), 'utf8'));
  }
  console.log(`[init] Loaded transitions for rounds: ${Object.keys(TRANSITIONS).sort().join(', ')}`);
} catch(e) { console.log(`[init] No transitions loaded: ${e.message}`); }

// Per-round ground truth data for round-specific lookups
let ROUND_GT = {}; // {roundNum: {seedIdx: {ground_truth, initial_grid}}}
try {
  const files = fs.readdirSync(CACHE_DIR).filter(f => f.match(/^r\d+_gt_s\d+\.json$/));
  for (const f of files) {
    const m = f.match(/r(\d+)_gt_s(\d+)/);
    const r = parseInt(m[1]), s = parseInt(m[2]);
    if (!ROUND_GT[r]) ROUND_GT[r] = {};
    const data = JSON.parse(fs.readFileSync(path.join(CACHE_DIR, f), 'utf8'));
    ROUND_GT[r][s] = { ground_truth: data.ground_truth, initial_grid: data.initial_grid };
  }
  console.log(`[init] Loaded GT for rounds: ${Object.keys(ROUND_GT).sort().join(', ')}`);
} catch(e) { console.log(`[init] No GT data loaded: ${e.message}`); }

// Per-round initial states
let ROUND_INIT = {};
try {
  const files = fs.readdirSync(CACHE_DIR).filter(f => f.match(/^r\d+_init\.json$/));
  for (const f of files) {
    const r = parseInt(f.match(/r(\d+)/)[1]);
    ROUND_INIT[r] = JSON.parse(fs.readFileSync(path.join(CACHE_DIR, f), 'utf8'));
  }
  console.log(`[init] Loaded init states for rounds: ${Object.keys(ROUND_INIT).sort().join(', ')}`);
} catch(e) {}

function reloadLookup() {
  try {
    LOOKUP = JSON.parse(fs.readFileSync(LOOKUP_PATH, 'utf8'));
    console.log(`[reload] Lookup: ${Object.keys(LOOKUP).length} bins`);
  } catch {}
}

// ── API helpers ─────────────────────────────────────────────────────────────
function apiCall(method, urlPath, body) {
  return new Promise((resolve, reject) => {
    const url = new URL(urlPath.startsWith('http') ? urlPath : `${API}${urlPath}`);
    const opts = {
      hostname: url.hostname, path: url.pathname + url.search,
      method, headers: {'Authorization': `Bearer ${TOKEN}`, 'Content-Type':'application/json'}
    };
    const req = https.request(opts, res => {
      let data = '';
      res.on('data', c => data += c);
      res.on('end', () => {
        if (res.statusCode >= 400) return reject(new Error(`${res.statusCode}: ${data.slice(0,300)}`));
        try { resolve(JSON.parse(data)); } catch { resolve(data); }
      });
    });
    req.on('error', reject);
    req.setTimeout(30000, () => { req.destroy(); reject(new Error('timeout')); });
    if (body) req.write(JSON.stringify(body));
    req.end();
  });
}
const sleep = ms => new Promise(r => setTimeout(r, ms));

// ── Spatial helpers ─────────────────────────────────────────────────────────
function cc(c) { return TTC[c] ?? 0; }

function isOcean(v) { return v === 10; }
function isMountain(v) { return v === 5; }
function isStatic(v) { return v === 10 || v === 5; }

function isCoast(g, H, W, y, x) {
  if (g[y][x] === 10 || g[y][x] === 5) return false;
  for (const [dy, dx] of [[-1,0],[1,0],[0,-1],[0,1]]) {
    const ny = y+dy, nx = x+dx;
    if (ny >= 0 && ny < H && nx >= 0 && nx < W && g[ny][nx] === 10) return true;
  }
  return false;
}

function foodPot(g, H, W, y, x) {
  let c = 0;
  for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
    if (!dy && !dx) continue;
    const ny = y+dy, nx = x+dx;
    if (ny >= 0 && ny < H && nx >= 0 && nx < W && g[ny][nx] === 4) c++;
  }
  return c;
}

function settDist(g, H, W, y, x) {
  let m = 999;
  for (let sy = 0; sy < H; sy++) for (let sx = 0; sx < W; sx++) {
    const v = g[sy][sx];
    if (v === 1 || v === 2) m = Math.min(m, Math.abs(y-sy) + Math.abs(x-sx));
  }
  return m;
}

function nSett(g, H, W, y, x) {
  let c = 0;
  for (let dy = -2; dy <= 2; dy++) for (let dx = -2; dx <= 2; dx++) {
    if (!dy && !dx) continue;
    const ny = y+dy, nx = x+dx;
    if (ny >= 0 && ny < H && nx >= 0 && nx < W) {
      const v = g[ny][nx];
      if (v === 1 || v === 2) c++;
    }
  }
  return c;
}

function ctxKey(ig, H, W, y, x) {
  const ic = cc(ig[y][x]);
  const f = Math.min(foodPot(ig, H, W, y, x), 4);
  const co = isCoast(ig, H, W, y, x) ? 1 : 0;
  const d = settDist(ig, H, W, y, x);
  const n = Math.min(nSett(ig, H, W, y, x), 3);
  const db = d <= 3 ? 'near' : d <= 7 ? 'mid' : d <= 12 ? 'far' : 'remote';
  return `${ic}_${f}_${co}_${db}_${n}`;
}

// ── Build round-specific lookup from GT data ────────────────────────────────
function buildRoundLookup(roundNum) {
  const gt = ROUND_GT[roundNum];
  const init = ROUND_INIT[roundNum];
  if (!gt || !init) return null;

  const bins = {}; // ctxKey -> [sumProbs[6], count]
  const seeds = Object.keys(gt).map(Number);
  const H = 40, W = 40;

  for (const si of seeds) {
    const ig = gt[si].initial_grid;
    const truth = gt[si].ground_truth;
    if (!ig || !truth) continue;

    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
      if (isStatic(ig[y][x])) continue;
      const key = ctxKey(ig, H, W, y, x);
      if (!bins[key]) bins[key] = [new Float64Array(NC), 0];
      for (let c = 0; c < NC; c++) bins[key][0][c] += truth[y][x][c];
      bins[key][1]++;
    }
  }

  // Average
  const lookup = {};
  for (const [key, [sums, cnt]] of Object.entries(bins)) {
    lookup[key] = [];
    for (let c = 0; c < NC; c++) lookup[key].push(sums[c] / cnt);
  }
  return lookup;
}

// Pre-build round lookups
const ROUND_LOOKUPS = {};
for (const r of Object.keys(ROUND_GT).map(Number)) {
  ROUND_LOOKUPS[r] = buildRoundLookup(r);
  if (ROUND_LOOKUPS[r]) {
    console.log(`[init] Built round ${r} lookup: ${Object.keys(ROUND_LOOKUPS[r]).length} bins`);
  }
}

// ── Lookup prediction ───────────────────────────────────────────────────────
function lookupPredFromTable(ig, H, W, table) {
  const pred = [];
  for (let y = 0; y < H; y++) {
    pred[y] = [];
    for (let x = 0; x < W; x++) {
      const raw = ig[y][x];
      if (raw === 5) { pred[y][x] = [PROB_FLOOR, PROB_FLOOR, PROB_FLOOR, PROB_FLOOR, PROB_FLOOR, 0.99]; continue; }
      if (raw === 10) { pred[y][x] = [0.99, PROB_FLOOR, PROB_FLOOR, PROB_FLOOR, PROB_FLOOR, PROB_FLOOR]; continue; }

      const key = ctxKey(ig, H, W, y, x);
      const ic = cc(raw), f = Math.min(foodPot(ig, H, W, y, x), 4);
      const co = isCoast(ig, H, W, y, x) ? 1 : 0;
      const d = settDist(ig, H, W, y, x);
      const db = d <= 3 ? 'near' : d <= 7 ? 'mid' : d <= 12 ? 'far' : 'remote';

      // Fallback keys: exact -> reduce nSett -> reduce food -> reduce food+nSett
      const keys = [
        key,
        `${ic}_${f}_${co}_${db}_0`,
        `${ic}_${Math.min(f,2)}_${co}_${db}_0`,
        `${ic}_0_${co}_${db}_0`,
      ];

      let p = null;
      for (const k of keys) {
        if (table[k]) { p = [...table[k]]; break; }
      }
      if (!p) {
        // Try global LOOKUP as final fallback
        for (const k of keys) {
          if (LOOKUP[k]) { p = [...LOOKUP[k]]; break; }
        }
      }
      if (!p) p = [0.5, 0.1, 0.05, 0.05, 0.25, 0.05];

      // Suppress port for non-coastal cells
      if (!co) p[2] = PROB_FLOOR;

      // Floor + normalize
      let s = 0;
      for (let c = 0; c < NC; c++) { p[c] = Math.max(p[c], PROB_FLOOR); s += p[c]; }
      for (let c = 0; c < NC; c++) p[c] /= s;
      pred[y][x] = p;
    }
  }
  return pred;
}

function lookupPred(ig, H, W) {
  return lookupPredFromTable(ig, H, W, LOOKUP);
}

// ── Stage 2: Query planning ─────────────────────────────────────────────────

function findSettlementCells(ig, H, W) {
  const cells = [];
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    const v = ig[y][x];
    if (v === 1 || v === 2) cells.push({x, y});
  }
  return cells;
}

function findDynamicZone(ig, H, W) {
  // All settlements + 3-cell expansion zone around them
  const isTarget = Array.from({length: H}, () => new Uint8Array(W));
  const setts = findSettlementCells(ig, H, W);

  // Mark settlements and their 3-cell neighborhood
  for (const s of setts) {
    for (let dy = -3; dy <= 3; dy++) for (let dx = -3; dx <= 3; dx++) {
      const ny = s.y + dy, nx = s.x + dx;
      if (ny >= 0 && ny < H && nx >= 0 && nx < W && !isStatic(ig[ny][nx])) {
        isTarget[ny][nx] = 1;
      }
    }
  }
  return isTarget;
}

function scoreViewport(ig, H, W, vx, vy, vw, vh, targetMask) {
  let score = 0;
  for (let y = vy; y < vy + vh && y < H; y++) {
    for (let x = vx; x < vx + vw && x < W; x++) {
      if (targetMask[y][x]) {
        const v = ig[y][x];
        if (v === 1 || v === 2) score += 10; // settlement/port
        else if (isCoast(ig, H, W, y, x)) score += 5;
        else score += 3; // expansion zone
      }
    }
  }
  return score;
}

function planQueries(initialStates, H, W, budget) {
  const VP = 15;
  const seedCount = initialStates.length;

  // For each seed: find minimal viewport set covering dynamic zone, scored by importance
  const seedViewports = []; // [{si, x, y, w, h, score}]

  for (let si = 0; si < seedCount; si++) {
    const ig = initialStates[si].grid;
    const targetMask = findDynamicZone(ig, H, W);
    const setts = findSettlementCells(ig, H, W);
    if (setts.length === 0) continue;

    // Find bounding box of all target cells
    let minY = H, maxY = 0, minX = W, maxX = 0;
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
      if (targetMask[y][x]) {
        minY = Math.min(minY, y); maxY = Math.max(maxY, y);
        minX = Math.min(minX, x); maxX = Math.max(maxX, x);
      }
    }

    // Greedy set cover: pick viewport with highest score, mark covered, repeat
    const covered = Array.from({length: H}, () => new Uint8Array(W));
    const vps = [];

    for (let iter = 0; iter < 6; iter++) { // max 6 viewports per seed
      let bestVP = null, bestScore = 0;

      // Search viewport positions (step by 3 for speed)
      for (let vy = Math.max(0, minY - 2); vy <= Math.min(H - VP, maxY); vy += 3) {
        for (let vx = Math.max(0, minX - 2); vx <= Math.min(W - VP, maxX); vx += 3) {
          // Score: only count uncovered target cells
          let sc = 0;
          for (let y = vy; y < vy + VP && y < H; y++) {
            for (let x = vx; x < vx + VP && x < W; x++) {
              if (targetMask[y][x] && !covered[y][x]) {
                const v = ig[y][x];
                if (v === 1 || v === 2) sc += 10;
                else if (isCoast(ig, H, W, y, x)) sc += 5;
                else sc += 3;
              }
            }
          }
          if (sc > bestScore) { bestScore = sc; bestVP = {x: vx, y: vy}; }
        }
      }

      if (!bestVP || bestScore < 5) break;

      // Refine: try 1-cell offsets around bestVP
      for (let ofy = -2; ofy <= 2; ofy++) for (let ofx = -2; ofx <= 2; ofx++) {
        const vy = Math.max(0, Math.min(H - VP, bestVP.y + ofy));
        const vx = Math.max(0, Math.min(W - VP, bestVP.x + ofx));
        let sc = 0;
        for (let y = vy; y < vy + VP && y < H; y++) {
          for (let x = vx; x < vx + VP && x < W; x++) {
            if (targetMask[y][x] && !covered[y][x]) {
              const v = ig[y][x];
              if (v === 1 || v === 2) sc += 10;
              else if (isCoast(ig, H, W, y, x)) sc += 5;
              else sc += 3;
            }
          }
        }
        if (sc > bestScore) { bestScore = sc; bestVP = {x: vx, y: vy}; }
      }

      vps.push({si, x: bestVP.x, y: bestVP.y, w: VP, h: VP, score: bestScore});

      // Mark covered
      for (let y = bestVP.y; y < bestVP.y + VP && y < H; y++) {
        for (let x = bestVP.x; x < bestVP.x + VP && x < W; x++) {
          covered[y][x] = 1;
        }
      }
    }

    seedViewports.push(...vps);
  }

  if (seedViewports.length === 0) {
    // Fallback: center viewport for each seed
    for (let si = 0; si < seedCount; si++) {
      seedViewports.push({si, x: 12, y: 12, w: VP, h: VP, score: 1});
    }
  }

  // Sort by score descending
  seedViewports.sort((a, b) => b.score - a.score);

  // Allocate queries: first pass = all unique viewports once, then repeat high-score ones
  const plan = [];
  // First: one of each viewport
  for (const vp of seedViewports) {
    plan.push({si: vp.si, x: vp.x, y: vp.y, w: vp.w, h: vp.h});
  }

  // Remaining budget: cycle through viewports weighted by score
  const remaining = budget - plan.length;
  if (remaining > 0) {
    // Weight by score for proportional allocation
    const totalScore = seedViewports.reduce((s, v) => s + v.score, 0);
    const allocations = seedViewports.map(v => Math.max(1, Math.round(remaining * v.score / totalScore)));

    // Flatten allocations into plan
    let added = 0;
    for (let round = 0; added < remaining; round++) {
      for (let i = 0; i < seedViewports.length && added < remaining; i++) {
        if (round < allocations[i]) {
          plan.push({si: seedViewports[i].si, x: seedViewports[i].x, y: seedViewports[i].y,
                      w: seedViewports[i].w, h: seedViewports[i].h});
          added++;
        }
      }
    }
  }

  return plan.slice(0, budget);
}

// ── Stage 3: Parameter inference (round matching) ───────────────────────────

function computeObservedRates(observations, initialStates, H, W) {
  // Compute transition rates: for each initial class, what fraction goes to each final class
  const trans = {};
  for (let c = 0; c < NC; c++) trans[c] = new Float64Array(NC);

  for (const obs of observations) {
    const {si, grid: obsGrid, viewport} = obs;
    const ig = initialStates[si].grid;
    for (let gy = 0; gy < obsGrid.length; gy++) {
      for (let gx = 0; gx < obsGrid[gy].length; gx++) {
        const ay = viewport.y + gy, ax = viewport.x + gx;
        if (ay >= H || ax >= W) continue;
        const ic = cc(ig[ay][ax]);
        const fc = cc(obsGrid[gy][gx]);
        trans[ic][fc]++;
      }
    }
  }

  // Normalize to rates
  const rates = {};
  for (let ic = 0; ic < NC; ic++) {
    const total = trans[ic].reduce((a, b) => a + b, 0);
    if (total < 5) continue;
    rates[ic] = [];
    for (let c = 0; c < NC; c++) rates[ic].push(trans[ic][c] / total);
  }
  return rates;
}

function findClosestRound(obsRates) {
  const roundNums = Object.keys(TRANSITIONS).map(Number);
  if (roundNums.length === 0) return null;

  let bestRound = null, bestDist = Infinity;

  for (const r of roundNums) {
    let dist = 0;
    let matched = 0;

    for (const [icStr, obsRate] of Object.entries(obsRates)) {
      const ic = parseInt(icStr);
      const rTrans = TRANSITIONS[r][ic];
      if (!rTrans) continue;

      // L2 distance weighted by class importance
      // Settlement (1) and port (2) transitions are most informative
      const weight = (ic === 1 || ic === 2) ? 3.0 : (ic === 0 ? 1.5 : 1.0);
      for (let c = 0; c < NC; c++) {
        const diff = obsRate[c] - rTrans[c];
        dist += weight * diff * diff;
      }
      matched++;
    }

    if (matched > 0 && dist < bestDist) {
      bestDist = dist;
      bestRound = r;
    }
  }

  return { round: bestRound, distance: bestDist };
}

// ── Stage 4: Bayesian prediction ────────────────────────────────────────────

function buildPerCellCounts(observations, H, W) {
  // Returns {seedIdx: counts[y][x][c]}
  const bySeed = {};
  for (const obs of observations) {
    const si = obs.si;
    if (!bySeed[si]) {
      bySeed[si] = Array.from({length: H}, () =>
        Array.from({length: W}, () => new Float64Array(NC))
      );
    }
    for (let gy = 0; gy < obs.grid.length; gy++) {
      for (let gx = 0; gx < obs.grid[gy].length; gx++) {
        const ay = obs.viewport.y + gy, ax = obs.viewport.x + gx;
        if (ay >= H || ax >= W) continue;
        bySeed[si][ay][ax][cc(obs.grid[gy][gx])]++;
      }
    }
  }
  return bySeed;
}

function bayesianPrediction(ig, H, W, counts, priorLookup, seedIdx) {
  const pred = [];
  const alpha = 1.0; // Dirichlet prior strength

  for (let y = 0; y < H; y++) {
    pred[y] = [];
    for (let x = 0; x < W; x++) {
      const raw = ig[y][x];

      // Static cells: hard constraints
      if (raw === 5) {
        pred[y][x] = [PROB_FLOOR, PROB_FLOOR, PROB_FLOOR, PROB_FLOOR, PROB_FLOOR, 0.99];
        continue;
      }
      if (raw === 10) {
        pred[y][x] = [0.99, PROB_FLOOR, PROB_FLOOR, PROB_FLOOR, PROB_FLOOR, PROB_FLOOR];
        continue;
      }

      // Get prior from round-matched lookup
      const key = ctxKey(ig, H, W, y, x);
      const ic = cc(raw), f = Math.min(foodPot(ig, H, W, y, x), 4);
      const co = isCoast(ig, H, W, y, x) ? 1 : 0;
      const d = settDist(ig, H, W, y, x);
      const db = d <= 3 ? 'near' : d <= 7 ? 'mid' : d <= 12 ? 'far' : 'remote';

      const keys = [
        key,
        `${ic}_${f}_${co}_${db}_0`,
        `${ic}_${Math.min(f,2)}_${co}_${db}_0`,
        `${ic}_0_${co}_${db}_0`,
      ];

      let prior = null;
      // First try round-specific lookup
      if (priorLookup) {
        for (const k of keys) {
          if (priorLookup[k]) { prior = [...priorLookup[k]]; break; }
        }
      }
      // Then global lookup
      if (!prior) {
        for (const k of keys) {
          if (LOOKUP[k]) { prior = [...LOOKUP[k]]; break; }
        }
      }
      if (!prior) prior = [0.5, 0.1, 0.05, 0.05, 0.25, 0.05];

      // Suppress port for non-coastal
      if (!co) prior[2] = PROB_FLOOR;

      // Floor prior
      for (let c = 0; c < NC; c++) prior[c] = Math.max(prior[c], PROB_FLOOR);
      let priorSum = 0;
      for (let c = 0; c < NC; c++) priorSum += prior[c];
      for (let c = 0; c < NC; c++) prior[c] /= priorSum;

      // Check observation data
      const cellCounts = counts ? counts[y][x] : null;
      const nObs = cellCounts ? cellCounts.reduce((a, b) => a + b, 0) : 0;

      let p;
      if (nObs >= 5) {
        // High confidence: Dirichlet posterior with prior
        // p_i = (count_i + alpha * prior_i) / (n + alpha)
        p = new Array(NC);
        const denom = nObs + alpha;
        for (let c = 0; c < NC; c++) {
          p[c] = (cellCounts[c] + alpha * prior[c]) / denom;
        }
      } else if (nObs >= 1) {
        // Medium confidence: blend Dirichlet posterior with lookup
        // Weight observations more as n increases: w = n/(n+3)
        const kt = new Array(NC);
        const denom = nObs + alpha;
        for (let c = 0; c < NC; c++) {
          kt[c] = (cellCounts[c] + alpha * prior[c]) / denom;
        }
        const w = nObs / (nObs + 3);
        p = new Array(NC);
        for (let c = 0; c < NC; c++) {
          p[c] = w * kt[c] + (1 - w) * prior[c];
        }
      } else {
        // Unobserved: pure prior (round-matched lookup)
        p = prior;
      }

      // Suppress port for non-coastal
      if (!co) p[2] = PROB_FLOOR;

      // Floor + normalize
      let s = 0;
      for (let c = 0; c < NC; c++) { p[c] = Math.max(p[c], PROB_FLOOR); s += p[c]; }
      for (let c = 0; c < NC; c++) p[c] /= s;
      pred[y][x] = p;
    }
  }
  return pred;
}

// ── Stage 5: Spatial smoothing (belief propagation) ─────────────────────────

function spatialSmooth(pred, ig, H, W, counts, iterations = 3, damping = 0.15) {
  // Only smooth unobserved non-static cells using neighbor information
  // Compatibility: settlements cluster, forest isolated
  const COMPAT = [
    // 0:empty 1:sett 2:port 3:ruin 4:forest 5:mountain
    [0.30, 0.08, 0.04, 0.04, 0.10, 0.02], // neighbor is empty
    [0.08, 0.12, 0.08, 0.06, 0.04, 0.02], // neighbor is settlement
    [0.06, 0.08, 0.10, 0.04, 0.04, 0.02], // neighbor is port
    [0.08, 0.06, 0.04, 0.10, 0.08, 0.02], // neighbor is ruin
    [0.12, 0.06, 0.04, 0.06, 0.20, 0.02], // neighbor is forest
    [0.06, 0.02, 0.02, 0.02, 0.04, 0.30], // neighbor is mountain
  ];

  const result = pred.map(row => row.map(p => [...p]));
  const dirs = [[-1,0],[1,0],[0,-1],[0,1]];

  for (let iter = 0; iter < iterations; iter++) {
    const prev = result.map(row => row.map(p => [...p]));

    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        if (isStatic(ig[y][x])) continue;

        // Only smooth cells with few or no observations
        const nObs = counts ? counts[y][x].reduce((a, b) => a + b, 0) : 0;
        if (nObs >= 3) continue; // well-observed cells don't need smoothing

        // Compute message from neighbors
        const msg = new Float64Array(NC);
        let nNeighbors = 0;

        for (const [dy, dx] of dirs) {
          const ny = y + dy, nx = x + dx;
          if (ny < 0 || ny >= H || nx < 0 || nx >= W) continue;
          nNeighbors++;

          // Message: for each class c of this cell, sum over neighbor classes
          for (let c = 0; c < NC; c++) {
            for (let nc = 0; nc < NC; nc++) {
              msg[c] += COMPAT[nc][c] * prev[ny][nx][nc];
            }
          }
        }

        if (nNeighbors === 0) continue;

        // Normalize message
        let msgSum = 0;
        for (let c = 0; c < NC; c++) msgSum += msg[c];
        if (msgSum > 0) for (let c = 0; c < NC; c++) msg[c] /= msgSum;

        // Blend with current belief (damping)
        const d = damping * (nObs === 0 ? 1.0 : 0.5); // less smoothing for partially observed
        for (let c = 0; c < NC; c++) {
          result[y][x][c] = (1 - d) * prev[y][x][c] + d * msg[c];
        }

        // Suppress port for non-coastal
        if (!isCoast(ig, H, W, y, x)) result[y][x][2] = PROB_FLOOR;

        // Floor + normalize
        let s = 0;
        for (let c = 0; c < NC; c++) { result[y][x][c] = Math.max(result[y][x][c], PROB_FLOOR); s += result[y][x][c]; }
        for (let c = 0; c < NC; c++) result[y][x][c] /= s;
      }
    }
  }

  return result;
}

// ── Apply transition shift to predictions ───────────────────────────────────

function applyTransitionShift(pred, obsRates, matchedRound, ig, H, W) {
  // If we have both observed rates and a matched round, compute shift multipliers
  // shift[ic][c] = obsRate[ic][c] / matchedRate[ic][c]
  if (!obsRates || !matchedRound || !TRANSITIONS[matchedRound]) return pred;

  const matchTrans = TRANSITIONS[matchedRound];
  const shift = {};

  for (const [icStr, obsRate] of Object.entries(obsRates)) {
    const ic = parseInt(icStr);
    if (!matchTrans[ic]) continue;
    shift[ic] = [];
    for (let c = 0; c < NC; c++) {
      if (matchTrans[ic][c] > 0.01) {
        shift[ic].push(Math.max(0.1, Math.min(5.0, obsRate[c] / matchTrans[ic][c])));
      } else {
        shift[ic].push(1.0);
      }
    }
  }

  const result = pred.map(row => row.map(p => [...p]));
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const raw = ig[y][x];
      if (isStatic(raw)) continue;
      const ic = cc(raw);
      if (!shift[ic]) continue;

      for (let c = 0; c < NC; c++) result[y][x][c] *= shift[ic][c];

      // Suppress port for non-coastal
      if (!isCoast(ig, H, W, y, x)) result[y][x][2] = PROB_FLOOR;

      let s = 0;
      for (let c = 0; c < NC; c++) { result[y][x][c] = Math.max(result[y][x][c], PROB_FLOOR); s += result[y][x][c]; }
      for (let c = 0; c < NC; c++) result[y][x][c] /= s;
    }
  }
  return result;
}

// ── Final normalization with strict floor ───────────────────────────────────

function finalNormalize(pred, ig, H, W) {
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const raw = ig[y][x];
      const co = !isStatic(raw) && isCoast(ig, H, W, y, x);

      // Static overrides
      if (raw === 5) { pred[y][x] = [PROB_FLOOR, PROB_FLOOR, PROB_FLOOR, PROB_FLOOR, PROB_FLOOR, 0.99]; }
      else if (raw === 10) { pred[y][x] = [0.99, PROB_FLOOR, PROB_FLOOR, PROB_FLOOR, PROB_FLOOR, PROB_FLOOR]; }
      else {
        // Suppress port for non-coastal
        if (!co) pred[y][x][2] = PROB_FLOOR;
      }

      // Floor + normalize
      let s = 0;
      for (let c = 0; c < NC; c++) { pred[y][x][c] = Math.max(pred[y][x][c], PROB_FLOOR); s += pred[y][x][c]; }
      for (let c = 0; c < NC; c++) pred[y][x][c] /= s;
    }
  }
  return pred;
}

// ── Main pipeline ───────────────────────────────────────────────────────────

// Persist completed rounds to file so restarts don't re-trigger
const COMPLETED_FILE = path.join(__dirname, 'completed_rounds.json');
let _completedList = [];
try { _completedList = JSON.parse(fs.readFileSync(COMPLETED_FILE, 'utf8')); } catch {}
const completed = new Set(_completedList);
function saveCompleted() { try { fs.writeFileSync(COMPLETED_FILE, JSON.stringify([...completed])); } catch {} }

async function processRound(round) {
  const t0 = Date.now();
  const log = (msg) => console.log(`[${((Date.now()-t0)/1000).toFixed(1)}s] ${msg}`);
  console.log(`\n${'='.repeat(70)}`);
  console.log(`[${new Date().toISOString().slice(11,19)}] Round ${round.round_number} (${round.id.slice(0,8)}) — ACTIVE`);
  console.log(`${'='.repeat(70)}`);

  reloadLookup();
  const detail = await apiCall('GET', `/rounds/${round.id}`);
  const H = detail.map_height, W = detail.map_width;
  const seedsCount = detail.seeds_count;
  log(`Map: ${W}x${H}, ${seedsCount} seeds`);

  for (let si = 0; si < seedsCount; si++) {
    const setts = detail.initial_states[si].settlements || [];
    log(`  Seed ${si}: ${setts.length} settlements`);
  }

  // ── STAGE 1: Instant lookup submission ────────────────────────────────────
  log('STAGE 1: Submitting lookup-only predictions (baseline)...');
  for (let si = 0; si < seedsCount; si++) {
    const pred = lookupPred(detail.initial_states[si].grid, H, W);
    try {
      await apiCall('POST', '/submit', {round_id: round.id, seed_index: si, prediction: pred});
      await sleep(SUBMIT_DELAY);
    } catch (e) {
      log(`  WARN: Submit seed ${si} failed: ${e.message}`);
    }
  }
  log(`STAGE 1 done: ${seedsCount} seeds submitted in ${((Date.now()-t0)/1000).toFixed(1)}s`);

  // ── STAGE 2: Focused observation ──────────────────────────────────────────
  let budget;
  try {
    budget = await apiCall('GET', '/budget');
  } catch (e) {
    log(`  Budget check failed: ${e.message}, using default 50`);
    budget = { queries_max: 50, queries_used: 0 };
  }
  const queriesLeft = budget.queries_max - budget.queries_used;
  log(`STAGE 2: Observing (${queriesLeft}/${budget.queries_max} queries available)...`);

  if (queriesLeft <= 0) {
    log('  No queries remaining. Skipping observation.');
    completed.add(round.id);
    return;
  }

  const plan = planQueries(detail.initial_states, H, W, queriesLeft);

  // Log query distribution
  const seedDist = {};
  for (const q of plan) { seedDist[q.si] = (seedDist[q.si] || 0) + 1; }
  log(`  Query plan: ${plan.length} queries — per seed: ${JSON.stringify(seedDist)}`);

  // Count unique viewports
  const uniqueVPs = new Set(plan.map(q => `${q.si}_${q.x}_${q.y}`));
  log(`  Unique viewports: ${uniqueVPs.size}, avg repeats: ${(plan.length / uniqueVPs.size).toFixed(1)}`);

  const observations = [];
  let failCount = 0;
  for (let qi = 0; qi < plan.length; qi++) {
    const q = plan[qi];
    try {
      await sleep(API_DELAY);
      const result = await apiCall('POST', '/simulate', {
        round_id: round.id,
        seed_index: q.si,
        viewport_x: q.x, viewport_y: q.y,
        viewport_w: q.w, viewport_h: q.h,
      });
      observations.push({si: q.si, grid: result.grid, viewport: result.viewport});
      if ((qi + 1) % 10 === 0) log(`    ${qi + 1}/${plan.length} queries done`);
    } catch (e) {
      failCount++;
      if (e.message.includes('429')) {
        rateLimitCount = (rateLimitCount || 0) + 1;
        if (rateLimitCount > 10) { log('    Too many rate limits, stopping queries'); break; }
        log(`    Rate limited at query ${qi}, waiting 2s... (${rateLimitCount}/10)`);
        await sleep(2000);
        qi--; // retry once
      } else if (e.message.includes('budget')) {
        log(`    Budget exhausted at query ${qi}`);
        break;
      } else {
        log(`    Query ${qi} failed: ${e.message}`);
        if (failCount > 5) { log('    Too many failures, stopping queries'); break; }
      }
    }
  }
  log(`STAGE 2 done: ${observations.length} observations in ${((Date.now()-t0)/1000).toFixed(1)}s`);

  // ── STAGE 3: Parameter inference ──────────────────────────────────────────
  log('STAGE 3: Inferring round parameters...');
  const obsRates = computeObservedRates(observations, detail.initial_states, H, W);

  // Log observed rates
  for (const [ic, rates] of Object.entries(obsRates)) {
    log(`  Class ${ic} transitions: [${rates.map(v => v.toFixed(3)).join(', ')}]`);
  }

  const match = findClosestRound(obsRates);
  let matchedRound = null;
  let roundLookup = null;
  if (match && match.round !== null) {
    matchedRound = match.round;
    roundLookup = ROUND_LOOKUPS[matchedRound] || null;
    log(`  Best matching round: R${matchedRound} (L2 distance: ${match.distance.toFixed(4)})`);
    if (roundLookup) log(`  Using round ${matchedRound} lookup (${Object.keys(roundLookup).length} bins)`);
  } else {
    log('  No round match found, using global lookup');
  }

  // ── STAGE 4: Bayesian prediction ──────────────────────────────────────────
  log('STAGE 4: Computing Bayesian predictions...');
  const cellCounts = buildPerCellCounts(observations, H, W);

  const predictions = [];
  for (let si = 0; si < seedsCount; si++) {
    const ig = detail.initial_states[si].grid;
    const counts = cellCounts[si] || null;

    // Use round-matched lookup as prior, falling back to global
    const priorTable = roundLookup || LOOKUP;

    let pred = bayesianPrediction(ig, H, W, counts, priorTable, si);

    // Apply transition shift for fine-tuning
    pred = applyTransitionShift(pred, obsRates, matchedRound, ig, H, W);

    predictions.push(pred);

    // Stats
    if (counts) {
      let obsCount = 0, highObs = 0;
      for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
        const n = counts[y][x].reduce((a, b) => a + b, 0);
        if (n > 0) obsCount++;
        if (n >= 5) highObs++;
      }
      log(`  Seed ${si}: ${obsCount} observed cells (${highObs} with n>=5)`);
    } else {
      log(`  Seed ${si}: no observations (pure lookup)`);
    }
  }

  // ── STAGE 5: Spatial smoothing + resubmit ─────────────────────────────────
  log('STAGE 5: Spatial smoothing + final submission...');
  for (let si = 0; si < seedsCount; si++) {
    const ig = detail.initial_states[si].grid;
    const counts = cellCounts[si] || null;

    let pred = spatialSmooth(predictions[si], ig, H, W, counts, 3, 0.15);
    pred = finalNormalize(pred, ig, H, W);

    try {
      await apiCall('POST', '/submit', {round_id: round.id, seed_index: si, prediction: pred});
      await sleep(SUBMIT_DELAY);
      log(`  Seed ${si} submitted`);
    } catch (e) {
      log(`  WARN: Final submit seed ${si} failed: ${e.message}`);
    }
  }

  const elapsed = ((Date.now() - t0) / 1000).toFixed(1);
  log(`ALL STAGES COMPLETE in ${elapsed}s`);
  completed.add(round.id);
}

// ── Poll loop ───────────────────────────────────────────────────────────────

let processing = false;

async function poll() {
  if (processing) { process.stdout.write('~'); return; }
  try {
    const rounds = await apiCall('GET', '/rounds');
    const active = rounds.filter(r => r.status === 'active' && !completed.has(r.id));
    if (!active.length) {
      process.stdout.write('.');
      return;
    }
    for (const round of active) {
      processing = true;
      completed.add(round.id); saveCompleted(); // Mark immediately to prevent re-trigger
      try {
        await processRound(round);
      } catch (e) {
        console.error(`\nRound processing error: ${e.message}`);
        console.error(e.stack);
      } finally {
        processing = false;
      }
    }
  } catch (e) {
    console.error(`\nPoll error: ${e.message}`);
  }
}

console.log('========================================');
console.log('  Auto-submit v3 — Production Agent');
console.log('  Target: score 99+');
console.log('  Polling every 30s...');
console.log('========================================');
console.log(`  Lookup bins: ${Object.keys(LOOKUP).length}`);
console.log(`  Transition rounds: ${Object.keys(TRANSITIONS).sort().join(', ') || 'none'}`);
console.log(`  Round lookups: ${Object.keys(ROUND_LOOKUPS).filter(r => ROUND_LOOKUPS[r]).join(', ') || 'none'}`);
console.log('');

poll();
setInterval(poll, POLL_INTERVAL);
