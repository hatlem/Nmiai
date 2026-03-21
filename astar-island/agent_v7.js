#!/usr/bin/env node
/**
 * agent_v7.js — Optimized Astar Island agent.
 *
 * Fixes from agent_final.js:
 * 1. Rate limit: use budget from API response, not rlCount heuristic
 * 2. Survival: compute from observed grid, not metadata (always 100%)
 * 3. KT threshold: n>=1 with adaptive alpha (was n>=3)
 * 4. Port suppression BEFORE KT blend (was after)
 * 5. Smarter query allocation: concentrate on fewer viewports
 * 6. SIM_DELAY 350ms (was 280ms)
 * 7. Weighted round ensemble (not winner-take-all)
 */
const fs = require('fs'), path = require('path'), https = require('https');

const NC = 6;
const TTC = {10:0, 11:0, 0:0, 1:1, 2:2, 3:3, 4:4, 5:5};
const FL = 0.0005;       // absolute minimum floor
const FL_NEAR = 0.02;    // floor for cells near settlements (high entropy)
const FL_MID = 0.01;     // floor for mid-range cells
const FL_FAR = 0.005;    // floor for remote cells
const TEMP = 1.15;       // temperature scaling >1 = softer predictions (safer for KL)
const DAMP = 0.5;        // lowered from 0.8 — ensemble prior does most work now
const COAST_DAMP = 0.2;  // lowered from 0.3
const CLIP = [0.1, 10.0];
const SIM_DELAY = 350;
const SUB_DELAY = 550;
const POLL = 30000;

const TOKEN = process.env.TOKEN || process.env.AINM_TOKEN;
if (!TOKEN) { console.error('Need TOKEN'); process.exit(1); }

// ── Load data ───────────────────────────────────────────────────────────────
let LOOKUP = JSON.parse(fs.readFileSync(path.join(__dirname, 'gt_lookup.json'), 'utf8'));

const CACHE = path.join(__dirname, 'cache');
const ROUND_PROFILES = {};
try {
  for (const f of fs.readdirSync(CACHE).filter(f => f.match(/^transitions_r\d+\.json$/))) {
    const r = parseInt(f.match(/r(\d+)/)[1]);
    ROUND_PROFILES[r] = JSON.parse(fs.readFileSync(path.join(CACHE, f), 'utf8'));
  }
} catch {}

const ROUND_LOOKUPS = {};
try {
  const rounds = [...new Set(fs.readdirSync(CACHE).filter(f=>f.match(/^r\d+_gt_s/)).map(f=>parseInt(f.match(/r(\d+)/)[1])))];
  for (const r of rounds) {
    const stats = {};
    for (let si = 0; si < 5; si++) {
      const p = path.join(CACHE, `r${r}_gt_s${si}.json`);
      if (!fs.existsSync(p)) continue;
      const d = JSON.parse(fs.readFileSync(p, 'utf8'));
      const ig = d.initial_grid, gt = d.ground_truth, H = d.height, W = d.width;
      for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
        const raw = ig[y][x];
        if (raw === 5 || raw === 10) continue;
        const k = cellKey(ig, H, W, y, x);
        if (!stats[k]) stats[k] = {c: new Float64Array(6), n: 0};
        for (let c = 0; c < 6; c++) stats[k].c[c] += gt[y][x][c];
        stats[k].n++;
      }
    }
    ROUND_LOOKUPS[r] = {};
    for (const [k, v] of Object.entries(stats)) {
      if (v.n < 2) continue;
      const tot = v.c.reduce((a,b) => a+b, 0);
      ROUND_LOOKUPS[r][k] = Array.from({length:6}, (_, c) => (v.c[c] + 0.005) / (tot + 0.03));
    }
  }
} catch {}

console.log(`[init] Lookup: ${Object.keys(LOOKUP).length} bins`);
console.log(`[init] Round profiles: ${Object.keys(ROUND_PROFILES).length}`);
console.log(`[init] Round lookups: ${Object.keys(ROUND_LOOKUPS).length}`);

// ── Completed rounds ────────────────────────────────────────────────────────
const DONE_FILE = path.join(__dirname, 'v7_completed.json');
let DONE = new Set();
try { DONE = new Set(JSON.parse(fs.readFileSync(DONE_FILE, 'utf8'))); } catch {}
function saveDone() { try { fs.writeFileSync(DONE_FILE, JSON.stringify([...DONE])); } catch {} }

// ── API ─────────────────────────────────────────────────────────────────────
function api(method, p, body) {
  return new Promise((resolve, reject) => {
    const url = new URL('https://api.ainm.no/astar-island' + p);
    const opts = { hostname: url.hostname, path: url.pathname, method,
      headers: { 'Authorization': `Bearer ${TOKEN}`, 'Content-Type': 'application/json' } };
    const req = https.request(opts, res => {
      let d = ''; res.on('data', c => d += c);
      res.on('end', () => {
        if (res.statusCode === 429) return reject(new Error('429'));
        if (res.statusCode >= 400) return reject(new Error(`${res.statusCode}: ${d.slice(0,200)}`));
        try { resolve(JSON.parse(d)); } catch { resolve(d); }
      });
    });
    req.on('error', reject);
    if (body) req.write(JSON.stringify(body));
    req.end();
  });
}
const sleep = ms => new Promise(r => setTimeout(r, ms));

// ── Spatial ─────────────────────────────────────────────────────────────────
function cc(c) { return TTC[c] ?? 0; }

function cellKey(ig, H, W, y, x) {
  const raw = ig[y][x], ic = cc(raw);
  let food = 0;
  for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
    if (!dy && !dx) continue;
    const ny = y+dy, nx = x+dx;
    if (ny >= 0 && ny < H && nx >= 0 && nx < W && ig[ny][nx] === 4) food++;
  }
  food = Math.min(food, 4);
  let co = 0;
  for (const [dy,dx] of [[-1,0],[1,0],[0,-1],[0,1]]) {
    const ny = y+dy, nx = x+dx;
    if (ny >= 0 && ny < H && nx >= 0 && nx < W && ig[ny][nx] === 10) { co = 1; break; }
  }
  if (ig[y][x] === 10 || ig[y][x] === 5) co = 0;
  let sd = 999;
  for (let sy = 0; sy < H; sy++) for (let sx = 0; sx < W; sx++)
    if (ig[sy][sx] === 1 || ig[sy][sx] === 2) sd = Math.min(sd, Math.abs(y-sy) + Math.abs(x-sx));
  const db = sd <= 3 ? 'near' : sd <= 7 ? 'mid' : sd <= 12 ? 'far' : 'remote';
  let ns = 0;
  for (let dy = -2; dy <= 2; dy++) for (let dx = -2; dx <= 2; dx++) {
    if (!dy && !dx) continue;
    const ny = y+dy, nx = x+dx;
    if (ny >= 0 && ny < H && nx >= 0 && nx < W && (ig[ny][nx] === 1 || ig[ny][nx] === 2)) ns++;
  }
  ns = Math.min(ns, 3);
  return `${ic}_${food}_${co}_${db}_${ns}`;
}

function precompute(ig, H, W) {
  const setts = [];
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++)
    if (ig[y][x] === 1 || ig[y][x] === 2) setts.push({y, x});
  const sd = Array.from({length: H}, () => new Int32Array(W).fill(999));
  for (const s of setts) for (let y = 0; y < H; y++) for (let x = 0; x < W; x++)
    sd[y][x] = Math.min(sd[y][x], Math.abs(y-s.y) + Math.abs(x-s.x));
  const coastal = Array.from({length: H}, () => new Uint8Array(W));
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    if (ig[y][x] === 10 || ig[y][x] === 5) continue;
    for (const [dy,dx] of [[-1,0],[1,0],[0,-1],[0,1]]) {
      const ny = y+dy, nx = x+dx;
      if (ny >= 0 && ny < H && nx >= 0 && nx < W && ig[ny][nx] === 10) { coastal[y][x] = 1; break; }
    }
  }
  const food = Array.from({length: H}, () => new Int32Array(W));
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    let c = 0;
    for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
      if (!dy && !dx) continue;
      const ny = y+dy, nx = x+dx;
      if (ny >= 0 && ny < H && nx >= 0 && nx < W && ig[ny][nx] === 4) c++;
    }
    food[y][x] = Math.min(c, 4);
  }
  const nsett = Array.from({length: H}, () => new Int32Array(W));
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    let c = 0;
    for (let dy = -2; dy <= 2; dy++) for (let dx = -2; dx <= 2; dx++) {
      if (!dy && !dx) continue;
      const ny = y+dy, nx = x+dx;
      if (ny >= 0 && ny < H && nx >= 0 && nx < W && (ig[ny][nx] === 1 || ig[ny][nx] === 2)) c++;
    }
    nsett[y][x] = Math.min(c, 3);
  }
  return { setts, sd, coastal, food, nsett };
}

// ── Prediction ──────────────────────────────────────────────────────────────
function resolveLookup(lookup, ic, f, co, db, ns) {
  const keys = [`${ic}_${f}_${co}_${db}_${ns}`, `${ic}_${f}_${co}_${db}_0`,
                 `${ic}_${Math.min(f,2)}_${co}_${db}_0`, `${ic}_0_${co}_${db}_0`,
                 `${ic}_${f}_${co}_${db}`, `${ic}_${co}_${db}`, `${ic}_${db}`, `${ic}`];
  for (const k of keys) { if (lookup[k]) return [...lookup[k]]; }
  return [.5, .1, .05, .05, .25, .05];
}

function predict(ig, H, W, pre, shift, counts, lookup) {
  const lk = lookup || LOOKUP;
  const pred = Array.from({length: H}, () => Array.from({length: W}, () => new Array(NC)));
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    const raw = ig[y][x];
    if (raw === 5) { pred[y][x] = [FL,FL,FL,FL,FL,1-5*FL]; continue; }
    if (raw === 10) { pred[y][x] = [1-5*FL,FL,FL,FL,FL,FL]; continue; }

    const ic = cc(raw), f = pre.food[y][x], co = pre.coastal[y][x];
    const d = pre.sd[y][x], ns = pre.nsett[y][x];
    const db = d <= 3 ? 'near' : d <= 7 ? 'mid' : d <= 12 ? 'far' : 'remote';

    let prior = resolveLookup(lk, ic, f, co, db, ns);

    // Apply shift
    if (shift && shift[ic]) {
      const damp = co ? COAST_DAMP : DAMP;
      for (let c = 0; c < NC; c++) {
        const s = Math.max(CLIP[0], Math.min(CLIP[1], shift[ic][c]));
        prior[c] *= Math.pow(s, damp);
      }
    }

    // Fix 4: Port suppression BEFORE KT blend
    if (!co) prior[2] = FL;

    // Gentle Blend: strength=30 — prior dominates, observations only nudge
    // Data shows queries HURT us 9/16 rounds with strength=2-6.
    // R13=91.3 with 0 queries. Lookup is our strength. Protect it.
    let p;
    const nObs = counts ? counts[y][x].reduce((a,b) => a+b, 0) : 0;
    if (nObs >= 1) {
      const strength = 30;
      const ktPred = new Array(NC);
      const denom = nObs + strength;
      for (let c = 0; c < NC; c++) ktPred[c] = (counts[y][x][c] + strength * prior[c]) / denom;
      p = ktPred; // with strength=30, n=5 → 86% prior, n=10 → 75% prior
    } else {
      p = prior;
    }

    if (!co) p[2] = FL;

    // Temperature scaling: soften predictions to reduce KL penalty on overconfident errors
    if (TEMP !== 1.0) {
      for (let c = 0; c < NC; c++) p[c] = Math.pow(Math.max(p[c], 1e-10), 1.0 / TEMP);
    }

    // Adaptive floor based on settlement distance (high-entropy cells get higher floor)
    const cellFloor = db === 'near' ? FL_NEAR : db === 'mid' ? FL_MID : db === 'far' ? FL_FAR : FL;
    let s = 0;
    for (let c = 0; c < NC; c++) { p[c] = Math.max(p[c], cellFloor); s += p[c]; }
    for (let c = 0; c < NC; c++) p[c] /= s;
    pred[y][x] = p;
  }

  // Spatial propagation DISABLED — data shows it spreads noisy obs and hurts score

  return pred;
}

// ── Shift computation ───────────────────────────────────────────────────────
function computeShift(obsTrans) {
  const avgRates = {};
  for (let ic = 0; ic < NC; ic++) {
    avgRates[ic] = new Array(NC).fill(0); let n = 0;
    for (const [k, dist] of Object.entries(LOOKUP)) {
      if (parseInt(k.split('_')[0]) === ic) { for (let c = 0; c < NC; c++) avgRates[ic][c] += dist[c]; n++; }
    }
    if (n > 0) for (let c = 0; c < NC; c++) avgRates[ic][c] /= n;
  }
  const shift = {};
  for (let ic = 0; ic < NC; ic++) {
    const tot = obsTrans[ic] ? obsTrans[ic].reduce((a,b) => a+b, 0) : 0;
    if (tot < 15) { shift[ic] = new Array(NC).fill(1); continue; }
    shift[ic] = [];
    for (let c = 0; c < NC; c++) {
      const obs = obsTrans[ic][c] / tot;
      const avg = avgRates[ic][c];
      shift[ic].push(avg > 0.01 ? obs / avg : 1.0);
    }
  }
  return shift;
}

// ── Fix 2: Compute survival from grid ───────────────────────────────────────
function computeSurvivalFromGrid(ig, grid, vpX, vpY, vpH, vpW, H, W) {
  let total = 0, dead = 0;
  for (let gy = 0; gy < vpH; gy++) for (let gx = 0; gx < vpW; gx++) {
    const ay = vpY + gy, ax = vpX + gx;
    if (ay >= H || ax >= W) continue;
    if (ig[ay][ax] === 1 || ig[ay][ax] === 2) {
      total++;
      if (grid[gy][gx] !== 1 && grid[gy][gx] !== 2) dead++;
    }
  }
  return total > 0 ? 1 - dead / total : 1;
}

// ── Weighted round ensemble using transition rate similarity ─────────────────
function weightedRoundLookup(obsTrans, survivalRates, regime) {
  if (Object.keys(ROUND_PROFILES).length === 0) return null;

  // Extract key transition rates from observations (ported from predictor.py)
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
    // Boost with regime survival rate (more accurate than grid-based)
    if (regime) {
      const refSurv = profile['1'] ? (profile['1'][1] + (profile['1'][2]||0)) : 0.5;
      d += Math.abs(regime.aliveRatio - refSurv) * 2;
    }
    const w = 1.0 / (d + 0.01);
    weights[r] = w;
    totalWeight += w;
  }

  if (totalWeight === 0) return null;

  // Blend round-specific lookups by weight
  const blended = {};
  for (const [r, w] of Object.entries(weights)) {
    const rl = ROUND_LOOKUPS[parseInt(r)];
    const nw = w / totalWeight;
    for (const [k, dist] of Object.entries(rl)) {
      if (!blended[k]) blended[k] = new Array(NC).fill(0);
      for (let c = 0; c < NC; c++) blended[k][c] += dist[c] * nw;
    }
  }

  const matchedRounds = Object.keys(weights).map(r => `R${r}(${(weights[r]/totalWeight*100).toFixed(0)}%)`).join('+');
  return { lookup: blended, desc: matchedRounds };
}

// ── Regime detection from settlement metadata ────────────────────────────────
function computeRegime(settlements, survivalRates) {
  if (!settlements.length) return null;
  const alive = settlements.filter(s => s.alive);
  // Use grid-based survival (more accurate than metadata alive field)
  const aliveRatio = survivalRates.length > 0
    ? survivalRates.reduce((a,b) => a+b, 0) / survivalRates.length
    : alive.length / settlements.length;
  const avgFood = alive.length ? alive.reduce((s,a) => s + (a.food||0), 0) / alive.length : 0;
  const avgPop = alive.length ? alive.reduce((s,a) => s + (a.population||0), 0) / alive.length : 0;
  const avgWealth = alive.length ? alive.reduce((s,a) => s + (a.wealth||0), 0) / alive.length : 0;
  const avgDefense = alive.length ? alive.reduce((s,a) => s + (a.defense||0), 0) / alive.length : 0;
  const factions = new Set(alive.map(s => s.owner_id).filter(Boolean));
  const portRatio = alive.length ? alive.filter(s => s.has_port).length / alive.length : 0;
  return { aliveRatio, avgFood, avgPop, avgWealth, avgDefense, factionCount: factions.size, portRatio };
}

// ── Submit ──────────────────────────────────────────────────────────────────
async function submitAll(roundId, detail, H, W, pres, shift, label, cellCounts, lookup) {
  for (let si = 0; si < detail.seeds_count; si++) {
    const counts = cellCounts ? cellCounts[si] : null;
    const pred = predict(detail.initial_states[si].grid, H, W, pres[si], shift, counts, lookup);
    for (let retry = 0; retry < 3; retry++) {
      try {
        await api('POST', '/submit', { round_id: roundId, seed_index: si, prediction: pred });
        await sleep(SUB_DELAY);
        break;
      } catch (e) {
        if (e.message === '429') { await sleep(2000); continue; }
        console.log(`  Submit ${si} err: ${e.message.slice(0,80)}`);
        break;
      }
    }
  }
  console.log(`  ${label}: 5 seeds submitted`);
}

// ── Process round ───────────────────────────────────────────────────────────
async function processRound(round) {
  const t0 = Date.now();
  const log = m => console.log(`[${((Date.now()-t0)/1000).toFixed(0)}s] ${m}`);
  console.log(`\n[${new Date().toISOString().slice(11,19)}] R${round.round_number} ACTIVE`);

  // Reload lookup if changed
  try {
    const fresh = JSON.parse(fs.readFileSync(path.join(__dirname, 'gt_lookup.json'), 'utf8'));
    if (Object.keys(fresh).length !== Object.keys(LOOKUP).length) {
      LOOKUP = fresh;
      log(`Lookup reloaded: ${Object.keys(LOOKUP).length} bins`);
    }
  } catch {}

  const detail = await api('GET', `/rounds/${round.id}`);
  const H = detail.map_height, W = detail.map_width;

  const pres = [];
  for (let si = 0; si < detail.seeds_count; si++) {
    pres.push(precompute(detail.initial_states[si].grid, H, W));
    log(`Seed ${si}: ${pres[si].setts.length} settlements`);
  }

  // ── PHASE 1: Instant lookup ────────────────────────────────────────────
  await submitAll(round.id, detail, H, W, pres, null, 'Phase 1 (lookup)', null, null);

  // Fix 1: Check budget from API
  let budget;
  try { budget = await api('GET', '/budget'); } catch { budget = { queries_used: 0, queries_max: 50 }; }
  let queriesUsed = budget.queries_used || 0;
  const queriesMax = budget.queries_max || 50;
  log(`Queries: ${queriesMax - queriesUsed} available (${queriesUsed}/${queriesMax} used)`);
  if (queriesUsed >= queriesMax) { log('No queries. Done.'); return; }

  // Grid-coverage viewports: 2 per seed — 1 settlement-focused + 1 grid coverage
  // META agent found: empty/forest cells are 60-65% of score, need full map coverage
  const viewports = [];
  for (let si = 0; si < detail.seeds_count; si++) {
    const s = pres[si].setts;
    // VP1: best settlement coverage
    if (s.length > 0) {
      let bestVP = null, bestCount = 0;
      for (const st of s) {
        const vx = Math.max(0, Math.min(W-15, st.x-7)), vy = Math.max(0, Math.min(H-15, st.y-7));
        let count = 0;
        for (const s2 of s) {
          if (s2.x >= vx && s2.x < vx+15 && s2.y >= vy && s2.y < vy+15) count++;
        }
        if (count > bestCount) { bestCount = count; bestVP = { si, x: vx, y: vy }; }
      }
      if (bestVP) viewports.push(bestVP);
    }
    // VP2: grid position that's different from VP1 (cover more of the map)
    const gridPositions = [[0,0],[13,0],[25,0],[0,13],[13,13],[25,13],[0,25],[13,25],[25,25]];
    const vp1 = viewports[viewports.length - 1];
    let bestGrid = null, bestDist = -1;
    for (const [gx, gy] of gridPositions) {
      if (vp1) {
        const dist = Math.abs(gx - vp1.x) + Math.abs(gy - vp1.y);
        if (dist > bestDist) { bestDist = dist; bestGrid = { si, x: gx, y: gy }; }
      } else {
        bestGrid = { si, x: gx, y: gy }; break;
      }
    }
    if (bestGrid) viewports.push(bestGrid);
  }
  log(`Viewports: ${viewports.length} (2 per seed: settlement + grid)`);

  // ── PHASE 2+3: Query loop ─────────────────────────────────────────────
  const obsTrans = {};
  for (let ic = 0; ic < NC; ic++) obsTrans[ic] = new Float64Array(NC);
  const cellCounts = {};
  for (let si = 0; si < detail.seeds_count; si++)
    cellCounts[si] = Array.from({length: H}, () => Array.from({length: W}, () => new Float64Array(NC)));

  const survivalRates = [];
  const allSettlements = [];
  let qCount = 0;

  // Helper: run N queries on given viewports
  async function runQueries(n, vps) {
    let done = 0;
    while (done < n && queriesUsed < queriesMax) {
      const vp = vps[done % vps.length];
      try {
        await sleep(SIM_DELAY);
        const result = await api('POST', '/simulate', {
          round_id: round.id, seed_index: vp.si,
          viewport_x: vp.x, viewport_y: vp.y, viewport_w: 15, viewport_h: 15
        });
        const ig = detail.initial_states[vp.si].grid;
        for (let gy = 0; gy < result.grid.length; gy++)
          for (let gx = 0; gx < result.grid[gy].length; gx++) {
            const ay = result.viewport.y + gy, ax = result.viewport.x + gx;
            if (ay < H && ax < W) {
              const cls = cc(result.grid[gy][gx]);
              obsTrans[cc(ig[ay][ax])][cls]++;
              cellCounts[vp.si][ay][ax][cls]++;
            }
          }
        const surv = computeSurvivalFromGrid(
          ig, result.grid, result.viewport.x, result.viewport.y,
          result.grid.length, result.grid[0].length, H, W
        );
        survivalRates.push(surv);
        if (result.settlements) allSettlements.push(...result.settlements);
        qCount++; done++;
        if (result.queries_used !== undefined) queriesUsed = result.queries_used;
        else queriesUsed++;
      } catch (e) {
        if (e.message === '429') {
          await sleep(2000);
          try {
            budget = await api('GET', '/budget');
            queriesUsed = budget.queries_used || queriesUsed;
            if (queriesUsed >= (budget.queries_max || queriesMax)) return done;
          } catch {}
          continue;
        }
        if (e.message.includes('budget') || e.message.includes('exceeded')) return done;
        log(`Query error: ${e.message.slice(0,80)}`);
        done++;
      }
    }
    return done;
  }

  function doSubmit(label) {
    const shift = computeShift(obsTrans);
    const regime = computeRegime(allSettlements, survivalRates);
    const wrl = weightedRoundLookup(obsTrans, survivalRates, regime);
    const avgSurv = survivalRates.length > 0
      ? (survivalRates.reduce((a,b) => a+b, 0) / survivalRates.length * 100).toFixed(0) : '?';
    log(`${label}: ${qCount}q, survival=${avgSurv}%, match=${wrl ? wrl.desc : '?'}`);
    if (regime) log(`  Regime: alive=${(regime.aliveRatio*100).toFixed(0)}% food=${regime.avgFood.toFixed(2)} pop=${regime.avgPop.toFixed(1)} factions=${regime.factionCount} ports=${(regime.portRatio*100).toFixed(0)}%`);
    return submitAll(round.id, detail, H, W, pres, shift, label, cellCounts, wrl ? wrl.lookup : null);
  }

  // ── PHASE 2: Explore (5q) — 1 per seed, detect regime ────────────────
  const exploreVPs = viewports.filter((_, i) => i % 2 === 0); // settlement viewports only
  await runQueries(5, exploreVPs);
  await doSubmit('Phase 2 (explore 5q)');

  // ── Analyze regime → decide strategy ──────────────────────────────────
  const regime = computeRegime(allSettlements, survivalRates);
  const avgSurv = survivalRates.length > 0
    ? survivalRates.reduce((a,b) => a+b, 0) / survivalRates.length : 1;

  // Harsh world (low survival) → focus on settlements (they change most)
  // Mild world (high survival) → spread wider (settlements stable, edges matter)
  let phase3VPs;
  if (avgSurv < 0.3) {
    // Harsh: concentrate on settlement areas for accurate dead/alive predictions
    phase3VPs = exploreVPs;
    log(`Strategy: HARSH world (survival=${(avgSurv*100).toFixed(0)}%) → concentrate on settlements`);
  } else if (avgSurv > 0.6) {
    // Mild: settlements stable, spread to cover empty/forest transitions
    phase3VPs = viewports; // all viewports including grid coverage
    log(`Strategy: MILD world (survival=${(avgSurv*100).toFixed(0)}%) → spread for coverage`);
  } else {
    // Moderate: balanced approach
    phase3VPs = viewports;
    log(`Strategy: MODERATE world (survival=${(avgSurv*100).toFixed(0)}%) → balanced coverage`);
  }

  // ── PHASE 3: Exploit (10q) — informed by regime ──────────────────────
  await runQueries(10, phase3VPs);
  await doSubmit('Phase 3 (exploit 15q)');

  // ── PHASE 4: Deep observation (15q) — build strong KT estimates ──────
  await runQueries(15, phase3VPs);
  await doSubmit('Phase 4 (deep 30q)');

  // ── PHASE 5: Final push (remaining ~20q) ─────────────────────────────
  const remaining = queriesMax - queriesUsed;
  if (remaining > 0) {
    await runQueries(remaining, phase3VPs);
  }
  await doSubmit('Final');

  log(`Done. ${qCount} queries, ${((Date.now()-t0)/1000).toFixed(0)}s`);
}

// ── Late resubmit: if lookup changed while round still active, resubmit ─────
let lastLookupSize = Object.keys(LOOKUP).length;
const RESUBMITTED = new Set(); // track rounds we've resubmitted

async function checkLateResubmit() {
  try {
    const fresh = JSON.parse(fs.readFileSync(path.join(__dirname, 'gt_lookup.json'), 'utf8'));
    const freshSize = Object.keys(fresh).length;
    if (freshSize === lastLookupSize) return;

    // Lookup changed! Check if there's an active round we already processed
    const rounds = await api('GET', '/rounds');
    const active = rounds.filter(r => r.status === 'active' && DONE.has(r.id) && !RESUBMITTED.has(r.id));
    if (!active.length) { lastLookupSize = freshSize; return; }

    LOOKUP = fresh;
    lastLookupSize = freshSize;
    console.log(`\n[LATE RESUBMIT] Lookup changed: ${freshSize} bins. Resubmitting active rounds...`);

    for (const round of active) {
      const detail = await api('GET', `/rounds/${round.id}`);
      const H = detail.map_height, W = detail.map_width;
      const pres = [];
      for (let si = 0; si < detail.seeds_count; si++)
        pres.push(precompute(detail.initial_states[si].grid, H, W));
      await submitAll(round.id, detail, H, W, pres, null, `Late resubmit (${freshSize} bins)`, null, null);
      RESUBMITTED.add(round.id);
    }
  } catch {}
}

// ── Poll ─────────────────────────────────────────────────────────────────
let processing = false;
async function poll() {
  if (processing) return;

  // Check for late resubmit opportunity
  await checkLateResubmit();

  try {
    const rounds = await api('GET', '/rounds');
    const active = rounds.filter(r => r.status === 'active' && !DONE.has(r.id));
    if (!active.length) { process.stdout.write('.'); return; }
    for (const round of active) {
      processing = true;
      DONE.add(round.id); saveDone();
      try { await processRound(round); }
      catch (e) { console.error(`Round error: ${e.message}`); }
      finally { processing = false; }
    }
  } catch (e) { /* silent */ }
}

console.log(`[init] agent_v7 started. Polling every ${POLL/1000}s...`);
console.log(`[init] Completed: ${DONE.size}`);
poll();
setInterval(poll, POLL);
