#!/usr/bin/env node
/**
 * auto_v6.js — Blended lookup + Monte Carlo agent.
 *
 * Extends v5 with MC simulator predictions:
 * - Phase 1: Submit lookup-only predictions immediately
 * - Phase 2: Query for transitions + run MC simulator in parallel
 * - Phase 3: Blend 60% lookup+shift + 40% MC, resubmit
 *
 * MC runs via subprocess: python3 mc_predict.py --round-id X --seed N
 * One process per seed, all 5 run in parallel.
 *
 * Usage: TOKEN=$(cat ../.env.ainm | cut -d= -f2) node auto_v6.js
 */
const fs = require('fs'), path = require('path'), https = require('https');
const { execSync, spawn } = require('child_process');

const NC = 6;
const TTC = {10:0, 11:0, 0:0, 1:1, 2:2, 3:3, 4:4, 5:5};
const FL = 0.0005;
const DAMP = 0.8;
const COAST_DAMP = 0.3;
const CLIP = [0.1, 10.0];
const SIM_DELAY = 280;
const SUB_DELAY = 600;
const POLL = 30000;

// Blend weight: 60% lookup+shift, 40% MC
const LOOKUP_WEIGHT = 0.60;
const MC_WEIGHT = 0.40;
const MC_RUNS = 200;

const TOKEN = process.env.TOKEN || process.env.AINM_TOKEN;
if (!TOKEN) { console.error('Need TOKEN env var'); process.exit(1); }

// Python path — try common locations
const PYTHON = (() => {
  for (const p of ['/opt/homebrew/bin/python3', '/usr/local/bin/python3', '/usr/bin/python3', 'python3']) {
    try { execSync(`${p} --version`, { stdio: 'pipe' }); return p; } catch {}
  }
  console.warn('[init] python3 not found — MC predictions disabled');
  return null;
})();
const MC_SCRIPT = path.join(__dirname, 'mc_predict.py');

// ── Load lookup ─────────────────────────────────────────────────────────────
// Try v2 (hierarchical Bayesian) first, fall back to v1
const LOOKUP_V2 = path.join(__dirname, 'gt_lookup_v2.json');
const LOOKUP_V1 = path.join(__dirname, 'gt_lookup.json');
const LOOKUP_FILE = fs.existsSync(LOOKUP_V2) ? LOOKUP_V2 : LOOKUP_V1;
let LOOKUP = JSON.parse(fs.readFileSync(LOOKUP_FILE, 'utf8'));
console.log(`[init] Lookup: ${Object.keys(LOOKUP).length} bins (${path.basename(LOOKUP_FILE)})`);

// ── Completed rounds (persisted) ────────────────────────────────────────────
const DONE_FILE = path.join(__dirname, 'v6_completed.json');
let DONE = new Set();
try { DONE = new Set(JSON.parse(fs.readFileSync(DONE_FILE, 'utf8'))); } catch {}
function saveDone() { try { fs.writeFileSync(DONE_FILE, JSON.stringify([...DONE])); } catch {} }

// ── API ─────────────────────────────────────────────────────────────────────
function api(method, p, body) {
  return new Promise((resolve, reject) => {
    const url = new URL('https://api.ainm.no/astar-island' + p);
    const opts = {
      hostname: url.hostname, path: url.pathname, method,
      headers: { 'Authorization': `Bearer ${TOKEN}`, 'Content-Type': 'application/json' }
    };
    const req = https.request(opts, res => {
      let d = ''; res.on('data', c => d += c);
      res.on('end', () => {
        if (res.statusCode === 429) return reject(new Error('429'));
        if (res.statusCode >= 400) return reject(new Error(`${res.statusCode}: ${d.slice(0,100)}`));
        try { resolve(JSON.parse(d)); } catch { resolve(d); }
      });
    });
    req.on('error', reject);
    if (body) req.write(JSON.stringify(body));
    req.end();
  });
}
const sleep = ms => new Promise(r => setTimeout(r, ms));

// ── Spatial (precomputed) ───────────────────────────────────────────────────
function cc(c) { return TTC[c] ?? 0; }

function precompute(ig, H, W) {
  const setts = [];
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++)
    if (ig[y][x] === 1 || ig[y][x] === 2) setts.push({y, x});

  const sd = Array.from({length: H}, () => new Int32Array(W).fill(999));
  for (const s of setts)
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++)
      sd[y][x] = Math.min(sd[y][x], Math.abs(y - s.y) + Math.abs(x - s.x));

  const coastal = Array.from({length: H}, () => new Uint8Array(W));
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    if (ig[y][x] === 10 || ig[y][x] === 5) continue;
    for (const [dy, dx] of [[-1,0],[1,0],[0,-1],[0,1]]) {
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

// ── Predict (lookup + shift) ────────────────────────────────────────────────
function predict(ig, H, W, pre, shift) {
  const pred = Array.from({length: H}, () => Array.from({length: W}, () => new Array(NC)));
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    const raw = ig[y][x];
    if (raw === 5) { pred[y][x] = [FL,FL,FL,FL,FL,1-5*FL]; continue; }
    if (raw === 10) { pred[y][x] = [1-5*FL,FL,FL,FL,FL,FL]; continue; }

    const ic = cc(raw), f = pre.food[y][x], co = pre.coastal[y][x];
    const d = pre.sd[y][x], n = pre.nsett[y][x];
    const db = d <= 3 ? 'near' : d <= 7 ? 'mid' : d <= 12 ? 'far' : 'remote';

    // Hierarchical fallback: finest -> coarsest
    const keys = [
      `${ic}_${f}_${co}_${db}_${n}`,   // level 0: full context
      `${ic}_${f}_${co}_${db}`,          // level 1: drop nsett
      `${ic}_${co}_${db}`,               // level 2: drop food
      `${ic}_${db}`,                      // level 3: drop coastal
      `${ic}`,                            // level 4: IC type only
      // Legacy fallbacks for old lookup format
      `${ic}_${f}_${co}_${db}_0`,
      `${ic}_${Math.min(f,2)}_${co}_${db}_0`,
      `${ic}_0_${co}_${db}_0`,
    ];
    let p = null;
    for (const k of keys) { if (LOOKUP[k]) { p = [...LOOKUP[k]]; break; } }
    if (!p) p = [.5, .1, .05, .05, .25, .05];

    if (shift && shift[ic]) {
      const damp = co ? COAST_DAMP : DAMP;
      for (let c = 0; c < NC; c++) {
        const s = Math.max(CLIP[0], Math.min(CLIP[1], shift[ic][c]));
        p[c] *= Math.pow(s, damp);
      }
    }

    if (!co) p[2] = FL;

    let s = 0;
    for (let c = 0; c < NC; c++) { p[c] = Math.max(p[c], FL); s += p[c]; }
    for (let c = 0; c < NC; c++) p[c] /= s;
    pred[y][x] = p;
  }
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

// ── MC prediction via subprocess ────────────────────────────────────────────
function runMcPredict(roundId, seedIndex) {
  return new Promise((resolve, reject) => {
    if (!PYTHON) return resolve(null);

    const child = spawn(PYTHON, [
      MC_SCRIPT,
      '--round-id', roundId,
      '--seed', String(seedIndex),
      '--runs', String(MC_RUNS),
    ], {
      env: { ...process.env, AINM_TOKEN: TOKEN },
      stdio: ['ignore', 'pipe', 'pipe'],
      timeout: 120000,
    });

    let stdout = '';
    let stderr = '';
    child.stdout.on('data', d => stdout += d);
    child.stderr.on('data', d => stderr += d);

    child.on('close', code => {
      if (code !== 0) {
        console.log(`  MC seed ${seedIndex} failed (code ${code}): ${stderr.slice(0, 100)}`);
        return resolve(null);
      }
      try {
        const parsed = JSON.parse(stdout);
        resolve(parsed);
      } catch (e) {
        console.log(`  MC seed ${seedIndex} parse error: ${e.message.slice(0, 60)}`);
        resolve(null);
      }
    });

    child.on('error', e => {
      console.log(`  MC seed ${seedIndex} spawn error: ${e.message.slice(0, 60)}`);
      resolve(null);
    });
  });
}

/**
 * Run MC predictions for all seeds in parallel.
 * Also saves round detail to cache for the subprocess to read.
 */
async function runAllMcPredictions(roundId, detail) {
  // Save detail to temp cache so subprocess can read it if API fails
  const cacheFile = path.join(__dirname, 'cache', `r_${roundId}_init.json`);
  try {
    fs.mkdirSync(path.join(__dirname, 'cache'), { recursive: true });
    fs.writeFileSync(cacheFile, JSON.stringify(detail));
  } catch {}

  const promises = [];
  for (let si = 0; si < detail.seeds_count; si++) {
    promises.push(runMcPredict(roundId, si));
  }

  const results = await Promise.all(promises);
  return results; // Array of (H,W,6) or null per seed
}

// ── Blend predictions ───────────────────────────────────────────────────────
function blendPredictions(lookupPred, mcPred, H, W) {
  if (!mcPred) return lookupPred;

  const blended = Array.from({length: H}, () => Array.from({length: W}, () => new Array(NC)));

  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    let s = 0;
    for (let c = 0; c < NC; c++) {
      const lv = lookupPred[y][x][c];
      const mv = mcPred[y][x][c];
      let v = LOOKUP_WEIGHT * lv + MC_WEIGHT * mv;
      v = Math.max(v, FL);
      blended[y][x][c] = v;
      s += v;
    }
    // Renormalize
    for (let c = 0; c < NC; c++) blended[y][x][c] /= s;
  }

  return blended;
}

// ── Submit all seeds ────────────────────────────────────────────────────────
async function submitPredictions(roundId, predictions, label) {
  for (let si = 0; si < predictions.length; si++) {
    if (!predictions[si]) continue;
    for (let retry = 0; retry < 3; retry++) {
      try {
        await api('POST', '/submit', { round_id: roundId, seed_index: si, prediction: predictions[si] });
        await sleep(SUB_DELAY);
        break;
      } catch (e) {
        if (e.message === '429') { await sleep(2000); continue; }
        console.log(`  Submit ${si} err: ${e.message.slice(0,60)}`);
        break;
      }
    }
  }
  console.log(`  ${label}: ${predictions.filter(Boolean).length} seeds submitted`);
}

async function submitAll(roundId, detail, H, W, pres, shift, label) {
  const preds = [];
  for (let si = 0; si < detail.seeds_count; si++) {
    preds.push(predict(detail.initial_states[si].grid, H, W, pres[si], shift));
  }
  await submitPredictions(roundId, preds, label);
  return preds;
}

// ── Process round ───────────────────────────────────────────────────────────
async function processRound(round) {
  const t0 = Date.now();
  const log = m => console.log(`[${((Date.now()-t0)/1000).toFixed(0)}s] ${m}`);
  console.log(`\n[${new Date().toISOString().slice(11,19)}] R${round.round_number} ACTIVE (v6 — blended)`);

  // Reload lookup if changed (try v2 first)
  try {
    const lf = fs.existsSync(LOOKUP_V2) ? LOOKUP_V2 : LOOKUP_V1;
    const fresh = JSON.parse(fs.readFileSync(lf, 'utf8'));
    if (Object.keys(fresh).length !== Object.keys(LOOKUP).length) {
      LOOKUP = fresh;
      log(`Lookup reloaded: ${Object.keys(LOOKUP).length} bins (${path.basename(lf)})`);
    }
  } catch {}

  const detail = await api('GET', `/rounds/${round.id}`);
  const H = detail.map_height, W = detail.map_width;

  // Precompute spatial features for all seeds
  const pres = [];
  for (let si = 0; si < detail.seeds_count; si++) {
    pres.push(precompute(detail.initial_states[si].grid, H, W));
    log(`Seed ${si}: ${pres[si].setts.length} settlements`);
  }

  // PHASE 1: Submit lookup immediately
  const lookupPreds = await submitAll(round.id, detail, H, W, pres, null, 'Phase 1 (lookup)');

  // PHASE 2: Start MC predictions in parallel with queries
  log('Starting MC predictions in parallel...');
  const mcPromise = runAllMcPredictions(round.id, detail);

  // PHASE 3: Query for transitions (same as v5)
  let budget;
  try { budget = await api('GET', '/budget'); } catch { budget = { queries_used: 0, queries_max: 50 }; }
  const left = budget.queries_max - budget.queries_used;
  log(`Queries available: ${left}`);

  const viewports = [];
  for (let si = 0; si < detail.seeds_count; si++) {
    const s = pres[si].setts;
    if (s.length === 0) continue;
    const cx = Math.round(s.reduce((a,b) => a+b.x, 0) / s.length);
    const cy = Math.round(s.reduce((a,b) => a+b.y, 0) / s.length);
    viewports.push({ si, x: Math.max(0, Math.min(W-15, cx-7)), y: Math.max(0, Math.min(H-15, cy-7)) });
  }

  const obsTrans = {};
  for (let ic = 0; ic < NC; ic++) obsTrans[ic] = new Float64Array(NC);
  let qCount = 0, rlCount = 0;

  for (let qi = 0; qi < left && left > 0; qi++) {
    const vp = viewports[qi % viewports.length];
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
          if (ay < H && ax < W) obsTrans[cc(ig[ay][ax])][cc(result.grid[gy][gx])]++;
        }
      qCount++;
    } catch (e) {
      if (e.message === '429') {
        rlCount++;
        if (rlCount > 15) { log(`Too many rate limits (${rlCount}). Stopping.`); break; }
        await sleep(2000);
        qi--; continue;
      }
      if (e.message.includes('budget')) { log('Budget exhausted.'); break; }
      log(`Query err: ${e.message.slice(0,50)}`);
    }

    // Resubmit every 10 queries (lookup+shift only, MC not ready yet maybe)
    if (qCount > 0 && qCount % 10 === 0) {
      const shift = computeShift(obsTrans);
      await submitAll(round.id, detail, H, W, pres, shift, `Phase 3 (${qCount}q shift)`);
    }
  }

  // Wait for MC to finish
  log('Waiting for MC predictions...');
  const mcResults = await mcPromise;
  const mcOk = mcResults.filter(Boolean).length;
  log(`MC predictions ready: ${mcOk}/${detail.seeds_count} seeds`);

  // PHASE 4: Final blended submission
  const shift = qCount > 0 ? computeShift(obsTrans) : null;
  const finalPreds = [];

  for (let si = 0; si < detail.seeds_count; si++) {
    const lookupPred = predict(detail.initial_states[si].grid, H, W, pres[si], shift);
    const mcPred = mcResults[si];

    if (mcPred) {
      finalPreds.push(blendPredictions(lookupPred, mcPred, H, W));
    } else {
      finalPreds.push(lookupPred);
    }
  }

  await submitPredictions(round.id, finalPreds, `Final blended (${qCount}q + ${mcOk}mc)`);

  log(`Done. ${qCount} queries, ${mcOk} MC seeds, ${((Date.now()-t0)/1000).toFixed(0)}s`);
}

// ── Poll loop ───────────────────────────────────────────────────────────────
let processing = false;

async function poll() {
  if (processing) return;
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
  } catch (e) { /* silent poll error */ }
}

console.log(`[init] v6 started (blended: ${LOOKUP_WEIGHT*100}% lookup + ${MC_WEIGHT*100}% MC)`);
console.log(`[init] Python: ${PYTHON || 'NOT FOUND'}`);
console.log(`[init] MC runs: ${MC_RUNS}`);
console.log(`[init] Lookup: ${Object.keys(LOOKUP).length} bins`);
console.log(`[init] Completed rounds: ${DONE.size}`);
poll();
setInterval(poll, POLL);
