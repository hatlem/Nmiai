#!/usr/bin/env node
/**
 * auto_v5.js — Final production agent.
 *
 * Lessons learned from v1-v4:
 * - NEVER crash (wrap everything in try/catch)
 * - Persist completed rounds to file (survive restarts)
 * - Never let queries go to waste
 * - Shift ALWAYS helps (+2-12 points)
 * - Floor 0.001 > 0.002
 * - XGBoost lookup (241 bins) > old lookup (209 bins)
 * - Settlement metadata is free information
 * - Progressive resubmit every 10 queries
 * - Rate limit: 250ms between simulate, 600ms between submit
 *
 * Usage: TOKEN=$(cat ../.env.ainm | cut -d= -f2) node auto_v5.js
 */
const fs = require('fs'), path = require('path'), https = require('https');

const NC = 6;
const TTC = {10:0, 11:0, 0:0, 1:1, 2:2, 3:3, 4:4, 5:5};
const FL = 0.001;
const DAMP = 0.7;
const COAST_DAMP = 0.5;
const CLIP = [0.5, 3.0];
const SIM_DELAY = 280;
const SUB_DELAY = 600;
const POLL = 30000;

const TOKEN = process.env.TOKEN || process.env.AINM_TOKEN;
if (!TOKEN) { console.error('Need TOKEN env var'); process.exit(1); }

// ── Load lookup ─────────────────────────────────────────────────────────────
const LOOKUP_FILE = path.join(__dirname, 'gt_lookup.json');
let LOOKUP = JSON.parse(fs.readFileSync(LOOKUP_FILE, 'utf8'));
console.log(`[init] Lookup: ${Object.keys(LOOKUP).length} bins`);

// ── Completed rounds (persisted) ────────────────────────────────────────────
const DONE_FILE = path.join(__dirname, 'v5_completed.json');
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
  // Settlement positions
  const setts = [];
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++)
    if (ig[y][x] === 1 || ig[y][x] === 2) setts.push({y, x});

  // Settlement distance (vectorized with early exit)
  const sd = Array.from({length: H}, () => new Int32Array(W).fill(999));
  for (const s of setts)
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++)
      sd[y][x] = Math.min(sd[y][x], Math.abs(y - s.y) + Math.abs(x - s.x));

  // Coastal
  const coastal = Array.from({length: H}, () => new Uint8Array(W));
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    if (ig[y][x] === 10 || ig[y][x] === 5) continue;
    for (const [dy, dx] of [[-1,0],[1,0],[0,-1],[0,1]]) {
      const ny = y+dy, nx = x+dx;
      if (ny >= 0 && ny < H && nx >= 0 && nx < W && ig[ny][nx] === 10) { coastal[y][x] = 1; break; }
    }
  }

  // Food (forest neighbors)
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

  // Neighbor settlements (radius 2)
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

// ── Predict ─────────────────────────────────────────────────────────────────
function predict(ig, H, W, pre, shift) {
  const pred = Array.from({length: H}, () => Array.from({length: W}, () => new Array(NC)));
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    const raw = ig[y][x];
    if (raw === 5) { pred[y][x] = [FL,FL,FL,FL,FL,1-5*FL]; continue; }
    if (raw === 10) { pred[y][x] = [1-5*FL,FL,FL,FL,FL,FL]; continue; }

    const ic = cc(raw), f = pre.food[y][x], co = pre.coastal[y][x];
    const d = pre.sd[y][x], n = pre.nsett[y][x];
    const db = d <= 3 ? 'near' : d <= 7 ? 'mid' : d <= 12 ? 'far' : 'remote';

    // Lookup with fallback
    const keys = [`${ic}_${f}_${co}_${db}_${n}`, `${ic}_${f}_${co}_${db}_0`,
                   `${ic}_${Math.min(f,2)}_${co}_${db}_0`, `${ic}_0_${co}_${db}_0`];
    let p = null;
    for (const k of keys) { if (LOOKUP[k]) { p = [...LOOKUP[k]]; break; } }
    if (!p) p = [.5, .1, .05, .05, .25, .05];

    // Shift
    if (shift && shift[ic]) {
      const damp = co ? COAST_DAMP : DAMP;
      for (let c = 0; c < NC; c++) {
        const s = Math.max(CLIP[0], Math.min(CLIP[1], shift[ic][c]));
        p[c] *= Math.pow(s, damp);
      }
    }

    // Port suppression for non-coastal
    if (!co) p[2] = FL;

    // Floor + normalize
    let s = 0;
    for (let c = 0; c < NC; c++) { p[c] = Math.max(p[c], FL); s += p[c]; }
    for (let c = 0; c < NC; c++) p[c] /= s;
    pred[y][x] = p;
  }
  return pred;
}

// ── Shift computation ───────────────────────────────────────────────────────
function computeShift(obsTrans) {
  // Average rates from lookup
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

// ── Submit all seeds ────────────────────────────────────────────────────────
async function submitAll(roundId, detail, H, W, pres, shift, label) {
  for (let si = 0; si < detail.seeds_count; si++) {
    const pred = predict(detail.initial_states[si].grid, H, W, pres[si], shift);
    for (let retry = 0; retry < 3; retry++) {
      try {
        await api('POST', '/submit', { round_id: roundId, seed_index: si, prediction: pred });
        await sleep(SUB_DELAY);
        break;
      } catch (e) {
        if (e.message === '429') { await sleep(2000); continue; }
        console.log(`  Submit ${si} err: ${e.message.slice(0,60)}`);
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
    const fresh = JSON.parse(fs.readFileSync(LOOKUP_FILE, 'utf8'));
    if (Object.keys(fresh).length !== Object.keys(LOOKUP).length) {
      LOOKUP = fresh;
      log(`Lookup reloaded: ${Object.keys(LOOKUP).length} bins`);
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
  await submitAll(round.id, detail, H, W, pres, null, 'Phase 1 (lookup)');

  // PHASE 2: Query
  let budget;
  try { budget = await api('GET', '/budget'); } catch { budget = { queries_used: 0, queries_max: 50 }; }
  const left = budget.queries_max - budget.queries_used;
  log(`Queries available: ${left}`);
  if (left <= 0) { log('No queries. Done.'); return; }

  // Plan: 1 viewport per seed centered on settlement cluster, repeat
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

  for (let qi = 0; qi < left; qi++) {
    const vp = viewports[qi % viewports.length];
    try {
      await sleep(SIM_DELAY);
      const result = await api('POST', '/simulate', {
        round_id: round.id, seed_index: vp.si,
        viewport_x: vp.x, viewport_y: vp.y, viewport_w: 15, viewport_h: 15
      });

      // Accumulate transitions
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

    // Resubmit every 10 queries
    if (qCount > 0 && qCount % 10 === 0) {
      const shift = computeShift(obsTrans);
      await submitAll(round.id, detail, H, W, pres, shift, `Phase 2 (${qCount}q)`);
    }
  }

  // Final resubmit
  if (qCount > 0 && qCount % 10 !== 0) {
    const shift = computeShift(obsTrans);
    await submitAll(round.id, detail, H, W, pres, shift, `Final (${qCount}q)`);
  }

  log(`Done. ${qCount} queries, ${((Date.now()-t0)/1000).toFixed(0)}s`);
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

console.log(`[init] v5 started. Polling every ${POLL/1000}s...`);
console.log(`[init] Completed rounds: ${DONE.size}`);
poll();
setInterval(poll, POLL);
