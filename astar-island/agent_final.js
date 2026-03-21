#!/usr/bin/env node
/**
 * agent_final.js — Best-in-class Astar Island agent.
 *
 * 3-phase strategy:
 * 1. INSTANT: Submit lookup predictions (0s)
 * 2. EXPLORE: 10 queries (2/seed) → infer hidden params from settlement metadata
 *    → select best matching historical round → resubmit with round-specific lookup
 * 3. EXPLOIT: 40 queries focused on settlements → per-cell KT with shifted prior
 *    → progressive resubmit every 10 queries
 *
 * Cross-seed: All observations contribute to global shift for ALL seeds.
 * Per-cell: Observed cells (n≥3) use Bayesian KT estimator.
 */
const fs = require('fs'), path = require('path'), https = require('https');

const NC = 6;
const TTC = {10:0, 11:0, 0:0, 1:1, 2:2, 3:3, 4:4, 5:5};
const FL = 0.0005;
const DAMP = 0.8;
const COAST_DAMP = 0.3;
const CLIP = [0.1, 10.0];
const KT_ALPHA = 2.0;
const KT_MIN_OBS = 3;
const SIM_DELAY = 280;
const SUB_DELAY = 550;
const POLL = 30000;

const TOKEN = process.env.TOKEN || process.env.AINM_TOKEN;
if (!TOKEN) { console.error('Need TOKEN'); process.exit(1); }

// ── Load data ───────────────────────────────────────────────────────────────
const LOOKUP = JSON.parse(fs.readFileSync(path.join(__dirname, 'gt_lookup.json'), 'utf8'));

// Per-round transition profiles (for round matching)
const CACHE = path.join(__dirname, 'cache');
const ROUND_PROFILES = {};
try {
  for (const f of fs.readdirSync(CACHE).filter(f => f.match(/^transitions_r\d+\.json$/))) {
    const r = parseInt(f.match(/r(\d+)/)[1]);
    ROUND_PROFILES[r] = JSON.parse(fs.readFileSync(path.join(CACHE, f), 'utf8'));
  }
} catch {}

// Per-round lookups (for round-specific predictions)
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
        const raw = ig[y][x], ic = cc(raw);
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
const DONE_FILE = path.join(__dirname, 'final_completed.json');
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

// ── Spatial ─────────────────────────────────────────────────────────────────
function cc(c) { return TTC[c] ?? 0; }

function cellKey(ig, H, W, y, x) {
  const raw = ig[y][x], ic = cc(raw);
  // Food
  let food = 0;
  for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
    if (!dy && !dx) continue;
    const ny = y+dy, nx = x+dx;
    if (ny >= 0 && ny < H && nx >= 0 && nx < W && ig[ny][nx] === 4) food++;
  }
  food = Math.min(food, 4);
  // Coastal
  let co = 0;
  for (const [dy,dx] of [[-1,0],[1,0],[0,-1],[0,1]]) {
    const ny = y+dy, nx = x+dx;
    if (ny >= 0 && ny < H && nx >= 0 && nx < W && ig[ny][nx] === 10) { co = 1; break; }
  }
  if (ig[y][x] === 10 || ig[y][x] === 5) co = 0;
  // Settlement distance
  let sd = 999;
  for (let sy = 0; sy < H; sy++) for (let sx = 0; sx < W; sx++)
    if (ig[sy][sx] === 1 || ig[sy][sx] === 2) sd = Math.min(sd, Math.abs(y-sy) + Math.abs(x-sx));
  const db = sd <= 3 ? 'near' : sd <= 7 ? 'mid' : sd <= 12 ? 'far' : 'remote';
  // Neighbor settlements
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

    // Apply shift to prior
    if (shift && shift[ic]) {
      const damp = co ? COAST_DAMP : DAMP;
      for (let c = 0; c < NC; c++) {
        const s = Math.max(CLIP[0], Math.min(CLIP[1], shift[ic][c]));
        prior[c] *= Math.pow(s, damp);
      }
    }

    // Per-cell KT: data + shifted prior
    let p;
    const nObs = counts ? counts[y][x].reduce((a,b) => a+b, 0) : 0;
    if (nObs >= KT_MIN_OBS) {
      p = new Array(NC);
      const denom = nObs + KT_ALPHA;
      for (let c = 0; c < NC; c++) p[c] = (counts[y][x][c] + KT_ALPHA * prior[c]) / denom;
    } else {
      p = prior;
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

// ── Round matching from metadata ────────────────────────────────────────────
function matchRound(metadata) {
  if (!metadata.length || Object.keys(ROUND_PROFILES).length === 0) return null;

  const alive = metadata.filter(s => s.alive);
  const dead = metadata.filter(s => !s.alive);
  const survivalRate = alive.length / Math.max(metadata.length, 1);
  const avgFood = alive.length ? alive.reduce((s,a) => s + (a.food||0), 0) / alive.length : 0;
  const portRate = alive.length ? alive.filter(s => s.has_port).length / alive.length : 0;

  // Match against historical round profiles
  let bestRound = null, bestDist = Infinity;
  for (const [r, profile] of Object.entries(ROUND_PROFILES)) {
    if (!profile['1']) continue; // no settlement data
    const histSurvival = profile['1'][1] + (profile['1'][2] || 0); // sett + port survival
    const dist = Math.abs(survivalRate - histSurvival) * 3 + Math.abs(portRate - (profile['1'][2] || 0));
    if (dist < bestDist) { bestDist = dist; bestRound = parseInt(r); }
  }
  return bestRound;
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
        console.log(`  Submit ${si} err: ${e.message.slice(0,50)}`);
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
      Object.assign(LOOKUP, fresh);
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

  let budget;
  try { budget = await api('GET', '/budget'); } catch { budget = { queries_used: 0, queries_max: 50 }; }
  const left = budget.queries_max - budget.queries_used;
  log(`Queries: ${left} available`);
  if (left <= 0) { log('No queries. Done.'); return; }

  // Plan viewports: 2-3 per seed covering most settlements
  const viewports = [];
  for (let si = 0; si < detail.seeds_count; si++) {
    const s = pres[si].setts;
    if (s.length === 0) continue;
    // Cluster settlements into 2-3 viewports
    const used = new Set();
    for (let vp = 0; vp < 3 && used.size < s.length; vp++) {
      let bestVP = null, bestCount = 0;
      for (const st of s) {
        if (used.has(`${st.y},${st.x}`)) continue;
        const vx = Math.max(0, Math.min(W-15, st.x-7)), vy = Math.max(0, Math.min(H-15, st.y-7));
        let count = 0;
        for (const s2 of s) {
          if (s2.x >= vx && s2.x < vx+15 && s2.y >= vy && s2.y < vy+15) count++;
        }
        if (count > bestCount) { bestCount = count; bestVP = { si, x: vx, y: vy }; }
      }
      if (!bestVP) break;
      viewports.push(bestVP);
      for (const s2 of s) {
        if (s2.x >= bestVP.x && s2.x < bestVP.x+15 && s2.y >= bestVP.y && s2.y < bestVP.y+15)
          used.add(`${s2.y},${s2.x}`);
      }
    }
  }
  log(`Viewports: ${viewports.length} (covering settlements)`);

  // ── PHASE 2: Explore (first 10 queries) ────────────────────────────────
  const obsTrans = {};
  for (let ic = 0; ic < NC; ic++) obsTrans[ic] = new Float64Array(NC);
  const cellCounts = {};
  for (let si = 0; si < detail.seeds_count; si++)
    cellCounts[si] = Array.from({length: H}, () => Array.from({length: W}, () => new Float64Array(NC)));
  const allMetadata = [];

  let qCount = 0, rlCount = 0;
  const EXPLORE = Math.min(10, left);

  for (let qi = 0; qi < left; qi++) {
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
          if (ay < H && ax < W) {
            const cls = cc(result.grid[gy][gx]);
            obsTrans[cc(ig[ay][ax])][cls]++;
            cellCounts[vp.si][ay][ax][cls]++;
          }
        }

      // Collect settlement metadata
      if (result.settlements) allMetadata.push(...result.settlements);
      qCount++;
    } catch (e) {
      if (e.message === '429') {
        rlCount++;
        if (rlCount > 15) { log(`Too many rate limits. Stopping.`); break; }
        await sleep(2000); qi--; continue;
      }
      if (e.message.includes('budget')) { log('Budget exhausted.'); break; }
    }

    // After explore phase: match round + resubmit
    if (qCount === EXPLORE) {
      const shift = computeShift(obsTrans);
      const matched = matchRound(allMetadata);
      const roundLookup = matched && ROUND_LOOKUPS[matched] ? ROUND_LOOKUPS[matched] : null;
      log(`Explore done. Matched round: R${matched || '?'}. Metadata: ${allMetadata.length} settlements`);
      if (allMetadata.length > 0) {
        const alive = allMetadata.filter(s => s.alive);
        log(`  Survival: ${(100*alive.length/allMetadata.length).toFixed(0)}% | Avg food: ${alive.length ? (alive.reduce((s,a)=>s+(a.food||0),0)/alive.length).toFixed(2) : '?'}`);
      }
      await submitAll(round.id, detail, H, W, pres, shift, `Phase 2 (explore ${qCount}q, match=R${matched||'?'})`, cellCounts, roundLookup);
    }

    // Progressive resubmit every 10 queries during exploit
    if (qCount > EXPLORE && (qCount - EXPLORE) % 10 === 0) {
      const shift = computeShift(obsTrans);
      const matched = matchRound(allMetadata);
      const roundLookup = matched && ROUND_LOOKUPS[matched] ? ROUND_LOOKUPS[matched] : null;
      await submitAll(round.id, detail, H, W, pres, shift, `Phase 3 (exploit ${qCount}q)`, cellCounts, roundLookup);
    }
  }

  // Final resubmit
  if (qCount > 0) {
    const shift = computeShift(obsTrans);
    const matched = matchRound(allMetadata);
    const roundLookup = matched && ROUND_LOOKUPS[matched] ? ROUND_LOOKUPS[matched] : null;
    await submitAll(round.id, detail, H, W, pres, shift, `Final (${qCount}q, R${matched||'?'})`, cellCounts, roundLookup);
  }

  log(`Done. ${qCount} queries, ${allMetadata.length} metadata, ${((Date.now()-t0)/1000).toFixed(0)}s`);
}

// ── Poll ─────────────────────────────────────────────────────────────────
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
  } catch (e) { /* silent */ }
}

console.log(`[init] agent_final started. Polling every ${POLL/1000}s...`);
console.log(`[init] Completed: ${DONE.size}`);
poll();
setInterval(poll, POLL);
