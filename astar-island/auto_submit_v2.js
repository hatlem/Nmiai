#!/usr/bin/env node
/**
 * Auto-submit v2: GT-lookup + observation-adaptive predictions.
 *
 * Strategy:
 * 1. Submit lookup-only predictions immediately (fast, ~82 score)
 * 2. Query simulator (50 queries focused on settlements)
 * 3. Estimate per-round transition rates from observations
 * 4. Blend lookup with per-round adapted predictions
 * 5. Resubmit improved predictions
 *
 * Usage: TOKEN=$(cat ../.env.ainm | cut -d= -f2) node auto_submit_v2.js
 */
const fs = require('fs');
const path = require('path');
const https = require('https');

const API = 'https://api.ainm.no/astar-island';
const TOKEN = process.env.TOKEN || process.env.AINM_TOKEN;
const POLL_INTERVAL = 30000;
const API_DELAY = 220; // ms between API calls (5 req/s limit)
const NC = 6;
const TTC = {10:0,11:0,0:0,1:1,2:2,3:3,4:4,5:5};
const CACHE_DIR = path.join(__dirname, 'cache');

if (!TOKEN) { console.error('Set TOKEN env var'); process.exit(1); }

const LOOKUP_PATH = path.join(__dirname, 'gt_lookup.json');
let LOOKUP = JSON.parse(fs.readFileSync(LOOKUP_PATH, 'utf8'));
console.log(`Loaded ${Object.keys(LOOKUP).length} context bins`);

// Also load per-round transition matrices for adaptation
let TRANSITIONS = {};
try {
  const files = fs.readdirSync(CACHE_DIR).filter(f => f.match(/^transitions_r\d+\.json$/));
  for (const f of files) {
    const r = parseInt(f.match(/r(\d+)/)[1]);
    TRANSITIONS[r] = JSON.parse(fs.readFileSync(path.join(CACHE_DIR, f), 'utf8'));
  }
  console.log(`Loaded transitions for ${Object.keys(TRANSITIONS).length} rounds`);
} catch {}

function reloadLookup() {
  try {
    LOOKUP = JSON.parse(fs.readFileSync(LOOKUP_PATH, 'utf8'));
    console.log(`Reloaded lookup: ${Object.keys(LOOKUP).length} bins`);
  } catch {}
}

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
        if (res.statusCode >= 400) return reject(new Error(`${res.statusCode}: ${data.slice(0,200)}`));
        try { resolve(JSON.parse(data)); } catch { resolve(data); }
      });
    });
    req.on('error', reject);
    if (body) req.write(JSON.stringify(body));
    req.end();
  });
}
const sleep = ms => new Promise(r => setTimeout(r, ms));

// ── Spatial helpers ──────────────────────────────────────────────────────────
function cc(c){return TTC[c]??0}
function isCoast(g,H,W,y,x){if(g[y][x]===10||g[y][x]===5)return false;for(const[dy,dx]of[[-1,0],[1,0],[0,-1],[0,1]]){const ny=y+dy,nx=x+dx;if(ny>=0&&ny<H&&nx>=0&&nx<W&&g[ny][nx]===10)return true}return false}
function foodPot(g,H,W,y,x){let c=0;for(let dy=-1;dy<=1;dy++)for(let dx=-1;dx<=1;dx++){if(!dy&&!dx)continue;const ny=y+dy,nx=x+dx;if(ny>=0&&ny<H&&nx>=0&&nx<W&&g[ny][nx]===4)c++}return c}
function settDist(g,H,W,y,x){let m=999;for(let sy=0;sy<H;sy++)for(let sx=0;sx<W;sx++)if(g[sy][sx]===1||g[sy][sx]===2)m=Math.min(m,Math.abs(y-sy)+Math.abs(x-sx));return m}
function nSett(g,H,W,y,x){let c=0;for(let dy=-2;dy<=2;dy++)for(let dx=-2;dx<=2;dx++){if(!dy&&!dx)continue;const ny=y+dy,nx=x+dx;if(ny>=0&&ny<H&&nx>=0&&nx<W){const v=g[ny][nx];if(v===1||v===2)c++}}return c}

function ctxKey(ig,H,W,y,x) {
  const ic=cc(ig[y][x]),f=Math.min(foodPot(ig,H,W,y,x),4);
  const co=isCoast(ig,H,W,y,x)?1:0,d=settDist(ig,H,W,y,x);
  const n=Math.min(nSett(ig,H,W,y,x),3);
  const db=d<=3?'near':d<=7?'mid':d<=12?'far':'remote';
  return `${ic}_${f}_${co}_${db}_${n}`;
}

function lookupPred(ig, H, W) {
  const pred = [];
  for(let y=0;y<H;y++){pred[y]=[];for(let x=0;x<W;x++){
    const raw=ig[y][x];
    if(raw===5){pred[y][x]=[.002,.002,.002,.002,.002,.990];continue}
    if(raw===10){pred[y][x]=[.990,.002,.002,.002,.002,.002];continue}
    const key = ctxKey(ig,H,W,y,x);
    const ic=cc(raw),f=Math.min(foodPot(ig,H,W,y,x),4);
    const co=isCoast(ig,H,W,y,x)?1:0;
    const d=settDist(ig,H,W,y,x);
    const db=d<=3?'near':d<=7?'mid':d<=12?'far':'remote';
    const keys=[key,`${ic}_${f}_${co}_${db}_0`,`${ic}_${Math.min(f,2)}_${co}_${db}_0`,`${ic}_0_${co}_${db}_0`];
    let p=null;
    for(const k of keys){if(LOOKUP[k]){p=[...LOOKUP[k]];break}}
    if(!p)p=[.5,.1,.05,.05,.25,.05];
    if(!co)p[2]=.002;
    let s=0;for(let c=0;c<NC;c++){p[c]=Math.max(p[c],.002);s+=p[c]}
    for(let c=0;c<NC;c++)p[c]/=s;
    pred[y][x]=p;
  }}
  return pred;
}

// ── Query strategy: focused on settlements ───────────────────────────────────

function planQueries(initialStates, H, W, budget) {
  const plan = [];
  const seedsCount = initialStates.length;
  const VP = 15;

  // For each seed, find viewports that cover settlements
  for (let si = 0; si < seedsCount; si++) {
    const setts = initialStates[si].settlements || [];
    if (setts.length === 0) continue;

    // Cluster settlements into viewport-sized groups
    const covered = new Set();
    const viewports = [];

    while (covered.size < setts.length) {
      // Find uncovered settlement furthest from any covered
      let bestVP = null, bestCount = 0;
      for (const s of setts) {
        if (covered.has(`${s.x},${s.y}`)) continue;
        const vx = Math.max(0, Math.min(W-VP, s.x - 7));
        const vy = Math.max(0, Math.min(H-VP, s.y - 7));
        let count = 0;
        for (const s2 of setts) {
          if (s2.x >= vx && s2.x < vx+VP && s2.y >= vy && s2.y < vy+VP) count++;
        }
        if (count > bestCount) { bestCount = count; bestVP = {x:vx, y:vy, si}; }
      }
      if (!bestVP) break;
      viewports.push(bestVP);
      for (const s of setts) {
        if (s.x >= bestVP.x && s.x < bestVP.x+VP && s.y >= bestVP.y && s.y < bestVP.y+VP)
          covered.add(`${s.x},${s.y}`);
      }
    }

    // Add viewports to plan
    for (const vp of viewports) plan.push({si, x:vp.x, y:vp.y, w:VP, h:VP});
  }

  // Fill remaining budget with repeated observations on settlement viewports
  const firstPass = plan.length;
  const remaining = budget - firstPass;
  if (remaining > 0 && firstPass > 0) {
    for (let i = 0; i < remaining; i++) {
      plan.push(plan[i % firstPass]); // cycle through settlement viewports
    }
  }

  return plan.slice(0, budget);
}

// ── Observation-based adaptation ─────────────────────────────────────────────

function estimateRoundShift(observations, initialStates, H, W) {
  // Compute observed transition rates for this round
  // Compare with lookup average → compute shift multipliers per class
  const obsTrans = {}; // initClass → [count per final class]
  for (let c = 0; c < NC; c++) obsTrans[c] = new Float64Array(NC);

  for (const obs of observations) {
    const {si, grid: obsGrid, viewport} = obs;
    const ig = initialStates[si].grid;
    for (let gy = 0; gy < obsGrid.length; gy++) {
      for (let gx = 0; gx < obsGrid[gy].length; gx++) {
        const ay = viewport.y + gy, ax = viewport.x + gx;
        if (ay >= H || ax >= W) continue;
        const ic = cc(ig[ay][ax]);
        const fc = cc(obsGrid[gy][gx]);
        obsTrans[ic][fc]++;
      }
    }
  }

  // Compute observed rates
  const obsRates = {};
  for (let ic = 0; ic < NC; ic++) {
    const total = obsTrans[ic].reduce((a,b) => a+b, 0);
    if (total < 10) continue; // not enough data
    obsRates[ic] = [];
    for (let c = 0; c < NC; c++) obsRates[ic].push(obsTrans[ic][c] / total);
  }

  // Compare with average transition from all historical rounds
  const avgTrans = {};
  const roundNums = Object.keys(TRANSITIONS).map(Number);
  if (roundNums.length === 0) return null;

  for (let ic = 0; ic < NC; ic++) {
    avgTrans[ic] = new Float64Array(NC);
    let count = 0;
    for (const r of roundNums) {
      if (TRANSITIONS[r][ic]) {
        for (let c = 0; c < NC; c++) avgTrans[ic][c] += TRANSITIONS[r][ic][c];
        count++;
      }
    }
    if (count > 0) for (let c = 0; c < NC; c++) avgTrans[ic][c] /= count;
  }

  // Compute shift: ratio of observed/average
  const shift = {};
  for (const [ic, rates] of Object.entries(obsRates)) {
    shift[ic] = [];
    const avg = avgTrans[ic];
    for (let c = 0; c < NC; c++) {
      if (avg[c] > 0.005) shift[ic].push(rates[c] / avg[c]);
      else shift[ic].push(1.0); // no shift for rare classes
    }
  }

  return shift;
}

function adaptPredictions(basePred, shift, ig, H, W) {
  if (!shift) return basePred;

  const pred = basePred.map(row => row.map(p => [...p]));
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const raw = ig[y][x];
      if (raw === 5 || raw === 10) continue;
      const ic = cc(raw);
      if (!shift[ic]) continue;

      // Apply shift: multiply each class probability by the shift ratio
      // Then renormalize
      for (let c = 0; c < NC; c++) {
        pred[y][x][c] *= shift[ic][c];
      }
      let s = 0;
      for (let c = 0; c < NC; c++) { pred[y][x][c] = Math.max(pred[y][x][c], 0.002); s += pred[y][x][c]; }
      for (let c = 0; c < NC; c++) pred[y][x][c] /= s;
    }
  }
  return pred;
}

// ── Also blend KT estimates for observed cells ───────────────────────────────

function blendWithObservations(pred, observations, initialStates, H, W, seedIdx) {
  // Build per-cell observation counts for this seed
  const counts = Array.from({length: H}, () => Array.from({length: W}, () => new Float64Array(NC)));

  for (const obs of observations) {
    if (obs.si !== seedIdx) continue;
    for (let gy = 0; gy < obs.grid.length; gy++) {
      for (let gx = 0; gx < obs.grid[gy].length; gx++) {
        const ay = obs.viewport.y + gy, ax = obs.viewport.x + gx;
        if (ay >= H || ax >= W) continue;
        counts[ay][ax][cc(obs.grid[gy][gx])]++;
      }
    }
  }

  const result = pred.map(row => row.map(p => [...p]));
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const nObs = counts[y][x].reduce((a,b) => a+b, 0);
      if (nObs === 0) continue;

      // KT estimator with lookup as prior
      const alpha = 0.5; // prior strength
      const kt = new Float64Array(NC);
      const denom = nObs + NC * alpha;
      for (let c = 0; c < NC; c++) {
        kt[c] = (counts[y][x][c] + alpha * pred[y][x][c] * NC) / denom;
      }

      // Blend: weight observations by n/(n+3)
      const w = nObs / (nObs + 3);
      for (let c = 0; c < NC; c++) {
        result[y][x][c] = w * kt[c] + (1-w) * pred[y][x][c];
      }
      let s = 0;
      for (let c = 0; c < NC; c++) { result[y][x][c] = Math.max(result[y][x][c], 0.002); s += result[y][x][c]; }
      for (let c = 0; c < NC; c++) result[y][x][c] /= s;
    }
  }
  return result;
}

// ── Main pipeline ────────────────────────────────────────────────────────────

const completed = new Set();

async function processRound(round) {
  const t0 = Date.now();
  console.log(`\n[${new Date().toISOString().slice(11,19)}] Round ${round.round_number} active!`);

  reloadLookup();
  const detail = await apiCall('GET', `/rounds/${round.id}`);
  const H = detail.map_height, W = detail.map_width;
  const seedsCount = detail.seeds_count;

  // Phase 1: Submit lookup-only immediately (baseline ~82)
  console.log('  Phase 1: Submitting lookup-only predictions...');
  for (let si = 0; si < seedsCount; si++) {
    const pred = lookupPred(detail.initial_states[si].grid, H, W);
    await apiCall('POST', '/submit', {round_id: round.id, seed_index: si, prediction: pred});
    await sleep(API_DELAY);
  }
  console.log('  Phase 1 done: 5 seeds submitted (lookup-only)');

  // Phase 2: Query simulator
  const budget = await apiCall('GET', '/budget');
  const queriesLeft = budget.queries_max - budget.queries_used;
  console.log(`  Phase 2: Querying (${queriesLeft} queries available)...`);

  if (queriesLeft <= 0) {
    console.log('  No queries left. Done.');
    completed.add(round.id);
    return;
  }

  const plan = planQueries(detail.initial_states, H, W, queriesLeft);
  console.log(`  Query plan: ${plan.length} queries across ${seedsCount} seeds`);

  const observations = [];
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
      observations.push({si: q.si, grid: result.grid, viewport: result.viewport, settlements: result.settlements});
      if ((qi+1) % 10 === 0) console.log(`    ${qi+1}/${plan.length} queries done`);
    } catch (e) {
      console.log(`    Query ${qi} failed: ${e.message}`);
      if (e.message.includes('429')) break;
    }
  }
  console.log(`  Phase 2 done: ${observations.length} observations collected`);

  // Phase 3: Estimate round shift from observations
  console.log('  Phase 3: Adapting predictions...');
  const shift = estimateRoundShift(observations, detail.initial_states, H, W);
  if (shift) {
    const shiftSummary = Object.entries(shift).map(([ic, s]) =>
      `class${ic}: [${s.map(v => v.toFixed(2)).join(',')}]`
    ).join(', ');
    console.log(`  Round shift: ${shiftSummary}`);
  }

  // Phase 4: Resubmit adapted predictions
  console.log('  Phase 4: Resubmitting adapted predictions...');
  for (let si = 0; si < seedsCount; si++) {
    const ig = detail.initial_states[si].grid;
    let pred = lookupPred(ig, H, W);

    // Apply round shift
    pred = adaptPredictions(pred, shift, ig, H, W);

    // Blend with direct observations
    pred = blendWithObservations(pred, observations, detail.initial_states, H, W, si);

    await apiCall('POST', '/submit', {round_id: round.id, seed_index: si, prediction: pred});
    await sleep(API_DELAY);
  }

  const elapsed = ((Date.now() - t0) / 1000).toFixed(1);
  console.log(`  Phase 4 done: 5 seeds resubmitted (lookup + observations + adaptation) in ${elapsed}s`);
  completed.add(round.id);
}

async function poll() {
  try {
    const rounds = await apiCall('GET', '/rounds');
    const active = rounds.filter(r => r.status === 'active' && !completed.has(r.id));
    if (!active.length) { process.stdout.write('.'); return; }
    for (const round of active) await processRound(round);
  } catch (e) { console.error(`\nPoll error: ${e.message}`); }
}

console.log('Auto-submit v2 started. Polling every 30s...');
poll();
setInterval(poll, POLL_INTERVAL);
