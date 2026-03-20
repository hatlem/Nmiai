#!/usr/bin/env node
/**
 * auto_v4.js — Simple, robust auto-submitter.
 * No model, no complex blending. Just lookup + shift.
 * Tested at 82-84 avg over 7 rounds, 300+ trials.
 *
 * Usage: TOKEN=$(cat ../.env.ainm | cut -d= -f2) node auto_v4.js
 */
const fs = require('fs'), path = require('path'), https = require('https');
const NC = 6, TTC = {10:0,11:0,0:0,1:1,2:2,3:3,4:4,5:5};
const TOKEN = process.env.TOKEN || process.env.AINM_TOKEN;
if (!TOKEN) { console.error('Need TOKEN'); process.exit(1); }

// Config
const FL = 0.001;
const DAMP = 0.7;
const CLIP = [0.5, 3.0];
const COAST_DAMP = 0.5;
const API_DELAY = 300;
const SUBMIT_DELAY = 600;
const POLL = 30000;

// Lookup
const LOOKUP = JSON.parse(fs.readFileSync(path.join(__dirname, 'gt_lookup.json'), 'utf8'));
console.log(`Lookup: ${Object.keys(LOOKUP).length} bins`);

// Completed rounds (persisted)
const DONE_FILE = path.join(__dirname, 'completed_rounds.json');
let done = new Set();
try { done = new Set(JSON.parse(fs.readFileSync(DONE_FILE, 'utf8'))); } catch {}
function saveDone() { fs.writeFileSync(DONE_FILE, JSON.stringify([...done])); }

// API
function api(method, p, body) {
  return new Promise((res, rej) => {
    const url = new URL('https://api.ainm.no/astar-island' + p);
    const opts = { hostname: url.hostname, path: url.pathname, method,
      headers: { 'Authorization': `Bearer ${TOKEN}`, 'Content-Type': 'application/json' } };
    const req = https.request(opts, r => {
      let d = ''; r.on('data', c => d += c);
      r.on('end', () => {
        if (r.statusCode === 429) return rej(new Error('429'));
        if (r.statusCode >= 400) return rej(new Error(`${r.statusCode}: ${d.slice(0,100)}`));
        try { res(JSON.parse(d)); } catch { res(d); }
      });
    });
    req.on('error', rej);
    if (body) req.write(JSON.stringify(body));
    req.end();
  });
}
const sleep = ms => new Promise(r => setTimeout(r, ms));

// Spatial
function cc(c) { return TTC[c] ?? 0; }
function isCoast(g,H,W,y,x) { if(g[y][x]===10||g[y][x]===5) return false; for(const[dy,dx]of[[-1,0],[1,0],[0,-1],[0,1]]){const ny=y+dy,nx=x+dx;if(ny>=0&&ny<H&&nx>=0&&nx<W&&g[ny][nx]===10)return true} return false; }
function foodPot(g,H,W,y,x) { let c=0; for(let dy=-1;dy<=1;dy++) for(let dx=-1;dx<=1;dx++){if(!dy&&!dx)continue;const ny=y+dy,nx=x+dx;if(ny>=0&&ny<H&&nx>=0&&nx<W&&g[ny][nx]===4)c++} return c; }
function sdist(g,H,W,y,x) { let m=999; for(let sy=0;sy<H;sy++) for(let sx=0;sx<W;sx++) if(g[sy][sx]===1||g[sy][sx]===2) m=Math.min(m,Math.abs(y-sy)+Math.abs(x-sx)); return m; }
function nsett(g,H,W,y,x) { let c=0; for(let dy=-2;dy<=2;dy++) for(let dx=-2;dx<=2;dx++){if(!dy&&!dx)continue;const ny=y+dy,nx=x+dx;if(ny>=0&&ny<H&&nx>=0&&nx<W){const v=g[ny][nx];if(v===1||v===2)c++}} return c; }

function predict(ig, H, W) {
  const pred = Array.from({length:H}, () => Array.from({length:W}, () => new Array(NC)));
  for (let y=0; y<H; y++) for (let x=0; x<W; x++) {
    const raw = ig[y][x];
    if (raw === 5) { pred[y][x] = [FL,FL,FL,FL,FL,1-5*FL]; continue; }
    if (raw === 10) { pred[y][x] = [1-5*FL,FL,FL,FL,FL,FL]; continue; }
    const ic=cc(raw), f=Math.min(foodPot(ig,H,W,y,x),4), co=isCoast(ig,H,W,y,x)?1:0;
    const sd=sdist(ig,H,W,y,x), n=Math.min(nsett(ig,H,W,y,x),3);
    const db=sd<=3?'near':sd<=7?'mid':sd<=12?'far':'remote';
    const keys=[`${ic}_${f}_${co}_${db}_${n}`,`${ic}_${f}_${co}_${db}_0`,`${ic}_${Math.min(f,2)}_${co}_${db}_0`,`${ic}_0_${co}_${db}_0`];
    let p=null; for (const k of keys) { if (LOOKUP[k]) { p=[...LOOKUP[k]]; break; } }
    if (!p) p=[.5,.1,.05,.05,.25,.05];
    if (!co) p[2]=FL;
    let s=0; for(let c=0;c<NC;c++){p[c]=Math.max(p[c],FL);s+=p[c]} for(let c=0;c<NC;c++)p[c]/=s;
    pred[y][x] = p;
  }
  return pred;
}

function shiftPredict(ig, H, W, shift) {
  const pred = predict(ig, H, W);
  for (let y=0; y<H; y++) for (let x=0; x<W; x++) {
    const raw = ig[y][x];
    if (raw === 5 || raw === 10) continue;
    const ic = cc(raw), co = isCoast(ig,H,W,y,x);
    if (!shift[ic]) continue;
    const d = co ? COAST_DAMP : DAMP;
    for (let c=0; c<NC; c++) pred[y][x][c] *= Math.pow(shift[ic][c], d);
    if (!co) pred[y][x][2] = FL;
    let s=0; for(let c=0;c<NC;c++){pred[y][x][c]=Math.max(pred[y][x][c],FL);s+=pred[y][x][c]} for(let c=0;c<NC;c++)pred[y][x][c]/=s;
  }
  return pred;
}

async function submitAll(roundId, detail, H, W, predFn, label) {
  for (let si=0; si<detail.seeds_count; si++) {
    const pred = predFn(detail.initial_states[si].grid, H, W);
    for (let retry=0; retry<3; retry++) {
      try {
        await api('POST', '/submit', {round_id: roundId, seed_index: si, prediction: pred});
        await sleep(SUBMIT_DELAY);
        break;
      } catch (e) {
        if (e.message === '429') { await sleep(2000); continue; }
        console.log(`  Submit ${si} failed: ${e.message}`);
        break;
      }
    }
  }
  console.log(`  ${label}: 5 seeds submitted`);
}

async function processRound(round) {
  const t0 = Date.now();
  const log = m => console.log(`[${((Date.now()-t0)/1000).toFixed(0)}s] ${m}`);
  console.log(`\n[${new Date().toISOString().slice(11,19)}] R${round.round_number} ACTIVE`);

  const detail = await api('GET', `/rounds/${round.id}`);
  const H = detail.map_height, W = detail.map_width;

  // 1. Submit lookup immediately
  await submitAll(round.id, detail, H, W, (ig,H,W) => predict(ig,H,W), 'Lookup-only');

  // 2. Query
  let budget;
  try { budget = await api('GET', '/budget'); } catch { budget = {queries_used:0,queries_max:50}; }
  const left = budget.queries_max - budget.queries_used;
  log(`Queries: ${left} available`);
  if (left <= 0) return;

  // Plan: cover all settlements
  const allSetts = [];
  for (let si=0; si<detail.seeds_count; si++) {
    const setts = detail.initial_states[si].settlements || [];
    for (const s of setts) allSetts.push({si, x: Math.max(0,Math.min(W-15,s.x-7)), y: Math.max(0,Math.min(H-15,s.y-7))});
  }
  // Deduplicate viewports
  const vpSet = new Map();
  for (const vp of allSetts) { const k=`${vp.si}_${vp.x}_${vp.y}`; if(!vpSet.has(k)) vpSet.set(k, vp); }
  const viewports = [...vpSet.values()];
  const plan = [];
  for (let i=0; i<left; i++) plan.push(viewports[i % viewports.length]);

  // Execute queries in batches of 10, shift + resubmit after each batch
  const obsTrans = {}; for(let ic=0;ic<NC;ic++) obsTrans[ic] = new Float64Array(NC);
  let queryCount = 0, rlCount = 0;

  for (let qi=0; qi<plan.length; qi++) {
    const q = plan[qi];
    try {
      await sleep(API_DELAY);
      const result = await api('POST', '/simulate', {
        round_id: round.id, seed_index: q.si,
        viewport_x: q.x, viewport_y: q.y, viewport_w: 15, viewport_h: 15
      });
      for (let gy=0; gy<result.grid.length; gy++)
        for (let gx=0; gx<result.grid[gy].length; gx++) {
          const ay=result.viewport.y+gy, ax=result.viewport.x+gx;
          if (ay<H && ax<W) obsTrans[cc(detail.initial_states[q.si].grid[ay][ax])][cc(result.grid[gy][gx])]++;
        }
      queryCount++;
    } catch (e) {
      if (e.message === '429') { rlCount++; if(rlCount>10) break; await sleep(2000); qi--; continue; }
      if (e.message.includes('budget')) break;
      console.log(`  Q${qi} failed: ${e.message}`);
    }

    // Every 10 queries: compute shift + resubmit
    if (queryCount > 0 && queryCount % 10 === 0) {
      const avgRates = {};
      for(let ic=0;ic<NC;ic++){avgRates[ic]=new Array(NC).fill(0);let n=0;
        for(const[k,dist]of Object.entries(LOOKUP)){if(parseInt(k.split('_')[0])===ic){for(let c=0;c<NC;c++)avgRates[ic][c]+=dist[c];n++}}
        if(n>0)for(let c=0;c<NC;c++)avgRates[ic][c]/=n}
      const shift = {};
      for(let ic=0;ic<NC;ic++){const tot=obsTrans[ic].reduce((a,b)=>a+b,0);
        if(tot<15){shift[ic]=new Array(NC).fill(1);continue}
        shift[ic]=[];for(let c=0;c<NC;c++){const obs=obsTrans[ic][c]/tot;const avg=avgRates[ic]?avgRates[ic][c]:0;
          let raw=avg>0.01?obs/avg:1;shift[ic].push(Math.max(CLIP[0],Math.min(CLIP[1],raw)))}}

      log(`${queryCount} queries → resubmitting with shift`);
      await submitAll(round.id, detail, H, W, (ig,H,W) => shiftPredict(ig,H,W,shift), `Shift (${queryCount}q)`);
    }
  }

  // Final shift + submit if not just done
  if (queryCount % 10 !== 0 && queryCount > 0) {
    const avgRates = {};
    for(let ic=0;ic<NC;ic++){avgRates[ic]=new Array(NC).fill(0);let n=0;
      for(const[k,dist]of Object.entries(LOOKUP)){if(parseInt(k.split('_')[0])===ic){for(let c=0;c<NC;c++)avgRates[ic][c]+=dist[c];n++}}
      if(n>0)for(let c=0;c<NC;c++)avgRates[ic][c]/=n}
    const shift = {};
    for(let ic=0;ic<NC;ic++){const tot=obsTrans[ic].reduce((a,b)=>a+b,0);
      if(tot<15){shift[ic]=new Array(NC).fill(1);continue}
      shift[ic]=[];for(let c=0;c<NC;c++){const obs=obsTrans[ic][c]/tot;const avg=avgRates[ic]?avgRates[ic][c]:0;
        let raw=avg>0.01?obs/avg:1;shift[ic].push(Math.max(CLIP[0],Math.min(CLIP[1],raw)))}}
    log(`Final: ${queryCount} queries → resubmitting`);
    await submitAll(round.id, detail, H, W, (ig,H,W) => shiftPredict(ig,H,W,shift), `Final (${queryCount}q)`);
  }

  log(`Done in ${((Date.now()-t0)/1000).toFixed(0)}s`);
}

// Poll
let processing = false;
async function poll() {
  if (processing) return;
  try {
    const rounds = await api('GET', '/rounds');
    const active = rounds.filter(r => r.status === 'active' && !done.has(r.id));
    if (!active.length) { process.stdout.write('.'); return; }
    for (const round of active) {
      processing = true;
      done.add(round.id); saveDone();
      try { await processRound(round); }
      catch (e) { console.error(`Error: ${e.message}`); }
      finally { processing = false; }
    }
  } catch (e) { console.error(`Poll: ${e.message}`); }
}

console.log('v4 started. Polling...');
poll();
setInterval(poll, POLL);
