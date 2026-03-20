#!/usr/bin/env node
/**
 * Watch for completed rounds, fetch GT, recalibrate lookup, and restart auto_submit.
 *
 * Usage: TOKEN=$(cat ../.env.ainm | cut -d= -f2) node watch_and_calibrate.js
 */
const fs = require('fs');
const path = require('path');
const https = require('https');
const { execSync } = require('child_process');

const API = 'https://api.ainm.no/astar-island';
const TOKEN = process.env.TOKEN || process.env.AINM_TOKEN;
const CACHE_DIR = path.join(__dirname, 'cache');
const NUM_CLASSES = 6;
const TTC = {10:0,11:0,0:0,1:1,2:2,3:3,4:4,5:5};

if (!TOKEN) { console.error('Set TOKEN env var'); process.exit(1); }
if (!fs.existsSync(CACHE_DIR)) fs.mkdirSync(CACHE_DIR, {recursive: true});

function apiCall(method, urlPath) {
  return new Promise((resolve, reject) => {
    const url = new URL(urlPath.startsWith('http') ? urlPath : `${API}${urlPath}`);
    const opts = {
      hostname: url.hostname, path: url.pathname + url.search,
      method, headers: {'Authorization': `Bearer ${TOKEN}`}
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
    req.end();
  });
}

// Spatial helpers
function cc(c){return TTC[c]??0}
function isCoast(g,H,W,y,x){if(g[y][x]===10||g[y][x]===5)return false;for(const[dy,dx]of[[-1,0],[1,0],[0,-1],[0,1]]){const ny=y+dy,nx=x+dx;if(ny>=0&&ny<H&&nx>=0&&nx<W&&g[ny][nx]===10)return true}return false}
function foodPot(g,H,W,y,x){let c=0;for(let dy=-1;dy<=1;dy++)for(let dx=-1;dx<=1;dx++){if(!dy&&!dx)continue;const ny=y+dy,nx=x+dx;if(ny>=0&&ny<H&&nx>=0&&nx<W&&g[ny][nx]===4)c++}return c}
function settDist(g,H,W,y,x){let m=999;for(let sy=0;sy<H;sy++)for(let sx=0;sx<W;sx++)if(g[sy][sx]===1||g[sy][sx]===2)m=Math.min(m,Math.abs(y-sy)+Math.abs(x-sx));return m}
function nSett(g,H,W,y,x){let c=0;for(let dy=-2;dy<=2;dy++)for(let dx=-2;dx<=2;dx++){if(!dy&&!dx)continue;const ny=y+dy,nx=x+dx;if(ny>=0&&ny<H&&nx>=0&&nx<W){const v=g[ny][nx];if(v===1||v===2)c++}}return c}

const calibrated = new Set();

async function fetchAndCacheGT(roundNum, roundId, seedsCount) {
  console.log(`\nFetching GT for Round ${roundNum}...`);

  // Cache init
  const initPath = path.join(CACHE_DIR, `r${roundNum}_init.json`);
  if (!fs.existsSync(initPath)) {
    const detail = await apiCall('GET', `/rounds/${roundId}`);
    fs.writeFileSync(initPath, JSON.stringify(detail));
    console.log(`  Init states: cached`);
  }

  // Cache GT per seed
  for (let si = 0; si < seedsCount; si++) {
    const gtPath = path.join(CACHE_DIR, `r${roundNum}_gt_s${si}.json`);
    if (fs.existsSync(gtPath)) { console.log(`  Seed ${si}: already cached`); continue; }
    await new Promise(r => setTimeout(r, 300));
    const analysis = await apiCall('GET', `/analysis/${roundId}/${si}`);
    fs.writeFileSync(gtPath, JSON.stringify(analysis));
    console.log(`  Seed ${si}: cached (score: ${analysis.score})`);
  }
}

function rebuildLookup() {
  console.log('\nRebuilding GT lookup from all cached data...');
  const files = fs.readdirSync(CACHE_DIR);
  const roundNums = [...new Set(
    files.filter(f => f.match(/^r\d+_init\.json$/)).map(f => parseInt(f.match(/^r(\d+)/)[1]))
  )].sort((a,b) => a-b);

  const ctxStats = {};
  let totalCells = 0;

  for (const r of roundNums) {
    const initPath = path.join(CACHE_DIR, `r${r}_init.json`);
    if (!fs.existsSync(initPath)) continue;

    for (let si = 0; si < 5; si++) {
      const gtPath = path.join(CACHE_DIR, `r${r}_gt_s${si}.json`);
      if (!fs.existsSync(gtPath)) continue;
      const data = JSON.parse(fs.readFileSync(gtPath, 'utf8'));
      const gt = data.ground_truth, ig = data.initial_grid;
      const H = data.height, W = data.width;

      for (let y = 0; y < H; y++) {
        for (let x = 0; x < W; x++) {
          const raw = ig[y][x];
          const ic = cc(raw);
          const f = Math.min(foodPot(ig,H,W,y,x), 4);
          const co = isCoast(ig,H,W,y,x) ? 1 : 0;
          const d = settDist(ig,H,W,y,x);
          const n = Math.min(nSett(ig,H,W,y,x), 3);
          const db = d<=3?'near':d<=7?'mid':d<=12?'far':'remote';
          const key = `${ic}_${f}_${co}_${db}_${n}`;
          if (!ctxStats[key]) ctxStats[key] = {counts: new Float64Array(6), n: 0};
          for (let c = 0; c < 6; c++) ctxStats[key].counts[c] += gt[y][x][c];
          ctxStats[key].n++;
          totalCells++;
        }
      }
    }
  }

  const lookup = {};
  for (const [k, v] of Object.entries(ctxStats)) {
    if (v.n < 3) continue;
    const total = v.counts.reduce((a,b) => a+b, 0);
    const dist = [];
    for (let c = 0; c < 6; c++) dist.push(+((v.counts[c]+0.01)/(total+0.06)).toFixed(6));
    lookup[k] = dist;
  }

  const outPath = path.join(__dirname, 'gt_lookup.json');
  fs.writeFileSync(outPath, JSON.stringify(lookup));
  console.log(`Rebuilt: ${Object.keys(lookup).length} bins from ${totalCells} cells (${roundNums.length} rounds)`);
  return lookup;
}

async function check() {
  try {
    const rounds = await apiCall('GET', '/my-rounds');
    const newCompleted = rounds.filter(r =>
      r.status === 'completed' && r.round_score && !calibrated.has(r.id)
    );

    if (newCompleted.length === 0) { process.stdout.write('.'); return; }

    for (const r of newCompleted) {
      console.log(`\n[${new Date().toISOString().slice(11,19)}] Round ${r.round_number} completed! Score: ${r.round_score}, Rank: ${r.rank}`);
      await fetchAndCacheGT(r.round_number, r.id, r.seeds_count || 5);
      calibrated.add(r.id);
    }

    // Rebuild lookup with new data
    rebuildLookup();
    console.log('Lookup updated. auto_submit.js will use new lookup on next restart.');

  } catch (e) { console.error(`\nError: ${e.message}`); }
}

// Mark already-calibrated rounds
async function init() {
  const existing = fs.readdirSync(CACHE_DIR)
    .filter(f => f.match(/^r\d+_gt_s0\.json$/))
    .map(f => parseInt(f.match(/^r(\d+)/)[1]));

  const rounds = await apiCall('GET', '/my-rounds');
  for (const r of rounds) {
    if (r.status === 'completed' && existing.includes(r.round_number)) {
      calibrated.add(r.id);
    }
  }
  console.log(`Already calibrated: ${calibrated.size} rounds. Watching for new completions...`);
}

init().then(() => {
  check();
  setInterval(check, 30000);
}).catch(e => { console.error(e); process.exit(1); });
