#!/usr/bin/env node
/**
 * Auto-submit for Astar Island using GT-calibrated lookup.
 * Polls for new rounds, builds predictions from gt_lookup.json, submits via API.
 *
 * Usage: TOKEN=$(cat ../.env.ainm | cut -d= -f2) node auto_submit.js
 */
const fs = require('fs');
const path = require('path');
const https = require('https');

const API = 'https://api.ainm.no/astar-island';
const TOKEN = process.env.TOKEN || process.env.AINM_TOKEN;
const POLL_INTERVAL = 30000;
const NUM_CLASSES = 6;
const TTC = {10:0,11:0,0:0,1:1,2:2,3:3,4:4,5:5};

if (!TOKEN) { console.error('Set TOKEN env var'); process.exit(1); }

// Load GT lookup (re-read each time a round is processed for hot-reload)
const LOOKUP_PATH = path.join(__dirname, 'gt_lookup.json');
let LOOKUP = JSON.parse(fs.readFileSync(LOOKUP_PATH, 'utf8'));
let lookupMtime = fs.statSync(LOOKUP_PATH).mtimeMs;
console.log(`Loaded ${Object.keys(LOOKUP).length} context bins`);

function reloadLookupIfChanged() {
  try {
    const mt = fs.statSync(LOOKUP_PATH).mtimeMs;
    if (mt > lookupMtime) {
      LOOKUP = JSON.parse(fs.readFileSync(LOOKUP_PATH, 'utf8'));
      lookupMtime = mt;
      console.log(`[HOT-RELOAD] Lookup updated: ${Object.keys(LOOKUP).length} bins`);
    }
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

// Spatial helpers
function cc(c){return TTC[c]??0}
function isCoast(g,H,W,y,x){if(g[y][x]===10||g[y][x]===5)return false;for(const[dy,dx]of[[-1,0],[1,0],[0,-1],[0,1]]){const ny=y+dy,nx=x+dx;if(ny>=0&&ny<H&&nx>=0&&nx<W&&g[ny][nx]===10)return true}return false}
function food(g,H,W,y,x){let c=0;for(let dy=-1;dy<=1;dy++)for(let dx=-1;dx<=1;dx++){if(!dy&&!dx)continue;const ny=y+dy,nx=x+dx;if(ny>=0&&ny<H&&nx>=0&&nx<W&&g[ny][nx]===4)c++}return c}
function sd(g,H,W,y,x){let m=999;for(let sy=0;sy<H;sy++)for(let sx=0;sx<W;sx++)if(g[sy][sx]===1||g[sy][sx]===2)m=Math.min(m,Math.abs(y-sy)+Math.abs(x-sx));return m}
function ns(g,H,W,y,x){let c=0;for(let dy=-2;dy<=2;dy++)for(let dx=-2;dx<=2;dx++){if(!dy&&!dx)continue;const ny=y+dy,nx=x+dx;if(ny>=0&&ny<H&&nx>=0&&nx<W){const v=g[ny][nx];if(v===1||v===2)c++}}return c}

function predict(ig, H, W) {
  const pred = [];
  for(let y=0;y<H;y++){pred[y]=[];for(let x=0;x<W;x++){
    const raw=ig[y][x];
    if(raw===5){pred[y][x]=[.002,.002,.002,.002,.002,.990];continue}
    if(raw===10){pred[y][x]=[.990,.002,.002,.002,.002,.002];continue}
    const ic=cc(raw),f=Math.min(food(ig,H,W,y,x),4);
    const co=isCoast(ig,H,W,y,x)?1:0,d=sd(ig,H,W,y,x);
    const n=Math.min(ns(ig,H,W,y,x),3);
    const db=d<=3?'near':d<=7?'mid':d<=12?'far':'remote';
    const keys=[`${ic}_${f}_${co}_${db}_${n}`,`${ic}_${f}_${co}_${db}_0`,`${ic}_${Math.min(f,2)}_${co}_${db}_0`,`${ic}_0_${co}_${db}_0`];
    let p=null;
    for(const k of keys){if(LOOKUP[k]){p=[...LOOKUP[k]];break}}
    if(!p)p=[.5,.1,.05,.05,.25,.05];
    if(!co)p[2]=.002;
    let s=0;for(let c=0;c<6;c++){p[c]=Math.max(p[c],.002);s+=p[c]}
    for(let c=0;c<6;c++)p[c]/=s;
    pred[y][x]=p;
  }}
  return pred;
}

const completed = new Set();

async function poll() {
  try {
    const rounds = await apiCall('GET', '/rounds');
    const active = rounds.filter(r => r.status === 'active' && !completed.has(r.id));
    if (!active.length) { process.stdout.write('.'); reloadLookupIfChanged(); return; }

    for (const round of active) {
      console.log(`\n[${new Date().toISOString().slice(11,19)}] Round ${round.round_number} active!`);
      const detail = await apiCall('GET', `/rounds/${round.id}`);
      const H = detail.map_height, W = detail.map_width;

      for (let si = 0; si < detail.seeds_count; si++) {
        const pred = predict(detail.initial_states[si].grid, H, W);
        await apiCall('POST', '/submit', {round_id: round.id, seed_index: si, prediction: pred});
        console.log(`  Seed ${si}: submitted`);
        await new Promise(r => setTimeout(r, 600));
      }

      completed.add(round.id);
      console.log(`  Round ${round.round_number} complete!`);
    }
  } catch (e) { console.error(`\nPoll error: ${e.message}`); }
}

console.log('Auto-submit started. Polling every 30s...');
poll();
setInterval(poll, POLL_INTERVAL);
