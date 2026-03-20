#!/usr/bin/env node
/**
 * Fast test runner with precomputed features.
 * Precompute ALL features once, then test 1000+ configs in seconds.
 *
 * Usage: node fast_test.js
 */
const fs = require('fs'), path = require('path');
const NC = 6, TTC = {10:0,11:0,0:0,1:1,2:2,3:3,4:4,5:5};
const CACHE = path.join(__dirname, 'cache');

function cc(c){return TTC[c]??0}
function isCoast(g,H,W,y,x){if(g[y][x]===10||g[y][x]===5)return false;for(const[dy,dx]of[[-1,0],[1,0],[0,-1],[0,1]]){const ny=y+dy,nx=x+dx;if(ny>=0&&ny<H&&nx>=0&&nx<W&&g[ny][nx]===10)return true}return false}
function foodPot(g,H,W,y,x){let c=0;for(let dy=-1;dy<=1;dy++)for(let dx=-1;dx<=1;dx++){if(!dy&&!dx)continue;const ny=y+dy,nx=x+dx;if(ny>=0&&ny<H&&nx>=0&&nx<W&&g[ny][nx]===4)c++}return c}
function sdist(g,H,W,y,x){let m=999;for(let sy=0;sy<H;sy++)for(let sx=0;sx<W;sx++)if(g[sy][sx]===1||g[sy][sx]===2)m=Math.min(m,Math.abs(y-sy)+Math.abs(x-sx));return m}
function nsett(g,H,W,y,x){let c=0;for(let dy=-2;dy<=2;dy++)for(let dx=-2;dx<=2;dx++){if(!dy&&!dx)continue;const ny=y+dy,nx=x+dx;if(ny>=0&&ny<H&&nx>=0&&nx<W){const v=g[ny][nx];if(v===1||v===2)c++}}return c}
function localDens(g,H,W,y,x){let c=0;for(let dy=-6;dy<=6;dy++)for(let dx=-6;dx<=6;dx++){if(!dy&&!dx)continue;const ny=y+dy,nx=x+dx;if(ny>=0&&ny<H&&nx>=0&&nx<W&&(g[ny][nx]===1||g[ny][nx]===2))c++}return c}

console.log('Precomputing features for all GT cells...');
const t0 = Date.now();

// Precompute EVERYTHING
const datasets = []; // [{ig, gt, H, W, round, seed, cells: [{ic, co, ctxKey, features, gt6, setts}]}]
const LOOKUP = JSON.parse(fs.readFileSync(path.join(__dirname, 'gt_lookup.json'), 'utf8'));

let MODEL = null;
try { const m=JSON.parse(fs.readFileSync(path.join(__dirname,'model_weights.json'),'utf8'));MODEL={W1:m.W1,W2:m.W2,NF:m.config.N_FEATURES,NH:m.config.HIDDEN}; } catch {}

const rounds = [...new Set(fs.readdirSync(CACHE).filter(f=>f.match(/^r\d+_init/)).map(f=>parseInt(f.match(/r(\d+)/)[1])))].sort((a,b)=>a-b);

for (const r of rounds) {
  for (let si = 0; si < 5; si++) {
    const p = path.join(CACHE, `r${r}_gt_s${si}.json`);
    if (!fs.existsSync(p)) continue;
    const d = JSON.parse(fs.readFileSync(p, 'utf8'));
    const ig = d.initial_grid, gt = d.ground_truth, H = d.height, W = d.width;

    const cells = [];
    const setts = [];
    let totalS = 0;
    for (let y=0;y<H;y++) for(let x=0;x<W;x++) if(ig[y][x]===1||ig[y][x]===2) { setts.push({y,x}); totalS++; }

    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
      const raw = ig[y][x];
      const ic = cc(raw);
      const co = isCoast(ig,H,W,y,x);
      const fd = foodPot(ig,H,W,y,x);
      const sd = sdist(ig,H,W,y,x);
      const ns = nsett(ig,H,W,y,x);
      const ld = localDens(ig,H,W,y,x);

      const fBin = Math.min(fd,4);
      const coBin = co?1:0;
      const db = sd<=3?'near':sd<=7?'mid':sd<=12?'far':'remote';
      const nBin = Math.min(ns,3);
      const keys = [`${ic}_${fBin}_${coBin}_${db}_${nBin}`,`${ic}_${fBin}_${coBin}_${db}_0`,`${ic}_${Math.min(fBin,2)}_${coBin}_${db}_0`,`${ic}_0_${coBin}_${db}_0`];

      let lookupDist = null;
      for (const k of keys) { if (LOOKUP[k]) { lookupDist = LOOKUP[k]; break; } }
      if (!lookupDist) lookupDist = [.5,.1,.05,.05,.25,.05];

      // Model features
      const features = new Float64Array(17);
      features[ic]=1; features[6]=fd/8; features[7]=co?1:0; features[8]=Math.exp(-sd/5);
      features[9]=ns/8; features[10]=totalS/60; features[11]=ld/20;
      features[13]=features[6]*features[8]; features[14]=features[7]*features[8]; features[15]=features[9]*features[8]; features[16]=1;

      // GT entropy
      let entropy = 0;
      for (let c=0;c<NC;c++) if(gt[y][x][c]>0) entropy -= gt[y][x][c]*Math.log(gt[y][x][c]);

      cells.push({ y, x, raw, ic, co, lookupDist: [...lookupDist], features, gt6: gt[y][x], entropy });
    }

    // Precompute query viewports (settlement-centered)
    const viewports = [];
    for (let q = 0; q < Math.min(setts.length, 15); q++) {
      const s = setts[q];
      viewports.push({ x: Math.max(0,Math.min(W-15,s.x-7)), y: Math.max(0,Math.min(H-15,s.y-7)) });
    }

    datasets.push({ ig, gt, H, W, round: r, seed: si, cells, setts, viewports, totalS });
  }
}

console.log(`Precomputed ${datasets.length} seeds, ${datasets.reduce((s,d)=>s+d.cells.length,0)} cells in ${((Date.now()-t0)/1000).toFixed(1)}s`);

// ── Fast scoring functions (no spatial recomputation) ───────────────────────

function modelPredFast(features) {
  if (!MODEL) return null;
  const h = new Float64Array(MODEL.NH);
  for(let i=0;i<MODEL.NH;i++){let s=0;for(let j=0;j<MODEL.NF;j++)s+=MODEL.W1[i][j]*features[j];h[i]=s>0?s:0}
  const logits = new Float64Array(NC);
  for(let i=0;i<NC;i++){let s=0;for(let j=0;j<MODEL.NH;j++)s+=MODEL.W2[i][j]*h[j];logits[i]=s}
  let mx=-Infinity;for(let i=0;i<NC;i++)if(logits[i]>mx)mx=logits[i];
  const p=new Float64Array(NC);let se=0;
  for(let i=0;i<NC;i++){p[i]=Math.exp(logits[i]-mx);se+=p[i]}
  for(let i=0;i<NC;i++)p[i]=Math.max(p[i]/se,0.002);
  let s2=0;for(let i=0;i<NC;i++)s2+=p[i];for(let i=0;i<NC;i++)p[i]/=s2;
  return p;
}

function sampleClass(gt6) {
  const r = Math.random(); let cum = 0;
  for (let c = 0; c < NC; c++) { cum += gt6[c]; if (r < cum) return c; }
  return NC - 1;
}

function simulateAndScore(ds, damping, modelWeight, nQueries) {
  const { cells, setts, viewports, ig, gt, H, W } = ds;

  // Simulate queries → observation counts per init class
  const obsTrans = {};
  for (let ic = 0; ic < NC; ic++) obsTrans[ic] = new Float64Array(NC);

  const queriesPerSeed = nQueries;
  for (let q = 0; q < queriesPerSeed; q++) {
    const vp = viewports[q % viewports.length];
    for (let gy = 0; gy < 15; gy++) for (let gx = 0; gx < 15; gx++) {
      const ay = vp.y + gy, ax = vp.x + gx;
      if (ay >= H || ax >= W) continue;
      const ic = cc(ig[ay][ax]);
      obsTrans[ic][sampleClass(gt[ay][ax])]++;
    }
  }

  // Compute avg rates
  const avgRates = {};
  for (let ic = 0; ic < NC; ic++) {
    avgRates[ic] = new Array(NC).fill(0); let n = 0;
    for (const [k,dist] of Object.entries(LOOKUP)) {
      if (parseInt(k.split('_')[0]) === ic) { for (let c = 0; c < NC; c++) avgRates[ic][c] += dist[c]; n++; }
    }
    if (n > 0) for (let c = 0; c < NC; c++) avgRates[ic][c] /= n;
  }

  // Compute shift
  const shift = {};
  for (let ic = 0; ic < NC; ic++) {
    const tot = obsTrans[ic].reduce((a,b) => a+b, 0);
    if (tot < 15) { shift[ic] = new Array(NC).fill(1); continue; }
    shift[ic] = [];
    for (let c = 0; c < NC; c++) {
      const obs = obsTrans[ic][c] / tot;
      const avg = avgRates[ic] ? avgRates[ic][c] : 0;
      shift[ic].push(1 + damping * ((avg > 0.01 ? obs / avg : 1) - 1));
    }
  }

  // Score
  let tw = 0, te = 0;
  for (const cell of cells) {
    if (cell.raw === 5) { // mountain
      const kl = cell.gt6[5] > 0 ? cell.gt6[5] * Math.log(cell.gt6[5] / 0.990) : 0;
      tw += cell.entropy * kl; te += cell.entropy; continue;
    }
    if (cell.raw === 10) { // ocean
      const kl = cell.gt6[0] > 0 ? cell.gt6[0] * Math.log(cell.gt6[0] / 0.990) : 0;
      tw += cell.entropy * kl; te += cell.entropy; continue;
    }

    const p = new Float64Array(NC);
    const shifted = new Float64Array(NC);
    for (let c = 0; c < NC; c++) shifted[c] = cell.lookupDist[c] * (shift[cell.ic] ? shift[cell.ic][c] : 1);

    if (MODEL && modelWeight > 0) {
      const mp = modelPredFast(cell.features);
      if (mp) {
        for (let c = 0; c < NC; c++) p[c] = (1-modelWeight) * shifted[c] + modelWeight * mp[c];
      } else {
        for (let c = 0; c < NC; c++) p[c] = shifted[c];
      }
    } else {
      for (let c = 0; c < NC; c++) p[c] = shifted[c];
    }

    if (!cell.co) p[2] = 0.002;
    let s = 0; for (let c = 0; c < NC; c++) { p[c] = Math.max(p[c], 0.002); s += p[c]; }
    for (let c = 0; c < NC; c++) p[c] /= s;

    let kl = 0;
    for (let c = 0; c < NC; c++) if (cell.gt6[c] > 0) kl += cell.gt6[c] * Math.log(cell.gt6[c] / p[c]);
    tw += cell.entropy * kl;
    te += cell.entropy;
  }

  return 100 * Math.exp(-3 * tw / te);
}

// ── Run massive grid search ─────────────────────────────────────────────────
console.log('\nRunning grid search...');
const t1 = Date.now();

const results = [];
const dampings = [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0];
const modelWeights = MODEL ? [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7] : [0.0];
const queryAmounts = [5, 10, 15, 20];
const TRIALS = 10;

for (const nQ of queryAmounts) {
  for (const d of dampings) {
    for (const mw of modelWeights) {
      let totalScore = 0, minScore = Infinity, n = 0;
      for (const ds of datasets) {
        for (let t = 0; t < TRIALS; t++) {
          const s = simulateAndScore(ds, d, mw, nQ);
          totalScore += s; n++;
          if (s < minScore) minScore = s;
        }
      }
      const mean = totalScore / n;
      results.push({ d, mw, nQ, mean, min: minScore, n });
    }
  }
  process.stdout.write(`nQ=${nQ} done. `);
}

const elapsed = ((Date.now() - t1) / 1000).toFixed(1);
console.log(`\n${results.length} configs tested in ${elapsed}s (${(results.length * datasets.length * TRIALS)} total evaluations)\n`);

// Sort by mean score
results.sort((a, b) => b.mean - a.mean);

console.log('=== TOP 20 CONFIGS ===');
console.log('Rank  Damping  Model  Queries  Mean   Min');
console.log('─'.repeat(55));
for (const [i, r] of results.slice(0, 20).entries()) {
  console.log(`#${String(i+1).padEnd(4)} d=${r.d.toFixed(1).padEnd(5)} m=${r.mw.toFixed(1).padEnd(5)} q=${String(r.nQ).padEnd(4)}   ${r.mean.toFixed(1).padStart(5)}  ${r.min.toFixed(1).padStart(5)}`);
}

// Per-round breakdown for #1 config
const best = results[0];
console.log(`\n=== BEST CONFIG: d=${best.d} m=${best.mw} q=${best.nQ} → ${best.mean.toFixed(1)} ===`);
for (const r of rounds) {
  const rScores = [];
  for (const ds of datasets.filter(d => d.round === r)) {
    for (let t = 0; t < TRIALS; t++) rScores.push(simulateAndScore(ds, best.d, best.mw, best.nQ));
  }
  const mean = rScores.reduce((a,b)=>a+b,0)/rScores.length;
  const min = Math.min(...rScores);
  console.log(`  R${r}: mean=${mean.toFixed(1)} min=${min.toFixed(1)}`);
}

// Save best config
fs.writeFileSync(path.join(__dirname, 'best_config.json'), JSON.stringify({
  damping: best.d, modelWeight: best.mw, queries: best.nQ, score: best.mean
}));
console.log(`\nSaved best_config.json`);
