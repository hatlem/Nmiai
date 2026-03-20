#!/usr/bin/env node
/**
 * Continuous improvement loop.
 * 1. Wait for new GT data (from watch_and_calibrate.js)
 * 2. Retrain model
 * 3. Rebuild lookups
 * 4. Run 100 tests
 * 5. If new strategy beats current → update auto_submit_v3 config
 * 6. Repeat
 *
 * Usage: TOKEN=$(cat ../.env.ainm | cut -d= -f2) node continuous_improve.js
 */
const fs = require('fs'), path = require('path'), { execSync } = require('child_process');
const CACHE = path.join(__dirname, 'cache');
const NC = 6, TTC = {10:0,11:0,0:0,1:1,2:2,3:3,4:4,5:5};

function log(msg) { console.log(`[${new Date().toISOString().slice(11,19)}] ${msg}`); }

function countGTSeeds() {
  try {
    return fs.readdirSync(CACHE).filter(f => f.match(/^r\d+_gt_s\d+\.json$/)).length;
  } catch { return 0; }
}

function cc(c){return TTC[c]??0}
function isCoast(g,H,W,y,x){if(g[y][x]===10||g[y][x]===5)return false;for(const[dy,dx]of[[-1,0],[1,0],[0,-1],[0,1]]){const ny=y+dy,nx=x+dx;if(ny>=0&&ny<H&&nx>=0&&nx<W&&g[ny][nx]===10)return true}return false}
function foodPot(g,H,W,y,x){let c=0;for(let dy=-1;dy<=1;dy++)for(let dx=-1;dx<=1;dx++){if(!dy&&!dx)continue;const ny=y+dy,nx=x+dx;if(ny>=0&&ny<H&&nx>=0&&nx<W&&g[ny][nx]===4)c++}return c}
function sdist(g,H,W,y,x){let m=999;for(let sy=0;sy<H;sy++)for(let sx=0;sx<W;sx++)if(g[sy][sx]===1||g[sy][sx]===2)m=Math.min(m,Math.abs(y-sy)+Math.abs(x-sx));return m}
function nsett(g,H,W,y,x){let c=0;for(let dy=-2;dy<=2;dy++)for(let dx=-2;dx<=2;dx++){if(!dy&&!dx)continue;const ny=y+dy,nx=x+dx;if(ny>=0&&ny<H&&nx>=0&&nx<W){const v=g[ny][nx];if(v===1||v===2)c++}}return c}
function localDens(g,H,W,y,x){let c=0;for(let dy=-6;dy<=6;dy++)for(let dx=-6;dx<=6;dx++){if(!dy&&!dx)continue;const ny=y+dy,nx=x+dx;if(ny>=0&&ny<H&&nx>=0&&nx<W&&(g[ny][nx]===1||g[ny][nx]===2))c++}return c}

function sampleFromGT(gt,y,x){const r=Math.random();let cum=0;for(let c=0;c<NC;c++){cum+=gt[y][x][c];if(r<cum)return c;}return NC-1;}
function scoreKL(pred,gt,H,W){let tw=0,te=0;for(let y=0;y<H;y++)for(let x=0;x<W;x++){let ent=0,kl=0;for(let c=0;c<NC;c++)if(gt[y][x][c]>0){ent-=gt[y][x][c]*Math.log(gt[y][x][c]);kl+=gt[y][x][c]*Math.log(gt[y][x][c]/Math.max(pred[y][x][c],1e-12))}tw+=ent*kl;te+=ent}return 100*Math.exp(-3*tw/te)}

function loadLookup() { return JSON.parse(fs.readFileSync(path.join(__dirname, 'gt_lookup.json'), 'utf8')); }
function loadModel() {
  try { const m=JSON.parse(fs.readFileSync(path.join(__dirname,'model_weights.json'),'utf8'));return{W1:m.W1,W2:m.W2,NF:m.config.N_FEATURES,NH:m.config.HIDDEN}; } catch { return null; }
}

function lookupPred(ig,H,W,LOOKUP){const pred=[];for(let y=0;y<H;y++){pred[y]=[];for(let x=0;x<W;x++){const raw=ig[y][x];if(raw===5){pred[y][x]=[.002,.002,.002,.002,.002,.990];continue}if(raw===10){pred[y][x]=[.990,.002,.002,.002,.002,.002];continue}const ic=cc(raw),f=Math.min(foodPot(ig,H,W,y,x),4),co=isCoast(ig,H,W,y,x)?1:0;const sd=sdist(ig,H,W,y,x),n=Math.min(nsett(ig,H,W,y,x),3);const db=sd<=3?'near':sd<=7?'mid':sd<=12?'far':'remote';const keys=[`${ic}_${f}_${co}_${db}_${n}`,`${ic}_${f}_${co}_${db}_0`,`${ic}_${Math.min(f,2)}_${co}_${db}_0`,`${ic}_0_${co}_${db}_0`];let p=null;for(const k of keys){if(LOOKUP[k]){p=[...LOOKUP[k]];break}}if(!p)p=[.5,.1,.05,.05,.25,.05];if(!co)p[2]=.002;let s=0;for(let c=0;c<NC;c++){p[c]=Math.max(p[c],.002);s+=p[c]}for(let c=0;c<NC;c++)p[c]/=s;pred[y][x]=p}}return pred}

function extractFeatures(ig,H,W,y,x){const raw=ig[y][x],ic=cc(raw);const f=new Float64Array(17);f[ic]=1;f[6]=foodPot(ig,H,W,y,x)/8;f[7]=isCoast(ig,H,W,y,x)?1:0;f[8]=Math.exp(-sdist(ig,H,W,y,x)/5);f[9]=nsett(ig,H,W,y,x)/8;let ts=0;for(let sy=0;sy<H;sy++)for(let sx=0;sx<W;sx++)if(ig[sy][sx]===1||ig[sy][sx]===2)ts++;f[10]=ts/60;f[11]=localDens(ig,H,W,y,x)/20;f[13]=f[6]*f[8];f[14]=f[7]*f[8];f[15]=f[9]*f[8];f[16]=1;return f}

function modelPredCell(features,MODEL){if(!MODEL)return null;const h=new Float64Array(MODEL.NH);for(let i=0;i<MODEL.NH;i++){let s=0;for(let j=0;j<MODEL.NF;j++)s+=MODEL.W1[i][j]*features[j];h[i]=s>0?s:0}const logits=new Float64Array(NC);for(let i=0;i<NC;i++){let s=0;for(let j=0;j<MODEL.NH;j++)s+=MODEL.W2[i][j]*h[j];logits[i]=s}let mx=-Infinity;for(let i=0;i<NC;i++)if(logits[i]>mx)mx=logits[i];const p=new Float64Array(NC);let se=0;for(let i=0;i<NC;i++){p[i]=Math.exp(logits[i]-mx);se+=p[i]}for(let i=0;i<NC;i++)p[i]=Math.max(p[i]/se,0.002);let s2=0;for(let i=0;i<NC;i++)s2+=p[i];for(let i=0;i<NC;i++)p[i]/=s2;return p}

function runTest(LOOKUP, MODEL, damping, modelWeight, nTrials) {
  const rounds = [];
  try {
    const files = fs.readdirSync(CACHE).filter(f => f.match(/^r\d+_init\.json$/));
    for (const f of files) rounds.push(parseInt(f.match(/r(\d+)/)[1]));
  } catch { return null; }

  const scores = [];
  for (const r of rounds) {
    for (let si = 0; si < 5; si++) {
      const p = path.join(CACHE, `r${r}_gt_s${si}.json`);
      if (!fs.existsSync(p)) continue;
      const d = JSON.parse(fs.readFileSync(p, 'utf8'));
      const ig = d.initial_grid, gt = d.ground_truth, H = d.height, W = d.width;

      for (let t = 0; t < nTrials; t++) {
        const basePred = lookupPred(ig, H, W, LOOKUP);

        // Simulate queries
        const obsTrans = {};
        for (let ic = 0; ic < NC; ic++) obsTrans[ic] = new Float64Array(NC);
        const setts = [];
        for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) if (ig[y][x] === 1 || ig[y][x] === 2) setts.push({y,x});
        for (let q = 0; q < 10; q++) {
          const s = setts[q % setts.length];
          const vx = Math.max(0, Math.min(W-15, s.x-7)), vy = Math.max(0, Math.min(H-15, s.y-7));
          for (let gy = 0; gy < 15; gy++) for (let gx = 0; gx < 15; gx++) {
            const ay = vy+gy, ax = vx+gx; if (ay >= H || ax >= W) continue;
            obsTrans[cc(ig[ay][ax])][sampleFromGT(gt, ay, ax)]++;
          }
        }

        // Compute shift
        const avgRates = {};
        for (let ic = 0; ic < NC; ic++) { avgRates[ic] = new Array(NC).fill(0); let n = 0;
          for (const [k,dist] of Object.entries(LOOKUP)) { if (parseInt(k.split('_')[0]) === ic) { for (let c = 0; c < NC; c++) avgRates[ic][c] += dist[c]; n++; } }
          if (n > 0) for (let c = 0; c < NC; c++) avgRates[ic][c] /= n; }

        const shift = {};
        for (let ic = 0; ic < NC; ic++) {
          const tot = obsTrans[ic].reduce((a,b) => a+b, 0);
          if (tot < 20) { shift[ic] = new Array(NC).fill(1); continue; }
          shift[ic] = [];
          for (let c = 0; c < NC; c++) {
            const obs = obsTrans[ic][c] / tot;
            const avg = avgRates[ic] ? avgRates[ic][c] : 0;
            shift[ic].push(1 + damping * ((avg > 0.01 ? obs / avg : 1) - 1));
          }
        }

        // Apply shift + model blend
        const pred = basePred.map(row => row.map(p => [...p]));
        for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
          const raw = ig[y][x]; if (raw === 5 || raw === 10) continue;
          const ic = cc(raw);

          const shifted = [...pred[y][x]];
          if (shift[ic]) for (let c = 0; c < NC; c++) shifted[c] *= shift[ic][c];

          if (MODEL && modelWeight > 0) {
            const mp = modelPredCell(extractFeatures(ig, H, W, y, x), MODEL);
            if (mp) for (let c = 0; c < NC; c++) pred[y][x][c] = (1-modelWeight) * shifted[c] + modelWeight * mp[c];
            else for (let c = 0; c < NC; c++) pred[y][x][c] = shifted[c];
          } else {
            for (let c = 0; c < NC; c++) pred[y][x][c] = shifted[c];
          }

          const co = isCoast(ig, H, W, y, x); if (!co) pred[y][x][2] = 0.002;
          let s = 0; for (let c = 0; c < NC; c++) { pred[y][x][c] = Math.max(pred[y][x][c], 0.002); s += pred[y][x][c]; }
          for (let c = 0; c < NC; c++) pred[y][x][c] /= s;
        }

        scores.push(scoreKL(pred, gt, H, W));
      }
    }
  }

  const mean = scores.reduce((a,b) => a+b, 0) / scores.length;
  const min = Math.min(...scores);
  return { mean, min, n: scores.length };
}

// ── Main loop ───────────────────────────────────────────────────────────────
let lastSeedCount = countGTSeeds();
let bestConfig = { damping: 0.5, modelWeight: 0.5, score: 0 };
let iteration = 0;

async function loop() {
  iteration++;
  const seedCount = countGTSeeds();
  const newData = seedCount > lastSeedCount;

  if (newData) {
    log(`New GT data: ${seedCount} seeds (was ${lastSeedCount})`);
    lastSeedCount = seedCount;

    // Retrain model
    log('Retraining model...');
    try {
      execSync('node train_model.js 2>&1', { cwd: __dirname, timeout: 180000 });
      log('Model retrained');
    } catch (e) { log('Retrain failed: ' + e.message.slice(0, 80)); }

    // Rebuild lookups
    log('Rebuilding lookups...');
    try {
      const TOKEN = process.env.TOKEN || process.env.AINM_TOKEN;
      execSync(`node calibrate_gt.js ${TOKEN ? '--token ' + TOKEN : ''} 2>&1`, { cwd: __dirname, timeout: 120000 });
      log('Lookups rebuilt');
    } catch (e) { log('Lookup rebuild failed: ' + e.message.slice(0, 80)); }
  }

  // Test different hyperparameters
  const LOOKUP = loadLookup();
  const MODEL = loadModel();

  const configs = [
    { damping: 0.3, modelWeight: 0.0 },
    { damping: 0.4, modelWeight: 0.3 },
    { damping: 0.5, modelWeight: 0.0 },
    { damping: 0.5, modelWeight: 0.3 },
    { damping: 0.5, modelWeight: 0.5 },
    { damping: 0.6, modelWeight: 0.3 },
    { damping: 0.6, modelWeight: 0.5 },
    { damping: 0.7, modelWeight: 0.3 },
  ];

  log(`Iteration ${iteration}: testing ${configs.length} configs × 5 trials...`);

  let bestScore = 0, bestCfg = null;
  for (const cfg of configs) {
    const result = runTest(LOOKUP, MODEL, cfg.damping, cfg.modelWeight, 5);
    if (result && result.mean > bestScore) {
      bestScore = result.mean;
      bestCfg = cfg;
    }
    process.stdout.write('.');
  }
  console.log('');

  if (bestCfg) {
    const improved = bestScore > bestConfig.score;
    log(`Best: damping=${bestCfg.damping} model=${bestCfg.modelWeight} → ${bestScore.toFixed(1)} ${improved ? '▲ NEW BEST' : ''}`);
    if (improved) {
      bestConfig = { ...bestCfg, score: bestScore };
      // Save config for v3 to read
      fs.writeFileSync(path.join(__dirname, 'best_config.json'), JSON.stringify(bestConfig));
      log(`Saved best_config.json: ${JSON.stringify(bestConfig)}`);
    }
  }

  log(`Current best: damping=${bestConfig.damping} model=${bestConfig.modelWeight} score=${bestConfig.score.toFixed(1)} | GT seeds: ${seedCount}`);
}

log('Continuous improvement started');
log(`Initial GT seeds: ${lastSeedCount}`);

// Run immediately, then every 5 minutes
loop().then(() => {
  setInterval(() => loop().catch(e => log('Error: ' + e.message)), 300000);
}).catch(e => log('Error: ' + e.message));
