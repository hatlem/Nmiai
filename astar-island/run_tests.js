#!/usr/bin/env node
/**
 * Run 100+ local simulation tests to validate our strategy.
 * For each GT round+seed, simulate the full pipeline N times with random observations.
 * Reports: mean, std, min, max score per strategy.
 *
 * Usage: node run_tests.js [--trials 20]
 */
const fs = require('fs'), path = require('path');
const NC = 6, TTC = {10:0,11:0,0:0,1:1,2:2,3:3,4:4,5:5};
const CACHE = path.join(__dirname, 'cache');

const TRIALS = parseInt(process.argv.find((_,i,a) => a[i-1] === '--trials') || '20');

function cc(c){return TTC[c]??0}
function isCoast(g,H,W,y,x){if(g[y][x]===10||g[y][x]===5)return false;for(const[dy,dx]of[[-1,0],[1,0],[0,-1],[0,1]]){const ny=y+dy,nx=x+dx;if(ny>=0&&ny<H&&nx>=0&&nx<W&&g[ny][nx]===10)return true}return false}
function foodPot(g,H,W,y,x){let c=0;for(let dy=-1;dy<=1;dy++)for(let dx=-1;dx<=1;dx++){if(!dy&&!dx)continue;const ny=y+dy,nx=x+dx;if(ny>=0&&ny<H&&nx>=0&&nx<W&&g[ny][nx]===4)c++}return c}
function sdist(g,H,W,y,x){let m=999;for(let sy=0;sy<H;sy++)for(let sx=0;sx<W;sx++)if(g[sy][sx]===1||g[sy][sx]===2)m=Math.min(m,Math.abs(y-sy)+Math.abs(x-sx));return m}
function nsett(g,H,W,y,x){let c=0;for(let dy=-2;dy<=2;dy++)for(let dx=-2;dx<=2;dx++){if(!dy&&!dx)continue;const ny=y+dy,nx=x+dx;if(ny>=0&&ny<H&&nx>=0&&nx<W){const v=g[ny][nx];if(v===1||v===2)c++}}return c}
function ctxKey(ig,H,W,y,x){
  const ic=cc(ig[y][x]),f=Math.min(foodPot(ig,H,W,y,x),4),co=isCoast(ig,H,W,y,x)?1:0;
  const sd=sdist(ig,H,W,y,x),n=Math.min(nsett(ig,H,W,y,x),3);
  const db=sd<=3?'near':sd<=7?'mid':sd<=12?'far':'remote';
  return `${ic}_${f}_${co}_${db}_${n}`;
}

function sampleFromGT(gt,y,x){const r=Math.random();let cum=0;for(let c=0;c<NC;c++){cum+=gt[y][x][c];if(r<cum)return c;}return NC-1;}

function scoreKL(pred,gt,H,W){
  let tw=0,te=0;
  for(let y=0;y<H;y++)for(let x=0;x<W;x++){
    let ent=0,kl=0;
    for(let c=0;c<NC;c++)if(gt[y][x][c]>0){ent-=gt[y][x][c]*Math.log(gt[y][x][c]);kl+=gt[y][x][c]*Math.log(gt[y][x][c]/Math.max(pred[y][x][c],1e-12));}
    tw+=ent*kl;te+=ent;
  }
  return 100*Math.exp(-3*tw/te);
}

// Load lookup
const LOOKUP = JSON.parse(fs.readFileSync(path.join(__dirname, 'gt_lookup.json'), 'utf8'));

// Load model
let MODEL = null;
try {
  const mw = JSON.parse(fs.readFileSync(path.join(__dirname, 'model_weights.json'), 'utf8'));
  MODEL = { W1: mw.W1, W2: mw.W2, NF: mw.config.N_FEATURES, NH: mw.config.HIDDEN };
} catch {}

function localDens(g,H,W,y,x){let c=0;for(let dy=-6;dy<=6;dy++)for(let dx=-6;dx<=6;dx++){if(!dy&&!dx)continue;const ny=y+dy,nx=x+dx;if(ny>=0&&ny<H&&nx>=0&&nx<W&&(g[ny][nx]===1||g[ny][nx]===2))c++}return c}

function extractFeatures(ig,H,W,y,x){
  const raw=ig[y][x],ic=cc(raw);const f=new Float64Array(17);
  f[ic]=1;f[6]=foodPot(ig,H,W,y,x)/8;f[7]=isCoast(ig,H,W,y,x)?1:0;
  f[8]=Math.exp(-sdist(ig,H,W,y,x)/5);f[9]=nsett(ig,H,W,y,x)/8;
  let ts=0;for(let sy=0;sy<H;sy++)for(let sx=0;sx<W;sx++)if(ig[sy][sx]===1||ig[sy][sx]===2)ts++;
  f[10]=ts/60;f[11]=localDens(ig,H,W,y,x)/20;f[13]=f[6]*f[8];f[14]=f[7]*f[8];f[15]=f[9]*f[8];f[16]=1;
  return f;
}

function modelPredict(features){
  if(!MODEL)return null;
  const h=new Float64Array(MODEL.NH);
  for(let i=0;i<MODEL.NH;i++){let s=0;for(let j=0;j<MODEL.NF;j++)s+=MODEL.W1[i][j]*features[j];h[i]=s>0?s:0}
  const logits=new Float64Array(NC);
  for(let i=0;i<NC;i++){let s=0;for(let j=0;j<MODEL.NH;j++)s+=MODEL.W2[i][j]*h[j];logits[i]=s}
  let mx=-Infinity;for(let i=0;i<NC;i++)if(logits[i]>mx)mx=logits[i];
  const p=new Float64Array(NC);let se=0;
  for(let i=0;i<NC;i++){p[i]=Math.exp(logits[i]-mx);se+=p[i]}
  for(let i=0;i<NC;i++)p[i]=Math.max(p[i]/se,0.002);
  let s2=0;for(let i=0;i<NC;i++)s2+=p[i];for(let i=0;i<NC;i++)p[i]/=s2;
  return p;
}

function lookupPred(ig,H,W){
  const pred=[];
  for(let y=0;y<H;y++){pred[y]=[];for(let x=0;x<W;x++){
    const raw=ig[y][x];
    if(raw===5){pred[y][x]=[.002,.002,.002,.002,.002,.990];continue}
    if(raw===10){pred[y][x]=[.990,.002,.002,.002,.002,.002];continue}
    const ic=cc(raw),f=Math.min(foodPot(ig,H,W,y,x),4),co=isCoast(ig,H,W,y,x)?1:0;
    const sd=sdist(ig,H,W,y,x),n=Math.min(nsett(ig,H,W,y,x),3);
    const db=sd<=3?'near':sd<=7?'mid':sd<=12?'far':'remote';
    const keys=[`${ic}_${f}_${co}_${db}_${n}`,`${ic}_${f}_${co}_${db}_0`,`${ic}_${Math.min(f,2)}_${co}_${db}_0`,`${ic}_0_${co}_${db}_0`];
    let p=null;for(const k of keys){if(LOOKUP[k]){p=[...LOOKUP[k]];break}}
    if(!p)p=[.5,.1,.05,.05,.25,.05];
    if(!co)p[2]=.002;
    let s=0;for(let c=0;c<NC;c++){p[c]=Math.max(p[c],.002);s+=p[c]}
    for(let c=0;c<NC;c++)p[c]/=s;
    pred[y][x]=p;
  }}
  return pred;
}

function simulateQueries(ig,gt,H,W,nQueries){
  const setts=[];
  for(let y=0;y<H;y++)for(let x=0;x<W;x++)if(ig[y][x]===1||ig[y][x]===2)setts.push({y,x});

  const obsTrans={};
  for(let ic=0;ic<NC;ic++)obsTrans[ic]=new Float64Array(NC);

  for(let q=0;q<nQueries;q++){
    const s=setts[q%setts.length];
    const vx=Math.max(0,Math.min(W-15,s.x-7)),vy=Math.max(0,Math.min(H-15,s.y-7));
    for(let gy=0;gy<15;gy++)for(let gx=0;gx<15;gx++){
      const ay=vy+gy,ax=vx+gx;if(ay>=H||ax>=W)continue;
      obsTrans[cc(ig[ay][ax])][sampleFromGT(gt,ay,ax)]++;
    }
  }
  return obsTrans;
}

function computeShift(obsTrans, damping){
  const avgRates={};
  for(let ic=0;ic<NC;ic++){avgRates[ic]=new Array(NC).fill(0);let n=0;
    for(const[k,dist]of Object.entries(LOOKUP)){if(parseInt(k.split('_')[0])===ic){for(let c=0;c<NC;c++)avgRates[ic][c]+=dist[c];n++}}
    if(n>0)for(let c=0;c<NC;c++)avgRates[ic][c]/=n}

  const shift={};
  for(let ic=0;ic<NC;ic++){
    const tot=obsTrans[ic].reduce((a,b)=>a+b,0);
    if(tot<20){shift[ic]=new Array(NC).fill(1);continue}
    shift[ic]=[];
    for(let c=0;c<NC;c++){
      const obs=obsTrans[ic][c]/tot;
      const avg=avgRates[ic]?avgRates[ic][c]:0;
      const raw=avg>0.01?obs/avg:1;
      shift[ic].push(1+damping*(raw-1));
    }
  }
  return shift;
}

function applyShift(pred,shift,ig,H,W){
  const out=pred.map(row=>row.map(p=>[...p]));
  for(let y=0;y<H;y++)for(let x=0;x<W;x++){
    const raw=ig[y][x];if(raw===5||raw===10)continue;
    const ic=cc(raw);if(!shift[ic])continue;
    for(let c=0;c<NC;c++)out[y][x][c]*=shift[ic][c];
    const co=isCoast(ig,H,W,y,x);if(!co)out[y][x][2]=0.002;
    let s=0;for(let c=0;c<NC;c++){out[y][x][c]=Math.max(out[y][x][c],0.002);s+=out[y][x][c]}
    for(let c=0;c<NC;c++)out[y][x][c]/=s;
  }
  return out;
}

function modelPred(ig,H,W){
  if(!MODEL)return null;
  const pred=[];
  for(let y=0;y<H;y++){pred[y]=[];for(let x=0;x<W;x++){
    const raw=ig[y][x];
    if(raw===5){pred[y][x]=[.002,.002,.002,.002,.002,.990];continue}
    if(raw===10){pred[y][x]=[.990,.002,.002,.002,.002,.002];continue}
    const f=extractFeatures(ig,H,W,y,x);
    const mp=modelPredict(f);
    pred[y][x]=mp?[...mp]:[.5,.1,.05,.05,.25,.05];
    const co=isCoast(ig,H,W,y,x);if(!co)pred[y][x][2]=0.002;
    let s=0;for(let c=0;c<NC;c++){pred[y][x][c]=Math.max(pred[y][x][c],0.002);s+=pred[y][x][c]}
    for(let c=0;c<NC;c++)pred[y][x][c]/=s;
  }}
  return pred;
}

function blendPred(p1,p2,w1,ig,H,W){
  const out=[];
  for(let y=0;y<H;y++){out[y]=[];for(let x=0;x<W;x++){
    out[y][x]=new Array(NC);
    for(let c=0;c<NC;c++)out[y][x][c]=w1*p1[y][x][c]+(1-w1)*p2[y][x][c];
    let s=0;for(let c=0;c<NC;c++){out[y][x][c]=Math.max(out[y][x][c],0.002);s+=out[y][x][c]}
    for(let c=0;c<NC;c++)out[y][x][c]/=s;
  }}
  return out;
}

// ── Run tests ───────────────────────────────────────────────────────────────
console.log(`Running ${TRIALS} trials per seed across all rounds...\n`);

const strategies = {
  'lookup-only': [],
  'shift-d0.3': [],
  'shift-d0.5': [],
  'shift-d0.7': [],
  'model-only': [],
  'shift+model': [],
};

const rounds = [1,2,4,5,6,7];
let testCount = 0;

for (const r of rounds) {
  for (let si = 0; si < 5; si++) {
    const p = path.join(CACHE, `r${r}_gt_s${si}.json`);
    if (!fs.existsSync(p)) continue;
    const d = JSON.parse(fs.readFileSync(p, 'utf8'));
    const ig = d.initial_grid, gt = d.ground_truth, H = d.height, W = d.width;

    const basePred = lookupPred(ig, H, W);
    const baseScore = scoreKL(basePred, gt, H, W);
    strategies['lookup-only'].push(baseScore);

    const mp = modelPred(ig, H, W);
    if (mp) strategies['model-only'].push(scoreKL(mp, gt, H, W));

    // Run trials with random observations
    for (const trial of Array(TRIALS).keys()) {
      const obs = simulateQueries(ig, gt, H, W, 10); // 10 queries per seed

      for (const [damping, key] of [[0.3,'shift-d0.3'],[0.5,'shift-d0.5'],[0.7,'shift-d0.7']]) {
        const shift = computeShift(obs, damping);
        const shifted = applyShift(basePred, shift, ig, H, W);
        strategies[key].push(scoreKL(shifted, gt, H, W));
      }

      if (mp) {
        const shift = computeShift(obs, 0.5);
        const shifted = applyShift(basePred, shift, ig, H, W);
        const blend = blendPred(shifted, mp, 0.5, ig, H, W);
        strategies['shift+model'].push(scoreKL(blend, gt, H, W));
      }
    }
    testCount++;
    if (testCount % 5 === 0) process.stdout.write('.');
  }
}

console.log('\n\n=== RESULTS (' + testCount + ' seeds, ' + TRIALS + ' trials each) ===\n');
console.log('Strategy'.padEnd(16) + 'Mean'.padStart(7) + 'Std'.padStart(7) + 'Min'.padStart(7) + 'Max'.padStart(7) + '  N');
console.log('─'.repeat(55));

for (const [name, scores] of Object.entries(strategies)) {
  if (scores.length === 0) continue;
  const mean = scores.reduce((a,b) => a+b, 0) / scores.length;
  const std = Math.sqrt(scores.reduce((a,b) => a + (b-mean)**2, 0) / scores.length);
  const min = Math.min(...scores);
  const max = Math.max(...scores);
  console.log(name.padEnd(16) + mean.toFixed(1).padStart(7) + std.toFixed(1).padStart(7) + min.toFixed(1).padStart(7) + max.toFixed(1).padStart(7) + ('  '+scores.length));
}

// Per-round breakdown for best strategy
console.log('\n=== PER-ROUND BREAKDOWN (shift-d0.5) ===\n');
for (const r of rounds) {
  const roundScores = [];
  for (let si = 0; si < 5; si++) {
    const p = path.join(CACHE, `r${r}_gt_s${si}.json`);
    if (!fs.existsSync(p)) continue;
    const d = JSON.parse(fs.readFileSync(p, 'utf8'));
    const ig = d.initial_grid, gt = d.ground_truth, H = d.height, W = d.width;
    const basePred = lookupPred(ig, H, W);
    for (let t = 0; t < TRIALS; t++) {
      const obs = simulateQueries(ig, gt, H, W, 10);
      const shift = computeShift(obs, 0.5);
      const shifted = applyShift(basePred, shift, ig, H, W);
      roundScores.push(scoreKL(shifted, gt, H, W));
    }
  }
  const mean = roundScores.reduce((a,b)=>a+b,0)/roundScores.length;
  const min = Math.min(...roundScores);
  const max = Math.max(...roundScores);
  console.log(`R${r}: mean=${mean.toFixed(1)} min=${min.toFixed(1)} max=${max.toFixed(1)} (${roundScores.length} trials)`);
}
