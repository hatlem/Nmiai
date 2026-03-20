#!/usr/bin/env node
/**
 * Train a neural network on GT data to predict terrain distributions.
 *
 * Input features per cell:
 *   - init_class (one-hot 6)
 *   - food potential (0-8, normalized)
 *   - coastal (0/1)
 *   - settlement distance (continuous, transformed)
 *   - neighbor settlements (0-8, normalized)
 *   - total settlements on map (normalized)
 *   - local density radius 6 (normalized)
 *
 * Output: 6-class probability distribution
 * Loss: KL divergence (directly optimizes competition metric)
 *
 * Architecture: 2-layer MLP with softmax output
 * No dependencies — pure Node.js
 *
 * Usage: node train_model.js
 * Output: astar-island/model_weights.json
 */
const fs = require('fs'), path = require('path');
const NC = 6;
const TTC = {10:0,11:0,0:0,1:1,2:2,3:3,4:4,5:5};

// ── Feature extraction ──────────────────────────────────────────────────────
function cc(c){return TTC[c]??0}
function isCoast(g,H,W,y,x){if(g[y][x]===10||g[y][x]===5)return false;for(const[dy,dx]of[[-1,0],[1,0],[0,-1],[0,1]]){const ny=y+dy,nx=x+dx;if(ny>=0&&ny<H&&nx>=0&&nx<W&&g[ny][nx]===10)return true}return false}
function foodPot(g,H,W,y,x){let c=0;for(let dy=-1;dy<=1;dy++)for(let dx=-1;dx<=1;dx++){if(!dy&&!dx)continue;const ny=y+dy,nx=x+dx;if(ny>=0&&ny<H&&nx>=0&&nx<W&&g[ny][nx]===4)c++}return c}
function settDist(g,H,W,y,x){let m=999;for(let sy=0;sy<H;sy++)for(let sx=0;sx<W;sx++)if(g[sy][sx]===1||g[sy][sx]===2)m=Math.min(m,Math.abs(y-sy)+Math.abs(x-sx));return m}
function nSett(g,H,W,y,x){let c=0;for(let dy=-2;dy<=2;dy++)for(let dx=-2;dx<=2;dx++){if(!dy&&!dx)continue;const ny=y+dy,nx=x+dx;if(ny>=0&&ny<H&&nx>=0&&nx<W){const v=g[ny][nx];if(v===1||v===2)c++}}return c}
function localDens(g,H,W,y,x){let c=0;for(let dy=-6;dy<=6;dy++)for(let dx=-6;dx<=6;dx++){if(!dy&&!dx)continue;const ny=y+dy,nx=x+dx;if(ny>=0&&ny<H&&nx>=0&&nx<W&&(g[ny][nx]===1||g[ny][nx]===2))c++}return c}
function oceanSides(g,H,W,y,x){let c=0;for(const[dy,dx]of[[-1,0],[1,0],[0,-1],[0,1]]){const ny=y+dy,nx=x+dx;if(ny<0||ny>=H||nx<0||nx>=W||g[ny][nx]===10)c++}return c}

const N_FEATURES = 17; // 6 one-hot + 11 continuous

function extractFeatures(ig, H, W, y, x) {
  const raw = ig[y][x];
  const ic = cc(raw);
  const f = new Float64Array(N_FEATURES);
  // One-hot init class
  f[ic] = 1.0;
  // Continuous features (normalized 0-1)
  f[6] = foodPot(ig,H,W,y,x) / 8;
  f[7] = isCoast(ig,H,W,y,x) ? 1 : 0;
  f[8] = Math.exp(-settDist(ig,H,W,y,x) / 5); // exponential decay, near=1, far=0
  f[9] = nSett(ig,H,W,y,x) / 8;
  let totalS=0;for(let sy=0;sy<H;sy++)for(let sx=0;sx<W;sx++)if(ig[sy][sx]===1||ig[sy][sx]===2)totalS++;
  f[10] = totalS / 60; // normalized by typical max
  f[11] = localDens(ig,H,W,y,x) / 20;
  f[12] = oceanSides(ig,H,W,y,x) / 4;
  // Interaction features
  f[13] = f[6] * f[8]; // food × proximity
  f[14] = f[7] * f[8]; // coastal × proximity
  f[15] = f[9] * f[8]; // neighbor_sett × proximity
  f[16] = 1.0; // bias
  return f;
}

// ── Neural Network ──────────────────────────────────────────────────────────
// 2-layer MLP: input(17) → hidden(32) → output(6) with softmax

function initWeights(rows, cols) {
  const w = [];
  const scale = Math.sqrt(2.0 / cols); // He initialization
  for (let i = 0; i < rows; i++) {
    w[i] = new Float64Array(cols);
    for (let j = 0; j < cols; j++) w[i][j] = (Math.random() - 0.5) * scale;
  }
  return w;
}

const HIDDEN = 32;
let W1 = initWeights(HIDDEN, N_FEATURES); // 32 x 17
let W2 = initWeights(NC, HIDDEN);          // 6 x 32

function relu(x) { return x > 0 ? x : 0; }
function reluGrad(x) { return x > 0 ? 1 : 0; }

function forward(features) {
  // Hidden layer
  const h = new Float64Array(HIDDEN);
  for (let i = 0; i < HIDDEN; i++) {
    let sum = 0;
    for (let j = 0; j < N_FEATURES; j++) sum += W1[i][j] * features[j];
    h[i] = relu(sum);
  }
  // Output layer (logits)
  const logits = new Float64Array(NC);
  for (let i = 0; i < NC; i++) {
    let sum = 0;
    for (let j = 0; j < HIDDEN; j++) sum += W2[i][j] * h[j];
    logits[i] = sum;
  }
  // Softmax
  let maxL = -Infinity;
  for (let i = 0; i < NC; i++) if (logits[i] > maxL) maxL = logits[i];
  const p = new Float64Array(NC);
  let sumExp = 0;
  for (let i = 0; i < NC; i++) { p[i] = Math.exp(logits[i] - maxL); sumExp += p[i]; }
  for (let i = 0; i < NC; i++) p[i] /= sumExp;
  // Floor
  for (let i = 0; i < NC; i++) p[i] = Math.max(p[i], 0.002);
  let s = 0; for (let i = 0; i < NC; i++) s += p[i];
  for (let i = 0; i < NC; i++) p[i] /= s;
  return { h, logits, p };
}

function backward(features, gt, fwd, lr) {
  const { h, logits, p } = fwd;
  // dL/dlogits = p - gt (cross-entropy gradient, equivalent to KL minimization)
  const dLogits = new Float64Array(NC);
  for (let i = 0; i < NC; i++) dLogits[i] = p[i] - gt[i];

  // dL/dW2
  for (let i = 0; i < NC; i++)
    for (let j = 0; j < HIDDEN; j++)
      W2[i][j] -= lr * dLogits[i] * h[j];

  // dL/dh
  const dH = new Float64Array(HIDDEN);
  for (let j = 0; j < HIDDEN; j++) {
    let sum = 0;
    for (let i = 0; i < NC; i++) sum += dLogits[i] * W2[i][j];
    dH[j] = sum * reluGrad(h[j]);
  }

  // dL/dW1
  for (let i = 0; i < HIDDEN; i++)
    for (let j = 0; j < N_FEATURES; j++)
      W1[i][j] -= lr * dH[i] * features[j];
}

// ── Load data ───────────────────────────────────────────────────────────────
function loadData() {
  const CACHE = path.join(__dirname, 'cache');
  const examples = [];

  for (const r of [1,2,4,5,6]) {
    for (let si = 0; si < 5; si++) {
      const p = path.join(CACHE, `r${r}_gt_s${si}.json`);
      if (!fs.existsSync(p)) continue;
      const d = JSON.parse(fs.readFileSync(p, 'utf8'));
      const ig = d.initial_grid, gt = d.ground_truth, H = d.height, W = d.width;

      for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
        const raw = ig[y][x];
        if (raw === 5 || raw === 10) continue; // skip static

        const features = extractFeatures(ig, H, W, y, x);
        const target = gt[y][x];

        // Weight by entropy (focus on dynamic cells)
        let entropy = 0;
        for (let c = 0; c < NC; c++) if (target[c] > 0) entropy -= target[c] * Math.log(target[c]);

        examples.push({ features, target, entropy, round: r });
      }
    }
  }
  return examples;
}

// ── Train ───────────────────────────────────────────────────────────────────
function train(examples, epochs, lr, holdoutRound) {
  const trainSet = holdoutRound ? examples.filter(e => e.round !== holdoutRound) : examples;
  const testSet = holdoutRound ? examples.filter(e => e.round === holdoutRound) : examples;

  for (let epoch = 0; epoch < epochs; epoch++) {
    // Shuffle
    for (let i = trainSet.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [trainSet[i], trainSet[j]] = [trainSet[j], trainSet[i]];
    }

    let totalLoss = 0, totalWeight = 0;
    for (const ex of trainSet) {
      const fwd = forward(ex.features);
      // KL loss weighted by entropy
      let kl = 0;
      for (let c = 0; c < NC; c++) {
        if (ex.target[c] > 0) kl += ex.target[c] * Math.log(ex.target[c] / Math.max(fwd.p[c], 1e-12));
      }
      totalLoss += ex.entropy * kl;
      totalWeight += ex.entropy;

      // Backprop (weight learning rate by entropy for important cells)
      const effectiveLR = lr * Math.max(0.1, Math.min(2.0, ex.entropy));
      backward(ex.features, ex.target, fwd, effectiveLR);
    }

    if ((epoch + 1) % 5 === 0 || epoch === 0) {
      const avgLoss = totalWeight > 0 ? totalLoss / totalWeight : 0;
      const score = 100 * Math.exp(-3 * avgLoss);

      // Test score
      let testLoss = 0, testWeight = 0;
      for (const ex of testSet) {
        const fwd = forward(ex.features);
        let kl = 0;
        for (let c = 0; c < NC; c++) {
          if (ex.target[c] > 0) kl += ex.target[c] * Math.log(ex.target[c] / Math.max(fwd.p[c], 1e-12));
        }
        testLoss += ex.entropy * kl;
        testWeight += ex.entropy;
      }
      const testScore = testWeight > 0 ? 100 * Math.exp(-3 * testLoss / testWeight) : 0;

      console.log(`Epoch ${epoch+1}: train=${score.toFixed(1)} test=${testScore.toFixed(1)} (wKL=${avgLoss.toFixed(4)})`);
    }
  }
}

function exportWeights() {
  const weights = {
    W1: W1.map(row => Array.from(row)),
    W2: W2.map(row => Array.from(row)),
    config: { N_FEATURES, HIDDEN, NC }
  };
  const outPath = path.join(__dirname, 'model_weights.json');
  fs.writeFileSync(outPath, JSON.stringify(weights));
  console.log(`\nExported to ${outPath} (${(fs.statSync(outPath).size/1024).toFixed(1)} KB)`);
}

// ── Main ────────────────────────────────────────────────────────────────────
console.log('Loading GT data...');
const examples = loadData();
console.log(`${examples.length} training examples (dynamic cells only)`);

// Leave-one-out validation first
console.log('\n=== Leave-one-out validation ===');
for (const holdout of [1, 2, 4, 5, 6]) {
  W1 = initWeights(HIDDEN, N_FEATURES);
  W2 = initWeights(NC, HIDDEN);
  train(examples, 20, 0.001, holdout);
  console.log('');
}

// Full training with early stopping
console.log('=== Full training (all data, early stopping) ===');
W1 = initWeights(HIDDEN, N_FEATURES);
W2 = initWeights(NC, HIDDEN);

// Save best weights
let bestScore = 0;
let bestW1 = null, bestW2 = null;

for (let epoch = 0; epoch < 30; epoch++) {
  // Shuffle
  for (let i = examples.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [examples[i], examples[j]] = [examples[j], examples[i]];
  }

  // Use 90% train, 10% validation for early stopping
  const split = Math.floor(examples.length * 0.9);
  const trainSet = examples.slice(0, split);
  const valSet = examples.slice(split);

  let totalLoss = 0, totalWeight = 0;
  const lr = 0.0008; // slightly lower LR for stability
  for (const ex of trainSet) {
    const fwd = forward(ex.features);
    let kl = 0;
    for (let c = 0; c < NC; c++) {
      if (ex.target[c] > 0) kl += ex.target[c] * Math.log(ex.target[c] / Math.max(fwd.p[c], 1e-12));
    }
    totalLoss += ex.entropy * kl;
    totalWeight += ex.entropy;

    // L2 regularization
    const effectiveLR = lr * Math.max(0.1, Math.min(1.5, ex.entropy));
    backward(ex.features, ex.target, fwd, effectiveLR);

    // Weight decay
    const decay = 0.0001;
    for (let i = 0; i < HIDDEN; i++)
      for (let j = 0; j < N_FEATURES; j++) W1[i][j] *= (1 - decay);
    for (let i = 0; i < NC; i++)
      for (let j = 0; j < HIDDEN; j++) W2[i][j] *= (1 - decay);
  }

  // Validation
  let valLoss = 0, valWeight = 0;
  for (const ex of valSet) {
    const fwd = forward(ex.features);
    let kl = 0;
    for (let c = 0; c < NC; c++) {
      if (ex.target[c] > 0) kl += ex.target[c] * Math.log(ex.target[c] / Math.max(fwd.p[c], 1e-12));
    }
    valLoss += ex.entropy * kl;
    valWeight += ex.entropy;
  }
  const trainScore = 100 * Math.exp(-3 * totalLoss / totalWeight);
  const valScore = 100 * Math.exp(-3 * valLoss / valWeight);

  if (valScore > bestScore) {
    bestScore = valScore;
    bestW1 = W1.map(r => Float64Array.from(r));
    bestW2 = W2.map(r => Float64Array.from(r));
  }

  if ((epoch + 1) % 3 === 0)
    console.log(`Epoch ${epoch+1}: train=${trainScore.toFixed(1)} val=${valScore.toFixed(1)} best=${bestScore.toFixed(1)}`);
}

// Restore best weights
W1 = bestW1;
W2 = bestW2;
console.log(`\nBest validation score: ${bestScore.toFixed(1)}`);
exportWeights();
