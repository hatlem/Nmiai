
(function () {
 "use strict";
 const API = "https://api.ainm.no/astar-island";
 const POLL_INTERVAL = 30000;
 const API_DELAY = 65;
 const NUM_CLASSES = 6;
 const PROB_FLOOR = 0.01;
 const STATIC_FLOOR = 0.002;
 const REMOTE_FLOOR = 0.003;
 const VIEWPORT_MAX = 15;

 // Calibrated priors per initial terrain class (from calibration.json + domain knowledge)
 const CALIBRATED_PRIORS = {
  0: [0.82, 0.13, 0.012, 0.010, 0.028, 0.005],  // Empty
  1: [0.37, 0.41, 0.008, 0.031, 0.181, 0.005],  // Settlement
  2: [0.36, 0.12, 0.319, 0.021, 0.176, 0.005],  // Port
  3: [0.15, 0.12, 0.03, 0.35, 0.30, 0.05],       // Ruin (CHANGE 6)
  4: [0.07, 0.16, 0.014, 0.012, 0.744, 0.005],  // Forest
  5: [0.002, 0.002, 0.002, 0.002, 0.002, 0.990], // Mountain
 };
 const TERRAIN_TO_CLASS = { 10: 0, 11: 0, 0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5 };

 // Ground truth calibrated context priors from R1 analysis (5 seeds, 8000 cells)
 // Key format: {initClass}_{foodBucket}_{coastal}_{distBucket}
 const GT_CTX_PRIORS = {"0_0_1_far":[0.997266,0.001641,0.000938,0.000039,0.000117,0],"0_1_1_far":[0.996621,0.001862,0.001241,0.000172,0.000103,0],"0_1_1_mid":[0.940438,0.023013,0.02745,0.002988,0.006112,0],"0_0_1_mid":[0.947645,0.019518,0.024174,0.002479,0.006185,0],"0_2_1_mid":[0.913814,0.032468,0.040513,0.003846,0.009359,0],"0_2_1_near":[0.802195,0.064573,0.106098,0.007805,0.019329,0],"0_1_1_near":[0.830683,0.056522,0.085714,0.008758,0.018323,0],"0_0_1_near":[0.8676,0.043567,0.066433,0.0059,0.0165,0],"0_2_1_far":[0.987353,0.007843,0.004412,0.000196,0.000196,0],"4_1_1_mid":[0.022805,0.062195,0.063659,0.006098,0.845244,0],"0_3_1_mid":[0.882935,0.045109,0.056957,0.004891,0.010109,0],"4_3_1_mid":[0.031667,0.071111,0.129444,0.005,0.762778,0],"4_2_1_near":[0.0648,0.105,0.1724,0.016,0.6418,0],"0_3_1_near":[0.694595,0.093784,0.176351,0.012297,0.022973,0],"4_1_1_near":[0.067742,0.12629,0.195,0.016452,0.594516,0],"4_3_1_near":[0.06,0.096667,0.164167,0.009167,0.67,0],"4_2_1_mid":[0.021129,0.064839,0.082581,0.004032,0.827419,0],"0_3_0_near":[0.711319,0.225434,0,0.015882,0.047365,0],"0_3_0_mid":[0.839451,0.126608,0,0.010285,0.023656,0],"0_2_0_mid":[0.835881,0.130559,0,0.01033,0.02323,0],"4_3_0_near":[0.100996,0.223996,0,0.017324,0.657684,0],"4_2_0_near":[0.100788,0.221362,0,0.016183,0.661667,0],"4_3_0_mid":[0.049462,0.139247,0,0.008656,0.802634,0],"4_2_0_mid":[0.050576,0.131212,0,0.009697,0.808515,0],"1_2_0_near":[0.366277,0.418936,0,0.030213,0.184574,0],"0_2_0_near":[0.719964,0.215418,0,0.015815,0.048802,0],"1_3_0_near":[0.353681,0.440417,0,0.034236,0.171667,0],"0_1_0_near":[0.720663,0.21623,0,0.015715,0.047392,0],"4_1_0_mid":[0.053733,0.128567,0,0.0107,0.807,0],"4_1_0_near":[0.097399,0.224753,0,0.017063,0.660785,0],"1_1_0_near":[0.381132,0.406887,0,0.029245,0.182736,0],"0_0_0_near":[0.719589,0.214929,0,0.015911,0.049571,0],"4_0_0_near":[0.1086,0.207733,0,0.0182,0.665467,0],"5_2_0_near":[0,0,0,0,0,1],"5_1_0_mid":[0,0,0,0,0,1],"5_0_0_mid":[0,0,0,0,0,1],"0_0_0_mid":[0.848254,0.119822,0,0.00929,0.022633,0],"5_1_0_near":[0,0,0,0,0,1],"5_2_0_mid":[0,0,0,0,0,1],"0_1_0_far":[0.984737,0.01307,0,0.000877,0.001316,0],"0_2_0_far":[0.986176,0.012882,0,0.000353,0.000588,0],"0_0_0_far":[0.990526,0.008421,0,0.000263,0.000789,0],"0_3_0_far":[0.984263,0.013947,0,0.000737,0.001053,0],"4_3_0_far":[0.002875,0.010875,0,0.001,0.98525,0],"4_1_0_far":[0.001833,0.0135,0,0.000333,0.984333,0],"4_2_0_far":[0.002,0.012143,0,0.000357,0.985500,0],"4_0_0_mid":[0.051058,0.129904,0,0.010673,0.808365,0],"1_0_0_near":[0.4152,0.3564,0,0.029,0.1994,0]};

 window.__pollLog = window.__pollLog || [];
 function log(msg) {
  const ts = new Date().toISOString().slice(11, 19);
  const line = `[${ts}] ${msg}`;
  console.log(line);
  window.__pollLog.push(line);
  if (window.__pollLog.length > 500) window.__pollLog.shift();
 }
 function sleep(ms) { return new Promise((r) => setTimeout(r, ms)); }

 async function apiFetch(path, opts = {}) {
  const url = path.startsWith("http") ? path : `${API}${path}`;
  const resp = await fetch(url, { credentials: "include", ...opts });
  if (!resp.ok) {
   const text = await resp.text().catch(() => "");
   throw new Error(`API ${resp.status}: ${text.slice(0, 200)}`);
  }
  return resp.json();
 }
 async function apiPost(path, body) {
  return apiFetch(path, {
   method: "POST",
   headers: { "Content-Type": "application/json" },
   body: JSON.stringify(body),
  });
 }

 function classifyCode(code) { return TERRAIN_TO_CLASS[code] ?? 0; }
 function classifyGrid(grid, H, W) {
  const out = Array.from({ length: H }, () => new Int32Array(W));
  for (let y = 0; y < H; y++)
   for (let x = 0; x < W; x++)
    out[y][x] = classifyCode(grid[y][x]);
  return out;
 }
 function make3D(H, W, D, val = 0) {
  return Array.from({ length: H }, () =>
   Array.from({ length: W }, () => new Float64Array(D).fill(val))
  );
 }
 function make2D(H, W, val = 0) {
  return Array.from({ length: H }, () => new Float64Array(W).fill(val));
 }
 function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }

 function normalize(p) {
  let s = 0;
  for (let i = 0; i < p.length; i++) s += p[i];
  if (s > 0) for (let i = 0; i < p.length; i++) p[i] /= s;
  return p;
 }

 // CHANGE 1: floorNorm with per-class floors array
 function floorNorm(p, floors) {
  if (floors) {
   for (let i = 0; i < p.length; i++) p[i] = Math.max(p[i], floors[i]);
  } else {
   for (let i = 0; i < p.length; i++) p[i] = Math.max(p[i], PROB_FLOOR);
  }
  return normalize(p);
 }

 // CHANGE 1: Compute per-cell, per-class floors
 function getCellFloors(initCls, isOcean, settDistVal) {
  const floors = new Float64Array(NUM_CLASSES).fill(PROB_FLOOR);
  if (initCls === 5) {
   // Mountain: all non-mountain get tight floor
   floors.fill(STATIC_FLOOR);
  } else if (isOcean) {
   // Ocean: all non-ocean get tight floor
   floors.fill(STATIC_FLOOR);
  } else {
   // All non-mountain cells: mountain class gets tight floor
   floors[5] = STATIC_FLOOR;
   if (initCls === 4) {
    // Forest: settlement/port/ruin unlikely
    floors[1] = REMOTE_FLOOR;
    floors[2] = REMOTE_FLOOR;
    floors[3] = REMOTE_FLOOR;
   }
   if (initCls === 0 && settDistVal > 8) {
    // Remote empty: settlement/port/ruin very unlikely
    floors[1] = REMOTE_FLOOR;
    floors[2] = REMOTE_FLOOR;
    floors[3] = REMOTE_FLOOR;
   }
   if (initCls === 4 && settDistVal > 8) {
    floors[1] = REMOTE_FLOOR;
    floors[2] = REMOTE_FLOOR;
    floors[3] = REMOTE_FLOOR;
   }
  }
  return floors;
 }

 function mountainPrior() {
  const p = new Float64Array(NUM_CLASSES).fill(STATIC_FLOOR);
  p[5] = 1.0 - 5 * STATIC_FLOOR;
  return p;
 }
 function oceanPrior() {
  const p = new Float64Array(NUM_CLASSES).fill(STATIC_FLOOR);
  p[0] = 1.0 - 5 * STATIC_FLOOR;
  return p;
 }

 function isCoastal(grid, H, W, y, x) {
  if (grid[y][x] === 10 || grid[y][x] === 5) return false;
  const dirs = [[-1,0],[1,0],[0,-1],[0,1]];
  for (const [dy,dx] of dirs) {
   const ny = y + dy, nx = x + dx;
   if (ny < 0 || ny >= H || nx < 0 || nx >= W) continue;
   if (grid[ny][nx] === 10) return true;
  }
  return false;
 }
 function coastalMask(grid, H, W) {
  const mask = Array.from({ length: H }, () => new Uint8Array(W));
  for (let y = 0; y < H; y++)
   for (let x = 0; x < W; x++)
    mask[y][x] = isCoastal(grid, H, W, y, x) ? 1 : 0;
  return mask;
 }
 function foodPotential(grid, H, W, y, x) {
  let count = 0;
  for (let dy = -1; dy <= 1; dy++)
   for (let dx = -1; dx <= 1; dx++) {
    if (dy === 0 && dx === 0) continue;
    const ny = y + dy, nx = x + dx;
    if (ny >= 0 && ny < H && nx >= 0 && nx < W && grid[ny][nx] === 4) count++;
   }
  return count;
 }
 function settDist(grid, H, W, settlements, y, x) {
  let minD = 999;
  for (const s of settlements) {
   const d = Math.abs(y - s.y) + Math.abs(x - s.x);
   if (d < minD) minD = d;
  }
  return minD;
 }
 function countAdj(grid, H, W, y, x, val) {
  let c = 0;
  for (let dy = -1; dy <= 1; dy++)
   for (let dx = -1; dx <= 1; dx++) {
    if (dy === 0 && dx === 0) continue;
    const ny = y + dy, nx = x + dx;
    if (ny >= 0 && ny < H && nx >= 0 && nx < W && grid[ny][nx] === val) c++;
   }
  return c;
 }
 // Count settlement/port neighbors within radius 2
 function neighborSettlements(grid, H, W, y, x) {
  let c = 0;
  for (let dy = -2; dy <= 2; dy++)
   for (let dx = -2; dx <= 2; dx++) {
    if (dy === 0 && dx === 0) continue;
    const ny = y + dy, nx = x + dx;
    if (ny >= 0 && ny < H && nx >= 0 && nx < W) {
     const v = grid[ny][nx];
     if (v === 1 || v === 2) c++;
    }
   }
  return c;
 }

 // CHANGE 2: Informative Dirichlet prior per cell
 function getCellPrior(initCls, sd, food, coastal, nSett) {
  const src = CALIBRATED_PRIORS[initCls] || CALIBRATED_PRIORS[0];
  const base = new Float64Array(NUM_CLASSES);
  for (let i = 0; i < NUM_CLASSES; i++) base[i] = src[i];

  // Coastal settlement/port: boost port
  if (coastal && (initCls === 1 || initCls === 2)) {
   base[2] += 0.08;
   base[0] -= 0.04;
  }
  // High food settlement: boost survival
  if (food >= 2 && initCls === 1) {
   base[1] += 0.10;
   base[0] -= 0.05;
   base[3] -= 0.03;
  }
  // Far from settlements: empty stays empty
  if (sd > 6 && initCls === 0) {
   base[0] = 0.92; base[1] = 0.01; base[2] = 0.01;
   base[3] = 0.01; base[4] = 0.04; base[5] = 0.01;
  }
  // Far from settlements: forest stays forest
  if (sd > 6 && initCls === 4) {
   base[4] = 0.90; base[0] = 0.04; base[1] = 0.01;
   base[2] = 0.01; base[3] = 0.01; base[5] = 0.01;
  }
  // Near settlements + empty: boost settlement/ruin
  if (sd <= 3 && initCls === 0) {
   base[1] += 0.06;
   base[3] += 0.03;
   base[0] -= 0.06;
  }
  // Near settlements with many neighbors
  if (nSett >= 2 && initCls === 0 && sd <= 4) {
   base[1] += 0.04;
   base[0] -= 0.03;
  }

  // Ensure non-negative and normalized
  for (let i = 0; i < NUM_CLASSES; i++) base[i] = Math.max(base[i], PROB_FLOOR);
  normalize(base);
  return base;
 }

 // CHANGE 2: Prior strength that decays with observations
 function getPriorStrength(initCls, sd, nObs) {
  let base;
  if (initCls === 5) base = 4.0;
  else if (sd > 6 && initCls === 0) base = 3.0;
  else if (initCls === 1 || initCls === 2) base = 1.5;
  else if (sd <= 3) base = 2.0;
  else base = 2.5;

  if (nObs > 0) base = base / (1.0 + nObs / base);
  return base;
 }

 // Build cross-seed contextual pooling model
 function buildContextPool(initialStates, allCounts, H, W) {
  const contextMap = {};
  for (let si = 0; si < initialStates.length; si++) {
   const grid = initialStates[si].grid;
   const settlements = initialStates[si].settlements || [];
   const counts = allCounts[si];
   if (!counts) continue;
   for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
     const raw = grid[y][x];
     if (raw === 5 || raw === 10) continue;
     const nObs = counts[y][x].reduce((a, b) => a + b, 0);
     if (nObs === 0) continue;
     const ic = classifyCode(raw);
     const food = Math.min(foodPotential(grid, H, W, y, x), 3);
     const coast = isCoastal(grid, H, W, y, x) ? 1 : 0;
     const sd = settDist(grid, H, W, settlements, y, x);
     const distBucket = sd <= 3 ? 'near' : sd <= 7 ? 'mid' : 'far';
     const key = `${ic}_${food}_${coast}_${distBucket}`;
     if (!contextMap[key]) contextMap[key] = new Float64Array(NUM_CLASSES);
     for (let c = 0; c < NUM_CLASSES; c++)
      contextMap[key][c] += counts[y][x][c];
    }
   }
  }
  return contextMap;
 }

 // Get contextual pooling prediction for one cell
 function contextPoolCell(contextMap, grid, H, W, settlements, y, x) {
  const raw = grid[y][x];
  const ic = classifyCode(raw);
  const food = Math.min(foodPotential(grid, H, W, y, x), 3);
  const coast = isCoastal(grid, H, W, y, x) ? 1 : 0;
  const sd = settDist(grid, H, W, settlements, y, x);
  const distBucket = sd <= 3 ? 'near' : sd <= 7 ? 'mid' : 'far';
  const key = `${ic}_${food}_${coast}_${distBucket}`;

  // Try current-round observations first
  const pooled = contextMap[key];
  if (pooled) {
   const total = pooled.reduce((a, b) => a + b, 0);
   if (total > 2) { // Need at least 3 obs for reliable pooling
    const p = new Float64Array(NUM_CLASSES);
    // Blend with GT prior: weight = n/(n+20) for current obs, rest from GT
    const gtPrior = GT_CTX_PRIORS[key];
    const obsWeight = total / (total + 20);
    const alpha = 0.5;
    const denom = total + NUM_CLASSES * alpha;
    for (let c = 0; c < NUM_CLASSES; c++) {
     const obsP = (pooled[c] + alpha) / denom;
     p[c] = gtPrior ? obsWeight * obsP + (1 - obsWeight) * gtPrior[c] : obsP;
    }
    normalize(p);
    return p;
   }
  }

  // Fallback: use ground truth calibrated prior directly
  const gtPrior = GT_CTX_PRIORS[key];
  if (gtPrior) {
   const p = new Float64Array(NUM_CLASSES);
   for (let c = 0; c < NUM_CLASSES; c++) p[c] = Math.max(gtPrior[c], 0.001);
   normalize(p);
   return p;
  }

  return null;
 }

 // CHANGE 5: Belief propagation for unobserved cells
 function beliefPropagation(pred, observedMask, initGrid, H, W) {
  // Compatibility matrix
  const compat = [];
  for (let i = 0; i < NUM_CLASSES; i++) {
   compat[i] = new Float64Array(NUM_CLASSES);
   for (let j = 0; j < NUM_CLASSES; j++)
    compat[i][j] = (i === j) ? 0.75 : (0.25 / NUM_CLASSES);
  }
  // Settlement types cross-compat
  for (const i of [1, 2, 3])
   for (const j of [1, 2, 3])
    if (i !== j) compat[i][j] = 0.12;
  // Forest barely propagates to non-forest
  for (const j of [0, 1, 2, 3]) {
   compat[4][j] = 0.02;
   compat[j][4] = 0.02;
  }
  // Mountain: strong self
  for (let j = 0; j < 5; j++) {
   compat[5][j] = 0.01;
   compat[j][5] = 0.01;
  }
  compat[5][5] = 0.95;

  // Static cells: mountains and ocean are anchored
  const anchorMask = Array.from({ length: H }, () => new Uint8Array(W));
  for (let y = 0; y < H; y++)
   for (let x = 0; x < W; x++)
    anchorMask[y][x] = (observedMask[y][x] || initGrid[y][x] === 5 || initGrid[y][x] === 10) ? 1 : 0;

  const damping = 0.15;
  const dirs = [[-1,0],[1,0],[0,-1],[0,1]];

  // 3 iterations for speed (CHANGE 5)
  for (let iter = 0; iter < 3; iter++) {
   const newPred = make3D(H, W, NUM_CLASSES, 0);
   for (let y = 0; y < H; y++)
    for (let x = 0; x < W; x++)
     for (let c = 0; c < NUM_CLASSES; c++)
      newPred[y][x][c] = pred[y][x][c];

   for (const [dy, dx] of dirs) {
    for (let y = 0; y < H; y++) {
     for (let x = 0; x < W; x++) {
      if (anchorMask[y][x]) continue; // Don't update anchored cells
      const ny = y - dy, nx = x - dx; // neighbor that sends message
      if (ny < 0 || ny >= H || nx < 0 || nx >= W) continue;
      // Message from neighbor: message[c] = sum_c' compat[c][c'] * neighbor[c']
      for (let c = 0; c < NUM_CLASSES; c++) {
       let msg = 0;
       for (let cp = 0; cp < NUM_CLASSES; cp++)
        msg += compat[c][cp] * pred[ny][nx][cp];
       newPred[y][x][c] = (1 - damping) * newPred[y][x][c] + damping * msg;
      }
     }
    }
   }

   // Re-normalize unanchored cells
   for (let y = 0; y < H; y++)
    for (let x = 0; x < W; x++) {
     if (anchorMask[y][x]) continue;
     let s = 0;
     for (let c = 0; c < NUM_CLASSES; c++) {
      newPred[y][x][c] = Math.max(newPred[y][x][c], 1e-12);
      s += newPred[y][x][c];
     }
     if (s > 0) for (let c = 0; c < NUM_CLASSES; c++) newPred[y][x][c] /= s;
    }

   pred = newPred;
  }
  return pred;
 }

 // Infer hidden params from cross-seed observation counts
 function inferParams(initialGrid, observedCounts, H, W) {
  const trans = Array.from({ length: NUM_CLASSES }, () => new Float64Array(NUM_CLASSES));
  for (let y = 0; y < H; y++)
   for (let x = 0; x < W; x++) {
    const ic = classifyCode(initialGrid[y][x]);
    const nObs = observedCounts[y][x].reduce((a, b) => a + b, 0);
    if (nObs > 0)
     for (let c = 0; c < NUM_CLASSES; c++)
      trans[ic][c] += observedCounts[y][x][c];
   }
  const rowSums = trans.map((row) => Math.max(row.reduce((a, b) => a + b, 0), 1));
  const tp = trans.map((row, i) => row.map((v) => v / rowSums[i]));
  const settToRuin = rowSums[1] > 10 ? tp[1][3] : 0.3;
  const settToPort = rowSums[1] > 10 ? tp[1][2] : 0.1;
  const portSurv = rowSums[2] > 5 ? tp[2][2] : 0.3;
  const emptyToForest = rowSums[0] > 20 ? tp[0][4] : 0.05;
  const emptyToSett = rowSums[0] > 20 ? tp[0][1] : 0.05;
  return {
   winter_severity: clamp(settToRuin * 1.5, 0.05, 0.95),
   faction_aggression: clamp(settToRuin * 2.0, 0.05, 0.95),
   trade_activity: clamp((settToPort + portSurv) * 1.2, 0.05, 0.95),
   forest_growth_rate: clamp(emptyToForest * 5.0, 0.02, 0.5),
   expansion_rate: clamp(emptyToSett * 5.0, 0.05, 0.5),
  };
 }

 // CHANGE 7: Query optimizer that concentrates on dynamic cells
 class QueryOptimizer {
  constructor(W, H, seedsCount, budget, initialStates) {
   this.W = W;
   this.H = H;
   this.seedsCount = seedsCount;
   this.budget = budget;
   this.initialStates = initialStates;
   this.settlementCoords = [];
   this.grids = [];
   this.settDistMaps = []; // precompute settlement distance for each seed

   for (let si = 0; si < seedsCount; si++) {
    const state = initialStates[si];
    const grid = state.grid;
    this.grids.push(grid);
    const coords = (state.settlements || [])
     .filter((s) => s.x >= 0 && s.x < W && s.y >= 0 && s.y < H)
     .map((s) => [s.x, s.y]);
    this.settlementCoords.push(coords);

    // Precompute settlement distance map
    const distMap = Array.from({ length: H }, () => new Float64Array(W).fill(999));
    for (const [sx, sy] of coords)
     for (let y = 0; y < H; y++)
      for (let x = 0; x < W; x++) {
       const d = Math.abs(x - sx) + Math.abs(y - sy);
       if (d < distMap[y][x]) distMap[y][x] = d;
      }
    this.settDistMaps.push(distMap);
   }

   // Precompute importance maps: 5=settlement, 2=expansion zone, 0.1=static, 1=other
   this.importanceMaps = [];
   for (let si = 0; si < seedsCount; si++) {
    const imp = Array.from({ length: H }, () => new Float64Array(W));
    const grid = this.grids[si];
    const dist = this.settDistMaps[si];
    for (let y = 0; y < H; y++)
     for (let x = 0; x < W; x++) {
      const v = grid[y][x];
      if (v === 10 || v === 5) { imp[y][x] = 0; continue; }
      if (v === 1 || v === 2 || v === 3) { imp[y][x] = 5; continue; }
      if (dist[y][x] <= 6) { imp[y][x] = 2; continue; }
      imp[y][x] = 0.3; // remote dynamic
     }
    this.importanceMaps.push(imp);
   }

   this.seedViewports = [];
   for (let si = 0; si < seedsCount; si++)
    this.seedViewports.push(this._findSettlementViewports(si));
  }

  _findSettlementViewports(seedIdx) {
   const coords = this.settlementCoords[seedIdx];
   if (coords.length === 0) {
    const cx = Math.max(0, Math.floor(this.W / 2) - Math.floor(VIEWPORT_MAX / 2));
    const cy = Math.max(0, Math.floor(this.H / 2) - Math.floor(VIEWPORT_MAX / 2));
    return [[cx, cy, Math.min(VIEWPORT_MAX, this.W - cx), Math.min(VIEWPORT_MAX, this.H - cy)]];
   }
   const uncovered = new Set(coords.map((_, i) => i));
   const viewports = [];
   while (uncovered.size > 0) {
    let bestVP = null, bestCov = new Set(), bestScore = -1;
    for (const i of uncovered) {
     const [sx, sy] = coords[i];
     const vx = clamp(sx - Math.floor(VIEWPORT_MAX / 2), 0, this.W - VIEWPORT_MAX);
     const vy = clamp(sy - Math.floor(VIEWPORT_MAX / 2), 0, this.H - VIEWPORT_MAX);
     const vw = Math.min(VIEWPORT_MAX, this.W - vx);
     const vh = Math.min(VIEWPORT_MAX, this.H - vy);
     const contained = new Set();
     for (const j of uncovered) {
      const [cx, cy] = coords[j];
      if (cx >= vx && cx < vx + vw && cy >= vy && cy < vy + vh) contained.add(j);
     }
     const dynScore = this._viewportImportance(seedIdx, vx, vy, vw, vh);
     const score = contained.size * 100 + dynScore;
     if (score > bestScore) {
      bestVP = [vx, vy, vw, vh];
      bestCov = contained;
      bestScore = score;
     }
    }
    if (bestVP && bestCov.size > 0) {
     viewports.push(bestVP);
     for (const j of bestCov) uncovered.delete(j);
    } else break;
   }
   return viewports;
  }

  _viewportImportance(seedIdx, x, y, w, h) {
   const imp = this.importanceMaps[seedIdx];
   let total = 0;
   for (let gy = y; gy < y + h && gy < this.H; gy++)
    for (let gx = x; gx < x + w && gx < this.W; gx++)
     total += imp[gy][gx];
   return total;
  }

  // CHANGE 7: Information gain = sum(importance / (1 + nObs)) for dynamic cells
  _viewportInfoGain(seedIdx, x, y, w, h, simCounts) {
   const imp = this.importanceMaps[seedIdx];
   const counts = simCounts[seedIdx];
   let gain = 0;
   for (let gy = y; gy < y + h && gy < this.H; gy++)
    for (let gx = x; gx < x + w && gx < this.W; gx++) {
     const importance = imp[gy][gx];
     if (importance === 0) continue;
     const nObs = counts ? counts[gy][gx] : 0;
     gain += importance / (1 + nObs);
    }
   return gain;
  }

  planQueries() {
   const plan = [];
   // Track per-cell observation counts (simplified)
   const simCounts = {};
   for (let si = 0; si < this.seedsCount; si++)
    simCounts[si] = Array.from({ length: this.H }, () => new Float64Array(this.W));

   const recordQuery = (si, x, y, w, h) => {
    for (let gy = y; gy < y + h && gy < this.H; gy++)
     for (let gx = x; gx < x + w && gx < this.W; gx++)
      simCounts[si][gy][gx] += 1;
   };

   // Phase 1 (10 queries): 1 best viewport per seed, then 2nd best
   const phase1Budget = Math.min(this.seedsCount * 2, Math.floor(this.budget * 0.2));
   for (let pass = 0; pass < 2 && plan.length < phase1Budget; pass++) {
    for (let si = 0; si < this.seedsCount && plan.length < phase1Budget; si++) {
     const vps = this.seedViewports[si];
     const vpIdx = Math.min(pass, vps.length - 1);
     const vp = vps[vpIdx];
     plan.push([si, ...vp]);
     recordQuery(si, vp[0], vp[1], vp[2], vp[3]);
    }
   }

   // Phase 2 (remaining): Adaptive — highest information gain
   // Build candidate viewports: settlement viewports + expansion viewports
   const candidates = [];
   for (let si = 0; si < this.seedsCount; si++) {
    for (const vp of this.seedViewports[si])
     candidates.push({ si, vp });
    // Also add shifted viewports to cover expansion zones
    for (const vp of this.seedViewports[si]) {
     const [vx, vy, vw, vh] = vp;
     // Shift in 4 directions by half viewport
     for (const [dy, dx] of [[-7,0],[7,0],[0,-7],[0,7]]) {
      const nx = clamp(vx + dx, 0, this.W - VIEWPORT_MAX);
      const ny = clamp(vy + dy, 0, this.H - VIEWPORT_MAX);
      const nw = Math.min(VIEWPORT_MAX, this.W - nx);
      const nh = Math.min(VIEWPORT_MAX, this.H - ny);
      candidates.push({ si, vp: [nx, ny, nw, nh] });
     }
    }
   }
   // Deduplicate
   const seen = new Set();
   const dedupCandidates = [];
   for (const c of candidates) {
    const key = `${c.si}_${c.vp.join("_")}`;
    if (!seen.has(key)) {
     seen.add(key);
     dedupCandidates.push(c);
    }
   }

   while (plan.length < this.budget) {
    let bestGain = -1, bestQuery = null;
    for (const c of dedupCandidates) {
     const [x, y, w, h] = c.vp;
     const gain = this._viewportInfoGain(c.si, x, y, w, h, simCounts);
     if (gain > bestGain) {
      bestGain = gain;
      bestQuery = [c.si, ...c.vp];
     }
    }
    if (!bestQuery) break;
    plan.push(bestQuery);
    recordQuery(bestQuery[0], bestQuery[1], bestQuery[2], bestQuery[3], bestQuery[4]);
   }

   return plan.slice(0, this.budget);
  }

  nextQuery(counts) {
   let bestGain = -1, bestQ = null;
   for (let si = 0; si < this.seedsCount; si++) {
    const seedCounts = counts[si];
    for (const vp of this.seedViewports[si]) {
     const [x, y, w, h] = vp;
     if (!seedCounts) return [si, x, y, w, h];
     let gain = 0;
     const imp = this.importanceMaps[si];
     for (let gy = y; gy < y + h && gy < this.H; gy++)
      for (let gx = x; gx < x + w && gx < this.W; gx++) {
       const importance = imp[gy][gx];
       if (importance === 0) continue;
       const nObs = seedCounts[gy][gx].reduce((a, b) => a + b, 0);
       gain += importance / (1 + nObs);
      }
     if (gain > bestGain) {
      bestGain = gain;
      bestQ = [si, x, y, w, h];
     }
    }
   }
   return bestQ || [0, 0, 0, VIEWPORT_MAX, VIEWPORT_MAX];
  }
 }

 // CHANGE 3: Layered prediction — replaces geometric ensemble
 function layeredPredict(
  grid, settlements, counts, contextMap, initialStates, seedIdx, H, W
 ) {
  const pred = make3D(H, W, NUM_CLASSES, 0);
  const coastal = coastalMask(grid, H, W);
  const observedMask = Array.from({ length: H }, () => new Uint8Array(W));

  for (let y = 0; y < H; y++) {
   for (let x = 0; x < W; x++) {
    const raw = grid[y][x];

    // Hard constraints for static terrain
    if (raw === 5) { pred[y][x] = mountainPrior(); continue; }
    if (raw === 10) { pred[y][x] = oceanPrior(); continue; }

    const ic = classifyCode(raw);
    const isOcean = false;
    const food = foodPotential(grid, H, W, y, x);
    const sd = settDist(grid, H, W, settlements, y, x);
    const isCoast = !!coastal[y][x];
    const nSett = neighborSettlements(grid, H, W, y, x);
    const nObs = counts[y][x].reduce((a, b) => a + b, 0);

    // Get informative prior (CHANGE 2)
    const prior = getCellPrior(ic, sd, food, isCoast, nSett);
    const strength = getPriorStrength(ic, sd, nObs);

    // Per-cell floors (CHANGE 1)
    const cellFloors = getCellFloors(ic, isOcean, sd);

    if (nObs >= 5) {
     // Layer 1: Pure KT with informative Dirichlet prior (data dominates)
     observedMask[y][x] = 1;
     const denom = nObs + strength;
     for (let c = 0; c < NUM_CLASSES; c++)
      pred[y][x][c] = (counts[y][x][c] + prior[c] * strength) / denom;
     floorNorm(pred[y][x], cellFloors);

    } else if (nObs >= 1) {
     // Layer 2: 60% KT + 40% contextual pooling (CHANGE 3)
     observedMask[y][x] = 1;
     const denom = nObs + strength;
     const kt = new Float64Array(NUM_CLASSES);
     for (let c = 0; c < NUM_CLASSES; c++)
      kt[c] = (counts[y][x][c] + prior[c] * strength) / denom;

     const ctxPred = contextPoolCell(contextMap, grid, H, W, settlements, y, x);
     if (ctxPred) {
      for (let c = 0; c < NUM_CLASSES; c++)
       pred[y][x][c] = 0.6 * kt[c] + 0.4 * ctxPred[c];
     } else {
      for (let c = 0; c < NUM_CLASSES; c++)
       pred[y][x][c] = kt[c];
     }
     floorNorm(pred[y][x], cellFloors);

    } else {
     // Layer 3: Pure contextual pooling or calibrated prior (CHANGE 3)
     const ctxPred = contextPoolCell(contextMap, grid, H, W, settlements, y, x);
     if (ctxPred) {
      for (let c = 0; c < NUM_CLASSES; c++)
       pred[y][x][c] = ctxPred[c];
     } else {
      for (let c = 0; c < NUM_CLASSES; c++)
       pred[y][x][c] = prior[c];
     }

     // Non-coastal: suppress port
     if (!isCoast) pred[y][x][2] = cellFloors[2];
     floorNorm(pred[y][x], cellFloors);
    }
   }
  }

  // CHANGE 5: Belief propagation for unobserved cells
  const result = beliefPropagation(pred, observedMask, grid, H, W);

  // Final: apply hard constraints + floors
  for (let y = 0; y < H; y++) {
   for (let x = 0; x < W; x++) {
    const raw = grid[y][x];
    if (raw === 5) { result[y][x] = mountainPrior(); continue; }
    if (raw === 10) { result[y][x] = oceanPrior(); continue; }
    const ic = classifyCode(raw);
    const sd = settDist(grid, H, W, settlements, y, x);
    const cellFloors = getCellFloors(ic, false, sd);
    if (!coastal[y][x]) result[y][x][2] = Math.max(result[y][x][2], cellFloors[2]);
    floorNorm(result[y][x], cellFloors);
   }
  }

  return result;
 }

 // ── Main pipeline ──────────────────────────────────────────────────────────
 async function runPipeline(roundId) {
  log(`Pipeline v8 starting for round ${roundId}`);
  const t0 = performance.now();
  const detail = await apiFetch(`/rounds/${roundId}`);
  const W = detail.map_width;
  const H = detail.map_height;
  const seedsCount = detail.seeds_count;
  const initialStates = detail.initial_states;
  log(`Round ${detail.round_number}: ${W}x${H}, ${seedsCount} seeds, ` +
   `${initialStates.map((s) => (s.settlements || []).length).join("/")} settlements`);

  let budget;
  try {
   budget = await apiFetch("/budget");
  } catch (e) {
   log(`Budget check failed: ${e.message}. Assuming 50 queries.`);
   budget = { queries_used: 0, queries_max: 50 };
  }
  const queriesLeft = budget.queries_max - budget.queries_used;
  log(`Budget: ${budget.queries_used}/${budget.queries_max} used, ${queriesLeft} remaining`);

  const allCounts = {};
  for (let si = 0; si < seedsCount; si++)
   allCounts[si] = make3D(H, W, NUM_CLASSES, 0);

  if (queriesLeft > 0) {
   const optimizer = new QueryOptimizer(W, H, seedsCount, queriesLeft, initialStates);
   const plan = optimizer.planQueries();
   log(`Query plan: ${plan.length} queries across ${seedsCount} seeds`);

   let queriesExecuted = 0;
   for (let qi = 0; qi < plan.length; qi++) {
    // After 60% of plan, switch to adaptive
    let query;
    if (qi >= plan.length * 0.6) {
     query = optimizer.nextQuery(allCounts);
    } else {
     query = plan[qi];
    }
    const [seedIdx, qx, qy, qw, qh] = query;
    try {
     await sleep(API_DELAY);
     const result = await apiPost("/simulate", {
      round_id: roundId,
      seed_index: seedIdx,
      viewport_x: qx,
      viewport_y: qy,
      viewport_w: qw,
      viewport_h: qh,
     });
     const vp = result.viewport;
     for (let gy = 0; gy < result.grid.length; gy++) {
      for (let gx = 0; gx < result.grid[gy].length; gx++) {
       const absY = vp.y + gy;
       const absX = vp.x + gx;
       if (absY < H && absX < W) {
        const cls = classifyCode(result.grid[gy][gx]);
        allCounts[seedIdx][absY][absX][cls] += 1;
       }
      }
     }
     queriesExecuted++;
     if (queriesExecuted % 10 === 0)
      log(` Queries: ${queriesExecuted}/${plan.length} (budget: ${result.queries_used}/${result.queries_max})`);
    } catch (e) {
     log(` Query ${qi} failed: ${e.message}`);
     if (e.message.includes("429")) {
      log(" Rate limited or budget exhausted, stopping queries.");
      break;
     }
    }
   }
   log(`Observation complete: ${queriesExecuted} queries executed`);
  }

  log("Building predictions (v8: layered + belief propagation)...");

  // Build cross-seed contextual pooling model
  const contextMap = buildContextPool(initialStates, allCounts, H, W);
  log(` Contextual pool: ${Object.keys(contextMap).length} context groups`);

  // Infer params for logging
  const globalCounts = make3D(H, W, NUM_CLASSES, 0);
  const refGrid = initialStates[0].grid;
  for (let si = 0; si < seedsCount; si++)
   for (let y = 0; y < H; y++)
    for (let x = 0; x < W; x++)
     for (let c = 0; c < NUM_CLASSES; c++)
      globalCounts[y][x][c] += allCounts[si][y][x][c];

  let inferredParams;
  try {
   inferredParams = inferParams(refGrid, globalCounts, H, W);
   log(`Inferred params: winter=${inferredParams.winter_severity.toFixed(2)}, ` +
    `aggression=${inferredParams.faction_aggression.toFixed(2)}, ` +
    `trade=${inferredParams.trade_activity.toFixed(2)}`);
  } catch (e) {
   log(`Param inference failed: ${e.message}`);
  }

  const predictions = {};
  for (let si = 0; si < seedsCount; si++) {
   const t1 = performance.now();
   const grid = initialStates[si].grid;
   const settlements = initialStates[si].settlements || [];
   const counts = allCounts[si];

   // CHANGE 3: Layered prediction (no ensemble, no temperature, no MC sim)
   predictions[si] = layeredPredict(
    grid, settlements, counts, contextMap, initialStates, si, H, W
   );

   const dt = performance.now() - t1;
   let obsCount = 0;
   for (let y = 0; y < H; y++)
    for (let x = 0; x < W; x++)
     if (counts[y][x].reduce((a, b) => a + b, 0) > 0) obsCount++;
   const obsPct = (100 * obsCount / (H * W)).toFixed(0);
   log(` Seed ${si}: ${obsPct}% observed, ${dt.toFixed(0)}ms`);
  }

  log("Submitting predictions...");
  for (let si = 0; si < seedsCount; si++) {
   try {
    await sleep(API_DELAY);
    const predArray = [];
    for (let y = 0; y < H; y++) {
     const row = [];
     for (let x = 0; x < W; x++)
      row.push(Array.from(predictions[si][y][x]));
     predArray.push(row);
    }
    const resp = await apiPost("/submit", {
     round_id: roundId,
     seed_index: si,
     prediction: predArray,
    });
    log(` Seed ${si}: ${resp.status || "accepted"}`);
   } catch (e) {
    log(` Seed ${si} submit FAILED: ${e.message}`);
   }
  }
  const totalTime = ((performance.now() - t0) / 1000).toFixed(1);
  log(`Pipeline v8 complete in ${totalTime}s`);
  return true;
 }

 // ── Auto-calibration from completed rounds ────────────────────────────────
 window.__calibratedRounds = window.__calibratedRounds || {};

 async function calibrateFromRound(roundId, seedsCount) {
  log(`Auto-calibrating from round ${roundId}...`);
  const ctxStats = {};

  for (let si = 0; si < seedsCount; si++) {
   try {
    await sleep(200);
    const data = await apiFetch(`/analysis/${roundId}/${si}`);
    const H = data.height, W = data.width;
    const gt = data.ground_truth;
    const ig = data.initial_grid;

    for (let y = 0; y < H; y++) {
     for (let x = 0; x < W; x++) {
      const raw = ig[y][x];
      const ic = classifyCode(raw);
      let food = 0;
      for (let dy = -1; dy <= 1; dy++)
       for (let dx = -1; dx <= 1; dx++) {
        if (dy === 0 && dx === 0) continue;
        const ny = y+dy, nx = x+dx;
        if (ny >= 0 && ny < H && nx >= 0 && nx < W && ig[ny][nx] === 4) food++;
       }
      food = Math.min(food, 3);
      let coastal = 0;
      for (const [dy,dx] of [[-1,0],[1,0],[0,-1],[0,1]]) {
       const ny = y+dy, nx = x+dx;
       if (ny >= 0 && ny < H && nx >= 0 && nx < W && ig[ny][nx] === 10) coastal = 1;
      }
      let sd = 999;
      for (let sy = Math.max(0,y-12); sy < Math.min(H,y+12); sy++)
       for (let sx = Math.max(0,x-12); sx < Math.min(W,x+12); sx++)
        if (ig[sy][sx] === 1 || ig[sy][sx] === 2) sd = Math.min(sd, Math.abs(y-sy)+Math.abs(x-sx));
      const distBucket = sd <= 3 ? 'near' : sd <= 7 ? 'mid' : 'far';
      const key = `${ic}_${food}_${coastal}_${distBucket}`;
      if (!ctxStats[key]) ctxStats[key] = {counts: new Float64Array(6), n: 0};
      for (let c = 0; c < 6; c++) ctxStats[key].counts[c] += gt[y][x][c];
      ctxStats[key].n++;
     }
    }
   } catch (e) { log(`  Seed ${si} analysis failed: ${e.message}`); }
  }

  // Update GT_CTX_PRIORS with new data (blend with existing)
  let updated = 0;
  for (const [key, stats] of Object.entries(ctxStats)) {
   if (stats.n < 3) continue;
   const avg = Array.from(stats.counts);
   const sum = avg.reduce((a,b) => a+b, 0);
   if (sum <= 0) continue;
   for (let i = 0; i < 6; i++) avg[i] /= sum;
   if (GT_CTX_PRIORS[key]) {
    // Blend: 40% old + 60% new (new data is more recent)
    for (let i = 0; i < 6; i++)
     GT_CTX_PRIORS[key][i] = 0.4 * GT_CTX_PRIORS[key][i] + 0.6 * avg[i];
   } else {
    GT_CTX_PRIORS[key] = avg;
   }
   updated++;
  }
  window.__calibratedRounds[roundId] = true;
  log(`Calibrated ${updated} context buckets from round ${roundId}`);
 }

 // ── Auto-poller ────────────────────────────────────────────────────────────
 const completedRounds = new Set();
 let pollTimer = null;
 let running = false;

 async function checkAndRun() {
  if (running) { log("Pipeline already running, skipping poll"); return; }
  try {
   try {
    const myRounds = await apiFetch("/my-rounds");
    for (const r of myRounds) {
     if (r.queries_used >= r.queries_max || r.seeds_submitted >= (r.seeds_count || 5))
      completedRounds.add(r.id);
     // Auto-calibrate from completed rounds with scores
     if (r.status === 'completed' && r.round_score && !window.__calibratedRounds?.[r.id]) {
      calibrateFromRound(r.id, r.seeds_count || 5).catch(e => log(`Calibration failed: ${e.message}`));
     }
    }
   } catch (e) {}
   const rounds = await apiFetch("/rounds");
   const active = rounds.filter((r) => r.status === "active" && !completedRounds.has(r.id));
   if (active.length === 0) {
    log("No new active rounds.");
    return;
   }
   for (const round of active) {
    log(`New active round detected: #${round.round_number} (${round.id})`);
    running = true;
    try {
     const success = await runPipeline(round.id);
     if (success) completedRounds.add(round.id);
    } catch (e) {
     log(`Pipeline error: ${e.message}`);
     console.error(e);
    } finally {
     running = false;
    }
   }
  } catch (e) {
   log(`Poll error: ${e.message}`);
  }
 }

 function startPoll() {
  if (pollTimer) clearInterval(pollTimer);
  log("v8 Auto-poller started (30s interval). window.__stopPoll() to stop.");
  checkAndRun();
  pollTimer = setInterval(checkAndRun, POLL_INTERVAL);
 }
 function stopPoll() {
  if (pollTimer) {
   clearInterval(pollTimer);
   pollTimer = null;
   log("Auto-poller stopped.");
  }
 }
 window.__stopPoll = stopPoll;
 window.__startPoll = startPoll;
 window.__runNow = async function (roundId) {
  if (running) { log("Already running"); return; }
  running = true;
  try { await runPipeline(roundId); }
  catch (e) { log(`Manual run error: ${e.message}`); console.error(e); }
  finally { running = false; }
 };
 window.__completedRounds = completedRounds;
 startPoll();
})();
