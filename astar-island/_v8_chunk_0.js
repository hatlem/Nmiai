
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
     const 