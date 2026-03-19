raw = grid[y][x];
     if (raw === 5 || raw === 10) continue;
     const nObs = counts[y][x].reduce((a, b) => a + b, 0);
     if (nObs === 0) continue;
     const ic = classifyCode(raw);
     const food = Math.min(foodPotential(grid, H, W, y, x), 4);
     const coast = isCoastal(grid, H, W, y, x) ? 1 : 0;
     const nSett = Math.min(countAdj(grid, H, W, y, x, 1) + countAdj(grid, H, W, y, x, 2), 3);
     const sd = Math.min(Math.floor(settDist(grid, H, W, settlements, y, x) / 3), 4);
     const key = `${ic}_${food}_${coast}_${nSett}_${sd}`;
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
  const food = Math.min(foodPotential(grid, H, W, y, x), 4);
  const coast = isCoastal(grid, H, W, y, x) ? 1 : 0;
  const nSett = Math.min(countAdj(grid, H, W, y, x, 1) + countAdj(grid, H, W, y, x, 2), 3);
  const sd = Math.min(Math.floor(settDist(grid, H, W, settlements, y, x) / 3), 4);
  const key = `${ic}_${food}_${coast}_${nSett}_${sd}`;
  const pooled = contextMap[key];
  if (pooled) {
   const total = pooled.reduce((a, b) => a + b, 0);
   if (total > 0) {
    const p = new Float64Array(NUM_CLASSES);
    const alpha = 0.5;
    const denom = total + NUM_CLASSES * alpha;
    for (let c = 0; c < NUM_CLASSES; c++)
     p[c] = (pooled[c] + alpha) / denom;
    return p;
   }
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
     const vy = clamp(sy - Mat