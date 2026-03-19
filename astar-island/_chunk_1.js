 s.alive = false;
 grid[s.y][s.x] = 3;
 const nearby = settlements.filter(
 (o) => o.alive && o.owner_id === s.owner_id && o !== s &&
 Math.abs(s.y - o.y) + Math.abs(s.x - o.x) <= 5
 );
 if (nearby.length > 0 && s.population > 0) {
 const perSett = (s.population * 0.5) / nearby.length;
 for (const nf of nearby) nf.population += perSett;
 }
 s.population = 0;
 }
 }
 }
 _phaseEnvironment(grid, settlements, rng) {
 const p = this.params;
 const coastal = coastalMask(grid, this.H, this.W);
 const aliveSettlements = settlements.filter((s) => s.alive);
 for (let y = 0; y < this.H; y++) {
 for (let x = 0; x < this.W; x++) {
 if (grid[y][x] === 3) {
 let reclaimed = false;
 for (const s of aliveSettlements) {
 const dist = Math.abs(y - s.y) + Math.abs(x - s.x);
 if (dist <= 3 && s.population > 1.5 && s.food > 1.0) {
 if (rng() < p.ruin_reclaim_rate * (0.5 + 0.5 * rng())) {
 const isPort = !!coastal[y][x];
 settlements.push({
 x, y,
 population: s.population * 0.2,
 food: s.food * 0.2,
 wealth: s.wealth * 0.1,
 defense: 0.3,
 has_port: isPort,
 alive: true,
 owner_id: s.owner_id,
 tech_level: s.tech_level * 0.3,
 has_longship: false,
 });
 s.population *= 0.8;
 s.food *= 0.8;
 grid[y][x] = isPort ? 2 : 1;
 reclaimed = true;
 break;
 }
 }
 }
 if (!reclaimed) {
 const adjForest = countAdj(grid, this.H, this.W, y, x, 4);
 const prob = adjForest > 0
 ? p.forest_growth_rate * 0.15 * Math.min(adjForest, 3)
 : p.forest_growth_rate * 0.02;
 if (rng() < prob) grid[y][x] = 4;
 else if (rng() < 0.03) grid[y][x] = 11;
 }
 } else if (grid[y][x] === 11 || grid[y][x] === 0) {
 const adjForest = countAdj(grid, this.H, this.W, y, x, 4);
 if (adjForest > 0) {
 const prob = p.forest_growth_rate * 0.02 * Math.min(adjForest, 3);
 if (rng() < prob) grid[y][x] = 4;
 }
 }
 }
 }
 }
 run(seed) {
 const rng = xoshiro128(seed);
 const grid = this._cloneGrid();
 let settlements = this._initSettlements();
 this._rebuildGrid(grid, settlements);
 for (let year = 0; year < 50; year++) {
 this._phaseGrowth(grid, settlements, rng);
 this._phaseConflict(grid, settlements, rng);
 this._phaseTrade(grid, settlements, rng);
 this._phaseWinter(grid, settlements, rng);
 this._phaseEnvironment(grid, settlements, rng);
 this._rebuildGrid(grid, settlements);
 if (year % 10 === 9) {
 settlements = settlements.filter((s) => s.alive);
 }
 }
 const cls = Array.from({ length: this.H }, () => new Int32Array(this.W));
 for (let y = 0; y < this.H; y++)
 for (let x = 0; x < this.W; x++)
 cls[y][x] = classifyCode(grid[y][x]);
 return cls;
 }
 runMonteCarlo(nRuns) {
 const counts = make3D(this.H, this.W, NUM_CLASSES, 0);
 for (let i = 0; i < nRuns; i++) {
 const cls = this.run(i * 7919 + 42); 
 for (let y = 0; y < this.H; y++)
 for (let x = 0; x < this.W; x++)
 counts[y][x][cls[y][x]] += 1;
 }
 const alpha = 0.5;
 const denom = nRuns + NUM_CLASSES * alpha;
 const probs = make3D(this.H, this.W, NUM_CLASSES, 0);
 for (let y = 0; y < this.H; y++)
 for (let x = 0; x < this.W; x++) {
 for (let c = 0; c < NUM_CLASSES; c++)
 probs[y][x][c] = (counts[y][x][c] + alpha) / denom;
 floorNorm(probs[y][x]);
 }
 return probs;
 }
 }
 function inferParams(initialGrid, observedCounts, H, W) {
 const trans = Array.from({ length: NUM_CLASSES }, () => new Float64Array(NUM_CLASSES));
 for (let y = 0; y < H; y++) {
 for (let x = 0; x < W; x++) {
 const ic = classifyCode(initialGrid[y][x]);
 const nObs = observedCounts[y][x].reduce((a, b) => a + b, 0);
 if (nObs > 0) {
 for (let c = 0; c < NUM_CLASSES; c++)
 trans[ic][c] += observedCounts[y][x][c];
 }
 }
 }
 const rowSums = trans.map((row) => Math.max(row.reduce((a, b) => a + b, 0), 1));
 const tp = trans.map((row, i) => row.map((v) => v / rowSums[i]));
 const settToRuin = rowSums[1] > 10 ? tp[1][3] : 0.3;
 const settToPort = rowSums[1] > 10 ? tp[1][2] : 0.1;
 const portSurv = rowSums[2] > 5 ? tp[2][2] : 0.3;
 const emptyToForest = rowSums[0] > 20 ? tp[0][4] : 0.05;
 const emptyToSett = rowSums[0] > 20 ? tp[0][1] : 0.05;
 const ruinToSett = rowSums[3] > 5 ? tp[3][1] : 0.1;
 const portToRuin = rowSums[2] > 5 ? tp[2][3] : 0.2;
 const settSurv = rowSums[1] > 10 ? tp[1][1] : 0.5;
 const params = {};
 params.winter_severity = clamp(settToRuin * 1.5 + portToRuin * 0.5, 0.05, 0.95);
 params.faction_aggression = clamp(settToRuin * 2.0, 0.05, 0.95);
 params.trade_activity = clamp((settToPort + portSurv) * 1.2, 0.05, 0.95);
 params.forest_growth_rate = clamp(emptyToForest * 5.0, 0.02, 0.5);
 params.expansion_rate = clamp(emptyToSett * 5.0, 0.05, 0.5);
 params.ruin_reclaim_rate = clamp(ruinToSett * 2.0, 0.05, 0.5);
 params.food_per_forest = clamp(settSurv * 0.5, 0.1, 0.6);
 params.port_development_threshold = clamp(1.0 - settToPort * 3.0, 0.2, 0.8);
 params.raid_range = clamp(3.0 + params.faction_aggression * 5.0, 3.0, 8.0);
 return params;
 }
 class QueryOptimizer {
 constructor(W, H, seedsCount, budget, initialStates) {
 this.W = W;
 this.H = H;
 this.seedsCount = seedsCount;
 this.budget = budget;
 this.initialStates = initialStates;
 this.settlementCoords = [];
 this.grids = [];
 for (let si = 0; si < seedsCount; si++) {
 const state = initialStates[si];
 const grid = state.grid;
 this.grids.push(grid);
 const coords = (state.settlements || [])
 .filter((s) => s.x >= 0 && s.x < W && s.y >= 0 && s.y < H)
 .map((s) => [s.x, s.y]);
 this.settlementCoords.push(coords);
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
 const dynScore = this._viewportDynScore(seedIdx, vx, vy, vw, vh);
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
 _viewportDynScore(seedIdx, x, y, w, h) {
 const grid = this.grids[seedIdx];
 let dyn = 0, sett = 0;
 for (let gy = y; gy < y + h && gy < this.H; gy++)
 for (let gx = x; gx < x + w && gx < this.W; gx++) {
 const v = grid[gy][gx];
 if (v !== 10 && v !== 5) dyn++;
 if (v === 1 || v === 2 || v === 3) sett++;
 }
 return dyn + sett * 5;
 }
 planQueries() {
 const plan = [];
 for (let si = 0; si < this.seedsCount; si++)
 for (const vp of this.seedViewports[si])
 plan.push([si, ...vp]);
 const remaining = this.budget - plan.length;
 if (remaining > 0) {
 const scoredVPs = [];
 for (let si = 0; si < this.seedsCount; si++)
 for (const vp of this.seedViewports[si]) {
 const score = this._viewportDynScore(si, vp[0], vp[1], vp[2], vp[3]);
 scoredVPs.push({ si, vp, score });
 }
 scoredVPs.sort((a, b) => b.score - a.score);
 for (let i = 0; i < remaining; i++) {
 const entry = scoredVPs[i % scoredVPs.length];
 plan.push([entry.si, ...entry.vp]);
 }
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
 for (let gy = y; gy < y + h && gy < this.H; gy++)
 for (let gx = x; gx < x + w && gx < this.W; gx++) {
 const v = this.grids[si][gy][gx];
 const imp = (v === 10 || v === 5) ? 0 : (v === 1 || v === 2 || v === 3) ? 5 : 1;
 const nObs = seedCounts[gy][gx].reduce((a, b) => a + b, 0);
 gain += imp / (1 + nObs);
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
 function statisticalPredict(initialGrid, counts, H, W) {
 const pred = make3D(H, W, NUM_CLASSES, 0);
 const ktAlpha = 0.5;
 for (let y = 0; y < H; y++) {
 for (let x = 0; x < W; x++) {
 const raw = initialGrid[y][x];
 if (raw === 5) { pred[y][x] = mountainPrior(); continue; }
 if (raw === 10) { pred[y][x] = oceanPrior(); continue; }
 const nObs = counts[y][x].reduce((a, b) => a + b, 0);
 if (nObs > 0) {
 const denom = nObs + NUM_CLASSES * ktAlpha;
 for (let c = 0; c < NUM_CLASSES; c++)
 pred[y][x][c] = (counts[y][x][c] + ktAlpha) / denom;
 } else {
 pred[y][x] = calibratedPrior(classifyCode(raw));
 }
 floorNorm(pred[y][x]);
 }
 }
 return pred;
 }
 function transitionPredict(initialGrid, counts, H, W) {
 const trans = Array.from({ length: NUM_CLASSES }, () => new Float64Array(NUM_CLASSES).fill(0.5));
 for (let i = 0; i < NUM_CLASSES; i++) trans[i][i] = 5.0;
 for (let y = 0; y < H; y++)
 for (let x = 0; x < W; x++) {
 const ic = classifyCode(initialGrid[y][x]);
 for (let c = 0; c < NUM_CLASSES; c++)
 trans[ic][c] += counts[y][x][c];
 }
 for (let i = 0; i < NUM_CLASSES; i++) {
 const s = trans[i].reduce((a, b) => a + b, 0);
 if (s > 0) for (let c = 0; c < NUM_CLASSES; c++) trans[i][c] /= s;
 }
 const pred = make3D(H, W, NUM_CLASSES, 0);
 for (let y = 0; y < H; y++)
 for (let x = 0; x < W; x++) {
 const raw = initialGrid[y][x];
 if (raw === 5) { pred[y][x] = mountainPrior(); continue; }
 if (raw === 10) { pred[y][x] = oceanPrior(); continue; }
 const ic = classifyCode(raw);
 for (let c = 0; c < NUM_CLASSES; c++) pred[y][x][c] = trans[ic][c];
 floorNorm(pred[y][x]);
 }
 return pred;
 }
 function heuristicPredict(initialGrid, settlements, inferredParams, H, W) {
 const pred = make3D(H, W, NUM_CLASSES, 0);
 const coastal = coastalMask(initialGrid, H, W);
 const aggression = inferredParams.faction_aggression ?? 0.3;
 const winter = inferredParams.winter_severity ?? 0.4;
 const trade = inferredParams.trade_activity ?? 0.5;
 const forestGrowth = inferredParams.forest_growth_rate ?? 0.15;
 for (let y = 0; y < H; y++) {
 for (let x = 0; x < W; x++) {
 const raw = initialGrid[y][x];
 if (raw === 5) { pred[y][x] = mountainPrior(); continue; }
 if (raw === 10) { pred[y][x] = oceanPrior(); continue; }
 const ic = classifyCode(raw);
 const dist = calibratedPrior(ic);
 const food = foodPotential(initialGrid, H, W, y, x);
 const sd = settDist(initialGrid, H, W, settlements, y, x);
 const isCoast = !!coastal[y][x];
 if (ic === 1 || ic === 2) {
 const deathRate = 0.15 * aggression + 0.12 * winter;
 let survival = Math.max(0.2, 1.0 - deathRate);
 if (food >= 2) survival += 0.15;
 else if (food < 1) survival -= 0.10;
 const portProb = (ic === 2 || isCoast) ? 0.15 + 0.10 * trade : 0.03;
 const ruinProb = Math.max(0.05, 1.0 - survival - portProb);
 dist[1] = ic === 1 ? survival * (1.0 - portProb) : survival * 0.3;