
(function () {
 "use strict";
 const API = "https://api.ainm.no/astar-island";
 const POLL_INTERVAL = 30000; 
 const API_DELAY = 65; 
 const NUM_CLASSES = 6;
 const PROB_FLOOR = 0.01;
 const TEMPERATURE = 1.1; 
 const MC_RUNS = 50; 
 const VIEWPORT_MAX = 15;
 const CALIBRATED_PRIORS = {
 0: [0.82, 0.13, 0.012, 0.010, 0.028, 0.01], 
 1: [0.37, 0.41, 0.008, 0.031, 0.181, 0.01], 
 2: [0.36, 0.12, 0.319, 0.021, 0.176, 0.01], 
 3: [0.17, 0.17, 0.17, 0.17, 0.17, 0.15], 
 4: [0.07, 0.16, 0.014, 0.012, 0.744, 0.01], 
 5: [0.005, 0.005, 0.005, 0.005, 0.005, 0.975], 
 };
 const TERRAIN_TO_CLASS = { 10: 0, 11: 0, 0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5 };
 const DEFAULT_PARAMS = {
 winter_severity: 0.4,
 faction_aggression: 0.3,
 trade_activity: 0.5,
 forest_growth_rate: 0.05,
 expansion_rate: 0.2,
 raid_range: 5.0,
 food_per_forest: 0.3,
 port_development_threshold: 0.5,
 ruin_reclaim_rate: 0.15,
 };
 window.__pollLog = window.__pollLog || [];
 function log(msg) {
 const ts = new Date().toISOString().slice(11, 19);
 const line = `[${ts}] ${msg}`;
 console.log(line);
 window.__pollLog.push(line);
 if (window.__pollLog.length > 500) window.__pollLog.shift();
 }
 function sleep(ms) {
 return new Promise((r) => setTimeout(r, ms));
 }
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
 function classifyCode(code) {
 return TERRAIN_TO_CLASS[code] ?? 0;
 }
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
 function clamp(v, lo, hi) {
 return Math.max(lo, Math.min(hi, v));
 }
 function normalize(p) {
 let s = 0;
 for (let i = 0; i < p.length; i++) s += p[i];
 if (s > 0) for (let i = 0; i < p.length; i++) p[i] /= s;
 return p;
 }
 function floorNorm(p, floor = PROB_FLOOR) {
 for (let i = 0; i < p.length; i++) p[i] = Math.max(p[i], floor);
 return normalize(p);
 }
 function calibratedPrior(initCls) {
 const src = CALIBRATED_PRIORS[initCls] || CALIBRATED_PRIORS[0];
 const p = new Float64Array(NUM_CLASSES);
 for (let i = 0; i < NUM_CLASSES; i++) p[i] = Math.max(src[i], PROB_FLOOR);
 normalize(p);
 return p;
 }
 function mountainPrior() {
 const p = new Float64Array(NUM_CLASSES).fill(PROB_FLOOR);
 p[5] = 1.0 - 5 * PROB_FLOOR;
 return p;
 }
 function oceanPrior() {
 const p = new Float64Array(NUM_CLASSES).fill(PROB_FLOOR);
 p[0] = 1.0 - 5 * PROB_FLOOR;
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
 if (ny >= 0 && ny < H && nx >= 0 && nx < W && grid[ny][nx] === 4)
 count++;
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
 function xoshiro128(seed) {
 let s0 = (seed ^ 0xDEADBEEF) >>> 0;
 let s1 = (seed * 1664525 + 1013904223) >>> 0;
 let s2 = (s1 * 1664525 + 1013904223) >>> 0;
 let s3 = (s2 * 1664525 + 1013904223) >>> 0;
 function rotl(x, k) { return ((x << k) | (x >>> (32 - k))) >>> 0; }
 return function random() {
 const result = (rotl((s1 * 5) >>> 0, 7) * 9) >>> 0;
 const t = (s1 << 9) >>> 0;
 s2 ^= s0; s3 ^= s1; s1 ^= s2; s0 ^= s3;
 s2 ^= t;
 s3 = rotl(s3, 11);
 return (result >>> 0) / 4294967296;
 };
 }
 class NorseSim {
 constructor(initialGrid, initialSettlements, params, H, W) {
 this.H = H;
 this.W = W;
 this.initialGrid = initialGrid;
 this.initialSettlements = initialSettlements;
 this.params = { ...DEFAULT_PARAMS, ...params };
 this.expansionOffsets = [];
 for (let dy = -4; dy <= 4; dy++)
 for (let dx = -4; dx <= 4; dx++) {
 const d = Math.abs(dy) + Math.abs(dx);
 if (d >= 2 && d <= 4) this.expansionOffsets.push([dy, dx]);
 }
 }
 _cloneGrid() {
 return this.initialGrid.map((row) => Int32Array.from(row));
 }
 _initSettlements() {
 return this.initialSettlements.map((s, i) => ({
 x: s.x ?? 0,
 y: s.y ?? 0,
 population: s.population ?? 1.0,
 food: s.food ?? 1.0,
 wealth: s.wealth ?? 0.0,
 defense: s.defense ?? 0.5,
 has_port: s.has_port ?? false,
 alive: s.alive ?? true,
 owner_id: s.owner_id ?? i,
 tech_level: s.tech_level ?? 0.0,
 has_longship: s.has_longship ?? false,
 }));
 }
 _strength(s) {
 return s.population * s.defense * (1.0 + 0.2 * s.tech_level);
 }
 _rebuildGrid(grid, settlements) {
 for (const s of settlements) {
 if (s.y < 0 || s.y >= this.H || s.x < 0 || s.x >= this.W) continue;
 if (s.alive) {
 grid[s.y][s.x] = s.has_port ? 2 : 1;
 } else if (grid[s.y][s.x] === 1 || grid[s.y][s.x] === 2) {
 grid[s.y][s.x] = 3;
 }
 }
 }
 _countForestFood(grid, y, x) {
 let food = 0;
 const fpf = this.params.food_per_forest;
 for (let dy = -1; dy <= 1; dy++)
 for (let dx = -1; dx <= 1; dx++) {
 if (dy === 0 && dx === 0) continue;
 const ny = y + dy, nx = x + dx;
 if (ny >= 0 && ny < this.H && nx >= 0 && nx < this.W && grid[ny][nx] === 4)
 food += fpf;
 }
 return food;
 }
 _phaseGrowth(grid, settlements, rng) {
 const p = this.params;
 const coastal = coastalMask(grid, this.H, this.W);
 for (const s of settlements) {
 if (!s.alive) continue;
 const forestFood = this._countForestFood(grid, s.y, s.x);
 const baseFood = 0.3 + forestFood;
 s.food += baseFood * (0.8 + 0.4 * rng());
 if (s.food > 1.0) {
 const growth = Math.min(s.food * 0.15, 0.5) * (0.7 + 0.6 * rng());
 s.population += growth;
 s.food -= growth * 0.5;
 }
 s.defense = Math.min(s.defense + 0.02 * rng(), 2.0);
 s.tech_level = Math.min(s.tech_level + 0.01 * rng(), 3.0);
 if (!s.has_port && coastal[s.y][s.x]) {
 if (s.wealth > p.port_development_threshold && rng() < 0.15 * (1 + s.tech_level)) {
 s.has_port = true;
 grid[s.y][s.x] = 2;
 }
 }
 if (!s.has_longship && s.has_port) {
 if (s.wealth > 0.5 && rng() < 0.1 * (1 + s.tech_level * 0.3))
 s.has_longship = true;
 }
 }
 const expandRate = p.expansion_rate * 0.3;
 const expandable = settlements.filter((s) => s.alive && s.population > 2.0 && s.food > 1.5);
 if (expandable.length === 0) return;
 const occ = Array.from({ length: this.H }, () => new Uint8Array(this.W));
 for (const s of settlements) {
 if (!s.alive) continue;
 for (let dy = -1; dy <= 1; dy++)
 for (let dx = -1; dx <= 1; dx++) {
 const ny = s.y + dy, nx = s.x + dx;
 if (ny >= 0 && ny < this.H && nx >= 0 && nx < this.W) occ[ny][nx] = 1;
 }
 }
 for (const s of expandable) {
 if (rng() >= expandRate) continue;
 const candidates = [];
 for (const [dy, dx] of this.expansionOffsets) {
 const ny = s.y + dy, nx = s.x + dx;
 if (ny >= 0 && ny < this.H && nx >= 0 && nx < this.W &&
 (grid[ny][nx] === 11 || grid[ny][nx] === 0 || grid[ny][nx] === 4) &&
 !occ[ny][nx]) {
 candidates.push([ny, nx]);
 }
 }
 if (candidates.length === 0) continue;
 const [ny, nx] = candidates[Math.floor(rng() * candidates.length)];
 const newS = {
 x: nx, y: ny,
 population: s.population * 0.3,
 food: s.food * 0.3,
 wealth: s.wealth * 0.2,
 defense: 0.3,
 has_port: false,
 alive: true,
 owner_id: s.owner_id,
 tech_level: s.tech_level * 0.5,
 has_longship: false,
 };
 s.population *= 0.7;
 s.food *= 0.7;
 s.wealth *= 0.8;
 settlements.push(newS);
 for (let ddy = -1; ddy <= 1; ddy++)
 for (let ddx = -1; ddx <= 1; ddx++) {
 const oy = ny + ddy, ox = nx + ddx;
 if (oy >= 0 && oy < this.H && ox >= 0 && ox < this.W) occ[oy][ox] = 1;
 }
 grid[ny][nx] = 1;
 }
 }
 _phaseConflict(grid, settlements, rng) {
 const p = this.params;
 const alive = settlements.filter((s) => s.alive);
 if (alive.length < 2) return;
 for (const attacker of alive) {
 if (!attacker.alive) continue;
 const desperation = Math.max(0, 1.0 - attacker.food) * 0.4;
 const raidProb = p.faction_aggression * 0.3 + desperation;
 if (rng() > raidProb) continue;
 const raidRange = p.raid_range * (attacker.has_longship ? 2.5 : 1.0);
 const targets = alive.filter((t) =>
 t.alive && t !== attacker && t.owner_id !== attacker.owner_id &&
 Math.abs(t.y - attacker.y) + Math.abs(t.x - attacker.x) <= raidRange
 );
 if (targets.length === 0) continue;
 const target = targets[Math.floor(rng() * targets.length)];
 const atkStr = this._strength(attacker) * (0.6 + 0.8 * rng());
 const defStr = this._strength(target) * (0.6 + 0.8 * rng());
 if (atkStr > defStr) {
 const lootFood = target.food * 0.3;
 const lootWealth = target.wealth * 0.3;
 attacker.food += lootFood;
 attacker.wealth += lootWealth;
 target.food -= lootFood;
 target.wealth -= lootWealth;
 target.population *= 0.8;
 target.defense *= 0.7;
 if (rng() < 0.2 * p.faction_aggression)
 target.owner_id = attacker.owner_id;
 } else {
 attacker.population *= 0.9;
 attacker.defense *= 0.85;
 }
 }
 }
 _phaseTrade(grid, settlements, rng) {
 const p = this.params;
 if (p.trade_activity < 0.05) return;
 const ports = settlements.filter((s) => s.alive && s.has_port);
 if (ports.length < 2) return;
 for (let i = 0; i < ports.length; i++) {
 for (let j = i + 1; j < ports.length; j++) {
 const a = ports[i], b = ports[j];
 if (a.owner_id === b.owner_id) continue;
 const dist = Math.abs(a.y - b.y) + Math.abs(a.x - b.x);
 if (dist > 12) continue;
 if (rng() > p.trade_activity * 0.6) continue;
 const tv = 0.1 * p.trade_activity * (0.5 + rng());
 a.wealth += tv; b.wealth += tv;
 a.food += tv * 0.5; b.food += tv * 0.5;
 if (a.tech_level > b.tech_level)
 b.tech_level += (a.tech_level - b.tech_level) * 0.05;
 else
 a.tech_level += (b.tech_level - a.tech_level) * 0.05;
 }
 }
 }
 _phaseWinter(grid, settlements, rng) {
 const p = this.params;
 for (const s of settlements) {
 if (!s.alive) continue;
 const foodLoss = p.winter_severity * (0.5 + 0.8 * rng());
 s.food -= foodLoss;
 if (s.food < 0) {
 const popLoss = Math.min(Math.abs(s.food) * 0.3, s.population * 0.4);
 s.population -= popLoss;
 s.food = 0;
 }
 if (s.population < 0.2 || (s.food < 0.1 && rng() < 0.3 * p.winter_severity)) {