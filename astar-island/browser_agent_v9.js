/**
 * Astar Island — Browser Agent v9
 *
 * Self-contained IIFE targeting 90+ score. Key improvements over v8:
 *   1. Full NorseSim MC simulator (faithful port of simulator.py)
 *   2. ABC parameter inference (20 candidates, select top 3)
 *   3. 3-layer prediction blending (KT + MC_sim + GT_lookup)
 *   4. Focused query strategy with importance-weighted viewports
 *   5. Auto-poller with calibration from completed rounds
 *
 * Usage: paste into browser console at app.ainm.no
 */
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
  const MC_RUNS_ABC = 10;
  const MC_RUNS_FINAL = 100;
  const ABC_CANDIDATES = 20;
  const ABC_TOP_K = 3;

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

  // Parameter ranges for ABC sampling
  const PARAM_RANGES = {
    winter_severity: [0.05, 0.95],
    faction_aggression: [0.05, 0.95],
    trade_activity: [0.05, 0.95],
    forest_growth_rate: [0.02, 0.50],
    expansion_rate: [0.05, 0.50],
    raid_range: [3.0, 8.0],
    food_per_forest: [0.10, 0.60],
    port_development_threshold: [0.20, 0.80],
    ruin_reclaim_rate: [0.05, 0.50],
  };

  const CALIBRATED_PRIORS = {
    0: [0.82, 0.13, 0.012, 0.010, 0.028, 0.005],
    1: [0.37, 0.41, 0.008, 0.031, 0.181, 0.005],
    2: [0.36, 0.12, 0.319, 0.021, 0.176, 0.005],
    3: [0.15, 0.12, 0.03, 0.35, 0.30, 0.05],
    4: [0.07, 0.16, 0.014, 0.012, 0.744, 0.005],
    5: [0.002, 0.002, 0.002, 0.002, 0.002, 0.990],
  };

  // Ground truth context priors from completed rounds
  // Key: {initClass}_{foodBucket}_{coastal}_{distBucket}_{nSett}
  // Will be populated from calibration or gt_lookup.json
  const GT_CTX_PRIORS = {"0_0_1_far":[0.997266,0.001641,0.000938,0.000039,0.000117,0],"0_1_1_far":[0.996621,0.001862,0.001241,0.000172,0.000103,0],"0_1_1_mid":[0.940438,0.023013,0.02745,0.002988,0.006112,0],"0_0_1_mid":[0.947645,0.019518,0.024174,0.002479,0.006185,0],"0_2_1_mid":[0.913814,0.032468,0.040513,0.003846,0.009359,0],"0_2_1_near":[0.802195,0.064573,0.106098,0.007805,0.019329,0],"0_1_1_near":[0.830683,0.056522,0.085714,0.008758,0.018323,0],"0_0_1_near":[0.8676,0.043567,0.066433,0.0059,0.0165,0],"0_2_1_far":[0.987353,0.007843,0.004412,0.000196,0.000196,0],"4_1_1_mid":[0.022805,0.062195,0.063659,0.006098,0.845244,0],"0_3_1_mid":[0.882935,0.045109,0.056957,0.004891,0.010109,0],"4_3_1_mid":[0.031667,0.071111,0.129444,0.005,0.762778,0],"4_2_1_near":[0.0648,0.105,0.1724,0.016,0.6418,0],"0_3_1_near":[0.694595,0.093784,0.176351,0.012297,0.022973,0],"4_1_1_near":[0.067742,0.12629,0.195,0.016452,0.594516,0],"4_3_1_near":[0.06,0.096667,0.164167,0.009167,0.67,0],"4_2_1_mid":[0.021129,0.064839,0.082581,0.004032,0.827419,0],"0_3_0_near":[0.711319,0.225434,0,0.015882,0.047365,0],"0_3_0_mid":[0.839451,0.126608,0,0.010285,0.023656,0],"0_2_0_mid":[0.835881,0.130559,0,0.01033,0.02323,0],"4_3_0_near":[0.100996,0.223996,0,0.017324,0.657684,0],"4_2_0_near":[0.100788,0.221362,0,0.016183,0.661667,0],"4_3_0_mid":[0.049462,0.139247,0,0.008656,0.802634,0],"4_2_0_mid":[0.050576,0.131212,0,0.009697,0.808515,0],"1_2_0_near":[0.366277,0.418936,0,0.030213,0.184574,0],"0_2_0_near":[0.719964,0.215418,0,0.015815,0.048802,0],"1_3_0_near":[0.353681,0.440417,0,0.034236,0.171667,0],"0_1_0_near":[0.720663,0.21623,0,0.015715,0.047392,0],"4_1_0_mid":[0.053733,0.128567,0,0.0107,0.807,0],"4_1_0_near":[0.097399,0.224753,0,0.017063,0.660785,0],"1_1_0_near":[0.381132,0.406887,0,0.029245,0.182736,0],"0_0_0_near":[0.719589,0.214929,0,0.015911,0.049571,0],"4_0_0_near":[0.1086,0.207733,0,0.0182,0.665467,0],"5_2_0_near":[0,0,0,0,0,1],"5_1_0_mid":[0,0,0,0,0,1],"5_0_0_mid":[0,0,0,0,0,1],"0_0_0_mid":[0.848254,0.119822,0,0.00929,0.022633,0],"5_1_0_near":[0,0,0,0,0,1],"5_2_0_mid":[0,0,0,0,0,1],"0_1_0_far":[0.984737,0.01307,0,0.000877,0.001316,0],"0_2_0_far":[0.986176,0.012882,0,0.000353,0.000588,0],"0_0_0_far":[0.990526,0.008421,0,0.000263,0.000789,0],"0_3_0_far":[0.984263,0.013947,0,0.000737,0.001053,0],"4_3_0_far":[0.002875,0.010875,0,0.001,0.98525,0],"4_1_0_far":[0.001833,0.0135,0,0.000333,0.984333,0],"4_2_0_far":[0.002,0.012143,0,0.000357,0.985500,0],"4_0_0_mid":[0.051058,0.129904,0,0.010673,0.808365,0],"1_0_0_near":[0.4152,0.3564,0,0.029,0.1994,0]};

  // ── Logging ────────────────────────────────────────────────────────────────

  window.__pollLog = window.__pollLog || [];
  function log(msg) {
    const ts = new Date().toISOString().slice(11, 19);
    const line = `[${ts}] ${msg}`;
    console.log(line);
    window.__pollLog.push(line);
    if (window.__pollLog.length > 500) window.__pollLog.shift();
  }

  // ── Utility functions ──────────────────────────────────────────────────────

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

  function floorNorm(p, floors) {
    if (floors) {
      for (let i = 0; i < p.length; i++) p[i] = Math.max(p[i], floors[i]);
    } else {
      for (let i = 0; i < p.length; i++) p[i] = Math.max(p[i], PROB_FLOOR);
    }
    return normalize(p);
  }

  function getCellFloors(initCls, isOcean, settDistVal) {
    const floors = new Float64Array(NUM_CLASSES).fill(PROB_FLOOR);
    if (initCls === 5) {
      floors.fill(STATIC_FLOOR);
    } else if (isOcean) {
      floors.fill(STATIC_FLOOR);
    } else {
      floors[5] = STATIC_FLOOR;
      if (initCls === 4) {
        floors[1] = REMOTE_FLOOR;
        floors[2] = REMOTE_FLOOR;
        floors[3] = REMOTE_FLOOR;
      }
      if ((initCls === 0 || initCls === 4) && settDistVal > 8) {
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

  // ── Spatial helpers ────────────────────────────────────────────────────────

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

  // ── Seeded PRNG (xoshiro128**) ────────────────────────────────────────────

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

  // ══════════════════════════════════════════════════════════════════════════
  // NorseSim — Faithful port of simulator.py
  // All 5 phases: Growth, Conflict, Trade, Winter, Environment
  // ══════════════════════════════════════════════════════════════════════════

  class NorseSim {
    constructor(initialGrid, initialSettlements, params, H, W) {
      this.H = H;
      this.W = W;
      this.initialGrid = initialGrid;
      this.initialSettlements = initialSettlements;
      this.params = { ...DEFAULT_PARAMS, ...params };

      // Precompute expansion offsets (manhattan 2-4)
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
          grid[s.y][s.x] = 3; // RUIN
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

    // ── Phase: Growth ──────────────────────────────────────────────────────

    _phaseGrowth(grid, settlements, rng, coastal) {
      const p = this.params;

      for (const s of settlements) {
        if (!s.alive) continue;

        // Food production from adjacent forests
        const forestFood = this._countForestFood(grid, s.y, s.x);
        const baseFood = 0.3 + forestFood;
        s.food += baseFood * (0.8 + 0.4 * rng());

        // Population growth based on food
        if (s.food > 1.0) {
          const growth = Math.min(s.food * 0.15, 0.5) * (0.7 + 0.6 * rng());
          s.population += growth;
          s.food -= growth * 0.5;
        }

        // Defense and tech slowly grow
        s.defense = Math.min(s.defense + 0.02 * rng(), 2.0);
        s.tech_level = Math.min(s.tech_level + 0.01 * rng(), 3.0);

        // Port development: coastal + enough wealth
        if (!s.has_port && coastal[s.y][s.x]) {
          const threshold = p.port_development_threshold;
          if (s.wealth > threshold && rng() < 0.15 * (1 + s.tech_level)) {
            s.has_port = true;
            grid[s.y][s.x] = 2;
          }
        }

        // Longship development
        if (!s.has_longship && s.has_port) {
          if (s.wealth > 0.5 && rng() < 0.1 * (1 + s.tech_level * 0.3))
            s.has_longship = true;
        }
      }

      // Expansion phase
      const expandRate = p.expansion_rate * 0.3;
      const expandable = settlements.filter((s) => s.alive && s.population > 2.0 && s.food > 1.5);
      if (expandable.length === 0) return;

      // Build occupancy (manhattan <= 1 from any alive settlement)
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
        // Update occupancy
        for (let ddy = -1; ddy <= 1; ddy++)
          for (let ddx = -1; ddx <= 1; ddx++) {
            const oy = ny + ddy, ox = nx + ddx;
            if (oy >= 0 && oy < this.H && ox >= 0 && ox < this.W) occ[oy][ox] = 1;
          }
        grid[ny][nx] = 1;
      }
    }

    // ── Phase: Conflict ────────────────────────────────────────────────────

    _phaseConflict(grid, settlements, rng) {
      const p = this.params;
      const alive = settlements.filter((s) => s.alive);
      if (alive.length < 2) return;

      for (const attacker of alive) {
        if (!attacker.alive) continue;

        // Raid probability: desperation + aggression
        const desperation = Math.max(0, 1.0 - attacker.food) * 0.4;
        const raidProb = p.faction_aggression * 0.3 + desperation;
        if (rng() > raidProb) continue;

        // Find targets within range
        const raidRange = p.raid_range * (attacker.has_longship ? 2.5 : 1.0);
        const targets = alive.filter((t) =>
          t.alive && t !== attacker && t.owner_id !== attacker.owner_id &&
          Math.abs(t.y - attacker.y) + Math.abs(t.x - attacker.x) <= raidRange
        );
        if (targets.length === 0) continue;

        const target = targets[Math.floor(rng() * targets.length)];

        // Combat resolution
        const atkStr = this._strength(attacker) * (0.6 + 0.8 * rng());
        const defStr = this._strength(target) * (0.6 + 0.8 * rng());

        if (atkStr > defStr) {
          // Attacker wins: loot + damage
          const lootFood = target.food * 0.3;
          const lootWealth = target.wealth * 0.3;
          attacker.food += lootFood;
          attacker.wealth += lootWealth;
          target.food -= lootFood;
          target.wealth -= lootWealth;
          target.population *= 0.8;
          target.defense *= 0.7;
          // Conquest chance
          if (rng() < 0.2 * p.faction_aggression)
            target.owner_id = attacker.owner_id;
        } else {
          // Defender wins
          attacker.population *= 0.9;
          attacker.defense *= 0.85;
        }
      }
    }

    // ── Phase: Trade ───────────────────────────────────────────────────────

    _phaseTrade(grid, settlements, rng) {
      const p = this.params;
      if (p.trade_activity < 0.05) return;
      const ports = settlements.filter((s) => s.alive && s.has_port);
      if (ports.length < 2) return;

      const tradeRange = 12.0;

      for (let i = 0; i < ports.length; i++) {
        for (let j = i + 1; j < ports.length; j++) {
          const a = ports[i], b = ports[j];
          if (a.owner_id === b.owner_id) continue; // Same faction
          const dist = Math.abs(a.y - b.y) + Math.abs(a.x - b.x);
          if (dist > tradeRange) continue;
          if (rng() > p.trade_activity * 0.6) continue;

          // Trade: both benefit
          const tv = 0.1 * p.trade_activity * (0.5 + rng());
          a.wealth += tv; b.wealth += tv;
          a.food += tv * 0.5; b.food += tv * 0.5;

          // Tech diffusion
          if (a.tech_level > b.tech_level)
            b.tech_level += (a.tech_level - b.tech_level) * 0.05;
          else
            a.tech_level += (b.tech_level - a.tech_level) * 0.05;
        }
      }
    }

    // ── Phase: Winter ──────────────────────────────────────────────────────

    _phaseWinter(grid, settlements, rng) {
      const p = this.params;

      for (const s of settlements) {
        if (!s.alive) continue;

        // Food loss from winter
        const foodLoss = p.winter_severity * (0.5 + 0.8 * rng());
        s.food -= foodLoss;

        // Population attrition when starving
        if (s.food < 0) {
          const popLoss = Math.min(Math.abs(s.food) * 0.3, s.population * 0.4);
          s.population -= popLoss;
          s.food = 0;
        }

        // Collapse check
        if (s.population < 0.2 || (s.food < 0.1 && rng() < 0.3 * p.winter_severity)) {
          s.alive = false;
          grid[s.y][s.x] = 3; // RUIN

          // Disperse population to nearby friendly settlements
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

    // ── Phase: Environment ─────────────────────────────────────────────────

    _phaseEnvironment(grid, settlements, rng, coastal) {
      const p = this.params;
      const aliveSettlements = settlements.filter((s) => s.alive);

      for (let y = 0; y < this.H; y++) {
        for (let x = 0; x < this.W; x++) {
          if (grid[y][x] === 3) {
            // Ruin: try reclaim by nearby thriving settlement
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
              // Forest reclaims ruin
              const adjForest = countAdj(grid, this.H, this.W, y, x, 4);
              const prob = adjForest > 0
                ? p.forest_growth_rate * 0.15 * Math.min(adjForest, 3)
                : p.forest_growth_rate * 0.02;
              if (rng() < prob) {
                grid[y][x] = 4; // FOREST
              } else if (rng() < 0.03) {
                grid[y][x] = 11; // PLAINS
              }
            }
          } else if (grid[y][x] === 11 || grid[y][x] === 0) {
            // Empty/Plains: forest spreading (very rare)
            const adjForest = countAdj(grid, this.H, this.W, y, x, 4);
            if (adjForest > 0) {
              const prob = p.forest_growth_rate * 0.02 * Math.min(adjForest, 3);
              if (rng() < prob) grid[y][x] = 4;
            }
          }
        }
      }
    }

    // ── Main simulation loop ───────────────────────────────────────────────

    /** Run one 50-year simulation. Returns class grid (H x W) with values 0-5. */
    run(seed) {
      const rng = xoshiro128(seed);
      const grid = this._cloneGrid();
      let settlements = this._initSettlements();
      this._rebuildGrid(grid, settlements);

      for (let year = 0; year < 50; year++) {
        // Compute coastal mask once per year
        const coastal = coastalMask(grid, this.H, this.W);

        this._phaseGrowth(grid, settlements, rng, coastal);
        this._phaseConflict(grid, settlements, rng);
        this._phaseTrade(grid, settlements, rng);
        this._phaseWinter(grid, settlements, rng);
        this._phaseEnvironment(grid, settlements, rng, coastal);
        this._rebuildGrid(grid, settlements);

        // Compact every 10 years
        if (year % 10 === 9) {
          settlements = settlements.filter((s) => s.alive);
        }
      }

      // Convert to class indices
      const cls = Array.from({ length: this.H }, () => new Int32Array(this.W));
      for (let y = 0; y < this.H; y++)
        for (let x = 0; x < this.W; x++)
          cls[y][x] = classifyCode(grid[y][x]);
      return cls;
    }

    /** Run Monte Carlo and return (H, W, 6) probability tensor */
    runMonteCarlo(nRuns) {
      const counts = make3D(this.H, this.W, NUM_CLASSES, 0);
      for (let i = 0; i < nRuns; i++) {
        const cls = this.run(i * 7919 + 42);
        for (let y = 0; y < this.H; y++)
          for (let x = 0; x < this.W; x++)
            counts[y][x][cls[y][x]] += 1;
      }

      // Jeffreys smoothing
      const alpha = 0.5;
      const denom = nRuns + NUM_CLASSES * alpha;
      const probs = make3D(this.H, this.W, NUM_CLASSES, 0);
      for (let y = 0; y < this.H; y++)
        for (let x = 0; x < this.W; x++) {
          for (let c = 0; c < NUM_CLASSES; c++)
            probs[y][x][c] = (counts[y][x][c] + alpha) / denom;
          normalize(probs[y][x]);
        }
      return probs;
    }
  }

  // ══════════════════════════════════════════════════════════════════════════
  // ABC Parameter Inference
  // ══════════════════════════════════════════════════════════════════════════

  /** Generate Latin hypercube-like random parameter sets */
  function generateCandidates(n, baseSeed) {
    const rng = xoshiro128(baseSeed);
    const keys = Object.keys(PARAM_RANGES);
    const candidates = [];

    for (let i = 0; i < n; i++) {
      const params = {};
      for (const key of keys) {
        const [lo, hi] = PARAM_RANGES[key];
        // Stratified: divide [0,1] into n strata, pick randomly within stratum
        const stratum = (i + rng()) / n;
        params[key] = lo + stratum * (hi - lo);
      }
      candidates.push(params);
    }
    return candidates;
  }

  /** Quick param inference from transition statistics (used as seed for ABC) */
  function inferParamsQuick(initialGrid, observedCounts, H, W) {
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
    const ruinToSett = rowSums[3] > 5 ? tp[3][1] : 0.1;
    const portToRuin = rowSums[2] > 5 ? tp[2][3] : 0.2;
    const settSurv = rowSums[1] > 10 ? tp[1][1] : 0.5;

    return {
      winter_severity: clamp(settToRuin * 1.5 + portToRuin * 0.5, 0.05, 0.95),
      faction_aggression: clamp(settToRuin * 2.0, 0.05, 0.95),
      trade_activity: clamp((settToPort + portSurv) * 1.2, 0.05, 0.95),
      forest_growth_rate: clamp(emptyToForest * 5.0, 0.02, 0.5),
      expansion_rate: clamp(emptyToSett * 5.0, 0.05, 0.5),
      raid_range: clamp(3.0 + settToRuin * 2.0 * 5.0, 3.0, 8.0),
      food_per_forest: clamp(settSurv * 0.5, 0.1, 0.6),
      port_development_threshold: clamp(1.0 - settToPort * 3.0, 0.2, 0.8),
      ruin_reclaim_rate: clamp(ruinToSett * 2.0, 0.05, 0.5),
    };
  }

  /** Compute KL divergence between observed counts and MC probs for observed cells only */
  function computeObservedKL(mcProbs, observedCounts, H, W) {
    let totalKL = 0;
    let nCells = 0;

    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        const nObs = observedCounts[y][x].reduce((a, b) => a + b, 0);
        if (nObs === 0) continue;

        nCells++;
        // Empirical distribution from observations
        const alpha = 0.5;
        const denom = nObs + NUM_CLASSES * alpha;
        for (let c = 0; c < NUM_CLASSES; c++) {
          const p = (observedCounts[y][x][c] + alpha) / denom;
          const q = Math.max(mcProbs[y][x][c], 1e-8);
          if (p > 1e-8) totalKL += p * Math.log(p / q);
        }
      }
    }
    return nCells > 0 ? totalKL / nCells : Infinity;
  }

  /** ABC parameter inference: generate candidates, run MC, pick best */
  function abcInferParams(initialGrid, initialSettlements, observedCounts, H, W) {
    const t0 = performance.now();

    // Start with quick inference as a baseline
    const quickParams = inferParamsQuick(initialGrid, observedCounts, H, W);

    // Generate candidates: half around quick inference, half random
    const candidates = [];

    // Add quick-inferred params as first candidate
    candidates.push({ ...quickParams });

    // Generate perturbations around quick params
    const pertRng = xoshiro128(12345);
    const keys = Object.keys(PARAM_RANGES);
    for (let i = 0; i < Math.floor(ABC_CANDIDATES / 2); i++) {
      const params = {};
      for (const key of keys) {
        const [lo, hi] = PARAM_RANGES[key];
        const base = quickParams[key] ?? (lo + hi) / 2;
        const spread = (hi - lo) * 0.3;
        params[key] = clamp(base + (pertRng() - 0.5) * 2 * spread, lo, hi);
      }
      candidates.push(params);
    }

    // Random candidates (Latin hypercube)
    const randomCands = generateCandidates(
      ABC_CANDIDATES - candidates.length, 54321
    );
    candidates.push(...randomCands);

    // Score each candidate: run MC sim, compute KL on observed cells
    const scored = [];
    for (const params of candidates) {
      const sim = new NorseSim(initialGrid, initialSettlements, params, H, W);
      const mcProbs = sim.runMonteCarlo(MC_RUNS_ABC);
      const kl = computeObservedKL(mcProbs, observedCounts, H, W);
      scored.push({ params, kl });
    }

    // Sort by KL (lower is better) and select top K
    scored.sort((a, b) => a.kl - b.kl);
    const topK = scored.slice(0, ABC_TOP_K);

    // Average the top K parameter sets
    const avgParams = {};
    for (const key of keys) {
      let sum = 0;
      for (const s of topK) sum += s.params[key];
      avgParams[key] = sum / topK.length;
    }

    const dt = performance.now() - t0;
    log(`  ABC inference: ${dt.toFixed(0)}ms, top KLs: ${topK.map(s => s.kl.toFixed(4)).join(", ")}`);

    return avgParams;
  }

  // ══════════════════════════════════════════════════════════════════════════
  // GT Lookup + Context Pooling
  // ══════════════════════════════════════════════════════════════════════════

  function getContextKey(ic, food, coastal, sd) {
    const foodBucket = Math.min(food, 3);
    const distBucket = sd <= 3 ? "near" : sd <= 7 ? "mid" : "far";
    const coastInt = coastal ? 1 : 0;
    return `${ic}_${foodBucket}_${coastInt}_${distBucket}`;
  }

  function getGTLookup(ic, food, coastal, sd) {
    const key = getContextKey(ic, food, coastal, sd);
    const gtPrior = GT_CTX_PRIORS[key];
    if (gtPrior) {
      const p = new Float64Array(NUM_CLASSES);
      for (let i = 0; i < NUM_CLASSES; i++) p[i] = Math.max(gtPrior[i], 0.002);
      normalize(p);
      return p;
    }
    // Fallback: calibrated prior
    const src = CALIBRATED_PRIORS[ic] || CALIBRATED_PRIORS[0];
    const p = new Float64Array(NUM_CLASSES);
    for (let i = 0; i < NUM_CLASSES; i++) p[i] = Math.max(src[i], PROB_FLOOR);
    normalize(p);
    return p;
  }

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
          const distBucket = sd <= 3 ? "near" : sd <= 7 ? "mid" : "far";
          const key = `${ic}_${food}_${coast}_${distBucket}`;
          if (!contextMap[key]) contextMap[key] = new Float64Array(NUM_CLASSES);
          for (let c = 0; c < NUM_CLASSES; c++)
            contextMap[key][c] += counts[y][x][c];
        }
      }
    }
    return contextMap;
  }

  function contextPoolCell(contextMap, grid, H, W, settlements, y, x) {
    const raw = grid[y][x];
    const ic = classifyCode(raw);
    const food = Math.min(foodPotential(grid, H, W, y, x), 3);
    const coast = isCoastal(grid, H, W, y, x) ? 1 : 0;
    const sd = settDist(grid, H, W, settlements, y, x);
    const distBucket = sd <= 3 ? "near" : sd <= 7 ? "mid" : "far";
    const key = `${ic}_${food}_${coast}_${distBucket}`;

    const pooled = contextMap[key];
    if (pooled) {
      const total = pooled.reduce((a, b) => a + b, 0);
      if (total > 2) {
        const p = new Float64Array(NUM_CLASSES);
        const gtPrior = GT_CTX_PRIORS[key];
        const obsWeight = total / (total + 8);
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

    const gtPrior = GT_CTX_PRIORS[key];
    if (gtPrior) {
      const p = new Float64Array(NUM_CLASSES);
      for (let c = 0; c < NUM_CLASSES; c++) p[c] = Math.max(gtPrior[c], 0.001);
      normalize(p);
      return p;
    }
    return null;
  }

  // ══════════════════════════════════════════════════════════════════════════
  // Belief Propagation
  // ══════════════════════════════════════════════════════════════════════════

  function beliefPropagation(pred, observedMask, initGrid, H, W) {
    const compat = [];
    for (let i = 0; i < NUM_CLASSES; i++) {
      compat[i] = new Float64Array(NUM_CLASSES);
      for (let j = 0; j < NUM_CLASSES; j++)
        compat[i][j] = (i === j) ? 0.75 : (0.25 / NUM_CLASSES);
    }
    for (const i of [1, 2, 3])
      for (const j of [1, 2, 3])
        if (i !== j) compat[i][j] = 0.12;
    for (const j of [0, 1, 2, 3]) {
      compat[4][j] = 0.02;
      compat[j][4] = 0.02;
    }
    for (let j = 0; j < 5; j++) {
      compat[5][j] = 0.01;
      compat[j][5] = 0.01;
    }
    compat[5][5] = 0.95;

    const anchorMask = Array.from({ length: H }, () => new Uint8Array(W));
    for (let y = 0; y < H; y++)
      for (let x = 0; x < W; x++)
        anchorMask[y][x] = (observedMask[y][x] || initGrid[y][x] === 5 || initGrid[y][x] === 10) ? 1 : 0;

    const damping = 0.15;
    const dirs = [[-1,0],[1,0],[0,-1],[0,1]];

    for (let iter = 0; iter < 3; iter++) {
      const newPred = make3D(H, W, NUM_CLASSES, 0);
      for (let y = 0; y < H; y++)
        for (let x = 0; x < W; x++)
          for (let c = 0; c < NUM_CLASSES; c++)
            newPred[y][x][c] = pred[y][x][c];

      for (const [dy, dx] of dirs) {
        for (let y = 0; y < H; y++) {
          for (let x = 0; x < W; x++) {
            if (anchorMask[y][x]) continue;
            const ny = y - dy, nx = x - dx;
            if (ny < 0 || ny >= H || nx < 0 || nx >= W) continue;
            for (let c = 0; c < NUM_CLASSES; c++) {
              let msg = 0;
              for (let cp = 0; cp < NUM_CLASSES; cp++)
                msg += compat[c][cp] * pred[ny][nx][cp];
              newPred[y][x][c] = (1 - damping) * newPred[y][x][c] + damping * msg;
            }
          }
        }
      }

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

  // ══════════════════════════════════════════════════════════════════════════
  // Query Optimizer — Focused on settlement viewports
  // ══════════════════════════════════════════════════════════════════════════

  class QueryOptimizer {
    constructor(W, H, seedsCount, budget, initialStates) {
      this.W = W;
      this.H = H;
      this.seedsCount = seedsCount;
      this.budget = budget;
      this.initialStates = initialStates;
      this.settlementCoords = [];
      this.grids = [];
      this.settDistMaps = [];

      for (let si = 0; si < seedsCount; si++) {
        const state = initialStates[si];
        const grid = state.grid;
        this.grids.push(grid);
        const coords = (state.settlements || [])
          .filter((s) => s.x >= 0 && s.x < W && s.y >= 0 && s.y < H)
          .map((s) => [s.x, s.y]);
        this.settlementCoords.push(coords);

        const distMap = Array.from({ length: H }, () => new Float64Array(W).fill(999));
        for (const [sx, sy] of coords)
          for (let y = 0; y < H; y++)
            for (let x = 0; x < W; x++) {
              const d = Math.abs(x - sx) + Math.abs(y - sy);
              if (d < distMap[y][x]) distMap[y][x] = d;
            }
        this.settDistMaps.push(distMap);
      }

      // Importance maps: 10=settlement, 5=coastal near sett, 3=adjacent, 0=static
      this.importanceMaps = [];
      for (let si = 0; si < seedsCount; si++) {
        const imp = Array.from({ length: H }, () => new Float64Array(W));
        const grid = this.grids[si];
        const dist = this.settDistMaps[si];
        for (let y = 0; y < H; y++)
          for (let x = 0; x < W; x++) {
            const v = grid[y][x];
            if (v === 10 || v === 5) { imp[y][x] = 0; continue; }
            if (v === 1 || v === 2 || v === 3) { imp[y][x] = 10; continue; }
            if (dist[y][x] <= 3 && isCoastal(grid, H, W, y, x)) { imp[y][x] = 5; continue; }
            if (dist[y][x] <= 3) { imp[y][x] = 3; continue; }
            if (dist[y][x] <= 6) { imp[y][x] = 1; continue; }
            imp[y][x] = 0;
          }
        this.importanceMaps.push(imp);
      }

      // Find minimum covering viewports per seed
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
      const buffer = 3;
      const uncovered = new Set(coords.map((_, i) => i));
      const viewports = [];
      while (uncovered.size > 0) {
        let bestVP = null, bestCov = new Set(), bestScore = -1;
        for (const i of uncovered) {
          const [sx, sy] = coords[i];
          const vx = clamp(sx - Math.floor(VIEWPORT_MAX / 2), 0, Math.max(0, this.W - VIEWPORT_MAX));
          const vy = clamp(sy - Math.floor(VIEWPORT_MAX / 2), 0, Math.max(0, this.H - VIEWPORT_MAX));
          const vw = Math.min(VIEWPORT_MAX, this.W - vx);
          const vh = Math.min(VIEWPORT_MAX, this.H - vy);
          const contained = new Set();
          for (const j of uncovered) {
            const [cx, cy] = coords[j];
            if (cx >= vx + buffer && cx < vx + vw - buffer &&
                cy >= vy + buffer && cy < vy + vh - buffer) contained.add(j);
          }
          if (contained.size === 0) {
            for (const j of uncovered) {
              const [cx, cy] = coords[j];
              if (cx >= vx && cx < vx + vw && cy >= vy && cy < vy + vh) contained.add(j);
            }
          }
          const dynScore = this._viewportImportance(seedIdx, vx, vy, vw, vh);
          const score = contained.size * 1000 + dynScore;
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
      const simCounts = {};
      for (let si = 0; si < this.seedsCount; si++)
        simCounts[si] = Array.from({ length: this.H }, () => new Float64Array(this.W));

      const recordQuery = (si, x, y, w, h) => {
        for (let gy = y; gy < y + h && gy < this.H; gy++)
          for (let gx = x; gx < x + w && gx < this.W; gx++)
            simCounts[si][gy][gx] += 1;
      };

      // Phase 1: One query per viewport per seed (~5-10 queries)
      for (let si = 0; si < this.seedsCount; si++) {
        for (const vp of this.seedViewports[si]) {
          if (plan.length >= this.budget) break;
          plan.push([si, ...vp]);
          recordQuery(si, vp[0], vp[1], vp[2], vp[3]);
        }
      }

      // Build candidate set: settlement viewports only
      const candidates = [];
      const seen = new Set();
      for (let si = 0; si < this.seedsCount; si++) {
        for (const vp of this.seedViewports[si]) {
          const key = `${si}_${vp.join("_")}`;
          if (!seen.has(key)) {
            seen.add(key);
            candidates.push({ si, vp });
          }
        }
      }

      // Phase 2: Remaining queries — highest marginal info gain
      while (plan.length < this.budget) {
        let bestGain = -1, bestQuery = null;
        for (const c of candidates) {
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

  // ══════════════════════════════════════════════════════════════════════════
  // 3-Layer Prediction Blending
  // ══════════════════════════════════════════════════════════════════════════

  function layeredPredict(
    grid, settlements, counts, contextMap, mcProbs, H, W
  ) {
    const pred = make3D(H, W, NUM_CLASSES, 0);
    const coastal = coastalMask(grid, H, W);
    const observedMask = Array.from({ length: H }, () => new Uint8Array(W));

    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        const raw = grid[y][x];

        // Hard constraints
        if (raw === 5) { pred[y][x] = mountainPrior(); continue; }
        if (raw === 10) { pred[y][x] = oceanPrior(); continue; }

        const ic = classifyCode(raw);
        const food = foodPotential(grid, H, W, y, x);
        const sd = settDist(grid, H, W, settlements, y, x);
        const isCoast = !!coastal[y][x];
        const nObs = counts[y][x].reduce((a, b) => a + b, 0);
        const cellFloors = getCellFloors(ic, false, sd);

        // KT estimator (Dirichlet-smoothed observation counts)
        const ktAlpha = 0.5;
        const kt = new Float64Array(NUM_CLASSES);
        if (nObs > 0) {
          const denom = nObs + NUM_CLASSES * ktAlpha;
          for (let c = 0; c < NUM_CLASSES; c++)
            kt[c] = (counts[y][x][c] + ktAlpha) / denom;
        }

        // MC simulator prediction
        const mc = mcProbs ? mcProbs[y][x] : null;

        // GT lookup
        const gt = getGTLookup(ic, food, isCoast, sd);

        if (nObs >= 3) {
          // Layer 1: Well-observed — 70% KT + 30% MC_sim
          observedMask[y][x] = 1;
          if (mc) {
            for (let c = 0; c < NUM_CLASSES; c++)
              pred[y][x][c] = 0.70 * kt[c] + 0.30 * mc[c];
          } else {
            for (let c = 0; c < NUM_CLASSES; c++)
              pred[y][x][c] = kt[c];
          }

        } else if (nObs >= 1) {
          // Layer 2: Sparse observations — 35% KT + 35% MC_sim + 30% GT_lookup
          observedMask[y][x] = 1;
          if (mc) {
            for (let c = 0; c < NUM_CLASSES; c++)
              pred[y][x][c] = 0.35 * kt[c] + 0.35 * mc[c] + 0.30 * gt[c];
          } else {
            for (let c = 0; c < NUM_CLASSES; c++)
              pred[y][x][c] = 0.50 * kt[c] + 0.50 * gt[c];
          }

        } else {
          // Layer 3: Unobserved dynamic cell — 60% MC_sim + 40% GT_lookup
          // Also blend with context pooling if available
          const ctxPred = contextPoolCell(contextMap, grid, H, W, settlements, y, x);

          if (mc && ctxPred) {
            // Have both MC and context pool
            for (let c = 0; c < NUM_CLASSES; c++)
              pred[y][x][c] = 0.40 * mc[c] + 0.30 * gt[c] + 0.30 * ctxPred[c];
          } else if (mc) {
            for (let c = 0; c < NUM_CLASSES; c++)
              pred[y][x][c] = 0.60 * mc[c] + 0.40 * gt[c];
          } else if (ctxPred) {
            for (let c = 0; c < NUM_CLASSES; c++)
              pred[y][x][c] = 0.50 * ctxPred[c] + 0.50 * gt[c];
          } else {
            for (let c = 0; c < NUM_CLASSES; c++)
              pred[y][x][c] = gt[c];
          }
        }

        // Suppress port on non-coastal
        if (!isCoast) pred[y][x][2] = Math.min(pred[y][x][2], cellFloors[2] * 2);

        floorNorm(pred[y][x], cellFloors);
      }
    }

    // Belief propagation for spatial smoothing of unobserved cells
    const result = beliefPropagation(pred, observedMask, grid, H, W);

    // Final hard constraints + floors
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

  // ══════════════════════════════════════════════════════════════════════════
  // Main Pipeline
  // ══════════════════════════════════════════════════════════════════════════

  async function runPipeline(roundId) {
    log(`Pipeline v9 starting for round ${roundId}`);
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

    // ── Phase 1: Observe ─────────────────────────────────────────────────

    const allCounts = {};
    for (let si = 0; si < seedsCount; si++)
      allCounts[si] = make3D(H, W, NUM_CLASSES, 0);

    if (queriesLeft > 0) {
      const optimizer = new QueryOptimizer(W, H, seedsCount, queriesLeft, initialStates);
      const plan = optimizer.planQueries();
      log(`Query plan: ${plan.length} queries across ${seedsCount} seeds`);

      let queriesExecuted = 0;
      for (let qi = 0; qi < plan.length; qi++) {
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

    // ── Phase 2: Parameter Inference + MC Simulation ─────────────────────

    log("Building predictions (v9: MC sim + ABC + 3-layer blending)...");

    // Build cross-seed context pool
    const contextMap = buildContextPool(initialStates, allCounts, H, W);
    log(` Contextual pool: ${Object.keys(contextMap).length} context groups`);

    // Aggregate observation counts across seeds for param inference
    const globalCounts = make3D(H, W, NUM_CLASSES, 0);
    const refGrid = initialStates[0].grid;
    for (let si = 0; si < seedsCount; si++)
      for (let y = 0; y < H; y++)
        for (let x = 0; x < W; x++)
          for (let c = 0; c < NUM_CLASSES; c++)
            globalCounts[y][x][c] += allCounts[si][y][x][c];

    // ABC parameter inference using first seed's grid/settlements
    let inferredParams;
    try {
      const refSettlements = (initialStates[0].settlements || []).map((s, i) => ({
        x: s.x, y: s.y,
        has_port: s.has_port || false,
        alive: true,
        owner_id: i,
      }));
      inferredParams = abcInferParams(refGrid, refSettlements, globalCounts, H, W);
      log(`Inferred params: winter=${inferredParams.winter_severity.toFixed(2)}, ` +
        `aggression=${inferredParams.faction_aggression.toFixed(2)}, ` +
        `trade=${inferredParams.trade_activity.toFixed(2)}, ` +
        `expansion=${inferredParams.expansion_rate.toFixed(2)}`);
    } catch (e) {
      log(`ABC inference failed: ${e.message}, using quick inference`);
      inferredParams = inferParamsQuick(refGrid, globalCounts, H, W);
    }

    // Run MC simulation per seed with inferred params
    const mcProbsPerSeed = {};
    for (let si = 0; si < seedsCount; si++) {
      const t1 = performance.now();
      const grid = initialStates[si].grid;
      const settlements = (initialStates[si].settlements || []).map((s, i) => ({
        x: s.x, y: s.y,
        has_port: s.has_port || false,
        alive: true,
        owner_id: i,
      }));

      try {
        const sim = new NorseSim(grid, settlements, inferredParams, H, W);
        mcProbsPerSeed[si] = sim.runMonteCarlo(MC_RUNS_FINAL);
        const dt = performance.now() - t1;
        log(` Seed ${si}: MC sim (${MC_RUNS_FINAL} runs) in ${dt.toFixed(0)}ms`);
      } catch (e) {
        log(` Seed ${si} MC sim failed: ${e.message}`);
        mcProbsPerSeed[si] = null;
      }
    }

    // ── Phase 3: 3-Layer Prediction Blending ─────────────────────────────

    const predictions = {};
    for (let si = 0; si < seedsCount; si++) {
      const t1 = performance.now();
      const grid = initialStates[si].grid;
      const settlements = initialStates[si].settlements || [];
      const counts = allCounts[si];

      predictions[si] = layeredPredict(
        grid, settlements, counts, contextMap, mcProbsPerSeed[si], H, W
      );

      const dt = performance.now() - t1;
      let obsCount = 0;
      for (let y = 0; y < H; y++)
        for (let x = 0; x < W; x++)
          if (counts[y][x].reduce((a, b) => a + b, 0) > 0) obsCount++;
      const obsPct = (100 * obsCount / (H * W)).toFixed(0);
      log(` Seed ${si}: ${obsPct}% observed, blend in ${dt.toFixed(0)}ms`);
    }

    // ── Phase 4: Submit ──────────────────────────────────────────────────

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
    log(`Pipeline v9 complete in ${totalTime}s`);
    return true;
  }

  // ══════════════════════════════════════════════════════════════════════════
  // Auto-calibration from completed rounds
  // ══════════════════════════════════════════════════════════════════════════

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
            let coast = 0;
            for (const [dy,dx] of [[-1,0],[1,0],[0,-1],[0,1]]) {
              const ny = y+dy, nx = x+dx;
              if (ny >= 0 && ny < H && nx >= 0 && nx < W && ig[ny][nx] === 10) coast = 1;
            }
            let sd = 999;
            for (let sy = Math.max(0,y-12); sy < Math.min(H,y+12); sy++)
              for (let sx = Math.max(0,x-12); sx < Math.min(W,x+12); sx++)
                if (ig[sy][sx] === 1 || ig[sy][sx] === 2) sd = Math.min(sd, Math.abs(y-sy)+Math.abs(x-sx));
            const distBucket = sd <= 3 ? "near" : sd <= 7 ? "mid" : "far";
            const key = `${ic}_${food}_${coast}_${distBucket}`;
            if (!ctxStats[key]) ctxStats[key] = { counts: new Float64Array(6), n: 0 };
            for (let c = 0; c < 6; c++) ctxStats[key].counts[c] += gt[y][x][c];
            ctxStats[key].n++;
          }
        }
      } catch (e) { log(`  Seed ${si} analysis failed: ${e.message}`); }
    }

    let updated = 0;
    for (const [key, stats] of Object.entries(ctxStats)) {
      if (stats.n < 3) continue;
      const avg = Array.from(stats.counts);
      const sum = avg.reduce((a,b) => a+b, 0);
      if (sum <= 0) continue;
      for (let i = 0; i < 6; i++) avg[i] /= sum;
      if (GT_CTX_PRIORS[key]) {
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

  // ══════════════════════════════════════════════════════════════════════════
  // Auto-poller
  // ══════════════════════════════════════════════════════════════════════════

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
          if (r.status === "completed" && r.round_score && !window.__calibratedRounds?.[r.id]) {
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
    log("v9 Auto-poller started (30s interval). window.__stopPoll() to stop.");
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
  window.__GT_CTX_PRIORS = GT_CTX_PRIORS;

  startPoll();
})();
