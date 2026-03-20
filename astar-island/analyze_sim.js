#!/usr/bin/env node
/**
 * analyze_sim.js — Calibrate NorseSimulator parameters against cached ground truth.
 *
 * Reads cached GT data from astar-island/cache/ directory,
 * runs MC simulation with various params, computes KL divergence,
 * and grid-searches for best-fit parameters per round.
 *
 * Usage: node analyze_sim.js
 */

"use strict";

const fs = require("fs");
const path = require("path");

// ── Terrain codes ──────────────────────────────────────────────────────────
const OCEAN = 10;
const PLAINS = 11;
const EMPTY = 0;
const SETTLEMENT = 1;
const PORT = 2;
const RUIN = 3;
const FOREST = 4;
const MOUNTAIN = 5;

const NUM_CLASSES = 6;
const TERRAIN_TO_CLASS = { 10: 0, 11: 0, 0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5 };
const IMMUTABLE = new Set([OCEAN, MOUNTAIN]);

const CLASS_NAMES = ["Empty", "Settlement", "Port", "Ruin", "Forest", "Mountain"];

// ── Default parameters ─────────────────────────────────────────────────────
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

// ── Seedable RNG (xoshiro128**) ────────────────────────────────────────────
// We need a seedable PRNG to match numpy's behavior deterministically per run.

class SeedableRNG {
  constructor(seed) {
    // SplitMix64 to initialize state from a single seed
    let s = BigInt(seed) | 0n;
    const sm = () => {
      s = BigInt.asUintN(64, s + 0x9e3779b97f4a7c15n);
      let z = s;
      z = BigInt.asUintN(64, (z ^ (z >> 30n)) * 0xbf58476d1ce4e5b9n);
      z = BigInt.asUintN(64, (z ^ (z >> 27n)) * 0x94d049bb133111ebn);
      return z ^ (z >> 31n);
    };
    const a = sm();
    const b = sm();
    this.s = [
      Number(a & 0xffffffffn),
      Number((a >> 32n) & 0xffffffffn),
      Number(b & 0xffffffffn),
      Number((b >> 32n) & 0xffffffffn),
    ];
  }

  _rotl(x, k) {
    return ((x << k) | (x >>> (32 - k))) >>> 0;
  }

  _next() {
    const s = this.s;
    const result = (Math.imul(this._rotl(Math.imul(s[1], 5) >>> 0, 7), 9)) >>> 0;
    const t = (s[1] << 9) >>> 0;
    s[2] = (s[2] ^ s[0]) >>> 0;
    s[3] = (s[3] ^ s[1]) >>> 0;
    s[1] = (s[1] ^ s[2]) >>> 0;
    s[0] = (s[0] ^ s[3]) >>> 0;
    s[2] = (s[2] ^ t) >>> 0;
    s[3] = this._rotl(s[3], 11);
    return result;
  }

  /** Returns float in [0, 1) */
  random() {
    return (this._next() >>> 0) / 4294967296;
  }

  /** Returns integer in [0, max) */
  integers(max) {
    return Math.floor(this.random() * max);
  }
}

// ── 2D array helpers ───────────────────────────────────────────────────────

function makeGrid(H, W, fill) {
  const g = new Array(H);
  for (let y = 0; y < H; y++) {
    g[y] = new Array(W).fill(fill);
  }
  return g;
}

function copyGrid(grid) {
  return grid.map((row) => row.slice());
}

function gridShape(grid) {
  return [grid.length, grid[0].length];
}

// ── Helper functions (ported from Python) ──────────────────────────────────

function coastalMask(grid) {
  const [H, W] = gridShape(grid);
  const mask = makeGrid(H, W, false);
  const dirs = [[-1, 0], [1, 0], [0, -1], [0, 1]];
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      if (grid[y][x] === OCEAN || grid[y][x] === MOUNTAIN) continue;
      for (const [dy, dx] of dirs) {
        const ny = y + dy;
        const nx = x + dx;
        if (ny >= 0 && ny < H && nx >= 0 && nx < W && grid[ny][nx] === OCEAN) {
          mask[y][x] = true;
          break;
        }
      }
    }
  }
  return mask;
}

function countAdjacent(grid, value) {
  const [H, W] = gridShape(grid);
  const count = makeGrid(H, W, 0);
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      let c = 0;
      for (let dy = -1; dy <= 1; dy++) {
        for (let dx = -1; dx <= 1; dx++) {
          if (dy === 0 && dx === 0) continue;
          const ny = y + dy;
          const nx = x + dx;
          if (ny >= 0 && ny < H && nx >= 0 && nx < W && grid[ny][nx] === value) {
            c++;
          }
        }
      }
      count[y][x] = c;
    }
  }
  return count;
}

function manhattanDist(y1, x1, y2, x2) {
  return Math.abs(y1 - y2) + Math.abs(x1 - x2);
}

// ── Settlement class ───────────────────────────────────────────────────────

class SettlementObj {
  constructor(opts) {
    this.x = opts.x || 0;
    this.y = opts.y || 0;
    this.population = opts.population != null ? opts.population : 1.0;
    this.food = opts.food != null ? opts.food : 1.0;
    this.wealth = opts.wealth != null ? opts.wealth : 0.0;
    this.defense = opts.defense != null ? opts.defense : 0.5;
    this.has_port = opts.has_port || false;
    this.alive = opts.alive != null ? opts.alive : true;
    this.owner_id = opts.owner_id != null ? opts.owner_id : 0;
    this.tech_level = opts.tech_level != null ? opts.tech_level : 0.0;
    this.has_longship = opts.has_longship || false;
  }

  strength() {
    return this.population * this.defense * (1.0 + 0.2 * this.tech_level);
  }
}

// ── NorseSimulator (faithful port from Python) ─────────────────────────────

class NorseSimulator {
  constructor(initialGrid, initialSettlements, params) {
    this.initialGrid = initialGrid;
    const [H, W] = gridShape(initialGrid);
    this.H = H;
    this.W = W;
    this.initialSettlements = initialSettlements;
    this.params = { ...DEFAULT_PARAMS, ...(params || {}) };

    // Precompute expansion candidate offsets (manhattan dist 2-4)
    this._expansionOffsets = [];
    for (let dy = -4; dy <= 4; dy++) {
      for (let dx = -4; dx <= 4; dx++) {
        const d = Math.abs(dy) + Math.abs(dx);
        if (d >= 2 && d <= 4) {
          this._expansionOffsets.push([dy, dx]);
        }
      }
    }
  }

  _initSettlements() {
    return this.initialSettlements.map((s, i) =>
      new SettlementObj({
        x: s.x || 0,
        y: s.y || 0,
        population: s.population != null ? s.population : 1.0,
        food: s.food != null ? s.food : 1.0,
        wealth: s.wealth != null ? s.wealth : 0.0,
        defense: s.defense != null ? s.defense : 0.5,
        has_port: s.has_port || false,
        alive: s.alive != null ? s.alive : true,
        owner_id: s.owner_id != null ? s.owner_id : i,
        tech_level: s.tech_level != null ? s.tech_level : 0.0,
        has_longship: s.has_longship || false,
      })
    );
  }

  _rebuildGrid(grid, settlements) {
    for (const s of settlements) {
      if (s.y < 0 || s.y >= this.H || s.x < 0 || s.x >= this.W) continue;
      if (s.alive) {
        grid[s.y][s.x] = s.has_port ? PORT : SETTLEMENT;
      } else {
        if (grid[s.y][s.x] === SETTLEMENT || grid[s.y][s.x] === PORT) {
          grid[s.y][s.x] = RUIN;
        }
      }
    }
  }

  _countForestFood(grid, y, x) {
    let food = 0;
    const fpf = this.params.food_per_forest;
    for (let dy = -1; dy <= 1; dy++) {
      for (let dx = -1; dx <= 1; dx++) {
        if (dy === 0 && dx === 0) continue;
        const ny = y + dy;
        const nx = x + dx;
        if (ny >= 0 && ny < this.H && nx >= 0 && nx < this.W && grid[ny][nx] === FOREST) {
          food += fpf;
        }
      }
    }
    return food;
  }

  _buildOccupancy(settlements) {
    const occ = makeGrid(this.H, this.W, false);
    for (const s of settlements) {
      if (!s.alive) continue;
      for (let dy = -1; dy <= 1; dy++) {
        for (let dx = -1; dx <= 1; dx++) {
          const ny = s.y + dy;
          const nx = s.x + dx;
          if (ny >= 0 && ny < this.H && nx >= 0 && nx < this.W) {
            occ[ny][nx] = true;
          }
        }
      }
    }
    return occ;
  }

  // ── Phase: Growth ──────────────────────────────────────────────────────
  _phaseGrowth(grid, settlements, rng, coastal) {
    const p = this.params;

    for (const s of settlements) {
      if (!s.alive) continue;

      // Food production from adjacent forests
      const forestFood = this._countForestFood(grid, s.y, s.x);
      const baseFood = 0.3 + forestFood;
      s.food += baseFood * (0.8 + 0.4 * rng.random());

      // Population growth based on food
      if (s.food > 1.0) {
        const growth = Math.min(s.food * 0.15, 0.5) * (0.7 + 0.6 * rng.random());
        s.population += growth;
        s.food -= growth * 0.5;
      }

      // Defense and tech slowly grow
      s.defense = Math.min(s.defense + 0.02 * rng.random(), 2.0);
      s.tech_level = Math.min(s.tech_level + 0.01 * rng.random(), 3.0);

      // Port development: coastal + enough wealth
      if (!s.has_port && coastal[s.y][s.x]) {
        const threshold = p.port_development_threshold;
        if (s.wealth > threshold && rng.random() < 0.15 * (1 + s.tech_level)) {
          s.has_port = true;
          grid[s.y][s.x] = PORT;
        }
      }

      // Longship development
      if (!s.has_longship && s.has_port) {
        if (s.wealth > 0.5 && rng.random() < 0.1 * (1 + s.tech_level * 0.3)) {
          s.has_longship = true;
        }
      }
    }

    // Expansion phase
    const expandRate = p.expansion_rate * 0.3;
    const expandable = settlements.filter(
      (s) => s.alive && s.population > 2.0 && s.food > 1.5
    );
    if (expandable.length > 0) {
      const occ = this._buildOccupancy(settlements);
      for (const s of expandable) {
        if (rng.random() >= expandRate) continue;

        // Find candidate land cells within range 2-4
        const candidates = [];
        for (const [dy, dx] of this._expansionOffsets) {
          const ny = s.y + dy;
          const nx = s.x + dx;
          if (
            ny >= 0 &&
            ny < this.H &&
            nx >= 0 &&
            nx < this.W &&
            (grid[ny][nx] === PLAINS || grid[ny][nx] === EMPTY || grid[ny][nx] === FOREST) &&
            !occ[ny][nx]
          ) {
            candidates.push([ny, nx]);
          }
        }

        if (candidates.length > 0) {
          const [ny, nx] = candidates[rng.integers(candidates.length)];
          const newSett = new SettlementObj({
            x: nx,
            y: ny,
            population: s.population * 0.3,
            food: s.food * 0.3,
            wealth: s.wealth * 0.2,
            defense: 0.3,
            has_port: false,
            alive: true,
            owner_id: s.owner_id,
            tech_level: s.tech_level * 0.5,
          });
          s.population *= 0.7;
          s.food *= 0.7;
          s.wealth *= 0.8;
          settlements.push(newSett);
          // Update occupancy for the new settlement
          for (let ddy = -1; ddy <= 1; ddy++) {
            for (let ddx = -1; ddx <= 1; ddx++) {
              const oy = ny + ddy;
              const ox = nx + ddx;
              if (oy >= 0 && oy < this.H && ox >= 0 && ox < this.W) {
                occ[oy][ox] = true;
              }
            }
          }
          grid[ny][nx] = SETTLEMENT;
        }
      }
    }
  }

  // ── Phase: Conflict ────────────────────────────────────────────────────
  _phaseConflict(grid, settlements, rng) {
    const p = this.params;
    const aggression = p.faction_aggression;
    const baseRange = p.raid_range;

    const alive = settlements.filter((s) => s.alive);
    if (alive.length < 2) return;

    for (let i = 0; i < alive.length; i++) {
      const attacker = alive[i];
      if (!attacker.alive) continue;

      // Raid probability
      const desperation = Math.max(0.0, 1.0 - attacker.food) * 0.4;
      const raidProb = aggression * 0.3 + desperation;
      if (rng.random() > raidProb) continue;

      // Find targets within range
      const raidRange = baseRange * (attacker.has_longship ? 2.5 : 1.0);
      const targetIndices = [];
      for (let j = 0; j < alive.length; j++) {
        if (j === i) continue;
        if (!alive[j].alive) continue;
        if (alive[j].owner_id === attacker.owner_id) continue;
        const dist = manhattanDist(attacker.y, attacker.x, alive[j].y, alive[j].x);
        if (dist <= raidRange && dist > 0) {
          targetIndices.push(j);
        }
      }
      if (targetIndices.length === 0) continue;

      const target = alive[targetIndices[rng.integers(targetIndices.length)]];

      // Resolve combat
      const atkStr = attacker.strength() * (0.6 + 0.8 * rng.random());
      const defStr = target.strength() * (0.6 + 0.8 * rng.random());

      if (atkStr > defStr) {
        // Attacker wins
        const lootFood = target.food * 0.3;
        const lootWealth = target.wealth * 0.3;
        attacker.food += lootFood;
        attacker.wealth += lootWealth;
        target.food -= lootFood;
        target.wealth -= lootWealth;
        target.population *= 0.8;
        target.defense *= 0.7;

        // Chance to conquer
        if (rng.random() < 0.2 * aggression) {
          target.owner_id = attacker.owner_id;
        }
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
    const tradeAct = p.trade_activity;
    if (tradeAct < 0.05) return;

    const ports = settlements.filter((s) => s.alive && s.has_port);
    if (ports.length < 2) return;

    const tradeRange = 12.0;

    for (let i = 0; i < ports.length; i++) {
      const portA = ports[i];
      for (let j = i + 1; j < ports.length; j++) {
        const portB = ports[j];
        if (portA.owner_id === portB.owner_id) continue;
        const dist = manhattanDist(portA.y, portA.x, portB.y, portB.x);
        if (dist > tradeRange) continue;
        if (rng.random() > tradeAct * 0.6) continue;

        // Trade: both benefit
        const tradeValue = 0.1 * tradeAct * (0.5 + rng.random());
        portA.wealth += tradeValue;
        portB.wealth += tradeValue;
        portA.food += tradeValue * 0.5;
        portB.food += tradeValue * 0.5;

        // Tech diffusion
        if (portA.tech_level > portB.tech_level) {
          portB.tech_level += (portA.tech_level - portB.tech_level) * 0.05;
        } else {
          portA.tech_level += (portB.tech_level - portA.tech_level) * 0.05;
        }
      }
    }
  }

  // ── Phase: Winter ──────────────────────────────────────────────────────
  _phaseWinter(grid, settlements, rng) {
    const p = this.params;
    const severity = p.winter_severity;

    for (const s of settlements) {
      if (!s.alive) continue;

      // Food loss from winter
      const foodLoss = severity * (0.5 + 0.8 * rng.random());
      s.food -= foodLoss;

      // Population attrition in harsh winters
      if (s.food < 0) {
        const popLoss = Math.min(Math.abs(s.food) * 0.3, s.population * 0.4);
        s.population -= popLoss;
        s.food = 0.0;
      }

      // Settlement collapse check
      if (s.population < 0.2 || (s.food < 0.1 && rng.random() < 0.3 * severity)) {
        s.alive = false;
        grid[s.y][s.x] = RUIN;

        // Disperse population to nearby friendly settlements
        const nearbyFriendly = settlements.filter(
          (other) =>
            other.alive &&
            other.owner_id === s.owner_id &&
            other !== s &&
            manhattanDist(s.y, s.x, other.y, other.x) <= 5
        );
        if (nearbyFriendly.length > 0 && s.population > 0) {
          const dispersedPop = s.population * 0.5;
          const perSett = dispersedPop / nearbyFriendly.length;
          for (const nf of nearbyFriendly) {
            nf.population += perSett;
          }
        }
        s.population = 0.0;
      }
    }
  }

  // ── Phase: Environment ─────────────────────────────────────────────────
  _phaseEnvironment(grid, settlements, rng, coastal) {
    const p = this.params;
    const forestRate = p.forest_growth_rate;
    const reclaimRate = p.ruin_reclaim_rate;

    // Find all ruins
    const ruins = [];
    for (let y = 0; y < this.H; y++) {
      for (let x = 0; x < this.W; x++) {
        if (grid[y][x] === RUIN) ruins.push([y, x]);
      }
    }

    const aliveSettlements = settlements.filter((s) => s.alive);

    // Precompute forest adjacency count
    const forestAdj = countAdjacent(grid, FOREST);

    for (const [ry, rx] of ruins) {
      // Check if any nearby thriving settlement can reclaim
      let reclaimed = false;
      for (const s of aliveSettlements) {
        const dist = manhattanDist(ry, rx, s.y, s.x);
        if (dist <= 3 && s.population > 1.5 && s.food > 1.0) {
          if (rng.random() < reclaimRate * (0.5 + 0.5 * rng.random())) {
            // Reclaim as new settlement
            const newSett = new SettlementObj({
              x: rx,
              y: ry,
              population: s.population * 0.2,
              food: s.food * 0.2,
              wealth: s.wealth * 0.1,
              defense: 0.3,
              has_port: coastal[ry][rx],
              alive: true,
              owner_id: s.owner_id,
              tech_level: s.tech_level * 0.3,
            });
            s.population *= 0.8;
            s.food *= 0.8;
            settlements.push(newSett);
            grid[ry][rx] = newSett.has_port ? PORT : SETTLEMENT;
            reclaimed = true;
            break;
          }
        }
      }

      if (!reclaimed) {
        // Forest reclaims ruin
        const adjForest = forestAdj[ry][rx];
        let prob;
        if (adjForest > 0) {
          prob = forestRate * 0.15 * Math.min(adjForest, 3);
        } else {
          prob = forestRate * 0.02;
        }
        if (rng.random() < prob) {
          grid[ry][rx] = FOREST;
        } else if (rng.random() < 0.03) {
          // Fade to plains
          grid[ry][rx] = PLAINS;
        }
      }
    }

    // Forest grows on empty/plains adjacent to existing forest
    for (let y = 0; y < this.H; y++) {
      for (let x = 0; x < this.W; x++) {
        if (
          (grid[y][x] === PLAINS || grid[y][x] === EMPTY) &&
          forestAdj[y][x] > 0
        ) {
          const prob = forestRate * 0.02 * Math.min(forestAdj[y][x], 3);
          if (rng.random() < prob) {
            grid[y][x] = FOREST;
          }
        }
      }
    }
  }

  // ── Main simulation loop ───────────────────────────────────────────────
  run(seed) {
    const rng = new SeedableRNG(seed != null ? seed : 0);
    const grid = copyGrid(this.initialGrid);
    let settlements = this._initSettlements();

    // Place initial settlements on grid
    this._rebuildGrid(grid, settlements);

    for (let year = 0; year < 50; year++) {
      // Compute coastal mask once per year
      const coastal = coastalMask(grid);

      this._phaseGrowth(grid, settlements, rng, coastal);
      this._phaseConflict(grid, settlements, rng);
      this._phaseTrade(grid, settlements, rng);
      this._phaseWinter(grid, settlements, rng);
      this._phaseEnvironment(grid, settlements, rng, coastal);

      // Sync grid with settlement states
      this._rebuildGrid(grid, settlements);

      // Compact settlement list every 10 years
      if (year % 10 === 9) {
        settlements = settlements.filter((s) => s.alive);
      }
    }

    return grid;
  }

  runToClasses(seed) {
    const grid = this.run(seed);
    const [H, W] = gridShape(grid);
    const classGrid = makeGrid(H, W, 0);
    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        classGrid[y][x] = TERRAIN_TO_CLASS[grid[y][x]] != null ? TERRAIN_TO_CLASS[grid[y][x]] : 0;
      }
    }
    return classGrid;
  }

  static runMonteCarlo(initialGrid, initialSettlements, params, nRuns, seeds) {
    const sim = new NorseSimulator(initialGrid, initialSettlements, params);
    const H = sim.H;
    const W = sim.W;

    if (!seeds) {
      seeds = [];
      for (let i = 0; i < nRuns; i++) seeds.push(i);
    }

    // Accumulate counts: [H][W][6]
    const counts = [];
    for (let y = 0; y < H; y++) {
      counts[y] = [];
      for (let x = 0; x < W; x++) {
        counts[y][x] = new Float64Array(NUM_CLASSES);
      }
    }

    for (const s of seeds) {
      const classGrid = sim.runToClasses(s);
      for (let y = 0; y < H; y++) {
        for (let x = 0; x < W; x++) {
          counts[y][x][classGrid[y][x]]++;
        }
      }
    }

    // Convert to probabilities with Jeffreys smoothing
    const alpha = 0.5;
    const denom = nRuns + NUM_CLASSES * alpha;
    const probs = [];
    for (let y = 0; y < H; y++) {
      probs[y] = [];
      for (let x = 0; x < W; x++) {
        const p = new Float64Array(NUM_CLASSES);
        let sum = 0;
        for (let c = 0; c < NUM_CLASSES; c++) {
          p[c] = (counts[y][x][c] + alpha) / denom;
          sum += p[c];
        }
        // Normalize
        for (let c = 0; c < NUM_CLASSES; c++) {
          p[c] = Math.max(p[c] / sum, 1e-6);
        }
        probs[y][x] = p;
      }
    }

    // Re-normalize after floor
    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        let sum = 0;
        for (let c = 0; c < NUM_CLASSES; c++) sum += probs[y][x][c];
        for (let c = 0; c < NUM_CLASSES; c++) probs[y][x][c] /= sum;
      }
    }

    return probs;
  }
}

// ── Scoring: KL divergence ─────────────────────────────────────────────────

function computeKLScore(gtProbs, predProbs, H, W) {
  // Entropy-weighted KL divergence (same as competition scoring)
  let totalWeightedKL = 0;
  let totalEntropy = 0;

  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const p = gtProbs[y][x]; // ground truth
      const q = predProbs[y][x]; // prediction

      // Compute entropy of ground truth
      let entropy = 0;
      for (let c = 0; c < NUM_CLASSES; c++) {
        if (p[c] > 1e-10) {
          entropy -= p[c] * Math.log(p[c]);
        }
      }

      // Skip static cells (entropy ~0)
      if (entropy < 1e-6) continue;

      // KL(p || q)
      let kl = 0;
      for (let c = 0; c < NUM_CLASSES; c++) {
        if (p[c] > 1e-10) {
          const qi = Math.max(q[c], 1e-10);
          kl += p[c] * Math.log(p[c] / qi);
        }
      }

      totalWeightedKL += entropy * kl;
      totalEntropy += entropy;
    }
  }

  if (totalEntropy < 1e-10) return 100;
  const weightedKL = totalWeightedKL / totalEntropy;
  return Math.max(0, Math.min(100, 100 * Math.exp(-3 * weightedKL)));
}

// ── Per-class KL analysis ──────────────────────────────────────────────────

function analyzePerClassBias(gtProbs, predProbs, H, W) {
  // Compute average predicted vs GT probability per class (weighted by GT mass)
  const gtSum = new Float64Array(NUM_CLASSES);
  const predSum = new Float64Array(NUM_CLASSES);
  let cellCount = 0;

  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const p = gtProbs[y][x];
      const q = predProbs[y][x];

      // Only count dynamic cells
      let entropy = 0;
      for (let c = 0; c < NUM_CLASSES; c++) {
        if (p[c] > 1e-10) entropy -= p[c] * Math.log(p[c]);
      }
      if (entropy < 1e-6) continue;

      cellCount++;
      for (let c = 0; c < NUM_CLASSES; c++) {
        gtSum[c] += p[c];
        predSum[c] += q[c];
      }
    }
  }

  const biases = [];
  for (let c = 0; c < NUM_CLASSES; c++) {
    const gtAvg = gtSum[c] / Math.max(cellCount, 1);
    const predAvg = predSum[c] / Math.max(cellCount, 1);
    const ratio = gtAvg > 0.001 ? predAvg / gtAvg : 0;
    biases.push({
      class: CLASS_NAMES[c],
      gtAvg: gtAvg,
      predAvg: predAvg,
      ratio: ratio,
    });
  }
  return biases;
}

// ── File I/O ───────────────────────────────────────────────────────────────

const CACHE_DIR = path.join(__dirname, "cache");

function loadJSON(filepath) {
  try {
    return JSON.parse(fs.readFileSync(filepath, "utf8"));
  } catch {
    return null;
  }
}

function discoverRounds() {
  const rounds = {};
  if (!fs.existsSync(CACHE_DIR)) {
    console.log("Cache directory not found:", CACHE_DIR);
    return rounds;
  }
  const files = fs.readdirSync(CACHE_DIR);

  // Find init files: r{N}_init.json
  for (const f of files) {
    const m = f.match(/^r(\d+)_init\.json$/);
    if (m) {
      const roundNum = parseInt(m[1]);
      if (!rounds[roundNum]) rounds[roundNum] = { init: null, gts: {} };
      rounds[roundNum].init = path.join(CACHE_DIR, f);
    }
  }

  // Find GT files: r{N}_gt_s{S}.json
  for (const f of files) {
    const m = f.match(/^r(\d+)_gt_s(\d+)\.json$/);
    if (m) {
      const roundNum = parseInt(m[1]);
      const seedIdx = parseInt(m[2]);
      if (!rounds[roundNum]) rounds[roundNum] = { init: null, gts: {} };
      rounds[roundNum].gts[seedIdx] = path.join(CACHE_DIR, f);
    }
  }

  return rounds;
}

// ── Grid search ────────────────────────────────────────────────────────────

function gridSearch(initData, gtData, seedIndex) {
  const winterValues = [0.2, 0.3, 0.4, 0.5, 0.6];
  const aggressionValues = [0.1, 0.2, 0.3, 0.4, 0.5];
  const tradeValues = [0.2, 0.4, 0.6, 0.8];
  const expansionValues = [0.1, 0.2, 0.3, 0.4];
  const forestValues = [0.02, 0.05, 0.1];

  const totalCombos =
    winterValues.length *
    aggressionValues.length *
    tradeValues.length *
    expansionValues.length *
    forestValues.length;

  console.log(`  Grid search: ${totalCombos} combos, 20 MC runs each, seed ${seedIndex} only`);

  // Parse GT data into probs array
  const H = gtData.height || gtData.ground_truth.length;
  const W = gtData.width || gtData.ground_truth[0].length;
  const gtProbs = gtData.ground_truth; // [H][W][6]

  const initialGrid = initData.grid;
  const initialSettlements = initData.settlements;

  let bestScore = -Infinity;
  let bestParams = null;
  let tested = 0;

  for (const winter of winterValues) {
    for (const aggression of aggressionValues) {
      for (const trade of tradeValues) {
        for (const expansion of expansionValues) {
          for (const forest of forestValues) {
            const params = {
              ...DEFAULT_PARAMS,
              winter_severity: winter,
              faction_aggression: aggression,
              trade_activity: trade,
              expansion_rate: expansion,
              forest_growth_rate: forest,
            };

            const predProbs = NorseSimulator.runMonteCarlo(
              initialGrid,
              initialSettlements,
              params,
              20 // 20 MC runs for speed
            );

            const score = computeKLScore(gtProbs, predProbs, H, W);

            if (score > bestScore) {
              bestScore = score;
              bestParams = {
                winter_severity: winter,
                faction_aggression: aggression,
                trade_activity: trade,
                expansion_rate: expansion,
                forest_growth_rate: forest,
              };
            }

            tested++;
            if (tested % 100 === 0) {
              process.stdout.write(
                `\r  Progress: ${tested}/${totalCombos} (best so far: ${bestScore.toFixed(1)})`
              );
            }
          }
        }
      }
    }
  }
  console.log(`\r  Progress: ${tested}/${totalCombos} (best: ${bestScore.toFixed(1)})        `);

  return { bestScore, bestParams };
}

// ── Main ───────────────────────────────────────────────────────────────────

function main() {
  console.log("=== Astar Island Simulator Calibration ===\n");

  const rounds = discoverRounds();
  const roundNums = Object.keys(rounds)
    .map(Number)
    .sort((a, b) => a - b);

  if (roundNums.length === 0) {
    console.log("No cached rounds found in", CACHE_DIR);
    console.log(
      "Expected files: r{N}_init.json and r{N}_gt_s{S}.json"
    );
    console.log("\nTo populate cache, run the main agent to fetch round data and GT analysis.");
    process.exit(0);
  }

  console.log(`Found ${roundNums.length} round(s): ${roundNums.join(", ")}\n`);

  const calibratedResults = {};

  for (const roundNum of roundNums) {
    const round = rounds[roundNum];
    console.log(`\n--- Round ${roundNum} ---`);

    // Load init data
    if (!round.init) {
      console.log("  No init file found, skipping.");
      continue;
    }
    const initDataRaw = loadJSON(round.init);
    if (!initDataRaw) {
      console.log("  Failed to load init file, skipping.");
      continue;
    }

    // initData has initial_states array (one per seed) or is the round detail
    const initialStates = initDataRaw.initial_states || [initDataRaw];

    const seedIndices = Object.keys(round.gts)
      .map(Number)
      .sort((a, b) => a - b);

    if (seedIndices.length === 0) {
      console.log("  No GT files found, skipping.");
      continue;
    }

    console.log(`  GT available for seeds: ${seedIndices.join(", ")}`);

    // Step 1: Analyze bias with DEFAULT_PARAMS for each seed
    console.log("\n  [1] Bias analysis with DEFAULT_PARAMS (50 MC runs):");
    for (const si of seedIndices) {
      const gtData = loadJSON(round.gts[si]);
      if (!gtData || !gtData.ground_truth) {
        console.log(`  Seed ${si}: no ground_truth in file, skipping.`);
        continue;
      }

      const H = gtData.height || gtData.ground_truth.length;
      const W = gtData.width || gtData.ground_truth[0].length;

      // Pick the right initial state for this seed
      const initState = initialStates[si] || initialStates[0];
      if (!initState || !initState.grid) {
        console.log(`  Seed ${si}: no init grid, skipping.`);
        continue;
      }

      const predProbs = NorseSimulator.runMonteCarlo(
        initState.grid,
        initState.settlements,
        DEFAULT_PARAMS,
        50
      );

      const score = computeKLScore(gtData.ground_truth, predProbs, H, W);
      const biases = analyzePerClassBias(gtData.ground_truth, predProbs, H, W);

      console.log(`\n  Seed ${si}: score=${score.toFixed(1)}`);
      for (const b of biases) {
        if (b.gtAvg < 0.001 && b.predAvg < 0.001) continue;
        const dir =
          b.ratio > 1.3
            ? `${b.ratio.toFixed(1)}x too HIGH`
            : b.ratio < 0.7
            ? `${(1 / b.ratio).toFixed(1)}x too LOW`
            : "OK";
        console.log(
          `    ${b.class.padEnd(12)} GT=${b.gtAvg.toFixed(4)} Pred=${b.predAvg.toFixed(4)} ratio=${b.ratio.toFixed(2)} ${dir}`
        );
      }
    }

    // Step 2: Grid search (use seed 0 of available GT, or first available)
    const searchSeed = seedIndices.includes(0) ? 0 : seedIndices[0];
    const gtData = loadJSON(round.gts[searchSeed]);
    if (!gtData || !gtData.ground_truth) {
      console.log("  Cannot run grid search: no valid GT data.");
      continue;
    }

    const initState = initialStates[searchSeed] || initialStates[0];
    if (!initState || !initState.grid) {
      console.log("  Cannot run grid search: no init grid.");
      continue;
    }

    console.log(`\n  [2] Grid search (seed ${searchSeed}):`);
    const { bestScore, bestParams } = gridSearch(initState, gtData, searchSeed);

    console.log(
      `\n  Round ${roundNum}: best_score=${bestScore.toFixed(1)}, params=${JSON.stringify(bestParams)}`
    );

    calibratedResults[`round_${roundNum}`] = {
      best_score: bestScore,
      best_params: bestParams,
      search_seed: searchSeed,
      gt_seeds_available: seedIndices,
    };
  }

  // Save results
  if (Object.keys(calibratedResults).length > 0) {
    const outPath = path.join(CACHE_DIR, "calibrated_params.json");
    fs.writeFileSync(outPath, JSON.stringify(calibratedResults, null, 2));
    console.log(`\nResults saved to ${outPath}`);
  }

  // Summary
  console.log("\n=== Summary ===");
  for (const [key, val] of Object.entries(calibratedResults)) {
    console.log(
      `${key}: best_score=${val.best_score.toFixed(1)}, params={winter: ${val.best_params.winter_severity}, aggression: ${val.best_params.faction_aggression}, trade: ${val.best_params.trade_activity}, expansion: ${val.best_params.expansion_rate}, forest: ${val.best_params.forest_growth_rate}}`
    );
  }
}

main();
