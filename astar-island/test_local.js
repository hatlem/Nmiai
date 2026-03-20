#!/usr/bin/env node
/**
 * Local test harness for Astar Island browser agent.
 *
 * Fetches ground truth from completed rounds, caches it locally,
 * then simulates our prediction pipeline using synthetic observations
 * (sampled from GT distributions) and scores against ground truth.
 *
 * Usage:
 *   node test_local.js --token YOUR_BEARER_TOKEN --round 2
 *   node test_local.js --round 1          # uses cached data
 *   node test_local.js --round 2 --fetch  # force re-fetch
 */

const fs = require("fs");
const path = require("path");
const https = require("https");

const API = "https://api.ainm.no/astar-island";
const CACHE_DIR = path.join(__dirname, "cache");
const NUM_CLASSES = 6;
const PROB_FLOOR = 0.01;
const STATIC_FLOOR = 0.002;
const REMOTE_FLOOR = 0.003;
const VIEWPORT_MAX = 15;

// ── CLI args ────────────────────────────────────────────────────────────────

function parseArgs() {
  const args = {
    round: null,
    token: null,
    fetch: false,
    syntheticN: 10, // samples per synthetic observation
    help: false,
  };
  const argv = process.argv.slice(2);
  for (let i = 0; i < argv.length; i++) {
    if (argv[i] === "--round" && argv[i + 1]) args.round = parseInt(argv[++i]);
    else if (argv[i] === "--token" && argv[i + 1]) args.token = argv[++i];
    else if (argv[i] === "--fetch") args.fetch = true;
    else if (argv[i] === "--synthetic-n" && argv[i + 1]) args.syntheticN = parseInt(argv[++i]);
    else if (argv[i] === "--help" || argv[i] === "-h") args.help = true;
  }
  return args;
}

// ── HTTP helpers ────────────────────────────────────────────────────────────

function httpGet(url, token) {
  return new Promise((resolve, reject) => {
    const headers = {};
    if (token) headers["Authorization"] = `Bearer ${token}`;
    const urlObj = new URL(url);
    const opts = {
      hostname: urlObj.hostname,
      path: urlObj.pathname + urlObj.search,
      method: "GET",
      headers,
    };
    const req = https.request(opts, (res) => {
      let data = "";
      res.on("data", (chunk) => (data += chunk));
      res.on("end", () => {
        if (res.statusCode >= 400)
          return reject(new Error(`HTTP ${res.statusCode}: ${data.slice(0, 200)}`));
        try { resolve(JSON.parse(data)); }
        catch (e) { reject(new Error(`JSON parse error: ${data.slice(0, 100)}`)); }
      });
    });
    req.on("error", reject);
    req.end();
  });
}

// ── Cache ───────────────────────────────────────────────────────────────────

function ensureCache() {
  if (!fs.existsSync(CACHE_DIR)) fs.mkdirSync(CACHE_DIR, { recursive: true });
}

function cacheKey(roundNum, type, seedIdx) {
  if (seedIdx !== undefined) return path.join(CACHE_DIR, `r${roundNum}_${type}_s${seedIdx}.json`);
  return path.join(CACHE_DIR, `r${roundNum}_${type}.json`);
}

function loadCache(roundNum, type, seedIdx) {
  const p = cacheKey(roundNum, type, seedIdx);
  if (fs.existsSync(p)) return JSON.parse(fs.readFileSync(p, "utf-8"));
  return null;
}

function saveCache(roundNum, type, data, seedIdx) {
  ensureCache();
  fs.writeFileSync(cacheKey(roundNum, type, seedIdx), JSON.stringify(data));
}

// ── Data fetching ───────────────────────────────────────────────────────────

async function fetchRoundData(roundNum, token, forceFetch) {
  // Try cache first
  let roundsList = loadCache(roundNum, "rounds_list");
  if (!roundsList || forceFetch) {
    if (!token) throw new Error("No token provided and no cached data. Use --token.");
    console.log("Fetching rounds list...");
    roundsList = await httpGet(`${API}/rounds`, token);
    saveCache(roundNum, "rounds_list", roundsList);
  }

  const roundInfo = roundsList.find((r) => r.round_number === roundNum);
  if (!roundInfo) throw new Error(`Round ${roundNum} not found. Available: ${roundsList.map(r => r.round_number).join(", ")}`);

  // Fetch round detail (initial states)
  let detail = loadCache(roundNum, "detail");
  if (!detail || forceFetch) {
    if (!token) throw new Error("No token and no cached detail. Use --token.");
    console.log(`Fetching round ${roundNum} detail (id: ${roundInfo.id})...`);
    detail = await httpGet(`${API}/rounds/${roundInfo.id}`, token);
    saveCache(roundNum, "detail", detail);
  }

  // Fetch analysis (ground truth) for each seed
  const seedsCount = detail.seeds_count || 5;
  const analyses = [];
  for (let si = 0; si < seedsCount; si++) {
    let analysis = loadCache(roundNum, "analysis", si);
    if (!analysis || forceFetch) {
      if (!token) throw new Error(`No token and no cached analysis for seed ${si}. Use --token.`);
      console.log(`Fetching analysis for round ${roundNum}, seed ${si}...`);
      analysis = await httpGet(`${API}/analysis/${roundInfo.id}/${si}`, token);
      saveCache(roundNum, "analysis", analysis, si);
      // Rate limit courtesy
      await new Promise((r) => setTimeout(r, 300));
    }
    analyses.push(analysis);
  }

  return { roundInfo, detail, analyses };
}

// ── Scoring (exact competition formula) ─────────────────────────────────────

function entropy(p) {
  let h = 0;
  for (let i = 0; i < p.length; i++) {
    if (p[i] > 1e-10) h -= p[i] * Math.log(p[i]);
  }
  return h;
}

function klDiv(gt, pred) {
  let kl = 0;
  const eps = 1e-10;
  for (let i = 0; i < gt.length; i++) {
    const p = Math.max(gt[i], eps);
    const q = Math.max(pred[i], eps);
    kl += p * Math.log(p / q);
  }
  return kl;
}

function scoreOneSeed(prediction, groundTruth, H, W) {
  let totalEntropy = 0;
  let weightedKL = 0;
  const perClassLoss = new Float64Array(NUM_CLASSES);
  const perClassWeight = new Float64Array(NUM_CLASSES);

  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const gt = groundTruth[y][x];
      const pred = prediction[y][x];
      const h = entropy(gt);
      totalEntropy += h;
      if (h < 1e-10) continue;

      const cellKL = klDiv(gt, pred);
      weightedKL += h * cellKL;

      // Per-class contribution to KL
      for (let c = 0; c < NUM_CLASSES; c++) {
        const p = Math.max(gt[c], 1e-10);
        const q = Math.max(pred[c], 1e-10);
        const contrib = h * p * Math.log(p / q);
        perClassLoss[c] += contrib;
        perClassWeight[c] += h * p;
      }
    }
  }

  if (totalEntropy < 1e-10) return { score: 100, wkl: 0, perClassLoss: [], totalEntropy: 0 };

  const wkl = weightedKL / totalEntropy;
  const score = Math.max(0, Math.min(100, 100 * Math.exp(-3 * wkl)));

  // Normalize per-class loss by total weighted entropy for interpretability
  const classBreakdown = [];
  for (let c = 0; c < NUM_CLASSES; c++) {
    classBreakdown.push({
      class: c,
      name: ["Empty", "Settlement", "Port", "Ruin", "Forest", "Mountain"][c],
      loss: perClassLoss[c] / totalEntropy,
      weight: perClassWeight[c] / totalEntropy,
    });
  }

  return { score, wkl, totalEntropy, classBreakdown };
}

// ── Synthetic observations ──────────────────────────────────────────────────

function multinomialSample(probs, n, rng) {
  // Sample n items from multinomial(probs)
  const counts = new Float64Array(probs.length);
  for (let i = 0; i < n; i++) {
    let r = rng();
    let cumSum = 0;
    for (let j = 0; j < probs.length; j++) {
      cumSum += probs[j];
      if (r < cumSum) {
        counts[j]++;
        break;
      }
      if (j === probs.length - 1) counts[j]++;
    }
  }
  return counts;
}

// Simple seeded RNG (xorshift128)
function makeRng(seed) {
  let s = seed | 0 || 1;
  return function () {
    s ^= s << 13;
    s ^= s >> 17;
    s ^= s << 5;
    return ((s >>> 0) / 4294967296);
  };
}

function generateSyntheticObservation(gt, vx, vy, vw, vh, H, W, nSamples, rng) {
  // Returns a grid of terrain codes sampled from GT distribution
  const grid = [];
  for (let gy = 0; gy < vh && vy + gy < H; gy++) {
    const row = [];
    for (let gx = 0; gx < vw && vx + gx < W; gx++) {
      const probs = gt[vy + gy][vx + gx];
      // Sample one outcome from the GT distribution
      let r = rng();
      let cumSum = 0;
      let cls = 0;
      for (let c = 0; c < NUM_CLASSES; c++) {
        cumSum += probs[c];
        if (r < cumSum) { cls = c; break; }
        if (c === NUM_CLASSES - 1) cls = c;
      }
      // Map class back to terrain code (reverse mapping)
      const classToCode = [11, 1, 2, 3, 4, 5]; // 0->Plains(11), 1->Settlement, etc.
      row.push(classToCode[cls]);
    }
    grid.push(row);
  }
  return grid;
}

// ── Import browser agent logic (inline since it's an IIFE) ──────────────────

// We replicate the key prediction functions from browser_agent_v8.js
// to test them without a browser environment.

const TERRAIN_TO_CLASS = { 10: 0, 11: 0, 0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5 };

const CALIBRATED_PRIORS = {
  0: [0.82, 0.13, 0.012, 0.010, 0.028, 0.005],
  1: [0.37, 0.41, 0.008, 0.031, 0.181, 0.005],
  2: [0.36, 0.12, 0.319, 0.021, 0.176, 0.005],
  3: [0.15, 0.12, 0.03, 0.35, 0.30, 0.05],
  4: [0.07, 0.16, 0.014, 0.012, 0.744, 0.005],
  5: [0.002, 0.002, 0.002, 0.002, 0.002, 0.990],
};

// Load GT_CTX_PRIORS from browser_agent_v8.js (copy the object)
const GT_CTX_PRIORS = {"0_0_1_far":[0.997266,0.001641,0.000938,0.000039,0.000117,0],"0_1_1_far":[0.996621,0.001862,0.001241,0.000172,0.000103,0],"0_1_1_mid":[0.940438,0.023013,0.02745,0.002988,0.006112,0],"0_0_1_mid":[0.947645,0.019518,0.024174,0.002479,0.006185,0],"0_2_1_mid":[0.913814,0.032468,0.040513,0.003846,0.009359,0],"0_2_1_near":[0.802195,0.064573,0.106098,0.007805,0.019329,0],"0_1_1_near":[0.830683,0.056522,0.085714,0.008758,0.018323,0],"0_0_1_near":[0.8676,0.043567,0.066433,0.0059,0.0165,0],"0_2_1_far":[0.987353,0.007843,0.004412,0.000196,0.000196,0],"4_1_1_mid":[0.022805,0.062195,0.063659,0.006098,0.845244,0],"0_3_1_mid":[0.882935,0.045109,0.056957,0.004891,0.010109,0],"4_3_1_mid":[0.031667,0.071111,0.129444,0.005,0.762778,0],"4_2_1_near":[0.0648,0.105,0.1724,0.016,0.6418,0],"0_3_1_near":[0.694595,0.093784,0.176351,0.012297,0.022973,0],"4_1_1_near":[0.067742,0.12629,0.195,0.016452,0.594516,0],"4_3_1_near":[0.06,0.096667,0.164167,0.009167,0.67,0],"4_2_1_mid":[0.021129,0.064839,0.082581,0.004032,0.827419,0],"0_3_0_near":[0.711319,0.225434,0,0.015882,0.047365,0],"0_3_0_mid":[0.839451,0.126608,0,0.010285,0.023656,0],"0_2_0_mid":[0.835881,0.130559,0,0.01033,0.02323,0],"4_3_0_near":[0.100996,0.223996,0,0.017324,0.657684,0],"4_2_0_near":[0.100788,0.221362,0,0.016183,0.661667,0],"4_3_0_mid":[0.049462,0.139247,0,0.008656,0.802634,0],"4_2_0_mid":[0.050576,0.131212,0,0.009697,0.808515,0],"1_2_0_near":[0.366277,0.418936,0,0.030213,0.184574,0],"0_2_0_near":[0.719964,0.215418,0,0.015815,0.048802,0],"1_3_0_near":[0.353681,0.440417,0,0.034236,0.171667,0],"0_1_0_near":[0.720663,0.21623,0,0.015715,0.047392,0],"4_1_0_mid":[0.053733,0.128567,0,0.0107,0.807,0],"4_1_0_near":[0.097399,0.224753,0,0.017063,0.660785,0],"1_1_0_near":[0.381132,0.406887,0,0.029245,0.182736,0],"0_0_0_near":[0.719589,0.214929,0,0.015911,0.049571,0],"4_0_0_near":[0.1086,0.207733,0,0.0182,0.665467,0],"5_2_0_near":[0,0,0,0,0,1],"5_1_0_mid":[0,0,0,0,0,1],"5_0_0_mid":[0,0,0,0,0,1],"0_0_0_mid":[0.848254,0.119822,0,0.00929,0.022633,0],"5_1_0_near":[0,0,0,0,0,1],"5_2_0_mid":[0,0,0,0,0,1],"0_1_0_far":[0.984737,0.01307,0,0.000877,0.001316,0],"0_2_0_far":[0.986176,0.012882,0,0.000353,0.000588,0],"0_0_0_far":[0.990526,0.008421,0,0.000263,0.000789,0],"0_3_0_far":[0.984263,0.013947,0,0.000737,0.001053,0],"4_3_0_far":[0.002875,0.010875,0,0.001,0.98525,0],"4_1_0_far":[0.001833,0.0135,0,0.000333,0.984333,0],"4_2_0_far":[0.002,0.012143,0,0.000357,0.985500,0],"4_0_0_mid":[0.051058,0.129904,0,0.010673,0.808365,0],"1_0_0_near":[0.4152,0.3564,0,0.029,0.1994,0]};

function classifyCode(code) { return TERRAIN_TO_CLASS[code] ?? 0; }

function make3D(H, W, D, val) {
  return Array.from({ length: H }, () =>
    Array.from({ length: W }, () => new Float64Array(D).fill(val))
  );
}

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
      floors[1] = REMOTE_FLOOR; floors[2] = REMOTE_FLOOR; floors[3] = REMOTE_FLOOR;
    }
    if (initCls === 0 && settDistVal > 8) {
      floors[1] = REMOTE_FLOOR; floors[2] = REMOTE_FLOOR; floors[3] = REMOTE_FLOOR;
    }
    if (initCls === 4 && settDistVal > 8) {
      floors[1] = REMOTE_FLOOR; floors[2] = REMOTE_FLOOR; floors[3] = REMOTE_FLOOR;
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

function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }

function getCellPrior(initCls, sd, food, coastal, nSett) {
  const foodBucket = Math.min(food, 3);
  const distBucket = sd <= 3 ? "near" : sd <= 7 ? "mid" : "far";
  const coastInt = coastal ? 1 : 0;
  const gtKey = `${initCls}_${foodBucket}_${coastInt}_${distBucket}`;
  const gtPrior = GT_CTX_PRIORS[gtKey];
  if (gtPrior) {
    const base = new Float64Array(NUM_CLASSES);
    for (let i = 0; i < NUM_CLASSES; i++) base[i] = Math.max(gtPrior[i], 0.002);
    normalize(base);
    return base;
  }
  const src = CALIBRATED_PRIORS[initCls] || CALIBRATED_PRIORS[0];
  const base = new Float64Array(NUM_CLASSES);
  for (let i = 0; i < NUM_CLASSES; i++) base[i] = src[i];
  if (coastal) {
    if (initCls === 1 || initCls === 2) {
      base[2] = 0.30; base[1] = 0.15; base[0] = 0.35; base[4] = 0.17; base[3] = 0.02;
    } else if (initCls === 0 && sd <= 3) {
      base[2] = 0.10; base[1] = 0.06; base[0] = 0.80;
    } else if (initCls === 4 && sd <= 3) {
      base[2] = 0.17; base[1] = 0.12; base[4] = 0.60; base[0] = 0.07;
    } else if (initCls === 0 && sd <= 7) {
      base[2] = 0.03; base[1] = 0.03; base[0] = 0.92;
    }
  } else {
    base[2] = 0.002;
    if (sd > 6 && initCls === 0) {
      base[0] = 0.92; base[1] = 0.01; base[3] = 0.01; base[4] = 0.04; base[5] = 0.01;
    } else if (sd > 6 && initCls === 4) {
      base[4] = 0.90; base[0] = 0.04; base[1] = 0.01; base[3] = 0.01; base[5] = 0.01;
    } else if (sd <= 3 && initCls === 0) {
      base[1] = 0.22; base[0] = 0.72; base[3] = 0.016; base[4] = 0.04;
    } else if (sd <= 7 && initCls === 0) {
      base[1] = 0.13; base[0] = 0.84; base[3] = 0.01; base[4] = 0.02;
    } else if (sd <= 3 && initCls === 4) {
      base[1] = 0.22; base[4] = 0.66; base[0] = 0.10; base[3] = 0.017;
    } else if (sd <= 7 && initCls === 4) {
      base[1] = 0.13; base[4] = 0.81; base[0] = 0.05; base[3] = 0.01;
    }
  }
  if (nSett >= 2 && initCls === 0 && sd <= 4) {
    base[1] += 0.04; base[0] -= 0.03;
  }
  for (let i = 0; i < NUM_CLASSES; i++) base[i] = Math.max(base[i], 0.002);
  base[5] = (initCls === 5) ? 0.99 : 0.002;
  normalize(base);
  return base;
}

function getPriorStrength(initCls, sd, nObs) {
  let base;
  if (initCls === 5) base = 4.0;
  else if (initCls === 1 || initCls === 2) base = 0.5;
  else if (sd <= 3) base = 0.8;
  else if (sd > 6 && initCls === 0) base = 1.5;
  else base = 1.0;
  if (nObs > 0) base = base / (1.0 + nObs / base);
  return base;
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
    compat[4][j] = 0.02; compat[j][4] = 0.02;
  }
  for (let j = 0; j < 5; j++) {
    compat[5][j] = 0.01; compat[j][5] = 0.01;
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

function layeredPredict(grid, settlements, counts, contextMap, initialStates, seedIdx, H, W) {
  const pred = make3D(H, W, NUM_CLASSES, 0);
  const coastal = coastalMask(grid, H, W);
  const observedMask = Array.from({ length: H }, () => new Uint8Array(W));

  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const raw = grid[y][x];
      if (raw === 5) { pred[y][x] = mountainPrior(); continue; }
      if (raw === 10) { pred[y][x] = oceanPrior(); continue; }

      const ic = classifyCode(raw);
      const food = foodPotential(grid, H, W, y, x);
      const sd = settDist(grid, H, W, settlements, y, x);
      const isCoast = !!coastal[y][x];
      const nSett = neighborSettlements(grid, H, W, y, x);
      const nObs = counts[y][x].reduce((a, b) => a + b, 0);

      const prior = getCellPrior(ic, sd, food, isCoast, nSett);
      const strength = getPriorStrength(ic, sd, nObs);
      const cellFloors = getCellFloors(ic, false, sd);

      if (nObs >= 5) {
        observedMask[y][x] = 1;
        const denom = nObs + strength;
        for (let c = 0; c < NUM_CLASSES; c++)
          pred[y][x][c] = (counts[y][x][c] + prior[c] * strength) / denom;
        floorNorm(pred[y][x], cellFloors);
      } else if (nObs >= 1) {
        observedMask[y][x] = 1;
        const denom = nObs + strength;
        const kt = new Float64Array(NUM_CLASSES);
        for (let c = 0; c < NUM_CLASSES; c++)
          kt[c] = (counts[y][x][c] + prior[c] * strength) / denom;
        const ctxPred = contextPoolCell(contextMap, grid, H, W, settlements, y, x);
        if (ctxPred) {
          const ktW = 0.25 + 0.125 * nObs;
          for (let c = 0; c < NUM_CLASSES; c++)
            pred[y][x][c] = ktW * kt[c] + (1 - ktW) * ctxPred[c];
        } else {
          for (let c = 0; c < NUM_CLASSES; c++) pred[y][x][c] = kt[c];
        }
        floorNorm(pred[y][x], cellFloors);
      } else {
        const ctxPred = contextPoolCell(contextMap, grid, H, W, settlements, y, x);
        if (ctxPred) {
          for (let c = 0; c < NUM_CLASSES; c++) pred[y][x][c] = ctxPred[c];
        } else {
          for (let c = 0; c < NUM_CLASSES; c++) pred[y][x][c] = prior[c];
        }
        if (!isCoast) pred[y][x][2] = cellFloors[2];
        floorNorm(pred[y][x], cellFloors);
      }
    }
  }

  const result = beliefPropagation(pred, observedMask, grid, H, W);

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

// ── New QueryOptimizer (matching the v8 fix) ────────────────────────────────

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

    // Importance: 10=settlement, 5=coastal near sett, 3=adjacent to sett, 1=other dynamic, 0=static
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
          imp[y][x] = 0; // No queries for remote cells
        }
      this.importanceMaps.push(imp);
    }

    // Find minimum covering viewports per seed (settlement + 3-cell buffer)
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

    // Greedy set cover: cover all settlements with minimum viewports, 3-cell buffer
    const buffer = 3;
    const uncovered = new Set(coords.map((_, i) => i));
    const viewports = [];

    while (uncovered.size > 0) {
      let bestVP = null, bestCov = new Set(), bestScore = -1;

      // Try centering on each uncovered settlement
      for (const i of uncovered) {
        const [sx, sy] = coords[i];
        // Try viewport centered on this settlement with buffer
        const vx = clamp(sx - Math.floor(VIEWPORT_MAX / 2), 0, Math.max(0, this.W - VIEWPORT_MAX));
        const vy = clamp(sy - Math.floor(VIEWPORT_MAX / 2), 0, Math.max(0, this.H - VIEWPORT_MAX));
        const vw = Math.min(VIEWPORT_MAX, this.W - vx);
        const vh = Math.min(VIEWPORT_MAX, this.H - vy);

        // Count settlements covered (with buffer check)
        const contained = new Set();
        for (const j of uncovered) {
          const [cx, cy] = coords[j];
          if (cx >= vx + buffer && cx < vx + vw - buffer &&
              cy >= vy + buffer && cy < vy + vh - buffer) {
            contained.add(j);
          }
        }
        // Fallback: if no settlement fits with buffer, count without buffer
        if (contained.size === 0) {
          for (const j of uncovered) {
            const [cx, cy] = coords[j];
            if (cx >= vx && cx < vx + vw && cy >= vy && cy < vy + vh) {
              contained.add(j);
            }
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

    // Phase 1: One query per unique viewport per seed
    for (let si = 0; si < this.seedsCount; si++) {
      for (const vp of this.seedViewports[si]) {
        if (plan.length >= this.budget) break;
        plan.push([si, ...vp]);
        recordQuery(si, vp[0], vp[1], vp[2], vp[3]);
      }
    }

    // Build candidate set: ONLY settlement viewports (no expansion)
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

    // Phase 2: ALL remaining queries — adaptive repeat, highest marginal info gain
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
}

// ── Main test ───────────────────────────────────────────────────────────────

async function main() {
  const args = parseArgs();

  if (args.help) {
    console.log(`
Astar Island Local Test Harness

Usage:
  node test_local.js --token TOKEN --round 2    Fetch + test round 2
  node test_local.js --round 1                  Test from cache
  node test_local.js --round 2 --fetch          Force re-fetch
  node test_local.js --round 2 --synthetic-n 5  Fewer samples per obs

Options:
  --round N          Round number (1 or 2)
  --token TOKEN      Bearer token for API auth
  --fetch            Force re-fetch even if cached
  --synthetic-n N    Samples per synthetic observation (default: 10)
  --help             Show this help
`);
    return;
  }

  if (!args.round) {
    console.error("ERROR: --round required. Use --help for usage.");
    process.exit(1);
  }

  console.log("=".repeat(60));
  console.log("ASTAR ISLAND LOCAL TEST HARNESS");
  console.log("=".repeat(60));

  // Fetch/load data
  const { roundInfo, detail, analyses } = await fetchRoundData(args.round, args.token, args.fetch);
  const H = detail.map_height;
  const W = detail.map_width;
  const seedsCount = detail.seeds_count || 5;
  const initialStates = detail.initial_states;

  console.log(`\nRound ${args.round}: ${W}x${H}, ${seedsCount} seeds`);
  for (let si = 0; si < seedsCount; si++) {
    const setts = initialStates[si].settlements || [];
    console.log(`  Seed ${si}: ${setts.length} settlements, score from API: ${analyses[si].score?.toFixed(2) || "N/A"}`);
  }

  // Build query plan using our optimizer
  const budget = 50;
  const optimizer = new QueryOptimizer(W, H, seedsCount, budget, initialStates);
  const plan = optimizer.planQueries();

  // Count queries per seed and unique viewports
  const seedQueryCounts = new Array(seedsCount).fill(0);
  const vpPerSeed = new Array(seedsCount).fill(0);
  const vpSeen = {};
  for (const [si, vx, vy, vw, vh] of plan) {
    seedQueryCounts[si]++;
    const key = `${si}_${vx}_${vy}`;
    if (!vpSeen[key]) { vpSeen[key] = true; vpPerSeed[si]++; }
  }

  console.log(`\nQuery plan: ${plan.length} queries`);
  for (let si = 0; si < seedsCount; si++)
    console.log(`  Seed ${si}: ${seedQueryCounts[si]} queries, ${vpPerSeed[si]} unique viewports, ${optimizer.seedViewports[si].length} settlement viewports`);

  // Simulate observations using GT distributions
  const rng = makeRng(42);
  const allCounts = {};
  for (let si = 0; si < seedsCount; si++)
    allCounts[si] = make3D(H, W, NUM_CLASSES, 0);

  for (const [seedIdx, qx, qy, qw, qh] of plan) {
    const gt = analyses[seedIdx].ground_truth;
    const synGrid = generateSyntheticObservation(gt, qx, qy, qw, qh, H, W, 1, rng);
    for (let gy = 0; gy < synGrid.length; gy++) {
      for (let gx = 0; gx < synGrid[gy].length; gx++) {
        const absY = qy + gy;
        const absX = qx + gx;
        if (absY < H && absX < W) {
          const cls = classifyCode(synGrid[gy][gx]);
          allCounts[seedIdx][absY][absX][cls] += 1;
        }
      }
    }
  }

  // Coverage stats
  for (let si = 0; si < seedsCount; si++) {
    let obsCells = 0, totalObs = 0;
    for (let y = 0; y < H; y++)
      for (let x = 0; x < W; x++) {
        const n = allCounts[si][y][x].reduce((a, b) => a + b, 0);
        if (n > 0) { obsCells++; totalObs += n; }
      }
    const avgObs = obsCells > 0 ? (totalObs / obsCells).toFixed(1) : 0;
    console.log(`  Seed ${si}: ${obsCells}/${W * H} cells observed, avg ${avgObs} obs/cell`);
  }

  // Build predictions
  console.log("\nBuilding predictions...");
  const contextMap = buildContextPool(initialStates, allCounts, H, W);
  console.log(`  Context pool: ${Object.keys(contextMap).length} groups`);

  const predictions = {};
  for (let si = 0; si < seedsCount; si++) {
    const grid = initialStates[si].grid;
    const settlements = initialStates[si].settlements || [];
    predictions[si] = layeredPredict(grid, settlements, allCounts[si], contextMap, initialStates, si, H, W);
  }

  // Score against ground truth
  console.log("\n" + "=".repeat(60));
  console.log("SCORES");
  console.log("=".repeat(60));

  const scores = [];
  const allClassBreakdown = new Array(NUM_CLASSES).fill(0).map(() => ({ loss: 0, weight: 0 }));

  for (let si = 0; si < seedsCount; si++) {
    const gt = analyses[si].ground_truth;
    const result = scoreOneSeed(predictions[si], gt, H, W);
    scores.push(result.score);

    console.log(`  Seed ${si}: ${result.score.toFixed(2)} (wkl=${result.wkl.toFixed(4)})`);
    if (result.classBreakdown) {
      for (const cb of result.classBreakdown) {
        console.log(`    ${cb.name.padEnd(12)} loss=${cb.loss.toFixed(4)}  weight=${cb.weight.toFixed(3)}`);
        allClassBreakdown[cb.class].loss += cb.loss;
        allClassBreakdown[cb.class].weight += cb.weight;
      }
    }
  }

  const avgScore = scores.reduce((a, b) => a + b, 0) / scores.length;
  console.log(`\n  AVERAGE SCORE: ${avgScore.toFixed(2)}`);

  // Aggregate class breakdown
  console.log("\n  Aggregate per-class loss breakdown:");
  const classNames = ["Empty", "Settlement", "Port", "Ruin", "Forest", "Mountain"];
  for (let c = 0; c < NUM_CLASSES; c++) {
    const avgLoss = allClassBreakdown[c].loss / seedsCount;
    const avgWeight = allClassBreakdown[c].weight / seedsCount;
    console.log(`    ${classNames[c].padEnd(12)} avgLoss=${avgLoss.toFixed(4)}  avgWeight=${avgWeight.toFixed(3)}`);
  }

  // Compare with API scores
  const apiScores = analyses.map((a) => a.score).filter((s) => s != null);
  if (apiScores.length > 0) {
    const apiAvg = apiScores.reduce((a, b) => a + b, 0) / apiScores.length;
    console.log(`\n  API scores (actual R${args.round}): ${apiScores.map(s => s.toFixed(2)).join(", ")} avg=${apiAvg.toFixed(2)}`);
    console.log(`  Local test (simulated):     ${scores.map(s => s.toFixed(2)).join(", ")} avg=${avgScore.toFixed(2)}`);
    console.log(`  Delta: ${(avgScore - apiAvg).toFixed(2)} (positive = improvement)`);
  }

  // Prior-only baseline
  console.log("\n--- Prior-only baseline ---");
  const priorOnlyCounts = {};
  for (let si = 0; si < seedsCount; si++)
    priorOnlyCounts[si] = make3D(H, W, NUM_CLASSES, 0);
  const emptyCtx = buildContextPool(initialStates, priorOnlyCounts, H, W);
  const priorScores = [];
  for (let si = 0; si < seedsCount; si++) {
    const grid = initialStates[si].grid;
    const settlements = initialStates[si].settlements || [];
    const pred = layeredPredict(grid, settlements, priorOnlyCounts[si], emptyCtx, initialStates, si, H, W);
    const gt = analyses[si].ground_truth;
    const result = scoreOneSeed(pred, gt, H, W);
    priorScores.push(result.score);
  }
  const priorAvg = priorScores.reduce((a, b) => a + b, 0) / priorScores.length;
  console.log(`  Prior-only avg: ${priorAvg.toFixed(2)}`);
  console.log(`  Observation gain: +${(avgScore - priorAvg).toFixed(2)}`);

  // Uniform baseline
  const uniformScores = [];
  for (let si = 0; si < seedsCount; si++) {
    const gt = analyses[si].ground_truth;
    const uniformPred = Array.from({ length: H }, () =>
      Array.from({ length: W }, () => new Float64Array(NUM_CLASSES).fill(1 / NUM_CLASSES))
    );
    const result = scoreOneSeed(uniformPred, gt, H, W);
    uniformScores.push(result.score);
  }
  const uniformAvg = uniformScores.reduce((a, b) => a + b, 0) / uniformScores.length;
  console.log(`  Uniform avg: ${uniformAvg.toFixed(2)}`);

  console.log("\n" + "=".repeat(60));
}

main().catch((e) => {
  console.error("FATAL:", e.message);
  process.exit(1);
});
