#!/usr/bin/env node
/**
 * GT Calibration Pipeline for Astar Island
 *
 * Fetches ground truth from ALL completed rounds, builds a comprehensive
 * context-based lookup table, and validates predictions against held-out data.
 *
 * Usage:
 *   node calibrate_gt.js --token YOUR_TOKEN
 *   node calibrate_gt.js                  # uses cached data only
 *   node calibrate_gt.js --validate       # leave-one-round-out validation
 */

const fs = require("fs");
const path = require("path");
const https = require("https");

const API = "https://api.ainm.no/astar-island";
const CACHE_DIR = path.join(__dirname, "cache");
const NUM_CLASSES = 6;
const TERRAIN_TO_CLASS = { 10: 0, 11: 0, 0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5 };

function parseArgs() {
  const args = { token: null, validate: false, fetch: false };
  const argv = process.argv.slice(2);
  for (let i = 0; i < argv.length; i++) {
    if (argv[i] === "--token" && argv[i + 1]) args.token = argv[++i];
    else if (argv[i] === "--validate") args.validate = true;
    else if (argv[i] === "--fetch") args.fetch = true;
  }
  return args;
}

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

function ensureCache() {
  if (!fs.existsSync(CACHE_DIR)) fs.mkdirSync(CACHE_DIR, { recursive: true });
}

// ── Spatial helpers ──────────────────────────────────────────────────────────

function classifyCode(code) { return TERRAIN_TO_CLASS[code] ?? 0; }

function isCoastal(grid, H, W, y, x) {
  if (grid[y][x] === 10 || grid[y][x] === 5) return false;
  for (const [dy, dx] of [[-1,0],[1,0],[0,-1],[0,1]]) {
    const ny = y + dy, nx = x + dx;
    if (ny >= 0 && ny < H && nx >= 0 && nx < W && grid[ny][nx] === 10) return true;
  }
  return false;
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

function settDist(grid, H, W, y, x) {
  let minD = 999;
  for (let sy = 0; sy < H; sy++)
    for (let sx = 0; sx < W; sx++)
      if (grid[sy][sx] === 1 || grid[sy][sx] === 2) {
        const d = Math.abs(y - sy) + Math.abs(x - sx);
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

function contextKey(ic, food, coastal, sd, nSett) {
  const foodBin = Math.min(food, 3);
  const coastBin = coastal ? 1 : 0;
  const distBin = sd <= 3 ? "near" : sd <= 7 ? "mid" : sd <= 12 ? "far" : "remote";
  const settBin = Math.min(nSett, 3);
  return `${ic}_${foodBin}_${coastBin}_${distBin}_${settBin}`;
}

// ── Fetch GT data ────────────────────────────────────────────────────────────

async function fetchAllGT(token) {
  ensureCache();

  // Get all rounds
  const rounds = await httpGet(`${API}/my-rounds`, token);
  const completed = rounds.filter(r => r.status === "completed" && r.round_score);
  console.log(`Found ${completed.length} completed rounds with scores`);

  const allData = [];

  for (const round of completed) {
    console.log(`\nRound ${round.round_number} (score: ${round.round_score}, rank: ${round.rank})`);

    // Get initial states
    const cacheInitPath = path.join(CACHE_DIR, `r${round.round_number}_init.json`);
    let detail;
    if (fs.existsSync(cacheInitPath)) {
      detail = JSON.parse(fs.readFileSync(cacheInitPath, "utf8"));
      console.log("  Initial states: cached");
    } else {
      detail = await httpGet(`${API}/rounds/${round.id}`, token);
      fs.writeFileSync(cacheInitPath, JSON.stringify(detail));
      console.log("  Initial states: fetched");
    }

    // Get GT for each seed
    const seedsCount = detail.seeds_count || 5;
    for (let si = 0; si < seedsCount; si++) {
      const cacheGTPath = path.join(CACHE_DIR, `r${round.round_number}_gt_s${si}.json`);
      let analysis;
      if (fs.existsSync(cacheGTPath)) {
        analysis = JSON.parse(fs.readFileSync(cacheGTPath, "utf8"));
        console.log(`  Seed ${si}: cached`);
      } else {
        await new Promise(r => setTimeout(r, 200)); // Rate limit
        analysis = await httpGet(`${API}/analysis/${round.id}/${si}`, token);
        fs.writeFileSync(cacheGTPath, JSON.stringify(analysis));
        console.log(`  Seed ${si}: fetched (score: ${analysis.score})`);
      }

      allData.push({
        round: round.round_number,
        roundId: round.id,
        seed: si,
        H: analysis.height,
        W: analysis.width,
        gt: analysis.ground_truth,
        initial_grid: analysis.initial_grid,
        score: analysis.score,
        prediction: analysis.prediction,
      });
    }
  }

  return allData;
}

// ── Build calibrated lookup table ────────────────────────────────────────────

function buildLookupTable(allData, excludeRound = null) {
  const ctxStats = {};

  for (const data of allData) {
    if (excludeRound && data.round === excludeRound) continue;

    const { H, W, gt, initial_grid: ig } = data;

    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        const raw = ig[y][x];
        const ic = classifyCode(raw);
        const food = foodPotential(ig, H, W, y, x);
        const coastal = isCoastal(ig, H, W, y, x);
        const sd = settDist(ig, H, W, y, x);
        const nSett = neighborSettlements(ig, H, W, y, x);
        const key = contextKey(ic, food, coastal, sd, nSett);

        if (!ctxStats[key]) ctxStats[key] = { counts: new Float64Array(NUM_CLASSES), n: 0 };
        for (let c = 0; c < NUM_CLASSES; c++) ctxStats[key].counts[c] += gt[y][x][c];
        ctxStats[key].n++;
      }
    }
  }

  // Convert to distributions with Jeffreys smoothing
  const lookup = {};
  for (const [key, stats] of Object.entries(ctxStats)) {
    const dist = new Array(NUM_CLASSES);
    const total = stats.counts.reduce((a, b) => a + b, 0);
    const alpha = 0.01; // Very light smoothing
    const denom = total + NUM_CLASSES * alpha;
    for (let c = 0; c < NUM_CLASSES; c++) {
      dist[c] = (stats.counts[c] + alpha) / denom;
    }
    lookup[key] = { dist, n: stats.n };
  }

  return lookup;
}

// ── Score predictions against GT ─────────────────────────────────────────────

function scoreAgainstGT(pred, gt, H, W) {
  let totalWKL = 0, totalEntropy = 0;
  const klByClass = new Float64Array(NUM_CLASSES);
  const countByClass = new Int32Array(NUM_CLASSES);

  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      let entropy = 0, kl = 0;
      for (let c = 0; c < NUM_CLASSES; c++) {
        if (gt[y][x][c] > 0) {
          entropy -= gt[y][x][c] * Math.log(gt[y][x][c]);
          kl += gt[y][x][c] * Math.log(gt[y][x][c] / Math.max(pred[y][x][c], 1e-12));
        }
      }
      const wkl = entropy * kl;
      totalWKL += wkl;
      totalEntropy += entropy;

      // Find dominant initial class for this cell
      const initVal = gt[y][x].indexOf(Math.max(...gt[y][x]));
      klByClass[initVal] += wkl;
      countByClass[initVal]++;
    }
  }

  const weightedKL = totalWKL / totalEntropy;
  const score = 100 * Math.exp(-3 * weightedKL);
  return { score, weightedKL, totalWKL, totalEntropy, klByClass, countByClass };
}

// ── Generate predictions from lookup table ───────────────────────────────────

function predictFromLookup(lookup, ig, H, W) {
  const pred = Array.from({ length: H }, () =>
    Array.from({ length: W }, () => new Array(NUM_CLASSES).fill(0))
  );

  const MOUNTAIN_PRIOR = [0.002, 0.002, 0.002, 0.002, 0.002, 0.990];
  const OCEAN_PRIOR = [0.990, 0.002, 0.002, 0.002, 0.002, 0.002];

  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const raw = ig[y][x];

      if (raw === 5) { pred[y][x] = [...MOUNTAIN_PRIOR]; continue; }
      if (raw === 10) { pred[y][x] = [...OCEAN_PRIOR]; continue; }

      const ic = classifyCode(raw);
      const food = foodPotential(ig, H, W, y, x);
      const coastal = isCoastal(ig, H, W, y, x);
      const sd = settDist(ig, H, W, y, x);
      const nSett = neighborSettlements(ig, H, W, y, x);
      const key = contextKey(ic, food, coastal, sd, nSett);

      const entry = lookup[key];
      if (entry && entry.n >= 3) {
        pred[y][x] = [...entry.dist];
      } else {
        // Fallback: try broader context (drop nSett)
        const broadKey = contextKey(ic, food, coastal, sd, 0);
        const broadEntry = lookup[broadKey];
        if (broadEntry && broadEntry.n >= 3) {
          pred[y][x] = [...broadEntry.dist];
        } else {
          // Very broad fallback
          pred[y][x] = [0.5, 0.1, 0.05, 0.05, 0.25, 0.05];
        }
      }

      // Floor and normalize
      let sum = 0;
      for (let c = 0; c < NUM_CLASSES; c++) {
        pred[y][x][c] = Math.max(pred[y][x][c], 0.002);
        sum += pred[y][x][c];
      }
      for (let c = 0; c < NUM_CLASSES; c++) pred[y][x][c] /= sum;

      // Non-coastal: suppress port
      if (!coastal) {
        pred[y][x][2] = 0.002;
        sum = pred[y][x].reduce((a, b) => a + b, 0);
        for (let c = 0; c < NUM_CLASSES; c++) pred[y][x][c] /= sum;
      }
    }
  }

  return pred;
}

// ── Main ─────────────────────────────────────────────────────────────────────

async function main() {
  const args = parseArgs();

  let allData;
  const allDataPath = path.join(CACHE_DIR, "all_gt_data.json");

  if (args.token || args.fetch) {
    if (!args.token) { console.error("Need --token for fetching"); process.exit(1); }
    allData = await fetchAllGT(args.token);
    ensureCache();
    // Don't cache full GT data (too large), cache individual files instead
    console.log(`\nTotal: ${allData.length} seed-level GT datasets`);
  } else if (fs.existsSync(path.join(CACHE_DIR, "r1_gt_s0.json"))) {
    // Load from individual cache files
    console.log("Loading from cache...");
    allData = [];
    for (let r = 1; r <= 10; r++) {
      const initPath = path.join(CACHE_DIR, `r${r}_init.json`);
      if (!fs.existsSync(initPath)) continue;
      const detail = JSON.parse(fs.readFileSync(initPath, "utf8"));
      const seedsCount = detail.seeds_count || 5;
      for (let si = 0; si < seedsCount; si++) {
        const gtPath = path.join(CACHE_DIR, `r${r}_gt_s${si}.json`);
        if (!fs.existsSync(gtPath)) continue;
        const analysis = JSON.parse(fs.readFileSync(gtPath, "utf8"));
        allData.push({
          round: r, seed: si,
          H: analysis.height, W: analysis.width,
          gt: analysis.ground_truth,
          initial_grid: analysis.initial_grid,
          score: analysis.score,
          prediction: analysis.prediction,
        });
      }
    }
    console.log(`Loaded ${allData.length} seed-level datasets from cache`);
  } else {
    console.error("No cached data. Run with --token YOUR_TOKEN first.");
    process.exit(1);
  }

  if (allData.length === 0) { console.error("No data!"); process.exit(1); }

  // Build full lookup table
  console.log("\n=== Building GT-Calibrated Lookup Table ===");
  const lookup = buildLookupTable(allData);
  console.log(`Context bins: ${Object.keys(lookup).length}`);
  console.log(`Total cells: ${Object.values(lookup).reduce((s, v) => s + v.n, 0)}`);

  // Show top context bins by count
  const sorted = Object.entries(lookup).sort((a, b) => b[1].n - a[1].n);
  console.log("\nTop 20 context bins:");
  for (const [key, val] of sorted.slice(0, 20)) {
    const d = val.dist.map(v => (v * 100).toFixed(1) + "%").join(", ");
    console.log(`  ${key} (n=${val.n}): [${d}]`);
  }

  // Score lookup-only predictions against GT
  console.log("\n=== Scoring Lookup-Only Predictions ===");
  const rounds = [...new Set(allData.map(d => d.round))];

  for (const r of rounds) {
    const seeds = allData.filter(d => d.round === r);
    let totalScore = 0;
    for (const seed of seeds) {
      const pred = predictFromLookup(lookup, seed.initial_grid, seed.H, seed.W);
      const result = scoreAgainstGT(pred, seed.gt, seed.H, seed.W);
      totalScore += result.score;
    }
    const avgScore = totalScore / seeds.length;
    const actualScore = seeds[0].score ? seeds.reduce((s, d) => s + d.score, 0) / seeds.length : "N/A";
    console.log(`  Round ${r}: lookup=${avgScore.toFixed(1)} (actual submitted=${actualScore})`);
  }

  if (args.validate) {
    console.log("\n=== Leave-One-Round-Out Validation ===");
    for (const holdout of rounds) {
      const valLookup = buildLookupTable(allData, holdout);
      const seeds = allData.filter(d => d.round === holdout);
      let totalScore = 0;
      for (const seed of seeds) {
        const pred = predictFromLookup(valLookup, seed.initial_grid, seed.H, seed.W);
        const result = scoreAgainstGT(pred, seed.gt, seed.H, seed.W);
        totalScore += result.score;
      }
      const avgScore = totalScore / seeds.length;
      console.log(`  Holdout R${holdout}: ${avgScore.toFixed(1)} (using lookup from other rounds)`);
    }
  }

  // Export lookup for browser agent
  const exportPath = path.join(__dirname, "gt_lookup.json");
  const exportData = {};
  for (const [key, val] of Object.entries(lookup)) {
    if (val.n >= 2) exportData[key] = val.dist.map(v => +v.toFixed(6));
  }
  fs.writeFileSync(exportPath, JSON.stringify(exportData));
  console.log(`\nExported ${Object.keys(exportData).length} context bins to gt_lookup.json`);
  console.log(`File size: ${(fs.statSync(exportPath).size / 1024).toFixed(1)} KB`);
}

main().catch(e => { console.error(e); process.exit(1); });
