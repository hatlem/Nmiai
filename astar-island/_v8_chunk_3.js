lements, y, x);
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
