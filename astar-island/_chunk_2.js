 dist[2] = isCoast ? portProb : PROB_FLOOR;
 dist[3] = ruinProb;
 dist[0] = 0.03;
 dist[4] = 0.03;
 dist[5] = PROB_FLOOR;
 } else if (ic === 0 && sd <= 4) {
 const expProb = 0.05 + 0.03 * (4 - sd);
 dist[1] = expProb;
 dist[3] = expProb * 0.3;
 dist[0] = Math.max(0.3, 1.0 - expProb * 1.5 - dist[4]);
 } else if (ic === 4 && sd <= 3) {
 dist[1] = 0.05;
 dist[0] = 0.08;
 dist[4] = 0.82;
 } else if (ic === 3) {
 if (sd <= 3) dist[1] = 0.15;
 dist[4] = 0.10 + 0.08 * forestGrowth;
 dist[3] = Math.max(0.3, 1.0 - dist[1] - dist[4] - 0.1);
 dist[0] = 0.08;
 }
 pred[y][x] = dist;
 floorNorm(pred[y][x]);
 }
 }
 return pred;
 }
 function contextualPoolPredict(initialStates, allCounts, seedIdx, H, W) {
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
 const grid = initialStates[seedIdx].grid;
 const settlements = initialStates[seedIdx].settlements || [];
 const pred = make3D(H, W, NUM_CLASSES, 0);
 const ktAlpha = 0.5;
 for (let y = 0; y < H; y++) {
 for (let x = 0; x < W; x++) {
 const raw = grid[y][x];
 if (raw === 5) { pred[y][x] = mountainPrior(); continue; }
 if (raw === 10) { pred[y][x] = oceanPrior(); continue; }
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
 const denom = total + NUM_CLASSES * ktAlpha;
 for (let c = 0; c < NUM_CLASSES; c++)
 pred[y][x][c] = (pooled[c] + ktAlpha) / denom;
 floorNorm(pred[y][x]);
 continue;
 }
 }
 pred[y][x] = calibratedPrior(ic);
 }
 }
 return pred;
 }
 function geometricEnsemble(preds, weights, H, W) {
 if (preds.length === 0) return make3D(H, W, NUM_CLASSES, 1 / NUM_CLASSES);
 const totalW = weights.reduce((a, b) => a + b, 0);
 const result = make3D(H, W, NUM_CLASSES, 0);
 for (let y = 0; y < H; y++) {
 for (let x = 0; x < W; x++) {
 for (let c = 0; c < NUM_CLASSES; c++) {
 let logSum = 0;
 for (let i = 0; i < preds.length; i++)
 logSum += (weights[i] / totalW) * Math.log(preds[i][y][x][c] + 1e-12);
 result[y][x][c] = Math.exp(logSum);
 }
 floorNorm(result[y][x]);
 }
 }
 return result;
 }
 function applyTemperature(pred, H, W, T) {
 for (let y = 0; y < H; y++) {
 for (let x = 0; x < W; x++) {
 let maxLog = -Infinity;
 const logP = new Float64Array(NUM_CLASSES);
 for (let c = 0; c < NUM_CLASSES; c++) {
 logP[c] = Math.log(pred[y][x][c] + 1e-12) / T;
 if (logP[c] > maxLog) maxLog = logP[c];
 }
 let sumExp = 0;
 for (let c = 0; c < NUM_CLASSES; c++) {
 pred[y][x][c] = Math.exp(logP[c] - maxLog);
 sumExp += pred[y][x][c];
 }
 for (let c = 0; c < NUM_CLASSES; c++) pred[y][x][c] /= sumExp;
 floorNorm(pred[y][x]);
 }
 }
 return pred;
 }
 function applyConstraints(pred, initialGrid, H, W) {
 const coastal = coastalMask(initialGrid, H, W);
 for (let y = 0; y < H; y++) {
 for (let x = 0; x < W; x++) {
 const raw = initialGrid[y][x];
 if (raw === 5) {
 pred[y][x] = mountainPrior();
 continue;
 }
 if (raw === 10) {
 pred[y][x] = oceanPrior();
 continue;
 }
 if (!coastal[y][x]) {
 pred[y][x][2] = PROB_FLOOR;
 }
 floorNorm(pred[y][x]);
 }
 }
 return pred;
 }
 async function runPipeline(roundId) {
 log(`Pipeline starting for round ${roundId}`);
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
 if (queriesLeft <= 0) {
 log("No queries left — skipping observation phase, using priors only.");
 }
 const allCounts = {};
 for (let si = 0; si < seedsCount; si++)
 allCounts[si] = make3D(H, W, NUM_CLASSES, 0);
 const allSettlementData = {};
 for (let si = 0; si < seedsCount; si++) allSettlementData[si] = [];
 if (queriesLeft > 0) {
 const optimizer = new QueryOptimizer(W, H, seedsCount, queriesLeft, initialStates);
 const plan = optimizer.planQueries();
 log(`Query plan: ${plan.length} queries across ${seedsCount} seeds`);
 let queriesExecuted = 0;
 for (let qi = 0; qi < plan.length; qi++) {
 const [si, vx, vy, vw, vh] = plan[qi];
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
 if (result.settlements && result.settlements.length > 0)
 allSettlementData[seedIdx].push({ settlements: result.settlements });
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
 log("Building predictions...");
 const predictions = {};
 const globalCounts = make3D(H, W, NUM_CLASSES, 0);
 const refGrid = initialStates[0].grid;
 for (let si = 0; si < seedsCount; si++) {
 for (let y = 0; y < H; y++)
 for (let x = 0; x < W; x++)
 for (let c = 0; c < NUM_CLASSES; c++)
 globalCounts[y][x][c] += allCounts[si][y][x][c];
 }
 let inferredParams;
 try {
 inferredParams = inferParams(refGrid, globalCounts, H, W);
 log(`Inferred params: winter=${inferredParams.winter_severity.toFixed(2)}, ` +
 `aggression=${inferredParams.faction_aggression.toFixed(2)}, ` +
 `trade=${inferredParams.trade_activity.toFixed(2)}`);
 } catch (e) {
 log(`Param inference failed: ${e.message}, using defaults`);
 inferredParams = { ...DEFAULT_PARAMS };
 }
 for (let si = 0; si < seedsCount; si++) {
 const t1 = performance.now();
 const grid = initialStates[si].grid;
 const settlements = initialStates[si].settlements || [];
 const counts = allCounts[si];
 const agentPreds = [];
 const agentWeights = [];
 try {
 agentPreds.push(statisticalPredict(grid, counts, H, W));
 agentWeights.push(2.5);
 } catch (e) { log(` Seed ${si}: Statistical agent failed: ${e.message}`); }
 try {
 agentPreds.push(contextualPoolPredict(initialStates, allCounts, si, H, W));
 agentWeights.push(2.0);
 } catch (e) { log(` Seed ${si}: Contextual pooling failed: ${e.message}`); }
 try {
 agentPreds.push(transitionPredict(grid, counts, H, W));
 agentWeights.push(0.8);
 } catch (e) { log(` Seed ${si}: Transition agent failed: ${e.message}`); }
 try {
 agentPreds.push(heuristicPredict(grid, settlements, inferredParams, H, W));
 agentWeights.push(1.0);
 } catch (e) { log(` Seed ${si}: Heuristic agent failed: ${e.message}`); }
 try {
 const mcT0 = performance.now();
 const sim = new NorseSim(grid, settlements, inferredParams, H, W);
 const mcPred = sim.runMonteCarlo(MC_RUNS);
 const mcTime = performance.now() - mcT0;
 agentPreds.push(mcPred);
 agentWeights.push(1.0);
 log(` Seed ${si}: MC sim (${MC_RUNS} runs) in ${mcTime.toFixed(0)}ms`);
 } catch (e) {
 log(` Seed ${si}: MC sim failed (falling back to stats-only): ${e.message}`);
 }
 let combined = geometricEnsemble(agentPreds, agentWeights, H, W);
 const ktAlpha = 0.5;
 for (let y = 0; y < H; y++) {
 for (let x = 0; x < W; x++) {
 const nObs = counts[y][x].reduce((a, b) => a + b, 0);
 if (nObs > 0) {
 const ktW = nObs / (nObs + 2.0);
 const denom = nObs + NUM_CLASSES * ktAlpha;
 for (let c = 0; c < NUM_CLASSES; c++) {
 const kt = (counts[y][x][c] + ktAlpha) / denom;
 combined[y][x][c] = ktW * kt + (1 - ktW) * combined[y][x][c];
 }
 floorNorm(combined[y][x]);
 }
 }
 }
 combined = applyTemperature(combined, H, W, TEMPERATURE);
 combined = applyConstraints(combined, grid, H, W);
 predictions[si] = combined;
 const dt = performance.now() - t1;
 let obsCount = 0;
 for (let y = 0; y < H; y++)
 for (let x = 0; x < W; x++)
 if (counts[y][x].reduce((a, b) => a + b, 0) > 0) obsCount++;
 const obsPct = (100 * obsCount / (H * W)).toFixed(0);
 log(` Seed ${si}: ${agentPreds.length} agents, ${obsPct}% observed, ${dt.toFixed(0)}ms`);
 }
 log("Submitting predictions...");
 for (let si = 0; si < seedsCount; si++) {
 try {
 await sleep(API_DELAY);
 const predArray = [];
 for (let y = 0; y < H; y++) {
 const row = [];
 for (let x = 0; x < W; x++) {
 row.push(Array.from(predictions[si][y][x]));
 }
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
 log(`Pipeline complete in ${totalTime}s`);
 return true;
 }
 const completedRounds = new Set();
 let pollTimer = null;
 let running = false;
 async function checkAndRun() {
 if (running) { log("Pipeline already running, skipping poll"); return; }
 try {
 try {
 const myRounds = await apiFetch("/my-rounds");
 for (const r of myRounds) {
 if (r.queries_used >= r.queries_max || r.seeds_submitted >= (r.seeds_count || 5)) {
 completedRounds.add(r.id);
 }
 }
 } catch (e) {
 }
 const rounds = await apiFetch("/rounds");
 const active = rounds.filter((r) => r.status === "active" && !completedRounds.has(r.id));
 if (active.length === 0) {
 log("No new active rounds.");
 return;
 }
 for (const round of active) {
 log(`New active round detected: #${round.round_number} (${round.id})`);