h.floor(VIEWPORT_MAX / 2), 0, this.H - VIEWPORT_MAX);
     const vw = Math.min(VIEWPORT_MAX, this.W - vx);
     const vh = Math.min(VIEWPORT_MAX, this.H - vy);
     const contained = new Set();
     for (const j of uncovered) {
      const [cx, cy] = coords[j];
      if (cx >= vx && cx < vx + vw && cy >= vy && cy < vy + vh) contained.add(j);
     }
     const dynScore = this._viewportImportance(seedIdx, vx, vy, vw, vh);
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

  _viewportImportance(seedIdx, x, y, w, h) {
   const imp = this.importanceMaps[seedIdx];
   let total = 0;
   for (let gy = y; gy < y + h && gy < this.H; gy++)
    for (let gx = x; gx < x + w && gx < this.W; gx++)
     total += imp[gy][gx];
   return total;
  }

  // CHANGE 7: Information gain = sum(importance / (1 + nObs)) for dynamic cells
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
   // Track per-cell observation counts (simplified)
   const simCounts = {};
   for (let si = 0; si < this.seedsCount; si++)
    simCounts[si] = Array.from({ length: this.H }, () => new Float64Array(this.W));

   const recordQuery = (si, x, y, w, h) => {
    for (let gy = y; gy < y + h && gy < this.H; gy++)
     for (let gx = x; gx < x + w && gx < this.W; gx++)
      simCounts[si][gy][gx] += 1;
   };

   // Phase 1 (10 queries): 1 best viewport per seed, then 2nd best
   const phase1Budget = Math.min(this.seedsCount * 2, Math.floor(this.budget * 0.2));
   for (let pass = 0; pass < 2 && plan.length < phase1Budget; pass++) {
    for (let si = 0; si < this.seedsCount && plan.length < phase1Budget; si++) {
     const vps = this.seedViewports[si];
     const vpIdx = Math.min(pass, vps.length - 1);
     const vp = vps[vpIdx];
     plan.push([si, ...vp]);
     recordQuery(si, vp[0], vp[1], vp[2], vp[3]);
    }
   }

   // Phase 2 (remaining): Adaptive — highest information gain
   // Build candidate viewports: settlement viewports + expansion viewports
   const candidates = [];
   for (let si = 0; si < this.seedsCount; si++) {
    for (const vp of this.seedViewports[si])
     candidates.push({ si, vp });
    // Also add shifted viewports to cover expansion zones
    for (const vp of this.seedViewports[si]) {
     const [vx, vy, vw, vh] = vp;
     // Shift in 4 directions by half viewport
     for (const [dy, dx] of [[-7,0],[7,0],[0,-7],[0,7]]) {
      const nx = clamp(vx + dx, 0, this.W - VIEWPORT_MAX);
      const ny = clamp(vy + dy, 0, this.H - VIEWPORT_MAX);
      const nw = Math.min(VIEWPORT_MAX, this.W - nx);
      const nh = Math.min(VIEWPORT_MAX, this.H - ny);
      candidates.push({ si, vp: [nx, ny, nw, nh] });
     }
    }
   }
   // Deduplicate
   const seen = new Set();
   const dedupCandidates = [];
   for (const c of candidates) {
    const key = `${c.si}_${c.vp.join("_")}`;
    if (!seen.has(key)) {
     seen.add(key);
     dedupCandidates.push(c);
    }
   }

   while (plan.length < this.budget) {
    let bestGain = -1, bestQuery = null;
    for (const c of dedupCandidates) {
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

 // CHANGE 3: Layered prediction — replaces geometric ensemble
 function layeredPredict(
  grid, settlements, counts, contextMap, initialStates, seedIdx, H, W
 ) {
  const pred = make3D(H, W, NUM_CLASSES, 0);
  const coastal = coastalMask(grid, H, W);
  const observedMask = Array.from({ length: H }, () => new Uint8Array(W));

  for (let y = 0; y < H; y++) {
   for (let x = 0; x < W; x++) {
    const raw = grid[y][x];

    // Hard constraints for static terrain
    if (raw === 5) { pred[y][x] = mountainPrior(); continue; }
    if (raw === 10) { pred[y][x] = oceanPrior(); continue; }

    const ic = classifyCode(raw);
    const isOcean = false;
    const food = foodPotential(grid, H, W, y, x);
    const sd = settDist(grid, H, W, settlements, y, x);
    const isCoast = !!coastal[y][x];
    const nSett = neighborSettlements(grid, H, W, y, x);
    const nObs = counts[y][x].reduce((a, b) => a + b, 0);

    // Get informative prior (CHANGE 2)
    const prior = getCellPrior(ic, sd, food, isCoast, nSett);
    const strength = getPriorStrength(ic, sd, nObs);

    // Per-cell floors (CHANGE 1)
    const cellFloors = getCellFloors(ic, isOcean, sd);

    if (nObs >= 5) {
     // Layer 1: Pure KT with informative Dirichlet prior (data dominates)
     observedMask[y][x] = 1;
     const denom = nObs + strength;
     for (let c = 0; c < NUM_CLASSES; c++)
      pred[y][x][c] = (counts[y][x][c] + prior[c] * strength) / denom;
     floorNorm(pred[y][x], cellFloors);

    } else if (nObs >= 1) {
     // Layer 2: 60% KT + 40% contextual pooling (CHANGE 3)
     observedMask[y][x] = 1;
     const denom = nObs + strength;
     const kt = new Float64Array(NUM_CLASSES);
     for (let c = 0; c < NUM_CLASSES; c++)
      kt[c] = (counts[y][x][c] + prior[c] * strength) / denom;

     const ctxPred = contextPoolCell(contextMap, grid, H, W, settlements, y, x);
     if (ctxPred) {
      for (let c = 0; c < NUM_CLASSES; c++)
       pred[y][x][c] = 0.6 * kt[c] + 0.4 * ctxPred[c];
     } else {
      for (let c = 0; c < NUM_CLASSES; c++)
       pred[y][x][c] = kt[c];
     }
     floorNorm(pred[y][x], cellFloors);

    } else {
     // Layer 3: Pure contextual pooling or calibrated prior (CHANGE 3)
     const ctxPred = contextPoolCell(contextMap, grid, H, W, settlements, y, x);
     if (ctxPred) {
      for (let c = 0; c < NUM_CLASSES; c++)
       pred[y][x][c] = ctxPred[c];
     } else {
      for (let c = 0; c < NUM_CLASSES; c++)
       pred[y][x][c] = prior[c];
     }

     // Non-coastal: suppress port
     if (!isCoast) pred[y][x][2] = cellFloors[2];
     floorNorm(pred[y][x], cellFloors);
    }
   }
  }

  // CHANGE 5: Belief propagation for unobserved cells
  const result = beliefPropagation(pred, observedMask, grid, H, W);

  // Final: apply hard constraints + floors
  for (let y = 0; y < H; y++) {
   for (let x = 0; x < W; x++) {
    const raw = grid[y][x];
    if (raw === 5) { result[y][x] = mountainPrior(); continue; }
    if (raw === 10) { result[y][x] = oceanPrior(); continue; }
    const ic = classifyCode(raw);
    const sd = settDist(grid, H, W, sett