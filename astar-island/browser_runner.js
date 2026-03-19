/**
 * Astar Island v4 — Browser Runner
 *
 * Paste this into the browser console on app.ainm.no to run the full agent.
 * Uses the browser's httpOnly cookies for authentication.
 *
 * Usage: Copy-paste into Chrome DevTools console, or inject via claude-in-chrome.
 */

(async function AstarIslandAgent() {
  const API = 'https://api.ainm.no';
  const NC = 6, FLR = 0.01, MAX_TC = 16;
  const T2C = {10:0, 11:0, 0:0, 1:1, 2:2, 3:3, 4:4, 5:5};

  const log = (msg) => console.log('[ASTAR] ' + msg);

  // ── API helpers ──
  async function apiGet(path) {
    const r = await fetch(API + path, {credentials:'include'});
    if (!r.ok) throw new Error(`GET ${path}: ${r.status}`);
    return r.json();
  }

  async function apiPost(path, body) {
    for (let attempt = 0; attempt < 3; attempt++) {
      const r = await fetch(API + path, {
        method: 'POST', credentials: 'include',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(body),
      });
      if (r.ok) return r.json();
      if (r.status === 429) {
        log(`Rate limited on ${path}, waiting ${1.5*(attempt+1)}s...`);
        await new Promise(r => setTimeout(r, 1500 * (attempt + 1)));
        continue;
      }
      throw new Error(`POST ${path}: ${r.status}`);
    }
    throw new Error(`Failed after 3 attempts: ${path}`);
  }

  // ── Map helpers ──
  function isOcean(grid, x, y, W, H) {
    return x<0||y<0||x>=W||y>=H||grid[y][x]===10;
  }
  function isCoastal(grid, x, y, W, H) {
    if (isOcean(grid,x,y,W,H)) return false;
    return [[-1,0],[1,0],[0,-1],[0,1]].some(([dx,dy]) => isOcean(grid,x+dx,y+dy,W,H));
  }
  function foodPotential(grid, x, y, W, H) {
    let f = 0;
    for (let dy=-1; dy<=1; dy++) for (let dx=-1; dx<=1; dx++) {
      if (dx===0&&dy===0) continue;
      const nx=x+dx, ny=y+dy;
      if (nx>=0&&ny>=0&&nx<W&&ny<H&&grid[ny][nx]===4) f++;
    }
    return f;
  }
  function settDist(x, y, setts) {
    let m = 999;
    for (const s of setts) m = Math.min(m, Math.abs(x-s.x)+Math.abs(y-s.y));
    return m;
  }
  function neighborSettlements(grid, x, y, W, H) {
    let n = 0;
    for (let dy=-2; dy<=2; dy++) for (let dx=-2; dx<=2; dx++) {
      if (dx===0&&dy===0) continue;
      const nx=x+dx, ny=y+dy;
      if (nx>=0&&ny>=0&&nx<W&&ny<H&&(grid[ny][nx]===1||grid[ny][nx]===2)) n++;
    }
    return Math.min(n, 4);
  }

  // ── Tiling ──
  function tilingOffsets(size, win) {
    if (size <= win) return [0];
    const offsets = [];
    for (let i = 0; i < size - win + 1; i += win) offsets.push(i);
    if (offsets[offsets.length-1] !== size - win) offsets.push(size - win);
    return offsets;
  }

  // ── Query planning (v4: 3-phase) ──
  function planQueries(W, H, seedsCount, budget, initialStates) {
    const xOff = tilingOffsets(W, 15);
    const yOff = tilingOffsets(H, 15);
    const tiles = [];
    for (const y of yOff) for (const x of xOff) {
      tiles.push([x, y, Math.min(15, W-x), Math.min(15, H-y)]);
    }

    // Rank seeds by settlement count
    const ranked = initialStates.map((s,i) => [s.settlements.length, i])
      .sort((a,b) => b[0]-a[0]).map(([_,i]) => i);

    const plan = [];

    // Phase 1: Full coverage of primary seed + settlement hotspots on others
    const primary = ranked[0];
    for (const [x,y,w,h] of tiles) plan.push([primary, x, y, w, h]);

    // Settlement viewports for other seeds (2-3 each)
    const others = ranked.slice(1);
    const perOther = Math.floor(11 / others.length);
    for (const si of others) {
      const setts = initialStates[si].settlements;
      // Score tiles by settlement content
      const scored = tiles.map(([x,y,w,h]) => {
        const count = setts.filter(s => s.x>=x&&s.x<x+w&&s.y>=y&&s.y<y+h).length;
        return {score: count, x, y, w, h};
      }).sort((a,b) => b.score-a.score);
      for (let i = 0; i < Math.min(perOther, scored.length); i++) {
        const {x,y,w,h} = scored[i];
        plan.push([si, x, y, w, h]);
      }
    }

    // Phase 2: Fill coverage gaps
    const covered = {};
    for (const [si,x,y,w,h] of plan) {
      if (!covered[si]) covered[si] = new Set();
      covered[si].add(x+','+y);
    }
    const remaining = [];
    for (const si of others) {
      for (const [x,y,w,h] of tiles) {
        if (!covered[si] || !covered[si].has(x+','+y)) {
          remaining.push([si, x, y, w, h]);
        }
      }
    }
    for (const q of remaining.slice(0, 20)) plan.push(q);

    // Phase 3: Re-observe highest-value viewports
    const phase3Budget = budget - plan.length;
    if (phase3Budget > 0) {
      const allVps = [];
      for (let si = 0; si < seedsCount; si++) {
        for (const [x,y,w,h] of tiles) {
          const setts = initialStates[si].settlements;
          const score = setts.filter(s => s.x>=x&&s.x<x+w&&s.y>=y&&s.y<y+h).length;
          allVps.push({score, si, x, y, w, h});
        }
      }
      allVps.sort((a,b) => b.score-a.score);
      const perSeed = {};
      let added = 0;
      for (const {si,x,y,w,h} of allVps) {
        if (added >= phase3Budget) break;
        perSeed[si] = (perSeed[si]||0);
        if (perSeed[si] >= 3) continue;
        plan.push([si, x, y, w, h]);
        perSeed[si]++;
        added++;
      }
    }

    return plan.slice(0, budget);
  }

  // ── Prediction: Jeffreys + transition + heuristics ──
  function buildPrediction(grid, setts, counts, obsTotals, W, H, globalTrans, params) {
    const pred = [];
    for (let y = 0; y < H; y++) {
      pred[y] = [];
      for (let x = 0; x < W; x++) {
        const cell = grid[y][x];
        const cls = T2C[cell] ?? 0;
        const nObs = obsTotals[y][x];

        if (nObs > 0) {
          // Jeffreys estimator: (count + 0.5) / (n + 3.0)
          const p = [];
          for (let c = 0; c < NC; c++) {
            p.push((counts[y][x][c] + 0.5) / (nObs + NC * 0.5));
          }

          // Blend with prior at low n
          if (nObs < 3 && globalTrans[cls]) {
            const pw = 0.4 / nObs;
            for (let c = 0; c < NC; c++) {
              p[c] = (1 - pw) * p[c] + pw * globalTrans[cls][c];
            }
          }
          pred[y][x] = p;
          continue;
        }

        // Unobserved cells
        if (cell === 10) { pred[y][x] = [0.985,0.003,0.003,0.003,0.003,0.003]; continue; }
        if (cell === 5) { pred[y][x] = [0.003,0.003,0.003,0.003,0.003,0.985]; continue; }

        // Start from transition matrix
        let d = globalTrans[cls] ? [...globalTrans[cls]] : [1/6,1/6,1/6,1/6,1/6,1/6];
        const food = foodPotential(grid, x, y, W, H);
        const coast = isCoastal(grid, x, y, W, H);
        const sd = settDist(x, y, setts);
        const ns = neighborSettlements(grid, x, y, W, H);
        const fa = params.faction_aggression || 0;
        const ws = params.winter_severity || 0;
        const ta = params.trade_activity || 0;
        const fg = params.forest_growth_rate || 0;

        if (cls === 1 || cls === 2) {
          if (food >= 2) { d[1] += 0.25; d[3] += 0.08; }
          else if (food >= 1) { d[1] += 0.15; d[3] += 0.20; }
          else { d[1] += 0.06; d[3] += 0.28; }
          if (cls === 2 || coast) d[2] += 0.10;
          d[3] += 0.18 * fa + 0.15 * ws;
          d[1] -= 0.08 * (fa + ws);
          if (coast) d[2] += 0.12 * ta;
        }
        if (cls === 0 && sd <= 3) { d[1] += 0.08; d[3] += 0.04; }
        if (cls === 4) { d[4] += 0.3; d[4] += 0.08 * fg; }
        if (cls === 0 && food >= 2) d[4] += 0.08 + 0.12 * fg;
        if (cls === 3) {
          if (sd <= 4) d[1] += 0.10;
          d[4] += 0.06 + 0.10 * fg;
          d[0] += 0.05;
        }
        if (cell !== 5) d[5] *= 0.2;
        if (!coast) d[2] = FLR;

        // Clamp negatives
        for (let c = 0; c < NC; c++) d[c] = Math.max(d[c], 0);
        pred[y][x] = d;
      }
    }

    // Floor and normalize
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
      let p = pred[y][x];
      for (let c = 0; c < NC; c++) p[c] = Math.max(p[c], FLR);
      const s = p.reduce((a,b) => a+b, 0);
      for (let c = 0; c < NC; c++) p[c] /= s;
    }
    return pred;
  }

  // ── Transition matrix (cross-seed) ──
  function buildGlobalTransition(initialStates, allCounts) {
    const trans = Array.from({length:NC}, () => Array(NC).fill(0.5));
    for (let c = 0; c < NC; c++) trans[c][c] = 5.0;

    for (let si = 0; si < initialStates.length; si++) {
      const grid = initialStates[si].grid;
      const counts = allCounts[si];
      if (!counts) continue;
      const H = grid.length, W = grid[0].length;
      for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
        const cls = T2C[grid[y][x]] ?? 0;
        for (let c = 0; c < NC; c++) trans[cls][c] += (counts[y]?.[x]?.[c] || 0);
      }
    }
    // Normalize
    for (let c = 0; c < NC; c++) {
      const s = trans[c].reduce((a,b) => a+b, 0);
      for (let j = 0; j < NC; j++) trans[c][j] /= s;
    }
    return trans;
  }

  // ── Hidden param estimation ──
  function estimateParams(trans) {
    const s2r = trans[1][3]; // settlement -> ruin
    const s2p = trans[1][2]; // settlement -> port
    const p2p = trans[2][2]; // port survival
    const e2f = trans[0][4]; // empty -> forest
    const e2s = trans[0][1]; // empty -> settlement
    return {
      faction_aggression: Math.min(s2r * 2.0, 1),
      winter_severity: Math.min(s2r * 1.5, 1),
      trade_activity: Math.min((s2p + p2p) * 1.5, 1),
      forest_growth_rate: Math.min(e2f * 5.0, 1),
      expansion_rate: Math.min(e2s * 5.0, 1),
    };
  }

  // ══════════════ MAIN ══════════════

  log('Loading round...');
  const rounds = await apiGet('/astar-island/rounds');
  const active = rounds.find(r => r.status === 'active');
  if (!active) { log('No active round'); return; }
  const roundId = active.id;
  const detail = await apiGet('/astar-island/rounds/' + roundId);
  const W = detail.map_width, H = detail.map_height;
  const seedsCount = detail.seeds_count;
  const IS = detail.initial_states;
  log(`Round ${active.round_number}: ${W}x${H}, ${seedsCount} seeds`);

  const budgetInfo = await apiGet('/astar-island/budget');
  const remaining = budgetInfo.queries_max - budgetInfo.queries_used;
  log(`Budget: ${budgetInfo.queries_used}/${budgetInfo.queries_max} (${remaining} left)`);

  // Initialize observation storage
  const allCounts = {};
  const obsTotals = {};
  for (let si = 0; si < seedsCount; si++) {
    allCounts[si] = Array.from({length:H}, () => Array.from({length:W}, () => Array(NC).fill(0)));
    obsTotals[si] = Array.from({length:H}, () => Array(W).fill(0));
  }

  // Execute queries
  if (remaining > 0) {
    const plan = planQueries(W, H, seedsCount, remaining, IS);
    log(`Executing ${plan.length} queries...`);

    for (let qi = 0; qi < plan.length; qi++) {
      const [si, vx, vy, vw, vh] = plan[qi];
      try {
        const result = await apiPost('/astar-island/simulate', {
          round_id: roundId, seed_index: si,
          viewport_x: vx, viewport_y: vy, viewport_w: vw, viewport_h: vh,
        });

        // Store observations
        for (let ry = 0; ry < result.grid.length; ry++) {
          for (let rx = 0; rx < result.grid[ry].length; rx++) {
            const y = vy + ry, x = vx + rx;
            if (y < H && x < W) {
              const cls = T2C[result.grid[ry][rx]] ?? 0;
              allCounts[si][y][x][cls]++;
              obsTotals[si][y][x]++;
            }
          }
        }

        log(`  Q${qi+1}/${plan.length}: seed=${si} (${vx},${vy}) [${result.queries_used}/${result.queries_max}]`);
        if (result.queries_used >= result.queries_max) { log('  Budget exhausted!'); break; }
        await new Promise(r => setTimeout(r, 60));
      } catch (e) {
        log(`  Q${qi+1} FAILED: ${e.message}`);
      }
    }
  } else {
    log('No queries remaining.');
  }

  // Build global transition matrix
  log('Building transition model...');
  const globalTrans = buildGlobalTransition(IS, allCounts);
  const params = estimateParams(globalTrans);
  log('Hidden params: ' + JSON.stringify(params));

  // Build and submit predictions
  log('Building and submitting predictions...');
  for (let si = 0; si < seedsCount; si++) {
    const pred = buildPrediction(
      IS[si].grid, IS[si].settlements, allCounts[si], obsTotals[si],
      W, H, globalTrans, params
    );
    try {
      const resp = await apiPost('/astar-island/submit', {
        round_id: roundId, seed_index: si, prediction: pred,
      });
      log(`  Seed ${si}: ${resp.status || JSON.stringify(resp)}`);
    } catch (e) {
      log(`  Seed ${si} SUBMIT FAILED: ${e.message}`);
    }
    await new Promise(r => setTimeout(r, 500));
  }

  log('Done!');
  return 'Agent complete';
})();
