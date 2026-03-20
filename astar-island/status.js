#!/usr/bin/env node
/**
 * Astar Island status dashboard. Run anytime to see where we stand.
 * Usage: TOKEN=$(cat ../.env.ainm | cut -d= -f2) node status.js
 */
const https = require('https');
const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');

const API = 'https://api.ainm.no/astar-island';
const TOKEN = process.env.TOKEN || process.env.AINM_TOKEN;
if (!TOKEN) { console.error('Set TOKEN env var'); process.exit(1); }

function get(p) {
  return new Promise((res, rej) => {
    const url = new URL(p.startsWith('http') ? p : API + p);
    https.get(url, {headers:{Authorization:`Bearer ${TOKEN}`}}, r => {
      let d = ''; r.on('data', c => d += c);
      r.on('end', () => { try { res(JSON.parse(d)); } catch { rej(new Error(d.slice(0,200))); } });
    }).on('error', rej);
  });
}

function bar(val, max, width = 30) {
  const filled = Math.round((val / max) * width);
  return '\u2588'.repeat(Math.min(filled, width)) + '\u2591'.repeat(Math.max(0, width - filled));
}

(async () => {
  const [rounds, lb] = await Promise.all([get('/my-rounds'), get('/leaderboard')]);

  console.log('\n\u2554' + '\u2550'.repeat(58) + '\u2557');
  console.log('\u2551  ASTAR ISLAND \u2014 STATUS DASHBOARD                          \u2551');
  console.log('\u2551  Team: Human-Like     ' + new Date().toISOString().slice(0,19) + '       \u2551');
  console.log('\u255a' + '\u2550'.repeat(58) + '\u255d\n');

  // Rounds
  const sorted = rounds.sort((a, b) => a.round_number - b.round_number);
  let bestWeighted = 0, bestRound = 0, bestRaw = 0;

  console.log('\u2500\u2500 ROUNDS ' + '\u2500'.repeat(50));
  console.log('Round  Status      Raw     Rank    Weight  Weighted  Agent');
  console.log('\u2500'.repeat(70));
  for (const r of sorted) {
    const raw = r.round_score ? r.round_score.toFixed(1) : '-';
    const rank = r.rank ? `#${r.rank}` : '-';
    const weight = r.round_weight ? r.round_weight.toFixed(3) : '-';
    const weighted = r.round_score && r.round_weight ? (r.round_score * r.round_weight).toFixed(1) : '-';
    const status = r.status.padEnd(10);
    if (r.round_score && r.round_weight && r.round_score * r.round_weight > bestWeighted) {
      bestWeighted = r.round_score * r.round_weight;
      bestRound = r.round_number;
    }
    if (r.round_score && r.round_score > bestRaw) bestRaw = r.round_score;
    // Guess which agent ran
    let agent = '-';
    if (r.round_number <= 3) agent = 'v7-browser';
    else if (r.round_number <= 5) agent = 'v8-browser';
    else if (r.round_number === 6) agent = 'v8+lookup';
    else agent = 'v3-node';
    console.log(`R${String(r.round_number).padEnd(4)} ${status}  ${raw.padStart(6)}  ${rank.padStart(6)}  ${String(weight).padStart(7)}  ${String(weighted).padStart(8)}  ${agent}`);
  }

  // Score progression
  const scored = sorted.filter(r => r.round_score);
  if (scored.length > 0) {
    console.log('\n\u2500\u2500 SCORE PROGRESSION \u2500'.padEnd(58, '\u2500'));
    const maxScore = Math.max(...scored.map(r => r.round_score));
    for (const r of scored) {
      const pct = r.round_score;
      const delta = scored.indexOf(r) > 0
        ? (pct - scored[scored.indexOf(r)-1].round_score).toFixed(1)
        : '-';
      const deltaStr = delta !== '-' ? (parseFloat(delta) >= 0 ? '+'+delta : delta) : '';
      console.log(`R${r.round_number}  ${bar(pct, 100)} ${pct.toFixed(1).padStart(5)}  ${deltaStr}`);
    }
  }

  // Leaderboard
  const us = lb.find(t => t.team_name === 'Human-Like');
  console.log('\n\u2500\u2500 LEADERBOARD \u2500'.padEnd(58, '\u2500'));
  console.log(`Our rank:     #${us ? us.rank : '?'} of ${lb.length}`);
  console.log(`Our weighted: ${us ? us.weighted_score.toFixed(1) : '?'}`);
  console.log(`Best raw:     ${bestRaw.toFixed(1)} (R${bestRound})`);
  console.log(`Best weighted: ${bestWeighted.toFixed(1)} (R${bestRound})\n`);

  const ourScore = us ? us.weighted_score : 0;
  console.log('Rank  Team                      Weighted  Gap     Raw~');
  console.log('\u2500'.repeat(65));
  for (const t of lb.slice(0, 20)) {
    const gap = t.weighted_score - ourScore;
    const gapStr = gap > 0 ? `+${gap.toFixed(1)}` : gap.toFixed(1);
    const name = t.team_name.slice(0, 24).padEnd(24);
    const marker = t.team_name === 'Human-Like' ? ' <<<' : '';
    // Estimate raw score: weighted / latest_weight
    const latestWeight = sorted.length > 0 ? Math.pow(1.05, sorted[sorted.length-1].round_number) : 1;
    const estRaw = (t.weighted_score / latestWeight).toFixed(1);
    console.log(`#${String(t.rank).padEnd(3)}  ${name}  ${t.weighted_score.toFixed(1).padStart(8)}  ${gapStr.padStart(6)}  ~${estRaw}${marker}`);
  }
  if (us && us.rank > 20) {
    console.log('  ...');
    const name = 'Human-Like'.padEnd(24);
    const estRaw = (us.weighted_score / Math.pow(1.05, sorted[sorted.length-1].round_number)).toFixed(1);
    console.log(`#${String(us.rank).padEnd(3)}  ${name}  ${us.weighted_score.toFixed(1).padStart(8)}    0.0  ~${estRaw} <<<`);
  }

  // Projections
  console.log('\n\u2500\u2500 PROJECTIONS \u2500'.padEnd(58, '\u2500'));
  const activeRound = sorted.find(r => r.status === 'active');
  const nextRoundNum = activeRound ? activeRound.round_number + 1 : (sorted.length > 0 ? sorted[sorted.length-1].round_number + 1 : 8);

  for (const futureR of [nextRoundNum, nextRoundNum+1, nextRoundNum+2]) {
    const w = Math.pow(1.05, futureR);
    const neededRaw = (lb[0].weighted_score + 1) / w;
    console.log(`R${futureR} (weight ${w.toFixed(3)}): raw ${neededRaw.toFixed(1)} needed for #1 | raw 82 = ${(82*w).toFixed(1)} | raw 90 = ${(90*w).toFixed(1)}`);
  }

  // Local validation results
  console.log('\n\u2500\u2500 LOCAL VALIDATION \u2500'.padEnd(58, '\u2500'));
  console.log('Leave-one-out cross-validation (seed 0 only):');
  console.log('  Average lookup floor:  ~82 (varies 70-86 per round)');
  console.log('  Round-matched lookup:  +0 to +7 when matched correctly');
  console.log('  In-sample ceiling:     ~86-94');
  console.log('  With observations:     +3 to +8 (never tested end-to-end)');

  // System
  console.log('\n\u2500\u2500 SYSTEM \u2500'.padEnd(58, '\u2500'));
  const cacheFiles = fs.readdirSync(path.join(__dirname, 'cache')).filter(f => f.endsWith('.json'));
  const gtSeeds = cacheFiles.filter(f => f.match(/gt_s/)).length;
  const gtRounds = [...new Set(cacheFiles.filter(f=>f.match(/gt_s/)).map(f=>f.match(/r(\d+)/)[1]))].length;
  const lookupBins = Object.keys(JSON.parse(fs.readFileSync(path.join(__dirname, 'gt_lookup.json'), 'utf8'))).length;
  const transRounds = cacheFiles.filter(f => f.startsWith('transitions')).length;

  let v3Running = false, v3Pid = '-', watchRunning = false, v3Log = '';
  try {
    const ps = execSync('ps aux', {encoding: 'utf8'});
    v3Running = ps.includes('auto_submit_v3');
    watchRunning = ps.includes('watch_and_calibrate');
    if (v3Running) {
      const line = ps.split('\n').find(l => l.includes('auto_submit_v3'));
      if (line) v3Pid = line.trim().split(/\s+/)[1];
    }
    // Last v3 log line
    try { v3Log = execSync('tail -1 /tmp/v3.log 2>/dev/null', {encoding:'utf8'}).trim(); } catch {}
  } catch {}

  console.log(`GT data:         ${gtSeeds} seeds from ${gtRounds} rounds`);
  console.log(`Lookup:          ${lookupBins} context bins`);
  console.log(`Transitions:     ${transRounds} rounds`);
  console.log(`auto_submit_v3:  ${v3Running ? '\u2713 RUNNING (PID '+v3Pid+')' : '\u2717 STOPPED'}`);
  console.log(`watch_calibrate: ${watchRunning ? '\u2713 RUNNING' : '\u2717 STOPPED'}`);
  if (v3Log) console.log(`v3 last log:     ${v3Log.slice(0, 70)}`);

  // Action items
  console.log('\n\u2500\u2500 ACTION ITEMS \u2500'.padEnd(58, '\u2500'));
  if (!v3Running) console.log('  [!] auto_submit_v3 is STOPPED - restart it!');
  if (!watchRunning) console.log('  [!] watch_and_calibrate is STOPPED - restart it!');
  if (activeRound) {
    console.log(`  [*] R${activeRound.round_number} is ACTIVE - v3 should be handling it`);
    if (activeRound.queries_used >= activeRound.queries_max)
      console.log(`  [!] R${activeRound.round_number} queries exhausted (${activeRound.queries_used}/${activeRound.queries_max})`);
    if (activeRound.seeds_submitted >= 5)
      console.log(`  [ok] R${activeRound.round_number} all seeds submitted`);
  }
  console.log('');

})().catch(e => console.error('Error:', e.message));
