#!/usr/bin/env python3
"""
Astar Island Dashboard — live tracking of competition status, scores, and progress.

Usage:
    python monitor.py                      # One-shot status
    python monitor.py --poll               # Poll every 60s
    python monitor.py --analyze            # Deep analysis of completed rounds

Reads AINM_TOKEN from ../.env.ainm automatically.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import requests

BASE = "https://api.ainm.no"
CLASS_NAMES = ["Empty", "Settlement", "Port", "Ruin", "Forest", "Mountain"]


def load_token():
    token = os.environ.get("AINM_TOKEN", "")
    if not token:
        env_file = Path(__file__).parent.parent / ".env.ainm"
        if env_file.exists():
            for line in env_file.read_text().splitlines():
                if line.startswith("AINM_TOKEN="):
                    token = line.split("=", 1)[1].strip()
    return token


def create_session(token):
    s = requests.Session()
    s.headers["Authorization"] = f"Bearer {token}"
    return s


def check_processes():
    """Check which agents are running."""
    processes = []
    try:
        result = subprocess.run(
            ["ps", "aux"], capture_output=True, text=True, timeout=5,
        )
        for line in result.stdout.splitlines():
            if "grep" in line:
                continue
            if "autopilot" in line and "python" in line.lower():
                processes.append(("autopilot.py", "v7 predictor (KT+lookup+adaptive)"))
            elif "auto_submit_v3" in line:
                processes.append(("auto_submit_v3.js", "JS lookup (OLD — should be stopped!)"))
            elif "auto_submit" in line and "python" in line.lower():
                processes.append(("auto_submit.py", "Python (OLD — should be stopped!)"))
            elif "watch_and_calibrate" in line:
                processes.append(("watch_and_calibrate.js", "GT cache updater (keep running)"))
    except Exception:
        pass
    return processes


def check_status(session):
    my_rounds = session.get(f"{BASE}/astar-island/my-rounds").json()
    if isinstance(my_rounds, dict):
        my_rounds = my_rounds.get("data", [])
    rounds = session.get(f"{BASE}/astar-island/rounds").json()
    if isinstance(rounds, dict):
        rounds = rounds.get("data", rounds.get("rounds", []))
    lb = session.get(f"{BASE}/astar-island/leaderboard").json()
    if isinstance(lb, dict):
        lb = lb.get("data", lb.get("leaderboard", []))

    budget = None
    try:
        budget = session.get(f"{BASE}/astar-island/budget").json()
    except Exception:
        pass

    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    print(f"\n{'='*70}")
    print(f"  ASTAR ISLAND DASHBOARD — {now}")
    print(f"{'='*70}")

    # ── Running Agents ───────────────────────────────────────────────────
    procs = check_processes()
    print(f"\n  AGENTS:")
    if procs:
        for name, desc in procs:
            print(f"    {name:<30} {desc}")
    else:
        print(f"    NO AGENTS RUNNING — start autopilot.py!")

    has_old = any("OLD" in desc for _, desc in procs)
    has_new = any("v7" in desc for _, desc in procs)
    if has_old:
        print(f"    WARNING: Old agents still running! Kill them.")
    if not has_new:
        print(f"    WARNING: New autopilot not running!")

    # ── Active Round ─────────────────────────────────────────────────────
    active = [r for r in rounds if r.get("status") == "active"]
    if active:
        r = active[0]
        weight = r.get("round_weight", 1.0)
        closes = r.get("closes_at", "?")
        print(f"\n  LIVE: Round {r.get('round_number')} | "
              f"weight {weight:.4f}x | closes {closes}")
        if budget:
            qu = budget.get("queries_used", 0)
            qm = budget.get("queries_max", 50)
            bar = "█" * (qu * 30 // qm) + "░" * (30 - qu * 30 // qm)
            print(f"    Queries: [{bar}] {qu}/{qm}")
        my_r = next((m for m in my_rounds if m.get("id") == r.get("id")), None)
        if my_r:
            submitted = my_r.get("seeds_submitted", 0)
            status = "COMPLETE" if submitted >= 5 else f"INCOMPLETE ({submitted}/5)!"
            print(f"    Seeds: {status}")
    else:
        scoring = [r for r in rounds if r.get("status") == "scoring"]
        if scoring:
            print(f"\n  SCORING: Round {scoring[0].get('round_number')} (waiting...)")
        else:
            print(f"\n  No active round. Next round TBD.")

    # ── Score History ────────────────────────────────────────────────────
    print(f"\n  SCORE HISTORY:")
    print(f"  {'Rnd':<5} {'Raw':>6} {'×Wt':>8} {'Rank':>10} {'Trend':>6}")
    print(f"  {'-'*40}")

    best_weighted = 0
    best_round = None
    prev_score = None
    scored_rounds = sorted(my_rounds, key=lambda x: x.get("round_number", 0))

    for r in scored_rounds:
        rn = r.get("round_number", "?")
        score = r.get("round_score")
        weight = r.get("round_weight", 1.0)
        rank = r.get("rank")
        total = r.get("total_teams")

        raw_str = f"{score:.1f}" if score else "--"
        weighted = score * weight if score else 0
        w_str = f"{weighted:.1f}" if score else "--"
        rank_str = f"#{rank}/{total}" if rank else "--"

        if score and weighted > best_weighted:
            best_weighted = weighted
            best_round = rn

        trend = ""
        if score and prev_score:
            d = score - prev_score
            if d > 3:
                trend = f"↑{d:.0f}"
            elif d < -3:
                trend = f"↓{abs(d):.0f}"
            else:
                trend = "→"
        if score:
            prev_score = score

        print(f"  R{rn:<4} {raw_str:>6} {w_str:>8} {rank_str:>10} {trend:>6}")

    if best_round:
        print(f"\n  Best: R{best_round} = {best_weighted:.1f} weighted")

    # ── Leaderboard ──────────────────────────────────────────────────────
    print(f"\n  LEADERBOARD (top 10):")
    print(f"  {'#':<4} {'Team':<30} {'WtScore':>8} {'Streak':>8}")
    print(f"  {'-'*52}")

    our_rank = None
    our_team = None
    for entry in lb:
        rank = entry.get("rank", 999)
        name = entry.get("team_name", "?")
        score = entry.get("weighted_score", 0)
        streak = entry.get("hot_streak_score", 0)

        if rank <= 10:
            print(f"  {rank:<4} {name[:29]:<30} {score:>8.1f} {streak:>8.1f}")

        if abs(score - best_weighted) < 2.0:
            our_team = name
            our_rank = rank

    total_teams = len(lb)
    if our_rank:
        print(f"\n  US: #{our_rank}/{total_teams} ({our_team})")

    # ── Gap Analysis ─────────────────────────────────────────────────────
    if lb and best_weighted:
        top = lb[0]
        top_score = top.get("weighted_score", 0)
        gap = top_score - best_weighted
        print(f"\n  GAP TO WIN:")
        print(f"    #1 {top.get('team_name','?')}: {top_score:.1f}")
        print(f"    Us: {best_weighted:.1f}  (gap: {gap:.1f})")

        max_weight = max((r.get("round_weight", 1.0) for r in rounds), default=1.0)
        next_weight = max_weight * 1.05
        needed = top_score / next_weight
        print(f"    Next round ~{next_weight:.3f}x → need {needed:.0f} raw to match #1")

        for target in [85, 90, 95]:
            tw = target * next_weight
            wr = sum(1 for e in lb if e.get("weighted_score", 0) > tw) + 1
            print(f"    {target} raw → ~#{wr}")

    # ── Per-Seed Breakdown ───────────────────────────────────────────────
    scored = [r for r in my_rounds if r.get("round_score") is not None]
    if scored:
        latest = max(scored, key=lambda r: r.get("round_number", 0))
        seeds = latest.get("seed_scores", [])
        if seeds:
            rn = latest.get("round_number")
            print(f"\n  R{rn} SEEDS: {' | '.join(f'{s:.1f}' for s in seeds)}")
            spread = max(seeds) - min(seeds)
            if spread > 10:
                print(f"    Spread: {spread:.1f} — INCONSISTENT!")
            else:
                print(f"    Spread: {spread:.1f} — consistent")

    print()
    return my_rounds, rounds, lb


def deep_analysis(session, my_rounds):
    scored = [r for r in my_rounds if r.get("round_score") is not None]
    if not scored:
        print("No scored rounds.")
        return

    for r in sorted(scored, key=lambda x: x.get("round_number", 0))[-2:]:
        round_id = r["id"]
        rn = r.get("round_number")
        seeds_count = r.get("seeds_count", 5)

        print(f"\n{'='*70}")
        print(f"  ROUND {rn} DEEP ANALYSIS — Score {r.get('round_score'):.1f}")
        print(f"{'='*70}")

        all_class_kl = {c: [] for c in range(6)}
        all_class_bias = {c: np.zeros(6) for c in range(6)}
        all_class_count = {c: 0 for c in range(6)}

        for seed_idx in range(seeds_count):
            try:
                data = session.get(
                    f"{BASE}/astar-island/analysis/{round_id}/{seed_idx}"
                ).json()
                if "ground_truth" not in data:
                    continue

                gt = np.array(data["ground_truth"])
                pred = np.array(data["prediction"])
                init = np.array(data.get("initial_grid", []))
                if init.size == 0:
                    continue

                eps = 1e-10
                p = np.clip(gt, eps, 1.0)
                q = np.clip(pred, eps, 1.0)
                kl = np.sum(p * np.log(p / q), axis=-1)
                entropy = -np.sum(p * np.log(p + eps), axis=-1)

                H, W = init.shape
                cls_map = {10: 0, 11: 0, 0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5}
                for y in range(H):
                    for x in range(W):
                        ic = cls_map.get(int(init[y, x]), 0)
                        if entropy[y, x] > 0.01:
                            all_class_kl[ic].append(float(kl[y, x]))
                            all_class_bias[ic] += pred[y, x] - gt[y, x]
                            all_class_count[ic] += 1

            except Exception as e:
                print(f"  Seed {seed_idx}: {e}")

        print(f"\n  Prediction bias (pred - GT) by terrain:")
        print(f"  {'Terrain':<12} {'AvgKL':>7} {'n':>6}  Bias per class (+ = overpredict)")
        for c in range(6):
            if all_class_kl[c]:
                avg_kl = np.mean(all_class_kl[c])
                n = all_class_count[c]
                if n > 0:
                    bias = all_class_bias[c] / n
                    bias_parts = []
                    for j in range(6):
                        if abs(bias[j]) > 0.005:
                            bias_parts.append(f"{CLASS_NAMES[j][:4]}:{bias[j]:+.3f}")
                    bias_str = " ".join(bias_parts) if bias_parts else "(minimal)"
                    print(f"  {CLASS_NAMES[c]:<12} {avg_kl:>7.4f} {n:>6}  {bias_str}")

        # Transition rates for this round
        print(f"\n  This round's GT transition rates:")
        detail = session.get(f"{BASE}/astar-island/rounds/{round_id}").json()
        initial_states = detail["initial_states"]

        trans = np.zeros((6, 6), dtype=np.float64)
        for si in range(seeds_count):
            try:
                data = session.get(
                    f"{BASE}/astar-island/analysis/{round_id}/{si}"
                ).json()
                gt = np.array(data["ground_truth"])
                init_grid = np.array(data.get("initial_grid", initial_states[si]["grid"]))
                H, W = init_grid.shape
                cls_map = {10: 0, 11: 0, 0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5}
                for y in range(H):
                    for x in range(W):
                        ic = cls_map.get(int(init_grid[y, x]), 0)
                        trans[ic] += gt[y, x]
            except Exception:
                pass

        row_sums = trans.sum(axis=1, keepdims=True)
        row_sums = np.maximum(row_sums, 1.0)
        trans_pct = trans / row_sums * 100

        header = f"  {'':>12}" + "".join(f" {CLASS_NAMES[j][:5]:>6}" for j in range(6))
        print(header)
        for i in range(6):
            if row_sums[i, 0] > 10:
                row = f"  {CLASS_NAMES[i]:<12}"
                for j in range(6):
                    v = trans_pct[i, j]
                    row += f" {v:>5.1f}%"
                print(row)


def main():
    parser = argparse.ArgumentParser(description="Astar Island Dashboard")
    parser.add_argument("--token", default=None)
    parser.add_argument("--poll", action="store_true")
    parser.add_argument("--analyze", action="store_true")
    args = parser.parse_args()

    token = args.token or load_token()
    if not token:
        print("No token. Set AINM_TOKEN or use --token.")
        sys.exit(1)

    session = create_session(token)

    while True:
        my_rounds, rounds, lb = check_status(session)
        if args.analyze:
            deep_analysis(session, my_rounds)
        if not args.poll:
            break
        print(f"Polling in 60s...")
        time.sleep(60)


if __name__ == "__main__":
    main()
