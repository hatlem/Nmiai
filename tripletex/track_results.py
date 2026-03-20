#!/usr/bin/env python3
"""Failure analysis for Tripletex agent results.

Reads results.jsonl and prints actionable insights:
- Success rate overall and by detected task type
- Most common API errors (path + status code)
- Slowest tasks
- Error pattern clustering

Usage:
    python track_results.py                  # analyze all
    python track_results.py --failures       # only show failures
    python track_results.py --last 20        # last N tasks
    python track_results.py --errors         # focus on API error patterns
"""

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

RESULTS_FILE = Path(__file__).parent / "results.jsonl"

# Heuristic task type detection from prompt text
TASK_PATTERNS = [
    # More specific patterns FIRST (before generic "invoice"/"faktura")
    (r"(leverand.r.*faktura|supplier.?invoice)", "supplier_invoice"),
    (r"(kreditnota|credit.?note)", "credit_note"),
    (r"(purring|reminder)", "reminder"),
    (r"(bilag|voucher|bokf)", "voucher"),
    (r"(innkj.psordre|purchase.?order)", "purchase_order"),
    (r"(reiseregning|travel.?expense)", "travel_expense"),
    (r"(l.nn|salary|payslip)", "salary"),
    (r"(faktura|invoice)", "invoice"),  # After supplier_invoice, credit_note
    (r"(ansatt|employee|ansettelse|employment)", "employee"),
    (r"(kunde|customer)", "customer"),
    (r"(leverand.r|supplier)", "supplier"),
    (r"(prosjekt|project)", "project"),
    (r"(avdeling|department)", "department"),
    (r"(produkt|product)", "product"),
    (r"(kontakt|contact)", "contact"),
    (r"(bankavstemming|bank.?reconcil)", "bank_reconciliation"),
    (r"(anleggsmiddel|fixed.?asset|asset)", "asset"),
    (r"(reverser|reverse)", "reversal"),
    (r"(godkjenn|approve)", "approval"),
    (r"(slett|delete)", "deletion"),
    (r"(.pningsbalanse|opening.?balance)", "opening_balance"),
]


def detect_task_type(prompt: str) -> str:
    """Detect task type from prompt using keyword matching."""
    p = prompt.lower()
    for pattern, task_type in TASK_PATTERNS:
        if re.search(pattern, p):
            return task_type
    return "unknown"


def load_results(last_n: int = 0) -> list[dict]:
    if not RESULTS_FILE.exists():
        print(f"No results file found at {RESULTS_FILE}")
        print("Results are logged when the agent processes tasks via /solve.")
        sys.exit(1)

    entries = []
    with open(RESULTS_FILE) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

    if not entries:
        print("results.jsonl is empty.")
        sys.exit(1)

    if last_n > 0:
        entries = entries[-last_n:]
    return entries


def analyze_overview(entries: list[dict]):
    total = len(entries)
    ok = sum(1 for e in entries if e.get("ok"))
    fail = total - ok
    avg_time = sum(e.get("elapsed", 0) for e in entries) / total if total else 0
    avg_calls = sum(e.get("api_calls", 0) for e in entries) / total if total else 0
    total_errors = sum(e.get("errors", 0) for e in entries)

    print("=" * 60)
    print(f"  TRIPLETEX AGENT — FAILURE ANALYSIS ({total} tasks)")
    print("=" * 60)
    print(f"  Success: {ok}/{total} ({ok/total*100:.0f}%)" if total else "  No data")
    print(f"  Failed:  {fail}/{total} ({fail/total*100:.0f}%)" if total else "")
    print(f"  Avg time: {avg_time:.1f}s | Avg API calls: {avg_calls:.1f} | Total API errors: {total_errors}")
    print()


def analyze_by_task_type(entries: list[dict]):
    by_type = defaultdict(lambda: {"total": 0, "ok": 0, "fail": 0, "times": [], "errors": 0})

    for e in entries:
        t = detect_task_type(e.get("prompt", ""))
        d = by_type[t]
        d["total"] += 1
        if e.get("ok"):
            d["ok"] += 1
        else:
            d["fail"] += 1
        d["times"].append(e.get("elapsed", 0))
        d["errors"] += e.get("errors", 0)

    print("  SUCCESS RATE BY TASK TYPE")
    print("  " + "-" * 56)
    print(f"  {'Type':<22} {'OK':>4} {'Fail':>4} {'Rate':>6} {'Avg(s)':>7} {'Errs':>5}")
    print("  " + "-" * 56)

    # Sort by failure rate (worst first)
    sorted_types = sorted(by_type.items(), key=lambda x: x[1]["ok"] / max(x[1]["total"], 1))
    for t, d in sorted_types:
        rate = d["ok"] / d["total"] * 100 if d["total"] else 0
        avg_t = sum(d["times"]) / len(d["times"]) if d["times"] else 0
        marker = " !!!" if rate < 50 else (" !" if rate < 80 else "")
        print(f"  {t:<22} {d['ok']:>4} {d['fail']:>4} {rate:>5.0f}% {avg_t:>6.1f}s {d['errors']:>5}{marker}")
    print()


def analyze_api_errors(entries: list[dict]):
    """Analyze which API paths fail most often."""
    error_counter = Counter()  # (method, path_pattern, status) -> count
    error_msgs = defaultdict(list)  # same key -> error snippets

    for e in entries:
        for call in e.get("call_log", []):
            if not call.get("ok"):
                # Normalize path: strip IDs
                path = re.sub(r"/\d+", "/{id}", call.get("path", ""))
                key = (call.get("method", "?"), path, call.get("status", 0))
                error_counter[key] += 1
                if call.get("error"):
                    error_msgs[key].append(call["error"][:150])

    if not error_counter:
        print("  No API errors recorded in call logs.")
        print()
        return

    print("  TOP API ERRORS (by frequency)")
    print("  " + "-" * 56)
    print(f"  {'Count':>5}  {'Method':<6} {'Status':>6}  {'Path'}")
    print("  " + "-" * 56)

    for (method, path, status), count in error_counter.most_common(15):
        print(f"  {count:>5}  {method:<6} {status:>6}  {path}")
        # Show one example error
        msgs = error_msgs[(method, path, status)]
        if msgs:
            print(f"         └─ {msgs[0][:80]}")
    print()


def analyze_failures(entries: list[dict]):
    """Show details of failed tasks."""
    failures = [e for e in entries if not e.get("ok")]
    if not failures:
        print("  No failures! All tasks succeeded.")
        print()
        return

    print(f"  FAILED TASKS ({len(failures)})")
    print("  " + "-" * 56)

    for e in failures:
        t = detect_task_type(e.get("prompt", ""))
        prompt_snip = e.get("prompt", "")[:80].replace("\n", " ")
        err = e.get("error_detail", "")[:100]
        time_str = e.get("time", "?")
        if "T" in time_str:
            time_str = time_str.split("T")[1][:8]

        print(f"  [{time_str}] {t:<20} {e.get('elapsed', 0):>5.1f}s  calls={e.get('api_calls', 0)} errs={e.get('errors', 0)}")
        print(f"    prompt: {prompt_snip}")
        if err:
            print(f"    error:  {err}")

        # Show failed API calls
        failed_calls = [c for c in e.get("call_log", []) if not c.get("ok")]
        if failed_calls:
            for c in failed_calls[:3]:
                print(f"    -> {c['method']} {c['path']} [{c['status']}] {c.get('error', '')[:60]}")
        print()


def analyze_slow_tasks(entries: list[dict], threshold: float = 120.0):
    """Show tasks that took too long."""
    slow = [e for e in entries if e.get("elapsed", 0) > threshold]
    if not slow:
        return

    slow.sort(key=lambda e: e.get("elapsed", 0), reverse=True)
    print(f"  SLOW TASKS (>{threshold:.0f}s) — {len(slow)} tasks")
    print("  " + "-" * 56)
    for e in slow[:10]:
        t = detect_task_type(e.get("prompt", ""))
        ok = "OK" if e.get("ok") else "FAIL"
        print(f"  {e.get('elapsed', 0):>6.1f}s  {ok:<4}  {t:<20} calls={e.get('api_calls', 0)}")
    print()


def main():
    last_n = 0
    show_failures_only = False
    show_errors_only = False

    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == "--last" and i + 1 < len(args):
            last_n = int(args[i + 1])
            i += 2
        elif args[i] == "--failures":
            show_failures_only = True
            i += 1
        elif args[i] == "--errors":
            show_errors_only = True
            i += 1
        else:
            i += 1

    entries = load_results(last_n)

    if show_errors_only:
        analyze_api_errors(entries)
        return

    analyze_overview(entries)

    if show_failures_only:
        analyze_failures(entries)
    else:
        analyze_by_task_type(entries)
        analyze_api_errors(entries)
        analyze_failures(entries)
        analyze_slow_tasks(entries)


if __name__ == "__main__":
    main()
