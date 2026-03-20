#!/usr/bin/env python3
"""Auto-submit loop for Tripletex competition — 3 CONCURRENT submissions.

Opens 3 browser tabs and submits in parallel for maximum throughput.
Logs all results to competition_results.jsonl.

Usage:
    /usr/local/bin/python3.11 auto_submit.py
"""
import asyncio
import json
import re
import time
from datetime import datetime
from pathlib import Path

ENDPOINT_URL = "https://tripletex-agent-174612781810.europe-north1.run.app/solve"
SUBMIT_URL = "https://app.ainm.no/submit/tripletex"
MAX_WAIT = 330  # 5.5 min
CONCURRENT_SLOTS = 3
RESULTS_FILE = Path(__file__).parent / "competition_results.jsonl"
LOG_LOCK = asyncio.Lock()
STATS = {"total": 0, "perfect": 0, "partial": 0, "failed": 0}


async def get_score_info(page) -> dict:
    body = await page.text_content("body") or ""
    info = {}
    m = re.search(r'(\d+)\s*/\s*(\d+)\s*daily submissions used', body)
    if m:
        info["used"] = int(m.group(1))
        info["limit"] = int(m.group(2))
    m = re.search(r'#(\d+)', body)
    if m:
        info["rank"] = int(m.group(1))
    m = re.search(r'(\d+)/30', body)
    if m:
        info["tasks_solved"] = int(m.group(1))
    return info


async def get_latest_result(page) -> dict | None:
    """Parse the most recent result from the page."""
    buttons = await page.query_selector_all("button")
    for btn in buttons:
        text = await btn.text_content() or ""
        m = re.search(r'Task \((\d+)/(\d+)\).*?(\d+\.?\d*)s.*?(\d+)/(\d+)\s*\((\d+)%\)', text)
        if m:
            return {
                "checks_passed": int(m.group(1)),
                "checks_total": int(m.group(2)),
                "time_seconds": float(m.group(3)),
                "score_pct": int(m.group(6)),
            }
        if "Rate limited" in text or "Daily limit" in text:
            return {"rate_limited": True}
    return None


async def log_result(slot: int, result: dict, info: dict, submission_num: int):
    """Thread-safe logging of results."""
    async with LOG_LOCK:
        pct = result["score_pct"]
        checks = f"{result['checks_passed']}/{result['checks_total']}"
        secs = result["time_seconds"]
        now = datetime.now().strftime("%H:%M:%S")

        STATS["total"] += 1
        if pct == 100:
            STATS["perfect"] += 1
            symbol = "PERFECT"
        elif pct == 0:
            STATS["failed"] += 1
            symbol = "FAILED"
        else:
            STATS["partial"] += 1
            symbol = f"PARTIAL {pct}%"

        print(f"  [{now}] Slot {slot}: {symbol} {checks} ({secs}s) | "
              f"Rank #{info.get('rank','?')} Tasks {info.get('tasks_solved','?')}/30 | "
              f"Session: {STATS['perfect']}ok {STATS['partial']}partial {STATS['failed']}fail / {STATS['total']}")

        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "slot": slot,
            "submission": submission_num,
            "checks": checks,
            "score_pct": pct,
            "time_s": secs,
            "rank": info.get("rank"),
            "tasks_solved": info.get("tasks_solved"),
        }
        with open(RESULTS_FILE, "a") as f:
            f.write(json.dumps(log_entry) + "\n")


async def submit_loop(slot: int, context, stop_event: asyncio.Event):
    """Single submission slot — runs in its own tab."""
    page = await context.new_page()
    submission_num = 0
    consecutive_fail = 0

    while not stop_event.is_set():
        try:
            await page.goto(SUBMIT_URL, wait_until="networkidle", timeout=30000)
            await asyncio.sleep(2)

            # Check daily limit
            info = await get_score_info(page)
            used = info.get("used", 0)
            limit = info.get("limit", 0)
            if limit > 0 and used >= limit:
                now = datetime.now().strftime("%H:%M:%S")
                print(f"  [{now}] Slot {slot}: Daily limit reached ({used}/{limit}). Waiting 5 min...")
                await asyncio.sleep(300)
                continue

            submission_num += 1

            # Fill URL and submit
            url_input = page.get_by_role("textbox", name="Endpoint URL")
            await url_input.fill(ENDPOINT_URL)
            await asyncio.sleep(0.5)
            btn = page.get_by_role("button", name="Submit")
            await btn.click()

            # Wait for result
            start = time.time()
            while time.time() - start < MAX_WAIT:
                await asyncio.sleep(10)
                body = await page.text_content("body") or ""
                if "Evaluating" not in body:
                    break

            await asyncio.sleep(2)

            # Parse result
            result = await get_latest_result(page)
            if not result:
                print(f"  Slot {slot}: Could not parse result")
                await asyncio.sleep(5)
                continue

            if result.get("rate_limited"):
                print(f"  Slot {slot}: Rate limited, waiting 60s...")
                await asyncio.sleep(60)
                continue

            # Refresh info after submission
            info = await get_score_info(page)
            await log_result(slot, result, info, submission_num)

            if result["score_pct"] == 0:
                consecutive_fail += 1
                if consecutive_fail >= 5:
                    print(f"  Slot {slot}: 5 consecutive fails! Pausing 2 min...")
                    await asyncio.sleep(120)
                    consecutive_fail = 0
            else:
                consecutive_fail = 0

            # Small stagger between submissions
            await asyncio.sleep(3)

        except Exception as e:
            print(f"  Slot {slot}: Error: {e}")
            await asyncio.sleep(15)
            try:
                await page.goto(SUBMIT_URL, wait_until="networkidle", timeout=30000)
            except Exception:
                pass


async def run():
    from playwright.async_api import async_playwright

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        context = await browser.new_context()

        # Login check on first page
        page = await context.new_page()
        print("Navigating to submission page...")
        await page.goto(SUBMIT_URL, wait_until="networkidle", timeout=30000)
        await asyncio.sleep(5)

        body = await page.text_content("body") or ""
        if "Endpoint URL" not in body:
            print("\n>>> NOT LOGGED IN — log in manually in the browser, then press Enter")
            await asyncio.to_thread(input)
            await page.goto(SUBMIT_URL, wait_until="networkidle")
            await asyncio.sleep(3)
        await page.close()

        # Check current status
        status_page = await context.new_page()
        await status_page.goto(SUBMIT_URL, wait_until="networkidle", timeout=30000)
        info = await get_score_info(status_page)
        await status_page.close()

        print(f"\nCurrent: Rank #{info.get('rank','?')} | Tasks {info.get('tasks_solved','?')}/30 | "
              f"Submissions {info.get('used','?')}/{info.get('limit','?')}")
        print(f"Launching {CONCURRENT_SLOTS} concurrent submission slots...\n")

        stop_event = asyncio.Event()
        tasks = [
            asyncio.create_task(submit_loop(i + 1, context, stop_event))
            for i in range(CONCURRENT_SLOTS)
        ]

        # Stagger start so we don't hit the page simultaneously
        await asyncio.sleep(1)

        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            pass
        finally:
            await browser.close()


if __name__ == "__main__":
    print("=" * 50)
    print(f"  TRIPLETEX AUTO-SUBMIT ({CONCURRENT_SLOTS}x CONCURRENT)")
    print(f"  Endpoint: {ENDPOINT_URL}")
    print(f"  Results: {RESULTS_FILE}")
    print("  Press Ctrl+C to stop")
    print("=" * 50)

    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        print("\n\nStopped.")
        if RESULTS_FILE.exists():
            lines = RESULTS_FILE.read_text().strip().split("\n")
            results = [json.loads(l) for l in lines if l.strip()]
            perfect = sum(1 for r in results if r.get("score_pct") == 100)
            failed = sum(1 for r in results if r.get("score_pct") == 0)
            partial = len(results) - perfect - failed
            print(f"\nTotal results: {len(results)} submissions")
            print(f"  Perfect: {perfect} | Partial: {partial} | Failed: {failed}")
