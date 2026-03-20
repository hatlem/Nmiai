#!/usr/bin/env python3
"""Auto-submit loop for Tripletex competition.

Submits ONE at a time, logs result, learns from failures.
Saves all results to results.jsonl for analysis.

Usage:
    # Close Chrome first, then:
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
RESULTS_FILE = Path(__file__).parent / "competition_results.jsonl"


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
        # Match pattern: "Task (N/M) HH:MM XM · Xs N/M (P%)"
        m = re.search(r'Task \((\d+)/(\d+)\).*?(\d+\.?\d*)s.*?(\d+)/(\d+)\s*\((\d+)%\)', text)
        if m:
            return {
                "checks_passed": int(m.group(1)),
                "checks_total": int(m.group(2)),
                "time_seconds": float(m.group(3)),
                "score_pct": int(m.group(6)),
            }
        # Rate limited
        if "Rate limited" in text or "Daily limit" in text:
            return {"rate_limited": True}
    return None


async def run():
    from playwright.async_api import async_playwright

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        context = await browser.new_context()
        page = await context.new_page()

        print("Navigating to submission page...")
        await page.goto(SUBMIT_URL, wait_until="networkidle", timeout=30000)
        await asyncio.sleep(5)

        # Check login
        body = await page.text_content("body") or ""
        if "Endpoint URL" not in body:
            print("\n>>> NOT LOGGED IN — log in manually in the browser, then press Enter")
            await asyncio.to_thread(input)
            await page.goto(SUBMIT_URL, wait_until="networkidle")
            await asyncio.sleep(5)

        submission_num = 0
        consecutive_100 = 0
        consecutive_fail = 0

        while True:
            # Refresh page
            await page.goto(SUBMIT_URL, wait_until="networkidle", timeout=30000)
            await asyncio.sleep(3)

            # Check score and limits
            info = await get_score_info(page)
            used = info.get("used", 0)
            limit = info.get("limit", 0)
            remaining = limit - used

            if remaining <= 0:
                now = datetime.now().strftime("%H:%M:%S")
                print(f"\n[{now}] Daily limit reached ({used}/{limit}). Waiting 5 min...")
                await asyncio.sleep(300)
                continue

            submission_num += 1
            now = datetime.now().strftime("%H:%M:%S")
            print(f"\n{'='*50}")
            print(f"[{now}] Submission #{submission_num} | Rank #{info.get('rank','?')} | "
                  f"Tasks {info.get('tasks_solved','?')}/30 | "
                  f"{remaining} submissions left")
            print(f"{'='*50}")

            # Fill URL and submit
            try:
                url_input = page.get_by_role("textbox", name="Endpoint URL")
                await url_input.fill(ENDPOINT_URL)
                await asyncio.sleep(1)
                btn = page.get_by_role("button", name="Submit")
                await btn.click()
                print("  Submitted. Waiting for evaluation...")
            except Exception as e:
                print(f"  Submit error: {e}")
                await asyncio.sleep(10)
                continue

            # Wait for result
            start = time.time()
            while time.time() - start < MAX_WAIT:
                await asyncio.sleep(15)
                body = await page.text_content("body") or ""
                if "Evaluating" not in body:
                    break
                elapsed = int(time.time() - start)
                eval_m = re.search(r'Evaluating \((\d+)\)', body)
                count = eval_m.group(1) if eval_m else "?"
                print(f"  Evaluating ({count})... {elapsed}s", end="\r")

            print()  # newline after \r
            await asyncio.sleep(3)

            # Get result
            result = await get_latest_result(page)
            if not result:
                print("  Could not parse result")
                continue

            if result.get("rate_limited"):
                print("  Rate limited!")
                continue

            pct = result["score_pct"]
            checks = f"{result['checks_passed']}/{result['checks_total']}"
            secs = result["time_seconds"]

            # Log result
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "submission": submission_num,
                "checks": checks,
                "score_pct": pct,
                "time_s": secs,
                "rank": info.get("rank"),
                "tasks_solved": info.get("tasks_solved"),
            }
            with open(RESULTS_FILE, "a") as f:
                f.write(json.dumps(log_entry) + "\n")

            # Print result with emphasis
            if pct == 100:
                consecutive_100 += 1
                consecutive_fail = 0
                print(f"  ✓ PERFECT {checks} ({secs}s) — streak: {consecutive_100}")
            elif pct == 0:
                consecutive_100 = 0
                consecutive_fail += 1
                print(f"  ✗ FAILED {checks} ({secs}s) — consecutive fails: {consecutive_fail}")
                if consecutive_fail >= 3:
                    print("  >>> 3 consecutive failures! Pausing 2 min for investigation...")
                    print(f"  >>> Check logs: gcloud run services logs read tripletex-agent --region europe-north1 --limit 30")
                    await asyncio.sleep(120)
                    consecutive_fail = 0
            else:
                consecutive_100 = 0
                consecutive_fail = 0
                print(f"  ~ PARTIAL {checks} = {pct}% ({secs}s)")

            # Brief pause
            await asyncio.sleep(5)


if __name__ == "__main__":
    print("=" * 50)
    print("  TRIPLETEX AUTO-SUBMIT (1 at a time)")
    print(f"  Endpoint: {ENDPOINT_URL}")
    print(f"  Results: {RESULTS_FILE}")
    print("  Press Ctrl+C to stop")
    print("=" * 50)

    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        print("\n\nStopped.")
        # Print summary
        if RESULTS_FILE.exists():
            lines = RESULTS_FILE.read_text().strip().split("\n")
            results = [json.loads(l) for l in lines]
            perfect = sum(1 for r in results if r["score_pct"] == 100)
            failed = sum(1 for r in results if r["score_pct"] == 0)
            partial = len(results) - perfect - failed
            print(f"\nSession summary: {len(results)} submissions")
            print(f"  Perfect: {perfect} | Partial: {partial} | Failed: {failed}")
