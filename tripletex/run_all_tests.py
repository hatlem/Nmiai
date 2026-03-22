#!/usr/bin/env python3
"""
Test runner for Tripletex agent.
Reads all prompts_*.json files, merges them, and sends each to the /solve endpoint.
Skips file-dependent prompts (PDF, CSV, attachments).
Runs 5 concurrent requests with 300s timeout.
"""

import asyncio
import glob
import json
import sys
import time
from dataclasses import dataclass, field

import httpx

SOLVE_URL = "https://tripletex-agent-174612781810.europe-north1.run.app/solve"
BASE_URL = "https://kkpqfuj-amager.tripletex.dev/v2"
SESSION_TOKEN = "eyJ0b2tlbklkIjoyMTQ3NjM4ODA2LCJ0b2tlbiI6ImNlZGY2OWJjLWQ4NTEtNGM1ZS1hMmUxLTJjMDU1YjNmYzQzMCJ9"
CONCURRENCY = 5
TIMEOUT = 300

SKIP_KEYWORDS = [
    "pdf", "csv", "vedlagt", "attached", "ci-joint",
    "adjunto", "anexo", "beigefugte",
]


@dataclass
class Result:
    index: int
    prompt: str
    status: str = ""       # "pass", "fail", "error", "skipped"
    elapsed: float = 0.0
    error: str = ""
    response_data: dict = field(default_factory=dict)


def should_skip(prompt: str) -> bool:
    lower = prompt.lower()
    return any(kw in lower for kw in SKIP_KEYWORDS)


def load_prompts(pattern: str) -> list[dict]:
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"No files matching {pattern}")
        sys.exit(1)

    prompts = []
    for f in files:
        print(f"Loading {f}")
        with open(f) as fh:
            data = json.load(fh)
            if isinstance(data, list):
                prompts.extend(data)
            elif isinstance(data, dict):
                prompts.append(data)
            else:
                print(f"  WARNING: unexpected format in {f}, skipping")
    print(f"Loaded {len(prompts)} prompts from {len(files)} file(s)\n")
    return prompts


def extract_prompt_text(item) -> str:
    """Extract the prompt string from various possible formats."""
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        for key in ("prompt", "text", "task", "description", "message", "content"):
            if key in item:
                return str(item[key])
        return json.dumps(item)
    return str(item)


async def send_request(client: httpx.AsyncClient, sem: asyncio.Semaphore, result: Result, prompt_text: str):
    async with sem:
        payload = {
            "prompt": prompt_text,
            "tripletex_credentials": {
                "base_url": BASE_URL,
                "session_token": SESSION_TOKEN,
            },
            "files": [],
        }

        t0 = time.monotonic()
        try:
            resp = await client.post(SOLVE_URL, json=payload, timeout=TIMEOUT)
            result.elapsed = time.monotonic() - t0
            result.response_data = resp.json() if resp.status_code == 200 else {}

            if resp.status_code == 200:
                result.status = "pass"
            else:
                result.status = "fail"
                result.error = f"HTTP {resp.status_code}: {resp.text[:300]}"
        except httpx.TimeoutException:
            result.elapsed = time.monotonic() - t0
            result.status = "error"
            result.error = "TIMEOUT (300s)"
        except Exception as e:
            result.elapsed = time.monotonic() - t0
            result.status = "error"
            result.error = f"{type(e).__name__}: {e}"

        # Live output
        icon = {"pass": "OK", "fail": "FAIL", "error": "ERR", "skipped": "SKIP"}[result.status]
        short = prompt_text[:80].replace("\n", " ")
        print(f"  [{icon}] #{result.index:3d} ({result.elapsed:5.1f}s) {short}")
        if result.error:
            print(f"         -> {result.error[:200]}")


async def main():
    pattern = "/Users/andreashatlem/Projects/nmiai/tripletex/prompts_*.json"
    prompts = load_prompts(pattern)

    results: list[Result] = []
    tasks = []
    sem = asyncio.Semaphore(CONCURRENCY)

    async with httpx.AsyncClient() as client:
        for i, item in enumerate(prompts):
            prompt_text = extract_prompt_text(item)
            r = Result(index=i, prompt=prompt_text[:200])

            if should_skip(prompt_text):
                r.status = "skipped"
                results.append(r)
                print(f"  [SKIP] #{i:3d}         {prompt_text[:80].replace(chr(10), ' ')}")
                continue

            results.append(r)
            tasks.append(send_request(client, sem, r, prompt_text))

        if tasks:
            print(f"Running {len(tasks)} requests ({CONCURRENCY} concurrent, {TIMEOUT}s timeout)...\n")
            await asyncio.gather(*tasks)

    # Summary
    passed = sum(1 for r in results if r.status == "pass")
    failed = sum(1 for r in results if r.status == "fail")
    errors = sum(1 for r in results if r.status == "error")
    skipped = sum(1 for r in results if r.status == "skipped")
    total = len(results)
    total_time = sum(r.elapsed for r in results)
    avg_time = total_time / max(1, total - skipped)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Total:   {total}")
    print(f"  Passed:  {passed}")
    print(f"  Failed:  {failed}")
    print(f"  Errors:  {errors}")
    print(f"  Skipped: {skipped}")
    print(f"  Avg time: {avg_time:.1f}s")
    print(f"  Total time: {total_time:.1f}s (wall clock lower due to concurrency)")

    if failed + errors > 0:
        print(f"\n--- Failed/Error details ---")
        for r in results:
            if r.status in ("fail", "error"):
                print(f"  #{r.index}: [{r.status.upper()}] {r.prompt[:100]}")
                print(f"    {r.error[:300]}")

    print()
    return 0 if (failed + errors) == 0 else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
