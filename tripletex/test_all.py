"""Test ALL 176 real competition prompts against our agent."""
import asyncio
import httpx
import time
import sys

AGENT_URL = "https://tripletex-agent-174612781810.europe-north1.run.app/solve"
SANDBOX_URL = "https://kkpqfuj-amager.tripletex.dev/v2"
SANDBOX_TOKEN = "eyJ0b2tlbklkIjoyMTQ3NjM4ODA2LCJ0b2tlbiI6ImNlZGY2OWJjLWQ4NTEtNGM1ZS1hMmUxLTJjMDU1YjNmYzQzMCJ9"

CONCURRENCY = 3  # Match competition concurrent limit


async def test_prompt(sem: asyncio.Semaphore, client: httpx.AsyncClient, idx: int, prompt: str):
    async with sem:
        start = time.monotonic()
        try:
            resp = await client.post(
                AGENT_URL,
                json={
                    "prompt": prompt,
                    "files": [],
                    "tripletex_credentials": {
                        "base_url": SANDBOX_URL,
                        "session_token": SANDBOX_TOKEN,
                    },
                },
                timeout=300.0,
            )
            elapsed = time.monotonic() - start
            ok = resp.status_code == 200
            status = "✓" if ok else "✗"
            print(f"  {status} [{idx:3d}] {elapsed:5.1f}s | {prompt[:80]}")
            return ok
        except Exception as e:
            elapsed = time.monotonic() - start
            print(f"  ✗ [{idx:3d}] {elapsed:5.1f}s | ERROR: {str(e)[:50]} | {prompt[:60]}")
            return False


async def main():
    with open("/tmp/all_prompts.txt") as f:
        prompts = [line.strip() for line in f if line.strip()]

    # Allow filtering by range
    start_idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    end_idx = int(sys.argv[2]) if len(sys.argv) > 2 else len(prompts)
    prompts = list(enumerate(prompts))[start_idx:end_idx]

    print(f"Testing {len(prompts)} prompts (idx {start_idx}-{end_idx-1}), concurrency={CONCURRENCY}")
    print()

    sem = asyncio.Semaphore(CONCURRENCY)
    async with httpx.AsyncClient() as client:
        tasks = [test_prompt(sem, client, idx, prompt) for idx, prompt in prompts]
        results = await asyncio.gather(*tasks)

    passed = sum(1 for r in results if r)
    failed = sum(1 for r in results if not r)
    print(f"\n{'='*60}")
    print(f"RESULTS: {passed}/{len(results)} passed, {failed} failed")


if __name__ == "__main__":
    asyncio.run(main())
