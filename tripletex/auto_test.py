#!/usr/bin/env python3
"""Auto-test harness for the Tripletex /solve endpoint.

Sends realistic task prompts to a local (or remote) /solve endpoint,
logs every request/response with timing, tracks success/failure rates,
and prints a summary report.

Usage:
    # Test against local dev server (default):
    python auto_test.py

    # Test against Cloud Run:
    python auto_test.py --url https://tripletex-agent-174612781810.europe-north1.run.app/solve

    # Filter by task type:
    python auto_test.py --filter invoice

    # Only tier 1 tasks:
    python auto_test.py --tier 1

    # Verbose mode (show full response bodies):
    python auto_test.py -v

Environment variables:
    TRIPLETEX_SESSION_TOKEN  — sandbox session token (required)
    TRIPLETEX_BASE_URL       — sandbox API base URL (default: https://kkpqfuj-amager.tripletex.dev/v2)
    AGENT_URL                — /solve endpoint URL (default: http://localhost:8000/solve)
"""

import argparse
import asyncio
import json
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import httpx

# ── Configuration ──

DEFAULT_AGENT_URL = "http://localhost:8000/solve"
DEFAULT_BASE_URL = "https://kkpqfuj-amager.tripletex.dev/v2"
LOG_DIR = Path(__file__).parent / "test_logs"
REQUEST_TIMEOUT = 320  # 5+ minutes, matching competition limit


# ── Test prompts organized by tier ──

TEST_PROMPTS: list[tuple[int, str, str]] = [
    # (tier, task_type, prompt)

    # ===== TIER 1 — Simple entity creation =====
    (1, "create_customer",
     "Opprett kunden Fjordtech AS med organisasjonsnummer 912345678. "
     "E-post: post@fjordtech.no. Telefon: 55667788. Adresse: Strandgata 12, 6800 Førde."),

    (1, "create_employee",
     "Opprett ansatt Kari Nordmann med e-post kari.nordmann@test.no, "
     "mobilnummer 91234567, født 1985-06-15."),

    (1, "create_product",
     "Opprett produktet 'Konsulenttimer' med produktnummer KT-001. "
     "Pris 1500 NOK eksklusiv MVA, standard MVA-sats 25%."),

    (1, "create_department",
     "Opprett avdelingen 'Økonomi' med avdelingsnummer 300."),

    (1, "create_supplier",
     "Registrer leverandøren Kontorservice AS med organisasjonsnummer 987654321. "
     "E-post: faktura@kontorservice.no."),

    (1, "create_contact",
     "Opprett kontaktperson Per Hansen for kunde Fjordtech AS. "
     "E-post: per@fjordtech.no, telefon 99887766."),

    (1, "create_internal_project",
     "Opprett et internt prosjekt med navn 'Systemoppgradering 2026'."),

    # ===== TIER 2 — Invoices, payments, projects =====
    (2, "create_invoice",
     "Opprett faktura til kunde Fjordtech AS (org.nr 912345678) for "
     "2 stk Konsulenttimer á 1500 kr. Fakturadato 2026-03-20, forfallsdato 2026-04-03."),

    (2, "create_invoice_existing_customer",
     "Opprett faktura til eksisterende kunde Fjordtech AS for "
     "1 stk Rådgivning á 3000 kr. Fakturadato 2026-03-20."),

    (2, "register_payment",
     "Registrer innbetaling på 3000 kr på faktura ID 1, betalingsdato 2026-03-20."),

    (2, "create_credit_note",
     "Opprett kreditnota for faktura nummer 1."),

    (2, "create_project",
     "Opprett prosjekt 'Websideutvikling' for kunde Fjordtech AS. Startdato 2026-03-20."),

    (2, "create_project_existing_customer",
     "Opprett prosjekt 'Vedlikehold' for eksisterende kunde Fjordtech AS."),

    (2, "create_travel_expense",
     "Registrer reiseregning for ansatt Kari Nordmann: Reise til Oslo 20. mars 2026, "
     "taxi 350 kr."),

    (2, "create_voucher",
     "Bokfør bilag: Kontorrekvisita 2500 kr, debet konto 6300, kredit konto 1920, "
     "dato 2026-03-20."),

    (2, "create_supplier_invoice",
     "Registrer leverandørfaktura fra Kontorservice AS. "
     "Fakturanummer F-2026-001, dato 2026-03-20, forfallsdato 2026-04-20, "
     "beløp 5000 kr på konto 6300."),

    (2, "update_employee",
     "Oppdater ansatt Kari Nordmann sin e-post til kari.ny@test.no."),

    (2, "update_customer",
     "Oppdater kunde Fjordtech AS sin telefon til 55112233."),

    (2, "create_customer_supplier",
     "Opprett Havglimt AS (org.nr 876543210) som både kunde og leverandør. "
     "E-post: post@havglimt.no."),

    # ===== TIER 3 — Complex multi-step =====
    (3, "create_invoice_with_payment",
     "Opprett faktura til kunde Strandvik AS (org.nr 111222333) for "
     "1 stk Konsultasjon á 5000 kr, og registrer full betaling med en gang. "
     "Fakturadato 2026-03-20."),

    (3, "create_opening_balance",
     "Sett åpningsbalanse per 2026-01-01: Bankkonto 1920 med 100000 kr, "
     "varelager konto 1400 med 50000 kr."),

    (3, "create_employment",
     "Registrer ansettelse for Kari Nordmann, startdato 2026-04-01, 100% stilling."),

    (3, "create_timesheet_entry",
     "Registrer 7.5 timer på prosjekt Websideutvikling, dato 2026-03-19, "
     "for ansatt Kari Nordmann."),

    # ===== Multi-language prompts =====
    (1, "create_employee_en",
     "Create employee John Smith, email john@example.com, phone +447911123456."),

    (2, "create_invoice_fr",
     "Créer une facture pour le client Paris Consulting (org.nr 998877665), "
     "1 article Service de conseil á 5000 NOK."),

    (1, "create_customer_de",
     "Erstellen Sie einen Kunden: Berliner Bäckerei GmbH, org.nr 112233445, "
     "E-Mail: info@berliner-baeckerei.de."),

    (1, "create_employee_es",
     "Crear empleado Carlos Garcia, email carlos@test.es, teléfono 612345678."),

    (1, "create_supplier_pt",
     "Criar fornecedor Lisboa Tech Lda, org.nr 554433221, email info@lisboatech.pt."),
]


# ── Result tracking ──

@dataclass
class TestResult:
    tier: int
    task_type: str
    prompt: str
    status_code: int | str
    success: bool
    elapsed: float
    response_body: dict | str | None = None
    error: str | None = None


@dataclass
class TestSession:
    started: str = field(default_factory=lambda: datetime.now().isoformat())
    agent_url: str = ""
    base_url: str = ""
    results: list[TestResult] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def passed(self) -> int:
        return sum(1 for r in self.results if r.success)

    @property
    def failed(self) -> int:
        return self.total - self.passed

    @property
    def avg_elapsed(self) -> float:
        if not self.results:
            return 0
        return sum(r.elapsed for r in self.results) / len(self.results)

    def by_tier(self, tier: int) -> list[TestResult]:
        return [r for r in self.results if r.tier == tier]

    def tier_pass_rate(self, tier: int) -> str:
        tier_results = self.by_tier(tier)
        if not tier_results:
            return "n/a"
        passed = sum(1 for r in tier_results if r.success)
        return f"{passed}/{len(tier_results)}"


# ── Test runner ──

async def run_single_test(
    client: httpx.AsyncClient,
    agent_url: str,
    base_url: str,
    session_token: str,
    tier: int,
    task_type: str,
    prompt: str,
    verbose: bool = False,
) -> TestResult:
    """Send one task to /solve and return the result."""
    payload = {
        "prompt": prompt,
        "tripletex_credentials": {
            "base_url": base_url,
            "session_token": session_token,
        },
    }

    start = time.monotonic()
    try:
        resp = await client.post(agent_url, json=payload)
        elapsed = time.monotonic() - start

        if resp.status_code == 200:
            body = resp.json()
            # The agent returns 200 even on some failures — check for error keys
            has_error = isinstance(body, dict) and body.get("error")
            success = not has_error
        else:
            body = resp.text[:500]
            success = False

        result = TestResult(
            tier=tier,
            task_type=task_type,
            prompt=prompt,
            status_code=resp.status_code,
            success=success,
            elapsed=round(elapsed, 1),
            response_body=body if verbose else None,
            error=body.get("error") if isinstance(body, dict) and body.get("error") else None,
        )
    except httpx.TimeoutException:
        elapsed = time.monotonic() - start
        result = TestResult(
            tier=tier, task_type=task_type, prompt=prompt,
            status_code="timeout", success=False, elapsed=round(elapsed, 1),
            error=f"Timeout after {REQUEST_TIMEOUT}s",
        )
    except Exception as e:
        elapsed = time.monotonic() - start
        result = TestResult(
            tier=tier, task_type=task_type, prompt=prompt,
            status_code="error", success=False, elapsed=round(elapsed, 1),
            error=str(e),
        )

    # Live progress
    icon = "\033[32mPASS\033[0m" if result.success else "\033[31mFAIL\033[0m"
    print(f"  [{icon}] T{tier} {task_type:40s} {result.elapsed:6.1f}s  {result.status_code}")
    if result.error:
        print(f"         Error: {result.error[:120]}")
    if verbose and result.response_body:
        print(f"         Response: {json.dumps(result.response_body, ensure_ascii=False)[:200]}")

    return result


async def run_all_tests(
    agent_url: str,
    base_url: str,
    session_token: str,
    tier_filter: int | None = None,
    name_filter: str | None = None,
    verbose: bool = False,
) -> TestSession:
    """Run all matching tests sequentially and return session results."""
    session = TestSession(agent_url=agent_url, base_url=base_url)

    # Filter prompts
    prompts = TEST_PROMPTS
    if tier_filter is not None:
        prompts = [(t, n, p) for t, n, p in prompts if t == tier_filter]
    if name_filter:
        prompts = [(t, n, p) for t, n, p in prompts if name_filter.lower() in n.lower()]

    if not prompts:
        print("No tests match the given filters.")
        return session

    total = len(prompts)
    print(f"\n{'=' * 80}")
    print(f"TRIPLETEX AUTO-TEST — {total} tasks")
    print(f"Agent: {agent_url}")
    print(f"Sandbox: {base_url}")
    print(f"Started: {session.started}")
    print(f"{'=' * 80}\n")

    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
        for i, (tier, task_type, prompt) in enumerate(prompts, 1):
            print(f"[{i}/{total}] {task_type}")
            result = await run_single_test(
                client, agent_url, base_url, session_token,
                tier, task_type, prompt, verbose,
            )
            session.results.append(result)
            # Small delay between tests to avoid hammering
            if i < total:
                await asyncio.sleep(1)

    return session


def print_summary(session: TestSession):
    """Print a formatted summary report."""
    print(f"\n{'=' * 80}")
    print("SUMMARY REPORT")
    print(f"{'=' * 80}")
    print(f"  Total:    {session.total}")
    print(f"  Passed:   \033[32m{session.passed}\033[0m")
    print(f"  Failed:   \033[31m{session.failed}\033[0m")
    print(f"  Avg time: {session.avg_elapsed:.1f}s")
    print()

    # Per-tier breakdown
    for tier in [1, 2, 3]:
        tier_results = session.by_tier(tier)
        if not tier_results:
            continue
        passed = sum(1 for r in tier_results if r.success)
        avg_t = sum(r.elapsed for r in tier_results) / len(tier_results)
        pct = passed / len(tier_results) * 100 if tier_results else 0
        bar = "\033[32m" + "#" * passed + "\033[31m" + "X" * (len(tier_results) - passed) + "\033[0m"
        print(f"  Tier {tier}: {passed}/{len(tier_results)} ({pct:.0f}%)  avg {avg_t:.1f}s  [{bar}]")

    # Failed tasks detail
    failed = [r for r in session.results if not r.success]
    if failed:
        print(f"\n{'─' * 80}")
        print("FAILED TASKS:")
        for r in failed:
            print(f"  T{r.tier} {r.task_type:40s} [{r.status_code}] {r.elapsed:.1f}s")
            if r.error:
                print(f"       {r.error[:150]}")
            print(f"       Prompt: {r.prompt[:100]}")

    # Timing outliers (slowest 5)
    sorted_by_time = sorted(session.results, key=lambda r: r.elapsed, reverse=True)
    print(f"\n{'─' * 80}")
    print("SLOWEST TASKS:")
    for r in sorted_by_time[:5]:
        icon = "\033[32mOK\033[0m" if r.success else "\033[31mFAIL\033[0m"
        print(f"  {r.elapsed:6.1f}s  [{icon}] T{r.tier} {r.task_type}")

    print(f"\n{'=' * 80}")


def save_log(session: TestSession):
    """Save results to a JSONL log file."""
    LOG_DIR.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = LOG_DIR / f"test_run_{ts}.jsonl"

    with open(log_file, "w") as f:
        # Header line
        header = {
            "type": "session",
            "started": session.started,
            "agent_url": session.agent_url,
            "total": session.total,
            "passed": session.passed,
            "failed": session.failed,
            "avg_elapsed": round(session.avg_elapsed, 1),
        }
        f.write(json.dumps(header) + "\n")

        for r in session.results:
            entry = {
                "type": "result",
                "tier": r.tier,
                "task_type": r.task_type,
                "prompt": r.prompt,
                "status_code": r.status_code,
                "success": r.success,
                "elapsed": r.elapsed,
                "error": r.error,
            }
            if r.response_body:
                entry["response"] = r.response_body
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"\nLog saved to: {log_file}")
    return log_file


def main():
    parser = argparse.ArgumentParser(description="Tripletex /solve auto-tester")
    parser.add_argument("--url", default=os.environ.get("AGENT_URL", DEFAULT_AGENT_URL),
                        help="Agent /solve endpoint URL")
    parser.add_argument("--base-url", default=os.environ.get("TRIPLETEX_BASE_URL", DEFAULT_BASE_URL),
                        help="Tripletex sandbox base URL")
    parser.add_argument("--token", default=os.environ.get("TRIPLETEX_SESSION_TOKEN"),
                        help="Tripletex session token")
    parser.add_argument("--tier", type=int, choices=[1, 2, 3],
                        help="Only run tasks of this tier")
    parser.add_argument("--filter", type=str,
                        help="Only run tasks whose name contains this string")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Show full response bodies")
    parser.add_argument("--no-log", action="store_true",
                        help="Skip writing log file")
    args = parser.parse_args()

    if not args.token:
        print("ERROR: Set TRIPLETEX_SESSION_TOKEN env var or pass --token")
        sys.exit(1)

    session = asyncio.run(run_all_tests(
        agent_url=args.url,
        base_url=args.base_url,
        session_token=args.token,
        tier_filter=args.tier,
        name_filter=args.filter,
        verbose=args.verbose,
    ))

    print_summary(session)

    if not args.no_log:
        save_log(session)

    # Exit code = number of failures (capped at 125)
    sys.exit(min(session.failed, 125))


if __name__ == "__main__":
    main()
