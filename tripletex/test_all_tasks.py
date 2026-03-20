"""Comprehensive test suite for ALL task types against the live Tripletex sandbox.
Run this BEFORE deploying to verify everything works."""

import asyncio
import json
import time
import httpx

# Sandbox credentials — load from env vars
import os
BASE_URL = os.environ.get("TRIPLETEX_BASE_URL", "https://kkpqfuj-amager.tripletex.dev/v2")
SESSION_TOKEN = os.environ["TRIPLETEX_SESSION_TOKEN"]  # Required

# Agent endpoint
AGENT_URL = "https://tripletex-agent-174612781810.europe-north1.run.app/solve"

# Test prompts for each task type — multilingual, realistic
TEST_PROMPTS = {
    # ===== TIER 1 =====
    "create_customer": "Opprett kunden Nordfjord AS med organisasjonsnummer 912345678. E-post: post@nordfjord.no. Telefon: 55667788. Adresse: Strandgata 12, 6800 Førde.",
    "create_employee": "Opprett ansatt Kari Nordmann med e-post kari.nordmann@test.no, mobilnummer 91234567, født 1985-06-15. Hun skal ha rollen som kontoadministrator.",
    "create_product": "Opprett produktet 'Konsulenttimer' med produktnummer 1234. Pris 1500 NOK eksklusiv MVA, standard MVA-sats 25%.",
    "create_department": "Opprett avdelingen 'Økonomi'.",
    "create_supplier": "Registrer leverandøren Kontorservice AS med organisasjonsnummer 987654321. E-post: faktura@kontorservice.no.",
    "create_contact": "Opprett kontaktperson Per Hansen for kunde Nordfjord AS. E-post: per@nordfjord.no, telefon 99887766.",
    "create_internal_project": "Opprett et internt prosjekt med navn 'Systemoppgradering'.",
    "send_invoice": "Send faktura nummer 1 til kunden via e-post.",
    "delete_travel_expense": "Slett reiseregning med ID 1.",

    # ===== TIER 2 =====
    "create_invoice": "Opprett faktura til kunde Nordfjord AS (org.nr 912345678) for 2 stk Konsulenttimer á 1500 kr. Fakturadato 2026-03-20, forfallsdato 2026-04-03.",
    "create_project": "Opprett prosjekt 'Websideutvikling' for kunde Nordfjord AS. Startdato 2026-03-20.",
    "create_travel_expense": "Registrer reiseregning for ansatt: Reise til Oslo 20. mars, taxi 350 kr.",
    "create_voucher": "Bokfør bilag: Kontorrekvisita 2500 kr, debet konto 6300, kredit konto 1920, dato 2026-03-20.",
    "update_employee": "Oppdater ansatt Kari Nordmann sin e-post til kari.ny@test.no.",
    "update_customer": "Oppdater kunde Nordfjord AS sin telefon til 55112233.",
    "create_supplier_invoice": "Registrer leverandørfaktura fra Kontorservice AS. Fakturanummer F-2026-001, dato 2026-03-20, forfallsdato 2026-04-20, beløp 5000 kr på konto 6300.",
    "create_credit_note": "Opprett kreditnota for faktura nummer 1.",

    # ===== TIER 2 (complex variants) =====
    "create_invoice_existing_customer": "Opprett faktura til eksisterende kunde Nordfjord AS for 1 stk Rådgivning á 3000 kr. Fakturadato 2026-03-20.",
    "register_payment": "Registrer innbetaling på 3000 kr på faktura ID 1, betalingsdato 2026-03-20.",
    "create_customer_supplier": "Opprett Havglimt AS (org.nr 876543210) som både kunde og leverandør. E-post: post@havglimt.no.",
    "create_project_existing_customer": "Opprett prosjekt 'Vedlikehold' for eksisterende kunde Nordfjord AS.",

    # ===== TIER 3 =====
    "create_invoice_with_payment": "Opprett faktura til kunde Strandvik AS (org.nr 111222333) for 1 stk Konsultasjon á 5000 kr, og registrer full betaling med en gang. Fakturadato 2026-03-20.",
    "create_opening_balance": "Sett åpningsbalanse per 2026-01-01: Bankkonto 1920 med 100000 kr, varelager konto 1400 med 50000 kr.",
}


async def test_task(prompt: str, task_name: str) -> dict:
    """Send a test task to the agent and return results."""
    payload = {
        "prompt": prompt,
        "tripletex_credentials": {
            "base_url": BASE_URL,
            "session_token": SESSION_TOKEN,
        },
    }

    start = time.monotonic()
    async with httpx.AsyncClient(timeout=320) as client:
        try:
            resp = await client.post(AGENT_URL, json=payload)
            elapsed = time.monotonic() - start
            return {
                "task": task_name,
                "status": resp.status_code,
                "response": resp.json() if resp.status_code == 200 else resp.text,
                "elapsed": round(elapsed, 1),
            }
        except Exception as e:
            elapsed = time.monotonic() - start
            return {
                "task": task_name,
                "status": "error",
                "response": str(e),
                "elapsed": round(elapsed, 1),
            }


async def run_tests(task_filter: str | None = None):
    """Run tests sequentially (to avoid sandbox state conflicts)."""
    prompts = TEST_PROMPTS
    if task_filter:
        prompts = {k: v for k, v in prompts.items() if task_filter in k}

    results = []
    total = len(prompts)

    for i, (task_name, prompt) in enumerate(prompts.items(), 1):
        print(f"\n[{i}/{total}] Testing {task_name}...")
        print(f"  Prompt: {prompt[:80]}...")
        result = await test_task(prompt, task_name)
        results.append(result)
        status = "OK" if result["status"] == 200 else f"FAIL ({result['status']})"
        print(f"  Result: {status} in {result['elapsed']}s")

    # Summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    passed = sum(1 for r in results if r["status"] == 200)
    failed = total - passed
    print(f"Passed: {passed}/{total}")
    print(f"Failed: {failed}/{total}")

    if failed > 0:
        print("\nFailed tasks:")
        for r in results:
            if r["status"] != 200:
                print(f"  - {r['task']}: {r['status']} ({r['elapsed']}s)")

    print("\nAll results:")
    for r in results:
        status = "PASS" if r["status"] == 200 else "FAIL"
        print(f"  [{status}] {r['task']}: {r['elapsed']}s")

    return results


if __name__ == "__main__":
    import sys
    task_filter = sys.argv[1] if len(sys.argv) > 1 else None
    asyncio.run(run_tests(task_filter))
