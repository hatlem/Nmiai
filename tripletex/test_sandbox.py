#!/usr/bin/env python3
"""Quick smoke test against a running agent (local or Cloud Run).

Usage:
  python3 test_sandbox.py                          # test localhost:8080
  python3 test_sandbox.py https://my-agent.run.app # test Cloud Run
"""
import sys
import json
import time
import httpx

AGENT_URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8080"

# These are fake credentials for testing the agent's plan generation only.
# The API calls will fail, but we can verify the agent doesn't crash.
FAKE_CREDS = {
    "base_url": "https://kkpqfuj-amager.tripletex.dev/v2",
    "session_token": "test-token-replace-me",
}

TEST_PROMPTS = [
    # Tier 1
    ("create_employee", "Opprett en ansatt med navn Ola Nordmann, e-post ola@example.com. Han skal vere kontoadministrator."),
    ("create_customer", "Registrer en ny kunde: Acme AS, org.nr 912345678, e-post post@acme.no"),
    ("create_product", "Opprett produkt 'Kontorstol Deluxe' med pris 4999 kr eks. mva"),
    ("create_department", "Opprett avdeling 'Salg' med avdelingsnummer 200"),
    ("create_supplier", "Registrer leverandor Staples Norge AS, e-post order@staples.no"),
    # Tier 2
    ("create_invoice", "Opprett faktura til kunde Bergen Bygg AS for 3 stk Murstein a 150 kr. Fakturadato 2026-03-20, forfall 2026-04-03."),
    ("create_project", "Opprett prosjekt 'Nettbutikk Redesign' for kunde Digital AS, start 2026-04-01, slutt 2026-06-30"),
    ("create_voucher", "Bokfor bilag: debet konto 1920 kr 10000, kredit konto 3000 kr 10000. Dato 2026-03-20. Beskrivelse: Innbetaling fra kunde."),
    ("create_travel_expense", "Registrer reiseregning: reise fra Oslo til Bergen 2026-03-15 til 2026-03-17. Formal: kundemote."),
    ("create_contact", "Opprett kontaktperson Kari Hansen, kari@acme.no, for kunde Acme AS"),
    # Multi-language
    ("create_employee", "Create an employee named John Smith, email john@example.com"),
    ("create_customer", "Erstellen Sie einen Kunden: Beispiel GmbH, E-Mail info@beispiel.de"),
    ("create_invoice", "Cree una factura para el cliente Madrid SL por 2 unidades de Servicio Premium a 500 EUR"),
]


def test_health():
    r = httpx.get(f"{AGENT_URL}/health", timeout=5)
    assert r.status_code == 200
    print(f"  /health: OK")


def test_solve(name: str, prompt: str):
    start = time.time()
    try:
        r = httpx.post(
            f"{AGENT_URL}/solve",
            json={"prompt": prompt, "files": [], "tripletex_credentials": FAKE_CREDS},
            timeout=300,
        )
        elapsed = time.time() - start
        data = r.json()
        status = data.get("status", "?")
        ok = r.status_code == 200 and status == "completed"
        print(f"  {name:40s} {'OK' if ok else 'FAIL':4s} {elapsed:5.1f}s  {r.status_code}")
        return ok
    except Exception as e:
        elapsed = time.time() - start
        print(f"  {name:40s} ERR  {elapsed:5.1f}s  {e}")
        return False


def main():
    print(f"Testing agent at {AGENT_URL}\n")

    print("Health check:")
    try:
        test_health()
    except Exception as e:
        print(f"  FAILED: {e}")
        print("  Is the agent running?")
        sys.exit(1)

    print(f"\nRunning {len(TEST_PROMPTS)} test prompts:")
    results = []
    for name, prompt in TEST_PROMPTS:
        ok = test_solve(name, prompt)
        results.append(ok)

    passed = sum(results)
    total = len(results)
    print(f"\nResults: {passed}/{total} passed")

    # Check stats
    try:
        stats = httpx.get(f"{AGENT_URL}/stats", timeout=5).json()
        print(f"\nAgent stats:")
        print(f"  Total: {stats['total']}")
        print(f"  Success: {stats['success']}")
        print(f"  Failed: {stats['failed']}")
        print(f"  API calls: {stats['total_api_calls']}")
        print(f"  Errors: {stats['total_errors']}")
        print(f"  Types seen: {len(stats['by_type'])}")
    except Exception:
        pass


if __name__ == "__main__":
    main()
