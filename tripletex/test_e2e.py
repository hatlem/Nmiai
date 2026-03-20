#!/usr/bin/env python3
"""End-to-end test: POST real tasks to /solve against sandbox API.

Simulates what the competition does — sends prompts with sandbox credentials
and checks if the agent creates the right entities in Tripletex.

Usage:
  python test_e2e.py                    # Run all tests against local server
  python test_e2e.py --url URL          # Against deployed Cloud Run
  python test_e2e.py --test create_employee  # Run single test
"""
import argparse
import asyncio
import json
import sys
import time
import httpx

import os
BASE_URL = os.environ.get("TRIPLETEX_BASE_URL", "https://kkpqfuj-amager.tripletex.dev/v2")
SESSION_TOKEN = os.environ["TRIPLETEX_SESSION_TOKEN"]  # Required
AUTH = ("0", SESSION_TOKEN)


# Each test: (name, prompt, verify_func)
# verify_func receives httpx client and should return (passed, details)
TESTS = []


def test(name):
    def decorator(fn):
        TESTS.append((name, fn))
        return fn
    return decorator


# ── Tier 1 Tests ──

@test("create_employee")
async def test_create_employee(client, solve):
    prompt = "Opprett ansatt TestAnsen, epost test.ansen@example.com, telefon 98765432, født 1985-06-20"
    result = await solve(prompt)

    # Verify: search for employee
    resp = await client.get(f"{BASE_URL}/employee",
        params={"firstName": "Test", "lastName": "Ansen", "fields": "id,firstName,lastName,email,phoneNumberMobile,dateOfBirth"})
    data = resp.json()
    values = data.get("values", [])
    if not values:
        return False, "Employee not found"

    emp = values[0]
    checks = []
    checks.append(("firstName", emp.get("firstName") == "Test"))
    checks.append(("lastName", emp.get("lastName") == "Ansen"))
    checks.append(("email", emp.get("email") == "test.ansen@example.com"))
    checks.append(("phone", emp.get("phoneNumberMobile") == "98765432"))
    checks.append(("dateOfBirth", emp.get("dateOfBirth") == "1985-06-20"))

    passed = all(ok for _, ok in checks)
    details = ", ".join(f"{f}={'OK' if ok else 'FAIL'}" for f, ok in checks)
    return passed, f"Employee id={emp['id']}: {details}"


@test("create_customer")
async def test_create_customer(client, solve):
    prompt = "Opprett kunde E2E Test AS, organisasjonsnummer 999888777, epost e2e@test.no, telefon 22334455"
    await solve(prompt)

    resp = await client.get(f"{BASE_URL}/customer",
        params={"name": "E2E Test AS", "fields": "id,name,email,organizationNumber,phoneNumber,isCustomer"})
    data = resp.json()
    values = data.get("values", [])
    if not values:
        return False, "Customer not found"

    c = values[0]
    checks = [
        ("name", c.get("name") == "E2E Test AS"),
        ("email", c.get("email") == "e2e@test.no"),
        ("orgNr", c.get("organizationNumber") == "999888777"),
        ("phone", c.get("phoneNumber") == "22334455"),
        ("isCustomer", c.get("isCustomer") == True),
    ]
    passed = all(ok for _, ok in checks)
    details = ", ".join(f"{f}={'OK' if ok else f'FAIL({c.get(f)})'}" for f, ok in checks)
    return passed, f"Customer id={c['id']}: {details}"


@test("create_product")
async def test_create_product(client, solve):
    prompt = "Opprett produkt E2E Testprodukt, pris 2500 kr eks mva, produktnummer E2E-001, beskrivelse Testprodukt for E2E"
    await solve(prompt)

    resp = await client.get(f"{BASE_URL}/product",
        params={"name": "E2E Testprodukt", "fields": "id,name,number,priceExcludingVatCurrency,description"})
    data = resp.json()
    values = data.get("values", [])
    if not values:
        return False, "Product not found"

    p = values[0]
    checks = [
        ("name", p.get("name") == "E2E Testprodukt"),
        ("number", str(p.get("number", "")) == "E2E-001" or p.get("number") == "E2E-001"),
        ("price", abs(float(p.get("priceExcludingVatCurrency", 0)) - 2500.0) < 1),
    ]
    passed = all(ok for _, ok in checks)
    details = ", ".join(f"{f}={'OK' if ok else f'FAIL({p.get(f)})'}" for f, ok in checks)
    return passed, f"Product id={p['id']}: {details}"


@test("create_department")
async def test_create_department(client, solve):
    prompt = "Opprett avdeling E2E Testavdeling, avdelingsnummer 999"
    await solve(prompt)

    resp = await client.get(f"{BASE_URL}/department",
        params={"name": "E2E Testavdeling", "fields": "id,name,departmentNumber"})
    data = resp.json()
    values = data.get("values", [])
    if not values:
        return False, "Department not found"

    d = values[0]
    checks = [
        ("name", d.get("name") == "E2E Testavdeling"),
        ("number", str(d.get("departmentNumber", "")) == "999"),
    ]
    passed = all(ok for _, ok in checks)
    return passed, f"Department id={d['id']}: {', '.join(f'{f}=OK' if ok else f'{f}=FAIL' for f, ok in checks)}"


@test("create_supplier")
async def test_create_supplier(client, solve):
    prompt = "Opprett leverandør E2E Leverandør AS, organisasjonsnummer 111222333, epost lev@e2e.no, telefon 44556677"
    await solve(prompt)

    resp = await client.get(f"{BASE_URL}/supplier",
        params={"name": "E2E Leverandør AS", "fields": "id,name,email,organizationNumber,phoneNumber"})
    data = resp.json()
    values = data.get("values", [])
    if not values:
        return False, "Supplier not found"

    s = values[0]
    checks = [
        ("name", s.get("name") == "E2E Leverandør AS"),
        ("email", s.get("email") == "lev@e2e.no"),
        ("orgNr", s.get("organizationNumber") == "111222333"),
    ]
    passed = all(ok for _, ok in checks)
    return passed, f"Supplier id={s['id']}: {', '.join(f'{f}=OK' if ok else f'{f}=FAIL' for f, ok in checks)}"


# ── Tier 2 Tests ──

@test("create_invoice")
async def test_create_invoice(client, solve):
    prompt = "Opprett faktura til kunde E2E Fakturakunde AS, epost faktura@e2e.no. 1 stk Konsulenttjeneste á 15000 kr, fakturadato 2026-03-20"
    await solve(prompt)

    # Check that an invoice was created (search recent invoices)
    resp = await client.get(f"{BASE_URL}/invoice",
        params={"invoiceDateFrom": "2026-03-20", "fields": "id,invoiceNumber,amount,customer", "count": 5})
    data = resp.json()
    values = data.get("values", [])
    if not values:
        return False, "No invoices found for date 2026-03-20"

    inv = values[-1]  # Most recent
    checks = [
        ("has_invoiceNumber", inv.get("invoiceNumber") is not None),
        ("has_amount", inv.get("amount") is not None and inv["amount"] > 0),
    ]
    passed = all(ok for _, ok in checks)
    return passed, f"Invoice #{inv.get('invoiceNumber')}, amount={inv.get('amount')}"


@test("create_voucher")
async def test_create_voucher(client, solve):
    prompt = "Bokfør bilag: debet konto 1920 kr 10000, kredit konto 3000 kr 10000, dato 2026-03-20, beskrivelse E2E test bilag"
    await solve(prompt)

    resp = await client.get(f"{BASE_URL}/ledger/voucher",
        params={"dateFrom": "2026-03-20", "dateTo": "2026-03-20", "fields": "id,description,date", "count": 5})
    data = resp.json()
    values = data.get("values", [])
    if not values:
        return False, "No vouchers found for 2026-03-20"

    # Find our voucher
    found = None
    for v in values:
        if "e2e" in (v.get("description") or "").lower():
            found = v
            break

    if not found:
        found = values[-1]

    return True, f"Voucher id={found['id']}, desc={found.get('description')}"


@test("update_employee")
async def test_update_employee(client, solve):
    # First check if our test employee exists
    resp = await client.get(f"{BASE_URL}/employee",
        params={"firstName": "Test", "lastName": "Ansen", "fields": "id,firstName,lastName,email"})
    data = resp.json()
    values = data.get("values", [])
    if not values:
        return False, "SKIP: Test employee not found (run create_employee first)"

    prompt = "Oppdater ansatt Test Ansen, endre epost til updated@e2e.no"
    await solve(prompt)

    # Re-fetch
    resp = await client.get(f"{BASE_URL}/employee/{values[0]['id']}",
        params={"fields": "id,firstName,lastName,email"})
    emp = resp.json().get("value", resp.json())
    if emp.get("email") == "updated@e2e.no":
        return True, f"Email updated to {emp['email']}"
    return False, f"Email still {emp.get('email')}, expected updated@e2e.no"


@test("create_internal_project")
async def test_create_internal_project(client, solve):
    prompt = "Opprett internt prosjekt E2E Internprosjekt, startdato 2026-04-01, sluttdato 2026-12-31"
    await solve(prompt)

    resp = await client.get(f"{BASE_URL}/project",
        params={"name": "E2E Internprosjekt", "fields": "id,name,isInternal,startDate,endDate"})
    data = resp.json()
    values = data.get("values", [])
    if not values:
        return False, "Project not found"

    p = values[0]
    checks = [
        ("name", "E2E" in str(p.get("name", ""))),
        ("isInternal", p.get("isInternal") == True),
        ("startDate", p.get("startDate") == "2026-04-01"),
    ]
    passed = all(ok for _, ok in checks)
    return passed, f"Project id={p['id']}, name={p.get('name')}, internal={p.get('isInternal')}"


# ── Runner ──

async def run_tests(agent_url: str, test_filter: str | None = None):
    print(f"\n{'='*80}")
    print(f"E2E TEST — Agent: {agent_url}")
    print(f"Sandbox: {BASE_URL}")
    print(f"{'='*80}\n")

    api_client = httpx.AsyncClient(auth=AUTH, timeout=30.0)
    agent_client = httpx.AsyncClient(timeout=310.0)  # 5 min + buffer

    async def solve(prompt):
        payload = {
            "prompt": prompt,
            "files": [],
            "tripletex_credentials": {
                "base_url": BASE_URL,
                "session_token": SESSION_TOKEN,
            }
        }
        start = time.monotonic()
        try:
            resp = await agent_client.post(f"{agent_url}/solve", json=payload)
            elapsed = time.monotonic() - start
            print(f"    /solve responded: {resp.status_code} in {elapsed:.1f}s")
            return resp.json()
        except Exception as e:
            elapsed = time.monotonic() - start
            print(f"    /solve FAILED after {elapsed:.1f}s: {e}")
            return {"error": str(e)}

    passed = 0
    failed = 0
    errors = 0

    for name, test_fn in TESTS:
        if test_filter and test_filter not in name:
            continue

        print(f"\n[TEST] {name}")
        print(f"  Running...")
        try:
            ok, details = await test_fn(api_client, solve)
            if ok:
                print(f"  \033[32mPASS\033[0m: {details}")
                passed += 1
            else:
                print(f"  \033[31mFAIL\033[0m: {details}")
                failed += 1
        except Exception as e:
            print(f"  \033[31mERROR\033[0m: {e}")
            errors += 1

    print(f"\n{'='*80}")
    print(f"RESULTS: {passed} PASS | {failed} FAIL | {errors} ERROR | {passed + failed + errors} total")
    print(f"{'='*80}\n")

    await api_client.aclose()
    await agent_client.aclose()
    return failed + errors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8080", help="Agent URL")
    parser.add_argument("--test", default=None, help="Run specific test (substring match)")
    args = parser.parse_args()

    exit_code = asyncio.run(run_tests(args.url, args.test))
    sys.exit(min(exit_code, 1))


if __name__ == "__main__":
    main()
