#!/usr/bin/env python3
"""Quick classification test — no external dependencies needed.

Usage:
  python test_local.py           # Test keyword classification
  python test_local.py --full    # Full test with LLM (requires vertexai)
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from templates import TEMPLATES, KEYWORD_HINTS

# ── Re-implement _quick_classify locally to avoid vertexai import ──

# Read TIER_MAP and high_conf_keywords from agent.py source
_agent_src = (Path(__file__).parent / "agent.py").read_text()

# Extract TIER_MAP
_tier_block = re.search(r'TIER_MAP:\s*dict\[str,\s*int\]\s*=\s*\{([^}]+)\}', _agent_src, re.DOTALL)
TIER_MAP = {}
if _tier_block:
    for m in re.finditer(r'"([^"]+)":\s*(\d+)', _tier_block.group(1)):
        TIER_MAP[m.group(1)] = int(m.group(2))


def _quick_classify(prompt: str):
    """Keyword-based classification (copied logic from agent.py)."""
    prompt_lower = prompt.lower()

    # Payment + invoice number pattern
    _has_payment = bool(re.search(r'\b(betal|betaling|innbetaling|payment|paiement|zahlung|pago)\b', prompt_lower))
    _has_invoice_number = bool(re.search(
        r'(faktura\s*(nr|nummer|#)\s*\d+|invoice\s*(nr|number|#|no\.?)\s*\d+)',
        prompt_lower,
    ))
    if _has_payment and _has_invoice_number:
        return "register_payment_by_search", 0.92

    # Extract high_conf_keywords from agent.py source
    # Parse the dict between "high_conf_keywords = {" and the closing "}"
    hck_start = _agent_src.find("high_conf_keywords = {")
    if hck_start >= 0:
        depth = 0
        for i in range(hck_start + 21, len(_agent_src)):
            if _agent_src[i] == '{':
                depth += 1
            elif _agent_src[i] == '}':
                if depth == 0:
                    break
                depth -= 1
        hck_block = _agent_src[hck_start:i+1]
        # Parse entries
        for m in re.finditer(r'"([^"]+)":\s*\("([^"]+)",\s*([\d.]+)\)', hck_block):
            phrase, task_type, conf = m.group(1), m.group(2), float(m.group(3))
            if phrase in prompt_lower:
                return task_type, conf

    # Keyword hints fallback
    best_type = None
    best_len = 0
    second_best_len = 0
    for task_type, keywords in KEYWORD_HINTS.items():
        for kw in keywords:
            if kw.lower() in prompt_lower:
                if len(kw) > best_len:
                    second_best_len = best_len
                    best_type = task_type
                    best_len = len(kw)
                elif len(kw) > second_best_len:
                    second_best_len = len(kw)

    if best_type and best_len > second_best_len + 2:
        conf = 0.75
        if best_len >= 8:
            conf = 0.70
        return best_type, conf
    if best_type and best_len >= 5:
        return best_type, 0.65
    return None


# ── Test prompts ──

TEST_PROMPTS = [
    # Tier 1
    ("create_employee", "Opprett ansatt Kari Nordmann, kari@test.no, tlf 99887766, født 1990-05-15"),
    ("create_customer", "Opprett kunde Bergen Bygg AS, org.nr 987654321, epost post@bergenbygg.no, tlf 55443322"),
    ("create_product", "Opprett produkt Kontorpult, pris 4999 kr eks mva, produktnummer P-001"),
    ("create_department", "Opprett avdeling Salg, avdelingsnummer 200"),
    ("create_supplier", "Opprett leverandør Stavanger Stål AS, org.nr 123456789, epost post@staal.no"),
    ("create_contact", "Opprett kontaktperson Per Hansen, epost per@test.no, for kunde Bergen Bygg AS"),
    ("create_customer_supplier", "Opprett Bergen Bygg AS som både kunde og leverandør, org.nr 987654321"),
    # Tier 2
    ("create_invoice", "Fakturér kunde Norsk Data AS for 2 stk Laptop á 12000 kr, fakturadato 2026-03-15"),
    ("create_invoice_existing_customer", "Opprett faktura for eksisterende kunde Norsk Data AS, 3 stk Monitor á 5000 kr"),
    ("create_invoice_with_payment", "Opprett faktura med betaling for kunde TestAS, 1 stk Vare á 1000 kr"),
    ("register_payment", "Registrer innbetaling på 24000 kr, betalingsdato 2026-03-20"),
    ("register_payment_by_search", "Registrer betaling på 24000 kr på faktura nr 10001, betalingsdato 2026-03-20"),
    ("create_credit_note", "Opprett kreditnota for faktura 10001, dato 2026-03-18"),
    ("send_invoice", "Send faktura 10001 til kunden via epost"),
    ("create_travel_expense", "Opprett reiseregning: Oslo til Bergen, 15.-17. mars 2026, forretningsreise"),
    ("delete_travel_expense", "Slett reiseregning 42"),
    ("deliver_travel_expense", "Lever reiseregning 42 for godkjenning"),
    ("approve_travel_expense", "Godkjenn reiseregning 42"),
    ("create_project", "Opprett prosjekt Nettside-redesign for kunde DigitalByrå AS, startdato 2026-04-01"),
    ("create_project_existing_customer", "Opprett prosjekt for eksisterende kunde Acme AS, startdato 2026-04-01"),
    ("create_internal_project", "Opprett internt prosjekt Infrastruktur-oppgradering, startdato 2026-04-01"),
    ("update_project", "Oppdater prosjekt Nettside-redesign, endre sluttdato til 2026-09-30"),
    ("create_voucher", "Bokfør bilag: debet konto 1920 kr 5000, kredit konto 3000 kr 5000, dato 2026-03-15"),
    ("reverse_voucher", "Reverser bilag 123, dato 2026-03-18"),
    ("update_employee", "Oppdater ansatt Kari Nordmann, endre epost til kari.nordmann@nyepost.no"),
    ("update_customer", "Oppdater kunde Bergen Bygg AS, endre telefonnummer til 55112233"),
    ("update_supplier", "Oppdater leverandør Stavanger Stål AS, endre epost til ny@staal.no"),
    ("update_department", "Oppdater avdeling Salg, endre navn til Salg og Marked"),
    ("update_product", "Oppdater produkt Kontorpult, endre pris til 5499 kr"),
    ("create_supplier_invoice", "Registrer leverandørfaktura fra Stavanger Stål AS, beløp 15000 kr, fakturanr F-2026-001"),
    ("create_purchase_order", "Opprett innkjøpsordre til Stavanger Stål AS, levering 2026-04-01"),
    ("create_timesheet_entry", "Registrer 7.5 timer på prosjekt Nettside-redesign, dato 2026-03-19"),
    ("create_reminder", "Send purring for faktura 10001"),
    ("create_employment", "Registrer ansettelse for Kari Nordmann, startdato 2026-04-01, 100% stilling"),
    ("create_salary_payment", "Registrer lønnsutbetaling for mars 2026"),
    # Tier 3
    ("create_opening_balance", "Opprett åpningsbalanse: konto 1920 kr 100000, konto 1500 kr 50000, dato 2026-01-01"),
    ("bank_reconciliation", "Gjennomfør bankavstemming for mars 2026, konto 1920"),
    ("create_asset", "Registrer anleggsmiddel: Kontormaskin, anskaffelseskost 45000, dato 2026-03-01"),
    # Multi-language
    ("create_employee", "Create employee John Smith, john@example.com, phone +447911123456"),
    ("create_customer", "Erstellen Sie einen Kunden: Berliner Bäckerei GmbH, org.nr 112233445"),
    ("create_invoice", "Créer une facture pour le client Paris Consulting, 1 article Service á 5000 EUR"),
    ("create_employee", "Crear empleado Carlos Garcia, email carlos@test.es, teléfono 612345678"),
    ("create_supplier", "Criar fornecedor Lisboa Tech Lda, email info@lisboatech.pt"),
    # Edge cases
    ("create_employee", "Ny tilsett Olav Haugen, olav@test.no"),  # Nynorsk
    ("create_invoice_with_payment", "Faktura og registrer betaling for kunde TestFirma AS, 1 stk Konsulenttime á 1500"),
    ("create_project_existing_customer", "Prosjekt til eksisterende kunde Acme Corp"),
]


def main():
    print("=" * 90)
    print("TRIPLETEX KEYWORD CLASSIFICATION TEST")
    print(f"Testing {len(TEST_PROMPTS)} prompts against _quick_classify")
    print("=" * 90)

    passed = 0
    failed = 0
    no_match = 0

    for expected, prompt in TEST_PROMPTS:
        result = _quick_classify(prompt)
        tier = TIER_MAP.get(expected, 3)

        if result is None:
            status = "\033[33mNONE \033[0m"
            no_match += 1
            print(f"  [{status}] T{tier} {expected:40s} <- {prompt[:55]}")
        elif result[0] == expected:
            status = "\033[32mOK   \033[0m"
            passed += 1
            print(f"  [{status}] T{tier} {expected:40s} (conf={result[1]:.2f}) <- {prompt[:55]}")
        else:
            status = "\033[31mWRONG\033[0m"
            failed += 1
            print(f"  [{status}] T{tier} {expected:40s} got {result[0]:30s} (conf={result[1]:.2f})")
            print(f"           Prompt: {prompt}")

    print()
    print("=" * 90)
    print(f"RESULTS: {passed} OK | {failed} WRONG | {no_match} NO MATCH | {len(TEST_PROMPTS)} total")

    threshold = 0.55
    needs_llm = no_match
    needs_pro = 0
    for expected, prompt in TEST_PROMPTS:
        result = _quick_classify(prompt)
        if result is None or result[1] < threshold:
            needs_llm += 1
            if result and result[1] < 0.45:
                needs_pro += 1

    print(f"Would need Flash-Lite: {needs_llm}/{len(TEST_PROMPTS)} prompts")
    print(f"Would need Pro escalation: {needs_pro}/{len(TEST_PROMPTS)} prompts")
    print("=" * 90)

    if failed > 0:
        print("\nFIX THESE MISCLASSIFICATIONS FIRST!")
        return 1
    if no_match > 3:
        print(f"\n{no_match} prompts need LLM — consider adding more keywords")
    return 0


if __name__ == "__main__":
    sys.exit(main())
