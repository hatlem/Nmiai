"""Test all 30 task types against our agent locally."""
import asyncio
import httpx
import json
import time
import sys

AGENT_URL = "https://tripletex-agent-174612781810.europe-north1.run.app/solve"
SANDBOX_URL = "https://kkpqfuj-amager.tripletex.dev/v2"
SANDBOX_TOKEN = "eyJ0b2tlbklkIjoyMTQ3NjM4ODA2LCJ0b2tlbiI6ImNlZGY2OWJjLWQ4NTEtNGM1ZS1hMmUxLTJjMDU1YjNmYzQzMCJ9"

# Representative prompts for each task type (no PDF tasks — can't test those)
TASKS = [
    ("create_employee", "Opprett en ansatt med navn Test Testersen, e-post test.testersen99@example.org. Fødselsdato 15. januar 1990. Telefon 99887711. Han skal være kontoadministrator."),
    ("create_customer", "Opprett kunden TestFirma AS med org.nr 999111222. E-post: post@testfirma.no. Adresse: Testveien 1, 0123 Oslo."),
    ("create_product", "Opprett produktet 'Testprodukt' med produktnummer 9999. Prisen er 15000 kr eksklusiv MVA, standard MVA-sats 25%."),
    ("create_supplier", "Registrer leverandøren TestLev AS med organisasjonsnummer 888111222. E-post: faktura@testlev.no."),
    ("create_department", "Opprett tre avdelinger i Tripletex: 'TestAvd1', 'TestAvd2' og 'TestAvd3'."),
    ("create_invoice", "Opprett en faktura til kunden TestKunde AS (org.nr 777111222) med en produktlinje: Konsulenttime til 1500 kr med standard MVA 25%."),
    ("create_project", "Opprett prosjektet 'Testprosjekt' knyttet til kunden TestProsjektKunde AS (org.nr 666111222). Prosjektleder er Ola Test, ola.test99@example.org."),
    ("create_voucher", "Bokfør et bilag: Debet konto 6300 med 5000 kr, kredit konto 1920 med 5000 kr. Beskrivelse: Testkostnad."),
    ("create_dimensions_voucher", "Opprett en fri regnskapsdimensjon 'TestDim' med verdiene 'Verdi1' og 'Verdi2'. Bokfør et bilag på konto 6300 (debet 3000 kr) og konto 1920 (kredit 3000 kr) knyttet til dimensjonsverdien 'Verdi1'."),
    ("create_travel_expense", "Registrer en reiseregning for Test Reisende (test.reisende99@example.org) for 'Kundebesøk Bergen'. Reisen varte 2 dager med avreise fra Oslo. Flybillett 3500 kr."),
    ("credit_note", "Kunden TestKreditKunde AS (org.nr 555111222) har reklamert på fakturaen for 'Konsulenttime' (10000 kr ekskl. MVA). Opprett en full kreditnota."),
    ("create_supplier_invoice", "Vi har mottatt faktura INV-TEST-001 fra leverandøren TestLevFakt AS (org.nr 444111222) på 25000 kr inklusiv MVA. Beløpet gjelder kontorrekvisita (konto 6540)."),
    ("salary", "Kjør lønn for Test Lansen (test.lansen99@example.org) for denne måneden. Grunnlønn er 42000 kr. Legg til en engangsbonus på 8000 kr."),
    ("month_end_closing", "Utfør månedsavslutning for mars 2026. Periodiser forskuddsbetalt kostnad (5000 kr per måned fra konto 1720 til kostkonto 6300). Bokfør avskrivning for kontormaskiner (120000 kr over 5 år, konto 6010/1209)."),
    ("overdue_reminder", "En av kundene dine har en forfalt faktura. Finn den forfalte fakturaen og bokfør et purregebyr på 50 kr. Debet kundefordringer 1500, kredit purregebyr 3400."),
    ("currency_exchange", "Vi sendte en faktura på 5000 EUR til TestValuta AS (org.nr 333111222) da kursen var 11.50 NOK/EUR. Kunden har nå betalt, men kursen var 11.20 NOK/EUR. Registrer betalingen og bokfør kurstapet."),
    ("reverse_payment", "Betalinga frå TestRevers AS (org.nr 222111222) for fakturaen 'Konsulenttimer' (20000 kr ekskl. MVA) vart returnert av banken. Reverser betalinga og opprett ein ny faktura."),
    ("fixed_price_project", "Sett fastpris 300000 kr på prosjektet 'TestFastpris' for TestFP AS (org.nr 111222333). Prosjektleder er Test PM, test.pm99@example.org. Fakturer 50%."),
    ("timesheet", "Registrer 8 timer for Test Timer (test.timer99@example.org) på aktiviteten 'Utvikling' i prosjektet 'Tidsregistrering' for TestTid AS (org.nr 111333444)."),
    ("update_employee", "Vi har en ansatt som heter Test Testersen med e-post test.testersen99@example.org. Oppdater telefonnummeret til 44556677."),
    ("update_customer", "Oppdater kunden TestFirma AS. Endre e-post til ny@testfirma.no."),
    ("create_contact", "Opprett kontaktperson Per Kontakt for kunden TestFirma AS. E-post: per.kontakt@testfirma.no, telefon 11223344."),
    ("invoice_with_payment", "Opprett en faktura til kunden TestBetal AS (org.nr 111555666) for Konsulenttimer til 25000 kr ekskl. MVA. Registrer betaling umiddelbart."),
    ("send_invoice", "Opprett og send en faktura til kunden TestSend AS (org.nr 111666777) på 15000 kr eksklusiv MVA. Fakturaen gjelder Nettverkstjeneste. Send via e-post."),
    ("year_end_closing", "Gjør forenklet årsoppgjør for 2025: 1) Beregn og bokfør årlige avskrivninger for to eiendeler: Kontormaskiner (150000 kr, 5 år, konto 6010/1209) og Kjøretøy (200000 kr, 4 år, konto 6010/1209). 2) Reverser forskuddsbetalte utgifter (24000 kr fra konto 1700 til konto 6300). 3) Beregn skatteavsetning på 22% av skattbart resultat 100000 kr."),
    ("ledger_analysis", "Totalkostnadene økte betydelig fra januar til februar 2026. Analyser hovedboken og finn de tre kostnadskontoene med størst økning."),
    ("voucher_correction", "Vi har oppdaget feil i hovedboken for januar og februar 2026. Gå gjennom alle bilag og finn de 4 feilene: en postering på feil konto, et duplikatbilag, en manglende MVA-postering, og et feil beløp. Korriger alle feilene."),
    ("project_lifecycle", "Gjennomfør hele prosjektsyklusen for 'TestSyklus' (TestSyklusKunde AS, org.nr 111777888): 1) Prosjektet har budsjett på 500000 kr. 2) Registrer 40 timer à 1200 kr. 3) Fakturer kunden for timene."),
    ("create_purchase_order", "Opprett en innkjøpsordre til leverandøren TestInnkjøp AS (org.nr 111888999). Bestill Kontorrekvisita til 5000 kr."),
    ("create_opening_balance", "Bokfør åpningsbalanse: Bankkonto (1920) 500000 kr, Kundefordringer (1500) 100000 kr, Egenkapital (2000) -600000 kr."),
]

async def test_task(client: httpx.AsyncClient, name: str, prompt: str) -> dict:
    """Test a single task against the agent."""
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
        return {
            "task": name,
            "status": resp.status_code,
            "time": f"{elapsed:.1f}s",
            "response": resp.json() if resp.status_code == 200 else resp.text[:200],
        }
    except Exception as e:
        elapsed = time.monotonic() - start
        return {"task": name, "status": "ERROR", "time": f"{elapsed:.1f}s", "error": str(e)[:200]}


async def main():
    # Run tasks specified by index, or all
    indices = [int(x) for x in sys.argv[1:]] if len(sys.argv) > 1 else range(len(TASKS))

    async with httpx.AsyncClient() as client:
        for i in indices:
            name, prompt = TASKS[i]
            print(f"\n[{i}] Testing: {name}...", flush=True)
            result = test_task(client, name, prompt)
            # Run sequentially to not overwhelm
            r = await result
            status = "✓" if r.get("status") == 200 else "✗"
            print(f"  {status} {r['task']}: {r['time']} — {r.get('response', r.get('error', '?'))}")


if __name__ == "__main__":
    asyncio.run(main())
