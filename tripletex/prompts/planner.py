"""Extraction-only prompts — LLM extracts values, never generates API steps."""

import json
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from templates import TEMPLATES

GLOSSARY = """\
Faktura/Rechnung/Facture/Factura=Invoice, Kreditnota=Credit note, Innbetaling/Betaling=Payment
Kunde/Kundin/Client/Cliente=Customer, Leverandør/Lieferant/Fournisseur=Supplier
Ansatt/Tilsett/Angestellte/Employé/Empleado/Empregado=Employee
Produkt=Product, Prosjekt/Projekt/Projet/Proyecto=Project, Avdeling/Abteilung=Department
Reiseregning/Reiserekning=Travel expense, Bilag/Beleg=Voucher, Mva/MwSt/TVA/IVA=VAT
Forfallsdato/Fälligkeitsdatum=Due date, Organisasjonsnummer=Org number
Kontoadministrator=ALL_PRIVILEGES, Regnskapsfører/Rekneskapsførar=ACCOUNTANT
Lønnansvarlig=PERSONELL_MANAGER, Fakturaansvarlig=INVOICING_MANAGER
Revisor/Auditeur=AUDITOR, Avdelingsleder=DEPARTMENT_LEADER
Fastpris/Festpreis/Prix forfaitaire/Precio fijo/Fixed price/Preço fixo=isFixedPrice+fixedprice on Project
Facturez X%/Fakturez X%/Invoice X%=invoicePercentage (partial invoice of fixed price)
Nynorsk: tilsett=ansatt, verksemd=virksomhet, reknskap=regnskap"""


def build_extraction_prompt(task_type: str, tier: int = 1) -> str:
    """Build extraction prompt for a known task type. Returns ~30 line system prompt."""
    template = TEMPLATES.get(task_type, TEMPLATES["unknown"])

    if task_type == "unknown":
        return _build_unknown_prompt()

    fields = template.get("extract_fields", [])
    conditional_fields = ""
    if template.get("conditional_steps"):
        cond = [k.removeprefix("if_") for k in template["conditional_steps"]]
        conditional_fields = f"\nAlso extract if mentioned: {json.dumps(cond)}"
        conditional_fields += "\nEntitlement values: ALL_PRIVILEGES, INVOICING_MANAGER, PERSONELL_MANAGER, ACCOUNTANT, AUDITOR, DEPARTMENT_LEADER"

    return f"""You extract values from accounting task prompts. Return ONLY a JSON object.

Task type: {task_type}
Description: {template["description"]}

Fields to extract: {json.dumps(fields)}{conditional_fields}
IMPORTANT: Extract ONLY the fields listed above. Do NOT invent extra fields like employmentType, percentageOfFullTimeEquivalent, userType (on employment), type, description (on timesheet), sendType, sendMethod, or status. These cause 422 errors.

Glossary:
{GLOSSARY}

Rules:
- Output ONLY a valid JSON object. No markdown fences, no explanation.
- Keys must match field names above exactly.
- Dates as YYYY-MM-DD. "today" or unspecified = {date.today().isoformat()}.
- Amounts as numbers (1500.00 not "1500.00"). Never calculate VAT yourself.
- "ekskl. mva" -> use amount as priceExcludingVatCurrency. "inkl. mva" -> priceIncludingVatCurrency.
- VAT-inclusive amounts (supplier invoices/vouchers): If the amount is TTC/inkl mva/inkl. mva/including VAT/brutto/IVA inclusa/com IVA/inkl. MwSt/mit MwSt → set amount_is_gross=true and amount to the FULL gross amount. The template engine calculates the net.
- VAT-exclusive amounts: If the amount is HT/ekskl mva/ekskl. mva/excluding VAT/netto/hors taxes/exkl. MwSt/sin IVA/sem IVA → set amount_is_gross=false (or omit).
- Default: If no VAT indication is given, assume amount_is_gross=false (amount is net/excluding VAT).
- Booleans as true/false.
- Preserve special chars exactly: Ø, Æ, Å, ñ, ü, etc.
- Phone numbers: preserve as-is from prompt. "telefon"/"tlf"/"mobil" -> phoneNumberMobile for employees, phoneNumber for customers.
- Addresses: extract addressLine1, postalCode, city as separate keys.
- For update tasks: put changed fields in a "fields_to_update" dict.
- For orderLines: array of {{"description": "...", "count": N, "unitPriceExcludingVatCurrency": N, "productNumber": "..." (if mentioned)}}.
  IMPORTANT: If the prompt mentions a product number (e.g. "Konsulenttimar (1874)"), include "productNumber": "1874" in that orderLine.
  The number in parentheses IS the product number. Example: "produkta Webdesign (5678) til 3000 kr" -> {{"description": "Webdesign", "productNumber": "5678", "count": 1, "unitPriceExcludingVatCurrency": 3000}}
- For voucher/opening balance: "accounts" list of {{"number": "1920", "amount": 100000}} (positive=debit, negative=credit).
- If the prompt mentions a "fri regnskapsdimensjon" or "accounting dimension", extract: dimension_name (the dimension name, e.g. "Kostsenter"), dimension_values (list of value names, e.g. ["Økonomi", "Kundeservice"]), dimension_link_value (which value to link the voucher posting to, e.g. "Kundeservice").
- If files attached, extract ALL data from them (every line, amount, account).
- Omit fields not mentioned in the prompt. But NEVER omit fields that ARE mentioned — every data point in the prompt MUST appear in the output.
- CRITICAL: Every field mentioned in the prompt MUST be extracted. Missing fields = lost points.
  A prompt like "org.nr 912345678" -> organizationNumber: "912345678"
  A prompt like "telefon 55112233" -> phoneNumber: "55112233" (customer) or phoneNumberMobile: "55112233" (employee)
  A prompt like "født 1990-05-15" -> dateOfBirth: "1990-05-15"
  A prompt like "adresse Strandgata 12, 6800 Førde" -> addressLine1: "Strandgata 12", postalCode: "6800", city: "Førde"

Example:
Task: "Opprett ansatt Kari Nordmann, kari@test.no, tlf 99887766, født 1990-05-15, Storgata 1, 0123 Oslo"
Output: {{"firstName": "Kari", "lastName": "Nordmann", "email": "kari@test.no", "phoneNumberMobile": "99887766", "dateOfBirth": "1990-05-15", "addressLine1": "Storgata 1", "postalCode": "0123", "city": "Oslo"}}

SCORING: Every field in the prompt that you miss = lost points. The scorer checks EVERY mentioned detail.

FIELD EXTRACTION CHECKLIST — scan the prompt for ALL of these:
- Name/company name: ALWAYS extract (name, firstName+lastName)
- organizationNumber / org.nr / org.nº / Organisationsnummer: 9-digit number
- email / e-post / epost / E-Mail / correo: email address
- phoneNumber / telefon / tlf / mobil / Telefon / teléfono: phone number
- dateOfBirth / født / geboren / nacido: birth date
- addressLine1 + postalCode + city: ANY address mentioned → extract ALL 3 parts
- description / beskrivelse: ANY description or note text
- departmentNumber / avdelingsnummer: department number
- number / produktnummer / number: product/item number
- startDate + endDate: project dates
- role: ALL_PRIVILEGES/ACCOUNTANT/INVOICING_MANAGER/PERSONELL_MANAGER/AUDITOR/DEPARTMENT_LEADER
- invoiceDueDate / forfallsdato: if not given, calculate as invoiceDate + 14 days
- deliveryDate / leveringsdato: if not given, use orderDate
- departureFrom / fra / from: departure city for travel
- title: travel expense title (use purpose if not explicit)
- costs: array of {{"amount": N, "description": "..."}} — extract ALL expense items including per diem.
  Per diem (diett/dagpenger): calculate total = daily_rate × days, add as cost item.
  Example: "3 days per diem 800 NOK, flight 2800, taxi 450" →
  costs: [{{"description": "Per diem (3 days x 800)", "amount": 2400}}, {{"description": "Flight ticket", "amount": 2800}}, {{"description": "Taxi", "amount": 450}}]
- firstName, lastName, email: ALWAYS extract for travel expenses. The prompt names a specific person who must be created as employee.
  Example: "for Alice Clark (alice.clark@example.org)" → firstName: "Alice", lastName: "Clark", email: "alice.clark@example.org"

COMMON EXTRACTION MISTAKES TO AVOID:
- "org.nr 912345678" → organizationNumber: "912345678" (NOT "org.nr 912345678")
- "tlf 55112233" → phoneNumber: "55112233" or phoneNumberMobile: "55112233"
- "Strandgata 12, 6800 Førde" → addressLine1: "Strandgata 12", postalCode: "6800", city: "Førde"
- "avdelingsnummer 200" → departmentNumber: "200" (string, not int)
- "kontoadministrator" → role: "ALL_PRIVILEGES"
- "pris 4999 kr eks mva" → priceExcludingVatCurrency: 4999

TRAVEL EXPENSE EXTRACTION:
- Multiple costs: "fly 2800 og taxi 450" → costs: [{{"amount": 2800, "description": "fly"}}, {{"amount": 450, "description": "taxi"}}]
- Per diem keywords: "diett", "dagpenger", "per diem", "daily rate", "daily allowance", "kostgodtgjørelse", "Tagegeld", "dieta"
- Per diem with explicit rate: extract BOTH as cost item AND as perDiem: {{"dailyRate": 780, "days": 3}}
  This enables the Tripletex perDiemCompensations API for proper per diem tracking.
- Employee name: "for Alice Clark (alice.clark@example.org)" → firstName: "Alice", lastName: "Clark", email: "alice.clark@example.org"
- Single cost (no array needed): "hotell 1200 kr" → cost_amount: 1200, cost_description_if_any: "hotell\""""


def build_repair_extraction_prompt(task_type: str, original_prompt: str, errors: list[dict]) -> str:
    """Build re-extraction prompt when execution failed due to bad/missing values."""
    template = TEMPLATES.get(task_type, TEMPLATES["unknown"])
    fields = template.get("extract_fields", [])

    return f"""A Tripletex accounting task failed. Re-extract the values considering the errors below.
Return ONLY a corrected JSON object with the extracted values.

Task type: {task_type}
Description: {template["description"]}
Fields to extract: {json.dumps(fields)}

Original task prompt:
{original_prompt}

Errors from execution:
{json.dumps(errors, indent=2, ensure_ascii=False)}

Rules:
- Fix the values that caused errors. Check field names, formats, missing fields.
- Dates as YYYY-MM-DD. Today = {date.today().isoformat()}.
- Amounts as numbers. Preserve special chars (Ø, Æ, Å).
- For update tasks: put changed fields in "fields_to_update" dict.
- Do NOT add fields that don't exist on the API endpoint (common 422 causes):
  * Employment: ONLY employee.id, startDate, endDate, division. NOT employmentType/percentageOfFullTimeEquivalent/userType/type
  * Timesheet: ONLY employee, project, activity, date, hours, comment. NOT description
  * PurchaseOrder: ONLY supplier.id, ourContact.id, deliveryDate. NOT status/currency/orderLines
  * createReminder: use dispatchType (NOT sendType/sendMethod)
- Output ONLY a valid JSON object. No markdown, no explanation."""


def _build_unknown_prompt() -> str:
    """For unknown/complex tasks, provide full API knowledge for free planning."""
    schemas = _get_relevant_schemas("unknown")
    return f"""You are an expert Tripletex accounting agent. Plan the EXACT API calls needed.

{GLOSSARY}

## Available API Endpoints
GET/POST /employee — create/search employees. POST body: firstName, lastName, email, phoneNumberMobile, dateOfBirth, userType ("STANDARD"), department ({{id}})
GET/POST /customer — POST body: name, isCustomer (true), email, organizationNumber, phoneNumber, postalAddress
GET/POST /supplier — POST body: name, email, organizationNumber
GET/POST /product — POST body: name, number, priceExcludingVatCurrency
GET/POST /department — POST body: name, departmentNumber
GET/POST /project — POST body: name, customer ({{id}}), projectManager ({{id}}), startDate, endDate, isInternal, isFixedPrice, fixedprice
POST /project/projectActivity — POST body: project ({{id}}), activity ({{id}})
GET/POST /order — POST body: customer ({{id}}), orderDate, deliveryDate, orderLines (array)
OrderLine in order body: description, count, unitPriceExcludingVatCurrency, product ({{id}}) [optional]
PUT /order/{{id}}/:invoice — query params: invoiceDate, invoiceDueDate, sendToCustomer
PUT /invoice/{{id}}/:payment — query params: paymentDate, paymentTypeId, paidAmount
PUT /invoice/{{id}}/:createCreditNote — query params: date, comment
PUT /invoice/{{id}}/:createReminder — query params: type, date, dispatchType (EMAIL)
PUT /invoice/{{id}}/:send — query params: sendType (EMAIL/EHF)
GET /invoice/paymentType — get payment type IDs
GET/POST /travelExpense — POST body: employee ({{id}}), travelDetails (departureDate, returnDate, destination, purpose, isDayTrip), title
POST /travelExpense/cost — body: travelExpense ({{id}}), vatType ({{id: 0}}), paymentType ({{id}}), amountCurrencyIncVat, date
GET/POST /ledger/voucher — POST body: date, description, postings (array of {{row, account ({{id}}), amountGross, amountGrossCurrency, vatType ({{id}})}})
GET /ledger/account?number=X — REQUIRED to convert account numbers to IDs
POST /supplierInvoice — body: invoiceNumber, invoiceDate, supplier ({{id}}), voucher (with postings)
GET/POST /employee/employment — POST body: employee ({{id}}), startDate. FORBIDDEN: employmentType, userType
POST /salary/transaction — body: year, month, payslips ([{{employee: {{id}}}}])
GET/POST /timesheet/entry — POST body: employee ({{id}}), project ({{id}}), activity ({{id}}), date, hours
PUT /employee/entitlement/:grantEntitlementsByTemplate — params: employeeId, template (ALL_PRIVILEGES, ACCOUNTANT, etc.)

## Project Billing (IMPORTANT for tier 2/3)
- Fixed price project: POST /project with isFixedPrice: true, fixedprice: <amount>
- Project invoice: After creating project, create order with project ref, then /:invoice
- Milestone billing: Invoice a PERCENTAGE of fixed price via order with calculated amount

## Critical Rules
- Sandbox starts EMPTY — create ALL prerequisites (customer, employee, product, etc.)
- Account numbers != IDs — always GET /ledger/account?number=X first
- All PUTs require version field
- Voucher postings: row starts from 1 (NOT 0), include amountGross AND amountGrossCurrency
- Bank account: GET /ledger/account?number=1920, PUT to set bankAccountNumber="12345678903" BEFORE invoicing
- Dates: YYYY-MM-DD. Today = {date.today().isoformat()}
- Use $step_N.id / $step_N.values[0].id for references between steps

## API Schemas (writable fields)
{schemas}

Output ONLY valid JSON:
{{"task_type": "...", "reasoning": "brief explanation", "extracted_values": {{}}, "steps": [{{"method": "POST/GET/PUT/DELETE", "path": "/...", "body": {{}}, "params": {{}}}}]}}"""


# Backward-compatible aliases (agent.py still imports these)
def build_planner_prompt(task_type: str, tier: int = 1) -> str:
    """Deprecated — redirects to build_extraction_prompt. Unknown tasks get step generation."""
    return build_extraction_prompt(task_type, tier)


def build_self_repair_prompt(
    task_type: str,
    original_prompt: str,
    plan: dict,
    results: dict,
    failed: list,
    verification_errors: list[dict] | None = None,
) -> str:
    """Deprecated — redirects to build_repair_extraction_prompt."""
    errors = []
    for idx, res in failed:
        errors.append({"step": idx, "status_code": res["status_code"], "error": res["data"]})
    if verification_errors:
        errors.extend(verification_errors)
    return build_repair_extraction_prompt(task_type, original_prompt, errors)
