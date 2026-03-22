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
- ALL amounts MUST be numbers (e.g. 42100, not "-" or "null"). If you cannot determine the exact amount, use 0.
- For voucher postings: ALWAYS include numeric amountGross on EVERY posting. Never use "-" or empty string.
- Amounts as numbers (1500.00 not "1500.00"). Never calculate VAT yourself.
- "ekskl. mva" -> use amount as priceExcludingVatCurrency. "inkl. mva" -> priceIncludingVatCurrency.
- Booleans as true/false.
- Preserve special chars exactly: Ø, Æ, Å, ñ, ü, etc.
- Phone numbers: preserve as-is from prompt. "telefon"/"tlf"/"mobil" -> phoneNumberMobile for employees, phoneNumber for customers.
- Addresses: extract addressLine1, postalCode, city as separate keys.
- For update tasks: put changed fields in a "fields_to_update" dict.
- For orderLines: array of {{"description": "...", "count": N, "unitPriceExcludingVatCurrency": N}}.
  If a product number is given (e.g. "Opplæring (7579)" or "producto 6042"), include "productNumber" in the orderLine: {{"description": "Opplæring", "productNumber": "7579", "count": 1, "unitPriceExcludingVatCurrency": 2350}}
  If different VAT rates are specified per line (e.g. "25% IVA", "15% IVA alimentos", "0% IVA exento"), include "vatType" in the orderLine: {{"vatType": 25}} or {{"vatType": 15}} or {{"vatType": 0}}
- For per diem / diett / dieta: include in costs array as {{"description": "Diett", "amount": DAILY_RATE * DAYS, "perDiem": true, "dailyRate": DAILY_RATE, "days": DAYS}}
- For voucher/opening balance: "accounts" list of {{"number": "1920", "amount": 100000}} (positive=debit, negative=credit).
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
- employeeFirstName + employeeLastName + employeeEmail: for travel expenses, extract the employee's name and email
- perDiem_dailyRate + perDiem_days: for per diem / diett / dieta, extract daily rate and number of days
- costs: array of cost items for travel expenses, each with description and amount. E.g. [{{"description": "Flybillett", "amount": 3800}}, {{"description": "Taxi", "amount": 200}}]

COMMON EXTRACTION MISTAKES TO AVOID:
- "org.nr 912345678" → organizationNumber: "912345678" (NOT "org.nr 912345678")
- "tlf 55112233" → phoneNumber: "55112233" or phoneNumberMobile: "55112233"
- "Strandgata 12, 6800 Førde" → addressLine1: "Strandgata 12", postalCode: "6800", city: "Førde"
- "avdelingsnummer 200" → departmentNumber: "200" (string, not int)
- "kontoadministrator" → role: "ALL_PRIVILEGES"
- "pris 4999 kr eks mva" → priceExcludingVatCurrency: 4999
- vatTypeId: If VAT rate specified (25%=3, 15% food=33, 12%=31, 0%=5), include the ID. E.g. "15% food and beverage VAT" → vatTypeId: 33
- expense_account_number: For supplier invoices, if no specific expense account is mentioned, default to "6500" (external services). For goods: "4300". For rent: "6300". For office supplies: "6500".
- dueDate: For supplier invoices, if no due date is mentioned, calculate as invoiceDate + 30 days.
- amount: For "TTC" (toutes taxes comprises) or "inkl. mva" or "brutto", this is the GROSS amount including VAT. Extract the exact number."""


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
    """For unknown tasks, allow step generation since we have no template."""
    return f"""You are an expert Tripletex accounting agent. This task could not be classified.

Analyze the prompt and produce a JSON response with:
1. "task_type": your best guess at the task type
2. "extracted_values": all values extracted from the prompt
3. "steps": array of API calls needed

{GLOSSARY}

Key patterns:
- Create entity: POST /entity with required fields
- Update entity: GET first (need ID + version), then PUT
- Delete entity: DELETE /entity/{{id}}
- Action on entity: PUT /entity/{{id}}/:action with query params
- Invoice flow: POST /customer -> POST /order (with orderLines + deliveryDate) -> PUT /order/{{id}}/:invoice
- Payment: GET invoice, GET /invoice/paymentType, PUT /invoice/{{id}}/:payment
- Voucher: GET /ledger/account?number=X per account, POST /ledger/voucher
- Sandbox starts EMPTY — create prerequisites first
- Account numbers (1920, 3000) are NOT IDs — must GET /ledger/account?number=X
- All PUTs require version field from GET response
- Action endpoints (/:payment, /:invoice, /:createCreditNote) use query params, not body
- Employee creation requires userType: "STANDARD" and department
- Orders require both orderDate and deliveryDate
- POST /order body MUST include orderLines array for invoicing to work
- POST /employee/employment ONLY accepts: employee.id, startDate, endDate, division. FORBIDDEN: employmentType, percentageOfFullTimeEquivalent, userType, type, jobTitle
- POST /timesheet/entry accepts: employee, project, activity, date, hours, comment. FORBIDDEN: description, title, name, type
- POST /purchaseOrder ONLY accepts: supplier.id, ourContact.id, deliveryDate. FORBIDDEN: status, currency, transportType, orderLineSorting, orderLines
- PUT /:createReminder uses dispatchType (NOT sendType, NOT sendMethod). Valid: EMAIL, SMS, LETTER

Rules:
- Dates as YYYY-MM-DD. Today = {date.today().isoformat()}.
- Amounts as numbers. Preserve special chars (Ø, Æ, Å).
- Use $step_N.id to reference IDs from previous steps.
- Output ONLY valid JSON.

Output format:
{{"task_type": "...", "reasoning": "brief", "extracted_values": {{}}, "steps": [{{"method": "POST", "path": "/...", "body": {{}}, "params": {{}}}}]}}"""


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
