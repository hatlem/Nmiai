"""Stage 2: Build plan with extracted values for execution and verification."""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from templates import TEMPLATES

_SCHEMA_PATH = Path(__file__).parent.parent / "schemas" / "api_reference.json"
_ALL_SCHEMAS: dict | None = None


def _load_schemas() -> dict:
    global _ALL_SCHEMAS
    if _ALL_SCHEMAS is None:
        try:
            with open(_SCHEMA_PATH) as f:
                _ALL_SCHEMAS = json.load(f)
        except FileNotFoundError:
            _ALL_SCHEMAS = {}
    return _ALL_SCHEMAS


def _get_relevant_schemas(task_type: str) -> str:
    all_schemas = _load_schemas()
    if not all_schemas:
        return "(Schema file not found - use your knowledge of the Tripletex API)"
    template = TEMPLATES.get(task_type, TEMPLATES["unknown"])
    relevant = {
        name: all_schemas[name]
        for name in template["relevant_schemas"]
        if name in all_schemas
    }
    return json.dumps(relevant, indent=2, ensure_ascii=False)


GLOSSARY = """## Norwegian Accounting Glossary (Bokmal / Nynorsk / English)
- Faktura = Invoice | Kreditnota = Credit note | Innbetaling/Betaling = Payment
- Kunde = Customer | Leverandor = Supplier | Ansatt/Tilsett = Employee
- Produkt = Product | Prosjekt = Project | Avdeling = Department
- Reiseregning/Reiserekning = Travel expense | Ordrelinje = Order line
- Bilag = Voucher | Kontoplan = Chart of accounts | Mva = VAT
- Kontoadministrator = ALL_PRIVILEGES | Regnskapsfor/Rekneskapsforar = ACCOUNTANT
- Lonnansvarlig = PERSONELL_MANAGER | Fakturaansvarlig = INVOICING_MANAGER
- Revisor = AUDITOR | Avdelingsleder = DEPARTMENT_LEADER
- Forfallsdato = Due date | Organisasjonsnummer = Org number
- Postering = Posting | Debet = Debit | Kredit = Credit
- Nynorsk: "tilsett/tilsatt" = ansatt, "verksemd" = virksomhet, "reknskap" = regnskap
"""

ACTION_ENDPOINTS = """## Key Action Endpoints (use query params, NOT body)
- PUT /order/{id}/:invoice — invoiceDate (REQUIRED), sendToCustomer (optional)
- PUT /invoice/{id}/:payment — paymentDate, paymentTypeId, paidAmount (ALL REQUIRED)
- PUT /invoice/{id}/:createCreditNote — date (REQUIRED), comment (optional)
- PUT /invoice/{id}/:send — sendType (REQUIRED: EMAIL, EHF, EFAKTURA, LETTER, MANUAL)
- PUT /invoice/{id}/:createReminder — type (REQUIRED), date (REQUIRED), comment (optional)
- PUT /employee/entitlement/:grantEntitlementsByTemplate — employeeId, template (REQUIRED)
- PUT /travelExpense/:deliver — id (REQUIRED)
- PUT /travelExpense/:approve — id (REQUIRED)
- PUT /ledger/voucher/{id}/:reverse — date (REQUIRED)
- DELETE /travelExpense/{id}

## Entitlement Templates
ALL_PRIVILEGES, INVOICING_MANAGER, PERSONELL_MANAGER, ACCOUNTANT, AUDITOR, DEPARTMENT_LEADER

## Employee.userType
STANDARD, EXTENDED, NO_ACCESS
"""

EFFICIENCY_RULES = """## Efficiency Rules (CRITICAL — affects your score!)
Every unnecessary API call REDUCES your score. The scoring system tracks:
1. Total number of API calls vs. the optimal solution
2. Number of 4xx error responses (EACH one hurts your score)

DO:
- POST directly for new entities with ALL fields in one call
- Combine operations where possible (order + orderLines in one call)
- Use the template steps as your guide — they represent near-optimal call sequences
- Include ALL required fields in the FIRST attempt to avoid 422 errors

DO NOT:
- Make GET calls to "check if something exists" before creating it (UNLESS task says "existing")
- Make exploratory GET calls to discover what's available
- Retry failed calls blindly — understand the error first
- Make unnecessary GET calls to verify your work — the system does this

EXCEPTIONS (GET-before-create IS correct):
- "existing customer/employee" -> GET first to find their ID
- Account numbers -> GET /ledger/account?number=X (required — account numbers are NOT IDs!)
- Payment types -> GET /invoice/paymentType (required for register_payment)
- Employee ID -> GET /employee when needed for travel expenses, timesheets, salary
- Department ID -> GET /department when creating employees (include if results exist)
"""

VERIFICATION_AWARENESS = """## Verification Awareness
After execution, the system verifies EVERY field against expected values.
- Include ALL fields from the prompt in your API calls — every missing field loses points
- Dates in YYYY-MM-DD format
- Amounts as numbers (not strings): 1500.00 not "1500.00"
- Preserve special characters in names exactly (Ø, Æ, Å, ñ, ü, etc.)
- Include organizationNumber if mentioned (format: 9 digits in Norway)
- Include phoneNumber/phoneNumberMobile if mentioned
- Include email if mentioned
- Include dateOfBirth if mentioned (for employees)
- Include description if mentioned (for products, projects)
- For updates: ALWAYS include the version field from the GET response in the PUT body
- Amounts with MVA/VAT: Extract the gross amount and let the API handle VAT calculation
"""

KNOWN_PITFALLS = """## CRITICAL PITFALLS
1. PUT with version: All PUT updates REQUIRE the version field from GET. Always GET first for updates.
2. isCustomer: true: ALWAYS set when creating a customer.
3. invoiceDueDate: MANDATORY when creating invoices. Default: 14 days after invoiceDate.
4. Action endpoints use query params: /:payment, /:createCreditNote, /:invoice take QUERY PARAMS, not body!
5. Account numbers != IDs: Must GET /ledger/account?number=X to find real ID.
6. VAT types vary: Never hardcode vatType IDs. Query GET /ledger/vatType.
7. Order before Invoice: Create Order with orderLines, then PUT /order/{id}/:invoice.
7b. Order REQUIRES deliveryDate AND orderDate: ALWAYS include both in POST /order body. Use invoiceDate or today's date if not explicitly specified in the prompt.
8. Empty sandbox: Each submission starts fresh — no pre-existing entities.
9. Supplier creation: Set name (required). Do NOT set isSupplier on /supplier endpoint.
10. Travel expense employee: Always GET /employee first for the employee ID.
11. Department on employee: If GET /department returns results, include "department": {"id": <first_dept_id>} in POST /employee body.
11b. userType on employee: ALWAYS include "userType": "STANDARD" when creating employees. Without it you get 422.
12. Version field for PUTs: ALL PUT requests require the 'version' field from the GET response. Include it in the body. Missing version causes 409 Conflict.

## TIER 3 PITFALLS (complex tasks)
13. Opening balance — postings MUST sum to zero: Total debit must equal total credit. If you only have
    asset accounts, add a balancing equity posting (e.g. account 2050). Format each posting as:
    {"account": {"id": <id>}, "amountGross": <amount>} where positive = debit, negative = credit.
    Do NOT fetch all accounts (count=1000) — only GET the specific account numbers mentioned in the task.
14. Bank reconciliation — accounting period must be open: The reconciliation date range must fall within
    an open accounting period. If you get a 422 error about closed period, the dates are wrong.
    After creating the reconciliation, you may need to POST individual payment/match entries.
15. Invoice with payment — the /:invoice action returns the created invoice: When you PUT
    /order/{id}/:invoice, the response contains the invoice ID. Use $step_N.id from that response
    for the subsequent /:payment call. Do NOT try to search for the invoice separately.
16. Bank account for invoicing: If invoice creation fails with "bankkontonummer" error, the company needs a bank account. This is usually pre-configured in competition sandboxes but may need: PUT /company with bankAccountNumber field.
17. deliveryDate on orders: REQUIRED field. If not specified in the prompt, use the same date as orderDate.
"""


def build_planner_prompt(task_type: str, tier: int = 1) -> str:
    """Build Stage 2 prompt with task-specific context."""
    template = TEMPLATES.get(task_type, TEMPLATES["unknown"])
    schemas = _get_relevant_schemas(task_type)
    template_json = json.dumps(template["steps"], indent=2, ensure_ascii=False)

    conditional_section = ""
    if template.get("conditional_steps"):
        conditional_json = json.dumps(template["conditional_steps"], indent=2, ensure_ascii=False)
        conditional_section = f"""
## Conditional Steps (include ONLY if the prompt mentions the relevant field)
{conditional_json}
If the prompt mentions a role, entitlement, or permission — append the matching conditional step.
If not mentioned, omit entirely.
"""

    return f"""You are an expert accounting agent for Tripletex. Produce a JSON plan of API calls AND extract all values from the prompt.

## Task Type: {task_type}
{template["description"]}

## Task Tier: {tier} ({"simple" if tier == 1 else "medium" if tier == 2 else "complex"})

## Template Steps (adapt — fill values from prompt)
{template_json}
{conditional_section}
## Fields to Extract
{json.dumps(template["extract_fields"])}

## Relevant API Schemas
{schemas}

{GLOSSARY}
{ACTION_ENDPOINTS}
{EFFICIENCY_RULES}
{VERIFICATION_AWARENESS}
{KNOWN_PITFALLS}

## Rules
1. Output ONLY valid JSON. No markdown, no explanation.
2. Use $step_N.id to reference IDs from previous steps.
3. Extract ALL values from the prompt — never guess or leave placeholders.
4. Dates in YYYY-MM-DD format. Use today's date if not specified.
5. Remove optional fields NOT mentioned in the prompt.
6. For entity references use {{"id": "$step_N.id"}} or {{"id": <known_id>}}.
7. If files are attached, extract ALL data from them.
8. Account numbers (1920, 3000) are NOT IDs. Must GET /ledger/account?number=X first.
9. Include ALL mentioned fields — missing fields are scored as errors.
10. For action endpoints (PUT with :), put parameters in "params" not "body".
11. Numbers should be numbers (not strings).

## Output Schema
```json
{{
  "task_type": "{task_type}",
  "reasoning": "Brief explanation",
  "steps": [
    {{"method": "POST|GET|PUT|DELETE", "path": "/...", "body": {{}}, "params": {{}}}}
  ],
  "extracted_values": {{
    "fieldName": "value from prompt"
  }}
}}
```

## CRITICAL: extracted_values
The extracted_values dict MUST contain EVERY piece of data you extracted from the prompt.
Keys MUST match Tripletex API field names exactly. For example:
- Employee: firstName, lastName, email, phoneNumberMobile, dateOfBirth
- Customer: name, email, organizationNumber, phoneNumber
- Product: name, priceExcludingVatCurrency, number, description
- Project: project_name, description, startDate, endDate
- Department: name, departmentNumber
- Invoice: customer_name, customer_email, invoiceDate, invoiceDueDate, orderLines
- Travel: departureDate, returnDate, departureFrom, destination, purpose
- Supplier: name, email, organizationNumber, phoneNumber
- Contact: firstName, lastName, email, phoneNumber
- Voucher: date, description, account numbers, amounts

If the prompt says "Kari Nordmann, kari@test.no, tlf 99887766" your extracted_values MUST include:
{{"firstName": "Kari", "lastName": "Nordmann", "email": "kari@test.no", "phoneNumberMobile": "99887766"}}
NEVER return an empty extracted_values if the prompt contains any data.
For update tasks, put all fields to change in a "fields_to_update" dict within extracted_values.
"""


def build_self_repair_prompt(
    task_type: str,
    original_prompt: str,
    plan: dict,
    results: dict,
    failed: list,
    verification_errors: list[dict] | None = None,
) -> str:
    """Build self-repair prompt with error analysis and optional verification mismatches."""
    schemas = _get_relevant_schemas(task_type)

    results_summary = {}
    for idx, res in results.items():
        results_summary[str(idx)] = {
            "ok": res["ok"],
            "status_code": res["status_code"],
            "data": res["data"],
        }

    failed_summary = [
        {"step": idx, "status_code": res["status_code"], "error": res["data"]}
        for idx, res in failed
    ]

    verification_section = ""
    if verification_errors:
        verification_section = f"""
## Verification Errors (field mismatches found after execution)
{json.dumps(verification_errors, indent=2, ensure_ascii=False)}

To fix field mismatches:
1. GET the entity first to obtain its current version number
2. PUT to update the entity with the correct field values + version
"""

    return f"""An accounting task failed during execution or verification. Produce a CORRECTED plan.

## Original Task
{original_prompt}

## Task Type: {task_type}

## Original Plan
{json.dumps(plan, indent=2, ensure_ascii=False)}

## Execution Results
{json.dumps(results_summary, indent=2, ensure_ascii=False)}

## Failed Steps
{json.dumps(failed_summary, indent=2, ensure_ascii=False)}
{verification_section}
## API Schemas
{schemas}

{ACTION_ENDPOINTS}

## Common Error Fixes
1. 422 "field is required" -> Add the missing field
2. 404 on $step_N.id -> Previous step failed, fix it
3. 400 "already exists" -> GET to find existing entity
4. 409 "version conflict" -> GET first, use version in PUT
5. 422 "invalid value" -> Check field types
6. Field mismatch -> GET entity, PUT with correct value + version
7. Account number as ID -> Must GET /ledger/account?number=X first
8. 422 "bankkontonummer" -> Company needs bank account. Fix: GET /company/1 to get version, then PUT /company/1 with bankAccountNumber (e.g. "15031750204")
9. 422 "Brukertype" on employee -> Add "userType": "STANDARD" to body
10. 422 "department" on employee -> GET /department first, include "department": {"id": <id>} in body
11. 422 "deliveryDate" or "orderDate" null on order -> Add deliveryDate and orderDate (use invoiceDate or today)

## Instructions
1. Analyze WHY each step failed
2. Produce corrected plan fixing ALL errors
3. Reuse IDs from successful steps with $step_N.id
4. Only include steps that need to be (re-)executed
5. Output ONLY valid JSON

## Output Schema
```json
{{
  "task_type": "{task_type}",
  "reasoning": "What went wrong and fix",
  "steps": [...],
  "extracted_values": {{}}
}}
```
"""
