"""Stage 2: Fill a plan template with values extracted from the prompt.

Only loads relevant entity schemas for the detected task type.
"""
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
        with open(_SCHEMA_PATH) as f:
            _ALL_SCHEMAS = json.load(f)
    return _ALL_SCHEMAS


def _get_relevant_schemas(task_type: str) -> str:
    """Load only the schemas relevant to this task type."""
    all_schemas = _load_schemas()
    template = TEMPLATES.get(task_type, TEMPLATES["unknown"])
    relevant = {
        name: all_schemas[name]
        for name in template["relevant_schemas"]
        if name in all_schemas
    }
    return json.dumps(relevant, indent=2, ensure_ascii=False)


GLOSSARY = """## Norwegian Accounting Glossary
- Faktura = Invoice | Kreditnota = Credit note | Innbetaling/Betaling = Payment
- Kunde = Customer | Leverandor = Supplier | Ansatt = Employee
- Produkt = Product | Prosjekt = Project | Avdeling = Department
- Reiseregning = Travel expense | Ordrelinje = Order line
- Bilag = Voucher | Kontoplan = Chart of accounts | Mva = VAT
- Kontoadministrator = ALL_PRIVILEGES | Regnskapsfor = ACCOUNTANT
- Lonnansvarlig = PERSONELL_MANAGER | Fakturaansvarlig = INVOICING_MANAGER
- Revisor = AUDITOR | Avdelingsleder = DEPARTMENT_LEADER
- Forfallsdato = Due date | Organisasjonsnummer = Org number
"""

ACTION_ENDPOINTS = """## Key Action Endpoints
- PUT /order/{id}/:invoice — invoiceDate (required), sendToCustomer (optional). NOTE: invoiceDueDate is set on the Invoice object AFTER creation, or use invoicesDueIn on the Order.
- PUT /invoice/{id}/:payment — paymentDate, paymentTypeId, paidAmount (all required)
- PUT /invoice/{id}/:createCreditNote — date (required), comment
- PUT /invoice/{id}/:send — sendType (required)
- PUT /employee/entitlement/:grantEntitlementsByTemplate — employeeId, template (both required)
- PUT /travelExpense/:deliver — id | PUT /travelExpense/:approve — id
- PUT /ledger/voucher/{id}/:reverse — date (required)
- DELETE /travelExpense/{id}

## Entitlement Templates
ALL_PRIVILEGES, INVOICING_MANAGER, PERSONELL_MANAGER, ACCOUNTANT, AUDITOR, DEPARTMENT_LEADER

## Employee.userType
STANDARD, EXTENDED, NO_ACCESS
"""


def build_planner_prompt(task_type: str) -> str:
    """Build Stage 2 prompt with task-specific context."""
    template = TEMPLATES.get(task_type, TEMPLATES["unknown"])
    schemas = _get_relevant_schemas(task_type)
    template_json = json.dumps(template["steps"], indent=2, ensure_ascii=False)

    return f"""You are an expert accounting agent for Tripletex. You must complete an accounting task by producing a JSON plan of API calls.

## Your Task Type: {task_type}
{template["description"]}

## Template Steps (adapt these — fill in values from the prompt)
{template_json}

## Fields to Extract from Prompt
{json.dumps(template["extract_fields"])}

## Relevant API Schemas (writable fields)
{schemas}

{GLOSSARY}

{ACTION_ENDPOINTS}

## Rules
1. Output ONLY valid JSON. No markdown, no explanation.
2. Use $step_N.id to reference IDs from previous steps' responses.
3. Extract ALL values from the prompt — never guess or leave placeholders.
4. Dates in YYYY-MM-DD format. Use today's date if not specified in prompt.
5. Remove optional fields that are not mentioned in the prompt.
6. For entity references use {{"id": "$step_N.id"}} or {{"id": <known_id>}}.
7. If files are attached (PDFs, images), extract ALL relevant data from them: names, amounts, dates, line items, account numbers. The file content IS the data source — use it.
8. Account numbers (e.g. 1920, 3000) are NOT account IDs. You must GET /ledger/account?number=X to find the real ID before using it in postings.

## Output Schema
```json
{{
  "task_type": "{task_type}",
  "reasoning": "Brief explanation",
  "steps": [
    {{"method": "POST|GET|PUT|DELETE", "path": "/...", "body": {{}}, "params": {{}}}}
  ]
}}
```
"""


def build_self_repair_prompt(
    task_type: str,
    original_prompt: str,
    plan: dict,
    results: dict,
    failed: list,
) -> str:
    """Build self-repair prompt: feed error back to LLM for correction."""
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

    return f"""An accounting task failed during execution. Analyze the errors and produce a CORRECTED complete plan.

## Original Task
{original_prompt}

## Task Type: {task_type}

## Original Plan
{json.dumps(plan, indent=2, ensure_ascii=False)}

## Execution Results (step index -> response)
{json.dumps(results_summary, indent=2, ensure_ascii=False)}

## Failed Steps
{json.dumps(failed_summary, indent=2, ensure_ascii=False)}

## Relevant API Schemas
{schemas}

{ACTION_ENDPOINTS}

## Instructions
1. Analyze WHY each step failed (read the error messages carefully)
2. Produce a corrected plan that fixes the errors
3. You may reuse IDs from successful steps using $step_N.id (original indices)
4. Output ONLY valid JSON in the same plan format
5. Include ALL steps (both already-succeeded and corrected ones)
"""
