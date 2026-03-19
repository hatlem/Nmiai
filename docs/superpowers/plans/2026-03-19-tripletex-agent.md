# Tripletex AI Agent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an AI agent with `/solve` endpoint that receives accounting tasks in natural language, plans Tripletex API calls via Gemini 3.1 Pro, executes them, and scores maximum points.

**Architecture:** Plan-then-Execute. Single LLM call parses prompt into structured JSON plan of API calls. Deterministic executor runs the plan, resolving inter-step dependencies. Optional LLM recovery on failure.

**Tech Stack:** Python 3.11, FastAPI, httpx, Vertex AI (Gemini 3.1 Pro), Docker, Cloud Run

**Spec:** `docs/superpowers/specs/2026-03-19-tripletex-agent-design.md`

---

## File Structure

```
tripletex/
├── Dockerfile                      # Python 3.11-slim, uvicorn
├── requirements.txt                # fastapi, uvicorn, httpx, google-cloud-aiplatform
├── main.py                         # FastAPI app: POST /solve, GET /health
├── agent.py                        # LLM planning: prompt → structured plan, recovery
├── executor.py                     # Execute plan steps, resolve deps, handle errors
├── tripletex_client.py             # Async HTTP client: auth, GET/POST/PUT/DELETE
├── prompts/
│   ├── __init__.py
│   └── system.py                   # System prompt builder: role + API ref + glossary + few-shots
├── schemas/
│   └── api_reference.json          # Compact Tripletex API reference (14 KB)
└── tests/
    ├── __init__.py
    ├── test_executor.py            # Unit tests for dependency resolution + error handling
    ├── test_agent.py               # Unit tests for plan parsing
    └── test_main.py                # Integration test for /solve endpoint
```

---

### Task 1: Tripletex HTTP Client

**Files:**
- Create: `tripletex/tripletex_client.py`
- Create: `tripletex/tests/test_executor.py` (dependency resolution tests used later)

The async HTTP wrapper for all Tripletex API calls. Handles auth, JSON parsing, error extraction.

- [ ] **Step 1: Create tripletex_client.py**

```python
# tripletex/tripletex_client.py
import httpx
import logging

logger = logging.getLogger(__name__)

class TripletexClient:
    """Async HTTP client for Tripletex API v2."""

    def __init__(self, base_url: str, session_token: str):
        self.base_url = base_url.rstrip("/")
        self.auth = ("0", session_token)
        self._client = httpx.AsyncClient(
            auth=self.auth,
            timeout=30.0,
            headers={"Content-Type": "application/json"},
        )

    async def request(self, method: str, path: str, body: dict | None = None, params: dict | None = None) -> dict:
        """Make an API request. Returns {"status_code": int, "ok": bool, "data": dict}."""
        url = f"{self.base_url}{path}"
        logger.info(f"{method} {path}")

        response = await self._client.request(
            method=method,
            url=url,
            json=body,
            params=params,
        )

        result = {
            "status_code": response.status_code,
            "ok": response.is_success,
        }

        try:
            result["data"] = response.json()
        except Exception:
            result["data"] = {"raw": response.text[:500]}

        if not response.is_success:
            logger.warning(f"{method} {path} → {response.status_code}: {result['data']}")

        return result

    async def get(self, path: str, params: dict | None = None) -> dict:
        # Default to fields=* so we see all available fields
        if params is None:
            params = {}
        if "fields" not in params:
            params["fields"] = "*"
        return await self.request("GET", path, params=params)

    async def post(self, path: str, body: dict | None = None, params: dict | None = None) -> dict:
        return await self.request("POST", path, body=body, params=params)

    async def put(self, path: str, body: dict | None = None, params: dict | None = None) -> dict:
        return await self.request("PUT", path, body=body, params=params)

    async def delete(self, path: str, params: dict | None = None) -> dict:
        return await self.request("DELETE", path, params=params)

    async def close(self):
        await self._client.aclose()
```

- [ ] **Step 2: Verify file created**

Run: `cat tripletex/tripletex_client.py | head -5`
Expected: Shows the import lines.

- [ ] **Step 3: Commit**

```bash
git add tripletex/tripletex_client.py
git commit -m "feat(tripletex): add async HTTP client wrapper"
```

---

### Task 2: Execution Engine

**Files:**
- Create: `tripletex/executor.py`
- Create: `tripletex/tests/__init__.py`
- Create: `tripletex/tests/test_executor.py`

The deterministic executor that runs a plan's API call steps, resolves `$step_N.field` dependencies, and handles errors.

- [ ] **Step 1: Write failing tests for dependency resolution**

```python
# tripletex/tests/test_executor.py
import pytest
from executor import resolve_ref, resolve_refs

def test_resolve_ref_simple_id():
    results = {0: {"data": {"value": {"id": 42}}}}
    assert resolve_ref("$step_0.id", results) == 42

def test_resolve_ref_nested_field():
    results = {0: {"data": {"value": {"id": 42, "name": "Test"}}}}
    assert resolve_ref("$step_0.name", results) == "Test"

def test_resolve_ref_no_match():
    assert resolve_ref("plain_string", {}) == "plain_string"

def test_resolve_refs_in_dict():
    results = {0: {"data": {"value": {"id": 10}}}}
    body = {"customer": {"id": "$step_0.id"}, "name": "Test"}
    resolved = resolve_refs(body, results)
    assert resolved == {"customer": {"id": 10}, "name": "Test"}

def test_resolve_refs_in_path():
    results = {1: {"data": {"value": {"id": 99}}}}
    path = "/order/$step_1.id/:invoice"
    resolved = resolve_ref(path, results)
    assert resolved == "/order/99/:invoice"

def test_resolve_ref_array_indexing():
    """Test $step_N.values[0].id pattern for list responses."""
    results = {0: {"data": {"values": [{"id": 7, "description": "Cash"}, {"id": 8, "description": "Bank"}]}}}
    assert resolve_ref("$step_0.values[0].id", results) == 7

def test_resolve_ref_deep_nested():
    """Test deeply nested path resolution."""
    results = {0: {"data": {"value": {"customer": {"id": 5}}}}}
    assert resolve_ref("$step_0.customer.id", results) == 5
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd tripletex && /opt/homebrew/bin/python3 -m pytest tests/test_executor.py -v`
Expected: FAIL — `executor` module not found.

- [ ] **Step 3: Implement executor.py**

```python
# tripletex/executor.py
import re
import logging
from tripletex_client import TripletexClient

logger = logging.getLogger(__name__)

def _deep_get(obj, path_parts: list[str]):
    """Navigate nested dicts/lists by dot-separated path parts.
    Supports array indexing like 'values[0]'."""
    current = obj
    for part in path_parts:
        if current is None:
            return None
        # Handle array indexing: values[0]
        array_match = re.match(r'(\w+)\[(\d+)\]', part)
        if array_match:
            key, idx = array_match.group(1), int(array_match.group(2))
            if isinstance(current, dict) and key in current:
                current = current[key]
                if isinstance(current, list) and idx < len(current):
                    current = current[idx]
                else:
                    return None
            else:
                return None
        elif isinstance(current, dict):
            current = current.get(part)
        else:
            return None
    return current


def resolve_ref(value: str, results: dict):
    """Resolve $step_N.path.to.field references in a string value.
    Supports nested paths like $step_0.values[0].id and $step_1.value.id"""
    if not isinstance(value, str):
        return value

    pattern = r'\$step_(\d+)\.([\w\[\]\.]+)'

    def _resolve_single(step_idx: int, field_path: str):
        step_data = results.get(step_idx, {}).get("data", {})
        parts = field_path.split(".")

        # Try value.path first (POST/PUT wrap in {"value": {...}})
        val = step_data.get("value", {})
        if isinstance(val, dict):
            resolved = _deep_get(val, parts)
            if resolved is not None:
                return resolved
            # Try without first part if it's "value" (user wrote $step_0.value.id)
            if parts[0] == "value" and len(parts) > 1:
                resolved = _deep_get(val, parts[1:])
                if resolved is not None:
                    return resolved

        # Try direct path on step_data
        resolved = _deep_get(step_data, parts)
        if resolved is not None:
            return resolved

        return None

    def replacer(match):
        step_idx = int(match.group(1))
        field_path = match.group(2)
        resolved = _resolve_single(step_idx, field_path)
        if resolved is not None:
            return str(resolved)
        logger.warning(f"Could not resolve $step_{step_idx}.{field_path}")
        return match.group(0)

    # If the entire string is a single reference, return typed value
    single_match = re.fullmatch(pattern, value)
    if single_match:
        step_idx = int(single_match.group(1))
        field_path = single_match.group(2)
        resolved = _resolve_single(step_idx, field_path)
        if resolved is not None:
            return resolved

    # Otherwise do string substitution
    resolved = re.sub(pattern, replacer, value)
    return resolved


def resolve_refs(obj, results: dict):
    """Recursively resolve all $step_N.field references in a dict/list/string."""
    if isinstance(obj, str):
        return resolve_ref(obj, results)
    if isinstance(obj, dict):
        return {k: resolve_refs(v, results) for k, v in obj.items()}
    if isinstance(obj, list):
        return [resolve_refs(item, results) for item in obj]
    return obj


async def execute_plan(plan: dict, client: TripletexClient) -> dict:
    """Execute a structured plan of API calls.

    Returns {"success": bool, "results": {step_idx: response}, "failed": [(idx, error)]}
    """
    steps = plan.get("steps", [])
    results = {}
    failed = []

    for i, step in enumerate(steps):
        method = step["method"].upper()
        path = resolve_ref(step["path"], results)
        body = resolve_refs(step.get("body"), results) if step.get("body") else None
        params = resolve_refs(step.get("params"), results) if step.get("params") else None

        logger.info(f"Step {i}: {method} {path}")
        response = await client.request(method, path, body=body, params=params)
        results[i] = response

        if not response["ok"]:
            # Try one programmatic retry
            fixed_body, fixed_params = try_fix_error(response, step, body, params)
            if fixed_body is not None or fixed_params is not None:
                logger.info(f"Step {i}: Retrying with fix")
                response = await client.request(
                    method, path,
                    body=fixed_body if fixed_body is not None else body,
                    params=fixed_params if fixed_params is not None else params,
                )
                results[i] = response

            if not response["ok"]:
                failed.append((i, response))
                logger.error(f"Step {i} failed: {response['status_code']}")

    return {
        "success": len(failed) == 0,
        "results": results,
        "failed": failed,
    }


def try_fix_error(response: dict, step: dict, body: dict | None, params: dict | None) -> tuple:
    """Attempt programmatic fix based on error response.
    Returns (fixed_body, fixed_params) or (None, None)."""
    status = response.get("status_code", 0)
    data = response.get("data", {})
    msg = str(data).lower()

    # 422 validation errors — parse message for missing/invalid fields
    if status == 422:
        if "iscustomer" in msg and body:
            return {**body, "isCustomer": True}, None
        if "isinternal" in msg and body:
            return {**body, "isInternal": False}, None
        if "orderdate" in msg and body and "orderDate" not in body:
            from datetime import date
            return {**body, "orderDate": date.today().isoformat()}, None
        if "deliverydate" in msg and body and "deliveryDate" not in body:
            from datetime import date
            return {**body, "deliveryDate": date.today().isoformat()}, None

    # 404 not found — likely bad ID in path, cannot fix programmatically
    # (will be handled by LLM recovery)

    # 409 duplicate — entity already exists
    if status == 409:
        # Can't fix without a GET call — flag for LLM recovery
        pass

    # 401 auth — re-check auth header (unlikely to be fixable)
    if status == 401:
        logger.error("Authentication failed — check session token")

    return None, None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd tripletex && /opt/homebrew/bin/python3 -m pytest tests/test_executor.py -v`
Expected: All 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add tripletex/executor.py tripletex/tests/
git commit -m "feat(tripletex): add execution engine with dependency resolution"
```

---

### Task 3: System Prompt Builder

**Files:**
- Create: `tripletex/prompts/__init__.py`
- Create: `tripletex/prompts/system.py`
- Verify: `tripletex/schemas/api_reference.json` (already created)

The system prompt with API reference, accounting glossary, few-shot examples, and output schema.

- [ ] **Step 1: Create prompts/__init__.py**

Empty file.

- [ ] **Step 2: Create prompts/system.py**

This is the core system prompt. It must contain:
1. Role + rules
2. Compact API reference (loaded from api_reference.json)
3. Norwegian accounting glossary
4. Few-shot examples for each task category
5. Output JSON schema

The prompt should be in English (strongest LLM reasoning). Include Norwegian glossary so the model can map Norwegian terms to API fields.

```python
# tripletex/prompts/system.py
import json
from pathlib import Path

_SCHEMA_PATH = Path(__file__).parent.parent / "schemas" / "api_reference.json"

def _load_api_reference() -> str:
    with open(_SCHEMA_PATH) as f:
        return f.read()

def build_system_prompt() -> str:
    api_ref = _load_api_reference()

    return f"""You are an expert accounting agent. You receive accounting tasks in natural language (Norwegian, English, Spanish, Portuguese, Nynorsk, German, or French) and must produce a structured JSON plan of Tripletex API calls to complete the task.

## Important
Prompts may arrive in any of these 7 languages: Norwegian Bokmal (nb), Norwegian Nynorsk (nn), English (en), Spanish (es), Portuguese (pt), German (de), French (fr). Extract all field values regardless of input language. Always output your JSON plan in the same format.

## Rules
1. Output ONLY valid JSON matching the schema below. No markdown, no explanation.
2. Use POST response IDs via $step_N.id references for subsequent steps.
3. Never guess field values — extract everything from the prompt.
4. Minimize API calls — plan the optimal sequence upfront.
5. Do NOT make GET calls to verify what you just created.
6. For references to other entities, use {{"id": "$step_N.id"}} or {{"id": <known_id>}}.
7. Dates should be in YYYY-MM-DD format. Use today's date if not specified.
8. For orders/invoices: always create customer first, then order with orderLines, then invoice via action endpoint.

## Output Schema
```json
{{
  "task_type": "string — e.g. create_employee, create_invoice, register_payment",
  "reasoning": "Brief explanation of what the task requires",
  "steps": [
    {{
      "method": "POST|GET|PUT|DELETE",
      "path": "/endpoint/path — use $step_N.id for dynamic IDs",
      "body": {{}},
      "params": {{}}
    }}
  ]
}}
```

## Tripletex API Reference (writable fields per entity)

{api_ref}

## Key Action Endpoints

- PUT /order/$order_id/:invoice — Convert order to invoice. Params: invoiceDate (required), sendToCustomer (optional, default false)
- PUT /invoice/$id/:payment — Register payment. Params: paymentDate, paymentTypeId, paidAmount (all required)
- PUT /invoice/$id/:createCreditNote — Credit note. Params: date (required), comment, creditNoteEmail
- PUT /invoice/$id/:send — Send invoice. Params: sendType (required), overrideEmailAddress
- PUT /employee/entitlement/:grantEntitlementsByTemplate — Set role. Params: employeeId (required), template (required)
- PUT /travelExpense/:deliver — Deliver travel expense. Params: id
- PUT /travelExpense/:approve — Approve travel expense. Params: id
- PUT /ledger/voucher/$id/:reverse — Reverse voucher. Params: date (required)
- DELETE /travelExpense/$id — Delete travel expense

## Entitlement Templates (Employee Roles)
ALL_PRIVILEGES, INVOICING_MANAGER, PERSONELL_MANAGER, ACCOUNTANT, AUDITOR, DEPARTMENT_LEADER, NONE_PRIVILEGES

## Employee.userType
STANDARD (normal user), EXTENDED (admin), NO_ACCESS

## Norwegian Accounting Glossary
- Faktura = Invoice (POST /order then PUT /order/:invoice)
- Kreditnota = Credit note (PUT /invoice/:createCreditNote)
- Innbetaling/Betaling = Payment (PUT /invoice/:payment)
- Kunde = Customer (POST /customer)
- Leverandor = Supplier (POST /supplier)
- Ansatt = Employee (POST /employee)
- Produkt = Product (POST /product)
- Prosjekt = Project (POST /project)
- Avdeling = Department (POST /department)
- Reiseregning = Travel expense (POST /travelExpense)
- Ordrelinje = Order line (part of order body)
- Bilag = Voucher (POST /ledger/voucher)
- Kontoplan = Chart of accounts (/ledger/account)
- Mva = VAT (/ledger/vatType)
- Kontoadministrator = Account administrator (ALL_PRIVILEGES template)
- Regnskapsfor = Accountant (ACCOUNTANT template)
- Lonnansvarlig = Payroll manager (PERSONELL_MANAGER template)
- Fakturaansvarlig = Invoice manager (INVOICING_MANAGER template)
- Revisor = Auditor (AUDITOR template)
- Avdelingsleder = Department leader (DEPARTMENT_LEADER template)
- Forfallsdato = Due date (invoiceDueDate)
- Organisasjonsnummer = Organization number
- Salgsinntekt = Sales revenue (account 3000)
- Bankinnskudd = Bank deposits (account 1920)

## Few-Shot Examples

### Example 1: Create employee with admin role
Prompt: "Opprett en ansatt med navn Ola Nordmann, e-post ola@example.org. Han skal vaere kontoadministrator."
```json
{{
  "task_type": "create_employee",
  "reasoning": "Create employee Ola Nordmann with email, then grant ALL_PRIVILEGES (kontoadministrator)",
  "steps": [
    {{
      "method": "POST",
      "path": "/employee",
      "body": {{"firstName": "Ola", "lastName": "Nordmann", "email": "ola@example.org"}}
    }},
    {{
      "method": "PUT",
      "path": "/employee/entitlement/:grantEntitlementsByTemplate",
      "params": {{"employeeId": "$step_0.id", "template": "ALL_PRIVILEGES"}}
    }}
  ]
}}
```

### Example 2: Create invoice
Prompt: "Opprett en faktura til kunde Acme AS for 10 timer konsulentarbeid a 1200 kr. Forfallsdato 2026-04-01."
```json
{{
  "task_type": "create_invoice",
  "reasoning": "Create customer Acme AS, then order with one order line (10 x 1200), then invoice the order",
  "steps": [
    {{
      "method": "POST",
      "path": "/customer",
      "body": {{"name": "Acme AS", "isCustomer": true}}
    }},
    {{
      "method": "POST",
      "path": "/order",
      "body": {{
        "customer": {{"id": "$step_0.id"}},
        "orderDate": "2026-03-19",
        "deliveryDate": "2026-03-19",
        "orderLines": [
          {{"description": "Konsulentarbeid", "count": 10, "unitPriceExcludingVatCurrency": 1200}}
        ]
      }}
    }},
    {{
      "method": "PUT",
      "path": "/order/$step_1.id/:invoice",
      "params": {{"invoiceDate": "2026-03-19", "sendToCustomer": false}}
    }}
  ]
}}
```

### Example 3: Register payment on invoice
Prompt: "Registrer en innbetaling pa faktura 1 pa 15000 kr, betalt i dag."
```json
{{
  "task_type": "register_payment",
  "reasoning": "Register payment of 15000 on invoice 1. Need to find paymentTypeId first.",
  "steps": [
    {{
      "method": "GET",
      "path": "/invoice/paymentType",
      "params": {{"fields": "id,description"}}
    }},
    {{
      "method": "PUT",
      "path": "/invoice/1/:payment",
      "params": {{"paymentDate": "2026-03-19", "paymentTypeId": "$step_0.values[0].id", "paidAmount": 15000}}
    }}
  ]
}}
```

### Example 4: Create travel expense
Prompt: "Registrer en reiseregning for reise fra Oslo til Bergen 15. mars 2026."
```json
{{
  "task_type": "create_travel_expense",
  "reasoning": "Create travel expense with travel details for Oslo-Bergen trip",
  "steps": [
    {{
      "method": "POST",
      "path": "/travelExpense",
      "body": {{
        "employee": {{"id": 1}},
        "travelDetails": {{
          "departureDate": "2026-03-15",
          "returnDate": "2026-03-15",
          "departureFrom": "Oslo",
          "destination": "Bergen",
          "isDayTrip": true,
          "isForeignTravel": false,
          "purpose": "Forretningsreise"
        }},
        "title": "Oslo - Bergen 15.03.2026"
      }}
    }}
  ]
}}
```

### Example 5: Create department
Prompt: "Opprett avdeling Salg med avdelingsnummer 200."
```json
{{
  "task_type": "create_department",
  "reasoning": "Create department named Salg with number 200",
  "steps": [
    {{
      "method": "POST",
      "path": "/department",
      "body": {{"name": "Salg", "departmentNumber": "200"}}
    }}
  ]
}}
```

### Example 6: Delete travel expense
Prompt: "Slett reiseregning med ID 5."
```json
{{
  "task_type": "delete_travel_expense",
  "reasoning": "Delete travel expense with ID 5",
  "steps": [
    {{
      "method": "DELETE",
      "path": "/travelExpense/5"
    }}
  ]
}}
```

### Example 7: Create project for customer
Prompt: "Opprett et prosjekt kalt 'Nettsideredesign' for kunde Bedrift AS. Prosjektet starter 1. april og slutter 30. juni 2026."
```json
{{
  "task_type": "create_project",
  "reasoning": "Create customer first, then project linked to that customer",
  "steps": [
    {{
      "method": "POST",
      "path": "/customer",
      "body": {{"name": "Bedrift AS", "isCustomer": true}}
    }},
    {{
      "method": "POST",
      "path": "/project",
      "body": {{
        "name": "Nettsideredesign",
        "customer": {{"id": "$step_0.id"}},
        "startDate": "2026-04-01",
        "endDate": "2026-06-30",
        "isInternal": false
      }}
    }}
  ]
}}
```
"""


def build_recovery_prompt(original_prompt: str, plan: dict, results: dict, failed: list) -> str:
    """Build a recovery prompt when execution fails."""
    return f"""The following accounting task failed during execution. Analyze the errors and provide a corrected plan for the remaining steps.

## Original Task
{original_prompt}

## Original Plan
{json.dumps(plan, indent=2, ensure_ascii=False)}

## Execution Results
{json.dumps({str(k): {{"ok": v["ok"], "status_code": v["status_code"], "data": v["data"]}} for k, v in results.items()}, indent=2, ensure_ascii=False)}

## Failed Steps
{json.dumps([(i, {{"status_code": r["status_code"], "data": r["data"]}}) for i, r in failed], indent=2, ensure_ascii=False)}

Provide a corrected plan in the same JSON format. Only include the steps that still need to be executed. You may reference results from already-completed steps using $step_N.field syntax (using the original step indices).
"""
```

- [ ] **Step 3: Commit**

```bash
git add tripletex/prompts/ tripletex/schemas/api_reference.json
git commit -m "feat(tripletex): add system prompt with API ref, glossary, few-shots"
```

---

### Task 4: LLM Agent (Planning + Recovery)

**Files:**
- Create: `tripletex/agent.py`

The LLM integration that sends prompts to Gemini 3.1 Pro and parses structured JSON responses.

- [ ] **Step 1: Create agent.py**

```python
# tripletex/agent.py
import json
import base64
import logging
from datetime import date

import vertexai
from vertexai.generative_models import GenerativeModel, Part

from prompts.system import build_system_prompt, build_recovery_prompt

logger = logging.getLogger(__name__)

# Initialize Vertex AI
vertexai.init(project="ainm26osl-710", location="europe-north1")

MODEL_ID = "gemini-3.1-pro"

def _get_model() -> GenerativeModel:
    return GenerativeModel(
        MODEL_ID,
        system_instruction=build_system_prompt(),
    )


def _parse_plan(text: str) -> dict:
    """Parse LLM response text into a plan dict."""
    # Strip markdown code fences if present
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:])  # remove first ```json line
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

    return json.loads(text)


async def create_plan(prompt: str, files: list[dict] | None = None) -> dict:
    """Send prompt to Gemini and get a structured execution plan."""
    model = _get_model()

    # Build content parts
    parts = []

    # Add file contents if present (multimodal)
    if files:
        for f in files:
            file_data = base64.b64decode(f["content_base64"])
            parts.append(Part.from_data(data=file_data, mime_type=f["mime_type"]))
            parts.append(Part.from_text(f"[Attached file: {f['filename']}]"))

    # Add the task prompt with today's date context
    today = date.today().isoformat()
    parts.append(Part.from_text(
        f"Today's date is {today}. Complete this accounting task:\n\n{prompt}"
    ))

    logger.info(f"Sending to {MODEL_ID}: {prompt[:100]}...")

    response = model.generate_content(
        parts,
        generation_config={
            "temperature": 0.0,
            "max_output_tokens": 4096,
        },
    )

    plan = _parse_plan(response.text)
    logger.info(f"Plan: {plan.get('task_type')} with {len(plan.get('steps', []))} steps")
    return plan


async def create_recovery_plan(
    original_prompt: str,
    original_plan: dict,
    results: dict,
    failed: list,
) -> dict:
    """Send recovery prompt to Gemini for corrected plan."""
    model = _get_model()

    recovery_prompt = build_recovery_prompt(original_prompt, original_plan, results, failed)
    logger.info("Sending recovery prompt to LLM...")

    response = model.generate_content(
        recovery_prompt,
        generation_config={
            "temperature": 0.0,
            "max_output_tokens": 4096,
        },
    )

    plan = _parse_plan(response.text)
    logger.info(f"Recovery plan: {len(plan.get('steps', []))} steps")
    return plan
```

- [ ] **Step 2: Commit**

```bash
git add tripletex/agent.py
git commit -m "feat(tripletex): add LLM agent with Gemini 3.1 Pro planning + recovery"
```

---

### Task 5: FastAPI Application (main.py)

**Files:**
- Create: `tripletex/main.py`

The main application that wires together the agent, executor, and client.

- [ ] **Step 1: Create main.py**

```python
# tripletex/main.py
import logging
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from agent import create_plan, create_recovery_plan
from executor import execute_plan
from tripletex_client import TripletexClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Tripletex AI Agent")


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/solve")
async def solve(request: Request):
    body = await request.json()
    prompt = body["prompt"]
    files = body.get("files", [])
    creds = body["tripletex_credentials"]

    base_url = creds["base_url"]
    session_token = creds["session_token"]

    logger.info(f"Received task: {prompt[:100]}...")

    client = TripletexClient(base_url, session_token)

    try:
        # Phase 1: Plan
        plan = await create_plan(prompt, files)
        logger.info(f"Plan created: {plan.get('task_type')} ({len(plan.get('steps', []))} steps)")

        # Phase 2: Execute
        result = await execute_plan(plan, client)

        # Phase 3: Recovery (if needed)
        if not result["success"]:
            logger.warning(f"Execution had {len(result['failed'])} failed steps, attempting recovery...")
            try:
                recovery_plan = await create_recovery_plan(
                    prompt, plan, result["results"], result["failed"]
                )
                recovery_result = await execute_plan(recovery_plan, client)
                if recovery_result["success"]:
                    logger.info("Recovery succeeded")
                else:
                    logger.error(f"Recovery also failed: {len(recovery_result['failed'])} steps")
            except Exception as e:
                logger.error(f"Recovery failed with exception: {e}")
        else:
            logger.info("Execution completed successfully")

    except Exception as e:
        logger.error(f"Agent error: {e}", exc_info=True)
    finally:
        await client.close()

    return JSONResponse({"status": "completed"})
```

- [ ] **Step 2: Commit**

```bash
git add tripletex/main.py
git commit -m "feat(tripletex): add FastAPI /solve endpoint with plan-execute-recover loop"
```

---

### Task 6: Dockerfile + requirements.txt

**Files:**
- Create: `tripletex/Dockerfile`
- Create: `tripletex/requirements.txt`

- [ ] **Step 1: Create requirements.txt**

```
fastapi
uvicorn[standard]
httpx
google-cloud-aiplatform
```

- [ ] **Step 2: Create Dockerfile**

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
```

- [ ] **Step 3: Commit**

```bash
git add tripletex/Dockerfile tripletex/requirements.txt
git commit -m "feat(tripletex): add Dockerfile and requirements for Cloud Run"
```

---

### Task 7: Local Testing + Deploy

**Files:**
- No new files — integration testing and deployment

- [ ] **Step 1: Test locally**

```bash
cd tripletex
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8080
```

Then in another terminal:
```bash
curl -X POST http://localhost:8080/solve \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Opprett en ansatt med navn Test Testesen, test@test.no", "files": [], "tripletex_credentials": {"base_url": "https://tx-proxy.ainm.no/v2", "session_token": "test-token"}}'
```

Expected: Returns `{"status": "completed"}` (API calls will fail with auth error but the flow works).

- [ ] **Step 2: Test /health**

```bash
curl http://localhost:8080/health
```

Expected: `{"status": "ok"}`

- [ ] **Step 3: Deploy to Cloud Run**

```bash
cd tripletex
gcloud run deploy tripletex-agent \
  --source . \
  --region europe-north1 \
  --allow-unauthenticated \
  --memory 1Gi \
  --timeout 300 \
  --min-instances 1
```

Expected: URL like `https://tripletex-agent-xxxxx-lz.a.run.app`

- [ ] **Step 4: Test deployed endpoint**

```bash
curl https://<CLOUD_RUN_URL>/health
```

Expected: `{"status": "ok"}`

- [ ] **Step 5: Submit URL on platform**

Go to https://app.ainm.no/submit/tripletex and submit the Cloud Run URL.

- [ ] **Step 6: Commit any fixes**

```bash
git add -A tripletex/
git commit -m "fix(tripletex): fixes from deployment testing"
```

---

### Task 8: Iterate Based on Submission Results

After initial deployment, analyze submission logs and improve:

- [ ] **Step 1: Check submission logs on app.ainm.no**

Review which tasks pass/fail and which fields are incorrect.

- [ ] **Step 2: Add more few-shot examples**

For task types that fail, add specific few-shot examples to `prompts/system.py`.

- [ ] **Step 3: Improve error handling in executor.py**

Add more programmatic fixes for common Tripletex validation errors.

- [ ] **Step 4: Optimize for efficiency bonus**

Remove any unnecessary GET calls. Ensure POST response IDs are used everywhere.

- [ ] **Step 5: Commit improvements**

```bash
git add tripletex/
git commit -m "fix(tripletex): improve accuracy based on submission feedback"
```
