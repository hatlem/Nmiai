# Tripletex AI Agent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an AI agent with `/solve` endpoint that receives accounting tasks in natural language, plans Tripletex API calls via Gemini 3.1 Pro, executes them, and scores maximum points.

**Architecture:** Two-stage LLM with template-based planning. Stage 1: classify task type (lightweight). Stage 2: fill pre-built plan template with extracted values (task-specific context only). Self-repair via LLM on error (no hardcoded fixes). Deterministic executor resolves dependencies.

**Tech Stack:** Python 3.11, FastAPI, httpx, Vertex AI (Gemini 3.1 Pro), Docker, Cloud Run

**Spec:** `docs/superpowers/specs/2026-03-19-tripletex-agent-design.md`

---

## File Structure

```
tripletex/
├── Dockerfile                      # Python 3.11-slim, uvicorn
├── requirements.txt                # fastapi, uvicorn, httpx, google-cloud-aiplatform
├── main.py                         # FastAPI app: POST /solve, GET /health
├── agent.py                        # Two-stage LLM: classify → fill template → self-repair
├── executor.py                     # Execute plan steps, resolve $step_N deps
├── tripletex_client.py             # Async HTTP client: auth, GET/POST/PUT/DELETE
├── templates.py                    # Pre-built plan templates for all 30 task types
├── prompts/
│   ├── __init__.py
│   ├── classifier.py               # Stage 1: classify task type (lightweight prompt)
│   └── planner.py                  # Stage 2: fill template (task-specific context only)
├── schemas/
│   └── api_reference.json          # Compact Tripletex API reference (14 KB, loaded selectively)
└── tests/
    ├── __init__.py
    ├── test_executor.py            # Unit tests for dependency resolution
    └── test_templates.py           # Verify all templates have valid structure
```

---

### Task 1: Tripletex HTTP Client

**Files:**
- Create: `tripletex/tripletex_client.py`

Async HTTP wrapper for all Tripletex API calls. Handles auth, JSON parsing, error extraction.

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

    async def request(
        self, method: str, path: str,
        body: dict | None = None, params: dict | None = None,
    ) -> dict:
        """Make API request. Returns {status_code, ok, data}."""
        url = f"{self.base_url}{path}"
        logger.info(f"{method} {path}")

        response = await self._client.request(
            method=method, url=url, json=body, params=params,
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
            logger.warning(f"{method} {path} -> {response.status_code}: {result['data']}")

        return result

    async def get(self, path: str, params: dict | None = None) -> dict:
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

- [ ] **Step 2: Commit**

```bash
git add tripletex/tripletex_client.py
git commit -m "feat(tripletex): add async HTTP client wrapper"
```

---

### Task 2: Plan Templates

**Files:**
- Create: `tripletex/templates.py`
- Create: `tripletex/tests/__init__.py`
- Create: `tripletex/tests/test_templates.py`

Pre-built plan templates for all known task types. The LLM only needs to extract values and fill placeholders — not design the API flow from scratch. This maximizes correctness and minimizes API calls.

Template placeholders use `{{field_name}}` syntax. The LLM fills these in Stage 2.

- [ ] **Step 1: Write test for template structure**

```python
# tripletex/tests/test_templates.py
import pytest
from templates import TEMPLATES


def test_all_templates_have_required_fields():
    for task_type, template in TEMPLATES.items():
        assert "steps" in template, f"{task_type} missing steps"
        assert "relevant_schemas" in template, f"{task_type} missing relevant_schemas"
        assert "description" in template, f"{task_type} missing description"
        assert len(template["steps"]) > 0, f"{task_type} has empty steps"
        for i, step in enumerate(template["steps"]):
            assert "method" in step, f"{task_type} step {i} missing method"
            assert "path" in step, f"{task_type} step {i} missing path"


def test_all_methods_are_valid():
    valid_methods = {"GET", "POST", "PUT", "DELETE"}
    for task_type, template in TEMPLATES.items():
        for step in template["steps"]:
            assert step["method"] in valid_methods, f"{task_type}: invalid method {step['method']}"


def test_minimum_task_types():
    """We need at least 15 templates to cover common task types."""
    assert len(TEMPLATES) >= 15, f"Only {len(TEMPLATES)} templates, need at least 15"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd tripletex && /opt/homebrew/bin/python3 -m pytest tests/test_templates.py -v`
Expected: FAIL — `templates` module not found.

- [ ] **Step 3: Create templates.py**

```python
# tripletex/templates.py
"""Pre-built plan templates for Tripletex task types.

Each template defines:
- description: What this task type does (shown to LLM in Stage 2)
- relevant_schemas: Which entity schemas to include in Stage 2 context
- steps: Ordered API calls with {{placeholder}} values for LLM to fill
- extract_fields: Fields the LLM must extract from the prompt

The LLM's job is ONLY to extract values and fill placeholders.
"""

TEMPLATES: dict[str, dict] = {

    # ===== EMPLOYEES =====

    "create_employee": {
        "description": "Create an employee, optionally assign a role/entitlement",
        "relevant_schemas": ["Employee"],
        "extract_fields": ["firstName", "lastName", "email", "dateOfBirth", "phoneNumberMobile", "role"],
        "steps": [
            {
                "method": "POST",
                "path": "/employee",
                "body": {
                    "firstName": "{{firstName}}",
                    "lastName": "{{lastName}}",
                    "email": "{{email}}",
                },
            },
        ],
        "conditional_steps": {
            "if_role": {
                "method": "PUT",
                "path": "/employee/entitlement/:grantEntitlementsByTemplate",
                "params": {"employeeId": "$step_0.id", "template": "{{role}}"},
            },
        },
    },

    "update_employee": {
        "description": "Update an existing employee's details (phone, email, address, etc.)",
        "relevant_schemas": ["Employee"],
        "extract_fields": ["search_name", "fields_to_update"],
        "steps": [
            {
                "method": "GET",
                "path": "/employee",
                "params": {"firstName": "{{search_firstName}}", "lastName": "{{search_lastName}}", "fields": "id,firstName,lastName"},
            },
            {
                "method": "PUT",
                "path": "/employee/$step_0.values[0].id",
                "body": "{{fields_to_update}}",
            },
        ],
    },

    # ===== CUSTOMERS =====

    "create_customer": {
        "description": "Create a customer with contact details",
        "relevant_schemas": ["Customer"],
        "extract_fields": ["name", "email", "organizationNumber", "phoneNumber", "isSupplier"],
        "steps": [
            {
                "method": "POST",
                "path": "/customer",
                "body": {
                    "name": "{{name}}",
                    "isCustomer": True,
                    "email": "{{email}}",
                },
            },
        ],
    },

    # ===== PRODUCTS =====

    "create_product": {
        "description": "Create a product with price and VAT settings",
        "relevant_schemas": ["Product"],
        "extract_fields": ["name", "number", "priceExcludingVatCurrency", "priceIncludingVatCurrency", "description"],
        "steps": [
            {
                "method": "POST",
                "path": "/product",
                "body": {
                    "name": "{{name}}",
                    "priceExcludingVatCurrency": "{{price}}",
                },
            },
        ],
    },

    # ===== INVOICING =====

    "create_invoice": {
        "description": "Create an invoice: customer -> order with orderLines -> invoice",
        "relevant_schemas": ["Customer", "Order", "OrderLine", "Invoice"],
        "extract_fields": ["customer_name", "orderLines", "invoiceDate", "invoiceDueDate", "customer_email"],
        "steps": [
            {
                "method": "POST",
                "path": "/customer",
                "body": {
                    "name": "{{customer_name}}",
                    "isCustomer": True,
                },
            },
            {
                "method": "POST",
                "path": "/order",
                "body": {
                    "customer": {"id": "$step_0.id"},
                    "orderDate": "{{orderDate}}",
                    "deliveryDate": "{{deliveryDate}}",
                    "orderLines": "{{orderLines}}",
                },
            },
            {
                "method": "PUT",
                "path": "/order/$step_1.id/:invoice",
                "params": {
                    "invoiceDate": "{{invoiceDate}}",
                    "sendToCustomer": False,
                },
            },
        ],
    },

    "create_invoice_existing_customer": {
        "description": "Create invoice for an existing customer (search by name first)",
        "relevant_schemas": ["Customer", "Order", "OrderLine", "Invoice"],
        "extract_fields": ["customer_name", "orderLines", "invoiceDate", "invoiceDueDate"],
        "steps": [
            {
                "method": "GET",
                "path": "/customer",
                "params": {"name": "{{customer_name}}", "fields": "id,name"},
            },
            {
                "method": "POST",
                "path": "/order",
                "body": {
                    "customer": {"id": "$step_0.values[0].id"},
                    "orderDate": "{{orderDate}}",
                    "deliveryDate": "{{deliveryDate}}",
                    "orderLines": "{{orderLines}}",
                },
            },
            {
                "method": "PUT",
                "path": "/order/$step_1.id/:invoice",
                "params": {
                    "invoiceDate": "{{invoiceDate}}",
                    "sendToCustomer": False,
                },
            },
        ],
    },

    "register_payment": {
        "description": "Register a payment on an existing invoice",
        "relevant_schemas": ["Invoice"],
        "extract_fields": ["invoice_id", "amount", "paymentDate"],
        "steps": [
            {
                "method": "GET",
                "path": "/invoice/paymentType",
                "params": {"fields": "id,description"},
            },
            {
                "method": "PUT",
                "path": "/invoice/{{invoice_id}}/:payment",
                "params": {
                    "paymentDate": "{{paymentDate}}",
                    "paymentTypeId": "$step_0.values[0].id",
                    "paidAmount": "{{amount}}",
                },
            },
        ],
    },

    "create_credit_note": {
        "description": "Create a credit note for an existing invoice",
        "relevant_schemas": ["Invoice"],
        "extract_fields": ["invoice_id", "date", "comment"],
        "steps": [
            {
                "method": "PUT",
                "path": "/invoice/{{invoice_id}}/:createCreditNote",
                "params": {
                    "date": "{{date}}",
                    "comment": "{{comment}}",
                },
            },
        ],
    },

    "send_invoice": {
        "description": "Send an invoice to the customer",
        "relevant_schemas": ["Invoice"],
        "extract_fields": ["invoice_id", "sendType", "email"],
        "steps": [
            {
                "method": "PUT",
                "path": "/invoice/{{invoice_id}}/:send",
                "params": {
                    "sendType": "{{sendType}}",
                },
            },
        ],
    },

    # ===== TRAVEL EXPENSES =====

    "create_travel_expense": {
        "description": "Register a travel expense report with travel details",
        "relevant_schemas": ["TravelExpense", "TravelDetails", "TravelExpenseCost"],
        "extract_fields": ["departureDate", "returnDate", "departureFrom", "destination", "purpose", "costs", "isDayTrip", "isForeignTravel"],
        "steps": [
            {
                "method": "GET",
                "path": "/employee",
                "params": {"fields": "id", "count": 1},
            },
            {
                "method": "POST",
                "path": "/travelExpense",
                "body": {
                    "employee": {"id": "$step_0.values[0].id"},
                    "travelDetails": {
                        "departureDate": "{{departureDate}}",
                        "returnDate": "{{returnDate}}",
                        "departureFrom": "{{departureFrom}}",
                        "destination": "{{destination}}",
                        "purpose": "{{purpose}}",
                        "isDayTrip": "{{isDayTrip}}",
                        "isForeignTravel": "{{isForeignTravel}}",
                    },
                    "title": "{{title}}",
                },
            },
        ],
    },

    "delete_travel_expense": {
        "description": "Delete a travel expense report",
        "relevant_schemas": ["TravelExpense"],
        "extract_fields": ["travel_expense_id"],
        "steps": [
            {
                "method": "DELETE",
                "path": "/travelExpense/{{travel_expense_id}}",
            },
        ],
    },

    "deliver_travel_expense": {
        "description": "Deliver (submit) a travel expense for approval",
        "relevant_schemas": ["TravelExpense"],
        "extract_fields": ["travel_expense_id"],
        "steps": [
            {
                "method": "PUT",
                "path": "/travelExpense/:deliver",
                "params": {"id": "{{travel_expense_id}}"},
            },
        ],
    },

    "approve_travel_expense": {
        "description": "Approve a travel expense",
        "relevant_schemas": ["TravelExpense"],
        "extract_fields": ["travel_expense_id"],
        "steps": [
            {
                "method": "PUT",
                "path": "/travelExpense/:approve",
                "params": {"id": "{{travel_expense_id}}"},
            },
        ],
    },

    # ===== PROJECTS =====

    "create_project": {
        "description": "Create a project, optionally linked to a customer",
        "relevant_schemas": ["Project", "Customer"],
        "extract_fields": ["name", "customer_name", "startDate", "endDate", "isInternal", "projectManager", "description"],
        "steps": [
            {
                "method": "POST",
                "path": "/customer",
                "body": {
                    "name": "{{customer_name}}",
                    "isCustomer": True,
                },
            },
            {
                "method": "POST",
                "path": "/project",
                "body": {
                    "name": "{{project_name}}",
                    "customer": {"id": "$step_0.id"},
                    "startDate": "{{startDate}}",
                    "endDate": "{{endDate}}",
                    "isInternal": False,
                },
            },
        ],
    },

    "create_internal_project": {
        "description": "Create an internal project (no customer)",
        "relevant_schemas": ["Project"],
        "extract_fields": ["name", "startDate", "endDate", "description"],
        "steps": [
            {
                "method": "POST",
                "path": "/project",
                "body": {
                    "name": "{{project_name}}",
                    "isInternal": True,
                    "startDate": "{{startDate}}",
                    "endDate": "{{endDate}}",
                },
            },
        ],
    },

    # ===== DEPARTMENTS =====

    "create_department": {
        "description": "Create a department",
        "relevant_schemas": ["Department"],
        "extract_fields": ["name", "departmentNumber", "departmentManager"],
        "steps": [
            {
                "method": "POST",
                "path": "/department",
                "body": {
                    "name": "{{name}}",
                    "departmentNumber": "{{departmentNumber}}",
                },
            },
        ],
    },

    # ===== SUPPLIERS =====

    "create_supplier": {
        "description": "Create a supplier",
        "relevant_schemas": ["Supplier"],
        "extract_fields": ["name", "organizationNumber", "email", "phoneNumber"],
        "steps": [
            {
                "method": "POST",
                "path": "/supplier",
                "body": {
                    "name": "{{name}}",
                    "email": "{{email}}",
                },
            },
        ],
    },

    # ===== CONTACTS =====

    "create_contact": {
        "description": "Create a contact person for a customer",
        "relevant_schemas": ["Contact", "Customer"],
        "extract_fields": ["firstName", "lastName", "email", "customer_name"],
        "steps": [
            {
                "method": "GET",
                "path": "/customer",
                "params": {"name": "{{customer_name}}", "fields": "id,name"},
            },
            {
                "method": "POST",
                "path": "/contact",
                "body": {
                    "firstName": "{{firstName}}",
                    "lastName": "{{lastName}}",
                    "email": "{{email}}",
                    "customer": {"id": "$step_0.values[0].id"},
                },
            },
        ],
    },

    # ===== LEDGER / VOUCHERS =====

    "create_voucher": {
        "description": "Create a ledger voucher with postings. IMPORTANT: account numbers (e.g. 1920) are NOT IDs — you must first GET /ledger/account?number=X to find the real account ID.",
        "relevant_schemas": ["Voucher", "Posting", "Account"],
        "extract_fields": ["date", "description", "postings_with_account_numbers"],
        "steps": [
            {
                "method": "GET",
                "path": "/ledger/account",
                "params": {"number": "{{debit_account_number}}", "fields": "id,number,name"},
            },
            {
                "method": "GET",
                "path": "/ledger/account",
                "params": {"number": "{{credit_account_number}}", "fields": "id,number,name"},
            },
            {
                "method": "POST",
                "path": "/ledger/voucher",
                "body": {
                    "date": "{{date}}",
                    "description": "{{description}}",
                    "postings": "{{postings_using_account_ids_from_step_0_and_1}}",
                },
            },
        ],
    },

    "reverse_voucher": {
        "description": "Reverse a voucher",
        "relevant_schemas": ["Voucher"],
        "extract_fields": ["voucher_id", "date"],
        "steps": [
            {
                "method": "PUT",
                "path": "/ledger/voucher/{{voucher_id}}/:reverse",
                "params": {"date": "{{date}}"},
            },
        ],
    },

    # ===== CORRECTIONS =====

    "delete_entity": {
        "description": "Delete an entity by type and ID",
        "relevant_schemas": [],
        "extract_fields": ["entity_type", "entity_id"],
        "steps": [
            {
                "method": "DELETE",
                "path": "/{{entity_type}}/{{entity_id}}",
            },
        ],
    },

    # ===== FALLBACK =====

    "unknown": {
        "description": "Task type not recognized — LLM generates plan from scratch using full API reference",
        "relevant_schemas": ["Employee", "Customer", "Product", "Order", "OrderLine", "Invoice", "TravelExpense", "Project", "Department", "Contact", "Supplier", "Voucher", "Posting"],
        "extract_fields": [],
        "steps": [],
    },
}


# Map Norwegian keywords to task types for fast classification
KEYWORD_HINTS: dict[str, list[str]] = {
    "create_employee": ["ansatt", "employee", "empleado", "empregado", "mitarbeiter", "employe"],
    "update_employee": ["oppdater ansatt", "endre ansatt", "update employee"],
    "create_customer": ["kunde", "customer", "cliente", "client", "Kunde"],
    "create_product": ["produkt", "product", "producto", "produto", "Produkt", "produit"],
    "create_invoice": ["faktura", "invoice", "factura", "fatura", "Rechnung", "facture"],
    "register_payment": ["innbetaling", "betaling", "payment", "pago", "pagamento", "Zahlung", "paiement"],
    "create_credit_note": ["kreditnota", "credit note", "nota de credito", "Gutschrift", "avoir"],
    "create_travel_expense": ["reiseregning", "travel expense", "gastos de viaje", "despesas de viagem", "Reisekosten", "note de frais"],
    "delete_travel_expense": ["slett reiseregning", "delete travel"],
    "create_project": ["prosjekt", "project", "proyecto", "projeto", "Projekt", "projet"],
    "create_department": ["avdeling", "department", "departamento", "Abteilung", "departement"],
    "create_supplier": ["leverandor", "supplier", "proveedor", "fornecedor", "Lieferant", "fournisseur"],
    "create_voucher": ["bilag", "voucher", "Beleg", "piece comptable"],
    "reverse_voucher": ["reverser", "reverse", "tilbakefor"],
    "send_invoice": ["send faktura", "send invoice"],
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd tripletex && /opt/homebrew/bin/python3 -m pytest tests/test_templates.py -v`
Expected: All 3 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add tripletex/templates.py tripletex/tests/
git commit -m "feat(tripletex): add pre-built plan templates for 20+ task types"
```

---

### Task 3: Execution Engine

**Files:**
- Create: `tripletex/executor.py`
- Create: `tripletex/tests/test_executor.py`

Deterministic executor that runs plan steps and resolves `$step_N.field` dependencies. No hardcoded error fixes — all error recovery goes through LLM self-repair in agent.py.

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
    results = {0: {"data": {"values": [{"id": 7, "description": "Cash"}, {"id": 8}]}}}
    assert resolve_ref("$step_0.values[0].id", results) == 7


def test_resolve_ref_deep_nested():
    """Test deeply nested path resolution."""
    results = {0: {"data": {"value": {"customer": {"id": 5}}}}}
    assert resolve_ref("$step_0.customer.id", results) == 5


def test_resolve_ref_non_string():
    """Non-string values pass through unchanged."""
    assert resolve_ref(42, {}) == 42
    assert resolve_ref(True, {}) is True
    assert resolve_ref(None, {}) is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd tripletex && /opt/homebrew/bin/python3 -m pytest tests/test_executor.py -v`
Expected: FAIL — `executor` module not found.

- [ ] **Step 3: Create executor.py**

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


def resolve_ref(value, results: dict):
    """Resolve $step_N.path.to.field references in a string value.
    Supports nested paths like $step_0.values[0].id and $step_1.value.id"""
    if not isinstance(value, str):
        return value

    pattern = r'\$step_(\d+)\.([\w\[\]\.]+)'

    def _resolve_single(step_idx: int, field_path: str):
        step_data = results.get(step_idx, {}).get("data", {})
        parts = field_path.split(".")

        # Try value.path first (POST/PUT responses wrap in {"value": {...}})
        val = step_data.get("value", {})
        if isinstance(val, dict):
            resolved = _deep_get(val, parts)
            if resolved is not None:
                return resolved
            if parts[0] == "value" and len(parts) > 1:
                resolved = _deep_get(val, parts[1:])
                if resolved is not None:
                    return resolved

        # Try direct path on step_data
        resolved = _deep_get(step_data, parts)
        if resolved is not None:
            return resolved

        return None

    # If entire string is a single reference, return typed value (int, not "42")
    single_match = re.fullmatch(pattern, value)
    if single_match:
        step_idx = int(single_match.group(1))
        field_path = single_match.group(2)
        resolved = _resolve_single(step_idx, field_path)
        if resolved is not None:
            return resolved

    # Otherwise do string substitution (for paths like "/order/$step_1.id/:invoice")
    def replacer(match):
        step_idx = int(match.group(1))
        field_path = match.group(2)
        resolved = _resolve_single(step_idx, field_path)
        if resolved is not None:
            return str(resolved)
        logger.warning(f"Could not resolve $step_{step_idx}.{field_path}")
        return match.group(0)

    return re.sub(pattern, replacer, value)


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
    Returns {success, results, failed}. No error fixing here — that's the LLM's job."""
    steps = plan.get("steps", [])
    results = {}
    failed = []

    for i, step in enumerate(steps):
        method = step["method"].upper()
        path = resolve_ref(step.get("path", ""), results)
        body = resolve_refs(step.get("body"), results) if step.get("body") else None
        params = resolve_refs(step.get("params"), results) if step.get("params") else None

        logger.info(f"Step {i}: {method} {path}")
        response = await client.request(method, path, body=body, params=params)
        results[i] = response

        if not response["ok"]:
            failed.append((i, response))
            logger.error(f"Step {i} failed: {response['status_code']} - {response['data']}")
            # Don't stop — continue with remaining steps (some may still work)

    return {
        "success": len(failed) == 0,
        "results": results,
        "failed": failed,
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd tripletex && /opt/homebrew/bin/python3 -m pytest tests/test_executor.py -v`
Expected: All 8 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add tripletex/executor.py tripletex/tests/
git commit -m "feat(tripletex): add execution engine with nested dependency resolution"
```

---

### Task 4: Prompts (Classifier + Planner)

**Files:**
- Create: `tripletex/prompts/__init__.py`
- Create: `tripletex/prompts/classifier.py`
- Create: `tripletex/prompts/planner.py`

Two-stage prompt system:
- Stage 1 (classifier): Lightweight prompt to identify task type from the 20+ known types
- Stage 2 (planner): Task-specific prompt with only relevant schemas + template to fill

- [ ] **Step 1: Create prompts/__init__.py**

Empty file.

- [ ] **Step 2: Create prompts/classifier.py**

```python
# tripletex/prompts/classifier.py
"""Stage 1: Classify the accounting task type.

Lightweight prompt — no API reference needed.
Returns just the task_type string.
"""

from templates import TEMPLATES, KEYWORD_HINTS

TASK_LIST = "\n".join(
    f"- {task_type}: {t['description']}"
    for task_type, t in TEMPLATES.items()
    if task_type != "unknown"
)

CLASSIFIER_PROMPT = f"""You are a task classifier for an accounting system. Given a task prompt (in any language: Norwegian, English, Spanish, Portuguese, Nynorsk, German, or French), identify which task type it belongs to.

## Available Task Types
{TASK_LIST}

## Output
Return ONLY a JSON object:
{{"task_type": "<one of the task types above>"}}

If the task doesn't match any known type, return:
{{"task_type": "unknown"}}

Do not include any other text, markdown, or explanation.
"""
```

- [ ] **Step 3: Create prompts/planner.py**

```python
# tripletex/prompts/planner.py
"""Stage 2: Fill a plan template with values extracted from the prompt.

Only loads relevant entity schemas for the detected task type.
"""
import json
from pathlib import Path
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
- PUT /order/{id}/:invoice — invoiceDate (required), sendToCustomer (optional). NOTE: invoiceDueDate is set on the Invoice object AFTER creation, or use invoicesDueIn on the Order to control due date.
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

    # Format results concisely
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
```

- [ ] **Step 4: Commit**

```bash
git add tripletex/prompts/
git commit -m "feat(tripletex): add two-stage prompts (classifier + planner + self-repair)"
```

---

### Task 5: LLM Agent (Two-Stage + Self-Repair)

**Files:**
- Create: `tripletex/agent.py`

Two-stage LLM agent:
- Stage 1: Classify task type (lightweight, ~100 tokens out)
- Stage 2: Fill template with task-specific context (focused, ~500 tokens out)
- Self-repair: On execution failure, feed errors to LLM for corrected plan (max 1 retry)

- [ ] **Step 1: Create agent.py**

```python
# tripletex/agent.py
import json
import base64
import logging
from datetime import date

import vertexai
from vertexai.generative_models import GenerativeModel, Part

from prompts.classifier import CLASSIFIER_PROMPT
from prompts.planner import build_planner_prompt, build_self_repair_prompt
from templates import TEMPLATES, KEYWORD_HINTS

logger = logging.getLogger(__name__)

vertexai.init(project="ainm26osl-710", location="europe-north1")

MODEL_ID = "gemini-3.1-pro"


def _get_model(system_instruction: str) -> GenerativeModel:
    return GenerativeModel(MODEL_ID, system_instruction=system_instruction)


def _parse_json(text: str) -> dict:
    """Parse LLM response, stripping markdown fences if present."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:])
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        logger.warning(f"JSON parse failed: {e}. Trying to extract JSON from response...")
        # Try to find JSON object in the response
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            return json.loads(text[start:end])
        raise


def _quick_classify(prompt: str) -> str | None:
    """Try keyword-based classification before calling LLM."""
    prompt_lower = prompt.lower()
    for task_type, keywords in KEYWORD_HINTS.items():
        for kw in keywords:
            if kw.lower() in prompt_lower:
                return task_type
    return None


async def classify_task(prompt: str) -> str:
    """Stage 1: Classify the task type. Try keywords first, fall back to LLM."""
    # Fast path: keyword match
    quick = _quick_classify(prompt)
    if quick:
        logger.info(f"Quick classify: {quick}")
        return quick

    # Slow path: LLM classification
    model = _get_model(CLASSIFIER_PROMPT)
    response = await model.generate_content_async(
        prompt,
        generation_config={"temperature": 0.0, "max_output_tokens": 100},
    )
    result = _parse_json(response.text)
    task_type = result.get("task_type", "unknown")
    logger.info(f"LLM classify: {task_type}")
    return task_type


async def create_plan(prompt: str, files: list[dict] | None = None) -> dict:
    """Two-stage planning: classify then fill template."""
    # Stage 1: Classify
    task_type = await classify_task(prompt)

    # Stage 2: Plan with task-specific context
    planner_prompt = build_planner_prompt(task_type)
    model = _get_model(planner_prompt)

    parts = []
    if files:
        for f in files:
            file_data = base64.b64decode(f["content_base64"])
            parts.append(Part.from_data(data=file_data, mime_type=f["mime_type"]))
            parts.append(Part.from_text(f"[Attached file: {f['filename']}]"))

    today = date.today().isoformat()
    parts.append(Part.from_text(
        f"Today's date is {today}. Complete this accounting task:\n\n{prompt}"
    ))

    logger.info(f"Planning {task_type}: {prompt[:80]}...")
    response = await model.generate_content_async(
        parts,
        generation_config={"temperature": 0.0, "max_output_tokens": 4096},
    )

    try:
        plan = _parse_json(response.text)
    except (json.JSONDecodeError, Exception) as e:
        logger.error(f"Failed to parse plan JSON: {e}. Raw: {response.text[:200]}")
        # Return a minimal plan that at least tries the template steps
        template = TEMPLATES.get(task_type, TEMPLATES["unknown"])
        plan = {"task_type": task_type, "reasoning": "Fallback — LLM JSON parse failed", "steps": template["steps"]}

    plan["task_type"] = plan.get("task_type", task_type)
    logger.info(f"Plan: {plan['task_type']} with {len(plan.get('steps', []))} steps")
    return plan


async def self_repair(
    original_prompt: str,
    plan: dict,
    results: dict,
    failed: list,
) -> dict:
    """Self-repair: feed errors to LLM for a corrected plan."""
    task_type = plan.get("task_type", "unknown")
    repair_prompt = build_self_repair_prompt(
        task_type, original_prompt, plan, results, failed
    )
    model = _get_model("You are an expert Tripletex API debugger. Fix the failed plan.")

    logger.info("Self-repair: sending errors to LLM...")
    response = await model.generate_content_async(
        repair_prompt,
        generation_config={"temperature": 0.0, "max_output_tokens": 4096},
    )

    repaired = _parse_json(response.text)
    logger.info(f"Repaired plan: {len(repaired.get('steps', []))} steps")
    return repaired
```

- [ ] **Step 2: Commit**

```bash
git add tripletex/agent.py
git commit -m "feat(tripletex): add two-stage LLM agent with self-repair"
```

---

### Task 6: FastAPI Application

**Files:**
- Create: `tripletex/main.py`

Wires together: classify → plan → execute → self-repair → execute again.

- [ ] **Step 1: Create main.py**

```python
# tripletex/main.py
import time
import logging
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from agent import create_plan, self_repair
from executor import execute_plan
from tripletex_client import TripletexClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Tripletex AI Agent")

# Time budget: 5 min total, reserve 60s for self-repair
MAX_PLAN_EXECUTE_SECONDS = 240
REPAIR_DEADLINE_SECONDS = 290  # leave 10s buffer before 300s timeout

# Simple task types where recovery cost > benefit (only 1 step, low tier)
SKIP_RECOVERY_TYPES = {
    "create_customer", "create_product", "create_department",
    "create_supplier", "delete_travel_expense", "delete_entity",
}


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/solve")
async def solve(request: Request):
    start = time.monotonic()
    body = await request.json()
    prompt = body["prompt"]
    files = body.get("files", [])
    creds = body["tripletex_credentials"]

    base_url = creds["base_url"]
    session_token = creds["session_token"]

    logger.info(f"Task: {prompt[:100]}...")

    client = TripletexClient(base_url, session_token)

    try:
        # Phase 1: Plan (two-stage: classify + fill template)
        plan = await create_plan(prompt, files)
        task_type = plan.get("task_type", "unknown")
        logger.info(f"Plan: {task_type} ({len(plan.get('steps', []))} steps)")

        # Phase 2: Execute
        result = await execute_plan(plan, client)

        # Phase 3: Self-repair (conditional)
        elapsed = time.monotonic() - start
        should_repair = (
            not result["success"]
            and task_type not in SKIP_RECOVERY_TYPES
            and elapsed < MAX_PLAN_EXECUTE_SECONDS
        )

        if should_repair:
            logger.warning(f"{len(result['failed'])} steps failed, self-repairing... ({elapsed:.0f}s elapsed)")
            try:
                repaired_plan = await self_repair(
                    prompt, plan, result["results"], result["failed"]
                )
                # Check time budget before executing repair
                if time.monotonic() - start < REPAIR_DEADLINE_SECONDS:
                    repair_result = await execute_plan(repaired_plan, client)
                    if repair_result["success"]:
                        logger.info("Self-repair succeeded")
                    else:
                        logger.error(f"Self-repair failed: {len(repair_result['failed'])} steps")
                else:
                    logger.warning("Skipping repair execution — time budget exceeded")
            except Exception as e:
                logger.error(f"Self-repair exception: {e}")
        elif not result["success"]:
            logger.warning(f"Skipping recovery for {task_type} (simple task or time exceeded)")
        else:
            logger.info(f"All steps succeeded in {elapsed:.1f}s")

    except Exception as e:
        logger.error(f"Agent error: {e}", exc_info=True)
    finally:
        await client.close()

    return JSONResponse({"status": "completed"})
```

- [ ] **Step 2: Commit**

```bash
git add tripletex/main.py
git commit -m "feat(tripletex): add FastAPI /solve with plan-execute-repair loop"
```

---

### Task 7: Dockerfile + Requirements + Deploy

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

- [ ] **Step 3: Create .dockerignore**

```
openapi.json
api-paths-relevant.json
api-reference-compact.json
api-reference-agent.json
api-schemas-key.json
api-schemas-writable.json
docs/
tests/
__pycache__/
*.pyc
.git/
```

- [ ] **Step 4: Commit**

```bash
git add tripletex/Dockerfile tripletex/requirements.txt tripletex/.dockerignore
git commit -m "feat(tripletex): add Dockerfile, requirements, .dockerignore"
```

- [ ] **Step 5: Test locally**

```bash
cd tripletex && pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8080
```

In another terminal:
```bash
curl http://localhost:8080/health
curl -X POST http://localhost:8080/solve \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Opprett en ansatt Ola Nordmann, ola@test.no", "files": [], "tripletex_credentials": {"base_url": "https://tx-proxy.ainm.no/v2", "session_token": "test"}}'
```

- [ ] **Step 6: Deploy to Cloud Run**

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

- [ ] **Step 7: Test deployed endpoint**

```bash
curl https://<CLOUD_RUN_URL>/health
```

- [ ] **Step 8: Submit URL at app.ainm.no**

Go to https://app.ainm.no/submit/tripletex and submit the Cloud Run URL.

- [ ] **Step 9: Commit any deployment fixes**

```bash
git add tripletex/
git commit -m "fix(tripletex): deployment fixes"
```

---

### Task 8: Iterate Based on Submissions

After initial deployment, analyze results and improve:

- [ ] **Step 1: Review submission logs at app.ainm.no**

Check which task types pass/fail and which fields are incorrect.

- [ ] **Step 2: Add templates for failing task types**

For tasks not covered by templates, add new entries in `templates.py`.

- [ ] **Step 3: Refine planner prompts**

If the LLM extracts wrong values, add task-specific hints in `planner.py`.

- [ ] **Step 4: Add more keyword hints**

Expand `KEYWORD_HINTS` in `templates.py` for faster classification.

- [ ] **Step 5: Test with sandbox credentials**

Use the sandbox account to manually test specific task types before resubmitting.

- [ ] **Step 6: Commit improvements**

```bash
git add tripletex/
git commit -m "fix(tripletex): improve accuracy based on submission feedback"
```
