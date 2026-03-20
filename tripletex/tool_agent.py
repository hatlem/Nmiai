"""Tripletex Tool-Use Agent — LLM as primary problem solver.

Instead of rigid templates with brittle step references, this gives
Gemini 3.1 Pro direct API access and lets it solve tasks dynamically.

The LLM:
1. Reads the task prompt (any of 7 languages)
2. Decides which API calls to make
3. Reads responses and adapts
4. Handles errors naturally (reads error message, tries different approach)
5. Continues until task is done

This eliminates all template bugs: no step references, no extraction step
that loses information, no rigid plans that can't handle edge cases.
"""

import asyncio
import base64
import json
import logging
import time
from datetime import date

import vertexai
from vertexai.generative_models import (
    FunctionDeclaration,
    GenerativeModel,
    Part,
    Tool,
)

import warnings
warnings.filterwarnings("ignore", message=".*REST async clients.*")
warnings.filterwarnings("ignore", message=".*deprecated.*")

from tripletex_client import TripletexClient

logger = logging.getLogger(__name__)

# NOTE: vertexai.init() is called in agent.py with location="global".
# Do NOT re-init here — it overrides the global location and breaks 3.1 models.
# tool_agent uses its own init when called directly (see tool_agent_solve).

MAX_TURNS = 20
DEADLINE_BUFFER = 25  # stop 25s before timeout

# ── Tripletex API tool definitions ────────────────────────────────────

_tripletex_get = FunctionDeclaration(
    name="tripletex_get",
    description="GET request to Tripletex API. Use for searching/listing entities. Response includes 'value' (single entity) or 'values' (list). Always use fields param to limit response size.",
    parameters={
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "API path, e.g. /customer, /employee/123, /ledger/account"},
            "params": {"type": "object", "description": "Query parameters as key-value pairs, e.g. {\"name\": \"Acme\", \"fields\": \"id,name\"}"},
        },
        "required": ["path"],
    },
)

_tripletex_post = FunctionDeclaration(
    name="tripletex_post",
    description="POST request to Tripletex API. Use for creating new entities. Returns the created entity with its ID.",
    parameters={
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "API path, e.g. /customer, /employee, /order"},
            "body": {"type": "object", "description": "Request body as JSON object"},
            "params": {"type": "object", "description": "Optional query parameters"},
        },
        "required": ["path", "body"],
    },
)

_tripletex_put = FunctionDeclaration(
    name="tripletex_put",
    description="PUT request to Tripletex API. Use for updating entities or triggering actions (/:invoice, /:payment, /:send, /:approve, /:deliver, /:reverse).",
    parameters={
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "API path, e.g. /customer/123, /order/456/:invoice, /invoice/789/:payment"},
            "body": {"type": "object", "description": "Request body (for updates, include id and version)"},
            "params": {"type": "object", "description": "Query parameters (for actions like /:payment, use params for paymentDate, paidAmount etc.)"},
        },
        "required": ["path"],
    },
)

_tripletex_delete = FunctionDeclaration(
    name="tripletex_delete",
    description="DELETE request to Tripletex API. Use for deleting entities.",
    parameters={
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "API path with entity ID, e.g. /travelExpense/123"},
        },
        "required": ["path"],
    },
)

TOOLS = [Tool(function_declarations=[_tripletex_get, _tripletex_post, _tripletex_put, _tripletex_delete])]

# ── System prompt with comprehensive API knowledge ────────────────────

SYSTEM_PROMPT = f"""\
You are an expert Tripletex accounting agent. Today is {date.today().isoformat()}.
You receive accounting tasks in Norwegian, English, German, French, Spanish, Portuguese, or Nynorsk.
Execute each task by making Tripletex API calls using the provided tools.

## CRITICAL RULES
- The sandbox starts EMPTY — create all prerequisites (customer, supplier, employee) before dependent entities.
- Account numbers are NOT account IDs. Always GET /ledger/account?number=X&fields=id first.
- Every invoice needs a bank account. If invoicing fails with "bankkontonummer", try POST /bank to register one.
- All IDs must be integers, not strings.
- When updating entities, include both 'id' and 'version' from the GET response.
- Amounts must be numbers, not strings.
- Dates must be YYYY-MM-DD format.
- Voucher postings: row starts from 1 (NEVER 0), MUST include amountGrossCurrency (same as amountGross), MUST include vatType.
- Contact: phone field is phoneNumberMobile (NOT phoneNumber — that field doesn't exist on contact).
- Order: MUST include both orderDate AND deliveryDate (both required).
- Project: projectManager MUST have ALL_PRIVILEGES entitlement before being assigned.
- Reminder: use dispatchType=EMAIL (NOT sendType/sendMethod).
- TTC/inkl mva amounts: for voucher postings with vatType 1 or 3, amountGross should be the NET amount (Tripletex adds VAT automatically).

## ENTITY CREATION PATTERNS

### Customer
POST /customer {{"name":"X", "isCustomer":true, "email":"x@y.no", "organizationNumber":"123456789", "phoneNumber":"12345678"}}
- postalAddress: {{"addressLine1":"X", "postalCode":"1234", "city":"Oslo"}}
- For both customer AND supplier: add "isSupplier":true

### Employee
POST /employee {{"firstName":"X", "lastName":"Y", "email":"x@y.no", "dateOfBirth":"1990-01-01", "phoneNumberMobile":"99887766", "userType":"STANDARD", "department":{{"id":DEPT_ID}}}}
- MUST GET /department first and include department.id (required field!)
- Phone field is phoneNumberMobile (NOT phoneNumber, NOT mobileNumber — these cause 422)
- email field is immutable after creation (cannot be changed via PUT)
- Role/admin: after creating, PUT /employee/entitlement/:grantEntitlementsByTemplate?employeeId=ID&template=ALL_PRIVILEGES
- Templates: ALL_PRIVILEGES, INVOICING_MANAGER, PERSONELL_MANAGER, ACCOUNTANT, AUDITOR, DEPARTMENT_LEADER
- For employment: POST /employee/employment {{"employee":{{"id":X}}, "startDate":"2026-01-01"}} — ONLY these 2 fields, NO employmentType/percentageOfFullTimeEquivalent/userType

### Invoice (create order → invoice it)
1. POST /customer (if new)
2. POST /order {{"customer":{{"id":X}}, "orderDate":"YYYY-MM-DD", "deliveryDate":"YYYY-MM-DD", "orderLines":[{{"description":"Item", "count":1, "unitPriceExcludingVatCurrency":1000}}]}}
3. PUT /order/ORDER_ID/:invoice?sendToCustomer=false&invoiceDate=YYYY-MM-DD
- With payment: GET /invoice/paymentType first, then PUT /invoice/INV_ID/:payment?paymentDate=YYYY-MM-DD&paymentTypeId=X&paidAmount=AMOUNT

### Voucher (bilag)
GET /ledger/account?number=XXXX&fields=id for each account number, then:
POST /ledger/voucher {{"date":"YYYY-MM-DD", "description":"X", "postings":[
  {{"row":1, "account":{{"id":DEBIT_ACCT_ID}}, "amountGross":AMOUNT, "amountGrossCurrency":AMOUNT, "vatType":{{"id":VAT_ID}}}},
  {{"row":2, "account":{{"id":CREDIT_ACCT_ID}}, "amountGross":-AMOUNT, "amountGrossCurrency":-AMOUNT, "vatType":{{"id":VAT_ID}}}}
]}}
- vatType: 0=no VAT (1xxx,2xxx,5xxx,8xxx accounts), 1=incoming 25% (4xxx,6xxx,7xxx), 3=outgoing 25% (3xxx)
- Postings MUST sum to zero. Row starts from 1.

### Supplier Invoice (leverandørfaktura)
POST /supplier (create supplier), GET /ledger/account for accounts, then:
POST /supplierInvoice {{"invoiceNumber":"X", "invoiceDate":"YYYY-MM-DD", "supplier":{{"id":X}},
  "voucher":{{"date":"YYYY-MM-DD", "description":"X", "postings":[...]}}
}}
- DO NOT include: orderDate, deliveryDate, dueDate (cause 422)

### Project
GET /employee?count=1&fields=id (for project manager), then:
POST /project {{"name":"X", "startDate":"YYYY-MM-DD", "projectManager":{{"id":EMPLOYEE_ID}}, "isInternal":false, "customer":{{"id":CUST_ID}}}}
- Internal: set isInternal:true, omit customer

### Other endpoints
- POST /department {{"name":"X", "departmentNumber":123}}
- POST /product {{"name":"X", "number":"P001", "priceExcludingVatCurrency":1000}}
- POST /supplier {{"name":"X", "organizationNumber":"123456789"}}
- POST /contact {{"firstName":"X", "lastName":"Y", "email":"x@y.no", "customer":{{"id":X}}}}
- PUT /invoice/ID/:send?sendType=EMAIL — send invoice (sendType UPPERCASE: EMAIL, EHF, EFAKTURA)
- PUT /invoice/ID/:createCreditNote — credit note
- PUT /ledger/voucher/ID/:reverse?date=YYYY-MM-DD — reverse voucher
### Travel Expense (reiseregning)
1. GET /employee?firstName=X&fields=id OR POST /employee to create
2. POST /travelExpense {{"title":"X", "employee":{{"id":X}}, "travelDetails":{{"departureDate":"X", "returnDate":"X", "destination":"X"}}, "isDayTrip":false, "isForeignTravel":false}}
3. GET /travelExpense/costCategory?fields=id,description to find cost category IDs
4. For EACH cost: POST /travelExpense/cost {{"travelExpense":{{"id":TE_ID}}, "date":"YYYY-MM-DD", "costCategory":{{"id":CAT_ID}}, "paymentType":{{"id":0}}, "currency":{{"code":"NOK"}}, "costCurrency":AMOUNT, "vatType":{{"id":0}}, "isRefund":false}}
- NEVER use /travelExpense/ID/expenses or /travelExpense/ID/expense — those don't exist!
- NEVER use /travelExpense/type or /expenseType — those don't exist!
- NEVER include "expenses" field in POST/PUT /travelExpense body — use /travelExpense/cost separately!
- costCategory IDs: check GET /travelExpense/costCategory first
- DELETE /travelExpense/ID — delete travel expense
- PUT /travelExpense/ID/:deliver — submit travel expense
- PUT /travelExpense/ID/:approve — approve travel expense
- POST /purchaseOrder {{"supplier":{{"id":X}}, "ourContact":{{"id":X}}, "deliveryDate":"X"}} then POST /purchaseOrder/orderline separately
- POST /salary/transaction {{"year":2026, "month":3, "payslips":[{{"employee":{{"id":X}}}}]}}
- POST /bank/reconciliation {{"account":{{"id":X}}, "type":"MANUAL", "dateFrom":"X"}}
- POST /asset {{"name":"X", "dateOfAcquisition":"X", "acquisitionCost":X}}
- PUT /invoice/ID/:createReminder?dispatchType=EMAIL
- PUT /company/modules {{"moduleAccountingInternal":true}} — enable modules

## LANGUAGE GLOSSARY
faktura=invoice, kunde=customer, ansatt=employee, leverandør=supplier, bilag=voucher, konto=account,
prosjekt=project, avdeling=department, produkt=product, reiseregning=travel expense, innbetaling=payment,
kreditnota=credit note, purring=reminder, bankavstemming=bank reconciliation, åpningsbalanse=opening balance,
anleggsmiddel=fixed asset, lønn=salary, innkjøpsordre=purchase order, kontaktperson=contact person,
ansettelse=employment, bokfør=post/book, reverser=reverse, godkjenn=approve, lever=deliver, slett=delete

## STRATEGY
1. Parse the task to understand what needs to be done
2. Create prerequisites first (customer before invoice, accounts before voucher)
3. Make API calls one at a time, using returned IDs in subsequent calls
4. If a call fails, read the error message and adapt (don't repeat the same call)
5. When done, stop. Don't make unnecessary verification calls.
"""


async def tool_agent_solve(
    prompt: str,
    files: list[dict] | None,
    client: TripletexClient,
    deadline: float,
) -> bool:
    """Run the tool-use agent. Returns True if task completed without errors."""

    vertexai.init(project="ainm26osl-710", location="global")
    model = GenerativeModel(
        "gemini-3.1-pro-preview",
        system_instruction=SYSTEM_PROMPT,
        tools=TOOLS,
    )
    logger.info("Tool agent using gemini-3.1-pro-preview (global)")

    # Build initial user message
    parts = []
    if files:
        for f in files:
            try:
                data = base64.b64decode(f.get("content_base64", ""))
                parts.append(Part.from_data(data=data, mime_type=f.get("mime_type", "application/octet-stream")))
                parts.append(Part.from_text(f"[Attached: {f.get('filename', 'file')}]"))
            except Exception:
                pass
    parts.append(Part.from_text(f"Execute this accounting task:\n\n{prompt}"))

    chat = model.start_chat()
    had_errors = False

    for turn in range(MAX_TURNS):
        remaining = deadline - time.monotonic()
        if remaining < DEADLINE_BUFFER:
            logger.warning(f"Tool agent: deadline approaching ({remaining:.0f}s), stopping at turn {turn}")
            break

        # Send message to LLM
        try:
            response = await asyncio.wait_for(
                chat.send_message_async(
                    parts,
                    generation_config={"temperature": 0.0, "max_output_tokens": 4096},
                ),
                timeout=min(60.0, remaining - DEADLINE_BUFFER),
            )
        except asyncio.TimeoutError:
            logger.error(f"Tool agent: LLM timeout at turn {turn}")
            had_errors = True
            break
        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "Resource exhausted" in error_str:
                logger.warning(f"Tool agent: 429 rate limit at turn {turn}, retrying in 3s")
                await asyncio.sleep(3)
                try:
                    response = await asyncio.wait_for(
                        chat.send_message_async(user_parts),
                        timeout=min(60.0, remaining - DEADLINE_BUFFER),
                    )
                except Exception as e2:
                    logger.error(f"Tool agent: retry also failed: {e2}")
                    had_errors = True
                    break
            else:
                logger.error(f"Tool agent: LLM error at turn {turn}: {e}")
                had_errors = True
                break

        # Check if LLM wants to call functions
        candidates = response.candidates
        if not candidates:
            logger.info("Tool agent: no candidates, assuming done")
            break

        content = candidates[0].content
        has_function_calls = any(
            hasattr(part, 'function_call') and part.function_call is not None and part.function_call.name
            for part in content.parts
        )

        if not has_function_calls:
            # LLM responded with text only — it's done
            text = response.text if hasattr(response, 'text') else ""
            logger.info(f"Tool agent: text response at turn {turn} (done): {text[:200]}")
            break

        # Execute all function calls
        function_responses = []
        for part in content.parts:
            if not hasattr(part, 'function_call') or not part.function_call.name:
                continue

            fc = part.function_call
            fn_name = fc.name
            args = dict(fc.args) if fc.args else {}

            path = args.get("path", "")
            body = args.get("body")
            params = args.get("params")

            # Ensure path starts with /
            if path and not path.startswith("/"):
                path = "/" + path

            # Map function name to HTTP method
            method_map = {
                "tripletex_get": "GET",
                "tripletex_post": "POST",
                "tripletex_put": "PUT",
                "tripletex_delete": "DELETE",
            }
            method = method_map.get(fn_name, "GET")

            logger.info(f"Tool agent turn {turn}: {method} {path}")

            # Execute
            try:
                result = await client.request(method, path, body=body, params=params)
            except Exception as e:
                result = {"ok": False, "status_code": 0, "data": {"error": str(e)}}

            ok = result.get("ok", False)
            status = result.get("status_code", 0)
            data = result.get("data", {})

            if not ok:
                had_errors = True
                logger.warning(f"Tool agent: {method} {path} -> {status} FAIL")

            # Summarize response for LLM (keep tokens low)
            summary = _compact_response(data, ok)

            function_responses.append(
                Part.from_function_response(
                    name=fn_name,
                    response={"result": summary},
                )
            )

        # Send function results back to LLM
        parts = function_responses

    return not had_errors


def _compact_response(data: dict, ok: bool, max_len: int = 1500) -> dict:
    """Compact API response to save tokens while keeping essential info."""
    if not ok:
        # For errors, keep full detail so LLM can learn
        return {"ok": False, "error": json.dumps(data, ensure_ascii=False, default=str)[:max_len]}

    # For success, extract the useful parts
    if isinstance(data, dict):
        if "value" in data:
            val = data["value"]
            if isinstance(val, dict):
                # Keep essential fields, drop verbose ones
                compact = {}
                for k, v in val.items():
                    if k in ("id", "version", "name", "number", "firstName", "lastName",
                             "email", "invoiceNumber", "amount", "status", "orderId",
                             "organizationNumber", "bankAccountNumber", "startDate",
                             "isCustomer", "isSupplier", "isInternal"):
                        compact[k] = v
                return {"ok": True, "value": compact}
            return {"ok": True, "value": val}

        if "values" in data:
            vals = data["values"]
            if isinstance(vals, list):
                compact_list = []
                for item in vals[:10]:
                    if isinstance(item, dict):
                        compact_list.append({
                            k: v for k, v in item.items()
                            if k in ("id", "version", "name", "number", "type",
                                     "description", "bankAccountNumber")
                        })
                    else:
                        compact_list.append(item)
                return {"ok": True, "count": len(vals), "values": compact_list}

    return {"ok": True, "data": json.dumps(data, ensure_ascii=False, default=str)[:max_len]}
