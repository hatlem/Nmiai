# tripletex/react_agent.py
"""ReAct-style tool-use agent for Tripletex. Fallback when templates fail."""

import json
import time
import base64
import logging
import re

import vertexai
from vertexai.generative_models import GenerativeModel, Part

import warnings
warnings.filterwarnings("ignore", message=".*REST async clients.*")
warnings.filterwarnings("ignore", message=".*deprecated.*")

from tripletex_client import TripletexClient

logger = logging.getLogger(__name__)

vertexai.init(project="ainm26osl-710", location="global")

MODEL = "gemini-3.1-pro-preview"
MAX_ROUNDS = 15
MAX_PARSE_FAILURES = 3
DEADLINE_BUFFER = 30  # seconds

SYSTEM_PROMPT = """\
You are a Tripletex accounting agent. Execute tasks via API calls one at a time.

## API Reference (base already handled)
POST /customer {"name","isCustomer":true,"email","phoneNumber","postalAddress":{"addressLine1","city","postalCode","country":{"id":161}}}
POST /supplier {"name","isSupplier":true,"email","phoneNumber","bankAccountNumber","postalAddress":{...}}
POST /employee {"firstName","lastName","email","dateOfBirth","startDate","department":{"id":X},"userType":"STANDARD","allowInformationRegistration":true}
POST /employee/:entitlementGrant {"employeeId","entitlementTemplateType":"ALL_PRIVILEGES"}
POST /product {"name","number","priceExcludingVat","vatType":{"id":3},"productUnit":{"id":1}}
POST /department {"name","departmentNumber"}
POST /project {"name","number","projectManagerId","department":{"id":X},"startDate","isInternal":false,"projectCategory":{"id":X}}
POST /order {"customer":{"id":X},"orderDate","deliveryDate","orderLines":[{"product":{"id":X},"count":N}]}
POST /invoice {"customer":{"id":X},"invoiceDate","invoiceDueDate","orders":[{"id":X}]}
POST /invoice/:invoice {"id":X} — register invoice
POST /invoice/:payment {"id":X,"paymentDate","paymentTypeId":X,"paidAmount":X}
POST /invoice/:send {"id":X,"sendType":"EMAIL","email":"x"}
POST /invoice/:createCreditNote {"id":X}
POST /ledger/voucher {"date","description","postings":[{"debit":{"id":acctId},"credit":{"id":acctId},"amount":X,"description":""}]}
GET /ledger/account?number=X — get account by number (returns id in values[0].id)
GET /ledger/vatType — list VAT types
GET /employee?count=1 — get default employee
GET /department?count=1 — get default department
GET /project/category — list project categories
GET /invoice/paymentType — list payment types
GET /company/1 — get company info
PUT /<entity>/{id} — update (include version from GET)
DELETE /<entity>/{id} — delete

## Rules
- Sandbox starts empty. Create prerequisites (customer, product, dept) before dependent entities.
- Account numbers != IDs. Use GET /ledger/account?number=X to resolve.
- Employee needs department.id and userType:"STANDARD".
- Orders need orderDate + deliveryDate.
- Invoices: create order first, then POST /invoice with orders:[{id:X}], then POST /invoice/:invoice to register.
- Voucher postings: row 0 has debit account, row 1 has credit account. Each posting needs debit OR credit (not both).
- Entitlement: POST /employee/:entitlementGrant?employeeId=X&template=ALL_PRIVILEGES (query params, no body).
- Country Norway: {"id":161}. VAT type 3 = 25% MVA.

## Norwegian glossary
faktura=invoice, kunde=customer, ansatt=employee, leverandor=supplier, bilag=voucher, konto=account, prosjekt=project, avdeling=department, produkt=product, ordre=order, kreditnota=credit note, reiseregning=travel expense, innbetaling=payment

## Response format
Reply with exactly ONE JSON object per turn:
{"action":"api_call","method":"POST","path":"/customer","body":{"name":"X","isCustomer":true},"params":null,"reasoning":"Create customer"}
or when done:
{"action":"done","reasoning":"All tasks completed"}
"""


def _extract_json(text: str) -> dict | None:
    """Extract JSON object from LLM response, handling markdown code blocks."""
    text = text.strip()
    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Try extracting from code block
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    # Try finding first { ... } block
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    return None


async def react_solve(
    prompt: str,
    files: list[dict] | None,
    client: TripletexClient,
    deadline: float,
) -> bool:
    """Run ReAct agent loop. Returns True if completed successfully."""
    model = GenerativeModel(MODEL, system_instruction=SYSTEM_PROMPT)

    # Build initial content parts
    parts: list[Part | str] = [f"Task:\n{prompt}"]

    # Add file attachments
    if files:
        for f in files:
            try:
                data = base64.b64decode(f.get("content_base64", ""))
                mime = f.get("mime_type", "application/octet-stream")
                parts.append(Part.from_data(data=data, mime_type=mime))
                parts.append(f"[Attached file: {f.get('filename', 'unknown')}]")
            except Exception as e:
                logger.warning(f"Failed to attach file: {e}")

    history = [{"role": "user", "parts": parts}]
    parse_failures = 0
    all_ok = True

    for round_num in range(MAX_ROUNDS):
        # Check deadline
        remaining = deadline - time.monotonic()
        if remaining < DEADLINE_BUFFER:
            logger.warning(f"ReAct: deadline approaching ({remaining:.0f}s left), stopping")
            break

        # Call LLM
        try:
            import asyncio
            chat = model.start_chat(history=history[:-1]) if len(history) > 1 else model.start_chat()
            response = await asyncio.wait_for(
                chat.send_message_async(
                    history[-1]["parts"],
                    generation_config={"temperature": 0.1, "max_output_tokens": 2048},
                ),
                timeout=min(45.0, remaining - DEADLINE_BUFFER),
            )
            response_text = response.text.strip()
        except Exception as e:
            logger.error(f"ReAct: LLM call failed: {e}")
            all_ok = False
            break

        logger.info(f"ReAct round {round_num+1}: {response_text[:200]}")

        # Parse response
        action = _extract_json(response_text)
        if action is None:
            parse_failures += 1
            if parse_failures >= MAX_PARSE_FAILURES:
                logger.error("ReAct: too many parse failures, aborting")
                all_ok = False
                break
            # Add assistant response and retry prompt
            history.append({"role": "model", "parts": [response_text]})
            history.append({"role": "user", "parts": ["Please respond with valid JSON only. One JSON object with action field."]})
            continue

        parse_failures = 0  # reset on success

        # Add assistant response to history
        history.append({"role": "model", "parts": [response_text]})

        # Handle done
        if action.get("action") == "done":
            logger.info(f"ReAct: done after {round_num+1} rounds. Reason: {action.get('reasoning', '')}")
            break

        # Handle api_call
        if action.get("action") == "api_call":
            method = action.get("method", "GET").upper()
            path = action.get("path", "")
            body = action.get("body")
            params = action.get("params")

            if not path:
                history.append({"role": "user", "parts": ["Error: missing 'path' in api_call. Try again."]})
                continue

            # Ensure path starts with /
            if not path.startswith("/"):
                path = "/" + path

            # Execute API call
            try:
                result = await client.request(method, path, body=body, params=params)
            except Exception as e:
                result = {"ok": False, "status_code": 0, "data": {"error": str(e)}}

            # Build concise result summary
            status = result.get("status_code", 0)
            ok = result.get("ok", False)
            data = result.get("data", {})

            if ok:
                # Trim response to avoid token bloat
                summary = _summarize_response(data)
                feedback = f"OK {status}: {summary}"
            else:
                all_ok = False
                # Include full error for LLM to learn from
                error_text = json.dumps(data, ensure_ascii=False, default=str)[:1500]
                feedback = f"ERROR {status}: {error_text}"

            logger.info(f"ReAct API: {method} {path} -> {status} {'OK' if ok else 'FAIL'}")
            history.append({"role": "user", "parts": [feedback]})
        else:
            # Unknown action
            history.append({"role": "user", "parts": [
                f"Unknown action '{action.get('action')}'. Use 'api_call' or 'done'."
            ]})

    return all_ok


def _summarize_response(data: dict, max_len: int = 800) -> str:
    """Summarize API response to keep context window manageable."""
    # Extract the value(s) from Tripletex response envelope
    if isinstance(data, dict):
        if "value" in data:
            val = data["value"]
            if isinstance(val, dict):
                # Single entity - keep id, version, and a few key fields
                keep = {k: v for k, v in val.items()
                        if k in ("id", "version", "name", "number", "firstName", "lastName",
                                 "invoiceNumber", "orderId", "url", "status") or v is None}
                if keep:
                    return json.dumps(keep, ensure_ascii=False, default=str)
            return json.dumps(val, ensure_ascii=False, default=str)[:max_len]
        if "values" in data:
            vals = data["values"]
            if isinstance(vals, list):
                # List - summarize count + first few items
                summaries = []
                for item in vals[:5]:
                    if isinstance(item, dict):
                        s = {k: v for k, v in item.items()
                             if k in ("id", "version", "name", "number", "type", "description")}
                        summaries.append(s)
                    else:
                        summaries.append(item)
                result = {"count": len(vals), "first": summaries}
                return json.dumps(result, ensure_ascii=False, default=str)[:max_len]

    text = json.dumps(data, ensure_ascii=False, default=str)
    return text[:max_len]
