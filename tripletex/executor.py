"""Plan executor with dependency-graph-based parallel execution.

Builds a DAG from $step_N references (and optional explicit depends_on),
then executes independent steps concurrently via asyncio.gather.
"""

import asyncio
import re
import time
import logging
from datetime import datetime, timedelta
from typing import Any

from tripletex_client import TripletexClient

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Reference helpers
# ---------------------------------------------------------------------------

def _deep_get(obj: Any, path_parts: list[str]) -> Any:
    """Navigate nested dicts/lists by dot-separated path parts.
    Supports array indexing like 'values[0]'."""
    current = obj
    for part in path_parts:
        if current is None:
            return None
        array_match = re.match(r'(\w+)\[(-?\d+)\]', part)
        if array_match:
            key, idx = array_match.group(1), int(array_match.group(2))
            if isinstance(current, dict) and key in current:
                current = current[key]
                if isinstance(current, list) and -len(current) <= idx < len(current):
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


def resolve_ref(value: Any, results: dict[int, dict]) -> Any:
    """Resolve $step_N.path.to.field references in a string value."""
    if not isinstance(value, str):
        return value

    pattern = r'\$step_(\d+)\.([\w\[\]\.]+)'

    def _resolve_single(step_idx: int, field_path: str) -> Any:
        step_result = results.get(step_idx, {})
        step_data = step_result.get("data", {})
        parts = field_path.split(".")

        # Try resolving from the "value" object first (POST/PUT responses)
        val = step_data.get("value", {})
        if isinstance(val, dict):
            resolved = _deep_get(val, parts)
            if resolved is not None:
                return resolved
            if parts[0] == "value" and len(parts) > 1:
                resolved = _deep_get(val, parts[1:])
                if resolved is not None:
                    return resolved

        # Try from the raw response data
        resolved = _deep_get(step_data, parts)
        if resolved is not None:
            return resolved

        # Shortcut: $step_N.id should resolve from value.id
        if parts == ["id"] and isinstance(val, dict) and "id" in val:
            return val["id"]

        # Debug logging when resolution fails
        data_keys = list(step_data.keys()) if isinstance(step_data, dict) else type(step_data).__name__
        values_info = f", values count={len(step_data['values'])}" if isinstance(step_data, dict) and "values" in step_data else ""
        logger.warning(
            f"resolve_ref: $step_{step_idx}.{field_path} returned None. "
            f"step ok={step_result.get('ok')}, status={step_result.get('status_code')}, "
            f"data keys={data_keys}, value type={type(val).__name__}{values_info}"
        )
        return None

    single_match = re.fullmatch(pattern, value)
    if single_match:
        step_idx = int(single_match.group(1))
        field_path = single_match.group(2)
        resolved = _resolve_single(step_idx, field_path)
        if resolved is not None:
            return resolved

    def replacer(match: re.Match) -> str:
        step_idx = int(match.group(1))
        field_path = match.group(2)
        resolved = _resolve_single(step_idx, field_path)
        if resolved is not None:
            return str(resolved)
        logger.warning(f"Could not resolve $step_{step_idx}.{field_path}")
        return match.group(0)

    return re.sub(pattern, replacer, value)


def resolve_refs(obj: Any, results: dict[int, dict]) -> Any:
    """Recursively resolve all $step_N.field references in a dict/list/string."""
    if isinstance(obj, str):
        return resolve_ref(obj, results)
    if isinstance(obj, dict):
        return {k: resolve_refs(v, results) for k, v in obj.items()}
    if isinstance(obj, list):
        return [resolve_refs(item, results) for item in obj]
    return obj


def _fill_placeholders(obj: Any, extracted_values: dict) -> Any:
    """Replace {{placeholder}} strings with values from extracted_values."""
    if not extracted_values:
        return obj
    if isinstance(obj, str):
        # Full match: "{{name}}" -> extracted_values["name"]
        m = re.fullmatch(r'\{\{(\w+)\}\}', obj)
        if m:
            key = m.group(1)
            if key in extracted_values:
                return extracted_values[key]
        # Partial match: "prefix {{name}} suffix" -> string interpolation
        def _replacer(match):
            key = match.group(1)
            return str(extracted_values.get(key, match.group(0)))
        result = re.sub(r'\{\{(\w+)\}\}', _replacer, obj)
        return result
    if isinstance(obj, dict):
        return {k: _fill_placeholders(v, extracted_values) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_fill_placeholders(item, extracted_values) for item in obj]
    return obj


def _strip_unresolved_placeholders(obj: Any) -> Any:
    """Remove fields that still contain {{placeholder}}, unresolved $step_N values,
    or None values from resolved references (e.g. empty search results)."""
    if isinstance(obj, dict):
        cleaned = {}
        for k, v in obj.items():
            v = _strip_unresolved_placeholders(v)
            if v is None:
                logger.debug(f"Stripping None field '{k}' (likely unresolved reference)")
                continue
            if isinstance(v, str) and re.search(r'\{\{.*?\}\}', v):
                logger.warning(f"Stripping unresolved placeholder field '{k}': {v}")
                continue
            if isinstance(v, str) and re.search(r'\$step_\d+', v):
                logger.warning(f"Stripping unresolved $step_N reference field '{k}': {v}")
                continue
            # Strip dict/list that became empty after recursive cleaning
            if isinstance(v, dict) and not v:
                logger.warning(f"Stripping empty dict field '{k}' (likely unresolved reference)")
                continue
            cleaned[k] = v
        return cleaned
    if isinstance(obj, list):
        cleaned_list = [_strip_unresolved_placeholders(item) for item in obj]
        return [item for item in cleaned_list if item is not None]
    return obj


# ---------------------------------------------------------------------------
# Dependency graph
# ---------------------------------------------------------------------------

_STEP_REF_RE = re.compile(r'\$step_(\d+)')


def _find_refs_in_obj(obj: Any) -> set[int]:
    """Recursively find all $step_N references in an arbitrary object."""
    refs: set[int] = set()
    if isinstance(obj, str):
        refs.update(int(m) for m in _STEP_REF_RE.findall(obj))
    elif isinstance(obj, dict):
        for v in obj.values():
            refs.update(_find_refs_in_obj(v))
    elif isinstance(obj, list):
        for item in obj:
            refs.update(_find_refs_in_obj(item))
    return refs


def _build_dependency_graph(steps: list[dict]) -> dict[int, set[int]]:
    """Parse $step_N references to build {step_idx: set(dependency_indices)}."""
    graph: dict[int, set[int]] = {}
    for i, step in enumerate(steps):
        explicit = step.get("depends_on")
        if explicit is not None:
            graph[i] = set(explicit)
            continue

        deps: set[int] = set()
        for field in ("path", "body", "params"):
            if field in step and step[field] is not None:
                deps.update(_find_refs_in_obj(step[field]))

        # Force sequential execution for /travelExpense/cost steps
        # (Tripletex locks the travel expense, parallel POSTs cause 409)
        path = step.get("path", "")
        if "/travelExpense/cost" in str(path) and i > 0:
            deps.add(i - 1)

        skip_ref = step.get("skip_if_exists")
        if isinstance(skip_ref, str):
            deps.update(int(m) for m in _STEP_REF_RE.findall(skip_ref))

        deps.discard(i)
        deps = {d for d in deps if d < i}
        graph[i] = deps
    return graph


def _topological_layers(graph: dict[int, set[int]], num_steps: int) -> list[list[int]]:
    """Group steps into layers for parallel execution."""
    completed: set[int] = set()
    remaining = set(range(num_steps))
    layers: list[list[int]] = []

    while remaining:
        ready = [i for i in sorted(remaining) if graph.get(i, set()).issubset(completed)]
        if not ready:
            logger.warning(f"Dependency cycle detected among steps {remaining}, falling back to sequential")
            layers.append(sorted(remaining))
            break
        layers.append(ready)
        completed.update(ready)
        remaining -= set(ready)
    return layers


# ---------------------------------------------------------------------------
# Pre-validation helpers
# ---------------------------------------------------------------------------

_ENDPOINT_INVALID_FIELDS: dict[str, set[str]] = {
    "/employee/employment": {"userType", "employmentType", "percentageOfFullTimeEquivalent", "type", "role", "department"},
    "/contact": {"phoneNumber"},  # Use phoneNumberMobile or phoneNumberWork
    "/supplier": {"isSupplier"},  # Never set on supplier endpoint
}

# Required fields per endpoint — add defaults for missing ones on POST
_REQUIRED_DEFAULTS: dict[str, dict[str, Any]] = {
    "/employee": {"userType": "STANDARD"},
    "/customer": {"isCustomer": True},
    "/supplier": {},  # isSupplier is auto-set by API, do not send
    "/order": {},  # orderDate and deliveryDate handled by date defaults below
    "/project": {},
    "/contact": {},
    "/product": {},
}


def _pre_validate_body(method: str, path: str, body: dict | None, params: dict | None) -> dict | None:
    """Pre-validate and clean request body/params to prevent 4xx errors.
    Returns cleaned body (or None if no body)."""
    if body is None:
        return None

    # Strip known-invalid fields for specific endpoints
    for endpoint_pattern, bad_fields in _ENDPOINT_INVALID_FIELDS.items():
        if endpoint_pattern in path:
            # /employee/employment fields are valid on /employee/employment/details
            if endpoint_pattern == "/employee/employment" and "/employment/details" in path:
                continue
            for bf in bad_fields:
                if bf in body:
                    logger.warning(f"Pre-validate: stripping invalid field '{bf}' from {path}")
                    body = {k: v for k, v in body.items() if k != bf}

    cleaned = {}
    for k, v in body.items():
        # Skip None values - API will reject them
        if v is None:
            continue
        # Skip empty strings for optional fields
        if v == "" and k not in ("name", "firstName", "lastName"):
            continue
        # orderLine prices must be positive (credit notes create positive invoice then reverse)
        if k in ("unitPriceExcludingVatCurrency", "unitPriceIncludingVatCurrency") and isinstance(v, (int, float)) and v < 0:
            v = abs(v)
        # Ensure amounts are numbers, not strings
        if k in ("amount", "amountGross", "amountGrossCurrency", "paidAmount",
                  "priceExcludingVatCurrency", "priceIncludingVatCurrency",
                  "amountCurrencyIncVat", "acquisitionCost", "hours",
                  "percentageOfFullTimeEquivalent"):
            if isinstance(v, str):
                try:
                    v = float(v)
                    if v == int(v):
                        v = int(v)
                except (ValueError, TypeError):
                    pass
        # Ensure boolean fields are actual booleans
        if k in ("isCustomer", "isSupplier", "isInternal", "isDayTrip",
                  "isForeignTravel", "sendToCustomer"):
            if isinstance(v, str):
                v = v.lower() in ("true", "1", "yes", "ja")
        # Clean nested dicts — but use "/" as path to avoid adding top-level defaults
        if isinstance(v, dict):
            v = _pre_validate_body(method, "/", v, None) or v
        # Clean lists of dicts
        if isinstance(v, list):
            v = [_pre_validate_body(method, "/", item, None) if isinstance(item, dict) else item for item in v]
        cleaned[k] = v

    # Fix voucher postings: row must start from 1 (row 0 is reserved/system-generated)
    if "postings" in cleaned and isinstance(cleaned["postings"], list):
        for i, posting in enumerate(cleaned["postings"]):
            if isinstance(posting, dict):
                if "row" not in posting or posting.get("row", 0) == 0:
                    posting["row"] = i + 1
                # Ensure amountGrossCurrency is set if amountGross is present
                if "amountGross" in posting and "amountGrossCurrency" not in posting:
                    posting["amountGrossCurrency"] = posting["amountGross"]
                # Auto-assign vatType based on account patterns if not set
                # This is a heuristic - the LLM should set it, but this catches misses
                if "vatType" not in posting:
                    # Default to no VAT (id=0) — safest default for balance sheet accounts
                    posting["vatType"] = {"id": 0}

    # POST-specific defaults to prevent 422 errors
    if method == "POST":
        path_lower = path.lower() if isinstance(path, str) else ""

        # Apply required defaults from the lookup table
        for endpoint, defaults in _REQUIRED_DEFAULTS.items():
            if endpoint in path:
                for field, default_val in defaults.items():
                    if field not in cleaned:
                        logger.info(f"Pre-validate: defaulting '{field}' to {default_val!r} for {path}")
                        cleaned[field] = default_val

        # Extra guard: employee (not employment) must have userType
        if "/employee" in path_lower and "/employment" not in path_lower and "userType" not in cleaned:
            cleaned["userType"] = "STANDARD"

        if "/order" in path_lower:
            from datetime import date as d
            today = d.today().isoformat()
            if "orderDate" not in cleaned:
                cleaned["orderDate"] = today
            if "deliveryDate" not in cleaned:
                cleaned["deliveryDate"] = cleaned.get("orderDate", today)

            # Validate order lines: ensure minimum required fields
            if "orderLines" in cleaned and isinstance(cleaned["orderLines"], list):
                for line in cleaned["orderLines"]:
                    if isinstance(line, dict):
                        line.setdefault("count", 1)
                        # productNumber is not a valid Tripletex orderLine field
                        line.pop("productNumber", None)
                        line.pop("product_number", None)
                        # vatType is REQUIRED on orderLines — default to 25% outgoing (id=3)
                        if "vatType" not in line:
                            line["vatType"] = {"id": 3}
                        elif isinstance(line.get("vatType"), (int, float)):
                            # Convert bare number to object
                            vat_n = int(line["vatType"])
                            _pct_map = {25: 3, 15: 33, 12: 31, 0: 5, 6: 6}
                            line["vatType"] = {"id": _pct_map.get(vat_n, vat_n)}
                        elif isinstance(line.get("vatType"), str) and line["vatType"].isdigit():
                            vat_n = int(line["vatType"])
                            _pct_map = {25: 3, 15: 33, 12: 31, 0: 5, 6: 6}
                            line["vatType"] = {"id": _pct_map.get(vat_n, vat_n)}

        if path.rstrip("/") == "/customer" and "isCustomer" not in cleaned:
            cleaned["isCustomer"] = True

    # Strip fields that don't exist on certain endpoints
    if isinstance(path, str):
        if "/employee/employment" in path and "/employment/details" not in path:
            # These fields are invalid on /employee/employment but VALID on /employee/employment/details
            for bad_field in ("userType", "employmentType", "percentageOfFullTimeEquivalent", "type", "role", "department", "email"):
                if bad_field in cleaned:
                    logger.warning(f"Pre-validate: stripping '{bad_field}' from /employee/employment")
                    cleaned.pop(bad_field, None)

    # Fix common LLM field name mistakes (synonyms the API doesn't accept)
    _FIELD_RENAMES = {"quantity": "count", "amount": "unitPriceExcludingVatCurrency"}
    for wrong, right in _FIELD_RENAMES.items():
        if wrong in cleaned and right not in cleaned:
            cleaned[right] = cleaned.pop(wrong)
            logger.info(f"Pre-validate: renamed '{wrong}' -> '{right}'")
    # Also fix in nested orderLines
    if "orderLines" in cleaned and isinstance(cleaned["orderLines"], list):
        for ol in cleaned["orderLines"]:
            if isinstance(ol, dict):
                for wrong, right in _FIELD_RENAMES.items():
                    if wrong in ol and right not in ol:
                        ol[right] = ol.pop(wrong)

    # Smart date defaults for invoice-related fields
    if "orderDate" in cleaned and "deliveryDate" not in cleaned:
        cleaned["deliveryDate"] = cleaned["orderDate"]
    if "invoiceDate" in cleaned and "invoiceDueDate" not in cleaned:
        try:
            inv_date = datetime.strptime(cleaned["invoiceDate"], "%Y-%m-%d")
            cleaned["invoiceDueDate"] = (inv_date + timedelta(days=14)).strftime("%Y-%m-%d")
        except (ValueError, TypeError):
            pass
    if "invoiceDate" in cleaned and "orderDate" not in cleaned:
        cleaned["orderDate"] = cleaned["invoiceDate"]

    # Strip fields invalid for POST /supplierInvoice (must run AFTER date defaults above)
    # KEEP invoiceDueDate and amountCurrency — they are REQUIRED for /supplierInvoice!
    if isinstance(path, str) and "/supplierInvoice" in path and "/orderline" not in path.lower():
        for bad_field in ("orderDate", "deliveryDate", "dueDate", "orderLines"):
            cleaned.pop(bad_field, None)
        # Ensure invoiceDueDate is set (required field)
        if "invoiceDueDate" not in cleaned and "invoiceDate" in cleaned:
            try:
                inv_date = datetime.strptime(cleaned["invoiceDate"], "%Y-%m-%d")
                cleaned["invoiceDueDate"] = (inv_date + timedelta(days=30)).strftime("%Y-%m-%d")
            except (ValueError, TypeError):
                pass
        # Ensure amountCurrency is set (required field) — compute from postings if missing
        if "amountCurrency" not in cleaned:
            voucher = cleaned.get("voucher", {})
            postings = voucher.get("postings", []) if isinstance(voucher, dict) else []
            for p in postings:
                if isinstance(p, dict):
                    amt = p.get("amountGross") or p.get("amountGrossCurrency")
                    if isinstance(amt, (int, float)) and amt > 0:
                        cleaned["amountCurrency"] = amt
                        break

    return cleaned


_ENTITLEMENT_NORMALIZE = {
    "administrator": "ALL_PRIVILEGES", "admin": "ALL_PRIVILEGES",
    "kontoadministrator": "ALL_PRIVILEGES", "all_privileges": "ALL_PRIVILEGES",
    "regnskapsforer": "ACCOUNTANT", "regnskapsfører": "ACCOUNTANT",
    "rekneskapsforar": "ACCOUNTANT", "accountant": "ACCOUNTANT",
    "revisor": "AUDITOR", "auditor": "AUDITOR",
    "lonnansvarlig": "PERSONELL_MANAGER", "lønnansvarlig": "PERSONELL_MANAGER",
    "personell_manager": "PERSONELL_MANAGER",
    "fakturaansvarlig": "INVOICING_MANAGER", "invoicing_manager": "INVOICING_MANAGER",
    "avdelingsleder": "DEPARTMENT_LEADER", "department_leader": "DEPARTMENT_LEADER",
}


def _pre_validate_params(params: dict | None, path: str = "") -> dict | None:
    """Clean query params - ensure proper types."""
    if params is None:
        return None
    cleaned = {}
    for k, v in params.items():
        if v is None or v == "":
            continue
        # Normalize entitlement template names
        if k == "template" and isinstance(v, str):
            normalized = _ENTITLEMENT_NORMALIZE.get(v.lower().strip())
            if normalized:
                logger.info(f"Pre-validate params: normalized template '{v}' -> '{normalized}'")
                v = normalized
        # Amount params should be numbers
        if k in ("paidAmount", "amount"):
            if isinstance(v, str):
                try:
                    v = float(v)
                except (ValueError, TypeError):
                    pass
        # Boolean params
        if k in ("sendToCustomer",):
            if isinstance(v, str):
                v = v.lower() in ("true", "1", "yes")
        # Integer params (IDs)
        if k in ("paymentTypeId", "employeeId", "id"):
            if isinstance(v, str) and v.isdigit():
                v = int(v)
        # Normalize dispatchType for reminders
        if k == "dispatchType" and isinstance(v, str):
            v = v.upper()
            if v not in ("EMAIL", "SMS", "OWN_PRINTER", "NETS_PRINT", "SFTP", "API", "LETTER"):
                v = "EMAIL"
        cleaned[k] = v
    # /:send uses sendType, /:createReminder uses dispatchType — don't mix them up
    if "/:send" in path:
        # For /:send, ensure sendType is used (not dispatchType)
        if "dispatchType" in cleaned and "sendType" not in cleaned:
            cleaned["sendType"] = cleaned.pop("dispatchType")
            logger.info("Pre-validate params: renamed 'dispatchType' -> 'sendType' for /:send")
    else:
        # Fix wrong param names for reminders
        for wrong in ("sendType", "sendTypes", "sendMethod", "selectedReminderSendTypes"):
            if wrong in cleaned and "dispatchType" not in cleaned:
                cleaned["dispatchType"] = cleaned.pop(wrong)
                logger.info(f"Pre-validate params: renamed '{wrong}' -> 'dispatchType'")
            elif wrong in cleaned:
                del cleaned[wrong]
    return cleaned


# ---------------------------------------------------------------------------
# Step execution
# ---------------------------------------------------------------------------

async def _execute_step(
    idx: int,
    step: dict,
    results: dict[int, dict],
    client: TripletexClient,
    extracted_values: dict | None = None,
) -> tuple[int, dict, bool]:
    """Execute a single step. Returns (index, response, ok)."""
    method = step.get("method", "GET").upper()
    path = resolve_ref(step.get("path", ""), results)

    # Strip double-brace placeholders from path (e.g. {{employee_id}} that didn't resolve)
    if isinstance(path, str) and re.search(r'\{\{.*?\}\}', path):
        logger.warning(f"Step {idx}: stripping unresolved placeholders from path '{path}'")
        path = re.sub(r'\{\{.*?\}\}', '', path)

    if "$step_" in str(path):
        logger.error(f"Step {idx}: unresolved reference in path '{path}'")
        response = {"status_code": 0, "ok": False, "data": {"error": f"unresolved path reference: {path}"}}
        return idx, response, False

    body = resolve_refs(step.get("body"), results) if step.get("body") else None
    params = resolve_refs(step.get("params"), results) if step.get("params") else None

    # Fill {{placeholders}} with extracted values from the prompt
    if extracted_values:
        if body:
            body = _fill_placeholders(body, extracted_values)
        if params:
            params = _fill_placeholders(params, extracted_values)

    if body:
        body = _strip_unresolved_placeholders(body)
    if params:
        params = _strip_unresolved_placeholders(params)

    if body:
        body = _pre_validate_body(method, path, body, params)
    if params:
        params = _pre_validate_params(params, path)

    # Log body for POST/PUT to help debug 422 errors
    if method in ("POST", "PUT") and body:
        import json
        logger.info(f"Step {idx}: {method} {path} body: {json.dumps(body, default=str, ensure_ascii=False)[:500]}")

    # Fallback: if this is a /:payment step and paymentTypeId was stripped (unresolved),
    # fetch it on-the-fly from GET /invoice/paymentType
    if "/:payment" in str(path) and (params is None or "paymentTypeId" not in params):
        logger.warning(f"Step {idx}: paymentTypeId missing for /:payment — fetching on-the-fly")
        if params is None:
            params = {}
        try:
            pt_resp = await client.request("GET", "/invoice/paymentType", params={"fields": "id,description"})
            if pt_resp.get("ok"):
                pt_data = pt_resp.get("data", {})
                pt_values = pt_data.get("values", [])
                if not pt_values and isinstance(pt_data.get("value"), dict):
                    pt_values = pt_data["value"].get("values", [])
                if pt_values and isinstance(pt_values[0], dict) and "id" in pt_values[0]:
                    params["paymentTypeId"] = pt_values[0]["id"]
                    logger.info(f"Step {idx}: resolved paymentTypeId={pt_values[0]['id']} via fallback")
                else:
                    logger.error(f"Step {idx}: GET /invoice/paymentType returned no values: {pt_data}")
            else:
                logger.error(f"Step {idx}: GET /invoice/paymentType failed: {pt_resp.get('status_code')}")
        except Exception as e:
            logger.error(f"Step {idx}: fallback GET /invoice/paymentType exception: {e}")

    # Fallback: if /:payment is missing paidAmount, try to get invoice amount
    if "/:payment" in str(path) and params and "paidAmount" not in params:
        # Extract invoice ID from path (e.g. /invoice/12345/:payment)
        inv_match = re.search(r'/invoice/(\d+)', str(path))
        if inv_match:
            inv_id = inv_match.group(1)
            logger.warning(f"Step {idx}: paidAmount missing — fetching invoice {inv_id} amount")
            try:
                inv_resp = await client.request("GET", f"/invoice/{inv_id}", params={"fields": "id,amount"})
                if inv_resp.get("ok"):
                    inv_data = inv_resp.get("data", {})
                    inv_val = inv_data.get("value", inv_data)
                    if isinstance(inv_val, dict) and "amount" in inv_val:
                        params["paidAmount"] = inv_val["amount"]
                        logger.info(f"Step {idx}: resolved paidAmount={inv_val['amount']} from invoice")
            except Exception as e:
                logger.error(f"Step {idx}: fallback GET invoice amount exception: {e}")

    logger.info(f"Step {idx}: {method} {path}")

    try:
        response = await client.request(method, path, body=body, params=params)
    except Exception as e:
        logger.error(f"Step {idx} exception: {e}")
        response = {"status_code": 0, "ok": False, "data": {"error": str(e)}}

    # Fallback: if step has fallback_on_500 and we got a 500, retry with alternate path/body
    fallback = step.get("fallback_on_500")
    if fallback and response.get("status_code") == 500:
        fb_path = fallback.get("path", path)
        fb_body = fallback.get("body")
        if fb_body:
            fb_body = resolve_refs(fb_body, results)
            if extracted_values:
                fb_body = _fill_placeholders(fb_body, extracted_values)
            fb_body = _strip_unresolved_placeholders(fb_body)
            fb_body = _pre_validate_body(method, fb_path, fb_body, params)
        logger.warning(f"Step {idx}: {path} returned 500, retrying with fallback {fb_path}")
        try:
            response = await client.request(method, fb_path, body=fb_body, params=params)
        except Exception as e2:
            logger.error(f"Step {idx} fallback exception: {e2}")
            response = {"status_code": 0, "ok": False, "data": {"error": str(e2)}}

    # Handle vatType lock errors: retry with vatType 0 on affected postings
    if response.get("status_code") == 422 and body:
        error_data = response.get("data", {})
        msgs = error_data.get("validationMessages", [])
        has_vat_lock = any("låst til mva-kode" in (m.get("message", "") or "").lower() for m in msgs)
        if has_vat_lock:
            # Find postings at top level or inside voucher (supplierInvoice)
            postings_lists = []
            if "postings" in body:
                postings_lists.append(body["postings"])
            if isinstance(body.get("voucher"), dict) and "postings" in body["voucher"]:
                postings_lists.append(body["voucher"]["postings"])
            if postings_lists:
                logger.warning(f"Step {idx}: vatType lock detected, retrying ALL postings with vatType 0")
                for postings in postings_lists:
                    for posting in postings:
                        if isinstance(posting, dict):
                            posting["vatType"] = {"id": 0}
                try:
                    response = await client.request(method, path, body=body, params=params)
                except Exception:
                    pass

    # Handle "already exists" 422 errors by searching for the existing entity
    if method == "POST" and response.get("status_code") == 422:
        error_data = response.get("data", {})
        validation_msgs = error_data.get("validationMessages", [])
        is_duplicate = any(
            "allerede" in (m.get("message", "") or "").lower()
            or "already" in (m.get("message", "") or "").lower()
            or "i bruk" in (m.get("message", "") or "").lower()
            or "in use" in (m.get("message", "") or "").lower()
            for m in validation_msgs
        )
        if is_duplicate and body:
            logger.warning(f"Step {idx}: POST {path} duplicate detected, searching for existing entity")
            # Build search params from body fields
            # Use only universally safe fields for each entity type
            _ENTITY_FIELDS = {
                "/employee": "id,firstName,lastName,email",
                "/customer": "id,name,email,organizationNumber",
                "/supplier": "id,name,email,organizationNumber",
                "/product": "id,name,number",
                "/department": "id,name",
                "/project": "id,name",
            }
            entity_fields = "id,name"
            for ep, fields in _ENTITY_FIELDS.items():
                if path.rstrip("/") == ep or path.startswith(ep + "/"):
                    entity_fields = fields
                    break
            search_params = {"fields": entity_fields}
            search_key = None
            # For products, prioritize "number" (unique product number) over "name"
            is_product = path.rstrip("/") == "/product" or path.startswith("/product/")
            if is_product:
                search_order = ("number", "name", "email", "organizationNumber", "firstName")
            else:
                search_order = ("name", "email", "organizationNumber", "number", "firstName")
            for key in search_order:
                if key in body and body[key]:
                    search_key = key
                    search_params[key] = str(body[key])
                    break
            # For products: if no search key found yet, try productNumber alias
            if is_product and search_key is None and "productNumber" in body and body["productNumber"]:
                search_key = "number"
                search_params["number"] = str(body["productNumber"])
            if search_key:
                try:
                    search_resp = await client.request("GET", path, params=search_params)
                    if search_resp.get("ok"):
                        s_data = search_resp.get("data", {})
                        s_values = s_data.get("values", [])
                        if not s_values:
                            inner = s_data.get("value", {})
                            if isinstance(inner, dict):
                                s_values = inner.get("values", [])
                        if s_values:
                            # Found existing entity — return it as if POST succeeded
                            existing = s_values[0]
                            logger.info(f"Step {idx}: found existing entity id={existing.get('id')} via search")
                            response = {
                                "status_code": 200, "ok": True,
                                "data": {"value": existing},
                                "recovered_duplicate": True,
                            }
                except Exception as e2:
                    logger.warning(f"Step {idx}: duplicate recovery search failed: {e2}")

    # Detect empty search results that will break downstream references
    if method == "GET" and response["ok"]:
        data = response.get("data", {})
        # Check both direct values and nested value.values
        values = data.get("values", [])
        if not values:
            inner = data.get("value", {})
            if isinstance(inner, dict):
                values = inner.get("values", [])
        # If this is a search endpoint (has params but no ID in path) with no results
        if not values and params and not re.search(r'/\d+', path) and "value" not in data:
            logger.warning(f"Step {idx}: GET {path} returned empty results")
            response["empty_search"] = True

    return idx, response, response["ok"]


def _should_skip_step(
    idx: int,
    step: dict,
    results: dict[int, dict],
    failed_set: set[int],
    skipped_set: set[int],
    graph: dict[int, set[int]],
) -> str | None:
    """Check if a step should be skipped. Returns reason string or None."""
    deps = graph.get(idx, set())
    failed_deps = deps & (failed_set | skipped_set)
    if failed_deps:
        return f"dependency step(s) {sorted(failed_deps)} failed/skipped"

    skip_if = step.get("skip_if_exists")
    if skip_if and isinstance(skip_if, str):
        resolved = resolve_ref(skip_if, results)
        if resolved is not None and resolved != skip_if:
            return f"skip_if_exists: data already exists"
    return None


# ---------------------------------------------------------------------------
# Main executor
# ---------------------------------------------------------------------------

VALID_METHODS = {"GET", "POST", "PUT", "DELETE"}


def validate_plan(plan: dict) -> list[str]:
    """Validate plan structure before execution."""
    issues = []
    steps = plan.get("steps")
    if not isinstance(steps, list):
        issues.append("Plan has no 'steps' list")
        return issues
    for i, step in enumerate(steps):
        if not isinstance(step, dict):
            issues.append(f"Step {i} is not a dict")
            continue
        method = step.get("method", "").upper()
        if method not in VALID_METHODS:
            issues.append(f"Step {i}: invalid method '{method}'")
        path = step.get("path", "")
        if not path or not isinstance(path, str):
            issues.append(f"Step {i}: missing or invalid path")
        elif not path.startswith("/") and not path.startswith("$"):
            issues.append(f"Step {i}: path should start with / (got '{path[:30]}')")
    return issues


async def execute_plan(
    plan: dict,
    client: TripletexClient,
    start_time: float | None = None,
    prior_results: dict[int, dict] | None = None,
) -> dict:
    """Execute a structured plan with parallel execution of independent steps.

    Returns {success, results, failed, skipped}.
    """
    if start_time is None:
        start_time = time.monotonic()

    DEADLINE = 280

    steps = plan.get("steps", [])
    extracted_values = plan.get("extracted_values", {})
    results: dict[int, dict] = {}
    failed: list[tuple[int, dict]] = []
    skipped: list[tuple[int, str]] = []
    failed_set: set[int] = set()
    skipped_set: set[int] = set()
    intentionally_skipped: set[int] = set()

    if prior_results:
        results.update(prior_results)

    if not steps:
        return {"success": True, "results": results, "failed": [], "skipped": []}

    # Validate
    issues = validate_plan(plan)
    if issues:
        for issue in issues:
            logger.warning(f"Plan validation: {issue}")

    # Build dependency graph and execution layers
    graph = _build_dependency_graph(steps)
    layers = _topological_layers(graph, len(steps))

    parallel_layers = [l for l in layers if len(l) > 1]
    if parallel_layers:
        logger.info(f"Execution: {len(steps)} steps in {len(layers)} layers, "
                    f"{len(parallel_layers)} parallel: {parallel_layers}")
    else:
        logger.info(f"Execution: {len(steps)} steps, fully sequential")

    for layer_idx, layer in enumerate(layers):
        elapsed = time.monotonic() - start_time
        if elapsed > DEADLINE:
            logger.warning(f"Global timeout ({elapsed:.0f}s) at layer {layer_idx}")
            for remaining_layer in layers[layer_idx:]:
                for idx in remaining_layer:
                    skipped.append((idx, "global timeout"))
                    skipped_set.add(idx)
            break

        runnable: list[int] = []
        for idx in layer:
            reason = _should_skip_step(idx, steps[idx], results, failed_set, skipped_set, graph)
            if reason:
                logger.warning(f"Step {idx} skipped: {reason}")
                skipped.append((idx, reason))
                skipped_set.add(idx)
                if reason.startswith("skip_if_exists:"):
                    intentionally_skipped.add(idx)
            else:
                runnable.append(idx)

        if not runnable:
            continue

        if len(runnable) == 1:
            idx = runnable[0]
            step_idx, response, ok = await _execute_step(idx, steps[idx], results, client, extracted_values)
            results[step_idx] = response
            if not ok:
                failed.append((step_idx, response))
                failed_set.add(step_idx)
                logger.error(f"Step {step_idx} failed: {response['status_code']}")
        else:
            logger.info(f"Executing steps {runnable} in parallel")
            tasks = [_execute_step(idx, steps[idx], results, client, extracted_values) for idx in runnable]
            step_results = await asyncio.gather(*tasks, return_exceptions=True)

            for result_item in step_results:
                if isinstance(result_item, Exception):
                    logger.error(f"Step execution raised: {result_item}")
                    continue
                step_idx, response, ok = result_item
                results[step_idx] = response
                if not ok:
                    failed.append((step_idx, response))
                    failed_set.add(step_idx)
                    logger.error(f"Step {step_idx} failed: {response['status_code']}")

    error_skips = len(skipped_set) - len(intentionally_skipped)
    return {
        "success": len(failed) == 0 and error_skips == 0,
        "results": results,
        "failed": failed,
        "skipped": skipped,
    }
