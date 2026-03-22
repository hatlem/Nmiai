"""Template engine: turns task_type + extracted_values into a concrete execution plan.

No LLM involvement. The template is law. $step_N references are preserved
for the executor to resolve at runtime.
"""

import copy
import json
import logging
import re
from datetime import date, datetime, timedelta

from templates import TEMPLATES

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# VAT type mapping by account number prefix
# ---------------------------------------------------------------------------

_VAT_TYPE_BY_PREFIX = {
    "1": 0,   # Assets — no VAT
    "2": 0,   # Equity/liabilities — no VAT
    "3": 3,   # Revenue — outgoing VAT 25%
    "4": 1,   # Cost of goods — incoming VAT 25%
    "5": 0,   # Salary costs — no VAT
    "6": 1,   # Operating expenses — incoming VAT 25%
    "7": 1,   # Other operating expenses — incoming VAT 25%
    "8": 0,   # Financial items — no VAT
    "9": 0,   # Tax — no VAT
}


def _infer_vat_type(account_number: str | int | None) -> int:
    """Infer vatType ID from account number prefix."""
    if account_number is None:
        return 0
    s = str(account_number).strip()
    if s and s[0] in _VAT_TYPE_BY_PREFIX:
        return _VAT_TYPE_BY_PREFIX[s[0]]
    return 0


# ---------------------------------------------------------------------------
# Value cleaning
# ---------------------------------------------------------------------------

def _clean_date(val: str | None) -> str | None:
    """Normalize a date value to YYYY-MM-DD. Returns None if unparseable."""
    if val is None:
        return None
    if isinstance(val, (date, datetime)):
        return val.strftime("%Y-%m-%d")
    s = str(val).strip()
    if not s:
        return None
    # Already correct format
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", s):
        return s
    # Common alternatives
    for fmt in ("%d.%m.%Y", "%d/%m/%Y", "%m/%d/%Y", "%Y/%m/%d", "%d-%m-%Y"):
        try:
            return datetime.strptime(s, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return s  # Return as-is if nothing matched


def _clean_amount(val) -> float | int | None:
    """Convert amount to numeric. Returns None if not parseable."""
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return int(val) if float(val) == int(float(val)) else val
    s = str(val).strip().replace(" ", "").replace("\u00a0", "")
    # Handle Norwegian format: 1.234,56 -> 1234.56
    if "," in s and "." in s:
        s = s.replace(".", "").replace(",", ".")
    elif "," in s:
        s = s.replace(",", ".")
    # Strip currency suffixes
    s = re.sub(r"\s*(kr|nok|eur|usd|sek|dkk)\.?\s*$", "", s, flags=re.IGNORECASE)
    s = re.sub(r"^(kr|nok|eur|usd|sek|dkk)\.?\s*", "", s, flags=re.IGNORECASE)
    try:
        f = float(s)
        return int(f) if f == int(f) else f
    except (ValueError, TypeError):
        return None


def _clean_bool(val) -> bool | None:
    """Convert to bool. Returns None if not interpretable."""
    if isinstance(val, bool):
        return val
    if val is None:
        return None
    s = str(val).strip().lower()
    if s in ("true", "1", "yes", "ja", "sant", "si", "oui"):
        return True
    if s in ("false", "0", "no", "nei", "usant", "non"):
        return False
    return None


# Fields that should be treated as dates
_DATE_FIELDS = {
    "date", "orderDate", "deliveryDate", "invoiceDate", "invoiceDueDate",
    "paymentDate", "startDate", "endDate", "dateOfBirth", "departureDate",
    "returnDate", "dateOfAcquisition", "dateFrom", "dateTo", "date_from",
    "date_to", "creditNoteDate", "reverseDate", "paymentDatePlusOne",
}

# Fields that should be numeric
_AMOUNT_FIELDS = {
    "amount", "amountGross", "amountGrossCurrency", "paidAmount",
    "priceExcludingVatCurrency", "priceIncludingVatCurrency",
    "amountCurrencyIncVat", "acquisitionCost", "hours", "count",
    "unitPriceExcludingVatCurrency", "paymentAmount", "cost_amount",
    "percentageOfFullTimeEquivalent", "closingBalance",
    "orderLine_count", "orderLine_unitPriceExcludingVatCurrency",
    "fixedprice", "invoice_percentage", "invoice_amount", "neg_amount",
    "perDiem_dailyRate", "perDiem_days",
}

# Fields that should be boolean
_BOOL_FIELDS = {
    "isCustomer", "isSupplier", "isInternal", "isDayTrip",
    "isForeignTravel", "sendToCustomer", "isPrivateIndividual",
    "isChargeable",
}


def _clean_extracted_values(values: dict) -> dict:
    """Type-coerce extracted values based on field names."""
    cleaned = {}
    for k, v in values.items():
        if v is None:
            continue
        if isinstance(v, str) and v.strip() == "":
            continue
        # Skip placeholder-like values
        if isinstance(v, str) and re.fullmatch(r"\{\{.*?\}\}", v):
            continue

        if k in _DATE_FIELDS:
            v = _clean_date(v)
        elif k in _AMOUNT_FIELDS:
            v = _clean_amount(v)
        elif k in _BOOL_FIELDS:
            v = _clean_bool(v)
        elif k in ("year", "month", "employeeNumber", "departmentNumber", "number"):
            if isinstance(v, str) and v.isdigit():
                v = int(v)

        if v is not None:
            cleaned[k] = v
    return cleaned


# ---------------------------------------------------------------------------
# Placeholder filling
# ---------------------------------------------------------------------------

def _fill_placeholders(obj, values: dict):
    """Replace {{placeholder}} with values from extracted_values.
    Preserves $step_N references as-is."""
    if isinstance(obj, str):
        # Full match: entire string is a placeholder
        m = re.fullmatch(r"\{\{(\w+)\}\}", obj)
        if m:
            key = m.group(1)
            return values.get(key)  # Returns None if missing (will be stripped later)
        # Partial match: interpolate within string
        def _replacer(match):
            key = match.group(1)
            val = values.get(key)
            if val is not None:
                return str(val)
            return match.group(0)  # Keep unresolved
        return re.sub(r"\{\{(\w+)\}\}", _replacer, obj)
    if isinstance(obj, dict):
        return {k: _fill_placeholders(v, values) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_fill_placeholders(item, values) for item in obj]
    return obj


def _strip_none_and_unresolved(obj):
    """Remove None values and unresolved {{placeholder}} strings recursively."""
    if isinstance(obj, dict):
        cleaned = {}
        for k, v in obj.items():
            v = _strip_none_and_unresolved(v)
            if v is None:
                continue
            if isinstance(v, str) and re.search(r"\{\{.*?\}\}", v):
                # If the ENTIRE string is a placeholder, strip the field
                if re.fullmatch(r"\{\{\w+\}\}", v.strip()):
                    logger.debug(f"Stripping unresolved placeholder '{k}': {v}")
                    continue
                # Partial placeholder in a longer string: remove just the placeholder parts
                cleaned_v = re.sub(r"\{\{\w+\}\}", "", v).strip()
                if cleaned_v:
                    cleaned[k] = cleaned_v
                    logger.debug(f"Cleaned partial placeholder in '{k}': {v} -> {cleaned_v}")
                else:
                    logger.debug(f"Stripping fully-unresolved '{k}': {v}")
                continue
            # Don't strip empty dicts/lists that might be intentional
            cleaned[k] = v
        return cleaned if cleaned else None
    if isinstance(obj, list):
        result = []
        for item in obj:
            item = _strip_none_and_unresolved(item)
            if item is not None:
                result.append(item)
        return result if result else None
    return obj


# ---------------------------------------------------------------------------
# Default application
# ---------------------------------------------------------------------------

def _apply_defaults(steps: list[dict], values: dict, task_type: str) -> list[dict]:
    """Apply smart defaults based on task type and existing values."""
    today = date.today().isoformat()

    for step in steps:
        method = step.get("method", "GET").upper()
        path = step.get("path", "")
        body = step.get("body")
        params = step.get("params")

        if method not in ("POST", "PUT") or body is None or not isinstance(body, dict):
            continue

        # Employee defaults
        if "/employee" in path and "/employment" not in path and method == "POST":
            if not body.get("userType"):
                body["userType"] = "STANDARD"

        # Customer defaults
        if path.rstrip("/") == "/customer" and method == "POST":
            if body.get("isCustomer") is None:
                body["isCustomer"] = True

        # Project defaults — startDate is REQUIRED (422 without it)
        if "/project" in path and method == "POST":
            if not body.get("startDate"):
                body["startDate"] = values.get("startDate") or today

        # Order date defaults
        if "/order" in path and "/orderline" not in path.lower() and method == "POST":
            if not body.get("orderDate"):
                body["orderDate"] = values.get("orderDate") or today
            if not body.get("deliveryDate"):
                body["deliveryDate"] = body.get("orderDate") or today

        # Voucher defaults — description is REQUIRED (422 without it)
        if "/ledger/voucher" in path:
            body.setdefault("description", values.get("description") or "Bilag")
            _apply_posting_defaults(body, values)

    # Date derivation in values (for params that reference these)
    if "orderDate" in values and "deliveryDate" not in values:
        values["deliveryDate"] = values["orderDate"]
    if "invoiceDate" in values and "invoiceDueDate" not in values:
        try:
            inv = datetime.strptime(values["invoiceDate"], "%Y-%m-%d")
            values["invoiceDueDate"] = (inv + timedelta(days=14)).strftime("%Y-%m-%d")
        except (ValueError, TypeError):
            pass
    if "invoiceDate" in values and "orderDate" not in values:
        values["orderDate"] = values["invoiceDate"]
        if "deliveryDate" not in values:
            values["deliveryDate"] = values["invoiceDate"]
    if "departureDate" in values and "returnDate" not in values:
        values["returnDate"] = values["departureDate"]

    # Reverse payment: compute paymentDatePlusOne for voucher search date range
    if "paymentDate" in values and "paymentDatePlusOne" not in values:
        try:
            pd = datetime.strptime(values["paymentDate"], "%Y-%m-%d")
            values["paymentDatePlusOne"] = (pd + timedelta(days=1)).strftime("%Y-%m-%d")
        except (ValueError, TypeError):
            pass

    # Reverse payment: default reverseDate to paymentDate if not set
    if "paymentDate" in values and "reverseDate" not in values:
        values["reverseDate"] = values["paymentDate"]

    # Credit note: default creditNoteDate to invoiceDate or today
    if "creditNoteDate" not in values:
        values["creditNoteDate"] = values.get("invoiceDate", today)

    return steps


def _apply_posting_defaults(body: dict, values: dict | None = None):
    """Ensure voucher postings have correct row numbers, amountGrossCurrency, and vatType.
    vatType is inferred from account numbers — LLM-set vatType is overridden because
    Tripletex locks accounts to specific VAT codes and mismatches cause 422."""
    postings = body.get("postings")
    if not isinstance(postings, list):
        return
    if values is None:
        values = {}

    # Map posting index to account number for vatType inference
    posting_account_numbers = []
    debit_acct = str(values.get("debit_account_number", ""))
    credit_acct = str(values.get("credit_account_number", ""))
    if debit_acct and credit_acct:
        posting_account_numbers = [debit_acct, credit_acct]
    # Supplier invoice: expense account + AP account (2400)
    elif values.get("expense_account_number"):
        posting_account_numbers = [str(values["expense_account_number"]), "2400"]
    # Generic: account_number_1, account_number_2
    elif values.get("account_number"):
        posting_account_numbers = [str(values["account_number"])]

    for i, posting in enumerate(postings):
        if not isinstance(posting, dict):
            continue
        posting["row"] = i + 1
        # amountGrossCurrency must equal amountGross and both must be numeric
        if "amountGross" in posting:
            amt = _clean_amount(posting["amountGross"])
            if amt is not None:
                posting["amountGross"] = amt
                posting["amountGrossCurrency"] = amt
        # vatType: infer from account number, OVERRIDE whatever LLM set
        if i < len(posting_account_numbers):
            vat_id = _infer_vat_type(posting_account_numbers[i])
        else:
            vat_id = 0  # Safe default: no VAT
        posting["vatType"] = {"id": vat_id}


# ---------------------------------------------------------------------------
# Dynamic step generation for tier 3 tasks
# ---------------------------------------------------------------------------

def _expand_voucher_steps(steps: list[dict], values: dict) -> list[dict]:
    """Expand voucher/opening-balance templates when postings_data has multiple accounts."""
    postings_data = values.get("postings_data")
    if not postings_data or not isinstance(postings_data, list):
        return steps

    # Collect unique account numbers from postings_data
    account_numbers = []
    seen = set()
    for p in postings_data:
        acct = p.get("account_number") or p.get("accountNumber")
        if acct and str(acct) not in seen:
            account_numbers.append(str(acct))
            seen.add(str(acct))

    if not account_numbers:
        return steps

    # Build N GET steps for accounts + 1 POST voucher
    new_steps = []
    account_step_map = {}  # account_number -> step index

    for i, acct_num in enumerate(account_numbers):
        new_steps.append({
            "method": "GET",
            "path": "/ledger/account",
            "params": {"number": acct_num, "fields": "id,number,name"},
        })
        account_step_map[acct_num] = i

    # Build postings with $step_N references
    postings = []
    for row_idx, p in enumerate(postings_data):
        acct_num = str(p.get("account_number") or p.get("accountNumber", ""))
        step_idx = account_step_map.get(acct_num, 0)
        amount = _clean_amount(p.get("amount") or p.get("amountGross"))
        vat_type = p.get("vatType")
        if vat_type is None:
            vat_type = {"id": _infer_vat_type(acct_num)}
        elif isinstance(vat_type, (int, float)):
            vat_type = {"id": int(vat_type)}

        posting = {
            "row": row_idx + 1,
            "account": {"id": f"$step_{step_idx}.values[0].id"},
            "amountGross": amount,
            "amountGrossCurrency": amount,
            "vatType": vat_type,
        }
        # Optional fields
        if "description" in p:
            posting["description"] = p["description"]
        if "supplier" in p:
            posting["supplier"] = p["supplier"]

        postings.append(posting)

    # Find the POST voucher step in the original template
    voucher_date = values.get("date", date.today().isoformat())
    voucher_desc = values.get("description", "Bilag")

    voucher_step = {
        "method": "POST",
        "path": "/ledger/voucher",
        "body": {
            "date": voucher_date,
            "description": voucher_desc,
            "postings": postings,
        },
    }
    new_steps.append(voucher_step)

    return new_steps


def _expand_opening_balance_steps(steps: list[dict], values: dict) -> list[dict]:
    """Expand opening balance when 'accounts' or 'entries' list is provided."""
    entries = values.get("entries") or values.get("accounts")
    if not entries or not isinstance(entries, list):
        return steps

    account_numbers = []
    seen = set()
    for entry in entries:
        acct = entry.get("account_number") or entry.get("accountNumber") or entry.get("number")
        if acct and str(acct) not in seen:
            account_numbers.append(str(acct))
            seen.add(str(acct))

    if not account_numbers:
        return steps

    # Check if we need a balancing equity account (2050)
    total = 0
    for entry in entries:
        amt = _clean_amount(entry.get("amount") or entry.get("amountGross") or 0)
        if amt is not None:
            total += amt

    needs_balancing = abs(total) > 0.01
    if needs_balancing and "2050" not in seen:
        account_numbers.append("2050")
        seen.add("2050")

    # Build GET steps
    new_steps = []
    account_step_map = {}
    for i, acct_num in enumerate(account_numbers):
        new_steps.append({
            "method": "GET",
            "path": "/ledger/account",
            "params": {"number": acct_num, "fields": "id,number,name"},
        })
        account_step_map[acct_num] = i

    # Build postings
    postings = []
    row = 1
    for entry in entries:
        acct_num = str(entry.get("account_number") or entry.get("accountNumber") or entry.get("number", ""))
        step_idx = account_step_map.get(acct_num, 0)
        amount = _clean_amount(entry.get("amount") or entry.get("amountGross") or 0)
        if amount is None:
            amount = 0

        postings.append({
            "row": row,
            "account": {"id": f"$step_{step_idx}.values[0].id"},
            "amountGross": amount,
            "amountGrossCurrency": amount,
            "vatType": {"id": _infer_vat_type(acct_num)},
        })
        row += 1

    # Add balancing entry if needed
    if needs_balancing:
        bal_step_idx = account_step_map.get("2050", 0)
        postings.append({
            "row": row,
            "account": {"id": f"$step_{bal_step_idx}.values[0].id"},
            "amountGross": -total,
            "amountGrossCurrency": -total,
            "vatType": {"id": 0},
        })

    voucher_date = values.get("date", date.today().isoformat())
    new_steps.append({
        "method": "POST",
        "path": "/ledger/voucher",
        "body": {
            "date": voucher_date,
            "description": "Åpningsbalanse",
            "postings": postings,
        },
    })

    return new_steps


def _expand_dimension_steps(steps: list[dict], values: dict) -> list[dict]:
    """Expand dimension value creation steps when multiple values are provided.

    The template has a single POST /ledger/customDimensionValue step.
    This expands it to N steps (one per dimension value) and adjusts
    subsequent step references accordingly.
    """
    dim_values = values.get("dimension_values")
    if not dim_values or not isinstance(dim_values, list) or len(dim_values) <= 1:
        return steps

    # Find the POST dimension value step (accountingDimensionValue or customDimensionValue)
    dimval_idx = None
    for i, step in enumerate(steps):
        path = step.get("path", "")
        if (path in ("/ledger/customDimensionValue", "/ledger/accountingDimensionValue")
                and step.get("method") == "POST"):
            dimval_idx = i
            break

    if dimval_idx is None:
        return steps

    # Build new steps: replace single dimval step with N steps
    # dimensionIndex references step 0 (POST /ledger/accountingDimensionName) response's number field
    new_steps = steps[:dimval_idx]
    for dv in dim_values:
        name = dv if isinstance(dv, str) else str(dv)
        new_steps.append({
            "method": "POST",
            "path": "/ledger/accountingDimensionValue",
            "body": {
                "displayName": name,
                "dimensionIndex": 1,
            },
        })

    # Number of extra steps inserted (N-1, since we replaced 1 with N)
    extra = len(dim_values) - 1

    # Adjust $step_N references in subsequent steps
    remaining = steps[dimval_idx + 1:]
    for step in remaining:
        step_str = json.dumps(step)
        # Shift step references that are >= dimval_idx+1
        def _shift_ref(m):
            idx = int(m.group(1))
            if idx > dimval_idx:
                return f"$step_{idx + extra}"
            return m.group(0)
        step_str = re.sub(r'\$step_(\d+)', _shift_ref, step_str)
        new_steps.append(json.loads(step_str))

    return new_steps


def _inject_product_steps(steps: list[dict], values: dict) -> list[dict]:
    """Inject POST /product steps before the order when orderLines have productNumber.

    Detects orderLines with productNumber and creates products first, then
    references product IDs in the orderLines. Also handles per-line vatType
    by mapping VAT percentage to Tripletex vatType IDs.
    """
    order_lines = values.get("orderLines")
    if not order_lines or not isinstance(order_lines, list):
        return steps

    # Check if any orderLine has a productNumber
    lines_with_product = [
        (i, ol) for i, ol in enumerate(order_lines)
        if isinstance(ol, dict) and ol.get("productNumber")
    ]
    if not lines_with_product:
        return steps

    # Find the POST /order step
    order_step_idx = None
    for i, step in enumerate(steps):
        if step.get("path", "").rstrip("/") == "/order" and step.get("method") == "POST":
            order_step_idx = i
            break

    if order_step_idx is None:
        return steps

    # Build product creation steps to insert BEFORE the order step
    product_steps = []
    product_step_base = order_step_idx  # product steps will be inserted starting at this index

    for line_idx, ol in lines_with_product:
        prod_num = ol["productNumber"]
        prod_name = ol.get("description", f"Product {prod_num}")
        prod_price = ol.get("unitPriceExcludingVatCurrency", 0)

        product_steps.append({
            "method": "POST",
            "path": "/product",
            "body": {
                "name": prod_name,
                "number": str(prod_num),
                "priceExcludingVatCurrency": prod_price,
            },
        })

    num_product_steps = len(product_steps)
    if num_product_steps == 0:
        return steps

    # Insert product steps before the order step
    new_steps = steps[:order_step_idx] + product_steps + steps[order_step_idx:]

    # Now shift all $step_N references in steps AFTER the inserted product steps
    # Steps from order_step_idx onward have shifted by num_product_steps
    for i in range(order_step_idx + num_product_steps, len(new_steps)):
        step_str = json.dumps(new_steps[i])

        def _shift_ref(m):
            idx = int(m.group(1))
            if idx >= order_step_idx:
                return f"$step_{idx + num_product_steps}"
            return m.group(0)

        step_str = re.sub(r'\$step_(\d+)', _shift_ref, step_str)
        new_steps[i] = json.loads(step_str)

    # Update orderLines in values dict to reference product IDs.
    # At this point the order step body still has "{{orderLines}}" as a placeholder,
    # so we must update values["orderLines"] directly — _fill_placeholders will
    # substitute it into the step body later.
    ol_value = values.get("orderLines")
    if isinstance(ol_value, list):
        for prod_idx, (line_idx, ol) in enumerate(lines_with_product):
            product_step_global = order_step_idx + prod_idx
            if line_idx < len(ol_value) and isinstance(ol_value[line_idx], dict):
                ol_value[line_idx]["product"] = {"id": f"$step_{product_step_global}.id"}
                # Remove productNumber from orderLine (not an API field)
                ol_value[line_idx].pop("productNumber", None)

        # Also strip productNumber from any remaining lines (safety)
        for ol in ol_value:
            if isinstance(ol, dict):
                ol.pop("productNumber", None)
        values["orderLines"] = ol_value

    return new_steps


def _expand_purchase_order_lines(steps: list[dict], values: dict) -> list[dict]:
    """Expand purchase order template when multiple orderLines are provided."""
    order_lines = values.get("orderLines")
    if not order_lines or not isinstance(order_lines, list) or len(order_lines) <= 1:
        return steps

    # Find the POST /purchaseOrder/orderline step
    orderline_idx = None
    for i, step in enumerate(steps):
        if step.get("path", "").endswith("/purchaseOrder/orderline"):
            orderline_idx = i
            break

    if orderline_idx is None:
        return steps

    # The orderline step references $step_2.id (the purchase order)
    # We need to find which step creates the purchase order
    po_step_ref = None
    original_step = steps[orderline_idx]
    body = original_step.get("body", {})
    po_ref = body.get("purchaseOrder", {}).get("id", "")
    if isinstance(po_ref, str) and po_ref.startswith("$step_"):
        po_step_ref = po_ref

    # Replace the single orderline step with N steps
    new_steps = steps[:orderline_idx]
    for line in order_lines:
        line_body = {
            "purchaseOrder": {"id": po_step_ref} if po_step_ref else body.get("purchaseOrder"),
            "description": line.get("description", ""),
            "count": _clean_amount(line.get("count", 1)),
            "unitPriceExcludingVatCurrency": _clean_amount(line.get("unitPriceExcludingVatCurrency", 0)),
        }
        new_steps.append({
            "method": "POST",
            "path": "/purchaseOrder/orderline",
            "body": line_body,
        })

    # Append any steps after the original orderline step
    new_steps.extend(steps[orderline_idx + 1:])
    return new_steps


def _expand_travel_costs(steps: list[dict], values: dict) -> list[dict]:
    """Expand travel expense template when costs are provided."""
    costs = values.get("costs")
    if not costs or not isinstance(costs, list) or len(costs) == 0:
        return steps

    # Find the POST /travelExpense/cost step
    cost_idx = None
    for i, step in enumerate(steps):
        if step.get("path", "") == "/travelExpense/cost" and step.get("method") == "POST":
            cost_idx = i
            break

    if cost_idx is None:
        return steps

    original_step = steps[cost_idx]
    original_body = original_step.get("body", {})

    new_steps = steps[:cost_idx]
    for cost_item in costs:
        cost_body = copy.deepcopy(original_body)
        # Override with specific cost data
        if "amount" in cost_item or "amountCurrencyIncVat" in cost_item:
            cost_body["amountCurrencyIncVat"] = _clean_amount(
                cost_item.get("amountCurrencyIncVat") or cost_item.get("amount")
            )
        if "date" in cost_item:
            cost_body["date"] = _clean_date(cost_item["date"])
        if "comments" in cost_item or "description" in cost_item:
            cost_body["comments"] = cost_item.get("comments") or cost_item.get("description")
        if "costCategory" in cost_item:
            cost_body["costCategory"] = cost_item["costCategory"]

        new_steps.append({
            "method": "POST",
            "path": "/travelExpense/cost",
            "body": cost_body,
        })

    new_steps.extend(steps[cost_idx + 1:])
    return new_steps


# ---------------------------------------------------------------------------
# Conditional steps
# ---------------------------------------------------------------------------

def _apply_conditional_steps(steps: list[dict], values: dict, template: dict) -> list[dict]:
    """Add conditional steps based on extracted values."""
    conditional = template.get("conditional_steps", {})
    if not conditional:
        return steps

    for trigger_key, step_or_steps in conditional.items():
        # trigger_key is like "if_role" or "if_cost_amount"
        field = trigger_key.removeprefix("if_")
        # Also trigger cost steps if we have a costs array
        triggered = field in values and values[field]
        if not triggered and field == "cost_amount" and isinstance(values.get("costs"), list) and values["costs"]:
            triggered = True
        if triggered:
            if isinstance(step_or_steps, list):
                # Array of steps (e.g. travel expense costs)
                steps.extend(copy.deepcopy(step_or_steps))
            else:
                # Single step (e.g. role entitlement)
                steps.append(copy.deepcopy(step_or_steps))

    return steps


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def build_concrete_plan(task_type: str, extracted_values: dict) -> dict:
    """Build a ready-to-execute plan from a template and extracted values.

    Args:
        task_type: Key into TEMPLATES (e.g. "create_employee", "create_voucher")
        extracted_values: Values extracted from the task prompt by the LLM

    Returns:
        {
            "task_type": str,
            "steps": [{"method", "path", "body", "params"}, ...],
            "extracted_values": dict (cleaned),
        }
    """
    template = TEMPLATES.get(task_type)
    if template is None:
        logger.warning(f"No template for task_type '{task_type}', falling back to 'unknown'")
        template = TEMPLATES.get("unknown", {"steps": []})

    # Clean and type-coerce extracted values
    values = _clean_extracted_values(extracted_values)

    # Derive missing fields from available data
    # Timesheet: ensure date >= project startDate (API rejects earlier dates)
    if task_type == "create_timesheet_entry":
        from datetime import date as _date
        ts_date = values.get("date", _date.today().isoformat())
        start_date = values.get("startDate") or values.get("project_startDate")
        if start_date and ts_date < start_date:
            values["date"] = start_date
            logger.info(f"Timesheet: adjusted date to project startDate {start_date}")
        elif not values.get("date"):
            values["date"] = _date.today().isoformat()

    # Fixed-price project invoice: compute invoice_amount from fixedprice * percentage
    if task_type == "fixed_price_project_invoice":
        from datetime import date as _date, timedelta
        fp = values.get("fixedprice", 0)
        pct = values.get("invoice_percentage", 100)
        if isinstance(fp, (int, float)) and isinstance(pct, (int, float)):
            values["invoice_amount"] = fp * pct / 100
        values.setdefault("today", _date.today().isoformat())
        if "invoiceDueDate" not in values:
            values["invoiceDueDate"] = (_date.today() + timedelta(days=14)).isoformat()

    # Dimensions voucher: compute neg_amount and expand dimension steps
    if task_type == "create_dimensions_voucher":
        amt = values.get("amount")
        if isinstance(amt, (int, float)):
            values["neg_amount"] = -amt
        dim_values = values.get("dimension_values")
        if isinstance(dim_values, list) and dim_values:
            values["first_dimension_value"] = dim_values[0] if isinstance(dim_values[0], str) else str(dim_values[0])

    # Voucher: credit_amount defaults to debit_amount (double-entry bookkeeping)
    if "debit_amount" in values and "credit_amount" not in values:
        values["credit_amount"] = values["debit_amount"]

    # Travel expense: defaults for REQUIRED travelDetails fields
    if task_type == "create_travel_expense":
        if "departureDate" not in values:
            values["departureDate"] = today
        if "returnDate" not in values:
            values["returnDate"] = values.get("departureDate", today)
        if "isDayTrip" not in values:
            values["isDayTrip"] = values.get("departureDate") == values.get("returnDate")
        if "isForeignTravel" not in values:
            values["isForeignTravel"] = False
        if "departureFrom" not in values:
            values["departureFrom"] = "Oslo"
        if "title" not in values:
            values["title"] = values.get("purpose", values.get("destination", "Reiseregning"))

        # Inject per diem into costs array if perDiem fields are present
        daily_rate = _clean_amount(values.get("perDiem_dailyRate"))
        days = _clean_amount(values.get("perDiem_days"))
        if daily_rate and days:
            per_diem_amount = daily_rate * days
            costs = values.get("costs")
            if not isinstance(costs, list):
                costs = []
            costs.append({
                "description": "Diett",
                "amount": per_diem_amount,
                "amountCurrencyIncVat": per_diem_amount,
                "comments": f"Diett {int(days)} dager x {int(daily_rate)} kr",
            })
            values["costs"] = costs

    if "cost_amount" not in values:
        # Try to get from costs list or amount field
        costs = values.get("costs", [])
        if isinstance(costs, list) and costs:
            first = costs[0] if isinstance(costs[0], dict) else {"amount": costs[0]}
            values["cost_amount"] = first.get("amountCurrencyIncVat") or first.get("amount")
        elif "amount" in values:
            values["cost_amount"] = values["amount"]

    # Deep copy steps to avoid mutating the template
    steps = copy.deepcopy(template.get("steps", []))

    # Inject product creation steps for invoices with product numbers
    if task_type in ("create_invoice", "create_invoice_existing_customer",
                     "create_invoice_with_payment", "create_full_credit_note",
                     "create_invoice_and_send", "reverse_payment"):
        order_lines = values.get("orderLines")
        if isinstance(order_lines, list) and any(
            isinstance(ol, dict) and ol.get("productNumber") for ol in order_lines
        ):
            steps = _inject_product_steps(steps, values)

    # Handle dynamic expansion for complex task types
    if task_type in ("create_voucher",) and "postings_data" in values:
        steps = _expand_voucher_steps(steps, values)

    elif task_type in ("create_opening_balance",) and ("entries" in values or "accounts" in values):
        steps = _expand_opening_balance_steps(steps, values)

    elif task_type == "create_purchase_order" and isinstance(values.get("orderLines"), list):
        steps = _expand_purchase_order_lines(steps, values)

    elif task_type == "create_dimensions_voucher" and isinstance(values.get("dimension_values"), list):
        steps = _expand_dimension_steps(steps, values)

    # reverse_voucher: if voucher_id is known, skip the search step and reverse directly
    if task_type == "reverse_voucher" and values.get("voucher_id"):
        vid = values["voucher_id"]
        steps = [
            {
                "method": "PUT",
                "path": f"/ledger/voucher/{vid}/:reverse",
                "params": {"date": "{{date}}"},
            },
        ]
    elif task_type == "reverse_voucher":
        # Ensure dateFrom/dateTo have defaults for the voucher search
        from datetime import date as _date, timedelta
        if "dateFrom" not in values:
            values["dateFrom"] = (_date.today() - timedelta(days=30)).isoformat()
        if "dateTo" not in values:
            values["dateTo"] = (_date.today() + timedelta(days=1)).isoformat()

    # Apply conditional steps BEFORE travel cost expansion
    # (travel cost step is in conditional_steps, needs to be in steps first)
    steps = _apply_conditional_steps(steps, values, template)

    # Expand travel costs AFTER conditional steps have been added
    if task_type == "create_travel_expense" and isinstance(values.get("costs"), list):
        steps = _expand_travel_costs(steps, values)

    # Fill {{placeholders}} with extracted values
    steps = _fill_placeholders(steps, values)

    # Inject id + version into PUT bodies for update tasks
    # The template uses "body": "{{fields_to_update}}" which only contains changed fields,
    # but Tripletex PUT requires id and version from the preceding GET step.
    if task_type.startswith("update_"):
        for i, step in enumerate(steps):
            if step.get("method", "").upper() == "PUT" and isinstance(step.get("body"), dict):
                body = step["body"]
                # Find the GET step this PUT depends on (usually step 0)
                get_step_idx = None
                path = step.get("path", "")
                for ref_match in re.finditer(r'\$step_(\d+)', str(path)):
                    get_step_idx = int(ref_match.group(1))
                    break
                if get_step_idx is None:
                    get_step_idx = 0  # Default: first step is usually the GET

                # Inject id and version references if not already present
                if "id" not in body:
                    body["id"] = f"$step_{get_step_idx}.values[0].id"
                if "version" not in body:
                    body["version"] = f"$step_{get_step_idx}.values[0].version"

    # Apply smart defaults (dates, booleans, etc.)
    steps = _apply_defaults(steps, values, task_type)

    # Strip None values and unresolved placeholders from bodies and params
    for step in steps:
        if "body" in step and isinstance(step["body"], dict):
            step["body"] = _strip_none_and_unresolved(step["body"])
        if "params" in step and isinstance(step["params"], dict):
            step["params"] = _strip_none_and_unresolved(step["params"])
        # Remove note fields — they're for documentation only
        step.pop("note", None)

    # Final cleanup: remove steps with no body AND no params for POST/PUT
    # (but keep GETs and DELETEs as they might just need the path)
    cleaned_steps = []
    for step in steps:
        method = step.get("method", "GET").upper()
        if method in ("POST", "PUT") and step.get("body") is None and step.get("params") is None:
            path = step.get("path", "")
            # Keep action endpoints like /:invoice, /:payment, /:deliver
            if "/:" in str(path):
                cleaned_steps.append(step)
            else:
                logger.warning(f"Dropping empty {method} {path} (no body or params)")
        else:
            cleaned_steps.append(step)

    return {
        "task_type": task_type,
        "steps": cleaned_steps,
        "extracted_values": values,
    }
