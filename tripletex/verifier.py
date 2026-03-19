"""Post-execution verification: GET created entities and check fields match expectations.

After the executor runs, the verifier queries the Tripletex API to confirm that
entities were actually created/modified with the correct field values. This mirrors
the competition's field-by-field scoring and enables targeted self-repair.
"""

from __future__ import annotations

import logging
from typing import Any

from tripletex_client import TripletexClient

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Verification config per task type
# ---------------------------------------------------------------------------
VERIFY_CONFIG: dict[str, dict[str, Any]] = {

    "create_employee": {
        "entity_path": "/employee",
        "id_from_step": 0,
        "search_params": {"fields": "id,firstName,lastName,email,phoneNumberMobile"},
        "check_fields": {
            "firstName": "extract:firstName",
            "lastName": "extract:lastName",
            "email": "extract:email",
        },
    },

    "update_employee": {
        "entity_path": "/employee",
        "id_from_step": None,
        "id_from_search_step": 0,
        "search_params": {"fields": "id,firstName,lastName,email,phoneNumberMobile"},
        "check_fields": {},
        "dynamic_checks": True,
    },

    "create_customer": {
        "entity_path": "/customer",
        "id_from_step": 0,
        "search_params": {"fields": "id,name,email,organizationNumber,phoneNumber,isCustomer"},
        "check_fields": {
            "name": "extract:name",
            "email": "extract:email",
            "isCustomer": "literal:True",
        },
    },

    "update_customer": {
        "entity_path": "/customer",
        "id_from_step": None,
        "id_from_search_step": 0,
        "search_params": {"fields": "id,name,email,phoneNumber,organizationNumber"},
        "check_fields": {},
        "dynamic_checks": True,
    },

    "create_product": {
        "entity_path": "/product",
        "id_from_step": 0,
        "search_params": {"fields": "id,name,number,priceExcludingVatCurrency"},
        "check_fields": {
            "name": "extract:name",
        },
    },

    "create_invoice": {
        "entity_path": "/invoice",
        "id_from_step": 2,
        "search_params": {"fields": "id,invoiceNumber,invoiceDate,invoiceDueDate,amount,customer"},
        "check_fields": {
            "exists:invoiceNumber": "exists",
        },
    },

    "create_invoice_existing_customer": {
        "entity_path": "/invoice",
        "id_from_step": 2,
        "search_params": {"fields": "id,invoiceNumber,invoiceDate,invoiceDueDate,amount,customer"},
        "check_fields": {
            "exists:invoiceNumber": "exists",
        },
    },

    "create_invoice_with_payment": {
        "entity_path": "/invoice",
        "id_from_step": 2,
        "search_params": {"fields": "id,invoiceNumber,amount,amountOutstanding,customer"},
        "check_fields": {
            "exists:invoiceNumber": "exists",
        },
    },

    "register_payment": {
        "entity_path": "/invoice",
        "id_from_step": None,
        "id_from_extract": "invoice_id",
        "search_params": {"fields": "id,invoiceNumber,amount,amountOutstanding"},
        "check_fields": {
            "exists:id": "exists",
        },
    },

    "create_credit_note": {"skip_verify": True},
    "send_invoice": {"skip_verify": True},

    "create_travel_expense": {
        "entity_path": "/travelExpense",
        "id_from_step": 1,
        "search_params": {"fields": "id,title,travelDetails,employee"},
        "check_fields": {
            "exists:id": "exists",
            "exists:travelDetails": "exists",
        },
    },

    "delete_travel_expense": {"skip_verify": True},
    "deliver_travel_expense": {"skip_verify": True},
    "approve_travel_expense": {"skip_verify": True},

    "create_project": {
        "entity_path": "/project",
        "id_from_step": 1,
        "search_params": {"fields": "id,name,startDate,endDate,isInternal,customer"},
        "check_fields": {
            "name": "extract:project_name",
        },
    },

    "create_internal_project": {
        "entity_path": "/project",
        "id_from_step": 0,
        "search_params": {"fields": "id,name,startDate,endDate,isInternal"},
        "check_fields": {
            "name": "extract:project_name",
            "isInternal": "literal:True",
        },
    },

    "update_project": {
        "entity_path": "/project",
        "id_from_step": None,
        "id_from_search_step": 0,
        "search_params": {"fields": "id,name,startDate,endDate,isInternal,customer"},
        "check_fields": {},
        "dynamic_checks": True,
    },

    "create_department": {
        "entity_path": "/department",
        "id_from_step": 0,
        "search_params": {"fields": "id,name,departmentNumber"},
        "check_fields": {
            "name": "extract:name",
        },
    },

    "create_supplier": {
        "entity_path": "/supplier",
        "id_from_step": 0,
        "search_params": {"fields": "id,name,email,organizationNumber,phoneNumber"},
        "check_fields": {
            "name": "extract:name",
        },
    },

    "create_contact": {
        "entity_path": "/contact",
        "id_from_step": 1,
        "search_params": {"fields": "id,firstName,lastName,email,customer"},
        "check_fields": {
            "firstName": "extract:firstName",
            "lastName": "extract:lastName",
            "email": "extract:email",
        },
    },

    "create_voucher": {
        "entity_path": "/ledger/voucher",
        "id_from_step": 2,
        "search_params": {"fields": "id,date,description,postings"},
        "check_fields": {
            "exists:id": "exists",
            "description": "extract:description",
        },
    },

    "reverse_voucher": {"skip_verify": True},
    "delete_entity": {"skip_verify": True},

    "create_supplier_invoice": {
        "entity_path": "/supplierInvoice",
        "id_from_step": 3,
        "search_params": {"fields": "id,invoiceNumber,amount,supplier"},
        "check_fields": {
            "exists:id": "exists",
        },
    },

    "create_purchase_order": {
        "entity_path": "/purchaseOrder",
        "id_from_step": 1,
        "search_params": {"fields": "id,supplier"},
        "check_fields": {
            "exists:id": "exists",
        },
    },

    "bank_reconciliation": {"skip_verify": True},

    "create_timesheet_entry": {
        "entity_path": "/timesheet/entry",
        "id_from_step": 3,
        "search_params": {"fields": "id,date,hours,employee,project,activity"},
        "check_fields": {
            "exists:id": "exists",
        },
    },

    "create_opening_balance": {"skip_verify": True},
    "create_asset": {"skip_verify": True},
    "create_salary_payment": {"skip_verify": True},

    "create_customer_supplier": {
        "entity_path": "/customer",
        "id_from_step": 0,
        "search_params": {"fields": "id,name,email,isCustomer,isSupplier"},
        "check_fields": {
            "name": "extract:name",
            "isCustomer": "literal:True",
            "isSupplier": "literal:True",
        },
    },

    "create_reminder": {"skip_verify": True},
    "create_employment": {"skip_verify": True},
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_entity_id(execution_results: dict, step_idx: int) -> int | None:
    """Pull the created entity's ID from an execution step's response."""
    if step_idx < 0:
        max_step = max(execution_results.keys()) if execution_results else 0
        step_idx = max_step + step_idx + 1

    step = execution_results.get(step_idx)
    if not step or not step.get("ok"):
        return None

    data = step.get("data", {})
    value = data.get("value")
    if isinstance(value, dict) and "id" in value:
        return value["id"]
    if "id" in data:
        return data["id"]
    return None


def _extract_entity_id_from_search(execution_results: dict, step_idx: int) -> int | None:
    """Pull entity ID from a GET/search step's response (values array)."""
    step = execution_results.get(step_idx)
    if not step or not step.get("ok"):
        return None

    data = step.get("data", {})
    values = data.get("values", [])
    if values and isinstance(values[0], dict) and "id" in values[0]:
        return values[0]["id"]

    value = data.get("value", {})
    if isinstance(value, dict):
        vals = value.get("values", [])
        if vals and isinstance(vals[0], dict) and "id" in vals[0]:
            return vals[0]["id"]
    return None


def _get_nested(obj: dict, path: str) -> Any:
    """Navigate nested dict by dot-separated path."""
    parts = path.split(".")
    current = obj
    for part in parts:
        if isinstance(current, dict):
            current = current.get(part)
        else:
            return None
    return current


def _normalize_for_comparison(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip().lower()


def _values_match(actual: Any, expected: Any) -> bool:
    """Check if actual matches expected, with some tolerance."""
    if actual is None and expected is None:
        return True
    if actual is None or expected is None:
        return False

    if isinstance(expected, bool):
        if isinstance(actual, bool):
            return actual == expected
        return str(actual).lower() == str(expected).lower()

    try:
        a_num = float(actual)
        e_num = float(expected)
        if abs(a_num - e_num) < 0.01:
            return True
    except (ValueError, TypeError):
        pass

    return _normalize_for_comparison(actual) == _normalize_for_comparison(expected)


# ---------------------------------------------------------------------------
# Main verify function
# ---------------------------------------------------------------------------

async def verify_execution(
    task_type: str,
    plan: dict,
    execution_results: dict,
    client: TripletexClient,
    extracted_values: dict,
) -> dict:
    """Verify execution results by querying the Tripletex API.

    Returns:
        {"verified": bool, "skipped": bool, "entity_id": int|None,
         "checks": [...], "passed": int, "failed": int, "missing_fields": [...]}
    """
    config = VERIFY_CONFIG.get(task_type)

    if not config or config.get("skip_verify"):
        logger.info(f"Verification skipped for {task_type}")
        return {"verified": True, "skipped": True, "entity_id": None,
                "checks": [], "passed": 0, "failed": 0, "missing_fields": []}

    entity_path = config.get("entity_path")
    if not entity_path:
        return {"verified": True, "skipped": True, "entity_id": None,
                "checks": [], "passed": 0, "failed": 0, "missing_fields": []}

    # --- Determine entity ID ---
    entity_id: int | None = None

    if config.get("id_from_extract"):
        key = config["id_from_extract"]
        raw_id = extracted_values.get(key)
        if raw_id is not None:
            try:
                entity_id = int(raw_id)
            except (ValueError, TypeError):
                pass
    elif config.get("id_from_step") is not None:
        entity_id = _extract_entity_id(execution_results, config["id_from_step"])
    elif config.get("id_from_search_step") is not None:
        entity_id = _extract_entity_id_from_search(execution_results, config["id_from_search_step"])

    # --- GET the entity ---
    entity: dict | None = None

    if entity_id is not None:
        try:
            search_params = dict(config.get("search_params", {}))
            clean_params = {}
            for k, v in search_params.items():
                if isinstance(v, str) and v.startswith("extract:"):
                    clean_params[k] = extracted_values.get(v[8:], "")
                else:
                    clean_params[k] = v

            resp = await client.get(f"{entity_path}/{entity_id}", params=clean_params)
            if resp["ok"]:
                data = resp.get("data", {})
                entity = data.get("value", data)
            else:
                logger.warning(f"Verify GET {entity_path}/{entity_id} failed: {resp['status_code']}")
        except Exception as e:
            logger.error(f"Verify GET exception: {e}")

    elif not config.get("id_from_extract"):
        # Try searching by params
        try:
            search_params = dict(config.get("search_params", {}))
            resolved_params = {}
            for k, v in search_params.items():
                if isinstance(v, str) and v.startswith("extract:"):
                    resolved_params[k] = extracted_values.get(v[8:], "")
                else:
                    resolved_params[k] = v

            resp = await client.get(entity_path, params=resolved_params)
            if resp["ok"]:
                data = resp.get("data", {})
                values = data.get("values", [])
                if not values:
                    inner = data.get("value", {})
                    if isinstance(inner, dict):
                        values = inner.get("values", [])
                if values:
                    entity = values[0]
                    entity_id = entity.get("id")
        except Exception as e:
            logger.error(f"Verify search exception: {e}")

    # --- If entity not found ---
    if entity is None:
        logger.warning(f"Verify: entity not found for {task_type}")
        return {"verified": False, "skipped": False, "entity_id": entity_id,
                "checks": [{"field": "_entity_found", "expected": True, "actual": False, "passed": False}],
                "passed": 0, "failed": 1, "missing_fields": ["_entity_found"]}

    # --- Run field checks ---
    check_fields = dict(config.get("check_fields", {}))

    if config.get("dynamic_checks") and "fields_to_update" in extracted_values:
        ftu = extracted_values["fields_to_update"]
        if isinstance(ftu, dict):
            for field_name, expected_val in ftu.items():
                check_fields[field_name] = f"literal:{expected_val}"

    checks: list[dict] = []
    passed = 0
    failed_count = 0
    missing_fields: list[str] = []

    checks.append({"field": "_entity_found", "expected": True, "actual": True, "passed": True})
    passed += 1

    for field_spec, value_spec in check_fields.items():
        if field_spec.startswith("exists:"):
            actual_field = field_spec[7:]
            actual_val = _get_nested(entity, actual_field)
            is_present = actual_val is not None and actual_val != "" and actual_val != []
            checks.append({"field": actual_field, "expected": "exists (non-empty)",
                          "actual": actual_val, "passed": is_present})
            if is_present:
                passed += 1
            else:
                failed_count += 1
                missing_fields.append(actual_field)
            continue

        actual_val = _get_nested(entity, field_spec)

        expected_val: Any = None
        if isinstance(value_spec, str):
            if value_spec.startswith("extract:"):
                key = value_spec[8:]
                expected_val = extracted_values.get(key)
                # Fallback: try to get from the POST response data
                if expected_val is None and config.get("id_from_step") is not None:
                    post_step = execution_results.get(config["id_from_step"], {})
                    post_data = post_step.get("data", {}).get("value", {})
                    if isinstance(post_data, dict):
                        expected_val = post_data.get(key)
                # Fallback 2: try common key variations
                if expected_val is None:
                    for alt_key in [key, key.lower(), key.replace("_", ""), key[0].lower() + key[1:]]:
                        if alt_key in extracted_values:
                            expected_val = extracted_values[alt_key]
                            break
            elif value_spec.startswith("literal:"):
                raw = value_spec[8:]
                if raw == "True":
                    expected_val = True
                elif raw == "False":
                    expected_val = False
                else:
                    try:
                        expected_val = int(raw)
                    except ValueError:
                        try:
                            expected_val = float(raw)
                        except ValueError:
                            expected_val = raw
            elif value_spec == "exists":
                is_present = actual_val is not None and actual_val != ""
                checks.append({"field": field_spec, "expected": "exists",
                              "actual": actual_val, "passed": is_present})
                if is_present:
                    passed += 1
                else:
                    failed_count += 1
                    missing_fields.append(field_spec)
                continue
            else:
                expected_val = value_spec
        else:
            expected_val = value_spec

        if expected_val is None:
            continue

        match = _values_match(actual_val, expected_val)
        checks.append({"field": field_spec, "expected": expected_val,
                       "actual": actual_val, "passed": match})
        if match:
            passed += 1
        else:
            failed_count += 1
            missing_fields.append(field_spec)

    verified = failed_count == 0
    logger.info(f"Verify {task_type}: {'PASS' if verified else 'FAIL'} ({passed}/{passed + failed_count} checks)")

    return {"verified": verified, "skipped": False, "entity_id": entity_id,
            "checks": checks, "passed": passed, "failed": failed_count,
            "missing_fields": missing_fields}


def format_verification_for_repair(verify_result: dict) -> str:
    """Format verification results into a string for the self-repair prompt."""
    if verify_result.get("skipped"):
        return "Verification was skipped for this task type."

    lines = []
    if verify_result["verified"]:
        lines.append("All verification checks PASSED.")
    else:
        lines.append(f"VERIFICATION FAILED: {verify_result['failed']} check(s) did not pass.")

    for check in verify_result.get("checks", []):
        status = "PASS" if check["passed"] else "FAIL"
        lines.append(f"  [{status}] {check['field']}: expected={check['expected']!r}, actual={check['actual']!r}")

    if verify_result.get("missing_fields"):
        lines.append(f"  Missing/wrong fields: {verify_result['missing_fields']}")

    return "\n".join(lines)
