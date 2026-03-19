import re
import time
import logging
from tripletex_client import TripletexClient

logger = logging.getLogger(__name__)

VALID_METHODS = {"GET", "POST", "PUT", "DELETE"}


def validate_plan(plan: dict) -> list[str]:
    """Validate plan structure before execution. Returns list of issues (empty = OK)."""
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
        # POST/PUT should have body or params
        if method in ("POST", "PUT") and not step.get("body") and not step.get("params"):
            # Some action endpoints (like :deliver, :approve) use only params, that's OK
            if ":" not in path:
                issues.append(f"Step {i}: {method} {path} has no body or params")
    return issues


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

    # Otherwise do string substitution
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


def _strip_unresolved_placeholders(obj):
    """Remove fields that still contain {{placeholder}} values — the LLM didn't fill them.
    Better to omit than send literal '{{foo}}' to the API."""
    if isinstance(obj, dict):
        cleaned = {}
        for k, v in obj.items():
            v = _strip_unresolved_placeholders(v)
            if isinstance(v, str) and re.search(r'\{\{.*?\}\}', v):
                logger.warning(f"Stripping unresolved placeholder field '{k}': {v}")
                continue
            cleaned[k] = v
        return cleaned
    if isinstance(obj, list):
        return [_strip_unresolved_placeholders(item) for item in obj]
    return obj


async def execute_plan(
    plan: dict,
    client: TripletexClient,
    start_time: float | None = None,
    prior_results: dict | None = None,
) -> dict:
    """Execute a structured plan of API calls.
    Returns {success, results, failed}.

    start_time: monotonic timestamp for global timeout tracking (280s deadline).
    prior_results: results from a previous run — allows $step_N refs to resolved IDs from succeeded steps.
    """
    if start_time is None:
        start_time = time.monotonic()

    DEADLINE = 280  # seconds, leave buffer for response

    steps = plan.get("steps", [])

    # Validate plan structure
    issues = validate_plan(plan)
    if issues:
        for issue in issues:
            logger.warning(f"Plan validation: {issue}")

    # Merge prior results so $step_N references from previous run still resolve
    results = dict(prior_results) if prior_results else {}
    failed = []

    for i, step in enumerate(steps):
        # Global timeout check
        if time.monotonic() - start_time > DEADLINE:
            logger.warning(f"Global timeout reached at step {i}, stopping execution")
            break

        method = step.get("method", "GET").upper()
        if method not in VALID_METHODS:
            logger.error(f"Step {i}: invalid method '{method}', skipping")
            results[i] = {"status_code": 0, "ok": False, "data": {"error": f"invalid method: {method}"}}
            failed.append((i, results[i]))
            continue

        path = resolve_ref(step.get("path", ""), results)

        # Check for unresolved path references
        if "$step_" in str(path):
            logger.error(f"Step {i}: unresolved reference in path '{path}'")
            results[i] = {"status_code": 0, "ok": False, "data": {"error": f"unresolved path reference: {path}"}}
            failed.append((i, results[i]))
            continue

        body = resolve_refs(step.get("body"), results) if step.get("body") else None
        params = resolve_refs(step.get("params"), results) if step.get("params") else None

        # Strip any unresolved {{placeholder}} fields
        if body:
            body = _strip_unresolved_placeholders(body)
        if params:
            params = _strip_unresolved_placeholders(params)

        logger.info(f"Step {i}: {method} {path}")

        try:
            response = await client.request(method, path, body=body, params=params)
        except Exception as e:
            logger.error(f"Step {i} exception: {e}")
            response = {"status_code": 0, "ok": False, "data": {"error": str(e)}}

        results[i] = response

        if not response["ok"]:
            failed.append((i, response))
            logger.error(f"Step {i} failed: {response['status_code']}")

    return {
        "success": len(failed) == 0,
        "results": results,
        "failed": failed,
    }
