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


async def execute_plan(plan: dict, client: TripletexClient, start_time: float | None = None) -> dict:
    """Execute a structured plan of API calls.
    Returns {success, results, failed}. No error fixing — that's the LLM's job.

    start_time: monotonic timestamp for global timeout tracking (280s deadline).
    """
    import time
    if start_time is None:
        start_time = time.monotonic()

    DEADLINE = 280  # seconds, leave buffer for response

    steps = plan.get("steps", [])
    results = {}
    failed = []

    for i, step in enumerate(steps):
        # Global timeout check
        if time.monotonic() - start_time > DEADLINE:
            logger.warning(f"Global timeout reached at step {i}, stopping execution")
            break

        method = step["method"].upper()
        path = resolve_ref(step.get("path", ""), results)
        body = resolve_refs(step.get("body"), results) if step.get("body") else None
        params = resolve_refs(step.get("params"), results) if step.get("params") else None

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
