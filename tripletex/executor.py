"""Plan executor with dependency-graph-based parallel execution.

Builds a DAG from $step_N references (and optional explicit depends_on),
then executes independent steps concurrently via asyncio.gather.
"""

import asyncio
import re
import time
import logging
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


def resolve_ref(value: Any, results: dict[int, dict]) -> Any:
    """Resolve $step_N.path.to.field references in a string value."""
    if not isinstance(value, str):
        return value

    pattern = r'\$step_(\d+)\.([\w\[\]\.]+)'

    def _resolve_single(step_idx: int, field_path: str) -> Any:
        step_data = results.get(step_idx, {}).get("data", {})
        parts = field_path.split(".")

        val = step_data.get("value", {})
        if isinstance(val, dict):
            resolved = _deep_get(val, parts)
            if resolved is not None:
                return resolved
            if parts[0] == "value" and len(parts) > 1:
                resolved = _deep_get(val, parts[1:])
                if resolved is not None:
                    return resolved

        resolved = _deep_get(step_data, parts)
        if resolved is not None:
            return resolved
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


def _strip_unresolved_placeholders(obj: Any) -> Any:
    """Remove fields that still contain {{placeholder}} values."""
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
# Step execution
# ---------------------------------------------------------------------------

async def _execute_step(
    idx: int,
    step: dict,
    results: dict[int, dict],
    client: TripletexClient,
) -> tuple[int, dict, bool]:
    """Execute a single step. Returns (index, response, ok)."""
    method = step.get("method", "GET").upper()
    path = resolve_ref(step.get("path", ""), results)

    if "$step_" in str(path):
        logger.error(f"Step {idx}: unresolved reference in path '{path}'")
        response = {"status_code": 0, "ok": False, "data": {"error": f"unresolved path reference: {path}"}}
        return idx, response, False

    body = resolve_refs(step.get("body"), results) if step.get("body") else None
    params = resolve_refs(step.get("params"), results) if step.get("params") else None

    if body:
        body = _strip_unresolved_placeholders(body)
    if params:
        params = _strip_unresolved_placeholders(params)

    logger.info(f"Step {idx}: {method} {path}")

    try:
        response = await client.request(method, path, body=body, params=params)
    except Exception as e:
        logger.error(f"Step {idx} exception: {e}")
        response = {"status_code": 0, "ok": False, "data": {"error": str(e)}}

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
            step_idx, response, ok = await _execute_step(idx, steps[idx], results, client)
            results[step_idx] = response
            if not ok:
                failed.append((step_idx, response))
                failed_set.add(step_idx)
                logger.error(f"Step {step_idx} failed: {response['status_code']}")
        else:
            logger.info(f"Executing steps {runnable} in parallel")
            tasks = [_execute_step(idx, steps[idx], results, client) for idx in runnable]
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
