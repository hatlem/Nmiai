"""
Self-learning module for the Tripletex agent.

Auto-template compilation: when the tool agent succeeds, record the exact API
call sequence. Next time a similar task arrives, replay it deterministically
instead of burning LLM calls.

Three layers:
1. Error Memory — remembers past errors for debugging
2. Compiled Templates — deterministic replay of proven API sequences
3. Adaptive Routing — tracks template vs tool_agent success rates
"""

import json
import logging
import re
import threading
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Common words to strip when extracting keywords
_STOP_WORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "must",
    "i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us",
    "my", "your", "his", "its", "our", "their",
    "this", "that", "these", "those", "what", "which", "who", "whom",
    "and", "but", "or", "nor", "not", "no", "so", "if", "then", "than",
    "of", "in", "on", "at", "to", "for", "with", "by", "from", "up",
    "about", "into", "through", "during", "before", "after", "above",
    "below", "between", "out", "off", "over", "under", "again", "further",
    "once", "here", "there", "when", "where", "why", "how", "all", "each",
    "every", "both", "few", "more", "most", "other", "some", "such",
    "only", "own", "same", "too", "very", "just", "because", "as", "until",
    "while", "also", "det", "en", "et", "er", "og", "i", "på", "for",
    "med", "til", "fra", "som", "den", "de", "å", "av", "har", "var",
    "kan", "vil", "skal", "må", "ikke", "om", "men", "eller", "da",
    "ved", "seg", "sin", "sitt", "sine", "ble", "bli", "blir",
})

MAX_ERROR_MEMORY = 200
MAX_COMPILED_TEMPLATES = 100

# ---------------------------------------------------------------------------
# In-memory stores
# ---------------------------------------------------------------------------

ERROR_MEMORY: list[dict] = []
COMPILED_TEMPLATES: dict[str, dict] = {}
TASK_STATS: dict[str, dict] = {}

_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_keywords(prompt: str) -> frozenset[str]:
    """Extract meaningful keywords from a prompt string."""
    tokens = re.findall(r"[a-zæøåäö0-9]+", prompt.lower())
    keywords = {t for t in tokens if t not in _STOP_WORDS and len(t) > 1}
    return frozenset(keywords)


def _match_score(keywords1: frozenset[str], keywords2: frozenset[str]) -> float:
    """Jaccard similarity between two keyword sets."""
    if not keywords1 or not keywords2:
        return 0.0
    intersection = keywords1 & keywords2
    union = keywords1 | keywords2
    return len(intersection) / len(union)


def _task_signature(prompt: str) -> str:
    """Derive a task signature from prompt keywords (sorted, joined with +)."""
    kw = _extract_keywords(prompt)
    return "+".join(sorted(kw))


def _value_in_prompt(value: Any, prompt_lower: str) -> bool:
    """Check if a value appears in the prompt text."""
    if value is None:
        return False
    s = str(value).strip().lower()
    if len(s) < 2:
        return False
    return s in prompt_lower


# Constants that should be kept as-is (not turned into placeholders)
_CONSTANT_VALUES = frozenset({
    True, False, "true", "false",
    "STANDARD", "NORMAL", "DEFAULT", "NOK", "PERCENTAGE", "FIXED",
    0, 1, -1,
})


def _is_constant(value: Any) -> bool:
    """Check if a value is a constant that shouldn't become a placeholder."""
    if value in _CONSTANT_VALUES:
        return True
    if isinstance(value, str) and value.isupper() and len(value) <= 20:
        return True  # Enum-like strings
    return False


# ---------------------------------------------------------------------------
# Layer 1: Error Memory
# ---------------------------------------------------------------------------

def record_error(endpoint: str, error_data: str | dict, prompt: str) -> None:
    """Parse an error response and store a lesson learned."""
    if isinstance(error_data, dict):
        error_msg = str(error_data.get("message", error_data.get("error", str(error_data))))
    else:
        error_msg = str(error_data)

    lesson = f"{endpoint}: {error_msg[:120]}"
    keywords = list(sorted(_extract_keywords(prompt)))

    with _lock:
        for entry in ERROR_MEMORY:
            if entry["endpoint"] == endpoint and entry["error"] == error_msg:
                entry["count"] += 1
                return

        ERROR_MEMORY.append({
            "endpoint": endpoint,
            "error": error_msg,
            "lesson": lesson,
            "task_keywords": keywords,
            "count": 1,
        })
        logger.info("Recorded error lesson: %s", lesson)

        while len(ERROR_MEMORY) > MAX_ERROR_MEMORY:
            ERROR_MEMORY.pop(0)


# ---------------------------------------------------------------------------
# Layer 2: Compiled Templates
# ---------------------------------------------------------------------------

def compile_template(prompt: str, api_log: list[dict]) -> None:
    """Convert a successful API call sequence into a replayable template.

    Args:
        prompt: The original task prompt.
        api_log: List of dicts with method, path, body (dict), status, response (dict).
                 Only successful calls (2xx) are included in the template.
    """
    sig = _task_signature(prompt)
    if not sig:
        return

    # Filter to successful mutating calls (POST/PUT/DELETE with 2xx)
    successful_calls = [
        c for c in api_log
        if c.get("ok", False) or (200 <= c.get("status", 0) < 300)
    ]
    if not successful_calls:
        return

    prompt_lower = prompt.lower()
    steps: list[dict] = []
    extract_fields: set[str] = set()
    # Track response IDs from previous steps for cross-reference detection
    prev_response_ids: dict[int, list[tuple[str, Any]]] = {}  # step_idx -> [(path, value)]

    for i, call in enumerate(successful_calls):
        method = call.get("method", "GET")
        path = call.get("path", "")
        body = call.get("body")
        response = call.get("response", call.get("data", {}))

        # Extract IDs from response for cross-referencing
        if isinstance(response, dict):
            _extract_ids(response, [], prev_response_ids, i)

        # Skip pure GET calls (lookups) — we only template mutations
        if method == "GET":
            # But still record response IDs for reference
            continue

        step: dict = {
            "method": method,
            "path": path,
        }

        if isinstance(body, dict):
            body_template, step_fields = _templatize_body(
                body, prompt_lower, prev_response_ids, i
            )
            step["body_template"] = body_template
            extract_fields.update(step_fields)

        params = call.get("params")
        if isinstance(params, dict):
            param_template, param_fields = _templatize_body(
                params, prompt_lower, prev_response_ids, i
            )
            step["params_template"] = param_template
            extract_fields.update(param_fields)

        steps.append(step)

    if not steps:
        return

    with _lock:
        existing = COMPILED_TEMPLATES.get(sig)
        if existing:
            existing["successes"] += 1
            existing["last_prompt"] = prompt[:300]
            logger.info("Updated compiled template: %s (successes=%d)", sig, existing["successes"])
        else:
            COMPILED_TEMPLATES[sig] = {
                "steps": steps,
                "extract_fields": sorted(extract_fields),
                "successes": 1,
                "last_prompt": prompt[:300],
            }
            logger.info("Compiled new template: %s (%d steps, %d fields)", sig, len(steps), len(extract_fields))

            # Evict oldest if over limit
            while len(COMPILED_TEMPLATES) > MAX_COMPILED_TEMPLATES:
                oldest_key = next(iter(COMPILED_TEMPLATES))
                del COMPILED_TEMPLATES[oldest_key]


def _extract_ids(obj: Any, path: list[str], store: dict[int, list[tuple[str, Any]]], step_idx: int):
    """Recursively extract id-like fields from a response dict."""
    if isinstance(obj, dict):
        for key, val in obj.items():
            current_path = path + [key]
            if key == "id" and isinstance(val, (int, float)):
                dotpath = ".".join(current_path)
                store.setdefault(step_idx, []).append((dotpath, val))
            elif isinstance(val, dict):
                _extract_ids(val, current_path, store, step_idx)
            elif key == "value" and isinstance(val, dict):
                _extract_ids(val, current_path, store, step_idx)


def _templatize_body(
    body: dict,
    prompt_lower: str,
    prev_response_ids: dict[int, list[tuple[str, Any]]],
    current_step: int,
) -> tuple[dict, set[str]]:
    """Convert a concrete body dict into a template with placeholders.

    Returns (body_template, set_of_extract_field_names).
    """
    template = {}
    fields: set[str] = set()

    for key, value in body.items():
        if isinstance(value, dict):
            # Nested object — recurse
            sub_template, sub_fields = _templatize_body(
                value, prompt_lower, prev_response_ids, current_step
            )
            template[key] = sub_template
            fields.update(sub_fields)
        elif isinstance(value, list):
            # Lists are kept as-is (too complex to templatize reliably)
            template[key] = value
        elif _is_constant(value):
            template[key] = value
        elif _is_step_reference(value, prev_response_ids, current_step):
            ref = _find_step_reference(value, prev_response_ids, current_step)
            template[key] = ref  # e.g. "$step_0.value.id"
        elif _value_in_prompt(value, prompt_lower):
            placeholder = f"{{{{{key}}}}}"
            template[key] = placeholder
            fields.add(key)
        else:
            # Value not in prompt and not a constant — keep as-is
            # (could be a default value the API needs)
            template[key] = value

    return template, fields


def _is_step_reference(value: Any, prev_ids: dict[int, list[tuple[str, Any]]], current_step: int) -> bool:
    """Check if a value matches an ID from a previous step's response."""
    if not isinstance(value, (int, float)):
        return False
    for step_idx in range(current_step):
        for _, id_val in prev_ids.get(step_idx, []):
            if value == id_val:
                return True
    return False


def _find_step_reference(value: Any, prev_ids: dict[int, list[tuple[str, Any]]], current_step: int) -> str:
    """Find which previous step produced this ID value."""
    for step_idx in range(current_step):
        for dotpath, id_val in prev_ids.get(step_idx, []):
            if value == id_val:
                return f"$step_{step_idx}.{dotpath}"
    return str(value)


def get_compiled_template(prompt: str) -> dict | None:
    """Find the best matching compiled template for this prompt.

    Returns the template dict if Jaccard similarity > 0.5, else None.
    """
    prompt_kw = _extract_keywords(prompt)
    if not prompt_kw:
        return None

    with _lock:
        best_score = 0.0
        best_template: dict | None = None
        for sig, tmpl in COMPILED_TEMPLATES.items():
            sig_kw = frozenset(sig.split("+"))
            score = _match_score(prompt_kw, sig_kw)
            if score > best_score:
                best_score = score
                best_template = tmpl

    if best_score < 0.5 or best_template is None:
        return None

    logger.info("Found compiled template (score=%.2f, %d steps)", best_score, len(best_template["steps"]))
    return best_template


async def execute_compiled_template(
    template: dict,
    prompt: str,
    client,
    files: list | None = None,
) -> dict:
    """Execute a compiled template by extracting values and replaying steps.

    Uses Gemini Flash-Lite for fast value extraction from the prompt.
    Returns {"success": True/False, "results": [...]}.
    """
    import vertexai
    from vertexai.generative_models import GenerativeModel

    extract_fields = template.get("extract_fields", [])
    steps = template.get("steps", [])

    if not steps:
        return {"success": False, "error": "empty template"}

    # --- Extract field values from prompt using Gemini Flash-Lite ---
    extracted_values: dict[str, Any] = {}
    if extract_fields:
        extraction_prompt = (
            f"Extract these fields from the text below. Return ONLY valid JSON, no markdown.\n"
            f"Fields: {json.dumps(extract_fields)}\n"
            f"Text: {prompt}\n"
        )
        if files:
            for f in files[:2]:  # Limit to 2 files
                content = f.get("content", "")
                if content:
                    extraction_prompt += f"\nAttached file ({f.get('name', 'file')}):\n{content[:2000]}\n"

        try:
            vertexai.init(project="ainm26osl-710", location="global")
            model = GenerativeModel("gemini-3.1-flash-lite-preview")
            response = model.generate_content(extraction_prompt)
            text = response.text.strip()
            # Strip markdown code fences if present
            if text.startswith("```"):
                text = re.sub(r"^```(?:json)?\s*", "", text)
                text = re.sub(r"\s*```$", "", text)
            extracted_values = json.loads(text)
            logger.info("Extracted %d values from prompt: %s", len(extracted_values), list(extracted_values.keys()))
        except Exception as e:
            logger.error("Flash-Lite extraction failed: %s", e)
            return {"success": False, "error": f"extraction failed: {e}"}

    # --- Execute steps sequentially ---
    step_results: list[dict] = []
    results_by_index: dict[int, dict] = {}  # For $step_N reference resolution

    for i, step in enumerate(steps):
        method = step["method"]
        path = step["path"]

        # Resolve body template
        body = None
        if "body_template" in step:
            body = _resolve_template(step["body_template"], extracted_values, results_by_index)

        params = None
        if "params_template" in step:
            params = _resolve_template(step["params_template"], extracted_values, results_by_index)

        # Execute the API call
        try:
            if method == "POST":
                resp = await client.post(path, body=body, params=params)
            elif method == "PUT":
                resp = await client.put(path, body=body, params=params)
            elif method == "DELETE":
                resp = await client.delete(path, params=params)
            else:
                resp = await client.get(path, params=params)

            ok = resp.get("ok", False)
            data = resp.get("data", {})

            step_results.append({
                "step": i,
                "method": method,
                "path": path,
                "ok": ok,
                "status": resp.get("status_code", 0),
            })

            # Store response for $step_N references
            # Unwrap nested value if present
            if isinstance(data, dict) and "value" in data:
                results_by_index[i] = data["value"]
            else:
                results_by_index[i] = data

            if not ok:
                logger.warning(
                    "Compiled template step %d failed: %s %s -> %d",
                    i, method, path, resp.get("status_code", 0),
                )
                return {"success": False, "results": step_results, "error": f"step {i} failed"}

        except Exception as e:
            logger.error("Compiled template step %d exception: %s", i, e)
            return {"success": False, "results": step_results, "error": str(e)}

    logger.info("Compiled template executed successfully (%d steps)", len(steps))
    return {"success": True, "results": step_results}


def _resolve_template(template: Any, values: dict[str, Any], step_results: dict[int, dict]) -> Any:
    """Recursively resolve placeholders and step references in a template."""
    if isinstance(template, dict):
        return {k: _resolve_template(v, values, step_results) for k, v in template.items()}
    if isinstance(template, list):
        return [_resolve_template(item, values, step_results) for item in template]
    if isinstance(template, str):
        # Check for {{fieldName}} placeholder
        placeholder_match = re.fullmatch(r"\{\{(\w+)\}\}", template)
        if placeholder_match:
            field = placeholder_match.group(1)
            if field in values:
                return values[field]
            logger.warning("Missing extracted value for placeholder: %s", field)
            return template

        # Check for $step_N.path references
        ref_match = re.match(r"\$step_(\d+)\.(.+)", template)
        if ref_match:
            step_idx = int(ref_match.group(1))
            dotpath = ref_match.group(2)
            step_data = step_results.get(step_idx)
            if step_data is not None:
                parts = dotpath.split(".")
                result = step_data
                for part in parts:
                    if isinstance(result, dict):
                        result = result.get(part)
                    else:
                        result = None
                        break
                if result is not None:
                    return result
            logger.warning("Could not resolve step reference: %s", template)
            return template

    return template


# ---------------------------------------------------------------------------
# Layer 3: Adaptive Routing
# ---------------------------------------------------------------------------

def record_result(task_type: str, path_used: str, success: bool) -> None:
    """Update success/failure stats for a task type and execution path."""
    key = f"{path_used}_{'ok' if success else 'fail'}"

    with _lock:
        if task_type not in TASK_STATS:
            TASK_STATS[task_type] = {
                "template_ok": 0,
                "template_fail": 0,
                "tool_ok": 0,
                "tool_fail": 0,
            }
        TASK_STATS[task_type][key] = TASK_STATS[task_type].get(key, 0) + 1
        logger.info(
            "Recorded result: %s via %s -> %s (stats: %s)",
            task_type, path_used, "ok" if success else "fail",
            TASK_STATS[task_type],
        )


def should_override_route(task_type: str) -> Optional[str]:
    """Check if we should override the default routing for a task type.

    Returns:
        "tool_agent" if template keeps failing but tool_agent works.
        "template" if tool_agent keeps failing but template works.
        None if no override recommended.
    """
    with _lock:
        stats = TASK_STATS.get(task_type)
        if not stats:
            return None

        t_fail = stats.get("template_fail", 0)
        t_ok = stats.get("template_ok", 0)
        a_fail = stats.get("tool_fail", 0)
        a_ok = stats.get("tool_ok", 0)

    if t_fail >= 2 and a_ok > 0 and a_ok > a_fail:
        logger.info("Override: %s -> tool_agent (template_fail=%d, tool_ok=%d)", task_type, t_fail, a_ok)
        return "tool_agent"

    if a_fail >= 2 and t_ok > 0 and t_ok > t_fail:
        logger.info("Override: %s -> template (tool_fail=%d, template_ok=%d)", task_type, a_fail, t_ok)
        return "template"

    return None
