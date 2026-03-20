"""
Self-learning module for the Tripletex agent.

Three-layer in-memory learning system:
1. Error Memory — remembers past errors and extracts lessons
2. Success Replay — stores proven API call sequences for task types
3. Adaptive Routing — tracks template vs tool_agent success rates
"""

import logging
import re
import threading
from typing import Optional

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
MAX_SUCCESS_PATTERNS = 100

# ---------------------------------------------------------------------------
# In-memory stores
# ---------------------------------------------------------------------------

ERROR_MEMORY: list[dict] = []
SUCCESS_PATTERNS: dict[str, list[dict]] = {}
TASK_STATS: dict[str, dict] = {}

_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_keywords(prompt: str) -> frozenset[str]:
    """Extract meaningful keywords from a prompt string.

    Lowercases, strips punctuation, removes stop words, returns sorted frozenset.
    """
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


# ---------------------------------------------------------------------------
# Layer 1: Error Memory
# ---------------------------------------------------------------------------

def record_error(endpoint: str, error_data: str | dict, prompt: str) -> None:
    """Parse an error response and store a lesson learned.

    If a similar lesson already exists (same endpoint + overlapping keywords),
    the existing entry's count is incremented instead of adding a duplicate.
    """
    if isinstance(error_data, dict):
        error_msg = str(error_data.get("message", error_data.get("error", str(error_data))))
    else:
        error_msg = str(error_data)

    # Build a concise lesson from the error
    lesson = f"{endpoint}: {error_msg[:120]}"
    keywords = list(sorted(_extract_keywords(prompt)))

    with _lock:
        # Check for similar existing entry
        for entry in ERROR_MEMORY:
            if entry["endpoint"] == endpoint and entry["error"] == error_msg:
                entry["count"] += 1
                logger.debug("Incremented error count for %s: %d", endpoint, entry["count"])
                return

        ERROR_MEMORY.append({
            "endpoint": endpoint,
            "error": error_msg,
            "lesson": lesson,
            "task_keywords": keywords,
            "count": 1,
        })
        logger.info("Recorded error lesson: %s", lesson)

        # FIFO eviction
        while len(ERROR_MEMORY) > MAX_ERROR_MEMORY:
            evicted = ERROR_MEMORY.pop(0)
            logger.debug("Evicted oldest error entry: %s", evicted["endpoint"])


def get_lessons(prompt: str) -> str:
    """Return formatted lessons relevant to this prompt, matched by keywords."""
    prompt_kw = _extract_keywords(prompt)
    if not prompt_kw:
        return ""

    with _lock:
        scored = []
        for entry in ERROR_MEMORY:
            entry_kw = frozenset(entry["task_keywords"])
            score = _match_score(prompt_kw, entry_kw)
            if score > 0.1:
                scored.append((score, entry))

    if not scored:
        return ""

    scored.sort(key=lambda x: (-x[0], -x[1]["count"]))
    lines = ["LEARNED FROM PREVIOUS ERRORS:"]
    for _, entry in scored[:10]:
        lines.append(f"- {entry['lesson']}")

    result = "\n".join(lines)
    logger.debug("Returning %d lessons for prompt", len(scored))
    return result


# ---------------------------------------------------------------------------
# Layer 2: Success Replay
# ---------------------------------------------------------------------------

def _task_signature(prompt: str) -> str:
    """Derive a task signature from prompt keywords (sorted, joined with +)."""
    kw = _extract_keywords(prompt)
    return "+".join(sorted(kw))


def record_success(prompt: str, api_log: list[dict]) -> None:
    """Store a proven API call sequence for a task type.

    api_log entries should have: method, path, and optionally body (dict).
    We store method + path + sorted body keys (not values) for privacy.
    """
    sig = _task_signature(prompt)
    if not sig:
        return

    compact_log = []
    for call in api_log:
        entry: dict = {
            "method": call.get("method", "GET"),
            "path": call.get("path", ""),
        }
        body = call.get("body")
        if isinstance(body, dict):
            entry["body_keys"] = sorted(body.keys())
        compact_log.append(entry)

    with _lock:
        SUCCESS_PATTERNS[sig] = compact_log
        logger.info("Recorded success pattern: %s (%d calls)", sig, len(compact_log))

        # Evict oldest entries if over limit
        while len(SUCCESS_PATTERNS) > MAX_SUCCESS_PATTERNS:
            oldest_key = next(iter(SUCCESS_PATTERNS))
            del SUCCESS_PATTERNS[oldest_key]
            logger.debug("Evicted oldest success pattern: %s", oldest_key)


def get_proven_pattern(prompt: str) -> str:
    """Find the best matching proven API sequence for this prompt."""
    prompt_kw = _extract_keywords(prompt)
    if not prompt_kw:
        return ""

    with _lock:
        best_score = 0.0
        best_log: list[dict] = []
        for sig, log in SUCCESS_PATTERNS.items():
            sig_kw = frozenset(sig.split("+"))
            score = _match_score(prompt_kw, sig_kw)
            if score > best_score:
                best_score = score
                best_log = log

    if best_score < 0.2 or not best_log:
        return ""

    lines = ["PROVEN PATTERN (follow this sequence):"]
    for i, call in enumerate(best_log, 1):
        desc = f"{i}. {call['method']} {call['path']}"
        if "body_keys" in call:
            desc += " {" + ", ".join(call["body_keys"]) + "}"
        lines.append(desc)

    result = "\n".join(lines)
    logger.debug("Returning proven pattern (score=%.2f, %d steps)", best_score, len(best_log))
    return result


# ---------------------------------------------------------------------------
# Layer 3: Adaptive Routing
# ---------------------------------------------------------------------------

def record_result(task_type: str, path_used: str, success: bool) -> None:
    """Update success/failure stats for a task type and execution path.

    Args:
        task_type: Identifier for the task (e.g. "create_employee").
        path_used: Either "template" or "tool_agent".
        success: Whether the execution succeeded.
    """
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

    # Template fails 2+ times and tool_agent has successes
    if t_fail >= 2 and a_ok > 0 and a_ok > a_fail:
        logger.info("Override: %s -> tool_agent (template_fail=%d, tool_ok=%d)", task_type, t_fail, a_ok)
        return "tool_agent"

    # Tool agent fails 2+ times and template has successes
    if a_fail >= 2 and t_ok > 0 and t_ok > t_fail:
        logger.info("Override: %s -> template (tool_fail=%d, template_ok=%d)", task_type, a_fail, t_ok)
        return "template"

    return None
