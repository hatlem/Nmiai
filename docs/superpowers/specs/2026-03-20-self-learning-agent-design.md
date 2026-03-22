# Self-Learning Tripletex Agent Design

## Problem
Tool agent makes the same mistakes repeatedly across tasks. No memory between requests. Same 422 errors happen again and again.

## Solution
Three-layer learning system, all in-memory (persists for instance lifetime with min-instances=1):

### Layer 1: Error Memory
After each failed API call, extract a lesson from the error message and store it.

```python
ERROR_MEMORY: list[dict] = []
# {"pattern": "salary_rate_null", "lesson": "POST /salary/transaction requires payslips[].specifications with rate and count", "count": 3}
```

Before each task: filter relevant lessons by task type keywords, inject into prompt as "LEARNED FROM PREVIOUS ERRORS" section.

### Layer 2: Success Replay (Auto-Templates)
After each SUCCESSFUL tool agent task, store the API call sequence.

```python
SUCCESS_PATTERNS: dict[str, list[dict]] = {}
# Key: normalized task signature (e.g. "salary+bonus")
# Value: [{"method": "POST", "path": "/employee", "body_keys": ["firstName","lastName","email"]}, ...]
```

Before each tool agent task: check if we have a successful pattern for a similar task. If yes, inject it as "PROVEN PATTERN (follow this exactly)" into the prompt.

### Layer 3: Adaptive Routing
Track success rates per task type. If tool agent fails 2+ times on a pattern that template path handles, auto-route to template next time.

```python
ROUTING_OVERRIDES: dict[str, str] = {}
# {"salary+bonus": "tool_agent"}  — learned from experience
# Updated when a path succeeds/fails for a task type
```

## Data Flow

```
Request arrives
  → Check ROUTING_OVERRIDES for forced path
  → Check SUCCESS_PATTERNS for proven replay
  → Normal router (keyword matching)
  → Execute (template or tool agent)
  → After execution:
      If errors: add to ERROR_MEMORY
      If success + tool agent: add to SUCCESS_PATTERNS
      Update ROUTING_OVERRIDES success/fail counts
```

## Implementation

Single new file: `tripletex/learning.py` (~100 lines)

```python
# learning.py
ERROR_MEMORY: list[dict] = []
SUCCESS_PATTERNS: dict[str, list[dict]] = {}
TASK_STATS: dict[str, dict] = {}  # {task_sig: {template_ok: N, template_fail: N, tool_ok: N, tool_fail: N}}

def record_error(task_type: str, error_msg: str, endpoint: str) -> None
def record_success(task_type: str, api_calls: list[dict], path: str) -> None
def get_lessons(prompt: str) -> str  # Returns formatted lessons for prompt injection
def get_proven_pattern(prompt: str) -> list[dict] | None  # Returns replay sequence
def should_override_route(task_type: str) -> str | None  # "template" or "tool_agent" or None
```

Integration points:
- `main.py`: call `record_error`/`record_success` after execution, check `should_override_route` before routing
- `tool_agent.py`: inject `get_lessons()` + `get_proven_pattern()` into system prompt
- `agent.py`: inject `get_lessons()` into extraction prompt

## Competition Rules Compliance
- No hardcoded answers (we store errors and API sequences, not task answers)
- No ground truth extraction (we learn from our own API errors)
- Genuine AI/ML work (adaptive learning from experience)
- All data is ephemeral (in-memory, resets on deploy)

## Files Changed
| File | Change |
|---|---|
| `tripletex/learning.py` | NEW — learning module |
| `tripletex/main.py` | Call learning functions after each task |
| `tripletex/tool_agent.py` | Inject lessons + patterns into prompt |
