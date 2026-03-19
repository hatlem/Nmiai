"""Stage 1: Classify the accounting task type.

Lightweight prompt — no API reference needed.
Returns just the task_type string.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from templates import TEMPLATES, KEYWORD_HINTS

TASK_LIST = "\n".join(
    f"- {task_type}: {t['description']}"
    for task_type, t in TEMPLATES.items()
    if task_type != "unknown"
)

CLASSIFIER_PROMPT = f"""You are a task classifier for an accounting system. Given a task prompt (in any language: Norwegian, English, Spanish, Portuguese, Nynorsk, German, or French), identify which task type it belongs to.

## Available Task Types
{TASK_LIST}

## Output
Return ONLY a JSON object:
{{"task_type": "<one of the task types above>"}}

If the task doesn't match any known type, return:
{{"task_type": "unknown"}}

Do not include any other text, markdown, or explanation.
"""
