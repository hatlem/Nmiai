# tripletex/agent.py
import json
import base64
import logging
from datetime import date

import vertexai
from vertexai.generative_models import GenerativeModel, Part

from prompts.classifier import CLASSIFIER_PROMPT
from prompts.planner import build_planner_prompt, build_self_repair_prompt
from templates import TEMPLATES, KEYWORD_HINTS

logger = logging.getLogger(__name__)

vertexai.init(project="ainm26osl-710", location="europe-north1")

MODEL_PRO = "gemini-2.5-pro"
MODEL_FLASH = "gemini-2.5-flash"


def _get_model(system_instruction: str, model_id: str = MODEL_PRO) -> GenerativeModel:
    return GenerativeModel(model_id, system_instruction=system_instruction)


def _parse_json(text: str) -> dict:
    """Parse LLM response, handling various markdown/fence formats."""
    text = text.strip()

    # Strip any markdown fences (```json, ```, etc.)
    import re
    text = re.sub(r'^```\w*\n?', '', text)
    text = re.sub(r'\n?```$', '', text)
    text = text.strip()

    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try to find JSON object with balanced braces
    start = text.find("{")
    if start >= 0:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i + 1])
                    except json.JSONDecodeError:
                        pass
                    break

    # Last resort: find first { to last }
    end = text.rfind("}") + 1
    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end])
        except json.JSONDecodeError:
            pass

    logger.error(f"Could not parse JSON from: {text[:200]}")
    raise json.JSONDecodeError("No valid JSON found", text, 0)


def _quick_classify(prompt: str) -> str | None:
    """Try keyword-based classification before calling LLM.
    Scores by keyword length — longer (more specific) matches win.
    Returns None on ambiguous matches to let LLM decide."""
    prompt_lower = prompt.lower()
    best_type = None
    best_len = 0
    second_best_len = 0
    for task_type, keywords in KEYWORD_HINTS.items():
        for kw in keywords:
            if kw.lower() in prompt_lower:
                if len(kw) > best_len:
                    second_best_len = best_len
                    best_type = task_type
                    best_len = len(kw)
                elif len(kw) > second_best_len:
                    second_best_len = len(kw)
    # If the best match is only marginally better than second, let LLM decide
    if best_type and best_len > second_best_len + 2:
        return best_type
    # Short keywords (< 5 chars) are too ambiguous on their own
    if best_type and best_len < 5:
        return None
    return best_type


async def classify_task(prompt: str) -> str:
    """Stage 1: Classify the task type. Try keywords first, fall back to LLM."""
    quick = _quick_classify(prompt)
    if quick:
        logger.info(f"Quick classify: {quick}")
        return quick

    model = _get_model(CLASSIFIER_PROMPT, model_id=MODEL_FLASH)
    response = await model.generate_content_async(
        prompt,
        generation_config={"temperature": 0.0, "max_output_tokens": 100},
    )
    result = _parse_json(response.text)
    task_type = result.get("task_type", "unknown")
    logger.info(f"LLM classify: {task_type}")
    return task_type


async def create_plan(prompt: str, files: list[dict] | None = None) -> dict:
    """Two-stage planning: classify then fill template."""
    task_type = await classify_task(prompt)

    planner_prompt = build_planner_prompt(task_type)
    model = _get_model(planner_prompt)

    parts = []
    if files:
        for f in files:
            file_data = base64.b64decode(f["content_base64"])
            parts.append(Part.from_data(data=file_data, mime_type=f["mime_type"]))
            parts.append(Part.from_text(f"[Attached file: {f['filename']}]"))

    today = date.today().isoformat()
    parts.append(Part.from_text(
        f"Today's date is {today}. Complete this accounting task:\n\n{prompt}"
    ))

    logger.info(f"Planning {task_type}: {prompt[:80]}...")
    response = await model.generate_content_async(
        parts,
        generation_config={"temperature": 0.0, "max_output_tokens": 4096},
    )

    try:
        plan = _parse_json(response.text)
    except (json.JSONDecodeError, Exception) as e:
        logger.error(f"Failed to parse plan JSON: {e}. Raw: {response.text[:200]}")
        template = TEMPLATES.get(task_type, TEMPLATES["unknown"])
        plan = {"task_type": task_type, "reasoning": "Fallback - LLM JSON parse failed", "steps": template["steps"]}

    plan["task_type"] = plan.get("task_type", task_type)
    logger.info(f"Plan: {plan['task_type']} with {len(plan.get('steps', []))} steps")
    return plan


async def self_repair(
    original_prompt: str,
    plan: dict,
    results: dict,
    failed: list,
) -> dict:
    """Self-repair: feed errors to LLM for a corrected plan."""
    task_type = plan.get("task_type", "unknown")
    repair_prompt = build_self_repair_prompt(
        task_type, original_prompt, plan, results, failed
    )
    model = _get_model("You are an expert Tripletex API debugger. Fix the failed plan.")

    logger.info("Self-repair: sending errors to LLM...")
    response = await model.generate_content_async(
        repair_prompt,
        generation_config={"temperature": 0.0, "max_output_tokens": 4096},
    )

    repaired = _parse_json(response.text)
    logger.info(f"Repaired plan: {len(repaired.get('steps', []))} steps")
    return repaired
