# tripletex/agent.py
"""Two-stage agent with model routing, confidence-based classification,
and extracted_values output for downstream verification."""

import json
import base64
import logging
import re
from datetime import date

import vertexai
from vertexai.generative_models import GenerativeModel, Part

from prompts.classifier import CLASSIFIER_PROMPT, CLASSIFIER_PROMPT_PRO
from prompts.planner import build_planner_prompt, build_self_repair_prompt
from templates import TEMPLATES, KEYWORD_HINTS

logger = logging.getLogger(__name__)

vertexai.init(project="ainm26osl-710", location="global")

# ---------- Model IDs ----------
MODEL_PRO = "gemini-3.1-pro-preview"
MODEL_FLASH_LITE = "gemini-3.1-flash-lite-preview"

# ---------- Tier mapping ----------
TIER_MAP: dict[str, int] = {
    "create_employee": 1,
    "create_customer": 1,
    "create_product": 1,
    "create_department": 1,
    "create_supplier": 1,
    "create_contact": 1,
    "update_employee": 2,
    "update_customer": 2,
    "create_invoice": 2,
    "create_invoice_existing_customer": 2,
    "create_invoice_with_payment": 3,
    "register_payment": 2,
    "create_credit_note": 2,
    "send_invoice": 1,
    "create_travel_expense": 2,
    "delete_travel_expense": 1,
    "deliver_travel_expense": 1,
    "approve_travel_expense": 1,
    "create_project": 2,
    "create_internal_project": 1,
    "update_project": 2,
    "create_voucher": 2,
    "reverse_voucher": 1,
    "delete_entity": 1,
    "create_supplier_invoice": 2,
    "create_purchase_order": 2,
    "bank_reconciliation": 3,
    "create_timesheet_entry": 2,
    "create_opening_balance": 3,
    "create_asset": 2,
    "create_salary_payment": 2,
    "create_customer_supplier": 1,
    "create_reminder": 2,
    "create_employment": 2,
    "unknown": 3,
}

CONFIDENCE_THRESHOLD = 0.7


def _get_model(model_id: str, system_instruction: str) -> GenerativeModel:
    return GenerativeModel(model_id, system_instruction=system_instruction)


def get_tier(task_type: str) -> int:
    return TIER_MAP.get(task_type, 3)


def _parse_json(text: str) -> dict:
    """Parse LLM response, handling various markdown/fence formats."""
    text = text.strip()
    text = re.sub(r'^```\w*\n?', '', text)
    text = re.sub(r'\n?```$', '', text)
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

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

    end = text.rfind("}") + 1
    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end])
        except json.JSONDecodeError:
            pass

    logger.error(f"Could not parse JSON from: {text[:200]}")
    raise json.JSONDecodeError("No valid JSON found", text, 0)


def _quick_classify(prompt: str) -> tuple[str, float] | None:
    """Try keyword-based classification before calling LLM."""
    prompt_lower = prompt.lower()

    high_conf_keywords = {
        "slett reiseregning": ("delete_travel_expense", 0.95),
        "delete travel": ("delete_travel_expense", 0.95),
        "lever reiseregning": ("deliver_travel_expense", 0.95),
        "deliver travel": ("deliver_travel_expense", 0.95),
        "godkjenn reiseregning": ("approve_travel_expense", 0.95),
        "approve travel": ("approve_travel_expense", 0.95),
        "send faktura": ("send_invoice", 0.95),
        "send invoice": ("send_invoice", 0.95),
        "oppdater ansatt": ("update_employee", 0.90),
        "endre ansatt": ("update_employee", 0.90),
        "update employee": ("update_employee", 0.90),
        "oppdater kunde": ("update_customer", 0.90),
        "endre kunde": ("update_customer", 0.90),
        "update customer": ("update_customer", 0.90),
        "oppdater prosjekt": ("update_project", 0.90),
        "update project": ("update_project", 0.90),
        "internt prosjekt": ("create_internal_project", 0.90),
        "internal project": ("create_internal_project", 0.90),
        "kontaktperson": ("create_contact", 0.90),
        "contact person": ("create_contact", 0.90),
        "reverser": ("reverse_voucher", 0.90),
        "reverse voucher": ("reverse_voucher", 0.90),
        "tilbakefor": ("reverse_voucher", 0.90),
        "kreditnota": ("create_credit_note", 0.90),
        "credit note": ("create_credit_note", 0.90),
        "gutschrift": ("create_credit_note", 0.90),
        "leverandorfaktura": ("create_supplier_invoice", 0.90),
        "supplier invoice": ("create_supplier_invoice", 0.90),
        "inngaende faktura": ("create_supplier_invoice", 0.90),
        "kunde og leverandor": ("create_customer_supplier", 0.90),
        "customer and supplier": ("create_customer_supplier", 0.90),
        "purring": ("create_reminder", 0.90),
        "reminder": ("create_reminder", 0.85),
        "ansettelse": ("create_employment", 0.85),
        "employment": ("create_employment", 0.85),
    }
    for phrase, (task_type, conf) in high_conf_keywords.items():
        if phrase in prompt_lower:
            return task_type, conf

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

    if best_type and best_len > second_best_len + 2:
        return best_type, 0.75
    if best_type and best_len >= 5:
        return best_type, 0.65
    return None


async def classify_task(prompt: str) -> tuple[str, float]:
    """Stage 1: Classify with confidence. Keyword -> Flash-Lite -> Pro escalation."""
    quick = _quick_classify(prompt)
    if quick:
        task_type, confidence = quick
        if confidence >= CONFIDENCE_THRESHOLD:
            logger.info(f"Quick classify: {task_type} (conf={confidence:.2f})")
            return task_type, confidence

    model = _get_model(MODEL_FLASH_LITE, CLASSIFIER_PROMPT)
    try:
        response = await model.generate_content_async(
            prompt,
            generation_config={"temperature": 0.0, "max_output_tokens": 200},
        )
        result = _parse_json(response.text)
        task_type = result.get("task_type", "unknown")
        confidence = float(result.get("confidence", 0.5))
        logger.info(f"Flash-Lite classify: {task_type} (conf={confidence:.2f})")
    except Exception as e:
        logger.error(f"Flash-Lite classification failed: {e}")
        task_type, confidence = "unknown", 0.0

    if confidence < CONFIDENCE_THRESHOLD:
        logger.info(f"Low confidence ({confidence:.2f}), escalating to Pro")
        pro_model = _get_model(MODEL_PRO, CLASSIFIER_PROMPT_PRO)
        try:
            response = await pro_model.generate_content_async(
                prompt,
                generation_config={"temperature": 0.0, "max_output_tokens": 300},
            )
            result = _parse_json(response.text)
            task_type = result.get("task_type", "unknown")
            confidence = float(result.get("confidence", 0.5))
            logger.info(f"Pro classify: {task_type} (conf={confidence:.2f})")
        except Exception as e:
            logger.error(f"Pro classification failed: {e}")

    if confidence < 0.5:
        logger.warning(f"Very low confidence ({confidence:.2f}) -> unknown")
        task_type = "unknown"

    return task_type, confidence


def _build_file_instructions(files: list[dict] | None) -> str:
    if not files:
        return ""
    instructions = []
    for f in files:
        mime = f.get("mime_type", "")
        name = f.get("filename", "unknown")
        if "pdf" in mime:
            instructions.append(
                f"ATTACHED PDF: '{name}' - Extract ALL data: every line item, "
                f"amounts (gross/net/VAT), dates, reference numbers, account numbers, "
                f"customer/supplier names, addresses, org numbers. "
                f"The PDF IS the data source - do not guess values."
            )
        elif "image" in mime:
            instructions.append(
                f"ATTACHED IMAGE: '{name}' - Read ALL visible text: invoice details, "
                f"amounts, dates, account numbers, names."
            )
        elif "csv" in mime or "spreadsheet" in mime or "excel" in mime:
            instructions.append(
                f"ATTACHED SPREADSHEET/CSV: '{name}' - Parse ALL rows/columns."
            )
        else:
            instructions.append(f"ATTACHED FILE: '{name}' ({mime}) - Extract all relevant data.")
    return "\n".join(instructions)


async def create_plan(prompt: str, files: list[dict] | None = None) -> dict:
    """Two-stage planning: classify then fill template."""
    task_type, confidence = await classify_task(prompt)
    tier = get_tier(task_type)

    planner_prompt = build_planner_prompt(task_type, tier)
    model = _get_model(MODEL_PRO, planner_prompt)

    parts = []
    if files:
        for f in files:
            file_data = base64.b64decode(f["content_base64"])
            parts.append(Part.from_data(data=file_data, mime_type=f["mime_type"]))
            parts.append(Part.from_text(f"[Attached file: {f['filename']}]"))

    file_instructions = _build_file_instructions(files)
    today = date.today().isoformat()

    task_text = f"Today's date is {today}.\n"
    if file_instructions:
        task_text += f"\n{file_instructions}\n\n"
    task_text += f"Complete this accounting task:\n\n{prompt}"
    parts.append(Part.from_text(task_text))

    logger.info(f"Planning {task_type} (tier={tier}): {prompt[:80]}...")
    response = await model.generate_content_async(
        parts,
        generation_config={"temperature": 0.0, "max_output_tokens": 4096},
    )

    try:
        plan = _parse_json(response.text)
    except (json.JSONDecodeError, Exception) as e:
        logger.error(f"Failed to parse plan JSON: {e}. Raw: {response.text[:300]}")
        template = TEMPLATES.get(task_type, TEMPLATES["unknown"])
        plan = {
            "task_type": task_type,
            "reasoning": "Fallback - LLM JSON parse failed",
            "steps": template["steps"],
            "extracted_values": {},
        }

    plan["task_type"] = plan.get("task_type", task_type)
    plan["tier"] = tier
    plan["classification_confidence"] = confidence
    plan.setdefault("extracted_values", {})
    logger.info(
        f"Plan: {plan['task_type']} with {len(plan.get('steps', []))} steps, "
        f"extracted {len(plan.get('extracted_values', {}))} values"
    )
    return plan


async def self_repair(
    original_prompt: str,
    plan: dict,
    results: dict,
    failed: list,
    verification_errors: list[dict] | None = None,
) -> dict:
    """Self-repair: feed errors + optional verification mismatches to Pro model."""
    task_type = plan.get("task_type", "unknown")
    repair_prompt = build_self_repair_prompt(
        task_type, original_prompt, plan, results, failed,
        verification_errors=verification_errors,
    )

    model = _get_model(MODEL_PRO, "You are an expert Tripletex API debugger. Fix the failed plan.")

    logger.info(f"Self-repair (verification_errors={bool(verification_errors)})")
    response = await model.generate_content_async(
        repair_prompt,
        generation_config={"temperature": 0.0, "max_output_tokens": 4096},
    )

    try:
        repaired = _parse_json(response.text)
    except (json.JSONDecodeError, Exception) as e:
        logger.error(f"Self-repair JSON parse failed: {e}. Raw: {response.text[:200]}")
        # Return empty plan so the repair loop knows to stop
        return {
            "task_type": task_type,
            "reasoning": "Self-repair JSON parse failed",
            "steps": [],
            "extracted_values": plan.get("extracted_values", {}),
        }

    repaired.setdefault("extracted_values", plan.get("extracted_values", {}))
    logger.info(f"Repaired plan: {len(repaired.get('steps', []))} steps")
    return repaired
