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
    "register_payment_by_search": 2,
    "create_credit_note": 2,
    "send_invoice": 1,
    "create_travel_expense": 2,
    "delete_travel_expense": 1,
    "deliver_travel_expense": 1,
    "approve_travel_expense": 1,
    "create_project": 2,
    "create_project_existing_customer": 2,
    "create_internal_project": 1,
    "update_project": 2,
    "update_supplier": 2,
    "update_department": 2,
    "update_product": 2,
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
    "enable_modules": 1,
    "unknown": 3,
}

CONFIDENCE_THRESHOLD = 0.55


def _get_model(model_id: str, system_instruction: str) -> GenerativeModel:
    return GenerativeModel(model_id, system_instruction=system_instruction)


def get_tier(task_type: str) -> int:
    return TIER_MAP.get(task_type, 3)


def _parse_json(text: str) -> dict:
    """Parse LLM response, handling various markdown/fence formats."""
    text = text.strip()
    # Remove all markdown code fences (possibly multiple)
    text = re.sub(r'```\w*\s*', '', text)
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Find the outermost { ... } using rfind for the closing brace
    start = text.find("{")
    end = text.rfind("}") + 1
    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end])
        except json.JSONDecodeError:
            pass

    # Try fixing common LLM issues: trailing commas
    if start >= 0 and end > start:
        candidate = text[start:end]
        candidate = re.sub(r',\s*([}\]])', r'\1', candidate)
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

        # Try replacing single quotes with double quotes
        candidate2 = candidate.replace("'", '"')
        try:
            return json.loads(candidate2)
        except json.JSONDecodeError:
            pass

    # Try to recover truncated JSON by closing open braces/brackets
    if start >= 0:
        candidate = text[start:]
        # Count unmatched braces
        open_braces = candidate.count("{") - candidate.count("}")
        open_brackets = candidate.count("[") - candidate.count("]")
        if open_braces > 0 or open_brackets > 0:
            # Truncate at last complete value (after last comma or colon+value)
            # Then close all open structures
            suffix = "]" * max(0, open_brackets) + "}" * max(0, open_braces)
            # Try removing partial trailing content after last complete entry
            # Look for last , or { or [ followed by incomplete content
            truncated = candidate.rstrip()
            # Remove trailing partial key-value pair
            truncated = re.sub(r',\s*"[^"]*"?\s*:?\s*"?[^"]*$', '', truncated)
            truncated = re.sub(r',\s*\{[^}]*$', '', truncated)
            truncated = truncated.rstrip().rstrip(",")
            suffix = "]" * max(0, truncated.count("[") - truncated.count("]"))
            suffix += "}" * max(0, truncated.count("{") - truncated.count("}"))
            try:
                result = json.loads(truncated + suffix)
                logger.warning(f"Recovered truncated JSON ({len(text)} chars -> {len(truncated)} used)")
                return result
            except json.JSONDecodeError:
                pass

    logger.error(f"Could not parse JSON from ({len(text)} chars): {text[:300]}")
    raise json.JSONDecodeError("No valid JSON found", text, 0)


def _quick_classify(prompt: str) -> tuple[str, float] | None:
    """Try keyword-based classification before calling LLM."""
    prompt_lower = prompt.lower()

    # Detect "payment on invoice NUMBER" pattern -> register_payment_by_search
    # This must come before high_conf_keywords since "betaling"/"payment" would match register_payment
    _has_payment = bool(re.search(r'\b(betal|betaling|innbetaling|payment|paiement|zahlung|pago)\b', prompt_lower))
    _has_invoice_number = bool(re.search(
        r'(faktura\s*(nr|nummer|#)\s*\d+|invoice\s*(nr|number|#|no\.?)\s*\d+|factura\s*(nr|numero|#)\s*\d+|rechnung\s*(nr|nummer|#)\s*\d+)',
        prompt_lower,
    ))
    if _has_payment and _has_invoice_number:
        return "register_payment_by_search", 0.92

    # Detect timesheet patterns: "N timer" or "N.N timer" (must come before "prosjekt" match)
    if re.search(r'\d+[\.,]?\d*\s*timer\b', prompt_lower):
        return "create_timesheet_entry", 0.90

    high_conf_keywords = {
        # Existing entity detection - must come BEFORE generic patterns
        "faktura for eksisterende": ("create_invoice_existing_customer", 0.95),
        "invoice for existing": ("create_invoice_existing_customer", 0.95),
        "faktura til eksisterende": ("create_invoice_existing_customer", 0.95),
        "invoice to existing": ("create_invoice_existing_customer", 0.95),
        "factura para cliente existente": ("create_invoice_existing_customer", 0.90),
        "rechnung fur bestehenden": ("create_invoice_existing_customer", 0.90),
        "facture pour client existant": ("create_invoice_existing_customer", 0.90),
        "prosjekt for eksisterende": ("create_project_existing_customer", 0.95),
        "project for existing": ("create_project_existing_customer", 0.95),
        "prosjekt til eksisterende": ("create_project_existing_customer", 0.95),
        "proyecto para cliente existente": ("create_project_existing_customer", 0.90),
        # Standard patterns
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
        "oppdater leverandor": ("update_supplier", 0.90),
        "endre leverandor": ("update_supplier", 0.90),
        "update supplier": ("update_supplier", 0.90),
        "oppdater avdeling": ("update_department", 0.90),
        "endre avdeling": ("update_department", 0.90),
        "update department": ("update_department", 0.90),
        "oppdater produkt": ("update_product", 0.90),
        "endre produkt": ("update_product", 0.90),
        "update product": ("update_product", 0.90),
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
        "betal faktura nummer": ("register_payment_by_search", 0.92),
        "betal faktura nr": ("register_payment_by_search", 0.92),
        "registrer betaling pa faktura": ("register_payment_by_search", 0.92),
        "registrer betaling på faktura": ("register_payment_by_search", 0.92),
        "payment on invoice number": ("register_payment_by_search", 0.92),
        "payment on invoice no": ("register_payment_by_search", 0.92),
        "betaling for faktura": ("register_payment_by_search", 0.90),
        "betaling på faktura": ("register_payment_by_search", 0.90),
        "pay invoice number": ("register_payment_by_search", 0.90),
        "purring": ("create_reminder", 0.90),
        "send purring": ("create_reminder", 0.92),
        "payment reminder": ("create_reminder", 0.88),
        "betalingspaminnelse": ("create_reminder", 0.90),
        "betalingspåminnelse": ("create_reminder", 0.90),
        "ansettelse": ("create_employment", 0.85),
        "employment": ("create_employment", 0.85),
        # Nynorsk patterns
        "opprett tilsett": ("create_employee", 0.90),
        "ny tilsett": ("create_employee", 0.90),
        "registrer tilsett": ("create_employee", 0.90),
        "opprett tilsatt": ("create_employee", 0.90),
        # German patterns
        "mitarbeiter erstellen": ("create_employee", 0.90),
        "kunde erstellen": ("create_customer", 0.90),
        "rechnung erstellen": ("create_invoice", 0.90),
        "lieferant erstellen": ("create_supplier", 0.90),
        "produkt erstellen": ("create_product", 0.90),
        "projekt erstellen": ("create_project", 0.90),
        "abteilung erstellen": ("create_department", 0.90),
        # French patterns
        "creer employe": ("create_employee", 0.90),
        "creer client": ("create_customer", 0.90),
        "creer facture": ("create_invoice", 0.90),
        "creer fournisseur": ("create_supplier", 0.90),
        "creer produit": ("create_product", 0.90),
        # Spanish patterns
        "crear empleado": ("create_employee", 0.90),
        "crear cliente": ("create_customer", 0.90),
        "crear factura": ("create_invoice", 0.90),
        "crear proveedor": ("create_supplier", 0.90),
        "crear producto": ("create_product", 0.90),
        # Portuguese patterns
        "criar empregado": ("create_employee", 0.90),
        "criar cliente": ("create_customer", 0.90),
        "criar fatura": ("create_invoice", 0.90),
        # Invoice with payment
        "faktura med betaling": ("create_invoice_with_payment", 0.95),
        "invoice with payment": ("create_invoice_with_payment", 0.95),
        "faktura og registrer betaling": ("create_invoice_with_payment", 0.92),
        "faktura og betal": ("create_invoice_with_payment", 0.90),
        # Opening balance
        "apningsbalanse": ("create_opening_balance", 0.95),
        "åpningsbalanse": ("create_opening_balance", 0.95),
        "opening balance": ("create_opening_balance", 0.95),
        "inngaende balanse": ("create_opening_balance", 0.90),
        "inngående balanse": ("create_opening_balance", 0.90),
        # Bank reconciliation
        "bankavstemming": ("bank_reconciliation", 0.95),
        "bank reconciliation": ("bank_reconciliation", 0.95),
        # Asset
        "anleggsmiddel": ("create_asset", 0.90),
        "fixed asset": ("create_asset", 0.90),
        # Salary
        "lønnsutbetaling": ("create_salary_payment", 0.90),
        "lonnsutbetaling": ("create_salary_payment", 0.90),
        "salary payment": ("create_salary_payment", 0.90),
        # Timesheet
        "timeregistrering": ("create_timesheet_entry", 0.90),
        "timesheet entry": ("create_timesheet_entry", 0.90),
        "registrer timer": ("create_timesheet_entry", 0.90),
        # Supplier invoice
        "inngående faktura": ("create_supplier_invoice", 0.90),
        # Purchase order
        "innkjøpsordre": ("create_purchase_order", 0.90),
        "bestilling fra leverandor": ("create_purchase_order", 0.88),
        "bestilling fra leverandør": ("create_purchase_order", 0.88),
        # Supplier invoice extras
        "opprett leverandorfaktura": ("create_supplier_invoice", 0.95),
        "registrer leverandørfaktura": ("create_supplier_invoice", 0.95),
        "ny leverandorfaktura": ("create_supplier_invoice", 0.90),
        "create supplier invoice": ("create_supplier_invoice", 0.95),
        "incoming invoice": ("create_supplier_invoice", 0.90),
        "factura del proveedor": ("create_supplier_invoice", 0.90),
        "lieferantenrechnung": ("create_supplier_invoice", 0.90),
        # Update supplier with ø
        "oppdater leverandør": ("update_supplier", 0.90),
        "endre leverandør": ("update_supplier", 0.90),
        # Travel expense extras
        "registrer reiseregning": ("create_travel_expense", 0.90),
        "ny reiseregning": ("create_travel_expense", 0.90),
        "create travel expense": ("create_travel_expense", 0.90),
        # Timesheet extras
        "timeforing": ("create_timesheet_entry", 0.85),
        "timeføring": ("create_timesheet_entry", 0.85),
        "register hours": ("create_timesheet_entry", 0.85),
        "timer på prosjekt": ("create_timesheet_entry", 0.92),
        "timer pa prosjekt": ("create_timesheet_entry", 0.92),
        "hours on project": ("create_timesheet_entry", 0.92),
        "timer på": ("create_timesheet_entry", 0.88),
        "timer pa": ("create_timesheet_entry", 0.88),
        # Salary extras — require longer phrases to avoid matching "lonnansvarlig" etc.
        "utbetal lonn": ("create_salary_payment", 0.88),
        "utbetal lønn": ("create_salary_payment", 0.88),
        "registrer lonn": ("create_salary_payment", 0.88),
        "registrer lønn": ("create_salary_payment", 0.88),
        # Invoice with payment (multilingual)
        "factura con pago": ("create_invoice_with_payment", 0.90),
        "rechnung mit zahlung": ("create_invoice_with_payment", 0.90),
        "facture avec paiement": ("create_invoice_with_payment", 0.90),
        # "opprette" (create) + entity patterns
        "opprette kunde": ("create_customer", 0.90),
        "opprette faktura": ("create_invoice", 0.90),
        "opprette ansatt": ("create_employee", 0.90),
        "opprette leverandor": ("create_supplier", 0.90),
        "opprette leverandør": ("create_supplier", 0.90),
        "opprette produkt": ("create_product", 0.90),
        "opprette prosjekt": ("create_project", 0.90),
        "opprette avdeling": ("create_department", 0.90),
        "opprette kontakt": ("create_contact", 0.90),
        # "lag" (make) patterns
        "lag faktura": ("create_invoice", 0.88),
        "lag kunde": ("create_customer", 0.88),
        "lag ansatt": ("create_employee", 0.88),
        "lag leverandor": ("create_supplier", 0.88),
        "lag leverandør": ("create_supplier", 0.88),
        "lag produkt": ("create_product", 0.88),
        "lag prosjekt": ("create_project", 0.88),
        "lag avdeling": ("create_department", 0.88),
        # "ny" (new) patterns
        "ny kunde": ("create_customer", 0.88),
        "ny faktura": ("create_invoice", 0.88),
        "ny ansatt": ("create_employee", 0.88),
        "ny leverandor": ("create_supplier", 0.88),
        "ny leverandør": ("create_supplier", 0.88),
        "ny produkt": ("create_product", 0.88),
        "nytt prosjekt": ("create_project", 0.88),
        "ny avdeling": ("create_department", 0.88),
        # "registrer" (register) patterns
        "registrer kunde": ("create_customer", 0.90),
        "registrer ansatt": ("create_employee", 0.90),
        "registrer leverandor": ("create_supplier", 0.90),
        "registrer leverandør": ("create_supplier", 0.90),
        "registrer produkt": ("create_product", 0.88),
        # "new" patterns (English)
        "new employee": ("create_employee", 0.88),
        "new customer": ("create_customer", 0.88),
        "new supplier": ("create_supplier", 0.88),
        "new product": ("create_product", 0.88),
        "new invoice": ("create_invoice", 0.88),
        "new project": ("create_project", 0.88),
        "new department": ("create_department", 0.88),
        # "create" patterns (English)
        "create employee": ("create_employee", 0.90),
        "create customer": ("create_customer", 0.90),
        "create supplier": ("create_supplier", 0.90),
        "create product": ("create_product", 0.90),
        "create invoice": ("create_invoice", 0.90),
        "create project": ("create_project", 0.90),
        "create department": ("create_department", 0.90),
        # "register" patterns (English)
        "register supplier": ("create_supplier", 0.88),
        "register customer": ("create_customer", 0.88),
        "register employee": ("create_employee", 0.88),
        # Portuguese patterns
        "criar fornecedor": ("create_supplier", 0.90),
        "criar produto": ("create_product", 0.90),
        "criar projeto": ("create_project", 0.90),
        # Enable modules
        "aktiver modul": ("enable_modules", 0.90),
        "enable module": ("enable_modules", 0.90),
        "aktivere modul": ("enable_modules", 0.90),
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
        # Longer keyword matches are more trustworthy
        if best_len >= 8:
            return best_type, 0.70
        return best_type, 0.65
    return None


async def classify_task(prompt: str) -> tuple[str, float]:
    """Stage 1: Classify with confidence. Keyword -> Flash-Lite -> Pro escalation."""
    quick = _quick_classify(prompt)
    if quick:
        task_type, confidence = quick
        if confidence >= CONFIDENCE_THRESHOLD:
            logger.info(f"Quick classify: {task_type} (conf={confidence:.2f})")
            # FAST PATH: skip LLM for keyword matches with decent confidence
            # High confidence (>=0.85) always skips regardless of prompt length
            # Medium confidence (>=0.60) skips for shorter prompts
            if confidence >= 0.85 or (len(prompt) < 200 and confidence >= 0.60):
                logger.info(f"Fast path (conf={confidence:.2f}, {len(prompt)} chars): skipping LLM")
            return task_type, confidence

    # Flash-Lite classification
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

    # Only escalate to Pro if Flash-Lite is very uncertain (< 0.45)
    if confidence < 0.45:
        logger.info(f"Very low Flash-Lite confidence ({confidence:.2f}), escalating to Pro")
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

    max_tokens = 8192
    logger.info(f"Planning {task_type} (tier={tier}): {prompt[:80]}...")
    response = await model.generate_content_async(
        parts,
        generation_config={"temperature": 0.0, "max_output_tokens": max_tokens},
    )

    try:
        raw_text = response.text
    except (ValueError, AttributeError):
        raw_text = ""

    try:
        plan = _parse_json(raw_text)
    except (json.JSONDecodeError, Exception) as e:
        logger.error(f"Failed to parse plan JSON: {e}. Raw: {raw_text[:300]}")
        # Retry once with a simpler prompt asking for just the JSON
        try:
            retry_model = _get_model(MODEL_PRO, "Return ONLY valid JSON. No markdown fences. Keep reasoning under 20 words.")
            retry_response = await retry_model.generate_content_async(
                f"Fix this truncated JSON and complete it:\n{raw_text[:800]}\n\nReturn the complete valid JSON object.",
                generation_config={"temperature": 0.0, "max_output_tokens": 8192},
            )
            retry_text = retry_response.text if hasattr(retry_response, 'text') else ""
            plan = _parse_json(retry_text)
            logger.info("Plan JSON recovered via retry")
        except Exception:
            logger.error("Plan JSON retry also failed, using template fallback")
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
    files: list[dict] | None = None,
) -> dict:
    """Self-repair: feed errors + optional verification mismatches to Pro model."""
    task_type = plan.get("task_type", "unknown")
    repair_prompt = build_self_repair_prompt(
        task_type, original_prompt, plan, results, failed,
        verification_errors=verification_errors,
    )

    model = _get_model(MODEL_PRO, "You are an expert Tripletex API debugger. Fix the failed plan.")

    parts = []
    if files:
        for f in files:
            file_data = base64.b64decode(f["content_base64"])
            parts.append(Part.from_data(data=file_data, mime_type=f["mime_type"]))
            parts.append(Part.from_text(f"[Attached file: {f['filename']}]"))
    parts.append(Part.from_text(repair_prompt))

    logger.info(f"Self-repair (verification_errors={bool(verification_errors)}, files={len(files or [])})")
    response = await model.generate_content_async(
        parts,
        generation_config={"temperature": 0.0, "max_output_tokens": 8192},
    )

    try:
        raw_text = response.text
    except (ValueError, AttributeError) as e:
        logger.error(f"Self-repair: empty response from LLM: {e}")
        return {
            "task_type": task_type,
            "reasoning": "Self-repair: empty LLM response",
            "steps": [],
            "extracted_values": plan.get("extracted_values", {}),
        }

    try:
        repaired = _parse_json(raw_text)
    except (json.JSONDecodeError, Exception) as e:
        logger.error(f"Self-repair JSON parse failed: {e}. Raw ({len(raw_text)} chars): {raw_text[:400]}")
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
