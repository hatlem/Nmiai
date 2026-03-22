# tripletex/agent.py
"""Two-stage agent: classify -> extract values -> template engine builds plan.

LLM never generates API steps for known task types. Only extracts field values.
"""

import asyncio
import json
import base64
import logging
import re
from datetime import date

import vertexai
from vertexai.generative_models import GenerativeModel, Part

from prompts.classifier import CLASSIFIER_PROMPT, CLASSIFIER_PROMPT_PRO
from prompts.planner import (
    build_extraction_prompt,
    build_repair_extraction_prompt,
    build_planner_prompt,
)
from template_engine import build_concrete_plan
from templates import TEMPLATES, KEYWORD_HINTS

logger = logging.getLogger(__name__)

vertexai.init(project="ainm26osl-710", location="global")

import warnings
warnings.filterwarnings("ignore", message=".*REST async clients.*")
warnings.filterwarnings("ignore", message=".*deprecated.*")

# ---------- Model IDs ----------
# Use 2.5 models — 3.1 is too slow (90s timeout on extraction)
MODEL_PRO = "gemini-2.5-pro"
MODEL_FLASH_LITE = "gemini-2.5-flash"

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
    "create_invoice_and_send": 2,
    "fixed_price_project_invoice": 3,
    "create_dimensions_voucher": 3,
    "create_full_credit_note": 3,
    "reverse_payment": 3,
    "unknown": 3,
}

CONFIDENCE_THRESHOLD = 0.55

# Fresh sandbox handling: each submission starts with an empty account.
_FRESH_SANDBOX_REMAP = {
    "register_payment": "create_invoice_with_payment",
    "register_payment_by_search": "create_invoice_with_payment",
    "create_invoice_existing_customer": "create_invoice",
    "create_project_existing_customer": "create_project",
    "create_credit_note": "create_full_credit_note",
}

# Pre-sorted keyword list (longest match first) — avoids re-sorting on every classify call
_HIGH_CONF_KEYWORDS = sorted({
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
    "reverser betaling": ("reverse_payment", 0.95),
    "reverse payment": ("reverse_payment", 0.95),
    "returnert av banken": ("reverse_payment", 0.95),
    "betaling returnert": ("reverse_payment", 0.95),
    "zurückgebucht": ("reverse_payment", 0.95),
    "stornieren zahlung": ("reverse_payment", 0.95),
    "stornieren betaling": ("reverse_payment", 0.92),
    "devuelto banco": ("reverse_payment", 0.92),
    "retourné banque": ("reverse_payment", 0.92),
    "annulez paiement": ("reverse_payment", 0.92),
    "payment returned": ("reverse_payment", 0.92),
    "reverser": ("reverse_voucher", 0.90),
    "reverse voucher": ("reverse_voucher", 0.90),
    "tilbakefor": ("reverse_voucher", 0.90),
    "kreditnota": ("create_full_credit_note", 0.92),
    "credit note": ("create_full_credit_note", 0.92),
    "gutschrift": ("create_full_credit_note", 0.92),
    "vollständige gutschrift": ("create_full_credit_note", 0.95),
    "reklamiert": ("create_full_credit_note", 0.90),
    "reklamert": ("create_full_credit_note", 0.90),
    "nota de crédito": ("create_full_credit_note", 0.92),
    "nota de credito": ("create_full_credit_note", 0.92),
    "note de crédit": ("create_full_credit_note", 0.92),
    "note de credit": ("create_full_credit_note", 0.90),
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
    "opprett tilsett": ("create_employee", 0.90),
    "ny tilsett": ("create_employee", 0.90),
    "registrer tilsett": ("create_employee", 0.90),
    "opprett tilsatt": ("create_employee", 0.90),
    "slett reiserekning": ("delete_travel_expense", 0.95),
    "lever reiserekning": ("deliver_travel_expense", 0.95),
    "godkjenn reiserekning": ("approve_travel_expense", 0.95),
    "oppdater tilsett": ("update_employee", 0.90),
    "oppdater tilsatt": ("update_employee", 0.90),
    "opprett leverandør": ("create_supplier", 0.90),
    "ny leverandørfaktura": ("create_supplier_invoice", 0.90),
    "bilag": ("create_voucher", 0.85),
    "innvendig prosjekt": ("create_internal_project", 0.85),
    "ny reiserekning": ("create_travel_expense", 0.90),
    "registrer reiserekning": ("create_travel_expense", 0.90),
    "mitarbeiter erstellen": ("create_employee", 0.90),
    "kunde erstellen": ("create_customer", 0.90),
    "rechnung erstellen": ("create_invoice", 0.90),
    "lieferant erstellen": ("create_supplier", 0.90),
    "produkt erstellen": ("create_product", 0.90),
    "projekt erstellen": ("create_project", 0.90),
    "abteilung erstellen": ("create_department", 0.90),
    "mitarbeiter aktualisieren": ("update_employee", 0.90),
    "kunde aktualisieren": ("update_customer", 0.90),
    "lieferant aktualisieren": ("update_supplier", 0.90),
    "produkt aktualisieren": ("update_product", 0.90),
    "abteilung aktualisieren": ("update_department", 0.90),
    "projekt aktualisieren": ("update_project", 0.90),
    "reisekosten erstellen": ("create_travel_expense", 0.90),
    "reisekosten loschen": ("delete_travel_expense", 0.90),
    "reisekosten löschen": ("delete_travel_expense", 0.90),
    "zahlung registrieren": ("register_payment", 0.90),
    "gehalt auszahlen": ("create_salary_payment", 0.88),
    "beleg erstellen": ("create_voucher", 0.90),
    "eröffnungsbilanz": ("create_opening_balance", 0.90),
    "creer employe": ("create_employee", 0.90),
    "creer client": ("create_customer", 0.90),
    "creer facture": ("create_invoice", 0.90),
    "creer fournisseur": ("create_supplier", 0.90),
    "creer produit": ("create_product", 0.90),
    "mettre a jour employe": ("update_employee", 0.88),
    "mettre a jour client": ("update_customer", 0.88),
    "mettre a jour fournisseur": ("update_supplier", 0.88),
    "mettre a jour produit": ("update_product", 0.88),
    "supprimer note de frais": ("delete_travel_expense", 0.90),
    "enregistrer paiement": ("register_payment", 0.88),
    "bon de commande": ("create_purchase_order", 0.88),
    "bilan d'ouverture": ("create_opening_balance", 0.88),
    "crear empleado": ("create_employee", 0.90),
    "crear cliente": ("create_customer", 0.90),
    "crear factura": ("create_invoice", 0.90),
    "crear proveedor": ("create_supplier", 0.90),
    "crear producto": ("create_product", 0.90),
    "actualizar empleado": ("update_employee", 0.88),
    "actualizar cliente": ("update_customer", 0.88),
    "actualizar proveedor": ("update_supplier", 0.88),
    "actualizar producto": ("update_product", 0.88),
    "eliminar gasto de viaje": ("delete_travel_expense", 0.90),
    "registrar pago": ("register_payment", 0.88),
    "orden de compra": ("create_purchase_order", 0.88),
    "balance de apertura": ("create_opening_balance", 0.88),
    "criar empregado": ("create_employee", 0.90),
    "criar cliente": ("create_customer", 0.90),
    "criar fatura": ("create_invoice", 0.90),
    "atualizar empregado": ("update_employee", 0.88),
    "atualizar cliente": ("update_customer", 0.88),
    "atualizar fornecedor": ("update_supplier", 0.88),
    "registrar pagamento": ("register_payment", 0.88),
    "faktura med betaling": ("create_invoice_with_payment", 0.95),
    "invoice with payment": ("create_invoice_with_payment", 0.95),
    "faktura og registrer betaling": ("create_invoice_with_payment", 0.92),
    "faktura og betal": ("create_invoice_with_payment", 0.90),
    "apningsbalanse": ("create_opening_balance", 0.95),
    "åpningsbalanse": ("create_opening_balance", 0.95),
    "opening balance": ("create_opening_balance", 0.95),
    "inngaende balanse": ("create_opening_balance", 0.90),
    "inngående balanse": ("create_opening_balance", 0.90),
    "bankavstemming": ("bank_reconciliation", 0.95),
    "bank reconciliation": ("bank_reconciliation", 0.95),
    "anleggsmiddel": ("create_asset", 0.90),
    "fixed asset": ("create_asset", 0.90),
    "lønnsutbetaling": ("create_salary_payment", 0.90),
    "lonnsutbetaling": ("create_salary_payment", 0.90),
    "salary payment": ("create_salary_payment", 0.90),
    "timeregistrering": ("create_timesheet_entry", 0.90),
    "timesheet entry": ("create_timesheet_entry", 0.90),
    "registrer timer": ("create_timesheet_entry", 0.90),
    "inngående faktura": ("create_supplier_invoice", 0.90),
    "bestilling fra leverandor": ("create_purchase_order", 0.88),
    "bestilling fra leverandør": ("create_purchase_order", 0.88),
    "opprett leverandorfaktura": ("create_supplier_invoice", 0.95),
    "opprett leverandørfaktura": ("create_supplier_invoice", 0.95),
    "registrer leverandørfaktura": ("create_supplier_invoice", 0.95),
    "ny leverandorfaktura": ("create_supplier_invoice", 0.90),
    "create supplier invoice": ("create_supplier_invoice", 0.95),
    "incoming invoice": ("create_supplier_invoice", 0.90),
    "factura del proveedor": ("create_supplier_invoice", 0.90),
    "lieferantenrechnung": ("create_supplier_invoice", 0.90),
    "oppdater leverandør": ("update_supplier", 0.90),
    "endre leverandør": ("update_supplier", 0.90),
    "registrer reiseregning": ("create_travel_expense", 0.90),
    "ny reiseregning": ("create_travel_expense", 0.90),
    "create travel expense": ("create_travel_expense", 0.90),
    "timeforing": ("create_timesheet_entry", 0.85),
    "timeføring": ("create_timesheet_entry", 0.85),
    "register hours": ("create_timesheet_entry", 0.85),
    "timer på prosjekt": ("create_timesheet_entry", 0.92),
    "timer pa prosjekt": ("create_timesheet_entry", 0.92),
    "hours on project": ("create_timesheet_entry", 0.92),
    "timer på": ("create_timesheet_entry", 0.88),
    "timer pa": ("create_timesheet_entry", 0.88),
    "utbetal lonn": ("create_salary_payment", 0.88),
    "utbetal lønn": ("create_salary_payment", 0.88),
    "registrer lonn": ("create_salary_payment", 0.88),
    "registrer lønn": ("create_salary_payment", 0.88),
    "factura con pago": ("create_invoice_with_payment", 0.90),
    "rechnung mit zahlung": ("create_invoice_with_payment", 0.90),
    "facture avec paiement": ("create_invoice_with_payment", 0.90),
    "opprette kunde": ("create_customer", 0.90),
    "opprette faktura": ("create_invoice", 0.90),
    "opprette ansatt": ("create_employee", 0.90),
    "opprette leverandor": ("create_supplier", 0.90),
    "opprette leverandør": ("create_supplier", 0.90),
    "opprette produkt": ("create_product", 0.90),
    "opprette prosjekt": ("create_project", 0.90),
    "opprette avdeling": ("create_department", 0.90),
    "opprette kontakt": ("create_contact", 0.90),
    "lag faktura": ("create_invoice", 0.88),
    "lag kunde": ("create_customer", 0.88),
    "lag ansatt": ("create_employee", 0.88),
    "lag leverandor": ("create_supplier", 0.88),
    "lag leverandør": ("create_supplier", 0.88),
    "lag produkt": ("create_product", 0.88),
    "lag prosjekt": ("create_project", 0.88),
    "lag avdeling": ("create_department", 0.88),
    "ny kunde": ("create_customer", 0.88),
    "ny faktura": ("create_invoice", 0.88),
    "ny ansatt": ("create_employee", 0.88),
    "ny leverandor": ("create_supplier", 0.88),
    "ny leverandør": ("create_supplier", 0.88),
    "ny produkt": ("create_product", 0.88),
    "nytt prosjekt": ("create_project", 0.88),
    "ny avdeling": ("create_department", 0.88),
    "registrer kunde": ("create_customer", 0.90),
    "registrer ansatt": ("create_employee", 0.90),
    "registrer leverandor": ("create_supplier", 0.90),
    "registrer leverandør": ("create_supplier", 0.90),
    "registrer produkt": ("create_product", 0.88),
    "new employee": ("create_employee", 0.88),
    "new customer": ("create_customer", 0.88),
    "new supplier": ("create_supplier", 0.88),
    "new product": ("create_product", 0.88),
    "new invoice": ("create_invoice", 0.88),
    "new project": ("create_project", 0.88),
    "new department": ("create_department", 0.88),
    "create employee": ("create_employee", 0.90),
    "create customer": ("create_customer", 0.90),
    "create supplier": ("create_supplier", 0.90),
    "create product": ("create_product", 0.90),
    "create invoice": ("create_invoice", 0.90),
    "create project": ("create_project", 0.90),
    "create department": ("create_department", 0.90),
    "register supplier": ("create_supplier", 0.88),
    "register customer": ("create_customer", 0.88),
    "register employee": ("create_employee", 0.88),
    "criar fornecedor": ("create_supplier", 0.90),
    "criar produto": ("create_product", 0.90),
    "criar projeto": ("create_project", 0.90),
    "aktiver modul": ("enable_modules", 0.90),
    "enable module": ("enable_modules", 0.90),
    "aktivere modul": ("enable_modules", 0.90),
    "innkjøpsordre": ("create_purchase_order", 0.92),
    "opprett innkjøpsordre": ("create_purchase_order", 0.95),
    "ny vare": ("create_product", 0.88),
    "legg inn vare": ("create_product", 0.90),
    "legg inn produkt": ("create_product", 0.90),
    "åpningsbalansen": ("create_opening_balance", 0.95),
    "føre opp": ("create_opening_balance", 0.85),
    "ajouter un employe": ("create_employee", 0.90),
    "ajouter un employé": ("create_employee", 0.90),
    "ajouter un client": ("create_customer", 0.90),
    "ajouter un fournisseur": ("create_supplier", 0.90),
    "ajouter un produit": ("create_product", 0.90),
    "ajouter une facture": ("create_invoice", 0.90),
    "ajouter un projet": ("create_project", 0.90),
    "opprett og send faktura": ("create_invoice_and_send", 0.95),
    "opprett og send ein faktura": ("create_invoice_and_send", 0.95),
    "create and send invoice": ("create_invoice_and_send", 0.92),
    "crea y envía": ("create_invoice_and_send", 0.92),
    "créez et envoyez": ("create_invoice_and_send", 0.92),
    "erstellen und senden": ("create_invoice_and_send", 0.92),
    "fastpris": ("fixed_price_project_invoice", 0.92),
    "prix forfaitaire": ("fixed_price_project_invoice", 0.92),
    "fixed price": ("fixed_price_project_invoice", 0.90),
    "festpreis": ("fixed_price_project_invoice", 0.90),
    "precio fijo": ("fixed_price_project_invoice", 0.90),
    "dimensjon": ("create_dimensions_voucher", 0.92),
    "dimensión contable": ("create_dimensions_voucher", 0.92),
    "regnskapsdimensjon": ("create_dimensions_voucher", 0.95),
    "dimensão contabilística": ("create_dimensions_voucher", 0.92),
    "dimensão contabilistica": ("create_dimensions_voucher", 0.92),
    "dimension comptable": ("create_dimensions_voucher", 0.92),
    "buchungsdimension": ("create_dimensions_voucher", 0.92),
    "accounting dimension": ("create_dimensions_voucher", 0.92),
    "custom dimension": ("create_dimensions_voucher", 0.90),
    "diett": ("create_travel_expense", 0.88),
    "per diem": ("create_travel_expense", 0.85),
    "dagsats": ("create_travel_expense", 0.85),
    "dieta": ("create_travel_expense", 0.85),
    "crie uma dimensão": ("create_dimensions_voucher", 0.95),
    "créez une dimension": ("create_dimensions_voucher", 0.95),
    "crea una dimensión": ("create_dimensions_voucher", 0.95),
    "create a dimension": ("create_dimensions_voucher", 0.95),
    "opprett dimensjon": ("create_dimensions_voucher", 0.95),
    "reise med diett": ("create_travel_expense", 0.92),
    "gastos de viaje": ("create_travel_expense", 0.90),
    "despesas de viagem": ("create_travel_expense", 0.90),
    "note de frais": ("create_travel_expense", 0.90),
    "crea una factura": ("create_invoice", 0.90),
    "créez une facture": ("create_invoice", 0.90),
    # PDF employee tasks
    "arbeitsvertrag": ("create_employee", 0.95),
    "angebotsschreiben": ("create_employee", 0.95),
    "arbeidskontrakt": ("create_employee", 0.95),
    "employment contract": ("create_employee", 0.95),
    "offer letter": ("create_employee", 0.95),
    "tilbudsbrev": ("create_employee", 0.95),
    "contrato de trabajo": ("create_employee", 0.95),
    "contrato de emprego": ("create_employee", 0.95),
    "contrat de travail": ("create_employee", 0.95),
    "erstellen sie den mitarbeiter": ("create_employee", 0.95),
    # PDF supplier invoice tasks
    "lieferantenrechnung erhalten": ("create_supplier_invoice", 0.95),
    "factura fournisseur": ("create_supplier_invoice", 0.95),
    "factura del proveedor recibida": ("create_supplier_invoice", 0.95),
    "fatura do fornecedor": ("create_supplier_invoice", 0.95),
    "supplier invoice received": ("create_supplier_invoice", 0.95),
    "leverandørfaktura mottatt": ("create_supplier_invoice", 0.95),
}.items(), key=lambda x: len(x[0]), reverse=True)


def _get_model(model_id: str, system_instruction: str) -> GenerativeModel:
    # All models now use europe-north1 (2.5 Pro/Flash)
    vertexai.init(project="ainm26osl-710", location="europe-north1")
    return GenerativeModel(model_id, system_instruction=system_instruction)


def get_tier(task_type: str) -> int:
    return TIER_MAP.get(task_type, 3)


def _parse_json(text: str) -> dict:
    """Parse LLM response, handling various markdown/fence formats."""
    text = text.strip()
    text = re.sub(r'```\w*\s*', '', text)
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}") + 1
    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end])
        except json.JSONDecodeError:
            pass

    if start >= 0 and end > start:
        candidate = text[start:end]
        candidate = re.sub(r',\s*([}\]])', r'\1', candidate)
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

        candidate2 = candidate.replace("'", '"')
        try:
            return json.loads(candidate2)
        except json.JSONDecodeError:
            pass

    if start >= 0:
        candidate = text[start:]
        open_braces = candidate.count("{") - candidate.count("}")
        open_brackets = candidate.count("[") - candidate.count("]")
        if open_braces > 0 or open_brackets > 0:
            truncated = candidate.rstrip()
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

    # Detect "payment on invoice NUMBER" -> register_payment_by_search
    _has_payment = bool(re.search(r'\b(betal|betaling|innbetaling|payment|paiement|zahlung|pago)\b', prompt_lower))
    _has_invoice_number = bool(re.search(
        r'(faktura\s*(nr|nummer|#)\s*\d+|invoice\s*(nr|number|#|no\.?)\s*\d+|factura\s*(nr|numero|#)\s*\d+|rechnung\s*(nr|nummer|#)\s*\d+)',
        prompt_lower,
    ))
    if _has_payment and _has_invoice_number:
        return "register_payment_by_search", 0.92

    # Detect timesheet patterns (must come before "prosjekt" match)
    if re.search(r'\d+[\.,]?\d*\s*timer\b', prompt_lower):
        if not re.search(r'\b(faktura|invoice|factura|rechnung|facture|ordre|order)\b', prompt_lower):
            return "create_timesheet_entry", 0.90

    # Delete verbs → delete_entity (unless travel expense which has its own type)
    if re.match(r'(slett|delete|eliminar|supprimer|löschen)\b', prompt_lower):
        if 'reiseregning' in prompt_lower or 'reiserekning' in prompt_lower or 'travel' in prompt_lower or 'note de frais' in prompt_lower or 'gasto de viaje' in prompt_lower:
            return "delete_travel_expense", 0.95
        return "delete_entity", 0.90

    # Combined: customer + invoice = invoice task (customer created as part of it)
    # But NOT if the prompt is primarily about a project (prosjekt/proyecto/projekt/projet/projeto)
    if re.search(r'\b(faktura|invoice|factura|rechnung|facture)\b', prompt_lower) and re.search(r'\b(kunde|customer|client|cliente)\b', prompt_lower):
        if not re.search(r'\b(prosjekt|proyecto|projekt|projet|projeto|project)\b', prompt_lower):
            if 'eksisterende' in prompt_lower or 'existing' in prompt_lower or 'existente' in prompt_lower or 'bestehenden' in prompt_lower:
                return "create_invoice_existing_customer", 0.88
            return "create_invoice", 0.88

    # Pre-sorted at module level (_HIGH_CONF_KEYWORDS)
    for phrase, (task_type, conf) in _HIGH_CONF_KEYWORDS:
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
        if best_len >= 8:
            return best_type, 0.70
        return best_type, 0.65
    return None


# ---------------------------------------------------------------------------
# Stage 1: Classification
# ---------------------------------------------------------------------------

async def classify_task(prompt: str) -> tuple[str, float]:
    """Classify with confidence. Keyword -> Flash-Lite -> Pro escalation."""
    quick = _quick_classify(prompt)
    if quick:
        task_type, confidence = quick
        if confidence >= 0.85:
            logger.info(f"Quick classify (high conf): {task_type} (conf={confidence:.2f})")
            return task_type, confidence
        elif confidence >= CONFIDENCE_THRESHOLD:
            logger.info(f"Quick classify (medium conf): {task_type} (conf={confidence:.2f}), validating with LLM")

    # Flash-Lite classification
    hint_prefix = ""
    if quick:
        task_type, confidence = quick
        if confidence >= CONFIDENCE_THRESHOLD:
            hint_prefix = f"[Keyword hint: {task_type} ({confidence:.0%})] "
    model = _get_model(MODEL_FLASH_LITE, CLASSIFIER_PROMPT)
    try:
        response = await asyncio.wait_for(
            model.generate_content_async(
                hint_prefix + prompt,
                generation_config={"temperature": 0.0, "max_output_tokens": 200},
            ),
            timeout=15.0,
        )
        result = _parse_json(response.text)
        task_type = result.get("task_type", "unknown")
        confidence = float(result.get("confidence", 0.5))
        logger.info(f"Flash-Lite classify: {task_type} (conf={confidence:.2f})")
    except asyncio.TimeoutError:
        logger.error("Flash-Lite classification timed out (15s)")
        task_type, confidence = "unknown", 0.0
    except Exception as e:
        logger.error(f"Flash-Lite classification failed: {e}")
        task_type, confidence = "unknown", 0.0

    # Escalate to Pro if Flash-Lite is very uncertain
    if confidence < 0.45:
        logger.info(f"Very low Flash-Lite confidence ({confidence:.2f}), escalating to Pro")
        pro_model = _get_model(MODEL_PRO, CLASSIFIER_PROMPT_PRO)
        try:
            response = await asyncio.wait_for(
                pro_model.generate_content_async(
                    prompt,
                    generation_config={"temperature": 0.0, "max_output_tokens": 300},
                ),
                timeout=20.0,
            )
            result = _parse_json(response.text)
            task_type = result.get("task_type", "unknown")
            confidence = float(result.get("confidence", 0.5))
            logger.info(f"Pro classify: {task_type} (conf={confidence:.2f})")
        except asyncio.TimeoutError:
            logger.error("Pro classification timed out (20s)")
        except Exception as e:
            logger.error(f"Pro classification failed: {e}")

    if confidence < 0.5:
        logger.warning(f"Very low confidence ({confidence:.2f}) -> unknown")
        task_type = "unknown"

    # Normalize LLM output: convert UPPER_CASE to lower_case
    if task_type != "unknown" and task_type == task_type.upper():
        task_type = task_type.lower()
        logger.info(f"Normalized uppercase task type to: {task_type}")

    if task_type in _FRESH_SANDBOX_REMAP:
        original = task_type
        task_type = _FRESH_SANDBOX_REMAP[original]
        logger.info(f"Fresh sandbox remap: {original} -> {task_type} (sandbox starts empty)")

    return task_type, confidence


# ---------------------------------------------------------------------------
# File handling
# ---------------------------------------------------------------------------

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


def _build_parts(prompt: str, files: list[dict] | None, prefix: str = "Extract values from") -> list[Part]:
    """Build Vertex AI Parts list from prompt text and optional file attachments."""
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
    task_text += f"{prefix} this accounting task:\n\n{prompt}"
    parts.append(Part.from_text(task_text))
    return parts


def _clean_extraction_result(parsed: dict) -> dict:
    """Strip meta keys from LLM extraction output, handle wrapped format."""
    if "extracted_values" in parsed and isinstance(parsed["extracted_values"], dict):
        values = parsed["extracted_values"]
    else:
        values = parsed
    for meta_key in ("task_type", "reasoning", "steps"):
        values.pop(meta_key, None)
    return values


def _rescue_missing_fields(prompt: str, values: dict) -> dict:
    """Regex-based rescue for fields the LLM failed to extract.
    Scans the prompt directly for common patterns and fills missing values."""
    prompt_lower = prompt.lower()

    # Organization number (9 digits)
    if not values.get("organizationNumber") and not values.get("customer_organizationNumber"):
        m = re.search(r'(?:org\.?\s*(?:nr|nº|no|nummer)?\.?\s*|organisasjonsnummer\s*|nº\s*org\.?\s*)(\d{9})', prompt_lower)
        if m:
            values["customer_organizationNumber"] = m.group(1)
            values["organizationNumber"] = m.group(1)

    # Email
    if not values.get("email") and not values.get("customer_email"):
        m = re.search(r'[\w.+-]+@[\w.-]+\.\w+', prompt)
        if m:
            email = m.group(0)
            values["customer_email"] = email
            values["email"] = email

    # Phone number
    if not values.get("phoneNumber") and not values.get("phoneNumberMobile"):
        m = re.search(r'(?:telefon|tlf|mobil|phone|téléphone|teléfono)\s*:?\s*\+?(\d[\d\s]{6,})', prompt_lower)
        if m:
            phone = re.sub(r'\s', '', m.group(1))
            values["phoneNumber"] = phone
            values["phoneNumberMobile"] = phone

    # Address
    if not values.get("addressLine1"):
        m = re.search(r'(?:adresse|address|dirección|adresse)\s*(?:er\s*|:\s*)?([A-ZÆØÅ][a-zæøåäö]+(?:gata|veien|gaten|vegen|gate|vei|veg|gade|straße|strasse|street|calle|rue)\s+\d+)', prompt)
        if m:
            values["addressLine1"] = m.group(1)

    # Postal code + city
    if not values.get("postalCode"):
        m = re.search(r'(\d{4})\s+([A-ZÆØÅ][a-zæøåäö]+(?:\s+[A-ZÆØÅ][a-zæøåäö]+)?)', prompt)
        if m:
            values["postalCode"] = m.group(1)
            values["city"] = m.group(2)

    # Date of birth
    if not values.get("dateOfBirth"):
        m = re.search(r'(?:født|born|nacido|né|geboren)\s+(?:den\s+)?(\d{1,2})[./\-]\s*(\d{1,2})[./\-]\s*(\d{4})', prompt_lower)
        if m:
            day, month, year = m.group(1), m.group(2), m.group(3)
            values["dateOfBirth"] = f"{year}-{month.zfill(2)}-{day.zfill(2)}"

    # invoiceDueDate (default to invoiceDate + 14 days)
    if not values.get("invoiceDueDate") and values.get("invoiceDate"):
        try:
            from datetime import datetime, timedelta
            inv_date = datetime.strptime(values["invoiceDate"], "%Y-%m-%d")
            values["invoiceDueDate"] = (inv_date + timedelta(days=14)).strftime("%Y-%m-%d")
        except (ValueError, TypeError):
            pass

    # Travel expense: rescue departureDate/returnDate from prompt
    if not values.get("departureDate"):
        # Look for date patterns like "15. mars", "15.03.2026", "2026-03-15", "March 15"
        m = re.search(r'(\d{4})-(\d{2})-(\d{2})', prompt)
        if m:
            values["departureDate"] = m.group(0)
        else:
            m = re.search(r'(\d{1,2})[./]\s*(\d{1,2})[./]\s*(\d{4})', prompt)
            if m:
                day, month, year = m.group(1), m.group(2), m.group(3)
                values["departureDate"] = f"{year}-{month.zfill(2)}-{day.zfill(2)}"

    # Travel expense: rescue destination
    if not values.get("destination"):
        m = re.search(r'(?:til|to|nach|à|a|hacia)\s+([A-ZÆØÅÄÖÜ][a-zæøåäöüß]+(?:\s+[A-ZÆØÅÄÖÜ][a-zæøåäöüß]+)?)', prompt)
        if m:
            values["destination"] = m.group(1)

    # Department number rescue
    if not values.get("departmentNumber"):
        m = re.search(r'(?:avdelingsnummer|dept\.?\s*(?:nr|no|num)?\.?|department\s*(?:number|no|nr))\s*:?\s*(\d+)', prompt_lower)
        if m:
            values["departmentNumber"] = m.group(1)

    # Credit note comment rescue — use the reason from the prompt
    if not values.get("comment"):
        if any(kw in prompt_lower for kw in ("reklamert", "reklamiert", "reklamasjon", "reklamation")):
            values["comment"] = "Reklamasjon"
        elif any(kw in prompt_lower for kw in ("kreditnota", "kreditert", "kreditering")):
            values["comment"] = "Kreditering"
        elif any(kw in prompt_lower for kw in ("gutschrift", "stornierung")):
            values["comment"] = "Gutschrift"
        elif any(kw in prompt_lower for kw in ("credit note", "credited")):
            values["comment"] = "Credit note"
        elif any(kw in prompt_lower for kw in ("nota de crédito", "nota de credito")):
            values["comment"] = "Nota de crédito"
        elif any(kw in prompt_lower for kw in ("note de crédit", "note de credit", "avoir")):
            values["comment"] = "Note de crédit"

    return values


# ---------------------------------------------------------------------------
# Stage 2: Value extraction
# ---------------------------------------------------------------------------

async def extract_values(prompt: str, task_type: str, files: list[dict] | None = None) -> dict:
    """Extract field values from prompt using LLM. No step generation.

    Uses Flash-Lite for tier 1, Pro for tier 2-3.
    """
    tier = get_tier(task_type)
    extraction_prompt = build_extraction_prompt(task_type, tier)
    model_id = MODEL_FLASH_LITE if tier <= 1 else MODEL_PRO
    model = _get_model(model_id, extraction_prompt)
    logger.info(f"Extracting values: {'Flash-Lite' if tier <= 1 else 'Pro'} for {task_type} (tier {tier})")

    parts = _build_parts(prompt, files, prefix="Extract values from")
    timeout = 90.0 if tier < 3 else 120.0

    try:
        response = await asyncio.wait_for(
            model.generate_content_async(
                parts,
                generation_config={"temperature": 0.0, "max_output_tokens": 4096},
            ),
            timeout=timeout,
        )
        try:
            raw_text = response.text
        except (ValueError, AttributeError):
            raw_text = ""

        try:
            parsed = _parse_json(raw_text)
            values = _clean_extraction_result(parsed)
            # Rescue any fields the LLM missed
            values = _rescue_missing_fields(prompt, values)
            return values
        except (json.JSONDecodeError, Exception) as e:
            logger.error(f"Failed to parse extraction JSON: {e}. Raw: {raw_text[:300]}")
            # Even on parse failure, try to extract from prompt directly
            return _rescue_missing_fields(prompt, {})

    except asyncio.TimeoutError:
        logger.error(f"Extraction LLM timed out ({timeout}s) for {task_type}")
        return _rescue_missing_fields(prompt, {})


# ---------------------------------------------------------------------------
# Stage 2b: Repair extraction
# ---------------------------------------------------------------------------

async def re_extract_values(
    prompt: str,
    task_type: str,
    errors: list[dict],
    files: list[dict] | None = None,
    original_values: dict | None = None,
) -> dict:
    """Re-extract values considering execution errors. Always uses Pro."""
    repair_prompt = build_repair_extraction_prompt(task_type, prompt, errors)
    model = _get_model(MODEL_PRO, repair_prompt)

    parts = _build_parts(prompt, files, prefix="Re-extract values from")
    timeout = 90.0 if get_tier(task_type) < 3 else 120.0

    try:
        response = await asyncio.wait_for(
            model.generate_content_async(
                parts,
                generation_config={"temperature": 0.0, "max_output_tokens": 4096},
            ),
            timeout=timeout,
        )
        try:
            raw_text = response.text
        except (ValueError, AttributeError):
            raw_text = ""

        try:
            parsed = _parse_json(raw_text)
            new_values = _clean_extraction_result(parsed)
            # Merge: new values override originals
            if original_values:
                merged = dict(original_values)
                merged.update(new_values)
                return merged
            return new_values
        except (json.JSONDecodeError, Exception) as e:
            logger.error(f"Failed to parse repair extraction JSON: {e}. Raw: {raw_text[:300]}")
            return original_values or {}

    except asyncio.TimeoutError:
        logger.error(f"Repair extraction timed out ({timeout}s) for {task_type}")
        return original_values or {}


# ---------------------------------------------------------------------------
# Full LLM planning (unknown tasks only)
# ---------------------------------------------------------------------------

async def _create_plan_full_llm(
    prompt: str, task_type: str, tier: int, confidence: float, files: list[dict] | None = None
) -> dict:
    """Full LLM planning for unknown task types -- LLM generates steps + extracted_values."""
    planner_prompt = build_planner_prompt(task_type, tier)
    model = _get_model(MODEL_PRO, planner_prompt)

    parts = _build_parts(prompt, files, prefix="Complete")
    timeout = 120.0

    logger.info(f"Full LLM planning {task_type} (tier={tier}, timeout={timeout}s): {prompt[:80]}...")
    try:
        response = await asyncio.wait_for(
            model.generate_content_async(
                parts,
                generation_config={"temperature": 0.0, "max_output_tokens": 8192},
            ),
            timeout=timeout,
        )
    except asyncio.TimeoutError:
        logger.error(f"Planner LLM timed out ({timeout}s) for {task_type}")
        return {
            "task_type": task_type,
            "tier": tier,
            "reasoning": "Planner LLM timeout",
            "steps": [],
            "extracted_values": {},
            "classification_confidence": confidence,
        }

    try:
        raw_text = response.text
    except (ValueError, AttributeError):
        raw_text = ""

    try:
        plan = _parse_json(raw_text)
    except (json.JSONDecodeError, Exception) as e:
        logger.error(f"Failed to parse plan JSON: {e}. Raw: {raw_text[:300]}")
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
            logger.error("Plan JSON retry also failed")
            plan = {
                "task_type": task_type,
                "reasoning": "Fallback - LLM JSON parse failed",
                "steps": [],
                "extracted_values": {},
            }

    plan["task_type"] = plan.get("task_type", task_type)
    plan["tier"] = tier
    plan["classification_confidence"] = confidence
    plan.setdefault("extracted_values", {})
    return plan


# ---------------------------------------------------------------------------
# Main entry: create_plan
# ---------------------------------------------------------------------------

async def create_plan(prompt: str, files: list[dict] | None = None) -> dict:
    """Classify -> extract values -> build concrete plan via template engine.

    For known task types: LLM only extracts values, template_engine builds steps.
    For unknown task types: falls back to full LLM planning.
    """
    task_type, confidence = await classify_task(prompt)
    tier = get_tier(task_type)

    # Unknown or missing template -> full LLM planning
    if task_type == "unknown" or task_type not in TEMPLATES:
        logger.info(f"{'Unknown' if task_type == 'unknown' else 'No template for ' + task_type} — full LLM planning")
        plan = await _create_plan_full_llm(prompt, task_type, tier, confidence, files)
        logger.info(f"Plan: {plan['task_type']} with {len(plan.get('steps', []))} steps")
        return plan

    # Known task type: extract values then build from template
    extracted_values = await extract_values(prompt, task_type, files)

    plan = build_concrete_plan(task_type, extracted_values)
    plan["tier"] = tier
    plan["classification_confidence"] = confidence
    plan["reasoning"] = f"Template-driven: {task_type}"

    logger.info(
        f"Plan: {plan['task_type']} with {len(plan.get('steps', []))} steps, "
        f"extracted {len(plan.get('extracted_values', {}))} values: {list(plan.get('extracted_values', {}).keys())}"
    )
    return plan
