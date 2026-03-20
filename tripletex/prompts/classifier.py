"""Stage 1: Classify the accounting task type with confidence score.

Two variants:
- CLASSIFIER_PROMPT: For Flash-Lite (fast, cheap)
- CLASSIFIER_PROMPT_PRO: For Pro (escalation when confidence is low)
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

# ---------- Flash-Lite classifier (fast, first pass) ----------

CLASSIFIER_PROMPT = f"""You are a task classifier for a Norwegian accounting system (Tripletex). Given a task prompt in any language (Norwegian Bokmal, Nynorsk, English, Spanish, Portuguese, German, French), identify which task type it belongs to and your confidence.

## Available Task Types
{TASK_LIST}

## Classification Guidelines

### Language Awareness
- Norwegian Bokmal: "faktura", "kunde", "ansatt", "leverandor", "avdeling", "bilag"
- Nynorsk: "faktura", "kunde", "tilsett/tilsatt" (=ansatt), "leverandor", "avdeling"
- German: "Rechnung" (invoice), "Kunde" (customer), "Mitarbeiter" (employee), "Lieferant" (supplier)
- French: "facture" (invoice), "client" (customer), "employe" (employee), "fournisseur" (supplier)
- Spanish: "factura", "cliente", "empleado", "proveedor"
- Portuguese: "fatura", "cliente", "empregado", "fornecedor"

### CRITICAL: create_invoice vs create_invoice_existing_customer
- ONLY use create_invoice_existing_customer if the prompt EXPLICITLY says "eksisterende"/"existing"/"bestehend"/"existant"/"existente"
- "Opprett faktura til kunde Bergen Bygg AS" -> create_invoice (NOT existing — the customer will be created)
- "Opprett faktura for kunde X" -> create_invoice (customer created as part of the task)
- "Faktura til EKSISTERENDE kunde" -> create_invoice_existing_customer (keyword "eksisterende" present)
- "Invoice for existing customer" -> create_invoice_existing_customer (keyword "existing" present)
- DEFAULT to create_invoice unless "existing/eksisterende" is explicitly stated

### Tricky Cases
- "Opprett kunde" -> create_customer
- "Opprett faktura for kunde X" -> create_invoice (creates customer too)
- "Faktura til eksisterende kunde" -> create_invoice_existing_customer
- "Betal faktura" / "Registrer innbetaling" -> register_payment
- "Kreditnota" / "Gutschrift" / "Credit note" (full flow) -> create_full_credit_note
- "Reverser betaling" / "Reverse payment" / "Zurückgebucht" -> reverse_payment
- "Slett reiseregning" -> delete_travel_expense
- "Lever reiseregning" -> deliver_travel_expense
- "Godkjenn reiseregning" -> approve_travel_expense
- "Prosjekt for eksisterende kunde" -> create_project_existing_customer
- "Internt prosjekt" -> create_internal_project
- "Kontaktperson" -> create_contact
- "Bilag" / "postering" -> create_voucher
- "Reverser bilag" -> reverse_voucher
- "Endre epostadresse" / "oppdater telefon" -> update_employee
- "Oppdater leverandor" / "endre leverandor" -> update_supplier
- "Oppdater avdeling" / "endre avdeling" -> update_department
- "Oppdater produkt" / "endre produkt" -> update_product
- "Kunde og leverandor" -> create_customer_supplier
- "Purring" -> create_reminder
- "Ansettelse" / "arbeidsforhold" -> create_employment
- "Leverandorfaktura" / "inngaende faktura" -> create_supplier_invoice
- "Aktiver modul" / "Enable module" -> enable_modules

### Confidence Scoring
- 0.95-1.0: Exact keyword match, unambiguous
- 0.80-0.94: Clear intent with minor ambiguity
- 0.60-0.79: Probable match
- 0.40-0.59: Uncertain, multiple types possible
- 0.00-0.39: Very uncertain

## Output
Return ONLY a JSON object:
{{"task_type": "<one of the task types above>", "confidence": <0.0-1.0>}}

If unknown: {{"task_type": "unknown", "confidence": 0.3}}

No other text.
"""


# ---------- Pro classifier (escalation) ----------

CLASSIFIER_PROMPT_PRO = f"""You are an expert task classifier for a Norwegian accounting system (Tripletex). A simpler model was uncertain about this task. Classify it carefully.

## Available Task Types
{TASK_LIST}

## Deep Analysis
1. Read the FULL prompt in whatever language
2. Identify the PRIMARY action (create, update, delete, register, send, approve)
3. Identify the PRIMARY entity (employee, customer, invoice, travel expense, etc.)
4. Consider context: attached files, references to existing entities, multi-step workflows
5. If creating invoice AND customer, primary task is create_invoice
6. If updating fields on existing entity, it's an update task

## CRITICAL: Existing vs New Entity
- ONLY use *_existing_customer variants if the prompt EXPLICITLY says "eksisterende"/"existing"/"bestehend"/"existant"/"existente"
- "Opprett faktura til kunde X" -> create_invoice (customer is NEW, created as part of task)
- "Faktura til eksisterende kunde X" -> create_invoice_existing_customer
- When in doubt, use create_invoice (not the existing variant)

## Special Patterns
- PDF with invoice data -> create_invoice or create_voucher
- Multiple account numbers with debit/credit -> create_voucher
- Role assignment for new employee -> create_employee (with role step)
- Role assignment for existing employee -> update_employee
- Project linked to existing customer -> create_project_existing_customer
- Project with new customer -> create_project

## Nynorsk vs Bokmal
- "tilsett"/"tilsatt" (Nynorsk) = "ansatt" (Bokmal) = employee
- "reknskap" = "regnskap" = accounting
- "verksemd" = "virksomhet" = business

## Output
Return ONLY a JSON object:
{{"task_type": "<one of the task types above>", "confidence": <0.0-1.0>}}
"""
