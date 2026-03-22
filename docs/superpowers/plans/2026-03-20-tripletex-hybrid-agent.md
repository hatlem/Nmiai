# Tripletex Hybrid Agent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Combine deterministic templates (83% of tasks) with LLM tool agent (17%) via keyword router for maximum competition score.

**Architecture:** Keyword router in main.py decides path. Template path: classify → extract values → template_engine → executor. Tool agent path: Gemini 2.5 Pro with function calling. No crossover between paths.

**Tech Stack:** Python/FastAPI, Vertex AI (Gemini), Tripletex REST API

**Spec:** `docs/superpowers/specs/2026-03-20-tripletex-hybrid-agent-design.md`

---

### Task 1: Restore template system files from git

**Files:**
- Restore: `tripletex/templates.py` (1411 lines from git `83089af`)
- Restore: `tripletex/template_engine.py` (727 lines from git `83089af`)
- Restore: `tripletex/executor.py` (751 lines from git `83089af`)
- Restore: `tripletex/agent.py` (802 lines from git `83089af`)
- Restore: `tripletex/prompts/planner.py` (195 lines from git `83089af`)
- Restore: `tripletex/prompts/classifier.py` (122 lines from git `83089af`)
- Restore: `tripletex/prompts/__init__.py` (empty)

- [ ] **Step 1: Restore all 7 files from git**

```bash
cd /Users/andreashatlem/Projects/nmiai
git show 83089af:tripletex/templates.py > tripletex/templates.py
git show 83089af:tripletex/template_engine.py > tripletex/template_engine.py
git show 83089af:tripletex/executor.py > tripletex/executor.py
git show 83089af:tripletex/agent.py > tripletex/agent.py
git show 83089af:tripletex/prompts/planner.py > tripletex/prompts/planner.py
git show 83089af:tripletex/prompts/classifier.py > tripletex/prompts/classifier.py
touch tripletex/prompts/__init__.py
```

- [ ] **Step 2: Verify files restored correctly**

```bash
wc -l tripletex/templates.py tripletex/template_engine.py tripletex/executor.py tripletex/agent.py tripletex/prompts/planner.py tripletex/prompts/classifier.py
```
Expected: ~4008 total lines

- [ ] **Step 3: Commit**

```bash
git add tripletex/templates.py tripletex/template_engine.py tripletex/executor.py tripletex/agent.py tripletex/prompts/planner.py tripletex/prompts/classifier.py tripletex/prompts/__init__.py
git commit -m "feat(tripletex): restore template system from 83089af for hybrid architecture"
```

---

### Task 2: Rewrite main.py with hybrid router

**Files:**
- Modify: `tripletex/main.py`

The router uses keyword matching (no LLM call) to decide between template path and tool agent path.

- [ ] **Step 1: Rewrite main.py /solve endpoint**

Replace the current `/solve` endpoint with hybrid router. Keep existing: FastAPI app, stats, history, health endpoint, `_record()`, `_ensure_bank_account()`, dashboard reporting.

The new `/solve` logic:

```python
# After parsing request and creating client:

# 1. Router decision (no LLM, just keywords)
TOOL_AGENT_SIGNALS = [
    ("timer", "faktura"), ("timar", "faktura"),
    ("hours", "invoice"), ("horas", "fatura"), ("horas", "factura"),
    ("heures", "facture"), ("stunden", "rechnung"),
    ("lønn", "bonus"), ("salary", "bonus"), ("løn", "bonus"),
    ("salario", "bonus"), ("gehalt", "bonus"),
    ("grunnlønn",), ("grunnløn",),
    ("reverser", "betaling"), ("reverse", "payment"),
    ("stornieren", "zahlung"), ("stornieren", "zurückgebucht"),
    ("annulez", "paiement"), ("revierta", "pago"),
    ("returnert", "banken"), ("zurückgebucht",), ("retourné",), ("devuelto",),
]

prompt_lower = prompt.lower()
use_tool_agent = any(
    all(word in prompt_lower for word in signal)
    for signal in TOOL_AGENT_SIGNALS
)

# 2. Pre-flight: bank account for invoice tasks
await _ensure_bank_account(client)

if use_tool_agent:
    # Tool agent path
    logger.info(f"Router: TOOL AGENT path")
    task_type = "tool_agent"
    success = await tool_agent_solve(prompt, files, client, start + 280)
else:
    # Template path
    from agent import create_plan
    from executor import execute_plan
    plan = await create_plan(prompt, files)
    task_type = plan.get("task_type", "unknown")
    logger.info(f"Router: TEMPLATE path -> {task_type}")
    result = await execute_plan(plan, client, start)
    success = result["success"]

    # Retry once if failed
    if not success and time.monotonic() - start < 150:
        from agent import re_extract_values
        from template_engine import build_concrete_plan
        errors = [{"step": idx, "status_code": res.get("status_code", 0), "error": res.get("data", {})} for idx, res in result.get("failed", [])]
        new_values = await re_extract_values(prompt, task_type, errors, files, original_values=plan.get("extracted_values", {}))
        new_plan = build_concrete_plan(task_type, new_values)
        result = await execute_plan(new_plan, client, start)
        success = result["success"]
```

- [ ] **Step 2: Update imports**

Add at top of main.py:
```python
from tool_agent import tool_agent_solve
from tripletex_client import TripletexClient
```

Template imports happen inside the if-block (lazy) to avoid circular imports.

- [ ] **Step 3: Verify main.py is syntactically valid**

```bash
cd /Users/andreashatlem/Projects/nmiai/tripletex && python3.11 -c "import ast; ast.parse(open('main.py').read()); print('OK')"
```

- [ ] **Step 4: Commit**

```bash
git add tripletex/main.py
git commit -m "feat(tripletex): hybrid router — template path + tool agent fallback"
```

---

### Task 3: Add new templates for complex-but-templateable tasks

**Files:**
- Modify: `tripletex/templates.py`
- Modify: `tripletex/template_engine.py` (if conditional step handling needed)

Three new templates based on real competition prompts.

- [ ] **Step 1: Add `create_invoice_and_send` template**

In `templates.py`, add after `create_invoice`:
```python
"create_invoice_and_send": {
    "description": "Create an invoice and send it to the customer via email",
    "relevant_schemas": ["Customer", "Order", "OrderLine", "Invoice"],
    "extract_fields": ["customer_name", "customer_email", "customer_organizationNumber", "orderLines", "invoiceDate", "invoiceDueDate", "orderDate", "deliveryDate"],
    "optimal_calls": 4,
    "steps": [
        {"method": "POST", "path": "/customer", "body": {"name": "{{customer_name}}", "isCustomer": True, "email": "{{customer_email}}", "organizationNumber": "{{customer_organizationNumber}}"}},
        {"method": "POST", "path": "/order", "body": {"customer": {"id": "$step_0.id"}, "orderDate": "{{orderDate}}", "deliveryDate": "{{deliveryDate}}", "orderLines": "{{orderLines}}"}},
        {"method": "PUT", "path": "/order/$step_1.id/:invoice", "params": {"invoiceDate": "{{invoiceDate}}", "invoiceDueDate": "{{invoiceDueDate}}", "sendToCustomer": False}},
        {"method": "PUT", "path": "/invoice/$step_2.id/:send", "params": {"sendType": "EMAIL"}},
    ],
},
```

Add keyword hints:
```python
"create_invoice_and_send": ["opprett og send faktura", "send faktura", "opprett og send ein faktura", "create and send invoice", "crea y envía", "créez et envoyez"],
```

- [ ] **Step 2: Add `fixed_price_project_invoice` template**

```python
"fixed_price_project_invoice": {
    "description": "Create a fixed-price project and invoice a percentage of the fixed price",
    "relevant_schemas": ["Customer", "Employee", "Project", "Order", "Invoice"],
    "extract_fields": ["customer_name", "customer_organizationNumber", "project_name", "fixedprice", "invoice_percentage", "projectManager_firstName", "projectManager_lastName", "projectManager_email"],
    "optimal_calls": 7,
    "steps": [
        {"method": "POST", "path": "/customer", "body": {"name": "{{customer_name}}", "isCustomer": True, "organizationNumber": "{{customer_organizationNumber}}"}},
        {"method": "GET", "path": "/department", "params": {"fields": "id", "count": 1}},
        {"method": "POST", "path": "/employee", "body": {"firstName": "{{projectManager_firstName}}", "lastName": "{{projectManager_lastName}}", "email": "{{projectManager_email}}", "userType": "STANDARD", "department": {"id": "$step_1.values[0].id"}}},
        {"method": "PUT", "path": "/employee/entitlement/:grantEntitlementsByTemplate", "params": {"employeeId": "$step_2.id", "template": "ALL_PRIVILEGES"}},
        {"method": "POST", "path": "/project", "body": {"name": "{{project_name}}", "startDate": "{{today}}", "projectManager": {"id": "$step_2.id"}, "customer": {"id": "$step_0.id"}, "isFixedPrice": True, "fixedprice": "{{fixedprice}}"}},
        {"method": "POST", "path": "/order", "body": {"customer": {"id": "$step_0.id"}, "orderDate": "{{today}}", "deliveryDate": "{{today}}", "orderLines": [{"description": "{{project_name}} - delbetaling", "count": 1, "unitPriceExcludingVatCurrency": "{{invoice_amount}}"}]}},
        {"method": "PUT", "path": "/order/$step_5.id/:invoice", "params": {"invoiceDate": "{{today}}", "invoiceDueDate": "{{invoiceDueDate}}", "sendToCustomer": False}},
    ],
},
```

Add keyword hints:
```python
"fixed_price_project_invoice": ["fastpris", "prix forfaitaire", "fixed price", "festpreis", "precio fijo", "preço fixo"],
```

- [ ] **Step 3: Add `create_dimensions_voucher` template**

```python
"create_dimensions_voucher": {
    "description": "Create accounting dimension with values and post a voucher linked to a dimension value",
    "relevant_schemas": ["Voucher", "Posting", "Account"],
    "extract_fields": ["dimension_name", "dimension_values", "dimension_link_value", "account_number", "amount", "date", "description"],
    "optimal_calls": 5,
    "steps": [
        {"method": "POST", "path": "/ledger/accountingDimensionName", "body": {"dimensionName": "{{dimension_name}}"}},
        # dimension_values expansion handled by template_engine
    ],
},
```

Note: This template needs dynamic step expansion in `template_engine.py` for multiple dimension values + voucher posting. Add `_expand_dimension_steps()` function.

- [ ] **Step 4: Add classification keywords in agent.py**

In `_quick_classify` high_conf_keywords, add:
```python
"opprett og send faktura": ("create_invoice_and_send", 0.95),
"opprett og send ein faktura": ("create_invoice_and_send", 0.95),
"crea y envía": ("create_invoice_and_send", 0.92),
"créez et envoyez": ("create_invoice_and_send", 0.92),
"fastpris": ("fixed_price_project_invoice", 0.92),
"prix forfaitaire": ("fixed_price_project_invoice", 0.92),
"fixed price": ("fixed_price_project_invoice", 0.90),
"dimensjon": ("create_dimensions_voucher", 0.92),
"dimensión": ("create_dimensions_voucher", 0.90),
"dimension contable": ("create_dimensions_voucher", 0.92),
```

- [ ] **Step 5: Add `invoice_amount` computation in template_engine.py**

In `build_concrete_plan`, add for `fixed_price_project_invoice`:
```python
if task_type == "fixed_price_project_invoice":
    fp = values.get("fixedprice", 0)
    pct = values.get("invoice_percentage", 100)
    values["invoice_amount"] = fp * pct / 100
    values["today"] = date.today().isoformat()
    if "invoiceDueDate" not in values:
        from datetime import timedelta
        values["invoiceDueDate"] = (date.today() + timedelta(days=14)).isoformat()
```

- [ ] **Step 6: Commit**

```bash
git add tripletex/templates.py tripletex/template_engine.py tripletex/agent.py
git commit -m "feat(tripletex): 3 new templates — invoice+send, fixed_price_project, dimensions"
```

---

### Task 4: Deploy and smoke test

**Files:** None (deployment only)

- [ ] **Step 1: Deploy to Cloud Run**

```bash
cd /Users/andreashatlem/Projects/nmiai/tripletex
gcloud run deploy tripletex-agent --source . --region europe-north1 --allow-unauthenticated --memory 1Gi --timeout 300 --min-instances 1 --project ainm26osl-710
```

- [ ] **Step 2: Smoke test — template path (simple customer)**

```bash
curl -s --max-time 60 -X POST https://tripletex-agent-174612781810.europe-north1.run.app/solve \
  -H "Content-Type: application/json" \
  -d '{"prompt":"Opprett kunden Test AS med org.nr 999000111.","tripletex_credentials":{"base_url":"https://kkpqfuj-amager.tripletex.dev/v2","session_token":"TOKEN"}}'
```
Expected: `{"status":"completed"}`, logs show "Router: TEMPLATE path", Done in <15s, 0 errors

- [ ] **Step 3: Smoke test — tool agent path (salary+bonus)**

```bash
curl -s --max-time 180 -X POST https://tripletex-agent-174612781810.europe-north1.run.app/solve \
  -H "Content-Type: application/json" \
  -d '{"prompt":"Kjør lønn for Test Person. Grunnlønn 35000 kr. Legg til bonus 5000 kr.","tripletex_credentials":{"base_url":"https://kkpqfuj-amager.tripletex.dev/v2","session_token":"TOKEN"}}'
```
Expected: `{"status":"completed"}`, logs show "Router: TOOL AGENT path"

- [ ] **Step 4: Smoke test — new template (fixed price project)**

```bash
curl -s --max-time 120 -X POST https://tripletex-agent-174612781810.europe-north1.run.app/solve \
  -H "Content-Type: application/json" \
  -d '{"prompt":"Sett fastpris 300000 kr på prosjekt Test for kunde Test AS. Fakturer 50%.","tripletex_credentials":{"base_url":"https://kkpqfuj-amager.tripletex.dev/v2","session_token":"TOKEN"}}'
```
Expected: `{"status":"completed"}`, logs show "Router: TEMPLATE path -> fixed_price_project_invoice"

- [ ] **Step 5: Check logs for all 3 tests**

```bash
gcloud run services logs read tripletex-agent --region europe-north1 --project ainm26osl-710 --limit 20 | grep "Done in"
```
Expected: 3 results, template paths <15s, tool agent <120s

- [ ] **Step 6: Commit any fixes needed**
