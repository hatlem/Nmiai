# Tripletex AI Accounting Agent — Design Spec

## Overview

AI agent with HTTPS `/solve` endpoint that receives accounting tasks in natural language (7 languages), plans a sequence of Tripletex API calls, executes them, and returns `{"status": "completed"}`. Deployed on Cloud Run in `europe-north1`.

## Architecture: Plan-then-Execute

```
POST /solve
  │
  ├─ 1. PARSE (Gemini 3.1 Pro)
  │    Input: prompt + decoded files (multimodal) + API schema reference
  │    Output: structured JSON plan with task_type + ordered API calls
  │
  ├─ 2. EXECUTE (deterministic, no LLM)
  │    Run API calls sequentially, resolve $step_N.field dependencies
  │    On 4xx: programmatic fix + 1 retry
  │
  ├─ 3. RECOVER (Gemini 3.1 Pro, only on failure)
  │    Send error context to LLM for corrected plan
  │    Execute corrected steps
  │
  └─ Return {"status": "completed"}
```

### Why Plan-then-Execute over ReAct

- Scoring penalizes 4xx errors → trial-and-error is expensive
- Efficiency bonus rewards fewer total API calls → plan upfront
- 2 LLM calls max vs N calls for ReAct (one per tool invocation)
- At 95% per-step accuracy, 10-step ReAct = 60% overall success

## Model

- **Gemini 3.1 Pro** for all tasks via Vertex AI (`europe-north1`)
- Free via GCP account, low latency same region as validators
- Structured JSON output via `response_schema` parameter

## LLM Prompt Strategy

### System prompt contains

1. Role definition (English — strongest reasoning language)
2. Compact API reference: entity schemas + action endpoints (extracted from OpenAPI spec)
3. Norwegian accounting glossary (50+ terms → API concepts)
4. Task patterns with 1-2 few-shot examples per category
5. Rules: valid JSON only, use POST response IDs, never guess field values

### Output schema

```json
{
  "task_type": "string — e.g. create_employee, create_invoice",
  "reasoning": "string — brief explanation of what needs to be done",
  "steps": [
    {
      "method": "POST|GET|PUT|DELETE",
      "path": "string — e.g. /customer, /order/$step_0.id/:invoice",
      "body": "object — request body, optional",
      "params": "object — query parameters, optional"
    }
  ]
}
```

### Dependency resolution

`$step_N.field` tokens in path/body/params are replaced at runtime with values from step N's response. Common patterns:
- `$step_0.id` → ID of entity created in step 0
- `$step_0.value.id` → nested ID extraction

## Execution Engine

### Step execution

```
for each step in plan:
    1. resolve_refs(step, previous_results)
    2. execute API call
    3. if 4xx error:
        a. try programmatic fix (parse error message)
        b. retry once
        c. if still fails → add to failed_steps
    4. store result
```

### Programmatic error handling (no LLM)

| Error | Strategy |
|---|---|
| 422 validation | Parse error message → identify missing/invalid field → fill default |
| 404 not found | Check ID reference resolution |
| 409 duplicate | GET existing entity → use its ID |
| 401 auth | Re-check auth header format |

Max 1 retry per step. If programmatic fix fails → LLM recovery phase.

### LLM Recovery

Only triggered when programmatic fix fails. Sends to Gemini 3.1 Pro:
- Original prompt
- Plan so far (what succeeded, what failed)
- Error message
- Asks for corrected remaining steps

## File Handling

- Decode base64 from request `files` array
- Send as multimodal input to Gemini Pro (images, PDFs)
- Gemini extracts values (invoice amounts, names, dates) directly from documents
- Extracted values included in the structured plan

## Tripletex API Client

- `httpx.AsyncClient` for async HTTP calls
- Basic Auth: username `0`, password = `session_token`
- Base URL from request `tripletex_credentials.base_url`
- All calls use `?fields=*` on GET to see available fields when needed
- POST/PUT responses return created/updated entity with ID

## API Reference (embedded in system prompt)

Pre-processed from OpenAPI spec (3.7 MB → ~15 KB compact reference):

### Key entity schemas (writable fields only)

- Employee: firstName, lastName, email, dateOfBirth, phoneNumberMobile, userType, department
- Customer: name, organizationNumber, email, phoneNumber, isCustomer, deliveryAddress
- Product: name, number, description, priceExcludingVatCurrency, vatType, account
- Order: customer, orderDate, deliveryDate, orderLines, isPrioritizeAmountsIncludingVat
- OrderLine: product, description, count, unitPriceExcludingVatCurrency, vatType
- Invoice: invoiceDate, invoiceDueDate, customer, orders, comment
- TravelExpense: employee, travelDetails, costs, perDiemCompensations, mileageAllowances, isCompleted
- Project: name, projectManager, customer, isInternal, startDate, endDate, projectCategory
- Department: name, departmentNumber, departmentManager

### Key action endpoints

- `PUT /order/{id}/:invoice` — Convert order to invoice
- `PUT /invoice/{id}/:payment` — Register payment on invoice
- `PUT /invoice/{id}/:createCreditNote` — Create credit note
- `PUT /invoice/{id}/:send` — Send invoice to customer
- `PUT /travelExpense/:deliver` — Deliver travel expense
- `PUT /travelExpense/:approve` — Approve travel expense
- `PUT /employee/entitlement/:grantEntitlementsByTemplate` — Set employee role
- `PUT /ledger/voucher/{id}/:reverse` — Reverse voucher

### Key enums

- Employee.userType: STANDARD, EXTENDED, NO_ACCESS
- Entitlement templates: ALL_PRIVILEGES, INVOICING_MANAGER, PERSONELL_MANAGER, ACCOUNTANT, AUDITOR, DEPARTMENT_LEADER, NONE_PRIVILEGES

## Task Categories & API Flows

| Category | Example | API Flow |
|---|---|---|
| Create employee | "Opprett ansatt Ola Nordmann" | POST /employee → PUT /employee/entitlement/:grantEntitlementsByTemplate |
| Create customer | "Registrer kunde Acme AS" | POST /customer |
| Create product | "Opprett produkt Konsulenttime" | POST /product |
| Create invoice | "Fakturer kunde for tjenester" | POST /customer → POST /order (with orderLines) → PUT /order/{id}/:invoice |
| Register payment | "Registrer innbetaling" | GET /invoice → PUT /invoice/{id}/:payment |
| Credit note | "Krediter faktura" | GET /invoice → PUT /invoice/{id}/:createCreditNote |
| Travel expense | "Registrer reiseregning" | POST /travelExpense (with travelDetails, costs) |
| Delete travel expense | "Slett reiseregning" | GET /travelExpense → DELETE /travelExpense/{id} |
| Create project | "Opprett prosjekt for kunde" | POST /customer (if needed) → POST /project |
| Create department | "Opprett avdeling" | POST /department |
| Update contact info | "Oppdater telefon for ansatt" | GET /employee → PUT /employee/{id} |

## Project Structure

```
tripletex/
├── Dockerfile
├── requirements.txt
├── main.py                 # FastAPI app: /solve + /health
├── agent.py                # LLM planning (parse prompt → plan) + recovery
├── executor.py             # Deterministic API call execution engine
├── tripletex_client.py     # httpx wrapper: auth, error parsing, retries
├── prompts/
│   └── system.py           # System prompt: role, API ref, glossary, few-shots
├── schemas/
│   └── api_reference.json  # Compact API reference from OpenAPI spec
└── tests/
    └── test_agent.py
```

## Deployment

- **Platform:** Google Cloud Run
- **Region:** europe-north1 (same as validators)
- **Memory:** 1Gi
- **Timeout:** 300s
- **Min instances:** 1 (avoid cold start)
- **Image:** python:3.11-slim

```bash
gcloud run deploy tripletex-agent \
  --source . \
  --region europe-north1 \
  --allow-unauthenticated \
  --memory 1Gi \
  --timeout 300 \
  --min-instances 1
```

## Dependencies

```
fastapi
uvicorn[standard]
httpx
google-cloud-aiplatform
```

## Key Constraints

- Fresh sandbox per submission — no pre-existing data
- All API calls through proxy (base_url from request)
- Basic Auth: username `0`, password = session_token
- 5 min timeout per task
- 7 languages: nb, en, es, pt, nn, de, fr
- Efficiency bonus only with perfect correctness
- Every 4xx error reduces efficiency bonus
