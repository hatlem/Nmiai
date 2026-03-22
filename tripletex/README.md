# Tripletex AI Accounting Agent

An AI agent that receives natural-language accounting tasks (in 7 languages) and executes them against the Tripletex API. Built for the NM i AI 2026 competition.

## Architecture

The agent uses a **hybrid architecture**: a fast template engine for known task types, with an LLM-powered tool agent as fallback for complex or unknown tasks.

```
                         POST /solve
                              |
                     +--------v--------+
                     |   Keyword Router |
                     |  (main.py)       |
                     +--------+---------+
                              |
               +--------------+--------------+
               |                             |
     salary, overdue,              everything else
     CSV, PDF, multi-step
               |                             |
               v                    +--------v--------+
     +------------------+          |   Classifier     |
     |   Tool Agent     |          |   (agent.py)     |
     |   (Gemini + tools)|         +--------+---------+
     |   25 turns max   |                   |
     +------------------+          +--------v---------+
                                   |  Template Engine  |
                                   |  (template_engine)|
                                   +--------+---------+
                                            |
                                   +--------v---------+
                                   |   DAG Executor    |
                                   |   (executor.py)   |
                                   +------------------+
                                            |
                                     if fails, retry
                                     with Tool Agent
                                            |
                                   +--------v---------+
                                   | TripletexClient   |
                                   | (guardrails +     |
                                   |  auto-fix + retry)|
                                   +------------------+
                                            |
                                     Tripletex API v2
```

## Request Flow

### 1. Keyword Router (`main.py`)

The router inspects the prompt for keyword signals that indicate complex, multi-step tasks:

- **Salary/payroll** (`lonn`, `salary`, `gehalt`, `salaire`, ...)
- **Overdue invoices + reminders** (`forfalt`, `purregebyr`, `overdue`, ...)
- **Bank reconciliation / CSV** (`bankutskrift`, `csv`, `kontoutskrift`, ...)
- **File attachments** (PDFs, images -- templates cannot read files)
- **Multi-entity creation** ("three departments", "to ansatte")
- **Complex flows** (reverse payments, currency exchange, year-end closing)

If any signal matches, the task goes directly to the **Tool Agent**. Otherwise, it goes to the **Classifier + Template** path.

### 2a. Template Path (fast, 1-10s)

**Stage 1 -- Classify + Extract** (`agent.py`)

1. **Keyword pre-classification**: A sorted list of ~200 multilingual keyword patterns is checked first. If a high-confidence match is found (e.g. "opprett kunde" -> `create_customer` at 0.90 confidence), the LLM classifier is skipped entirely.

2. **LLM classifier** (Gemini 2.5 Flash): If keywords are inconclusive, the LLM classifies the task into one of ~35 task types and extracts all field values in a single call.

3. **Fresh sandbox remapping**: Each competition submission starts with an empty sandbox. Tasks like `register_payment` (which assume an existing invoice) are automatically remapped to `create_invoice_with_payment` (which creates the invoice first).

**Stage 2 -- Build Plan** (`template_engine.py`)

The template engine takes the task type + extracted values and produces a concrete execution plan. No LLM involvement -- the template is law.

- Fills `{{placeholder}}` values from extracted fields
- Cleans dates (supports `DD.MM.YYYY`, `DD/MM/YYYY`, ISO, and more)
- Normalizes amounts (Norwegian `1.234,56` format, currency suffixes like `kr`, `NOK`)
- Infers vatType from account numbers (Norwegian chart of accounts)
- Preserves `$step_N.field` references for runtime resolution

**Stage 3 -- Execute DAG** (`executor.py`)

The executor builds a dependency graph from `$step_N` references and executes independent steps in parallel via `asyncio.gather`.

- Topological sort into execution layers for maximum parallelism
- `$step_N.values[0].id` references resolved at runtime from previous step responses
- Unresolved placeholders are stripped (optional fields the prompt didn't mention)
- Conditional steps (`if_role`, `if_startDate`) execute only when relevant values were extracted
- 290-second hard deadline with per-step timeout

**Fallback**: If the template path fails, the task is handed off to the Tool Agent with a **fresh** `TripletexClient` (clean state, no dirty API calls polluting context from failed template execution).

### 2b. Tool Agent Path (complex, 10-60s)

`tool_agent.py` -- A Gemini model with function calling that can make arbitrary API calls. Used for tasks too complex for templates (multi-step workflows, file parsing, calculations).

**Tools available to the LLM:**

| Tool | Description |
|---|---|
| `tripletex_get` | GET request -- search/list entities. Free (no efficiency penalty). |
| `tripletex_post` | POST request -- create entities. |
| `tripletex_put` | PUT request -- update entities or trigger actions (`:invoice`, `:payment`, `:approve`). |
| `tripletex_delete` | DELETE request -- remove entities. |
| `get_api_guide(topic)` | Retrieve step-by-step API documentation for a topic. |
| `get_api_schema(entity)` | Retrieve exact field names from OpenAPI spec. |

**Key design: on-demand knowledge retrieval.** Instead of stuffing all API documentation into the system prompt (which would waste context and confuse the LLM), the agent calls `get_api_guide("invoice")` to retrieve step-by-step instructions for that specific topic. This keeps the context lean and focused.

The tool agent runs for up to 25 turns with a 290-second deadline (15s safety buffer before the competition's 300s limit).

## Template System (`templates.py`)

Templates define the exact API call sequence for each task type:

```python
"create_invoice": {
    "extract_fields": ["customer_name", "orderLines", "invoiceDate", ...],
    "steps": [
        {"method": "POST", "path": "/customer",
         "body": {"name": "{{customer_name}}", "isCustomer": True}},
        {"method": "POST", "path": "/order",
         "body": {"customer": {"id": "$step_0.id"}, "orderLines": "{{orderLines}}"}},
        {"method": "PUT",  "path": "/order/$step_1.id/:invoice",
         "params": {"invoiceDate": "{{invoiceDate}}"}},
    ],
}
```

**Placeholder types:**
- `{{field}}` -- filled from LLM-extracted values at plan-build time
- `$step_N.field` -- resolved at runtime from previous API responses
- `$step_N.values[0].id` -- array indexing into list responses

**Features:**
- `depends_on` -- explicit dependency declaration (auto-inferred from `$step_N` refs if omitted)
- `conditional_steps` -- executed only when relevant extracted values are present
- `skip_if_exists` -- skip step if a referenced value already exists
- `fallback_on_500` -- special error handling for flaky endpoints

Currently ~35 templates covering: employees, customers, suppliers, products, departments, contacts, invoices, orders, payments, credit notes, vouchers, travel expenses, projects, timesheets, supplier invoices, purchase orders, assets, employment, and more.

## Guardrails System (`tripletex_client.py`)

The `TripletexClient` wraps every API call with multiple layers of automatic error prevention and recovery.

### Auto-Fix (`_fix_body`)

Corrects common LLM mistakes before sending requests:

| Problem | Auto-fix |
|---|---|
| `nationalIdNumber` | Renamed to `nationalIdentityNumber` |
| `mobile`, `mobilePhone` | Renamed to `phoneNumberMobile` |
| `vatType: 25` (percentage) | Mapped to `{"id": 3}` (the vatType ID for 25%) |
| `vatType: 3` (bare number) | Wrapped to `{"id": 3}` |
| `product: 42` (bare ID) | Wrapped to `{"id": 42}` |
| Amount as string `"1500"` | Coerced to number `1500` |
| Top-level `addressLine1` | Moved into `postalAddress` object |
| Locked account with wrong vatType | Overridden to `{"id": 0}` based on account number ranges |
| Missing voucher `description` | Defaulted to `"Bilag"` |
| Missing voucher `date` | Defaulted to today |

### OpenAPI Field Validation (`_validate_fields`)

Uses a pre-extracted OpenAPI schema (`schemas/openapi_fields.json`) to strip fields that don't exist on each entity type. Prevents 422 errors from hallucinated field names.

### Auto-Retry

- Retries on status 429, 500, 502, 503, 504 with backoff (0.5s, 1.5s)
- Max 2 retries per request
- GET response caching for identical requests

### VAT Type Resolution

On startup, fetches all vatType records via `GET /ledger/vatType` and builds a number-to-ID mapping. Handles sandbox environments where vatType IDs may differ from production.

### Account-Locked vatType Inference

Norwegian chart of accounts rules enforced automatically:

- 1000-1999 (balance sheet): always vatType 0
- 2000-2999 (liabilities): always vatType 0
- 5000-5999 (payroll): always vatType 0
- 8000-8999 (finance): always vatType 0
- 3000-3999 (revenue): vatType 3 (25% outgoing VAT)
- 4000-4999 (COGS): vatType 1 (25% incoming VAT)

## API Guides

The tool agent has ~40 pre-written API guides embedded in `tool_agent.py` as the `API_GUIDES` dictionary. Each guide is a concise, step-by-step recipe with exact endpoints, field names, and common pitfalls.

**Standard guides:** `customer`, `employee`, `invoice`, `voucher`, `project`, `supplier`, `product`, `department`, `contact`, `payment`, `credit_note`, `reminder`, `timesheet`, `salary`, `employment`, `opening_balance`, `supplier_invoice`, `purchase_order`, `asset`, `bank_reconciliation`, `dimensions`, `fixed_price_project`, `update_entity`

**Complex workflow guides:**
- `currency_exchange` -- Foreign currency invoices with agio/disagio postings
- `month_end_closing` -- Salary accruals, prepaid expenses, depreciation vouchers
- `overdue_invoice_reminder` -- Create invoice, mark overdue, add reminder fee
- `bank_reconciliation_csv` -- Parse CSV, create bank statement, match transactions
- `project_lifecycle` -- Budget, hours, supplier costs, final invoicing
- `year_end_closing` -- Depreciation, tax provisions, equity transfers
- `ledger_correction` -- Reverse incorrect vouchers, post corrections
- `voucher_correction` -- Fix posting errors

All guides support multilingual aliases (Norwegian, German, French, Spanish, Portuguese). For example, `get_api_guide("faktura")` returns the same guide as `get_api_guide("invoice")`.

## Self-Learning (`learning.py`)

Three adaptive layers that improve performance over time within a session:

1. **Error Memory** -- Records past API errors with full context. Prevents repeating the same mistakes.
2. **Compiled Templates** -- When the tool agent succeeds on a task type that has no template, the exact API call sequence is recorded. Future identical tasks replay the sequence deterministically without any LLM calls.
3. **Adaptive Routing** -- Tracks success rates of template vs. tool agent paths per task type. Can override the default routing decision based on historical performance.

## Model Selection

| Model | Used For | Why |
|---|---|---|
| Gemini 2.5 Flash | Classification + field extraction | Fast (1-3s), sufficient for structured extraction |
| Gemini 2.5 Pro | Tool agent (complex tasks) | Better reasoning for multi-step workflows |

Gemini 3.1 models were tested but rejected due to high latency (~90s for extraction calls), leaving too little time for API execution within the 300s deadline.

## Pre-flight Setup

Before every task, the agent runs two pre-flight checks in parallel:

1. **Bank account** -- Ensures account 1920 has a `bankAccountNumber` set (required for invoicing)
2. **VAT type resolution** -- Fetches all vatType records and maps number -> database ID

## File Structure

```
tripletex/
  main.py                -- FastAPI router, /solve endpoint, keyword signals
  agent.py               -- Classifier + extractor (LLM stage)
  template_engine.py     -- Builds concrete plans from templates (no LLM)
  templates.py           -- ~35 task type definitions with API step sequences
  executor.py            -- DAG executor with parallel step execution
  tool_agent.py          -- Gemini function-calling agent + API guides
  tripletex_client.py    -- HTTP client with guardrails, auto-fix, retry
  learning.py            -- Self-learning: error memory, compiled templates
  prompts/
    classifier.py        -- LLM prompts for task classification
    planner.py           -- LLM prompts for field extraction
  schemas/
    openapi_fields.json  -- Valid fields per entity (from OpenAPI spec)
    field_reference.json -- Compact field reference for tool agent
    account_vat_map.json -- Account number -> vatType mapping
  docs/                  -- Competition task documentation
  Dockerfile             -- Cloud Run deployment image
  requirements.txt       -- Python dependencies
```

## Deployment

### Cloud Run

```bash
gcloud run deploy tripletex-agent \
  --source . \
  --region europe-north1 \
  --allow-unauthenticated \
  --memory 1Gi \
  --timeout 300 \
  --min-instances 1
```

The agent listens on port 8080 (`uvicorn main:app --host 0.0.0.0 --port 8080`). Cloud Run injects the `PORT` environment variable automatically.

### Endpoint

```
POST /solve
Content-Type: application/json

{
  "prompt": "Opprett en ny ansatt med navn Ola Nordmann...",
  "tripletex_credentials": {
    "base_url": "https://tx-proxy.ainm.no/...",
    "session_token": "..."
  },
  "files": []  // optional: base64-encoded PDFs/images
}
```

### Monitoring

- `GET /health` -- Health check
- `GET /stats` -- Aggregated success/failure counts, API call totals, per-type breakdown
- `GET /history` -- Last 100 task results with timing and error details

## Key Design Decisions

1. **Templates over pure LLM**: LLMs hallucinate field names and forget steps. Templates guarantee the correct API call sequence. The LLM only extracts field values -- a much simpler and more reliable task.

2. **On-demand API knowledge**: Instead of a massive system prompt with all API docs (~750 lines of guides), the tool agent retrieves documentation via `get_api_guide()` on demand. This keeps context focused and avoids confusion from irrelevant endpoints.

3. **Guardrails at the HTTP layer**: Rather than relying on prompt engineering to prevent mistakes, the client automatically fixes common errors (field renames, type wrapping, vatType inference). This catches errors that even good prompts cannot prevent.

4. **Fresh client on fallback**: When the template path fails and falls back to the tool agent, a new `TripletexClient` is created. This prevents the tool agent from being confused by error state from failed template API calls.

5. **Keyword routing before classification**: Certain task types (salary, bank reconciliation, overdue flows) are too complex for templates and too important to risk misclassification. Hard-coded keyword signals bypass the classifier entirely and go straight to the tool agent.

6. **Multilingual keyword coverage**: Tasks arrive in Norwegian (Bokmal + Nynorsk), English, German, French, Spanish, and Portuguese. Keyword lists, classifier prompts, and API guides include translations for all 7 languages.

7. **DAG-based parallel execution**: Independent API calls (e.g., GET department + GET employee) execute concurrently via `asyncio.gather`, reducing total latency. The dependency graph is auto-inferred from `$step_N` references.

## License

MIT
