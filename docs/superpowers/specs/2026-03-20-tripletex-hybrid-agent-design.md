# Tripletex Hybrid Agent Design

## Problem

Pure tool agent (LLM with function calling) is unstable: last 11 competition submissions show 1 success, 10 failures. Average 80-245s with 8-31 API calls. Template engine was previously 100% reliable for known tasks (3-15s, 1-5 calls) but was removed because it couldn't handle complex multi-step tasks.

## Solution

Hybrid architecture with keyword-based router:

```
Prompt → Classifier → Router
                        ├── Template Path (83% of tasks)
                        │   Deterministic, 3-15s, 0 errors, max efficiency bonus
                        └── Tool Agent Path (17% of tasks)
                            Gemini 2.5 Pro function calling, 30-120s
```

## Router Logic

No LLM call. Pure keyword matching in `main.py`:

```python
TOOL_AGENT_SIGNALS = [
    # timesheet + project invoice
    ("timer", "faktura"), ("timar", "faktura"),
    ("hours", "invoice"), ("horas", "fatura"), ("horas", "factura"),
    ("heures", "facture"), ("stunden", "rechnung"),
    # salary + bonus
    ("lønn", "bonus"), ("salary", "bonus"), ("løn", "bonus"),
    ("salario", "bonus"), ("gehalt", "bonus"),
    ("grunnlønn",), ("grunnløn",),
    # reverse payment
    ("reverser", "betaling"), ("reverse", "payment"),
    ("stornieren", "zahlung"), ("stornieren", "zurückgebucht"),
    ("annulez", "paiement"), ("revierta", "pago"),
    ("returnert", "banken"), ("zurückgebucht",), ("retourné",), ("devuelto",),
]

use_tool_agent = any(
    all(word in prompt.lower() for word in signal)
    for signal in TOOL_AGENT_SIGNALS
)
```

Tested against 21 real competition prompts: 0 misroutes.

## Template Path

Restored from git commit `83089af`. Components:

- `templates.py` (1411 lines) — 40+ task type definitions with exact field names
- `template_engine.py` (727 lines) — fills placeholders, applies defaults, expands dynamic steps
- `agent.py` — classifier (keyword + Flash-Lite + Pro) and value extraction

Flow: classify → extract values (LLM) → template_engine builds concrete plan → executor runs it.

New templates to add:
- `create_invoice_and_send` — invoice + PUT /:send
- `fixed_price_project_invoice` — project(isFixedPrice) + order(% amount) + /:invoice
- `create_dimensions_voucher` — POST /dimensionName + /dimensionValue + voucher with dimension ref

## Tool Agent Path

Kept as-is (`tool_agent.py`, 806 lines). Used for:
- timesheet + project invoice (dynamic amount from logged hours)
- salary + bonus (complex salary API)
- reverse payment (create full entity chain then reverse)

## Error Handling

- Template path: retry once with re-extracted values. 150s deadline.
- Tool agent path: internal error handling (reads errors, adapts). 280s deadline.
- No crossover between paths.

## Expected Impact

Based on 29 real competition prompts:
- Pure tool agent: ~24 points
- Hybrid: ~90 points (3.7x improvement)
- Perfect hybrid: ~105 points (4.3x improvement)

## Files Changed

| File | Action |
|---|---|
| `main.py` | New router logic, restore template imports |
| `templates.py` | Restore from git + 3 new templates |
| `template_engine.py` | Restore from git |
| `agent.py` | Restore classifier + extract_values |
| `tool_agent.py` | Keep as-is |

## Bank Account Pre-flight

Both paths run `_ensure_bank_account(client)` before execution. Sets `bankAccountNumber` on ledger account 1920 via `PUT /ledger/account/{id}`.
