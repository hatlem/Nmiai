# Tripletex Win Improvements — Implementation Plan

> **For agentic workers:** Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Maximize competition score from 22.3 to 60+ by fixing the 3 biggest point-losers in 4 hours.

**Architecture:** Three targeted fixes to the hybrid template+tool agent system. No architectural changes — just fill gaps that cause 0/8 scores.

**Tech Stack:** Python, FastAPI, Gemini 2.5 Pro, Tripletex API v2

**Priority order by expected point impact:**

---

### Task 1: Pre-create products in template path (HIGHEST IMPACT)

**Why:** Orders with product numbers like "Opplæring (7579)" score 0/8 because template path doesn't create products. This affects ~30% of invoice tasks.

**Files:**
- Modify: `tripletex/main.py` (already has `_create_products_from_plan` — verify it works)
- Modify: `tripletex/executor.py` (inject product.id into orderLines before execution)

- [ ] **Step 1: Verify _create_products_from_plan works**

Test locally with curl:
```bash
curl -s -X POST https://tripletex-agent-174612781810.europe-north1.run.app/solve \
  -H "Content-Type: application/json" \
  -d '{"prompt":"Opprett ordre for kunde Test AS (org.nr 999111222) med produkta Rådgivning (5001) til 15000 kr og Support (5002) til 8000 kr. Konverter til faktura og registrer full betaling.","tripletex_credentials":{"base_url":"https://kkpqfuj-amager.tripletex.dev/v2","session_token":"eyJ0b2tlbklkIjoyMTQ3NjM4ODA2LCJ0b2tlbiI6ImNlZGY2OWJjLWQ4NTEtNGM1ZS1hMmUxLTJjMDU1YjNmYzQzMCJ9"}}'
```

Check logs for "Pre-created product" messages.

- [ ] **Step 2: Fix if products not injected into orderLines**

The `_create_products_from_plan` function adds `product.id` to extracted_values orderLines, but the executor builds the order body from template steps using `{{orderLines}}` placeholder. The product.id must flow through.

In `executor.py` `_pre_validate_body`, add: if an orderLine has a `product` key with an `id`, keep it in the body.

- [ ] **Step 3: Deploy and test with competition submission**

```bash
gcloud run deploy tripletex-agent --source tripletex/ --region europe-north1 --allow-unauthenticated --memory 2Gi --timeout 300 --min-instances 1 --project ainm26osl-710
```

- [ ] **Step 4: Commit**

---

### Task 2: Entity pre-fetch cache (MEDIUM IMPACT)

**Why:** Every template task starts with GET /department, GET /employee, etc. These cost 1-3 API calls + latency per task. Pre-fetching saves calls and time = better efficiency bonus.

**Files:**
- Modify: `tripletex/main.py` (add pre-fetch before routing)
- Modify: `tripletex/tripletex_client.py` (add warm_cache method)

- [ ] **Step 1: Add warm_cache to TripletexClient**

```python
async def warm_cache(self):
    """Pre-fetch commonly needed entities to cache."""
    await asyncio.gather(
        self.request("GET", "/department", params={"fields": "id,name", "count": 1}),
        self.request("GET", "/employee", params={"fields": "id,firstName,lastName", "count": 1}),
        self.request("GET", "/invoice/paymentType", params={"fields": "id,description"}),
        self.request("GET", "/activity", params={"fields": "id,name,isProjectActivity"}),
    )
```

These endpoints are already cached by TripletexClient — calling them once populates the cache for all subsequent uses.

- [ ] **Step 2: Call warm_cache after _ensure_bank_account in main.py**

```python
await _ensure_bank_account(client)
await client.warm_cache()  # Pre-fetch common lookups
```

- [ ] **Step 3: Deploy and verify cache hits in logs**

Look for "Cache hit" debug messages in logs.

- [ ] **Step 4: Commit**

---

### Task 3: Payload validation before execution (MEDIUM IMPACT)

**Why:** LLM sometimes generates bodies missing required fields (userType, department.id, orderDate). Catching these before API call saves a turn + error.

**Files:**
- Modify: `tripletex/executor.py` (add validation in `_execute_step`)

- [ ] **Step 1: Add REQUIRED_FIELDS validation**

```python
REQUIRED_FIELDS = {
    "/employee": ["firstName", "lastName", "userType", "department"],
    "/customer": ["name", "isCustomer"],
    "/order": ["customer", "orderDate", "deliveryDate", "orderLines"],
    "/product": ["name"],
    "/department": ["name"],
    "/supplier": ["name"],
    "/project": ["name", "projectManager", "startDate"],
    "/travelExpense": ["employee", "travelDetails"],
    "/ledger/voucher": ["date", "postings"],
}
```

Before POST, check body has required fields. If missing, add sensible defaults (userType="STANDARD", isCustomer=true, dates=today).

- [ ] **Step 2: Deploy and verify fewer 422 errors**

- [ ] **Step 3: Commit**

---

### Task 4: Fix remaining template bugs (HIGH IMPACT)

**Why:** Some templates have wrong field names or missing steps that cause 0 scores.

**Files:**
- Modify: `tripletex/templates.py`

- [ ] **Step 1: Verify all templates produce valid API calls**

Run the template test script against sandbox for every task type. Fix any that fail.

Key known issues:
- `create_dimensions_voucher`: wrong endpoint paths (already fixed to /ledger/accountingDimensionName)
- `create_full_credit_note`: negative amounts in orderLines
- `fixed_price_project_invoice`: wrong entitlement template name (already fixed to ALL_PRIVILEGES)
- `reverse_payment`: voucher search dateFrom=dateTo issue

- [ ] **Step 2: Deploy and test**

- [ ] **Step 3: Commit**
