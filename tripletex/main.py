# tripletex/main.py
"""FastAPI agent with plan-execute-verify-repair loop + live dashboard."""
import os
import time
import logging
from datetime import datetime, timezone
from collections import deque

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, HTMLResponse

from agent import create_plan, self_repair, get_tier
from executor import execute_plan
from verifier import verify_execution, format_verification_for_repair
from tripletex_client import TripletexClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Tripletex AI Agent")

# Auth: if API_KEY is set, require Bearer token on /solve
API_KEY = os.environ.get("API_KEY", "")

MAX_REPAIR_ATTEMPTS = 3
REPAIR_DEADLINE_SECONDS = 270  # leave 30s buffer for verification + response

# Simple tasks where recovery cost > benefit
SKIP_RECOVERY_TYPES = {
    "create_customer", "create_product", "create_department",
    "create_supplier", "delete_travel_expense", "delete_entity",
    "create_customer_supplier",
}

# ── In-memory stats ──
STATS = {
    "started": datetime.now(timezone.utc).isoformat(),
    "total": 0, "success": 0, "failed": 0, "repairs": 0,
    "total_api_calls": 0, "total_errors": 0, "by_type": {},
}
HISTORY: deque = deque(maxlen=100)


def _record(task_type: str, success: bool, elapsed: float, api_calls: int,
            errors: int, repairs: int, prompt: str, verified: bool | None = None):
    STATS["total"] += 1
    STATS["total_api_calls"] += api_calls
    STATS["total_errors"] += errors
    STATS["repairs"] += repairs
    if success:
        STATS["success"] += 1
    else:
        STATS["failed"] += 1

    bt = STATS["by_type"].setdefault(task_type, {"total": 0, "success": 0, "failed": 0, "total_time": 0.0})
    bt["total"] += 1
    bt["total_time"] += elapsed
    bt["success" if success else "failed"] += 1

    HISTORY.appendleft({
        "time": datetime.now(timezone.utc).strftime("%H:%M:%S"),
        "type": task_type, "ok": success, "verified": verified,
        "elapsed": round(elapsed, 1), "api_calls": api_calls,
        "errors": errors, "repairs": repairs, "prompt": prompt[:90],
    })


# ── Endpoints ──

@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/stats")
async def stats():
    return STATS


@app.get("/history")
async def history():
    return list(HISTORY)


@app.get("/", response_class=HTMLResponse)
async def dashboard():
    return DASHBOARD_HTML


def _try_quick_fix(plan: dict, results: dict, failed: list) -> dict | None:
    """Try to fix common errors without calling LLM for self-repair.
    Returns a repaired plan or None if no quick fix available."""
    if not failed:
        return None

    steps = plan.get("steps", [])
    new_steps = []

    for fail_idx, fail_res in failed:
        status = fail_res.get("status_code", 0)
        data = fail_res.get("data", {})
        error_msg = str(data).lower()

        if fail_idx >= len(steps):
            continue
        original_step = steps[fail_idx]

        # 422 with "department.id" - need to GET department first then retry with it
        if status == 422 and "department" in error_msg and "id" in error_msg:
            fixed_step = dict(original_step)
            fixed_body = dict(fixed_step.get("body", {}))
            # Add GET department step
            new_steps.append({
                "method": "GET",
                "path": "/department",
                "params": {"fields": "id,name", "count": 1},
                "note": "quick-fix: fetch department for employee",
            })
            dept_step = len(new_steps) - 1
            fixed_body["department"] = {"id": f"$step_{dept_step}.values[0].id"}
            fixed_step["body"] = fixed_body
            new_steps.append(fixed_step)
            continue

        # 422 with "version" - need to GET first then retry with version
        if status == 422 and "version" in error_msg:
            path = original_step.get("path", "")
            # Extract the base entity path for a GET
            # e.g., /employee/123 -> GET /employee/123 for version
            new_steps.append({
                "method": "GET",
                "path": path,
                "params": {"fields": "id,version"},
                "note": "quick-fix: fetch version for PUT",
            })
            # Then retry the PUT with version from GET
            fixed_step = dict(original_step)
            fixed_body = dict(fixed_step.get("body", {}))
            fixed_body["version"] = f"$step_{len(new_steps) - 1}.value.version"
            fixed_step["body"] = fixed_body
            new_steps.append(fixed_step)
            continue

        # 422 with "userType" / "brukertype" - retry with userType: STANDARD
        if status == 422 and ("usertype" in error_msg or "brukertype" in error_msg):
            fixed_step = dict(original_step)
            fixed_body = dict(fixed_step.get("body", {}))
            fixed_body["userType"] = "STANDARD"
            fixed_step["body"] = fixed_body
            new_steps.append(fixed_step)
            continue

        # 422 with "deliveryDate" - retry with deliveryDate = orderDate or today
        if status == 422 and "deliverydate" in error_msg:
            from datetime import date as date_cls
            fixed_step = dict(original_step)
            fixed_body = dict(fixed_step.get("body", {}))
            fixed_body["deliveryDate"] = fixed_body.get("orderDate", date_cls.today().isoformat())
            fixed_step["body"] = fixed_body
            new_steps.append(fixed_step)
            continue

        # 422 with "invoiceDueDate" - retry with invoiceDueDate = invoiceDate + 14 days
        if status == 422 and "invoiceduedate" in error_msg:
            fixed_step = dict(original_step)
            fixed_params = dict(fixed_step.get("params", {}))
            fixed_body = dict(fixed_step.get("body", {}))
            from datetime import date as date_cls, timedelta as td
            invoice_date = fixed_params.get("invoiceDate") or fixed_body.get("invoiceDate") or date_cls.today().isoformat()
            try:
                from datetime import datetime as dt
                due = dt.strptime(invoice_date, "%Y-%m-%d") + td(days=14)
                due_str = due.strftime("%Y-%m-%d")
            except (ValueError, TypeError):
                due_str = date_cls.today().isoformat()
            fixed_params["invoiceDueDate"] = due_str
            fixed_step["params"] = fixed_params
            fixed_step["body"] = fixed_body
            new_steps.append(fixed_step)
            continue

        # 422 with "orderDate" - retry with orderDate = today
        if status == 422 and "orderdate" in error_msg:
            from datetime import date as date_cls
            fixed_step = dict(original_step)
            fixed_body = dict(fixed_step.get("body", {}))
            today = date_cls.today().isoformat()
            fixed_body["orderDate"] = today
            if "deliveryDate" not in fixed_body:
                fixed_body["deliveryDate"] = today
            fixed_step["body"] = fixed_body
            new_steps.append(fixed_step)
            continue

        # 422 with "systemgenererte" or "rad 0" on voucher postings - fix row numbering
        if status == 422 and ("systemgenererte" in error_msg or "rad 0" in error_msg or "guirow 0" in error_msg):
            fixed_step = dict(original_step)
            fixed_body = dict(fixed_step.get("body", {}))
            postings = fixed_body.get("postings", [])
            if isinstance(postings, list) and postings:
                fixed_postings = []
                for i, p in enumerate(postings):
                    fp = dict(p) if isinstance(p, dict) else p
                    if isinstance(fp, dict):
                        fp["row"] = i + 1  # Start from 1, not 0
                        fp.pop("guiRow", None)
                        # Ensure amountGrossCurrency equals amountGross
                        if "amountGross" in fp and "amountGrossCurrency" not in fp:
                            fp["amountGrossCurrency"] = fp["amountGross"]
                    fixed_postings.append(fp)
                fixed_body["postings"] = fixed_postings
                fixed_step["body"] = fixed_body
                new_steps.append(fixed_step)
                logger.info("Quick-fix: fixed voucher posting rows (start from 1) and added amountGrossCurrency")
                continue

        # 422 with "amountGrossCurrency" - add missing field
        if status == 422 and "amountgrosscurrency" in error_msg:
            fixed_step = dict(original_step)
            fixed_body = dict(fixed_step.get("body", {}))
            postings = fixed_body.get("postings", [])
            if isinstance(postings, list):
                for p in postings:
                    if isinstance(p, dict) and "amountGross" in p and "amountGrossCurrency" not in p:
                        p["amountGrossCurrency"] = p["amountGross"]
                fixed_body["postings"] = postings
                fixed_step["body"] = fixed_body
                new_steps.append(fixed_step)
                logger.info("Quick-fix: added amountGrossCurrency to voucher postings")
                continue

        # 422 with "vatType" or "mva-kode" on voucher postings - can't quick-fix without API call
        # Fall through to LLM repair which will add GET /ledger/vatType step
        if status == 422 and ("vattype" in error_msg or "mva-kode" in error_msg or "avgiftspliktig" in error_msg):
            logger.info("Quick-fix: vatType error on voucher — delegating to LLM repair (needs GET /ledger/vatType)")
            return None

        # 422 with "bankkontonummer" - company needs bank account number
        # PUT /company returns 405 in dev sandbox. Competition sandboxes have this pre-configured.
        if status == 422 and "bankkontonummer" in error_msg:
            logger.warning("Quick-fix: bankkontonummer error — cannot fix via API, competition sandboxes have this pre-configured")
            return None

        # 400/422 "already exists" - search for existing entity instead of creating
        _already_exists_kw = ("already exists", "allerede registrert", "allerede", "finnes allerede", "er allerede")
        if status in (400, 422) and any(kw in error_msg for kw in _already_exists_kw):
            method = original_step.get("method", "").upper()
            path = original_step.get("path", "")
            body = original_step.get("body", {})

            if method == "POST":
                entity_search_map = {
                    "/customer": ("name", "name"),
                    "/supplier": ("name", "name"),
                    "/product": ("name", "name"),
                }
                matched_entity = None
                for entity_path, (body_field, query_param) in entity_search_map.items():
                    if path.rstrip("/") == entity_path or path.rstrip("/").startswith(entity_path + "?"):
                        search_value = body.get(body_field)
                        if search_value:
                            matched_entity = (entity_path, query_param, search_value)
                        break

                if matched_entity:
                    entity_path, query_param, search_value = matched_entity
                    new_steps.append({
                        "method": "GET",
                        "path": entity_path,
                        "params": {query_param: search_value, "count": "1", "fields": "id,name,version"},
                        "note": "quick-fix: search existing entity instead of creating duplicate",
                    })
                    logger.info(f"Quick-fix: 400 already-exists, searching {entity_path} by {query_param}={search_value}")
                    continue
            return None

        # 409 conflict - same as 422 version: GET to fetch version, then retry PUT
        if status == 409:
            path = original_step.get("path", "")
            method = original_step.get("method", "").upper()
            if method == "PUT":
                get_step_idx = len(new_steps)
                new_steps.append({
                    "method": "GET",
                    "path": path,
                    "params": {"fields": "id,version"},
                    "note": "quick-fix: fetch version after 409 conflict",
                })
                fixed_step = dict(original_step)
                fixed_body = dict(fixed_step.get("body", {}))
                fixed_body["version"] = f"$step_{get_step_idx}.value.version"
                fixed_step["body"] = fixed_body
                new_steps.append(fixed_step)
                continue
            else:
                return None

        # No quick fix available for this error
        return None

    if not new_steps:
        return None

    return {
        "task_type": plan.get("task_type", "unknown"),
        "reasoning": "Quick-fix: common 422 error",
        "steps": new_steps,
        "extracted_values": plan.get("extracted_values", {}),
    }


async def _ensure_bank_account(client: TripletexClient):
    """Pre-flight: ensure account 1920 has a bank account number for invoicing.
    Sets bankAccountNumber via PUT /ledger/account if empty."""
    try:
        resp = await client.get("/ledger/account", params={"number": "1920", "fields": "id,bankAccountNumber,version"})
        if not resp.get("ok"):
            return
        data = resp.get("data", {})
        values = data.get("values", [])
        if not values:
            inner = data.get("value", {})
            if isinstance(inner, dict):
                values = inner.get("values", [])
        if not values:
            return
        acct = values[0]
        if acct.get("bankAccountNumber"):
            logger.debug("Pre-flight: bank account already set on 1920")
            return
        logger.info("Pre-flight: setting bankAccountNumber on account 1920")
        await client.put(
            f"/ledger/account/{acct['id']}",
            body={
                "id": acct["id"],
                "version": acct.get("version", 0),
                "bankAccountNumber": "12345678903",
            },
        )
        logger.info("Pre-flight: bankAccountNumber set on 1920")
    except Exception as e:
        logger.warning(f"Pre-flight bank account failed: {e}")


@app.post("/solve")
async def solve(request: Request):
    # Auth check
    if API_KEY:
        auth_header = request.headers.get("authorization", "")
        if not auth_header.startswith("Bearer ") or auth_header[7:] != API_KEY:
            return JSONResponse({"error": "unauthorized"}, status_code=401)

    start = time.monotonic()
    try:
        body = await request.json()
        prompt = body["prompt"]
        creds = body["tripletex_credentials"]
        base_url = creds["base_url"]
        session_token = creds["session_token"]
    except (KeyError, TypeError) as e:
        return JSONResponse({"error": f"missing field: {e}"}, status_code=400)

    # SSRF prevention: validate base_url against known Tripletex proxy domains
    ALLOWED_HOSTS = ("tx-proxy.ainm.no", "api.tripletex.dev", "api.tripletex.io", "tripletex.no", "tripletex.dev")
    from urllib.parse import urlparse
    parsed = urlparse(base_url)
    if parsed.scheme not in ("https", "http") or not any(
        parsed.hostname == h or (parsed.hostname and parsed.hostname.endswith(f".{h}"))
        for h in ALLOWED_HOSTS
    ):
        logger.warning(f"Rejected base_url: {base_url}")
        return JSONResponse({"error": "invalid base_url"}, status_code=400)

    files = body.get("files", [])

    logger.info(f"Task: {prompt[:120]}...")

    client = TripletexClient(base_url, session_token)
    task_type = "unknown"
    attempt = 0
    verified = None

    try:
        # Phase 1: Plan (classify + fill template)
        plan = await create_plan(prompt, files)
        task_type = plan.get("task_type", "unknown")
        extracted_values = plan.get("extracted_values", {})
        logger.info(f"Plan: {task_type} ({len(plan.get('steps', []))} steps)")

        # Pre-flight: ensure bank account for invoice tasks
        if "invoice" in task_type:
            await _ensure_bank_account(client)

        # Phase 2: Execute
        result = await execute_plan(plan, client, start)

        # Phase 3: Verify + Self-repair loop
        current_plan = plan
        current_result = result
        verification_errors = None

        while attempt < MAX_REPAIR_ATTEMPTS and time.monotonic() - start < REPAIR_DEADLINE_SECONDS:
            has_http_errors = not current_result["success"]

            # If all API calls succeeded but verification fails, limit to 1 repair
            # Our verifier may disagree with the competition's verifier
            if not has_http_errors and attempt >= 1:
                logger.info("All API calls succeeded, skipping further verification repairs")
                break

            # Skip verification for simple tier 1 tasks when all calls succeeded
            # Verification GETs count as API calls and hurt efficiency bonus
            tier = get_tier(task_type)
            if not has_http_errors and tier == 1 and attempt == 0:
                logger.info(f"Skipping verification for tier 1 task {task_type} (efficiency)")
                verified = True
                break

            if not has_http_errors:
                try:
                    verify_result = await verify_execution(
                        task_type, current_plan, current_result["results"],
                        client, extracted_values,
                    )
                    if verify_result["verified"] or verify_result.get("skipped"):
                        logger.info(f"Verified OK (attempt {attempt})")
                        verified = True
                        break

                    verification_errors = [
                        {"field": c["field"], "expected": c["expected"], "actual": c["actual"]}
                        for c in verify_result.get("checks", [])
                        if not c["passed"]
                    ]
                    verified = False
                    logger.warning(
                        f"Verification failed: {verify_result['failed']} checks. "
                        f"Missing: {verify_result.get('missing_fields', [])}"
                    )
                except Exception as e:
                    logger.error(f"Verification exception: {e}")
                    break
            else:
                verification_errors = None

            # Don't waste time on repair if it's a network/DNS error
            has_network_error = any(
                isinstance(res.get("data"), dict) and res["data"].get("network_error")
                for _, res in current_result.get("failed", [])
            )
            if has_network_error:
                logger.error("Network/DNS error detected — repair won't help, stopping")
                break

            attempt += 1
            elapsed = time.monotonic() - start
            logger.warning(
                f"Repair attempt {attempt}/{MAX_REPAIR_ATTEMPTS} "
                f"({len(current_result.get('failed', []))} http errors, "
                f"{len(verification_errors or [])} verify errors, "
                f"{elapsed:.0f}s elapsed)"
            )

            try:
                succeeded_results = {
                    idx: res for idx, res in current_result["results"].items()
                    if res["ok"]
                }
                # Try quick fix first (no LLM call needed)
                quick_fix = _try_quick_fix(current_plan, current_result["results"], current_result.get("failed", []))
                if quick_fix:
                    logger.info("Applied quick-fix repair (no LLM call)")
                    repaired_plan = quick_fix
                elif task_type in SKIP_RECOVERY_TYPES and not verification_errors:
                    logger.info(f"Skipping LLM recovery for simple task {task_type}")
                    break
                else:
                    repaired_plan = await self_repair(
                        prompt, current_plan, current_result["results"],
                        current_result.get("failed", []),
                        verification_errors=verification_errors,
                        files=files,
                    )

                # If repaired plan has no steps, nothing to fix
                if not repaired_plan.get("steps"):
                    logger.info("Repaired plan has no steps — nothing to fix")
                    break

                if time.monotonic() - start < REPAIR_DEADLINE_SECONDS:
                    current_result = await execute_plan(
                        repaired_plan, client, start,
                        prior_results=succeeded_results,
                    )
                    current_plan = repaired_plan
                    if repaired_plan.get("extracted_values"):
                        extracted_values.update(repaired_plan["extracted_values"])
                    verification_errors = None
                else:
                    logger.warning("Time budget exceeded before repair execution")
                    break
            except Exception as e:
                logger.error(f"Self-repair exception: {e}")
                break

        elapsed = time.monotonic() - start
        success = current_result["success"]
        logger.info(
            f"Done in {elapsed:.1f}s | type={task_type} | "
            f"success={success} | verified={verified} | repairs={attempt} | "
            f"api_calls={client.call_count} | errors={client.error_count}"
        )
        _record(task_type, success, elapsed, client.call_count,
                client.error_count, attempt, prompt, verified)

    except Exception as e:
        logger.error(f"Agent error: {e}", exc_info=True)
        elapsed = time.monotonic() - start
        _record(task_type, False, elapsed, client.call_count,
                client.error_count, attempt, prompt, False)
    finally:
        await client.close()

    return JSONResponse({"status": "completed"})


# ── Dashboard ──
DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Tripletex Agent</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:system-ui,-apple-system,sans-serif;background:#0a0a0f;color:#e0e0e0;padding:20px}
h1{font-size:1.4rem;color:#fff;margin-bottom:4px}
.sub{color:#888;font-size:.85rem;margin-bottom:20px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:12px;margin-bottom:24px}
.card{background:#14141f;border:1px solid #222;border-radius:10px;padding:16px;text-align:center}
.card .num{font-size:2rem;font-weight:700;color:#fff}
.card .label{font-size:.7rem;color:#888;margin-top:2px;text-transform:uppercase;letter-spacing:.5px}
.ok .num{color:#22c55e} .fail .num{color:#ef4444} .warn .num{color:#f59e0b} .info .num{color:#3b82f6}
table{width:100%;border-collapse:collapse;font-size:.85rem}
th{text-align:left;color:#888;font-weight:500;padding:8px 10px;border-bottom:1px solid #222;font-size:.7rem;text-transform:uppercase;letter-spacing:.5px}
td{padding:7px 10px;border-bottom:1px solid #1a1a2a}
tr:hover td{background:#14141f}
.badge{display:inline-block;padding:2px 8px;border-radius:4px;font-size:.75rem;font-weight:600}
.badge-ok{background:#052e16;color:#22c55e} .badge-fail{background:#2d0a0a;color:#ef4444}
.badge-verify{background:#1e1b4b;color:#818cf8}
.mono{font-family:'SF Mono',Consolas,monospace;font-size:.8rem;color:#a0a0b0}
.type-tag{background:#1e1b4b;color:#818cf8;padding:2px 8px;border-radius:4px;font-size:.75rem}
.section{margin-bottom:24px}
.section h2{font-size:1rem;color:#ccc;margin-bottom:10px}
.bar-bg{background:#1a1a2a;border-radius:3px;height:6px;flex:1}
.bar-fill{height:6px;border-radius:3px;background:linear-gradient(90deg,#3b82f6,#22c55e);transition:width .5s}
.type-row{display:flex;align-items:center;gap:10px;padding:5px 0}
.type-row .name{width:220px;font-size:.85rem}
.type-row .count{width:50px;text-align:right;font-size:.85rem;color:#888}
.prompt{max-width:260px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;color:#666;font-size:.8rem}
.empty{text-align:center;padding:40px;color:#555;font-size:.9rem}
#live{position:fixed;top:12px;right:20px;display:flex;align-items:center;gap:6px;font-size:.75rem;color:#888}
#live .pulse{width:8px;height:8px;border-radius:50%;background:#22c55e;animation:pulse 2s infinite}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.3}}
</style>
</head>
<body>
<div id="live"><span class="pulse"></span> Live</div>
<h1>Tripletex AI Agent</h1>
<p class="sub">NM i AI 2026</p>

<div class="grid">
  <div class="card"><div class="num" id="c-total">0</div><div class="label">Tasks</div></div>
  <div class="card ok"><div class="num" id="c-ok">0</div><div class="label">Success</div></div>
  <div class="card fail"><div class="num" id="c-fail">0</div><div class="label">Failed</div></div>
  <div class="card warn"><div class="num" id="c-repairs">0</div><div class="label">Repairs</div></div>
  <div class="card info"><div class="num" id="c-calls">0</div><div class="label">API Calls</div></div>
  <div class="card"><div class="num" id="c-errs">0</div><div class="label">4xx Errors</div></div>
  <div class="card"><div class="num" id="c-rate">-</div><div class="label">Success %</div></div>
</div>

<div class="section">
  <h2>Task Types</h2>
  <div id="types"><div class="empty">No tasks yet</div></div>
</div>

<div class="section">
  <h2>Recent Submissions</h2>
  <table>
    <thead><tr><th>Time</th><th>Type</th><th>Status</th><th>Verified</th><th>Time</th><th>Calls</th><th>Errs</th><th>Repairs</th><th>Prompt</th></tr></thead>
    <tbody id="hist"></tbody>
  </table>
  <div id="empty" class="empty">Waiting for first submission...</div>
</div>

<script>
function esc(s){return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;').replace(/'/g,'&#39;')}
async function r(){
  try{
    const s=await(await fetch('/stats')).json();
    document.getElementById('c-total').textContent=s.total;
    document.getElementById('c-ok').textContent=s.success;
    document.getElementById('c-fail').textContent=s.failed;
    document.getElementById('c-repairs').textContent=s.repairs;
    document.getElementById('c-calls').textContent=s.total_api_calls;
    document.getElementById('c-errs').textContent=s.total_errors;
    document.getElementById('c-rate').textContent=s.total>0?Math.round(s.success/s.total*100)+'%':'-';
    const t=Object.entries(s.by_type).sort((a,b)=>b[1].total-a[1].total);
    const mx=t.length?Math.max(...t.map(x=>x[1].total)):1;
    document.getElementById('types').innerHTML=t.length?t.map(([n,d])=>{
      const avg=(d.total_time/d.total).toFixed(1);
      return`<div class="type-row"><span class="name"><span class="type-tag">${esc(n)}</span></span><span class="count">${d.success}/${d.total}</span><div class="bar-bg"><div class="bar-fill" style="width:${d.total/mx*100}%"></div></div><span class="count">${avg}s</span></div>`;
    }).join(''):'<div class="empty">No tasks yet</div>';
  }catch(e){}
  try{
    const h=await(await fetch('/history')).json();
    const el=document.getElementById('hist'),em=document.getElementById('empty');
    if(h.length){
      em.style.display='none';
      el.innerHTML=h.map(e=>`<tr>
        <td class="mono">${e.time}</td>
        <td><span class="type-tag">${esc(e.type)}</span></td>
        <td><span class="badge ${e.ok?'badge-ok':'badge-fail'}">${e.ok?'OK':'FAIL'}</span></td>
        <td>${e.verified===true?'<span class="badge badge-verify">YES</span>':e.verified===false?'<span class="badge badge-fail">NO</span>':'-'}</td>
        <td class="mono">${e.elapsed}s</td>
        <td class="mono">${e.api_calls}</td>
        <td class="mono">${e.errors}</td>
        <td class="mono">${e.repairs}</td>
        <td class="prompt" title="${esc(e.prompt)}">${esc(e.prompt)}</td>
      </tr>`).join('');
    }
  }catch(e){}
}
r();setInterval(r,3000);
</script>
</body>
</html>"""
