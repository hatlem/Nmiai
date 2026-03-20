# tripletex/main.py
"""FastAPI agent with clean plan-execute-retry architecture + live dashboard."""
import os
import time
import logging
from datetime import datetime, timezone
from collections import deque
from urllib.parse import urlparse

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, HTMLResponse

from agent import create_plan, classify_task, re_extract_values, get_tier
from executor import execute_plan
from template_engine import build_concrete_plan
from tripletex_client import TripletexClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Tripletex AI Agent")

API_KEY = os.environ.get("API_KEY", "")

ALLOWED_HOSTS = (
    "tx-proxy.ainm.no", "api.tripletex.dev", "api.tripletex.io",
    "tripletex.no", "tripletex.dev",
)

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

    # SSRF prevention
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
    retried = False

    try:
        # 1. Classify
        task_type, confidence = await classify_task(prompt)

        # 2. Extract values + build plan from template
        plan = await create_plan(prompt, files)
        task_type = plan.get("task_type", task_type)
        logger.info(f"Plan: {task_type} ({len(plan.get('steps', []))} steps)")

        # 3. Execute
        result = await execute_plan(plan, client, start)

        # 4. If failed and time permits, re-extract and retry ONCE
        if not result["success"] and time.monotonic() - start < 240:
            retried = True
            errors = result.get("failed", [])
            new_values = await re_extract_values(prompt, task_type, errors, files)
            new_plan = build_concrete_plan(task_type, new_values)
            result = await execute_plan(new_plan, client, start)

        elapsed = time.monotonic() - start
        success = result["success"]
        logger.info(
            f"Done in {elapsed:.1f}s | type={task_type} | "
            f"success={success} | retried={retried} | "
            f"api_calls={client.call_count} | errors={client.error_count}"
        )
        _record(task_type, success, elapsed, client.call_count,
                client.error_count, int(retried), prompt)

    except Exception as e:
        logger.error(f"Agent error: {e}", exc_info=True)
        elapsed = time.monotonic() - start
        _record(task_type, False, elapsed, client.call_count,
                client.error_count, int(retried), prompt, False)
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
