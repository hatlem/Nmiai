# tripletex/main.py
# Hybrid router: template engine for known tasks, tool agent for complex ones.
"""FastAPI agent — hybrid router with template engine + Gemini tool agent."""
import json
import os
import time
import logging
import urllib.request
from datetime import datetime, timezone
from collections import deque
from urllib.parse import urlparse

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from tool_agent import tool_agent_solve
from agent import create_plan
from executor import execute_plan
from tripletex_client import TripletexClient
from learning import record_error, record_success, record_result, get_lessons, get_proven_pattern, should_override_route

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
try:
    from dashboard_report import report_test, update_test, report_score
except ImportError:
    # Fallback stubs if dashboard_report is not available (e.g. in Cloud Run)
    def report_test(*a, **kw): return None
    def update_test(*a, **kw): return None
    def report_score(*a, **kw): return None

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Tripletex AI Agent")

API_KEY = os.environ.get("API_KEY", "")

ALLOWED_HOSTS = (
    "tx-proxy.ainm.no", "api.tripletex.dev", "api.tripletex.io",
    "tripletex.no", "tripletex.dev", "a.run.app",
)

# ── Router signals: if ALL words in a tuple match, route to tool agent ──
TOOL_AGENT_SIGNALS = [
    ("timer", "faktura"), ("timar", "faktura"),
    ("hours", "invoice"), ("horas", "fatura"), ("horas", "factura"),
    ("heures", "facture"), ("stunden", "rechnung"),
    ("lønn", "bonus"), ("salary", "bonus"), ("løn", "bonus"),
    ("salario", "bonus"), ("gehalt", "bonus"),
    ("grunnlønn",), ("grunnløn",),
]

# ── In-memory stats ──
STATS = {
    "started": datetime.now(timezone.utc).isoformat(),
    "total": 0, "success": 0, "failed": 0, "repairs": 0,
    "total_api_calls": 0, "total_errors": 0, "by_type": {},
    "last_proxy": "",
}
HISTORY: deque = deque(maxlen=100)

DASHBOARD_URL = os.environ.get("DASHBOARD_URL", "http://localhost:8090")
RESULTS_LOG = os.path.join(os.path.dirname(__file__), "results.jsonl")


def _report_task_result(task_type: str, tier: int, success: bool, elapsed: float,
                        error_detail: str | None = None):
    """POST task result to dashboard. Fails silently."""
    try:
        payload = {
            "task_type": task_type,
            "tier": max(tier, 1),
            "points_earned": 1 if success else 0,
            "points_max": 1,
            "time_seconds": round(elapsed, 2),
            "error": error_detail,
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{DASHBOARD_URL}/api/tripletex/task-result",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        urllib.request.urlopen(req, timeout=3)
    except Exception:
        pass


def _log_to_jsonl(entry: dict):
    """Append a result entry to results.jsonl for offline analysis."""
    try:
        with open(RESULTS_LOG, "a") as f:
            f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
    except Exception as e:
        logger.warning(f"Failed to write results.jsonl: {e}")


def _record(task_type: str, success: bool, elapsed: float, api_calls: int,
            errors: int, repairs: int, prompt: str, verified: bool | None = None,
            tier: int = 0, confidence: float = 0, extracted_keys: list | None = None,
            error_detail: str = "", call_log: list | None = None):
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

    entry = {
        "time": datetime.now(timezone.utc).isoformat(),
        "type": task_type, "ok": success, "verified": verified,
        "elapsed": round(elapsed, 1), "api_calls": api_calls,
        "errors": errors, "repairs": repairs, "prompt": prompt[:500],
        "tier": tier, "confidence": round(confidence, 2),
        "extracted_keys": extracted_keys or [],
        "error_detail": error_detail[:500],
        "call_log": call_log or [],
    }
    HISTORY.appendleft(entry)
    _log_to_jsonl(entry)


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


@app.get("/")
async def root():
    """Redirect to /stats — dashboard is at dashboard/ (port 8090)."""
    return {"status": "ok", "dashboard": "http://localhost:8090", "endpoints": ["/health", "/stats", "/history", "/solve"]}


async def _ensure_bank_account(client: TripletexClient):
    """Set bankAccountNumber on account 1920 if not already set.
    Competition sandboxes don't have this pre-configured, causing invoice 422."""
    try:
        # Try primary query format
        resp = await client.get("/ledger/account", params={"number": "1920", "fields": "id,bankAccountNumber,version"})
        if not resp.get("ok"):
            # Fallback: some proxies use numberFrom/numberTo instead of number
            logger.info("Pre-flight: GET /ledger/account?number=1920 failed, trying numberFrom/numberTo")
            resp = await client.get("/ledger/account", params={
                "numberFrom": "1920", "numberTo": "1920",
                "fields": "id,bankAccountNumber,version",
            })
        if not resp.get("ok"):
            logger.warning(f"Pre-flight: could not fetch account 1920: status={resp.get('status_code')}")
            return
        data = resp.get("data", {})
        values = data.get("values", [])
        if not values:
            inner = data.get("value", {})
            if isinstance(inner, dict):
                values = inner.get("values", [])
        if not values:
            logger.warning("Pre-flight: account 1920 not found, trying PUT /company fallback")
            try:
                co_resp = await client.get("/company/1", params={"fields": "id,version"})
                if co_resp.get("ok"):
                    co_data = co_resp.get("data", {})
                    co_val = co_data.get("value", co_data)
                    if isinstance(co_val, dict) and "id" in co_val:
                        await client.put(f"/company/{co_val['id']}", body={
                            "id": co_val["id"],
                            "version": co_val.get("version", 0),
                            "bankAccountNumber": "12345678903",
                        })
                        logger.info("Pre-flight: set bankAccountNumber via PUT /company")
            except Exception as e2:
                logger.warning(f"Pre-flight: PUT /company fallback failed: {e2}")
            return
        acct = values[0]
        if acct.get("bankAccountNumber"):
            logger.info(f"Pre-flight: account 1920 already has bankAccountNumber={acct['bankAccountNumber']}")
            return  # Already set
        logger.info("Pre-flight: setting bankAccountNumber on account 1920")
        put_resp = await client.put(
            f"/ledger/account/{acct['id']}",
            body={
                "id": acct["id"],
                "version": acct.get("version", 0),
                "bankAccountNumber": "12345678903",
            },
        )
        if not put_resp.get("ok"):
            logger.warning(f"Pre-flight: PUT account 1920 failed: {put_resp.get('status_code')} {put_resp.get('data', {})}")
            # Fallback: try via POST /bank
            try:
                bank_resp = await client.request("POST", "/bank", body={
                    "accountNumber": "86011117947",
                    "name": "Driftskonto",
                })
                if bank_resp.get("ok"):
                    logger.info("Pre-flight: registered bank account via POST /bank fallback")
                else:
                    logger.warning(f"Pre-flight: POST /bank also failed: {bank_resp.get('status_code')}")
            except Exception as e3:
                logger.warning(f"Pre-flight: POST /bank exception: {e3}")
        else:
            logger.info("Pre-flight: bankAccountNumber set successfully on account 1920")
    except Exception as e:
        logger.warning(f"Pre-flight bank account failed (non-fatal): {e}")


def _should_use_tool_agent(prompt: str) -> bool:
    """Keyword-based router. Returns True if the prompt needs the tool agent."""
    prompt_lower = prompt.lower()
    return any(
        all(word in prompt_lower for word in signal)
        for signal in TOOL_AGENT_SIGNALS
    )


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

    # Log base_url for debugging (don't reject — competition controls the URL)
    parsed = urlparse(base_url)
    if not any(
        parsed.hostname == h or (parsed.hostname and parsed.hostname.endswith(f".{h}"))
        for h in ALLOWED_HOSTS
    ):
        logger.warning(f"Unexpected base_url (proceeding anyway): {base_url}")

    files = body.get("files", [])
    STATS["last_proxy"] = urlparse(base_url).hostname or ""
    logger.info(f"Task [{urlparse(base_url).hostname}]: {prompt}")

    # Report test start to dashboard
    test_id = report_test("tripletex", prompt[:80], status="running")

    client = TripletexClient(base_url, session_token)

    try:
        # Pre-flight: ensure bank account exists (prevents invoice 422 errors)
        await _ensure_bank_account(client)

        # ── Router decision (keyword matching, no LLM call) ──
        use_tool_agent = _should_use_tool_agent(prompt)

        if use_tool_agent:
            # ── TOOL AGENT PATH ──
            logger.info("Router: TOOL AGENT path")
            task_type = "tool_agent"
            agent_deadline = start + 280  # 280s = 300 - 20s buffer
            success = await tool_agent_solve(prompt, files, client, agent_deadline)
            record_result("tool_agent", "tool_agent", success)

        else:
            # ── TEMPLATE PATH ──
            task_type = "template"
            try:
                plan = await create_plan(prompt, files)
                plan_task_type = plan.get("task_type", "unknown")
                task_type = plan_task_type
                logger.info(f"Router: TEMPLATE path -> {task_type}")

                # Check learning-based routing override
                route_override = should_override_route(task_type)
                if route_override == "tool_agent":
                    logger.info(f"Learning override: {task_type} -> tool_agent")
                    use_tool_agent = True
                    agent_deadline = start + 280
                    success = await tool_agent_solve(prompt, files, client, agent_deadline)
                    record_result(task_type, "tool_agent", success)
                else:
                    result = await execute_plan(plan, client, start)
                    success = result.get("success", False)

                    # Retry once if failed and we have time
                    if not success and (time.monotonic() - start) < 150:
                        logger.warning(f"Template path failed for {task_type}, retrying with re-extraction")
                        STATS["repairs"] += 1
                        plan2 = await create_plan(prompt, files)
                        result2 = await execute_plan(plan2, client, start, prior_results=result.get("results"))
                        success = result2.get("success", False)

                    # Record result for learning
                    record_result(task_type, "template", success)
                    if success:
                        record_success(prompt, client.call_log or [])
                    else:
                        # Record errors from failed API calls
                        for call in (client.call_log or []):
                            status = call.get("status", 0)
                            if status >= 400:
                                record_error(call.get("path", ""), call.get("response", ""), prompt)

            except Exception as tmpl_err:
                logger.error(f"Template path error: {tmpl_err}", exc_info=True)
                record_result(task_type, "template", False)
                record_error("template_crash", str(tmpl_err), prompt)
                # Fallback to tool agent if template path crashes and we have time
                remaining = 280 - (time.monotonic() - start)
                if remaining > 60:
                    logger.info(f"Router: TEMPLATE crashed, falling back to TOOL AGENT ({remaining:.0f}s left)")
                    task_type = f"template_fallback_tool_agent"
                    agent_deadline = start + 280
                    success = await tool_agent_solve(prompt, files, client, agent_deadline)
                    record_result(task_type, "tool_agent", success)
                else:
                    raise

        elapsed = time.monotonic() - start
        logger.info(
            f"Done in {elapsed:.1f}s | path={task_type} | "
            f"success={success} | "
            f"api_calls={client.call_count} | errors={client.error_count}"
        )
        _record(task_type, success, elapsed, client.call_count,
                client.error_count, 0, prompt, call_log=client.call_log)

        update_test(
            test_id,
            status="passed" if success else "failed",
            details=f"path={task_type} elapsed={elapsed:.1f}s calls={client.call_count}",
            metadata={"api_calls": client.call_count, "errors": client.error_count, "path": task_type},
        )

    except Exception as e:
        logger.error(f"Agent error: {e}", exc_info=True)
        elapsed = time.monotonic() - start
        _record(task_type, False, elapsed, client.call_count,
                client.error_count, 0, prompt, False,
                error_detail=str(e)[:300], call_log=client.call_log)
        update_test(test_id, status="failed", details=f"Error: {e}")
    finally:
        await client.close()

    return JSONResponse({"status": "completed"})
