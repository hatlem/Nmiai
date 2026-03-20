# tripletex/main.py
"""FastAPI agent with clean plan-execute-retry architecture."""
import os
import time
import logging
from datetime import datetime, timezone
from collections import deque
from urllib.parse import urlparse

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from agent import create_plan, classify_task, re_extract_values, get_tier
from executor import execute_plan
from template_engine import build_concrete_plan
from tripletex_client import TripletexClient

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

# ── In-memory stats ──
STATS = {
    "started": datetime.now(timezone.utc).isoformat(),
    "total": 0, "success": 0, "failed": 0, "repairs": 0,
    "total_api_calls": 0, "total_errors": 0, "by_type": {},
    "last_proxy": "",
}
HISTORY: deque = deque(maxlen=100)

DASHBOARD_URL = os.environ.get("DASHBOARD_URL", "http://localhost:8090")


def _report_task_result(task_type: str, tier: int, success: bool, elapsed: float,
                        error_detail: str | None = None):
    """POST task result to dashboard. Fails silently."""
    import json
    import urllib.request
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


def _record(task_type: str, success: bool, elapsed: float, api_calls: int,
            errors: int, repairs: int, prompt: str, verified: bool | None = None,
            tier: int = 0, confidence: float = 0, extracted_keys: list | None = None,
            error_detail: str = ""):
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

    from templates import TEMPLATES
    optimal = TEMPLATES.get(task_type, {}).get("optimal_calls", 0)
    efficiency = round(optimal / api_calls * 100) if api_calls > 0 and optimal > 0 else None

    HISTORY.appendleft({
        "time": datetime.now(timezone.utc).strftime("%H:%M:%S"),
        "type": task_type, "ok": success, "verified": verified,
        "elapsed": round(elapsed, 1), "api_calls": api_calls,
        "optimal_calls": optimal, "efficiency": efficiency,
        "errors": errors, "repairs": repairs, "prompt": prompt[:300],
        "tier": tier, "confidence": round(confidence, 2),
        "extracted_keys": extracted_keys or [],
        "error_detail": error_detail[:300],
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


@app.get("/")
async def root():
    """Redirect to /stats — dashboard is at dashboard/ (port 8090)."""
    return {"status": "ok", "dashboard": "http://localhost:8090", "endpoints": ["/health", "/stats", "/history", "/solve"]}


async def _ensure_bank_account(client: TripletexClient):
    """Set bankAccountNumber on account 1920 if not already set.
    Competition sandboxes don't have this pre-configured, causing invoice 422."""
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
            return  # Already set
        logger.info("Pre-flight: setting bankAccountNumber on account 1920")
        await client.put(
            f"/ledger/account/{acct['id']}",
            body={
                "id": acct["id"],
                "version": acct.get("version", 0),
                "bankAccountNumber": "12345678903",
            },
        )
    except Exception as e:
        logger.warning(f"Pre-flight bank account failed (non-fatal): {e}")


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
    task_type = "unknown"
    retried = False

    try:
        # 1. Classify + extract values + build plan from template
        plan = await create_plan(prompt, files)
        task_type = plan.get("task_type", task_type)
        logger.info(f"Plan: {task_type} ({len(plan.get('steps', []))} steps)")

        # 2. Pre-flight: ensure bank account for invoice tasks
        if "invoice" in task_type and "supplier" not in task_type:
            await _ensure_bank_account(client)

        # 3. Execute
        result = await execute_plan(plan, client, start)

        # 4. If failed and time permits, re-extract and retry ONCE
        if not result["success"] and time.monotonic() - start < 150:
            retried = True
            raw_errors = result.get("failed", [])
            errors = [{"step": idx, "status_code": res.get("status_code", 0), "error": res.get("data", {})} for idx, res in raw_errors]
            new_values = await re_extract_values(prompt, task_type, errors, files, original_values=plan.get("extracted_values", {}))
            new_plan = build_concrete_plan(task_type, new_values)
            result = await execute_plan(new_plan, client, start)

        elapsed = time.monotonic() - start
        success = result["success"]

        # Collect debug info
        tier = plan.get("tier", 0)
        confidence = plan.get("classification_confidence", 0)
        extracted_keys = list(plan.get("extracted_values", {}).keys())
        error_detail = ""
        if not success:
            for _, res in result.get("failed", []):
                detail = str(res.get("data", ""))[:150]
                if detail:
                    error_detail = detail
                    break

        logger.info(
            f"Done in {elapsed:.1f}s | type={task_type} | tier={tier} | "
            f"success={success} | retried={retried} | "
            f"api_calls={client.call_count} | errors={client.error_count} | "
            f"extracted={extracted_keys}"
        )
        _record(task_type, success, elapsed, client.call_count,
                client.error_count, int(retried), prompt,
                tier=tier, confidence=confidence,
                extracted_keys=extracted_keys, error_detail=error_detail)

        # Report result to dashboard
        update_test(
            test_id,
            status="passed" if success else "failed",
            details=f"type={task_type} elapsed={elapsed:.1f}s calls={client.call_count} retried={retried}",
            metadata={"task_type": task_type, "api_calls": client.call_count,
                       "errors": client.error_count, "retried": retried},
        )

        # Report task-level result to dashboard
        _report_task_result(
            task_type=task_type,
            tier=tier,
            success=success,
            elapsed=elapsed,
            error_detail=error_detail if not success else None,
        )

    except Exception as e:
        logger.error(f"Agent error: {e}", exc_info=True)
        elapsed = time.monotonic() - start
        _record(task_type, False, elapsed, client.call_count,
                client.error_count, int(retried), prompt, False,
                error_detail=str(e)[:300])
        update_test(test_id, status="failed", details=f"Error: {e}")
        _report_task_result(
            task_type=task_type,
            tier=0,
            success=False,
            elapsed=elapsed,
            error_detail=str(e)[:300],
        )
    finally:
        await client.close()

    return JSONResponse({"status": "completed"})

