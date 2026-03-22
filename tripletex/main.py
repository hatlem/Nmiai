# tripletex/main.py
# Hybrid router: template engine for known tasks, tool agent for complex ones.
"""FastAPI agent — hybrid router with template engine + Gemini tool agent."""
import asyncio
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
from learning import record_error, record_result, compile_template, get_compiled_template, execute_compiled_template, should_override_route

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
    # Timesheet + invoice
    ("timer", "faktura"), ("timar", "faktura"),
    ("hours", "invoice"), ("horas", "fatura"), ("horas", "factura"),
    ("heures", "facture"), ("stunden", "rechnung"),
    # Salary
    ("lønn",), ("salary",), ("løn",), ("lön",),
    ("salario",), ("gehalt",), ("salaire",), ("salário",),
    ("grunnlønn",), ("grunnløn",), ("payroll",), ("gehaltsabrechnung",),
    ("lønnskjøring",), ("lønnskøyring",), ("kjør lønn",), ("run payroll",),
    # Bank reconciliation / CSV
    ("bankutskrift",), ("kontoutskrift",), ("bank statement",),
    ("extracto bancario",), ("extrait bancaire",), ("kontoauszug",),
    ("csv",), ("vedlagt csv",), ("attached csv",),
    # Supplier invoice — handled by template with fallback_on_500, NOT tool agent
    # ("leverandørfaktura",), ("leverandorfaktura",), ("supplier invoice",),
    # ("facture fournisseur",), ("factura del proveedor",), ("fatura do fornecedor",),
    # ("lieferantenrechnung",), ("eingangsrechnung",),
    # ("inngående faktura",), ("inngaende faktura",),
    # Complex multi-step (overdue + reminder/partial payment in all languages)
    ("vencida", "lembrete"), ("vencida", "parcial"),
    ("overdue",), ("forfalt",), ("überfällig",), ("vencida",), ("impayé",), ("en mora",),
    ("reminder fee",), ("purregebyr",), ("partial payment",), ("delbetaling",), ("delbetalinger",),
    ("Mahngebühr",), ("tasa de recordatorio",), ("frais de rappel",),
    ("inkasso",), ("forsinkelsesrente",), ("late fee",),
    ("teilzahlung",), ("paiement partiel",), ("pago parcial",), ("pagamento parcial",),
    # Overdue invoice + reminder combo (need full flow, not just reminder template)
    ("purring", "faktura"), ("purregebyr", "faktura"), ("reminder", "invoice"),
    ("purring", "forfalt"), ("reminder", "overdue"),
    ("tipo de cambio",), ("valutakurs",), ("exchange rate",), ("agio",),
    # Contract PDF
    ("arbeidskontrakt",), ("employment contract",), ("contrato de trabajo",),
    ("vedlagt pdf",), ("attached pdf",), ("voir pdf",), ("ver pdf",),
    # Receipt to voucher
    ("kvittering",), ("receipt",), ("recibo",), ("quittung",),
    # Ledger analysis
    ("analyser", "regnskap"), ("analyse", "konto"), ("analyze", "ledger"),
    ("analyser", "konto"), ("analysez", "compte"), ("analise", "razão"),
    # Year-end closing / depreciation / complex accounting
    ("årsoppgjør",), ("årsoppgjor",), ("årsavslutning",), ("year-end",), ("encerramento",), ("jahresabschluss",), ("cierre anual",),
    ("avskrivning",), ("depreciation",), ("depreciação",), ("abschreibung",), ("amortización",),
    ("skatteavsetning",), ("provisão fiscal",), ("tax provision",),
    # Project lifecycle (budget + hours + supplier + invoice)
    ("lifecycle",), ("livssyklus",), ("ciclo de vida",),
    ("budget", "hours", "invoice"), ("budsjett", "timer", "faktura"),
    # Complex voucher tasks (multiple postings, calculations)
    ("beregn",), ("calcule",), ("calculate",), ("berechne",),
    # Reverse payment (complex multi-step: create→invoice→pay→find voucher→reverse)
    ("reverser",), ("reverse",), ("stornieren",), ("zurückgebucht",), ("zuruckgebucht",),
    ("returnert",), ("returned",), ("retourné",), ("devuelto",), ("devolvido",),
    # Ledger errors / corrections
    ("feil", "hovedbok"), ("feil", "bilag"), ("errors", "ledger"), ("errors", "voucher"),
    ("erros", "livro"), ("errores", "libro"), ("fehler", "hauptbuch"), ("erreurs", "grand livre"),
    # Month-end closing
    ("månedsavslutning",), ("month-end",), ("monthly closing",), ("encerramento mensal",),
    ("monatsabschluss",), ("cierre mensual",), ("clôture mensuelle",),
    ("periodiser",), ("accrual",), ("periodisering",),
    # Reminder fee as standalone (not just with faktura)
    ("purregebyr",), ("reminder fee",), ("Mahngebühr",), ("forfallen",), ("forfalt",),
    # Currency / exchange rate
    ("eur ",), ("usd ",), ("gbp ",),
    ("exchange rate",), ("valutakurs",), ("wechselkurs",), ("taux de change",),
    ("tipo de cambio",), ("taxa de câmbio",),
    ("disagio",), ("agio",),
]


def _is_multi_entity_task(prompt: str) -> bool:
    """Detect tasks that create multiple top-level entities (e.g. '3 avdelinger').
    Only matches explicit number words (not digits in amounts) near plural entity nouns."""
    import re
    prompt_lower = prompt.lower()
    # Only number WORDS (not digits like "55 kr") to avoid false positives
    _NUMBERS = r'(?:two|three|four|five|to|tre|fire|fem|deux|trois|quatre|cinq|zwei|drei|vier|fünf|dos|tres|cuatro|cinco|dois|três)'
    # Only PLURAL entity nouns (not singular like "fournisseur")
    _ENTITIES = r'(?:departments|avdelinger|avdelingar|departamentos|abteilungen|départements|employees|ansatte|empleados|funcionários|mitarbeiter|employés|customers|kunder|clientes|clients|suppliers|leverandører|fornecedores|proveedores|lieferanten|fournisseurs|produkter|productos|produkte|produits|kontakter|contacts)'
    return bool(re.search(
        rf'\b{_NUMBERS}\b\s+(?:\w+\s+){{0,2}}{_ENTITIES}\b',
        prompt_lower,
    ))

async def _create_products_from_plan(plan: dict, client) -> None:
    """Pre-create products from orderLines that have productNumber before executing plan."""
    import re as _re
    extracted = plan.get("extracted_values", {})
    order_lines = extracted.get("orderLines", [])
    if not isinstance(order_lines, list):
        return

    for line in order_lines:
        if not isinstance(line, dict):
            continue
        prod_num = line.get("productNumber") or line.get("product_number")
        if not prod_num:
            continue

        name = line.get("description", line.get("name", f"Product {prod_num}"))
        price = line.get("unitPriceExcludingVatCurrency", 0)

        # Try to create product
        resp = await client.request("POST", "/product", body={
            "name": name,
            "number": str(prod_num),
            "priceExcludingVatCurrency": price,
        })

        if resp.get("ok"):
            prod_id = resp.get("data", {}).get("value", {}).get("id")
            if prod_id:
                line["product"] = {"id": prod_id}
                logger.info(f"Pre-created product {name} ({prod_num}) -> id={prod_id}")
        elif resp.get("status_code") == 422 and "i bruk" in str(resp.get("data", "")):
            # Product number already exists — find it
            search = await client.request("GET", "/product", params={"number": str(prod_num), "fields": "id,name"})
            if search.get("ok"):
                vals = search.get("data", {}).get("values", [])
                if vals:
                    line["product"] = {"id": vals[0]["id"]}
                    logger.info(f"Found existing product {prod_num} -> id={vals[0]['id']}")


def _has_product_numbers(prompt: str) -> bool:
    """Detect if prompt has product numbers in parentheses like 'Opplæring (7579)'."""
    import re
    return bool(re.search(r'\w+\s*\(\d{3,5}\)', prompt))

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
    """Set bankAccountNumber on account 1920 if not set."""
    try:
        resp = await client.get("/ledger/account", params={"number": "1920", "fields": "id,bankAccountNumber,version"})
        if not resp.get("ok"):
            return
        values = resp.get("data", {}).get("values", [])
        if not values:
            return
        acct = values[0]
        if acct.get("bankAccountNumber"):
            return
        await client.put(f"/ledger/account/{acct['id']}", body={
            "id": acct["id"],
            "version": acct.get("version", 0),
            "bankAccountNumber": "12345678903",
        })
    except Exception:
        pass  # Non-fatal


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

    parsed = urlparse(base_url)
    if not any(
        parsed.hostname == h or (parsed.hostname and parsed.hostname.endswith(f".{h}"))
        for h in ALLOWED_HOSTS
    ):
        logger.warning(f"Unexpected base_url (proceeding anyway): {base_url}")

    files = body.get("files", [])
    STATS["last_proxy"] = urlparse(base_url).hostname or ""
    logger.info(f"Task [{urlparse(base_url).hostname}]: {prompt[:120]}...")

    test_id = report_test("tripletex", prompt[:80], status="running")
    client = TripletexClient(base_url, session_token)
    task_type = "unknown"
    success = False

    try:
        # ── Pre-flight: vatType resolution (GET = free) + bank account (1 PUT, only if needed) ──
        # GET calls are free. Bank account PUT costs 1 write but is needed for invoicing.
        await asyncio.gather(
            _ensure_bank_account(client),
            client.resolve_vat_types(),
        )

        # ══════════════════════════════════════════════════════════
        # TIER 0: Keyword-based routing to tool agent for complex tasks
        #   - Overdue, reminder, partial payment, supplier invoice, etc.
        #   - These are too complex for simple templates
        # ══════════════════════════════════════════════════════════

        force_tool_agent = _should_use_tool_agent(prompt) or _is_multi_entity_task(prompt)
        # PDF/image tasks MUST go to tool agent — templates can't read files
        if files and len(files) > 0:
            force_tool_agent = True
            logger.info(f"Has {len(files)} file(s) → routing to tool agent (templates can't read files)")
        if force_tool_agent:
            logger.info(f"Keyword/multi-entity/file match → routing directly to tool agent")

        # ══════════════════════════════════════════════════════════
        # TIER 1: Template path (fast, 1-10s, no tool agent needed)
        #   - Classify + extract with LLM (1 call)
        #   - Build plan from template (no LLM)
        #   - Execute via DAG (parallel API calls)
        #   - If fails: tool agent fallback with FRESH client
        # ══════════════════════════════════════════════════════════

        if force_tool_agent:
            plan = {"task_type": "tool_agent", "steps": []}
        else:
            plan = await create_plan(prompt, files)
        task_type = plan.get("task_type", "unknown")
        has_template = task_type != "unknown" and task_type != "tool_agent" and len(plan.get("steps", [])) > 0
        logger.info(f"Classify: {task_type} (conf={plan.get('classification_confidence', 0):.2f}, steps={len(plan.get('steps', []))})")

        if has_template:
            # Pre-create products if needed
            await _create_products_from_plan(plan, client)

            # Execute template
            result = await execute_plan(plan, client, start)
            success = result.get("success", False)

            if success:
                logger.info(f"Template OK: {task_type} in {time.monotonic()-start:.1f}s")
            else:
                # ── Template failed → tool agent with FRESH client ──
                remaining = 290 - (time.monotonic() - start)
                if remaining > 60:
                    logger.warning(f"Template FAILED for {task_type}, handing off to tool agent ({remaining:.0f}s left)")
                    # Fresh client = clean state, no dirty API calls polluting context
                    agent_client = TripletexClient(base_url, session_token)
                    agent_client.vat_number_to_id = client.vat_number_to_id  # Share vatType map
                    try:
                        agent_deadline = start + 290
                        success = await tool_agent_solve(prompt, files, agent_client, agent_deadline)
                        task_type = f"{task_type}→agent"
                        # Merge stats
                        client.call_count += agent_client.call_count
                        client.error_count += agent_client.error_count
                    except Exception as agent_err:
                        logger.error(f"Tool agent fallback error: {agent_err}")
                    finally:
                        await agent_client.close()
                else:
                    logger.warning(f"Template FAILED, no time for agent ({remaining:.0f}s left)")

        else:
            # ══════════════════════════════════════════════════════
            # NO TEMPLATE: Go straight to tool agent
            # ══════════════════════════════════════════════════════
            logger.info(f"No template for {task_type} → tool agent")
            task_type = "tool_agent"
            agent_deadline = start + 290
            success = await tool_agent_solve(prompt, files, client, agent_deadline)

        elapsed = time.monotonic() - start
        logger.info(
            f"Done in {elapsed:.1f}s | path={task_type} | "
            f"success={success} | "
            f"api_calls={client.call_count} | errors={client.error_count}"
        )
        _record(task_type, success, elapsed, client.call_count,
                client.error_count, 0, prompt, call_log=getattr(client, 'call_log', None))

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
                error_detail=str(e)[:300], call_log=getattr(client, 'call_log', None))
        update_test(test_id, status="failed", details=f"Error: {e}")
    finally:
        await client.close()

    return JSONResponse({"status": "completed"})
