# tripletex/main.py
import time
import logging
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from agent import create_plan, self_repair
from executor import execute_plan
from tripletex_client import TripletexClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Tripletex AI Agent")

# Time budget: 5 min total, reserve 60s for self-repair
MAX_PLAN_EXECUTE_SECONDS = 240
REPAIR_DEADLINE_SECONDS = 290  # leave 10s buffer before 300s timeout

# Simple task types where recovery cost > benefit
SKIP_RECOVERY_TYPES = {
    "create_customer", "create_product", "create_department",
    "create_supplier", "delete_travel_expense", "delete_entity",
}


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/solve")
async def solve(request: Request):
    start = time.monotonic()
    body = await request.json()
    prompt = body["prompt"]
    files = body.get("files", [])
    creds = body["tripletex_credentials"]

    base_url = creds["base_url"]
    session_token = creds["session_token"]

    logger.info(f"Task: {prompt[:100]}...")

    client = TripletexClient(base_url, session_token)

    try:
        # Phase 1: Plan (two-stage: classify + fill template)
        plan = await create_plan(prompt, files)
        task_type = plan.get("task_type", "unknown")
        logger.info(f"Plan: {task_type} ({len(plan.get('steps', []))} steps)")

        # Phase 2: Execute
        result = await execute_plan(plan, client)

        # Phase 3: Self-repair (conditional)
        elapsed = time.monotonic() - start
        should_repair = (
            not result["success"]
            and task_type not in SKIP_RECOVERY_TYPES
            and elapsed < MAX_PLAN_EXECUTE_SECONDS
        )

        if should_repair:
            logger.warning(f"{len(result['failed'])} steps failed, self-repairing... ({elapsed:.0f}s elapsed)")
            try:
                repaired_plan = await self_repair(
                    prompt, plan, result["results"], result["failed"]
                )
                if time.monotonic() - start < REPAIR_DEADLINE_SECONDS:
                    repair_result = await execute_plan(repaired_plan, client)
                    if repair_result["success"]:
                        logger.info("Self-repair succeeded")
                    else:
                        logger.error(f"Self-repair failed: {len(repair_result['failed'])} steps")
                else:
                    logger.warning("Skipping repair execution - time budget exceeded")
            except Exception as e:
                logger.error(f"Self-repair exception: {e}")
        elif not result["success"]:
            logger.warning(f"Skipping recovery for {task_type} (simple task or time exceeded)")
        else:
            logger.info(f"All steps succeeded in {elapsed:.1f}s")

    except Exception as e:
        logger.error(f"Agent error: {e}", exc_info=True)
    finally:
        await client.close()

    return JSONResponse({"status": "completed"})
