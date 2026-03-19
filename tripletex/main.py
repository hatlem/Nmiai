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

MAX_REPAIR_ATTEMPTS = 3
REPAIR_DEADLINE_SECONDS = 280  # leave 20s buffer before 300s timeout

# Simple single-step task types where recovery cost > benefit
SKIP_RECOVERY_TYPES = {
    "create_customer", "create_product", "create_department",
    "create_supplier", "delete_travel_expense", "delete_entity",
    "create_customer_supplier",
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
        result = await execute_plan(plan, client, start)

        # Phase 3: Self-repair loop (up to MAX_REPAIR_ATTEMPTS)
        attempt = 0
        current_plan = plan
        current_result = result

        while (
            not current_result["success"]
            and task_type not in SKIP_RECOVERY_TYPES
            and attempt < MAX_REPAIR_ATTEMPTS
            and time.monotonic() - start < REPAIR_DEADLINE_SECONDS
        ):
            attempt += 1
            elapsed = time.monotonic() - start
            logger.warning(
                f"Repair attempt {attempt}/{MAX_REPAIR_ATTEMPTS} "
                f"({len(current_result['failed'])} failed, {elapsed:.0f}s elapsed)"
            )
            try:
                # Collect results from succeeded steps so LLM can skip them
                succeeded_results = {
                    idx: res for idx, res in current_result["results"].items()
                    if res["ok"]
                }
                repaired_plan = await self_repair(
                    prompt, current_plan, current_result["results"], current_result["failed"]
                )
                if time.monotonic() - start < REPAIR_DEADLINE_SECONDS:
                    # Pass succeeded results so $step_N refs from prior run still resolve
                    current_result = await execute_plan(
                        repaired_plan, client, start, prior_results=succeeded_results
                    )
                    current_plan = repaired_plan
                    if current_result["success"]:
                        logger.info(f"Self-repair succeeded on attempt {attempt}")
                        break
                else:
                    logger.warning("Time budget exceeded before repair execution")
                    break
            except Exception as e:
                logger.error(f"Self-repair exception: {e}")
                break

        if not current_result["success"] and attempt == 0:
            logger.warning(f"Skipping recovery for {task_type}")

        elapsed = time.monotonic() - start
        logger.info(
            f"Done in {elapsed:.1f}s | type={task_type} | "
            f"success={current_result['success']} | repairs={attempt} | "
            f"api_calls={len(current_result['results'])} | "
            f"errors={len(current_result['failed'])}"
        )

    except Exception as e:
        logger.error(f"Agent error: {e}", exc_info=True)
    finally:
        await client.close()

    return JSONResponse({"status": "completed"})
