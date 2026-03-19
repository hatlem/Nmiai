# tripletex/main.py
"""FastAPI agent with plan-execute-verify-repair loop."""
import time
import logging
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from agent import create_plan, self_repair
from executor import execute_plan
from verifier import verify_execution, format_verification_for_repair
from tripletex_client import TripletexClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Tripletex AI Agent")

MAX_REPAIR_ATTEMPTS = 3
REPAIR_DEADLINE_SECONDS = 260  # leave 40s buffer for verification + response

# Simple tasks where recovery cost > benefit
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

    logger.info(f"Task: {prompt[:120]}...")

    client = TripletexClient(base_url, session_token)

    try:
        # Phase 1: Plan (classify + fill template)
        plan = await create_plan(prompt, files)
        task_type = plan.get("task_type", "unknown")
        extracted_values = plan.get("extracted_values", {})
        logger.info(f"Plan: {task_type} ({len(plan.get('steps', []))} steps)")

        # Phase 2: Execute
        result = await execute_plan(plan, client, start)

        # Phase 3: Verify + Self-repair loop
        attempt = 0
        current_plan = plan
        current_result = result
        verification_errors = None

        while attempt < MAX_REPAIR_ATTEMPTS and time.monotonic() - start < REPAIR_DEADLINE_SECONDS:
            has_http_errors = not current_result["success"]

            # If execution succeeded, verify the result
            if not has_http_errors:
                try:
                    verify_result = await verify_execution(
                        task_type, current_plan, current_result["results"],
                        client, extracted_values,
                    )
                    if verify_result["verified"] or verify_result.get("skipped"):
                        logger.info(f"Verified OK (attempt {attempt})")
                        break

                    verification_errors = [
                        {"field": c["field"], "expected": c["expected"], "actual": c["actual"]}
                        for c in verify_result.get("checks", [])
                        if not c["passed"]
                    ]
                    logger.warning(
                        f"Verification failed: {verify_result['failed']} checks. "
                        f"Missing: {verify_result.get('missing_fields', [])}"
                    )
                except Exception as e:
                    logger.error(f"Verification exception: {e}")
                    break
            else:
                verification_errors = None

            # Skip recovery for simple tasks (unless verification caught issues)
            if task_type in SKIP_RECOVERY_TYPES and not verification_errors:
                logger.info(f"Skipping recovery for simple task {task_type}")
                break

            # Self-repair
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
                repaired_plan = await self_repair(
                    prompt, current_plan, current_result["results"],
                    current_result.get("failed", []),
                    verification_errors=verification_errors,
                )

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
        logger.info(
            f"Done in {elapsed:.1f}s | type={task_type} | "
            f"success={current_result['success']} | repairs={attempt} | "
            f"api_calls={client.call_count} | errors={client.error_count}"
        )

    except Exception as e:
        logger.error(f"Agent error: {e}", exc_info=True)
    finally:
        await client.close()

    return JSONResponse({"status": "completed"})
