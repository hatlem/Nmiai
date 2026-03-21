import asyncio
import httpx
import logging

logger = logging.getLogger(__name__)

MAX_RETRIES = 2
RETRY_BACKOFF = [0.5, 1.5]  # seconds between retries
RETRYABLE_STATUS = {429, 500, 502, 503, 504}


class TripletexClient:
    """Async HTTP client for Tripletex API v2 with retry and backoff."""

    def __init__(self, base_url: str, session_token: str):
        self.base_url = base_url.rstrip("/")
        self.auth = ("0", session_token)
        self._client = httpx.AsyncClient(
            auth=self.auth,
            timeout=httpx.Timeout(45.0, connect=10.0),
            headers={"Content-Type": "application/json"},
            limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
        )
        self.call_count = 0
        self.error_count = 0
        self._cache: dict[str, dict] = {}
        self.call_log: list[dict] = []  # Per-call details for failure tracking

    def _log_call(self, method: str, path: str, status: int, ok: bool,
                  error_snippet: str = "", body: dict | None = None,
                  response_data: dict | None = None, params: dict | None = None):
        """Record API call for failure tracking and template compilation."""
        entry = {
            "method": method,
            "path": path,
            "status": status,
            "ok": ok,
            "error": error_snippet[:200] if error_snippet else "",
        }
        if body is not None:
            entry["body"] = body
        if response_data is not None:
            entry["response"] = response_data
        if params is not None:
            entry["params"] = params
        self.call_log.append(entry)

    @staticmethod
    def _fix_body(body: dict | None, path: str) -> dict | None:
        """Auto-fix common LLM body mistakes before sending."""
        if body is None:
            return None
        # costCategory/category must be {"id": X}, not a string
        for field in ("costCategory", "category"):
            val = body.get(field)
            if isinstance(val, str):
                body.pop(field)  # Remove invalid string — API will use default
            elif isinstance(val, (int, float)):
                body[field] = {"id": int(val)}
        # paymentType must be {"id": X}
        val = body.get("paymentType")
        if isinstance(val, (int, float)):
            body["paymentType"] = {"id": int(val)}
        # vatType must be {"id": X}, not a bare number
        val = body.get("vatType")
        if isinstance(val, (int, float)):
            body["vatType"] = {"id": int(val)}
        elif isinstance(val, str) and val.isdigit():
            body["vatType"] = {"id": int(val)}
        # Fix vatType in nested orderLines
        for line in body.get("orderLines", []):
            if isinstance(line, dict):
                vt = line.get("vatType")
                if isinstance(vt, (int, float)):
                    line["vatType"] = {"id": int(vt)}
                elif isinstance(vt, str) and vt.isdigit():
                    line["vatType"] = {"id": int(vt)}
        # Fix vatType in nested postings
        for posting in body.get("postings", []):
            if isinstance(posting, dict):
                vt = posting.get("vatType")
                if isinstance(vt, (int, float)):
                    posting["vatType"] = {"id": int(vt)}
                elif isinstance(vt, str) and vt.isdigit():
                    posting["vatType"] = {"id": int(vt)}
        # Bug fix 1: Auto-convert postings amount fields from string to number
        if "postings" in body and isinstance(body["postings"], list):
            for posting in body["postings"]:
                if isinstance(posting, dict):
                    for field in ("amountGross", "amountGrossCurrency", "amount"):
                        if field in posting and isinstance(posting[field], str):
                            try:
                                posting[field] = float(posting[field])
                            except (ValueError, TypeError):
                                pass
        # Bug fix 2: Voucher description must not be null
        if "/ledger/voucher" in path and isinstance(body, dict):
            body.setdefault("description", "Bilag")
        # Bug fix 3: Ensure orderLines vatType wrapping handles all cases
        if "orderLines" in body and isinstance(body["orderLines"], list):
            for line in body["orderLines"]:
                if isinstance(line, dict):
                    vt = line.get("vatType")
                    # Handle plain dict without "id" key (e.g. {"number": 3})
                    if isinstance(vt, dict) and "id" not in vt:
                        line["vatType"] = {"id": 3}  # default 25% outgoing VAT
                    elif vt is None:
                        line["vatType"] = {"id": 3}  # default 25% outgoing VAT
        return body

    async def request(
        self, method: str, path: str, body: dict | None = None, params: dict | None = None
    ) -> dict:
        if body and method in ("POST", "PUT"):
            body = self._fix_body(body, path)
        # Bug fix 6: Auto-add dateTo if dateFrom is present but dateTo is missing on GET /ledger/voucher
        if "/ledger/voucher" in path and method == "GET" and params:
            if "dateFrom" in params and "dateTo" not in params:
                from datetime import datetime, timedelta
                try:
                    d = datetime.strptime(params["dateFrom"], "%Y-%m-%d")
                    params["dateTo"] = (d + timedelta(days=1)).strftime("%Y-%m-%d")
                except Exception:
                    params["dateTo"] = params["dateFrom"]
        # Fix dateFrom=dateTo on GET requests (dateTo must be > dateFrom)
        if params and method == "GET" and "dateFrom" in params and "dateTo" in params:
            if params["dateFrom"] == params["dateTo"]:
                # Add 1 day to dateTo
                try:
                    from datetime import datetime, timedelta
                    dt = datetime.strptime(params["dateTo"], "%Y-%m-%d")
                    params["dateTo"] = (dt + timedelta(days=1)).strftime("%Y-%m-%d")
                except (ValueError, TypeError):
                    pass
        # Bug fix: Auto-default invoiceDate/invoiceDueDate/sendToCustomer for /:invoice action
        if "/:invoice" in path and method == "PUT":
            if params is None:
                params = {}
            if "invoiceDate" not in params:
                from datetime import date
                params["invoiceDate"] = date.today().isoformat()
            if "invoiceDueDate" not in params:
                from datetime import date, timedelta
                inv = params.get("invoiceDate", date.today().isoformat())
                try:
                    from datetime import datetime as _dt
                    d = _dt.strptime(inv, "%Y-%m-%d")
                    params["invoiceDueDate"] = (d + timedelta(days=14)).strftime("%Y-%m-%d")
                except Exception:
                    params["invoiceDueDate"] = (date.today() + timedelta(days=14)).isoformat()
            if "sendToCustomer" not in params:
                params["sendToCustomer"] = "false"

        # Bug fix: Block POST /customer without name (extraction failure)
        if "/customer" in path and method == "POST" and body:
            if not body.get("name"):
                logger.warning("POST /customer without name — extraction likely failed, skipping call")
                self._log_call(method, path, 400, False, "name missing — blocked by client")
                return {"status_code": 400, "ok": False, "data": {"error": "Customer name is required but was not extracted from the task"}}

        url = f"{self.base_url}{path}"
        self.call_count += 1

        # Cache specific GET endpoints that return constant data within a session
        cache_key = None
        if method == "GET":
            # Normalize cache key from path + sorted params
            param_str = "&".join(f"{k}={v}" for k, v in sorted((params or {}).items()))
            candidate_key = f"{path}?{param_str}"
            # Only cache specific stable endpoints
            cacheable_paths = ("/invoice/paymentType", "/activity", "/salary/type", "/ledger/vatType")
            if any(path == cp for cp in cacheable_paths):
                cache_key = candidate_key
            elif path == "/employee" and params and params.get("count") in (1, "1") and "firstName" not in (params or {}):
                cache_key = candidate_key
            elif path == "/department" and params and params.get("count") in (1, "1") and "name" not in (params or {}):
                cache_key = candidate_key
            # Cache ledger account lookups by number (constant within a session)
            elif path == "/ledger/account" and params and "number" in params:
                cache_key = candidate_key
            # Cache bank lookups
            elif path == "/bank":
                cache_key = candidate_key
            # Cache company lookups by ID
            elif path.startswith("/company/") and not params:
                cache_key = candidate_key

            if cache_key and cache_key in self._cache:
                self.call_count -= 1  # Don't count cached responses
                logger.debug(f"Cache hit: {method} {path}")
                return self._cache[cache_key]

        for attempt in range(MAX_RETRIES + 1):
            try:
                response = await self._client.request(
                    method=method, url=url, json=body, params=params
                )
            except (httpx.ConnectError, httpx.ReadTimeout, httpx.WriteTimeout) as e:
                err_str = str(e)
                # DNS errors won't resolve with retry — fail fast
                if "Name or service not known" in err_str or "nodename nor servname" in err_str:
                    logger.error(f"{method} {path} DNS error (no retry): {e}")
                    self.error_count += 1
                    self._log_call(method, path, 0, False, err_str)
                    return {"status_code": 0, "ok": False, "data": {"error": err_str, "network_error": True}}
                if attempt < MAX_RETRIES:
                    wait = RETRY_BACKOFF[attempt]
                    logger.warning(f"{method} {path} network error, retry {attempt+1} in {wait}s: {e}")
                    await asyncio.sleep(wait)
                    continue
                self.error_count += 1
                self._log_call(method, path, 0, False, err_str)
                return {"status_code": 0, "ok": False, "data": {"error": err_str, "network_error": True}}

            result = {
                "status_code": response.status_code,
                "ok": response.is_success,
            }

            try:
                result["data"] = response.json()
            except Exception:
                result["data"] = {"raw": response.text[:500]}

            # Retry on rate-limit or server errors
            if response.status_code in RETRYABLE_STATUS and attempt < MAX_RETRIES:
                wait = RETRY_BACKOFF[attempt]
                if response.status_code == 429:
                    # Respect Retry-After header if present
                    retry_after = response.headers.get("Retry-After")
                    if retry_after:
                        try:
                            wait = min(float(retry_after), 5.0)
                        except (ValueError, TypeError):
                            pass  # keep default wait
                logger.warning(f"{method} {path} -> {response.status_code}, retry {attempt+1} in {wait}s")
                await asyncio.sleep(wait)
                continue

            if not response.is_success:
                self.error_count += 1
                logger.warning(f"{method} {path} -> {response.status_code}: {result['data']}")

            # Cache successful GET responses for stable endpoints
            if cache_key and result["ok"]:
                self._cache[cache_key] = result

            err_snip = ""
            if not result["ok"]:
                import json as _json
                err_snip = _json.dumps(result.get("data", {}), ensure_ascii=False, default=str)[:200]
            self._log_call(
                method, path, response.status_code, result["ok"], err_snip,
                body=body, response_data=result.get("data"), params=params,
            )
            return result

        # Should not reach here, but safety net
        self._log_call(method, path, 0, False, "max retries exhausted")
        return {"status_code": 0, "ok": False, "data": {"error": "max retries exhausted"}}

    async def get(self, path: str, params: dict | None = None) -> dict:
        if params is None:
            params = {"fields": "*"}
        elif "fields" not in params:
            params = {**params, "fields": "*"}
        return await self.request("GET", path, params=params)

    async def post(self, path: str, body: dict | None = None, params: dict | None = None) -> dict:
        return await self.request("POST", path, body=body, params=params)

    async def put(self, path: str, body: dict | None = None, params: dict | None = None) -> dict:
        return await self.request("PUT", path, body=body, params=params)

    async def delete(self, path: str, params: dict | None = None) -> dict:
        return await self.request("DELETE", path, params=params)

    async def warm_cache(self):
        """Pre-fetch commonly needed entities to populate cache."""
        await asyncio.gather(
            self.request("GET", "/department", params={"fields": "id,name", "count": "1"}),
            self.request("GET", "/employee", params={"fields": "id,firstName,lastName", "count": "1"}),
            self.request("GET", "/invoice/paymentType", params={"fields": "id,description"}),
            self.request("GET", "/activity", params={"fields": "id,name"}),
            self.request("GET", "/ledger/vatType", params={"fields": "id,name,number", "count": "5"}),
        )

    async def close(self):
        await self._client.aclose()
