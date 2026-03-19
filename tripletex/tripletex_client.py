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
        )
        self.call_count = 0
        self.error_count = 0
        self._cache: dict[str, dict] = {}

    async def request(
        self, method: str, path: str, body: dict | None = None, params: dict | None = None
    ) -> dict:
        url = f"{self.base_url}{path}"
        self.call_count += 1

        # Cache specific GET endpoints that return constant data within a session
        cache_key = None
        if method == "GET":
            # Normalize cache key from path + sorted params
            param_str = "&".join(f"{k}={v}" for k, v in sorted((params or {}).items()))
            candidate_key = f"{path}?{param_str}"
            # Only cache specific stable endpoints
            cacheable_paths = ("/invoice/paymentType", "/activity", "/salary/type")
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
                    self._dns_ok = False
                    return {"status_code": 0, "ok": False, "data": {"error": err_str, "network_error": True}}
                if attempt < MAX_RETRIES:
                    wait = RETRY_BACKOFF[attempt]
                    logger.warning(f"{method} {path} network error, retry {attempt+1} in {wait}s: {e}")
                    await asyncio.sleep(wait)
                    continue
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
                        wait = min(float(retry_after), 5.0)
                logger.warning(f"{method} {path} -> {response.status_code}, retry {attempt+1} in {wait}s")
                await asyncio.sleep(wait)
                continue

            if not response.is_success:
                self.error_count += 1
                logger.warning(f"{method} {path} -> {response.status_code}: {result['data']}")

            # Cache successful GET responses for stable endpoints
            if cache_key and result["ok"]:
                self._cache[cache_key] = result

            return result

        # Should not reach here, but safety net
        return {"status_code": 0, "ok": False, "data": {"error": "max retries exhausted"}}

    async def get(self, path: str, params: dict | None = None) -> dict:
        if params is None:
            params = {}
        if "fields" not in params:
            params["fields"] = "*"
        return await self.request("GET", path, params=params)

    async def post(self, path: str, body: dict | None = None, params: dict | None = None) -> dict:
        return await self.request("POST", path, body=body, params=params)

    async def put(self, path: str, body: dict | None = None, params: dict | None = None) -> dict:
        return await self.request("PUT", path, body=body, params=params)

    async def delete(self, path: str, params: dict | None = None) -> dict:
        return await self.request("DELETE", path, params=params)

    async def close(self):
        await self._client.aclose()
