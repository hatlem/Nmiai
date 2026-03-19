import httpx
import logging

logger = logging.getLogger(__name__)


class TripletexClient:
    """Async HTTP client for Tripletex API v2."""

    def __init__(self, base_url: str, session_token: str):
        self.base_url = base_url.rstrip("/")
        self.auth = ("0", session_token)
        self._client = httpx.AsyncClient(
            auth=self.auth,
            timeout=30.0,
            headers={"Content-Type": "application/json"},
        )
        self.call_count = 0
        self.error_count = 0

    async def request(
        self, method: str, path: str, body: dict | None = None, params: dict | None = None
    ) -> dict:
        url = f"{self.base_url}{path}"
        logger.info(f"{method} {path}")
        self.call_count += 1

        response = await self._client.request(
            method=method, url=url, json=body, params=params
        )

        result = {
            "status_code": response.status_code,
            "ok": response.is_success,
        }

        try:
            result["data"] = response.json()
        except Exception:
            result["data"] = {"raw": response.text[:500]}

        if not response.is_success:
            self.error_count += 1
            logger.warning(f"{method} {path} -> {response.status_code}: {result['data']}")

        return result

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
