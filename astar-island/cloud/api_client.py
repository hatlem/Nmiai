"""Robust Astar Island API client with retry on 429. Never throws."""

import httpx
import time
import logging
from typing import Any, Optional

log = logging.getLogger(__name__)

NC = 6


class AstarClient:
    BASE = "https://api.ainm.no/astar-island"

    def __init__(self, token: str, timeout: float = 30.0):
        self.client = httpx.Client(
            headers={"Authorization": f"Bearer {token}"},
            timeout=timeout,
        )

    def _req(
        self,
        method: str,
        path: str,
        json: Optional[dict] = None,
        retries: int = 3,
    ) -> Optional[Any]:
        """Make API request with retry on 429 and error swallowing."""
        for attempt in range(retries):
            try:
                r = self.client.request(method, f"{self.BASE}{path}", json=json)
                if r.status_code == 429:
                    wait = 2 ** attempt
                    log.warning(f"429 on {method} {path}, retry in {wait}s")
                    time.sleep(wait)
                    continue
                r.raise_for_status()
                return r.json()
            except Exception as e:
                log.warning(f"{method} {path} attempt {attempt}: {e}")
                if attempt < retries - 1:
                    time.sleep(1)
        return None

    def get_rounds(self) -> list:
        return self._req("GET", "/rounds") or []

    def get_round(self, rid: str) -> Optional[dict]:
        return self._req("GET", f"/rounds/{rid}")

    def get_budget(self) -> Optional[dict]:
        return self._req("GET", "/budget")

    def get_my_rounds(self) -> list:
        return self._req("GET", "/my-rounds") or []

    def simulate(
        self, rid: str, si: int, x: int, y: int, w: int = 15, h: int = 15
    ) -> Optional[dict]:
        return self._req(
            "POST",
            "/simulate",
            {
                "round_id": rid,
                "seed_index": si,
                "viewport_x": x,
                "viewport_y": y,
                "viewport_w": w,
                "viewport_h": h,
            },
        )

    def submit(self, rid: str, si: int, prediction: list) -> Optional[dict]:
        return self._req(
            "POST",
            "/submit",
            {"round_id": rid, "seed_index": si, "prediction": prediction},
        )

    def get_analysis(self, rid: str, si: int) -> Optional[dict]:
        return self._req("GET", f"/analysis/{rid}/{si}")
