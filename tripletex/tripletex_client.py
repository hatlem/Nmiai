import asyncio
import httpx
import json as _json
import logging
from pathlib import Path as _Path

logger = logging.getLogger(__name__)

# --- OpenAPI field validation ---
_OPENAPI_FIELDS: dict[str, dict] = {}
try:
    _fields_path = _Path(__file__).parent / "schemas" / "openapi_fields.json"
    if _fields_path.exists():
        _OPENAPI_FIELDS = _json.loads(_fields_path.read_text())
except Exception:
    pass

# Map API paths to entity names
_PATH_TO_ENTITY = {
    "/employee": "Employee",
    "/customer": "Customer",
    "/supplier": "Supplier",
    "/product": "Product",
    "/department": "Department",
    "/project": "Project",
    "/order": "Order",
    "/invoice": "Invoice",
    "/contact": "Contact",
    "/ledger/voucher": "Voucher",
    "/travelExpense": "TravelExpense",
    "/travelExpense/cost": "TravelExpenseCost",
    "/employee/employment": "Employment",
    "/employee/employment/details": "EmploymentDetails",
    "/salary/transaction": "SalaryTransaction",
    "/timesheet/entry": "TimesheetEntry",
    "/purchaseOrder": "PurchaseOrder",
    "/asset": "Asset",
    "/supplierInvoice": "SupplierInvoice",
}

# Known field renames (wrong -> correct)
_FIELD_RENAMES = {
    # Employee fields
    "nationalIdNumber": "nationalIdentityNumber",
    "mobilePhone": "phoneNumberMobile",
    "mobile": "phoneNumberMobile",
    "phoneNumberWork": "phoneNumberWork",
    "phone": "phoneNumber",
    "birthDate": "dateOfBirth",
    "birthday": "dateOfBirth",
    # Customer/Supplier fields
    "orgNumber": "organizationNumber",
    "orgNo": "organizationNumber",
    "organisationNumber": "organizationNumber",
    "address1": "addressLine1",
    "address": "addressLine1",
    "zipCode": "postalCode",
    "zip": "postalCode",
    "postalAddress": "postalAddress",
    # Product fields
    "quantity": "count",
    "price": "priceExcludingVatCurrency",
    "unitPrice": "unitPriceExcludingVatCurrency",
    "salesPrice": "priceExcludingVatCurrency",
    "productId": "product",
    "productNumber": "number",
    # Project fields
    "fixedPrice": "fixedprice",
    "budget": "fixedprice",
    # Invoice/Order fields
    "dueDate": "invoiceDueDate",
    "totalAmount": "amount",
    # Voucher/Posting fields
    "isDebit": None,  # strip — not a valid field
    "debit": None,
    "credit": None,
}


def _validate_fields(path: str, body: dict) -> dict:
    """Strip invalid fields and rename known wrong names based on OpenAPI spec."""
    if not body or not isinstance(body, dict):
        return body

    # Skip voucher postings validation (nested, different rules)
    if "/ledger/voucher" in path:
        return body

    # Find entity for this path
    entity_name = None
    # Match longest path first (e.g. /employee/employment/details before /employee/employment)
    best_match = ""
    for api_path, entity in _PATH_TO_ENTITY.items():
        stripped = path.rstrip("/").split("?")[0]
        if stripped == api_path or stripped.startswith(api_path + "/"):
            if len(api_path) > len(best_match):
                best_match = api_path
                entity_name = entity

    if not entity_name or entity_name not in _OPENAPI_FIELDS:
        return body

    valid_fields = set(_OPENAPI_FIELDS[entity_name].keys())
    cleaned = {}
    stripped_fields = []

    for key, value in body.items():
        # Try rename first
        if key in _FIELD_RENAMES:
            new_key = _FIELD_RENAMES[key]
            if new_key is None:
                stripped_fields.append(key)
                continue  # Strip field entirely
            if new_key in valid_fields:
                logger.info(f"OpenAPI: renamed field '{key}' -> '{new_key}' for {entity_name}")
                cleaned[new_key] = value
                continue

        # Keep if valid, strip if not
        if key in valid_fields:
            cleaned[key] = value
        else:
            stripped_fields.append(key)

    if stripped_fields:
        logger.warning(f"OpenAPI: stripped invalid fields from {entity_name}: {stripped_fields}")

    return cleaned

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
        self.vat_number_to_id: dict[int, int] = {}  # vatType number → actual DB id

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

    def _fix_body(self, body: dict | None, path: str) -> dict | None:
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
        # LLM sometimes sends VAT percentages (25, 15) instead of vatType IDs
        _VAT_PCT_TO_NUMBER = {25: 3, 15: 5, 10: 4, 12: 6, 6: 6, 8: 7}
        def _fix_vat_value(v):
            """Convert bare vatType (int/str) to {"id": X}, mapping percentages to IDs."""
            if isinstance(v, (int, float)):
                n = int(v)
                return {"id": _VAT_PCT_TO_NUMBER.get(n, n)}
            elif isinstance(v, str) and v.isdigit():
                n = int(v)
                return {"id": _VAT_PCT_TO_NUMBER.get(n, n)}
            return v

        val = body.get("vatType")
        if isinstance(val, (int, float, str)) and not isinstance(val, dict):
            body["vatType"] = _fix_vat_value(val)
        # Fix vatType in nested orderLines
        for line in body.get("orderLines", []):
            if isinstance(line, dict):
                vt = line.get("vatType")
                if isinstance(vt, (int, float)):
                    line["vatType"] = _fix_vat_value(vt)
                elif isinstance(vt, str) and vt.isdigit():
                    line["vatType"] = _fix_vat_value(vt)
        # Fix vatType in nested postings
        for posting in body.get("postings", []):
            if isinstance(posting, dict):
                vt = posting.get("vatType")
                if isinstance(vt, (int, float)):
                    posting["vatType"] = _fix_vat_value(vt)
                elif isinstance(vt, str) and vt.isdigit():
                    posting["vatType"] = _fix_vat_value(vt)
        # Bug fix 1: Auto-convert postings amount fields from string to number
        if "postings" in body and isinstance(body["postings"], list):
            for posting in body["postings"]:
                if isinstance(posting, dict):
                    for field in ("amountGross", "amountGrossCurrency", "amount", "amountCurrencyIncVat"):
                        val = posting.get(field)
                        if val is not None and not isinstance(val, (int, float)):
                            try:
                                posting[field] = float(str(val).replace(",", ".").strip())
                            except (ValueError, TypeError):
                                posting.pop(field, None)
                    # amountGrossCurrency MUST equal amountGross
                    if "amountGross" in posting:
                        posting["amountGrossCurrency"] = posting["amountGross"]
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
                        line["vatType"] = {"id": self.resolve_vat_id(3)}  # default 25% outgoing VAT
                    elif vt is None:
                        line["vatType"] = {"id": self.resolve_vat_id(3)}  # default 25% outgoing VAT
        # Fix: employmentType must be integer, not string or object
        if "employmentType" in body:
            val = body["employmentType"]
            if isinstance(val, dict) and "id" in val:
                body["employmentType"] = val["id"]
            elif isinstance(val, str):
                try:
                    body["employmentType"] = int(val)
                except (ValueError, TypeError):
                    del body["employmentType"]

        # Same for workingHoursScheme, remunerationType, occupationCode
        for field in ("workingHoursScheme", "remunerationType", "occupationCode"):
            if field in body:
                val = body[field]
                if isinstance(val, dict) and "id" in val:
                    body[field] = val["id"]
                elif isinstance(val, str):
                    try:
                        body[field] = int(val)
                    except (ValueError, TypeError):
                        del body[field]

        # Fix: Strip invalid fields from employment/details requests
        if "/employment/details" in path:
            for bad in ("position", "title", "jobTitle", "role", "userType"):
                body.pop(bad, None)

        # Fix: Strip fields that cause 500 on POST /supplierInvoice
        if "/supplierInvoice" in path and "/orderline" not in path.lower():
            for bad in ("orderDate", "deliveryDate", "dueDate", "invoiceDueDate",
                        "amount", "amountCurrency", "orderLines"):
                body.pop(bad, None)

        return body

    async def request(
        self, method: str, path: str, body: dict | None = None, params: dict | None = None
    ) -> dict:
        if body and method in ("POST", "PUT"):
            body = self._fix_body(body, path)
            # Resolve vatType number→id for all vatType references
            if self.vat_number_to_id:
                body = self._resolve_all_vat_ids(body)
            # Validate fields against OpenAPI spec (top-level only)
            body = _validate_fields(path, body)
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

        # /:send uses sendType, /:createReminder uses dispatchType
        if "/:send" in path and params:
            if "dispatchType" in params and "sendType" not in params:
                params["sendType"] = params.pop("dispatchType")

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

    def _resolve_all_vat_ids(self, body: dict) -> dict:
        """Walk body and resolve all vatType {"id": number} to actual DB ids."""
        def _resolve_vat(obj):
            if isinstance(obj, dict):
                vt = obj.get("vatType")
                if isinstance(vt, dict) and "id" in vt:
                    vid = vt["id"]
                    if isinstance(vid, int) and vid in self.vat_number_to_id:
                        vt["id"] = self.vat_number_to_id[vid]
                for v in obj.values():
                    _resolve_vat(v)
            elif isinstance(obj, list):
                for item in obj:
                    _resolve_vat(item)
        _resolve_vat(body)
        return body

    async def resolve_vat_types(self):
        """Fetch vatType list and build number→id mapping.
        Must be called once per session before executing plans."""
        resp = await self.request("GET", "/ledger/vatType", params={
            "fields": "id,name,number", "count": "100",
        })
        if resp.get("ok"):
            for vt in resp.get("data", {}).get("values", []):
                num = vt.get("number")
                vid = vt.get("id")
                if num is not None and vid is not None:
                    try:
                        self.vat_number_to_id[int(num)] = int(vid)
                    except (ValueError, TypeError):
                        pass  # Skip non-numeric vatType numbers like "UT-2"
            logger.info(f"Resolved {len(self.vat_number_to_id)} vatType mappings: {self.vat_number_to_id}")
        else:
            logger.warning(f"Failed to fetch vatTypes: {resp}")

    def resolve_vat_id(self, number: int) -> int:
        """Convert vatType number to actual DB id. Falls back to number if not resolved."""
        return self.vat_number_to_id.get(number, number)

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
