"""Report test results and scores to the NM i AI dashboard.

Silent fail if dashboard is unreachable. Uses only stdlib (no requests/httpx).
"""
import json
import os
import urllib.request
import urllib.error

DASHBOARD = os.environ.get("DASHBOARD_URL", "http://localhost:8090")
_TIMEOUT = 3  # seconds


def _post(path, payload):
    """POST JSON to dashboard. Returns parsed response or None on failure."""
    try:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{DASHBOARD}{path}",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception:
        return None


def _put(path, payload):
    """PUT JSON to dashboard. Returns parsed response or None on failure."""
    try:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{DASHBOARD}{path}",
            data=data,
            headers={"Content-Type": "application/json"},
            method="PUT",
        )
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception:
        return None


def report_test(task, label, status="running", score=None, details=None, metadata=None):
    """Create a new test entry. Returns test_id (str) or None."""
    payload = {"task": task, "label": label, "status": status}
    if score is not None:
        payload["score"] = score
    if details is not None:
        payload["details"] = details
    if metadata is not None:
        payload["metadata"] = metadata
    result = _post("/api/test", payload)
    if result and "id" in result:
        return result["id"]
    return None


def update_test(test_id, status=None, score=None, details=None, metadata=None):
    """Update an existing test. Returns updated record or None."""
    if test_id is None:
        return None
    payload = {}
    if status is not None:
        payload["status"] = status
    if score is not None:
        payload["score"] = score
    if details is not None:
        payload["details"] = details
    if metadata is not None:
        payload["metadata"] = metadata
    if not payload:
        return None
    return _put(f"/api/test/{test_id}", payload)


def report_score(task, raw, norm=None, rank=None, total_teams=None, note="", **kwargs):
    """Report a competition score. Returns response dict or None."""
    payload = {"task": task, "raw": raw, "note": note}
    if norm is not None:
        payload["norm"] = norm
    if rank is not None:
        payload["rank"] = rank
    if total_teams is not None:
        payload["total_teams"] = total_teams
    # Pass through any extra kwargs the API accepts
    for key in ("det_map", "cls_map", "tier", "tasks_solved", "efficiency",
                "kl_div", "queries_left", "leader_score", "submission_id"):
        if key in kwargs and kwargs[key] is not None:
            payload[key] = kwargs[key]
    return _post("/api/score", payload)
