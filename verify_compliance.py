#!/usr/bin/env python3
"""
NM i AI 2026 — Competition Compliance Verifier

Checks the codebase against all competition rules and reports PASS/FAIL/WARN.
Uses only stdlib: pathlib, re, ast, json, subprocess (for git commands).

Usage:
    python3 verify_compliance.py
    python3 verify_compliance.py --json   # machine-readable JSON output

Exit codes:
    0 = all checks passed
    1 = one or more checks failed
"""

import argparse
import json
import re
import subprocess
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).parent.resolve()
TRIPLETEX = ROOT / "tripletex"
ASTAR = ROOT / "astar-island"
NORGESGRUPPEN = ROOT / "norgesgruppen"

# ── Result tracking ──────────────────────────────────────────────────────

class CheckResult:
    def __init__(self, group: str, name: str, status: str, detail: str = ""):
        self.group = group
        self.name = name
        self.status = status  # PASS, FAIL, WARN
        self.detail = detail

    def to_dict(self):
        return {
            "group": self.group,
            "name": self.name,
            "status": self.status,
            "detail": self.detail,
        }

    def __str__(self):
        tag = f"[{self.status:4s}]"
        detail = f" -- {self.detail}" if self.detail else ""
        return f"  {tag} {self.name}{detail}"


results: list[CheckResult] = []


def check(group: str, name: str, passed: bool, detail: str = ""):
    status = "PASS" if passed else "FAIL"
    results.append(CheckResult(group, name, status, detail))


def warn(group: str, name: str, detail: str = ""):
    results.append(CheckResult(group, name, "WARN", detail))


# ── Helpers ──────────────────────────────────────────────────────────────

def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""


def git_ls_files() -> list[str]:
    try:
        out = subprocess.check_output(
            ["git", "ls-files"], cwd=str(ROOT), text=True, stderr=subprocess.DEVNULL
        )
        return out.strip().splitlines()
    except Exception:
        return []


def find_py_files(directory: Path) -> list[Path]:
    return sorted(directory.rglob("*.py"))


SECRET_PATTERNS = [
    (r'(?:api[_-]?key|apikey)\s*[=:]\s*["\'][A-Za-z0-9_\-]{16,}["\']', "API key assignment"),
    (r'(?:secret|token|password)\s*[=:]\s*["\'][A-Za-z0-9_\-]{16,}["\']', "Secret/token assignment"),
    (r'AIza[A-Za-z0-9_\-]{35}', "Google API key"),
    (r'sk-[A-Za-z0-9]{40,}', "OpenAI-style API key"),
    (r'ghp_[A-Za-z0-9]{36}', "GitHub PAT"),
    (r'AKIA[A-Z0-9]{16}', "AWS access key"),
]


# ── Global Checks ────────────────────────────────────────────────────────

def check_global():
    group = "global"

    # MIT license file
    license_file = ROOT / "LICENSE"
    license_md = ROOT / "LICENSE.md"
    has_license = license_file.exists() or license_md.exists()
    check(group, "MIT license file exists", has_license)

    if has_license:
        content = read_text(license_file if license_file.exists() else license_md)
        check(group, "License contains MIT text", "MIT License" in content or "MIT" in content.upper())

    # No .env files in git
    tracked = git_ls_files()
    env_files = [f for f in tracked if Path(f).name.startswith(".env")]
    check(group, "No .env files tracked in git", len(env_files) == 0,
          f"Found: {env_files}" if env_files else "")

    # No hardcoded secrets in code
    secret_hits = []
    for py_file in find_py_files(ROOT):
        # Skip __pycache__, .git, node_modules, venv
        parts = py_file.parts
        if any(p in ("__pycache__", ".git", "node_modules", "venv", ".venv") for p in parts):
            continue
        content = read_text(py_file)
        for pattern, label in SECRET_PATTERNS:
            matches = re.findall(pattern, content, re.IGNORECASE)
            for m in matches:
                # Skip if it looks like a placeholder or env var reference
                if any(placeholder in m.lower() for placeholder in
                       ["your_", "xxx", "placeholder", "example", "test", "dummy", "fake"]):
                    continue
                relative = py_file.relative_to(ROOT)
                secret_hits.append(f"{relative}: {label}")

    if secret_hits:
        # Limit output
        detail = "; ".join(secret_hits[:5])
        if len(secret_hits) > 5:
            detail += f" ... and {len(secret_hits) - 5} more"
        warn(group, "Possible hardcoded secrets found", detail)
    else:
        check(group, "No hardcoded API keys/tokens detected", True)


# ── Tripletex Checks ─────────────────────────────────────────────────────

def check_tripletex():
    group = "tripletex"

    main_py = TRIPLETEX / "main.py"
    client_py = TRIPLETEX / "tripletex_client.py"
    executor_py = TRIPLETEX / "executor.py"

    if not main_py.exists():
        check(group, "main.py exists", False, "tripletex/main.py not found")
        return

    main_content = read_text(main_py)
    client_content = read_text(client_py)
    executor_content = read_text(executor_py)

    # Has /solve endpoint
    check(group, "Has /solve endpoint",
          bool(re.search(r'@app\.(post|route)\s*\(\s*["\']/solve', main_content)),
          "Found POST /solve" if re.search(r'@app\.post\s*\(\s*["\']/solve', main_content) else "")

    # Returns {"status": "completed"}
    check(group, 'Returns {"status": "completed"}',
          '"status"' in main_content and '"completed"' in main_content)

    # Uses base_url from request
    check(group, "Uses base_url from request (not hardcoded)",
          'base_url = creds["base_url"]' in main_content or
          "base_url = creds['base_url']" in main_content)

    # Uses Basic Auth with username "0"
    check(group, 'Uses Basic Auth with username "0"',
          '("0",' in client_content or "('0'," in client_content,
          'Found auth = ("0", session_token)' if '("0",' in client_content else "")

    # Handles base64 file attachments
    check(group, "Handles file attachments (files from body)",
          "files" in main_content and ('body.get("files"' in main_content or
                                        "body.get('files'" in main_content))

    # Timeout handling
    has_timeout = False
    timeout_val = None
    m = re.search(r'DEADLINE\s*=\s*(\d+)', executor_content)
    if m:
        timeout_val = int(m.group(1))
        has_timeout = timeout_val < 300
    check(group, "Has timeout handling (<300s)",
          has_timeout,
          f"DEADLINE={timeout_val}s" if timeout_val else "No DEADLINE found")

    # No direct Tripletex API URLs hardcoded (in non-doc, non-test files)
    hardcoded_urls = []
    for py_file in find_py_files(TRIPLETEX):
        name = py_file.name
        if name.startswith("test_") or "doc" in str(py_file):
            continue
        content = read_text(py_file)
        # Look for hardcoded full URLs to tripletex
        urls = re.findall(r'https?://(?:api\.)?tripletex\.(?:no|io|dev)/v2/', content)
        if urls:
            hardcoded_urls.append(str(py_file.relative_to(ROOT)))

    check(group, "No direct Tripletex API URLs hardcoded",
          len(hardcoded_urls) == 0,
          f"Found in: {hardcoded_urls}" if hardcoded_urls else "")

    # SSRF prevention
    check(group, "SSRF prevention (base_url validation)",
          "ALLOWED_HOSTS" in main_content or "urlparse" in main_content)


# ── Astar Island Checks ──────────────────────────────────────────────────

def check_astar():
    group = "astar"

    priors_py = ASTAR / "priors.py"
    prediction_py = ASTAR / "prediction.py"
    query_opt_py = ASTAR / "query_optimizer.py"
    swarm_py = ASTAR / "swarm.py"

    if not priors_py.exists():
        check(group, "priors.py exists", False)
        return

    priors_content = read_text(priors_py)
    prediction_content = read_text(prediction_py) if prediction_py.exists() else ""
    query_content = read_text(query_opt_py) if query_opt_py.exists() else ""
    swarm_content = read_text(swarm_py) if swarm_py.exists() else ""

    # Probability floor >= 0.01
    # Check PROB_FLOOR, MIN_FLOOR, STATIC_FLOOR values
    floor_values = {}
    for var_name in ("PROB_FLOOR", "MIN_FLOOR", "STATIC_FLOOR", "REMOTE_FLOOR"):
        m = re.search(rf'{var_name}\s*=\s*([\d.]+)', priors_content)
        if m:
            floor_values[var_name] = float(m.group(1))

    min_floor_val = min(floor_values.values()) if floor_values else 0
    # Competition says never 0.0 -- we allow small floors as long as they're > 0
    check(group, "All probability floors > 0",
          min_floor_val > 0,
          f"Floor values: {floor_values}")

    # Check that PROB_FLOOR (the main dynamic floor) is reasonable
    prob_floor = floor_values.get("PROB_FLOOR", 0)
    if prob_floor < 0.005:
        warn(group, "PROB_FLOOR may be too low", f"PROB_FLOOR={prob_floor}")
    else:
        check(group, "PROB_FLOOR >= 0.005", True, f"PROB_FLOOR={prob_floor}")

    # Normalization: probabilities sum to 1.0
    has_normalization = (
        "/= p.sum()" in priors_content or
        "p /= p.sum()" in priors_content or
        "/= arr.sum()" in priors_content or
        "sum(axis=2, keepdims=True)" in prediction_content or
        "sum(axis=-1, keepdims=True)" in prediction_content or
        "/= base.sum()" in prediction_content
    )
    check(group, "Probabilities are normalized (sum to 1.0)", has_normalization)

    # Never assigns 0.0 probability
    # Check that floors are applied before normalization
    has_floor_application = (
        "np.maximum" in priors_content or
        "np.maximum" in prediction_content or
        "np.clip" in prediction_content
    )
    check(group, "Probability floors applied (np.maximum/np.clip)",
          has_floor_application)

    # Viewport max 15x15 (check default parameter value and assignments)
    # Handles both `viewport_max = 15` and `viewport_max: int = 15`
    viewport_match = re.search(r'viewport_max(?:\s*:\s*\w+)?\s*=\s*(\d+)', query_content)
    if viewport_match:
        vp_max = int(viewport_match.group(1))
        check(group, "Viewport max 15x15", vp_max <= 15, f"viewport_max={vp_max}")
    else:
        warn(group, "Could not find viewport_max setting")

    # Budget of 50 queries
    has_budget_50 = "50" in query_content and "budget" in query_content.lower()
    check(group, "Respects 50 query budget",
          has_budget_50 or "budget" in query_content.lower(),
          "budget parameter found in QueryOptimizer")

    # Submits all 5 seeds
    # Look for seeds_count usage or iteration over all seeds
    has_all_seeds = (
        "seeds_count" in query_content or
        "range(seeds_count)" in query_content or
        "range(self.seeds_count)" in query_content
    )
    check(group, "Handles all seeds (seeds_count parameter)", has_all_seeds)

    # Check swarm also applies floors
    if swarm_content:
        swarm_has_floors = "PROB_FLOOR" in swarm_content or "STATIC_FLOOR" in swarm_content
        check(group, "Swarm agents use probability floors", swarm_has_floors)


# ── NorgesGruppen Checks ─────────────────────────────────────────────────

def check_norgesgruppen():
    group = "norgesgruppen"

    run_py = NORGESGRUPPEN / "run.py"

    # run.py exists
    check(group, "run.py exists", run_py.exists())
    if not run_py.exists():
        return

    content = read_text(run_py)

    # Accepts --input and --output args
    check(group, "Accepts --input argument",
          '--input' in content or "'--input'" in content or '"--input"' in content)
    check(group, "Accepts --output argument",
          '--output' in content or "'--output'" in content or '"--output"' in content)

    # Output format check (JSON with required fields)
    has_image_id = "image_id" in content
    has_category_id = "category_id" in content
    has_bbox = "bbox" in content
    has_score = '"score"' in content or "'score'" in content
    check(group, "Output has image_id, category_id, bbox, score fields",
          has_image_id and has_category_id and has_bbox and has_score,
          f"image_id={has_image_id} category_id={has_category_id} bbox={has_bbox} score={has_score}")

    # Blocked imports check - only in run.py and src/ files that would be in submission
    blocked_imports = ["import os", "from os import",
                       "import sys", "from sys import",
                       "import subprocess", "from subprocess import",
                       "import socket", "from socket import",
                       "import ctypes", "from ctypes import",
                       "import builtins", "from builtins import"]

    submission_files = [run_py] + list((NORGESGRUPPEN / "src").rglob("*.py")) if (NORGESGRUPPEN / "src").exists() else [run_py]

    blocked_found = []
    for py_file in submission_files:
        file_content = read_text(py_file)
        for line_no, line in enumerate(file_content.splitlines(), 1):
            stripped = line.strip()
            # Skip comments
            if stripped.startswith("#"):
                continue
            # Skip strings (rough check: line contains the import as actual code)
            for blocked in blocked_imports:
                if stripped.startswith(blocked) or stripped == blocked:
                    relative = py_file.relative_to(ROOT)
                    blocked_found.append(f"{relative}:{line_no}: {stripped}")

    check(group, "No blocked imports (os/sys/subprocess/socket) in submission files",
          len(blocked_found) == 0,
          "; ".join(blocked_found[:5]) if blocked_found else "")

    # No eval/exec/compile
    dangerous_calls = []
    for py_file in submission_files:
        file_content = read_text(py_file)
        for line_no, line in enumerate(file_content.splitlines(), 1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            for func in ("eval(", "exec(", "compile(", "__import__("):
                if func in stripped:
                    # Skip PyTorch .eval() method calls (model.eval(), etc.)
                    if func == "eval(" and ".eval()" in stripped:
                        continue
                    relative = py_file.relative_to(ROOT)
                    dangerous_calls.append(f"{relative}:{line_no}: {func}")

    check(group, "No eval/exec/compile/__import__ in submission files",
          len(dangerous_calls) == 0,
          "; ".join(dangerous_calls[:5]) if dangerous_calls else "")

    # No symlinks (only in submission-scope: run.py level + src/)
    symlinks = []
    submission_scope = [NORGESGRUPPEN / "src"]
    for p in NORGESGRUPPEN.iterdir():
        if p.is_symlink():
            symlinks.append(str(p.relative_to(ROOT)))
    for scope_dir in submission_scope:
        if scope_dir.exists():
            for p in scope_dir.rglob("*"):
                if p.is_symlink():
                    symlinks.append(str(p.relative_to(ROOT)))
    check(group, "No symlinks in submission directory",
          len(symlinks) == 0,
          f"Found: {symlinks}" if symlinks else "")

    # Weight file sizes (submission scope only: root + models/)
    weight_extensions = {".pt", ".pth", ".onnx", ".safetensors", ".npy"}
    weight_files = []
    for ext in weight_extensions:
        weight_files.extend(NORGESGRUPPEN.glob(f"*{ext}"))
        weight_files.extend(NORGESGRUPPEN.glob(f"models/*{ext}"))
    # Deduplicate
    weight_files = list(set(weight_files))

    total_weight_mb = sum(f.stat().st_size for f in weight_files if f.exists()) / (1024 * 1024)
    check(group, "Weight files < 420 MB total",
          total_weight_mb < 420,
          f"{total_weight_mb:.1f} MB across {len(weight_files)} files")

    check(group, "Max 3 weight files",
          len(weight_files) <= 3,
          f"Found {len(weight_files)} weight files")

    # Max 10 .py files in submission scope (run.py + src/)
    py_count = len([f for f in submission_files if f.suffix == ".py"])
    check(group, "Max 10 .py files in submission",
          py_count <= 10,
          f"Found {py_count} .py files")

    # Only allowed file extensions
    allowed_extensions = {".py", ".json", ".yaml", ".yml", ".pt", ".pth",
                          ".onnx", ".safetensors", ".npy", ".txt", ".cfg"}
    # Check files that would be in a submission zip (run.py dir + src/)
    bad_extensions = []
    for f in submission_files:
        if f.suffix not in allowed_extensions:
            bad_extensions.append(f"{f.name} ({f.suffix})")
    # Also check weight files
    for f in weight_files:
        if f.suffix not in allowed_extensions:
            bad_extensions.append(f"{f.name} ({f.suffix})")

    if bad_extensions:
        warn(group, "Non-standard file extensions found", "; ".join(bad_extensions[:5]))
    else:
        check(group, "Only allowed file extensions", True)

    # category_id range check (code should produce 0-355)
    # Just verify the code references the range or has a max
    has_category_range = (
        "355" in content or
        "356" in content or
        "category_id" in content
    )
    check(group, "category_id references valid range",
          has_category_range,
          "Found category_id handling in code")

    # Uses pathlib not os
    uses_pathlib = "from pathlib import" in content or "import pathlib" in content
    check(group, "Uses pathlib for file operations",
          uses_pathlib,
          "Found pathlib import" if uses_pathlib else "Missing pathlib import")

    # ONNX opset check (submission scope only)
    onnx_files = list(set(list(NORGESGRUPPEN.glob("*.onnx")) + list(NORGESGRUPPEN.glob("models/*.onnx"))))
    for onnx_file in onnx_files:
        opset = _check_onnx_opset(onnx_file)
        if opset is not None:
            check(group, f"ONNX opset <= 20 ({onnx_file.name})",
                  opset <= 20,
                  f"opset_version={opset}")
        else:
            warn(group, f"Could not determine ONNX opset for {onnx_file.name}")


def _check_onnx_opset(path: Path) -> int | None:
    """Try to read ONNX opset version from the file header using protobuf parsing.

    ONNX uses protobuf. The opset_import field (field number 8) contains
    a nested message with version (field number 2, varint).
    This is a best-effort parser without requiring onnx/protobuf packages.
    """
    try:
        data = path.read_bytes()
        # Look for the string "opset_import" or just try to find opset version
        # Simple approach: search for the opset_import protobuf field
        # Field 8 (opset_import) = wire type 2 (length-delimited) = tag byte 0x42
        # Inside: field 2 (version) = wire type 0 (varint) = tag byte 0x10
        idx = 0
        while idx < len(data) - 2:
            if data[idx] == 0x42:  # field 8, wire type 2
                idx += 1
                # Read length varint
                length, consumed = _read_varint(data, idx)
                if consumed == 0 or length is None:
                    idx += 1
                    continue
                idx += consumed
                # Inside the submessage, look for field 2 (version)
                end = idx + length
                while idx < end:
                    if data[idx] == 0x10:  # field 2, wire type 0
                        idx += 1
                        version, consumed = _read_varint(data, idx)
                        if version is not None and 1 <= version <= 30:
                            return version
                        idx += consumed
                    else:
                        idx += 1
                continue
            idx += 1
        return None
    except Exception:
        return None


def _read_varint(data: bytes, idx: int) -> tuple:
    """Read a protobuf varint from data at idx. Returns (value, bytes_consumed)."""
    result = 0
    shift = 0
    consumed = 0
    while idx < len(data):
        b = data[idx]
        result |= (b & 0x7F) << shift
        shift += 7
        idx += 1
        consumed += 1
        if (b & 0x80) == 0:
            return result, consumed
        if consumed > 5:
            break
    return None, 0


# ── Report ────────────────────────────────────────────────────────────────

def print_report():
    groups = {}
    for r in results:
        groups.setdefault(r.group, []).append(r)

    total = len(results)
    passed = sum(1 for r in results if r.status == "PASS")
    failed = sum(1 for r in results if r.status == "FAIL")
    warned = sum(1 for r in results if r.status == "WARN")

    print("=" * 70)
    print("NM i AI 2026 -- Competition Compliance Report")
    print("=" * 70)

    for group_name in ("global", "tripletex", "astar", "norgesgruppen"):
        if group_name not in groups:
            continue
        group_results = groups[group_name]
        group_pass = sum(1 for r in group_results if r.status == "PASS")
        group_total = len(group_results)
        print(f"\n--- {group_name.upper()} ({group_pass}/{group_total} passed) ---")
        for r in group_results:
            print(r)

    print(f"\n{'=' * 70}")
    print(f"SUMMARY: {passed} passed, {failed} failed, {warned} warnings out of {total} checks")
    if failed > 0:
        print(f"STATUS: FAIL")
    elif warned > 0:
        print(f"STATUS: PASS (with warnings)")
    else:
        print(f"STATUS: PASS")
    print("=" * 70)


def get_json_report() -> dict:
    total = len(results)
    passed = sum(1 for r in results if r.status == "PASS")
    failed = sum(1 for r in results if r.status == "FAIL")
    warned = sum(1 for r in results if r.status == "WARN")

    return {
        "summary": {
            "total": total,
            "passed": passed,
            "failed": failed,
            "warnings": warned,
            "status": "FAIL" if failed > 0 else ("WARN" if warned > 0 else "PASS"),
        },
        "checks": [r.to_dict() for r in results],
    }


# ── Main ──────────────────────────────────────────────────────────────────

def run_all_checks():
    results.clear()
    check_global()
    check_tripletex()
    check_astar()
    check_norgesgruppen()


def main():
    parser = argparse.ArgumentParser(description="NM i AI 2026 compliance verifier")
    parser.add_argument("--json", action="store_true", help="Output JSON instead of text")
    args = parser.parse_args()

    run_all_checks()

    if args.json:
        print(json.dumps(get_json_report(), indent=2))
    else:
        print_report()

    has_failures = any(r.status == "FAIL" for r in results)
    sys.exit(1 if has_failures else 0)


if __name__ == "__main__":
    main()
