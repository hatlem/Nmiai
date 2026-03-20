#!/usr/bin/env python3
"""
NM i AI 2026 -- Competition Compliance Verifier (STRICT MODE)

Goes beyond grep-based checking. Actually extracts ZIPs, AST-parses Python,
validates tensor outputs, and tests real endpoints where possible.

Usage:
    python3 verify_compliance.py
    python3 verify_compliance.py --json

Exit codes:
    0 = all checks passed
    1 = one or more checks failed
"""

import argparse
import ast
import json
import re
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).parent.resolve()
TRIPLETEX = ROOT / "tripletex"
ASTAR = ROOT / "astar-island"
NORGESGRUPPEN = ROOT / "norgesgruppen"

# ---- Result tracking --------------------------------------------------------

class CheckResult:
    def __init__(self, group, name, status, detail=""):
        self.group = group
        self.name = name
        self.status = status
        self.detail = detail

    def to_dict(self):
        return {"group": self.group, "name": self.name,
                "status": self.status, "detail": self.detail}

    def __str__(self):
        d = f" -- {self.detail}" if self.detail else ""
        return f"  [{self.status:4s}] {self.name}{d}"


results = []


def check(group, name, passed, detail=""):
    results.append(CheckResult(group, name, "PASS" if passed else "FAIL", detail))


def warn(group, name, detail=""):
    results.append(CheckResult(group, name, "WARN", detail))


def read_text(path):
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""


def git_ls_files():
    try:
        return subprocess.check_output(
            ["git", "ls-files"], cwd=str(ROOT), text=True,
            stderr=subprocess.DEVNULL
        ).strip().splitlines()
    except Exception:
        return []


# ---- AST-based import checker -----------------------------------------------

BLOCKED_MODULES = {
    "os", "sys", "subprocess", "socket", "ctypes", "builtins", "importlib",
    "pickle", "marshal", "shelve", "shutil", "yaml", "requests", "urllib",
    "http.client", "multiprocessing", "threading", "signal", "gc",
    "code", "codeop", "pty",
}

BLOCKED_FUNCTIONS = {"eval", "exec", "compile", "__import__"}


def ast_check_file(filepath):
    """AST-parse a Python file and find blocked imports and dangerous calls."""
    source = read_text(filepath)
    if not source.strip():
        return [], []

    try:
        tree = ast.parse(source, filename=str(filepath))
    except SyntaxError:
        return [], [f"SyntaxError in {filepath.name}"]

    blocked_imports = []
    blocked_calls = []

    for node in ast.walk(tree):
        # Check imports
        if isinstance(node, ast.Import):
            for alias in node.names:
                mod = alias.name.split(".")[0]
                if mod in BLOCKED_MODULES:
                    blocked_imports.append(
                        f"{filepath.name}:{node.lineno} import {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                mod = node.module.split(".")[0]
                if mod in BLOCKED_MODULES:
                    blocked_imports.append(
                        f"{filepath.name}:{node.lineno} from {node.module} import ...")

        # Check dangerous function calls
        elif isinstance(node, ast.Call):
            func_name = None
            is_method_call = False
            if isinstance(node.func, ast.Name):
                func_name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                func_name = node.func.attr
                is_method_call = True  # something.eval() is a method call

            # Skip method calls like model.eval() -- only flag bare eval()
            if func_name in BLOCKED_FUNCTIONS:
                if is_method_call and func_name == "eval":
                    continue  # model.eval() is safe (PyTorch)
                if not is_method_call or func_name != "eval":
                    blocked_calls.append(
                        f"{filepath.name}:{node.lineno} {func_name}()")

    return blocked_imports, blocked_calls


# ---- ZIP structure validator -------------------------------------------------

def validate_zip(zip_path):
    """Extract and validate a submission ZIP. Returns dict of findings."""
    findings = {
        "exists": zip_path.exists(),
        "is_symlink": zip_path.is_symlink() if zip_path.exists() else False,
        "run_py_at_root": False,
        "run_py_nested": False,
        "py_files": [],
        "weight_files": [],
        "weight_total_mb": 0.0,
        "onnx_files": [],
        "blocked_imports": [],
        "blocked_calls": [],
        "all_files": [],
        "bad_extensions": [],
        "has_dirs_only_at_root": True,
    }

    if not zip_path.exists() or zip_path.is_symlink():
        return findings

    weight_ext = {".pt", ".pth", ".onnx", ".safetensors", ".npy"}
    allowed_ext = {".py", ".json", ".yaml", ".yml", ".cfg",
                   ".pt", ".pth", ".onnx", ".safetensors", ".npy", ".txt"}

    try:
        with zipfile.ZipFile(zip_path) as zf:
            for info in zf.infolist():
                name = info.filename
                p = Path(name)
                findings["all_files"].append(name)

                # Check if run.py is at root
                if name == "run.py":
                    findings["run_py_at_root"] = True
                elif p.name == "run.py" and "/" in name:
                    findings["run_py_nested"] = True

                # Categorize files
                if p.suffix == ".py":
                    findings["py_files"].append(name)
                if p.suffix in weight_ext:
                    findings["weight_files"].append(name)
                    findings["weight_total_mb"] += info.file_size / (1024 * 1024)
                if p.suffix == ".onnx":
                    findings["onnx_files"].append(name)

                # Check extensions (skip directories)
                if not info.is_dir() and p.suffix and p.suffix not in allowed_ext:
                    findings["bad_extensions"].append(f"{name} ({p.suffix})")

            # AST-parse all .py files from the ZIP
            with tempfile.TemporaryDirectory() as tmp_dir:
                for py_name in findings["py_files"]:
                    try:
                        data = zf.read(py_name)
                        tmp_file = Path(tmp_dir) / Path(py_name).name
                        tmp_file.write_bytes(data)
                        imports, calls = ast_check_file(tmp_file)
                        findings["blocked_imports"].extend(imports)
                        findings["blocked_calls"].extend(calls)
                    except Exception:
                        pass

                # Check ONNX opsets
                for onnx_name in findings["onnx_files"]:
                    try:
                        data = zf.read(onnx_name)
                        tmp_file = Path(tmp_dir) / Path(onnx_name).name
                        tmp_file.write_bytes(data)
                        opset = _check_onnx_opset(tmp_file)
                        findings["onnx_files"]  # already tracked
                        if opset is not None:
                            findings.setdefault("onnx_opsets", {})[onnx_name] = opset
                    except Exception:
                        pass

    except zipfile.BadZipFile:
        findings["bad_zip"] = True

    return findings


# ---- ONNX opset reader ------------------------------------------------------

def _check_onnx_opset(path):
    """Read ONNX opset version from protobuf header (no dependencies)."""
    try:
        data = path.read_bytes()
        idx = 0
        while idx < min(len(data) - 2, 2000):  # opset is in the header
            if data[idx] == 0x42:
                idx += 1
                length, consumed = _read_varint(data, idx)
                if consumed == 0 or length is None:
                    idx += 1
                    continue
                idx += consumed
                end = idx + length
                while idx < end:
                    if data[idx] == 0x10:
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


def _read_varint(data, idx):
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


# ==== CHECKS =================================================================

SECRET_PATTERNS = [
    (r'AIza[A-Za-z0-9_\-]{35}', "Google API key"),
    (r'sk-[A-Za-z0-9]{40,}', "OpenAI-style API key"),
    (r'ghp_[A-Za-z0-9]{36}', "GitHub PAT"),
    (r'AKIA[A-Z0-9]{16}', "AWS access key"),
]


def check_global():
    group = "global"

    # MIT license
    lic = ROOT / "LICENSE"
    lic_md = ROOT / "LICENSE.md"
    has = lic.exists() or lic_md.exists()
    check(group, "MIT license file exists", has)
    if has:
        content = read_text(lic if lic.exists() else lic_md)
        check(group, "License contains MIT text",
              "MIT" in content.upper(),
              "Found MIT in license" if "MIT" in content.upper() else "MIT not found")

    # No .env in git
    tracked = git_ls_files()
    env_files = [f for f in tracked if Path(f).name.startswith(".env")]
    check(group, "No .env files tracked in git",
          len(env_files) == 0,
          f"TRACKED: {env_files}" if env_files else "")

    # No hardcoded secrets (scan all tracked .py files)
    secret_hits = []
    for f in tracked:
        if not f.endswith(".py"):
            continue
        fp = ROOT / f
        if not fp.exists():
            continue
        content = read_text(fp)
        for pattern, label in SECRET_PATTERNS:
            for m in re.finditer(pattern, content):
                val = m.group()
                if any(p in val.lower() for p in ["your_", "xxx", "placeholder", "example"]):
                    continue
                secret_hits.append(f"{f}: {label}")

    if secret_hits:
        check(group, "No hardcoded API keys/tokens", False,
              "; ".join(secret_hits[:3]))
    else:
        check(group, "No hardcoded API keys/tokens", True)

    # Check .gitignore covers sensitive files
    gitignore = ROOT / ".gitignore"
    if gitignore.exists():
        gi_content = read_text(gitignore)
        covers_env = ".env" in gi_content
        covers_weights = "*.pt" in gi_content or "*.onnx" in gi_content or "models/" in gi_content
        check(group, ".gitignore covers .env files", covers_env,
              "'.env' pattern found" if covers_env else ".env not in .gitignore")
    else:
        warn(group, "No .gitignore file found")


def check_tripletex():
    group = "tripletex"

    main_py = TRIPLETEX / "main.py"
    if not main_py.exists():
        check(group, "main.py exists", False)
        return

    main_content = read_text(main_py)
    client_content = read_text(TRIPLETEX / "tripletex_client.py")
    executor_content = read_text(TRIPLETEX / "executor.py")

    # AST-parse to find the /solve endpoint properly
    try:
        tree = ast.parse(main_content)
        has_solve = False
        returns_completed = False
        for node in ast.walk(tree):
            # Match both def solve and async def solve
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "solve":
                has_solve = True
                for child in ast.walk(node):
                    if isinstance(child, ast.Constant) and child.value == "completed":
                        returns_completed = True
        # Also check decorator pattern
        if not has_solve:
            has_solve = bool(re.search(r'@app\.post\s*\(\s*["\']/solve', main_content))
        check(group, "Has /solve endpoint (AST verified)", has_solve)
        check(group, 'Returns "completed" status (AST verified)', returns_completed)
    except SyntaxError:
        check(group, "main.py parses without errors", False)

    # base_url from request (not hardcoded)
    check(group, "Uses base_url from request",
          'base_url' in main_content and 'creds' in main_content)

    # Basic Auth username "0"
    has_auth_zero = bool(re.search(r'["\']0["\']\s*,', client_content))
    check(group, 'Basic Auth username = "0"', has_auth_zero,
          "Found in tripletex_client.py" if has_auth_zero else "Not found")

    # Handles file attachments
    check(group, "Handles file attachments",
          "files" in main_content and ("base64" in main_content or "file" in main_content.lower()),
          "Checks for files in request body")

    # Timeout < 300s
    m = re.search(r'DEADLINE\s*=\s*(\d+)', executor_content)
    if m:
        val = int(m.group(1))
        check(group, f"Timeout < 300s (found {val}s)", val < 300)
    else:
        warn(group, "Could not find DEADLINE/timeout value")

    # No hardcoded Tripletex URLs
    hardcoded = []
    for py in TRIPLETEX.rglob("*.py"):
        if "__pycache__" in str(py):
            continue
        c = read_text(py)
        urls = re.findall(r'https?://[a-z0-9.-]*tripletex\.[a-z]+/v2/', c)
        if urls:
            hardcoded.extend([f"{py.name}: {u}" for u in urls])
    check(group, "No hardcoded Tripletex API URLs",
          len(hardcoded) == 0,
          "; ".join(hardcoded[:3]) if hardcoded else "")

    # Try hitting the deployed endpoint (if running)
    try:
        import urllib.request
        req = urllib.request.Request(
            "http://localhost:8080/health",
            method="GET",
        )
        urllib.request.urlopen(req, timeout=3)
        check(group, "Local endpoint responds", True)
    except Exception:
        warn(group, "Local endpoint not reachable (not running or different port)")


def check_astar():
    group = "astar"

    priors_py = ASTAR / "priors.py"
    prediction_py = ASTAR / "prediction.py"
    query_opt_py = ASTAR / "query_optimizer.py"

    if not priors_py.exists():
        check(group, "priors.py exists", False)
        return

    # ACTUAL VALIDATION: import priors and test the output
    # Run in subprocess to avoid polluting this process
    test_code = '''
import sys, json
sys.path.insert(0, "{astar_dir}")
try:
    import numpy as np
    from priors import CALIBRATED_PRIORS, get_domain_prior, NUM_CLASSES, PROB_FLOOR, STATIC_FLOOR, MIN_FLOOR, REMOTE_FLOOR

    issues = []

    # Test 1: All calibrated priors sum to ~1.0
    for cls_idx, prior in CALIBRATED_PRIORS.items():
        s = prior.sum()
        if abs(s - 1.0) > 0.02:
            issues.append(f"Class {{cls_idx}} prior sums to {{s:.4f}}")

    # Test 2: No zeros in any prior
    for cls_idx, prior in CALIBRATED_PRIORS.items():
        if (prior <= 0).any():
            issues.append(f"Class {{cls_idx}} has zero/negative: {{prior.tolist()}}")

    # Test 3: get_domain_prior returns valid distributions
    for cls_idx in range(NUM_CLASSES):
        p = get_domain_prior(cls_idx)
        s = p.sum()
        if abs(s - 1.0) > 0.02:
            issues.append(f"get_domain_prior({{cls_idx}}) sums to {{s:.4f}}")
        if (p <= 0).any():
            issues.append(f"get_domain_prior({{cls_idx}}) has zero/negative")
        if p.min() < 0.0005:
            issues.append(f"get_domain_prior({{cls_idx}}) min={{p.min():.6f}} (dangerously low)")

    # Test 4: Floor values
    floors = dict(PROB_FLOOR=PROB_FLOOR, STATIC_FLOOR=STATIC_FLOOR, MIN_FLOOR=MIN_FLOOR, REMOTE_FLOOR=REMOTE_FLOOR)

    result = dict(
        issues=issues,
        floors=floors,
        num_classes=NUM_CLASSES,
        calibrated_classes=list(CALIBRATED_PRIORS.keys()),
    )
    print(json.dumps(result))
except Exception as e:
    print(json.dumps(dict(error=str(e))))
'''.format(astar_dir=str(ASTAR))

    try:
        proc = subprocess.run(
            [sys.executable, "-c", test_code],
            capture_output=True, text=True, timeout=10,
        )
        data = json.loads(proc.stdout.strip())

        if "error" in data:
            check(group, "Priors module loads without errors", False, data["error"])
        else:
            issues = data.get("issues", [])
            floors = data.get("floors", {})

            check(group, "Priors module loads without errors", True)

            check(group, "All calibrated priors sum to 1.0 (TESTED)",
                  not any("sums to" in i for i in issues),
                  "; ".join(i for i in issues if "sums to" in i) or "All valid")

            check(group, "No zero probabilities in any prior (TESTED)",
                  not any("zero" in i.lower() or "negative" in i.lower() for i in issues),
                  "; ".join(i for i in issues if "zero" in i.lower() or "negative" in i.lower()) or "All > 0")

            check(group, "No dangerously low probabilities (TESTED)",
                  not any("dangerously" in i for i in issues),
                  "; ".join(i for i in issues if "dangerously" in i) or "All floors safe")

            # Floor values
            prob_floor = floors.get("PROB_FLOOR", 0)
            min_floor = floors.get("MIN_FLOOR", 0)
            check(group, f"PROB_FLOOR={prob_floor} (>= 0.005)",
                  prob_floor >= 0.005)
            check(group, f"All floors > 0",
                  all(v > 0 for v in floors.values()),
                  json.dumps(floors))

    except subprocess.TimeoutExpired:
        check(group, "Priors module loads within 10s", False, "Timeout")
    except (json.JSONDecodeError, Exception) as e:
        check(group, "Priors validation ran", False, str(e))

    # Static code checks
    priors_content = read_text(priors_py)
    prediction_content = read_text(prediction_py)
    query_content = read_text(query_opt_py) if query_opt_py.exists() else ""

    # Viewport max
    vp_match = re.search(r'viewport_max(?:\s*:\s*\w+)?\s*=\s*(\d+)', query_content)
    if vp_match:
        vp = int(vp_match.group(1))
        check(group, f"Viewport max = {vp} (<= 15)", vp <= 15)
    else:
        warn(group, "Could not find viewport_max")

    # Budget check
    budget_match = re.search(r'budget(?:\s*:\s*\w+)?\s*=\s*(\d+)', query_content)
    if budget_match:
        budget = int(budget_match.group(1))
        check(group, f"Query budget = {budget} (<= 50)", budget <= 50)
    else:
        check(group, "Budget parameter exists",
              "budget" in query_content.lower())

    # Seeds count
    check(group, "Handles all seeds",
          "seeds_count" in query_content or "seeds_count" in prediction_content)

    # Normalization in prediction engine
    check(group, "Final normalization (sum to 1.0)",
          "sum(axis=2, keepdims=True)" in prediction_content or
          "sum(axis=-1, keepdims=True)" in prediction_content)

    # Calibration.json validation
    cal_file = ASTAR / "calibration.json"
    if cal_file.exists():
        try:
            cal = json.loads(read_text(cal_file))
            cal_issues = []
            for key, values in cal.items():
                if key.startswith("_"):
                    continue
                if not isinstance(values, list) or len(values) != 6:
                    cal_issues.append(f"Class {key}: expected 6 values, got {len(values) if isinstance(values, list) else type(values)}")
                    continue
                s = sum(values)
                if abs(s - 1.0) > 0.05:
                    cal_issues.append(f"Class {key}: sums to {s:.3f}")
                if any(v < 0 for v in values):
                    cal_issues.append(f"Class {key}: has negative values")
            check(group, "calibration.json is valid (6 classes, sums ~1.0)",
                  len(cal_issues) == 0,
                  "; ".join(cal_issues) if cal_issues else "All classes valid")
        except json.JSONDecodeError:
            check(group, "calibration.json is valid JSON", False)
    else:
        warn(group, "calibration.json not found")


def check_norgesgruppen():
    group = "norgesgruppen"

    # ---- ZIP-based validation (the REAL test) ----
    submission_zip = NORGESGRUPPEN / "submission.zip"
    check(group, "submission.zip exists", submission_zip.exists())
    check(group, "submission.zip is not a symlink",
          not submission_zip.is_symlink() if submission_zip.exists() else False)

    if not submission_zip.exists():
        warn(group, "Cannot run ZIP validation without submission.zip")
        return

    zf = validate_zip(submission_zip)

    # run.py at root (NOT nested)
    check(group, "run.py at ZIP root (not nested)",
          zf["run_py_at_root"],
          "NESTED in subfolder!" if zf["run_py_nested"] else
          ("Not found in ZIP" if not zf["run_py_at_root"] else ""))

    # Weight files
    check(group, f"Weight files: {zf['weight_total_mb']:.1f} MB (< 420 MB)",
          zf["weight_total_mb"] < 420,
          ", ".join(zf["weight_files"]))

    check(group, f"Weight file count: {len(zf['weight_files'])} (<= 3)",
          len(zf["weight_files"]) <= 3,
          ", ".join(zf["weight_files"]))

    # Python files
    check(group, f"Python files: {len(zf['py_files'])} (<= 10)",
          len(zf["py_files"]) <= 10,
          ", ".join(zf["py_files"]))

    # File extensions
    check(group, "All file extensions allowed",
          len(zf["bad_extensions"]) == 0,
          "; ".join(zf["bad_extensions"]) if zf["bad_extensions"] else "")

    # AST-based blocked import check (from actual ZIP contents)
    check(group, "No blocked imports in ZIP (AST-parsed)",
          len(zf["blocked_imports"]) == 0,
          "; ".join(zf["blocked_imports"][:5]) if zf["blocked_imports"] else "")

    check(group, "No eval/exec/compile in ZIP (AST-parsed)",
          len(zf["blocked_calls"]) == 0,
          "; ".join(zf["blocked_calls"][:5]) if zf["blocked_calls"] else "")

    # ONNX opset check (from ZIP)
    onnx_opsets = zf.get("onnx_opsets", {})
    for name, opset in onnx_opsets.items():
        check(group, f"ONNX opset <= 20 ({Path(name).name})",
              opset <= 20,
              f"opset_version={opset}")

    # Total file count
    file_count = len([f for f in zf["all_files"] if not f.endswith("/")])
    check(group, f"Total files in ZIP: {file_count} (<= 1000)",
          file_count <= 1000)

    # ---- Validate run.py content from ZIP ----
    try:
        with zipfile.ZipFile(submission_zip) as z:
            run_py_content = z.read("run.py").decode("utf-8", errors="replace")
    except Exception:
        run_py_content = ""

    if run_py_content:
        # Check --input and --output arguments
        check(group, "run.py accepts --input argument",
              "--input" in run_py_content)
        check(group, "run.py accepts --output argument",
              "--output" in run_py_content)

        # Check output format fields
        has_fields = all(f in run_py_content for f in
                         ["image_id", "category_id", "bbox", '"score"'])
        check(group, "Output JSON has required fields (image_id, category_id, bbox, score)",
              has_fields)

        # Uses pathlib
        check(group, "Uses pathlib (not os) for file operations",
              "pathlib" in run_py_content,
              "pathlib imported" if "pathlib" in run_py_content else "Missing")

        # Dry-run syntax check
        try:
            ast.parse(run_py_content)
            check(group, "run.py has valid Python syntax (AST-parsed)", True)
        except SyntaxError as e:
            check(group, "run.py has valid Python syntax", False,
                  f"Line {e.lineno}: {e.msg}")

    # ---- Validate ZIP can be extracted cleanly ----
    try:
        with tempfile.TemporaryDirectory() as tmp:
            with zipfile.ZipFile(submission_zip) as z:
                z.extractall(tmp)
            # Verify run.py actually exists after extraction
            extracted_run = Path(tmp) / "run.py"
            check(group, "run.py extractable and exists after unzip",
                  extracted_run.exists())

            # Check for path traversal attempts
            bad_paths = [f for f in zf["all_files"]
                         if ".." in f or f.startswith("/")]
            check(group, "No path traversal in ZIP entries",
                  len(bad_paths) == 0,
                  f"Suspicious: {bad_paths}" if bad_paths else "")

            # Check no symlinks after extraction
            symlinks = list(Path(tmp).rglob("*"))
            symlinks = [p for p in symlinks if p.is_symlink()]
            check(group, "No symlinks after extraction",
                  len(symlinks) == 0)
    except Exception as e:
        check(group, "ZIP extracts cleanly", False, str(e))


# ---- Report -----------------------------------------------------------------

def print_report():
    groups = {}
    for r in results:
        groups.setdefault(r.group, []).append(r)

    total = len(results)
    passed = sum(1 for r in results if r.status == "PASS")
    failed = sum(1 for r in results if r.status == "FAIL")
    warned = sum(1 for r in results if r.status == "WARN")

    print("=" * 70)
    print("NM i AI 2026 -- Strict Compliance Report")
    print("=" * 70)

    for gn in ("global", "tripletex", "astar", "norgesgruppen"):
        if gn not in groups:
            continue
        gr = groups[gn]
        gp = sum(1 for r in gr if r.status == "PASS")
        print(f"\n--- {gn.upper()} ({gp}/{len(gr)} passed) ---")
        for r in gr:
            print(r)

    print(f"\n{'=' * 70}")
    print(f"SUMMARY: {passed} passed, {failed} failed, {warned} warnings out of {total} checks")
    status = "FAIL" if failed else ("PASS (with warnings)" if warned else "PASS")
    print(f"STATUS: {status}")
    print("=" * 70)


def get_json_report():
    total = len(results)
    passed = sum(1 for r in results if r.status == "PASS")
    failed = sum(1 for r in results if r.status == "FAIL")
    warned = sum(1 for r in results if r.status == "WARN")
    return {
        "summary": {
            "total": total, "passed": passed, "failed": failed,
            "warnings": warned,
            "status": "FAIL" if failed else ("WARN" if warned else "PASS"),
        },
        "checks": [r.to_dict() for r in results],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    results.clear()
    check_global()
    check_tripletex()
    check_astar()
    check_norgesgruppen()

    if args.json:
        print(json.dumps(get_json_report(), indent=2))
    else:
        print_report()

    sys.exit(1 if any(r.status == "FAIL" for r in results) else 0)


if __name__ == "__main__":
    main()
