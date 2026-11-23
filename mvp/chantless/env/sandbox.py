"""Run model-generated code against tests in an isolated subprocess.

Isolation: fresh temp dir as cwd, `python -I -S`, wall-clock timeout, CPU/memory rlimits where
the OS supports them, no inherited environment. Process isolation only (not a security boundary).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile

HARNESS = r'''
import json, os, sys
try:
    import resource
    resource.setrlimit(resource.RLIMIT_CPU, (5, 5))
    if sys.platform.startswith("linux"):
        resource.setrlimit(resource.RLIMIT_AS, (1 << 30, 1 << 30))
    resource.setrlimit(resource.RLIMIT_FSIZE, (1 << 20, 1 << 20))
except Exception:
    pass
_RES_PATH = os.path.abspath("result.json")
def _write_result(obj):
    with open(_RES_PATH, "w", encoding="utf-8") as f:
        json.dump(obj, f)
with open("spec.json", encoding="utf-8") as f:
    spec = json.load(f)
with open("solution.py", encoding="utf-8") as f:
    src = f.read()
ns = {}
try:
    exec(compile(src, "solution.py", "exec"), ns)
except BaseException as e:
    _write_result({"load_error": type(e).__name__ + ": " + str(e)[:200]}); sys.exit(0)
fn = ns.get(spec["fname"])
if fn is None:
    _write_result({"results": [], "missing": True}); sys.exit(0)
if not callable(fn):
    _write_result({"load_error": f"{spec['fname']} is not callable ({type(fn).__name__})"}); sys.exit(0)
res = []
for args, expected in spec["tests"]:
    try:
        got = fn(*args)
        try:
            got = json.loads(json.dumps(got))
        except Exception:
            got = repr(got)
        res.append({"ok": got == expected, "got": repr(got)[:120]})
    except BaseException as e:
        res.append({"ok": False, "error": type(e).__name__ + ": " + str(e)[:120]})
_write_result({"results": res, "missing": False})
'''


def run_tests(code: str, fname: str, tests: list, timeout: float = 10.0) -> dict:
    """Return {'passed': bool, 'n_ok': int, 'n': int, 'results': [...], 'error': str|None}."""
    with tempfile.TemporaryDirectory(prefix="chantless-sbx-") as d:
        with open(os.path.join(d, "solution.py"), "w", encoding="utf-8") as f:
            f.write(code)
        with open(os.path.join(d, "spec.json"), "w", encoding="utf-8") as f:
            json.dump({"fname": fname, "tests": tests}, f)
        with open(os.path.join(d, "harness.py"), "w", encoding="utf-8") as f:
            f.write(HARNESS)
        try:
            p = subprocess.run([sys.executable, "-I", "-S", "harness.py"], cwd=d, capture_output=True, text=True,
                               timeout=timeout, env={"PATH": "/usr/bin:/bin"})
        except subprocess.TimeoutExpired:
            return {"passed": False, "n_ok": 0, "n": len(tests), "results": [], "error": "timeout"}
        result_path = os.path.join(d, "result.json")
        if not os.path.exists(result_path):
            err_tail = (p.stderr or "").strip()[-200:]
            prefix = f"harness crashed (rc={p.returncode})" if p.returncode != 0 else "harness produced no result.json"
            msg = f"{prefix}: {err_tail}" if err_tail else prefix
            return {"passed": False, "n_ok": 0, "n": len(tests), "results": [], "error": msg}
        try:
            with open(result_path, encoding="utf-8") as f:
                out = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            return {"passed": False, "n_ok": 0, "n": len(tests), "results": [], "error": f"harness invalid result.json: {type(e).__name__}: {e}"}
        if not isinstance(out, dict):
            return {"passed": False, "n_ok": 0, "n": len(tests), "results": [], "error": "harness invalid result.json: non-object"}
        if "load_error" in out:
            return {"passed": False, "n_ok": 0, "n": len(tests), "results": [], "error": out["load_error"]}
        if out.get("missing"):
            return {"passed": False, "n_ok": 0, "n": len(tests), "results": [], "error": f"function {fname} not defined"}
        res = out.get("results")
        if not isinstance(res, list) or len(res) != len(tests):
            return {"passed": False, "n_ok": 0, "n": len(tests), "results": [], "error": "harness result count mismatch"}
        n_ok = sum(1 for r in res if isinstance(r, dict) and r.get("ok"))
        return {"passed": n_ok == len(tests), "n_ok": n_ok, "n": len(tests), "results": res, "error": None}
