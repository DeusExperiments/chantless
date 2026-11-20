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
import json, sys
try:
    import resource
    resource.setrlimit(resource.RLIMIT_CPU, (5, 5))
    if sys.platform.startswith("linux"):
        resource.setrlimit(resource.RLIMIT_AS, (1 << 30, 1 << 30))
    resource.setrlimit(resource.RLIMIT_FSIZE, (1 << 20, 1 << 20))
except Exception:
    pass
spec = json.load(open("spec.json"))
ns = {}
try:
    exec(compile(open("solution.py").read(), "solution.py", "exec"), ns)
except BaseException as e:
    print(json.dumps({"load_error": type(e).__name__ + ": " + str(e)[:200]})); sys.exit(0)
fn = ns.get(spec["fname"])
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
print(json.dumps({"results": res, "missing": fn is None}))
'''


def run_tests(code: str, fname: str, tests: list, timeout: float = 10.0) -> dict:
    """Return {'passed': bool, 'n_ok': int, 'n': int, 'results': [...], 'error': str|None}."""
    with tempfile.TemporaryDirectory(prefix="chantless-sbx-") as d:
        with open(os.path.join(d, "solution.py"), "w") as f:
            f.write(code)
        with open(os.path.join(d, "spec.json"), "w") as f:
            json.dump({"fname": fname, "tests": tests}, f)
        with open(os.path.join(d, "harness.py"), "w") as f:
            f.write(HARNESS)
        try:
            p = subprocess.run([sys.executable, "-I", "-S", "harness.py"], cwd=d, capture_output=True, text=True,
                               timeout=timeout, env={"PATH": "/usr/bin:/bin"})
        except subprocess.TimeoutExpired:
            return {"passed": False, "n_ok": 0, "n": len(tests), "results": [], "error": "timeout"}
        try:
            out = json.loads(p.stdout.strip().splitlines()[-1])
        except Exception:
            return {"passed": False, "n_ok": 0, "n": len(tests), "results": [], "error": "harness: " + p.stderr[-200:]}
        if "load_error" in out:
            return {"passed": False, "n_ok": 0, "n": len(tests), "results": [], "error": out["load_error"]}
        if out.get("missing"):
            return {"passed": False, "n_ok": 0, "n": len(tests), "results": [], "error": f"function {fname} not defined"}
        res = out["results"]
        n_ok = sum(r["ok"] for r in res)
        return {"passed": n_ok == len(tests), "n_ok": n_ok, "n": len(tests), "results": res, "error": None}
