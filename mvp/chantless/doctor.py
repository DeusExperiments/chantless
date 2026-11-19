from __future__ import annotations

import importlib
import importlib.metadata
import importlib.util
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

PINNED_LLAMA_COMMIT = "a01dff9"
REQUIRED_PY_DEPS = ("torch", "transformers", "gguf", "pydantic", "yaml", "numpy", "scipy", "safetensors")
LLAMA_BINARIES = ("llama-server", "llama-quantize", "llama-imatrix")


def _env_with_lib(llama_bin: Path | None) -> dict[str, str]:
    env = dict(os.environ)
    if llama_bin and llama_bin.is_dir():
        ld = str(llama_bin)
        cur = env.get("LD_LIBRARY_PATH", "")
        if ld not in cur.split(":"):
            env["LD_LIBRARY_PATH"] = f"{ld}:{cur}" if cur else ld
    return env


def _run_probe(cmd: list[str], env: dict[str, str] | None = None, timeout: int = 10) -> tuple[int, str]:
    try:
        r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env, timeout=timeout)
        return r.returncode, (r.stdout or "").strip()
    except (OSError, subprocess.SubprocessError) as e:
        return -1, str(e)


def _check_llama_bin(llama_bin: str | Path | None) -> dict[str, Any]:
    if not llama_bin:
        return {"status": "fail", "required": True, "detail": "llama_bin not provided"}
    bdir = Path(llama_bin)
    if not bdir.is_dir():
        return {"status": "fail", "required": True, "path": str(bdir), "detail": "directory not found"}
    env = _env_with_lib(bdir)
    bins: dict[str, Any] = {}
    missing: list[str] = []
    for name in LLAMA_BINARIES:
        p = bdir / name
        ok = p.is_file() and os.access(p, os.X_OK)
        bins[name] = {"path": str(p), "executable": ok}
        if not ok:
            missing.append(name)
    if missing:
        return {"status": "fail", "required": True, "path": str(bdir), "binaries": bins, "detail": f"missing executable(s): {missing}"}

    rc_v, out_v = _run_probe([str(bdir / "llama-server"), "--version"], env=env)
    commit_ok = rc_v == 0 and PINNED_LLAMA_COMMIT in out_v
    rc_h, out_h = _run_probe([str(bdir / "llama-quantize"), "--help"], env=env)
    ttf_ok = "--tensor-type-file" in out_h

    status = "pass" if (commit_ok and ttf_ok) else "fail"
    details: list[str] = []
    if not commit_ok:
        details.append(f"llama-server --version missing {PINNED_LLAMA_COMMIT} (rc={rc_v})")
    if not ttf_ok:
        details.append("llama-quantize --help missing --tensor-type-file")
    return {
        "status": status,
        "required": True,
        "path": str(bdir),
        "binaries": bins,
        "version_line": out_v.splitlines()[0] if out_v else "",
        "pinned_commit_match": commit_ok,
        "tensor_type_file_support": ttf_ok,
        "detail": "; ".join(details) if details else "ok",
    }


def _check_tool(name: str, path: str | Path | None, required: bool, llama_bin: Path | None) -> dict[str, Any]:
    if not path:
        return {
            "status": "fail" if required else "warn",
            "required": required,
            "path": None,
            "detail": f"{name} not configured",
        }
    p = Path(path)
    ok = p.is_file() and os.access(p, os.X_OK)
    if not ok:
        return {
            "status": "fail" if required else "warn",
            "required": required,
            "path": str(p),
            "detail": "not found or not executable",
        }
    return {"status": "pass", "required": required, "path": str(p), "detail": "ok"}


def _query_nvidia_smi() -> list[dict[str, Any]]:
    if not shutil.which("nvidia-smi"):
        return []
    rc, out = _run_probe([
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,memory.free,compute_cap",
        "--format=csv,noheader,nounits",
    ])
    if rc != 0 or not out:
        return []
    gpus: list[dict[str, Any]] = []
    for line in out.splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) < 5:
            continue
        try:
            gpus.append({
                "index": int(parts[0]),
                "name": parts[1],
                "memory_total_mib": int(float(parts[2])),
                "memory_free_mib": int(float(parts[3])),
                "compute_cap": parts[4],
            })
        except ValueError:
            continue
    return gpus


def _check_accelerator(require_cuda: bool = False) -> dict[str, Any]:
    nvidia_gpus = _query_nvidia_smi()
    torch_cuda = False
    torch_cuda_count = 0
    torch_cuda_version = None
    torch_mps = False
    try:
        import torch
        torch_cuda = bool(torch.cuda.is_available())
        torch_cuda_count = int(torch.cuda.device_count()) if torch_cuda else 0
        torch_cuda_version = torch.version.cuda
        torch_mps = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
    except Exception:
        pass

    fla_spec = importlib.util.find_spec("fla")
    fla_ver = None
    if fla_spec is not None:
        try:
            fla_ver = importlib.metadata.version("flash-linear-attention")
        except importlib.metadata.PackageNotFoundError:
            fla_ver = "installed"

    backend = "cuda" if (torch_cuda or nvidia_gpus) else ("metal" if torch_mps else "cpu")
    if require_cuda and not (torch_cuda and nvidia_gpus):
        status = "fail"
        detail = "NVIDIA CUDA required but not available in both nvidia-smi and torch.cuda"
    elif backend == "cpu":
        status = "warn"
        detail = "no CUDA or Metal accelerator detected (CPU-only)"
    elif nvidia_gpus and not torch_cuda:
        status = "warn"
        detail = "nvidia-smi sees GPUs but torch.cuda.is_available() is False"
    else:
        status = "pass"
        detail = f"{backend} ready"

    return {
        "status": status,
        "required": require_cuda,
        "backend": backend,
        "nvidia_smi_gpus": nvidia_gpus,
        "torch_cuda_available": torch_cuda,
        "torch_cuda_count": torch_cuda_count,
        "torch_cuda_version": torch_cuda_version,
        "torch_mps_available": torch_mps,
        "fla": {"installed": fla_spec is not None, "version": fla_ver, "status": "pass" if fla_spec else "warn"},
        "detail": detail,
    }


def _check_python_deps() -> dict[str, Any]:
    pkgs: dict[str, Any] = {}
    missing: list[str] = []
    for mod_name in REQUIRED_PY_DEPS:
        try:
            mod = importlib.import_module(mod_name)
            ver = getattr(mod, "__version__", None)
            if ver is None:
                dist = "PyYAML" if mod_name == "yaml" else mod_name
                try:
                    ver = importlib.metadata.version(dist)
                except importlib.metadata.PackageNotFoundError:
                    ver = "installed"
            pkgs[mod_name] = {"installed": True, "version": str(ver)}
        except Exception as e:
            pkgs[mod_name] = {"installed": False, "error": str(e)}
            missing.append(mod_name)
    return {
        "status": "pass" if not missing else "fail",
        "required": True,
        "python": sys.version.split()[0],
        "packages": pkgs,
        "detail": "ok" if not missing else f"missing python package(s): {missing}",
    }


def _check_storage(run_root: str | Path, model_gb: float | None = None, n_gpus: int = 1) -> dict[str, Any]:
    rr = Path(run_root).resolve()
    probe_rr = rr if rr.exists() else rr.parent
    while not probe_rr.exists() and probe_rr != probe_rr.parent:
        probe_rr = probe_rr.parent
    du_rr = shutil.disk_usage(probe_rr)
    shm_p = Path("/dev/shm")
    shm_info: dict[str, Any] = {"exists": shm_p.exists()}
    shm_free = 0
    if shm_p.exists():
        du_s = shutil.disk_usage(shm_p)
        shm_free = du_s.free
        shm_info.update({
            "total_bytes": du_s.total,
            "free_bytes": du_s.free,
            "total_gib": round(du_s.total / (1 << 30), 2),
            "free_gib": round(du_s.free / (1 << 30), 2),
        })

    out: dict[str, Any] = {
        "status": "pass",
        "required": False,
        "run_root": {
            "path": str(rr),
            "total_bytes": du_rr.total,
            "free_bytes": du_rr.free,
            "total_gib": round(du_rr.total / (1 << 30), 2),
            "free_gib": round(du_rr.free / (1 << 30), 2),
        },
        "dev_shm": shm_info,
    }
    if model_gb is not None and model_gb > 0:
        from . import pipeline as pipe
        src_b = int(model_gb * (1 << 30))
        sc_dir, stage = pipe.choose_scratch_layout("auto", src_b, rr / "doctor-probe", "doctor-probe", shm_free=shm_free)
        sc_free = shm_free if str(sc_dir).startswith("/dev/shm") else du_rr.free
        w_eff, w_cap, _ = pipe.clamp_probe_workers(None, sc_dir, src_b, n_gpus=max(1, n_gpus), free_bytes=sc_free)
        tc_dir, tc_est, tc_where = pipe.choose_tier_cache_dir(
            sc_dir, rr / "doctor-probe", src_b, ["Q4_K", "Q3_K", "Q2_K"], shm_free=shm_free, nvme_free=du_rr.free
        )
        out["model_layout_preview"] = {
            "model_gib": model_gb,
            "scratch": str(sc_dir),
            "stage_source": stage,
            "probe_workers_auto": w_eff,
            "probe_workers_cap": w_cap,
            "tier_cache_location": tc_where,
            "tier_cache_dir": str(tc_dir) if tc_dir else None,
            "tier_cache_est_gib": round(tc_est / (1 << 30), 2),
        }
    return out


def run_doctor(
    llama_bin: str | Path | None = None,
    traj_kl: str | Path | None = None,
    logodds: str | Path | None = None,
    tokdocs: str | Path | None = None,
    run_root: str | Path = ".",
    model_gb: float | None = None,
    require_classifier: bool = False,
    require_cuda: bool = False,
) -> dict[str, Any]:
    lbin = Path(llama_bin) if llama_bin else None
    accel = _check_accelerator(require_cuda=require_cuda)
    n_gpus = max(1, accel["torch_cuda_count"] or len(accel["nvidia_smi_gpus"]) or 1)
    checks: dict[str, Any] = {
        "llama_bin": _check_llama_bin(lbin),
        "traj_kl": _check_tool("traj-kl", traj_kl, required=True, llama_bin=lbin),
        "logodds": _check_tool("logodds", logodds, required=require_classifier or bool(logodds), llama_bin=lbin),
        "tokdocs": _check_tool("tokdocs", tokdocs, required=require_classifier or bool(tokdocs), llama_bin=lbin),
        "accelerator": accel,
        "python_deps": _check_python_deps(),
        "storage": _check_storage(run_root, model_gb=model_gb, n_gpus=n_gpus),
    }
    ok = all(c["status"] != "fail" for c in checks.values())
    return {
        "ok": ok,
        "pinned_llama_commit": PINNED_LLAMA_COMMIT,
        "platform": {
            "system": platform.system(),
            "machine": platform.machine(),
            "release": platform.release(),
        },
        "checks": checks,
    }
