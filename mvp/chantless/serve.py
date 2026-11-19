from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import signal
import statistics
import subprocess
import tempfile
import threading
import time
import urllib.request
from pathlib import Path
from typing import Any

from . import builder as bmod, gguf_ops, mask as mk
from .runtime import LlamaServer, Toolchain
from .util import sha256_file


def default_cache_dir() -> Path:
    env = os.environ.get("CHANTLESS_CACHE_DIR", "").strip()
    if env:
        return Path(env).expanduser()
    return Path.home() / ".cache" / "chantless" / "builds"


def _mask_needs_imatrix(m: dict) -> bool:
    for e in (m.get("groups") or {}).values():
        if e.get("tier") in mk.IMATRIX_REQUIRED:
            return True
    for t in (m.get("tensors") or {}).values():
        if t in mk.IMATRIX_REQUIRED:
            return True
    return False


def resolve_model(
    llama_bin: str | Path,
    model: str | Path | None = None,
    source: str | Path | None = None,
    mask: str | Path | None = None,
    imatrix: str | Path | None = None,
    hf_dir: str | Path | None = None,
    cache_dir: str | Path | None = None,
    threads: int = 8,
    shards: int = 0,
) -> dict[str, Any]:
    if model and (source or mask):
        raise ValueError("pass either --model or (--source and --mask), not both")
    if model:
        mp = Path(model).resolve()
        if not mp.exists():
            raise FileNotFoundError(f"model not found: {mp}")
        return {
            "model_path": str(mp),
            "bytes": mp.stat().st_size,
            "sha256": sha256_file(mp),
            "cache_hit": None,
            "cache_key": None,
            "build_s": 0.0,
        }
    if not source or not mask:
        raise ValueError("either --model or both --source and --mask are required")

    src_p = Path(source).resolve()
    msk_p = Path(mask).resolve()
    if not src_p.exists():
        raise FileNotFoundError(f"source GGUF not found: {src_p}")
    if not msk_p.exists():
        raise FileNotFoundError(f"mask JSON not found: {msk_p}")
    m = mk.load(msk_p)
    if _mask_needs_imatrix(m) and not imatrix:
        raise ValueError(f"mask {msk_p.name} uses IQ tiers and requires --imatrix")

    src_sha = sha256_file(src_p)
    msk_sha = sha256_file(msk_p)
    im_sha = sha256_file(Path(imatrix).resolve()) if imatrix else "none"
    key = hashlib.sha256(f"{src_sha}:{msk_sha}:{im_sha}".encode("utf-8")).hexdigest()[:24]

    cdir = Path(cache_dir).expanduser().resolve() if cache_dir else default_cache_dir().resolve()
    entry_dir = cdir / key
    out_gguf = entry_dir / "specialist.gguf"
    meta_p = entry_dir / "meta.json"
    if out_gguf.exists() and meta_p.exists():
        meta = json.loads(meta_p.read_text(encoding="utf-8"))
        if meta.get("bytes") == out_gguf.stat().st_size and meta.get("sha256"):
            return {
                "model_path": str(out_gguf),
                "bytes": int(meta["bytes"]),
                "sha256": str(meta["sha256"]),
                "cache_hit": True,
                "cache_key": key,
                "build_s": 0.0,
            }

    entry_dir.mkdir(parents=True, exist_ok=True)
    tc = Toolchain(Path(llama_bin).resolve(), Path("/bin/true"), [""])
    groups, _ = gguf_ops.adapter_from_gguf(str(src_p), str(hf_dir) if hf_dir else None)
    work_p = entry_dir / ".work-build"
    tmp_gguf = entry_dir / "specialist.gguf.tmp"
    t0 = time.perf_counter()
    try:
        res = bmod.Builder(tc, str(src_p), groups, str(imatrix) if imatrix else None, work_p, threads=threads, quantize_shards=shards).build(m, tmp_gguf)
        os.replace(tmp_gguf, out_gguf)
    finally:
        shutil.rmtree(work_p, ignore_errors=True)
        tmp_gguf.unlink(missing_ok=True)
    dt = round(time.perf_counter() - t0, 3)
    out_sha = sha256_file(out_gguf)
    meta = {
        "cache_key": key,
        "source_path": str(src_p),
        "source_sha256": src_sha,
        "mask_path": str(msk_p),
        "mask_sha256": msk_sha,
        "imatrix_sha256": im_sha,
        "bytes": int(res.file_bytes),
        "sha256": out_sha,
        "build_s": dt,
    }
    tmp_m = meta_p.with_suffix(".json.tmp")
    tmp_m.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp_m, meta_p)
    return {
        "model_path": str(out_gguf),
        "bytes": int(res.file_bytes),
        "sha256": out_sha,
        "cache_hit": False,
        "cache_key": key,
        "build_s": dt,
    }


def _sample_pid_rss_mib(pid: int) -> float | None:
    try:
        r = subprocess.run(["ps", "-o", "rss=", "-p", str(pid)], capture_output=True, text=True, timeout=2)
        if r.returncode == 0 and r.stdout.strip():
            return round(int(r.stdout.strip()) / 1024.0, 2)
    except Exception:
        pass
    return None


def _sample_cuda_vram_mib(pid: int) -> float | None:
    if not shutil.which("nvidia-smi"):
        return None
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_gpu_memory", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=2,
        )
        if r.returncode != 0:
            return None
        total = 0.0
        found = False
        for line in r.stdout.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) == 2 and parts[0].isdigit() and int(parts[0]) == pid:
                try:
                    total += float(parts[1])
                    found = True
                except ValueError:
                    pass
        return round(total, 2) if found else None
    except Exception:
        return None


def _parse_server_buffers(log_text: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    dev_matches = re.findall(
        r"\|\s*-\s*((?:CUDA|ROCm|MTL|Vulkan)\d+[^|]*?)\s*\|\s*([0-9.]+)\s*=\s*([0-9.]+)\s*\+\s*\(\s*([0-9.]+)\s*=\s*([0-9.]+)\s*\+\s*([0-9.]+)\s*\+\s*([0-9.]+)\s*\)",
        log_text,
    )
    if dev_matches:
        dev_name, _tot, _free, self_m, mod_m, ctx_m, cmp_m = dev_matches[-1]
        out["device"] = dev_name.strip()
        out["device_self_mib"] = float(self_m)
        out["device_model_mib"] = float(mod_m)
        out["device_kv_mib"] = float(ctx_m)
        out["device_compute_mib"] = float(cmp_m)
    host_matches = re.findall(
        r"\|\s*-\s*Host\s*\|\s*([0-9.]+)\s*=\s*([0-9.]+)\s*\+\s*([0-9.]+)\s*\+\s*([0-9.]+)",
        log_text,
    )
    if host_matches:
        h_self, h_mod, h_ctx, h_cmp = host_matches[-1]
        out["host_self_mib"] = float(h_self)
        out["host_model_mib"] = float(h_mod)
        out["host_compute_mib"] = float(h_cmp)
    return out


def _make_bench_prompt(target_tokens: int) -> str:
    chunk = "The quick brown fox analyzes matrix quantization block strides and memory bandwidth across layers. "
    reps = max(1, (target_tokens + 15) // 16)
    return (chunk * reps).strip() + "\nSummarize the key systems bottleneck in plain prose:\n"


def _p95(vals: list[float]) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    idx = min(len(s) - 1, int(0.95 * (len(s) - 1) + 0.9999))
    return float(s[idx])


def bench_one(
    llama_bin: str | Path,
    model_info: dict[str, Any],
    gpu: str = "",
    slots: int = 1,
    ctx_per_slot: int = 4096,
    kv_type: str = "f16",
    prompt_tokens: int = 256,
    gen_tokens: int = 128,
    runs: int = 3,
    extra: list[str] | None = None,
) -> dict[str, Any]:
    tc = Toolchain(Path(llama_bin).resolve(), Path("/bin/true"), [gpu])
    prompt = _make_bench_prompt(prompt_tokens)
    with tempfile.TemporaryDirectory() as td:
        log_p = Path(td) / "server.log"
        peak_rss = 0.0
        peak_vram: float | None = None
        stop_ev = threading.Event()

        def _poll(pid: int) -> None:
            nonlocal peak_rss, peak_vram
            while not stop_ev.is_set():
                rss = _sample_pid_rss_mib(pid)
                if rss is not None and rss > peak_rss:
                    peak_rss = rss
                vram = _sample_cuda_vram_mib(pid)
                if vram is not None and (peak_vram is None or vram > peak_vram):
                    peak_vram = vram
                stop_ev.wait(0.15)

        bench_extra = list(extra or [])
        if "-lv" not in bench_extra:
            bench_extra += ["-lv", "4"]
        t_boot0 = time.perf_counter()
        with LlamaServer(
            tc,
            model_info["model_path"],
            gpu=gpu,
            slots=slots,
            ctx_per_slot=ctx_per_slot,
            kv_type=kv_type,
            log=log_p,
            extra=bench_extra,
            metrics=True,
            cache_prompt=False,
            webui=False,
        ) as srv:
            boot_s = round(time.perf_counter() - t_boot0, 3)
            pid = srv.proc.pid if srv.proc else -1
            poller = threading.Thread(target=_poll, args=(pid,), daemon=True)
            if pid > 0:
                poller.start()
            try:
                body = {
                    "prompt": prompt,
                    "max_tokens": gen_tokens,
                    "temperature": 0.0,
                    "seed": 42,
                    "cache_prompt": False,
                }
                _ = srv.post("/v1/completions", {**body, "max_tokens": min(16, gen_tokens)})
                prefill_tps: list[float] = []
                decode_tps: list[float] = []
                ttft_ms: list[float] = []
                prompt_n_vals: list[int] = []
                pred_n_vals: list[int] = []
                wall_ms_vals: list[float] = []

                for _ in range(runs):
                    t0 = time.perf_counter()
                    resp = srv.post("/v1/completions", body)
                    dt_ms = (time.perf_counter() - t0) * 1000.0
                    wall_ms_vals.append(dt_ms)
                    tim = resp.get("timings") or {}
                    p_n = int(tim.get("prompt_n") or 0)
                    p_ms = float(tim.get("prompt_ms") or 0.0)
                    g_n = int(tim.get("predicted_n") or 0)
                    g_ms = float(tim.get("predicted_ms") or 0.0)
                    p_tps = float(tim.get("prompt_per_second") or (1000.0 * p_n / p_ms if p_ms > 0 else 0.0))
                    g_tps = float(tim.get("predicted_per_second") or (1000.0 * g_n / g_ms if g_ms > 0 else 0.0))
                    first_tok_ms = p_ms + (g_ms / g_n if g_n > 0 else 0.0)
                    prompt_n_vals.append(p_n)
                    pred_n_vals.append(g_n)
                    prefill_tps.append(p_tps)
                    decode_tps.append(g_tps)
                    ttft_ms.append(first_tok_ms if first_tok_ms > 0 else dt_ms)
            finally:
                stop_ev.set()
                if pid > 0:
                    poller.join(timeout=2)

        log_txt = log_p.read_text(encoding="utf-8", errors="ignore") if log_p.exists() else ""
        bufs = _parse_server_buffers(log_txt)

    return {
        "model_path": model_info["model_path"],
        "bytes": model_info["bytes"],
        "sha256": model_info["sha256"],
        "cache_hit": model_info["cache_hit"],
        "cache_key": model_info["cache_key"],
        "build_s": model_info["build_s"],
        "boot_s": boot_s,
        "config": {"gpu": gpu, "slots": slots, "ctx_per_slot": ctx_per_slot, "kv_type": kv_type, "runs": runs, "gen_tokens": gen_tokens},
        "prompt_tokens_measured": int(statistics.median(prompt_n_vals)) if prompt_n_vals else 0,
        "gen_tokens_measured": int(statistics.median(pred_n_vals)) if pred_n_vals else 0,
        "prefill_tok_s_median": round(statistics.median(prefill_tps), 2),
        "prefill_tok_s_p95": round(_p95(prefill_tps), 2),
        "decode_tok_s_median": round(statistics.median(decode_tps), 2),
        "decode_tok_s_p95": round(_p95(decode_tps), 2),
        "ttft_ms_median": round(statistics.median(ttft_ms), 2),
        "ttft_ms_p95": round(_p95(ttft_ms), 2),
        "wall_ms_median": round(statistics.median(wall_ms_vals), 2),
        "peak_vram_mib": peak_vram,
        "peak_rss_mib": round(peak_rss, 2) if peak_rss > 0 else None,
        "buffers": bufs,
    }


def run_bench(
    llama_bin: str | Path,
    model_info: dict[str, Any],
    compare_path: str | Path | None = None,
    out_path: str | Path | None = None,
    **bench_kwargs,
) -> dict[str, Any]:
    spec = bench_one(llama_bin, model_info, **bench_kwargs)
    receipt: dict[str, Any] = {"mode": "bench", "specialist": spec}
    if compare_path:
        base_info = resolve_model(llama_bin, model=compare_path)
        base = bench_one(llama_bin, base_info, **bench_kwargs)
        receipt["baseline"] = base
        s_dev = spec["buffers"].get("device_self_mib")
        b_dev = base["buffers"].get("device_self_mib")
        receipt["delta"] = {
            "size_ratio": round(spec["bytes"] / max(1, base["bytes"]), 4),
            "size_pct_of_baseline": round(100.0 * spec["bytes"] / max(1, base["bytes"]), 2),
            "prefill_speedup": round(spec["prefill_tok_s_median"] / max(1e-6, base["prefill_tok_s_median"]), 3),
            "decode_speedup": round(spec["decode_tok_s_median"] / max(1e-6, base["decode_tok_s_median"]), 3),
            "ttft_ratio": round(spec["ttft_ms_median"] / max(1e-6, base["ttft_ms_median"]), 3),
            "vram_ratio": round(spec["peak_vram_mib"] / base["peak_vram_mib"], 3) if (spec["peak_vram_mib"] and base["peak_vram_mib"]) else None,
            "device_mem_ratio": round(s_dev / b_dev, 3) if (s_dev and b_dev) else None,
            "rss_ratio": round(spec["peak_rss_mib"] / base["peak_rss_mib"], 3) if (spec["peak_rss_mib"] and base["peak_rss_mib"]) else None,
        }
    if out_path:
        op = Path(out_path)
        op.parent.mkdir(parents=True, exist_ok=True)
        tmp = op.with_suffix(op.suffix + ".tmp")
        tmp.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
        os.replace(tmp, op)
    return receipt


def serve_live(
    llama_bin: str | Path,
    model_info: dict[str, Any],
    host: str = "127.0.0.1",
    port: int = 8080,
    gpu: str = "",
    slots: int = 4,
    ctx_per_slot: int = 8192,
    kv_type: str = "f16",
    alias: str | None = None,
    metrics: bool = True,
    webui: bool = False,
    log_path: str | Path | None = None,
    extra: list[str] | None = None,
) -> dict[str, Any]:
    tc = Toolchain(Path(llama_bin).resolve(), Path("/bin/true"), [gpu])
    stop_ev = threading.Event()
    prev_int = signal.getsignal(signal.SIGINT)
    prev_term = signal.getsignal(signal.SIGTERM)

    def _handle(signum, frame):
        stop_ev.set()

    signal.signal(signal.SIGINT, _handle)
    signal.signal(signal.SIGTERM, _handle)

    t0 = time.perf_counter()
    peak_rss = 0.0
    peak_vram: float | None = None
    try:
        with LlamaServer(
            tc,
            model_info["model_path"],
            gpu=gpu,
            slots=slots,
            ctx_per_slot=ctx_per_slot,
            kv_type=kv_type,
            log=log_path,
            extra=extra,
            host=host,
            port=port,
            metrics=metrics,
            alias=alias,
            cache_prompt=True,
            webui=webui,
        ) as srv:
            pid = srv.proc.pid if srv.proc else -1
            startup = {
                "status": "ready",
                "url": srv.url,
                "openai_base_url": f"{srv.url}/v1",
                "metrics_url": f"{srv.url}/metrics" if metrics else None,
                "pid": pid,
                "model_path": model_info["model_path"],
                "bytes": model_info["bytes"],
                "sha256": model_info["sha256"],
                "cache_hit": model_info["cache_hit"],
                "alias": alias,
            }
            print(json.dumps(startup), flush=True)
            while not stop_ev.is_set():
                if srv.proc and srv.proc.poll() is not None:
                    break
                if pid > 0:
                    rss = _sample_pid_rss_mib(pid)
                    if rss is not None and rss > peak_rss:
                        peak_rss = rss
                    vram = _sample_cuda_vram_mib(pid)
                    if vram is not None and (peak_vram is None or vram > peak_vram):
                        peak_vram = vram
                stop_ev.wait(0.5)
    finally:
        signal.signal(signal.SIGINT, prev_int)
        signal.signal(signal.SIGTERM, prev_term)

    return {
        "status": "stopped",
        "uptime_s": round(time.perf_counter() - t0, 3),
        "peak_vram_mib": peak_vram,
        "peak_rss_mib": round(peak_rss, 2) if peak_rss > 0 else None,
    }
