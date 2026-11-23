"""Chantless CLI entrypoint."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
from pathlib import Path

from .pipeline import PHASES, Run


def _add_extract_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", required=True, help="extraction run configuration JSON/YAML")
    p.add_argument("--set", dest="overrides", action="append", default=[], help="repeatable key.path=value override")
    p.add_argument("--upto", choices=PHASES, help="stop after completing this phase")
    p.add_argument("--reset", choices=PHASES, help="clear this phase marker and all later ones")
    p.add_argument("--out-mask", help="optional path to export the resulting chantless.mask/v3 JSON")
    p.add_argument("--out", help="optional path to export the built specialist GGUF")
    p.add_argument("--cache-prompt", action=argparse.BooleanOptionalAction, default=None, help="override serving.cache_prompt from config")


def _load_validated(path: str, kind: str, overrides: list[str] | None):
    from pydantic import ValidationError
    from .config import load_config

    try:
        return load_config(path, kind, overrides=overrides)
    except (ValidationError, ValueError, FileNotFoundError) as e:
        raise SystemExit(f"config validation failed ({path}):\n{e}")


def main():
    ap = argparse.ArgumentParser(prog="chantless")
    sub = ap.add_subparsers(dest="cmd", required=True)

    ex = sub.add_parser("extract", help="extract a chantless.mask/v3 specialist mask under a byte budget and verify gates")
    _add_extract_args(ex)

    r = sub.add_parser("run", help="alias for `extract` (end-to-end pipeline execution)")
    _add_extract_args(r)

    bd = sub.add_parser("build", help="materialize a standalone GGUF deterministically from frozen source + thin mask")
    bd.add_argument("--source", required=True, help="frozen BF16/F16 source GGUF")
    bd.add_argument("--mask", required=True, help="chantless.mask/v3 (or v2/v1) JSON file")
    bd.add_argument("--out", required=True, help="output specialist GGUF path")
    bd.add_argument("--imatrix", help="importance-matrix GGUF (required when mask uses IQ tiers or width cuts with IQ)")
    bd.add_argument("--hf-dir", help="optional HF config directory for shared-KV metadata")
    bd.add_argument("--llama-bin", default="", help="directory containing llama-quantize")
    bd.add_argument("--threads", type=int, default=8)
    bd.add_argument("--shards", type=int, default=0, help="parallel llama-quantize CPU worker shards (0 = auto)")
    bd.add_argument("--work", help="deterministic scratch directory (defaults next to --out)")

    pl = sub.add_parser("plan", help="estimate moves, episodes, tokens, wall-clock and cost without running")
    pl.add_argument("--config", required=True)
    pl.add_argument("--set", dest="overrides", action="append", default=[])
    pl.add_argument("--rates", help="JSON {backend: {field: value}} overriding default backend rate estimates")
    pl.add_argument("--source", help="local copy of the source GGUF, read for its header in place of the config's path")

    ln = sub.add_parser("learn", help="adapt a built specialist from few-shot examples via neuron switches or grid updates")
    ln.add_argument("--config", required=True)
    ln.add_argument("--set", dest="overrides", action="append", default=[])
    ln.add_argument("--fold", type=int, help="hold this feedback fold out of training; default trains on all of it")
    ln.add_argument("--out-mask", help="optional path to export the updated chantless.mask/v3 JSON")
    ln.add_argument("--out", help="optional path to export the updated specialist GGUF")

    an = sub.add_parser("anneal", help="run staged quantization-aware training with progressive structural pruning")
    an.add_argument("--config", required=True)
    an.add_argument("--set", dest="overrides", action="append", default=[])
    an.add_argument("--stage", choices=("setup", "sweep", "pick", "arm", "judge", "report"), default="arm")
    an.add_argument("--arm", help="arm name")
    an.add_argument("--gpu", default="0", help="GPU index for scoring (isolate process via CUDA_VISIBLE_DEVICES)")
    an.add_argument("--lr", type=float, help="learning rate for --stage sweep")
    an.add_argument("--model", help="GGUF path for --stage judge (defaults to arm's config row)")

    sw = sub.add_parser("sweep", help="run a declarative multi-cell sweep across local GPUs or pluggable cloud launchers")
    sw.add_argument("--config", required=True, help="sweep configuration YAML/JSON")
    sw.add_argument("--set", dest="overrides", action="append", default=[])
    sw.add_argument("--launcher", help="override launcher backend (local, ssh, modal, colab, or registered custom)")
    sw.add_argument("--export-dir", help="override durable per-cell export directory")
    sw.add_argument("--dry-run", action="store_true", help="expand and validate all cells, print launch plan, do not execute")
    sw.add_argument("--no-resume", action="store_true", help="ignore existing per-cell done.json receipts")
    sw.add_argument("--stop-on-failure", action="store_true", help="halt scheduling remaining cells after first failure")

    sc = sub.add_parser("schema", help="export Pydantic v2 JSON Schema for Chantless config files")
    sc.add_argument("kind", nargs="?", default="all", choices=("extract", "learn", "sweep", "anneal", "all"))
    sc.add_argument("--out", help="optional output path for schema JSON")

    vl = sub.add_parser("validate", help="strictly validate a Chantless JSON/YAML config file (with extends and --set)")
    vl.add_argument("--config", required=True)
    vl.add_argument("--kind", default="extract", choices=("extract", "learn", "sweep", "anneal"))
    vl.add_argument("--set", dest="overrides", action="append", default=[])

    mkp = sub.add_parser("mask", help="inspect, diff, fork, merge, or pull chantless.mask/v3 thin masks")
    mk_sub = mkp.add_subparsers(dest="mask_cmd", required=True)

    mk_i = mk_sub.add_parser("inspect", help="summarize a chantless.mask/v3 JSON file")
    mk_i.add_argument("mask", help="path to mask JSON")

    mk_d = mk_sub.add_parser("diff", help="diff two chantless.mask/v3 JSON files")
    mk_d.add_argument("mask_a", help="first mask JSON")
    mk_d.add_argument("mask_b", help="second mask JSON")

    mk_f = mk_sub.add_parser("fork", help="clone a thin mask with parent provenance metadata")
    mk_f.add_argument("base", help="parent mask JSON")
    mk_f.add_argument("--out", required=True, help="child mask JSON path")
    mk_f.add_argument("--note", help="optional fork note")

    mk_m = mk_sub.add_parser("merge", help="merge two thin masks from the same frozen source checkpoint")
    mk_m.add_argument("mask_a", help="first mask JSON")
    mk_m.add_argument("mask_b", help="second mask JSON")
    mk_m.add_argument("--mode", default="apprentice", choices=("apprentice", "specialist"))
    mk_m.add_argument("--op", default="union", choices=("union", "intersection"), help="switch merge op for --mode apprentice")
    mk_m.add_argument("--out", required=True, help="merged mask JSON path")

    mk_p = mk_sub.add_parser("pull", help="fetch a remote thin mask (https:// or hf://) with SHA256 verification")
    mk_p.add_argument("uri", help="https://... or hf://<org>/<repo>/<path>[@<rev>]")
    mk_p.add_argument("--out", required=True, help="local destination JSON path")
    mk_p.add_argument("--sha256", help="expected SHA256 hex digest (required unless --allow-unpinned)")
    mk_p.add_argument("--allow-unpinned", action="store_true", help="allow pull without --sha256")
    mk_p.add_argument("--allow-http", action="store_true", help="allow plain http:// (for localhost verification)")

    mk_im = mk_sub.add_parser("import", help="convert an existing GGUF, tensor-type/audit recipe, or legacy spec into chantless.mask/v3")
    mk_im.add_argument("--from-gguf", help="existing mixed-precision GGUF whose header tensor types are imported")
    mk_im.add_argument("--from-tensor-types", help="tensor-audit.json, *_types.json, or llama-quantize tensor_types.txt")
    mk_im.add_argument("--from-spec", help="legacy spec JSON ({tiers: {group: tier}})")
    mk_im.add_argument("--source-gguf", help="optional frozen base GGUF for canonical group/drop/regex resolution")
    mk_im.add_argument("--hf-dir", help="optional HF config dir for Gemma 4 shared-KV metadata")
    mk_im.add_argument("--override-recipe", help="optional tensor-protection override JSON ({protect_qtype, protected: [...]})")
    mk_im.add_argument("--repo", default="", help="pinned source model repo id")
    mk_im.add_argument("--revision", default="", help="pinned source checkpoint revision")
    mk_im.add_argument("--source-sha256", default="", help="pinned base GGUF SHA256")
    mk_im.add_argument("--objective-mode", default="preserve", choices=("preserve", "improve"))
    mk_im.add_argument("--note", help="provenance note attached to the imported mask")
    mk_im.add_argument("--measured-json", help="optional JSON file of recorded evaluation/size measurements to attach")
    mk_im.add_argument("--out", required=True, help="output chantless.mask/v3 JSON path")

    def _add_serve_args(p: argparse.ArgumentParser, default_bench: bool) -> None:
        p.add_argument("--model", help="built standalone specialist GGUF")
        p.add_argument("--source", help="frozen stock BF16 GGUF (paired with --mask for auto-build cache)")
        p.add_argument("--mask", help="chantless.mask/v3 JSON (paired with --source)")
        p.add_argument("--imatrix", help="importance-matrix GGUF (required for IQ-tier masks)")
        p.add_argument("--hf-dir", help="optional HF config dir for Gemma 4 shared-KV metadata")
        p.add_argument("--cache-dir", help="auto-build cache dir (default ~/.cache/chantless/builds)")
        p.add_argument("--llama-bin", required=True, help="directory containing llama-server and llama-quantize")
        p.add_argument("--gpu", default="", help="CUDA_VISIBLE_DEVICES id (empty for Metal/CPU)")
        p.add_argument("--host", default="127.0.0.1")
        p.add_argument("--port", type=int, default=8080)
        p.add_argument("--slots", type=int, default=1 if default_bench else 4)
        p.add_argument("--ctx-per-slot", type=int, default=4096 if default_bench else 8192)
        p.add_argument("--kv-type", default="f16")
        p.add_argument("--alias", help="OpenAI API model alias")
        p.add_argument("--webui", action="store_true", help="enable llama-server WebUI (off by default)")
        p.add_argument("--special", action=argparse.BooleanOptionalAction, default=None, help="pass --special to llama-server (auto-detected for Inkling-2 when omitted; use --no-special to force off)")
        p.add_argument("--cache-prompt", action=argparse.BooleanOptionalAction, default=False, help="enable llama-server prompt caching (disabled by default)")
        p.add_argument("--flash-attn", default="on", choices=("on", "off", "auto"), help="llama-server flash attention mode (on, off, auto; default: on)")
        p.add_argument("--no-metrics", action="store_true", help="disable /metrics endpoint")
        p.add_argument("--log", help="optional llama-server log path")
        p.add_argument("--threads", type=int, default=8, help="quantize threads on auto-build cache miss")
        p.add_argument("--shards", type=int, default=0, help="parallel quantize shards on cache miss")
        p.add_argument("--bench", action="store_true", default=default_bench, help="run benchmark sweep and emit JSON telemetry receipt")
        p.add_argument("--prompt-tokens", type=int, default=256)
        p.add_argument("--gen-tokens", type=int, default=128)
        p.add_argument("--runs", type=int, default=3)
        p.add_argument("--compare", help="optional baseline GGUF for sequential side-by-side benchmark diff")
        p.add_argument("--out", help="optional JSON receipt output path for --bench")

    sv = sub.add_parser("serve", help="serve a built GGUF or thin mask (auto-build cached) with telemetry or --bench")
    _add_serve_args(sv, default_bench=False)

    bn = sub.add_parser("bench", help="benchmark a built GGUF or thin mask and emit JSON telemetry receipt")
    _add_serve_args(bn, default_bench=True)

    dr = sub.add_parser("doctor", help="validate llama.cpp@afe77e18 binaries, C++ tools, CUDA/Metal, Python deps, and /dev/shm")
    dr.add_argument("--config", help="optional extract/learn config JSON/YAML to read toolchain paths from")
    dr.add_argument("--llama-bin", help="directory containing llama-server, llama-quantize, llama-imatrix")
    dr.add_argument("--traj-kl", help="path to traj-kl binary")
    dr.add_argument("--logodds", help="path to logodds binary")
    dr.add_argument("--tokdocs", help="path to tokdocs binary")
    dr.add_argument("--run-root", default=".", help="directory to check disk headroom against")
    dr.add_argument("--model-gb", type=float, help="optional BF16 model size in GiB to preview scratch and tier-cache layout")
    dr.add_argument("--require-classifier", action="store_true", help="require logodds and tokdocs binaries")
    dr.add_argument("--require-cuda", action="store_true", help="require NVIDIA CUDA availability")
    dr.add_argument("--llama-commit", default=None, help="expected llama-server --version commit substring (default: afe77e1)")

    a = ap.parse_args()
    if a.cmd == "doctor":
        from . import config as cfgmod, doctor as drmod
        llama_bin, traj_kl, logodds, tokdocs, run_root = a.llama_bin, a.traj_kl, a.logodds, a.tokdocs, a.run_root
        if a.config:
            try:
                raw = cfgmod.load_raw_with_extends(a.config)
            except (FileNotFoundError, ValueError, OSError) as e:
                raise SystemExit(f"doctor failed reading config ({a.config}): {e}")
            tc = raw.get("toolchain") or {}
            llama_bin = llama_bin or tc.get("llama_bin")
            traj_kl = traj_kl or tc.get("traj_kl")
            logodds = logodds or tc.get("logodds")
            tokdocs = tokdocs or tc.get("tokdocs")
            if run_root == "." and raw.get("run_root"):
                run_root = raw["run_root"]
        res = drmod.run_doctor(
            llama_bin=llama_bin,
            traj_kl=traj_kl,
            logodds=logodds,
            tokdocs=tokdocs,
            run_root=run_root,
            model_gb=a.model_gb,
            require_classifier=a.require_classifier,
            require_cuda=a.require_cuda,
            llama_commit=a.llama_commit or drmod.PINNED_LLAMA_COMMIT,
        )
        print(json.dumps(res, indent=2))
        if not res["ok"]:
            raise SystemExit(1)
        return
    if a.cmd in ("serve", "bench"):
        from . import serve as svmod
        try:
            minfo = svmod.resolve_model(
                a.llama_bin,
                model=a.model,
                source=a.source,
                mask=a.mask,
                imatrix=a.imatrix,
                hf_dir=a.hf_dir,
                cache_dir=a.cache_dir,
                threads=a.threads,
                shards=a.shards,
            )
            if a.bench:
                out = svmod.run_bench(
                    a.llama_bin,
                    minfo,
                    compare_path=a.compare,
                    out_path=a.out,
                    gpu=a.gpu,
                    slots=a.slots,
                    ctx_per_slot=a.ctx_per_slot,
                    kv_type=a.kv_type,
                    prompt_tokens=a.prompt_tokens,
                    gen_tokens=a.gen_tokens,
                    runs=a.runs,
                    special=a.special,
                    cache_prompt=a.cache_prompt,
                    flash_attn=a.flash_attn,
                )
            else:
                out = svmod.serve_live(
                    a.llama_bin,
                    minfo,
                    host=a.host,
                    port=a.port,
                    gpu=a.gpu,
                    slots=a.slots,
                    ctx_per_slot=a.ctx_per_slot,
                    kv_type=a.kv_type,
                    alias=a.alias,
                    metrics=not a.no_metrics,
                    webui=a.webui,
                    log_path=a.log,
                    special=a.special,
                    cache_prompt=a.cache_prompt,
                    flash_attn=a.flash_attn,
                )
        except (ValueError, FileNotFoundError, RuntimeError) as e:
            raise SystemExit(f"{a.cmd} failed: {e}")
        print(json.dumps(out, indent=2))
        return
    if a.cmd == "mask":
        from . import mask as mk
        try:
            if a.mask_cmd == "inspect":
                out = mk.inspect_mask(a.mask)
            elif a.mask_cmd == "diff":
                out = mk.diff_masks(a.mask_a, a.mask_b)
            elif a.mask_cmd == "fork":
                out = mk.fork_mask(a.base, a.out, note=a.note)
            elif a.mask_cmd == "merge":
                out = mk.merge_masks(a.mask_a, a.mask_b, a.out, mode=a.mode, op=a.op)
            elif a.mask_cmd == "pull":
                out = mk.pull_mask(a.uri, a.out, expected_sha256=a.sha256, allow_unpinned=a.allow_unpinned, allow_http=a.allow_http)
            elif a.mask_cmd == "import":
                meas = json.loads(Path(a.measured_json).read_text(encoding="utf-8")) if a.measured_json else None
                out = mk.import_mask(
                    a.out,
                    from_gguf=a.from_gguf,
                    from_tensor_types=a.from_tensor_types,
                    from_spec=a.from_spec,
                    source_gguf=a.source_gguf,
                    hf_dir=a.hf_dir,
                    override_recipe=a.override_recipe,
                    repo=a.repo,
                    revision=a.revision,
                    source_sha256=a.source_sha256,
                    objective_mode=a.objective_mode,
                    note=a.note,
                    measured=meas,
                )
            else:
                raise SystemExit(f"unknown mask subcommand {a.mask_cmd!r}")
        except (mk.MaskError, FileNotFoundError, ValueError) as e:
            raise SystemExit(f"mask {a.mask_cmd} failed: {e}")
        print(json.dumps(out, indent=2))
        return
    if a.cmd == "schema":
        from .config import export_json_schema
        text = json.dumps(export_json_schema(a.kind), indent=2)
        if a.out:
            Path(a.out).parent.mkdir(parents=True, exist_ok=True)
            Path(a.out).write_text(text + "\n", encoding="utf-8")
        else:
            print(text)
        return

    if a.cmd == "validate":
        model = _load_validated(a.config, a.kind, a.overrides)
        if a.kind == "sweep":
            from .sweep import expand_cells
            cells = expand_cells(model, sweep_file=a.config)
            print(json.dumps({"valid": True, "kind": a.kind, "sweep_id": model.sweep_id, "cells": len(cells)}, indent=2))
        else:
            print(json.dumps({"valid": True, "kind": a.kind}, indent=2))
        return

    if a.cmd == "sweep":
        from .sweep import run_sweep
        sw_cfg = _load_validated(a.config, "sweep", a.overrides)
        res = run_sweep(
            sw_cfg,
            sweep_file=a.config,
            launcher_override=a.launcher,
            export_dir_override=a.export_dir,
            dry_run=a.dry_run,
            resume_override=False if a.no_resume else None,
            stop_on_failure_override=True if a.stop_on_failure else None,
        )
        print(json.dumps(res, indent=2))
        if not a.dry_run and not res.get("ok", False):
            raise SystemExit(1)
        return

    if a.cmd == "build":
        from . import builder as bmod, gguf_ops, mask as mk
        from .runtime import Toolchain
        groups, _ = gguf_ops.adapter_from_gguf(a.source, a.hf_dir)
        m = mk.load(a.mask)
        tc = Toolchain(Path(a.llama_bin), Path("/bin/true"), [""])
        out_p = Path(a.out).resolve()
        work_p = Path(a.work) if a.work else (out_p.parent / f".work-{out_p.stem}")
        created_work = not work_p.exists()
        try:
            res = bmod.Builder(tc, a.source, groups, a.imatrix, work_p, threads=a.threads, quantize_shards=a.shards).build(m, a.out)
        finally:
            if created_work and not a.work:
                shutil.rmtree(work_p, ignore_errors=True)
        print(json.dumps(res.to_dict(), indent=2))
        return

    overrides = list(getattr(a, "overrides", None) or [])
    if a.cmd in ("extract", "run") and getattr(a, "cache_prompt", None) is not None:
        overrides.append(f"serving.cache_prompt={'true' if a.cache_prompt else 'false'}")
    cfg = _load_validated(a.config, a.cmd, overrides or None).to_raw_dict()
    if a.cmd == "anneal":
        from .anneal import main as anneal_main
        anneal_main(cfg, a)
        return
    if a.cmd == "learn":
        from .learn import run as learn_run
        if getattr(a, "out_mask", None):
            cfg.setdefault("learn", {})["mask_out"] = a.out_mask
        res = learn_run(cfg, a.fold)
        if getattr(a, "out", None):
            tag = "all" if a.fold is None else f"fold{a.fold}"
            out_dir = Path(cfg["learn"]["out_dir"])
            for cand in (out_dir / f"apprentice-{tag}.gguf", out_dir / f"model-{tag}.gguf"):
                if cand.exists():
                    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(cand, a.out)
                    break
        if getattr(a, "out_mask", None) and res.get("mask_out") and Path(res["mask_out"]).resolve() != Path(a.out_mask).resolve():
            Path(a.out_mask).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(res["mask_out"], a.out_mask)
        return
    if a.cmd == "plan":
        from .plan import estimate, format_table
        try:
            est = estimate(cfg, json.loads(Path(a.rates).read_text()) if a.rates else None, a.source)
        except (FileNotFoundError, ValueError) as e:
            raise SystemExit(f"plan: {e}")
        print(format_table(est))
        return

    # `extract` / `run`: own process group so SIGTERM also stops llama-server / traj-kl / quantize children
    try:
        os.setpgrp()
    except OSError:
        pass
    signal.signal(signal.SIGTERM, lambda *_: os.killpg(os.getpgrp(), signal.SIGKILL))
    run = Run(cfg)
    if a.reset:
        for p in PHASES[PHASES.index(a.reset):]:
            (run.dir / "phases" / f"{p}.done").unlink(missing_ok=True)
    run.run_phases(a.upto)
    if getattr(a, "out_mask", None):
        for m_cand in (run.dir / "final" / "mask.json",):
            if m_cand.exists():
                Path(a.out_mask).parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(m_cand, a.out_mask)
                break
        else:
            sr_p = run.dir / "search_result.json"
            if sr_p.exists():
                Path(a.out_mask).parent.mkdir(parents=True, exist_ok=True)
                Path(a.out_mask).write_text(json.dumps(json.loads(sr_p.read_text())["mask"], indent=2))
    if getattr(a, "out", None):
        sp = run.dir / "final" / "specialist.gguf"
        if sp.exists():
            Path(a.out).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(sp, a.out)


if __name__ == "__main__":
    main()
