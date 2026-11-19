"""Chantless CLI:
  chantless extract --config configs/X.json [--upto PHASE] [--reset PHASE] [--out-mask MASK.json] [--out SPECIALIST.gguf]
  chantless build   --source BASE.gguf --mask MASK.json --out SPECIALIST.gguf [--imatrix IM.gguf] [--hf-dir DIR] [--llama-bin BIN] [--threads N] [--shards K]
  chantless learn   --config configs/X.json [--fold N] [--out-mask MASK.json] [--out SPECIALIST.gguf]
  chantless plan    --config configs/X.json [--rates R.json] [--source MODEL.gguf]
  chantless run     --config configs/X.json [--upto PHASE] [--reset PHASE]
  chantless anneal  --config configs/X.json [--stage setup|sweep|pick|arm|judge|report] [--arm NAME] [--gpu ID] [--lr RATE] [--model MODEL.gguf]
"""
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
    pl.add_argument("--rates", help="JSON {backend: {field: value}} merged over plan.RATES")
    pl.add_argument("--source", help="local copy of the source GGUF, read for its header in place of the config's path")

    ln = sub.add_parser("learn", help="teach a built specialist from <=25 examples via switches/grid moves and judge controls")
    ln.add_argument("--config", required=True)
    ln.add_argument("--set", dest="overrides", action="append", default=[])
    ln.add_argument("--fold", type=int, help="hold this feedback fold out of training; default trains on all of it")
    ln.add_argument("--out-mask", help="optional path to export the updated chantless.mask/v3 JSON")
    ln.add_argument("--out", help="optional path to export the updated specialist GGUF")

    an = sub.add_parser("anneal", help="train the whole source while cuts arrive in steps, then judge the real files")
    an.add_argument("--config", required=True)
    an.add_argument("--set", dest="overrides", action="append", default=[])
    an.add_argument("--stage", choices=("setup", "sweep", "pick", "arm", "judge", "report"), default="arm")
    an.add_argument("--arm", help="the arm to run, or the name a judged file goes under")
    an.add_argument("--gpu", default="0", help="the real GPU id, for the scorer; the caller isolates the process with CUDA_VISIBLE_DEVICES")
    an.add_argument("--lr", type=float, help="the rate for --stage sweep")
    an.add_argument("--model", help="the file for --stage judge; default the arm's entry in the config's rows")

    sw = sub.add_parser("sweep", help="run a declarative multi-cell sweep across local GPUs or pluggable cloud launchers")
    sw.add_argument("--config", required=True, help="sweep configuration YAML/JSON")
    sw.add_argument("--set", dest="overrides", action="append", default=[])
    sw.add_argument("--launcher", help="override launcher backend (local, ssh, modal, colab, or registered custom)")
    sw.add_argument("--export-dir", help="override durable per-cell export directory")
    sw.add_argument("--dry-run", action="store_true", help="expand and validate all cells, print launch plan, do not execute")
    sw.add_argument("--no-resume", action="store_true", help="ignore existing per-cell done.json receipts")
    sw.add_argument("--stop-on-failure", action="store_true", help="halt scheduling remaining cells after first failure")

    sc = sub.add_parser("schema", help="export Pydantic v2 JSON Schema for Chantless config files")
    sc.add_argument("kind", nargs="?", default="all", choices=("extract", "learn", "sweep", "all"))
    sc.add_argument("--out", help="optional output path for schema JSON")

    vl = sub.add_parser("validate", help="strictly validate a Chantless JSON/YAML config file (with extends and --set)")
    vl.add_argument("--config", required=True)
    vl.add_argument("--kind", default="extract", choices=("extract", "learn", "sweep", "anneal"))
    vl.add_argument("--set", dest="overrides", action="append", default=[])

    a = ap.parse_args()
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

    cfg = _load_validated(a.config, a.cmd, getattr(a, "overrides", None)).to_raw_dict()
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
