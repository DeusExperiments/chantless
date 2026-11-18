"""python -m chantless.cli run --config configs/X.json [--upto PHASE] [--reset PHASE]
python -m chantless.cli build --source BASE.gguf --mask MASK.json --out SPECIALIST.gguf [--imatrix IM.gguf] [--hf-dir DIR] [--llama-bin BIN] [--threads N]
python -m chantless.cli plan --config configs/X.json [--rates R.json] [--source MODEL.gguf]
python -m chantless.cli learn --config configs/X.json [--fold N]
python -m chantless.cli anneal --config configs/X.json [--stage setup|sweep|pick|arm|judge|report] [--arm NAME] [--gpu ID] [--lr RATE] [--model MODEL.gguf]"""
from __future__ import annotations

import argparse
import json
import os
import signal
from pathlib import Path

from .pipeline import PHASES, Run


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--config", required=True)
    r.add_argument("--upto", choices=PHASES)
    r.add_argument("--reset", choices=PHASES, help="clear this phase marker and all later ones")
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
    pl.add_argument("--rates", help="JSON {backend: {field: value}} merged over plan.RATES")
    pl.add_argument("--source", help="local copy of the source GGUF, read for its header in place of the config's path")
    ln = sub.add_parser("learn", help="teach a built model from labelled examples on its own grid, then judge the real file")
    ln.add_argument("--config", required=True)
    ln.add_argument("--fold", type=int, help="hold this feedback fold out of training; default trains on all of it")
    an = sub.add_parser("anneal", help="train the whole source while cuts arrive in steps, then judge the real files")
    an.add_argument("--config", required=True)
    an.add_argument("--stage", choices=("setup", "sweep", "pick", "arm", "judge", "report"), default="arm")
    an.add_argument("--arm", help="the arm to run, or the name a judged file goes under")
    an.add_argument("--gpu", default="0", help="the real GPU id, for the scorer; the caller isolates the process with CUDA_VISIBLE_DEVICES")
    an.add_argument("--lr", type=float, help="the rate for --stage sweep")
    an.add_argument("--model", help="the file for --stage judge; default the arm's entry in the config's rows")
    a = ap.parse_args()
    if a.cmd == "build":
        import shutil
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
    cfg = json.loads(Path(a.config).read_text())
    if a.cmd == "anneal":
        from .anneal import main as anneal_main
        anneal_main(cfg, a)
        return
    if a.cmd == "learn":
        from .learn import run as learn_run
        learn_run(cfg, a.fold)
        return
    if a.cmd == "plan":
        from .plan import estimate, format_table
        try:
            est = estimate(cfg, json.loads(Path(a.rates).read_text()) if a.rates else None, a.source)
        except (FileNotFoundError, ValueError) as e:
            raise SystemExit(f"plan: {e}")
        print(format_table(est))
        return
    # own process group: a SIGTERM to the run also stops its llama-server / traj-kl / quantize children
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


if __name__ == "__main__":
    main()
