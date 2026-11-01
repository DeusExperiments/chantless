"""python -m chantless.cli run --config configs/X.json [--upto PHASE] [--reset PHASE]
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
