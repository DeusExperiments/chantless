"""python -m chantless.cli run --config configs/X.json [--upto PHASE] [--reset PHASE]
python -m chantless.cli plan --config configs/X.json [--rates R.json] [--source MODEL.gguf]"""
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
    a = ap.parse_args()
    cfg = json.loads(Path(a.config).read_text())
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
