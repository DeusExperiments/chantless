"""python -m chantless.cli run --config configs/X.json [--upto PHASE] [--reset PHASE]"""
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
    a = ap.parse_args()
    # own process group: a SIGTERM to the run also stops its llama-server / traj-kl / quantize children
    try:
        os.setpgrp()
    except OSError:
        pass
    signal.signal(signal.SIGTERM, lambda *_: os.killpg(os.getpgrp(), signal.SIGKILL))
    cfg = json.loads(Path(a.config).read_text())
    run = Run(cfg)
    if a.reset:
        for p in PHASES[PHASES.index(a.reset):]:
            (run.dir / "phases" / f"{p}.done").unlink(missing_ok=True)
    run.run_phases(a.upto)


if __name__ == "__main__":
    main()
