"""Diagnostic: success rate of several independent source seed sets on the same reliable select tasks."""
import json, os, sys
from chantless.pipeline import Run
from chantless.evaluator import ok_rows
cfg = json.load(open(sys.argv[1]))
R = Run(cfg); R.guard.phase = "aa"
sel = R.reliable(os.environ.get("SPLIT", "select"))
import os
if os.environ.get("SEED_RUN_ID"):
    R.cfg = dict(R.cfg, run_id=os.environ["SEED_RUN_ID"])
with R.ev.servers(R.src) as s:
    for tag in sys.argv[2:]:
        rows = ok_rows(R.ev.run(R.src, R.spread(tag, sel, cfg["stats"]["n_aa"], 12), "diag-" + tag, "src", save=False, srvs=s))
        print(tag, len(rows), round(sum(r["success"] for r in rows) / len(rows), 3), flush=True)
