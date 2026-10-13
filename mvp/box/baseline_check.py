"""Shakedown check of exam baselines + report on a COPY of a finished smoke run (not an accepted result)."""
import json, sys
from chantless.pipeline import Run
from chantless.util import read_jsonl
cfg = json.load(open(sys.argv[1])); cfg["run_id"] = sys.argv[2]
R = Run(cfg); R.guard.phase = "exam"
per = {}
for r in read_jsonl(R.dir / "episodes" / "exam-pool.jsonl"):
    per.setdefault(r["task"], []).append(r["success"])
R.exam_pool = [R.by_id[t] for t, xs in per.items() if len(xs) >= 7 and sum(bool(x) for x in xs) / len(xs) >= 0.75]
R.exam_baselines(str(R.dir / "final" / "specialist.gguf"))
R.phase_report()
print(open(R.dir / "report.md").read())
