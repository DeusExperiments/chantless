"""AI caught at 1% human false positives, per model, with bootstrap intervals and the win-or-stop verdict.

Usage: python detect/measure.py SCORES [--boot 1000]
SCORES holds one folder per model (qwen, modernbert, proselens) of score.py shard files; writes SCORES/measure.json.
Cutoff: a model flags a document when p_human < t, with t set so 1% of the calibration humans are flagged.
Reported on WildOutlines test and on Beemo (all AI, and each Beemo kind): the share of AI documents caught,
and the share of humans flagged, which should sit near 1%. Only documents every model scored are counted.
Intervals: 2.5th to 97.5th percentile over bootstrap resamples of documents, the same resamples for every model.
Verdict: the 0.8B wins only if, on both test and Beemo, its interval for AI caught lies wholly above ModernBERT-L's.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from common import LICENCE_NOTE, read_jsonl, write_json

ap = argparse.ArgumentParser()
ap.add_argument("scores", type=Path)
ap.add_argument("--boot", type=int, default=1000)
ap.add_argument("--models", default="qwen,modernbert,proselens", help="contender first; it is judged against modernbert")
args = ap.parse_args()
models = [m for m in args.models.split(",") if (args.scores / m).is_dir()]
rng = np.random.default_rng(0)


def load(model, stem):
    return {r["id"]: r for p in sorted((args.scores / model).glob(f"{stem}.shard*.jsonl")) for r in read_jsonl(p)}


cut = {}
for m in models:
    cal = np.sort([r["p_human"] for r in load(m, "calibration").values() if r["label"] == "human"])
    cut[m] = float(cal[int(0.01 * len(cal))])  # strictly below this: the lowest 1%

results = {}
for stem in ("test", "beemo"):
    data = {m: load(m, stem) for m in models}
    ids = sorted(set.intersection(*(set(d) for d in data.values())))
    first = data[models[0]]
    ai = np.array([first[i]["label"] == "ai" for i in ids])
    kind = np.array([first[i]["kind"] for i in ids])
    groups = {"ai": ai} | ({k: kind == k for k in ("generated", "expert-edited")} if stem == "beemo" else {})
    boot = rng.integers(0, len(ids), (args.boot, len(ids)))
    results[stem] = {"docs": len(ids), "unmatched": {m: len(data[m]) - len(ids) for m in models}}
    for m in models:
        flag = np.array([data[m][i]["p_human"] for i in ids]) < cut[m]
        res = {"human_flagged": float(flag[~ai].mean())}
        for g, sel in groups.items():
            fb, sb = flag[boot], sel[boot]
            dist = (fb & sb).sum(1) / np.maximum(sb.sum(1), 1)
            res[g] = {"caught": float(flag[sel].mean()), "ci": [float(np.percentile(dist, 2.5)),
                                                                 float(np.percentile(dist, 97.5))]}
        results[stem][m] = res

if "modernbert" in models and models[0] != "modernbert":
    wins = {s: results[s][models[0]]["ai"]["ci"][0] > results[s]["modernbert"]["ai"]["ci"][1] for s in results}
    verdict = f"{models[0]} wins" if all(wins.values()) else f"stop: {models[0]} does not clearly beat ModernBERT-L"
else:
    wins, verdict = {}, "incomplete: contender or modernbert scores missing"

write_json(args.scores / "measure.json", {"cutoffs": cut, "results": results, "wins": wins, "verdict": verdict,
                                          "bootstrap": args.boot, "licence": LICENCE_NOTE})
for stem, res in results.items():
    print(f"\n{stem} ({res['docs']} docs)")
    for m in models:
        cells = "  ".join(f"{g} {v['caught']:.3f} [{v['ci'][0]:.3f}, {v['ci'][1]:.3f}]"
                          for g, v in res[m].items() if g != "human_flagged")
        print(f"  {m:<11} humans flagged {res[m]['human_flagged']:.3f}  {cells}")
print(f"\nverdict: {verdict}")
