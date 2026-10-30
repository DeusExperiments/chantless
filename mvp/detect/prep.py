"""Write the run's data files from pinned WildOutlines and Beemo revisions.

Usage: python detect/prep.py OUT
Writes OUT/{train,validation,calibration,test,beemo}.parquet with columns id, text, label (human/ai),
kind, format, plus OUT/prep.json (revisions, counts, hashes).
- train, test: WildOutlines splits as published.
- validation: 1,000 human + 1,000 AI from the validation split, for the training loss curve only.
- calibration: 2,500 human documents per format (20,000) from the all-human calibration split; each model's
  cutoff is set here so that 1% of these get flagged.
- beemo: per prompt, the human answer (human), the model output (ai, generated) and the expert-edited
  model output (ai, expert-edited). The LLM-edited variants are left out. Exact duplicate texts are dropped.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset

from common import BEEMO, LICENCE_NOTE, WILDOUTLINES, sha256_file, write_json

OUT = Path(sys.argv[1])
SEED = 0
CAL_PER_FORMAT = 2500
VAL_PER_LABEL = 1000
COLUMNS = ["id", "text", "label", "kind", "format"]
OUT.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(SEED)

wo = load_dataset(WILDOUTLINES[0], revision=WILDOUTLINES[1])


def wo_split(name, idx=None):
    ds = wo[name] if idx is None else wo[name].select(sorted(idx))
    ds = ds.map(lambda b: {"kind": b["label"]}, batched=True).select_columns(COLUMNS)
    ds.to_parquet(str(OUT / f"{name}.parquet"))
    return len(ds)


def pick(values, per_value):
    values = np.asarray(list(values))
    return [int(i) for v in sorted(set(values)) for i in rng.permutation(np.flatnonzero(values == v))[:per_value]]


counts = {
    "train": wo_split("train"),
    "test": wo_split("test"),
    "validation": wo_split("validation", pick(wo["validation"]["label"], VAL_PER_LABEL)),
    "calibration": wo_split("calibration", pick(wo["calibration"]["format"], CAL_PER_FORMAT)),
}

rows, seen = [], set()
for r in load_dataset(BEEMO[0], revision=BEEMO[1])["train"]:
    for rid, text, label, kind in ((f"beemo-h-{r['prompt_id']}", r["human_output"], "human", "human"),
                                   (f"beemo-{r['id']}-g", r["model_output"], "ai", "generated"),
                                   (f"beemo-{r['id']}-e", r["human_edits"], "ai", "expert-edited")):
        if text and text.strip() and text not in seen:
            seen.add(text)
            rows.append({"id": rid, "text": text, "label": label, "kind": kind, "format": r["category"]})
pq.write_table(pa.Table.from_pylist(rows), OUT / "beemo.parquet")
counts["beemo"] = len(rows)

files = {p.name: sha256_file(p) for p in sorted(OUT.glob("*.parquet"))}
write_json(OUT / "prep.json", {"wildoutlines": WILDOUTLINES, "beemo": BEEMO, "seed": SEED, "counts": counts,
                               "sha256": files, "licence": LICENCE_NOTE})
print(counts)
