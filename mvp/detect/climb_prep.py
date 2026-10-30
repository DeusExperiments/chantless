"""Tokenize every document the mask climb needs into TRJ1 files (one target: the answer position).

Usage: python detect/climb_prep.py HF_RAW SCORES TOKENIZER OUT
Writes OUT/{val,cal,test,beemo}.trj and OUT/{val,cal,test,beemo}.json (id, label, kind, format per trajectory,
in trajectory order), plus OUT/prep.json.
- val: WildOutlines validation minus the 2,000 documents the detector run used for its loss curve.
- cal: the 20,000 calibration humans every earlier cutoff was set on (ids from the source's score files).
- test: all 48,869 WildOutlines test documents.
- beemo: every Beemo document, with split "search" for 30% of prompts and "exam" for the rest. A prompt's
  human, generated and expert-edited texts share one split.
The prompt is the ProseLens chat prompt with the document cut after its first 4,096 tokens, tokenized as one
string, exactly as detect/score.py did; the answer token is appended so its prediction is the target.
"""
from __future__ import annotations

import glob
import hashlib
import json
import struct
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent))
from common import MAX_DOC_TOKENS, chat_prompt  # noqa: E402

RAW, SCORES, TOK, OUT = (Path(a) for a in sys.argv[1:5])
OUT.mkdir(parents=True, exist_ok=True)
tok = AutoTokenizer.from_pretrained(str(TOK))
HUMAN = tok.encode("human", add_special_tokens=False)[0]


def loss_curve_rows(labels) -> set[int]:
    rng = np.random.default_rng(0)  # mvp/detect/prep.py: validation pick, seed 0, 1,000 per label
    v = np.asarray(labels)
    return {int(i) for x in sorted(set(v)) for i in rng.permutation(np.flatnonzero(v == x))[:1000]}


def write(name: str, rows: list[dict]) -> None:
    texts = [r.pop("text") for r in rows]
    enc = tok(texts, add_special_tokens=False, return_offsets_mapping=True)
    heads = [t if len(ids) <= MAX_DOC_TOKENS else t[: off[MAX_DOC_TOKENS - 1][1]]
             for t, ids, off in zip(texts, enc["input_ids"], enc["offset_mapping"])]
    ids = tok([chat_prompt(h) for h in heads], add_special_tokens=False)["input_ids"]
    with open(OUT / f"{name}.trj", "wb") as f:
        f.write(b"TRJ1" + struct.pack("<I", len(ids)))
        for x in ids:
            seq = x + [HUMAN]
            f.write(struct.pack("<II", len(seq), 1) + struct.pack(f"<{len(seq)}i", *seq) + struct.pack("<I", len(x)))
    for r, x, h, t in zip(rows, ids, heads, texts):
        r["tokens"], r["truncated"] = len(x), len(h) < len(t)
    (OUT / f"{name}.json").write_text(json.dumps(rows))
    print(name, len(rows), "docs", sum(len(x) for x in ids), "tokens", flush=True)


cols = ["id", "text", "label", "format"]
val = pq.read_table(RAW / "WildOutlines/data/validation-00000-of-00001.parquet", columns=cols).to_pylist()
skip = loss_curve_rows([r["label"] for r in val])
write("val", [dict(r, kind=r["label"]) for i, r in enumerate(val) if i not in skip])

cal_ids = {json.loads(l)["id"] for f in glob.glob(str(SCORES / "calibration.shard*.jsonl")) for l in open(f)}
cal = [r for r in pq.read_table(RAW / "WildOutlines/data/calibration-00000-of-00001.parquet", columns=cols).to_pylist()
       if r["id"] in cal_ids]
assert len(cal) == len(cal_ids) == 20000, len(cal)
write("cal", [dict(r, kind="human") for r in cal])

write("test", [dict(r, kind=r["label"]) for r in
               pq.read_table(RAW / "WildOutlines/data/test-00000-of-00001.parquet", columns=cols).to_pylist()])

beemo, seen = [], set()  # same row construction as mvp/detect/prep.py
for r in pq.read_table(RAW / "beemo/data/train-00000-of-00001.parquet").to_pylist():
    split = "search" if int(hashlib.sha256(str(r["prompt_id"]).encode()).hexdigest(), 16) % 10 < 3 else "exam"
    for rid, text, label, kind in ((f"beemo-h-{r['prompt_id']}", r["human_output"], "human", "human"),
                                   (f"beemo-{r['id']}-g", r["model_output"], "ai", "generated"),
                                   (f"beemo-{r['id']}-e", r["human_edits"], "ai", "expert-edited")):
        if text and text.strip() and text not in seen:
            seen.add(text)
            beemo.append({"id": rid, "text": text, "label": label, "kind": kind, "format": r["category"], "split": split})
write("beemo", beemo)
(OUT / "prep.json").write_text(json.dumps({"max_doc_tokens": MAX_DOC_TOKENS, "answer_token": HUMAN,
                                           "beemo_split": "sha256(prompt_id) % 10 < 3 -> search"}))
