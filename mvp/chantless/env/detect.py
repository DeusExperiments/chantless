"""Detector environment: each task is one document, and an episode is one forward pass of the detector.

Entry point: make_environment(**kwargs). The model reads the document in the ProseLens chat prompt and the
episode records p(human) from the "human" and "ai" probabilities at the answer position. Success means the
label at the fixed cutoff is right (AI when p(human) < cutoff). Search and select documents come from the
WildOutlines validation split, minus the 2,000 documents the detector run used for its loss curve; the exam
is WildOutlines test plus Beemo. Families are split/source/format/label, so no family crosses splits.
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from .protocol import Episode, Task, Verdict

# ProseLens model-card prompt, as in mvp/detect/common.py
SYSTEM = ("Given a document, answer with one word: human if the document was "
          "human-written, ai if it was AI-generated.")
SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
ANSWERS = ("human", "ai")
GRAMMAR = 'root ::= "human" | "ai"'


def chat_prompt(text: str) -> str:
    return f"<|im_start|>system\n{SYSTEM}<|im_end|>\n<|im_start|>user\n{text}{SUFFIX}"


def _loss_curve_rows(labels) -> set[int]:
    """The validation rows mvp/detect/prep.py picked (seed 0, 1,000 per label): never used here."""
    rng = np.random.default_rng(0)
    values = np.asarray(labels)
    return {int(i) for v in sorted(set(values)) for i in rng.permutation(np.flatnonzero(values == v))[:1000]}


def _sample(rows: list[dict], key, per: int, rng) -> list[dict]:
    by: dict = {}
    for r in rows:
        by.setdefault(key(r), []).append(r)
    out = []
    for k in sorted(by):
        idx = rng.permutation(len(by[k]))[:per]
        out += [by[k][i] for i in sorted(idx)]
    return out


class DetectEnvironment:
    name = "detect"
    version = "1"
    tools = None

    def __init__(self, data_dir: str, hf_dir: str, cutoff: float, seed: int = 20261029, max_doc_tokens: int = 4096,
                 search_per_cell: int = 20, select_per_cell: int = 50, exam_test_per_cell: int = 75,
                 exam_beemo_per_kind: int = 200, context: int = 4608, kv_type: str = "f16",
                 max_total_memory_bytes: int | None = None):
        from transformers import AutoTokenizer
        self.cutoff = float(cutoff)
        self.budget = {"max_tokens": 1, "max_doc_tokens": max_doc_tokens}
        self.deployment = {"context": context, "concurrency": 1, "kv_type": kv_type,
                           "max_total_memory_bytes": max_total_memory_bytes}
        self.tok = AutoTokenizer.from_pretrained(hf_dir)
        self.max_doc_tokens = max_doc_tokens
        d = Path(data_dir)
        rng = np.random.default_rng(seed)
        cols = ["id", "text", "label", "format"]
        val = pq.read_table(d / "WildOutlines/data/validation-00000-of-00001.parquet", columns=cols).to_pylist()
        skip = _loss_curve_rows([r["label"] for r in val])
        val = [r for i, r in enumerate(val) if i not in skip]
        cell = lambda r: (r["format"], r["label"])
        order = rng.permutation(len(val))
        half = {i for i in order[: len(val) // 2]}
        a = [r for i, r in enumerate(val) if i in half]
        b = [r for i, r in enumerate(val) if i not in half]
        picks = [("search", "wo-val", _sample(a, cell, search_per_cell, rng)),
                 ("select", "wo-val", _sample(b, cell, select_per_cell, rng))]
        test = pq.read_table(d / "WildOutlines/data/test-00000-of-00001.parquet", columns=cols).to_pylist()
        picks.append(("exam", "wo-test", _sample(test, cell, exam_test_per_cell, rng)))
        beemo, seen = [], set()
        for r in pq.read_table(d / "beemo/data/train-00000-of-00001.parquet").to_pylist():
            for rid, text, label, kind in ((f"beemo-h-{r['prompt_id']}", r["human_output"], "human", "human"),
                                           (f"beemo-{r['id']}-g", r["model_output"], "ai", "generated"),
                                           (f"beemo-{r['id']}-e", r["human_edits"], "ai", "expert-edited")):
                if text and text.strip() and text not in seen:
                    seen.add(text)
                    beemo.append({"id": rid, "text": text, "label": label, "kind": kind, "format": r["category"]})
        picks.append(("exam", "beemo", _sample(beemo, lambda r: r["kind"], exam_beemo_per_kind, rng)))
        self._tasks = []
        for split, source, rows in picks:
            for r in rows:
                text, cut = self._head(r["text"])
                kind = r.get("kind", r["label"])
                self._tasks.append(Task(id=f"{source}:{r['id']}", family=f"{split}/{source}/{r['format']}/{kind}",
                                        split=split, prompt={"text": text},
                                        meta={"label": r["label"], "kind": kind, "truncated": cut}))

    def _head(self, text: str) -> tuple[str, bool]:
        enc = self.tok(text, add_special_tokens=False, return_offsets_mapping=True)
        if len(enc["input_ids"]) <= self.max_doc_tokens:
            return text, False
        return text[: enc["offset_mapping"][self.max_doc_tokens - 1][1]], True

    def tasks(self) -> list[Task]:
        return list(self._tasks)

    def run(self, task: Task, client, seed: int) -> Episode:
        """One forward pass through llama-server's raw completion endpoint; the answer is greedy."""
        t0 = time.time()
        # The grammar keeps only tokens that start "human" or "ai"; at temperature 1 with every other sampler off,
        # the post-sampling probabilities are the model's own softmax renormalised over those tokens, so the
        # human/ai ratio is exact
        body = {"prompt": chat_prompt(task.prompt["text"]), "n_predict": 1, "n_probs": 10, "temperature": 1.0,
                "top_k": 0, "top_p": 1.0, "min_p": 0.0, "grammar": GRAMMAR, "post_sampling_probs": True,
                "seed": int(seed), "cache_prompt": False}
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": task.prompt["text"]}]
        try:
            req = urllib.request.Request(client.base_url + "/completion", data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=client.timeout) as r:
                out = json.loads(r.read())
            p = _p_human(out["completion_probabilities"][0])
        except (urllib.error.URLError, ConnectionError, TimeoutError, KeyError, IndexError, json.JSONDecodeError) as e:
            return Episode(task.id, seed, msgs, None, None, ["error"], 0, 0, time.time() - t0, infra_error=repr(e))
        word = "ai" if p < self.cutoff else "human"
        msgs.append({"role": "assistant", "content": word})
        return Episode(task.id, seed, msgs, None, json.dumps({"p_human": p}), ["stop"], 1, 1, time.time() - t0)

    def verify(self, task: Task, episode: Episode) -> Verdict:
        if episode.infra_error:
            return Verdict(task.id, episode.seed, False, {}, infra_error=episode.infra_error)
        p = json.loads(episode.final_text)["p_human"]
        pred = "ai" if p < self.cutoff else "human"
        return Verdict(task.id, episode.seed, pred == task.meta["label"], {},
                       details={"p_human": p, "pred": pred, "label": task.meta["label"], "kind": task.meta["kind"]})


def _p_human(entry: dict) -> float:
    """p(human) over the two answer tokens, from the grammar-restricted post-sampling probabilities."""
    pr = {t["token"]: t["prob"] for t in entry["top_probs"]}
    h, a = pr.get(ANSWERS[0], 0.0), pr.get(ANSWERS[1], 0.0)
    if h + a <= 0:
        raise KeyError("neither answer token among the returned probabilities")
    return h / (h + a)


def make_environment(**kwargs) -> DetectEnvironment:
    return DetectEnvironment(**kwargs)
