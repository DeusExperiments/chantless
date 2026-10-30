"""Detector environment: each task is one document, and an episode is one forward pass of the detector.

Entry point: make_environment(**kwargs). The model reads the document in the ProseLens chat prompt and the
episode records p(human) from the "human" and "ai" probabilities at the answer position. Success means the
label at the fixed cutoff is right (AI when p(human) < cutoff). Search and select documents come from the
WildOutlines validation split, minus the 2,000 documents the detector run used for its loss curve; the exam
is WildOutlines test plus Beemo. Families are split/source/format/label, so no family crosses splits.

With classifier=True the environment is judged at each model's own operating point instead (see
CLASSIFIER-CONTRACT.md): a calibration split of the source's 20,000 calibration humans sets the cutoff, the
evaluator scores tasks in batch through score() and decides with calibrate()/decide(), and Beemo's search
prompts (sha256(prompt_id) % 10 < 3) join the search and select splits while the other 70% join the exam.
"""
from __future__ import annotations

import glob
import hashlib
import json
import math
import os
import struct
import threading
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


def _beemo_rows(path) -> list[dict]:
    """Every distinct Beemo text, as in mvp/detect/prep.py; split "search" for 30% of prompts (climb_prep.py)."""
    beemo, seen = [], set()
    for r in pq.read_table(path).to_pylist():
        split = "search" if int(hashlib.sha256(str(r["prompt_id"]).encode()).hexdigest(), 16) % 10 < 3 else "exam"
        for rid, text, label, kind in ((f"beemo-h-{r['prompt_id']}", r["human_output"], "human", "human"),
                                       (f"beemo-{r['id']}-g", r["model_output"], "ai", "generated"),
                                       (f"beemo-{r['id']}-e", r["human_edits"], "ai", "expert-edited")):
            if text and text.strip() and text not in seen:
                seen.add(text)
                beemo.append({"id": rid, "text": text, "label": label, "kind": kind, "format": r["category"],
                              "prompt_id": str(r["prompt_id"]), "split": split})
    return beemo


def _read_ids(pattern: str) -> set[str]:
    """Document ids from score files (a path or glob): JSONL rows with an "id", or a .json list of ids."""
    files = sorted(glob.glob(str(pattern)))
    if not files:
        raise FileNotFoundError(f"no calibration id files match {pattern}")
    ids: set[str] = set()
    for f in files:
        if f.endswith(".json"):
            ids |= set(json.loads(Path(f).read_text()))
        else:
            ids |= {json.loads(line)["id"] for line in open(f) if line.strip()}
    return ids


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

    def __init__(self, data_dir: str, hf_dir: str, cutoff: float | None = None, seed: int = 20261029,
                 max_doc_tokens: int = 4096, search_per_cell: int | None = 20, select_per_cell: int | None = 50,
                 exam_test_per_cell: int | None = 75, exam_beemo_per_kind: int | None = 200, context: int = 4608,
                 kv_type: str = "f16", max_total_memory_bytes: int | None = None, classifier: bool = False,
                 calibration_ids_file: str | None = None, cache_dir: str | None = None,
                 search_beemo_per_kind: int | None = None, select_beemo_per_kind: int | None = None):
        """A per-cell or per-kind size of None takes every document of that cell."""
        from transformers import AutoTokenizer
        self.classifier = bool(classifier)
        if cutoff is None and not self.classifier:
            raise ValueError("cutoff is required unless classifier=True")
        self.cutoff = None if cutoff is None else float(cutoff)
        self.budget = {"max_tokens": 1, "max_doc_tokens": max_doc_tokens}
        self.deployment = {"context": context, "concurrency": 1, "kv_type": kv_type,
                           "max_total_memory_bytes": max_total_memory_bytes}
        self.tok = AutoTokenizer.from_pretrained(hf_dir)
        self.max_doc_tokens = max_doc_tokens
        d = Path(data_dir)
        rng = np.random.default_rng(seed)
        if self.classifier:
            if not calibration_ids_file or not cache_dir:
                raise ValueError("classifier=True needs calibration_ids_file and cache_dir")
            self.cache_dir = Path(cache_dir)
            self.pair = tuple(self._single_id(w) for w in ANSWERS)
            # Cache identity: the full tokenizer serialisation (vocab, merges, added tokens) when there is one
            backend = getattr(self.tok, "backend_tokenizer", None)
            ident = backend.to_str() if backend is not None else f"{Path(hf_dir).resolve()}:{len(self.tok)}"
            self._tok_id = hashlib.sha256(ident.encode()).hexdigest()
            self._classifier_tasks(d, rng, _read_ids(calibration_ids_file), search_per_cell, select_per_cell,
                                   exam_test_per_cell, search_beemo_per_kind, select_beemo_per_kind,
                                   exam_beemo_per_kind)
            return
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
        beemo = _beemo_rows(d / "beemo/data/train-00000-of-00001.parquet")
        picks.append(("exam", "beemo", _sample(beemo, lambda r: r["kind"], exam_beemo_per_kind, rng)))
        self._tasks = []
        for split, source, rows in picks:
            for r in rows:
                text, cut = self._head(r["text"])
                kind = r.get("kind", r["label"])
                self._tasks.append(Task(id=f"{source}:{r['id']}", family=f"{split}/{source}/{r['format']}/{kind}",
                                        split=split, prompt={"text": text},
                                        meta={"label": r["label"], "kind": kind, "truncated": cut}))

    def _classifier_tasks(self, d: Path, rng, cal_ids: set[str], search_per_cell, select_per_cell, exam_test_per_cell,
                          search_beemo_per_kind, select_beemo_per_kind, exam_beemo_per_kind) -> None:
        """Classifier-mode splits. Beemo's search prompts are halved by prompt between search and select, so a
        prompt's texts stay in one split; the calibration humans are the ids the source's cutoff was set on."""
        cols = ["id", "text", "label", "format"]
        wo = d / "WildOutlines/data"
        val = pq.read_table(wo / "validation-00000-of-00001.parquet", columns=cols).to_pylist()
        skip = _loss_curve_rows([r["label"] for r in val])
        val = [r for i, r in enumerate(val) if i not in skip]
        cell, kind = (lambda r: (r["format"], r["label"])), (lambda r: r["kind"])
        half = set(rng.permutation(len(val))[: len(val) // 2].tolist())
        a = [r for i, r in enumerate(val) if i in half]
        b = [r for i, r in enumerate(val) if i not in half]
        beemo = _beemo_rows(d / "beemo/data/train-00000-of-00001.parquet")
        pool = [r for r in beemo if r["split"] == "search"]
        prompts = sorted({r["prompt_id"] for r in pool})
        first = {prompts[i] for i in rng.permutation(len(prompts))[: len(prompts) // 2]}
        test = pq.read_table(wo / "test-00000-of-00001.parquet", columns=cols).to_pylist()
        cal = [r for r in pq.read_table(wo / "calibration-00000-of-00001.parquet", columns=cols).to_pylist()
               if r["id"] in cal_ids]
        if len(cal) != len(cal_ids):
            raise ValueError(f"{len(cal_ids) - len(cal)} calibration ids are not in the calibration parquet")
        first_half = [r for r in pool if r["prompt_id"] in first]
        second_half = [r for r in pool if r["prompt_id"] not in first]
        picks = [("search", "wo-val", _sample(a, cell, search_per_cell, rng)),
                 ("search", "beemo", _sample(first_half, kind, search_beemo_per_kind, rng)),
                 ("select", "wo-val", _sample(b, cell, select_per_cell, rng)),
                 ("select", "beemo", _sample(second_half, kind, select_beemo_per_kind, rng)),
                 ("exam", "wo-test", _sample(test, cell, exam_test_per_cell, rng)),
                 ("exam", "beemo", _sample([r for r in beemo if r["split"] == "exam"], kind, exam_beemo_per_kind, rng)),
                 ("calibration", "wo-cal", cal)]
        self._tasks = []
        for split, source, rows in picks:
            for r, (text, cut) in zip(rows, self._heads([r["text"] for r in rows])):
                k = r.get("kind", r["label"])
                self._tasks.append(Task(id=f"{source}:{r['id']}", family=f"{split}/{source}/{r['format']}/{k}",
                                        split=split, prompt={"text": text},
                                        meta={"label": r["label"], "kind": k, "truncated": cut}))
        self._cal_ids = {t.id for t in self._tasks if t.split == "calibration"}

    def _single_id(self, word: str) -> int:
        ids = self.tok.encode(word, add_special_tokens=False)
        if len(ids) != 1:
            raise RuntimeError(f"answer word {word!r} is not one token: {ids}")
        return ids[0]

    def _heads(self, texts: list[str], chunk: int = 256) -> list[tuple[str, bool]]:
        """_head over many texts, tokenised in batches (the same cut, faster)."""
        out, n = [], self.max_doc_tokens
        for i in range(0, len(texts), chunk):
            part = texts[i:i + chunk]
            enc = self.tok(part, add_special_tokens=False, return_offsets_mapping=True)
            out += [(t, False) if len(ids) <= n else (t[: off[n - 1][1]], True)
                    for t, ids, off in zip(part, enc["input_ids"], enc["offset_mapping"])]
        return out

    def _head(self, text: str) -> tuple[str, bool]:
        enc = self.tok(text, add_special_tokens=False, return_offsets_mapping=True)
        if len(enc["input_ids"]) <= self.max_doc_tokens:
            return text, False
        return text[: enc["offset_mapping"][self.max_doc_tokens - 1][1]], True

    def tasks(self) -> list[Task]:
        return list(self._tasks)

    # ------------------------------------------------------------------ classifier mode
    def _encoded(self, tasks: list[Task], chunk: int = 256):
        """Prompt token ids per task, in order: the chat prompt tokenised as one string, as climb_prep.py does."""
        for i in range(0, len(tasks), chunk):
            yield from self.tok([chat_prompt(t.prompt["text"]) for t in tasks[i:i + chunk]],
                                add_special_tokens=False)["input_ids"]

    def _key(self, tasks: list[Task]) -> str:
        """Cache key: tokenizer, answer pair, prompt template, document cut and the ordered task ids."""
        head = [self._tok_id, repr(self.pair), chat_prompt(""), str(self.max_doc_tokens)]
        return hashlib.sha256("\n".join(head + [t.id for t in tasks]).encode()).hexdigest()[:16]

    def _tmp(self, path: Path) -> Path:
        return path.with_name(f"{path.name}.{os.getpid()}-{threading.get_ident()}.tmp")

    def trj_for(self, tasks: list[Task]) -> Path:
        """A TRJ1 file of these tasks in order, the answer token appended as the one target; cached by task ids."""
        path = self.cache_dir / f"{self._key(tasks)}.trj"
        if path.exists():
            return path
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        tmp = self._tmp(path)
        with open(tmp, "wb") as f:
            f.write(b"TRJ1" + struct.pack("<I", len(tasks)))
            for x in self._encoded(tasks):
                seq = x + [self.pair[0]]
                f.write(struct.pack("<II", len(seq), 1) + struct.pack(f"<{len(seq)}i", *seq)
                        + struct.pack("<I", len(x)))
        os.replace(tmp, path)
        return path

    def score(self, tasks: list[Task], model: str, scorer) -> dict[str, float]:
        """log p(human) - log p(ai) at the answer per task id; lower is more AI-like (the flagged class)."""
        if not tasks:
            return {}
        v = np.asarray(scorer.score_trj(str(model), str(self.trj_for(tasks)), self.pair), dtype=float)
        if v.shape != (len(tasks),) or not np.isfinite(v).all():
            raise RuntimeError(f"scorer returned {v.shape} values for {len(tasks)} tasks, or non-finite ones")
        return {t.id: float(x) for t, x in zip(tasks, v)}

    def calibrate(self, scores: dict[str, float]) -> dict:
        """The cutoff that flags the lowest 1% of this model's calibration humans (flag strictly below)."""
        cal = sorted(float(v) for k, v in scores.items() if k in self._cal_ids)
        if not cal:
            raise ValueError("no calibration scores to calibrate on")
        return {"cutoff": cal[int(0.01 * len(cal))], "n": len(cal)}

    def decide(self, task: Task, score: float, rule: dict) -> Verdict:
        flagged = score < rule["cutoff"]
        return Verdict(task.id, 0, flagged == (task.meta["label"] == "ai"), {},
                       details={"score": score, "p_human": _sigmoid(score), "cutoff": rule["cutoff"],
                                "pred": "ai" if flagged else "human", "label": task.meta["label"],
                                "kind": task.meta["kind"]})

    def proxy_spec(self, source_scores: dict[str, float], n_neg: int = 1000, n_pos: int = 600,
                   seed: int = 0) -> list[dict]:
        """Parts of the log-odds store, in store order: a random sample of calibration humans, and the search
        split's AI documents whose source scores are closest to the source's cutoff. The humans are a random
        sample, never the low tail, so the objective can pick each candidate's own hardest ones."""
        cut = self.calibrate(source_scores)["cutoff"]
        cal = [t for t in self._tasks if t.split == "calibration"]
        neg = np.sort(np.random.default_rng(seed).choice(len(cal), min(n_neg, len(cal)), replace=False))
        ai = [(i, t) for i, t in enumerate(self._tasks) if t.split == "search" and t.meta["label"] == "ai"]
        near = sorted(sorted(ai, key=lambda x: (abs(source_scores[x[1].id] - cut), x[0]))[:n_pos], key=lambda x: x[0])
        return [{"name": "humans", "role": "neg", "n": len(neg), "ids": [cal[i].id for i in neg]},
                {"name": "ai", "role": "pos", "n": len(near), "ids": [t.id for _, t in near]}]

    def stats_texts(self, n: int, seed: int = 0) -> list[str]:
        """n raw prompt strings from the search split (a fixed random sample) for the importance matrix."""
        pool = [t for t in self._tasks if t.split == "search"]
        pick = np.sort(np.random.default_rng(seed).permutation(len(pool))[:n])
        return [chat_prompt(pool[i].prompt["text"]) for i in pick]

    def token_counts(self) -> dict[str, dict]:
        """{split: {"docs", "tokens"}}, tokens being the prompt plus answer token each forward pass reads."""
        path = None
        if getattr(self, "cache_dir", None):
            path = self.cache_dir / f"tokens-{self._key(self._tasks)}.json"
        if path and path.exists():
            return json.loads(path.read_text())
        out: dict[str, dict] = {}
        for t, x in zip(self._tasks, self._encoded(self._tasks)):
            c = out.setdefault(t.split, {"docs": 0, "tokens": 0})
            c["docs"], c["tokens"] = c["docs"] + 1, c["tokens"] + len(x) + 1
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._tmp(path)
            tmp.write_text(json.dumps(out))
            os.replace(tmp, path)
        return out

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


def _sigmoid(x: float) -> float:
    """1 / (1 + e^-x) without overflow at large |x|."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    z = math.exp(x)
    return z / (1.0 + z)


def make_environment(**kwargs) -> DetectEnvironment:
    return DetectEnvironment(**kwargs)
