"""Detector environment shared across architectures (Qwen, Gemma 4, etc.): one document -> one forward pass.

Entry point: `make_environment(**kwargs)`. Supports both ChatML (`prompt_style="qwen_chatml"`, raw
ChatML) and Hugging Face chat templates (`prompt_style="chat_template"` or `"auto"`, used for
Gemma 4 and any non-ChatML tokenizer). When scoring a vocabulary-trimmed GGUF (`n_vocab < source_vocab`), `score()`
automatically resolves the remapped `(human, ai)` token IDs in the candidate's GGUF header and re-tokenizes the
rendered prompts with the candidate's own vocabulary (`tokdocs` when present, or a pruned BPE fallback).
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

from .. import vocab as vocabmod
from .protocol import Episode, Task, Verdict

# ProseLens model-card prompt
SYSTEM = ("Given a document, answer with one word: human if the document was "
          "human-written, ai if it was AI-generated.")
SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
ANSWERS = ("human", "ai")
GRAMMAR = 'root ::= "human" | "ai"'


def chat_prompt(text: str) -> str:
    return f"<|im_start|>system\n{SYSTEM}<|im_end|>\n<|im_start|>user\n{text}{SUFFIX}"


def _loss_curve_rows(labels) -> set[int]:
    """Validation rows excluded from search/select (seed 0, 1,000 per label)."""
    rng = np.random.default_rng(0)
    values = np.asarray(labels)
    return {int(i) for v in sorted(set(values)) for i in rng.permutation(np.flatnonzero(values == v))[:1000]}


def _beemo_rows(path) -> list[dict]:
    """Load distinct Beemo texts; hash 30% of prompt_ids to 'search' and the rest to 'exam'."""
    import pyarrow.parquet as pq
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
    """Document ids from score files (a path or glob): JSONL rows with an 'id', or a .json list of ids."""
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


def _sample(rows: list[dict], key, per: int | None, rng) -> list[dict]:
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
    version = "2"
    tools = None

    def __init__(self, data_dir: str, hf_dir: str, cutoff: float | None = None, seed: int = 20261029,
                 max_doc_tokens: int = 4096, search_per_cell: int | None = 20, select_per_cell: int | None = 50,
                 exam_test_per_cell: int | None = 75, exam_beemo_per_kind: int | None = 200, context: int = 4608,
                 kv_type: str = "f16", max_total_memory_bytes: int | None = None, classifier: bool = False,
                 calibration_ids_file: str | None = None, cache_dir: str | None = None,
                 search_beemo_per_kind: int | None = None, select_beemo_per_kind: int | None = None,
                 prompt_style: str = "auto"):
        """A per-cell or per-kind size of None takes every document of that cell."""
        from transformers import AutoTokenizer
        self.classifier = bool(classifier)
        if cutoff is None and not self.classifier:
            raise ValueError("cutoff is required unless classifier=True")
        self.cutoff = None if cutoff is None else float(cutoff)
        self.budget = {"max_tokens": 1, "max_doc_tokens": max_doc_tokens}
        self.deployment = {"context": context, "concurrency": 1, "kv_type": kv_type,
                           "max_total_memory_bytes": max_total_memory_bytes}
        self.hf_dir = str(hf_dir)
        self.tok = AutoTokenizer.from_pretrained(hf_dir)
        self.max_doc_tokens = max_doc_tokens
        if prompt_style == "auto":
            vocab = self.tok.get_vocab()
            self.prompt_style = "qwen_chatml" if "<|im_start|>" in vocab and "<think>" in vocab else "chat_template"
        elif prompt_style in ("qwen_chatml", "chat_template"):
            self.prompt_style = prompt_style
        else:
            raise ValueError(f"unknown prompt_style {prompt_style!r}")

        d = Path(data_dir)
        rng = np.random.default_rng(seed)
        if self.classifier:
            if not calibration_ids_file or not cache_dir:
                raise ValueError("classifier=True needs calibration_ids_file and cache_dir")
            self.cache_dir = Path(cache_dir)
            self.pair = tuple(self._single_id(w) for w in ANSWERS)
            self.source_vocab_size = len(self.tok)
            backend = getattr(self.tok, "backend_tokenizer", None)
            ident = backend.to_str() if backend is not None else f"{Path(hf_dir).resolve()}:{len(self.tok)}"
            self._tok_id = hashlib.sha256(ident.encode()).hexdigest()
            self._pruned_tok_cache: dict[str, object] = {}
            self._classifier_tasks(d, rng, _read_ids(calibration_ids_file), search_per_cell, select_per_cell,
                                   exam_test_per_cell, search_beemo_per_kind, select_beemo_per_kind,
                                   exam_beemo_per_kind)
            return

        import pyarrow.parquet as pq
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

    def render_prompt(self, text: str) -> str:
        if self.prompt_style == "qwen_chatml":
            return chat_prompt(text)
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": text}]
        try:
            return self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            return self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    def _classifier_tasks(self, d: Path, rng, cal_ids: set[str], search_per_cell, select_per_cell, exam_test_per_cell,
                          search_beemo_per_kind, select_beemo_per_kind, exam_beemo_per_kind) -> None:
        """Build classifier-mode splits (search, select, exam, calibration)."""
        import pyarrow.parquet as pq
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
        """Prompt token ids per task, in order: the chat prompt tokenised as one string."""
        for i in range(0, len(tasks), chunk):
            yield from self.tok([self.render_prompt(t.prompt["text"]) for t in tasks[i:i + chunk]],
                                add_special_tokens=False)["input_ids"]

    def _key(self, tasks: list[Task]) -> str:
        """Cache key: tokenizer, answer pair, prompt template, document cut and the ordered task ids."""
        head = [self._tok_id, repr(self.pair), self.render_prompt(""), str(self.max_doc_tokens)]
        return hashlib.sha256("\n".join(head + [t.id for t in tasks]).encode()).hexdigest()[:16]

    def _tmp(self, path: Path) -> Path:
        return path.with_name(f"{path.name}.{os.getpid()}-{threading.get_ident()}.tmp")

    def _write_trj_seqs(self, path: Path, seqs, first_ans_id: int) -> Path:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        tmp = self._tmp(path)
        seqs = list(seqs)
        with open(tmp, "wb") as f:
            f.write(b"TRJ1" + struct.pack("<I", len(seqs)))
            for x in seqs:
                arr = np.asarray(x, dtype="<i4")
                seq = np.append(arr, np.int32(first_ans_id)).astype("<i4")
                f.write(struct.pack("<II", len(seq), 1) + seq.tobytes() + struct.pack("<I", len(arr)))
        os.replace(tmp, path)
        return path

    def trj_for(self, tasks: list[Task]) -> Path:
        """A TRJ1 file of these tasks in order for the untrimmed source vocabulary, cached by task ids."""
        path = self.cache_dir / f"{self._key(tasks)}.trj"
        if path.exists():
            return path
        return self._write_trj_seqs(path, self._encoded(tasks), self.pair[0])

    def _model_vocab_plan(self, model: str) -> tuple[tuple[int, int], str | None, dict | None]:
        """Inspect `model` GGUF vocabulary: returns `((h_id, a_id), vocab_hash_if_trimmed_else_None, meta)`."""
        if not os.path.exists(model):
            return self.pair, None, None
        try:
            meta = vocabmod.read_tokenizer(model)
        except Exception:
            return self.pair, None, None
        if meta["vocab_size"] == self.source_vocab_size:
            return self.pair, None, meta
        # Resolve exact source answer token byte strings inside the trimmed vocabulary
        idx = {b: i for i, b in enumerate(meta["tokens"])}
        src_toks = self._source_token_bytes()
        remapped = []
        for w, pid in zip(ANSWERS, self.pair):
            tb = src_toks[pid]
            if tb in idx:
                remapped.append(idx[tb])
            elif w.encode("utf-8") in idx:
                remapped.append(idx[w.encode("utf-8")])
            else:
                raise RuntimeError(f"trimmed GGUF {model} dropped answer token {pid} ({w!r} / {tb!r})")
        vh = hashlib.sha256(b"\x00".join(meta["tokens"])).hexdigest()[:12]
        return (remapped[0], remapped[1]), vh, meta

    def _source_token_bytes(self) -> list[bytes]:
        if not hasattr(self, "_src_tok_bytes"):
            # Fallback table mapping source token id -> raw GGUF-style token string bytes
            vocab = self.tok.get_vocab()
            inv = [""] * max(len(self.tok), max(vocab.values()) + 1)
            for s, i in vocab.items():
                inv[i] = s
            self._src_tok_bytes = [s.encode("utf-8") for s in inv]
        return self._src_tok_bytes

    def _pruned_hf_encode(self, vh: str, meta: dict, prompts: list[str]) -> list[list[int]]:
        """Fallback HF retokenizer when `toolchain.tokdocs` is unavailable."""
        if vh not in self._pruned_tok_cache:
            from tokenizers import Tokenizer
            backend = getattr(self.tok, "backend_tokenizer", None)
            if backend is None:
                raise RuntimeError("trimmed-GGUF scoring needs either toolchain.tokdocs or a fast HF tokenizer")
            spec = json.loads(backend.to_str())
            kept_strs = [t.decode("utf-8", errors="surrogateescape") for t in meta["tokens"]]
            new_vocab = {s: i for i, s in enumerate(kept_strs)}
            if spec.get("model", {}).get("type") == "BPE":
                spec["model"]["vocab"] = new_vocab
                spec["model"]["merges"] = [m.decode("utf-8") for m in meta["merges"]]
            if "added_tokens" in spec:
                kept_added = []
                for at in spec["added_tokens"]:
                    if at["content"] in new_vocab:
                        at = dict(at)
                        at["id"] = new_vocab[at["content"]]
                        kept_added.append(at)
                spec["added_tokens"] = kept_added
            self._pruned_tok_cache[vh] = Tokenizer.from_str(json.dumps(spec))
        tk = self._pruned_tok_cache[vh]
        return [e.ids for e in tk.encode_batch(prompts, add_special_tokens=False)]

    def trj_for_model(self, tasks: list[Task], model: str, tc=None) -> tuple[Path, tuple[int, int]]:
        """Return `(trj_path, answer_pair)` matching `model`'s vocabulary (re-tokenizing when trimmed)."""
        pair, vh, meta = self._model_vocab_plan(model)
        if vh is None:
            return self.trj_for(tasks), self.pair
        path = self.cache_dir / f"{self._key(tasks)}-v{vh}.trj"
        if path.exists():
            return path, pair
        prompts = [self.render_prompt(t.prompt["text"]) for t in tasks]
        if tc is not None and getattr(tc, "tokdocs", None) is not None:
            from ..runtime import tokenize_docs
            seqs = tokenize_docs(tc, model, prompts)
        else:
            seqs = self._pruned_hf_encode(vh, meta, prompts)
        return self._write_trj_seqs(path, seqs, pair[0]), pair

    def score(self, tasks: list[Task], model: str, scorer) -> dict[str, float]:
        """log p(human) - log p(ai) at the answer per task id; automatically retokenizes trimmed GGUFs."""
        if not tasks:
            return {}
        tc = getattr(scorer, "tc", None)
        trj_path, pair = self.trj_for_model(tasks, str(model), tc=tc)
        v = np.asarray(scorer.score_trj(str(model), str(trj_path), pair), dtype=float)
        if v.shape != (len(tasks),) or not np.isfinite(v).all():
            raise RuntimeError(f"scorer returned {v.shape} values for {len(tasks)} tasks, or non-finite ones")
        return {t.id: float(x) for t, x in zip(tasks, v)}

    def fit_token_counts(self, gguf_path: str | None = None, tc=None) -> np.ndarray:
        """Token occurrence counts on `calibration`, `search`, and WildOutlines `select` (excluding `exam` and Beemo `select`)."""
        fit_tasks = [
            t for t in self._tasks
            if t.split in ("calibration", "search") or (t.split == "select" and t.id.startswith("wo-val:"))
        ]
        if gguf_path and tc is not None and getattr(tc, "tokdocs", None) is not None:
            from ..runtime import tokenize_docs
            v = vocabmod.read_tokenizer(gguf_path)["vocab_size"]
            seqs = tokenize_docs(tc, gguf_path, [self.render_prompt(t.prompt["text"]) for t in fit_tasks])
            return np.bincount(np.concatenate(seqs), minlength=v)
        v = self.source_vocab_size if hasattr(self, "source_vocab_size") else len(self.tok)
        all_ids = [np.asarray(x, dtype=np.int64) for x in self._encoded(fit_tasks)]
        return np.bincount(np.concatenate(all_ids), minlength=v) if all_ids else np.zeros(v, dtype=np.int64)

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
        """Parts of the log-odds store, in store order: random calibration humans + near-cutoff search AI docs."""
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
        return [self.render_prompt(pool[i].prompt["text"]) for i in pick]

    def token_counts(self) -> dict[str, dict]:
        """{split: {'docs', 'tokens'}}, tokens being the prompt plus answer token each forward pass reads."""
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
        body = {"prompt": self.render_prompt(task.prompt["text"]), "n_predict": 1, "n_probs": 10, "temperature": 1.0,
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
    pr = {t["token"]: t["prob"] for t in entry["top_probs"]}
    h, a = pr.get(ANSWERS[0], 0.0), pr.get(ANSWERS[1], 0.0)
    if h + a <= 0:
        raise KeyError("neither answer token among the returned probabilities")
    return h / (h + a)


def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    z = math.exp(x)
    return z / (1.0 + z)


def make_environment(**kwargs) -> DetectEnvironment:
    return DetectEnvironment(**kwargs)
