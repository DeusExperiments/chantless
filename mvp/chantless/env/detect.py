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
import http.client
import json
import math
import numbers
import os
import struct
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

from .. import vocab as vocabmod
from ..client import _DECODING_SKIP
from .bfcl import detect_dialect, render_chat_prompt
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
            ids |= set(json.loads(Path(f).read_text(encoding="utf-8")))
        else:
            with open(f, encoding="utf-8") as fh:
                ids |= {json.loads(line)["id"] for line in fh if line.strip()}
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
    unflagged_word = ANSWERS[0]
    flagged_word = ANSWERS[1]

    def __init__(self, data_dir: str, hf_dir: str, cutoff: float | None = None, seed: int = 20261029,
                 max_doc_tokens: int = 4096, search_per_cell: int | None = 20, select_per_cell: int | None = 50,
                 exam_test_per_cell: int | None = 75, exam_beemo_per_kind: int | None = 200, context: int = 4608,
                 kv_type: str = "f16", max_total_memory_bytes: int | None = None, classifier: bool = False,
                 calibration_ids_file: str | None = None, cache_dir: str | None = None,
                 search_beemo_per_kind: int | None = None, select_beemo_per_kind: int | None = None,
                 prompt_style: str = "auto", reasoning_effort: str | float | None = "none"):
        """A per-cell or per-kind size of None takes every document of that cell."""
        from transformers import AutoTokenizer
        self.classifier = bool(classifier)
        if cutoff is None and not self.classifier:
            raise ValueError("cutoff is required unless classifier=True")
        if cutoff is not None:
            if isinstance(cutoff, bool):
                raise ValueError(f"cutoff must be a finite float or None, got {cutoff!r}")
            self.cutoff = float(cutoff)
            if not math.isfinite(self.cutoff):
                raise ValueError(f"cutoff must be finite, got {cutoff!r}")
        else:
            self.cutoff = None
        self.budget = {"max_tokens": 1, "max_doc_tokens": max_doc_tokens}
        self.deployment = {"context": context, "concurrency": 1, "kv_type": kv_type,
                           "max_total_memory_bytes": max_total_memory_bytes}
        self.hf_dir = str(hf_dir)
        self.tok = AutoTokenizer.from_pretrained(hf_dir)
        self.max_doc_tokens = max_doc_tokens
        self.reasoning_effort = reasoning_effort
        det_dialect, _ = detect_dialect(self.tok)
        vocab = self.tok.get_vocab()
        if prompt_style == "auto":
            if det_dialect == "inkling2":
                self.prompt_style = "inkling2"
                self.dialect = "inkling2"
            elif "<|im_start|>" in vocab and "<think>" in vocab:
                self.prompt_style = "qwen_chatml"
                self.dialect = det_dialect
            else:
                self.prompt_style = "chat_template"
                self.dialect = det_dialect
        elif prompt_style == "inkling2":
            self.prompt_style = "inkling2"
            self.dialect = "inkling2"
        elif prompt_style in ("qwen_chatml", "chat_template"):
            self.prompt_style = prompt_style
            self.dialect = "qwen3_json" if (prompt_style == "qwen_chatml" and det_dialect == "inkling2") else det_dialect
        else:
            raise ValueError(f"unknown prompt_style {prompt_style!r}")
        self.needs_special_tokens = self.dialect == "inkling2"

        d = Path(data_dir)
        rng = np.random.default_rng(seed)
        self._pruned_tok_cache: dict[str, object] = {}
        self._prompt_hash: dict[tuple[str, str], bytes] = {}
        if self.classifier:
            if not calibration_ids_file or not cache_dir:
                raise ValueError("classifier=True needs calibration_ids_file and cache_dir")
            self.cache_dir = Path(cache_dir)
            self.pair = tuple(self._single_id(w) for w in ANSWERS)
            self.source_vocab_size = len(self.tok)
            backend = getattr(self.tok, "backend_tokenizer", None)
            ident = backend.to_str() if backend is not None else f"{Path(hf_dir).resolve()}:{len(self.tok)}"
            self._tok_id = hashlib.sha256(ident.encode()).hexdigest()
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
        if self.prompt_style == "inkling2":
            pstr = render_chat_prompt(self.tok, msgs, "inkling2", reasoning_effort=self.reasoning_effort)
            if pstr.endswith("<|end_message|>"):
                pstr += "<|message_model|><|content_text|>"
            return pstr
        return render_chat_prompt(self.tok, msgs, self.dialect, reasoning_effort=self.reasoning_effort)

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
        if not cal:
            raise ValueError("calibration split is empty")
        if len(cal) != len(cal_ids):
            raise ValueError(f"{len(cal_ids) - len(cal)} calibration ids are not in the calibration parquet")
        non_human = [r["id"] for r in cal if r.get("label") != ANSWERS[0]]
        if non_human:
            raise ValueError(f"calibration rows must have label {ANSWERS[0]!r}, got non-human ids: {non_human[:5]}")
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
            try:
                enc = self.tok(part, add_special_tokens=False, return_offsets_mapping=True)
                offsets = enc["offset_mapping"]
                for t, ids, off in zip(part, enc["input_ids"], offsets):
                    if len(ids) <= n:
                        out.append((t, False))
                    else:
                        end = int(off[n - 1][1]) if n - 1 < len(off) else 0
                        if end > 0:
                            out.append((t[:end], True))
                        else:
                            out.append((self.tok.decode(ids[:n], skip_special_tokens=True), True))
            except (NotImplementedError, TypeError, ValueError, KeyError):
                enc = self.tok(part, add_special_tokens=False)
                for t, ids in zip(part, enc["input_ids"]):
                    if len(ids) <= n:
                        out.append((t, False))
                    else:
                        out.append((self.tok.decode(ids[:n], skip_special_tokens=True), True))
        return out

    def _head(self, text: str) -> tuple[str, bool]:
        return self._heads([text])[0]

    def tasks(self) -> list[Task]:
        return list(self._tasks)

    # ------------------------------------------------------------------ classifier mode
    def _encoded(self, tasks: list[Task], chunk: int = 256):
        """Prompt token ids per task, in order: the chat prompt tokenised as one string."""
        for i in range(0, len(tasks), chunk):
            yield from self.tok([self.render_prompt(t.prompt["text"]) for t in tasks[i:i + chunk]],
                                add_special_tokens=False)["input_ids"]

    def _key(self, tasks: list[Task]) -> str:
        """Cache key: tokenizer, answer pair, prompt template, document cut and ordered (task_id, prompt_hash) pairs."""
        h = hashlib.sha256()
        for part in (self._tok_id, repr(self.pair), self.render_prompt(""), str(self.max_doc_tokens)):
            b = part.encode("utf-8")
            h.update(struct.pack("<I", len(b)))
            h.update(b)
        h.update(struct.pack("<I", len(tasks)))
        for t in tasks:
            tid_b = t.id.encode("utf-8")
            sp_b = t.split.encode("utf-8")
            txt = str(t.prompt["text"])
            ck = (t.id, txt)
            ph = self._prompt_hash.get(ck)
            if ph is None:
                ph = hashlib.sha256(self.render_prompt(txt).encode("utf-8")).digest()
                self._prompt_hash[ck] = ph
            h.update(struct.pack("<I", len(tid_b)))
            h.update(tid_b)
            h.update(struct.pack("<I", len(sp_b)))
            h.update(sp_b)
            h.update(ph)
        return h.hexdigest()[:16]

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
        meta = vocabmod.read_tokenizer(model)
        src_toks = self._source_token_bytes()
        v = int(meta["vocab_size"])
        if (
            v == self.source_vocab_size
            and self.pair[0] < v
            and self.pair[1] < v
            and meta["tokens"][self.pair[0]] == src_toks[self.pair[0]]
            and meta["tokens"][self.pair[1]] == src_toks[self.pair[1]]
        ):
            return self.pair, None, meta
        # Resolve exact source answer token byte strings inside the trimmed/remapped vocabulary
        idx = {b: i for i, b in enumerate(meta["tokens"])}
        remapped = []
        for w, pid in zip(ANSWERS, self.pair):
            tb = src_toks[pid]
            if tb in idx:
                remapped.append(idx[tb])
            else:
                raise RuntimeError(f"trimmed GGUF {model} dropped answer token {pid} ({w!r} / {tb!r})")
        vh = _gguf_vocab_hash(meta)
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
            mtype = spec.get("model", {}).get("type")
            if mtype == "BPE":
                spec["model"]["vocab"] = new_vocab
                spec["model"]["merges"] = [m.decode("utf-8") for m in meta["merges"]]
            elif mtype in ("WordLevel", "WordPiece"):
                spec["model"]["vocab"] = new_vocab
                unk = spec["model"].get("unk_token")
                if unk is not None and unk not in new_vocab:
                    raise RuntimeError(f"trimmed GGUF dropped unk_token {unk!r} for {mtype} tokenizer")
            else:
                raise RuntimeError(
                    f"trimmed-GGUF scoring without toolchain.tokdocs does not support tokenizer model type {mtype!r}"
                )
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
        if gguf_path is not None:
            meta = vocabmod.read_tokenizer(gguf_path)
            v = int(meta["vocab_size"])
            if tc is not None and getattr(tc, "tokdocs", None) is not None:
                from ..runtime import tokenize_docs
                if not fit_tasks:
                    return np.zeros(v, dtype=np.int64)
                seqs = [
                    np.asarray(s, dtype=np.int64)
                    for s in tokenize_docs(tc, gguf_path, [self.render_prompt(t.prompt["text"]) for t in fit_tasks])
                    if len(s)
                ]
                return np.bincount(np.concatenate(seqs), minlength=v)[:v] if seqs else np.zeros(v, dtype=np.int64)

            if hasattr(self, "source_vocab_size"):
                src_toks = self._source_token_bytes()
                untrimmed_or_padded = (
                    v >= self.source_vocab_size
                    and self.pair[0] < v
                    and self.pair[1] < v
                    and meta["tokens"][self.pair[0]] == src_toks[self.pair[0]]
                    and meta["tokens"][self.pair[1]] == src_toks[self.pair[1]]
                )
            else:
                untrimmed_or_padded = v >= len(self.tok)

            if untrimmed_or_padded:
                if not fit_tasks:
                    return np.zeros(v, dtype=np.int64)
                all_ids = [np.asarray(x, dtype=np.int64) for x in self._encoded(fit_tasks) if len(x)]
                return np.bincount(np.concatenate(all_ids), minlength=v)[:v] if all_ids else np.zeros(v, dtype=np.int64)

            if not fit_tasks:
                return np.zeros(v, dtype=np.int64)
            vh = _gguf_vocab_hash(meta)
            prompts = [self.render_prompt(t.prompt["text"]) for t in fit_tasks]
            encoded = self._pruned_hf_encode(vh, meta, prompts)
            seqs = [np.asarray(s, dtype=np.int64) for s in encoded if len(s)]
            return np.bincount(np.concatenate(seqs), minlength=v)[:v] if seqs else np.zeros(v, dtype=np.int64)

        v = int(self.source_vocab_size if hasattr(self, "source_vocab_size") else len(self.tok))
        if not fit_tasks:
            return np.zeros(v, dtype=np.int64)
        all_ids = [np.asarray(x, dtype=np.int64) for x in self._encoded(fit_tasks) if len(x)]
        return np.bincount(np.concatenate(all_ids), minlength=v)[:v] if all_ids else np.zeros(v, dtype=np.int64)

    def calibrate(self, scores: dict[str, float]) -> dict:
        """The cutoff that flags the lowest 1% of this model's calibration humans (flag strictly below)."""
        raw: list[float] = []
        for k, v in scores.items():
            if k in self._cal_ids:
                if isinstance(v, bool):
                    raise ValueError(f"invalid calibration score for {k!r}: {v!r}")
                try:
                    f = float(v)
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"invalid calibration score for {k!r}: {v!r}") from exc
                if not math.isfinite(f):
                    raise ValueError(f"non-finite calibration score for {k!r}: {v!r}")
                raw.append(f)
        if not raw:
            raise ValueError("no calibration scores to calibrate on")
        cal = sorted(raw)
        idx = min(len(cal) - 1, max(0, int(0.01 * len(cal))))
        return {"cutoff": cal[idx], "n": len(cal)}

    def decide(self, task: Task, score: float, rule: dict) -> Verdict:
        sc = float(score)
        cut = float(rule["cutoff"])
        if not (math.isfinite(sc) and math.isfinite(cut)):
            raise ValueError(f"non-finite decide input: score={score!r}, cutoff={rule.get('cutoff')!r}")
        flagged = sc < cut
        return Verdict(task.id, 0, flagged == (task.meta["label"] == "ai"), {},
                       details={"score": sc, "p_human": _sigmoid(sc), "cutoff": cut,
                                "pred": "ai" if flagged else "human", "label": task.meta["label"],
                                "kind": task.meta["kind"]})

    def proxy_spec(self, source_scores: dict[str, float], n_neg: int = 1000, n_pos: int = 600,
                   seed: int = 0) -> list[dict]:
        """Parts of the log-odds store, in store order: random calibration humans + near-cutoff search AI docs."""
        if isinstance(n_neg, bool) or not isinstance(n_neg, numbers.Integral) or int(n_neg) < 0:
            raise ValueError(f"n_neg must be a non-negative int, got {n_neg!r}")
        if isinstance(n_pos, bool) or not isinstance(n_pos, numbers.Integral) or int(n_pos) < 0:
            raise ValueError(f"n_pos must be a non-negative int, got {n_pos!r}")
        n_neg, n_pos = int(n_neg), int(n_pos)
        cut = self.calibrate(source_scores)["cutoff"]
        cal = [t for t in self._tasks if t.split == "calibration"]
        neg = np.sort(np.random.default_rng(seed).choice(len(cal), min(n_neg, len(cal)), replace=False))
        ai = [(i, t) for i, t in enumerate(self._tasks) if t.split == "search" and t.meta["label"] == "ai"]
        ai_scores: dict[str, float] = {}
        for _, t in ai:
            if t.id not in source_scores:
                raise KeyError(f"missing search score for {t.id!r}")
            try:
                val = float(source_scores[t.id])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"invalid search score for {t.id!r}: {source_scores[t.id]!r}") from exc
            if not math.isfinite(val):
                raise ValueError(f"non-finite search score for {t.id!r}: {source_scores[t.id]!r}")
            ai_scores[t.id] = val
        near = sorted(sorted(ai, key=lambda x: (abs(ai_scores[x[1].id] - cut), x[0]))[:n_pos], key=lambda x: x[0])
        return [{"name": "humans", "role": "neg", "n": len(neg), "ids": [cal[i].id for i in neg]},
                {"name": "ai", "role": "pos", "n": len(near), "ids": [t.id for _, t in near]}]

    def stats_texts(self, n: int, seed: int = 0) -> list[str]:
        """n raw prompt strings from the search split (a fixed random sample) for the importance matrix."""
        if isinstance(n, bool) or not isinstance(n, numbers.Integral):
            raise ValueError(f"n must be an int, got {n!r}")
        if int(n) <= 0:
            return []
        n = int(n)
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
        prompt_str = self.render_prompt(task.prompt["text"])
        bos = getattr(self.tok, "bos_token", None)
        srv_prompt = prompt_str[len(bos):] if (bos and prompt_str.startswith(bos)) else prompt_str
        skip = set(_DECODING_SKIP) | {
            "prompt", "n_predict", "n_probs", "grammar", "post_sampling_probs",
            "temperature", "top_k", "top_p", "min_p",
        }
        extra = {k: v for k, v in getattr(client, "decoding", {}).items() if k not in skip}
        body = {
            **extra,
            "prompt": srv_prompt,
            "n_predict": 1,
            "n_probs": 10,
            "temperature": 1.0,
            "top_k": 0,
            "top_p": 1.0,
            "min_p": 0.0,
            "grammar": GRAMMAR,
            "post_sampling_probs": True,
            "seed": int(seed),
            "cache_prompt": bool(getattr(client, "cache_prompt", False)),
        }
        payload = json.dumps(body).encode("utf-8")
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": task.prompt["text"]}]
        retries = max(1, int(getattr(client, "retries", 3)))
        last_err: Exception | None = None
        p: float | None = None
        for attempt in range(retries):
            try:
                req = urllib.request.Request(client.base_url + "/completion", data=payload,
                                             headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=client.timeout) as r:
                    out = json.loads(r.read())
                if not isinstance(out, dict):
                    raise TypeError(f"expected JSON object, got {type(out).__name__}")
                probs = out["completion_probabilities"]
                if not isinstance(probs, list) or not probs:
                    raise KeyError("completion_probabilities")
                p = _p_human(probs[0])
                break
            except (
                urllib.error.URLError,
                http.client.HTTPException,
                ConnectionError,
                TimeoutError,
                KeyError,
                IndexError,
                TypeError,
                AttributeError,
                UnicodeDecodeError,
                ValueError,
                json.JSONDecodeError,
            ) as e:
                last_err = e
                if attempt + 1 < retries:
                    time.sleep(1.0 + attempt)
        if p is None:
            return Episode(task.id, seed, msgs, None, None, ["error"], 0, 0, time.time() - t0, infra_error=repr(last_err))
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
    if not isinstance(entry, dict):
        raise TypeError(f"expected probability entry object, got {type(entry).__name__}")
    top = entry.get("top_probs")
    if not isinstance(top, list):
        raise ValueError("missing top_probs list")
    pr = {}
    for t in top:
        if not isinstance(t, dict):
            raise TypeError(f"expected top_probs element object, got {type(t).__name__}")
        pr[t["token"]] = float(t["prob"])
    h, a = pr.get(ANSWERS[0], 0.0), pr.get(ANSWERS[1], 0.0)
    if not (math.isfinite(h) and math.isfinite(a)):
        raise ValueError("non-finite answer token probabilities")
    if h + a <= 0:
        raise KeyError("neither answer token among the returned probabilities")
    return h / (h + a)


def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    z = math.exp(x)
    return z / (1.0 + z)


def _gguf_vocab_hash(meta: dict) -> str:
    h = hashlib.sha256()
    for section, items in ((b"T", meta["tokens"]), (b"M", meta["merges"])):
        h.update(section)
        h.update(struct.pack("<I", len(items)))
        for item in items:
            h.update(struct.pack("<I", len(item)))
            h.update(item)
    if "token_type" in meta:
        h.update(b"Y")
        h.update(np.asarray(meta["token_type"], dtype="<i4").tobytes())
    return h.hexdigest()[:16]


def make_environment(**kwargs) -> DetectEnvironment:
    return DetectEnvironment(**kwargs)
