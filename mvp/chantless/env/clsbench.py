"""Calibrated single-token classifier benchmark across Qwen3, Qwen3.5, and Gemma 4.

Implements `ClassifierHooks` over HuggingFace polarity datasets (default: `fancyzhx/yelp_polarity`)
or a local `jsonl_file`, with answer-ID remapping and `tokdocs` retokenization for trimmed vocabularies.
"""
from __future__ import annotations

import hashlib
import json
import math
import numbers
import os
import struct
import threading
from pathlib import Path
from typing import Any

import numpy as np

from .. import vocab as vocabmod
from .bfcl import detect_dialect, render_chat_prompt
from .protocol import Episode, Task, Verdict

DEFAULT_DATASET = "fancyzhx/yelp_polarity"
DEFAULT_SYSTEM = "Classify the sentiment of the user's review with one word: positive or negative."
DEFAULT_ANSWERS = ("negative", "positive")  # lower score = more toward `pos` class (`negative` = flagged class)
_EVAL_CYCLE = (
    "search", "select", "train",
    "search", "select",
    "search", "select",
    "search", "select",
    "search", "select", "train",
)


def _iter_shuffled(ds: Any, seed: int):
    if callable(getattr(ds, "shuffle", None)):
        return enumerate(ds.shuffle(seed=seed))
    if hasattr(ds, "__len__") and hasattr(ds, "__getitem__"):
        order = np.random.default_rng(seed).permutation(len(ds))
        return ((int(i), ds[int(i)]) for i in order)
    return enumerate(ds)


def _tok_bytes(tok: Any, tok_id: int, fallback_word: str) -> bytes:
    if callable(getattr(tok, "convert_ids_to_tokens", None)):
        tok_s = tok.convert_ids_to_tokens(int(tok_id))
        if isinstance(tok_s, str):
            return tok_s.encode("utf-8")
        if isinstance(tok_s, (bytes, bytearray)):
            return bytes(tok_s)
    if callable(getattr(tok, "get_vocab", None)):
        for s, i in tok.get_vocab().items():
            if int(i) == int(tok_id):
                return s.encode("utf-8")
    return fallback_word.encode("utf-8")


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


def _load_public_cls(
    cache_dir: Path,
    dataset_name: str,
    max_cal: int,
    max_eval: int,
    seed: int = 20261118,
) -> list[dict[str, Any]]:
    if (
        isinstance(max_cal, bool) or not isinstance(max_cal, numbers.Integral) or int(max_cal) < 0
        or isinstance(max_eval, bool) or not isinstance(max_eval, numbers.Integral) or int(max_eval) < 0
    ):
        raise ValueError(f"max_cal and max_eval must be non-negative ints, got {max_cal!r}, {max_eval!r}")
    max_cal, max_eval = int(max_cal), int(max_eval)
    slug = dataset_name.replace("/", "__")
    cache_p = cache_dir / f"cls_{slug}_{max_cal}_{max_eval}.jsonl"
    if cache_p.exists():
        return [json.loads(line) for line in cache_p.read_text(encoding="utf-8").splitlines() if line.strip()]
    from datasets import load_dataset
    ds_train = load_dataset(dataset_name, split="train")
    ds_test = load_dataset(dataset_name, split="test")

    need_eval_pos = max_eval - (max_eval // 2)
    need_eval_neg = max_eval // 2
    cal_pos: list[tuple[int, str]] = []
    eval_pos: list[tuple[int, str]] = []
    eval_neg: list[tuple[int, str]] = []
    for idx, r in _iter_shuffled(ds_train, seed):
        txt = str(r.get("text", "")).strip()
        if len(txt) < 40 or len(txt) > 3000:
            continue
        raw_lbl = int(r["label"])  # yelp_polarity: 0 = negative (flagged pos), 1 = positive (unflagged neg)
        cls_lbl = "negative" if raw_lbl == 0 else "positive"
        if cls_lbl == "positive":
            if len(cal_pos) < max_cal and len(eval_pos) < need_eval_pos:
                if len(cal_pos) * need_eval_pos <= len(eval_pos) * max_cal:
                    cal_pos.append((idx, txt))
                else:
                    eval_pos.append((idx, txt))
            elif len(cal_pos) < max_cal:
                cal_pos.append((idx, txt))
            elif len(eval_pos) < need_eval_pos:
                eval_pos.append((idx, txt))
        else:
            if len(eval_neg) < need_eval_neg:
                eval_neg.append((idx, txt))
        if len(cal_pos) >= max_cal and len(eval_pos) >= need_eval_pos and len(eval_neg) >= need_eval_neg:
            break

    rows: list[dict[str, Any]] = []
    for idx, txt in cal_pos:
        rows.append({
            "id": f"cal-{idx}",
            "text": txt,
            "label": "positive",
            "split": "calibration",
            "family": f"cal/b{idx % 32:02d}",
        })

    def _assign_eval(items: list[tuple[int, str]], cls_lbl: str) -> list[dict[str, Any]]:
        out_r = []
        for i, (idx, txt) in enumerate(items):
            sp = _EVAL_CYCLE[i % len(_EVAL_CYCLE)]
            b = int(hashlib.sha256(f"tr:{idx}".encode()).hexdigest()[:8], 16) % 60
            out_r.append({
                "id": f"tr-{idx}",
                "text": txt,
                "label": cls_lbl,
                "split": sp,
                "family": f"{sp}/b{b:02d}",
            })
        return out_r

    pos_eval_rows = _assign_eval(eval_pos, "positive")
    neg_eval_rows = _assign_eval(eval_neg, "negative")
    for i in range(max(len(pos_eval_rows), len(neg_eval_rows))):
        if i < len(pos_eval_rows):
            rows.append(pos_eval_rows[i])
        if i < len(neg_eval_rows):
            rows.append(neg_eval_rows[i])

    need_ex_total = max_eval // 2
    need_ex_pos = need_ex_total - (need_ex_total // 2)
    need_ex_neg = need_ex_total // 2
    ex_pos: list[dict[str, Any]] = []
    ex_neg: list[dict[str, Any]] = []
    for idx, r in _iter_shuffled(ds_test, seed + 1):
        txt = str(r.get("text", "")).strip()
        if len(txt) < 40 or len(txt) > 3000:
            continue
        cls_lbl = "negative" if int(r["label"]) == 0 else "positive"
        b = int(hashlib.sha256(f"ex:{idx}".encode()).hexdigest()[:8], 16) % 40
        row = {"id": f"ex-{idx}", "text": txt, "label": cls_lbl, "split": "exam", "family": f"exam/b{b:02d}"}
        if cls_lbl == "positive" and len(ex_pos) < need_ex_pos:
            ex_pos.append(row)
        elif cls_lbl == "negative" and len(ex_neg) < need_ex_neg:
            ex_neg.append(row)
        if len(ex_pos) >= need_ex_pos and len(ex_neg) >= need_ex_neg:
            break
    for i in range(max(len(ex_pos), len(ex_neg))):
        if i < len(ex_pos):
            rows.append(ex_pos[i])
        if i < len(ex_neg):
            rows.append(ex_neg[i])

    cache_dir.mkdir(parents=True, exist_ok=True)
    tmp = cache_p.with_name(f"{cache_p.name}.{os.getpid()}-{threading.get_ident()}.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, cache_p)
    return rows


class PublicClassifierEnvironment:
    name = "clsbench"
    version = "1"
    classifier = True
    tools = None

    def __init__(
        self,
        hf_dir: str,
        cache_dir: str = "/tmp/chantless-clsbench-cache",
        jsonl_file: str | None = None,
        dataset_name: str = DEFAULT_DATASET,
        system_prompt: str = DEFAULT_SYSTEM,
        answers: tuple[str, str] | list[str] = DEFAULT_ANSWERS,
        fpr_target: float = 0.05,
        max_cal: int = 4000,
        max_eval: int = 6000,
        max_doc_tokens: int = 1024,
        context: int = 2048,
        kv_type: str = "f16",
        max_total_memory_bytes: int | None = None,
    ):
        from transformers import AutoTokenizer
        if isinstance(answers, (str, bytes)) or not hasattr(answers, "__len__") or len(answers) != 2:
            raise ValueError(f"answers must have 2 elements (flagged, unflagged), got {answers!r}")
        flagged_word, unflagged_word = str(answers[0]), str(answers[1])
        if not flagged_word or not unflagged_word or flagged_word == unflagged_word:
            raise ValueError(f"answers must be two distinct non-empty strings (flagged, unflagged), got {answers!r}")
        if isinstance(fpr_target, bool) or not math.isfinite(float(fpr_target)) or not (0.0 <= float(fpr_target) <= 1.0):
            raise ValueError(f"fpr_target must be finite in [0, 1], got {fpr_target!r}")
        if isinstance(max_doc_tokens, bool) or not isinstance(max_doc_tokens, numbers.Integral) or int(max_doc_tokens) <= 0:
            raise ValueError(f"max_doc_tokens must be a positive int, got {max_doc_tokens!r}")
        if isinstance(context, bool) or not isinstance(context, numbers.Integral) or int(context) <= 0:
            raise ValueError(f"context must be a positive int, got {context!r}")
        if not isinstance(kv_type, str) or not kv_type.strip():
            raise ValueError(f"kv_type must be a non-empty str, got {kv_type!r}")
        if max_total_memory_bytes is not None and (
            isinstance(max_total_memory_bytes, bool)
            or not isinstance(max_total_memory_bytes, numbers.Integral)
            or int(max_total_memory_bytes) <= 0
        ):
            raise ValueError(f"max_total_memory_bytes must be a positive int or None, got {max_total_memory_bytes!r}")
        if jsonl_file is None and (
            isinstance(max_cal, bool) or not isinstance(max_cal, numbers.Integral) or int(max_cal) < 0
            or isinstance(max_eval, bool) or not isinstance(max_eval, numbers.Integral) or int(max_eval) < 0
        ):
            raise ValueError(f"max_cal and max_eval must be non-negative ints, got {max_cal!r}, {max_eval!r}")
        self.hf_dir = str(hf_dir)
        self.tok = AutoTokenizer.from_pretrained(self.hf_dir)
        self.dialect, _ = detect_dialect(self.tok)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.system_prompt = str(system_prompt)
        # `answers = (flagged_word, unflagged_word)`; `score = logit[unflagged] - logit[flagged]` so lower = more flagged
        self.flagged_word, self.unflagged_word = flagged_word, unflagged_word
        self.fpr_target = float(fpr_target)
        self.max_doc_tokens = int(max_doc_tokens)
        self.budget = {"max_tokens": 1, "max_doc_tokens": self.max_doc_tokens}
        mem_limit = int(max_total_memory_bytes) if max_total_memory_bytes is not None else None
        self.deployment = {"context": int(context), "concurrency": 1, "kv_type": kv_type, "max_total_memory_bytes": mem_limit}

        # Resolve single-token IDs `(unflagged_id, flagged_id)` in source HF tokenizer
        u_ids = self.tok.encode(self.unflagged_word, add_special_tokens=False)
        f_ids = self.tok.encode(self.flagged_word, add_special_tokens=False)
        if len(u_ids) != 1 or len(f_ids) != 1:
            raise ValueError(f"answers {(self.flagged_word, self.unflagged_word)} must each be 1 token, got {f_ids}, {u_ids}")
        self.pair = (int(u_ids[0]), int(f_ids[0]))
        if self.pair[0] == self.pair[1]:
            raise ValueError(
                f"answers {(self.flagged_word, self.unflagged_word)} collapsed to identical token id {self.pair[0]}"
            )
        self._orig_tok_bytes = {
            self.unflagged_word: _tok_bytes(self.tok, self.pair[0], self.unflagged_word),
            self.flagged_word: _tok_bytes(self.tok, self.pair[1], self.flagged_word),
        }
        self._src_vocab_n: int | None = (
            len(self.tok)
            if hasattr(self.tok, "__len__")
            else (len(self.tok.get_vocab()) if callable(getattr(self.tok, "get_vocab", None)) else None)
        )
        backend = getattr(self.tok, "backend_tokenizer", None)
        ident = backend.to_str() if backend is not None else f"{Path(self.hf_dir).resolve()}:{self._src_vocab_n}"
        self._tok_id = hashlib.sha256(ident.encode("utf-8")).hexdigest()
        self._template_sig = self._render_template("")

        if jsonl_file:
            rows = [json.loads(line) for line in Path(jsonl_file).read_text(encoding="utf-8").splitlines() if line.strip()]
            seen_ids: set[str] = set()
            for i, r in enumerate(rows):
                if not isinstance(r, dict) or any(k not in r for k in ("id", "text", "label", "split")):
                    raise ValueError(f"jsonl row {i} missing required keys (id, text, label, split)")
                tid = str(r["id"])
                if tid in seen_ids:
                    raise ValueError(f"duplicate task id {tid!r} in jsonl_file at row {i}")
                seen_ids.add(tid)
                lbl = str(r["label"])
                if lbl not in (self.flagged_word, self.unflagged_word):
                    raise ValueError(f"jsonl row {r['id']!r} has invalid label {lbl!r}; expected {(self.flagged_word, self.unflagged_word)}")
                if str(r["split"]) == "calibration" and lbl != self.unflagged_word:
                    raise ValueError(f"calibration row {r['id']!r} must have unflagged label {self.unflagged_word!r}, got {lbl!r}")
        else:
            rows = _load_public_cls(self.cache_dir, dataset_name, int(max_cal), int(max_eval))

        self._tasks: list[Task] = []
        self._prompt_str: dict[str, str] = {}
        self._prompt_hash: dict[str, bytes] = {}
        heads = self._heads([str(r["text"]) for r in rows])
        for r, (txt, cut) in zip(rows, heads):
            tid = str(r["id"])
            sp = str(r["split"])
            lbl = str(r["label"])
            fam = str(r.get("family", f"{sp}/{tid}"))
            pstr = self._render_template(txt)
            self._prompt_str[tid] = pstr
            self._prompt_hash[tid] = hashlib.sha256(pstr.encode("utf-8")).digest()
            self._tasks.append(Task(
                id=tid,
                family=fam,
                split=sp,
                prompt={"text": txt},
                meta={"label": lbl, "truncated": cut},
            ))
        self._by_id = {t.id: t for t in self._tasks}
        self._cal_ids = {t.id for t in self._tasks if t.split == "calibration" and t.meta.get("label") == self.unflagged_word}
        self._lock = threading.Lock()

    def _render_template(self, txt: str) -> str:
        msgs = [{"role": "system", "content": self.system_prompt}, {"role": "user", "content": txt}]
        pstr = render_chat_prompt(self.tok, msgs, self.dialect)
        if self.dialect == "inkling2" and pstr.endswith("<|end_message|>"):
            pstr += "<|message_model|><|content_text|>"
        return pstr

    def _heads(self, texts: list[str], chunk: int = 256) -> list[tuple[str, bool]]:
        out: list[tuple[str, bool]] = []
        n = self.max_doc_tokens
        for i in range(0, len(texts), chunk):
            part = texts[i : i + chunk]
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

    def tasks(self) -> list[Task]:
        return list(self._tasks)

    def stats_texts(self, n: int = 200, seed: int = 0) -> list[str]:
        if isinstance(n, bool) or not isinstance(n, numbers.Integral):
            raise ValueError(f"n must be an int, got {n!r}")
        if int(n) <= 0:
            return []
        by_lbl: dict[str, list[Task]] = {}
        for t in self._tasks:
            if t.split == "search":
                by_lbl.setdefault(str(t.meta.get("label", "")), []).append(t)
        if not by_lbl:
            return []
        rng = np.random.default_rng(seed)
        buckets: dict[str, list[str]] = {}
        for lbl in sorted(by_lbl):
            ts = sorted(by_lbl[lbl], key=lambda t: t.id)
            order = rng.permutation(len(ts))
            buckets[lbl] = [self._prompt_str[ts[int(idx)].id] for idx in order]
        out: list[str] = []
        pos = 0
        labels = sorted(buckets)
        while len(out) < int(n):
            added = False
            for lbl in labels:
                if pos < len(buckets[lbl]):
                    out.append(buckets[lbl][pos])
                    added = True
                    if len(out) >= int(n):
                        break
            if not added:
                break
            pos += 1
        return out

    def _encode_prompts(self, prompts: list[str], chunk: int = 256) -> list[list[int]]:
        seqs: list[list[int]] = []
        for i in range(0, len(prompts), chunk):
            seqs.extend(self.tok(prompts[i : i + chunk], add_special_tokens=False)["input_ids"])
        return seqs

    def _task_prompt(self, t: Task) -> str:
        p = self._prompt_str.get(t.id)
        if p is not None:
            return p
        return self._render_template(str(t.prompt["text"]))

    def _key(self, tasks: list[Task], vocab_sig: str, answer_tok: int) -> str:
        h = hashlib.sha256()
        for part in (vocab_sig, str(int(answer_tok)), repr(self.pair), self._template_sig, str(self.max_doc_tokens)):
            b = part.encode("utf-8")
            h.update(struct.pack("<I", len(b)))
            h.update(b)
        h.update(struct.pack("<I", len(tasks)))
        for t in tasks:
            tid_b = t.id.encode("utf-8")
            sp_b = t.split.encode("utf-8")
            ph = self._prompt_hash.get(t.id)
            if ph is None:
                ph = hashlib.sha256(self._task_prompt(t).encode("utf-8")).digest()
            h.update(struct.pack("<I", len(tid_b)))
            h.update(tid_b)
            h.update(struct.pack("<I", len(sp_b)))
            h.update(sp_b)
            h.update(ph)
        return h.hexdigest()[:24]

    def token_counts(self) -> dict[str, dict[str, int]]:
        """{split: {'docs', 'tokens'}}, tokens being prompt plus answer token each forward pass reads."""
        path = self.cache_dir / f"tokens-{self._key(self._tasks, f'hf:{self._tok_id}', self.pair[0])}.json"
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        out: dict[str, dict[str, int]] = {}
        prompts = [self._prompt_str[t.id] for t in self._tasks]
        for t, ids in zip(self._tasks, self._encode_prompts(prompts)):
            c = out.setdefault(t.split, {"docs": 0, "tokens": 0})
            c["docs"] += 1
            c["tokens"] += len(ids) + 1
        tmp = path.with_name(f"{path.name}.{os.getpid()}-{threading.get_ident()}.tmp")
        tmp.write_text(json.dumps(out), encoding="utf-8")
        os.replace(tmp, path)
        return out

    def _is_exact_untrimmed(self, meta: dict) -> bool:
        v = int(meta["vocab_size"])
        u_raw = self._orig_tok_bytes[self.unflagged_word]
        f_raw = self._orig_tok_bytes[self.flagged_word]
        return (
            self._src_vocab_n is not None
            and v == self._src_vocab_n
            and self.pair[0] < v
            and self.pair[1] < v
            and meta["tokens"][self.pair[0]] == u_raw
            and meta["tokens"][self.pair[1]] == f_raw
        )

    def fit_token_counts(self, gguf_path: str | None = None, tc: Any = None) -> np.ndarray:
        sample = [self._prompt_str[t.id] for t in self._tasks if t.split in ("calibration", "search")]
        if gguf_path is not None:
            meta = vocabmod.read_tokenizer(gguf_path)
            v = int(meta["vocab_size"])
            counts = np.zeros(v, dtype=np.int64)
            if tc is not None and getattr(tc, "tokdocs", None) is not None:
                from .. import runtime
                if sample:
                    for arr in runtime.tokenize_docs(tc, gguf_path, sample):
                        if len(arr):
                            counts += np.bincount(arr, minlength=v)[:v]
                return counts
            u_raw = self._orig_tok_bytes[self.unflagged_word]
            f_raw = self._orig_tok_bytes[self.flagged_word]
            untrimmed_or_padded = (
                self._src_vocab_n is not None
                and v >= self._src_vocab_n
                and self.pair[0] < v
                and self.pair[1] < v
                and meta["tokens"][self.pair[0]] == u_raw
                and meta["tokens"][self.pair[1]] == f_raw
            )
            if not untrimmed_or_padded:
                raise RuntimeError(f"trimmed GGUF {gguf_path} requires toolchain.tokdocs (refusing untrimmed self.tok fallback)")
        else:
            v = int(self._src_vocab_n if self._src_vocab_n is not None else len(self.tok))
            counts = np.zeros(v, dtype=np.int64)
        for ids in self._encode_prompts(sample):
            if ids:
                counts += np.bincount(np.asarray(ids, dtype=np.int64), minlength=v)[:v]
        return counts

    def trj_for(self, tasks: list[Task], model_gguf: str | None = None, tc: Any = None, answer_tok: int | None = None) -> Path:
        ans = int(self.pair[0] if answer_tok is None else answer_tok)
        if model_gguf is not None:
            meta = vocabmod.read_tokenizer(model_gguf)
            vh = _gguf_vocab_hash(meta)
            vocab_sig = f"gguf:{vh}"
            exact_untrimmed = self._is_exact_untrimmed(meta)
            use_tokdocs = tc is not None and getattr(tc, "tokdocs", None) is not None
            if not exact_untrimmed and not use_tokdocs:
                raise RuntimeError(f"trimmed GGUF {model_gguf} requires toolchain.tokdocs (refusing untrimmed self.tok fallback)")
        else:
            vocab_sig = f"hf:{self._tok_id}"
            use_tokdocs = False
        out = self.cache_dir / f"trj-{self._key(tasks, vocab_sig, ans)}.bin"
        if out.exists():
            return out
        prompts = [self._task_prompt(t) for t in tasks]
        if use_tokdocs:
            from .. import runtime
            tok_seqs = [x.tolist() for x in runtime.tokenize_docs(tc, model_gguf, prompts)]
        else:
            tok_seqs = self._encode_prompts(prompts)
        part = out.with_name(f"{out.name}.{os.getpid()}-{threading.get_ident()}.tmp")
        with part.open("wb") as f:
            f.write(b"TRJ1" + struct.pack("<I", len(tok_seqs)))
            for seq in tok_seqs:
                full = seq + [ans]
                f.write(struct.pack("<II", len(full), 1))
                f.write(struct.pack(f"<{len(full)}i", *full))
                f.write(struct.pack("<I", len(seq)))
        os.replace(part, out)
        return out

    def _pair_for_model(self, model_gguf: str) -> tuple[tuple[int, int], bool]:
        meta = vocabmod.read_tokenizer(model_gguf)
        if self._is_exact_untrimmed(meta):
            return self.pair, False
        raw_idx = {b: i for i, b in enumerate(meta["tokens"])}
        u_raw = self._orig_tok_bytes[self.unflagged_word]
        f_raw = self._orig_tok_bytes[self.flagged_word]
        remapped: list[int] = []
        for w, pid, tb in ((self.unflagged_word, self.pair[0], u_raw), (self.flagged_word, self.pair[1], f_raw)):
            if tb in raw_idx:
                remapped.append(int(raw_idx[tb]))
            else:
                raise RuntimeError(f"trimmed GGUF {model_gguf} dropped answer token {pid} ({w!r} / {tb!r})")
        return (remapped[0], remapped[1]), True

    def score(self, tasks: list[Task], model: str, scorer: Any) -> dict[str, float]:
        if not tasks:
            return {}
        pair_use, trimmed = self._pair_for_model(model)
        tc = getattr(scorer, "tc", None)
        trj = self.trj_for(tasks, model_gguf=model if trimmed else None, tc=tc if trimmed else None, answer_tok=pair_use[0])
        vals = np.asarray(scorer.score_trj(model, str(trj), pair_use), dtype=np.float64)
        if vals.shape != (len(tasks),) or not np.isfinite(vals).all():
            raise RuntimeError(f"scorer returned {vals.shape} values for {len(tasks)} tasks, or non-finite ones")
        return {t.id: float(v) for t, v in zip(tasks, vals)}

    def calibrate(self, scores: dict[str, float]) -> dict:
        cal_vals: list[float] = []
        for tid, val in scores.items():
            t = self._by_id.get(tid)
            if t is not None and t.split == "calibration":
                if t.meta.get("label") != self.unflagged_word:
                    raise ValueError(f"calibration task {tid!r} has flagged label {t.meta.get('label')!r}")
                if isinstance(val, bool):
                    raise ValueError(f"invalid calibration score for {tid!r}: {val!r}")
                try:
                    cal_vals.append(float(val))
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"invalid calibration score for {tid!r}: {val!r}") from exc
        if not cal_vals:
            raise ValueError("no calibration scores to calibrate on")
        arr = np.sort(np.asarray(cal_vals, dtype=np.float64))
        if not np.isfinite(arr).all():
            raise ValueError(f"{int((~np.isfinite(arr)).sum())} non-finite calibration scores")
        idx = min(len(arr) - 1, max(0, int(self.fpr_target * len(arr))))
        return {"cutoff": float(arr[idx]), "fpr_target": self.fpr_target, "n_cal": len(arr)}

    def decide(self, task: Task, score: float, rule: dict) -> Verdict:
        sc = float(score)
        cut = float(rule["cutoff"])
        if not (math.isfinite(sc) and math.isfinite(cut)):
            raise ValueError(f"non-finite decide input: score={score!r}, cutoff={rule.get('cutoff')!r}")
        pred = self.flagged_word if sc < cut else self.unflagged_word
        ok = pred == task.meta["label"]
        return Verdict(task.id, 0, ok, {}, details={"score": sc, "pred": pred, "cutoff": cut})

    def proxy_spec(self, src_scores: dict[str, float], n_neg: int = 400, n_pos: int = 200) -> list[dict]:
        if isinstance(n_neg, bool) or not isinstance(n_neg, numbers.Integral) or int(n_neg) < 0:
            raise ValueError(f"n_neg must be a non-negative int, got {n_neg!r}")
        if isinstance(n_pos, bool) or not isinstance(n_pos, numbers.Integral) or int(n_pos) < 0:
            raise ValueError(f"n_pos must be a non-negative int, got {n_pos!r}")
        n_neg, n_pos = int(n_neg), int(n_pos)
        cal_cut = float(self.calibrate(src_scores)["cutoff"])
        cal_ids = sorted(i for i in src_scores if i in self._cal_ids)
        rng = np.random.default_rng(20261118)
        neg_pick = [cal_ids[i] for i in sorted(rng.permutation(len(cal_ids))[: min(n_neg, len(cal_ids))])]
        s_pos = [
            i for i in src_scores
            if i in self._by_id and self._by_id[i].split == "search" and self._by_id[i].meta["label"] == self.flagged_word
        ]
        for i in s_pos:
            if not math.isfinite(float(src_scores[i])):
                raise ValueError(f"non-finite search score for {i!r}: {src_scores[i]!r}")
        # Pick positives closest to the calibration cutoff
        s_pos.sort(key=lambda i: (abs(float(src_scores[i]) - cal_cut), i))
        pos_pick = s_pos[: min(n_pos, len(s_pos))]
        return [
            {"name": "cal_neg", "role": "neg", "n": len(neg_pick), "ids": neg_pick},
            {"name": "search_pos", "role": "pos", "n": len(pos_pick), "ids": pos_pick},
        ]

    def run(self, task: Task, client: Any, seed: int) -> Episode:
        raise NotImplementedError("classifier environment uses BatchScorer")

    def verify(self, task: Task, episode: Episode) -> Verdict:
        raise NotImplementedError("classifier environment uses decide()")


def make_environment(**kwargs) -> PublicClassifierEnvironment:
    return PublicClassifierEnvironment(**kwargs)
