"""Public calibrated single-token classifier benchmark across Qwen3, Qwen3.5, and Gemma 4.

Implements the full Chantless `ClassifierHooks` contract (`logodds` + `BatchScorer` + automatic
`tokdocs` retokenization and answer-ID remapping on `vocab_keep`-trimmed GGUFs) over public
HuggingFace polarity/moderation datasets (defaulting to `fancyzhx/yelp_polarity`, Apache/public,
zero private files) or a local `jsonl_file` override for offline testing.
"""
from __future__ import annotations

import hashlib
import json
import struct
import threading
from pathlib import Path
from typing import Any

import numpy as np

from .. import vocab as vocabmod
from .protocol import Episode, Task, Verdict

DEFAULT_DATASET = "fancyzhx/yelp_polarity"
DEFAULT_SYSTEM = "Classify the sentiment of the user's review with one word: positive or negative."
DEFAULT_ANSWERS = ("negative", "positive")  # lower score = more toward `pos` class (`negative` = flagged class)


def _load_public_cls(cache_dir: Path, dataset_name: str, max_cal: int, max_eval: int) -> list[dict[str, Any]]:
    slug = dataset_name.replace("/", "__")
    cache_p = cache_dir / f"cls_{slug}_{max_cal}_{max_eval}.jsonl"
    if cache_p.exists():
        return [json.loads(line) for line in cache_p.read_text(encoding="utf-8").splitlines() if line.strip()]
    from datasets import load_dataset
    ds_train = load_dataset(dataset_name, split="train")
    ds_test = load_dataset(dataset_name, split="test")
    rows: list[dict[str, Any]] = []
    # Calibration negatives (`positive` reviews, label==1 in yelp_polarity) + search/train pool from train split
    n_cal_neg = 0
    n_pool = 0
    for idx, r in enumerate(ds_train):
        txt = str(r.get("text", "")).strip()
        if len(txt) < 40 or len(txt) > 3000:
            continue
        raw_lbl = int(r["label"])  # yelp_polarity: 0 = negative (flagged pos), 1 = positive (unflagged neg)
        cls_lbl = "negative" if raw_lbl == 0 else "positive"
        if cls_lbl == "positive" and n_cal_neg < max_cal:
            rows.append({"id": f"cal-{idx}", "text": txt, "label": cls_lbl, "split": "calibration", "family": f"cal/b{idx % 32:02d}"})
            n_cal_neg += 1
        elif n_pool < max_eval:
            b = int(hashlib.sha256(f"tr:{idx}".encode()).hexdigest()[:8], 16) % 60
            sp = "search" if b < 25 else ("select" if b < 50 else "train")
            rows.append({"id": f"tr-{idx}", "text": txt, "label": cls_lbl, "split": sp, "family": f"{sp}/b{b:02d}"})
            n_pool += 1
        if n_cal_neg >= max_cal and n_pool >= max_eval:
            break
    n_ex = 0
    for idx, r in enumerate(ds_test):
        txt = str(r.get("text", "")).strip()
        if len(txt) < 40 or len(txt) > 3000:
            continue
        cls_lbl = "negative" if int(r["label"]) == 0 else "positive"
        b = int(hashlib.sha256(f"ex:{idx}".encode()).hexdigest()[:8], 16) % 40
        rows.append({"id": f"ex-{idx}", "text": txt, "label": cls_lbl, "split": "exam", "family": f"exam/b{b:02d}"})
        n_ex += 1
        if n_ex >= max_eval // 2:
            break
    cache_dir.mkdir(parents=True, exist_ok=True)
    with cache_p.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
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
        self.hf_dir = str(hf_dir)
        self.tok = AutoTokenizer.from_pretrained(self.hf_dir)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.system_prompt = str(system_prompt)
        # `answers = (flagged_word, unflagged_word)`; `score = logit[unflagged] - logit[flagged]` so lower = more flagged
        self.flagged_word, self.unflagged_word = str(answers[0]), str(answers[1])
        self.fpr_target = float(fpr_target)
        self.max_doc_tokens = int(max_doc_tokens)
        self.budget = {"max_tokens": 1, "max_doc_tokens": self.max_doc_tokens}
        self.deployment = {"context": int(context), "concurrency": 1, "kv_type": str(kv_type), "max_total_memory_bytes": max_total_memory_bytes}

        # Resolve single-token IDs `(unflagged_id, flagged_id)` in source HF tokenizer
        u_ids = self.tok.encode(self.unflagged_word, add_special_tokens=False)
        f_ids = self.tok.encode(self.flagged_word, add_special_tokens=False)
        if len(u_ids) != 1 or len(f_ids) != 1:
            raise ValueError(f"answers {(self.flagged_word, self.unflagged_word)} must each be 1 token, got {f_ids}, {u_ids}")
        self.pair = (int(u_ids[0]), int(f_ids[0]))
        self._orig_tok_bytes = {
            self.unflagged_word: self.tok.convert_ids_to_tokens(self.pair[0]).encode("utf-8") if isinstance(self.tok.convert_ids_to_tokens(self.pair[0]), str) else bytes(self.tok.convert_ids_to_tokens(self.pair[0])),
            self.flagged_word: self.tok.convert_ids_to_tokens(self.pair[1]).encode("utf-8") if isinstance(self.tok.convert_ids_to_tokens(self.pair[1]), str) else bytes(self.tok.convert_ids_to_tokens(self.pair[1])),
        }

        if jsonl_file:
            rows = [json.loads(line) for line in Path(jsonl_file).read_text(encoding="utf-8").splitlines() if line.strip()]
        else:
            rows = _load_public_cls(self.cache_dir, dataset_name, int(max_cal), int(max_eval))

        self._tasks: list[Task] = []
        self._prompt_str: dict[str, str] = {}
        for r in rows:
            tid = str(r["id"])
            txt = str(r["text"])
            enc = self.tok(txt, add_special_tokens=False)
            if len(enc["input_ids"]) > self.max_doc_tokens:
                txt = self.tok.decode(enc["input_ids"][: self.max_doc_tokens], skip_special_tokens=True)
            msgs = [{"role": "system", "content": self.system_prompt}, {"role": "user", "content": txt}]
            try:
                pstr = self.tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, enable_thinking=False)
            except TypeError:
                pstr = self.tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
            self._prompt_str[tid] = pstr
            self._tasks.append(Task(
                id=tid,
                family=str(r["family"]),
                split=str(r["split"]),
                prompt={"text": txt},
                meta={"label": str(r["label"])},
            ))
        self._lock = threading.Lock()
        self._src_vocab_n: int | None = None

    def tasks(self) -> list[Task]:
        return list(self._tasks)

    def stats_texts(self, n: int = 200) -> list[str]:
        return [self._prompt_str[t.id] for t in self._tasks if t.split == "search"][:n]

    def fit_token_counts(self, gguf_path: str, tc: Any) -> np.ndarray:
        from .. import runtime
        v = vocabmod.read_tokenizer(gguf_path)["vocab_size"]
        counts = np.zeros(v, dtype=np.int64)
        sample = [self._prompt_str[t.id] for t in self._tasks if t.split in ("calibration", "search")]
        for arr in runtime.tokenize_docs(tc, gguf_path, sample):
            if len(arr):
                counts += np.bincount(arr, minlength=v)[:v]
        return counts

    def trj_for(self, tasks: list[Task], model_gguf: str | None = None, tc: Any = None, answer_tok: int | None = None) -> Path:
        key_src = f"{model_gguf or 'hf'}:{len(tasks)}:" + hashlib.sha256(",".join(t.id for t in tasks).encode()).hexdigest()[:16]
        out = self.cache_dir / f"trj-{hashlib.sha256(key_src.encode()).hexdigest()[:16]}.bin"
        if out.exists():
            return out
        prompts = [self._prompt_str[t.id] for t in tasks]
        ans = int(self.pair[0] if answer_tok is None else answer_tok)
        if model_gguf is not None and tc is not None and getattr(tc, "tokdocs", None) is not None:
            from .. import runtime
            tok_seqs = [x.tolist() for x in runtime.tokenize_docs(tc, model_gguf, prompts)]
        else:
            tok_seqs = [self.tok(p, add_special_tokens=False)["input_ids"] for p in prompts]
        part = out.with_suffix(".part")
        with part.open("wb") as f:
            f.write(b"TRJ1" + struct.pack("<I", len(tok_seqs)))
            for seq in tok_seqs:
                full = seq + [ans]
                f.write(struct.pack("<II", len(full), 1))
                f.write(struct.pack(f"<{len(full)}i", *full))
                f.write(struct.pack("<I", len(seq)))
        part.replace(out)
        return out

    def _pair_for_model(self, model_gguf: str) -> tuple[tuple[int, int], bool]:
        meta = vocabmod.read_tokenizer(model_gguf)
        if self._src_vocab_n is None:
            self._src_vocab_n = len(self.tok)
        v = meta["vocab_size"]
        # Check exact bytes at `self.pair` first
        if (
            self.pair[0] < v and self.pair[1] < v
            and self.unflagged_word.encode("utf-8") in meta["tokens"][self.pair[0]]
            and self.flagged_word.encode("utf-8") in meta["tokens"][self.pair[1]]
        ):
            return self.pair, False
        # Resolve in trimmed GGUF vocabulary (matching raw UTF-8 or SentencePiece/GPT2 space-prefixed form)
        idx = {}
        for i, b in enumerate(meta["tokens"]):
            clean = b.replace("▁".encode("utf-8"), b"").replace("Ġ".encode("utf-8"), b"")
            if clean not in idx:
                idx[clean] = i
        u_id = idx[self.unflagged_word.encode("utf-8")]
        f_id = idx[self.flagged_word.encode("utf-8")]
        return (int(u_id), int(f_id)), True

    def score(self, tasks: list[Task], model: str, scorer: Any) -> dict[str, float]:
        pair_use, trimmed = self._pair_for_model(model)
        trj = self.trj_for(tasks, model_gguf=model if trimmed else None, tc=scorer.tc if trimmed else None, answer_tok=pair_use[0])
        vals = scorer.score_trj(model, str(trj), pair_use)
        return {t.id: float(v) for t, v in zip(tasks, vals)}

    def calibrate(self, scores: dict[str, float]) -> dict:
        arr = np.sort(np.asarray(list(scores.values()), dtype=np.float64))
        idx = min(len(arr) - 1, max(0, int(self.fpr_target * len(arr))))
        return {"cutoff": float(arr[idx]), "fpr_target": self.fpr_target, "n_cal": len(arr)}

    def decide(self, task: Task, score: float, rule: dict) -> Verdict:
        pred = self.flagged_word if float(score) < float(rule["cutoff"]) else self.unflagged_word
        ok = pred == task.meta["label"]
        return Verdict(task.id, 0, ok, {}, details={"score": float(score), "pred": pred, "cutoff": rule["cutoff"]})

    def proxy_spec(self, src_scores: dict[str, float], n_neg: int = 400, n_pos: int = 200) -> list[dict]:
        by_id = {t.id: t for t in self._tasks}
        cal_ids = sorted(i for i in src_scores if by_id[i].split == "calibration")
        rng = np.random.default_rng(20261118)
        neg_pick = [cal_ids[i] for i in sorted(rng.permutation(len(cal_ids))[: min(n_neg, len(cal_ids))])]
        s_pos = [i for i in src_scores if by_id[i].split == "search" and by_id[i].meta["label"] == self.flagged_word]
        # Pick boundary/hard positives closest to calibration cutoff plus random positives
        cal_cut = float(np.sort([src_scores[i] for i in cal_ids])[max(0, int(self.fpr_target * len(cal_ids)))]) if cal_ids else 0.0
        s_pos.sort(key=lambda i: abs(src_scores[i] - cal_cut))
        pos_pick = s_pos[: min(n_pos, len(s_pos))]
        return [
            {"name": "cal_neg", "role": "neg", "n": len(neg_pick), "ids": neg_pick},
            {"name": "search_pos", "role": "pos", "n": len(pos_pick), "ids": pos_pick},
        ]

    def run(self, task: Task, client: Any, seed: int) -> Episode:
        raise NotImplementedError("classifier environment uses BatchScorer via Evaluator._run_classifier")

    def verify(self, task: Task, episode: Episode) -> Verdict:
        raise NotImplementedError("classifier environment uses decide()")


def make_environment(**kwargs) -> PublicClassifierEnvironment:
    return PublicClassifierEnvironment(**kwargs)
