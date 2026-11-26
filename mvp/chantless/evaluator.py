"""Run episodes for one model across all GPUs (independent evaluations in parallel; results identical to serial).

Infra errors are retried on a fresh request and, if they persist, excluded from statistics and
logged; they never count as task failures.

A classifier environment (env.classifier true) runs no episodes: each distinct task is scored once per
model by the batch scorer and judged at that model's own rule, computed from its calibration-split scores.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import math
import struct
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import gguf_ops
from .client import ChatClient, InfraError
from .env.protocol import Episode, Task
from .runtime import BatchScorer, LlamaServer, Toolchain
from .util import append_jsonl, read_json, sha256_file, write_json

_LOG = logging.getLogger(__name__)


class NoServer:
    """Placeholder for a llama-server on a classifier environment, which scores without one."""
    url = None

    def __init__(self, gpu: str):
        self.gpu = gpu


class Evaluator:
    def __init__(self, tc: Toolchain, env, decoding: dict, slots_per_gpu: int, ctx_per_slot: int, kv_type: str,
                 log_dir: str | Path, infra_retries: int = 2, scorer: BatchScorer | None = None,
                 cache_prompt: bool = False, special: bool | None = None, dialect: str | None = None,
                 flash_attn: str = "on", extra: list[str] | None = None, request_timeout: float = 600.0,
                 use_chat_completions: bool | None = None):
        self.tc, self.env, self.decoding = tc, env, decoding
        self.slots, self.ctx, self.kv = slots_per_gpu, ctx_per_slot, kv_type
        self.cache_prompt = bool(cache_prompt)
        self.flash_attn = "on" if flash_attn is None else str(flash_attn)
        self.extra = list(extra) if extra is not None else []
        self.request_timeout = float(request_timeout)
        self.dialect = dialect if dialect is not None else getattr(env, "dialect", None)
        self._use_chat_completions_override: bool | None = (
            None if use_chat_completions is None else bool(use_chat_completions)
        )
        self.use_chat_completions: bool = self._resolve_use_chat_completions(self.dialect)
        self._special_override = None if special is None else bool(special)
        env_special = getattr(env, "needs_special_tokens", None)
        self._env_special: bool | None = (
            bool(env_special)
            if env_special is not None
            else (True if self.dialect == "inkling2" else None)
        )
        self.special = (
            self._special_override
            if self._special_override is not None
            else bool(self._env_special)
        )
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        if isinstance(infra_retries, bool) or not isinstance(infra_retries, int) or int(infra_retries) < 0:
            raise ValueError(f"infra_retries must be a non-negative int, got {infra_retries!r}")
        self.retries = int(infra_retries)
        self._lock = threading.Lock()
        self._seen_tags: set[str] = set()
        self.scorer = scorer
        self.classifier = bool(getattr(env, "classifier", False))
        self._cls_lock = threading.Lock()
        self._hashes: dict[tuple, str] = {}               # (path, size, mtime_ns) -> model hash
        self._rules: dict[tuple[str, str], dict] = {}     # (model hash, cal fingerprint) -> rule
        self._scores: dict[str, dict[str, float]] = {}    # model hash -> {task id: score}

    def _resolve_special(self, model: str | Path | None) -> bool:
        if self._special_override is not None:
            return self._special_override
        if self._env_special is not None:
            return self._env_special
        if model:
            try:
                return gguf_ops.needs_special_tokens(model)
            except (OSError, ValueError, struct.error) as exc:
                _LOG.warning("cannot auto-detect --special from %s (%s); defaulting to False", model, exc)
                return False
        return False

    def _resolve_dialect(self, model: str | Path | None, srvs) -> str | None:
        if self.dialect is not None:
            return self.dialect
        if model and Path(model).is_file():
            try:
                return "inkling2" if gguf_ops.needs_special_tokens(model) else None
            except (OSError, ValueError, struct.error) as exc:
                _LOG.warning("cannot auto-detect dialect from %s (%s)", model, exc)
                return None
        if self._special_override is None and srvs and any(getattr(s, "special", False) for s in srvs):
            return "inkling2"
        return None

    def _resolve_use_chat_completions(self, dialect: str | None) -> bool:
        if self._use_chat_completions_override is not None:
            return self._use_chat_completions_override
        return dialect == "inkling2" and getattr(self.env, "name", None) == "bfcl"

    @contextlib.contextmanager
    def servers(self, model: str):
        if self.classifier:
            yield [NoServer(g) for g in self.tc.gpus]
            return
        eff_special = self._resolve_special(model)
        with contextlib.ExitStack() as st:
            srvs = [st.enter_context(LlamaServer(self.tc, model, gpu=g, slots=self.slots, ctx_per_slot=self.ctx,
                                                 kv_type=self.kv, log=self.log_dir / f"server-gpu{g or 0}.log",
                                                 cache_prompt=self.cache_prompt, special=eff_special,
                                                 flash_attn=self.flash_attn, extra=self.extra))
                    for g in self.tc.gpus]
            yield srvs

    def run(self, model: str, jobs: list[tuple[Task, int]], tag: str, side: str = "cand", save: bool = True,
            srvs=None) -> list[dict]:
        """jobs: [(task, seed)] -> rows {task, family, split, seed, side, success, tests_passed, detectors, ...}."""
        if self.classifier:
            return self._run_classifier(model, jobs, tag, side, save)
        if srvs is None:
            with self.servers(model) as s:
                return self.run(model, jobs, tag, side, save, s)
        if not srvs:
            raise ValueError("Evaluator.run requires at least one server")
        eff_dialect = self._resolve_dialect(model, srvs)
        eff_use_chat = self._resolve_use_chat_completions(eff_dialect)
        use_bfcl_chat = (
            eff_use_chat
            and getattr(self.env, "name", None) == "bfcl"
            and (callable(getattr(self.env, "run_chat", None)) or callable(getattr(self.env, "prepare_request_messages_and_tools", None)))
        )
        clients = [
            ChatClient(s.url, self.decoding, timeout=self.request_timeout, dialect=eff_dialect, cache_prompt=self.cache_prompt)
            for s in srvs
        ]
        out_path = self._episode_log(tag, save)

        def one(i_job):
            i, (task, seed) = i_job
            cl = clients[i % len(clients)]
            for attempt in range(self.retries + 1):
                try:
                    if use_bfcl_chat:
                        if callable(getattr(self.env, "run_chat", None)):
                            ep = self.env.run_chat(task, cl, seed)
                        else:
                            from .env.bfcl import run_chat_episode
                            ep = run_chat_episode(self.env, task, cl, seed)
                    else:
                        ep = self.env.run(task, cl, seed)
                except InfraError as e:
                    ep = Episode(task.id, seed, [], None, None, ["error"], 0, 0, 0.0, infra_error=str(e))
                if not ep.infra_error:
                    break
            v = self.env.verify(task, ep)
            row = {"task": task.id, "family": task.family, "split": task.split, "seed": seed, "side": side,
                   "success": v.success if not v.infra_error else None, "tests_passed": v.tests_passed,
                   "detectors": v.detectors, "infra_error": v.infra_error, "turns": ep.n_turns,
                   "tokens": ep.completion_tokens, "wall_s": round(ep.wall_s, 2)}
            if save:
                with self._lock:
                    rec = {**row, "messages": ep.messages, "final": ep.final_text, "verdict": v.details}
                    rc = getattr(ep, "reasoning_content", None)
                    if rc is not None:
                        rec["reasoning_content"] = rc
                    append_jsonl(out_path, rec)
            return row, ep

        n = self.slots * len(clients)
        with ThreadPoolExecutor(n) as ex:
            res = list(ex.map(one, enumerate(jobs)))
        self.last_episodes = [ep for _, ep in res]
        return [r for r, _ in res]

    def _episode_log(self, tag: str, save: bool) -> Path:
        out_path = self.log_dir / f"{tag}.jsonl"
        if save and tag not in self._seen_tags:  # a re-run phase replaces, never duplicates, its episode log
            out_path.unlink(missing_ok=True)
            self._seen_tags.add(tag)
        return out_path

    # ------------------------------------------------------------------ classifier environments

    def model_hash(self, model: str) -> str:
        """First 16 hex of the model file's sha256, memoised on (path, size, mtime)."""
        st = Path(model).stat()
        k = (str(Path(model).resolve()), st.st_size, st.st_mtime_ns)
        if k not in self._hashes:
            self._hashes[k] = sha256_file(model)[:16]
        return self._hashes[k]

    def _score(self, model: str, tasks: list[Task]) -> dict[str, float]:
        if self.scorer is None:
            raise ValueError("a classifier environment needs Evaluator(..., scorer=BatchScorer(...))")
        for attempt in range(self.retries + 1):
            try:
                got = self.env.score(tasks, str(model), self.scorer)
                break
            except Exception:
                if attempt == self.retries:
                    raise
        bad = [t.id for t in tasks if not _finite(got.get(t.id))]
        if bad:
            raise RuntimeError(f"env.score gave no finite score for {len(bad)} of {len(tasks)} tasks (first {bad[0]})")
        return {t.id: float(got[t.id]) for t in tasks}

    def rule(self, model: str) -> dict:
        """The decision rule for this model, from its calibration-split scores; cached in memory and on disk.

        The disk copy is reused only when its calibration fingerprint (env name, version, sorted task ids,
        fpr_target, pair/answer tokens, decoding, dialect, special) and size match the current evaluation;
        otherwise the rule is recomputed and the file replaced.
        """
        h = self.model_hash(model)
        cal = [t for t in self.env.tasks() if t.split == "calibration"]
        if not cal:
            raise ValueError("classifier environment has no calibration tasks")
        pair = getattr(self.env, "pair", None)
        u_tok = getattr(self.env, "unflagged_word", None)
        f_tok = getattr(self.env, "flagged_word", None)
        fp_obj = {
            "env_name": getattr(self.env, "name", None),
            "env_version": getattr(self.env, "version", None),
            "cal_ids": sorted(t.id for t in cal),
            "fpr_target": getattr(self.env, "fpr_target", None),
            "pair": list(pair) if pair is not None else None,
            "answer_tokens": [u_tok, f_tok] if (u_tok is not None or f_tok is not None) else None,
            "decoding": self.decoding,
            "dialect": self._resolve_dialect(model, None),
            "special": self._resolve_special(model),
        }
        fp = hashlib.sha256(json.dumps(fp_obj, sort_keys=True, default=str).encode()).hexdigest()
        rk = (h, fp)
        with self._cls_lock:
            if rk not in self._rules:
                path = self.log_dir / "rules" / f"{h}.json"
                old: dict = {}
                if path.exists():
                    try:
                        loaded = read_json(path)
                        if isinstance(loaded, dict):
                            old = loaded
                    except (json.JSONDecodeError, ValueError, OSError):
                        old = {}
                if old.get("cal_sha") == fp and old.get("n") == len(cal) and "rule" in old:
                    self._rules[rk] = old["rule"]
                else:
                    known = self._scores.setdefault(h, {})
                    todo = [t for t in cal if t.id not in known]   # seeded or earlier-scored calibration tasks are reused
                    if todo:
                        known.update(self._score(model, todo))
                    scores = {t.id: known[t.id] for t in cal}
                    rule = self.env.calibrate(scores)
                    write_json(path, {"model": str(model), "sha256_16": h, "n": len(cal), "cal_sha": fp, "rule": rule})
                    self._rules[rk] = rule
            return self._rules[rk]

    def scores(self, model: str, tasks: list[Task]) -> dict[str, float]:
        """Classifier scores of `tasks` under `model`, through the same per-model cache the episodes use."""
        h = self.model_hash(model)
        with self._cls_lock:
            known = self._scores.setdefault(h, {})
            todo = list({t.id: t for t in tasks if t.id not in known}.values())
            if todo:
                known.update(self._score(model, todo))
            return {t.id: known[t.id] for t in tasks}

    def seed(self, model: str, scores: dict[str, float]) -> None:
        """Preload recorded scores for `model` (same file, same tokenisation); the caller verifies a sample first."""
        bad = [k for k, v in scores.items() if not _finite(v)]
        if bad:
            raise ValueError(f"seed() received non-finite score for {len(bad)} tasks (first {bad[0]})")
        with self._cls_lock:
            self._scores.setdefault(self.model_hash(model), {}).update({k: float(v) for k, v in scores.items()})

    def _run_classifier(self, model: str, jobs: list[tuple[Task, int]], tag: str, side: str, save: bool) -> list[dict]:
        rule = self.rule(model)
        h = self.model_hash(model)
        with self._cls_lock:
            known = self._scores.setdefault(h, {})
            todo = list({t.id: t for t, _ in jobs if t.id not in known}.values())
            if todo:
                known.update(self._score(model, todo))
        out_path = self._episode_log(tag, save)
        rows, self.last_episodes = [], []
        for task, seed in jobs:
            s = known[task.id]
            v = self.env.decide(task, s, rule)
            row = {"task": task.id, "family": task.family, "split": task.split, "seed": seed, "side": side,
                   "success": v.success if not v.infra_error else None, "tests_passed": v.tests_passed,
                   "detectors": v.detectors, "infra_error": v.infra_error, "turns": 0, "tokens": 0, "wall_s": 0.0}
            if save:
                append_jsonl(out_path, {**row, "messages": [], "final": None, "verdict": v.details, "score": s,
                                        "rule": rule})
            rows.append(row)
            self.last_episodes.append(Episode(task.id, seed, [], None, None, [], 0, 0, 0.0))
        return rows


def _finite(x) -> bool:
    if isinstance(x, bool):
        return False
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def ok_rows(rows: list[dict]) -> list[dict]:
    """Drop infra-error rows (excluded, not failures)."""
    return [r for r in rows if r["success"] is not None]
