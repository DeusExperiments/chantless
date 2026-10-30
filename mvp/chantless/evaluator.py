"""Run episodes for one model across all GPUs (independent evaluations in parallel; results identical to serial).

Infra errors are retried on a fresh request and, if they persist, excluded from statistics and
logged; they never count as task failures (decision S12).

A classifier environment (env.classifier true) runs no episodes: each distinct task is scored once per
model by the batch scorer and judged at that model's own rule, computed from its calibration-split scores.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import math
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .client import ChatClient
from .env.protocol import Episode, Task
from .runtime import BatchScorer, LlamaServer, Toolchain
from .util import append_jsonl, read_json, sha256_file, write_json


class NoServer:
    """Placeholder for a llama-server on a classifier environment, which scores without one."""
    url = None

    def __init__(self, gpu: str):
        self.gpu = gpu


class Evaluator:
    def __init__(self, tc: Toolchain, env, decoding: dict, slots_per_gpu: int, ctx_per_slot: int, kv_type: str,
                 log_dir: str | Path, infra_retries: int = 2, scorer: BatchScorer | None = None):
        self.tc, self.env, self.decoding = tc, env, decoding
        self.slots, self.ctx, self.kv = slots_per_gpu, ctx_per_slot, kv_type
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.retries = infra_retries
        self._lock = threading.Lock()
        self._seen_tags: set[str] = set()
        self.scorer = scorer
        self.classifier = bool(getattr(env, "classifier", False))
        self._cls_lock = threading.Lock()
        self._hashes: dict[tuple, str] = {}               # (path, size, mtime_ns) -> model hash
        self._rules: dict[str, dict] = {}                 # model hash -> rule
        self._scores: dict[str, dict[str, float]] = {}    # model hash -> {task id: score}

    @contextlib.contextmanager
    def servers(self, model: str):
        if self.classifier:
            yield [NoServer(g) for g in self.tc.gpus]
            return
        with contextlib.ExitStack() as st:
            srvs = [st.enter_context(LlamaServer(self.tc, model, gpu=g, slots=self.slots, ctx_per_slot=self.ctx,
                                                 kv_type=self.kv, log=self.log_dir / f"server-gpu{g or 0}.log"))
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
        clients = [ChatClient(s.url, self.decoding) for s in srvs]
        out_path = self._episode_log(tag, save)

        def one(i_job):
            i, (task, seed) = i_job
            cl = clients[i % len(clients)]
            for attempt in range(self.retries + 1):
                ep = self.env.run(task, cl, seed)
                if not ep.infra_error:
                    break
            v = self.env.verify(task, ep)
            row = {"task": task.id, "family": task.family, "split": task.split, "seed": seed, "side": side,
                   "success": v.success if not v.infra_error else None, "tests_passed": v.tests_passed,
                   "detectors": v.detectors, "infra_error": v.infra_error, "turns": ep.n_turns,
                   "tokens": ep.completion_tokens, "wall_s": round(ep.wall_s, 2)}
            if save:
                with self._lock:
                    append_jsonl(out_path, {**row, "messages": ep.messages, "final": ep.final_text,
                                            "verdict": v.details})
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

        The disk copy is reused only when its calibration fingerprint (env name, version and sorted task ids)
        and size match the current environment; otherwise the rule is recomputed and the file replaced.
        """
        h = self.model_hash(model)
        with self._cls_lock:
            if h not in self._rules:
                cal = [t for t in self.env.tasks() if t.split == "calibration"]
                if not cal:
                    raise ValueError("classifier environment has no calibration tasks")
                fp = hashlib.sha256(json.dumps([self.env.name, self.env.version, sorted(t.id for t in cal)])
                                    .encode()).hexdigest()
                path = self.log_dir / "rules" / f"{h}.json"
                old = read_json(path) if path.exists() else {}
                if old.get("cal_sha") == fp and old.get("n") == len(cal):
                    self._rules[h] = old["rule"]
                else:
                    scores = self._score(model, cal)
                    self._scores.setdefault(h, {}).update(scores)  # a calibration task run as a job is not rescored
                    rule = self.env.calibrate(scores)
                    write_json(path, {"model": str(model), "sha256_16": h, "n": len(cal), "cal_sha": fp, "rule": rule})
                    self._rules[h] = rule
            return self._rules[h]

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
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def ok_rows(rows: list[dict]) -> list[dict]:
    """Drop infra-error rows (excluded, not failures)."""
    return [r for r in rows if r["success"] is not None]
