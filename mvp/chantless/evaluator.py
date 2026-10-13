"""Run episodes for one model across all GPUs (independent evaluations in parallel; results identical to serial).

Infra errors are retried on a fresh request and, if they persist, excluded from statistics and
logged; they never count as task failures (decision S12).
"""
from __future__ import annotations

import contextlib
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .client import ChatClient
from .env.protocol import Task
from .runtime import LlamaServer, Toolchain
from .util import append_jsonl


class Evaluator:
    def __init__(self, tc: Toolchain, env, decoding: dict, slots_per_gpu: int, ctx_per_slot: int, kv_type: str,
                 log_dir: str | Path, infra_retries: int = 2):
        self.tc, self.env, self.decoding = tc, env, decoding
        self.slots, self.ctx, self.kv = slots_per_gpu, ctx_per_slot, kv_type
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.retries = infra_retries
        self._lock = threading.Lock()
        self._seen_tags: set[str] = set()

    @contextlib.contextmanager
    def servers(self, model: str):
        with contextlib.ExitStack() as st:
            srvs = [st.enter_context(LlamaServer(self.tc, model, gpu=g, slots=self.slots, ctx_per_slot=self.ctx,
                                                 kv_type=self.kv, log=self.log_dir / f"server-gpu{g or 0}.log"))
                    for g in self.tc.gpus]
            yield srvs

    def run(self, model: str, jobs: list[tuple[Task, int]], tag: str, side: str = "cand", save: bool = True,
            srvs=None) -> list[dict]:
        """jobs: [(task, seed)] -> rows {task, family, split, seed, side, success, tests_passed, detectors, ...}."""
        if srvs is None:
            with self.servers(model) as s:
                return self.run(model, jobs, tag, side, save, s)
        clients = [ChatClient(s.url, self.decoding) for s in srvs]
        out_path = self.log_dir / f"{tag}.jsonl"
        if save and tag not in self._seen_tags:  # a re-run phase replaces, never duplicates, its episode log
            out_path.unlink(missing_ok=True)
            self._seen_tags.add(tag)

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


def ok_rows(rows: list[dict]) -> list[dict]:
    """Drop infra-error rows (excluded, not failures)."""
    return [r for r in rows if r["success"] is not None]
