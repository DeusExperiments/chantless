"""Probe = build a candidate from a mask, then score it against frozen-source reference stores.

Stores (computed once from the frozen source; never moved with candidate weights):
  task     KL on source-successful search-split trajectories (preserve proxy)
  general  KL on general text + code (size-matched sanity cap)
  fb_k     KL on candidate failure trajectories added by learner feedback (decision S4)
In improve mode the task proxy is the candidate's NLL on verified-successful trajectories instead of KL.
A store of kind "logodds" (classifier environments) instead scores the candidate's log-odds at each
trajectory's answer and records an objectives.OBJECTIVES damage against the source's, as both kl and nll.
"""
from __future__ import annotations

import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import kl, objectives
from .builder import Builder, mask_key
from .runtime import Toolchain


@dataclass
class Store:
    name: str
    trj: str
    dir: str
    n_targets: int
    weight: float = 1.0
    kind: str = "kl"        # "logodds": trj is meta["parts"] concatenated, dir holds source.npy
    meta: dict = field(default_factory=dict)


@dataclass
class ProbeResult:
    key: str
    bytes: int
    bpw: float
    kl: dict = field(default_factory=dict)     # store -> kl
    nll: dict = field(default_factory=dict)    # store -> candidate nll
    fallbacks: dict = field(default_factory=dict)

    def task_proxy(self, stores: list[Store], mode: str) -> float:
        num = den = 0.0
        for s in stores:
            if s.name == "general":
                continue
            val = self.kl[s.name] if mode == "preserve" else self.nll[s.name]
            num += val * s.n_targets * s.weight
            den += s.n_targets * s.weight
        return num / den


class Prober:
    def __init__(self, tc: Toolchain, builder: Builder, work: str | Path, log: str | None, workers: int | None = None,
                 scratch: str | Path | None = None, scorer=None):
        self.tc, self.builder, self.scorer = tc, builder, scorer   # scorer: runtime.BatchScorer, for logodds stores
        self.work = Path(work)                       # receipts (persist; resume reads them)
        self.work.mkdir(parents=True, exist_ok=True)
        self.scratch = Path(scratch) if scratch else self.work   # built candidates (deleted after scoring)
        self.scratch.mkdir(parents=True, exist_ok=True)
        self.log = log
        self.stores: list[Store] = []
        self._gpu_lock = threading.Lock()
        self._free = list(tc.gpus)
        self._cv = threading.Condition()
        self.cache: dict[str, ProbeResult] = {}
        self.workers = workers or max(1, len(tc.gpus))   # > n_gpus overlaps CPU builds with GPU scoring
        self._src: dict[str, dict] = {}                   # logodds store -> source parts

    def add_store(self, s: Store) -> None:
        if s.kind not in ("kl", "logodds"):
            raise ValueError(f"store {s.name}: unknown kind {s.kind!r}")
        if s.kind == "logodds" and self.scorer is None:
            raise ValueError(f"store {s.name}: a logodds store needs Prober(scorer=...)")
        self.stores.append(s)

    def _gpu(self):
        with self._cv:
            while not self._free:
                self._cv.wait()
            return self._free.pop()

    def _release(self, g):
        with self._cv:
            self._free.append(g)
            self._cv.notify()

    def probe(self, mask: dict, stores: list[str] | None = None, keep: bool = False, gate=None) -> ProbeResult:
        """`gate(res)` is checked after the general store; when it returns False the other stores are skipped."""
        key = mask_key(mask)
        want = stores or [s.name for s in self.stores]
        cached = self.cache.get(key) or self._from_disk(key)
        if cached and (all(n in cached.kl for n in want) or (gate and "general" in cached.kl and not gate(cached))):
            self.cache[key] = cached
            return cached
        b = self.builder.build(mask, self.scratch / f"p-{key}.gguf")
        res = cached or ProbeResult(key, b.file_bytes, b.bpw, fallbacks=b.fallbacks)
        g = self._gpu()
        try:
            for s in sorted(self.stores, key=lambda s: s.name != "general"):
                if s.name not in want or s.name in res.kl:
                    continue
                if gate and s.name != "general" and "general" in res.kl and not gate(res):
                    break
                if s.kind == "logodds":
                    res.kl[s.name] = res.nll[s.name] = self._logodds(b.path, s, key, g)
                    continue
                r = kl.traj_kl(self.tc, "kl", b.path, s.trj, s.dir, str(self.work / f"p-{key}-{s.name}.json"), self.log, gpu=g)
                res.kl[s.name], res.nll[s.name] = float(r["kl"]), float(r["nll_cand"])
        finally:
            self._release(g)
            if not keep:
                Path(b.path).unlink(missing_ok=True)
        self.cache[key] = res
        (self.work / f"p-{key}-meta.json").write_text(json.dumps({"bytes": res.bytes, "bpw": res.bpw, "fallbacks": res.fallbacks}))
        return res

    @staticmethod
    def _split(s: Store, x: np.ndarray) -> dict:
        """Per-trajectory scores in meta["parts"] order -> {"pos": array, "neg": array}."""
        parts = s.meta["parts"]
        if len(x) != sum(p["n"] for p in parts):
            raise RuntimeError(f"store {s.name}: {len(x)} scores for {sum(p['n'] for p in parts)} trajectories")
        out, i = {"pos": [], "neg": []}, 0
        for p in parts:
            out[p["role"]].append(x[i: i + p["n"]])
            i += p["n"]
        return {k: np.concatenate(v) if v else np.zeros(0) for k, v in out.items()}

    def _logodds(self, model: str, s: Store, key: str, gpu: str) -> float:
        """Objective damage of the candidate on a logodds store. The receipt keeps "kl" and "nll_cand" for _from_disk."""
        if s.name not in self._src:
            self._src[s.name] = self._split(s, np.load(Path(s.dir) / "source.npy"))
        x = np.asarray(self.scorer.score_trj(model, s.trj, tuple(s.meta["pair"]), gpu=gpu), dtype=np.float64)
        cand = self._split(s, x)
        v = float(objectives.OBJECTIVES[s.meta["objective"]](cand, self._src[s.name], **s.meta.get("objective_args", {})))
        np.save(self.work / f"p-{key}-{s.name}.npy", x)
        (self.work / f"p-{key}-{s.name}.json").write_text(json.dumps(
            {"kl": v, "nll_cand": v, "kind": "logodds", "objective": s.meta["objective"],
             "n_pos": int(len(cand["pos"])), "n_neg": int(len(cand["neg"]))}))
        return v

    def _from_disk(self, key: str):
        """Resume support: reload receipts of probes already measured in this run/tag (same weights, same stores)."""
        meta = self.work / f"p-{key}-meta.json"
        if not meta.exists():
            return None
        m = json.loads(meta.read_text())
        res = ProbeResult(key, m["bytes"], m["bpw"], fallbacks=m.get("fallbacks", {}))
        for s in self.stores:
            f = self.work / f"p-{key}-{s.name}.json"
            if f.exists():
                r = json.loads(f.read_text())
                res.kl[s.name], res.nll[s.name] = float(r["kl"]), float(r["nll_cand"])
        return res

    def probe_many(self, masks: list[dict], stores: list[str] | None = None, gate=None) -> list[ProbeResult]:
        with ThreadPoolExecutor(self.workers) as ex:
            return list(ex.map(lambda m: self.probe(m, stores, gate=gate), masks))

    def invalidate(self) -> None:
        """Called when weights change (repair) or stores change; cached KL values become stale."""
        self.cache.clear()
        self._src.clear()
