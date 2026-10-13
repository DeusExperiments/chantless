"""Materialize a mask into a standalone GGUF and account its actual bytes (spec §9, decision S1).

Order: physically remove dropped groups from the (source or repaired) weights, then quantize every
remaining tensor with explicit per-tensor types. The built file is then audited against the mask.
Objective bytes = the built file's size on disk. Zero-filled tensors earn nothing; only absent
tensors reduce bytes.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path

from . import gguf_ops, mask as maskmod
from .runtime import Toolchain, quantize


@dataclass
class BuildResult:
    path: str
    file_bytes: int
    tensor_bytes: int
    source_params: int
    bpw: float                     # file_bytes * 8 / source parameter count (size-matched cap uses this)
    types: dict[str, str]
    dropped: list[str]
    fallbacks: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("types")
        return d


def mask_key(mask: dict) -> str:
    core = {"groups": mask["groups"], "tensors": mask.get("tensors", {}), "weights": mask.get("weights", {})}
    return hashlib.sha256(json.dumps(core, sort_keys=True).encode()).hexdigest()[:20]


class Builder:
    def __init__(self, tc: Toolchain, weights_gguf: str, groups: dict, imatrix: str | None, work: str | os.PathLike,
                 log: str | None = None, threads: int = 8):
        self.tc, self.weights, self.groups, self.imatrix = tc, str(weights_gguf), groups, imatrix
        self.work = Path(work)
        self.work.mkdir(parents=True, exist_ok=True)
        self.log = log
        self.threads = threads
        self.max_pruned = 24
        self._prune_lock = threading.Lock()
        inv = gguf_ops.inventory(self.weights)
        self.source_params = sum(t["n_elements"] for t in inv)
        self.source_types = {t["name"]: t["type"] for t in inv}

    def _pruned(self, drop: set[str]) -> str:
        if not drop:
            return self.weights
        with self._prune_lock:
            return self._pruned_locked(drop)

    def _pruned_locked(self, drop: set[str]) -> str:
        h = hashlib.sha256((self.weights + "|" + "|".join(sorted(drop))).encode()).hexdigest()[:16]
        p = self.work / f"pruned-{h}.gguf"
        if not p.exists():
            gguf_ops.copy_without(self.weights, p, drop)
            old = sorted(self.work.glob("pruned-*.gguf"), key=lambda q: q.stat().st_mtime)
            for q in old[: max(0, len(old) - self.max_pruned)]:  # bounded disk use; rebuilt on demand
                if q != p:
                    q.unlink(missing_ok=True)
        os.utime(p)
        return str(p)

    def build(self, m: dict, out: str | os.PathLike | None = None, keep_pruned: bool = False) -> BuildResult:
        maskmod.validate(m, self.groups)
        types, drop = maskmod.tensor_plan(m, self.groups)
        out = Path(out) if out else self.work / f"cand-{mask_key(m)}.gguf"
        src = self._pruned(drop)
        if all(t == "BF16" for t in types.values()) and all(self.source_types[n] == "BF16" for n in types):
            Path(out).unlink(missing_ok=True)
            try:
                os.link(src, out)        # no extra disk for pure-BF16 (drop-only) candidates
            except OSError:
                shutil.copyfile(src, out)
        else:
            tok = types.pop("token_embd.weight", None)
            quantize(self.tc, src, str(out), "Q8_0", {n: t for n, t in types.items()}, tok,
                     self.imatrix, self.log, self.threads)
            if tok:
                types["token_embd.weight"] = tok
        if src != self.weights and not keep_pruned:
            pass  # pruned sources are cached by drop-set; removed by cleanup()
        inv = {t["name"]: t for t in gguf_ops.inventory(out)}
        fallbacks = {}
        for n, t in types.items():
            if n not in inv:
                raise RuntimeError(f"built file lacks expected tensor {n}")
            if inv[n]["type"] != t.upper():
                fallbacks[n] = inv[n]["type"]
        present_dropped = sorted(n for n in drop if n in inv)
        if present_dropped:
            raise RuntimeError(f"dropped tensors still present: {present_dropped[:3]}")
        fb = gguf_ops.file_bytes(out)
        return BuildResult(str(out), fb, sum(t["n_bytes"] for t in inv.values()), self.source_params,
                           fb * 8 / self.source_params, types, sorted(drop), fallbacks)

    def cleanup(self) -> None:
        for p in self.work.glob("pruned-*.gguf"):
            p.unlink(missing_ok=True)


def kv_bytes(info: dict, n_attn_kept: int, context: int, concurrency: int, kv_type: str) -> int:
    per = {"f16": 2.0, "q8_0": 34 / 32, "q4_0": 18 / 32}[kv_type]
    return int(n_attn_kept * 2 * info["n_head_kv"] * info["head_dim"] * per * context * concurrency)
