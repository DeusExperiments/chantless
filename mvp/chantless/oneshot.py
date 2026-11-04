"""One-shot mask search (`learner.algo: "oneshot"`): the search reuses the sanity phase's solo probes and builds once.

The sanity phase already builds every single-group setting (one group changed from the BF16 mask) and scores it on
every proxy store, saving per-trajectory scores. This module turns those into a surrogate:

    scores(mask) = source scores + sum over groups of that group's solo score shift

and minimises the pipeline's own task proxy (each logodds store's objective, weighted by its trajectory count) on
the predicted scores under a byte budget. The search is greedy descent by damage per byte with a full re-rank at
every step, then single and pair exchanges at the budget. Every candidate costs microseconds, because no file is
built until the end. Then one real build checks the bytes. If the file is over budget, the solve runs again with the
overage taken off the budget, at most three times.

On the Qwen3.5-0.8B classifier run, this surrogate predicted the measured proxy damage of all 576 greedy-search masks
with Spearman 0.97. Its own 396 MB mask was predicted at -0.01244 and measured at -0.01283.

Bytes are exact per tensor (block sizes of each ggml type, each tensor padded to the file's alignment). A constant
header part is calibrated on the real uniform builds of the sanity phase.
"""
from __future__ import annotations

import math

import numpy as np

from . import gguf_ops, mask as mk, objectives

BLOCK = {"F32": (1, 4), "BF16": (1, 2), "F16": (1, 2), "Q8_0": (32, 34), "Q6_K": (256, 210), "Q5_K": (256, 176),
         "Q4_K": (256, 144), "Q3_K": (256, 110), "Q2_K": (256, 84), "IQ2_XS": (256, 74), "IQ1_S": (256, 50)}


def _pad(n: int, a: int = 32) -> int:
    return (n + a - 1) // a * a


def tensor_bytes(dims, ty: str) -> int:
    bs, ts = BLOCK[ty.upper()]
    if dims[0] % bs:
        raise ValueError(f"row length {dims[0]} is not a multiple of {ty} block {bs}")
    return _pad(ts * (dims[0] // bs) * int(np.prod(dims[1:], dtype=np.int64)))


class ByteModel:
    """bytes(mask) = header + sum of tensor bytes; header calibrated on one real build of a known mask."""
    def __init__(self, source_gguf: str, groups: dict):
        self.inv = {t["name"]: t for t in gguf_ops.inventory(source_gguf)}
        self.groups, self.header = groups, 0

    def tensors(self, m: dict) -> int:
        types, drop = mk.tensor_plan(m, self.groups)
        tot = 0
        for n, t in self.inv.items():
            if n in drop:
                continue
            tot += tensor_bytes(t["dims"], types.get(n, t["type"]))
        return tot

    def calibrate(self, m: dict, real_bytes: int) -> None:
        self.header = real_bytes - self.tensors(m)

    def __call__(self, m: dict) -> int:
        return self.header + self.tensors(m)


class Surrogate:
    """stores: list of (name, weight, objective fn(cand_scores) -> damage, source scores array)."""
    def __init__(self, stores, solo: dict):
        self.stores = stores
        self.src = np.concatenate([s[3] for s in stores])
        self.cuts = np.cumsum([0] + [len(s[3]) for s in stores])
        self.solo = solo                      # (group, tier) -> score shift over all stores, concatenated
        self.zero = np.zeros(len(self.src))

    def shift(self, g: str, t: str) -> np.ndarray:
        return self.solo.get((g, t), self.zero) if t not in ("BF16",) else self.zero

    def damage(self, x: np.ndarray) -> float:
        num = den = 0.0
        for (name, w, fn, src), a, b in zip(self.stores, self.cuts, self.cuts[1:]):
            num += w * len(src) * fn(x[a:b]); den += w * len(src)
        return num / den


def solve(sur: Surrogate, options: dict, cost: dict, budget: int, header: int, pair_top: int = 60, log=print) -> dict:
    """options: group -> list of tiers; cost: (group, tier) -> bytes of that group's tensors. Returns {group: tier}."""
    rich = {g: max(o, key=lambda t: cost[(g, t)]) for g, o in options.items()}
    m = dict(rich)
    B = lambda m: header + sum(cost[(g, t)] for g, t in m.items())
    x = sur.src + sum(sur.shift(g, t) for g, t in m.items()); cur = sur.damage(x)
    while B(m) > budget:
        best = None
        for g, o in options.items():
            for t in o:
                db = cost[(g, m[g])] - cost[(g, t)]
                if db <= 0:
                    continue
                y = x - sur.shift(g, m[g]) + sur.shift(g, t); d = sur.damage(y) - cur
                sc = (d + 1e-12) / db if d >= 0 else d * db
                if best is None or sc < best[0]:
                    best = (sc, g, t, y, d)
        if best is None:
            raise RuntimeError(f"budget {budget} is below the cheapest mask {B(m)}")
        _, g, t, x, d = best; m[g] = t; cur += d
    log(f"oneshot: descent reached {B(m)} bytes, predicted damage {cur:.6f}")
    for _ in range(100):
        b = B(m); singles = []
        for g, o in options.items():
            for t in o:
                if t != m[g]:
                    singles.append((sur.damage(x - sur.shift(g, m[g]) + sur.shift(g, t)) - cur, cost[(g, t)] - cost[(g, m[g])], g, t))
        best = (-1e-7, None)
        for d, db, g, t in singles:
            if b + db <= budget and d < best[0]:
                best = (d, [(g, t)])
        ups = sorted([s for s in singles if s[1] > 0], key=lambda s: s[0])[:pair_top]
        downs = sorted([s for s in singles if s[1] < 0], key=lambda s: s[0] / -s[1])[:pair_top]
        for d1, db1, g1, t1 in ups:
            y1 = x - sur.shift(g1, m[g1]) + sur.shift(g1, t1)
            for d2, db2, g2, t2 in downs:
                if g1 != g2 and b + db1 + db2 <= budget:
                    d = sur.damage(y1 - sur.shift(g2, m[g2]) + sur.shift(g2, t2)) - cur
                    if d < best[0]:
                        best = (d, [(g1, t1), (g2, t2)])
        if best[1] is None:
            break
        for g, t in best[1]:
            x = x - sur.shift(g, m[g]) + sur.shift(g, t); m[g] = t
        cur = sur.damage(x)
    log(f"oneshot: exchange settled at {B(m)} bytes, predicted damage {cur:.6f}")
    return m


def pauc_store(src_split: dict, parts: list[dict], args: dict):
    """Damage fn on one logodds store's concatenated scores, in its parts order (the prober's own objective)."""
    roles = np.concatenate([np.full(p["n"], p["role"] == "pos") for p in parts])
    def fn(x):
        return objectives.pauc_damage({"pos": x[roles], "neg": x[~roles]}, src_split, **args)
    return fn

def _rank(a: np.ndarray) -> np.ndarray:
    r = np.argsort(np.argsort(a, kind="stable"), kind="stable").astype(np.float64)
    return (r - r.mean()) / r.std()


class FidelitySurrogate(Surrogate):
    """Damage = 1 - Spearman(predicted scores, source scores) over every store's trajectories.

    The gate recalibrates each candidate's own cutoff and counts decisions that differ from the source, so what it
    rewards is keeping the source's ordering. On the Qwen3.5-0.8B classifier, store fidelity ranked seven gated masks
    against their losses at Spearman 0.93, against 0.61 for the pAUC proxy, which ranked the hand-built 200 MB mask last.
    """
    def __init__(self, stores, solo: dict):
        super().__init__(stores, solo)
        self.rs = _rank(self.src)

    def damage(self, x: np.ndarray) -> float:
        return 1.0 - float((_rank(x) * self.rs).mean())
