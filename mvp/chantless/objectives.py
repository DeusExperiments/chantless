"""Objectives for log-odds proxy stores: candidate scores vs the frozen source's, as damage (higher is worse).

Each objective takes `cand` and `src` as `{"pos": array, "neg": array}`: one score per trajectory, where a
lower score means more toward the flagged class (pos).
"""
from __future__ import annotations

import math

import numpy as np


def pauc(x: dict, tail: float = 0.05, tau: float = 0.5) -> float:
    """Soft partial AUC: mean over pos p and hard negatives n of sigmoid((n - p) / tau).

    The hard negatives are the lowest ceil(tail * len(x["neg"])) of x's own negative scores. They must be
    chosen from the scored model's own values, never from a fixed index set (such as the source's lowest
    tail): a fixed lowest-tail set regresses toward the middle under any perturbation, which raises its
    apparent cutoff and rewards noisier candidates. For the same reason x["neg"] must be a random sample of
    negatives, not a tail-selected one.
    """
    pos, neg = np.asarray(x["pos"], dtype=np.float64), np.asarray(x["neg"], dtype=np.float64)
    if not len(pos) or not len(neg):
        raise ValueError(f"pauc needs both parts: {len(pos)} pos, {len(neg)} neg")
    k = math.ceil(round(tail * len(neg), 9))   # round first: 0.05 * 60 is 3.0000000000000004 in floating point
    hard = np.sort(neg)[: max(1, k)]
    gap = (hard[None, :] - pos[:, None]) / tau
    return float(np.mean(0.5 * (1.0 + np.tanh(0.5 * gap))))   # sigmoid, without exp overflow


def pauc_damage(cand: dict, src: dict, tail: float = 0.05, tau: float = 0.5) -> float:
    """pauc(src) - pauc(cand): higher is worse; negative when the candidate separates better than the source."""
    return pauc(src, tail, tau) - pauc(cand, tail, tau)


OBJECTIVES = {"pauc": pauc_damage}
