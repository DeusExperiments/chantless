"""External comparison of classifiers, each calibrated at its own 1% false-positive operating point.

A model's cutoff is the score at index int(0.01 * n) of its sorted negative-class calibration scores, and a
document is flagged when its score is strictly below it. Only documents every model scored are counted.
Intervals are the 2.5th to 97.5th percentile over bootstrap resamples of documents, the same resamples for
every model, so two models' intervals are directly comparable.
"""
from __future__ import annotations

import math
import numbers

import numpy as np


def cutoff(cal, rate: float = 0.01) -> float:
    """Flag strictly below this: the lowest `rate` of the calibration negatives."""
    if isinstance(rate, bool) or not isinstance(rate, (int, float, np.number)):
        raise ValueError(f"invalid rate {rate!r}")
    frate = float(rate)
    if not math.isfinite(frate) or not (0.0 <= frate <= 1.0):
        raise ValueError(f"rate must be finite in [0, 1], got {rate!r}")
    if (isinstance(cal, np.ndarray) and cal.dtype == bool) or (
        not isinstance(cal, np.ndarray) and any(isinstance(x, bool) for x in cal)
    ):
        raise ValueError("calibration scores must be numeric, not bool")
    cal = np.asarray(cal, dtype=float)
    if cal.ndim != 1 or not len(cal):
        raise ValueError("empty calibration set")
    if not np.isfinite(cal).all():
        raise ValueError(f"{int((~np.isfinite(cal)).sum())} non-finite calibration scores")
    cal = np.sort(cal)
    idx = min(len(cal) - 1, max(0, int(frate * len(cal))))
    return float(cal[idx])


def _rate(flag, sel, boot):
    if not sel.any():   # nothing to measure: NaN, never a measured zero
        return {"caught": float("nan"), "ci": [float("nan"), float("nan")], "n": 0}
    point = float(flag[sel].mean())
    fb, sb = flag[boot], sel[boot]
    denom = sb.sum(1)
    valid = denom > 0
    if not valid.any():
        return {"caught": point, "ci": [point, point], "n": int(sel.sum())}
    dist = (fb & sb)[valid].sum(1) / denom[valid]
    return {"caught": point, "ci": [float(np.percentile(dist, 2.5)), float(np.percentile(dist, 97.5))],
            "n": int(sel.sum())}


def compare(models: dict, labels: dict, kinds: dict, groups_of=None, boot: int = 1000, seed=0) -> dict:
    """Per model: negatives flagged, positives caught overall and per kind, with paired bootstrap intervals.

    models = {name: {"cal": negative-class calibration scores, "scores": {doc_id: score}}}; a lower score
    means more toward the flagged ("pos") class. labels = {doc_id: "pos" | "neg"}, kinds = {doc_id: str}.
    groups_of: the kinds reported separately (each restricted to positives); None reports every positive
    kind when there are at least two. `seed` may be a numpy Generator, advanced in place so successive calls
    share a deterministic RNG stream.
    A label other than "pos"/"neg", or a non-finite score, is an error. A selection with no documents reports
    NaN with a NaN interval.
    """
    names = list(models)
    if not names:
        raise ValueError("no models")
    if isinstance(boot, bool) or not isinstance(boot, numbers.Integral) or int(boot) < 1:
        raise ValueError(f"boot must be an integer >= 1, got {boot!r}")
    boot = int(boot)
    rng = np.random.default_rng(seed)
    cuts = {}
    for m in names:
        try:
            cuts[m] = cutoff(models[m]["cal"])
        except ValueError as e:
            raise ValueError(f"model {m}: {e}") from None
    ids = sorted(set(labels).intersection(*(set(models[m]["scores"]) for m in names)))
    if not ids:
        raise ValueError("no document was scored by every model")
    bad = next((i for i in ids if labels[i] not in ("pos", "neg")), None)
    if bad is not None:
        raise ValueError(f"label of {bad!r} is {labels[bad]!r}; labels must be 'pos' or 'neg'")
    pos = np.array([labels[i] == "pos" for i in ids])
    kind = np.array([kinds.get(i, "") for i in ids])
    if groups_of is None:
        ks = sorted(str(k) for k in set(kind[pos]))
        groups_of = ks if len(ks) > 1 else []
    sel_k = {k: pos & (kind == k) for k in groups_of}
    idx = rng.integers(0, len(ids), (boot, len(ids)))
    out = {"docs": len(ids), "n_pos": int(pos.sum()), "n_neg": int((~pos).sum()),
           "unmatched": {m: len(models[m]["scores"]) - len(ids) for m in names}, "cutoffs": cuts, "bootstrap": boot,
           "models": {}}
    for m in names:
        x = np.array([models[m]["scores"][i] for i in ids], dtype=float)
        if not np.isfinite(x).all():
            j = int(np.flatnonzero(~np.isfinite(x))[0])
            raise ValueError(f"model {m}: score of {ids[j]!r} is {x[j]}")
        flag = x < cuts[m]
        neg = _rate(flag, ~pos, idx)
        out["models"][m] = {"neg_flagged": neg["caught"], "neg_ci": neg["ci"], "pos": _rate(flag, pos, idx),
                            "kinds": {k: _rate(flag, s, idx) for k, s in sel_k.items()}}
    return out


def beats(res: dict, a: str, b: str, kind: str | None = None) -> bool:
    """True when a's interval for positives caught (overall, or for one kind) lies wholly above b's."""
    get = (lambda m: res["models"][m]["pos"]) if kind is None else (lambda m: res["models"][m]["kinds"][kind])
    return get(a)["ci"][0] > get(b)["ci"][1]
