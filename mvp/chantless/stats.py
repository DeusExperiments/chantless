"""Paired, task-matched non-inferiority acceptance statistics.

`delta` = mean over tasks of (candidate rate - source rate). Uncertainty uses family-clustered CR1
standard errors with Student-t(F-1) bounds. Sequential stopping uses K equally spaced looks with
Bonferroni alpha spending (alpha/K per look): stop for success when L > -margin, or for futility when
U < -margin.
"""
from __future__ import annotations

import math
import random
from collections import defaultdict
from statistics import NormalDist

Z = NormalDist().inv_cdf


def sample_size(v: float, margin: float, alpha: float, power: float, deff: float = 1.0, looks: int = 1,
                df: int | None = None) -> int:
    """Required sample size: n = ((q_{1-a} + q_{power}) * sqrt(v) / margin)^2 * deff, with alpha split over
    looks. With df (families - 1) the quantiles are Student-t, matching the cluster-robust gate."""
    a = alpha / max(1, looks)
    if df:
        from scipy.stats import t as tdist
        q = float(tdist.ppf(1 - a, df)) + float(tdist.ppf(power, df))
    else:
        q = Z(1 - a) + Z(power)
    return int(math.ceil((q * math.sqrt(max(v, 1e-9)) / margin) ** 2 * deff))


def _task_rates(rows: list[dict]) -> dict[str, dict]:
    """rows: {'task','family','side' in {cand,src}, 'success': bool}. -> per task rates and family."""
    acc: dict[str, dict] = {}
    for r in rows:
        a = acc.setdefault(r["task"], {"family": r["family"], "cand": [], "src": []})
        a[r["side"]].append(1.0 if r["success"] else 0.0)
    return {t: a for t, a in acc.items() if a["cand"] and a["src"]}


def delta_and_bounds(rows: list[dict], alpha: float, n_boot: int = 0, seed: int = 0) -> dict:
    """delta = mean over tasks of (cand rate - src rate); one-sided (1-alpha) bounds from a cluster-robust
    (family-level) standard error with a t critical value on F-1 degrees of freedom.

    Each task's delta already contains its episode noise, and families are the independent units, so the
    between-family spread of family sums is the right variance (CR1 sandwich). `n_boot` and `seed` are
    accepted for API compatibility and ignored.
    """
    from scipy.stats import t as tdist
    tr = _task_rates(rows)
    nan = float("nan")
    if not tr:
        return {"delta": nan, "L": nan, "U": nan, "n_tasks": 0, "n_families": 0}
    d = {t: sum(a["cand"]) / len(a["cand"]) - sum(a["src"]) / len(a["src"]) for t, a in tr.items()}
    fam = defaultdict(list)
    for t, a in tr.items():
        fam[a["family"]].append(d[t])
    N, F = len(d), len(fam)
    delta = sum(d.values()) / N
    out = {"delta": delta, "n_tasks": N, "n_families": F, "n_cand": sum(len(a["cand"]) for a in tr.values()),
           "n_src": sum(len(a["src"]) for a in tr.values())}
    if F < 2:
        out.update(L=nan, U=nan, se=nan)
        return out
    ss = sum((sum(v) - len(v) * delta) ** 2 for v in fam.values())
    se = math.sqrt(F / (F - 1) * ss) / N
    crit = float(tdist.ppf(1 - alpha, F - 1))
    out.update(L=delta - crit * se, U=delta + crit * se, se=se, t_crit=crit, df=F - 1)
    return out


def empty_ok(b: dict, margin: float) -> bool:
    """Non-inferiority for bounds that may be empty: True when nothing was measured (n_tasks == 0),
    otherwise L > -margin."""
    return b["n_tasks"] == 0 or b["L"] > -margin


def discordance(rows: list[dict]) -> dict:
    """v = variance of a single paired difference; deff = (CR1 variance of the task-mean delta) / (iid variance)."""
    tr = _task_rates(rows)
    diffs = []
    for a in tr.values():
        for i in range(min(len(a["cand"]), len(a["src"]))):
            diffs.append(a["cand"][i] - a["src"][i])
    if len(diffs) < 2 or len(tr) < 3:
        return {"v": float("nan"), "deff": float("nan"), "n_pairs": len(diffs)}
    mean = sum(diffs) / len(diffs)
    v = sum((x - mean) ** 2 for x in diffs) / (len(diffs) - 1)
    d = [sum(a["cand"]) / len(a["cand"]) - sum(a["src"]) / len(a["src"]) for a in tr.values()]
    N = len(d)
    md = sum(d) / N
    iid = sum((x - md) ** 2 for x in d) / (N - 1) / N
    b = delta_and_bounds(rows, 0.05)
    deff = (b["se"] ** 2 / iid) if iid > 0 and b.get("se") == b.get("se") else float("nan")
    return {"v": v, "deff": deff, "n_pairs": len(diffs), "n_tasks": N, "n_families": b["n_families"], "mean": mean}


def sequential_decision(rows: list[dict], look: int, looks: int, margin: float, alpha: float, final: bool) -> dict:
    a = alpha / looks
    b = delta_and_bounds(rows, a)
    if b["n_tasks"] == 0:
        b.update(decision="continue")
        return b
    if b["L"] > -margin:
        b["decision"] = "pass"
    elif b["U"] < -margin:
        b["decision"] = "fail"
    else:
        b["decision"] = "fail" if final else "continue"
    b.update(look=look, looks=looks, alpha_look=a, margin=margin)
    return b


def detector_bounds(rows: list[dict], alpha: float, bound: float) -> dict:
    """Check that the upper (1-alpha) bound on (cand - src) rate per detector is <= `bound`."""
    out = {}
    names = sorted({k for r in rows for k in r.get("detectors", {})})
    for k in names:
        sub = [dict(r, success=bool(r["detectors"].get(k))) for r in rows]
        b = delta_and_bounds(sub, alpha, n_boot=2000)
        out[k] = {"delta": b["delta"], "U": b["U"], "ok": not (b["U"] > bound)}
    return out
