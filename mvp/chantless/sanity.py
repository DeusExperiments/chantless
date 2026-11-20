"""General-data sanity: text corpus, executable behavioral checks, size-matched cap, solo-application floors.

Computes per-group minimum precisions from solo tests, enforces a size-matched general-KL cap, and runs
executable behavioral checks on screens and gates.
"""
from __future__ import annotations

import bisect
import glob
import json
import os
import random
import sysconfig
from pathlib import Path

from .client import InfraError
from .util import stable_seed


# ---------------------------------------------------------------- general text + code corpus

def general_texts(n_docs: int, seed: int, cache: str | Path) -> list[str]:
    """FineWeb-Edu (streamed, pinned config) + Python stdlib source (general code, PSF licence, offline)."""
    cache = Path(cache)
    if cache.exists():
        return json.loads(cache.read_text())
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
    web = []
    for row in ds:
        t = row["text"]
        if 800 < len(t) < 6000:
            web.append(t)
        if len(web) >= n_docs:
            break
    lib = sorted(glob.glob(os.path.join(sysconfig.get_paths()["stdlib"], "*.py")))
    rng = random.Random(seed)
    rng.shuffle(lib)
    code = []
    for p in lib:
        s = Path(p).read_text(errors="ignore")
        if 2000 < len(s) < 40000:
            code.append(s[:6000])
        if len(code) >= n_docs // 2:
            break
    mixed = []
    for i in range(max(len(web), len(code))):
        if i < len(web):
            mixed.append(web[i])
        if i < len(code):
            mixed.append(code[i])
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(mixed))
    return mixed


# ---------------------------------------------------------------- behavioral checks

CAPITALS = [("France", "Paris"), ("Japan", "Tokyo"), ("Italy", "Rome"), ("Germany", "Berlin"), ("Spain", "Madrid"),
            ("Canada", "Ottawa"), ("Egypt", "Cairo"), ("Kenya", "Nairobi"), ("Peru", "Lima"), ("Norway", "Oslo")]


def behavioral_items(n: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    items = []
    for i in range(n):
        k = i % 6
        if k == 0:
            a, b = rng.randint(10, 99), rng.randint(10, 99)
            items.append({"id": f"arith-{i}", "prompt": f"What is {a} + {b}? Reply with only the number.", "check": "exact", "answer": str(a + b)})
        elif k == 1:
            c, cap = rng.choice(CAPITALS)
            items.append({"id": f"cap-{i}", "prompt": f"What is the capital of {c}? Reply with one word.", "check": "contains", "answer": cap})
        elif k == 2:
            w = rng.choice(["apple", "river", "planet", "window", "guitar", "silver"])
            items.append({"id": f"upper-{i}", "prompt": f"Write the word {w} in all capital letters and nothing else.", "check": "exact", "answer": w.upper()})
        elif k == 3:
            name, age = rng.choice(["Ana", "Bo", "Cy", "Dee"]), rng.randint(5, 90)
            items.append({"id": f"json-{i}", "prompt": f'Return a JSON object with keys "name" set to "{name}" and "age" set to {age}. Output only the JSON.',
                          "check": "json", "answer": {"name": name, "age": age}})
        elif k == 4:
            xs = [rng.randint(1, 9) for _ in range(rng.randint(3, 6))]
            items.append({"id": f"count-{i}", "prompt": f"How many numbers are in this list: {xs}? Reply with only the number.", "check": "exact", "answer": str(len(xs))})
        else:
            w = rng.choice(["stone", "light", "drama", "paper", "metal"])
            items.append({"id": f"rev-{i}", "prompt": f"Reverse the letters of the word '{w}'. Reply with only the reversed word.", "check": "exact_lower", "answer": w[::-1]})
    return items


def _check(item: dict, text: str) -> bool:
    t = (text or "").strip().strip("`").strip()
    if item["check"] == "exact":
        return t.rstrip(".") == item["answer"]
    if item["check"] == "exact_lower":
        return t.rstrip(".").lower() == item["answer"]
    if item["check"] == "contains":
        return item["answer"].lower() in t.lower() and len(t.split()) <= 4
    if item["check"] == "json":
        s = t[t.find("{") : t.rfind("}") + 1] if "{" in t else t
        try:
            return json.loads(s) == item["answer"]
        except Exception:
            return False
    raise ValueError(item["check"])


def run_behavioral(srvs, decoding: dict, items: list[dict], seeds: list[int], side: str) -> list[dict]:
    from concurrent.futures import ThreadPoolExecutor
    jobs = [(it, s) for it in items for s in seeds]
    if not jobs:
        return []
    from .client import ChatClient
    clients = [ChatClient(s.url, decoding) for s in srvs]

    def one(ij):
        i, (it, s) = ij
        try:
            r = clients[i % len(clients)].chat([{"role": "user", "content": it["prompt"]}], seed=stable_seed("beh", it["id"], s), max_tokens=64)
            txt = r["message"].get("content") or ""
            ok = _check(it, txt) and r["finish_reason"] != "length"
            return {"task": it["id"], "family": it["id"].split("-")[0], "seed": s, "side": side, "success": ok,
                    "messages": [{"role": "user", "content": it["prompt"]}, {"role": "assistant", "content": txt}]}
        except InfraError as e:
            return {"task": it["id"], "family": it["id"].split("-")[0], "seed": s, "side": side, "success": None, "infra_error": str(e)}

    with ThreadPoolExecutor(max(1, 4 * len(clients))) as ex:
        return list(ex.map(one, enumerate(jobs)))


# ---------------------------------------------------------------- size-matched cap + floors

class SizeMatchedCap:
    """cap(bpw) = general KL of a uniform quant at the same average bpw (piecewise linear between ladder points)."""

    def __init__(self, points: list[tuple[float, float]]):
        pts = sorted(points)
        self.bpw = [p[0] for p in pts]
        self.kl = [p[1] for p in pts]

    def __call__(self, bpw: float) -> float:
        if bpw >= self.bpw[-1]:
            return self.kl[-1]
        if bpw <= self.bpw[0]:
            return self.kl[0]
        j = bisect.bisect_left(self.bpw, bpw)
        x0, x1, y0, y1 = self.bpw[j - 1], self.bpw[j], self.kl[j - 1], self.kl[j]
        return y0 + (y1 - y0) * (bpw - x0) / (x1 - x0)

    def to_dict(self) -> dict:
        return {"points": list(zip(self.bpw, self.kl))}


class NoCap:
    """No general-KL cap: every bpw is allowed."""

    def __call__(self, bpw: float) -> float:
        return float("inf")

    def to_dict(self) -> dict:
        return {"kind": "off"}


def make_cap(kind: str, points=None):
    """Return `SizeMatchedCap(points)` for `'general_kl'` or `NoCap()` for `'off'`."""
    if kind == "general_kl":
        return SizeMatchedCap(points)
    if kind == "off":
        return NoCap()
    raise ValueError(f"unknown cap kind {kind!r}")


def floors_from_solo(solo: dict[str, dict[str, dict]], uniform: dict[str, dict], ladder: list[str], factor: float,
                     key: str = "kl_gen") -> dict:
    """solo[group][setting] = {key, 'bytes_saved'}; uniform[tier] = {key, 'bytes_saved'}.

    A solo setting is forbidden when its cost (`key`: general KL by default, or e.g. the task proxy) per
    byte saved exceeds `factor` times the uniform-quant average at that tier (drop is compared with the
    cheapest uniform tier). The floor is the cheapest tier reached before the first forbidden tier; drop
    is allowed only if it passes too.
    """
    floors = {}
    cheapest = ladder[-1]
    for g, settings in solo.items():
        min_tier, evidence = ladder[1] if len(ladder) > 1 else ladder[0], {}
        allowed_chain = True
        for t in ladder[1:]:
            if t not in settings:
                continue
            s, u = settings[t], uniform[t]
            slope_u = u[key] / max(1, u["bytes_saved"])
            ok = s["bytes_saved"] > 0 and s[key] <= factor * slope_u * s["bytes_saved"]
            evidence[t] = {**s, "limit": factor * slope_u * s["bytes_saved"], "ok": ok}
            if allowed_chain and ok:
                min_tier = t
            else:
                allowed_chain = False
        drop_ok = False
        if "drop" in settings:
            s, u = settings["drop"], uniform[cheapest]
            slope_u = u[key] / max(1, u["bytes_saved"])
            drop_ok = s[key] <= factor * slope_u * s["bytes_saved"]
            evidence["drop"] = {**s, "limit": factor * slope_u * s["bytes_saved"], "ok": drop_ok}
        floors[g] = {"min_tier": min_tier, "drop_ok": drop_ok, "evidence": evidence}
    return floors
