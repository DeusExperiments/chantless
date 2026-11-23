"""General-data sanity: text corpus, executable behavioral checks, size-matched cap, solo-application floors.

Computes per-group minimum precisions from solo tests, enforces a size-matched general-KL cap, and runs
executable behavioral checks on screens and gates.
"""
from __future__ import annotations

import bisect
import glob
import json
import logging
import math
import os
import random
import re
import struct
import sysconfig
from pathlib import Path
from typing import Any

from .client import InfraError
from .util import stable_seed

_LOG = logging.getLogger(__name__)


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


_INKLING2_WIRE_TAGS = (
    "<|message_model|>",
    "<|content_text|>",
    "<|end_message|>",
    "<|content_model_end_sampling|>",
)


def _strip_reasoning_and_wire(text: str | None) -> str:
    if not isinstance(text, str) or not text:
        return ""
    t = text
    t = re.sub(r"<think>.*?(?:</think>|$)", "", t, flags=re.DOTALL)
    t = re.sub(r"<\|channel>thought.*?(?:<channel\|>|$)", "", t, flags=re.DOTALL)
    t = re.sub(
        r"(?:<\|message_model\|>[^<\n]*)?<\|content_thinking\|>.*?(?:<\|end_message\|>|$)",
        "",
        t,
        flags=re.DOTALL,
    )
    for tag in _INKLING2_WIRE_TAGS:
        t = t.replace(tag, "")
    return t


def _check(item: dict, text: str) -> bool:
    if not isinstance(text, str):
        return False
    t = _strip_reasoning_and_wire(text).strip().strip("`").strip()
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


def _infer_dialect(srvs, dialect: str | None) -> str | None:
    if dialect is not None:
        return dialect
    if srvs:
        for s in srvs:
            m = getattr(s, "model", None)
            if m and Path(m).is_file():
                try:
                    from . import gguf_ops
                    return "inkling2" if gguf_ops.needs_special_tokens(m) else None
                except (OSError, ValueError, struct.error) as exc:
                    _LOG.warning("cannot auto-detect dialect from %s (%s)", m, exc)
                    return None
        if any(bool(getattr(s, "special", False)) for s in srvs):
            return "inkling2"
    return None


def run_behavioral(
    srvs,
    decoding: dict,
    items: list[dict],
    seeds: list[int],
    side: str,
    dialect: str | None = None,
    cache_prompt: bool = False,
) -> list[dict]:
    from concurrent.futures import ThreadPoolExecutor
    jobs = [(it, s) for it in items for s in seeds]
    if not jobs:
        return []
    from .client import ChatClient
    eff_dialect = _infer_dialect(srvs, dialect)
    clients = [ChatClient(s.url, decoding, dialect=eff_dialect, cache_prompt=cache_prompt) for s in srvs]

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
        if not points:
            raise ValueError("SizeMatchedCap requires at least one point")
        by_bpw: dict[float, float] = {}
        for x, y in points:
            fx, fy = _finite(x, "bpw"), _finite(y, "kl")
            by_bpw[fx] = fy if fx not in by_bpw else min(by_bpw[fx], fy)
        pts = sorted(by_bpw.items())
        self.bpw = [p[0] for p in pts]
        self.kl = [p[1] for p in pts]

    def __call__(self, bpw: float) -> float:
        fbpw = _finite(bpw, "bpw")
        if fbpw >= self.bpw[-1]:
            return self.kl[-1]
        if fbpw <= self.bpw[0]:
            return self.kl[0]
        j = bisect.bisect_left(self.bpw, fbpw)
        x0, x1, y0, y1 = self.bpw[j - 1], self.bpw[j], self.kl[j - 1], self.kl[j]
        if x1 == x0:
            return float(min(y0, y1))
        return y0 + (y1 - y0) * (fbpw - x0) / (x1 - x0)

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


def _finite(val: Any, name: str) -> float:
    if isinstance(val, bool):
        raise ValueError(f"invalid boolean for {name}: {val!r}")
    f = float(val)
    if not math.isfinite(f):
        raise ValueError(f"non-finite value for {name}: {val!r}")
    return f


def floors_from_solo(solo: dict[str, dict[str, dict]], uniform: dict[str, dict], ladder: list[str], factor: float,
                     key: str = "kl_gen") -> dict:
    """solo[group][setting] = {key, 'bytes_saved'}; uniform[tier] = {key, 'bytes_saved'}.

    A solo setting is forbidden when its cost (`key`: general KL by default, or e.g. the task proxy) per
    byte saved exceeds `factor` times the uniform-quant average at that tier (drop is compared with the
    cheapest uniform tier). The floor is the cheapest tier reached before the first forbidden tier; drop
    is allowed only if it passes too.
    """
    if not ladder:
        raise ValueError("ladder must be non-empty")
    ffactor = _finite(factor, "factor")
    if ffactor < 0.0:
        raise ValueError(f"factor must be non-negative, got {factor!r}")
    floors = {}
    cheapest = ladder[-1]

    def _eval_one(s: dict, u: dict, label: str) -> tuple[float, bool]:
        s_cost = _finite(s[key], f"{label}.{key}")
        s_bytes = _finite(s["bytes_saved"], f"{label}.bytes_saved")
        u_cost = _finite(u[key], f"{label}.uniform.{key}")
        u_bytes = _finite(u["bytes_saved"], f"{label}.uniform.bytes_saved")
        if u_bytes <= 0.0 or s_bytes <= 0.0:
            return 0.0, False
        if u_cost <= 0.0:
            return 0.0, s_cost <= 0.0
        slope_u = u_cost / u_bytes
        limit = ffactor * slope_u * s_bytes
        return limit, s_cost <= limit

    for g, settings in solo.items():
        min_tier, evidence = ladder[1] if len(ladder) > 1 else ladder[0], {}
        allowed_chain = True
        for t in ladder[1:]:
            if t not in settings:
                continue
            s, u = settings[t], uniform[t]
            limit, ok = _eval_one(s, u, f"{g}.{t}")
            evidence[t] = {**s, "limit": limit, "ok": ok}
            if allowed_chain and ok:
                min_tier = t
            else:
                allowed_chain = False
        drop_ok = False
        if "drop" in settings:
            s, u = settings["drop"], uniform[cheapest]
            limit, drop_ok = _eval_one(s, u, f"{g}.drop")
            evidence["drop"] = {**s, "limit": limit, "ok": drop_ok}
        floors[g] = {"min_tier": min_tier, "drop_ok": drop_ok, "evidence": evidence}
    return floors
