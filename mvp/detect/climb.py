"""Mask-only hill climb for the detector, scored on the decision itself, parallel over every GPU.

Usage: python detect/climb.py CONFIG.json STEP     (STEP: source, sets, climb, finalists, exam)
Every step writes its receipts under cfg["out"] and the next step reads them.
- source: the BF16 source's log-odds (log p(human) - log p(ai) at the answer) for every document in
  data/{val,cal,test,beemo}.trj, sharded over all GPU slots.
- sets: the reference set R the climb scores and the select set S that picks the finalist. R is a random
  sample of calibration humans (its 1% cutoff stands in for the 20,000's) plus the AI
  documents within 3 of the source cutoff from WildOutlines validation and the Beemo search split. S is every
  other AI document of those two pools; S's cutoff comes from all 20,000 calibration humans.
- climb: start at uniform Q4_K, greedy down to the byte budget, then exchange moves (one group up, one down or
  removed) while the R objective improves. Objective: partial AUC of the AI documents against the candidate's
  own hardest 5% of humans, so a uniform shift of the log-odds costs nothing. Move score = objective lost per byte saved.
- finalists: the best masks under the budget, rebuilt and scored on S at their own cutoff; the best is frozen.
- exam: the frozen mask on all WildOutlines test documents and the Beemo exam split, written as score-shard
  files beside the source's and ModernBERT-L's, for detect/measure.py.
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from chantless import mask as mk  # noqa: E402
from chantless.builder import Builder, mask_key  # noqa: E402
from chantless.runtime import Toolchain  # noqa: E402

cfg = json.loads(Path(sys.argv[1]).read_text())
STEP = sys.argv[2]
OUT, DATA = Path(cfg["out"]), Path(cfg["data"])
OUT.mkdir(parents=True, exist_ok=True)
PAIR = "24509,2075"
TAU = 0.5           # softness of "caught" in the R objective, in log-odds units
LADDER = ["Q8_0", "Q4_K", "IQ1_S"]
SLOTS = [(g, k) for g in cfg["gpus"] for k in range(cfg["procs_per_gpu"])]
_free, _cv = list(SLOTS), threading.Condition()


def log(msg):
    line = f"{time.strftime('%H:%M:%S')}  {msg}"
    print(line, flush=True)
    with open(OUT / "climb.log", "a") as f:
        f.write(line + "\n")


def slot():
    with _cv:
        while not _free:
            _cv.wait()
        return _free.pop()


def release(s):
    with _cv:
        _free.append(s)
        _cv.notify()


def run_logodds(model, trj, out, shard="0/1", s=None):
    own = s is None
    s = s or slot()
    try:
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(s[0]))
        subprocess.run([cfg["logodds"], "-m", model, "--traj", trj, "--pair", PAIR, "--out", out, "--shard", shard],
                       check=True, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    finally:
        if own:
            release(s)


def score_all(model, name, tag):
    """Log-odds for every trajectory of data/<name>.trj, sharded over all slots; cached as <tag>-<name>.npy."""
    dst = OUT / "scores" / f"{tag}-{name}.npy"
    if dst.exists():
        return np.load(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    n = len(SLOTS)
    parts = [OUT / "scores" / f"{tag}-{name}.{i}.tsv" for i in range(n)]
    with ThreadPoolExecutor(n) as ex:
        list(ex.map(lambda i: run_logodds(model, str(DATA / f"{name}.trj"), str(parts[i]), f"{i}/{n}"), range(n)))
    meta = json.loads((DATA / f"{name}.json").read_text())
    lo = np.full(len(meta), np.nan)
    for p in parts:
        for line in open(p):
            i, _, v = line.split("\t")
            lo[int(i)] = float(v)
        p.unlink()
    assert not np.isnan(lo).any(), f"{name}: missing scores"
    np.save(dst, lo)
    return lo


def cut_of(cal_lo):
    return float(np.sort(cal_lo)[int(0.01 * len(cal_lo))])  # flag strictly below: the lowest 1%


def meta_of(name):
    return json.loads((DATA / f"{name}.json").read_text())


# ---------------------------------------------------------------- steps
def step_source():
    for name in ("cal", "val", "test", "beemo"):
        t = time.time()
        lo = score_all(cfg["source"], name, "source")
        log(f"source {name}: {len(lo)} docs in {time.time() - t:.0f}s")
    log(f"source cutoff (log-odds) {cut_of(np.load(OUT / 'scores/source-cal.npy')):.4f}")


def write_subset(src_name, idx, dst):
    """A TRJ1 file holding trajectories idx (in that order) of data/<src_name>.trj."""
    import struct
    want = {int(i): k for k, i in enumerate(idx)}
    recs = [None] * len(idx)
    with open(DATA / f"{src_name}.trj", "rb") as f:
        f.read(4)
        n, = struct.unpack("<I", f.read(4))
        for i in range(n):
            lt, lg = struct.unpack("<II", f.read(8))
            body = f.read(4 * lt + 4 * lg)
            if i in want:
                recs[want[i]] = struct.pack("<II", lt, lg) + body
    with open(dst, "wb") as o:
        o.write(b"TRJ1" + struct.pack("<I", len(recs)))
        for r in recs:
            o.write(r)


def step_sets():
    s = {k: np.load(OUT / f"scores/source-{k}.npy") for k in ("cal", "val", "beemo")}
    c = cut_of(s["cal"])
    val, bee = meta_of("val"), meta_of("beemo")
    # A random sample, never the lowest-scoring tail: a tail sample regresses toward the middle under any
    # perturbation, which raises the estimated cutoff and rewards noisier candidates
    hum = np.sort(np.random.default_rng(0).choice(len(s["cal"]), cfg["r_humans"], replace=False))
    ai_pool = [("val", i) for i, m in enumerate(val) if m["label"] == "ai"] + \
              [("beemo", i) for i, m in enumerate(bee) if m["label"] == "ai" and m["split"] == "search"]
    near = sorted(((abs(s[src][i] - c), src, i) for src, i in ai_pool if abs(s[src][i] - c) <= 3.0))
    r_ai = {(src, i) for _, src, i in near[: cfg["r_ai_max"]]}  # the closest to the cutoff when there are more
    sel = [(src, i) for src, i in ai_pool if (src, i) not in r_ai]
    sets = {"cut_source": c, "R_humans": [int(i) for i in hum], "R_ai": sorted(r_ai), "S_ai": sel}
    (OUT / "sets.json").write_text(json.dumps(sets))
    for src in ("val", "beemo"):
        write_subset(src, [i for s_, i in sorted(r_ai) if s_ == src], OUT / f"R-ai-{src}.trj")
    write_subset("cal", [int(i) for i in hum], OUT / "R-humans.trj")
    log(f"sets: R = {len(hum)} random calibration humans + {len(r_ai)} AI near the cutoff "
        f"({sum(1 for s_, _ in r_ai if s_ == 'beemo')} Beemo); S = {len(sel)} AI + 20,000 calibration humans")


class Scorer:
    """R-objective of a mask: build, score R on one slot, cache the log-odds per mask key."""

    def __init__(self):
        sets = json.loads((OUT / "sets.json").read_text())
        self.src = {k: np.load(OUT / f"scores/source-{k}.npy") for k in ("cal", "val", "beemo")}
        self.r_parts = [("humans", OUT / "R-humans.trj")] + \
                       [(f"ai-{s}", OUT / f"R-ai-{s}.trj") for s in ("val", "beemo") if (OUT / f"R-ai-{s}.trj").exists()]
        groups = json.loads(Path(cfg["groups"]).read_text())
        self.groups = groups
        tc = Toolchain.from_config({"llama_bin": cfg["llama_bin"], "traj_kl": cfg["logodds"], "gpus": [""]})
        self.builder = Builder(tc, cfg["source"], groups, cfg["imatrix"], OUT / "build", log=str(OUT / "build.log"),
                               threads=cfg["quantize_threads"])
        self.builder.max_pruned = 64
        self.cache_dir = OUT / "probe"
        self.cache_dir.mkdir(exist_ok=True)
        self.lock = threading.Lock()
        self.src_obj = self.objective({"humans": self.src["cal"][sets["R_humans"]],
                                       "ai": np.concatenate([self.src[s][[i for s_, i in sets["R_ai"] if s_ == s]]
                                                             for s in ("val", "beemo")])})
        self.n = 0

    @staticmethod
    def objective(lo):
        """Partial AUC: over every pair of an AI document and one of the candidate's own hardest humans (its
        lowest 5% of the random human sample), a soft reward for ranking the AI document as more AI-like.
        Choosing the hard humans by the candidate's own scores keeps the tail estimate unbiased."""
        h = np.sort(lo["humans"])[: max(1, int(0.05 * len(lo["humans"])))]
        gap = h[None, :] - lo["ai"][:, None]
        return float(np.mean(1.0 / (1.0 + np.exp(-gap / TAU))))

    def probe(self, m):
        key = mask_key(m)
        f = self.cache_dir / f"{key}.json"
        if f.exists():
            return json.loads(f.read_text())
        b = self.builder.build(m, OUT / "build" / f"p-{key}.gguf")
        s = slot()
        try:
            lo = {}
            for name, trj in self.r_parts:
                out = OUT / "build" / f"p-{key}-{name}.tsv"
                run_logodds(b.path, str(trj), str(out), s=s)
                lo[name] = np.array([float(l.split("\t")[2]) for l in open(out)])
                out.unlink()
        finally:
            release(s)
            Path(b.path).unlink(missing_ok=True)
        lo_all = {"humans": lo["humans"], "ai": np.concatenate([lo[k] for k in ("ai-val", "ai-beemo") if k in lo])}
        res = {"key": key, "bytes": b.file_bytes, "obj": self.objective(lo_all), "groups": m["groups"]}
        f.write_text(json.dumps(res))
        with self.lock:
            self.n += 1
        return res

    def many(self, masks):
        with ThreadPoolExecutor(len(SLOTS)) as ex:
            return list(ex.map(self.probe, masks))


def moves(m, groups, allow_up):
    out = []
    n_attn = sum(1 for g, e in m["groups"].items() if g.endswith(".attn") and e["tier"] != mk.DROP)
    for g, e in m["groups"].items():
        t = e["tier"]
        if cfg.get("only_groups") and g not in cfg["only_groups"]:  # smoke tests only
            continue
        if t != mk.DROP:
            i = LADDER.index(t)
            if i + 1 < len(LADDER) and mk.tier_allowed(groups[g], LADDER[i + 1]):
                out.append((g, LADDER[i + 1]))
            if groups[g]["droppable"] and not (g.endswith(".attn") and n_attn <= 1):
                out.append((g, mk.DROP))
            if allow_up and i > 0:
                out.append((g, LADDER[i - 1]))
        elif allow_up:
            out.append((g, LADDER[-1] if mk.tier_allowed(groups[g], LADDER[-1]) else LADDER[-2]))
    return out


def step_climb():
    S = Scorer()
    groups, budget = S.groups, cfg["budget_bytes"]
    src_pin = {"repo": cfg["source_repo"], "revision": cfg["source_revision"]}
    cur = mk.new_mask(groups, "Q4_K", src_pin, "preserve")
    base = S.probe(cur)
    log(f"source R objective {S.src_obj:.4f}; start uniform Q4_K: {base['bytes']:,} B, objective {base['obj']:.4f}")
    history = [{"phase": "start", "bytes": base["bytes"], "obj": base["obj"], "key": base["key"]}]
    seen = {base["key"]: base}

    def rate(r, b):  # objective lost per byte saved; lower is better
        db = b["bytes"] - r["bytes"]
        return (b["obj"] - r["obj"]) / db if db > 0 else float("inf")

    # greedy descent: sweep every move once, then re-probe the 2 x slots best stale moves each step
    stale = {}
    sweep = moves(cur, groups, allow_up=False)
    for mv, r in zip(sweep, S.many([mk.apply(cur, *mv) for mv in sweep])):
        stale[mv] = rate(r, base)
        seen[r["key"]] = r
    log(f"greedy sweep: {len(sweep)} moves")
    while base["bytes"] > budget:
        cand = sorted((s, mv) for mv, s in stale.items() if mv in set(moves(cur, groups, False)))[: 2 * len(SLOTS)]
        res = S.many([mk.apply(cur, *mv) for _, mv in cand])
        fresh = [(rate(r, base), mv, r) for (_, mv), r in zip(cand, res)]
        for s, mv, r in fresh:
            stale[mv] = s
            seen[r["key"]] = r
        s, mv, r = min(fresh, key=lambda x: x[0])
        cur, base = mk.apply(cur, *mv), r
        stale.pop(mv, None)
        history.append({"phase": "greedy", "move": f"{mv[0]}->{mv[1]}", "bytes": r["bytes"], "obj": r["obj"], "key": r["key"]})
        log(f"greedy {mv[0]}->{mv[1]}: {r['bytes']:,} B objective {r['obj']:.4f}")

    # exchange: one group up, one group down or removed, staying under the budget, while the objective improves
    for rnd in range(cfg["exchange_rounds"]):
        singles = moves(cur, groups, allow_up=True)
        rs = S.many([mk.apply(cur, *mv) for mv in singles])
        for r in rs:
            seen[r["key"]] = r
        d = {mv: (r["obj"] - base["obj"], r["bytes"] - base["bytes"]) for mv, r in zip(singles, rs)}
        pairs = []
        for a, (oa, ba) in d.items():
            for b_, (ob, bb) in d.items():
                if a[0] != b_[0] and ba > 0 > bb and base["bytes"] + ba + bb <= budget:
                    pairs.append((oa + ob, a, b_))
        pairs += [(o, mv, None) for mv, (o, bb) in d.items() if base["bytes"] + bb <= budget]
        pairs.sort(key=lambda x: -x[0])
        trial = []
        for est, a, b_ in pairs[: 2 * len(SLOTS)]:
            m = mk.apply(cur, *a)
            if b_:
                m = mk.apply(m, *b_)
            trial.append((m, a, b_))
        res = S.many([t[0] for t in trial])
        best = max(zip(res, trial), key=lambda x: x[0]["obj"])
        for r in res:
            seen[r["key"]] = r
        if best[0]["obj"] <= base["obj"] + 1e-4:
            log(f"exchange round {rnd + 1}: no improving exchange among {len(trial)} tried; stop")
            break
        r, (m, a, b_) = best
        cur, base = m, r
        label = f"{a[0]}->{a[1]}" + (f" + {b_[0]}->{b_[1]}" if b_ else "")
        history.append({"phase": "exchange", "move": label, "bytes": r["bytes"], "obj": r["obj"], "key": r["key"]})
        log(f"exchange round {rnd + 1}: {label}: {r['bytes']:,} B objective {r['obj']:.4f}")
    (OUT / "history.json").write_text(json.dumps(history, indent=1))
    under = sorted((r for r in seen.values() if r["bytes"] <= budget), key=lambda r: -r["obj"])[: cfg["finalists"]]
    (OUT / "finalists.json").write_text(json.dumps(under, indent=1))
    log(f"climb done: {S.n} probes; finalists {[(r['key'], r['bytes'], round(r['obj'], 4)) for r in under]}")


def full_mask(groups_tiers):
    return {"schema": mk.SCHEMA, "source": {"repo": cfg["source_repo"], "revision": cfg["source_revision"]},
            "objective_mode": "preserve", "groups": groups_tiers, "tensors": {}, "floors": {},
            "weights": {"lineage": []}, "deployment": {}, "measured": {}}


def step_finalists():
    groups = json.loads(Path(cfg["groups"]).read_text())
    tc = Toolchain.from_config({"llama_bin": cfg["llama_bin"], "traj_kl": cfg["logodds"], "gpus": [""]})
    B = Builder(tc, cfg["source"], groups, cfg["imatrix"], OUT / "build", threads=cfg["quantize_threads"])
    sets = json.loads((OUT / "sets.json").read_text())
    write_subset("val", [i for s_, i in sets["S_ai"] if s_ == "val"], OUT / "S-ai-val.trj")
    write_subset("beemo", [i for s_, i in sets["S_ai"] if s_ == "beemo"], OUT / "S-ai-beemo.trj")
    src = {k: np.load(OUT / f"scores/source-{k}.npy") for k in ("cal", "val", "beemo")}
    s_src = np.concatenate([src[s][[i for s_, i in sets["S_ai"] if s_ == s]] for s in ("val", "beemo")])
    rows = [{"name": "source", "bytes": None, "cut": cut_of(src["cal"]),
             "S_caught": float(np.mean(s_src < cut_of(src["cal"])))}]
    for r in json.loads((OUT / "finalists.json").read_text()):
        m = full_mask(r["groups"])
        path = OUT / "build" / f"final-{r['key']}.gguf"
        b = B.build(m, path)
        cal = score_all(str(path), "cal", f"f-{r['key']}")
        lo = []
        for s in ("val", "beemo"):
            n = len(SLOTS)
            parts = [OUT / "build" / f"S-{r['key']}-{s}.{i}.tsv" for i in range(n)]
            with ThreadPoolExecutor(n) as ex:
                list(ex.map(lambda i: run_logodds(str(path), str(OUT / f"S-ai-{s}.trj"), str(parts[i]), f"{i}/{n}"), range(n)))
            v = {}
            for p in parts:
                for line in open(p):
                    i, _, x = line.split("\t")
                    v[int(i)] = float(x)
                p.unlink()
            lo.append(np.array([v[i] for i in range(len(v))]))
        lo = np.concatenate(lo)
        c = cut_of(cal)
        rows.append({"name": r["key"], "bytes": b.file_bytes, "R_obj": r["obj"], "cut": c, "S_caught": float(np.mean(lo < c)),
                     "groups": r["groups"]})
        log(f"finalist {r['key']}: {b.file_bytes:,} B, cutoff {c:.3f}, S AI caught {rows[-1]['S_caught']:.4f} "
            f"(source {rows[0]['S_caught']:.4f})")
    best = max(rows[1:], key=lambda x: (x["S_caught"], -x["bytes"]))
    (OUT / "finalists_S.json").write_text(json.dumps({"rows": rows, "chosen": best["name"]}, indent=1))
    final = OUT / "final"
    final.mkdir(exist_ok=True)
    os.replace(OUT / "build" / f"final-{best['name']}.gguf", final / "detector-mask.gguf")
    (final / "mask.json").write_text(json.dumps(full_mask(best["groups"]), indent=1))
    (final / "cutoff.json").write_text(json.dumps({"log_odds": best["cut"], "p_human": 1 / (1 + math.exp(-best["cut"])),
                                                   "rule": "AI when log p(human) - log p(ai) < log_odds; 1% of 20,000 calibration humans"}))
    for p in (OUT / "build").glob("final-*.gguf"):
        p.unlink()
    log(f"chosen {best['name']}: S AI caught {best['S_caught']:.4f}")


def step_exam():
    """Score the frozen mask on all test documents and the Beemo exam split; write score shards for measure.py."""
    final = OUT / "final" / "detector-mask.gguf"
    dst = OUT / "measure" / "mask"
    dst.mkdir(parents=True, exist_ok=True)
    for name, keep in (("cal", None), ("test", None), ("beemo", "exam")):
        lo = score_all(str(final), name, "mask")
        meta = meta_of(name)
        with open(dst / f"{'calibration' if name == 'cal' else name}.shard0.jsonl", "w") as f:
            for m, x in zip(meta, lo):
                if keep and m.get("split") != keep:
                    continue
                f.write(json.dumps({"id": m["id"], "label": m["label"], "kind": m["kind"], "format": m["format"],
                                    "p_human": 1 / (1 + math.exp(-x)), "tokens": m["tokens"]}) + "\n")
    log("exam scores written; run detect/measure.py on the measure folder")


{"source": step_source, "sets": step_sets, "climb": step_climb, "finalists": step_finalists, "exam": step_exam}[STEP]()
