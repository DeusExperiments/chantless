"""Estimate the work of a run config without running it: moves, probes, episodes, tokens, wall-clock and cost.

Nothing is built, served or scored. Counts follow pipeline.Run's phase logic. Sizes come from the GGUF header,
the config and, when it loads, the environment; receipts an earlier attempt of the same run left behind
(sanity mask, proxy store sizes, A/A gate size) are reused. Every other number is an assumption, and each one
is listed in the result. Time is a throughput model: prefill FLOPs = 2 x parameters x tokens at an effective
rate per GPU, decode bounded by memory bandwidth, builds at a rate per source GB.
"""
from __future__ import annotations

import copy
import json
import math
from collections import Counter
from pathlib import Path

import gguf

from . import gguf_ops, mask as mk
from .learner import LearnerState, legal_moves

# Effective rates, not peaks. tflops: prefill per GPU at 2 x params FLOPs per token; decode_gbps: weight bytes
# read per second per GPU during decode; build_s_per_gb: one llama-quantize build per GB of source.
RATES = {
    "laptop": {"label": "laptop M2 Pro", "gpus": 1, "tflops": 3.5, "decode_gbps": 120, "build_s_per_gb": 20,
               "build_parallel": 2, "usd_per_hour": 0.0,
               "basis": "measured: logodds 166k tokens of the BF16 0.75B detector in 65 s (3.8 TFLOPS); llama-server "
                        "gates of run detect-400mb-002 at 2.9-3.5 TFLOPS; build rate from its 96 s per 4-probe step; "
                        "owned hardware, no hourly cost"},
    "h100x1": {"label": "1xH100", "gpus": 1, "tflops": 57, "decode_gbps": 2000, "build_s_per_gb": 8,
               "build_parallel": 2, "usd_per_hour": 2.5,
               "basis": "prefill from the climb exam on 8xH100 (129M tokens in 427 s, 2 logodds processes per GPU); "
                        "decode and build rates assumed; price a typical on-demand rate, not a quote"},
    "h100x8": {"label": "8xH100", "gpus": 8, "tflops": 57, "decode_gbps": 2000, "build_s_per_gb": 8,
               "build_parallel": 16, "usd_per_hour": 20.0,
               "basis": "as 1xH100 per GPU, with work spread over all 8 GPUs; price a typical on-demand rate, not a quote"},
}


def tensor_bytes(t: dict, tier: str) -> int:
    """Bytes of one quantizable tensor at `tier`; a row length the block size does not divide falls back to Q8_0."""
    if tier == mk.DROP:
        return 0
    qt = gguf.GGMLQuantizationType[tier]
    blk, size = gguf.GGML_QUANT_SIZES[qt]
    if t["shape"][0] % blk:
        blk, size = gguf.GGML_QUANT_SIZES[gguf.GGMLQuantizationType.Q8_0]
    return t["n_elements"] // blk * size


class SizeModel:
    """Built-file bytes of a mask, from the source header: quantizable tensors at their tier, the rest as stored."""

    def __init__(self, path: str):
        self.inv = gguf_ops.inventory(path)
        self.by_name = {t["name"]: t for t in self.inv}
        self.groups = gguf_ops.groups_from_inventory(self.inv)
        self.params = sum(t["n_elements"] for t in self.inv)
        self.file_bytes = gguf_ops.file_bytes(path)
        self.overhead = self.file_bytes - sum(t["n_bytes"] for t in self.inv)   # metadata and alignment
        self.fixed = sum(self.by_name[n]["n_bytes"] for n in self.groups["_fixed"]["fixed"])

    def group_bytes(self, g: str, tier: str) -> int:
        if tier == mk.DROP:
            return 0
        info = self.groups[g]
        return sum(tensor_bytes(self.by_name[n], tier) for n in info["quantizable"]) + \
            sum(self.by_name[n]["n_bytes"] for n in info["fixed"])

    def mask_bytes(self, m: dict) -> int:
        return self.overhead + self.fixed + sum(self.group_bytes(g, e["tier"]) for g, e in m["groups"].items())


def _token_stats(env, A: list) -> dict:
    """{split: {"n": tasks, "mean": tokens per task}} from env.token_counts(), whatever shape it returns."""
    n_split = Counter(t.split for t in env.tasks())
    raw = env.token_counts()
    out = {}
    for s, v in raw.items():
        n = n_split.get(s, 0)
        if isinstance(v, (int, float)):
            total = float(v)
        elif isinstance(v, dict) and "mean" in v:
            out[s] = {"n": int(v.get("n", n)), "mean": float(v["mean"])}
            continue
        elif isinstance(v, dict) and ("tokens" in v or "total" in v):
            total, n = float(v.get("tokens", v.get("total"))), int(v.get("n", v.get("docs", v.get("tasks", n))))
        elif isinstance(v, dict):
            vals = list(v.values())
            total, n = float(sum(vals)), len(vals)
        else:
            vals = list(v)
            total, n = float(sum(vals)), len(vals)
        out[s] = {"n": n, "mean": total / max(1, n)}
    A.append("tokens per task from env.token_counts(): " + ", ".join(f"{s} {d['mean']:.0f} x {d['n']}" for s, d in out.items()))
    return out


def _receipt(run_dir: Path, rel: str):
    p = run_dir / rel
    return json.loads(p.read_text()) if p.exists() else None


def _moves(cfg, sm: SizeModel, start: dict, A: list) -> dict:
    ladder, lc = cfg["ladder"], cfg.get("learner", {})
    groups = sm.groups
    sweep = legal_moves(LearnerState(mask=start), groups, ladder)
    # one rung up as the exchange phase allows it: never above ladder[1]; a dropped group returns at its cheapest tier
    up = sum(1 for g, e in start["groups"].items()
             if (e["tier"] == mk.DROP) or ladder.index(e["tier"]) > 1)
    start_b = sm.mask_bytes(start)
    target = lc.get("target_bytes", 0)

    def greedy(allow_drop: bool, stop: int) -> int:
        m, b, n = copy.deepcopy(start), start_b, 0
        while b > stop:
            mv = [x for x in legal_moves(LearnerState(mask=m), groups, ladder) if allow_drop or x.tier != mk.DROP]
            if not mv:
                return -1 if stop else n
            best = max(mv, key=lambda x: sm.group_bytes(x.group, m["groups"][x.group]["tier"]) - sm.group_bytes(x.group, x.tier))
            b -= sm.group_bytes(best.group, m["groups"][best.group]["tier"]) - sm.group_bytes(best.group, best.tier)
            m = mk.apply(m, best.group, best.tier)
            n += 1
        return n

    rung = [sm.group_bytes(x.group, start["groups"][x.group]["tier"]) - sm.group_bytes(x.group, x.tier)
            for x in sweep if x.tier != mk.DROP]
    out = {"start_bytes": start_b, "target_bytes": target or None, "legal_per_sweep": len(sweep),
           "legal_per_sweep_with_up": len(sweep) + up, "rung_moves": len(rung), "drop_moves": len(sweep) - len(rung),
           "mean_rung_saving": int(sum(rung) / max(1, len(rung)))}
    max_moves = lc.get("max_moves", 10 ** 9)
    if target and start_b > target:
        out["fewest"] = greedy(True, target)
        out["fewest_rungs_only"] = greedy(False, target)
        out["typical"] = math.ceil((start_b - target) / max(1, out["mean_rung_saving"]))
        planned = min(max_moves, out["typical"])
        A.append(f"accepted moves = min(max_moves {max_moves}, typical {out['typical']}): the bytes to shed over the mean "
                 f"one-rung saving at the start; fewest possible {out['fewest']} (largest saving first, drops allowed)")
    elif target:
        planned = 0
        A.append("the start mask is already at or under target_bytes: no accepted moves")
    else:
        out["exhaust"] = greedy(False, 0)
        planned = min(max_moves, out["exhaust"])
        A.append(f"no target_bytes: accepted moves = min(max_moves, {out['exhaust']} one-rung moves to the ladder's end)")
    out["planned"] = planned
    return out


def estimate(cfg: dict, rates: dict | None = None) -> dict:
    """Counts, tokens and per-backend wall-clock and cost of running `cfg`; `rates` overrides or extends RATES."""
    A: list[str] = []
    rt = copy.deepcopy(RATES)
    for k, v in (rates or {}).items():
        rt[k] = {**rt.get(k, {}), **v}
    run_dir = Path(cfg["run_root"]) / cfg["run_id"]
    lc, st, pc, sc = cfg.get("learner", {}), cfg.get("stats", {}), cfg.get("pool", {}), cfg.get("sanity", {})
    px, scr = cfg.get("proxy", {}), cfg.get("screen", {})
    ladder = cfg["ladder"]

    # ---- model and groups
    sm = SizeModel(cfg["source"]["gguf"])
    groups = sm.groups
    imx = run_dir / "proxy" / "imatrix.gguf"
    if imx.exists():
        names = {t.name.rsplit(".", 1)[0] for t in gguf.GGUFReader(str(imx)).tensors}
        for g, info in groups.items():
            if not g.startswith("_"):
                info["imatrix_ok"] = all(n in names for n in info["quantizable"])
        A.append("importance-data coverage per group read from the run's imatrix.gguf")
    else:
        for g in ("token_embd", "output"):
            if g in groups:
                groups[g]["imatrix_ok"] = False
        A.append("importance data assumed for every block group and not for token_embd/output (llama-imatrix collects none there)")
    sanity_mask = _receipt(run_dir, "sanity/sanity_mask.json")
    if sanity_mask:
        start = sanity_mask
        A.append("start mask: this run's sanity/sanity_mask.json (its floors apply)")
    else:
        start = mk.new_mask(groups, cfg["start_tier"], {}, cfg.get("objective_mode", "preserve"))
        start["groups"] = {g: {"tier": mk.clip(groups[g], cfg["start_tier"], ladder)} for g in start["groups"]}
        A.append(f"start mask: uniform {cfg['start_tier']} with no sanity floors (floors only remove moves)")
    A.append("built bytes from the header: quantizable tensors at their tier (Q8_0 where the block size does not divide a row), "
             "other tensors as stored, plus the source's metadata overhead")
    moves = _moves(cfg, sm, start, A)
    acc = moves["planned"]

    # ---- environment
    env, toks, classifier = None, None, bool(cfg["env"].get("kwargs", {}).get("classifier", False))
    try:
        from .env.protocol import load_environment
        env = load_environment(cfg["env"]["module"], **cfg["env"].get("kwargs", {}))
        classifier = bool(getattr(env, "classifier", False))
        avail = Counter(t.split for t in env.tasks())
        if callable(getattr(env, "token_counts", None)):
            toks = _token_stats(env, A)
    except Exception as e:  # noqa: BLE001 - a plan must still print without the data
        avail = Counter()
        A.append(f"environment not loaded ({type(e).__name__}: {str(e)[:120]}); split sizes taken from the config")
    if toks is None:
        toks = {}
        A.append("no env.token_counts(): 1,000 prompt tokens per task")
    budget = env.budget if env is not None else {}
    if classifier:
        dec = 0.0
    elif budget.get("max_turns"):
        dec = 0.5 * budget["max_turns"] * budget.get("max_tokens_per_turn", 512)
        A.append(f"{dec:.0f} decoded tokens per episode: half the turn budget")
    elif "max_tokens" in budget:
        dec = float(budget["max_tokens"])
    else:
        dec = 256.0
        A.append("256 decoded tokens per episode (no environment budget known)")
    mean = lambda s: toks.get(s, {}).get("mean", 1000.0)
    tps = pc.get("tasks_per_split", {})
    n_of = lambda s, n: min(n, avail[s]) if avail.get(s) else n
    pool_n = {s: n_of(s, n) for s, n in tps.items() if s != "exam"}
    sel = pool_n.get("select", 0)
    n_search = pool_n.get("search", 0)
    A.append("every screened task counts as reliable (the pool's pass rule only shrinks the counts)")
    aa = _receipt(run_dir, "aa.json")
    if st.get("n_override"):
        n_gate = st["n_override"]
    elif aa:
        n_gate = aa["n_gate"]
        A.append("gate size from this run's aa.json")
    else:
        n_gate = st.get("n_floor", 0)
        A.append(f"gate size = n_floor {n_gate}: the A/A sample-size rule can only raise it")
    k_gate = lambda n: min(st.get("max_eps_per_task", 1), max(1, math.ceil(n_gate / max(1, n))))
    exam_n = n_of("exam", tps.get("exam", tps.get("select", 0)))
    n_screens = acc // lc.get("screen_every", 4) if lc.get("screen_every") else 0
    n_gates = acc // math.lcm(lc.get("screen_every", 4), lc.get("gate_every", 10 ** 9)) if lc.get("screen_every") else 0
    n_beh = sc.get("behavioral_items", 0) * sc.get("behavioral_seeds", 1)

    # ---- store sizes (tokens scored per probe)
    gen_tok = px.get("general_seq_len", 1024) * px.get("general_n_seq", 4)
    cap_kind = sc.get("cap", "general_kl")
    pj = _receipt(run_dir, "proxy/proxy.json")
    if classifier:
        n_neg, n_pos = px.get("n_neg", 1000), px.get("n_pos", 600)
        task_tok = n_neg * mean("calibration") + n_pos * mean("search")
        A.append(f"log-odds store: {n_neg} calibration negatives + {n_pos} search positives (proxy.n_neg / proxy.n_pos)")
        if cap_kind == "off":
            gen_tok = 0
            A.append("sanity.cap is off: no general store is scored")
    elif pj:
        task_tok, gen_tok = pj["task"]["n_tokens"], pj["general"]["n_tokens"]
        A.append("store sizes from this run's proxy/proxy.json")
    elif toks:
        task_tok = px.get("task_max_targets", 3000) * mean("search")
        A.append("task store: one target per document (task_max_targets x mean search tokens)")
    else:
        task_tok = 2 * px.get("task_max_targets", 3000)
        A.append("task store: half the rendered tokens are targets (2 x task_max_targets)")
    probe_tok = gen_tok + task_tok

    # ---- probes and builds
    L, L_up = moves["legal_per_sweep"], moves["legal_per_sweep_with_up"]
    real = [g for g in groups if not g.startswith("_")]
    solo_tiers = sc.get("solo_tiers") or ladder[1:]
    n_solo = sum(sum(1 for t in solo_tiers if t in ladder and mk.tier_allowed(groups[g], t)) + groups[g]["droppable"]
                 for g in real)
    per_step = L if lc.get("reprobe_every", 8) == 1 else lc.get("top_k", 16) + 1
    n_fb = n_screens if lc.get("feedback", True) else 0
    rounds = lc.get("exchange_rounds", 0)
    search_probes = 1 + L + acc * per_step + n_screens * (1 + (L if lc.get("reprobe_after_screen", True) else 0))
    fb_probes = n_fb * (lc.get("screen_every", 4) + 1)
    ex_probes = rounds * (L_up + lc.get("exchange_top_k", 32))
    A.append(f"search probes: a full sweep at the start{' and after every screen' if lc.get('reprobe_after_screen', True) else ''}, "
             f"then {per_step} per accepted move; every probe scores every store (the general-cap shortcut ignored)")
    if n_fb:
        A.append("feedback: every screen is assumed to yield failures (one failure store and one revert trial per windowed move)")
    if rounds:
        A.append(f"exchange: {rounds} rounds (learner.exchange_rounds), each a sweep with one rung up plus exchange_top_k pairs")
    # exam baselines, as exam_baselines() picks them, at the estimated bytes of the target
    target_b = moves["target_bytes"] or moves["start_bytes"]
    blk = [g for g in real if g.startswith("blk.")]
    base_probes = base_builds = 0
    for b_t, rank_t, dep_t in (("Q4_K", "Q8_0", "Q8_0"), ("Q2_K", "Q8_0", "Q4_K")):
        if not {b_t, rank_t, dep_t} <= set(ladder):
            continue
        ub = sm.mask_bytes({"groups": {g: {"tier": mk.clip(groups[g], b_t, ladder)} for g in real}})
        if ub > target_b and mk.rank(dep_t) < mk.rank(b_t):
            continue
        base_probes += 1 + len(blk)
        base_builds += math.ceil(math.log2(len(blk) + 1)) + 2
    n_base_models = 1 + (1 if base_probes else 0)
    builds = {"sanity": len(ladder) - 1 + n_solo, "search": search_probes + fb_probes + ex_probes + n_screens + n_gates + 1,
              "freeze": 2, "exam": base_probes + base_builds + n_base_models}
    probes = {"sanity": len(ladder) - 1 + n_solo, "search": search_probes, "feedback": fb_probes, "exchange": ex_probes,
              "baselines": base_probes}

    # ---- evaluation work: (stage, items, prefill tokens, decode tokens)
    work = []

    def add(stage, items, tok, dec_tok=0.0):
        if items:
            work.append({"stage": stage, "items": int(items), "prefill_tokens": float(tok), "decode_tokens": float(dec_tok)})

    src_scores = cfg.get("exam", {}).get("source_scores")
    if classifier:
        # one score per distinct task per model; each new model is calibrated once on the calibration split
        n_cal = avail.get("calibration") or toks.get("calibration", {}).get("n", 0)
        cal_tok = n_cal * mean("calibration")
        n_models = 1 + n_screens + 1 + n_base_models
        A.append(f"classifier: each distinct task scored once per model and per evaluation call; {n_models} models calibrated "
                 f"on {n_cal} calibration documents (source, one per screen, the final mask, the exam baselines); a gate "
                 "re-uses the rule of the screen on the same mask")
        add("calibration", n_models * n_cal, n_models * cal_tok)
        add("pool (source)", sum(pool_n.values()), sum(n * mean(s) for s, n in pool_n.items()))
        add("A/A (source)", 2 * sel, 2 * sel * mean("select"))
        add("screens", n_screens * (min(scr.get("pairs", 0), sel) + min(lc.get("feedback_tasks", 0), n_search)),
            n_screens * (min(scr.get("pairs", 0), sel) * mean("select") + min(lc.get("feedback_tasks", 0), n_search) * mean("search")))
        g_mult = 1 if st.get("reuse_source") else 2
        add("gates", (n_gates + 1) * sel * g_mult, (n_gates + 1) * sel * g_mult * mean("select"))
        ex_src = 0 if src_scores else exam_n
        add("exam", exam_n + ex_src + (n_base_models * exam_n), (exam_n + ex_src + n_base_models * exam_n) * mean("exam"))
        add("proxy store (source)", 1, task_tok)
    else:
        def eps(stage, n, split):
            add(stage, n, n * mean(split), n * dec)
        eps("pool (source)", sum(n * pc.get("runs", 1) for n in pool_n.values()), "search")
        k_aa = min(st.get("max_eps_per_task", 1), max(1, math.ceil(st.get("n_aa", 0) / max(1, sel))))
        eps("A/A (source)", 2 * sel * k_aa, "select")
        eps("screens", n_screens * min(scr.get("pairs", 0), sel), "select")
        eps("feedback", n_screens * min(lc.get("feedback_tasks", 0), n_search), "search")
        g_mult = 1 if st.get("reuse_source") else 2
        eps("gates", (n_gates + 1) * sel * k_gate(sel) * g_mult, "select")
        ex_pool = 0 if src_scores else exam_n * pc.get("runs", 1)
        ex_src = 0 if (st.get("reuse_source") or src_scores) else exam_n * k_gate(exam_n)
        eps("exam", ex_pool + exam_n * k_gate(exam_n) + ex_src + n_base_models * exam_n * cfg.get("exam", {}).get("baseline_eps_per_task", 1), "exam")
        nb = n_beh * (1 + n_screens + n_gates + 2)
        add("behavioral", nb, nb * 40, nb * 16)
        add("proxy stores (source)", 2, 2 * probe_tok)
        A.append("behavioral items: 40 prompt and 16 decoded tokens each; all gate looks run (no early stop)")
    n_probes = sum(probes.values())
    n_uni = len(ladder) - 1
    add("probes: sanity", probes["sanity"], n_uni * probe_tok + n_solo * gen_tok)
    add("probes: search", search_probes + ex_probes, (search_probes + ex_probes) * probe_tok)
    add("probes: feedback", fb_probes, fb_probes * task_tok)
    add("probes: baselines", base_probes, base_probes * task_tok)
    A.append(f"probe tokens: general {gen_tok:,} + task {task_tok:,.0f}; solo sanity probes score the general store "
             "only, baseline and feedback probes one task-sized store")

    # ---- time and cost per backend
    tot_pre = sum(w["prefill_tokens"] for w in work)
    tot_dec = sum(w["decode_tokens"] for w in work)
    n_builds = sum(builds.values())
    slots = cfg.get("serving", {}).get("slots_per_gpu", 4)
    gb = sm.file_bytes / 1e9
    A.append(f"decode reads the source's {gb:.2f} GB of weights per step on each of {slots} batched slots per GPU; "
             "server start-up, model loads and imatrix are not counted")
    A.append("builds run beside scoring only up to build_parallel at once; scoring spreads over every GPU of a backend")
    backends = {}
    for name, r in rt.items():
        pre_s = tot_pre * 2 * sm.params / (r["tflops"] * 1e12 * r["gpus"])
        dec_s = tot_dec / (r["gpus"] * slots * r["decode_gbps"] / gb) if tot_dec else 0.0
        bld_s = n_builds * r["build_s_per_gb"] * gb / r["build_parallel"]
        h = (pre_s + dec_s + bld_s) / 3600
        backends[name] = {"label": r.get("label", name), "hours": h, "usd": h * r["usd_per_hour"],
                          "prefill_h": pre_s / 3600, "decode_h": dec_s / 3600, "build_h": bld_s / 3600,
                          "within_budget": h <= cfg.get("budget", {}).get("max_hours", float("inf")), "basis": r.get("basis", "")}
    return {"run_id": cfg["run_id"], "env": cfg["env"]["module"], "classifier": classifier,
            "model": {"params": sm.params, "source_bytes": sm.file_bytes, "groups": len(real), "ladder": ladder},
            "moves": moves, "probes": probes, "builds": builds, "work": work,
            "totals": {"prefill_tokens": tot_pre, "decode_tokens": tot_dec, "builds": n_builds, "probes": n_probes},
            "backends": backends, "rates": rt, "assumptions": A}


def format_table(est: dict) -> str:
    m, mv = est["model"], est["moves"]
    lines = [f"plan {est['run_id']}: {est['env']} ({'classifier' if est['classifier'] else 'episode'} path)",
             f"model: {m['params']:,} params, {m['source_bytes']:,} B source, {m['groups']} groups, ladder {' > '.join(m['ladder'])}",
             f"moves: start {mv['start_bytes']:,} B -> target {mv['target_bytes'] or 0:,} B; legal per sweep {mv['legal_per_sweep']} "
             f"({mv['legal_per_sweep_with_up']} with one rung up); fewest {mv.get('fewest', '-')}, typical {mv.get('typical', '-')}, planned {mv['planned']}",
             "probes: " + ", ".join(f"{k} {v:,}" for k, v in est["probes"].items()) + f"; builds {est['totals']['builds']:,}", "",
             f"{'stage':<22}{'items':>10}{'prefill tok':>16}{'decode tok':>14}"]
    for w in est["work"]:
        lines.append(f"{w['stage']:<22}{w['items']:>10,}{w['prefill_tokens']:>16,.0f}{w['decode_tokens']:>14,.0f}")
    t = est["totals"]
    lines += [f"{'total':<22}{'':>10}{t['prefill_tokens']:>16,.0f}{t['decode_tokens']:>14,.0f}", "",
              f"{'backend':<16}{'hours':>9}{'prefill':>9}{'decode':>9}{'build':>9}{'cost $':>10}  budget"]
    for b in est["backends"].values():
        lines.append(f"{b['label']:<16}{b['hours']:>9.2f}{b['prefill_h']:>9.2f}{b['decode_h']:>9.2f}{b['build_h']:>9.2f}"
                     f"{b['usd']:>10.2f}  {'ok' if b['within_budget'] else 'OVER'}")
    lines += ["", "assumptions:"] + [f"- {a}" for a in est["assumptions"]]
    lines += [f"- rates {k}: {r.get('basis', '')}" for k, r in est["rates"].items()]
    return "\n".join(lines)
