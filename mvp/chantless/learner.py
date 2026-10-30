"""Iterative mask learner with re-probing and failure feedback (spec §2/§6, decision S4).

Moves are ranked by proxy cost per byte saved (Dettmers-style re-probing on the *current*
allocation). Task-verifier and sanity failures change later allocations: failed trajectories join
the scoring proxy, moves implicated by those failures are re-measured, then reverted or penalised,
and reverted (group, tier) settings are blocked. A proxy-only search (feedback disabled) is kept as
an ablation and does not satisfy the contract.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field

from . import mask as mk


@dataclass
class Move:
    group: str
    tier: str  # target tier (one rung down) or 'drop'

    @property
    def key(self) -> str:
        return f"{self.group}->{self.tier}"


@dataclass
class LearnerState:
    mask: dict
    cache: dict = field(default_factory=dict)          # move key -> score measured on some past mask
    penalties: dict = field(default_factory=dict)      # group -> multiplier increment from feedback
    blocked: set = field(default_factory=set)          # move keys that may not be re-applied
    window: list = field(default_factory=list)         # moves accepted since the last feedback
    accepted: int = 0


def legal_moves(state: LearnerState, groups: dict, ladder: list[str]) -> list[Move]:
    out = []
    m = state.mask
    for g, e in m["groups"].items():
        t = e["tier"]
        if t == mk.DROP:
            continue
        fl = m.get("floors", {}).get(g, {"min_tier": ladder[-1], "drop_ok": groups[g]["droppable"]})
        i = ladder.index(t)
        if i + 1 < len(ladder) and mk.rank(ladder[i + 1]) <= mk.rank(fl["min_tier"]) and mk.tier_allowed(groups[g], ladder[i + 1]):
            mv = Move(g, ladder[i + 1])
            if mv.key not in state.blocked:
                out.append(mv)
        if groups[g]["droppable"] and fl.get("drop_ok"):
            mv = Move(g, mk.DROP)
            if mv.key not in state.blocked:
                out.append(mv)
    # keep at least one attention block
    n_attn_left = sum(1 for g, e in m["groups"].items() if g.endswith(".attn") and e["tier"] != mk.DROP)
    if n_attn_left <= 1:
        out = [mv for mv in out if not (mv.tier == mk.DROP and mv.group.endswith(".attn"))]
    return out


class Learner:
    def __init__(self, prober, groups, ladder, cap, mode: str, cfg: dict, log, record):
        self.prober, self.groups, self.ladder, self.cap, self.mode = prober, groups, ladder, cap, mode
        self.cfg, self.log, self.record = cfg, log, record

    # ---- scoring
    def evaluate(self, state: LearnerState, moves: list[Move], base_res) -> list[dict]:
        masks = [mk.apply(state.mask, mv.group, mv.tier) for mv in moves]
        # An illegal move's task score is never used, so its task stores are skipped once the general cap fails
        res = self.prober.probe_many(masks, gate=lambda r: r.kl["general"] <= self.cap(r.bpw))
        base_p = base_res.task_proxy(self.prober.stores, self.mode)
        out = []
        for mv, r in zip(moves, res):
            d_bytes = base_res.bytes - r.bytes
            cap = self.cap(r.bpw)
            legal = d_bytes > 0 and r.kl["general"] <= cap
            d_proxy = r.task_proxy(self.prober.stores, self.mode) - base_p if legal else float("nan")
            pen = 1.0 + state.penalties.get(mv.group, 0.0)
            score = (max(d_proxy, 0.0) * pen + 1e-12) / d_bytes if legal else float("inf")
            row = {"move": mv.key, "score": score, "d_proxy": d_proxy, "d_bytes": d_bytes, "kl_general": r.kl["general"],
                   "cap": cap, "bpw": r.bpw, "legal": legal, "penalty": pen - 1.0, "bytes": r.bytes}
            state.cache[mv.key] = score if legal else float("inf")
            self.record("probes", row)
            out.append({**row, "move": mv, "res": r})
        return out

    def full_reprobe(self, state: LearnerState, base_res) -> None:
        moves = legal_moves(state, self.groups, self.ladder)
        self.log(f"full re-probe: {len(moves)} moves")
        state.cache = {}
        self.evaluate(state, moves, base_res)

    def step(self, state: LearnerState, base_res):
        """One accepted move, or None if the move set is exhausted."""
        moves = legal_moves(state, self.groups, self.ladder)
        if not moves:
            return None
        exact = self.cfg.get("reprobe_every", 8) == 1
        if exact:
            cand = moves
        else:
            known = sorted((state.cache.get(mv.key, float("inf")), i) for i, mv in enumerate(moves))
            unknown = [mv for mv in moves if mv.key not in state.cache]
            cand = [moves[i] for s, i in known[: self.cfg.get("top_k", 16)] if s != float("inf")] + unknown
            cand = list({mv.key: mv for mv in cand}.values())
            if not cand:
                cand = moves
        rows = self.evaluate(state, cand, base_res)
        legal = [r for r in rows if r["legal"]]
        if not legal:
            if not exact:  # cached ranking may be stale: try everything once before declaring exhaustion
                rows = self.evaluate(state, [mv for mv in moves if mv.key not in {r['move'].key for r in rows}], base_res)
                legal = [r for r in rows if r["legal"]]
            if not legal:
                return None
        best = min(legal, key=lambda r: (r["score"], -r["d_bytes"]))
        mv = best["move"]
        prev = state.mask["groups"][mv.group]["tier"]
        state.mask = mk.apply(state.mask, mv.group, mv.tier)
        state.accepted += 1
        state.window.append({"group": mv.group, "from": prev, "to": mv.tier})
        state.cache.pop(mv.key, None)
        return best

    # ---- failure feedback (decision S4)
    def feedback(self, state: LearnerState, fb_store_name: str, base_res, revert: bool) -> dict:
        """Re-measure each move in the current window by reverting it alone, scored on the failure store.

        Moves whose revert lowers the failure-store proxy are implicated. Implicated moves get a penalty;
        if `revert`, the top implicated moves are undone and blocked.
        """
        if not state.window:
            return {"implicated": [], "reverted": []}
        cur = self.prober.probe(state.mask)
        cur_fb = cur.kl[fb_store_name] if self.mode == "preserve" else cur.nll[fb_store_name]
        trials = []
        for w in state.window:
            if state.mask["groups"][w["group"]]["tier"] != w["to"]:
                continue
            m = mk.apply(state.mask, w["group"], w["from"])
            trials.append((w, m))
        res = self.prober.probe_many([m for _, m in trials], [fb_store_name])  # the decision reads only this store
        gains = []
        for (w, _), r in zip(trials, res):
            v = r.kl[fb_store_name] if self.mode == "preserve" else r.nll[fb_store_name]
            gains.append({"group": w["group"], "from": w["from"], "to": w["to"], "gain": cur_fb - v})
        gains.sort(key=lambda x: -x["gain"])
        total = sum(max(0.0, g["gain"]) for g in gains)
        implicated = [g for g in gains if g["gain"] > 0 and g["gain"] >= self.cfg.get("implicate_share", 0.2) * total]
        lam = self.cfg.get("penalty_step", 1.0)
        for g in implicated:
            state.penalties[g["group"]] = state.penalties.get(g["group"], 0.0) + lam
        reverted = []
        if revert:
            for g in implicated[: self.cfg.get("max_revert", 2)]:
                state.mask = mk.apply(state.mask, g["group"], g["from"])
                state.blocked.add(f"{g['group']}->{g['to']}")
                reverted.append(g)
        state.window = []
        out = {"failure_store": fb_store_name, "current": cur_fb, "trials": gains, "implicated": implicated, "reverted": reverted}
        self.record("feedback", out)
        return out
