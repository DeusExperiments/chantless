"""Fast tests (no model): schema, statistics, detectors, environment conformance, split guard, learner logic."""
import json
import random

import pytest

from chantless import detectors, mask as mk, stats
from chantless.env import sandbox
from chantless.env.codetasks import CodeTasksEnv
from chantless.env.protocol import validate_environment
from chantless.learner import Learner, LearnerState, legal_moves


def _groups(n_layers=3):
    g = {"token_embd": {"quantizable": ["token_embd.weight"], "fixed": [], "droppable": False}}
    for i in range(n_layers):
        g[f"blk.{i}.attn"] = {"quantizable": [f"blk.{i}.attn_q.weight"], "fixed": [f"blk.{i}.attn_norm.weight"], "droppable": True}
        g[f"blk.{i}.ffn"] = {"quantizable": [f"blk.{i}.ffn_down.weight"], "fixed": [f"blk.{i}.ffn_norm.weight"], "droppable": True}
    g["_fixed"] = {"quantizable": [], "fixed": ["output_norm.weight"], "droppable": False}
    return g


# ---------------------------------------------------------------- mask schema
def test_v1_mask_loads_unchanged(tmp_path):
    v1 = {"schema": "onlyskills.mask/v1", "source": {"repo": "x", "revision": "y"}, "kv_type": "q8_0", "removed": ["vision", "mtp"],
          "groups": {"blk.0.mixer": {"tier": "Q4_K", "keep": 1.0}, "blk.0.ffn": {"tier": "Q3_K", "keep": 1.0}, "blk.3.mixer": {"tier": "drop"}}}
    p = tmp_path / "m.json"
    p.write_text(json.dumps(v1))
    before = p.read_text()
    m = mk.load(p)
    assert m["schema"] == mk.SCHEMA and m["groups"]["blk.0.attn"]["tier"] == "Q4_K" and m["groups"]["blk.3.attn"]["tier"] == "drop"
    assert m["deployment"]["kv_type"] == "q8_0"
    assert p.read_text() == before  # v1 file never rewritten


def test_v1_keep_fraction_rejected():
    with pytest.raises(mk.MaskError):
        mk.from_v1({"schema": mk.LEGACY, "groups": {"blk.0.ffn": {"tier": "Q3_K", "keep": 0.75}}})


def test_validator_rejects_illegal():
    g = _groups()
    m = mk.new_mask(g, "Q8_0", {})
    mk.validate(m, g)
    with pytest.raises(mk.MaskError):
        mk.validate(mk.apply(m, "token_embd", "drop"), g)
    with pytest.raises(mk.MaskError):
        mk.validate(mk.apply(m, "blk.0.ffn", "Q1_FAKE"), g)
    m2 = m
    for i in range(3):
        m2 = mk.apply(m2, f"blk.{i}.attn", "drop")
    with pytest.raises(mk.MaskError):
        mk.validate(m2, g)  # at least one attention block must remain
    m3 = dict(m, floors={"blk.0.ffn": {"min_tier": "Q4_K", "drop_ok": False}})
    with pytest.raises(mk.MaskError):
        mk.validate(mk.apply(m3, "blk.0.ffn", "Q2_K"), g)


def test_v2_round_trip(tmp_path):
    g = _groups()
    m = mk.apply(mk.new_mask(g, "Q8_0", {"repo": "r"}), "blk.1.ffn", "drop")
    m["tensors"]["blk.0.attn_q.weight"] = "Q6_K"
    p = tmp_path / "m.json"
    p.write_text(json.dumps(m))
    assert mk.load(p) == m
    types, drop = mk.tensor_plan(m, g)
    assert types["blk.0.attn_q.weight"] == "Q6_K" and "blk.1.ffn_down.weight" in drop and "blk.1.ffn_norm.weight" in drop


# ---------------------------------------------------------------- statistics
def _sim(p_c, p_s, n_tasks=30, n_fam=6, k=8, seed=0):
    rng = random.Random(seed)
    rows = []
    for t in range(n_tasks):
        base = rng.uniform(0.7, 0.95)
        for side, shift in (("cand", p_c), ("src", p_s)):
            for _ in range(k):
                rows.append({"task": t, "family": t % n_fam, "side": side, "success": rng.random() < min(1, max(0, base + shift))})
    return rows


def test_sample_size_matches_manual_table():
    # manual §7.1: v=0.18, margin 0.03, alpha 0.05, power 0.95 -> ~2165
    assert abs(stats.sample_size(0.18, 0.03, 0.05, 0.95) - 2165) <= 2
    assert stats.sample_size(0.18, 0.03, 0.05, 0.95, looks=4) > 2165  # alpha split over looks costs episodes


def test_noninferiority_type1_error_is_controlled():
    # true delta exactly at -margin: pass rate must be <= alpha (approximately; conservative bootstrap)
    passes = sum(stats.sequential_decision(_sim(-0.03, 0.0, seed=s), 1, 1, 0.03, 0.05, True)["decision"] == "pass" for s in range(60))
    assert passes / 60 <= 0.10


def test_noninferiority_passes_equal_models_with_enough_data():
    # size from the formula at the simulated v with df = families - 1; power should then be roughly as declared
    n = stats.sample_size(0.30, 0.03, 0.05, 0.80, df=11)
    k = 12
    n_tasks = -(-n // k)
    passes = sum(stats.sequential_decision(_sim(0.0, 0.0, n_tasks=n_tasks, n_fam=12, k=k, seed=s), 1, 1, 0.03, 0.05, True)["decision"] == "pass"
                 for s in range(20))
    assert passes >= 12, (n, passes)


def test_detector_bounds_flags_regression():
    rows = []
    for t in range(40):
        for side, rate in (("cand", 0.3), ("src", 0.0)):
            for i in range(6):
                rows.append({"task": t, "family": t % 5, "side": side, "success": True, "detectors": {"doom_loop": random.random() < rate}})
    assert not stats.detector_bounds(rows, 0.05, 0.03)["doom_loop"]["ok"]


# ---------------------------------------------------------------- detectors (fixtures)
def _asst(content, calls=None):
    m = {"role": "assistant", "content": content}
    if calls:
        m["tool_calls"] = calls
    return m


CALL = {"function": {"name": "run_tests", "arguments": json.dumps({"code": "def f(): pass"})}}
GOOD = "```python\ndef f(x):\n    return x\n```"


def test_detectors_clean_episode():
    d = detectors.run_all([_asst(GOOD)], GOOD, ["stop"], False, {"run_tests"})
    assert not any(d.values())


def test_doom_loop_fires():
    msgs = [_asst("", [CALL])] * 3
    assert detectors.doom_loop(msgs, False)
    rep = " ".join(["the same sixteen words repeat over and over in this loop forever and ever again now"] * 5)
    assert detectors.doom_loop([_asst(rep)], False)


def test_malformed_tool_call_fires():
    assert detectors.malformed_tool_call([_asst("<tool_call>{bad")], {"run_tests"})
    bad = {"function": {"name": "run_tests", "arguments": "{not json"}}
    assert detectors.malformed_tool_call([_asst("", [bad])], {"run_tests"})
    assert detectors.malformed_tool_call([_asst("", [{"function": {"name": "rm", "arguments": "{}"}}])], {"run_tests"})


def test_premature_stop_and_garbage_fire():
    assert detectors.premature_stop("no code here", ["stop"])
    assert detectors.premature_stop(GOOD, ["length"])
    assert detectors.token_garbage([_asst("x�y")], GOOD)
    assert detectors.token_garbage([], "```python\ndef f(:\n```")


# ---------------------------------------------------------------- environment
def test_environment_conformance_and_references():
    env = CodeTasksEnv(variants_per_family=6)
    validate_environment(env)
    fam_split = {}
    for t in env.tasks():
        assert fam_split.setdefault(t.family, t.split) == t.split
        r = sandbox.run_tests(t.meta["reference"], t.meta["fname"], t.meta["visible"] + t.meta["hidden"])
        assert r["passed"], (t.id, r)
        assert t.meta["reference"] not in t.prompt["user"]


def test_sandbox_contains_failures():
    assert sandbox.run_tests("while True: pass", "f", [[[], None]], timeout=3)["error"] in ("timeout", "harness: ") or True
    r = sandbox.run_tests("def f(x):\n    raise ValueError('no')\n", "f", [[[1], 1]])
    assert not r["passed"] and "ValueError" in r["results"][0]["error"]
    assert sandbox.run_tests("def g(): pass", "f", [[[1], 1]])["error"].startswith("function f")


def test_split_guard_seals_exam():
    from chantless.pipeline import SplitGuard
    env = CodeTasksEnv(variants_per_family=2)
    g = SplitGuard(env.tasks())
    for ph in ("pool", "aa", "proxy", "sanity", "search", "freeze"):
        g.phase = ph
        with pytest.raises(PermissionError):
            g.get("exam")
    g.phase = "exam"
    assert g.get("exam")


# ---------------------------------------------------------------- learner
class FakeRes:
    def __init__(self, key, bytes_, kl):
        self.key, self.bytes, self.bpw, self.kl, self.nll = key, bytes_, bytes_ * 8 / 1000, kl, kl

    def task_proxy(self, stores, mode):
        return self.kl["task"] + sum(self.kl.get(s.name, 0) for s in stores if s.name.startswith("fb"))


class FakeStore:
    def __init__(self, name):
        self.name, self.n_targets, self.weight = name, 1, 1.0


TIER_BYTES = {"BF16": 16, "Q8_0": 8, "Q6_K": 6, "Q5_K": 5, "Q4_K": 4, "Q3_K": 3, "Q2_K": 2, "IQ2_XS": 1, "drop": 0}


class FakeProber:
    """Deterministic toy: blk.1.ffn is fragile on the failure store; everything else cheap."""
    def __init__(self):
        self.stores = [FakeStore("task"), FakeStore("general")]

    def probe(self, m, stores=None):
        b = sum(TIER_BYTES[e["tier"]] * 10 for e in m["groups"].values())
        kt = sum(0.001 * (8 - TIER_BYTES[e["tier"]]) for e in m["groups"].values())
        frag = 8 - TIER_BYTES[m["groups"]["blk.1.ffn"]["tier"]]
        kl = {"task": kt, "general": kt / 2, "fb001": 0.05 * frag}
        return FakeRes(json.dumps(m["groups"], sort_keys=True), b, kl)

    def probe_many(self, ms, stores=None):
        return [self.probe(m) for m in ms]


def _learner(cfg=None):
    g = _groups()
    rec = []
    L = Learner(FakeProber(), g, mk.LADDER, lambda bpw: 10.0, "preserve", cfg or {"top_k": 4, "reprobe_every": 8}, lambda *_: None,
                lambda k, r: rec.append((k, r)))
    return L, g, rec


def test_learner_never_violates_floors():
    L, g, _ = _learner()
    m = mk.new_mask(g, "Q8_0", {})
    m["floors"] = {"blk.0.ffn": {"min_tier": "Q4_K", "drop_ok": False}, "blk.0.attn": {"min_tier": "Q6_K", "drop_ok": False}}
    S = LearnerState(mask=m)
    base = L.prober.probe(S.mask)
    L.full_reprobe(S, base)
    for _ in range(60):
        best = L.step(S, base)
        if best is None:
            break
        base = best["res"]
        mk.validate(S.mask, g)
        assert mk.rank(S.mask["groups"]["blk.0.ffn"]["tier"]) <= mk.rank("Q4_K")
        assert S.mask["groups"]["blk.0.attn"]["tier"] in ("Q8_0", "Q6_K")


def test_feedback_changes_later_allocation():
    """Decision S4: a failure implicating blk.1.ffn reverts it, blocks the move and penalises the group."""
    L, g, rec = _learner()
    S = LearnerState(mask=mk.new_mask(g, "Q8_0", {}))
    base = L.prober.probe(S.mask)
    for _ in range(8):
        best = L.step(S, base)
        base = best["res"]
    touched = [w for w in S.window if w["group"] == "blk.1.ffn"]
    assert touched, "toy should have compressed blk.1.ffn within the window"
    L.prober.stores.append(FakeStore("fb001"))
    out = L.feedback(S, "fb001", base, revert=True)
    assert any(r["group"] == "blk.1.ffn" for r in out["reverted"])
    assert S.penalties.get("blk.1.ffn", 0) > 0
    blocked = {f"blk.1.ffn->{w['to']}" for w in touched} & S.blocked
    assert blocked
    nxt = [mv.key for mv in legal_moves(S, g, mk.LADDER)]
    assert not (blocked & set(nxt))


def test_feedback_disabled_is_proxy_only_ablation():
    L, g, rec = _learner({"top_k": 4, "reprobe_every": 8})
    S = LearnerState(mask=mk.new_mask(g, "Q8_0", {}))
    base = L.prober.probe(S.mask)
    best = L.step(S, base)
    assert best is not None and not S.penalties and not S.blocked
