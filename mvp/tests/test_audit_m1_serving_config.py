"""Deterministic offline regression tests for Milestone 1 (client, config, runtime, serve, evaluator, cli)."""
from __future__ import annotations

import json
import logging
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from pydantic import ValidationError

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "mvp"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from chantless import cli as climod, config as cfgmod, evaluator as evmod, runtime, serve
from chantless.client import ChatClient, InfraError
from chantless.env.protocol import Episode, Task
from test_stage0_cli_and_plan import _write_minimal_gguf


def _make_loopback_server(handler_fn):
    records: list[dict] = []
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length) if length else b"{}"
            body = json.loads(raw.decode("utf-8"))
            with lock:
                idx = len(records)
                records.append({"path": self.path, "body": body})
            res = handler_fn(self, self.path, body, idx)
            if res is None:
                return
            status, payload = res
            data = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, format, *args):
            return

    srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    return srv, th, f"http://127.0.0.1:{srv.server_port}", records


# ------------------------------------------------------------------ 1. client.py
def test_client_seed_max_tokens_precedence_and_cache_prompt():
    def handler(req_h, path, body, idx):
        if path == "/v1/chat/completions":
            return 200, {
                "choices": [{"message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
                "usage": {"completion_tokens": 2},
            }
        return 200, {
            "choices": [{"text": "ok", "finish_reason": "stop"}],
            "usage": {"completion_tokens": 2},
        }

    srv, th, url, recs = _make_loopback_server(handler)
    try:
        decoding = {
            "temperature": 0.2,
            "seed": 999,
            "max_tokens": 999,
            "reasoning_effort": "low",
            "enable_thinking": True,
        }
        cl = ChatClient(url, decoding)
        cl.chat([{"role": "user", "content": "hi"}], seed=7, max_tokens=64)
        chat_body = recs[-1]["body"]
        assert chat_body["seed"] == 7
        assert chat_body["max_tokens"] == 64
        assert chat_body["temperature"] == 0.2
        assert chat_body["cache_prompt"] is False
        assert chat_body["chat_template_kwargs"] == {"reasoning_effort": "low"}
        for skipped in ("reasoning_effort", "enable_thinking"):
            assert skipped not in chat_body

        cl.complete("hello", seed=8, max_tokens=32)
        comp_body = recs[-1]["body"]
        assert comp_body["seed"] == 8
        assert comp_body["max_tokens"] == 32
        assert comp_body["temperature"] == 0.2
        assert comp_body["cache_prompt"] is False
        for skipped in ("reasoning_effort", "enable_thinking", "chat_template_kwargs"):
            assert skipped not in comp_body

        # Explicit cache_prompt=True is respected, and per-call cache_prompt=False overrides it
        cl_cached = ChatClient(url, {"temperature": 0.0}, cache_prompt=True)
        cl_cached.chat([{"role": "user", "content": "hi"}], seed=1, max_tokens=8)
        assert recs[-1]["body"].get("cache_prompt") is not False
        cl_cached.chat([{"role": "user", "content": "hi"}], seed=1, max_tokens=8, cache_prompt=False)
        assert recs[-1]["body"]["cache_prompt"] is False

        # decoding["cache_prompt"] = False propagates when init arg is omitted
        cl_dec_off = ChatClient(url, {"temperature": 0.0, "cache_prompt": False})
        cl_dec_off.complete("p", seed=1, max_tokens=8)
        assert recs[-1]["body"]["cache_prompt"] is False
    finally:
        srv.shutdown()
        srv.server_close()
        th.join(timeout=2)


def test_client_retries_remote_disconnected_and_empty_choices(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr("chantless.client.time.sleep", lambda _s: None)
    mode = {"kind": "transient"}

    def handler(req_h, path, body, idx):
        if mode["kind"] == "transient":
            attempt = idx % 3
            if attempt == 0:
                # Close socket without writing HTTP status/body -> RemoteDisconnected (HTTPException)
                req_h.connection.close()
                return None
            if attempt == 1:
                return 200, {"choices": []}
            if path == "/v1/chat/completions":
                return 200, {"choices": [{"message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}]}
            return 200, {"choices": [{"text": "ok", "finish_reason": "stop"}]}
        if mode["kind"] == "always_disconnect":
            req_h.connection.close()
            return None
        return 200, {"choices": []}

    srv, th, url, recs = _make_loopback_server(handler)
    try:
        cl = ChatClient(url, {"temperature": 0.0}, retries=3)
        res_chat = cl.chat([{"role": "user", "content": "hi"}], seed=1, max_tokens=8)
        assert res_chat["message"]["content"] == "ok"
        res_comp = cl.complete("hi", seed=2, max_tokens=8)
        assert res_comp["text"] == "ok"
        assert len(recs) == 6

        cl_fail = ChatClient(url, {"temperature": 0.0}, retries=2)
        mode["kind"] = "always_disconnect"
        with pytest.raises(InfraError):
            cl_fail.chat([{"role": "user", "content": "hi"}], seed=1, max_tokens=8)
        with pytest.raises(InfraError):
            cl_fail.complete("hi", seed=1, max_tokens=8)

        mode["kind"] = "always_empty"
        with pytest.raises(InfraError):
            cl_fail.chat([{"role": "user", "content": "hi"}], seed=1, max_tokens=8)
        with pytest.raises(InfraError):
            cl_fail.complete("hi", seed=1, max_tokens=8)
    finally:
        srv.shutdown()
        srv.server_close()
        th.join(timeout=2)


def test_client_retries_null_choices_and_non_serializable_fails_fast(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr("chantless.client.time.sleep", lambda _s: None)
    mode = {"kind": "transient_null"}

    def handler(req_h, path, body, idx):
        if mode["kind"] == "transient_null" and idx % 2 == 0:
            return 200, {"choices": None}
        if mode["kind"] == "always_null":
            return 200, {"choices": None}
        if path == "/v1/chat/completions":
            return 200, {"choices": [{"message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}]}
        return 200, {"choices": [{"text": "ok", "finish_reason": "stop"}]}

    srv, th, url, recs = _make_loopback_server(handler)
    try:
        cl = ChatClient(url, {"temperature": 0.0}, retries=2)
        res_chat = cl.chat([{"role": "user", "content": "hi"}], seed=1, max_tokens=8)
        assert res_chat["message"]["content"] == "ok"
        res_comp = cl.complete("hi", seed=2, max_tokens=8)
        assert res_comp["text"] == "ok"
        assert len(recs) == 4

        mode["kind"] = "always_null"
        before_chat = len(recs)
        with pytest.raises(InfraError):
            cl.chat([{"role": "user", "content": "hi"}], seed=1, max_tokens=8)
        assert len(recs) - before_chat == 2

        before_comp = len(recs)
        with pytest.raises(InfraError):
            cl.complete("hi", seed=1, max_tokens=8)
        assert len(recs) - before_comp == 2

        # Non-serializable inputs fail fast during json.dumps(body) with zero HTTP requests
        before_bad = len(recs)
        with pytest.raises(TypeError):
            cl.chat([{"role": "user", "content": object()}], seed=1, max_tokens=8)
        with pytest.raises(TypeError):
            cl.complete(object(), seed=1, max_tokens=8)  # type: ignore[arg-type]
        assert len(recs) == before_bad
    finally:
        srv.shutdown()
        srv.server_close()
        th.join(timeout=2)


# ------------------------------------------------------------------ 2. config.py
def test_config_serving_decoding_and_schema_validation():
    sc = cfgmod.ServingConfig()
    assert sc.cache_prompt is False
    assert sc.flash_attn == "on"
    assert sc.special is None
    assert cfgmod.ServingConfig(flash_attn="off", special=True).flash_attn == "off"
    with pytest.raises(ValidationError):
        cfgmod.ServingConfig(flash_attn="invalid")

    dc = cfgmod.DecodingConfig(
        chat_template_kwargs={"reasoning_effort": "low"},
        enable_thinking=False,
        reasoning_effort="none",
    )
    assert dc.reasoning_effort == "none"
    assert cfgmod.DecodingConfig(reasoning_effort=0.25).reasoning_effort == 0.25
    for bad_re in (True, False, -0.1, float("nan"), float("inf"), "  "):
        with pytest.raises(ValidationError):
            cfgmod.DecodingConfig(reasoning_effort=bad_re)

    # SanityConfig & ProbeConfig
    assert cfgmod.SanityConfig().cap == "off"
    assert cfgmod.SanityConfig(cap="general_kl", floor_key="kl_gen").cap == "general_kl"
    for bad_cap in (True, False, 0.5, None, "unknown"):
        with pytest.raises(ValidationError):
            cfgmod.SanityConfig(cap=bad_cap)
    with pytest.raises(ValidationError):
        cfgmod.SanityConfig(floor_key="bogus")
    with pytest.raises(ValidationError):
        cfgmod.SanityConfig(solo_task_tiers=["BOGUS"])
    with pytest.raises(ValidationError):
        cfgmod.ProbeConfig(solo_task_tiers=["BOGUS"])
    assert cfgmod.ProbeConfig(want_solo=["general", "task"]).want_solo == ["general", "task"]
    for bad_ws in ([], ["bogus"], ["task", "task"]):
        with pytest.raises(ValidationError):
            cfgmod.ProbeConfig(want_solo=bad_ws)

    # LearnerConfig preserves standalone oneshot without target_bytes (test_stage0_cli_and_plan.py:217 parity)
    lc_standalone = cfgmod.LearnerConfig(algo="oneshot", vocab_k="auto", experts=True, expert_counts=[32, 48, 64])
    assert lc_standalone.algo == "oneshot"
    lc_ok = cfgmod.LearnerConfig(
        oneshot_tiers=["Q8_0", "Q4_K"],
        group_tiers={"blk.0.ffn": ["Q4_K"]},
        width_step=256,
        allow_drop=False,
        enforce_general_cap=False,
    )
    assert lc_ok.allow_drop is False
    for bad_lc in (
        {"oneshot_tiers": ["BOGUS"]},
        {"oneshot_tiers": []},
        {"group_tiers": {"blk.0.ffn": ["BOGUS"]}},
        {"group_tiers": {"blk.0.ffn": []}},
        {"width_step": 0},
    ):
        with pytest.raises(ValidationError):
            cfgmod.LearnerConfig(**bad_lc)

    # ExtractConfig cross-field validation
    base = {
        "run_id": "test",
        "run_root": "/tmp/runs",
        "source": {"gguf": "m.gguf", "repo": "org/m", "revision": "rev"},
        "toolchain": {"llama_bin": "/bin", "traj_kl": "/bin/true"},
        "env": {"module": "chantless.env.bfcl"},
        "ladder": ["BF16", "Q8_0", "Q4_K", "Q2_K"],
        "start_tier": "Q4_K",
    }
    assert cfgmod.ExtractConfig.model_validate(base).serving.cache_prompt is False

    # oneshot in ExtractConfig requires target_bytes
    with pytest.raises(ValidationError):
        cfgmod.ExtractConfig.model_validate({**base, "learner": {"algo": "oneshot"}})

    # floor_tier, solo_tiers, solo_task_tiers, oneshot_tiers, group_tiers must be in ladder
    for bad_patch in (
        {"proxy": {"floor_tier": "IQ4_NL"}},
        {"proxy": {"floor_tier": "BF16"}},
        {"proxy": {"floor_tier": "drop"}},
        {"sanity": {"solo_tiers": ["IQ4_NL"]}},
        {"sanity": {"solo_task_tiers": ["IQ4_NL"]}},
        {"probe": {"solo_task_tiers": ["IQ4_NL"]}},
        {"learner": {"algo": "oneshot", "target_bytes": 1000, "oneshot_tiers": ["IQ4_NL"]}},
        {"learner": {"group_tiers": {"blk.0.ffn": ["IQ4_NL"]}}},
        {"sanity": {"allow_drop": True}, "learner": {"allow_drop": False}},
        {"sanity": {"enforce_general_cap": False}, "learner": {"enforce_general_cap": True}},
        {"learner": {"algo": "oneshot", "target_bytes": 1000}, "probe": {"want_solo": ["general"]}},
    ):
        with pytest.raises(ValidationError):
            cfgmod.ExtractConfig.model_validate({**base, **bad_patch})

    # Reconciling single-sided allow_drop / enforce_general_cap
    ec_from_sanity = cfgmod.ExtractConfig.model_validate({**base, "sanity": {"allow_drop": False}})
    assert ec_from_sanity.learner.allow_drop is False
    ec_from_learner = cfgmod.ExtractConfig.model_validate({**base, "learner": {"enforce_general_cap": False}})
    assert ec_from_learner.sanity.enforce_general_cap is False

    # _parse_override_value preserves "on"/"off" strings for --set
    ov = cfgmod.apply_overrides(base, ["serving.flash_attn=off", "sanity.cap=off"])
    assert ov["serving"]["flash_attn"] == "off"
    assert ov["sanity"]["cap"] == "off"
    ec_ov = cfgmod.ExtractConfig.model_validate(ov)
    assert ec_ov.serving.flash_attn == "off"
    assert ec_ov.sanity.cap == "off"

    # export_json_schema("all") includes "anneal"
    all_schemas = cfgmod.export_json_schema("all")
    assert "anneal" in all_schemas


def test_all_six_repo_configs_validate_with_zero_trust_defaults():
    cfg_dir = ROOT / "mvp" / "configs"
    extract_files = (
        "extract-example.yaml",
        "inkling2-mini-bfcl.yaml",
        "inkling2-mini-clsbench.yaml",
        "inkling2-mini-deepmath.yaml",
        "local-smoke.json",
    )
    for name in extract_files:
        cfg = cfgmod.load_config(cfg_dir / name, "extract")
        assert cfg.serving.cache_prompt is False, f"{name} must default cache_prompt=False"
    learn_cfg = cfgmod.load_config(cfg_dir / "learn-example.yaml", "learn")
    assert learn_cfg.learn.method == "gates_seq"


# ------------------------------------------------------------------ 3. runtime.py & serve.py
def test_runtime_llama_server_defaults_and_cmd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    tc = runtime.Toolchain(llama_bin=tmp_path, traj_kl=tmp_path / "traj-kl", gpus=[""])
    srv = runtime.LlamaServer(tc, tmp_path / "m.gguf")
    assert srv.cache_prompt is False
    assert srv.flash_attn == "on"
    assert srv.special is False

    seen_cmds: list[list[str]] = []

    class FakePopen:
        def __init__(self, cmd, **kw):
            seen_cmds.append(cmd)
            self.pid = 12345

        def poll(self):
            return None

        def terminate(self):
            return None

        def wait(self, timeout=None):
            return 0

    class FakeResp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return b'{"status": "ok"}'

    monkeypatch.setattr(runtime.subprocess, "Popen", FakePopen)
    monkeypatch.setattr(runtime.urllib.request, "urlopen", lambda *a, **kw: FakeResp())

    with runtime.LlamaServer(tc, tmp_path / "m.gguf", flash_attn="off", extra=["--ctx-check"]) as s:
        assert s.cache_prompt is False
    cmd = seen_cmds[-1]
    assert "--no-cache-prompt" in cmd
    assert cmd[cmd.index("-fa") + 1] == "off"
    assert "--ctx-check" in cmd


def test_serve_live_bench_defaults_and_resolve_special_warning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
):
    captured_init: list[dict] = []

    class FakeLlamaServer:
        def __init__(self, tc, model, **kw):
            captured_init.append(kw)
            self.url = "http://127.0.0.1:1"
            self.proc = SimpleNamespace(pid=-1, poll=lambda: 0)

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def post(self, path, body):
            return {
                "timings": {"prompt_n": 10, "prompt_ms": 5.0, "predicted_n": 5, "predicted_ms": 10.0}
            }

    monkeypatch.setattr(serve, "LlamaServer", FakeLlamaServer)
    m = tmp_path / "plain.gguf"
    _write_minimal_gguf(m, "qwen35")
    minfo = {"model_path": str(m), "bytes": 100, "sha256": "0", "cache_hit": True, "cache_key": "k", "build_s": 0.0}

    serve.serve_live(tmp_path, minfo)
    assert captured_init[-1]["cache_prompt"] is False
    assert captured_init[-1]["flash_attn"] == "on"

    serve.serve_live(tmp_path, minfo, cache_prompt=True, flash_attn="off")
    assert captured_init[-1]["cache_prompt"] is True
    assert captured_init[-1]["flash_attn"] == "off"

    b_res = serve.bench_one(tmp_path, minfo, runs=1, flash_attn="off", cache_prompt=True)
    assert captured_init[-1]["flash_attn"] == "off"
    assert captured_init[-1]["cache_prompt"] is True
    assert b_res["config"]["flash_attn"] == "off"
    assert b_res["config"]["cache_prompt"] is True

    corrupt = tmp_path / "corrupt.gguf"
    corrupt.write_bytes(b"NOTGGUF!12345678")
    with caplog.at_level(logging.WARNING, logger="chantless.serve"):
        assert serve._resolve_special(corrupt, None) is False
    assert any("special" in rec.message.lower() or "corrupt.gguf" in rec.message for rec in caplog.records)


# ------------------------------------------------------------------ 4. evaluator.py
def test_evaluator_defaults_special_fallback_and_dialect_decoupling(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    ink_gguf = tmp_path / "ink.gguf"
    _write_minimal_gguf(ink_gguf, "inkling2_loop")
    qwen_gguf = tmp_path / "qwen.gguf"
    _write_minimal_gguf(qwen_gguf, "qwen35")

    seen_srv_kw: list[dict] = []

    class FakeLlamaServer:
        def __init__(self, tc, model, **kw):
            seen_srv_kw.append(kw)
            self.url = "http://127.0.0.1:1"
            self.special = bool(kw.get("special", False))

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(evmod, "LlamaServer", FakeLlamaServer)

    class StubEnv:
        classifier = False

        def run(self, task, client, seed):
            res = client.chat([{"role": "user", "content": "hi"}], seed=seed, max_tokens=8)
            return Episode(task.id, seed, [], None, res["message"]["content"], [res["finish_reason"]], 1, 1, 0.01)

        def verify(self, task, ep):
            return SimpleNamespace(success=True, tests_passed=True, detectors={}, infra_error=None, details={})

    tc = runtime.Toolchain(llama_bin=tmp_path, traj_kl=tmp_path / "traj-kl", gpus=[""])
    ev_def = evmod.Evaluator(
        tc, StubEnv(), {"temperature": 0.0}, 1, 1024, "f16", tmp_path / "l1",
        flash_attn="off", extra=["--log-disable"],
    )
    assert ev_def.cache_prompt is False
    with ev_def.servers(str(ink_gguf)) as srvs:
        assert srvs[0].special is True
    assert seen_srv_kw[-1]["cache_prompt"] is False
    assert seen_srv_kw[-1]["special"] is True
    assert seen_srv_kw[-1]["flash_attn"] == "off"
    assert seen_srv_kw[-1]["extra"] == ["--log-disable"]

    # Explicit special=True vs special=None across real GGUFs and mock non-file model
    srv, th, url, recs = _make_loopback_server(
        lambda req_h, path, body, idx: (
            200,
            {"choices": [{"message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}]},
        )
    )
    try:
        mock_srvs = [SimpleNamespace(url=url, special=True)]
        t = Task(id="t0", family="f0", split="select", prompt={})
        ev_sp = evmod.Evaluator(tc, StubEnv(), {"temperature": 0.0}, 1, 1024, "f16", tmp_path / "l2", special=True)

        # 1. special=True + real ink_gguf -> dialect="inkling2" (reasoning_effort: "none")
        assert ev_sp._resolve_dialect(str(ink_gguf), mock_srvs) == "inkling2"
        ev_sp.run(str(ink_gguf), [(t, 1)], "tag_ink", save=False, srvs=mock_srvs)
        assert recs[-1]["body"]["cache_prompt"] is False
        assert recs[-1]["body"]["chat_template_kwargs"] == {"reasoning_effort": "none"}

        # 2. special=True + real qwen_gguf -> dialect=None (enable_thinking: False)
        assert ev_sp._resolve_dialect(str(qwen_gguf), mock_srvs) is None
        ev_sp.run(str(qwen_gguf), [(t, 1)], "tag_qwen", save=False, srvs=mock_srvs)
        assert recs[-1]["body"]["cache_prompt"] is False
        assert recs[-1]["body"]["chat_template_kwargs"] == {"enable_thinking": False}

        # 3. special=True + non-file mock "m" + srvs.special=True -> dialect=None (enable_thinking: False)
        assert ev_sp._resolve_dialect("m", mock_srvs) is None
        ev_sp.run("m", [(t, 1)], "tag_mock_sp", save=False, srvs=mock_srvs)
        assert recs[-1]["body"]["chat_template_kwargs"] == {"enable_thinking": False}

        # 4. special=None + non-file mock "m" + srvs.special=True -> dialect="inkling2" (preserves stage0 heuristic)
        assert ev_def._resolve_dialect("m", mock_srvs) == "inkling2"
        ev_def.run("m", [(t, 1)], "tag_mock_def", save=False, srvs=mock_srvs)
        assert recs[-1]["body"]["chat_template_kwargs"] == {"reasoning_effort": "none"}
    finally:
        srv.shutdown()
        srv.server_close()
        th.join(timeout=2)


def test_evaluator_rule_cache_invalidation_and_finite_scores(tmp_path: Path):
    model = tmp_path / "m.gguf"
    _write_minimal_gguf(model, "qwen35")
    tc = runtime.Toolchain(llama_bin=tmp_path, traj_kl=tmp_path / "traj-kl", gpus=[""])
    cal_tasks = [
        Task(id="c0", family="f0", split="calibration", prompt={"label": "human"}),
        Task(id="c1", family="f1", split="calibration", prompt={"label": "ai"}),
    ]

    class FakeClsEnv:
        name = "clsbench"
        version = "1"
        classifier = True

        def __init__(self, fpr_target=0.05, pair=(10, 11), unflagged="human", flagged="ai", dialect=None):
            self.fpr_target = fpr_target
            self.pair = pair
            self.unflagged_word = unflagged
            self.flagged_word = flagged
            if dialect is not None:
                self.dialect = dialect
            self.calibrate_calls = 0

        def tasks(self):
            return cal_tasks

        def score(self, tasks, model_path, scorer):
            # Lower score = flagged (ai), higher score = unflagged (human): score = logit[unflagged] - logit[flagged]
            return {t.id: (2.0 if t.prompt["label"] == "human" else -2.0) for t in tasks}

        def calibrate(self, scores):
            self.calibrate_calls += 1
            return {"fpr_target": self.fpr_target, "pair": list(self.pair), "calls": self.calibrate_calls}

        def decide(self, task, score, rule):
            flagged = score < 0.0
            is_flagged_label = task.prompt["label"] == "ai"
            return SimpleNamespace(
                success=(flagged == is_flagged_label),
                tests_passed=(flagged == is_flagged_label),
                detectors={},
                infra_error=None,
                details={"score": score, "flagged": flagged},
            )

    log_dir = tmp_path / "eval_logs"
    dummy_scorer = SimpleNamespace()
    env0 = FakeClsEnv(fpr_target=0.05, pair=(10, 11))
    ev0 = evmod.Evaluator(tc, env0, {"temperature": 0.0}, 1, 1024, "f16", log_dir, scorer=dummy_scorer)
    r0 = ev0.rule(str(model))
    assert r0["fpr_target"] == 0.05
    assert env0.calibrate_calls == 1

    # Varying each axis on the same log_dir + model + cal IDs must invalidate disk cache & recompute
    for idx, (env_kw, ev_kw) in enumerate([
        ({"fpr_target": 0.01, "pair": (10, 11)}, {"decoding": {"temperature": 0.0}}),
        ({"fpr_target": 0.01, "pair": (20, 21)}, {"decoding": {"temperature": 0.0}}),
        ({"fpr_target": 0.01, "pair": (20, 21), "unflagged": "hum", "flagged": "bot"}, {"decoding": {"temperature": 0.0}}),
        ({"fpr_target": 0.01, "pair": (20, 21), "unflagged": "hum", "flagged": "bot"}, {"decoding": {"temperature": 0.7}}),
        ({"fpr_target": 0.01, "pair": (20, 21), "unflagged": "hum", "flagged": "bot", "dialect": "inkling2"}, {"decoding": {"temperature": 0.7}}),
        ({"fpr_target": 0.01, "pair": (20, 21), "unflagged": "hum", "flagged": "bot", "dialect": "inkling2"}, {"decoding": {"temperature": 0.7}, "special": False}),
    ], start=1):
        env_i = FakeClsEnv(**env_kw)
        ev_i = evmod.Evaluator(
            tc, env_i, ev_kw["decoding"], 1, 1024, "f16", log_dir,
            scorer=dummy_scorer, special=ev_kw.get("special"),
        )
        r_i = ev_i.rule(str(model))
        assert env_i.calibrate_calls == 1, f"case {idx} hit stale rule cache"
        assert r_i["fpr_target"] == env_kw["fpr_target"]
        assert r_i["pair"] == list(env_kw["pair"])

    # seed() and _score() reject non-finite scores immediately
    for bad_val in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError):
            ev0.seed(str(model), {"c0": bad_val})

    class NanEnv(FakeClsEnv):
        def score(self, tasks, model_path, scorer):
            return {t.id: float("nan") for t in tasks}

    ev_nan = evmod.Evaluator(tc, NanEnv(), {"temperature": 0.0}, 1, 1024, "f16", tmp_path / "nan_logs", scorer=dummy_scorer)
    with pytest.raises(RuntimeError):
        ev_nan._score(str(model), cal_tasks)

    # Classifier orientation check: unflagged > flagged (no 1 - AUROC inversion)
    rows = ev0.run(str(model), [(cal_tasks[0], 0), (cal_tasks[1], 0)], "cls_orient", save=False)
    assert all(r["success"] is True for r in rows)


# ------------------------------------------------------------------ 5. cli.py
def test_cli_serve_bench_flags_and_schema_anneal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]):
    qwen_plain = tmp_path / "qwen.gguf"
    _write_minimal_gguf(qwen_plain, "qwen35")
    minfo = {"model_path": str(qwen_plain), "bytes": 100, "sha256": "0", "cache_hit": True, "cache_key": "k", "build_s": 0.0}
    captured: list[dict] = []
    monkeypatch.setattr(serve, "resolve_model", lambda *a, **kw: minfo)
    monkeypatch.setattr(serve, "serve_live", lambda *a, **kw: captured.append(kw) or {"ok": True})
    monkeypatch.setattr(serve, "run_bench", lambda *a, **kw: captured.append(kw) or {"ok": True})

    cases = [
        (["chantless", "serve", "--model", str(qwen_plain), "--llama-bin", str(tmp_path)], None, False, "on"),
        (["chantless", "serve", "--model", str(qwen_plain), "--llama-bin", str(tmp_path), "--no-special", "--no-cache-prompt", "--flash-attn", "off"], False, False, "off"),
        (["chantless", "serve", "--model", str(qwen_plain), "--llama-bin", str(tmp_path), "--cache-prompt"], None, True, "on"),
        (["chantless", "bench", "--model", str(qwen_plain), "--llama-bin", str(tmp_path), "--no-special", "--cache-prompt", "--flash-attn", "auto"], False, True, "auto"),
    ]
    for argv, exp_sp, exp_cp, exp_fa in cases:
        monkeypatch.setattr(sys, "argv", argv)
        climod.main()
        assert captured[-1]["special"] is exp_sp
        assert captured[-1]["cache_prompt"] is exp_cp
        assert captured[-1]["flash_attn"] == exp_fa
    capsys.readouterr()

    # schema anneal and schema all
    monkeypatch.setattr(sys, "argv", ["chantless", "schema", "anneal"])
    climod.main()
    anneal_schema = json.loads(capsys.readouterr().out)
    assert "anneal" in anneal_schema.get("properties", {})

    monkeypatch.setattr(sys, "argv", ["chantless", "schema", "all"])
    climod.main()
    all_schemas = json.loads(capsys.readouterr().out)
    assert "anneal" in all_schemas


def test_cli_extract_run_cache_prompt_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    cfg_path = tmp_path / "extract.json"
    cfg_path.write_text(json.dumps({
        "run_id": "cli_cp_test",
        "run_root": str(tmp_path / "runs"),
        "source": {"gguf": "m.gguf", "repo": "org/m", "revision": "rev"},
        "toolchain": {"llama_bin": "/bin", "traj_kl": "/bin/true"},
        "env": {"module": "chantless.env.bfcl"},
        "ladder": ["BF16", "Q8_0", "Q4_K", "Q2_K"],
        "start_tier": "Q4_K",
    }), encoding="utf-8")

    captured_cfgs: list[dict] = []

    class FakeRun:
        def __init__(self, cfg: dict):
            captured_cfgs.append(cfg)
            self.dir = tmp_path / "runs" / "cli_cp_test"

        def run_phases(self, upto=None):
            return None

    monkeypatch.setattr(climod, "Run", FakeRun)
    monkeypatch.setattr(climod.signal, "signal", lambda *_a, **_kw: None)

    cases = [
        (["chantless", "extract", "--config", str(cfg_path)], False),
        (["chantless", "extract", "--config", str(cfg_path), "--cache-prompt"], True),
        (["chantless", "extract", "--config", str(cfg_path), "--set", "serving.cache_prompt=true", "--no-cache-prompt"], False),
        (["chantless", "run", "--config", str(cfg_path), "--set", "serving.cache_prompt=true", "--no-cache-prompt"], False),
    ]
    for argv, expected_cp in cases:
        monkeypatch.setattr(sys, "argv", argv)
        climod.main()
        assert captured_cfgs[-1]["serving"]["cache_prompt"] is expected_cp
