"""Milestone 5 deterministic offline regression tests for Tier 5 hardening (F1-F12 & Gaps 1-7)."""
from __future__ import annotations

import json
import math
import os
import shutil
import stat
import struct
import subprocess
import sys
import threading
import typing
from contextlib import nullcontext
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from pydantic import ValidationError
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "mvp") not in sys.path:
    sys.path.insert(0, str(ROOT / "mvp"))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from chantless import (  # noqa: E402
    cli as climod,
    compare as cmpmod,
    config as cfgmod,
    detectors,
    evaluator as evmod,
    kl,
    learn,
    pipeline,
    runtime,
    sanity,
    stats,
)
from chantless.client import ChatClient, InfraError  # noqa: E402
from chantless.env import clsbench, codetasks, deepmath, detect  # noqa: E402
from chantless.env.protocol import Task  # noqa: E402
from test_audit_m4_pipeline_tools_configs import _make_hf_dir, _write_gguf  # noqa: E402
from test_stage0_cli_and_plan import _write_minimal_gguf  # noqa: E402
from test_stage0_renderer_and_envs import _build_inkling2_hf_dir  # noqa: E402


def _make_raw_http_server(responder_fn):
    """Loopback HTTP server that can return arbitrary raw bytes & status codes."""
    calls: list[dict[str, Any]] = []
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length) if length else b"{}"
            body = json.loads(raw.decode("utf-8"))
            with lock:
                idx = len(calls)
                calls.append({"path": self.path, "body": body})
            status, payload_bytes = responder_fn(self.path, body, idx)
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload_bytes)))
            self.end_headers()
            self.wfile.write(payload_bytes)

        def log_message(self, format, *args):
            return

    srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    return srv, th, f"http://127.0.0.1:{srv.server_port}", calls


def _make_cls_jsonl(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return path


def _make_detect_data_dir(tmp_path: Path) -> tuple[Path, Path]:
    d = tmp_path / "detect_data"
    wo = d / "WildOutlines" / "data"
    be = d / "beemo" / "data"
    wo.mkdir(parents=True)
    be.mkdir(parents=True)
    val_rows = [
        {"id": f"v{i}", "text": f"alpha beta gamma delta {i}", "label": ("human" if i % 2 == 0 else "ai"), "format": "essay"}
        for i in range(8)
    ]
    test_rows = [
        {"id": f"t{i}", "text": f"test doc {i}", "label": ("human" if i % 2 == 0 else "ai"), "format": "essay"}
        for i in range(4)
    ]
    cal_rows = [
        {"id": f"c{i}", "text": f"cal doc {i}", "label": "human", "format": "essay"}
        for i in range(4)
    ]
    beemo_rows = [
        {
            "id": "b0",
            "prompt_id": "p0",
            "human_output": "human beemo text zero",
            "model_output": "model beemo text zero",
            "human_edits": "edited beemo text zero",
            "category": "story",
        }
    ]
    pq.write_table(pa.Table.from_pylist(val_rows), wo / "validation-00000-of-00001.parquet")
    pq.write_table(pa.Table.from_pylist(test_rows), wo / "test-00000-of-00001.parquet")
    pq.write_table(pa.Table.from_pylist(cal_rows), wo / "calibration-00000-of-00001.parquet")
    pq.write_table(pa.Table.from_pylist(beemo_rows), be / "train-00000-of-00001.parquet")
    cal_ids_p = tmp_path / "cal_ids.json"
    cal_ids_p.write_text(json.dumps(["c0", "c1"]), encoding="utf-8")
    return d, cal_ids_p


# =============================================================================
# F1 - F12: Challenger 1 Findings
# =============================================================================


@pytest.mark.parametrize(
    "bad_bytes",
    [
        b"null",
        b"[]",
        b'{"choices": [null]}',
        b'{"choices": [{"message": {"role": "assistant", "content": "ok"}, "text": "ok"}], "usage": "bad"}',
        b'{"choices": "\xff',
    ],
)
def test_f1_client_malformed_envelopes_utf8_retry_and_evaluator_infra_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, bad_bytes: bytes
):
    monkeypatch.setattr("chantless.client.time.sleep", lambda _s: None)
    mode = {"transient": True}

    def responder(path: str, body: dict, idx: int):
        if mode["transient"] and idx % 2 == 0:
            return 200, bad_bytes
        if not mode["transient"]:
            return 200, bad_bytes
        if path == "/v1/chat/completions":
            return 200, json.dumps({"choices": [{"message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}], "usage": {"completion_tokens": 1}}).encode()
        return 200, json.dumps({"choices": [{"text": "ok", "finish_reason": "stop"}], "usage": {"completion_tokens": 1}}).encode()

    srv, th, url, calls = _make_raw_http_server(responder)
    try:
        cl = ChatClient(url, {"temperature": 0.0}, retries=2)
        assert cl.chat([{"role": "user", "content": "hi"}], seed=1, max_tokens=8)["message"]["content"] == "ok"
        assert cl.complete("hi", seed=2, max_tokens=8)["text"] == "ok"

        mode["transient"] = False
        before = len(calls)
        with pytest.raises(InfraError):
            cl.chat([{"role": "user", "content": "hi"}], seed=1, max_tokens=8)
        assert len(calls) - before == 2

        before_c = len(calls)
        with pytest.raises(InfraError):
            cl.complete("hi", seed=1, max_tokens=8)
        assert len(calls) - before_c == 2

        # Evaluator.run records infra_error cleanly when server persists in returning malformed body
        if bad_bytes == b'{"choices": [null]}':
            class StubGenEnv:
                classifier = False
                def run(self, task, client, seed):
                    res = client.chat([{"role": "user", "content": "hi"}], seed=seed, max_tokens=8)
                    return SimpleNamespace(
                        task_id=task.id, seed=seed, messages=[], tools=None, final_text=res["message"]["content"],
                        finish_reasons=["stop"], n_turns=1, completion_tokens=1, wall_s=0.01, infra_error=None,
                    )
                def verify(self, task, ep):
                    return SimpleNamespace(success=False, tests_passed=False, detectors={}, infra_error=ep.infra_error, details={})

            tc = runtime.Toolchain(llama_bin=tmp_path, traj_kl=tmp_path / "tk", gpus=[""])
            ev = evmod.Evaluator(tc, StubGenEnv(), {"temperature": 0.0}, 1, 512, "f16", tmp_path / "logs", infra_retries=0)
            rows = ev.run("m", [(Task(id="t0", family="f0", split="select", prompt={}), 0)], "tag", save=False, srvs=[SimpleNamespace(url=url, special=False)])
            assert len(rows) == 1
            assert rows[0]["success"] is None
            assert rows[0]["infra_error"]
    finally:
        srv.shutdown()
        srv.server_close()
        th.join(timeout=2)


def test_f2_detect_malformed_completion_probabilities_null_elem_and_utf8(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr("chantless.env.detect.time.sleep", lambda _s: None)
    hf_dir = _make_hf_dir(tmp_path)
    data_dir, _ = _make_detect_data_dir(tmp_path)
    env = detect.DetectEnvironment(
        data_dir=str(data_dir),
        hf_dir=str(hf_dir),
        cutoff=0.5,
        classifier=False,
        search_per_cell=1,
        select_per_cell=1,
        exam_test_per_cell=1,
        exam_beemo_per_kind=1,
    )
    task = env.tasks()[0]
    mode = {"payload": b'{"completion_probabilities": [null]}'}

    def responder(path: str, body: dict, idx: int):
        return 200, mode["payload"]

    srv, th, url, calls = _make_raw_http_server(responder)
    try:
        cl = ChatClient(url, {"temperature": 0.0}, retries=2)
        ep = env.run(task, cl, seed=0)
        assert ep.infra_error is not None
        assert len(calls) == 2

        mode["payload"] = b'{"completion_probabilities": "\xff'
        ep_utf8 = env.run(task, cl, seed=0)
        assert ep_utf8.infra_error is not None
        assert len(calls) == 4
    finally:
        srv.shutdown()
        srv.server_close()
        th.join(timeout=2)


@pytest.mark.parametrize("bad_cutoff", [float("nan"), float("inf"), float("-inf"), True, False])
def test_f3_detect_rejects_nonfinite_and_bool_cutoff(tmp_path: Path, bad_cutoff: Any):
    hf_dir = _make_hf_dir(tmp_path)
    data_dir, _ = _make_detect_data_dir(tmp_path)
    with pytest.raises(ValueError):
        detect.DetectEnvironment(
            data_dir=str(data_dir),
            hf_dir=str(hf_dir),
            cutoff=bad_cutoff,
            classifier=False,
        )


def test_f4_clsbench_and_detect_key_include_split(tmp_path: Path):
    hf_dir = _make_hf_dir(tmp_path)
    cache_dir = tmp_path / "cache"

    # PublicClassifierEnvironment: same id & text, swapped split -> different token_counts & _key
    j1 = _make_cls_jsonl(tmp_path / "j1.jsonl", [
        {"id": "c0", "text": "hello world", "label": "positive", "split": "calibration"},
        {"id": "t1", "text": "alpha", "label": "positive", "split": "search"},
        {"id": "t2", "text": "alpha beta gamma delta", "label": "negative", "split": "select"},
    ])
    j2 = _make_cls_jsonl(tmp_path / "j2.jsonl", [
        {"id": "c0", "text": "hello world", "label": "positive", "split": "calibration"},
        {"id": "t1", "text": "alpha", "label": "positive", "split": "select"},
        {"id": "t2", "text": "alpha beta gamma delta", "label": "negative", "split": "search"},
    ])
    env1 = clsbench.PublicClassifierEnvironment(hf_dir=str(hf_dir), cache_dir=str(cache_dir), jsonl_file=str(j1))
    env2 = clsbench.PublicClassifierEnvironment(hf_dir=str(hf_dir), cache_dir=str(cache_dir), jsonl_file=str(j2))
    tc1 = env1.token_counts()
    tc2 = env2.token_counts()
    assert tc1["search"]["tokens"] != tc2["search"]["tokens"]
    assert env1._key(env1.tasks(), f"hf:{env1._tok_id}", env1.pair[0]) != env2._key(env2.tasks(), f"hf:{env2._tok_id}", env2.pair[0])

    # DetectEnvironment._key: same id & text, different split -> different key
    data_dir, cal_ids_p = _make_detect_data_dir(tmp_path)
    # Add "human" and "ai" to HF vocab for DetectEnvironment
    vocab_p = hf_dir / "tokenizer.json"
    tok_json = json.loads(vocab_p.read_text())
    for w in ("human", "ai"):
        if w not in tok_json["model"]["vocab"]:
            tok_json["model"]["vocab"][w] = len(tok_json["model"]["vocab"])
    vocab_p.write_text(json.dumps(tok_json))
    det = detect.DetectEnvironment(
        data_dir=str(data_dir),
        hf_dir=str(hf_dir),
        classifier=True,
        calibration_ids_file=str(cal_ids_p),
        cache_dir=str(tmp_path / "det_cache"),
        search_per_cell=1,
        select_per_cell=1,
        exam_test_per_cell=1,
        exam_beemo_per_kind=1,
    )
    t_a = Task(id="x0", family="f", split="search", prompt={"text": "hello world"}, meta={})
    t_b = Task(id="x0", family="f", split="select", prompt={"text": "hello world"}, meta={})
    assert det._key([t_a]) != det._key([t_b])


def test_f5_client_reasoning_effort_non_inkling2_dialects_and_precedence():
    def responder(path: str, body: dict, idx: int):
        return 200, json.dumps({"choices": [{"message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}]}).encode()

    srv, th, url, calls = _make_raw_http_server(responder)
    try:
        for dialect in ("qwen3_json", "gemma4", "qwen35_xml"):
            ChatClient(url, {"reasoning_effort": "low"}, dialect=dialect).chat([{"role": "user", "content": "hi"}], seed=1, max_tokens=8)
            assert calls[-1]["body"]["chat_template_kwargs"] == {"enable_thinking": True}, dialect

            ChatClient(url, {"reasoning_effort": "none"}, dialect=dialect).chat([{"role": "user", "content": "hi"}], seed=1, max_tokens=8)
            assert calls[-1]["body"]["chat_template_kwargs"] == {"enable_thinking": False}, dialect

            # Explicit enable_thinking takes precedence over reasoning_effort on non-inkling2 dialects
            ChatClient(url, {"reasoning_effort": "low", "enable_thinking": False}, dialect=dialect).chat(
                [{"role": "user", "content": "hi"}], seed=1, max_tokens=8
            )
            assert calls[-1]["body"]["chat_template_kwargs"] == {"enable_thinking": False}, dialect

        # inkling2 & None preserve reasoning_effort directly
        ChatClient(url, {"reasoning_effort": "low"}, dialect="inkling2").chat([{"role": "user", "content": "hi"}], seed=1, max_tokens=8)
        assert calls[-1]["body"]["chat_template_kwargs"] == {"reasoning_effort": "low"}

        ChatClient(url, {"reasoning_effort": "low"}, dialect=None).chat([{"role": "user", "content": "hi"}], seed=1, max_tokens=8)
        assert calls[-1]["body"]["chat_template_kwargs"] == {"reasoning_effort": "low"}

        # Explicit chat_template_kwargs overrides
        ChatClient(url, {"reasoning_effort": "low"}, chat_template_kwargs={"enable_thinking": False}, dialect="qwen3_json").chat(
            [{"role": "user", "content": "hi"}], seed=1, max_tokens=8
        )
        assert calls[-1]["body"]["chat_template_kwargs"] == {"enable_thinking": False}
    finally:
        srv.shutdown()
        srv.server_close()
        th.join(timeout=2)


def test_f6_config_parse_override_value_yaml12_on_off_yes_no_true_false():
    for s in ("on", "ON", "On", "off", "OFF", "Off", "yes", "YES", "Yes", "no", "NO", "No"):
        val = cfgmod._parse_override_value(s)
        assert isinstance(val, str) and val == s, (s, val)
    assert cfgmod._parse_override_value("[no,yes]") == ["no", "yes"]
    assert cfgmod._parse_override_value("true") is True
    assert cfgmod._parse_override_value("True") is True
    assert cfgmod._parse_override_value("false") is False
    assert cfgmod._parse_override_value("False") is False

    base = {
        "run_id": "test",
        "run_root": "/tmp/runs",
        "source": {"gguf": "m.gguf", "repo": "org/m", "revision": "rev"},
        "toolchain": {"llama_bin": "/bin", "traj_kl": "/bin/true"},
        "env": {"module": "chantless.env.bfcl"},
        "ladder": ["BF16", "Q8_0", "Q4_K"],
        "start_tier": "Q4_K",
    }
    ov = cfgmod.apply_overrides(base, ["run_id=ON", "source.revision=no", "serving.cache_prompt=true", "serving.flash_attn=off"])
    ec = cfgmod.ExtractConfig.model_validate(ov)
    assert ec.run_id == "ON"
    assert ec.source.revision == "no"
    assert ec.serving.cache_prompt is True
    assert ec.serving.flash_attn == "off"


def test_f7_config_decoding_and_sanity_floor_factor_numeric_validation():
    # Valid boundary values preserved
    dc_ok = cfgmod.DecodingConfig(temperature=0.0, top_p=1.0, top_k=0, min_p=0.0, max_tokens=1, seed=0)
    assert dc_ok.temperature == 0.0 and dc_ok.top_k == 0 and dc_ok.min_p == 0.0
    assert cfgmod.SanityConfig(floor_factor=0.0).floor_factor == 0.0

    for bad_kw in (
        {"temperature": float("nan")},
        {"temperature": float("inf")},
        {"temperature": -0.1},
        {"temperature": True},
        {"top_p": float("nan")},
        {"top_p": float("inf")},
        {"top_p": -0.1},
        {"top_p": 1.1},
        {"top_p": False},
        {"top_k": -1},
        {"top_k": True},
        {"min_p": float("-inf")},
        {"min_p": -0.01},
        {"min_p": 1.5},
        {"min_p": True},
        {"max_tokens": 0},
        {"max_tokens": -5},
        {"max_tokens": True},
        {"seed": False},
    ):
        with pytest.raises(ValidationError):
            cfgmod.DecodingConfig(**bad_kw)

    for bad_ff in (float("nan"), float("inf"), float("-inf"), -1.0, True, False):
        with pytest.raises(ValidationError):
            cfgmod.SanityConfig(floor_factor=bad_ff)


def test_f8_runtime_llama_server_closes_log_fh_on_early_exit(tmp_path: Path):
    llama_bin = tmp_path / "bin"
    llama_bin.mkdir()
    srv_exe = llama_bin / "llama-server"
    srv_exe.write_text("#!/bin/sh\necho 'boom' >&2\nexit 1\n", encoding="utf-8")
    srv_exe.chmod(srv_exe.stat().st_mode | stat.S_IXUSR)

    tc = runtime.Toolchain(llama_bin=llama_bin, traj_kl=tmp_path / "traj-kl", gpus=[""])
    log_p = tmp_path / "srv.log"
    srv = runtime.LlamaServer(tc, tmp_path / "m.gguf", log=log_p)
    opened_fhs = []
    orig_open = open

    def spy_open(file, mode="r", *a, **kw):
        fh = orig_open(file, mode, *a, **kw)
        if Path(file) == log_p:
            opened_fhs.append(fh)
        return fh

    import builtins
    orig_builtin_open = builtins.open
    builtins.open = spy_open
    try:
        with pytest.raises(RuntimeError, match="exited early"):
            srv.__enter__()
    finally:
        builtins.open = orig_builtin_open

    assert srv._log_fh is None
    assert opened_fhs and all(fh.closed for fh in opened_fhs)


def test_f9_detect_slow_tokenizer_fallback(tmp_path: Path):
    hf_dir = _make_hf_dir(tmp_path)
    data_dir, _ = _make_detect_data_dir(tmp_path)
    env = detect.DetectEnvironment(
        data_dir=str(data_dir),
        hf_dir=str(hf_dir),
        cutoff=0.5,
        classifier=False,
        max_doc_tokens=2,
        search_per_cell=1,
        select_per_cell=1,
        exam_test_per_cell=1,
        exam_beemo_per_kind=1,
    )

    class SlowTok:
        def __call__(self, texts, add_special_tokens=False, return_offsets_mapping=False):
            if return_offsets_mapping:
                raise NotImplementedError("return_offsets_mapping is not available when using Python tokenizers")
            single = isinstance(texts, str)
            items = [texts] if single else texts
            ids = [[11, 12, 13][: len(t.split())] for t in items]
            return {"input_ids": ids[0] if single else ids}

        def decode(self, ids, skip_special_tokens=True):
            return " ".join(f"w{i}" for i in ids)

    env.tok = SlowTok()
    res_many = env._heads(["one", "one two three"])
    assert res_many[0] == ("one", False)
    assert res_many[1][1] is True and res_many[1][0] == "w11 w12"
    res_one = env._head("one two three")
    assert res_one == ("w11 w12", True)


def test_f10_clsbench_rejects_duplicate_ids_in_jsonl(tmp_path: Path):
    hf_dir = _make_hf_dir(tmp_path)
    jpath = _make_cls_jsonl(tmp_path / "dup.jsonl", [
        {"id": "d0", "text": "hello world", "label": "positive", "split": "calibration"},
        {"id": "d0", "text": "alpha beta", "label": "negative", "split": "search"},
    ])
    with pytest.raises(ValueError, match="duplicate"):
        clsbench.PublicClassifierEnvironment(hf_dir=str(hf_dir), cache_dir=str(tmp_path / "c"), jsonl_file=str(jpath))


def test_f11_sanity_get_type_hints_succeeds():
    hints = typing.get_type_hints(sanity._finite)
    assert hints["name"] is str and hints["return"] is float


def test_f12_bool_none_empty_edge_guards(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    hf_dir = _make_hf_dir(tmp_path)
    jpath = _make_cls_jsonl(tmp_path / "cls.jsonl", [
        {"id": "c0", "text": "hello world", "label": "positive", "split": "calibration"},
        {"id": "s0", "text": "alpha beta", "label": "negative", "split": "search"},
    ])
    cls_env = clsbench.PublicClassifierEnvironment(hf_dir=str(hf_dir), cache_dir=str(tmp_path / "c"), jsonl_file=str(jpath))
    for bad_n in (True, False):
        with pytest.raises(ValueError):
            cls_env.stats_texts(bad_n)

    # detect.calibrate rejects None & bool with ValueError; fit_token_counts handles empty fit_tasks on Unigram trimmed GGUF
    data_dir, cal_ids_p = _make_detect_data_dir(tmp_path)
    vocab_p = hf_dir / "tokenizer.json"
    tok_json = json.loads(vocab_p.read_text())
    for w in ("human", "ai"):
        if w not in tok_json["model"]["vocab"]:
            tok_json["model"]["vocab"][w] = len(tok_json["model"]["vocab"])
    vocab_p.write_text(json.dumps(tok_json))
    det_env = detect.DetectEnvironment(
        data_dir=str(data_dir),
        hf_dir=str(hf_dir),
        classifier=True,
        calibration_ids_file=str(cal_ids_p),
        cache_dir=str(tmp_path / "dc"),
        search_per_cell=1,
        select_per_cell=1,
        exam_test_per_cell=1,
        exam_beemo_per_kind=1,
    )
    for bad_v in (None, True, False):
        with pytest.raises(ValueError):
            det_env.calibrate({"wo-cal:c0": bad_v})

    # Empty fit_tasks with Unigram trimmed GGUF without tokdocs must return zero array
    det_env._tasks = []
    uni_tok = Tokenizer(models.Unigram([("<unk>", 0.0), ("human", -1.0), ("ai", -1.0)]))
    uni_tok.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    uni_dir = tmp_path / "uni_hf"
    uni_dir.mkdir()
    PreTrainedTokenizerFast(tokenizer_object=uni_tok, unk_token="<unk>").save_pretrained(str(uni_dir))
    det_env.tok = PreTrainedTokenizerFast.from_pretrained(str(uni_dir))
    det_env.source_vocab_size = len(det_env.tok)
    trimmed_gguf = tmp_path / "trimmed.gguf"
    _write_minimal_gguf(trimmed_gguf, "qwen35")
    monkeypatch.setattr(
        detect.vocabmod,
        "read_tokenizer",
        lambda p: {"vocab_size": 2, "tokens": [b"human", b"ai"], "merges": [], "token_type": [1, 1]},
    )
    cnt = det_env.fit_token_counts(str(trimmed_gguf), tc=None)
    assert cnt.shape == (2,) and int(cnt.sum()) == 0

    # evaluator guards: infra_retries < 0, srvs=[], _finite(True), rules/{h}.json == null
    tc = runtime.Toolchain(llama_bin=tmp_path, traj_kl=tmp_path / "tk", gpus=[""])
    m_gguf = tmp_path / "m.gguf"
    _write_minimal_gguf(m_gguf, "qwen35")
    assert evmod._finite(True) is False
    with pytest.raises(ValueError):
        evmod.Evaluator(tc, cls_env, {"temperature": 0.0}, 1, 512, "f16", tmp_path / "l1", infra_retries=-1, scorer=SimpleNamespace())
    ev_ok = evmod.Evaluator(tc, cls_env, {"temperature": 0.0}, 1, 512, "f16", tmp_path / "l2", scorer=SimpleNamespace())
    with pytest.raises(ValueError):
        ev_ok.seed(str(m_gguf), {"c0": True})
    h = ev_ok.model_hash(str(m_gguf))
    rules_p = tmp_path / "l2" / "rules" / f"{h}.json"
    rules_p.parent.mkdir(parents=True, exist_ok=True)
    rules_p.write_text("null", encoding="utf-8")
    ev_ok.seed(str(m_gguf), {"c0": 1.0})
    rule = ev_ok.rule(str(m_gguf))
    assert "cutoff" in rule

    ev_gen = evmod.Evaluator(tc, SimpleNamespace(classifier=False), {"temperature": 0.0}, 1, 512, "f16", tmp_path / "l3")
    with pytest.raises(ValueError):
        ev_gen.run("m", [], "tag", save=False, srvs=[])

    # compare & sanity guards
    with pytest.raises(ValueError):
        cmpmod.cutoff([True, False])
    with pytest.raises(ValueError):
        cmpmod.cutoff(np.array([True, False]))
    for bad_boot in (1.5, True, False, 0):
        with pytest.raises(ValueError):
            cmpmod.compare(
                {"m0": {"cal": [1.0, 2.0], "scores": {"d0": 0.5}}},
                {"d0": "pos"},
                {"d0": "k"},
                boot=bad_boot,
            )
    with pytest.raises(ValueError):
        sanity.SizeMatchedCap([(True, False)])
    cap = sanity.SizeMatchedCap([(2.0, 0.5)])
    with pytest.raises(ValueError):
        cap(True)

    # cli doctor --config missing.yaml -> SystemExit
    monkeypatch.setattr(sys, "argv", ["chantless", "doctor", "--config", str(tmp_path / "missing.yaml")])
    with pytest.raises(SystemExit):
        climod.main()


# =============================================================================
# Gaps 1 - 7: Challenger 2 Findings
# =============================================================================


def test_gap1_pipeline_gate_generative_behavioral_items_zero(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    gguf_p = _write_gguf(tmp_path / "m.gguf")
    hf_p = _make_hf_dir(tmp_path)
    tasks = [
        Task(id=f"sel-{fam}-{i}", family=fam, split="select", prompt={"text": "hi"}, meta={})
        for fam in ("famA", "famB", "famC")
        for i in range(2)
    ]
    stub_env = SimpleNamespace(
        name="stub-gen", version="1", classifier=False,
        deployment={"context": 512, "concurrency": 1, "kv_type": "f16", "max_total_memory_bytes": None},
        budget={"max_turns": 1, "max_tokens_per_turn": 64},
        tasks=lambda: list(tasks),
    )
    monkeypatch.setattr(pipeline, "load_environment", lambda mod, **kw: stub_env)

    def _run_gate(run_id: str, beh_items_n: int):
        cfg = {
            "run_id": run_id,
            "run_root": str(tmp_path / "runs"),
            "source": {"gguf": str(gguf_p), "repo": "test/r", "revision": "abc", "hf_dir": str(hf_p)},
            "toolchain": {"llama_bin": str(tmp_path), "traj_kl": str(tmp_path / "tk"), "gpus": [""]},
            "env": {"module": "stub.env", "kwargs": {}},
            "ladder": ["BF16", "Q8_0", "Q4_K"],
            "start_tier": "Q4_K",
            "sanity": {"behavioral_items": beh_items_n},
        }
        r = pipeline.Run(cfg)
        (r.dir / "pool.json").write_text(json.dumps({"reliable": {"select": [t.id for t in tasks]}}))
        (r.dir / "aa.json").write_text(json.dumps({"n_gate": 6}))
        (r.dir / "behavioral_bank.json").write_text("[]")
        (r.dir / "episodes").mkdir(parents=True, exist_ok=True)
        (r.dir / "gates").mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr(r.ev, "servers", lambda m: nullcontext([]))
        monkeypatch.setattr(
            r.ev, "run",
            lambda m, jobs, tag, side, srvs=None: [
                {"task": t.id, "family": t.family, "split": t.split, "seed": s, "side": side, "success": True, "detectors": {}}
                for t, s in jobs
            ],
        )
        monkeypatch.setattr(sanity, "run_behavioral", lambda *a, **kw: [])
        return r.gate(str(gguf_p), "g1")

    res_disabled = _run_gate("gap1-disabled", 0)
    assert res_disabled["behavioral_ok"] is True
    assert res_disabled["pass"] is True

    # When behavioral_items > 0 and all behavioral checks fail/infra_error (brows == []), must fail!
    res_all_infra = _run_gate("gap1-infra", 4)
    assert res_all_infra["behavioral_ok"] is False
    assert res_all_infra["pass"] is False


def test_gap2_kl_malformed_tool_calls_raises_render_error_and_pipeline_skips_cleanly(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    # Build HF tokenizer using real Inkling-2 chat template if available, else strict Jinja template enforcing parsed dict arguments
    hf_p = _make_hf_dir(tmp_path, dialect="inkling2")
    strict_tpl = (
        "{%- for m in messages -%}"
        "{%- if m.role == 'user' -%}{{- '<|message_user|><|content_text|>' ~ m.content ~ '<|end_message|>' -}}"
        "{%- elif m.role == 'assistant' -%}"
        "{%- if m.tool_calls -%}"
        "{%- for tc in m.tool_calls -%}"
        "{%- if tc.function.arguments is string -%}{{- raise_exception('tool call arguments must be a parsed object, not a JSON string') -}}{%- endif -%}"
        "{{- '<|message_model|><|content_invoke_tool_json|>{\"name\":\"' ~ tc.function.name ~ '\"}<|end_message|><|content_model_end_sampling|>' -}}"
        "{%- endfor -%}"
        "{%- else -%}{{- '<|message_model|><|content_text|>' ~ (m.content or '') ~ '<|end_message|><|content_model_end_sampling|>' -}}{%- endif -%}"
        "{%- endif -%}"
        "{%- endfor -%}"
    )
    tok = PreTrainedTokenizerFast.from_pretrained(str(hf_p))
    tok.chat_template = strict_tpl
    tok.save_pretrained(str(hf_p))

    R = kl.Renderer(str(hf_p), reasoning_effort="none")
    bad_json_msgs = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "run_tests", "arguments": "{bad json"}}]},
    ]
    bad_elem_msgs = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "", "tool_calls": ["not_a_dict"]},
    ]
    with pytest.raises(kl.RenderError):
        R.episode(bad_json_msgs, None)
    with pytest.raises(kl.RenderError):
        R.episode(bad_elem_msgs, None)

    # render_trajectories skips malformed one and keeps valid one
    valid_msgs = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "ok"}]
    recs = R.render_trajectories([bad_json_msgs, bad_elem_msgs, valid_msgs])
    assert len(recs) == 1

    # pipeline._failure_store skips cleanly instead of crashing
    gguf_p = _write_gguf(tmp_path / "m.gguf")
    monkeypatch.setattr(pipeline, "load_environment", lambda mod, **kw: SimpleNamespace(
        name="stub", version="1", classifier=False,
        deployment={"context": 512, "concurrency": 1, "kv_type": "f16", "max_total_memory_bytes": None},
        budget={"max_turns": 1, "max_tokens_per_turn": 64}, tasks=lambda: [],
    ))
    r = pipeline.Run({
        "run_id": "r-gap2",
        "run_root": str(tmp_path / "runs"),
        "source": {"gguf": str(gguf_p), "repo": "test/r", "revision": "abc", "hf_dir": str(hf_p)},
        "toolchain": {"llama_bin": str(tmp_path), "traj_kl": str(tmp_path / "tk"), "gpus": [""]},
        "env": {"module": "stub.env", "kwargs": {}},
        "ladder": ["BF16", "Q8_0", "Q4_K"],
        "start_tier": "Q4_K",
        "sanity": {"behavioral_items": 0},
    })
    out = r._failure_store([{"task": "t0", "family": "f0", "messages": bad_json_msgs, "tools": None}], "fb_test")
    assert out is None


def test_gap3_stats_detector_bounds_rejects_nan_u():
    one_fam = [
        {"task": "t0", "family": "famA", "side": "cand", "detectors": {"doom_loop": True}},
        {"task": "t0", "family": "famA", "side": "src", "detectors": {"doom_loop": False}},
    ]
    res_1f = stats.detector_bounds(one_fam, alpha=0.05, bound=0.03)
    assert math.isnan(res_1f["doom_loop"]["U"])
    assert res_1f["doom_loop"]["ok"] is False

    zero_cand = [
        {"task": "t0", "family": "famA", "side": "src", "detectors": {"doom_loop": False}},
        {"task": "t1", "family": "famB", "side": "src", "detectors": {"doom_loop": False}},
    ]
    res_0c = stats.detector_bounds(zero_cand, alpha=0.05, bound=0.03)
    assert math.isnan(res_0c["doom_loop"]["U"])
    assert res_0c["doom_loop"]["ok"] is False


def test_gap4_learn_gates_seq_feedback_batch_zero(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    class TinySeqStudent(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = torch.nn.Linear(4, 4, bias=False)
            self.lm_head = torch.nn.Linear(4, 8, bias=False)
            self.config = SimpleNamespace(final_logit_softcapping=None, model_type="qwen3")
            self.hook_mod = torch.nn.Identity()
            self.model = SimpleNamespace(
                layers=[SimpleNamespace(mlp=SimpleNamespace(switch_hook=self.hook_mod))],
                config=SimpleNamespace(num_hidden_layers=1, hidden_size=4),
            )
        def forward_bb(self, seqs, round_to=32):
            assert len(seqs) > 0
            x = torch.ones((len(seqs), 4, 4), dtype=torch.float32)
            return self.hook_mod(self.lin(x))

    tiny = TinySeqStudent()
    monkeypatch.setattr(learn.lattice, "Artifact", lambda p: SimpleNamespace(tensors={"blk.0.ffn_up.weight": {"dims": [4, 4]}}, close=lambda: None))
    monkeypatch.setattr(learn, "build_student", lambda *a, **kw: (tiny, None))
    monkeypatch.setattr(learn, "_switchable_layers", lambda *a, **kw: [0])
    monkeypatch.setattr(learn, "_layer_ffn_width", lambda art, il: 4)
    monkeypatch.setattr(learn, "_switch_hook_module", lambda layer: tiny.hook_mod)
    monkeypatch.setattr(learn.PrefixCache, "build", staticmethod(lambda *a, **kw: None))
    monkeypatch.setattr(learn, "_forward_backbone", lambda model, seqs, dev, amp, round_to=32, **kw: tiny.forward_bb(seqs, round_to=round_to))
    monkeypatch.setattr(
        learn, "_apply_switches_to_file_and_mask",
        lambda art, ap, out, layers, theta, sm, **kw: (Path(out).write_bytes(b"G"), ({0: []}, 4, {0: 4}, None))[1],
    )
    monkeypatch.setattr(learn, "sha256", lambda p: "0")

    in_p = tmp_path / "in.gguf"
    in_p.write_bytes(b"IN")
    out_p = tmp_path / "out.gguf"
    rec = learn.learn_gates_seq(
        str(in_p),
        None,
        [{"pos_tokens": [1, 2, 3, 4], "pos_targets": [2, 3]}],
        [(np.array([1, 2, 3, 4], dtype=np.int32), np.array([2, 3], dtype=np.uint32))],
        str(out_p),
        cfg={"steps": 1, "feedback_batch": 0, "replay_batch": 1, "amp": False, "frozen_bf16": False, "stream_prefix": False},
        device="cpu",
    )
    assert rec["steps"] == 1 and out_p.exists()


def test_gap5_learn_train_shares_exceeding_feedback_batch_and_zero_feedback_batch():
    torch.manual_seed(0)
    lin = torch.nn.Linear(4, 2)
    model = SimpleNamespace(parameters=lin.parameters)

    def fake_scores(m, docs, pair, dev, amp, prefix_cache=None):
        assert len(docs) > 0
        x = torch.stack([torch.tensor(d, dtype=torch.float32)[:4] for d in docs])
        out = lin(x)
        return out[:, 0] - out[:, 1]

    orig_scores = learn.scores
    learn.scores = fake_scores
    try:
        feedback = [
            {"tokens": np.array([1.0, 2.0, 3.0, 4.0]), "label": ("human" if i % 2 == 0 else "ai"), "share": f"s{i}"}
            for i in range(9)
        ]
        replay = [np.array([0.5, 0.5, 0.5, 0.5]) for _ in range(4)]
        hist = learn._train(
            model, list(lin.parameters()), feedback, replay, (0, 1), 0.0,
            {"steps": 2, "feedback_batch": 8, "replay_batch": 2, "train_budget": 0, "amp": False},
            log=lambda _m: None, device="cpu", amp=False, strata=["human"] * 4, lr=1e-2,
        )
        assert len(hist) == 2

        hist0 = learn._train(
            model, list(lin.parameters()), feedback, replay, (0, 1), 0.0,
            {"steps": 1, "feedback_batch": 0, "replay_batch": 2, "train_budget": 0, "amp": False},
            log=lambda _m: None, device="cpu", amp=False, strata=["human"] * 4, lr=1e-2,
        )
        assert len(hist0) == 1 and hist0[0][0] == 0.0

        # Iteration 2 Finding 1: feedback_batch=0 with project=True (A-GEM) must not raise RuntimeError
        hist_proj = learn._train(
            model, list(lin.parameters()), feedback, replay, (0, 1), 0.0,
            {"steps": 1, "feedback_batch": 0, "replay_batch": 2, "train_budget": 0, "amp": False, "project": True},
            log=lambda _m: None, device="cpu", amp=False, strata=["human"] * 4, lr=1e-2,
        )
        assert len(hist_proj) == 1 and hist_proj[0][0] == 0.0

        # Iteration 2 Finding 1: feedback_batch=0, replay_batch=0 with train_budget > 0 must not raise IndexError in capped()
        hist_empty_cap = learn._train(
            model, list(lin.parameters()), feedback, replay, (0, 1), 0.0,
            {"steps": 1, "feedback_batch": 0, "replay_batch": 0, "train_budget": 1024, "amp": False},
            log=lambda _m: None, device="cpu", amp=False, strata=["human"] * 4, lr=1e-2,
        )
        assert len(hist_empty_cap) == 1 and hist_empty_cap[0] == (0.0, 0.0, 0.0)

        # Empty batch with train_budget=0 must also skip backward/step cleanly
        hist_empty_nocap = learn._train(
            model, list(lin.parameters()), feedback, replay, (0, 1), 0.0,
            {"steps": 1, "feedback_batch": 0, "replay_batch": 0, "train_budget": 0, "amp": False},
            log=lambda _m: None, device="cpu", amp=False, strata=["human"] * 4, lr=1e-2,
        )
        assert len(hist_empty_nocap) == 1 and hist_empty_nocap[0] == (0.0, 0.0, 0.0)

        # feedback_batch > 0, replay_batch=0, project=True (has_new=True, has_old=False) must step cleanly
        hist_new_only_proj = learn._train(
            model, list(lin.parameters()), feedback, replay, (0, 1), 0.0,
            {"steps": 1, "feedback_batch": 2, "replay_batch": 0, "train_budget": 1024, "amp": False, "project": True},
            log=lambda _m: None, device="cpu", amp=False, strata=["human"] * 4, lr=1e-2,
        )
        assert len(hist_new_only_proj) == 1
    finally:
        learn.scores = orig_scores


def test_gap6_deepmath_and_deepmath_select_difficulty_family_and_displaystyle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    hf_dir = _make_hf_dir(tmp_path)

    # Whitespace difficulty defaults to 5.0
    ws_j = tmp_path / "ws.jsonl"
    ws_j.write_text(json.dumps({"id": "0", "question": "1+1?", "answer": 2, "difficulty": "   "}) + "\n", encoding="utf-8")
    dm_ws = deepmath.DeepMathEnvironment(hf_dir=str(hf_dir), cache_dir=str(tmp_path / "c"), jsonl_file=str(ws_j))
    assert dm_ws.tasks()[0].meta["difficulty"] == 5.0

    # Bool / NaN / Inf difficulty rejected; blank family rejected
    for bad_row in (
        {"id": "0", "question": "q", "answer": 2, "difficulty": True},
        {"id": "0", "question": "q", "answer": 2, "difficulty": False},
        {"id": "0", "question": "q", "answer": 2, "difficulty": "nan"},
        {"id": "0", "question": "q", "answer": 2, "difficulty": float("inf")},
        {"id": "0", "question": "q", "answer": 2, "difficulty": 4.0, "split": "search", "family": ""},
        {"id": "0", "question": "q", "answer": 2, "difficulty": 4.0, "split": "search", "family": "   "},
    ):
        bad_j = tmp_path / "bad.jsonl"
        bad_j.write_text(json.dumps(bad_row) + "\n", encoding="utf-8")
        with pytest.raises(ValueError):
            deepmath.DeepMathEnvironment(hf_dir=str(hf_dir), cache_dir=str(tmp_path / "c"), jsonl_file=str(bad_j))

    stub_pkg = tmp_path / "stub_pkgs" / "sklearn"
    stub_pkg.mkdir(parents=True)
    (stub_pkg / "__init__.py").write_text("")
    (stub_pkg / "cluster.py").write_text(
        "import numpy as np\n"
        "class KMeans:\n"
        "    def __init__(self, n_clusters, **kw): self.k = int(n_clusters)\n"
        "    def fit(self, X):\n"
        "        n = len(X)\n"
        "        self.labels_ = np.array([(i * self.k) // n for i in range(n)], dtype=int)\n"
        "        self.cluster_centers_ = np.zeros((self.k, X.shape[1]), dtype=X.dtype)\n"
        "        self.inertia_ = 1.0\n"
        "        return self\n"
    )
    # deepmath_select: \displaystyle parsed, whitespace difficulty skipped, NaN/bool difficulty rejected
    monkeypatch.syspath_prepend(str(stub_pkg.parent))
    monkeypatch.syspath_prepend(str(ROOT / "mvp/tools"))
    import deepmath_select  # noqa: E402
    assert deepmath_select._parse_int_answer(r"\boxed{\displaystyle 42}") == "42"
    assert deepmath_select._parse_int_answer(r"\displaystyle -17") == "-17"
    monkeypatch.delitem(sys.modules, "deepmath_select", raising=False)
    monkeypatch.delitem(sys.modules, "sklearn.cluster", raising=False)
    monkeypatch.delitem(sys.modules, "sklearn", raising=False)
    env_sub = {**os.environ, "PYTHONPATH": f"{stub_pkg.parent}:{ROOT / 'mvp'}"}
    base_rows = [
        {"question": f"Q{i}", "final_answer": str(i + 10), "difficulty": "4.0", "topic": "Math -> Algebra", "shard": 0, "row": i}
        for i in range(10)
    ]
    out_ws = tmp_path / "out_ws"
    out_ws.mkdir()
    np.save(out_ws / "pool-embeddings.npy", np.ones((10, 4), dtype=np.float32))
    qa_ws = tmp_path / "qa_ws.parquet"
    pq.write_table(pa.Table.from_pylist(base_rows + [{"question": "WS", "final_answer": "99", "difficulty": "   ", "topic": "Math -> Algebra", "shard": 0, "row": 10}]), qa_ws)
    p_ws = subprocess.run([sys.executable, str(ROOT / "mvp/tools/deepmath_select.py"), str(qa_ws), str(tmp_path / "m"), str(out_ws), "10", "3", "3"], capture_output=True, text=True, env=env_sub)
    assert p_ws.returncode == 0, p_ws.stderr

    for bad_diff in (float("nan"), True):
        qa_bad = tmp_path / f"qa_bad_{bad_diff}.parquet"
        pq.write_table(pa.Table.from_pylist([{"question": "Bad", "final_answer": "99", "difficulty": bad_diff, "topic": "Math -> Algebra", "shard": 0, "row": 0}]), qa_bad)
        out_bad = tmp_path / f"out_bad_{bad_diff}"
        p_bad = subprocess.run([sys.executable, str(ROOT / "mvp/tools/deepmath_select.py"), str(qa_bad), str(tmp_path / "m"), str(out_bad), "10", "3", "3"], capture_output=True, text=True, env=env_sub)
        assert p_bad.returncode != 0


def test_gap7_traj_kl_and_logodds_strict_cli_and_empty_trj(tmp_path: Path):
    cxx = shutil.which("clang++") or shutil.which("g++")
    if not cxx:
        pytest.skip("no C++ compiler available")
    inc = tmp_path / "inc"
    inc.mkdir()
    (inc / "ggml.h").write_text("""#pragma once
#include <cstdint>
#include <cstring>
enum ggml_type { GGML_TYPE_F16 = 1, GGML_TYPE_Q8_0 = 8, GGML_TYPE_Q4_0 = 2 };
typedef uint16_t ggml_fp16_t;
inline void ggml_fp32_to_fp16_row(const float * x, ggml_fp16_t * y, int64_t n) {
    for (int64_t i = 0; i < n; ++i) { uint32_t u; memcpy(&u, &x[i], 4); y[i] = (uint16_t)(u >> 16); }
}
inline void ggml_fp16_to_fp32_row(const ggml_fp16_t * x, float * y, int64_t n) {
    for (int64_t i = 0; i < n; ++i) { uint32_t u = ((uint32_t)x[i]) << 16; memcpy(&y[i], &u, 4); }
}
""")
    (inc / "llama.h").write_text("""#pragma once
#define LLAMA_H
#include "ggml.h"
#include <cstdint>
#include <vector>
typedef int32_t llama_token; typedef int32_t llama_pos; typedef int32_t llama_seq_id;
enum llama_flash_attn_type { LLAMA_FLASH_ATTN_TYPE_AUTO = -1, LLAMA_FLASH_ATTN_TYPE_DISABLED = 0, LLAMA_FLASH_ATTN_TYPE_ENABLED = 1 };
struct llama_model_params { int32_t n_gpu_layers; };
struct llama_context_params { uint32_t n_ctx, n_batch, n_ubatch, n_seq_max; llama_flash_attn_type flash_attn_type; ggml_type type_k, type_v; int32_t n_threads, n_threads_batch; bool kv_unified; };
struct llama_batch { int32_t n_tokens; llama_token * token; float * embd; llama_pos * pos; int32_t * n_seq_id; llama_seq_id ** seq_id; int8_t * logits; };
struct llama_vocab { int n; };
struct llama_model { llama_vocab v{32}; };
struct llama_memory_t { int d; };
typedef llama_memory_t * llama_memory_i;
struct llama_context { llama_context_params cp; llama_memory_t mem; std::vector<std::vector<float>> lb; };
inline void llama_backend_init() {}
inline void llama_backend_free() {}
inline llama_model_params llama_model_default_params() { return {0}; }
inline llama_context_params llama_context_default_params() { return {512, 512, 512, 1, LLAMA_FLASH_ATTN_TYPE_AUTO, GGML_TYPE_F16, GGML_TYPE_F16, 1, 1, false}; }
inline llama_model * llama_model_load_from_file(const char *, llama_model_params) { return new llama_model(); }
inline void llama_model_free(llama_model * m) { delete m; }
inline const llama_vocab * llama_model_get_vocab(const llama_model * m) { return &m->v; }
inline int llama_vocab_n_tokens(const llama_vocab * v) { return v->n; }
inline llama_context * llama_init_from_model(llama_model *, llama_context_params cp) { auto * c = new llama_context(); c->cp = cp; return c; }
inline void llama_free(llama_context * c) { delete c; }
inline llama_memory_i llama_get_memory(llama_context * c) { return &c->mem; }
inline void llama_memory_clear(llama_memory_i, bool) {}
inline void llama_synchronize(llama_context *) {}
inline llama_batch llama_batch_init(int32_t n, int32_t, int32_t s) {
    llama_batch b{}; b.token = new llama_token[n]; b.pos = new llama_pos[n]; b.n_seq_id = new int32_t[n];
    b.seq_id = new llama_seq_id*[n]; for (int i = 0; i < n; ++i) b.seq_id[i] = new llama_seq_id[s];
    b.logits = new int8_t[n]; return b;
}
inline void llama_batch_free(llama_batch b) { delete[] b.token; delete[] b.pos; delete[] b.n_seq_id; delete[] b.logits; delete[] b.seq_id; }
inline int llama_decode(llama_context * c, llama_batch b) { c->lb.assign(b.n_tokens, std::vector<float>(32, 0.1f)); return 0; }
inline float * llama_get_logits_ith(llama_context * c, int32_t i) { return c->lb.at(i).data(); }
""")
    tk_bin = tmp_path / "traj-kl"
    lo_bin = tmp_path / "logodds"
    subprocess.run([cxx, "-O2", "-std=c++17", f"-I{inc}", str(ROOT / "mvp/tools/traj-kl.cpp"), "-o", str(tk_bin)], check=True)
    subprocess.run([cxx, "-O2", "-std=c++17", f"-I{inc}", str(ROOT / "mvp/tools/logodds.cpp"), "-o", str(lo_bin)], check=True)

    m_p = tmp_path / "m.gguf"
    m_p.write_bytes(b"stub")
    store_p = tmp_path / "store"
    store_p.mkdir()
    valid_trj = tmp_path / "valid.bin"
    with open(valid_trj, "wb") as f:
        f.write(b"TRJ1" + struct.pack("<I", 1) + struct.pack("<II", 3, 1) + struct.pack("<3i", 1, 2, 3) + struct.pack("<I", 2))
    empty_trj = tmp_path / "empty.bin"
    with open(empty_trj, "wb") as f:
        f.write(b"TRJ1" + struct.pack("<I", 0))
    zero_tgt_trj = tmp_path / "zero_tgt.bin"
    with open(zero_tgt_trj, "wb") as f:
        f.write(b"TRJ1" + struct.pack("<I", 1) + struct.pack("<II", 3, 0) + struct.pack("<3i", 1, 2, 3))

    for args in (
        ["--mode", "base", "-m", str(m_p), "--traj", str(valid_trj), "--store", str(store_p), "--ubatch", "abc"],
        ["--mode", "base", "-m", str(m_p), "--traj", str(valid_trj), "--store", str(store_p), "--threads", "1abc"],
        ["--mode", "base", "-m", str(m_p), "--traj", str(empty_trj), "--store", str(store_p)],
        ["--mode", "base", "-m", str(m_p), "--traj", str(zero_tgt_trj), "--store", str(store_p)],
    ):
        p = subprocess.run([str(tk_bin), *args], capture_output=True, text=True)
        assert p.returncode > 0, (args, p.returncode, p.stderr)

    out_tsv = tmp_path / "out.tsv"
    for args in (
        ["-m", str(m_p), "--traj", str(valid_trj), "--pair", "1,2", "--out", str(out_tsv), "--ubatch", "abc"],
        ["-m", str(m_p), "--traj", str(valid_trj), "--pair", "1,2", "--out", str(out_tsv), "--n-seq", "2abc"],
        ["-m", str(m_p), "--traj", str(valid_trj), "--pair", "1,2", "--out", str(out_tsv), "--shard", "0/1abc"],
        ["-m", str(m_p), "--traj", str(valid_trj), "--pair", "1,2abc", "--out", str(out_tsv)],
    ):
        p = subprocess.run([str(lo_bin), *args], capture_output=True, text=True)
        assert p.returncode > 0, (args, p.returncode, p.stderr)


# =============================================================================
# Iteration 2 Findings 2 - 4: Challenger 4 & Reviewer 2 Findings
# =============================================================================


def test_iter2_finding2_kl_non_iterable_content_raises_render_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    # Real .local/inkling2-colab/chat_template.jinja + Qwen3 HF dir + fallback tokenizer without effort/thinking kwargs
    ink_hf = _build_inkling2_hf_dir(tmp_path, name="hf_ink_f2")
    qwen_hf = _make_hf_dir(tmp_path / "qwen_f2", dialect="qwen3")

    bad_int_msgs = [{"role": "user", "content": 123}, {"role": "assistant", "content": "ok"}]
    bad_list_msgs = [{"role": "user", "content": [123]}, {"role": "assistant", "content": "ok"}]
    good_msgs = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "ok"}]

    for hf_dir in (ink_hf, qwen_hf):
        R = kl.Renderer(str(hf_dir), reasoning_effort="none")
        for bad_msgs in (bad_int_msgs, bad_list_msgs):
            with pytest.raises(kl.RenderError):
                R.episode(bad_msgs, None)
        recs = R.render_trajectories([bad_int_msgs, good_msgs])
        assert len(recs) == 1

    # Direct _render() against .local/inkling2-colab/chat_template.jinja must also convert TypeError -> RenderError
    R_ink = kl.Renderer(str(ink_hf), reasoning_effort="none")
    with pytest.raises(kl.RenderError):
        R_ink._render(bad_int_msgs, None, False)

    # pipeline._failure_store catches kl.RenderError on non-iterable content against Inkling-2 template
    gguf_p = _write_gguf(tmp_path / "m_f2.gguf")
    monkeypatch.setattr(pipeline, "load_environment", lambda mod, **kw: SimpleNamespace(
        name="stub", version="1", classifier=False,
        deployment={"context": 512, "concurrency": 1, "kv_type": "f16", "max_total_memory_bytes": None},
        budget={"max_turns": 1, "max_tokens_per_turn": 64}, tasks=lambda: [],
    ))
    r = pipeline.Run({
        "run_id": "r-iter2-f2",
        "run_root": str(tmp_path / "runs"),
        "source": {"gguf": str(gguf_p), "repo": "test/r", "revision": "abc", "hf_dir": str(ink_hf)},
        "toolchain": {"llama_bin": str(tmp_path), "traj_kl": str(tmp_path / "tk"), "gpus": [""]},
        "env": {"module": "stub.env", "kwargs": {}},
        "ladder": ["BF16", "Q8_0", "Q4_K"],
        "start_tier": "Q4_K",
        "sanity": {"behavioral_items": 0},
    })
    assert r._failure_store([{"task": "t0", "family": "f0", "messages": bad_int_msgs, "tools": None}], "fb_f2") is None

    # Kwarg-probing TypeError fallback must NOT be swallowed prematurely when tokenizer rejects optional kwargs
    R_fb = kl.Renderer(str(qwen_hf), reasoning_effort="none")
    orig_apply = R_fb.tok.apply_chat_template

    def strict_no_extra_kwargs(msgs, tools=None, add_generation_prompt=False, tokenize=False):
        return orig_apply(msgs, tools=tools, add_generation_prompt=add_generation_prompt, tokenize=tokenize)

    R_fb.tok.apply_chat_template = strict_no_extra_kwargs
    recs_fb = R_fb.episode(good_msgs, None)
    assert len(recs_fb) == 1
    with pytest.raises(kl.RenderError):
        R_fb.episode(bad_int_msgs, None)


def test_iter2_finding3_codetasks_and_detectors_non_list_tool_calls(tmp_path: Path):
    msgs = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "ok", "tool_calls": 123}]
    assert detectors.malformed_tool_call(msgs, {"run_tests"}) is True
    assert detectors.doom_loop(msgs, False) is False
    all_det = detectors.run_all(msgs, "ok", ["stop"], False, {"run_tests"})
    assert all_det["malformed_tool_call"] is True and all_det["doom_loop"] is False

    # CodeTasksEnv.run() + verify() + Evaluator.run() must not raise TypeError on non-list tool_calls=123
    env = codetasks.CodeTasksEnv(seed=20261012, variants_per_family=1, max_turns=2, max_tokens_per_turn=64)
    task = env.tasks()[0]

    def responder(_path, _body, _idx):
        return 200, json.dumps({
            "choices": [{
                "message": {"role": "assistant", "content": "```python\ndef f():\n    return 1\n```", "tool_calls": 123},
                "finish_reason": "stop",
            }],
            "usage": {"completion_tokens": 5},
        }).encode("utf-8")

    srv, th, url, _calls = _make_raw_http_server(responder)
    try:
        cl = ChatClient(url, {"temperature": 0.0})
        ep = env.run(task, cl, seed=0)
        assert ep.infra_error is None
        assert ep.n_turns == 1
        assert ep.final_text is None
        v = env.verify(task, ep)
        assert v.success is False
        assert v.detectors["malformed_tool_call"] is True

        tc = runtime.Toolchain(llama_bin=tmp_path, traj_kl=tmp_path / "tk", gpus=[""])
        ev = evmod.Evaluator(tc, env, {"temperature": 0.0}, 1, 512, "f16", tmp_path / "evlog")
        rows = ev.run("stub.gguf", [(task, 0)], "ct-nonlist", save=False, srvs=[SimpleNamespace(url=url)])
        assert len(rows) == 1
        assert rows[0]["infra_error"] is None
        assert rows[0]["success"] is False
        assert rows[0]["detectors"]["malformed_tool_call"] is True
    finally:
        srv.shutdown()
        srv.server_close()
        th.join(timeout=2)


def test_iter2_finding4_phase_exam_improve_mode_claim_holds_requires_pass(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    gguf_p = _write_gguf(tmp_path / "m_f4.gguf")
    hf_p = _make_hf_dir(tmp_path / "hf_f4")
    tasks = [
        Task(id=f"ex-{fam}-{i}", family=fam, split="exam", prompt={"text": f"q{i}"}, meta={})
        for fam in ("famA", "famB")
        for i in range(5)
    ]
    stub_env = SimpleNamespace(
        name="stub-improve", version="1", classifier=False,
        deployment={"context": 512, "concurrency": 1, "kv_type": "f16", "max_total_memory_bytes": None},
        budget={"max_turns": 1, "max_tokens_per_turn": 64},
        tasks=lambda: list(tasks),
    )
    monkeypatch.setattr(pipeline, "load_environment", lambda mod, **kw: stub_env)
    cfg = {
        "run_id": "iter2-f4-improve",
        "run_root": str(tmp_path / "runs"),
        "source": {"gguf": str(gguf_p), "repo": "test/r", "revision": "abc", "hf_dir": str(hf_p)},
        "toolchain": {"llama_bin": str(tmp_path), "traj_kl": str(tmp_path / "tk"), "gpus": [""]},
        "env": {"module": "stub.env", "kwargs": {}},
        "objective_mode": "improve",
        "ladder": ["BF16", "Q8_0", "Q4_K"],
        "start_tier": "Q4_K",
        "pool": {"tasks_per_split": {"exam": 10}, "runs": 1, "min_pass": 1},
        "stats": {"alpha": 0.05, "margin": 0.02, "detector_bound": 0.03},
        "sanity": {"behavioral_items": 0},
    }
    r = pipeline.Run(cfg)
    (r.dir / "aa.json").write_text(json.dumps({"n_gate": 10}))
    (r.dir / "behavioral_bank.json").write_text("[]")
    (r.dir / "episodes").mkdir(parents=True, exist_ok=True)
    (r.dir / "gates").mkdir(parents=True, exist_ok=True)
    (r.dir / "final").mkdir(parents=True, exist_ok=True)
    shutil.copyfile(gguf_p, r.dir / "final" / "specialist.gguf")

    monkeypatch.setattr(r.ev, "servers", lambda m: nullcontext([]))
    monkeypatch.setattr(r, "exam_baselines", lambda spec: None)

    def fake_run(m, jobs, tag, side, srvs=None):
        if tag == "exam-pool":
            return [{"task": t.id, "family": t.family, "split": t.split, "seed": s, "side": side, "success": True, "detectors": {}} for t, s in jobs]
        # gate exam: cand passes 10/10 but 100% doom_loop=True; src passes 1/10 with doom_loop=False -> L > 0, pass=False
        out = []
        for idx, (t, s) in enumerate(jobs):
            if side == "cand":
                out.append({"task": t.id, "family": t.family, "split": t.split, "seed": s, "side": side, "success": True, "detectors": {"doom_loop": True}})
            else:
                out.append({"task": t.id, "family": t.family, "split": t.split, "seed": s, "side": side, "success": (idx == 0), "detectors": {"doom_loop": False}})
        return out

    monkeypatch.setattr(r.ev, "run", fake_run)
    r.guard.phase = "exam"
    r.phase_exam()

    exam_res = json.loads((r.dir / "gates" / "exam.json").read_text())
    assert exam_res["L"] > 0
    assert exam_res["detectors"]["doom_loop"]["ok"] is False
    assert exam_res["pass"] is False
    assert exam_res["claim"] == "superior to source"
    assert exam_res["claim_holds"] is False


# =============================================================================
# Iteration 3: Challenger 5 & Challenger 6 Findings (Test-First Hardening)
# =============================================================================


def test_iter3_c5_bug1_codetasks_and_detectors_unhashable_tool_name(tmp_path: Path):
    for bad_name in (["run_tests"], {"bad": 1}):
        msgs = [{
            "role": "assistant",
            "content": "```python\ndef f():\n    return 1\n```",
            "tool_calls": [{"id": "tc0", "function": {"name": bad_name, "arguments": "{}"}}],
        }]
        assert detectors.malformed_tool_call(msgs, codetasks.TOOL_NAMES) is True

    env = codetasks.CodeTasksEnv(seed=20261012, variants_per_family=1, max_turns=2, max_tokens_per_turn=64)
    task = env.tasks()[0]

    def responder(_path, _body, _idx):
        return 200, json.dumps({
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "```python\ndef solve():\n    return 1\n```",
                    "tool_calls": [{"id": "tc0", "function": {"name": ["run_tests"], "arguments": "{}"}}],
                },
                "finish_reason": "stop",
            }],
            "usage": {"completion_tokens": 5},
        }).encode("utf-8")

    srv, th, url, _calls = _make_raw_http_server(responder)
    try:
        cl = ChatClient(url, {"temperature": 0.0})
        ep = env.run(task, cl, seed=0)
        assert ep.infra_error is None
        v = env.verify(task, ep)
        assert v.success is False
        assert v.detectors["malformed_tool_call"] is True

        tc = runtime.Toolchain(llama_bin=tmp_path, traj_kl=tmp_path / "tk", gpus=[""])
        ev = evmod.Evaluator(tc, env, {"temperature": 0.0}, 1, 512, "f16", tmp_path / "evlog")
        rows = ev.run("stub.gguf", [(task, 0)], "c5-bug1", save=False, srvs=[SimpleNamespace(url=url)])
        assert len(rows) == 1
        assert rows[0]["infra_error"] is None
        assert rows[0]["success"] is False
        assert rows[0]["detectors"]["malformed_tool_call"] is True
    finally:
        srv.shutdown()
        srv.server_close()
        th.join(timeout=2)


def test_iter3_c5_bug2_codetasks_falsy_non_list_tool_calls(tmp_path: Path):
    env = codetasks.CodeTasksEnv(seed=20261012, variants_per_family=1, max_turns=2, max_tokens_per_turn=64)
    task = env.tasks()[0]
    good_code = f"```python\n{task.meta['reference'].rstrip()}\n```"

    mode = {"tool_calls": {}}

    def responder(_path, _body, _idx):
        msg: dict[str, Any] = {"role": "assistant", "content": good_code}
        if mode["tool_calls"] != "__omit__":
            msg["tool_calls"] = mode["tool_calls"]
        return 200, json.dumps({
            "choices": [{"message": msg, "finish_reason": "stop"}],
            "usage": {"completion_tokens": 5},
        }).encode("utf-8")

    srv, th, url, _calls = _make_raw_http_server(responder)
    try:
        cl = ChatClient(url, {"temperature": 0.0})
        tc = runtime.Toolchain(llama_bin=tmp_path, traj_kl=tmp_path / "tk", gpus=[""])
        ev = evmod.Evaluator(tc, env, {"temperature": 0.0}, 1, 512, "f16", tmp_path / "evlog")

        for bad_tc in ({}, 0, False, ""):
            mode["tool_calls"] = bad_tc
            ep = env.run(task, cl, seed=0)
            assert ep.infra_error is None
            assert ep.n_turns == 1
            assert ep.final_text is None
            assert ep.messages[-1]["tool_calls"] == bad_tc
            assert detectors.malformed_tool_call(ep.messages, codetasks.TOOL_NAMES) is True
            v = env.verify(task, ep)
            assert v.success is False
            assert v.detectors["malformed_tool_call"] is True

            rows = ev.run("stub.gguf", [(task, 0)], "c5-bug2-bad", save=False, srvs=[SimpleNamespace(url=url)])
            assert len(rows) == 1
            assert rows[0]["infra_error"] is None
            assert rows[0]["success"] is False
            assert rows[0]["detectors"]["malformed_tool_call"] is True

        # None and [] remain valid non-tool turns
        for ok_tc in (None, []):
            mode["tool_calls"] = ok_tc
            ep_ok = env.run(task, cl, seed=0)
            assert ep_ok.infra_error is None
            assert ep_ok.final_text == good_code
            assert "tool_calls" not in ep_ok.messages[-1]
            v_ok = env.verify(task, ep_ok)
            assert v_ok.success is True
            assert v_ok.detectors["malformed_tool_call"] is False
    finally:
        srv.shutdown()
        srv.server_close()
        th.join(timeout=2)


@pytest.mark.parametrize(
    "endpoint,payload",
    [
        ("chat", {"choices": [{"message": {"role": "assistant", "content": 123}, "finish_reason": "stop"}], "usage": {"completion_tokens": 2}}),
        ("chat", {"choices": [{"message": {"role": "assistant", "content": [{"type": "text", "text": "ok"}]}, "finish_reason": "stop"}], "usage": {"completion_tokens": 2}}),
        ("chat", {"choices": [{"message": {"role": "assistant", "content": "ok", "reasoning_content": 123}, "finish_reason": "stop"}], "usage": {"completion_tokens": 2}}),
        ("chat", {"choices": [{"message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}], "usage": {"completion_tokens": "bad"}}),
        ("chat", {"choices": [{"message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}], "usage": {"completion_tokens": True}}),
        ("complete", {"choices": [{"text": 123, "finish_reason": "stop"}], "usage": {"completion_tokens": 2}}),
        ("complete", {"choices": [{"text": "ok", "finish_reason": "stop"}], "usage": {"completion_tokens": "bad"}}),
        ("complete", {"choices": [{"text": "ok", "finish_reason": "stop"}], "usage": {"completion_tokens": False}}),
    ],
)
def test_iter3_c5_bug3_client_payload_type_validation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, endpoint: str, payload: dict[str, Any]
):
    monkeypatch.setattr("chantless.client.time.sleep", lambda _s: None)
    mode = {"transient": True}

    def responder(path: str, _body: dict, idx: int):
        if mode["transient"] and idx == 0:
            return 200, json.dumps(payload).encode("utf-8")
        if not mode["transient"]:
            return 200, json.dumps(payload).encode("utf-8")
        if path == "/v1/chat/completions":
            return 200, json.dumps({"choices": [{"message": {"role": "assistant", "content": None, "reasoning_content": None}, "finish_reason": "stop"}], "usage": {"completion_tokens": 0}}).encode("utf-8")
        return 200, json.dumps({"choices": [{"text": None, "finish_reason": "stop"}], "usage": {"completion_tokens": 0}}).encode("utf-8")

    srv, th, url, calls = _make_raw_http_server(responder)
    try:
        cl = ChatClient(url, {"temperature": 0.0}, retries=2)
        if endpoint == "chat":
            res = cl.chat([{"role": "user", "content": "hi"}], seed=1, max_tokens=8)
            assert res["message"]["content"] is None and res["completion_tokens"] == 0
        else:
            res = cl.complete("hi", seed=1, max_tokens=8)
            assert res["text"] == "" and res["completion_tokens"] == 0
        assert len(calls) == 2

        mode["transient"] = False
        before = len(calls)
        with pytest.raises(InfraError):
            if endpoint == "chat":
                cl.chat([{"role": "user", "content": "hi"}], seed=1, max_tokens=8)
            else:
                cl.complete("hi", seed=1, max_tokens=8)
        assert len(calls) - before == 2

        if endpoint == "chat":
            env = codetasks.CodeTasksEnv(seed=20261012, variants_per_family=1, max_turns=1, max_tokens_per_turn=32)
            task = env.tasks()[0]
            tc = runtime.Toolchain(llama_bin=tmp_path, traj_kl=tmp_path / "tk", gpus=[""])
            ev = evmod.Evaluator(tc, env, {"temperature": 0.0}, 1, 512, "f16", tmp_path / "evlog", infra_retries=0)
            rows = ev.run("stub.gguf", [(task, 0)], "c5-bug3", save=False, srvs=[SimpleNamespace(url=url)])
            assert len(rows) == 1
            assert rows[0]["success"] is None
            assert rows[0]["infra_error"]
    finally:
        srv.shutdown()
        srv.server_close()
        th.join(timeout=2)


def test_iter3_c6_bug2_direct_callers_non_str_content():
    for bad_content in (123, [{"type": "text", "text": "hello"}]):
        msgs = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": bad_content, "reasoning_content": 456}]
        assert detectors.strip_thinking(bad_content) == ""
        assert detectors.extract_code(bad_content) is None
        res = detectors.run_all(msgs, bad_content, ["stop"], False, codetasks.TOOL_NAMES)
        assert isinstance(res, dict) and set(res) == {"doom_loop", "malformed_tool_call", "premature_stop", "token_garbage"}
        assert res["premature_stop"] is True

        assert sanity._strip_reasoning_and_wire(bad_content) == ""
        assert sanity._check({"check": "exact", "answer": "123"}, bad_content) is False

        env = codetasks.CodeTasksEnv(seed=20261012, variants_per_family=1, max_turns=1, max_tokens_per_turn=32)
        task = env.tasks()[0]

        class DirectBadClient:
            dialect = None
            def chat(self, *_a, **_kw):
                return {"message": {"role": "assistant", "content": bad_content, "reasoning_content": 456}, "finish_reason": "stop", "completion_tokens": "bad"}

        ep = env.run(task, DirectBadClient(), seed=0)
        assert ep.infra_error is None
        v = env.verify(task, ep)
        assert v.success is False
        assert isinstance(v.detectors, dict)


def test_iter3_c6_bug1_learn_run_generative_pos_tokens_controls(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    gguf_p = _write_gguf(tmp_path / "m_c6b1.gguf")
    lessons = [
        {"pos_tokens": [1, 2, 3, 4], "pos_targets": [2, 3], "neg_tokens": [1, 2, 5, 6], "neg_targets": [2, 3]},
        {"pos_tokens": [10, 11, 12, 13, 14], "pos_targets": [3, 4], "neg_tokens": [10, 11, 15, 16, 17], "neg_targets": [3, 4]},
    ]
    lessons_p = tmp_path / "lessons.json"
    lessons_p.write_text(json.dumps(lessons), encoding="utf-8")

    captured: list[tuple[str, list[dict]]] = []

    def fake_learn_gates_seq(model_path, cfg_dir, train_lessons, replay_trajs, out_path, cfg=None, log=print, mask_in=None, mask_out=None):
        Path(out_path).write_bytes(b"GGUF")
        name = Path(out_path).name
        captured.append((name, [dict(r) for r in train_lessons]))
        for r in train_lessons:
            nl = learn._norm_seq_lesson(r)
            assert len(nl["pos_targets"]) > 0
        return {"method": "gates_seq", "steps": 1, "off": 1, "n_feedback": len(train_lessons)}

    def fake_traj_kl(tc, mode, model, trj, store, out, log_p, gpu="", **_kw):
        n_trj = struct.unpack("<I", Path(trj).read_bytes()[4:8])[0]
        res = {"mode": mode, "n_tgt": n_trj, "kl": 0.0, "nll_src": 1.0, "nll_cand": 0.5,
               "trajs": [{"i": i, "n_tok": 4, "n_tgt": 2, "kl": 0.0, "nll_src": 1.0, "nll_cand": 0.5 if "apprentice" in str(model) else 1.5} for i in range(n_trj)]}
        Path(out).write_text(json.dumps(res), encoding="utf-8")
        return res

    monkeypatch.setattr(learn, "learn_gates_seq", fake_learn_gates_seq)
    monkeypatch.setattr(kl, "traj_kl", fake_traj_kl)

    out_dir = tmp_path / "out_c6b1"
    cfg = {
        "toolchain": {"llama_bin": str(tmp_path), "traj_kl": str(tmp_path / "tk"), "gpus": [""]},
        "learn": {
            "mode": "generative",
            "model": str(gguf_p),
            "out_dir": str(out_dir),
            "lessons": str(lessons_p),
            "controls": True,
            "steps": 1,
        },
    }
    res = learn.run_generative(cfg, log=lambda _m: None)
    assert set(res["controls_lineage"]) == {"empty", "flipped", "shuffled"}
    assert set(res["cpp_traj_kl"]) == {"base", "learned", "empty", "flipped", "shuffled"}

    by_arm = dict(captured)
    flp_rows = by_arm["control-flipped-all.gguf"]
    assert len(flp_rows) == 2
    assert list(flp_rows[0]["pos_tokens"]) == [1, 2, 5, 6]
    assert list(flp_rows[0]["neg_tokens"]) == [1, 2, 3, 4]

    shf_rows = by_arm["control-shuffled-all.gguf"]
    assert len(shf_rows) == 2
    assert list(shf_rows[0]["pos_tokens"]) == [1, 2, 13, 14]
    assert list(shf_rows[0]["pos_targets"]) == [2, 3]
    assert list(shf_rows[1]["pos_tokens"]) == [10, 11, 12, 3, 4]
    assert list(shf_rows[1]["pos_targets"]) == [3, 4]


def test_iter3_c6_bug3_kl_normalise_and_pipeline_failure_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    hf_p = _make_hf_dir(tmp_path / "hf_c6b3", dialect="qwen3")
    R = kl.Renderer(str(hf_p), reasoning_effort="none")

    with pytest.raises(kl.RenderError):
        kl._normalise(123)
    with pytest.raises(kl.RenderError):
        R.episode(123, None)
    with pytest.raises(kl.RenderError):
        R.render_trajectories([{"messages": 123}])
    with pytest.raises(kl.RenderError):
        R.render_trajectories([123])

    valid_msgs = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "ok"}]
    recs = R.render_trajectories([{"messages": 123}, valid_msgs])
    assert len(recs) == 1

    gguf_p = _write_gguf(tmp_path / "m_c6b3.gguf")
    monkeypatch.setattr(pipeline, "load_environment", lambda mod, **kw: SimpleNamespace(
        name="stub", version="1", classifier=False,
        deployment={"context": 512, "concurrency": 1, "kv_type": "f16", "max_total_memory_bytes": None},
        budget={"max_turns": 1, "max_tokens_per_turn": 64}, tasks=lambda: [],
    ))
    r = pipeline.Run({
        "run_id": "r-c6b3",
        "run_root": str(tmp_path / "runs"),
        "source": {"gguf": str(gguf_p), "repo": "test/r", "revision": "abc", "hf_dir": str(hf_p)},
        "toolchain": {"llama_bin": str(tmp_path), "traj_kl": str(tmp_path / "tk"), "gpus": [""]},
        "env": {"module": "stub.env", "kwargs": {}},
        "ladder": ["BF16", "Q8_0", "Q4_K"],
        "start_tier": "Q4_K",
        "sanity": {"behavioral_items": 0},
    })
    out = r._failure_store([
        {"task": "t0", "family": "f0", "messages": 123},
        {"task": "t1", "family": "f0"},
        "not_a_dict",
    ], "fb_c6b3")
    assert out is None


def test_iter3_c6_bug4_learn_train_strata_none_single_class_and_empty_replay(monkeypatch: pytest.MonkeyPatch):
    torch.manual_seed(0)
    proj = torch.nn.Linear(4, 4, bias=False)
    head = torch.nn.Linear(4, 2, bias=False)
    model = SimpleNamespace(lm_head=head, config=SimpleNamespace(final_logit_softcapping=None))
    params = list(proj.parameters()) + list(head.parameters())

    def fake_vec_scores(m, docs, pair, dev, amp, prefix_cache=None):
        assert len(docs) > 0
        x = torch.stack([torch.tensor(d, dtype=torch.float32)[:4] for d in docs])
        h = proj(x)
        lg = m.lm_head(h)
        return lg[:, 0] - lg[:, 1], h

    def fake_scores(m, docs, pair, dev, amp, prefix_cache=None):
        return fake_vec_scores(m, docs, pair, dev, amp, prefix_cache=prefix_cache)[0]

    monkeypatch.setattr(learn, "scores", fake_scores)
    monkeypatch.setattr(learn, "vec_scores", fake_vec_scores)

    feedback = [
        {"tokens": np.array([1.0, 2.0, 3.0, 4.0]), "label": ("human" if i % 2 == 0 else "ai")}
        for i in range(4)
    ]
    replay = [np.array([0.5, 0.25 * (i + 1), 0.5, 0.5]) for i in range(4)]

    for hk in ("kl", "oneside", "klvec"):
        hist = learn._train(
            model, params, feedback, replay, (0, 1), 0.0,
            {"steps": 2, "feedback_batch": 2, "replay_batch": 2, "train_budget": 0, "amp": False, "hold_kind": hk},
            log=lambda _m: None, device="cpu", amp=False, strata=None, lr=1e-2,
        )
        assert len(hist) == 2 and all(math.isfinite(x) for row in hist for x in row)

    # Single-class strata with klvec and mixed human/ai feedback must not emit NaN centre or FloatingPointError
    hist_sc = learn._train(
        model, params, feedback, replay, (0, 1), 0.0,
        {"steps": 2, "feedback_batch": 4, "replay_batch": 2, "train_budget": 0, "amp": False, "hold_kind": "klvec"},
        log=lambda _m: None, device="cpu", amp=False, strata=["human"] * 4, lr=1e-2,
    )
    assert len(hist_sc) == 2 and all(math.isfinite(x) for row in hist_sc for x in row)

    # Empty replay with replay_batch=2 must not raise ZeroDivisionError
    for hk in (None, "kl", "oneside", "klvec"):
        hist_er = learn._train(
            model, params, feedback, [], (0, 1), 0.0,
            {"steps": 2, "feedback_batch": 2, "replay_batch": 2, "train_budget": 0, "amp": False, "hold_kind": hk},
            log=lambda _m: None, device="cpu", amp=False, strata=None, lr=1e-2,
        )
        assert len(hist_er) == 2 and all(row[2] == 0.0 for row in hist_er)

