"""Milestone 4 deterministic offline regression tests (`pipeline.py`, `plan.py`, `learn.py`, `mvp/tools/*`, `mvp/configs/*`).

All tests run 100% offline in `tmp_path` with zero `.local/` coupling.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import struct
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "mvp") not in sys.path:
    sys.path.insert(0, str(ROOT / "mvp"))

from chantless import (  # noqa: E402
    config as cfgmod,
    gguf_ops,
    kl,
    learn,
    mask as mk,
    pipeline,
    plan,
    runtime,
    sanity,
)
from chantless.env import codetasks, protocol  # noqa: E402
from chantless.env.protocol import Task  # noqa: E402


# =============================================================================
# Offline Synthetic Fixtures & Helpers
# =============================================================================


def _pack_kv_str(s: str) -> tuple[int, bytes]:
    b = s.encode("utf-8")
    return 8, struct.pack("<Q", len(b)) + b


def _pack_kv_u32(v: int) -> tuple[int, bytes]:
    return 4, struct.pack("<I", v)


def _pack_kv_i32_arr(vals: list[int]) -> tuple[int, bytes]:
    return 9, struct.pack("<IQ", 5, len(vals)) + struct.pack(f"<{len(vals)}i", *vals)


def _pack_kv_f32_arr(vals: list[float]) -> tuple[int, bytes]:
    return 9, struct.pack("<IQ", 6, len(vals)) + struct.pack(f"<{len(vals)}f", *vals)


def _pack_kv_str_arr(items: list[bytes]) -> tuple[int, bytes]:
    b = bytearray(struct.pack("<IQ", 8, len(items)))
    for x in items:
        b += struct.pack("<Q", len(x)) + x
    return 9, bytes(b)


def _f32_to_bf16_bytes(arr: np.ndarray) -> bytes:
    f32 = np.asarray(arr, dtype=np.float32)
    return (f32.view(np.uint32) >> 16).astype(np.uint16).tobytes()


_INKLING2_TEMPLATE = (
    "{%- set eff = reasoning_effort | default('high') -%}"
    "{%- set effort_num = '0' if eff == 'none' else ('0.2' if eff == 'low' else '0.9') -%}"
    "{%- if messages and messages[0].role != 'system' -%}"
    "{{- '<|message_system|><|content_text|>Thinking effort level: ' ~ effort_num ~ '<|end_message|>' -}}"
    "{%- endif -%}"
    "{%- for m in messages -%}"
    "{%- if m.role == 'system' -%}"
    "{{- '<|message_system|><|content_text|>' ~ m.content ~ '<|end_message|>' -}}"
    "{{- '<|message_system|><|content_text|>Thinking effort level: ' ~ effort_num ~ '<|end_message|>' -}}"
    "{%- elif m.role == 'user' -%}"
    "{{- '<|message_user|><|content_text|>' ~ m.content ~ '<|end_message|>' -}}"
    "{%- elif m.role == 'assistant' -%}"
    "{{- '<|message_model|><|content_text|>' ~ m.content ~ '<|end_message|><|content_model_end_sampling|>' -}}"
    "{%- endif -%}"
    "{%- endfor -%}"
)

_QWEN3_TEMPLATE = (
    "{%- for m in messages -%}"
    "{{- '<|im_start|>' ~ m.role ~ '\\n' ~ (m.content or '') ~ '<|im_end|>\\n' -}}"
    "{%- endfor -%}"
    "{%- if add_generation_prompt -%}"
    "{{- '<|im_start|>assistant\\n' -}}"
    "{%- endif -%}"
)


def _make_hf_dir(tmp_path: Path, name: str = "hf", dialect: str = "qwen3_json") -> Path:
    hf_dir = tmp_path / name
    hf_dir.mkdir(parents=True, exist_ok=True)
    vocab = {
        "<unk>": 0,
        "<|endoftext|>": 1,
        "<|im_start|>": 2,
        "<|im_end|>": 3,
        "<|message_system|>": 4,
        "<|message_user|>": 5,
        "<|message_model|>": 6,
        "<|content_text|>": 7,
        "<|end_message|>": 8,
        "<|content_model_end_sampling|>": 9,
        "<|content_invoke_tool_json|>": 10,
        " ": 11,
        "\n": 12,
    }
    words = [
        "Thinking", "effort", "level:", "0", "0.2", "0.9", "user", "assistant",
        "system", "hello", "world", "ok", "yes", "no", "positive", "negative",
        "custom_tool", "hi", "test", "done", "1", "2", "3",
    ]
    for w in words:
        if w not in vocab:
            vocab[w] = len(vocab)
    tok_obj = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    tok_obj.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    specials = [
        "<|im_start|>", "<|im_end|>", "<|message_system|>", "<|message_user|>",
        "<|message_model|>", "<|content_text|>", "<|end_message|>",
        "<|content_model_end_sampling|>", "<|content_invoke_tool_json|>",
    ]
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tok_obj,
        bos_token="<|endoftext|>",
        eos_token="<|content_model_end_sampling|>" if dialect == "inkling2" else "<|im_end|>",
        unk_token="<unk>",
        additional_special_tokens=specials,
    )
    fast.chat_template = _INKLING2_TEMPLATE if dialect == "inkling2" else _QWEN3_TEMPLATE
    fast.save_pretrained(str(hf_dir))
    return hf_dir


def _write_gguf(
    path: Path,
    arch: str = "qwen3",
    n_layer: int = 2,
    n_embd: int = 256,
    n_ff: int = 256,
    vocab_size: int = 32,
    extra_kvs: list[tuple[str, tuple[int, bytes]]] | None = None,
    n_expert: int = 0,
    n_expert_used: int = 0,
) -> Path:
    tokens = [b"<unk>", b"<|endoftext|>", b"<|im_start|>", b"<|im_end|>"] + [
        f"tok_{i}".encode("ascii") for i in range(max(0, vocab_size - 4))
    ]
    ttypes = [3, 3, 3, 3] + [1] * (len(tokens) - 4)
    scores = [0.0] * len(tokens)
    kvs: list[tuple[str, tuple[int, bytes]]] = [
        ("general.architecture", _pack_kv_str(arch)),
        ("general.alignment", _pack_kv_u32(32)),
        (f"{arch}.block_count", _pack_kv_u32(n_layer)),
        (f"{arch}.embedding_length", _pack_kv_u32(n_embd)),
        (f"{arch}.attention.head_count", _pack_kv_u32(4)),
        (f"{arch}.attention.head_count_kv", _pack_kv_u32(2)),
        (f"{arch}.attention.key_length", _pack_kv_u32(n_embd // 4)),
        (f"{arch}.feed_forward_length", _pack_kv_u32(n_ff)),
        ("tokenizer.ggml.tokens", _pack_kv_str_arr(tokens)),
        ("tokenizer.ggml.token_type", _pack_kv_i32_arr(ttypes)),
        ("tokenizer.ggml.scores", _pack_kv_f32_arr(scores)),
        ("tokenizer.ggml.bos_token_id", _pack_kv_u32(0)),
        ("tokenizer.ggml.eos_token_id", _pack_kv_u32(1)),
    ]
    if n_expert > 0:
        kvs.append((f"{arch}.expert_count", _pack_kv_u32(n_expert)))
        kvs.append((f"{arch}.expert_used_count", _pack_kv_u32(max(1, n_expert_used))))
    if extra_kvs:
        kvs.extend(extra_kvs)

    rng = np.random.default_rng(7)
    tensors: list[tuple[str, list[int], int, bytes]] = [
        ("token_embd.weight", [n_embd, len(tokens)], 30, _f32_to_bf16_bytes(rng.standard_normal((len(tokens), n_embd)))),
        ("output_norm.weight", [n_embd], 0, np.ones(n_embd, dtype=np.float32).tobytes()),
    ]
    for il in range(n_layer):
        p = f"blk.{il}."
        sq = _f32_to_bf16_bytes(rng.standard_normal((n_embd, n_embd)))
        sf = _f32_to_bf16_bytes(rng.standard_normal((n_embd, n_ff)))
        sd = _f32_to_bf16_bytes(rng.standard_normal((n_ff, n_embd)))
        nrm = np.ones(n_embd, dtype=np.float32).tobytes()
        tensors.extend([
            (p + "attn_norm.weight", [n_embd], 0, nrm),
            (p + "attn_q.weight", [n_embd, n_embd], 30, sq),
            (p + "attn_k.weight", [n_embd, n_embd], 30, sq),
            (p + "attn_v.weight", [n_embd, n_embd], 30, sq),
            (p + "attn_output.weight", [n_embd, n_embd], 30, sq),
            (p + "post_attention_norm.weight", [n_embd], 0, nrm),
        ])
        if n_expert > 0:
            se = _f32_to_bf16_bytes(rng.standard_normal((n_expert, n_ff, n_embd)))
            sde = _f32_to_bf16_bytes(rng.standard_normal((n_expert, n_embd, n_ff)))
            sgate = _f32_to_bf16_bytes(rng.standard_normal((n_expert, n_embd)))
            tensors.extend([
                (p + "ffn_gate_inp.weight", [n_embd, n_expert], 30, sgate),
                (p + "ffn_gate_exps.weight", [n_embd, n_ff, n_expert], 30, se),
                (p + "ffn_up_exps.weight", [n_embd, n_ff, n_expert], 30, se),
                (p + "ffn_down_exps.weight", [n_ff, n_embd, n_expert], 30, sde),
            ])
        else:
            tensors.extend([
                (p + "ffn_gate.weight", [n_embd, n_ff], 30, sf),
                (p + "ffn_up.weight", [n_embd, n_ff], 30, sf),
                (p + "ffn_down.weight", [n_ff, n_embd], 30, sd),
            ])

    kv_buf = bytearray()
    for k, (vt, raw) in kvs:
        kb = k.encode("utf-8")
        kv_buf += struct.pack("<Q", len(kb)) + kb + struct.pack("<I", vt) + raw
    gguf_ops._write_gguf_raw(3, 32, len(kvs), bytes(kv_buf), tensors, path)
    return path


def _make_stub_env(classifier: bool = False, with_sample_tasks: bool = False, reasoning_effort: str = "none"):
    tasks = []
    for sp in ("search", "select", "exam", "train"):
        for fam in ("famA", "famB", "famC"):
            for idx in range(10):
                meta: dict[str, Any] = {"label": "positive" if idx % 2 == 0 else "negative"}
                tasks.append(
                    Task(
                        id=f"{sp}-{fam}-{idx}",
                        family=fam,
                        split=sp,
                        prompt={"system": "sys", "user": f"hello {idx}"},
                        meta=meta,
                    )
                )
    if classifier:
        for idx in range(4):
            tasks.append(Task(id=f"cal-{idx}", family="cal", split="calibration", prompt={"text": f"doc {idx}"}, meta={"label": "human"}))

    class _StubEnv:
        name = "stub-env"
        version = "1"
        deployment = {"context": 512, "concurrency": 1, "kv_type": "f16", "max_total_memory_bytes": None}
        budget = {"max_turns": 1, "max_tokens_per_turn": 64}

        def __init__(self):
            self.classifier = classifier
            self.reasoning_effort = reasoning_effort
            self.sample_calls: list[tuple[str, int, bool]] = []
            if classifier:
                self.pair = (14, 15)

        def tasks(self):
            return list(tasks)

    if with_sample_tasks:
        def sample_tasks(self, split: str, n: int, *, stratified: bool = True):
            self.sample_calls.append((split, n, stratified))
            pool = [t for t in tasks if t.split == split]
            return pool[:n]

        _StubEnv.sample_tasks = sample_tasks  # type: ignore[attr-defined]
    return _StubEnv()


# =============================================================================
# 1–10: pipeline.py & plan.py Tests
# =============================================================================


def test_pipeline_run_evaluator_zero_trust_defaults_and_serving_forwarding(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    gguf_p = _write_gguf(tmp_path / "m.gguf")
    hf_p = _make_hf_dir(tmp_path)

    for is_cls in (False, True):
        monkeypatch.setattr(pipeline, "load_environment", lambda mod, **kw: _make_stub_env(classifier=is_cls))
        base_cfg = {
            "run_id": f"r-{is_cls}",
            "run_root": str(tmp_path / "runs"),
            "source": {"gguf": str(gguf_p), "repo": "test/r", "revision": "abc", "hf_dir": str(hf_p)},
            "toolchain": {"llama_bin": str(tmp_path), "traj_kl": str(tmp_path / "tk"), "logodds": str(tmp_path / "lo"), "gpus": [""]},
            "env": {"module": "stub.env", "kwargs": {}},
            "ladder": ["BF16", "Q8_0", "Q4_K"],
            "start_tier": "Q4_K",
            "decoding": {"temperature": 0.0},
            "serving": {"slots_per_gpu": 2, "ctx_per_slot": 512},
            "sanity": {"behavioral_items": 0},
            "scoring": None,
        }
        r_def = pipeline.Run(base_cfg)
        assert r_def.ev.cache_prompt is False
        assert r_def.ev.flash_attn == "on"
        assert r_def.ev._special_override is None

        exp_cfg = {
            **base_cfg,
            "run_id": f"r-exp-{is_cls}",
            "serving": {"slots_per_gpu": 2, "ctx_per_slot": 512, "cache_prompt": True, "flash_attn": "off", "special": True},
        }
        r_exp = pipeline.Run(exp_cfg)
        assert r_exp.ev.cache_prompt is True
        assert r_exp.ev.flash_attn == "off"
        assert r_exp.ev._special_override is True


def test_pipeline_measure_memory_forwards_cache_prompt_false_and_flash_attn(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    gguf_p = _write_gguf(tmp_path / "ink.gguf", arch="inkling2_loop")
    srv_calls: list[dict[str, Any]] = []
    client_calls: list[dict[str, Any]] = []

    class FakeLlamaServer:
        def __init__(self, tc, model, **kw):
            srv_calls.append(kw)
            self.url = "http://127.0.0.1:0"
            log = Path(kw["log"])
            log.parent.mkdir(parents=True, exist_ok=True)
            log.write_text("model buffer size = 4.0 MiB\nllama_kv_cache: size = 1.0 MiB\ncompute buffer size = 0.5 MiB\n")

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    class FakeChatClient:
        def __init__(self, url, decoding, **kw):
            client_calls.append(kw)

        def chat(self, msgs, **kw):
            return {"message": {"content": "1 2"}, "completion_tokens": 2}

    monkeypatch.setattr(pipeline, "LlamaServer", FakeLlamaServer)
    import chantless.client as client_mod
    monkeypatch.setattr(client_mod, "ChatClient", FakeChatClient)

    r_obj = object.__new__(pipeline.Run)
    r_obj.dir = tmp_path / "run"
    (r_obj.dir / "final").mkdir(parents=True, exist_ok=True)
    r_obj.tc = runtime.Toolchain(tmp_path, tmp_path / "tk", [""])
    r_obj.dep = {"concurrency": 1, "context": 256, "kv_type": "f16"}
    r_obj.info = {"arch": "inkling2_loop"}
    r_obj.cfg = {"serving": {"flash_attn": "off", "special": False}}
    r_obj.ev = SimpleNamespace(dialect="inkling2", _special_override=False, special=False, flash_attn="off")

    r_obj.measure_memory(str(gguf_p))
    assert srv_calls[-1]["cache_prompt"] is False
    assert srv_calls[-1]["flash_attn"] == "off"
    assert srv_calls[-1]["special"] is False
    assert client_calls[-1]["cache_prompt"] is False


def test_pipeline_kl_renderer_reasoning_effort_and_episode_tools(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    hf_p = _make_hf_dir(tmp_path, dialect="inkling2")
    gguf_p = _write_gguf(tmp_path / "m.gguf", arch="inkling2")
    env_obj = _make_stub_env(classifier=False, reasoning_effort="low")
    monkeypatch.setattr(pipeline, "load_environment", lambda mod, **kw: env_obj)

    cfg = {
        "run_id": "r-kl",
        "run_root": str(tmp_path / "runs"),
        "source": {"gguf": str(gguf_p), "repo": "test/r", "revision": "abc", "hf_dir": str(hf_p)},
        "toolchain": {"llama_bin": str(tmp_path), "traj_kl": str(tmp_path / "tk"), "logodds": str(tmp_path / "lo"), "gpus": [""]},
        "env": {"module": "stub.env", "kwargs": {}},
        "ladder": ["BF16", "Q8_0", "Q4_K"],
        "start_tier": "Q4_K",
        "decoding": {"temperature": 1.0, "reasoning_effort": "low"},
        "serving": {"slots_per_gpu": 2, "ctx_per_slot": 512, "cache_prompt": False},
        "proxy": {"general_docs": 2, "general_seq_len": 16, "general_n_seq": 1, "self_kl_tol": 1e-3},
        "sanity": {"behavioral_items": 0},
    }
    raw = cfgmod.ExtractConfig.model_validate(cfg).to_raw_dict()
    assert "task_max_targets" not in raw["proxy"]  # excluded when None

    r = pipeline.Run(raw)
    (r.dir / "episodes").mkdir(parents=True, exist_ok=True)
    ep_tools = [{"type": "function", "function": {"name": "ep_specific_tool", "description": "d", "parameters": {"type": "object", "properties": {}}}}]
    (r.dir / "pool.json").write_text(json.dumps({"reliable": {"search": ["search-famA-0"]}}))
    (r.dir / "episodes" / "pool.jsonl").write_text(
        json.dumps({
            "task": "search-famA-0",
            "split": "search",
            "success": True,
            "tools": ep_tools,
            "messages": [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "ok"},
            ],
        }) + "\n"
    )

    seen_tools: list[Any] = []
    orig_ep = kl.Renderer.episode

    def wrapped_ep(self_r, msgs, tools):
        seen_tools.append(tools)
        assert self_r.reasoning_effort == "low"
        return orig_ep(self_r, msgs, tools)

    def fake_traj_kl(tc, mode, model, trj, store, out, log, **kw):
        res = {"kl": 0.0, "n_tgt": 1, "trajs": []}
        Path(out).write_text(json.dumps(res))
        return res

    monkeypatch.setattr(kl.Renderer, "episode", wrapped_ep)
    monkeypatch.setattr(sanity, "general_texts", lambda *a, **kw: ["hello world ok yes no " * 10])
    monkeypatch.setattr(kl, "traj_kl", fake_traj_kl)
    monkeypatch.setattr(pipeline, "run_imatrix", lambda *a, **kw: None)
    monkeypatch.setattr(pipeline, "sha256_file", lambda p: "deadbeef")
    monkeypatch.setattr(r, "annotate_imatrix", lambda p: None)

    r.phase_proxy()
    assert seen_tools == [ep_tools]

    # When 0 trajectories render, phase_proxy must raise kl.RenderError cleanly
    (r.dir / "episodes" / "pool.jsonl").write_text("")
    with pytest.raises(kl.RenderError):
        r.phase_proxy()


def test_pipeline_tools_helper_never_attaches_codetasks_to_unrelated_env():
    class UnrelatedEnv:
        pass

    class CustomToolEnv:
        tools = [{"type": "function", "function": {"name": "custom"}}]

    assert pipeline._tools(UnrelatedEnv()) is None
    assert pipeline._tools(CustomToolEnv()) == CustomToolEnv.tools
    ct = codetasks.CodeTasksEnv(variants_per_family=1)
    assert pipeline._tools(ct) == codetasks.TOOLS


def test_pipeline_sampling_delegates_to_env_sample_tasks_and_stratifies_fallbacks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    gguf_p = _write_gguf(tmp_path / "m.gguf")
    hf_p = _make_hf_dir(tmp_path)

    # 1) Env WITH sample_tasks: phase_pool & phase_exam delegate, SplitGuard enforced
    env_s = _make_stub_env(classifier=False, with_sample_tasks=True)
    monkeypatch.setattr(pipeline, "load_environment", lambda mod, **kw: env_s)
    cfg = {
        "run_id": "r-samp",
        "run_root": str(tmp_path / "runs"),
        "source": {"gguf": str(gguf_p), "repo": "test/r", "revision": "abc", "hf_dir": str(hf_p)},
        "toolchain": {"llama_bin": str(tmp_path), "traj_kl": str(tmp_path / "tk"), "logodds": str(tmp_path / "lo"), "gpus": [""]},
        "env": {"module": "stub.env", "kwargs": {}},
        "ladder": ["BF16", "Q8_0", "Q4_K"],
        "start_tier": "Q4_K",
        "pool": {"tasks_per_split": {"search": 6, "select": 6, "exam": 6}, "runs": 1, "min_pass": 1},
        "sanity": {"behavioral_items": 0},
    }
    r = pipeline.Run(cfg)
    with pytest.raises(PermissionError):
        r._sample_split("exam", 6)
    monkeypatch.setattr(r.ev, "run", lambda m, jobs, tag, side, srvs=None: [
        {"task": t.id, "family": t.family, "split": t.split, "seed": s, "side": side, "success": True}
        for t, s in jobs
    ])
    r.guard.phase = "pool"
    r.phase_pool()
    assert ("search", 6, True) in env_s.sample_calls
    assert ("select", 6, True) in env_s.sample_calls

    # 2) Env WITHOUT sample_tasks: fallback stratifies across families (not [:32] prefix)
    env_ns = _make_stub_env(classifier=False, with_sample_tasks=False)
    monkeypatch.setattr(pipeline, "load_environment", lambda mod, **kw: env_ns)
    r2 = pipeline.Run({**cfg, "run_id": "r-samp-fb"})
    r2.guard.phase = "pool"
    sel_pool = r2._sample_split("search", 6)
    assert {t.family for t in sel_pool} == {"famA", "famB", "famC"}

    # screen() stratified over reliable("select")
    (r2.dir / "pool.json").write_text(json.dumps({
        "reliable": {"select": [t.id for t in env_ns.tasks() if t.split == "select"]}
    }))
    (r2.dir / "episodes").mkdir(parents=True, exist_ok=True)
    (r2.dir / "episodes" / "aa-B.jsonl").write_text("")
    (r2.dir / "behavioral_bank.json").write_text("[]")
    seen_screen_tasks: list[Task] = []

    class _DummyCtx:
        def __enter__(self):
            return []
        def __exit__(self, *a):
            return False

    monkeypatch.setattr(r2.ev, "servers", lambda m: _DummyCtx())
    def fake_ev_run(m, jobs, tag, side, srvs=None):
        seen_screen_tasks.extend(t for t, _ in jobs)
        return [{"task": t.id, "family": t.family, "split": t.split, "seed": s, "side": side, "success": True} for t, s in jobs]
    monkeypatch.setattr(r2.ev, "run", fake_ev_run)
    r2.cfg["screen"] = {"pairs": 6, "fail_below": -0.5}
    r2.screen(str(gguf_p), "t1")
    assert {t.family for t in seen_screen_tasks} == {"famA", "famB", "famC"}


@pytest.mark.parametrize("bad_val", [float("nan"), float("inf"), float("-inf")])
def test_pipeline_kl_gates_reject_nan_inf_and_handle_float_self_kl_tol(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, bad_val: float):
    hf_p = _make_hf_dir(tmp_path)
    gguf_p = _write_gguf(tmp_path / "m.gguf")
    env_obj = _make_stub_env(classifier=False)
    monkeypatch.setattr(pipeline, "load_environment", lambda mod, **kw: env_obj)

    cfg = {
        "run_id": f"r-nan-{bad_val}",
        "run_root": str(tmp_path / "runs"),
        "source": {"gguf": str(gguf_p), "repo": "test/r", "revision": "abc", "hf_dir": str(hf_p)},
        "toolchain": {"llama_bin": str(tmp_path), "traj_kl": str(tmp_path / "tk"), "logodds": str(tmp_path / "lo"), "gpus": [""]},
        "env": {"module": "stub.env", "kwargs": {}},
        "ladder": ["BF16", "Q8_0", "Q4_K"],
        "start_tier": "Q4_K",
        "proxy": {"general_docs": 2, "general_seq_len": 16, "general_n_seq": 1, "self_kl_tol": 1e-3, "floor_tier": "Q4_K"},
        "sanity": {"behavioral_items": 0},
    }
    r = pipeline.Run(cfg)
    (r.dir / "episodes").mkdir(parents=True, exist_ok=True)
    (r.dir / "pool.json").write_text(json.dumps({"reliable": {"search": ["search-famA-0"]}}))
    (r.dir / "episodes" / "pool.jsonl").write_text(
        json.dumps({"task": "search-famA-0", "split": "search", "success": True,
                    "messages": [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "ok"}]}) + "\n"
    )
    monkeypatch.setattr(sanity, "general_texts", lambda *a, **kw: ["hello world ok yes no " * 10])
    monkeypatch.setattr(kl, "traj_kl", lambda *a, **kw: {"kl": bad_val, "n_tgt": 1, "trajs": []})
    with pytest.raises(RuntimeError):
        r.phase_proxy()

    # _prepass_dead_tensors rejects NaN/Inf
    r.classifier = True
    r.scorer = object()
    r.dead_tensors = ["dummy"]
    r.env.score = lambda sample, m, sc: {sample[0].id: 1.0 if m == r.src else bad_val}
    monkeypatch.setattr(gguf_ops, "copy_without", lambda src, dst, drop: Path(dst).write_bytes(b"stub"))
    with pytest.raises(RuntimeError):
        r._prepass_dead_tensors()

    # phase_sanity selfkl zero-guard: ref == 0.0 with 1e-12 passes, bad_val raises
    (r.dir / "proxy").mkdir(parents=True, exist_ok=True)
    (r.dir / "proxy" / "proxy.json").write_text(json.dumps({
        "task": {"n_targets": 1}, "general": {"n_targets": 1},
        "self_kl": {"task": 1e-12, "general": 1e-12}, "imatrix_sha256": "abc",
    }))
    (r.dir / "proxy" / "task.bin").write_bytes(b"stub")
    (r.dir / "proxy" / "general.bin").write_bytes(b"stub")
    (r.dir / "proxy" / "imatrix.gguf").write_bytes(b"stub")
    (r.dir / "feedback_stores.jsonl").write_text("")
    (r.dir / "run.json").write_text(json.dumps({"pins": {"source_gguf_sha256": "abc"}}))

    class FakeProber:
        def __init__(self):
            self.workers = 1
            self.builder = SimpleNamespace(source_params=1000, cleanup=lambda: None)
        def probe_many(self, masks, stores):
            return [SimpleNamespace(kl={s: 0.0 for s in stores}, bytes=500, bpw=4.0) for _ in masks]

    monkeypatch.setattr(r, "make_prober", lambda w, tag: FakeProber())
    r.classifier = False
    r.cfg["sanity"]["solo_tiers"] = ["Q4_K"]
    r.cfg["sanity"]["floor_key"] = "task"
    r.cfg["sanity"]["cap"] = "off"
    r.cfg["probe"] = {"splice_cache": False}
    r.phase_sanity()  # ref == 0.0, selfkl == 1e-12 <= 1e-6 -> passes

    (r.dir / "proxy" / "proxy.json").write_text(json.dumps({
        "task": {"n_targets": 1}, "general": {"n_targets": 1},
        "self_kl": {"task": bad_val, "general": 0.0}, "imatrix_sha256": "abc",
    }))
    with pytest.raises((RuntimeError, ValueError)):
        r.phase_sanity()


def test_pipeline_oneshot_blocks_unprobed_q8_0(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    gguf_p = _write_gguf(tmp_path / "m.gguf")
    hf_p = _make_hf_dir(tmp_path)
    monkeypatch.setattr(pipeline, "load_environment", lambda mod, **kw: _make_stub_env(classifier=False))
    cfg = {
        "run_id": "r-q8",
        "run_root": str(tmp_path / "runs"),
        "source": {"gguf": str(gguf_p), "repo": "test/r", "revision": "abc", "hf_dir": str(hf_p)},
        "toolchain": {"llama_bin": str(tmp_path), "traj_kl": str(tmp_path / "tk"), "logodds": str(tmp_path / "lo"), "gpus": [""]},
        "env": {"module": "stub.env", "kwargs": {}},
        "ladder": ["BF16", "Q8_0", "Q4_K"],
        "start_tier": "Q4_K",
        "sanity": {"solo_tiers": ["Q4_K"], "behavioral_items": 0},
    }
    r = pipeline.Run(cfg)
    (r.dir / "run.json").write_text(json.dumps({"pins": {"source_gguf_sha256": "a"}}))
    base = mk.new_mask(r.groups, "BF16", {"repo": "r", "revision": "1", "gguf_sha256": "a"}, "preserve")
    (r.dir / "sanity").mkdir(parents=True, exist_ok=True)
    (r.dir / "sanity" / "uniform.json").write_text(json.dumps({"Q4_K": {"bytes": 1000}}))
    (r.dir / "probe" / "sanity").mkdir(parents=True, exist_ok=True)
    for g in base["groups"]:
        k_q4 = pipeline.bmod.mask_key(mk.apply(base, g, "Q4_K"))
        (r.dir / "probe" / "sanity" / f"p-{k_q4}-task.json").write_text(
            json.dumps({"trajs": [{"kl": 0.01, "n_tgt": 2, "nll_src": 1.0, "nll_cand": 1.01}]})
        )
    P = SimpleNamespace(
        work=r.dir / "probe" / "sanity",
        stores=[SimpleNamespace(name="task", kind="kl", weight=1.0, meta={}, dir=str(r.dir / "proxy" / "store_task"))],
        builder=SimpleNamespace(source_params=1000, groups=r.groups, cleanup=lambda: None),
    )
    lc = {"algo": "oneshot", "target_bytes": 500, "oneshot_tiers": ["Q8_0"], "allow_drop": False}
    with pytest.raises((ValueError, RuntimeError)):
        r._search_oneshot(P, base, lc, cap=sanity.NoCap())


def test_pipeline_expert_and_wanda_order_caches_validate_imatrix_sha256(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    gguf_p = _write_gguf(tmp_path / "moe.gguf", arch="qwen3moe", n_layer=1, n_embd=256, n_ff=256, n_expert=4, n_expert_used=2)
    hf_p = _make_hf_dir(tmp_path)
    monkeypatch.setattr(pipeline, "load_environment", lambda mod, **kw: _make_stub_env(classifier=False))
    cfg = {
        "run_id": "r-cache",
        "run_root": str(tmp_path / "runs"),
        "source": {"gguf": str(gguf_p), "repo": "test/r", "revision": "abc", "hf_dir": str(hf_p)},
        "toolchain": {"llama_bin": str(tmp_path), "traj_kl": str(tmp_path / "tk"), "logodds": str(tmp_path / "lo"), "gpus": [""]},
        "env": {"module": "stub.env", "kwargs": {}},
        "ladder": ["BF16", "Q8_0", "Q4_K"],
        "start_tier": "Q4_K",
        "sanity": {"solo_tiers": ["Q4_K"], "behavioral_items": 0},
    }
    r = pipeline.Run(cfg)
    (r.dir / "run.json").write_text(json.dumps({"pins": {"source_gguf_sha256": "a"}}))
    im_p = r.dir / "proxy" / "imatrix.gguf"
    im_p.parent.mkdir(parents=True, exist_ok=True)
    im_p.write_bytes(b"IMATRIX_V1")

    base = mk.new_mask(r.groups, "BF16", {"repo": "r", "revision": "1", "gguf_sha256": "a"}, "preserve")
    (r.dir / "sanity").mkdir(parents=True, exist_ok=True)
    (r.dir / "sanity" / "uniform.json").write_text(json.dumps({"Q4_K": {"bytes": 1_000_000}}))
    (r.dir / "probe" / "sanity").mkdir(parents=True, exist_ok=True)
    for g in base["groups"]:
        k_q4 = pipeline.bmod.mask_key(mk.apply(base, g, "Q4_K"))
        (r.dir / "probe" / "sanity" / f"p-{k_q4}-task.json").write_text(
            json.dumps({"trajs": [{"kl": 0.01, "n_tgt": 2, "nll_src": 1.0, "nll_cand": 1.01}]})
        )

    calls = {"exp": 0, "wanda": 0}
    monkeypatch.setattr(gguf_ops, "expert_importance_from_imatrix", lambda s, im: (calls.__setitem__("exp", calls["exp"] + 1) or {0: np.array([0, 1, 2, 3], dtype=np.int64)}))
    monkeypatch.setattr(gguf_ops, "wanda_order", lambda s, im, kept_experts=None: (calls.__setitem__("wanda", calls["wanda"] + 1) or {0: np.arange(256, dtype=np.int64)}))

    def fake_probe(m, stores=None):
        k = pipeline.bmod.mask_key(m)
        for sn in (stores or ["task"]):
            (r.dir / "probe" / "sanity" / f"p-{k}-{sn}.json").write_text(
                json.dumps({"trajs": [{"kl": 0.02, "n_tgt": 2, "nll_src": 1.0, "nll_cand": 1.02}]})
            )
        return SimpleNamespace(bytes=100, bpw=4.0, key=k, kl={"task": 0.02, "general": 0.0})

    P = SimpleNamespace(
        work=r.dir / "probe" / "sanity",
        stores=[SimpleNamespace(name="task", kind="kl", weight=1.0, meta={}, dir=str(r.dir / "proxy" / "store_task"))],
        builder=SimpleNamespace(source_params=1000, groups=r.groups, cleanup=lambda: None, build=lambda m, out=None, keep_pruned=False: SimpleNamespace(path=str(out), file_bytes=100, bpw=4.0)),
        probe=fake_probe,
        probe_many=lambda ms, stores=None: [fake_probe(m, stores) for m in ms],
    )
    from chantless import oneshot as oneshot_mod
    monkeypatch.setattr(oneshot_mod, "solve", lambda sur, opts, cost, budget, hdr, **kw: {g: opts[g][0] for g in opts})
    monkeypatch.setattr(pipeline.bmod, "Builder", lambda *a, **kw: P.builder)

    lc = {"algo": "oneshot", "target_bytes": 10**9, "oneshot_tiers": ["Q4_K"], "experts": True, "expert_counts": [2, 4], "widths": True, "width_fractions": [0.5], "width_step": 128, "allow_drop": False}
    r._search_oneshot(P, base, lc, cap=sanity.NoCap())
    assert calls["exp"] == 1 and calls["wanda"] == 2

    # Same imatrix -> cache hit (counts unchanged)
    r._search_oneshot(P, base, lc, cap=sanity.NoCap())
    assert calls["exp"] == 1 and calls["wanda"] == 2

    # Mutate imatrix -> SHA256 mismatch invalidates caches
    im_p.write_bytes(b"IMATRIX_V2_DIFFERENT")
    r._search_oneshot(P, base, lc, cap=sanity.NoCap())
    assert calls["exp"] == 2 and calls["wanda"] == 4


def test_pipeline_want_solo_and_allow_drop_enforce_general_cap_wiring(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    gguf_p = _write_gguf(tmp_path / "m.gguf")
    hf_p = _make_hf_dir(tmp_path)
    monkeypatch.setattr(pipeline, "load_environment", lambda mod, **kw: _make_stub_env(classifier=False))

    cfg = {
        "run_id": "r-wiring",
        "run_root": str(tmp_path / "runs"),
        "source": {"gguf": str(gguf_p), "repo": "test/r", "revision": "abc", "hf_dir": str(hf_p)},
        "toolchain": {"llama_bin": str(tmp_path), "traj_kl": str(tmp_path / "tk"), "logodds": str(tmp_path / "lo"), "gpus": [""]},
        "env": {"module": "stub.env", "kwargs": {}},
        "ladder": ["BF16", "Q8_0", "Q4_K"],
        "start_tier": "Q4_K",
        "proxy": {"floor_tier": "Q4_K"},
        "sanity": {"solo_tiers": ["Q4_K"], "floor_key": "task", "cap": "general_kl", "behavioral_items": 0, "allow_drop": False, "floor_factor": 4.0},
        "learner": {"algo": "oneshot", "target_bytes": 10**9},
        "probe": {"want_solo": ["task", "general"], "solo_task_tiers": ["Q4_K"], "splice_cache": False},
    }
    r = pipeline.Run(cfg)
    (r.dir / "proxy").mkdir(parents=True, exist_ok=True)
    (r.dir / "proxy" / "proxy.json").write_text(json.dumps({"task": {"n_targets": 1}, "general": {"n_targets": 1}, "self_kl": {"task": 0.0, "general": 0.0}, "imatrix_sha256": "a"}))
    (r.dir / "feedback_stores.jsonl").write_text("")
    (r.dir / "run.json").write_text(json.dumps({"pins": {"source_gguf_sha256": "a"}}))

    probed_calls: list[tuple[int, list[str]]] = []

    class FakeProber:
        def __init__(self):
            self.workers = 1
            self.builder = SimpleNamespace(source_params=1000, cleanup=lambda: None)
        def probe_many(self, masks, stores):
            probed_calls.append((len(masks), list(stores)))
            return [SimpleNamespace(kl={s: 0.01 for s in stores}, bytes=500, bpw=4.0) for _ in masks]

    monkeypatch.setattr(r, "make_prober", lambda w, tag: FakeProber())
    r.phase_sanity()
    # 1 uniform batch + 1 task+general batch (solo_task_tiers=['Q4_K']), NO drop solo probes because allow_drop=False
    n_groups = len([g for g in r.groups if not g.startswith("_")])
    assert probed_calls[1] == (n_groups, ["general", "task"])

    # Invalid raw want_solo without "task" when floor_key=="task" raises ValueError
    r.cfg["probe"]["want_solo"] = ["general"]
    with pytest.raises(ValueError):
        r.phase_sanity()

    # Conflicting sanity.allow_drop != learner.allow_drop raises ValueError
    r.cfg["probe"]["want_solo"] = ["task", "general"]
    r.cfg["learner"]["allow_drop"] = True
    with pytest.raises(ValueError):
        r.phase_sanity()


def test_plan_estimate_none_fields_safety(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    gguf_p = _write_gguf(tmp_path / "m.gguf")
    for is_cls in (False, True):
        monkeypatch.setattr(protocol, "load_environment", lambda mod, **kw: _make_stub_env(classifier=is_cls))
        cfg = {
            "run_id": f"plan-{is_cls}",
            "run_root": str(tmp_path / "runs"),
            "source": {"gguf": str(gguf_p), "repo": "test/r", "revision": "abc"},
            "toolchain": {"llama_bin": str(tmp_path), "traj_kl": str(tmp_path / "tk"), "logodds": str(tmp_path / "lo"), "gpus": [""]},
            "env": {"module": "stub.env", "kwargs": {"classifier": is_cls}},
            "ladder": ["BF16", "Q8_0", "Q4_K"],
            "start_tier": "Q4_K",
            "proxy": {"task_max_targets": None, "spec": None, "general_seq_len": None, "general_n_seq": None},
            "learner": {"gate_every": None, "max_moves": None, "target_bytes": None, "screen_every": None, "top_k": None},
            "stats": {"n_floor": None, "max_eps_per_task": None, "runs": None, "n_aa": None},
            "pool": {"tasks_per_split": {"search": 4, "select": 4, "exam": 4}, "runs": None},
            "serving": {"slots_per_gpu": None},
            "budget": {"max_hours": None},
            "exam": None,
            "screen": None,
            "sanity": None,
        }
        est = plan.estimate(cfg, source=str(gguf_p))
        assert est["totals"]["prefill_tokens"] > 0


# =============================================================================
# 11–16: learn.py Tests
# =============================================================================


def test_learn_kl_vector_applies_final_logit_softcapping():
    torch.manual_seed(0)
    head = torch.nn.Linear(4, 6, bias=False)
    with torch.no_grad():
        head.weight.mul_(10.0)
    model = SimpleNamespace(config=SimpleNamespace(final_logit_softcapping=5.0), lm_head=head)
    h = torch.randn(3, 4)
    target_h = torch.randn(3, 4)

    got = learn.kl_vector(model, h, target_h, temp=2.0)
    lq = torch.log_softmax(learn._head_logits(model, h) / 2.0, dim=-1)
    lt = torch.log_softmax(learn._head_logits(model, target_h) / 2.0, dim=-1)
    expected = (lt.exp() * (lt - lq)).sum(dim=-1).mean() * 4.0
    assert torch.allclose(got, expected, atol=1e-6)

    raw_lq = torch.log_softmax(head(h) / 2.0, dim=-1)
    raw_lt = torch.log_softmax(head(target_h) / 2.0, dim=-1)
    raw_kl = (raw_lt.exp() * (raw_lt - raw_lq)).sum(dim=-1).mean() * 4.0
    assert abs(float(got.item()) - float(raw_kl.item())) > 1e-2

    with pytest.raises(ValueError):
        learn.kl_vector(model, h, target_h, temp=0.0)


def test_learn_select_limit_is_stratified_and_preserves_multi_family(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    # Ensure side_seed works deterministically
    assert learn._side("task-1", seed=None) == learn._side("task-1")
    diffs = sum(learn._side(f"t-{i}", seed=42) != learn._side(f"t-{i}", seed=99) for i in range(40))
    assert diffs > 0

    sel_tasks_all: list[Task] = []
    for fam in ("famA", "famB", "famC"):
        kept = 0
        for idx in range(50):
            tid = f"{fam}-{idx}"
            if learn._side(tid) == 1:
                sel_tasks_all.append(Task(id=tid, family=fam, split="select", prompt="p", meta={}))
                kept += 1
                if kept >= 10:
                    break
    train_tasks = [Task(id=f"tr-{i}", family="famA", split="train", prompt="p", meta={}) for i in range(6)]
    all_tasks = sel_tasks_all + train_tasks

    class MockEnv:
        deployment = {"context": 512, "concurrency": 1, "kv_type": "f16"}
        def tasks(self):
            return list(all_tasks)

    captured_sel: list[Task] = []
    monkeypatch.setattr("chantless.env.protocol.load_environment", lambda mod, **kw: MockEnv())
    monkeypatch.setattr(learn, "_tokenize_seq_lessons", lambda tc, m, rows, **kw: rows)
    monkeypatch.setattr(learn, "_load_replay_trajs", lambda tc, m, sp, **kw: [])
    monkeypatch.setattr(learn, "learn_gates_seq", lambda *a, **kw: (Path(a[4]).write_bytes(b"G"), {"off": 1})[1])
    monkeypatch.setattr(learn, "sha256", lambda p: "0")
    monkeypatch.setattr(kl, "write_trj", lambda p, recs: {"n_seq": len(recs), "n_targets": len(recs)})
    monkeypatch.setattr(kl, "traj_kl", lambda *a, **kw: {"trajs": [{"nll_cand": 1.0}]})

    class FakeEv:
        def __init__(self, *a, **kw):
            pass
        def run(self, model, jobs, tag, side):
            if tag == "learn-ref-sel":
                captured_sel.extend(t for t, _ in jobs)
            return [{"task": t.id, "family": t.family, "split": t.split, "seed": s, "side": side, "success": True} for t, s in jobs]

    import chantless.evaluator as evmod
    monkeypatch.setattr(evmod, "Evaluator", FakeEv)

    model_p = tmp_path / "m.gguf"
    model_p.write_bytes(b"stub")
    cfg = {
        "toolchain": {"llama_bin": str(tmp_path), "traj_kl": str(tmp_path / "tk"), "gpus": [""]},
        "env": {"module": "mock.env", "kwargs": {}},
        "learn": {
            "mode": "generative",
            "method": "gates_seq",
            "model": str(model_p),
            "out_dir": str(tmp_path / "out"),
            "lessons": [{"pos_tokens": [1, 2, 3], "pos_targets": [1, 2]}],
            "controls": False,
            "select_limit": 6,
            "heldout_ids": ["tr-0"],
        },
    }
    learn.run_generative(cfg)
    assert len(captured_sel) == 6
    assert {t.family for t in captured_sel} == {"famA", "famB", "famC"}


def test_learn_bpe_boundary_logging_and_all_dropped_replay_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        runtime,
        "tokenize_docs",
        lambda tc, m, strs, threads=8: [
            np.array(x, dtype=np.int32)
            for x in (
                # For _tokenize_seq_lessons (5 items): pr, pr+good, good_raw, pr+bad, bad_raw
                ([1, 2, 3], [1, 2, 99, 4], [88, 4], [1, 2, 98, 5], [77, 5])
                if len(strs) == 5
                # For _load_replay_trajs (4 items = 2 trajectories: 1st clean, 2nd BPE mismatch)
                else ([1, 2, 3], [1, 2, 3, 4], [1, 2, 3], [1, 2, 99, 4])
                if len(strs) == 4
                # For _load_replay_trajs (2 items = 1 trajectory: all dropped)
                else ([1, 2, 3], [1, 2, 99, 4])
            )
        ],
    )
    logs_l: list[str] = []
    out_l = learn._tokenize_seq_lessons(None, "m.gguf", [{"prompt": "p", "corrected": "g", "mistaken": "b"}], log=logs_l.append)
    assert out_l[0]["corrected_tokens"] == [88, 4]
    assert any("BPE" in msg or "boundary" in msg.lower() for msg in logs_l)

    logs_r: list[str] = []
    kept = learn._load_replay_trajs(None, "m.gguf", [{"prompt": "p", "completion": "c1"}, {"prompt": "p", "completion": "c2"}], log=logs_r.append)
    assert len(kept) == 1
    assert any("1/2" in msg or "drop" in msg.lower() for msg in logs_r)

    with pytest.raises(ValueError):
        learn._load_replay_trajs(None, "m.gguf", [{"prompt": "p", "completion": "c"}], log=logs_r.append)


def test_learn_gates_seq_raises_on_nonfinite_loss(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    class FakeArt:
        def __init__(self, p):
            pass
        def close(self):
            pass

    monkeypatch.setattr(learn.lattice, "Artifact", FakeArt)
    dummy_model = SimpleNamespace(
        model=SimpleNamespace(layers=[SimpleNamespace()]),
        lm_head=torch.nn.Linear(4, 8, bias=False),
        config=SimpleNamespace(final_logit_softcapping=None),
    )
    monkeypatch.setattr(learn, "build_student", lambda *a, **kw: (dummy_model, None))
    monkeypatch.setattr(learn, "_switchable_layers", lambda *a, **kw: [0])
    monkeypatch.setattr(learn, "_layer_ffn_width", lambda art, il: 4)
    monkeypatch.setattr(learn, "_switch_hook_module", lambda layer: SimpleNamespace(register_forward_pre_hook=lambda fn: SimpleNamespace(remove=lambda: None)))
    monkeypatch.setattr(learn.PrefixCache, "build", staticmethod(lambda *a, **kw: None))
    monkeypatch.setattr(learn, "_forward_backbone", lambda model, seqs, dev, amp, **kw: torch.full((len(seqs), 4, 4), float("nan"), requires_grad=True))

    out_p = tmp_path / "out.gguf"
    with pytest.raises(FloatingPointError):
        learn.learn_gates_seq(
            str(tmp_path / "in.gguf"),
            None,
            [{"pos_tokens": [1, 2, 3], "pos_targets": [1, 2]}],
            [],
            str(out_p),
            cfg={"steps": 1, "amp": False, "frozen_bf16": False, "stream_prefix": False},
            device="cpu",
        )
    assert not out_p.exists()
    assert not Path(str(out_p) + ".lineage.json").exists()


def test_learn_run_generative_evaluator_flags_and_infra_error_gating(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    class MockEnv:
        deployment = {"context": 512, "concurrency": 1, "kv_type": "f16"}
        def tasks(self):
            return [
                Task(id="tr-0", family="famA", split="train", prompt="p", meta={}),
                Task(id="sel-0", family="famA", split="select", prompt="p", meta={}),
            ] + [Task(id=f"sel-{i}", family="famA", split="select", prompt="p", meta={}) for i in range(1, 20)]

    ev_inits: list[dict[str, Any]] = []

    class FakeEv:
        def __init__(self, tc, env, decoding, slots, ctx, kv, log_dir, **kw):
            ev_inits.append({"decoding": decoding, **kw})
        def run(self, model, jobs, tag, side):
            return []  # 100% infra error

    import chantless.evaluator as evmod
    monkeypatch.setattr("chantless.env.protocol.load_environment", lambda mod, **kw: MockEnv())
    monkeypatch.setattr(evmod, "Evaluator", FakeEv)
    monkeypatch.setattr(learn, "_tokenize_seq_lessons", lambda tc, m, rows, **kw: rows)
    monkeypatch.setattr(learn, "_load_replay_trajs", lambda tc, m, sp, **kw: [])
    monkeypatch.setattr(learn, "learn_gates_seq", lambda *a, **kw: (Path(a[4]).write_bytes(b"G"), {"off": 1})[1])
    monkeypatch.setattr(learn, "sha256", lambda p: "0")
    monkeypatch.setattr(kl, "write_trj", lambda p, recs: {"n_seq": len(recs), "n_targets": len(recs)})
    monkeypatch.setattr(kl, "traj_kl", lambda *a, **kw: {"trajs": [{"nll_cand": 1.0 if "base" in a[6] else 0.5}] * 30})

    model_p = tmp_path / "m.gguf"
    model_p.write_bytes(b"stub")
    base_cfg = {
        "toolchain": {"llama_bin": str(tmp_path), "traj_kl": str(tmp_path / "tk"), "gpus": [""]},
        "env": {"module": "mock.env", "kwargs": {}},
        "serving": None,
        "decoding": None,
        "learn": {
            "mode": "generative",
            "method": "gates_seq",
            "model": str(model_p),
            "out_dir": str(tmp_path / "out"),
            "lessons": [{"pos_tokens": [1, 2, 3], "pos_targets": [1, 2]}],
            "controls": False,
            "heldout_ids": ["tr-0"],
        },
    }
    res = learn.run_generative({**base_cfg, "serving": {"slots_per_gpu": 2, "ctx_per_slot": 512, "special": True, "flash_attn": "off"}, "decoding": None})
    assert ev_inits[-1]["special"] is True
    assert ev_inits[-1]["flash_attn"] == "off"
    assert res["env_eval"]["learned"]["gate_pass"] is False
    assert res["accepted"] is False

    # None serving/decoding + unknown heldout_ids raises ValueError
    with pytest.raises(ValueError):
        learn.run_generative({**base_cfg, "learn": {**base_cfg["learn"], "heldout_ids": ["nonexistent-id"]}})


def test_learn_edge_cases_run_tails_and_task_kind(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    # 1) train_names with last_blocks=None
    art = SimpleNamespace(tensors={"blk.0.ffn_up.weight": {"type": "Q4_K"}, "blk.1.ffn_up.weight": {"type": "Q4_K"}})
    names = learn.train_names(art, {"last_blocks": None})
    assert len(names) == 2

    # 2) learn_gates with 0 switchable layers raises RuntimeError
    monkeypatch.setattr(learn.lattice, "Artifact", lambda p: SimpleNamespace(close=lambda: None))
    monkeypatch.setattr(learn, "build_student", lambda *a, **kw: (SimpleNamespace(model=SimpleNamespace(layers=[SimpleNamespace()])), None))
    monkeypatch.setattr(learn, "_switchable_layers", lambda *a, **kw: [])
    with pytest.raises(RuntimeError):
        learn.learn_gates(str(tmp_path / "m.gguf"), None, [], [], (1, 2), 0.0, str(tmp_path / "o.gguf"), cfg={"amp": False}, device="cpu")

    # 3) run() dispatches on task_kind == "generative"
    called = []
    monkeypatch.setattr(learn, "run_generative", lambda cfg, fold=None, log=print: (called.append(cfg) or {"ok": True}))
    out = learn.run({"learn": {"task_kind": "generative"}})
    assert out == {"ok": True} and len(called) == 1


# =============================================================================
# 17–21: mvp/tools/* & mvp/configs/* Tests
# =============================================================================


def test_m4_deepmath_select_dynamic_bands_select_split_and_answer_parsing(tmp_path: Path):
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
        "        for c in range(self.k):\n"
        "            m = self.labels_ == c\n"
        "            if m.any(): self.cluster_centers_[c] = X[m].mean(axis=0)\n"
        "        self.inertia_ = 1.0\n"
        "        return self\n"
    )

    # Mix 2 bands, +, commas, \boxed, \boxed{{...}}, $, guessable 0/1, bad answers, None difficulty/topic
    rows = []
    answers = ["+10", "-17", "1,234", r"\boxed{-42}", "$99$"]
    for i in range(30):
        area = "Algebra" if i < 15 else "Geometry"
        diff = 3.5 if i % 2 == 0 else 5.5
        ans = answers[i % len(answers)] if i < 5 else (r"\boxed{{42}}" if i == 5 else str(i + 20))
        topic = None if i == 6 else f"Mathematics -> {area} -> Sub"
        rows.append({
            "question": f"Q {i} {area}",
            "final_answer": ans,
            "difficulty": diff,
            "topic": topic,
            "shard": 0,
            "row": i,
        })
    # Rows that must be skipped cleanly (guessable, non-integer, None difficulty)
    rows.extend([
        {"question": "Skip 0", "final_answer": "0", "difficulty": 4.0, "topic": "Mathematics -> Algebra", "shard": 0, "row": 30},
        {"question": "Skip 1", "final_answer": "-0", "difficulty": 4.0, "topic": "Mathematics -> Algebra", "shard": 0, "row": 31},
        {"question": "Skip comma space", "final_answer": "1, 2", "difficulty": 4.0, "topic": "Mathematics -> Algebra", "shard": 0, "row": 32},
        {"question": "Skip float", "final_answer": "3.14", "difficulty": 4.0, "topic": "Mathematics -> Algebra", "shard": 0, "row": 33},
        {"question": "Skip none diff", "final_answer": "42", "difficulty": None, "topic": "Mathematics -> Algebra", "shard": 0, "row": 34},
    ])
    qa_p = tmp_path / "qa.parquet"
    pq.write_table(pa.Table.from_pylist(rows), qa_p)

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    rng = np.random.default_rng(0)
    emb = rng.standard_normal((32, 8)).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    np.save(out_dir / "pool-embeddings.npy", emb)

    env = {**os.environ, "PYTHONPATH": f"{stub_pkg.parent}:{ROOT / 'mvp'}"}
    # 3-way split with positional K N_EXAM N_SELECT on 2-band data
    p = subprocess.run(
        [sys.executable, str(ROOT / "mvp/tools/deepmath_select.py"), str(qa_p), str(tmp_path / "model"), str(out_dir), "30", "10", "10"],
        capture_output=True, text=True, env=env,
    )
    assert p.returncode == 0, p.stderr
    rep = json.loads((out_dir / "report.json").read_text())
    assert rep["split"]["n_exam"] == 10
    assert rep["split"]["n_select"] == 10
    assert rep["split"]["n_search"] == 10
    assert (out_dir / "deepmath-int-rep3k-select.jsonl").exists()
    all_picked = [
        json.loads(ln)
        for name in ("search", "select", "exam")
        for ln in (out_dir / f"deepmath-int-rep3k-{name}.jsonl").read_text().splitlines()
        if ln.strip()
    ]
    ans_set = {r["answer"] for r in all_picked}
    assert {"10", "-17", "1234", "-42", "99", "42"} <= ans_set
    assert "0" not in ans_set and "1" not in ans_set

    # Disjoint coarse groups across splits
    groups_by_split = {
        name: {json.loads(ln)["group"] for ln in (out_dir / f"deepmath-int-rep3k-{name}.jsonl").read_text().splitlines() if ln.strip()}
        for name in ("search", "select", "exam")
    }
    assert not (groups_by_split["exam"] & groups_by_split["select"])
    assert not (groups_by_split["exam"] & groups_by_split["search"])
    assert not (groups_by_split["select"] & groups_by_split["search"])


def test_m4_traj_kl_cpp_nan_rejection_fopen_null_and_flash_attn(tmp_path: Path):
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
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>
typedef int32_t llama_token; typedef int32_t llama_pos; typedef int32_t llama_seq_id;
enum llama_flash_attn_type { LLAMA_FLASH_ATTN_TYPE_AUTO = -1, LLAMA_FLASH_ATTN_TYPE_DISABLED = 0, LLAMA_FLASH_ATTN_TYPE_ENABLED = 1 };
struct llama_model_params { int32_t n_gpu_layers; };
struct llama_context_params {
    uint32_t n_ctx, n_batch, n_ubatch, n_seq_max;
    llama_flash_attn_type flash_attn_type;
    ggml_type type_k, type_v;
    int32_t n_threads, n_threads_batch;
};
struct llama_batch { int32_t n_tokens; llama_token * token; float * embd; llama_pos * pos; int32_t * n_seq_id; llama_seq_id ** seq_id; int8_t * logits; };
struct llama_vocab { int n; };
struct llama_model { llama_vocab v{16}; };
struct llama_memory_t { int d; };
typedef llama_memory_t * llama_memory_i;
struct llama_context { llama_context_params cp; llama_memory_t mem; std::vector<std::vector<float>> lb; bool inject_nan = false; bool inject_inf = false; };
inline void llama_backend_init() {}
inline void llama_backend_free() {}
inline llama_model_params llama_model_default_params() { return {0}; }
inline llama_context_params llama_context_default_params() { return {512, 512, 512, 1, LLAMA_FLASH_ATTN_TYPE_AUTO, GGML_TYPE_F16, GGML_TYPE_F16, 1, 1}; }
inline llama_model * llama_model_load_from_file(const char *, llama_model_params) { return new llama_model(); }
inline void llama_model_free(llama_model * m) { delete m; }
inline const llama_vocab * llama_model_get_vocab(const llama_model * m) { return &m->v; }
inline int llama_vocab_n_tokens(const llama_vocab * v) { return v->n; }
inline llama_context * llama_init_from_model(llama_model *, llama_context_params cp) {
    auto * c = new llama_context();
    c->cp = cp;
    c->inject_nan = getenv("STUB_INJECT_NAN") != nullptr;
    c->inject_inf = getenv("STUB_INJECT_INF") != nullptr;
    fprintf(stderr, "FA=%d\\n", (int)cp.flash_attn_type);
    return c;
}
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
inline int llama_decode(llama_context * c, llama_batch b) {
    c->lb.assign(b.n_tokens, {});
    for (int i = 0; i < b.n_tokens; ++i) if (b.logits[i]) {
        c->lb[i].assign(16, 0.1f);
        if (c->inject_nan) c->lb[i][2] = std::numeric_limits<float>::quiet_NaN();
        if (c->inject_inf) c->lb[i][2] = std::numeric_limits<float>::infinity();
    }
    return 0;
}
inline float * llama_get_logits_ith(llama_context * c, int32_t i) { return c->lb.at(i).data(); }
""")
    bin_p = tmp_path / "traj-kl"
    subprocess.run([cxx, "-O2", "-std=c++17", f"-I{inc}", str(ROOT / "mvp/tools/traj-kl.cpp"), "-o", str(bin_p)], check=True)

    trj_p = tmp_path / "t.bin"
    with open(trj_p, "wb") as f:
        f.write(b"TRJ1" + struct.pack("<I", 1) + struct.pack("<II", 3, 1) + struct.pack("<3i", 1, 2, 3) + struct.pack("<I", 2))
    store_p = tmp_path / "store"
    store_p.mkdir()
    m_p = tmp_path / "m.gguf"
    m_p.write_bytes(b"stub")

    # Finite base & kl succeed; flash-attn defaults to 1 (ENABLED), respects --flash-attn off & CHANTLESS_FLASH_ATTN
    p_base = subprocess.run([str(bin_p), "--mode", "base", "-m", str(m_p), "--traj", str(trj_p), "--store", str(store_p)], capture_output=True, text=True)
    assert p_base.returncode == 0, p_base.stderr
    assert "FA=1" in p_base.stderr

    p_off = subprocess.run([str(bin_p), "--mode", "kl", "-m", str(m_p), "--traj", str(trj_p), "--store", str(store_p), "--flash-attn", "off"], capture_output=True, text=True)
    assert p_off.returncode == 0 and "FA=0" in p_off.stderr

    # NaN / +Inf in kl mode MUST exit non-zero (never coerce NaN to 0.0)
    for var in ("STUB_INJECT_NAN", "STUB_INJECT_INF"):
        env = {**os.environ, var: "1"}
        p_kl_bad = subprocess.run([str(bin_p), "--mode", "kl", "-m", str(m_p), "--traj", str(trj_p), "--store", str(store_p)], capture_output=True, text=True, env=env)
        assert p_kl_bad.returncode != 0
        p_base_bad = subprocess.run([str(bin_p), "--mode", "base", "-m", str(m_p), "--traj", str(trj_p), "--store", str(store_p)], capture_output=True, text=True, env=env)
        assert p_base_bad.returncode != 0

    # Unwritable --out must exit non-zero cleanly (no segfault)
    p_out_bad = subprocess.run([str(bin_p), "--mode", "kl", "-m", str(m_p), "--traj", str(trj_p), "--store", str(store_p), "--out", str(tmp_path / "no_dir" / "out.json")], capture_output=True, text=True)
    assert p_out_bad.returncode != 0 and p_out_bad.returncode > 0

    # Out-of-vocab token ID rejected
    oob_trj = tmp_path / "oob.bin"
    with open(oob_trj, "wb") as f:
        f.write(b"TRJ1" + struct.pack("<I", 1) + struct.pack("<II", 3, 1) + struct.pack("<3i", 1, 999, 3) + struct.pack("<I", 2))
    p_oob = subprocess.run([str(bin_p), "--mode", "base", "-m", str(m_p), "--traj", str(oob_trj), "--store", str(store_p)], capture_output=True, text=True)
    assert p_oob.returncode != 0


def test_m4_logodds_cpp_vocab_bounds_nan_and_flash_attn(tmp_path: Path):
    cxx = shutil.which("clang++") or shutil.which("g++")
    if not cxx:
        pytest.skip("no C++ compiler available")
    inc = tmp_path / "inc"
    inc.mkdir()
    (inc / "llama.h").write_text("""#pragma once
#define LLAMA_H
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <vector>
typedef int32_t llama_token; typedef int32_t llama_pos; typedef int32_t llama_seq_id;
enum llama_flash_attn_type { LLAMA_FLASH_ATTN_TYPE_AUTO = -1, LLAMA_FLASH_ATTN_TYPE_DISABLED = 0, LLAMA_FLASH_ATTN_TYPE_ENABLED = 1 };
struct llama_model_params { int32_t n_gpu_layers; };
struct llama_context_params { uint32_t n_ctx, n_batch, n_ubatch, n_seq_max; llama_flash_attn_type flash_attn_type; bool kv_unified; };
struct llama_batch { int32_t n_tokens; llama_token * token; float * embd; llama_pos * pos; int32_t * n_seq_id; llama_seq_id ** seq_id; int8_t * logits; };
struct llama_vocab { int n; };
struct llama_model { llama_vocab v{32}; };
struct llama_memory_t { int d; };
typedef llama_memory_t * llama_memory_i;
struct llama_context { llama_context_params cp; llama_memory_t mem; std::vector<std::vector<float>> lb; bool inject_nan = false; };
inline void llama_backend_init() {}
inline void llama_backend_free() {}
inline llama_model_params llama_model_default_params() { return {0}; }
inline llama_context_params llama_context_default_params() { return {512, 512, 512, 1, LLAMA_FLASH_ATTN_TYPE_AUTO, false}; }
inline llama_model * llama_model_load_from_file(const char *, llama_model_params) { return new llama_model(); }
inline void llama_model_free(llama_model * m) { delete m; }
inline const llama_vocab * llama_model_get_vocab(const llama_model * m) { return &m->v; }
inline int llama_vocab_n_tokens(const llama_vocab * v) { return v->n; }
inline llama_context * llama_init_from_model(llama_model *, llama_context_params cp) {
    auto * c = new llama_context();
    c->cp = cp;
    c->inject_nan = getenv("STUB_INJECT_NAN") != nullptr;
    fprintf(stderr, "FA=%d\\n", (int)cp.flash_attn_type);
    return c;
}
inline void llama_free(llama_context * c) { delete c; }
inline llama_memory_i llama_get_memory(llama_context * c) { return &c->mem; }
inline void llama_memory_clear(llama_memory_i, bool) {}
inline llama_batch llama_batch_init(int32_t n, int32_t, int32_t s) {
    llama_batch b{}; b.token = new llama_token[n]; b.pos = new llama_pos[n]; b.n_seq_id = new int32_t[n];
    b.seq_id = new llama_seq_id*[n]; for (int i = 0; i < n; ++i) b.seq_id[i] = new llama_seq_id[s];
    b.logits = new int8_t[n]; return b;
}
inline void llama_batch_free(llama_batch b) { delete[] b.token; delete[] b.pos; delete[] b.n_seq_id; delete[] b.logits; delete[] b.seq_id; }
inline int llama_decode(llama_context * c, llama_batch b) {
    c->lb.assign(b.n_tokens, {});
    for (int i = 0; i < b.n_tokens; ++i) if (b.logits[i]) {
        c->lb[i].assign(32, 0.25f);
        if (c->inject_nan) c->lb[i][1] = std::numeric_limits<float>::quiet_NaN();
    }
    return 0;
}
inline float * llama_get_logits_ith(llama_context * c, int32_t i) { return c->lb.at(i).data(); }
""")
    bin_p = tmp_path / "logodds"
    subprocess.run([cxx, "-O2", "-std=c++17", f"-I{inc}", str(ROOT / "mvp/tools/logodds.cpp"), "-o", str(bin_p)], check=True)

    trj_p = tmp_path / "t.bin"
    with open(trj_p, "wb") as f:
        f.write(b"TRJ1" + struct.pack("<I", 1) + struct.pack("<II", 3, 1) + struct.pack("<3i", 1, 2, 3) + struct.pack("<I", 2))
    m_p = tmp_path / "m.gguf"
    m_p.write_bytes(b"stub")
    out_p = tmp_path / "out.tsv"

    for bad_pair in ("-1,2", "32,2", "2,999"):
        p_bad = subprocess.run([str(bin_p), "-m", str(m_p), "--traj", str(trj_p), "--pair", bad_pair, "--out", str(out_p)], capture_output=True, text=True)
        assert p_bad.returncode != 0, bad_pair

    p_nseq = subprocess.run([str(bin_p), "-m", str(m_p), "--traj", str(trj_p), "--pair", "1,2", "--n-seq", "0", "--out", str(out_p)], capture_output=True, text=True)
    assert p_nseq.returncode != 0

    p_nan = subprocess.run([str(bin_p), "-m", str(m_p), "--traj", str(trj_p), "--pair", "1,2", "--out", str(out_p)], capture_output=True, text=True, env={**os.environ, "STUB_INJECT_NAN": "1"})
    assert p_nan.returncode != 0

    p_fa = subprocess.run([str(bin_p), "-m", str(m_p), "--traj", str(trj_p), "--pair", "1,2", "--out", str(out_p), "--flash-attn", "off"], capture_output=True, text=True)
    assert p_fa.returncode == 0 and "FA=0" in p_fa.stderr


def test_m4_tokdocs_cpp_nth_validation_fopen_checks_and_negative_tokenize(tmp_path: Path):
    cxx = shutil.which("clang++") or shutil.which("g++")
    if not cxx:
        pytest.skip("no C++ compiler available")
    inc = tmp_path / "inc"
    inc.mkdir()
    (inc / "llama.h").write_text("""#pragma once
#include <cstdint>
#include <cstdlib>
typedef int32_t llama_token;
struct llama_model_params { bool vocab_only; int32_t n_gpu_layers; };
struct llama_vocab { int d; };
struct llama_model { llama_vocab v{1}; };
inline void llama_backend_init() {}
inline void llama_backend_free() {}
inline llama_model_params llama_model_default_params() { return {false, 0}; }
inline llama_model * llama_model_load_from_file(const char *, llama_model_params) { return new llama_model(); }
inline void llama_model_free(llama_model * m) { delete m; }
inline const llama_vocab * llama_model_get_vocab(const llama_model * m) { return &m->v; }
inline int32_t llama_tokenize(const llama_vocab *, const char * text, int32_t len, llama_token * out, int32_t n_max, bool, bool) {
    if (getenv("STUB_FAIL_TOK")) return -100;
    if (n_max < len + 1) return -(len + 1);
    for (int32_t i = 0; i < len; ++i) out[i] = (llama_token)(unsigned char)text[i];
    return len;
}
""")
    bin_p = tmp_path / "tokdocs"
    subprocess.run([cxx, "-O2", "-std=c++17", f"-I{inc}", str(ROOT / "mvp/tools/tokdocs.cpp"), "-o", str(bin_p)], check=True)

    in_p = tmp_path / "in.bin"
    docs = [b"hello", b"world!"]
    with open(in_p, "wb") as f:
        f.write(struct.pack("<I", len(docs)))
        for d in docs:
            f.write(struct.pack("<I", len(d)) + d)
    m_p = tmp_path / "m.gguf"
    m_p.write_bytes(b"stub")
    out_p = tmp_path / "out.bin"

    p_ok = subprocess.run([str(bin_p), str(m_p), str(in_p), str(out_p), "2"], capture_output=True, text=True)
    assert p_ok.returncode == 0, p_ok.stderr
    assert out_p.exists()

    for bad_nth in ("0", "-1", "abc"):
        p_nth = subprocess.run([str(bin_p), str(m_p), str(in_p), str(out_p), bad_nth], capture_output=True, text=True)
        assert p_nth.returncode != 0, bad_nth

    p_no_in = subprocess.run([str(bin_p), str(m_p), str(tmp_path / "missing.bin"), str(out_p), "2"], capture_output=True, text=True)
    assert p_no_in.returncode != 0 and p_no_in.returncode > 0

    p_no_out = subprocess.run([str(bin_p), str(m_p), str(in_p), str(tmp_path / "no_dir" / "out.bin"), "2"], capture_output=True, text=True)
    assert p_no_out.returncode != 0 and p_no_out.returncode > 0

    p_fail_tok = subprocess.run([str(bin_p), str(m_p), str(in_p), str(out_p), "2"], capture_output=True, text=True, env={**os.environ, "STUB_FAIL_TOK": "1"})
    assert p_fail_tok.returncode != 0 and p_fail_tok.returncode > 0


def test_m4_configs_zero_trust_alignment_and_env_instantiation(tmp_path: Path):
    cfg_dir = ROOT / "mvp" / "configs"
    extract_names = (
        "extract-example.yaml",
        "inkling2-mini-bfcl.yaml",
        "inkling2-mini-clsbench.yaml",
        "inkling2-mini-deepmath.yaml",
        "local-smoke.json",
    )
    loaded = {name: cfgmod.load_config(cfg_dir / name, "extract") for name in extract_names}
    for name, c in loaded.items():
        assert c.serving.cache_prompt is False, name
        assert c.serving.flash_attn == "on", name

    # extract-example.yaml
    ex = loaded["extract-example.yaml"]
    assert ex.pool.tasks_per_split == {"search": 80, "select": 80, "exam": 160}
    assert ex.env.kwargs.get("hf_dir") == ".local/models/qwen3-4b"
    assert ex.decoding.max_tokens == 1024
    assert ex.decoding.enable_thinking is False

    # inkling2-mini-clsbench.yaml
    cls = loaded["inkling2-mini-clsbench.yaml"]
    assert cls.env.kwargs.get("context") == 2048
    assert cls.serving.ctx_per_slot == 2048
    assert cls.env.kwargs.get("fpr_target") == 0.01

    # local-smoke.json: deployment concurrency=1 decoupled from eval slots_per_gpu=4, fits 4GB budget
    smk = loaded["local-smoke.json"]
    assert smk.env.kwargs.get("hf_dir") == ".local/models/qwen3-0.6b"
    assert smk.env.kwargs.get("concurrency") == 1
    assert smk.serving.slots_per_gpu == 4
    from chantless import builder as bmod
    smk_kv = bmod.kv_bytes(
        {"n_layer": 28, "n_head_kv": 8, "head_dim": 128},
        28,
        smk.env.kwargs["context"],
        smk.env.kwargs["concurrency"],
        smk.env.kwargs["kv_type"],
    )
    assert smk_kv + int(595_984_384 * 34 / 32) < smk.env.kwargs["max_total_memory_bytes"]
    assert smk.decoding.max_tokens == 768
    assert smk.decoding.enable_thinking is False

    # inkling2-mini-bfcl.yaml & inkling2-mini-deepmath.yaml: deployment concurrency default 1 distinct from slots_per_gpu=8
    assert loaded["inkling2-mini-bfcl.yaml"].env.kwargs.get("concurrency", 1) == 1
    assert loaded["inkling2-mini-bfcl.yaml"].serving.slots_per_gpu == 8
    assert loaded["inkling2-mini-deepmath.yaml"].env.kwargs.get("concurrency", 1) == 1
    assert loaded["inkling2-mini-deepmath.yaml"].serving.slots_per_gpu == 8

    # learn-example.yaml
    lrn = cfgmod.load_config(cfg_dir / "learn-example.yaml", "learn")
    assert lrn.learn.judge_cache_prompt is False
    assert lrn.serving is not None and lrn.serving.cache_prompt is False and lrn.serving.flash_attn == "on"
    assert lrn.decoding is not None and lrn.decoding.max_tokens == 1024 and lrn.decoding.enable_thinking is False

    # Instantiate codetasks from local-smoke.json kwargs (offline) and validate protocol
    hf_p = _make_hf_dir(tmp_path)
    ct_kw = {**smk.env.kwargs, "hf_dir": str(hf_p), "variants_per_family": 2}
    ct_env = codetasks.CodeTasksEnv(**ct_kw)
    protocol.validate_environment(ct_env)



# =============================================================================
# Iteration 2 Remediation Regression Tests (Strict Test-First)
# =============================================================================


def test_pipeline_multi_element_nan_masking_prepass_and_seed_scores(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    hf_p = _make_hf_dir(tmp_path)
    gguf_p = _write_gguf(tmp_path / "m.gguf")
    env_obj = _make_stub_env(classifier=True)
    monkeypatch.setattr(pipeline, "load_environment", lambda mod, **kw: env_obj)

    cfg = {
        "run_id": "r-nan-multi",
        "run_root": str(tmp_path / "runs"),
        "source": {"gguf": str(gguf_p), "repo": "test/r", "revision": "abc", "hf_dir": str(hf_p)},
        "toolchain": {"llama_bin": str(tmp_path), "traj_kl": str(tmp_path / "tk"), "logodds": str(tmp_path / "lo"), "gpus": [""]},
        "env": {"module": "stub.env", "kwargs": {}},
        "ladder": ["BF16", "Q8_0", "Q4_K"],
        "start_tier": "Q4_K",
        "sanity": {"behavioral_items": 0},
    }
    r = pipeline.Run(cfg)

    # 1) _prepass_dead_tensors: task 0 has diff=0.0, task 1+ has NaN -> must raise RuntimeError (never mask NaN in max)
    r.dead_tensors = ["dummy"]
    monkeypatch.setattr(gguf_ops, "copy_without", lambda src, dst, drop: Path(dst).write_bytes(b"stub"))
    r.env.score = lambda sample, m, sc: {
        sample[0].id: 1.0,
        **{t.id: (1.0 if m == r.src else float("nan")) for t in sample[1:]},
    }
    with pytest.raises(RuntimeError):
        r._prepass_dead_tensors()

    # 2) _seed_source_scores: task 0 matches (1.0 vs 1.0), task 1+ has fresh=NaN -> must raise/abort seeding
    ids_p = tmp_path / "seed_ids.json"
    npy_p = tmp_path / "seed_vals.npy"
    ids_p.write_text(json.dumps([{"id": "famA-0"}, {"id": "famA-1"}]))
    np.save(npy_p, np.array([1.0, 1.0], dtype=np.float64))
    r.by_id["s:famA-0"] = r.by_id["search-famA-0"]
    r.by_id["s:famA-1"] = r.by_id["search-famA-1"]
    r.env.score = lambda tasks, m, sc: {
        tasks[0].id: 1.0,
        **{t.id: float("nan") for t in tasks[1:]},
    }
    r.ev._scores.clear()
    try:
        r._seed_source_scores({"files": [{"prefix": "s", "ids": str(ids_p), "npy": str(npy_p)}], "check": 2})
    except RuntimeError:
        pass
    assert not r.ev._scores, "seeding must be rejected when fresh verification contains NaN after index 0"

    # Also when recorded .npy itself has NaN at index 1+ while fresh is finite -> must abort seeding
    np.save(npy_p, np.array([1.0, float("nan")], dtype=np.float64))
    r.env.score = lambda tasks, m, sc: {t.id: 1.0 for t in tasks}
    r.ev._scores.clear()
    try:
        r._seed_source_scores({"files": [{"prefix": "s", "ids": str(ids_p), "npy": str(npy_p)}], "check": 2})
    except RuntimeError:
        pass
    assert not r.ev._scores, "seeding must be refused when recorded scores contain NaN"


def test_learn_run_generative_heldout_100pct_infra_error_on_base_and_controls_rejects(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    class MockEnv:
        deployment = {"context": 512, "concurrency": 1, "kv_type": "f16"}
        def tasks(self):
            return [
                Task(id="tr-0", family="famA", split="train", prompt="p", meta={}),
                Task(id="tr-1", family="famB", split="train", prompt="p", meta={}),
            ] + [Task(id=f"sel-{i}", family="famA" if i % 2 == 0 else "famB", split="select", prompt="p", meta={}) for i in range(40)]

    import chantless.evaluator as evmod
    monkeypatch.setattr("chantless.env.protocol.load_environment", lambda mod, **kw: MockEnv())
    monkeypatch.setattr(learn, "_tokenize_seq_lessons", lambda tc, m, rows, **kw: rows)
    monkeypatch.setattr(learn, "_load_replay_trajs", lambda tc, m, sp, **kw: [])
    monkeypatch.setattr(learn, "learn_gates_seq", lambda *a, **kw: (Path(a[4]).write_bytes(b"G"), {"off": 1})[1])
    monkeypatch.setattr(learn, "sha256", lambda p: "0")
    monkeypatch.setattr(kl, "write_trj", lambda p, recs: {"n_seq": len(recs), "n_targets": len(recs)})
    monkeypatch.setattr(kl, "traj_kl", lambda *a, **kw: {"trajs": [{"nll_cand": 1.0 if "base" in a[6] else 0.5}] * 30})

    # Case A: controls=False, base 100% infra error on heldout ([]), learned heldout=0.0, select passes -> accepted MUST be False
    class FakeEvBaseInfra:
        def __init__(self, *a, **kw):
            pass
        def run(self, model, jobs, tag, side):
            if tag == "learn-base-held":
                return []
            if tag == "learn-learned-held":
                return [{"task": t.id, "family": t.family, "split": t.split, "seed": s, "side": side, "success": False} for t, s in jobs]
            return [{"task": t.id, "family": t.family, "split": t.split, "seed": s, "side": side, "success": True} for t, s in jobs]

    monkeypatch.setattr(evmod, "Evaluator", FakeEvBaseInfra)
    model_p = tmp_path / "m.gguf"
    model_p.write_bytes(b"stub")
    cfg = {
        "toolchain": {"llama_bin": str(tmp_path), "traj_kl": str(tmp_path / "tk"), "gpus": [""]},
        "env": {"module": "mock.env", "kwargs": {}},
        "learn": {
            "mode": "generative",
            "method": "gates_seq",
            "model": str(model_p),
            "out_dir": str(tmp_path / "out_a"),
            "lessons": [{"pos_tokens": [1, 2, 3], "pos_targets": [1, 2]}],
            "controls": False,
            "heldout_ids": ["tr-0"],
        },
    }
    res_a = learn.run_generative(cfg)
    assert res_a["env_eval"]["base"]["heldout_rate"] is None
    assert res_a["env_eval"]["learned"]["heldout_rate"] == 0.0
    assert res_a["env_eval"]["learned"]["gate_pass"] is True
    assert res_a["accepted"] is False

    # Case B: controls=True, base=0.5, learned=1.0, but empty control has 100% infra error (heldout_rate is None) -> accepted MUST be False
    class FakeEvCtrlInfra:
        def __init__(self, *a, **kw):
            pass
        def run(self, model, jobs, tag, side):
            if tag == "learn-empty-held":
                return []
            if tag == "learn-base-held":
                return [
                    {"task": t.id, "family": t.family, "split": t.split, "seed": s, "side": side, "success": (i == 0)}
                    for i, (t, s) in enumerate(jobs)
                ]
            return [{"task": t.id, "family": t.family, "split": t.split, "seed": s, "side": side, "success": True} for t, s in jobs]

    monkeypatch.setattr(evmod, "Evaluator", FakeEvCtrlInfra)
    res_b = learn.run_generative({
        **cfg,
        "learn": {
            **cfg["learn"],
            "out_dir": str(tmp_path / "out_b"),
            "controls": True,
            "heldout_ids": ["tr-0", "tr-1"],
        },
    })
    assert res_b["env_eval"]["empty"]["heldout_rate"] is None
    assert res_b["accepted"] is False


def test_pipeline_phase_sanity_partial_solo_task_tiers_with_floor_key_task_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    gguf_p = _write_gguf(tmp_path / "m.gguf")
    hf_p = _make_hf_dir(tmp_path)
    monkeypatch.setattr(pipeline, "load_environment", lambda mod, **kw: _make_stub_env(classifier=False))

    cfg = {
        "run_id": "r-partial-stt",
        "run_root": str(tmp_path / "runs"),
        "source": {"gguf": str(gguf_p), "repo": "test/r", "revision": "abc", "hf_dir": str(hf_p)},
        "toolchain": {"llama_bin": str(tmp_path), "traj_kl": str(tmp_path / "tk"), "logodds": str(tmp_path / "lo"), "gpus": [""]},
        "env": {"module": "stub.env", "kwargs": {}},
        "ladder": ["BF16", "Q8_0", "Q4_K"],
        "start_tier": "Q4_K",
        "proxy": {"floor_tier": "Q4_K"},
        "sanity": {"solo_tiers": ["Q8_0", "Q4_K"], "floor_key": "task", "cap": "general_kl", "behavioral_items": 0, "allow_drop": False},
        "probe": {"want_solo": ["task", "general"], "solo_task_tiers": ["Q4_K"], "splice_cache": False},
    }
    r = pipeline.Run(cfg)
    (r.dir / "proxy").mkdir(parents=True, exist_ok=True)
    (r.dir / "proxy" / "proxy.json").write_text(json.dumps({
        "task": {"n_targets": 1}, "general": {"n_targets": 1},
        "self_kl": {"task": 0.0, "general": 0.0}, "imatrix_sha256": "a",
    }))
    (r.dir / "feedback_stores.jsonl").write_text("")
    (r.dir / "run.json").write_text(json.dumps({"pins": {"source_gguf_sha256": "a"}}))

    probed_batches: list[int] = []

    class FakeProber:
        def __init__(self):
            self.workers = 1
            self.builder = SimpleNamespace(source_params=1000, cleanup=lambda: None)
        def probe_many(self, masks, stores):
            probed_batches.append(len(masks))
            return [SimpleNamespace(kl={s: 0.01 for s in stores}, bytes=500, bpw=4.0) for _ in masks]

    monkeypatch.setattr(r, "make_prober", lambda w, tag: FakeProber())
    with pytest.raises(ValueError, match="solo_task_tiers"):
        r.phase_sanity()
    assert probed_batches == [], "validation must fail upfront before running probes"


def test_pipeline_explicit_none_fields_sanity_search_greedy_and_exam(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    gguf_p = _write_gguf(tmp_path / "m.gguf")
    hf_p = _make_hf_dir(tmp_path)
    monkeypatch.setattr(pipeline, "load_environment", lambda mod, **kw: _make_stub_env(classifier=False))

    # 1) Run with sanity: None and omitted sanity (including classifier=True)
    cfg_none = {
        "run_id": "r-none-fields",
        "run_root": str(tmp_path / "runs"),
        "source": {"gguf": str(gguf_p), "repo": "test/r", "revision": "abc", "hf_dir": str(hf_p)},
        "toolchain": {"llama_bin": str(tmp_path), "traj_kl": str(tmp_path / "tk"), "logodds": str(tmp_path / "lo"), "gpus": [""]},
        "env": {"module": "stub.env", "kwargs": {}},
        "ladder": ["BF16", "Q8_0", "Q4_K"],
        "start_tier": "Q4_K",
        "probe": None,
        "sanity": None,
        "repair": None,
        "exam": None,
        "serving": None,
        "decoding": None,
        "scoring": None,
        "stats": {},
        "budget": {"max_hours": None},
        "learner": {
            "algo": "greedy",
            "max_moves": None,
            "target_bytes": None,
            "screen_every": None,
            "gate_every": None,
            "exchange_rounds": None,
            "feedback_tasks": None,
            "top_k": None,
            "reprobe_every": None,
            "max_revert": None,
            "implicate_share": None,
            "penalty_step": None,
            "feedback": None,
            "reprobe_after_screen": None,
        },
        "pool": {"tasks_per_split": {"exam": None, "select": 4}, "runs": 1, "min_pass": 1},
    }
    r = pipeline.Run(cfg_none)
    r_omitted = pipeline.Run({k: v for k, v in cfg_none.items() if k != "sanity"} | {"run_id": "r-omitted-sanity"})
    assert r_omitted.ev.cache_prompt is False

    # Also classifier=True with sanity: None and omitted sanity (hits line 302)
    monkeypatch.setattr(pipeline, "load_environment", lambda mod, **kw: _make_stub_env(classifier=True))
    r_cls_none = pipeline.Run({**cfg_none, "run_id": "r-cls-none-sanity"})
    r_cls_omit = pipeline.Run({k: v for k, v in cfg_none.items() if k != "sanity"} | {"run_id": "r-cls-omit-sanity"})
    assert r_cls_none.classifier is True and r_cls_omit.classifier is True
    monkeypatch.setattr(pipeline, "load_environment", lambda mod, **kw: _make_stub_env(classifier=False))

    # 1b) Unmocked make_prober with probe: None
    (r.dir / "proxy" / "store_task").mkdir(parents=True, exist_ok=True)
    (r.dir / "proxy" / "store_general").mkdir(parents=True, exist_ok=True)
    (r.dir / "proxy" / "proxy.json").write_text(json.dumps({
        "task": {"n_targets": 1, "kind": "kl", "meta": {}},
        "general": {"n_targets": 1},
    }))
    unmocked_prober = r.make_prober(str(gguf_p), "test")
    assert unmocked_prober.builder.threads == 8
    assert [s.name for s in unmocked_prober.stores] == ["task", "general"]

    # 2) phase_search greedy with explicit None learner/budget fields
    (r.dir / "sanity").mkdir(parents=True, exist_ok=True)
    init_mask = mk.new_mask(r.groups, "Q4_K", {"repo": "test/r", "revision": "abc", "gguf_sha256": "a"}, "preserve")
    (r.dir / "sanity" / "sanity_mask.json").write_text(json.dumps(init_mask))
    (r.dir / "sanity" / "cap.json").write_text(json.dumps({"kind": "off"}))
    (r.dir / "pool.json").write_text(json.dumps({
        "sizes": {"search": {"screened": 1, "reliable": 1}, "select": {"screened": 1, "reliable": 1}},
        "reliable": {"search": ["search-famA-0"], "select": ["select-famA-0"]},
    }))

    fake_res = SimpleNamespace(key="k0", bytes=1000, bpw=4.0, kl={"task": 0.0, "general": 0.0})
    fake_prober = SimpleNamespace(
        stores=[],
        probe=lambda m: fake_res,
        builder=SimpleNamespace(build=lambda m, out: SimpleNamespace(path=str(gguf_p)), cleanup=lambda: None),
    )

    step_calls = {"n": 0}
    class StubLearner:
        def __init__(self, *a, **kw):
            pass
        def full_reprobe(self, S, base):
            pass
        def step(self, S, base):
            step_calls["n"] += 1
            if step_calls["n"] == 1:
                S.accepted += 1
                return {"move": SimpleNamespace(key="g->Q4_K"), "res": fake_res, "score": 0.0}
            return None

    def fake_gate(cand, tag, split="select"):
        if split == "exam":
            src_ep = r.dir / "episodes" / "gate-exam-src.jsonl"
            src_ep.write_text("\n".join(
                json.dumps({"task": t.id, "family": t.family, "split": t.split, "seed": 0, "side": "src", "success": True})
                for t in r.exam_pool
            ) + "\n")
        return {
            "pass": True, "L": 0.0, "U": 0.0, "delta": 0.0,
            "n_cand": 4, "n_src": 4, "n_tasks": 4,
            "detectors": {}, "behavioral": {"delta": 0.0, "L": 0.0}, "behavioral_ok": True,
        }

    monkeypatch.setattr(pipeline, "Learner", StubLearner)
    monkeypatch.setattr(r, "screen", lambda cand, tag: {"delta": 0.0, "behavioral_rate": 1.0, "behavioral_source": 1.0, "pass": True, "_beh_fail": []})
    monkeypatch.setattr(r, "gate", fake_gate)
    monkeypatch.setattr(r.ev, "run", lambda m, jobs, tag, side: [])
    (r.dir / "episodes").mkdir(parents=True, exist_ok=True)
    (r.dir / "episodes" / "fb-0001.jsonl").write_text("")
    with monkeypatch.context() as mp:
        mp.setattr(r, "make_prober", lambda w, tag: fake_prober)
        mp.setattr(pipeline.bmod, "Builder", lambda *a, **kw: fake_prober.builder)
        r.guard.phase = "search"
        r.phase_search()
    sr = json.loads((r.dir / "search_result.json").read_text())
    assert sr["stop_reason"] == "exhausted" and sr["accepted"] == 1

    # 3) phase_exam with tasks_per_split["exam"] = None falling back to select=4 AND unmocked exam_baselines
    (r.dir / "run.json").write_text(json.dumps({"pins": {"source_gguf_sha256": "a"}}))
    (r.dir / "final").mkdir(parents=True, exist_ok=True)
    spec_p = r.dir / "final" / "specialist.gguf"
    shutil.copyfile(gguf_p, spec_p)
    spec_sz = spec_p.stat().st_size
    (r.dir / "sanity" / "uniform.json").write_text(json.dumps({
        "Q8_0": {"bytes": spec_sz * 2},
        "Q4_K": {"bytes": spec_sz},
    }))

    def _stub_build(self_b, m, out):
        out_p = Path(out)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(gguf_p, out_p)
        return SimpleNamespace(path=str(out_p), file_bytes=spec_sz, bpw=4.0)

    monkeypatch.setattr(pipeline.bmod.Builder, "build", _stub_build)
    monkeypatch.setattr(pipeline.Prober, "probe", lambda self_p, m, stores=None: SimpleNamespace(key="b0", bytes=spec_sz, bpw=4.0, kl={"task": 0.1, "general": 0.0}))
    monkeypatch.setattr(
        pipeline.Prober,
        "probe_many",
        lambda self_p, ms, stores=None: [SimpleNamespace(key=f"b{i}", bytes=spec_sz, bpw=4.0, kl={"task": 0.05, "general": 0.0}) for i, _ in enumerate(ms)],
    )

    seen_exam_tasks: list[Task] = []
    baseline_tags: list[str] = []
    def fake_exam_run(m, jobs, tag, side):
        if tag == "exam-pool":
            seen_exam_tasks.extend(t for t, _ in jobs)
        elif tag.startswith("exam-"):
            baseline_tags.append(tag)
        return [{"task": t.id, "family": t.family, "split": t.split, "seed": s, "side": side, "success": True} for t, s in jobs]
    monkeypatch.setattr(r.ev, "run", fake_exam_run)
    r.guard.phase = "exam"
    r.phase_exam()
    assert len(seen_exam_tasks) == 4
    assert baseline_tags and (r.dir / "gates" / "exam_baselines.json").exists()

    # 4) Unmocked _weights_outcome() and phase_report() when repair=None and stats={}
    assert r._weights_outcome() == "mask-only"
    (r.dir / "aa.json").write_text(json.dumps({"aa_delta": 0.0, "v": 0.1, "deff_used": 1.0, "n_gate": 4}))
    (r.dir / "final" / "freeze.json").write_text(json.dumps({
        "build": {"file_bytes": spec_sz, "bpw": 4.0, "dropped": []},
        "audit": {"rebuild_sha256_equal": True, "types_match": True, "dropped_absent": True},
        "deployment_memory": {"measured_total": 1000, "kv_measured_bytes": 100, "kv_formula_bytes": 100, "ok": True},
    }))
    r.phase_report()
    report_txt = (r.dir / "report.md").read_text()
    assert "weights: mask-only" in report_txt


def test_plan_estimate_explicit_none_tasks_per_split(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    import chantless.env.protocol as proto_mod

    gguf_p = _write_gguf(tmp_path / "m.gguf")
    hf_p = _make_hf_dir(tmp_path)
    monkeypatch.setattr(proto_mod, "load_environment", lambda mod, **kw: _make_stub_env(classifier=False))

    base_cfg = {
        "run_id": "plan-none-tps",
        "run_root": str(tmp_path / "runs"),
        "source": {"gguf": str(gguf_p), "repo": "test/r", "revision": "abc", "hf_dir": str(hf_p)},
        "toolchain": {"llama_bin": str(tmp_path), "traj_kl": str(tmp_path / "tk"), "gpus": [""]},
        "env": {"module": "stub.env", "kwargs": {}},
        "ladder": ["BF16", "Q8_0", "Q4_K"],
        "start_tier": "Q4_K",
        "probe": None,
        "sanity": None,
        "repair": None,
        "exam": None,
        "serving": None,
        "decoding": None,
        "scoring": None,
        "budget": None,
        "stats": None,
        "learner": None,
        "proxy": None,
        "screen": None,
        "pool": {"tasks_per_split": {"exam": None, "select": 4}, "runs": 1, "min_pass": 1},
    }
    est1 = plan.estimate(base_cfg)
    exam_work1 = next(w for w in est1["work"] if w["stage"] == "exam")
    # select=4 -> exam falls back to 4; ex_pool (4) + cand (4) + src (4) + baselines (2*4) = 20 items
    assert exam_work1["items"] == 20

    cfg_all_none = {
        **base_cfg,
        "pool": {"tasks_per_split": {"exam": None, "select": None, "search": None}, "runs": 1, "min_pass": 1},
    }
    est2 = plan.estimate(cfg_all_none)
    assert all(w["stage"] != "exam" for w in est2["work"])

    cfg_zero_exam = {
        **base_cfg,
        "pool": {"tasks_per_split": {"exam": 0, "select": 4}, "runs": 1, "min_pass": 1},
    }
    est3 = plan.estimate(cfg_zero_exam)
    assert all(w["stage"] != "exam" for w in est3["work"])


# =============================================================================
# Iteration 4 Remediation Regression Tests (Strict Test-First)
# =============================================================================


def test_pipeline_phase_aa_explicit_none_and_aa_enforce_security(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from contextlib import nullcontext

    gguf_p = _write_gguf(tmp_path / "m.gguf")
    hf_p = _make_hf_dir(tmp_path)
    monkeypatch.setattr(pipeline, "load_environment", lambda mod, **kw: _make_stub_env(classifier=False))
    monkeypatch.setattr(sanity, "run_behavioral", lambda *a, **kw: [])

    def _make_run(run_id: str, stats_cfg: Any):
        cfg = {
            "run_id": run_id,
            "run_root": str(tmp_path / "runs"),
            "source": {"gguf": str(gguf_p), "repo": "test/r", "revision": "abc", "hf_dir": str(hf_p)},
            "toolchain": {"llama_bin": str(tmp_path), "traj_kl": str(tmp_path / "tk"), "logodds": str(tmp_path / "lo"), "gpus": [""]},
            "env": {"module": "stub.env", "kwargs": {}},
            "ladder": ["BF16", "Q8_0", "Q4_K"],
            "start_tier": "Q4_K",
            "stats": stats_cfg,
            "sanity": {"behavioral_items": 0},
        }
        r = pipeline.Run(cfg)
        rel_ids = [f"select-{fam}-{i}" for fam in ("famA", "famB", "famC") for i in range(7)]
        (r.dir / "pool.json").write_text(json.dumps({"reliable": {"select": rel_ids}}))
        monkeypatch.setattr(r.ev, "servers", lambda m: nullcontext([]))
        return r

    # 1) stats["aa_enforce"] = None with biased A/A rows (aa_ok=False) MUST raise RuntimeError
    r_enf_none = _make_run(
        "r-aa-enf-none",
        {
            "margin": 0.02, "alpha": 0.05, "power": 0.9, "n_floor": 2, "looks": 1,
            "max_eps_per_task": 1, "aa_tol": 0.01, "aa_enforce": None, "aa_replication": False,
            "detector_bound": 0.03, "n_aa": 2, "v_min": 0.05, "allow_small_pool": True,
        },
    )
    monkeypatch.setattr(
        r_enf_none.ev,
        "run",
        lambda m, jobs, tag, side, srvs=None: [
            {"task": t.id, "family": t.family, "split": t.split, "seed": s, "side": side, "success": (tag == "aa-A")}
            for t, s in jobs
        ],
    )
    with pytest.raises(RuntimeError, match="A/A check failed"):
        r_enf_none.phase_aa()

    # 2) Explicit None fields (v_min=None, aa_replication=None, etc.) with explicit aa_enforce=False preserved
    r_vmin_none = _make_run(
        "r-aa-vmin-none",
        {
            "margin": None, "alpha": None, "power": None, "n_floor": 2, "looks": None,
            "max_eps_per_task": None, "aa_tol": 0.01, "aa_enforce": False, "aa_replication": None,
            "detector_bound": None, "n_aa": None, "v_min": None, "allow_small_pool": True,
        },
    )
    seen_tags: list[str] = []
    def _biased_with_rep(m, jobs, tag, side, srvs=None):
        seen_tags.append(tag)
        ok_val = tag in ("aa-A", "aa-C")
        return [{"task": t.id, "family": t.family, "split": t.split, "seed": s, "side": side, "success": ok_val} for t, s in jobs]

    monkeypatch.setattr(r_vmin_none.ev, "run", _biased_with_rep)
    r_vmin_none.phase_aa()
    aa_rep_data = json.loads((r_vmin_none.dir / "aa.json").read_text())
    assert "replication" in aa_rep_data and aa_rep_data["aa_ok"] is False
    assert seen_tags == ["aa-A", "aa-B", "aa-D", "aa-C"]

    # 3) stats=None with unbiased A/A rows succeeds cleanly
    r_stats_none = _make_run("r-aa-stats-none", None)
    monkeypatch.setattr(
        r_stats_none.ev,
        "run",
        lambda m, jobs, tag, side, srvs=None: [
            {"task": t.id, "family": t.family, "split": t.split, "seed": s, "side": side, "success": True}
            for t, s in jobs
        ],
    )
    r_stats_none.phase_aa()
    aa_none_data = json.loads((r_stats_none.dir / "aa.json").read_text())
    assert aa_none_data["aa_ok"] is True

    # 4) Explicit v_min=0.0 is preserved (not clobbered by default 0.05/0.18)
    r_vmin_zero = _make_run(
        "r-aa-vmin-zero",
        {
            "margin": 0.02, "alpha": 0.05, "power": 0.9, "n_floor": 2, "looks": 1,
            "max_eps_per_task": 1, "aa_tol": 0.01, "aa_enforce": True, "aa_replication": False,
            "n_aa": 2, "v_min": 0.0, "allow_small_pool": True,
        },
    )
    monkeypatch.setattr(
        r_vmin_zero.ev,
        "run",
        lambda m, jobs, tag, side, srvs=None: [
            {"task": t.id, "family": t.family, "split": t.split, "seed": s, "side": side, "success": True}
            for t, s in jobs
        ],
    )
    r_vmin_zero.phase_aa()
    aa_zero_data = json.loads((r_vmin_zero.dir / "aa.json").read_text())
    assert aa_zero_data["v_used"] == 0.0


def test_pipeline_gate_explicit_none_stats_and_detector_bound(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from contextlib import nullcontext

    gguf_p = _write_gguf(tmp_path / "m.gguf")
    hf_p = _make_hf_dir(tmp_path)
    monkeypatch.setattr(pipeline, "load_environment", lambda mod, **kw: _make_stub_env(classifier=False))
    beh_rows = [
        {"task": "b0", "family": "famA", "seed": 0, "success": True},
        {"task": "b1", "family": "famB", "seed": 0, "success": True},
    ]
    monkeypatch.setattr(sanity, "run_behavioral", lambda srvs, dec, items, seeds, side, **kw: [dict(r, side=side) for r in beh_rows])

    def _run_gate(run_id: str, stats_cfg: Any, cand_fmt_trigger: bool):
        cfg = {
            "run_id": run_id,
            "run_root": str(tmp_path / "runs"),
            "source": {"gguf": str(gguf_p), "repo": "test/r", "revision": "abc", "hf_dir": str(hf_p)},
            "toolchain": {"llama_bin": str(tmp_path), "traj_kl": str(tmp_path / "tk"), "logodds": str(tmp_path / "lo"), "gpus": [""]},
            "env": {"module": "stub.env", "kwargs": {}},
            "ladder": ["BF16", "Q8_0", "Q4_K"],
            "start_tier": "Q4_K",
            "stats": stats_cfg,
            "sanity": {"behavioral_items": 0},
        }
        r = pipeline.Run(cfg)
        rel_ids = [f"select-{fam}-{i}" for fam in ("famA", "famB", "famC") for i in range(2)]
        (r.dir / "pool.json").write_text(json.dumps({"reliable": {"select": rel_ids}}))
        (r.dir / "aa.json").write_text(json.dumps({"n_gate": 6}))
        (r.dir / "behavioral_bank.json").write_text(json.dumps([dict(x, side="src") for x in beh_rows]))
        (r.dir / "episodes").mkdir(parents=True, exist_ok=True)
        (r.dir / "gates").mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr(r.ev, "servers", lambda m: nullcontext([]))
        monkeypatch.setattr(
            r.ev,
            "run",
            lambda m, jobs, tag, side, srvs=None: [
                {
                    "task": t.id, "family": t.family, "split": t.split, "seed": s, "side": side,
                    "success": True, "detectors": {"fmt": (cand_fmt_trigger if side == "cand" else False)},
                }
                for t, s in jobs
            ],
        )
        return r.gate(str(gguf_p), "g1")

    # 1) Explicit None detector_bound / margin / alpha / looks / max_eps_per_task
    res_none_fields = _run_gate(
        "r-gate-none-fields",
        {"detector_bound": None, "margin": None, "alpha": None, "looks": None, "max_eps_per_task": None},
        cand_fmt_trigger=True,
    )
    assert res_none_fields["detectors"]["fmt"]["ok"] is False
    assert res_none_fields["pass"] is False

    # 2) stats=None altogether
    res_stats_none = _run_gate("r-gate-stats-none", None, cand_fmt_trigger=False)
    assert res_stats_none["detectors"]["fmt"]["ok"] is True
    assert res_stats_none["pass"] is True

    # 3) Explicit detector_bound=0.0 and margin=0.0 preserved
    res_zero = _run_gate("r-gate-zero", {"detector_bound": 0.0, "margin": 0.0}, cand_fmt_trigger=False)
    assert res_zero["detectors"]["fmt"]["ok"] is True
    assert res_zero["margin"] == 0.0


def test_pipeline_proxy_and_proxy_classifier_explicit_none_and_zero_preservation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    gguf_p = _write_gguf(tmp_path / "m.gguf")
    hf_p = _make_hf_dir(tmp_path)

    # Part A: classifier proxy (_proxy_classifier) with stats_docs=None, proxy=None, and general_docs=0/general_n_seq=0
    seen_stats_docs: list[Any] = []
    seen_gen_docs: list[Any] = []
    seen_pack: list[tuple[int, int]] = []

    cls_env = _make_stub_env(classifier=True)
    cls_env.proxy_spec = lambda scores, **kw: [{"name": "neg", "role": "neg", "n": 1, "ids": ["cal-0"]}, {"name": "pos", "role": "pos", "n": 1, "ids": ["search-famA-0"]}]
    def _trj_for(tasks):
        p = tmp_path / "stub_trj.bin"
        p.write_bytes(b"TRJ1")
        return str(p)
    cls_env.trj_for = _trj_for
    def _stats_texts(n):
        seen_stats_docs.append(n)
        if not isinstance(n, int):
            raise ValueError(f"n must be int, got {n!r}")
        return ["sample doc"] * max(0, n)
    cls_env.stats_texts = _stats_texts

    monkeypatch.setattr(pipeline, "load_environment", lambda mod, **kw: cls_env)
    monkeypatch.setattr(sanity, "general_texts", lambda n, seed, cache: (seen_gen_docs.append(n), ["gen text"] * n)[1])
    monkeypatch.setattr(kl, "pack_text", lambda R, texts, seq_len, n_seq: (seen_pack.append((seq_len, n_seq)), [([1, 2], [1])] * n_seq)[1])
    monkeypatch.setattr(kl, "write_trj", lambda p, recs: {"n_seq": len(recs), "n_targets": len(recs)})
    monkeypatch.setattr(kl, "traj_kl", lambda tc, mode, m, trj, st, out, log, **kw: (Path(out).write_text(json.dumps({"kl": 0.0})), {"kl": 0.0})[1])
    monkeypatch.setattr(pipeline, "run_imatrix", lambda tc, src, txt, dst, log, **kw: Path(dst).write_bytes(b"IM"))

    for run_id, proxy_cfg in (
        ("r-cls-proxy-sd-none", {"stats_docs": None, "general_docs": None, "general_seq_len": None, "general_n_seq": None}),
        ("r-cls-proxy-none", None),
        ("r-cls-proxy-zeros", {"stats_docs": 5, "general_docs": 0, "general_seq_len": 16, "general_n_seq": 0}),
    ):
        cfg = {
            "run_id": run_id,
            "run_root": str(tmp_path / "runs"),
            "source": {"gguf": str(gguf_p), "repo": "test/r", "revision": "abc", "hf_dir": str(hf_p)},
            "toolchain": {"llama_bin": str(tmp_path), "traj_kl": str(tmp_path / "tk"), "logodds": str(tmp_path / "lo"), "gpus": [""]},
            "env": {"module": "stub.env", "kwargs": {}},
            "ladder": ["BF16", "Q8_0", "Q4_K"],
            "start_tier": "Q4_K",
            "proxy": proxy_cfg,
            "sanity": {"behavioral_items": 0},
        }
        r = pipeline.Run(cfg)
        monkeypatch.setattr(r.ev, "scores", lambda m, ts: {t.id: 0.0 for t in ts})
        monkeypatch.setattr(r.scorer, "score_trj", lambda m, trj, pair: np.array([0.0, 0.0]))
        monkeypatch.setattr(r, "annotate_imatrix", lambda p: None)
        r.phase_proxy()
        assert (r.dir / "proxy" / "proxy.json").exists()

    assert seen_stats_docs == [200, 200, 5]
    assert seen_gen_docs == [30, 30, 0]
    assert seen_pack == [(1024, 4), (1024, 4), (16, 0)]

    # Part B: generative phase_proxy with proxy=None
    gen_env = _make_stub_env(classifier=False)
    monkeypatch.setattr(pipeline, "load_environment", lambda mod, **kw: gen_env)
    r_gen = pipeline.Run({
        "run_id": "r-gen-proxy-none",
        "run_root": str(tmp_path / "runs"),
        "source": {"gguf": str(gguf_p), "repo": "test/r", "revision": "abc", "hf_dir": str(hf_p)},
        "toolchain": {"llama_bin": str(tmp_path), "traj_kl": str(tmp_path / "tk"), "logodds": str(tmp_path / "lo"), "gpus": [""]},
        "env": {"module": "stub.env", "kwargs": {}},
        "ladder": ["BF16", "Q8_0", "Q4_K"],
        "start_tier": "Q4_K",
        "proxy": None,
        "sanity": {"behavioral_items": 0},
    })
    (r_gen.dir / "episodes").mkdir(parents=True, exist_ok=True)
    (r_gen.dir / "pool.json").write_text(json.dumps({"reliable": {"search": ["search-famA-0"]}}))
    (r_gen.dir / "episodes" / "pool.jsonl").write_text(json.dumps({
        "split": "search", "success": True, "task": "search-famA-0",
        "messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "ok"}],
        "tools": None,
    }) + "\n")
    monkeypatch.setattr(r_gen, "annotate_imatrix", lambda p: None)
    r_gen.phase_proxy()
    assert (r_gen.dir / "proxy" / "proxy.json").exists()


def test_pipeline_pool_and_screen_explicit_none_and_falsy_preservation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from contextlib import nullcontext

    gguf_p = _write_gguf(tmp_path / "m.gguf")
    hf_p = _make_hf_dir(tmp_path)
    monkeypatch.setattr(pipeline, "load_environment", lambda mod, **kw: _make_stub_env(classifier=False))
    monkeypatch.setattr(sanity, "run_behavioral", lambda *a, **kw: [])

    base_cfg = {
        "run_root": str(tmp_path / "runs"),
        "source": {"gguf": str(gguf_p), "repo": "test/r", "revision": "abc", "hf_dir": str(hf_p)},
        "toolchain": {"llama_bin": str(tmp_path), "traj_kl": str(tmp_path / "tk"), "logodds": str(tmp_path / "lo"), "gpus": [""]},
        "env": {"module": "stub.env", "kwargs": {}},
        "ladder": ["BF16", "Q8_0", "Q4_K"],
        "start_tier": "Q4_K",
        "sanity": {"behavioral_items": 0},
    }

    # 1) phase_pool with pool=None, pool={"runs": None, "min_pass": None}, and min_pass=0 preserved
    r_pool_none = pipeline.Run({**base_cfg, "run_id": "r-pool-none", "pool": None})
    monkeypatch.setattr(r_pool_none.ev, "run", lambda m, jobs, tag, side: [])
    r_pool_none.phase_pool()
    assert json.loads((r_pool_none.dir / "pool.json").read_text())["criterion"] == ">= 1/1"

    r_pool_subnone = pipeline.Run({**base_cfg, "run_id": "r-pool-subnone", "pool": {"tasks_per_split": {"select": 2}, "runs": None, "min_pass": None}})
    monkeypatch.setattr(
        r_pool_subnone.ev,
        "run",
        lambda m, jobs, tag, side: [{"task": t.id, "family": t.family, "split": t.split, "seed": s, "side": side, "success": True} for t, s in jobs],
    )
    r_pool_subnone.phase_pool()
    p_sub = json.loads((r_pool_subnone.dir / "pool.json").read_text())
    assert p_sub["criterion"] == ">= 1/1" and len(p_sub["reliable"]["select"]) == 2

    r_pool_zero = pipeline.Run({**base_cfg, "run_id": "r-pool-zero", "pool": {"tasks_per_split": {"select": 2}, "runs": 2, "min_pass": 0}})
    monkeypatch.setattr(
        r_pool_zero.ev,
        "run",
        lambda m, jobs, tag, side: [{"task": t.id, "family": t.family, "split": t.split, "seed": s, "side": side, "success": False} for t, s in jobs],
    )
    r_pool_zero.phase_pool()
    p_zero = json.loads((r_pool_zero.dir / "pool.json").read_text())
    assert p_zero["criterion"] == ">= 0/2" and len(p_zero["reliable"]["select"]) == 2

    # 2) screen with screen=None, screen={"pairs": None, "fail_below": None}, and pairs=0 / fail_below=0.0 preserved
    def _prep_screen(run_id: str, screen_cfg: Any):
        r = pipeline.Run({**base_cfg, "run_id": run_id, "screen": screen_cfg})
        rel_ids = [f"select-{fam}-{i}" for fam in ("famA", "famB", "famC") for i in range(2)]
        (r.dir / "pool.json").write_text(json.dumps({"reliable": {"select": rel_ids}}))
        (r.dir / "behavioral_bank.json").write_text("[]")
        (r.dir / "episodes").mkdir(parents=True, exist_ok=True)
        (r.dir / "episodes" / "aa-B.jsonl").write_text(
            "\n".join(json.dumps({"task": tid, "family": tid.split("-")[1], "split": "select", "seed": 0, "side": "src", "success": True}) for tid in rel_ids) + "\n"
        )
        monkeypatch.setattr(r.ev, "servers", lambda m: nullcontext([]))
        monkeypatch.setattr(
            r.ev,
            "run",
            lambda m, jobs, tag, side, srvs=None: [
                {"task": t.id, "family": t.family, "split": t.split, "seed": s, "side": side, "success": True}
                for t, s in jobs
            ],
        )
        return r

    out_none = _prep_screen("r-scr-none", None).screen(str(gguf_p), "s1")
    assert out_none["n"] == 6 and out_none["task_ok"] is True

    out_subnone = _prep_screen("r-scr-subnone", {"pairs": None, "fail_below": None}).screen(str(gguf_p), "s1")
    assert out_subnone["n"] == 6 and out_subnone["task_ok"] is True

    out_zero = _prep_screen("r-scr-zero", {"pairs": 0, "fail_below": 0.0}).screen(str(gguf_p), "s1")
    assert out_zero["n"] == 0 and math.isnan(out_zero["delta"]) and out_zero["task_ok"] is False


def test_pipeline_exam_compare_explicit_none_and_make_renderer_reasoning_effort(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    gguf_p = _write_gguf(tmp_path / "m.gguf")
    hf_p = _make_hf_dir(tmp_path, dialect="inkling2")

    # 1) exam_compare with exam=None and exam={"external": None, "boot": None}
    tasks_cmp = [
        Task(id="cal-0", family="cal", split="calibration", prompt={"text": "c0"}, meta={"label": "human"}),
        Task(id="cal-1", family="cal", split="calibration", prompt={"text": "c1"}, meta={"label": "human"}),
        Task(id="srcA:e0", family="famA", split="exam", prompt={"text": "e0"}, meta={"label": "ai", "kind": "k1"}),
        Task(id="srcA:e1", family="famA", split="exam", prompt={"text": "e1"}, meta={"label": "human", "kind": "k1"}),
        Task(id="srcB:e2", family="famB", split="exam", prompt={"text": "e2"}, meta={"label": "ai", "kind": "k2"}),
        Task(id="srcB:e3", family="famB", split="exam", prompt={"text": "e3"}, meta={"label": "human", "kind": "k2"}),
    ]
    class _CmpEnv:
        name = "cmp-env"
        version = "1"
        classifier = True
        pair = (14, 15)
        deployment = {"context": 512, "concurrency": 1, "kv_type": "f16"}
        budget = {"max_turns": 1, "max_tokens_per_turn": 1}
        def tasks(self):
            return list(tasks_cmp)
        def decide(self, t, score, rule):
            return SimpleNamespace(tests_passed=(t.meta["label"] == "ai"))

    monkeypatch.setattr(pipeline, "load_environment", lambda mod, **kw: _CmpEnv())
    for run_id, exam_cfg in (("r-excmp-none", None), ("r-excmp-extnone", {"external": None, "boot": None})):
        r = pipeline.Run({
            "run_id": run_id,
            "run_root": str(tmp_path / "runs"),
            "source": {"gguf": str(gguf_p), "repo": "test/r", "revision": "abc", "hf_dir": str(hf_p)},
            "toolchain": {"llama_bin": str(tmp_path), "traj_kl": str(tmp_path / "tk"), "logodds": str(tmp_path / "lo"), "gpus": [""]},
            "env": {"module": "stub.env", "kwargs": {}},
            "ladder": ["BF16", "Q8_0", "Q4_K"],
            "start_tier": "Q4_K",
            "exam": exam_cfg,
            "sanity": {"behavioral_items": 0},
        })
        (r.dir / "gates").mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr(r.ev, "rule", lambda p: 0.0)
        monkeypatch.setattr(r.ev, "scores", lambda path, ts: {t.id: (-1.0 if t.meta["label"] == "ai" else 1.0) for t in ts})
        r.exam_compare(str(gguf_p))
        assert (r.dir / "gates" / "exam_compare.json").exists()

    # 2) _make_renderer reasoning_effort fallback & decoding precedence
    class _NoReEnv:
        name = "nore"
        version = "1"
        classifier = False
        deployment = {"context": 512, "concurrency": 1, "kv_type": "f16"}
        budget = {"max_turns": 1, "max_tokens_per_turn": 64}
        def tasks(self):
            return []

    monkeypatch.setattr(pipeline, "load_environment", lambda mod, **kw: _NoReEnv())
    r_nore = pipeline.Run({
        "run_id": "r-rend-nore",
        "run_root": str(tmp_path / "runs"),
        "source": {"gguf": str(gguf_p), "repo": "test/r", "revision": "abc", "hf_dir": str(hf_p)},
        "toolchain": {"llama_bin": str(tmp_path), "traj_kl": str(tmp_path / "tk"), "logodds": str(tmp_path / "lo"), "gpus": [""]},
        "env": {"module": "stub.env", "kwargs": {}},
        "ladder": ["BF16", "Q8_0", "Q4_K"],
        "start_tier": "Q4_K",
        "decoding": {"temperature": 0.0, "reasoning_effort": None},
        "sanity": {"behavioral_items": 0},
    })
    assert r_nore._make_renderer().reasoning_effort == "none"

    monkeypatch.setattr(pipeline, "load_environment", lambda mod, **kw: _make_stub_env(classifier=False, reasoning_effort="none"))
    r_dec_low = pipeline.Run({
        "run_id": "r-rend-declow",
        "run_root": str(tmp_path / "runs"),
        "source": {"gguf": str(gguf_p), "repo": "test/r", "revision": "abc", "hf_dir": str(hf_p)},
        "toolchain": {"llama_bin": str(tmp_path), "traj_kl": str(tmp_path / "tk"), "logodds": str(tmp_path / "lo"), "gpus": [""]},
        "env": {"module": "stub.env", "kwargs": {}},
        "ladder": ["BF16", "Q8_0", "Q4_K"],
        "start_tier": "Q4_K",
        "decoding": {"temperature": 0.0, "reasoning_effort": "low"},
        "sanity": {"behavioral_items": 0},
    })
    assert r_dec_low._make_renderer().reasoning_effort == "low"

    monkeypatch.setattr(pipeline, "load_environment", lambda mod, **kw: _make_stub_env(classifier=False, reasoning_effort="low"))
    r_env_low = pipeline.Run({
        "run_id": "r-rend-envlow",
        "run_root": str(tmp_path / "runs"),
        "source": {"gguf": str(gguf_p), "repo": "test/r", "revision": "abc", "hf_dir": str(hf_p)},
        "toolchain": {"llama_bin": str(tmp_path), "traj_kl": str(tmp_path / "tk"), "logodds": str(tmp_path / "lo"), "gpus": [""]},
        "env": {"module": "stub.env", "kwargs": {}},
        "ladder": ["BF16", "Q8_0", "Q4_K"],
        "start_tier": "Q4_K",
        "decoding": {"temperature": 0.0, "reasoning_effort": None},
        "sanity": {"behavioral_items": 0},
    })
    assert r_env_low._make_renderer().reasoning_effort == "low"


# =============================================================================
# Iteration 5 Remediation Regression Tests (Strict Test-First)
# =============================================================================


def test_learn_auto_checkpointing_explicit_none_and_bool_preservation(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda dev: SimpleNamespace(total_memory=16 << 30))

    m_q3 = SimpleNamespace(config=SimpleNamespace(model_type="qwen3"))
    assert learn._auto_checkpointing(m_q3, {}, "cuda:0") is True
    assert learn._auto_checkpointing(m_q3, {"checkpointing": None}, "cuda:0") is True
    assert learn._auto_checkpointing(m_q3, {"checkpointing": False}, "cuda:0") is False
    assert learn._auto_checkpointing(m_q3, {"checkpointing": True}, "cuda:0") is True

    # >55 GiB GPU with qwen3_5 (without fla) still auto-enables when checkpointing is None
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda dev: SimpleNamespace(total_memory=80 << 30))
    monkeypatch.setitem(sys.modules, "fla", None)
    m_q35 = SimpleNamespace(config=SimpleNamespace(model_type="qwen3_5"))
    assert learn._auto_checkpointing(m_q35, {"checkpointing": None}, "cuda:0") is True
    assert learn._auto_checkpointing(m_q35, {"checkpointing": False}, "cuda:0") is False

    # CPU always disables unless explicitly True
    assert learn._auto_checkpointing(m_q3, {"checkpointing": None}, "cpu") is False
    assert learn._auto_checkpointing(m_q3, {"checkpointing": True}, "cpu") is True


def test_learn_run_generative_explicit_none_and_falsy_preservation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    sel_tasks = []
    for fam in ("famA", "famB", "famC"):
        for idx in range(20):
            tid = f"{fam}-{idx}"
            if learn._side(tid) == 1:
                sel_tasks.append(Task(id=tid, family=fam, split="select", prompt="p", meta={}))
                if len([t for t in sel_tasks if t.family == fam]) >= 2:
                    break
    train_tasks = [Task(id="tr-0", family="famA", split="train", prompt="p", meta={})]
    all_tasks = sel_tasks + train_tasks

    class MockEnv:
        deployment = {"context": 512, "concurrency": 1, "kv_type": "f16"}
        def tasks(self):
            return list(all_tasks)

    ev_calls: list[dict[str, Any]] = []
    gate_calls: list[dict[str, Any]] = []
    sel_jobs_count: list[int] = []

    class FakeEv:
        def __init__(self, tc, env, decoding, slots, ctx, kv, log_dir, **kw):
            ev_calls.append({"decoding": decoding, "slots": slots, "ctx": ctx, "kv": kv, **kw})
        def run(self, model, jobs, tag, side):
            if tag == "learn-learned-sel":
                sel_jobs_count.append(len(jobs))
            return [{"task": t.id, "family": t.family, "split": t.split, "seed": s, "side": side, "success": True} for t, s in jobs]

    import chantless.evaluator as evmod
    monkeypatch.setattr("chantless.env.protocol.load_environment", lambda mod, **kw: MockEnv())
    monkeypatch.setattr(evmod, "Evaluator", FakeEv)
    monkeypatch.setattr(learn, "_tokenize_seq_lessons", lambda tc, m, rows, **kw: rows)
    monkeypatch.setattr(learn, "_load_replay_trajs", lambda tc, m, sp, **kw: [])
    def _fake_lgs(art_path, config_dir, feedback, replay, out, cfg=None, **kw):
        Path(out).write_bytes(b"G")
        gate_calls.append({"out": Path(out).name, "n_lessons": len(feedback), "cfg": dict(cfg or {})})
        return {"off": 1}
    monkeypatch.setattr(learn, "learn_gates_seq", _fake_lgs)
    monkeypatch.setattr(learn, "sha256", lambda p: "0")
    monkeypatch.setattr(kl, "write_trj", lambda p, recs: {"n_seq": len(recs), "n_targets": len(recs)})
    monkeypatch.setattr(kl, "traj_kl", lambda *a, **kw: {"trajs": [{"nll_cand": 1.0 if "base" in a[6] else 0.5}] * 30})

    model_p = tmp_path / "m.gguf"
    model_p.write_bytes(b"stub")
    lessons_30 = [
        {"id": f"l-{i}", "family": f"f{i % 3}", "prompt_tokens": [1, 2], "corrected_tokens": [3]}
        for i in range(30)
    ]

    # Case A: All nullable/defaultable fields explicitly None (must not crash, defaults apply: slots=1, max_lessons=25, controls=True, alpha=0.05, gate_margin=0.03)
    cfg_none = {
        "toolchain": {"llama_bin": str(tmp_path), "traj_kl": str(tmp_path / "tk"), "gpus": [""]},
        "env": {"module": "mock.env", "kwargs": None},
        "serving": {"slots_per_gpu": None, "ctx_per_slot": None, "special": None, "flash_attn": None},
        "decoding": None,
        "learn": {
            "mode": "generative",
            "method": "gates_seq",
            "model": str(model_p),
            "out_dir": str(tmp_path / "out-none"),
            "lessons": lessons_30,
            "judge_slots": None,
            "max_lessons": None,
            "gate_margin": None,
            "alpha": None,
            "controls": None,
            "judge_cache_prompt": None,
            "select_limit": None,
            "switch_mode": None,
            "side_seed": None,
            "heldout_ids": ["tr-0"],
            "gate_ref_model": None,
        },
    }
    res_none = learn.run_generative(cfg_none)
    assert ev_calls[-1]["slots"] == 1
    assert ev_calls[-1]["ctx"] == 512
    assert ev_calls[-1]["cache_prompt"] is False
    assert ev_calls[-1]["flash_attn"] == "on"
    assert ev_calls[-1]["special"] is None
    assert gate_calls[0]["n_lessons"] == 25
    assert any(c["out"].startswith("control-empty-") for c in gate_calls)
    assert sel_jobs_count[-1] == len(sel_tasks)
    assert res_none["env_eval"]["learned"]["gate_pass"] is True

    # Case B: Preserve explicit falsy values (gate_margin=0.0 -> L=0.0 > -0.0 is False; controls=False; judge_cache_prompt=False; select_limit=0)
    gate_calls.clear()
    cfg_falsy = {
        **cfg_none,
        "learn": {
            **cfg_none["learn"],
            "out_dir": str(tmp_path / "out-falsy"),
            "gate_margin": 0.0,
            "controls": False,
            "judge_cache_prompt": False,
        },
    }
    res_falsy = learn.run_generative(cfg_falsy)
    assert res_falsy["env_eval"]["learned"]["gate_pass"] is False
    assert len(gate_calls) == 1
    assert ev_calls[-1]["cache_prompt"] is False

    # Case C: select_limit=0 preserved (runs 0 select tasks)
    sel_jobs_count.clear()
    cfg_lim0 = {
        **cfg_falsy,
        "learn": {**cfg_falsy["learn"], "out_dir": str(tmp_path / "out-lim0"), "select_limit": 0},
    }
    learn.run_generative(cfg_lim0)
    assert sel_jobs_count == []


def test_learn_gates_and_learn_gates_seq_explicit_none_and_falsy_preservation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    class _CapturedBuild(RuntimeError):
        pass

    class FakeArt:
        tensors = {"blk.0.ffn_up.weight": {"dims": [4, 4]}}
        def close(self):
            pass

    seen_frozen: list[bool] = []
    monkeypatch.setattr(learn.lattice, "Artifact", lambda p: FakeArt())
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda dev: SimpleNamespace(total_memory=80 << 30))

    def _capture_build_student(art, config_dir, train, ggml_lib, device="cpu", latents=None, frozen_bf16=False):
        seen_frozen.append(bool(frozen_bf16))
        raise _CapturedBuild("captured")

    monkeypatch.setattr(learn, "build_student", _capture_build_student)

    # 1) frozen_bf16=None when amp=True (on CUDA) MUST resolve to True, while frozen_bf16=False -> False, amp=False -> False
    for fn_name in ("learn_gates_seq", "learn_gates"):
        for cfg_in, expected in (
            ({"frozen_bf16": None, "amp": None}, True),
            ({"frozen_bf16": None, "amp": False}, False),
            ({"frozen_bf16": False, "amp": True}, False),
        ):
            seen_frozen.clear()
            with pytest.raises(_CapturedBuild):
                if fn_name == "learn_gates_seq":
                    learn.learn_gates_seq(
                        str(tmp_path / "m.gguf"), None,
                        [{"pos_tokens": [1, 2, 3], "pos_targets": [1, 2]}], [],
                        str(tmp_path / "o.gguf"), cfg=cfg_in, device="cuda:0",
                    )
                else:
                    learn.learn_gates(
                        str(tmp_path / "m.gguf"), None,
                        [{"tokens": np.array([1, 2, 3]), "label": "human"}],
                        [np.array([1, 2, 3])], (0, 1), 0.0,
                        str(tmp_path / "o.gguf"), cfg=cfg_in, device="cuda:0",
                    )
            assert seen_frozen == [expected], (fn_name, cfg_in, seen_frozen)

    # 2) Full execution on CPU with ALL nullable/defaultable fields set to None (must not crash with TypeError)
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
            bs = len(seqs)
            max_l = max(len(s) for s in seqs)
            n = int(-(-max_l // round_to) * round_to) if round_to > 1 else max_l
            x = torch.ones((bs, n, 4), dtype=torch.float32)
            y = self.lin(x)
            return self.hook_mod(y)

    tiny = TinySeqStudent()
    seen_round_to: list[int] = []
    seen_adam_eps: list[float] = []
    orig_adamw = torch.optim.AdamW

    def _spy_adamw(params, *a, **kw):
        seen_adam_eps.append(float(kw.get("eps", -1.0)))
        return orig_adamw(params, *a, **kw)

    monkeypatch.setattr(torch.optim, "AdamW", _spy_adamw)
    monkeypatch.setattr(learn, "build_student", lambda *a, **kw: (tiny, None))
    monkeypatch.setattr(learn, "_switchable_layers", lambda *a, **kw: [0])
    monkeypatch.setattr(learn, "_layer_ffn_width", lambda art, il: 4)
    monkeypatch.setattr(learn, "_switch_hook_module", lambda layer: tiny.hook_mod)
    monkeypatch.setattr(learn.PrefixCache, "build", staticmethod(lambda *a, **kw: None))
    monkeypatch.setattr(learn, "_forward_backbone", lambda model, seqs, dev, amp, round_to=32, **kw: (seen_round_to.append(round_to), tiny.forward_bb(seqs, round_to=round_to))[1])
    monkeypatch.setattr(
        learn,
        "_apply_switches_to_file_and_mask",
        lambda art, ap, out, layers, theta, sm, **kw: (
            Path(out).write_bytes(b"G"),
            ({il: [i for i, v in enumerate(theta[il].detach().tolist()) if v <= 0] for il in layers}, 4, {0: 4}, None),
        )[1],
    )
    monkeypatch.setattr(learn, "sha256", lambda p: "0")

    all_none_cfg = {
        "pad_multiple": None, "steps": None, "gate_lr": None, "adam_eps": None,
        "nll_weight": None, "margin_weight": None, "margin": None, "replay_weight": None,
        "gate_init": None, "project": None, "amp": None, "frozen_bf16": None,
        "checkpointing": None, "switch_mode": None, "seed": None, "kl_temp": None,
        "feedback_batch": None, "replay_batch": None, "gate_last_blocks": None,
        "prefix_cache_max_mb": None, "free_prefix": None, "stream_prefix": None,
        "score_budget": None, "train_budget": None, "pair_margin": None,
        "pair_weight": None, "guard_weight": None, "keep_tau": None, "keep_window": None,
        "sep_humans": None,
    }

    in_gguf = tmp_path / "in.gguf"
    in_gguf.write_bytes(b"IN")
    out_seq = tmp_path / "out_seq_none.gguf"
    rec_seq = learn.learn_gates_seq(
        str(in_gguf), None,
        [{"pos_tokens": [1, 2, 3, 4], "pos_targets": [2, 3], "neg_tokens": [1, 2, 4, 5], "neg_targets": [2, 3]}],
        [(np.array([1, 2, 3, 4], dtype=np.int32), np.array([2, 3], dtype=np.uint32))],
        str(out_seq), cfg={**all_none_cfg, "steps": 1}, device="cpu",
    )
    assert rec_seq["steps"] == 1 and seen_round_to[-1] == 32 and seen_adam_eps[-1] == pytest.approx(1e-4)

    # Also test learn_gates (classifier path) with all_none_cfg
    monkeypatch.setattr(
        learn, "scores",
        lambda model, docs, pair, dev, amp, prefix_cache=None: torch.stack([tiny.lm_head(tiny.hook_mod(tiny.lin(torch.ones((1, 4)))))[0, :2].sum() for _ in docs]),
    )
    out_cls = tmp_path / "out_cls_none.gguf"
    rec_cls = learn.learn_gates(
        str(in_gguf), None,
        [
            {"tokens": np.array([1, 2, 3], dtype=np.int64), "label": "human", "pair": 0},
            {"tokens": np.array([1, 2, 4], dtype=np.int64), "label": "ai", "pair": 0},
        ],
        [np.array([1, 2, 3], dtype=np.int64)], (0, 1), 0.0,
        str(out_cls), cfg={**all_none_cfg, "steps": 1}, strata=["human"], device="cpu",
    )
    assert rec_cls["steps"] == 1 and seen_adam_eps[-1] == pytest.approx(1e-8)

    # 3) Explicit 0 / 0.0 / False preservation in learn_gates_seq (zero weights -> zero gradient update -> loss_last == loss_first or zero grads)
    seen_round_to.clear()
    out_zero = tmp_path / "out_seq_zero.gguf"
    rec_zero = learn.learn_gates_seq(
        str(in_gguf), None,
        [{"pos_tokens": [1, 2, 3, 4], "pos_targets": [2, 3], "neg_tokens": [1, 2, 4, 5], "neg_targets": [2, 3]}],
        [(np.array([1, 2, 3, 4], dtype=np.int32), np.array([2, 3], dtype=np.uint32))],
        str(out_zero),
        cfg={
            "steps": 1, "pad_multiple": 1, "nll_weight": 0.0, "margin_weight": 0.0,
            "margin": 0.0, "replay_weight": 0.0, "gate_init": 0.0, "project": False, "amp": False,
        },
        device="cpu",
    )
    assert seen_round_to == [1, 1]
    assert rec_zero["agem_conflicts"] == 0
    assert rec_zero["off"] == 4  # gate_init=0.0 preserved -> theta <= 0 -> all 4 switched off when weights=0.0


def test_learn_run_classifier_and_judge_explicit_none_and_falsy_preservation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from chantless import runtime as rtmod

    sel_rows = [
        {"task": f"beemo-{i}", "family": "beemo", "verdict": "human", "src": True}
        for i in range(700)
    ] + [
        {"task": f"beemo-ai-{i}", "family": "beemo", "verdict": "ai", "src": True}
        for i in range(700)
    ] + [
        {"task": f"wo-{i}", "family": "wo", "verdict": "human", "src": True}
        for i in range(700)
    ] + [
        {"task": f"wo-ai-{i}", "family": "wo", "verdict": "ai", "src": True}
        for i in range(700)
    ]
    fb_rows = [
        {"index": 0, "label": "human", "pair": 0, "fold": 0},
        {"index": 1, "label": "ai", "pair": 0, "fold": 0},
        {"index": 2, "label": "human", "pair": 1, "fold": 1},
        {"index": 3, "label": "ai", "pair": 1, "fold": 1},
    ]
    sel_p = tmp_path / "select.json"
    fb_p = tmp_path / "fb.json"
    sel_p.write_text(json.dumps(sel_rows))
    fb_p.write_text(json.dumps(fb_rows))

    pool_trj = tmp_path / "pool.trj"
    fb_trj = tmp_path / "fb.trj"
    n_cal = 1000
    learn.write_trj([np.array([1, 2], dtype=np.int32) for _ in range(n_cal + len(sel_rows))], str(pool_trj), 0)
    learn.write_trj([np.array([1, 2], dtype=np.int32) for _ in range(len(fb_rows))], str(fb_trj), 0)

    base_m = tmp_path / "base.gguf"
    base_m.write_bytes(b"BASE")

    class FakeScorer:
        def __init__(self, tc, slots):
            self.slots = slots
        def score_trj(self, model, trj_path, pair):
            if str(trj_path) == str(pool_trj):
                arr = np.ones(n_cal + len(sel_rows), dtype=np.float32)
                for idx, r in enumerate(sel_rows):
                    arr[n_cal + idx] = -1.0 if r["verdict"] == "ai" else 1.0
                return arr
            if Path(model).name.startswith("model-"):
                return np.array([2.0, -2.0, 2.0, -2.0], dtype=np.float32)
            return np.array([1.0, -1.0, 1.0, -1.0], dtype=np.float32)

    monkeypatch.setattr(rtmod, "BatchScorer", FakeScorer)
    monkeypatch.setattr(learn, "learn_gates", lambda *a, **kw: (Path(a[6]).write_bytes(b"CAND"), {"ok": True})[1])

    cfg_none = {
        "toolchain": {"llama_bin": str(tmp_path), "traj_kl": str(tmp_path / "tk"), "gpus": [""]},
        "learn": {
            "mode": "classifier",
            "method": "gates",
            "model": str(base_m),
            "out_dir": str(tmp_path / "out-cls-none"),
            "pair": [0, 1],
            "n_calibration": n_cal,
            "cutoff": 0.0,
            "select_rows": str(sel_p),
            "feedback_rows": str(fb_p),
            "pool_trj": str(pool_trj),
            "feedback_trj": str(fb_trj),
            "seed": None,
            "replay_humans": None,
            "replay_select": None,
            "stratify": None,
            "slots_per_gpu": None,
            "tails": None,
            "tail_n": None,
            "near_n": None,
            "max_human_rise": None,
            "max_ai_drop": None,
            "gate_margin": None,
            "cache_dir": None,
            "side_seed": None,
        },
    }
    res = learn.run(cfg_none, fold=0)
    assert res["accepted"] is True

    # Explicit 0.0 gate_margin -> cand L=0.0 > -0.0 is False -> gate_pass=False -> accepted=False
    cfg_zero = {
        **cfg_none,
        "learn": {
            **cfg_none["learn"],
            "out_dir": str(tmp_path / "out-cls-zero"),
            "gate_margin": 0.0,
            "max_human_rise": 0.0,
            "max_ai_drop": 0.0,
        },
    }
    res_zero = learn.run(cfg_zero, fold=0)
    assert res_zero["candidate"]["gate_pass"] is False
    assert res_zero["accepted"] is False



