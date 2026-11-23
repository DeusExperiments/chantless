"""Deterministic offline regression tests for Milestone 3 (`M3_generative_envs_kl`).

Covers:
- `mvp/chantless/env/deepmath.py`
- `mvp/chantless/kl.py`
- `mvp/chantless/env/bfcl.py`
- `mvp/chantless/env/codetasks.py`
- `mvp/chantless/env/sandbox.py`
- `mvp/chantless/detectors.py`
- `mvp/chantless/env/families.py`
"""
from __future__ import annotations

import copy
import json
import math
import os
import random
import struct
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import numpy as np
import pytest
from tokenizers import Tokenizer, decoders, models, pre_tokenizers
from transformers import AutoTokenizer, PreTrainedTokenizerFast

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "mvp") not in sys.path:
    sys.path.insert(0, str(ROOT / "mvp"))

from chantless import client as client_mod
from chantless import detectors, evaluator, gguf_ops, kl, runtime
from chantless.env import bfcl, codetasks, deepmath, families, protocol, sandbox
from chantless.env.protocol import Episode, Task


# =============================================================================
# Offline Synthetic Helpers (self-contained, no `.local/` or network required)
# =============================================================================

_INKLING2_TEMPLATE = (
    "{%- set eff = reasoning_effort if reasoning_effort is defined and reasoning_effort is not none else 0.9 -%}"
    "{%- set effort_num = '0' if eff == 'none' else ('0.2' if eff == 'low' else ('0.9' if eff == 'high' else eff)) -%}"
    "{%- if tools -%}"
    "{{- '<|message_system|>tool_declare<|content_xml|>' ~ (tools | tojson) ~ '<|end_message|>' -}}"
    "{%- endif -%}"
    "{%- if messages and messages[0].role != 'system' -%}"
    "{{- '<|message_system|><|content_text|>Thinking effort level: ' ~ effort_num ~ '<|end_message|>' -}}"
    "{%- endif -%}"
    "{%- for m in messages -%}"
    "{%- if m.role == 'system' -%}"
    "{{- '<|message_system|><|content_text|>' ~ m.content ~ '<|end_message|>' -}}"
    "{{- '<|message_system|><|content_text|>Thinking effort level: ' ~ effort_num ~ '<|end_message|>' -}}"
    "{%- elif m.role == 'user' -%}"
    "{{- '<|message_user|><|content_text|>' ~ m.content ~ '<|end_message|>' -}}"
    "{%- elif m.role == 'tool' -%}"
    "{{- '<|message_tool|>' ~ (m.name if m.name is defined and m.name else '') ~ '<|content_text|>' ~ m.content ~ '<|end_message|>' -}}"
    "{%- elif m.role == 'assistant' -%}"
    "{%- if m.reasoning_content is defined and m.reasoning_content -%}"
    "{{- '<|message_model|><|content_thinking|>' ~ m.reasoning_content ~ '<|end_message|>' -}}"
    "{%- endif -%}"
    "{%- if m.tool_calls -%}"
    "{%- for tc in m.tool_calls -%}"
    "{{- '<|message_model|>' ~ tc.function.name ~ '<|content_invoke_tool_json|>' ~ ({'name': tc.function.name, 'args': tc.function.arguments} | tojson) ~ '<|end_message|>' -}}"
    "{%- endfor -%}"
    "{{- '<|content_model_end_sampling|>' -}}"
    "{%- elif m.content or not (m.reasoning_content is defined and m.reasoning_content) -%}"
    "{{- '<|message_model|><|content_text|>' ~ (m.content or '') ~ '<|end_message|><|content_model_end_sampling|>' -}}"
    "{%- else -%}"
    "{{- '<|content_model_end_sampling|>' -}}"
    "{%- endif -%}"
    "{%- endif -%}"
    "{%- endfor -%}"
)

_QWEN3_TEMPLATE = (
    "{%- for m in messages -%}"
    "{{- '<|im_start|>' ~ m.role ~ '\\n' ~ (m.content or '') ~ '<|im_end|>\\n' -}}"
    "{%- endfor -%}"
    "{%- if add_generation_prompt -%}"
    "{%- if enable_thinking is defined and not enable_thinking -%}"
    "{{- '<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n' -}}"
    "{%- else -%}"
    "{{- '<|im_start|>assistant\\n' -}}"
    "{%- endif -%}"
    "{%- endif -%}"
)


def _build_synthetic_hf_dir(out_dir: Path, dialect: str = "inkling2") -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    alphabet = list(pre_tokenizers.ByteLevel.alphabet())
    vocab = {tok: i for i, tok in enumerate(alphabet)}
    if dialect == "inkling2":
        special = [
            "<unk>",
            "<|content_model_end_sampling|>",
            "<|message_system|>",
            "<|message_user|>",
            "<|message_model|>",
            "<|message_tool|>",
            "<|content_text|>",
            "<|content_thinking|>",
            "<|content_invoke_tool_json|>",
            "<|content_xml|>",
            "<|end_message|>",
        ]
        eos_tok = "<|content_model_end_sampling|>"
        ct = _INKLING2_TEMPLATE
    else:
        special = ["<unk>", "<|im_end|>", "<|im_start|>", "<think>", "</think>", "<tool_call>", "</tool_call>"]
        eos_tok = "<|im_end|>"
        ct = _QWEN3_TEMPLATE

    for tok_s in special:
        if tok_s not in vocab:
            vocab[tok_s] = len(vocab)
    bpe = Tokenizer(models.BPE(vocab=vocab, merges=[], unk_token="<unk>"))
    bpe.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    bpe.decoder = decoders.ByteLevel()
    tok = PreTrainedTokenizerFast(
        tokenizer_object=bpe,
        eos_token=eos_tok,
        unk_token="<unk>",
        additional_special_tokens=[s for s in special if s != "<unk>"],
        chat_template=ct,
    )
    tok.save_pretrained(out_dir)
    return out_dir


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


def _write_minimal_gguf(path: Path, arch: str, tokens: list[bytes]) -> Path:
    v_size = len(tokens)
    ttypes = [3 if (t.startswith(b"<") and t.endswith(b">")) else 1 for t in tokens]
    scores = [0.0] * v_size
    kvs: list[tuple[str, tuple[int, bytes]]] = [
        ("general.architecture", _pack_kv_str(arch)),
        ("general.alignment", _pack_kv_u32(32)),
        (f"{arch}.block_count", _pack_kv_u32(1)),
        (f"{arch}.embedding_length", _pack_kv_u32(64)),
        (f"{arch}.attention.head_count", _pack_kv_u32(2)),
        (f"{arch}.attention.head_count_kv", _pack_kv_u32(2)),
        (f"{arch}.attention.key_length", _pack_kv_u32(32)),
        (f"{arch}.feed_forward_length", _pack_kv_u32(64)),
        ("tokenizer.ggml.tokens", _pack_kv_str_arr(tokens)),
        ("tokenizer.ggml.token_type", _pack_kv_i32_arr(ttypes)),
        ("tokenizer.ggml.scores", _pack_kv_f32_arr(scores)),
        ("tokenizer.ggml.bos_token_id", _pack_kv_u32(0)),
        ("tokenizer.ggml.eos_token_id", _pack_kv_u32(1)),
    ]
    n_embd = 64
    arr = np.zeros((v_size, n_embd), dtype=np.float32)
    bf16 = (arr.view(np.uint32) >> 16).astype(np.uint16).tobytes()
    tensors = [("token_embd.weight", [n_embd, v_size], 30, bf16)]
    kv_buf = bytearray()
    for k, (vt, raw) in kvs:
        kb = k.encode("utf-8")
        kv_buf += struct.pack("<Q", len(kb)) + kb + struct.pack("<I", vt) + raw
    gguf_ops._write_gguf_raw(3, 32, len(kvs), bytes(kv_buf), tensors, path)
    return path


class _StubServer:
    def __init__(self, responder: Callable[[str, dict[str, Any]], tuple[int, Any]]):
        self.records: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        parent = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers.get("Content-Length", "0"))
                raw = self.rfile.read(length)
                body = json.loads(raw.decode("utf-8")) if raw else {}
                with parent._lock:
                    parent.records.append({"path": self.path, "body": body})
                status, payload = responder(self.path, body)
                data = json.dumps(payload).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, format, *args):
                return

        self._srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._th = threading.Thread(target=self._srv.serve_forever, daemon=True)
        self.url = f"http://127.0.0.1:{self._srv.server_port}"

    def __enter__(self) -> "_StubServer":
        self._th.start()
        return self

    def __exit__(self, *args) -> None:
        self._srv.shutdown()
        self._srv.server_close()
        self._th.join(timeout=2)


# =============================================================================
# Explorer 1 Tests: `deepmath.py`, `kl.py`, `bfcl.py`
# =============================================================================


def test_deepmath_difficulty_zero_and_null_handling(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    rows = [
        {"question": "Zero diff problem?", "final_answer": "42", "difficulty": 0.0, "topic": "Mathematics -> Algebra -> Linear"},
        {"question": "Valid diff problem?", "final_answer": "1,234", "difficulty": 4.0, "topic": "Mathematics -> Algebra -> Linear"},
        {"question": "Null diff problem?", "final_answer": "77", "difficulty": None, "topic": "Mathematics -> Geometry -> Circles"},
    ]
    monkeypatch.setitem(sys.modules, "datasets", SimpleNamespace(load_dataset=lambda *a, **kw: rows))
    cache_dir = tmp_path / "dm_cache"
    picked = deepmath._load_or_build_deepmath(cache_dir, max_items=10)
    qs = {r["question"]: r for r in picked}
    assert "Zero diff problem?" not in qs
    assert qs["Valid diff problem?"]["answer"] == 1234
    assert qs["Valid diff problem?"]["difficulty"] == 4.0
    assert qs["Null diff problem?"]["answer"] == 77
    assert qs["Null diff problem?"]["difficulty"] == 5.0

    # Also test JSONL loading with difficulty=0.0, difficulty=null, and comma-formatted answer
    hf_dir = _build_synthetic_hf_dir(tmp_path / "hf", dialect="inkling2")
    jpath = tmp_path / "dm.jsonl"
    jrows = [
        {"id": "z0", "question": "Q0?", "answer": "42", "difficulty": 0.0, "area": "Algebra", "split": "search"},
        {"id": "n1", "question": "Q1?", "answer": "1,234", "difficulty": None, "area": "Geometry", "split": "select"},
    ]
    jpath.write_text("\n".join(json.dumps(r) for r in jrows) + "\n", encoding="utf-8")
    env = deepmath.DeepMathEnvironment(hf_dir=str(hf_dir), jsonl_file=str(jpath))
    by_id = {t.id: t for t in env.tasks()}
    assert by_id["dm:z0"].meta["difficulty"] == 0.0
    assert by_id["dm:n1"].meta["difficulty"] == 5.0
    assert by_id["dm:n1"].meta["answer"] == 1234


def test_deepmath_jsonl_preserves_split_without_family_or_group(tmp_path: Path):
    hf_dir = _build_synthetic_hf_dir(tmp_path / "hf", dialect="inkling2")
    jpath = tmp_path / "dm_splits.jsonl"
    rows = [
        {"id": "a0", "question": "1+1?", "answer": "2", "difficulty": 4.0, "area": "Algebra", "split": "search"},
        {"id": "a1", "question": "1+2?", "answer": "3", "difficulty": 4.0, "area": "Algebra", "split": "select"},
        {"id": "a2", "question": "1+3?", "answer": "4", "difficulty": 4.0, "area": "Algebra", "split": "exam"},
    ]
    jpath.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    env = deepmath.DeepMathEnvironment(hf_dir=str(hf_dir), jsonl_file=str(jpath))
    by_id = {t.id: t for t in env.tasks()}
    assert by_id["dm:a0"].split == "search"
    assert by_id["dm:a1"].split == "select"
    assert by_id["dm:a2"].split == "exam"
    protocol.validate_environment(env)


def test_deepmath_extract_boxed_int_and_bfcl_strip_thinking_edge_cases():
    assert deepmath.extract_boxed_int(r"\boxed{{42}}") == 42
    assert deepmath.extract_boxed_int(r"\boxed{\textbf{42}}") == 42
    assert deepmath.extract_boxed_int(r"\boxed{\mbox{42}}") == 42
    assert deepmath.extract_boxed_int(r"\boxed{\text{\textbf{-1,234}}}") == -1234
    assert deepmath.extract_boxed_int(r"\boxed{1{,}000}") == 1000

    # Unclosed thinking blocks must be stripped so draft boxed numbers don't score
    assert deepmath.extract_boxed_int(r"<think>draft \boxed{99}") is None
    assert deepmath.extract_boxed_int("<|channel>thought\n\\boxed{99}") is None
    assert deepmath.extract_boxed_int(r"<|message_model|><|content_thinking|>\boxed{99}") is None

    # Closed thinking followed by valid boxed answer still extracts cleanly
    assert deepmath.extract_boxed_int(r"<think>draft \boxed{99}</think>\boxed{42}") == 42

    # Unclosed thinking in bfcl.strip_thinking / parse_tool_calls
    assert bfcl.parse_tool_calls('<think>{"name": "f", "arguments": {}}') == []


def test_deepmath_verify_empty_text_premature_stop(tmp_path: Path):
    hf_dir = _build_synthetic_hf_dir(tmp_path / "hf", dialect="inkling2")
    jpath = tmp_path / "dm.jsonl"
    jpath.write_text(
        json.dumps({"id": "q0", "question": "Q", "answer": "42", "difficulty": 4.0, "group": 1, "split": "search", "area": "Algebra"}) + "\n",
        encoding="utf-8",
    )
    env = deepmath.DeepMathEnvironment(hf_dir=str(hf_dir), jsonl_file=str(jpath))
    task = env.tasks()[0]

    for empty_txt in ("", "   \n\t "):
        ep = Episode(task.id, 0, [], None, empty_txt, ["stop"], 1, 0, 0.1)
        v = env.verify(task, ep)
        assert v.detectors["premature_stop"] is True
        assert v.success is False


def test_deepmath_sample_tasks_and_stats_texts_stratified(tmp_path: Path):
    hf_dir = _build_synthetic_hf_dir(tmp_path / "hf", dialect="inkling2")
    jpath = tmp_path / "dm_strat.jsonl"
    rows = []
    for area in ("Algebra", "Geometry", "Number Theory"):
        for i in range(6):
            rows.append({
                "id": f"{area[:3].lower()}_{i:02d}",
                "question": f"{area} question {i}?",
                "answer": str(i + 10),
                "difficulty": 4.0,
                "area": area,
                "split": "search",
            })
    jpath.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    env = deepmath.DeepMathEnvironment(hf_dir=str(hf_dir), jsonl_file=str(jpath))

    strat = env.sample_tasks("search", 6, stratified=True)
    assert len(strat) == 6
    counts = {}
    for t in strat:
        counts[t.meta["area"]] = counts.get(t.meta["area"], 0) + 1
    assert counts == {"Algebra": 2, "Geometry": 2, "Number Theory": 2}

    lex = env.sample_tasks("search", 6, stratified=False)
    assert [t.id for t in lex] == sorted(t.id for t in env.tasks() if t.split == "search")[:6]
    assert env.sample_tasks("search", 0) == []

    st = env.stats_texts(6)
    assert len(st) == 6
    joined = "\n".join(st)
    assert "Algebra question" in joined
    assert "Geometry question" in joined
    assert "Number Theory question" in joined


def test_deepmath_run_preserves_reasoning_and_kl_renderer_parity(tmp_path: Path):
    hf_dir = _build_synthetic_hf_dir(tmp_path / "hf", dialect="inkling2")
    jpath = tmp_path / "dm.jsonl"
    jpath.write_text(
        json.dumps({"id": "q0", "question": "2+3?", "answer": "5", "difficulty": 4.0, "group": 1, "split": "select", "area": "Algebra"}) + "\n",
        encoding="utf-8",
    )
    env = deepmath.DeepMathEnvironment(hf_dir=str(hf_dir), jsonl_file=str(jpath), reasoning_effort="low")
    task = env.tasks()[0]

    wire_reply = "<|message_model|><|content_thinking|>2+3=5<|end_message|><|message_model|><|content_text|>\\boxed{5}<|end_message|>"

    class StubComp:
        def complete(self, prompt, seed, max_tokens, stop=None):
            assert "Thinking effort level: 0.2<|end_message|>" in prompt
            return {"text": wire_reply, "finish_reason": "stop", "completion_tokens": 12}

    ep = env.run(task, StubComp(), seed=7)
    assert ep.messages[-1]["content"] == wire_reply
    assert "<|content_thinking|>2+3=5<|end_message|>" in ep.messages[-1]["content"]
    assert env.verify(task, ep).success is True

    R = kl.Renderer(str(hf_dir), reasoning_effort=env.reasoning_effort)
    pre_str = R._render(ep.messages[:-1], None, True)
    assert "Thinking effort level: 0.2<|end_message|>" in pre_str
    assert pre_str == env.render_prompt(task)
    recs = R.episode(ep.messages, None)
    assert len(recs) == 1
    seq, tg = recs[0]
    target_txt = R.tok.decode([seq[j] for j in tg], skip_special_tokens=False)
    assert "<|content_thinking|>2+3=5<|end_message|>" in target_txt
    assert "\\boxed{5}" in target_txt


def test_kl_renderer_params_typeerror_fallback_and_empty_guards(tmp_path: Path):
    q3_dir = _build_synthetic_hf_dir(tmp_path / "q3", dialect="qwen3_json")
    R_def = kl.Renderer(str(q3_dir))
    R_think = kl.Renderer(str(q3_dir), enable_thinking=True)
    msgs = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "ok"}]
    assert "<think>\n\n</think>" in R_def._render(msgs[:1], None, True)
    assert "<think>\n\n</think>" not in R_think._render(msgs[:1], None, True)

    # Dummy tokenizers raising TypeError on reasoning_effort / enable_thinking
    class NoEffortInkTok:
        eos_token_id = 1
        chat_template = "<|message_model|>"
        def get_vocab(self):
            return {"<|message_model|>": 0, "<|content_model_end_sampling|>": 1}
        def apply_chat_template(self, m, **kw):
            if "reasoning_effort" in kw:
                raise TypeError("unexpected kwarg reasoning_effort")
            return f"ok:{kw.get('enable_thinking')}"

    R_ink_fb = kl.Renderer.__new__(kl.Renderer)
    R_ink_fb.tok = NoEffortInkTok()
    R_ink_fb.dialect = "inkling2"
    R_ink_fb.reasoning_effort = "low"
    R_ink_fb.enable_thinking = True
    R_ink_fb.end_ids = [1]
    R_ink_fb.im_end = 1
    assert R_ink_fb._render([{"role": "user", "content": "hi"}], None, True) == "ok:True"

    class BareTok:
        eos_token_id = 1
        chat_template = ""
        def get_vocab(self):
            return {"<|im_end|>": 1}
        def apply_chat_template(self, m, **kw):
            if "enable_thinking" in kw or "reasoning_effort" in kw:
                raise TypeError("no thinking kwargs")
            return "bare"

    R_bare = kl.Renderer.__new__(kl.Renderer)
    R_bare.tok = BareTok()
    R_bare.dialect = "qwen3_json"
    R_bare.reasoning_effort = "low"
    R_bare.enable_thinking = False
    R_bare.end_ids = [1]
    R_bare.im_end = 1
    assert R_bare._render([{"role": "user", "content": "hi"}], None, True) == "bare"

    # Empty batch / empty write_trj / empty pack_text guards
    with pytest.raises(kl.RenderError):
        R_def.render_trajectories([])
    with pytest.raises(kl.RenderError):
        R_def.render_trajectories([{"messages": [{"role": "user", "content": "only user"}]}])
    with pytest.raises(kl.RenderError):
        kl.write_trj(tmp_path / "empty.trj", [])
    with pytest.raises(kl.RenderError):
        kl.pack_text(R_def, ["hi"], seq_len=64, n_seq=1)


def test_bfcl_stats_texts_stratified_and_pairs_split_preserved(tmp_path: Path):
    hf_dir = _build_synthetic_hf_dir(tmp_path / "hf", dialect="inkling2")
    pairs = tmp_path / "bfcl_pairs.jsonl"
    rows = []
    # Sequential categories in search (4 each) without "family" key, plus select & exam
    for cat in ("simple", "multiple", "sql"):
        for i in range(4):
            rows.append({
                "id": f"{cat}:{cat}_{i:02d}",
                "category": cat,
                "split": "search",
                "messages": [{"role": "user", "content": f"call {cat} {i}"}],
                "tools": [{"type": "function", "function": {"name": "fn", "parameters": {"type": "object", "properties": {}}}}],
                "reference_calls": [{"name": "fn", "arguments": {}}],
            })
    for sp in ("select", "exam"):
        rows.append({
            "id": f"simple:{sp}_0",
            "category": "simple",
            "split": sp,
            "messages": [{"role": "user", "content": f"call {sp}"}],
            "tools": [{"type": "function", "function": {"name": "fn", "parameters": {"type": "object", "properties": {}}}}],
            "reference_calls": [{"name": "fn", "arguments": {}}],
        })
    pairs.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    env = bfcl.BfclEnvironment(hf_dir=str(hf_dir), pairs_file=str(pairs))
    by_id = {t.id: t for t in env.tasks()}
    assert by_id["simple:select_0"].split == "select"
    assert by_id["simple:exam_0"].split == "exam"
    assert by_id["sql:sql_00"].split == "search"
    protocol.validate_environment(env)

    st = env.stats_texts(3)
    assert len(st) == 3
    joined = "\n".join(st)
    assert "call simple" in joined
    assert "call multiple" in joined
    assert "call sql" in joined


# =============================================================================
# Explorer 2 Tests: `codetasks.py`, `sandbox.py`, `detectors.py`
# =============================================================================


def test_codetasks_init_metadata_dialect_and_evaluator_compat(tmp_path: Path):
    env0 = codetasks.CodeTasksEnv(seed=20261012, variants_per_family=1)
    assert env0.dialect is None
    assert env0.needs_special_tokens is None
    assert env0.tools == codetasks.TOOLS
    assert env0.reasoning_effort == "none"

    # Evaluator MUST still fall back to gguf_ops.needs_special_tokens when env.needs_special_tokens is None
    ink_hf = _build_synthetic_hf_dir(tmp_path / "ink_hf", dialect="inkling2")
    ink_tok = AutoTokenizer.from_pretrained(str(ink_hf))
    vocab = ink_tok.get_vocab()
    ink_tokens = [b""] * len(vocab)
    for tok_s, idx in vocab.items():
        ink_tokens[idx] = tok_s.encode("utf-8")
    ink_gguf = _write_minimal_gguf(tmp_path / "ink.gguf", "inkling2", ink_tokens)

    tc = runtime.Toolchain(tmp_path, tmp_path / "tk", [""])
    ev0 = evaluator.Evaluator(tc, env0, {"temperature": 0.0}, 1, 1024, "f16", tmp_path / "logs0")
    assert ev0._resolve_special(ink_gguf) is True

    # With Inkling-2 hf_dir and reasoning_effort="low"
    env_ink = codetasks.CodeTasksEnv(seed=20261012, variants_per_family=1, hf_dir=str(ink_hf), reasoning_effort="low")
    assert env_ink.dialect == "inkling2"
    assert env_ink.needs_special_tokens is True
    assert env_ink.tools == codetasks.TOOLS

    task = env_ink.tasks()[0]

    def responder_ok(path, body):
        return 200, {
            "choices": [{
                "message": {"role": "assistant", "content": f"```python\n{task.meta['reference']}\n```"},
                "finish_reason": "stop",
            }],
            "usage": {"completion_tokens": 5},
        }

    with _StubServer(responder_ok) as srv:
        cl_ink = client_mod.ChatClient(srv.url, {"temperature": 0.0}, dialect=env_ink.dialect)
        ep = env_ink.run(task, cl_ink, seed=1)
        assert env_ink.verify(task, ep).success is True
        assert srv.records[-1]["body"]["chat_template_kwargs"] == {"reasoning_effort": "low"}

    # With Qwen3 hf_dir
    q3_hf = _build_synthetic_hf_dir(tmp_path / "q3_hf", dialect="qwen3_json")
    env_q3 = codetasks.CodeTasksEnv(seed=20261012, variants_per_family=1, hf_dir=str(q3_hf))
    assert env_q3.dialect == "qwen3_json"
    assert env_q3.needs_special_tokens is False
    with _StubServer(responder_ok) as srv:
        cl_q3 = client_mod.ChatClient(srv.url, {"temperature": 0.0}, dialect=env_q3.dialect)
        env_q3.run(env_q3.tasks()[0], cl_q3, seed=1)
        assert srv.records[-1]["body"]["chat_template_kwargs"] == {"enable_thinking": False}


def test_codetasks_sample_tasks_and_stats_texts(tmp_path: Path):
    env0 = codetasks.CodeTasksEnv(seed=20261012, variants_per_family=3)
    strat = env0.sample_tasks("search", 14, stratified=True)
    assert len(strat) == 14
    assert len({t.family for t in strat}) == 14

    lex = env0.sample_tasks("search", 14, stratified=False)
    assert len({t.family for t in lex}) < 14
    assert [t.id for t in lex] == sorted(t.id for t in env0.tasks() if t.split == "search")[:14]
    assert env0.sample_tasks("search", 0) == []

    st0 = env0.stats_texts(14)
    assert len(st0) == 14 and all("```python" in s for s in st0)

    ink_hf = _build_synthetic_hf_dir(tmp_path / "ink_hf", dialect="inkling2")
    env_ink = codetasks.CodeTasksEnv(seed=20261012, variants_per_family=2, hf_dir=str(ink_hf), reasoning_effort="low")
    st_ink = env_ink.stats_texts(14)
    assert len(st_ink) == 14
    assert all("Thinking effort level: 0.2" in s for s in st_ink)


def test_codetasks_run_tool_validation_and_reasoning_content(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    ink_hf = _build_synthetic_hf_dir(tmp_path / "ink_hf", dialect="inkling2")
    env = codetasks.CodeTasksEnv(seed=20261012, variants_per_family=1, hf_dir=str(ink_hf), reasoning_effort="low")
    task = env.tasks()[0]

    # 1. Unknown tool must NOT invoke sandbox.run_tests
    sbx_calls = []
    orig_run_tests = sandbox.run_tests

    def spy_run_tests(*a, **kw):
        sbx_calls.append(a)
        return orig_run_tests(*a, **kw)

    monkeypatch.setattr(sandbox, "run_tests", spy_run_tests)

    turn_bad = {"n": 0}

    class BadToolClient:
        def chat(self, msgs, seed, max_tokens, tools=None, **kw):
            turn_bad["n"] += 1
            if turn_bad["n"] == 1:
                return {
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {"id": "c1", "function": {"name": "evil_tool", "arguments": json.dumps({"code": task.meta["reference"]})}},
                            {"id": "c2", "function": {"name": "run_tests", "arguments": "{not json"}},
                        ],
                    },
                    "finish_reason": "tool_calls",
                    "completion_tokens": 8,
                }
            return {
                "message": {"role": "assistant", "content": f"```python\n{task.meta['reference']}\n```"},
                "finish_reason": "stop",
                "completion_tokens": 6,
            }

    ep_bad = env.run(task, BadToolClient(), seed=0)
    assert len(sbx_calls) == 0
    tool_msgs = [m for m in ep_bad.messages if m.get("role") == "tool"]
    assert len(tool_msgs) == 2
    assert "unknown tool 'evil_tool'" in tool_msgs[0]["content"]
    assert "JSONDecodeError" in tool_msgs[1]["content"]
    assert detectors.malformed_tool_call(ep_bad.messages, codetasks.TOOL_NAMES) is True

    # 2. Valid multi-turn with reasoning_content preserved + kl.Renderer parity
    turn_good = {"n": 0}

    class GoodToolClient:
        def chat(self, msgs, seed, max_tokens, tools=None, **kw):
            turn_good["n"] += 1
            if turn_good["n"] == 1:
                return {
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "reasoning_content": "Let me test candidate first.",
                        "tool_calls": [{"id": "c1", "function": {"name": "run_tests", "arguments": json.dumps({"code": task.meta["reference"]})}}],
                    },
                    "finish_reason": "tool_calls",
                    "completion_tokens": 10,
                }
            assert any(m.get("reasoning_content") == "Let me test candidate first." for m in msgs)
            assert any(m.get("role") == "tool" and m.get("name") == "run_tests" for m in msgs)
            return {
                "message": {
                    "role": "assistant",
                    "reasoning_content": "Tests passed.",
                    "content": f"```python\n{task.meta['reference']}\n```",
                },
                "finish_reason": "stop",
                "completion_tokens": 12,
            }

    ep_good = env.run(task, GoodToolClient(), seed=0)
    assert len(sbx_calls) == 1
    assert ep_good.messages[2]["reasoning_content"] == "Let me test candidate first."
    assert ep_good.messages[3]["name"] == "run_tests"
    assert ep_good.messages[4]["reasoning_content"] == "Tests passed."
    assert env.verify(task, ep_good).success is True

    R = kl.Renderer(str(ink_hf), reasoning_effort="low")
    recs = R.episode(ep_good.messages, env.tools)
    assert len(recs) == 2
    dec0 = R.tok.decode(recs[0][0], skip_special_tokens=False)
    assert "<|content_thinking|>Let me test candidate first.<|end_message|>" in dec0


def test_sandbox_result_json_chdir_crash_and_errors():
    # 1. stdout without trailing newline must pass
    code_print = 'print("x", end="")\ndef f(x):\n    return x + 1\n'
    res_print = sandbox.run_tests(code_print, "f", [[[1], 2], [[5], 6]])
    assert res_print["passed"] is True and res_print["n_ok"] == 2 and res_print["error"] is None

    # 2. os.chdir inside solution must still write result.json to sandbox temp dir
    code_chdir = 'import os\nos.chdir("/tmp")\ndef f(x):\n    return x * 2\n'
    res_chdir = sandbox.run_tests(code_chdir, "f", [[[3], 6]])
    assert res_chdir["passed"] is True and res_chdir["n_ok"] == 1 and res_chdir["error"] is None

    # 3. Missing function reported cleanly
    res_missing = sandbox.run_tests("def g(x): return x\n", "f", [[[1], 1]])
    assert res_missing["passed"] is False and res_missing["error"] == "function f not defined"

    # 4. SyntaxError reported cleanly
    res_syn = sandbox.run_tests("def f(:\n", "f", [[[1], 1]])
    assert res_syn["passed"] is False and res_syn["error"].startswith("SyntaxError:")

    # 5. Subprocess crash via os._exit(1) distinguished from missing result.json and timeout
    res_crash = sandbox.run_tests("import os\nos._exit(1)\n", "f", [[[1], 1]])
    assert res_crash["passed"] is False
    assert res_crash["error"].startswith("harness crashed (rc=1)")


def test_detectors_thinking_stripping_and_multidialect_tags():
    # Closed and unclosed thinking blocks must be stripped before extract_code
    assert detectors.extract_code("<think>\n```python\ndef f(): return 1\n```\n</think>\nNo code") is None
    assert detectors.extract_code("<think>\n```python\ndef f(): return 1\n```") is None
    assert detectors.extract_code("<|channel>thought\n```python\ndef f(): return 1\n```") is None
    assert detectors.extract_code("<|message_model|><|content_thinking|>\n```python\ndef f(): return 1\n```") is None

    # Broken code inside <think>...</think> must not trigger token_garbage SyntaxError when valid code follows
    valid_msg = "<think>\n```python\ndef broken(:\n```\n</think>\n```python\ndef ok():\n    return 1\n```"
    assert detectors.extract_code(valid_msg) == "def ok():\n    return 1\n"
    assert detectors.token_garbage([{"role": "assistant", "content": valid_msg}], valid_msg) is False

    # Thinking-only output triggers premature_stop
    assert detectors.premature_stop("<think>```python\ndef f(): pass\n```", ["stop"]) is True

    # U+FFFD in reasoning_content triggers token_garbage
    assert detectors.token_garbage(
        [{"role": "assistant", "content": "ok", "reasoning_content": "bad\ufffdtok"}],
        "```python\ndef f(): return 1\n```",
    ) is True

    # All 6 unparsed tool-call markers detected
    for tag in (
        "<tool_call>{}",
        "</tool_call>",
        "<|tool_call>call:run_tests{}",
        "<tool_call|>",
        "<function=run_tests>",
        "<|content_invoke_tool_json|>{}",
    ):
        assert detectors.malformed_tool_call([{"role": "assistant", "content": tag}], codetasks.TOOL_NAMES) is True, tag

    # Non-run_tests tools without "code" key must NOT be flagged malformed
    assert detectors.malformed_tool_call(
        [{
            "role": "assistant",
            "content": "",
            "tool_calls": [{"function": {"name": "calc", "arguments": {"x": 1}}}],
        }],
        {"calc"},
    ) is False


# =============================================================================
# Explorer 3 Tests: `families.py` (All 42 Generators)
# =============================================================================


def test_families_all_42_determinism_and_hashseed_invariance():
    assert len(families.FAMILIES) == 42

    state_before = random.getstate()
    env_a = codetasks.CodeTasksEnv(seed=20261012, variants_per_family=5)
    state_after = random.getstate()
    assert state_before == state_after

    script = (
        "import json, sys\n"
        f"sys.path.insert(0, {str(ROOT / 'mvp')!r})\n"
        "from chantless.env import codetasks\n"
        "env = codetasks.CodeTasksEnv(seed=20261012, variants_per_family=5)\n"
        "dump = [{'id': t.id, 'family': t.family, 'split': t.split, 'prompt': t.prompt, 'meta': t.meta} for t in env.tasks()]\n"
        "print(json.dumps(dump, sort_keys=True))\n"
    )
    out1 = subprocess.check_output([sys.executable, "-c", script], env={**os.environ, "PYTHONHASHSEED": "1"}, text=True)
    out2 = subprocess.check_output([sys.executable, "-c", script], env={**os.environ, "PYTHONHASHSEED": "99999"}, text=True)
    assert out1 == out2
    assert json.loads(out1) == [
        {"id": t.id, "family": t.family, "split": t.split, "prompt": t.prompt, "meta": t.meta}
        for t in env_a.tasks()
    ]


def test_families_all_42_variants_json_roundtrip_and_immutability():
    env = codetasks.CodeTasksEnv(seed=20261012, variants_per_family=40)
    tasks = env.tasks()
    assert len(tasks) == 1680

    by_fam: dict[str, list[Task]] = {}
    for t in tasks:
        by_fam.setdefault(t.family, []).append(t)
        assert len(t.meta["visible"]) == 2
        assert len(t.meta["hidden"]) == 10
        ns: dict[str, Any] = {}
        exec(t.meta["reference"], ns)
        fn = ns[t.meta["fname"]]
        for args, exp in t.meta["visible"] + t.meta["hidden"]:
            orig = copy.deepcopy(args)
            got = fn(*args)
            assert args == orig
            if isinstance(got, float):
                assert math.isfinite(got)
            gj = json.loads(json.dumps(got))
            assert gj == exp and type(gj) is type(exp)

    assert len(by_fam) == 42
    assert all(len(ts) == 40 for ts in by_fam.values())


def test_families_all_42_sandbox_self_consistency():
    # Verify all 42 generators import & execute cleanly under python -I -S
    script = (
        "import random, sys\n"
        f"sys.path.insert(0, {str(ROOT / 'mvp')!r})\n"
        "from chantless.env import families\n"
        "assert len(families.FAMILIES) == 42\n"
        "for gen in families.FAMILIES.values():\n"
        "    fname, sig, desc, ref, inputs = gen(random.Random(42))\n"
        "    ns = {}\n"
        "    exec(ref, ns)\n"
        "    for a in inputs:\n"
        "        ns[fname](*a)\n"
    )
    p = subprocess.run([sys.executable, "-I", "-S", "-c", script], capture_output=True, text=True)
    assert p.returncode == 0, p.stderr

    # Verify 1 variant per family (42 tasks x 12 tests = 504 checks) through sandbox.run_tests
    env = codetasks.CodeTasksEnv(seed=20261012, variants_per_family=1)
    tasks = env.tasks()
    assert len(tasks) == 42
    for t in tasks:
        res = sandbox.run_tests(t.meta["reference"], t.meta["fname"], t.meta["visible"] + t.meta["hidden"])
        assert res["passed"] is True, (t.family, res)
        assert res["n_ok"] == 12, (t.family, res)
        assert res["error"] is None, (t.family, res)


# =============================================================================
# Iteration 2 Remediation Tests (Verifiers' Findings 1-6)
# =============================================================================


def test_detectors_and_codetasks_verify_non_dict_tool_calls():
    code_ok = "```python\ndef f():\n    return 1\n```"
    for bad in ("bad_entry", None, 123):
        res = detectors.run_all(
            [{"role": "assistant", "content": "", "tool_calls": [bad]}],
            code_ok,
            ["stop"],
            False,
            codetasks.TOOL_NAMES,
        )
        assert res["malformed_tool_call"] is True, bad
        assert res["doom_loop"] is False, bad

    env = codetasks.CodeTasksEnv(seed=20261012, variants_per_family=1)
    task = env.tasks()[0]

    class NonDictToolClient:
        def __init__(self):
            self.n = 0

        def chat(self, msgs, seed, max_tokens, tools=None, **kw):
            self.n += 1
            if self.n == 1:
                return {
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": ["bad_entry", None, 123],
                    },
                    "finish_reason": "tool_calls",
                    "completion_tokens": 3,
                }
            return {
                "message": {"role": "assistant", "content": "no code"},
                "finish_reason": "stop",
                "completion_tokens": 2,
            }

    ep = env.run(task, NonDictToolClient(), seed=0)
    v = env.verify(task, ep)
    assert v.detectors["malformed_tool_call"] is True
    assert v.detectors["doom_loop"] is False
    assert v.success is False


def test_bfcl_run_records_prepared_messages_and_tools_kl_parity(tmp_path: Path):
    ink_hf = _build_synthetic_hf_dir(tmp_path / "ink_hf", dialect="inkling2")
    pairs = tmp_path / "bfcl_java.jsonl"
    rows = [
        {
            "id": "java:j0",
            "category": "java",
            "split": "select",
            "messages": [{"role": "user", "content": "Decode input."}],
            "tools": [{"type": "function", "function": {"name": "Foo.bar", "description": "Decodes.", "parameters": {"type": "object", "properties": {}, "required": []}}}],
            "reference_calls": [{"name": "Foo.bar", "arguments": {}}],
        },
        {
            "id": "simple:s0",
            "category": "simple",
            "split": "search",
            "messages": [{"role": "user", "content": "Ping."}],
            "tools": [{"type": "function", "function": {"name": "ping", "description": "Ping.", "parameters": {"type": "object", "properties": {}, "required": []}}}],
            "reference_calls": [{"name": "ping", "arguments": {}}],
        },
        {
            "id": "simple:e0",
            "category": "simple",
            "split": "exam",
            "messages": [{"role": "user", "content": "Pong."}],
            "tools": [{"type": "function", "function": {"name": "pong", "description": "Pong.", "parameters": {"type": "object", "properties": {}, "required": []}}}],
            "reference_calls": [{"name": "pong", "arguments": {}}],
        },
    ]
    pairs.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    env = bfcl.BfclEnvironment(hf_dir=str(ink_hf), pairs_file=str(pairs), reasoning_effort="low")
    task = env.tasks()[0]

    class StubComp:
        def complete(self, prompt, seed, max_tokens, stop=None):
            return {
                "text": '<|message_model|><|content_thinking|>think<|end_message|><|message_model|>Foo.bar<|content_invoke_tool_json|>{"name":"Foo.bar","args":{}}<|end_message|>',
                "finish_reason": "stop",
                "completion_tokens": 10,
            }

    class FailingComp:
        def complete(self, prompt, seed, max_tokens, stop=None):
            raise client_mod.InfraError("down")

    ep = env.run(task, StubComp(), seed=0)
    assert ep.messages[0]["role"] == "system"
    assert ep.messages[0]["content"] == bfcl.DEFAULT_INKLING2_TOOL_SYSTEM_PROMPT
    assert "Java 8 SDK syntax" in ep.tools[0]["function"]["description"]

    ep_err = env.run(task, FailingComp(), seed=0)
    assert ep_err.infra_error == "down"
    assert ep_err.messages[0]["role"] == "system"
    assert ep_err.messages[0]["content"] == bfcl.DEFAULT_INKLING2_TOOL_SYSTEM_PROMPT
    assert "Java 8 SDK syntax" in ep_err.tools[0]["function"]["description"]

    R = kl.Renderer(str(ink_hf), reasoning_effort=env.reasoning_effort)
    assert R._render(ep.messages[:-1], ep.tools, True) == env.render_prompt(task)
    recs = R.episode(ep.messages, ep.tools)
    assert len(recs) == 1 and len(recs[0][1]) > 0


def test_codetasks_and_kl_renderer_reasoning_effort_parity(tmp_path: Path):
    # A. Inkling-2 with reasoning_effort=None must NOT emit enable_thinking=False and must match effort 0.9
    ink_hf = _build_synthetic_hf_dir(tmp_path / "ink_hf", dialect="inkling2")
    env_ink = codetasks.CodeTasksEnv(seed=20261012, variants_per_family=1, hf_dir=str(ink_hf), reasoning_effort=None)
    task_ink = env_ink.tasks()[0]

    def responder_ink(path, body):
        return 200, {
            "choices": [{
                "message": {"role": "assistant", "content": f"```python\n{task_ink.meta['reference']}\n```"},
                "finish_reason": "stop",
            }],
            "usage": {"completion_tokens": 5},
        }

    with _StubServer(responder_ink) as srv:
        cl_ink = client_mod.ChatClient(srv.url, {"temperature": 0.0}, dialect=env_ink.dialect)
        ep_ink = env_ink.run(task_ink, cl_ink, seed=1)
        ctk = srv.records[-1]["body"]["chat_template_kwargs"]
        assert "enable_thinking" not in ctk
        rendered_via_server_ctk = env_ink.tok.apply_chat_template(
            ep_ink.messages[:-1],
            tools=ep_ink.tools,
            add_generation_prompt=True,
            tokenize=False,
            **ctk,
        )
        assert "Thinking effort level: 0.9<|end_message|>" in rendered_via_server_ctk
        assert rendered_via_server_ctk == env_ink.render_prompt(task_ink)

    # B. Qwen3 with reasoning_effort="low" must align run(), render_prompt(), and kl.Renderer
    q3_hf = _build_synthetic_hf_dir(tmp_path / "q3_hf", dialect="qwen3_json")
    env_q3 = codetasks.CodeTasksEnv(seed=20261012, variants_per_family=1, hf_dir=str(q3_hf), reasoning_effort="low")
    task_q3 = env_q3.tasks()[0]

    def responder_q3(path, body):
        return 200, {
            "choices": [{
                "message": {"role": "assistant", "content": f"```python\n{task_q3.meta['reference']}\n```"},
                "finish_reason": "stop",
            }],
            "usage": {"completion_tokens": 5},
        }

    with _StubServer(responder_q3) as srv:
        cl_q3 = client_mod.ChatClient(srv.url, {"temperature": 0.0}, dialect=env_q3.dialect)
        ep_q3 = env_q3.run(task_q3, cl_q3, seed=1)
        assert srv.records[-1]["body"]["chat_template_kwargs"] == {"enable_thinking": True}

    prompt_q3 = env_q3.render_prompt(task_q3)
    assert "<think>\n\n</think>" not in prompt_q3
    R_q3 = kl.Renderer(str(q3_hf), reasoning_effort=env_q3.reasoning_effort)
    assert R_q3._render(ep_q3.messages[:-1], ep_q3.tools, True) == prompt_q3

    # DeepMath on Qwen3 with reasoning_effort="low" must also preserve thinking and match kl.Renderer
    dm_p = tmp_path / "dm_q3.jsonl"
    dm_p.write_text(
        json.dumps({"id": "q0", "question": "1+1?", "answer": "2", "difficulty": 4.0, "area": "Algebra", "split": "search"}) + "\n",
        encoding="utf-8",
    )
    dm_env = deepmath.DeepMathEnvironment(hf_dir=str(q3_hf), jsonl_file=str(dm_p), reasoning_effort="low")
    dm_t = dm_env.tasks()[0]
    assert "<think>\n\n</think>" not in dm_env.render_prompt(dm_t)

    class StubQ3Comp:
        def complete(self, prompt, seed, max_tokens, stop=None):
            return {"text": "<think>1+1=2</think>\\boxed{2}", "finish_reason": "stop", "completion_tokens": 5}

    dm_ep = dm_env.run(dm_t, StubQ3Comp(), seed=0)
    assert "<think>1+1=2</think>" in dm_ep.messages[-1]["content"]
    assert R_q3._render(dm_ep.messages[:-1], None, True) == dm_env.render_prompt(dm_t)


def test_deepmath_extract_boxed_int_displaystyle_brace():
    assert deepmath.extract_boxed_int(r"\boxed{\displaystyle{42}}") == 42
    assert deepmath.extract_boxed_int(r"\boxed{\text{\displaystyle{-1,234}}}") == -1234


def test_deepmath_jsonl_integer_id_without_family_or_group(tmp_path: Path):
    hf_dir = _build_synthetic_hf_dir(tmp_path / "hf", dialect="inkling2")
    jpath = tmp_path / "dm_int_ids.jsonl"
    rows = [
        {"id": 1001, "question": "1+1?", "answer": "2", "difficulty": 4.0, "area": "Algebra", "split": "search"},
        {"id": 1002, "question": "2+2?", "answer": "4", "difficulty": 4.0, "area": "Algebra", "split": "select"},
        {"id": 1003, "question": "3+3?", "answer": "6", "difficulty": 4.0, "area": "Algebra", "split": "exam"},
    ]
    jpath.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    env = deepmath.DeepMathEnvironment(hf_dir=str(hf_dir), jsonl_file=str(jpath))
    assert [t.id for t in env.tasks()] == ["dm:1001", "dm:1002", "dm:1003"]
    protocol.validate_environment(env)


def test_m3_robustness_split_sizes_tool_call_id_thinking_strings_and_jsonl_fields(tmp_path: Path):
    # 1. Empty split_sizes={} must raise ValueError, not silently use DEFAULT_SPLIT_SIZES
    with pytest.raises(ValueError):
        codetasks.CodeTasksEnv(seed=20261012, variants_per_family=1, split_sizes={})

    # 2. tool_calls with {"id": None} must produce tool_call_id="" (not None)
    env = codetasks.CodeTasksEnv(seed=20261012, variants_per_family=1)
    task = env.tasks()[0]

    class NullIdClient:
        def __init__(self):
            self.n = 0

        def chat(self, msgs, seed, max_tokens, tools=None, **kw):
            self.n += 1
            if self.n == 1:
                return {
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {"id": None, "function": None},
                            {"id": None, "function": {"name": "evil", "arguments": {}}},
                            {"id": None, "function": {"name": "run_tests", "arguments": json.dumps({"code": task.meta["reference"]})}},
                        ],
                    },
                    "finish_reason": "tool_calls",
                    "completion_tokens": 5,
                }
            return {
                "message": {"role": "assistant", "content": f"```python\n{task.meta['reference']}\n```"},
                "finish_reason": "stop",
                "completion_tokens": 5,
            }

    ep = env.run(task, NullIdClient(), seed=0)
    tool_msgs = [m for m in ep.messages if m.get("role") == "tool"]
    assert len(tool_msgs) == 3
    assert all(m["tool_call_id"] == "" for m in tool_msgs)

    # 3. _thinking_enabled handles numeric strings like "0" and "0.0"
    assert bfcl._thinking_enabled("inkling2", "0") is False
    assert bfcl._thinking_enabled("inkling2", "0.0") is False
    assert bfcl._thinking_enabled("inkling2", "0.2") is True
    assert bfcl._thinking_enabled("qwen3_json", "0") is False
    assert bfcl._thinking_enabled("qwen3_json", "low") is True

    # 4. Whitespace-padded split in deepmath & bfcl JSONL, plus final_answer fallback in deepmath JSONL
    ink_hf = _build_synthetic_hf_dir(tmp_path / "ink_hf", dialect="inkling2")
    dm_p = tmp_path / "dm_ws.jsonl"
    dm_p.write_text(
        "\n".join([
            json.dumps({"id": "a0", "question": "Q0?", "final_answer": "42", "split": " search "}),
            json.dumps({"id": "a1", "question": "Q1?", "answer": "7", "split": " select "}),
            json.dumps({"id": "a2", "question": "Q2?", "answer": "9", "split": " exam "}),
        ]) + "\n",
        encoding="utf-8",
    )
    dm_env = deepmath.DeepMathEnvironment(hf_dir=str(ink_hf), jsonl_file=str(dm_p))
    assert [t.split for t in dm_env.tasks()] == ["search", "select", "exam"]
    assert dm_env.tasks()[0].meta["answer"] == 42

    bf_p = tmp_path / "bf_ws.jsonl"
    bf_p.write_text(
        "\n".join([
            json.dumps({
                "id": f"simple:{i}",
                "category": "simple",
                "split": f" {sp} ",
                "messages": [{"role": "user", "content": "Go."}],
                "tools": [{"type": "function", "function": {"name": "fn", "parameters": {"type": "object", "properties": {}}}}],
                "reference_calls": [{"name": "fn", "arguments": {}}],
            })
            for i, sp in enumerate(("search", "select", "exam"))
        ]) + "\n",
        encoding="utf-8",
    )
    bf_env = bfcl.BfclEnvironment(hf_dir=str(ink_hf), pairs_file=str(bf_p))
    assert [t.split for t in bf_env.tasks()] == ["search", "select", "exam"]
