"""Opaque-box 4-tier E2E contract test suite for Chantless non-BFCL zero-trust audit.

Tiers:
- test_t1_*: Tier 1 Feature Coverage (Category-Partition across F1-F6)
- test_t2_*: Tier 2 Boundary & Corner Cases (BVA & Adversarial Edge Inputs)
- test_t3_*: Tier 3 Cross-Feature Combinations (Pairwise Integration)
- test_t4_*: Tier 4 Real-World Offline E2E Scenarios

All tests are 100% self-contained in `tmp_path`, completely offline, and use zero `.local/` assets.
"""
from __future__ import annotations

import copy
import hashlib
import http.client
import importlib
import inspect
import json
import math
import os
import stat
import struct
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT / "mvp") not in sys.path:
    sys.path.insert(0, str(ROOT / "mvp"))

from chantless import (
    anneal,
    client as client_mod,
    compare,
    config as cfgmod,
    detectors,
    evaluator,
    gguf_ops,
    kl,
    objectives,
    plan,
    runtime,
    sanity,
    serve,
    vocab as vocabmod,
)
from chantless.env import clsbench, codetasks, deepmath, detect, families, protocol, sandbox


# =============================================================================
# Offline Harness Helpers (100% self-contained in tmp_path)
# =============================================================================


class _StubServer:
    """Ephemeral loopback OpenAI-compatible HTTP server with request recording."""

    def __init__(self, responder: Callable[[str, dict[str, Any]], tuple[int, Any]]):
        self.records: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        parent = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path == "/health":
                    data = json.dumps({"status": "ok"}).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                    return
                self.send_response(404)
                self.end_headers()

            def do_POST(self):
                length = int(self.headers.get("Content-Length", "0"))
                raw = self.rfile.read(length)
                body = json.loads(raw.decode("utf-8")) if raw else {}
                with parent._lock:
                    parent.records.append({"path": self.path, "body": body})
                res = responder(self.path, body)
                if res is None:
                    # Simulate abrupt connection drop (triggers http.client.RemoteDisconnected)
                    self.connection.close()
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


def _write_gguf(
    path: Path,
    arch: str,
    tokens: list[bytes],
    ttypes: list[int] | None = None,
    merges: list[bytes] | None = None,
    extra_kvs: list[tuple[str, tuple[int, bytes]]] | None = None,
    n_layer: int = 2,
    n_embd: int = 256,
    n_ff: int = 256,
) -> Path:
    v_size = len(tokens)
    if ttypes is None:
        ttypes = [3 if (t.startswith(b"<") and t.endswith(b">")) else 1 for t in tokens]
    scores = [0.0] * v_size
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
    if merges is not None:
        kvs.append(("tokenizer.ggml.merges", _pack_kv_str_arr(merges)))
    if extra_kvs:
        kvs.extend(extra_kvs)

    rng = np.random.default_rng(7)
    tensors: list[tuple[str, list[int], int, bytes]] = [
        ("token_embd.weight", [n_embd, v_size], 30, _f32_to_bf16_bytes(rng.standard_normal((v_size, n_embd)))),
        ("output_norm.weight", [n_embd], 0, np.ones(n_embd, dtype=np.float32).tobytes()),
    ]
    for il in range(n_layer):
        p = f"blk.{il}."
        sq = _f32_to_bf16_bytes(rng.standard_normal((n_embd, n_embd)))
        sf = _f32_to_bf16_bytes(rng.standard_normal((n_ff, n_embd)))
        sd = _f32_to_bf16_bytes(rng.standard_normal((n_embd, n_ff)))
        nrm = np.ones(n_embd, dtype=np.float32).tobytes()
        tensors.extend([
            (p + "attn_norm.weight", [n_embd], 0, nrm),
            (p + "attn_q.weight", [n_embd, n_embd], 30, sq),
            (p + "attn_k.weight", [n_embd, n_embd], 30, sq),
            (p + "attn_v.weight", [n_embd, n_embd], 30, sq),
            (p + "attn_output.weight", [n_embd, n_embd], 30, sq),
            (p + "post_attention_norm.weight", [n_embd], 0, nrm),
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


_INKLING2_TEMPLATE = (
    "{%- set eff = reasoning_effort | default('high') -%}"
    "{%- set effort_num = '0' if eff == 'none' else ('0.2' if eff == 'low' else '0.9') -%}"
    "{%- if messages and messages[0].role != 'system' -%}"
    "{{- '<|message_system|><|content_text|>Thinking effort level: ' ~ effort_num ~ '<|end_message|>' -}}"
    "{%- endif -%}"
    "{%- for m in messages -%}"
    "{%- if m.role == 'system' -%}"
    "{{- '<|message_system|><|content_text|>' ~ m.content ~ '<|end_message|>' -}}"
    "{%- if eff != 'high' -%}{{- '<|message_system|><|content_text|>Thinking effort level: ' ~ effort_num ~ '<|end_message|>' -}}{%- endif -%}"
    "{%- elif m.role == 'user' -%}"
    "{{- '<|message_user|><|content_text|>' ~ m.content ~ '<|end_message|>' -}}"
    "{%- elif m.role == 'tool' -%}"
    "{{- '<|message_tool|><|content_text|>' ~ m.content ~ '<|end_message|>' -}}"
    "{%- elif m.role == 'assistant' -%}"
    "{%- if m.tool_calls -%}"
    "{%- for tc in m.tool_calls -%}"
    "{{- '<|message_model|>' ~ tc.function.name ~ '<|content_invoke_tool_json|>' ~ (tc.function.arguments | tojson) ~ '<|end_message|>' -}}"
    "{%- endfor -%}"
    "{{- '<|content_model_end_sampling|>' -}}"
    "{%- else -%}"
    "{{- '<|message_model|><|content_text|>' ~ m.content ~ '<|end_message|><|content_model_end_sampling|>' -}}"
    "{%- endif -%}"
    "{%- endif -%}"
    "{%- endfor -%}"
)

_QWEN3_TEMPLATE = (
    "{%- for m in messages -%}"
    "{{- '<|im_start|>' ~ m.role ~ '\n' ~ (m.content or '') ~ '<|im_end|>\n' -}}"
    "{%- endfor -%}"
    "{%- if add_generation_prompt -%}"
    "{%- if enable_thinking is defined and not enable_thinking -%}"
    "{{- '<|im_start|>assistant\n<think>\n\n</think>\n\n' -}}"
    "{%- else -%}"
    "{{- '<|im_start|>assistant\n' -}}"
    "{%- endif -%}"
    "{%- endif -%}"
)

_GEMMA4_TEMPLATE = (
    "{%- for m in messages -%}"
    "{{- '<|turn>' ~ m.role ~ '\n' ~ (m.content or '') ~ '<turn|>\n' -}}"
    "{%- endfor -%}"
    "{%- if add_generation_prompt -%}{{- '<|turn>model\n' -}}{%- endif -%}"
)

_QWEN35_XML_TEMPLATE = (
    "{%- if tools -%}{{- '<tools><function=placeholder></function></tools>\n' -}}{%- endif -%}"
    "{%- for m in messages -%}"
    "{{- '<|im_start|>' ~ m.role ~ '\n' ~ (m.content or '') ~ '<|im_end|>\n' -}}"
    "{%- endfor -%}"
    "{%- if add_generation_prompt -%}{{- '<|im_start|>assistant\n' -}}{%- endif -%}"
)


def _make_hf_dir(
    out_dir: Path,
    dialect: str = "qwen3_json",
    extra_words: list[str] | None = None,
    bpe: bool = False,
) -> tuple[Path, list[bytes], list[bytes]]:
    """Build a synthetic offline HF fast tokenizer (WordLevel or BPE) + matching GGUF token/merge lists."""
    out_dir.mkdir(parents=True, exist_ok=True)
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
            "<|end_message|>",
        ]
        eos_tok = "<|content_model_end_sampling|>"
        ct = _INKLING2_TEMPLATE
    elif dialect == "gemma4":
        special = ["<unk>", "<turn|>", "<|turn>", "<|tool_call>", "<tool_call|>", "<|channel>", "<channel|>"]
        eos_tok = "<turn|>"
        ct = _GEMMA4_TEMPLATE
    elif dialect == "qwen35_xml":
        special = ["<unk>", "<|im_end|>", "<|im_start|>", "<think>", "</think>", "<function=", "</function>"]
        eos_tok = "<|im_end|>"
        ct = _QWEN35_XML_TEMPLATE
    else:
        special = ["<unk>", "<|im_end|>", "<|im_start|>", "<think>", "</think>", "<tool_call>", "</tool_call>"]
        eos_tok = "<|im_end|>"
        ct = _QWEN3_TEMPLATE

    base_words = [
        "negative",
        "positive",
        "human",
        "ai",
        "yes",
        "no",
        "Classify",
        "review",
        "Solve",
        "Thinking",
        "effort",
        "level:",
        "0",
        "0.2",
        "0.9",
        "unused_extra",
        "hello",
        "world",
        "ok",
        "calc",
        "\\boxed{7}",
        "\\boxed{42}",
    ] + list(extra_words or [])

    if bpe:
        # BPE vocab containing printable single chars, full merge chains for answer words, and special tokens
        vocab_list: list[str] = list(special)
        chars = [chr(c) for c in range(32, 127)]
        for ch in chars:
            if ch not in vocab_list:
                vocab_list.append(ch)
        merges_str: list[str] = ["a b", "ab c"]
        for m in merges_str:
            merged = m.replace(" ", "")
            if merged not in vocab_list:
                vocab_list.append(merged)

        # Build sequential left-to-right BPE merges so every answer word encodes as 1 token
        single_tok_words = ["negative", "positive", "human", "ai", "no", "yes"] + list(extra_words or [])
        seen_merges = set(merges_str)
        for w in single_tok_words:
            if len(w) >= 2:
                cur = w[0]
                for ch in w[1:]:
                    pair = f"{cur} {ch}"
                    cur = cur + ch
                    if pair not in seen_merges:
                        seen_merges.add(pair)
                        merges_str.append(pair)
                    if cur not in vocab_list:
                        vocab_list.append(cur)
        for w in base_words:
            if w not in vocab_list:
                vocab_list.append(w)
        vocab_map = {t: i for i, t in enumerate(vocab_list)}
        merges_pairs = [tuple(m.split(" ")) for m in merges_str]
        tk_obj = Tokenizer(models.BPE(vocab=vocab_map, merges=merges_pairs, unk_token="<unk>"))
        tok = PreTrainedTokenizerFast(
            tokenizer_object=tk_obj,
            unk_token="<unk>",
            eos_token=eos_tok,
            additional_special_tokens=[t for t in special if t not in ("<unk>", eos_tok)],
            chat_template=ct,
        )
        tok.save_pretrained(out_dir)
        return out_dir, [t.encode("utf-8") for t in vocab_list], [m.encode("utf-8") for m in merges_str]

    vocab_list = []
    for t in special + base_words:
        if t not in vocab_list:
            vocab_list.append(t)
    vocab_map = {t: i for i, t in enumerate(vocab_list)}
    tk_obj = Tokenizer(models.WordLevel(vocab=vocab_map, unk_token="<unk>"))
    tk_obj.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tok = PreTrainedTokenizerFast(
        tokenizer_object=tk_obj,
        unk_token="<unk>",
        eos_token=eos_tok,
        additional_special_tokens=[t for t in special if t not in ("<unk>", eos_tok)],
        chat_template=ct,
    )
    tok.save_pretrained(out_dir)
    return out_dir, [t.encode("utf-8") for t in vocab_list], []


def _make_logodds_shim(bin_dir: Path, score_cfg_path: Path | None = None) -> Path:
    """Create executable Python `logodds` shim implementing BatchScorer CLI wire protocol."""
    bin_dir.mkdir(parents=True, exist_ok=True)
    p = bin_dir / "logodds"
    cfg_repr = repr(str(score_cfg_path)) if score_cfg_path else "None"
    p.write_text(
        f"""#!{sys.executable}
import hashlib, json, os, struct, sys
sys.path.insert(0, {str(ROOT / "mvp")!r})
from chantless import vocab as vocabmod

args = sys.argv[1:]
model = None
jobs = []
shard_i, shard_n = 0, 1
i = 0
while i < len(args):
    a = args[i]
    if a == "-m":
        model = args[i + 1]; i += 2
    elif a == "--traj":
        trj = args[i + 1]; i += 2
        assert args[i] == "--pair"
        pa, pb = map(int, args[i + 1].split(",")); i += 2
        assert args[i] == "--out"
        out = args[i + 1]; i += 2
        jobs.append((trj, (pa, pb), out))
    elif a == "--shard":
        shard_i, shard_n = map(int, args[i + 1].split("/")); i += 2
    elif a in ("--ubatch", "--n-seq"):
        i += 2
    else:
        i += 1

cfg_p = {cfg_repr}
cfg = json.loads(open(cfg_p).read()) if cfg_p and os.path.exists(cfg_p) else {{}}

# Validate answer token bounds when model is a real GGUF
if model and os.path.exists(model):
    try:
        v = vocabmod.read_tokenizer(model)["vocab_size"]
        for _, (pa, pb), _ in jobs:
            if pa < 0 or pb < 0 or pa >= v or pb >= v:
                sys.stderr.write(f"pair {{pa}},{{pb}} out of vocab bounds {{v}}\\n")
                sys.exit(2)
    except Exception:
        pass

for trj, pair, out_p in jobs:
    with open(trj, "rb") as f:
        head = f.read(8)
        assert head[:4] == b"TRJ1"
        n_seqs = struct.unpack("<I", head[4:])[0]
        seqs = []
        for _ in range(n_seqs):
            nt, ng = struct.unpack("<II", f.read(8))
            toks = struct.unpack(f"<{{nt}}i", f.read(4 * nt))
            tgts = struct.unpack(f"<{{ng}}I", f.read(4 * ng))
            seqs.append((toks, tgts))

    m_key = os.path.basename(model) if model else ""
    m_rules = cfg.get("by_model", {{}}).get(m_key) or cfg.get("by_path", {{}}).get(str(model), {{}})
    pair_rule = m_rules.get(f"{{pair[0]}},{{pair[1]}}")
    with open(out_p, "w") as out_f:
        for ti, (toks, tgts) in enumerate(seqs):
            if ti % shard_n != shard_i:
                continue
            for k in range(len(tgts)):
                if isinstance(m_rules, dict) and "values" in m_rules:
                    val = m_rules["values"][ti % len(m_rules["values"])]
                elif isinstance(pair_rule, list):
                    val = pair_rule[ti % len(pair_rule)]
                elif isinstance(m_rules, dict) and "token_scores" in m_rules:
                    tok_key = ",".join(map(str, toks[:-1]))
                    val = m_rules["token_scores"].get(tok_key, 0.0)
                else:
                    # Deterministic fallback keyed by token content
                    h = int(hashlib.sha256(repr(toks).encode()).hexdigest()[:8], 16)
                    val = ((h % 2000) / 1000.0) - 1.0
                out_f.write(f"{{ti}}\\t{{k}}\\t{{val}}\\n")
""",
        encoding="utf-8",
    )
    p.chmod(p.stat().st_mode | stat.S_IEXEC)
    return p


def _make_tokdocs_shim(bin_dir: Path) -> Path:
    """Create executable Python `tokdocs` shim reading GGUF vocabulary and writing binary token streams."""
    bin_dir.mkdir(parents=True, exist_ok=True)
    p = bin_dir / "tokdocs"
    p.write_text(
        f"""#!{sys.executable}
import struct, sys
sys.path.insert(0, {str(ROOT / "mvp")!r})
from chantless import vocab as vocabmod

gguf_path, in_bin, out_bin, _threads = sys.argv[1:5]
meta = vocabmod.read_tokenizer(gguf_path)
tok_to_id = {{t.decode("utf-8", errors="surrogateescape"): i for i, t in enumerate(meta["tokens"])}}
unk_id = tok_to_id.get("<unk>", 0)

with open(in_bin, "rb") as f:
    n_docs = struct.unpack("<I", f.read(4))[0]
    prompts = []
    for _ in range(n_docs):
        blen = struct.unpack("<I", f.read(4))[0]
        prompts.append(f.read(blen).decode("utf-8"))

# Greedy longest-match over GGUF token strings
sorted_toks = sorted([k for k in tok_to_id if k], key=len, reverse=True)
encoded = []
for s in prompts:
    ids = []
    pos = 0
    while pos < len(s):
        hit = None
        for cand in sorted_toks:
            if s.startswith(cand, pos):
                hit = cand
                break
        if hit is not None:
            ids.append(tok_to_id[hit])
            pos += len(hit)
        else:
            ids.append(unk_id)
            pos += 1
    encoded.append(ids)

with open(out_bin, "wb") as f:
    f.write(struct.pack("<I", len(encoded)))
    for ids in encoded:
        f.write(struct.pack("<I", len(ids)))
        if ids:
            f.write(struct.pack(f"<{{len(ids)}}i", *ids))
""",
        encoding="utf-8",
    )
    p.chmod(p.stat().st_mode | stat.S_IEXEC)
    return p


def _make_clsbench_jsonl(
    path: Path,
    n_cal: int = 12,
    n_search: int = 8,
    n_select: int = 8,
    n_exam: int = 8,
    flagged: str = "negative",
    unflagged: str = "positive",
) -> Path:
    rows = []
    for i in range(n_cal):
        rows.append({
            "id": f"cal-{i}",
            "text": f"{unflagged} review number {i} with clear pleasant quality and good service.",
            "label": unflagged,
            "split": "calibration",
            "family": f"cal/b{i % 2:02d}",
        })
    for sp, n in (("search", n_search), ("select", n_select), ("exam", n_exam)):
        for i in range(n):
            is_flag = (i % 2 == 0)
            lbl = flagged if is_flag else unflagged
            rows.append({
                "id": f"{sp}-{i}",
                "text": f"{lbl} sample document {sp} index {i} for classifier evaluation.",
                "label": lbl,
                "split": sp,
                "family": f"{sp}/b{i % 2:02d}",
            })
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return path


def _make_detect_parquet_dir(data_dir: Path, n_cal: int = 10, n_extra_val: int = 8) -> tuple[Path, Path]:
    """Create minimal WildOutlines + Beemo parquet files in `tmp_path` (>1000/label in val so _loss_curve_rows leaves docs)."""
    wo = data_dir / "WildOutlines" / "data"
    be = data_dir / "beemo" / "data"
    wo.mkdir(parents=True, exist_ok=True)
    be.mkdir(parents=True, exist_ok=True)

    # validation: >1000 per label so _loss_curve_rows (which skips 1000/label) leaves n_extra_val docs
    n_each = 1000 + n_extra_val
    val_rows = []
    for lbl in ("human", "ai"):
        for i in range(n_each):
            val_rows.append({
                "id": f"val-{lbl}-{i}",
                "text": f"Validation {lbl} document {i} text.",
                "label": lbl,
                "format": "essay" if i % 2 == 0 else "news",
            })
    pq.write_table(pa.Table.from_pylist(val_rows), wo / "validation-00000-of-00001.parquet")

    test_rows = []
    for lbl in ("human", "ai"):
        for i in range(6):
            test_rows.append({
                "id": f"test-{lbl}-{i}",
                "text": f"Test {lbl} document {i} text.",
                "label": lbl,
                "format": "essay" if i % 2 == 0 else "news",
            })
    pq.write_table(pa.Table.from_pylist(test_rows), wo / "test-00000-of-00001.parquet")

    cal_rows = [
        {"id": f"cal-{i}", "text": f"Calibration human document {i} text.", "label": "human", "format": "essay"}
        for i in range(n_cal)
    ]
    pq.write_table(pa.Table.from_pylist(cal_rows), wo / "calibration-00000-of-00001.parquet")

    beemo_rows = [
        {
            "id": f"b{i}",
            "prompt_id": f"p{i}",
            "category": "summary",
            "human_output": f"Beemo human output {i}.",
            "model_output": f"Beemo ai output {i}.",
            "human_edits": f"Beemo edited output {i}.",
        }
        for i in range(12)
    ]
    pq.write_table(pa.Table.from_pylist(beemo_rows), be / "train-00000-of-00001.parquet")

    cal_ids_path = data_dir / "cal_ids.json"
    cal_ids_path.write_text(json.dumps([r["id"] for r in cal_rows]), encoding="utf-8")
    return data_dir, cal_ids_path


def _make_deepmath_jsonl(path: Path) -> Path:
    rows = []
    areas = ["Algebra", "Geometry", "Number Theory"]
    splits = ["search", "select", "exam", "train"]
    for i, sp in enumerate(splits):
        for j, area in enumerate(areas):
            rows.append({
                "id": f"{sp}_{j}_{i}",
                "question": f"Compute {j + 2} + {i + 3} in {area}.",
                "answer": str((j + 2) + (i + 3)),
                "difficulty": 4.0 + j,
                "area": area,
                "subtopic": f"{area}Sub",
                "split": sp,
                "family": f"{area.lower().replace(' ', '_')}/{sp}",
            })
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return path


# =============================================================================
# TIER 1: Feature Coverage (`test_t1_*`)
# =============================================================================


def test_t1_classifier_auroc_orientation_and_decide(tmp_path: Path):
    """CLASSIFIER-CONTRACT.md: AUROC = P(score(flagged) < score(unflagged)) == 1.0 on perfect separation."""
    hf_dir, _, _ = _make_hf_dir(tmp_path, dialect="qwen3_json")
    jpath = _make_clsbench_jsonl(tmp_path / "cls.jsonl", n_cal=4, n_search=4, n_select=4, n_exam=4)
    env = clsbench.PublicClassifierEnvironment(hf_dir=str(hf_dir), cache_dir=str(tmp_path / "cache"), jsonl_file=str(jpath))

    # pair = (unflagged_id, flagged_id); score = logit[unflagged] - logit[flagged]; lower = flagged
    assert env.pair == (env.tok.encode("positive", add_special_tokens=False)[0], env.tok.encode("negative", add_special_tokens=False)[0])
    rule = {"cutoff": 0.0, "fpr_target": 0.05, "n_cal": 4}
    neg_task = next(t for t in env.tasks() if t.split == "exam" and t.meta["label"] == "negative")
    pos_task = next(t for t in env.tasks() if t.split == "exam" and t.meta["label"] == "positive")
    assert env.decide(neg_task, -1.5, rule).success is True
    assert env.decide(pos_task, 1.5, rule).success is True
    assert env.decide(pos_task, -0.1, rule).success is False  # strictly below cutoff flags

    # anneal.auc(s, y): y=True is flagged (AI), s higher = unflagged (human)
    s_perfect = np.array([10.0, 8.0, -5.0, -9.0])
    y = np.array([False, False, True, True])
    assert anneal.auc(s_perfect, y) == pytest.approx(1.0)
    assert anneal.auc(-s_perfect, y) == pytest.approx(0.0)

    # compare.compare flags score < cutoff
    res = compare.compare(
        models={"m1": {"cal": [0.0, 1.0, 2.0, 3.0], "scores": {"a": -2.0, "b": -1.0, "c": 1.5, "d": 2.5}}},
        labels={"a": "pos", "b": "pos", "c": "neg", "d": "neg"},
        kinds={"a": "k", "b": "k", "c": "k", "d": "k"},
        boot=20,
        seed=0,
    )
    assert res["models"]["m1"]["pos"]["caught"] == pytest.approx(1.0)
    assert res["models"]["m1"]["neg_flagged"] == pytest.approx(0.0)


def test_t1_evaluator_rule_cache_isolation_and_zero_trust_default(tmp_path: Path):
    """Evaluator.cache_prompt defaults False; rule() cache key isolates fpr_target, answers, decoding, dialect, special."""
    hf_dir, tokens, _ = _make_hf_dir(tmp_path, dialect="qwen3_json")
    gguf_path = _write_gguf(tmp_path / "m.gguf", "qwen3", tokens)
    jpath = _make_clsbench_jsonl(tmp_path / "cls.jsonl", n_cal=10, n_search=4, n_select=4, n_exam=4)

    score_calls = 0

    class CountingEnv(clsbench.PublicClassifierEnvironment):
        def score(self, tasks, model, scorer):
            nonlocal score_calls
            score_calls += 1
            return {t.id: float(i) for i, t in enumerate(tasks)}

    env1 = CountingEnv(hf_dir=str(hf_dir), cache_dir=str(tmp_path / "c1"), jsonl_file=str(jpath), fpr_target=0.1)
    tc = runtime.Toolchain(llama_bin=tmp_path, traj_kl=tmp_path / "traj-kl", gpus=[""], logodds=_make_logodds_shim(tmp_path / "bin"))
    scorer = runtime.BatchScorer(tc, slots_per_gpu=1)

    ev1 = evaluator.Evaluator(tc, env1, {"temperature": 0.0}, 1, 512, "f16", tmp_path / "logs", scorer=scorer)
    assert ev1.cache_prompt is False  # Zero-trust default

    r1 = ev1.rule(str(gguf_path))
    assert score_calls == 1
    # Memory cache hit
    assert ev1.rule(str(gguf_path)) == r1
    assert score_calls == 1

    # New Evaluator with same env/settings hits disk cache
    ev1_disk = evaluator.Evaluator(tc, env1, {"temperature": 0.0}, 1, 512, "f16", tmp_path / "logs", scorer=scorer)
    assert ev1_disk.rule(str(gguf_path)) == r1
    assert score_calls == 1

    # Changing fpr_target, decoding, dialect, special, or answer pair MUST NOT reuse stale disk rule
    env_fpr = CountingEnv(hf_dir=str(hf_dir), cache_dir=str(tmp_path / "c2"), jsonl_file=str(jpath), fpr_target=0.5)
    ev_fpr = evaluator.Evaluator(tc, env_fpr, {"temperature": 0.0}, 1, 512, "f16", tmp_path / "logs", scorer=scorer)
    r_fpr = ev_fpr.rule(str(gguf_path))
    assert score_calls == 2
    assert r_fpr["cutoff"] != r1["cutoff"]

    jpath_yesno = _make_clsbench_jsonl(
        tmp_path / "cls_yesno.jsonl", n_cal=10, n_search=4, n_select=4, n_exam=4, flagged="no", unflagged="yes",
    )
    env_pair = CountingEnv(
        hf_dir=str(hf_dir), cache_dir=str(tmp_path / "c3"), jsonl_file=str(jpath_yesno),
        answers=("no", "yes"), fpr_target=0.1,
    )
    ev_pair = evaluator.Evaluator(tc, env_pair, {"temperature": 0.0}, 1, 512, "f16", tmp_path / "logs", scorer=scorer)
    ev_pair.rule(str(gguf_path))
    assert score_calls == 3

    ev_dec = evaluator.Evaluator(tc, env1, {"temperature": 0.7}, 1, 512, "f16", tmp_path / "logs", scorer=scorer)
    ev_dec.rule(str(gguf_path))
    assert score_calls == 4

    ev_dial = evaluator.Evaluator(tc, env1, {"temperature": 0.0}, 1, 512, "f16", tmp_path / "logs", scorer=scorer, dialect="gemma4")
    ev_dial.rule(str(gguf_path))
    assert score_calls == 5

    ev_spec = evaluator.Evaluator(tc, env1, {"temperature": 0.0}, 1, 512, "f16", tmp_path / "logs", scorer=scorer, special=True)
    ev_spec.rule(str(gguf_path))
    assert score_calls == 6


def test_t1_objectives_pauc_damage_dynamic_hard_negatives():
    """objectives.pauc_damage dynamically selects lowest ceil(tail * len(neg)) of candidate's OWN negative scores."""
    src = {
        "pos": np.array([-2.0, -1.5]),
        "neg": np.array([0.1, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0, 10.5, 11.0]),
    }
    # Index 0 is src's hardest negative (0.1). In cand, index 0 moves to 11.0 while index 5 drops to -10.0!
    cand_neg = src["neg"].copy()
    cand_neg[0] = 11.0
    cand_neg[5] = -10.0
    cand = {"pos": src["pos"].copy(), "neg": cand_neg}

    # Dynamic hard-negative selection picks -10.0 (cand's own lowest 5%), giving much worse pauc(cand) -> positive damage
    dmg = objectives.pauc_damage(cand, src, tail=0.05, tau=0.5)
    assert dmg > 0.5

    # Better-separated candidate gives negative damage
    better = {"pos": np.array([-5.0, -5.0]), "neg": src["neg"] + 2.0}
    assert objectives.pauc_damage(better, src, tail=0.05, tau=0.5) < 0.0


def test_t1_clsbench_hooks_token_counts_proxy_stats_calibrate_decide(tmp_path: Path):
    """PublicClassifierEnvironment exposes token_counts, fit_token_counts, proxy_spec, stats_texts, calibrate, decide."""
    hf_dir, tokens, _ = _make_hf_dir(tmp_path, dialect="qwen3_json")
    gguf_path = _write_gguf(tmp_path / "m.gguf", "qwen3", tokens)
    jpath = _make_clsbench_jsonl(tmp_path / "cls.jsonl", n_cal=10, n_search=6, n_select=4, n_exam=4)
    tc = runtime.Toolchain(
        llama_bin=tmp_path,
        traj_kl=tmp_path / "traj-kl",
        gpus=[""],
        tokdocs=_make_tokdocs_shim(tmp_path / "bin"),
        logodds=_make_logodds_shim(tmp_path / "bin"),
    )

    env = clsbench.PublicClassifierEnvironment(hf_dir=str(hf_dir), cache_dir=str(tmp_path / "cache"), jsonl_file=str(jpath))
    assert hasattr(env, "token_counts") and callable(env.token_counts)
    tc_dict = env.token_counts()
    assert set(tc_dict) >= {"calibration", "search", "select", "exam"}
    for s in ("calibration", "search", "select", "exam"):
        d = tc_dict[s]
        assert isinstance(d, dict)
        assert ("docs" in d or "n" in d) and d.get("docs", d.get("n", 0)) > 0
        assert ("tokens" in d or "total" in d or "mean" in d) and d.get("tokens", d.get("total", d.get("mean", 0))) > 0

    counts = env.fit_token_counts(str(gguf_path), tc)
    assert counts.shape == (len(tokens),) and counts.dtype == np.int64 and counts.sum() > 0

    st = env.stats_texts(4)
    assert len(st) == 4 and all(isinstance(x, str) for x in st)

    # Only calibration tasks should affect calibrate(), even if search/exam scores are passed
    scores = {t.id: (10.0 if t.split == "calibration" else -999.0) for t in env.tasks()}
    rule = env.calibrate(scores)
    assert rule["cutoff"] == pytest.approx(10.0)
    assert rule["n_cal"] == 10

    spec = env.proxy_spec(scores, n_neg=5, n_pos=2)
    assert [p["role"] for p in spec] == ["neg", "pos"]
    assert spec[0]["n"] == 5 and spec[1]["n"] == 2

    ex_neg = next(t for t in env.tasks() if t.split == "exam" and t.meta["label"] == "negative")
    assert env.decide(ex_neg, -5.0, rule).success is True


def test_t1_detect_classifier_and_generative_and_inkling2_dialect(tmp_path: Path):
    """DetectEnvironment supports classifier=True, classifier=False, and inkling2 dialect."""
    hf_dir, tokens, _ = _make_hf_dir(tmp_path, dialect="inkling2")
    gguf_path = _write_gguf(tmp_path / "ink.gguf", "inkling2", tokens)
    data_dir, cal_ids = _make_detect_parquet_dir(tmp_path / "data", n_cal=10)
    tc = runtime.Toolchain(
        llama_bin=tmp_path,
        traj_kl=tmp_path / "traj-kl",
        gpus=[""],
        tokdocs=_make_tokdocs_shim(tmp_path / "bin"),
        logodds=_make_logodds_shim(tmp_path / "bin"),
    )

    # Classifier mode with auto-detected inkling2 dialect
    env_cls = detect.DetectEnvironment(
        data_dir=str(data_dir),
        hf_dir=str(hf_dir),
        classifier=True,
        calibration_ids_file=str(cal_ids),
        cache_dir=str(tmp_path / "det_cache"),
        search_per_cell=2,
        select_per_cell=2,
        exam_test_per_cell=2,
        search_beemo_per_kind=2,
        select_beemo_per_kind=2,
        exam_beemo_per_kind=2,
        prompt_style="auto",
    )
    rendered = env_cls.render_prompt("hello")
    assert "<|message_model|><|content_text|>" in rendered
    assert env_cls.prompt_style == "inkling2"

    scorer = runtime.BatchScorer(tc, slots_per_gpu=1)
    cal_tasks = [t for t in env_cls.tasks() if t.split == "calibration"]
    cal_scores = env_cls.score(cal_tasks, str(gguf_path), scorer)
    assert len(cal_scores) == len(cal_tasks)
    rule = env_cls.calibrate(cal_scores)
    assert "cutoff" in rule and rule["n"] == len(cal_tasks)
    assert env_cls.token_counts()["calibration"]["docs"] == len(cal_tasks)

    all_scores = {t.id: (1.0 if t.meta["label"] == "human" else -1.0) for t in env_cls.tasks()}
    pspec = env_cls.proxy_spec(all_scores, n_neg=4, n_pos=2)
    assert pspec[0]["n"] == 4 and pspec[1]["n"] == 2
    ai_task = next(t for t in env_cls.tasks() if t.meta["label"] == "ai")
    assert env_cls.decide(ai_task, -2.0, {"cutoff": 0.0}).success is True

    # Non-classifier generative mode via loopback /completion server
    env_gen = detect.DetectEnvironment(
        data_dir=str(data_dir),
        hf_dir=str(hf_dir),
        cutoff=0.5,
        classifier=False,
        search_per_cell=2,
        select_per_cell=2,
        exam_test_per_cell=2,
        exam_beemo_per_kind=2,
        prompt_style="inkling2",
    )

    def responder(path, body):
        assert path == "/completion"
        assert body["cache_prompt"] is False
        return 200, {"completion_probabilities": [{"top_probs": [{"token": "human", "prob": 0.1}, {"token": "ai", "prob": 0.9}]}]}

    with _StubServer(responder) as srv:
        cl = client_mod.ChatClient(srv.url, {"temperature": 0.0}, timeout=5.0, retries=2)
        t_ai = next(t for t in env_gen.tasks() if t.meta["label"] == "ai")
        ep = env_gen.run(t_ai, cl, seed=7)
        v = env_gen.verify(t_ai, ep)
        assert ep.infra_error is None and v.tests_passed is True


def test_t1_protocol_validate_environment_classifier_and_generative():
    """protocol.validate_environment enforces classifier=True calibration split + pair + generative invariants."""
    base_tasks = [
        protocol.Task("s0", "f_s", "search", {}),
        protocol.Task("l0", "f_l", "select", {}),
        protocol.Task("e0", "f_e", "exam", {}),
    ]

    class GoodGen:
        name = "good_gen"
        version = "1"
        budget = {"max_turns": 1, "max_tokens_per_turn": 16}
        deployment = {"context": 512, "concurrency": 1, "kv_type": "f16"}
        def tasks(self): return list(base_tasks)
        def run(self, task, client, seed): ...
        def verify(self, task, ep): ...

    protocol.validate_environment(GoodGen())

    class BadClsNoCal(GoodGen):
        classifier = True
        pair = (1, 2)
        def score(self, tasks, model, scorer): return {}
        def calibrate(self, scores): return {}
        def decide(self, task, score, rule): ...

    with pytest.raises(ValueError, match="calibration"):
        protocol.validate_environment(BadClsNoCal())

    class BadClsNoPair(GoodGen):
        classifier = True
        def score(self, tasks, model, scorer): return {}
        def calibrate(self, scores): return {}
        def decide(self, task, score, rule): ...
        def tasks(self): return base_tasks + [protocol.Task("c0", "f_c", "calibration", {})]

    with pytest.raises(ValueError, match="pair"):
        protocol.validate_environment(BadClsNoPair())

    class GoodCls(BadClsNoPair):
        pair = (1, 2)

    protocol.validate_environment(GoodCls())


def test_t1_deepmath_parsing_sampling_split_prompt_verify(tmp_path: Path):
    """DeepMathEnvironment preserves JSONL split, supports stratified sample_tasks, render_prompt, verify."""
    hf_dir, _, _ = _make_hf_dir(tmp_path, dialect="inkling2")
    jpath = tmp_path / "dm.jsonl"
    rows = [
        {"id": "a0", "question": "1+1?", "answer": "2", "difficulty": 4.0, "area": "Algebra", "split": "search"},
        {"id": "a1", "question": "1+2?", "answer": "3", "difficulty": 4.0, "area": "Algebra", "split": "select"},
        {"id": "a2", "question": "1+3?", "answer": "4", "difficulty": 4.0, "area": "Algebra", "split": "select"},
        {"id": "g0", "question": "2+2?", "answer": "4", "difficulty": 5.0, "area": "Geometry", "split": "select"},
        {"id": "n0", "question": "3+3?", "answer": "6", "difficulty": 6.0, "area": "Number Theory", "split": "select"},
        {"id": "e0", "question": "4+4?", "answer": "8", "difficulty": 5.0, "area": "Geometry", "split": "exam"},
    ]
    jpath.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    assert deepmath.extract_boxed_int(r"The answer is \boxed{-1,234}.") == -1234
    assert deepmath.extract_boxed_int(r"\boxed{x = 42}") == 42

    env = deepmath.DeepMathEnvironment(hf_dir=str(hf_dir), jsonl_file=str(jpath), reasoning_effort="low")
    by_id = {t.id: t for t in env.tasks()}
    # Even without "family" or "group" keys, r["split"] MUST be preserved
    assert by_id["dm:a0"].split == "search"
    assert by_id["dm:e0"].split == "exam"

    assert hasattr(env, "sample_tasks") and callable(env.sample_tasks)
    sampled = env.sample_tasks("select", 3, stratified=True)
    assert len(sampled) == 3
    assert {t.meta["area"] for t in sampled} == {"Algebra", "Geometry", "Number Theory"}

    prompt = env.render_prompt(by_id["dm:a0"])
    assert "Thinking effort level: 0.2<|end_message|>" in prompt

    ep_ok = protocol.Episode("dm:a0", 0, [], None, r"<|content_thinking|>step<|end_message|>\boxed{2}", ["stop"], 1, 5, 0.1)
    assert env.verify(by_id["dm:a0"], ep_ok).success is True
    ep_bad = protocol.Episode("dm:a0", 0, [], None, r"\boxed{99}", ["stop"], 1, 5, 0.1)
    assert env.verify(by_id["dm:a0"], ep_bad).success is False


def test_t1_codetasks_families_sandbox_detectors():
    """42 deterministic families, deterministic CodeTasksEnv, sandbox.run_tests, fn.name==run_tests enforcement."""
    assert len(families.FAMILIES) == 42
    env1 = codetasks.CodeTasksEnv(seed=123, variants_per_family=2)
    env2 = codetasks.CodeTasksEnv(seed=123, variants_per_family=2)
    assert [t.id for t in env1.tasks()] == [t.id for t in env2.tasks()]
    assert len(env1.tasks()) == 84

    t = env1.tasks()[0]
    res_ok = sandbox.run_tests(t.meta["reference"], t.meta["fname"], t.meta["visible"])
    assert res_ok["passed"] is True and res_ok["n_ok"] == len(t.meta["visible"])

    # CodeTasksEnv.run MUST reject unknown tool names instead of running sandbox.run_tests
    calls = {"n": 0}

    class FakeClient:
        def chat(self, msgs, seed, max_tokens, tools=None, **kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                return {
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [{"id": "c1", "function": {"name": "evil_tool", "arguments": json.dumps({"code": t.meta["reference"]})}}],
                    },
                    "finish_reason": "tool_calls",
                    "completion_tokens": 5,
                }
            return {
                "message": {"role": "assistant", "content": f"```python\n{t.meta['reference']}\n```"},
                "finish_reason": "stop",
                "completion_tokens": 5,
            }

    ep = env1.run(t, FakeClient(), seed=0)
    tool_msgs = [m for m in ep.messages if m.get("role") == "tool"]
    assert len(tool_msgs) == 1
    assert "examples passed" not in tool_msgs[0]["content"]
    assert "error" in tool_msgs[0]["content"].lower() or "unknown" in tool_msgs[0]["content"].lower()
    assert detectors.malformed_tool_call(ep.messages, codetasks.TOOL_NAMES) is True


def test_t1_configs_serving_defaults_client_precedence_and_retries(tmp_path: Path):
    """Zero-trust ServingConfig/LlamaServer/serve_live defaults, ChatClient precedence/non-leak/retries."""
    assert cfgmod.ServingConfig().cache_prompt is False
    assert inspect.signature(runtime.LlamaServer.__init__).parameters["cache_prompt"].default is False
    assert inspect.signature(serve.serve_live).parameters["cache_prompt"].default is False

    # Validate all 6 shipped config files in mvp/configs/
    expected_kinds = {
        "extract-example.yaml": "extract",
        "inkling2-mini-bfcl.yaml": "extract",
        "inkling2-mini-clsbench.yaml": "extract",
        "inkling2-mini-deepmath.yaml": "extract",
        "learn-example.yaml": "learn",
        "local-smoke.json": "extract",
    }
    for fname, kind in expected_kinds.items():
        loaded = cfgmod.load_config(ROOT / "mvp" / "configs" / fname, kind)
        if kind == "extract":
            assert loaded.serving.cache_prompt is False, fname
        else:
            assert loaded.learn.judge_cache_prompt is False, fname

    # ChatClient explicit seed/max_tokens precedence, no top-level reasoning_effort leak, retries on HTTPException & empty choices
    attempts = {"chat": 0, "comp": 0}

    def responder(path, body):
        if path == "/v1/chat/completions":
            attempts["chat"] += 1
            if attempts["chat"] == 1:
                return None  # triggers http.client.RemoteDisconnected (HTTPException)
            if attempts["chat"] == 2:
                return 200, {"choices": []}  # triggers IndexError
            return 200, {"choices": [{"message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}], "usage": {"completion_tokens": 1}}
        attempts["comp"] += 1
        if attempts["comp"] == 1:
            return 200, {"choices": []}
        return 200, {"choices": [{"text": "ok", "finish_reason": "stop"}], "usage": {"completion_tokens": 1}}

    with _StubServer(responder) as srv:
        cl = client_mod.ChatClient(
            srv.url,
            decoding={"temperature": 0.0, "seed": 999, "max_tokens": 1, "reasoning_effort": "high", "cache_prompt": False},
            retries=4,
            dialect="inkling2",
        )
        r_chat = cl.chat([{"role": "user", "content": "hi"}], seed=42, max_tokens=128)
        assert r_chat["message"]["content"] == "ok"
        chat_body = srv.records[-1]["body"]
        assert chat_body["seed"] == 42
        assert chat_body["max_tokens"] == 128
        assert "reasoning_effort" not in chat_body
        assert chat_body["cache_prompt"] is False

        r_comp = cl.complete("prompt", seed=77, max_tokens=256)
        assert r_comp["text"] == "ok"
        comp_body = srv.records[-1]["body"]
        assert comp_body["seed"] == 77
        assert comp_body["max_tokens"] == 256
        assert "reasoning_effort" not in comp_body
        assert comp_body["cache_prompt"] is False


def test_t1_cli_flags_and_schema_anneal(tmp_path: Path):
    """cli.py supports --no-special, --no-cache-prompt, --flash-attn, schema anneal; export_json_schema('all') includes anneal."""
    all_schemas = cfgmod.export_json_schema("all")
    assert "anneal" in all_schemas

    env = {**os.environ, "PYTHONPATH": str(ROOT / "mvp")}
    p_help = subprocess.run(
        [sys.executable, "-m", "chantless.cli", "serve", "--help"],
        capture_output=True, text=True, env=env,
    )
    assert p_help.returncode == 0
    for flag in ("--no-special", "--no-cache-prompt", "--flash-attn"):
        assert flag in p_help.stdout

    p_schema = subprocess.run(
        [sys.executable, "-m", "chantless.cli", "schema", "anneal"],
        capture_output=True, text=True, env=env,
    )
    assert p_schema.returncode == 0, p_schema.stderr
    parsed = json.loads(p_schema.stdout)
    assert "properties" in parsed and "anneal" in parsed["properties"]


def test_t1_kl_renderer_reasoning_effort_and_empty_batch(tmp_path: Path):
    """kl.Renderer accepts reasoning_effort/enable_thinking and raises on empty trajectory batches."""
    hf_ink, _, _ = _make_hf_dir(tmp_path / "ink", dialect="inkling2")
    R_ink = kl.Renderer(str(hf_ink), reasoning_effort="low")
    msgs = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "ok"}]
    pre = R_ink._render(msgs[:1], None, True)
    assert "Thinking effort level: 0.2<|end_message|>" in pre
    assert R_ink._render(msgs, None, False).startswith(pre)

    # Verify valid episode (without system message) renders cleanly and produces non-empty TRJ1 record
    recs = R_ink.episode(msgs, None)
    assert len(recs) == 1 and len(recs[0][1]) > 0
    meta = kl.write_trj(tmp_path / "one.bin", recs)
    assert meta["n_seq"] == 1 and meta["n_targets"] > 0

    hf_q3, _, _ = _make_hf_dir(tmp_path / "q3", dialect="qwen3_json")
    R_q3 = kl.Renderer(str(hf_q3), enable_thinking=True)
    pre_q3 = R_q3._render(msgs[:1], None, True)
    assert "<think>\n\n</think>" not in pre_q3

    with pytest.raises((ValueError, kl.RenderError)):
        R_ink.episode([], None)
    with pytest.raises((ValueError, kl.RenderError)):
        R_ink.episode([{"role": "user", "content": "hi"}], None)

    if hasattr(R_ink, "render_trajectories"):
        sig = inspect.signature(R_ink.render_trajectories)
        extra = (None,) if len([p for p in sig.parameters.values() if p.default is inspect._empty]) >= 2 else ()
        # PROJECT.md: render_trajectories raises ValueError / RenderError when input is non-empty and 0 trajectories render cleanly
        try:
            with pytest.raises((ValueError, kl.RenderError)):
                R_ink.render_trajectories([{"messages": [{"role": "user", "content": "hi"}]}], *extra)
        except TypeError:
            with pytest.raises((ValueError, kl.RenderError)):
                R_ink.render_trajectories([[{"role": "user", "content": "hi"}]], *extra)


def test_t1_deepmath_select_cli_dynamic_bands_and_select_split(tmp_path: Path):
    """deepmath_select.py handles != 3 difficulty bands dynamically and supports 'select' split via subprocess."""
    # Provide lightweight pure-numpy sklearn.cluster.KMeans stub so tool runs 100% offline without scikit-learn
    stub_pkg = tmp_path / "py_stubs" / "sklearn"
    stub_pkg.mkdir(parents=True)
    (stub_pkg / "__init__.py").write_text("", encoding="utf-8")
    (stub_pkg / "cluster.py").write_text(
        """import numpy as np
class KMeans:
    def __init__(self, n_clusters=8, n_init=1, max_iter=100, random_state=0):
        self.n_clusters = int(n_clusters)
    def fit(self, X):
        X = np.asarray(X, dtype=float)
        n = len(X)
        self.labels_ = np.arange(n) % self.n_clusters
        self.cluster_centers_ = np.array([X[self.labels_ == c].mean(0) for c in range(self.n_clusters)])
        self.inertia_ = float(((X - self.cluster_centers_[self.labels_]) ** 2).sum())
        return self
""",
        encoding="utf-8",
    )

    # 30 rows across 2 areas (15 each) and ONLY 2 difficulty bands (3.5 and 5.5 -> never 7+)
    # K=30 -> K // 10 = 3 coarse groups (10 rows each).
    rows = []
    for i in range(30):
        area = "Algebra" if i < 15 else "Geometry"
        diff = 3.5 if i % 2 == 0 else 5.5
        rows.append({
            "question": f"Solve problem {i} in {area}",
            "final_answer": str(i + 10),
            "difficulty": diff,
            "topic": f"Mathematics -> {area} -> Sub",
            "shard": 0,
            "row": i,
        })
    qa_p = tmp_path / "qa.parquet"
    pq.write_table(pa.Table.from_pylist(rows), qa_p)

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    rng = np.random.default_rng(0)
    emb = rng.standard_normal((30, 8)).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    np.save(out_dir / "pool-embeddings.npy", emb)

    env = {**os.environ, "PYTHONPATH": f"{stub_pkg.parent}:{ROOT / 'mvp'}"}
    # K=30, N_EXAM=16: with hardcoded len(cells)-3 (=1) on 2 areas, area0 total is 15 < 16 so pre-fix crashes;
    # post-fix (n_areas=2), 2 of 3 groups (20 items) go to exam and 1 group (10 items) stays in search.
    proc = subprocess.run(
        [sys.executable, str(ROOT / "mvp/tools/deepmath_select.py"), str(qa_p), str(tmp_path / "model"), str(out_dir), "30", "16"],
        capture_output=True, text=True, env=env,
    )
    assert proc.returncode == 0, f"deepmath_select failed with 2 bands:\n{proc.stderr}"
    report = json.loads((out_dir / "report.json").read_text())
    assert report["split"]["n_exam"] >= 16
    assert report["split"]["n_search"] >= 10

    # Check "select" split support (K=30, N_EXAM=10, N_SELECT=10 -> 1 coarse group per split: search, select, exam all non-empty)
    has_select = (out_dir / "deepmath-int-rep3k-select.jsonl").exists() and report["split"].get("n_select", 0) > 0
    if not has_select:
        out2 = tmp_path / "out2"
        out2.mkdir()
        np.save(out2 / "pool-embeddings.npy", emb)
        for extra_args in (["30", "10", "10"], ["30", "10", "--n-select", "10"]):
            p2 = subprocess.run(
                [sys.executable, str(ROOT / "mvp/tools/deepmath_select.py"), str(qa_p), str(tmp_path / "model"), str(out2), *extra_args],
                capture_output=True, text=True, env=env,
            )
            if p2.returncode == 0 and (out2 / "deepmath-int-rep3k-select.jsonl").exists():
                r2 = json.loads((out2 / "report.json").read_text())
                if r2["split"]["n_search"] > 0 and r2["split"].get("n_select", 0) > 0 and r2["split"]["n_exam"] > 0:
                    has_select = True
                    break
    assert has_select, "deepmath_select.py does not produce or support a non-empty 'select' split"


# =============================================================================
# TIER 2: Boundary & Corner Cases (`test_t2_*`)
# =============================================================================


@pytest.mark.parametrize("bad_val", [float("nan"), float("inf"), float("-inf")])
def test_t2_nan_inf_rejection_across_scoring_and_calibration(tmp_path: Path, bad_val: float):
    """NaN and +/-Inf must raise immediately across BatchScorer, Evaluator.seed, compare, objectives, calibrate, sanity."""
    # compare
    with pytest.raises(ValueError):
        compare.cutoff([1.0, bad_val])
    with pytest.raises(ValueError):
        compare.compare(
            models={"m": {"cal": [1.0, 2.0], "scores": {"d0": bad_val}}},
            labels={"d0": "pos"},
            kinds={"d0": "k"},
            boot=5,
        )

    # objectives
    with pytest.raises(ValueError):
        objectives.pauc_damage({"pos": [bad_val], "neg": [1.0, 2.0]}, {"pos": [0.0], "neg": [1.0, 2.0]})
    with pytest.raises(ValueError):
        objectives.pauc_damage({"pos": [0.0], "neg": [1.0, bad_val]}, {"pos": [0.0], "neg": [1.0, 2.0]})

    # sanity.floors_from_solo
    with pytest.raises(ValueError):
        sanity.floors_from_solo(
            solo={"blk.0.attn": {"Q4_K": {"pauc": bad_val, "bytes_saved": 100}}},
            uniform={"Q4_K": {"pauc": 0.1, "bytes_saved": 100}},
            ladder=["BF16", "Q4_K"],
            factor=4.0,
            key="pauc",
        )

    # clsbench.calibrate & Evaluator.seed & BatchScorer.score_trj
    hf_dir, tokens, _ = _make_hf_dir(tmp_path, dialect="qwen3_json")
    gguf_path = _write_gguf(tmp_path / "m.gguf", "qwen3", tokens)
    jpath = _make_clsbench_jsonl(tmp_path / "cls.jsonl", n_cal=4, n_search=2, n_select=2, n_exam=2)
    env = clsbench.PublicClassifierEnvironment(hf_dir=str(hf_dir), cache_dir=str(tmp_path / "cache"), jsonl_file=str(jpath))
    with pytest.raises(ValueError):
        env.calibrate({"cal-0": bad_val, "cal-1": 1.0})

    # detect.calibrate
    data_dir, cal_ids = _make_detect_parquet_dir(tmp_path / "data", n_cal=4)
    det_env = detect.DetectEnvironment(
        data_dir=str(data_dir), hf_dir=str(hf_dir), classifier=True,
        calibration_ids_file=str(cal_ids), cache_dir=str(tmp_path / "dcache"),
        search_per_cell=1, select_per_cell=1, exam_test_per_cell=1,
        search_beemo_per_kind=1, select_beemo_per_kind=1, exam_beemo_per_kind=1,
    )
    with pytest.raises(ValueError):
        det_env.calibrate({"wo-cal:cal-0": bad_val, "wo-cal:cal-1": 1.0})

    # BatchScorer.score_trj
    cfg_p = tmp_path / "scores.json"
    cfg_p.write_text(json.dumps({"by_model": {"m.gguf": {"values": [bad_val]}}}))
    tc = runtime.Toolchain(tmp_path, tmp_path / "traj-kl", [""], logodds=_make_logodds_shim(tmp_path / "bin", cfg_p))
    scorer = runtime.BatchScorer(tc, slots_per_gpu=1)
    trj_p = tmp_path / "t.bin"
    kl.write_trj(trj_p, [([2, 3, 4], [2])])
    with pytest.raises(RuntimeError):
        scorer.score_trj(str(gguf_path), str(trj_p), (2, 3))

    # Evaluator.seed
    ev = evaluator.Evaluator(tc, env, {"temperature": 0.0}, 1, 512, "f16", tmp_path / "logs", scorer=scorer)
    with pytest.raises((ValueError, RuntimeError)):
        ev.seed(str(gguf_path), {"cal-0": bad_val})


def test_t2_empty_calibration_and_dataset_rejection(tmp_path: Path):
    """Empty calibration/dataset inputs raise ValueError cleanly (never IndexError or silent pass)."""
    with pytest.raises(ValueError):
        compare.cutoff([])

    hf_dir, _, _ = _make_hf_dir(tmp_path, dialect="qwen3_json")
    jpath = _make_clsbench_jsonl(tmp_path / "cls.jsonl", n_cal=2, n_search=2, n_select=2, n_exam=2)
    env = clsbench.PublicClassifierEnvironment(hf_dir=str(hf_dir), cache_dir=str(tmp_path / "cache"), jsonl_file=str(jpath))
    with pytest.raises(ValueError):
        env.calibrate({})

    data_dir, cal_ids = _make_detect_parquet_dir(tmp_path / "data", n_cal=2)
    det_env = detect.DetectEnvironment(
        data_dir=str(data_dir), hf_dir=str(hf_dir), classifier=True,
        calibration_ids_file=str(cal_ids), cache_dir=str(tmp_path / "dcache"),
        search_per_cell=1, select_per_cell=1, exam_test_per_cell=1,
        search_beemo_per_kind=1, select_beemo_per_kind=1, exam_beemo_per_kind=1,
    )
    with pytest.raises(ValueError):
        det_env.calibrate({})

    class EmptyCalEnv:
        name = "e"
        version = "1"
        classifier = True
        pair = (1, 2)
        budget = {}
        deployment = {"context": 512, "concurrency": 1, "kv_type": "f16"}
        def tasks(self):
            return [protocol.Task("s", "fs", "search", {}), protocol.Task("l", "fl", "select", {}), protocol.Task("e", "fe", "exam", {})]
        def run(self, t, c, s): ...
        def verify(self, t, e): ...
        def score(self, ts, m, sc): return {}
        def calibrate(self, sc): return {}
        def decide(self, t, s, r): ...

    with pytest.raises(ValueError):
        protocol.validate_environment(EmptyCalEnv())


def test_t2_compare_cutoff_rate_one_and_bounds():
    """compare.cutoff(cal, rate=1.0) clamps safely without IndexError and rejects invalid rates."""
    assert compare.cutoff([0.9, 0.1, 0.5], rate=1.0) == pytest.approx(0.9)
    assert compare.cutoff([0.9, 0.1, 0.5], rate=0.0) == pytest.approx(0.1)
    with pytest.raises(ValueError):
        compare.cutoff([0.1, 0.5], rate=-0.1)
    with pytest.raises(ValueError):
        compare.cutoff([0.1, 0.5], rate=1.1)


def test_t2_sandbox_stdout_without_trailing_newline():
    """sandbox.run_tests must not fail when candidate code prints to stdout without trailing newline."""
    code = 'def add_one(x):\n    print("debug", end="")\n    return x + 1\n'
    res = sandbox.run_tests(code, "add_one", [[[2], 3], [[-1], 0]])
    assert res["passed"] is True, res
    assert res["n_ok"] == 2
    assert res["error"] is None


def test_t2_deepmath_difficulty_zero_and_boxed_edge_cases_and_premature_stop(tmp_path: Path, monkeypatch):
    """difficulty=0.0 preserved; extract_boxed_int handles {{42}}, textbf, mbox, unclosed <think>; empty final_text -> premature_stop."""
    # 1. _load_or_build_deepmath must NOT coerce difficulty=0.0 -> 5.0
    fake_ds = [
        {"question": "Easy zero difficulty problem?", "final_answer": "42", "difficulty": 0.0, "topic": "Math -> Algebra -> Lin"},
        {"question": "Valid difficulty four problem?", "final_answer": "17", "difficulty": 4.0, "topic": "Math -> Algebra -> Lin"},
    ]
    import datasets
    monkeypatch.setattr(datasets, "load_dataset", lambda *a, **kw: fake_ds)
    picked = deepmath._load_or_build_deepmath(tmp_path / "dm_cache", max_items=10)
    assert [r["answer"] for r in picked] == [17]

    # 2. JSONL difficulty=0.0 preserved in task.meta
    hf_dir, _, _ = _make_hf_dir(tmp_path, dialect="qwen3_json")
    jpath = tmp_path / "dm0.jsonl"
    jpath.write_text(json.dumps({"id": "z0", "question": "Q", "answer": "42", "difficulty": 0.0, "split": "search"}) + "\n")
    env = deepmath.DeepMathEnvironment(hf_dir=str(hf_dir), jsonl_file=str(jpath))
    t = env.tasks()[0]
    assert t.meta["difficulty"] == pytest.approx(0.0)

    # 3. extract_boxed_int edge cases
    assert deepmath.extract_boxed_int(r"\boxed{{42}}") == 42
    assert deepmath.extract_boxed_int(r"\boxed{\textbf{42}}") == 42
    assert deepmath.extract_boxed_int(r"\boxed{\mbox{42}}") == 42
    assert deepmath.extract_boxed_int(r"<think>draft \boxed{99}") is None

    # 4. empty final_text -> premature_stop=True
    ep_empty = protocol.Episode(t.id, 0, [], None, "", ["stop"], 1, 0, 0.05)
    v_empty = env.verify(t, ep_empty)
    assert v_empty.detectors["premature_stop"] is True
    assert v_empty.success is False


def test_t2_plan_estimate_task_max_targets_none(tmp_path: Path):
    """plan.estimate must not crash with TypeError when proxy.task_max_targets is None."""
    hf_dir, tokens, _ = _make_hf_dir(tmp_path, dialect="qwen3_json")
    gguf_path = _write_gguf(tmp_path / "m.gguf", "qwen3", tokens)
    jpath = _make_deepmath_jsonl(tmp_path / "dm.jsonl")
    cfg = {
        "run_id": "plan-none",
        "run_root": str(tmp_path / "runs"),
        "source": {"gguf": str(gguf_path), "repo": "test/m", "revision": "0", "hf_dir": str(hf_dir)},
        "toolchain": {"llama_bin": str(tmp_path), "traj_kl": str(tmp_path / "tk"), "gpus": [""]},
        "env": {"module": "chantless.env.deepmath", "kwargs": {"hf_dir": str(hf_dir), "jsonl_file": str(jpath)}},
        "ladder": ["BF16", "Q8_0", "Q4_K", "Q2_K"],
        "start_tier": "Q4_K",
        "pool": {"tasks_per_split": {"search": 3, "select": 3, "exam": 3}},
        "proxy": {"task_max_targets": None},
    }
    est = plan.estimate(cfg, source=str(gguf_path))
    assert est["totals"]["prefill_tokens"] > 0
    assert "plan-none" in plan.format_table(est)


def test_t2_sanity_floors_from_solo_signed_pauc_bytes_saved_and_duplicate_bpw():
    """sanity.floors_from_solo handles u[key] <= 0 and drop bytes_saved <= 0; SizeMatchedCap handles duplicate bpw."""
    # Negative uniform pauc_damage: factor=4.0 must allow up to less-negative/zero appropriately, not tighten negative limit.
    # Place Q8_0 at ladder[1] and Q4_K at ladder[2] so min_tier == "Q4_K" proves Q4_K passed!
    res_neg = sanity.floors_from_solo(
        solo={
            "blk.0.ffn": {
                "Q8_0": {"pauc": -0.04, "bytes_saved": 50},
                "Q4_K": {"pauc": -0.02, "bytes_saved": 100},
                "drop": {"pauc": -0.01, "bytes_saved": 0},
            }
        },
        uniform={
            "Q8_0": {"pauc": -0.06, "bytes_saved": 50},
            "Q4_K": {"pauc": -0.05, "bytes_saved": 100},
        },
        ladder=["BF16", "Q8_0", "Q4_K"],
        factor=4.0,
        key="pauc",
    )
    assert res_neg["blk.0.ffn"]["evidence"]["Q4_K"]["ok"] is True
    assert res_neg["blk.0.ffn"]["min_tier"] == "Q4_K"
    assert res_neg["blk.0.ffn"]["drop_ok"] is False  # bytes_saved <= 0 must forbid drop

    # Zero uniform pauc_damage: pauc <= 0.0 passes, while tiny positive 1e-9 candidate fails
    res_zero = sanity.floors_from_solo(
        solo={
            "blk.0.ffn": {
                "Q8_0": {"pauc": 0.0, "bytes_saved": 50},
                "Q4_K": {"pauc": -0.01, "bytes_saved": 100},
                "Q2_K": {"pauc": 1e-9, "bytes_saved": 1024},
            }
        },
        uniform={
            "Q8_0": {"pauc": 0.0, "bytes_saved": 50},
            "Q4_K": {"pauc": 0.0, "bytes_saved": 100},
            "Q2_K": {"pauc": 0.0, "bytes_saved": 1024},
        },
        ladder=["BF16", "Q8_0", "Q4_K", "Q2_K"],
        factor=4.0,
        key="pauc",
    )
    assert res_zero["blk.0.ffn"]["evidence"]["Q4_K"]["ok"] is True
    assert res_zero["blk.0.ffn"]["evidence"]["Q2_K"]["ok"] is False
    assert res_zero["blk.0.ffn"]["min_tier"] == "Q4_K"

    # Duplicate bpw (x1 == x0) in SizeMatchedCap must not raise ZeroDivisionError
    cap = sanity.SizeMatchedCap([(2.0, 0.5), (2.0, 0.7), (4.0, 0.1)])
    assert math.isfinite(cap(2.0))
    assert math.isfinite(cap(3.0))


def test_t2_clsbench_label_sorted_input_skew(tmp_path: Path, monkeypatch):
    """_load_public_cls must not produce all-positive/all-negative skew when HF dataset is label-sorted."""
    import datasets

    # 30 positive (label=1) followed by 30 negative (label=0) in both train and test
    train_rows = [
        {"text": f"Train review {i} with sufficient length to exceed forty characters easily.", "label": 1 if i < 30 else 0}
        for i in range(60)
    ]
    test_rows = [
        {"text": f"Test review {i} with sufficient length to exceed forty characters easily.", "label": 1 if i < 20 else 0}
        for i in range(40)
    ]
    monkeypatch.setattr(datasets, "load_dataset", lambda name, split: train_rows if split == "train" else test_rows)

    rows = clsbench._load_public_cls(tmp_path / "cls_skew", "dummy/yelp", max_cal=10, max_eval=16)
    pool_labels = {r["label"] for r in rows if r["split"] in ("search", "select", "train")}
    exam_labels = {r["label"] for r in rows if r["split"] == "exam"}
    assert pool_labels == {"positive", "negative"}
    assert exam_labels == {"positive", "negative"}


def test_t2_clsbench_trj_for_forbids_untrimmed_fallback_and_pair_for_model_exact_match(tmp_path: Path):
    """trj_for forbids untrimmed self.tok fallback on trimmed GGUF; _pair_for_model requires v==_src_vocab_n & exact bytes."""
    hf_dir, tokens, merges = _make_hf_dir(tmp_path, dialect="qwen3_json", bpe=True)
    src_gguf = _write_gguf(tmp_path / "src.gguf", "qwen3", tokens, merges=merges)
    jpath = _make_clsbench_jsonl(tmp_path / "cls.jsonl", n_cal=2, n_search=2, n_select=2, n_exam=2)
    env = clsbench.PublicClassifierEnvironment(hf_dir=str(hf_dir), cache_dir=str(tmp_path / "cache"), jsonl_file=str(jpath))

    # Build trimmed GGUF keeping answer tokens at different positions AND injecting a substring-collision token at old pair[0]
    u_id, f_id = env.pair
    keep_ids = sorted(set(range(10)) | {u_id, f_id})
    trimmed_gguf = tmp_path / "trimmed.gguf"
    vocabmod.trim_gguf(src_gguf, trimmed_gguf, keep_ids)

    # Remap check: v < _src_vocab_n -> MUST report trimmed=True and exact remapped pair
    remapped_pair, is_trimmed = env._pair_for_model(str(trimmed_gguf))
    assert is_trimmed is True
    assert remapped_pair == (keep_ids.index(u_id), keep_ids.index(f_id))

    # Full source GGUF -> untrimmed (v == _src_vocab_n and exact bytes match)
    src_pair, src_trimmed = env._pair_for_model(str(src_gguf))
    assert src_trimmed is False and src_pair == env.pair

    # Smaller-vocab GGUF (v < _src_vocab_n) where u_id/f_id keep exact bytes at original indices -> MUST still report trimmed=True
    small_same_tokens = list(tokens[:-4])
    assert len(small_same_tokens) < len(tokens)
    assert small_same_tokens[u_id] == b"positive" and small_same_tokens[f_id] == b"negative"
    small_same_gguf = _write_gguf(tmp_path / "small_same.gguf", "qwen3", small_same_tokens, merges=merges)
    small_pair, small_trimmed = env._pair_for_model(str(small_same_gguf))
    assert small_trimmed is True
    assert small_pair == env.pair

    # Substring trap: GGUF with strictly smaller vocab where token at self.pair[0] contains "positive" as substring ("positive_trap")
    trap_tokens = list(tokens[:-4])
    while len(trap_tokens) <= max(u_id, f_id):
        trap_tokens.append(b"pad")
    trap_tokens[u_id] = b"positive_trap"
    trap_tokens[f_id] = b"negative_trap"
    trap_tokens.extend([b"positive", b"negative"])
    assert len(trap_tokens) < len(tokens)
    trap_gguf = _write_gguf(tmp_path / "trap.gguf", "qwen3", trap_tokens, merges=merges)
    trap_pair, trap_trimmed = env._pair_for_model(str(trap_gguf))
    assert trap_trimmed is True
    assert trap_pair != env.pair

    # trj_for on a trimmed GGUF WITHOUT tc.tokdocs MUST raise (forbid untrimmed self.tok fallback)
    tasks = env.tasks()[:2]
    with pytest.raises((ValueError, RuntimeError)):
        env.trj_for(tasks, model_gguf=str(trimmed_gguf), tc=None)


# =============================================================================
# TIER 3: Cross-Feature Combinations (`test_t3_*`)
# =============================================================================


def test_t3_config_load_to_chat_client_precedence_and_zero_trust(tmp_path: Path):
    """config.load_config -> ChatClient decoding precedence, non-leakage, and cache_prompt=False."""
    cfg_path = tmp_path / "extract.yaml"
    cfg_path.write_text(
        json.dumps({
            "run_id": "t3",
            "run_root": str(tmp_path / "runs"),
            "source": {"gguf": "m.gguf", "repo": "r", "revision": "1"},
            "toolchain": {"llama_bin": str(tmp_path), "traj_kl": str(tmp_path / "tk"), "gpus": [""]},
            "env": {"module": "chantless.env.bfcl", "kwargs": {}},
            "decoding": {"temperature": 0.2, "seed": 999, "max_tokens": 8, "reasoning_effort": "low"},
        }),
        encoding="utf-8",
    )
    loaded = cfgmod.load_config(cfg_path, "extract")
    assert loaded.serving.cache_prompt is False
    decoding = loaded.decoding.model_dump(exclude_none=True)

    def responder(path, body):
        if path == "/v1/chat/completions":
            return 200, {"choices": [{"message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}], "usage": {"completion_tokens": 1}}
        return 200, {"choices": [{"text": "ok", "finish_reason": "stop"}], "usage": {"completion_tokens": 1}}

    with _StubServer(responder) as srv:
        cl = client_mod.ChatClient(srv.url, decoding, dialect="inkling2")
        cl.chat([{"role": "user", "content": "hi"}], seed=12, max_tokens=64)
        b_chat = srv.records[-1]["body"]
        assert b_chat["seed"] == 12 and b_chat["max_tokens"] == 64
        assert "reasoning_effort" not in b_chat
        assert b_chat.get("chat_template_kwargs", {}).get("reasoning_effort") == "low"
        assert b_chat["cache_prompt"] is False

        cl.complete("p", seed=13, max_tokens=32)
        b_comp = srv.records[-1]["body"]
        assert b_comp["seed"] == 13 and b_comp["max_tokens"] == 32
        assert "reasoning_effort" not in b_comp
        assert b_comp["cache_prompt"] is False


def test_t3_evaluator_clsbench_vocab_trim_tokdocs_and_rule_cache_isolation(tmp_path: Path):
    """Evaluator + clsbench vocab-trimmed tokdocs retokenization + rule() cache key isolation."""
    hf_dir, tokens, merges = _make_hf_dir(tmp_path, dialect="qwen3_json", bpe=True)
    src_gguf = _write_gguf(tmp_path / "src.gguf", "qwen3", tokens, merges=merges)
    jpath = _make_clsbench_jsonl(tmp_path / "cls.jsonl", n_cal=6, n_search=4, n_select=4, n_exam=4)

    env1 = clsbench.PublicClassifierEnvironment(hf_dir=str(hf_dir), cache_dir=str(tmp_path / "c1"), jsonl_file=str(jpath), fpr_target=0.2)
    u_id, f_id = env1.pair
    keep_ids = sorted(set(range(10)) | {u_id, f_id} | {env1.tok.encode(w, add_special_tokens=False)[0] for w in ("no", "yes")})
    trimmed_gguf = tmp_path / "trimmed.gguf"
    vocabmod.trim_gguf(src_gguf, trimmed_gguf, keep_ids)

    remapped_u, remapped_f = keep_ids.index(u_id), keep_ids.index(f_id)
    score_cfg = tmp_path / "scores.json"
    score_cfg.write_text(json.dumps({
        "by_model": {
            "trimmed.gguf": {
                f"{remapped_u},{remapped_f}": [0.5, 1.0, 2.0, 3.0, 4.0, 5.0],
                f"{keep_ids.index(env1.tok.encode('yes', add_special_tokens=False)[0])},{keep_ids.index(env1.tok.encode('no', add_special_tokens=False)[0])}": [9.0, 9.0, 9.0, 9.0, 9.0, 9.0],
            }
        }
    }))
    tc = runtime.Toolchain(
        tmp_path, tmp_path / "tk", [""],
        logodds=_make_logodds_shim(tmp_path / "bin", score_cfg),
        tokdocs=_make_tokdocs_shim(tmp_path / "bin"),
    )
    scorer = runtime.BatchScorer(tc, slots_per_gpu=1)
    ev1 = evaluator.Evaluator(tc, env1, {"temperature": 0.0}, 1, 512, "f16", tmp_path / "logs", scorer=scorer)
    r1 = ev1.rule(str(trimmed_gguf))
    assert r1["cutoff"] == pytest.approx(1.0)

    # Same log_dir, different answers -> MUST NOT hit stale disk rule
    jpath_yesno = _make_clsbench_jsonl(
        tmp_path / "cls_yesno.jsonl", n_cal=6, n_search=4, n_select=4, n_exam=4, flagged="no", unflagged="yes",
    )
    env2 = clsbench.PublicClassifierEnvironment(
        hf_dir=str(hf_dir), cache_dir=str(tmp_path / "c2"), jsonl_file=str(jpath_yesno),
        answers=("no", "yes"), fpr_target=0.2,
    )
    ev2 = evaluator.Evaluator(tc, env2, {"temperature": 0.0}, 1, 512, "f16", tmp_path / "logs", scorer=scorer)
    r2 = ev2.rule(str(trimmed_gguf))
    assert r2["cutoff"] == pytest.approx(9.0)


def test_t3_deepmath_reasoning_effort_to_kl_renderer_prompt_parity(tmp_path: Path):
    """DeepMathEnvironment(reasoning_effort=...) -> kl.Renderer prompt parity and valid target generation."""
    hf_dir, _, _ = _make_hf_dir(tmp_path, dialect="inkling2")
    jpath = _make_deepmath_jsonl(tmp_path / "dm.jsonl")
    env = deepmath.DeepMathEnvironment(hf_dir=str(hf_dir), jsonl_file=str(jpath), reasoning_effort="low")
    task = env.tasks()[0]

    def responder(path, body):
        assert path == "/v1/completions"
        assert "Thinking effort level: 0.2<|end_message|>" in body["prompt"]
        return 200, {
            "choices": [{
                "text": "<|message_model|><|content_text|>\\boxed{5}<|end_message|><|content_model_end_sampling|>",
                "finish_reason": "stop",
            }],
            "usage": {"completion_tokens": 6},
        }

    with _StubServer(responder) as srv:
        cl = client_mod.ChatClient(srv.url, {"temperature": 0.0}, dialect="inkling2")
        ep = env.run(task, cl, seed=1)
        assert env.verify(task, ep).success is True

    R = kl.Renderer(str(hf_dir), reasoning_effort=env.reasoning_effort)
    assert R._render(ep.messages[:-1], None, True) == env.render_prompt(task)
    recs = R.episode(ep.messages, None)
    assert len(recs) == 1 and len(recs[0][1]) > 0


def test_t3_codetasks_detectors_sandbox_multiturn_and_multidialect_tags():
    """CodeTasksEnv + detectors + sandbox multi-turn execution, <think> stripping, and multi-dialect malformed tags."""
    env = codetasks.CodeTasksEnv(seed=20261012, variants_per_family=1)
    task = env.tasks()[0]
    turn_idx = {"n": 0}

    def responder(path, body):
        turn_idx["n"] += 1
        if turn_idx["n"] == 1:
            # Candidate prints without trailing newline during tool call
            cand = f'print("trace", end="")\n{task.meta["reference"]}'
            return 200, {
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [{"id": "tc1", "function": {"name": "run_tests", "arguments": json.dumps({"code": cand})}}],
                    },
                    "finish_reason": "tool_calls",
                }],
                "usage": {"completion_tokens": 10},
            }
        # Final reply includes decoy broken code inside <think> block and valid code outside
        final_msg = f"<think>\n```python\ndef broken(:\n```\n</think>\n```python\n{task.meta['reference']}\n```"
        return 200, {
            "choices": [{"message": {"role": "assistant", "content": final_msg}, "finish_reason": "stop"}],
            "usage": {"completion_tokens": 12},
        }

    with _StubServer(responder) as srv:
        cl = client_mod.ChatClient(srv.url, {"temperature": 0.0})
        ep = env.run(task, cl, seed=42)
        v = env.verify(task, ep)
        assert v.tests_passed is True
        assert v.detectors["token_garbage"] is False
        assert v.success is True

    # Code inside <think>...</think> only MUST NOT be extracted
    assert detectors.extract_code("<think>\n```python\ndef f(): pass\n```\n</think>\nNo code") is None

    # Multi-dialect malformed tool-call detection
    for tag in ("<tool_call>{}", "<|tool_call>call:run_tests{}", "<function=run_tests>", "<|content_invoke_tool_json|>{}"):
        assert detectors.malformed_tool_call([{"role": "assistant", "content": tag}], codetasks.TOOL_NAMES) is True, tag


# =============================================================================
# TIER 4: Real-World Offline E2E Scenarios (`test_t4_*`)
# =============================================================================


def test_t4_e2e_clsbench_eval_compare_pauc_and_auroc(tmp_path: Path):
    """Scenario 1: Offline clsbench eval + compare + pauc_damage + AUROC pipeline."""
    hf_dir, tokens, merges = _make_hf_dir(tmp_path, dialect="qwen3_json", bpe=True)
    src_gguf = _write_gguf(tmp_path / "src.gguf", "qwen3", tokens, merges=merges)
    jpath = _make_clsbench_jsonl(tmp_path / "cls.jsonl", n_cal=20, n_search=10, n_select=10, n_exam=10)
    env = clsbench.PublicClassifierEnvironment(
        hf_dir=str(hf_dir), cache_dir=str(tmp_path / "cache"), jsonl_file=str(jpath), fpr_target=0.05,
    )
    protocol.validate_environment(env)

    u_id, f_id = env.pair
    keep_ids = sorted(set(range(10)) | {u_id, f_id})
    cand_gguf = tmp_path / "cand.gguf"
    vocabmod.trim_gguf(src_gguf, cand_gguf, keep_ids)
    cu, cf = keep_ids.index(u_id), keep_ids.index(f_id)

    # Configure deterministic scores: flagged ("negative") lower than unflagged ("positive")
    all_tasks = env.tasks()
    # Partition by split using task order in trj: cal=20, search=10, select=10, exam=10
    # Calibration all positive -> high scores; exam even indices negative -> low scores
    src_scores_by_tok: dict[str, float] = {}
    cand_scores_by_tok: dict[str, float] = {}

    tc = runtime.Toolchain(
        tmp_path, tmp_path / "tk", [""],
        tokdocs=_make_tokdocs_shim(tmp_path / "bin"),
        logodds=None,
    )
    # Pre-compute tokenized prompts for src and cand to map exact token sequences in logodds shim
    for t in all_tasks:
        is_flag = t.meta["label"] == "negative"
        src_val = -2.0 if is_flag else 2.0
        cand_val = -3.0 if is_flag else 3.0
        s_ids = env.tok(env._prompt_str[t.id], add_special_tokens=False)["input_ids"]
        src_scores_by_tok[",".join(map(str, s_ids))] = src_val
    c_seqs = runtime.tokenize_docs(tc, str(cand_gguf), [env._prompt_str[t.id] for t in all_tasks])
    for t, seq in zip(all_tasks, c_seqs):
        is_flag = t.meta["label"] == "negative"
        cand_scores_by_tok[",".join(map(str, seq.tolist()))] = -3.0 if is_flag else 3.0

    score_cfg = tmp_path / "scores.json"
    score_cfg.write_text(json.dumps({
        "by_model": {
            "src.gguf": {"token_scores": src_scores_by_tok},
            "cand.gguf": {"token_scores": cand_scores_by_tok},
        }
    }))
    tc.logodds = _make_logodds_shim(tmp_path / "bin", score_cfg)
    scorer = runtime.BatchScorer(tc, slots_per_gpu=1)
    ev = evaluator.Evaluator(tc, env, {"temperature": 0.0}, 1, 512, "f16", tmp_path / "logs", scorer=scorer)

    exam_tasks = [t for t in all_tasks if t.split == "exam"]
    jobs = [(t, 0) for t in exam_tasks]
    src_rows = ev.run(str(src_gguf), jobs, "src_exam", side="src")
    cand_rows = ev.run(str(cand_gguf), jobs, "cand_exam", side="cand")
    assert all(r["success"] is True for r in src_rows)
    assert all(r["success"] is True for r in cand_rows)

    cal_tasks = [t for t in all_tasks if t.split == "calibration"]
    s_src_cal = ev.scores(str(src_gguf), cal_tasks)
    s_src_ex = ev.scores(str(src_gguf), exam_tasks)
    s_cand_cal = ev.scores(str(cand_gguf), cal_tasks)
    s_cand_ex = ev.scores(str(cand_gguf), exam_tasks)

    cmp_res = compare.compare(
        models={
            "src": {"cal": list(s_src_cal.values()), "scores": s_src_ex},
            "cand": {"cal": list(s_cand_cal.values()), "scores": s_cand_ex},
        },
        labels={t.id: ("pos" if t.meta["label"] == "negative" else "neg") for t in exam_tasks},
        kinds={t.id: "sentiment" for t in exam_tasks},
        boot=50,
        seed=0,
    )
    assert cmp_res["models"]["cand"]["pos"]["caught"] == pytest.approx(1.0)
    assert cmp_res["models"]["cand"]["neg_flagged"] == pytest.approx(0.0)

    # AUROC orientation == 1.0 (not 0.0) and pauc_damage negative (cand separates even wider than src)
    y_exam = np.array([t.meta["label"] == "negative" for t in exam_tasks])
    sc_exam = np.array([s_cand_ex[t.id] for t in exam_tasks])
    assert anneal.auc(sc_exam, y_exam) == pytest.approx(1.0)

    src_part = {"pos": np.array([s_src_ex[t.id] for t in exam_tasks if t.meta["label"] == "negative"]),
                "neg": np.array([s_src_ex[t.id] for t in exam_tasks if t.meta["label"] == "positive"])}
    cand_part = {"pos": np.array([s_cand_ex[t.id] for t in exam_tasks if t.meta["label"] == "negative"]),
                 "neg": np.array([s_cand_ex[t.id] for t in exam_tasks if t.meta["label"] == "positive"])}
    assert objectives.pauc_damage(cand_part, src_part, tail=0.2, tau=0.5) < 0.0


def test_t4_e2e_deepmath_eval_kl_renderer_and_write_trj(tmp_path: Path):
    """Scenario 2: Offline deepmath eval via loopback /v1/completions + kl.Renderer + TRJ1 verification."""
    hf_dir, _, _ = _make_hf_dir(tmp_path, dialect="inkling2")
    jpath = _make_deepmath_jsonl(tmp_path / "dm.jsonl")
    env = deepmath.DeepMathEnvironment(hf_dir=str(hf_dir), jsonl_file=str(jpath), reasoning_effort="none")
    protocol.validate_environment(env)

    select_tasks = env.sample_tasks("select", 3, stratified=True)
    ans_by_q = {t.prompt["messages"][1]["content"]: t.meta["answer"] for t in select_tasks}

    def responder(path, body):
        assert path == "/v1/completions"
        q = next(q for q in ans_by_q if q in body["prompt"])
        ans = ans_by_q[q]
        return 200, {
            "choices": [{
                "text": f"<|message_model|><|content_text|>\\boxed{{{ans}}}<|end_message|><|content_model_end_sampling|>",
                "finish_reason": "stop",
            }],
            "usage": {"completion_tokens": 5},
        }

    with _StubServer(responder) as srv:
        tc = runtime.Toolchain(tmp_path, tmp_path / "tk", [""])
        ev = evaluator.Evaluator(tc, env, {"temperature": 0.0}, 1, 512, "f16", tmp_path / "logs")
        rows = ev.run("m", [(t, 0) for t in select_tasks], "dm_select", srvs=[SimpleNamespace(url=srv.url, special=True)])
        assert len(rows) == 3 and all(r["success"] is True for r in rows)

    R = kl.Renderer(str(hf_dir), reasoning_effort=env.reasoning_effort)
    recs = []
    for ep in ev.last_episodes:
        recs.extend(R.episode(ep.messages, None))
    trj_p = tmp_path / "dm.trj"
    meta = kl.write_trj(trj_p, recs)
    assert meta["n_seq"] == 3 and meta["n_targets"] > 0
    assert runtime.trj_targets(trj_p) == [len(tg) for _, tg in recs]


def test_t4_e2e_codetasks_multiturn_tool_use_sandbox_verify(tmp_path: Path):
    """Scenario 3: Offline codetasks multi-turn tool-use (run_tests) + sandbox + verify via /v1/chat/completions."""
    env = codetasks.CodeTasksEnv(seed=20261012, variants_per_family=1, max_turns=3)
    protocol.validate_environment(env)
    select_tasks = [t for t in env.tasks() if t.split == "select"][:2]
    ref_by_fname = {t.meta["fname"]: t.meta["reference"] for t in select_tasks}

    def responder(path, body):
        assert path == "/v1/chat/completions"
        assert "tools" in body and body["tools"]
        assert body.get("cache_prompt") is False
        msgs = body["messages"]
        user_msg = msgs[1]["content"]
        fname = next(fn for fn in ref_by_fname if f"`{fn}(" in user_msg)
        ref = ref_by_fname[fname]
        if not any(m.get("role") == "tool" for m in msgs):
            return 200, {
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [{"id": "call_1", "function": {"name": "run_tests", "arguments": json.dumps({"code": ref})}}],
                    },
                    "finish_reason": "tool_calls",
                }],
                "usage": {"completion_tokens": 10},
            }
        # Confirm sandbox feedback was delivered back to the model in turn 2
        tool_out = next(m["content"] for m in msgs if m.get("role") == "tool")
        assert "2/2 examples passed" in tool_out
        return 200, {
            "choices": [{"message": {"role": "assistant", "content": f"```python\n{ref}\n```"}, "finish_reason": "stop"}],
            "usage": {"completion_tokens": 10},
        }

    with _StubServer(responder) as srv:
        tc = runtime.Toolchain(tmp_path, tmp_path / "tk", [""])
        ev = evaluator.Evaluator(tc, env, {"temperature": 0.0}, 1, 2048, "f16", tmp_path / "logs")
        rows = ev.run("m", [(t, 1) for t in select_tasks], "code_sel", srvs=[SimpleNamespace(url=srv.url, special=False)])
        assert len(rows) == 2 and all(r["success"] is True and r["turns"] == 2 for r in rows)


def test_t4_e2e_detect_classifier_sanity_floors_and_plan_estimate(tmp_path: Path):
    """Scenario 4: Offline detect (classifier=True) + sanity.floors_from_solo + plan.estimate."""
    hf_dir, tokens, _ = _make_hf_dir(tmp_path, dialect="inkling2")
    gguf_path = _write_gguf(tmp_path / "ink.gguf", "inkling2", tokens)
    data_dir, cal_ids = _make_detect_parquet_dir(tmp_path / "data", n_cal=10)
    tc = runtime.Toolchain(
        tmp_path, tmp_path / "tk", [""],
        tokdocs=_make_tokdocs_shim(tmp_path / "bin"),
        logodds=_make_logodds_shim(tmp_path / "bin"),
    )

    env = detect.DetectEnvironment(
        data_dir=str(data_dir),
        hf_dir=str(hf_dir),
        classifier=True,
        calibration_ids_file=str(cal_ids),
        cache_dir=str(tmp_path / "dcache"),
        search_per_cell=2,
        select_per_cell=2,
        exam_test_per_cell=2,
        search_beemo_per_kind=2,
        select_beemo_per_kind=2,
        exam_beemo_per_kind=2,
    )
    protocol.validate_environment(env)

    scorer = runtime.BatchScorer(tc, slots_per_gpu=1)
    ev = evaluator.Evaluator(tc, env, {"temperature": 0.0}, 1, 512, "f16", tmp_path / "logs", scorer=scorer)
    exam_jobs = [(t, 0) for t in env.tasks() if t.split == "exam"]
    rows = ev.run(str(gguf_path), exam_jobs, "det_exam")
    assert len(rows) == len(exam_jobs)

    floors = sanity.floors_from_solo(
        solo={
            "blk.0.attn": {
                "Q8_0": {"pauc": -0.03, "bytes_saved": 512},
                "Q4_K": {"pauc": -0.02, "bytes_saved": 1024},
                "IQ2_XS": {"pauc": 0.8, "bytes_saved": 2048},
            }
        },
        uniform={
            "Q8_0": {"pauc": -0.04, "bytes_saved": 512},
            "Q4_K": {"pauc": -0.03, "bytes_saved": 1024},
            "IQ2_XS": {"pauc": 0.05, "bytes_saved": 2048},
        },
        ladder=["BF16", "Q8_0", "Q4_K", "IQ2_XS"],
        factor=4.0,
        key="pauc",
    )
    assert floors["blk.0.attn"]["evidence"]["Q4_K"]["ok"] is True
    assert floors["blk.0.attn"]["evidence"]["IQ2_XS"]["ok"] is False
    assert floors["blk.0.attn"]["min_tier"] == "Q4_K"

    cfg = {
        "run_id": "e2e-det",
        "run_root": str(tmp_path / "runs"),
        "source": {"gguf": str(gguf_path), "repo": "test/ink", "revision": "0", "hf_dir": str(hf_dir)},
        "toolchain": {"llama_bin": str(tmp_path), "traj_kl": str(tmp_path / "tk"), "gpus": [""]},
        "env": {
            "module": "chantless.env.detect",
            "kwargs": {
                "data_dir": str(data_dir),
                "hf_dir": str(hf_dir),
                "classifier": True,
                "calibration_ids_file": str(cal_ids),
                "cache_dir": str(tmp_path / "dcache"),
                "search_per_cell": 2,
                "select_per_cell": 2,
                "exam_test_per_cell": 2,
                "search_beemo_per_kind": 2,
                "select_beemo_per_kind": 2,
                "exam_beemo_per_kind": 2,
            },
        },
        "ladder": ["BF16", "Q8_0", "Q4_K", "Q2_K"],
        "start_tier": "Q4_K",
        "pool": {"tasks_per_split": {"search": 4, "select": 4, "exam": 4}},
        "proxy": {"task_max_targets": None, "spec": {"n_neg": 4, "n_pos": 2}},
    }
    est = plan.estimate(cfg, source=str(gguf_path))
    assert est["classifier"] is True
    assert est["totals"]["prefill_tokens"] > 0
    assert "classifier path" in plan.format_table(est)


def test_t4_e2e_mvp_configs_load_and_cli_validate_schema(tmp_path: Path):
    """Scenario 5: Validate all 6 mvp/configs/* via config.load_config and CLI `validate` + `schema`."""
    configs = [
        ("extract-example.yaml", "extract"),
        ("inkling2-mini-bfcl.yaml", "extract"),
        ("inkling2-mini-clsbench.yaml", "extract"),
        ("inkling2-mini-deepmath.yaml", "extract"),
        ("learn-example.yaml", "learn"),
        ("local-smoke.json", "extract"),
    ]
    env = {**os.environ, "PYTHONPATH": str(ROOT / "mvp")}

    for fname, kind in configs:
        cpath = ROOT / "mvp" / "configs" / fname
        model = cfgmod.load_config(cpath, kind)
        if kind == "extract":
            assert model.serving.cache_prompt is False, fname
        else:
            assert model.learn.judge_cache_prompt is False, fname

        proc = subprocess.run(
            [sys.executable, "-m", "chantless.cli", "validate", "--config", str(cpath), "--kind", kind],
            capture_output=True, text=True, env=env,
        )
        assert proc.returncode == 0, f"{fname}: {proc.stderr}"
        assert json.loads(proc.stdout)["valid"] is True

    # Verify CLI `schema all` and `schema anneal`
    out_schema = tmp_path / "all_schema.json"
    p_all = subprocess.run(
        [sys.executable, "-m", "chantless.cli", "schema", "all", "--out", str(out_schema)],
        capture_output=True, text=True, env=env,
    )
    assert p_all.returncode == 0, p_all.stderr
    schemas = json.loads(out_schema.read_text())
    assert {"extract", "learn", "sweep", "anneal"} <= set(schemas)

    p_ann = subprocess.run(
        [sys.executable, "-m", "chantless.cli", "schema", "anneal"],
        capture_output=True, text=True, env=env,
    )
    assert p_ann.returncode == 0, p_ann.stderr
    assert "anneal" in json.loads(p_ann.stdout)["properties"]
