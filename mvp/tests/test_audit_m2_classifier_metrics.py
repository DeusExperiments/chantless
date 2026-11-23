"""Deterministic offline M2 regression tests: clsbench, detect, protocol, compare, sanity.

All tests are 100% self-contained in `tmp_path` and run completely offline without `.local/` assets.
"""
from __future__ import annotations

import json
import math
import stat
import struct
import sys
import threading
import time
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

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "mvp") not in sys.path:
    sys.path.insert(0, str(ROOT / "mvp"))

from chantless import anneal, client as client_mod, compare, gguf_ops, objectives, plan, runtime, sanity, vocab as vocabmod
from chantless.env import clsbench, detect, protocol


# ---------------------------------------------------------------------------
# Offline synthetic helpers (GGUF, HF tokenizers, loopback server, shims)
# ---------------------------------------------------------------------------


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
    merges: list[bytes] | None = None,
    extra_kvs: list[tuple[str, tuple[int, bytes]]] | None = None,
    n_layer: int = 1,
    n_embd: int = 64,
    n_ff: int = 64,
) -> Path:
    v_size = len(tokens)
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

    rng = np.random.default_rng(13)
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
    "{%- elif m.role == 'assistant' -%}"
    "{{- '<|message_model|><|content_text|>' ~ m.content ~ '<|end_message|><|content_model_end_sampling|>' -}}"
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


def _make_hf_dir(
    out_dir: Path,
    dialect: str = "qwen3_json",
    kind: str = "bpe",
    extra_words: list[str] | None = None,
    bos_token: str | None = None,
    metaspace: bool = False,
) -> tuple[Path, list[bytes], list[bytes]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    if dialect == "inkling2":
        special = [
            "<unk>",
            "<|content_model_end_sampling|>",
            "<|message_system|>",
            "<|message_user|>",
            "<|message_model|>",
            "<|content_text|>",
            "<|content_thinking|>",
            "<|content_invoke_tool_json|>",
            "<|end_message|>",
        ]
        eos_tok = "<|content_model_end_sampling|>"
        ct = _INKLING2_TEMPLATE
    else:
        special = ["<unk>", "<|im_end|>", "<|im_start|>", "<think>", "</think>"]
        if bos_token and bos_token not in special:
            special.insert(1, bos_token)
        eos_tok = "<|im_end|>"
        ct = _QWEN3_TEMPLATE

    words = [
        "negative",
        "positive",
        "human",
        "ai",
        "yes",
        "no",
        "alpha",
        "beta",
        "gamma",
        "delta",
        "epsilon",
        "zeta",
        "eta",
        "theta",
        "hello",
        "world",
        "review",
        "quality",
        "service",
        "good",
        "bad",
        "document",
    ] + list(extra_words or [])

    if kind == "bpe":
        vocab_list: list[str] = list(special)
        for c in range(32, 127):
            ch = chr(c)
            if ch not in vocab_list:
                vocab_list.append(ch)
        merges_str: list[str] = ["a b", "ab c"]
        for m in merges_str:
            merged = m.replace(" ", "")
            if merged not in vocab_list:
                vocab_list.append(merged)
        for w in words:
            if len(w) >= 2:
                cur = w[0]
                for ch in w[1:]:
                    pair = f"{cur} {ch}"
                    cur = cur + ch
                    if pair not in merges_str:
                        merges_str.append(pair)
                    if cur not in vocab_list:
                        vocab_list.append(cur)
            elif w not in vocab_list:
                vocab_list.append(w)
        vocab_map = {t: i for i, t in enumerate(vocab_list)}
        merges_pairs = [tuple(m.split(" ")) for m in merges_str]
        tk_obj = Tokenizer(models.BPE(vocab=vocab_map, merges=merges_pairs, unk_token="<unk>"))
        tok = PreTrainedTokenizerFast(
            tokenizer_object=tk_obj,
            unk_token="<unk>",
            eos_token=eos_tok,
            bos_token=bos_token,
            additional_special_tokens=[t for t in special if t not in ("<unk>", eos_tok, bos_token)],
            chat_template=ct,
        )
        tok.save_pretrained(out_dir)
        return out_dir, [t.encode("utf-8") for t in vocab_list], [m.encode("utf-8") for m in merges_str]

    if kind == "wordlevel":
        vocab_list = []
        for t in special + words:
            if t not in vocab_list:
                vocab_list.append(t)
        vocab_map = {t: i for i, t in enumerate(vocab_list)}
        tk_obj = Tokenizer(models.WordLevel(vocab=vocab_map, unk_token="<unk>"))
        tk_obj.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
        tok = PreTrainedTokenizerFast(
            tokenizer_object=tk_obj,
            unk_token="<unk>",
            eos_token=eos_tok,
            bos_token=bos_token,
            additional_special_tokens=[t for t in special if t not in ("<unk>", eos_tok, bos_token)],
            chat_template=ct,
        )
        tok.save_pretrained(out_dir)
        return out_dir, [t.encode("utf-8") for t in vocab_list], []

    if kind == "wordpiece":
        vocab_list = []
        for t in special + words:
            if t not in vocab_list:
                vocab_list.append(t)
        vocab_map = {t: i for i, t in enumerate(vocab_list)}
        tk_obj = Tokenizer(models.WordPiece(vocab=vocab_map, unk_token="<unk>"))
        tk_obj.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
        tok = PreTrainedTokenizerFast(
            tokenizer_object=tk_obj,
            unk_token="<unk>",
            eos_token=eos_tok,
            bos_token=bos_token,
            additional_special_tokens=[t for t in special if t not in ("<unk>", eos_tok, bos_token)],
            chat_template=ct,
        )
        tok.save_pretrained(out_dir)
        return out_dir, [t.encode("utf-8") for t in vocab_list], []

    if kind == "unigram":
        vocab_items: list[tuple[str, float]] = [(t, 0.0) for t in special]
        if metaspace:
            vocab_items += [(f"▁{w}", -1.0) for w in words] + [(w, -4.0) for w in words] + [("▁", -2.0)]
        else:
            vocab_items += [(w, -1.0) for w in words]
        seen_u = {t for t, _ in vocab_items}
        for c in range(32, 127):
            ch = chr(c)
            if ch not in seen_u:
                vocab_items.append((ch, -5.0))
                seen_u.add(ch)
        tk_obj = Tokenizer(models.Unigram(vocab_items, unk_id=0))
        if metaspace:
            tk_obj.pre_tokenizer = pre_tokenizers.Metaspace()
        tok = PreTrainedTokenizerFast(
            tokenizer_object=tk_obj,
            unk_token="<unk>",
            eos_token=eos_tok,
            bos_token=bos_token,
            additional_special_tokens=[t for t in special if t not in ("<unk>", eos_tok, bos_token)],
            chat_template=ct,
        )
        tok.save_pretrained(out_dir)
        return out_dir, [t.encode("utf-8") for t, _ in vocab_items], []

    raise ValueError(kind)


def _make_tokdocs_shim(bin_dir: Path) -> Path:
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
    prompts = [f.read(struct.unpack("<I", f.read(4))[0]).decode("utf-8") for _ in range(n_docs)]
sorted_toks = sorted([k for k in tok_to_id if k], key=len, reverse=True)
encoded = []
for s in prompts:
    ids, pos = [], 0
    while pos < len(s):
        hit = next((cand for cand in sorted_toks if s.startswith(cand, pos)), None)
        if hit is not None:
            ids.append(tok_to_id[hit]); pos += len(hit)
        else:
            ids.append(unk_id); pos += 1
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


def _make_logodds_shim(bin_dir: Path, score_cfg_path: Path | None = None) -> Path:
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
for trj, pair, out_p in jobs:
    with open(trj, "rb") as f:
        head = f.read(8)
        n_seqs = struct.unpack("<I", head[4:])[0]
        seqs = []
        for _ in range(n_seqs):
            nt, ng = struct.unpack("<II", f.read(8))
            toks = struct.unpack(f"<{{nt}}i", f.read(4 * nt))
            tgts = struct.unpack(f"<{{ng}}I", f.read(4 * ng))
            seqs.append((toks, tgts))
    m_key = os.path.basename(model) if model else ""
    m_rules = cfg.get("by_model", {{}}).get(m_key, {{}})
    mode = m_rules.get("mode", "values")
    tokens = vocabmod.read_tokenizer(model)["tokens"] if mode == "vocab_decode" else None
    v = len(tokens) if tokens is not None else None
    if mode == "vocab_decode":
        assert 0 <= pair[0] < v and 0 <= pair[1] < v and pair[0] != pair[1]
        if "expected_pair_bytes" in m_rules:
            assert tokens[pair[0]].decode("utf-8") == m_rules["expected_pair_bytes"][0]
            assert tokens[pair[1]].decode("utf-8") == m_rules["expected_pair_bytes"][1]
    with open(out_p, "w") as out_f:
        for ti, (toks, tgts) in enumerate(seqs):
            if ti % shard_n != shard_i:
                continue
            if mode == "vocab_decode":
                assert all(0 <= tid < v for tid in toks)
                assert len(tgts) == 1 and tgts[0] == len(toks) - 1 and toks[tgts[0]] == pair[0]
                raw_txt = b"".join(tokens[tid] for tid in toks[:-1]).decode("utf-8", errors="replace")
                if "<|im_start|>user" in raw_txt:
                    user_txt = raw_txt.split("<|im_start|>user", 1)[1].split("<|im_end|>", 1)[0]
                elif "<|message_user|><|content_text|>" in raw_txt:
                    user_txt = raw_txt.split("<|message_user|><|content_text|>", 1)[1].split("<|end_message|>", 1)[0]
                else:
                    user_txt = raw_txt
                if any(s in user_txt for s in m_rules["flagged_substrs"]):
                    base = float(m_rules["flagged_score"])
                elif any(s in user_txt for s in m_rules.get("cal_substrs", [])):
                    base = float(m_rules["cal_score"])
                elif any(s in user_txt for s in m_rules["unflagged_substrs"]):
                    base = float(m_rules["unflagged_score"])
                else:
                    raise AssertionError(f"unexpected user_txt: {{user_txt!r}}")
                jitter = (hashlib.sha256(user_txt.encode("utf-8")).digest()[0] % 7) * float(m_rules.get("step", 0.05))
                val = base + jitter
                for k in range(len(tgts)):
                    out_f.write(f"{{ti}}\\t{{k}}\\t{{val}}\\n")
            else:
                for k in range(len(tgts)):
                    val = m_rules["values"][ti % len(m_rules["values"])] if "values" in m_rules else 1.0
                    out_f.write(f"{{ti}}\\t{{k}}\\t{{val}}\\n")
""",
        encoding="utf-8",
    )
    p.chmod(p.stat().st_mode | stat.S_IEXEC)
    return p


def _make_cls_jsonl(path: Path, n_cal: int = 6, n_each: int = 4, cal_label: str | None = None) -> Path:
    rows = []
    for i in range(n_cal):
        rows.append({
            "id": f"cal-{i}",
            "text": f"positive review number {i} with good service and quality",
            "label": cal_label or "positive",
            "split": "calibration",
            "family": f"cal/b{i % 2:02d}",
        })
    for sp in ("search", "select", "exam"):
        for i in range(n_each):
            lbl = "negative" if i % 2 == 0 else "positive"
            rows.append({
                "id": f"{sp}-{i}",
                "text": f"{lbl} review number {i} in {sp}",
                "label": lbl,
                "split": sp,
                "family": f"{sp}/b{i % 2:02d}",
            })
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return path


def _make_detect_data(data_dir: Path, n_cal: int = 8, cal_labels: list[str] | None = None) -> tuple[Path, Path]:
    wo = data_dir / "WildOutlines" / "data"
    be = data_dir / "beemo" / "data"
    wo.mkdir(parents=True, exist_ok=True)
    be.mkdir(parents=True, exist_ok=True)
    val_rows = []
    for lbl in ("human", "ai"):
        for i in range(1006):
            val_rows.append({"id": f"val-{lbl}-{i}", "text": f"alpha beta {lbl} {i}", "label": lbl, "format": "essay" if i % 2 == 0 else "news"})
    pq.write_table(pa.Table.from_pylist(val_rows), wo / "validation-00000-of-00001.parquet")

    test_rows = [
        {"id": f"test-{lbl}-{i}", "text": f"gamma delta {lbl} {i}", "label": lbl, "format": "essay" if i % 2 == 0 else "news"}
        for lbl in ("human", "ai")
        for i in range(6)
    ]
    pq.write_table(pa.Table.from_pylist(test_rows), wo / "test-00000-of-00001.parquet")

    cal_rows = [
        {"id": f"cal-{i}", "text": f"hello world human {i}", "label": (cal_labels[i] if cal_labels else "human"), "format": "essay"}
        for i in range(n_cal)
    ]
    pq.write_table(pa.Table.from_pylist(cal_rows), wo / "calibration-00000-of-00001.parquet")

    beemo_rows = [
        {
            "id": f"b{i}",
            "prompt_id": f"p{i}",
            "category": "summary",
            "human_output": f"Beemo human {i}",
            "model_output": f"Beemo ai {i}",
            "human_edits": f"Beemo edited {i}",
        }
        for i in range(12)
    ]
    pq.write_table(pa.Table.from_pylist(beemo_rows), be / "train-00000-of-00001.parquet")

    cal_ids = data_dir / "cal_ids.json"
    cal_ids.write_text(json.dumps([r["id"] for r in cal_rows]), encoding="utf-8")
    return data_dir, cal_ids


class _LoopbackServer:
    def __init__(self, responder: Callable[[str, dict[str, Any]], tuple[int, Any] | None]):
        self.records: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        parent = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                n = int(self.headers.get("Content-Length", "0"))
                raw = self.rfile.read(n)
                body = json.loads(raw.decode("utf-8")) if raw else {}
                with parent._lock:
                    parent.records.append({"path": self.path, "body": body})
                res = responder(self.path, body)
                if res is None:
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

    def __enter__(self):
        self._th.start()
        return self

    def __exit__(self, *args):
        self._srv.shutdown()
        self._srv.server_close()
        self._th.join(timeout=2)


# ---------------------------------------------------------------------------
# 1. clsbench regression tests
# ---------------------------------------------------------------------------


def test_clsbench_load_public_cls_label_sorted_no_skew(tmp_path: Path, monkeypatch):
    import datasets

    train_rows = [
        {"text": f"Train review {i} with plenty of characters to pass length filtering.", "label": 1 if i < 60 else 0}
        for i in range(120)
    ]
    test_rows = [
        {"text": f"Test review {i} with plenty of characters to pass length filtering.", "label": 1 if i < 30 else 0}
        for i in range(60)
    ]
    monkeypatch.setattr(datasets, "load_dataset", lambda name, split: train_rows if split == "train" else test_rows)

    rows = clsbench._load_public_cls(tmp_path / "cache", "dummy/yelp", max_cal=20, max_eval=36)
    cal = [r for r in rows if r["split"] == "calibration"]
    assert len(cal) == 20
    assert {r["label"] for r in cal} == {"positive"}

    cal_texts = {r["text"] for r in cal}
    eval_texts = {r["text"] for r in rows if r["split"] != "calibration"}
    assert cal_texts.isdisjoint(eval_texts)

    for sp in ("search", "select", "train", "exam"):
        sp_rows = [r for r in rows if r["split"] == sp]
        assert sp_rows, f"split {sp} should be non-empty"
        assert {r["label"] for r in sp_rows} == {"positive", "negative"}, f"split {sp} skewed"


def test_clsbench_trj_for_cache_invalidation_and_forbids_untrimmed_fallback(tmp_path: Path):
    hf_dir, tokens, merges = _make_hf_dir(tmp_path / "hf", dialect="qwen3_json", kind="bpe")
    src_gguf = _write_gguf(tmp_path / "src.gguf", "qwen3", tokens, merges=merges)
    jpath = _make_cls_jsonl(tmp_path / "cls.jsonl", n_cal=2, n_each=2)
    cache_dir = tmp_path / "cache"
    env = clsbench.PublicClassifierEnvironment(hf_dir=str(hf_dir), cache_dir=str(cache_dir), jsonl_file=str(jpath))
    tc = runtime.Toolchain(tmp_path, tmp_path / "tk", [""], tokdocs=_make_tokdocs_shim(tmp_path / "bin"))

    u_id, f_id = env.pair
    a_id = env.tok.encode("alpha", add_special_tokens=False)[0]
    b_id = env.tok.encode("beta", add_special_tokens=False)[0]
    cand_gguf = tmp_path / "cand.gguf"
    tasks = env.tasks()[:3]

    # Same model_gguf path overwritten with two different trimmed vocabs -> distinct cache files
    vocabmod.trim_gguf(src_gguf, cand_gguf, sorted(set(range(10)) | {u_id, f_id, a_id}))
    trj_a = env.trj_for(tasks, model_gguf=str(cand_gguf), tc=tc)
    vocabmod.trim_gguf(src_gguf, cand_gguf, sorted(set(range(10)) | {u_id, f_id, b_id}))
    trj_b = env.trj_for(tasks, model_gguf=str(cand_gguf), tc=tc)
    assert trj_a != trj_b

    # Changing answer_tok -> distinct cache file
    trj_ans = env.trj_for(tasks, model_gguf=str(cand_gguf), tc=tc, answer_tok=f_id)
    assert trj_ans != trj_b

    # Changing system_prompt or max_doc_tokens in same cache_dir -> distinct cache files
    env_sys = clsbench.PublicClassifierEnvironment(
        hf_dir=str(hf_dir), cache_dir=str(cache_dir), jsonl_file=str(jpath), system_prompt="Different system prompt."
    )
    env_len = clsbench.PublicClassifierEnvironment(
        hf_dir=str(hf_dir), cache_dir=str(cache_dir), jsonl_file=str(jpath), max_doc_tokens=4
    )
    assert env_sys.trj_for(tasks) != env.trj_for(tasks)
    assert env_len.trj_for(tasks) != env.trj_for(tasks)

    # Trimmed GGUF without tc.tokdocs MUST raise RuntimeError (no untrimmed self.tok fallback)
    with pytest.raises(RuntimeError):
        env.trj_for(tasks, model_gguf=str(cand_gguf), tc=None)

    # Regression #6a: identical meta["tokens"] with different meta["merges"] at same path -> distinct cache paths
    merge_tokens = list(tokens[:-1])
    cand_m = tmp_path / "cand_merges.gguf"
    _write_gguf(cand_m, "qwen3", merge_tokens, merges=[b"a b"])
    trj_m1 = env.trj_for(tasks, model_gguf=str(cand_m), tc=tc)
    _write_gguf(cand_m, "qwen3", merge_tokens, merges=[b"c d"])
    trj_m2 = env.trj_for(tasks, model_gguf=str(cand_m), tc=tc)
    assert trj_m1 != trj_m2

    # Regression #6b: two jsonl_file datasets sharing cache_dir & task IDs with different text -> distinct trj_for & token_counts
    alt_rows = []
    for r_line in jpath.read_text(encoding="utf-8").splitlines():
        row = json.loads(r_line)
        row["text"] = f"{row['text']} alpha beta gamma delta epsilon"
        alt_rows.append(json.dumps(row))
    j_alt = tmp_path / "cls_alt.jsonl"
    j_alt.write_text("\n".join(alt_rows) + "\n", encoding="utf-8")
    env_alt = clsbench.PublicClassifierEnvironment(hf_dir=str(hf_dir), cache_dir=str(cache_dir), jsonl_file=str(j_alt))
    assert env_alt.trj_for(env_alt.tasks()) != env.trj_for(env.tasks())
    assert env_alt.token_counts()["search"]["tokens"] != env.token_counts()["search"]["tokens"]


def test_clsbench_pair_for_model_strict_and_errors(tmp_path: Path):
    hf_dir, tokens, merges = _make_hf_dir(tmp_path / "hf", dialect="qwen3_json", kind="bpe")
    src_gguf = _write_gguf(tmp_path / "src.gguf", "qwen3", tokens, merges=merges)
    jpath = _make_cls_jsonl(tmp_path / "cls.jsonl", n_cal=2, n_each=2)
    env = clsbench.PublicClassifierEnvironment(hf_dir=str(hf_dir), cache_dir=str(tmp_path / "cache"), jsonl_file=str(jpath))
    u_id, f_id = env.pair

    # 1. Exact untrimmed GGUF -> (env.pair, False)
    assert env._pair_for_model(str(src_gguf)) == (env.pair, False)

    # 2. Smaller-vocab GGUF with original bytes at env.pair -> MUST report trimmed=True
    small_tokens = list(tokens[:-2])
    assert len(small_tokens) < len(tokens)
    small_gguf = _write_gguf(tmp_path / "small.gguf", "qwen3", small_tokens, merges=merges)
    assert env._pair_for_model(str(small_gguf)) == (env.pair, True)

    # 3. Substring trap at env.pair (same vocab size!) -> MUST detect mismatch and remap
    trap_tokens = list(tokens)
    trap_tokens[u_id] = b"positive_trap"
    trap_tokens[f_id] = b"negative_trap"
    trap_tokens.extend([b"positive", b"negative"])
    # Make same vocab size by replacing 2 unused tokens
    trap_tokens = trap_tokens[:-2]
    # Put exact b"positive" and b"negative" at unused indices inside vocab
    u_alt = trap_tokens.index(b"alpha")
    f_alt = trap_tokens.index(b"beta")
    trap_tokens[u_alt] = b"positive"
    trap_tokens[f_alt] = b"negative"
    assert len(trap_tokens) == len(tokens)
    trap_gguf = _write_gguf(tmp_path / "trap.gguf", "qwen3", trap_tokens, merges=merges)
    pair_trap, trimmed_trap = env._pair_for_model(str(trap_gguf))
    assert trimmed_trap is True
    assert pair_trap == (u_alt, f_alt)

    # 4. Dropped answer token -> raises RuntimeError (not KeyError)
    dropped_gguf = tmp_path / "dropped.gguf"
    vocabmod.trim_gguf(src_gguf, dropped_gguf, sorted(set(range(10)) | {u_id}))
    with pytest.raises(RuntimeError):
        env._pair_for_model(str(dropped_gguf))

    # 5. Cross-space mismatch (only "Ġnegative" present, exact b"negative" dropped) -> raises RuntimeError
    space_tokens = [b"\xc4\xa0negative" if t == b"negative" else t for t in tokens]
    space_gguf = _write_gguf(tmp_path / "space.gguf", "qwen3", space_tokens, merges=merges)
    with pytest.raises(RuntimeError):
        env._pair_for_model(str(space_gguf))

    # 6. Regression #4 (Metaspace/Unigram): dropped "▁negative" while bare "negative" remains -> RuntimeError; kept -> valid remap
    hf_meta, meta_tokens, _ = _make_hf_dir(
        tmp_path / "hf_meta",
        dialect="qwen3_json",
        kind="unigram",
        extra_words=["number", "with", "in", "search", "select", "exam"],
        metaspace=True,
    )
    src_meta = _write_gguf(tmp_path / "src_meta.gguf", "qwen3", meta_tokens)
    env_meta = clsbench.PublicClassifierEnvironment(hf_dir=str(hf_meta), cache_dir=str(tmp_path / "c_meta"), jsonl_file=str(jpath))
    sp_neg = "▁negative".encode("utf-8")
    sp_pos = "▁positive".encode("utf-8")
    assert env_meta._orig_tok_bytes["negative"] == sp_neg
    assert env_meta._orig_tok_bytes["positive"] == sp_pos
    assert b"negative" in meta_tokens
    dropped_meta_tokens = [b"<dropped>" if t == sp_neg else t for t in meta_tokens]
    dropped_meta_gguf = _write_gguf(tmp_path / "dropped_meta.gguf", "qwen3", dropped_meta_tokens)
    with pytest.raises(RuntimeError):
        env_meta._pair_for_model(str(dropped_meta_gguf))

    keep_meta = sorted({0, 1, 2, meta_tokens.index(sp_pos), meta_tokens.index(sp_neg)})
    valid_meta_gguf = tmp_path / "valid_meta.gguf"
    vocabmod.trim_gguf(src_meta, valid_meta_gguf, keep_meta)
    pair_meta, trimmed_meta = env_meta._pair_for_model(str(valid_meta_gguf))
    assert trimmed_meta is True
    assert pair_meta == (keep_meta.index(meta_tokens.index(sp_pos)), keep_meta.index(meta_tokens.index(sp_neg)))


def test_clsbench_calibrate_proxy_stats_token_counts_offset_score_decide_auroc(tmp_path: Path):
    hf_dir, tokens, merges = _make_hf_dir(tmp_path / "hf", dialect="qwen3_json", kind="bpe")
    src_gguf = _write_gguf(tmp_path / "src.gguf", "qwen3", tokens, merges=merges)
    jpath = _make_cls_jsonl(tmp_path / "cls.jsonl", n_cal=6, n_each=4)
    env = clsbench.PublicClassifierEnvironment(hf_dir=str(hf_dir), cache_dir=str(tmp_path / "cache"), jsonl_file=str(jpath), fpr_target=1.0)

    # Calibrate restricts to _cal_ids and clamps fpr_target=1.0
    scores = {t.id: (float(i) if t.split == "calibration" else -999.0) for i, t in enumerate(env.tasks())}
    rule = env.calibrate(scores)
    assert rule["cutoff"] == pytest.approx(5.0)
    assert rule["n_cal"] == 6

    # Empty or non-finite calibration scores raise ValueError
    with pytest.raises(ValueError):
        env.calibrate({})
    with pytest.raises(ValueError):
        env.calibrate({"cal-0": float("nan"), "cal-1": 1.0})
    with pytest.raises(ValueError):
        env.calibrate({"cal-0": float("inf"), "cal-1": 1.0})

    # Flagged calibration task rejected in both __init__ and calibrate()
    bad_cal_jsonl = _make_cls_jsonl(tmp_path / "bad_cal.jsonl", n_cal=2, n_each=2, cal_label="negative")
    with pytest.raises(ValueError):
        clsbench.PublicClassifierEnvironment(hf_dir=str(hf_dir), cache_dir=str(tmp_path / "c2"), jsonl_file=str(bad_cal_jsonl))

    bad_task = next(t for t in env.tasks() if t.split == "calibration")
    orig_lbl = bad_task.meta["label"]
    bad_task.meta["label"] = env.flagged_word
    try:
        with pytest.raises(ValueError):
            env.calibrate({"cal-0": 1.0, "cal-1": 2.0})
    finally:
        bad_task.meta["label"] = orig_lbl

    # proxy_spec clamps fpr_target=1.0, uses unflagged _cal_ids, rejects non-finite and negative/bool counts
    spec = env.proxy_spec(scores, n_neg=4, n_pos=2)
    assert spec[0]["n"] == 4 and spec[1]["n"] == 2
    assert env.proxy_spec(scores, n_neg=0, n_pos=0) == [
        {"name": "cal_neg", "role": "neg", "n": 0, "ids": []},
        {"name": "search_pos", "role": "pos", "n": 0, "ids": []},
    ]
    with pytest.raises(ValueError):
        env.proxy_spec({**scores, "search-0": float("nan")})
    for bad_kw in ({"n_neg": -1, "n_pos": 2}, {"n_neg": 2, "n_pos": -1}, {"n_neg": True, "n_pos": 1}, {"n_neg": 1, "n_pos": False}):
        with pytest.raises(ValueError):
            env.proxy_spec(scores, **bad_kw)

    # Regression #5: duplicate answers (using all-"positive" JSONL so constructor check is exercised), str answers, and bool dims
    j_all_pos = tmp_path / "all_pos.jsonl"
    j_all_pos.write_text(
        "\n".join(
            json.dumps({"id": f"{sp}-0", "text": f"positive review {sp}", "label": "positive", "split": sp, "family": f"{sp}/0"})
            for sp in ("calibration", "search", "select", "exam")
        ) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        clsbench.PublicClassifierEnvironment(hf_dir=str(hf_dir), cache_dir=str(tmp_path / "c_dup"), jsonl_file=str(j_all_pos), answers=("positive", "positive"))
    with pytest.raises(ValueError):
        clsbench.PublicClassifierEnvironment(hf_dir=str(hf_dir), cache_dir=str(tmp_path / "c_str"), jsonl_file=str(jpath), answers="no")
    for bad_init in (
        {"max_doc_tokens": True},
        {"max_doc_tokens": False},
        {"context": True},
        {"context": False},
        {"max_total_memory_bytes": True},
        {"max_total_memory_bytes": 0},
    ):
        with pytest.raises(ValueError):
            clsbench.PublicClassifierEnvironment(hf_dir=str(hf_dir), cache_dir=str(tmp_path / "c_bad"), jsonl_file=str(jpath), **bad_init)
    with pytest.raises(ValueError):
        clsbench._load_public_cls(tmp_path / "c_load", "dummy/yelp", max_cal=True, max_eval=0)
    with pytest.raises(ValueError):
        clsbench._load_public_cls(tmp_path / "c_load", "dummy/yelp", max_cal=0, max_eval=False)

    # fit_token_counts works on untrimmed GGUF without tokdocs and raises on trimmed without tokdocs
    cnt_src = env.fit_token_counts(str(src_gguf), tc=None)
    assert cnt_src.shape == (len(tokens),) and cnt_src.sum() > 0
    trimmed_gguf = tmp_path / "trim.gguf"
    vocabmod.trim_gguf(src_gguf, trimmed_gguf, sorted(set(range(10)) | set(env.pair)))
    with pytest.raises(RuntimeError):
        env.fit_token_counts(str(trimmed_gguf), tc=None)

    # stats_texts stratified across positive/negative in search split
    st = env.stats_texts(4)
    assert len(st) == 4
    assert any("negative" in s for s in st) and any("positive" in s for s in st)

    # token_counts() matches plan._token_stats schema
    tc_dict = env.token_counts()
    assert set(tc_dict) == {"calibration", "search", "select", "exam"}
    for d in tc_dict.values():
        assert d["docs"] > 0 and d["tokens"] > d["docs"]

    # Offset-mapping truncation preserves exact spacing in kept prefix without decode-roundtrip drift
    long_j = tmp_path / "long.jsonl"
    spaced = "alpha   beta   gamma   delta   epsilon   zeta"
    long_j.write_text(json.dumps({"id": "s0", "text": spaced, "label": "positive", "split": "search", "family": "s/0"}) + "\n")
    env_trunc = clsbench.PublicClassifierEnvironment(hf_dir=str(hf_dir), cache_dir=str(tmp_path / "ct"), jsonl_file=str(long_j), max_doc_tokens=5)
    t0 = env_trunc.tasks()[0]
    assert t0.meta["truncated"] is True
    assert spaced.startswith(t0.prompt["text"])
    assert "alpha   beta" in t0.prompt["text"]

    # score + decide validate finiteness
    cfg_p = tmp_path / "scores.json"
    cfg_p.write_text(json.dumps({"by_model": {"src.gguf": {"values": [float("nan")]}}}))
    tc = runtime.Toolchain(tmp_path, tmp_path / "tk", [""], logodds=_make_logodds_shim(tmp_path / "bin", cfg_p))
    scorer = runtime.BatchScorer(tc, slots_per_gpu=1)
    with pytest.raises(RuntimeError):
        env.score(env.tasks()[:1], str(src_gguf), scorer)

    with pytest.raises(ValueError):
        env.decide(env.tasks()[0], float("nan"), {"cutoff": 0.0})


def test_clsbench_score_decide_compare_pauc_auroc_e2e(tmp_path: Path):
    hf_dir, tokens, merges = _make_hf_dir(tmp_path / "hf", dialect="qwen3_json", kind="bpe")
    src_gguf = _write_gguf(tmp_path / "src.gguf", "qwen3", tokens, merges=merges)
    jpath = _make_cls_jsonl(tmp_path / "cls.jsonl", n_cal=10, n_each=6)
    env = clsbench.PublicClassifierEnvironment(hf_dir=str(hf_dir), cache_dir=str(tmp_path / "cache"), jsonl_file=str(jpath), fpr_target=0.1)

    drop_toks = {tokens.index(b"ab"), tokens.index(b"abc"), tokens.index(b"theta"), tokens.index(b"zeta")}
    keep = [i for i in range(len(tokens)) if i not in drop_toks]
    trim_gguf = tmp_path / "trim.gguf"
    vocabmod.trim_gguf(src_gguf, trim_gguf, keep)
    pair_trim, is_trimmed = env._pair_for_model(str(trim_gguf))
    assert is_trimmed is True and pair_trim != env.pair

    cfg_p = tmp_path / "vocab_decode.json"
    cfg_p.write_text(
        json.dumps({
            "by_model": {
                "src.gguf": {
                    "mode": "vocab_decode",
                    "expected_pair_bytes": ["positive", "negative"],
                    "flagged_substrs": ["negative"],
                    "cal_substrs": ["good service"],
                    "unflagged_substrs": ["positive"],
                    "flagged_score": -2.0,
                    "cal_score": 1.5,
                    "unflagged_score": 2.0,
                },
                "trim.gguf": {
                    "mode": "vocab_decode",
                    "expected_pair_bytes": ["positive", "negative"],
                    "flagged_substrs": ["negative"],
                    "cal_substrs": ["good service"],
                    "unflagged_substrs": ["positive"],
                    "flagged_score": -2.5,
                    "cal_score": 1.6,
                    "unflagged_score": 2.3,
                },
            }
        }),
        encoding="utf-8",
    )
    bin_dir = tmp_path / "bin"
    tc = runtime.Toolchain(tmp_path, tmp_path / "tk", [""], tokdocs=_make_tokdocs_shim(bin_dir), logodds=_make_logodds_shim(bin_dir, cfg_p))
    scorer = runtime.BatchScorer(tc, slots_per_gpu=1)

    tasks = env.tasks()
    sc_src = env.score(tasks, str(src_gguf), scorer)
    sc_trim = env.score(tasks, str(trim_gguf), scorer)
    assert set(sc_src) == {t.id for t in tasks} and set(sc_trim) == {t.id for t in tasks}

    rule_src = env.calibrate(sc_src)
    rule_trim = env.calibrate(sc_trim)
    eval_tasks = [t for t in tasks if t.split != "calibration"]
    for t in eval_tasks:
        assert env.decide(t, sc_src[t.id], rule_src).success is True
        assert env.decide(t, sc_trim[t.id], rule_trim).success is True

    y_flag = np.array([t.meta["label"] == env.flagged_word for t in eval_tasks], dtype=bool)
    vec_src = np.array([sc_src[t.id] for t in eval_tasks], dtype=float)
    vec_trim = np.array([sc_trim[t.id] for t in eval_tasks], dtype=float)
    assert anneal.auc(vec_src, y_flag) == pytest.approx(1.0)
    assert anneal.auc(vec_trim, y_flag) == pytest.approx(1.0)

    cal_ids = [t.id for t in tasks if t.split == "calibration"]
    labels = {t.id: ("pos" if t.meta["label"] == env.flagged_word else "neg") for t in eval_tasks}
    kinds = {t.id: t.split for t in eval_tasks}
    cmp_res = compare.compare(
        models={
            "src": {"cal": [sc_src[i] for i in cal_ids], "scores": {t.id: sc_src[t.id] for t in eval_tasks}},
            "trim": {"cal": [sc_trim[i] for i in cal_ids], "scores": {t.id: sc_trim[t.id] for t in eval_tasks}},
        },
        labels=labels,
        kinds=kinds,
        boot=30,
        seed=0,
    )
    for m in ("src", "trim"):
        assert cmp_res["models"][m]["pos"]["caught"] == pytest.approx(1.0)
        assert cmp_res["models"][m]["neg_flagged"] == pytest.approx(0.0)

    spec = env.proxy_spec(sc_src, n_neg=6, n_pos=3)
    src_parts = {
        "neg": np.array([sc_src[i] for i in spec[0]["ids"]], dtype=float),
        "pos": np.array([sc_src[i] for i in spec[1]["ids"]], dtype=float),
    }
    trim_parts = {
        "neg": np.array([sc_trim[i] for i in spec[0]["ids"]], dtype=float),
        "pos": np.array([sc_trim[i] for i in spec[1]["ids"]], dtype=float),
    }
    dmg = objectives.pauc_damage(trim_parts, src_parts, tail=0.5, tau=0.5)
    assert math.isfinite(dmg) and dmg < 0.0


# ---------------------------------------------------------------------------
# 2. detect regression tests
# ---------------------------------------------------------------------------


def test_detect_inkling2_dialect_auto_and_explicit_and_reasoning_effort(tmp_path: Path):
    hf_dir, _, _ = _make_hf_dir(tmp_path / "hf_ink", dialect="inkling2", kind="wordlevel")
    data_dir, cal_ids = _make_detect_data(tmp_path / "data", n_cal=4)

    for ps in ("auto", "inkling2"):
        env = detect.DetectEnvironment(
            data_dir=str(data_dir),
            hf_dir=str(hf_dir),
            classifier=True,
            calibration_ids_file=str(cal_ids),
            cache_dir=str(tmp_path / f"c_{ps}"),
            search_per_cell=1,
            select_per_cell=1,
            exam_test_per_cell=1,
            search_beemo_per_kind=1,
            select_beemo_per_kind=1,
            exam_beemo_per_kind=1,
            prompt_style=ps,
        )
        assert env.prompt_style == "inkling2"
        assert env.dialect == "inkling2"
        assert env.needs_special_tokens is True
        assert env.reasoning_effort == "none"
        rendered = env.render_prompt("hello world")
        assert "Thinking effort level: 0<|end_message|>" in rendered
        assert rendered.endswith("<|message_model|><|content_text|>")

    env_low = detect.DetectEnvironment(
        data_dir=str(data_dir),
        hf_dir=str(hf_dir),
        cutoff=0.5,
        classifier=False,
        search_per_cell=1,
        select_per_cell=1,
        exam_test_per_cell=1,
        exam_beemo_per_kind=1,
        prompt_style="inkling2",
        reasoning_effort="low",
    )
    assert "Thinking effort level: 0.2<|end_message|>" in env_low.render_prompt("hello world")


def test_detect_model_vocab_plan_corrupt_gguf_and_pruned_hf_encode(tmp_path: Path):
    data_dir, cal_ids = _make_detect_data(tmp_path / "data", n_cal=4)

    # WordLevel trimmed GGUF without tokdocs -> _pruned_hf_encode trims vocab cleanly (< trimmed_v)
    hf_wl, tokens_wl, _ = _make_hf_dir(tmp_path / "hf_wl", dialect="qwen3_json", kind="wordlevel")
    src_wl = _write_gguf(tmp_path / "src_wl.gguf", "qwen3", tokens_wl)
    env_wl = detect.DetectEnvironment(
        data_dir=str(data_dir), hf_dir=str(hf_wl), classifier=True,
        calibration_ids_file=str(cal_ids), cache_dir=str(tmp_path / "c_wl"),
        search_per_cell=1, select_per_cell=1, exam_test_per_cell=1,
        search_beemo_per_kind=1, select_beemo_per_kind=1, exam_beemo_per_kind=1,
    )

    # Corrupt existing GGUF raises instead of silently returning untrimmed pair
    corrupt = tmp_path / "corrupt.gguf"
    corrupt.write_bytes(b"NOT_A_GGUF_FILE")
    with pytest.raises((OSError, ValueError, RuntimeError, struct.error)):
        env_wl._model_vocab_plan(str(corrupt))

    # Regression #2: nonexistent GGUF raises FileNotFoundError (no silent untrimmed fallback)
    with pytest.raises(FileNotFoundError):
        env_wl._model_vocab_plan(str(tmp_path / "missing.gguf"))

    # Same vocab_size but swapped answer token bytes -> must NOT treat as untrimmed
    h_id, a_id = env_wl.pair
    swapped_tokens = list(tokens_wl)
    swapped_tokens[h_id], swapped_tokens[a_id] = swapped_tokens[a_id], swapped_tokens[h_id]
    swapped_gguf = _write_gguf(tmp_path / "swapped.gguf", "qwen3", swapped_tokens)
    remapped_pair, vh_swap, _ = env_wl._model_vocab_plan(str(swapped_gguf))
    assert vh_swap is not None
    assert remapped_pair == (a_id, h_id)

    # Trimmed WordLevel without tokdocs produces valid < trimmed_v token IDs
    keep_wl = sorted(set(range(6)) | {h_id, a_id})
    trim_wl = tmp_path / "trim_wl.gguf"
    vocabmod.trim_gguf(src_wl, trim_wl, keep_wl)
    _, vh_wl, meta_wl = env_wl._model_vocab_plan(str(trim_wl))
    assert vh_wl is not None
    seqs = env_wl._pruned_hf_encode(vh_wl, meta_wl, ["hello world human ai"])
    assert all(0 <= tid < len(keep_wl) for tid in seqs[0])

    # Regression #1 (WordLevel/BPE/Unigram/Missing): fit_token_counts on trimmed GGUF without tokdocs
    cnt_wl = env_wl.fit_token_counts(str(trim_wl), tc=None)
    assert cnt_wl.shape == (len(keep_wl),) and int(cnt_wl.sum()) > 0
    with pytest.raises(FileNotFoundError):
        env_wl.fit_token_counts(str(tmp_path / "missing_fit.gguf"), tc=None)

    hf_bpe, tokens_bpe, merges_bpe = _make_hf_dir(tmp_path / "hf_bpe_fit", dialect="qwen3_json", kind="bpe")
    src_bpe = _write_gguf(tmp_path / "src_bpe_fit.gguf", "qwen3", tokens_bpe, merges=merges_bpe)
    env_bpe = detect.DetectEnvironment(
        data_dir=str(data_dir), hf_dir=str(hf_bpe), classifier=True,
        calibration_ids_file=str(cal_ids), cache_dir=str(tmp_path / "c_bpe_fit"),
        search_per_cell=1, select_per_cell=1, exam_test_per_cell=1,
        search_beemo_per_kind=1, select_beemo_per_kind=1, exam_beemo_per_kind=1,
    )
    drop_bpe = {tokens_bpe.index(b"ab"), tokens_bpe.index(b"abc"), tokens_bpe.index(b"theta"), tokens_bpe.index(b"zeta")}
    keep_bpe = [i for i in range(len(tokens_bpe)) if i not in drop_bpe]
    trim_bpe = tmp_path / "trim_bpe_fit.gguf"
    vocabmod.trim_gguf(src_bpe, trim_bpe, keep_bpe)
    cnt_bpe = env_bpe.fit_token_counts(str(trim_bpe), tc=None)
    assert cnt_bpe.shape == (len(keep_bpe),) and int(cnt_bpe.sum()) > 0

    # Regression #6 (detect): identical meta["tokens"] with different meta["merges"] -> distinct trj_for_model paths
    merge_toks = list(tokens_bpe[:-1])
    cand_det_m = tmp_path / "cand_det_merges.gguf"
    _write_gguf(cand_det_m, "qwen3", merge_toks, merges=[b"a b"])
    trj_dm1, _ = env_bpe.trj_for_model(env_bpe.tasks()[:2], str(cand_det_m), tc=None)
    _write_gguf(cand_det_m, "qwen3", merge_toks, merges=[b"a b", b"ab c"])
    trj_dm2, _ = env_bpe.trj_for_model(env_bpe.tasks()[:2], str(cand_det_m), tc=None)
    assert trj_dm1 != trj_dm2

    # Unigram trimmed GGUF without tokdocs raises RuntimeError cleanly in both _pruned_hf_encode and fit_token_counts
    hf_uni, tokens_uni, _ = _make_hf_dir(tmp_path / "hf_uni", dialect="qwen3_json", kind="unigram")
    src_uni = _write_gguf(tmp_path / "src_uni.gguf", "qwen3", tokens_uni)
    env_uni = detect.DetectEnvironment(
        data_dir=str(data_dir), hf_dir=str(hf_uni), classifier=True,
        calibration_ids_file=str(cal_ids), cache_dir=str(tmp_path / "c_uni"),
        search_per_cell=1, select_per_cell=1, exam_test_per_cell=1,
        search_beemo_per_kind=1, select_beemo_per_kind=1, exam_beemo_per_kind=1,
    )
    keep_uni = sorted(set(range(6)) | set(env_uni.pair))
    trim_uni = tmp_path / "trim_uni.gguf"
    vocabmod.trim_gguf(src_uni, trim_uni, keep_uni)
    _, vh_uni, meta_uni = env_uni._model_vocab_plan(str(trim_uni))
    with pytest.raises(RuntimeError):
        env_uni._pruned_hf_encode(vh_uni, meta_uni, ["hello world"])
    with pytest.raises(RuntimeError):
        env_uni.fit_token_counts(str(trim_uni), tc=None)

    # Regression #4 (detect Metaspace/Unigram): dropped "▁human" while bare "human" exists -> RuntimeError; kept -> valid remap
    hf_dmeta, dmeta_tokens, _ = _make_hf_dir(
        tmp_path / "hf_dmeta", dialect="qwen3_json", kind="unigram", metaspace=True
    )
    src_dmeta = _write_gguf(tmp_path / "src_dmeta.gguf", "qwen3", dmeta_tokens)
    env_dmeta = detect.DetectEnvironment(
        data_dir=str(data_dir), hf_dir=str(hf_dmeta), classifier=True,
        calibration_ids_file=str(cal_ids), cache_dir=str(tmp_path / "c_dmeta"),
        search_per_cell=1, select_per_cell=1, exam_test_per_cell=1,
        search_beemo_per_kind=1, select_beemo_per_kind=1, exam_beemo_per_kind=1,
    )
    sp_hum = "▁human".encode("utf-8")
    sp_ai = "▁ai".encode("utf-8")
    assert b"human" in dmeta_tokens
    dropped_dmeta_toks = [b"<dropped>" if t == sp_hum else t for t in dmeta_tokens]
    dropped_dmeta_gguf = _write_gguf(tmp_path / "dropped_dmeta.gguf", "qwen3", dropped_dmeta_toks)
    with pytest.raises(RuntimeError):
        env_dmeta._model_vocab_plan(str(dropped_dmeta_gguf))
    keep_dmeta = sorted({0, 1, 2, dmeta_tokens.index(sp_hum), dmeta_tokens.index(sp_ai)})
    valid_dmeta_gguf = tmp_path / "valid_dmeta.gguf"
    vocabmod.trim_gguf(src_dmeta, valid_dmeta_gguf, keep_dmeta)
    pair_dm, vh_dm, _ = env_dmeta._model_vocab_plan(str(valid_dmeta_gguf))
    assert vh_dm is not None
    assert pair_dm == (keep_dmeta.index(dmeta_tokens.index(sp_hum)), keep_dmeta.index(dmeta_tokens.index(sp_ai)))


def test_detect_classifier_tasks_calibrate_decide_and_empty_fit_token_counts(tmp_path: Path):
    hf_dir, tokens, _ = _make_hf_dir(tmp_path / "hf", dialect="qwen3_json", kind="wordlevel")

    # Non-human calibration row rejected
    bad_dir, bad_cal = _make_detect_data(tmp_path / "bad_data", n_cal=2, cal_labels=["human", "ai"])
    with pytest.raises(ValueError, match="human"):
        detect.DetectEnvironment(
            data_dir=str(bad_dir), hf_dir=str(hf_dir), classifier=True,
            calibration_ids_file=str(bad_cal), cache_dir=str(tmp_path / "cb"),
            search_per_cell=1, select_per_cell=1, exam_test_per_cell=1,
            search_beemo_per_kind=1, select_beemo_per_kind=1, exam_beemo_per_kind=1,
        )

    # Empty calibration IDs rejected
    empty_dir, empty_cal = _make_detect_data(tmp_path / "empty_data", n_cal=2)
    empty_cal.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="empty"):
        detect.DetectEnvironment(
            data_dir=str(empty_dir), hf_dir=str(hf_dir), classifier=True,
            calibration_ids_file=str(empty_cal), cache_dir=str(tmp_path / "ce"),
            search_per_cell=1, select_per_cell=1, exam_test_per_cell=1,
            search_beemo_per_kind=1, select_beemo_per_kind=1, exam_beemo_per_kind=1,
        )

    good_dir, good_cal = _make_detect_data(tmp_path / "good_data", n_cal=3)
    env = detect.DetectEnvironment(
        data_dir=str(good_dir), hf_dir=str(hf_dir), classifier=True,
        calibration_ids_file=str(good_cal), cache_dir=str(tmp_path / "cg"),
        search_per_cell=1, select_per_cell=1, exam_test_per_cell=1,
        search_beemo_per_kind=1, select_beemo_per_kind=1, exam_beemo_per_kind=1,
    )
    with pytest.raises(ValueError):
        env.calibrate({})
    with pytest.raises(ValueError):
        env.calibrate({"wo-cal:cal-0": float("nan")})
    with pytest.raises(ValueError):
        env.calibrate({"wo-cal:cal-0": float("-inf")})
    with pytest.raises(ValueError):
        env.decide(env.tasks()[0], float("nan"), {"cutoff": 0.0})

    # Regression #3: detect.proxy_spec rejects NaN/inf search "ai" scores and negative/bool n_neg/n_pos
    valid_scores = {t.id: (1.0 if t.split == "calibration" else -1.0) for t in env.tasks()}
    ai_t = next(t for t in env.tasks() if t.split == "search" and t.meta["label"] == "ai")
    for bad_sc in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError):
            env.proxy_spec({**valid_scores, ai_t.id: bad_sc}, n_neg=2, n_pos=1)
    for bad_kw in ({"n_neg": -1, "n_pos": 1}, {"n_neg": 1, "n_pos": -1}, {"n_neg": True, "n_pos": 1}, {"n_neg": 1, "n_pos": False}):
        with pytest.raises(ValueError):
            env.proxy_spec(valid_scores, **bad_kw)

    # Regression #6 (detect): shared cache_dir & identical task IDs with different texts -> distinct trj_for & token_counts
    alt_dir = tmp_path / "alt_data"
    for p_rel in ("WildOutlines/data/validation-00000-of-00001.parquet", "WildOutlines/data/test-00000-of-00001.parquet", "WildOutlines/data/calibration-00000-of-00001.parquet"):
        dst = alt_dir / p_rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        rows = pq.read_table(good_dir / p_rel).to_pylist()
        for r in rows:
            r["text"] = f"{r['text']} extra alpha beta gamma delta"
        pq.write_table(pa.Table.from_pylist(rows), dst)
    be_dst = alt_dir / "beemo/data/train-00000-of-00001.parquet"
    be_dst.parent.mkdir(parents=True, exist_ok=True)
    be_rows = pq.read_table(good_dir / "beemo/data/train-00000-of-00001.parquet").to_pylist()
    for r in be_rows:
        r["human_output"] = f"{r['human_output']} extra alpha beta"
        r["model_output"] = f"{r['model_output']} extra alpha beta"
        r["human_edits"] = f"{r['human_edits']} extra alpha beta"
    pq.write_table(pa.Table.from_pylist(be_rows), be_dst)
    env_alt = detect.DetectEnvironment(
        data_dir=str(alt_dir), hf_dir=str(hf_dir), classifier=True,
        calibration_ids_file=str(good_cal), cache_dir=str(tmp_path / "cg"),
        search_per_cell=1, select_per_cell=1, exam_test_per_cell=1,
        search_beemo_per_kind=1, select_beemo_per_kind=1, exam_beemo_per_kind=1,
    )
    assert env_alt.trj_for(env_alt.tasks()) != env.trj_for(env.tasks())
    assert env_alt.token_counts()["search"]["tokens"] != env.token_counts()["search"]["tokens"]

    # Empty fit_tasks handled cleanly
    env._tasks = []
    counts = env.fit_token_counts()
    assert counts.shape == (len(tokens),) and int(counts.sum()) == 0


def test_detect_score_trj_calibrate_decide_proxy_stats_token_counts_e2e(tmp_path: Path):
    hf_dir, tokens, merges = _make_hf_dir(tmp_path / "hf_det_e2e", dialect="qwen3_json", kind="bpe")
    src_gguf = _write_gguf(tmp_path / "src_det.gguf", "qwen3", tokens, merges=merges)
    data_dir, cal_ids = _make_detect_data(tmp_path / "data_e2e", n_cal=8)
    env = detect.DetectEnvironment(
        data_dir=str(data_dir),
        hf_dir=str(hf_dir),
        classifier=True,
        calibration_ids_file=str(cal_ids),
        cache_dir=str(tmp_path / "c_det_e2e"),
        search_per_cell=2,
        select_per_cell=2,
        exam_test_per_cell=2,
        search_beemo_per_kind=2,
        select_beemo_per_kind=2,
        exam_beemo_per_kind=2,
    )

    drop_toks = {tokens.index(b"ab"), tokens.index(b"abc"), tokens.index(b"theta"), tokens.index(b"zeta")}
    keep = [i for i in range(len(tokens)) if i not in drop_toks]
    trim_bpe = tmp_path / "trim_det.gguf"
    vocabmod.trim_gguf(src_gguf, trim_bpe, keep)

    tasks = env.tasks()
    trj_trim, pair_trim = env.trj_for_model(tasks, str(trim_bpe), tc=None)
    assert trj_trim.exists() and pair_trim != env.pair

    cfg_p = tmp_path / "det_vocab_decode.json"
    cfg_p.write_text(
        json.dumps({
            "by_model": {
                "src_det.gguf": {
                    "mode": "vocab_decode",
                    "expected_pair_bytes": ["human", "ai"],
                    "flagged_substrs": [" ai ", "edited"],
                    "cal_substrs": ["hello world"],
                    "unflagged_substrs": ["human"],
                    "flagged_score": -2.0,
                    "cal_score": 1.5,
                    "unflagged_score": 2.0,
                },
                "trim_det.gguf": {
                    "mode": "vocab_decode",
                    "expected_pair_bytes": ["human", "ai"],
                    "flagged_substrs": [" ai ", "edited"],
                    "cal_substrs": ["hello world"],
                    "unflagged_substrs": ["human"],
                    "flagged_score": -0.6,
                    "cal_score": 0.8,
                    "unflagged_score": 1.3,
                },
            }
        }),
        encoding="utf-8",
    )
    tc = runtime.Toolchain(tmp_path, tmp_path / "tk", [""], logodds=_make_logodds_shim(tmp_path / "bin", cfg_p))
    scorer = runtime.BatchScorer(tc, slots_per_gpu=1)

    sc_src = env.score(tasks, str(src_gguf), scorer)
    sc_trim = env.score(tasks, str(trim_bpe), scorer)
    assert set(sc_src) == {t.id for t in tasks} and set(sc_trim) == {t.id for t in tasks}

    rule_src = env.calibrate(sc_src)
    rule_trim = env.calibrate(sc_trim)
    eval_tasks = [t for t in tasks if t.split != "calibration"]
    for t in eval_tasks:
        assert env.decide(t, sc_src[t.id], rule_src).success is True
        assert env.decide(t, sc_trim[t.id], rule_trim).success is True

    y_flag = np.array([t.meta["label"] == "ai" for t in eval_tasks], dtype=bool)
    vec_src = np.array([sc_src[t.id] for t in eval_tasks], dtype=float)
    vec_trim = np.array([sc_trim[t.id] for t in eval_tasks], dtype=float)
    assert anneal.auc(vec_src, y_flag) == pytest.approx(1.0)
    assert anneal.auc(vec_trim, y_flag) == pytest.approx(1.0)

    cal_ids_list = [t.id for t in tasks if t.split == "calibration"]
    labels = {t.id: ("pos" if t.meta["label"] == "ai" else "neg") for t in eval_tasks}
    kinds = {t.id: t.meta["kind"] for t in eval_tasks}
    cmp_res = compare.compare(
        models={
            "src": {"cal": [sc_src[i] for i in cal_ids_list], "scores": {t.id: sc_src[t.id] for t in eval_tasks}},
            "trim": {"cal": [sc_trim[i] for i in cal_ids_list], "scores": {t.id: sc_trim[t.id] for t in eval_tasks}},
        },
        labels=labels,
        kinds=kinds,
        boot=30,
        seed=0,
    )
    for m in ("src", "trim"):
        assert cmp_res["models"][m]["pos"]["caught"] == pytest.approx(1.0)
        assert cmp_res["models"][m]["neg_flagged"] == pytest.approx(0.0)

    spec = env.proxy_spec(sc_src, n_neg=4, n_pos=2)
    assert spec[0]["n"] == 4 and spec[1]["n"] == 2
    src_parts = {
        "neg": np.array([sc_src[i] for i in spec[0]["ids"]], dtype=float),
        "pos": np.array([sc_src[i] for i in spec[1]["ids"]], dtype=float),
    }
    trim_parts = {
        "neg": np.array([sc_trim[i] for i in spec[0]["ids"]], dtype=float),
        "pos": np.array([sc_trim[i] for i in spec[1]["ids"]], dtype=float),
    }
    dmg = objectives.pauc_damage(trim_parts, src_parts, tail=0.5, tau=0.5)
    assert math.isfinite(dmg) and dmg > 0.0

    st = env.stats_texts(4)
    assert len(st) == 4 and all(isinstance(s, str) and "<|im_start|>user" in s for s in st)
    assert env.stats_texts(0) == []

    tc_dict = env.token_counts()
    assert set(tc_dict) == {"calibration", "search", "select", "exam"}
    for d in tc_dict.values():
        assert d["docs"] > 0 and d["tokens"] > d["docs"]


def test_detect_non_classifier_run_bos_decoding_cache_prompt_and_retries(tmp_path: Path, monkeypatch):
    hf_dir, _, _ = _make_hf_dir(tmp_path / "hf_bos", dialect="qwen3_json", kind="wordlevel", bos_token="<s>")
    data_dir, _ = _make_detect_data(tmp_path / "data", n_cal=2)
    env = detect.DetectEnvironment(
        data_dir=str(data_dir), hf_dir=str(hf_dir), cutoff=0.5, classifier=False,
        search_per_cell=1, select_per_cell=1, exam_test_per_cell=1, exam_beemo_per_kind=1,
        prompt_style="chat_template",
    )
    # Prepend BOS in render_prompt to verify run() strips it
    orig_render = env.render_prompt
    monkeypatch.setattr(env, "render_prompt", lambda txt: "<s>" + orig_render(txt))
    monkeypatch.setattr(time, "sleep", lambda _: None)

    attempts = {"n": 0}

    def responder(path, body):
        attempts["n"] += 1
        if attempts["n"] == 1:
            return None  # RemoteDisconnected -> http.client.HTTPException
        if attempts["n"] == 2:
            return 500, {"error": "transient"}
        return 200, {"completion_probabilities": [{"top_probs": [{"token": "human", "prob": 0.1}, {"token": "ai", "prob": 0.9}]}]}

    task = next(t for t in env.tasks() if t.meta["label"] == "ai")
    with _LoopbackServer(responder) as srv:
        cl = client_mod.ChatClient(
            srv.url,
            {"temperature": 0.0, "repeat_penalty": 1.15, "seed": 999, "max_tokens": 999, "chat_template_kwargs": {"x": 1}},
            timeout=5.0,
            retries=3,
        )
        ep = env.run(task, cl, seed=7)
        assert ep.infra_error is None
        assert env.verify(task, ep).tests_passed is True
        body = srv.records[-1]["body"]
        assert not body["prompt"].startswith("<s>")
        assert body["temperature"] == 1.0
        assert body["top_k"] == 0
        assert body["top_p"] == 1.0
        assert body["min_p"] == 0.0
        assert body["repeat_penalty"] == pytest.approx(1.15)
        assert body["seed"] == 7
        assert body["cache_prompt"] is False

        # Respects explicit client.cache_prompt=True
        attempts["n"] = 2
        cl_cp = client_mod.ChatClient(srv.url, {"temperature": 0.0}, timeout=5.0, retries=1, cache_prompt=True)
        ep2 = env.run(task, cl_cp, seed=9)
        assert ep2.infra_error is None
        assert srv.records[-1]["body"]["cache_prompt"] is True


# ---------------------------------------------------------------------------
# 3. protocol regression tests
# ---------------------------------------------------------------------------


def test_protocol_validate_environment_classifier_and_budget():
    base = [
        protocol.Task("s0", "f_s", "search", {}),
        protocol.Task("l0", "f_l", "select", {}),
        protocol.Task("e0", "f_e", "exam", {}),
    ]

    class GoodGen:
        name = "g"
        version = "1"
        budget = {"max_turns": 1, "max_tokens_per_turn": 16}
        deployment = {"context": 512, "concurrency": 1, "kv_type": "f16"}
        def tasks(self): return list(base)
        def run(self, t, c, s): ...
        def verify(self, t, e): ...

    protocol.validate_environment(GoodGen())

    class GoodCls(GoodGen):
        classifier = True
        pair = (1, 2)
        def score(self, ts, m, sc): return {}
        def calibrate(self, sc): return {}
        def decide(self, t, s, r): ...
        def tasks(self): return base + [protocol.Task("c0", "f_c", "calibration", {})]

    protocol.validate_environment(GoodCls())

    class GoodClsDocBudget(GoodCls):
        budget = {"max_tokens": 1, "max_doc_tokens": 512}

    protocol.validate_environment(GoodClsDocBudget())

    class EmptyCalEnv:
        name = "e"
        version = "1"
        classifier = True
        pair = (1, 2)
        budget = {}
        deployment = {"context": 512, "concurrency": 1, "kv_type": "f16"}
        def tasks(self): return list(base)
        def run(self, t, c, s): ...
        def verify(self, t, e): ...
        def score(self, ts, m, sc): return {}
        def calibrate(self, sc): return {}
        def decide(self, t, s, r): ...

    with pytest.raises(ValueError, match="calibration"):
        protocol.validate_environment(EmptyCalEnv())

    for bad_pair in (None, (1,), (1, 1), (-1, 2), (True, 1)):
        class BadPair(GoodCls):
            pair = bad_pair
        with pytest.raises(ValueError, match="pair"):
            protocol.validate_environment(BadPair())

    for bad_budget in (None, {"max_turns": 0}, {"max_tokens_per_turn": -5}, {"max_tokens": True}):
        class BadBudget(GoodGen):
            budget = bad_budget
        with pytest.raises(ValueError):
            protocol.validate_environment(BadBudget())

    # Regression #8: single-pass generator tasks() & deployment["max_total_memory_bytes"] validation
    class GenTasksEnv(GoodGen):
        def tasks(self):
            return (t for t in base)

    protocol.validate_environment(GenTasksEnv())

    class GenClsTasksEnv(GoodCls):
        def tasks(self):
            return (t for t in base + [protocol.Task("c0", "f_c", "calibration", {})])

    protocol.validate_environment(GenClsTasksEnv())

    class EmptyGenEnv(GoodGen):
        def tasks(self):
            return (t for t in ())

    with pytest.raises(ValueError, match="no tasks"):
        protocol.validate_environment(EmptyGenEnv())

    class MemNoneEnv(GoodGen):
        deployment = {"context": 512, "concurrency": 1, "kv_type": "f16", "max_total_memory_bytes": None}

    class MemOkEnv(GoodGen):
        deployment = {"context": 512, "concurrency": 1, "kv_type": "f16", "max_total_memory_bytes": 1024}

    protocol.validate_environment(MemNoneEnv())
    protocol.validate_environment(MemOkEnv())

    for bad_mem in (0, -1, True, False, "4GB"):
        class BadMemEnv(GoodGen):
            deployment = {"context": 512, "concurrency": 1, "kv_type": "f16", "max_total_memory_bytes": bad_mem}
        with pytest.raises(ValueError, match="max_total_memory_bytes"):
            protocol.validate_environment(BadMemEnv())


# ---------------------------------------------------------------------------
# 4. compare regression tests
# ---------------------------------------------------------------------------


def test_compare_cutoff_bounds_rate_masking_and_polarity():
    assert compare.cutoff([0.9, 0.1, 0.5], rate=1.0) == pytest.approx(0.9)
    assert compare.cutoff([0.9, 0.1, 0.5], rate=0.0) == pytest.approx(0.1)

    for bad_rate in (-0.1, 1.1, float("nan"), float("inf"), True, False):
        with pytest.raises(ValueError):
            compare.cutoff([0.1, 0.5], rate=bad_rate)

    for bad_cal in ([], [1.0, float("nan")], [float("inf")]):
        with pytest.raises(ValueError):
            compare.cutoff(bad_cal)

    # _rate excludes bootstrap draws where sb.sum(1) == 0 instead of injecting 0.0
    flag = np.array([True] + [False] * 199)
    sel = np.array([True] + [False] * 199)
    rng = np.random.default_rng(0)
    boot = rng.integers(0, 200, (500, 200))
    # Ensure some draws actually have sb.sum(1) == 0
    assert (sel[boot].sum(1) == 0).any()
    r = compare._rate(flag, sel, boot)
    assert r["caught"] == pytest.approx(1.0)
    assert r["ci"][0] == pytest.approx(1.0)
    assert r["ci"][1] == pytest.approx(1.0)

    # Polarity: score < cutoff flags pos
    res = compare.compare(
        models={"m": {"cal": [0.0, 1.0, 2.0], "scores": {"p0": -1.0, "n0": 2.0}}},
        labels={"p0": "pos", "n0": "neg"},
        kinds={"p0": "k", "n0": "k"},
        boot=20,
        seed=0,
    )
    assert res["models"]["m"]["pos"]["caught"] == pytest.approx(1.0)
    assert res["models"]["m"]["neg_flagged"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 5. sanity regression tests
# ---------------------------------------------------------------------------


def test_sanity_infer_dialect_precedence_and_run_behavioral_cache_prompt(tmp_path: Path):
    _, q_tokens, _ = _make_hf_dir(tmp_path / "hf_q", dialect="qwen3_json", kind="wordlevel")
    q_gguf = _write_gguf(tmp_path / "qwen.gguf", "qwen3", q_tokens)
    _, i_tokens, _ = _make_hf_dir(tmp_path / "hf_i", dialect="inkling2", kind="wordlevel")
    i_gguf = _write_gguf(tmp_path / "ink.gguf", "inkling2", i_tokens)

    # Real GGUF model inspected BEFORE s.special
    assert sanity._infer_dialect([SimpleNamespace(url="http://127.0.0.1:1", model=str(q_gguf), special=True)], None) is None
    assert sanity._infer_dialect([SimpleNamespace(url="http://127.0.0.1:1", model=str(i_gguf), special=False)], None) == "inkling2"
    # Mock without .model preserves special=True fallback
    assert sanity._infer_dialect([SimpleNamespace(url="http://127.0.0.1:1", special=True)], None) == "inkling2"

    def responder(path, body):
        return 200, {"choices": [{"message": {"role": "assistant", "content": "30"}, "finish_reason": "stop"}], "usage": {"completion_tokens": 1}}

    items = [{"id": "arith-0", "prompt": "10+20?", "check": "exact", "answer": "30"}]
    with _LoopbackServer(responder) as srv:
        rows = sanity.run_behavioral(
            [SimpleNamespace(url=srv.url, model=str(q_gguf), special=True)],
            {"temperature": 0.0, "cache_prompt": True},
            items,
            [0],
            "cand",
        )
        assert rows[0]["success"] is True
        assert srv.records[0]["body"]["chat_template_kwargs"] == {"enable_thinking": False}
        assert srv.records[0]["body"]["cache_prompt"] is False


def test_sanity_size_matched_cap_guards():
    with pytest.raises(ValueError):
        sanity.SizeMatchedCap([])
    with pytest.raises(ValueError):
        sanity.SizeMatchedCap([(2.0, float("nan"))])
    with pytest.raises(ValueError):
        sanity.SizeMatchedCap([(float("inf"), 0.1)])

    cap = sanity.SizeMatchedCap([(1.0, 0.9), (2.0, 0.5), (2.0, 0.7), (4.0, 0.1)])
    assert math.isfinite(cap(2.0))
    assert math.isfinite(cap(3.0))
    with pytest.raises(ValueError):
        cap(float("nan"))


def test_sanity_floors_from_solo_signed_bytes_and_nan():
    # u[key] < 0 and u[key] == 0 set limit = 0.0 (never inverting under factor)
    res_neg = sanity.floors_from_solo(
        solo={
            "g0": {
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
    assert res_neg["g0"]["evidence"]["Q4_K"]["limit"] == 0.0
    assert res_neg["g0"]["evidence"]["Q4_K"]["ok"] is True
    assert res_neg["g0"]["min_tier"] == "Q4_K"
    assert res_neg["g0"]["drop_ok"] is False

    res_zero = sanity.floors_from_solo(
        solo={"g0": {"Q8_0": {"pauc": 0.0, "bytes_saved": 50}, "Q4_K": {"pauc": 1e-6, "bytes_saved": 100}}},
        uniform={"Q8_0": {"pauc": 0.0, "bytes_saved": 50}, "Q4_K": {"pauc": 0.0, "bytes_saved": 100}},
        ladder=["BF16", "Q8_0", "Q4_K"],
        factor=4.0,
        key="pauc",
    )
    assert res_zero["g0"]["evidence"]["Q8_0"]["ok"] is True
    assert res_zero["g0"]["evidence"]["Q4_K"]["ok"] is False

    # u["bytes_saved"] <= 0 fails cleanly
    res_zero_bytes = sanity.floors_from_solo(
        solo={"g0": {"Q8_0": {"pauc": -0.01, "bytes_saved": 50}}},
        uniform={"Q8_0": {"pauc": 0.01, "bytes_saved": 0}},
        ladder=["BF16", "Q8_0"],
        factor=4.0,
        key="pauc",
    )
    assert res_zero_bytes["g0"]["evidence"]["Q8_0"]["ok"] is False

    # Non-finite values raise ValueError
    for bad in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError):
            sanity.floors_from_solo(
                solo={"g0": {"Q8_0": {"pauc": bad, "bytes_saved": 50}}},
                uniform={"Q8_0": {"pauc": 0.1, "bytes_saved": 50}},
                ladder=["BF16", "Q8_0"],
                factor=4.0,
                key="pauc",
            )
        with pytest.raises(ValueError):
            sanity.floors_from_solo(
                solo={"g0": {"Q8_0": {"pauc": 0.1, "bytes_saved": 50}}},
                uniform={"Q8_0": {"pauc": bad, "bytes_saved": 50}},
                ladder=["BF16", "Q8_0"],
                factor=4.0,
                key="pauc",
            )
