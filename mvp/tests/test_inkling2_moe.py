"""End-to-end tests for `thinkingmachines/Inkling-2-Mini` (`inkling2_loop`) support:
  1. 45-pattern `inkling2_loop` GGUF group adaptation & loop/attn_res `droppable=False` guard
  2. `chantless.mask/v3` `experts` + `widths` + `vocab_keep` validation & backward-compatible `mask_key`
  3. Composed MoE expert pruning (`8 -> 4`), fused `ffn_gate_up_exps` width slicing (`320 -> 256`),
     and padded vocabulary trimming (`260/256 -> 160`) with `imatrix.gguf` subsetting & exact `ByteModel`
  4. Real `PR #32704` (`inkling2_loop`) `llama-quantize` + `traj-kl` verification on built GGUFs
  5. `bfcl`, `deepmath`, `clsbench` `"inkling2"` dialect handling & Rule-P special-token retention
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "mvp"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from chantless import builder, config as cfgmod, gguf_ops, learn, lattice, mask as mk, oneshot, plan, runtime, util, vocab
from chantless.env import bfcl, clsbench, deepmath


def _kv_str(s: str) -> tuple[int, bytes]:
    b = s.encode("utf-8")
    return 8, struct.pack("<Q", len(b)) + b


def _kv_u32(v: int) -> tuple[int, bytes]:
    return 4, struct.pack("<I", v)


def _kv_f32(v: float) -> tuple[int, bytes]:
    return 6, struct.pack("<f", v)


def _kv_bool(v: bool) -> tuple[int, bytes]:
    return 7, struct.pack("<?", v)


def _kv_i32_arr(vals: list[int]) -> tuple[int, bytes]:
    return 9, struct.pack("<IQ", 5, len(vals)) + struct.pack(f"<{len(vals)}i", *vals)


def _kv_bool_arr(vals: list[bool]) -> tuple[int, bytes]:
    return 9, struct.pack("<IQ", 7, len(vals)) + struct.pack(f"<{len(vals)}?", *vals)


def _kv_str_arr(items: list[bytes]) -> tuple[int, bytes]:
    b = bytearray(struct.pack("<IQ", 8, len(items)))
    for x in items:
        b += struct.pack("<Q", len(x)) + x
    return 9, bytes(b)


def _kv_f32_arr(vals: list[float]) -> tuple[int, bytes]:
    return 9, struct.pack("<IQ", 6, len(vals)) + struct.pack(f"<{len(vals)}f", *vals)


def _f32_to_bf16_u16(arr: np.ndarray) -> np.ndarray:
    f32 = np.asarray(arr, dtype=np.float32)
    return (f32.view(np.uint32) >> 16).astype(np.uint16)


def _write_gguf(path: Path, kvs: list[tuple[str, tuple[int, bytes]]], tensors: list[tuple[str, list[int], int, bytes]]) -> None:
    kv_buf = bytearray()
    for k, (vt, raw) in kvs:
        kb = k.encode("utf-8")
        kv_buf += struct.pack("<Q", len(kb)) + kb + struct.pack("<I", vt) + raw
    gguf_ops._write_gguf_raw(3, 32, len(kvs), bytes(kv_buf), tensors, path)


def _write_trj1(path: Path, seqs: list[list[int]]) -> None:
    buf = bytearray(b"TRJ1" + struct.pack("<I", len(seqs)))
    for seq in seqs:
        tgts = list(range(1, len(seq)))
        buf += struct.pack("<II", len(seq), len(tgts))
        buf += struct.pack(f"<{len(seq)}i", *seq)
        buf += struct.pack(f"<{len(tgts)}I", *tgts)
    path.write_bytes(bytes(buf))


def _resolve_inkling2_toolchain() -> runtime.Toolchain | None:
    from conftest import resolve_toolchain

    llama_dir, tools_bin = resolve_toolchain()
    llama_bin = llama_dir / "build/bin"
    if (
        (llama_dir / "src/models/inkling2-loop.cpp").exists()
        and (llama_bin / "llama-quantize").exists()
        and (tools_bin / "traj-kl").exists()
    ):
        return runtime.Toolchain(
            llama_bin=llama_bin,
            traj_kl=tools_bin / "traj-kl",
            gpus=[""],
            logodds=tools_bin / "logodds",
            tokdocs=tools_bin / "tokdocs",
        )
    root = ROOT / ".local/toolchain-inkling2"
    llama_bin = root / "llama-inkling2/build/bin"
    tools_bin = root / "bin"
    if not (llama_bin / "llama-quantize").exists() or not (tools_bin / "traj-kl").exists():
        return None
    return runtime.Toolchain(
        llama_bin=llama_bin,
        traj_kl=tools_bin / "traj-kl",
        gpus=[""],
        logodds=tools_bin / "logodds",
        tokdocs=tools_bin / "tokdocs",
    )


def _build_mini_inkling2_fixture(tmp_path: Path) -> tuple[Path, Path, dict]:
    n_layer = 6
    n_embd = 256
    n_head = 4
    n_head_kv = 2
    head_dim = 64
    n_rel = 4
    n_rel_ext = 16
    n_swa = 16
    n_conv = 4
    n_ff_exp = 320  # 10 * 32: compatible with IQ4_NL/Q8_0, not divisible by 256 (Q4_K/Q2_K require width trim)
    n_expert = 8
    n_expert_used = 2
    v_unpadded = 256
    v_padded = 260
    swa_pattern = [True, False, True, False, True, False]

    ctrl_toks = [
        b"<|unused|>",
        b"<|endoftext|>",
        b"<|message_user|>",
        b"<|message_model|>",
        b"<|message_system|>",
        b"<|message_tool|>",
        b"<|content_text|>",
        b"<|content_thinking|>",
        b"<|end_message|>",
        b"<|content_model_end_sampling|>",
        b"<|content_invoke_tool_json|>",
    ]
    n_special = len(ctrl_toks)
    tokens: list[bytes] = []
    ttypes: list[int] = []
    for i in range(v_unpadded - n_special):
        tok = chr(97 + (i % 26)).encode() if i < 26 else f"w{i}".encode()
        tokens.append(tok)
        ttypes.append(1)
    tokens[0] = b"a"
    tokens[1] = b"b"
    tokens[2] = b"ab"
    tokens[3] = b","
    tokens[200] = b"negative"
    tokens[201] = b"positive"
    eos_id = (v_unpadded - n_special) + ctrl_toks.index(b"<|content_model_end_sampling|>")
    pad_id = (v_unpadded - n_special) + ctrl_toks.index(b"<|endoftext|>")
    for tok in ctrl_toks:
        tokens.append(tok)
        ttypes.append(3)
    merges = [b"a b"]

    kvs = [
        ("general.architecture", _kv_str("inkling2_loop")),
        ("general.alignment", _kv_u32(32)),
        ("general.name", _kv_str("Inkling-2-Mini-Test")),
        ("inkling2_loop.block_count", _kv_u32(n_layer)),
        ("inkling2_loop.context_length", _kv_u32(256)),
        ("inkling2_loop.embedding_length", _kv_u32(n_embd)),
        ("inkling2_loop.feed_forward_length", _kv_u32(n_ff_exp)),
        ("inkling2_loop.expert_feed_forward_length", _kv_u32(n_ff_exp)),
        ("inkling2_loop.expert_count", _kv_u32(n_expert)),
        ("inkling2_loop.expert_used_count", _kv_u32(n_expert_used)),
        ("inkling2_loop.attention.head_count", _kv_i32_arr([n_head] * n_layer)),
        ("inkling2_loop.attention.head_count_kv", _kv_i32_arr([n_head_kv] * n_layer)),
        ("inkling2_loop.attention.key_length", _kv_u32(head_dim)),
        ("inkling2_loop.attention.value_length", _kv_u32(head_dim)),
        ("inkling2_loop.attention.key_length_swa", _kv_u32(head_dim)),
        ("inkling2_loop.attention.value_length_swa", _kv_u32(head_dim)),
        ("inkling2_loop.attention.layer_norm_rms_epsilon", _kv_f32(1e-6)),
        ("inkling2_loop.rope.dimension_count", _kv_u32(0)),
        ("inkling2_loop.vocab_size", _kv_u32(v_padded)),
        ("inkling2_loop.vocab_size_unpadded", _kv_u32(v_unpadded)),
        ("inkling2_loop.attention.sliding_window", _kv_u32(n_swa)),
        ("inkling2_loop.attention.sliding_window_pattern", _kv_bool_arr(swa_pattern)),
        ("inkling2_loop.shortconv.l_cache", _kv_u32(n_conv)),
        ("inkling2_loop.expert_weights_scale", _kv_f32(8.0)),
        ("inkling2_loop.activation.situ_beta", _kv_f32(4.0)),
        ("inkling2_loop.activation.situ_linear_beta", _kv_f32(25.0)),
        ("inkling2_loop.attn_res.block_size", _kv_u32(2)),
        ("inkling2_loop.final_logit_softcapping", _kv_f32(30.0)),
        ("inkling2_loop.attention.relative_length", _kv_u32(n_rel)),
        ("inkling2_loop.attention.relative_extent", _kv_u32(n_rel_ext)),
        ("inkling2_loop.attention.qk_clip", _kv_f32(22.627417)),
        ("inkling2_loop.attention.relative_clip", _kv_f32(8.0)),
        ("inkling2_loop.attention.log_scale_floor", _kv_u32(8192)),
        ("inkling2_loop.attention.log_scale_alpha", _kv_f32(0.1)),
        ("inkling2_loop.loop.start", _kv_u32(2)),
        ("inkling2_loop.loop.end", _kv_u32(4)),
        ("inkling2_loop.loop.count", _kv_u32(2)),
        ("inkling2_loop.loop.gate_rank", _kv_u32(4)),
        ("inkling2_loop.loop.varying", _kv_bool(False)),
        ("tokenizer.ggml.model", _kv_str("gpt2")),
        ("tokenizer.ggml.pre", _kv_str("inkling2")),
        ("tokenizer.ggml.tokens", _kv_str_arr(tokens)),
        ("tokenizer.ggml.token_type", _kv_i32_arr(ttypes)),
        ("tokenizer.ggml.merges", _kv_str_arr(merges)),
        ("tokenizer.ggml.eos_token_id", _kv_u32(eos_id)),
        ("tokenizer.ggml.eot_token_id", _kv_u32(eos_id)),
        ("tokenizer.ggml.padding_token_id", _kv_u32(pad_id)),
        ("tokenizer.ggml.add_bos_token", _kv_bool(False)),
        ("tokenizer.ggml.add_eos_token", _kv_bool(False)),
    ]

    rng = np.random.default_rng(20261121)

    def bf16(shape_rev: tuple[int, ...], scale: float = 0.05) -> bytes:
        arr = (rng.standard_normal(shape_rev) * scale).astype(np.float32)
        return _f32_to_bf16_u16(arr).tobytes()

    def f32(shape_rev: tuple[int, ...], fill: float | None = None, scale: float = 0.05) -> bytes:
        if fill is not None:
            return np.full(shape_rev, fill, dtype=np.float32).tobytes()
        return (rng.standard_normal(shape_rev) * scale).astype(np.float32).tobytes()

    tensors: list[tuple[str, list[int], int, bytes]] = [
        ("token_embd.weight", [n_embd, v_padded], 30, bf16((v_padded, n_embd))),
        ("token_embd_norm.weight", [n_embd], 0, f32((n_embd,), fill=1.0)),
        ("output.weight", [n_embd, v_padded], 30, bf16((v_padded, n_embd))),
        ("output_norm.weight", [n_embd], 0, f32((n_embd,), fill=1.0)),
        ("output_res_query.weight", [n_embd], 0, f32((n_embd,))),
    ]
    im_tensors: list[tuple[str, list[int], int, bytes]] = []

    def add_im(wname: str, cols: int, n_mat: int = 1, custom_sum2: np.ndarray | None = None, custom_cnt: np.ndarray | None = None) -> None:
        s2 = np.ones((n_mat, cols), dtype=np.float32) if custom_sum2 is None else custom_sum2.astype(np.float32)
        cnt = np.full((n_mat, 1), 32.0, dtype=np.float32) if custom_cnt is None else custom_cnt.astype(np.float32)
        im_tensors.append((f"{wname}.in_sum2", [cols, n_mat], 0, s2.tobytes()))
        im_tensors.append((f"{wname}.counts", [1, n_mat], 0, cnt.tobytes()))
        im_tensors.append((f"{wname}.in_sum", [cols, n_mat], 0, (s2 * 0.1).astype(np.float32).tobytes()))

    for il in range(n_layer):
        p = f"blk.{il}."
        is_swa = swa_pattern[il]
        n_ext = n_swa if is_swa else n_rel_ext

        # Make experts 4..7 strictly higher energy & activation than 0..3, and neurons 64..319 higher than 0..63
        down_w = (rng.standard_normal((n_expert, n_embd, n_ff_exp)) * 0.05).astype(np.float32)
        down_w[4:, :, :] *= 4.0
        down_w[:, :, 64:] *= 3.0
        gu_w = (rng.standard_normal((n_expert, 2 * n_ff_exp, n_embd)) * 0.05).astype(np.float32)
        # Put distinct pattern in up half vs gate half so contiguous-half slicing is verified
        gu_w[:, n_ff_exp:, :] += 0.25

        im_down = np.ones((n_expert, n_ff_exp), dtype=np.float32)
        im_down[4:, :] = 10.0
        im_down[:, 64:] *= 5.0
        cnt_exp = np.array([[5.0], [5.0], [5.0], [5.0], [50.0], [50.0], [50.0], [50.0]], dtype=np.float32)
        im_gu = np.full((n_expert, n_embd), 2.0, dtype=np.float32)

        tensors += [
            (p + "attn_q.weight", [n_embd, n_head * head_dim], 30, bf16((n_head * head_dim, n_embd))),
            (p + "attn_k.weight", [n_embd, n_head_kv * head_dim], 30, bf16((n_head_kv * head_dim, n_embd))),
            (p + "attn_v.weight", [n_embd, n_head_kv * head_dim], 30, bf16((n_head_kv * head_dim, n_embd))),
            (p + "attn_output.weight", [n_head * head_dim, n_embd], 30, bf16((n_embd, n_head * head_dim))),
            (p + "attn_gate.weight", [n_embd, n_head * head_dim], 30, bf16((n_head * head_dim, n_embd))),
            (p + "attn_q_clip.weight", [head_dim, n_head], 0, f32((n_head, head_dim), fill=1.0)),
            (p + "attn_sink_proj.weight", [n_embd, n_head], 30, bf16((n_head, n_embd))),
            (p + "attn_rel.weight", [n_embd, n_rel * n_head], 30, bf16((n_rel * n_head, n_embd))),
            (p + "attn_rel_bias.weight", [n_rel, n_ext], 0, f32((n_ext, n_rel))),
            (p + "attn_rel_clip.weight", [n_rel, n_head], 0, f32((n_head, n_rel), fill=1.0)),
            (p + "attn_k_conv.weight", [n_conv, n_head_kv * head_dim], 0, f32((n_head_kv * head_dim, n_conv))),
            (p + "attn_v_conv.weight", [n_conv, n_head_kv * head_dim], 0, f32((n_head_kv * head_dim, n_conv))),
            (p + "attn_post_conv.weight", [n_conv, n_embd], 0, f32((n_embd, n_conv))),
            (p + "ffn_post_conv.weight", [n_conv, n_embd], 0, f32((n_embd, n_conv))),
        ]
        if is_swa:
            tensors.append((p + "attn_norm.weight", [n_embd], 0, f32((n_embd,), fill=1.0)))
        else:
            tensors += [
                (p + "attn_q_inp_norm.weight", [n_embd], 0, f32((n_embd,), fill=1.0)),
                (p + "attn_k_inp_norm.weight", [n_embd], 0, f32((n_embd,), fill=1.0)),
                (p + "attn_v_inp_norm.weight", [n_embd], 0, f32((n_embd,), fill=1.0)),
                (p + "attn_k_conv_proj.weight", [n_embd, n_conv * n_head_kv], 30, bf16((n_conv * n_head_kv, n_embd))),
                (p + "attn_v_conv_proj.weight", [n_embd, n_conv * n_head_kv], 30, bf16((n_conv * n_head_kv, n_embd))),
            ]
        if il > 0:
            tensors.append((p + "attn_res_query.weight", [n_embd, 1 if is_swa else 3], 0, f32((1 if is_swa else 3, n_embd))))
        tensors += [
            (p + "ffn_res_query.weight", [n_embd], 0, f32((n_embd,))),
            (p + "ffn_norm.weight", [n_embd], 0, f32((n_embd,), fill=1.0)),
            (p + "ffn_gate_inp.weight", [n_embd, n_expert], 0, f32((n_expert, n_embd))),
            (p + "exp_probs_b.bias", [n_expert], 0, f32((n_expert,), fill=0.0)),
            (p + "ffn_gate_up_exps.weight", [n_embd, 2 * n_ff_exp, n_expert], 30, _f32_to_bf16_u16(gu_w).tobytes()),
            (p + "ffn_down_exps.weight", [n_ff_exp, n_embd, n_expert], 30, _f32_to_bf16_u16(down_w).tobytes()),
        ]
        if 2 <= il < 4:
            tensors.append((p + "loop_msg_query.weight", [n_embd, 4], 0, f32((4, n_embd))))
            for mod in ("q", "k", "v", "ffn"):
                tensors += [
                    (p + f"loop_{mod}_ctrl_down.weight", [2 * n_embd, 4], 0, f32((4, 2 * n_embd))),
                    (p + f"loop_{mod}_ctrl_gain.weight", [4, n_embd], 0, f32((n_embd, 4))),
                    (p + f"loop_{mod}_ctrl_forget.weight", [4, n_embd], 0, f32((n_embd, 4))),
                ]

        for nm, cols in (
            ("attn_q.weight", n_embd), ("attn_k.weight", n_embd), ("attn_v.weight", n_embd),
            ("attn_output.weight", n_head * head_dim), ("attn_gate.weight", n_embd),
            ("attn_sink_proj.weight", n_embd), ("attn_rel.weight", n_embd),
        ):
            add_im(p + nm, cols)
        add_im(p + "ffn_gate_up_exps.weight", n_embd, n_mat=n_expert, custom_sum2=im_gu, custom_cnt=cnt_exp)
        add_im(p + "ffn_down_exps.weight", n_ff_exp, n_mat=n_expert, custom_sum2=im_down, custom_cnt=cnt_exp)

    src_gguf = tmp_path / "inkling2-mini-test-bf16.gguf"
    im_gguf = tmp_path / "inkling2-mini-imatrix.gguf"
    _write_gguf(src_gguf, kvs, tensors)
    im_kvs = [
        ("general.type", _kv_str("imatrix")),
        ("imatrix.datasets", _kv_str_arr([b"test"])),
        ("imatrix.chunk_count", _kv_u32(1)),
        ("imatrix.chunk_size", _kv_u32(64)),
    ]
    _write_gguf(im_gguf, im_kvs, im_tensors)

    dims = {
        "n_layer": n_layer, "n_embd": n_embd, "n_ff_exp": n_ff_exp, "n_expert": n_expert,
        "v_unpadded": v_unpadded, "v_padded": v_padded, "eos_id": eos_id, "pad_id": pad_id,
        "n_special": n_special,
    }
    return src_gguf, im_gguf, dims


# ------------------------------------------------------------------ 1. Adapter, ordering & mask validation
def test_inkling2_adapter_ordering_and_mask_validation(tmp_path: Path):
    src, im, dims = _build_mini_inkling2_fixture(tmp_path)
    groups, dead = gguf_ops.adapter_from_gguf(src)
    assert not dead
    real = {g: v for g, v in groups.items() if not g.startswith("_")}
    assert len(real) == 2 + 2 * dims["n_layer"]

    # Looped / AttnRes guard prevents dropping any block sub-group
    assert all(not real[g]["droppable"] for g in real)
    for il in range(dims["n_layer"]):
        ffn_q = set(real[f"blk.{il}.ffn"]["quantizable"])
        assert ffn_q == {f"blk.{il}.ffn_gate_up_exps.weight", f"blk.{il}.ffn_down_exps.weight"}
        ffn_f = set(real[f"blk.{il}.ffn"]["fixed"])
        assert f"blk.{il}.ffn_gate_inp.weight" in ffn_f
        assert f"blk.{il}.exp_probs_b.bias" in ffn_f
        attn_q = set(real[f"blk.{il}.attn"]["quantizable"])
        assert {f"blk.{il}.attn_rel.weight", f"blk.{il}.attn_sink_proj.weight"} <= attn_q

    info = gguf_ops.arch_info(src)
    assert info["arch"] == "inkling2_loop"
    assert info["n_expert"] == dims["n_expert"]
    assert info["n_ff_exp"] == dims["n_ff_exp"]
    assert gguf_ops.ffn_widths(src) == {il: dims["n_ff_exp"] for il in range(dims["n_layer"])}

    exp_ord = gguf_ops.expert_importance_from_imatrix(src, im)
    for il in range(dims["n_layer"]):
        assert set(exp_ord[il][:4].tolist()) == {4, 5, 6, 7}

    kept_exp = {il: exp_ord[il][:4] for il in range(dims["n_layer"])}
    w_ord = gguf_ops.wanda_order(src, im, kept_experts=kept_exp)
    for il in range(dims["n_layer"]):
        assert len(w_ord[il]) == dims["n_ff_exp"]
        assert set(w_ord[il][:256].tolist()) <= set(range(64, dims["n_ff_exp"]))

    # Mask validation: absent `experts` preserves v3 hash; `320` fails Q4_K, passes IQ4_NL; `256` passes Q4_K/Q2_K
    m = mk.new_mask(groups, "IQ4_NL", {"repo": "test", "revision": "0", "gguf_sha256": "0"})
    m["groups"]["token_embd"]["tier"] = "Q8_0"
    m["groups"]["output"]["tier"] = "Q8_0"
    mk.validate(m, groups, ladder=mk.ALL_TIERS)
    key_no_exp = builder.mask_key(m)
    assert "experts" not in m

    m_exp = json.loads(json.dumps(m))
    m_exp["experts"] = {str(il): mk.pack_bitmap(kept_exp[il], dims["n_expert"]) for il in range(dims["n_layer"])}
    mk.validate(m_exp, groups, ladder=mk.ALL_TIERS)
    assert builder.mask_key(m_exp) != key_no_exp

    # Non-uniform expert counts must be rejected
    m_bad_exp = json.loads(json.dumps(m_exp))
    m_bad_exp["experts"]["0"] = mk.pack_bitmap([4, 5, 6], dims["n_expert"])
    with pytest.raises(mk.MaskError, match="uniform expert count"):
        mk.validate(m_bad_exp, groups, ladder=mk.ALL_TIERS)

    # 320 is not divisible by Q4_K block 256, whereas 256 is
    m_q4 = json.loads(json.dumps(m_exp))
    m_q4["groups"]["blk.0.ffn"]["tier"] = "Q4_K"
    m_q4["widths"]["0"] = mk.pack_bitmap(np.arange(320), dims["n_ff_exp"])
    with pytest.raises(mk.MaskError, match="not divisible by Q4_K"):
        mk.validate(m_q4, groups, ladder=mk.ALL_TIERS)
    m_q4["widths"]["0"] = mk.pack_bitmap(w_ord[0][:256], dims["n_ff_exp"])
    mk.validate(m_q4, groups, ladder=mk.ALL_TIERS)


# ------------------------------------------------------------------ 2. Real Builder + PR #32704 llama-quantize & traj-kl
def test_inkling2_real_build_bytemodel_and_traj_kl(tmp_path: Path):
    tc = _resolve_inkling2_toolchain()
    if tc is None:
        pytest.skip("local .local/toolchain-inkling2 PR #32704 toolchain not built")

    src, im, dims = _build_mini_inkling2_fixture(tmp_path)
    groups, _ = gguf_ops.adapter_from_gguf(src)
    exp_ord = gguf_ops.expert_importance_from_imatrix(src, im)
    kept_exp = {il: np.sort(exp_ord[il][:4]) for il in range(dims["n_layer"])}
    w_ord = gguf_ops.wanda_order(src, im, kept_experts=kept_exp)

    # Verify vocab keep set preserves all Inkling-2 control tokens & closes BPE merges
    counts = np.zeros(dims["v_unpadded"], dtype=np.int64)
    counts[2] = 500  # "ab" -> must pull in "a" (0) & "b" (1)
    counts[140:190] = 20
    keep_vocab = vocab.rule_p_keep_set(src, counts, k=160, must_keep=(200, 201))
    assert len(keep_vocab) <= 160
    remapped = vocab.remap_token_ids(keep_vocab, (200, 201, dims["eos_id"]))
    assert remapped[0] != 200 and remapped[1] != 201  # shifted IDs test real remapping

    # Also test unpadding alone (260 -> 256)
    unpad_out = tmp_path / "unpadded.gguf"
    res_unpad = vocab.trim_gguf(src, unpad_out, np.arange(dims["v_unpadded"], dtype=np.int64))
    assert res_unpad["kept_tokens"] == dims["v_unpadded"]
    h_unp = gguf_ops._Hdr(unpad_out)
    try:
        kv_unp = h_unp.parsed_kvs()
        assert kv_unp["inkling2_loop.vocab_size"] == dims["v_unpadded"]
        assert kv_unp["inkling2_loop.vocab_size_unpadded"] == dims["v_unpadded"]
        assert {t["name"]: t for t in h_unp.tensors}["token_embd.weight"]["dims"][1] == dims["v_unpadded"]
    finally:
        h_unp.close()

    bm = oneshot.ByteModel(str(src), groups)
    B = builder.Builder(tc, str(src), groups, str(im), tmp_path / "work", threads=4, quantize_shards=1)

    # Build A: IQ4_NL MoE (native 1152/320 compatible) + 8->4 expert pruning + Q8_0 attn/embd + padded vocab trim
    mask_a = mk.new_mask(groups, "Q8_0", {"repo": "test", "revision": "0", "gguf_sha256": "0"})
    for il in range(dims["n_layer"]):
        mask_a["groups"][f"blk.{il}.ffn"]["tier"] = "IQ4_NL"
    mask_a["experts"] = {str(il): mk.pack_bitmap(kept_exp[il], dims["n_expert"]) for il in range(dims["n_layer"])}
    mask_a["vocab_keep"] = mk.pack_bitmap(keep_vocab, dims["v_unpadded"])

    out_a = tmp_path / "cand_iq4nl_exp4_vocab.gguf"
    res_a = B.build(mask_a, out_a)
    # Calibrate ByteModel on one real IQ4_NL expert-pruned build (matching real pipeline flow), then check vocab prediction
    mask_a_cal = json.loads(json.dumps(mask_a))
    mask_a_cal["vocab_keep"] = None
    cal_a = B.build(mask_a_cal, tmp_path / "cand_cal.gguf")
    bm.calibrate(mask_a_cal, cal_a.file_bytes)
    assert bm(mask_a) == res_a.file_bytes

    # Build B: 320 -> 256 MoE width slicing + 8->4 expert pruning + Q4_K on MoE exps!
    mask_b = mk.new_mask(groups, "Q4_K", {"repo": "test", "revision": "0", "gguf_sha256": "0"})
    mask_b["groups"]["token_embd"]["tier"] = "Q8_0"
    mask_b["experts"] = {str(il): mk.pack_bitmap(kept_exp[il], dims["n_expert"]) for il in range(dims["n_layer"])}
    mask_b["widths"] = {str(il): mk.pack_bitmap(w_ord[il][:256], dims["n_ff_exp"]) for il in range(dims["n_layer"])}

    out_b = tmp_path / "cand_q4k_exp4_w256.gguf"
    res_b = B.build(mask_b, out_b)
    assert bm(mask_b) == res_b.file_bytes

    # Verify built GGUF KVs and tensor shapes
    hb = gguf_ops._Hdr(out_b)
    try:
        kvb = hb.parsed_kvs()
        assert kvb["inkling2_loop.expert_count"] == 4
        assert kvb["inkling2_loop.expert_feed_forward_length"] == 256
        invb = {t["name"]: t for t in hb.tensors}
        assert invb["blk.0.ffn_down_exps.weight"]["type"] == "Q4_K"
        assert invb["blk.0.ffn_down_exps.weight"]["dims"] == [256, dims["n_embd"], 4]
        assert invb["blk.0.ffn_gate_up_exps.weight"]["type"] == "Q4_K"
        assert invb["blk.0.ffn_gate_up_exps.weight"]["dims"] == [dims["n_embd"], 512, 4]
    finally:
        hb.close()

    # Run compiled PR #32704 `traj-kl` self-KL check + PyTorch Inkling2LoopStudent parity (< 1e-3)
    # Include len >= 20 (> n_swa=16, n_rel_ext=16) so SWA masking, relative-bias clamping, and shortconv history execute non-vacuously.
    trj_full = tmp_path / "seqs_full.trj"
    trj_trim = tmp_path / "seqs_trim.trj"
    r_a, r_b, r_ab, r_comma, tok_s0, tok_s1, eos_id = vocab.remap_token_ids(
        keep_vocab, (0, 1, 2, 3, 200, 201, dims["eos_id"])
    )
    _write_trj1(
        trj_full,
        [
            [200, 201, 0, 1, 2, dims["eos_id"]],
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 200, 201, 0, 1, 2, 3, 200, 201, dims["eos_id"]],
        ],
    )
    _write_trj1(
        trj_trim,
        [
            [tok_s0, tok_s1, eos_id],
            [r_a, r_b, r_ab, r_comma] * 5 + [eos_id],
        ],
    )

    has_torch = importlib.util.find_spec("torch") is not None
    for tag, gg, trj in (("bf16", src, trj_full), ("iq4nl_exp4_vocab", out_a, trj_trim), ("q4k_exp4_w256", out_b, trj_full)):
        store = tmp_path / f"store_{tag}"
        store.mkdir()
        subprocess.run(
            [str(tc.traj_kl), "--mode", "base", "-m", str(gg), "--traj", str(trj), "--store", str(store), "--threads", "2"],
            env=tc.env_for(""),
            capture_output=True,
            text=True,
            check=True,
        )
        proc = subprocess.run(
            [str(tc.traj_kl), "--mode", "kl", "-m", str(gg), "--traj", str(trj), "--store", str(store), "--threads", "2"],
            env=tc.env_for(""),
            capture_output=True,
            text=True,
            check=True,
        )
        rep = json.loads(proc.stdout)
        assert math.isfinite(float(rep["nll_cand"])), (gg.name, rep)
        # `base_0.f16` stores FP16 log-probs; uniform-ish logits yield ~1e-4 FP16 rounding noise
        assert abs(float(rep["kl"])) < 1e-3, (gg.name, rep)
        assert float(rep["top1"]) == 1.0, (gg.name, rep)
        assert abs(float(rep["nll_cand"]) - float(rep["nll_src"])) < 2e-3, (gg.name, rep)

        if has_torch:
            art = lattice.Artifact(str(gg))
            try:
                stu, _ = learn.build_student(art, None, [], None if tag == "bf16" else str(tc.llama_bin), device="cpu")
            finally:
                art.close()
            stu_trajs = learn.student_traj_nll(stu, learn.read_trj_multi(trj), device="cpu")
            assert len(stu_trajs) == len(rep["trajs"])
            if tag in ("iq4nl_exp4_vocab", "q4k_exp4_w256"):
                # Random-weight fixture uses `exp_probs_b.bias = 0.0` (router near-ties can flip between Metal FP16 FA and CPU FP32 on long sequences);
                # verify seq 0 (< 1e-3) here and full multi-length (< 1e-3, including len=21 > 16) with non-tied router bias right below.
                assert all(math.isfinite(float(s["nll_cand"])) for s in stu_trajs)
                assert abs(float(stu_trajs[0]["nll_cand"]) - float(rep["trajs"][0]["nll_cand"])) < 1e-3
            else:
                max_nll_diff = max(abs(float(s["nll_cand"]) - float(c["nll_cand"])) for s, c in zip(stu_trajs, rep["trajs"]))
                assert max_nll_diff < 1e-3, (tag, max_nll_diff, stu_trajs, rep["trajs"])
            if tag == "bf16":
                art_bf = lattice.Artifact(str(gg))
                try:
                    stu_bf, _ = learn.build_student(art_bf, None, [], None, device="cpu", frozen_bf16=True)
                finally:
                    art_bf.close()
                stu_bf_trajs = learn.student_traj_nll(stu_bf, learn.read_trj_multi(trj), device="cpu")
                assert all(math.isfinite(s["nll_cand"]) for s in stu_bf_trajs)
                bf_diff = max(abs(float(s["nll_cand"]) - float(c["nll_cand"])) for s, c in zip(stu_bf_trajs, rep["trajs"]))
                assert bf_diff < 0.05, ("bf16_frozen_bf16", bf_diff)

    if has_torch:
        # Gate (a) IQ4_NL (expert prune + vocab trim) and Q4_K (expert prune + width slice) parity (< 1e-3 across all lengths including len=21 > n_swa=16):
        # Set distinct exp_probs_b.bias so Metal FP16 FA vs PyTorch CPU FP32 cannot flip near-tie routers.
        src_sep = tmp_path / "src_moe_sep.gguf"
        src_sep.write_bytes(src.read_bytes())
        art_sep = lattice.Artifact(str(src_sep))
        bias_updates = []
        for tname, ti in art_sep.tensors.items():
            if "exp_probs_b.bias" in tname:
                bvals = np.arange(ti["dims"][0], dtype=np.float32) * 5.0
                bias_updates.append((art_sep.h.data_start + ti["offset"], bvals.tobytes()))
        art_sep.close()
        with open(src_sep, "r+b") as bf:
            for off_pos, bbytes in bias_updates:
                bf.seek(off_pos)
                bf.write(bbytes)
        mask_q4k_moe = mk.new_mask(groups, "Q8_0", {"repo": "test", "revision": "0", "gguf_sha256": "0"})
        for il in range(dims["n_layer"]):
            mask_q4k_moe["groups"][f"blk.{il}.ffn"]["tier"] = "Q4_K"
        mask_q4k_moe["experts"] = {str(il): mk.pack_bitmap(kept_exp[il], dims["n_expert"]) for il in range(dims["n_layer"])}
        mask_q4k_moe["widths"] = {str(il): mk.pack_bitmap(w_ord[il][:256], dims["n_ff_exp"]) for il in range(dims["n_layer"])}
        B_sep = builder.Builder(tc, str(src_sep), groups, str(im), tmp_path / "work_moe_sep", threads=4, quantize_shards=1)
        out_iq4nl_sep = tmp_path / "cand_iq4nl_sep.gguf"
        out_q4k_moe = tmp_path / "cand_q4k_moe_w256.gguf"
        B_sep.build(mask_a, out_iq4nl_sep)
        B_sep.build(mask_q4k_moe, out_q4k_moe)
        for mtag, mgg, mtrj in (("iq4nl_sep", out_iq4nl_sep, trj_trim), ("q4k_moe_sep", out_q4k_moe, trj_full)):
            mstore = tmp_path / f"store_{mtag}"
            mstore.mkdir()
            subprocess.run(
                [str(tc.traj_kl), "--mode", "base", "-m", str(mgg), "--traj", str(mtrj), "--store", str(mstore), "--threads", "2"],
                env=tc.env_for(""), capture_output=True, text=True, check=True,
            )
            mproc = subprocess.run(
                [str(tc.traj_kl), "--mode", "kl", "-m", str(mgg), "--traj", str(mtrj), "--store", str(mstore), "--threads", "2"],
                env=tc.env_for(""), capture_output=True, text=True, check=True,
            )
            mrep = json.loads(mproc.stdout)
            art_m = lattice.Artifact(str(mgg))
            try:
                stu_m, _ = learn.build_student(art_m, None, [], str(tc.llama_bin), device="cpu")
            finally:
                art_m.close()
            stu_m_trajs = learn.student_traj_nll(stu_m, learn.read_trj_multi(mtrj), device="cpu")
            m_diff = max(abs(float(s["nll_cand"]) - float(c["nll_cand"])) for s, c in zip(stu_m_trajs, mrep["trajs"]))
            assert m_diff < 1e-3, (mtag, m_diff, stu_m_trajs, mrep["trajs"])

    # Also score cross-model KL from original BF16 (`src`) -> Q4_K expert-pruned & width-256 (`out_b`)
    cross_proc = subprocess.run(
        [str(tc.traj_kl), "--mode", "kl", "-m", str(out_b), "--traj", str(trj_full), "--store", str(tmp_path / "store_bf16"), "--threads", "2"],
        env=tc.env_for(""),
        capture_output=True,
        text=True,
        check=True,
    )
    cross_rep = json.loads(cross_proc.stdout)
    assert math.isfinite(float(cross_rep["kl"])) and float(cross_rep["kl"]) >= 0.0, cross_rep

    # Verify clsbench detects `inkling2`, sets `<|message_model|><|content_text|>` suffix, and remaps trimmed vocab cleanly
    from tokenizers import Tokenizer, models as tok_models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    hf_dir = tmp_path / "hf_inkling2_cls"
    hf_dir.mkdir()
    src_tokens = vocab.read_tokenizer(src)["tokens"]
    vocab_map = {t.decode("utf-8"): i for i, t in enumerate(src_tokens)}
    wl = Tokenizer(tok_models.WordLevel(vocab=vocab_map, unk_token="<|endoftext|>"))
    wl.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    ct = (
        "{%- set eff = 0 if reasoning_effort == 'none' else 0.9 -%}"
        "{{- '<|message_system|><|content_text|>Thinking effort level: ' ~ eff ~ '<|end_message|>' -}}"
        "{%- for m in messages -%}{{- '<|message_' ~ m.role ~ '|><|content_text|>' ~ m.content ~ '<|end_message|>' -}}{%- endfor -%}"
    )
    hf_tok = PreTrainedTokenizerFast(tokenizer_object=wl, eos_token="<|content_model_end_sampling|>", pad_token="<|endoftext|>", chat_template=ct)
    hf_tok.save_pretrained(hf_dir)
    cls_jsonl = tmp_path / "cls_ink.jsonl"
    cls_jsonl.write_text(json.dumps({"id": "s0", "text": "a b ab", "label": "positive", "split": "search", "family": "srch/0"}) + "\n")
    env = clsbench.PublicClassifierEnvironment(hf_dir=str(hf_dir), cache_dir=str(tmp_path / "cls_cache"), jsonl_file=str(cls_jsonl))
    assert env.dialect == "inkling2"
    pstr = env._prompt_str["s0"]
    assert "Thinking effort level: 0" in pstr
    assert pstr.endswith("<|message_model|><|content_text|>")
    pair_trim, trimmed = env._pair_for_model(str(out_a))
    assert trimmed and pair_trim == (remapped[1], remapped[0])  # (unflagged="positive", flagged="negative")


# ------------------------------------------------------------------ 3. `inkling2` dialects (BFCL, DeepMath, ClsBench)
def test_inkling2_env_dialects(tmp_path: Path):
    from tokenizers import Tokenizer, models as tok_models
    from transformers import PreTrainedTokenizerFast

    vocab_map = {
        "Hello": 0, "Sys": 1, "<|endoftext|>": 2, "<|message_user|>": 3,
        "<|message_model|>": 4, "<|message_system|>": 5, "<|content_text|>": 6,
        "<|content_thinking|>": 7, "<|end_message|>": 8,
        "<|content_model_end_sampling|>": 9, "<|content_invoke_tool_json|>": 10,
    }
    bpe = Tokenizer(tok_models.BPE(vocab=vocab_map, merges=[]))
    ct = (
        "{%- set eff = 0 if reasoning_effort == 'none' else 0.9 -%}"
        "{{- '<|message_system|><|content_text|>Thinking effort level: ' ~ eff ~ '<|end_message|>' -}}"
        "{%- for m in messages -%}{{- '<|message_' ~ m.role ~ '|><|content_text|>' ~ m.content ~ '<|end_message|>' -}}{%- endfor -%}"
    )
    tok = PreTrainedTokenizerFast(tokenizer_object=bpe, chat_template=ct)
    dialect, stop = bfcl.detect_dialect(tok)
    assert dialect == "inkling2"
    assert stop == ["<|content_model_end_sampling|>"]

    call = {"name": "math.integrate", "arguments": {"expr": "x**2", "bounds": [0, 1]}}
    formatted = bfcl.format_tool_call(call, "inkling2")
    assert '"args":' in formatted
    assert "<|content_invoke_tool_json|>" in formatted

    with_think = "<|message_model|><|content_thinking|>let me compute<|end_message|>" + formatted
    parsed = bfcl.parse_tool_calls(with_think)
    assert len(parsed) == 1 and bfcl.calls_match(parsed, [call])

    # Missing explicit `"name"` falls back cleanly to `<|message_model|>NAME` prefix
    parsed_fallback = bfcl.parse_tool_calls('<|message_model|>calc<|content_invoke_tool_json|>{"args":{"x":7}}<|end_message|>')
    assert parsed_fallback == [{"name": "calc", "arguments": {"x": 7}}]

    assert deepmath.extract_boxed_int("<|message_model|><|content_thinking|>\\boxed{999}<|end_message|><|message_model|><|content_text|>\\boxed{42}<|end_message|>") == 42

    rendered = bfcl.render_chat_prompt(tok, [{"role": "user", "content": "Hello"}], "inkling2")
    assert "Thinking effort level: 0" in rendered

    raw_kin = (
        "<|message_model|><|content_thinking|>Let's compute final velocity.<|end_message|>"
        "<|message_model|>kinematics.final_velocity_from_distance<|content_invoke_tool_json|>"
        '{"name":"kinematics.final_velocity_from_distance","args":{"a":4,"d":300,"v0":0}}<|end_message|>'
    )
    assert bfcl.strip_thinking(raw_kin).startswith("<|message_model|>kinematics.final_velocity_from_distance")
    kin_parsed = bfcl.parse_tool_calls(raw_kin)
    assert kin_parsed == [{"arguments": {"a": 4, "d": 300, "v0": 0}, "name": "kinematics.final_velocity_from_distance"}]

    from types import SimpleNamespace
    from chantless.evaluator import Evaluator

    tc_stub = runtime.Toolchain(llama_bin=tmp_path, traj_kl=tmp_path / "traj-kl", gpus=[""])
    assert runtime.LlamaServer(tc_stub, tmp_path / "m.gguf").special is False
    assert runtime.LlamaServer(tc_stub, tmp_path / "m.gguf", special=True).special is True
    assert Evaluator(tc_stub, SimpleNamespace(dialect="inkling2"), {}, 1, 1024, "f16", tmp_path / "l1").special is True
    assert Evaluator(tc_stub, SimpleNamespace(dialect="qwen3_json"), {}, 1, 1024, "f16", tmp_path / "l2").special is False


# ------------------------------------------------------------------ 4. Issue 07 Gate (b): PrefixCache parity (< 1e-4)
def test_inkling2_prefix_cache_parity(tmp_path: Path):
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch not installed")
    import torch

    src, _, dims = _build_mini_inkling2_fixture(tmp_path)
    art = lattice.Artifact(str(src))
    try:
        stu, _ = learn.build_student(art, None, [], None, device="cpu")
    finally:
        art.close()

    seqs = [
        np.array([200, 201, 0, 1, 2], dtype=np.int64),
        np.arange(10, 23, dtype=np.int64),
        np.array([0, 1, 2, 3, 4, 5, 6] * 3, dtype=np.int64),
    ]
    with torch.no_grad():
        ref_logits = [
            learn._head_logits(stu, learn._forward_backbone(stu, [s], "cpu", False, round_to=1)[0])
            for s in seqs
        ]

    # start_layer < loop_end=4 must skip caching and return None
    assert learn.PrefixCache.build(stu, seqs, start_layer=2, device="cpu", amp=False, cfg={"free_prefix": False}, stream_prefix=False) is None

    pc_ns = learn.PrefixCache.build(stu, seqs, start_layer=4, device="cpu", amp=False, cfg={"free_prefix": False}, stream_prefix=False)
    assert pc_ns is not None
    assert pc_ns.kind == "inkling2_loop"
    for s in seqs:
        ent = pc_ns.entries[pc_ns._key(s)]
        assert ent.blocks is not None
        assert tuple(ent.blocks.shape) == (2, len(s), dims["n_embd"])

    with torch.no_grad():
        h_ns = pc_ns.forward_tail(stu, seqs, "cpu", False, round_to=32)
        assert tuple(h_ns.shape) == (3, 32, dims["n_embd"])
        for k, s in enumerate(seqs):
            lg_ns = learn._head_logits(stu, h_ns[k, : len(s)])
            diff = float((lg_ns - ref_logits[k]).abs().max().item())
            assert diff < 1e-4, (k, len(s), diff)

    pc_st = learn.PrefixCache.build(stu, seqs, start_layer=4, device="cpu", amp=False, cfg={"free_prefix": True}, stream_prefix=True)
    assert pc_st is not None
    assert pc_st.kind == "inkling2_loop"
    assert stu.model.loop_layers is None
    assert stu.model.tok_norm is None
    assert stu.model.embed_tokens is None
    for s in seqs:
        key = pc_st._key(s)
        assert float((pc_st.entries[key].h - pc_ns.entries[key].h).abs().max().item()) < 1e-5
        assert float((pc_st.entries[key].blocks - pc_ns.entries[key].blocks).abs().max().item()) < 1e-5

    with torch.no_grad():
        h_st = pc_st.forward_tail(stu, seqs, "cpu", False, round_to=32)
        for k, s in enumerate(seqs):
            lg_st = learn._head_logits(stu, h_st[k, : len(s)])
            diff = float((lg_st - ref_logits[k]).abs().max().item())
            assert diff < 1e-4, (k, len(s), diff)


# ------------------------------------------------------------------ 5. Issue 07 Gate (c1): 3D zero_rows & _apply_apprentice unit gate
def test_zero_rows_3d_moe_iq4nl_and_q4k(tmp_path: Path):
    n_embd, W, K = 256, 256, 4
    specs = [
        (0, 20, (n_embd // 32) * 18, np.array([0, 17, 255], dtype=np.int64)),   # IQ4_NL
        (1, 12, (n_embd // 256) * 144, np.array([3, 64, 128, 200], dtype=np.int64)),  # Q4_K
    ]
    rng = np.random.default_rng(20261121)
    tensors = []
    for il, ttype, rbytes, _dead in specs:
        nbytes = K * (2 * W) * rbytes
        tensors.append(
            (f"blk.{il}.ffn_gate_up_exps.weight", [n_embd, 2 * W, K], ttype, rng.integers(1, 255, size=nbytes, dtype=np.uint8).tobytes())
        )
    src = tmp_path / "moe3d.gguf"
    _write_gguf(src, [("general.architecture", _kv_str("inkling2_loop"))], tensors)

    dst_zr = tmp_path / "moe3d_zr.gguf"
    dst_app = tmp_path / "moe3d_app.gguf"
    lattice.zero_rows(
        src,
        dst_zr,
        {f"blk.{il}.ffn_gate_up_exps.weight": np.concatenate([dead, W + dead]) for il, _, _, dead in specs},
    )
    builder._apply_apprentice(
        src,
        dst_app,
        {
            "switch_mode": "up_row_zero",
            "switches": {str(il): mk.pack_bitmap(dead, W) for il, _, _, dead in specs},
            "levels": {},
        },
    )

    assert dst_zr.stat().st_size == src.stat().st_size
    assert util.sha256_file(dst_zr) == util.sha256_file(dst_app)

    art_s, art_d = lattice.Artifact(src), lattice.Artifact(dst_zr)
    try:
        for il, _ttype, rbytes, dead in specs:
            name = f"blk.{il}.ffn_gate_up_exps.weight"
            raw_s = np.frombuffer(bytes(art_s.raw(name)), dtype=np.uint8).reshape(K, 2 * W, rbytes)
            raw_d = np.frombuffer(bytes(art_d.raw(name)), dtype=np.uint8).reshape(K, 2 * W, rbytes)
            dead_set = set(dead.tolist()) | set((W + dead).tolist())
            for r in range(2 * W):
                if r in dead_set:
                    assert np.all(raw_d[:, r, :] == 0)
                else:
                    assert np.array_equal(raw_d[:, r, :], raw_s[:, r, :])
    finally:
        art_s.close()
        art_d.close()

    with pytest.raises(ValueError, match="out of range"):
        lattice.zero_rows(src, tmp_path / "bad_idx.gguf", {"blk.0.ffn_gate_up_exps.weight": np.array([2 * W])})

    src_4d = tmp_path / "bad_rank4.gguf"
    _write_gguf(src_4d, [("general.architecture", _kv_str("inkling2_loop"))], [("t4", [2, 2, 2, 2], 0, b"\x01" * (16 * 4))])
    with pytest.raises(ValueError, match="2D or 3D"):
        lattice.zero_rows(src_4d, tmp_path / "bad_4d_out.gguf", {"t4": np.array([0])})


# ------------------------------------------------------------------ 6. Issue 07 Gate (c2): learn_gates_seq -> Builder.build SHA256 & switched NLL parity
def test_inkling2_learn_gates_seq_sha256_and_switched_nll_parity(tmp_path: Path):
    tc = _resolve_inkling2_toolchain()
    if tc is None:
        pytest.skip("local .local/toolchain-inkling2 PR #32704 toolchain not built")
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch not installed")

    src, im, dims = _build_mini_inkling2_fixture(tmp_path)
    # Give routers non-zero separation in exp_probs_b.bias so Metal FP16 FA and CPU FP32 do not flip near-tie expert picks
    art_sep = lattice.Artifact(str(src))
    bias_updates = []
    for tname, ti in art_sep.tensors.items():
        if "exp_probs_b.bias" in tname:
            bvals = np.arange(ti["dims"][0], dtype=np.float32) * 5.0
            bias_updates.append((art_sep.h.data_start + ti["offset"], bvals.tobytes()))
    art_sep.close()
    with open(src, "r+b") as bf:
        for off_pos, bbytes in bias_updates:
            bf.seek(off_pos)
            bf.write(bbytes)

    groups, _ = gguf_ops.adapter_from_gguf(src)
    exp_ord = gguf_ops.expert_importance_from_imatrix(src, im)
    kept_exp = {il: np.sort(exp_ord[il][:4]) for il in range(dims["n_layer"])}
    w_ord = gguf_ops.wanda_order(src, im, kept_experts=kept_exp)

    mask_b = mk.new_mask(groups, "Q8_0", {"repo": "test", "revision": "0", "gguf_sha256": "0"})
    for il in range(dims["n_layer"]):
        mask_b["groups"][f"blk.{il}.ffn"]["tier"] = "Q4_K"
    mask_b["experts"] = {str(il): mk.pack_bitmap(kept_exp[il], dims["n_expert"]) for il in range(dims["n_layer"])}
    mask_b["widths"] = {str(il): mk.pack_bitmap(w_ord[il][:256], dims["n_ff_exp"]) for il in range(dims["n_layer"])}

    out_b = tmp_path / "cand_q4k.gguf"
    B = builder.Builder(tc, str(src), groups, str(im), tmp_path / "work", threads=4, quantize_shards=1)
    B.build(mask_b, out_b)

    lessons = [
        {"prompt_tokens": [200, 10, 11, 12, 13, 14], "corrected_tokens": [30, 31, 32, 33], "mistaken_tokens": [60, 61, 62, 63]},
        {"prompt_tokens": [201, 20, 21, 22, 23, 24], "corrected_tokens": [70, 71, 72, 73], "mistaken_tokens": [40, 41, 42, 43]},
    ]
    replay_recs = [
        {"tokens": [0, 1, 2, 3, 4, 5, 6, 7], "targets": [3, 4, 5, 6, 7]},
        {"tokens": list(range(110, 131)), "targets": list(range(10, 21))},
    ]

    sw_gg = tmp_path / "switched.gguf"
    sw_mask_p = tmp_path / "switched.mask.json"
    rec = learn.learn_gates_seq(
        str(out_b),
        None,
        lessons,
        replay_recs,
        str(sw_gg),
        cfg={
            "ggml_lib": str(tc.llama_bin),
            "steps": 40,
            "gate_lr": 0.15,
            "gate_init": 0.25,
            "margin": 1.0,
            "nll_weight": 1.0,
            "margin_weight": 2.0,
            "replay_weight": 1.0,
            "project": True,
            "gate_last_blocks": 2,
            "switch_mode": "up_row_zero",
        },
        device="cpu",
        mask_in=mask_b,
        mask_out=sw_mask_p,
    )
    assert rec["off"] > 0
    assert rec["bytes"] == out_b.stat().st_size
    assert len(rec["mask"]["apprentice"]["switches"]) >= 1
    assert all(sp["n"] == 256 for sp in rec["mask"]["apprentice"]["switches"].values())

    reb_gg = tmp_path / "rebuilt.gguf"
    builder.Builder(tc, str(src), groups, str(im), tmp_path / "work_reb", threads=4, quantize_shards=1).build(
        mk.load(sw_mask_p), reb_gg
    )
    assert util.sha256_file(reb_gg) == util.sha256_file(sw_gg)

    eval_trj = tmp_path / "eval_sw.trj"
    _write_trj1(
        eval_trj,
        [
            [200, 10, 11, 12, 13, 14, 30, 31, 32, 33],
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 200, 201, 0, 1, 2, 3, 200, 201, dims["eos_id"]],
        ],
    )
    store_sw = tmp_path / "store_sw"
    store_sw.mkdir()
    subprocess.run(
        [str(tc.traj_kl), "--mode", "base", "-m", str(out_b), "--traj", str(eval_trj), "--store", str(store_sw), "--threads", "2"],
        env=tc.env_for(""), capture_output=True, text=True, check=True,
    )
    proc_sw = subprocess.run(
        [str(tc.traj_kl), "--mode", "kl", "-m", str(reb_gg), "--traj", str(eval_trj), "--store", str(store_sw), "--threads", "2"],
        env=tc.env_for(""), capture_output=True, text=True, check=True,
    )
    rep_sw = json.loads(proc_sw.stdout)

    art_reb = lattice.Artifact(str(reb_gg))
    try:
        stu_reb, _ = learn.build_student(art_reb, None, [], str(tc.llama_bin), device="cpu")
    finally:
        art_reb.close()
    stu_trajs = learn.student_traj_nll(stu_reb, learn.read_trj_multi(eval_trj), device="cpu")
    max_diff = max(abs(float(s["nll_cand"]) - float(c["nll_cand"])) for s, c in zip(stu_trajs, rep_sw["trajs"]))
    assert max_diff < 5e-3, (max_diff, stu_trajs, rep_sw["trajs"])


# ------------------------------------------------------------------ 7. Issue 07 Gate (d): 4-control protocol + config_dir=None
def test_inkling2_learn_generative_four_controls_and_sha256(tmp_path: Path):
    tc = _resolve_inkling2_toolchain()
    if tc is None:
        pytest.skip("local .local/toolchain-inkling2 PR #32704 toolchain not built")
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch not installed")

    src, im, dims = _build_mini_inkling2_fixture(tmp_path)
    groups, _ = gguf_ops.adapter_from_gguf(src)
    exp_ord = gguf_ops.expert_importance_from_imatrix(src, im)
    kept_exp = {il: np.sort(exp_ord[il][:4]) for il in range(dims["n_layer"])}
    w_ord = gguf_ops.wanda_order(src, im, kept_experts=kept_exp)

    mask_b = mk.new_mask(groups, "Q4_K", {"repo": "test", "revision": "0", "gguf_sha256": "0"})
    mask_b["groups"]["token_embd"]["tier"] = "Q8_0"
    mask_b["experts"] = {str(il): mk.pack_bitmap(kept_exp[il], dims["n_expert"]) for il in range(dims["n_layer"])}
    mask_b["widths"] = {str(il): mk.pack_bitmap(w_ord[il][:256], dims["n_ff_exp"]) for il in range(dims["n_layer"])}

    out_b = tmp_path / "base_q4k.gguf"
    builder.Builder(tc, str(src), groups, str(im), tmp_path / "work", threads=4, quantize_shards=1).build(mask_b, out_b)
    mask_b_path = tmp_path / "base_q4k.mask.json"
    mask_b_path.write_text(json.dumps(mask_b))

    lessons_path = tmp_path / "lessons.jsonl"
    lessons_rows = [
        {"prompt_tokens": [200, 10, 11, 12, 13, 14], "corrected_tokens": [30, 31, 32, 33], "mistaken_tokens": [60, 61, 62, 63]},
        {"prompt_tokens": [201, 20, 21, 22, 23, 24], "corrected_tokens": [70, 71, 72, 73], "mistaken_tokens": [40, 41, 42, 43]},
        {"prompt_tokens": [5, 80, 81, 82, 83, 84], "corrected_tokens": [90, 91, 92, 93], "mistaken_tokens": [100, 101, 102, 103]},
    ]
    lessons_path.write_text("\n".join(json.dumps(r) for r in lessons_rows) + "\n")

    replay_path = tmp_path / "replay.jsonl"
    replay_rows = [
        {"tokens": [0, 1, 2, 3, 4, 5, 6, 7], "targets": [3, 4, 5, 6, 7]},
        {"tokens": list(range(110, 131)), "targets": list(range(10, 21))},
    ]
    replay_path.write_text("\n".join(json.dumps(r) for r in replay_rows) + "\n")

    out_dir = tmp_path / "learn_out"
    cfg_obj = cfgmod.LearnConfig.model_validate(
        {
            "toolchain": {
                "llama_bin": str(tc.llama_bin),
                "traj_kl": str(tc.traj_kl),
                "gpus": [""],
            },
            "learn": {
                "mode": "generative",
                "model": str(out_b),
                "config_dir": None,
                "source_gguf": str(src),
                "imatrix": str(im),
                "mask_in": str(mask_b_path),
                "mask_out": str(out_dir / "learned.mask.json"),
                "out_dir": str(out_dir),
                "lessons": str(lessons_path),
                "replay": str(replay_path),
                "controls": True,
                "gate_last_blocks": 2,
                "switch_mode": "up_row_zero",
                "steps": 40,
                "gate_lr": 0.15,
                "gate_init": 0.25,
                "margin": 1.0,
                "nll_weight": 1.0,
                "margin_weight": 2.0,
                "replay_weight": 1.0,
                "project": True,
                "amp": False,
                "ggml_lib": str(tc.llama_bin),
            },
        }
    )
    raw_cfg = cfg_obj.to_raw_dict()
    assert "config_dir" not in raw_cfg["learn"]

    res = learn.run_generative(raw_cfg)
    assert res["accepted"] is True
    assert res["lineage"]["off"] > 0
    assert set(res["cpp_traj_kl"]) == {"base", "learned", "empty", "flipped", "shuffled"}
    learned_margin = float(res["cpp_traj_kl"]["learned"]["margin_mean"])
    for ctrl in ("base", "empty", "flipped", "shuffled"):
        ctrl_margin = float(res["cpp_traj_kl"][ctrl]["margin_mean"])
        assert learned_margin > ctrl_margin, (ctrl, learned_margin, ctrl_margin, res["cpp_traj_kl"])


# ------------------------------------------------------------------ 8. Issue 07 Gate (e): plan.estimate & cli plan drop_moves == 0
def test_inkling2_plan_zero_drop_moves(tmp_path: Path):
    src, _, _ = _build_mini_inkling2_fixture(tmp_path)
    raw_cfg = cfgmod.load_config(ROOT / "mvp/configs/inkling2-mini-bfcl.yaml", "extract").to_raw_dict()
    raw_cfg["run_root"] = str(tmp_path / "runs")
    raw_cfg["source"]["gguf"] = str(src)

    est = plan.estimate(raw_cfg, source=str(src))
    assert est["moves"]["drop_moves"] == 0
    assert est["moves"]["legal_per_sweep"] == 14

    cfg_copy = tmp_path / "inkling2-plan.yaml"
    cfg_copy.write_text(
        (ROOT / "mvp/configs/inkling2-mini-bfcl.yaml")
        .read_text(encoding="utf-8")
        .replace("run_root: .local/runs", f"run_root: {tmp_path / 'runs'}"),
        encoding="utf-8",
    )
    proc = subprocess.run(
        [sys.executable, "-m", "chantless.cli", "plan", "--config", str(cfg_copy), "--source", str(src)],
        cwd=str(ROOT),
        env={**os.environ, "PYTHONPATH": str(ROOT / "mvp")},
        capture_output=True,
        text=True,
        check=True,
    )
    assert "14 groups" in proc.stdout
    assert "legal per sweep 14" in proc.stdout


# ------------------------------------------------------------------ 9. Gorilla BFCL AST rules & Inkling-2 FC prompt/config defaults
def test_bfcl_gorilla_ast_matching_and_inkling2_prompt_defaults(tmp_path: Path):
    # Optional `""` omission (`simple_104`, `live_simple_58`)
    assert bfcl.calls_match(
        [{"name": "geometry.area_triangle", "arguments": {"base": 6, "height": 10}}],
        [{"name": "geometry.area_triangle", "arguments": {"base": 6, "height": 10, "unit": ""}}],
    )
    assert bfcl.calls_match(
        [{"name": "get_movies", "arguments": {"city": "Mumbai"}}],
        [{"name": "get_movies", "arguments": {"city": "Mumbai", "cinema_hall": "", "movie_date": ""}}],
    )
    # Non-optional key still fails if omitted
    assert not bfcl.calls_match(
        [{"name": "geometry.area_triangle", "arguments": {"base": 6}}],
        [{"name": "geometry.area_triangle", "arguments": {"base": 6, "height": 10, "unit": ""}}],
    )
    # Nested dict option list + Gorilla standardize_string
    assert bfcl.calls_match(
        [{"name": "ChaDri.change_drink", "arguments": {"drink_id": "latte", "new_preferences": {"milk_type": "Coconut", "special_instructions": "served boiling-hot"}}}],
        [{"name": "ChaDri.change_drink", "arguments": {"drink_id": "latte", "new_preferences": {"milk_type": ["coconut"], "special_instructions": ["served boiling hot", "boiling hot"], "size": [""]}}}],
    )
    # Java/JS stringified literal matching (`5000L`, `3.5f`, `true`, `new ArrayList<>(Arrays.asList(...))`, JS dict)
    assert bfcl.calls_match(
        [{"name": "Foo.run", "arguments": {"n": "5000L", "rate": "3.5f", "flag": "true", "xs": 'new ArrayList<String>(Arrays.asList("a", "b"))'}}],
        [{"name": "Foo.run", "arguments": {"n": 5000, "rate": 3.5, "flag": True, "xs": ["a", "b"]}}],
    )
    assert bfcl.calls_match(
        [{"name": "EventScheduler", "arguments": {"events": "{setupStage: setupStageFunction, cleanupStage: cleanStageFunction}", "concurrencyLimit": "3.0"}}],
        [{"name": "EventScheduler", "arguments": {"events": {"setupStage": ["setupStageFunction"], "cleanupStage": ["cleanStageFunction"]}, "concurrencyLimit": 3.0}}],
    )
    # Dotted Inkling-2 prefix name preserved when model shortens `"name"` inside JSON (`java_11`)
    dotted_txt = '<|message_model|>BasePolicyDataProvider.getRegistryPolicyValue<|content_invoke_tool_json|>{"name":"BasePolicyDataProvider","args":{"property":"EnableExtensions","root":"WinReg.HKEY_LOCAL_MACHINE"}}<|end_message|>'
    parsed_dotted = bfcl.parse_tool_calls(dotted_txt)
    assert parsed_dotted == [{"name": "BasePolicyDataProvider.getRegistryPolicyValue", "arguments": {"property": "EnableExtensions", "root": "WinReg.HKEY_LOCAL_MACHINE"}}]
    parsed_allowed = bfcl.parse_tool_calls(dotted_txt, allowed_names={"BasePolicyDataProvider.getRegistryPolicyValue"})
    assert parsed_allowed[0]["name"] == "BasePolicyDataProvider.getRegistryPolicyValue"

    # Gorilla `func_doc_language_specific_pre_processing`: Java/JS params become `string` with per-param hints
    java_raw = [{
        "type": "function",
        "function": {
            "name": "Basic.checkSizes",
            "description": "Checks map sizes.",
            "parameters": {
                "type": "dict",
                "properties": {
                    "environ": {"type": "HashMap", "description": "Env map."},
                    "items": {"type": "ArrayList", "items": {"type": "String"}, "description": "Items."},
                    "obj": {"type": "any", "description": "Obj."},
                },
                "required": ["environ", "items", "obj"],
            },
        },
    }]
    java_hinted = bfcl._apply_gorilla_lang_hints(java_raw, "java", raw_tools=java_raw)
    jp = java_hinted[0]["function"]["parameters"]
    assert jp["type"] == "object"
    assert jp["properties"]["environ"]["type"] == "string"
    assert "Java HashMap type parameter in string representation." in jp["properties"]["environ"]["description"]
    assert jp["properties"]["items"]["type"] == "string" and "items" not in jp["properties"]["items"]
    assert "Java ArrayList type parameter in string representation." in jp["properties"]["items"]["description"]
    assert "The list elements are of type String; they are not in string representation." in jp["properties"]["items"]["description"]
    assert "any type of Java object in string representation." in jp["properties"]["obj"]["description"]

    js_raw = [{
        "type": "function",
        "function": {
            "name": "prioritizeAndSort",
            "description": "Sorts items.",
            "parameters": {
                "type": "dict",
                "properties": {
                    "items": {"type": "array", "items": {"type": "String"}, "description": "Array."},
                    "cfg": {"type": "dict", "properties": {"k": {"type": "String"}}, "description": "Cfg."},
                },
                "required": ["items"],
            },
        },
    }]
    js_hinted = bfcl._apply_gorilla_lang_hints(js_raw, "javascript", raw_tools=js_raw)
    jsp = js_hinted[0]["function"]["parameters"]["properties"]
    assert jsp["items"]["type"] == "string" and "items" not in jsp["items"]
    assert "JavaScript array type parameter in string representation." in jsp["items"]["description"]
    assert jsp["cfg"]["type"] == "string" and "properties" not in jsp["cfg"]
    assert "The dictionary entries have the following schema; they are not in string representation." in jsp["cfg"]["description"]

    # LlamaServer stores configurable `flash_attn`
    tc_dummy = runtime.Toolchain(Path("/tmp"), Path("/tmp/traj-kl"), ["0"])
    srv_fa_off = runtime.LlamaServer(tc_dummy, "/tmp/m.gguf", flash_attn="off")
    assert srv_fa_off.flash_attn == "off"

    # Inkling-2 auto-enables Gorilla FC system prompt + language hints while Qwen3 stays untouched
    from tokenizers import Tokenizer, models as tok_models
    from transformers import PreTrainedTokenizerFast

    ink_dir = tmp_path / "hf_ink"
    ink_dir.mkdir()
    ink_tok = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(tok_models.BPE(vocab={
            "x": 0, "<|message_system|>": 1, "<|message_user|>": 2, "<|message_model|>": 3,
            "<|content_text|>": 4, "<|content_xml|>": 5, "<|end_message|>": 6,
            "<|content_invoke_tool_json|>": 7, "<|content_model_end_sampling|>": 8,
        }, merges=[])),
        chat_template=(
            "{%- for m in messages -%}{{- m.role ~ ':' ~ m.content ~ '\n' -}}{%- endfor -%}"
            "{%- if tools -%}TOOLS:{{ tools[0].function.description }}{%- endif -%}"
        ),
    )
    ink_tok.save_pretrained(ink_dir)

    q_dir = tmp_path / "hf_qwen"
    q_dir.mkdir()
    q_tok = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(tok_models.BPE(vocab={"x": 0, "<|im_end|>": 1}, merges=[])),
        chat_template="{%- for m in messages -%}{{- m.role ~ ':' ~ m.content ~ '\n' -}}{%- endfor -%}",
    )
    q_tok.save_pretrained(q_dir)

    pairs = tmp_path / "pairs.jsonl"
    pairs.write_text(json.dumps({
        "id": "java:j0",
        "category": "java",
        "family": "java/b00",
        "split": "select",
        "messages": [{"role": "user", "content": "How to decode?"}],
        "tools": [{"type": "function", "function": {"name": "Foo.bar", "description": "Decodes input.", "parameters": {"type": "object", "properties": {}, "required": []}}}],
        "reference_calls": [{"name": "Foo.bar", "arguments": {}}],
    }) + "\n")

    ink_env = bfcl.BfclEnvironment(hf_dir=str(ink_dir), pairs_file=str(pairs))
    assert ink_env.system_prompt == bfcl.DEFAULT_INKLING2_TOOL_SYSTEM_PROMPT
    assert ink_env.lang_hints is True
    ink_p = ink_env.render_prompt(ink_env.tasks()[0])
    assert "expert in composing functions" in ink_p
    assert "Java 8 SDK syntax" in ink_p

    q_env = bfcl.BfclEnvironment(hf_dir=str(q_dir), pairs_file=str(pairs))
    assert q_env.system_prompt is None
    assert q_env.lang_hints is False
    assert "expert in composing functions" not in q_env.render_prompt(q_env.tasks()[0])

    # Configs validate with official temperature=1.0 and cache_prompt=false
    bfcl_cfg = cfgmod.load_config(ROOT / "mvp/configs/inkling2-mini-bfcl.yaml", "extract")
    dm_cfg = cfgmod.load_config(ROOT / "mvp/configs/inkling2-mini-deepmath.yaml", "extract")
    assert bfcl_cfg.decoding.temperature == 1.0 and bfcl_cfg.serving.cache_prompt is False
    assert dm_cfg.decoding.temperature == 1.0 and dm_cfg.serving.cache_prompt is False
    assert dm_cfg.env.kwargs["reasoning_effort"] == "low" and dm_cfg.env.kwargs["max_tokens"] == 4096


def test_shipped_inkling2_masks_validate_v3_schema():
    expected = {
        "inkling2-mini-clsbench-mask.json": {"experts": 48, "widths": 24, "vocab_keep": 32768, "switches": 0},
        "inkling2-mini-clsbench-apprentice-mask.json": {"experts": 48, "widths": 24, "vocab_keep": 32768, "switches": 424},
        "inkling2-mini-bfcl-mask.json": {"experts": 0, "widths": 0, "vocab_keep": None, "switches": 0},
        "inkling2-mini-deepmath-mask.json": {"experts": 96, "widths": 0, "vocab_keep": None, "switches": 0},
    }
    groups = {"token_embd": {"droppable": False}, "output": {"droppable": False}}
    for il in range(24):
        groups[f"blk.{il}.attn"] = {"droppable": False}
        groups[f"blk.{il}.ffn"] = {"droppable": False}

    for fn, spec in expected.items():
        m = json.loads((ROOT / "masks" / fn).read_text(encoding="utf-8"))
        assert m["schema"] == "chantless.mask/v3"
        assert m["source"]["repo"] == "thinkingmachines/Inkling-2-Mini"
        assert m["source"]["gguf_sha256"] == "7074c9762e6769a608c7b23960a7fd2444880b1174a3910149749f6d01389aec"
        assert m["weights"] == {"lineage": []}
        assert m["deployment"]["kv_type"] == "f16"
        assert m["measured"]["file_bytes"] > 0
        assert len(m["measured"]["built_sha256"]) == 64
        mk.validate(m, groups, ladder=mk.ALL_TIERS)
        if spec["experts"]:
            kept = mk.kept_experts(m)
            assert len(kept) == 24 and all(len(v) == spec["experts"] for v in kept.values())
        else:
            assert not m.get("experts")
        assert len(m.get("widths") or {}) == spec["widths"]
        if spec["vocab_keep"] is None:
            assert m.get("vocab_keep") is None
        else:
            assert m["vocab_keep"]["count"] == spec["vocab_keep"]
        sw = (m.get("apprentice") or {}).get("switches") or {}
        total_sw = sum(len(mk.unpack_bitmap(s)) for s in sw.values())
        assert total_sw == spec["switches"]
        if spec["switches"]:
            assert m["apprentice"]["switch_mode"] == "up_row_zero"
            assert m["measured"]["cli_rebuild_sha256_equal"] is True
            assert m["measured"]["apprentice_eval"]["beats_all_controls"] is True


# ------------------------------------------------------------------ 10. Optional `/v1/chat/completions` BFCL wiring
def _make_bfcl_loopback_server(handler_fn):
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

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
            status, payload = handler_fn(self.path, body, idx)
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


def _build_bfcl_dialect_hf_dir(out_dir: Path, dialect: str) -> Path:
    from tokenizers import Tokenizer, decoders, models as tok_models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    out_dir.mkdir(parents=True, exist_ok=True)
    alphabet = list(pre_tokenizers.ByteLevel.alphabet())
    vocab_map = {tok: i for i, tok in enumerate(alphabet)}
    if dialect == "inkling2":
        specials = [
            "<|endoftext|>",
            "<|message_system|>",
            "<|message_user|>",
            "<|message_model|>",
            "<|message_tool|>",
            "<|content_text|>",
            "<|content_thinking|>",
            "<|content_xml|>",
            "<|content_invoke_tool_json|>",
            "<|end_message|>",
            "<|content_model_end_sampling|>",
        ]
        ct = (
            "{%- set eff = reasoning_effort if reasoning_effort is defined and reasoning_effort is not none else 0.9 -%}"
            "{%- set effort_num = '0' if eff == 'none' else ('0.2' if eff == 'low' else ('0.9' if eff == 'high' else eff)) -%}"
            "{%- if tools -%}"
            "{{- '<|message_system|>tool_declare<|content_xml|>' ~ (tools | tojson) ~ '<|end_message|>' -}}"
            "{%- endif -%}"
            "{%- for m in messages -%}"
            "{%- if m.role == 'system' -%}"
            "{{- '<|message_system|><|content_text|>' ~ m.content ~ '<|end_message|>' -}}"
            "{{- '<|message_system|><|content_text|>Thinking effort level: ' ~ effort_num ~ '<|end_message|>' -}}"
            "{%- elif m.role == 'user' -%}"
            "{{- '<|message_user|><|content_text|>' ~ m.content ~ '<|end_message|>' -}}"
            "{%- elif m.role == 'assistant' -%}"
            "{%- if m.reasoning_content is defined and m.reasoning_content -%}"
            "{{- '<|message_model|><|content_thinking|>' ~ m.reasoning_content ~ '<|end_message|>' -}}"
            "{%- endif -%}"
            "{%- if m.tool_calls -%}"
            "{%- for tc in m.tool_calls -%}"
            "{{- '<|message_model|>' ~ tc.function.name ~ '<|content_invoke_tool_json|>' ~ ({'name': tc.function.name, 'args': tc.function.arguments} | tojson) ~ '<|end_message|>' -}}"
            "{%- endfor -%}"
            "{{- '<|content_model_end_sampling|>' -}}"
            "{%- else -%}"
            "{{- '<|message_model|><|content_text|>' ~ (m.content or '') ~ '<|end_message|><|content_model_end_sampling|>' -}}"
            "{%- endif -%}"
            "{%- endif -%}"
            "{%- endfor -%}"
        )
        eos = "<|content_model_end_sampling|>"
    elif dialect == "gemma4":
        specials = ["<|endoftext|>", "<|tool_call>", "<tool_call|>", "<turn|>", "<end_of_turn>", "<|tool_response>"]
        ct = "{%- for m in messages -%}{{- m.role ~ ':' ~ (m.content or '') ~ '<turn|>' -}}{%- endfor -%}"
        eos = "<turn|>"
    elif dialect == "qwen35_xml":
        specials = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]
        ct = "{%- for m in messages -%}{{- '<|im_start|>' ~ m.role ~ '\n' ~ (m.content or '') ~ '<|im_end|>\n' -}}{%- endfor -%}{%- if tools -%}<function=example>{%- endif -%}"
        eos = "<|im_end|>"
    else:
        specials = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]
        ct = "{%- for m in messages -%}{{- '<|im_start|>' ~ m.role ~ '\n' ~ (m.content or '') ~ '<|im_end|>\n' -}}{%- endfor -%}"
        eos = "<|im_end|>"

    for sp in specials:
        if sp not in vocab_map:
            vocab_map[sp] = len(vocab_map)
    tk = Tokenizer(tok_models.BPE(vocab=vocab_map, merges=[]))
    tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tk.decoder = decoders.ByteLevel()
    hf_tok = PreTrainedTokenizerFast(tokenizer_object=tk, eos_token=eos, pad_token="<|endoftext|>", chat_template=ct)
    hf_tok.add_special_tokens({"additional_special_tokens": specials})
    hf_tok.save_pretrained(out_dir)
    return out_dir


def test_bfcl_chat_completions_inkling2_wire_payload_and_grading(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from types import SimpleNamespace
    import urllib.request
    from chantless import kl
    from chantless.evaluator import Evaluator

    ink_dir = _build_bfcl_dialect_hf_dir(tmp_path / "hf_ink", "inkling2")
    java_raw_tools = [{
        "type": "function",
        "function": {
            "name": "BasePolicyDataProvider.getRegistryPolicyValue",
            "description": "Reads a registry policy.",
            "parameters": {
                "type": "dict",
                "properties": {
                    "root": {"type": "String", "description": "Registry root."},
                    "property": {"type": "String", "description": "Policy property."},
                },
                "required": ["root", "property"],
            },
        },
    }]
    pairs_rows = [
        {
            "id": "simple:s0",
            "category": "simple",
            "family": "simple/b00",
            "split": "select",
            "messages": [{"role": "user", "content": "Area of triangle base 6 height 10?"}],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "geometry.area_triangle",
                    "description": "Compute triangle area.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "base": {"type": "integer"},
                            "height": {"type": "integer"},
                            "unit": {"type": "string"},
                        },
                        "required": ["base", "height"],
                    },
                },
            }],
            "reference_calls": [{"name": "geometry.area_triangle", "arguments": {"base": 6, "height": 10, "unit": ""}}],
        },
        {
            "id": "java:j0",
            "category": "java",
            "family": "java/b00",
            "split": "select",
            "messages": [{"role": "user", "content": "Read EnableExtensions from HKLM."}],
            "tools": [bfcl._norm_schema(java_raw_tools[0])],
            "raw_tools": java_raw_tools,
            "reference_calls": [{
                "name": "BasePolicyDataProvider.getRegistryPolicyValue",
                "arguments": {"property": "EnableExtensions", "root": "WinReg.HKEY_LOCAL_MACHINE"},
            }],
        },
        {
            "id": "simple:s1_bad_json",
            "category": "simple",
            "family": "simple/b01",
            "split": "select",
            "messages": [{"role": "user", "content": "Bare JSON text case."}],
            "tools": [{"type": "function", "function": {"name": "ping", "description": "Ping.", "parameters": {"type": "object", "properties": {}}}}],
            "reference_calls": [{"name": "ping", "arguments": {}}],
        },
        {
            "id": "simple:s2_len",
            "category": "simple",
            "family": "simple/b02",
            "split": "select",
            "messages": [{"role": "user", "content": "Truncated length case."}],
            "tools": [{"type": "function", "function": {"name": "ping", "description": "Ping.", "parameters": {"type": "object", "properties": {}}}}],
            "reference_calls": [{"name": "ping", "arguments": {}}],
        },
        {
            "id": "simple:s3_garb",
            "category": "simple",
            "family": "simple/b03",
            "split": "select",
            "messages": [{"role": "user", "content": "Garbage reasoning case."}],
            "tools": [{"type": "function", "function": {"name": "ping", "description": "Ping.", "parameters": {"type": "object", "properties": {}}}}],
            "reference_calls": [{"name": "ping", "arguments": {}}],
        },
        {
            "id": "simple:s4_infra",
            "category": "simple",
            "family": "simple/b04",
            "split": "select",
            "messages": [{"role": "user", "content": "Server 500 case."}],
            "tools": [{"type": "function", "function": {"name": "ping", "description": "Ping.", "parameters": {"type": "object", "properties": {}}}}],
            "reference_calls": [{"name": "ping", "arguments": {}}],
        },
        {
            "id": "simple:s5_unknown",
            "category": "simple",
            "family": "simple/b05",
            "split": "select",
            "messages": [{"role": "user", "content": "Unknown tool case."}],
            "tools": [{"type": "function", "function": {"name": "ping", "description": "Ping.", "parameters": {"type": "object", "properties": {}}}}],
            "reference_calls": [{"name": "ping", "arguments": {}}],
        },
        {
            "id": "simple:s6_bad_args",
            "category": "simple",
            "family": "simple/b06",
            "split": "select",
            "messages": [{"role": "user", "content": "Malformed tool args case."}],
            "tools": [{"type": "function", "function": {"name": "ping", "description": "Ping.", "parameters": {"type": "object", "properties": {}}}}],
            "reference_calls": [{"name": "ping", "arguments": {}}],
        },
        {
            "id": "simple:s7_loop",
            "category": "simple",
            "family": "simple/b07",
            "split": "select",
            "messages": [{"role": "user", "content": "Doom loop reasoning case."}],
            "tools": [{"type": "function", "function": {"name": "ping", "description": "Ping.", "parameters": {"type": "object", "properties": {}}}}],
            "reference_calls": [{"name": "ping", "arguments": {}}],
        },
    ]
    pairs_p = tmp_path / "bfcl_pairs.jsonl"
    pairs_p.write_text("\n".join(json.dumps(r) for r in pairs_rows) + "\n", encoding="utf-8")

    seen_timeouts: list[float] = []
    orig_urlopen = urllib.request.urlopen

    def spy_urlopen(req, *args, **kwargs):
        if "timeout" in kwargs:
            seen_timeouts.append(kwargs["timeout"])
        return orig_urlopen(req, *args, **kwargs)

    monkeypatch.setattr(urllib.request, "urlopen", spy_urlopen)
    monkeypatch.setattr("chantless.client.time.sleep", lambda _s: None)

    def responder(path, body, idx):
        user_txt = body["messages"][-1]["content"]
        if "Area of triangle" in user_txt:
            return 200, {
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "reasoning_content": "Triangle area needs base=6 and height=10.",
                        "tool_calls": [{
                            "id": "call_tri_1",
                            "type": "function",
                            "function": {
                                "name": "geometry.area_triangle",
                                "arguments": '{"base": "6", "height": 10}',
                            },
                        }],
                    },
                    "finish_reason": "tool_calls",
                }],
                "usage": {"completion_tokens": 27},
            }
        if "Read EnableExtensions" in user_txt:
            # Fallback to Inkling-2 wire tag in `content` when `tool_calls` is empty
            return 200, {
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": (
                            "<|message_model|>BasePolicyDataProvider.getRegistryPolicyValue"
                            "<|content_invoke_tool_json|>"
                            '{"name":"BasePolicyDataProvider","args":{"property":"EnableExtensions","root":"WinReg.HKEY_LOCAL_MACHINE"}}'
                            "<|end_message|>"
                        ),
                        "tool_calls": [],
                    },
                    "finish_reason": "stop",
                }],
                "usage": {"completion_tokens": 31},
            }
        if "Bare JSON text case" in user_txt:
            # Plain-text JSON without `<|content_invoke_tool_json|>` must NOT pass on Inkling-2
            return 200, {
                "choices": [{
                    "message": {"role": "assistant", "content": '{"name": "ping", "arguments": {}}'},
                    "finish_reason": "stop",
                }],
                "usage": {"completion_tokens": 9},
            }
        if "Truncated length case" in user_txt:
            return 200, {
                "choices": [{
                    "message": {"role": "assistant", "content": "", "reasoning_content": "still thinking..."},
                    "finish_reason": "length",
                }],
                "usage": {"completion_tokens": 1024},
            }
        if "Garbage reasoning case" in user_txt:
            return 200, {
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "reasoning_content": "bad utf8 \ufffd token",
                        "tool_calls": [{"id": "c_g", "type": "function", "function": {"name": "ping", "arguments": "{}"}}],
                    },
                    "finish_reason": "tool_calls",
                }],
                "usage": {"completion_tokens": 14},
            }
        if "Unknown tool case" in user_txt:
            return 200, {
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [{"id": "c_u", "type": "function", "function": {"name": "not_a_tool", "arguments": "{}"}}],
                    },
                    "finish_reason": "tool_calls",
                }],
                "usage": {"completion_tokens": 8},
            }
        if "Malformed tool args case" in user_txt:
            return 200, {
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [{"id": "c_m", "type": "function", "function": {"name": "ping", "arguments": "{broken"}}],
                    },
                    "finish_reason": "tool_calls",
                }],
                "usage": {"completion_tokens": 8},
            }
        if "Doom loop reasoning case" in user_txt:
            span = " ".join(f"tok{k}" for k in range(16))
            return 200, {
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "reasoning_content": " ".join([span] * 4),
                        "tool_calls": [{"id": "c_l", "type": "function", "function": {"name": "ping", "arguments": "{}"}}],
                    },
                    "finish_reason": "tool_calls",
                }],
                "usage": {"completion_tokens": 80},
            }
        return 500, {"error": "grammar error"}

    srv, th, url, recs = _make_bfcl_loopback_server(responder)
    tc = runtime.Toolchain(llama_bin=tmp_path, traj_kl=tmp_path / "traj-kl", gpus=[""])
    try:
        env = bfcl.BfclEnvironment(hf_dir=str(ink_dir), pairs_file=str(pairs_p), max_tokens=1024, reasoning_effort="none")
        ev = Evaluator(
            tc,
            env,
            {"temperature": 1.0},
            slots_per_gpu=1,
            ctx_per_slot=8192,
            kv_type="f16",
            log_dir=tmp_path / "logs_none",
            infra_retries=0,
            request_timeout=321.5,
        )
        assert ev.use_chat_completions is True
        tasks = env.tasks()
        rows = ev.run("mock-inkling2", [(t, 17 + i) for i, t in enumerate(tasks)], "bfcl_oai", save=True, srvs=[SimpleNamespace(url=url, special=True)])
        by_id = {r["task"]: r for r in rows}
        eps_by_id = {ep.task_id: ep for ep in ev.last_episodes}

        # Wire HTTP payload checks
        assert all(r["path"] == "/v1/chat/completions" for r in recs)
        req0 = recs[0]["body"]
        assert req0["messages"][0] == {"role": "system", "content": bfcl.DEFAULT_INKLING2_TOOL_SYSTEM_PROMPT}
        assert req0["messages"][1]["role"] == "user"
        assert "Python 3 syntax" in req0["tools"][0]["function"]["description"]
        assert req0["chat_template_kwargs"] == {"reasoning_effort": "none"}
        assert req0["temperature"] == 1.0
        assert req0["max_tokens"] == 1024
        assert req0["seed"] == 17
        assert req0["stop"] == ["<|content_model_end_sampling|>"]
        assert req0["cache_prompt"] is False
        assert 321.5 in seen_timeouts

        # Java raw_tools Gorilla hints sent on the wire
        req_java = recs[1]["body"]
        assert "Java 8 SDK syntax" in req_java["tools"][0]["function"]["description"]
        assert req_java["tools"][0]["function"]["parameters"]["properties"]["root"]["type"] == "string"

        # Grading & Episode population
        assert by_id["simple:s0"]["success"] is True
        assert by_id["simple:s0"]["tokens"] == 27
        assert eps_by_id["simple:s0"].reasoning_content == "Triangle area needs base=6 and height=10."
        assert eps_by_id["simple:s0"].finish_reasons == ["tool_calls"]

        assert by_id["java:j0"]["success"] is True
        assert by_id["java:j0"]["tokens"] == 31

        assert by_id["simple:s1_bad_json"]["success"] is False
        assert by_id["simple:s1_bad_json"]["detectors"]["malformed_tool_call"] is True

        assert by_id["simple:s2_len"]["success"] is False
        assert by_id["simple:s2_len"]["detectors"]["premature_stop"] is True
        assert by_id["simple:s2_len"]["detectors"]["malformed_tool_call"] is True

        assert by_id["simple:s3_garb"]["success"] is False
        assert by_id["simple:s3_garb"]["tests_passed"] is True
        assert by_id["simple:s3_garb"]["detectors"]["token_garbage"] is True

        assert by_id["simple:s4_infra"]["success"] is None
        assert by_id["simple:s4_infra"]["infra_error"] is not None

        assert by_id["simple:s5_unknown"]["success"] is False
        assert by_id["simple:s5_unknown"]["detectors"]["malformed_tool_call"] is True

        assert by_id["simple:s6_bad_args"]["success"] is False
        assert by_id["simple:s6_bad_args"]["detectors"]["malformed_tool_call"] is True

        assert by_id["simple:s7_loop"]["success"] is False
        assert by_id["simple:s7_loop"]["tests_passed"] is True
        assert by_id["simple:s7_loop"]["detectors"]["doom_loop"] is True

        # Saved JSONL preserves `verdict.pred_calls` (Gorilla replay compatible) and `reasoning_content`
        saved = {
            json.loads(line)["task"]: json.loads(line)
            for line in (tmp_path / "logs_none" / "bfcl_oai.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
        assert saved["simple:s0"]["verdict"]["pred_calls"] == [
            {"arguments": {"base": 6, "height": 10}, "name": "geometry.area_triangle"}
        ]
        assert saved["simple:s0"]["reasoning_content"] == "Triangle area needs base=6 and height=10."
        assert saved["java:j0"]["verdict"]["pred_calls"] == [{
            "arguments": {"property": "EnableExtensions", "root": "WinReg.HKEY_LOCAL_MACHINE"},
            "name": "BasePolicyDataProvider.getRegistryPolicyValue",
        }]
        assert "reasoning_content" not in saved["java:j0"]

        # High reasoning_effort propagates to `chat_template_kwargs` AND `kl.Renderer.episode` renders cleanly
        recs.clear()
        env_hi = bfcl.BfclEnvironment(hf_dir=str(ink_dir), pairs_file=str(pairs_p), max_tokens=2048, reasoning_effort="high")
        ev_hi = Evaluator(tc, env_hi, {"temperature": 1.0}, 1, 8192, "f16", tmp_path / "logs_hi", infra_retries=0)
        hi_rows = ev_hi.run("mock-inkling2", [(env_hi.tasks()[0], 99)], "bfcl_hi", save=False, srvs=[SimpleNamespace(url=url, special=True)])
        assert hi_rows[0]["success"] is True
        assert recs[0]["body"]["chat_template_kwargs"] == {"reasoning_effort": "high"}
        assert recs[0]["body"]["max_tokens"] == 2048
        ep_hi = ev_hi.last_episodes[0]
        R = kl.Renderer(str(ink_dir), reasoning_effort="high")
        kl_recs = R.episode(ep_hi.messages, ep_hi.tools)
        assert len(kl_recs) == 1 and len(kl_recs[0][1]) > 0
    finally:
        srv.shutdown()
        srv.server_close()
        th.join(timeout=2)


def test_bfcl_chat_completions_rule0_and_explicit_overrides(tmp_path: Path):
    from types import SimpleNamespace
    from chantless.evaluator import Evaluator

    pairs_p = tmp_path / "bfcl_r0.jsonl"
    pairs_p.write_text(
        json.dumps({
            "id": "simple:s0",
            "category": "simple",
            "family": "simple/b00",
            "split": "select",
            "messages": [{"role": "user", "content": "Call ping."}],
            "tools": [{"type": "function", "function": {"name": "ping", "description": "Ping.", "parameters": {"type": "object", "properties": {}}}}],
            "reference_calls": [{"name": "ping", "arguments": {}}],
        }) + "\n",
        encoding="utf-8",
    )
    dm_p = tmp_path / "dm_r0.jsonl"
    dm_p.write_text(
        json.dumps({"id": "q0", "question": "1+1?", "answer": "2", "difficulty": 4.0, "group": 1, "split": "select", "area": "Algebra"}) + "\n",
        encoding="utf-8",
    )

    def responder(path, body, idx):
        if path == "/v1/chat/completions":
            return 200, {
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [{"id": "c0", "type": "function", "function": {"name": "ping", "arguments": "{}"}}],
                    },
                    "finish_reason": "tool_calls",
                }],
                "usage": {"completion_tokens": 5},
            }
        # `/v1/completions`
        prompt = body["prompt"]
        if "<|message_user|>" in prompt and "1+1?" in prompt:
            txt = r"<|message_model|><|content_text|>\boxed{2}<|end_message|>"
        elif "<|message_user|>" in prompt:
            txt = '<|message_model|>ping<|content_invoke_tool_json|>{"name":"ping","args":{}}<|end_message|>'
        elif "<turn|>" in prompt:
            txt = "<|tool_call>call:ping{}<tool_call|>"
        elif "<function=" in prompt:
            txt = "<tool_call>\n<function=ping>\n</function>\n</tool_call>"
        else:
            txt = '<tool_call>\n{"name": "ping", "arguments": {}}\n</tool_call>'
        return 200, {"choices": [{"text": txt, "finish_reason": "stop"}], "usage": {"completion_tokens": 6}}

    srv, th, url, recs = _make_bfcl_loopback_server(responder)
    tc = runtime.Toolchain(llama_bin=tmp_path, traj_kl=tmp_path / "traj-kl", gpus=[""])
    try:
        # Rule 0: non-inkling2 BFCL (`qwen3_json`, `qwen35_xml`, `gemma4`) stays 100% on `/v1/completions`
        for dial in ("qwen3_json", "qwen35_xml", "gemma4"):
            d_dir = _build_bfcl_dialect_hf_dir(tmp_path / f"hf_{dial}", dial)
            env_d = bfcl.BfclEnvironment(hf_dir=str(d_dir), pairs_file=str(pairs_p))
            assert env_d.dialect == dial
            ev_d = Evaluator(tc, env_d, {"temperature": 0.0}, 1, 2048, "f16", tmp_path / f"log_{dial}")
            assert ev_d.use_chat_completions is False
            recs.clear()
            rows_d = ev_d.run("m", [(env_d.tasks()[0], 0)], f"r0_{dial}", save=False, srvs=[SimpleNamespace(url=url, special=False)])
            assert rows_d[0]["success"] is True
            assert len(recs) == 1 and recs[0]["path"] == "/v1/completions"

        # Rule 0: Inkling-2 DeepMath stays 100% on `/v1/completions`
        ink_dir = _build_bfcl_dialect_hf_dir(tmp_path / "hf_ink_r0", "inkling2")
        dm_env = deepmath.DeepMathEnvironment(hf_dir=str(ink_dir), jsonl_file=str(dm_p))
        ev_dm = Evaluator(tc, dm_env, {"temperature": 1.0}, 1, 4096, "f16", tmp_path / "log_dm")
        assert ev_dm.use_chat_completions is False
        recs.clear()
        dm_rows = ev_dm.run("m", [(dm_env.tasks()[0], 0)], "r0_dm", save=False, srvs=[SimpleNamespace(url=url, special=True)])
        assert dm_rows[0]["success"] is True
        assert len(recs) == 1 and recs[0]["path"] == "/v1/completions"

        # Explicit `use_chat_completions=False` on Inkling-2 BFCL forces `/v1/completions`
        ink_bfcl = bfcl.BfclEnvironment(hf_dir=str(ink_dir), pairs_file=str(pairs_p))
        ev_force_comp = Evaluator(
            tc, ink_bfcl, {"temperature": 1.0}, 1, 4096, "f16", tmp_path / "log_ink_comp", use_chat_completions=False
        )
        assert ev_force_comp.use_chat_completions is False
        recs.clear()
        rows_fc = ev_force_comp.run("m", [(ink_bfcl.tasks()[0], 0)], "ink_comp", save=False, srvs=[SimpleNamespace(url=url, special=True)])
        assert rows_fc[0]["success"] is True
        assert len(recs) == 1 and recs[0]["path"] == "/v1/completions"

        # Explicit `use_chat_completions=True` on Qwen3 BFCL enables `/v1/chat/completions` without leaking `reasoning_effort`
        q3_env = bfcl.BfclEnvironment(hf_dir=str(tmp_path / "hf_qwen3_json"), pairs_file=str(pairs_p))
        ev_force_chat = Evaluator(
            tc, q3_env, {"temperature": 0.0}, 1, 4096, "f16", tmp_path / "log_q3_chat", use_chat_completions=True
        )
        assert ev_force_chat.use_chat_completions is True
        recs.clear()
        rows_q3c = ev_force_chat.run("m", [(q3_env.tasks()[0], 0)], "q3_chat", save=False, srvs=[SimpleNamespace(url=url, special=False)])
        assert rows_q3c[0]["success"] is True
        assert len(recs) == 1 and recs[0]["path"] == "/v1/chat/completions"
        assert recs[0]["body"]["chat_template_kwargs"] == {"enable_thinking": False}
        assert recs[0]["body"]["cache_prompt"] is False

        # ServingConfig optional `use_chat_completions` flag validates and stays omitted when None
        sc_default = cfgmod.ServingConfig()
        assert sc_default.use_chat_completions is None
        assert "use_chat_completions" not in sc_default.model_dump(exclude_none=True)
        assert cfgmod.ServingConfig(use_chat_completions=True).use_chat_completions is True
        assert cfgmod.ServingConfig(use_chat_completions=False).use_chat_completions is False
        bfcl_cfg = cfgmod.load_config(
            ROOT / "mvp/configs/inkling2-mini-bfcl.yaml", "extract", overrides=["serving.use_chat_completions=false"]
        )
        assert bfcl_cfg.serving.use_chat_completions is False
    finally:
        srv.shutdown()
        srv.server_close()
        th.join(timeout=2)
