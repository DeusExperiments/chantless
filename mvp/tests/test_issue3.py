"""Tests for Issue #3: unified `chantless.mask/v3` thin mask, architecture adapter, Wanda width cuts,
merge-closed Rule-P vocabulary trim, exact byte model, and golden reproduction of Qwen `f200` and Gemma `mask10`.
"""
from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "mvp"))

from chantless import builder, gguf_ops, mask as mk, oneshot, vocab


# ------------------------------------------------------------------ synthetic GGUF writer for unit tests
def _pack_kv_str(s: str) -> tuple[int, bytes]:
    b = s.encode("utf-8")
    return 8, struct.pack("<Q", len(b)) + b


def _pack_kv_u32(v: int) -> tuple[int, bytes]:
    return 4, struct.pack("<I", v)


def _pack_kv_u32_arr(vals: list[int]) -> tuple[int, bytes]:
    return 9, struct.pack("<IQ", 4, len(vals)) + struct.pack(f"<{len(vals)}I", *vals)


def _pack_kv_i32_arr(vals: list[int]) -> tuple[int, bytes]:
    return 9, struct.pack("<IQ", 5, len(vals)) + struct.pack(f"<{len(vals)}i", *vals)


def _pack_kv_f32_arr(vals: list[float]) -> tuple[int, bytes]:
    return 9, struct.pack("<IQ", 6, len(vals)) + struct.pack(f"<{len(vals)}f", *vals)


def _pack_kv_str_arr(items: list[bytes]) -> tuple[int, bytes]:
    b = bytearray(struct.pack("<IQ", 8, len(items)))
    for x in items:
        b += struct.pack("<Q", len(x)) + x
    return 9, bytes(b)


def _f32_to_bf16_u16(arr: np.ndarray) -> np.ndarray:
    f32 = np.asarray(arr, dtype=np.float32)
    return (f32.view(np.uint32) >> 16).astype(np.uint16)


def _write_test_gguf(path: Path, kvs: list[tuple[str, tuple[int, bytes]]], tensors: list[tuple[str, list[int], int, bytes]]) -> None:
    kv_buf = bytearray()
    for k, (vt, raw) in kvs:
        kb = k.encode("utf-8")
        kv_buf += struct.pack("<Q", len(kb)) + kb + struct.pack("<I", vt) + raw
    gguf_ops._write_gguf_raw(3, 32, len(kvs), bytes(kv_buf), tensors, path)


# ------------------------------------------------------------------ 1. Architecture adapter (Qwen3.5 & Gemma 4)
def test_gemma4_adapter_matches_golden_groups_and_dead():
    """Verify automatic adapter derives exact 108 groups, {13, 14} KV donors, and 60 dead tensors of Gemma 4 E2B."""
    inv = [
        {"name": "token_embd.weight", "type": "BF16", "shape": [1536, 262144], "n_elements": 1536 * 262144, "n_bytes": 2 * 1536 * 262144},
        {"name": "per_layer_token_embd.weight", "type": "BF16", "shape": [8960, 262144], "n_elements": 8960 * 262144, "n_bytes": 2 * 8960 * 262144},
        {"name": "per_layer_model_proj.weight", "type": "BF16", "shape": [1536, 8960], "n_elements": 1536 * 8960, "n_bytes": 2 * 1536 * 8960},
        {"name": "per_layer_proj_norm.weight", "type": "F32", "shape": [256], "n_elements": 256, "n_bytes": 1024},
        {"name": "rope_freqs.weight", "type": "F32", "shape": [256], "n_elements": 256, "n_bytes": 1024},
        {"name": "output_norm.weight", "type": "F32", "shape": [1536], "n_elements": 1536, "n_bytes": 6144},
    ]
    layer_types = []
    for il in range(35):
        is_global = (il + 1) % 5 == 0
        layer_types.append("full_attention" if is_global else "sliding_attention")
        p = f"blk.{il}."
        for suf in ("attn_q.weight", "attn_k.weight", "attn_v.weight", "attn_output.weight",
                    "attn_norm.weight", "attn_q_norm.weight", "attn_k_norm.weight", "post_attention_norm.weight",
                    "ffn_gate.weight", "ffn_up.weight", "ffn_down.weight", "ffn_norm.weight", "post_ffw_norm.weight",
                    "inp_gate.weight", "proj.weight", "post_norm.weight", "layer_output_scale.weight"):
            inv.append({"name": p + suf, "type": "BF16", "shape": [256, 256], "n_elements": 65536, "n_bytes": 131072})

    hf_cfg = {"text_config": {"num_hidden_layers": 35, "num_kv_shared_layers": 20, "layer_types": layer_types}}
    groups, dead = gguf_ops._build_groups_and_dead(inv, kvs={"general.architecture": "gemma4"}, hf_config=hf_cfg)
    assert len(dead) == 60
    real_groups = {k: v for k, v in groups.items() if not k.startswith("_")}
    assert len(real_groups) == 108
    assert not real_groups["blk.13.attn"]["droppable"]
    assert not real_groups["blk.14.attn"]["droppable"]
    assert real_groups["blk.12.attn"]["droppable"]
    assert real_groups["blk.15.attn"]["droppable"]
    assert all(real_groups[f"blk.{i}.attn"]["kv"] for i in range(15))
    assert all(not real_groups[f"blk.{i}.attn"]["kv"] for i in range(15, 35))

    g_receipt = ROOT / ".local/gemma4/mask10/receipts/groups.json"
    d_receipt = ROOT / ".local/gemma4/mask10/receipts/dead.json"
    if g_receipt.exists() and d_receipt.exists():
        golden_groups = json.loads(g_receipt.read_text())
        golden_dead = json.loads(d_receipt.read_text())
        assert dead == golden_dead
        assert set(real_groups) == set(golden_groups)
        for g in golden_groups:
            assert sorted(real_groups[g]["quantizable"]) == sorted(golden_groups[g]["quantizable"])
            assert sorted(real_groups[g]["fixed"]) == sorted(golden_groups[g]["fixed"])
            assert real_groups[g]["droppable"] == golden_groups[g]["droppable"]
            assert real_groups[g]["kv"] == golden_groups[g]["kv"]
            assert real_groups[g]["imatrix_ok"] == golden_groups[g]["imatrix_ok"]


# ------------------------------------------------------------------ 2. Wanda order, variable width, vocab trim & ByteModel
@pytest.mark.parametrize("ff_array_mode", [False, True])
def test_wanda_variable_width_vocab_trim_and_exact_bytemodel(tmp_path: Path, ff_array_mode: bool):
    arch = "gemma4" if ff_array_mode else "qwen35"
    n_layer, n_embd, n_ff = 2, 256, 512
    tokens = [b"<pad>", b"<bos>", b"!", b",", b"a", b"b", b"c", b"ab", b"bc", b"abc", b"human", b"ai"]
    ttypes = [3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    merges = [b"a b", b"b c", b"ab c"]
    scores = [0.0] * len(tokens)
    v_size = len(tokens)

    kvs = [
        ("general.architecture", _pack_kv_str(arch)),
        ("general.alignment", _pack_kv_u32(32)),
        (f"{arch}.block_count", _pack_kv_u32(n_layer)),
        (f"{arch}.embedding_length", _pack_kv_u32(n_embd)),
        (f"{arch}.attention.head_count", _pack_kv_u32(4)),
        (f"{arch}.attention.head_count_kv", _pack_kv_u32(2)),
        (f"{arch}.feed_forward_length", _pack_kv_u32_arr([n_ff] * n_layer) if ff_array_mode else _pack_kv_u32(n_ff)),
        ("tokenizer.ggml.tokens", _pack_kv_str_arr(tokens)),
        ("tokenizer.ggml.token_type", _pack_kv_i32_arr(ttypes)),
        ("tokenizer.ggml.merges", _pack_kv_str_arr(merges)),
        ("tokenizer.ggml.scores", _pack_kv_f32_arr(scores)),
        ("tokenizer.ggml.bos_token_id", _pack_kv_u32(1)),
    ]

    rng = np.random.default_rng(42)
    tensors = [
        ("token_embd.weight", [n_embd, v_size], 30, _f32_to_bf16_u16(rng.standard_normal((v_size, n_embd))).tobytes()),
        ("output_norm.weight", [n_embd], 0, np.ones(n_embd, dtype=np.float32).tobytes()),
    ]
    im_tensors = []
    for il in range(n_layer):
        p = f"blk.{il}."
        # Construct ffn_down so neurons 256..511 have 10x larger Wanda score than 0..255 in layer 0
        down_f32 = np.ones((n_embd, n_ff), dtype=np.float32)
        in_sum2 = np.ones(n_ff, dtype=np.float32)
        if il == 0:
            down_f32[:, 256:] = 5.0
            in_sum2[256:] = 4.0
        else:
            in_sum2 = np.linspace(10.0, 1.0, n_ff, dtype=np.float32)
        tensors += [
            (p + "attn_norm.weight", [n_embd], 0, np.ones(n_embd, dtype=np.float32).tobytes()),
            (p + "attn_q.weight", [n_embd, n_embd], 30, _f32_to_bf16_u16(rng.standard_normal((n_embd, n_embd))).tobytes()),
            (p + "attn_k.weight", [n_embd, n_embd], 30, _f32_to_bf16_u16(rng.standard_normal((n_embd, n_embd))).tobytes()),
            (p + "attn_v.weight", [n_embd, n_embd], 30, _f32_to_bf16_u16(rng.standard_normal((n_embd, n_embd))).tobytes()),
            (p + "attn_output.weight", [n_embd, n_embd], 30, _f32_to_bf16_u16(rng.standard_normal((n_embd, n_embd))).tobytes()),
            (p + "post_attention_norm.weight", [n_embd], 0, np.ones(n_embd, dtype=np.float32).tobytes()),
            (p + "ffn_gate.weight", [n_embd, n_ff], 30, _f32_to_bf16_u16(rng.standard_normal((n_ff, n_embd))).tobytes()),
            (p + "ffn_up.weight", [n_embd, n_ff], 30, _f32_to_bf16_u16(rng.standard_normal((n_ff, n_embd))).tobytes()),
            (p + "ffn_down.weight", [n_ff, n_embd], 30, _f32_to_bf16_u16(down_f32).tobytes()),
        ]
        im_tensors.append((p + "ffn_down.weight.in_sum2", [n_ff], 0, in_sum2.tobytes()))

    src_gguf = tmp_path / "src.gguf"
    im_gguf = tmp_path / "imatrix.gguf"
    _write_test_gguf(src_gguf, kvs, tensors)
    _write_test_gguf(im_gguf, [("general.architecture", _pack_kv_str("imatrix"))], im_tensors)

    # Check Wanda ordering puts neurons 256..511 ahead of 0..255 in layer 0
    order = gguf_ops.wanda_order(src_gguf, im_gguf)
    assert set(order[0][:256].tolist()) == set(range(256, 512))

    # Check Rule-P keep set closes BPE merge parents: high count on "abc" (id 9) must pull in "ab" (7), "a" (4), "b" (5), "c" (6)
    counts = np.zeros(v_size, dtype=np.int64)
    counts[9] = 500  # "abc"
    keep = vocab.rule_p_keep_set(src_gguf, counts, k=11, must_keep=(10, 11))
    for req in (0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11):
        assert req in set(keep.tolist())
    assert 8 not in set(keep.tolist())  # "bc" was unused and not a parent of "abc"
    assert vocab.remap_token_ids(keep, (10, 11)) == (9, 10)

    # Slice layer 0 width to 256, drop layer 1 FFN, and trim vocabulary to `keep`
    groups, dead = gguf_ops.adapter_from_gguf(src_gguf)
    assert not dead
    m = mk.new_mask(groups, "BF16", {"repo": "test", "revision": "0", "gguf_sha256": "0"})
    m["groups"]["blk.1.ffn"]["tier"] = "drop"
    m["widths"]["0"] = mk.pack_bitmap(order[0][:256], n_ff)
    m["vocab_keep"] = mk.pack_bitmap(keep, v_size)
    mk.validate(m, groups)

    var_gguf = tmp_path / "var.gguf"
    im_sub = tmp_path / "im_sub.gguf"
    _, drop_set = mk.tensor_plan(m, groups)
    gguf_ops.write_variable_width(src_gguf, var_gguf, {0: order[0][:256]}, drop_tensors=drop_set,
                                  imatrix_src=im_gguf, imatrix_dst=im_sub)
    final_gguf = tmp_path / "final.gguf"
    vocab.trim_gguf(var_gguf, final_gguf, keep)

    # ByteModel must predict the exact byte count (0-byte error!) across width cut + FFN drop + vocab trim
    bm = oneshot.ByteModel(str(src_gguf), groups)
    assert bm(m) == final_gguf.stat().st_size


# ------------------------------------------------------------------ 3. Thin-mask v3 packing & real receipt sizes
def test_thin_mask_v3_roundtrip_and_golden_receipt_sizes():
    g_dir = ROOT / ".local/gemma4/mask10/receipts"
    q_spec = ROOT / ".local/detect/oneshot-20261104/spec_f200.json"
    if not g_dir.exists() or not q_spec.exists():
        pytest.skip("local receipts not present")

    # Pack real Gemma 4 E2B 10% + Apprentice switch mask into a single `chantless.mask/v3` JSON
    groups = json.loads((g_dir / "groups.json").read_text())
    dead = json.loads((g_dir / "dead.json").read_text())
    spec = json.loads((g_dir / "spec_g10.json").read_text())
    order = json.loads((g_dir / "order.json").read_text())
    keep_80k = np.load(g_dir / "keep-80000.npy")

    gm = mk.new_mask(groups, "BF16", {"repo": "gemma4-e2b-detector", "revision": "local", "gguf_sha256": "src"})
    for g, t in spec["tiers"].items():
        gm["groups"][g]["tier"] = t
    gm["dead_tensors"] = dead
    for k, w in spec["widths"].items():
        full_w = len(order[k])
        gm["widths"][k] = mk.pack_bitmap(np.sort(np.asarray(order[k][:w], dtype=np.int64)), full_w)
    gm["vocab_keep"] = mk.pack_bitmap(keep_80k, 262144)

    sw_path = ROOT / ".local/gemma4/apprentice/receipts/i1.gguf.switches.json"
    if sw_path.exists():
        sw = json.loads(sw_path.read_text())
        for k, dead_idx in sw.items():
            full_w = int(spec["widths"].get(k, len(order[k])))
            gm["apprentice"]["switches"][k] = mk.pack_bitmap(dead_idx, full_w)

    mk.validate(gm, groups)
    encoded = json.dumps(gm, separators=(",", ":")).encode("utf-8")
    assert len(encoded) < 80_000, f"Gemma thin mask exceeded ~78 KB target: {len(encoded)} B"
    reloaded = json.loads(encoded.decode("utf-8"))
    assert np.array_equal(mk.unpack_bitmap(reloaded["vocab_keep"]), keep_80k)
    for k, w in spec["widths"].items():
        expected = np.sort(np.asarray(order[k][:w], dtype=np.int64))
        assert np.array_equal(mk.unpack_bitmap(reloaded["widths"][k]), expected)

    # Also verify v2 mask_key stability for existing cached Qwen sanity probe filenames
    q_groups = json.loads((ROOT / ".local/detect/cls-run/detect-cls-400mb-001/sanity/sanity_mask.json").read_text())["groups"]
    v2 = {"schema": "chantless.mask/v2", "groups": {g: {"tier": "BF16"} for g in q_groups}, "tensors": {}, "weights": {"lineage": []}}
    v3_plain = mk.upgrade_v2(v2)
    assert builder.mask_key(v2) == builder.mask_key(v3_plain)


# ------------------------------------------------------------------ 4. Golden Qwen f200 & Gemma mask10 verification
def test_reproduce_qwen_f200_and_gemma_mask10_from_saved_receipts():
    headers_p = ROOT / ".local/detect/neuron/solver/inputs/headers.json"
    q_groups_p = ROOT / ".local/detect/neuron/solver/inputs/groups.json"
    cls_run = ROOT / ".local/detect/cls-run/detect-cls-400mb-001"
    os_dir = ROOT / ".local/detect/oneshot-20261104"
    g_dir = ROOT / ".local/gemma4/mask10/receipts"
    if not (headers_p.exists() and cls_run.exists() and os_dir.exists() and g_dir.exists()):
        pytest.skip("local Qwen/Gemma receipts not present")

    # A. Verify unified ByteModel matches q4cut50 (200,296,000 B) and f200 (200,240,128 B) to 0 bytes
    type_name = {0: "F32", 1: "F16", 8: "Q8_0", 12: "Q4_K", 19: "IQ1_S", 30: "BF16"}
    hdr = json.loads(headers_p.read_text())
    q_groups = json.loads(q_groups_p.read_text())
    src_tensors = [{"name": n, "shape": d, "type": type_name[t]} for n, d, t, _ in hdr["source"]["tensors"]]
    bm = oneshot.ByteModel(None, q_groups, tensors=src_tensors, kv_end=hdr["cls400"]["kv_end"], ff_is_array=False, n_layer=24)
    bm.set_vocab_trim(np.arange(79999, dtype=np.int64), kv_cut=8137472)

    fin_groups = json.loads((cls_run / "final/mask.json").read_text())["groups"]
    q4_tiers = {g: ("Q4_K" if v["tier"] == "Q8_0" else v["tier"]) for g, v in fin_groups.items()}
    q4_widths = {7: 3072, 10: 1024, 11: 1024, 13: 512, 15: 512, 17: 256, 20: 256, 21: 256, 23: 256}
    assert abs(bm.spec_bytes(q4_tiers, q4_widths) - 200_296_000) <= 32

    # B. Run unified FidelitySurrogate + solve() on saved Qwen solo + width probes -> must reproduce spec_f200.json!
    src_scores = np.load(cls_run / "proxy/store_task/source.npy")
    gs = list(json.loads((cls_run / "sanity/sanity_mask.json").read_text())["groups"])
    solo = {}
    for g in gs:
        for t in ("Q4_K", "IQ1_S", "drop"):
            m = {"groups": {x: {"tier": "BF16"} for x in gs}, "tensors": {}, "weights": {"lineage": []}}
            m["groups"][g]["tier"] = t
            pf = cls_run / f"probe/sanity/p-{builder.mask_key(m)}-task.npy"
            if pf.exists():
                solo[(g, t)] = np.load(pf) - src_scores

    wz = np.load(os_dir / "w.npz")
    width_cuts = {tuple(map(int, k.split("-"))): wz[k] - wz["src"] for k in wz.files if k != "src"}
    n_ff0 = {il: 3584 for il in range(24)}
    allowed = {g: ["Q8_0", "Q4_K", "IQ1_S"] for g in gs}
    options = oneshot.build_joint_options(q_groups, allowed, solo, width_cuts, n_ff0, width_step=256, allow_drop=True)

    ref_sol = {g: ("Q8_0", 3584 if g.endswith(".ffn") else 0) for g in gs}
    ref_t, ref_w = oneshot.unpack_solution(ref_sol)
    ref_b = bm.spec_bytes(ref_t, ref_w)
    cost = {}
    for g, opts in options.items():
        for o in opts:
            s1 = dict(ref_sol)
            s1[g] = o
            cost[(g, o)] = bm.spec_bytes(*oneshot.unpack_solution(s1)) - ref_b

    sur = oneshot.FidelitySurrogate([("task", 1.0, lambda x: 0.0, src_scores)], solo, width_cuts=width_cuts, n_ff0=n_ff0)
    sol = oneshot.solve(sur, options, cost, budget=200_300_000 - 64, header=ref_b, pair_top=60, max_polish=100, log=lambda *_: None)
    tiers_out, widths_out = oneshot.unpack_solution(sol)
    widths_cut = {str(k): v for k, v in widths_out.items() if v < 3584}

    golden_f200 = json.loads((os_dir / "spec_f200.json").read_text())
    assert tiers_out == golden_f200["tiers"]
    assert widths_cut == {str(k): int(v) for k, v in golden_f200["widths"].items()}
    assert bm.spec_bytes(tiers_out, widths_out) == 200_240_192
    assert abs(bm.spec_bytes(tiers_out, widths_out) - 200_240_128) <= 64

    gates_q = json.loads((os_dir / "gates_20261104.json").read_text())
    assert gates_q["f200"]["lost"] == 47
    assert gates_q["f200"]["bytes"] == 200_240_128

    # C. Verify Gemma 4 E2B mask10 gate receipt & score reproduction
    gate_g = json.loads((g_dir / "gate_select.json").read_text())
    src_cal = np.load(g_dir / "src_cal.npy")
    cand_cal = np.load(g_dir / "cal.npy")
    assert float(np.sort(src_cal)[200]) == pytest.approx(gate_g["src_cutoff"])
    assert float(np.sort(cand_cal)[200]) == pytest.approx(gate_g["cutoff"])
    assert gate_g["bytes"] == 930_971_136
    assert gate_g["lost"] == 42
    assert gate_g["L"] == pytest.approx(-0.0055489498, abs=1e-7)
    assert gate_g["pass"] is True


# ------------------------------------------------------------------ 5. End-to-end Builder.build(v3) + pick_vocab_k
def test_builder_v3_end_to_end_deterministic_rebuild_and_auto_vocab(tmp_path: Path):
    from chantless.runtime import Toolchain
    from chantless.util import sha256_file

    n_layer, n_embd, n_ff = 2, 256, 512
    tokens = [b"<pad>", b"<bos>", b".", b"a", b"b", b"ab", b"rare1", b"rare2", b"human", b"ai"]
    ttypes = [3, 3, 1, 1, 1, 1, 1, 1, 1, 1]
    merges = [b"a b"]
    v_size = len(tokens)
    kvs = [
        ("general.architecture", _pack_kv_str("gemma4")),
        ("general.alignment", _pack_kv_u32(32)),
        ("gemma4.block_count", _pack_kv_u32(n_layer)),
        ("gemma4.embedding_length", _pack_kv_u32(n_embd)),
        ("gemma4.attention.head_count", _pack_kv_u32(4)),
        ("gemma4.attention.head_count_kv", _pack_kv_u32(1)),
        ("gemma4.attention.shared_kv_layers", _pack_kv_u32(1)),
        ("gemma4.feed_forward_length", _pack_kv_u32_arr([n_ff, n_ff])),
        ("tokenizer.ggml.tokens", _pack_kv_str_arr(tokens)),
        ("tokenizer.ggml.token_type", _pack_kv_i32_arr(ttypes)),
        ("tokenizer.ggml.merges", _pack_kv_str_arr(merges)),
        ("tokenizer.ggml.bos_token_id", _pack_kv_u32(1)),
    ]
    rng = np.random.default_rng(7)
    tensors = [
        ("token_embd.weight", [n_embd, v_size], 30, _f32_to_bf16_u16(rng.standard_normal((v_size, n_embd))).tobytes()),
        ("per_layer_token_embd.weight", [n_embd, v_size], 30, _f32_to_bf16_u16(rng.standard_normal((v_size, n_embd))).tobytes()),
        ("output_norm.weight", [n_embd], 0, np.ones(n_embd, dtype=np.float32).tobytes()),
    ]
    for il in range(n_layer):
        p = f"blk.{il}."
        for suf, sh in (
            ("attn_norm.weight", [n_embd]), ("attn_q.weight", [n_embd, n_embd]), ("attn_k.weight", [n_embd, n_embd]),
            ("attn_v.weight", [n_embd, n_embd]), ("attn_output.weight", [n_embd, n_embd]), ("attn_q_norm.weight", [64]),
            ("attn_k_norm.weight", [64]), ("post_attention_norm.weight", [n_embd]), ("ffn_norm.weight", [n_embd]),
            ("ffn_gate.weight", [n_embd, n_ff]), ("ffn_up.weight", [n_embd, n_ff]), ("ffn_down.weight", [n_ff, n_embd]),
            ("post_ffw_norm.weight", [n_embd]),
        ):
            dt = 0 if len(sh) == 1 else 30
            blob = np.ones(sh[0], dtype=np.float32).tobytes() if dt == 0 else _f32_to_bf16_u16(rng.standard_normal((sh[1], sh[0]))).tobytes()
            tensors.append((p + suf, sh, dt, blob))

    src = tmp_path / "base.gguf"
    _write_test_gguf(src, kvs, tensors)

    groups, dead = gguf_ops.adapter_from_gguf(src)
    assert dead == ["blk.1.attn_k.weight", "blk.1.attn_v.weight", "blk.1.attn_k_norm.weight"]
    assert not groups["blk.0.attn"]["droppable"]  # sole KV donor for shared layer 1

    # Auto-pick vocab K: K=8 hits 0.999 fidelity, K=6 hits 0.990 -> chooses K=8
    counts = np.array([1, 1, 5, 20, 20, 30, 0, 0, 10, 10], dtype=np.int64)
    chosen_k, keep_vocab, _ = vocab.pick_vocab_k(
        src, counts, [6, 8, 9], lambda kp: 0.999 if len(kp) >= 8 else 0.990, min_fidelity=0.998, must_keep=(8, 9)
    )
    assert chosen_k == 8
    assert len(keep_vocab) == 8

    m = mk.new_mask(groups, "BF16", {"repo": "test", "revision": "1", "gguf_sha256": sha256_file(src)})
    m["dead_tensors"] = dead
    m["groups"]["blk.1.attn"]["tier"] = "drop"
    m["widths"]["0"] = mk.pack_bitmap(np.arange(256, dtype=np.int64), n_ff)
    m["vocab_keep"] = mk.pack_bitmap(keep_vocab, v_size)

    tc = Toolchain(tmp_path, tmp_path / "traj-kl", [""])
    B = builder.Builder(tc, str(src), groups, None, tmp_path / "work")
    r1 = B.build(m, tmp_path / "out1.gguf")
    r2 = B.build(m, tmp_path / "out2.gguf")
    assert sha256_file(r1.path) == sha256_file(r2.path)

    bm = oneshot.ByteModel(str(src), groups)
    assert bm(m) == r1.file_bytes

    # Also test `python -m chantless.cli build` end-to-end from saved thin mask JSON
    import subprocess
    mask_json = tmp_path / "thin.mask.json"
    mask_json.write_text(json.dumps(m))
    cli_out = tmp_path / "cli_out.gguf"
    proc = subprocess.run(
        [sys.executable, "-m", "chantless.cli", "build", "--source", str(src), "--mask", str(mask_json), "--out", str(cli_out)],
        cwd=str(ROOT / "mvp"), capture_output=True, text=True, check=True,
    )
    assert json.loads(proc.stdout)["file_bytes"] == r1.file_bytes
    assert sha256_file(cli_out) == sha256_file(r1.path)


# ------------------------------------------------------------------ 7. Sharded parallel_quantize bit-identity
def test_parallel_quantize_bit_identical_to_single(tmp_path: Path):
    """Verify multi-process `parallel_quantize` produces bit-identical SHA256 to single-process `quantize`."""
    import hashlib
    import os
    import stat
    from chantless.runtime import Toolchain

    n_embd, n_ff, v_size = 256, 512, 16
    tokens = [f"tok{i}".encode() for i in range(v_size)]
    kvs = [
        ("general.architecture", _pack_kv_str("qwen35")),
        ("qwen35.block_count", _pack_kv_u32(2)),
        ("qwen35.feed_forward_length", _pack_kv_u32(n_ff)),
        ("tokenizer.ggml.tokens", _pack_kv_str_arr(tokens)),
        ("tokenizer.ggml.token_type", _pack_kv_i32_arr([1] * v_size)),
        ("tokenizer.ggml.merges", _pack_kv_str_arr([])),
    ]
    rng = np.random.default_rng(123)
    tensors = [
        ("token_embd.weight", [n_embd, v_size], 30, _f32_to_bf16_u16(rng.standard_normal((v_size, n_embd))).tobytes()),
        ("output_norm.weight", [n_embd], 0, np.ones(n_embd, dtype=np.float32).tobytes()),
    ]
    for il in range(2):
        p = f"blk.{il}."
        for suf, sh in (
            ("attn_norm.weight", [n_embd]), ("attn_q.weight", [n_embd, n_embd]),
            ("attn_k.weight", [n_embd, n_embd]), ("attn_v.weight", [n_embd, n_embd]),
            ("attn_output.weight", [n_embd, n_embd]), ("ffn_norm.weight", [n_embd]),
            ("ffn_gate.weight", [n_embd, n_ff]), ("ffn_up.weight", [n_embd, n_ff]),
            ("ffn_down.weight", [n_ff, n_embd]),
        ):
            dt = 0 if len(sh) == 1 else 30
            blob = np.ones(sh[0], dtype=np.float32).tobytes() if dt == 0 else _f32_to_bf16_u16(rng.standard_normal((sh[1], sh[0]))).tobytes()
            tensors.append((p + suf, sh, dt, blob))

    src = tmp_path / "src.gguf"
    _write_test_gguf(src, kvs, tensors)
    im = tmp_path / "imatrix.gguf"
    im_tensors = [
        (f"blk.{il}.ffn_down.weight.in_sum2", [n_ff], 0, np.ones(n_ff, dtype=np.float32).tobytes())
        for il in range(2)
    ]
    _write_test_gguf(im, [("general.architecture", _pack_kv_str("imatrix"))], im_tensors)

    # Deterministic stub `llama-quantize` implementing real GGUF type/block-size conversion + imatrix KV injection
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake_q = bin_dir / "llama-quantize"
    fake_q.write_text(f"""#!{sys.executable}
import hashlib, os, re, struct, sys
sys.path.insert(0, {str(ROOT / "mvp")!r})
from chantless import gguf_ops

NAME_TO_TTYPE = {{name.upper(): (tt, bs, ts) for tt, (name, bs, ts) in gguf_ops.GGML_TYPES.items()}}

args = sys.argv[1:]
ov_rules = []
tok_type = None
im_path = None
pos = []
i = 0
while i < len(args):
    a = args[i]
    if a == "--pure":
        i += 1
    elif a == "--tensor-type-file":
        for line in open(args[i + 1]):
            line = line.strip()
            if line and "=" in line:
                pat, ty = line.split("=", 1)
                ov_rules.append((re.compile(pat), ty.upper()))
        i += 2
    elif a == "--token-embedding-type":
        tok_type = args[i + 1].upper(); i += 2
    elif a == "--imatrix":
        im_path = args[i + 1]; i += 2
    else:
        pos.append(a); i += 1

src_p, dst_p, base_ty, _threads = pos[0], pos[1], pos[2].upper(), int(pos[3])
h = gguf_ops._Hdr(src_p)
try:
    kv_bytes = bytearray(h.m[h.kv_start : h.kv_end])
    n_kv = h.n_kv
    for k, s in [("general.quantization_version", None), ("quantize.imatrix.file", im_path or "")]:
        if k == "quantize.imatrix.file" and not im_path:
            continue
        kb = k.encode()
        if s is None:
            kv_bytes += struct.pack("<Q", len(kb)) + kb + struct.pack("<II", 4, 2)
        else:
            sb = s.encode()
            kv_bytes += struct.pack("<Q", len(kb)) + kb + struct.pack("<IQ", 8, len(sb)) + sb
        n_kv += 1
    out_t = []
    for t in h.tensors:
        name = t["name"]
        s0 = h.data_start + t["offset"]
        raw = bytes(h.m[s0 : s0 + t["n_bytes"]])
        if len(t["dims"]) < 2:
            out_t.append((name, list(t["dims"]), t["ttype"], raw))
            continue
        req = base_ty
        if name == "token_embd.weight" and tok_type:
            req = tok_type
        else:
            for rx, ty in ov_rules:
                if rx.search(name):
                    req = ty
                    break
        if req == t["type"].upper():
            out_t.append((name, list(t["dims"]), t["ttype"], raw))
        else:
            tt, bs, ts = NAME_TO_TTYPE[req]
            nb = (t["n_elements"] // bs) * ts
            digest = hashlib.sha256(name.encode() + b"|" + req.encode() + b"|" + raw[:64]).digest()
            qraw = (digest * ((nb + 31) // 32))[:nb]
            out_t.append((name, list(t["dims"]), tt, qraw))
    gguf_ops._write_gguf_raw(h.version, h.alignment, n_kv, bytes(kv_bytes), out_t, dst_p)
finally:
    h.close()
""")
    fake_q.chmod(fake_q.stat().st_mode | stat.S_IEXEC)

    groups, _ = gguf_ops.adapter_from_gguf(src)
    m = mk.new_mask(groups, "Q4_K", {"repo": "t", "revision": "1", "gguf_sha256": "0" * 64})
    m["groups"]["token_embd"]["tier"] = "Q2_K"
    m["groups"]["blk.0.ffn"]["tier"] = "IQ2_XS"
    m["groups"]["blk.1.attn"]["tier"] = "Q8_0"
    m["widths"]["0"] = mk.pack_bitmap(np.arange(256, dtype=np.int64), n_ff)
    m["vocab_keep"] = mk.pack_bitmap(np.array([0, 1, 2, 3, 4, 5, 6, 7], dtype=np.int64), v_size)

    tc = Toolchain(bin_dir, tmp_path / "traj-kl", [""])
    out_single = tmp_path / "single.gguf"
    out_sharded = tmp_path / "sharded.gguf"
    r_single = builder.Builder(tc, str(src), groups, str(im), tmp_path / "w1", threads=8, quantize_shards=1).build(m, out_single)
    r_shard = builder.Builder(tc, str(src), groups, str(im), tmp_path / "w1", threads=32, quantize_shards=5).build(m, out_sharded)

    assert r_single.file_bytes == r_shard.file_bytes
    assert hashlib.sha256(out_single.read_bytes()).hexdigest() == hashlib.sha256(out_shard := out_sharded.read_bytes()).hexdigest()


# ------------------------------------------------------------------ 8. Packed multi-seq logodds.cpp + BatchScorer
def test_logodds_cpp_packed_multiseq_and_batchscorer(tmp_path: Path):
    """Compile real `mvp/tools/logodds.cpp` against a causal sequence-isolated stub runtime and verify exact parity."""
    import shutil
    import subprocess
    from chantless.runtime import BatchScorer, Toolchain

    cxx = shutil.which("clang++") or shutil.which("g++")
    if not cxx:
        pytest.skip("no C++ compiler available")

    inc = tmp_path / "inc"
    inc.mkdir()
    (inc / "llama.h").write_text("""#pragma once
#include <cstdint>
#include <cstdlib>
#include <vector>

typedef int32_t llama_token;
typedef int32_t llama_pos;
typedef int32_t llama_seq_id;

enum llama_flash_attn_type {
    LLAMA_FLASH_ATTN_TYPE_AUTO = -1,
    LLAMA_FLASH_ATTN_TYPE_DISABLED = 0,
    LLAMA_FLASH_ATTN_TYPE_ENABLED = 1,
};

struct llama_model_params { int32_t n_gpu_layers; };
struct llama_context_params {
    uint32_t n_ctx;
    uint32_t n_batch;
    uint32_t n_ubatch;
    uint32_t n_seq_max;
    llama_flash_attn_type flash_attn_type;
    bool kv_unified;
};

struct llama_batch {
    int32_t n_tokens;
    llama_token * token;
    float * embd;
    llama_pos * pos;
    int32_t * n_seq_id;
    llama_seq_id ** seq_id;
    int8_t * logits;
};

struct llama_model { int dummy; };
struct llama_memory_t {
    std::vector<double> seq_acc;
    std::vector<int32_t> expected_pos;
};
typedef llama_memory_t * llama_memory_i;

struct llama_context {
    llama_context_params cp;
    llama_memory_t mem;
    std::vector<std::vector<float>> logits_buf;
};

inline void llama_backend_init() {}
inline void llama_backend_free() {}
inline llama_model_params llama_model_default_params() { return {0}; }
inline llama_context_params llama_context_default_params() {
    return {512, 512, 512, 1, LLAMA_FLASH_ATTN_TYPE_AUTO, false};
}
inline llama_model * llama_model_load_from_file(const char *, llama_model_params) { return new llama_model{1}; }
inline void llama_model_free(llama_model * m) { delete m; }
inline llama_context * llama_init_from_model(llama_model *, llama_context_params cp) {
    auto * c = new llama_context();
    c->cp = cp;
    c->mem.seq_acc.assign(cp.n_seq_max, 0.0);
    c->mem.expected_pos.assign(cp.n_seq_max, 0);
    return c;
}
inline void llama_free(llama_context * c) { delete c; }
inline llama_memory_i llama_get_memory(llama_context * c) { return &c->mem; }
inline void llama_memory_clear(llama_memory_i m, bool) {
    std::fill(m->seq_acc.begin(), m->seq_acc.end(), 0.0);
    std::fill(m->expected_pos.begin(), m->expected_pos.end(), 0);
}
inline llama_batch llama_batch_init(int32_t n_tokens, int32_t, int32_t n_seq_max) {
    llama_batch b{};
    b.token = new llama_token[n_tokens];
    b.pos = new llama_pos[n_tokens];
    b.n_seq_id = new int32_t[n_tokens];
    b.seq_id = new llama_seq_id *[n_tokens];
    for (int32_t i = 0; i < n_tokens; ++i) b.seq_id[i] = new llama_seq_id[n_seq_max];
    b.logits = new int8_t[n_tokens];
    return b;
}
inline void llama_batch_free(llama_batch b) {
    delete[] b.token; delete[] b.pos; delete[] b.n_seq_id;
    delete[] b.logits; delete[] b.seq_id;
}
inline int llama_decode(llama_context * c, llama_batch b) {
    if ((uint32_t) b.n_tokens > c->cp.n_ubatch) return 1;
    c->logits_buf.assign(b.n_tokens, {});
    for (int32_t j = 0; j < b.n_tokens; ++j) {
        int s = b.seq_id[j][0];
        if (s < 0 || (uint32_t) s >= c->cp.n_seq_max) return 2;
        if (b.pos[j] != c->mem.expected_pos[s]) return 3;
        c->mem.expected_pos[s] += 1;
        c->mem.seq_acc[s] = c->mem.seq_acc[s] * 0.997 + (double)(b.token[j] + 1) * 0.125 + 0.01 * b.pos[j];
        if (b.logits[j]) {
            c->logits_buf[j].assign(64, 0.0f);
            c->logits_buf[j][10] = (float)(c->mem.seq_acc[s] * 1.5);
            c->logits_buf[j][20] = (float)(-c->mem.seq_acc[s] * 0.5 + 0.25);
            c->logits_buf[j][30] = (float)(c->mem.seq_acc[s] * 0.75 - 1.0);
        }
    }
    return 0;
}
inline float * llama_get_logits_ith(llama_context * c, int32_t i) {
    return c->logits_buf.at(i).data();
}
""")

    bin_path = tmp_path / "logodds"
    subprocess.run(
        [cxx, "-O2", "-std=c++17", f"-I{inc}", str(ROOT / "mvp/tools/logodds.cpp"), "-o", str(bin_path)],
        check=True, capture_output=True, text=True,
    )

    def write_trj(path: Path, seqs: list[tuple[list[int], list[int]]]) -> None:
        with open(path, "wb") as f:
            f.write(b"TRJ1" + struct.pack("<I", len(seqs)))
            for toks, tgts in seqs:
                f.write(struct.pack("<II", len(toks), len(tgts)))
                f.write(struct.pack(f"<{len(toks)}i", *toks))
                f.write(struct.pack(f"<{len(tgts)}I", *tgts))

    rng = np.random.default_rng(99)
    # Include mixed short lengths (packed together) AND trajectories > ubatch=64 (chunked single-seq fallback)
    seqs1 = []
    for k in range(19):
        ln = int(rng.integers(5, 95))
        toks = rng.integers(1, 50, size=ln).tolist()
        tgts = sorted({max(1, ln // 2), ln - 1})
        seqs1.append((toks, tgts))
    seqs2 = [
        (rng.integers(1, 50, size=int(rng.integers(6, 80))).tolist(), [int(ln_t)])
        for ln_t in (3, 5, 2, 4, 1, 5, 3)
    ]

    trj1 = tmp_path / "t1.trj"
    trj2 = tmp_path / "t2.trj"
    write_trj(trj1, seqs1)
    write_trj(trj2, seqs2)
    dummy_model = tmp_path / "m.gguf"
    dummy_model.write_bytes(b"stub")

    tc = Toolchain(tmp_path, tmp_path / "traj-kl", ["", ""], logodds=bin_path)
    s_ref = BatchScorer(tc, slots_per_gpu=1, ubatch=64, n_seq=1)
    s_packed = BatchScorer(tc, slots_per_gpu=2, ubatch=64, n_seq=4)

    ref1 = s_ref.score_trj(str(dummy_model), str(trj1), (10, 20), gpu="")
    ref2 = s_ref.score_trj(str(dummy_model), str(trj2), (30, 20), gpu="")
    got1, got2 = s_packed.score_trjs(str(dummy_model), [(str(trj1), (10, 20)), (str(trj2), (30, 20))])

    np.testing.assert_allclose(got1, ref1, rtol=0, atol=0)
    np.testing.assert_allclose(got2, ref2, rtol=0, atol=0)

