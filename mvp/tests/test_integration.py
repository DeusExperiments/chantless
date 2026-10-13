"""Model-backed integration tests. Require CHANTLESS_MVP_ROOT (built llama.cpp fork, traj-kl, Qwen3-0.6B GGUF).

Run: CHANTLESS_MVP_ROOT=.local/mvp pytest -q tests/test_integration.py
"""
import os
import re
import tempfile
from pathlib import Path

import pytest

ROOT = os.environ.get("CHANTLESS_MVP_ROOT")
pytestmark = pytest.mark.skipif(not ROOT, reason="set CHANTLESS_MVP_ROOT")


@pytest.fixture(scope="module")
def ctx():
    from chantless import gguf_ops
    from chantless.runtime import Toolchain
    root = Path(ROOT)
    gpus = os.environ.get("CHANTLESS_GPUS", "").split(",") if os.environ.get("CHANTLESS_GPUS") else [""]
    tc = Toolchain(root / "llama.cpp/build/bin", root / "bin/traj-kl", gpus)
    src = str(root / "models/qwen3-0.6b-bf16.gguf")
    G = gguf_ops.groups_from_inventory(gguf_ops.inventory(src))
    d = Path(tempfile.mkdtemp(prefix="chantless-it-", dir=root))
    return tc, src, G, d, root


def _trj(ctx):
    """Two short chat sequences rendered with the real template, all assistant tokens scored."""
    from chantless import kl
    tc, src, G, d, root = ctx
    R = kl.Renderer(str(root / "models/qwen3-0.6b"))
    msgs = [[{"role": "user", "content": "Write a python function that adds two numbers."},
             {"role": "assistant", "content": "```python\ndef add(a, b):\n    return a + b\n```"}],
            [{"role": "user", "content": "Name three primary colours."},
             {"role": "assistant", "content": "Red, yellow and blue are the traditional primary colours."}]]
    recs = [r for m in msgs for r in R.episode(m, None)]
    kl.write_trj(d / "t.bin", recs)
    return str(d / "t.bin")


def test_group_map_partitions_header(ctx):
    tc, src, G, d, _ = ctx
    assert len([g for g in G if g.endswith(".attn")]) == 28 and len([g for g in G if g.endswith(".ffn")]) == 28


def test_bytes_golden_and_zero_credit(ctx):
    from chantless import builder, gguf_ops, mask as mk
    tc, src, G, d, _ = ctx
    B = builder.Builder(tc, src, G, None, d / "b")
    q8 = B.build(mk.new_mask(G, "Q8_0", {}), d / "q8.gguf")
    assert q8.file_bytes == gguf_ops.file_bytes(d / "q8.gguf") and not q8.fallbacks
    z = d / "zero.gguf"
    gguf_ops.copy_with_zeroed(src, z, {"blk.3.ffn_down.weight"})
    assert gguf_ops.file_bytes(z) == gguf_ops.file_bytes(src)  # zeros earn nothing
    dr = B.build(mk.apply(mk.new_mask(G, "Q8_0", {}), "blk.3.ffn", "drop"), d / "dr.gguf")
    assert dr.file_bytes < q8.file_bytes and "blk.3.ffn_down.weight" in dr.dropped


def test_rebuild_is_bit_identical(ctx):
    from chantless import builder, mask as mk, util
    tc, src, G, d, _ = ctx
    B = builder.Builder(tc, src, G, None, d / "b2")
    m = mk.apply(mk.apply(mk.new_mask(G, "Q6_K", {}), "blk.2.attn", "Q3_K"), "blk.7.ffn", "drop")
    a, b = B.build(m, d / "a.gguf"), B.build(m, d / "b.gguf")
    assert util.sha256_file(a.path) == util.sha256_file(b.path)


def test_drop_equivalent_to_zeroed_output(ctx):
    """Patch acceptance (manual L3 analogue): dropped sub-block == source with that sub-block's output zeroed."""
    from chantless import builder, gguf_ops, kl, mask as mk
    tc, src, G, d, _ = ctx
    t = _trj(ctx)
    m = mk.apply(mk.apply(mk.new_mask(G, "BF16", {}), "blk.5.ffn", "drop"), "blk.9.attn", "drop")
    dropped = builder.Builder(tc, src, G, None, d / "b3").build(m, d / "dropped.gguf")
    gguf_ops.copy_with_zeroed(src, d / "zeroed.gguf", {"blk.5.ffn_down.weight", "blk.9.attn_output.weight"})
    kl.traj_kl(tc, "base", str(d / "zeroed.gguf"), t, str(d / "zs"), str(d / "zb.json"), None)
    r = kl.traj_kl(tc, "kl", dropped.path, t, str(d / "zs"), str(d / "eq.json"), None)
    # numeric floor of this backend on the same sequences: zeroed model against its own store
    floor = kl.traj_kl(tc, "kl", str(d / "zeroed.gguf"), t, str(d / "zs"), str(d / "zself.json"), None)["kl"]
    assert r["kl"] <= max(1e-5, 3 * floor) and r["top1"] == 1, (r["kl"], floor)


def test_dropped_attention_allocates_no_kv(ctx):
    from chantless import builder, mask as mk
    from chantless.runtime import LlamaServer
    tc, src, G, d, _ = ctx
    m = mk.apply(mk.new_mask(G, "Q8_0", {}), "blk.9.attn", "drop")
    p = builder.Builder(tc, src, G, None, d / "b4").build(m, d / "kv.gguf").path
    log = d / "kv.log"
    with LlamaServer(tc, p, slots=1, ctx_per_slot=4096, log=log, extra=["-v"]):
        pass
    txt = log.read_text(errors="ignore")
    assert re.search(r"layer\s+9: does not have KV cache", txt)
    size = float(re.findall(r"llama_kv_cache: size =\s+([\d.]+) MiB", txt)[-1])
    assert abs(size * 2 ** 20 - builder.kv_bytes({"n_head_kv": 8, "head_dim": 128}, 27, 4096, 1, "f16")) < 1


def test_self_kl_floor(ctx):
    from chantless import kl
    tc, src, G, d, _ = ctx
    t = _trj(ctx)
    kl.traj_kl(tc, "base", src, t, str(d / "ss"), str(d / "sb.json"), None)
    assert kl.traj_kl(tc, "kl", src, t, str(d / "ss"), str(d / "sk.json"), None)["kl"] <= 5e-5
