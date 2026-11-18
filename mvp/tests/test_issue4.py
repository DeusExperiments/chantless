"""Tests for Issue #4: generative `traj-kl` one-shot search (`kind == "kl"`), universal `up_row_zero` switch Apprentice
(`learn_gates_seq` across all ggml quant types including `IQ2_XS`/`IQ1_S` with replay KL + A-GEM), chat/tool token
protection in `vocab.rule_p_keep_set`, and unified `llama.cpp` `a01dff9` runtime verification.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "mvp"))

from chantless import builder, gguf_ops, lattice, mask as mk, oneshot, sanity, vocab


def _sha256(p: str | Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _pack_kv_str(s: str) -> tuple[int, bytes]:
    b = s.encode("utf-8")
    return 8, struct.pack("<Q", len(b)) + b


def _pack_kv_u32(v: int) -> tuple[int, bytes]:
    return 4, struct.pack("<I", v)


def _pack_kv_i32_arr(vals: list[int]) -> tuple[int, bytes]:
    return 9, struct.pack("<IQ", 5, len(vals)) + struct.pack(f"<{len(vals)}i", *vals)


def _pack_kv_str_arr(items: list[bytes]) -> tuple[int, bytes]:
    b = bytearray(struct.pack("<IQ", 8, len(items)))
    for x in items:
        b += struct.pack("<Q", len(x)) + x
    return 9, bytes(b)


def _write_gguf(path: Path, kvs: list[tuple[str, tuple[int, bytes]]], tensors: list[tuple[str, list[int], int, bytes]]) -> None:
    kv_buf = bytearray()
    for k, (vt, raw) in kvs:
        kb = k.encode("utf-8")
        kv_buf += struct.pack("<Q", len(kb)) + kb + struct.pack("<I", vt) + raw
    gguf_ops._write_gguf_raw(3, 32, len(kvs), bytes(kv_buf), tensors, path)


# ------------------------------------------------------------------ 1. Unified runtime patch check
def test_unified_a01dff9_patch_structure_and_clean_apply(tmp_path: Path):
    patch_file = ROOT / "mvp/patches/llama-cpp-a01dff9-chantless.patch"
    assert patch_file.exists()
    txt = patch_file.read_text()
    # Must cover all 3 architectures + hparams KV drop tracking + offline server UI build default
    for required in (
        "src/llama-hparams.h",
        "src/llama-hparams.cpp",
        "src/llama-model.cpp",
        "src/models/qwen3.cpp",
        "src/models/qwen35.cpp",
        "src/models/gemma4.cpp",
        "hparams.attn_dropped",
        "hparams.n_ff(",
    ):
        assert required in txt, f"unified patch missing {required}"

    lc_repo = ROOT / ".local/mvp/llama-a01dff9"
    if lc_repo.exists():
        wt = tmp_path / "wt"
        subprocess.run(["git", "worktree", "add", "--detach", str(wt), "a01dff908b9c9311d7af316bba3c918063749bd0"],
                       cwd=lc_repo, check=True, capture_output=True)
        try:
            subprocess.run(["git", "apply", "--check", str(patch_file)], cwd=wt, check=True, capture_output=True)
        finally:
            subprocess.run(["git", "worktree", "remove", "--force", str(wt)], cwd=lc_repo, capture_output=True)


# ------------------------------------------------------------------ 2. Chat / tool / reasoning token protection
def test_vocab_rule_p_protects_chat_and_tool_tags(tmp_path: Path):
    tokens = [
        b"<unk>", b"<s>", b"<tool_call>", b"</tool_call>", b"<think>", b"</think>",
        b"<|im_end|>", b"common0", b"common1", b"rare0", b"rare1", b"rare2",
    ]
    # Mark `<tool_call>`..`<|im_end|>` as `ttype = 1` (NORMAL) with `count = 0`, mimicking HF `special=false` added tokens
    ttypes = [3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    kvs = [
        ("general.architecture", _pack_kv_str("qwen3")),
        ("tokenizer.ggml.tokens", _pack_kv_str_arr(tokens)),
        ("tokenizer.ggml.token_type", _pack_kv_i32_arr(ttypes)),
        ("tokenizer.ggml.merges", _pack_kv_str_arr([])),
    ]
    gg = tmp_path / "tok.gguf"
    _write_gguf(gg, kvs, [("output_norm.weight", [32], 0, np.ones(32, dtype=np.float32).tobytes())])
    counts = np.array([0, 0, 0, 0, 0, 0, 0, 500, 400, 10, 5, 1], dtype=np.int64)
    keep = vocab.rule_p_keep_set(gg, counts, k=9)
    kept_bytes = {tokens[int(i)] for i in keep}
    for tag in (b"<tool_call>", b"</tool_call>", b"<think>", b"</think>", b"<|im_end|>"):
        assert tag in kept_bytes, f"control tag {tag!r} was dropped!"
    assert b"common0" in kept_bytes and b"common1" in kept_bytes
    assert len(keep) == 9


# ------------------------------------------------------------------ 3. Universal `up_row_zero` neuron switches across IQ1_S / IQ2_XXS / IQ2_XS / Q4_K
def test_up_row_zero_switches_across_iq_and_k_quant_types(tmp_path: Path):
    n_embd, n_ff = 256, 8
    # Layer 0: IQ1_S (ttype=19, 50B), Layer 1: IQ2_XS (ttype=17, 74B), Layer 2: IQ2_XXS (ttype=16, 66B), Layer 3: Q4_K (ttype=12, 144B)
    specs = [(0, 19, 50, {1, 5}), (1, 17, 74, {0, 3, 7}), (2, 16, 66, {2, 6}), (3, 12, 144, {4})]
    rng = np.random.default_rng(42)
    tensors = []
    for il, ttype, rbytes, _dead in specs:
        nbytes = (n_embd // 256) * rbytes * n_ff
        tensors.extend([
            (f"blk.{il}.ffn_gate.weight", [n_embd, n_ff], ttype, rng.integers(1, 255, size=nbytes, dtype=np.uint8).tobytes()),
            (f"blk.{il}.ffn_up.weight", [n_embd, n_ff], ttype, rng.integers(1, 255, size=nbytes, dtype=np.uint8).tobytes()),
            (f"blk.{il}.ffn_down.weight", [n_ff, n_embd], ttype, rng.integers(1, 255, size=nbytes, dtype=np.uint8).tobytes()),
        ])
    src = tmp_path / "iq.gguf"
    _write_gguf(src, [("general.architecture", _pack_kv_str("qwen35"))], tensors)

    dst = tmp_path / "iq_switched.gguf"
    app = {
        "switch_mode": "up_row_zero",
        "switches": {str(il): mk.pack_bitmap(sorted(dead), n_ff) for il, _, _, dead in specs},
        "levels": {},
    }
    builder._apply_apprentice(src, dst, app)
    assert dst.stat().st_size == src.stat().st_size

    art_s, art_d = lattice.Artifact(src), lattice.Artifact(dst)
    try:
        for il, _ttype, rbytes, dead in specs:
            for suf in ("ffn_up.weight", "ffn_gate.weight"):
                raw_s = np.frombuffer(bytes(art_s.raw(f"blk.{il}.{suf}")), dtype=np.uint8).reshape(n_ff, rbytes)
                raw_d = np.frombuffer(bytes(art_d.raw(f"blk.{il}.{suf}")), dtype=np.uint8).reshape(n_ff, rbytes)
                for r in range(n_ff):
                    if r in dead:
                        assert np.all(raw_d[r] == 0)
                    else:
                        assert np.array_equal(raw_d[r], raw_s[r])
    finally:
        art_s.close()
        art_d.close()


# ------------------------------------------------------------------ 4. Generative KL one-shot surrogate + SizeMatchedCap guard
def test_generative_kl_oneshot_solver_and_general_cap_guard():
    receipt_preserve = {
        "mode": "kl", "n_tgt": 15, "kl": 0.14, "nll_src": 1.2, "nll_cand": 1.35,
        "trajs": [
            {"i": 0, "n_tok": 20, "n_tgt": 5, "kl": 0.06, "nll_src": 1.0, "nll_cand": 1.1},
            {"i": 1, "n_tok": 30, "n_tgt": 10, "kl": 0.18, "nll_src": 1.3, "nll_cand": 1.475},
        ],
    }
    cand_p, src_p, tw = oneshot.extract_traj_kl(receipt_preserve, "preserve")
    fn_p = oneshot.kl_store_damage(tw)
    assert np.allclose(src_p, 0.0)
    assert fn_p(cand_p) == pytest.approx((5 * 0.06 + 10 * 0.18) / 15.0)

    cand_i, src_i, tw_i = oneshot.extract_traj_kl(receipt_preserve, "improve")
    assert fn_p(cand_i) == pytest.approx((5 * 1.1 + 10 * 1.475) / 15.0)
    assert fn_p(src_i) == pytest.approx((5 * 1.0 + 10 * 1.3) / 15.0)

    # Verify `guard_fn` prevents `oneshot.solve()` from picking a move that blows the general-KL cap when an alternative fits
    options = {"g0": ["Q8_0", "drop"], "g1": ["Q8_0", "drop"]}
    cost = {("g0", "Q8_0"): 1000, ("g0", "drop"): 0, ("g1", "Q8_0"): 1000, ("g1", "drop"): 0}
    # g0=drop has slightly lower task KL (0.01 vs 0.02), BUT blows general KL (5.0 vs 0.05)!
    task_solo = {("g0", "drop"): np.array([0.01, 0.01]), ("g1", "drop"): np.array([0.02, 0.02])}
    gen_solo = {("g0", "drop"): np.array([5.0, 5.0]), ("g1", "drop"): np.array([0.05, 0.05])}
    tw2 = np.array([1.0, 1.0])
    sur_task = oneshot.Surrogate([("task", 1.0, oneshot.kl_store_damage(tw2), np.zeros(2))], task_solo)
    sur_gen = oneshot.Surrogate([("general", 1.0, oneshot.kl_store_damage(tw2), np.zeros(2))], gen_solo)
    cap = sanity.SizeMatchedCap([(2.0, 0.50), (8.0, 0.01)])

    sol_unguarded = oneshot.solve(sur_task, options, cost, budget=1000, header=0, log=lambda *_: None)
    assert sol_unguarded == {"g0": "drop", "g1": "Q8_0"}

    def guard(m_cand, b_cand):
        gx = sum(sur_gen.shift(g, t) for g, t in m_cand.items())
        return sur_gen.damage(gx) - cap(b_cand * 8 / 2000.0)

    sol_guarded = oneshot.solve(sur_task, options, cost, budget=1000, header=0, log=lambda *_: None, guard_fn=guard)
    assert sol_guarded == {"g0": "Q8_0", "g1": "drop"}


# ------------------------------------------------------------------ 5. Live end-to-end generative smoke run on unified `a01dff9`
def test_live_end_to_end_generative_oneshot_and_learn_gates_seq_smoke(tmp_path: Path):
    """End-to-end generative smoke proof using real unified `a01dff9` C++ binaries + PyTorch Student (`IQ2_XS` + drop + width + A-GEM)."""
    venv_py = ROOT / ".local/mvp/.venv/bin/python"
    llama_dir = ROOT / ".local/mvp/llama-a01dff9"
    llama_bin = llama_dir / "build/bin"
    tools_bin = ROOT / ".local/mvp/bin-a01dff9"
    qwen_tok_dir = ROOT / ".local/mvp/models/qwen3-0.6b"

    if not (venv_py.exists() and (llama_bin / "llama-quantize").exists() and (tools_bin / "traj-kl").exists() and qwen_tok_dir.exists()):
        pytest.skip("local unified a01dff9 toolchain / venv not present")

    # Run inside `.local/mvp/.venv/bin/python` (which has torch, transformers, scipy, gguf)
    driver = tmp_path / "smoke_e2e.py"
    driver.write_text(f"""
import hashlib, json, shutil, subprocess, sys
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM

ROOT = Path({str(ROOT)!r})
TMP = Path({str(tmp_path)!r})
sys.path.insert(0, str(ROOT / "mvp"))

from chantless import builder, gguf_ops, kl, lattice, learn, mask as mk, oneshot, prober, runtime, sanity, util

llama_bin = Path({str(llama_bin)!r})
tools_bin = Path({str(tools_bin)!r})
tok_src = Path({str(qwen_tok_dir)!r})

# 1. Create a compact 3-layer Qwen3 HF checkpoint (`n_embd=256`, `n_ff=512`, `vocab_size=2048` synthetic or full Qwen3 vocab)
# Use `vocab_size=2048` with a clean subset or full vocab? Using `vocab_size=1024`? Wait: `convert_hf_to_gguf.py` reads tokenizer!
# With `n_embd=256`, even full Qwen3 `vocab_size=151936` is only `256 * 151936 * 2 = 77 MB` in BF16 (converts in 1.5s, quantizes in 0.4s)!
hf_dir = TMP / "hf_tiny"
hf_dir.mkdir()
for fn in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "special_tokens_map.json"):
    if (tok_src / fn).exists():
        shutil.copyfile(tok_src / fn, hf_dir / fn)

torch.manual_seed(20261118)
cfg = Qwen3Config(
    vocab_size=151936,
    hidden_size=256,
    intermediate_size=512,
    num_hidden_layers=3,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=64,
    max_position_embeddings=2048,
    tie_word_embeddings=True,
    torch_dtype="bfloat16",
)
cfg.save_pretrained(hf_dir)
m_hf = Qwen3ForCausalLM(cfg).to(torch.bfloat16)
m_hf.save_pretrained(hf_dir)
del m_hf

src_gguf = TMP / "src-bf16.gguf"
subprocess.run([sys.executable, str(Path({str(llama_dir / 'convert_hf_to_gguf.py')!r})), str(hf_dir), "--outtype", "bf16", "--outfile", str(src_gguf)],
               check=True, capture_output=True, text=True)
assert src_gguf.exists()

tc = runtime.Toolchain(llama_bin, tools_bin / "traj-kl", [""], logodds=tools_bin / "logodds", tokdocs=tools_bin / "tokdocs")

# 2. Build generative `task` and `general` TRJ1 stores + `imatrix.gguf` on the unified `a01dff9` binaries
R = kl.Renderer(str(hf_dir))
task_convos = [
    [{{"role": "user", "content": "Return JSON status ok."}}, {{"role": "assistant", "content": '{{"status": "ok", "code": 200}}'}}],
    [{{"role": "user", "content": "Add 2 and 3 in Python."}}, {{"role": "assistant", "content": "def add():\\n    return 2 + 3"}}],
    [{{"role": "user", "content": "Call weather tool for Tokyo."}}, {{"role": "assistant", "content": '<tool_call>\\n{{"name": "weather", "arguments": {{"city": "Tokyo"}}}}\\n</tool_call>'}}],
]
task_recs = [rec for msgs in task_convos for rec in R.episode(msgs, None)]
gen_texts = [
    "def binary_search(a, x):\\n    lo, hi = 0, len(a) - 1\\n    while lo <= hi:\\n        mid = (lo + hi) // 2\\n        if a[mid] == x: return mid\\n        elif a[mid] < x: lo = mid + 1\\n        else: hi = mid - 1\\n    return -1\\n" * 6,
    "The Fourier transform decomposes a function of time into its constituent frequencies. " * 20,
]
gen_recs = kl.pack_text(R, gen_texts, seq_len=128, n_seq=2)

pdir = TMP / "run/proxy"
pdir.mkdir(parents=True)
meta_t = kl.write_trj(pdir / "task.bin", task_recs)
meta_g = kl.write_trj(pdir / "general.bin", gen_recs)

for name in ("task", "general"):
    kl.traj_kl(tc, "base", str(src_gguf), str(pdir / f"{{name}}.bin"), str(pdir / f"store_{{name}}"), str(pdir / f"base_{{name}}.json"), str(pdir / "kl.log"))
    s_chk = kl.traj_kl(tc, "kl", str(src_gguf), str(pdir / f"{{name}}.bin"), str(pdir / f"store_{{name}}"), str(pdir / f"self_{{name}}.json"), str(pdir / "kl.log"))
    assert s_chk["kl"] < 1e-4, s_chk

(pdir / "stats_corpus.txt").write_text("\\n\\n".join(gen_texts))
runtime.imatrix(tc, str(src_gguf), str(pdir / "stats_corpus.txt"), str(pdir / "imatrix.gguf"), str(pdir / "im.log"), ctx=128)

groups, dead = gguf_ops.adapter_from_gguf(src_gguf)
assert len(dead) == 0
# Annotate imatrix_ok from imatrix tensor names
im_names = {{t["name"].split(".in_sum")[0] for t in gguf_ops.inventory(pdir / "imatrix.gguf")}}
for g, info in groups.items():
    if not g.startswith("_"):
        info["imatrix_ok"] = all(t in im_names for t in info["quantizable"])
assert groups["blk.0.ffn"]["imatrix_ok"] is True
assert groups["token_embd"]["imatrix_ok"] is False

# 3. Run sanity uniform + solo probes (`Q4_K`, `IQ2_XS`, `drop`) + Wanda width probes (`512 -> 256`) and solve via `_search_oneshot`!
B_san = builder.Builder(tc, str(src_gguf), groups, str(pdir / "imatrix.gguf"), TMP / "work_san", threads=4)
P_san = prober.Prober(tc, B_san, TMP / "run/probe/sanity", str(TMP / "run/probe.log"), workers=1, scratch=TMP / "cand_san")
P_san.add_store(prober.Store("task", str(pdir / "task.bin"), str(pdir / "store_task"), meta_t["n_targets"], kind="kl"))
P_san.add_store(prober.Store("general", str(pdir / "general.bin"), str(pdir / "store_general"), meta_g["n_targets"], kind="kl"))

base_m = mk.new_mask(groups, "BF16", {{"repo": "tiny-qwen3", "revision": "local", "gguf_sha256": util.sha256_file(src_gguf)}})
ladder = ["BF16", "Q8_0", "Q4_K", "IQ2_XS"]
uniform = {{}}
for t in ladder[1:]:
    um = mk.new_mask(groups, "BF16", base_m["source"])
    for g in um["groups"]:
        um["groups"][g]["tier"] = mk.clip(groups[g], t, ladder)
    r = P_san.probe(um, ["general", "task"])
    uniform[t] = {{"kl_gen": r.kl["general"], "kl_task": r.kl["task"], "bytes": r.bytes, "bpw": r.bpw}}
(TMP / "run/sanity").mkdir(parents=True)
util.write_json(TMP / "run/sanity/uniform.json", uniform)
util.write_json(TMP / "run/sanity/sanity_mask.json", base_m)
util.write_json(pdir / "proxy.json", {{"task": meta_t, "general": meta_g, "self_kl": {{"task": 0.0, "general": 0.0}}}})
util.write_json(TMP / "run/feedback_stores.jsonl", []) if False else (TMP / "run/feedback_stores.jsonl").write_text("")

solo_masks = []
for g in base_m["groups"]:
    for t in [x for x in ("Q4_K", "IQ2_XS") if mk.tier_allowed(groups[g], x)] + (["drop"] if groups[g]["droppable"] else []):
        solo_masks.append(mk.apply(base_m, g, t))
P_san.probe_many(solo_masks, ["general", "task"])

# Exercise `Run._search_oneshot` directly (`kind == "kl"`, Wanda `widths=True`, `IQ2_XS` + `drop` + `Q4_K` under a tight budget!)
from chantless.pipeline import Run
dummy_run = object.__new__(Run)
dummy_run.dir = TMP / "run"
dummy_run.scratch = TMP / "scratch"
dummy_run.scratch.mkdir(parents=True)
dummy_run.tc = tc
dummy_run.src = str(src_gguf)
dummy_run.probe_src = str(src_gguf)
dummy_run.groups = groups
dummy_run.dead_tensors = []
dummy_run.ladder = ladder
dummy_run.mode = "preserve"
dummy_run.classifier = False
dummy_run.env = None
dummy_run.scorer = None
dummy_run.info = gguf_ops.arch_info(src_gguf)
dummy_run.dep = {{"context": 512, "concurrency": 1, "kv_type": "f16", "max_total_memory_bytes": None}}
dummy_run.cfg = {{"sanity": {{"solo_tiers": ["Q4_K", "IQ2_XS"]}}, "probe": {{"quantize_threads": 4}}}}
dummy_run.log = print
dummy_run.records = []
dummy_run.record = lambda k, v: dummy_run.records.append((k, v))
dummy_run.mark = lambda phase, extra=None: dummy_run.records.append(("mark", phase, extra))
dummy_run._source_pin = lambda: base_m["source"]

P_search = dummy_run.make_prober(str(src_gguf), "search")
# Set `vocab_k = 8192` + `target_b` strictly below trimmed uniform Q4_K so solver exercises Rule-P vocab trim + IQ2_XS / width cuts / drop!
from chantless import vocab as vocabmod
bm = oneshot.ByteModel(str(src_gguf), groups)
bm.calibrate(dummy_run.uniform_mask(base_m, "Q4_K"), uniform["Q4_K"]["bytes"])
keep_est = vocabmod.rule_p_keep_set(str(src_gguf), np.zeros(151936, dtype=np.int64), k=8192)
m_q4_trim = mk.upgrade_v2(dummy_run.uniform_mask(base_m, "Q4_K"))
m_q4_trim["vocab_keep"] = mk.pack_bitmap(keep_est, 151936)
target_b = bm(m_q4_trim) - 450_000  # ~450 KB tighter than trimmed uniform Q4_K (layers alone are ~1.1 MB in Q4_K)
cap = sanity.NoCap()
lc = {{
    "algo": "oneshot",
    "target_bytes": target_b,
    "vocab_k": 8192,
    "oneshot_tiers": ["Q8_0", "Q4_K", "IQ2_XS"],
    "widths": True,
    "width_fractions": [0.5],
    "width_step": 256,
    "allow_drop": True,
    "enforce_general_cap": False,
}}
v3_mask = dummy_run._search_oneshot(P_search, base_m, lc, cap=cap)
assert v3_mask["schema"] == "chantless.mask/v3"
assert v3_mask["vocab_keep"] is not None

# Ensure at least one surviving MLP is IQ2_XS so we also prove IQ2_XS Student + `up_row_zero` switch parity!
kept_ffns = [int(g.split(".")[1]) for g, e in v3_mask["groups"].items() if g.endswith(".ffn") and e["tier"] != "drop"]
if not kept_ffns:
    v3_mask["groups"]["blk.0.ffn"]["tier"] = "IQ2_XS"
    kept_ffns = [0]
else:
    v3_mask["groups"][f"blk.{{kept_ffns[0]}}.ffn"]["tier"] = "IQ2_XS"
mk.validate(v3_mask, groups)

spec_base = TMP / "specialist_oneshot.gguf"
br_base = builder.Builder(tc, str(src_gguf), groups, str(pdir / "imatrix.gguf"), TMP / "work_build", threads=4).build(v3_mask, spec_base)
assert abs(bm(v3_mask) - br_base.file_bytes) <= 32

# Verify `<tool_call>` and `</tool_call>` survived Rule-P vocabulary trimming as single special tokens in `spec_base`!
tag_toks = runtime.tokenize_docs(tc, spec_base, ["<tool_call>", "</tool_call>", "<|im_end|>"], threads=2)
assert all(len(x) == 1 for x in tag_toks), f"chat/tool control tags split in trimmed GGUF: {{tag_toks}}"

# 4. Tokenize lessons & replay trajectories against `spec_base`'s trimmed GGUF vocab, then run `learn_gates_seq` (`up_row_zero` + A-GEM)!
prompt_str = R._render([{{"role": "user", "content": "Output magic token sequence."}}], None, True)
prompt_ids, good_ids, bad_ids = [x.tolist() for x in runtime.tokenize_docs(tc, spec_base, [prompt_str, "ALPHA_OK", "BETA_BAD"], threads=2)]
lessons = [{{"prompt_tokens": prompt_ids, "corrected_tokens": good_ids, "mistaken_tokens": bad_ids}} for _ in range(4)]

replay_strs = []
for msgs in task_convos:
    replay_strs.append(R._render(msgs[:-1], None, True))
    replay_strs.append(R._render(msgs, None, False))
replay_toks = [x.tolist() for x in runtime.tokenize_docs(tc, spec_base, replay_strs, threads=2)]
spec_replay_recs = []
for i in range(0, len(replay_toks), 2):
    pre, full = replay_toks[i], replay_toks[i + 1]
    assert full[:len(pre)] == pre and len(full) > len(pre)
    spec_replay_recs.append((full, list(range(len(pre), len(full)))))

switched_gguf = TMP / "specialist_switched.gguf"
switched_mask_path = TMP / "specialist_switched.mask.json"
rec_learn = learn.learn_gates_seq(
    str(spec_base),
    str(hf_dir),
    lessons,
    spec_replay_recs,
    str(switched_gguf),
    cfg={{
        "ggml_lib": str(llama_bin),
        "steps": 45,
        "gate_lr": 0.15,
        "gate_init": 0.25,
        "margin": 1.0,
        "nll_weight": 1.0,
        "margin_weight": 2.0,
        "replay_weight": 1.0,
        "project": True,
        "switch_mode": "up_row_zero",
    }},
    device="cpu",
    mask_in=v3_mask,
    mask_out=switched_mask_path,
)
assert rec_learn["off"] > 0, f"expected >0 switched-off neurons, got {{rec_learn}}"
assert rec_learn["bytes"] == br_base.file_bytes
assert rec_learn["loss_last"][0] < rec_learn["loss_first"][0]  # NLL on corrected output strictly improved!

# 5. Rebuild from `(frozen BF16 source + thin v3 mask)` via `python -m chantless.cli build` and verify bit-identical SHA256!
cli_rebuilt = TMP / "cli_rebuilt.gguf"
subprocess.run([
    sys.executable, "-m", "chantless.cli", "build",
    "--source", str(src_gguf),
    "--mask", str(switched_mask_path),
    "--imatrix", str(pdir / "imatrix.gguf"),
    "--llama-bin", str(llama_bin),
    "--threads", "4",
    "--shards", "1",
    "--work", str(TMP / "work_build"),
    "--out", str(cli_rebuilt),
], cwd=str(ROOT / "mvp"), check=True, capture_output=True, text=True)

assert cli_rebuilt.stat().st_size == switched_gguf.stat().st_size
assert util.sha256_file(cli_rebuilt) == util.sha256_file(switched_gguf), "cli build from thin v3 mask must match learn_gates_seq GGUF byte-for-byte!"

# 6. Verify PyTorch Student vs REAL C++ `traj-kl` trajectory log-prob / NLL parity on the rebuilt GGUF!
eval_trj = TMP / "eval_lesson.bin"
pos_seq = prompt_ids + good_ids
neg_seq = prompt_ids + bad_ids
eval_recs = [
    (pos_seq, list(range(len(prompt_ids), len(pos_seq)))),
    (neg_seq, list(range(len(prompt_ids), len(neg_seq)))),
    *spec_replay_recs[:2],
]
kl.write_trj(eval_trj, eval_recs)
kl.traj_kl(tc, "base", str(spec_base), str(eval_trj), str(TMP / "store_eval"), str(TMP / "base_eval.json"), str(TMP / "eval.log"))
cpp_eval = kl.traj_kl(tc, "kl", str(cli_rebuilt), str(eval_trj), str(TMP / "store_eval"), str(TMP / "cand_eval.json"), str(TMP / "eval.log"))

art_reb = lattice.Artifact(str(cli_rebuilt))
try:
    stu_reb, _ = learn.build_student(art_reb, str(hf_dir), [], str(llama_bin), device="cpu")
finally:
    art_reb.close()
stu_eval = learn.student_traj_nll(stu_reb, eval_recs, device="cpu")

diffs = [abs(s["nll_cand"] - c["nll_cand"]) for s, c in zip(stu_eval, cpp_eval["trajs"])]
print("Student vs C++ traj-kl max |NLL diff|:", max(diffs), "per-traj:", list(zip([round(s["nll_cand"], 5) for s in stu_eval], [round(c["nll_cand"], 5) for c in cpp_eval["trajs"]])))
assert max(diffs) < 5e-3, f"Student vs C++ GGUF NLL diverged: {{diffs}}"

# Verify lesson improvement on REAL C++ GGUF (`nll_cand` vs `nll_src` on `pos_seq` vs `neg_seq`)
pos_tr, neg_tr = cpp_eval["trajs"][0], cpp_eval["trajs"][1]
gap_before = (-pos_tr["nll_src"]) - (-neg_tr["nll_src"])
gap_after = (-pos_tr["nll_cand"]) - (-neg_tr["nll_cand"])
print(f"Real C++ GGUF lesson log-prob margin: {{gap_before:.6f}} -> {{gap_after:.6f}} (delta={{gap_after - gap_before:+.6f}}), thin mask={{switched_mask_path.stat().st_size}} B, off={{rec_learn['off']}}")
assert gap_after > gap_before, f"Real C++ GGUF margin did not improve: {{gap_before}} -> {{gap_after}}"

# 7. Run `Run.phase_freeze()` on the switched `chantless.mask/v3` mask (double-build SHA256, header audit, `llama-server` chat + memory)!
util.write_json(TMP / "run/search_result.json", {{"mask": mk.load(switched_mask_path), "weights": str(src_gguf)}})
dummy_run.phase_freeze()
freeze_rec = util.read_json(TMP / "run/final/freeze.json")
assert freeze_rec["audit"]["rebuild_sha256_equal"] is True
assert freeze_rec["audit"]["types_match"] is True
assert freeze_rec["audit"]["dropped_absent"] is True
assert freeze_rec["audit"]["sha256"] == util.sha256_file(switched_gguf)
assert freeze_rec["deployment_memory"]["ok"] is True
assert freeze_rec["build"]["file_bytes"] == switched_gguf.stat().st_size
assert util.read_json(TMP / "run/final/mask.json")["measured"]["memory"]["decode_tokens"] >= 1
print("Freeze audit & LlamaServer memory receipt:", json.dumps(freeze_rec))

print("E2E_GENERATIVE_ISSUE4_OK")
""")
    proc = subprocess.run([str(venv_py), str(driver)], capture_output=True, text=True)
    if proc.returncode != 0:
        raise AssertionError(f"E2E smoke failed (rc={proc.returncode}):\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")
    print(proc.stdout)
    assert "E2E_GENERATIVE_ISSUE4_OK" in proc.stdout


# ------------------------------------------------------------------ 6. Qwen3.5 VLM config + GatedDeltaNet (num_k != num_v) parity on unified `a01dff9`
def test_qwen35_vlm_config_and_gdn_v_head_unreorder_parity(tmp_path: Path):
    """Verify stock VLM `Qwen3_5Config` (`text_config`), `num_k_heads != num_v_heads` V-head un-reordering, variable width, vocab trim, and `up_row_zero`."""
    venv_py = ROOT / ".local/mvp/.venv/bin/python"
    llama_dir = ROOT / ".local/mvp/llama-a01dff9"
    llama_bin = llama_dir / "build/bin"
    tools_bin = ROOT / ".local/mvp/bin-a01dff9"
    qwen_tok_dir = ROOT / ".local/mvp/models/qwen3-0.6b"

    if not (venv_py.exists() and (llama_bin / "llama-quantize").exists() and (tools_bin / "traj-kl").exists() and qwen_tok_dir.exists()):
        pytest.skip("local unified a01dff9 toolchain / venv not present")

    driver = tmp_path / "q35_parity.py"
    driver.write_text(f"""
import json, subprocess, sys
from pathlib import Path
import numpy as np, torch
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config, Qwen3_5TextConfig
from transformers import AutoModelForCausalLM

ROOT = Path({str(ROOT)!r})
TMP = Path({str(tmp_path)!r})
sys.path.insert(0, str(ROOT / "mvp"))
from chantless import builder, gguf_ops, kl, lattice, learn, mask as mk, runtime, vocab

hf = TMP / "hf_q35"
hf.mkdir()
tok_src = Path({str(qwen_tok_dir)!r})
for fn in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"):
    if (tok_src / fn).exists():
        (hf / fn).write_bytes((tok_src / fn).read_bytes())

torch.manual_seed(7)
tcfg = Qwen3_5TextConfig(
    vocab_size=151936, hidden_size=256, intermediate_size=512, num_hidden_layers=2,
    num_attention_heads=4, num_key_value_heads=2, head_dim=64,
    linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=64, linear_value_head_dim=64,
    layer_types=["linear_attention", "full_attention"], tie_word_embeddings=True, torch_dtype="bfloat16",
)
m = AutoModelForCausalLM.from_config(tcfg, dtype=torch.bfloat16)
with torch.no_grad():
    m.model.norm.weight.uniform_(-0.2, 0.2)
    for l in m.model.layers:
        l.input_layernorm.weight.uniform_(-0.2, 0.2)
        l.post_attention_layernorm.weight.uniform_(-0.2, 0.2)
m.save_pretrained(hf)
vcfg = Qwen3_5Config(text_config=tcfg.to_dict(), tie_word_embeddings=True)
d = vcfg.to_dict()
d["architectures"] = ["Qwen3_5ForConditionalGeneration"]
(hf / "config.json").write_text(json.dumps(d, indent=2))

gg = TMP / "q35.gguf"
subprocess.run([sys.executable, {str(llama_dir / 'convert_hf_to_gguf.py')!r}, str(hf), "--no-mtp", "--outtype", "bf16", "--outfile", str(gg)],
               check=True, capture_output=True, text=True)

tc = runtime.Toolchain(Path({str(llama_bin)!r}), Path({str(tools_bin / 'traj-kl')!r}), [""], tokdocs=Path({str(tools_bin / 'tokdocs')!r}))
groups, _ = gguf_ops.adapter_from_gguf(gg)
m3 = mk.new_mask(groups, "Q4_K", {{"repo": "tiny-q35", "revision": "local", "gguf_sha256": "0"}})
keep = vocab.rule_p_keep_set(gg, np.zeros(151936, dtype=np.int64), k=4096)
m3["vocab_keep"] = mk.pack_bitmap(keep, 151936)
m3["widths"] = {{"0": mk.pack_bitmap(list(range(256)), 512)}}
m3["apprentice"] = {{"switch_mode": "up_row_zero", "switches": {{"0": mk.pack_bitmap([1, 7, 42], 256), "1": mk.pack_bitmap([0, 19], 512)}}, "levels": {{}}, "decision": {{}}}}
built = TMP / "q35_built.gguf"
builder.Builder(tc, str(gg), groups, None, TMP / "wk", threads=4).build(m3, built)

recs = [([10, 20, 30, 40, 50, 60], [2, 3, 4, 5]), ([15, 25, 35, 45, 55], [1, 2, 3, 4])]
kl.write_trj(TMP / "t.bin", recs)
kl.traj_kl(tc, "base", str(built), str(TMP / "t.bin"), str(TMP / "st"), str(TMP / "b.json"), str(TMP / "l.log"))
cpp = kl.traj_kl(tc, "kl", str(built), str(TMP / "t.bin"), str(TMP / "st"), str(TMP / "c.json"), str(TMP / "l.log"))
art = lattice.Artifact(str(built))
try:
    stu, _ = learn.build_student(art, str(hf), [], {str(llama_bin)!r}, device="cpu")
finally:
    art.close()
st_nll = learn.student_traj_nll(stu, recs, device="cpu")
diffs = [abs(s["nll_cand"] - c["nll_cand"]) for s, c in zip(st_nll, cpp["trajs"])]
print("Qwen3.5 VLM+GDN(k=2,v=4)+cut+vocab+up_row_zero max |NLL diff|:", max(diffs))
assert max(diffs) < 5e-3, f"Qwen3.5 Student vs C++ NLL diverged: {{diffs}}"
print("QWEN35_PARITY_OK")
""")
    proc = subprocess.run([str(venv_py), str(driver)], capture_output=True, text=True)
    if proc.returncode != 0:
        raise AssertionError(f"Qwen3.5 parity failed (rc={proc.returncode}):\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")
    print(proc.stdout)
    assert "QWEN35_PARITY_OK" in proc.stdout
