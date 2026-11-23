"""Local integration tests for environments, Gemma 4 parity, and CLI workflows:
  1. Multi-dialect BFCL tool-call AST environment (`Qwen3` JSON, `Qwen3.5` XML, `Gemma 4` `<|tool_call>call:...`)
  2. DeepMath integer-reasoning environment (`\\boxed{INT}` extraction & SplitGuard conformance)
  3. Calibrated classifier environment (`clsbench` + `BatchScorer` + `vocab_keep` retokenization)
  4. Gemma 4 asymmetric tag protection (`<|turn>`, `<turn|>`, `<|tool_call>`, `<tool_call|>`, `<|"|>`)
  5. Unified `learn.build_student` + `learn_gates_seq` on `Gemma 4` (PLE, shared-KV, `Q3_K`, dropped PLE/attn, width cut, softcap, C++ `traj-kl` parity)
  6. `chantless {extract, build, learn}` CLI + generative `empty`/`flipped`/`shuffled` controls
"""
from __future__ import annotations

import importlib.util
import json
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

from conftest import resolve_model_dir, resolve_toolchain
from chantless import gguf_ops, vocab
from chantless.env import bfcl, deepmath, protocol


def _pack_kv_str(s: str) -> tuple[int, bytes]:
    b = s.encode("utf-8")
    return 8, struct.pack("<Q", len(b)) + b


def _pack_kv_i32_arr(vals: list[int]) -> tuple[int, bytes]:
    return 9, struct.pack("<IQ", 5, len(vals)) + struct.pack(f"<{len(vals)}i", *vals)


def _pack_kv_str_arr(items: list[bytes]) -> tuple[int, bytes]:
    b = bytearray(struct.pack("<IQ", 8, len(items)))
    for x in items:
        b += struct.pack("<Q", len(x)) + x
    return 9, bytes(b)


# ------------------------------------------------------------------ 1. BFCL 3-dialect parser/formatter + env conformance
def test_bfcl_three_dialect_roundtrip_and_env_conformance(tmp_path: Path):
    call = {
        "name": "math.integrate",
        "arguments": {
            "expr": "x**2 + 3*x",
            "bounds": [0, 10],
            "exact": True,
            "meta": {"unit": "m"},
        },
    }
    for dialect in ("qwen3_json", "qwen35_xml", "gemma4"):
        rendered = bfcl.format_tool_call(call, dialect)
        parsed = bfcl.parse_tool_calls(rendered)
        assert len(parsed) == 1, (dialect, rendered, parsed)
        assert bfcl.calls_match(parsed, [call]), (dialect, parsed, call)

    # Check formula normalization (`x^2` -> `x**2`) and numeric string coercion
    pred_variant = bfcl.parse_tool_calls(
        '<tool_call>\n<function=math.integrate>\n<parameter=expr>\nx^2 + 3*x\n</parameter>\n'
        '<parameter=bounds>\n[0, 10]\n</parameter>\n<parameter=exact>\ntrue\n</parameter>\n'
        '<parameter=meta>\n{"unit": "m"}\n</parameter>\n</function>\n</tool_call>'
    )
    assert bfcl.calls_match(pred_variant, [call])

    tok_dir = resolve_model_dir("qwen3-0.6b")
    has_ml_deps = importlib.util.find_spec("transformers") is not None
    if tok_dir.exists() and has_ml_deps:
        pairs = tmp_path / "bfcl_pairs.jsonl"
        rows = []
        for i, sp in enumerate(("search", "select", "exam", "train")):
            rows.append({
                "id": f"simple:{i}",
                "category": "simple",
                "family": f"simple/{sp}",
                "split": sp,
                "messages": [{"role": "user", "content": f"Compute f({i})."}],
                "tools": [{"type": "function", "function": {"name": "calc", "parameters": {"type": "dict", "properties": {"x": {"type": "int"}}}}}],
                "reference_calls": [{"name": "calc", "arguments": {"x": i}}],
            })
        pairs.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        proc = subprocess.run([
            sys.executable, "-c",
            f"import sys; sys.path.insert(0, {str(ROOT / 'mvp')!r}); "
            f"from chantless.env.protocol import load_environment; "
            f"env = load_environment('chantless.env.bfcl', hf_dir={str(tok_dir)!r}, pairs_file={str(pairs)!r}); "
            f"t0 = env.tasks()[0]; "
            f"from chantless.env.protocol import Episode; "
            f"ep = Episode(t0.id, 0, t0.prompt['messages'] + [{{'role': 'assistant', 'content': env.format_reference(t0)}}], t0.prompt['tools'], env.format_reference(t0), ['stop'], 1, 12, 0.1); "
            f"v = env.verify(t0, ep); assert v.success, v; print('BFCL_ENV_OK')"
        ], capture_output=True, text=True)
        assert proc.returncode == 0 and "BFCL_ENV_OK" in proc.stdout, proc.stderr


# ------------------------------------------------------------------ 2. DeepMath `\boxed{INT}` extraction + env conformance
def test_deepmath_boxed_int_and_env_conformance(tmp_path: Path):
    assert deepmath.extract_boxed_int("Step 1... \\boxed{42}") == 42
    assert deepmath.extract_boxed_int("<think>\\boxed{99}</think> Therefore \\boxed{-1,234}.") == -1234
    assert deepmath.extract_boxed_int("<|channel>thought\ntry 5<channel|>Final: \\boxed{\\text{256}}") == 256
    assert deepmath.extract_boxed_int("First \\boxed{10}, corrected \\boxed{2048}") == 2048
    assert deepmath.extract_boxed_int("No boxed answer 42") is None

    tok_dir = resolve_model_dir("qwen3-0.6b")
    has_ml_deps = importlib.util.find_spec("transformers") is not None
    if tok_dir.exists() and has_ml_deps:
        dm_file = tmp_path / "dm.jsonl"
        rows = [
            {"id": f"q{i}", "question": f"What is {i+2} * 3?", "answer": (i + 2) * 3, "difficulty": 5.0, "area": "Algebra", "family": f"alg/{sp}", "split": sp}
            for i, sp in enumerate(("search", "select", "exam", "train"))
        ]
        dm_file.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        proc = subprocess.run([
            sys.executable, "-c",
            f"import sys; sys.path.insert(0, {str(ROOT / 'mvp')!r}); "
            f"from chantless.env.protocol import load_environment, Episode; "
            f"env = load_environment('chantless.env.deepmath', hf_dir={str(tok_dir)!r}, jsonl_file={str(dm_file)!r}); "
            f"t0 = env.tasks()[0]; ans = f'Reasoning... \\\\boxed{{{{{{t0.meta[\"answer\"]}}}}}}'; "
            f"ep = Episode(t0.id, 0, t0.prompt['messages'] + [{{'role': 'assistant', 'content': ans}}], None, ans, ['stop'], 1, 10, 0.1); "
            f"v = env.verify(t0, ep); assert v.success, v; print('DEEPMATH_ENV_OK')"
        ], capture_output=True, text=True)
        assert proc.returncode == 0 and "DEEPMATH_ENV_OK" in proc.stdout, proc.stderr


# ------------------------------------------------------------------ 3. `clsbench` conformance + vocab_keep scoring
def test_clsbench_conformance_and_vocab_trim_scoring(tmp_path: Path):
    tok_dir = resolve_model_dir("qwen3-0.6b")
    llama_dir, tools_bin = resolve_toolchain()
    has_ml_deps = importlib.util.find_spec("torch") is not None and importlib.util.find_spec("transformers") is not None
    if not (has_ml_deps and tok_dir.exists() and (tools_bin / "logodds").exists() and (llama_dir / "build/bin/llama-quantize").exists()):
        pytest.skip("local toolchain, tokenizer model, or ML dependencies not present")

    cls_file = tmp_path / "cls.jsonl"
    rows = [
        {"id": "c0", "text": "Wonderful service and delicious food!", "label": "positive", "split": "calibration", "family": "cal/0"},
        {"id": "c1", "text": "Great atmosphere and friendly staff.", "label": "positive", "split": "calibration", "family": "cal/1"},
        {"id": "s0", "text": "Terrible wait time and cold meal.", "label": "negative", "split": "search", "family": "srch/0"},
        {"id": "l0", "text": "Awful experience, never coming back.", "label": "negative", "split": "select", "family": "sel/0"},
        {"id": "e0", "text": "Fantastic dessert and quick checkout.", "label": "positive", "split": "exam", "family": "ex/0"},
        {"id": "t0", "text": "Rude manager and dirty table.", "label": "negative", "split": "train", "family": "tr/0"},
    ]
    cls_file.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    driver = tmp_path / "run_cls.py"
    driver.write_text(f"""
import shutil, subprocess, sys
from pathlib import Path
import numpy as np, torch
from transformers import Qwen3Config, Qwen3ForCausalLM
sys.path.insert(0, {str(ROOT / 'mvp')!r})
from chantless import runtime, vocab
from chantless.env.protocol import load_environment

TMP = Path({str(tmp_path)!r})
hf = TMP / "hf_cls"; hf.mkdir()
tok_src = Path({str(tok_dir)!r})
for fn in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"):
    if (tok_src / fn).exists():
        shutil.copyfile(tok_src / fn, hf / fn)
cfg = Qwen3Config(vocab_size=151936, hidden_size=256, intermediate_size=512, num_hidden_layers=2,
                  num_attention_heads=4, num_key_value_heads=2, head_dim=64, tie_word_embeddings=True, torch_dtype="bfloat16")
cfg.save_pretrained(hf)
Qwen3ForCausalLM(cfg).to(torch.bfloat16).save_pretrained(hf)
gg = TMP / "cls.gguf"
subprocess.run([sys.executable, {str(llama_dir / 'convert_hf_to_gguf.py')!r}, str(hf), "--outtype", "bf16", "--outfile", str(gg)], check=True, capture_output=True)

env = load_environment("chantless.env.clsbench", hf_dir=str(hf), cache_dir=str(TMP / "cache"), jsonl_file={str(cls_file)!r}, fpr_target=0.5)
tc = runtime.Toolchain(Path({str(llama_dir / 'build/bin')!r}), Path({str(tools_bin / 'traj-kl')!r}), [""],
                       logodds=Path({str(tools_bin / 'logodds')!r}), tokdocs=Path({str(tools_bin / 'tokdocs')!r}))
scorer = runtime.BatchScorer(tc, slots_per_gpu=1)
sc_full = env.score(env.tasks(), str(gg), scorer)
assert len(sc_full) == 6

# Trim vocab to 4096 keeping `env.pair` and verify `clsbench.score()` remaps & retokenizes cleanly
keep = vocab.rule_p_keep_set(gg, np.zeros(151936, dtype=np.int64), k=4096, must_keep=env.pair)
gg_trim = TMP / "cls_trim.gguf"
vocab.trim_gguf(gg, gg_trim, keep)
sc_trim = env.score(env.tasks(), str(gg_trim), scorer)
assert len(sc_trim) == 6 and all(np.isfinite(v) for v in sc_trim.values())
print("CLSBENCH_OK")
""")
    proc = subprocess.run([sys.executable, str(driver)], capture_output=True, text=True)
    assert proc.returncode == 0 and "CLSBENCH_OK" in proc.stdout, f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"


# ------------------------------------------------------------------ 4. Gemma 4 asymmetric chat/tool tag protection
def test_vocab_rule_p_protects_gemma4_asymmetric_tags(tmp_path: Path):
    tokens = [
        b"<unk>", b"<bos>", b"<|turn>", b"<turn|>", b"<|tool_call>", b"<tool_call|>",
        b"<|channel>", b"<channel|>", b'<|"|>', b"wordA", b"wordB", b"wordC", b"wordD",
    ]
    ttypes = [3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    kvs = [
        ("general.architecture", _pack_kv_str("gemma4")),
        ("tokenizer.ggml.tokens", _pack_kv_str_arr(tokens)),
        ("tokenizer.ggml.token_type", _pack_kv_i32_arr(ttypes)),
        ("tokenizer.ggml.merges", _pack_kv_str_arr([])),
    ]
    kv_buf = bytearray()
    for k, (vt, raw) in kvs:
        kb = k.encode("utf-8")
        kv_buf += struct.pack("<Q", len(kb)) + kb + struct.pack("<I", vt) + raw
    gg = tmp_path / "g4_tok.gguf"
    gguf_ops._write_gguf_raw(3, 32, len(kvs), bytes(kv_buf), [("output_norm.weight", [32], 0, np.ones(32, dtype=np.float32).tobytes())], gg)
    counts = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 100, 50, 10, 1], dtype=np.int64)
    keep = vocab.rule_p_keep_set(gg, counts, k=10)
    kept = {tokens[int(i)] for i in keep}
    for req in (b"<|turn>", b"<turn|>", b"<|tool_call>", b"<tool_call|>", b"<|channel>", b"<channel|>", b'<|"|>'):
        assert req in kept, f"Gemma 4 control tag {req!r} dropped!"


# ------------------------------------------------------------------ 5. Unified Gemma 4 Student + `learn_gates_seq` + C++ parity
def test_gemma4_unified_student_ple_shared_kv_softcap_and_learn_gates_seq_parity(tmp_path: Path):
    llama_dir, tools_bin = resolve_toolchain()
    llama_bin = llama_dir / "build/bin"
    has_ml_deps = importlib.util.find_spec("torch") is not None and importlib.util.find_spec("transformers") is not None
    if not (has_ml_deps and (llama_bin / "llama-quantize").exists() and (tools_bin / "traj-kl").exists()):
        pytest.skip("local unified afe77e18 toolchain or ML dependencies not present")

    driver = tmp_path / "g4_e2e.py"
    driver.write_text(f"""
import subprocess, sys
from pathlib import Path
import numpy as np, torch, gguf
from transformers import AutoConfig, AutoModelForCausalLM
sys.path.insert(0, {str(ROOT / 'mvp')!r})
from chantless import builder, gguf_ops, kl, lattice, learn, mask as mk, runtime, util

TMP = Path({str(tmp_path)!r})
hf = TMP / "hf_g4"; hf.mkdir()
torch.manual_seed(19)
V = 512
Cfg = type(AutoConfig.for_model("gemma4_text"))
c = Cfg(
    vocab_size=V, vocab_size_per_layer_input=V,
    hidden_size=256, intermediate_size=512, hidden_size_per_layer_input=256,
    num_hidden_layers=4, num_kv_shared_layers=2,
    num_attention_heads=4, num_key_value_heads=1, num_global_key_value_heads=1,
    head_dim=64, global_head_dim=128, sliding_window=512,
    attention_k_eq_v=False,
    layer_types=["sliding_attention", "full_attention", "sliding_attention", "full_attention"],
    use_double_wide_mlp=False, final_logit_softcapping=30.0,
)
m = AutoModelForCausalLM.from_config(c, dtype=torch.bfloat16)
with torch.no_grad():
    for il, l in enumerate(m.model.layers):
        l.layer_scalar.fill_(0.85 + 0.05 * il)
c.save_pretrained(hf)

src_gg = TMP / "g4_src.gguf"
w = gguf.GGUFWriter(str(src_gg), "gemma4")
w.add_context_length(2048); w.add_embedding_length(256); w.add_block_count(4); w.add_feed_forward_length(512)
w.add_head_count(4); w.add_head_count_kv([1, 1, 1, 1])
w.add_key_length(128); w.add_value_length(128); w.add_key_length_swa(64); w.add_value_length_swa(64)
w.add_layer_norm_rms_eps(float(c.rms_norm_eps))
w.add_rope_freq_base(float(c.rope_parameters["full_attention"]["rope_theta"]))
w.add_rope_freq_base_swa(float(c.rope_parameters["sliding_attention"]["rope_theta"]))
w.add_rope_dimension_count(128); w.add_rope_dimension_count_swa(64)
w.add_sliding_window(512); w.add_sliding_window_pattern([True, False, True, False])
w.add_shared_kv_layers(2); w.add_embedding_length_per_layer_input(256)
w.add_final_logit_softcapping(30.0)
w.add_tokenizer_model("llama"); w.add_token_list([f"t{{i}}".encode() for i in range(V)])
w.add_token_scores([0.0]*V); w.add_token_types([1]*V)
sd = dict(m.named_parameters()); bd = dict(m.named_buffers())
def add_t(name, t):
    dt = t.detach().to(torch.float32).numpy() if t.ndim == 1 else (t.detach().to(torch.bfloat16).view(torch.uint16).numpy())
    w.add_tensor(name, dt, raw_dtype=gguf.GGMLQuantizationType.F32 if t.ndim == 1 else gguf.GGMLQuantizationType.BF16)
for gk, hfn in learn.TOPLEVEL_NAMES.items():
    if gk != "output.weight": add_t(gk, sd[hfn])
rev_lay = {{v: k for k, v in learn.LAYER_NAMES.items() if k != "ffn_norm"}}
for il in range(4):
    for hsuf, gsuf in list(rev_lay.items()) + [("pre_feedforward_layernorm.weight", "ffn_norm")]:
        key = f"model.layers.{{il}}.{{hsuf}}"
        if key in sd: add_t(f"blk.{{il}}.{{gsuf}}.weight", sd[key])
        elif key in bd: add_t(f"blk.{{il}}.{{gsuf}}.weight", bd[key].reshape(1))
w.add_tensor("rope_freqs.weight", np.array([1.0]*16 + [1e30]*48, dtype=np.float32), raw_dtype=gguf.GGMLQuantizationType.F32)
w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()

tc = runtime.Toolchain(Path({str(llama_bin)!r}), Path({str(tools_bin / 'traj-kl')!r}), [""])
groups, dead = gguf_ops.adapter_from_gguf(src_gg, hf)
base_mask = mk.new_mask(groups, "Q4_K", {{"repo": "tiny-g4", "revision": "local", "gguf_sha256": util.sha256_file(src_gg)}})
base_mask["dead_tensors"] = dead
base_mask["groups"]["blk.0.ffn"]["tier"] = "Q3_K"   # exercises `dequantize_row_q3_K` + `up_row_zero` on Q3_K
base_mask["groups"]["blk.2.ple"]["tier"] = "drop"
base_mask["groups"]["blk.3.attn"]["tier"] = "drop"
base_mask["widths"] = {{"1": mk.pack_bitmap(list(range(256)), 512)}}

spec_gg = TMP / "g4_spec.gguf"
builder.Builder(tc, str(src_gg), groups, None, TMP / "wk", threads=4).build(base_mask, spec_gg)

# Train `learn_gates_seq` directly on the Gemma 4 specialist
lessons = [{{"prompt_tokens": [5, 10, 15, 20], "corrected_tokens": [30, 31, 32], "mistaken_tokens": [40, 41, 42]}} for _ in range(4)]
replay = [([6, 12, 18, 24, 28, 29], [2, 3, 4, 5]), ([7, 14, 21, 25, 26], [1, 2, 3, 4])]
sw_gg = TMP / "g4_switched.gguf"
sw_mask = TMP / "g4_switched.mask.json"
rec = learn.learn_gates_seq(
    str(spec_gg), str(hf), lessons, replay, str(sw_gg),
    cfg={{"ggml_lib": {str(llama_bin)!r}, "steps": 35, "gate_lr": 0.15, "gate_init": 0.25, "margin": 1.0, "switch_mode": "up_row_zero"}},
    device="cpu", mask_in=base_mask, mask_out=sw_mask,
)
assert rec["off"] > 0 and rec["bytes"] == spec_gg.stat().st_size

# Verify `chantless build` from `(src_gg, sw_mask)` is byte-identical
reb_gg = TMP / "g4_rebuilt.gguf"
subprocess.run([
    sys.executable, "-m", "chantless.cli", "build",
    "--source", str(src_gg), "--mask", str(sw_mask), "--hf-dir", str(hf),
    "--llama-bin", {str(llama_bin)!r}, "--threads", "4", "--shards", "1", "--out", str(reb_gg),
], cwd={str(ROOT / 'mvp')!r}, check=True, capture_output=True)
assert util.sha256_file(reb_gg) == util.sha256_file(sw_gg)

# Verify architectural parity (< 1e-3 nats) on BF16 before Q8_K activation-dot quantization,
# plus quantized Q4_K/Q3_K switched Student vs C++ `traj-kl` parity (< 2e-2 nats) and lesson margin gain
eval_recs = [([5, 10, 15, 20, 30, 31, 32], [4, 5, 6]), ([5, 10, 15, 20, 40, 41, 42], [4, 5, 6]), *replay]
kl.write_trj(TMP / "ev.bin", eval_recs)
kl.traj_kl(tc, "base", str(src_gg), str(TMP / "ev.bin"), str(TMP / "st0"), str(TMP / "b0.json"), str(TMP / "kl0.log"))
cpp0 = kl.traj_kl(tc, "kl", str(src_gg), str(TMP / "ev.bin"), str(TMP / "st0"), str(TMP / "c0.json"), str(TMP / "kl0.log"))
art0 = lattice.Artifact(str(src_gg))
try:
    stu0, _ = learn.build_student(art0, str(hf), [], {str(llama_bin)!r}, device="cpu")
finally:
    art0.close()
st0 = learn.student_traj_nll(stu0, eval_recs, device="cpu")
assert max(abs(s["nll_cand"] - c_["nll_cand"]) for s, c_ in zip(st0, cpp0["trajs"])) < 1e-3

kl.traj_kl(tc, "base", str(spec_gg), str(TMP / "ev.bin"), str(TMP / "st"), str(TMP / "b.json"), str(TMP / "kl.log"))
cpp = kl.traj_kl(tc, "kl", str(reb_gg), str(TMP / "ev.bin"), str(TMP / "st"), str(TMP / "c.json"), str(TMP / "kl.log"))

art = lattice.Artifact(str(reb_gg))
try:
    stu, _ = learn.build_student(art, str(hf), [], {str(llama_bin)!r}, device="cpu")
finally:
    art.close()
st_nll = learn.student_traj_nll(stu, eval_recs, device="cpu")
diffs = [abs(s["nll_cand"] - c_["nll_cand"]) for s, c_ in zip(st_nll, cpp["trajs"])]
print("Gemma4 switched Student vs C++ max |NLL diff|:", max(diffs))
assert max(diffs) < 2e-2
p_tr, n_tr = cpp["trajs"][0], cpp["trajs"][1]
assert ((-p_tr["nll_cand"]) - (-n_tr["nll_cand"])) > ((-p_tr["nll_src"]) - (-n_tr["nll_src"]))
print("GEMMA4_LEARN_SEQ_OK")
""")
    proc = subprocess.run([sys.executable, str(driver)], capture_output=True, text=True)
    assert proc.returncode == 0 and "GEMMA4_LEARN_SEQ_OK" in proc.stdout, f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"


# ------------------------------------------------------------------ 6. CLI `{extract, build, learn}` + generative controls (`empty`, `flipped`, `shuffled`)
def test_cli_extract_build_learn_generative_controls_e2e(tmp_path: Path):
    llama_dir, tools_bin = resolve_toolchain()
    llama_bin = llama_dir / "build/bin"
    tok_dir = resolve_model_dir("qwen3-0.6b")
    has_ml_deps = importlib.util.find_spec("torch") is not None and importlib.util.find_spec("transformers") is not None
    if not (has_ml_deps and (llama_bin / "llama-quantize").exists() and (tools_bin / "traj-kl").exists() and tok_dir.exists()):
        pytest.skip("local unified afe77e18 toolchain, tokenizer model, or ML dependencies not present")

    cli_env = {**os.environ, "PYTHONPATH": str(ROOT / "mvp")}
    for subcmd in ("extract", "build", "learn", "plan"):
        h = subprocess.run([sys.executable, "-m", "chantless.cli", subcmd, "--help"], env=cli_env, capture_output=True, text=True)
        assert h.returncode == 0, (subcmd, h.stderr)

    driver = tmp_path / "cli_learn_ctrl.py"
    driver.write_text(f"""
import json, os, shutil, subprocess, sys
from pathlib import Path
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM
sys.path.insert(0, {str(ROOT / 'mvp')!r})
from chantless import builder, gguf_ops, mask as mk, runtime, util

TMP = Path({str(tmp_path)!r})
hf = TMP / "hf_q3"; hf.mkdir()
tok_src = Path({str(tok_dir)!r})
for fn in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"):
    if (tok_src / fn).exists():
        shutil.copyfile(tok_src / fn, hf / fn)
torch.manual_seed(23)
cfg = Qwen3Config(vocab_size=151936, hidden_size=256, intermediate_size=512, num_hidden_layers=2,
                  num_attention_heads=4, num_key_value_heads=2, head_dim=64, tie_word_embeddings=True, torch_dtype="bfloat16")
cfg.save_pretrained(hf)
Qwen3ForCausalLM(cfg).to(torch.bfloat16).save_pretrained(hf)
src_gg = TMP / "src.gguf"
subprocess.run([sys.executable, {str(llama_dir / 'convert_hf_to_gguf.py')!r}, str(hf), "--outtype", "bf16", "--outfile", str(src_gg)], check=True, capture_output=True)

tc = runtime.Toolchain(Path({str(llama_bin)!r}), Path({str(tools_bin / 'traj-kl')!r}), [""], tokdocs=Path({str(tools_bin / 'tokdocs')!r}))
groups, _ = gguf_ops.adapter_from_gguf(src_gg)
m0 = mk.new_mask(groups, "Q4_K", {{"repo": "tiny-q3", "revision": "local", "gguf_sha256": util.sha256_file(src_gg)}})
m0_path = TMP / "base.mask.json"
m0_path.write_text(json.dumps(m0))
spec_gg = TMP / "spec.gguf"
builder.Builder(tc, str(src_gg), groups, None, TMP / "wk", threads=4).build(m0, spec_gg)

lessons = [
    {{"prompt": "<|im_start|>user\\nCall calc x=1<|im_end|>\\n<|im_start|>assistant\\n", "corrected": '<tool_call>{{"name":"calc","arguments":{{"x":1}}}}</tool_call>', "mistaken": "Sure! calc(1)"}},
    {{"prompt": "<|im_start|>user\\nCall ping host=db<|im_end|>\\n<|im_start|>assistant\\n", "corrected": '<tool_call>{{"name":"ping","arguments":{{"host":"db"}}}}</tool_call>', "mistaken": "Pinging db now"}},
]
replay = [
    {{"prompt": "<|im_start|>user\\n2+2?<|im_end|>\\n<|im_start|>assistant\\n", "completion": "4<|im_end|>"}},
]
(TMP / "lessons.json").write_text(json.dumps(lessons))
(TMP / "replay.json").write_text(json.dumps(replay))
learn_cfg = {{
    "toolchain": {{"llama_bin": {str(llama_bin)!r}, "traj_kl": {str(tools_bin / 'traj-kl')!r}, "tokdocs": {str(tools_bin / 'tokdocs')!r}, "gpus": [""]}},
    "learn": {{
        "mode": "generative",
        "model": str(spec_gg),
        "source_gguf": str(src_gg),
        "mask_in": str(m0_path),
        "config_dir": str(hf),
        "out_dir": str(TMP / "learn_out"),
        "lessons": str(TMP / "lessons.json"),
        "replay": str(TMP / "replay.json"),
        "controls": True,
        "steps": 35,
        "gate_lr": 0.15,
        "gate_init": 0.25,
        "margin": 1.0,
    }}
}}
cfg_p = TMP / "learn_cfg.json"
cfg_p.write_text(json.dumps(learn_cfg))
exp_mask = TMP / "exported.mask.json"
exp_gguf = TMP / "exported.gguf"
cli_env = {{**os.environ, "PYTHONPATH": {str(ROOT / 'mvp')!r}}}
subprocess.run([sys.executable, "-m", "chantless.cli", "learn", "--config", str(cfg_p), "--out-mask", str(exp_mask), "--out", str(exp_gguf)], env=cli_env, check=True, capture_output=True, text=True)

res = json.loads((TMP / "learn_out/result-all.json").read_text())
cpp = res["cpp_traj_kl"]
assert set(cpp) == {{"base", "learned", "empty", "flipped", "shuffled"}}
# `learned` must strictly beat `base`, `empty`, and `flipped` on C++ `traj-kl` margin
assert cpp["learned"]["margin_mean"] > cpp["base"]["margin_mean"]
assert cpp["learned"]["margin_mean"] > cpp["empty"]["margin_mean"]
assert cpp["learned"]["margin_mean"] > cpp["flipped"]["margin_mean"]
assert exp_mask.exists() and exp_gguf.exists() and exp_gguf.stat().st_size == spec_gg.stat().st_size
print("CLI_CONTROLS_OK", json.dumps(cpp))
""")
    proc = subprocess.run([sys.executable, str(driver)], capture_output=True, text=True)
    assert proc.returncode == 0 and "CLI_CONTROLS_OK" in proc.stdout, f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
