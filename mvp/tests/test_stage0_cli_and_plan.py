"""Stage 0 deterministic $0 tests for M2 (Features #5-#12: config, plan, kv_bytes, oneshot MoE, serve/measure_memory --special, mask experts algebra, doctor --llama-commit, and inkling2-mini YAML configs)."""
from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
import threading

import numpy as np
import pytest
from pydantic import ValidationError

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "mvp"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from chantless import (
    builder,
    config as cfgmod,
    doctor,
    gguf_ops,
    mask as mk,
    oneshot,
    pipeline,
    plan,
    prober,
    runtime,
    sanity,
    serve,
)
from test_inkling2_moe import (
    _build_mini_inkling2_fixture,
    _kv_i32_arr,
    _kv_str,
    _kv_u32,
    _resolve_inkling2_toolchain,
    _write_gguf,
)


def _write_minimal_gguf(path: Path, arch: str, extra_kvs: list[tuple[str, tuple[int, bytes]]] | None = None, n_layer: int = 2) -> None:
    n_embd = 64
    kvs = [
        ("general.architecture", _kv_str(arch)),
        ("general.alignment", _kv_u32(32)),
        (f"{arch}.block_count", _kv_u32(n_layer)),
        (f"{arch}.embedding_length", _kv_u32(n_embd)),
        (f"{arch}.attention.head_count", _kv_u32(2)),
        (f"{arch}.attention.head_count_kv", _kv_u32(2)),
        (f"{arch}.attention.key_length", _kv_u32(32)),
    ]
    if extra_kvs:
        kvs.extend(extra_kvs)
    zeros = np.zeros(n_embd * n_embd, dtype=np.uint16).tobytes()
    tensors = [("token_embd.weight", [n_embd, n_embd], 30, zeros)]
    for il in range(n_layer):
        tensors.extend([
            (f"blk.{il}.attn_q.weight", [n_embd, n_embd], 30, zeros),
            (f"blk.{il}.attn_k.weight", [n_embd, n_embd], 30, zeros),
            (f"blk.{il}.attn_v.weight", [n_embd, n_embd], 30, zeros),
            (f"blk.{il}.attn_output.weight", [n_embd, n_embd], 30, zeros),
            (f"blk.{il}.ffn_gate.weight", [n_embd, n_embd], 30, zeros),
            (f"blk.{il}.ffn_up.weight", [n_embd, n_embd], 30, zeros),
            (f"blk.{il}.ffn_down.weight", [n_embd, n_embd], 30, zeros),
        ])
    _write_gguf(path, kvs, tensors)


def _make_fake_q(bin_dir: Path) -> Path:
    bin_dir.mkdir(parents=True, exist_ok=True)
    fake_q = bin_dir / "llama-quantize"
    fake_q.write_text(f"""#!{sys.executable}
import re, struct, sys
sys.path.insert(0, {str(ROOT / "mvp")!r})
from chantless import gguf_ops

NAME_TO_TTYPE = {{name.upper(): (tt, bs, ts) for tt, (name, bs, ts) in gguf_ops.GGML_TYPES.items()}}
TENSOR_TYPE_FALLBACK = {{
    "IQ1_S": "IQ4_NL",
    "IQ1_M": "IQ4_NL",
    "IQ2_XXS": "IQ4_NL",
    "IQ2_XS": "IQ4_NL",
    "IQ2_S": "IQ4_NL",
    "IQ3_XXS": "IQ4_NL",
    "IQ3_S": "IQ4_NL",
    "IQ4_XS": "IQ4_NL",
    "Q2_0": "Q4_0",
    "Q2_K": "Q4_0",
    "Q3_K": "Q4_0",
    "TQ1_0": "Q4_0",
    "TQ2_0": "Q4_0",
    "Q4_K": "Q5_0",
    "Q5_K": "Q5_1",
    "Q6_K": "Q8_0",
}}
args = sys.argv[1:]
ov_rules = []
tok_type = None
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
        i += 2
    else:
        pos.append(a); i += 1

src_p, dst_p, base_ty, _threads = pos[0], pos[1], pos[2].upper(), int(pos[3])
h = gguf_ops._Hdr(src_p)
try:
    kv_bytes = bytes(h.m[h.kv_start : h.kv_end])
    items = []
    for t in h.tensors:
        name, dims, old_tt = t["name"], list(t["dims"]), t["ttype"]
        if t["type"] in ("F32", "F16"):
            tt_id = old_tt
        else:
            ty_name = base_ty
            if name == "token_embd.weight" and tok_type:
                ty_name = tok_type
            for pat, r_ty in ov_rules:
                if pat.search(name):
                    ty_name = r_ty
            tt_id, bs, ts = NAME_TO_TTYPE[ty_name]
            if dims[0] % bs != 0:
                fb_name = TENSOR_TYPE_FALLBACK.get(ty_name)
                if fb_name is None:
                    if bs <= 32:
                        fb_name = ty_name
                    else:
                        raise RuntimeError(f"no tensor type fallback is defined for type {{ty_name}}")
                tt_id, bs, ts = NAME_TO_TTYPE[fb_name]
                if dims[0] % bs != 0:
                    tt_id, bs, ts = NAME_TO_TTYPE["F16"]
            n_el = 1
            for d in dims:
                n_el *= d
            raw = b"\\x00" * ((n_el // bs) * ts)
            items.append((name, dims, tt_id, raw))
            continue
        s0 = h.data_start + t["offset"]
        items.append((name, dims, tt_id, bytes(h.m[s0 : s0 + t["n_bytes"]])))
    gguf_ops._write_gguf_raw(h.version, h.alignment, h.n_kv, kv_bytes, items, dst_p)
finally:
    h.close()
""", encoding="utf-8")
    fake_q.chmod(fake_q.stat().st_mode | stat.S_IEXEC)
    return fake_q


def _make_stub_server(responder):
    records: list[dict] = []
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(length).decode("utf-8"))
            with lock:
                records.append({"path": self.path, "body": body})
            status, payload = responder(self.path, body)
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


# ------------------------------------------------------------------ 1. Configs & schema validation
def test_configs_and_learner_schema_validation():
    cli_env = {**os.environ, "PYTHONPATH": str(ROOT / "mvp")}
    for cfg_name, kind in (
        ("extract-example.yaml", "extract"),
        ("inkling2-mini-bfcl.yaml", "extract"),
        ("inkling2-mini-clsbench.yaml", "extract"),
        ("inkling2-mini-deepmath.yaml", "extract"),
        ("learn-example.yaml", "learn"),
    ):
        cfg_p = ROOT / "mvp" / "configs" / cfg_name
        proc = subprocess.run(
            [sys.executable, "-m", "chantless.cli", "validate", "--config", str(cfg_p), "--kind", kind],
            cwd=str(ROOT),
            env=cli_env,
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 0, f"{cfg_name}: {proc.stderr}"
        parsed = json.loads(proc.stdout)
        assert parsed == {"valid": True, "kind": kind}

    # Positive LearnerConfig & LearnSpecConfig checks
    lc = cfgmod.LearnerConfig(algo="oneshot", vocab_k="auto", experts=True, expert_counts=[32, 48, 64])
    assert lc.vocab_k == "auto"
    assert lc.experts is True
    assert lc.expert_counts == [32, 48, 64]
    assert cfgmod.LearnerConfig(vocab_k=None).vocab_k is None
    assert cfgmod.LearnerConfig(vocab_k=50000).vocab_k == 50000

    ls = cfgmod.LearnSpecConfig(model="/tmp/spec.gguf", out_dir="/tmp/out")
    assert ls.config_dir is None

    # Negative LearnerConfig validation checks (including bool & <= 0 vocab_k / bool expert_counts)
    for bad_vocab_k in ("bad", 0, -1, True, False):
        with pytest.raises(ValidationError):
            cfgmod.LearnerConfig(vocab_k=bad_vocab_k)
    for bad_counts in ([], [0, 32], [-4, 32], [32, 32], [64, 32], [True, 32], [False, 32]):
        with pytest.raises(ValidationError):
            cfgmod.LearnerConfig(expert_counts=bad_counts)


# ------------------------------------------------------------------ 2. plan.SizeModel kvs fix & zero drop_moves
def test_plan_inkling2_loop_zero_drop_moves(tmp_path: Path):
    src, _, _ = _build_mini_inkling2_fixture(tmp_path)
    sm = plan.SizeModel(str(src))
    non_fixed = {g: v for g, v in sm.groups.items() if not g.startswith("_")}
    assert len(non_fixed) == 14
    assert all(not v["droppable"] for v in non_fixed.values())

    cfg_raw = cfgmod.load_config(ROOT / "mvp" / "configs" / "inkling2-mini-bfcl.yaml", "extract").to_raw_dict()
    cfg_raw["run_root"] = str(tmp_path / "runs")
    cfg_raw["source"]["gguf"] = str(src)
    est = plan.estimate(cfg_raw, source=str(src))
    assert est["moves"]["drop_moves"] == 0

    cfg_copy = tmp_path / "bfcl_test.yaml"
    cfg_copy.write_text(
        (ROOT / "mvp" / "configs" / "inkling2-mini-bfcl.yaml")
        .read_text(encoding="utf-8")
        .replace("run_root: .local/runs", f"run_root: {tmp_path / 'runs'}"),
        encoding="utf-8",
    )
    cli_env = {**os.environ, "PYTHONPATH": str(ROOT / "mvp")}
    proc = subprocess.run(
        [sys.executable, "-m", "chantless.cli", "plan", "--config", str(cfg_copy), "--source", str(src)],
        cwd=str(ROOT),
        env=cli_env,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert "14 groups" in proc.stdout
    assert "legal per sweep 14" in proc.stdout

    # Gemma 4 shared_kv_layers: SizeModel keeps partition complete & marks KV donors non-droppable
    gemma_p = tmp_path / "gemma4_shared.gguf"
    _write_minimal_gguf(
        gemma_p,
        "gemma4",
        extra_kvs=[
            ("gemma4.attention.shared_kv_layers", _kv_u32(2)),
            ("gemma4.attention.sliding_window", _kv_i32_arr([512, 0, 512, 0])),
        ],
        n_layer=4,
    )
    sm_g = plan.SizeModel(str(gemma_p))
    assert sm_g.groups["blk.0.attn"]["droppable"] is False
    assert sm_g.groups["blk.1.attn"]["droppable"] is False
    assert sm_g.groups["blk.2.attn"]["kv"] is False
    m_bf16 = mk.new_mask(sm_g.groups, "BF16", {"repo": "g4", "revision": "0", "gguf_sha256": "0"})
    assert sm_g.mask_bytes(m_bf16) == sm_g.file_bytes


# ------------------------------------------------------------------ 3. arch_info & kv_bytes loop/SWA awareness
def test_arch_info_and_kv_bytes_loop_swa(tmp_path: Path):
    src, _, _ = _build_mini_inkling2_fixture(tmp_path)
    info = gguf_ops.arch_info(src)
    assert info["loop_start"] == 2
    assert info["loop_end"] == 4
    assert info["loop_count"] == 2
    assert info["n_swa"] == 16
    assert info["swa_pattern"] == [True, False, True, False, True, False]
    assert info["key_length_swa"] == 64
    assert info["value_length"] == 64
    assert info["value_length_swa"] == 64

    # 6-layer fixture at context=256: 4 SWA slots @ 16 toks + 4 full slots @ 256 toks = 557056 B
    assert builder.kv_bytes(info, 6, 256, 1, "f16") == 557056
    # At context=128: 4*16 + 4*128 = 576 token-slots across 8 expanded KV slots
    assert builder.kv_bytes(info, 6, 128, 1, "f16") == 576 * 2 * 2 * 64 * 2

    # 24-layer Inkling-2-Mini (loop 6..18, count=3 -> 48 slots): 24 SWA @ 2048 + 24 full @ 4096
    mini_info = {
        "n_layer": 24,
        "n_head_kv": 4,
        "head_dim": 128,
        "value_length": 128,
        "key_length_swa": 128,
        "value_length_swa": 128,
        "n_swa": 2048,
        "swa_pattern": [i % 2 == 0 for i in range(24)],
        "loop_start": 6,
        "loop_end": 18,
        "loop_count": 3,
    }
    expected_mini = (24 * 2048 + 24 * 4096) * (128 + 128) * 4 * 2.0 * 8
    assert builder.kv_bytes(mini_info, 24, 4096, 8, "f16") == int(expected_mini)

    # Gemma 4 asymmetric SWA dims + kept_attn_layers=[0, 1]
    g4_info = {
        "n_layer": 4,
        "n_head_kv": 2,
        "head_dim": 128,
        "value_length": 128,
        "key_length_swa": 64,
        "value_length_swa": 64,
        "n_swa": 512,
        "swa_pattern": [True, False, True, False],
    }
    expected_g4 = ((64 + 64) * 512 + (128 + 128) * 2048) * 2 * 2.0 * 1
    assert builder.kv_bytes(g4_info, 2, 2048, 1, "f16", kept_attn_layers=[0, 1]) == int(expected_g4)

    # Legacy plain dict exact bit-for-bit compatibility
    plain = {"n_head_kv": 2, "head_dim": 64}
    assert builder.kv_bytes(plain, 4, 512, 2, "f16") == 4 * 2 * 2 * 64 * 2 * 512 * 2


# ------------------------------------------------------------------ 4. oneshot.build_joint_options non-divisible widths
def test_oneshot_build_joint_options_non_divisible_widths():
    groups = {
        "blk.0.attn": {"droppable": False},
        "blk.0.ffn": {"droppable": False},
    }
    allowed = {
        "blk.0.attn": ["Q8_0", "Q4_K"],
        "blk.0.ffn": ["Q8_0", "IQ4_NL", "Q4_K", "Q2_K"],
    }
    z = np.zeros(4, dtype=np.float64)
    solo = {("blk.0.attn", "Q4_K"): z, ("blk.0.ffn", "IQ4_NL"): z, ("blk.0.ffn", "Q4_K"): z, ("blk.0.ffn", "Q2_K"): z}

    # full=1152 (Inkling-2-Mini): aligned widths [1152, 1024, 768, 512, 256]
    wc_1152 = {(0, w): z for w in (1024, 768, 512, 256)}
    opts_1152 = oneshot.build_joint_options(groups, allowed, solo, wc_1152, {0: 1152}, width_step=256, allow_drop=False)
    assert ("IQ4_NL", 1152) in opts_1152["blk.0.ffn"]
    q4_w_1152 = [w for t, w in opts_1152["blk.0.ffn"] if t == "Q4_K"]
    q2_w_1152 = [w for t, w in opts_1152["blk.0.ffn"] if t == "Q2_K"]
    assert q4_w_1152 == [1024, 768, 512, 256]
    assert q2_w_1152 == [1024, 768, 512, 256]

    # When min measured width is 512 and droppable=False, unmeasured 256 is excluded so cut_shift won't raise
    wc_partial = {(0, 1024): z, (0, 512): z}
    opts_partial = oneshot.build_joint_options(groups, allowed, solo, wc_partial, {0: 1152}, width_step=256, allow_drop=False)
    assert all(w >= 512 for _t, w in opts_partial["blk.0.ffn"])

    # full=320 (fixture): aligned widths [320, 256]
    wc_320 = {(0, 256): z}
    opts_320 = oneshot.build_joint_options(groups, allowed, solo, wc_320, {0: 320}, width_step=256, allow_drop=False)
    assert [w for t, w in opts_320["blk.0.ffn"] if t == "IQ4_NL"] == [320, 256]
    assert [w for t, w in opts_320["blk.0.ffn"] if t == "Q4_K"] == [256]
    assert [w for t, w in opts_320["blk.0.ffn"] if t == "Q2_K"] == [256]

    # Regression: non-256-divisible full (128 or unmeasured 1152) with 256-block-only tiers & allow_drop=False
    # must fall back to full width (Q8_0 fallback in Builder/ByteModel) rather than emitting [] and crashing solve().
    for full_w in (128, 1152):
        bm_inv = [
            {"name": "blk.0.attn_q.weight", "type": "BF16", "shape": [256, 256], "n_elements": 256 * 256, "n_bytes": 2 * 256 * 256},
            {"name": "blk.0.ffn_gate.weight", "type": "BF16", "shape": [256, full_w], "n_elements": 256 * full_w, "n_bytes": 2 * 256 * full_w},
            {"name": "blk.0.ffn_down.weight", "type": "BF16", "shape": [full_w, 256], "n_elements": full_w * 256, "n_bytes": 2 * full_w * 256},
        ]
        bm_groups = {
            "blk.0.attn": {"droppable": False, "quantizable": ["blk.0.attn_q.weight"], "fixed": []},
            "blk.0.ffn": {"droppable": False, "quantizable": ["blk.0.ffn_gate.weight", "blk.0.ffn_down.weight"], "fixed": []},
            "_fixed": {"droppable": False, "quantizable": [], "fixed": []},
        }
        bm_fb = oneshot.ByteModel(None, bm_groups, tensors=bm_inv, kv_end=1024, ff_is_array=False, n_layer=1)
        # Exact llama-quant.cpp tensor_type_fallback parity for both oneshot.tensor_bytes and plan.tensor_bytes
        fallback_cases = (
            ("Q4_K", "Q5_0"),
            ("Q5_K", "Q5_1"),
            ("Q6_K", "Q8_0"),
            ("Q2_K", "Q4_0"),
            ("Q3_K", "Q4_0"),
            ("IQ1_S", "IQ4_NL"),
            ("IQ1_M", "IQ4_NL"),
            ("IQ2_XXS", "IQ4_NL"),
            ("IQ2_XS", "IQ4_NL"),
            ("IQ2_S", "IQ4_NL"),
            ("IQ3_XXS", "IQ4_NL"),
            ("IQ3_S", "IQ4_NL"),
            ("IQ4_XS", "IQ4_NL"),
        )
        for src_ty, fb_ty in fallback_cases:
            assert oneshot.tensor_bytes([full_w, 256], src_ty, padded=False) == oneshot.tensor_bytes([full_w, 256], fb_ty, padded=False)
            assert plan.tensor_bytes({"shape": [full_w, 256], "n_elements": full_w * 256}, src_ty) == oneshot.tensor_bytes([full_w, 256], fb_ty, padded=False)
        # Non-32-divisible rows fall back to F16; Q8_K has no fallback defined in llama-quant.cpp
        assert oneshot.tensor_bytes([300, 256], "Q4_K", padded=False) == oneshot.tensor_bytes([300, 256], "F16", padded=False)
        assert oneshot.tensor_bytes([300, 256], "Q8_0", padded=False) == oneshot.tensor_bytes([300, 256], "F16", padded=False)
        assert plan.tensor_bytes({"shape": [300, 256], "n_elements": 300 * 256}, "Q4_K") == oneshot.tensor_bytes([300, 256], "F16", padded=False)
        with pytest.raises(ValueError, match="no tensor type fallback"):
            oneshot.tensor_bytes([full_w, 256], "Q8_K", padded=False)

        only_256 = {"blk.0.attn": ["Q4_K"], "blk.0.ffn": ["Q4_K", "Q2_K"]}
        opts_fb = oneshot.build_joint_options(bm_groups, only_256, solo, {}, {0: full_w}, width_step=256, allow_drop=False)
        assert opts_fb["blk.0.ffn"] == [("Q4_K", full_w), ("Q2_K", full_w)]
        ref_sol = {g: opts_fb[g][0] for g in opts_fb}
        ref_b = bm_fb.spec_bytes(*oneshot.unpack_solution(ref_sol))
        cost_fb = {
            (g, opt): bm_fb.spec_bytes(*oneshot.unpack_solution({**ref_sol, g: opt})) - ref_b
            for g, o in opts_fb.items() for opt in o
        }
        sur_fb = oneshot.Surrogate([("task", 1.0, lambda x: float(np.mean(x ** 2)), np.zeros(4))], solo, width_cuts={}, n_ff0={0: full_w})
        assert np.allclose(sur_fb.cut_shift(0, full_w), 0.0)
        sol_fb = oneshot.solve(sur_fb, opts_fb, cost_fb, budget=ref_b, header=ref_b, log=lambda *_: None)
        assert sol_fb["blk.0.ffn"] in (("Q4_K", full_w), ("Q2_K", full_w))

    # Non-32-divisible full with 256-block-only tiers & allow_drop=False raises a clear ValueError before solve()
    with pytest.raises(ValueError, match="no valid options"):
        oneshot.build_joint_options(groups, {"blk.0.attn": ["Q4_K"], "blk.0.ffn": ["Q4_K", "Q2_K"]}, solo, {}, {0: 100}, width_step=256, allow_drop=False)


# ------------------------------------------------------------------ 5. pipeline._search_oneshot MoE experts sweep
def test_pipeline_search_oneshot_moe_experts_sweep(tmp_path: Path):
    src, im, dims = _build_mini_inkling2_fixture(tmp_path)
    groups, dead = gguf_ops.adapter_from_gguf(src)
    info = gguf_ops.arch_info(src)
    fake_q = _make_fake_q(tmp_path / "fake_bin")
    tc = runtime.Toolchain(fake_q.parent, Path("/bin/true"), [""])

    run_dir = tmp_path / "run"
    for sub in ("sanity", "proxy", "probe/search", "probe/sanity"):
        (run_dir / sub).mkdir(parents=True, exist_ok=True)
    shutil_im = run_dir / "proxy" / "imatrix.gguf"
    shutil_im.write_bytes(im.read_bytes())

    base_m = mk.new_mask(groups, "BF16", {"repo": "inkling2-test", "revision": "0", "gguf_sha256": "0"}, "preserve")
    bm = oneshot.ByteModel(str(src), groups)

    # Measure uniform IQ4_NL & K=4 IQ4_NL budgets
    b_work = tmp_path / "cal_build"
    cal_b = builder.Builder(tc, str(src), groups, str(shutil_im), b_work, threads=2)
    m_iq4 = {**base_m, "groups": {g: {"tier": mk.clip(groups[g], "IQ4_NL", ["BF16", "Q8_0", "IQ4_NL", "Q4_K"])} for g in base_m["groups"]}}
    r_iq4 = cal_b.build(m_iq4, tmp_path / "cal_iq4.gguf")
    bm.calibrate(m_iq4, r_iq4.file_bytes)
    Path(r_iq4.path).unlink()

    exp_ord = gguf_ops.expert_importance_from_imatrix(src, im)
    kept4 = {il: np.sort(exp_ord[il][:4]) for il in range(dims["n_layer"])}
    target_bytes = bm.spec_bytes(
        {g: e["tier"] for g, e in m_iq4["groups"].items()},
        {il: 256 for il in range(dims["n_layer"])},
        experts={il: 4 for il in range(dims["n_layer"])},
    ) + 4096

    (run_dir / "sanity" / "uniform.json").write_text(
        json.dumps({"IQ4_NL": {"bytes": r_iq4.file_bytes, "bpw": r_iq4.bpw}}),
        encoding="utf-8",
    )

    dummy = object.__new__(pipeline.Run)
    dummy.dir = run_dir
    dummy.scratch = tmp_path / "scratch"
    dummy.scratch.mkdir()
    dummy.cfg = {
        "source": {"hf_dir": None},
        "sanity": {"solo_tiers": ["Q8_0", "IQ4_NL", "Q4_K"]},
        "probe": {"quantize_threads": 2},
    }
    dummy.tc = tc
    dummy.src = str(src)
    dummy.probe_src = str(src)
    dummy.groups = groups
    dummy.dead_tensors = dead
    dummy.info = info
    dummy.ladder = ["BF16", "Q8_0", "IQ4_NL", "Q4_K"]
    dummy.mode = "preserve"
    dummy.classifier = False
    dummy.env = None
    dummy.scorer = None
    dummy.log = lambda msg: None
    dummy.records = []
    dummy.record = lambda k, v: dummy.records.append((k, v))
    dummy._source_pin = lambda: base_m["source"]
    dummy.uniform_mask = lambda b, t: pipeline.Run.uniform_mask(dummy, b, t)

    b_main = builder.Builder(tc, str(src), groups, str(shutil_im), tmp_path / "probe_build", threads=2)
    P = prober.Prober(tc, b_main, run_dir / "probe" / "search", None, workers=1, scratch=tmp_path / "p_scratch")
    P.add_store(prober.Store("task", "/dev/null", str(run_dir / "proxy"), 2, kind="kl"))
    P.add_store(prober.Store("general", "/dev/null", str(run_dir / "proxy"), 2, kind="kl"))

    # Write sanity solo receipts for K=8 (all experts)
    def _write_kl(key_s: str, store_name: str, vals: list[float]) -> None:
        rec = {"trajs": [{"kl": v, "n_tgt": 10} for v in vals]}
        (run_dir / "probe" / "sanity" / f"p-{key_s}-{store_name}.json").write_text(json.dumps(rec), encoding="utf-8")

    for g in base_m["groups"]:
        for t in ("Q8_0", "IQ4_NL", "Q4_K"):
            if mk.tier_allowed(groups[g], t):
                key_gt = builder.mask_key(mk.apply(base_m, g, t))
                kl_v = 0.01 if t == "Q8_0" else (0.04 if t == "IQ4_NL" else 0.08)
                _write_kl(key_gt, "task", [kl_v, kl_v])
                _write_kl(key_gt, "general", [kl_v, kl_v])

    # Stub P.probe / P.probe_many so K=4 probes build through _SliceCacheBuilder (testing full-SHA256 cache & bpw)
    built_bpws: list[float] = []

    def _stub_probe(mask: dict, stores: list[str] | None = None, keep: bool = False, gate=None):
        key = builder.mask_key(mask)
        br = P.builder.build(mask, tmp_path / "p_scratch" / f"p-{key}.gguf")
        built_bpws.append(br.bpw)
        assert br.source_params == b_main.source_params
        Path(br.path).unlink(missing_ok=True)
        has_exp = bool(mask.get("experts"))
        has_w = bool(mask.get("widths"))
        # Make K=4 + w=256 low-KL so solver prefers K=4 within target_bytes
        kl_t = (0.02 if has_exp else 0.0) + (0.01 if has_w else 0.0)
        kl_g = kl_t
        for sname in (stores or ["task", "general"]):
            _write_kl(key, sname, [kl_t, kl_t] if sname == "task" else [kl_g, kl_g])
        return prober.ProbeResult(key, br.file_bytes, br.bpw, kl={"task": kl_t, "general": kl_g}, nll={"task": kl_t, "general": kl_g})

    P.probe = _stub_probe
    P.probe_many = lambda masks, stores=None, gate=None: [_stub_probe(mm, stores=stores, gate=gate) for mm in masks]

    # Verify _SliceCacheBuilder._slice_key distinguishes same-length different expert sets
    k_a = pipeline._SliceCacheBuilder._slice_key({}, {0: np.array([0, 1, 2, 3])}, set(), str(shutil_im))
    k_b = pipeline._SliceCacheBuilder._slice_key({}, {0: np.array([4, 5, 6, 7])}, set(), str(shutil_im))
    assert k_a != k_b and len(k_a) == 64

    lc = {
        "algo": "oneshot",
        "target_bytes": target_bytes,
        "oneshot_tiers": ["Q8_0", "IQ4_NL", "Q4_K"],
        "experts": True,
        "expert_counts": [4, 8],
        "widths": True,
        "width_fractions": [0.75],
        "width_step": 256,
        "allow_drop": False,
        "enforce_general_cap": True,
    }
    cap = sanity.SizeMatchedCap([(16.0, 0.0), (4.0, 2.0)])
    win_m = dummy._search_oneshot(P, base_m, lc, cap=cap)
    mk.validate(win_m, groups)
    assert "experts" in win_m
    kept = mk.kept_experts(win_m)
    assert all(set(kept[il].tolist()) == {4, 5, 6, 7} for il in range(dims["n_layer"]))
    assert all(len(idx) == 256 for idx in mk.kept_widths(win_m).values())
    assert (run_dir / "expert_order.json").exists()
    assert (run_dir / "wanda_order_k4.json").exists()
    assert not (dummy.scratch / "slice_cache").exists()
    assert built_bpws and all(b > 0 for b in built_bpws)
    assert P.builder is b_main

    # Optional real PR #32704 build verification when local toolchain exists
    real_tc = _resolve_inkling2_toolchain()
    if real_tc is not None:
        rb = builder.Builder(real_tc, str(src), groups, str(shutil_im), tmp_path / "real_b", threads=2).build(win_m, tmp_path / "real.gguf")
        assert rb.source_params == b_main.source_params
        assert rb.file_bytes <= target_bytes

    # Regression: uniform.json containing "Q4_K" (as produced by real phase_sanity on inkling2-mini-*.yaml)
    # and widths=False with Q4_K in oneshot_tiers on the n_ff_exp=320 fixture must not crash ByteModel.
    m_q4 = {**base_m, "groups": {g: {"tier": mk.clip(groups[g], "Q4_K", dummy.ladder)} for g in base_m["groups"]}}
    r_q4 = cal_b.build(m_q4, tmp_path / "cal_q4.gguf")
    (run_dir / "sanity" / "uniform.json").write_text(
        json.dumps({"Q4_K": {"bytes": r_q4.file_bytes, "bpw": r_q4.bpw}, "IQ4_NL": {"bytes": r_iq4.file_bytes, "bpw": r_iq4.bpw}}),
        encoding="utf-8",
    )
    Path(r_q4.path).unlink()

    win_q4_cal = dummy._search_oneshot(P, base_m, lc, cap=cap)
    mk.validate(win_q4_cal, groups)
    assert "experts" in win_q4_cal

    nowidth_budget = bm.spec_bytes(
        {g: e["tier"] for g, e in m_iq4["groups"].items()},
        {},
        experts={il: 4 for il in range(dims["n_layer"])},
    ) + 4096
    lc_nowidth = {
        **lc,
        "widths": False,
        "target_bytes": nowidth_budget,
    }
    win_nowidth = dummy._search_oneshot(P, base_m, lc_nowidth, cap=cap)
    mk.validate(win_nowidth, groups)
    assert win_nowidth["widths"] == {}
    assert "experts" in win_nowidth
    assert P.builder is b_main
    assert not (dummy.scratch / "slice_cache").exists()


# ------------------------------------------------------------------ 5b. Real PR #32704 toolchain ByteModel parity & _search_oneshot
def test_real_toolchain_bytemodel_parity_and_oneshot(tmp_path: Path):
    real_tc = _resolve_inkling2_toolchain()
    if real_tc is None:
        pytest.skip("local .local/toolchain-inkling2 PR #32704 toolchain not built")

    src, im, dims = _build_mini_inkling2_fixture(tmp_path)
    groups, dead = gguf_ops.adapter_from_gguf(src)
    info = gguf_ops.arch_info(src)
    exp_ord = gguf_ops.expert_importance_from_imatrix(src, im)
    kept4 = {il: np.sort(exp_ord[il][:4]) for il in range(dims["n_layer"])}
    w_ord = gguf_ops.wanda_order(src, im, kept_experts=kept4)

    ladder = ["BF16", "Q8_0", "IQ4_NL", "Q4_K", "Q3_K", "Q2_K"]
    base_m = mk.new_mask(groups, "BF16", {"repo": "inkling2-test", "revision": "0", "gguf_sha256": "0"}, "preserve")
    B = builder.Builder(real_tc, str(src), groups, str(im), tmp_path / "real_b_work", threads=4, quantize_shards=1)

    # 1. Calibrate ByteModel on "Q4_K" (matching pipeline.py:1244) and assert exact 0-byte diff across all tiers/cuts
    m_q4 = {**base_m, "groups": {g: {"tier": mk.clip(groups[g], "Q4_K", ladder)} for g in base_m["groups"]}}
    r_q4 = B.build(m_q4, tmp_path / "parity_cal_q4k.gguf")
    Path(r_q4.path).unlink(missing_ok=True)
    bm = oneshot.ByteModel(str(src), groups)
    bm.calibrate(m_q4, r_q4.file_bytes)

    real_bytes: dict[tuple[str, str], int] = {}
    for t in ("Q8_0", "IQ4_NL", "Q4_K", "Q3_K", "Q2_K"):
        m_full = {**base_m, "groups": {g: {"tier": mk.clip(groups[g], t, ladder)} for g in base_m["groups"]}}
        rb_full = B.build(m_full, tmp_path / f"parity_{t}_full.gguf")
        Path(rb_full.path).unlink(missing_ok=True)
        assert bm(m_full) == rb_full.file_bytes, (t, "full", bm(m_full), rb_full.file_bytes)
        real_bytes[(t, "full")] = rb_full.file_bytes

        m_k4 = {**m_full, "experts": {str(il): mk.pack_bitmap(kept4[il], dims["n_expert"]) for il in range(dims["n_layer"])}}
        rb_k4 = B.build(m_k4, tmp_path / f"parity_{t}_k4.gguf")
        Path(rb_k4.path).unlink(missing_ok=True)
        assert bm(m_k4) == rb_k4.file_bytes, (t, "k4", bm(m_k4), rb_k4.file_bytes)
        real_bytes[(t, "k4")] = rb_k4.file_bytes

        m_k4_w256 = {**m_k4, "widths": {str(il): mk.pack_bitmap(w_ord[il][:256], dims["n_ff_exp"]) for il in range(dims["n_layer"])}}
        rb_k4_w256 = B.build(m_k4_w256, tmp_path / f"parity_{t}_k4_w256.gguf")
        Path(rb_k4_w256.path).unlink(missing_ok=True)
        assert bm(m_k4_w256) == rb_k4_w256.file_bytes, (t, "k4_w256", bm(m_k4_w256), rb_k4_w256.file_bytes)
        real_bytes[(t, "k4_w256")] = rb_k4_w256.file_bytes

    for extra_t in ("Q6_K", "Q5_K"):
        m_ex = {**base_m, "groups": {g: {"tier": mk.clip(groups[g], extra_t, ["BF16", "Q8_0", extra_t])} for g in base_m["groups"]}}
        rb_ex = B.build(m_ex, tmp_path / f"parity_{extra_t}_full.gguf")
        Path(rb_ex.path).unlink(missing_ok=True)
        assert bm(m_ex) == rb_ex.file_bytes, (extra_t, "full", bm(m_ex), rb_ex.file_bytes)

    # 2. Real PR #32704 _search_oneshot with "Q4_K" in uniform.json for both widths=True and widths=False
    run_dir = tmp_path / "real_run"
    for sub in ("sanity", "proxy", "probe/search", "probe/sanity"):
        (run_dir / sub).mkdir(parents=True, exist_ok=True)
    im_copy = run_dir / "proxy" / "imatrix.gguf"
    im_copy.write_bytes(im.read_bytes())

    (run_dir / "sanity" / "uniform.json").write_text(
        json.dumps({"Q4_K": {"bytes": r_q4.file_bytes, "bpw": r_q4.bpw}}),
        encoding="utf-8",
    )

    dummy = object.__new__(pipeline.Run)
    dummy.dir = run_dir
    dummy.scratch = tmp_path / "real_scratch"
    dummy.scratch.mkdir()
    dummy.cfg = {
        "source": {"hf_dir": None},
        "sanity": {"solo_tiers": ["Q8_0", "IQ4_NL", "Q4_K", "Q3_K", "Q2_K"]},
        "probe": {"quantize_threads": 4},
    }
    dummy.tc = real_tc
    dummy.src = str(src)
    dummy.probe_src = str(src)
    dummy.groups = groups
    dummy.dead_tensors = dead
    dummy.info = info
    dummy.ladder = ladder
    dummy.mode = "preserve"
    dummy.classifier = False
    dummy.env = None
    dummy.scorer = None
    dummy.log = lambda msg: None
    dummy.records = []
    dummy.record = lambda k, v: dummy.records.append((k, v))
    dummy._source_pin = lambda: base_m["source"]
    dummy.uniform_mask = lambda b, t: pipeline.Run.uniform_mask(dummy, b, t)

    b_main = builder.Builder(real_tc, str(src), groups, str(im_copy), tmp_path / "real_probe_build", threads=4, quantize_shards=1)
    P = prober.Prober(real_tc, b_main, run_dir / "probe" / "search", None, workers=1, scratch=tmp_path / "real_p_scratch")
    P.add_store(prober.Store("task", "/dev/null", str(run_dir / "proxy"), 2, kind="kl"))
    P.add_store(prober.Store("general", "/dev/null", str(run_dir / "proxy"), 2, kind="kl"))

    def _write_kl(key_s: str, store_name: str, vals: list[float]) -> None:
        rec = {"trajs": [{"kl": v, "n_tgt": 10} for v in vals]}
        (run_dir / "probe" / "sanity" / f"p-{key_s}-{store_name}.json").write_text(json.dumps(rec), encoding="utf-8")

    tier_kl = {"Q8_0": 0.01, "IQ4_NL": 0.03, "Q4_K": 0.04, "Q3_K": 0.06, "Q2_K": 0.09}
    for g in base_m["groups"]:
        for t, kl_v in tier_kl.items():
            if mk.tier_allowed(groups[g], t):
                key_gt = builder.mask_key(mk.apply(base_m, g, t))
                _write_kl(key_gt, "task", [kl_v, kl_v])
                _write_kl(key_gt, "general", [kl_v, kl_v])

    def _real_build_probe(mask: dict, stores: list[str] | None = None, keep: bool = False, gate=None):
        key = builder.mask_key(mask)
        br = P.builder.build(mask, tmp_path / "real_p_scratch" / f"p-{key}.gguf")
        Path(br.path).unlink(missing_ok=True)
        has_exp = bool(mask.get("experts"))
        has_w = bool(mask.get("widths"))
        kl_v = (0.02 if has_exp else 0.0) + (0.01 if has_w else 0.0)
        for sname in (stores or ["task", "general"]):
            _write_kl(key, sname, [kl_v, kl_v])
        return prober.ProbeResult(key, br.file_bytes, br.bpw, kl={"task": kl_v, "general": kl_v}, nll={"task": kl_v, "general": kl_v})

    P.probe = _real_build_probe
    P.probe_many = lambda masks, stores=None, gate=None: [_real_build_probe(mm, stores=stores, gate=gate) for mm in masks]

    cap = sanity.SizeMatchedCap([(16.0, 0.0), (4.0, 2.0)])
    oneshot_tiers = ["Q8_0", "IQ4_NL", "Q4_K", "Q3_K", "Q2_K"]

    target_w = real_bytes[("IQ4_NL", "k4_w256")] + 4096
    lc_w = {
        "algo": "oneshot",
        "target_bytes": target_w,
        "oneshot_tiers": oneshot_tiers,
        "experts": True,
        "expert_counts": [4, 8],
        "widths": True,
        "width_fractions": [0.75],
        "width_step": 256,
        "allow_drop": False,
        "enforce_general_cap": True,
    }
    win_w = dummy._search_oneshot(P, base_m, lc_w, cap=cap)
    mk.validate(win_w, groups)
    assert "experts" in win_w
    rb_win_w = B.build(win_w, tmp_path / "win_w.gguf")
    Path(rb_win_w.path).unlink(missing_ok=True)
    assert rb_win_w.file_bytes <= target_w

    target_nw = real_bytes[("Q2_K", "k4")] + 4096
    lc_nw = {**lc_w, "widths": False, "target_bytes": target_nw}
    win_nw = dummy._search_oneshot(P, base_m, lc_nw, cap=cap)
    mk.validate(win_nw, groups)
    assert win_nw["widths"] == {}
    assert "experts" in win_nw
    rb_win_nw = B.build(win_nw, tmp_path / "win_nw.gguf")
    Path(rb_win_nw.path).unlink(missing_ok=True)
    assert rb_win_nw.file_bytes <= target_nw
    assert P.builder is b_main
    assert not (dummy.scratch / "slice_cache").exists()


# ------------------------------------------------------------------ 6. serve, cli, & measure_memory --special
def test_serve_cli_and_measure_memory_special(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]):
    ink_gguf, _, _ = _build_mini_inkling2_fixture(tmp_path)
    qwen_plain = tmp_path / "qwen_plain.gguf"
    qwen_ink = tmp_path / "qwen_ink.gguf"
    _write_minimal_gguf(qwen_plain, "qwen35")
    _write_minimal_gguf(qwen_ink, "qwen35", extra_kvs=[("tokenizer.ggml.pre", _kv_str("inkling2"))])

    assert gguf_ops.needs_special_tokens(ink_gguf) is True
    assert gguf_ops.needs_special_tokens(qwen_ink) is True
    assert gguf_ops.needs_special_tokens(qwen_plain) is False

    srv, th, url, reqs = _make_stub_server(
        lambda path, body: (
            200,
            {
                "choices": [{"message": {"content": "ok"}, "text": "ok"}],
                "usage": {"completion_tokens": 5},
                "timings": {"prompt_n": 10, "prompt_ms": 5.0, "predicted_n": 5, "predicted_ms": 10.0},
            },
        )
    )
    recorded_special: list[bool] = []

    class FakeLlamaServer:
        def __init__(self, tc, model, **kw):
            recorded_special.append(bool(kw.get("special", False)))
            self.url = url
            self.proc = SimpleNamespace(pid=-1, poll=lambda: 0)
            log = kw.get("log")
            if log:
                Path(log).parent.mkdir(parents=True, exist_ok=True)
                Path(log).write_text(
                    "model buffer size = 10.0 MiB\nllama_kv_cache: size = 2.0 MiB\ncompute buffer size = 1.0 MiB\n",
                    encoding="utf-8",
                )

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def post(self, path, body):
            import urllib.request
            req = urllib.request.Request(self.url + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req) as r:
                return json.loads(r.read().decode())

    try:
        monkeypatch.setattr(serve, "LlamaServer", FakeLlamaServer)
        monkeypatch.setattr(pipeline, "LlamaServer", FakeLlamaServer)

        # Auto-detection in serve.bench_one & serve.serve_live
        for path, exp_sp in ((ink_gguf, True), (qwen_plain, False)):
            minfo = {"model_path": str(path), "bytes": 100, "sha256": "0", "cache_hit": True, "cache_key": "k", "build_s": 0.0}
            b_res = serve.bench_one(tmp_path, minfo, runs=1)
            assert b_res["config"]["special"] is exp_sp
            assert recorded_special[-1] is exp_sp
            capsys.readouterr()
            serve.serve_live(tmp_path, minfo)
            startup = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
            assert startup["special"] is exp_sp
            assert recorded_special[-1] is exp_sp

        # Explicit overrides
        minfo_q = {"model_path": str(qwen_plain), "bytes": 100, "sha256": "0", "cache_hit": True, "cache_key": "k", "build_s": 0.0}
        assert serve.bench_one(tmp_path, minfo_q, runs=1, special=True)["config"]["special"] is True
        capsys.readouterr()
        serve.serve_live(tmp_path, minfo_q, special=False)
        assert json.loads(capsys.readouterr().out.strip().splitlines()[-1])["special"] is False

        # serve.run_bench(..., compare_path=..., special=None) on mixed pair (inkling2_loop vs qwen35)
        recorded_special.clear()
        minfo_ink = {"model_path": str(ink_gguf), "bytes": 200, "sha256": "0", "cache_hit": True, "cache_key": "k1", "build_s": 0.0}
        cmp_res = serve.run_bench(tmp_path, minfo_ink, compare_path=qwen_plain, runs=1, special=None)
        assert cmp_res["specialist"]["config"]["special"] is True
        assert cmp_res["baseline"]["config"]["special"] is False
        assert recorded_special == [True, False]

        # CLI argument forwarding (--special vs omitted)
        from chantless import cli as climod
        captured: list[dict] = []
        monkeypatch.setattr(serve, "resolve_model", lambda *a, **kw: minfo_q)
        monkeypatch.setattr(serve, "run_bench", lambda *a, **kw: captured.append(kw) or {"ok": True})
        monkeypatch.setattr(serve, "serve_live", lambda *a, **kw: captured.append(kw) or {"ok": True})
        for argv, exp in (
            (["chantless", "bench", "--model", str(qwen_plain), "--llama-bin", str(tmp_path)], None),
            (["chantless", "bench", "--model", str(qwen_plain), "--llama-bin", str(tmp_path), "--special"], True),
            (["chantless", "serve", "--model", str(qwen_plain), "--llama-bin", str(tmp_path)], None),
            (["chantless", "serve", "--model", str(qwen_plain), "--llama-bin", str(tmp_path), "--special"], True),
        ):
            monkeypatch.setattr(sys, "argv", argv)
            climod.main()
            assert captured[-1]["special"] is exp

        # Run.measure_memory: Inkling-2 passes special=True & reasoning_effort=none; Qwen3.5 preserves special=False & enable_thinking=False
        for model_p, arch_name, exp_sp, exp_kw in (
            (ink_gguf, "inkling2_loop", True, {"reasoning_effort": "none"}),
            (qwen_plain, "qwen35", False, {"enable_thinking": False}),
        ):
            reqs.clear()
            r_obj = object.__new__(pipeline.Run)
            r_obj.dir = tmp_path / f"mem_{arch_name}"
            (r_obj.dir / "final").mkdir(parents=True, exist_ok=True)
            r_obj.tc = runtime.Toolchain(tmp_path, Path("/bin/true"), [""])
            r_obj.dep = {"concurrency": 1, "context": 256, "kv_type": "f16"}
            r_obj.info = {"arch": arch_name}
            out_mem = r_obj.measure_memory(str(model_p))
            assert out_mem["decode_tokens"] == 5
            assert recorded_special[-1] is exp_sp
            assert reqs[-1]["body"]["chat_template_kwargs"] == exp_kw

        # measure_memory with ev=None, info=None reaches needs_special_tokens(model) directly
        reqs.clear()
        r_direct = object.__new__(pipeline.Run)
        r_direct.dir = tmp_path / "mem_direct"
        (r_direct.dir / "final").mkdir(parents=True, exist_ok=True)
        r_direct.tc = runtime.Toolchain(tmp_path, Path("/bin/true"), [""])
        r_direct.dep = {"concurrency": 1, "context": 256, "kv_type": "f16"}
        r_direct.ev = None
        r_direct.info = None
        out_direct = r_direct.measure_memory(str(ink_gguf))
        assert out_direct["decode_tokens"] == 5
        assert recorded_special[-1] is True
        assert reqs[-1]["body"]["chat_template_kwargs"] == {"reasoning_effort": "none"}

        # Truncated / missing / corrupt GGUF resilience (no uncaught struct.error) in _resolve_special & measure_memory
        trunc_gguf = tmp_path / "truncated.gguf"
        trunc_gguf.write_bytes(b"GGUF\x03\x00\x00\x00\x01\x00")
        bad_gguf = tmp_path / "bad_magic.gguf"
        bad_gguf.write_bytes(b"NOTGGUF!12345678")
        empty_gguf = tmp_path / "empty.gguf"
        empty_gguf.write_bytes(b"")
        for corrupt_p in (trunc_gguf, bad_gguf, empty_gguf, tmp_path / "missing.gguf"):
            assert serve._resolve_special(corrupt_p, None) is False
        reqs.clear()
        out_trunc = r_direct.measure_memory(str(trunc_gguf))
        assert out_trunc["decode_tokens"] == 5
        assert recorded_special[-1] is False
        assert reqs[-1]["body"]["chat_template_kwargs"] == {"enable_thinking": False}
    finally:
        srv.shutdown()
        srv.server_close()
        th.join(timeout=2)


# ------------------------------------------------------------------ 7. mask.py experts algebra & CLI
def test_mask_experts_algebra_and_cli(tmp_path: Path):
    # Self-contained 6-layer synthetic masks
    groups = {
        **{f"blk.{il}.attn": {"droppable": False, "quantizable": [f"blk.{il}.attn_q.weight"], "fixed": [], "kv": True} for il in range(6)},
        **{f"blk.{il}.ffn": {"droppable": False, "quantizable": [f"blk.{il}.ffn_down_exps.weight"], "fixed": [], "kv": False} for il in range(6)},
        "token_embd": {"droppable": False, "quantizable": ["token_embd.weight"], "fixed": [], "kv": False},
        "output": {"droppable": False, "quantizable": ["output.weight"], "fixed": [], "kv": False},
        "_fixed": {"droppable": False, "quantizable": [], "fixed": []},
    }
    pin = {"repo": "inkling2-test", "revision": "r0", "gguf_sha256": "abc"}
    m_a = mk.new_mask(groups, "IQ4_NL", pin)
    m_a["experts"] = {str(il): mk.pack_bitmap([4, 5, 6, 7], 8) for il in range(6)}
    m_b_same = json.loads(json.dumps(m_a))
    m_b_diff = json.loads(json.dumps(m_a))
    m_b_diff["experts"] = {str(il): mk.pack_bitmap([2, 3, 6, 7], 8) for il in range(6)}
    m_dense = mk.new_mask(groups, "IQ4_NL", pin)

    ins_a = mk.inspect_mask(m_a)
    assert ins_a["experts"] == {
        "pruned_layers": 6,
        "kept_experts": 24,
        "orig_experts": 48,
        "kept_per_layer": 4,
        "orig_per_layer": 8,
        "pct_kept": 50.0,
    }
    assert mk.inspect_mask(m_dense)["experts"]["pruned_layers"] == 0

    d_same = mk.diff_masks(m_a, m_b_same)
    assert d_same["expert_diffs"] == {}
    assert d_same["identical_structure"] is True

    d_diff = mk.diff_masks(m_a, m_b_diff)
    assert len(d_diff["expert_diffs"]) == 6
    assert d_diff["expert_diffs"]["0"] == {"a_kept": 4, "b_kept": 4, "only_a": 2, "only_b": 2}
    assert d_diff["identical_structure"] is False

    pa, pb_same, pb_diff = tmp_path / "a.json", tmp_path / "b_same.json", tmp_path / "b_diff.json"
    pa.write_text(json.dumps(m_a), encoding="utf-8")
    pb_same.write_text(json.dumps(m_b_same), encoding="utf-8")
    pb_diff.write_text(json.dumps(m_b_diff), encoding="utf-8")

    # Apprentice merge rejects mismatched experts, preserves matching experts
    with pytest.raises(mk.MaskError, match="expert_diffs=6"):
        mk.merge_masks(pa, pb_diff, tmp_path / "app_bad.json", mode="apprentice")
    mk.merge_masks(pa, pb_same, tmp_path / "app_ok.json", mode="apprentice")
    assert mk.load(tmp_path / "app_ok.json")["experts"] == m_a["experts"]

    # Specialist merge preserves matching experts, clears mismatched experts
    st_same = mk.merge_masks(pa, pb_same, tmp_path / "spec_same.json", mode="specialist")
    assert st_same["experts_preserved"] is True and st_same["experts_reset_full"] is False
    merged_same = mk.load(tmp_path / "spec_same.json")
    assert merged_same["experts"] == m_a["experts"]
    mk.validate(merged_same, groups, ladder=mk.ALL_TIERS)

    st_diff = mk.merge_masks(pa, pb_diff, tmp_path / "spec_diff.json", mode="specialist")
    assert st_diff["experts_preserved"] is False and st_diff["experts_reset_full"] is True
    merged_diff = mk.load(tmp_path / "spec_diff.json")
    assert "experts" not in merged_diff
    mk.validate(merged_diff, groups, ladder=mk.ALL_TIERS)

    # CLI mask inspect / diff / merge on synthetic & real .local/inkling2-colab masks if present
    cli_env = {**os.environ, "PYTHONPATH": str(ROOT / "mvp")}
    for args in (
        ["mask", "inspect", str(pa)],
        ["mask", "diff", str(pa), str(pb_diff)],
        ["mask", "merge", str(pa), str(pb_diff), "--mode", "specialist", "--out", str(tmp_path / "cli_merged.json")],
    ):
        proc = subprocess.run([sys.executable, "-m", "chantless.cli", *args], env=cli_env, capture_output=True, text=True)
        assert proc.returncode == 0, proc.stderr

    # DROPped .ffn + experts, mismatched experts presence, and same-count different IDs
    groups_drop = {
        **{f"blk.{il}.attn": {"droppable": False, "quantizable": [f"blk.{il}.attn_q.weight"], "fixed": [], "kv": True} for il in range(6)},
        **{f"blk.{il}.ffn": {"droppable": True, "quantizable": [f"blk.{il}.ffn_down_exps.weight"], "fixed": [], "kv": False} for il in range(6)},
        "token_embd": {"droppable": False, "quantizable": ["token_embd.weight"], "fixed": [], "kv": False},
        "output": {"droppable": False, "quantizable": ["output.weight"], "fixed": [], "kv": False},
        "_fixed": {"droppable": False, "quantizable": [], "fixed": []},
    }
    m_drop1 = mk.new_mask(groups_drop, "IQ4_NL", pin)
    m_drop1["groups"]["blk.2.ffn"]["tier"] = mk.DROP
    m_drop1["experts"] = {str(il): mk.pack_bitmap([0, 1, 2, 3], 8) for il in range(6) if il != 2}
    assert mk.inspect_mask(m_drop1)["experts"] == {
        "pruned_layers": 5, "kept_experts": 20, "orig_experts": 40,
        "kept_per_layer": 4, "orig_per_layer": 8, "pct_kept": 50.0,
    }
    m_drop2 = json.loads(json.dumps(m_drop1))
    p_d1, p_d2 = tmp_path / "d1.json", tmp_path / "d2.json"
    p_d1.write_text(json.dumps(m_drop1), encoding="utf-8")
    p_d2.write_text(json.dumps(m_drop2), encoding="utf-8")
    st_both_drop = mk.merge_masks(p_d1, p_d2, tmp_path / "drop_same.json", mode="specialist")
    assert st_both_drop["experts_preserved"] is True and st_both_drop["experts_reset_full"] is False
    m_both_drop = mk.load(tmp_path / "drop_same.json")
    assert set(m_both_drop["experts"]) == {"0", "1", "3", "4", "5"}
    mk.validate(m_both_drop, groups_drop, ladder=mk.ALL_TIERS)

    # One parent drops blk.2.ffn while the other keeps it -> specialist merge resets experts
    m_keep2 = mk.new_mask(groups_drop, "IQ4_NL", pin)
    m_keep2["experts"] = {str(il): mk.pack_bitmap([0, 1, 2, 3], 8) for il in range(6)}
    p_k2 = tmp_path / "k2.json"
    p_k2.write_text(json.dumps(m_keep2), encoding="utf-8")
    st_drop_mismatch = mk.merge_masks(p_d1, p_k2, tmp_path / "drop_mismatch.json", mode="specialist")
    assert st_drop_mismatch["experts_preserved"] is False and st_drop_mismatch["experts_reset_full"] is True
    m_drop_mm = mk.load(tmp_path / "drop_mismatch.json")
    assert "experts" not in m_drop_mm
    mk.validate(m_drop_mm, groups_drop, ladder=mk.ALL_TIERS)

    # Mismatched experts presence (one has experts, one dense) -> specialist merge resets experts
    p_dense = tmp_path / "dense.json"
    p_dense.write_text(json.dumps(m_dense), encoding="utf-8")
    for lhs, rhs, out_name in ((pa, p_dense, "pres_lr.json"), (p_dense, pa, "pres_rl.json")):
        st_pres = mk.merge_masks(lhs, rhs, tmp_path / out_name, mode="specialist")
        assert st_pres["experts_preserved"] is False and st_pres["experts_reset_full"] is True
        m_pres = mk.load(tmp_path / out_name)
        assert "experts" not in m_pres
        mk.validate(m_pres, groups, ladder=mk.ALL_TIERS)

    # Same expert count (4) with shifted IDs ([0..3] vs [1..4])
    m_shift = mk.new_mask(groups, "IQ4_NL", pin)
    m_shift["experts"] = {str(il): mk.pack_bitmap([1, 2, 3, 4], 8) for il in range(6)}
    m_base0 = mk.new_mask(groups, "IQ4_NL", pin)
    m_base0["experts"] = {str(il): mk.pack_bitmap([0, 1, 2, 3], 8) for il in range(6)}
    d_shift = mk.diff_masks(m_base0, m_shift)
    assert d_shift["identical_structure"] is False
    assert d_shift["expert_diffs"]["0"] == {"a_kept": 4, "b_kept": 4, "only_a": 1, "only_b": 1}

    cls_p = ROOT / ".local/inkling2-colab/results/inkling2-mini-clsbench-mask.json"
    bfcl_p = ROOT / ".local/inkling2-colab/results/inkling2-mini-bfcl-mask.json"
    if cls_p.exists() and bfcl_p.exists():
        ins_cls = mk.inspect_mask(cls_p)
        ins_bfcl = mk.inspect_mask(bfcl_p)
        assert ins_cls["experts"] == {
            "pruned_layers": 24,
            "kept_experts": 1152,
            "orig_experts": 3072,
            "kept_per_layer": 48,
            "orig_per_layer": 128,
            "pct_kept": 37.5,
        }
        assert ins_bfcl["experts"] == {
            "pruned_layers": 24,
            "kept_experts": 2304,
            "orig_experts": 3072,
            "kept_per_layer": 96,
            "orig_per_layer": 128,
            "pct_kept": 75.0,
        }
        d_real = mk.diff_masks(cls_p, bfcl_p)
        assert len(d_real["expert_diffs"]) == 24
        assert d_real["identical_structure"] is False

        groups_24 = {
            **{f"blk.{il}.attn": {"droppable": False, "quantizable": [f"blk.{il}.attn_q.weight"], "fixed": [], "kv": True, "imatrix_ok": True} for il in range(24)},
            **{f"blk.{il}.ffn": {"droppable": False, "quantizable": [f"blk.{il}.ffn_down_exps.weight"], "fixed": [], "kv": False, "imatrix_ok": True} for il in range(24)},
            "token_embd": {"droppable": False, "quantizable": ["token_embd.weight"], "fixed": [], "kv": False, "imatrix_ok": False},
            "output": {"droppable": False, "quantizable": ["output.weight"], "fixed": [], "kv": False, "imatrix_ok": True},
            "_fixed": {"droppable": False, "quantizable": [], "fixed": []},
        }
        with pytest.raises(mk.MaskError, match="expert_diffs=24"):
            mk.merge_masks(cls_p, bfcl_p, tmp_path / "real_app_bad.json", mode="apprentice")
        st_real = mk.merge_masks(cls_p, bfcl_p, tmp_path / "real_spec.json", mode="specialist")
        assert st_real["experts_preserved"] is False and st_real["experts_reset_full"] is True
        m_real = mk.load(tmp_path / "real_spec.json")
        assert "experts" not in m_real
        mk.validate(m_real, groups_24, ladder=mk.ALL_TIERS)


# ------------------------------------------------------------------ 8. doctor opt-in --llama-commit
def test_doctor_opt_in_llama_commit(tmp_path: Path):
    def _make_stub_bin(name: str, commit: str) -> tuple[Path, Path]:
        b = tmp_path / name
        b.mkdir()
        (b / "llama-server").write_text(f"#!/bin/sh\necho 'version: 1.0.0-dev (build 1, commit {commit})'\n", encoding="utf-8")
        (b / "llama-quantize").write_text("#!/bin/sh\necho 'usage: --tensor-type-file tensor_types.txt'\n", encoding="utf-8")
        (b / "llama-imatrix").write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        tkl = tmp_path / f"traj-kl-{name}"
        tkl.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        for p in (b / "llama-server", b / "llama-quantize", b / "llama-imatrix", tkl):
            p.chmod(p.stat().st_mode | stat.S_IEXEC)
        return b, tkl

    b_pin, t_pin = _make_stub_bin("bin_pin", "a01dff9")
    b_new, t_new = _make_stub_bin("bin_new", "2a28df0")

    r_pin = doctor.run_doctor(llama_bin=b_pin, traj_kl=t_pin, run_root=tmp_path)
    assert r_pin["ok"] is True
    assert r_pin["pinned_llama_commit"] == "a01dff9"
    assert r_pin["expected_llama_commit"] == "a01dff9"
    assert r_pin["checks"]["llama_bin"]["pinned_commit_match"] is True

    r_fail = doctor.run_doctor(llama_bin=b_new, traj_kl=t_new, run_root=tmp_path)
    assert r_fail["ok"] is False
    assert r_fail["checks"]["llama_bin"]["pinned_commit_match"] is False

    r_opt = doctor.run_doctor(llama_bin=b_new, traj_kl=t_new, run_root=tmp_path, llama_commit="2a28df0")
    assert r_opt["ok"] is True
    assert r_opt["pinned_llama_commit"] == "a01dff9"
    assert r_opt["expected_llama_commit"] == "2a28df0"
    assert r_opt["checks"]["llama_bin"]["pinned_commit_match"] is True
    assert r_opt["checks"]["llama_bin"]["expected_commit"] == "2a28df0"

    # Empty or whitespace-only llama_commit falls back consistently to PINNED_LLAMA_COMMIT
    for blank in ("", "   "):
        r_blk_ok = doctor.run_doctor(llama_bin=b_pin, traj_kl=t_pin, run_root=tmp_path, llama_commit=blank)
        assert r_blk_ok["ok"] is True
        assert r_blk_ok["expected_llama_commit"] == "a01dff9"
        assert r_blk_ok["checks"]["llama_bin"]["expected_commit"] == "a01dff9"
        r_blk_fail = doctor.run_doctor(llama_bin=b_new, traj_kl=t_new, run_root=tmp_path, llama_commit=blank)
        assert r_blk_fail["ok"] is False
        assert r_blk_fail["expected_llama_commit"] == "a01dff9"
        assert r_blk_fail["checks"]["llama_bin"]["expected_commit"] == "a01dff9"
        chk_ws = doctor._check_llama_bin(b_new, llama_commit=blank)
        assert chk_ws["status"] == "fail"
        assert chk_ws["expected_commit"] == "a01dff9"

    cli_env = {**os.environ, "PYTHONPATH": str(ROOT / "mvp")}
    proc = subprocess.run(
        [
            sys.executable, "-m", "chantless.cli", "doctor",
            "--llama-bin", str(b_new), "--traj-kl", str(t_new),
            "--run-root", str(tmp_path), "--llama-commit", "2a28df0",
        ],
        env=cli_env,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert json.loads(proc.stdout)["ok"] is True

    real_tc = _resolve_inkling2_toolchain()
    if real_tc is not None:
        proc_real = subprocess.run(
            [
                sys.executable, "-m", "chantless.cli", "doctor",
                "--llama-bin", str(real_tc.llama_bin), "--traj-kl", str(real_tc.traj_kl),
                "--run-root", str(tmp_path), "--llama-commit", "2a28df0",
            ],
            env=cli_env,
            capture_output=True,
            text=True,
        )
        assert proc_real.returncode == 0, proc_real.stdout + proc_real.stderr
        assert json.loads(proc_real.stdout)["ok"] is True
