"""Issue #1: acceptance checks for the qwen35 sub-block removal patch, on one heavy mask.

Usage: python box/qwen35_check.py ROOT OUT   (ROOT has llama.cpp/build/bin, bin/traj-kl, models/qwen3.5-4b[-bf16.gguf])
Writes OUT/report.json plus logs; every check records its own verdict and continues on failure.
"""
import concurrent.futures as cf
import json
import re
import subprocess
import sys
import time
import traceback
from pathlib import Path

from chantless import builder, gguf_ops, kl, mask as mk, util
from chantless.client import ChatClient
from chantless.runtime import LlamaServer, Toolchain

ROOT, OUT = Path(sys.argv[1]), Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)
tc = Toolchain(ROOT / "llama.cpp/build/bin", ROOT / "bin/traj-kl", ["0"])
HF, SRC = ROOT / "models/qwen3.5-4b", str(ROOT / "models/qwen3.5-4b-bf16.gguf")
inv = gguf_ops.inventory(SRC)
G = gguf_ops.groups_from_inventory(inv)
info = gguf_ops.arch_info(SRC)
n = info["n_layer"]
kind = {i: ("full" if G[f"blk.{i}.attn"]["kv"] else "delta") for i in range(n)}
deltas = [i for i in range(n) if kind[i] == "delta"]
consec = next(i for i in deltas if i + 1 in deltas and 0 < i < n // 2)
both = next(i for i in deltas if i > consec + 2 and i < n - 2)
DROP = sorted({"blk.0.attn", f"blk.{n - 1}.attn", f"blk.{consec}.attn", f"blk.{consec + 1}.attn",
               f"blk.{both}.attn", f"blk.{both}.ffn", "blk.5.ffn", f"blk.{n // 2 + 3}.ffn", f"blk.{n - 3}.ffn"})
OUTPROJ = (".attn_output.weight", ".ssm_out.weight", ".ffn_down.weight")
report = {"pins": {"llama_cpp_commit": subprocess.run(["git", "-C", str(ROOT / "llama.cpp"), "rev-parse", "HEAD"],
                                                      capture_output=True, text=True).stdout.strip(),
                   "patch_sha256": util.sha256_file(ROOT / "mvp/patches/llama-cpp-9adc7f4-subblock-drop.patch"),
                   "model_revision": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a", "source_gguf_sha256": util.sha256_file(SRC)},
          "arch": info, "layer_kinds": kind, "mask_M_drop": DROP, "checks": {}}


def save():
    (OUT / "report.json").write_text(json.dumps(report, indent=1, default=str))


def check(name):
    def wrap(f):
        t0 = time.time()
        try:
            r = f()
        except Exception as e:
            r = {"pass": False, "error": repr(e), "trace": traceback.format_exc()[-2000:]}
        r["seconds"] = round(time.time() - t0, 1)
        report["checks"][name] = r
        save()
        print(name, "PASS" if r.get("pass") else "FAIL", flush=True)
        return f
    return wrap


def drop_mask(start):
    m = mk.new_mask(G, start, {})
    for g in DROP:
        m = mk.apply(m, g, "drop")
    return m


@check("1_group_map")
def _():
    blk_fixed = [t for t in G["_fixed"]["fixed"] if t.startswith("blk.")]
    names = sorted({re.sub(r"^blk\.\d+\.", "", t["name"]) for t in inv if t["name"].startswith("blk.")})
    n_mix = sum(1 for g in G if g.endswith(".attn")); n_ffn = sum(1 for g in G if g.endswith(".ffn"))
    outs = [g for g in G if g.endswith((".attn", ".ffn")) and not any(x.endswith(OUTPROJ) for x in G[g]["quantizable"])]
    return {"pass": not blk_fixed and n_mix == n and n_ffn == n and not outs,
            "blk_tensors_outside_groups": blk_fixed, "groups_without_output_projection": outs,
            "n_mixer": n_mix, "n_ffn": n_ffn, "n_full_attention": sum(v == "full" for v in kind.values()),
            "blk_tensor_suffixes": names}


B = builder.Builder(tc, SRC, G, None, OUT / "build")
paths = {}


def rendered_trj():
    R = kl.Renderer(str(HF))
    msgs = [[{"role": "user", "content": f"Compute {a} * {b} and explain briefly."},
             {"role": "assistant", "content": f"{a} * {b} = {a * b}. Multiply the tens and the units, then add the partial products."},
             {"role": "user", "content": "Now add 17 to that and give only the number."},
             {"role": "assistant", "content": str(a * b + 17)}] for a, b in [(37, 48), (125, 9), (64, 64), (19, 73)]]
    kl.write_trj(OUT / "t.bin", [r for m in msgs for r in R.episode(m, None)])
    return str(OUT / "t.bin")


@check("2a_teacher_forced_equivalence")
def _():
    paths["removed"] = B.build(drop_mask("BF16"), OUT / "removed-bf16.gguf").path
    paths["zeroed"] = str(OUT / "zeroed-bf16.gguf")
    gguf_ops.copy_with_zeroed(SRC, paths["zeroed"], {x for g in DROP for x in G[g]["quantizable"] if x.endswith(OUTPROJ)})
    t = rendered_trj()
    lg = str(OUT / "traj-kl.log")
    kl.traj_kl(tc, "base", paths["zeroed"], t, str(OUT / "zstore"), str(OUT / "zbase.json"), lg, gpu="0")
    floor = kl.traj_kl(tc, "kl", paths["zeroed"], t, str(OUT / "zstore"), str(OUT / "zself.json"), lg, gpu="0")
    r = kl.traj_kl(tc, "kl", paths["removed"], t, str(OUT / "zstore"), str(OUT / "eq.json"), lg, gpu="0")
    return {"pass": r["kl"] <= max(1e-5, 3 * floor["kl"]) and r["top1"] == 1, "kl": r["kl"], "top1": r["top1"],
            "floor_kl": floor["kl"], "removed_bytes": gguf_ops.file_bytes(paths["removed"]),
            "source_bytes": gguf_ops.file_bytes(SRC)}


PROMPTS = [f"What is the remainder when {a} is divided by {b}? Answer with a short explanation." for a, b in
           [(1234, 7), (999, 13), (2026, 17), (4567, 23), (8191, 31), (3141, 59), (2718, 28), (1618, 33)]]
GREEDY = {"temperature": 0.0, "top_k": 1, "top_p": 1.0, "min_p": 0.0}


def transcripts(model, tag):
    """8 two-turn episodes, 4 concurrent slots, prompt cache reused across turns."""
    with LlamaServer(tc, model, gpu="0", slots=4, ctx_per_slot=8192, log=OUT / f"srv-{tag}.log") as s:
        c = ChatClient(s.url, GREEDY)

        def ep(i):
            m = [{"role": "user", "content": PROMPTS[i]}]
            a = c.chat(m, seed=i, max_tokens=256)
            m += [{"role": "assistant", "content": a["message"].get("content") or ""},
                  {"role": "user", "content": "Now double that remainder. Number only."}]
            b = c.chat(m, seed=i, max_tokens=64)
            return [a["message"].get("content"), b["message"].get("content")], a["completion_tokens"] + b["completion_tokens"]
        t0 = time.time()
        with cf.ThreadPoolExecutor(4) as ex:
            res = list(ex.map(ep, range(len(PROMPTS))))
        dt = time.time() - t0
    return [r[0] for r in res], sum(r[1] for r in res) / dt


@check("2b_serving_equivalence")
def _():
    z1, tps = transcripts(paths["zeroed"], "zeroed-1")
    z2, _ = transcripts(paths["zeroed"], "zeroed-2")
    rm, tps_rm = transcripts(paths["removed"], "removed")
    d_zz = sum(a != b for a, b in zip(z1, z2)); d_rz = sum(a != b for a, b in zip(rm, z1))
    (OUT / "transcripts.json").write_text(json.dumps({"zeroed_1": z1, "zeroed_2": z2, "removed": rm}, indent=1))
    return {"pass": d_rz <= d_zz, "diverged_removed_vs_zeroed": d_rz, "diverged_zeroed_vs_zeroed": d_zz,
            "episodes": len(PROMPTS), "serving_tokens_per_s_4slots": {"zeroed": round(tps, 1), "removed": round(tps_rm, 1)}}


@check("3_memory")
def _():
    log = OUT / "srv-memory.log"
    with LlamaServer(tc, paths["removed"], gpu="0", slots=1, ctx_per_slot=8192, log=log, extra=["-v"]):
        pass
    txt = log.read_text(errors="ignore")
    kv = [float(x) for x in re.findall(r"llama_kv_cache: size =\s+([\d.]+) MiB", txt)]
    # the summary line prints n_layer; allocated layers are the per-layer "dev =" lines of the first allocation
    first = txt.split("RS buffer size")[0]
    alloc = len(set(re.findall(r"llama_memory_recurrent, layer\s+(\d+): dev =", first)))
    rs = [(float(a), alloc) for a in re.findall(r"llama_memory_recurrent: size =\s+([\d.]+) MiB", txt)]
    full_kept = sum(1 for i in range(n) if kind[i] == "full" and f"blk.{i}.attn" not in DROP)
    delta_kept = sum(1 for i in range(n) if kind[i] == "delta" and f"blk.{i}.attn" not in DROP)
    kv_formula = builder.kv_bytes(info, full_kept, 8192, 1, "f16")
    model_mib = sum(float(x) for x in re.findall(r"model buffer size =\s+([\d.]+) MiB", txt))
    compute_mib = sum(float(x) for x in re.findall(r"compute buffer size =\s+([\d.]+) MiB", txt))
    ok = bool(kv) and abs(kv[-1] * 2 ** 20 - kv_formula) < 1 and bool(rs) and rs[-1][1] == delta_kept
    return {"pass": ok, "kv_measured_mib": kv[-1] if kv else None, "kv_formula_mib": kv_formula / 2 ** 20,
            "full_attention_kept": full_kept, "recurrent_layers_measured": rs[-1][1] if rs else None,
            "recurrent_layers_expected": delta_kept, "recurrent_mib": rs[-1][0] if rs else None,
            "total_mib_at_8k": round(model_mib + (kv[-1] if kv else 0) + (rs[-1][0] if rs else 0) + compute_mib, 1)}


@check("4_quantized_removal")
def _():
    r = B.build(drop_mask("Q4_K"), OUT / "removed-q4k.gguf")
    inv_q = {t["name"]: t for t in gguf_ops.inventory(r.path)}
    _, drop = mk.tensor_plan(drop_mask("Q4_K"), G)
    absent = all(x not in inv_q for x in drop)
    with LlamaServer(tc, r.path, gpu="0", slots=1, ctx_per_slot=4096, log=OUT / "srv-q4k.log") as s:
        a = ChatClient(s.url, GREEDY).chat([{"role": "user", "content": PROMPTS[0]}], seed=0, max_tokens=128)
    served = bool((a["message"].get("content") or "").strip())
    return {"pass": absent and served and not r.fallbacks, "dropped_absent": absent, "fallbacks": r.fallbacks,
            "served_one_episode": served, "file_bytes": r.file_bytes, "bpw": round(r.bpw, 3),
            "sha256": util.sha256_file(r.path)}


report["all_pass"] = all(c.get("pass") for c in report["checks"].values())
save()
print("ALL_PASS" if report["all_pass"] else "SOME_FAIL", flush=True)
