"""Materialize a mask (`v2` or `v3`) into a standalone GGUF and verify its header and byte count.

Order for a full `chantless.mask/v3` mask:
  1. Remove dropped groups + `dead_tensors`, and physically slice per-layer MLP `widths` (and `imatrix` columns).
  2. Quantize every remaining tensor with explicit per-tensor types via `llama-quantize --pure`.
  3. Trim vocabulary tables (`token_embd`, `per_layer_token_embd`, untied `output`) and tokenizer KV if `vocab_keep` is set.
  4. Apply `apprentice` neuron switches (zeroed on grid) and/or sparse grid-level deltas if present.
  5. Audit the built GGUF header against the mask (absent tensors, tensor shapes/widths, vocab size, and quant types).
"""
from __future__ import annotations

import concurrent.futures as cf
import hashlib
import json
import os
import shutil
import struct
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

from . import gguf_ops, lattice, mask as maskmod, vocab
from .runtime import Toolchain, parallel_quantize


@dataclass
class BuildResult:
    path: str
    file_bytes: int
    tensor_bytes: int
    source_params: int
    bpw: float                     # file_bytes * 8 / source parameter count (size-matched cap uses this)
    types: dict[str, str]
    dropped: list[str]
    fallbacks: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("types")
        return d


def mask_key(mask: dict) -> str:
    """Stable 20-char cache key. Preserves exact `v2` hashes when no `v3` structural fields are active."""
    core = {"groups": mask["groups"], "tensors": mask.get("tensors", {}), "weights": mask.get("weights", {})}
    if maskmod.has_v3_cuts(mask):
        core["dead_tensors"] = sorted(mask.get("dead_tensors") or ())
        core["widths"] = mask.get("widths") or {}
        if mask.get("experts"):
            core["experts"] = mask["experts"]
        core["vocab_keep"] = mask.get("vocab_keep")
        app = mask.get("apprentice") or {}
        if app.get("switches") or app.get("levels"):
            core["apprentice"] = {
                "switches": app.get("switches", {}),
                "levels": app.get("levels", {}),
                **({"switch_mode": app["switch_mode"]} if app.get("switch_mode") else {}),
            }
    return hashlib.sha256(json.dumps(core, sort_keys=True).encode()).hexdigest()[:20]


def _apply_apprentice(gguf_in: str | Path, gguf_out: str | Path, app: dict) -> None:
    switches = app.get("switches") or {}
    sparse = app.get("levels") or {}
    switch_mode = app.get("switch_mode", "down_grid")
    if not switches and not sparse:
        if Path(gguf_in).resolve() != Path(gguf_out).resolve():
            shutil.copyfile(gguf_in, gguf_out)
        return
    art = lattice.Artifact(str(gguf_in))
    row_zeros: dict[str, np.ndarray] = {}
    updates: dict[str, np.ndarray] = {}
    try:
        for name, spec in sparse.items():
            idx, vals = maskmod.unpack_sparse_levels(spec)
            lv = art.grid(name).levels.copy()
            lv[idx] = vals
            updates[name] = lv
        for k, spec in switches.items():
            il = int(k)
            dead = maskmod.unpack_bitmap(spec)
            if len(dead) == 0:
                continue
            if switch_mode == "up_row_zero":
                for suf in ("ffn_up.weight", "ffn_gate.weight", "ffn_up_exps.weight", "ffn_gate_exps.weight"):
                    tname = f"blk.{il}.{suf}"
                    if tname in art.tensors:
                        row_zeros[tname] = dead
                gu_name = f"blk.{il}.ffn_gate_up_exps.weight"
                if gu_name in art.tensors:
                    half = int(art.tensors[gu_name]["dims"][1]) // 2
                    row_zeros[gu_name] = np.concatenate([dead, half + dead])
            else:
                name = f"blk.{il}.ffn_down.weight"
                shape = art.shape(name)
                g = art.grid(name)
                lv = updates.get(name, g.levels.copy()).reshape(shape)
                zero = np.clip(np.round(-g.lo / np.maximum(g.step, 1e-30)), 0, g.top).astype(np.uint8).reshape(shape)
                lv[:, dead] = zero[:, dead]
                updates[name] = lv.reshape(-1)
    finally:
        art.close()
    cur_src = str(gguf_in)
    tmp_mid = None
    try:
        if updates:
            dst1 = str(gguf_out) + (".lvl.part" if row_zeros else ".app.part")
            lattice.write_levels(cur_src, dst1, updates)
            if row_zeros:
                tmp_mid = dst1
                cur_src = dst1
            else:
                os.replace(dst1, gguf_out)
        if row_zeros:
            tmp2 = str(gguf_out) + ".row.part"
            lattice.zero_rows(cur_src, tmp2, row_zeros)
            os.replace(tmp2, gguf_out)
        elif not updates and Path(gguf_in).resolve() != Path(gguf_out).resolve():
            shutil.copyfile(gguf_in, gguf_out)
    finally:
        if tmp_mid is not None:
            Path(tmp_mid).unlink(missing_ok=True)


class Builder:
    def __init__(self, tc: Toolchain, weights_gguf: str, groups: dict, imatrix: str | None, work: str | os.PathLike,
                 log: str | None = None, threads: int = 8, quantize_shards: int = 0):
        self.tc, self.weights, self.groups, self.imatrix = tc, str(weights_gguf), groups, imatrix
        self.work = Path(work)
        self.work.mkdir(parents=True, exist_ok=True)
        self.log = log
        self.threads = threads
        self.quantize_shards = quantize_shards
        self.max_pruned = 24
        self._prune_lock = threading.Lock()
        inv = gguf_ops.inventory(self.weights)
        self.source_params = sum(t["n_elements"] for t in inv)
        self.source_types = {t["name"]: t["type"] for t in inv}
        self.source_ndims = {t["name"]: len(t["shape"]) for t in inv}

    def _pruned(self, drop: set[str]) -> str:
        if not drop:
            return self.weights
        with self._prune_lock:
            return self._pruned_locked(drop)

    def _pruned_locked(self, drop: set[str]) -> str:
        h = hashlib.sha256((self.weights + "|" + "|".join(sorted(drop))).encode()).hexdigest()[:16]
        p = self.work / f"pruned-{h}.gguf"
        if not p.exists():
            gguf_ops.copy_without(self.weights, p, drop)
            old = sorted(self.work.glob("pruned-*.gguf"), key=lambda q: q.stat().st_mtime)
            for q in old[: max(0, len(old) - self.max_pruned)]:
                if q != p:
                    q.unlink(missing_ok=True)
        os.utime(p)
        return str(p)

    def build(self, m: dict, out: str | os.PathLike | None = None, keep_pruned: bool = False) -> BuildResult:
        maskmod.validate(m, self.groups)
        types, drop = maskmod.tensor_plan(m, self.groups)
        widths = maskmod.kept_widths(m)
        experts = maskmod.kept_experts(m)
        vocab_spec = m.get("vocab_keep")
        app = m.get("apprentice") or {}
        has_app = bool(app.get("switches") or app.get("levels"))
        out = Path(out) if out else self.work / f"cand-{mask_key(m)}.gguf"
        out.parent.mkdir(parents=True, exist_ok=True)
        fixed_pins = {
            n: self.source_types[n]
            for g in self.groups.values()
            for n in g["fixed"]
            if n not in drop and n in self.source_types and self.source_ndims[n] >= 2
        }

        tmp_var = tmp_im = tmp_q = tmp_trim = None
        try:
            if widths or experts:
                key = mask_key(m)
                tmp_var = self.work / f"var-{key}.gguf"
                if self.imatrix:
                    cand_im = self.work / "im.gguf"
                    with self._prune_lock:
                        if not cand_im.exists():
                            cand_im.touch()
                            tmp_im = cand_im
                        else:
                            tmp_im = self.work / f"im-{key}.gguf"
                else:
                    tmp_im = None
                gguf_ops.write_variable_width(self.weights, tmp_var, widths, drop_tensors=drop,
                                              imatrix_src=self.imatrix, imatrix_dst=tmp_im, kept_experts=experts)
                src = str(tmp_var)
                im_use = str(tmp_im) if tmp_im else None
            else:
                src = self._pruned(drop)
                im_use = self.imatrix

            final_from_q = vocab_spec is None and not has_app
            q_dst = Path(str(out) + ".part") if final_from_q else (self.work / f"q-{mask_key(m)}.gguf")
            tmp_q = q_dst

            if all(t == "BF16" for t in types.values()) and all(self.source_types[n] == "BF16" for n in types):
                Path(q_dst).unlink(missing_ok=True)
                if tmp_var is not None and src == str(tmp_var):
                    try:
                        os.replace(src, q_dst)
                        tmp_var = None
                    except OSError:
                        shutil.copyfile(src, q_dst)
                elif not widths and not experts and final_from_q:
                    try:
                        os.link(src, q_dst)
                    except OSError:
                        shutil.copyfile(src, q_dst)
                else:
                    shutil.copyfile(src, q_dst)
            else:
                overrides = {**fixed_pins, **types}
                tok = overrides.pop("token_embd.weight", None)
                shards = self.quantize_shards if self.quantize_shards > 0 else (min(16, max(1, self.threads // 4)) if self.threads >= 12 else 1)
                parallel_quantize(self.tc, src, str(q_dst), "Q8_0", overrides, tok,
                                  im_use, self.log, self.threads, shards=shards)

            if final_from_q:
                os.replace(q_dst, out)
                cur_file = out
            else:
                cur_file = q_dst
            expected_vocab = None
            if vocab_spec is not None:
                keep_ids = maskmod.unpack_bitmap(vocab_spec)
                expected_vocab = len(keep_ids)
                trim_dst = (self.work / f"trim-{mask_key(m)}.gguf") if has_app else out
                if trim_dst != out:
                    tmp_trim = trim_dst
                vocab.trim_gguf(cur_file, trim_dst, keep_ids)
                cur_file = trim_dst

            if has_app:
                _apply_apprentice(cur_file, out, app)
        finally:
            for p in (tmp_var, tmp_im, tmp_q, tmp_trim):
                if p is not None:
                    Path(p).unlink(missing_ok=True)

        inv = {t["name"]: t for t in gguf_ops.inventory(out)}
        fallbacks = {}
        for n, t in types.items():
            if n not in inv:
                raise RuntimeError(f"built file lacks expected tensor {n}")
            if inv[n]["type"] != t.upper():
                fallbacks[n] = inv[n]["type"]
        for n, t in fixed_pins.items():
            if n in types:
                continue
            if n not in inv:
                raise RuntimeError(f"built file lacks expected fixed tensor {n}")
            if inv[n]["type"] != t.upper():
                raise RuntimeError(f"fixed tensor {n}: built type {inv[n]['type']} != expected {t.upper()}")
        present_dropped = sorted(n for n in drop if n in inv)
        if present_dropped:
            raise RuntimeError(f"dropped tensors still present: {present_dropped[:3]}")
        for il, idx in widths.items():
            for suf in ("ffn_down.weight", "ffn_down_exps.weight"):
                dn = f"blk.{il}.{suf}"
                if dn in inv and inv[dn]["shape"][0] != len(idx):
                    raise RuntimeError(f"{dn}: built width {inv[dn]['shape'][0]} != mask width {len(idx)}")
            gu = f"blk.{il}.ffn_gate_up_exps.weight"
            if gu in inv and inv[gu]["shape"][1] != 2 * len(idx):
                raise RuntimeError(f"{gu}: built gate_up rows {inv[gu]['shape'][1]} != {2 * len(idx)}")
        for il, e_idx in experts.items():
            for suf, axis in (
                ("ffn_down_exps.weight", 2),
                ("ffn_gate_up_exps.weight", 2),
                ("ffn_gate_inp.weight", 1),
                ("exp_probs_b.bias", 0),
            ):
                en = f"blk.{il}.{suf}"
                if en in inv and inv[en]["shape"][axis] != len(e_idx):
                    raise RuntimeError(f"{en}: built expert dim {inv[en]['shape'][axis]} != mask {len(e_idx)}")
        if expected_vocab is not None and "token_embd.weight" in inv:
            if inv["token_embd.weight"]["shape"][1] != expected_vocab:
                raise RuntimeError(f"token_embd vocab {inv['token_embd.weight']['shape'][1]} != {expected_vocab}")
        fb = gguf_ops.file_bytes(out)
        return BuildResult(str(out), fb, sum(t["n_bytes"] for t in inv.values()), self.source_params,
                           fb * 8 / self.source_params, types, sorted(drop), fallbacks)

    def cleanup(self) -> None:
        for p in self.work.glob("pruned-*.gguf"):
            p.unlink(missing_ok=True)


TIER_BPW = {
    "F32": 32.0, "BF16": 16.0, "F16": 16.0,
    "Q8_0": 8.5, "Q6_K": 6.6,
    "Q5_K": 5.7, "Q5_1": 6.0, "Q5_0": 5.5,
    "Q4_K": 4.9, "Q4_1": 5.0, "Q4_0": 4.5, "IQ4_NL": 4.5, "IQ4_XS": 4.25,
    "Q3_K": 3.9, "IQ3_S": 3.5, "IQ3_XXS": 3.1,
    "Q2_K": 3.3, "IQ2_S": 2.5, "IQ2_XS": 2.4, "IQ2_XXS": 2.1,
    "IQ1_M": 1.75, "IQ1_S": 1.6,
}


def estimate_tier_cache_bytes(src_bytes: int, tiers: list[str]) -> int:
    total = 0.0
    for t in set(tiers):
        if t in ("BF16", "F16", maskmod.DROP):
            continue
        total += src_bytes * (TIER_BPW.get(t.upper(), 6.0) / 16.0) * 1.10
    return int(total)


def reachable_tiers(masks: list[dict], groups: dict[str, dict], ladder: list[str]) -> list[str]:
    need: set[str] = set()
    for m in masks:
        types, _ = maskmod.tensor_plan(m, groups)
        for t in types.values():
            if t not in ("BF16", "F16", maskmod.DROP):
                need.add(t)
    order = {t: i for i, t in enumerate(ladder)}
    return sorted(need, key=lambda t: order.get(t, 999))


class SpliceProbeBuilder:
    def __init__(self, inner: Builder, tier_paths: dict[str, Path], cache_dir: Path | None = None,
                 exact_paths: dict[str, tuple[str, Path]] | None = None):
        self.inner = inner
        self.groups = inner.groups
        self.source_params = inner.source_params
        self.weights = inner.weights
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self.tier_paths = dict(tier_paths)
        self.exact_paths = dict(exact_paths or {})
        self.base_h = gguf_ops._Hdr(inner.weights)
        self.tier_h = {k: gguf_ops._Hdr(v) for k, v in tier_paths.items()}
        self.tier_map = {k: {t["name"]: t for t in h.tensors} for k, h in self.tier_h.items()}
        self.hits = 0
        self.fallbacks = 0

    @classmethod
    def from_builder(cls, inner: Builder, base_mask: dict, tiers: list[str], ladder: list[str],
                     cache_dir: str | os.PathLike, workers: int = 4,
                     existing_paths: dict[str, Path] | None = None) -> "SpliceProbeBuilder":
        cdir = Path(cache_dir)
        cdir.mkdir(parents=True, exist_ok=True)
        tier_paths: dict[str, Path] = dict(existing_paths or {})
        req_tiers = [t for t in dict.fromkeys(tiers) if t not in ("BF16", "F16", maskmod.DROP)]
        to_build = [t for t in req_tiers if t not in tier_paths]

        def _uniform_mask(tier_name: str) -> dict:
            return {**base_mask, "groups": {g: {"tier": maskmod.clip(inner.groups[g], tier_name, ladder)} for g in base_mask["groups"]}}

        def _build_one(tier_name: str) -> tuple[str, Path]:
            tp = cdir / f"u-{tier_name}.gguf"
            if not tp.exists():
                inner.build(_uniform_mask(tier_name), tp)
            return tier_name, tp

        if to_build:
            with cf.ThreadPoolExecutor(max_workers=max(1, min(workers, len(to_build)))) as ex:
                for tn, tp in ex.map(_build_one, to_build):
                    tier_paths[tn] = tp
        exact_paths = {mask_key(_uniform_mask(tn)): (tn, tp) for tn, tp in tier_paths.items()}
        return cls(inner, tier_paths, cache_dir=cdir, exact_paths=exact_paths)

    def build(self, m: dict, out: str | os.PathLike | None = None, keep_pruned: bool = False) -> BuildResult:
        if (
            m.get("widths")
            or m.get("experts")
            or m.get("vocab_keep") is not None
            or (m.get("apprentice") or {}).get("switches")
            or (m.get("apprentice") or {}).get("levels")
        ):
            self.fallbacks += 1
            return self.inner.build(m, out, keep_pruned=keep_pruned)
        maskmod.validate(m, self.inner.groups)
        types, drop = maskmod.tensor_plan(m, self.inner.groups)
        drop_set = set(drop)
        mkey = mask_key(m)
        if mkey in self.exact_paths and not drop_set:
            tn, tp = self.exact_paths[mkey]
            self.hits += 1
            out_p = Path(out) if out else self.inner.work / f"cand-{mkey}.gguf"
            out_p.parent.mkdir(parents=True, exist_ok=True)
            if out_p.exists() or out_p.is_symlink():
                out_p.unlink()
            os.symlink(tp.resolve(), out_p)
            sh = self.tier_h[tn]
            fb = gguf_ops.file_bytes(tp)
            data_b = sum(t["n_bytes"] for t in sh.tensors)
            fallbacks = {t["name"]: t["type"] for t in sh.tensors if t["name"] in types and t["type"] != types[t["name"]].upper()}
            return BuildResult(str(out_p), fb, data_b, self.inner.source_params, fb * 8 / self.inner.source_params, types, [], fallbacks)
        for n, t in types.items():
            if t != "BF16" and (t not in self.tier_map or n not in self.tier_map[t]):
                self.fallbacks += 1
                return self.inner.build(m, out, keep_pruned=keep_pruned)
        self.hits += 1
        out_p = Path(out) if out else self.inner.work / f"cand-{mask_key(m)}.gguf"
        out_p.parent.mkdir(parents=True, exist_ok=True)
        bh = self.base_h
        A = bh.alignment
        plan = []
        ti, off = bytearray(), 0
        fallbacks: dict[str, str] = {}
        for bt in bh.tensors:
            nm = bt["name"]
            if nm in drop_set:
                continue
            want_t = types.get(nm, "BF16")
            if want_t != "BF16" and want_t in self.tier_map and nm in self.tier_map[want_t]:
                sh = self.tier_h[want_t]
                st_t = self.tier_map[want_t][nm]
                if st_t["type"] != want_t.upper():
                    fallbacks[nm] = st_t["type"]
            else:
                sh, st_t = bh, bt
            nb = nm.encode()
            ti += struct.pack("<Q", len(nb)) + nb + struct.pack("<I", len(st_t["dims"])) + struct.pack(f"<{len(st_t['dims'])}Q", *st_t["dims"])
            ti += struct.pack("<IQ", st_t["ttype"], off)
            plan.append((sh, st_t, off))
            off = (off + st_t["n_bytes"] + A - 1) // A * A
        tmp = Path(str(out_p) + ".part")
        with open(tmp, "wb") as f:
            f.write(b"GGUF" + struct.pack("<IQQ", bh.version, len(plan), bh.n_kv))
            f.write(bh.m[bh.kv_start : bh.kv_end])
            f.write(ti)
            pos = f.tell()
            f.write(b"\0" * ((pos + A - 1) // A * A - pos))
            base_off = f.tell()
            for sh, st_t, new_off in plan:
                cur = f.tell() - base_off
                if new_off > cur:
                    f.write(b"\0" * (new_off - cur))
                s = sh.data_start + st_t["offset"]
                f.write(sh.m[s : s + st_t["n_bytes"]])
            end_pos = f.tell()
            f.write(b"\0" * ((end_pos + A - 1) // A * A - end_pos))
        os.replace(tmp, out_p)
        fb = gguf_ops.file_bytes(out_p)
        data_b = sum(st_t["n_bytes"] for _, st_t, _ in plan)
        return BuildResult(str(out_p), fb, data_b, self.inner.source_params, fb * 8 / self.inner.source_params, types, sorted(drop_set), fallbacks)

    def release_cache(self) -> None:
        self.base_h.close()
        for h in self.tier_h.values():
            h.close()
        self.tier_h.clear()
        self.tier_map.clear()
        if self.cache_dir is not None:
            shutil.rmtree(self.cache_dir, ignore_errors=True)

    def cleanup(self) -> None:
        self.release_cache()
        self.inner.cleanup()


def _layer_is_swa(info: dict, il: int) -> bool:
    pat = info.get("swa_pattern")
    if isinstance(pat, list):
        return bool(pat[il]) if il < len(pat) else False
    if isinstance(pat, int) and not isinstance(pat, bool) and pat > 0:
        return (il + 1) % pat != 0
    return False


def kv_bytes(
    info: dict,
    n_attn_kept: int,
    context: int,
    concurrency: int,
    kv_type: str,
    kept_attn_layers: Sequence[int] | None = None,
) -> int:
    per = {"f16": 2.0, "q8_0": 34 / 32, "q4_0": 18 / 32}[kv_type]
    n_head_kv = int(info["n_head_kv"])
    head_dim_k = int(info["head_dim"])
    head_dim_v = int(info.get("value_length") or head_dim_k)
    k_swa = int(info.get("key_length_swa") or head_dim_k)
    v_swa = int(info.get("value_length_swa") or head_dim_v)
    n_swa = int(info.get("n_swa") or 0)
    loop_start = int(info.get("loop_start") or 0)
    loop_end = int(info.get("loop_end") or 0)
    loop_count = int(info.get("loop_count") or 1)
    n_layer = int(info.get("n_layer") or 0)

    def _layer_kv_bytes(il: int) -> float:
        mult = loop_count if (loop_count > 1 and loop_start <= il < loop_end) else 1
        is_swa = _layer_is_swa(info, il)
        kd = k_swa if is_swa else head_dim_k
        vd = v_swa if is_swa else head_dim_v
        toks = min(context, n_swa) if (is_swa and n_swa > 0) else context
        return mult * (kd + vd) * n_head_kv * per * toks * concurrency

    if kept_attn_layers is not None:
        return int(sum(_layer_kv_bytes(int(il)) for il in kept_attn_layers))
    if n_layer > 0 and n_attn_kept == n_layer:
        return int(sum(_layer_kv_bytes(il) for il in range(n_layer)))
    has_loop_or_swa = (
        (loop_count > 1 and loop_end > loop_start)
        or (n_swa > 0 and info.get("swa_pattern") is not None)
        or (head_dim_k != head_dim_v)
    )
    if n_layer > 0 and has_loop_or_swa:
        if n_attn_kept <= 0:
            return 0
        full_b = sum(_layer_kv_bytes(il) for il in range(n_layer))
        return int(full_b * n_attn_kept / n_layer)
    return int(n_attn_kept * (head_dim_k + head_dim_v) * n_head_kv * per * context * concurrency)
