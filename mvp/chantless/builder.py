"""Materialize a mask (`v2` or `v3`) into a standalone GGUF and account its actual bytes (spec §9, decision S1).

Order for a full `chantless.mask/v3` mask:
  1. Remove dropped groups + `dead_tensors`, and physically slice per-layer MLP `widths` (and `imatrix` columns).
  2. Quantize every remaining tensor with explicit per-tensor types via `llama-quantize --pure`.
  3. Trim vocabulary tables (`token_embd`, `per_layer_token_embd`, untied `output`) and tokenizer KV if `vocab_keep` is set.
  4. Apply `apprentice` neuron switches (zeroed on grid) and/or sparse grid-level deltas if present.
  5. Audit the built GGUF header against the mask (absent tensors, tensor shapes/widths, vocab size, and quant types).
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path

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
                for suf in ("ffn_up.weight", "ffn_gate.weight"):
                    tname = f"blk.{il}.{suf}"
                    if tname in art.tensors:
                        row_zeros[tname] = dead
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
        vocab_spec = m.get("vocab_keep")
        app = m.get("apprentice") or {}
        has_app = bool(app.get("switches") or app.get("levels"))
        out = Path(out) if out else self.work / f"cand-{mask_key(m)}.gguf"
        out.parent.mkdir(parents=True, exist_ok=True)

        tmp_var = tmp_im = tmp_q = tmp_trim = None
        try:
            if widths:
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
                                              imatrix_src=self.imatrix, imatrix_dst=tmp_im)
                src = str(tmp_var)
                im_use = str(tmp_im) if tmp_im else None
            else:
                src = self._pruned(drop)
                im_use = self.imatrix

            q_dst = (self.work / f"q-{mask_key(m)}.gguf") if (vocab_spec is not None or has_app) else out
            if q_dst != out:
                tmp_q = q_dst

            if all(t == "BF16" for t in types.values()) and all(self.source_types[n] == "BF16" for n in types):
                Path(q_dst).unlink(missing_ok=True)
                if not widths and vocab_spec is None and not has_app:
                    try:
                        os.link(src, q_dst)
                    except OSError:
                        shutil.copyfile(src, q_dst)
                else:
                    shutil.copyfile(src, q_dst)
            else:
                tok = types.pop("token_embd.weight", None)
                shards = self.quantize_shards if self.quantize_shards > 0 else (min(16, max(1, self.threads // 4)) if self.threads >= 12 else 1)
                parallel_quantize(self.tc, src, str(q_dst), "Q8_0", {n: t for n, t in types.items()}, tok,
                                  im_use, self.log, self.threads, shards=shards)
                if tok:
                    types["token_embd.weight"] = tok

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
        present_dropped = sorted(n for n in drop if n in inv)
        if present_dropped:
            raise RuntimeError(f"dropped tensors still present: {present_dropped[:3]}")
        for il, idx in widths.items():
            dn = f"blk.{il}.ffn_down.weight"
            if dn in inv and inv[dn]["shape"][0] != len(idx):
                raise RuntimeError(f"{dn}: built width {inv[dn]['shape'][0]} != mask width {len(idx)}")
        if expected_vocab is not None and "token_embd.weight" in inv:
            if inv["token_embd.weight"]["shape"][1] != expected_vocab:
                raise RuntimeError(f"token_embd vocab {inv['token_embd.weight']['shape'][1]} != {expected_vocab}")
        fb = gguf_ops.file_bytes(out)
        return BuildResult(str(out), fb, sum(t["n_bytes"] for t in inv.values()), self.source_params,
                           fb * 8 / self.source_params, types, sorted(drop), fallbacks)

    def cleanup(self) -> None:
        for p in self.work.glob("pruned-*.gguf"):
            p.unlink(missing_ok=True)


def kv_bytes(info: dict, n_attn_kept: int, context: int, concurrency: int, kv_type: str) -> int:
    per = {"f16": 2.0, "q8_0": 34 / 32, "q4_0": 18 / 32}[kv_type]
    return int(n_attn_kept * 2 * info["n_head_kv"] * info["head_dim"] * per * context * concurrency)
