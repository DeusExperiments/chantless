"""GGUF inspection, architecture group adaptation, Wanda neuron ordering, and physical tensor/width slicing."""
from __future__ import annotations

import json
import mmap
import os
import re
import struct
from pathlib import Path
from typing import Sequence

import numpy as np

# ggml_type id -> (name, block_size, type_size_bytes); avoids requiring `gguf-py` for fast header operations.
GGML_TYPES: dict[int, tuple[str, int, int]] = {
    0: ("F32", 1, 4), 1: ("F16", 1, 2), 2: ("Q4_0", 32, 18), 3: ("Q4_1", 32, 20),
    6: ("Q5_0", 32, 22), 7: ("Q5_1", 32, 24), 8: ("Q8_0", 32, 34), 9: ("Q8_1", 32, 36),
    10: ("Q2_K", 256, 84), 11: ("Q3_K", 256, 110), 12: ("Q4_K", 256, 144), 13: ("Q5_K", 256, 176),
    14: ("Q6_K", 256, 210), 15: ("Q8_K", 256, 292), 16: ("IQ2_XXS", 256, 66), 17: ("IQ2_XS", 256, 74),
    18: ("IQ3_XXS", 256, 98), 19: ("IQ1_S", 256, 50), 20: ("IQ4_NL", 32, 18), 21: ("IQ3_S", 256, 110),
    22: ("IQ2_S", 256, 82), 23: ("IQ4_XS", 256, 136), 24: ("I8", 1, 1), 25: ("I16", 1, 2),
    26: ("I32", 1, 4), 27: ("I64", 1, 8), 28: ("F64", 1, 8), 29: ("IQ1_M", 256, 56), 30: ("BF16", 1, 2),
}

# The "attn" group is the token-mixer sub-block: full attention (qwen3, qwen35, gemma4) or gated delta net (qwen35).
# When a block has no separate `ffn_norm` (qwen35), `post_attention_norm` is the FFN's pre-norm and belongs to `.ffn`;
# when `ffn_norm` is present (gemma4), `post_attention_norm` is attention's post-norm and belongs to `.attn`.
QUANTIZABLE_ATTN = (
    "attn_q", "attn_k", "attn_v", "attn_output", "attn_qkv", "attn_gate",
    "attn_rel", "attn_sink_proj",
    "ssm_out", "ssm_alpha", "ssm_beta",
)
QUANTIZABLE_FFN = (
    "ffn_gate", "ffn_up", "ffn_down",
    "ffn_gate_up_exps", "ffn_down_exps", "ffn_gate_exps", "ffn_up_exps",
    "ffn_gate_shexp", "ffn_up_shexp", "ffn_down_shexp",
)
QUANTIZABLE_PLE = ("inp_gate", "proj")
ATTN_FIXED = (
    "attn_norm", "attn_q_norm", "attn_k_norm",
    "attn_q_inp_norm", "attn_k_inp_norm", "attn_v_inp_norm",
    "attn_k_conv", "attn_v_conv", "attn_k_conv_proj", "attn_v_conv_proj",
    "attn_post_conv", "attn_q_clip", "attn_rel_bias", "attn_rel_clip", "attn_res_query",
    "ssm_conv1d", "ssm_dt", "ssm_a", "ssm_norm",
)
FFN_FIXED = ("ffn_norm", "post_ffw_norm", "ffn_gate_inp", "exp_probs_b", "ffn_post_conv", "ffn_res_query")
PLE_FIXED = ("post_norm",)
BLK = re.compile(r"^blk\.(\d+)\.([a-z0-9_]+?)(?:\.(weight|bias))?$")

_SCALAR = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
_SCALAR_FMT = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}
_STR, _ARR = 8, 9


def inventory_slow(path: str | os.PathLike) -> list[dict]:
    import gguf
    r = gguf.GGUFReader(str(path))
    return [{"name": t.name, "type": t.tensor_type.name, "shape": [int(x) for x in t.shape], "n_elements": int(t.n_elements),
             "n_bytes": int(t.n_bytes)} for t in r.tensors]


def arch_info(path: str | os.PathLike) -> dict:
    h = _Hdr(path)
    try:
        meta = h.parsed_kvs()
        arch = h.arch or str(meta.get("general.architecture", ""))
        def g(k, default=None):
            v = meta.get(f"{arch}.{k}", default)
            if isinstance(v, list):
                return max(v) if v else default
            return v
        n_embd = int(g("embedding_length"))
        n_head = int(g("attention.head_count"))
        n_head_kv = int(g("attention.head_count_kv", 1))
        head_dim = int(g("attention.key_length", n_embd // max(1, n_head)))
        out = {
            "arch": arch,
            "n_layer": int(g("block_count")),
            "n_embd": n_embd,
            "n_head": n_head,
            "n_head_kv": n_head_kv,
            "head_dim": head_dim,
            "ff_is_array": isinstance(meta.get(f"{arch}.feed_forward_length"), list),
        }
        if f"{arch}.expert_count" in meta:
            out["n_expert"] = int(g("expert_count", 0) or 0)
            out["n_expert_used"] = int(g("expert_used_count", 0) or 0)
            out["n_ff_exp"] = int(g("expert_feed_forward_length", 0) or 0)
        n_layer = out["n_layer"]
        if f"{arch}.attention.value_length" in meta:
            out["value_length"] = int(g("attention.value_length", head_dim))
        if f"{arch}.attention.key_length_swa" in meta:
            out["key_length_swa"] = int(g("attention.key_length_swa", head_dim))
        if f"{arch}.attention.value_length_swa" in meta:
            out["value_length_swa"] = int(g("attention.value_length_swa", out.get("value_length", head_dim)))
        raw_sw = meta.get(f"{arch}.attention.sliding_window")
        if isinstance(raw_sw, list):
            pos_sw = [int(w) for w in raw_sw if int(w) > 0]
            if pos_sw:
                out["n_swa"] = max(pos_sw)
            if f"{arch}.attention.sliding_window_pattern" not in meta and raw_sw:
                out["swa_pattern"] = [int(w) > 0 for w in raw_sw]
        elif raw_sw is not None and int(raw_sw) > 0:
            out["n_swa"] = int(raw_sw)
        raw_pat = meta.get(f"{arch}.attention.sliding_window_pattern")
        if isinstance(raw_pat, list):
            out["swa_pattern"] = [bool(x) for x in raw_pat]
        elif raw_pat is not None and not isinstance(raw_pat, bool) and int(raw_pat) > 0:
            pat_n = int(raw_pat)
            out["swa_pattern"] = [((il + 1) % pat_n != 0) for il in range(n_layer)]
        if f"{arch}.loop.start" in meta:
            out["loop_start"] = int(g("loop.start", 0) or 0)
        if f"{arch}.loop.end" in meta:
            out["loop_end"] = int(g("loop.end", 0) or 0)
        if f"{arch}.loop.count" in meta:
            out["loop_count"] = int(g("loop.count", 1) or 1)
        return out
    finally:
        h.close()


def needs_special_tokens(path: str | os.PathLike) -> bool:
    """Return True when the GGUF architecture/tokenizer requires `llama-server --special` (e.g. Inkling-2)."""
    h = _Hdr(path)
    try:
        kvs = h.parsed_kvs(skip_large_arrays=True)
        arch = h.arch or str(kvs.get("general.architecture", ""))
        return arch in ("inkling2", "inkling2_loop") or kvs.get("tokenizer.ggml.pre") == "inkling2"
    finally:
        h.close()


def _shared_kv_spec(n_layer: int, kvs: dict | None, hf_cfg: dict | None) -> tuple[int, set[int]]:
    """Return `(first_shared_layer, undroppable_kv_source_layers)`.

    When no layers share KV (`first_shared == n_layer`), returns `(n_layer, set())`.
    """
    kvs = kvs or {}
    cfg = (hf_cfg.get("text_config", hf_cfg) if isinstance(hf_cfg, dict) else {}) or {}
    arch = kvs.get("general.architecture", "")
    n_shared = int(cfg.get("num_kv_shared_layers", 0) or 0)
    if not n_shared and arch:
        for key in (f"{arch}.attention.shared_kv_layers", f"{arch}.attention.num_kv_shared_layers"):
            if key in kvs and int(kvs[key]) > 0:
                n_shared = int(kvs[key])
                break
        if not n_shared and f"{arch}.attention.layer_count_kv" in kvs:
            lkv = int(kvs[f"{arch}.attention.layer_count_kv"])
            if 0 < lkv < n_layer:
                n_shared = n_layer - lkv
    if n_shared <= 0 or n_shared >= n_layer:
        return n_layer, set()

    first_shared = n_layer - n_shared
    layer_types = cfg.get("layer_types")
    if not layer_types and arch and isinstance(kvs.get(f"{arch}.attention.sliding_window"), list):
        sw = kvs[f"{arch}.attention.sliding_window"]
        if len(sw) >= first_shared:
            layer_types = ["sliding_attention" if int(w) > 0 else "full_attention" for w in sw]
    if layer_types and len(layer_types) >= first_shared:
        prev = list(layer_types[:first_shared])
        sources = {first_shared - 1 - prev[::-1].index(kind) for kind in set(prev)}
        return first_shared, sources
    return first_shared, {i for i in (first_shared - 2, first_shared - 1) if i >= 0}


def groups_from_inventory(
    inv: list[dict],
    kvs: dict | None = None,
    hf_config: dict | None = None,
    strip_dead_shared_kv: bool = True,
) -> dict[str, dict]:
    """Group map generated from the header: every non-dead tensor lands in one group or `_fixed`."""
    groups, _ = _build_groups_and_dead(inv, kvs=kvs, hf_config=hf_config, strip_dead=False)
    return groups


def _build_groups_and_dead(
    inv: list[dict],
    kvs: dict | None = None,
    hf_config: dict | None = None,
    strip_dead: bool = True,
) -> tuple[dict[str, dict], list[str]]:
    groups: dict[str, dict] = {}
    fixed: list[str] = []
    mtp_blocks = {p[1] for t in inv if len(p := t["name"].split(".")) > 3 and p[0] == "blk" and p[2] == "nextn"}
    ffn_norms = {int(m.group(1)) for t in inv if (m := BLK.match(t["name"])) and m.group(2) == "ffn_norm"}
    block_ids = {int(m.group(1)) for t in inv if (m := BLK.match(t["name"])) and t["name"].split(".")[1] not in mtp_blocks}
    n_layer = (max(block_ids) + 1) if block_ids else 0
    first_shared, kv_sources = _shared_kv_spec(n_layer, kvs, hf_config)
    arch = str((kvs or {}).get("general.architecture", "") or "")
    has_loop = (
        arch in ("inkling2", "inkling2_loop")
        or int((kvs or {}).get(f"{arch}.loop.count", 0) or 0) > 1
        or int((kvs or {}).get(f"{arch}.num_loops", 0) or 0) > 1
        or int((kvs or {}).get(f"{arch}.attn_res.block_size", 0) or 0) > 0
    )

    names_present = {t["name"] for t in inv}
    dead: list[str] = []
    if strip_dead and first_shared < n_layer:
        for il in range(first_shared, n_layer):
            for suf in ("attn_k.weight", "attn_v.weight", "attn_k_norm.weight"):
                nm = f"blk.{il}.{suf}"
                if nm in names_present:
                    dead.append(nm)
    dead_set = set(dead)

    for t in inv:
        name = t["name"]
        if name in dead_set:
            continue
        m = BLK.match(name)
        if name.startswith("blk.") and name.split(".")[1] in mtp_blocks:
            fixed.append(name)
        elif name == "token_embd.weight":
            groups.setdefault("token_embd", {"quantizable": [], "fixed": [], "droppable": False, "kv": False, "imatrix_ok": False})["quantizable"].append(name)
        elif name == "per_layer_token_embd.weight":
            groups.setdefault("per_layer_token_embd", {"quantizable": [], "fixed": [], "droppable": False, "kv": False, "imatrix_ok": False})["quantizable"].append(name)
        elif name == "per_layer_model_proj.weight":
            groups.setdefault("per_layer_model_proj", {"quantizable": [], "fixed": [], "droppable": False, "kv": False, "imatrix_ok": True})["quantizable"].append(name)
        elif name == "output.weight":
            groups.setdefault("output", {"quantizable": [], "fixed": [], "droppable": False})["quantizable"].append(name)
        elif m:
            i, kind = int(m.group(1)), m.group(2)
            if kind == "post_attention_norm":
                target = "attn" if i in ffn_norms else "ffn"
                drop_ok = False if has_loop else ((i not in kv_sources) if target == "attn" else True)
                g = groups.setdefault(f"blk.{i}.{target}", {"quantizable": [], "fixed": [], "droppable": drop_ok})
                g["fixed"].append(name)
            elif kind in QUANTIZABLE_ATTN or kind in ATTN_FIXED:
                g = groups.setdefault(f"blk.{i}.attn", {"quantizable": [], "fixed": [], "droppable": (False if has_loop else (i not in kv_sources))})
                (g["quantizable"] if kind in QUANTIZABLE_ATTN else g["fixed"]).append(name)
            elif kind in QUANTIZABLE_FFN or kind in FFN_FIXED:
                g = groups.setdefault(f"blk.{i}.ffn", {"quantizable": [], "fixed": [], "droppable": (not has_loop), "kv": False, "imatrix_ok": True})
                (g["quantizable"] if kind in QUANTIZABLE_FFN else g["fixed"]).append(name)
            elif kind in QUANTIZABLE_PLE or kind in PLE_FIXED:
                g = groups.setdefault(f"blk.{i}.ple", {"quantizable": [], "fixed": [], "droppable": (not has_loop), "kv": False, "imatrix_ok": True})
                (g["quantizable"] if kind in QUANTIZABLE_PLE else g["fixed"]).append(name)
            else:
                fixed.append(name)
        else:
            fixed.append(name)

    for g, e in groups.items():
        if g.endswith(".attn"):
            il = int(g.split(".")[1])
            has_full_out = any(n.endswith(".attn_output.weight") for n in e["quantizable"])
            e["kv"] = bool(has_full_out and il < first_shared)
            e["droppable"] = False if has_loop else (il not in kv_sources)
            e.setdefault("imatrix_ok", True)
        elif has_loop and not g.startswith("_"):
            e["droppable"] = False

    groups["_fixed"] = {"quantizable": [], "fixed": fixed, "droppable": False}
    seen = [n for g in groups.values() for n in g["quantizable"] + g["fixed"]] + dead
    assert sorted(seen) == sorted(t["name"] for t in inv), "group map does not partition the tensors"
    return groups, dead


def adapter_from_gguf(
    path: str | os.PathLike,
    config_dir: str | os.PathLike | None = None,
    strip_dead_shared_kv: bool = True,
) -> tuple[dict[str, dict], list[str]]:
    """Derive `(groups, dead_tensors)` automatically from a GGUF header and optional HF `config.json`."""
    hf_cfg = None
    if config_dir:
        cp = Path(config_dir)
        cp = cp / "config.json" if cp.is_dir() else cp
        if cp.exists():
            hf_cfg = json.loads(cp.read_text())
    h = _Hdr(path)
    try:
        inv = [{"name": t["name"], "type": t["type"], "shape": t["dims"], "n_elements": t["n_elements"], "n_bytes": t["n_bytes"]} for t in h.tensors]
        kvs = h.parsed_kvs(skip_large_arrays=True)
    finally:
        h.close()
    return _build_groups_and_dead(inv, kvs=kvs, hf_config=hf_cfg, strip_dead=strip_dead_shared_kv)


def copy_without_slow(src: str | os.PathLike, dst: str | os.PathLike, drop: set[str]) -> dict:
    """Write `dst` = `src` minus the named tensors (metadata copied exactly). Returns a removal receipt."""
    import gguf
    reader = gguf.GGUFReader(str(src))
    names = {t.name for t in reader.tensors}
    missing = drop - names
    if missing:
        raise ValueError(f"cannot drop tensors absent from source: {sorted(missing)[:5]}")
    arch = reader.get_field("general.architecture").contents()
    tmp = Path(str(dst) + ".part")
    writer = gguf.GGUFWriter(str(tmp), arch=arch, endianess=reader.endianess)
    align = reader.get_field("general.alignment")
    if align:
        writer.data_alignment = int(align.contents())
    for field in reader.fields.values():
        if field.name == gguf.Keys.General.ARCHITECTURE or field.name.startswith("GGUF."):
            continue
        vt = field.types[0]
        st = field.types[-1] if vt == gguf.GGUFValueType.ARRAY else None
        writer.add_key_value(field.name, field.contents(), vt, sub_type=st)
    kept = [t for t in reader.tensors if t.name not in drop]
    for t in kept:
        writer.add_tensor_info(t.name, t.data.shape, t.data.dtype, t.data.nbytes, t.tensor_type)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()
    for t in kept:
        writer.write_tensor_data(t.data, tensor_endianess=reader.endianess)
    writer.close()
    os.replace(tmp, dst)
    return {"dropped": sorted(drop), "n_dropped": len(drop), "dropped_bytes": int(sum(t.n_bytes for t in reader.tensors if t.name in drop))}


def file_bytes(path: str | os.PathLike) -> int:
    return os.path.getsize(path)


def copy_with_zeroed(src: str | os.PathLike, dst: str | os.PathLike, zero: set[str]) -> None:
    """Copy `src` to `dst` with the named tensors zeroed (preserves file size)."""
    import gguf
    shutil_copy = __import__("shutil").copyfile
    shutil_copy(src, dst)
    r = gguf.GGUFReader(str(dst), "r+")
    for t in r.tensors:
        if t.name in zero:
            t.data[...] = np.zeros_like(t.data)
    del r


# ---------------------------------------------------------------- fast raw GGUF header handling
class _Hdr:
    def __init__(self, path):
        self.f = open(path, "rb")
        self.m = mmap.mmap(self.f.fileno(), 0, access=mmap.ACCESS_READ)
        m = self.m
        if m[:4] != b"GGUF":
            raise ValueError("not a GGUF file")
        self.version, self.n_tensors, self.n_kv = struct.unpack_from("<IQQ", m, 4)
        p = 24
        self.kv_start = p
        self.alignment = 32
        self.arch = None
        for _ in range(self.n_kv):
            (kl,) = struct.unpack_from("<Q", m, p); p += 8
            key = m[p : p + kl].decode(); p += kl
            (vt,) = struct.unpack_from("<I", m, p); p += 4
            if key == "general.alignment" and vt == 4:
                (self.alignment,) = struct.unpack_from("<I", m, p)
            if key == "general.architecture" and vt == _STR:
                (sl,) = struct.unpack_from("<Q", m, p)
                self.arch = m[p + 8 : p + 8 + sl].decode()
            p = self._skip(p, vt)
        self.kv_end = p
        self.tensors = []
        for _ in range(self.n_tensors):
            (nl,) = struct.unpack_from("<Q", m, p); p += 8
            name = m[p : p + nl].decode(); p += nl
            (nd,) = struct.unpack_from("<I", m, p); p += 4
            dims = list(struct.unpack_from(f"<{nd}Q", m, p)); p += 8 * nd
            ttype, off = struct.unpack_from("<IQ", m, p); p += 12
            self.tensors.append({"name": name, "dims": dims, "ttype": ttype, "offset": off})
        self.ti_end = p
        self.data_start = (p + self.alignment - 1) // self.alignment * self.alignment
        for t in self.tensors:
            if t["ttype"] in GGML_TYPES:
                tname, bs, ts = GGML_TYPES[t["ttype"]]
            else:
                import gguf
                qt = gguf.GGMLQuantizationType(t["ttype"])
                tname = qt.name
                bs, ts = gguf.GGML_QUANT_SIZES[qt]
            n = 1
            for d in t["dims"]:
                n *= d
            t["type"], t["n_elements"], t["n_bytes"] = tname, n, n // bs * ts

    def _skip(self, p, vt):
        m = self.m
        if vt in _SCALAR:
            return p + _SCALAR[vt]
        if vt == _STR:
            (sl,) = struct.unpack_from("<Q", m, p)
            return p + 8 + sl
        if vt == _ARR:
            et, cnt = struct.unpack_from("<IQ", m, p); p += 12
            if et in _SCALAR:
                return p + _SCALAR[et] * cnt
            for _ in range(cnt):
                p = self._skip(p, et)
            return p
        raise ValueError(f"unknown GGUF value type {vt}")

    def read_kvs(self) -> list[tuple[str, int, bytes]]:
        """Return ordered `[(key, value_type, raw_value_bytes)]` from the GGUF KV section."""
        m, p = self.m, self.kv_start
        out = []
        for _ in range(self.n_kv):
            (kl,) = struct.unpack_from("<Q", m, p); p += 8
            key = m[p : p + kl].decode(); p += kl
            (vt,) = struct.unpack_from("<I", m, p); p += 4
            s = p
            p = self._skip(p, vt)
            out.append((key, vt, bytes(m[s:p])))
        return out

    def parsed_kvs(self, skip_large_arrays: bool = True, max_array: int = 4096) -> dict:
        """Decode scalar, string, and bounded numeric-array KV metadata."""
        out = {}
        for k, vt, raw in self.read_kvs():
            if vt in _SCALAR_FMT:
                out[k] = struct.unpack_from(_SCALAR_FMT[vt], raw, 0)[0]
            elif vt == _STR:
                (sl,) = struct.unpack_from("<Q", raw, 0)
                out[k] = raw[8 : 8 + sl].decode("utf-8", errors="replace")
            elif vt == _ARR:
                et, cnt = struct.unpack_from("<IQ", raw, 0)
                if skip_large_arrays and cnt > max_array:
                    continue
                if et in _SCALAR_FMT:
                    fmt = "<" + str(cnt) + _SCALAR_FMT[et][1:]
                    out[k] = list(struct.unpack_from(fmt, raw, 12))
        return out

    def close(self):
        self.m.close(); self.f.close()


def inventory_fast(path) -> list[dict]:
    h = _Hdr(path)
    try:
        return [{"name": t["name"], "type": t["type"], "shape": t["dims"], "n_elements": t["n_elements"], "n_bytes": t["n_bytes"]}
                for t in h.tensors]
    finally:
        h.close()


def _write_gguf_raw(version: int, alignment: int, n_kv: int, kv_bytes: bytes, tensors: list[tuple[str, list[int], int, bytes]], dst: str | os.PathLike) -> None:
    ti = bytearray()
    off = 0
    offs = []
    for name, dims, ttype, raw in tensors:
        nb = name.encode()
        ti += struct.pack("<Q", len(nb)) + nb + struct.pack("<I", len(dims)) + struct.pack(f"<{len(dims)}Q", *dims)
        ti += struct.pack("<IQ", ttype, off)
        offs.append(off)
        off = (off + len(raw) + alignment - 1) // alignment * alignment
    tmp = str(dst) + ".part"
    with open(tmp, "wb") as f:
        f.write(b"GGUF" + struct.pack("<IQQ", version, len(tensors), n_kv))
        f.write(kv_bytes)
        f.write(ti)
        pos = f.tell()
        f.write(b"\0" * ((pos + alignment - 1) // alignment * alignment - pos))
        base = f.tell()
        for (_, _, _, raw), o in zip(tensors, offs):
            f.write(b"\0" * (o - (f.tell() - base)))
            f.write(raw)
        end_pos = f.tell()
        f.write(b"\0" * ((end_pos + alignment - 1) // alignment * alignment - end_pos))
    os.replace(tmp, dst)


def copy_without_fast(src, dst, drop: set[str]) -> dict:
    """Byte-level removal: KV section copied verbatim, tensor table rewritten, kept tensor bytes copied."""
    h = _Hdr(src)
    try:
        names = {t["name"] for t in h.tensors}
        missing = drop - names
        if missing:
            raise ValueError(f"cannot drop tensors absent from source: {sorted(missing)[:5]}")
        kept = [t for t in h.tensors if t["name"] not in drop]
        A = h.alignment
        ti, off = bytearray(), 0
        for t in kept:
            nb = t["name"].encode()
            ti += struct.pack("<Q", len(nb)) + nb + struct.pack("<I", len(t["dims"])) + struct.pack(f"<{len(t['dims'])}Q", *t["dims"])
            ti += struct.pack("<IQ", t["ttype"], off)
            t["_new"] = off
            off = (off + t["n_bytes"] + A - 1) // A * A
        tmp = Path(str(dst) + ".part")
        with open(tmp, "wb") as f:
            f.write(b"GGUF" + struct.pack("<IQQ", h.version, len(kept), h.n_kv))
            f.write(h.m[h.kv_start : h.kv_end])
            f.write(ti)
            pos = f.tell()
            f.write(b"\0" * ((pos + A - 1) // A * A - pos))
            base = f.tell()
            for t in kept:
                cur = f.tell() - base
                f.write(b"\0" * (t["_new"] - cur))
                s = h.data_start + t["offset"]
                f.write(h.m[s : s + t["n_bytes"]])
            end_pos = f.tell()
            f.write(b"\0" * ((end_pos + A - 1) // A * A - end_pos))
        os.replace(tmp, dst)
        return {"dropped": sorted(drop), "n_dropped": len(drop),
                "dropped_bytes": int(sum(t["n_bytes"] for t in h.tensors if t["name"] in drop))}
    finally:
        h.close()


# ---------------------------------------------------------------- Wanda neuron ordering & variable-width / MoE slicing
def ffn_widths(path: str | os.PathLike) -> dict[int, int]:
    """Return `{layer_idx: n_ff}` for every present `blk.<il>.ffn_down.weight` or `ffn_down_exps.weight` tensor."""
    h = _Hdr(path)
    try:
        out: dict[int, int] = {}
        for t in h.tensors:
            if t["name"].endswith(("ffn_down.weight", "ffn_down_exps.weight")):
                out[int(t["name"].split(".")[1])] = int(t["dims"][0])
        return out
    finally:
        h.close()


def _decode_unquantized_f32(buf: mmap.mmap, offset: int, n_elements: int, tname: str) -> np.ndarray:
    if tname == "BF16":
        u16 = np.frombuffer(buf, dtype=np.uint16, count=n_elements, offset=offset).copy()
        return (u16.astype(np.uint32) << 16).view(np.float32)
    if tname == "F16":
        return np.frombuffer(buf, dtype=np.float16, count=n_elements, offset=offset).astype(np.float32)
    if tname == "F32":
        return np.frombuffer(buf, dtype=np.float32, count=n_elements, offset=offset).copy()
    raise ValueError(f"expects unquantized BF16/F16/F32 source tensor, got {tname}")


def wanda_order(
    src_gguf: str | os.PathLike,
    imatrix_gguf: str | os.PathLike | None = None,
    kept_experts: dict[int, Sequence[int] | np.ndarray] | None = None,
) -> dict[int, np.ndarray]:
    """Per-layer neuron importance order (descending): `||W_down[:, j]||_2 * sqrt(in_sum2_j)` (supports MoE `ffn_down_exps`)."""
    im: dict[int, np.ndarray] = {}
    im_exp: dict[int, np.ndarray] = {}
    if imatrix_gguf is not None:
        hi = _Hdr(imatrix_gguf)
        try:
            for t in hi.tensors:
                nm = t["name"]
                if nm.endswith("ffn_down.weight.in_sum2"):
                    il = int(nm.split(".")[1])
                    arr = np.frombuffer(hi.m, dtype=np.float32, count=t["n_elements"], offset=hi.data_start + t["offset"]).reshape(-1, t["dims"][0]).sum(axis=0).astype(np.float64)
                    im[il] = arr
                elif nm.endswith("ffn_down_exps.weight.in_sum2"):
                    il = int(nm.split(".")[1])
                    rows = int(t["dims"][1]) if len(t["dims"]) > 1 else 1
                    arr = np.frombuffer(hi.m, dtype=np.float32, count=t["n_elements"], offset=hi.data_start + t["offset"]).reshape(rows, t["dims"][0]).astype(np.float64)
                    im_exp[il] = arr
        finally:
            hi.close()

    hs = _Hdr(src_gguf)
    try:
        out: dict[int, np.ndarray] = {}
        for t in hs.tensors:
            nm = t["name"]
            if nm.endswith("ffn_down.weight"):
                il = int(nm.split(".")[1])
                if il not in im:
                    continue
                w = _decode_unquantized_f32(hs.m, hs.data_start + t["offset"], t["n_elements"], t["type"]).reshape(t["dims"][1], t["dims"][0])
                col_norm = np.sqrt((w.astype(np.float64) ** 2).sum(axis=0))
                score = col_norm * np.sqrt(np.maximum(im[il], 0.0))
                out[il] = np.argsort(-score, kind="stable").astype(np.int64)
            elif nm.endswith("ffn_down_exps.weight"):
                il = int(nm.split(".")[1])
                n_ff_exp, n_embd, n_exp = int(t["dims"][0]), int(t["dims"][1]), int(t["dims"][2])
                w = _decode_unquantized_f32(hs.m, hs.data_start + t["offset"], t["n_elements"], t["type"]).reshape(n_exp, n_embd, n_ff_exp)
                e_idx = None
                if kept_experts is not None and il in kept_experts:
                    e_idx = np.asarray(kept_experts[il], dtype=np.int64)
                    w = w[e_idx]
                col_sq = (w.astype(np.float64) ** 2).sum(axis=1)  # (K, n_ff_exp)
                if il in im_exp:
                    ex_im = np.maximum(im_exp[il], 0.0)
                    if e_idx is not None and ex_im.shape[0] == n_exp:
                        ex_im = ex_im[e_idx]
                    score = np.sqrt((col_sq * ex_im).sum(axis=0))
                else:
                    score = np.sqrt(col_sq.sum(axis=0))
                out[il] = np.argsort(-score, kind="stable").astype(np.int64)
        return out
    finally:
        hs.close()


def expert_importance_from_imatrix(
    src_gguf: str | os.PathLike,
    imatrix_gguf: str | os.PathLike | None = None,
) -> dict[int, np.ndarray]:
    """Per-layer routed expert importance order (descending): activation-weighted Frobenius energy with weight-norm fallback."""
    im_exp: dict[int, np.ndarray] = {}
    if imatrix_gguf is not None:
        hi = _Hdr(imatrix_gguf)
        try:
            for t in hi.tensors:
                if t["name"].endswith("ffn_down_exps.weight.in_sum2"):
                    il = int(t["name"].split(".")[1])
                    rows = int(t["dims"][1]) if len(t["dims"]) > 1 else 1
                    arr = np.frombuffer(hi.m, dtype=np.float32, count=t["n_elements"], offset=hi.data_start + t["offset"]).reshape(rows, t["dims"][0]).astype(np.float64)
                    im_exp[il] = arr
        finally:
            hi.close()

    hs = _Hdr(src_gguf)
    try:
        out: dict[int, np.ndarray] = {}
        for t in hs.tensors:
            if t["name"].endswith("ffn_down_exps.weight"):
                il = int(t["name"].split(".")[1])
                n_ff_exp, n_embd, n_exp = int(t["dims"][0]), int(t["dims"][1]), int(t["dims"][2])
                w = _decode_unquantized_f32(hs.m, hs.data_start + t["offset"], t["n_elements"], t["type"]).reshape(n_exp, n_embd, n_ff_exp)
                col_sq = (w.astype(np.float64) ** 2).sum(axis=1)  # (n_exp, n_ff_exp)
                if il in im_exp and im_exp[il].shape == col_sq.shape:
                    score = (col_sq * np.maximum(im_exp[il], 0.0)).sum(axis=1) + 1e-12 * col_sq.sum(axis=1)
                else:
                    score = col_sq.sum(axis=1)
                out[il] = np.argsort(-score, kind="stable").astype(np.int64)
        return out
    finally:
        hs.close()


def _splice_u32_kvs(h: _Hdr, updates: dict[str, int | Sequence[int]]) -> bytes:
    """Rewrite u32 scalar or u32 array KV entries in-place within `h.m[h.kv_start:h.kv_end]`."""
    if not updates:
        return bytes(h.m[h.kv_start : h.kv_end])
    m, p = h.m, h.kv_start
    out = bytearray()
    seen: set[str] = set()
    for _ in range(h.n_kv):
        p0 = p
        (kl,) = struct.unpack_from("<Q", m, p); p += 8
        k = m[p : p + kl].decode(); p += kl
        (vt,) = struct.unpack_from("<I", m, p); p += 4
        e = h._skip(p, vt)
        if k in updates:
            val = updates[k]
            out += bytes(m[p0 : p - 4])
            if isinstance(val, int):
                out += struct.pack("<II", 4, int(val))
            else:
                arr = [int(x) for x in val]
                out += struct.pack("<IIQ", _ARR, 4, len(arr)) + struct.pack(f"<{len(arr)}I", *arr)
            seen.add(k)
        else:
            out += bytes(m[p0:e])
        p = e
    missing = set(updates) - seen
    if missing:
        raise KeyError(f"missing GGUF KV key(s): {sorted(missing)}")
    return bytes(out)


def _splice_ff_kv(h: _Hdr, widths_per_layer: Sequence[int]) -> bytes:
    """Rewrite `{arch}.feed_forward_length` (scalar u32 OR existing u32 array) as a u32 array."""
    return _splice_u32_kvs(h, {f"{h.arch}.feed_forward_length": list(widths_per_layer)})


def subset_imatrix(
    imatrix_src: str | os.PathLike,
    imatrix_dst: str | os.PathLike,
    kept_neurons: dict[int, Sequence[int] | np.ndarray],
    kept_experts: dict[int, Sequence[int] | np.ndarray] | None = None,
) -> None:
    """Subset imatrix stats (`in_sum2`, `in_sum`, `counts`) for kept MLP neurons and/or kept MoE experts."""
    keep = {int(il): np.asarray(idx, dtype=np.int64) for il, idx in (kept_neurons or {}).items()}
    exp_keep = {int(il): np.asarray(idx, dtype=np.int64) for il, idx in (kept_experts or {}).items()}
    hi = _Hdr(imatrix_src)
    try:
        items = []
        for t in hi.tensors:
            p = t["name"].split(".")
            il = int(p[1]) if len(p) > 1 and p[0] == "blk" and p[1].isdigit() else -1
            if il in keep and len(keep[il]) == 0 and len(p) > 2 and p[2] in ("ffn_down", "ffn_gate", "ffn_up"):
                continue
            s0 = hi.data_start + t["offset"]
            raw = bytes(hi.m[s0 : s0 + t["n_bytes"]])
            dims = list(t["dims"])
            if len(p) > 3 and p[2].endswith("_exps") and p[-1] in ("in_sum2", "in_sum", "counts"):
                rows = int(dims[1]) if len(dims) > 1 else 1
                arr = np.frombuffer(raw, dtype=np.float32).reshape(rows, int(dims[0]))
                if il in exp_keep:
                    arr = arr[exp_keep[il], :]
                    dims = [int(dims[0]), int(arr.shape[0])]
                if il in keep and p[2] == "ffn_down_exps" and p[-1] in ("in_sum2", "in_sum"):
                    arr = arr[:, keep[il]]
                    dims = [int(arr.shape[1])] + dims[1:]
                raw = np.ascontiguousarray(arr).tobytes()
            elif il in keep and len(p) > 3 and p[2] == "ffn_down" and p[-1] in ("in_sum2", "in_sum"):
                arr = np.frombuffer(raw, dtype=np.float32).reshape(-1, dims[0])[:, keep[il]]
                raw = np.ascontiguousarray(arr).tobytes()
                dims = [int(arr.shape[1])] + dims[1:]
            items.append((t["name"], dims, t["ttype"], raw))
        _write_gguf_raw(hi.version, hi.alignment, hi.n_kv, bytes(hi.m[hi.kv_start : hi.kv_end]), items, imatrix_dst)
    finally:
        hi.close()


def _write_mv(f, mv: memoryview, start: int, nbytes: int, chunk: int = 16 << 20) -> None:
    end = start + nbytes
    p = start
    while p < end:
        q = min(p + chunk, end)
        f.write(mv[p:q])
        p = q


def _write_arr(f, arr: np.ndarray) -> None:
    c = np.ascontiguousarray(arr)
    with memoryview(c.ravel().view(np.uint8)) as amv:
        _write_mv(f, amv, 0, c.nbytes)


def write_variable_width(
    src_gguf: str | os.PathLike,
    dst_gguf: str | os.PathLike,
    kept_neurons: dict[int, Sequence[int] | np.ndarray],
    drop_tensors: set[str] | None = None,
    imatrix_src: str | os.PathLike | None = None,
    imatrix_dst: str | os.PathLike | None = None,
    kept_experts: dict[int, Sequence[int] | np.ndarray] | None = None,
) -> dict:
    """Write `dst_gguf` from unquantized `src_gguf` with dropped tensors removed and MLP widths / MoE experts sliced."""
    drop = set(drop_tensors or ())
    keep = {int(il): np.sort(np.asarray(idx, dtype=np.int64)) for il, idx in (kept_neurons or {}).items()}
    exp_keep = {int(il): np.sort(np.asarray(idx, dtype=np.int64)) for il, idx in (kept_experts or {}).items()}
    h = _Hdr(src_gguf)
    try:
        kvs = h.parsed_kvs()
        arch = h.arch or ""
        n_layer = int(kvs[f"{arch}.block_count"])
        raw_ff_exp = kvs.get(f"{arch}.expert_feed_forward_length")
        raw_ff = kvs.get(f"{arch}.feed_forward_length", raw_ff_exp)
        is_moe_ff = raw_ff_exp is not None or any(t["name"].endswith("ffn_down_exps.weight") for t in h.tensors)
        base_ff = raw_ff_exp if (is_moe_ff and raw_ff_exp is not None) else raw_ff
        n0 = list(base_ff) if isinstance(base_ff, list) else [int(base_ff)] * n_layer
        e0: dict[int, int] = {}
        for t in h.tensors:
            if t["name"].endswith("ffn_down.weight"):
                n0[int(t["name"].split(".")[1])] = int(t["dims"][0])
            elif t["name"].endswith("ffn_down_exps.weight"):
                il = int(t["name"].split(".")[1])
                n0[il] = int(t["dims"][0])
                e0[il] = int(t["dims"][2])
            elif t["name"].endswith("ffn_gate_inp.weight"):
                e0.setdefault(int(t["name"].split(".")[1]), int(t["dims"][1]))

        active_keep = {il: idx for il, idx in keep.items() if len(idx) < n0[il]}
        active_exp = {il: idx for il, idx in exp_keep.items() if il in e0 and len(idx) < e0[il]}

        kv_updates: dict[str, int | list[int]] = {}
        if active_keep:
            widths_arr = [int(len(active_keep[il])) if il in active_keep else int(n0[il]) for il in range(n_layer)]
            if is_moe_ff:
                if len(set(widths_arr)) == 1 and not isinstance(raw_ff_exp, list):
                    kv_updates[f"{arch}.expert_feed_forward_length"] = widths_arr[0]
                    if f"{arch}.feed_forward_length" in kvs and not isinstance(kvs[f"{arch}.feed_forward_length"], list):
                        kv_updates[f"{arch}.feed_forward_length"] = widths_arr[0]
                else:
                    kv_updates[f"{arch}.expert_feed_forward_length"] = widths_arr
                    if f"{arch}.feed_forward_length" in kvs:
                        kv_updates[f"{arch}.feed_forward_length"] = widths_arr
            else:
                kv_updates[f"{arch}.feed_forward_length"] = widths_arr

        if active_exp:
            kept_exp_counts = {len(active_exp[il]) if il in active_exp else e0[il] for il in e0}
            if len(kept_exp_counts) != 1:
                raise ValueError(f"MoE expert_count must remain uniform across layers, got counts {sorted(kept_exp_counts)}")
            k_exp = next(iter(kept_exp_counts))
            n_used = kvs.get(f"{arch}.expert_used_count")
            n_used_max = max(n_used) if isinstance(n_used, list) else int(n_used or 1)
            if k_exp < n_used_max:
                raise ValueError(f"kept expert_count={k_exp} < expert_used_count={n_used_max}")
            kv_updates[f"{arch}.expert_count"] = int(k_exp)

        kv_bytes = _splice_u32_kvs(h, kv_updates) if kv_updates else bytes(h.m[h.kv_start : h.kv_end])

        elem_dtype = {"BF16": np.uint16, "F16": np.uint16, "F32": np.uint32}
        elem_size = {"BF16": 2, "F16": 2, "F32": 4}
        A = h.alignment
        ti = bytearray()
        off = 0
        plan: list[tuple[dict, int, str, list[int], int, int, bool, bool]] = []
        for t in h.tensors:
            name = t["name"]
            if name in drop:
                continue
            p = name.split(".")
            il = int(p[1]) if len(p) > 1 and p[0] == "blk" and p[1].isdigit() else -1
            kind = p[2] if il >= 0 and len(p) > 2 else ""
            if il in active_keep and len(active_keep[il]) == 0 and kind in ("ffn_gate", "ffn_up", "ffn_down", "ffn_norm", "post_ffw_norm"):
                continue
            need_w = il in active_keep and kind in ("ffn_gate", "ffn_up", "ffn_down", "ffn_gate_up_exps", "ffn_down_exps", "ffn_gate_exps", "ffn_up_exps")
            need_e = il in active_exp and kind in ("ffn_gate_up_exps", "ffn_down_exps", "ffn_gate_exps", "ffn_up_exps", "ffn_gate_inp", "exp_probs_b")
            dims = list(t["dims"])
            if need_w or need_e:
                es = elem_size.get(t["type"])
                if es is None:
                    raise ValueError(f"write_variable_width requires BF16/F16/F32 source tensor, got {name} ({t['type']})")
                if kind in ("ffn_gate", "ffn_up", "ffn_down"):
                    if kind == "ffn_down":
                        new_dims = [int(len(active_keep[il])), int(dims[1])]
                    else:
                        new_dims = [int(dims[0]), int(len(active_keep[il]))]
                elif kind in ("ffn_gate_up_exps", "ffn_down_exps", "ffn_gate_exps", "ffn_up_exps"):
                    d0, d1, d2 = int(dims[0]), int(dims[1]), int(dims[2])
                    if need_e:
                        d2 = int(len(active_exp[il]))
                    if need_w:
                        wn = int(len(active_keep[il]))
                        if kind == "ffn_down_exps":
                            d0 = wn
                        elif kind == "ffn_gate_up_exps":
                            d1 = 2 * wn
                        else:
                            d1 = wn
                    new_dims = [d0, d1, d2]
                elif kind == "ffn_gate_inp":
                    new_dims = [int(dims[0]), int(len(active_exp[il]))]
                else:
                    new_dims = [int(len(active_exp[il]))]
                n_el = 1
                for d in new_dims:
                    n_el *= d
                new_n_bytes = n_el * es
            else:
                new_dims = dims
                new_n_bytes = int(t["n_bytes"])
            nb = name.encode()
            ti += struct.pack("<Q", len(nb)) + nb + struct.pack("<I", len(new_dims)) + struct.pack(f"<{len(new_dims)}Q", *new_dims)
            ti += struct.pack("<IQ", t["ttype"], off)
            plan.append((t, il, kind, new_dims, new_n_bytes, off, need_w, need_e))
            off = (off + new_n_bytes + A - 1) // A * A

        def _stream_write() -> None:
            tmp = str(dst_gguf) + ".part"
            try:
                with memoryview(h.m) as mv, open(tmp, "wb") as f:
                    f.write(b"GGUF" + struct.pack("<IQQ", h.version, len(plan), h.n_kv))
                    f.write(kv_bytes)
                    f.write(ti)
                    pos = f.tell()
                    f.write(b"\0" * ((pos + A - 1) // A * A - pos))
                    base = f.tell()
                    for t, il, kind, _new_dims, new_n_bytes, new_off, need_w, need_e in plan:
                        pad = new_off - (f.tell() - base)
                        if pad:
                            f.write(b"\0" * pad)
                        s0 = h.data_start + t["offset"]
                        if not (need_w or need_e):
                            _write_mv(f, mv, s0, t["n_bytes"])
                        else:
                            dt = elem_dtype[t["type"]]
                            es = elem_size[t["type"]]
                            dims = t["dims"]
                            if kind in ("ffn_gate", "ffn_up", "ffn_down"):
                                v = np.frombuffer(h.m, dtype=dt, count=t["n_elements"], offset=s0).reshape(dims[1], dims[0])
                                idx = active_keep[il]
                                sub = v[:, idx] if kind == "ffn_down" else v[idx, :]
                                del v
                                _write_arr(f, sub)
                                del sub
                            elif kind in ("ffn_gate_up_exps", "ffn_down_exps", "ffn_gate_exps", "ffn_up_exps"):
                                d0, d1, d2 = int(dims[0]), int(dims[1]), int(dims[2])
                                exp_elems = d1 * d0
                                exp_bytes = exp_elems * es
                                e_indices = active_exp[il].tolist() if need_e else range(d2)
                                if not need_w:
                                    for e in e_indices:
                                        _write_mv(f, mv, s0 + e * exp_bytes, exp_bytes)
                                else:
                                    w_idx = active_keep[il]
                                    idx_gu = np.concatenate([w_idx, (d1 // 2) + w_idx]) if kind == "ffn_gate_up_exps" else None
                                    for e in e_indices:
                                        v = np.frombuffer(h.m, dtype=dt, count=exp_elems, offset=s0 + e * exp_bytes).reshape(d1, d0)
                                        if kind == "ffn_down_exps":
                                            sub = v[:, w_idx]
                                        elif kind == "ffn_gate_up_exps":
                                            sub = v[idx_gu, :]
                                        else:
                                            sub = v[w_idx, :]
                                        del v
                                        _write_arr(f, sub)
                                        del sub
                            elif kind == "ffn_gate_inp":
                                d0, d1 = int(dims[0]), int(dims[1])
                                v = np.frombuffer(h.m, dtype=dt, count=t["n_elements"], offset=s0).reshape(d1, d0)
                                sub = v[active_exp[il], :]
                                del v
                                _write_arr(f, sub)
                                del sub
                            else:
                                v = np.frombuffer(h.m, dtype=dt, count=t["n_elements"], offset=s0)
                                sub = v[active_exp[il]]
                                del v
                                _write_arr(f, sub)
                                del sub
                        if (f.tell() - base) != new_off + new_n_bytes:
                            raise RuntimeError(f"{t['name']}: wrote {(f.tell() - base) - new_off} B != planned {new_n_bytes} B")
                    end_pos = f.tell()
                    f.write(b"\0" * ((end_pos + A - 1) // A * A - end_pos))
                os.replace(tmp, dst_gguf)
            except BaseException:
                Path(tmp).unlink(missing_ok=True)
                raise

        _stream_write()
    finally:
        h.close()

    if imatrix_src and imatrix_dst:
        if active_keep or active_exp:
            subset_imatrix(imatrix_src, imatrix_dst, active_keep, kept_experts=active_exp)
        elif Path(imatrix_src).resolve() != Path(imatrix_dst).resolve():
            __import__("shutil").copyfile(imatrix_src, imatrix_dst)
    res = {"widths": {il: int(len(idx)) for il, idx in active_keep.items()}, "file_bytes": os.path.getsize(dst_gguf)}
    if active_exp:
        res["experts"] = {il: int(len(idx)) for il, idx in active_exp.items()}
    return res


def arch_info_fast(path) -> dict:
    return arch_info(path)


inventory = inventory_fast
copy_without = copy_without_fast
