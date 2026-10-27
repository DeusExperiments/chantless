"""GGUF inspection and physical tensor removal (zero-bit = absent tensor, never stored zeros)."""
from __future__ import annotations

import os
import re
from pathlib import Path

import gguf

# The "attn" group is the token-mixer sub-block: full attention (qwen3, qwen35) or gated delta net (qwen35).
# qwen35's post_attention_norm is the FFN's pre-norm, so it belongs to the FFN group.
QUANTIZABLE_ATTN = ("attn_q", "attn_k", "attn_v", "attn_output", "attn_qkv", "attn_gate", "ssm_out", "ssm_alpha", "ssm_beta")
QUANTIZABLE_FFN = ("ffn_gate", "ffn_up", "ffn_down")
ATTN_FIXED = ("attn_norm", "attn_q_norm", "attn_k_norm", "ssm_conv1d", "ssm_dt", "ssm_a", "ssm_norm")
FFN_FIXED = ("ffn_norm", "post_attention_norm")
BLK = re.compile(r"^blk\.(\d+)\.([a-z0-9_]+?)(?:\.(weight|bias))?$")


def inventory_slow(path: str | os.PathLike) -> list[dict]:
    r = gguf.GGUFReader(str(path))
    return [{"name": t.name, "type": t.tensor_type.name, "shape": [int(x) for x in t.shape], "n_elements": int(t.n_elements),
             "n_bytes": int(t.n_bytes)} for t in r.tensors]


def arch_info(path: str | os.PathLike) -> dict:
    r = gguf.GGUFReader(str(path))
    arch = r.get_field("general.architecture").contents()
    def g(k, default=None):
        f = r.get_field(f"{arch}.{k}")
        v = f.contents() if f else default
        return max(v) if isinstance(v, list) else v  # per-layer arrays (hybrids store 0 for recurrent layers)
    return {"arch": arch, "n_layer": int(g("block_count")), "n_embd": int(g("embedding_length")),
            "n_head": int(g("attention.head_count")), "n_head_kv": int(g("attention.head_count_kv")),
            "head_dim": int(g("attention.key_length", int(g("embedding_length")) // int(g("attention.head_count"))))}


def groups_from_inventory(inv: list[dict]) -> dict[str, dict]:
    """Group map generated from the header: every tensor lands in exactly one group or the fixed set."""
    groups: dict[str, dict] = {}
    fixed = []
    # MTP / NextN draft blocks are never part of the searched trunk
    mtp_blocks = {p[1] for t in inv if len(p := t["name"].split(".")) > 3 and p[0] == "blk" and p[2] == "nextn"}
    for t in inv:
        name = t["name"]
        m = BLK.match(name)
        if name.startswith("blk.") and name.split(".")[1] in mtp_blocks:
            fixed.append(name)
        elif name == "token_embd.weight":
            groups.setdefault("token_embd", {"quantizable": [], "fixed": [], "droppable": False})["quantizable"].append(name)
        elif name == "output.weight":
            groups.setdefault("output", {"quantizable": [], "fixed": [], "droppable": False})["quantizable"].append(name)
        elif m:
            i, kind = int(m.group(1)), m.group(2)
            if kind in QUANTIZABLE_ATTN or kind in ATTN_FIXED:
                g = groups.setdefault(f"blk.{i}.attn", {"quantizable": [], "fixed": [], "droppable": True})
                (g["quantizable"] if kind in QUANTIZABLE_ATTN else g["fixed"]).append(name)
            elif kind in QUANTIZABLE_FFN or kind in FFN_FIXED:
                g = groups.setdefault(f"blk.{i}.ffn", {"quantizable": [], "fixed": [], "droppable": True})
                (g["quantizable"] if kind in QUANTIZABLE_FFN else g["fixed"]).append(name)
            else:
                fixed.append(name)
        else:
            fixed.append(name)
    for g, e in groups.items():
        if g.endswith(".attn"):  # full attention keeps a KV cache; a gated delta net keeps constant recurrent state
            e["kv"] = any(n.endswith(".attn_output.weight") for n in e["quantizable"])
    groups["_fixed"] = {"quantizable": [], "fixed": fixed, "droppable": False}
    seen = [n for g in groups.values() for n in g["quantizable"] + g["fixed"]]
    assert sorted(seen) == sorted(t["name"] for t in inv), "group map does not partition the tensors"
    return groups


def copy_without_slow(src: str | os.PathLike, dst: str | os.PathLike, drop: set[str]) -> dict:
    """Write `dst` = `src` minus the named tensors (metadata copied exactly). Returns a removal receipt."""
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
    """Test helper: same file with the named tensors' data set to zero (same size; earns no byte credit)."""
    import numpy as np
    shutil_copy = __import__("shutil").copyfile
    shutil_copy(src, dst)
    r = gguf.GGUFReader(str(dst), "r+")
    for t in r.tensors:
        if t.name in zero:
            t.data[...] = np.zeros_like(t.data)
    del r


# ---------------------------------------------------------------- fast raw GGUF header handling
# gguf-py's reader materialises every tokenizer string (~151k) as numpy objects; parsing the header
# directly is orders of magnitude faster and lets removal copy the KV section verbatim.
import mmap
import struct

_SCALAR = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}   # GGUF value type -> bytes
_STR, _ARR = 8, 9


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
            qt = gguf.GGMLQuantizationType(t["ttype"])
            bs, ts = gguf.GGML_QUANT_SIZES[qt]
            n = 1
            for d in t["dims"]:
                n *= d
            t["type"], t["n_elements"], t["n_bytes"] = qt.name, n, n // bs * ts

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

    def close(self):
        self.m.close(); self.f.close()


def inventory_fast(path) -> list[dict]:
    h = _Hdr(path)
    try:
        return [{"name": t["name"], "type": t["type"], "shape": t["dims"], "n_elements": t["n_elements"], "n_bytes": t["n_bytes"]}
                for t in h.tensors]
    finally:
        h.close()


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
        os.replace(tmp, dst)
        return {"dropped": sorted(drop), "n_dropped": len(drop),
                "dropped_bytes": int(sum(t["n_bytes"] for t in h.tensors if t["name"] in drop))}
    finally:
        h.close()


def arch_info_fast(path) -> dict:
    return arch_info(path)


inventory = inventory_fast
copy_without = copy_without_fast
