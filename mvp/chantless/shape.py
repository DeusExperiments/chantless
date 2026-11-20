"""Materialize variable-width, mixed-precision, vocab-trimmed GGUFs and price candidate structures.

Pipeline: write variable-width GGUF (omitting dropped sub-blocks) -> quantize via `Builder` -> trim
vocabulary. Kept neurons use source indices so cuts compose cumulatively; `tiers` maps every group to a
ggml tier or `'drop'`.
"""
from __future__ import annotations

import hashlib
import os
import shutil
import struct
from pathlib import Path

import gguf
import numpy as np

from . import gguf_ops, lattice, mask as maskmod
from .builder import Builder
from .learn import floats

FFN = ("ffn_gate", "ffn_up", "ffn_down")
STR, ARR, U32, I32, F32 = 8, 9, 4, 5, 0
MB = 1_000_000


def _find(h, key: str) -> tuple[int, int, int, int]:
    """-> (entry start, value start, entry end, value type) of one KV."""
    m, p = h.m, h.kv_start
    for _ in range(h.n_kv):
        p0 = p
        (kl,) = struct.unpack_from("<Q", m, p); p += 8
        k = m[p:p + kl].decode(); p += kl
        (vt,) = struct.unpack_from("<I", m, p); p += 4
        e = h._skip(p, vt)
        if k == key:
            return p0, p, e, vt
        p = e
    raise KeyError(key)


def widths_of(h) -> list[int]:
    """Neurons per layer, from either form of the KV. A layer with no FFN reads as full width."""
    _, p, _, vt = _find(h, f"{h.arch}.feed_forward_length")
    if vt == ARR:
        n = struct.unpack_from("<IQ", h.m, p)[1]
        return list(struct.unpack_from(f"<{n}I", h.m, p + 12))
    n_layer = struct.unpack_from("<I", h.m, _find(h, f"{h.arch}.block_count")[1])[0]
    return [struct.unpack_from("<I", h.m, p)[0]] * n_layer


def splice_kv(h, widths: list[int]) -> bytes:
    """The KV section with the scalar width replaced by a uint32 array. The rest stays verbatim, tokenizer included."""
    _, p, e, vt = _find(h, f"{h.arch}.feed_forward_length")
    assert vt == U32
    return h.m[h.kv_start:p - 4] + struct.pack("<IIQ", ARR, U32, len(widths)) + struct.pack(f"<{len(widths)}I", *widths) + h.m[e:h.kv_end]


def _write(path: str, version: int, n_kv: int, kv: bytes, items: list[tuple], A: int, tail: bool = False) -> None:
    """A GGUF from (name, dims, ggml type, buffer) items. `tail` pads the last tensor as well, which gguf_init wants when it reads the data."""
    ti, offs, off = bytearray(), [], 0
    for name, dims, ttype, raw in items:
        n = name.encode()
        ti += struct.pack("<Q", len(n)) + n + struct.pack("<I", len(dims)) + struct.pack(f"<{len(dims)}Q", *dims) + struct.pack("<IQ", ttype, off)
        offs.append(off)
        off = (off + raw.nbytes + A - 1) // A * A
    tmp = str(path) + ".part"
    with open(tmp, "wb") as f:
        f.write(b"GGUF" + struct.pack("<IQQ", version, len(items), n_kv))
        f.write(kv)
        f.write(ti)
        pos = f.tell()
        f.write(b"\0" * ((pos + A - 1) // A * A - pos))
        base = f.tell()
        for (_, _, _, raw), o in zip(items, offs):
            f.write(b"\0" * (o - (f.tell() - base)))
            f.write(raw)
        if tail:
            pos = f.tell()
            f.write(b"\0" * ((pos + A - 1) // A * A - pos))
    os.replace(tmp, path)


def _raw(h, t: dict) -> memoryview:
    s = h.data_start + t["offset"]
    return memoryview(h.m)[s:s + t["n_bytes"]]


def _layer(name: str) -> tuple[int, str]:
    p = name.split(".")
    return (int(p[1]), p[2]) if p[0] == "blk" else (-1, "")


def carve(a: np.ndarray, kind: str, idx: np.ndarray) -> np.ndarray:
    """Neurons are rows of ffn_gate and ffn_up, columns of ffn_down."""
    return a[:, idx] if kind == "ffn_down" else a[idx]


def bf16_floats(h, t: dict) -> np.ndarray:
    """A BF16 tensor as float32 in torch shape, exactly."""
    a = (np.frombuffer(_raw(h, t), "<u2").astype(np.uint32) << 16).view(np.float32)
    return a.reshape(tuple(reversed(t["dims"])))


def write_var(h, keep: dict[int, np.ndarray], path: str, latents: dict[str, np.ndarray] | None = None,
              basis: dict[int, np.ndarray] | None = None, drop: set[str] = frozenset()) -> None:
    """`h` at the widths of `keep`, without the tensors in `drop`. A layer not in `keep` stays full width.

    `latents` are float32 in torch shape by GGUF name and are written as F32 in place of the source's bytes. A layer's FFN
    latents hold the neurons `basis[il]` (source indices, default exactly `keep[il]`), so weights trained at one width can
    be written at a narrower one. Everything else is the source's bytes, untouched.
    """
    latents, basis = latents or {}, basis or {}
    names = {t["name"] for t in h.tensors}
    assert latents.keys() <= names, sorted(latents.keys() - names)[:3]
    widths = widths_of(h)
    full = widths[0]
    widths = [len(keep[i]) if i in keep else full for i in range(len(widths))]
    items = []
    for t in h.tensors:
        name = t["name"]
        if name in drop:
            continue
        il, kind = _layer(name)
        cut = il in keep and kind in FFN
        dims, ttype, a = list(t["dims"]), t["ttype"], latents.get(name)
        if cut:
            dims[0 if kind == "ffn_down" else 1] = len(keep[il])
        if a is not None:
            if cut:
                src = np.asarray(basis.get(il, keep[il]))
                assert np.isin(keep[il], src).all(), f"layer {il}: keep is not inside basis"
                a = carve(a, kind, np.searchsorted(src, keep[il]))
            assert a.shape == tuple(reversed(dims)), (name, a.shape, dims)
            raw, ttype = memoryview(np.ascontiguousarray(a, "<f4")), F32
        elif cut:
            assert t["type"] == "BF16", name
            v = np.frombuffer(_raw(h, t), "<u2").reshape(tuple(reversed(t["dims"])))
            raw = memoryview(np.ascontiguousarray(carve(v, kind, keep[il])))
        else:
            raw = _raw(h, t)
        items.append((name, dims, ttype, raw))
    _write(path, h.version, h.n_kv, splice_kv(h, widths), items, h.alignment)


def sub_imatrix(imatrix: str, keep: dict[int, np.ndarray], path: str) -> None:
    """The imatrix for a var GGUF: ffn_down.in_sum2 at the kept columns. Gate and up in_sum2 are input-side, width n_embd, unchanged."""
    h = gguf_ops._Hdr(imatrix)
    items = []
    for t in h.tensors:
        il, kind = _layer(t["name"])
        raw, dims = _raw(h, t), t["dims"]
        if il in keep and kind == "ffn_down" and t["name"].endswith("in_sum2"):
            raw, dims = memoryview(np.frombuffer(raw, "<f4")[keep[il]]), [len(keep[il])]
        items.append((t["name"], dims, t["ttype"], raw))
    _write(path, h.version, h.n_kv, h.m[h.kv_start:h.kv_end], items, h.alignment, tail=True)


def _kvs(h) -> list[tuple[str, int, bytes]]:
    m, p, out = h.m, h.kv_start, []
    for _ in range(h.n_kv):
        (kl,) = struct.unpack_from("<Q", m, p); p += 8
        key = m[p:p + kl].decode(); p += kl
        (vt,) = struct.unpack_from("<I", m, p); p += 4
        s = p
        p = h._skip(p, vt)
        out.append((key, vt, bytes(m[s:p])))
    return out


def _strs(raw: bytes) -> list[bytes]:
    et, n = struct.unpack_from("<IQ", raw, 0)
    assert et == STR
    p, out = 12, []
    for _ in range(n):
        (sl,) = struct.unpack_from("<Q", raw, p); p += 8
        out.append(raw[p:p + sl]); p += sl
    return out


def _pack_strs(items: list[bytes]) -> bytes:
    b = bytearray(struct.pack("<IQ", STR, len(items)))
    for s in items:
        b += struct.pack("<Q", len(s)) + s
    return bytes(b)


def trim(src: str, dst: str, keep: np.ndarray) -> None:
    """`src` shrunk to a kept token set at any tier, with no requantisation.

    Rows of token_embd are quantised independently, so the kept rows are copied byte for byte. tokens, token_type and merges
    are subset in order, which preserves merge ranks; token ids are renumbered. `keep` must be sorted, closed under merge
    parents, and hold every special token and all 256 byte tokens. New id = position in `keep`.
    """
    h = gguf_ops._Hdr(src)
    keep = np.asarray(keep, np.int64)
    assert (np.diff(keep) > 0).all()
    kvs = _kvs(h)
    kvd = {k: raw for k, _, raw in kvs}
    tokens = _strs(kvd["tokenizer.ggml.tokens"])
    V = len(tokens)
    new_id = np.full(V, -1, np.int64)
    new_id[keep] = np.arange(len(keep))
    ttype = np.frombuffer(kvd["tokenizer.ggml.token_type"], "<i4", V, 12)
    ks = {tokens[i] for i in keep.tolist()}
    merges = []
    for r in _strs(kvd["tokenizer.ggml.merges"]):
        a, b = r.split(b" ", 1)
        if a in ks and b in ks and a + b in ks:
            merges.append(r)
    kv = bytearray()
    for k, vt, raw in kvs:
        if k == "tokenizer.ggml.tokens":
            raw = _pack_strs([tokens[i] for i in keep.tolist()])
        elif k == "tokenizer.ggml.token_type":
            raw = struct.pack("<IQ", I32, len(keep)) + ttype[keep].astype("<i4").tobytes()
        elif k == "tokenizer.ggml.merges":
            raw = _pack_strs(merges)
        elif k.startswith("tokenizer.ggml.") and k.endswith("_token_id"):
            (o,) = struct.unpack("<I", raw)
            assert new_id[o] >= 0, f"{k}={o} is not in the keep set"
            raw = struct.pack("<I", int(new_id[o]))
        b = k.encode()
        kv += struct.pack("<Q", len(b)) + b + struct.pack("<I", vt) + raw
    items = []
    for t in h.tensors:
        raw, dims = _raw(h, t), t["dims"]
        if t["name"] == "token_embd.weight":
            rows = np.frombuffer(raw, np.uint8).reshape(dims[1], t["n_bytes"] // dims[1])
            raw, dims = memoryview(rows[keep]), [dims[0], len(keep)]
        items.append((t["name"], dims, t["ttype"], raw))
    _write(dst, h.version, len(kvs), bytes(kv), items, h.alignment)


def as_mask(tiers: dict[str, str], groups: dict) -> dict:
    m = maskmod.new_mask(groups, "Q4_K", {})
    m["groups"] = {g: {"tier": t} for g, t in tiers.items()}
    return m


def tensor_bytes(dims: list[int], tier: str) -> int:
    blck, size = gguf.GGML_QUANT_SIZES[gguf.GGMLQuantizationType[tier]]
    return int(np.prod(dims)) // blck * size


class Shaper:
    """Builds GGUFs from a fixed source, imatrix, and vocab keep-set in a fresh per-output work directory
    (deterministic path keeps `quantize.imatrix.file` reproducible; prefer tmpfs/RAM disk for `work`)."""

    def __init__(self, tc, src: str, groups: dict, imatrix: str, vocab: np.ndarray | str, work: str | os.PathLike, threads: int = 16, log: str | None = None):
        self.tc, self.src, self.groups, self.imatrix, self.threads, self.log = tc, str(src), groups, str(imatrix), threads, log
        self.vocab = np.load(vocab) if isinstance(vocab, (str, Path)) else vocab
        self.work = Path(work)
        self.work.mkdir(parents=True, exist_ok=True)

    def build(self, keep: dict[int, np.ndarray], tiers: dict[str, str], out: str | os.PathLike,
              latents: dict[str, np.ndarray] | None = None, basis: dict[int, np.ndarray] | None = None) -> tuple[str, int]:
        """-> (`out`, its bytes on disk). `keep` is {layer: sorted source neurons}; a layer it omits is full width."""
        drop = {n for g, t in tiers.items() if t == maskmod.DROP for n in self.groups[g]["quantizable"] + self.groups[g]["fixed"]}
        keep = {il: np.asarray(k, np.int64) for il, k in keep.items() if tiers[f"blk.{il}.ffn"] != maskmod.DROP}
        for il, k in keep.items():
            blck = gguf.GGML_QUANT_SIZES[gguf.GGMLQuantizationType[tiers[f"blk.{il}.ffn"]]][0]
            assert len(k) and len(k) % blck == 0 and (np.diff(k) > 0).all(), f"layer {il}: {len(k)} neurons, {tiers[f'blk.{il}.ffn']}"
        # write_var has already left dropped tensors out, so the groups Builder sees no longer name them
        live = {g: dict(e, quantizable=[n for n in e["quantizable"] if n not in drop], fixed=[n for n in e["fixed"] if n not in drop]) for g, e in self.groups.items()}
        d = self.work / f"shape-{hashlib.sha256(str(out).encode()).hexdigest()[:16]}"
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir()
        try:
            write_var(gguf_ops._Hdr(self.src), keep, str(d / "var.gguf"), latents, basis, drop)
            sub_imatrix(self.imatrix, keep, str(d / "im.gguf"))
            r = Builder(self.tc, d / "var.gguf", live, str(d / "im.gguf"), d / "work", log=self.log, threads=self.threads).build(as_mask(tiers, self.groups), d / "q.gguf")
            assert not r.fallbacks, r.fallbacks
            trim(str(d / "q.gguf"), str(d / "final.gguf"), self.vocab)
            tmp = str(out) + ".part"
            shutil.copyfile(d / "final.gguf", tmp)
            os.replace(tmp, out)
        finally:
            shutil.rmtree(d, ignore_errors=True)
        return str(out), gguf_ops.file_bytes(out)


def structure_of(built: str, src: str, groups: dict, ggml_lib: str | None) -> tuple[dict[int, np.ndarray], dict[str, str]]:
    """(keep, tiers) of a built file. Each cut neuron is matched to its nearest source neuron on all three FFN tensors together;
    a cut keeps source order, so the matches must rise. Works on learned files too, whose rows have moved only a little."""
    a, h = lattice.Artifact(built), gguf_ops._Hdr(src)
    tiers = {g: a.tensors[e["quantizable"][0]]["type"] if e["quantizable"][0] in a.tensors else maskmod.DROP for g, e in groups.items() if not g.startswith("_")}
    full, ts, keep = widths_of(h)[0], {t["name"]: t for t in h.tensors}, {}

    def unit(x: np.ndarray) -> np.ndarray:
        return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-30)

    for il, w in enumerate(widths_of(a.h)):
        if tiers[f"blk.{il}.ffn"] == maskmod.DROP:
            continue
        if w == full:
            keep[il] = np.arange(full)
            continue
        names = [f"blk.{il}.{k}.weight" for k in FFN]
        x = unit(np.hstack([floats(a, names[0], ggml_lib), floats(a, names[1], ggml_lib), floats(a, names[2], ggml_lib).T]))
        y = unit(np.hstack([bf16_floats(h, ts[names[0]]), bf16_floats(h, ts[names[1]]), bf16_floats(h, ts[names[2]]).T]))
        m = (x @ y.T).argmax(1)
        assert (np.diff(m) > 0).all(), f"layer {il}: matches do not rise"
        keep[il] = m
    return keep, tiers


class Ledger:
    """Prices structures from tensor dims alone, with the KV section read from `ref`, any file built at this vocabulary.
    `size` matches a file sharing `ref`'s header; another imatrix path, or no quantiser keys at all, moves it by a few hundred bytes."""

    def __init__(self, src: str, groups: dict, rows: int, ref: str):
        h, r = gguf_ops._Hdr(src), gguf_ops._Hdr(ref)
        owner = {n: (g, n in e["quantizable"]) for g, e in groups.items() for n in e["quantizable"] + e["fixed"]}
        self.items = [(t["name"], t["dims"], t["type"], *owner[t["name"]]) for t in h.tensors]
        self.src, self.groups, self.rows, self.kv, self.A = src, groups, rows, r.kv_end - r.kv_start, h.alignment

    def size(self, widths: dict[int, int], tiers: dict[str, str]) -> int:
        """Bytes of the built file with `widths` neurons per cut layer."""
        ti, data, A = 0, 0, self.A
        for name, dims, kind, g, quantized in self.items:
            if tiers.get(g) == maskmod.DROP:
                continue
            d, (il, k) = list(dims), _layer(name)
            if name == "token_embd.weight":
                d[1] = self.rows
            elif k in FFN and il in widths:
                d[0 if k == "ffn_down" else 1] = widths[il]
            ti += 8 + len(name.encode()) + 4 + 8 * len(d) + 12
            data += (tensor_bytes(d, tiers[g] if quantized else kind) + A - 1) // A * A
        return (24 + self.kv + ti + A - 1) // A * A + data

    def saved(self, widths: dict[int, int], tiers: dict[str, str], group: str | None = None, il: int | None = None, neurons: int = 256) -> int:
        """Bytes one move frees: dropping `group`, or taking `neurons` from layer `il`."""
        if group:
            return self.size(widths, tiers) - self.size(widths, {**tiers, group: maskmod.DROP})
        return self.size(widths, tiers) - self.size({**widths, il: widths[il] - neurons}, tiers)

    def stages(self, target: int = 200 * MB) -> dict[str, float]:
        """The budget in MB: source, source at the kept vocabulary, that at uniform Q4_K, and what is left to remove."""
        real = [g for g in self.groups if not g.startswith("_")]
        q4 = self.size({}, dict.fromkeys(real, "Q4_K"))
        return {"source": os.path.getsize(self.src) / MB, "vocabulary": self.size({}, dict.fromkeys(real, "BF16")) / MB, "q4": q4 / MB, "remove": (q4 - target) / MB}
