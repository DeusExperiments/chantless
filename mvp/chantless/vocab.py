"""Merge-closed Rule-P vocabulary keep-set selection and lossless GGUF vocabulary trimming.

Rows of `token_embd.weight`, `per_layer_token_embd.weight`, and untied `output.weight` are quantized
independently per row in every ggml quantization format, so kept rows can be copied byte for byte
without requantization. Tokenizer arrays (`tokens`, `token_type`, `scores`, `merges`) are subset in
order so BPE merge ranks are preserved, and `tokenizer.ggml.*_token_id` scalars are renumbered.
"""
from __future__ import annotations

import os
import struct
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from . import gguf_ops

_STR = 8
VOCAB_TABLES = ("token_embd.weight", "per_layer_token_embd.weight", "output.weight")
BLOCK_BYTES = {
    "F32": (1, 4), "BF16": (1, 2), "F16": (1, 2), "Q8_0": (32, 34), "Q6_K": (256, 210),
    "Q5_K": (256, 176), "Q4_K": (256, 144), "Q3_K": (256, 110), "Q2_K": (256, 84),
    "IQ2_XS": (256, 74), "IQ1_S": (256, 50),
}


def _str_array(raw: bytes) -> list[bytes]:
    et, n = struct.unpack_from("<IQ", raw, 0)
    if et != _STR:
        raise ValueError(f"expected string array (type 8), got {et}")
    p, out = 12, []
    for _ in range(n):
        (sl,) = struct.unpack_from("<Q", raw, p)
        p += 8
        out.append(raw[p : p + sl])
        p += sl
    return out


def _pack_str_array(items: Sequence[bytes]) -> bytes:
    b = bytearray(struct.pack("<IQ", _STR, len(items)))
    for s in items:
        b += struct.pack("<Q", len(s)) + s
    return bytes(b)


def read_tokenizer(gguf_path: str | os.PathLike) -> dict:
    """Read raw tokenizer arrays from a GGUF header without materializing numpy string objects."""
    h = gguf_ops._Hdr(gguf_path)
    try:
        kvd = {k: (vt, raw) for k, vt, raw in h.read_kvs()}
        tokens = _str_array(kvd["tokenizer.ggml.tokens"][1])
        v = len(tokens)
        ttype = (
            np.frombuffer(kvd["tokenizer.ggml.token_type"][1], "<i4", v, 12).copy()
            if "tokenizer.ggml.token_type" in kvd
            else np.ones(v, dtype=np.int32)
        )
        merges = _str_array(kvd["tokenizer.ggml.merges"][1]) if "tokenizer.ggml.merges" in kvd else []
        has_scores = "tokenizer.ggml.scores" in kvd
        special_ids = []
        for k, (_, raw) in kvd.items():
            if k.startswith("tokenizer.ggml.") and k.endswith("_token_id") and len(raw) == 4:
                (oid,) = struct.unpack("<I", raw)
                if 0 <= oid < v:
                    special_ids.append(int(oid))
        return {
            "vocab_size": v,
            "tokens": tokens,
            "token_type": ttype,
            "merges": merges,
            "has_scores": has_scores,
            "special_ids": sorted(set(special_ids)),
        }
    finally:
        h.close()


def _ascii_punct(tok_bytes: bytes) -> bool:
    s = tok_bytes.replace("▁".encode("utf-8"), b" ").replace("Ġ".encode("utf-8"), b" ")
    try:
        u = s.decode("utf-8")
    except UnicodeDecodeError:
        return False
    return len(u) > 0 and all(ord(c) < 128 and not c.isalnum() for c in u)


def rule_p_keep_set(
    gguf_path: str | os.PathLike,
    counts: np.ndarray,
    k: int,
    must_keep: Sequence[int] = (),
) -> np.ndarray:
    """Merge-closed Rule-P keep set of at most `k` tokens (sorted int64 old token IDs).

    Base set = all non-normal tokens (`token_type != 1`: control, byte, user-defined, unused)
    + all normal ASCII non-alphanumeric tokens + `must_keep` + header `*_token_id` entries.
    Remaining budget is filled by descending occurrence frequency in `counts`, closed under BPE merge parents.
    """
    meta = read_tokenizer(gguf_path)
    tokens, ttype, merges, v = meta["tokens"], meta["token_type"], meta["merges"], meta["vocab_size"]
    if k >= v:
        return np.arange(v, dtype=np.int64)
    counts = np.asarray(counts, dtype=np.int64)
    if len(counts) < v:
        counts = np.pad(counts, (0, v - len(counts)))
    tid = {t: i for i, t in enumerate(tokens)}
    parents: dict[int, tuple[int, int]] = {}
    for r in merges:
        sp = r.find(b" ", 1)
        if sp > 0:
            a, b = r[:sp], r[sp + 1 :]
            ab = a + b
            if ab in tid and a in tid and b in tid:
                parents.setdefault(tid[ab], (tid[a], tid[b]))

    def close(seed_set: set[int]) -> set[int]:
        s = set(seed_set)
        stack = list(s)
        while stack:
            x = stack.pop()
            for p in parents.get(x, ()):
                if p not in s:
                    s.add(p)
                    stack.append(p)
        return s

    base = (
        set(np.flatnonzero(ttype != 1).tolist())
        | {i for i, t in enumerate(tokens) if ttype[i] == 1 and _ascii_punct(t)}
        | set(meta["special_ids"])
        | {int(i) for i in must_keep}
    )
    closed_base = close(base)
    if len(closed_base) > k:
        raise ValueError(f"mandatory merge-closed base vocabulary ({len(closed_base)}) exceeds target K={k}")

    order = np.lexsort((np.arange(v), -counts[:v]))
    lo, hi = 0, v
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if len(close(closed_base | set(order[:mid].tolist()))) <= k:
            lo = mid
        else:
            hi = mid - 1
    return np.array(sorted(close(closed_base | set(order[:lo].tolist()))), dtype=np.int64)


def pick_vocab_k(
    gguf_path: str | os.PathLike,
    counts: np.ndarray,
    candidate_ks: Sequence[int],
    eval_fidelity: Callable[[np.ndarray], float],
    min_fidelity: float = 0.998,
    must_keep: Sequence[int] = (),
) -> tuple[int, np.ndarray, dict[int, float]]:
    """Evaluate candidate vocabulary sizes (ascending) and return the smallest K meeting `min_fidelity`."""
    v = read_tokenizer(gguf_path)["vocab_size"]
    measured: dict[int, float] = {}
    ks = sorted({int(k) for k in candidate_ks if 0 < int(k) < v})
    for k in ks:
        try:
            keep = rule_p_keep_set(gguf_path, counts, k, must_keep=must_keep)
        except ValueError:
            continue
        fid = float(eval_fidelity(keep))
        measured[int(k)] = fid
        if fid >= min_fidelity:
            return int(k), keep, measured
    return v, np.arange(v, dtype=np.int64), measured


def remap_token_ids(keep: np.ndarray, ids: Sequence[int]) -> tuple[int, ...]:
    """Map old token IDs into a sorted `keep` array; raises if any ID was dropped."""
    keep = np.asarray(keep, dtype=np.int64)
    pos = np.searchsorted(keep, ids)
    out = []
    for old_id, p in zip(ids, pos):
        if p >= len(keep) or int(keep[p]) != int(old_id):
            raise ValueError(f"required token id {old_id} is absent from vocab_keep (size {len(keep)})")
        out.append(int(p))
    return tuple(out)


def find_answer_ids(gguf_path: str | os.PathLike, words: Sequence[str], orig_tokens: dict[str, bytes] | None = None) -> tuple[int, ...]:
    """Resolve exact single-token byte representations in a (possibly trimmed) GGUF tokenizer."""
    meta = read_tokenizer(gguf_path)
    idx = {b: i for i, b in enumerate(meta["tokens"])}
    out = []
    for w in words:
        target = orig_tokens[w] if orig_tokens and w in orig_tokens else w.encode("utf-8")
        if target in idx:
            out.append(idx[target])
            continue
        raise KeyError(f"token bytes {target!r} for {w!r} not found in {gguf_path}")
    return tuple(out)


def kv_trim_savings(gguf_path: str | os.PathLike, keep: np.ndarray) -> int:
    """Exact KV-section byte reduction when trimming `gguf_path` to `keep`."""
    meta = read_tokenizer(gguf_path)
    keep = np.asarray(keep, dtype=np.int64)
    v = meta["vocab_size"]
    if len(keep) >= v:
        return 0
    tokens, merges = meta["tokens"], meta["merges"]
    ks = {tokens[int(i)] for i in keep}
    kept_merge_bytes = 0
    for r in merges:
        sp = r.find(b" ", 1)
        if sp > 0 and r[:sp] in ks and r[sp + 1 :] in ks and (r[:sp] + r[sp + 1 :]) in ks:
            kept_merge_bytes += 8 + len(r)
    tok_cut = sum(8 + len(t) for t in tokens) - sum(8 + len(tokens[int(i)]) for i in keep)
    arr_cut = 4 * (v - len(keep)) * (1 + int(meta["has_scores"]))
    merge_cut = (sum(8 + len(r) for r in merges) - kept_merge_bytes) if merges else 0
    return int(tok_cut + arr_cut + merge_cut)


def vocab_trim_byte_savings(
    gguf_path: str | os.PathLike,
    keep: np.ndarray,
    table_tiers: dict[str, str],
) -> int:
    """Exact total file byte reduction (KV + aligned tensor payloads) from trimming to `keep`.

    Note: caller's `ByteModel` handles 32-byte header alignment directly when given `kv_cut`; this helper
    returns the raw KV + row savings matching `gsolve.py` / `bytes200.py` when alignment mod-32 is preserved.
    """
    keep = np.asarray(keep, dtype=np.int64)
    h = gguf_ops._Hdr(gguf_path)
    try:
        ten = {t["name"]: t for t in h.tensors}
    finally:
        h.close()
    if "token_embd.weight" not in ten:
        return 0
    v = int(ten["token_embd.weight"]["dims"][1])
    if len(keep) >= v:
        return 0
    cut = kv_trim_savings(gguf_path, keep)
    rows_removed = v - len(keep)
    for tab in VOCAB_TABLES:
        if tab in ten:
            cols = int(ten[tab]["dims"][0])
            ty = table_tiers.get(tab, ten[tab]["type"]).upper()
            bs, ts = BLOCK_BYTES[ty]
            cut += rows_removed * (ts * (cols // bs))
    return int(cut)


def trim_gguf(src: str | os.PathLike, dst: str | os.PathLike, keep: Sequence[int] | np.ndarray) -> dict:
    """Shrink `src` GGUF to sorted token keep-set `keep` across all vocab tables and tokenizer KV."""
    keep = np.asarray(keep, dtype=np.int64)
    if keep.ndim != 1 or len(keep) == 0 or not (np.diff(keep) > 0).all():
        raise ValueError("keep must be a strictly increasing 1D array of token IDs")
    h = gguf_ops._Hdr(src)
    try:
        kvs = h.read_kvs()
        kvd = {k: (vt, raw) for k, vt, raw in kvs}
        tokens = _str_array(kvd["tokenizer.ggml.tokens"][1])
        v = len(tokens)
        if int(keep[-1]) >= v or int(keep[0]) < 0:
            raise ValueError(f"keep indices out of bounds for vocab size {v}")
        if len(keep) == v:
            if Path(src).resolve() != Path(dst).resolve():
                __import__("shutil").copyfile(src, dst)
            return {"kept_tokens": v, "kept_merges": 0, "file_bytes": os.path.getsize(dst)}

        new_id = np.full(v, -1, dtype=np.int64)
        new_id[keep] = np.arange(len(keep), dtype=np.int64)
        ks = {tokens[int(i)] for i in keep}
        new_tokens = [tokens[int(i)] for i in keep]

        out_kvs = []
        kept_merges = 0
        for k, vt, raw in kvs:
            if k == "tokenizer.ggml.tokens":
                raw = _pack_str_array(new_tokens)
            elif k == "tokenizer.ggml.token_type":
                ttype = np.frombuffer(raw, "<i4", v, 12)[keep]
                raw = struct.pack("<IQ", 5, len(keep)) + np.asarray(ttype, "<i4").tobytes()
            elif k == "tokenizer.ggml.merges":
                merges = _str_array(raw)
                nm = []
                for r in merges:
                    sp = r.find(b" ", 1)
                    if sp > 0 and r[:sp] in ks and r[sp + 1 :] in ks and (r[:sp] + r[sp + 1 :]) in ks:
                        nm.append(r)
                kept_merges = len(nm)
                raw = _pack_str_array(nm)
            elif k == "tokenizer.ggml.scores":
                sc = np.frombuffer(raw, "<f4", v, 12)[keep]
                raw = struct.pack("<IQ", 6, len(keep)) + np.asarray(sc, "<f4").tobytes()
            elif k.startswith("tokenizer.ggml.") and k.endswith("_token_id") and len(raw) == 4:
                (oid,) = struct.unpack("<I", raw)
                nid = int(new_id[oid])
                if nid < 0:
                    raise ValueError(f"{k}={oid} is not in vocab_keep")
                raw = struct.pack("<I", nid)
            out_kvs.append((k, vt, raw))

        align = h.alignment
        tensors = []
        for t in h.tensors:
            d = dict(t)
            if t["name"] in VOCAB_TABLES and t["dims"][1] == v:
                row_bytes = t["n_bytes"] // v
                base = h.data_start + t["offset"]
                rows = np.frombuffer(h.m, np.uint8, t["n_bytes"], base).reshape(v, row_bytes)
                blob = np.ascontiguousarray(rows[keep]).tobytes()
                del rows
                d["dims"] = [t["dims"][0], int(len(keep))]
                d["n_bytes"] = len(blob)
                d["_blob"] = blob
            tensors.append(d)

        ti = bytearray()
        off = 0
        for t in tensors:
            nb = t["name"].encode()
            ti += struct.pack("<Q", len(nb)) + nb + struct.pack("<I", len(t["dims"]))
            ti += struct.pack(f"<{len(t['dims'])}Q", *t["dims"]) + struct.pack("<IQ", t["ttype"], off)
            t["_new"] = off
            off = (off + t["n_bytes"] + align - 1) // align * align

        tmp = str(dst) + ".part"
        with open(tmp, "wb") as f:
            f.write(b"GGUF" + struct.pack("<IQQ", h.version, len(tensors), len(out_kvs)))
            for k, vt, raw in out_kvs:
                kb = k.encode()
                f.write(struct.pack("<Q", len(kb)) + kb + struct.pack("<I", vt) + raw)
            kv_end = f.tell()
            f.write(ti)
            pos = f.tell()
            f.write(b"\0" * ((pos + align - 1) // align * align - pos))
            data_pos = f.tell()
            for t in tensors:
                f.write(b"\0" * (t["_new"] - (f.tell() - data_pos)))
                if "_blob" in t:
                    f.write(t["_blob"])
                else:
                    s = h.data_start + t["offset"]
                    f.write(h.m[s : s + t["n_bytes"]])
        os.replace(tmp, dst)
        return {
            "kept_tokens": int(len(keep)),
            "kept_merges": int(kept_merges),
            "file_bytes": os.path.getsize(dst),
            "kv_bytes": kv_end - 24,
        }
    finally:
        h.close()
