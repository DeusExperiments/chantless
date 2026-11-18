"""The grid a quantised tensor lives on (mask-plus-weights, decision S3).

A Q4_K or Q8_0 tensor is a fixed grid: value = lo + step * level, with lo and step fixed per sub-block and
only the integer levels free. Learning moves levels and leaves lo and step alone, so training simulates the
shipped weights exactly, a level change cannot be erased by rounding, and writing the result back changes
tensor bytes only: same header, same offsets, same file size. Other tiers stay frozen.
"""
from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import gguf_ops

Q4_BLOCK, Q4_BYTES = 256, 144
Q8_BLOCK, Q8_BYTES = 32, 34
TOP = {"Q4_K": 15, "Q8_0": 255}


def _half(b: np.ndarray) -> np.ndarray:
    return np.frombuffer(b.tobytes(), dtype="<f2").astype(np.float32)


@dataclass
class Grid:
    levels: np.ndarray    # uint8, one per weight, row order
    lo: np.ndarray        # float32, one per weight
    step: np.ndarray      # float32, one per weight
    top: int

    def values(self) -> np.ndarray:
        return self.lo + self.step * self.levels.astype(np.float32)


def _q4_scales(s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """6-bit scale and min of the 8 sub-blocks, from the 12 packed bytes; same order as get_scale_min_k4."""
    sc = np.zeros((len(s), 8), np.float32)
    mn = np.zeros((len(s), 8), np.float32)
    for j in range(4):
        sc[:, j] = s[:, j] & 63
        mn[:, j] = s[:, j + 4] & 63
    for j in range(4, 8):
        sc[:, j] = (s[:, j + 4] & 0xF) | ((s[:, j - 4] >> 6) << 4)
        mn[:, j] = (s[:, j + 4] >> 4) | ((s[:, j] >> 6) << 4)
    return sc, mn


def decode(raw: bytes | memoryview, kind: str) -> Grid:
    a = np.frombuffer(raw, dtype=np.uint8)
    if kind == "Q4_K":
        b = a.reshape(-1, Q4_BYTES)
        d, dmin = _half(b[:, :2]), _half(b[:, 2:4])
        sc, mn = _q4_scales(b[:, 4:16])
        q = b[:, 16:].reshape(-1, 4, 32)
        lv = np.stack([q & 0xF, q >> 4], axis=2).reshape(-1, 8, 32)     # pairs of sub-blocks share 32 bytes
        step = np.repeat(sc * d[:, None], 32, axis=1).reshape(-1)
        lo = -np.repeat(mn * dmin[:, None], 32, axis=1).reshape(-1)
        return Grid(lv.reshape(-1).astype(np.uint8), lo, step, 15)
    if kind == "Q8_0":
        b = a.reshape(-1, Q8_BYTES)
        d = _half(b[:, :2])
        lv = (b[:, 2:].astype(np.int16) ^ 0x80).astype(np.uint8)        # int8 q -> level q + 128
        step = np.repeat(d, Q8_BLOCK)
        return Grid(lv.reshape(-1), -128 * step, step, 255)
    raise ValueError(f"no grid for {kind}")


def encode(raw: bytes | memoryview, kind: str, levels: np.ndarray) -> bytes:
    """`raw` with its levels replaced; every scale byte is kept."""
    b = np.frombuffer(raw, dtype=np.uint8).copy()
    if kind == "Q4_K":
        b = b.reshape(-1, Q4_BYTES)
        lv = levels.reshape(-1, 4, 2, 32)
        b[:, 16:] = (lv[:, :, 0] | (lv[:, :, 1] << 4)).reshape(-1, 128)
    elif kind == "Q8_0":
        b = b.reshape(-1, Q8_BYTES)
        b[:, 2:] = (levels.reshape(-1, Q8_BLOCK) ^ 0x80)
    else:
        raise ValueError(f"no grid for {kind}")
    return b.tobytes()


class Artifact:
    """A built GGUF read tensor by tensor."""

    def __init__(self, path: str | os.PathLike):
        self.path = str(path)
        self.h = gguf_ops._Hdr(self.path)
        self.tensors = {t["name"]: t for t in self.h.tensors}

    def raw(self, name: str) -> memoryview:
        t = self.tensors[name]
        s = self.h.data_start + t["offset"]
        return memoryview(self.h.m)[s:s + t["n_bytes"]]

    def grid(self, name: str) -> Grid:
        return decode(self.raw(name), self.tensors[name]["type"])

    def shape(self, name: str) -> tuple[int, ...]:
        return tuple(reversed(self.tensors[name]["dims"]))    # torch order: rows first

    def close(self) -> None:
        self.h.close()


def write_levels(src: str | os.PathLike, dst: str | os.PathLike, levels: dict[str, np.ndarray]) -> None:
    """Copy `src` to `dst` with new levels in the named tensors. Only tensor bytes change."""
    art = Artifact(src)
    tmp = str(dst) + ".part"
    shutil.copyfile(src, tmp)
    with open(tmp, "r+b") as f:
        for name, lv in levels.items():
            t = art.tensors[name]
            f.seek(art.h.data_start + t["offset"])
            f.write(encode(art.raw(name), t["type"], lv))
    art.close()
    os.replace(tmp, dst)
    assert Path(src).stat().st_size == Path(dst).stat().st_size


def zero_rows(src: str | os.PathLike, dst: str | os.PathLike, row_indices: dict[str, np.ndarray]) -> None:
    """Zero whole rows `row_indices[name]` of 2D GGUF tensors across ANY ggml quant type (`IQ1_S`..`BF16`).

    For `ffn_up.weight` and `ffn_gate.weight` (`dims = [n_embd, n_ff]`), `n_embd` is a multiple of 256 so each
    neuron row `j` owns an integer number of unshared quantization super-blocks; writing `0x00` bytes sets the
    fp16 block scale `d = 0.0`, which dequantizes to exact `0.0f` in every ggml kernel without adding 1 byte.
    """
    art = Artifact(src)
    tmp = str(dst) + ".part"
    shutil.copyfile(src, tmp)
    try:
        with open(tmp, "r+b") as f:
            for name, rows in row_indices.items():
                idx = np.unique(np.asarray(rows, dtype=np.int64))
                if len(idx) == 0:
                    continue
                t = art.tensors[name]
                if len(t["dims"]) != 2:
                    raise ValueError(f"zero_rows expects a 2D tensor, got {name} dims={t['dims']}")
                n_rows = int(t["dims"][1])
                if t["n_bytes"] % n_rows != 0:
                    raise ValueError(f"{name}: n_bytes={t['n_bytes']} not divisible by n_rows={n_rows}")
                if int(idx[0]) < 0 or int(idx[-1]) >= n_rows:
                    raise ValueError(f"{name}: row index out of range [0, {n_rows})")
                row_bytes = t["n_bytes"] // n_rows
                buf = np.frombuffer(bytes(art.raw(name)), dtype=np.uint8).reshape(n_rows, row_bytes).copy()
                buf[idx] = 0
                f.seek(art.h.data_start + t["offset"])
                f.write(buf.tobytes())
    finally:
        art.close()
    os.replace(tmp, dst)
    assert Path(src).stat().st_size == Path(dst).stat().st_size
