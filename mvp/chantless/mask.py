"""Precision & structural mask schema `chantless.mask/v3`, backward-compatible with `v2` and `onlyskills.mask/v1`.

A `v3` mask is the complete, deterministic record turning the frozen reference base into the Chantless model:
  - `groups` / `tensors`: per-group and per-tensor precision tiers or `drop` (physical sub-block removal)
  - `dead_tensors`: losslessly removed tensors (e.g. dead shared-KV projections)
  - `widths`: per-layer kept MLP neuron bitmaps (`{str(layer): packed_bitmap}`)
  - `vocab_keep`: kept vocabulary token bitmap (`packed_bitmap` or `None` for full vocabulary)
  - `apprentice`: post-build lessons (`switches` dead-neuron bitmaps, optional sparse grid `levels`, `decision` rule)
"""
from __future__ import annotations

import base64
import copy
import json
import zlib
from pathlib import Path
from typing import Sequence

import numpy as np

SCHEMA = "chantless.mask/v3"
SCHEMA_V2 = "chantless.mask/v2"
LEGACY = "onlyskills.mask/v1"

# Cheapening order. BF16 is the source representation; `drop` is physical removal.
LADDER = ["BF16", "Q8_0", "Q6_K", "Q5_K", "Q4_K", "Q3_K", "Q2_K", "IQ2_XS", "IQ1_S"]
DROP = "drop"
IMATRIX_REQUIRED = {"IQ2_XS", "IQ2_XXS", "IQ1_M", "IQ1_S"}
BLOCK_SIZES = {
    "F32": 1, "BF16": 1, "F16": 1, "Q8_0": 32, "Q6_K": 256, "Q5_K": 256,
    "Q4_K": 256, "Q3_K": 256, "Q2_K": 256, "IQ2_XS": 256, "IQ1_S": 256,
}


class MaskError(ValueError):
    pass


def tier_allowed(group_info: dict, tier: str) -> bool:
    """IQ tiers need importance data for every tensor of the group (llama-imatrix collects none for embeddings/output)."""
    return tier == DROP or tier not in IMATRIX_REQUIRED or group_info.get("imatrix_ok", True)


def clip(group_info: dict, tier: str, ladder: list[str] | None = None) -> str:
    """Cheapest allowed tier at or above `tier` for this group."""
    ladder = ladder or LADDER
    i = ladder.index(tier)
    while not tier_allowed(group_info, ladder[i]):
        i -= 1
    return ladder[i]


def rank(tier: str) -> int:
    """Higher rank = cheaper. drop is cheapest."""
    return len(LADDER) if tier == DROP else ladder_index(tier)


def ladder_index(tier: str, ladder: list[str] | None = None) -> int:
    return (ladder or LADDER).index(tier)


# ------------------------------------------------------------------ compact bitmap & sparse-level serialization
def pack_bitmap(indices: Sequence[int] | np.ndarray, total: int, compress: bool = True) -> dict:
    """Pack a subset of `0..total-1` into a self-contained JSON-safe bitmap record."""
    total = int(total)
    if total <= 0:
        raise MaskError(f"bitmap total length must be positive, got {total}")
    idx = np.asarray(indices, dtype=np.int64)
    if idx.ndim != 1:
        raise MaskError("bitmap indices must be 1D")
    if len(idx):
        idx = np.unique(idx)
        if int(idx[0]) < 0 or int(idx[-1]) >= total:
            raise MaskError(f"bitmap index out of range [0, {total})")
    bits = np.zeros(total, dtype=np.uint8)
    if len(idx):
        bits[idx] = 1
    raw = np.packbits(bits).tobytes()
    if compress:
        comp = zlib.compress(raw, level=9)
        if len(comp) < len(raw):
            return {"n": total, "count": int(len(idx)), "enc": "packbits+zlib", "b64": base64.b64encode(comp).decode("ascii")}
    return {"n": total, "count": int(len(idx)), "enc": "packbits", "b64": base64.b64encode(raw).decode("ascii")}


def unpack_bitmap(spec: dict | list[int]) -> np.ndarray:
    """Unpack a `pack_bitmap` dict (or plain integer list) into a sorted `int64` index array."""
    if isinstance(spec, list):
        return np.array(sorted({int(x) for x in spec}), dtype=np.int64)
    if not isinstance(spec, dict) or "n" not in spec or "b64" not in spec:
        raise MaskError(f"invalid bitmap spec: {type(spec)!r}")
    total = int(spec["n"])
    raw = base64.b64decode(spec["b64"])
    enc = spec.get("enc", "packbits")
    if enc == "packbits+zlib":
        raw = zlib.decompress(raw)
    elif enc != "packbits":
        raise MaskError(f"unknown bitmap encoding {enc!r}")
    bits = np.unpackbits(np.frombuffer(raw, dtype=np.uint8))[:total]
    idx = np.flatnonzero(bits).astype(np.int64)
    if "count" in spec and len(idx) != int(spec["count"]):
        raise MaskError(f"bitmap count mismatch: expected {spec['count']}, unpacked {len(idx)}")
    return idx


def pack_sparse_levels(indices: Sequence[int] | np.ndarray, values: Sequence[int] | np.ndarray, n_elements: int) -> dict:
    """Pack sparse quantized grid-level overrides `(flat_index -> uint8_level)`."""
    idx = np.asarray(indices, dtype=np.uint32)
    val = np.asarray(values, dtype=np.uint8)
    if idx.shape != val.shape or idx.ndim != 1:
        raise MaskError("sparse level indices and values must be 1D arrays of equal length")
    if len(idx):
        order = np.argsort(idx)
        idx, val = idx[order], val[order]
        if int(idx[-1]) >= int(n_elements):
            raise MaskError("sparse level index out of bounds")
    blob = zlib.compress(idx.tobytes() + val.tobytes(), level=9)
    return {"n": int(n_elements), "count": int(len(idx)), "b64": base64.b64encode(blob).decode("ascii")}


def unpack_sparse_levels(spec: dict) -> tuple[np.ndarray, np.ndarray]:
    """Return `(flat_indices_int64, levels_uint8)` from a `pack_sparse_levels` dict."""
    c = int(spec["count"])
    if c == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.uint8)
    raw = zlib.decompress(base64.b64decode(spec["b64"]))
    if len(raw) != 5 * c:
        raise MaskError(f"corrupt sparse level payload: {len(raw)} bytes for {c} entries")
    idx = np.frombuffer(raw, dtype="<u4", count=c).astype(np.int64)
    val = np.frombuffer(raw, dtype=np.uint8, count=c, offset=4 * c).copy()
    return idx, val


# ------------------------------------------------------------------ construction, upgrade & validation
def new_mask(groups: dict[str, dict], start_tier: str, source: dict, objective_mode: str = "preserve", schema: str = SCHEMA) -> dict:
    m = {
        "schema": schema,
        "source": source,
        "objective_mode": objective_mode,
        "groups": {g: {"tier": start_tier} for g in groups if not g.startswith("_")},
        "tensors": {},
        "floors": {},
        "weights": {"lineage": []},
        "deployment": {},
        "measured": {},
    }
    if schema == SCHEMA:
        m["dead_tensors"] = []
        m["widths"] = {}
        m["vocab_keep"] = None
        m["apprentice"] = {"switches": {}, "levels": {}, "decision": {}}
    return m


def upgrade_v2(d: dict) -> dict:
    """Return a `chantless.mask/v3` copy of a `v2` or `v3` mask without mutating the input."""
    out = copy.deepcopy(d)
    out["schema"] = SCHEMA
    out.setdefault("dead_tensors", [])
    out.setdefault("widths", {})
    out.setdefault("vocab_keep", None)
    app = out.setdefault("apprentice", {})
    app.setdefault("switches", {})
    app.setdefault("levels", {})
    app.setdefault("decision", {})
    return out


def has_v3_cuts(mask: dict) -> bool:
    """True when `mask` carries any structural/thin-mask transform beyond `v2` group/tensor tiers."""
    app = mask.get("apprentice") or {}
    return bool(
        mask.get("dead_tensors")
        or mask.get("widths")
        or mask.get("vocab_keep") is not None
        or app.get("switches")
        or app.get("levels")
    )


def validate(mask: dict, groups: dict[str, dict], ladder: list[str] | None = None) -> None:
    ladder = ladder or LADDER
    sc = mask.get("schema")
    if sc not in (SCHEMA, SCHEMA_V2):
        raise MaskError(f"schema must be {SCHEMA} or {SCHEMA_V2}, got {sc!r}")
    if mask.get("objective_mode") not in ("preserve", "improve"):
        raise MaskError("objective_mode must be preserve or improve")
    real = {g for g in groups if not g.startswith("_")}
    if set(mask["groups"]) != real:
        raise MaskError(f"mask groups differ from model groups: {sorted(set(mask['groups']) ^ real)[:5]}")
    n_attn = sum(1 for g in real if g.endswith(".attn"))
    dropped_attn = 0
    for g, e in mask["groups"].items():
        t = e["tier"]
        if t == DROP:
            if not groups[g]["droppable"]:
                raise MaskError(f"{g} cannot be dropped")
            dropped_attn += g.endswith(".attn")
        elif t not in ladder:
            raise MaskError(f"{g}: unknown tier {t}")
        elif not tier_allowed(groups[g], t):
            raise MaskError(f"{g}: tier {t} needs importance data this group lacks")
        fl = mask.get("floors", {}).get(g)
        if fl and (len(ladder) if t == DROP else ladder.index(t)) > ladder.index(fl["min_tier"]) and not (t == DROP and fl.get("drop_ok")):
            raise MaskError(f"{g}: tier {t} below sanity floor {fl['min_tier']}")
    if n_attn > 0 and dropped_attn >= n_attn:
        raise MaskError("at least one attention block must remain")
    for name, t in mask.get("tensors", {}).items():
        if t not in ladder:
            raise MaskError(f"tensor {name}: unknown tier {t}")

    for k, spec in (mask.get("widths") or {}).items():
        il = int(k)
        fg = f"blk.{il}.ffn"
        if fg not in mask["groups"]:
            raise MaskError(f"width specified for unknown FFN group {fg}")
        idx = unpack_bitmap(spec)
        t = mask["groups"][fg]["tier"]
        if t != DROP:
            if len(idx) == 0:
                raise MaskError(f"{fg}: width 0 requires tier='drop'")
            down_tier = mask.get("tensors", {}).get(f"blk.{il}.ffn_down.weight", t).upper()
            bs = BLOCK_SIZES.get(down_tier, 1)
            if len(idx) % bs != 0:
                raise MaskError(f"{fg}: kept width {len(idx)} is not divisible by {down_tier} block size {bs}")

    if mask.get("vocab_keep") is not None:
        vk = unpack_bitmap(mask["vocab_keep"])
        if len(vk) == 0:
            raise MaskError("vocab_keep cannot be empty")

    app = mask.get("apprentice") or {}
    for k, spec in (app.get("switches") or {}).items():
        _ = int(k)
        unpack_bitmap(spec)
    for name, spec in (app.get("levels") or {}).items():
        unpack_sparse_levels(spec)


def tensor_plan(mask: dict, groups: dict[str, dict]) -> tuple[dict[str, str], set[str]]:
    """-> ({quantizable tensor: tier}, {tensors to remove including dead_tensors})."""
    types: dict[str, str] = {}
    drop: set[str] = set(mask.get("dead_tensors") or ())
    for g, e in mask["groups"].items():
        if e["tier"] == DROP:
            drop.update(groups[g]["quantizable"] + groups[g]["fixed"])
        else:
            for n in groups[g]["quantizable"]:
                if n not in drop:
                    types[n] = mask.get("tensors", {}).get(n, e["tier"])
    return types, drop


def kept_widths(mask: dict) -> dict[int, np.ndarray]:
    """Return `{layer_idx: sorted_kept_neuron_indices}` for layers with width cuts (excluding dropped FFNs)."""
    out: dict[int, np.ndarray] = {}
    for k, spec in (mask.get("widths") or {}).items():
        il = int(k)
        if mask["groups"].get(f"blk.{il}.ffn", {}).get("tier") != DROP:
            out[il] = unpack_bitmap(spec)
    return out


def load(path: str | Path) -> dict:
    d = json.loads(Path(path).read_text())
    if d.get("schema") in (SCHEMA, SCHEMA_V2):
        return d
    if d.get("schema") == LEGACY:
        return from_v1(d)
    raise MaskError(f"unknown mask schema {d.get('schema')!r}")


def from_v1(d: dict) -> dict:
    """Read a legacy v1 mask (manual §11.1) into v2 form without touching the original file."""
    out = {
        "schema": SCHEMA_V2, "source": d.get("source", {}), "objective_mode": "preserve", "groups": {},
        "tensors": {}, "floors": {}, "weights": {"lineage": []}, "deployment": {"kv_type": d.get("kv_type")},
        "measured": d.get("measured", {}), "legacy": {"schema": LEGACY, "removed": d.get("removed", [])},
    }
    for g, e in d.get("groups", {}).items():
        if e.get("tier") != DROP and float(e.get("keep", 1.0)) != 1.0:
            raise MaskError(f"{g}: keep-fraction pruning is not supported by this builder")
        name = g.replace(".mixer", ".attn")
        out["groups"][name] = {"tier": e["tier"]}
    return out


def apply(mask: dict, group: str, tier: str) -> dict:
    m = copy.deepcopy(mask)
    m["groups"][group]["tier"] = tier
    return m
