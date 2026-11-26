"""Precision & structural mask schema `chantless.mask/v3` (reads `v2` and legacy v1 masks).

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
ALL_TIERS = [
    "F32", "BF16", "F16",
    "Q8_0", "Q6_K",
    "Q5_K", "Q5_1", "Q5_0",
    "Q4_K", "Q4_1", "Q4_0", "IQ4_NL", "IQ4_XS",
    "Q3_K", "IQ3_S", "IQ3_XXS",
    "Q2_K", "IQ2_S", "IQ2_XS", "IQ2_XXS",
    "IQ1_M", "IQ1_S",
]
VALID_GGML_TYPES = frozenset(ALL_TIERS)
DROP = "drop"
IMATRIX_REQUIRED = {"IQ2_S", "IQ2_XS", "IQ2_XXS", "IQ1_M", "IQ1_S"}
BLOCK_SIZES = {
    "F32": 1, "BF16": 1, "F16": 1,
    "Q8_0": 32, "Q6_K": 256,
    "Q5_K": 256, "Q5_1": 32, "Q5_0": 32,
    "Q4_K": 256, "Q4_1": 32, "Q4_0": 32, "IQ4_NL": 32, "IQ4_XS": 256,
    "Q3_K": 256, "IQ3_S": 256, "IQ3_XXS": 256,
    "Q2_K": 256, "IQ2_S": 256, "IQ2_XS": 256, "IQ2_XXS": 256,
    "IQ1_M": 256, "IQ1_S": 256,
}


class MaskError(ValueError):
    pass


def normalize_tier(tier: str) -> str:
    s = str(tier).strip()
    if s.lower() == DROP:
        return DROP
    u = s.upper()
    if u not in VALID_GGML_TYPES:
        raise MaskError(f"unknown ggml tier {tier!r}")
    return u


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
    if tier == DROP:
        return len(ALL_TIERS)
    u = tier.upper()
    if u in ALL_TIERS:
        return ALL_TIERS.index(u)
    raise MaskError(f"unknown tier {tier!r}")


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
        or mask.get("experts")
        or mask.get("vocab_keep") is not None
        or app.get("switches")
        or app.get("levels")
    )


def validate(mask: dict, groups: dict[str, dict], ladder: list[str] | None = None) -> None:
    allowed = VALID_GGML_TYPES if ladder is None else (set(ladder) | VALID_GGML_TYPES)
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
        elif t not in allowed:
            raise MaskError(f"{g}: unknown tier {t}")
        elif not tier_allowed(groups[g], t):
            raise MaskError(f"{g}: tier {t} needs importance data this group lacks")
        fl = mask.get("floors", {}).get(g)
        if fl and rank(t) > rank(fl["min_tier"]) and not (t == DROP and fl.get("drop_ok")):
            raise MaskError(f"{g}: tier {t} below sanity floor {fl['min_tier']}")
    if n_attn > 0 and dropped_attn >= n_attn:
        raise MaskError("at least one attention block must remain")
    for name, t in mask.get("tensors", {}).items():
        if t not in allowed:
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
            tensors_m = mask.get("tensors", {})
            down_tier = (tensors_m.get(f"blk.{il}.ffn_down_exps.weight") or tensors_m.get(f"blk.{il}.ffn_down.weight") or t).upper()
            bs = BLOCK_SIZES.get(down_tier, 1)
            if len(idx) % bs != 0:
                raise MaskError(f"{fg}: kept width {len(idx)} is not divisible by {down_tier} block size {bs}")

    if mask.get("experts"):
        exp_counts: set[int] = set()
        shrunk = False
        exp_layers: set[int] = set()
        for k, spec in (mask.get("experts") or {}).items():
            il = int(k)
            fg = f"blk.{il}.ffn"
            if fg not in mask["groups"]:
                raise MaskError(f"experts specified for unknown FFN group {fg}")
            idx = unpack_bitmap(spec)
            if mask["groups"][fg]["tier"] != DROP:
                if len(idx) == 0:
                    raise MaskError(f"{fg}: kept expert count cannot be 0")
                exp_counts.add(len(idx))
                exp_layers.add(il)
                if isinstance(spec, dict) and "n" in spec and len(idx) < int(spec["n"]):
                    shrunk = True
        if len(exp_counts) > 1:
            raise MaskError(f"MoE experts must keep a uniform expert count across layers, got {sorted(exp_counts)}")
        if shrunk:
            moe_layers = {
                int(g.split(".")[1])
                for g, info in groups.items()
                if g.endswith(".ffn")
                and mask["groups"].get(g, {}).get("tier") != DROP
                and any(n.endswith("ffn_down_exps.weight") for n in info.get("quantizable", ()))
            }
            missing_layers = sorted(moe_layers - exp_layers)
            if missing_layers:
                raise MaskError(f"uniform MoE expert pruning requires all MoE layers in `experts`; missing layers {missing_layers[:5]}")

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


def kept_experts(mask: dict) -> dict[int, np.ndarray]:
    """Return `{layer_idx: sorted_kept_expert_indices}` for layers with MoE expert pruning."""
    out: dict[int, np.ndarray] = {}
    for k, spec in (mask.get("experts") or {}).items():
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
    """Convert a legacy v1 mask dict into `chantless.mask/v2` format without mutating the input."""
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


def _mask_sha256(path: str | Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _source_key(m: dict) -> tuple[str | None, str | None, str | None]:
    s = m.get("source") or {}
    return (s.get("gguf_sha256"), s.get("repo"), s.get("revision"))


def _assert_same_source(a: dict, b: dict) -> None:
    sa, sb = _source_key(a), _source_key(b)
    if sa[0] and sb[0] and sa[0] != sb[0]:
        raise MaskError(f"source gguf_sha256 mismatch: {sa[0]} vs {sb[0]}")
    if sa[1] and sb[1] and (sa[1] != sb[1] or sa[2] != sb[2]):
        raise MaskError(f"source repo/revision mismatch: {sa[1]}@{sa[2]} vs {sb[1]}@{sb[2]}")
    if set(a.get("groups") or {}) != set(b.get("groups") or {}):
        raise MaskError("mask group keysets differ (different architecture or layer count)")


def inspect_mask(path_or_mask: str | Path | dict) -> dict:
    file_bytes = None
    file_sha256 = None
    if isinstance(path_or_mask, dict):
        m = upgrade_v2(path_or_mask)
    else:
        p = Path(path_or_mask)
        file_bytes = p.stat().st_size
        file_sha256 = _mask_sha256(p)
        m = upgrade_v2(load(p))

    tier_counts: dict[str, int] = {}
    dropped_groups: list[str] = []
    for g, e in sorted((m.get("groups") or {}).items()):
        t = e["tier"]
        tier_counts[t] = tier_counts.get(t, 0) + 1
        if t == DROP:
            dropped_groups.append(g)

    widths = m.get("widths") or {}
    width_layers = 0
    width_kept_total = 0
    width_orig_total = 0
    for k, spec in widths.items():
        if (m.get("groups") or {}).get(f"blk.{int(k)}.ffn", {}).get("tier") == DROP:
            continue
        idx = unpack_bitmap(spec)
        width_layers += 1
        width_kept_total += len(idx)
        width_orig_total += int(spec["n"]) if isinstance(spec, dict) and "n" in spec else len(idx)

    experts = m.get("experts") or {}
    exp_layers = 0
    exp_kept_total = 0
    exp_orig_total = 0
    exp_kept_per_layer: set[int] = set()
    exp_orig_per_layer: set[int] = set()
    for k, spec in experts.items():
        if (m.get("groups") or {}).get(f"blk.{int(k)}.ffn", {}).get("tier") == DROP:
            continue
        idx = unpack_bitmap(spec)
        n_orig = int(spec["n"]) if isinstance(spec, dict) and "n" in spec else len(idx)
        exp_layers += 1
        exp_kept_total += len(idx)
        exp_orig_total += n_orig
        exp_kept_per_layer.add(len(idx))
        exp_orig_per_layer.add(n_orig)

    vk = m.get("vocab_keep")
    vocab_info = None
    if vk is not None:
        vidx = unpack_bitmap(vk)
        vn = int(vk["n"]) if isinstance(vk, dict) and "n" in vk else len(vidx)
        vocab_info = {"kept": len(vidx), "total": vn, "pct_kept": round(100.0 * len(vidx) / max(1, vn), 2)}

    app = m.get("apprentice") or {}
    sw = app.get("switches") or {}
    sw_off = 0
    sw_total = 0
    for spec in sw.values():
        idx = unpack_bitmap(spec)
        sw_off += len(idx)
        sw_total += int(spec["n"]) if isinstance(spec, dict) and "n" in spec else len(idx)

    return {
        "schema": m.get("schema"),
        "mask_file_bytes": file_bytes,
        "mask_sha256": file_sha256,
        "source": m.get("source") or {},
        "objective_mode": m.get("objective_mode"),
        "weights_lineage_len": len((m.get("weights") or {}).get("lineage") or []),
        "groups_total": len(m.get("groups") or {}),
        "tier_counts": tier_counts,
        "dropped_groups": dropped_groups,
        "tensor_overrides": len(m.get("tensors") or {}),
        "dead_tensors": len(m.get("dead_tensors") or []),
        "widths": {
            "cut_layers": width_layers,
            "kept_neurons": width_kept_total,
            "orig_neurons": width_orig_total,
            "pct_kept": round(100.0 * width_kept_total / max(1, width_orig_total), 2) if width_orig_total else 100.0,
        },
        "experts": {
            "pruned_layers": exp_layers,
            "kept_experts": exp_kept_total,
            "orig_experts": exp_orig_total,
            "kept_per_layer": next(iter(exp_kept_per_layer)) if len(exp_kept_per_layer) == 1 else (sorted(exp_kept_per_layer) if exp_kept_per_layer else None),
            "orig_per_layer": next(iter(exp_orig_per_layer)) if len(exp_orig_per_layer) == 1 else (sorted(exp_orig_per_layer) if exp_orig_per_layer else None),
            "pct_kept": round(100.0 * exp_kept_total / max(1, exp_orig_total), 2) if exp_orig_total else 100.0,
        },
        "vocab_keep": vocab_info,
        "apprentice": {
            "switch_mode": app.get("switch_mode"),
            "switch_layers": len(sw),
            "switches_off": sw_off,
            "switches_total": sw_total,
            "sparse_level_tensors": len(app.get("levels") or {}),
            "has_decision": bool(app.get("decision")),
        },
        "measured": m.get("measured") or {},
        "provenance": m.get("provenance"),
    }


def diff_masks(a_in: str | Path | dict, b_in: str | Path | dict) -> dict:
    a = upgrade_v2(a_in if isinstance(a_in, dict) else load(a_in))
    b = upgrade_v2(b_in if isinstance(b_in, dict) else load(b_in))

    ga, gb = a.get("groups") or {}, b.get("groups") or {}
    group_diffs = {}
    for g in sorted(set(ga) | set(gb)):
        ta = ga.get(g, {}).get("tier")
        tb = gb.get(g, {}).get("tier")
        if ta != tb:
            group_diffs[g] = {"a": ta, "b": tb}

    ta_m, tb_m = a.get("tensors") or {}, b.get("tensors") or {}
    tensor_diffs = {}
    for t in sorted(set(ta_m) | set(tb_m)):
        if ta_m.get(t) != tb_m.get(t):
            tensor_diffs[t] = {"a": ta_m.get(t), "b": tb_m.get(t)}

    da, db = set(a.get("dead_tensors") or []), set(b.get("dead_tensors") or [])
    wa, wb = a.get("widths") or {}, b.get("widths") or {}
    width_diffs = {}
    for k in sorted(set(wa) | set(wb), key=lambda x: int(x)):
        ia = unpack_bitmap(wa[k]) if k in wa else None
        ib = unpack_bitmap(wb[k]) if k in wb else None
        if ia is None or ib is None or not np.array_equal(ia, ib):
            width_diffs[k] = {
                "a_kept": int(len(ia)) if ia is not None else None,
                "b_kept": int(len(ib)) if ib is not None else None,
                "only_a": int(len(np.setdiff1d(ia, ib))) if (ia is not None and ib is not None) else None,
                "only_b": int(len(np.setdiff1d(ib, ia))) if (ia is not None and ib is not None) else None,
            }

    ea, eb = a.get("experts") or {}, b.get("experts") or {}
    expert_diffs = {}
    for k in sorted(set(ea) | set(eb), key=lambda x: int(x)):
        ia = unpack_bitmap(ea[k]) if k in ea else None
        ib = unpack_bitmap(eb[k]) if k in eb else None
        if ia is None or ib is None or not np.array_equal(ia, ib):
            expert_diffs[k] = {
                "a_kept": int(len(ia)) if ia is not None else None,
                "b_kept": int(len(ib)) if ib is not None else None,
                "only_a": int(len(np.setdiff1d(ia, ib))) if (ia is not None and ib is not None) else None,
                "only_b": int(len(np.setdiff1d(ib, ia))) if (ia is not None and ib is not None) else None,
            }

    vka, vkb = a.get("vocab_keep"), b.get("vocab_keep")
    vocab_diff = None
    if (vka is None) != (vkb is None):
        vocab_diff = {
            "a_kept": int(len(unpack_bitmap(vka))) if vka is not None else None,
            "b_kept": int(len(unpack_bitmap(vkb))) if vkb is not None else None,
        }
    elif vka is not None and vkb is not None:
        ia, ib = unpack_bitmap(vka), unpack_bitmap(vkb)
        if not np.array_equal(ia, ib):
            vocab_diff = {
                "a_kept": int(len(ia)),
                "b_kept": int(len(ib)),
                "only_a": int(len(np.setdiff1d(ia, ib))),
                "only_b": int(len(np.setdiff1d(ib, ia))),
            }

    swa = (a.get("apprentice") or {}).get("switches") or {}
    swb = (b.get("apprentice") or {}).get("switches") or {}
    switch_diffs = {}
    total_sw_only_a = 0
    total_sw_only_b = 0
    for k in sorted(set(swa) | set(swb), key=lambda x: int(x)):
        ia = unpack_bitmap(swa[k]) if k in swa else np.empty(0, dtype=np.int64)
        ib = unpack_bitmap(swb[k]) if k in swb else np.empty(0, dtype=np.int64)
        if not np.array_equal(ia, ib):
            oa = int(len(np.setdiff1d(ia, ib)))
            ob = int(len(np.setdiff1d(ib, ia)))
            total_sw_only_a += oa
            total_sw_only_b += ob
            switch_diffs[k] = {"a_off": int(len(ia)), "b_off": int(len(ib)), "only_a": oa, "only_b": ob}

    return {
        "same_source": _source_key(a) == _source_key(b),
        "source_a": a.get("source") or {},
        "source_b": b.get("source") or {},
        "group_diffs": group_diffs,
        "tensor_diffs": tensor_diffs,
        "dead_tensors_only_a": sorted(da - db),
        "dead_tensors_only_b": sorted(db - da),
        "width_diffs": width_diffs,
        "expert_diffs": expert_diffs,
        "vocab_diff": vocab_diff,
        "switch_diffs": switch_diffs,
        "switches_only_a_total": total_sw_only_a,
        "switches_only_b_total": total_sw_only_b,
        "identical_structure": not (group_diffs or tensor_diffs or (da ^ db) or width_diffs or expert_diffs or vocab_diff or switch_diffs),
    }


def fork_mask(base_path: str | Path, out_path: str | Path, note: str | None = None) -> dict:
    bp = Path(base_path)
    m = upgrade_v2(load(bp))
    parent_sha = _mask_sha256(bp)
    prov = {
        "op": "fork",
        "parent_sha256": parent_sha,
        "parent_name": bp.name,
    }
    if note:
        prov["note"] = note
    m["provenance"] = prov
    op = Path(out_path)
    op.parent.mkdir(parents=True, exist_ok=True)
    tmp = op.with_suffix(op.suffix + ".tmp")
    tmp.write_text(json.dumps(m, indent=2) + "\n", encoding="utf-8")
    tmp.replace(op)
    return {"out": str(op), "parent_sha256": parent_sha, "out_sha256": _mask_sha256(op)}


def _higher_precision(ta: str, tb: str) -> str:
    return ta if rank(ta) <= rank(tb) else tb


def merge_masks(
    a_path: str | Path,
    b_path: str | Path,
    out_path: str | Path,
    mode: str = "apprentice",
    op: str = "union",
) -> dict:
    if mode not in ("apprentice", "specialist"):
        raise MaskError(f"merge mode must be 'apprentice' or 'specialist', got {mode!r}")
    if op not in ("union", "intersection"):
        raise MaskError(f"merge op must be 'union' or 'intersection', got {op!r}")

    ap, bp = Path(a_path), Path(b_path)
    a, b = upgrade_v2(load(ap)), upgrade_v2(load(bp))
    _assert_same_source(a, b)
    sha_a, sha_b = _mask_sha256(ap), _mask_sha256(bp)

    if mode == "apprentice":
        d = diff_masks(a, b)
        if d["group_diffs"] or d["tensor_diffs"] or d["dead_tensors_only_a"] or d["dead_tensors_only_b"] or d["width_diffs"] or d["expert_diffs"] or d["vocab_diff"]:
            raise MaskError(
                "apprentice merge requires identical specialist skeletons "
                f"(group_diffs={len(d['group_diffs'])}, tensor_diffs={len(d['tensor_diffs'])}, "
                f"width_diffs={len(d['width_diffs'])}, expert_diffs={len(d['expert_diffs'])}, vocab_diff={d['vocab_diff'] is not None})"
            )
        app_a, app_b = a.get("apprentice") or {}, b.get("apprentice") or {}
        sm_a, sm_b = app_a.get("switch_mode"), app_b.get("switch_mode")
        if sm_a and sm_b and sm_a != sm_b:
            raise MaskError(f"incompatible switch_mode: {sm_a!r} vs {sm_b!r}")
        if (app_a.get("levels") or {}) or (app_b.get("levels") or {}):
            raise MaskError("apprentice merge supports binary switches only (non-empty sparse levels found)")

        merged = copy.deepcopy(a)
        swa, swb = app_a.get("switches") or {}, app_b.get("switches") or {}
        new_sw: dict[str, dict] = {}
        total_off = 0
        for k in sorted(set(swa) | set(swb), key=lambda x: int(x)):
            sa = swa.get(k)
            sb = swb.get(k)
            n = int(sa["n"]) if isinstance(sa, dict) and "n" in sa else int(sb["n"])
            if isinstance(sa, dict) and isinstance(sb, dict) and int(sa["n"]) != int(sb["n"]):
                raise MaskError(f"layer {k}: switch bitmap length mismatch ({sa['n']} vs {sb['n']})")
            ia = unpack_bitmap(sa) if sa is not None else np.empty(0, dtype=np.int64)
            ib = unpack_bitmap(sb) if sb is not None else np.empty(0, dtype=np.int64)
            comb = np.union1d(ia, ib) if op == "union" else np.intersect1d(ia, ib)
            if len(comb):
                new_sw[k] = pack_bitmap(comb, n)
                total_off += len(comb)
        merged["apprentice"] = {
            "switch_mode": sm_a or sm_b or "up_row_zero",
            "switches": new_sw,
            "levels": {},
            "decision": {},
        }
        meas = dict(merged.get("measured") or {})
        merged["measured"] = {k: meas[k] for k in ("file_bytes", "pct_of_bf16", "bpw") if k in meas}
        merged["provenance"] = {
            "op": f"merge:apprentice:{op}",
            "parents": [{"name": ap.name, "sha256": sha_a}, {"name": bp.name, "sha256": sha_b}],
        }
        stats_out = {"mode": mode, "op": op, "switch_layers": len(new_sw), "switches_off": total_off}
    else:
        merged = copy.deepcopy(a)
        ga, gb = a.get("groups") or {}, b.get("groups") or {}
        for g in sorted(ga):
            merged["groups"][g] = {"tier": _higher_precision(ga[g]["tier"], gb[g]["tier"])}

        ta_m, tb_m = a.get("tensors") or {}, b.get("tensors") or {}
        new_tensors = {}
        for t in sorted(set(ta_m) | set(tb_m)):
            if t in ta_m and t in tb_m:
                new_tensors[t] = _higher_precision(ta_m[t], tb_m[t])
            else:
                new_tensors[t] = ta_m.get(t, tb_m.get(t))
        merged["tensors"] = new_tensors

        merged["dead_tensors"] = sorted(set(a.get("dead_tensors") or []) & set(b.get("dead_tensors") or []))

        wa, wb = a.get("widths") or {}, b.get("widths") or {}
        new_widths = {}
        widths_kept_identical = 0
        widths_reset_full = 0
        for k in sorted(set(wa) | set(wb), key=lambda x: int(x)):
            il = int(k)
            if merged["groups"].get(f"blk.{il}.ffn", {}).get("tier") == DROP:
                continue
            if k in wa and k in wb:
                ia, ib = unpack_bitmap(wa[k]), unpack_bitmap(wb[k])
                if np.array_equal(ia, ib):
                    new_widths[k] = copy.deepcopy(wa[k])
                    widths_kept_identical += 1
                else:
                    widths_reset_full += 1
            else:
                widths_reset_full += 1
        merged["widths"] = new_widths

        ea, eb = a.get("experts") or {}, b.get("experts") or {}
        experts_preserved = False
        experts_reset_full = False
        if ea or eb:
            active_exp_keys = [
                k for k in sorted(set(ea) | set(eb), key=lambda x: int(x))
                if merged["groups"].get(f"blk.{int(k)}.ffn", {}).get("tier") != DROP
            ]
            all_match = bool(ea) and bool(eb) and bool(active_exp_keys) and all(
                (k in ea and k in eb and np.array_equal(unpack_bitmap(ea[k]), unpack_bitmap(eb[k])))
                for k in active_exp_keys
            )
            if all_match:
                merged["experts"] = {k: copy.deepcopy(ea[k]) for k in active_exp_keys}
                experts_preserved = True
            else:
                merged.pop("experts", None)
                experts_reset_full = True
        else:
            merged.pop("experts", None)

        vka, vkb = a.get("vocab_keep"), b.get("vocab_keep")
        if vka is None or vkb is None:
            merged["vocab_keep"] = None
            vocab_kept_n = None
        else:
            na = int(vka["n"]) if isinstance(vka, dict) else None
            nb = int(vkb["n"]) if isinstance(vkb, dict) else None
            if na is None or nb is None or na != nb:
                raise MaskError(f"vocab_keep total size mismatch: {na} vs {nb}")
            comb_v = np.union1d(unpack_bitmap(vka), unpack_bitmap(vkb))
            if len(comb_v) >= na:
                merged["vocab_keep"] = None
                vocab_kept_n = na
            else:
                merged["vocab_keep"] = pack_bitmap(comb_v, na)
                vocab_kept_n = int(len(comb_v))

        merged["apprentice"] = {"switches": {}, "levels": {}, "decision": {}}
        merged["measured"] = {}
        merged["provenance"] = {
            "op": "merge:specialist",
            "parents": [{"name": ap.name, "sha256": sha_a}, {"name": bp.name, "sha256": sha_b}],
        }
        stats_out = {
            "mode": mode,
            "widths_kept_identical": widths_kept_identical,
            "widths_reset_full": widths_reset_full,
            "experts_preserved": experts_preserved,
            "experts_reset_full": experts_reset_full,
            "vocab_kept": vocab_kept_n,
            "dead_tensors": len(merged["dead_tensors"]),
        }

    op_p = Path(out_path)
    op_p.parent.mkdir(parents=True, exist_ok=True)
    tmp = op_p.with_suffix(op_p.suffix + ".tmp")
    tmp.write_text(json.dumps(merged, indent=2) + "\n", encoding="utf-8")
    tmp.replace(op_p)
    return {"out": str(op_p), "out_sha256": _mask_sha256(op_p), **stats_out}


def _resolve_pull_url(uri: str, allow_http: bool = False) -> str:
    if uri.startswith("hf://"):
        rest = uri[len("hf://"):]
        rev = "main"
        if "@" in rest:
            rest, rev = rest.rsplit("@", 1)
        parts = rest.split("/", 2)
        if len(parts) < 3 or not parts[2]:
            raise MaskError(f"invalid hf:// URI {uri!r}; expected hf://<org>/<repo>/<path>[@<rev>]")
        org, repo, subpath = parts
        return f"https://huggingface.co/{org}/{repo}/resolve/{rev}/{subpath}"
    if uri.startswith("https://"):
        return uri
    if allow_http and uri.startswith("http://"):
        return uri
    raise MaskError(f"unsupported or insecure URI {uri!r}; use https:// or hf:// (or --allow-http for local tests)")


def pull_mask(
    uri: str,
    out_path: str | Path,
    expected_sha256: str | None = None,
    allow_unpinned: bool = False,
    allow_http: bool = False,
    max_bytes: int = 16 << 20,
    timeout: float = 60.0,
) -> dict:
    import hashlib
    import urllib.request

    if not expected_sha256 and not allow_unpinned:
        raise MaskError("--sha256 is required for mask pull (pass --allow-unpinned to opt out)")
    url = _resolve_pull_url(uri, allow_http=allow_http)
    req = urllib.request.Request(url, headers={"User-Agent": "chantless-mask-pull/0.3"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = resp.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise MaskError(f"mask payload exceeds {max_bytes} byte cap")
    got_sha = hashlib.sha256(data).hexdigest()
    if expected_sha256 and got_sha.lower() != expected_sha256.strip().lower():
        raise MaskError(f"SHA256 mismatch for {uri}: expected {expected_sha256}, got {got_sha}")
    try:
        parsed = json.loads(data.decode("utf-8"))
    except Exception as e:
        raise MaskError(f"downloaded payload is not valid UTF-8 JSON: {e}") from e
    if not isinstance(parsed, dict) or parsed.get("schema") not in (SCHEMA, SCHEMA_V2, LEGACY):
        raise MaskError(f"downloaded JSON is not a valid Chantless mask (schema={parsed.get('schema') if isinstance(parsed, dict) else None!r})")
    op_p = Path(out_path)
    op_p.parent.mkdir(parents=True, exist_ok=True)
    tmp = op_p.with_suffix(op_p.suffix + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(op_p)
    return {"uri": uri, "resolved_url": url, "out": str(op_p), "bytes": len(data), "sha256": got_sha, "schema": parsed.get("schema")}


def _parse_tensor_types_input(path: Path, source_names: list[str] | None = None) -> tuple[dict[str, str], list[dict], str]:
    import re
    raw_txt = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        obj = json.loads(raw_txt)
        if isinstance(obj, dict) and isinstance(obj.get("tensors"), list):
            tmap: dict[str, str] = {}
            inv: list[dict] = []
            for entry in obj["tensors"]:
                nm = entry["name"]
                ty = normalize_tier(entry.get("output_type") or entry.get("type"))
                tmap[nm] = ty
                inv.append({"name": nm, "type": ty, "shape": entry.get("shape") or []})
            return tmap, inv, "tensor_audit_json"
        if isinstance(obj, dict) and isinstance(obj.get("types"), dict):
            tmap = {nm: normalize_tier(ty) for nm, ty in obj["types"].items()}
            inv = [{"name": nm, "type": ty, "shape": []} for nm, ty in tmap.items()]
            return tmap, inv, "types_json"
        if isinstance(obj, dict) and all(isinstance(v, str) for v in obj.values()):
            tmap = {nm: normalize_tier(ty) for nm, ty in obj.items()}
            inv = [{"name": nm, "type": ty, "shape": []} for nm, ty in tmap.items()]
            return tmap, inv, "flat_types_json"
        raise MaskError(f"unrecognized JSON tensor-types layout in {path}")

    entries: list[tuple[str, str]] = []
    for line in raw_txt.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        for tok in s.split():
            if "=" not in tok:
                continue
            pat, ty = tok.rsplit("=", 1)
            entries.append((pat.strip(), normalize_tier(ty)))
    if not entries:
        raise MaskError(f"no tensor=type entries found in {path}")

    tmap = {}
    if source_names:
        for pat, ty in entries:
            exact = pat.lstrip("^").rstrip("$").replace("\\.", ".")
            if exact in source_names:
                tmap[exact] = ty
            else:
                rx = re.compile(pat)
                for nm in source_names:
                    if rx.search(nm):
                        tmap[nm] = ty
    else:
        for pat, ty in entries:
            nm = pat.lstrip("^").rstrip("$").replace("\\.", ".")
            tmap[nm] = ty
    inv = [{"name": nm, "type": ty, "shape": []} for nm, ty in tmap.items()]
    return tmap, inv, "tensor_types_txt"


def _apply_dynquant_override(tmap: dict[str, str], groups: dict[str, dict], recipe: dict) -> dict:
    import re
    protect_raw = recipe.get("protect_qtype") or recipe.get("protect_type")
    if not protect_raw:
        raise MaskError("override recipe missing protect_qtype / protect_type")
    protect_t = normalize_tier(protect_raw)
    protected = recipe.get("protected") or []
    grp_rx = re.compile(r"^L(\d+)_(attn|mixer|ffn)", re.IGNORECASE)
    upgraded_groups: list[str] = []
    upgraded_tensors = 0
    for item in protected:
        if isinstance(item, dict) and item.get("overrides"):
            for pair in str(item["overrides"]).split(","):
                if "=" not in pair:
                    continue
                pat, ty = pair.rsplit("=", 1)
                nt = normalize_tier(ty)
                rx = re.compile(pat)
                for nm in list(tmap):
                    if rx.search(nm):
                        tmap[nm] = nt
                        upgraded_tensors += 1
            if item.get("label"):
                upgraded_groups.append(str(item["label"]))
            continue
        lbl = item.get("label") if isinstance(item, dict) else str(item)
        m = grp_rx.match(lbl)
        if not m:
            raise MaskError(f"unrecognized protected group label {lbl!r}")
        il, kind = int(m.group(1)), m.group(2).lower()
        gname = f"blk.{il}.{'attn' if kind in ('attn', 'mixer') else 'ffn'}"
        if gname not in groups:
            raise MaskError(f"protected group {gname!r} (from {lbl!r}) not found in model groups")
        for nm in groups[gname]["quantizable"]:
            tmap[nm] = protect_t
            upgraded_tensors += 1
        upgraded_groups.append(gname)
    return {"protect_type": protect_t, "protected_groups": upgraded_groups, "upgraded_tensors": upgraded_tensors}


def import_mask(
    out_path: str | Path,
    from_gguf: str | Path | None = None,
    from_tensor_types: str | Path | None = None,
    from_spec: str | Path | None = None,
    source_gguf: str | Path | None = None,
    hf_dir: str | Path | None = None,
    override_recipe: str | Path | None = None,
    repo: str = "",
    revision: str = "",
    source_sha256: str = "",
    objective_mode: str = "preserve",
    note: str | None = None,
    measured: dict | None = None,
    extra_provenance: dict | None = None,
) -> dict:
    from collections import Counter
    from . import gguf_ops

    modes = [bool(from_gguf), bool(from_tensor_types), bool(from_spec)]
    if sum(modes) != 1:
        raise MaskError("specify exactly one of from_gguf, from_tensor_types, or from_spec")

    source_info = {"repo": repo, "revision": revision, "gguf_sha256": source_sha256}
    groups: dict[str, dict]
    dead: list[str] = []
    override_info = None

    if from_spec:
        sp = Path(from_spec)
        raw = json.loads(sp.read_text(encoding="utf-8"))
        tiers = raw.get("tiers")
        if not isinstance(tiers, dict) or not tiers:
            raise MaskError(f"{sp}: expected non-empty 'tiers' dict")
        if source_gguf:
            groups, dead = gguf_ops.adapter_from_gguf(source_gguf, hf_dir)
            m = new_mask(groups, "BF16", source_info, objective_mode=objective_mode, schema=SCHEMA)
            m["dead_tensors"] = list(dead)
            for g, t in tiers.items():
                if g in m["groups"]:
                    m["groups"][g] = {"tier": normalize_tier(t)}
            validate(m, groups)
        else:
            m = {
                "schema": SCHEMA,
                "source": source_info,
                "objective_mode": objective_mode,
                "groups": {g: {"tier": normalize_tier(t)} for g, t in sorted(tiers.items())},
                "tensors": {},
                "floors": {},
                "weights": {"lineage": []},
                "deployment": {},
                "measured": dict(measured or {}),
                "dead_tensors": [],
                "widths": {},
                "vocab_keep": None,
                "apprentice": {"switches": {}, "levels": {}, "decision": {}},
            }
        src_format = "legacy_spec_json"
        in_name = sp.name
    else:
        if from_gguf:
            gp = Path(from_gguf)
            inv_built = gguf_ops.inventory(gp)
            tmap = {t["name"]: normalize_tier(t["type"]) for t in inv_built}
            src_format = "gguf"
            in_name = gp.name
            if source_gguf:
                groups, dead = gguf_ops.adapter_from_gguf(source_gguf, hf_dir)
            else:
                groups, dead = gguf_ops.adapter_from_gguf(gp, hf_dir)
        else:
            tp = Path(from_tensor_types)
            src_names = [t["name"] for t in gguf_ops.inventory(source_gguf)] if source_gguf else None
            tmap, inv_tt, src_format = _parse_tensor_types_input(tp, source_names=src_names)
            in_name = tp.name
            if source_gguf:
                groups, dead = gguf_ops.adapter_from_gguf(source_gguf, hf_dir)
            else:
                groups, dead = gguf_ops._build_groups_and_dead(inv_tt, strip_dead=False)

        if override_recipe:
            rec = json.loads(Path(override_recipe).read_text(encoding="utf-8"))
            override_info = _apply_dynquant_override(tmap, groups, rec)

        m = new_mask(groups, "BF16", source_info, objective_mode=objective_mode, schema=SCHEMA)
        m["dead_tensors"] = list(dead)
        for g in sorted(k for k in groups if not k.startswith("_")):
            qnames = groups[g]["quantizable"]
            present = [tmap[n] for n in qnames if n in tmap]
            if not present:
                if source_gguf and groups[g]["droppable"]:
                    m["groups"][g] = {"tier": DROP}
                else:
                    raise MaskError(f"group {g} has no quantizable tensors in imported input")
                continue
            counts = Counter(present)
            mode_tier = min(counts, key=lambda t: (-counts[t], rank(t)))
            m["groups"][g] = {"tier": mode_tier}
            for n in qnames:
                if n in tmap and tmap[n] != mode_tier:
                    m["tensors"][n] = tmap[n]

        validate(m, groups)
        planned, _ = tensor_plan(m, groups)
        mismatches = [n for n, t in planned.items() if n in tmap and tmap[n] != t]
        if mismatches:
            raise MaskError(f"imported mask tensor_plan mismatch on {len(mismatches)} tensors: {mismatches[:5]}")

    if measured:
        m["measured"] = dict(measured)
    prov: dict = {"op": "import", "from_format": src_format, "input": in_name}
    if override_info:
        prov["override_recipe"] = override_info
    if note:
        prov["note"] = note
    if extra_provenance:
        prov.update(extra_provenance)
    m["provenance"] = prov

    op_p = Path(out_path)
    op_p.parent.mkdir(parents=True, exist_ok=True)
    tmp = op_p.with_suffix(op_p.suffix + ".tmp")
    tmp.write_text(json.dumps(m, indent=2) + "\n", encoding="utf-8")
    tmp.replace(op_p)
    return {
        "out": str(op_p),
        "out_sha256": _mask_sha256(op_p),
        "schema": m["schema"],
        "from_format": src_format,
        "groups": len(m["groups"]),
        "tensor_overrides": len(m["tensors"]),
        "tier_counts": inspect_mask(m)["tier_counts"],
    }
