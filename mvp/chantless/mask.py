"""Precision mask schema `chantless.mask/v2`, with a read-only loader for legacy `onlyskills.mask/v1`.

v1 files are never rewritten. A v2 mask assigns each group a tier from the ladder or `drop`
(physical removal of the group's tensors); optional per-tensor entries override the group tier.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

SCHEMA = "chantless.mask/v2"
LEGACY = "onlyskills.mask/v1"

# Cheapening order. BF16 is the source representation; `drop` is physical removal.
LADDER = ["BF16", "Q8_0", "Q6_K", "Q5_K", "Q4_K", "Q3_K", "Q2_K", "IQ2_XS", "IQ1_S"]
DROP = "drop"
IMATRIX_REQUIRED = {"IQ2_XS", "IQ2_XXS", "IQ1_M", "IQ1_S"}


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
    return len(LADDER) if tier == DROP else LADDER.index(tier)


class MaskError(ValueError):
    pass


def new_mask(groups: dict[str, dict], start_tier: str, source: dict, objective_mode: str = "preserve") -> dict:
    return {"schema": SCHEMA, "source": source, "objective_mode": objective_mode,
            "groups": {g: {"tier": start_tier} for g in groups if not g.startswith("_")},
            "tensors": {}, "floors": {}, "weights": {"lineage": []}, "deployment": {}, "measured": {}}


def validate(mask: dict, groups: dict[str, dict], ladder: list[str] | None = None) -> None:
    ladder = ladder or LADDER
    if mask.get("schema") != SCHEMA:
        raise MaskError(f"schema must be {SCHEMA}")
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
        if fl and rank(t) > rank(fl["min_tier"]) and not (t == DROP and fl.get("drop_ok")):
            raise MaskError(f"{g}: tier {t} below sanity floor {fl['min_tier']}")
    if dropped_attn >= n_attn:
        raise MaskError("at least one attention block must remain")
    for name, t in mask.get("tensors", {}).items():
        if t not in ladder:
            raise MaskError(f"tensor {name}: unknown tier {t}")


def tensor_plan(mask: dict, groups: dict[str, dict]) -> tuple[dict[str, str], set[str]]:
    """-> ({quantizable tensor: tier}, {tensors to remove})."""
    types, drop = {}, set()
    for g, e in mask["groups"].items():
        if e["tier"] == DROP:
            drop.update(groups[g]["quantizable"] + groups[g]["fixed"])
        else:
            for n in groups[g]["quantizable"]:
                types[n] = mask.get("tensors", {}).get(n, e["tier"])
    return types, drop


def load(path: str | Path) -> dict:
    d = json.loads(Path(path).read_text())
    if d.get("schema") == SCHEMA:
        return d
    if d.get("schema") == LEGACY:
        return from_v1(d)
    raise MaskError(f"unknown mask schema {d.get('schema')!r}")


def from_v1(d: dict) -> dict:
    """Read a legacy v1 mask (manual §11.1) into v2 form without touching the original file."""
    out = {"schema": SCHEMA, "source": d.get("source", {}), "objective_mode": "preserve", "groups": {}, "tensors": {},
           "floors": {}, "weights": {"lineage": []}, "deployment": {"kv_type": d.get("kv_type")},
           "measured": d.get("measured", {}), "legacy": {"schema": LEGACY, "removed": d.get("removed", [])}}
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
