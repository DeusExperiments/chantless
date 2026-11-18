"""One-shot mask search (`learner.algo: "oneshot"`): solo + width surrogate under an exact GGUF byte model.

Supports both:
  - Tier-only one-shot search (`options: {group: [tier, ...]}`), backward-compatible with `v2`.
  - Joint `(tier, width)` + `vocab_keep` + `dead_tensors` search (`chantless.mask/v3`), where each FFN group's
    predicted score shift at `(tier, width)` is `shift(group, tier) + cut_shift(layer, width)` (piecewise-linear
    interpolation between measured width probes, with `width == 0` mapped to `'drop'`), minimizing either
    `1 - Spearman(predicted, source)` (`FidelitySurrogate`) or weighted store damage (`Surrogate`).
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from . import gguf_ops, mask as mk, objectives, vocab

BLOCK = {
    "F32": (1, 4), "BF16": (1, 2), "F16": (1, 2), "Q8_0": (32, 34), "Q6_K": (256, 210),
    "Q5_K": (256, 176), "Q4_K": (256, 144), "Q3_K": (256, 110), "Q2_K": (256, 84),
    "IQ2_XS": (256, 74), "IQ1_S": (256, 50),
}


def _pad(n: int, a: int = 32) -> int:
    return (n + a - 1) // a * a


def ti_bytes(name: str, ndim: int) -> int:
    return 8 + len(name.encode()) + 4 + 8 * ndim + 4 + 8


def tensor_bytes(dims, ty: str, padded: bool = True) -> int:
    bs, ts = BLOCK[ty.upper()]
    if dims[0] % bs:
        raise ValueError(f"row length {dims[0]} is not a multiple of {ty} block {bs}")
    raw = ts * (dims[0] // bs) * int(np.prod(dims[1:], dtype=np.int64))
    return _pad(raw) if padded else raw


class ByteModel:
    """Exact GGUF byte model accounting dropped/dead tensors, tensor-info table, MLP widths, and vocabulary trim."""

    def __init__(
        self,
        source_gguf: str | None,
        groups: dict,
        *,
        tensors: list[dict] | None = None,
        kv_end: int | None = None,
        ff_is_array: bool | None = None,
        n_layer: int | None = None,
        alignment: int = 32,
    ):
        if source_gguf is not None:
            h = gguf_ops._Hdr(source_gguf)
            try:
                kvs = h.parsed_kvs()
                arch = h.arch or ""
                self._tensor_list = [{"name": t["name"], "shape": list(t["dims"]), "type": t["type"]} for t in h.tensors]
                self.kv_end = h.kv_end if kv_end is None else int(kv_end)
                self.alignment = h.alignment
                self.ff_is_array = isinstance(kvs.get(f"{arch}.feed_forward_length"), list) if ff_is_array is None else bool(ff_is_array)
                self.n_layer = int(kvs.get(f"{arch}.block_count", 0)) if n_layer is None else int(n_layer)
            finally:
                h.close()
        else:
            if tensors is None or kv_end is None:
                raise ValueError("ByteModel needs either source_gguf or (tensors, kv_end)")
            self._tensor_list = [{"name": t["name"], "shape": list(t.get("shape", t.get("dims"))), "type": t["type"]} for t in tensors]
            self.kv_end = int(kv_end)
            self.alignment = int(alignment)
            self.ff_is_array = bool(ff_is_array)
            self.n_layer = int(n_layer or len({int(t["name"].split(".")[1]) for t in self._tensor_list if t["name"].startswith("blk.") and t["name"].split(".")[1].isdigit()}))

        self.source_gguf = source_gguf
        self.inv = {t["name"]: {**t, "dims": list(t["shape"])} for t in self._tensor_list}
        self.groups = groups
        self.kv_q_delta = 0
        self.n_ff0 = {
            int(t["name"].split(".")[1]): int(t["shape"][0])
            for t in self._tensor_list if t["name"].endswith("ffn_down.weight")
        }
        self.vocab_size = int(self.inv["token_embd.weight"]["shape"][1]) if "token_embd.weight" in self.inv else 0
        self._vocab_kv_cut = 0
        self._vocab_rows_kept: int | None = None
        self._vocab_cache_key: object = None

    def set_vocab_trim(self, keep: Sequence[int] | np.ndarray | None, kv_cut: int | None = None) -> None:
        """Configure vocabulary trim accounting (`keep` array, or explicit `(kept_count, kv_cut)`)."""
        self._vocab_cache_key = None
        if keep is None:
            self._vocab_kv_cut = 0
            self._vocab_rows_kept = None
            return
        keep_arr = np.asarray(keep, dtype=np.int64)
        if self.vocab_size and len(keep_arr) >= self.vocab_size:
            self._vocab_kv_cut = 0
            self._vocab_rows_kept = None
            return
        self._vocab_rows_kept = int(len(keep_arr))
        if kv_cut is not None:
            self._vocab_kv_cut = int(kv_cut)
        elif self.source_gguf is not None:
            self._vocab_kv_cut = vocab.kv_trim_savings(self.source_gguf, keep_arr)
        else:
            raise ValueError("kv_cut is required when source_gguf is not provided")

    def _raw_file_bytes(self, tiers: dict[str, str], widths: dict[int, int], dead: set[str], kv_end: int, tensors_override: dict[str, str] | None = None) -> int:
        tier_map: dict[str, str] = {}
        drop = set(dead)
        for g, t in tiers.items():
            if t == "drop":
                drop.update(self.groups[g]["quantizable"] + self.groups[g]["fixed"])
            else:
                for n in self.groups[g]["quantizable"]:
                    if n not in drop:
                        tier_map[n] = (tensors_override or {}).get(n, t)

        active_widths = {int(il): int(w) for il, w in widths.items() if int(w) < self.n_ff0.get(int(il), 0) and tiers.get(f"blk.{int(il)}.ffn") != "drop"}
        ff_kv_delta = (8 + 4 * self.n_layer) if (active_widths and not self.ff_is_array) else 0

        tot_payload = 0
        tot_ti = 0
        for t in self._tensor_list:
            n = t["name"]
            if n in drop:
                continue
            d = list(t["shape"])
            p = n.split(".")
            if p[0] == "blk" and p[1].isdigit() and int(p[1]) in active_widths and p[2] in ("ffn_gate", "ffn_up", "ffn_down"):
                w = active_widths[int(p[1])]
                d = [w, d[1]] if p[2] == "ffn_down" else [d[0], w]
            ty = tier_map.get(n, t["type"])
            tot_payload += _pad(tensor_bytes(d, ty, padded=False), self.alignment)
            tot_ti += ti_bytes(n, len(d))

        kv_cut = self._vocab_kv_cut if (self._vocab_rows_kept is not None and self.vocab_size > self._vocab_rows_kept) else 0
        b = _pad(kv_end + ff_kv_delta - kv_cut + tot_ti, self.alignment) + tot_payload
        if self._vocab_rows_kept is not None and self.vocab_size > self._vocab_rows_kept:
            rows_cut = self.vocab_size - self._vocab_rows_kept
            for tab in vocab.VOCAB_TABLES:
                if tab in self.inv and tab not in drop and self.inv[tab]["shape"][1] == self.vocab_size:
                    cols = int(self.inv[tab]["shape"][0])
                    ty = tier_map.get(tab, self.inv[tab]["type"]).upper()
                    bs, ts = BLOCK[ty]
                    b -= rows_cut * (ts * (cols // bs))
        return int(b)

    def spec_bytes(self, tiers: dict[str, str], widths: dict[int, int] | None = None, dead: Sequence[str] = ()) -> int:
        return self._raw_file_bytes(tiers, widths or {}, set(dead), self.kv_end + self.kv_q_delta)

    def calibrate(self, m: dict, real_bytes: int) -> None:
        """Calibrate `llama-quantize` KV overhead (`imatrix` metadata keys) on one real quantized build."""
        saved_kept, saved_kv = self._vocab_rows_kept, self._vocab_kv_cut
        self._vocab_rows_kept, self._vocab_kv_cut = None, 0
        try:
            tiers = {g: e["tier"] for g, e in m["groups"].items()}
            widths = {il: len(idx) for il, idx in mk.kept_widths(m).items()}
            dead = set(m.get("dead_tensors") or ())
            uncal = self._raw_file_bytes(tiers, widths, dead, self.kv_end, m.get("tensors"))
            self.kv_q_delta = int(real_bytes) - uncal
        finally:
            self._vocab_rows_kept, self._vocab_kv_cut = saved_kept, saved_kv

    def __call__(self, m: dict) -> int:
        tiers = {g: e["tier"] for g, e in m["groups"].items()}
        widths = {il: len(idx) for il, idx in mk.kept_widths(m).items()}
        dead = set(m.get("dead_tensors") or ())
        vk = m.get("vocab_keep")
        if vk is not None and self.source_gguf is not None:
            vkey = vk.get("b64") if isinstance(vk, dict) else tuple(vk)
            if vkey != self._vocab_cache_key or self._vocab_rows_kept is None:
                self.set_vocab_trim(mk.unpack_bitmap(vk))
                self._vocab_cache_key = vkey
        elif "vocab_keep" in m and vk is None and self._vocab_cache_key is not None:
            self.set_vocab_trim(None)
        return self._raw_file_bytes(tiers, widths, dead, self.kv_end + self.kv_q_delta, m.get("tensors"))


class Surrogate:
    """Additive per-trajectory surrogate supporting both plain tier choices and joint `(tier, width)` choices."""

    def __init__(
        self,
        stores,
        solo: dict,
        width_cuts: dict[tuple[int, int], np.ndarray] | None = None,
        n_ff0: dict[int, int] | None = None,
    ):
        self.stores = stores
        self.src = np.concatenate([s[3] for s in stores])
        self.cuts = np.cumsum([0] + [len(s[3]) for s in stores])
        self.solo = solo                      # (group, tier) -> score shift over all stores, concatenated
        self.width_cuts = dict(width_cuts or {})  # (layer_idx, width) -> score shift over all stores
        self.n_ff0 = dict(n_ff0 or {})
        self.zero = np.zeros(len(self.src))
        for (g, t), v in self.solo.items():
            if t == "drop" and g.endswith(".ffn"):
                il = int(g.split(".")[1])
                self.width_cuts.setdefault((il, 0), v)

    def cut_shift(self, il: int, w: int) -> np.ndarray:
        full = self.n_ff0.get(il)
        if full is None or w >= full:
            return self.zero
        pts = sorted([(ww, v) for (l, ww), v in self.width_cuts.items() if l == il] + [(full, self.zero)], key=lambda q: q[0])
        for (w0, v0), (w1, v1) in zip(pts, pts[1:]):
            if w0 <= w <= w1:
                if w1 == w0:
                    return v0
                a = (w - w0) / (w1 - w0)
                return (1.0 - a) * v0 + a * v1
        raise ValueError(f"width {w} for layer {il} falls outside measured range {[p[0] for p in pts]}")

    def shift(self, g: str, opt) -> np.ndarray:
        if isinstance(opt, tuple):
            t, w = opt
            if t == "drop":
                return self.solo.get((g, "drop"), self.zero)
            x = self.zero if t in ("BF16", "Q8_0") and (g, t) not in self.solo else self.solo.get((g, t), self.zero)
            if g.endswith(".ffn"):
                il = int(g.split(".")[1])
                if w < self.n_ff0.get(il, w):
                    x = x + self.cut_shift(il, w)
            return x
        return self.solo.get((g, opt), self.zero) if opt not in ("BF16",) else self.zero

    def damage(self, x: np.ndarray) -> float:
        num = den = 0.0
        for (_, w, fn, src), a, b in zip(self.stores, self.cuts, self.cuts[1:]):
            num += w * len(src) * fn(x[a:b])
            den += w * len(src)
        return num / den


def solve(
    sur: Surrogate,
    options: dict,
    cost: dict,
    budget: int,
    header: int,
    pair_top: int = 60,
    max_polish: int = 100,
    log=print,
    guard_fn=None,
) -> dict:
    """Solve `min sur.damage(m)` s.t. `header + sum cost[(g, m[g])] <= budget`.

    Each `options[g]` may contain tier strings (`"Q4_K"`) or `(tier, width)` tuples (`("Q4_K", 3072)`).
    Optional `guard_fn(m, bytes_m) -> float` returns `<= 0.0` when satisfied and `> 0.0` excess violation
    (e.g. predicted `general` KL minus `SizeMatchedCap(bpw)`), preferred whenever feasible.
    """
    rich = {g: o[0] if len(set(cost[(g, x)] for x in o)) == 1 else max(o, key=lambda t: cost[(g, t)]) for g, o in options.items()}
    m = dict(rich)
    dvc = {(g, t): sur.shift(g, t) for g, o in options.items() for t in o}
    B = lambda m_: header + sum(cost[(g, t)] for g, t in m_.items())
    x = sur.src + sum(dvc[(g, t)] for g, t in m.items())
    cur = sur.damage(x)
    while B(m) > budget:
        best_ok = None
        best_any = None
        cur_b = B(m)
        for g, o in options.items():
            for t in o:
                db = cost[(g, m[g])] - cost[(g, t)]
                if db <= 0:
                    continue
                y = x - dvc[(g, m[g])] + dvc[(g, t)]
                d = sur.damage(y) - cur
                sc = (d + 1e-12) / db if d >= 0 else d * db
                cand = (sc, g, t, y, d)
                if best_any is None or sc < best_any[0]:
                    best_any = cand
                if guard_fn is not None:
                    m_try = dict(m); m_try[g] = t
                    if guard_fn(m_try, cur_b - db) <= 0.0 and (best_ok is None or sc < best_ok[0]):
                        best_ok = cand
        best = best_ok if best_ok is not None else best_any
        if best is None:
            raise RuntimeError(f"budget {budget} is below the cheapest mask {B(m)}")
        _, g, t, x, d = best
        m[g] = t
        cur += d
    log(f"oneshot: descent reached {B(m)} bytes, predicted damage {cur:.6f}")
    for _ in range(max_polish):
        b = B(m)
        cur_viol = max(0.0, float(guard_fn(m, b))) if guard_fn is not None else 0.0
        singles = []
        for g, o in options.items():
            for t in o:
                if t != m[g]:
                    singles.append((sur.damage(x - dvc[(g, m[g])] + dvc[(g, t)]) - cur, cost[(g, t)] - cost[(g, m[g])], g, t))
        best = (-1e-7, None)

        def _guard_ok(changes, new_b):
            if guard_fn is None:
                return True
            m_try = dict(m)
            for gg, tt in changes:
                m_try[gg] = tt
            return float(guard_fn(m_try, new_b)) <= cur_viol + 1e-9

        for d, db, g, t in singles:
            if b + db <= budget and d < best[0] and _guard_ok([(g, t)], b + db):
                best = (d, [(g, t)])
        ups = sorted([s for s in singles if s[1] > 0], key=lambda s: s[0])[:pair_top]
        downs = sorted([s for s in singles if s[1] < 0], key=lambda s: s[0] / -s[1])[:pair_top]
        for d1, db1, g1, t1 in ups:
            y1 = x - dvc[(g1, m[g1])] + dvc[(g1, t1)]
            for d2, db2, g2, t2 in downs:
                if g1 != g2 and b + db1 + db2 <= budget:
                    d = sur.damage(y1 - dvc[(g2, m[g2])] + dvc[(g2, t2)]) - cur
                    if d < best[0] and _guard_ok([(g1, t1), (g2, t2)], b + db1 + db2):
                        best = (d, [(g1, t1), (g2, t2)])
        if best[1] is None:
            break
        for g, t in best[1]:
            x = x - dvc[(g, m[g])] + dvc[(g, t)]
            m[g] = t
        cur = sur.damage(x)
    log(f"oneshot: exchange settled at {B(m)} bytes, predicted damage {cur:.6f}")
    return m


def unpack_solution(sol: dict) -> tuple[dict[str, str], dict[int, int]]:
    """Split a `solve()` output `{group: tier | (tier, width)}` into `(tiers, cut_widths)`."""
    tiers: dict[str, str] = {}
    widths: dict[int, int] = {}
    for g, v in sol.items():
        if isinstance(v, tuple):
            t, w = v
            tiers[g] = t
            if g.endswith(".ffn") and t != "drop" and w > 0:
                widths[int(g.split(".")[1])] = int(w)
        else:
            tiers[g] = v
    return tiers, widths


def build_joint_options(
    groups: dict[str, dict],
    allowed_tiers: dict[str, list[str]],
    solo: dict,
    width_cuts: dict[tuple[int, int], np.ndarray],
    n_ff0: dict[int, int],
    width_step: int = 256,
    allow_drop: bool = True,
) -> dict[str, list[tuple[str, int]]]:
    """Construct `(tier, width)` candidate lists per group, respecting block-size constraints."""
    measured_layers = {il for (il, w) in width_cuts if w > 0}
    opts: dict[str, list[tuple[str, int]]] = {}
    for g, info in groups.items():
        if g.startswith("_"):
            continue
        ts = [t for t in allowed_tiers.get(g, []) if t in ("BF16", "Q8_0") or (g, t) in solo]
        if g.endswith(".ffn"):
            il = int(g.split(".")[1])
            full = n_ff0[il]
            ws = list(range(full, 0, -width_step)) if il in measured_layers else [full]
            o = [(t, w) for t in ts for w in ws if w % BLOCK[t.upper()][0] == 0]
        else:
            o = [(t, 0) for t in ts]
        if allow_drop and info.get("droppable") and (g, "drop") in solo:
            o.append(("drop", 0))
        opts[g] = o
    return opts


def pauc_store(src_split: dict, parts: list[dict], args: dict):
    """Damage fn on one logodds store's concatenated scores, in its parts order."""
    roles = np.concatenate([np.full(p["n"], p["role"] == "pos") for p in parts])
    def fn(x):
        return objectives.pauc_damage({"pos": x[roles], "neg": x[~roles]}, src_split, **args)
    return fn


def kl_store_damage(tgt_weights: np.ndarray):
    """Target-weighted mean KL or NLL damage function for a generative `traj-kl` store."""
    tw = np.asarray(tgt_weights, dtype=np.float64)
    denom = max(1.0, float(tw.sum()))
    def fn(x: np.ndarray) -> float:
        return float(np.dot(x, tw) / denom)
    return fn


def extract_traj_kl(receipt_json: dict, mode: str = "preserve") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return `(cand_vec, src_vec, tgt_weights)` from a `traj-kl` JSON receipt containing `'trajs'`."""
    trajs = receipt_json.get("trajs")
    if not isinstance(trajs, list) or not trajs:
        # Scalar fallback if an older/stub receipt lacks per-trajectory breakdown
        n_t = max(1, int(receipt_json.get("n_tgt", 1)))
        if mode == "preserve":
            return np.array([float(receipt_json["kl"])], dtype=np.float64), np.zeros(1, dtype=np.float64), np.array([float(n_t)], dtype=np.float64)
        return (
            np.array([float(receipt_json["nll_cand"])], dtype=np.float64),
            np.array([float(receipt_json.get("nll_src", 0.0))], dtype=np.float64),
            np.array([float(n_t)], dtype=np.float64),
        )
    tw = np.array([float(r.get("n_tgt", 1)) for r in trajs], dtype=np.float64)
    if mode == "preserve":
        cand = np.array([float(r["kl"]) for r in trajs], dtype=np.float64)
        src = np.zeros(len(trajs), dtype=np.float64)
    else:
        cand = np.array([float(r["nll_cand"]) for r in trajs], dtype=np.float64)
        src = np.array([float(r.get("nll_src", 0.0)) for r in trajs], dtype=np.float64)
    return cand, src, tw


def _rank(a: np.ndarray) -> np.ndarray:
    r = np.argsort(np.argsort(a, kind="stable"), kind="stable").astype(np.float64)
    return (r - r.mean()) / r.std()


class FidelitySurrogate(Surrogate):
    """Damage = 1 - Spearman(predicted scores, source scores) over every store's trajectories."""

    def __init__(self, stores, solo: dict, width_cuts: dict | None = None, n_ff0: dict[int, int] | None = None):
        super().__init__(stores, solo, width_cuts=width_cuts, n_ff0=n_ff0)
        self.rs = _rank(self.src)

    def damage(self, x: np.ndarray) -> float:
        return 1.0 - float((_rank(x) * self.rs).mean())
