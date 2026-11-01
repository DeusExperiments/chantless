"""Learning from labelled examples at the mask stage (mask-plus-weights for a classifier, decision S3).

The built model is the starting point and the only input besides its config: `Student` rebuilds it in PyTorch
from the GGUF, with its removed sub-blocks, cut neurons and trimmed vocabulary, so the student scores what
the file scores. Training then moves levels on each tensor's fixed grid (lattice.py) with a straight-through
gradient, and `write_levels` puts them back. Size cannot change, and rounding cannot erase what was learned.

Feedback arrives as labelled documents. A wrong score costs a margin loss, a human and AI document from the
same origin form a contrastive pair, and a replay set of ordinary documents is held near its starting scores
so a few corrections cannot move everything else. The real built file is judged afterwards, never the student.
"""
from __future__ import annotations

import ctypes
import hashlib
import json
import random
import re
import struct
from collections import deque
from pathlib import Path

import numpy as np
from torch import nn

from . import lattice

LAYER_NAMES = {
    "attn_norm": "input_layernorm.weight", "post_attention_norm": "post_attention_layernorm.weight",
    "ffn_gate": "mlp.gate_proj.weight", "ffn_up": "mlp.up_proj.weight", "ffn_down": "mlp.down_proj.weight",
    "attn_q": "self_attn.q_proj.weight", "attn_k": "self_attn.k_proj.weight", "attn_v": "self_attn.v_proj.weight",
    "attn_output": "self_attn.o_proj.weight", "attn_q_norm": "self_attn.q_norm.weight", "attn_k_norm": "self_attn.k_norm.weight",
    "attn_qkv": "linear_attn.in_proj_qkv.weight", "attn_gate": "linear_attn.in_proj_z.weight",
    "ssm_alpha": "linear_attn.in_proj_a.weight", "ssm_beta": "linear_attn.in_proj_b.weight",
    "ssm_out": "linear_attn.out_proj.weight", "ssm_norm": "linear_attn.norm.weight",
    "ssm_conv1d": "linear_attn.conv1d.weight", "ssm_a": "linear_attn.A_log", "ssm_dt": "linear_attn.dt_bias",
}


class OnGrid(nn.Module):
    """Rounds a weight onto a tensor's fixed grid by straight-through, so the student scores what the file holds."""

    def __init__(self, lo, step, top):
        super().__init__()
        self.register_buffer("lo", lo, persistent=False)
        self.register_buffer("step", step, persistent=False)
        self.top = top

    def levels(self, w):
        return ((w - self.lo) / self.step.clamp_min(1e-30)).round().clamp(0, self.top)

    def forward(self, w):
        return w + (self.lo + self.step * self.levels(w) - w).detach()


def hf_name(gg: str) -> str:
    if gg == "token_embd.weight":
        return "model.embed_tokens.weight"
    if gg == "output_norm.weight":
        return "model.norm.weight"
    p = gg.split(".")
    return f"model.layers.{p[1]}.{LAYER_NAMES[p[2]]}"


def floats(art: lattice.Artifact, name: str, ggml_lib: str | None) -> np.ndarray:
    """A tensor as float32 in torch shape. Tiers without a grid are decoded by ggml itself."""
    t, shape = art.tensors[name], art.shape(name)
    if t["type"] == "F32":
        return np.frombuffer(art.raw(name), dtype="<f4").reshape(shape)
    if t["type"] == "BF16":    # the top half of a float32; ggml has no dequantize_row_bf16
        return (np.frombuffer(art.raw(name), dtype="<u2").astype("<u4") << 16).view("<f4").reshape(shape)
    if t["type"] in lattice.TOP:
        return art.grid(name).values().reshape(shape)
    lib = ctypes.CDLL(ggml_lib)
    fn = getattr(lib, f"dequantize_row_{t['type'].lower()}")
    fn.argtypes, fn.restype = [ctypes.c_char_p, ctypes.c_void_p, ctypes.c_int64], None
    y = np.empty(t["n_elements"], np.float32)
    fn(bytes(art.raw(name)), y.ctypes.data, t["n_elements"])
    return y.reshape(shape)


def train_names(art: lattice.Artifact, cfg: dict) -> list[str]:
    """Grid tensors in the last `last_blocks` blocks still present, or those matching `train_regex`."""
    grid = [n for n, t in art.tensors.items() if t["type"] in lattice.TOP and n.startswith("blk.")]
    if cfg.get("train_regex"):
        return [n for n in grid if re.search(cfg["train_regex"], n)]
    kept = sorted({int(n.split(".")[1]) for n in grid})[-cfg.get("last_blocks", 2):]
    return [n for n in grid if int(n.split(".")[1]) in kept]


def build_student(art: lattice.Artifact, config_dir: str, train: list[str], ggml_lib: str | None, device: str = "cpu",
                  latents: dict[str, np.ndarray] | None = None):
    """The file as a PyTorch model. `latents` holds float32 arrays in torch shape, one per `train` tensor, the floats this file
    was built from: they sit behind its grids in place of the grid centres, so what training learned below one level survives
    a rebuild. The rounded student must still equal the file, level for level."""
    import torch
    from torch.nn.utils import parametrize
    from transformers import AutoConfig, AutoModelForCausalLM

    class Zero(nn.Module):
        def __init__(self, pair: bool):
            super().__init__()
            self.pair = pair

        def forward(self, *a, **k):
            x = k["hidden_states"] if "hidden_states" in k else a[0]
            z = torch.zeros_like(x)
            return (z, None) if self.pair else z

    conf = AutoConfig.from_pretrained(config_dir)
    conf.vocab_size = art.tensors["token_embd.weight"]["dims"][1]
    conf.use_cache = False
    model = AutoModelForCausalLM.from_config(conf, dtype=torch.float32)
    for il, layer in enumerate(model.model.layers):
        if f"blk.{il}.ffn_down.weight" not in art.tensors:
            layer.mlp = Zero(False)
        else:
            for nm, gg in (("gate_proj", "ffn_gate"), ("up_proj", "ffn_up"), ("down_proj", "ffn_down")):
                rows, cols = art.shape(f"blk.{il}.{gg}.weight")
                setattr(layer.mlp, nm, nn.Linear(cols, rows, bias=False))
        if not ({f"blk.{il}.ssm_out.weight", f"blk.{il}.attn_output.weight"} & art.tensors.keys()):
            if layer.block_type == "linear_attention":
                layer.linear_attn = Zero(False)
            else:
                layer.self_attn = Zero(True)
    params = dict(model.named_parameters())
    for name in art.tensors:
        p, v = params[hf_name(name)], torch.from_numpy(floats(art, name, ggml_lib).copy())
        # the three places conversion to GGUF changes a value (conversion/qwen.py, Qwen3NextModel.modify_tensors)
        if "ssm_conv1d" in name:
            v = v.unsqueeze(1)
        elif name.endswith(".ssm_a"):
            v = torch.log(-v)
        elif name.endswith("norm.weight") and "ssm_norm" not in name:
            v = v - 1
        assert p.shape == v.shape, (name, p.shape, v.shape)
        p.data = v
    for p in model.parameters():
        p.requires_grad_(False)
    model.lm_head.weight = model.model.embed_tokens.weight
    model.to(device)
    mods = dict(model.named_modules())
    grids = {}
    for name in train:
        g = art.grid(name)
        path = hf_name(name).rsplit(".", 1)[0]
        m, shape = mods[path], art.shape(name)
        fq = OnGrid(torch.from_numpy(g.lo).reshape(shape).to(device), torch.from_numpy(g.step).reshape(shape).to(device), g.top)
        if latents is not None:
            w = torch.from_numpy(np.array(latents[name], dtype=np.float32)).to(device)
            assert w.shape == fq.lo.shape, (name, w.shape, fq.lo.shape)
            # a zero step has one value at every level, and within 1e-4 of a tie ggml's (x + dm) * id can round the other way
            x = (w - fq.lo) / fq.step.clamp_min(1e-30)
            off = (fq.levels(w) != torch.from_numpy(g.levels.copy()).reshape(shape).to(device)) & (fq.step > 0) & ((x - x.floor() - .5).abs() > 1e-4)
            assert not off.any(), (name, int(off.sum()), "latents are not the floats this file was built from")
            m.weight.data = w
        parametrize.register_parametrization(m, "weight", fq, unsafe=True)
        m.parametrizations.weight.original.requires_grad_(True)
        grids[name] = (m.parametrizations.weight.original, fq, torch.from_numpy(g.levels.copy()))
    model.eval()
    return model, grids


def read_trj(path: str) -> list[np.ndarray]:
    """Prompt tokens of each document in a TRJ1 file: the sequence up to its single target."""
    out = []
    with open(path, "rb") as f:
        assert f.read(4) == b"TRJ1"
        n = struct.unpack("<I", f.read(4))[0]
        for _ in range(n):
            nt, ng = struct.unpack("<II", f.read(8))
            assert ng == 1
            seq = np.frombuffer(f.read(4 * nt), dtype=np.int32)
            out.append(seq[:struct.unpack("<I", f.read(4))[0]].copy())
    return out


def collate(docs: list[np.ndarray], device: str, round_to: int = 256):
    """Left-padded batch. Lengths round up to a multiple of 256 so the kernels are not rebuilt for each length."""
    import torch
    n = int(-(-max(map(len, docs)) // round_to) * round_to)
    ids = torch.zeros((len(docs), n), dtype=torch.long)
    m = torch.zeros((len(docs), n), dtype=torch.long)
    for k, d in enumerate(docs):
        ids[k, n - len(d):] = torch.from_numpy(d.astype(np.int64))
        m[k, n - len(d):] = 1
    return ids.to(device), m.to(device)


def scores(model, docs: list[np.ndarray], pair: tuple[int, int], device: str, amp: bool):
    import torch
    ids, m = collate(docs, device)
    with torch.autocast(device.split(":")[0], dtype=torch.bfloat16, enabled=amp):
        logits = model(input_ids=ids, attention_mask=m, logits_to_keep=1).logits[:, -1]
    x = logits[:, list(pair)].float()
    return x[:, 0] - x[:, 1]


def vec_scores(model, docs: list[np.ndarray], pair: tuple[int, int], device: str, amp: bool):
    """The score and the final 1,024-number state, from which the whole output distribution follows."""
    import torch
    ids, m = collate(docs, device)
    with torch.autocast(device.split(":")[0], dtype=torch.bfloat16, enabled=amp):
        h = model.model(input_ids=ids, attention_mask=m).last_hidden_state[:, -1]
        x = model.lm_head(h)[:, list(pair)]
    x = x.float()
    return x[:, 0] - x[:, 1], h.float()


def capped(fn, model, docs: list[np.ndarray], pair: tuple[int, int], device: str, amp: bool, cap: int):
    """`fn` (scores or vec_scores) in sub-batches of at most `cap` padded tokens, longest first, back in the order of `docs`.
    One graph spans them, so a single backward sees the same loss while padding and peak memory both fall."""
    import torch
    parts = batches(docs, cap, len(docs))
    out = [fn(model, [docs[i] for i in b], pair, device, amp) for b in parts]
    back = torch.from_numpy(np.argsort([i for b in parts for i in b])).to(device)
    return tuple(torch.cat(x)[back] for x in zip(*out)) if isinstance(out[0], tuple) else torch.cat(out)[back]


def batches(docs: list[np.ndarray], budget: int, maxbs: int = 64) -> list[list[int]]:
    """Longest first, so each batch pads little and holds at most `budget` padded tokens."""
    order = np.argsort([-len(d) for d in docs]).tolist()
    out, cur, pad = [], [], 0
    for i in order:
        p = int(-(-len(docs[i]) // 256) * 256)
        q = max(pad, p)
        if cur and ((len(cur) + 1) * q > budget or len(cur) >= maxbs):
            out.append(cur)
            cur, q = [], p
        cur.append(i)
        pad = q
    return out + [cur] if cur else out


def score_all(model, docs, pair, device, amp, budget=40000) -> np.ndarray:
    import torch
    v = np.full(len(docs), np.nan)
    with torch.no_grad():
        for idx in batches(docs, budget):
            v[idx] = scores(model, [docs[i] for i in idx], pair, device, amp).cpu().numpy()
    assert np.isfinite(v).all()
    return v


def margin_loss(s, y, cutoff, margin):
    """Human documents (y = +1) must score above cutoff + margin, AI documents (y = -1) below cutoff - margin."""
    import torch
    return torch.nn.functional.softplus(margin - y * (s - cutoff)).mean()


def kl_vector(model, h, target_h, temp):
    """KL between whole output distributions, softened by `temp` so the mass beyond the two answer tokens counts."""
    import torch
    logq = torch.log_softmax(model.lm_head(h).float() / temp, -1)
    t = torch.softmax(model.lm_head(target_h).float() / temp, -1)
    return (t * (t.clamp_min(1e-12).log() - logq)).sum(-1).mean() * temp * temp


def kl_hold(s, s0, y, cutoff, temp):
    """KL to the starting model's answer, weighted to the right answers. Probabilities are centred on the cutoff, so documents
    near it count most and settled ones barely move the loss. Where the start was right it holds that answer; where it was
    wrong there is nothing to keep, and the document is pulled to its label."""
    import torch
    q = torch.sigmoid((s - cutoff) / temp).clamp(1e-6, 1 - 1e-6)
    p = torch.sigmoid((s0 - cutoff) / temp).clamp(1e-6, 1 - 1e-6)
    right = ((s0 > cutoff).float() * 2 - 1) == y
    target = torch.where(right, p, (y + 1) / 2)
    return (target * (target.clamp_min(1e-6) / q).log() + (1 - target) * ((1 - target).clamp_min(1e-6) / (1 - q)).log()).mean()


def keep_loss(drift, y, tau):
    """One-sided hold: a document moves the right way freely and the wrong way by up to `tau`, then pays the square of the rest."""
    import torch
    return torch.relu(-y * drift - tau).square().mean()


def pair_loss(s_human, s_ai, margin):
    """Contrastive: a human document must outscore the AI document it is paired with by at least `margin`."""
    import torch
    return torch.nn.functional.softplus(margin - (s_human - s_ai)).mean()


def _train(model, params, feedback, replay, pair, cutoff, cfg, log, device, amp, strata, lr, anchor=None, on_step=None) -> list[tuple]:
    """The shared loop: feedback margin, contrastive pairs, replay held near its start, known documents kept on their side.

    All of these are opt-in. `params` may be AdamW parameter groups, when gates and weights take different rates. `anchor` is a
    frozen teacher to hold to in place of this model's own start: {"base": real score per replay document, "cut", "med": its
    own, "h0": its final states, for klvec}. cfg "z_units" {"cut", "med", "tau"} measures every loss in z = (s - cut) / (med - cut),
    with this model's human 1% and median, margins at 0 and lesson, pair and separation terms over tau. Feedback rows naming a
    "share" are sampled in equal shares, and rows without one then feed the pairs only. hold_kind "oneside" is `keep_loss` around
    each stratum's running median drift, so a whole population sliding together costs nothing. cfg "train_budget" scores each step's
    documents together under that many padded tokens. `on_step(step, opt, losses)` runs after each update, gradients in place."""
    import torch

    budget = cfg.get("score_budget", 40000)
    if anchor is None:
        base = torch.from_numpy(score_all(model, replay, pair, device, amp, budget)).float().to(device)
    else:
        base = torch.from_numpy(np.asarray(anchor["base"], dtype=np.float32)).to(device)
        assert len(base) == len(replay)
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    flat = [w for g in params for w in g["params"]] if isinstance(params[0], dict) else params
    rng = random.Random(cfg.get("seed", 20261031))
    steps, nf, nr = cfg.get("steps", 200), cfg.get("feedback_batch", 8), cfg.get("replay_batch", 16)
    zu = cfg.get("z_units")
    unit = (lambda s, c=zu: (s - c["cut"]) / (c["med"] - c["cut"]) / zu["tau"]) if zu else (lambda s: s)
    if zu:
        base, cutoff = unit(base, anchor or zu), 0.0
    margin, hold = cfg.get("margin", 0.0 if zu else 1.0), cfg.get("replay_weight", 1.0)
    pair_margin, pair_weight = cfg.get("pair_margin", 0.0 if zu else 2.0), cfg.get("pair_weight", 1.0)
    by_pair, shares = {}, {}
    for r in feedback:
        if r.get("pair") is not None:
            by_pair.setdefault(r["pair"], {})[r["label"]] = r
        if r.get("share"):
            shares.setdefault(r["share"], []).append(r)
    pairs = [(v["human"], v["ai"]) for v in by_pair.values() if len(v) == 2]
    pools = {}
    for i, g in enumerate(strata or ["all"] * len(replay)):
        pools.setdefault(g, []).append(i)
    names_s, names_f = sorted(pools), sorted(shares)
    guard = cfg.get("guard_weight", 0.0)
    log(f"learn: {sum(w.numel() for w in flat):,} parameters, {len(feedback)} feedback, {len(pairs)} pairs, {len(replay)} replay")
    kind, cap = cfg.get("hold_kind"), cfg.get("train_budget")
    vec = kind == "klvec"
    if vec:
        if anchor is not None and "h0" in anchor:
            h0 = torch.from_numpy(np.asarray(anchor["h0"], dtype=np.float32)).to(device)
        else:
            hs = []
            with torch.no_grad():
                for b in batches(replay, budget):
                    hs.append((b, vec_scores(model, [replay[i] for i in b], pair, device, amp)[1]))
            h0 = torch.zeros(len(replay), hs[0][1].shape[1], device=device)
            for b, h in hs:
                h0[b] = h
        y0 = torch.tensor([1.0 if g.startswith("human") else -1.0 for g in strata], device=device)
        right = ((base > cutoff).float() * 2 - 1) == y0
        centre = {}
        for v, cls in ((1.0, y0 > 0), (-1.0, y0 < 0)):
            good = right & cls
            centre[v] = h0[good if good.any() else cls].mean(0)
        log(f"learn: vector KL, {int(right.sum())} of {len(replay)} replay documents start right")
    tau, recent = zu["tau"] if zu else cfg.get("keep_tau", 0.5), {g: deque(maxlen=cfg.get("keep_window", 64)) for g in names_s}
    hist, conflicts = [], 0
    for step in range(steps):
        fb = [r for g in names_f for r in rng.sample(shares[g], min(nf // len(names_f), len(shares[g])))] if shares else rng.sample(feedback, min(nf, len(feedback)))
        ps = rng.sample(pairs, min(nf // 2, len(pairs))) if pairs else []
        idx = []
        for k in range(nr):
            idx.append(rng.choice(pools[names_s[(step + k) % len(names_s)]]))
        fd, hd, ad, rd = [r["tokens"] for r in fb], [h["tokens"] for h, _ in ps], [a["tokens"] for _, a in ps], [replay[i] for i in idx]
        if cap:
            # one pass, longest first: padding is what a step spends its time on, and four small batches each pad to their own longest
            sizes = [len(fd), len(hd), len(ad), len(rd)]
            out = capped(vec_scores if vec else scores, model, fd + hd + ad + rd, pair, device, amp, cap)
            s, sh, sa, sr = (out[0] if vec else out).split(sizes)
            hf, _, _, hr = out[1].split(sizes) if vec else (None,) * 4
        else:
            s, hf = vec_scores(model, fd, pair, device, amp) if vec else (scores(model, fd, pair, device, amp), None)
            if ps:
                sh, sa = scores(model, hd, pair, device, amp), scores(model, ad, pair, device, amp)
            sr, hr = vec_scores(model, rd, pair, device, amp) if vec else (scores(model, rd, pair, device, amp), None)
        s, sr = unit(s), unit(sr)
        y = torch.tensor([1.0 if r["label"] == "human" else -1.0 for r in fb], device=device)
        loss_fb = margin_loss(s, y, cutoff, margin)
        if vec:
            loss_fb = loss_fb + kl_vector(model, hf, torch.stack([centre[float(v)] for v in y]), cfg.get("kl_temp", 4.0))
        loss_pair = torch.zeros((), device=device)
        if ps:
            loss_pair = pair_loss(unit(sh), unit(sa), pair_margin)
        yr = torch.tensor([1.0 if strata[i].startswith("human") else -1.0 for i in idx], device=device) if kind in ("kl", "oneside") else None
        if vec:
            tg = torch.stack([h0[i] if right[i] else centre[float(y0[i])] for i in idx])
            loss_hold = kl_vector(model, hr, tg, cfg.get("kl_temp", 4.0))
        elif kind == "kl":
            loss_hold = kl_hold(sr, base[idx], yr, cutoff, cfg.get("kl_temp", 1.0))
        elif kind == "oneside":
            drift = (sr - base[idx]) * (tau if zu else 1.0)
            for i, d in zip(idx, drift.tolist()):
                recent[strata[i]].append(d)
            c = torch.tensor([0.0 if cfg.get("keep_abs") else float(np.median(recent[strata[i]])) for i in idx], device=device)
            loss_hold = keep_loss(drift - c, yr, tau)
        else:
            loss_hold = ((sr - base[idx]) ** 2).mean()
        new = loss_fb + pair_weight * loss_pair
        loss = new + hold * loss_hold
        if cfg.get("batch_pairs") and strata:
            # every human against every AI document in the batch: right scores pull apart, however few pairs the feedback holds
            hs = [k for k, i in enumerate(idx) if strata[i].startswith(cfg.get("sep_humans", "human"))]
            ai = [k for k, i in enumerate(idx) if strata[i].startswith("ai")]
            if hs and ai:
                gap = sr[hs][:, None] - sr[ai][None, :]
                loss = loss + cfg["batch_pairs"] * torch.nn.functional.softplus(pair_margin - gap).mean()
        sides = [(k, 1.0 if strata[i].startswith("human") else -1.0) for k, i in enumerate(idx) if strata and strata[i].startswith(("human", "ai"))]
        if guard and sides:
            ks = [k for k, _ in sides]
            loss = loss + guard * margin_loss(sr[ks], torch.tensor([y for _, y in sides], device=device), cutoff, margin)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"loss is not finite at step {step}")
        opt.zero_grad()
        if cfg.get("project"):
            # A-GEM: where the new lesson would hurt what must be kept, remove that part and move along the plane that does not
            old = loss - new
            g_new = torch.autograd.grad(new, flat, retain_graph=True, allow_unused=True)
            g_old = torch.autograd.grad(old, flat, allow_unused=True)
            g_new = [torch.zeros_like(p) if g is None else g for g, p in zip(g_new, flat)]
            g_old = [torch.zeros_like(p) if g is None else g for g, p in zip(g_old, flat)]
            dot = sum((a * b).sum() for a, b in zip(g_new, g_old))
            if dot < 0:
                ratio = dot / sum((b * b).sum() for b in g_old).clamp_min(1e-30)
                g_new = [a - ratio * b for a, b in zip(g_new, g_old)]
                conflicts += 1
            for p, a, b in zip(flat, g_new, g_old):
                p.grad = a + b
        else:
            loss.backward()
        opt.step()
        hist.append((loss_fb.item(), loss_pair.item(), loss_hold.item()))
        if step % 10 == 0:
            r = hist[-10:]
            log(f"learn step {step}/{steps} fb {sum(x[0] for x in r) / len(r):.4f} pair {sum(x[1] for x in r) / len(r):.4f} hold {sum(x[2] for x in r) / len(r):.4f}")
        if on_step:
            on_step(step, opt, hist[-1])
    if cfg.get("project"):
        log(f"learn: the new lesson conflicted with protected documents on {conflicts} of {steps} steps")
    return hist


def learn(art_path: str, config_dir: str, feedback: list[dict], replay: list[np.ndarray], pair: tuple[int, int],
          cutoff: float, out: str, cfg: dict, log=print, device: str | None = None, strata: list[str] | None = None) -> dict:
    """Train on `feedback` rows {"tokens", "label": "human"|"ai", "pair": group or None}, write `out`, return the lineage.
    `replay` is ordinary documents held near their starting scores. With `strata`, one name per replay document,
    each population gets the same share of every batch, and every "human..." or "ai..." document is also kept on its
    side of the cutoff: a mean penalty alone lets a small population fall through it."""
    import torch

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.startswith("cuda")
    art = lattice.Artifact(art_path)
    names = train_names(art, cfg)
    model, grids = build_student(art, config_dir, names, cfg.get("ggml_lib"), device)
    if cfg.get("checkpointing", True) and amp:
        model.gradient_checkpointing_enable()
        model.train()    # HF checkpoints only in training mode; the model has no dropout
    params = [w for w, _, _ in grids.values()]
    hist = _train(model, params, feedback, replay, pair, cutoff, cfg, log, device, amp, strata, cfg.get("lr", 2e-5))
    levels, moved = {}, {}
    for name, (w, fq, start) in grids.items():
        lv = fq.levels(w.detach()).reshape(-1).to("cpu", torch.uint8).numpy()
        levels[name], moved[name] = lv, int((lv != start.numpy()).sum())
    lattice.write_levels(art_path, out, levels)
    art.close()
    n = sum(g[0].numel() for g in grids.values())
    rec = {"method": "grid", "steps": cfg.get("steps", 200), "lr": cfg.get("lr", 2e-5), "tensors": names, "weights": n, "levels_moved": sum(moved.values()),
           "fraction_moved": sum(moved.values()) / max(1, n), "loss_first": hist[0] if hist else None, "loss_last": hist[-1] if hist else None,
           "n_feedback": len(feedback), "n_pairs": len({r["pair"] for r in feedback if r.get("pair") is not None}), "n_replay": len(replay),
           "base_sha256": sha256(art_path), "out_sha256": sha256(out), "bytes": Path(out).stat().st_size}
    Path(str(out) + ".lineage.json").write_text(json.dumps(rec, indent=1))
    return rec


def learn_gates(art_path: str, config_dir: str, feedback: list[dict], replay: list[np.ndarray], pair: tuple[int, int],
                cutoff: float, out: str, cfg: dict, log=print, device: str | None = None, strata: list[str] | None = None) -> dict:
    """Weights stay fixed and only one on/off switch per kept neuron learns, by a straight-through gradient. A switched-off
    neuron is written as its outgoing column set as near zero as the grid allows, so the real file can be judged as usual."""
    import torch

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.startswith("cuda")
    art = lattice.Artifact(art_path)
    model, _ = build_student(art, config_dir, [], cfg.get("ggml_lib"), device)
    if cfg.get("checkpointing", True) and amp:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.train()
    layers = [il for il in range(len(model.model.layers)) if art.tensors.get(f"blk.{il}.ffn_down.weight", {}).get("type") in lattice.TOP]
    theta = {il: torch.nn.Parameter(torch.ones(art.shape(f"blk.{il}.ffn_down.weight")[1], device=device)) for il in layers}

    def hook(il):
        def f(mod, args):
            s = torch.sigmoid(theta[il])
            g = (theta[il] > 0).float() + s - s.detach()
            return (args[0] * g.to(args[0].dtype),)
        return f

    hooks = [model.model.layers[il].mlp.down_proj.register_forward_pre_hook(hook(il)) for il in layers]
    params = list(theta.values())
    hist = _train(model, params, feedback, replay, pair, cutoff, cfg, log, device, amp, strata, cfg.get("gate_lr", 0.03))
    for h in hooks:
        h.remove()
    levels, off, bits = {}, {}, 0
    for il in layers:
        name, shape = f"blk.{il}.ffn_down.weight", art.shape(f"blk.{il}.ffn_down.weight")
        dead = np.flatnonzero(theta[il].detach().cpu().numpy() <= 0)
        off[il], bits = dead.tolist(), bits + shape[1]
        if len(dead):
            g = art.grid(name)
            lv = g.levels.reshape(shape).copy()
            zero = np.clip(np.round(-g.lo / np.maximum(g.step, 1e-30)), 0, g.top).astype(np.uint8).reshape(shape)
            lv[:, dead] = zero[:, dead]
            levels[name] = lv.reshape(-1)
    lattice.write_levels(art_path, out, levels)
    art.close()
    n = sum(map(len, off.values()))
    rec = {"method": "gates", "steps": cfg.get("steps", 200), "gate_lr": cfg.get("gate_lr", 0.03), "neurons": bits, "off": n, "fraction_moved": n / max(1, bits), "off_by_layer": {str(k): len(v) for k, v in off.items()},
           "loss_first": hist[0] if hist else None, "loss_last": hist[-1] if hist else None, "n_feedback": len(feedback), "n_replay": len(replay),
           "base_sha256": sha256(art_path), "out_sha256": sha256(out), "bytes": Path(out).stat().st_size}
    Path(str(out) + ".lineage.json").write_text(json.dumps(rec, indent=1))
    Path(str(out) + ".switches.json").write_text(json.dumps({str(k): v for k, v in off.items()}))
    return rec


def merge(base: str, a: str, b: str, mode: str, out: str) -> dict:
    """Combine two models learned from the same base file. Each is a list of level changes on the same grid, so merging is exact:
    "union" keeps every change and leaves a weight at base where the two disagree in direction; "agreement" keeps only changes
    both made in the same direction, at the smaller size. Any other mode is a JSON map of tensor name to "a" or "b"."""
    A, B, C = lattice.Artifact(a), lattice.Artifact(b), lattice.Artifact(base)
    pick = json.loads(mode) if mode.startswith("{") else None
    levels, kept, clash, total = {}, 0, 0, 0
    for name, t in C.tensors.items():
        if t["type"] not in lattice.TOP:
            continue
        l0 = C.grid(name).levels.astype(np.int16)
        da, db = A.grid(name).levels.astype(np.int16) - l0, B.grid(name).levels.astype(np.int16) - l0
        if not da.any() and not db.any():
            continue
        same = (np.sign(da) == np.sign(db)) & (da != 0)
        opposite = (np.sign(da) * np.sign(db)) < 0
        if pick is not None:
            d = {"a": da, "b": db}[pick.get(name, "a")]
        elif mode == "union":
            d = np.where(same, np.sign(da) * np.maximum(np.abs(da), np.abs(db)), np.where(opposite, 0, da + db))
        elif mode == "agreement":
            d = np.where(same, np.sign(da) * np.minimum(np.abs(da), np.abs(db)), 0)
        else:
            raise ValueError(mode)
        levels[name] = (l0 + d).astype(np.uint8)
        kept, clash, total = kept + int((d != 0).sum()), clash + int(opposite.sum()), total + l0.size
    lattice.write_levels(base, out, levels)
    for x in (A, B, C):
        x.close()
    rec = {"method": "merge", "mode": mode, "a": a, "b": b, "levels_kept": kept, "levels_clashing": clash, "weights": total,
           "base_sha256": sha256(base), "out_sha256": sha256(out), "bytes": Path(out).stat().st_size}
    Path(str(out) + ".lineage.json").write_text(json.dumps(rec, indent=1))
    return rec


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def _side(task: str, prompt_id: str | None = None) -> int:
    """Select documents split in two by task id: side 0 may be replayed in training, side 1 alone judges. With `prompt_id`, by
    prompt, so one Beemo prompt's human, generated and edited texts cannot land on both sides."""
    return int(hashlib.sha256((task if prompt_id is None else str(prompt_id)).encode()).hexdigest()[:8], 16) % 2


def row_side(r: dict, c: dict) -> int:
    """`_side` of a select row. With `side_by_prompt` every Beemo row must carry `prompt_id`: Beemo task ids do not hold it."""
    return _side(r["task"], r["prompt_id"] if c.get("side_by_prompt") and r["task"].startswith("beemo") else None)


def real_scores(model: str, c: dict, scorer) -> tuple[np.ndarray, np.ndarray]:
    """The real file's scores on the pool and the feedback, once per file."""
    pair = tuple(c["pair"])
    quick = c.get("quick_idx") and sha256(model) != sha256(c["model"])
    cache = Path(c.get("cache_dir", c["out_dir"])) / f"scores-{sha256(model)[:16]}{'-quick' if quick else ''}.npz"
    if cache.exists():
        z = np.load(cache)
        return z["pool"], z["held"]
    if quick:
        # a screen, not a verdict: only documents that can cross the cutoff are scored, the rest keep the base file's scores
        pool = real_scores(c["model"], c, scorer)[0].copy()
        pool[np.load(c["quick_idx"])] = scorer.score_trj(model, c["quick_trj"], pair)
    else:
        pool = scorer.score_trj(model, c["pool_trj"], pair)
    held = scorer.score_trj(model, c["feedback_trj"], pair)
    np.savez(cache, pool=pool, held=held)
    return pool, held


def write_trj(docs: list[np.ndarray], path: str, target: int) -> None:
    """Documents as a TRJ1 file with one target each, so read_trj returns the same tokens."""
    with open(path, "wb") as f:
        f.write(b"TRJ1" + struct.pack("<I", len(docs)))
        for d in docs:
            x = d.tolist() + [target]
            f.write(struct.pack("<II", len(x), 1) + struct.pack(f"<{len(x)}i", *x) + struct.pack("<I", len(d)))


def make_panel(c: dict, rows: list[dict], humans: list[int] | None = None) -> None:
    """Write the fixed panel once, to `panel_trj` and `panel_idx`: `panel_humans` random calibration humans from `humans` (default
    all of calibration), then all of select side 1. Pool indices are kept in `panel_idx`, and no document is chosen by its score."""
    ncal = c["n_calibration"]
    rng = random.Random(c.get("seed", 20261031))
    idx = sorted(rng.sample(range(ncal) if humans is None else humans, c.get("panel_humans", 4000)))
    idx += [ncal + i for i, r in enumerate(rows) if row_side(r, c) == 1]
    pool = read_trj(c["pool_trj"])
    assert len(pool) == ncal + len(rows)
    write_trj([pool[i] for i in idx], c["panel_trj"], c["pair"][0])
    np.save(c["panel_idx"], np.array(idx))


def real_panel(model: str, c: dict, scorer) -> np.ndarray:
    """The real file's scores on the fixed panel, in `panel_idx` order, once per file. Beside the quick screen, not a form of it:
    nothing is imputed from a base file, so it stays honest for a file far from any base and a cutoff may be read from it."""
    cache = Path(c.get("cache_dir", c["out_dir"])) / f"panel-{sha256(model)[:16]}-{sha256(c['panel_trj'])[:8]}.npy"
    if cache.exists():
        return np.load(cache)
    v = scorer.score_trj(model, c["panel_trj"], tuple(c["pair"]))
    tmp = cache.with_suffix(".part.npy")
    np.save(tmp, v)
    tmp.replace(cache)
    return v


def judge(model: str, c: dict, scorer, rows: list[dict], fb: list[dict], hold: list[int]) -> dict:
    """The real built file on the gate side of select and on held-out feedback, at the file's own 1% cutoff."""
    from . import stats
    pair, ncal = tuple(c["pair"]), c["n_calibration"]
    pool, held = real_scores(model, c, scorer)
    cut = float(np.sort(pool[:ncal])[int(.01 * ncal)])
    gate = [i for i, r in enumerate(rows) if row_side(r, c) == 1]
    s, R = pool[ncal:], []
    for i in gate:
        r = rows[i]
        ok = bool((s[i] < cut) == (r["verdict"] == "ai"))
        R += [{"task": r["task"], "family": r["family"], "side": "cand", "success": ok},
              {"task": r["task"], "family": r["family"], "side": "src", "success": r["src"]}]
    b = stats.delta_and_bounds(R, .05 / 4)
    def rate(ai: bool, beemo: bool) -> float:
        x = [s[i] < cut for i in gate if (rows[i]["verdict"] == "ai") == ai and rows[i]["task"].startswith("beemo") == beemo]
        return float(np.mean(x)) if x else float("nan")
    out = {"model": model, "bytes": Path(model).stat().st_size, "cutoff": cut, "n_gate": len(gate), "delta": b["delta"], "L": b["L"], "U": b["U"],
           "gate_pass": bool(b["L"] > -c.get("gate_margin", .02)), "caught_wo": rate(True, False), "caught_beemo": rate(True, True),
           "human_flagged_wo": rate(False, False), "human_flagged_beemo": rate(False, True)}
    h = [fb[i] for i in hold]
    v = {r["index"]: held[r["index"]] for r in h}
    ai = [v[r["index"]] for r in h if r["label"] == "ai"]
    hu = [v[r["index"]] for r in h if r["label"] == "human"]
    g = {}
    for r in h:
        if r.get("pair") is not None:
            g.setdefault(r["pair"], {})[r["label"]] = v[r["index"]]
    gaps = [p["human"] - p["ai"] for p in g.values() if len(p) == 2]
    out["held"] = {"n_ai": len(ai), "n_human": len(hu), "ai_flagged": int(sum(x < cut for x in ai)), "human_flagged": int(sum(x < cut for x in hu)),
                   "pairs": len(gaps), "pairs_right": int(sum(x > 0 for x in gaps)), "mean_gap": float(np.mean(gaps)) if gaps else None}
    return out


def run(cfg: dict, fold: int | None = None, log=print) -> dict:
    """`chantless.cli learn`: train on feedback, build by writing levels, judge the real file, accept or reject."""
    from .runtime import BatchScorer, Toolchain

    c = cfg["learn"]
    out_dir = Path(c["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    pair, ncal = tuple(c["pair"]), c["n_calibration"]
    rows = json.loads(Path(c["select_rows"]).read_text())
    fb = json.loads(Path(c["feedback_rows"]).read_text())
    train = [i for i, r in enumerate(fb) if fold is None or r.get("fold") != fold]
    hold = [i for i, r in enumerate(fb) if fold is None or r.get("fold") == fold]
    fdocs, pool = read_trj(c["feedback_trj"]), read_trj(c["pool_trj"])
    assert len(pool) == ncal + len(rows)
    rng = random.Random(c.get("seed", 20261031))
    side0 = [ncal + i for i, r in enumerate(rows) if row_side(r, c) == 0]
    picks = rng.sample(range(ncal), c.get("replay_humans", 1000)) + rng.sample(side0, c.get("replay_select", 1000))
    strata = None
    if c.get("stratify"):
        kind = lambda i: ("beemo" if rows[i - ncal]["task"].startswith("beemo") else "wo", "human" if rows[i - ncal]["verdict"] != "ai" else "ai")
        beemo = [i for i in side0 if kind(i)[0] == "beemo"]
        wo = [i for i in side0 if kind(i)[0] == "wo"]
        picks = rng.sample(range(ncal), c.get("replay_humans", 1000)) + beemo + rng.sample(wo, c.get("replay_select", 1000))
        strata = ["human-cal"] * c.get("replay_humans", 1000) + [f"{kind(i)[1]}-{kind(i)[0]}" for i in picks[c.get("replay_humans", 1000):]]
    scorer = BatchScorer(Toolchain.from_config(cfg["toolchain"]), c.get("slots_per_gpu", 2))
    if c.get("tails"):
        # the documents that decide the operating point: the humans who set the 1% cutoff, and the AI just under it
        p, _ = real_scores(c["model"], c, scorer)
        tail = np.argsort(p[:ncal])[:c.get("tail_n", 500)].tolist()
        ai = [i for i in side0 if rows[i - ncal]["verdict"] == "ai"]
        near = sorted(ai, key=lambda i: -p[i])[:c.get("near_n", 300)]
        picks += tail + near
        strata += ["human-tail"] * len(tail) + ["ai-near"] * len(near)
    feedback = [dict(fb[i], tokens=fdocs[fb[i]["index"]]) for i in train]
    tag = "all" if fold is None else f"fold{fold}"
    out = str(out_dir / f"model-{tag}.gguf")
    trainer = learn_gates if c.get("method") == "gates" else learn
    rec = trainer(c["model"], c["config_dir"], feedback, [pool[i] for i in picks], pair, c["cutoff"], out, c, log, strata=strata)
    base = judge(c["model"], c, scorer, rows, fb, hold)
    cand = judge(out, c, scorer, rows, fb, hold)
    rise = cand["human_flagged_beemo"] - base["human_flagged_beemo"]
    better = hold and cand["held"]["pairs"] and cand["held"]["mean_gap"] > base["held"]["mean_gap"]
    drop = max(base["caught_wo"] - cand["caught_wo"], base["caught_beemo"] - cand["caught_beemo"])    # the pooled gate hides Beemo
    accepted = bool(cand["gate_pass"] and rise <= c.get("max_human_rise", .01) and drop <= c.get("max_ai_drop", .02) and better)
    res = {"accepted": accepted, "fold": fold, "lineage": rec, "base": base, "candidate": cand, "claim": "preview on one side of select; not a Chantless exam result"}
    (out_dir / f"result-{tag}.json").write_text(json.dumps(res, indent=1))
    log(f"learn {tag}: accepted={accepted} gate L {cand['L']:.4f} held pairs right {base['held']['pairs_right']} -> {cand['held']['pairs_right']}, "
        f"flagged AI {base['held']['ai_flagged']} -> {cand['held']['ai_flagged']}, Beemo humans {base['human_flagged_beemo']:.3f} -> {cand['human_flagged_beemo']:.3f}")
    return res
