"""Few-shot specialist adaptation via fixed-grid level updates and binary MLP neuron switches.

`Student` reconstructs a built GGUF in PyTorch with its exact removed sub-blocks, cut neurons, and trimmed
vocabulary. Adaptation optimizes either quantized grid levels (`lattice.py`) or universal `+0 B` binary
`up_row_zero` MLP neuron switches (`learn_gates`, `learn_gates_seq`) using margin/NLL losses with `replay_h0`
token-KL retention and A-GEM gradient projection, writes back the GGUF/mask, and evaluates the built file.
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

from . import lattice, mask as maskmod

LAYER_NAMES = {
    "attn_norm": "input_layernorm.weight", "post_attention_norm": "post_attention_layernorm.weight",
    "ffn_norm": "post_attention_layernorm.weight", "post_ffw_norm": "post_feedforward_layernorm.weight",
    "ffn_gate": "mlp.gate_proj.weight", "ffn_up": "mlp.up_proj.weight", "ffn_down": "mlp.down_proj.weight",
    "attn_q": "self_attn.q_proj.weight", "attn_k": "self_attn.k_proj.weight", "attn_v": "self_attn.v_proj.weight",
    "attn_output": "self_attn.o_proj.weight", "attn_q_norm": "self_attn.q_norm.weight", "attn_k_norm": "self_attn.k_norm.weight",
    "attn_qkv": "linear_attn.in_proj_qkv.weight", "attn_gate": "linear_attn.in_proj_z.weight",
    "ssm_alpha": "linear_attn.in_proj_a.weight", "ssm_beta": "linear_attn.in_proj_b.weight",
    "ssm_out": "linear_attn.out_proj.weight", "ssm_norm": "linear_attn.norm.weight",
    "ssm_conv1d": "linear_attn.conv1d.weight", "ssm_a": "linear_attn.A_log", "ssm_dt": "linear_attn.dt_bias",
    "inp_gate": "per_layer_input_gate.weight", "proj": "per_layer_projection.weight",
    "post_norm": "post_per_layer_input_norm.weight", "layer_output_scale": "layer_scalar",
}
TOPLEVEL_NAMES = {
    "token_embd.weight": "model.embed_tokens.weight",
    "output_norm.weight": "model.norm.weight",
    "output.weight": "lm_head.weight",
    "per_layer_token_embd.weight": "model.embed_tokens_per_layer.weight",
    "per_layer_model_proj.weight": "model.per_layer_model_projection.weight",
    "per_layer_proj_norm.weight": "model.per_layer_projection_norm.weight",
}

_GGML_LIBS_CACHE: dict[str, list[ctypes.CDLL]] = {}


def _load_ggml_libs(ggml_lib: str | None) -> list[ctypes.CDLL]:
    if not ggml_lib:
        raise ValueError("ggml_lib path (directory or shared library) is required to dequantize non-grid tensors")
    key = str(Path(ggml_lib).resolve())
    if key in _GGML_LIBS_CACHE:
        return _GGML_LIBS_CACHE[key]
    p = Path(key)
    lib_dir = p if p.is_dir() else p.parent
    cands: list[Path] = ([] if p.is_dir() else [p])
    for stem in ("libggml-base", "libggml-cpu", "libggml", "libllama"):
        for ext in (".so", ".dylib"):
            cand = lib_dir / f"{stem}{ext}"
            if cand.exists() and cand not in cands:
                cands.append(cand)
    loaded: list[ctypes.CDLL] = []
    for c in cands:
        try:
            loaded.append(ctypes.CDLL(str(c), mode=getattr(ctypes, "RTLD_GLOBAL", 0)))
        except OSError:
            loaded.append(ctypes.CDLL(str(c)))
    # Initialize IQ2_XS / IQ1_S lookup grids once before calling `dequantize_row_*`
    for lib in loaded:
        init_fn = getattr(lib, "llama_backend_init", None) or getattr(lib, "ggml_cpu_init", None)
        if init_fn is not None:
            init_fn.argtypes, init_fn.restype = [], None
            init_fn()
            break
    _GGML_LIBS_CACHE[key] = loaded
    return loaded


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


def hf_name(gg: str, is_gemma4: bool = False) -> str:
    if gg in TOPLEVEL_NAMES:
        return TOPLEVEL_NAMES[gg]
    p = gg.split(".")
    if is_gemma4 and p[2] == "ffn_norm":
        return f"model.layers.{p[1]}.pre_feedforward_layernorm.weight"
    return f"model.layers.{p[1]}.{LAYER_NAMES[p[2]]}"


def floats(art: lattice.Artifact, name: str, ggml_lib: str | None) -> np.ndarray:
    """A tensor as float32 in torch shape. Tiers without a grid are decoded by ggml itself."""
    t, shape = art.tensors[name], art.shape(name)
    if t["type"] == "F32":
        return np.frombuffer(art.raw(name), dtype="<f4").reshape(shape)
    if t["type"] == "F16":
        return np.frombuffer(art.raw(name), dtype="<f2").astype(np.float32).reshape(shape)
    if t["type"] == "BF16":    # the top half of a float32; ggml has no dequantize_row_bf16
        return (np.frombuffer(art.raw(name), dtype="<u2").astype("<u4") << 16).view("<f4").reshape(shape)
    if t["type"] in lattice.TOP:
        return art.grid(name).values().reshape(shape)
    low = t["type"].lower()
    syms = [f"dequantize_row_{low}"]
    if t["type"].endswith("_K"):
        syms.append(f"dequantize_row_{low[:-1]}K")
    fn = None
    for lib in _load_ggml_libs(ggml_lib):
        for s in syms:
            fn = getattr(lib, s, None)
            if fn is not None:
                break
        if fn is not None:
            break
    if fn is None:
        raise RuntimeError(f"could not find {syms[0]} in {ggml_lib}")
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
                  latents: dict[str, np.ndarray] | None = None, frozen_bf16: bool = False):
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
    if hasattr(conf, "get_text_config"):
        conf = conf.get_text_config()
    elif hasattr(conf, "text_config"):
        conf = conf.text_config
    conf.vocab_size = art.tensors["token_embd.weight"]["dims"][1]
    if "per_layer_token_embd.weight" in art.tensors and hasattr(conf, "vocab_size_per_layer_input"):
        conf.vocab_size_per_layer_input = art.tensors["per_layer_token_embd.weight"]["dims"][1]
    conf.use_cache = False
    is_gemma4 = getattr(art.h, "arch", "") == "gemma4" or "per_layer_token_embd.weight" in art.tensors or any(n.endswith(".post_ffw_norm.weight") for n in art.tensors)
    model = AutoModelForCausalLM.from_config(conf, dtype=torch.float32)
    # `(1 + weight)` RMSNorm models (`Qwen3_5`) init `norm.weight` to zeros; standard `RMSNorm` (`Qwen3`, `Gemma4`) inits to ones.
    norm_offset = 1.0 if float(model.model.norm.weight.detach().abs().max()) == 0.0 else 0.0
    num_k_heads = int(getattr(conf, "linear_num_key_heads", 0) or 0)
    num_v_heads = int(getattr(conf, "linear_num_value_heads", 0) or 0)
    head_k_dim = int(getattr(conf, "linear_key_head_dim", 0) or 0)
    head_v_dim = int(getattr(conf, "linear_value_head_dim", 0) or 0)
    unreorder_gdn_v = (num_k_heads > 0 and num_v_heads > 0 and num_k_heads != num_v_heads
                       and getattr(art.h, "arch", "") in ("qwen35", "qwen3next"))
    num_v_per_k = (num_v_heads // num_k_heads) if unreorder_gdn_v else 1
    qk_half = head_k_dim * num_k_heads

    def _unreorder_v(t, dim: int, hdim: int):
        sh = list(t.shape)
        if dim < 0:
            dim += len(sh)
        ns = sh[:dim] + [num_v_per_k, num_k_heads, hdim] + sh[dim + 1:]
        perm = list(range(len(ns)))
        perm[dim], perm[dim + 1] = perm[dim + 1], perm[dim]
        return t.reshape(*ns).permute(*perm).contiguous().reshape(*sh)

    for il, layer in enumerate(model.model.layers):
        if f"blk.{il}.ffn_down.weight" not in art.tensors:
            layer.mlp = Zero(False)
        else:
            for nm, gg in (("gate_proj", "ffn_gate"), ("up_proj", "ffn_up"), ("down_proj", "ffn_down")):
                rows, cols = art.shape(f"blk.{il}.{gg}.weight")
                setattr(layer.mlp, nm, nn.Linear(cols, rows, bias=False))
        if not ({f"blk.{il}.ssm_out.weight", f"blk.{il}.attn_output.weight"} & art.tensors.keys()):
            if getattr(layer, "block_type", "attention") == "linear_attention":
                layer.linear_attn = Zero(False)
            else:
                layer.self_attn = Zero(True)
        if hasattr(layer, "per_layer_input_gate") and f"blk.{il}.inp_gate.weight" not in art.tensors:
            layer.hidden_size_per_layer_input = 0
    params = dict(model.named_parameters())
    buffers = dict(model.named_buffers())
    for name in art.tensors:
        if name == "rope_freqs.weight":
            continue
        hfn = hf_name(name, is_gemma4=is_gemma4)
        if hfn == "lm_head.weight" and hfn not in params:
            model.lm_head = nn.Linear(art.shape(name)[1], art.shape(name)[0], bias=False)
            params[hfn] = model.lm_head.weight
        v = torch.from_numpy(floats(art, name, ggml_lib).copy())
        if hfn in buffers and hfn not in params:
            b = buffers[hfn]
            b.copy_(v.reshape(b.shape))
            continue
        p = params[hfn]
        # Match llama.cpp GGUF weight layout conversions (conv1d, A_log, RMSNorm offset, GDN V-head order)
        if "ssm_conv1d" in name:
            v = v.unsqueeze(1)
        elif name.endswith(".ssm_a"):
            v = torch.log(-v)
        elif name.endswith("norm.weight") and "ssm_norm" not in name and norm_offset != 0.0:
            v = v - norm_offset
        if unreorder_gdn_v and "linear_attn." in hfn:
            if ".in_proj_qkv." in hfn or ".conv1d." in hfn:
                v = torch.cat([v[: 2 * qk_half], _unreorder_v(v[2 * qk_half :], 0, head_v_dim)], dim=0)
            elif ".in_proj_z." in hfn:
                v = _unreorder_v(v, 0, head_v_dim)
            elif ".in_proj_a." in hfn or ".in_proj_b." in hfn:
                v = _unreorder_v(v, 0, 1)
            elif hfn.endswith(".A_log") or hfn.endswith(".dt_bias"):
                v = _unreorder_v(v.unsqueeze(-1), 0, 1).squeeze(-1)
            elif ".out_proj." in hfn:
                v = _unreorder_v(v, 1, head_v_dim)
        assert p.shape == v.shape, (name, p.shape, v.shape)
        p.data = v
    for p in model.parameters():
        p.requires_grad_(False)
    if "output.weight" not in art.tensors:
        model.lm_head.weight = model.model.embed_tokens.weight
    if frozen_bf16 and not train:
        for p in model.parameters():
            if p.ndim >= 2 and p.dtype == torch.float32:
                p.data = p.data.to(torch.bfloat16)
    model.to(device)
    mods = dict(model.named_modules())
    grids = {}
    for name in train:
        g = art.grid(name)
        path = hf_name(name, is_gemma4=is_gemma4).rsplit(".", 1)[0]
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


def scores(model, docs: list[np.ndarray], pair: tuple[int, int], device: str, amp: bool, prefix_cache=None):
    import torch
    if prefix_cache is not None:
        h_seq = prefix_cache.forward_tail(model, docs, device, amp, round_to=32)
        idx = torch.tensor([len(d) - 1 for d in docs], device=h_seq.device, dtype=torch.long)
        h = h_seq[torch.arange(len(docs), device=h_seq.device), idx]
        with torch.autocast(device.split(":")[0], dtype=torch.bfloat16, enabled=amp):
            x = _head_logits(model, h)[:, list(pair)]
        return x[:, 0] - x[:, 1]
    ids, m = collate(docs, device)
    with torch.autocast(device.split(":")[0], dtype=torch.bfloat16, enabled=amp):
        h = model.model(input_ids=ids, attention_mask=m, use_cache=False).last_hidden_state[:, -1]
        x = _head_logits(model, h)[:, list(pair)]
    return x[:, 0] - x[:, 1]


def _head_logits(model, h):
    import torch
    lg = model.lm_head(h.to(model.lm_head.weight.dtype)).float()
    cap = getattr(getattr(model, "config", None), "final_logit_softcapping", None)
    if cap:
        lg = torch.tanh(lg / float(cap)) * float(cap)
    return lg


def vec_scores(model, docs: list[np.ndarray], pair: tuple[int, int], device: str, amp: bool, prefix_cache=None):
    """The score and the final hidden state, from which the whole output distribution follows."""
    import torch
    if prefix_cache is not None:
        h_seq = prefix_cache.forward_tail(model, docs, device, amp, round_to=32)
        idx = torch.tensor([len(d) - 1 for d in docs], device=h_seq.device, dtype=torch.long)
        h = h_seq[torch.arange(len(docs), device=h_seq.device), idx]
        with torch.autocast(device.split(":")[0], dtype=torch.bfloat16, enabled=amp):
            x = _head_logits(model, h)[:, list(pair)]
        return x[:, 0] - x[:, 1], h.float()
    ids, m = collate(docs, device)
    with torch.autocast(device.split(":")[0], dtype=torch.bfloat16, enabled=amp):
        h = model.model(input_ids=ids, attention_mask=m, use_cache=False).last_hidden_state[:, -1]
        x = _head_logits(model, h)[:, list(pair)]
    return x[:, 0] - x[:, 1], h.float()


def capped(fn, model, docs: list[np.ndarray], pair: tuple[int, int], device: str, amp: bool, cap: int, prefix_cache=None):
    """`fn` (scores or vec_scores) in sub-batches of at most `cap` padded tokens, longest first, back in the order of `docs`.
    One graph spans them, so a single backward sees the same loss while padding and peak memory both fall."""
    import torch
    parts = batches(docs, cap, len(docs))
    out = [fn(model, [docs[i] for i in b], pair, device, amp, prefix_cache=prefix_cache) for b in parts]
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


def score_all(model, docs, pair, device, amp, budget=40000, prefix_cache=None) -> np.ndarray:
    import torch
    v = np.full(len(docs), np.nan)
    with torch.no_grad():
        for idx in batches(docs, budget):
            v[idx] = scores(model, [docs[i] for i in idx], pair, device, amp, prefix_cache=prefix_cache).cpu().numpy()
    assert np.isfinite(v).all()
    return v


def margin_loss(s, y, cutoff, margin):
    """Human documents (y = +1) must score above cutoff + margin, AI documents (y = -1) below cutoff - margin."""
    import torch
    return torch.nn.functional.softplus(margin - y * (s - cutoff)).mean()


def kl_vector(model, h, target_h, temp):
    """KL between whole output distributions, softened by `temp` so the mass beyond the two answer tokens counts."""
    import torch
    dt = model.lm_head.weight.dtype
    logq = torch.log_softmax(model.lm_head(h.to(dt)).float() / temp, -1)
    t = torch.softmax(model.lm_head(target_h.to(dt)).float() / temp, -1)
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


def _train(model, params, feedback, replay, pair, cutoff, cfg, log, device, amp, strata, lr, anchor=None, on_step=None, prefix_cache=None) -> list[tuple]:
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
        base = torch.from_numpy(score_all(model, replay, pair, device, amp, budget, prefix_cache=prefix_cache)).float().to(device)
    else:
        base = torch.from_numpy(np.asarray(anchor["base"], dtype=np.float32)).to(device)
        assert len(base) == len(replay)
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0, eps=float(cfg.get("adam_eps", 1e-8)))
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
    kind, cap = cfg.get("hold_kind"), cfg.get("train_budget", 3072 if amp else None)
    vec = kind == "klvec"
    if vec:
        if anchor is not None and "h0" in anchor:
            h0 = torch.from_numpy(np.asarray(anchor["h0"], dtype=np.float32)).to(device)
        else:
            hs = []
            with torch.no_grad():
                for b in batches(replay, budget):
                    hs.append((b, vec_scores(model, [replay[i] for i in b], pair, device, amp, prefix_cache=prefix_cache)[1]))
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
            # pack all sub-batches into one length-sorted pass to minimize padding
            sizes = [len(fd), len(hd), len(ad), len(rd)]
            out = capped(vec_scores if vec else scores, model, fd + hd + ad + rd, pair, device, amp, cap, prefix_cache=prefix_cache)
            s, sh, sa, sr = (out[0] if vec else out).split(sizes)
            hf, _, _, hr = out[1].split(sizes) if vec else (None,) * 4
        else:
            s, hf = vec_scores(model, fd, pair, device, amp, prefix_cache=prefix_cache) if vec else (scores(model, fd, pair, device, amp, prefix_cache=prefix_cache), None)
            if ps:
                sh, sa = scores(model, hd, pair, device, amp, prefix_cache=prefix_cache), scores(model, ad, pair, device, amp, prefix_cache=prefix_cache)
            sr, hr = vec_scores(model, rd, pair, device, amp, prefix_cache=prefix_cache) if vec else (scores(model, rd, pair, device, amp, prefix_cache=prefix_cache), None)
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
            # all-pairs human vs AI margin loss within the replay batch
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
            # A-GEM gradient projection against replay-hold gradients
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
        log(f"learn: A-GEM projected {conflicts}/{steps} steps")
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


def _apply_switches_to_file_and_mask(
    art: lattice.Artifact,
    art_path: str,
    out: str,
    layers: list[int],
    theta: dict,
    switch_mode: str,
    mask_in: dict | str | Path | None = None,
    mask_out: str | Path | None = None,
) -> tuple[dict[int, list[int]], int, dict[int, int], dict | None]:
    off: dict[int, list[int]] = {}
    widths_by_layer: dict[int, int] = {}
    bits = 0
    if switch_mode == "up_row_zero":
        row_zeros: dict[str, np.ndarray] = {}
        for il in layers:
            w_il = int(art.shape(f"blk.{il}.ffn_up.weight")[0])
            widths_by_layer[il] = w_il
            dead = np.flatnonzero(theta[il].detach().cpu().numpy() <= 0)
            off[il] = dead.tolist()
            bits += w_il
            if len(dead):
                for suf in ("ffn_up.weight", "ffn_gate.weight"):
                    tname = f"blk.{il}.{suf}"
                    if tname in art.tensors:
                        row_zeros[tname] = dead
        lattice.zero_rows(art_path, out, row_zeros)
    else:
        levels: dict[str, np.ndarray] = {}
        for il in layers:
            name, shape = f"blk.{il}.ffn_down.weight", art.shape(f"blk.{il}.ffn_down.weight")
            widths_by_layer[il] = int(shape[1])
            dead = np.flatnonzero(theta[il].detach().cpu().numpy() <= 0)
            off[il], bits = dead.tolist(), bits + int(shape[1])
            if len(dead):
                g = art.grid(name)
                lv = g.levels.reshape(shape).copy()
                zero = np.clip(np.round(-g.lo / np.maximum(g.step, 1e-30)), 0, g.top).astype(np.uint8).reshape(shape)
                lv[:, dead] = zero[:, dead]
                levels[name] = lv.reshape(-1)
        lattice.write_levels(art_path, out, levels)

    m_v3 = None
    if mask_in is not None:
        raw_m = maskmod.load(mask_in) if isinstance(mask_in, (str, Path)) else mask_in
        m_v3 = maskmod.upgrade_v2(raw_m)
        m_v3["apprentice"]["switch_mode"] = switch_mode
        m_v3["apprentice"]["switches"] = {
            str(il): maskmod.pack_bitmap(dead, widths_by_layer[il])
            for il, dead in sorted(off.items()) if len(dead) > 0
        }
        if mask_out is not None:
            Path(mask_out).write_text(json.dumps(m_v3))
    return off, bits, widths_by_layer, m_v3


def _switchable_layers(art: lattice.Artifact, n_layers: int, switch_mode: str, last_blocks: int | None = None) -> list[int]:
    if switch_mode == "up_row_zero":
        layers = [
            il for il in range(n_layers)
            if f"blk.{il}.ffn_up.weight" in art.tensors and f"blk.{il}.ffn_gate.weight" in art.tensors and f"blk.{il}.ffn_down.weight" in art.tensors
        ]
    else:
        layers = [il for il in range(n_layers) if art.tensors.get(f"blk.{il}.ffn_down.weight", {}).get("type") in lattice.TOP]
    if last_blocks is not None and last_blocks > 0:
        layers = layers[-int(last_blocks):]
    return layers


def _auto_checkpointing(model, cfg: dict, device: str) -> bool:
    if "checkpointing" in cfg:
        return bool(cfg["checkpointing"])
    if not device.startswith("cuda"):
        return False
    import torch
    if torch.cuda.is_available() and torch.cuda.get_device_properties(torch.device(device)).total_memory < (55 << 30):
        return True
    mtype = str(getattr(getattr(model, "config", None), "model_type", "")).lower()
    if mtype.startswith("qwen3_5"):
        try:
            import fla  # noqa: F401
        except ImportError:
            return True
    return False


def _should_stream_prefix(art: lattice.Artifact, cfg: dict, device: str, frozen_bf16: bool) -> bool:
    sp = cfg.get("stream_prefix", "auto")
    if sp is True:
        return True
    if sp is False or not device.startswith("cuda"):
        return False
    import math, torch
    if not torch.cuda.is_available():
        return False
    est_bytes = sum(math.prod(t["dims"]) for t in art.tensors.values()) * (2 if frozen_bf16 else 4)
    total_vram = torch.cuda.get_device_properties(torch.device(device)).total_memory
    return est_bytes > int(0.65 * total_vram)


def _default_seq_batch(model, device: str) -> int:
    if device.startswith("cuda") and str(getattr(getattr(model, "config", None), "model_type", "")).lower().startswith("qwen3_5"):
        try:
            import fla  # noqa: F401
        except ImportError:
            return 2
    return 4


def make_shuffled_classifier_labels(feedback: list[dict], seed: int = 20261118) -> list[dict]:
    """Build a non-degenerate random-label control for binary classifier feedback.

    When all feedback mistakes share a single class label, permuting them would reproduce the
    learned labels identically; fall back to balanced 50/50 random labels in that case.
    """
    import random as _rnd
    lbls = [r["label"] for r in feedback]
    if len(set(lbls)) <= 1:
        n_half = len(feedback) // 2
        pool = ["human"] * (len(feedback) - n_half) + ["ai"] * n_half
    else:
        pool = list(lbls)
    _rnd.Random(seed).shuffle(pool)
    return [{**r, "label": pool[i]} for i, r in enumerate(feedback)]


def learn_gates(art_path: str, config_dir: str, feedback: list[dict], replay: list[np.ndarray], pair: tuple[int, int] | None = None,
                cutoff: float = 0.0, out: str = "", cfg: dict | None = None, log=print, device: str | None = None,
                strata: list[str] | None = None, mask_in: dict | str | Path | None = None, mask_out: str | Path | None = None) -> dict:
    """Weights stay fixed and only one on/off switch per kept neuron learns, by a straight-through gradient.

    When `pair is None` or `cfg.get("task_kind") == "generative"`, dispatches to `learn_gates_seq` on sequence trajectories.
    With `switch_mode == "up_row_zero"` (default), switched-off neurons zero contiguous super-block rows in `ffn_up` & `ffn_gate`
    across ALL ggml quant types (`IQ1_S`, `IQ2_XS`, `Q2_K`..`Q8_0`, `BF16`) for exact `0.0f` parity with zero bytes added.
    """
    cfg = cfg or {}
    if pair is None or cfg.get("task_kind") == "generative":
        return learn_gates_seq(art_path, config_dir, feedback, replay, out, cfg, log=log, device=device, mask_in=mask_in, mask_out=mask_out)

    import torch

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.startswith("cuda")
    switch_mode = cfg.get("switch_mode", "up_row_zero")
    art = lattice.Artifact(art_path)
    frozen_bf16 = bool(cfg.get("frozen_bf16", amp))
    stream_pref = _should_stream_prefix(art, cfg, device, frozen_bf16)
    model, _ = build_student(art, config_dir, [], cfg.get("ggml_lib"), "cpu" if stream_pref else device, frozen_bf16=frozen_bf16)
    layers = _switchable_layers(art, len(model.model.layers), switch_mode, cfg.get("gate_last_blocks"))
    all_seqs = [r["tokens"] for r in feedback] + list(replay)
    prefix_cache = PrefixCache.build(model, all_seqs, min(layers), device, amp, cfg, log=log, stream_prefix=stream_pref) if layers else None
    if stream_pref and prefix_cache is None:
        model.to(device)
    if _auto_checkpointing(model, cfg, device) and amp:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.train()
    init_val = float(cfg.get("gate_init", 1.0))
    theta = {il: torch.nn.Parameter(torch.full((art.shape(f"blk.{il}.ffn_down.weight")[1],), init_val, device=device)) for il in layers}

    def hook(il):
        def f(mod, args):
            s = torch.sigmoid(theta[il])
            g = (theta[il] > 0).float() + s - s.detach()
            return (args[0] * g.to(args[0].dtype),)
        return f

    hooks = [model.model.layers[il].mlp.down_proj.register_forward_pre_hook(hook(il)) for il in layers]
    params = list(theta.values())
    hist = _train(model, params, feedback, replay, pair, cutoff, cfg, log, device, amp, strata, cfg.get("gate_lr", 0.03), prefix_cache=prefix_cache)
    for h in hooks:
        h.remove()
    off, bits, _, m_v3 = _apply_switches_to_file_and_mask(
        art, art_path, out, layers, theta, switch_mode,
        mask_in=mask_in or cfg.get("mask_in"), mask_out=mask_out or cfg.get("mask_out"),
    )
    art.close()
    del model
    import gc
    gc.collect()
    if amp:
        torch.cuda.empty_cache()
    n = sum(map(len, off.values()))
    rec = {"method": "gates", "switch_mode": switch_mode, "steps": cfg.get("steps", 200), "gate_lr": cfg.get("gate_lr", 0.03),
           "neurons": bits, "off": n, "fraction_moved": n / max(1, bits), "off_by_layer": {str(k): len(v) for k, v in off.items()},
           "loss_first": hist[0] if hist else None, "loss_last": hist[-1] if hist else None, "n_feedback": len(feedback), "n_replay": len(replay),
           "base_sha256": sha256(art_path), "out_sha256": sha256(out), "bytes": Path(out).stat().st_size}
    Path(str(out) + ".lineage.json").write_text(json.dumps(rec, indent=1))
    Path(str(out) + ".switches.json").write_text(json.dumps({str(k): v for k, v in off.items()}))
    if m_v3 is not None:
        rec["mask"] = m_v3
    return rec


def read_trj_multi(path: str | Path) -> list[tuple[np.ndarray, np.ndarray]]:
    """Read any single- or multi-target `TRJ1` file into `[(tokens_i32, targets_u32), ...]`."""
    out: list[tuple[np.ndarray, np.ndarray]] = []
    with open(path, "rb") as f:
        if f.read(4) != b"TRJ1":
            raise ValueError(f"{path}: bad TRJ1 magic")
        (n,) = struct.unpack("<I", f.read(4))
        for _ in range(n):
            nt, ng = struct.unpack("<II", f.read(8))
            tok = np.frombuffer(f.read(4 * nt), dtype=np.int32).copy()
            tgt = np.frombuffer(f.read(4 * ng), dtype=np.uint32).copy()
            out.append((tok, tgt))
    return out


def _norm_seq_lesson(r: dict) -> dict:
    if "pos_tokens" in r and "pos_targets" in r:
        pos_tok = np.asarray(r["pos_tokens"], dtype=np.int64)
        pos_tgt = np.asarray(r["pos_targets"], dtype=np.int64)
        neg_tok = np.asarray(r["neg_tokens"], dtype=np.int64) if r.get("neg_tokens") is not None else None
        neg_tgt = np.asarray(r["neg_targets"], dtype=np.int64) if r.get("neg_targets") is not None else None
    elif "prompt_tokens" in r and "corrected_tokens" in r:
        pr = list(map(int, r["prompt_tokens"]))
        good = list(map(int, r["corrected_tokens"]))
        if not pr or not good:
            raise ValueError("prompt_tokens and corrected_tokens must be non-empty")
        pos_tok = np.asarray(pr + good, dtype=np.int64)
        pos_tgt = np.arange(len(pr), len(pos_tok), dtype=np.int64)
        bad = list(map(int, r["mistaken_tokens"])) if r.get("mistaken_tokens") else []
        if bad:
            neg_tok = np.asarray(pr + bad, dtype=np.int64)
            neg_tgt = np.arange(len(pr), len(neg_tok), dtype=np.int64)
        else:
            neg_tok = neg_tgt = None
    else:
        raise ValueError("generative lesson needs ('pos_tokens', 'pos_targets') or ('prompt_tokens', 'corrected_tokens')")
    if len(pos_tgt) == 0 or int(pos_tgt[0]) < 1 or int(pos_tgt[-1]) >= len(pos_tok):
        raise ValueError("pos_targets must satisfy 1 <= t < len(pos_tokens)")
    if neg_tok is not None and (neg_tgt is None or len(neg_tgt) == 0 or int(neg_tgt[0]) < 1 or int(neg_tgt[-1]) >= len(neg_tok)):
        raise ValueError("neg_targets must satisfy 1 <= t < len(neg_tokens)")
    return {"pos_tokens": pos_tok, "pos_targets": pos_tgt, "neg_tokens": neg_tok, "neg_targets": neg_tgt}


def _norm_replay_traj(item) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(item, dict):
        tok, tgt = np.asarray(item["tokens"], dtype=np.int64), np.asarray(item["targets"], dtype=np.int64)
    else:
        tok, tgt = np.asarray(item[0], dtype=np.int64), np.asarray(item[1], dtype=np.int64)
    if len(tgt) == 0 or int(tgt[0]) < 1 or int(tgt[-1]) >= len(tok):
        raise ValueError("replay trajectory targets must satisfy 1 <= t < len(tokens)")
    return tok, tgt


def collate_right(seqs: list[np.ndarray], device: str, round_to: int = 32):
    """Right-padded batch so causal `position_ids = 0..L-1` and `logits[k, t - 1]` match unpadded prefixes."""
    import torch
    max_l = max(len(s) for s in seqs)
    n = int(-(-max_l // round_to) * round_to) if round_to > 1 else max_l
    ids = torch.zeros((len(seqs), n), dtype=torch.long)
    mask = torch.zeros((len(seqs), n), dtype=torch.long)
    for k, s in enumerate(seqs):
        ln = len(s)
        ids[k, :ln] = torch.from_numpy(np.asarray(s, dtype=np.int64))
        mask[k, :ln] = 1
    return ids.to(device), mask.to(device)


class _PrefixEntry:
    __slots__ = ("length", "h", "ple", "skv")

    def __init__(self, length: int, h, ple=None, skv=None):
        self.length = length
        self.h = h
        self.ple = ple
        self.skv = skv


class PrefixCache:
    def __init__(self, start_layer: int, kind: str, entries: dict[bytes, _PrefixEntry], offloaded: list[str], total_bytes: int):
        self.start_layer = start_layer
        self.kind = kind
        self.entries = entries
        self.offloaded = offloaded
        self.total_bytes = total_bytes

    @staticmethod
    def _key(seq: np.ndarray) -> bytes:
        return np.asarray(seq, dtype=np.int64).tobytes()

    @staticmethod
    def _arch_kind(model) -> str:
        inner = model.model
        cfg = inner.config
        if hasattr(inner, "unique_layer_types") and hasattr(inner, "get_per_layer_inputs"):
            return "gemma4"
        if hasattr(cfg, "layer_types") and "linear_attention" in (cfg.layer_types or ()):
            return "qwen35"
        return "decoder"

    @staticmethod
    def _free_prefix(model, start_layer: int, kind: str) -> list[str]:
        inner = model.model
        freed: list[str] = []
        for i in range(start_layer):
            inner.layers[i] = nn.Identity()
        freed.append(f"layers[0:{start_layer}]")
        if kind == "gemma4":
            for attr in ("embed_tokens_per_layer", "per_layer_model_projection", "per_layer_projection_norm"):
                if getattr(inner, attr, None) is not None:
                    setattr(inner, attr, None)
                    freed.append(attr)
        if getattr(inner, "embed_tokens", None) is not None and model.lm_head.weight.data_ptr() != inner.embed_tokens.weight.data_ptr():
            inner.embed_tokens = None
            freed.append("embed_tokens")
        return freed

    @classmethod
    def build(cls, model, seqs: list[np.ndarray], start_layer: int, device: str, amp: bool, cfg: dict | None = None, log=print, stream_prefix: bool = False):
        import gc
        from collections import UserDict
        import torch
        from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask

        cfg = cfg or {}
        if cfg.get("prefix_cache") is False or start_layer <= 0 or not seqs:
            if stream_prefix:
                model.to(device)
            return None
        inner = model.model
        mcfg = inner.config
        n_layers = int(mcfg.num_hidden_layers)
        if start_layer >= n_layers:
            if stream_prefix:
                model.to(device)
            return None

        uniq: dict[bytes, np.ndarray] = {}
        for s in seqs:
            arr = np.asarray(s, dtype=np.int64)
            uniq.setdefault(arr.tobytes(), arr)

        kind = cls._arch_kind(model)
        dev_type = device.split(":")[0]
        has_bf16 = any(p.dtype == torch.bfloat16 for p in model.parameters())
        use_amp = bool((amp and dev_type == "cuda") or has_bf16)
        pin = bool(dev_type == "cuda")
        store_dtype = torch.bfloat16 if use_amp else torch.float32
        elem_b = 2 if store_dtype == torch.bfloat16 else 4
        total_tok = sum(len(a) for a in uniq.values())
        per_tok_elems = int(mcfg.hidden_size)
        if kind == "gemma4":
            d_ple = int(getattr(inner, "hidden_size_per_layer_input", 0) or 0)
            if d_ple:
                per_tok_elems += (n_layers - start_layer) * d_ple
            n_kv = int(getattr(mcfg, "num_key_value_heads", 0) or 0)
            hd_full = int(getattr(mcfg, "global_head_dim", getattr(mcfg, "head_dim", 0)) or 0)
            hd_slide = int(getattr(mcfg, "head_dim", 0) or 0)
            per_tok_elems += 2 * n_kv * (hd_full + hd_slide)
        est_mb = (total_tok * per_tok_elems * elem_b) / (1024 * 1024)
        max_mb = float(cfg.get("prefix_cache_max_mb", 4096))
        if est_mb > max_mb:
            log(f"prefix_cache: skipped ({est_mb:.1f} MB > limit {max_mb:.1f} MB)")
            if stream_prefix:
                model.to(device)
            return None

        if kind == "qwen35":
            from transformers.models.qwen3_5.modeling_qwen3_5 import create_recurrent_attention_mask

        def _to_host(t):
            out = t.detach().to("cpu", store_dtype).contiguous()
            return out.pin_memory() if pin else out

        def _clear_dev():
            gc.collect()
            if dev_type == "cuda":
                torch.cuda.empty_cache()
            elif dev_type == "mps":
                torch.mps.empty_cache()

        entries: dict[bytes, _PrefixEntry] = {}
        actual_bytes = 0
        free_prefix = bool(cfg.get("free_prefix", True))
        freed: list[str] = []

        if stream_prefix:
            keys = list(uniq.keys())
            inner.rotary_emb.to(device)
            inner.embed_tokens.to(device)
            if kind == "gemma4":
                for attr in ("embed_tokens_per_layer", "per_layer_model_projection", "per_layer_projection_norm"):
                    mod = getattr(inner, attr, None)
                    if mod is not None and hasattr(mod, "to"):
                        mod.to(device)
            hs_cpu: dict[bytes, torch.Tensor] = {}
            ple_pref_cpu: dict[bytes, torch.Tensor | None] = {}
            ple_tail_cpu: dict[bytes, torch.Tensor | None] = {}
            skv_cpu_map: dict[bytes, dict] = {k: {} for k in keys}
            with torch.no_grad():
                for key in keys:
                    arr = uniq[key]
                    ids = torch.from_numpy(arr).unsqueeze(0).to(device)
                    with torch.autocast(dev_type, dtype=torch.bfloat16, enabled=use_amp):
                        emb = inner.embed_tokens(ids)
                        hs_cpu[key] = emb[0].detach().to("cpu", store_dtype).contiguous()
                        if kind == "gemma4" and getattr(inner, "hidden_size_per_layer_input", 0):
                            pli_all = inner.project_per_layer_inputs(emb, inner.get_per_layer_inputs(ids, emb))
                            ple_pref_cpu[key] = pli_all[0, :, :start_layer, :].detach().to("cpu", store_dtype).contiguous()
                            ple_tail_cpu[key] = _to_host(pli_all[0, :, start_layer:, :])
                            actual_bytes += ple_tail_cpu[key].numel() * ple_tail_cpu[key].element_size()
                        else:
                            ple_pref_cpu[key] = None
                            ple_tail_cpu[key] = None
            if kind == "gemma4":
                for attr in ("embed_tokens_per_layer", "per_layer_model_projection", "per_layer_projection_norm"):
                    if getattr(inner, attr, None) is not None:
                        setattr(inner, attr, None)
                        freed.append(attr)
            if model.lm_head.weight.data_ptr() != inner.embed_tokens.weight.data_ptr():
                inner.embed_tokens = None
                freed.append("embed_tokens")
            else:
                inner.embed_tokens.to("cpu")
            _clear_dev()

            with torch.no_grad():
                for i in range(start_layer):
                    layer_mod = inner.layers[i].to(device)
                    for key in keys:
                        h_i = hs_cpu[key].unsqueeze(0).to(device, non_blocking=pin)
                        L = h_i.shape[1]
                        attn = torch.ones((1, L), dtype=torch.long, device=device)
                        with torch.autocast(dev_type, dtype=torch.bfloat16, enabled=use_amp):
                            if kind == "gemma4":
                                position_ids = torch.arange(L, device=device).unsqueeze(0)
                                mk = {"config": mcfg, "inputs_embeds": h_i, "attention_mask": attn, "past_key_values": None, "position_ids": position_ids}
                                lt = mcfg.layer_types[i]
                                cm = create_causal_mask(**mk) if lt == "full_attention" else create_sliding_window_causal_mask(**mk)
                                pe = inner.rotary_emb(h_i, position_ids, lt)
                                pli = ple_pref_cpu[key][:, i, :].unsqueeze(0).to(device, non_blocking=pin) if ple_pref_cpu[key] is not None else None
                                skv = UserDict({t_lt: (kv[0].unsqueeze(0).to(device, non_blocking=pin), kv[1].unsqueeze(0).to(device, non_blocking=pin)) for t_lt, kv in skv_cpu_map[key].items()})
                                h_out = layer_mod(h_i, pli, shared_kv_states=skv, position_embeddings=pe, attention_mask=cm, position_ids=position_ids, past_key_values=None)
                                if getattr(getattr(layer_mod, "self_attn", None), "store_full_length_kv", False):
                                    skv_cpu_map[key][lt] = (skv[lt][0][0].detach().to("cpu", store_dtype).contiguous(), skv[lt][1][0].detach().to("cpu", store_dtype).contiguous())
                            elif kind == "qwen35":
                                position_ids = torch.arange(L, device=device).view(1, 1, -1).expand(4, 1, -1)
                                text_pos, mrope_pos = position_ids[0], position_ids[1:]
                                mk = {"config": mcfg, "inputs_embeds": h_i, "attention_mask": attn, "past_key_values": None, "position_ids": text_pos}
                                lt = mcfg.layer_types[i]
                                cm = create_causal_mask(**mk) if lt == "full_attention" else create_recurrent_attention_mask(**mk)
                                pe = inner.rotary_emb(h_i, mrope_pos)
                                h_out = layer_mod(h_i, position_embeddings=pe, attention_mask=cm, position_ids=text_pos, past_key_values=None, use_cache=False)
                            else:
                                position_ids = torch.arange(L, device=device).unsqueeze(0)
                                mk = {"config": mcfg, "inputs_embeds": h_i, "attention_mask": attn, "past_key_values": None, "position_ids": position_ids}
                                ltypes = getattr(mcfg, "layer_types", None)
                                lt = ltypes[i] if ltypes else "full_attention"
                                cm = create_sliding_window_causal_mask(**mk) if lt == "sliding_attention" else create_causal_mask(**mk)
                                pe = inner.rotary_emb(h_i, position_ids)
                                h_out = layer_mod(h_i, attention_mask=cm, position_embeddings=pe, position_ids=position_ids, past_key_values=None, use_cache=False)
                        hs_cpu[key] = h_out[0].detach().to("cpu", store_dtype).contiguous()
                    inner.layers[i] = nn.Identity()
                    del layer_mod
                    _clear_dev()
            freed.append(f"layers[0:{start_layer}]")
            for key in keys:
                hc = _to_host(hs_cpu.pop(key))
                actual_bytes += hc.numel() * hc.element_size()
                skv_final = None
                if kind == "gemma4":
                    skv_final = {}
                    for t_lt, (kc, vc) in skv_cpu_map[key].items():
                        kp, vp = _to_host(kc), _to_host(vc)
                        skv_final[t_lt] = (kp, vp)
                        actual_bytes += (kp.numel() + vp.numel()) * kp.element_size()
                entries[key] = _PrefixEntry(len(uniq[key]), hc, ple=ple_tail_cpu.get(key), skv=skv_final)
            for i in range(start_layer, n_layers):
                inner.layers[i].to(device)
            inner.norm.to(device)
            model.lm_head.to(device)
            _clear_dev()
        else:
            with torch.no_grad():
                for key, arr in uniq.items():
                    L = len(arr)
                    ids = torch.from_numpy(arr).unsqueeze(0).to(device)
                    attn = torch.ones((1, L), dtype=torch.long, device=device)
                    with torch.autocast(dev_type, dtype=torch.bfloat16, enabled=use_amp):
                        inputs_embeds = inner.embed_tokens(ids)
                        h = inputs_embeds
                        ple_tail = None
                        skv_cpu = None
                        if kind == "gemma4":
                            per_layer_inputs = None
                            if getattr(inner, "hidden_size_per_layer_input", 0):
                                per_layer_inputs = inner.get_per_layer_inputs(ids, inputs_embeds)
                                per_layer_inputs = inner.project_per_layer_inputs(inputs_embeds, per_layer_inputs)
                            position_ids = torch.arange(L, device=device).unsqueeze(0)
                            mk = {"config": mcfg, "inputs_embeds": inputs_embeds, "attention_mask": attn, "past_key_values": None, "position_ids": position_ids}
                            cmm = {"full_attention": create_causal_mask(**mk), "sliding_attention": create_sliding_window_causal_mask(**mk)}
                            pe = {lt: inner.rotary_emb(h, position_ids, lt) for lt in inner.unique_layer_types}
                            skv = UserDict()
                            for i in range(start_layer):
                                pli = per_layer_inputs[:, :, i, :] if per_layer_inputs is not None else None
                                lt = mcfg.layer_types[i]
                                h = inner.layers[i](h, pli, shared_kv_states=skv, position_embeddings=pe[lt], attention_mask=cmm[lt], position_ids=position_ids, past_key_values=None)
                            if per_layer_inputs is not None:
                                ple_tail = _to_host(per_layer_inputs[0, :, start_layer:, :])
                                actual_bytes += ple_tail.numel() * ple_tail.element_size()
                            skv_cpu = {}
                            for lt, kv in skv.items():
                                kc, vc = _to_host(kv[0][0]), _to_host(kv[1][0])
                                skv_cpu[lt] = (kc, vc)
                                actual_bytes += (kc.numel() + vc.numel()) * kc.element_size()
                        elif kind == "qwen35":
                            position_ids = torch.arange(L, device=device).view(1, 1, -1).expand(4, 1, -1)
                            text_pos = position_ids[0]
                            mrope_pos = position_ids[1:]
                            mk = {"config": mcfg, "inputs_embeds": inputs_embeds, "attention_mask": attn, "past_key_values": None, "position_ids": text_pos}
                            cmm = {"full_attention": create_causal_mask(**mk), "linear_attention": create_recurrent_attention_mask(**mk)}
                            pe = inner.rotary_emb(h, mrope_pos)
                            for i in range(start_layer):
                                h = inner.layers[i](h, position_embeddings=pe, attention_mask=cmm[mcfg.layer_types[i]], position_ids=text_pos, past_key_values=None, use_cache=False)
                        else:
                            position_ids = torch.arange(L, device=device).unsqueeze(0)
                            mk = {"config": mcfg, "inputs_embeds": inputs_embeds, "attention_mask": attn, "past_key_values": None, "position_ids": position_ids}
                            cmm = {"full_attention": create_causal_mask(**mk)}
                            if getattr(inner, "has_sliding_layers", False):
                                cmm["sliding_attention"] = create_sliding_window_causal_mask(**mk)
                            pe = inner.rotary_emb(h, position_ids)
                            ltypes = getattr(mcfg, "layer_types", None)
                            for i in range(start_layer):
                                lt = ltypes[i] if ltypes else "full_attention"
                                h = inner.layers[i](h, attention_mask=cmm[lt], position_embeddings=pe, position_ids=position_ids, past_key_values=None, use_cache=False)
                        h_cpu = _to_host(h[0])
                        actual_bytes += h_cpu.numel() * h_cpu.element_size()
                    entries[key] = _PrefixEntry(L, h_cpu, ple=ple_tail, skv=skv_cpu)

            if free_prefix:
                freed = cls._free_prefix(model, start_layer, kind)
                _clear_dev()

        log(f"prefix_cache: cached {len(entries)} seqs at layer {start_layer}/{n_layers} ({actual_bytes / (1024 * 1024):.1f} MB, stream={stream_prefix}, freed={freed})")
        return cls(start_layer, kind, entries, freed, actual_bytes)

    def forward_tail(self, model, seqs: list[np.ndarray], device: str, amp: bool, round_to: int = 32):
        from collections import UserDict
        import torch
        from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask

        ents = [self.entries[self._key(s)] for s in seqs]
        B = len(ents)
        max_l = max(e.length for e in ents)
        n = int(-(-max_l // round_to) * round_to) if round_to > 1 else max_l
        dev_type = device.split(":")[0]
        nb = bool(dev_type == "cuda")
        h_dtype = ents[0].h.dtype
        d = ents[0].h.shape[1]
        h_buf = torch.zeros((B, n, d), dtype=h_dtype)
        attn_buf = torch.zeros((B, n), dtype=torch.long)
        for k, e in enumerate(ents):
            h_buf[k, :e.length] = e.h
            attn_buf[k, :e.length] = 1
        h = h_buf.to(device, non_blocking=nb)
        attn = attn_buf.to(device, non_blocking=nb)
        if torch.is_grad_enabled() and getattr(model, "is_gradient_checkpointing", False):
            h.requires_grad_(True)

        inner = model.model
        mcfg = inner.config
        n_layers = int(mcfg.num_hidden_layers)
        use_amp = bool((amp and dev_type == "cuda") or ents[0].h.dtype == torch.bfloat16)

        with torch.autocast(dev_type, dtype=torch.bfloat16, enabled=use_amp):
            if self.kind == "gemma4":
                ple = None
                if ents[0].ple is not None:
                    n_tail, d_ple = ents[0].ple.shape[1], ents[0].ple.shape[2]
                    ple_buf = torch.zeros((B, n, n_tail, d_ple), dtype=ents[0].ple.dtype)
                    for k, e in enumerate(ents):
                        ple_buf[k, :e.length] = e.ple
                    ple = ple_buf.to(device, non_blocking=nb)
                skv = UserDict()
                if ents[0].skv:
                    for lt in ents[0].skv:
                        k0 = ents[0].skv[lt][0]
                        n_kv, _, hd = k0.shape
                        kb = torch.zeros((B, n_kv, n, hd), dtype=k0.dtype)
                        vb = torch.zeros((B, n_kv, n, hd), dtype=k0.dtype)
                        for k, e in enumerate(ents):
                            ek, ev = e.skv[lt]
                            kb[k, :, :e.length, :] = ek
                            vb[k, :, :e.length, :] = ev
                        skv[lt] = (kb.to(device, non_blocking=nb), vb.to(device, non_blocking=nb))
                position_ids = torch.arange(n, device=device).unsqueeze(0)
                mk = {"config": mcfg, "inputs_embeds": h, "attention_mask": attn, "past_key_values": None, "position_ids": position_ids}
                cmm = {"full_attention": create_causal_mask(**mk), "sliding_attention": create_sliding_window_causal_mask(**mk)}
                pe = {lt: inner.rotary_emb(h, position_ids, lt) for lt in inner.unique_layer_types}
                for i in range(self.start_layer, n_layers):
                    pli = ple[:, :, i - self.start_layer, :] if ple is not None else None
                    lt = mcfg.layer_types[i]
                    h = inner.layers[i](h, pli, shared_kv_states=skv, position_embeddings=pe[lt], attention_mask=cmm[lt], position_ids=position_ids, past_key_values=None)
            elif self.kind == "qwen35":
                from transformers.models.qwen3_5.modeling_qwen3_5 import create_recurrent_attention_mask
                position_ids = torch.arange(n, device=device).view(1, 1, -1).expand(4, B, -1)
                text_pos = position_ids[0]
                mrope_pos = position_ids[1:]
                mk = {"config": mcfg, "inputs_embeds": h, "attention_mask": attn, "past_key_values": None, "position_ids": text_pos}
                cmm = {"full_attention": create_causal_mask(**mk), "linear_attention": create_recurrent_attention_mask(**mk)}
                pe = inner.rotary_emb(h, mrope_pos)
                for i in range(self.start_layer, n_layers):
                    h = inner.layers[i](h, position_embeddings=pe, attention_mask=cmm[mcfg.layer_types[i]], position_ids=text_pos, past_key_values=None, use_cache=False)
            else:
                position_ids = torch.arange(n, device=device).unsqueeze(0)
                mk = {"config": mcfg, "inputs_embeds": h, "attention_mask": attn, "past_key_values": None, "position_ids": position_ids}
                cmm = {"full_attention": create_causal_mask(**mk)}
                if getattr(inner, "has_sliding_layers", False):
                    cmm["sliding_attention"] = create_sliding_window_causal_mask(**mk)
                pe = inner.rotary_emb(h, position_ids)
                ltypes = getattr(mcfg, "layer_types", None)
                for i in range(self.start_layer, n_layers):
                    lt = ltypes[i] if ltypes else "full_attention"
                    h = inner.layers[i](h, attention_mask=cmm[lt], position_embeddings=pe, position_ids=position_ids, past_key_values=None, use_cache=False)
            return inner.norm(h)


def _forward_backbone(model, seqs: list[np.ndarray], device: str, amp: bool, round_to: int = 32, prefix_cache=None):
    import torch
    if prefix_cache is not None:
        return prefix_cache.forward_tail(model, seqs, device, amp, round_to=round_to)
    ids, attn = collate_right(seqs, device, round_to=round_to)
    dev_type = device.split(":")[0]
    if amp and dev_type == "cuda":
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return model.model(input_ids=ids, attention_mask=attn).last_hidden_state
    return model.model(input_ids=ids, attention_mask=attn).last_hidden_state


def _seq_mean_logprob(model, h_batch, seqs: list[np.ndarray], tgts: list[np.ndarray]):
    """Compute length-normalized mean target log-prob `mean_{t in tgt} log p(tok[t] | tok[:t])` per sequence."""
    import torch
    out = []
    for k, (tok, tgt) in enumerate(zip(seqs, tgts)):
        pos_idx = torch.from_numpy(tgt - 1).to(h_batch.device, dtype=torch.long)
        lbl = torch.from_numpy(tok[tgt]).to(h_batch.device, dtype=torch.long)
        logits = _head_logits(model, h_batch[k, pos_idx])
        logp = torch.log_softmax(logits, dim=-1)
        picked = logp.gather(-1, lbl.unsqueeze(-1)).squeeze(-1)
        out.append(picked.mean())
    return torch.stack(out)


def student_traj_nll(model, trajs: list[tuple[np.ndarray, np.ndarray]], device: str = "cpu", amp: bool = False) -> list[dict]:
    """Evaluate exact per-trajectory `(n_tok, n_tgt, nll_cand, top1)` on `TRJ1` pairs using the PyTorch student."""
    import torch
    res = []
    with torch.no_grad():
        for i, (tok_raw, tgt_raw) in enumerate(trajs):
            tok = np.asarray(tok_raw, dtype=np.int64)
            tgt = np.asarray(tgt_raw, dtype=np.int64)
            h = _forward_backbone(model, [tok], device, amp, round_to=1)[0]
            pos_idx = torch.from_numpy(tgt - 1).to(device, dtype=torch.long)
            lbl = torch.from_numpy(tok[tgt]).to(device, dtype=torch.long)
            logits = _head_logits(model, h[pos_idx])
            logp = torch.log_softmax(logits, dim=-1)
            nll = float(-logp.gather(-1, lbl.unsqueeze(-1)).squeeze(-1).mean().item())
            top1 = float((logits.argmax(dim=-1) == lbl).float().mean().item())
            res.append({"i": i, "n_tok": int(len(tok)), "n_tgt": int(len(tgt)), "nll_cand": nll, "top1": top1})
    return res


def learn_gates_seq(
    art_path: str,
    config_dir: str,
    feedback: list[dict],
    replay: list,
    out: str,
    cfg: dict | None = None,
    log=print,
    device: str | None = None,
    mask_in: dict | str | Path | None = None,
    mask_out: str | Path | None = None,
) -> dict:
    """Generative / tool-calling switch Apprentice (`<=25` mistake lessons + replay KL hold + A-GEM projection).

    Optimizes straight-through binary MLP neuron switches `theta` (`1` bit/neuron, zero weight values changed, zero bytes added):
      - Lesson objective: `-nll_weight * mean_logp(y+ | x) + margin_weight * softplus(margin - (mean_logp(y+ | x) - mean_logp(y- | x)))`
      - Replay hold: exact full-vocabulary token KL at replay target positions from cached pre-`lm_head` states `h0`
      - A-GEM projection: projects conflicting lesson gradients orthogonal to replay-hold gradients whenever `dot(g_new, g_hold) < 0`
    """
    import torch

    cfg = cfg or {}
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    amp = device.startswith("cuda") and bool(cfg.get("amp", True))
    switch_mode = cfg.get("switch_mode", "up_row_zero")
    pad_mult = int(cfg.get("pad_multiple", 32))

    lessons = [_norm_seq_lesson(r) for r in feedback]
    rep_trajs = [_norm_replay_traj(x) for x in replay]
    if not lessons and not rep_trajs:
        raise ValueError("learn_gates_seq requires at least one feedback lesson or replay trajectory")

    art = lattice.Artifact(art_path)
    frozen_bf16 = bool(cfg.get("frozen_bf16", amp))
    stream_pref = _should_stream_prefix(art, cfg, device, frozen_bf16)
    model, _ = build_student(art, config_dir, [], cfg.get("ggml_lib"), "cpu" if stream_pref else device, frozen_bf16=frozen_bf16)
    layers = _switchable_layers(art, len(model.model.layers), switch_mode, cfg.get("gate_last_blocks"))
    if not layers:
        art.close()
        raise RuntimeError(f"no switchable FFN layers found in {art_path} under switch_mode={switch_mode!r}")

    all_seqs = [r["pos_tokens"] for r in lessons] + [r["neg_tokens"] for r in lessons if r["neg_tokens"] is not None] + [t for t, _ in rep_trajs]
    prefix_cache = PrefixCache.build(model, all_seqs, min(layers), device, amp, cfg, log=log, stream_prefix=stream_pref)
    if stream_pref and prefix_cache is None:
        model.to(device)
    if _auto_checkpointing(model, cfg, device) and amp:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.train()

    init_val = float(cfg.get("gate_init", 1.0))
    theta = {il: torch.nn.Parameter(torch.full((art.shape(f"blk.{il}.ffn_down.weight")[1],), init_val, device=device)) for il in layers}

    replay_h0: list[torch.Tensor] = []
    with torch.no_grad():
        for tok, tgt in rep_trajs:
            h = _forward_backbone(model, [tok], device, amp, round_to=pad_mult, prefix_cache=prefix_cache)[0]
            pos_idx = torch.from_numpy(tgt - 1).to(device, dtype=torch.long)
            replay_h0.append(h[pos_idx].detach().float().cpu())

    def hook(il):
        def f(mod, args):
            s = torch.sigmoid(theta[il])
            g = (theta[il] > 0).float() + s - s.detach()
            return (args[0] * g.to(args[0].dtype),)
        return f

    hooks = [model.model.layers[il].mlp.down_proj.register_forward_pre_hook(hook(il)) for il in layers]
    flat = list(theta.values())
    opt = torch.optim.AdamW(flat, lr=float(cfg.get("gate_lr", 0.05)), weight_decay=0.0, eps=float(cfg.get("adam_eps", 1e-4)))
    rng = random.Random(cfg.get("seed", 20261031))

    steps = int(cfg.get("steps", 80))
    def_bs = _default_seq_batch(model, device)
    nf = int(cfg.get("feedback_batch", min(def_bs, len(lessons))))
    nr = int(cfg.get("replay_batch", min(def_bs, len(rep_trajs)))) if rep_trajs else 0
    nll_w = float(cfg.get("nll_weight", 1.0))
    margin_w = float(cfg.get("margin_weight", 1.0))
    margin = float(cfg.get("margin", 0.5))
    hold_w = float(cfg.get("replay_weight", 2.0))
    kl_temp = float(cfg.get("kl_temp", 1.0))
    use_agem = bool(cfg.get("project", True))

    log(f"learn_gates_seq: {sum(w.numel() for w in flat):,} switches across {len(layers)} FFN layers, {len(lessons)} lessons, {len(rep_trajs)} replay")
    hist: list[tuple[float, float, float]] = []
    conflicts = 0

    for step in range(steps):
        if lessons and (nll_w > 0 or margin_w > 0):
            fb_batch = rng.sample(lessons, min(nf, len(lessons)))
            pos_seqs = [r["pos_tokens"] for r in fb_batch]
            pos_tgts = [r["pos_targets"] for r in fb_batch]
            h_pos = _forward_backbone(model, pos_seqs, device, amp, round_to=pad_mult, prefix_cache=prefix_cache)
            lp_pos = _seq_mean_logprob(model, h_pos, pos_seqs, pos_tgts)
            loss_nll = -lp_pos.mean()

            neg_idx = [k for k, r in enumerate(fb_batch) if r["neg_tokens"] is not None]
            if neg_idx and margin_w > 0:
                neg_seqs = [fb_batch[k]["neg_tokens"] for k in neg_idx]
                neg_tgts = [fb_batch[k]["neg_targets"] for k in neg_idx]
                h_neg = _forward_backbone(model, neg_seqs, device, amp, round_to=pad_mult, prefix_cache=prefix_cache)
                lp_neg = _seq_mean_logprob(model, h_neg, neg_seqs, neg_tgts)
                loss_margin = torch.nn.functional.softplus(margin - (lp_pos[neg_idx] - lp_neg)).mean()
            else:
                loss_margin = torch.zeros((), device=device)
            loss_new = nll_w * loss_nll + margin_w * loss_margin
        else:
            loss_nll = torch.zeros((), device=device)
            loss_margin = torch.zeros((), device=device)
            loss_new = torch.zeros((), device=device)

        if nr > 0:
            r_ids = rng.sample(range(len(rep_trajs)), min(nr, len(rep_trajs)))
            r_seqs = [rep_trajs[i][0] for i in r_ids]
            h_rep = _forward_backbone(model, r_seqs, device, amp, round_to=pad_mult, prefix_cache=prefix_cache)
            kl_terms = []
            for k, i in enumerate(r_ids):
                tgt = rep_trajs[i][1]
                pos_idx = torch.from_numpy(tgt - 1).to(device, dtype=torch.long)
                h_cur = h_rep[k, pos_idx]
                h_ref = replay_h0[i].to(device=device, dtype=h_cur.dtype)
                with torch.no_grad():
                    log_p0 = torch.log_softmax(_head_logits(model, h_ref) / kl_temp, dim=-1)
                    p0 = log_p0.exp()
                log_pc = torch.log_softmax(_head_logits(model, h_cur) / kl_temp, dim=-1)
                kl_terms.append((p0 * (log_p0 - log_pc)).sum(dim=-1).mean() * (kl_temp * kl_temp))
            loss_hold = torch.stack(kl_terms).mean()
        else:
            loss_hold = torch.zeros((), device=device)

        opt.zero_grad()
        has_new = bool(loss_new.requires_grad)
        has_hold = bool(nr > 0 and hold_w > 0 and loss_hold.requires_grad)
        if use_agem and has_new and has_hold:
            scaled_hold = hold_w * loss_hold
            g_new = torch.autograd.grad(loss_new, flat, retain_graph=True, allow_unused=True)
            g_old = torch.autograd.grad(scaled_hold, flat, allow_unused=True)
            g_new = [torch.zeros_like(p) if g is None else g for g, p in zip(g_new, flat)]
            g_old = [torch.zeros_like(p) if g is None else g for g, p in zip(g_old, flat)]
            dot = sum((a * b).sum() for a, b in zip(g_new, g_old))
            if dot < 0:
                ratio = dot / sum((b * b).sum() for b in g_old).clamp_min(1e-30)
                g_new = [a - ratio * b for a, b in zip(g_new, g_old)]
                conflicts += 1
            for p, a, b in zip(flat, g_new, g_old):
                p.grad = a + b
            opt.step()
        elif has_new or has_hold:
            tot = (loss_new if has_new else 0.0) + (hold_w * loss_hold if has_hold else 0.0)
            tot.backward()
            opt.step()

        hist.append((float(loss_nll.item()), float(loss_margin.item()), float(loss_hold.item())))
        if step % max(1, steps // 5) == 0 or step == steps - 1:
            log(f"learn_gates_seq step {step}/{steps}: nll={hist[-1][0]:.4f} margin={hist[-1][1]:.4f} replay_kl={hist[-1][2]:.5f}")

    for h in hooks:
        h.remove()

    off, bits, _, m_v3 = _apply_switches_to_file_and_mask(
        art, art_path, out, layers, theta, switch_mode,
        mask_in=mask_in or cfg.get("mask_in"), mask_out=mask_out or cfg.get("mask_out"),
    )
    art.close()
    del model
    import gc
    gc.collect()
    if amp:
        torch.cuda.empty_cache()
    n_off = sum(map(len, off.values()))
    rec = {
        "method": "gates_seq",
        "switch_mode": switch_mode,
        "steps": steps,
        "gate_lr": float(cfg.get("gate_lr", 0.05)),
        "neurons": bits,
        "off": n_off,
        "fraction_moved": n_off / max(1, bits),
        "off_by_layer": {str(k): len(v) for k, v in off.items()},
        "agem_conflicts": conflicts,
        "loss_first": hist[0] if hist else None,
        "loss_last": hist[-1] if hist else None,
        "n_feedback": len(lessons),
        "n_replay": len(rep_trajs),
        "base_sha256": sha256(art_path),
        "out_sha256": sha256(out),
        "bytes": Path(out).stat().st_size,
    }
    Path(str(out) + ".lineage.json").write_text(json.dumps(rec, indent=1))
    Path(str(out) + ".switches.json").write_text(json.dumps({str(k): v for k, v in off.items()}))
    if m_v3 is not None:
        rec["mask"] = m_v3
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
    """Deterministic 50/50 split by `prompt_id` (when provided) or `task` id so related items stay on one side."""
    return int(hashlib.sha256((task if prompt_id is None else str(prompt_id)).encode()).hexdigest()[:8], 16) % 2


def row_side(r: dict, c: dict) -> int:
    """Resolve select side (0 = replay pool, 1 = judge), grouping multi-variant rows by `prompt_id` when configured."""
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
        # quick screen: score boundary-adjacent indices and reuse base scores elsewhere
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
    """Score the built GGUF directly on the fixed panel (cached by model and panel hash; no base-score imputation)."""
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


def _load_json_or_jsonl(path_or_list) -> list[dict]:
    if isinstance(path_or_list, list):
        return list(path_or_list)
    p = Path(path_or_list)
    txt = p.read_text().strip()
    if not txt:
        return []
    if txt[0] == "[":
        return json.loads(txt)
    return [json.loads(line) for line in txt.splitlines() if line.strip()]


def _tokenize_seq_lessons(tc, model_gguf: str, raw_rows: list[dict], threads: int = 8) -> list[dict]:
    from . import runtime
    need_str = [i for i, r in enumerate(raw_rows) if "prompt_tokens" not in r and "pos_tokens" not in r]
    if not need_str:
        return [dict(r) for r in raw_rows]
    strs: list[str] = []
    for i in need_str:
        r = raw_rows[i]
        pr = str(r["prompt"])
        good_s = str(r["corrected"])
        bad_s = str(r.get("mistaken") or "")
        strs.extend([pr, pr + good_s, good_s, (pr + bad_s) if bad_s else "", bad_s])
    toks = [x.tolist() for x in runtime.tokenize_docs(tc, model_gguf, strs, threads=threads)]
    out = [dict(r) for r in raw_rows]
    for k, i in enumerate(need_str):
        pr, pr_good, good_raw, pr_bad, bad_raw = toks[5 * k : 5 * k + 5]
        good = pr_good[len(pr):] if (len(pr_good) > len(pr) and pr_good[: len(pr)] == pr) else good_raw
        bad = (pr_bad[len(pr):] if (len(pr_bad) > len(pr) and pr_bad[: len(pr)] == pr) else bad_raw) if bad_raw else []
        out[i]["prompt_tokens"] = pr
        out[i]["corrected_tokens"] = good
        if bad:
            out[i]["mistaken_tokens"] = bad
    return out


def _load_replay_trajs(tc, model_gguf: str, spec, threads: int = 8) -> list[tuple[np.ndarray, np.ndarray]]:
    from . import runtime
    if not spec:
        return []
    if isinstance(spec, (str, Path)) and str(spec).endswith((".bin", ".trj")):
        return read_trj_multi(spec)
    rows = _load_json_or_jsonl(spec)
    if not rows:
        return []
    if "tokens" in rows[0] and "targets" in rows[0]:
        return [(np.asarray(r["tokens"], dtype=np.int32), np.asarray(r["targets"], dtype=np.uint32)) for r in rows]
    strs = []
    for r in rows:
        pr = str(r["prompt"])
        full = pr + str(r["completion"])
        strs.extend([pr, full])
    toks = [x.tolist() for x in runtime.tokenize_docs(tc, model_gguf, strs, threads=threads)]
    out = []
    for k in range(0, len(toks), 2):
        pre, full = toks[k], toks[k + 1]
        if len(full) > len(pre) and full[: len(pre)] == pre:
            out.append((np.asarray(full, dtype=np.int32), np.arange(len(pre), len(full), dtype=np.uint32)))
    return out


def run_generative(cfg: dict, fold: int | None = None, log=print) -> dict:
    """`chantless learn` for generative / tool-calling models (`learn_gates_seq` + `empty`/`flipped`/`shuffled` controls)."""
    from . import builder as bmod, gguf_ops, kl, stats, util
    from .env.protocol import load_environment
    from .evaluator import Evaluator, ok_rows
    from .runtime import Toolchain

    c = cfg["learn"]
    tc = Toolchain.from_config(cfg["toolchain"])
    out_dir = Path(c["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    base_model = str(c["model"])
    config_dir = str(c["config_dir"])
    ggml_lib = str(c.get("ggml_lib") or cfg["toolchain"]["llama_bin"])
    base_cfg = {**c, "ggml_lib": ggml_lib, "switch_mode": c.get("switch_mode", "up_row_zero")}

    all_lessons = _tokenize_seq_lessons(tc, base_model, _load_json_or_jsonl(c["lessons"]))
    train_lessons = [r for r in all_lessons if fold is None or r.get("fold") != fold]
    held_lessons = [r for r in all_lessons if fold is not None and r.get("fold") == fold]
    if len(train_lessons) > int(c.get("max_lessons", 25)):
        train_lessons = train_lessons[: int(c.get("max_lessons", 25))]
    replay_trajs = _load_replay_trajs(tc, base_model, c.get("replay"))

    tag = "all" if fold is None else f"fold{fold}"
    learned_gguf = str(out_dir / f"apprentice-{tag}.gguf")
    learned_mask = str(c.get("mask_out") or (out_dir / f"apprentice-{tag}.mask.json"))
    rec_learned = learn_gates_seq(
        base_model, config_dir, train_lessons, replay_trajs, learned_gguf,
        cfg=base_cfg, log=log, mask_in=c.get("mask_in"), mask_out=learned_mask if c.get("mask_in") else None,
    )

    if c.get("source_gguf") and c.get("mask_in"):
        groups, _ = gguf_ops.adapter_from_gguf(c["source_gguf"], config_dir)
        chk_p = out_dir / f"rebuild-check-{tag}.gguf"
        bmod.Builder(tc, c["source_gguf"], groups, c.get("imatrix"), out_dir / ".work-rebuild").build(maskmod.load(learned_mask), chk_p)
        if sha256(str(chk_p)) != sha256(learned_gguf):
            raise RuntimeError("rebuilt GGUF from exported v3 apprentice mask did not match in-place switched GGUF")
        chk_p.unlink(missing_ok=True)

    arms = {"base": base_model, "learned": learned_gguf}
    control_recs = {}
    if c.get("controls", True):
        # 1. Empty control (0 lessons, replay/init only)
        emp_gguf = str(out_dir / f"control-empty-{tag}.gguf")
        emp_cfg = {**base_cfg, "nll_weight": 0.0, "margin_weight": 0.0}
        emp_cfg.pop("mask_in", None); emp_cfg.pop("mask_out", None)
        control_recs["empty"] = learn_gates_seq(base_model, config_dir, [], replay_trajs or [(np.array([1, 2, 3], dtype=np.int32), np.array([1, 2], dtype=np.uint32))], emp_gguf, cfg=emp_cfg, log=log)
        arms["empty"] = emp_gguf

        # 2. Flipped control (swap corrected <-> mistaken on lessons that carry mistaken_tokens)
        flipped_lessons = []
        for r in train_lessons:
            if r.get("mistaken_tokens"):
                flipped_lessons.append({**r, "corrected_tokens": r["mistaken_tokens"], "mistaken_tokens": r["corrected_tokens"]})
        if flipped_lessons:
            flp_gguf = str(out_dir / f"control-flipped-{tag}.gguf")
            flp_cfg = dict(base_cfg); flp_cfg.pop("mask_in", None); flp_cfg.pop("mask_out", None)
            control_recs["flipped"] = learn_gates_seq(base_model, config_dir, flipped_lessons, replay_trajs, flp_gguf, cfg=flp_cfg, log=log)
            arms["flipped"] = flp_gguf

        # 3. Shuffled control (cyclic shift of `(corrected_tokens, mistaken_tokens)` pairs together across prompts)
        if len(train_lessons) >= 2:
            shuf_lessons = []
            for i, r in enumerate(train_lessons):
                other = train_lessons[(i + 1) % len(train_lessons)]
                shuf_lessons.append({
                    **r,
                    "corrected_tokens": other["corrected_tokens"],
                    "mistaken_tokens": other.get("mistaken_tokens"),
                })
            shf_gguf = str(out_dir / f"control-shuffled-{tag}.gguf")
            shf_cfg = dict(base_cfg); shf_cfg.pop("mask_in", None); shf_cfg.pop("mask_out", None)
            control_recs["shuffled"] = learn_gates_seq(base_model, config_dir, shuf_lessons, replay_trajs, shf_gguf, cfg=shf_cfg, log=log)
            arms["shuffled"] = shf_gguf

    # Score C++ `traj-kl` NLL / margin across all arms on held-out (or train) lessons
    probe_lessons = held_lessons or train_lessons
    trj_eval = out_dir / f"eval-lessons-{tag}.bin"
    eval_recs = []
    pair_indices = []
    for r in probe_lessons:
        nl = _norm_seq_lesson(r)
        i_pos = len(eval_recs)
        eval_recs.append((nl["pos_tokens"].tolist(), nl["pos_targets"].tolist()))
        i_neg = None
        if nl["neg_tokens"] is not None:
            i_neg = len(eval_recs)
            eval_recs.append((nl["neg_tokens"].tolist(), nl["neg_targets"].tolist()))
        pair_indices.append((i_pos, i_neg))
    kl.write_trj(trj_eval, eval_recs)
    kl.traj_kl(tc, "base", base_model, str(trj_eval), str(out_dir / "store_eval"), str(out_dir / "base_eval.json"), str(out_dir / "kl.log"), gpu=tc.gpus[0])

    cpp_metrics = {}
    for arm_name, arm_path in arms.items():
        r_kl = kl.traj_kl(tc, "kl", arm_path, str(trj_eval), str(out_dir / "store_eval"), str(out_dir / f"kl-{arm_name}.json"), str(out_dir / "kl.log"), gpu=tc.gpus[0])
        pos_nlls = [r_kl["trajs"][ip]["nll_cand"] for ip, _ in pair_indices]
        margins = [(-r_kl["trajs"][ip]["nll_cand"]) - (-r_kl["trajs"][ineg]["nll_cand"]) for ip, ineg in pair_indices if ineg is not None]
        cpp_metrics[arm_name] = {
            "bytes": Path(arm_path).stat().st_size,
            "sha256": sha256(arm_path),
            "pos_nll_mean": float(np.mean(pos_nlls)) if pos_nlls else None,
            "margin_mean": float(np.mean(margins)) if margins else None,
            "pairs_won": int(sum(m > 0 for m in margins)) if margins else None,
            "n_pairs": len(margins),
        }

    # Optional live `Evaluator` task evaluation (`heldout_ids` + `select` side-1 retention gate)
    env_metrics = {}
    accepted = bool(cpp_metrics["learned"]["pos_nll_mean"] < cpp_metrics["base"]["pos_nll_mean"])
    if cfg.get("env"):
        env = load_environment(cfg["env"]["module"], **cfg["env"].get("kwargs", {}))
        by_id = {t.id: t for t in env.tasks()}
        ev = Evaluator(tc, env, cfg.get("decoding", {"temperature": 0.0}),
                       int(c.get("judge_slots", 1)),
                       cfg.get("serving", {}).get("ctx_per_slot", env.deployment["context"]),
                       env.deployment["kv_type"], out_dir / "episodes",
                       cache_prompt=bool(c.get("judge_cache_prompt", False)))
        held_ids = c.get("heldout_ids") or [t.id for t in env.tasks() if t.split == "train" and _side(t.id) == 1]
        held_tasks = [by_id[i] for i in held_ids if i in by_id]
        sel_tasks = [t for t in env.tasks() if t.split == "select" and _side(t.id) == 1]
        if c.get("select_limit"):
            sel_tasks = sel_tasks[: int(c["select_limit"])]
        ref_model = str(c.get("gate_ref_model") or base_model)
        ref_sel = ok_rows(ev.run(ref_model, [(t, util.stable_seed("learn-sel", t.id)) for t in sel_tasks], "learn-ref-sel", "src")) if sel_tasks else []

        for arm_name, arm_path in arms.items():
            h_rows = ok_rows(ev.run(arm_path, [(t, util.stable_seed("learn-held", t.id)) for t in held_tasks], f"learn-{arm_name}-held", "cand")) if held_tasks else []
            s_rows = ok_rows(ev.run(arm_path, [(t, util.stable_seed("learn-sel", t.id)) for t in sel_tasks], f"learn-{arm_name}-sel", "cand")) if sel_tasks else []
            bnd = stats.delta_and_bounds(s_rows + ref_sel, float(c.get("alpha", 0.05))) if (s_rows and ref_sel) else {"delta": 0.0, "L": 0.0, "U": 0.0}
            env_metrics[arm_name] = {
                "heldout_n": len(h_rows),
                "heldout_passed": int(sum(bool(r["success"]) for r in h_rows)),
                "heldout_rate": float(sum(bool(r["success"]) for r in h_rows) / max(1, len(h_rows))) if h_rows else None,
                "select_n": len(s_rows),
                "select_rate": float(sum(bool(r["success"]) for r in s_rows) / max(1, len(s_rows))) if s_rows else None,
                "select_delta": bnd["delta"],
                "select_L": bnd["L"],
                "gate_pass": bool(bnd["L"] > -float(c.get("gate_margin", 0.03))),
            }
        if held_tasks:
            l_rate = env_metrics["learned"]["heldout_rate"]
            ctrl_rates = [env_metrics[k]["heldout_rate"] for k in ("base", "empty", "flipped", "shuffled") if k in env_metrics and env_metrics[k]["heldout_rate"] is not None]
            accepted = bool(env_metrics["learned"]["gate_pass"] and all(l_rate > cr for cr in ctrl_rates))

    res = {
        "accepted": accepted,
        "fold": fold,
        "lineage": rec_learned,
        "controls_lineage": control_recs,
        "cpp_traj_kl": cpp_metrics,
        "env_eval": env_metrics,
        "mask_out": learned_mask if c.get("mask_in") else None,
    }
    (out_dir / f"result-{tag}.json").write_text(json.dumps(res, indent=2))
    log(f"learn_generative {tag}: accepted={accepted} cpp={cpp_metrics} env={env_metrics}")
    return res


def run(cfg: dict, fold: int | None = None, log=print) -> dict:
    """`chantless.cli learn`: train on feedback, materialize switches/levels, judge the real file + controls."""
    c = cfg["learn"]
    if c.get("mode") == "generative" or c.get("method") == "gates_seq":
        return run_generative(cfg, fold=fold, log=log)
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
    drop = max(base["caught_wo"] - cand["caught_wo"], base["caught_beemo"] - cand["caught_beemo"])    # enforce per-subset recall drop cap alongside the pooled gate
    accepted = bool(cand["gate_pass"] and rise <= c.get("max_human_rise", .01) and drop <= c.get("max_ai_drop", .02) and better)
    res = {"accepted": accepted, "fold": fold, "lineage": rec, "base": base, "candidate": cand, "split": "select_side1"}
    (out_dir / f"result-{tag}.json").write_text(json.dumps(res, indent=1))
    log(f"learn {tag}: accepted={accepted} gate L {cand['L']:.4f} held pairs right {base['held']['pairs_right']} -> {cand['held']['pairs_right']}, "
        f"flagged AI {base['held']['ai_flagged']} -> {cand['held']['ai_flagged']}, Beemo humans {base['human_flagged_beemo']:.3f} -> {cand['human_flagged_beemo']:.3f}")
    return res
