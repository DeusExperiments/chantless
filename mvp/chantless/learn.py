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
import logging
import math
import random
import re
import struct
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from . import lattice, mask as maskmod

_LOG = logging.getLogger(__name__)

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
    lb = cfg.get("last_blocks")
    lb = 2 if lb is None else int(lb)
    kept = sorted({int(n.split(".")[1]) for n in grid})[-lb:] if lb > 0 else []
    return [n for n in grid if int(n.split(".")[1]) in kept]


def _inkling2_rms_norm(x: torch.Tensor, weight: torch.Tensor | None, eps: float) -> torch.Tensor:
    xf = x.float()
    y = xf * torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + eps)
    if weight is not None:
        y = y * weight.float()
    return y.to(x.dtype)


def _inkling2_clip(x: torch.Tensor, scale: torch.Tensor | None, threshold: float) -> torch.Tensor:
    dt = x.dtype
    xf = x.float()
    if scale is not None:
        xf = xf * scale.float()
    norm = torch.sqrt(xf.pow(2).sum(dim=-1, keepdim=True)) + 1e-6
    return (xf * torch.clamp(threshold / norm, min=0.0, max=1.0)).to(dt)


def _inkling2_mix(blocks: list[torch.Tensor], cur: torch.Tensor, query: torch.Tensor, n_embd: int) -> torch.Tensor:
    states = torch.stack([*blocks, cur], dim=2)  # [B, L, S, D]
    states_n = _inkling2_rms_norm(states, None, 1e-6).float()
    scores = torch.matmul(states_n, query.float().t()) * (1.0 / math.sqrt(n_embd))
    probs = torch.softmax(scores, dim=2)
    return torch.einsum("blsc,blsd->blcd", probs, states_n).to(cur.dtype)


def _inkling2_convolve(
    cur: torch.Tensor,
    weight: torch.Tensor,
    dynamic: torch.Tensor | None,
    n_conv: int,
    attn_bool: torch.Tensor | None = None,
) -> torch.Tensor:
    dt = cur.dtype
    cf = cur.float()
    if attn_bool is not None:
        cf = cf.masked_fill(~attn_bool.unsqueeze(-1), 0.0)
    x = F.pad(cf, (0, 0, n_conv - 1, 0))  # [B, n_conv - 1 + L, C]
    out = F.conv1d(x.transpose(1, 2), weight.float().unsqueeze(1), groups=cf.shape[-1]).transpose(1, 2)
    if dynamic is not None:
        B, L, C = cf.shape
        n_h = dynamic.shape[-1] // n_conv
        dyn = dynamic.float().view(B, L, n_h, n_conv)
        for lag in range(n_conv):
            s = n_conv - 1 - lag
            win = x[:, s : s + L, :].view(B, L, n_h, C // n_h)
            out = out + (win * dyn[..., lag : lag + 1]).reshape(B, L, C)
    denom = math.sqrt((2 if dynamic is not None else 1) * n_conv + 1)
    return ((cf + out) / denom).to(dt)


class Inkling2RMSNorm(nn.Module):
    def __init__(self, n_embd: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(n_embd, dtype=torch.float32))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _inkling2_rms_norm(x, self.weight, self.eps)


class Inkling2MoE(nn.Module):
    def __init__(
        self,
        n_embd: int,
        n_ff_exp: int,
        n_expert: int,
        n_expert_used: int,
        situ_beta: float,
        situ_linear_beta: float,
        expert_weights_scale: float,
        has_exp_probs_b: bool,
    ):
        super().__init__()
        self.n_embd = n_embd
        self.n_ff_exp = n_ff_exp
        self.n_expert = n_expert
        self.n_expert_used = n_expert_used
        self.situ_beta = situ_beta
        self.situ_linear_beta = situ_linear_beta
        self.expert_weights_scale = expert_weights_scale
        self.gate_inp = nn.Parameter(torch.empty(n_expert, n_embd, dtype=torch.float32))
        self.exp_probs_b = nn.Parameter(torch.zeros(n_expert, dtype=torch.float32)) if has_exp_probs_b else None
        self.gate_up_exps = nn.Parameter(torch.empty(n_expert, 2 * n_ff_exp, n_embd, dtype=torch.float32))
        self.down_exps = nn.Parameter(torch.empty(n_expert, n_embd, n_ff_exp, dtype=torch.float32))
        self.switch_hook = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        x_flat = x.reshape(-1, self.n_embd)
        T = x_flat.shape[0]
        if T == 0:
            return torch.zeros_like(x)

        logits = F.linear(x_flat.float(), self.gate_inp.float())
        probs = torch.sigmoid(logits)
        sel_probs = probs + self.exp_probs_b.float() if self.exp_probs_b is not None else probs
        _, topk_idx = torch.topk(sel_probs, self.n_expert_used, dim=-1)
        weights = torch.gather(probs, dim=-1, index=topk_idx)
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
        if self.expert_weights_scale != 0.0 and self.expert_weights_scale != 1.0:
            weights = weights * self.expert_weights_scale

        flat_exp = topk_idx.reshape(-1)
        flat_tok = torch.arange(T, device=x.device).unsqueeze(1).expand(T, self.n_expert_used).reshape(-1)
        flat_w = weights.reshape(-1)
        order = torch.argsort(flat_exp, stable=True)
        exp_s = flat_exp[order]
        tok_s = flat_tok[order]
        w_s = flat_w[order]
        counts = torch.bincount(exp_s, minlength=self.n_expert).tolist()

        act_parts: list[torch.Tensor] = []
        slices: list[tuple[int, int, int]] = []
        off = 0
        W = self.n_ff_exp
        beta = self.situ_beta
        lb = self.situ_linear_beta
        for e, cnt in enumerate(counts):
            if cnt == 0:
                continue
            x_e = x_flat[tok_s[off : off + cnt]]
            gu = F.linear(x_e.to(self.gate_up_exps.dtype), self.gate_up_exps[e])
            gate, up = gu[:, :W].float(), gu[:, W:].float()
            act = (beta * torch.tanh(gate / beta) * torch.sigmoid(gate)) * (lb * torch.tanh(up / lb) if lb > 0.0 else up)
            act_parts.append(act.to(x.dtype))
            slices.append((e, off, cnt))
            off += cnt

        act_all = self.switch_hook(torch.cat(act_parts, dim=0))
        out = torch.zeros((T, self.n_embd), device=x.device, dtype=torch.float32)
        for e, s_off, cnt in slices:
            act_e = act_all[s_off : s_off + cnt]
            down_e = F.linear(act_e.to(self.down_exps.dtype), self.down_exps[e]).float()
            out.index_add_(0, tok_s[s_off : s_off + cnt], down_e * w_s[s_off : s_off + cnt, None])
        return out.reshape(orig_shape).to(x.dtype)


class Inkling2LoopController(nn.Module):
    def __init__(self, n_embd: int, gate_rank: int):
        super().__init__()
        self.msg_query = nn.Parameter(torch.empty(4, n_embd, dtype=torch.float32))
        self.ctrl_down = nn.ParameterList([nn.Parameter(torch.empty(gate_rank, 2 * n_embd, dtype=torch.float32)) for _ in range(4)])
        self.ctrl_gain = nn.ParameterList([nn.Parameter(torch.empty(n_embd, gate_rank, dtype=torch.float32)) for _ in range(4)])
        self.ctrl_forget = nn.ParameterList([nn.Parameter(torch.empty(n_embd, gate_rank, dtype=torch.float32)) for _ in range(4)])

    def control(self, cur: torch.Tensor, message: torch.Tensor, ctrl: int) -> torch.Tensor:
        gate_in = torch.cat([_inkling2_rms_norm(cur, None, 1e-6), _inkling2_rms_norm(message, None, 1e-6)], dim=-1)
        gate = F.linear(gate_in.to(self.ctrl_down[ctrl].dtype), self.ctrl_down[ctrl])
        gain = torch.tanh(F.linear(gate.to(self.ctrl_gain[ctrl].dtype), self.ctrl_gain[ctrl]).float())
        forget = torch.tanh(F.linear(gate.to(self.ctrl_forget[ctrl].dtype), self.ctrl_forget[ctrl]).float())
        return (cur.float() - forget * cur.float() + gain * message.float()).to(cur.dtype)


class Inkling2Attention(nn.Module):
    def __init__(
        self,
        il: int,
        n_embd: int,
        n_head: int,
        n_head_kv: int,
        head_dim: int,
        n_rel: int,
        n_rel_ext: int,
        n_swa: int,
        n_conv: int,
        is_swa: bool,
        rms_eps: float,
        f_qk_clip: float,
        f_rel_clip: float,
    ):
        super().__init__()
        self.il = il
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_head_kv = n_head_kv
        self.head_dim = head_dim
        self.n_rel = n_rel
        self.n_extent = n_swa if is_swa else n_rel_ext
        self.n_swa = n_swa
        self.n_conv = n_conv
        self.is_swa = is_swa
        self.rms_eps = rms_eps
        self.f_qk_clip = f_qk_clip
        self.f_rel_clip = f_rel_clip

        self.wq = nn.Parameter(torch.empty(n_head * head_dim, n_embd, dtype=torch.float32))
        self.wk = nn.Parameter(torch.empty(n_head_kv * head_dim, n_embd, dtype=torch.float32))
        self.wv = nn.Parameter(torch.empty(n_head_kv * head_dim, n_embd, dtype=torch.float32))
        self.wo = nn.Parameter(torch.empty(n_embd, n_head * head_dim, dtype=torch.float32))
        self.wqkv_gate = nn.Parameter(torch.empty(n_head * head_dim, n_embd, dtype=torch.float32))

        self.attn_q_clip = nn.Parameter(torch.empty(n_head, head_dim, dtype=torch.float32))
        self.attn_sink_proj = nn.Parameter(torch.empty(n_head, n_embd, dtype=torch.float32))
        self.attn_rel = nn.Parameter(torch.empty(n_rel * n_head, n_embd, dtype=torch.float32))
        self.attn_rel_bias = nn.Parameter(torch.empty(self.n_extent, n_rel, dtype=torch.float32))
        self.attn_rel_clip = nn.Parameter(torch.empty(n_head, n_rel, dtype=torch.float32))

        self.attn_k_conv = nn.Parameter(torch.empty(n_head_kv * head_dim, n_conv, dtype=torch.float32))
        self.attn_v_conv = nn.Parameter(torch.empty(n_head_kv * head_dim, n_conv, dtype=torch.float32))
        self.attn_post_conv = nn.Parameter(torch.empty(n_embd, n_conv, dtype=torch.float32))

        if is_swa:
            self.attn_norm = nn.Parameter(torch.empty(n_embd, dtype=torch.float32))
            self.attn_q_inp_norm = None
            self.attn_k_inp_norm = None
            self.attn_v_inp_norm = None
            self.attn_k_conv_proj = None
            self.attn_v_conv_proj = None
        else:
            self.attn_norm = None
            self.attn_q_inp_norm = nn.Parameter(torch.empty(n_embd, dtype=torch.float32))
            self.attn_k_inp_norm = nn.Parameter(torch.empty(n_embd, dtype=torch.float32))
            self.attn_v_inp_norm = nn.Parameter(torch.empty(n_embd, dtype=torch.float32))
            self.attn_k_conv_proj = nn.Parameter(torch.empty(n_conv * n_head_kv, n_embd, dtype=torch.float32))
            self.attn_v_conv_proj = nn.Parameter(torch.empty(n_conv * n_head_kv, n_embd, dtype=torch.float32))

        self.attn_res_query = nn.Parameter(torch.empty(1 if is_swa else 3, n_embd, dtype=torch.float32)) if il > 0 else None

    def forward(
        self,
        cur: torch.Tensor,
        blocks: list[torch.Tensor],
        pos: torch.Tensor,
        tau: torch.Tensor,
        attn_bool: torch.Tensor,
        messages: torch.Tensor | None = None,
        loop: Inkling2LoopController | None = None,
    ) -> torch.Tensor:
        B, L, _ = cur.shape
        if self.attn_res_query is not None:
            mixed = _inkling2_mix(blocks, cur, self.attn_res_query, self.n_embd)
            if self.is_swa:
                x0 = mixed[:, :, 0, :]
                inputs = [x0, x0, x0]
            else:
                inputs = [mixed[:, :, 0, :], mixed[:, :, 1, :], mixed[:, :, 2, :]]
        else:
            inputs = [cur, cur, cur]

        if messages is not None and loop is not None:
            inputs = [loop.control(inputs[j], messages[:, :, j, :], j) for j in range(3)]

        if self.is_swa and messages is not None:
            inputs = [_inkling2_rms_norm(inputs[j], self.attn_norm, self.rms_eps) for j in range(3)]
        elif self.is_swa:
            n0 = _inkling2_rms_norm(inputs[0], self.attn_norm, self.rms_eps)
            inputs = [n0, n0, n0]
        else:
            inputs = [
                _inkling2_rms_norm(inputs[0], self.attn_q_inp_norm, self.rms_eps),
                _inkling2_rms_norm(inputs[1], self.attn_k_inp_norm, self.rms_eps),
                _inkling2_rms_norm(inputs[2], self.attn_v_inp_norm, self.rms_eps),
            ]

        q = F.linear(inputs[0].to(self.wq.dtype), self.wq)
        k_proj = F.linear(inputs[1].to(self.wk.dtype), self.wk)
        k_dyn = None if self.is_swa else F.linear(inputs[1].to(self.attn_k_conv_proj.dtype), self.attn_k_conv_proj)
        k = _inkling2_convolve(k_proj, self.attn_k_conv, k_dyn, self.n_conv, attn_bool)
        v_proj = F.linear(inputs[2].to(self.wv.dtype), self.wv)
        v_dyn = None if self.is_swa else F.linear(inputs[2].to(self.attn_v_conv_proj.dtype), self.attn_v_conv_proj)
        v = _inkling2_convolve(v_proj, self.attn_v_conv, v_dyn, self.n_conv, attn_bool)
        r = F.linear(inputs[0].to(self.attn_rel.dtype), self.attn_rel)

        q = _inkling2_clip(q.view(B, L, self.n_head, self.head_dim), self.attn_q_clip, self.f_qk_clip)
        k = _inkling2_clip(k.view(B, L, self.n_head_kv, self.head_dim), None, self.f_qk_clip)
        v = _inkling2_rms_norm(v.view(B, L, self.n_head_kv, self.head_dim), None, self.rms_eps)
        r = _inkling2_clip(r.view(B, L, self.n_head, self.n_rel), self.attn_rel_clip, self.f_rel_clip)

        if not self.is_swa:
            tau_b = tau[:, :, None, None]
            q = (q.float() * tau_b).to(q.dtype)
            r = (r.float() * tau_b).to(r.dtype)

        sinks = F.linear(inputs[0].to(self.attn_sink_proj.dtype), self.attn_sink_proj)
        bias_pad = F.pad(self.attn_rel_bias.float(), (0, 0, 0, 1))  # [n_extent + 1, n_rel]
        table = torch.matmul(r.float(), bias_pad.t()) - sinks.float().unsqueeze(-1)  # [B, L, H, n_extent + 1]

        dist = pos[:, :, None] - pos[:, None, :]
        rel_pos = torch.where((dist >= 0) & (dist < self.n_extent), dist, self.n_extent)
        kq_b = torch.gather(table, dim=3, index=rel_pos.unsqueeze(2).expand(B, L, self.n_head, L)).permute(0, 2, 1, 3)

        if self.n_head != self.n_head_kv:
            rep = self.n_head // self.n_head_kv
            k_exp = k.unsqueeze(3).expand(B, L, self.n_head_kv, rep, self.head_dim).reshape(B, L, self.n_head, self.head_dim)
            v_exp = v.unsqueeze(3).expand(B, L, self.n_head_kv, rep, self.head_dim).reshape(B, L, self.n_head, self.head_dim)
        else:
            k_exp, v_exp = k, v

        scores = torch.matmul(q.permute(0, 2, 1, 3).float(), k_exp.permute(0, 2, 3, 1).float()) * (1.0 / math.sqrt(self.head_dim)) + kq_b

        idx = torch.arange(L, device=cur.device)
        allowed = (idx[None, :] <= idx[:, None]).unsqueeze(0).unsqueeze(0) & attn_bool[:, None, None, :]
        if self.is_swa:
            allowed = allowed & ((pos[:, :, None] - pos[:, None, :]) < self.n_swa).unsqueeze(1)
        scores = scores.masked_fill(~allowed, float("-inf"))

        m = torch.clamp(scores.max(dim=-1, keepdim=True).values, min=0.0)
        ex = torch.exp(scores - m)
        probs = ex / (ex.sum(dim=-1, keepdim=True) + torch.exp(-m))

        attn_v = torch.matmul(probs, v_exp.permute(0, 2, 1, 3).float()).permute(0, 2, 1, 3).reshape(B, L, self.n_head * self.head_dim).to(inputs[0].dtype)
        gate = torch.sigmoid(F.linear(inputs[0].to(self.wqkv_gate.dtype), self.wqkv_gate).float()).to(attn_v.dtype)
        wo_out = F.linear((attn_v * gate).to(self.wo.dtype), self.wo)
        return _inkling2_convolve(wo_out, self.attn_post_conv, None, self.n_conv, attn_bool)


class Inkling2Layer(nn.Module):
    def __init__(
        self,
        il: int,
        n_embd: int,
        n_head: int,
        n_head_kv: int,
        head_dim: int,
        n_ff_exp: int,
        n_expert: int,
        n_expert_used: int,
        n_rel: int,
        n_rel_ext: int,
        n_swa: int,
        n_conv: int,
        is_swa: bool,
        rms_eps: float,
        f_qk_clip: float,
        f_rel_clip: float,
        situ_beta: float,
        situ_linear_beta: float,
        expert_weights_scale: float,
        block_size: int,
        has_exp_probs_b: bool,
    ):
        super().__init__()
        self.il = il
        self.n_embd = n_embd
        self.n_conv = n_conv
        self.rms_eps = rms_eps
        self.block_size = block_size
        self.self_attn = Inkling2Attention(
            il, n_embd, n_head, n_head_kv, head_dim, n_rel, n_rel_ext, n_swa, n_conv, is_swa, rms_eps, f_qk_clip, f_rel_clip
        )
        self.mlp = Inkling2MoE(
            n_embd, n_ff_exp, n_expert, n_expert_used, situ_beta, situ_linear_beta, expert_weights_scale, has_exp_probs_b
        )
        self.ffn_res_query = nn.Parameter(torch.empty(n_embd, dtype=torch.float32))
        self.ffn_norm = nn.Parameter(torch.empty(n_embd, dtype=torch.float32))
        self.ffn_post_conv = nn.Parameter(torch.empty(n_embd, n_conv, dtype=torch.float32))

    def forward(
        self,
        ip: int,
        cur: torch.Tensor,
        blocks: list[torch.Tensor],
        pos: torch.Tensor,
        tau: torch.Tensor,
        attn_bool: torch.Tensor,
        messages: torch.Tensor | None = None,
        loop: Inkling2LoopController | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        inp_sa = cur
        attn_out = self.self_attn(cur, blocks, pos, tau, attn_bool, messages=messages, loop=loop)
        if ip % self.block_size == 0:
            new_block = inp_sa
            ffn_inp = attn_out
            blocks_for_ffn = [*blocks, new_block]
        else:
            new_block = None
            ffn_inp = inp_sa + attn_out
            blocks_for_ffn = blocks

        ffn_mix = _inkling2_mix(blocks_for_ffn, ffn_inp, self.ffn_res_query.unsqueeze(0), self.n_embd)[:, :, 0, :]
        if messages is not None and loop is not None:
            ffn_mix = loop.control(ffn_mix, messages[:, :, 3, :], 3)
        ffn_in = _inkling2_rms_norm(ffn_mix, self.ffn_norm, self.rms_eps)
        moe_out = self.mlp(ffn_in)
        ffn_out = _inkling2_convolve(moe_out, self.ffn_post_conv, None, self.n_conv, attn_bool)
        return ffn_inp + ffn_out, new_block


class Inkling2LoopBackbone(nn.Module):
    def __init__(
        self,
        config: SimpleNamespace,
        n_vocab_padded: int,
        n_embd: int,
        rms_eps: float,
        log_scale_floor: int,
        log_scale_alpha: float,
        layers: nn.ModuleList,
        loop_layers: nn.ModuleList,
    ):
        super().__init__()
        self.config = config
        self.n_embd = n_embd
        self.log_scale_floor = log_scale_floor
        self.log_scale_alpha = log_scale_alpha
        self.loop_start = int(getattr(config, "loop_start", 0) or 0)
        self.loop_end = int(getattr(config, "loop_end", 0) or 0)
        self.loop_count = int(getattr(config, "loop_count", 1) or 1)
        self._grad_ckpt = False
        self._gc_kwargs: dict | None = None

        self.embed_tokens = nn.Embedding(n_vocab_padded, n_embd, dtype=torch.float32)
        self.tok_norm = Inkling2RMSNorm(n_embd, rms_eps)
        self.layers = layers
        self.loop_layers = loop_layers
        self.output_res_query = nn.Parameter(torch.empty(n_embd, dtype=torch.float32))
        self.norm = Inkling2RMSNorm(n_embd, rms_eps)

    def _pos_and_tau(self, attention_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        attn_bool = attention_mask > 0
        pos = torch.clamp(attn_bool.long().cumsum(dim=-1) - 1, min=0)
        tau = 1.0 + self.log_scale_alpha * torch.log(torch.clamp((pos.float() + 1.0) / float(self.log_scale_floor), min=1.0))
        return pos, attn_bool, tau

    def _run_layer_ckpt(
        self,
        layer: Inkling2Layer,
        ip: int,
        h: torch.Tensor,
        blocks: list[torch.Tensor],
        pos: torch.Tensor,
        tau: torch.Tensor,
        attn_bool: torch.Tensor,
        messages: torch.Tensor | None = None,
        loop: Inkling2LoopController | None = None,
        grad_ckpt: bool = False,
        gc_kwargs: dict | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if grad_ckpt and self.training and torch.is_grad_enabled():
            n_b = len(blocks)

            def _fn(cur_in: torch.Tensor, *blk_args: torch.Tensor):
                res_h, res_b = layer(ip, cur_in, list(blk_args[:n_b]), pos, tau, attn_bool, messages=messages, loop=loop)
                if res_b is None:
                    return (res_h,)
                return (res_h, res_b)

            kw = dict(gc_kwargs or {})
            kw.setdefault("use_reentrant", False)
            out = torch.utils.checkpoint.checkpoint(_fn, h, *blocks, **kw)
            return (out[0], out[1]) if len(out) == 2 else (out[0], None)
        return layer(ip, h, blocks, pos, tau, attn_bool, messages=messages, loop=loop)

    def forward_prefix(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        start_layer: int,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        pos, attn_bool, tau = self._pos_and_tau(attention_mask)
        h = self.tok_norm(self.embed_tokens(input_ids))
        blocks: list[torch.Tensor] = []
        has_loop = self.loop_end > self.loop_start and self.loop_layers is not None and len(self.loop_layers) > 0 and start_layer >= self.loop_end
        pre_end = self.loop_start if has_loop else start_layer
        for ip in range(pre_end):
            h, new_b = self.layers[ip](ip, h, blocks, pos, tau, attn_bool)
            if new_b is not None:
                blocks.append(new_b)
        if has_loop:
            prelude = h
            prelude_blocks = list(blocks)
            for pass_idx in range(self.loop_count):
                if pass_idx > 0:
                    bank = blocks[len(prelude_blocks) + 1 :]
                    bank_out = h
                    h = prelude
                    blocks = list(prelude_blocks)
                for ip in range(self.loop_start, self.loop_end):
                    if pass_idx == 0:
                        h, new_b = self.layers[ip](ip, h, blocks, pos, tau, attn_bool)
                    else:
                        loop_mod = self.loop_layers[ip - self.loop_start]
                        msgs = _inkling2_mix(bank, bank_out, loop_mod.msg_query, self.n_embd)
                        h, new_b = self.layers[ip](ip, h, blocks, pos, tau, attn_bool, messages=msgs, loop=loop_mod)
                    if new_b is not None:
                        blocks.append(new_b)
            for ip in range(self.loop_end, start_layer):
                h, new_b = self.layers[ip](ip, h, blocks, pos, tau, attn_bool)
                if new_b is not None:
                    blocks.append(new_b)
        return h, blocks

    def forward_tail(
        self,
        h: torch.Tensor,
        blocks: list[torch.Tensor],
        attention_mask: torch.Tensor,
        start_layer: int,
        grad_ckpt: bool = False,
        gc_kwargs: dict | None = None,
    ) -> torch.Tensor:
        pos, attn_bool, tau = self._pos_and_tau(attention_mask)
        for ip in range(start_layer, len(self.layers)):
            h, new_b = self._run_layer_ckpt(
                self.layers[ip], ip, h, blocks, pos, tau, attn_bool,
                grad_ckpt=grad_ckpt, gc_kwargs=gc_kwargs,
            )
            if new_b is not None:
                blocks.append(new_b)
        return _inkling2_mix(blocks, h, self.output_res_query.unsqueeze(0), self.n_embd)[:, :, 0, :]

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None, use_cache: bool = False, **_):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.long)
        pos, attn_bool, tau = self._pos_and_tau(attention_mask)
        h = self.tok_norm(self.embed_tokens(input_ids))
        blocks: list[torch.Tensor] = []
        has_loop = self.loop_end > self.loop_start and self.loop_layers is not None and len(self.loop_layers) > 0
        n_phys = len(self.layers)
        pre_end = self.loop_start if has_loop else n_phys
        for ip in range(pre_end):
            h, new_b = self._run_layer_ckpt(
                self.layers[ip], ip, h, blocks, pos, tau, attn_bool,
                grad_ckpt=self._grad_ckpt, gc_kwargs=self._gc_kwargs,
            )
            if new_b is not None:
                blocks.append(new_b)
        if has_loop:
            prelude = h
            prelude_blocks = list(blocks)
            for pass_idx in range(self.loop_count):
                if pass_idx > 0:
                    bank = blocks[len(prelude_blocks) + 1 :]
                    bank_out = h
                    h = prelude
                    blocks = list(prelude_blocks)
                for ip in range(self.loop_start, self.loop_end):
                    if pass_idx == 0:
                        h, new_b = self._run_layer_ckpt(
                            self.layers[ip], ip, h, blocks, pos, tau, attn_bool,
                            grad_ckpt=self._grad_ckpt, gc_kwargs=self._gc_kwargs,
                        )
                    else:
                        loop_mod = self.loop_layers[ip - self.loop_start]
                        msgs = _inkling2_mix(bank, bank_out, loop_mod.msg_query, self.n_embd)
                        h, new_b = self._run_layer_ckpt(
                            self.layers[ip], ip, h, blocks, pos, tau, attn_bool,
                            messages=msgs, loop=loop_mod,
                            grad_ckpt=self._grad_ckpt, gc_kwargs=self._gc_kwargs,
                        )
                    if new_b is not None:
                        blocks.append(new_b)
            for ip in range(self.loop_end, n_phys):
                h, new_b = self._run_layer_ckpt(
                    self.layers[ip], ip, h, blocks, pos, tau, attn_bool,
                    grad_ckpt=self._grad_ckpt, gc_kwargs=self._gc_kwargs,
                )
                if new_b is not None:
                    blocks.append(new_b)
        h = _inkling2_mix(blocks, h, self.output_res_query.unsqueeze(0), self.n_embd)[:, :, 0, :]
        h = self.norm(h)
        return SimpleNamespace(last_hidden_state=h)


class Inkling2LoopStudent(nn.Module):
    def __init__(self, config: SimpleNamespace, model: Inkling2LoopBackbone, n_embd: int, n_vocab_unpadded: int):
        super().__init__()
        self.config = config
        self.model = model
        self.lm_head = nn.Linear(n_embd, n_vocab_unpadded, bias=False, dtype=torch.float32)
        self.is_gradient_checkpointing = False
        self._gc_kwargs: dict | None = None

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs: dict | None = None) -> None:
        self.is_gradient_checkpointing = True
        self._gc_kwargs = dict(gradient_checkpointing_kwargs) if gradient_checkpointing_kwargs else None
        self.model._grad_ckpt = True
        self.model._gc_kwargs = self._gc_kwargs


def _build_inkling2_student(
    art: lattice.Artifact,
    kvs: dict,
    arch: str,
    ggml_lib: str | None,
    device: str = "cpu",
    frozen_bf16: bool = False,
) -> Inkling2LoopStudent:
    p = f"{arch}."
    n_phys = int(kvs[f"{p}block_count"])
    n_embd = int(kvs[f"{p}embedding_length"])
    n_vocab_padded = int(kvs[f"{p}vocab_size"])
    n_vocab_unpadded = int(kvs[f"{p}vocab_size_unpadded"])
    n_expert = int(kvs[f"{p}expert_count"])
    rms_eps = float(kvs[f"{p}attention.layer_norm_rms_epsilon"])
    f_qk_clip = float(kvs[f"{p}attention.qk_clip"])
    block_size = int(kvs[f"{p}attn_res.block_size"])
    n_conv = int(kvs[f"{p}shortconv.l_cache"])
    n_rel = int(kvs[f"{p}attention.relative_length"])
    n_rel_ext = int(kvs[f"{p}attention.relative_extent"])
    f_rel_clip = float(kvs[f"{p}attention.relative_clip"])
    log_scale_floor = int(kvs[f"{p}attention.log_scale_floor"])
    log_scale_alpha = float(kvs[f"{p}attention.log_scale_alpha"])
    situ_beta = float(kvs[f"{p}activation.situ_beta"])
    situ_linear_beta = float(kvs[f"{p}activation.situ_linear_beta"])
    n_swa = int(kvs[f"{p}attention.sliding_window"])
    swa_pattern = [bool(x) for x in kvs[f"{p}attention.sliding_window_pattern"]]
    expert_weights_scale = float(kvs.get(f"{p}expert_weights_scale", 0.0) or 0.0)
    softcap = float(kvs.get(f"{p}final_logit_softcapping", 0.0) or 0.0)

    def _per_layer(key: str, default=None) -> list[int]:
        v = kvs.get(key, default)
        if isinstance(v, (list, tuple)):
            return [int(v[i]) for i in range(n_phys)]
        return [int(v)] * n_phys

    n_head_arr = _per_layer(f"{p}attention.head_count")
    n_head_kv_arr = _per_layer(f"{p}attention.head_count_kv", kvs[f"{p}attention.head_count"])
    n_expert_used_arr = _per_layer(f"{p}expert_used_count")
    n_ff_exp_arr = _per_layer(f"{p}expert_feed_forward_length")
    head_dim_full = int(kvs.get(f"{p}attention.key_length") or (n_embd // n_head_arr[0]))
    head_dim_swa = int(kvs.get(f"{p}attention.key_length_swa") or head_dim_full)

    if arch == "inkling2_loop":
        loop_start = int(kvs[f"{p}loop.start"])
        loop_end = int(kvs[f"{p}loop.end"])
        loop_count = int(kvs[f"{p}loop.count"])
        gate_rank = int(kvs[f"{p}loop.gate_rank"])
    else:
        loop_start = 0
        loop_end = 0
        loop_count = 1
        gate_rank = 0

    cfg = SimpleNamespace(
        model_type=arch,
        num_hidden_layers=n_phys,
        hidden_size=n_embd,
        final_logit_softcapping=softcap,
        loop_start=loop_start,
        loop_end=loop_end,
        loop_count=loop_count,
        attn_res_block_size=block_size,
    )

    layers = nn.ModuleList()
    for il in range(n_phys):
        is_swa = swa_pattern[il]
        hd = head_dim_swa if is_swa else head_dim_full
        layers.append(
            Inkling2Layer(
                il=il,
                n_embd=n_embd,
                n_head=n_head_arr[il],
                n_head_kv=n_head_kv_arr[il],
                head_dim=hd,
                n_ff_exp=n_ff_exp_arr[il],
                n_expert=n_expert,
                n_expert_used=n_expert_used_arr[il],
                n_rel=n_rel,
                n_rel_ext=n_rel_ext,
                n_swa=n_swa,
                n_conv=n_conv,
                is_swa=is_swa,
                rms_eps=rms_eps,
                f_qk_clip=f_qk_clip,
                f_rel_clip=f_rel_clip,
                situ_beta=situ_beta,
                situ_linear_beta=situ_linear_beta,
                expert_weights_scale=expert_weights_scale,
                block_size=block_size,
                has_exp_probs_b=(f"blk.{il}.exp_probs_b.bias" in art.tensors),
            )
        )

    loop_layers = nn.ModuleList(
        [Inkling2LoopController(n_embd, gate_rank) for _ in range(loop_start, loop_end)]
        if arch == "inkling2_loop" and loop_end > loop_start
        else []
    )

    backbone = Inkling2LoopBackbone(cfg, n_vocab_padded, n_embd, rms_eps, log_scale_floor, log_scale_alpha, layers, loop_layers)
    student = Inkling2LoopStudent(cfg, backbone, n_embd, n_vocab_unpadded)

    f32_params: set[int] = set()

    def _load(param: torch.Tensor, gg_name: str, sl: slice | None = None) -> None:
        if art.tensors[gg_name]["type"] == "F32":
            f32_params.add(id(param))
        arr = floats(art, gg_name, ggml_lib).copy()
        if sl is not None:
            arr = arr[sl]
        t = torch.from_numpy(arr)
        if t.shape != param.shape and tuple(d for d in param.shape if d != 1) == tuple(d for d in t.shape if d != 1):
            t = t.reshape(param.shape)
        assert param.shape == t.shape, (gg_name, param.shape, t.shape)
        param.data.copy_(t)

    _load(backbone.embed_tokens.weight, "token_embd.weight")
    _load(backbone.tok_norm.weight, "token_embd_norm.weight")
    _load(student.lm_head.weight, "output.weight", slice(0, n_vocab_unpadded))
    _load(backbone.norm.weight, "output_norm.weight")
    _load(backbone.output_res_query, "output_res_query.weight")

    for il, layer in enumerate(layers):
        bp = f"blk.{il}."
        sa = layer.self_attn
        _load(sa.wq, bp + "attn_q.weight")
        _load(sa.wk, bp + "attn_k.weight")
        _load(sa.wv, bp + "attn_v.weight")
        _load(sa.wo, bp + "attn_output.weight")
        _load(sa.wqkv_gate, bp + "attn_gate.weight")
        _load(sa.attn_q_clip, bp + "attn_q_clip.weight")
        _load(sa.attn_sink_proj, bp + "attn_sink_proj.weight")
        _load(sa.attn_rel, bp + "attn_rel.weight")
        _load(sa.attn_rel_bias, bp + "attn_rel_bias.weight")
        _load(sa.attn_rel_clip, bp + "attn_rel_clip.weight")
        _load(sa.attn_k_conv, bp + "attn_k_conv.weight")
        _load(sa.attn_v_conv, bp + "attn_v_conv.weight")
        _load(sa.attn_post_conv, bp + "attn_post_conv.weight")
        _load(layer.ffn_post_conv, bp + "ffn_post_conv.weight")
        if sa.is_swa:
            _load(sa.attn_norm, bp + "attn_norm.weight")
        else:
            _load(sa.attn_q_inp_norm, bp + "attn_q_inp_norm.weight")
            _load(sa.attn_k_inp_norm, bp + "attn_k_inp_norm.weight")
            _load(sa.attn_v_inp_norm, bp + "attn_v_inp_norm.weight")
            _load(sa.attn_k_conv_proj, bp + "attn_k_conv_proj.weight")
            _load(sa.attn_v_conv_proj, bp + "attn_v_conv_proj.weight")
        if sa.attn_res_query is not None:
            _load(sa.attn_res_query, bp + "attn_res_query.weight")
        _load(layer.ffn_res_query, bp + "ffn_res_query.weight")
        _load(layer.ffn_norm, bp + "ffn_norm.weight")
        _load(layer.mlp.gate_inp, bp + "ffn_gate_inp.weight")
        if layer.mlp.exp_probs_b is not None:
            _load(layer.mlp.exp_probs_b, bp + "exp_probs_b.bias")
        _load(layer.mlp.gate_up_exps, bp + "ffn_gate_up_exps.weight")
        _load(layer.mlp.down_exps, bp + "ffn_down_exps.weight")

    if arch == "inkling2_loop":
        for idx_l, lc in enumerate(loop_layers):
            ip = loop_start + idx_l
            bp = f"blk.{ip}."
            _load(lc.msg_query, bp + "loop_msg_query.weight")
            for c_idx, cname in enumerate(("q", "k", "v", "ffn")):
                _load(lc.ctrl_down[c_idx], bp + f"loop_{cname}_ctrl_down.weight")
                _load(lc.ctrl_gain[c_idx], bp + f"loop_{cname}_ctrl_gain.weight")
                _load(lc.ctrl_forget[c_idx], bp + f"loop_{cname}_ctrl_forget.weight")

    for param in student.parameters():
        param.requires_grad_(False)
    if frozen_bf16:
        for param in student.parameters():
            if param.ndim >= 2 and param.dtype == torch.float32 and id(param) not in f32_params:
                param.data = param.data.to(torch.bfloat16)
    student.to(device)
    student.eval()
    return student


def build_student(art: lattice.Artifact, config_dir: str | Path | None, train: list[str], ggml_lib: str | None, device: str = "cpu",
                  latents: dict[str, np.ndarray] | None = None, frozen_bf16: bool = False):
    """The file as a PyTorch model. `latents` holds float32 arrays in torch shape, one per `train` tensor, the floats this file
    was built from: they sit behind its grids in place of the grid centres, so what training learned below one level survives
    a rebuild. The rounded student must still equal the file, level for level."""
    from torch.nn.utils import parametrize
    from transformers import AutoConfig, AutoModelForCausalLM

    kvs = art.h.parsed_kvs(skip_large_arrays=True)
    arch = str(getattr(art.h, "arch", None) or kvs.get("general.architecture", "") or "")
    if arch in ("inkling2", "inkling2_loop"):
        if train or latents:
            raise NotImplementedError("grid-level learning is not supported for inkling2/inkling2_loop; use learn_gates_seq")
        return _build_inkling2_student(art, kvs, arch, ggml_lib, device=device, frozen_bf16=frozen_bf16), {}
    if not config_dir:
        raise ValueError(f"config_dir is required to build student for architecture {arch!r}")
    config_dir = str(config_dir)

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
        p_parts = name.split(".")
        if len(p_parts) >= 3 and p_parts[0] == "blk" and p_parts[2] in ("attn_qkv_gate", "ssm_beta_alpha"):
            il = p_parts[1]
            v = torch.from_numpy(floats(art, name, ggml_lib).copy())
            if p_parts[2] == "attn_qkv_gate":
                n_z = num_v_heads * head_v_dim
                n_qkv = v.shape[0] - n_z
                split_pairs = (
                    (f"model.layers.{il}.linear_attn.in_proj_qkv.weight", v[:n_qkv]),
                    (f"model.layers.{il}.linear_attn.in_proj_z.weight", v[n_qkv:]),
                )
            else:
                split_pairs = (
                    (f"model.layers.{il}.linear_attn.in_proj_b.weight", v[:num_v_heads]),
                    (f"model.layers.{il}.linear_attn.in_proj_a.weight", v[num_v_heads:]),
                )
            for hfn_s, part in split_pairs:
                part = part.contiguous()
                if unreorder_gdn_v:
                    if ".in_proj_qkv." in hfn_s:
                        part = torch.cat([part[: 2 * qk_half], _unreorder_v(part[2 * qk_half :], 0, head_v_dim)], dim=0)
                    elif ".in_proj_z." in hfn_s:
                        part = _unreorder_v(part, 0, head_v_dim)
                    elif ".in_proj_a." in hfn_s or ".in_proj_b." in hfn_s:
                        part = _unreorder_v(part, 0, 1)
                p = params[hfn_s]
                assert p.shape == part.shape, (name, hfn_s, p.shape, part.shape)
                p.data = part
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
    t_val = float(temp)
    if not math.isfinite(t_val) or t_val <= 0:
        raise ValueError(f"kl_vector requires finite temp > 0, got {temp!r}")
    logq = torch.log_softmax(_head_logits(model, h) / t_val, dim=-1)
    logt = torch.log_softmax(_head_logits(model, target_h) / t_val, dim=-1)
    t = logt.exp()
    return (t * (logt - logq)).sum(dim=-1).mean() * (t_val * t_val)


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

    budget = 40000 if cfg.get("score_budget") is None else int(cfg["score_budget"])
    if anchor is None:
        base = torch.from_numpy(score_all(model, replay, pair, device, amp, budget, prefix_cache=prefix_cache)).float().to(device)
    else:
        base = torch.from_numpy(np.asarray(anchor["base"], dtype=np.float32)).to(device)
        assert len(base) == len(replay)
    adam_eps = 1e-8 if cfg.get("adam_eps") is None else float(cfg["adam_eps"])
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0, eps=adam_eps)
    flat = [w for g in params for w in g["params"]] if isinstance(params[0], dict) else params
    rng = random.Random(20261031 if cfg.get("seed") is None else cfg["seed"])
    steps = 200 if cfg.get("steps") is None else int(cfg["steps"])
    nf = 8 if cfg.get("feedback_batch") is None else int(cfg["feedback_batch"])
    nr = 16 if cfg.get("replay_batch") is None else int(cfg["replay_batch"])
    if nf < 0 or nr < 0:
        raise ValueError(f"feedback_batch and replay_batch must be >= 0, got {nf!r}, {nr!r}")
    zu = cfg.get("z_units")
    unit = (lambda s, c=zu: (s - c["cut"]) / (c["med"] - c["cut"]) / zu["tau"]) if zu else (lambda s: s)
    if zu:
        base, cutoff = unit(base, anchor or zu), 0.0
    margin = (0.0 if zu else 1.0) if cfg.get("margin") is None else float(cfg["margin"])
    hold = 1.0 if cfg.get("replay_weight") is None else float(cfg["replay_weight"])
    pair_margin = (0.0 if zu else 2.0) if cfg.get("pair_margin") is None else float(cfg["pair_margin"])
    pair_weight = 1.0 if cfg.get("pair_weight") is None else float(cfg["pair_weight"])
    by_pair, shares = {}, {}
    for r in feedback:
        if r.get("pair") is not None:
            by_pair.setdefault(r["pair"], {})[r["label"]] = r
        if r.get("share"):
            shares.setdefault(r["share"], []).append(r)
    pairs = [(v["human"], v["ai"]) for v in by_pair.values() if len(v) == 2]
    strata_eff = list(strata) if strata is not None else ["all"] * len(replay)
    pools = {}
    for i, g in enumerate(strata_eff):
        pools.setdefault(g, []).append(i)
    names_s, names_f = sorted(pools), sorted(shares)
    guard = 0.0 if cfg.get("guard_weight") is None else float(cfg["guard_weight"])
    log(f"learn: {sum(w.numel() for w in flat):,} parameters, {len(feedback)} feedback, {len(pairs)} pairs, {len(replay)} replay")
    kind = cfg.get("hold_kind")
    cap = (3072 if amp else None) if cfg.get("train_budget") is None else cfg["train_budget"]
    vec = kind == "klvec"
    if vec:
        if not replay:
            h0 = torch.empty((0, 0), device=device)
        elif anchor is not None and "h0" in anchor:
            h0 = torch.from_numpy(np.asarray(anchor["h0"], dtype=np.float32)).to(device)
        else:
            hs = []
            with torch.no_grad():
                for b in batches(replay, budget):
                    hs.append((b, vec_scores(model, [replay[i] for i in b], pair, device, amp, prefix_cache=prefix_cache)[1]))
            h0 = torch.zeros(len(replay), hs[0][1].shape[1], device=device)
            for b, h in hs:
                h0[b] = h
        y0 = torch.tensor([1.0 if g.startswith("human") else -1.0 for g in strata_eff], device=device)
        right = ((base > cutoff).float() * 2 - 1) == y0
        fallback_h = h0.mean(0) if len(h0) > 0 else None
        centre = {}
        for v, cls in ((1.0, y0 > 0), (-1.0, y0 < 0)):
            good = right & cls
            if good.any():
                centre[v] = h0[good].mean(0)
            elif cls.any():
                centre[v] = h0[cls].mean(0)
            elif fallback_h is not None:
                centre[v] = fallback_h
        log(f"learn: vector KL, {int(right.sum())} of {len(replay)} replay documents start right")
    keep_tau = 0.5 if cfg.get("keep_tau") is None else float(cfg["keep_tau"])
    keep_window = 64 if cfg.get("keep_window") is None else int(cfg["keep_window"])
    tau, recent = zu["tau"] if zu else keep_tau, {g: deque(maxlen=keep_window) for g in names_s}
    kl_temp_vec = 4.0 if cfg.get("kl_temp") is None else float(cfg["kl_temp"])
    kl_temp_hold = 1.0 if cfg.get("kl_temp") is None else float(cfg["kl_temp"])
    sep_humans = "human" if cfg.get("sep_humans") is None else str(cfg["sep_humans"])
    hist, conflicts = [], 0
    for step in range(steps):
        if nf <= 0 or not feedback:
            fb = []
        elif shares:
            if nf >= len(names_f):
                per_g = nf // len(names_f)
                fb = [r for g in names_f for r in rng.sample(shares[g], min(per_g, len(shares[g])))]
            else:
                fb = [rng.choice(shares[g]) for g in rng.sample(names_f, nf)]
        else:
            fb = rng.sample(feedback, min(nf, len(feedback)))
        ps = rng.sample(pairs, min(nf // 2, len(pairs))) if (pairs and nf >= 2) else []
        idx = []
        if names_s:
            for k in range(nr):
                idx.append(rng.choice(pools[names_s[(step + k) % len(names_s)]]))
        fd, hd, ad, rd = [r["tokens"] for r in fb], [h["tokens"] for h, _ in ps], [a["tokens"] for _, a in ps], [replay[i] for i in idx]
        docs_all = fd + hd + ad + rd
        if cap and docs_all:
            # pack all sub-batches into one length-sorted pass to minimize padding
            sizes = [len(fd), len(hd), len(ad), len(rd)]
            out = capped(vec_scores if vec else scores, model, docs_all, pair, device, amp, cap, prefix_cache=prefix_cache)
            s, sh, sa, sr = (out[0] if vec else out).split(sizes)
            hf, _, _, hr = out[1].split(sizes) if vec else (None,) * 4
        elif cap:
            s = sh = sa = sr = torch.empty(0, device=device)
            hf = hr = None
        else:
            if fd:
                s, hf = vec_scores(model, fd, pair, device, amp, prefix_cache=prefix_cache) if vec else (scores(model, fd, pair, device, amp, prefix_cache=prefix_cache), None)
            else:
                s, hf = torch.empty(0, device=device), None
            if ps:
                sh, sa = scores(model, hd, pair, device, amp, prefix_cache=prefix_cache), scores(model, ad, pair, device, amp, prefix_cache=prefix_cache)
            if rd:
                sr, hr = vec_scores(model, rd, pair, device, amp, prefix_cache=prefix_cache) if vec else (scores(model, rd, pair, device, amp, prefix_cache=prefix_cache), None)
            else:
                sr, hr = torch.empty(0, device=device), None
        s, sr = unit(s), unit(sr)
        if fb:
            y = torch.tensor([1.0 if r["label"] == "human" else -1.0 for r in fb], device=device)
            loss_fb = margin_loss(s, y, cutoff, margin)
            if vec and all(float(v) in centre for v in y):
                loss_fb = loss_fb + kl_vector(model, hf, torch.stack([centre[float(v)] for v in y]), kl_temp_vec)
        else:
            loss_fb = torch.zeros((), device=device)
        loss_pair = torch.zeros((), device=device)
        if ps:
            loss_pair = pair_loss(unit(sh), unit(sa), pair_margin)
        yr = torch.tensor([1.0 if strata_eff[i].startswith("human") else -1.0 for i in idx], device=device) if (idx and kind in ("kl", "oneside")) else None
        if not idx:
            loss_hold = torch.zeros((), device=device)
        elif vec:
            tg = torch.stack([h0[i] if right[i] else centre[float(y0[i])] for i in idx])
            loss_hold = kl_vector(model, hr, tg, kl_temp_vec)
        elif kind == "kl":
            loss_hold = kl_hold(sr, base[idx], yr, cutoff, kl_temp_hold)
        elif kind == "oneside":
            drift = (sr - base[idx]) * (tau if zu else 1.0)
            for i, d in zip(idx, drift.tolist()):
                recent[strata_eff[i]].append(d)
            c = torch.tensor([0.0 if cfg.get("keep_abs") else float(np.median(recent[strata_eff[i]])) for i in idx], device=device)
            loss_hold = keep_loss(drift - c, yr, tau)
        else:
            loss_hold = ((sr - base[idx]) ** 2).mean()
        new = loss_fb + pair_weight * loss_pair
        loss = new + hold * loss_hold
        if cfg.get("batch_pairs") and strata:
            # all-pairs human vs AI margin loss within the replay batch
            hs = [k for k, i in enumerate(idx) if strata[i].startswith(sep_humans)]
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
        has_new = bool(new.requires_grad)
        old = loss - new
        has_old = bool(old.requires_grad)
        if cfg.get("project") and has_new and has_old:
            # A-GEM gradient projection against replay-hold gradients
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
            opt.step()
        elif loss.requires_grad:
            loss.backward()
            opt.step()
        else:
            opt.zero_grad(set_to_none=True)
        hist.append((loss_fb.item(), loss_pair.item(), loss_hold.item()))
        if step % 10 == 0:
            r = hist[-10:]
            log(f"learn step {step}/{steps} fb {sum(x[0] for x in r) / len(r):.4f} pair {sum(x[1] for x in r) / len(r):.4f} hold {sum(x[2] for x in r) / len(r):.4f}")
        if on_step:
            on_step(step, opt, hist[-1])
    if cfg.get("project"):
        log(f"learn: A-GEM projected {conflicts}/{steps} steps")
    return hist


def learn(art_path: str, config_dir: str | Path | None, feedback: list[dict], replay: list[np.ndarray], pair: tuple[int, int],
          cutoff: float, out: str, cfg: dict, log=print, device: str | None = None, strata: list[str] | None = None) -> dict:
    """Train on `feedback` rows {"tokens", "label": "human"|"ai", "pair": group or None}, write `out`, return the lineage.
    `replay` is ordinary documents held near their starting scores. With `strata`, one name per replay document,
    each population gets the same share of every batch, and every "human..." or "ai..." document is also kept on its
    side of the cutoff: a mean penalty alone lets a small population fall through it."""
    import torch

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.startswith("cuda") and (True if cfg.get("amp") is None else bool(cfg["amp"]))
    art = lattice.Artifact(art_path)
    names = train_names(art, cfg)
    model, grids = build_student(art, config_dir, names, cfg.get("ggml_lib"), device)
    ckpt = True if cfg.get("checkpointing") is None else bool(cfg["checkpointing"])
    if ckpt and amp:
        model.gradient_checkpointing_enable()
        model.train()    # HF checkpoints only in training mode; the model has no dropout
    params = [w for w, _, _ in grids.values()]
    lr = 2e-5 if cfg.get("lr") is None else float(cfg["lr"])
    steps_rec = 200 if cfg.get("steps") is None else int(cfg["steps"])
    hist = _train(model, params, feedback, replay, pair, cutoff, cfg, log, device, amp, strata, lr)
    levels, moved = {}, {}
    for name, (w, fq, start) in grids.items():
        lv = fq.levels(w.detach()).reshape(-1).to("cpu", torch.uint8).numpy()
        levels[name], moved[name] = lv, int((lv != start.numpy()).sum())
    lattice.write_levels(art_path, out, levels)
    art.close()
    n = sum(g[0].numel() for g in grids.values())
    rec = {"method": "grid", "steps": steps_rec, "lr": lr, "tensors": names, "weights": n, "levels_moved": sum(moved.values()),
           "fraction_moved": sum(moved.values()) / max(1, n), "loss_first": hist[0] if hist else None, "loss_last": hist[-1] if hist else None,
           "n_feedback": len(feedback), "n_pairs": len({r["pair"] for r in feedback if r.get("pair") is not None}), "n_replay": len(replay),
           "base_sha256": sha256(art_path), "out_sha256": sha256(out), "bytes": Path(out).stat().st_size}
    Path(str(out) + ".lineage.json").write_text(json.dumps(rec, indent=1))
    return rec


def _layer_ffn_width(art: lattice.Artifact, il: int) -> int:
    for suf in ("ffn_down.weight", "ffn_down_exps.weight"):
        tname = f"blk.{il}.{suf}"
        if tname in art.tensors:
            return int(art.tensors[tname]["dims"][0])
    for suf in ("ffn_up.weight", "ffn_up_exps.weight"):
        tname = f"blk.{il}.{suf}"
        if tname in art.tensors:
            return int(art.tensors[tname]["dims"][1])
    gu = f"blk.{il}.ffn_gate_up_exps.weight"
    if gu in art.tensors:
        return int(art.tensors[gu]["dims"][1]) // 2
    raise KeyError(f"blk.{il}: no FFN width tensor found")


def _switch_hook_module(layer):
    mlp = layer.mlp
    return mlp.switch_hook if hasattr(mlp, "switch_hook") else mlp.down_proj


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
            w_il = _layer_ffn_width(art, il)
            widths_by_layer[il] = w_il
            dead = np.flatnonzero(theta[il].detach().cpu().numpy() <= 0)
            off[il] = dead.tolist()
            bits += w_il
            if len(dead):
                for suf in ("ffn_up.weight", "ffn_gate.weight", "ffn_up_exps.weight", "ffn_gate_exps.weight"):
                    tname = f"blk.{il}.{suf}"
                    if tname in art.tensors:
                        row_zeros[tname] = dead
                gu_name = f"blk.{il}.ffn_gate_up_exps.weight"
                if gu_name in art.tensors:
                    half = int(art.tensors[gu_name]["dims"][1]) // 2
                    row_zeros[gu_name] = np.concatenate([dead, half + dead])
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
        layers = []
        for il in range(n_layers):
            dense_ok = (
                f"blk.{il}.ffn_up.weight" in art.tensors
                and f"blk.{il}.ffn_gate.weight" in art.tensors
                and f"blk.{il}.ffn_down.weight" in art.tensors
            )
            moe_ok = (
                f"blk.{il}.ffn_down_exps.weight" in art.tensors
                and (
                    f"blk.{il}.ffn_gate_up_exps.weight" in art.tensors
                    or (f"blk.{il}.ffn_up_exps.weight" in art.tensors and f"blk.{il}.ffn_gate_exps.weight" in art.tensors)
                )
            )
            if dense_ok or moe_ok:
                layers.append(il)
    else:
        layers = [il for il in range(n_layers) if art.tensors.get(f"blk.{il}.ffn_down.weight", {}).get("type") in lattice.TOP]
    if last_blocks is not None and last_blocks > 0:
        layers = layers[-int(last_blocks):]
    return layers


def _auto_checkpointing(model, cfg: dict, device: str) -> bool:
    if cfg.get("checkpointing") is not None:
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
    sp = "auto" if cfg.get("stream_prefix") is None else cfg["stream_prefix"]
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


def learn_gates(art_path: str, config_dir: str | Path | None, feedback: list[dict], replay: list[np.ndarray], pair: tuple[int, int] | None = None,
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
    amp = device.startswith("cuda") and (True if cfg.get("amp") is None else bool(cfg["amp"]))
    switch_mode = "up_row_zero" if cfg.get("switch_mode") is None else cfg["switch_mode"]
    art = lattice.Artifact(art_path)
    frozen_bf16 = amp if cfg.get("frozen_bf16") is None else bool(cfg["frozen_bf16"])
    stream_pref = _should_stream_prefix(art, cfg, device, frozen_bf16)
    model, _ = build_student(art, config_dir, [], cfg.get("ggml_lib"), "cpu" if stream_pref else device, frozen_bf16=frozen_bf16)
    layers = _switchable_layers(art, len(model.model.layers), switch_mode, cfg.get("gate_last_blocks"))
    if not layers:
        art.close()
        raise RuntimeError(f"no switchable FFN layers found in {art_path} under switch_mode={switch_mode!r}")
    all_seqs = [r["tokens"] for r in feedback] + list(replay)
    prefix_cache = PrefixCache.build(model, all_seqs, min(layers), device, amp, cfg, log=log, stream_prefix=stream_pref) if layers else None
    if stream_pref and prefix_cache is None:
        model.to(device)
    if _auto_checkpointing(model, cfg, device) and amp:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.train()
    init_val = 1.0 if cfg.get("gate_init") is None else float(cfg["gate_init"])
    theta = {il: torch.nn.Parameter(torch.full((_layer_ffn_width(art, il),), init_val, device=device)) for il in layers}

    def hook(il):
        def f(mod, args):
            s = torch.sigmoid(theta[il])
            g = (theta[il] > 0).float() + s - s.detach()
            return (args[0] * g.to(args[0].dtype),)
        return f

    hooks = [_switch_hook_module(model.model.layers[il]).register_forward_pre_hook(hook(il)) for il in layers]
    params = list(theta.values())
    gate_lr = 0.03 if cfg.get("gate_lr") is None else float(cfg["gate_lr"])
    steps_rec = 200 if cfg.get("steps") is None else int(cfg["steps"])
    hist = _train(model, params, feedback, replay, pair, cutoff, cfg, log, device, amp, strata, gate_lr, prefix_cache=prefix_cache)
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
    rec = {"method": "gates", "switch_mode": switch_mode, "steps": steps_rec, "gate_lr": gate_lr,
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
    __slots__ = ("length", "h", "ple", "skv", "blocks")

    def __init__(self, length: int, h, ple=None, skv=None, blocks=None):
        self.length = length
        self.h = h
        self.ple = ple
        self.skv = skv
        self.blocks = blocks


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
        if str(getattr(cfg, "model_type", "")) in ("inkling2", "inkling2_loop"):
            return "inkling2_loop"
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
        elif kind == "inkling2_loop":
            for attr in ("loop_layers", "tok_norm"):
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

        kind = cls._arch_kind(model)
        if kind == "inkling2_loop":
            loop_end = int(getattr(mcfg, "loop_end", 0) or 0)
            if start_layer < loop_end:
                log(f"prefix_cache: skipped (start_layer={start_layer} < loop_end={loop_end})")
                if stream_prefix:
                    model.to(device)
                return None

        uniq: dict[bytes, np.ndarray] = {}
        for s in seqs:
            arr = np.asarray(s, dtype=np.int64)
            uniq.setdefault(arr.tobytes(), arr)

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
        elif kind == "inkling2_loop":
            bs = max(1, int(getattr(mcfg, "attn_res_block_size", 1) or 1))
            n_blocks = (start_layer + bs - 1) // bs
            per_tok_elems += n_blocks * int(mcfg.hidden_size)
        est_mb = (total_tok * per_tok_elems * elem_b) / (1024 * 1024)
        max_mb = 4096.0 if cfg.get("prefix_cache_max_mb") is None else float(cfg["prefix_cache_max_mb"])
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
        free_prefix = True if cfg.get("free_prefix") is None else bool(cfg["free_prefix"])
        freed: list[str] = []

        if stream_prefix:
            keys = list(uniq.keys())
            if getattr(inner, "rotary_emb", None) is not None:
                inner.rotary_emb.to(device)
            inner.embed_tokens.to(device)
            if kind == "gemma4":
                for attr in ("embed_tokens_per_layer", "per_layer_model_projection", "per_layer_projection_norm"):
                    mod = getattr(inner, attr, None)
                    if mod is not None and hasattr(mod, "to"):
                        mod.to(device)
            elif kind == "inkling2_loop" and getattr(inner, "tok_norm", None) is not None:
                inner.tok_norm.to(device)
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
                        if kind == "inkling2_loop" and getattr(inner, "tok_norm", None) is not None:
                            emb = inner.tok_norm(emb)
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
            elif kind == "inkling2_loop" and getattr(inner, "tok_norm", None) is not None:
                inner.tok_norm = None
                freed.append("tok_norm")
            if model.lm_head.weight.data_ptr() != inner.embed_tokens.weight.data_ptr():
                inner.embed_tokens = None
                freed.append("embed_tokens")
            else:
                inner.embed_tokens.to("cpu")
            _clear_dev()

            if kind == "inkling2_loop":
                blocks_cpu: dict[bytes, list[torch.Tensor]] = {k: [] for k in keys}
                loop_start = int(getattr(mcfg, "loop_start", 0) or 0)
                loop_end = int(getattr(mcfg, "loop_end", 0) or 0)
                loop_count = int(getattr(mcfg, "loop_count", 1) or 1)
                with torch.no_grad():
                    def _step_inkling2(layer_mod, key: bytes, ip: int, loop_mod=None, bank_h=None, bank_blocks=None):
                        h_i = hs_cpu[key].unsqueeze(0).to(device, non_blocking=pin)
                        L = h_i.shape[1]
                        attn = torch.ones((1, L), dtype=torch.long, device=device)
                        pos, attn_bool, tau = inner._pos_and_tau(attn)
                        b_dev = [b.unsqueeze(0).to(device, non_blocking=pin) for b in blocks_cpu[key]]
                        with torch.autocast(dev_type, dtype=torch.bfloat16, enabled=use_amp):
                            msgs = None
                            if loop_mod is not None:
                                bh_dev = bank_h.unsqueeze(0).to(device, non_blocking=pin)
                                bb_dev = [b.unsqueeze(0).to(device, non_blocking=pin) for b in bank_blocks]
                                msgs = _inkling2_mix(bb_dev, bh_dev, loop_mod.msg_query, inner.n_embd)
                            h_out, new_blk = layer_mod(ip, h_i, b_dev, pos, tau, attn_bool, messages=msgs, loop=loop_mod)
                        hs_cpu[key] = h_out[0].detach().to("cpu", store_dtype).contiguous()
                        if new_blk is not None:
                            blocks_cpu[key].append(new_blk[0].detach().to("cpu", store_dtype).contiguous())

                    pre_end = loop_start if loop_end > loop_start else start_layer
                    for ip in range(pre_end):
                        layer_mod = inner.layers[ip].to(device)
                        for key in keys:
                            _step_inkling2(layer_mod, key, ip)
                        inner.layers[ip] = nn.Identity()
                        del layer_mod
                        _clear_dev()

                    if loop_end > loop_start:
                        prelude_h = {k: hs_cpu[k] for k in keys}
                        prelude_blocks = {k: list(blocks_cpu[k]) for k in keys}
                        for pass_idx in range(loop_count):
                            bank_h_map = None
                            bank_b_map = None
                            if pass_idx > 0:
                                bank_h_map = {k: hs_cpu[k] for k in keys}
                                bank_b_map = {k: blocks_cpu[k][len(prelude_blocks[k]) + 1 :] for k in keys}
                                hs_cpu = {k: prelude_h[k] for k in keys}
                                blocks_cpu = {k: list(prelude_blocks[k]) for k in keys}
                            for ip in range(loop_start, loop_end):
                                layer_mod = inner.layers[ip].to(device)
                                loop_mod = inner.loop_layers[ip - loop_start].to(device) if (pass_idx > 0 and inner.loop_layers is not None) else None
                                for key in keys:
                                    _step_inkling2(
                                        layer_mod, key, ip, loop_mod=loop_mod,
                                        bank_h=(bank_h_map[key] if bank_h_map is not None else None),
                                        bank_blocks=(bank_b_map[key] if bank_b_map is not None else None),
                                    )
                                if pass_idx == loop_count - 1:
                                    inner.layers[ip] = nn.Identity()
                                else:
                                    layer_mod.to("cpu")
                                    if loop_mod is not None:
                                        loop_mod.to("cpu")
                                del layer_mod
                                if loop_mod is not None:
                                    del loop_mod
                                _clear_dev()
                        if getattr(inner, "loop_layers", None) is not None:
                            inner.loop_layers = None
                            freed.append("loop_layers")
                        for ip in range(loop_end, start_layer):
                            layer_mod = inner.layers[ip].to(device)
                            for key in keys:
                                _step_inkling2(layer_mod, key, ip)
                            inner.layers[ip] = nn.Identity()
                            del layer_mod
                            _clear_dev()
                freed.append(f"layers[0:{start_layer}]")
                for key in keys:
                    hc = _to_host(hs_cpu.pop(key))
                    bc = _to_host(torch.stack(blocks_cpu.pop(key), dim=0))
                    actual_bytes += hc.numel() * hc.element_size() + bc.numel() * bc.element_size()
                    entries[key] = _PrefixEntry(len(uniq[key]), hc, blocks=bc)
                for i in range(start_layer, n_layers):
                    inner.layers[i].to(device)
                if getattr(inner, "output_res_query", None) is not None:
                    inner.output_res_query.data = inner.output_res_query.data.to(device)
                inner.norm.to(device)
                model.lm_head.to(device)
                _clear_dev()
            else:
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
                        ple_tail = None
                        skv_cpu = None
                        blocks_cpu_entry = None
                        if kind == "inkling2_loop":
                            h, blocks_list = inner.forward_prefix(ids, attn, start_layer)
                            blocks_cpu_entry = _to_host(torch.stack([b[0] for b in blocks_list], dim=0))
                            actual_bytes += blocks_cpu_entry.numel() * blocks_cpu_entry.element_size()
                        else:
                            inputs_embeds = inner.embed_tokens(ids)
                            h = inputs_embeds
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
                    entries[key] = _PrefixEntry(L, h_cpu, ple=ple_tail, skv=skv_cpu, blocks=blocks_cpu_entry)

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
            if self.kind == "inkling2_loop":
                n_blocks = ents[0].blocks.shape[0]
                b_buf = torch.zeros((B, n_blocks, n, d), dtype=ents[0].blocks.dtype)
                for k, e in enumerate(ents):
                    b_buf[k, :, :e.length, :] = e.blocks
                b_dev = b_buf.to(device, non_blocking=nb)
                blocks = [b_dev[:, j, :, :] for j in range(n_blocks)]
                h = inner.forward_tail(
                    h, blocks, attn, self.start_layer,
                    grad_ckpt=getattr(model, "is_gradient_checkpointing", False),
                    gc_kwargs=getattr(model, "_gc_kwargs", None),
                )
            elif self.kind == "gemma4":
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
    config_dir: str | Path | None,
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
    amp = device.startswith("cuda") and (True if cfg.get("amp") is None else bool(cfg["amp"]))
    switch_mode = "up_row_zero" if cfg.get("switch_mode") is None else cfg["switch_mode"]
    pad_mult = 32 if cfg.get("pad_multiple") is None else int(cfg["pad_multiple"])

    lessons = [_norm_seq_lesson(r) for r in feedback]
    rep_trajs = [_norm_replay_traj(x) for x in replay]
    if not lessons and not rep_trajs:
        raise ValueError("learn_gates_seq requires at least one feedback lesson or replay trajectory")

    art = lattice.Artifact(art_path)
    frozen_bf16 = amp if cfg.get("frozen_bf16") is None else bool(cfg["frozen_bf16"])
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

    init_val = 1.0 if cfg.get("gate_init") is None else float(cfg["gate_init"])
    theta = {il: torch.nn.Parameter(torch.full((_layer_ffn_width(art, il),), init_val, device=device)) for il in layers}

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

    hooks = [_switch_hook_module(model.model.layers[il]).register_forward_pre_hook(hook(il)) for il in layers]
    flat = list(theta.values())
    gate_lr = 0.05 if cfg.get("gate_lr") is None else float(cfg["gate_lr"])
    adam_eps = 1e-4 if cfg.get("adam_eps") is None else float(cfg["adam_eps"])
    opt = torch.optim.AdamW(flat, lr=gate_lr, weight_decay=0.0, eps=adam_eps)
    rng = random.Random(20261031 if cfg.get("seed") is None else cfg["seed"])

    steps = 80 if cfg.get("steps") is None else int(cfg["steps"])
    def_bs = _default_seq_batch(model, device)
    fb_raw = cfg.get("feedback_batch")
    nr_raw = cfg.get("replay_batch")
    nf = min(def_bs, len(lessons)) if fb_raw is None else int(fb_raw)
    nr = (min(def_bs, len(rep_trajs)) if nr_raw is None else int(nr_raw)) if rep_trajs else 0
    if nf < 0 or nr < 0:
        raise ValueError(f"feedback_batch and replay_batch must be >= 0, got {nf!r}, {nr!r}")
    nll_w = 1.0 if cfg.get("nll_weight") is None else float(cfg["nll_weight"])
    margin_w = 1.0 if cfg.get("margin_weight") is None else float(cfg["margin_weight"])
    margin = 0.5 if cfg.get("margin") is None else float(cfg["margin"])
    hold_w = 2.0 if cfg.get("replay_weight") is None else float(cfg["replay_weight"])
    kl_temp = float(1.0 if cfg.get("kl_temp") is None else cfg["kl_temp"])
    if not math.isfinite(kl_temp) or kl_temp <= 0:
        raise ValueError(f"kl_temp must be finite > 0, got {kl_temp!r}")
    use_agem = True if cfg.get("project") is None else bool(cfg["project"])

    log(f"learn_gates_seq: {sum(w.numel() for w in flat):,} switches across {len(layers)} FFN layers, {len(lessons)} lessons, {len(rep_trajs)} replay")
    hist: list[tuple[float, float, float]] = []
    conflicts = 0

    for step in range(steps):
        if lessons and nf > 0 and (nll_w > 0 or margin_w > 0):
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

        if not (torch.isfinite(loss_new) and torch.isfinite(loss_hold)):
            for h in hooks:
                h.remove()
            art.close()
            raise FloatingPointError(
                f"non-finite loss in learn_gates_seq step {step}: loss_new={loss_new.item()}, loss_hold={loss_hold.item()}"
            )

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
        "gate_lr": gate_lr,
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


def _side(task: str, prompt_id: str | None = None, seed: int | None = None) -> int:
    """Deterministic 50/50 split by `prompt_id` (when provided) or `task` id so related items stay on one side."""
    key = task if prompt_id is None else str(prompt_id)
    if seed is not None:
        key = f"{seed}:{key}"
    return int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) % 2


def row_side(r: dict, c: dict) -> int:
    """Resolve select side (0 = replay pool, 1 = judge), grouping multi-variant rows by `prompt_id` when configured."""
    return _side(
        r["task"],
        r["prompt_id"] if c.get("side_by_prompt") and r["task"].startswith("beemo") else None,
        seed=c.get("side_seed"),
    )


def real_scores(model: str, c: dict, scorer) -> tuple[np.ndarray, np.ndarray]:
    """The real file's scores on the pool and the feedback, once per file."""
    pair = tuple(c["pair"])
    quick = c.get("quick_idx") and sha256(model) != sha256(c["model"])
    cache_dir = c["out_dir"] if c.get("cache_dir") is None else c["cache_dir"]
    cache = Path(cache_dir) / f"scores-{sha256(model)[:16]}{'-quick' if quick else ''}.npz"
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
    rng = random.Random(20261031 if c.get("seed") is None else c["seed"])
    panel_humans = 4000 if c.get("panel_humans") is None else int(c["panel_humans"])
    idx = sorted(rng.sample(range(ncal) if humans is None else humans, panel_humans))
    idx += [ncal + i for i, r in enumerate(rows) if row_side(r, c) == 1]
    pool = read_trj(c["pool_trj"])
    assert len(pool) == ncal + len(rows)
    write_trj([pool[i] for i in idx], c["panel_trj"], c["pair"][0])
    np.save(c["panel_idx"], np.array(idx))


def real_panel(model: str, c: dict, scorer) -> np.ndarray:
    """Score the built GGUF directly on the fixed panel (cached by model and panel hash; no base-score imputation)."""
    cache_dir = c["out_dir"] if c.get("cache_dir") is None else c["cache_dir"]
    cache = Path(cache_dir) / f"panel-{sha256(model)[:16]}-{sha256(c['panel_trj'])[:8]}.npy"
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
    gate_margin = 0.02 if c.get("gate_margin") is None else float(c["gate_margin"])
    out = {"model": model, "bytes": Path(model).stat().st_size, "cutoff": cut, "n_gate": len(gate), "delta": b["delta"], "L": b["L"], "U": b["U"],
           "gate_pass": bool(b["L"] > -gate_margin), "caught_wo": rate(True, False), "caught_beemo": rate(True, True),
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


def _stratified_select(items: list, limit: int, key_fn, id_fn=None) -> list:
    """Deterministically select up to `limit` items stratified round-robin across buckets."""
    if limit <= 0 or not items:
        return []
    if len(items) <= limit:
        return list(items)
    buckets: dict[str, list] = {}
    for idx, it in enumerate(items):
        k = str(key_fn(it) or "")
        buckets.setdefault(k, []).append((idx, it))
    keys = sorted(buckets)
    if id_fn is not None:
        for k in keys:
            buckets[k].sort(key=lambda pair: str(id_fn(pair[1])))
    out = []
    pos = 0
    while len(out) < limit:
        progressed = False
        for k in keys:
            b = buckets[k]
            if pos < len(b):
                out.append(b[pos][1])
                progressed = True
                if len(out) >= limit:
                    break
        if not progressed:
            break
        pos += 1
    return out


def _tokenize_seq_lessons(tc, model_gguf: str, raw_rows: list[dict], threads: int = 8, log=None) -> list[dict]:
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
    good_fb = 0
    bad_fb = 0
    for k, i in enumerate(need_str):
        pr, pr_good, good_raw, pr_bad, bad_raw = toks[5 * k : 5 * k + 5]
        if len(pr_good) > len(pr) and pr_good[: len(pr)] == pr:
            good = pr_good[len(pr):]
        else:
            good = good_raw
            good_fb += 1
        if bad_raw:
            if len(pr_bad) > len(pr) and pr_bad[: len(pr)] == pr:
                bad = pr_bad[len(pr):]
            else:
                bad = bad_raw
                bad_fb += 1
        else:
            bad = []
        out[i]["prompt_tokens"] = pr
        out[i]["corrected_tokens"] = good
        if bad:
            out[i]["mistaken_tokens"] = bad
    if good_fb or bad_fb:
        msg = (
            f"_tokenize_seq_lessons: BPE-boundary prefix mismatch fell back to unmerged tokenization "
            f"(corrected={good_fb}/{len(need_str)}, mistaken={bad_fb}/{len(need_str)})"
        )
        if callable(log):
            log(f"WARNING: {msg}")
        else:
            _LOG.warning(msg)
    return out


def _load_replay_trajs(tc, model_gguf: str, spec, threads: int = 8, log=None) -> list[tuple[np.ndarray, np.ndarray]]:
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
    dropped_bpe = 0
    dropped_empty = 0
    for k in range(0, len(toks), 2):
        pre, full = toks[k], toks[k + 1]
        if len(full) > len(pre) and full[: len(pre)] == pre:
            out.append((np.asarray(full, dtype=np.int32), np.arange(len(pre), len(full), dtype=np.uint32)))
        elif len(full) <= len(pre):
            dropped_empty += 1
        else:
            dropped_bpe += 1
    dropped = dropped_bpe + dropped_empty
    if dropped:
        msg = (
            f"_load_replay_trajs: dropped {dropped}/{len(rows)} replay trajectories "
            f"(bpe_mismatch={dropped_bpe}, empty_completion={dropped_empty})"
        )
        if callable(log):
            log(f"WARNING: {msg}")
        else:
            _LOG.warning(msg)
    if rows and not out:
        raise ValueError(
            f"all {len(rows)} replay trajectories were dropped "
            f"(bpe_mismatch={dropped_bpe}, empty_completion={dropped_empty})"
        )
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
    config_dir = str(c["config_dir"]) if c.get("config_dir") else None
    ggml_lib = str(c.get("ggml_lib") or cfg["toolchain"]["llama_bin"])
    switch_mode = "up_row_zero" if c.get("switch_mode") is None else c["switch_mode"]
    base_cfg = {**c, "ggml_lib": ggml_lib, "switch_mode": switch_mode}

    all_lessons = _tokenize_seq_lessons(tc, base_model, _load_json_or_jsonl(c["lessons"]), log=log)
    train_lessons = [r for r in all_lessons if fold is None or r.get("fold") != fold]
    held_lessons = [r for r in all_lessons if fold is not None and r.get("fold") == fold]
    max_l = 25 if c.get("max_lessons") is None else int(c["max_lessons"])
    if len(train_lessons) > max_l:
        n_before = len(train_lessons)
        if any(any(k in r for k in ("family", "category", "share")) for r in train_lessons):
            train_lessons = _stratified_select(
                train_lessons,
                max_l,
                key_fn=lambda r: r.get("family") or r.get("category") or r.get("share") or "",
                id_fn=lambda r: r.get("id") or r.get("task") or "",
            )
        else:
            train_lessons = train_lessons[:max_l]
        log(f"run_generative: truncated train_lessons {n_before} -> {len(train_lessons)} (max_lessons={max_l})")
    replay_trajs = _load_replay_trajs(tc, base_model, c.get("replay"), log=log)

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
    if (True if c.get("controls") is None else bool(c["controls"])):
        # 1. Empty control (0 lessons, replay/init only)
        emp_gguf = str(out_dir / f"control-empty-{tag}.gguf")
        emp_cfg = {**base_cfg, "nll_weight": 0.0, "margin_weight": 0.0}
        emp_cfg.pop("mask_in", None); emp_cfg.pop("mask_out", None)
        control_recs["empty"] = learn_gates_seq(base_model, config_dir, [], replay_trajs or [(np.array([1, 2, 3], dtype=np.int32), np.array([1, 2], dtype=np.uint32))], emp_gguf, cfg=emp_cfg, log=log)
        arms["empty"] = emp_gguf

        # 2. Flipped control (swap positive <-> negative on lessons that carry a negative trajectory)
        normed_lessons = [_norm_seq_lesson(r) for r in train_lessons]
        flipped_lessons = []
        for nl in normed_lessons:
            if nl["neg_tokens"] is not None:
                flipped_lessons.append({
                    "pos_tokens": nl["neg_tokens"],
                    "pos_targets": nl["neg_targets"],
                    "neg_tokens": nl["pos_tokens"],
                    "neg_targets": nl["pos_targets"],
                })
        if flipped_lessons:
            flp_gguf = str(out_dir / f"control-flipped-{tag}.gguf")
            flp_cfg = dict(base_cfg); flp_cfg.pop("mask_in", None); flp_cfg.pop("mask_out", None)
            control_recs["flipped"] = learn_gates_seq(base_model, config_dir, flipped_lessons, replay_trajs, flp_gguf, cfg=flp_cfg, log=log)
            arms["flipped"] = flp_gguf

        # 3. Shuffled control (cyclic shift of positive/negative completions across prompts)
        if len(normed_lessons) >= 2:
            shuf_lessons = []
            for i, nl_r in enumerate(normed_lessons):
                nl_o = normed_lessons[(i + 1) % len(normed_lessons)]
                r_prompt = nl_r["pos_tokens"][: int(nl_r["pos_targets"].min())]
                o_pos_start = int(nl_o["pos_targets"].min())
                pos_tok = np.concatenate([r_prompt, nl_o["pos_tokens"][o_pos_start:]])
                pos_tgt = len(r_prompt) + (nl_o["pos_targets"] - o_pos_start)
                neg_tok = neg_tgt = None
                if nl_o["neg_tokens"] is not None:
                    o_neg_start = int(nl_o["neg_targets"].min())
                    neg_tok = np.concatenate([r_prompt, nl_o["neg_tokens"][o_neg_start:]])
                    neg_tgt = len(r_prompt) + (nl_o["neg_targets"] - o_neg_start)
                shuf_lessons.append({
                    "pos_tokens": pos_tok,
                    "pos_targets": pos_tgt,
                    "neg_tokens": neg_tok,
                    "neg_targets": neg_tgt,
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
        pos_nlls = [float(r_kl["trajs"][ip]["nll_cand"]) for ip, _ in pair_indices]
        margins = [float((-r_kl["trajs"][ip]["nll_cand"]) - (-r_kl["trajs"][ineg]["nll_cand"])) for ip, ineg in pair_indices if ineg is not None]
        if not all(math.isfinite(x) for x in pos_nlls) or not all(math.isfinite(m) for m in margins):
            raise FloatingPointError(f"non-finite traj-kl metrics for arm {arm_name!r}")
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
    l_nll = cpp_metrics["learned"]["pos_nll_mean"]
    b_nll = cpp_metrics["base"]["pos_nll_mean"]
    accepted = bool(l_nll is not None and b_nll is not None and l_nll < b_nll)
    if cfg.get("env"):
        env_kw = (cfg["env"].get("kwargs") or {})
        env = load_environment(cfg["env"]["module"], **env_kw)
        by_id = {t.id: t for t in env.tasks()}
        decoding = cfg.get("decoding") or {"temperature": 0.0}
        serving = cfg.get("serving") or {}
        slots = int(c["judge_slots"] if c.get("judge_slots") is not None else (1 if serving.get("slots_per_gpu") is None else serving["slots_per_gpu"]))
        ctx_per_slot = int(serving["ctx_per_slot"] if serving.get("ctx_per_slot") is not None else env.deployment["context"])
        ev = Evaluator(
            tc, env, decoding, slots, ctx_per_slot, env.deployment["kv_type"], out_dir / "episodes",
            cache_prompt=(False if c.get("judge_cache_prompt") is None else bool(c["judge_cache_prompt"])),
            special=serving.get("special"),
            flash_attn=("on" if serving.get("flash_attn") is None else serving["flash_attn"]),
            use_chat_completions=serving.get("use_chat_completions"),
        )
        side_seed = c.get("side_seed")
        if c.get("heldout_ids") is not None:
            held_ids = list(c["heldout_ids"])
            missing_ids = [i for i in held_ids if i not in by_id]
            if missing_ids:
                raise ValueError(f"unknown heldout_ids: {missing_ids[:10]}")
        else:
            held_ids = [t.id for t in env.tasks() if t.split == "train" and _side(t.id, seed=side_seed) == 1]
        held_tasks = [by_id[i] for i in held_ids]
        sel_tasks = [t for t in env.tasks() if t.split == "select" and _side(t.id, seed=side_seed) == 1]
        if c.get("select_limit") is not None:
            sel_tasks = _stratified_select(
                sel_tasks,
                int(c["select_limit"]),
                key_fn=lambda t: (t.meta or {}).get("category") or (t.meta or {}).get("area") or t.family,
                id_fn=lambda t: t.id,
            )
        ref_model = str(c.get("gate_ref_model") or base_model)
        ref_sel = ok_rows(ev.run(ref_model, [(t, util.stable_seed("learn-sel", t.id)) for t in sel_tasks], "learn-ref-sel", "src")) if sel_tasks else []

        gate_margin = 0.03 if c.get("gate_margin") is None else float(c["gate_margin"])
        alpha = 0.05 if c.get("alpha") is None else float(c["alpha"])
        for arm_name, arm_path in arms.items():
            h_rows = ok_rows(ev.run(arm_path, [(t, util.stable_seed("learn-held", t.id)) for t in held_tasks], f"learn-{arm_name}-held", "cand")) if held_tasks else []
            s_rows = ok_rows(ev.run(arm_path, [(t, util.stable_seed("learn-sel", t.id)) for t in sel_tasks], f"learn-{arm_name}-sel", "cand")) if sel_tasks else []
            if not sel_tasks:
                bnd = {"delta": 0.0, "L": 0.0, "U": 0.0}
                g_pass = True
            elif not s_rows or not ref_sel:
                bnd = {"delta": None, "L": None, "U": None}
                g_pass = False
            else:
                bnd = stats.delta_and_bounds(s_rows + ref_sel, alpha)
                g_pass = bool(math.isfinite(bnd["L"]) and bnd["L"] > -gate_margin)
            env_metrics[arm_name] = {
                "heldout_n": len(h_rows),
                "heldout_passed": int(sum(bool(r["success"]) for r in h_rows)),
                "heldout_rate": float(sum(bool(r["success"]) for r in h_rows) / len(h_rows)) if h_rows else None,
                "select_n": len(s_rows),
                "select_rate": float(sum(bool(r["success"]) for r in s_rows) / len(s_rows)) if s_rows else None,
                "select_delta": bnd["delta"],
                "select_L": bnd["L"],
                "gate_pass": g_pass,
            }
        if held_tasks:
            l_rate = env_metrics["learned"]["heldout_rate"]
            ctrl_keys = [k for k in ("base", "empty", "flipped", "shuffled") if k in env_metrics]
            ctrl_rates = [env_metrics[k]["heldout_rate"] for k in ctrl_keys]
            accepted = bool(
                l_rate is not None
                and math.isfinite(l_rate)
                and env_metrics["learned"]["gate_pass"]
                and "base" in env_metrics
                and all(cr is not None and math.isfinite(cr) and l_rate > cr for cr in ctrl_rates)
            )
        elif sel_tasks:
            accepted = bool(accepted and env_metrics["learned"]["gate_pass"])

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
    if c.get("mode") == "generative" or c.get("method") == "gates_seq" or c.get("task_kind") == "generative":
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
    rng = random.Random(20261031 if c.get("seed") is None else c["seed"])
    side0 = [ncal + i for i, r in enumerate(rows) if row_side(r, c) == 0]
    rh = 1000 if c.get("replay_humans") is None else int(c["replay_humans"])
    rs = 1000 if c.get("replay_select") is None else int(c["replay_select"])
    picks = rng.sample(range(ncal), rh) + rng.sample(side0, rs)
    strata = None
    if c.get("stratify"):
        kind = lambda i: ("beemo" if rows[i - ncal]["task"].startswith("beemo") else "wo", "human" if rows[i - ncal]["verdict"] != "ai" else "ai")
        beemo = [i for i in side0 if kind(i)[0] == "beemo"]
        wo = [i for i in side0 if kind(i)[0] == "wo"]
        picks = rng.sample(range(ncal), rh) + beemo + rng.sample(wo, rs)
        strata = ["human-cal"] * rh + [f"{kind(i)[1]}-{kind(i)[0]}" for i in picks[rh:]]
    slots_pg = 2 if c.get("slots_per_gpu") is None else int(c["slots_per_gpu"])
    scorer = BatchScorer(Toolchain.from_config(cfg["toolchain"]), slots_pg)
    if c.get("tails"):
        # the documents that decide the operating point: the humans who set the 1% cutoff, and the AI just under it
        p, _ = real_scores(c["model"], c, scorer)
        tail_n = 500 if c.get("tail_n") is None else int(c["tail_n"])
        near_n = 300 if c.get("near_n") is None else int(c["near_n"])
        tail = np.argsort(p[:ncal])[:tail_n].tolist()
        ai = [i for i in side0 if rows[i - ncal]["verdict"] == "ai"]
        near = sorted(ai, key=lambda i: -p[i])[:near_n]
        picks += tail + near
        if strata is None:
            strata = ["all"] * (len(picks) - len(tail) - len(near))
        strata += ["human-tail"] * len(tail) + ["ai-near"] * len(near)
    feedback = [dict(fb[i], tokens=fdocs[fb[i]["index"]]) for i in train]
    tag = "all" if fold is None else f"fold{fold}"
    out = str(out_dir / f"model-{tag}.gguf")
    trainer = learn_gates if c.get("method") == "gates" else learn
    config_dir = str(c["config_dir"]) if c.get("config_dir") else None
    rec = trainer(c["model"], config_dir, feedback, [pool[i] for i in picks], pair, c["cutoff"], out, c, log, strata=strata)
    base = judge(c["model"], c, scorer, rows, fb, hold)
    cand = judge(out, c, scorer, rows, fb, hold)
    rise = cand["human_flagged_beemo"] - base["human_flagged_beemo"]
    better = hold and cand["held"]["pairs"] and cand["held"]["mean_gap"] > base["held"]["mean_gap"]
    drop_wo = base["caught_wo"] - cand["caught_wo"]
    drop_beemo = base["caught_beemo"] - cand["caught_beemo"]
    finite_drop = math.isfinite(drop_wo) and math.isfinite(drop_beemo)
    drop = max(drop_wo, drop_beemo) if finite_drop else float("nan")    # enforce per-subset recall drop cap alongside the pooled gate
    max_human_rise = 0.01 if c.get("max_human_rise") is None else float(c["max_human_rise"])
    max_ai_drop = 0.02 if c.get("max_ai_drop") is None else float(c["max_ai_drop"])
    accepted = bool(cand["gate_pass"] and math.isfinite(rise) and finite_drop and rise <= max_human_rise and drop <= max_ai_drop and better)
    res = {"accepted": accepted, "fold": fold, "lineage": rec, "base": base, "candidate": cand, "split": "select_side1"}
    (out_dir / f"result-{tag}.json").write_text(json.dumps(res, indent=1))
    log(f"learn {tag}: accepted={accepted} gate L {cand['L']:.4f} held pairs right {base['held']['pairs_right']} -> {cand['held']['pairs_right']}, "
        f"flagged AI {base['held']['ai_flagged']} -> {cand['held']['ai_flagged']}, Beemo humans {base['human_flagged_beemo']:.3f} -> {cand['human_flagged_beemo']:.3f}")
    return res
