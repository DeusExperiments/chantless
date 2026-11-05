"""Gemma 4 E2B text model in plain PyTorch, for training the detector. Same maths as the reference implementation:
pre and post norms around attention and MLP, per-layer embeddings, shared keys and values from layer 15 on, dual RoPE
(windowed layers: 256-wide heads, base 1e4; global layers: 512-wide heads, proportional RoPE on a quarter of the
dimensions, base 1e6), attention scale 1, per-layer scalars, tied output with a soft cap of 30.
Documents are packed into one stream with no padding; one FlexAttention mask carries document boundaries, causality
and the 512-token window. Only the "human" and "ai" output rows are computed."""
import json, math
from pathlib import Path
import torch, torch.nn as nn, torch.nn.functional as F
from safetensors import safe_open
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

_flex = torch.compile(flex_attention, dynamic=False)
H100_GLOBAL = {'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_M1': 32, 'BLOCK_N1': 32, 'BLOCK_M2': 32, 'BLOCK_N2': 32, 'num_stages': 1}
SMALL_SMEM = {'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_M1': 16, 'BLOCK_N1': 32, 'BLOCK_M2': 32, 'BLOCK_N2': 16, 'num_stages': 1}


class Norm(nn.Module):
    def __init__(s, n, eps, scale=True):
        super().__init__(); s.eps = eps; s.weight = nn.Parameter(torch.ones(n)) if scale else None

    def forward(s, x):
        y = x.float(); y = y * torch.pow(y.pow(2).mean(-1, keepdim=True) + s.eps, -0.5)
        if s.weight is not None: y = y * s.weight.float()
        return y.type_as(x)


def rotate_half(x):
    a, b = x.chunk(2, dim=-1); return torch.cat((-b, a), dim=-1)


_DOC = None
_WIN = 512


def _full_mod(b, h, qi, ki):
    return (_DOC[qi] == _DOC[ki]) & (ki <= qi)


def _slide_mod(b, h, qi, ki):
    return (_DOC[qi] == _DOC[ki]) & (ki <= qi) & (qi - ki < _WIN)


class Layer(nn.Module):
    def __init__(s, c, i):
        super().__init__()
        d, eps = c['hidden_size'], c['rms_norm_eps']; s.kind = c['layer_types'][i]; s.sliding = s.kind == 'sliding_attention'
        s.hd = c['head_dim'] if s.sliding else c['global_head_dim']; s.nh = c['num_attention_heads']
        first_shared = c['num_hidden_layers'] - c['num_kv_shared_layers']; s.shared = i >= first_shared
        prev = c['layer_types'][:first_shared]; s.store = (not s.shared) and i == len(prev) - 1 - prev[::-1].index(s.kind)
        s.q_proj = nn.Linear(d, s.nh * s.hd, bias=False); s.q_norm = Norm(s.hd, eps)
        if not s.shared:
            s.k_proj = nn.Linear(d, s.hd, bias=False); s.v_proj = nn.Linear(d, s.hd, bias=False); s.k_norm = Norm(s.hd, eps); s.v_norm = Norm(s.hd, eps, scale=False)
        s.o_proj = nn.Linear(s.nh * s.hd, d, bias=False)
        f = c['intermediate_size'] * (2 if c['use_double_wide_mlp'] and s.shared and first_shared > 0 else 1)
        s.gate_proj = nn.Linear(d, f, bias=False); s.up_proj = nn.Linear(d, f, bias=False); s.down_proj = nn.Linear(f, d, bias=False)
        s.input_layernorm, s.post_attention_layernorm, s.pre_feedforward_layernorm, s.post_feedforward_layernorm = (Norm(d, eps) for _ in range(4))
        p = c['hidden_size_per_layer_input']; s.per_layer_input_gate = nn.Linear(d, p, bias=False); s.per_layer_projection = nn.Linear(p, d, bias=False); s.post_per_layer_input_norm = Norm(d, eps)
        s.register_buffer('layer_scalar', torch.ones(1))

    def forward(s, x, ple, cos, sin, mask, kv, kopt):
        T = x.shape[0]; h = s.input_layernorm(x)
        q = s.q_norm(s.q_proj(h).view(T, s.nh, s.hd)); q = q * cos[:, None] + rotate_half(q) * sin[:, None]
        if s.shared: k, v = kv[s.kind]
        else:
            k = s.k_norm(s.k_proj(h).view(T, 1, s.hd)); k = k * cos[:, None] + rotate_half(k) * sin[:, None]
            v = s.v_norm(s.v_proj(h).view(T, 1, s.hd))
            if s.store: kv[s.kind] = (k, v)
        if isinstance(mask, tuple):  # reference path: plain attention one document at a time
            segs, win = mask; outs = []
            for lo, hi in segs:
                n = hi - lo; ii = torch.arange(n, device=x.device); m_ = ii[None, :] <= ii[:, None]
                if s.sliding: m_ = m_ & (ii[:, None] - ii[None, :] < win)
                outs.append(F.scaled_dot_product_attention(q[lo:hi].transpose(0, 1)[None], k[lo:hi].transpose(0, 1)[None], v[lo:hi].transpose(0, 1)[None], attn_mask=m_, scale=1.0, enable_gqa=True)[0])
            if T > segs[-1][1]: outs.append(torch.zeros(s.nh, T - segs[-1][1], s.hd, device=x.device, dtype=q.dtype))
            a = torch.cat(outs, 1)[None]
        else:
            a = _flex(q.transpose(0, 1)[None], k.transpose(0, 1)[None], v.transpose(0, 1)[None], block_mask=mask, scale=1.0, enable_gqa=True, kernel_options=kopt)
        a = s.o_proj(a[0].transpose(0, 1).reshape(T, s.nh * s.hd))
        x = x + s.post_attention_layernorm(a)
        h = s.pre_feedforward_layernorm(x); m = s.down_proj(F.gelu(s.gate_proj(h), approximate='tanh') * s.up_proj(h))
        x = x + s.post_feedforward_layernorm(m)
        g = F.gelu(s.per_layer_input_gate(x), approximate='tanh') * ple
        x = x + s.post_per_layer_input_norm(s.per_layer_projection(g))
        return x * s.layer_scalar


class GemmaE2B(nn.Module):
    def __init__(s, c, answer_ids):
        super().__init__(); s.c = c; d, L, p = c['hidden_size'], c['num_hidden_layers'], c['hidden_size_per_layer_input']
        s.embed_tokens = nn.Embedding(c['vocab_size'], d); s.embed_tokens_per_layer = nn.Embedding(c['vocab_size_per_layer_input'], L * p)
        s.per_layer_model_projection = nn.Linear(d, L * p, bias=False); s.per_layer_projection_norm = Norm(p, c['rms_norm_eps'])
        s.layers = nn.ModuleList(Layer(c, i) for i in range(L)); s.norm = Norm(d, c['rms_norm_eps']); s.answer = list(answer_ids)
        s.kopt = None
        global _WIN; _WIN = c['sliding_window']
        for t, hd in (('sliding_attention', c['head_dim']), ('full_attention', c['global_head_dim'])):
            r = c['rope_parameters'][t]
            if r.get('rope_type', 'default') == 'proportional':
                n = int(r.get('partial_rotary_factor', 1.0) * hd // 2)
                inv = 1.0 / (r['rope_theta'] ** (torch.arange(0, 2 * n, 2, dtype=torch.int64).float() / hd)); inv = torch.cat((inv, torch.zeros(hd // 2 - n)))
            else: inv = 1.0 / (r['rope_theta'] ** (torch.arange(0, hd, 2, dtype=torch.float) / hd))
            s.register_buffer(f'inv_{t}', inv, persistent=False)

    def rope(s, pos, t, dtype):
        f = pos.float()[:, None] * getattr(s, f'inv_{t}')[None]; e = torch.cat((f, f), -1); return e.cos().to(dtype), e.sin().to(dtype)

    def masks(s, doc, T):
        # One persistent buffer and two fixed mask functions: a new closure per call makes the compiler see new code
        # every time and recompile until it gives up (measured: 64 recompiles, then unfused attention, 15x slower).
        global _DOC
        if _DOC is None or _DOC.numel() < T or _DOC.device != doc.device: _DOC = torch.full((max(T, 32768),), -1, dtype=torch.long, device=doc.device)
        _DOC[:T].copy_(doc)
        return {'full_attention': create_block_mask(_full_mod, None, None, T, T, device=doc.device), 'sliding_attention': create_block_mask(_slide_mod, None, None, T, T, device=doc.device)}

    def forward(s, ids, pos, doc, ends):
        """ids, pos, doc: [T] packed stream (pad tokens carry doc -1); ends: index of each document's last token. Returns [n, 2] logits."""
        T = ids.shape[0]; c = s.c; L, p = c['num_hidden_layers'], c['hidden_size_per_layer_input']
        x = s.embed_tokens(ids) * torch.tensor(c['hidden_size'] ** 0.5, dtype=s.embed_tokens.weight.dtype)
        tok = (s.embed_tokens_per_layer(ids) * torch.tensor(p ** 0.5, dtype=s.embed_tokens_per_layer.weight.dtype)).view(T, L, p)
        pr = s.per_layer_projection_norm((s.per_layer_model_projection(x) * c['hidden_size'] ** -0.5).view(T, L, p))
        ple = (pr + tok) * 2 ** -0.5
        rope = {t: s.rope(pos, t, x.dtype) for t in ('sliding_attention', 'full_attention')}; kv = {}
        if getattr(s, 'reference_attention', False):
            e = ends.tolist(); segs = list(zip([0] + [v + 1 for v in e[:-1]], [v + 1 for v in e])); M = {t: (segs, c['sliding_window']) for t in rope}
        else: M = s.masks(doc, T)
        for i, layer in enumerate(s.layers):
            kind = s.c['layer_types'][i]; cos, sin = rope[kind]; ko = s.kopt.get(kind) if isinstance(s.kopt, dict) and ('BLOCK_M' not in s.kopt) else s.kopt
            x = layer(x, ple[:, i], cos, sin, M[kind], kv, ko)
        h = s.norm(x)[ends]; lg = h @ s.embed_tokens.weight[s.answer].to(h.dtype).T
        cap = c['final_logit_softcapping']; return (torch.tanh(lg.float() / cap) * cap) if cap else lg.float()


def load(path, answer_ids, device='cuda', trained_dtype=torch.float32, frozen_dtype=torch.bfloat16):
    c = json.load(open(Path(path) / 'config.json'))['text_config']
    with torch.device('meta'): m = GemmaE2B(c, answer_ids)
    sd = {}
    with safe_open(str(Path(path) / 'model.safetensors'), 'pt', device='cpu') as f:
        for k in f.keys():
            if k.startswith('model.language_model.'): sd[k[len('model.language_model.'):].replace('.self_attn.', '.').replace('.mlp.', '.')] = f.get_tensor(k)
    params, bufs = dict(m.named_parameters()), [k for k, _ in m.named_buffers() if not k.startswith('inv_')]
    missing = [k for k in list(params) + bufs if k not in sd]; assert not missing, missing[:8]
    m = m.to_empty(device=device)
    with torch.no_grad():
        for k in params:
            m.get_parameter(k).data = sd[k].to(device, frozen_dtype if k.startswith('embed_tokens') else trained_dtype)
        for k in bufs: m.get_buffer(k).copy_(sd[k].to(device, torch.float32))
        for t in ('sliding_attention', 'full_attention'):
            hd = c['head_dim'] if t == 'sliding_attention' else c['global_head_dim']; r = c['rope_parameters'][t]
            if r.get('rope_type', 'default') == 'proportional':
                n = int(r.get('partial_rotary_factor', 1.0) * hd // 2); inv = 1.0 / (r['rope_theta'] ** (torch.arange(0, 2 * n, 2, dtype=torch.int64).float() / hd)); inv = torch.cat((inv, torch.zeros(hd // 2 - n)))
            else: inv = 1.0 / (r['rope_theta'] ** (torch.arange(0, hd, 2, dtype=torch.float) / hd))
            getattr(m, f'inv_{t}').data = inv.to(device)
    for n, q in m.named_parameters(): q.requires_grad = not n.startswith('embed_tokens')
    # Measured: Ada (101 KB on-chip) needs small blocks everywhere; H100 (228 KB) runs default blocks on the windowed
    # layers but needs 64x32 single-stage blocks on the 512-wide global layers once Inductor fuses the norm and RoPE in.
    if torch.cuda.get_device_properties(device).shared_memory_per_block_optin < 160 * 1024: m.kopt = SMALL_SMEM
    else: m.kopt = {'full_attention': H100_GLOBAL}
    return m


def pack(docs, device='cuda', multiple=128):
    """docs: list of token-id lists. Returns ids, pos, doc, ends, padded to a multiple of 128 with doc -1."""
    ids, pos, doc, ends, t = [], [], [], [], 0
    for j, x in enumerate(docs):
        ids += x; pos += range(len(x)); doc += [j] * len(x); t += len(x); ends.append(t - 1)
    pad = (-t) % multiple; ids += [0] * pad; pos += [0] * pad; doc += [-1] * pad
    T = lambda v: torch.tensor(v, device=device)
    return T(ids), T(pos), T(doc), T(ends)


def export(m, src, out):
    """Write a full checkpoint in the original file layout: every tensor of src/model.safetensors, with the trained
    text weights replaced (BF16) under their original names, plus the config and tokenizer files."""
    import shutil
    from safetensors.torch import save_file
    out = Path(out); out.mkdir(parents=True, exist_ok=True); src = Path(src)
    # torch.compile wraps each layer, so names carry an extra '_orig_mod.' segment; drop it.
    mine = {n.replace('._orig_mod', ''): p for n, p in m.named_parameters() if p.requires_grad}
    def orig(n):
        parts = n.split('.')
        if parts[0] == 'layers' and parts[2] in ('q_proj', 'k_proj', 'v_proj', 'o_proj', 'q_norm', 'k_norm'): parts.insert(2, 'self_attn')
        if parts[0] == 'layers' and parts[2] in ('gate_proj', 'up_proj', 'down_proj'): parts.insert(2, 'mlp')
        return 'model.language_model.' + '.'.join(parts)
    names = {orig(n): n for n in mine}; full = {}; replaced = 0
    with safe_open(str(src / 'model.safetensors'), 'pt', device='cpu') as f:
        for k in f.keys():
            t = f.get_tensor(k)
            if k in names: t = mine[names[k]].detach().to('cpu', t.dtype); replaced += 1
            full[k] = t.contiguous()
    assert replaced == len(mine), (replaced, len(mine))
    save_file(full, str(out / 'model.safetensors'), metadata={'format': 'pt'})
    for f in src.iterdir():
        if f.suffix in ('.json', '.jinja') or f.name.startswith('tokenizer'):
            if f.name != 'model.safetensors.index.json': shutil.copy(f, out / f.name)
    return replaced
