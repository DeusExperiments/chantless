"""The Gemma 4 E2B Chantless mask as a PyTorch student, for mask-stage learning (the Gemma counterpart of
learn.build_student). Maths is lean.py's, which was checked against the reference implementation; the weights are the
mask file's own, decoded per tensor, and the structure follows the patched llama.cpp graph:
  - attention removed (no attn_output): the residual stream passes through;
  - per-layer embedding removed (no inp_gate): nothing is added;
  - MLP width per layer as in the file; vocabulary as trimmed in the file (80,000 rows);
  - layers 15-34 read layer 13's (sliding) or 14's (full) keys and values.
Gemma 4 conversion to GGUF changes no values (norm_shift 0), so tensors load as they are.
Only the human and ai output rows are computed; the score is their soft-capped logit difference, as logodds reports."""
import sys
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
sys.path.insert(0, '/root/g'); sys.path.insert(0, '/root/g/mvp')
import lean
from chantless import lattice, learn

GGML = '/root/g/llama.cpp/build/bin/libggml-base.so'
LAYER = {'attn_norm': 'input_layernorm', 'attn_q': 'q_proj', 'attn_k': 'k_proj', 'attn_v': 'v_proj', 'attn_output': 'o_proj',
         'attn_q_norm': 'q_norm', 'attn_k_norm': 'k_norm', 'post_attention_norm': 'post_attention_layernorm',
         'ffn_norm': 'pre_feedforward_layernorm', 'ffn_gate': 'gate_proj', 'ffn_up': 'up_proj', 'ffn_down': 'down_proj',
         'post_ffw_norm': 'post_feedforward_layernorm', 'inp_gate': 'per_layer_input_gate', 'proj': 'per_layer_projection',
         'post_norm': 'post_per_layer_input_norm', 'layer_output_scale': 'layer_scalar'}
TOPLEVEL = {'token_embd': 'embed_tokens', 'per_layer_token_embd': 'embed_tokens_per_layer', 'per_layer_model_proj': 'per_layer_model_projection',
            'per_layer_proj_norm': 'per_layer_projection_norm', 'output_norm': 'norm'}
SKIP = {'rope_freqs.weight'}    # lean computes proportional RoPE itself


def floats(art, name):
    """learn.floats, with ggml's own spelling of the k-quant decoders (dequantize_row_q2_K, not q2_k)."""
    t = art.tensors[name]
    if t['type'] in ('F32', 'BF16') or t['type'] in lattice.TOP: return learn.floats(art, name, GGML)
    import ctypes
    fn = getattr(ctypes.CDLL(GGML), 'dequantize_row_' + (t['type'].lower()[:-1] + 'K' if t['type'].endswith('_K') else t['type'].lower()))
    fn.argtypes, fn.restype = [ctypes.c_char_p, ctypes.c_void_p, ctypes.c_int64], None
    y = np.empty(t['n_elements'], np.float32); fn(bytes(art.raw(name)), y.ctypes.data, t['n_elements'])
    return y.reshape(art.shape(name))


def mod_path(gg):
    """GGUF tensor name -> (module path in the student, parameter name)."""
    p = gg.split('.')
    if p[0] == 'blk':
        return f'layers.{p[1]}.{LAYER[p[2]]}', 'weight'
    return TOPLEVEL[p[0]], 'weight'


class SLayer(nn.Module):
    def __init__(s, c, i, has_attn, ffn, has_ple):
        super().__init__()
        d, eps = c['hidden_size'], c['rms_norm_eps']; s.kind = c['layer_types'][i]; s.sliding = s.kind == 'sliding_attention'
        s.hd = c['head_dim'] if s.sliding else c['global_head_dim']; s.nh = c['num_attention_heads']
        first_shared = c['num_hidden_layers'] - c['num_kv_shared_layers']; s.shared = i >= first_shared
        prev = c['layer_types'][:first_shared]; s.store = (not s.shared) and i == len(prev) - 1 - prev[::-1].index(s.kind)
        s.has_attn, s.ffn, s.has_ple = has_attn, ffn, has_ple
        assert has_attn or not s.store, 'a layer whose keys and values are shared lost its attention'
        if has_attn:
            s.input_layernorm, s.post_attention_layernorm = lean.Norm(d, eps), lean.Norm(d, eps)
            s.q_proj = nn.Linear(d, s.nh * s.hd, bias=False); s.q_norm = lean.Norm(s.hd, eps); s.o_proj = nn.Linear(s.nh * s.hd, d, bias=False)
            if not s.shared:
                s.k_proj = nn.Linear(d, s.hd, bias=False); s.v_proj = nn.Linear(d, s.hd, bias=False); s.k_norm = lean.Norm(s.hd, eps)
                s.v_norm = lean.Norm(s.hd, eps, scale=False)
        if ffn:
            s.pre_feedforward_layernorm, s.post_feedforward_layernorm = lean.Norm(d, eps), lean.Norm(d, eps)
            s.gate_proj = nn.Linear(d, ffn, bias=False); s.up_proj = nn.Linear(d, ffn, bias=False); s.down_proj = nn.Linear(ffn, d, bias=False)
        if has_ple:
            p = c['hidden_size_per_layer_input']
            s.per_layer_input_gate = nn.Linear(d, p, bias=False); s.per_layer_projection = nn.Linear(p, d, bias=False); s.post_per_layer_input_norm = lean.Norm(d, eps)
        s.layer_scalar = nn.Parameter(torch.ones(1))

    def forward(s, x, ple, cos, sin, mask, kv, kopt):
        T = x.shape[0]
        if s.has_attn:
            h = s.input_layernorm(x)
            q = s.q_norm(s.q_proj(h).view(T, s.nh, s.hd)); q = q * cos[:, None] + lean.rotate_half(q) * sin[:, None]
            if s.shared: k, v = kv[s.kind]
            else:
                k = s.k_norm(s.k_proj(h).view(T, 1, s.hd)); k = k * cos[:, None] + lean.rotate_half(k) * sin[:, None]
                v = s.v_norm(s.v_proj(h).view(T, 1, s.hd))
                if s.store: kv[s.kind] = (k, v)
            dt = torch.bfloat16 if torch.is_autocast_enabled() else q.dtype    # the residual stream stays float32; attention runs in bf16
            a = lean._flex(q.to(dt).transpose(0, 1)[None], k.to(dt).transpose(0, 1)[None], v.to(dt).transpose(0, 1)[None], block_mask=mask, scale=1.0, enable_gqa=True, kernel_options=kopt)
            x = x + s.post_attention_layernorm(s.o_proj(a[0].transpose(0, 1).reshape(T, s.nh * s.hd)))
        if s.ffn:
            h = s.pre_feedforward_layernorm(x)
            x = x + s.post_feedforward_layernorm(s.down_proj(F.gelu(s.gate_proj(h), approximate='tanh') * s.up_proj(h)))
        if s.has_ple:
            g = F.gelu(s.per_layer_input_gate(x), approximate='tanh') * ple
            x = x + s.post_per_layer_input_norm(s.per_layer_projection(g))
        return x * s.layer_scalar


class Student(lean.GemmaE2B):
    """lean.GemmaE2B with the file's structure. `ckpt` recomputes each layer in the backward pass (switch training)."""

    def __init__(s, c, answer_ids, art):
        nn.Module.__init__(s); s.c = c; d, L, p = c['hidden_size'], c['num_hidden_layers'], c['hidden_size_per_layer_input']
        V = art.tensors['token_embd.weight']['dims'][1]; Vp = art.tensors['per_layer_token_embd.weight']['dims'][1]
        s.embed_tokens = nn.Embedding(V, d); s.embed_tokens_per_layer = nn.Embedding(Vp, L * p)
        s.per_layer_model_projection = nn.Linear(d, L * p, bias=False); s.per_layer_projection_norm = lean.Norm(p, c['rms_norm_eps'])
        T = art.tensors
        s.layers = nn.ModuleList(SLayer(c, i, f'blk.{i}.attn_output.weight' in T, T[f'blk.{i}.ffn_down.weight']['dims'][0] if f'blk.{i}.ffn_down.weight' in T else 0,
                                        f'blk.{i}.inp_gate.weight' in T) for i in range(L))
        s.norm = lean.Norm(d, c['rms_norm_eps']); s.answer = list(answer_ids); s.kopt = None; s.ckpt = False
        lean._WIN = c['sliding_window']
        for t, hd in (('sliding_attention', c['head_dim']), ('full_attention', c['global_head_dim'])):
            r = c['rope_parameters'][t]
            if r.get('rope_type', 'default') == 'proportional':
                n = int(r.get('partial_rotary_factor', 1.0) * hd // 2)
                inv = 1.0 / (r['rope_theta'] ** (torch.arange(0, 2 * n, 2, dtype=torch.int64).float() / hd)); inv = torch.cat((inv, torch.zeros(hd // 2 - n)))
            else: inv = 1.0 / (r['rope_theta'] ** (torch.arange(0, hd, 2, dtype=torch.float) / hd))
            s.register_buffer(f'inv_{t}', inv, persistent=False)

    def forward(s, ids, pos, doc, ends):
        T = ids.shape[0]; c = s.c; L, p = c['num_hidden_layers'], c['hidden_size_per_layer_input']
        x = s.embed_tokens(ids) * torch.tensor(c['hidden_size'] ** 0.5, dtype=s.embed_tokens.weight.dtype)
        tok = (s.embed_tokens_per_layer(ids) * torch.tensor(p ** 0.5, dtype=s.embed_tokens_per_layer.weight.dtype)).view(T, L, p)
        pr = s.per_layer_projection_norm((s.per_layer_model_projection(x) * c['hidden_size'] ** -0.5).view(T, L, p))
        ple = (pr + tok) * 2 ** -0.5
        rope = {t: s.rope(pos, t, x.dtype) for t in ('sliding_attention', 'full_attention')}; kv = {}
        M = s.masks(doc, T)
        for i, layer in enumerate(s.layers):
            kind = c['layer_types'][i]; cos, sin = rope[kind]; ko = s.kopt.get(kind) if isinstance(s.kopt, dict) and ('BLOCK_M' not in s.kopt) else s.kopt
            if s.ckpt and torch.is_grad_enabled():
                if layer.store:     # the stored keys and values must come out of the checkpoint as outputs
                    def f(x, ple_i, layer=layer, cos=cos, sin=sin, m=M[kind], ko=ko):
                        d = {}; y = layer(x, ple_i, cos, sin, m, d, ko); return (y,) + d[layer.kind]
                    x, k_, v_ = checkpoint(f, x, ple[:, i], use_reentrant=False); kv[kind] = (k_, v_)
                else:
                    x = checkpoint(layer, x, ple[:, i], cos, sin, M[kind], kv, ko, use_reentrant=False)
            else:
                x = layer(x, ple[:, i], cos, sin, M[kind], kv, ko)
        h = s.norm(x)[ends]; lg = h @ s.embed_tokens.weight[s.answer].to(h.dtype).T
        cap = c['final_logit_softcapping']; return (torch.tanh(lg.float() / cap) * cap) if cap else lg.float()


def build(art, cfg_text, answer_ids, train, device='cuda'):
    """The file as a Student. `train` names grid tensors that get the straight-through on-grid parametrization (learn.OnGrid).
    Returns (model, grids) as learn.build_student does."""
    from torch.nn.utils import parametrize
    with torch.device('meta'):
        m = Student(cfg_text, answer_ids, art)
    m = m.to_empty(device=device)
    mods = dict(m.named_modules()); seen = set()
    with torch.no_grad():
        for name in art.tensors:
            if name in SKIP: continue
            path, attr = mod_path(name)
            v = torch.from_numpy(floats(art, name).copy())
            if path.endswith('layer_scalar'):
                mods[path.rsplit('.', 1)[0]].layer_scalar.data = v.reshape(1).to(device); seen.add(path); continue
            w = getattr(mods[path], attr); assert tuple(w.shape) == tuple(v.shape), (name, tuple(w.shape), tuple(v.shape))
            w.data = v.to(device); seen.add(path)
        for t in ('sliding_attention', 'full_attention'):
            c = cfg_text; hd = c['head_dim'] if t == 'sliding_attention' else c['global_head_dim']; r = c['rope_parameters'][t]
            if r.get('rope_type', 'default') == 'proportional':
                n = int(r.get('partial_rotary_factor', 1.0) * hd // 2); inv = 1.0 / (r['rope_theta'] ** (torch.arange(0, 2 * n, 2, dtype=torch.int64).float() / hd)); inv = torch.cat((inv, torch.zeros(hd // 2 - n)))
            else: inv = 1.0 / (r['rope_theta'] ** (torch.arange(0, hd, 2, dtype=torch.float) / hd))
            getattr(m, f'inv_{t}').data = inv.to(device)
    have = {n.rsplit('.', 1)[0] if not n.endswith('layer_scalar') else n for n, _ in m.named_parameters()}
    missing = sorted(have - seen); assert not missing, missing[:10]
    for q in m.parameters(): q.requires_grad_(False)
    m.kopt = lean.SMALL_SMEM if torch.cuda.get_device_properties(device).shared_memory_per_block_optin < 160 * 1024 else {'full_attention': lean.H100_GLOBAL}
    grids = {}
    for name in train:
        g = art.grid(name); path, _ = mod_path(name); mod, shape = mods[path], art.shape(name)
        fq = learn.OnGrid(torch.from_numpy(g.lo).reshape(shape).to(device), torch.from_numpy(g.step).reshape(shape).to(device), g.top)
        parametrize.register_parametrization(mod, 'weight', fq, unsafe=True)
        mod.parametrizations.weight.original.requires_grad_(True)
        grids[name] = (mod.parametrizations.weight.original, fq, torch.from_numpy(g.levels.copy()))
    m.eval()
    return m, grids


def gscores(model, docs, pair=None, device='cuda', amp=True):
    """learn.scores for the packed student: one stream, no padding, length rounded to 1,024 so few shapes compile."""
    ids, pos, doc, ends = lean.pack([d.tolist() for d in docs], device=device, multiple=1024)
    with torch.autocast('cuda', dtype=torch.bfloat16, enabled=amp):
        lg = model(ids, pos, doc, ends)
    return lg[:, 0] - lg[:, 1]


def install():
    """Route learn's scoring through the packed student and allow the many compiled shapes FlexAttention needs."""
    import torch._dynamo
    torch._dynamo.config.recompile_limit = 1024; torch._dynamo.config.accumulated_recompile_limit = 65536
    learn.scores = gscores
