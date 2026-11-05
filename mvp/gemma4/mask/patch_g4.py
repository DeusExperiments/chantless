p='/root/g/llama.cpp/src/models/gemma4.cpp'; s=open(p).read(); n0=len(s)
def rep(a,b,cnt=1):
    global s
    assert s.count(a)==cnt, (a[:80], s.count(a)); s=s.replace(a,b)
# loader: every per-layer sub-block may be absent (Chantless sub-block removal)
rep('layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);','layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, TENSOR_NOT_REQUIRED);')
rep('const int     kv_flags    = hparams.has_kv(i) ? 0 : TENSOR_NOT_REQUIRED;','const int     kv_flags    = TENSOR_NOT_REQUIRED;')
rep('layer.wq = create_tensor(tn(LLM_TENSOR_ATTN_Q, "weight", i), {n_embd, n_embd_head * n_head}, 0);','layer.wq = create_tensor(tn(LLM_TENSOR_ATTN_Q, "weight", i), {n_embd, n_embd_head * n_head}, TENSOR_NOT_REQUIRED);')
rep('layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd_head * n_head, n_embd}, 0);','layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd_head * n_head, n_embd}, TENSOR_NOT_REQUIRED);')
rep('layer.attn_q_norm    = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM,    "weight", i), {n_embd_head}, 0);','layer.attn_q_norm    = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM,    "weight", i), {n_embd_head}, TENSOR_NOT_REQUIRED);')
rep('layer.attn_post_norm = create_tensor(tn(LLM_TENSOR_ATTN_POST_NORM, "weight", i), {n_embd}, 0);','layer.attn_post_norm = create_tensor(tn(LLM_TENSOR_ATTN_POST_NORM, "weight", i), {n_embd}, TENSOR_NOT_REQUIRED);')
for t in ('FFN_NORM, "weight", i), {n_embd}, 0);','FFN_GATE, "weight", i), {n_embd,   n_ff_cur}, 0);','FFN_UP,   "weight", i), {n_embd,   n_ff_cur}, 0);','FFN_DOWN, "weight", i), {n_ff_cur, n_embd}, 0);','FFN_POST_NORM, "weight", i), {n_embd}, 0);'):
    rep(t, t[:-4]+'TENSOR_NOT_REQUIRED);')
for t in ('PER_LAYER_INP_GATE,  "weight", i), {n_embd, n_embd_per_layer}, 0);','PER_LAYER_PROJ,      "weight", i), {n_embd_per_layer, n_embd}, 0);','PER_LAYER_POST_NORM, "weight", i), {n_embd}, 0);'):
    rep(t, t[:-4]+'TENSOR_NOT_REQUIRED);')
# graph: attention block
rep('        // norm\n        cur = build_norm(inpL, model.layers[il].attn_norm, nullptr, LLM_NORM_RMS, il);',
    '        ggml_tensor * attn_out = nullptr;\n        if (model.layers[il].wo) {\n        // norm\n        cur = build_norm(inpL, model.layers[il].attn_norm, nullptr, LLM_NORM_RMS, il);')
rep('        ggml_tensor * attn_out = ggml_add(ctx0, cur, inpL);\n        cb(attn_out, "attn_out", il);\n',
    '        attn_out = ggml_add(ctx0, cur, inpL);\n        cb(attn_out, "attn_out", il);\n        } else {\n            // Chantless: attention sub-block removed, residual stream passes through\n            if (!prune_shared && il == n_layer - 1 && crop_before_nextn(inp_out_ids)) {\n                inpL = ggml_get_rows(ctx0, inpL, inp_out_ids);\n            }\n            attn_out = inpL;\n        }\n')
# graph: FFN block
a='        // feed-forward network\n'; b='        // residual connection\n        cur = ggml_add(ctx0, cur, attn_out);\n'
i=s.index(a); j=s.index(b,i)+len(b)
s=s[:i]+'        if (model.layers[il].ffn_down || model.layers[il].ffn_gate_inp) {\n'+s[i:j]+'        } else {\n            // Chantless: FFN sub-block removed\n            cur = attn_out;\n        }\n'+s[j:]
rep('        if (inp_per_layer) {\n            ggml_tensor * pe_in = cur;','        if (inp_per_layer && model.layers[il].per_layer_inp_gate) {\n            ggml_tensor * pe_in = cur;')
open(p,'w').write(s); print('patched', n0, '->', len(s))
