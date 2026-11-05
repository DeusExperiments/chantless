import sys, json; sys.path.insert(0, '/root/g/mvp'); from chantless import gguf_ops as G
h = G._Hdr('/root/g/src.gguf'); names = {t['name'] for t in h.tensors}
KVSRC = {13, 14}; NKV = 15
groups = {
  'token_embd': {'quantizable': ['token_embd.weight'], 'fixed': [], 'droppable': False, 'kv': False, 'imatrix_ok': False},
  'per_layer_token_embd': {'quantizable': ['per_layer_token_embd.weight'], 'fixed': [], 'droppable': False, 'kv': False, 'imatrix_ok': False},
  'per_layer_model_proj': {'quantizable': ['per_layer_model_proj.weight'], 'fixed': [], 'droppable': False, 'kv': False, 'imatrix_ok': True}}
dead = []
for il in range(35):
    p = f'blk.{il}.'
    aq = [p + x for x in ('attn_q.weight', 'attn_k.weight', 'attn_v.weight', 'attn_output.weight') if p + x in names]
    af = [p + x for x in ('attn_norm.weight', 'attn_q_norm.weight', 'attn_k_norm.weight', 'post_attention_norm.weight') if p + x in names]
    if il >= NKV:   # shared-KV layers never read their own K/V: dead weights, removed losslessly
        dead += [p + 'attn_k.weight', p + 'attn_v.weight', p + 'attn_k_norm.weight']
        aq = [x for x in aq if x not in dead]; af = [x for x in af if x not in dead]
    groups[f'blk.{il}.attn'] = {'quantizable': aq, 'fixed': af, 'droppable': il not in KVSRC, 'kv': il < NKV, 'imatrix_ok': True}
    groups[f'blk.{il}.ffn'] = {'quantizable': [p + x for x in ('ffn_gate.weight', 'ffn_up.weight', 'ffn_down.weight')], 'fixed': [p + 'ffn_norm.weight', p + 'post_ffw_norm.weight'], 'droppable': True, 'kv': False, 'imatrix_ok': True}
    groups[f'blk.{il}.ple'] = {'quantizable': [p + 'inp_gate.weight', p + 'proj.weight'], 'fixed': [p + 'post_norm.weight'], 'droppable': True, 'kv': False, 'imatrix_ok': True}
covered = {n for g in groups.values() for n in g['quantizable'] + g['fixed']} | set(dead)
rest = sorted(names - covered); print('ungrouped (always kept):', rest[:8], len(rest))
missing = [n for g in groups.values() for n in g['quantizable'] + g['fixed'] if n not in names]; assert not missing, missing
json.dump(groups, open('/root/g/groups.json', 'w'), indent=0); json.dump(dead, open('/root/g/dead.json', 'w'))
print('groups', len(groups), 'dead tensors', len(dead))
