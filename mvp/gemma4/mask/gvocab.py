"""Keep sets for the Gemma vocabulary trim, rule P as for Qwen: base = every non-normal token (control, user-defined,
byte, unused) + tokens whose text is ASCII and has no letter or digit; then tokens by frequency on fit documents
(calibration + WildOutlines validation used in select), closed under BPE merge parents. Writes keep-<K>.npy."""
import sys, json, struct, numpy as np
sys.path.insert(0, '/root/g/mvp'); from chantless import gguf_ops as G
exec(open('/root/g/gtrim.py').read().split('def trim(')[0].split("sys.path.insert(0, '/root/g/mvp')")[1].split('\n', 1)[1]) if False else None
def read_kvs(h):
    m = h.m; p = h.kv_start; out = {}
    for _ in range(h.n_kv):
        (kl,) = struct.unpack_from('<Q', m, p); p += 8; k = m[p:p + kl].decode(); p += kl
        (vt,) = struct.unpack_from('<I', m, p); p += 4; s = p; p = h._skip(p, vt); out[k] = bytes(m[s:p])
    return out
def strs(raw):
    et, n = struct.unpack_from('<IQ', raw, 0); p = 12; o = []
    for _ in range(n): (sl,) = struct.unpack_from('<Q', raw, p); p += 8; o.append(raw[p:p + sl]); p += sl
    return o
h = G._Hdr('/root/g/src0.gguf'); kv = read_kvs(h)
tokens = strs(kv['tokenizer.ggml.tokens']); V = len(tokens); tt = np.frombuffer(kv['tokenizer.ggml.token_type'], '<i4', V, 12); merges = strs(kv['tokenizer.ggml.merges'])
tid = {t: i for i, t in enumerate(tokens)}; parents = {}
for r in merges:
    k = r.find(b' ', 1); a, b = r[:k], r[k + 1:]
    if a + b in tid and a in tid and b in tid: parents.setdefault(tid[a + b], (tid[a], tid[b]))
def read_trj(path):
    b = open(path, 'rb').read(); n = struct.unpack_from('<I', b, 4)[0]; p = 8; out = []
    for _ in range(n):
        L, nt = struct.unpack_from('<II', b, p); p += 8; out.append(np.frombuffer(b, '<i4', L, p)); p += 4 * L + 4 * nt
    return out
rows = json.load(open('/root/g/gd/select.json'))
docs = read_trj('/root/g/gd/cal.trj') + [d for d, r in zip(read_trj('/root/g/gd/select.trj'), rows) if r['task'].startswith('wo-val:')]
cnt = np.bincount(np.concatenate(docs), minlength=V)
def ascii_punct(t):
    s = t.replace('▁'.encode(), b' ')
    try: u = s.decode('utf-8')
    except UnicodeDecodeError: return False
    return len(u) > 0 and all(ord(c) < 128 and not c.isalnum() for c in u)
base = set(np.flatnonzero(tt != 1).tolist()) | {i for i, t in enumerate(tokens) if tt[i] == 1 and ascii_punct(t)}
order = np.lexsort((np.arange(V), -cnt))
def close(s):
    s = set(s); stack = list(s)
    while stack:
        x = stack.pop()
        for p in parents.get(x, ()):
            if p not in s: s.add(p); stack.append(p)
    return s
print('vocab', V, 'base', len(base), 'tokens seen in fit docs', int((cnt > 0).sum()), 'fit tokens', int(cnt.sum()))
for K in tuple(int(x) for x in sys.argv[1:]) or (50000, 65000, 80000):
    lo, hi = 0, V
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if len(close(base | set(order[:mid].tolist()))) <= K: lo = mid
        else: hi = mid - 1
    keep = np.array(sorted(close(base | set(order[:lo].tolist()))), np.int64); np.save(f'/root/g/keep-{K}.npy', keep)
    cov = cnt[keep].sum() / cnt.sum(); print('K', K, 'kept', len(keep), 'top-n', lo, 'fit occurrence coverage %.6f' % cov)
