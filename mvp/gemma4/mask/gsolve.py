"""One-shot fidelity solve for the Gemma 4 E2B detector (adopted classifier path).
Predicted scores = source scores + sum of solo shifts measured from the BF16 base (src0); an MLP at tier t and width w
adds shift(t) + cut(w), cut linear between measured widths, width 0 = drop. Objective: 1 - Spearman with the source
on the 900-document store. Bytes: exact per tensor + calibrated header, minus the vocabulary trim.
Usage: gsolve.py BUDGET K TAG [--no-iq1]"""
import sys, json, math, struct, glob, os, numpy as np
sys.path.insert(0, '/root/g/mvp'); from chantless import gguf_ops as G
R = '/root/g'; budget, K, tag = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]; noiq1 = '--no-iq1' in sys.argv
GR = json.load(open(f'{R}/groups.json')); dead = set(json.load(open(f'{R}/dead.json')))
BLK = {'F32': (1, 4), 'BF16': (1, 2), 'F16': (1, 2), 'Q8_0': (32, 34), 'Q6_K': (256, 210), 'Q5_K': (256, 176), 'Q4_K': (256, 144),
       'Q3_K': (256, 110), 'Q2_K': (256, 84), 'IQ2_XS': (256, 74), 'IQ1_S': (256, 50)}
pad = lambda n: (n + 31) // 32 * 32
def tbytes(dims, ty):
    bs, ts = BLK[ty]; assert dims[0] % bs == 0, (dims, ty); return pad(ts * (dims[0] // bs) * int(np.prod(dims[1:], dtype=np.int64)))
ti = lambda name, nd: 8 + len(name.encode()) + 4 + 8 * nd + 4 + 8
h = G._Hdr(f'{R}/src0.gguf'); TEN = [(t['name'], list(t['dims']), t['type']) for t in h.tensors]
N0 = {int(n.split('.')[1]): d[0] for n, d, _ in TEN if n.endswith('ffn_down.weight')}
def file_bytes(tiers, widths, kv_end):
    tier = {}; drop = set()
    for g, t in tiers.items():
        if t == 'drop': drop |= set(GR[g]['quantizable'] + GR[g]['fixed'])
        else:
            for n in GR[g]['quantizable']: tier[n] = t
    tot = 0; tib = 0
    for n, d, ty in TEN:
        if n in drop: continue
        p = n.split('.'); d = list(d)
        if p[0] == 'blk' and int(p[1]) in widths and p[2] in ('ffn_gate', 'ffn_up', 'ffn_down'):
            w = widths[int(p[1])]; d = [w, d[1]] if p[2] == 'ffn_down' else [d[0], w]
        tot += tbytes(d, tier.get(n, ty)); tib += ti(n, len(d))
    return pad(kv_end + tib) + tot
# calibrate the KV end of a quantized build (llama-quantize adds imatrix keys) on a real probe
pj = json.load(open(f'{R}/sc/p/blk.0.attn=Q4_K.json'))['bytes']
base_t = {g: 'BF16' for g in GR}; t1 = dict(base_t); t1['blk.0.attn'] = 'Q4_K'
kv_q = h.kv_end + (pj - file_bytes(t1, {}, h.kv_end))
assert file_bytes(t1, {}, kv_q) == pj
# vocabulary trim: rows removed from both token tables, plus the tokenizer KV that goes away
keep = np.load(f'{R}/keep-{K}.npy'); V = 262144
def kvs(h):
    m = h.m; p = h.kv_start; out = {}
    for _ in range(h.n_kv):
        (kl,) = struct.unpack_from('<Q', m, p); p += 8; k = m[p:p + kl].decode(); p += kl
        (vt,) = struct.unpack_from('<I', m, p); p += 4; s = p; p = h._skip(p, vt); out[k] = bytes(m[s:p])
    return out
def strs(raw):
    et, n = struct.unpack_from('<IQ', raw, 0); p = 12; o = []
    for _ in range(n): (sl,) = struct.unpack_from('<Q', raw, p); p += 8; o.append(raw[p:p + sl]); p += sl
    return o
kv = kvs(h); toks = strs(kv['tokenizer.ggml.tokens']); merges = strs(kv['tokenizer.ggml.merges']); ks = {toks[i] for i in keep.tolist()}
km = 0
for r in merges:
    k = r.find(b' ', 1)
    if r[:k] in ks and r[k + 1:] in ks and r[:k] + r[k + 1:] in ks: km += 8 + len(r)
kv_cut = sum(8 + len(t) for t in toks) - sum(8 + len(toks[i]) for i in keep.tolist()) + 4 * (V - len(keep)) * 2 + (sum(8 + len(r) for r in merges) - km)
def final_bytes(tiers, widths):
    b = file_bytes(tiers, widths, kv_q) - kv_cut
    for tab, cols in (('token_embd', 1536), ('per_layer_token_embd', 8960)):
        bs, ts = BLK[tiers[tab] if tiers[tab] != 'BF16' else 'BF16']; b -= (V - len(keep)) * (ts * cols // bs)
    return b
# probes
src = np.load(f'{R}/sc/src_both.npy')[:900]
shift, cut = {}, {}
for f in glob.glob(f'{R}/sc/p/*.npy'):
    t = os.path.basename(f)[:-4]; x = np.load(f)[:900] - src
    if t.startswith('w'): il, w = t[1:].split('='); cut[(int(il), int(w))] = x
    else: g, ti_ = t.split('='); shift[(g, ti_)] = x
for il in range(35):
    if (f'blk.{il}.ffn', 'drop') in shift: cut[(il, 0)] = shift[(f'blk.{il}.ffn', 'drop')]
print('probes', len(shift), 'width', len(cut), 'kv_cut', kv_cut, flush=True)
Z = np.zeros(900)
def cutv(il, w):
    if w >= N0[il]: return Z
    pts = sorted([(ww, v) for (l, ww), v in cut.items() if l == il] + [(N0[il], Z)], key=lambda q: q[0])
    for (w0, v0), (w1, v1) in zip(pts, pts[1:]):
        if w0 <= w <= w1: a = (w - w0) / (w1 - w0); return (1 - a) * v0 + a * v1
    raise ValueError((il, w))
opts = {}
for g, info in GR.items():
    if g == 'per_layer_model_proj': opts[g] = [('BF16', 0)]; continue
    if g == 'token_embd': ts = ['Q8_0', 'Q4_K', 'Q2_K']
    elif g == 'per_layer_token_embd': ts = ['Q8_0', 'Q4_K', 'Q3_K', 'Q2_K']
    else: ts = ['Q8_0', 'Q4_K', 'IQ2_XS'] + ([] if noiq1 else ['IQ1_S'])
    ts = [t for t in ts if t == 'Q8_0' or (g, t) in shift]
    if g.endswith('.ffn'):
        il = int(g.split('.')[1]); measured = any(l == il and w > 0 for (l, w) in cut)
        o = [(t, w) for t in ts for w in (range(N0[il], 0, -256) if measured else [N0[il]])]
    else: o = [(t, 0) for t in ts]
    if info['droppable'] and (g, 'drop') in shift: o.append(('drop', 0))
    opts[g] = o
def dv(g, o):
    t, w = o
    if t == 'drop': return shift[(g, 'drop')]
    x = Z if t in ('Q8_0', 'BF16') else shift[(g, t)]
    if g.endswith('.ffn'): il = int(g.split('.')[1]); x = x + cutv(il, w)
    return x
rk = lambda a: np.argsort(np.argsort(a, kind='stable'), kind='stable').astype(float)
rs = rk(src); rs = (rs - rs.mean()) / rs.std()
def dmg(x): r = rk(x); r = (r - r.mean()) / r.std(); return 1.0 - float((r * rs).mean())
def spec(m):
    tiers = {g: o[0] for g, o in m.items()}
    widths = {int(g.split('.')[1]): o[1] for g, o in m.items() if g.endswith('.ffn') and o[0] != 'drop' and o[1] < N0[int(g.split('.')[1])]}
    return tiers, widths
FB = lambda m: final_bytes(*spec(m))
ref = {g: (opts[g][0][0], N0[int(g.split('.')[1])] if g.endswith('.ffn') else 0) for g in GR}
F0 = FB(ref); c = {}
for g in GR:
    for o in opts[g]:
        m = dict(ref); m[g] = o; c[(g, o)] = FB(m) - F0
B = lambda m: F0 + sum(c[(g, o)] for g, o in m.items())
m = dict(ref); x = src.copy(); cur = dmg(x); dvc = {(g, o): dv(g, o) for g in GR for o in opts[g]}
print('rich start', B(m), flush=True)
while B(m) > budget:
    best = None
    for g in GR:
        for o in opts[g]:
            db = c[(g, m[g])] - c[(g, o)]
            if db <= 0: continue
            y = x - dvc[(g, m[g])] + dvc[(g, o)]; d = dmg(y) - cur; sc = (d + 1e-12) / db if d >= 0 else d * db
            if best is None or sc < best[0]: best = (sc, g, o, y, d)
    _, g, o, x, d = best; m[g] = o; cur += d
print('descent', B(m), 'pred fidelity %.4f' % (1 - cur), flush=True)
for r in range(60):
    b = B(m); singles = []
    for g in GR:
        for o in opts[g]:
            if o != m[g]: singles.append((dmg(x - dvc[(g, m[g])] + dvc[(g, o)]) - cur, c[(g, o)] - c[(g, m[g])], g, o))
    best = (-1e-7, None)
    for d, db, g, o in singles:
        if b + db <= budget and d < best[0]: best = (d, [(g, o)])
    ups = sorted([s for s in singles if s[1] > 0], key=lambda s: s[0])[:50]; downs = sorted([s for s in singles if s[1] < 0], key=lambda s: s[0] / -s[1])[:50]
    for d1, db1, g1, o1 in ups:
        y1 = x - dvc[(g1, m[g1])] + dvc[(g1, o1)]
        for d2, db2, g2, o2 in downs:
            if g1 != g2 and b + db1 + db2 <= budget:
                d = dmg(y1 - dvc[(g2, m[g2])] + dvc[(g2, o2)]) - cur
                if d < best[0]: best = (d, [(g1, o1), (g2, o2)])
    if best[1] is None: break
    for g, o in best[1]: x = x - dvc[(g, m[g])] + dvc[(g, o)]; m[g] = o
    cur = dmg(x)
tiers, widths = spec(m)
import collections
print(tag, 'pred fidelity %.4f' % (1 - cur), 'final bytes', FB(m), 'budget', budget, dict(collections.Counter(tiers.values())), 'widths', widths, flush=True)
json.dump({'tag': tag, 'K': K, 'tiers': tiers, 'widths': {str(k): v for k, v in widths.items()}, 'pred_fidelity': 1 - cur, 'pred_bytes': FB(m)}, open(f'{R}/spec_{tag}.json', 'w'), indent=0)
