"""Gemma 4 E2B Chantless probes and builds.
  solos               write the probe list (precision tiers and drops per group, widths per MLP)
  work GPU            probe worker on one GPU: build once to /dev/shm, score gd/both.trj, save sc/p/<tag>.npy
  build SPEC GPUS     build a candidate (tiers + widths + keep set), report bytes, score main store
Base for every probe: src0.gguf (BF16 source without the 60 dead shared-layer K/V tensors)."""
import sys, os, json, struct, time, subprocess
from pathlib import Path
import numpy as np
sys.path.insert(0, '/root/g/mvp')
from chantless import gguf_ops as G
from chantless.builder import Builder
from chantless.runtime import Toolchain
R = Path('/root/g'); SRC = str(R / 'src0.gguf'); IM = str(R / 'imatrix.gguf'); GR = json.load(open(R / 'groups.json'))
TC = Toolchain(R / 'llama.cpp/build/bin', Path('/bin/true'), ['0'], R / 'bin/logodds')
KEYFF = 'gemma4.feed_forward_length'
TIERS_BLOCK = ['Q4_K', 'IQ2_XS', 'IQ1_S']

def mask(tiers):
    return {'schema': 'chantless.mask/v2', 'objective_mode': 'preserve', 'groups': {g: {'tier': tiers.get(g, 'BF16')} for g in GR},
            'tensors': {}, 'floors': {}, 'weights': {'lineage': []}}

def nff0():
    h = G._Hdr(SRC); return {int(t['name'].split('.')[1]): t['dims'][0] for t in h.tensors if t['name'].endswith('ffn_down.weight')}

def order():
    """Neuron order per MLP, most important first: sqrt(mean squared input to ffn_down) x column norm of ffn_down (Wanda)."""
    p = R / 'order.json'
    if p.exists(): return {int(k): np.array(v) for k, v in json.load(open(p)).items()}
    hi = G._Hdr(IM); im = {}
    for t in hi.tensors:
        if t['name'].endswith('ffn_down.weight.in_sum2'):
            im[int(t['name'].split('.')[1])] = np.frombuffer(hi.m, np.float32, t['n_elements'], hi.data_start + t['offset'])
    hs = G._Hdr(SRC); out = {}
    for t in hs.tensors:
        if t['name'].endswith('ffn_down.weight'):
            il = int(t['name'].split('.')[1]); v = np.frombuffer(hs.m, np.uint16, t['n_elements'], hs.data_start + t['offset']).reshape(t['dims'][1], t['dims'][0])
            w = (v.astype(np.uint32) << 16).view(np.float32); col = np.sqrt((w.astype(np.float64) ** 2).sum(0))
            out[il] = np.argsort(-(col * np.sqrt(im[il].astype(np.float64))))
    json.dump({k: v.tolist() for k, v in out.items()}, open(p, 'w')); return out

def kv_with_widths(h, widths):
    m, p = h.m, h.kv_start
    for _ in range(h.n_kv):
        p0 = p; (kl,) = struct.unpack_from('<Q', m, p); p += 8; k = m[p:p + kl].decode(); p += kl
        (vt,) = struct.unpack_from('<I', m, p); p += 4; e = h._skip(p, vt)
        if k == KEYFF:
            return bytes(m[h.kv_start:p0]) + bytes(m[p0:p - 4]) + struct.pack('<I', 9) + struct.pack('<IQ', 4, len(widths)) + struct.pack(f'<{len(widths)}I', *widths) + bytes(m[e:h.kv_end])
        p = e
    raise KeyError(KEYFF)

def write_gguf(h, kv, tens, path):
    A = h.alignment; ti = bytearray(); off = 0; offs = []
    for n, d, tt, raw in tens:
        nb = n.encode(); ti += struct.pack('<Q', len(nb)) + nb + struct.pack('<I', len(d)) + struct.pack(f'<{len(d)}Q', *d) + struct.pack('<IQ', tt, off)
        offs.append(off); off = (off + len(raw) + A - 1) // A * A
    with open(path + '.part', 'wb') as f:
        f.write(b'GGUF' + struct.pack('<IQQ', h.version, len(tens), h.n_kv)); f.write(kv); f.write(ti)
        pos = f.tell(); f.write(b'\0' * ((pos + A - 1) // A * A - pos)); base = f.tell()
        for (n, d, tt, raw), o in zip(tens, offs): f.write(b'\0' * (o - (f.tell() - base))); f.write(raw)
        pos = f.tell(); f.write(b'\0' * ((pos + A - 1) // A * A - pos))
    os.replace(path + '.part', path)

def write_var(widths, path, im_path=None):
    """src0 with MLP widths cut to the top-w neurons of order(); widths {il: w}. Optionally the matching imatrix subset."""
    h = G._Hdr(SRC); od = order(); n0 = nff0(); keep = {il: np.sort(od[il][:w]) for il, w in widths.items() if w < n0[il]}
    wl = [int(widths.get(il, n0[il])) for il in range(35)]; kv = kv_with_widths(h, wl); tens = []
    for t in h.tensors:
        n = t['name']; p = n.split('.'); s0 = h.data_start + t['offset']
        if p[0] == 'blk' and int(p[1]) in keep and p[2] in ('ffn_gate', 'ffn_up', 'ffn_down'):
            il = int(p[1]); v = np.frombuffer(h.m, np.uint16, t['n_elements'], s0).reshape(t['dims'][1], t['dims'][0])
            if p[2] == 'ffn_down': blob, d = np.ascontiguousarray(v[:, keep[il]]), [len(keep[il]), t['dims'][1]]
            else: blob, d = np.ascontiguousarray(v[keep[il], :]), [t['dims'][0], len(keep[il])]
            tens.append((n, d, t['ttype'], blob.tobytes()))
        else: tens.append((n, list(t['dims']), t['ttype'], bytes(h.m[s0:s0 + t['n_bytes']])))
    write_gguf(h, kv, tens, path)
    if im_path:
        hi = G._Hdr(IM); it = []
        for t in hi.tensors:
            s0 = hi.data_start + t['offset']; raw = bytes(hi.m[s0:s0 + t['n_bytes']]); d = list(t['dims']); p = t['name'].split('.')
            if p[0] == 'blk' and int(p[1]) in keep and p[2] == 'ffn_down' and p[-1] in ('in_sum2', 'in_sum'):
                a = np.frombuffer(raw, np.float32).reshape(-1, d[0])[:, keep[int(p[1])]]; raw, d = np.ascontiguousarray(a).tobytes(), [a.shape[1]] + list(d[1:])
            it.append((t['name'], d, t['ttype'], raw))
        write_gguf(hi, bytes(hi.m[hi.kv_start:hi.kv_end]), it, im_path)
    return keep

def score(model, trj, gpu, n):
    out = f'/dev/shm/s-{os.getpid()}.tsv'; env = dict(os.environ, LD_LIBRARY_PATH=str(R / 'llama.cpp/build/bin'), CUDA_VISIBLE_DEVICES=str(gpu))
    r = subprocess.run([str(R / 'bin/logodds'), '-m', model, '--traj', trj, '--pair', '23214,1389', '--out', out, '--shard', '0/1'], env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-1500:]
    v = np.full(n, np.nan)
    for l in open(out): a, _, x = l.split('\t'); v[int(a)] = float(x)
    os.unlink(out); assert np.isfinite(v).all(); return v

def probes():
    P = []
    for g, info in GR.items():
        if g in ('token_embd',): ts = ['Q4_K', 'Q2_K']
        elif g == 'per_layer_token_embd': ts = ['Q4_K', 'Q3_K', 'Q2_K']
        else: ts = list(TIERS_BLOCK)
        if info['droppable']: ts.append('drop')
        P += [{'tag': f'{g}={t}', 'tiers': {g: t}} for t in ts]
    n0 = nff0()
    for il in range(35):
        for f in (0.75, 0.5, 0.25, 0.0625):
            P.append({'tag': f'w{il}={int(n0[il] * f)}', 'widths': {il: int(n0[il] * f)}})
    return P

def work(gpu):
    os.makedirs(R / 'sc/p', exist_ok=True); n = struct.unpack_from('<I', open(R / 'gd/both.trj', 'rb').read(8), 4)[0]
    for it in json.load(open(R / 'probes.json')):
        out = R / 'sc/p' / f"{it['tag']}.npy"
        if out.exists(): continue
        try: os.close(os.open(str(out) + '.lock', os.O_CREAT | os.O_EXCL))
        except FileExistsError: continue
        t0 = time.time(); f = f"/dev/shm/p-{gpu}.gguf"
        try:
            if 'widths' in it: write_var({int(k): v for k, v in it['widths'].items()}, f)
            else:
                B = Builder(TC, SRC, GR, IM, Path(f'/dev/shm/w-{gpu}'), log=f'/root/g/logs/build-{gpu}.log', threads=14)
                r = B.build(mask(it['tiers']), f); assert not r.fallbacks, r.fallbacks; B.cleanup()
            b = os.path.getsize(f); v = score(f, str(R / 'gd/both.trj'), gpu, n); np.save(out, v)
            json.dump({'bytes': b}, open(str(out)[:-4] + '.json', 'w')); print('probe', it['tag'], b, round(time.time() - t0), 's', flush=True)
        except Exception as e:
            print('FAIL', it['tag'], repr(e)[:400], flush=True); os.unlink(str(out) + '.lock')
        finally:
            if os.path.exists(f): os.unlink(f)

if __name__ == '__main__':
    c = sys.argv[1]
    if c == 'solos': P = probes(); json.dump(P, open(R / 'probes.json', 'w')); print(len(P), 'probes')
    elif c == 'order': o = order(); print('order', len(o))
    elif c == 'work': work(int(sys.argv[2]))
