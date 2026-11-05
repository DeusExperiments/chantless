#!/usr/bin/env python3
"""Shrink a Qwen3.5 GGUF to a kept token set, at any quant tier, with no requantisation.

Rows of token_embd.weight are independent at every tier (each row is quantised on its own), so the kept rows
are copied byte for byte. tokens, token_type and merges are subset in order, which preserves merge ranks; ids
in tokenizer.ggml.*_token_id are renumbered. `keep` must be sorted and closed under merge parents, must hold
every special token and all 256 byte tokens, and new id = position in `keep`.

--head a,b        also write a short output.weight holding only rows a,b (old ids), in that order. Needs the
                  Chantless tiny-head patch; logits then have two columns and logodds takes --pair 0,1.
--head-from F     take the head rows from GGUF F (for example the BF16 source) instead of the file's own table.
--fallback-row C  append one row, the count-weighted mean of the dropped rows, as control token <|unk_row|>.
                  BF16 sources only. Counts come from C (.npy, one count per old id).
"""
import argparse, json, os, struct, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, '/root/g/mvp')
from chantless import gguf_ops

STR, ARR = 8, 9

def read_kvs(h):
    m = h.m; p = h.kv_start; out = []
    for _ in range(h.n_kv):
        (kl,) = struct.unpack_from('<Q', m, p); p += 8
        key = m[p:p + kl].decode(); p += kl
        (vt,) = struct.unpack_from('<I', m, p); p += 4
        s = p; p = h._skip(p, vt)
        out.append((key, vt, bytes(m[s:p])))
    return out

def str_array(raw):
    et, n = struct.unpack_from('<IQ', raw, 0); assert et == STR
    p = 12; out = []
    for _ in range(n):
        (sl,) = struct.unpack_from('<Q', raw, p); p += 8
        out.append(raw[p:p + sl]); p += sl
    return out

def pack_str_array(items):
    b = bytearray(struct.pack('<IQ', STR, len(items)))
    for s in items:
        b += struct.pack('<Q', len(s)); b += s
    return bytes(b)

def bf16_mean(rows, w):
    """Weighted mean of BF16 rows given as uint16 patterns; returns float32."""
    x = (rows.astype(np.uint32) << 16).view(np.float32).astype(np.float64)
    return ((x * w[:, None]).sum(0) / w.sum()).astype(np.float32)

def _bf(v):
    # round to nearest with ties to even in the upper 16 bits of the float32 pattern
    a = v.view(np.uint32).astype(np.uint64)
    lo = a & 0xFFFF; hi = a >> 16
    hi = hi + ((lo > 0x8000) | ((lo == 0x8000) & ((hi & 1) == 1)))
    return hi.astype(np.uint16).tobytes()

def trim(src, dst, keep, head=None, head_from=None, fallback=None):
    h = gguf_ops._Hdr(src)
    keep = np.asarray(keep, np.int64); assert (np.diff(keep) > 0).all()
    kvs = read_kvs(h)
    kvd = {k: (vt, raw) for k, vt, raw in kvs}
    V = struct.unpack_from('<Q', kvd['tokenizer.ggml.tokens'][1], 4)[0]
    new_id = np.full(V, -1, np.int64); new_id[keep] = np.arange(len(keep))

    tokens = str_array(kvd['tokenizer.ggml.tokens'][1])
    ttype = np.frombuffer(kvd['tokenizer.ggml.token_type'][1], '<i4', V, 12)
    merges = str_array(kvd['tokenizer.ggml.merges'][1])
    ks = {tokens[i] for i in keep.tolist()}
    new_merges = []
    for r in merges:
        k = r.find(b' ', 1); a, b = r[:k], r[k + 1:]
        if a in ks and b in ks and (a + b) in ks:
            new_merges.append(r)
    new_tokens = [tokens[i] for i in keep.tolist()]; new_types = ttype[keep]
    if fallback is not None:
        new_tokens.append(b'<|unk_row|>'); new_types = np.append(new_types, 3)

    out_kvs = []
    for k, vt, raw in kvs:
        if k == 'tokenizer.ggml.tokens': raw = pack_str_array(new_tokens)
        elif k == 'tokenizer.ggml.token_type': raw = struct.pack('<IQ', 5, len(new_types)) + np.asarray(new_types, '<i4').tobytes()
        elif k == 'tokenizer.ggml.merges': raw = pack_str_array(new_merges)
        elif k == 'tokenizer.ggml.scores': raw = struct.pack('<IQ', 6, len(keep)) + np.frombuffer(raw, '<f4', V, 12)[keep].tobytes()
        elif k.startswith('tokenizer.ggml.') and k.endswith('_token_id'):
            (o,) = struct.unpack('<I', raw); n = int(new_id[o])
            if n < 0: raise SystemExit(f'{k}={o} is not in the keep set')
            raw = struct.pack('<I', n)
        out_kvs.append((k, vt, raw))

    A = h.alignment; tensors = []; off = 0
    for t in h.tensors:
        d = dict(t)
        if t['name'] in ('token_embd.weight', 'per_layer_token_embd.weight'):
            row = t['n_bytes'] // t['dims'][1]
            base = h.data_start + t['offset']
            rows = np.frombuffer(h.m, np.uint8, t['n_bytes'], base).reshape(t['dims'][1], row)
            blob = rows[keep].tobytes()
            if fallback is not None:
                if t['type'] != 'BF16': raise SystemExit('--fallback-row works on a BF16 table only')
                dropped = np.setdiff1d(np.arange(V), keep)
                r16 = np.frombuffer(h.m, '<u2', t['n_bytes'] // 2, base).reshape(t['dims'][1], t['dims'][0])
                w = fallback[dropped].astype(np.float64)
                if w.sum() == 0: w = np.ones_like(w)
                sel = w > 0
                mean = bf16_mean(r16[dropped[sel]], w[sel])
                blob += _bf(mean)
            d['dims'] = [t['dims'][0], len(keep) + (fallback is not None)]
            d['n_bytes'] = len(blob); d['_blob'] = blob
        tensors.append(d)
    if head:
        hs = gguf_ops._Hdr(head_from) if head_from else h
        te = next(t for t in hs.tensors if t['name'] == 'token_embd.weight')
        row = te['n_bytes'] // te['dims'][1]
        base = hs.data_start + te['offset']
        blob = b''.join(bytes(hs.m[base + row * r: base + row * (r + 1)]) for r in head)
        tensors.append({'name': 'output.weight', 'dims': [te['dims'][0], len(head)], 'ttype': te['ttype'], 'n_bytes': len(blob), '_blob': blob})

    ti = bytearray()
    for t in tensors:
        nb = t['name'].encode()
        ti += struct.pack('<Q', len(nb)) + nb + struct.pack('<I', len(t['dims'])) + struct.pack(f"<{len(t['dims'])}Q", *t['dims'])
        ti += struct.pack('<IQ', t['ttype'], off)
        t['_new'] = off
        off = (off + t['n_bytes'] + A - 1) // A * A
    tmp = str(dst) + '.part'
    with open(tmp, 'wb') as f:
        f.write(b'GGUF' + struct.pack('<IQQ', h.version, len(tensors), len(out_kvs)))
        for k, vt, raw in out_kvs:
            kb = k.encode(); f.write(struct.pack('<Q', len(kb)) + kb + struct.pack('<I', vt) + raw)
        kv_end = f.tell()
        f.write(ti)
        pos = f.tell(); f.write(b'\0' * ((pos + A - 1) // A * A - pos))
        data = f.tell()
        for t in tensors:
            cur = f.tell() - data; f.write(b'\0' * (t['_new'] - cur))
            if '_blob' in t: f.write(t['_blob'])
            else:
                s = h.data_start + t['offset']; f.write(h.m[s:s + t['n_bytes']])
    os.replace(tmp, dst)
    return {'kept_tokens': len(new_tokens), 'kept_merges': len(new_merges), 'file_bytes': os.path.getsize(dst), 'src_bytes': os.path.getsize(src),
            'kv_bytes': kv_end - 24, 'src_kv_bytes': h.kv_end - h.kv_start, 'token_embd_bytes': next(t['n_bytes'] for t in tensors if t['name'] == 'token_embd.weight'),
            'src_token_embd_bytes': next(t['n_bytes'] for t in h.tensors if t['name'] == 'token_embd.weight')}

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('src'); ap.add_argument('dst'); ap.add_argument('--keep', required=True)
    ap.add_argument('--head'); ap.add_argument('--head-from'); ap.add_argument('--fallback-row')
    a = ap.parse_args()
    r = trim(a.src, a.dst, np.load(a.keep), [int(x) for x in a.head.split(',')] if a.head else None, a.head_from,
             np.load(a.fallback_row) if a.fallback_row else None)
    print(json.dumps(r))
