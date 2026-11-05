"""Build a Gemma candidate from a spec, trim its vocabulary, and gate it against the source.
Usage: gbuild.py SPEC.json [exam]
Gate: each model's own cutoff = sorted(scores of 20,000 calibration humans)[200]; on the documents the source gets right
at its own cutoff, family-clustered paired difference, L = delta - t * se at alpha 0.05/4, pass when L > -0.02."""
import sys, os, json, struct, time, subprocess, numpy as np
from pathlib import Path
sys.path.insert(0, '/root/g'); sys.argv_ = sys.argv; import gdrv as D
from chantless.builder import Builder
from chantless import stats
R = Path('/root/g'); spec = json.load(open(sys.argv[1])); tag = spec['tag']; K = spec['K']; d = R / 'cand' / tag; d.mkdir(parents=True, exist_ok=True)
keep = np.load(R / f'keep-{K}.npy'); H, A = int(np.searchsorted(keep, 23214)), int(np.searchsorted(keep, 1389))
GPUS = '0,1,2,3,4,5,6,7'; t0 = time.time()
final = d / 'final.gguf'
if not final.exists():
    widths = {int(k): v for k, v in spec.get('widths', {}).items()}
    if widths: D.write_var(widths, str(d / 'var.gguf'), str(d / 'im.gguf')); w_src, im = str(d / 'var.gguf'), str(d / 'im.gguf')
    else: w_src, im = D.SRC, D.IM
    B = Builder(D.TC, w_src, D.GR, im, d / 'work', log=str(d / 'build.log'), threads=96)
    r = B.build(D.mask(spec['tiers']), d / 'q.gguf'); assert not r.fallbacks, r.fallbacks; B.cleanup()
    print('pre-trim bytes', os.path.getsize(d / 'q.gguf'), flush=True)
    subprocess.run([str(R / '.venv/bin/python'), str(R / 'gtrim.py'), str(d / 'q.gguf'), str(final), '--keep', str(R / f'keep-{K}.npy')], check=True, stdout=subprocess.DEVNULL)
    for x in ('var.gguf', 'q.gguf'): (d / x).unlink(missing_ok=True)
print(tag, 'final bytes', os.path.getsize(final), 'built in', round(time.time() - t0), 's', flush=True)
def retok(s):
    out = R / 'gd' / f'{s}.{K}.trj'
    if not out.exists():
        env = dict(os.environ, LD_LIBRARY_PATH=str(R / 'llama.cpp/build/bin'))
        subprocess.run([str(R / 'bin/tokdocs'), str(final), str(R / 'gd' / f'{s}.docs.bin'), f'/dev/shm/{s}.{K}.tok', '64'], env=env, check=True, capture_output=True)
        b = open(f'/dev/shm/{s}.{K}.tok', 'rb').read(); n = struct.unpack_from('<I', b, 0)[0]; p = 4
        with open(str(out) + '.part', 'wb') as f:
            f.write(b'TRJ1' + struct.pack('<I', n))
            for _ in range(n):
                (l,) = struct.unpack_from('<I', b, p); p += 4; x = np.frombuffer(b, '<i4', l, p); p += 4 * l
                seq = np.append(x, H).astype('<i4'); f.write(struct.pack('<II', len(seq), 1) + seq.tobytes() + struct.pack('<I', l))
        os.replace(str(out) + '.part', out); os.unlink(f'/dev/shm/{s}.{K}.tok')
    return str(out)
def sc(s):
    out = d / f'{s}.npy'
    if not out.exists(): subprocess.run([str(R / '.venv/bin/python'), str(R / 'gscore.py'), str(final), retok(s), str(out), GPUS, f'{H},{A}'], check=True, stdout=subprocess.DEVNULL)
    return np.load(out)
def gate(setname):
    rows = json.load(open(R / 'gd' / f'{setname}.json')); y = np.array([r['label'] == 'ai' for r in rows]); bee = np.array([r['task'].startswith('beemo') for r in rows])
    src_cal = np.load(R / 'sc/src_cal.npy'); src = np.load(R / f'sc/src_{setname}.npy'); scut = float(np.sort(src_cal)[200]); src_ok = (src < scut) == y
    cal = sc('cal'); v = sc(setname); cut = float(np.sort(cal)[200]); ok = (v < cut) == y
    rel = src_ok; Rr = []
    for r, o in zip([r for r, k in zip(rows, rel) if k], ok[rel]):
        Rr += [{'task': r['task'], 'family': r['family'], 'side': 'cand', 'success': bool(o)}, {'task': r['task'], 'family': r['family'], 'side': 'src', 'success': True}]
    b = stats.delta_and_bounds(Rr, .05 / 4)
    def sub(mask):
        RR = [z for z, m in zip([Rr[i:i + 2] for i in range(0, len(Rr), 2)], rel.nonzero()[0]) if mask[m]]
        bb = stats.delta_and_bounds([z for p in RR for z in p], .05 / 4); return {'delta': bb['delta'], 'L': bb['L'], 'n': len(RR)}
    out = {'tag': tag, 'set': setname, 'bytes': os.path.getsize(final), 'K': K, 'cutoff': cut, 'src_cutoff': scut, 'n_reliable': int(rel.sum()),
           'delta': b['delta'], 'L': b['L'], 'U': b['U'], 'pass': b['L'] > -0.02, 'lost': int((~ok & rel).sum()),
           'wo': sub(~bee), 'beemo': sub(bee),
           'caught_wo': float((v[rel & y & ~bee] < cut).mean()), 'human_flagged_wo': float((v[rel & ~y & ~bee] < cut).mean()),
           'caught_beemo': float((v[rel & y & bee] < cut).mean()), 'human_flagged_beemo': float((v[rel & ~y & bee] < cut).mean()),
           'src_caught_wo_all': float((src[y & ~bee] < scut).mean()), 'cand_caught_wo_all': float((v[y & ~bee] < cut).mean()),
           'src_caught_beemo_all': float((src[y & bee] < scut).mean()), 'cand_caught_beemo_all': float((v[y & bee] < cut).mean())}
    json.dump(out, open(d / f'gate_{setname}.json', 'w'), indent=1); print('GATE', json.dumps(out), flush=True); return out
gate('select')
if len(sys.argv) > 2 and sys.argv[2] == 'exam': gate('exam')
print('DONE', round(time.time() - t0), 's', flush=True)
