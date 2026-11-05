"""Score a GGUF on a TRJ file split across GPUs. Usage: gscore.py MODEL TRJ OUT.npy GPUS(comma) [PAIR]"""
import subprocess, sys, os, struct, numpy as np
model, trj, out, gpus = sys.argv[1:5]; pair = sys.argv[5] if len(sys.argv) > 5 else '23214,1389'
gpus = gpus.split(','); n = struct.unpack_from('<I', open(trj, 'rb').read(8), 4)[0]; P = len(gpus); tag = os.path.basename(out)
env = dict(os.environ, LD_LIBRARY_PATH='/root/g/llama.cpp/build/bin')
ps = [subprocess.Popen(['/root/g/bin/logodds', '-m', model, '--traj', trj, '--pair', pair, '--out', f'/dev/shm/{tag}.{i}.tsv', '--shard', f'{i}/{P}'],
                       stdout=subprocess.DEVNULL, stderr=open(f'/dev/shm/{tag}.{i}.err', 'w'), env=dict(env, CUDA_VISIBLE_DEVICES=g)) for i, g in enumerate(gpus)]
rc = [p.wait() for p in ps]; assert all(r == 0 for r in rc), (rc, open(f'/dev/shm/{tag}.0.err').read()[-1500:])
v = np.full(n, np.nan)
for i in range(P):
    for l in open(f'/dev/shm/{tag}.{i}.tsv'): a, _, x = l.split('\t'); v[int(a)] = float(x)
    os.unlink(f'/dev/shm/{tag}.{i}.tsv'); os.unlink(f'/dev/shm/{tag}.{i}.err')
assert np.isfinite(v).all(), int((~np.isfinite(v)).sum()); np.save(out, v); print('scored', out, n)
