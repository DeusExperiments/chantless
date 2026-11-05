"""Gemma token files for the apprentice: lessons (80), fresh (100), calibration (20,000), select; same prompt as gprompts.py,
tokenised by tokdocs against the 80k-trimmed mask so ids match the file. Then score the mask on all four as a check."""
import json, struct, os, subprocess, sys, numpy as np
from multiprocessing import Pool
import pyarrow.parquet as pq
from transformers import AutoTokenizer
sys.path.insert(0, '/root/g/mvp'); from chantless.learn import _side
G = '/root/g'; O = f'{G}/ap'; os.makedirs(O, exist_ok=True); MASK = f'{G}/mask10.gguf'
SYSTEM = "Given a document, answer with one word: human if the document was human-written, ai if it was AI-generated."
keep = np.load(f'{G}/rc/keep-80000.npy'); H, A = int(np.searchsorted(keep, 23214)), int(np.searchsorted(keep, 1389)); assert keep[H] == 23214 and keep[A] == 1389
tok = None
def init():
    global tok; tok = AutoTokenizer.from_pretrained(f'{G}/det')
def prompt(t):
    e = tok(t, add_special_tokens=False, return_offsets_mapping=True)
    if len(e['input_ids']) > 4096: t = t[:e['offset_mapping'][4095][1]]
    return tok.apply_chat_template([{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': t}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
def trj(name, texts, P):
    out = f'{O}/{name}.trj'
    if os.path.exists(out): return out
    ps = P.map(prompt, texts, chunksize=32)
    with open(f'/dev/shm/{name}.bin', 'wb') as f:
        f.write(struct.pack('<I', len(ps)))
        for p in ps: b = p.encode('utf-8'); f.write(struct.pack('<I', len(b)) + b)
    env = dict(os.environ, LD_LIBRARY_PATH=f'{G}/llama.cpp/build/bin:/usr/local/lib/python3.13/site-packages/nvidia/cu13/lib')
    subprocess.run([f'{G}/bin/tokdocs', MASK, f'/dev/shm/{name}.bin', f'/dev/shm/{name}.tok', '4'], env=env, check=True, capture_output=True)
    b = open(f'/dev/shm/{name}.tok', 'rb').read(); n = struct.unpack_from('<I', b, 0)[0]; p = 4; mx = 0
    with open(out + '.part', 'wb') as f:
        f.write(b'TRJ1' + struct.pack('<I', n))
        for _ in range(n):
            (l,) = struct.unpack_from('<I', b, p); p += 4; x = np.frombuffer(b, '<i4', l, p); p += 4 * l; mx = max(mx, l)
            seq = np.append(x, H).astype('<i4'); f.write(struct.pack('<II', len(seq), 1) + seq.tobytes() + struct.pack('<I', l))
    os.replace(out + '.part', out); os.unlink(f'/dev/shm/{name}.tok'); os.unlink(f'/dev/shm/{name}.bin'); print(name, n, 'docs, longest', mx, flush=True); return out
if __name__ == '__main__':
    T = json.load(open(f'{G}/texts.json'))
    text, meta = {}, {}
    for split, pre in (('validation', 'wo-val'), ('calibration', 'wo-cal'), ('test', 'wo-test')):
        for r in pq.read_table(f'{G}/raw/WildOutlines/data/{split}-00000-of-00001.parquet', columns=['id', 'text', 'label', 'format']).to_pylist():
            text[f'{pre}:{r["id"]}'] = r['text']; meta[f'{pre}:{r["id"]}'] = {'label': 'human' if str(r['label']).lower().startswith('human') else 'ai', 'format': r['format'], 'kind': r['label']}
    seen = set()
    for r in pq.read_table(f'{G}/raw/beemo/data/train-00000-of-00001.parquet').to_pylist():
        for rid, t, label, kind in ((f"beemo-h-{r['prompt_id']}", r['human_output'], 'human', 'human'), (f"beemo-{r['id']}-g", r['model_output'], 'ai', 'generated'), (f"beemo-{r['id']}-e", r['human_edits'], 'ai', 'expert-edited')):
            if t and t.strip() and t not in seen: seen.add(t); text[f'beemo:{rid}'] = t; meta[f'beemo:{rid}'] = {'label': label, 'format': r['category'], 'kind': kind, 'prompt_id': r['prompt_id']}
    pool = json.load(open(f'{G}/cls/pool.json')); select = sorted(pool['per_task']['select'])
    cal = [f'wo-cal:{r["id"]}' for r in json.load(open(f'{G}/data/cal.json'))]; assert len(cal) == 20000
    rows = [{'task': i, 'label': meta[i]['label'], 'family': f"select/{i.split(':')[0]}/{meta[i]['format']}/{meta[i]['kind']}", 'side': _side(i)} for i in select]
    json.dump(rows, open(f'{O}/select.json', 'w')); print('select rows', len(rows), 'side1', sum(r['side'] for r in rows), flush=True)
    with Pool(4, initializer=init) as P:
        for name, texts in (('lesson', T['lesson']), ('fresh', T['fresh']), ('cal', [text[i] for i in cal]), ('select', [text[i] for i in select])): trj(name, texts, P)
    env = dict(os.environ)
    for name in ('lesson', 'fresh', 'cal', 'select'):
        out = f'{O}/mask10-{name}.npy'
        if not os.path.exists(out): subprocess.run([f'{G}/.venv/bin/python', f'{G}/gscore.py', MASK, f'{O}/{name}.trj', out, '0', f'{H},{A}'], check=True, stdout=subprocess.DEVNULL)
    for name in ('cal', 'select'):
        v, r = np.load(f'{O}/mask10-{name}.npy'), np.load(f'{G}/rc/{name}.npy'); print('CHECK', name, 'max |diff| vs saved', float(np.abs(v - r).max()), flush=True)
    c = np.load(f'{O}/mask10-cal.npy'); cut = float(np.sort(c)[200]); fr = np.load(f'{O}/mask10-fresh.npy'); rows = json.load(open(f'{G}/fresh.json'))
    k = lambda kind: [i for i, r in enumerate(rows) if r['kind'] == kind]
    print('BASE cutoff', cut, {kind: int((fr[k(kind)] < cut).sum()) for kind in ('portion-synth', 'portion-human', 'article-mixed', 'article-human')}, flush=True)
    print('APREP_DONE', flush=True)
