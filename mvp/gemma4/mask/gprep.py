"""Gemma 4 E2B token files for the Chantless detector stores and gates, same document ids as the Qwen run.
Prompt exactly as prep_e2b.py: system instruction, document cut to its first 4,096 tokens, chat template, thinking off.
TRJ1 record: prompt ids + [human id] as a sacrificial last token, one target at len(prompt)."""
import json, struct, hashlib, os, sys
from multiprocessing import Pool
import pyarrow.parquet as pq
from transformers import AutoTokenizer
G = '/root/g'; M = f'{G}/det'; O = f'{G}/gd'; os.makedirs(O, exist_ok=True)
SYSTEM = "Given a document, answer with one word: human if the document was human-written, ai if it was AI-generated."
tok = None
def init():
    global tok; tok = AutoTokenizer.from_pretrained(M)
def enc(t):
    e = tok(t, add_special_tokens=False, return_offsets_mapping=True)
    if len(e['input_ids']) > 4096: t = t[:e['offset_mapping'][4095][1]]
    p = tok.apply_chat_template([{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': t}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    return tok.encode(p, add_special_tokens=False)
init(); H, A = [tok.encode(w, add_special_tokens=False) for w in ('human', 'ai')]; assert H == [23214] and A == [1389], (H, A)
cols = ['id', 'text', 'label', 'format']; R = f'{G}/raw'; text, meta = {}, {}
for split, pre in (('validation', 'wo-val'), ('calibration', 'wo-cal'), ('test', 'wo-test')):
    for r in pq.read_table(f'{R}/WildOutlines/data/{split}-00000-of-00001.parquet', columns=cols).to_pylist():
        text[f'{pre}:{r["id"]}'] = r['text']; meta[f'{pre}:{r["id"]}'] = {'label': 'human' if str(r['label']).lower().startswith('human') else 'ai', 'format': r['format'], 'kind': r['label']}
seen = set()
for r in pq.read_table(f'{R}/beemo/data/train-00000-of-00001.parquet').to_pylist():
    for rid, t, label, kind in ((f"beemo-h-{r['prompt_id']}", r['human_output'], 'human', 'human'), (f"beemo-{r['id']}-g", r['model_output'], 'ai', 'generated'), (f"beemo-{r['id']}-e", r['human_edits'], 'ai', 'expert-edited')):
        if t and t.strip() and t not in seen:
            seen.add(t); text[f'beemo:{rid}'] = t; meta[f'beemo:{rid}'] = {'label': label, 'format': r['category'], 'kind': kind}
pool = json.load(open(f'{G}/cls/pool.json'))
cal = [f'wo-cal:{r["id"]}' for r in json.load(open(f'{G}/data/cal.json'))]
assert len(cal) == 20000
select = sorted(pool['per_task']['select']); search = sorted(pool['per_task']['search'])
exam = []
for l in open(f'{G}/cls/exam-pool.jsonl'):
    r = json.loads(l); t = r.get('task') or r.get('task_id') or r.get('id')
    if t not in exam: exam.append(t)
parts = json.load(open(f'{G}/parts.json')); store = [i for p in parts for i in p['ids']]
bee_search = [i for i in pool['reliable']['search'] if i.startswith('beemo:')]
bee = [i for i in bee_search if meta[i]['label'] == 'human'] + [i for i in bee_search if meta[i]['label'] == 'ai']
sets = {'cal': cal, 'select': select, 'exam': exam, 'task': store, 'bee': bee}
need = sorted({i for v in sets.values() for i in v}); miss = [i for i in need if i not in text]
print({k: len(v) for k, v in sets.items()}, 'distinct', len(need), 'missing text', len(miss), miss[:3], flush=True)
with Pool(120, initializer=init) as P: ids = dict(zip(need, P.map(enc, [text[i] for i in need], chunksize=64)))
for name, order in sets.items():
    with open(f'{O}/{name}.trj', 'wb') as f:
        f.write(b'TRJ1' + struct.pack('<I', len(order)))
        for i in order:
            x = ids[i]; seq = x + H; f.write(struct.pack('<II', len(seq), 1) + struct.pack(f'<{len(seq)}i', *seq) + struct.pack('<I', len(x)))
    fam = lambda i, sp: f"{sp}/{i.split(':')[0]}/{meta[i]['format']}/{meta[i]['kind']}"
    json.dump([{'task': i, 'label': meta[i]['label'], 'family': fam(i, name)} for i in order], open(f'{O}/{name}.json', 'w'))
    print(name, len(order), 'docs', sum(len(ids[i]) + 1 for i in order), 'tokens', flush=True)
print('GPREP_DONE', flush=True)
