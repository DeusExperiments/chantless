"""Prompt strings (chat template applied, document cut at 4,096 Hugging Face tokens) for each set, as tokdocs input
(u32 n, then u32 len + UTF-8 bytes), in the same order as gd/<set>.trj."""
import json, struct, sys, os
from multiprocessing import Pool
import pyarrow.parquet as pq
from transformers import AutoTokenizer
G = '/root/g'; SYSTEM = "Given a document, answer with one word: human if the document was human-written, ai if it was AI-generated."
tok = None
def init():
    global tok; tok = AutoTokenizer.from_pretrained(f'{G}/det')
def prompt(t):
    e = tok(t, add_special_tokens=False, return_offsets_mapping=True)
    if len(e['input_ids']) > 4096: t = t[:e['offset_mapping'][4095][1]]
    return tok.apply_chat_template([{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': t}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
text = {}
for split, pre in (('validation', 'wo-val'), ('calibration', 'wo-cal'), ('test', 'wo-test')):
    for r in pq.read_table(f'{G}/raw/WildOutlines/data/{split}-00000-of-00001.parquet', columns=['id', 'text']).to_pylist(): text[f'{pre}:{r["id"]}'] = r['text']
seen = set()
for r in pq.read_table(f'{G}/raw/beemo/data/train-00000-of-00001.parquet').to_pylist():
    for rid, t in ((f"beemo-h-{r['prompt_id']}", r['human_output']), (f"beemo-{r['id']}-g", r['model_output']), (f"beemo-{r['id']}-e", r['human_edits'])):
        if t and t.strip() and t not in seen: seen.add(t); text[f'beemo:{rid}'] = t
sets = sys.argv[1:] or ['task', 'bee', 'cal', 'select', 'exam']
init()
with Pool(120, initializer=init) as P:
    for s in sets:
        ids = [r['task'] for r in json.load(open(f'{G}/gd/{s}.json'))]
        ps = P.map(prompt, [text[i] for i in ids], chunksize=64)
        with open(f'{G}/gd/{s}.docs.bin', 'wb') as f:
            f.write(struct.pack('<I', len(ps)))
            for p in ps: b = p.encode('utf-8'); f.write(struct.pack('<I', len(b)) + b)
        print('prompts', s, len(ps), flush=True)
