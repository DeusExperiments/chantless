"""Pilot analysis v2: tolerant of any trajs layout; adds tokens and wall time per episode."""
import json, glob, os, collections, statistics as st, sys, re, datetime as dt
root = sys.argv[1]
rows = []
for tp in sorted(glob.glob(root + '/*/rollout_*/*.traj.json')):
    d = os.path.dirname(tp); t = json.load(open(tp)); info = t.get('info', {})
    trajs = t.get('trajs', {}) or {}
    msgs = [m for k in trajs for m in (trajs[k].get('messages') or [])]
    calls = [json.dumps(c.get('function', c), sort_keys=True)
             for m in msgs if m.get('role') == 'assistant' for c in (m.get('tool_calls') or [])]
    run = mx = 1
    for a, b in zip(calls, calls[1:]):
        run = run + 1 if a == b else 1; mx = max(mx, run)
    try: rei = json.load(open(glob.glob(d + '/*reward_extra_info.json')[0]))
    except Exception: rei = {}
    errs = sum(sum(v) for v in (rei.get('tool_call_errors') or {}).values())
    ts = re.findall(r'^(\d\d-\d\d \d\d:\d\d:\d\d)', open(d + '/instance.log').read(), re.M)
    wall = (dt.datetime.strptime(ts[-1], '%m-%d %H:%M:%S') - dt.datetime.strptime(ts[0], '%m-%d %H:%M:%S')).seconds / 60 if len(ts) > 1 else None
    ms = info.get('model_stats') or {}
    rows.append(dict(task=d.split('/')[-2], ro=d.split('/')[-1], traj_keys=list(trajs),
                     reward=info.get('reward'), exit=info.get('exit_status'), n_calls=len(calls),
                     max_repeat=mx, tool_errs=errs, wall_min=wall, model_stats=ms,
                     submission_empty=not (info.get('submission') or '').strip()))
json.dump(rows, open(root + '/summary_v2.json', 'w'), default=str, indent=1)
by = collections.defaultdict(list)
for r in rows: by[r['task']].append(1 if (r['reward'] or 0) >= 1 else 0)
n = len(rows); p = sum(sum(v) for v in by.values()) / max(1, n)
print(f'episodes {n}  tasks {len(by)}  pass rate {p:.3f}')
print('passes per task (of 2):', dict(sorted(collections.Counter(sum(v) for v in by.values()).items())))
print('exit status:', dict(collections.Counter(r['exit'] for r in rows)))
w = [r['wall_min'] for r in rows if r['wall_min'] is not None]
c = [r['n_calls'] for r in rows]
print(f'wall min: median {st.median(w):.1f}  max {max(w):.1f} | tool calls: median {st.median(c)}  max {max(c)}')
print('doom proxy (same call >=3 in a row):', sum(r['max_repeat'] >= 3 for r in rows),
      '| episodes with tool-call errors:', sum(r['tool_errs'] > 0 for r in rows),
      '| empty submissions:', sum(r['submission_empty'] for r in rows))
print('traj layouts:', dict(collections.Counter(tuple(r['traj_keys']) for r in rows)))
print('model_stats example:', json.dumps(rows[0]['model_stats'])[:300])
for t_, v in sorted(by.items()): print(' ', t_[-6:], v)
