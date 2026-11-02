import json,glob,os,collections,statistics as st,sys
root=sys.argv[1]
rows=[]
for tp in sorted(glob.glob(root+'/*/rollout_*/*.traj.json')+glob.glob(root+'/*/*.traj.json')):
    d=os.path.dirname(tp); t=json.load(open(tp)); info=t['info']
    msgs=t['trajs']['main']['messages']
    calls=[json.dumps(c.get('function',c),sort_keys=True) for m in msgs if m.get('role')=='assistant' for c in (m.get('tool_calls') or [])]
    run=mx=1
    for a,b in zip(calls,calls[1:]):
        run=run+1 if a==b else 1; mx=max(mx,run)
    try: rei=json.load(open(glob.glob(d+'/*reward_extra_info.json')[0]))
    except Exception: rei={}
    errs=sum(sum(v) for v in (rei.get('tool_call_errors') or {}).values())
    ms=info.get('model_stats') or {}
    rows.append(dict(task=d.split('/')[-2] if 'rollout_' in d else d.split('/')[-1], ro=d.split('/')[-1],
        reward=info.get('reward'), exit=info.get('exit_status'), n_calls=len(calls), max_repeat=mx,
        tool_errs=errs, test_s=info.get('test_duration'), stats=ms))
json.dump(rows,open(root+'/summary.json','w'),default=str,indent=1)
by=collections.defaultdict(list)
for r in rows: by[r['task']].append(1.0 if (r['reward'] or 0)>=1 else 0.0)
print('episodes',len(rows),'tasks',len(by))
print('pass rate %.3f'%(sum(sum(v) for v in by.values())/max(1,len(rows))))
print('per-task:',collections.Counter(int(sum(v)) for v in by.values()))
print('exit:',collections.Counter(r['exit'] for r in rows))
print('calls median',st.median([r['n_calls'] for r in rows]) if rows else '-', 'max',max([r['n_calls'] for r in rows] or [0]))
print('doom(>=3 repeats):',sum(r['max_repeat']>=3 for r in rows),' tool_err eps:',sum(r['tool_errs']>0 for r in rows))
print('stats keys:', list(rows[0]['stats'])[:12] if rows else '')
