"""Learned Apprentice on the Gemma 4 E2B 10% mask: the Qwen recipe (CHANTLESS-REPLICATION.md C5-C9) on Gemma.
  check            student vs llama.cpp on lesson, fresh and 200 select documents
  train o0|i1      o0 = value mask (KL + plane, last two grid blocks); i1 = switch mask (neuron gates)
  combo            value levels, then wherever the switch mask changed a level, the switch wins
  judge NAME...    real file (llama.cpp): own cutoff from 20,000 calibration humans, select side-1 gate vs the source, fresh 25
Same seeds, replay draws, strata, tails, terms, steps and rates as the Qwen run; fold 0 (trains on odd articles).
Each step scores its feedback, pair and replay documents in one packed forward pass (train_budget): lean.py keeps one
document-boundary buffer, so four passes before one backward would overwrite what the first three need. Same loss, same gradient."""
import sys, os, json, random, struct, subprocess, time
import numpy as np
sys.path.insert(0, '/root/g'); sys.path.insert(0, '/root/g/mvp'); sys.path.insert(0, '/root/g/ap')
import gstudent as GS
from chantless import lattice, learn, stats
G = '/root/g'; AP = f'{G}/ap'; MASK = f'{G}/mask10.gguf'; NCAL = 20000
keep = np.load(f'{G}/rc/keep-80000.npy'); PAIR = (int(np.searchsorted(keep, 23214)), int(np.searchsorted(keep, 1389)))
CFG = json.load(open(f'{G}/det/config.json'))['text_config']
SPECS = {'o0': {'fold': 0, 'hold_kind': 'kl', 'replay_weight': 8.0, 'project': True, 'lr': 2e-5, 'steps': 100, 'stratify': True, 'tails': True, 'guard_weight': 0.0, 'train_budget': 10**9},
         'i1': {'fold': 0, 'method': 'gates', 'gate_lr': 0.03, 'steps': 100, 'stratify': True, 'tails': True, 'guard_weight': 1.0, 'feedback_batch': 4, 'replay_batch': 8, 'score_budget': 16000, 'train_budget': 10**9}}
log = lambda *a: print(time.strftime('%H:%M:%S'), *a, flush=True)


def real(model, name):
    """llama.cpp scores of a GGUF on ap/<name>.trj, cached by file hash."""
    if model == MASK and os.path.exists(f'{AP}/mask10-{name}.npy'): return np.load(f'{AP}/mask10-{name}.npy')
    out = f'{AP}/sc-{learn.sha256(model)[:16]}-{name}.npy'
    if not os.path.exists(out):
        subprocess.run([f'{G}/.venv/bin/python', f'{G}/gscore.py', model, f'{AP}/{name}.trj', out, '0', f'{PAIR[0]},{PAIR[1]}'], check=True, stdout=subprocess.DEVNULL)
    return np.load(out)


def replay_set(rows, base_pool):
    """run() of learn.py, lines 669-687, with this run's rows: same RNG, same draws, same order."""
    rng = random.Random(20261031)
    side0 = [NCAL + i for i, r in enumerate(rows) if r['side'] == 0]
    picks = rng.sample(range(NCAL), 1000) + rng.sample(side0, 1000)
    kind = lambda i: ('beemo' if rows[i - NCAL]['task'].startswith('beemo') else 'wo', 'human' if rows[i - NCAL]['label'] != 'ai' else 'ai')
    beemo = [i for i in side0 if kind(i)[0] == 'beemo']; wo = [i for i in side0 if kind(i)[0] == 'wo']
    picks = rng.sample(range(NCAL), 1000) + beemo + rng.sample(wo, 1000)
    strata = ['human-cal'] * 1000 + [f'{kind(i)[1]}-{kind(i)[0]}' for i in picks[1000:]]
    tail = np.argsort(base_pool[:NCAL])[:500].tolist()
    ai = [i for i in side0 if rows[i - NCAL]['label'] == 'ai']
    near = sorted(ai, key=lambda i: -base_pool[i])[:300]
    return picks + tail + near, strata + ['human-tail'] * len(tail) + ['ai-near'] * len(near)


def inputs():
    rows = json.load(open(f'{AP}/select.json')); fb = json.load(open(f'{G}/feedback.json'))
    pool = learn.read_trj(f'{AP}/cal.trj') + learn.read_trj(f'{AP}/select.trj'); assert len(pool) == NCAL + len(rows)
    base_pool = np.concatenate([real(MASK, 'cal'), real(MASK, 'select')])
    cutoff = float(np.sort(base_pool[:NCAL])[200])
    return rows, fb, pool, base_pool, cutoff


def check():
    import torch
    GS.install(); art = lattice.Artifact(MASK); m, _ = GS.build(art, CFG, PAIR, []); art.close()
    out = {}
    for name, idx in (('lesson', None), ('fresh', None), ('select', list(range(0, 14000, 70)))):
        docs = learn.read_trj(f'{AP}/{name}.trj'); r = real(MASK, name)
        if idx is not None: docs, r = [docs[i] for i in idx], r[idx]
        s = learn.score_all(m, docs, PAIR, 'cuda', True, 16000)
        d = np.abs(s - r); out[name] = {'n': len(docs), 'max': float(d.max()), 'mean': float(d.mean()), 'p95': float(np.quantile(d, .95)), 'spearman': float(__import__('scipy.stats').stats.spearmanr(s, r)[0])}
        log('CHECK', name, json.dumps(out[name]))
    json.dump(out, open(f'{AP}/check.json', 'w'), indent=1)


def train(tag):
    import torch
    GS.install(); c = SPECS[tag]
    rows, fb, pool, base_pool, cutoff = inputs()
    picks, strata = replay_set(rows, base_pool); replay = [pool[i] for i in picks]
    fdocs = learn.read_trj(f'{AP}/lesson.trj'); assert len(fdocs) == 80
    feedback = [dict(fb[i], tokens=fdocs[fb[i]['index']]) for i in range(len(fb)) if fb[i].get('fold') != c['fold']]
    log(tag, 'cutoff', cutoff, 'replay', len(replay), {s: strata.count(s) for s in sorted(set(strata))}, 'feedback', len(feedback))
    out = f'{AP}/{tag}.gguf'; art = lattice.Artifact(MASK)
    if c.get('method') == 'gates':
        m, _ = GS.build(art, CFG, PAIR, []); m.ckpt = True
        layers = [il for il in range(CFG['num_hidden_layers']) if art.tensors.get(f'blk.{il}.ffn_down.weight', {}).get('type') in lattice.TOP]
        theta = {il: torch.nn.Parameter(torch.ones(art.shape(f'blk.{il}.ffn_down.weight')[1], device='cuda')) for il in layers}
        def hook(il):
            def f(mod, args):
                s = torch.sigmoid(theta[il]); g = (theta[il] > 0).float() + s - s.detach(); return (args[0] * g.to(args[0].dtype),)
            return f
        hooks = [m.layers[il].down_proj.register_forward_pre_hook(hook(il)) for il in layers]
        log('gates on layers', layers, 'neurons', sum(t.numel() for t in theta.values()))
        hist = learn._train(m, list(theta.values()), feedback, replay, PAIR, cutoff, c, log, 'cuda', True, strata, c['gate_lr'])
        for h in hooks: h.remove()
        levels, off, bits = {}, {}, 0
        for il in layers:
            name, shape = f'blk.{il}.ffn_down.weight', art.shape(f'blk.{il}.ffn_down.weight')
            dead = np.flatnonzero(theta[il].detach().cpu().numpy() <= 0); off[il], bits = dead.tolist(), bits + shape[1]
            if len(dead):
                g = art.grid(name); lv = g.levels.reshape(shape).copy()
                zero = np.clip(np.round(-g.lo / np.maximum(g.step, 1e-30)), 0, g.top).astype(np.uint8).reshape(shape)
                lv[:, dead] = zero[:, dead]; levels[name] = lv.reshape(-1)
        lattice.write_levels(MASK, out, levels)
        n = sum(map(len, off.values()))
        rec = {'method': 'gates', **c, 'neurons': bits, 'off': n, 'fraction_off': n / bits, 'off_by_layer': {str(k): len(v) for k, v in off.items()}}
        json.dump({str(k): v for k, v in off.items()}, open(out + '.switches.json', 'w'))
    else:
        names = learn.train_names(art, {'last_blocks': 2})
        m, grids = GS.build(art, CFG, PAIR, names)
        log('value tensors', names, 'weights', sum(g[0].numel() for g in grids.values()))
        hist = learn._train(m, [w for w, _, _ in grids.values()], feedback, replay, PAIR, cutoff, c, log, 'cuda', True, strata, c['lr'])
        levels, moved = {}, {}
        for name, (w, fq, start) in grids.items():
            lv = fq.levels(w.detach()).reshape(-1).to('cpu', torch.uint8).numpy(); levels[name], moved[name] = lv, int((lv != start.numpy()).sum())
        lattice.write_levels(MASK, out, levels)
        nw = sum(g[0].numel() for g in grids.values())
        rec = {'method': 'grid', **c, 'tensors': names, 'weights': nw, 'levels_moved': sum(moved.values()), 'fraction_moved': sum(moved.values()) / nw}
    art.close()
    rec.update(loss_first=hist[0], loss_last=hist[-1], n_feedback=len(feedback), n_replay=len(replay), cutoff=cutoff, base_sha256=learn.sha256(MASK), out_sha256=learn.sha256(out), bytes=os.path.getsize(out))
    json.dump(rec, open(out + '.lineage.json', 'w'), indent=1); log('TRAINED', tag, json.dumps({k: v for k, v in rec.items() if k not in ('tensors', 'off_by_layer')}))


def combo():
    B, V, S = lattice.Artifact(MASK), lattice.Artifact(f'{AP}/o0.gguf'), lattice.Artifact(f'{AP}/i1.gguf')
    levels, both = {}, 0
    for name, t in B.tensors.items():
        if t['type'] not in lattice.TOP: continue
        l0, lv, ls = B.grid(name).levels, V.grid(name).levels, S.grid(name).levels
        if (lv != l0).any() or (ls != l0).any():
            sw = ls != l0; both += int((sw & (lv != l0)).sum()); out = lv.copy(); out[sw] = ls[sw]; levels[name] = out
    lattice.write_levels(MASK, f'{AP}/combo.gguf', levels)
    log('COMBO', len(levels), 'tensors, overlapping weights', both)


def judge(names):
    rows = json.load(open(f'{AP}/select.json')); fr = json.load(open(f'{G}/fresh.json')); fb = json.load(open(f'{G}/feedback.json'))
    src_cal, src_sel = np.load(f'{G}/rc/src_cal.npy'), np.load(f'{G}/rc/src_select.npy'); scut = float(np.sort(src_cal)[200])
    y = np.array([r['label'] == 'ai' for r in rows]); bee = np.array([r['task'].startswith('beemo') for r in rows]); side1 = np.array([r['side'] == 1 for r in rows])
    src_ok = (src_sel < scut) == y
    res = {}
    for nm in names:
        path = MASK if nm == 'base' else f'{AP}/{nm}.gguf'
        cal, sel, fre, les = real(path, 'cal'), real(path, 'select'), real(path, 'fresh'), real(path, 'lesson')
        cut = float(np.sort(cal)[200]); ok = (sel < cut) == y
        R = []
        for i in np.flatnonzero(side1):
            R += [{'task': rows[i]['task'], 'family': rows[i]['family'], 'side': 'cand', 'success': bool(ok[i])}, {'task': rows[i]['task'], 'family': rows[i]['family'], 'side': 'src', 'success': bool(src_ok[i])}]
        b = stats.delta_and_bounds(R, .05 / 4)
        rate = lambda ai, bm: float((sel[side1 & (y == ai) & (bee == bm)] < cut).mean())
        k = lambda kind: [i for i, r in enumerate(fr) if r['kind'] == kind]
        flag = lambda kind: int((fre[k(kind)] < cut).sum())
        wr = {w: f"{int((fre[[i for i in k('portion-synth') if fr[i]['writer'] == w]] < cut).sum())}/{len([i for i in k('portion-synth') if fr[i]['writer'] == w])}" for w in ('opus', 'sonnet', 'haiku', 'fable')}
        held = [r for r in fb if r['fold'] == 0]; hv = {r['index']: les[r['index']] for r in held}
        res[nm] = {'bytes': os.path.getsize(path), 'sha256': learn.sha256(path), 'cutoff': cut, 'gate_n_side1': int(side1.sum()), 'delta': b['delta'], 'L': b['L'], 'pass': bool(b['L'] > -0.02),
                   'caught_wo': rate(True, False), 'humans_flagged_wo': rate(False, False), 'caught_beemo': rate(True, True), 'humans_flagged_beemo': rate(False, True),
                   'fresh_synth': flag('portion-synth'), 'fresh_mixed': flag('article-mixed'), 'fresh_human_portion': flag('portion-human'), 'fresh_human_article': flag('article-human'), 'by_writer': wr,
                   'held_ai_flagged': int(sum(hv[r['index']] < cut for r in held if r['label'] == 'ai')), 'held_human_flagged': int(sum(hv[r['index']] < cut for r in held if r['label'] == 'human'))}
        log('JUDGE', nm, json.dumps(res[nm]))
    prev = json.load(open(f'{AP}/judge.json')) if os.path.exists(f'{AP}/judge.json') else {}
    prev.update(res); json.dump(prev, open(f'{AP}/judge.json', 'w'), indent=1)


if __name__ == '__main__':
    cmd = sys.argv[1]
    if cmd == 'check': check()
    elif cmd == 'train': train(sys.argv[2])
    elif cmd == 'combo': combo()
    elif cmd == 'judge': judge(sys.argv[2:])
