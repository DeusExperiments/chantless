"""One-Budget Anneal: staged quantization-aware training with progressive structural pruning.

`setup` builds shared fixed inputs once and `sweep`/`pick` choose the learning rate. Each arm then walks its
schedule starting from the previous built GGUF with float latents behind its quantization grids. A `cut` or
`rung` stage masks out target widths/blocks during training, prunes them, and builds a new standalone GGUF;
`finish` updates quantization levels in place on the final byte size. Stages checkpoint locally and optionally
to remote storage for crash recovery. Built GGUFs are evaluated at their calibrated cutoff.

Calibration humans split by hash (`cal-train` for tails/replay/training cutoff; `cal-judge` for evaluation).
Multi-variant prompts stay on one side of the split, feedback examples train from side 0 only with a held-out
probe fold, and raw tokens/text under `common/` stay local.
"""
from __future__ import annotations

import gc
import glob
import hashlib
import json
import os
import queue
import random
import shutil
import struct
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from scipy.stats import rankdata, spearmanr
from torch import nn

from . import lattice, learn, shape, stats
from .env.detect import _beemo_rows, _loss_curve_rows, chat_prompt
from .mask import DROP
from .runtime import BatchScorer, Toolchain

CHUNK, FULL = 256, 3584


def _sha(s: str) -> int:
    return int(hashlib.sha256(s.encode()).hexdigest()[:12], 16)


def _dump(path: str | Path, v) -> None:
    tmp = f"{path}.part"
    Path(tmp).write_text(json.dumps(v))
    os.replace(tmp, path)


def groups_of(c: dict) -> dict:
    g = json.loads(Path(c["groups"]).read_text())
    return g.get("groups", g)


def panel_cfg(c: dict) -> dict:
    com = Path(c["root"]) / "common"
    return dict(n_calibration=c["n_calibration"], pool_trj=c["pool_trj"], panel_trj=str(com / "panel.trj"), panel_idx=str(com / "panel_idx.npy"),
                panel_humans=c["panel_humans"], pair=c["pair"], seed=c["seed"], side_by_prompt=True, out_dir=str(com), cache_dir=str(com / "cache"))


def wq(x: np.ndarray, w: np.ndarray, q: float) -> float:
    o = np.argsort(x)
    return float(x[o][min(np.searchsorted(np.cumsum(w[o]), q * w.sum()), len(x) - 1)])


def auc(s: np.ndarray, y: np.ndarray) -> float:
    """P(human outscores AI). `y` is True for AI."""
    r, a = rankdata(s), int(y.sum())
    return float((r[~y].sum() - (len(s) - a) * (len(s) - a + 1) / 2) / (a * (len(s) - a)))


def caught_at(s: np.ndarray, y: np.ndarray, rate: float) -> float:
    """AI caught when the cutoff flags `rate` of these humans: threshold-free, so a slide along the ROC earns nothing."""
    return float((s[y] < np.quantile(s[~y], rate)).mean())


def tokenise(c: dict, texts: list[str], gguf: str, path: str | Path) -> list[np.ndarray]:
    """Chat prompts through tokdocs at `gguf`'s vocabulary into a TRJ1 file, keeping head+tail above 4,300 tokens."""
    tmp = Path(f"{path}.bin")
    with open(tmp, "wb") as f:
        f.write(struct.pack("<I", len(texts)))
        for t in texts:
            b = chat_prompt(t).encode()
            f.write(struct.pack("<I", len(b)) + b)
    tok = f"{path}.tok"
    subprocess.run([c["tokdocs"], gguf, str(tmp), tok, str(c["threads"])], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    b, docs = open(tok, "rb").read(), []
    q = 4
    for _ in range(struct.unpack_from("<I", b)[0]):
        n = struct.unpack_from("<I", b, q)[0]
        x = np.frombuffer(b, "<i4", n, q + 4)
        q += 4 + 4 * n
        docs.append(np.concatenate([x[:4250], x[-12:]]) if n > 4300 else x)
    learn.write_trj(docs, path, c["pair"][0])
    tmp.unlink()
    os.unlink(tok)
    return docs


def setup(cfg: dict, log=print) -> None:
    """Build shared baseline artifacts, splits, calibration anchors, lesson, replay, probes, panel, and judge set."""
    c, tc = cfg["anneal"], Toolchain.from_config(cfg["toolchain"])
    com, pair, ncal, groups = Path(c["root"]) / "common", tuple(c["pair"]), c["n_calibration"], groups_of(c)
    if (com / "spec.json").exists():
        return
    (com / "cache").mkdir(parents=True, exist_ok=True)
    scorer, vocab, pc = BatchScorer(tc, c["setup_slots"]), np.load(c["vocab"]), panel_cfg(c)
    s80, u331, uf = str(com / "s80.gguf"), str(com / "u331.gguf"), str(com / "uf.gguf")
    real = [g for g in groups if not g.startswith("_")]
    sh = shape.Shaper(tc, c["src"], groups, c["imatrix"], vocab, Path(c["work"]) / "common", c["threads"], str(com / "build.log"))
    if not Path(s80).exists():
        shape.trim(c["src"], s80, vocab)
    if not (com / "structure.json").exists():
        keep, tiers = shape.structure_of(c["q4cut"], c["src"], groups, c["ggml_lib"])
        _dump(com / "structure.json", {"keep": {str(k): v.tolist() for k, v in keep.items()}, "tiers": tiers})
    tiers = json.loads((com / "structure.json").read_text())["tiers"]
    if not Path(u331).exists():
        sh.build({}, dict.fromkeys(real, "Q4_K"), u331)
    if not Path(uf).exists():
        sh.build({}, {g: tiers[g] if tiers[g].startswith("IQ") else "Q4_K" for g in real}, uf)
    log(f"setup: files {[round(os.path.getsize(p) / shape.MB, 2) for p in (s80, u331, uf)]} MB")

    beemo = _beemo_rows(c["beemo"])
    pid = {r["id"]: r["prompt_id"] for r in beemo}
    rows = json.loads(Path(c["select_rows"]).read_text())
    for r in rows:
        if r["task"].startswith("beemo"):
            r["prompt_id"] = pid[r["task"].split(":", 1)[1]]
    _dump(com / "select.json", rows)
    val = pq.read_table(c["wo_val"], columns=["id", "text", "label", "format"]).to_pylist()
    skip = _loss_curve_rows([r["label"] for r in val])
    uni = [dict(task=f"wo-val:{r['id']}", family=f"wo-val/{r['format']}/{r['label']}", label=r["label"], kind=r["label"], text=r["text"]) for i, r in enumerate(val) if i not in skip]
    bee = lambda r: dict(task=f"beemo:{r['id']}", family=f"beemo/{r['format']}/{r['kind']}", label=r["label"], kind=r["kind"], prompt_id=r["prompt_id"], text=r["text"])
    uni += [bee(r) for r in beemo if r["split"] == "search"]
    exam = [bee(r) for r in beemo if r["split"] == "exam"]
    udocs = tokenise(c, [r.pop("text") for r in uni], u331, com / "universe.trj")
    edocs = tokenise(c, [r.pop("text") for r in exam], u331, com / "exam.trj")
    _dump(com / "exam.json", exam)
    del val, beemo
    log(f"setup: universe {len(uni)} documents, exam {len(exam)}")

    t = time.time()
    S = {k: scorer.score_trj(s80, p, pair) for k, p in (("pool", c["pool_trj"]), ("universe", str(com / "universe.trj")), ("exam", str(com / "exam.trj")),
                                                         ("fresh", c["fresh_trj"]), ("feedback", c["feedback_trj"]))}
    np.savez(com / "s80.npz", **S)
    log(f"setup: S80 scored in {time.time() - t:.0f} s")
    order = sorted(range(ncal), key=lambda i: _sha(f"cal:{i}"))
    train, cj = sorted(order[:ncal // 2]), sorted(order[ncal // 2:])
    p, rng = S["pool"], random.Random(c["seed"])
    tail = np.array(train)[np.argsort(p[train])[:c["tail_n"]]].tolist()
    rest = sorted(set(train) - set(tail))
    nh = c["panel_humans"]
    learn.make_panel(dict(pc, panel_humans=nh), rows, tail + rng.sample(rest, nh - len(tail)))
    idx = np.load(pc["panel_idx"])
    w = np.where(np.isin(idx[:nh], tail), 1.0, len(rest) / (nh - len(tail)))
    np.save(com / "panel_w.npy", w)
    cut, med = wq(p[idx[:nh]], w, .01), wq(p[idx[:nh]], w, .5)
    tau = .5 / (med - cut)

    folded = lambda r: _sha("probe:" + (r.get("prompt_id") or r["task"])) % 100 < c["probe_pct"]
    wrong = lambda i: bool((S["universe"][i] < cut) != (uni[i]["label"] == "ai"))
    u0 = [i for i, r in enumerate(uni) if learn.row_side(r, pc) == 0 and not folded(r)]
    probe = [i for i, r in enumerate(uni) if learn.row_side(r, pc) == 0 and folded(r)]
    side1 = [i for i, r in enumerate(uni) if learn.row_side(r, pc) == 1]
    fb, fdocs = json.loads(Path(c["feedback_rows"]).read_text()), learn.read_trj(c["feedback_trj"])
    docs, ids, ls = [], {}, []

    def doc(key, tokens) -> int:
        if key not in ids:
            ids[key] = len(docs)
            docs.append(tokens)
        return ids[key]

    for r in fb:
        if r.get("fold") != c["held_fold"]:
            ls.append(dict(doc=doc(("f", r["index"]), fdocs[r["index"]]), label=r["label"], pair=f"infill-{r['pair']}", share="infill", unit=f"infill-{r['pair']}", group="infill"))
    prompts = {}
    for i in u0:
        if uni[i]["task"].startswith("beemo"):
            prompts.setdefault(uni[i]["prompt_id"], []).append(i)
        elif uni[i]["label"] == "human" and wrong(i):
            ls.append(dict(doc=doc(("u", i), udocs[i]), label="human", pair=None, share="wo-human", unit=uni[i]["task"], group="wo"))
    for pr, idx_p in sorted(prompts.items()):
        hu = [i for i in idx_p if uni[i]["label"] == "human"][:1]
        ai = [i for i in idx_p if uni[i]["label"] == "ai"]
        group = "beemo-" + "".join(sorted({uni[i]["kind"][0] for i in ai if wrong(i)}))
        for i in ai:
            share = f"beemo-{uni[i]['kind']}" if wrong(i) else None
            if not hu and not share:
                continue
            ls.append(dict(doc=doc(("u", i), udocs[i]), label="ai", pair=uni[i]["task"] if hu else None, share=share, unit=pr, group=group))
            if hu:
                ls.append(dict(doc=doc(("u", hu[0]), udocs[hu[0]]), label="human", pair=uni[i]["task"], share=None, unit=pr, group=group))
    units = sorted({(r["group"], r["unit"]) for r in ls}, key=lambda u: (u[0], _sha(f"twin:{c['seed']}:{u[1]}")))
    flipped = {u[1] for u in units[::2]}
    for r in ls:
        r["flip"] = r["unit"] in flipped
    learn.write_trj(docs, com / "lesson.trj", pair[0])
    _dump(com / "lesson.json", ls)
    count = {}
    for r in ls:
        count[r["share"]] = count.get(r["share"], 0) + 1
    log(f"setup: lesson {len(ls)} rows on {len(docs)} documents, shares {count}, {len(units)} twin units, {len(flipped)} flipped, S80 cut {cut:.3f} med {med:.3f} tau {tau:.4f}")

    pool = learn.read_trj(c["pool_trj"])
    side0 = [ncal + i for i, r in enumerate(rows) if learn.row_side(r, pc) == 0 and not folded(r)]
    who = lambda i: ("beemo" if rows[i - ncal]["task"].startswith("beemo") else "wo", "human" if rows[i - ncal]["verdict"] != "ai" else "ai")
    picks = rng.sample(train, c["replay_humans"]) + [i for i in side0 if who(i)[0] == "beemo"] + rng.sample([i for i in side0 if who(i)[0] == "wo"], c["replay_select"])
    strata = ["human-cal"] * c["replay_humans"] + [f"{who(i)[1]}-{who(i)[0]}" for i in picks[c["replay_humans"]:]]
    near = sorted((i for i in side0 if who(i)[1] == "ai"), key=lambda i: -p[i])[:c["near_n"]]
    picks, strata = picks + tail + near, strata + ["human-tail"] * len(tail) + ["ai-near"] * len(near)
    learn.write_trj([pool[i] for i in picks], com / "replay.trj", pair[0])
    _dump(com / "replay.json", dict(strata=strata, base=p[picks].tolist()))
    shock = rng.sample(range(len(docs)), min(100, len(docs)))
    sdocs = [docs[i] for i in shock] + [pool[i] for i in rng.sample(picks, 200)]
    learn.write_trj(sdocs, com / "shock.trj", pair[0])
    learn.write_trj([udocs[i] for i in probe], com / "probe.trj", pair[0])
    _dump(com / "probe.json", dict(ai=[uni[i]["label"] == "ai" for i in probe], s80=S["universe"][probe].tolist()))
    _dump(com / "side1.json", [dict(uni[i], s80=float(S["universe"][i])) for i in side1])

    parts, n, jd = {}, 0, []
    for k, v in (("calj", [pool[i] for i in cj]), ("exam", edocs), ("fresh", learn.read_trj(c["fresh_trj"])), ("side1", [udocs[i] for i in side1]), ("lesson", docs), ("feedback", fdocs)):
        parts[k] = [n, n + len(v)]
        n += len(v)
        jd += v
    learn.write_trj(jd, com / "judge.trj", pair[0])
    _dump(com / "judge.json", parts)
    key = {("f", r["index"]): S["feedback"][r["index"]] for r in fb} | {("u", i): S["universe"][i] for i in range(len(uni))}
    np.save(com / "cache" / f"judge-{learn.sha256(s80)[:16]}.npy", np.concatenate([p[cj], S["exam"], S["fresh"], S["universe"][side1], [key[k] for k in ids], S["feedback"]]))
    for m in (u331, uf, c["q4cut"]):    # pre-score shared baseline models once
        learn.real_panel(m, pc, scorer)
    spec = dict(cut=cut, med=med, tau=tau, cal_train=len(train), cal_judge=len(cj), universe=len(uni), side1=len(side1), probe=len(probe), lesson_rows=len(ls), lesson_docs=len(docs), shares=count,
                replay=len(picks), panel=len(idx), human_tail_weight=1.0, human_rest_weight=float(w.max()), s80_cal_train_rate=float((p[train] < cut).mean()))
    _dump(com / "spec.json", spec)
    log(f"SETUP_DONE {json.dumps(spec)}")


class Uploader:
    """Ordered background checkpoint uploader that excludes local dataset artifacts in `common/`."""

    def __init__(self, c: dict, arm: str, log):
        os.environ.setdefault("MODELSCOPE_ENDPOINT", c["modelscope_endpoint"])
        os.environ.setdefault("MODELSCOPE_PREFER_AI_SITE", "true")
        self.repo, self.prefix, self.log, self.q = c["repo"], f"{c['repo_dir']}/{arm}", log, queue.Queue()
        self.t = threading.Thread(target=self._run, daemon=True)
        self.t.start()

    def put(self, *paths) -> None:
        assert all(str(p).endswith((".gguf", ".safetensors", ".json")) and "common" not in Path(p).parts for p in paths), paths
        self.q.put(paths)

    def _run(self) -> None:
        from modelscope.hub.api import HubApi
        api = HubApi()
        while (paths := self.q.get()) is not None:
            for p in paths:
                q = f"{self.prefix}/{Path(p).name}"
                for k in range(6):
                    try:
                        api.upload_file(repo_id=self.repo, path_or_fileobj=str(p), path_in_repo=q, commit_message=f"add {q}")
                        self.log(f"up {q}")
                        break
                    except Exception as e:
                        self.log(f"upload retry {k} {q} {type(e).__name__}")
                        time.sleep(30 * (k + 1))
                else:
                    self.log(f"UPLOAD_FAILED {q}: the rest of this batch stays local")
                    break

    def close(self) -> None:
        self.q.put(None)
        self.t.join()


def fetch(c: dict, arm: str, name: str, dst: Path) -> bool:
    """Fetch a checkpoint file from remote storage into `dst`, returning False if absent."""
    from modelscope.hub.errors import NotExistError
    from modelscope.hub.file_download import model_file_download
    os.environ.setdefault("MODELSCOPE_ENDPOINT", c["modelscope_endpoint"])
    os.environ.setdefault("MODELSCOPE_PREFER_AI_SITE", "true")
    d = dst.parent / ".ms"
    try:
        p = model_file_download(c["repo"], f"{c['repo_dir']}/{arm}/{name}", cache_dir=str(d))
    except Exception as e:
        if isinstance(e, NotExistError) or "404" in str(e):
            return False
        raise
    shutil.move(p, dst)
    shutil.rmtree(d, ignore_errors=True)
    return True


class Off(nn.Module):
    """A sub-block switched off but still held by the student: zero residual, as when the file has no such tensors."""

    def __init__(self, pair: bool):
        super().__init__()
        self.pair = pair

    def forward(self, *a, **k):
        z = torch.zeros_like(k["hidden_states"] if "hidden_states" in k else a[0])
        return (z, None) if self.pair else z


class Gates:
    """What a student has switched off. `fixed[il]` is an on/off mask over the file's neurons; with `learned`, `theta[il]` picks the
    top `k[il]` neurons each pass, so bytes stay exact while off and on neurons swap. Multiplying by the gate leaves a gradient at
    g = 0, so an off neuron can still ask to return. `ema` is the running raw dJ/dg per neuron, `res_ema` the same per sub-block."""

    def __init__(self, model, widths: dict[int, int], device: str, learned: bool, saved: dict | None = None):
        self.model, self.saved, self.fixed, self.k, self.decay = model, {}, {}, dict(widths), .98
        self.theta, self.ema, self.prev, self.res, self.res_ema = {}, {}, {}, {}, {}
        for il in widths:
            model.model.layers[il].mlp.down_proj.register_forward_pre_hook(self._pre(il))
        if not learned:
            return
        saved = saved or {}
        for il, n in widths.items():
            self.theta[il] = nn.Parameter(torch.ones(n, device=device))
            self.ema[il] = torch.as_tensor(saved["ema"][il], device=device) if il in saved.get("ema", {}) else torch.zeros(n, device=device)
            self.prev[il] = self.theta[il].detach().clone()
        for il, layer in enumerate(model.model.layers):
            for kind, m in (("ffn", layer.mlp), ("attn", self._mixer(layer))):
                if not type(m).__name__ == "Zero":
                    key = f"{il}.{kind}"
                    self.res[key] = nn.Parameter(torch.ones(1, device=device))
                    self.res_ema[key] = saved.get("res", {}).get(key, 0.0)
                    m.register_forward_hook(self._post(key))

    @staticmethod
    def _mixer(layer):
        return layer.linear_attn if layer.block_type == "linear_attention" else layer.self_attn

    def _pre(self, il: int):
        def f(mod, args):
            x = args[0]
            if il in self.fixed:
                return (x * self.fixed[il].to(x.dtype),)
            th = self.theta.get(il)
            if th is None:
                return None
            s = torch.sigmoid(th)
            m = torch.zeros_like(s)
            m[th.topk(self.k[il]).indices] = 1
            return (x * (m + s - s.detach()).to(x.dtype),)
        return f

    def _post(self, key: str):
        def f(mod, args, out):
            r = self.res[key]
            return (out[0] * r.to(out[0].dtype), *out[1:]) if isinstance(out, tuple) else out * r.to(out.dtype)
        return f

    def block(self, il: int, kind: str, off: bool) -> None:
        layer, key = self.model.model.layers[il], (il, kind)
        attr = "mlp" if kind == "ffn" else "linear_attn" if layer.block_type == "linear_attention" else "self_attn"
        if off and key not in self.saved:
            self.saved[key] = getattr(layer, attr)
            setattr(layer, attr, Off(attr == "self_attn"))
        elif not off and key in self.saved:
            setattr(layer, attr, self.saved.pop(key))

    def on_step(self, step, opt, losses) -> None:
        for il, th in self.theta.items():
            if th.grad is not None:
                s = torch.sigmoid(self.prev[il])
                self.ema[il].mul_(self.decay).add_((1 - self.decay) * th.grad / (s * (1 - s)).clamp_min(1e-6))
            self.prev[il] = th.detach().clone()
        for key, r in self.res.items():
            if r.grad is not None:
                self.res_ema[key] = self.decay * self.res_ema[key] + (1 - self.decay) * float(r.grad)

    def on(self, il: int) -> np.ndarray:
        """Positions of the neurons on now."""
        if il in self.fixed:
            return np.flatnonzero(self.fixed[il].cpu().numpy() > 0)
        if il in self.theta:
            return np.sort(self.theta[il].topk(self.k[il]).indices.cpu().numpy())
        raise KeyError(il)


class Arm:
    def __init__(self, cfg: dict, name: str, gpu: str, log=print):
        c = self.c = cfg["anneal"]
        self.name, self.a, self.gpu = name, c["arms"].get(name, {}), str(gpu)
        self.com, self.dir = Path(c["root"]) / "common", Path(c["root"]) / name
        self.dir.mkdir(parents=True, exist_ok=True)
        self.log = lambda m: log(f"{time.strftime('%H:%M:%S')} {name} {m}")
        self.tc = Toolchain.from_config(dict(cfg["toolchain"], gpus=[self.gpu]))
        self.scorer, self.pc, self.groups = BatchScorer(self.tc, c["slots"]), panel_cfg(c), groups_of(c)
        self.spec, self.pair, self.seed = json.loads((self.com / "spec.json").read_text()), tuple(c["pair"]), self.a.get("seed", c["seed"])
        self.sh = shape.Shaper(self.tc, c["src"], self.groups, c["imatrix"], c["vocab"], Path(c["work"]) / name, c["threads"], str(self.dir / "build.log"))
        self.ledger = shape.Ledger(c["src"], self.groups, len(self.sh.vocab), str(self.com / "u331.gguf"))
        self.w, self.j = np.load(self.com / "panel_w.npy"), {**c["j"], **self.a.get("j", {})}
        self.device, self.amp, self.budget = "cuda", True, c["j"]["score_budget"]
        self.stages = c["schedules"].get(self.a.get("schedule", "auction"), [])
        ldocs = learn.read_trj(str(self.com / "lesson.trj"))
        flip = {"human": "ai", "ai": "human"}
        self.lesson = [dict(r, tokens=ldocs[r["doc"]], label=flip[r["label"]] if self.a.get("twin") and r["flip"] else r["label"]) for r in json.loads((self.com / "lesson.json").read_text())]
        if self.a.get("lesson"):    # filter lesson rows to the configured group
            self.lesson = [r for r in self.lesson if r["group"] == self.a["lesson"]]
        self.replay, rp = learn.read_trj(str(self.com / "replay.trj")), json.loads((self.com / "replay.json").read_text())
        self.strata, self.anchor = rp["strata"], dict(base=rp["base"], cut=self.spec["cut"], med=self.spec["med"])
        self.shock, self.up, self.ready, self.res = learn.read_trj(str(self.com / "shock.trj")), None, None, {}
        self.side1 = [r for r in json.loads((self.com / "select.json").read_text()) if learn.row_side(r, self.pc) == 1]
        self.source, self.cfg, self.pool = None, cfg, ThreadPoolExecutor(1)

    def source_latents(self, names: list[str], keep: dict[int, np.ndarray]) -> dict[str, np.ndarray]:
        self.source = self.source or lattice.Artifact(self.c["src"])
        out = {}
        for n in names:
            a, (il, kind) = learn.floats(self.source, n, self.c["ggml_lib"]), shape._layer(n)
            out[n] = shape.carve(a, kind, keep[il]).astype(np.float32) if kind in shape.FFN else a.copy()
        return out

    def start(self) -> dict:
        s = json.loads((self.com / "structure.json").read_text())
        order = self.a.get("order", "auction")
        full = {il: np.arange(FULL) for il in range(24)}
        keep = {int(k): np.array(v) for k, v in s["keep"].items()} if order == "patched" else full
        real = [g for g in self.groups if not g.startswith("_")]
        tiers = {"patched": s["tiers"], "scheduled": {g: s["tiers"][g] if s["tiers"][g].startswith("IQ") else "Q4_K" for g in real}, "auction": dict.fromkeys(real, "Q4_K")}[order]
        file = {"patched": self.c["q4cut"], "scheduled": str(self.com / "uf.gguf"), "auction": str(self.com / "u331.gguf")}[order]
        art = lattice.Artifact(file)
        self.lat = self.source_latents(learn.train_names(art, {"train_regex": r"^blk\."}), keep)
        art.close()
        cut, med, _ = self.rates(file)
        lr = json.loads((self.com / "lr.json").read_text())["lr"] if (self.com / "lr.json").exists() else 1e-6
        self.gsave = {}
        return dict(stage=0, file=file, keep=keep, tiers=tiers, cut=cut, med=med, lr=self.a.get("lr", lr), fails=0, history=[], latents=None, gates=None, res={})

    def resume(self) -> dict | None:
        from safetensors.numpy import load_file
        p = self.dir / "state.json"
        if not p.exists() and not fetch(self.c, self.name, "state.json", p):
            return None
        st = json.loads(p.read_text())
        st["keep"] = {int(k): np.array(v) for k, v in st["keep"].items()}
        for f in (st["latents"], st["gates"], st["file"]):
            if f and not Path(f).exists():
                assert fetch(self.c, self.name, Path(f).name, Path(f)), f"checkpoint file {f} not found locally or in remote storage"
        self.lat = load_file(st["latents"]) if st["latents"] else None
        self.gsave = dict(ema={int(k): v for k, v in load_file(st["gates"]).items()}, res=st["res"]) if st["gates"] else {}
        if self.lat is None and st["stage"] < len(self.stages):
            art = lattice.Artifact(st["file"])
            self.lat = self.source_latents(learn.train_names(art, {"train_regex": r"^blk\."}), st["keep"])
            art.close()
        self.log(f"resume at stage {st['stage']} of {len(self.stages)}")
        return st

    def save(self, i: int, file: str) -> None:
        """Latents, gates and the file first, the state last, so the state always names a whole checkpoint. `finish` leaves only its file."""
        from safetensors.numpy import save_file
        st = self.st
        lat, gt = None, None
        if i + 1 < len(self.stages):
            lat = self.dir / f"latents-{i}.safetensors"
            save_file({n: np.ascontiguousarray(v) for n, v in self.lat.items()}, f"{lat}.part")    # carve leaves ffn_down in column order
            os.replace(f"{lat}.part", lat)
            if self.gsave:
                gt = self.dir / f"gates-{i}.safetensors"
                save_file({str(k): v for k, v in self.gsave["ema"].items()}, f"{gt}.part")
                os.replace(f"{gt}.part", gt)
                st["res"] = self.gsave["res"]
        old = st["latents"], st["gates"]
        st.update(stage=i + 1, file=file, latents=str(lat) if lat else None, gates=str(gt) if gt else None, fails=0)
        self.write()
        self.up.put(*[p for p in (lat, gt, file, self.dir / "state.json") if p])
        for f in old:    # the upload above may still be reading only the newest pair
            if f and f not in (st["latents"], st["gates"]):
                Path(f).unlink(missing_ok=True)

    def write(self) -> None:
        _dump(self.dir / "state.json", {**self.st, "keep": {str(k): np.asarray(v).tolist() for k, v in self.st["keep"].items()}})

    def rates(self, file: str) -> tuple[float, float, np.ndarray]:
        """cut and med from the panel's cal-train humans, weighted back to all of cal-train, and the panel's scores."""
        v = learn.real_panel(file, self.pc, self.scorer)
        h = v[:self.c["panel_humans"]]
        return wq(h, self.w, .01), wq(h, self.w, .5), v

    def student(self, file: str, lat: dict, gsave: dict):
        art = lattice.Artifact(file)
        names = learn.train_names(art, {"train_regex": r"^blk\."})
        model, grids = learn.build_student(art, self.c["config_dir"], names, self.c["ggml_lib"], self.device, latents={n: lat[n] for n in names})
        if self.j.get("checkpointing", True):
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            model.train()    # HF checkpoints only in training mode; the model has no dropout
        widths = {il: art.shape(f"blk.{il}.ffn_down.weight")[1] for il in range(24) if f"blk.{il}.ffn_down.weight" in art.tensors}
        g = Gates(model, widths, self.device, self.a.get("order", "auction") == "auction", gsave)
        art.close()
        return model, grids, g

    def train(self, model, grids, g: Gates, steps: int, seed: int, lr: float | None = None) -> list:
        ws = [w for w, _, _ in grids.values()]
        params = [dict(params=ws), dict(params=list(g.theta.values()), lr=self.c["gate_lr"]), dict(params=list(g.res.values()), lr=0.0)] if g.theta else ws
        j = dict(self.j, steps=steps, seed=seed, z_units=dict(cut=self.st["cut"], med=self.st["med"], tau=self.spec["tau"]))
        return learn._train(model, params, self.lesson, self.replay, self.pair, 0.0, j, self.log, self.device, self.amp, self.strata, lr or self.st["lr"], anchor=self.anchor, on_step=g.on_step if g.theta else None)

    def h0(self) -> None:
        """Compute and cache frozen teacher hidden states when hold_kind == 'klvec'."""
        p = self.com / "h0.npy"
        if self.j.get("hold_kind") != "klvec":
            return
        if not p.exists():
            art = lattice.Artifact(self.com / "s80.gguf")
            model, _ = learn.build_student(art, self.c["config_dir"], [], self.c["ggml_lib"], self.device)
            with torch.no_grad():
                hs = [(b, learn.vec_scores(model, [self.replay[i] for i in b], self.pair, self.device, self.amp)[1].cpu().numpy()) for b in learn.batches(self.replay, self.budget)]
            h = np.zeros((len(self.replay), hs[0][1].shape[1]), np.float32)
            for b, x in hs:
                h[b] = x
            np.save(f"{p}.tmp.npy", h)
            os.replace(f"{p}.tmp.npy", p)
            del model
            gc.collect()
            torch.cuda.empty_cache()
        self.anchor["h0"] = np.load(p)

    def scheduled(self, frac: float) -> tuple[dict, dict]:
        """Interpolate `frac` toward the target structure in structure.json (lowest-Fisher neurons first, mixers in layer order)."""
        s = json.loads((self.com / "structure.json").read_text())
        fisher = np.fromfile(self.c["fisher"], dtype=np.float32).reshape(24, FULL)
        st, keep, tiers = self.st, {}, dict(self.st["tiers"])
        for il, cur in st["keep"].items():
            g = f"blk.{il}.ffn"
            if st["tiers"][g] == DROP:
                continue
            fin = np.array(s["keep"].get(str(il), []), dtype=np.int64) if s["tiers"][g] != DROP else np.empty(0, np.int64)
            w = len(fin) if frac >= 1 else int(-(-round(FULL - (FULL - len(fin)) * frac) // CHUNK) * CHUNK)
            order = cur[np.lexsort((fisher[il][cur], np.isin(cur, fin)))]
            keep[il] = np.sort(order[len(order) - w:])
            assert np.isin(fin, keep[il]).all() or w == 0, il
            if w == 0:
                tiers[g] = DROP
        mixers = sorted((g for g, t in s["tiers"].items() if t == DROP and g.endswith(".attn")), key=lambda g: int(g.split(".")[1]))
        for g in mixers[:round(len(mixers) * frac)]:
            tiers[g] = DROP
        return keep, tiers

    def probes(self, i: int) -> tuple[list[np.ndarray], np.ndarray]:
        """Sample training-disjoint near-cutoff probes for stage `i`."""
        docs, p = learn.read_trj(str(self.com / "probe.trj")), json.loads((self.com / "probe.json").read_text())
        y, s, rng = np.array(p["ai"]), np.abs(np.array(p["s80"]) - self.spec["cut"]), np.random.default_rng(self.seed + i)
        n, idx = self.c["probe_n"], []
        for a in (False, True):
            near = np.flatnonzero(y == a)[np.argsort(s[y == a])[:2 * n]]
            idx += rng.choice(near, min(n, len(near)), replace=False).tolist()
        return [docs[j] for j in idx], y[idx]

    def auction(self, model, grids, g: Gates, i: int, s: dict) -> tuple[dict, dict, dict]:
        """Moves by damage per MB. Near-cut AUC prices FFN drops, half-width cuts and mixer drops directly on the probes. 256-neuron
        chunks are priced by the EMA of dJ/dg, up to 20% of a layer per rung, on a scale set by four measured chunks. The chosen set is
        then measured whole against a do-nothing probe; past 1.5 times what its parts add to, its dearest part goes and it is measured again."""
        st, led = self.st, self.ledger
        keep, tiers = st["keep"], st["tiers"]
        widths = {il: len(v) for il, v in keep.items() if tiers[f"blk.{il}.ffn"] != DROP}
        docs, y = self.probes(i)
        score = lambda d=docs: learn.score_all(model, d, self.pair, self.device, self.amp, self.budget)
        t0, rank = time.time(), {il: np.argsort(-g.ema[il].cpu().numpy()) for il in widths}

        def trial(ms: list[dict]) -> np.ndarray:
            n = {}
            for m in ms:
                if m["k"] in ("ffn", "mixer"):
                    g.block(m["il"], "ffn" if m["k"] == "ffn" else "attn", True)
                else:
                    n[m["il"]] = n.get(m["il"], 0) + m["n"]
            for il, x in n.items():
                on = torch.ones(widths[il], device=self.device)
                on[torch.as_tensor(rank[il][:x].copy(), device=self.device)] = 0
                g.fixed[il] = on
            v = score()
            g.fixed.clear()
            for m in ms:
                if m["k"] in ("ffn", "mixer"):
                    g.block(m["il"], "ffn" if m["k"] == "ffn" else "attn", False)
            return v

        base = score()
        a0 = auc(base, y)
        back = np.random.default_rng(i).permutation(len(docs))
        v = score([docs[j] for j in back])
        null = abs(auc(v, y[back]) - a0)
        imp = lambda key: -g.res_ema.get(key, 0.0)
        ffns = sorted(widths, key=lambda il: imp(f"{il}.ffn"))[:self.c["probe_layers"]]
        mix = sorted((il for il in range(24) if tiers[f"blk.{il}.attn"] != DROP and self.groups[f"blk.{il}.attn"]["droppable"]), key=lambda il: imp(f"{il}.attn"))[:self.c["probe_layers"]]
        moves = [dict(k="mixer", il=il, bytes=led.saved(widths, tiers, group=f"blk.{il}.attn")) for il in mix]
        for il in ffns:
            h = -(-widths[il] // 2 // CHUNK) * CHUNK
            if h < widths[il] and h:
                moves.append(dict(k="half", il=il, n=widths[il] - h, bytes=led.saved(widths, tiers, il=il, neurons=widths[il] - h)))
        checked = list(moves)
        moves += [dict(k="ffn", il=il, bytes=led.saved(widths, tiers, group=f"blk.{il}.ffn")) for il in ffns if self.groups[f"blk.{il}.ffn"]["droppable"]]
        S = {}
        for m in moves:
            S[id(m)] = trial([m])
            m["dmg"] = a0 - auc(S[id(m)], y)
        chunks = []
        for il in widths:
            e = g.ema[il].cpu().numpy()[rank[il]]
            cap = max(1, int(.2 * widths[il]) // CHUNK)
            for j in range(widths[il] // CHUNK - 1):
                chunks.append(dict(k="chunk", il=il, j=j, n=CHUNK, bytes=led.saved(widths, tiers, il=il), pred=float(-e[j * CHUNK:(j + 1) * CHUNK].sum()), over=j >= cap))
        first = [m for m in chunks if m["j"] == 0]
        cal = first[::max(1, len(first) // 4)][:4]
        for m in cal:
            m["dmg"] = a0 - auc(trial([m]), y)
        scale = sum(abs(m["dmg"]) for m in cal) / max(sum(abs(m["pred"]) for m in cal), 1e-30)
        for m in chunks:
            m["dmg"] = m.get("dmg", scale * m["pred"])
        boot = np.random.default_rng(self.seed + i).integers(0, len(y), (200, len(y)))
        ban, rec = [], {}
        for attempt in range(3):
            plan, w2, t2 = self.choose(moves + chunks, widths, tiers, int(s["mb"] * shape.MB), ban, null)
            v = trial(plan)
            actual, pred = a0 - auc(v, y), sum(m["dmg"] for m in plan)
            se = float(np.std([auc(base[b], y[b]) - auc(v[b], y[b]) for b in boot]))
            rec = dict(a0=a0, null=null, se=se, actual=actual, predicted=pred, scale=scale, attempt=attempt, moves=[{k: x for k, x in m.items() if k != "over"} for m in plan], price_per_mb=max(m["dmg"] / m["bytes"] * shape.MB for m in plan), sec=round(time.time() - t0))
            self.log(f"auction {s['mb']} MB attempt {attempt}: {len(plan)} moves, damage {actual:.4f} against predicted {pred:.4f}, null {null:.4f}, se {se:.4f}, lambda {rec['price_per_mb']:.5f}/MB, ledger {led.size(w2, t2) / shape.MB:.2f} MB")
            if actual <= 1.5 * max(pred, null, se) or attempt == 2:
                break
            ban.append(max(plan, key=lambda m: m["dmg"] / m["bytes"]))
        if s.get("ranking") and not self.a.get("twin"):
            rec["ranking"] = self.ranking(model, grids, g, checked, S, trial, score, y)
        new = {il: keep[il][np.sort(rank[il][widths[il] - w:])] for il, w in w2.items()}
        return new, t2, rec

    def choose(self, moves: list[dict], widths: dict, tiers: dict, target: int, ban: list[dict], floor: float) -> tuple[list, dict, dict]:
        """Cheapest damage per byte until the ledger reaches `target`. Damage within the noise floor ties, so bigger moves win those.
        A move far larger than what is left waits unless nothing else can finish. Chunks past the 20% cap are a last resort."""
        led, w, t, plan, nc = self.ledger, dict(widths), dict(tiers), [], {}
        price = lambda m: max(m["dmg"], floor, 1e-12) / m["bytes"] * (1e6 if m.get("over") else 1)
        order = sorted((m for m in moves if not any(m is b for b in ban)), key=price)
        for strict in (True, False):
            for m in order:
                left, il = led.size(w, t) - target, m["il"]
                if left <= 0:
                    break
                g = f"blk.{il}.{'attn' if m['k'] == 'mixer' else 'ffn'}"
                if any(m is x for x in plan) or t[g] == DROP or (strict and m["bytes"] > left + 2 * CHUNK * 1728):
                    continue
                if m["k"] == "chunk":
                    if m["j"] != nc.get(il, 0) or w[il] <= CHUNK:
                        continue
                    w[il] -= CHUNK
                    nc[il] = m["j"] + 1
                elif m["k"] == "half":
                    if nc.get(il):
                        continue
                    w[il] -= m["n"]
                    nc[il] = 1 << 20
                else:
                    t[g] = DROP
                    if m["k"] == "ffn":
                        w.pop(il, None)
                plan.append(m)
        assert led.size(w, t) <= target, "no moves left to reach the rung's target"
        return plan, w, t

    def ranking(self, model, grids, g: Gates, moves: list[dict], taught: dict, trial, score, y: np.ndarray) -> dict:
        """The same moves on taught and untaught weights. If they rank alike, learning does not steer structure."""
        src = self.source_latents(list(grids), self.st["keep"])
        for n, (w, _, _) in grids.items():
            w.data.copy_(torch.from_numpy(src[n]).to(w.device))
        base0 = score()
        un = {id(m): trial([m]) for m in moves}
        for n, (w, _, _) in grids.items():
            w.data.copy_(torch.from_numpy(self.lat[n]).to(w.device))
        base1, rng, rows = score(), np.random.default_rng(self.seed), []
        for b in rng.integers(0, len(y), (200, len(y))):
            d1 = [auc(base1[b], y[b]) - auc(taught[id(m)][b], y[b]) for m in moves]
            d0 = [auc(base0[b], y[b]) - auc(un[id(m)][b], y[b]) for m in moves]
            rows.append(spearmanr(d1, d0)[0])
        out = dict(moves=len(moves), spearman=float(np.nanmean(rows)), lower=float(np.nanquantile(rows, .025)))
        self.log(f"ranking {out}")
        return out

    def stage(self, i: int, s: dict) -> None:
        st, t0 = self.st, time.time()
        model, grids, g = self.ready.result() if self.ready else self.student(st["file"], self.lat, self.gsave)
        self.ready = None
        keep, tiers, rec = st["keep"], st["tiers"], {}
        if s["kind"] == "cut":
            keep, tiers = self.scheduled(s["frac"])
        elif s["kind"] == "rung":
            keep, tiers, rec = self.auction(model, grids, g, i, s)
            if rec.get("ranking", {}).get("lower", 0) > self.c["ranking_bound"] and self.c.get("ranking_stop"):
                st["history"].append(dict(stage=i, verdict="auction dropped: learning does not steer structure", **rec))
                st["stage"] = len(self.stages)
                self.write()
                self.up.put(self.dir / "state.json")
                return
        for gr, t in tiers.items():
            if t == DROP and st["tiers"][gr] != DROP:
                il, kind = gr.split(".")[1:]
                g.block(int(il), kind, True)
        for il, k in keep.items():
            if il in g.k and tiers[f"blk.{il}.ffn"] != DROP:
                on = np.isin(st["keep"][il], k)
                if il in g.theta:
                    g.k[il] = int(on.sum())
                    g.theta[il].data = torch.as_tensor(np.where(on, .5, -.5), dtype=torch.float32, device=self.device)    # initial gate margin +/-0.5
                    g.prev[il] = g.theta[il].detach().clone()
                elif not on.all():
                    g.fixed[il] = torch.as_tensor(on, dtype=torch.float32, device=self.device)
        hist = self.train(model, grids, g, s["steps"], self.seed + i + 1000 * st["fails"])
        if g.theta:
            keep = {il: st["keep"][il][g.on(il)] for il in keep if il in g.k and tiers[f"blk.{il}.ffn"] != DROP}
        with torch.no_grad():
            sh0 = learn.score_all(model, self.shock, self.pair, self.device, self.amp, self.budget)
        lat = {n: w.detach().cpu().numpy() for n, (w, _, _) in grids.items()}
        rec.update(stage=i, kind=s["kind"], loss_first=hist[0], loss_last=hist[-1], loss_mean=np.mean(hist, 0).tolist())
        if s["kind"] == "finish":
            lv = {n: fq.levels(w.detach()).reshape(-1).to("cpu", torch.uint8).numpy() for n, (w, fq, _) in grids.items()}
            rec["levels_moved"] = sum(int((lv[n] != x[2].numpy()).sum()) for n, x in grids.items())
            del model, grids, g
            gc.collect()
            torch.cuda.empty_cache()
            out = str(self.dir / "final.gguf")
            lattice.write_levels(st["file"], out, lv)
            self.close_stage(i, out, rec, sh0, t0, False, keep, tiers, lat, None)
            return
        gsave = {}
        if g.theta:
            gsave = dict(ema={il: g.ema[il].cpu().numpy()[g.on(il)] for il in g.k if il in keep}, res=dict(g.res_ema))
        del model, grids, g
        gc.collect()
        torch.cuda.empty_cache()
        out = str(self.dir / f"stage-{i}.gguf")
        built, nb = self.sh.build(keep, tiers, out, lat, st["keep"])
        carved = {}
        for n, a in lat.items():
            il, kind = shape._layer(n)
            if tiers[f"blk.{il}.{'ffn' if kind in shape.FFN else 'attn'}"] != DROP:
                carved[n] = shape.carve(a, kind, np.searchsorted(st["keep"][il], keep[il])) if kind in shape.FFN else a
        rec["bytes"] = nb
        self.close_stage(i, built, rec, sh0, t0, bool(s.get("check")), keep, tiers, carved, gsave)

    def close_stage(self, i: int, file: str, rec: dict, sh0: np.ndarray, t0: float, check: bool, keep: dict, tiers: dict, lat: dict, gsave: dict | None) -> None:
        """Score the real file while the next student builds, then keep the stage, or give it up if it failed its check."""
        st = self.st
        if gsave is not None and i + 1 < len(self.stages):
            self.ready = self.pool.submit(self.student, file, lat, gsave)
        with ThreadPoolExecutor(2) as ex:
            p, sh = ex.submit(self.rates, file), ex.submit(learn.real_panel, file, dict(self.pc, panel_trj=str(self.com / "shock.trj")), self.scorer)
            cut, med, v = p.result()
            sh1 = sh.result()
        idx = len(self.w)
        sel, R = v[idx:], []
        for r, x in zip(self.side1, sel):
            R += [{"task": r["task"], "family": r["family"], "side": "cand", "success": bool((x < cut) == (r["verdict"] == "ai"))},
                  {"task": r["task"], "family": r["family"], "side": "src", "success": r["src"]}]
        b = stats.delta_and_bounds(R, .05 / 4)
        rec.update(cut=cut, med=med, L=b["L"], delta=b["delta"], shock_rms=float(np.sqrt(np.mean((sh1 - sh0) ** 2))), shock_max=float(np.abs(sh1 - sh0).max()), bytes=rec.get("bytes", os.path.getsize(file)), sec=round(time.time() - t0))
        failed = check and (b["L"] <= -self.c["gate_margin"] or abs(med - st["med"]) > self.c["median_guard"])
        self.log(f"stage {i} {rec['kind']}: {rec['bytes'] / shape.MB:.2f} MB, L {b['L']:.4f}, cut {cut:.3f}, med {med:.3f}, shock rms {rec['shock_rms']:.3f} max {rec['shock_max']:.3f}, loss {rec['loss_first']} -> {rec['loss_last']}, {rec['sec']} s{' FAILED' if failed else ''}")
        st["history"].append(rec)
        if failed:
            self.ready = self.ready.result() and None
            gc.collect()
            torch.cuda.empty_cache()
            st["fails"] += 1
            if st["fails"] >= 2:
                st["floor"] = os.path.getsize(st["file"])
                self.log(f"two failed rungs; stopping at size floor {st['floor'] / shape.MB:.2f} MB")
                st["stage"] = next(k for k, x in enumerate(self.stages) if x["kind"] == "finish")
                st["fails"] = 0
            self.write()
            self.up.put(self.dir / "state.json")
            return
        st.update(keep=keep, tiers=tiers, cut=cut, med=med)
        self.lat, self.gsave = lat, gsave or {}
        self.save(i, file)

    def run(self) -> None:
        self.up = Uploader(self.c, self.name, self.log)
        self.st = self.resume() or self.start()
        self.h0()
        while self.st["stage"] < len(self.stages):
            i = self.st["stage"]
            self.stage(i, self.stages[i])
        if self.st["history"] and not str(self.st["history"][-1].get("verdict", "")).startswith("auction dropped"):
            judge(self.cfg, self.name, self.st["file"], self.gpu, self.log, self.st)
            self.up.put(self.dir / "result.json")
        self.up.close()
        self.log("ARM_DONE")


def sweep(cfg: dict, lr: float, gpu: str, log=print) -> None:
    """Run a short LR sweep from the initial scheduled checkpoint and record per-step score RMS."""
    arm = Arm(cfg, f"sweep-{lr:g}", gpu, log)
    arm.a = dict(order="scheduled")
    arm.st = arm.start()
    model, grids, g = arm.student(arm.st["file"], arm.lat, {})
    docs = arm.shock[:cfg["anneal"]["sweep_docs"]]
    with torch.no_grad():
        s0 = learn.score_all(model, docs, arm.pair, arm.device, arm.amp, arm.budget)
    t, n = time.time(), cfg["anneal"]["sweep_steps"]
    hist = arm.train(model, grids, g, n, arm.seed, lr)
    sec = (time.time() - t) / n
    with torch.no_grad():
        s1 = learn.score_all(model, docs, arm.pair, arm.device, arm.amp, arm.budget)
    d = Path(cfg["anneal"]["root"]) / "common" / "sweep"
    d.mkdir(exist_ok=True)
    rms = float(np.sqrt(np.mean((s1 - s0) ** 2)) / n)
    _dump(d / f"{lr:g}.json", dict(lr=lr, rms_per_step=rms, sec_per_step=sec, loss_first=hist[0], loss_last=hist[-1]))
    arm.log(f"sweep {lr:g}: {rms:.4f} rms per step, {sec:.2f} s/step")


def pick(cfg: dict, log=print) -> None:
    c = cfg["anneal"]
    com = Path(c["root"]) / "common"
    rows = sorted((json.loads(p.read_text()) for p in (com / "sweep").glob("*.json")), key=lambda r: r["lr"])
    ok = [r for r in rows if r["rms_per_step"] < c["sweep_limit"]]
    best = ok[-1] if ok else rows[0]
    _dump(com / "lr.json", dict(lr=best["lr"], rows=rows))
    log(f"PICK lr {best['lr']:g} from {[(r['lr'], round(r['rms_per_step'], 4)) for r in rows]}")


def judge(cfg: dict, name: str, model: str, gpu: str, log=print, st: dict | None = None) -> dict:
    """Evaluate a built GGUF across exam, fresh, side-1, and held-out splits at its own cal-judge cutoff."""
    c = cfg["anneal"]
    root, com, pair = Path(c["root"]), Path(c["root"]) / "common", tuple(c["pair"])
    parts, spec = json.loads((com / "judge.json").read_text()), json.loads((com / "spec.json").read_text())
    cache = com / "cache" / f"judge-{learn.sha256(model)[:16]}.npy"
    if not cache.exists():
        v = BatchScorer(Toolchain.from_config(dict(cfg["toolchain"], gpus=[str(gpu)])), c["slots"]).score_trj(model, str(com / "judge.trj"), pair)
        np.save(f"{cache}.part.npy", v)
        os.replace(f"{cache}.part.npy", cache)
    v = np.load(cache)
    part = lambda k: v[slice(*parts[k])]
    cal, s = part("calj"), part("exam")
    cut, med = float(np.sort(cal)[int(.01 * len(cal))]), float(np.median(cal))
    exam = json.loads((com / "exam.json").read_text())
    y, ed = np.array([r["label"] == "ai" for r in exam]), np.array([r["kind"] == "expert-edited" for r in exam])
    rate = c["beemo_human_rate"]
    B = boots(exam, c["n_boot"], c["seed"])
    box = np.array([(caught_at(s[b], y[b], rate), auc(s[b], y[b])) for b in B])
    ci = lambda x: [float(q) for q in np.quantile(x, [.025, .975])]
    t = float(np.quantile(s[~y], rate))
    beemo = dict(caught_at_rate=caught_at(s, y, rate), caught_ci=ci(box[:, 0]), auc=auc(s, y), auc_ci=ci(box[:, 1]), generated=float((s[y & ~ed] < t).mean()), edited=float((s[y & ed] < t).mean()),
                 human_flagged_at_cutoff=float((s[~y] < cut).mean()), ai_caught_at_cutoff=float((s[y] < cut).mean()))
    side, u = json.loads((com / "side1.json").read_text()), part("side1")
    ai, wo, src = np.array([r["label"] == "ai" for r in side]), np.array([r["task"].startswith("wo") for r in side]), np.array([r["s80"] for r in side])
    R = []
    for r, a, x, x0 in zip(side, ai, u, src):
        R += [{"task": r["task"], "family": r["family"], "side": "cand", "success": bool((x < cut) == a)}, {"task": r["task"], "family": r["family"], "side": "src", "success": bool((x0 < spec["cut"]) == a)}]
    g = stats.delta_and_bounds(R, .05 / 4)
    near = np.concatenate([np.flatnonzero(ai == a)[np.argsort(np.abs(u[ai == a] - cut))[:c["probe_n"]]] for a in (False, True)])
    fresh, f = json.loads(Path(c["fresh_rows"]).read_text()), part("fresh")
    count = lambda k, w=None: [int(sum(f[i] < cut for i, r in enumerate(fresh) if r["kind"] == k and (w is None or r["writer"] == w))), sum(r["kind"] == k and (w is None or r["writer"] == w) for r in fresh)]
    gaps = [float(f[i] - f[i + 1]) for i, r in enumerate(fresh) if r["kind"] == "portion-human"]
    ls, h = json.loads((com / "lesson.json").read_text()), part("lesson")
    z = lambda x: (x - cut) / (med - cut) / spec["tau"]
    loss, by = np.logaddexp(0, -np.array([1.0 if r["label"] == "human" else -1.0 for r in ls]) * z(h[[r["doc"] for r in ls]])), {}
    for r, x in zip(ls, loss):
        if r["share"]:
            by.setdefault(r["share"], []).append(float(x))
    fb, held = json.loads(Path(c["feedback_rows"]).read_text()), part("feedback")
    hs = {}
    for r in fb:
        if r.get("fold") == c["held_fold"]:
            hs.setdefault(r["pair"], {})[r["label"]] = held[r["index"]]
    hg = [float(p["human"] - p["ai"]) for p in hs.values() if len(p) == 2]
    out = dict(arm=name, model=str(model), bytes=os.path.getsize(model), sha256=learn.sha256(model), cutoff=cut, median=med, scores=str(cache), beemo=beemo,
               unfiltered=dict(n=len(side), delta=g["delta"], L=g["L"], U=g["U"], gate_pass=bool(g["L"] > -c["gate_margin"]), near_auc=auc(u[near], ai[near]), caught_wo=float((u[ai & wo] < cut).mean()), caught_beemo=float((u[ai & ~wo] < cut).mean()),
                               human_flagged_wo=float((u[~ai & wo] < cut).mean()), human_flagged_beemo=float((u[~ai & ~wo] < cut).mean())),
               fresh=dict(synth=count("portion-synth"), human=count("portion-human"), mixed=count("article-mixed"), article_human=count("article-human"), by_writer={w: count("portion-synth", w) for w in ("opus", "sonnet", "haiku", "fable")},
                          pairs_right=sum(x > 0 for x in gaps), n_pairs=len(gaps), mean_gap=float(np.mean(gaps))),
               lesson_loss=dict(all=float(np.mean([x for q in by.values() for x in q])), by_share={k: float(np.mean(x)) for k, x in by.items()}),
               held=dict(pairs=len(hg), pairs_right=sum(x > 0 for x in hg), mean_gap=float(np.mean(hg)) if hg else None),
               split="dev_select")
    if st:
        out.update(lr=st["lr"], history=st["history"], floor=st.get("floor"), widths={str(k): len(v) for k, v in st["keep"].items()}, dropped=sorted(g for g, t in st["tiers"].items() if t == DROP))
    (root / name).mkdir(exist_ok=True)
    _dump(root / name / "result.json", out)
    log(f"JUDGE {name}: caught {beemo['caught_at_rate']:.4f} at {rate} humans, AUC {beemo['auc']:.4f}, {out['bytes'] / shape.MB:.2f} MB, unfiltered L {g['L']:.4f}, WO humans {out['unfiltered']['human_flagged_wo']:.4f}, fresh {out['fresh']['synth']}")
    return out


def boots(exam: list[dict], n: int, seed: int) -> list[np.ndarray]:
    """Prompt-cluster resamples, the same ones for every file, so that differences between files stay paired."""
    by = {}
    for i, r in enumerate(exam):
        by.setdefault(r["prompt_id"], []).append(i)
    keys = sorted(by)
    cl = [np.array(by[k]) for k in keys]
    rng = np.random.default_rng(seed)
    return [np.concatenate([cl[j] for j in rng.integers(0, len(cl), len(cl))]) for _ in range(n)]


def report(cfg: dict, log=print) -> dict:
    """Compare built arm results under the decision rule, compute G(x) vs twin deltas, and write report.json."""
    c = cfg["anneal"]
    root, com = Path(c["root"]), Path(c["root"]) / "common"
    parts, exam = json.loads((com / "judge.json").read_text()), json.loads((com / "exam.json").read_text())
    y, B = np.array([r["label"] == "ai" for r in exam]), boots(exam, c["n_boot"], c["seed"])
    res = {p.parent.name: json.loads(p.read_text()) for p in root.glob("*/result.json")}
    S = {k: np.load(r["scores"])[slice(*parts["exam"])] for k, r in res.items()}
    C = {k: np.array([caught_at(s[b], y[b], c["beemo_human_rate"]) for b in B]) for k, s in S.items()}
    point = {k: res[k]["beemo"]["caught_at_rate"] for k in res}
    diff = lambda a, b: dict(est=point[a] - point[b], lo=float(np.quantile(C[a] - C[b], .025)), hi=float(np.quantile(C[a] - C[b], .975)))
    out, rule = dict(caught=point), []
    for x in ("P", "F", "L"):
        if x in res and f"{x}-twin" in res:
            out[f"G({x})"] = diff(x, f"{x}-twin")
            if res[x]["lesson_loss"]["all"] >= res[f"{x}-twin"]["lesson_loss"]["all"]:
                rule.append(f"{x}: lesson loss did not drop below twin baseline")
    if "F" in res and "P" in res:
        out["F-P"] = diff("F", "P")
        if out["F-P"]["est"] <= 0:
            rule.append("F <= P on target gain")
    if "G(F)" in out and "G(P)" in out:
        d = C["F"] - C["F-twin"] - C["P"] + C["P-twin"]
        out["G(F)-G(P)"] = dict(est=out["G(F)"]["est"] - out["G(P)"]["est"], lo=float(np.quantile(d, .025)), hi=float(np.quantile(d, .975)))
        if out["G(F)-G(P)"]["est"] < .01 or out["G(F)-G(P)"]["lo"] <= 0:
            rule.append("G(F) - G(P) below 1 pt or CI <= 0")
    rows = [k for k in c["patched_rows"] if k in res]
    rank = [h["ranking"] for h in res.get("L", {}).get("history", []) if "ranking" in h]
    if rank and rank[0]["lower"] > c["ranking_bound"]:
        rule.append(f"ranking lower bound {rank[0]['lower']:.3f}: auction gain below threshold; fixed schedule preferred")
    if "L" in res and rows:
        best = max(rows, key=lambda k: point[k])
        out["L-best_row"] = dict(diff("L", best), row=best)
        u = res["L"]["unfiltered"]
        ok = out["L-best_row"]["lo"] > 0 and u["human_flagged_wo"] <= .015 and u["gate_pass"]
        rule.append(f"full ladder {'eligible' if ok else 'ineligible'}: L vs {best} {out['L-best_row']}, WO humans {u['human_flagged_wo']:.4f}, unfiltered gate {u['gate_pass']}")
    out["rule"] = rule
    _dump(root / "report.json", out)
    log(f"REPORT {json.dumps(out)}")
    return out


def main(cfg: dict, a) -> None:
    if a.stage == "setup":
        setup(cfg)
    elif a.stage == "sweep":
        sweep(cfg, a.lr, a.gpu)
    elif a.stage == "pick":
        pick(cfg)
    elif a.stage == "judge":
        rows = cfg["anneal"]["rows"]
        judge(cfg, a.arm, a.model or sorted(glob.glob(rows[a.arm]))[0], a.gpu)
    elif a.stage == "report":
        report(cfg)
    else:
        Arm(cfg, a.arm, a.gpu).run()
