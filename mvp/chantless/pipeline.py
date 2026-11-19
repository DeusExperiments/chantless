"""End-to-end Chantless MVP run: prepare -> pool -> aa -> proxy -> sanity -> search -> freeze -> exam -> report.

Every phase writes receipts under runs/<run_id>/ and a completion marker, so a run can resume.
The exam split is unreadable before the exam phase (SplitGuard). Nothing from the exam feeds back.
"""
from __future__ import annotations

import json
import math
import random
import re
import shutil
import struct
import subprocess
import time
from pathlib import Path

from . import builder as bmod, gguf_ops, kl, mask as mk, sanity, stats
from .env.protocol import load_environment
from .evaluator import Evaluator, ok_rows
from .learner import Learner, LearnerState
from .prober import Prober, Store
from .runtime import LlamaServer, Toolchain, imatrix as run_imatrix
from .util import Log, append_jsonl, read_json, read_jsonl, sha256_file, stable_seed, write_json

PHASES = ["prepare", "pool", "aa", "proxy", "sanity", "search", "freeze", "exam", "report"]


def choose_scratch_layout(
    sr_cfg: str | None,
    src_sz: int,
    run_dir: Path,
    run_id: str,
    auto_shm: bool = False,
    stage_source: bool | None = None,
    shm_path: Path = Path("/dev/shm"),
    shm_free: int | None = None,
) -> tuple[Path, bool]:
    def _shm_free() -> int:
        if shm_free is not None:
            return shm_free
        return shutil.disk_usage(shm_path).free if shm_path.exists() else 0

    if sr_cfg == "auto" or (sr_cfg is None and auto_shm):
        free_b = _shm_free() if src_sz > 0 else 0
        if src_sz > 0 and free_b >= max(30 << 30, int(3.5 * src_sz)):
            return shm_path / "chantless" / run_id, (True if stage_source is None else bool(stage_source))
        if src_sz > 0 and free_b >= int(1.5 * src_sz):
            return shm_path / "chantless" / run_id, bool(stage_source)
        return run_dir, False

    if sr_cfg:
        scratch = Path(sr_cfg) / run_id
        if stage_source is not None:
            return scratch, bool(stage_source)
        if str(scratch).startswith(str(shm_path)):
            return scratch, (src_sz > 0 and _shm_free() >= max(30 << 30, int(3.5 * src_sz)))
        return scratch, True

    return run_dir, False


def clamp_probe_workers(
    requested: int | None,
    scratch_dir: Path,
    weights_sz: int,
    n_gpus: int = 1,
    clamp_enabled: bool = True,
    max_auto: int = 5,
    free_bytes: int | None = None,
) -> tuple[int, int, int]:
    probe_path = scratch_dir if scratch_dir.exists() else scratch_dir.parent
    if free_bytes is not None:
        free_b = free_bytes
    else:
        while not probe_path.exists() and probe_path != probe_path.parent:
            probe_path = probe_path.parent
        free_b = shutil.disk_usage(probe_path).free
    if weights_sz <= 0:
        eff = requested if requested is not None else max(1, n_gpus)
        return eff, eff, free_b
    cand_bytes = max(1, int(1.05 * weights_sz))
    cap = max(1, int(free_b * 0.85 // cand_bytes))
    is_shm = str(scratch_dir).startswith("/dev/shm")
    if requested is None:
        if is_shm:
            eff = min(max_auto, cap) if clamp_enabled else max(1, n_gpus)
        else:
            default_w = max(1, n_gpus)
            eff = min(default_w, cap) if clamp_enabled else default_w
    else:
        eff = min(requested, cap) if clamp_enabled else requested
    return max(1, eff), cap, free_b


def choose_tier_cache_dir(
    scratch_dir: Path,
    run_dir: Path,
    src_bytes: int,
    tiers: list[str],
    cfg_dir: str | None = None,
    shm_path: Path = Path("/dev/shm"),
    shm_free: int | None = None,
    nvme_free: int | None = None,
) -> tuple[Path | None, int, str]:
    est = bmod.estimate_tier_cache_bytes(src_bytes, tiers)
    cand_bytes = max(1, int(1.05 * src_bytes))
    need_shm = est + 2 * cand_bytes
    need_nvme = est + cand_bytes
    if cfg_dir:
        p = Path(cfg_dir)
        p.mkdir(parents=True, exist_ok=True)
        free_b = nvme_free if nvme_free is not None else shutil.disk_usage(p).free
        if free_b >= need_nvme:
            return p, est, "explicit"
        return None, est, f"explicit_insufficient_free({free_b}<{need_nvme})"
    s_free = shm_free if shm_free is not None else (shutil.disk_usage(shm_path).free if shm_path.exists() else 0)
    if s_free >= need_shm:
        return shm_path / "chantless" / run_dir.name / "tier_cache" / "sanity", est, "shm"
    n_free = nvme_free if nvme_free is not None else shutil.disk_usage(run_dir).free
    if n_free >= need_nvme:
        return run_dir / "tier_cache" / "sanity", est, "nvme"
    return None, est, f"insufficient_disk(shm={s_free},nvme={n_free},need={need_nvme})"


class SplitGuard:
    def __init__(self, tasks):
        self.tasks = tasks
        self.phase = None

    def get(self, split: str):
        if split == "exam" and self.phase != "exam":
            raise PermissionError(f"exam split is sealed during phase {self.phase!r}")
        return [t for t in self.tasks if t.split == split]


class Run:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.dir = Path(cfg["run_root"]) / cfg["run_id"]
        self.dir.mkdir(parents=True, exist_ok=True)
        self.log = Log(self.dir / "log.md")
        self.tc = Toolchain.from_config(cfg["toolchain"])
        self.env = load_environment(cfg["env"]["module"], **cfg["env"].get("kwargs", {}))
        self.guard = SplitGuard(self.env.tasks())
        self.by_id = {t.id: t for t in self.env.tasks()}
        self.src = cfg["source"]["gguf"]
        sr_cfg = cfg.get("scratch_root")
        probe_cfg = cfg.get("probe", {})
        src_sz = Path(self.src).stat().st_size if Path(self.src).exists() else 0
        self.scratch, should_stage = choose_scratch_layout(
            sr_cfg, src_sz, self.dir, cfg["run_id"],
            auto_shm=bool(probe_cfg.get("auto_shm", False)),
            stage_source=probe_cfg.get("stage_source"),
        )
        if should_stage and self.scratch != self.dir:
            staged = self.scratch / "source" / Path(self.src).name
            if not staged.exists() or staged.stat().st_size != Path(self.src).stat().st_size:
                staged.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(self.src, staged)
            self.src = str(staged)
            pin = self.dir / "run.json"
            if pin.exists() and sha256_file(self.src) != read_json(pin)["pins"]["source_gguf_sha256"]:
                raise RuntimeError("staged source GGUF does not match the pinned sha256")
        if self.scratch != self.dir:
            self.log(f"scratch: {self.scratch} (stage_source={should_stage})")
        self.inv = gguf_ops.inventory(self.src)
        self.groups, self.dead_tensors = gguf_ops.adapter_from_gguf(self.src, cfg["source"].get("hf_dir"))
        self.probe_src = self.src
        src0 = self.scratch / "source" / "src0-nodead.gguf"
        if self.dead_tensors and src0.exists():
            self.probe_src = str(src0)
        imx = Path(cfg["run_root"]) / cfg["run_id"] / "proxy" / "imatrix.gguf"
        if imx.exists():
            self.annotate_imatrix(imx)
        self.info = gguf_ops.arch_info(self.src)
        self.ladder = cfg["ladder"]
        self.mode = cfg.get("objective_mode", "preserve")
        self.dep = self.env.deployment
        # classifier environments (CLASSIFIER-CONTRACT.md) score documents in batches and judge each model at
        # its own calibrated rule; every other environment keeps the episode path
        self.classifier = getattr(self.env, "classifier", False)
        self.scorer = None
        if self.classifier and cfg["sanity"].get("behavioral_items", 0):
            raise ValueError("classifier environments have no chat servers: set sanity.behavioral_items to 0")
        if self.classifier:
            from .runtime import BatchScorer
            self.scorer = BatchScorer(self.tc, cfg.get("scoring", {}).get("slots_per_gpu", 2))
            self.ev = Evaluator(self.tc, self.env, cfg["decoding"], cfg["serving"]["slots_per_gpu"],
                                cfg["serving"]["ctx_per_slot"], self.dep["kv_type"], self.dir / "episodes", scorer=self.scorer)
            if cfg.get("scoring", {}).get("seed"):
                self._seed_source_scores(cfg["scoring"]["seed"])
        else:
            self.ev = Evaluator(self.tc, self.env, cfg["decoding"], cfg["serving"]["slots_per_gpu"],
                                cfg["serving"]["ctx_per_slot"], self.dep["kv_type"], self.dir / "episodes")
        self.t0 = time.time()

    def _seed_source_scores(self, sd: dict):
        """Reuse the source's recorded scores from an earlier run of the same source file and tokenisation
        (scoring.seed: {"files": [{"prefix", "npy", "ids"}], "check": n}). A random sample is rescored first;
        any disagreement beyond 1e-3 in log-odds refuses the whole seed and the run scores fresh."""
        import numpy as np
        rec = {}
        for f in sd["files"]:
            ids = [m["id"] for m in json.loads(Path(f["ids"]).read_text())]
            vals = np.load(f["npy"])
            assert len(ids) == len(vals), f["npy"]
            rec.update({f"{f['prefix']}:{i}": float(v) for i, v in zip(ids, vals) if f"{f['prefix']}:{i}" in self.by_id})
        sample = random.Random(20261030).sample(sorted(rec), min(sd.get("check", 64), len(rec)))
        fresh = self.env.score([self.by_id[i] for i in sample], self.src, self.scorer)
        worst = max(abs(fresh[i] - rec[i]) for i in sample)
        if worst > 1e-3:   # another backend or build: score fresh instead
            self.log(f"seed refused: recorded source scores disagree with a fresh pass by {worst:.2e}; scoring fresh")
            return
        self.ev.seed(self.src, rec)
        self.log(f"seeded {len(rec):,} source scores from earlier receipts; {len(sample)} rescored, worst |diff| {worst:.1e}")

    def annotate_imatrix(self, path):
        names = {t["name"].rsplit(".", 1)[0] for t in gguf_ops.inventory(path)}
        for g, info in self.groups.items():
            if not g.startswith("_"):
                info["imatrix_ok"] = all(n in names for n in info["quantizable"])

    def uniform_mask(self, base, tier):
        return {**base, "groups": {g: {"tier": mk.clip(self.groups[g], tier, self.ladder)} for g in base["groups"]}}

    # ------------------------------------------------------------------ helpers
    def done(self, phase):
        return (self.dir / "phases" / f"{phase}.done").exists()

    def mark(self, phase, info=None):
        write_json(self.dir / "phases" / f"{phase}.done", {"t": time.time(), **(info or {})})

    def record(self, kind, row):
        append_jsonl(self.dir / f"{kind}.jsonl", {"t": round(time.time() - self.t0, 1), **row})

    def seeds(self, tag, task_id, k):
        return [stable_seed(self.cfg["run_id"], tag, task_id, i) for i in range(k)]

    def jobs(self, tag, tasks, k):
        return [(t, s) for t in tasks for s in self.seeds(tag, t.id, k)]

    def spread(self, tag, tasks, n, max_per_task):
        """n episodes spread evenly over tasks (equal episodes per task), <= max_per_task each."""
        k = min(max_per_task, max(1, math.ceil(n / max(1, len(tasks)))))
        return self.jobs(tag, tasks, k)

    def run_phases(self, upto: str | None = None):
        for p in PHASES:
            self.guard.phase = p
            if self.done(p):
                continue
            self.log(f"== phase {p}")
            getattr(self, "phase_" + p)()
            if upto and p == upto:
                break

    # ------------------------------------------------------------------ prepare
    def phase_prepare(self):
        c = self.cfg
        lc = Path(c["toolchain"]["llama_bin"]).parents[1]
        commit = subprocess.run(["git", "-C", str(lc), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        diff = subprocess.run(["git", "-C", str(lc), "diff"], capture_output=True, text=True).stdout
        run = {"run_id": c["run_id"], "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "config": c,
               "pins": {"source_repo": c["source"]["repo"], "source_revision": c["source"]["revision"],
                        "source_gguf_sha256": sha256_file(self.src), "llama_cpp_commit": commit,
                        "llama_cpp_patch_sha256": __import__("hashlib").sha256(diff.encode()).hexdigest(),
                        "traj_kl_sha256": sha256_file(c["toolchain"]["traj_kl"]),
                        "env": {"name": self.env.name, "version": self.env.version, "kwargs": c["env"].get("kwargs", {})}},
               "deployment": self.dep, "budget": self.env.budget}
        for k in ("source_revision",):
            if not run["pins"][k]:
                raise ValueError(f"unpinned input: {k}")
        exp = c["source"].get("expected_gguf_sha256")
        if exp and exp != run["pins"]["source_gguf_sha256"]:
            raise ValueError(f"source GGUF sha256 {run['pins']['source_gguf_sha256']} != pinned {exp}")
        (self.dir / "llama-cpp.patch").write_text(diff)
        write_json(self.dir / "run.json", run)
        splits = {}
        for t in self.env.tasks():
            splits.setdefault(t.split, {}).setdefault(t.family, 0)
            splits[t.split][t.family] += 1
        write_json(self.dir / "splits.json", splits)
        write_json(self.dir / "groups.json", self.groups)
        if self.dead_tensors:
            write_json(self.dir / "dead.json", self.dead_tensors)
            self._prepass_dead_tensors()
        self.mark("prepare")

    def _prepass_dead_tensors(self):
        """Lossless pre-pass: remove dead shared-KV tensors once and verify scores do not move."""
        src0 = self.scratch / "source" / "src0-nodead.gguf"
        src0.parent.mkdir(parents=True, exist_ok=True)
        if not src0.exists():
            gguf_ops.copy_without(self.src, src0, set(self.dead_tensors))
        if self.classifier and self.scorer is not None:
            sample = self.guard.get("search")[:32]
            if sample:
                s_full = self.env.score(sample, self.src, self.scorer)
                s_cut = self.env.score(sample, str(src0), self.scorer)
                worst = max(abs(s_full[k] - s_cut[k]) for k in s_full)
                if worst > 1e-5:
                    raise RuntimeError(f"dead-tensor pre-pass moved scores by {worst:.2e}; aborting lossless removal")
                self.log(f"pre-pass: removed {len(self.dead_tensors)} dead tensors losslessly (max |diff|={worst:.2e})")
        self.probe_src = str(src0)

    # ------------------------------------------------------------------ pool (reliability screen, source only)
    def phase_pool(self):
        pc = self.cfg["pool"]
        pool, rng = {}, random.Random(stable_seed(self.cfg["run_id"], "pool"))
        jobs = []
        for split, n in pc["tasks_per_split"].items():
            if split == "exam":  # exam reliability is measured inside the single-use exam phase
                continue
            ts = self.guard.get(split)
            rng.shuffle(ts)
            jobs += self.jobs("pool", ts[:n], pc["runs"])
        rows = ok_rows(self.ev.run(self.src, jobs, "pool", side="src"))
        per = {}
        for r in rows:
            per.setdefault(r["task"], []).append(r["success"])
        for tid, xs in per.items():
            t = self.by_id[tid]
            pool.setdefault(t.split, {})[tid] = {"n": len(xs), "k": sum(xs)}
        reliable = {s: sorted(t for t, v in d.items() if v["n"] >= pc["runs"] - 1 and v["k"] / v["n"] >= pc["min_pass"] / pc["runs"])
                    for s, d in pool.items()}
        out = {"criterion": f">= {pc['min_pass']}/{pc['runs']}", "per_task": pool, "reliable": reliable,
               "sizes": {s: {"screened": len(pool[s]), "reliable": len(reliable[s])} for s in pool},
               "pass_rate": sum(r["success"] for r in rows) / max(1, len(rows))}
        write_json(self.dir / "pool.json", out)
        self.log(f"pool: {out['sizes']} pass_rate={out['pass_rate']:.3f}")
        self.mark("pool", out["sizes"])

    def reliable(self, split):
        return [self.by_id[t] for t in read_json(self.dir / "pool.json")["reliable"].get(split, [])]

    # ------------------------------------------------------------------ A/A calibration + gate sizing
    def phase_aa(self):
        st = self.cfg["stats"]
        sel = self.reliable("select")
        n_aa = st["n_aa"]
        with self.ev.servers(self.src) as srvs:
            a = ok_rows(self.ev.run(self.src, self.spread("aa-A", sel, n_aa, st["max_eps_per_task"]), "aa-A", "src", srvs=srvs))
            b = ok_rows(self.ev.run(self.src, self.spread("aa-B", sel, n_aa, st["max_eps_per_task"]), "aa-B", "src", srvs=srvs))
            beh_items = sanity.behavioral_items(self.cfg["sanity"]["behavioral_items"], 7)
            bank = sanity.run_behavioral(srvs, self.cfg["decoding"], beh_items, list(range(self.cfg["sanity"]["behavioral_seeds"])), "src")
        rows = [dict(r, side="cand") for r in a] + [dict(r, side="src") for r in b]
        d = stats.discordance(rows)
        bnd = stats.delta_and_bounds(rows, st["alpha"])
        v = max(d["v"], st.get("v_min", 0.0)) if not math.isnan(d["v"]) else st.get("v_min", 0.18)
        deff = max(1.0, d["deff"]) if not math.isnan(d["deff"]) else 1.0   # never size below iid
        n_fam_sel = len({t.family for t in sel})
        n_req = stats.sample_size(v, st["margin"], st["alpha"], st["power"], deff, st["looks"], df=max(1, n_fam_sel - 1))
        n_gate = max(st["n_floor"], n_req) if not st.get("n_override") else st["n_override"]
        cap_sel = len(sel) * st["max_eps_per_task"]
        two = stats.delta_and_bounds(rows, 0.025)  # two-sided 95%
        # Harness bias check: zero inside the two-sided 95% interval; the point tolerance (manual 0.015)
        # applies only at gate scale, where it is meaningful.
        point_ok = abs(bnd["delta"]) <= st["aa_tol"] if d["n_pairs"] >= st["n_floor"] else None
        out = {"v": d["v"], "deff": d["deff"], "v_used": v, "deff_used": deff, "aa_delta": bnd["delta"], "aa_bounds": bnd,
               "aa_two_sided": {"L": two["L"], "U": two["U"]}, "aa_point_ok": point_ok, "n_pairs": d["n_pairs"],
               "aa_ok": (two["L"] <= 0 <= two["U"]) and point_ok is not False, "n_required": n_req, "n_gate": n_gate,
               "select_capacity": cap_sel, "pool_rule_ok": cap_sel >= n_gate, "n_select_reliable": len(sel),
               "source_rate_A": sum(r["success"] for r in a) / max(1, len(a)), "source_rate_B": sum(r["success"] for r in b) / max(1, len(b)),
               "behavioral_source_rate": sum(bool(r["success"]) for r in bank) / max(1, len(bank))}
        write_json(self.dir / "aa.json", out)
        write_json(self.dir / "behavioral_bank.json", bank)
        self.log(f"A/A: delta={bnd['delta']:+.4f} v={d['v']:.4f} deff={deff:.2f} n_gate={n_gate} capacity={cap_sel} aa_ok={out['aa_ok']}")
        if not out["aa_ok"] and st.get("aa_replication"):
            # pre-registered single replication with fresh seed sets, run in reverse order (order-effect check);
            # proceed only if the replication passes; both results are kept.
            self.log("A/A failed; running the pre-registered replication (sets D then C)")
            with self.ev.servers(self.src) as srvs:
                d_ = ok_rows(self.ev.run(self.src, self.spread("aa-D", sel, n_aa, st["max_eps_per_task"]), "aa-D", "src", srvs=srvs))
                c_ = ok_rows(self.ev.run(self.src, self.spread("aa-C", sel, n_aa, st["max_eps_per_task"]), "aa-C", "src", srvs=srvs))
            rr = [dict(r, side="cand") for r in c_] + [dict(r, side="src") for r in d_]
            t2 = stats.delta_and_bounds(rr, 0.025)
            pt2 = abs(t2["delta"]) <= st["aa_tol"] if len(c_) >= st["n_floor"] else None
            rep = {"delta": t2["delta"], "L": t2["L"], "U": t2["U"], "point_ok": pt2, "n_pairs": min(len(c_), len(d_)),
                   "rate_C": sum(r["success"] for r in c_) / max(1, len(c_)), "rate_D": sum(r["success"] for r in d_) / max(1, len(d_))}
            rep["ok"] = (t2["L"] <= 0 <= t2["U"]) and pt2 is not False
            out["replication"] = rep
            out["aa_ok_after_replication"] = rep["ok"]
            self.log(f"A/A replication: delta(C-D)={t2['delta']:+.4f} [{t2['L']:+.4f},{t2['U']:+.4f}] ok={rep['ok']}")
            write_json(self.dir / "aa.json", out)
            if rep["ok"]:
                out["aa_ok"] = True
        if not out["aa_ok"]:
            if st.get("aa_enforce", True):
                raise RuntimeError("A/A check failed: harness, seeding or task isolation is biased; fix before search")
            self.log("WARNING: A/A check failed but aa_enforce=false (smoke config); recorded, not enforced")
        if not out["pool_rule_ok"] and not st.get("allow_small_pool"):
            raise RuntimeError(f"pool-size rule failed: select capacity {cap_sel} < n_gate {n_gate}")
        self.mark("aa", {"n_gate": n_gate})

    # ------------------------------------------------------------------ proxy stores + imatrix
    def phase_proxy(self):
        if self.classifier:
            return self._proxy_classifier()
        pc = self.cfg["proxy"]
        R = kl.Renderer(self.cfg["source"]["hf_dir"])
        pdir = self.dir / "proxy"
        pdir.mkdir(exist_ok=True)
        pool = read_json(self.dir / "pool.json")
        rel_search = set(pool["reliable"].get("search", []))
        recs, stats_text = [], []
        for r in read_jsonl(self.dir / "episodes" / "pool.jsonl"):
            if r["split"] == "search" and r["success"] and r["task"] in rel_search:
                try:
                    rr = R.episode(r["messages"], self._task_tools(r.get("task")))
                except kl.RenderError as e:
                    self.log(f"proxy: reject {r['task']}: {e}")
                    continue
                recs += rr
                stats_text.append(R.tok.decode(rr[-1][0]))
        recs = kl.subsample(recs, pc["task_max_targets"], 20261012)
        meta_t = kl.write_trj(pdir / "task.bin", recs)
        texts = sanity.general_texts(pc["general_docs"], 20261012, pdir / "general_texts.json")
        grecs = kl.pack_text(R, texts, pc["general_seq_len"], pc["general_n_seq"])
        meta_g = kl.write_trj(pdir / "general.bin", grecs)
        for name in ("task", "general"):
            kl.traj_kl(self.tc, "base", self.src, str(pdir / f"{name}.bin"), str(pdir / f"store_{name}"), str(pdir / f"base_{name}.json"), str(pdir / "kl.log"), gpu=self.tc.gpus[0])
            s = kl.traj_kl(self.tc, "kl", self.src, str(pdir / f"{name}.bin"), str(pdir / f"store_{name}"), str(pdir / f"self_{name}.json"), str(pdir / "kl.log"), gpu=self.tc.gpus[0])
            tol = pc.get("self_kl_tol", {}).get(name, 1e-5)
            if s["kl"] > tol:
                raise RuntimeError(f"self-KL check failed for {name}: {s['kl']} > {tol}")
        (pdir / "stats_corpus.txt").write_text("\n\n".join(stats_text + texts[: pc["general_docs"] // 2]))
        run_imatrix(self.tc, self.probe_src, str(pdir / "stats_corpus.txt"), str(pdir / "imatrix.gguf"), str(pdir / "imatrix.log"), gpu=self.tc.gpus[0])
        self.annotate_imatrix(pdir / "imatrix.gguf")
        write_json(self.dir / "groups.json", self.groups)
        out = {"task": meta_t, "general": meta_g, "self_kl": {n: read_json(pdir / f"self_{n}.json")["kl"] for n in ("task", "general")},
               "imatrix_sha256": sha256_file(pdir / "imatrix.gguf")}
        write_json(pdir / "proxy.json", out)
        self.log(f"proxy: task {meta_t} general {meta_g} self-KL {out['self_kl']}")
        self.mark("proxy")

    def _proxy_classifier(self):
        """Classifier proxy: a log-odds store of the parts env.proxy_spec picks (random calibration humans, AI
        documents near the source's cutoff), scored by the configured objective; plus the usual general store."""
        pc = self.cfg["proxy"]
        pdir = self.dir / "proxy"
        pdir.mkdir(exist_ok=True)
        by_split = lambda sp: [t for t in self.env.tasks() if t.split == sp]
        src_scores = self.ev.scores(self.src, by_split("calibration") + by_split("search"))
        parts = self.env.proxy_spec(src_scores, **pc.get("spec", {}))
        tasks = [self.by_id[i] for p in parts for i in p["ids"]]
        trj = pdir / "task.bin"
        shutil.copyfile(self.env.trj_for(tasks), trj)
        sdir = pdir / "store_task"
        sdir.mkdir(exist_ok=True)
        import numpy as np
        from . import objectives
        src = np.asarray(self.scorer.score_trj(self.src, str(trj), tuple(self.env.pair)), dtype=float)
        np.save(sdir / "source.npy", src)
        again = np.asarray(self.scorer.score_trj(self.src, str(trj), tuple(self.env.pair)), dtype=float)
        meta_task = {"kind": "logodds", "n_seq": len(tasks), "n_targets": len(tasks),
                     "meta": {"parts": [{"name": p["name"], "role": p["role"], "n": p["n"]} for p in parts],
                              "pair": list(self.env.pair), "objective": pc.get("objective", "pauc"),
                              "objective_args": pc.get("objective_args", {})}}
        split = lambda x: {r: np.concatenate([x[sum(q["n"] for q in parts[:k]):][: p["n"]] for k, p in enumerate(parts)
                                              if p["role"] == r]) for r in ("pos", "neg")}
        self_damage = abs(objectives.OBJECTIVES[meta_task["meta"]["objective"]](split(again), split(src),
                                                                                **meta_task["meta"]["objective_args"]))
        write_json(pdir / "parts.json", parts)
        R = kl.Renderer(self.cfg["source"]["hf_dir"])
        texts = sanity.general_texts(pc["general_docs"], 20261012, pdir / "general_texts.json")
        meta_g = kl.write_trj(pdir / "general.bin", kl.pack_text(R, texts, pc["general_seq_len"], pc["general_n_seq"]))
        kl.traj_kl(self.tc, "base", self.src, str(pdir / "general.bin"), str(pdir / "store_general"), str(pdir / "base_general.json"), str(pdir / "kl.log"), gpu=self.tc.gpus[0])
        sg = kl.traj_kl(self.tc, "kl", self.src, str(pdir / "general.bin"), str(pdir / "store_general"), str(pdir / "self_general.json"), str(pdir / "kl.log"), gpu=self.tc.gpus[0])
        tol = pc.get("self_kl_tol", {})
        if self_damage > tol.get("task", 1e-5) or sg["kl"] > tol.get("general", 1e-4):
            raise RuntimeError(f"self-check failed: task damage {self_damage}, general KL {sg['kl']}")
        (pdir / "stats_corpus.txt").write_text("\n\n".join(self.env.stats_texts(pc.get("stats_docs", 200)) + texts[: pc["general_docs"] // 2]))
        run_imatrix(self.tc, self.probe_src, str(pdir / "stats_corpus.txt"), str(pdir / "imatrix.gguf"), str(pdir / "imatrix.log"), gpu=self.tc.gpus[0])
        self.annotate_imatrix(pdir / "imatrix.gguf")
        write_json(self.dir / "groups.json", self.groups)
        out = {"task": meta_task, "general": meta_g, "self_kl": {"task": self_damage, "general": sg["kl"]},
               "imatrix_sha256": sha256_file(pdir / "imatrix.gguf")}
        write_json(pdir / "proxy.json", out)
        self.log(f"proxy (classifier): {[(p['name'], p['role'], p['n']) for p in parts]}; self-damage {self_damage:.2e}, general self-KL {sg['kl']:.2e}")
        self.mark("proxy")

    def make_prober(self, weights: str, tag: str) -> Prober:
        pdir = self.dir / "proxy"
        meta = read_json(pdir / "proxy.json")
        b_dir = self.scratch / "build" / tag
        c_dir = self.scratch / "cand" / tag
        shutil.rmtree(b_dir, ignore_errors=True)
        shutil.rmtree(c_dir, ignore_errors=True)
        B = bmod.Builder(self.tc, weights, self.groups, str(pdir / "imatrix.gguf"), b_dir, log=str(self.dir / "build.log"))
        probe_cfg = self.cfg.get("probe", {})
        B.threads = probe_cfg.get("quantize_threads", 8)
        B.max_pruned = probe_cfg.get("max_pruned", 1 if str(self.scratch).startswith("/dev/shm") else 4)
        req_workers = probe_cfg.get("workers")
        clamp_enabled = probe_cfg.get("clamp_workers", True)
        w_sz = Path(weights).stat().st_size if Path(weights).exists() else 0
        workers, cap, free_b = clamp_probe_workers(
            req_workers, self.scratch, w_sz, n_gpus=max(1, len(self.tc.gpus)), clamp_enabled=clamp_enabled,
        )
        if req_workers is not None and workers < req_workers:
            self.log(f"probe({tag}): clamped workers {req_workers} -> {workers} (cap={cap}, free={free_b / (1 << 30):.1f} GiB)")
        P = Prober(self.tc, B, self.dir / "probe" / tag, str(self.dir / "probe.log"), workers=workers,
                   scratch=c_dir, scorer=self.scorer)
        if self.cfg.get("probe", {}).get("slots_per_gpu"):   # several concurrent probes per GPU
            P._free = [g for g in self.tc.gpus for _ in range(self.cfg["probe"]["slots_per_gpu"])]
        P.add_store(Store("task", str(pdir / "task.bin"), self._staged(pdir / "store_task"), meta["task"]["n_targets"],
                          kind=meta["task"].get("kind", "kl"), meta=meta["task"].get("meta", {})))
        P.add_store(Store("general", str(pdir / "general.bin"), self._staged(pdir / "store_general"), meta["general"]["n_targets"]))
        for fb in read_jsonl(self.dir / "feedback_stores.jsonl"):
            P.add_store(Store(fb["name"], fb["trj"], fb["dir"], fb["n_targets"], fb.get("weight", 1.0),
                              kind=fb.get("kind", "kl"), meta=fb.get("meta", {})))
        return P

    def _staged(self, store_dir) -> str:
        """Reference stores are read on every probe; keep a verified copy on scratch storage (fast, unencrypted)."""
        store_dir = Path(store_dir)
        if self.scratch == self.dir:
            return str(store_dir)
        dst = self.scratch / "stores" / store_dir.relative_to(self.dir)
        src_files = sorted(store_dir.glob("*.f16"))
        if not dst.exists() or sorted(p.name for p in dst.glob("*.f16")) != [p.name for p in src_files] or \
                any((dst / p.name).stat().st_size != p.stat().st_size for p in src_files):
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(store_dir, dst)
        return str(dst)

    # ------------------------------------------------------------------ sanity: cap curve, floors, initial mask
    def phase_sanity(self):
        sc = self.cfg["sanity"]
        probe_cfg = self.cfg.get("probe", {})
        P = self.make_prober(self.probe_src, "sanity")
        src_bytes = gguf_ops.file_bytes(self.probe_src)
        base = mk.new_mask(self.groups, "BF16", self._source_pin(), self.mode)
        uniform = {}
        um = {t: self.uniform_mask(base, t) for t in self.ladder[1:]}

        solo, masks, keys = {}, [], []
        tiers_all = sc.get("solo_tiers") or self.ladder[1:]
        for g in base["groups"]:
            for t in [x for x in tiers_all if mk.tier_allowed(self.groups[g], x)] + (["drop"] if self.groups[g]["droppable"] else []):
                masks.append(mk.apply(base, g, t))
                keys.append((g, t))
        fkey = sc.get("floor_key", "kl_gen")
        cap_off = sc.get("cap", "general_kl") == "off"
        if fkey == "task" and cap_off:
            want_solo = ["task"]
        elif fkey == "task" or self.cfg.get("learner", {}).get("algo") == "oneshot":
            want_solo = ["general", "task"]
        else:
            want_solo = ["general"]

        splice_flag = probe_cfg["splice_cache"] if "splice_cache" in probe_cfg else sc.get("splice_cache", True)
        solo_needed = bmod.reachable_tiers(masks, self.groups, self.ladder)
        splice_b = None
        if splice_flag and len(masks) >= 16 and solo_needed and Path(self.probe_src).exists():
            tc_dir, est_b, tc_where = choose_tier_cache_dir(
                self.scratch, self.dir, src_bytes, solo_needed, cfg_dir=probe_cfg.get("tier_cache_dir"),
            )
            if tc_dir is not None:
                self.log(f"sanity: splice tier_cache on {tc_where} ({tc_dir}) for {solo_needed} (~{est_b / (1 << 30):.2f} GiB)")
                splice_b = bmod.SpliceProbeBuilder.from_builder(
                    P.builder, base, solo_needed, self.ladder, tc_dir, workers=min(4, max(1, P.workers)),
                )
                P.builder = splice_b
            else:
                self.log(f"sanity: splice tier_cache disabled ({tc_where}; est ~{est_b / (1 << 30):.2f} GiB)")

        um_hits = um_fb = 0
        try:
            for t, r in zip(um, P.probe_many(list(um.values()), ["general", "task"])):
                uniform[t] = {"kl_gen": r.kl["general"], "kl_task": r.kl["task"], "bytes": r.bytes, "bpw": r.bpw, "bytes_saved": src_bytes - r.bytes}
            um_hits = splice_b.hits if splice_b is not None else 0
            um_fb = splice_b.fallbacks if splice_b is not None else 0
            pts = [(bmod_bpw(src_bytes, P.builder.source_params), 0.0)] + [(u["bpw"], u["kl_gen"]) for u in uniform.values()]
            cap = sanity.make_cap(sc.get("cap", "general_kl"), pts)
            if fkey == "task":
                for u in uniform.values():
                    u["task"] = u["kl_task"]
            selfkl = read_json(self.dir / "proxy" / "proxy.json")["self_kl"]
            ft = self.cfg["proxy"].get("floor_tier", "Q8_0")
            for name, key in (("general", "kl_gen"), ("task", "kl_task")):
                ref = uniform[ft][key]
                if self.classifier and name == "task":
                    ref = abs(ref)
                if selfkl[name] > 0.05 * ref:
                    raise RuntimeError(f"measurement floor too high for {name}: self-KL {selfkl[name]} > 5% of uniform {ft} KL {ref}")
            self.log(f"sanity: {len(masks)} solo probes")

            raw_task_tiers = probe_cfg.get("solo_task_tiers") if probe_cfg.get("solo_task_tiers") is not None else sc.get("solo_task_tiers")
            task_tiers = set(raw_task_tiers or [])
            if task_tiers and want_solo == ["general", "task"]:
                full_i = [i for i, (_g, t) in enumerate(keys) if t in task_tiers]
                gen_i = [i for i, (_g, t) in enumerate(keys) if t not in task_tiers]
                res_map = {}
                for i, r in zip(full_i, P.probe_many([masks[i] for i in full_i], ["general", "task"])):
                    res_map[i] = r
                for i, r in zip(gen_i, P.probe_many([masks[i] for i in gen_i], ["general"])):
                    res_map[i] = r
                solo_results = [res_map[i] for i in range(len(keys))]
            else:
                solo_results = P.probe_many(masks, want_solo)
        finally:
            if splice_b is not None:
                self.log(
                    f"sanity: splice_cache hits={splice_b.hits} fallbacks={splice_b.fallbacks} "
                    f"(uniform_hits={um_hits} solo_hits={splice_b.hits - um_hits} solo_fallbacks={splice_b.fallbacks - um_fb})"
                )
                splice_b.release_cache()
                P.builder = splice_b.inner

        for (g, t), r in zip(keys, solo_results):
            solo.setdefault(g, {})[t] = {"kl_gen": r.kl.get("general", 0.0), "bytes_saved": src_bytes - r.bytes}
            if "task" in r.kl:
                solo[g][t]["kl_task"] = r.kl["task"]
            if fkey == "task":
                solo[g][t]["task"] = r.kl["task"]
        floors = (sanity.floors_from_solo(solo, uniform, self.ladder, sc["floor_factor"]) if fkey == "kl_gen" else
                  sanity.floors_from_solo(solo, uniform, self.ladder, sc["floor_factor"], key=fkey))
        init = mk.new_mask(self.groups, self.cfg["start_tier"], self._source_pin(), self.mode)
        for g, f in floors.items():
            init["floors"][g] = {"min_tier": f["min_tier"], "drop_ok": f["drop_ok"]}
            if mk.rank(init["groups"][g]["tier"]) > mk.rank(f["min_tier"]):
                init["groups"][g]["tier"] = f["min_tier"]
        init["deployment"] = self.dep
        init["importance_stats"] = read_json(self.dir / "proxy" / "proxy.json")["imatrix_sha256"]
        mk.validate(init, self.groups)
        write_json(self.dir / "sanity" / "uniform.json", uniform)
        write_json(self.dir / "sanity" / "cap.json", cap.to_dict())
        write_json(self.dir / "sanity" / "floors.json", floors)
        write_json(self.dir / "sanity" / "sanity_mask.json", init)
        n_forbid = sum(1 for f in floors.values() for e in f["evidence"].values() if not e["ok"])
        self.log(f"sanity: cap {cap.to_dict().get('points', cap.to_dict())}; forbidden solo settings {n_forbid}")
        P.builder.cleanup()
        self.mark("sanity")

    def _source_pin(self):
        return {"repo": self.cfg["source"]["repo"], "revision": self.cfg["source"]["revision"], "gguf_sha256": read_json(self.dir / "run.json")["pins"]["source_gguf_sha256"]}

    # ------------------------------------------------------------------ screens and gates
    def screen(self, model: str, tag: str) -> dict:
        sc = self.cfg["screen"]
        sel = self.reliable("select")
        rng = random.Random(stable_seed(self.cfg["run_id"], "screen", tag))
        tasks = rng.sample(sel, min(sc["pairs"], len(sel)))
        src_bank = {}
        for r in read_jsonl(self.dir / "episodes" / "aa-B.jsonl"):
            if r["success"] is not None:
                src_bank.setdefault(r["task"], []).append(r)
        with self.ev.servers(model) as srvs:
            cand = ok_rows(self.ev.run(model, [(t, stable_seed(tag, t.id)) for t in tasks], f"screen-{tag}", "cand", srvs=srvs))
            beh = sanity.run_behavioral(srvs, self.cfg["decoding"], sanity.behavioral_items(self.cfg["sanity"]["behavioral_items"], 7),
                                        list(range(self.cfg["sanity"]["behavioral_seeds"])), "cand")
        rows = cand + [dict(src_bank[t.id][0], side="src") for t in tasks if t.id in src_bank]
        b = stats.delta_and_bounds(rows, 0.05, n_boot=1000)
        bank = read_json(self.dir / "behavioral_bank.json")
        beh_rate = sum(bool(r["success"]) for r in beh if r["success"] is not None) / max(1, len(beh))
        src_beh = sum(bool(r["success"]) for r in bank if r["success"] is not None) / max(1, len(bank))
        beh_ok = beh_rate >= src_beh - self.cfg["sanity"]["behavioral_margin"]
        task_ok = b["delta"] >= sc["fail_below"]
        out = {"tag": tag, "delta": b["delta"], "task_ok": task_ok, "behavioral_rate": beh_rate, "behavioral_source": src_beh,
               "behavioral_ok": beh_ok, "pass": task_ok and beh_ok, "n": len(cand)}
        # only candidate-specific failures carry feedback signal: items the source passed on the same seed
        src_ok = {(r["task"], r["seed"]) for r in bank if r.get("success")}
        out["_beh_fail"] = [r for r in beh if r["success"] is False and (r["task"], r["seed"]) in src_ok]
        self.record("screens", {k: v for k, v in out.items() if not k.startswith("_")})
        return out

    def gate(self, model: str, tag: str, split: str = "select") -> dict:
        st = self.cfg["stats"]
        n_gate = read_json(self.dir / "aa.json")["n_gate"]
        tasks = self.reliable(split) if split != "exam" else self.exam_pool
        k = min(st["max_eps_per_task"], max(1, math.ceil(n_gate / max(1, len(tasks)))))
        looks = min(st["looks"], k) if split != "exam" else 1
        rows, res = [], None
        import contextlib
        stack = contextlib.ExitStack()
        cs = stack.enter_context(self.ev.servers(model))
        sbank = self._source_bank(split, tasks)
        ss = stack.enter_context(self.ev.servers(self.src)) if sbank is None else None
        (self.dir / "episodes" / f"gate-{tag}-src.jsonl").unlink(missing_ok=True)
        for look in range(1, looks + 1):
            lo, hi = (look - 1) * k // looks, look * k // looks
            if hi <= lo:
                continue
            cj = [(t, s) for t in tasks for s in self.seeds(f"gate-{tag}-cand", t.id, k)[lo:hi]]
            sj = [(t, s) for t in tasks for s in self.seeds(f"gate-{tag}-src", t.id, k)[lo:hi]]
            rows += ok_rows(self.ev.run(model, cj, f"gate-{tag}-cand", "cand", srvs=cs))
            if sbank is None:
                rows += ok_rows(self.ev.run(self.src, sj, f"gate-{tag}-src", "src", srvs=ss))
            else:
                reused = [dict(sbank[t.id], seed=s, side="src") for t, s in sj]
                for r in reused:
                    append_jsonl(self.dir / "episodes" / f"gate-{tag}-src.jsonl", r)
                rows += reused
            res = stats.sequential_decision(rows, look, looks, st["margin"], st["alpha"], final=(look == looks))
            self.log(f"gate {tag} look {look}/{looks}: delta={res['delta']:+.4f} L={res['L']:+.4f} U={res['U']:+.4f} -> {res['decision']}")
            if res["decision"] != "continue":
                break
        det = stats.detector_bounds(rows, st["alpha"], st["detector_bound"])
        res["detectors"] = det
        res["pass"] = res["decision"] == "pass" and all(d["ok"] for d in det.values())
        beh = sanity.run_behavioral(cs, self.cfg["decoding"], sanity.behavioral_items(self.cfg["sanity"]["behavioral_items"], 7),
                                    list(range(self.cfg["sanity"]["behavioral_seeds"])), "cand")
        stack.close()
        bank = read_json(self.dir / "behavioral_bank.json")
        brows = [dict(r, side="cand") for r in beh if r["success"] is not None] + [dict(r, side="src") for r in bank if r["success"] is not None]
        bb = stats.delta_and_bounds(brows, st["alpha"], n_boot=2000)
        res["behavioral"] = bb
        res["behavioral_ok"] = (stats.empty_ok(bb, self.cfg["sanity"]["behavioral_margin"]) if self.classifier and not brows
                                else bb["L"] > -self.cfg["sanity"]["behavioral_margin"])
        res["pass"] = res["pass"] and res["behavioral_ok"]
        res.update(tag=tag, split=split, n_gate=n_gate, eps_per_task=k, n_tasks=len(tasks))
        write_json(self.dir / "gates" / f"{tag}.json", res)
        self.record("gates", {k_: v for k_, v in res.items() if k_ not in ("detectors",)})
        return res

    def _source_bank(self, split: str, tasks) -> dict | None:
        """The source's recorded result per task, when stats.reuse_source is set and every task has one.

        Only valid for a deterministic environment (detect: one greedy forward pass), where re-scoring the
        frozen source returns the same row. Select rows come from the pool and A/A episodes, exam rows from
        the exam-pool episodes."""
        if not self.cfg["stats"].get("reuse_source"):
            return None
        files = ["pool.jsonl", "aa-A.jsonl", "aa-B.jsonl"] if split != "exam" else ["exam-pool.jsonl"]
        bank = {}
        for f in files:
            p = self.dir / "episodes" / f
            for r in (read_jsonl(p) if p.exists() else []):
                if r["success"] is not None and r["split"] == split:
                    bank.setdefault(r["task"], {k: v for k, v in r.items() if k not in ("messages", "final", "verdict")})
        return bank if all(t.id in bank for t in tasks) else None

    def _exam_pool_from_scores(self, ex_tasks) -> list[dict]:
        """Exam-pool source rows from the source's recorded per-document scores (exam.source_scores), instead
        of a fresh source pass; a row counts as success when the label at the environment's cutoff is right."""
        import glob
        sc = {}
        for f in glob.glob(str(Path(self.cfg["exam"]["source_scores"]) / "*.jsonl")):
            for r in read_jsonl(f):
                sc[r["id"]] = r
        rows = []
        out = self.dir / "episodes" / "exam-pool.jsonl"
        out.unlink(missing_ok=True)
        for t in ex_tasks:
            r = sc[t.id.split(":", 1)[1]]
            assert r["label"] == t.meta["label"], t.id
            pred = "ai" if r["p_human"] < self.env.cutoff else "human"
            row = {"task": t.id, "family": t.family, "split": t.split, "seed": 0, "side": "src", "success": pred == r["label"],
                   "tests_passed": pred == r["label"], "detectors": {}, "infra_error": None, "turns": 1, "tokens": 1, "wall_s": 0.0,
                   "p_human": r["p_human"], "reused_from": self.cfg["exam"]["source_scores"]}
            append_jsonl(out, row)
            rows.append(row)
        return rows

    # ------------------------------------------------------------------ search
    def _failure_store(self, eps_rows: list[dict], name: str) -> dict | None:
        """Render failed trajectories, compute the frozen-source reference on them, register as a store."""
        if self.classifier:
            return self._failure_store_classifier(eps_rows, name)
        R = kl.Renderer(self.cfg["source"]["hf_dir"])
        recs = []
        for r in eps_rows:
            try:
                t_tools = self._task_tools(r.get("task")) if r.get("family") in self._task_families() else None
                recs += R.episode(r["messages"], t_tools)
            except kl.RenderError:
                continue
        if not recs:
            return None
        cap_t = read_json(self.dir / "proxy" / "proxy.json")["task"]["n_targets"]
        recs = kl.subsample(recs, max(500, cap_t // 4), stable_seed(name))
        d = self.scratch / "proxy" / name   # failure stores are written straight to scratch storage
        d.mkdir(parents=True, exist_ok=True)
        meta = kl.write_trj(d / "fb.bin", recs)
        kl.traj_kl(self.tc, "base", self.src, str(d / "fb.bin"), str(d / "store"), str(d / "base.json"), str(d / "kl.log"), gpu=self.tc.gpus[0])
        row = {"name": name, "trj": str(d / "fb.bin"), "dir": str(d / "store"), "n_targets": meta["n_targets"], "weight": 1.0}
        append_jsonl(self.dir / "feedback_stores.jsonl", row)
        return row

    def _failure_store_classifier(self, eps_rows: list[dict], name: str) -> dict | None:
        """Classifier failure store: the proxy's objective on the failed documents. Failed positives (missed AI)
        become the pos part, or the proxy's own pos part if none failed; failed negatives join the proxy's
        random human sample, where the objective's own-tail choice will count them when they are hard."""
        import numpy as np
        parts = read_json(self.dir / "proxy" / "parts.json")
        failed = [self.by_id[r["task"]] for r in eps_rows if r.get("task") in self.by_id]
        if not failed:
            return None
        rule = self.ev.rule(self.src)
        fpos = [t.id for t in failed if self.env.decide(t, -1e9, rule).tests_passed]   # a flag is right: a positive
        fneg = [t.id for t in failed if t.id not in set(fpos)]
        neg = next(p for p in parts if p["role"] == "neg")["ids"] + fneg
        pos = fpos or next(p for p in parts if p["role"] == "pos")["ids"]
        tasks = [self.by_id[i] for i in neg + pos]
        d = self.scratch / "proxy" / name
        d.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(self.env.trj_for(tasks), d / "fb.bin")
        (d / "store").mkdir(exist_ok=True)
        np.save(d / "store" / "source.npy", np.asarray(self.scorer.score_trj(self.src, str(d / "fb.bin"), tuple(self.env.pair)), dtype=float))
        pm = read_json(self.dir / "proxy" / "proxy.json")["task"]["meta"]
        meta = {**pm, "parts": [{"name": "neg", "role": "neg", "n": len(neg)}, {"name": "pos", "role": "pos", "n": len(pos)}]}
        row = {"name": name, "trj": str(d / "fb.bin"), "dir": str(d / "store"), "n_targets": len(tasks), "weight": 1.0,
               "kind": "logodds", "meta": meta}
        append_jsonl(self.dir / "feedback_stores.jsonl", row)
        return row

    def _task_families(self):
        return {t.family for t in self.env.tasks()}

    def _task_tools(self, task_id: str | None):
        if task_id and task_id in self.by_id:
            p = self.by_id[task_id].prompt
            if isinstance(p, dict) and "tools" in p:
                return p["tools"]
        return _tools(self.env)

    def phase_search(self):
        lc = self.cfg["learner"]
        init = read_json(self.dir / "sanity" / "sanity_mask.json")
        cd = read_json(self.dir / "sanity" / "cap.json")
        cap = sanity.make_cap("off") if cd.get("kind") == "off" else sanity.SizeMatchedCap([tuple(p) for p in cd["points"]])
        P = self.make_prober(self.probe_src, "search")
        L = Learner(P, self.groups, self.ladder, cap, self.mode, lc, self.log, self.record)
        S = LearnerState(mask=init)
        last_gate, last_screen = None, init      # source's gate on record is the A/A
        algo = lc.get("algo", "greedy")
        if algo == "oneshot":
            S.mask, stop_reason = self._search_oneshot(P, init, lc, cap=cap), "oneshot"
        elif algo == "greedy":
            base = P.probe(S.mask)
            self.record("history", {"step": 0, "mask_key": base.key, "bytes": base.bytes, "bpw": base.bpw, "kl": base.kl, "event": "start"})
            L.full_reprobe(S, base)
            fb_i, consecutive_fail, deadline = 0, 0, self.t0 + 3600 * self.cfg["budget"]["max_hours"]
            stop_reason = "exhausted"
            exch, n_exch = False, 0   # exchange phase (learner.exchange_rounds > 0): at the budget, one up + one down per round
            while True:
                if time.time() > deadline:
                    stop_reason = "budget"
                    break
                if S.accepted >= lc.get("max_moves", 10 ** 9):
                    stop_reason = "max_moves"
                    break
                if base.bytes <= lc.get("target_bytes", 0) and not exch:
                    if lc.get("exchange_rounds", 0) <= 0:
                        stop_reason = "target_bytes"
                        break
                    exch = True
                if exch:
                    if n_exch >= lc["exchange_rounds"]:
                        stop_reason = "exchange_rounds_done"
                        break
                    n_exch += 1
                    best = L.exchange(S, base, lc["target_bytes"])
                    if best is None:
                        stop_reason = "exchange_converged"
                        break
                    label = " + ".join(m.key for m in best["pair"] if m)
                else:
                    best = L.step(S, base)
                    if best is None:
                        stop_reason = "exhausted"
                        break
                    label = best["move"].key
                base = best["res"]
                self.record("history", {"step": S.accepted, "move": label, "bytes": base.bytes, "bpw": base.bpw, "kl": base.kl,
                                        "score": best["score"], "event": "exchange" if exch else "accept"})
                self.log(f"{'exchange' if exch else 'move'} {S.accepted}: {label} bytes={base.bytes} bpw={base.bpw:.3f} kl_task={base.kl['task']:.5f} kl_gen={base.kl['general']:.5f}")
                if S.accepted % lc["screen_every"] == 0 or exch:
                    cand = P.builder.build(S.mask, self.scratch / "build" / "screen.gguf").path
                    sc = self.screen(cand, f"m{S.accepted:04d}")
                    # failure feedback from search-split episodes (never the select split)
                    fb_tasks = random.Random(stable_seed("fb", S.accepted)).sample(self.reliable("search"), min(lc["feedback_tasks"], len(self.reliable("search"))))
                    fb_rows = ok_rows(self.ev.run(cand, [(t, stable_seed("fb", S.accepted, t.id)) for t in fb_tasks], f"fb-{S.accepted:04d}", "cand"))
                    fails = [r for r in read_jsonl(self.dir / "episodes" / f"fb-{S.accepted:04d}.jsonl") if r["success"] is False]
                    fails += sc["_beh_fail"]
                    self.log(f"screen m{S.accepted}: delta={sc['delta']:+.3f} beh={sc['behavioral_rate']:.3f}/{sc['behavioral_source']:.3f} pass={sc['pass']}; feedback failures={len(fails)}/{len(fb_rows) + len(sc['_beh_fail'])}")
                    if lc.get("feedback", True) and fails:
                        fb_i += 1
                        st_row = self._failure_store(fails, f"fb{fb_i:03d}")
                        if st_row:
                            P.add_store(Store(st_row["name"], st_row["trj"], st_row["dir"], st_row["n_targets"],
                                              kind=st_row.get("kind", "kl"), meta=st_row.get("meta", {})))
                            L.feedback(S, st_row["name"], base, revert=not sc["pass"])
                    base = P.probe(S.mask)
                    if not sc["pass"]:
                        consecutive_fail += 1
                        if consecutive_fail >= 2 or not lc.get("feedback", True):
                            S.mask = last_screen
                            stop_reason = "screen_failed"
                            break
                        cand = P.builder.build(S.mask, self.scratch / "build" / "screen.gguf").path
                        sc2 = self.screen(cand, f"m{S.accepted:04d}-after-feedback")
                        if not sc2["pass"]:
                            S.mask = last_screen
                            stop_reason = "screen_failed_after_feedback"
                            break
                    consecutive_fail = 0
                    last_screen = S.mask
                    if lc.get("reprobe_after_screen", True):
                        L.full_reprobe(S, base)
                    if S.accepted % lc["gate_every"] == 0:
                        g = self.gate(P.builder.build(S.mask, self.scratch / "build" / "gate.gguf").path, f"m{S.accepted:04d}")
                        if not g["pass"]:
                            S.mask = last_gate or init
                            stop_reason = "gate_failed"
                            break
                        last_gate = S.mask
        else:
            raise ValueError(f"learner.algo must be greedy or oneshot, got {algo!r}")
        self.log(f"search stopped: {stop_reason} after {S.accepted} accepted moves")
        # optional weight repair on the selected mask
        final_weights, lineage = self.src, []
        if self.cfg.get("repair", {}).get("enabled"):
            final_weights, lineage, S.mask = self._repair(S.mask, P)
        cand_mask = S.mask
        sel_mask = cand_mask
        if last_gate is None or bmod.mask_key(last_gate) != bmod.mask_key(cand_mask) or lineage:
            gb = bmod.Builder(self.tc, final_weights, self.groups, str(self.dir / "proxy" / "imatrix.gguf"), self.scratch / "build" / "final-gate")
            g = self.gate(gb.build(cand_mask, self.scratch / "build" / "final-gate.gguf").path, "final")
            gated = g["pass"]
            if not g["pass"]:
                sel_mask = last_gate or init
                final_weights, lineage = self.src, []   # the gated incumbent was built from the source weights
                gated = last_gate is not None
                self.log("final gate failed: returning last gate-passing mask")
                if last_gate is None:
                    self.log("no searched mask passed a gate; the returned mask is the initial sanity mask, which has not passed a gate")
        else:
            gated = True
        sel_mask["weights"] = {"lineage": lineage, "path": final_weights if lineage else None}
        write_json(self.dir / "search_result.json", {"stop_reason": stop_reason, "accepted": S.accepted, "mask": sel_mask, "gate_on_record": gated,
                                                     "weights": final_weights, "penalties": S.penalties, "blocked": sorted(S.blocked)})
        P.builder.cleanup()
        self.mark("search", {"stop_reason": stop_reason, "accepted": S.accepted})

    def _search_oneshot(self, P, init, lc, cap=None):
        """One-shot search (`logodds` or generative `kl`): solo + Wanda width probes + Rule-P vocab -> exact-byte solve -> 1 build."""
        import numpy as np
        from . import oneshot, runtime as rtmod, vocab as vocabmod
        base = mk.new_mask(self.groups, "BF16", self._source_pin(), self.mode)
        sdir = self.dir / "probe" / "sanity"
        probe_dirs = [P.work, sdir] if P.work != sdir else [sdir]

        def _find_probe(fname: str) -> Path | None:
            for d in probe_dirs:
                cand_p = d / fname
                if cand_p.exists():
                    return cand_p
            return None

        logodds_stores = [s for s in P.stores if s.kind == "logodds" and s.name != "general"]
        is_kl_mode = len(logodds_stores) == 0
        task_stores = [s for s in P.stores if s.kind == "kl" and s.name != "general"] if is_kl_mode else logodds_stores
        tiers = self.cfg["sanity"].get("solo_tiers") or self.ladder[1:]

        def _load_vec(key_str: str, s_name: str, expected_src: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
            if not is_kl_mode:
                fp = _find_probe(f"p-{key_str}-{s_name}.npy")
                if fp is None:
                    return None
                v = np.asarray(np.load(fp), dtype=np.float64)
                return v, expected_src, np.ones(len(v), dtype=np.float64)
            fp = _find_probe(f"p-{key_str}-{s_name}.json")
            if fp is None:
                return None
            return oneshot.extract_traj_kl(read_json(fp), self.mode)

        solo, used = {}, []
        for s in task_stores:
            first_probes = [
                (g, t) for g in base["groups"]
                for t in [x for x in tiers if mk.tier_allowed(self.groups[g], x)][:1]
            ]
            if not is_kl_mode:
                src = np.asarray(np.load(Path(s.dir) / "source.npy"), dtype=np.float64)
                have = all(_find_probe(f"p-{bmod.mask_key(mk.apply(base, g, t))}-{s.name}.npy") is not None for g, t in first_probes)
                if not have:
                    self.log(f"oneshot: store {s.name} has no sanity solo scores; left out of the surrogate")
                    continue
                used.append((s.name, s.weight, oneshot.pauc_store(P._split(s, src), s.meta["parts"], s.meta.get("objective_args", {})), src))
            else:
                sample = None
                have = True
                for g, t in first_probes:
                    got = _load_vec(bmod.mask_key(mk.apply(base, g, t)), s.name)
                    if got is None:
                        have = False
                        break
                    if sample is None:
                        sample = got
                if not have or sample is None:
                    self.log(f"oneshot: kl store {s.name} has no sanity solo receipts; left out of the surrogate")
                    continue
                _cand0, src_vec, tw = sample
                used.append((s.name, s.weight * float(tw.sum()) / max(1, len(src_vec)), oneshot.kl_store_damage(tw), src_vec))

        if not used:
            raise RuntimeError("oneshot needs per-trajectory sanity solo receipts on at least one task store")

        # Optional general-KL cap surrogate for generative runs (`SizeMatchedCap(bpw)`)
        gen_store = next((s for s in P.stores if s.name == "general" and s.kind == "kl"), None)
        use_gen_guard = is_kl_mode and gen_store is not None and cap is not None and lc.get("enforce_general_cap", True)
        gen_solo: dict = {}
        gen_used = []
        if use_gen_guard:
            for g in base["groups"]:
                for t in [x for x in tiers if mk.tier_allowed(self.groups[g], x)][:1]:
                    fp = _find_probe(f"p-{bmod.mask_key(mk.apply(base, g, t))}-general.json")
                    if fp is not None:
                        _cg, sg, twg = oneshot.extract_traj_kl(read_json(fp), "preserve")
                        gen_used = [("general", 1.0, oneshot.kl_store_damage(twg), sg)]
                        break
                if gen_used:
                    break

        default_tiers = lc.get("oneshot_tiers", self.ladder[1:] if is_kl_mode else ["Q8_0", "Q4_K", "IQ2_XS", "IQ1_S"])
        group_tier_overrides = lc.get("group_tiers", {})
        respect_floors = lc.get("respect_floors", False)
        floors = init.get("floors", {}) if respect_floors else {}
        allowed_tiers: dict[str, list[str]] = {}
        for g in base["groups"]:
            cand_t = group_tier_overrides.get(g, default_tiers)
            fl = floors.get(g)
            max_idx = self.ladder.index(fl["min_tier"]) if fl and fl.get("min_tier") in self.ladder else len(self.ladder)
            o = [t for t in self.ladder[: max_idx + 1] if (t in cand_t) and mk.tier_allowed(self.groups[g], t)]
            allowed_tiers[g] = []
            for t in o:
                key_gt = bmod.mask_key(mk.apply(base, g, t))
                shifts = []
                for name, _, _, src in used:
                    got = _load_vec(key_gt, name, src)
                    if got is not None:
                        shifts.append(got[0] - src)
                if len(shifts) == len(used):
                    solo[(g, t)] = np.concatenate(shifts)
                    allowed_tiers[g].append(t)
                    if gen_used:
                        fg = _find_probe(f"p-{key_gt}-general.json")
                        if fg is not None:
                            cg, sg, _ = oneshot.extract_traj_kl(read_json(fg), "preserve")
                            gen_solo[(g, t)] = cg - sg
                elif t in ("BF16", "Q8_0"):
                    allowed_tiers[g].append(t)
            drop_ok_by_floor = (fl is None) or bool(fl.get("drop_ok", True))
            if self.groups[g]["droppable"] and lc.get("allow_drop", True) and drop_ok_by_floor:
                key_dr = bmod.mask_key(mk.apply(base, g, "drop"))
                shifts = []
                for name, _, _, src in used:
                    got = _load_vec(key_dr, name, src)
                    if got is not None:
                        shifts.append(got[0] - src)
                if len(shifts) == len(used):
                    solo[(g, "drop")] = np.concatenate(shifts)
                    if gen_used:
                        fg = _find_probe(f"p-{key_dr}-general.json")
                        if fg is not None:
                            cg, sg, _ = oneshot.extract_traj_kl(read_json(fg), "preserve")
                            gen_solo[(g, "drop")] = cg - sg

        # Optional Wanda neuron ordering + per-MLP width probes
        n_ff0 = gguf_ops.ffn_widths(self.probe_src)
        width_cuts: dict[tuple[int, int], np.ndarray] = {}
        gen_width_cuts: dict[tuple[int, int], np.ndarray] = {}
        neuron_order: dict[int, np.ndarray] = {}
        if lc.get("widths"):
            order_file = self.dir / "wanda_order.json"
            if order_file.exists():
                neuron_order = {int(k): np.asarray(v, dtype=np.int64) for k, v in read_json(order_file).items()}
            else:
                neuron_order = gguf_ops.wanda_order(self.probe_src, str(self.dir / "proxy" / "imatrix.gguf"))
                write_json(order_file, {str(k): v.tolist() for k, v in neuron_order.items()})
            fracs = lc.get("width_fractions", [0.75, 0.5, 0.25, 0.0625])
            step = int(lc.get("width_step", 256))
            ext = "json" if is_kl_mode else "npy"
            want_stores = (["general"] if gen_used else []) + [name for name, _, _, _ in used]
            w_masks, w_keys = [], []
            for il, full in sorted(n_ff0.items()):
                if il not in neuron_order:
                    continue
                for f in fracs:
                    w = int(full * f) // step * step
                    if 0 < w < full:
                        mw = json.loads(json.dumps(base))
                        mw["widths"] = {str(il): mk.pack_bitmap(neuron_order[il][:w], full)}
                        key_w = bmod.mask_key(mw)
                        need = any(_find_probe(f"p-{key_w}-{sname}.{ext}") is None for sname in want_stores)
                        if need:
                            w_masks.append(mw)
                        w_keys.append((il, w, mw))
            if w_masks:
                self.log(f"oneshot: probing {len(w_masks)} MLP width settings")
                P.probe_many(w_masks, want_stores)
            for il, w, mw in w_keys:
                key_w = bmod.mask_key(mw)
                shifts = []
                for name, _, _, src in used:
                    got = _load_vec(key_w, name, src)
                    if got is not None:
                        shifts.append(got[0] - src)
                if len(shifts) == len(used):
                    width_cuts[(il, w)] = np.concatenate(shifts)
                if gen_used:
                    fg = _find_probe(f"p-{key_w}-general.json")
                    if fg is not None:
                        cg, sg, _ = oneshot.extract_traj_kl(read_json(fg), "preserve")
                        gen_width_cuts[(il, w)] = cg - sg

        # Optional Merge-Closed Rule-P vocabulary keep-set (fixed K or automatic scan)
        keep_vocab = None
        vocab_cfg = lc.get("vocab_k")
        if vocab_cfg:
            env_obj = getattr(self, "env", None)
            if env_obj is not None and hasattr(env_obj, "fit_token_counts"):
                counts = env_obj.fit_token_counts(self.probe_src, self.tc)
            else:
                v_sz = vocabmod.read_tokenizer(self.probe_src)["vocab_size"]
                counts = np.zeros(v_sz, dtype=np.int64)
                corpus_p = self.dir / "proxy" / "stats_corpus.txt"
                if corpus_p.exists() and self.tc.tokdocs is not None:
                    for arr in rtmod.tokenize_docs(self.tc, self.probe_src, corpus_p.read_text().split("\n\n")):
                        if len(arr):
                            counts += np.bincount(arr, minlength=v_sz)[:v_sz]
                task_p = self.dir / "proxy" / "task.bin"
                if is_kl_mode and task_p.exists():
                    raw_t = task_p.read_bytes()
                    (ntrj,) = struct.unpack_from("<I", raw_t, 4)
                    pb = 8
                    for _ in range(ntrj):
                        nt, ng = struct.unpack_from("<II", raw_t, pb); pb += 8
                        sa = np.frombuffer(raw_t, dtype="<i4", count=nt, offset=pb); pb += 4 * (nt + ng)
                        if len(sa):
                            counts += 10 * np.bincount(sa, minlength=v_sz)[:v_sz]
            must_keep = list(getattr(env_obj, "pair", ()))
            if vocab_cfg == "auto":
                cand_ks = lc.get("vocab_candidates", [50000, 65000, 80000, 100000])
                min_fid = float(lc.get("vocab_min_fidelity", 0.998 if not is_kl_mode else 0.995))
                if not is_kl_mode:
                    parts = read_json(self.dir / "proxy" / "parts.json")
                    proxy_tasks = [self.by_id[i] for p in parts for i in p["ids"]]
                    src_vec = used[0][3]
                    rs = oneshot._rank(src_vec)
                    def _eval_vocab(keep_arr: np.ndarray) -> float:
                        mv = json.loads(json.dumps(base))
                        mv["vocab_keep"] = mk.pack_bitmap(keep_arr, vocabmod.read_tokenizer(self.probe_src)["vocab_size"])
                        built = P.builder.build(mv, self.scratch / "build" / f"vocab-{len(keep_arr)}.gguf").path
                        try:
                            sc = self.env.score(proxy_tasks, built, self.scorer)
                            vec = np.array([sc[t.id] for t in proxy_tasks], dtype=np.float64)
                            return float((oneshot._rank(vec) * rs).mean())
                        finally:
                            Path(built).unlink(missing_ok=True)
                else:
                    # Unsplit target-token coverage on `task.bin` (avoids `n_vocab` store mismatch while preserving BPE generation)
                    raw_trj = (self.dir / "proxy" / "task.bin").read_bytes()
                    (n_seq,) = struct.unpack_from("<I", raw_trj, 4)
                    pos_b, tgt_toks = 8, []
                    for _ in range(n_seq):
                        nt, ng = struct.unpack_from("<II", raw_trj, pos_b); pos_b += 8
                        seq_a = np.frombuffer(raw_trj, dtype="<i4", count=nt, offset=pos_b); pos_b += 4 * nt
                        tg_a = np.frombuffer(raw_trj, dtype="<u4", count=ng, offset=pos_b); pos_b += 4 * ng
                        if ng:
                            tgt_toks.append(seq_a[tg_a])
                    all_tgt = np.concatenate(tgt_toks) if tgt_toks else np.zeros(0, dtype=np.int32)
                    def _eval_vocab(keep_arr: np.ndarray) -> float:
                        if len(all_tgt) == 0:
                            return 1.0
                        return float(np.isin(all_tgt, keep_arr).mean())
                chosen_k, keep_vocab, v_meas = vocabmod.pick_vocab_k(self.probe_src, counts, cand_ks, _eval_vocab, min_fidelity=min_fid, must_keep=must_keep)
                self.log(f"oneshot: auto-selected vocab K={chosen_k} (kept={len(keep_vocab)}), fidelities={v_meas}")
            else:
                keep_vocab = vocabmod.rule_p_keep_set(self.probe_src, counts, int(vocab_cfg), must_keep=must_keep)
                self.log(f"oneshot: Rule-P vocab K={vocab_cfg} -> kept {len(keep_vocab)} tokens")

        bm = oneshot.ByteModel(self.probe_src, self.groups)
        u = read_json(self.dir / "sanity" / "uniform.json")
        cal_tier = "Q4_K" if "Q4_K" in u else next(iter(u))
        bm.calibrate(self.uniform_mask(base, cal_tier), u[cal_tier]["bytes"])
        if keep_vocab is not None:
            bm.set_vocab_trim(keep_vocab)

        use_joint = bool(width_cuts)
        if use_joint:
            options = oneshot.build_joint_options(self.groups, allowed_tiers, solo, width_cuts, n_ff0,
                                                  width_step=int(lc.get("width_step", 256)), allow_drop=lc.get("allow_drop", True))
            ref_sol = {g: options[g][0] for g in options}
            ref_t, ref_w = oneshot.unpack_solution(ref_sol)
            ref_bytes = bm.spec_bytes(ref_t, ref_w)
            cost = {}
            for g, o in options.items():
                for opt in o:
                    s1 = dict(ref_sol); s1[g] = opt
                    cost[(g, opt)] = bm.spec_bytes(*oneshot.unpack_solution(s1)) - ref_bytes
            hdr = ref_bytes
        else:
            options = {g: list(allowed_tiers[g]) + (["drop"] if (g, "drop") in solo else []) for g in base["groups"]}
            cost = {}
            ref_tier = "Q8_0" if "Q8_0" in self.ladder else self.ladder[0]
            ref_m = self.uniform_mask(base, ref_tier)
            ref_bytes = bm(ref_m)
            for g, o in options.items():
                for t in o:
                    cost[(g, t)] = bm(mk.apply(ref_m, g, t)) - ref_bytes
            hdr = ref_bytes - sum(cost.get((g, ref_tier), 0) for g in options)

        default_obj = "proxy" if is_kl_mode else "fidelity"
        use_fid = (not is_kl_mode) and (lc.get("oneshot_objective", default_obj) == "fidelity")
        sur_cls = oneshot.FidelitySurrogate if use_fid else oneshot.Surrogate
        sur = sur_cls(used, solo, width_cuts=width_cuts, n_ff0=n_ff0)

        guard_fn = None
        if gen_used and cap is not None:
            gen_sur = oneshot.Surrogate(gen_used, gen_solo, width_cuts=gen_width_cuts, n_ff0=n_ff0)
            src_params = P.builder.source_params
            gen_dvc = {(g, t): gen_sur.shift(g, t) for g, o in options.items() for t in o}
            def _gen_guard(m_cand: dict, b_cand: int) -> float:
                vs = 0
                if keep_vocab is not None:
                    tiers_c, _ = oneshot.unpack_solution(m_cand)
                    t_tiers = {
                        tab: tiers_c.get("output" if tab == "output.weight" else "token_embd", "Q4_K")
                        for tab in ("token_embd.weight", "per_layer_token_embd.weight", "output.weight")
                        if tab in bm.inv
                    }
                    vs = vocabmod.vocab_trim_byte_savings(self.probe_src, keep_vocab, t_tiers)
                gx = gen_sur.src + sum(gen_dvc[(g, t)] for g, t in m_cand.items())
                return float(gen_sur.damage(gx) - cap(bmod_bpw(b_cand + vs, src_params)))
            guard_fn = _gen_guard

        budget, target = lc["target_bytes"], lc["target_bytes"]
        full_vocab_n = vocabmod.read_tokenizer(self.src)["vocab_size"] if keep_vocab is not None else 0
        final_builder = bmod.Builder(self.tc, self.src, self.groups, str(self.dir / "proxy" / "imatrix.gguf"), self.scratch / "build" / "oneshot-final", log=str(self.dir / "build.log"))

        for attempt in range(3):
            sol = oneshot.solve(sur, options, cost, budget, hdr, log=self.log, guard_fn=guard_fn)
            tiers_of, widths_of = oneshot.unpack_solution(sol)
            m = mk.upgrade_v2({**init, "groups": {g: {"tier": t} for g, t in tiers_of.items()}, "floors": {}})
            m["dead_tensors"] = list(self.dead_tensors)
            m["widths"] = {str(il): mk.pack_bitmap(neuron_order[il][:w], n_ff0[il]) for il, w in sorted(widths_of.items()) if w < n_ff0[il]}
            if keep_vocab is not None and len(keep_vocab) < full_vocab_n:
                m["vocab_keep"] = mk.pack_bitmap(keep_vocab, full_vocab_n)
            mk.validate(m, self.groups)
            if mk.has_v3_cuts(m):
                br = final_builder.build(m, self.scratch / "build" / "oneshot-final" / f"attempt-{attempt}.gguf")
                built_bytes, built_bpw, built_key = br.file_bytes, br.bpw, bmod.mask_key(m)
                Path(br.path).unlink(missing_ok=True)
                kl_rec = {}
            else:
                r = P.probe(m)
                built_bytes, built_bpw, built_key, kl_rec = r.bytes, r.bpw, r.key, r.kl
            self.record("history", {"step": 0, "mask_key": built_key, "bytes": built_bytes, "bpw": built_bpw, "kl": kl_rec, "event": "oneshot", "budget": budget})
            self.log(f"oneshot attempt {attempt + 1}: built {built_bytes} bytes for budget {budget} (target {target})")
            if built_bytes <= target:
                final_builder.cleanup()
                return m
            budget -= built_bytes - target
        final_builder.cleanup()
        raise RuntimeError("oneshot: the built file stayed over target after three solves")

    def _repair(self, m, P):
        from . import repair
        rc = self.cfg["repair"]
        R = kl.Renderer(self.cfg["source"]["hf_dir"])
        recs = []
        for r in read_jsonl(self.dir / "episodes" / "pool.jsonl"):
            if r["split"] == "train" and r["success"]:
                try:
                    recs += R.episode(r["messages"], self._task_tools(r.get("task")))
                except kl.RenderError:
                    pass
        if not recs:
            self.log("repair: no train-split trajectories; skipped")
            return self.src, [], m
        out = self.dir / "repair" / "hf"
        rec = repair.repair(self.cfg["source"]["hf_dir"], None, m, recs, self.mode, out, rc, self.log)
        if rec is None:
            self.log("repair: selected mask has no removed sub-blocks; training cannot improve on the frozen source under an unsimulated quantiser, skipped")
            write_json(self.dir / "repair" / "lineage.json", {"skipped": "no removal in selected mask"})
            return self.src, [], m
        gg = self.dir / "repair" / "repaired-bf16.gguf"
        repair.to_gguf(self.cfg["toolchain"]["convert_py"], out, gg, str(self.dir / "repair" / "convert.log"))
        rec["gguf_sha256"] = sha256_file(gg)
        P2 = self.make_prober(str(gg), "repair")
        before, after = P.probe(m), P2.probe(m)
        rec["task_proxy_before"], rec["task_proxy_after"] = before.task_proxy(P.stores, self.mode), after.task_proxy(P2.stores, self.mode)
        rec["kl_general_before"], rec["kl_general_after"] = before.kl["general"], after.kl["general"]
        cand = P2.builder.build(m, self.scratch / "build" / "repair-screen.gguf").path
        sc = self.screen(cand, "repair")
        rec["screen"] = {k: v for k, v in sc.items() if not k.startswith("_")}
        accept = sc["pass"] and rec["task_proxy_after"] <= rec["task_proxy_before"]
        rec["accepted"] = accept
        write_json(self.dir / "repair" / "lineage.json", rec)
        self.log(f"repair: proxy {rec['task_proxy_before']:.5f} -> {rec['task_proxy_after']:.5f}, screen pass={sc['pass']} -> accepted={accept}")
        P2.builder.cleanup()
        return (str(gg), [rec], m) if accept else (self.src, [], m)

    # ------------------------------------------------------------------ freeze
    def phase_freeze(self):
        sr = read_json(self.dir / "search_result.json")
        m, weights = sr["mask"], sr["weights"]
        fdir = self.dir / "final"
        fdir.mkdir(exist_ok=True)
        B = bmod.Builder(self.tc, weights, self.groups, str(self.dir / "proxy" / "imatrix.gguf"), self.scratch / "build" / "freeze")
        r1 = B.build(m, fdir / "specialist.gguf")
        r2 = B.build(m, self.scratch / "build" / "freeze" / "rebuild.gguf")
        h1, h2 = sha256_file(r1.path), sha256_file(r2.path)
        Path(r2.path).unlink()
        types, drop = mk.tensor_plan(m, self.groups)
        inv = {t["name"]: t for t in gguf_ops.inventory(r1.path)}
        audit = {"types_match": all(inv[n]["type"] == t.upper() or n in r1.fallbacks for n, t in types.items()),
                 "dropped_absent": all(n not in inv for n in drop), "fallbacks": r1.fallbacks,
                 "rebuild_sha256_equal": h1 == h2, "sha256": h1}
        mem = self.measure_memory(r1.path)
        n_attn_kept = sum(1 for g, e in m["groups"].items()
                          if g.endswith(".attn") and self.groups[g].get("kv", True) and e["tier"] != "drop")
        kvb = bmod.kv_bytes(self.info, n_attn_kept, self.dep["context"], self.dep["concurrency"], self.dep["kv_type"])
        limit = self.dep.get("max_total_memory_bytes")
        constraint = {"limit": limit, "measured_total": mem.get("total_bytes"), "kv_formula_bytes": kvb,
                      "kv_measured_bytes": mem.get("kv_bytes"), "ok": (limit is None) or (mem.get("total_bytes", 0) <= limit)}
        m["measured"] = {"file_bytes": r1.file_bytes, "bpw": r1.bpw, "memory": mem}
        write_json(fdir / "mask.json", m)
        out = {"build": r1.to_dict(), "audit": audit, "deployment_memory": constraint, "weights": weights}
        write_json(fdir / "freeze.json", out)
        self.log(f"freeze: {r1.file_bytes} bytes ({r1.bpw:.3f} bpw), audit={audit}, memory={constraint}")
        if not (audit["types_match"] and audit["dropped_absent"] and audit["rebuild_sha256_equal"]):
            raise RuntimeError("reproduction/audit check failed")
        if not constraint["ok"]:
            raise RuntimeError("deployment memory constraint violated")
        B.cleanup()
        self.mark("freeze")

    def measure_memory(self, model: str) -> dict:
        log = self.dir / "final" / "memory-server.log"
        log.unlink(missing_ok=True)
        with LlamaServer(self.tc, model, gpu=self.tc.gpus[0], slots=self.dep["concurrency"], ctx_per_slot=self.dep["context"],
                         kv_type=self.dep["kv_type"], log=log, extra=["-v"]) as s:
            t0 = time.time()
            from .client import ChatClient
            r = ChatClient(s.url, {"temperature": 0.0}).chat([{"role": "user", "content": "Count from 1 to 50."}], seed=1, max_tokens=128)
            dt = time.time() - t0
        txt = log.read_text(errors="ignore")
        mib = lambda pat: sum(float(x) for x in re.findall(pat, txt))
        model_b = mib(r"model buffer size =\s+([\d.]+) MiB")
        kv = re.findall(r"llama_kv_cache: size =\s+([\d.]+) MiB", txt)
        comp = mib(r"compute buffer size =\s+([\d.]+) MiB")
        out = {"model_buffers_bytes": int(model_b * 2 ** 20), "kv_bytes": int(float(kv[-1]) * 2 ** 20) if kv else None,
               "compute_buffers_bytes": int(comp * 2 ** 20), "decode_tokens": r["completion_tokens"], "decode_wall_s": round(dt, 3)}
        out["total_bytes"] = out["model_buffers_bytes"] + (out["kv_bytes"] or 0) + out["compute_buffers_bytes"]
        return out

    # ------------------------------------------------------------------ exam (single use)
    def phase_exam(self):
        if (self.dir / "gates" / "exam.json").exists():
            raise RuntimeError("exam split already spent")
        pc, st = self.cfg["pool"], self.cfg["stats"]
        ex_tasks = self.guard.get("exam")
        rng = random.Random(stable_seed(self.cfg["run_id"], "exam-pool"))
        rng.shuffle(ex_tasks)
        ex_tasks = ex_tasks[: pc["tasks_per_split"].get("exam", pc["tasks_per_split"]["select"])]
        if self.cfg["exam"].get("source_scores"):
            rows = self._exam_pool_from_scores(ex_tasks)
        else:
            rows = ok_rows(self.ev.run(self.src, self.jobs("exam-pool", ex_tasks, pc["runs"]), "exam-pool", "src"))
        per = {}
        for r in rows:
            per.setdefault(r["task"], []).append(r["success"])
        self.exam_pool = [self.by_id[t] for t, xs in per.items() if len(xs) >= pc["runs"] - 1 and sum(xs) / len(xs) >= pc["min_pass"] / pc["runs"]]
        self.log(f"exam pool: {len(self.exam_pool)} reliable of {len(ex_tasks)}")
        spec = str(self.dir / "final" / "specialist.gguf")
        res = self.gate(spec, "exam", split="exam")
        res["claim"] = ("non-inferior to source" if self.mode == "preserve" else "superior to source")
        res["claim_holds"] = res["pass"] if self.mode == "preserve" else (res["L"] > 0)
        write_json(self.dir / "gates" / "exam.json", res)
        # descriptive baselines at matched bytes (observed, not claimed)
        self.exam_baselines(spec)
        if self.classifier and self.cfg["exam"].get("external"):
            self.exam_compare(spec)
        self.mark("exam", {"claim_holds": res["claim_holds"]})

    def exam_compare(self, spec: str):
        """Classifier exam, descriptive: the specialist, the source and external models (exam.external), each at
        its own 1%-of-calibration-negatives cutoff, on every exam document, with shared bootstrap resamples."""
        import glob
        import numpy as np
        from . import compare
        tasks = [t for t in self.env.tasks() if t.split == "exam"]
        cal = [t for t in self.env.tasks() if t.split == "calibration"]
        strip = lambda i: i.split(":", 1)[1]
        models = {}
        for name, path in (("specialist", spec), ("source", self.src)):
            sc = self.ev.scores(path, cal + tasks)
            models[name] = {"cal": np.array([sc[t.id] for t in cal]), "scores": {strip(t.id): sc[t.id] for t in tasks}}
        for ext in self.cfg["exam"]["external"]:
            rows = [r for f in glob.glob(str(Path(ext["scores"]) / "*.jsonl")) for r in read_jsonl(f)]
            cal_ids = {strip(t.id) for t in cal}
            models[ext["name"]] = {"cal": np.array([r["p_human"] for r in rows if r["id"] in cal_ids]),
                                   "scores": {r["id"]: r["p_human"] for r in rows}}
        rng = np.random.default_rng(0)
        out = {}
        for source in sorted({t.id.split(":", 1)[0] for t in tasks}):
            sel = [t for t in tasks if t.id.split(":", 1)[0] == source]
            labels = {strip(t.id): "pos" if self.env.decide(t, -1e9, self.ev.rule(self.src)).tests_passed else "neg" for t in sel}
            kinds = {strip(t.id): t.meta.get("kind", "") for t in sel}
            ms = {m: {"cal": v["cal"], "scores": {i: x for i, x in v["scores"].items() if i in labels}} for m, v in models.items()}
            out[source] = compare.compare(ms, labels, kinds, boot=self.cfg["exam"].get("boot", 1000), seed=rng)
            for m in models:
                if m != "specialist":
                    out[source].setdefault("beats", {})[m] = compare.beats(out[source], "specialist", m)
        write_json(self.dir / "gates" / "exam_compare.json", out)
        self.log(f"exam compare: {json.dumps({s_: {m: v.get('pos', v) for m, v in r.get('models', r).items()} for s_, r in out.items()}, default=str)[:600]}")

    def exam_baselines(self, spec: str):
        target = gguf_ops.file_bytes(spec)
        P = self.make_prober(self.src, "baselines")
        base = mk.new_mask(self.groups, "BF16", self._source_pin(), self.mode)
        uniform = read_json(self.dir / "sanity" / "uniform.json")
        tier = min(uniform, key=lambda t: abs(uniform[t]["bytes"] - target))
        cands = {f"uniform-{tier}": self.uniform_mask(base, tier)}
        # dynquant-style recipes (isolated restore gains on the task proxy, ranked once, no re-ranking):
        #   K34-style: Q4_K base, top-K groups restored to Q8_0;  K94-style: Q2_K base, Q8_0 ranking, top-K deployed at Q4_K
        blk = [g for g in base["groups"] if g.startswith("blk.")]
        recipes = {"k34": ("Q4_K", "Q8_0", "Q8_0"), "k94": ("Q2_K", "Q8_0", "Q4_K")}
        best = None
        rankings = {}
        for name, (b_t, rank_t, dep_t) in recipes.items():
            if not {b_t, rank_t, dep_t} <= set(self.ladder):  # a recipe needs every tier it uses on this run's ladder
                continue
            bm = self.uniform_mask(base, b_t)
            if uniform.get(b_t, {}).get("bytes", 0) > target and mk.rank(dep_t) < mk.rank(b_t):
                self.log(f"baseline {name}: skipped, its {b_t} base ({uniform[b_t]['bytes']:,} B) already exceeds the target and restores only add bytes")
                continue
            floor = P.probe(bm, ["task"]).kl["task"]
            gains = [(floor - r.kl["task"], g) for g, r in zip(blk, P.probe_many([mk.apply(bm, g, rank_t) for g in blk], ["task"]))]
            ranked = [g for gain, g in sorted(gains, reverse=True) if gain > 0]
            rankings[name] = ranked
            lo, hi = 0, len(ranked)
            def bytes_k(k):
                mm = bm
                for g in ranked[:k]:
                    mm = mk.apply(mm, g, dep_t)
                return P.builder.build(mm, self.scratch / "build" / "dq.gguf").file_bytes, mm
            while lo < hi:          # bytes grow monotonically with K: binary search for the nearest K
                mid = (lo + hi) // 2
                if bytes_k(mid)[0] < target:
                    lo = mid + 1
                else:
                    hi = mid
            for k in {max(0, lo - 1), lo}:
                bk, mm = bytes_k(k)
                if best is None or abs(bk - target) < best[0]:
                    best = (abs(bk - target), f"dynquant-{name}-K{k}", mm)
        if best:
            cands[best[1]] = best[2]
        ranked = rankings
        out = {"target_bytes": target, "ranking": ranked, "points": {}}
        n = self.cfg["exam"]["baseline_eps_per_task"]
        for name, m in cands.items():
            path = P.builder.build(m, self.scratch / "build" / f"{name}.gguf").path
            rows = ok_rows(self.ev.run(path, self.jobs(f"exam-{name}", self.exam_pool, n), f"exam-{name}", "cand"))
            src = [dict(r, side="src") for r in read_jsonl(self.dir / "episodes" / "gate-exam-src.jsonl") if r["success"] is not None]
            b = stats.delta_and_bounds(rows + src, 0.05, n_boot=2000)
            nb = gguf_ops.file_bytes(path)
            out["points"][name] = {"bytes": nb, "bytes_ratio_to_target": nb / target, "matched_within_5pct": abs(nb / target - 1) <= 0.05, **b}
            Path(path).unlink(missing_ok=True)
            self.log(f"baseline {name}: bytes={out['points'][name]['bytes']} delta={b['delta']:+.4f} [{b['L']:+.4f},{b['U']:+.4f}]")
        write_json(self.dir / "gates" / "exam_baselines.json", out)
        P.builder.cleanup()

    def _weights_outcome(self):
        if not self.cfg.get("repair", {}).get("enabled"):
            return "mask-only"
        lp = self.dir / "repair" / "lineage.json"
        if not lp.exists():
            return "mask+weights enabled, repair not reached"
        rec = read_json(lp)
        if rec.get("skipped"):
            return f"mask+weights enabled, repair skipped ({rec['skipped']}); shipped weights = frozen source"
        return f"mask+weights; repair {'ACCEPTED' if rec.get('accepted') else 'rejected'} (task proxy {rec.get('task_proxy_before', float('nan')):.5f} -> {rec.get('task_proxy_after', float('nan')):.5f})"

    # ------------------------------------------------------------------ report
    def phase_report(self):
        g = lambda p: read_json(self.dir / p) if (self.dir / p).exists() else None
        run, pool, aa, fr, ex, bl, sr = g("run.json"), g("pool.json"), g("aa.json"), g("final/freeze.json"), g("gates/exam.json"), g("gates/exam_baselines.json"), g("search_result.json")
        src_bytes = gguf_ops.file_bytes(self.src)
        lines = [f"# Chantless MVP run `{self.cfg['run_id']}`", "",
                 f"Source `{self.cfg['source']['repo']}` @ `{self.cfg['source']['revision'][:12]}`; env `{self.env.name}` v{self.env.version}; objective mode {self.mode}; weights: {self._weights_outcome()}.",
                 f"Returned mask has a passing gate on record: **{sr.get('gate_on_record')}**" + ("" if sr.get("gate_on_record") else " (no searched mask passed a gate; the sanity mask was frozen instead)") + ".",
                 "Everything below is measured in this run. Claims are limited to the exam test stated; all else is observed, not claimed.", "",
                 "## Exam (single use)",
                 f"- Claim tested: **{ex['claim']}** at margin {self.cfg['stats']['margin']}, one-sided alpha {self.cfg['stats']['alpha']}. Result: **{'HOLDS' if ex['claim_holds'] else 'DOES NOT HOLD'}**.",
                 f"- delta (specialist - source) = {ex['delta']:+.4f}, one-sided bounds L={ex['L']:+.4f}, U={ex['U']:+.4f}; {ex.get('n_cand')} specialist / {ex.get('n_src')} source episodes on {ex['n_tasks']} reliable exam tasks.",
                 f"- Detectors (upper bound on cand-src rate <= {self.cfg['stats']['detector_bound']}): " + ", ".join(f"{k} {'ok' if v['ok'] else 'FAIL'} (U={v['U']:+.3f})" for k, v in ex["detectors"].items()),
                 f"- Behavioral sanity: delta={ex['behavioral']['delta']:+.4f}, L={ex['behavioral']['L']:+.4f} -> {'ok' if ex['behavioral_ok'] else 'FAIL'}.", "",
                 "## Artifact",
                 f"- `final/specialist.gguf`: {fr['build']['file_bytes']:,} bytes ({fr['build']['bpw']:.3f} bpw) vs source {src_bytes:,} bytes ({fr['build']['file_bytes'] / src_bytes:.1%}). Dropped tensors: {len(fr['build']['dropped'])}.",
                 f"- Rebuild reproduces sha256: {fr['audit']['rebuild_sha256_equal']}; header matches mask: {fr['audit']['types_match']}; dropped tensors absent: {fr['audit']['dropped_absent']}.",
                 f"- Measured runtime memory at context {self.dep['context']} x {self.dep['concurrency']}, KV {self.dep['kv_type']}: {fr['deployment_memory']['measured_total']:,} bytes (KV measured {fr['deployment_memory']['kv_measured_bytes']}, formula {fr['deployment_memory']['kv_formula_bytes']}); constraint ok: {fr['deployment_memory']['ok']}.", "",
                 "## Search",
                 f"- Stop: {sr['stop_reason']} after {sr['accepted']} accepted moves; feedback penalties {sr['penalties']}; blocked {sr['blocked']}.",
                 f"- Pool: {pool['sizes']}; A/A delta {aa['aa_delta']:+.4f}, v={aa['v']:.4f}, deff={aa['deff_used']:.2f}, n_gate={aa['n_gate']}.", "",
                 "## Baselines at matched bytes (descriptive, not claimed)"]
        for k, v in (bl or {}).get("points", {}).items():
            match = "" if v.get("matched_within_5pct", True) else f" (NOT byte-matched: {v['bytes_ratio_to_target']:.2f}x target; nearest this recipe reaches)"
            lines.append(f"- {k}: {v['bytes']:,} bytes, delta={v['delta']:+.4f} [{v['L']:+.4f}, {v['U']:+.4f}]{match}")
        cmp_ = g("gates/exam_compare.json")
        if cmp_:
            lines += ["", "## Classifier comparison (descriptive, not claimed)",
                      "Each model at its own cutoff (1% of calibration negatives); positives caught, 95% bootstrap intervals over shared resamples."]
            for source, r in cmp_.items():
                lines.append(f"- {source} ({r['docs']} docs): " + "; ".join(
                    f"{m} {v['pos']['caught']:.3f} [{v['pos']['ci'][0]:.3f}, {v['pos']['ci'][1]:.3f}], negatives flagged {v['neg_flagged']:.3f}"
                    for m, v in r["models"].items()) + "; interval wholly above: " + ", ".join(f"{m} {b}" for m, b in r.get("beats", {}).items()))
        lines += ["", "## Not established", "No global optimum, no generalisation beyond this environment and model, no latency/throughput claim, no MiMo/qwen35 result."]
        (self.dir / "report.md").write_text("\n".join(lines) + "\n")
        self.log("report written")
        self.mark("report")


def bmod_bpw(nbytes: int, params: int) -> float:
    return nbytes * 8 / params


def _tools(env):
    """The tool schema the environment's episodes were rendered with; codetasks predates the attribute."""
    if hasattr(env, "tools"):
        return env.tools
    from .env.codetasks import TOOLS
    return TOOLS
