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
        self.scratch = Path(cfg["scratch_root"]) / cfg["run_id"] if cfg.get("scratch_root") else self.dir
        if cfg.get("scratch_root"):
            # stage the source on fast unencrypted storage; served, quantized and pruned from there
            staged = self.scratch / "source" / Path(self.src).name
            if not staged.exists() or staged.stat().st_size != Path(self.src).stat().st_size:
                staged.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(self.src, staged)
            self.src = str(staged)
            pin = self.dir / "run.json"
            if pin.exists() and sha256_file(self.src) != read_json(pin)["pins"]["source_gguf_sha256"]:
                raise RuntimeError("staged source GGUF does not match the pinned sha256")
        self.inv = gguf_ops.inventory(self.src)
        self.groups = gguf_ops.groups_from_inventory(self.inv)
        imx = Path(cfg["run_root"]) / cfg["run_id"] / "proxy" / "imatrix.gguf"
        if imx.exists():
            self.annotate_imatrix(imx)
        self.info = gguf_ops.arch_info(self.src)
        self.ladder = cfg["ladder"]
        self.mode = cfg.get("objective_mode", "preserve")
        self.dep = self.env.deployment
        self.ev = Evaluator(self.tc, self.env, cfg["decoding"], cfg["serving"]["slots_per_gpu"],
                            cfg["serving"]["ctx_per_slot"], self.dep["kv_type"], self.dir / "episodes")
        self.t0 = time.time()

    def annotate_imatrix(self, path):
        import gguf
        names = {t.name.rsplit(".", 1)[0] for t in gguf.GGUFReader(str(path)).tensors}
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
        self.mark("prepare")

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
                    rr = R.episode(r["messages"], _tools())
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
        run_imatrix(self.tc, self.src, str(pdir / "stats_corpus.txt"), str(pdir / "imatrix.gguf"), str(pdir / "imatrix.log"), gpu=self.tc.gpus[0])
        self.annotate_imatrix(pdir / "imatrix.gguf")
        write_json(self.dir / "groups.json", self.groups)
        out = {"task": meta_t, "general": meta_g, "self_kl": {n: read_json(pdir / f"self_{n}.json")["kl"] for n in ("task", "general")},
               "imatrix_sha256": sha256_file(pdir / "imatrix.gguf")}
        write_json(pdir / "proxy.json", out)
        self.log(f"proxy: task {meta_t} general {meta_g} self-KL {out['self_kl']}")
        self.mark("proxy")

    def make_prober(self, weights: str, tag: str) -> Prober:
        pdir = self.dir / "proxy"
        meta = read_json(pdir / "proxy.json")
        B = bmod.Builder(self.tc, weights, self.groups, str(pdir / "imatrix.gguf"), self.scratch / "build" / tag, log=str(self.dir / "build.log"))
        B.threads = self.cfg.get("probe", {}).get("quantize_threads", 8)
        B.max_pruned = self.cfg.get("probe", {}).get("max_pruned", 4)
        P = Prober(self.tc, B, self.dir / "probe" / tag, str(self.dir / "probe.log"), workers=self.cfg.get("probe", {}).get("workers"),
                   scratch=self.scratch / "cand" / tag)
        P.add_store(Store("task", str(pdir / "task.bin"), self._staged(pdir / "store_task"), meta["task"]["n_targets"]))
        P.add_store(Store("general", str(pdir / "general.bin"), self._staged(pdir / "store_general"), meta["general"]["n_targets"]))
        for fb in read_jsonl(self.dir / "feedback_stores.jsonl"):
            P.add_store(Store(fb["name"], fb["trj"], fb["dir"], fb["n_targets"], fb.get("weight", 1.0)))
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
        P = self.make_prober(self.src, "sanity")
        src_bytes = gguf_ops.file_bytes(self.src)
        base = mk.new_mask(self.groups, "BF16", self._source_pin(), self.mode)
        uniform = {}
        um = {t: self.uniform_mask(base, t) for t in self.ladder[1:]}
        for t, r in zip(um, P.probe_many(list(um.values()), ["general", "task"])):
            uniform[t] = {"kl_gen": r.kl["general"], "kl_task": r.kl["task"], "bytes": r.bytes, "bpw": r.bpw, "bytes_saved": src_bytes - r.bytes}
        pts = [(bmod_bpw(src_bytes, P.builder.source_params), 0.0)] + [(u["bpw"], u["kl_gen"]) for u in uniform.values()]
        cap = sanity.SizeMatchedCap(pts)
        selfkl = read_json(self.dir / "proxy" / "proxy.json")["self_kl"]
        for name, key in (("general", "kl_gen"), ("task", "kl_task")):
            q8 = uniform["Q8_0"][key]
            if selfkl[name] > 0.05 * q8:
                raise RuntimeError(f"measurement floor too high for {name}: self-KL {selfkl[name]} > 5% of uniform Q8_0 KL {q8}")
        solo, masks, keys = {}, [], []
        for g in base["groups"]:
            tiers = self.cfg["sanity"].get("solo_tiers") or self.ladder[1:]
            for t in [x for x in tiers if mk.tier_allowed(self.groups[g], x)] + (["drop"] if self.groups[g]["droppable"] else []):
                masks.append(mk.apply(base, g, t))
                keys.append((g, t))
        self.log(f"sanity: {len(masks)} solo probes")
        for (g, t), r in zip(keys, P.probe_many(masks, ["general"])):
            solo.setdefault(g, {})[t] = {"kl_gen": r.kl["general"], "bytes_saved": src_bytes - r.bytes}
        floors = sanity.floors_from_solo(solo, uniform, self.ladder, sc["floor_factor"])
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
        self.log(f"sanity: cap points {cap.to_dict()['points']}; forbidden solo settings {n_forbid}")
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
        looks = st["looks"] if split != "exam" else 1
        rows, res = [], None
        import contextlib
        stack = contextlib.ExitStack()
        cs = stack.enter_context(self.ev.servers(model))
        ss = stack.enter_context(self.ev.servers(self.src))
        for look in range(1, looks + 1):
            lo, hi = (look - 1) * k // looks, look * k // looks
            if hi <= lo:
                continue
            cj = [(t, s) for t in tasks for s in self.seeds(f"gate-{tag}-cand", t.id, k)[lo:hi]]
            sj = [(t, s) for t in tasks for s in self.seeds(f"gate-{tag}-src", t.id, k)[lo:hi]]
            rows += ok_rows(self.ev.run(model, cj, f"gate-{tag}-cand", "cand", srvs=cs))
            rows += ok_rows(self.ev.run(self.src, sj, f"gate-{tag}-src", "src", srvs=ss))
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
        res["behavioral_ok"] = bb["L"] > -self.cfg["sanity"]["behavioral_margin"]
        res["pass"] = res["pass"] and res["behavioral_ok"]
        res.update(tag=tag, split=split, n_gate=n_gate, eps_per_task=k, n_tasks=len(tasks))
        write_json(self.dir / "gates" / f"{tag}.json", res)
        self.record("gates", {k_: v for k_, v in res.items() if k_ not in ("detectors",)})
        return res

    # ------------------------------------------------------------------ search
    def _failure_store(self, eps_rows: list[dict], name: str) -> dict | None:
        """Render failed trajectories, compute the frozen-source reference on them, register as a store."""
        R = kl.Renderer(self.cfg["source"]["hf_dir"])
        recs = []
        for r in eps_rows:
            try:
                recs += R.episode(r["messages"], _tools() if r.get("family") in self._task_families() else None)
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

    def _task_families(self):
        return {t.family for t in self.env.tasks()}

    def phase_search(self):
        lc = self.cfg["learner"]
        init = read_json(self.dir / "sanity" / "sanity_mask.json")
        cap = sanity.SizeMatchedCap([tuple(p) for p in read_json(self.dir / "sanity" / "cap.json")["points"]])
        weights = self.src
        P = self.make_prober(weights, "search")
        L = Learner(P, self.groups, self.ladder, cap, self.mode, lc, self.log, self.record)
        S = LearnerState(mask=init)
        last_gate, last_screen = None, init      # source's gate on record is the A/A
        base = P.probe(S.mask)
        self.record("history", {"step": 0, "mask_key": base.key, "bytes": base.bytes, "bpw": base.bpw, "kl": base.kl, "event": "start"})
        L.full_reprobe(S, base)
        fb_i, consecutive_fail, deadline = 0, 0, self.t0 + 3600 * self.cfg["budget"]["max_hours"]
        stop_reason = "exhausted"
        while True:
            if time.time() > deadline:
                stop_reason = "budget"
                break
            if S.accepted >= lc.get("max_moves", 10 ** 9):
                stop_reason = "max_moves"
                break
            best = L.step(S, base)
            if best is None:
                stop_reason = "exhausted"
                break
            base = best["res"]
            self.record("history", {"step": S.accepted, "move": best["move"].key, "bytes": base.bytes, "bpw": base.bpw, "kl": base.kl,
                                    "score": best["score"], "event": "accept"})
            self.log(f"move {S.accepted}: {best['move'].key} bytes={base.bytes} bpw={base.bpw:.3f} kl_task={base.kl['task']:.5f} kl_gen={base.kl['general']:.5f}")
            if S.accepted % lc["screen_every"] == 0:
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
                        P.add_store(Store(st_row["name"], st_row["trj"], st_row["dir"], st_row["n_targets"]))
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
                L.full_reprobe(S, base)
                if S.accepted % lc["gate_every"] == 0:
                    g = self.gate(P.builder.build(S.mask, self.scratch / "build" / "gate.gguf").path, f"m{S.accepted:04d}")
                    if not g["pass"]:
                        S.mask = last_gate or init
                        stop_reason = "gate_failed"
                        break
                    last_gate = S.mask
        self.log(f"search stopped: {stop_reason} after {S.accepted} accepted moves")
        # optional weight repair on the selected mask
        final_weights, lineage = self.src, []
        if self.cfg.get("repair", {}).get("enabled"):
            final_weights, lineage, S.mask = self._repair(S.mask, P)
        cand_mask = S.mask
        sel_mask = cand_mask
        if last_gate is None or json.dumps(last_gate["groups"], sort_keys=True) != json.dumps(cand_mask["groups"], sort_keys=True) or lineage:
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

    def _repair(self, m, P):
        from . import repair
        rc = self.cfg["repair"]
        R = kl.Renderer(self.cfg["source"]["hf_dir"])
        recs = []
        for r in read_jsonl(self.dir / "episodes" / "pool.jsonl"):
            if r["split"] == "train" and r["success"]:
                try:
                    recs += R.episode(r["messages"], _tools())
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
        n_attn_kept = sum(1 for g, e in m["groups"].items() if g.endswith(".attn") and e["tier"] != "drop")
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
        self.mark("exam", {"claim_holds": res["claim_holds"]})

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
            bm = self.uniform_mask(base, b_t)
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
        lines += ["", "## Not established", "No global optimum, no generalisation beyond this environment and model, no latency/throughput claim, no MiMo/qwen35 result."]
        (self.dir / "report.md").write_text("\n".join(lines) + "\n")
        self.log("report written")
        self.mark("report")


def bmod_bpw(nbytes: int, params: int) -> float:
    return nbytes * 8 / params


def _tools():
    from .env.codetasks import TOOLS
    return TOOLS
