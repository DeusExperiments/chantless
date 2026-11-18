"""llama.cpp process wrappers: llama-server lifecycle, llama-quantize, llama-imatrix, traj-kl, logodds."""
from __future__ import annotations

import json
import os
import shutil
import socket
import struct
import subprocess
import tempfile
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Toolchain:
    llama_bin: Path      # directory with llama-server, llama-quantize, llama-imatrix
    traj_kl: Path        # traj-kl binary
    gpus: list[str]      # device ids for CUDA_VISIBLE_DEVICES; [""] on Metal/CPU
    logodds: Path | None = None  # logodds binary; needed only by classifier environments
    tokdocs: Path | None = None  # tokdocs binary; fast GGUF-native retokenizer for vocab-trimmed models

    @classmethod
    def from_config(cls, cfg: dict) -> "Toolchain":
        return cls(Path(cfg["llama_bin"]), Path(cfg["traj_kl"]), [str(g) for g in cfg.get("gpus", [""])],
                   Path(cfg["logodds"]) if cfg.get("logodds") else None,
                   Path(cfg["tokdocs"]) if cfg.get("tokdocs") else None)

    def env_for(self, gpu: str) -> dict:
        env = dict(os.environ)
        ld = str(self.llama_bin)
        if ld and os.path.isdir(ld):
            cur = env.get("LD_LIBRARY_PATH", "")
            if ld not in cur.split(":"):
                env["LD_LIBRARY_PATH"] = f"{ld}:{cur}" if cur else ld
        if gpu != "":
            env["CUDA_VISIBLE_DEVICES"] = gpu
        return env


def free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


class LlamaServer:
    """One llama-server process. Used as a context manager."""

    def __init__(self, tc: Toolchain, model: str | os.PathLike, gpu: str = "", slots: int = 4, ctx_per_slot: int = 8192,
                 kv_type: str = "f16", log: str | os.PathLike | None = None, extra: list[str] | None = None):
        self.tc, self.model, self.gpu, self.slots = tc, str(model), gpu, slots
        self.ctx = ctx_per_slot * slots
        self.kv_type = kv_type
        self.port = free_port()
        self.log = log
        self.extra = extra or []
        self.proc: subprocess.Popen | None = None

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def __enter__(self):
        cmd = [str(self.tc.llama_bin / "llama-server"), "-m", self.model, "--host", "127.0.0.1", "--port", str(self.port),
               "-c", str(self.ctx), "-np", str(self.slots), "-fa", "on", "--jinja", "-ngl", "999",
               "-ctk", self.kv_type, "-ctv", self.kv_type, "--no-webui", *self.extra]
        out = open(self.log, "a") if self.log else subprocess.DEVNULL
        self.proc = subprocess.Popen(cmd, stdout=out, stderr=subprocess.STDOUT, env=self.tc.env_for(self.gpu))
        deadline = time.time() + 300
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(f"llama-server exited early ({self.proc.returncode}); see {self.log}")
            try:
                with urllib.request.urlopen(self.url + "/health", timeout=2) as r:
                    if json.loads(r.read()).get("status") == "ok":
                        return self
            except Exception:
                pass
            time.sleep(0.5)
        self.__exit__(None, None, None)
        raise RuntimeError("llama-server did not become healthy")

    def __exit__(self, *a):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(30)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.proc = None

    def post(self, path: str, body: dict) -> dict:
        req = urllib.request.Request(self.url + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=600) as r:
            return json.loads(r.read())


def run(cmd: list[str], log: str | os.PathLike | None, env: dict | None = None, check: bool = True) -> subprocess.CompletedProcess:
    with open(log, "a") if log else open(os.devnull, "w") as f:
        f.write("$ " + " ".join(map(str, cmd)) + "\n")
        f.flush()
        p = subprocess.run(list(map(str, cmd)), stdout=f, stderr=subprocess.STDOUT, env=env)
    if check and p.returncode != 0:
        raise RuntimeError(f"command failed ({p.returncode}): {' '.join(map(str, cmd[:3]))} ... see {log}")
    return p


def quantize(tc: Toolchain, src: str, dst: str, base_type: str, overrides: dict[str, str], token_embd_type: str | None,
             imatrix: str | None, log: str | None, threads: int = 8) -> None:
    tt = Path(dst).with_suffix(".tensor-types.txt")
    tt.write_text("\n".join(f"^{name.replace('.', chr(92) + '.')}$={t.lower()}" for name, t in sorted(overrides.items())) + "\n")
    cmd = [tc.llama_bin / "llama-quantize", "--pure", "--tensor-type-file", tt]
    if token_embd_type:
        cmd += ["--token-embedding-type", token_embd_type.lower()]
    if imatrix:
        cmd += ["--imatrix", imatrix]
    cmd += [src, dst, base_type.upper(), str(threads)]
    run(cmd, log, env=tc.env_for(""))
    tt.unlink(missing_ok=True)


def parallel_quantize(
    tc: Toolchain,
    src: str,
    dst: str,
    base_type: str,
    overrides: dict[str, str],
    token_embd_type: str | None,
    imatrix: str | None,
    log: str | None,
    threads: int = 8,
    shards: int = 1,
) -> None:
    """Bit-identical multi-process `llama-quantize --pure` across `shards` CPU workers + zero-copy mmap splice."""
    if shards <= 1:
        quantize(tc, src, dst, base_type, overrides, token_embd_type, imatrix, log, threads)
        return

    from . import gguf_ops

    hs_src = gguf_ops._Hdr(src)
    try:
        src_tensors = [dict(t) for t in hs_src.tensors]
    finally:
        hs_src.close()

    req_of: dict[str, str] = {}
    active: list[tuple[str, str, str, int, float]] = []
    for t in src_tensors:
        name = t["name"]
        stype = t["type"].upper()
        if name == "token_embd.weight":
            req = (token_embd_type or base_type).upper()
        elif name in overrides:
            req = overrides[name].upper()
        elif len(t["dims"]) >= 2 and stype in ("BF16", "F16", "F32"):
            req = base_type.upper()
        else:
            req = stype
        req_of[name] = req
        if len(t["dims"]) >= 2 and req != stype:
            mult = 4.0 if req.startswith("IQ") else (1.5 if "_K" in req else 1.0)
            active.append((name, req, stype, int(t["n_elements"]), float(t["n_elements"]) * mult))

    p_actual = min(int(shards), len(active))
    if p_actual <= 1:
        quantize(tc, src, dst, base_type, overrides, token_embd_type, imatrix, log, threads)
        return

    active.sort(key=lambda x: (-x[4], x[0]))
    buckets: list[list[tuple[str, str]]] = [[] for _ in range(p_actual)]
    loads = [0.0] * p_actual
    owner_of: dict[str, int] = {}
    for name, req, _stype, _ne, cost in active:
        k = min(range(p_actual), key=loads.__getitem__)
        buckets[k].append((name, req))
        loads[k] += cost
        owner_of[name] = k

    tot_load = sum(loads) or 1.0
    worker_threads = [max(2, min(32, round(threads * ld / tot_load))) for ld in loads]

    src_size = os.path.getsize(src)
    need_bytes = int(src_size * p_actual * 1.25)
    scratch_parent = Path(dst).resolve().parent
    scratch_parent.mkdir(parents=True, exist_ok=True)
    if os.path.isdir("/dev/shm") and os.access("/dev/shm", os.W_OK):
        try:
            if shutil.disk_usage("/dev/shm").free > need_bytes:
                scratch_parent = Path("/dev/shm")
        except OSError:
            pass

    with tempfile.TemporaryDirectory(prefix="chantless-pq-", dir=str(scratch_parent)) as tmp_dir:
        tmp_p = Path(tmp_dir)
        w_paths = [tmp_p / f"w{k}.gguf" for k in range(p_actual)]
        w_logs = [tmp_p / f"w{k}.log" for k in range(p_actual)]

        def run_worker(k: int) -> None:
            bset = {n: r for n, r in buckets[k]}
            w_ov: dict[str, str] = {}
            tok_arg: str | None = token_embd_type
            for t in src_tensors:
                n = t["name"]
                if len(t["dims"]) < 2:
                    continue
                if n == "token_embd.weight":
                    tok_arg = bset[n] if n in bset else t["type"]
                elif n in bset:
                    w_ov[n] = bset[n]
                else:
                    w_ov[n] = t["type"]
            quantize(tc, src, str(w_paths[k]), base_type, w_ov, tok_arg, imatrix, str(w_logs[k]), worker_threads[k])

        with ThreadPoolExecutor(p_actual) as ex:
            list(ex.map(run_worker, range(p_actual)))

        if log:
            with open(log, "a") as lf:
                for k in range(p_actual):
                    if w_logs[k].exists():
                        lf.write(f"--- parallel_quantize shard {k}/{p_actual} (threads={worker_threads[k]}) ---\n")
                        lf.write(w_logs[k].read_text())

        hdrs = [gguf_ops._Hdr(p) for p in w_paths]
        try:
            h0 = hdrs[0]
            align = h0.alignment
            ti = bytearray()
            off = 0
            plan: list[tuple[int, int, int, int]] = []
            for idx, t0 in enumerate(h0.tensors):
                w = owner_of.get(t0["name"], 0)
                tw = hdrs[w].tensors[idx]
                if tw["name"] != t0["name"]:
                    raise RuntimeError(f"shard tensor order mismatch at {idx}: {tw['name']} != {t0['name']}")
                nb = tw["name"].encode("utf-8")
                dims = tw["dims"]
                ti += struct.pack("<Q", len(nb)) + nb + struct.pack("<I", len(dims)) + struct.pack(f"<{len(dims)}Q", *dims)
                ti += struct.pack("<IQ", tw["ttype"], off)
                plan.append((w, hdrs[w].data_start + tw["offset"], tw["n_bytes"], off))
                off = (off + tw["n_bytes"] + align - 1) // align * align

            last0 = h0.tensors[-1]
            w0_eof_pad = os.path.getsize(w_paths[0]) - (h0.data_start + last0["offset"] + last0["n_bytes"])

            part = str(dst) + ".part"
            with open(part, "wb") as f:
                f.write(b"GGUF" + struct.pack("<IQQ", h0.version, len(h0.tensors), h0.n_kv))
                f.write(memoryview(h0.m)[h0.kv_start : h0.kv_end])
                f.write(ti)
                pos = f.tell()
                f.write(b"\0" * ((pos + align - 1) // align * align - pos))
                base = f.tell()
                for w, s0, nbytes, new_off in plan:
                    f.write(b"\0" * (new_off - (f.tell() - base)))
                    f.write(memoryview(hdrs[w].m)[s0 : s0 + nbytes])
                if w0_eof_pad > 0:
                    end_pos = f.tell()
                    f.write(b"\0" * ((end_pos + align - 1) // align * align - end_pos))
            os.replace(part, dst)
        finally:
            for h in hdrs:
                h.close()


def imatrix(tc: Toolchain, model: str, text_file: str, out: str, log: str | None, ctx: int = 512, gpu: str = "") -> None:
    run([tc.llama_bin / "llama-imatrix", "-m", model, "-f", text_file, "-o", out, "-c", str(ctx), "-ngl", "999",
         "--parse-special", "--no-ppl"], log, env=tc.env_for(gpu))


def trj_targets(path: str | os.PathLike) -> list[int]:
    """Number of targets of each trajectory of a TRJ1 file, in file order (tokens are skipped, not read)."""
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        head = f.read(8)
        if len(head) != 8 or head[:4] != b"TRJ1":
            raise ValueError(f"{path}: not a TRJ1 file")
        out = []
        for _ in range(struct.unpack("<I", head[4:])[0]):
            rec = f.read(8)
            if len(rec) != 8:
                raise ValueError(f"{path}: truncated")
            nt, ng = struct.unpack("<II", rec)
            if f.seek(4 * (nt + ng), 1) > size:  # seeking past EOF does not fail by itself
                raise ValueError(f"{path}: truncated")
            out.append(ng)
    return out


class BatchScorer:
    """logit[a] - logit[b] at every target of one or more TRJ1 files, from the logodds binary.

    With gpu=None files are sharded over every slot of every GPU (slots_per_gpu processes per GPU) and
    merged back in file order; with a gpu given, one process runs there. A missing value raises.
    """

    def __init__(self, tc: Toolchain, slots_per_gpu: int = 2, ubatch: int | None = None, n_seq: int | None = None):
        if tc.logodds is None:
            raise ValueError("toolchain has no logodds binary (toolchain.logodds)")
        self.tc, self.slots_per_gpu = tc, slots_per_gpu
        self.ubatch, self.n_seq = ubatch, n_seq

    def score_trjs(self, model: str, jobs: list[tuple[str, tuple[int, int]]], gpu: str | None = None) -> list[np.ndarray]:
        """Score multiple `(trj_path, (tok_a, tok_b))` files in one model load per GPU shard."""
        if not jobs:
            return []
        meta = []
        max_active_trajs = 0
        for trj, pair in jobs:
            counts = trj_targets(trj)
            offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
            out = np.full(int(offsets[-1]), np.nan)
            max_active_trajs = max(max_active_trajs, sum(1 for c in counts if c))
            meta.append((str(trj), pair, counts, offsets, out))
        if max_active_trajs == 0:
            return [m[4] for m in meta]

        gpus = [g for g in self.tc.gpus for _ in range(self.slots_per_gpu)] if gpu is None else [gpu]
        n = max(1, min(len(gpus), max_active_trajs))
        with tempfile.TemporaryDirectory(prefix="logodds-") as tmp:
            parts = [[Path(tmp) / f"f{fi}_s{i}.tsv" for i in range(n)] for fi in range(len(meta))]

            def one(i: int) -> None:
                cmd = [str(self.tc.logodds), "-m", str(model)]
                for fi, (trj, pair, _c, _off, _out) in enumerate(meta):
                    cmd += ["--traj", trj, "--pair", f"{pair[0]},{pair[1]}", "--out", str(parts[fi][i])]
                cmd += ["--shard", f"{i}/{n}"]
                if self.ubatch is not None:
                    cmd += ["--ubatch", str(self.ubatch)]
                if self.n_seq is not None:
                    cmd += ["--n-seq", str(self.n_seq)]
                p = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True,
                                   env=self.tc.env_for(gpus[i]))
                if p.returncode != 0:
                    tail = p.stderr.strip().splitlines()[-3:]
                    raise RuntimeError(f"logodds failed ({p.returncode}) shard {i}/{n}: {' | '.join(tail)}")

            with ThreadPoolExecutor(n) as ex:
                list(ex.map(one, range(n)))
            for fi, (trj, _pair, _counts, offsets, out) in enumerate(meta):
                if not len(out):
                    continue
                for part in parts[fi]:
                    for line in open(part):
                        ti, k, v = line.split("\t")
                        out[offsets[int(ti)] + int(k)] = float(v)
                miss = np.flatnonzero(~np.isfinite(out))
                if len(miss):
                    raise RuntimeError(f"logodds left {len(miss)} of {len(out)} targets of {trj} unscored or non-finite "
                                       f"(first at {miss[0]})")
        return [m[4] for m in meta]

    def score_trj(self, model: str, trj: str, pair: tuple[int, int], gpu: str | None = None) -> np.ndarray:
        return self.score_trjs(model, [(trj, pair)], gpu=gpu)[0]


def tokenize_docs(tc: Toolchain, model_gguf: str | os.PathLike, prompts: list[str], threads: int = 32) -> list[np.ndarray]:
    """Tokenize raw prompt strings using `model_gguf`'s own GGUF vocabulary via `tokdocs`."""
    if tc.tokdocs is None:
        raise ValueError("toolchain has no tokdocs binary (toolchain.tokdocs)")
    with tempfile.TemporaryDirectory(prefix="tokdocs-") as tmp:
        in_bin = Path(tmp) / "in.bin"
        out_bin = Path(tmp) / "out.bin"
        with open(in_bin, "wb") as f:
            f.write(struct.pack("<I", len(prompts)))
            for s in prompts:
                b = s.encode("utf-8")
                f.write(struct.pack("<I", len(b)) + b)
        p = subprocess.run([str(tc.tokdocs), str(model_gguf), str(in_bin), str(out_bin), str(threads)],
                           stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, env=tc.env_for(""))
        if p.returncode != 0:
            raise RuntimeError(f"tokdocs failed ({p.returncode}) on {model_gguf}: {p.stderr.strip()[-400:]}")
        raw = out_bin.read_bytes()
    (n,) = struct.unpack_from("<I", raw, 0)
    if n != len(prompts):
        raise RuntimeError(f"tokdocs returned {n} docs for {len(prompts)} inputs")
    pos, out = 4, []
    for _ in range(n):
        (l,) = struct.unpack_from("<I", raw, pos); pos += 4
        out.append(np.frombuffer(raw, dtype="<i4", count=l, offset=pos).copy())
        pos += 4 * l
    return out
