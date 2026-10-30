"""llama.cpp process wrappers: llama-server lifecycle, llama-quantize, llama-imatrix, traj-kl, logodds."""
from __future__ import annotations

import json
import os
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

    @classmethod
    def from_config(cls, cfg: dict) -> "Toolchain":
        return cls(Path(cfg["llama_bin"]), Path(cfg["traj_kl"]), [str(g) for g in cfg.get("gpus", [""])],
                   Path(cfg["logodds"]) if cfg.get("logodds") else None)

    def env_for(self, gpu: str) -> dict:
        env = dict(os.environ)
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
    run(cmd, log)
    tt.unlink(missing_ok=True)


def imatrix(tc: Toolchain, model: str, text_file: str, out: str, log: str | None, ctx: int = 512, gpu: str = "") -> None:
    run([tc.llama_bin / "llama-imatrix", "-m", model, "-f", text_file, "-o", out, "-c", str(ctx), "-ngl", "999",
         "--parse-special", "--no-ppl"], log, env=tc.env_for(gpu))


def trj_targets(path: str | os.PathLike) -> list[int]:
    """Number of targets of each trajectory of a TRJ1 file, in file order (tokens are skipped, not read)."""
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
            f.seek(4 * nt, 1)
            if len(f.read(4 * ng)) != 4 * ng:
                raise ValueError(f"{path}: truncated")
            out.append(ng)
    return out


class BatchScorer:
    """logit[a] - logit[b] at every target of a TRJ1 file, from the logodds binary.

    With gpu=None one file is sharded over every slot of every GPU (slots_per_gpu processes per GPU) and
    merged back in file order; with a gpu given, one process runs there. A missing value raises.
    """

    def __init__(self, tc: Toolchain, slots_per_gpu: int = 2):
        if tc.logodds is None:
            raise ValueError("toolchain has no logodds binary (toolchain.logodds)")
        self.tc, self.slots_per_gpu = tc, slots_per_gpu

    def score_trj(self, model: str, trj: str, pair: tuple[int, int], gpu: str | None = None) -> np.ndarray:
        counts = trj_targets(trj)
        offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
        out = np.full(int(offsets[-1]), np.nan)
        if not len(out):
            return out
        gpus = [g for g in self.tc.gpus for _ in range(self.slots_per_gpu)] if gpu is None else [gpu]
        n = max(1, min(len(gpus), sum(1 for c in counts if c)))  # never start a process with nothing to score
        with tempfile.TemporaryDirectory(prefix="logodds-") as tmp:
            parts = [Path(tmp) / f"{i}.tsv" for i in range(n)]

            def one(i):
                cmd = [str(self.tc.logodds), "-m", str(model), "--traj", str(trj), "--pair", f"{pair[0]},{pair[1]}",
                       "--out", str(parts[i]), "--shard", f"{i}/{n}"]
                p = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True,
                                   env=self.tc.env_for(gpus[i]))
                if p.returncode != 0:
                    tail = p.stderr.strip().splitlines()[-3:]
                    raise RuntimeError(f"logodds failed ({p.returncode}) on {trj} shard {i}/{n}: {' | '.join(tail)}")

            with ThreadPoolExecutor(n) as ex:
                list(ex.map(one, range(n)))
            for part in parts:
                for line in open(part):
                    ti, k, v = line.split("\t")
                    out[offsets[int(ti)] + int(k)] = float(v)
        miss = np.flatnonzero(np.isnan(out))
        if len(miss):
            raise RuntimeError(f"logodds left {len(miss)} of {len(out)} targets of {trj} unscored (first at {miss[0]})")
        return out
