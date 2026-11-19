from __future__ import annotations

import json
import os
import queue
import shlex
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from .config import SweepConfig, deep_merge, load_raw_with_extends, validate_dict


@dataclass
class ExpandedCell:
    id: str
    command: str
    config: dict[str, Any]
    config_path: Path
    export_dir: Path
    out_mask: Path
    out_gguf: Path | None
    upto: str | None = None
    fold: int | None = None
    gpus: list[str] = field(default_factory=lambda: [""])


@dataclass
class CellResult:
    id: str
    command: str
    status: str
    returncode: int
    wall_s: float
    gpus: list[str]
    export_dir: str
    config_path: str
    mask_path: str | None = None
    gguf_path: str | None = None
    detail: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@runtime_checkable
class SweepLauncher(Protocol):
    name: str

    def plan(self, sweep: SweepConfig, cells: list[ExpandedCell]) -> list[dict[str, Any]]: ...
    def run(self, sweep: SweepConfig, cells: list[ExpandedCell], dry_run: bool = False) -> dict[str, Any]: ...


_LAUNCHERS: dict[str, SweepLauncher] = {}


def register_launcher(launcher: SweepLauncher) -> None:
    _LAUNCHERS[launcher.name] = launcher


def get_launcher(name: str) -> SweepLauncher:
    if name not in _LAUNCHERS:
        raise ValueError(f"unknown sweep launcher {name!r}; registered: {sorted(_LAUNCHERS)}")
    return _LAUNCHERS[name]


def list_launchers() -> list[str]:
    return sorted(_LAUNCHERS)


def _discover_local_gpus(spec_gpus: list[str] | None) -> list[str]:
    if spec_gpus is not None and len(spec_gpus) > 0:
        return [str(g) for g in spec_gpus]
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if cvd:
        return [g.strip() for g in cvd.split(",") if g.strip()]
    return [""]


def _cell_cli_cmd(cell: ExpandedCell, python_exe: str = sys.executable) -> list[str]:
    cmd = [python_exe, "-m", "chantless.cli", cell.command, "--config", str(cell.config_path)]
    if cell.command == "extract":
        if cell.upto:
            cmd += ["--upto", cell.upto]
        cmd += ["--out-mask", str(cell.out_mask)]
        if cell.out_gguf is not None:
            cmd += ["--out", str(cell.out_gguf)]
    elif cell.command == "learn":
        if cell.fold is not None:
            cmd += ["--fold", str(cell.fold)]
        cmd += ["--out-mask", str(cell.out_mask)]
        if cell.out_gguf is not None:
            cmd += ["--out", str(cell.out_gguf)]
    return cmd


def expand_cells(sweep: SweepConfig, sweep_file: str | Path | None = None) -> list[ExpandedCell]:
    base_dir = Path(sweep_file).resolve().parent if sweep_file else Path.cwd()
    base_raw: dict[str, Any] = {}
    if sweep.base_config:
        bp = Path(sweep.base_config)
        if not bp.is_absolute():
            bp = (base_dir / bp).resolve()
        base_raw = load_raw_with_extends(bp)
    if sweep.base:
        base_raw = deep_merge(base_raw, sweep.base)

    export_root = Path(sweep.export_dir)
    if not export_root.is_absolute():
        export_root = (base_dir / export_root).resolve()

    avail_gpus = _discover_local_gpus(sweep.launcher.gpus)
    gpc = sweep.launcher.gpus_per_cell
    gpu_slices: list[list[str]] = []
    if len(avail_gpus) >= gpc:
        for i in range(0, len(avail_gpus) - gpc + 1, gpc):
            gpu_slices.append(avail_gpus[i : i + gpc])
    if not gpu_slices:
        gpu_slices = [avail_gpus[:gpc] or [""]]

    expanded: list[ExpandedCell] = []
    for idx, cell in enumerate(sweep.cells):
        cmd = cell.command or sweep.command
        merged = deep_merge(base_raw, cell.overrides)
        c_export = export_root / cell.id
        c_gpus = gpu_slices[idx % len(gpu_slices)]

        if cmd == "extract":
            merged.setdefault("run_id", f"{sweep.sweep_id}-{cell.id}")
            if cell.overrides.get("run_id") is None and "run_id" in base_raw:
                merged["run_id"] = f"{base_raw['run_id']}-{cell.id}"
            merged.setdefault("toolchain", {})["gpus"] = list(c_gpus)
        elif cmd == "learn":
            merged.setdefault("toolchain", {})["gpus"] = list(c_gpus)
            if "learn" in merged and isinstance(merged["learn"], dict):
                merged["learn"].setdefault("out_dir", str(c_export / "learn_work"))
                if cell.overrides.get("learn", {}).get("out_dir") is None and "out_dir" in base_raw.get("learn", {}):
                    merged["learn"]["out_dir"] = str(Path(base_raw["learn"]["out_dir"]) / cell.id)

        validated = validate_dict(merged, cmd)
        cfg_dict = validated.model_dump(exclude_none=True)
        cfg_path = c_export / "cell_config.json"
        out_mask = c_export / f"{cell.id}-mask.json"
        out_gguf = None if cell.upto else (c_export / f"{cell.id}.gguf")
        expanded.append(
            ExpandedCell(
                id=cell.id,
                command=cmd,
                config=cfg_dict,
                config_path=cfg_path,
                export_dir=c_export,
                out_mask=out_mask,
                out_gguf=out_gguf,
                upto=cell.upto,
                fold=cell.fold,
                gpus=list(c_gpus),
            )
        )
    return expanded


def _write_cell_config(cell: ExpandedCell) -> None:
    cell.export_dir.mkdir(parents=True, exist_ok=True)
    tmp = cell.config_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(cell.config, indent=2), encoding="utf-8")
    os.replace(tmp, cell.config_path)


def _write_done_receipt(cell: ExpandedCell, res: CellResult) -> None:
    cell.export_dir.mkdir(parents=True, exist_ok=True)
    done_p = cell.export_dir / "done.json"
    tmp = done_p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(res.to_dict(), indent=2), encoding="utf-8")
    os.replace(tmp, done_p)


class LocalLauncher:
    name = "local"

    def plan(self, sweep: SweepConfig, cells: list[ExpandedCell]) -> list[dict[str, Any]]:
        plans = []
        for c in cells:
            done_p = c.export_dir / "done.json"
            plans.append(
                {
                    "id": c.id,
                    "backend": self.name,
                    "command": c.command,
                    "gpus": c.gpus,
                    "will_skip_resume": bool(sweep.resume and done_p.exists()),
                    "export_dir": str(c.export_dir),
                    "argv": _cell_cli_cmd(c),
                }
            )
        return plans

    def run(self, sweep: SweepConfig, cells: list[ExpandedCell], dry_run: bool = False) -> dict[str, Any]:
        plans = self.plan(sweep, cells)
        if dry_run:
            return {"sweep_id": sweep.sweep_id, "backend": self.name, "dry_run": True, "cells": plans}

        avail_gpus = _discover_local_gpus(sweep.launcher.gpus)
        gpc = sweep.launcher.gpus_per_cell
        slots: list[list[str]] = []
        if len(avail_gpus) >= gpc:
            for i in range(0, len(avail_gpus) - gpc + 1, gpc):
                slots.append(avail_gpus[i : i + gpc])
        if not slots:
            slots = [avail_gpus[:gpc] or [""]]
        max_workers = max(1, min(sweep.launcher.max_parallel, len(slots), len(cells)))
        slot_q: queue.Queue[list[str]] = queue.Queue()
        for sl in slots[:max_workers]:
            slot_q.put(sl)

        stop_ev = threading.Event()
        results: dict[str, CellResult] = {}

        def _exec(cell: ExpandedCell) -> CellResult:
            done_p = cell.export_dir / "done.json"
            if sweep.resume and done_p.exists():
                prev = json.loads(done_p.read_text(encoding="utf-8"))
                return CellResult(
                    id=cell.id,
                    command=cell.command,
                    status="skipped_resume",
                    returncode=0,
                    wall_s=0.0,
                    gpus=prev.get("gpus", cell.gpus),
                    export_dir=str(cell.export_dir),
                    config_path=str(cell.config_path),
                    mask_path=prev.get("mask_path"),
                    gguf_path=prev.get("gguf_path"),
                    detail="resumed from existing done.json",
                )
            if stop_ev.is_set():
                return CellResult(
                    id=cell.id,
                    command=cell.command,
                    status="aborted_stop_on_failure",
                    returncode=-1,
                    wall_s=0.0,
                    gpus=[],
                    export_dir=str(cell.export_dir),
                    config_path=str(cell.config_path),
                )
            assigned = slot_q.get()
            t0 = time.time()
            try:
                cell.gpus = list(assigned)
                if "toolchain" in cell.config:
                    cell.config["toolchain"]["gpus"] = list(assigned)
                _write_cell_config(cell)
                cmd = _cell_cli_cmd(cell)
                log_p = cell.export_dir / "cell.log"
                env = dict(os.environ)
                if assigned and assigned != [""]:
                    env["CUDA_VISIBLE_DEVICES"] = ",".join(assigned)
                with open(log_p, "w", encoding="utf-8") as lf:
                    proc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env)
                dt = time.time() - t0
                ok = proc.returncode == 0
                res = CellResult(
                    id=cell.id,
                    command=cell.command,
                    status="ok" if ok else "failed",
                    returncode=int(proc.returncode),
                    wall_s=round(dt, 3),
                    gpus=list(assigned),
                    export_dir=str(cell.export_dir),
                    config_path=str(cell.config_path),
                    mask_path=str(cell.out_mask) if cell.out_mask.exists() else None,
                    gguf_path=str(cell.out_gguf) if (cell.out_gguf and cell.out_gguf.exists()) else None,
                    detail=None if ok else f"see {log_p}",
                )
                if ok:
                    _write_done_receipt(cell, res)
                elif sweep.stop_on_failure:
                    stop_ev.set()
                return res
            finally:
                slot_q.put(assigned)

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futs = {pool.submit(_exec, c): c.id for c in cells}
            for f in as_completed(futs):
                r = f.result()
                results[r.id] = r

        ordered = [results[c.id].to_dict() for c in cells]
        summary = {
            "sweep_id": sweep.sweep_id,
            "backend": self.name,
            "dry_run": False,
            "ok": all(r["status"] in ("ok", "skipped_resume") for r in ordered),
            "cells": ordered,
        }
        export_root = Path(cells[0].export_dir).parent
        export_root.mkdir(parents=True, exist_ok=True)
        (export_root / "sweep_receipt.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary


class SSHLauncher:
    name = "ssh"

    def plan(self, sweep: SweepConfig, cells: list[ExpandedCell]) -> list[dict[str, Any]]:
        hosts = sweep.launcher.hosts or ["localhost"]
        rwd = sweep.launcher.remote_workdir or "."
        py = sweep.launcher.remote_python
        plans = []
        for i, c in enumerate(cells):
            host = hosts[i % len(hosts)]
            rcfg = f"{rwd}/sweep-{sweep.sweep_id}/{c.id}/cell_config.json"
            rexport = f"{rwd}/sweep-{sweep.sweep_id}/{c.id}"
            cli = f"{shlex.quote(py)} -m chantless.cli {c.command} --config {shlex.quote(rcfg)} --out-mask {shlex.quote(rexport + '/' + c.id + '-mask.json')}"
            if c.upto:
                cli += f" --upto {shlex.quote(c.upto)}"
            if c.fold is not None:
                cli += f" --fold {c.fold}"
            plans.append(
                {
                    "id": c.id,
                    "backend": self.name,
                    "host": host,
                    "remote_export_dir": rexport,
                    "local_export_dir": str(c.export_dir),
                    "remote_cmd": f"cd {shlex.quote(rwd)} && {cli}",
                }
            )
        return plans

    def run(self, sweep: SweepConfig, cells: list[ExpandedCell], dry_run: bool = False) -> dict[str, Any]:
        plans = self.plan(sweep, cells)
        if dry_run:
            return {"sweep_id": sweep.sweep_id, "backend": self.name, "dry_run": True, "cells": plans}
        if not sweep.launcher.hosts:
            raise ValueError("ssh launcher requires launcher.hosts")
        if not shutil.which("ssh") or not shutil.which("scp"):
            raise RuntimeError("ssh/scp binaries not found in PATH")
        results = []
        for c, pl in zip(cells, plans):
            done_p = c.export_dir / "done.json"
            if sweep.resume and done_p.exists():
                results.append(CellResult(c.id, c.command, "skipped_resume", 0, 0.0, c.gpus, str(c.export_dir), str(c.config_path)).to_dict())
                continue
            _write_cell_config(c)
            t0 = time.time()
            host = pl["host"]
            rexport = pl["remote_export_dir"]
            subprocess.run(["ssh", host, f"mkdir -p {shlex.quote(rexport)}"], check=True)
            subprocess.run(["scp", str(c.config_path), f"{host}:{rexport}/cell_config.json"], check=True)
            p = subprocess.run(["ssh", host, pl["remote_cmd"]])
            ok = p.returncode == 0
            if ok:
                subprocess.run(["scp", "-r", f"{host}:{rexport}/.", str(c.export_dir)], check=False)
            res = CellResult(c.id, c.command, "ok" if ok else "failed", int(p.returncode), round(time.time() - t0, 3), c.gpus, str(c.export_dir), str(c.config_path))
            if ok:
                _write_done_receipt(c, res)
            results.append(res.to_dict())
            if not ok and sweep.stop_on_failure:
                break
        return {"sweep_id": sweep.sweep_id, "backend": self.name, "dry_run": False, "ok": all(r["status"] in ("ok", "skipped_resume") for r in results), "cells": results}


class ModalLauncher:
    name = "modal"

    def plan(self, sweep: SweepConfig, cells: list[ExpandedCell]) -> list[dict[str, Any]]:
        app = sweep.launcher.modal_app or "chantless_modal.py"
        gpu = sweep.launcher.modal_gpu or "A100-80GB:1"
        vol = sweep.launcher.modal_volume or f"chantless-{sweep.sweep_id}"
        plans = []
        for c in cells:
            argv = ["modal", "run", app, "--cell-id", c.id, "--config", str(c.config_path), "--gpu", gpu, "--volume", vol, *sweep.launcher.extra_args]
            plans.append({"id": c.id, "backend": self.name, "gpu": gpu, "volume": vol, "export_dir": str(c.export_dir), "argv": argv})
        return plans

    def run(self, sweep: SweepConfig, cells: list[ExpandedCell], dry_run: bool = False) -> dict[str, Any]:
        plans = self.plan(sweep, cells)
        if dry_run:
            return {"sweep_id": sweep.sweep_id, "backend": self.name, "dry_run": True, "cells": plans}
        if not shutil.which("modal"):
            raise RuntimeError("modal CLI not found in PATH")
        if not sweep.launcher.modal_app:
            raise ValueError("modal launcher requires launcher.modal_app path")
        results = []
        for c, pl in zip(cells, plans):
            if sweep.resume and (c.export_dir / "done.json").exists():
                results.append(CellResult(c.id, c.command, "skipped_resume", 0, 0.0, c.gpus, str(c.export_dir), str(c.config_path)).to_dict())
                continue
            _write_cell_config(c)
            t0 = time.time()
            p = subprocess.run(pl["argv"])
            ok = p.returncode == 0
            res = CellResult(c.id, c.command, "ok" if ok else "failed", int(p.returncode), round(time.time() - t0, 3), c.gpus, str(c.export_dir), str(c.config_path))
            if ok:
                _write_done_receipt(c, res)
            results.append(res.to_dict())
            if not ok and sweep.stop_on_failure:
                break
        return {"sweep_id": sweep.sweep_id, "backend": self.name, "dry_run": False, "ok": all(r["status"] in ("ok", "skipped_resume") for r in results), "cells": results}


class ColabLauncher:
    name = "colab"

    def plan(self, sweep: SweepConfig, cells: list[ExpandedCell]) -> list[dict[str, Any]]:
        sessions = sweep.launcher.colab_sessions or ["default"]
        rwd = sweep.launcher.remote_workdir or "/content/chantless"
        plans = []
        for i, c in enumerate(cells):
            sess = sessions[i % len(sessions)]
            rcfg = f"{rwd}/sweep-{sweep.sweep_id}/{c.id}/cell_config.json"
            cli = f"python3 -m chantless.cli {c.command} --config {shlex.quote(rcfg)} --out-mask {shlex.quote(rwd + '/export/' + c.id + '-mask.json')}"
            if c.upto:
                cli += f" --upto {shlex.quote(c.upto)}"
            argv = ["colab", "--auth=oauth2", "exec", "-s", sess, cli, *sweep.launcher.extra_args]
            plans.append({"id": c.id, "backend": self.name, "session": sess, "export_dir": str(c.export_dir), "argv": argv})
        return plans

    def run(self, sweep: SweepConfig, cells: list[ExpandedCell], dry_run: bool = False) -> dict[str, Any]:
        plans = self.plan(sweep, cells)
        if dry_run:
            return {"sweep_id": sweep.sweep_id, "backend": self.name, "dry_run": True, "cells": plans}
        if not shutil.which("colab"):
            raise RuntimeError("colab CLI not found in PATH")
        results = []
        for c, pl in zip(cells, plans):
            if sweep.resume and (c.export_dir / "done.json").exists():
                results.append(CellResult(c.id, c.command, "skipped_resume", 0, 0.0, c.gpus, str(c.export_dir), str(c.config_path)).to_dict())
                continue
            _write_cell_config(c)
            t0 = time.time()
            p = subprocess.run(pl["argv"])
            ok = p.returncode == 0
            res = CellResult(c.id, c.command, "ok" if ok else "failed", int(p.returncode), round(time.time() - t0, 3), c.gpus, str(c.export_dir), str(c.config_path))
            if ok:
                _write_done_receipt(c, res)
            results.append(res.to_dict())
            if not ok and sweep.stop_on_failure:
                break
        return {"sweep_id": sweep.sweep_id, "backend": self.name, "dry_run": False, "ok": all(r["status"] in ("ok", "skipped_resume") for r in results), "cells": results}


register_launcher(LocalLauncher())
register_launcher(SSHLauncher())
register_launcher(ModalLauncher())
register_launcher(ColabLauncher())


def run_sweep(
    sweep: SweepConfig,
    sweep_file: str | Path | None = None,
    launcher_override: str | None = None,
    export_dir_override: str | None = None,
    dry_run: bool = False,
    resume_override: bool | None = None,
    stop_on_failure_override: bool | None = None,
) -> dict[str, Any]:
    if export_dir_override:
        sweep = sweep.model_copy(update={"export_dir": export_dir_override})
    if resume_override is not None:
        sweep = sweep.model_copy(update={"resume": resume_override})
    if stop_on_failure_override is not None:
        sweep = sweep.model_copy(update={"stop_on_failure": stop_on_failure_override})
    backend = launcher_override or sweep.launcher.backend
    launcher = get_launcher(backend)
    cells = expand_cells(sweep, sweep_file=sweep_file)
    return launcher.run(sweep, cells, dry_run=dry_run)
