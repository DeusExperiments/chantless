"""Small shared helpers: hashing, atomic JSON, JSONL logs, deterministic seeds."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Iterable


def sha256_file(path: str | os.PathLike, chunk: int = 8 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def stable_seed(*parts: Any, bits: int = 31) -> int:
    """Deterministic seed from arbitrary parts (manual §6.1: hash(run_id, episode_id, turn))."""
    h = hashlib.sha256("\x1f".join(map(str, parts)).encode()).digest()
    return int.from_bytes(h[:8], "little") % (1 << bits)


def write_json(path: str | os.PathLike, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".")
    with os.fdopen(fd, "w") as f:
        json.dump(value, f, indent=1, sort_keys=False, default=str)
        f.write("\n")
    os.replace(tmp, path)


def read_json(path: str | os.PathLike) -> Any:
    return json.loads(Path(path).read_text())


def append_jsonl(path: str | os.PathLike, row: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(row, default=str) + "\n")


def read_jsonl(path: str | os.PathLike) -> list[Any]:
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


class Log:
    """Human run log (manual §11.2 log.md) plus stderr echo."""

    def __init__(self, path: str | os.PathLike | None):
        self.path = Path(path) if path else None

    def __call__(self, msg: str) -> None:
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {msg}"
        print(line, flush=True)
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a") as f:
                f.write(line + "\n")


def chunked(xs: list, n: int) -> Iterable[list]:
    for i in range(0, len(xs), n):
        yield xs[i : i + n]
