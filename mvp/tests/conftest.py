"""Shared test path helpers for Chantless local integration tests."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def resolve_toolchain() -> tuple[Path, Path]:
    """Locate `(llama_dir, tools_bin)` from env override or local `.local/{toolchain,mvp}`."""
    candidates: list[tuple[Path, Path]] = []
    if env_prefix := os.environ.get("CHANTLESS_TOOLCHAIN_PREFIX"):
        p = Path(env_prefix).expanduser().resolve()
        candidates.append((p / "llama-a01dff9", p / "bin"))
        candidates.append((p / "llama-a01dff9", p / "bin-a01dff9"))
    candidates.extend([
        (ROOT / ".local/toolchain/llama-a01dff9", ROOT / ".local/toolchain/bin"),
        (ROOT / ".local/mvp/llama-a01dff9", ROOT / ".local/mvp/bin-a01dff9"),
    ])
    for llama_dir, tools_bin in candidates:
        if (llama_dir / "build/bin/llama-quantize").exists() and (tools_bin / "traj-kl").exists():
            return llama_dir, tools_bin
    return candidates[0]


def resolve_model_dir(name: str = "qwen3-0.6b") -> Path:
    """Locate tokenizer/HF reference directory from env override or `.local/{models,mvp/models}`."""
    if env_dir := os.environ.get("CHANTLESS_MODEL_DIR"):
        return Path(env_dir).expanduser().resolve()
    for cand in (ROOT / ".local/models" / name, ROOT / ".local/mvp/models" / name):
        if cand.exists():
            return cand
    return ROOT / ".local/models" / name
