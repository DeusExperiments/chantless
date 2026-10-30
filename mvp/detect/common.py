"""Shared pieces for the detector run: pinned sources, the prompt, the 4,096-token rule, file I/O."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import pyarrow.parquet as pq

MAX_DOC_TOKENS = 4096
LICENCE_NOTE = "research-only: trained and evaluated on WildOutlines (CC BY-NC-SA 4.0)"

WILDOUTLINES = ("rishanthrajendhran/WildOutlines", "2fb690de919dcaf6a12f53d52b6f3ccacb5cd10d")
BEEMO = ("toloka/beemo", "9c014107fe9b85c4c784c1ce3a43b0b7b0a6d162")
MODELS = {
    "qwen": ("Qwen/Qwen3.5-0.8B", "2fc06364715b967f1860aea9cf38778875588b17"),
    "modernbert": ("rishanthrajendhran/ProseLens-ModernBERT-L", "c00074fd13f8603d246d2338f79d60498c52b4e8"),
    "proselens": ("rishanthrajendhran/ProseLens", "f3018f8241847b5d3a6c210356813818ccdc3fc1"),
}

# ProseLens model-card prompt, as used in the earlier full-precision run. The trained Qwen uses the same
# prompt, so both causal models answer the same question in the same chat format.
SYSTEM = ("Given a document, answer with one word: human if the document was "
          "human-written, ai if it was AI-generated.")
SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
PROSELENS_IDS = (50755, 2464)  # "human", "ai" in the ProseLens tokenizer


def chat_prompt(text: str) -> str:
    return f"<|im_start|>system\n{SYSTEM}<|im_end|>\n<|im_start|>user\n{text}{SUFFIX}"


def head_text(tokenizer, text: str, n: int = MAX_DOC_TOKENS) -> tuple[str, bool]:
    """The document cut after its first n tokens under this model's tokenizer; untouched if shorter."""
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    if len(enc["input_ids"]) <= n:
        return text, False
    return text[:enc["offset_mapping"][n - 1][1]], True


def answer_ids(tokenizer) -> tuple[int, int]:
    """Single-token ids for "human" and "ai" right after the assistant header."""
    ids = [tokenizer.encode(w, add_special_tokens=False) for w in ("human", "ai")]
    if any(len(i) != 1 for i in ids):
        raise RuntimeError(f"answer words are not single tokens: {ids}")
    return ids[0][0], ids[1][0]


def read_rows(path: str | Path) -> list[dict[str, Any]]:
    return pq.read_table(path).to_pylist()


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def append_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    with open(path, "a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def write_json(path: str | Path, value: Any) -> None:
    Path(path).write_text(json.dumps(value, indent=1, default=str) + "\n")


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()
