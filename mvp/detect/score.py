"""Score documents with one detector and write p(human) per document.

Usage: python detect/score.py MODEL OUT DATA... [--path P] [--shard I --nshard N] [--batch B] [--preflight]
MODEL is qwen, modernbert or proselens; --path points qwen at the trained model (default: the base model).
Each DATA parquet is scored into OUT/<file stem>.shard<I>.jsonl with id, label, kind, format, p_human, tokens,
truncated. Rows I, I+N, I+2N, ... go to shard I, so N processes, one per GPU, cover each file exactly once.
Results are appended per batch and a rerun skips ids already written. Every model sees the same document,
cut after its first 4,096 tokens under its own tokenizer.
--preflight scores the first two rows of the first file one at a time and as one batch, prints both and a rough
projected time for this shard, and exits non-zero if --batch > 1 and the two disagree by more than 0.02.
"""
from __future__ import annotations

import argparse
import platform
import sys
import time
from pathlib import Path

import torch
import transformers
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer

from common import (LICENCE_NOTE, MODELS, PROSELENS_IDS, append_jsonl, answer_ids, chat_prompt, head_text,
                    read_jsonl, read_rows, write_json)

DEV = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


def pad_batch(encoded, pad, left, multiple=1):
    n = -(-max(map(len, encoded)) // multiple) * multiple
    ids = torch.full((len(encoded), n), pad)
    mask = torch.zeros((len(encoded), n), dtype=torch.long)
    for k, e in enumerate(encoded):
        span = slice(n - len(e), n) if left else slice(0, len(e))
        ids[k, span] = torch.tensor(e)
        mask[k, span] = 1
    return ids.to(DEV), mask.to(DEV)


class Causal:
    """qwen and proselens: p(human) from the "human" and "ai" logits after the chat prompt."""

    def __init__(self, name, path):
        repo, rev = MODELS[name]
        src = {"pretrained_model_name_or_path": path} if path else {"pretrained_model_name_or_path": repo, "revision": rev}
        self.tok = AutoTokenizer.from_pretrained(**src)
        # straight to the GPU: eight processes staging 66 GB of ProseLens in host memory would not fit
        self.model = AutoModelForCausalLM.from_pretrained(**src, dtype=torch.bfloat16, device_map=str(DEV)).eval()
        self.ids = list(PROSELENS_IDS) if name == "proselens" else list(answer_ids(self.tok))
        words = [self.tok.decode([i]) for i in self.ids]
        if words != ["human", "ai"]:
            raise RuntimeError(f"unexpected class tokens: {words}")
        self.pad = self.tok.pad_token_id if self.tok.pad_token_id is not None else self.tok.eos_token_id
        # Qwen's fla kernels re-tune for every new length; a few padded lengths are tuned once
        self.multiple = 256 if name == "qwen" else 1

    def encode(self, text):
        doc, cut = head_text(self.tok, text)
        return self.tok.encode(chat_prompt(doc), add_special_tokens=False), cut

    @torch.inference_mode()
    def p_human(self, encoded):
        ids, mask = pad_batch(encoded, self.pad, left=True, multiple=self.multiple)
        logits = self.model(input_ids=ids, attention_mask=mask, logits_to_keep=1).logits[:, -1][:, self.ids]
        return torch.softmax(logits.float(), -1)[:, 0].tolist()


class Encoder:
    """modernbert: p(human) is class 0 of the sequence classifier, loaded as in the earlier pilot (FP32, SDPA)."""

    def __init__(self, name, path):
        repo, rev = MODELS[name]
        self.tok = AutoTokenizer.from_pretrained(repo, revision=rev)
        config = AutoConfig.from_pretrained(repo, revision=rev)
        if hasattr(config, "reference_compile"):
            config.reference_compile = False
        print("id2label:", config.id2label, flush=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            repo, revision=rev, config=config, dtype=torch.float32, attn_implementation="sdpa").to(DEV).eval()
        self.ids = None

    def encode(self, text):
        doc, cut = head_text(self.tok, text)
        return self.tok.encode(doc, add_special_tokens=True), cut

    @torch.inference_mode()
    def p_human(self, encoded):
        ids, mask = pad_batch(encoded, self.tok.pad_token_id, left=False)
        return torch.softmax(self.model(input_ids=ids, attention_mask=mask).logits.float(), -1)[:, 0].tolist()


ap = argparse.ArgumentParser()
ap.add_argument("model", choices=sorted(MODELS))
ap.add_argument("out", type=Path)
ap.add_argument("data", type=Path, nargs="+")
ap.add_argument("--path", default=None)
ap.add_argument("--shard", type=int, default=0)
ap.add_argument("--nshard", type=int, default=1)
ap.add_argument("--batch", type=int, default=16)
ap.add_argument("--preflight", action="store_true")
args = ap.parse_args()
args.out.mkdir(parents=True, exist_ok=True)

det = (Encoder if args.model == "modernbert" else Causal)(args.model, args.path)
shards = {p: read_rows(p)[args.shard::args.nshard] for p in args.data}

if args.preflight:
    rows = shards[args.data[0]][:2]
    enc = [det.encode(r["text"])[0] for r in rows]
    t0 = time.time()
    single = [det.p_human([e])[0] for e in enc]
    per_doc = (time.time() - t0) / len(enc)
    batched = det.p_human(enc)
    gap = max(abs(a - b) for a, b in zip(single, batched))
    total = sum(map(len, shards.values()))
    print(f"preflight {args.model}: tokens {[len(e) for e in enc]} single {single} batched {batched} gap {gap:.4f}; "
          f"rough time for {total} docs at batch {args.batch}: {per_doc * total / max(1, args.batch) / 60:.0f} to "
          f"{per_doc * total / 60:.0f} min", flush=True)
    sys.exit(1 if args.batch > 1 and gap > 0.02 else 0)

if args.shard == 0:
    write_json(args.out / "score.json", {
        "model": args.model, "source": args.path or MODELS[args.model], "class_token_ids": det.ids,
        "batch": args.batch, "nshard": args.nshard, "data": [str(p) for p in args.data],
        "python": sys.version, "platform": platform.platform(), "torch": torch.__version__,
        "transformers": transformers.__version__,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else str(DEV), "licence": LICENCE_NOTE})

for path, rows in shards.items():
    dest = args.out / f"{path.stem}.shard{args.shard}.jsonl"
    done = {r["id"] for r in read_jsonl(dest)} if dest.exists() else set()
    todo = sorted((r for r in rows if r["id"] not in done), key=lambda r: -len(r["text"]))
    t0 = time.time()
    for k in range(0, len(todo), args.batch):
        chunk = todo[k:k + args.batch]
        enc = [det.encode(r["text"]) for r in chunk]
        p = det.p_human([e for e, _ in enc])
        append_jsonl(dest, ({"id": r["id"], "label": r["label"], "kind": r["kind"], "format": r["format"],
                             "p_human": ph, "tokens": len(e), "truncated": cut}
                            for r, (e, cut), ph in zip(chunk, enc, p)))
    print(f"{args.model} shard {args.shard} {path.stem}: {len(todo)} new, {len(done)} kept, "
          f"{time.time() - t0:.0f}s", flush=True)
