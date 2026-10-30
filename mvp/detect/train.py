"""Fine-tune Qwen3.5-0.8B to answer "human" or "ai" for a document: one pass over WildOutlines train.

Usage: torchrun --nproc-per-node 8 detect/train.py DATA OUT [--batch 8] [--lr 1e-5] [--max-steps N]
DATA holds prep.py's parquet files; the model is saved to OUT/final.
The prompt is the ProseLens chat prompt. The loss is two-way cross-entropy over the "human" and "ai" logits at
the answer position, so training and scoring read the same two numbers. Full fine-tune with FP32 master weights
and BF16 autocast. Left padding keeps the answer position last. Batches are grouped by length and the longest
batch runs first, so an out-of-memory error shows at step 1. --max-steps is for the preflight: a few steps on
the real data to check memory and pace. A rerun resumes from the newest checkpoint in OUT.
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

# One Triton cache per rank: eight ranks compiling the same fla kernel into one cache race and crash at step 1
os.environ.setdefault("TRITON_CACHE_DIR", str(Path.home() / f".triton/cache-rank{os.environ.get('LOCAL_RANK', '0')}"))

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainerCallback, TrainingArguments
from transformers.trainer_pt_utils import LengthGroupedSampler

from common import LICENCE_NOTE, MODELS, answer_ids, chat_prompt, head_text, write_json

ap = argparse.ArgumentParser()
ap.add_argument("data", type=Path)
ap.add_argument("out", type=Path)
ap.add_argument("--batch", type=int, default=8)
ap.add_argument("--lr", type=float, default=1e-5)
ap.add_argument("--max-steps", type=int, default=-1)
args = ap.parse_args()
full = args.max_steps < 0

if torch.cuda.is_available():
    import fla  # noqa: F401  fast Gated DeltaNet kernels; the PyTorch fallback is far too slow for this budget

repo, rev = MODELS["qwen"]
tok = AutoTokenizer.from_pretrained(repo, revision=rev)
HUMAN, AI = answer_ids(tok)
PAD = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id


def stamp(msg):
    if os.environ.get("LOCAL_RANK", "0") == "0":
        print(time.strftime("%H:%M:%S"), msg, flush=True)


def encode(batch):
    ids = [tok.encode(chat_prompt(head_text(tok, t)[0]), add_special_tokens=False) for t in batch["text"]]
    return {"input_ids": ids, "length": [len(i) for i in ids],
            "target": [0 if label == "human" else 1 for label in batch["label"]]}


def collate(rows):
    # Lengths round up to a multiple of 256: the fla kernels re-tune for every new length, 22 s a step unpadded
    n = -(-max(len(r["input_ids"]) for r in rows) // 256) * 256
    ids = torch.full((len(rows), n), PAD)
    mask = torch.zeros((len(rows), n), dtype=torch.long)
    for k, r in enumerate(rows):
        m = len(r["input_ids"])
        ids[k, n - m:] = torch.tensor(r["input_ids"])
        mask[k, n - m:] = 1
    return {"input_ids": ids, "attention_mask": mask, "target": torch.tensor([r["target"] for r in rows])}


class AnswerTrainer(Trainer):
    def _get_train_sampler(self, train_dataset=None):
        # The stock group_by_length sampler, given the lengths as a list: the stock code reads the lazy datasets
        # Column one row at a time, which spent ~35 minutes on 842k rows before step 1
        lengths = (train_dataset or self.train_dataset).select_columns(["length"]).to_dict()["length"]
        return LengthGroupedSampler(self.args.train_batch_size * self.args.gradient_accumulation_steps, lengths=lengths)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        target = inputs.pop("target")
        logits = model(**inputs, logits_to_keep=1).logits[:, -1][:, [HUMAN, AI]].float()
        loss = F.cross_entropy(logits, target)
        return (loss, logits) if return_outputs else loss


class Pace(TrainerCallback):
    """Prints steps per second and the projected time left, once, about 5 minutes in (or at the last step)."""
    t0 = s0 = None
    told = False

    def on_step_end(self, a, state, control, **kw):
        if self.t0 is None:  # skip step 1: warm-up and the longest batch
            self.t0, self.s0 = time.time(), state.global_step
            return
        elapsed = time.time() - self.t0
        if not self.told and (elapsed > 300 or state.global_step == state.max_steps):
            rate = (state.global_step - self.s0) / elapsed
            if state.is_world_process_zero:
                print(f"pace: {rate:.2f} steps/s, about {(state.max_steps - state.global_step) / rate / 60:.0f} min left "
                      f"of {state.max_steps} steps", flush=True)
            self.told = True


targs = TrainingArguments(
    output_dir=str(args.out), seed=0, report_to="none",
    per_device_train_batch_size=args.batch, per_device_eval_batch_size=args.batch,
    learning_rate=args.lr, lr_scheduler_type="cosine", warmup_steps=0.01, num_train_epochs=1, max_steps=args.max_steps,
    bf16=True, logging_steps=20, eval_strategy="steps" if full else "no", eval_steps=500, prediction_loss_only=True,
    save_strategy="steps" if full else "no", save_steps=2000, save_total_limit=1,
    train_sampling_strategy="group_by_length", length_column_name="length", remove_unused_columns=False,
    label_names=["target"],
    dataloader_num_workers=4, ddp_find_unused_parameters=False)

with targs.main_process_first(desc="tokenize"):  # rank 0 tokenizes; the others load its cache
    data = load_dataset("parquet", data_files={s: str(args.data / f"{s}.parquet") for s in ("train", "validation")})
    data = data.map(encode, batched=True, num_proc=max(1, (os.cpu_count() or 2) // 2),
                    remove_columns=["id", "text", "label", "kind", "format"])
stamp("tokenized")

model = AutoModelForCausalLM.from_pretrained(repo, revision=rev, dtype=torch.float32)
model.config.use_cache = False
stamp("model loaded")
trainer = AnswerTrainer(model=model, args=targs, train_dataset=data["train"], eval_dataset=data["validation"],
                        data_collator=collate, callbacks=[Pace()])
resume = full and any(args.out.glob("checkpoint-*"))
t0 = time.time()
stamp("training")
result = trainer.train(resume_from_checkpoint=True if resume else None)
metrics = {**result.metrics, **(trainer.evaluate() if full else {})}
if full:
    trainer.save_model(str(args.out / "final"))
if trainer.is_world_process_zero():
    if full:
        tok.save_pretrained(args.out / "final")
    write_json(args.out / "train.json", {
        "model": [repo, rev], "answer_ids": {"human": HUMAN, "ai": AI}, "args": vars(args), "resumed": resume,
        "world_size": targs.world_size, "train_docs": len(data["train"]), "seconds": round(time.time() - t0),
        "peak_gpu_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None,
        "metrics": metrics, "licence": LICENCE_NOTE})
    print(metrics, flush=True)
