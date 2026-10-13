"""Optional weight adaptation (mask-plus-weights mode, decision S3 / Q31 option B).

LoRA repair on the training split under the candidate's exact removal structure (dropped sub-blocks
are identity during training, matching the built runtime), merged into full weights, converted to
GGUF and then materialised by the SAME builder with the same mask and re-measured in the loop.
Quantisation is not simulated during training; the loop judges the real built artifact. The KL
reference store stays the frozen source.
"""
from __future__ import annotations

import json
import math
import random
import shutil
import subprocess
import sys
from pathlib import Path


class _Zero:
    pass


def _zero_modules(model, mask: dict):
    import torch

    class ZeroAttn(torch.nn.Module):
        def forward(self, hidden_states, *a, **k):
            return torch.zeros_like(hidden_states), None

    class ZeroMLP(torch.nn.Module):
        def forward(self, x):
            return torch.zeros_like(x)

    saved = {}
    for g, e in mask["groups"].items():
        if e["tier"] != "drop":
            continue
        i = int(g.split(".")[1])
        layer = model.model.layers[i]
        if g.endswith(".attn"):
            saved[g] = layer.self_attn
            layer.self_attn = ZeroAttn()
        else:
            saved[g] = layer.mlp
            layer.mlp = ZeroMLP()
    return saved


def _restore(model, saved):
    for g, mod in saved.items():
        layer = model.model.layers[int(g.split(".")[1])]
        if g.endswith(".attn"):
            layer.self_attn = mod
        else:
            layer.mlp = mod


def repair(hf_dir: str, weights_hf_dir: str | None, mask: dict, recs: list[tuple[list[int], list[int]]], mode: str,
           out_dir: str | Path, cfg: dict, log) -> dict:
    """Train LoRA on `recs` (token seqs + target positions), merge, save HF dir. Returns lineage record."""
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM

    if not any(e["tier"] == "drop" for e in mask["groups"].values()) and not cfg.get("repair_without_drops"):
        return None
    dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    dtype = torch.bfloat16 if dev != "cpu" else torch.float32
    src = weights_hf_dir or hf_dir
    student = AutoModelForCausalLM.from_pretrained(src, torch_dtype=dtype).to(dev)
    teacher = None
    if mode == "preserve":
        teacher = AutoModelForCausalLM.from_pretrained(hf_dir, torch_dtype=dtype).to(dev).eval()  # frozen source
    saved = _zero_modules(student, mask)
    kept_layers = sorted({int(g.split(".")[1]) for g, e in mask["groups"].items() if e["tier"] != "drop" and g.startswith("blk.")})
    targets = []
    for i in kept_layers:
        for nm in ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"):
            if (nm.startswith("self_attn") and mask["groups"].get(f"blk.{i}.attn", {}).get("tier") != "drop") or \
               (nm.startswith("mlp") and mask["groups"].get(f"blk.{i}.ffn", {}).get("tier") != "drop"):
                targets.append(f"model.layers.{i}.{nm}")
    lcfg = LoraConfig(r=cfg.get("rank", 16), lora_alpha=cfg.get("alpha", 32), lora_dropout=0.0, target_modules=targets)
    model = get_peft_model(student, lcfg)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg.get("lr", 1e-4))
    rng = random.Random(cfg.get("seed", 1729))
    steps = cfg.get("steps", 200)
    losses = []
    model.train()
    for step in range(steps):
        toks, tg = recs[rng.randrange(len(recs))]
        ids = torch.tensor([toks], device=dev)
        pos = torch.tensor([t - 1 for t in tg], device=dev)
        logits = model(input_ids=ids).logits[0, pos].float()
        if mode == "preserve":
            with torch.no_grad():
                tl = teacher(input_ids=ids).logits[0, pos].float()
            loss = torch.nn.functional.kl_div(torch.log_softmax(logits, -1), torch.log_softmax(tl, -1), log_target=True, reduction="batchmean")
        else:
            y = torch.tensor([toks[t] for t in tg], device=dev)
            loss = torch.nn.functional.cross_entropy(logits, y)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(float(loss))
        if step % 25 == 0:
            log(f"repair step {step}/{steps} loss {sum(losses[-25:]) / len(losses[-25:]):.5f}")
    merged = model.merge_and_unload()
    _restore(merged, saved)
    if getattr(merged.config, "tie_word_embeddings", False):
        # keep the source tensor layout (an explicit output.weight), so the repaired GGUF has the same groups
        merged.config.tie_word_embeddings = False
        merged.lm_head.weight = torch.nn.Parameter(merged.model.embed_tokens.weight.detach().clone())
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    merged.to(torch.bfloat16).save_pretrained(out_dir, safe_serialization=True)
    for f in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "generation_config.json"):
        p = Path(hf_dir) / f
        if p.exists():
            shutil.copy(p, out_dir / f)
    rec = {"method": "lora-merge", "mode": mode, "steps": steps, "rank": lcfg.r, "lr": cfg.get("lr", 1e-4),
           "loss_first": sum(losses[:10]) / max(1, len(losses[:10])), "loss_last": sum(losses[-10:]) / max(1, len(losses[-10:])),
           "n_train_seqs": len(recs), "base": src, "out": str(out_dir), "device": dev}
    (out_dir / "chantless-lineage.json").write_text(json.dumps(rec, indent=1))
    return rec


def to_gguf(convert_py: str, hf_dir: str | Path, out: str | Path, log_path: str) -> None:
    with open(log_path, "a") as f:
        subprocess.run([sys.executable, convert_py, str(hf_dir), "--outtype", "bf16", "--outfile", str(out)],
                       stdout=f, stderr=subprocess.STDOUT, check=True)
