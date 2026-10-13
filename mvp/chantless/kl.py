"""KL proxy: render episodes to token sequences with assistant-token targets, then score with traj-kl.

traj-kl (tools/traj-kl/traj-kl.cpp) runs one forward pass per sequence and stores or compares
full-vocabulary log-probs only at target positions. The reference store is computed once from
the frozen source and reused for every candidate (reference-logit reuse, pilot-measured).
"""
from __future__ import annotations

import json
import random
import struct
from pathlib import Path

from .runtime import Toolchain, run


class RenderError(ValueError):
    pass


class Renderer:
    """Qwen3 chat-template rendering through the HF tokenizer, non-thinking mode."""

    def __init__(self, hf_dir: str):
        from transformers import AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(hf_dir)
        self.im_end = self.tok.convert_tokens_to_ids("<|im_end|>")

    def _render(self, msgs, tools, gen: bool) -> str:
        return self.tok.apply_chat_template(msgs, tools=tools or None, add_generation_prompt=gen, tokenize=False,
                                            enable_thinking=False)

    def ids(self, text: str) -> list[int]:
        return self.tok(text, add_special_tokens=False)["input_ids"]

    def episode(self, messages: list[dict], tools: list[dict] | None) -> list[tuple[list[int], list[int]]]:
        """One scored sequence per assistant turn, rendered exactly as the model saw it at generation time.

        Qwen3's template drops the empty think block from earlier assistant turns when re-rendering history,
        so a single whole-conversation render would not match what was generated. Each turn i is therefore
        rendered as messages[:i+1] (turn i last), whose prefix must equal the generation prompt for messages[:i].
        Targets run from the end of that prompt through the turn's <|im_end|>.
        """
        msgs = _normalise(messages)
        recs = []
        for i, m in enumerate(msgs):
            if m["role"] != "assistant":
                continue
            seq = self.ids(self._render(msgs[: i + 1], tools, False))
            pre = self.ids(self._render(msgs[:i], tools, True))
            if seq[: len(pre)] != pre:
                raise RenderError(f"prefix mismatch at message {i}")
            try:
                end = seq.index(self.im_end, len(pre))
            except ValueError:
                raise RenderError(f"no <|im_end|> after assistant message {i}")
            recs.append((seq[: end + 1], list(range(len(pre), end + 1))))
        if not recs:
            raise RenderError("no assistant targets")
        return recs


def _normalise(messages):
    out = []
    for m in messages:
        m = dict(m)
        if m.get("tool_calls"):
            tcs = []
            for c in m["tool_calls"]:
                c = dict(c)
                fn = dict(c.get("function") or {})
                if isinstance(fn.get("arguments"), str):
                    try:
                        fn["arguments"] = json.loads(fn["arguments"])
                    except json.JSONDecodeError:
                        pass
                c["function"] = fn
                tcs.append(c)
            m["tool_calls"] = tcs
        out.append(m)
    return out


def write_trj(path: str | Path, recs: list[tuple[list[int], list[int]]]) -> dict:
    with open(path, "wb") as f:
        f.write(b"TRJ1" + struct.pack("<I", len(recs)))
        for toks, tg in recs:
            f.write(struct.pack("<II", len(toks), len(tg)))
            f.write(struct.pack(f"<{len(toks)}i", *toks))
            f.write(struct.pack(f"<{len(tg)}I", *tg))
    return {"n_seq": len(recs), "n_tokens": sum(len(t) for t, _ in recs), "n_targets": sum(len(g) for _, g in recs),
            "max_len": max((len(t) for t, _ in recs), default=0)}


def subsample(recs, max_targets: int, seed: int):
    allt = [(ri, t) for ri, (_, ts) in enumerate(recs) for t in ts]
    if not max_targets or len(allt) <= max_targets:
        return recs
    keep = set(random.Random(seed).sample(range(len(allt)), max_targets))
    per = [[] for _ in recs]
    for k, (ri, t) in enumerate(allt):
        if k in keep:
            per[ri].append(t)
    return [(toks, sorted(p)) for (toks, _), p in zip(recs, per) if p]


def pack_text(renderer: Renderer, texts: list[str], seq_len: int, n_seq: int) -> list[tuple[list[int], list[int]]]:
    """Pack general text into fixed sequences; score the second half (positions seq_len/2 .. seq_len-1)."""
    stream: list[int] = []
    eos = renderer.tok.eos_token_id or renderer.im_end
    for t in texts:
        stream.extend(renderer.ids(t) + [eos])
        if len(stream) >= seq_len * n_seq:
            break
    recs = []
    for i in range(min(n_seq, len(stream) // seq_len)):
        toks = stream[i * seq_len : (i + 1) * seq_len]
        recs.append((toks, list(range(seq_len // 2, seq_len))))
    return recs


def traj_kl(tc: Toolchain, mode: str, model: str, trj: str, store: str, out: str, log: str | None, gpu: str = "",
            ubatch: int = 2048, threads: int = 8, kv: str = "f16") -> dict:
    Path(store).mkdir(parents=True, exist_ok=True)
    env = dict(tc.env_for(gpu), OMP_NUM_THREADS=str(threads))   # bound the row-parallel post-processing pool
    run([tc.traj_kl, "--mode", mode, "-m", model, "--traj", trj, "--store", store, "--out", out, "--ubatch", str(ubatch),
         "--threads", str(threads), "-ctk", kv, "-ctv", kv], log, env=env)
    return json.loads(Path(out).read_text())
