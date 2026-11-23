"""KL proxy: render episodes to token sequences with assistant-token targets, then score with traj-kl.

traj-kl (mvp/tools/traj-kl.cpp) runs one forward pass per sequence and stores or compares
full-vocabulary log-probs only at target positions. The reference store is computed once from
the frozen source and reused across candidates.
"""
from __future__ import annotations

import json
import random
import struct
from pathlib import Path
from typing import Any

from jinja2 import TemplateError

from .env.bfcl import _thinking_enabled
from .runtime import Toolchain, run


class RenderError(ValueError):
    pass


class Renderer:
    """Multi-family chat-template rendering through the HF tokenizer (`Qwen3`, `Qwen3.5`, `Gemma 4`, `Inkling-2`)."""

    def __init__(
        self,
        hf_dir: str,
        reasoning_effort: str | float | None = "none",
        enable_thinking: bool | None = None,
    ):
        from transformers import AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(hf_dir)
        self.reasoning_effort = reasoning_effort
        self.enable_thinking = enable_thinking
        vocab = self.tok.get_vocab()
        ct = str(getattr(self.tok, "chat_template", "") or "")
        if (
            "<|content_invoke_tool_json|>" in vocab
            or "<|message_model|>" in vocab
            or "<|content_invoke_tool_json|>" in ct
            or "<|message_model|>" in ct
        ):
            self.dialect = "inkling2"
        elif "<|tool_call>" in vocab or "<turn|>" in vocab:
            self.dialect = "gemma4"
        elif "<function=" in ct:
            self.dialect = "qwen35_xml"
        else:
            self.dialect = "qwen3_json"
        self.end_ids = [
            int(vocab[t])
            for t in ("<|content_model_end_sampling|>", "<|im_end|>", "<turn|>", "<end_of_turn>")
            if t in vocab
        ]
        if not self.end_ids and self.tok.eos_token_id is not None:
            self.end_ids = [int(self.tok.eos_token_id)]
        self.im_end = self.end_ids[0] if self.end_ids else None

    @staticmethod
    def _is_inkling2_wire(content: object) -> bool:
        return isinstance(content, str) and content.lstrip().startswith(("<|message_model|>", "<|content_"))

    def _render(self, msgs, tools, gen: bool) -> str:
        kw = {"tools": tools or None, "add_generation_prompt": gen, "tokenize": False}
        try:
            if self.dialect == "inkling2":
                render_msgs = []
                replacements: list[tuple[str, str]] = []
                for idx, m in enumerate(msgs):
                    if m.get("role") == "assistant" and not m.get("tool_calls") and self._is_inkling2_wire(m.get("content")):
                        wire = m["content"]
                        if not wire.rstrip().endswith("<|content_model_end_sampling|>"):
                            wire += "<|content_model_end_sampling|>"
                        sentinel = f"__CHANTLESS_INK2_WIRE_{idx}_{id(msgs)}__"
                        placeholder = {"role": "assistant", "content": sentinel}
                        if m.get("reasoning_content"):
                            placeholder["reasoning_content"] = m["reasoning_content"]
                        render_msgs.append(placeholder)
                        replacements.append(
                            (f"<|message_model|><|content_text|>{sentinel}<|end_message|><|content_model_end_sampling|>", wire)
                        )
                    else:
                        render_msgs.append(m)
                eff_et = _thinking_enabled(self.dialect, self.reasoning_effort) if self.enable_thinking is None else bool(self.enable_thinking)
                try:
                    if self.reasoning_effort is not None:
                        out = self.tok.apply_chat_template(render_msgs, reasoning_effort=self.reasoning_effort, **kw)
                    else:
                        out = self.tok.apply_chat_template(render_msgs, **kw)
                except TypeError:
                    try:
                        out = self.tok.apply_chat_template(render_msgs, enable_thinking=eff_et, **kw)
                    except TypeError:
                        out = self.tok.apply_chat_template(render_msgs, **kw)
                for placeholder_str, wire in replacements:
                    if placeholder_str not in out:
                        raise RenderError("failed to substitute Inkling-2 wire assistant turn")
                    out = out.replace(placeholder_str, wire, 1)
                return out
            eff_et = _thinking_enabled(self.dialect, self.reasoning_effort) if self.enable_thinking is None else bool(self.enable_thinking)
            try:
                return self.tok.apply_chat_template(msgs, enable_thinking=eff_et, **kw)
            except TypeError:
                if self.reasoning_effort is not None:
                    try:
                        return self.tok.apply_chat_template(msgs, reasoning_effort=self.reasoning_effort, **kw)
                    except TypeError:
                        pass
                return self.tok.apply_chat_template(msgs, **kw)
        except (TemplateError, TypeError, ValueError) as exc:
            raise RenderError(f"chat template error: {exc}") from exc

    def ids(self, text: str) -> list[int]:
        return self.tok(text, add_special_tokens=False)["input_ids"]

    def episode(self, messages: list[dict], tools: list[dict] | None) -> list[tuple[list[int], list[int]]]:
        """One scored sequence per assistant turn, rendered exactly as the model saw it at generation time.

        Templates like Qwen3/Qwen3.5/Gemma4 strip empty reasoning blocks from earlier assistant turns when re-rendering
        history, so each turn i is rendered as messages[:i+1] (turn i last), whose prefix must equal the generation
        prompt for messages[:i]. Targets run from the end of that prompt through the turn-end delimiter.
        """
        msgs = _normalise(messages)
        recs = []
        for i, m in enumerate(msgs):
            if m["role"] != "assistant":
                continue
            seq_str = self._render(msgs[: i + 1], tools, False)
            pre_str = self._render(msgs[:i], tools, True)
            seq = self.ids(seq_str)
            pre = self.ids(pre_str)
            if seq[: len(pre)] != pre:
                if seq_str.startswith(pre_str):
                    seq = pre + self.ids(seq_str[len(pre_str) :])
                else:
                    raise RenderError(f"prefix mismatch at message {i}")
            end = None
            for eid in self.end_ids:
                try:
                    cand = seq.index(eid, len(pre))
                    if end is None or cand < end:
                        end = cand
                except ValueError:
                    pass
            if end is None:
                raise RenderError(f"no turn-end token after assistant message {i}")
            recs.append((seq[: end + 1], list(range(len(pre), end + 1))))
        if not recs:
            raise RenderError("no assistant targets")
        return recs

    def render_trajectories(
        self,
        trajectories: list[Any],
        tools_fn: Any = None,
        on_error: Any = None,
    ) -> list[tuple[list[int], list[int]]]:
        """Render a batch of episodes/dicts/message-lists; raises RenderError if 0 records render."""
        if not trajectories:
            raise RenderError("no trajectories to render")
        recs: list[tuple[list[int], list[int]]] = []
        last_err: Exception | None = None
        for traj in trajectories:
            if isinstance(traj, dict):
                msgs = traj.get("messages") or []
                t_tools = traj.get("tools")
                task_id = traj.get("task") or traj.get("task_id")
            elif hasattr(traj, "messages"):
                msgs = traj.messages
                t_tools = getattr(traj, "tools", None)
                task_id = getattr(traj, "task_id", None)
            else:
                msgs = traj
                t_tools = None
                task_id = None
            if t_tools is None and callable(tools_fn):
                t_tools = tools_fn(task_id)
            try:
                recs.extend(self.episode(msgs, t_tools))
            except RenderError as e:
                last_err = e
                if callable(on_error):
                    on_error(traj, e)
        if not recs:
            raise RenderError(
                f"all {len(trajectories)} trajectories failed to render"
                + (f" (last: {last_err})" if last_err else "")
            )
        return recs


def _normalise(messages):
    if not isinstance(messages, (list, tuple)):
        raise RenderError(f"messages must be a list or tuple, got {type(messages).__name__}")
    out = []
    for m in messages:
        if not isinstance(m, dict):
            raise RenderError(f"message must be a dict, got {type(m).__name__}")
        m = dict(m)
        content = m.get("content")
        if content is not None and not isinstance(content, (str, list)):
            raise RenderError(f"message content must be None, str, or list, got {type(content).__name__}")
        if isinstance(content, list):
            for part in content:
                if not isinstance(part, dict):
                    raise RenderError(f"message content parts must be dicts, got {type(part).__name__}")
        if m.get("tool_calls"):
            if not isinstance(m["tool_calls"], list):
                raise RenderError(f"tool_calls must be a list, got {type(m['tool_calls']).__name__}")
            tcs = []
            for c in m["tool_calls"]:
                if not isinstance(c, dict):
                    raise RenderError(f"tool_call entry must be a dict, got {type(c).__name__}")
                c = dict(c)
                raw_fn = c.get("function")
                if raw_fn is not None and not isinstance(raw_fn, dict):
                    raise RenderError(f"tool_call function must be a dict, got {type(raw_fn).__name__}")
                fn = dict(raw_fn or {})
                if isinstance(fn.get("arguments"), str):
                    try:
                        parsed = json.loads(fn["arguments"])
                    except json.JSONDecodeError as exc:
                        raise RenderError(f"malformed tool_call arguments JSON: {exc}") from exc
                    if not isinstance(parsed, dict):
                        raise RenderError(f"tool_call arguments must decode to object, got {type(parsed).__name__}")
                    fn["arguments"] = parsed
                elif fn.get("arguments") is not None and not isinstance(fn.get("arguments"), dict):
                    raise RenderError(f"tool_call arguments must be a dict or JSON string, got {type(fn.get('arguments')).__name__}")
                c["function"] = fn
                tcs.append(c)
            m["tool_calls"] = tcs
        out.append(m)
    return out


def write_trj(path: str | Path, recs: list[tuple[list[int], list[int]]]) -> dict:
    if not recs:
        raise RenderError("cannot write empty trajectory store")
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
    eos = renderer.im_end if renderer.tok.eos_token_id is None else renderer.tok.eos_token_id
    if eos is None:
        raise RenderError("no eos_token_id or turn-end token for pack_text")
    eos = int(eos)
    for t in texts:
        stream.extend(renderer.ids(t) + [eos])
        if len(stream) >= seq_len * n_seq:
            break
    recs = []
    for i in range(min(n_seq, len(stream) // seq_len)):
        toks = stream[i * seq_len : (i + 1) * seq_len]
        recs.append((toks, list(range(seq_len // 2, seq_len))))
    if n_seq > 0 and not recs:
        raise RenderError("pack_text produced zero sequences")
    return recs


def traj_kl(tc: Toolchain, mode: str, model: str, trj: str, store: str, out: str, log: str | None, gpu: str = "",
            ubatch: int = 2048, threads: int = 8, kv: str = "f16") -> dict:
    Path(store).mkdir(parents=True, exist_ok=True)
    env = dict(tc.env_for(gpu), OMP_NUM_THREADS=str(threads))   # bound the row-parallel post-processing pool
    run([tc.traj_kl, "--mode", mode, "-m", model, "--traj", trj, "--store", store, "--out", out, "--ubatch", str(ubatch),
         "--threads", str(threads), "-ctk", kv, "-ctv", kv], log, env=env)
    return json.loads(Path(out).read_text())
