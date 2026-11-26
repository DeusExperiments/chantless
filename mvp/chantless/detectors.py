"""Behavioral failure detectors for agent episodes (doom loops, malformed tool calls, premature stops, token corruption).

Every detector is part of episode success; each is also reported separately per side.
"""
from __future__ import annotations

import ast
import json
import re

DEFAULTS = {
    "repeat_call_run": 3,      # same normalised tool call N times in a row
    "span_words": 16,          # repeated n-gram span (whitespace tokens)
    "span_repeats": 4,         # ... repeated at least this many times
}

CODE_BLOCK = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.S)
_THINK_PATTERNS = (
    re.compile(r"<think>.*?(?:</think>|$)", re.DOTALL),
    re.compile(r"<\|channel>thought.*?(?:<channel\|>|$)", re.DOTALL),
    re.compile(r"(?:<\|message_model\|>[^<\n]*)?<\|content_thinking\|>.*?(?:<\|end_message\|>|$)", re.DOTALL),
)
_UNPARSED_TOOL_MARKERS = (
    "<tool_call>",
    "</tool_call>",
    "<|tool_call>",
    "<tool_call|>",
    "<function=",
    "<|content_invoke_tool_json|>",
)


def strip_thinking(text: str | None) -> str:
    if not isinstance(text, str) or not text:
        return ""
    cleaned = text
    for pat in _THINK_PATTERNS:
        cleaned = pat.sub("", cleaned)
    return cleaned


def extract_code(text: str | None) -> str | None:
    if not isinstance(text, str) or not text:
        return None
    cleaned = strip_thinking(text)
    blocks = CODE_BLOCK.findall(cleaned)
    return blocks[-1] if blocks else None


def _assistant_texts(messages: list[dict]) -> list[str]:
    return [m["content"] for m in messages if m.get("role") == "assistant" and isinstance(m.get("content"), str)]


def _tool_calls(messages: list[dict]) -> list[dict]:
    out = []
    for m in messages:
        if m.get("role") == "assistant":
            tcs = m.get("tool_calls")
            if isinstance(tcs, list):
                out.extend(tcs)
    return out


def doom_loop(messages: list[dict], hit_turn_cap: bool, cfg: dict = DEFAULTS) -> bool:
    calls = [json.dumps(c.get("function", c) if isinstance(c, dict) else c, sort_keys=True, default=str) for c in _tool_calls(messages)]
    run = 1
    for a, b in zip(calls, calls[1:]):
        run = run + 1 if a == b else 1
        if run >= cfg["repeat_call_run"]:
            return True
    n = cfg["span_words"]
    for text in _assistant_texts(messages):
        words = text.split()
        if len(words) < n * cfg["span_repeats"]:
            continue
        counts: dict[tuple, int] = {}
        for i in range(len(words) - n + 1):
            k = tuple(words[i : i + n])
            counts[k] = counts.get(k, 0) + 1
            if counts[k] >= cfg["span_repeats"]:
                return True
    return hit_turn_cap


def malformed_tool_call(messages: list[dict], tool_names: set[str]) -> bool:
    for m in messages:
        if m.get("role") != "assistant":
            continue
        raw_content = m.get("content")
        if raw_content is not None and not isinstance(raw_content, str):
            return True
        content = raw_content or ""
        if any(tag in content for tag in _UNPARSED_TOOL_MARKERS):  # template-level call the server failed to parse
            return True
        tcs = m.get("tool_calls")
        if tcs is not None and not isinstance(tcs, list):
            return True
        for c in tcs or []:
            if not isinstance(c, dict):
                return True
            fn = c.get("function")
            if not isinstance(fn, dict):
                return True
            tname = fn.get("name")
            if not isinstance(tname, str) or tname not in tool_names:
                return True
            args = fn.get("arguments")
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    return True
            if not isinstance(args, dict):
                return True
            if tname == "run_tests" and not isinstance(args.get("code"), str):
                return True
    return False


def premature_stop(final_text: str | None, finish_reasons: list[str]) -> bool:
    if any(r == "length" for r in finish_reasons):
        return True
    code = extract_code(final_text)
    return code is None or not code.strip()


def token_garbage(messages: list[dict], final_text: str | None) -> bool:
    extra = [
        m["reasoning_content"]
        for m in messages
        if m.get("role") == "assistant" and isinstance(m.get("reasoning_content"), str)
    ]
    tail = [final_text] if isinstance(final_text, str) else []
    for text in _assistant_texts(messages) + extra + tail:
        if "\ufffd" in text:
            return True
    code = extract_code(final_text)
    if code:
        try:
            ast.parse(code)
        except SyntaxError:
            return True
    return False


def run_all(messages, final_text, finish_reasons, hit_turn_cap, tool_names) -> dict[str, bool]:
    return {
        "doom_loop": doom_loop(messages, hit_turn_cap),
        "malformed_tool_call": malformed_tool_call(messages, tool_names),
        "premature_stop": premature_stop(final_text, finish_reasons),
        "token_garbage": token_garbage(messages, final_text),
    }
