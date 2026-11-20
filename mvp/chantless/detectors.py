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


def extract_code(text: str | None) -> str | None:
    if not text:
        return None
    blocks = CODE_BLOCK.findall(text)
    return blocks[-1] if blocks else None


def _assistant_texts(messages: list[dict]) -> list[str]:
    return [m.get("content") or "" for m in messages if m.get("role") == "assistant"]


def _tool_calls(messages: list[dict]) -> list[dict]:
    out = []
    for m in messages:
        if m.get("role") == "assistant":
            out.extend(m.get("tool_calls") or [])
    return out


def doom_loop(messages: list[dict], hit_turn_cap: bool, cfg: dict = DEFAULTS) -> bool:
    calls = [json.dumps(c.get("function", c), sort_keys=True) for c in _tool_calls(messages)]
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
        content = m.get("content") or ""
        if "<tool_call>" in content:  # template-level call the server failed to parse
            return True
        for c in m.get("tool_calls") or []:
            fn = c.get("function") or {}
            if fn.get("name") not in tool_names:
                return True
            args = fn.get("arguments")
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    return True
            if not isinstance(args, dict) or not isinstance(args.get("code"), str):
                return True
    return False


def premature_stop(final_text: str | None, finish_reasons: list[str]) -> bool:
    if any(r == "length" for r in finish_reasons):
        return True
    code = extract_code(final_text)
    return code is None or not code.strip()


def token_garbage(messages: list[dict], final_text: str | None) -> bool:
    for text in _assistant_texts(messages) + [final_text or ""]:
        if "�" in text:
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
