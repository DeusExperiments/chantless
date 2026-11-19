"""OpenAI-compatible chat client for llama-server, with per-request seeds and infra-error typing."""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request


class InfraError(RuntimeError):
    """Server/transport failure. Retried; never counted as a task failure (decision S12)."""


class ChatClient:
    def __init__(self, base_url: str, decoding: dict, timeout: float = 600.0, retries: int = 3):
        self.base_url = base_url.rstrip("/")
        self.decoding = dict(decoding)
        self.timeout = timeout
        self.retries = retries

    def chat(self, messages: list[dict], seed: int, max_tokens: int, tools: list[dict] | None = None) -> dict:
        body = {
            "messages": messages,
            "seed": int(seed),
            "max_tokens": int(max_tokens),
            "chat_template_kwargs": {"enable_thinking": False},
            **{k: v for k, v in self.decoding.items() if k != "enable_thinking"},
        }
        if tools:
            body["tools"] = tools
        last = None
        for attempt in range(self.retries):
            try:
                req = urllib.request.Request(
                    self.base_url + "/v1/chat/completions",
                    data=json.dumps(body).encode(),
                    headers={"Content-Type": "application/json"},
                )
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    out = json.loads(r.read())
                choice = out["choices"][0]
                return {
                    "message": choice["message"],
                    "finish_reason": choice.get("finish_reason"),
                    "completion_tokens": (out.get("usage") or {}).get("completion_tokens", 0),
                }
            except (urllib.error.URLError, ConnectionError, TimeoutError, KeyError, json.JSONDecodeError) as e:
                last = e
                time.sleep(1.0 + attempt)
        raise InfraError(f"chat failed after {self.retries} attempts: {last!r}")

    def complete(self, prompt: str, seed: int, max_tokens: int, stop: list[str] | None = None) -> dict:
        """Unconstrained `/v1/completions` call on an already-rendered template prompt (no automatic GBNF tool grammar)."""
        body = {
            "prompt": prompt,
            "seed": int(seed),
            "max_tokens": int(max_tokens),
            **{k: v for k, v in self.decoding.items() if k != "enable_thinking"},
        }
        if stop:
            body["stop"] = stop
        last = None
        for attempt in range(self.retries):
            try:
                req = urllib.request.Request(
                    self.base_url + "/v1/completions",
                    data=json.dumps(body).encode(),
                    headers={"Content-Type": "application/json"},
                )
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    out = json.loads(r.read())
                choice = out["choices"][0]
                return {
                    "text": choice.get("text", ""),
                    "finish_reason": choice.get("finish_reason"),
                    "completion_tokens": (out.get("usage") or {}).get("completion_tokens", 0),
                }
            except (urllib.error.URLError, ConnectionError, TimeoutError, KeyError, json.JSONDecodeError) as e:
                last = e
                time.sleep(1.0 + attempt)
        raise InfraError(f"completion failed after {self.retries} attempts: {last!r}")
