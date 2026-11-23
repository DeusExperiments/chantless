"""OpenAI-compatible chat client for llama-server, with per-request seeds and infra-error typing."""
from __future__ import annotations

import http.client
import json
import time
import urllib.error
import urllib.request


class InfraError(RuntimeError):
    """Server or transport failure; retried and excluded from task failure counts."""


_DECODING_SKIP = (
    "enable_thinking",
    "chat_template_kwargs",
    "reasoning_effort",
    "seed",
    "max_tokens",
    "cache_prompt",
)


def _thinking_enabled(dialect: str | None, reasoning_effort: str | float | None) -> bool:
    if reasoning_effort is None:
        return dialect == "inkling2"
    if isinstance(reasoning_effort, bool):
        return reasoning_effort
    if isinstance(reasoning_effort, str):
        s = reasoning_effort.strip().lower()
        if s in ("", "none", "false", "off"):
            return False
        try:
            return float(s) > 0.0
        except ValueError:
            return True
    return float(reasoning_effort) > 0.0


def _default_chat_template_kwargs(dialect: str | None) -> dict:
    if dialect == "inkling2":
        return {"reasoning_effort": "none"}
    return {"enable_thinking": False}


class ChatClient:
    def __init__(
        self,
        base_url: str,
        decoding: dict,
        timeout: float = 600.0,
        retries: int = 3,
        chat_template_kwargs: dict | None = None,
        dialect: str | None = None,
        cache_prompt: bool | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.decoding = dict(decoding)
        self.timeout = timeout
        self.retries = retries
        self.dialect = dialect
        self._explicit_chat_template_kwargs = dict(chat_template_kwargs) if chat_template_kwargs is not None else None
        self.chat_template_kwargs = (
            dict(self._explicit_chat_template_kwargs)
            if self._explicit_chat_template_kwargs is not None
            else _default_chat_template_kwargs(dialect)
        )
        self._explicit_cache_prompt = None if cache_prompt is None else bool(cache_prompt)
        self.cache_prompt: bool = (
            self._explicit_cache_prompt
            if self._explicit_cache_prompt is not None
            else bool(self.decoding.get("cache_prompt", False))
        )

    def chat(
        self,
        messages: list[dict],
        seed: int,
        max_tokens: int,
        tools: list[dict] | None = None,
        chat_template_kwargs: dict | None = None,
        cache_prompt: bool | None = None,
    ) -> dict:
        dec_ctk = self.decoding.get("chat_template_kwargs")
        if isinstance(dec_ctk, dict):
            if self._explicit_chat_template_kwargs is None and self.dialect is None:
                ctk = dict(dec_ctk)
            else:
                ctk = {**self.chat_template_kwargs, **dec_ctk}
        else:
            ctk = dict(self.chat_template_kwargs)
            if self._explicit_chat_template_kwargs is None:
                if self.dialect == "inkling2" and self.decoding.get("reasoning_effort") is not None:
                    ctk["reasoning_effort"] = self.decoding["reasoning_effort"]
                elif self.dialect is None and self.decoding.get("reasoning_effort") is not None:
                    ctk = {"reasoning_effort": self.decoding["reasoning_effort"]}
                elif self.dialect != "inkling2":
                    if self.decoding.get("enable_thinking") is not None:
                        ctk["enable_thinking"] = bool(self.decoding["enable_thinking"])
                    elif self.dialect is not None and self.decoding.get("reasoning_effort") is not None:
                        ctk["enable_thinking"] = _thinking_enabled(self.dialect, self.decoding["reasoning_effort"])
        if chat_template_kwargs is not None:
            ctk = {**ctk, **chat_template_kwargs}
        body = {
            **{k: v for k, v in self.decoding.items() if k not in _DECODING_SKIP},
            "messages": messages,
            "seed": int(seed),
            "max_tokens": int(max_tokens),
            "chat_template_kwargs": ctk,
        }
        eff_cp = self.cache_prompt if cache_prompt is None else bool(cache_prompt)
        if not eff_cp:
            body["cache_prompt"] = False
        if tools:
            body["tools"] = tools
        payload = json.dumps(body).encode()
        last = None
        for attempt in range(self.retries):
            try:
                req = urllib.request.Request(
                    self.base_url + "/v1/chat/completions",
                    data=payload,
                    headers={"Content-Type": "application/json"},
                )
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    out = json.loads(r.read())
                if not isinstance(out, dict):
                    raise TypeError(f"expected JSON object, got {type(out).__name__}")
                choices = out["choices"]
                if not isinstance(choices, list):
                    raise KeyError("choices")
                choice = choices[0]
                if not isinstance(choice, dict):
                    raise TypeError(f"expected choice object, got {type(choice).__name__}")
                msg = choice["message"]
                if not isinstance(msg, dict):
                    raise TypeError(f"expected message object, got {type(msg).__name__}")
                content = msg.get("content")
                if content is not None and not isinstance(content, str):
                    raise TypeError(f"expected message content to be str or None, got {type(content).__name__}")
                rc = msg.get("reasoning_content")
                if rc is not None and not isinstance(rc, str):
                    raise TypeError(f"expected message reasoning_content to be str or None, got {type(rc).__name__}")
                usage = out.get("usage")
                if usage is not None and not isinstance(usage, dict):
                    raise TypeError(f"expected usage object, got {type(usage).__name__}")
                ctok = (usage or {}).get("completion_tokens")
                if ctok is not None and (isinstance(ctok, bool) or not isinstance(ctok, int)):
                    raise TypeError(f"expected completion_tokens to be int or None, got {type(ctok).__name__}")
                return {
                    "message": msg,
                    "finish_reason": choice.get("finish_reason"),
                    "completion_tokens": ctok or 0,
                }
            except (
                urllib.error.URLError,
                http.client.HTTPException,
                ConnectionError,
                TimeoutError,
                KeyError,
                IndexError,
                TypeError,
                AttributeError,
                UnicodeDecodeError,
                json.JSONDecodeError,
            ) as e:
                last = e
                if attempt + 1 < self.retries:
                    time.sleep(1.0 + attempt)
        raise InfraError(f"chat failed after {self.retries} attempts: {last!r}")

    def complete(
        self,
        prompt: str,
        seed: int,
        max_tokens: int,
        stop: list[str] | None = None,
        cache_prompt: bool | None = None,
    ) -> dict:
        """Unconstrained `/v1/completions` call on an already-rendered template prompt (no automatic GBNF tool grammar)."""
        body = {
            **{k: v for k, v in self.decoding.items() if k not in _DECODING_SKIP},
            "prompt": prompt,
            "seed": int(seed),
            "max_tokens": int(max_tokens),
        }
        eff_cp = self.cache_prompt if cache_prompt is None else bool(cache_prompt)
        if not eff_cp:
            body["cache_prompt"] = False
        if stop:
            body["stop"] = stop
        payload = json.dumps(body).encode()
        last = None
        for attempt in range(self.retries):
            try:
                req = urllib.request.Request(
                    self.base_url + "/v1/completions",
                    data=payload,
                    headers={"Content-Type": "application/json"},
                )
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    out = json.loads(r.read())
                if not isinstance(out, dict):
                    raise TypeError(f"expected JSON object, got {type(out).__name__}")
                choices = out["choices"]
                if not isinstance(choices, list):
                    raise KeyError("choices")
                choice = choices[0]
                if not isinstance(choice, dict):
                    raise TypeError(f"expected choice object, got {type(choice).__name__}")
                text = choice.get("text")
                if text is not None and not isinstance(text, str):
                    raise TypeError(f"expected choice text to be str or None, got {type(text).__name__}")
                usage = out.get("usage")
                if usage is not None and not isinstance(usage, dict):
                    raise TypeError(f"expected usage object, got {type(usage).__name__}")
                ctok = (usage or {}).get("completion_tokens")
                if ctok is not None and (isinstance(ctok, bool) or not isinstance(ctok, int)):
                    raise TypeError(f"expected completion_tokens to be int or None, got {type(ctok).__name__}")
                return {
                    "text": text or "",
                    "finish_reason": choice.get("finish_reason"),
                    "completion_tokens": ctok or 0,
                }
            except (
                urllib.error.URLError,
                http.client.HTTPException,
                ConnectionError,
                TimeoutError,
                KeyError,
                IndexError,
                TypeError,
                AttributeError,
                UnicodeDecodeError,
                json.JSONDecodeError,
            ) as e:
                last = e
                if attempt + 1 < self.retries:
                    time.sleep(1.0 + attempt)
        raise InfraError(f"completion failed after {self.retries} attempts: {last!r}")
