"""Public single-turn BFCL AST tool-calling environment across Qwen3, Qwen3.5, and Gemma 4.

Downloads and caches pinned Berkeley Function Calling Leaderboard v3 single-call tasks from
`ShishirPatil/gorilla@70b6a4a2144597b1f99d1f4d3185d35d7ee532a4` (Apache-2.0; zero private data),
renders each model family's native tool template (`Qwen3` JSON, `Qwen3.5` XML `<function=...>`,
`Gemma 4` `<|tool_call>call:...<tool_call|>`), executes unconstrained `/v1/completions` (no GBNF
grammar rescue), and verifies canonicalized AST tool invocations.
"""
from __future__ import annotations

import ast
import hashlib
import itertools
import json
import re
import time
import urllib.request
from pathlib import Path
from typing import Any

from ..client import InfraError
from .protocol import Episode, Task, Verdict

GORILLA_COMMIT = "70b6a4a2144597b1f99d1f4d3185d35d7ee532a4"
BFCL_RAW_BASE = f"https://raw.githubusercontent.com/ShishirPatil/gorilla/{GORILLA_COMMIT}/berkeley-function-call-leaderboard/data"
DEFAULT_CATEGORIES = (
    "simple",
    "multiple",
    "live_simple",
    "live_multiple",
    "java",
    "javascript",
    "sql",
)

JSON_TOOL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
XML_FUNC_RE = re.compile(r"<function=([^>\s]+)>\s*(.*?)\s*</function>", re.DOTALL)
XML_PARAM_RE = re.compile(r"<parameter=([^>\s]+)>\n?(.*?)\n?</parameter>", re.DOTALL)
GEMMA4_CALL_RE = re.compile(r"<\|tool_call>call:([A-Za-z0-9_.-]+)\{(.*?)\}<tool_call\|>", re.DOTALL)


def _read_records(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    s = text.lstrip()
    if not s:
        return []
    if s[0] == "[":
        return json.loads(text)
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def _parse_maybe(val: Any) -> Any:
    if not isinstance(val, str):
        return val
    s = val.strip()
    if not s:
        return s
    for fn in (json.loads, ast.literal_eval):
        try:
            return fn(s)
        except Exception:
            pass
    return val


def canonical(val: Any) -> Any:
    val = _parse_maybe(val)
    if isinstance(val, dict):
        return {str(k): canonical(v) for k, v in sorted(val.items())}
    if isinstance(val, (list, tuple)):
        return [canonical(v) for v in val]
    if isinstance(val, set):
        return sorted(canonical(v) for v in val)
    return val


def _norm_schema(val: Any) -> Any:
    if isinstance(val, dict):
        out = {str(k): _norm_schema(v) for k, v in val.items()}
        if out.get("type") in ("dict", "HashMap", "Map"):
            out["type"] = "object"
        elif out.get("type") in ("tuple", "ArrayList", "array"):
            out["type"] = "array"
        elif out.get("type") in ("float", "double"):
            out["type"] = "number"
        elif out.get("type") in ("int", "long", "short", "byte"):
            out["type"] = "integer"
        elif out.get("type") in ("bool", "Boolean"):
            out["type"] = "boolean"
        elif out.get("type") in ("str", "String", "char", "any"):
            out["type"] = "string"
        return out
    if isinstance(val, list):
        return [_norm_schema(v) for v in val]
    return val


def _norm_messages(row: dict[str, Any]) -> list[dict[str, str]]:
    val = _parse_maybe(row.get("messages") or row.get("question") or row.get("prompt"))
    if isinstance(val, list):
        if len(val) == 1 and isinstance(val[0], list):
            val = val[0]
        out = []
        for item in val:
            if isinstance(item, dict):
                out.append({"role": str(item.get("role", "user")), "content": str(item.get("content", ""))})
            else:
                out.append({"role": "user", "content": str(item)})
        return out
    return [{"role": "user", "content": str(val)}]


def _norm_tools(row: dict[str, Any]) -> list[dict[str, Any]]:
    val = _parse_maybe(row.get("tools") or row.get("function") or row.get("functions"))
    if val is None:
        return []
    if isinstance(val, dict):
        val = [val]
    out = []
    for t in val:
        t = _parse_maybe(t)
        if not isinstance(t, dict):
            continue
        if t.get("type") == "function" and isinstance(t.get("function"), dict):
            out.append(_norm_schema(t))
        else:
            out.append({"type": "function", "function": _norm_schema(t)})
    return out


def _parse_py_call(expr: str) -> list[dict[str, Any]]:
    try:
        tree = ast.parse(expr.strip(), mode="eval")
    except SyntaxError:
        return []
    if not isinstance(tree.body, ast.Call):
        return []
    parts, fn = [], tree.body.func
    while isinstance(fn, ast.Attribute):
        parts.append(fn.attr)
        fn = fn.value
    if not isinstance(fn, ast.Name):
        return []
    parts.append(fn.id)
    name = ".".join(reversed(parts))
    args = {}
    for kw in tree.body.keywords:
        if kw.arg:
            try:
                args[kw.arg] = canonical(ast.literal_eval(kw.value))
            except Exception:
                return []
    return [{"name": name, "arguments": args}]


def expand_ground_truth(target: Any) -> list[dict[str, Any]]:
    target = _parse_maybe(target)
    if isinstance(target, str):
        return _parse_py_call(target)
    if not isinstance(target, list):
        return [canonical(target)] if isinstance(target, dict) else []
    if target and isinstance(target[0], str):
        out = []
        for s in target:
            out.extend(_parse_py_call(s))
        return out
    calls = []
    for item in target:
        item = _parse_maybe(item)
        if not isinstance(item, dict):
            continue
        if "name" in item and "arguments" in item:
            calls.append(canonical(item))
            continue
        for name, params in item.items():
            params = _parse_maybe(params)
            if not isinstance(params, dict):
                calls.append({"name": name, "arguments": params})
                continue
            keys = list(params)
            vlists = []
            for k in keys:
                vs = _parse_maybe(params[k])
                if not isinstance(vs, list):
                    vs = [vs]
                elif not vs:
                    vs = [[]]
                vlists.append(vs)
            if len(vlists) > 12:
                # Cap combinatorial explosion on huge optional cartesian products
                vlists = [vs[:3] if len(vs) > 3 else vs for vs in vlists]
            for prod in itertools.islice(itertools.product(*vlists), 512):
                calls.append(canonical({"name": name, "arguments": dict(zip(keys, prod))}))
    return calls


def _parse_gemma4_dict(inner: str) -> dict[str, Any]:
    """Convert Gemma 4 `k:<|"|>str<|"|>,n:1` body into a Python dict."""
    s = inner.strip()
    if not s:
        return {}
    # Replace `<|"|>...<|"|>` with valid JSON string literals
    def _repl_str(m: re.Match) -> str:
        return json.dumps(m.group(1))
    s_json = re.sub(r'<\|"\|>(.*?)<\|"\|>', _repl_str, s, flags=re.DOTALL)
    # Quote bare object keys (`{foo:` or `, bar:`) outside JSON strings
    out_chars = []
    in_str, esc = False, False
    i, n = 0, len(s_json)
    s_wrapped = "{" + s_json + "}"
    n = len(s_wrapped)
    while i < n:
        ch = s_wrapped[i]
        if in_str:
            out_chars.append(ch)
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            i += 1
            continue
        if ch == '"':
            in_str = True
            out_chars.append(ch)
            i += 1
            continue
        if ch in "{,":
            out_chars.append(ch)
            i += 1
            while i < n and s_wrapped[i].isspace():
                out_chars.append(s_wrapped[i])
                i += 1
            if i < n and s_wrapped[i] not in ('"', "}", "{", "["):
                m_key = re.match(r"([A-Za-z0-9_.-]+)\s*:", s_wrapped[i:])
                if m_key:
                    out_chars.append(json.dumps(m_key.group(1)) + ":")
                    i += m_key.end()
                    continue
            continue
        out_chars.append(ch)
        i += 1
    return json.loads("".join(out_chars))


def parse_tool_calls(text: str) -> list[dict[str, Any]]:
    """Parse tool calls across `Qwen3.5` XML, `Gemma 4` `<|tool_call>call:...`, and `Qwen3` JSON."""
    if not text:
        return []
    # Strip any `<think>...</think>` or `<|channel>thought...<channel|>` reasoning prefix
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    cleaned = re.sub(r"<\|channel>thought.*?<channel\|>", "", cleaned, flags=re.DOTALL).strip()

    # 1. Gemma 4 `<|tool_call>call:NAME{...}<tool_call|>`
    g4 = GEMMA4_CALL_RE.findall(cleaned)
    if g4:
        calls = []
        for fn_name, body in g4:
            try:
                calls.append(canonical({"name": fn_name.strip(), "arguments": _parse_gemma4_dict(body)}))
            except Exception:
                return []
        return calls

    # 2. Qwen3.5 / Qwen3-Coder XML `<function=NAME><parameter=K>V</parameter></function>`
    xml_blocks = XML_FUNC_RE.findall(cleaned)
    if xml_blocks:
        calls = []
        for fn_name, body in xml_blocks:
            args = {}
            for p_name, p_val in XML_PARAM_RE.findall(body):
                args[p_name.strip()] = canonical(_parse_maybe(p_val.strip()))
            calls.append(canonical({"name": fn_name.strip(), "arguments": args}))
        return calls

    # 3. Qwen3 JSON `<tool_call>{"name": ..., "arguments": ...}</tool_call>` (or bare JSON object)
    blocks = JSON_TOOL_RE.findall(cleaned)
    if not blocks and cleaned.startswith("{") and cleaned.endswith("}"):
        blocks = [cleaned]
    calls = []
    for b in blocks:
        obj = _parse_maybe(b)
        if isinstance(obj, dict) and "name" in obj:
            args = _parse_maybe(obj.get("arguments", {}))
            if isinstance(args, dict):
                calls.append(canonical({"name": str(obj["name"]), "arguments": args}))
    return calls


def format_tool_call(call: dict[str, Any], dialect: str) -> str:
    """Serialize a canonical `{"name": ..., "arguments": {...}}` call in the model family's native syntax."""
    name = str(call["name"])
    args = call.get("arguments") or {}
    if dialect == "qwen35_xml":
        lines = ["<tool_call>", f"<function={name}>"]
        for k, v in args.items():
            v_str = json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else str(v)
            lines.append(f"<parameter={k}>\n{v_str}\n</parameter>")
        lines.extend(["</function>", "</tool_call>"])
        return "\n".join(lines)
    if dialect == "gemma4":
        def _g4_val(v: Any) -> str:
            if isinstance(v, str):
                return f'<|"|>{v}<|"|>'
            if isinstance(v, bool):
                return "true" if v else "false"
            if v is None:
                return "null"
            if isinstance(v, (int, float)):
                return str(v)
            if isinstance(v, list):
                return "[" + ",".join(_g4_val(x) for x in v) + "]"
            if isinstance(v, dict):
                return "{" + ",".join(f"{kk}:{_g4_val(vv)}" for kk, vv in v.items()) + "}"
            return f'<|"|>{v}<|"|>'
        inner = ",".join(f"{k}:{_g4_val(v)}" for k, v in args.items())
        return f"<|tool_call>call:{name}{{{inner}}}<tool_call|>"
    return f"<tool_call>\n{json.dumps({'name': name, 'arguments': args}, ensure_ascii=False)}\n</tool_call>"


def _norm_against_target(val: Any, tgt: Any) -> Any:
    val, tgt = _parse_maybe(val), _parse_maybe(tgt)
    if isinstance(val, dict) and isinstance(tgt, dict):
        return {str(k): _norm_against_target(val[k], tgt.get(k)) for k in sorted(val)}
    if isinstance(val, list) and isinstance(tgt, list) and tgt:
        return [_norm_against_target(v, tgt[i] if i < len(tgt) else tgt[0]) for i, v in enumerate(val)]
    if not isinstance(val, list) and isinstance(tgt, list) and tgt:
        return [_norm_against_target(val, tgt[0])]
    if isinstance(val, str):
        s = val.strip()
        if len(s) >= 2 and s[0] == s[-1] and s[0] in ("'", '"'):
            s = s[1:-1]
        if not isinstance(tgt, bool) and isinstance(tgt, int) and re.fullmatch(r"[-+]?\d+", s):
            return int(s)
        if not isinstance(tgt, bool) and isinstance(tgt, float):
            try:
                return float(s)
            except ValueError:
                pass
        if isinstance(tgt, str) and "**" in tgt and "^" in s:
            s = re.sub(r"(?<=\w)\s*\^\s*(?=[\w(+-])", "**", s)
        return s
    if not isinstance(val, bool) and isinstance(val, int) and isinstance(tgt, float):
        return float(val)
    if not isinstance(val, bool) and isinstance(val, float) and isinstance(tgt, int) and val.is_integer():
        return int(val)
    return val


def calls_match(pred_calls: list[dict[str, Any]], ref_calls: list[dict[str, Any]]) -> bool:
    if len(pred_calls) != 1 or not ref_calls:
        return False
    p = canonical(pred_calls[0])
    for ref in ref_calls:
        r = canonical(ref)
        if p == r:
            return True
        # Also allow optional fields omitted in prediction when reference marks them "" or default
        np_ = canonical(_norm_against_target(p, r))
        nr_ = canonical(_norm_against_target(r, r))
        if np_ == nr_:
            return True
    return False


def detect_dialect(tok) -> tuple[str, list[str]]:
    vocab = tok.get_vocab()
    ct = str(getattr(tok, "chat_template", "") or "")
    if "<|tool_call>" in vocab or "<turn|>" in vocab:
        return "gemma4", ["<turn|>", "<end_of_turn>", "<|tool_response>"]
    if "<function=" in ct:
        return "qwen35_xml", ["<|im_end|>"]
    return "qwen3_json", ["<|im_end|>"]


def _download_cached(cache_dir: Path, categories: tuple[str, ...]) -> list[dict[str, Any]]:
    cache_file = cache_dir / f"bfcl_v3_single_{GORILLA_COMMIT[:8]}_{'-'.join(categories)}.jsonl"
    if cache_file.exists():
        return _read_records(cache_file)
    cache_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for cat in categories:
        fn = f"BFCL_v3_{cat}.json"
        q_p = cache_dir / "raw" / fn
        a_p = cache_dir / "raw" / "possible_answer" / fn
        q_p.parent.mkdir(parents=True, exist_ok=True)
        a_p.parent.mkdir(parents=True, exist_ok=True)
        if not q_p.exists():
            with urllib.request.urlopen(f"{BFCL_RAW_BASE}/{fn}", timeout=60) as r:
                q_p.write_bytes(r.read())
        if not a_p.exists():
            with urllib.request.urlopen(f"{BFCL_RAW_BASE}/possible_answer/{fn}", timeout=60) as r:
                a_p.write_bytes(r.read())
        qs = _read_records(q_p)
        ans_map = {}
        for ar in _read_records(a_p):
            kid = str(ar.get("id") or ar.get("question_id"))
            ans_map[kid] = ar.get("ground_truth") or ar.get("possible_answer") or ar.get("answer")
        for qr in qs:
            q_val = qr.get("question")
            if isinstance(q_val, list) and len(q_val) > 1 and all(isinstance(x, list) for x in q_val):
                continue
            kid = str(qr.get("id") or qr.get("question_id"))
            tgt = ans_map.get(kid)
            refs = expand_ground_truth(tgt)
            if not refs:
                continue
            # Require single-invocation ground truth
            t_parsed = _parse_maybe(tgt)
            if isinstance(t_parsed, list) and sum(1 for x in t_parsed if isinstance(_parse_maybe(x), (dict, str))) != 1:
                continue
            tools = _norm_tools(qr)
            if not tools:
                continue
            rows.append({
                "id": f"{cat}:{kid}",
                "category": cat,
                "messages": _norm_messages(qr),
                "tools": tools,
                "reference_calls": refs[:64],
            })
    with cache_file.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return rows


def _assign_family_and_split(row: dict[str, Any], n_buckets_per_cat: int = 16) -> tuple[str, str]:
    cat = str(row.get("category", "bfcl"))
    fn0 = str((row.get("reference_calls") or [{}])[0].get("name", "fn")).split(".")[0]
    b = int(hashlib.sha256(f"{cat}:{fn0}".encode()).hexdigest()[:8], 16) % n_buckets_per_cat
    fam = f"{cat}/b{b:02d}"
    h = int(hashlib.sha256(f"split:20261118:{fam}".encode()).hexdigest()[:8], 16) % 20
    if h < 6:
        sp = "search"
    elif h < 12:
        sp = "select"
    elif h < 18:
        sp = "exam"
    else:
        sp = "train"
    return fam, sp


class BfclEnvironment:
    name = "bfcl"
    version = "1"

    def __init__(
        self,
        hf_dir: str,
        cache_dir: str = "/tmp/chantless-bfcl-cache",
        pairs_file: str | None = None,
        categories: list[str] | tuple[str, ...] = DEFAULT_CATEGORIES,
        max_tokens: int = 256,
        context: int = 4096,
        concurrency: int = 1,
        kv_type: str = "f16",
        max_total_memory_bytes: int | None = None,
        n_buckets_per_cat: int = 16,
    ):
        from transformers import AutoTokenizer
        self.hf_dir = str(hf_dir)
        self.tok = AutoTokenizer.from_pretrained(self.hf_dir)
        self.dialect, self.stop = detect_dialect(self.tok)
        self.max_tokens = int(max_tokens)
        self.budget = {"max_turns": 1, "max_tokens_per_turn": self.max_tokens}
        self.deployment = {
            "context": int(context),
            "concurrency": int(concurrency),
            "kv_type": str(kv_type),
            "max_total_memory_bytes": max_total_memory_bytes,
        }
        rows = _read_records(Path(pairs_file)) if pairs_file else _download_cached(Path(cache_dir), tuple(categories))
        self._tasks: list[Task] = []
        for r in rows:
            fam, sp = (r["family"], r["split"]) if ("family" in r and "split" in r) else _assign_family_and_split(r, n_buckets_per_cat)
            self._tasks.append(Task(
                id=str(r["id"]),
                family=fam,
                split=sp,
                prompt={"messages": r["messages"], "tools": r["tools"]},
                meta={"category": r.get("category", "bfcl"), "reference_calls": r["reference_calls"]},
            ))

    def render_prompt(self, task: Task) -> str:
        msgs = task.prompt["messages"]
        tools = task.prompt["tools"]
        try:
            return self.tok.apply_chat_template(msgs, tools=tools, add_generation_prompt=True, tokenize=False, enable_thinking=False)
        except TypeError:
            return self.tok.apply_chat_template(msgs, tools=tools, add_generation_prompt=True, tokenize=False)

    def format_reference(self, task: Task) -> str:
        return format_tool_call(task.meta["reference_calls"][0], self.dialect)

    def tasks(self) -> list[Task]:
        return list(self._tasks)

    def stats_texts(self, n: int = 200) -> list[str]:
        search_ts = [t for t in self._tasks if t.split == "search"][:n]
        return [self.render_prompt(t) + self.format_reference(t) for t in search_ts]

    def run(self, task: Task, client: Any, seed: int) -> Episode:
        t0 = time.time()
        prompt_str = self.render_prompt(task)
        srv_prompt = prompt_str[len(self.tok.bos_token):] if (self.tok.bos_token and prompt_str.startswith(self.tok.bos_token)) else prompt_str
        try:
            res = client.complete(srv_prompt, seed=seed, max_tokens=self.max_tokens, stop=self.stop)
        except InfraError as e:
            return Episode(task.id, seed, list(task.prompt["messages"]), task.prompt["tools"], None, [], 0, 0, time.time() - t0, str(e))
        raw_txt = (res["text"] or "").strip()
        clean_txt = re.sub(r"<\|channel>thought.*?<channel\|>", "", re.sub(r"<think>.*?</think>", "", raw_txt, flags=re.DOTALL), flags=re.DOTALL).strip()
        msgs = list(task.prompt["messages"]) + [{"role": "assistant", "content": clean_txt or raw_txt}]
        return Episode(
            task_id=task.id,
            seed=seed,
            messages=msgs,
            tools=task.prompt["tools"],
            final_text=raw_txt,
            finish_reasons=[res.get("finish_reason") or "stop"],
            n_turns=1,
            completion_tokens=int(res.get("completion_tokens", 0)),
            wall_s=time.time() - t0,
        )

    def verify(self, task: Task, episode: Episode) -> Verdict:
        if episode.infra_error:
            return Verdict(task.id, episode.seed, False, {}, infra_error=episode.infra_error)
        txt = episode.final_text or ""
        pred_calls = parse_tool_calls(txt)
        allowed_names = {t["function"]["name"] for t in (task.prompt.get("tools") or []) if "function" in t}
        words = txt.split()
        span_rep = False
        if len(words) >= 64:
            seen: dict[tuple[str, ...], int] = {}
            for i in range(len(words) - 15):
                k = tuple(words[i : i + 16])
                seen[k] = seen.get(k, 0) + 1
                if seen[k] >= 4:
                    span_rep = True
                    break
        malformed = len(pred_calls) == 0 or any(c.get("name") not in allowed_names for c in pred_calls)
        premature = any(r == "length" for r in episode.finish_reasons) or not txt.strip()
        garbage = "\ufffd" in txt
        ok = (not malformed) and calls_match(pred_calls, task.meta["reference_calls"])
        dets = {
            "doom_loop": span_rep,
            "malformed_tool_call": malformed,
            "premature_stop": premature,
            "token_garbage": garbage,
        }
        return Verdict(task.id, episode.seed, ok, dets, details={"pred_calls": pred_calls})


def make_environment(**kwargs) -> BfclEnvironment:
    return BfclEnvironment(**kwargs)
