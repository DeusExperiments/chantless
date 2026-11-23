"""Single-turn BFCL AST tool-calling environment across Qwen3, Qwen3.5, and Gemma 4.

Downloads and caches pinned Berkeley Function Calling Leaderboard v3 single-call tasks from
`ShishirPatil/gorilla@70b6a4a2144597b1f99d1f4d3185d35d7ee532a4` (Apache-2.0), renders each model
family's native tool template (`Qwen3` JSON, `Qwen3.5` XML `<function=...>`, `Gemma 4`
`<|tool_call>call:...<tool_call|>`), executes unconstrained `/v1/completions`, and verifies
canonicalized AST tool invocations.
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

from ..client import InfraError, _thinking_enabled
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
INKLING2_THINK_RE = re.compile(
    r"(?:<\|message_model\|>[^<\n]*)?<\|content_thinking\|>.*?(?:<\|end_message\|>|$)",
    re.DOTALL,
)
INKLING2_CALL_RE = re.compile(
    r"(?:<\|message_model\|>([^<\s]*))?<\|content_invoke_tool_json\|>\s*(.*?)\s*(?:<\|end_message\|>|<\|content_model_end_sampling\|>|$)",
    re.DOTALL,
)


def strip_thinking(text: str) -> str:
    """Strip Qwen, Gemma 4, and Inkling 2 reasoning blocks (including unclosed trailing blocks) from raw model output."""
    cleaned = re.sub(r"<think>.*?(?:</think>|$)", "", text, flags=re.DOTALL)
    cleaned = re.sub(r"<\|channel>thought.*?(?:<channel\|>|$)", "", cleaned, flags=re.DOTALL)
    return INKLING2_THINK_RE.sub("", cleaned)


def render_chat_prompt(
    tok: Any,
    msgs: list[dict[str, Any]],
    dialect: str,
    tools: list[dict[str, Any]] | None = None,
    reasoning_effort: str | float | None = "none",
) -> str:
    """Render an HF chat template with thinking configured for the detected model dialect."""
    kwargs: dict[str, Any] = {"add_generation_prompt": True, "tokenize": False}
    if tools is not None:
        kwargs["tools"] = tools
    try:
        if dialect == "inkling2":
            if reasoning_effort is not None:
                return tok.apply_chat_template(msgs, reasoning_effort=reasoning_effort, **kwargs)
            return tok.apply_chat_template(msgs, **kwargs)
        return tok.apply_chat_template(msgs, enable_thinking=_thinking_enabled(dialect, reasoning_effort), **kwargs)
    except TypeError:
        return tok.apply_chat_template(msgs, **kwargs)


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
        elif out.get("type") in ("tuple", "Array", "ArrayList", "array"):
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


def _raw_tools(row: dict[str, Any]) -> list[dict[str, Any]]:
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
            out.append(t)
        else:
            out.append({"type": "function", "function": t})
    return out


def _norm_tools(row: dict[str, Any]) -> list[dict[str, Any]]:
    return [_norm_schema(t) for t in _raw_tools(row)]


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


def parse_tool_calls(text: str, *, allowed_names: set[str] | None = None) -> list[dict[str, Any]]:
    """Parse tool calls across `Inkling 2`, `Qwen3.5` XML, `Gemma 4` `<|tool_call>call:...`, and `Qwen3` JSON."""
    if not text:
        return []
    cleaned = strip_thinking(text).strip()

    # 1. Inkling 2 `<|message_model|>NAME<|content_invoke_tool_json|>{"name":"NAME","args":{...}}<|end_message|>`
    ink = INKLING2_CALL_RE.findall(cleaned)
    if ink:
        calls = []
        for prefix_name, body in ink:
            obj = _parse_maybe(body)
            if isinstance(obj, dict):
                jname = str(obj.get("name") or "").strip()
                pname = prefix_name.strip()
                if allowed_names and jname in allowed_names:
                    fn_name = jname
                elif allowed_names and pname in allowed_names:
                    fn_name = pname
                elif pname and (not jname or ("." in pname and pname.startswith(jname + "."))):
                    fn_name = pname
                else:
                    fn_name = jname or pname
                raw_args = obj["args"] if "args" in obj else obj.get("arguments", {})
                args = _parse_maybe(raw_args)
                if fn_name and isinstance(args, dict):
                    calls.append(canonical({"name": fn_name, "arguments": args}))
        return calls

    # 2. Gemma 4 `<|tool_call>call:NAME{...}<tool_call|>`
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
    if dialect == "inkling2":
        args_json = json.dumps({str(k): v for k, v in sorted(args.items())}, ensure_ascii=False, separators=(",", ":"))
        payload = f'{{"name":{json.dumps(name, ensure_ascii=False)},"args":{args_json}}}'
        return f"<|message_model|>{name}<|content_invoke_tool_json|>{payload}<|end_message|><|content_model_end_sampling|>"
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


def _parse_lang_collection_str(s: str, want_dict: bool) -> Any:
    """Parse common Gorilla Java/JavaScript stringified list/map expressions into Python collections."""
    s = s.strip()
    if want_dict:
        m_hm = re.search(r"new\s+HashMap<[^>]*>\s*\(\)\s*\{\s*\{?\s*(.*?)\s*\}?\s*\}", s, re.DOTALL)
        if m_hm:
            out: dict[str, Any] = {}
            for km, vm in re.findall(r"put\(\s*([^,]+?)\s*,\s*([^)]+?)\s*\)", m_hm.group(1)):
                k = str(_parse_maybe(km.strip())).strip("'\"")
                out[k] = _parse_maybe(vm.strip())
            return out
        if s.startswith("{") and s.endswith("}"):
            inner = s[1:-1].strip()
            if not inner:
                return {}
            pairs = re.findall(r"([A-Za-z0-9_$\"'\.-]+)\s*:\s*([^,{}]+|\[[^\]]*\]|\{[^}]*\})", inner)
            if pairs:
                out = {}
                for k, v in pairs:
                    kk = k.strip().strip("'\"")
                    vv = v.strip()
                    if vv.startswith("{") and vv.endswith("}"):
                        out[kk] = _parse_lang_collection_str(vv, want_dict=True)
                    elif vv.startswith("[") and vv.endswith("]"):
                        out[kk] = _parse_lang_collection_str(vv, want_dict=False)
                    else:
                        out[kk] = _parse_maybe(vv)
                return out
        return s

    m_al = re.search(r"new\s+ArrayList<[^>]*>\s*\(\s*Arrays\.asList\((.+?)\)\s*\)", s, re.DOTALL)
    if m_al:
        return [_parse_maybe(x.strip().strip("'\"")) for x in m_al.group(1).split(",") if x.strip()]
    if re.fullmatch(r"new\s+ArrayList<[^>]*>\s*\(\s*\)", s):
        return []
    m_arr = re.search(r"new\s+[A-Za-z0-9_]+\[\]\s*\{(.*?)\}", s, re.DOTALL)
    if m_arr:
        body = m_arr.group(1).strip()
        return [] if not body else [_parse_maybe(x.strip().strip("'\"")) for x in body.split(",") if x.strip()]
    m_js = re.fullmatch(r"(?:new\s+Array\((.*)\)|\[(.*)\])", s, re.DOTALL)
    if m_js:
        body = (m_js.group(1) if m_js.group(1) is not None else m_js.group(2)).strip()
        return [] if not body else [_parse_maybe(x.strip().strip("'\"")) for x in body.split(",") if x.strip()]
    return s


def _norm_against_target(val: Any, tgt: Any) -> Any:
    val, tgt = _parse_maybe(val), _parse_maybe(tgt)
    if isinstance(val, str) and isinstance(tgt, dict):
        parsed = _parse_lang_collection_str(val, want_dict=True)
        if isinstance(parsed, dict):
            val = parsed
    elif isinstance(val, str) and isinstance(tgt, list):
        parsed = _parse_lang_collection_str(val, want_dict=False)
        if isinstance(parsed, list):
            val = parsed
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
        if not isinstance(tgt, bool) and isinstance(tgt, int) and re.fullmatch(r"[-+]?\d+[lLn]?", s):
            return int(s.rstrip("lLn"))
        if not isinstance(tgt, bool) and isinstance(tgt, float):
            s_num = s[:-1] if re.fullmatch(r"[-+]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?[fFdD]", s) else s
            try:
                return float(s_num)
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


DEFAULT_INKLING2_TOOL_SYSTEM_PROMPT = (
    "You are an expert in composing functions. You are given a question and a set of possible functions. "
    "Based on the question, call the appropriate function tool to achieve the purpose."
)


def _gorilla_standardize_str(s: str) -> str:
    return re.sub(r"[ \,\.\/\-\_\*\^]", "", s).lower().replace("'", '"')


def _gorilla_scalar_eq(pv: Any, rv: Any) -> bool:
    if pv == rv:
        return True
    if not isinstance(pv, bool) and not isinstance(rv, bool) and isinstance(pv, (int, float)) and isinstance(rv, (int, float)):
        return float(pv) == float(rv)
    if isinstance(pv, str) and isinstance(rv, str):
        return _gorilla_standardize_str(pv) == _gorilla_standardize_str(rv)
    return False


def _gorilla_val_match(pv: Any, rv: Any, *, allow_option_list: bool = False) -> bool:
    if _gorilla_scalar_eq(pv, rv):
        return True
    if isinstance(pv, dict) and isinstance(rv, dict):
        return _gorilla_dict_match(pv, rv, nested=True)
    if isinstance(pv, list) and isinstance(rv, list):
        if len(pv) == len(rv) and all(_gorilla_val_match(a, b, allow_option_list=False) for a, b in zip(pv, rv)):
            return True
    if allow_option_list and isinstance(rv, list):
        return any(_gorilla_val_match(pv, opt, allow_option_list=False) for opt in rv if opt != "")
    return False


def _gorilla_dict_match(pa: dict[str, Any], ra: dict[str, Any], *, nested: bool = False) -> bool:
    for k, pv in pa.items():
        if k not in ra:
            return False
        if not _gorilla_val_match(pv, ra[k], allow_option_list=nested):
            return False
    for k, rv in ra.items():
        if k not in pa:
            if rv == "" or (nested and isinstance(rv, list) and "" in rv):
                continue
            return False
    return True


def _apply_gorilla_lang_hints(
    tools: list[dict[str, Any]],
    category: str,
    *,
    raw_tools: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Mirror Gorilla BFCL's `func_doc_language_specific_pre_processing` on a copy of `tools`."""
    cat = str(category or "").lower()
    src = raw_tools if (raw_tools and cat in ("java", "javascript")) else tools
    if not src:
        return []
    if cat == "java":
        hint = " Note that the provided function is in Java 8 SDK syntax."
    elif cat == "javascript":
        hint = " Note that the provided function is in JavaScript syntax."
    else:
        hint = " Note that the provided function is in Python 3 syntax."
    out: list[dict[str, Any]] = []
    for t in src:
        tc = json.loads(json.dumps(t))
        fn = tc.get("function") if isinstance(tc.get("function"), dict) else tc
        desc = str(fn.get("description") or "")
        if hint.strip() not in desc:
            fn["description"] = desc + hint
        params = fn.get("parameters")
        if isinstance(params, dict):
            if params.get("type") in (None, "dict"):
                params["type"] = "object"
            props = params.get("properties")
            if isinstance(props, dict) and cat in ("java", "javascript"):
                lang_label = "Java" if cat == "java" else "JavaScript"
                for _k, val in props.items():
                    if not isinstance(val, dict):
                        continue
                    orig_type = str(val.get("type") or "any")
                    pdesc = str(val.get("description") or "")
                    if "in string representation." not in pdesc:
                        if orig_type == "any":
                            pdesc += f" This parameter can be of any type of {lang_label} object in string representation."
                        else:
                            pdesc += f" This is {lang_label} {orig_type} type parameter in string representation."
                        if cat == "java" and orig_type in ("ArrayList", "Array", "array") and isinstance(val.get("items"), dict):
                            item_t = val["items"].get("type", "any")
                            pdesc += f" The list elements are of type {item_t}; they are not in string representation."
                            val.pop("items", None)
                        elif cat == "javascript" and orig_type == "array" and isinstance(val.get("items"), dict):
                            item_t = val["items"].get("type", "any")
                            pdesc += f" The list elements are of type {item_t}; they are not in string representation."
                            val.pop("items", None)
                        elif cat == "javascript" and orig_type in ("dict", "object") and isinstance(val.get("properties"), dict):
                            pdesc += (
                                " The dictionary entries have the following schema; they are not in string representation. "
                                + json.dumps(val["properties"])
                            )
                            val.pop("properties", None)
                        val["description"] = pdesc
                    val["type"] = "string"
        out.append(tc)
    return out


def calls_match(pred_calls: list[dict[str, Any]], ref_calls: list[dict[str, Any]]) -> bool:
    if len(pred_calls) != 1 or not ref_calls:
        return False
    p = canonical(pred_calls[0])
    for ref in ref_calls:
        r = canonical(ref)
        if p == r:
            return True
        # Retry after normalizing scalar/list types against the reference
        np_ = canonical(_norm_against_target(p, r))
        nr_ = canonical(_norm_against_target(r, r))
        if np_ == nr_:
            return True
        # Retry with Gorilla BFCL AST rules: `""`-optional key omission, nested dict option lists, standardize_string
        for cand_p, cand_r in ((p, r), (np_, nr_)):
            if (
                isinstance(cand_p, dict)
                and isinstance(cand_r, dict)
                and cand_p.get("name") == cand_r.get("name")
                and isinstance(cand_p.get("arguments"), dict)
                and isinstance(cand_r.get("arguments"), dict)
                and _gorilla_dict_match(cand_p["arguments"], cand_r["arguments"], nested=False)
            ):
                return True
    return False


def detect_dialect(tok) -> tuple[str, list[str]]:
    vocab = tok.get_vocab()
    ct = str(getattr(tok, "chat_template", "") or "")
    if (
        "<|content_invoke_tool_json|>" in vocab
        or "<|message_model|>" in vocab
        or "<|content_invoke_tool_json|>" in ct
        or "<|message_model|>" in ct
    ):
        return "inkling2", ["<|content_model_end_sampling|>"]
    if "<|tool_call>" in vocab or "<turn|>" in vocab:
        return "gemma4", ["<turn|>", "<end_of_turn>", "<|tool_response>"]
    if "<function=" in ct:
        return "qwen35_xml", ["<|im_end|>"]
    return "qwen3_json", ["<|im_end|>"]


def _download_cached(cache_dir: Path, categories: tuple[str, ...]) -> list[dict[str, Any]]:
    cache_file = cache_dir / f"bfcl_v3_single_{GORILLA_COMMIT[:8]}_{'-'.join(categories)}.jsonl"
    if cache_file.exists():
        cached = _read_records(cache_file)
        needs_raw = any(r.get("category") in ("java", "javascript") and "raw_tools" not in r for r in cached)
        if not needs_raw:
            return cached
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
            raw_tools = _raw_tools(qr)
            tools = [_norm_schema(t) for t in raw_tools]
            if not tools:
                continue
            row_obj: dict[str, Any] = {
                "id": f"{cat}:{kid}",
                "category": cat,
                "messages": _norm_messages(qr),
                "tools": tools,
                "reference_calls": refs[:64],
            }
            if cat in ("java", "javascript"):
                row_obj["raw_tools"] = raw_tools
            rows.append(row_obj)
    tmp_file = cache_file.with_suffix(cache_file.suffix + ".tmp")
    with tmp_file.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp_file.replace(cache_file)
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


def stratified_sample(
    tasks: list[Task],
    n: int,
    *,
    split: str | None = None,
    categories: tuple[str, ...] | list[str] = DEFAULT_CATEGORIES,
) -> list[Task]:
    """Deterministically round-robin sample up to `n` tasks across categories, sorted by `t.id` within each category."""
    if n <= 0:
        return []
    pool = [t for t in tasks if t.split == split] if split is not None else list(tasks)
    buckets: dict[str, list[Task]] = {}
    for t in pool:
        cat = str((t.meta or {}).get("category") or (t.id.split(":", 1)[0] if ":" in t.id else "bfcl"))
        buckets.setdefault(cat, []).append(t)
    cat_order = [c for c in categories if buckets.get(c)] + sorted(c for c in buckets if c not in categories)
    for c in cat_order:
        buckets[c].sort(key=lambda t: t.id)
    out: list[Task] = []
    idx = 0
    while len(out) < n:
        added = False
        for c in cat_order:
            if idx < len(buckets[c]):
                out.append(buckets[c][idx])
                added = True
                if len(out) == n:
                    break
        if not added:
            break
        idx += 1
    return out


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
        reasoning_effort: str | float | None = "none",
        system_prompt: str | None = None,
        lang_hints: bool | None = None,
    ):
        from transformers import AutoTokenizer
        self.hf_dir = str(hf_dir)
        self.tok = AutoTokenizer.from_pretrained(self.hf_dir)
        self.dialect, self.stop = detect_dialect(self.tok)
        self.max_tokens = int(max_tokens)
        self.reasoning_effort = reasoning_effort
        if system_prompt is None and self.dialect == "inkling2":
            self.system_prompt: str | None = DEFAULT_INKLING2_TOOL_SYSTEM_PROMPT
        else:
            self.system_prompt = system_prompt
        self.lang_hints = (self.dialect == "inkling2") if lang_hints is None else bool(lang_hints)
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
            raw_split = r.get("split")
            sp_s = str(raw_split).strip() if raw_split is not None else ""
            has_split = sp_s != ""
            if has_split and r.get("family") is not None:
                fam, sp = str(r["family"]), sp_s
            elif has_split:
                raw_fam, _ = _assign_family_and_split(r, n_buckets_per_cat)
                sp = sp_s
                fam = f"{sp}/{raw_fam}"
            else:
                fam, sp = _assign_family_and_split(r, n_buckets_per_cat)
            meta: dict[str, Any] = {"category": r.get("category", "bfcl"), "reference_calls": r["reference_calls"]}
            if r.get("raw_tools"):
                meta["raw_tools"] = r["raw_tools"]
            self._tasks.append(Task(
                id=str(r["id"]),
                family=fam,
                split=sp,
                prompt={"messages": r["messages"], "tools": r["tools"]},
                meta=meta,
            ))

    def prepare_request_messages_and_tools(self, task: Task) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        msgs = [dict(m) for m in task.prompt["messages"]]
        if self.system_prompt and (not msgs or msgs[0].get("role") != "system"):
            msgs = [{"role": "system", "content": self.system_prompt}] + msgs
        tools = task.prompt["tools"]
        if self.lang_hints:
            meta = task.meta or {}
            tools = _apply_gorilla_lang_hints(
                tools,
                str(meta.get("category", "")),
                raw_tools=meta.get("raw_tools"),
            )
        return msgs, tools

    def render_prompt(self, task: Task) -> str:
        msgs, tools = self.prepare_request_messages_and_tools(task)
        return render_chat_prompt(
            self.tok,
            msgs,
            self.dialect,
            tools=tools,
            reasoning_effort=self.reasoning_effort,
        )

    def format_reference(self, task: Task) -> str:
        return format_tool_call(task.meta["reference_calls"][0], self.dialect)

    def tasks(self) -> list[Task]:
        return list(self._tasks)

    def sample_tasks(self, split: str, n: int, *, stratified: bool = True) -> list[Task]:
        pool = [t for t in self._tasks if t.split == split]
        if not stratified:
            return sorted(pool, key=lambda t: t.id)[: max(0, int(n))]
        return stratified_sample(pool, int(n))

    def stats_texts(self, n: int = 200) -> list[str]:
        search_ts = self.sample_tasks("search", int(n), stratified=True)
        return [self.render_prompt(t) + self.format_reference(t) for t in search_ts]

    def run(self, task: Task, client: Any, seed: int) -> Episode:
        t0 = time.time()
        req_msgs, req_tools = self.prepare_request_messages_and_tools(task)
        prompt_str = render_chat_prompt(
            self.tok,
            req_msgs,
            self.dialect,
            tools=req_tools,
            reasoning_effort=self.reasoning_effort,
        )
        srv_prompt = prompt_str[len(self.tok.bos_token):] if (self.tok.bos_token and prompt_str.startswith(self.tok.bos_token)) else prompt_str
        try:
            res = client.complete(srv_prompt, seed=seed, max_tokens=self.max_tokens, stop=self.stop)
        except InfraError as e:
            return Episode(task.id, seed, list(req_msgs), req_tools, None, [], 0, 0, time.time() - t0, str(e))
        raw_txt = (res["text"] or "").strip()
        if _thinking_enabled(self.dialect, self.reasoning_effort):
            asst_txt = raw_txt
        else:
            clean_txt = strip_thinking(raw_txt).strip()
            asst_txt = clean_txt or raw_txt
        msgs = list(req_msgs) + [{"role": "assistant", "content": asst_txt}]
        return Episode(
            task_id=task.id,
            seed=seed,
            messages=msgs,
            tools=req_tools,
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
        allowed_names = {t["function"]["name"] for t in (task.prompt.get("tools") or []) if "function" in t}
        pred_calls = parse_tool_calls(txt, allowed_names=allowed_names)
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
