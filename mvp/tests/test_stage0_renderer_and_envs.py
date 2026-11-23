"""Stage 0 unit tests for M1 (`kl.Renderer`, `ChatClient`, `Evaluator`, `sanity`, `bfcl`, `deepmath`)."""
from __future__ import annotations

import copy
import json
import shutil
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest
from tokenizers import Tokenizer, decoders, models, pre_tokenizers
from transformers import AutoTokenizer, PreTrainedTokenizerFast
from transformers.utils.chat_template_utils import render_jinja_template

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "mvp"))

from chantless import kl, runtime, sanity
from chantless.client import ChatClient
from chantless.env import bfcl, deepmath
from chantless.env.protocol import Episode, Task
from chantless.evaluator import Evaluator

INKLING2_COLAB_DIR = ROOT / ".local/inkling2-colab"
DEEPMATH_REP3K_DIR = ROOT / ".local/mvp/deepmath/rep3k-no01"
BFCL_CACHE_FILE = Path(
    "/tmp/chantless-bfcl-cache/bfcl_v3_single_70b6a4a2_simple-multiple-live_simple-live_multiple-java-javascript-sql.jsonl"
)


def _build_inkling2_hf_dir(tmp_path: Path, name: str = "hf_inkling2") -> Path:
    """Build an offline HF tokenizer dir using real `.local/inkling2-colab/` configs + padded ByteLevel BPE."""
    hf_dir = tmp_path / name
    hf_dir.mkdir(parents=True, exist_ok=True)
    for fn in ("chat_template.jinja", "tokenizer_config.json", "special_tokens_map.json"):
        shutil.copyfile(INKLING2_COLAB_DIR / fn, hf_dir / fn)
    tcfg = json.loads((hf_dir / "tokenizer_config.json").read_text(encoding="utf-8"))
    min_added = min(int(k) for k in tcfg["added_tokens_decoder"])
    alphabet = list(pre_tokenizers.ByteLevel.alphabet())
    vocab = {tok: i for i, tok in enumerate(alphabet)}
    for i in range(len(alphabet), min_added):
        vocab[f"<|pad_vocab_{i}|>"] = i
    bpe = Tokenizer(models.BPE(vocab=vocab, merges=[]))
    bpe.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    bpe.decoder = decoders.ByteLevel()
    bpe.save(str(hf_dir / "tokenizer.json"))
    return hf_dir


def _make_recording_server(responder):
    records: list[dict] = []
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length)
            body = json.loads(raw.decode("utf-8"))
            with lock:
                records.append({"path": self.path, "body": body})
            status, payload = responder(self.path, body)
            data = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, format, *args):
            return

    srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    return srv, th, f"http://127.0.0.1:{srv.server_port}", records


# ------------------------------------------------------------------ 1. `kl.Renderer` Inkling-2 support
def test_kl_renderer_inkling2_effort_end_ids_and_no_double_wrap(tmp_path: Path):
    hf_dir = _build_inkling2_hf_dir(tmp_path)
    R = kl.Renderer(str(hf_dir))

    assert R.dialect == "inkling2"
    assert R.end_ids == [200006]
    assert R.im_end == 200006
    assert R.tok.eos_token_id == 200006

    tools = [
        {
            "type": "function",
            "function": {
                "name": "calc",
                "description": "Compute x",
                "parameters": {"type": "object", "properties": {"x": {"type": "integer"}}},
            },
        }
    ]
    bfcl_wire_stripped = '<|message_model|>calc<|content_invoke_tool_json|>{"name":"calc","args":{"x":2}}<|end_message|>'
    bfcl_wire_full = bfcl_wire_stripped + "<|content_model_end_sampling|>"
    dm_wire_stripped = "<|message_model|><|content_text|>\\boxed{42}<|end_message|>"
    trunc_wire = "<|message_model|><|content_thinking|>partial"

    for label, wire, use_tools in (
        ("bfcl_stripped", bfcl_wire_stripped, tools),
        ("bfcl_full", bfcl_wire_full, tools),
        ("deepmath_stripped", dm_wire_stripped, None),
        ("truncated_thinking", trunc_wire, None),
    ):
        msgs = [
            {"role": "system", "content": "Solve concisely."},
            {"role": "user", "content": "Go."},
            {"role": "assistant", "content": wire},
        ]
        pre_str = R._render(msgs[:2], use_tools, True)
        seq_str = R._render(msgs, use_tools, False)
        assert "Thinking effort level: 0<|end_message|>" in pre_str, label
        assert "Thinking effort level: 0.9" not in pre_str, label
        assert "Thinking effort level: 0.9" not in seq_str, label

        recs = R.episode(msgs, use_tools)
        assert len(recs) == 1, label
        seq, tg = recs[0]
        assert seq[-1] == 200006, (label, seq[-1])
        decoded = R.tok.decode(seq, skip_special_tokens=False)
        assert "<|message_model|><|content_text|><|message_model|>" not in decoded, (label, decoded)
        assert "<|content_model_end_sampling|><|content_model_end_sampling|>" not in decoded, (label, decoded)

        pre_ids = R.ids(pre_str)
        assert seq[: len(pre_ids)] == pre_ids, label
        assert tg == list(range(len(pre_ids), len(seq))), label
        target_dec = R.tok.decode([seq[j] for j in tg], skip_special_tokens=False)
        expected_wire = wire if wire.endswith("<|content_model_end_sampling|>") else wire + "<|content_model_end_sampling|>"
        assert target_dec == expected_wire, (label, target_dec, expected_wire)


def test_kl_renderer_inkling2_multiturn_and_structured_tool_calls(tmp_path: Path):
    hf_dir = _build_inkling2_hf_dir(tmp_path)
    R = kl.Renderer(str(hf_dir))

    tools = [
        {
            "type": "function",
            "function": {
                "name": "calc",
                "description": "Compute x",
                "parameters": {"type": "object", "properties": {"x": {"type": "integer"}}},
            },
        }
    ]
    a1_wire = '<|message_model|>calc<|content_invoke_tool_json|>{"name":"calc","args":{"x":7}}<|end_message|>'
    a2_wire = "<|message_model|><|content_text|>Result is 14.<|end_message|>"
    mt_msgs = [
        {"role": "system", "content": "Be helpful."},
        {"role": "user", "content": "Compute for 7."},
        {"role": "assistant", "content": a1_wire},
        {"role": "tool", "name": "calc", "content": '{"result": 14}'},
        {"role": "assistant", "content": a2_wire},
    ]
    recs = R.episode(mt_msgs, tools)
    assert len(recs) == 2

    dec1 = R.tok.decode(recs[0][0], skip_special_tokens=False)
    dec2 = R.tok.decode(recs[1][0], skip_special_tokens=False)
    for dec in (dec1, dec2):
        assert "<|message_model|><|content_text|><|message_model|>" not in dec
        assert "<|content_model_end_sampling|><|content_model_end_sampling|>" not in dec
    assert dec2.count("<|content_model_end_sampling|>") == 2
    assert recs[0][0][-1] == 200006 and recs[1][0][-1] == 200006
    t2_str = R.tok.decode([recs[1][0][j] for j in recs[1][1]], skip_special_tokens=False)
    assert t2_str == a2_wire + "<|content_model_end_sampling|>"

    # Structured `tool_calls` with JSON-string `arguments` + plain-text assistant turn
    structured_msgs = [
        {"role": "user", "content": "Compute for 3."},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"type": "function", "function": {"name": "calc", "arguments": '{"x": 3}'}}],
        },
        {"role": "tool", "name": "calc", "content": "6"},
        {"role": "assistant", "content": "Done."},
    ]
    s_recs = R.episode(structured_msgs, tools)
    assert len(s_recs) == 2
    s_dec2 = R.tok.decode(s_recs[1][0], skip_special_tokens=False)
    assert '<|message_model|>calc<|content_invoke_tool_json|>{"name":"calc","args":{"x":3}}<|end_message|><|content_model_end_sampling|>' in s_dec2
    assert "<|message_model|><|content_text|>Done.<|end_message|><|content_model_end_sampling|>" in s_dec2


def test_kl_renderer_bpe_boundary_merge_fallback_and_eos_id_zero(tmp_path: Path):
    # 1. BPE boundary-merge across `pre_str` (ending with ":") and assistant reply (starting with " ")
    hf_merge = tmp_path / "hf_merge"
    hf_merge.mkdir()
    vocab = {
        "<|im_start|>": 0,
        "<|im_end|>": 1,
        "<unk>": 2,
        "user": 3,
        "assistant": 4,
        "hi": 5,
        ":": 6,
        " ": 7,
        ": ": 8,
        "ok": 9,
    }
    bpe = Tokenizer(models.BPE(vocab=vocab, merges=[(":", " ")], unk_token="<unk>"))
    ct = (
        "{%- for m in messages -%}"
        "{{- '<|im_start|>' ~ m.role ~ ':' ~ m.content ~ '<|im_end|>' -}}"
        "{%- endfor -%}"
        "{%- if add_generation_prompt -%}{{- '<|im_start|>assistant:' -}}{%- endif -%}"
    )
    tok = PreTrainedTokenizerFast(tokenizer_object=bpe, eos_token="<|im_end|>", unk_token="<unk>", chat_template=ct)
    tok.save_pretrained(hf_merge)
    R_merge = kl.Renderer(str(hf_merge))
    # Confirm raw tokenization genuinely merges `:` + ` ` into `: ` (8) instead of `[6, 7]`
    pre_str = R_merge._render([{"role": "user", "content": "hi"}], None, True)
    seq_str = R_merge._render([{"role": "user", "content": "hi"}, {"role": "assistant", "content": " ok"}], None, False)
    assert seq_str.startswith(pre_str)
    assert R_merge.ids(seq_str)[: len(R_merge.ids(pre_str))] != R_merge.ids(pre_str)
    recs = R_merge.episode([{"role": "user", "content": "hi"}, {"role": "assistant", "content": " ok"}], None)
    assert len(recs) == 1
    assert recs[0][0][: len(R_merge.ids(pre_str))] == R_merge.ids(pre_str)
    assert recs[0][0][-1] == 1

    # 2. `eos_token_id == 0` in `kl.pack_text` must NOT fall back to `im_end` (1)
    hf_eos0 = tmp_path / "hf_eos0"
    hf_eos0.mkdir()
    wl = Tokenizer(models.WordLevel(vocab={"<eos>": 0, "<|im_end|>": 1, "<unk>": 2, "hello": 3, "world": 4}, unk_token="<unk>"))
    wl.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tok0 = PreTrainedTokenizerFast(tokenizer_object=wl, eos_token="<eos>", unk_token="<unk>", chat_template="{{ messages[0].content }}")
    tok0.save_pretrained(hf_eos0)
    R0 = kl.Renderer(str(hf_eos0))
    assert R0.tok.eos_token_id == 0
    assert R0.im_end == 1
    packed = kl.pack_text(R0, ["hello", "world", "hello", "world"], seq_len=8, n_seq=1)
    assert len(packed) == 1
    assert packed[0][0] == [3, 0, 4, 0, 3, 0, 4, 0]

    # Missing both `eos_token_id` and `im_end` raises RenderError
    R0.tok.eos_token_id = None
    R0.im_end = None
    with pytest.raises(kl.RenderError):
        kl.pack_text(R0, ["hello"], seq_len=4, n_seq=1)


# ------------------------------------------------------------------ 2. `ChatClient`, `Evaluator`, `sanity`
def test_chat_client_dialect_and_override_no_leak():
    def responder(path, body):
        if path == "/v1/chat/completions":
            return 200, {"choices": [{"message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}], "usage": {"completion_tokens": 1}}
        return 200, {"choices": [{"text": "ok", "finish_reason": "stop"}], "usage": {"completion_tokens": 1}}

    srv, th, url, recs = _make_recording_server(responder)
    try:
        # Default (`dialect=None`)
        c_def = ChatClient(url, {"temperature": 0.0})
        c_def.chat([{"role": "user", "content": "hi"}], seed=1, max_tokens=16)
        assert recs[-1]["body"]["chat_template_kwargs"] == {"enable_thinking": False}

        # Inkling-2 dialect + verify against real `.local/inkling2-colab/chat_template.jinja`
        c_ink = ChatClient(url, {"temperature": 0.0}, dialect="inkling2")
        c_ink.chat([{"role": "user", "content": "Count from 1 to 50."}], seed=1, max_tokens=16)
        ink_body = recs[-1]["body"]
        assert ink_body["chat_template_kwargs"] == {"reasoning_effort": "none"}
        ct_text = (INKLING2_COLAB_DIR / "chat_template.jinja").read_text(encoding="utf-8")
        rendered, _ = render_jinja_template(
            [ink_body["messages"]],
            chat_template=ct_text,
            add_generation_prompt=True,
            **ink_body["chat_template_kwargs"],
        )
        assert "Thinking effort level: 0<|end_message|>" in rendered[0]
        assert "Thinking effort level: 0.9" not in rendered[0]

        # `decoding["chat_template_kwargs"]` + `"enable_thinking"` must not leak or mutate caller dicts
        decoding = {"temperature": 0.0, "chat_template_kwargs": {"reasoning_effort": "low"}, "enable_thinking": True}
        snap = copy.deepcopy(decoding)
        c_dec = ChatClient(url, decoding)
        c_dec.chat([{"role": "user", "content": "hi"}], seed=2, max_tokens=16)
        chat_body = recs[-1]["body"]
        assert chat_body["chat_template_kwargs"] == {"reasoning_effort": "low"}
        assert chat_body["temperature"] == 0.0
        assert "enable_thinking" not in chat_body

        c_dec.complete("prompt text", seed=3, max_tokens=16)
        comp_body = recs[-1]["body"]
        assert "chat_template_kwargs" not in comp_body
        assert "enable_thinking" not in comp_body
        assert comp_body["temperature"] == 0.0
        assert decoding == snap

        # Explicit `__init__` + per-call override without mutating `client.chat_template_kwargs`
        c_ovr = ChatClient(url, {"temperature": 0.0}, chat_template_kwargs={"reasoning_effort": "minimal"})
        c_ovr.chat([{"role": "user", "content": "hi"}], seed=4, max_tokens=16, chat_template_kwargs={"reasoning_effort": "none"})
        assert recs[-1]["body"]["chat_template_kwargs"] == {"reasoning_effort": "none"}
        assert c_ovr.chat_template_kwargs == {"reasoning_effort": "minimal"}
    finally:
        srv.shutdown()
        srv.server_close()
        th.join(timeout=2)


def test_evaluator_passes_dialect_to_chat_client(tmp_path: Path):
    seen_ctk: list[dict] = []

    def responder(path, body):
        seen_ctk.append(body.get("chat_template_kwargs"))
        return 200, {"choices": [{"message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}], "usage": {"completion_tokens": 1}}

    class StubEnv:
        classifier = False

        def __init__(self, dialect=None):
            if dialect is not None:
                self.dialect = dialect

        def run(self, task, client, seed):
            res = client.chat([{"role": "user", "content": "hi"}], seed=seed, max_tokens=8)
            return Episode(task.id, seed, [], None, res["message"]["content"], [res["finish_reason"]], 1, 1, 0.01)

        def verify(self, task, ep):
            return SimpleNamespace(success=True, tests_passed=True, detectors={}, infra_error=None, details={})

    srv, th, url, _ = _make_recording_server(responder)
    tc = runtime.Toolchain(llama_bin=tmp_path, traj_kl=tmp_path / "traj-kl", gpus=[""])
    t = Task(id="t0", family="f0", split="select", prompt={})
    try:
        ev_ink = Evaluator(tc, StubEnv("inkling2"), {"temperature": 0.0}, 1, 1024, "f16", tmp_path / "l1")
        ev_ink.run("m", [(t, 0)], "tag1", save=False, srvs=[SimpleNamespace(url=url, special=True)])
        assert seen_ctk[-1] == {"reasoning_effort": "none"}

        ev_q3 = Evaluator(tc, StubEnv("qwen3_json"), {"temperature": 0.0}, 1, 1024, "f16", tmp_path / "l2")
        ev_q3.run("m", [(t, 0)], "tag2", save=False, srvs=[SimpleNamespace(url=url, special=False)])
        assert seen_ctk[-1] == {"enable_thinking": False}

        ev_ovr = Evaluator(tc, StubEnv(None), {"temperature": 0.0}, 1, 1024, "f16", tmp_path / "l3", dialect="inkling2")
        assert ev_ovr.dialect == "inkling2" and ev_ovr.special is True
        ev_ovr.run("m", [(t, 0)], "tag3", save=False, srvs=[SimpleNamespace(url=url, special=False)])
        assert seen_ctk[-1] == {"reasoning_effort": "none"}

        ev_srv = Evaluator(tc, StubEnv(None), {"temperature": 0.0}, 1, 1024, "f16", tmp_path / "l4")
        ev_srv.run("m", [(t, 0)], "tag4", save=False, srvs=[SimpleNamespace(url=url, special=True)])
        assert seen_ctk[-1] == {"reasoning_effort": "none"}
    finally:
        srv.shutdown()
        srv.server_close()
        th.join(timeout=2)


def test_sanity_check_and_run_behavioral_inkling2():
    # Direct `_check` cases
    it_exact = {"check": "exact", "answer": "30"}
    assert sanity._check(it_exact, "<|message_model|><|content_thinking|>10+20=30<|end_message|><|message_model|><|content_text|>30<|end_message|><|content_model_end_sampling|>")
    assert sanity._check(it_exact, "<|content_thinking|>x<|end_message|><|content_text|>30<|end_message|>")
    assert sanity._check(it_exact, "<think>10+20</think>\n30")
    assert not sanity._check(it_exact, "<think>30")
    assert sanity._check(it_exact, "<|channel>thought\n30<channel|>30")

    assert sanity._check({"check": "exact_lower", "answer": "enots"}, "<|message_model|><|content_text|>ENOTS.<|end_message|>")
    assert sanity._check({"check": "contains", "answer": "Paris"}, "<|message_model|><|content_text|>Paris<|end_message|>")
    assert not sanity._check({"check": "contains", "answer": "Paris"}, "<|message_model|><|content_thinking|>Paris")
    assert sanity._check(
        {"check": "json", "answer": {"name": "Ana", "age": 25}},
        '<|message_model|><|content_thinking|>plan<|end_message|><|message_model|><|content_text|>```json\n{"name": "Ana", "age": 25}\n```<|end_message|><|content_model_end_sampling|>',
    )

    # Loopback `run_behavioral`
    items = [
        {"id": "arith-0", "prompt": "10+20?", "check": "exact", "answer": "30"},
        {"id": "cap-1", "prompt": "Capital of France?", "check": "contains", "answer": "Paris"},
        {"id": "json-2", "prompt": "JSON", "check": "json", "answer": {"name": "Ana", "age": 25}},
    ]
    mode = {"finish": "stop"}

    def responder(path, body):
        prompt = body["messages"][0]["content"]
        if mode["finish"] == "length":
            return 200, {"choices": [{"message": {"role": "assistant", "content": "<|message_model|><|content_thinking|>Paris"}, "finish_reason": "length"}], "usage": {"completion_tokens": 64}}
        if "10+20" in prompt:
            txt = "<|message_model|><|content_text|>30<|end_message|><|content_model_end_sampling|>"
        elif "France" in prompt:
            txt = "<|message_model|><|content_text|>Paris<|end_message|>"
        else:
            txt = '<|message_model|><|content_text|>{"name": "Ana", "age": 25}<|end_message|>'
        return 200, {"choices": [{"message": {"role": "assistant", "content": txt}, "finish_reason": "stop"}], "usage": {"completion_tokens": 5}}

    srv, th, url, recs = _make_recording_server(responder)
    try:
        rows_ink = sanity.run_behavioral([SimpleNamespace(url=url, special=True)], {"temperature": 0.0}, items, [0], "src")
        assert all(r["success"] is True for r in rows_ink)
        assert all(r["body"]["chat_template_kwargs"] == {"reasoning_effort": "none"} for r in recs)

        recs.clear()
        rows_q3 = sanity.run_behavioral([SimpleNamespace(url=url, special=False)], {"temperature": 0.0}, items, [0], "src")
        assert all(r["success"] is True for r in rows_q3)
        assert all(r["body"]["chat_template_kwargs"] == {"enable_thinking": False} for r in recs)

        mode["finish"] = "length"
        rows_len = sanity.run_behavioral([SimpleNamespace(url=url, special=True)], {"temperature": 0.0}, items, [0], "cand")
        assert all(r["success"] is False for r in rows_len)
    finally:
        srv.shutdown()
        srv.server_close()
        th.join(timeout=2)


# ------------------------------------------------------------------ 3. `bfcl.py` & `deepmath.py`
def test_bfcl_render_chat_prompt_reasoning_effort(tmp_path: Path):
    hf_dir = _build_inkling2_hf_dir(tmp_path)
    tok = AutoTokenizer.from_pretrained(str(hf_dir))
    msgs = [{"role": "user", "content": "Hello"}]

    assert "Thinking effort level: 0<|end_message|>" in bfcl.render_chat_prompt(tok, msgs, "inkling2")
    assert "Thinking effort level: 0<|end_message|>" in bfcl.render_chat_prompt(tok, msgs, "inkling2", reasoning_effort="none")
    assert "Thinking effort level: 0.2<|end_message|>" in bfcl.render_chat_prompt(tok, msgs, "inkling2", reasoning_effort="low")
    assert "Thinking effort level: 0.2<|end_message|>" in bfcl.render_chat_prompt(tok, msgs, "inkling2", reasoning_effort=0.2)
    assert "Thinking effort level: 0.9<|end_message|>" in bfcl.render_chat_prompt(tok, msgs, "inkling2", reasoning_effort=None)

    # Non-inkling2 dialect keeps `enable_thinking=False` path cleanly
    seen = {}

    class DummyTok:
        def apply_chat_template(self, m, **kw):
            seen.update(kw)
            return "rendered"

    assert bfcl.render_chat_prompt(DummyTok(), msgs, "qwen3_json") == "rendered"
    assert seen["enable_thinking"] is False and "reasoning_effort" not in seen


def test_bfcl_stratified_sample_round_robin_and_default_order(tmp_path: Path):
    hf_dir = _build_inkling2_hf_dir(tmp_path)
    pairs = tmp_path / "bfcl_pairs.jsonl"
    rows = []
    # Put 3 tasks per category in `select` except only 1 in `javascript`, and put `sql` only in `select`
    counts_sel = {"simple": 3, "multiple": 3, "live_simple": 3, "live_multiple": 3, "java": 3, "javascript": 1, "sql": 3}
    for cat, k in counts_sel.items():
        for i in range(k):
            rows.append({
                "id": f"{cat}:{cat}_{i:02d}",
                "category": cat,
                "family": f"{cat}/select",
                "split": "select",
                "messages": [{"role": "user", "content": f"{cat} {i}"}],
                "tools": [{"type": "function", "function": {"name": "fn", "parameters": {"type": "object", "properties": {}}}}],
                "reference_calls": [{"name": "fn", "arguments": {}}],
            })
    # Add required search/exam rows
    for sp in ("search", "exam"):
        rows.append({
            "id": f"simple:{sp}_0",
            "category": "simple",
            "family": f"simple/{sp}",
            "split": sp,
            "messages": [{"role": "user", "content": sp}],
            "tools": [{"type": "function", "function": {"name": "fn", "parameters": {"type": "object", "properties": {}}}}],
            "reference_calls": [{"name": "fn", "arguments": {}}],
        })
    pairs.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    env = bfcl.BfclEnvironment(hf_dir=str(hf_dir), pairs_file=str(pairs))
    # Default `.tasks()` ordering preserves input order
    assert [t.id for t in env.tasks()] == [r["id"] for r in rows]

    sampled = env.sample_tasks("select", 14, stratified=True)
    assert len(sampled) == 14
    by_cat = {}
    for t in sampled:
        by_cat[t.meta["category"]] = by_cat.get(t.meta["category"], 0) + 1
    assert set(by_cat) == set(bfcl.DEFAULT_CATEGORIES)
    assert by_cat["javascript"] == 1
    assert all(by_cat[c] >= 2 for c in bfcl.DEFAULT_CATEGORIES if c != "javascript")

    lex = env.sample_tasks("select", 5, stratified=False)
    assert [t.id for t in lex] == sorted([t.id for t in env.tasks() if t.split == "select"])[:5]
    assert bfcl.stratified_sample(env.tasks(), 0, split="select") == []

    # Verify against the local BFCL cache when present
    if BFCL_CACHE_FILE.exists():
        real_env = bfcl.BfclEnvironment(hf_dir=str(hf_dir), cache_dir=str(BFCL_CACHE_FILE.parent))
        strat80 = real_env.sample_tasks("select", 80, stratified=True)
        assert len(strat80) == 80
        cat_counts = {c: sum(1 for t in strat80 if t.meta["category"] == c) for c in bfcl.DEFAULT_CATEGORIES}
        assert all(cat_counts[c] in (11, 12) for c in bfcl.DEFAULT_CATEGORIES), cat_counts
        lex80 = real_env.sample_tasks("select", 80, stratified=False)
        lex_cats = {c: sum(1 for t in lex80 if t.meta["category"] == c) for c in bfcl.DEFAULT_CATEGORIES}
        assert lex_cats["simple"] == 0 and lex_cats["sql"] == 0 and lex_cats["java"] == 33


def test_deepmath_extract_boxed_int_assignments():
    assert deepmath.extract_boxed_int(r"\boxed{=42}") == 42
    assert deepmath.extract_boxed_int(r"\boxed{=-17}") == -17
    assert deepmath.extract_boxed_int(r"\boxed{x = 42}") == 42
    assert deepmath.extract_boxed_int(r"\boxed{x=-17}") == -17
    assert deepmath.extract_boxed_int(r"\boxed{ans = +7}") == 7
    assert deepmath.extract_boxed_int(r"\boxed{x_1 = 42}") == 42
    assert deepmath.extract_boxed_int(r"\boxed{\text{x} = -17}") == -17
    assert deepmath.extract_boxed_int(r"\boxed{-1,234}") == -1234
    assert deepmath.extract_boxed_int(r"\boxed{42\,}") == 42
    assert deepmath.extract_boxed_int(r"\boxed{1\,000}") == 1000
    assert deepmath.extract_boxed_int(r"\boxed{x = $42$}") == 42
    assert deepmath.extract_boxed_int(r"\boxed{1=2}") is None
    assert deepmath.extract_boxed_int(r"\boxed{x = }") is None
    assert deepmath.extract_boxed_int(r"\boxed{x=y=5}") is None


def test_deepmath_rep3k_group_split_topic_and_reasoning_effort(tmp_path: Path):
    hf_dir = _build_inkling2_hf_dir(tmp_path)
    syn_jsonl = tmp_path / "syn_rep3k.jsonl"
    syn_rows = [
        {"id": "r0", "question": "1+1?", "answer": "2", "difficulty": 4.0, "group": 0, "split": "search", "topic": "Mathematics -> Calculus -> Integrals"},
        {"id": "r1", "question": "2+2?", "answer": "4", "difficulty": 5.0, "group": 7, "split": "exam", "topic": "Mathematics -> Algebra -> Linear"},
    ]
    syn_jsonl.write_text("\n".join(json.dumps(r) for r in syn_rows) + "\n", encoding="utf-8")

    env_none = deepmath.DeepMathEnvironment(hf_dir=str(hf_dir), jsonl_file=str(syn_jsonl))
    t0, t1 = env_none.tasks()
    assert (t0.family, t0.split, t0.meta["area"]) == ("g0", "search", "Calculus")
    assert (t1.family, t1.split, t1.meta["area"]) == ("g7", "exam", "Algebra")
    assert "Thinking effort level: 0<|end_message|>" in env_none.render_prompt(t0)

    env_low = deepmath.DeepMathEnvironment(hf_dir=str(hf_dir), jsonl_file=str(syn_jsonl), reasoning_effort="low")
    assert "Thinking effort level: 0.2<|end_message|>" in env_low.render_prompt(t0)

    env_default = deepmath.DeepMathEnvironment(hf_dir=str(hf_dir), jsonl_file=str(syn_jsonl), reasoning_effort=None)
    assert "Thinking effort level: 0.9<|end_message|>" in env_default.render_prompt(t0)

    # Real `.local/mvp/deepmath/rep3k-no01/` files
    search_p = DEEPMATH_REP3K_DIR / "deepmath-int-rep3k-search.jsonl"
    exam_p = DEEPMATH_REP3K_DIR / "deepmath-int-rep3k-exam.jsonl"
    assert search_p.exists() and exam_p.exists()
    env_srch = deepmath.DeepMathEnvironment(hf_dir=str(hf_dir), jsonl_file=str(search_p))
    env_exam = deepmath.DeepMathEnvironment(hf_dir=str(hf_dir), jsonl_file=str(exam_p))
    srch_tasks = env_srch.tasks()
    exam_tasks = env_exam.tasks()
    assert len(srch_tasks) == 2009 and len({t.family for t in srch_tasks}) == 249
    assert all(t.split == "search" and t.family.startswith("g") and t.meta["area"] != "Math" for t in srch_tasks)
    assert len(exam_tasks) == 991 and len({t.family for t in exam_tasks}) == 51
    assert all(t.split == "exam" and t.family.startswith("g") and t.meta["area"] != "Math" for t in exam_tasks)
    assert not ({t.family for t in srch_tasks} & {t.family for t in exam_tasks})


def test_deepmath_verify_separates_premature_stop_and_no_boxed(tmp_path: Path):
    hf_dir = _build_inkling2_hf_dir(tmp_path)
    syn_jsonl = tmp_path / "dm.jsonl"
    syn_jsonl.write_text(json.dumps({"id": "q0", "question": "Q", "answer": "42", "difficulty": 4.0, "group": 1, "split": "search", "topic": "Mathematics -> Algebra"}) + "\n", encoding="utf-8")
    env = deepmath.DeepMathEnvironment(hf_dir=str(hf_dir), jsonl_file=str(syn_jsonl))
    task = env.tasks()[0]

    # Case A: finish_reason="stop" with no `\boxed{...}` -> premature_stop False, no_boxed True
    ep_a = Episode(task.id, 0, [], None, "The answer is 42.", ["stop"], 1, 5, 0.1)
    v_a = env.verify(task, ep_a)
    assert v_a.tests_passed is False
    assert v_a.detectors["premature_stop"] is False
    assert v_a.details["no_boxed"] is True
    assert v_a.success is False

    # Case B: finish_reason="length" with valid `\boxed{42}` -> premature_stop True, no_boxed False, success False
    ep_b = Episode(task.id, 0, [], None, r"\boxed{42}", ["length"], 1, 1024, 0.1)
    v_b = env.verify(task, ep_b)
    assert v_b.tests_passed is True
    assert v_b.detectors["premature_stop"] is True
    assert v_b.details["no_boxed"] is False
    assert v_b.success is False

    # Case C: finish_reason="length" with no `\boxed{...}` -> premature_stop True, no_boxed True
    ep_c = Episode(task.id, 0, [], None, "<|message_model|><|content_thinking|>still thinking", ["length"], 1, 1024, 0.1)
    v_c = env.verify(task, ep_c)
    assert v_c.tests_passed is False
    assert v_c.detectors["premature_stop"] is True
    assert v_c.details["no_boxed"] is True
    assert v_c.success is False
