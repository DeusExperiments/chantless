"""Procedurally generated Python function tasks with hidden unit tests.

Entry point: make_environment(**kwargs). Each episode is a short tool-using interaction: the model
may call `run_tests` on the visible examples, then must answer with one ```python block. Success
is decided by hidden tests (never shown) and the gating detectors.
"""
from __future__ import annotations

import json
import random
import time

from .. import detectors
from ..client import InfraError
from ..util import stable_seed
from . import sandbox
from .families import FAMILIES
from .protocol import Episode, Task, Verdict

SYSTEM = "You are a careful Python programmer. Write correct, simple code."

TOOLS = [{
    "type": "function",
    "function": {
        "name": "run_tests",
        "description": "Run your candidate function against the visible examples and report which pass.",
        "parameters": {
            "type": "object",
            "properties": {"code": {"type": "string", "description": "Complete Python source defining the function."}},
            "required": ["code"],
        },
    },
}]
TOOL_NAMES = {t["function"]["name"] for t in TOOLS}

DEFAULT_SPLIT_SIZES = {"search": 14, "select": 10, "exam": 10, "train": 8}


def _json_safe(x):
    return json.loads(json.dumps(x))


def build_tasks(seed: int, variants_per_family: int, split_sizes: dict) -> list[Task]:
    fams = sorted(FAMILIES)
    if sum(split_sizes.values()) != len(fams):
        raise ValueError(f"split sizes {split_sizes} must cover all {len(fams)} families")
    order = fams[:]
    random.Random(seed).shuffle(order)
    fam_split, i = {}, 0
    for split in ("search", "select", "exam", "train"):
        for f in order[i : i + split_sizes[split]]:
            fam_split[f] = split
        i += split_sizes[split]
    tasks = []
    for fam in fams:
        made = set()
        for v in range(variants_per_family * 4):
            if len(made) >= variants_per_family:
                break
            rng = random.Random(stable_seed(seed, fam, v))
            fname, sig, desc, ref, inputs = FAMILIES[fam](rng)
            ns: dict = {}
            exec(ref, ns)
            fn = ns[fname]
            tests = [[_json_safe(list(a)), _json_safe(fn(*a))] for a in inputs]
            key = json.dumps([fname, desc, tests])
            if key in made:
                continue
            made.add(key)
            tid = f"{fam}-{len(made) - 1:03d}"
            visible, hidden = tests[:2], tests[2:]
            ex = "\n".join(f"  {fname}({', '.join(map(repr, a))}) == {e!r}" for a, e in visible)
            user = (f"Write a Python function `{fname}{sig}` that {desc}.\n\nExamples:\n{ex}\n\n"
                    f"You may call the `run_tests` tool with your code to check it against the examples. "
                    f"When you are done, reply with the final function in a single ```python code block.")
            tasks.append(Task(id=tid, family=fam, split=fam_split[fam],
                              prompt={"system": SYSTEM, "user": user},
                              meta={"fname": fname, "visible": visible, "hidden": hidden, "reference": ref}))
    return tasks


class CodeTasksEnv:
    name = "chantless-codetasks"
    version = "1"

    def __init__(self, seed: int = 20261012, variants_per_family: int = 40, split_sizes: dict | None = None,
                 max_turns: int = 3, max_tokens_per_turn: int = 768, context: int = 8192, concurrency: int = 1,
                 kv_type: str = "f16", max_total_memory_bytes: int | None = None):
        self.seed = seed
        self.budget = {"max_turns": max_turns, "max_tokens_per_turn": max_tokens_per_turn}
        self.deployment = {"context": context, "concurrency": concurrency, "kv_type": kv_type,
                           "max_total_memory_bytes": max_total_memory_bytes}
        self._tasks = build_tasks(seed, variants_per_family, split_sizes or DEFAULT_SPLIT_SIZES)

    def tasks(self) -> list[Task]:
        return list(self._tasks)

    def run(self, task: Task, client, seed: int) -> Episode:
        msgs = [{"role": "system", "content": task.prompt["system"]}, {"role": "user", "content": task.prompt["user"]}]
        finish, ntok, final = [], 0, None
        t0 = time.time()
        turns = 0
        try:
            for turn in range(self.budget["max_turns"]):
                turns += 1
                r = client.chat(msgs, seed=stable_seed(seed, turn), max_tokens=self.budget["max_tokens_per_turn"], tools=TOOLS)
                m = r["message"]
                finish.append(r["finish_reason"] or "")
                ntok += r["completion_tokens"] or 0
                am = {"role": "assistant", "content": m.get("content") or ""}
                if m.get("tool_calls"):
                    am["tool_calls"] = m["tool_calls"]
                msgs.append(am)
                if not m.get("tool_calls"):
                    final = am["content"]
                    break
                for c in m["tool_calls"]:
                    fn = c.get("function") or {}
                    try:
                        args = fn.get("arguments")
                        args = json.loads(args) if isinstance(args, str) else args
                        code = args["code"]
                        res = sandbox.run_tests(code, task.meta["fname"], task.meta["visible"])
                        out = (f"{res['n_ok']}/{res['n']} examples passed" + (f"; error: {res['error']}" if res["error"] else "")
                               + "".join(f"\n- example {i + 1}: {'ok' if r_['ok'] else 'FAIL ' + r_.get('error', 'got ' + r_.get('got', ''))}"
                                         for i, r_ in enumerate(res["results"])))
                    except Exception as e:
                        out = f"tool error: {type(e).__name__}: {e}"
                    msgs.append({"role": "tool", "tool_call_id": c.get("id", ""), "content": out})
        except InfraError as e:
            return Episode(task.id, seed, msgs, TOOLS, None, finish, turns, ntok, time.time() - t0, infra_error=str(e))
        return Episode(task.id, seed, msgs, TOOLS, final, finish, turns, ntok, time.time() - t0)

    def verify(self, task: Task, ep: Episode) -> Verdict:
        if ep.infra_error:
            return Verdict(task.id, ep.seed, False, {}, infra_error=ep.infra_error)
        hit_cap = ep.final_text is None
        det = detectors.run_all(ep.messages, ep.final_text, ep.finish_reasons, hit_cap, TOOL_NAMES)
        code = detectors.extract_code(ep.final_text)
        res = sandbox.run_tests(code, task.meta["fname"], task.meta["hidden"]) if code else {"passed": False, "error": "no code"}
        return Verdict(task.id, ep.seed, bool(res["passed"]), det,
                       details={"n_ok": res.get("n_ok"), "n": res.get("n"), "error": res.get("error")})


def make_environment(**kwargs) -> CodeTasksEnv:
    return CodeTasksEnv(**kwargs)
