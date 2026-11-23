"""DeepMath-103K integer reasoning environment across Qwen3, Qwen3.5, and Gemma 4.

Uses pinned `zwhe99/DeepMath-103K@5cf055d1fe3d7a2eb19719ac020211469736ae44` filtered to integer answers
outside `{0, 1}` (`difficulty >= 3.0`), clustered into disjoint topic-hash families across `search`,
`select`, `exam`, and `train`, and verified deterministically via `\\boxed{INT}`.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import time
from pathlib import Path
from typing import Any

from ..client import InfraError
from .bfcl import _thinking_enabled, detect_dialect, render_chat_prompt, strip_thinking
from .protocol import Episode, Task, Verdict

DEEPMATH_REPO = "zwhe99/DeepMath-103K"
DEEPMATH_REV = "5cf055d1fe3d7a2eb19719ac020211469736ae44"
SYSTEM_PROMPT = "Solve the math problem concisely step by step and put your final integer answer inside \\boxed{...}."
INT_RE = re.compile(r"^[-+]?\d+$")
_LATEX_WRAP_RE = re.compile(
    r"\\(?:text|textbf|textit|textrm|mathrm|mathbf|mathit|boldsymbol|mbox|hbox|operatorname)\s*\{([^{}]*)\}"
)


def extract_boxed_int(text: str | None) -> int | None:
    """Extract the last `\\boxed{...}` integer from `text`, stripping LaTeX spacing/commas."""
    if not text:
        return None
    cleaned = strip_thinking(text)
    # Match nested braces inside \boxed{...}
    starts = [m.end() for m in re.finditer(r"\\boxed\s*\{", cleaned)]
    for pos in reversed(starts):
        depth, i = 1, pos
        while i < len(cleaned) and depth > 0:
            if cleaned[i] == "{":
                depth += 1
            elif cleaned[i] == "}":
                depth -= 1
            i += 1
        if depth == 0:
            inner = cleaned[pos : i - 1]
            inner = re.sub(r"\\displaystyle\b", "", inner)
            for _ in range(8):
                nxt = _LATEX_WRAP_RE.sub(r"\1", inner)
                if nxt == inner:
                    nxt = re.sub(r"(?<![A-Za-z])\{([^{}]*)\}", r"\1", inner)
                if nxt == inner:
                    break
                inner = nxt
            for sp_tok in ("\\,", "\\!", "\\;", "\\:"):
                inner = inner.replace(sp_tok, "")
            inner = inner.replace(",", "").replace(" ", "").strip("$").strip()
            inner = re.sub(r"^(?:[A-Za-z_][A-Za-z0-9_]*)?=", "", inner).strip("$").strip()
            if INT_RE.match(inner):
                try:
                    return int(inner)
                except ValueError:
                    pass
    return None


def _parse_difficulty(raw_diff: Any) -> float:
    if raw_diff is None:
        return 5.0
    if isinstance(raw_diff, bool):
        raise ValueError(f"difficulty must be numeric or None (not bool), got {raw_diff!r}")
    if isinstance(raw_diff, str) and not raw_diff.strip():
        return 5.0
    val = float(raw_diff)
    if not math.isfinite(val):
        raise ValueError(f"difficulty must be finite, got {raw_diff!r}")
    return val


def _load_or_build_deepmath(cache_dir: Path, max_items: int = 3200) -> list[dict[str, Any]]:
    cache_p = cache_dir / f"deepmath_int_{DEEPMATH_REV[:8]}_{max_items}.jsonl"
    if cache_p.exists():
        return [json.loads(line) for line in cache_p.read_text(encoding="utf-8").splitlines() if line.strip()]
    from datasets import load_dataset
    ds = load_dataset(DEEPMATH_REPO, revision=DEEPMATH_REV, split="train")
    seen_q: set[str] = set()
    by_area: dict[str, list[dict[str, Any]]] = {}
    for idx, row in enumerate(ds):
        raw_ans = row.get("final_answer")
        if raw_ans is None:
            raw_ans = row.get("answer", "")
        ans_s = str(raw_ans).strip().replace(",", "")
        if not INT_RE.match(ans_s):
            continue
        ans_i = int(ans_s)
        if ans_i in (0, 1):
            continue
        raw_diff = row.get("difficulty")
        diff = _parse_difficulty(raw_diff)
        if diff < 3.0:
            continue
        q = str(row.get("question", "")).strip()
        if not q or len(q) > 2200 or q in seen_q:
            continue
        seen_q.add(q)
        topic = str(row.get("topic", "Mathematics -> Other"))
        parts = [p.strip() for p in topic.split("->")]
        area = parts[1] if len(parts) > 1 else parts[0]
        sub = parts[2] if len(parts) > 2 else area
        qid = hashlib.sha256(f"{idx}:{q}".encode()).hexdigest()[:12]
        by_area.setdefault(area, []).append({
            "id": qid,
            "question": q,
            "answer": ans_i,
            "difficulty": diff,
            "area": area,
            "subtopic": sub,
        })
    # Deterministic stratified round-robin across areas up to `max_items`
    for area in by_area:
        by_area[area].sort(key=lambda r: hashlib.sha256(r["id"].encode()).hexdigest())
    picked: list[dict[str, Any]] = []
    areas = sorted(by_area)
    ptrs = {a: 0 for a in areas}
    while len(picked) < max_items and any(ptrs[a] < len(by_area[a]) for a in areas):
        for a in areas:
            if ptrs[a] < len(by_area[a]) and len(picked) < max_items:
                picked.append(by_area[a][ptrs[a]])
                ptrs[a] += 1
    cache_dir.mkdir(parents=True, exist_ok=True)
    tmp_p = cache_p.with_suffix(".jsonl.tmp")
    with tmp_p.open("w", encoding="utf-8") as f:
        for r in picked:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp_p.replace(cache_p)
    return picked


def _row_area_and_subtopic(row: dict[str, Any]) -> tuple[str, str]:
    if row.get("area"):
        area = str(row["area"]).strip()
        sub = str(row.get("subtopic") or area).strip()
        return area, sub
    if row.get("topic"):
        parts = [p.strip() for p in str(row["topic"]).split("->")]
        area = parts[1] if len(parts) > 1 else parts[0]
        sub = parts[2] if len(parts) > 2 else area
        return area, sub
    return "Math", "Math"


def _family_and_split(row: dict[str, Any], buckets_per_area: int = 16) -> tuple[str, str]:
    raw_area, raw_sub = _row_area_and_subtopic(row)
    area = re.sub(r"[^a-z0-9]+", "_", raw_area.lower()).strip("_") or "math"
    sub = str(row.get("subtopic") or (raw_sub if raw_sub != raw_area else area))
    b = int(hashlib.sha256(f"{area}:{sub}:{str(row['id'])[:3]}".encode()).hexdigest()[:8], 16) % buckets_per_area
    fam = f"{area}/c{b:02d}"
    h = int(hashlib.sha256(f"dm_split:20261118:{fam}".encode()).hexdigest()[:8], 16) % 20
    if h < 6:
        sp = "search"
    elif h < 12:
        sp = "select"
    elif h < 18:
        sp = "exam"
    else:
        sp = "train"
    return fam, sp


class DeepMathEnvironment:
    name = "deepmath"
    version = "1"
    tools = None

    def __init__(
        self,
        hf_dir: str,
        cache_dir: str = "/tmp/chantless-deepmath-cache",
        jsonl_file: str | None = None,
        max_items: int = 3200,
        max_tokens: int = 1024,
        context: int = 4096,
        concurrency: int = 1,
        kv_type: str = "f16",
        max_total_memory_bytes: int | None = None,
        buckets_per_area: int = 16,
        system_prompt: str = SYSTEM_PROMPT,
        reasoning_effort: str | float | None = "none",
    ):
        from transformers import AutoTokenizer
        self.hf_dir = str(hf_dir)
        self.tok = AutoTokenizer.from_pretrained(self.hf_dir)
        self.dialect, self.stop = detect_dialect(self.tok)
        self.max_tokens = int(max_tokens)
        self.system_prompt = str(system_prompt)
        self.reasoning_effort = reasoning_effort
        self.budget = {"max_turns": 1, "max_tokens_per_turn": self.max_tokens}
        self.deployment = {
            "context": int(context),
            "concurrency": int(concurrency),
            "kv_type": str(kv_type),
            "max_total_memory_bytes": max_total_memory_bytes,
        }
        if jsonl_file:
            rows = [json.loads(line) for line in Path(jsonl_file).read_text(encoding="utf-8").splitlines() if line.strip()]
        else:
            rows = _load_or_build_deepmath(Path(cache_dir), max_items=int(max_items))
        self._tasks: list[Task] = []
        for r in rows:
            raw_split = r.get("split")
            sp_s = str(raw_split).strip() if raw_split is not None else ""
            has_split = sp_s != ""
            if has_split and r.get("family") is not None:
                fam_s = str(r["family"]).strip()
                if not fam_s:
                    raise ValueError(f"deepmath row {r.get('id')!r} has blank/empty family")
                fam, sp = fam_s, sp_s
            elif has_split and r.get("group") is not None:
                fam, sp = f"g{r['group']}", sp_s
            elif has_split:
                raw_fam, _ = _family_and_split(r, buckets_per_area)
                sp = sp_s
                fam = f"{sp}/{raw_fam}"
            else:
                fam, sp = _family_and_split(r, buckets_per_area)
            area, _ = _row_area_and_subtopic(r)
            diff_val = _parse_difficulty(r.get("difficulty"))
            raw_ans = r.get("answer") if r.get("answer") is not None else r.get("final_answer", "")
            ans_val = int(str(raw_ans).replace(",", "").strip())
            msgs = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": str(r["question"])},
            ]
            self._tasks.append(Task(
                id=f"dm:{r['id']}",
                family=fam,
                split=sp,
                prompt={"messages": msgs},
                meta={"answer": ans_val, "difficulty": diff_val, "area": area},
            ))

    def render_prompt(self, task: Task) -> str:
        return render_chat_prompt(self.tok, task.prompt["messages"], self.dialect, reasoning_effort=self.reasoning_effort)

    def tasks(self) -> list[Task]:
        return list(self._tasks)

    def sample_tasks(self, split: str, n: int, *, stratified: bool = True) -> list[Task]:
        if int(n) <= 0:
            return []
        pool = [t for t in self._tasks if t.split == split]
        if not stratified:
            return sorted(pool, key=lambda t: t.id)[: int(n)]
        buckets: dict[str, list[Task]] = {}
        for t in pool:
            area = str((t.meta or {}).get("area") or "Math")
            buckets.setdefault(area, []).append(t)
        areas = sorted(buckets)
        for a in areas:
            buckets[a].sort(key=lambda t: t.id)
        out: list[Task] = []
        idx = 0
        while len(out) < int(n):
            added = False
            for a in areas:
                if idx < len(buckets[a]):
                    out.append(buckets[a][idx])
                    added = True
                    if len(out) == int(n):
                        break
            if not added:
                break
            idx += 1
        return out

    def stats_texts(self, n: int = 200) -> list[str]:
        search_ts = self.sample_tasks("search", int(n), stratified=True)
        return [self.render_prompt(t) + f"\\boxed{{{t.meta['answer']}}}" for t in search_ts]

    def run(self, task: Task, client: Any, seed: int) -> Episode:
        t0 = time.time()
        prompt_str = self.render_prompt(task)
        srv_prompt = prompt_str[len(self.tok.bos_token):] if (self.tok.bos_token and prompt_str.startswith(self.tok.bos_token)) else prompt_str
        try:
            res = client.complete(srv_prompt, seed=seed, max_tokens=self.max_tokens, stop=self.stop)
        except InfraError as e:
            return Episode(task.id, seed, list(task.prompt["messages"]), None, None, [], 0, 0, time.time() - t0, str(e))
        raw_txt = (res["text"] or "").strip()
        if _thinking_enabled(self.dialect, self.reasoning_effort):
            asst_txt = raw_txt
        else:
            clean_txt = strip_thinking(raw_txt).strip()
            asst_txt = clean_txt or raw_txt
        msgs = list(task.prompt["messages"]) + [{"role": "assistant", "content": asst_txt}]
        return Episode(
            task_id=task.id,
            seed=seed,
            messages=msgs,
            tools=None,
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
        pred = extract_boxed_int(txt)
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
        premature = any(r == "length" for r in episode.finish_reasons) or not txt.strip()
        no_boxed = pred is None
        garbage = "\ufffd" in txt
        ok = (not no_boxed) and int(pred) == int(task.meta["answer"])
        dets = {
            "doom_loop": span_rep,
            "malformed_tool_call": False,
            "premature_stop": premature,
            "token_garbage": garbage,
        }
        return Verdict(
            task.id,
            episode.seed,
            ok,
            dets,
            details={"pred": pred, "answer": task.meta["answer"], "no_boxed": no_boxed},
        )


def make_environment(**kwargs) -> DeepMathEnvironment:
    return DeepMathEnvironment(**kwargs)
