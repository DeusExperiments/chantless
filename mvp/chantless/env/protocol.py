"""Environment contract.

One entry point supplies tasks, a run function and a verifier, and declares the budget,
deployment conditions and split membership. Infrastructure errors are retried and never
counted as task failures.
"""
from __future__ import annotations

import importlib
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol, runtime_checkable

SPLITS = ("search", "select", "exam", "train", "calibration")  # calibration: optional, for classifier rules


@dataclass
class Task:
    id: str
    family: str          # cluster key; splits are assigned per family (no family shared across splits)
    split: str
    prompt: dict          # environment-defined prompt payload
    meta: dict = field(default_factory=dict)  # verifier-only data (hidden tests); never shown to the model


@dataclass
class Episode:
    task_id: str
    seed: int
    messages: list[dict]
    tools: list[dict] | None
    final_text: str | None
    finish_reasons: list[str]
    n_turns: int
    completion_tokens: int
    wall_s: float
    infra_error: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Verdict:
    task_id: str
    seed: int
    tests_passed: bool
    detectors: dict[str, bool]
    infra_error: str | None = None
    details: dict = field(default_factory=dict)

    @property
    def success(self) -> bool:
        """Episode success = verifier pass and no gating detector fired."""
        return self.tests_passed and not any(self.detectors.values())

    def to_dict(self) -> dict:
        d = asdict(self)
        d["success"] = self.success
        return d


@runtime_checkable
class Environment(Protocol):
    name: str
    version: str
    budget: dict        # e.g. {"max_turns": 3, "max_tokens_per_turn": 768}
    deployment: dict    # e.g. {"context": 8192, "concurrency": 1, "kv_type": "f16", "max_total_memory_bytes": ...}

    def tasks(self) -> list[Task]: ...
    def run(self, task: Task, client: Any, seed: int) -> Episode: ...
    def verify(self, task: Task, episode: Episode) -> Verdict: ...


@runtime_checkable
class ClassifierHooks(Protocol):
    """Optional hooks of an environment with `classifier = True`, judged at each model's own operating point.

    score: a score per task id (lower = more toward the flagged class); calibrate: a JSON-serialisable rule
    from this model's scores on the calibration split; decide: the verdict for one task under that rule.
    """
    classifier: bool

    def score(self, tasks: list[Task], model: str, scorer: Any) -> dict[str, float]: ...
    def calibrate(self, scores: dict[str, float]) -> dict: ...
    def decide(self, task: Task, score: float, rule: dict) -> Verdict: ...


def load_environment(module: str, **kwargs) -> Environment:
    mod = importlib.import_module(module)
    env = mod.make_environment(**kwargs)
    validate_environment(env)
    return env


def validate_environment(env: Any) -> None:
    """Conformance checks every environment must pass before a run may start."""
    for attr in ("name", "version", "budget", "deployment"):
        if not hasattr(env, attr):
            raise ValueError(f"environment lacks {attr!r}")
    for fn in ("tasks", "run", "verify"):
        if not callable(getattr(env, fn, None)):
            raise ValueError(f"environment lacks callable {fn}()")
    if getattr(env, "classifier", False):
        for fn in ("score", "calibrate", "decide"):
            if not callable(getattr(env, fn, None)):
                raise ValueError(f"classifier environment lacks callable {fn}()")
    for k in ("context", "concurrency", "kv_type"):
        if k not in env.deployment:
            raise ValueError(f"environment deployment block lacks {k!r}")
    tasks = env.tasks()
    if not tasks:
        raise ValueError("environment declares no tasks")
    seen, fam_split = set(), {}
    for t in tasks:
        if t.split not in SPLITS:
            raise ValueError(f"task {t.id} has no valid split ({t.split!r})")
        if t.id in seen:
            raise ValueError(f"duplicate task id {t.id}")
        seen.add(t.id)
        if fam_split.setdefault(t.family, t.split) != t.split:
            raise ValueError(f"family {t.family} spans splits {fam_split[t.family]} and {t.split}")
    for s in ("search", "select", "exam"):
        if not any(t.split == s for t in tasks):
            raise ValueError(f"no tasks in required split {s!r}")
