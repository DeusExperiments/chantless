from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .mask import DROP, LADDER

VALID_TIERS = set(LADDER) | {DROP}


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class SourceConfig(StrictModel):
    gguf: str
    repo: str
    revision: str
    hf_dir: str | None = None
    expected_gguf_sha256: str | None = None


class ToolchainConfig(StrictModel):
    llama_bin: str
    traj_kl: str
    gpus: list[str] = Field(default_factory=lambda: [""])
    logodds: str | None = None
    tokdocs: str | None = None
    convert_py: str | None = None

    @field_validator("gpus", mode="before")
    @classmethod
    def _coerce_gpus(cls, v: Any) -> list[str]:
        if v is None:
            return [""]
        if isinstance(v, (str, int)):
            return [str(v)]
        return [str(x) for x in v]


class EnvConfig(StrictModel):
    module: str
    kwargs: dict[str, Any] = Field(default_factory=dict)


class DecodingConfig(StrictModel):
    temperature: float = 0.0
    top_p: float | None = None
    top_k: int | None = None
    min_p: float | None = None
    max_tokens: int | None = None
    seed: int | None = None


class ServingConfig(StrictModel):
    slots_per_gpu: int = Field(default=4, ge=1)
    ctx_per_slot: int = Field(default=8192, ge=128)


class PoolConfig(StrictModel):
    tasks_per_split: dict[str, int] = Field(default_factory=dict)
    runs: int = Field(default=1, ge=1)
    min_pass: int = Field(default=1, ge=0)


class StatsConfig(StrictModel):
    margin: float = 0.02
    alpha: float = 0.05
    power: float = 0.90
    n_floor: int = Field(default=20, ge=1)
    looks: int = Field(default=4, ge=1)
    max_eps_per_task: int = Field(default=1, ge=1)
    aa_tol: float = 0.02
    aa_enforce: bool = True
    aa_replication: bool = True
    detector_bound: float = 0.03
    n_aa: int = Field(default=20, ge=1)
    n_override: int | None = None
    v_min: float = 0.05
    allow_small_pool: bool = False
    reuse_source: bool = True


class ProxyConfig(StrictModel):
    objective: str | None = None
    objective_args: dict[str, Any] = Field(default_factory=dict)
    spec: dict[str, Any] | None = None
    stats_docs: int = Field(default=200, ge=1)
    general_docs: int = Field(default=30, ge=0)
    general_seq_len: int = Field(default=1024, ge=1)
    general_n_seq: int = Field(default=4, ge=0)
    task_max_targets: int | None = None
    floor_tier: str = "Q4_K"
    self_kl_tol: float | dict[str, float] = 1e-3

    @field_validator("floor_tier")
    @classmethod
    def _check_tier(cls, v: str) -> str:
        if v not in VALID_TIERS:
            raise ValueError(f"unknown tier {v!r}")
        return v


class SanityConfig(StrictModel):
    floor_factor: float = 4.0
    behavioral_items: int = Field(default=0, ge=0)
    behavioral_seeds: int = Field(default=1, ge=1)
    behavioral_margin: float = 0.1
    cap: str | float | bool | None = "off"
    floor_key: str = "task"
    solo_tiers: list[str] | None = None
    solo_task_tiers: list[str] | None = None
    splice_cache: bool | None = None
    allow_drop: bool = True
    enforce_general_cap: bool = True

    @field_validator("solo_tiers")
    @classmethod
    def _check_solo_tiers(cls, v: list[str] | None) -> list[str] | None:
        if v is not None:
            for t in v:
                if t not in VALID_TIERS:
                    raise ValueError(f"unknown solo tier {t!r}")
        return v


class ScreenConfig(StrictModel):
    pairs: int = Field(default=32, ge=1)
    fail_below: float = -0.05


class LearnerConfig(StrictModel):
    algo: Literal["greedy", "oneshot"] = "greedy"
    reprobe_every: int = Field(default=16, ge=1)
    top_k: int = Field(default=16, ge=1)
    screen_every: int = Field(default=4, ge=1)
    gate_every: int = Field(default=16, ge=1)
    reprobe_after_screen: bool = False
    max_moves: int = Field(default=200, ge=0)
    target_bytes: int | None = Field(default=None, ge=1)
    feedback: bool = True
    feedback_tasks: int = Field(default=200, ge=0)
    implicate_share: float = 0.2
    penalty_step: float = 1.0
    max_revert: int = Field(default=2, ge=0)
    exchange_rounds: int = Field(default=0, ge=0)
    allow_negative_proxy: bool = True
    exchange_eps: float = 1e-4
    exchange_top_k: int = Field(default=16, ge=1)
    oneshot_objective: str | None = None
    oneshot_tiers: list[str] | None = None
    group_tiers: dict[str, list[str]] | None = None
    respect_floors: bool = True
    widths: bool = False
    width_fractions: list[float] | None = None
    width_step: int | None = None
    vocab_k: int | None = None
    vocab_candidates: list[int] | None = None
    vocab_min_fidelity: float | None = None


class RepairConfig(StrictModel):
    enabled: bool = False
    lr: float = 1e-4
    rank: int = 8
    alpha: float = 16.0
    steps: int = 50
    seed: int = 20261012
    repair_without_drops: bool = False


class ExternalBaselineConfig(StrictModel):
    name: str
    scores: str


class ExamConfig(StrictModel):
    baseline_eps_per_task: int = Field(default=1, ge=1)
    boot: int = Field(default=1000, ge=1)
    external: list[ExternalBaselineConfig] = Field(default_factory=list)
    source_scores: str | None = None


class BudgetConfig(StrictModel):
    max_hours: float = Field(default=4.0, gt=0)


class ProbeConfig(StrictModel):
    workers: int = Field(default=4, ge=1)
    slots_per_gpu: int = Field(default=2, ge=1)
    quantize_threads: int = Field(default=8, ge=1)
    max_pruned: int = Field(default=64, ge=0)
    auto_shm: bool = False
    stage_source: bool | Literal["auto"] = "auto"
    clamp_workers: bool = True
    splice_cache: bool = True
    tier_cache_dir: str | None = None
    want_solo: list[str] | None = None
    solo_task_tiers: list[str] | None = None


class ScoringSeedFileConfig(StrictModel):
    prefix: str
    npy: str
    ids: str


class ScoringSeedConfig(StrictModel):
    check: int = Field(default=64, ge=1)
    files: list[ScoringSeedFileConfig]


class ScoringConfig(StrictModel):
    slots_per_gpu: int = Field(default=2, ge=1)
    seed: ScoringSeedConfig | None = None


class ExtractConfig(StrictModel):
    run_id: str
    run_root: str
    source: SourceConfig
    toolchain: ToolchainConfig
    env: EnvConfig
    decoding: DecodingConfig = Field(default_factory=DecodingConfig)
    serving: ServingConfig = Field(default_factory=ServingConfig)
    objective_mode: Literal["preserve", "improve"] = "preserve"
    ladder: list[str] = Field(default_factory=lambda: list(LADDER))
    start_tier: str = "Q4_K"
    pool: PoolConfig = Field(default_factory=PoolConfig)
    stats: StatsConfig = Field(default_factory=StatsConfig)
    proxy: ProxyConfig = Field(default_factory=ProxyConfig)
    sanity: SanityConfig = Field(default_factory=SanityConfig)
    screen: ScreenConfig = Field(default_factory=ScreenConfig)
    learner: LearnerConfig = Field(default_factory=LearnerConfig)
    repair: RepairConfig = Field(default_factory=RepairConfig)
    exam: ExamConfig = Field(default_factory=ExamConfig)
    budget: BudgetConfig = Field(default_factory=BudgetConfig)
    probe: ProbeConfig = Field(default_factory=ProbeConfig)
    scoring: ScoringConfig | None = None
    scratch_root: str | None = None

    @model_validator(mode="after")
    def _validate_tiers(self) -> "ExtractConfig":
        for t in self.ladder:
            if t not in VALID_TIERS:
                raise ValueError(f"ladder contains unknown tier {t!r}")
        if self.start_tier not in self.ladder:
            raise ValueError(f"start_tier {self.start_tier!r} must be in ladder {self.ladder}")
        return self

    def to_raw_dict(self) -> dict[str, Any]:
        return self.model_dump(exclude_none=True)


class LearnSpecConfig(StrictModel):
    model: str
    config_dir: str
    out_dir: str
    mode: Literal["classifier", "generative"] | None = None
    method: Literal["grid", "gates", "gates_seq"] | None = None
    task_kind: Literal["classifier", "generative"] | None = None
    switch_mode: Literal["up_row_zero", "down_grid"] = "up_row_zero"
    ggml_lib: str | None = None
    source_gguf: str | None = None
    imatrix: str | None = None
    mask_in: str | None = None
    mask_out: str | None = None
    pair: list[int] | None = None
    cutoff: float | None = None
    n_calibration: int | None = None
    pool_trj: str | None = None
    select_rows: str | None = None
    feedback_trj: str | None = None
    feedback_rows: str | None = None
    lessons: str | list[dict[str, Any]] | None = None
    replay: str | list[Any] | None = None
    max_lessons: int = Field(default=25, ge=1)
    heldout_ids: list[str] | None = None
    select_limit: int | None = None
    gate_ref_model: str | None = None
    controls: bool = True
    alpha: float = 0.05
    gate_margin: float = 0.03
    last_blocks: int | None = None
    gate_last_blocks: int | None = None
    train_regex: str | None = None
    lr: float = 2e-5
    gate_lr: float = 0.03
    gate_init: float = 1.0
    adam_eps: float = 2e-4
    steps: int = Field(default=80, ge=1)
    feedback_batch: int | None = None
    replay_batch: int | None = None
    score_budget: int = 40000
    train_budget: int | None = None
    pad_multiple: int = 32
    margin: float = 1.0
    pair_margin: float = 2.0
    pair_weight: float = 1.0
    nll_weight: float = 1.0
    margin_weight: float = 1.0
    replay_weight: float = 1.0
    guard_weight: float = 0.0
    hold_kind: str | None = None
    kl_temp: float = 1.0
    keep_tau: float = 0.5
    keep_window: int = 64
    keep_abs: bool = False
    batch_pairs: float | None = None
    sep_humans: str = "human"
    z_units: dict[str, float] | None = None
    project: bool = True
    seed: int = 20261031
    amp: bool = True
    checkpointing: bool | None = None
    frozen_bf16: bool | None = None
    prefix_cache: bool = True
    prefix_cache_max_mb: float = 4096.0
    free_prefix: bool = True
    stream_prefix: bool | Literal["auto"] = "auto"
    replay_humans: int = 1000
    replay_select: int = 1000
    stratify: bool = False
    tails: bool = False
    tail_n: int = 500
    near_n: int = 300
    slots_per_gpu: int = 2
    max_human_rise: float = 0.01
    max_ai_drop: float = 0.02
    side_seed: int | None = None


class LearnConfig(StrictModel):
    toolchain: ToolchainConfig
    learn: LearnSpecConfig
    env: EnvConfig | None = None
    serving: ServingConfig | None = None
    decoding: DecodingConfig | None = None

    def to_raw_dict(self) -> dict[str, Any]:
        return self.model_dump(exclude_none=True)


class AnnealConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    toolchain: ToolchainConfig
    anneal: dict[str, Any]

    def to_raw_dict(self) -> dict[str, Any]:
        return self.model_dump(exclude_none=True)


class LauncherSpecConfig(StrictModel):
    backend: str = "local"
    gpus: list[str] | None = None
    gpus_per_cell: int = Field(default=1, ge=1)
    max_parallel: int = Field(default=1, ge=1)
    hosts: list[str] = Field(default_factory=list)
    remote_workdir: str | None = None
    remote_python: str = "python3"
    modal_app: str | None = None
    modal_gpu: str | None = None
    modal_volume: str | None = None
    colab_sessions: list[str] = Field(default_factory=list)
    extra_args: list[str] = Field(default_factory=list)

    @field_validator("gpus", mode="before")
    @classmethod
    def _coerce_gpus(cls, v: Any) -> list[str] | None:
        if v is None:
            return None
        if isinstance(v, (str, int)):
            return [str(v)]
        return [str(x) for x in v]


class SweepCellConfig(StrictModel):
    id: str
    command: Literal["extract", "learn"] | None = None
    upto: str | None = None
    fold: int | None = None
    overrides: dict[str, Any] = Field(default_factory=dict)


class SweepConfig(StrictModel):
    sweep_id: str
    command: Literal["extract", "learn"] = "extract"
    export_dir: str
    base_config: str | None = None
    base: dict[str, Any] = Field(default_factory=dict)
    launcher: LauncherSpecConfig = Field(default_factory=LauncherSpecConfig)
    resume: bool = True
    stop_on_failure: bool = False
    cells: list[SweepCellConfig] = Field(min_length=1)

    @model_validator(mode="after")
    def _unique_cell_ids(self) -> "SweepConfig":
        seen: set[str] = set()
        for c in self.cells:
            if c.id in seen:
                raise ValueError(f"duplicate sweep cell id {c.id!r}")
            seen.add(c.id)
        return self

    def to_raw_dict(self) -> dict[str, Any]:
        return self.model_dump(exclude_none=True)


SCHEMA_MODELS: dict[str, type[BaseModel]] = {
    "extract": ExtractConfig,
    "run": ExtractConfig,
    "plan": ExtractConfig,
    "learn": LearnConfig,
    "sweep": SweepConfig,
    "anneal": AnnealConfig,
}


def deep_merge(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in patch.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _read_mapping(path: Path) -> dict[str, Any]:
    txt = path.read_text(encoding="utf-8")
    if path.suffix.lower() in (".yaml", ".yml"):
        data = yaml.safe_load(txt)
    else:
        data = json.loads(txt)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: top-level config must be a mapping")
    return data


def load_raw_with_extends(path: str | Path, _seen: set[Path] | None = None) -> dict[str, Any]:
    p = Path(path).resolve()
    seen = set() if _seen is None else set(_seen)
    if p in seen:
        raise ValueError(f"cyclic extends chain involving {p}")
    seen.add(p)
    raw = _read_mapping(p)
    ext = raw.pop("extends", None)
    if ext is None:
        return raw
    parents = [ext] if isinstance(ext, str) else list(ext)
    merged: dict[str, Any] = {}
    for parent in parents:
        pp = Path(parent)
        if not pp.is_absolute():
            pp = (p.parent / pp).resolve()
        merged = deep_merge(merged, load_raw_with_extends(pp, seen))
    return deep_merge(merged, raw)


def _parse_override_value(raw_val: str) -> Any:
    try:
        return yaml.safe_load(raw_val)
    except yaml.YAMLError:
        return raw_val


def apply_overrides(data: dict[str, Any], overrides: list[str] | None) -> dict[str, Any]:
    if not overrides:
        return data
    out = copy.deepcopy(data)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"invalid --set override {item!r}: expected KEY=VALUE")
        key_path, raw_val = item.split("=", 1)
        parts = [p for p in key_path.strip().split(".") if p]
        if not parts:
            raise ValueError(f"invalid --set override {item!r}: empty key")
        cur = out
        for part in parts[:-1]:
            nxt = cur.get(part)
            if not isinstance(nxt, dict):
                nxt = {}
                cur[part] = nxt
            cur = nxt
        cur[parts[-1]] = _parse_override_value(raw_val)
    return out


def validate_dict(data: dict[str, Any], kind: str) -> BaseModel:
    if kind not in SCHEMA_MODELS:
        raise ValueError(f"unknown config kind {kind!r}; expected one of {sorted(SCHEMA_MODELS)}")
    return SCHEMA_MODELS[kind].model_validate(data)


def load_config(path: str | Path, kind: str, overrides: list[str] | None = None) -> BaseModel:
    raw = load_raw_with_extends(path)
    raw = apply_overrides(raw, overrides)
    return validate_dict(raw, kind)


def export_json_schema(kind: str = "all") -> dict[str, Any]:
    if kind == "all":
        return {k: SCHEMA_MODELS[k].model_json_schema() for k in ("extract", "learn", "sweep")}
    if kind not in SCHEMA_MODELS:
        raise ValueError(f"unknown schema kind {kind!r}")
    return SCHEMA_MODELS[kind].model_json_schema()
