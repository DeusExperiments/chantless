# Classifier extension: interface contract

Chantless runs one search per environment. Today each episode is a pass/fail judged by a fixed rule, and the
search is steered by KL to the source. That works for generative tasks. It does not work for a classifier
judged at its own operating point: a detector flags at the cutoff that hits 1% of calibration negatives, so
the rule belongs to each model, and the decision depends on a score gap, not on the whole next-token
distribution. This contract adds that case without changing anything else.

**Rule 0: every existing config runs exactly as before.** Each new behaviour is opt-in, through an
environment flag or a config key whose default is today's behaviour. Anything an old config touches must
stay byte-for-byte the same: arguments, defaults, receipts and file names.

## Opt-in switch

An environment is a *classifier* when `getattr(env, "classifier", False)` is True. The detect environment
sets it from its `classifier` kwarg, which defaults to False, so run 002's config keeps the episode path.
Every classifier code path is guarded by this flag.

## Piece A: protocol, evaluator, batch scorer
Files: `env/protocol.py`, `evaluator.py`, `runtime.py`.

- `SPLITS` gains `"calibration"`. It is optional in `validate_environment`: search, select and exam are
  still required, and a family may not span splits.
- `runtime.Toolchain` gains `logodds: Path | None = None`, read from `toolchain.logodds` when present.
  `from_config` must still accept old configs.
- `runtime.BatchScorer(tc, slots_per_gpu: int = 2)`:
  - `score_trj(model: str, trj: str, pair: tuple[int, int], gpu: str | None = None) -> np.ndarray`. It
    returns one float per target, in file order: `logit[pair[0]] - logit[pair[1]]` at each target,
    computed by the `logodds` binary. The `--pair a,b --shard i/n --out FILE` CLI is in
    `mvp/tools/logodds.cpp`.
  - With `gpu=None` the work is sharded over every slot of every GPU in `tc.gpus`, `slots_per_gpu` per GPU,
    each with `CUDA_VISIBLE_DEVICES` set, then merged in order. With a `gpu` given, one process runs on
    that GPU.
  - A missing value is an error, never a silent NaN.
- Optional classifier hooks the evaluator calls when `env.classifier` is true:
  - `env.score(tasks: list[Task], model: str, scorer: BatchScorer) -> dict[str, float]`: a score per task
    id, where a lower score means more toward the flagged class.
  - `env.calibrate(scores: dict[str, float]) -> dict`: a JSON-serialisable rule, computed from this model's
    scores on the calibration split.
  - `env.decide(task: Task, score: float, rule: dict) -> Verdict`.
- `Evaluator(..., scorer: BatchScorer | None = None)`. A new keyword with a default of None; the existing
  positional arguments are unchanged.
- `Evaluator.run(model, jobs, tag, side, save, srvs)` on a classifier environment:
  - It starts no llama-server. `servers()` yields a placeholder list of the same length.
  - The rule for `model` is computed once. It is cached in memory and on disk at
    `log_dir/rules/<sha256 of the model file, first 16 hex>.json`, keyed by that hash.
  - Each distinct task is scored once; repeated (task, seed) jobs reuse the score.
  - The returned rows have the same keys as today's rows, so stats, gates and feedback see the same shape.
    Saved episode lines also carry `"score"` and `"rule"`. `messages`, `final` and `verdict` are present
    but may be minimal.
  - `infra_error` handling stays as it is.

## Piece B: log-odds proxy stores and objectives
Files: `prober.py`, new `objectives.py`.

- `Store` gains `kind: str = "kl"` and `meta: dict = field(default_factory=dict)`. `kind == "kl"` is
  today's behaviour, unchanged.
- `kind == "logodds"`:
  - `store.trj` is a TRJ1 file whose trajectories are the concatenation of named parts, in the order
    `meta["parts"]`, a list of `{"name", "role": "pos" | "neg", "n"}`.
  - `store.dir` holds `source.npy`, the source's score per trajectory in the same order.
  - The prober scores the candidate with `BatchScorer.score_trj(cand, trj, meta["pair"], gpu=g)` on the
    probe's GPU. It then sets `res.kl[name] = res.nll[name] = objectives.OBJECTIVES[meta["objective"]](cand_parts, src_parts, **meta.get("objective_args", {}))`,
    where `*_parts = {"pos": array, "neg": array}`.
  - The `p-<key>-<store>.json` receipt is still written, with `"kl"` and `"nll_cand"`, so `_from_disk`
    works. Raw candidate scores go to `p-<key>-<store>.npy`.
  - `Prober` gains an optional `scorer` keyword (default None). It is required only when a logodds store
    exists.
- `objectives.pauc_damage(cand, src, tail=0.05, tau=0.5) -> float` returns `pauc(src) - pauc(cand)`.
  Higher is worse, and the value can be negative. The partial AUC is
  `pauc(x) = mean over pos p and hard negatives n of sigmoid((n - p) / tau)`. The hard negatives are **the
  lowest `ceil(tail * len(x["neg"]))` of x's own negative scores**.
  - They must be chosen from the scored model's own values, never from a fixed index set. A fixed
    lowest-tail set regresses toward the middle under any perturbation and rewards noisier candidates. The
    docstring must say so.
  - The neg part must be a random sample, not tail-selected.
  - `OBJECTIVES = {"pauc": pauc_damage}`.

## Piece C: learner exchange phase and signed proxy
File: `learner.py`.

- `legal_moves(state, groups, ladder, allow_up=False)`. The default output is unchanged.
  - With `allow_up`, the moves also include one rung up: tier i to i-1, but never above `ladder[1]`, so
    BF16 is never a target.
  - A dropped group can come back at the cheapest tier `tier_allowed` permits.
- `Learner` reads `cfg.get("allow_negative_proxy", False)`. When True the move score uses the signed
  `d_proxy`; when False it stays `max(d_proxy, 0)`, as today.
- `Learner.exchange(state, base_res, budget_bytes) -> dict | None` runs one round:
  1. Probe every legal single move, with `allow_up=True`, from the current mask.
  2. Additively estimate every pair (one move that adds bytes plus one that removes bytes, on different
     groups), and every single move, that keeps `bytes <= budget_bytes`.
  3. Probe the best `cfg.get("exchange_top_k", 32)` of those for real.
  4. Accept the best one if its task proxy is lower than the base's by more than
     `cfg.get("exchange_eps", 0.0)`, and it passes the existing legality (cap).
  5. On acceptance, update `state.mask`, add both moves to `state.window` so feedback can implicate them,
     add 1 to `state.accepted`, and return the same dict shape as `step()` plus `"pair": [move_a, move_b|None]`.
     Otherwise return None.
  - It records `self.record("probes", ...)` rows like `evaluate()`.

## Piece D: sanity options and empty-safe stats
Files: `sanity.py`, `stats.py`.

- `sanity.make_cap(kind: str, points) -> callable`. `"general_kl"` returns `SizeMatchedCap(points)`, as
  today; `"off"` returns a callable that always gives `float("inf")`.
- `floors_from_solo(..., key="kl_gen")`. The new `key` argument picks which measured value the floors use
  (`"kl_gen"` today, or `"task"`), with the same rule. The default output is identical.
- `behavioral_items(0, seed)` returns `[]`, and `run_behavioral` with no items returns `[]` without
  starting threads.
- `stats.delta_and_bounds` behaviour is unchanged. Add `stats.empty_ok(b: dict, margin: float) -> bool`:
  True when `b["n_tasks"] == 0` (nothing was measured), otherwise `b["L"] > -margin`. It is a helper only;
  the pipeline decides when to use it.

## Piece E: external comparison and the plan command
Files: new `compare.py`, new `plan.py`, `cli.py`.

- `compare.compare(models, labels, kinds, groups_of=None, boot=1000, seed=0) -> dict`:
  - `models = {name: {"cal": np.ndarray of negative-class calibration scores, "scores": {doc_id: score}}}`;
    `labels = {doc_id: "pos" | "neg"}`; `kinds = {doc_id: str}`.
  - Each model's cutoff is `sorted(cal)[int(0.01 * len(cal))]`, and a document is flagged when its score is
    strictly below the cutoff. This is the same rule as `mvp/detect/measure.py`.
  - The comparison uses only documents every model scored. It reports per model the negatives-flagged
    rate, plus the positives caught overall and per kind, with 2.5–97.5% intervals. The intervals use the
    **same bootstrap resamples of documents for every model**.
  - `compare.beats(res, a, b) -> bool` is True when a's caught interval lies wholly above b's.
  - It must reproduce `measure.py`'s numbers when given the same inputs, which is the test.
- `plan.estimate(cfg: dict, rates: dict | None = None) -> dict` estimates the work of a config without
  running it:
  - from the groups and ladder in the GGUF header: legal moves per sweep, and moves to reach `target_bytes`;
  - from the pool, A/A, screen, gate and exam settings: episodes;
  - tokens, from the environment's task token counts when `env.token_counts()` exists, otherwise a
    stated assumption;
  - wall-clock and cost on named backends (laptop M2 Pro, 1×H100, 8×H100), from a rates table in
    `plan.py` that `rates` can override.
  - Every assumption is printed.
- `cli.py` gains `plan --config X [--rates R.json]`, which prints a table. `run` is unchanged.

## Piece F: detect environment, classifier mode
File: `env/detect.py`.

- The new kwarg `classifier: bool = False` sets `self.classifier`. With False, nothing changes: the same
  tasks, the same `run`/`verify`, the same receipts.
- With True:
  - **Calibration split:** the 20,000 calibration humans, by id from `calibration_ids_file` (a kwarg; the
    ids from the source's earlier score files), from `WildOutlines/data/calibration-00000-of-00001.parquet`.
  - **`score`:** documents are tokenised exactly as `mvp/detect/climb_prep.py` does (ProseLens prompt, a
    4,096-token head, the answer token appended as the single target). The TRJ1 files are cached under a
    `cache_dir` kwarg, keyed by a hash of the task ids. Scores come from `scorer.score_trj(model, trj, (24509, 2075))`.
  - **`calibrate`:** returns `{"cutoff": sorted(cal)[int(0.01 * n)], "n": n}`.
  - **`decide`:** flags when `score < cutoff`; success means flagged == (label == "ai").
  - **`proxy_spec(source_scores)`** returns the parts for the log-odds store: a random sample of `n_neg`
    calibration humans (seed fixed), plus the `n_pos` AI tasks of the search split whose source scores are
    closest to the source cutoff. Defaults: `n_neg=1000`, `n_pos=600`.
  - **`stats_texts(n)`:** raw prompt strings for the importance matrix.
  - **`token_counts()`:** token counts per split.
  - **Splits:** search and select are drawn from WildOutlines validation (minus the 2,000 loss-curve
    documents) plus Beemo's 30% search split by prompt (`sha256(prompt_id) % 10 < 3`). The exam is
    WildOutlines test plus Beemo's other 70%. Sizes come from kwargs. Families are
    split/source/format/kind, as today.

## Piece G: learning from labelled examples
Files: new `lattice.py`, new `learn.py`, `cli.py`.

- `cli learn` is a new command, so `run` and `plan` and every existing config are untouched.
- Input is a built GGUF. Its structure (removed sub-blocks, cut neurons, trimmed vocabulary) is read from the
  file. Only levels on Q4_K and Q8_0 grids move; scales, other tiers and the header stay byte-identical.
- Select is split by task id: side 0 may be replayed in training, side 1 alone judges. Feedback folds are held
  out by `--fold`. The result is a preview, not an exam claim.

## Ownership
The pipeline wiring (`pipeline.py`), integration and golden checks belong to the integrator. No piece edits
a file it does not own.
