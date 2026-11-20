# Classifier Mode Interface

Chantless supports calibrated single-token classifier environments alongside generative multi-token tasks through opt-in `ClassifierHooks` on an environment (`getattr(env, "classifier", False) is True`).

## Environment Protocol (`env/protocol.py`, `evaluator.py`, `runtime.py`)

- `SPLITS` includes optional `"calibration"` alongside `"search"`, `"select"`, `"exam"`, and `"train"`. Families must not cross splits.
- `runtime.Toolchain` reads optional `toolchain.logodds` for fast batched logit scoring via `mvp/tools/logodds.cpp`.
- `runtime.BatchScorer(tc, slots_per_gpu: int = 2)`:
  - `score_trj(model: str, trj: str, pair: tuple[int, int], gpu: str | None = None) -> np.ndarray` returns `logit[pair[0]] - logit[pair[1]]` per target across `TRJ1` trajectories, sharded cleanly over available GPUs when `gpu=None`.
  - Non-finite or missing scores raise immediately; silent NaNs are forbidden.
- When `env.classifier` is `True`, `Evaluator.run(...)` invokes:
  - `env.score(tasks, model, scorer) -> dict[str, float]` (lower score = stronger signal toward flagged class).
  - `env.calibrate(scores) -> dict` to compute the JSON-serializable cutoff rule on `"calibration"`.
  - `env.decide(task, score, rule) -> Verdict`.
- Calibration rules are cached in memory and on disk keyed by the built GGUF's SHA256 prefix, and each distinct task is scored once per evaluation.

## Log-Odds Proxy Stores & Partial AUC Damage (`prober.py`, `objectives.py`)

- `Store(kind="logodds", meta=...)` pairs a concatenated `TRJ1` file (`meta["parts"]` specifying `{"name", "role": "pos" | "neg", "n"}`) with `source.npy` reference scores.
- `Prober` scores each candidate via `BatchScorer.score_trj` and evaluates `objectives.OBJECTIVES[meta["objective"]](cand_parts, src_parts, ...)`.
- `objectives.pauc_damage(cand, src, tail=0.05, tau=0.5) -> float` returns `pauc(src) - pauc(cand)` (higher is worse; negative indicates better separation than source), where `pauc(x)` averages `sigmoid((n - p) / tau)` over positives `p` and the hard-negative tail `n`.
  - **Statistical invariant:** Hard negatives MUST be selected dynamically as the lowest `ceil(tail * len(x["neg"]))` of the **scored model's own negative values**, never from a frozen source index set, and `x["neg"]` must be randomly sampled rather than pre-filtered by tail. A fixed tail regresses toward the mean under perturbation and rewards noisy candidates.

## Search, Exchange, Sanity & Planning (`learner.py`, `sanity.py`, `stats.py`, `compare.py`, `plan.py`)

- `learner.legal_moves(..., allow_up=False)`: with `allow_up=True`, includes single-rung upgrades up to `ladder[1]` and re-materializing dropped groups at their cheapest allowed tier.
- `Learner` supports signed proxy deltas (`allow_negative_proxy: true`) and `Learner.exchange(state, base_res, budget_bytes)` to trade an upgrade on one group against a downgrade on another within budget.
- `sanity.make_cap(kind, points)` selects `SizeMatchedCap(points)` (`"general_kl"`) or an unconstrained cap (`"off"`); `floors_from_solo(..., key=...)` computes floors from either `"kl_gen"` or `"task"`.
- `stats.empty_ok(b, margin)` returns `True` when `n_tasks == 0` and checks `b["L"] > -margin` otherwise.
- `compare.compare(models, labels, kinds, ...)` evaluates models strictly on the intersection of scored document IDs, computing each model's 1% false-positive calibration cutoff and paired bootstrap confidence intervals using identical resamples across all models.
- `plan.estimate(cfg, rates=None, source=None)` estimates moves, probes, episodes, tokens, wall-clock, and cost from GGUF headers and environment token counts (`chantless plan`).

## Classifier Mode on `detect` (`env/detect.py`, `env/clsbench.py`)

- Both `detect` (`classifier=True`) and `clsbench` implement `ClassifierHooks`, automatic `tokdocs` re-tokenization (with HF fallback) and answer-token ID remapping on vocabulary-trimmed GGUFs, calibration at the lowest 1% of calibration negatives, and `proxy_spec` / `fit_token_counts` / `token_counts` hooks.

## Few-Shot Switch & Grid Adaptation (`learn.py`, `lattice.py`, `cli.py`)

- `chantless learn` reconstructs any built GGUF (`IQ1_S..BF16`) as an exact PyTorch student, optimizing either universal `+0 B` binary `up_row_zero` MLP switches or sparse `Q4_K`/`Q8_0` quantization-grid updates, with strict `select` half-splitting (`side 0` replay / `side 1` evaluation, fold isolation, and `base`/`empty`/`flipped`/`shuffled` controls).
