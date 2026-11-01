# Chantless MVP

This is a small but complete implementation of the tuned Chantless spec (`../explainer/design-notes.md`, decisions in `.local/architecture-implementation-planning-20261012/DECISIONS.md`). It runs on a deliberately small target so the algorithm and infrastructure can be frozen before porting to MiMo.

- **Source model.** `Qwen/Qwen3-0.6B` @ `c1899de2…`, converted to BF16 GGUF. The GGUF sha256 is `fd29a1cc…` when built with the pinned converter.
- **Runtime.** llama.cpp `9adc7f4` plus `patches/llama-cpp-9adc7f4-subblock-drop.patch` (`qwen3` and `qwen35`). The patch lets the runtime run a model whose token-mixer or FFN sub-blocks are physically absent. An absent sub-block runs as identity; an absent full-attention block allocates no KV cache and an absent gated delta net no recurrent state. The MVP run used the qwen3-only subset, kept as `patches/llama-cpp-9adc7f4-qwen3-subblock-drop.patch`.
- **Environment.** `chantless.env.codetasks`: 42 families of procedurally generated Python function tasks, graded by hidden unit tests. The model has a `run_tests` tool turn. Splits are assigned per family (search / select / exam / train).

## How each architecture box is implemented

| Spec box | Module | What it does |
|---|---|---|
| Environment contract (§4, S12) | `env/protocol.py`, `env/codetasks.py` | One entry point supplies tasks, run and verify, plus the budget and deployment block. Every task has a split. Infra errors are retried and excluded, never counted as failures |
| Detectors (§7) | `detectors.py` | Doom loop, malformed tool call, premature stop, token garbage. All four gate episode success |
| Data guardrails (§4) | `pipeline.SplitGuard` | The exam split cannot be read before the exam phase. Reliability is screened on the source only |
| Sanity suite + initial mask (§5, S5) | `sanity.py`, `pipeline.phase_sanity` | FineWeb-Edu + stdlib code KL store, plus executable behavioral checks. Uniform-ladder size-matched cap. Per-group solo-application floors give `sanity_mask.json` |
| Mask schema (§3) | `mask.py` | `chantless.mask/v2`, which can also record per-tensor entries. Reads `onlyskills.mask/v1` without rewriting it |
| Precision + removal operators (§3) | `builder.py`, `gguf_ops.py` | Removes dropped groups from the GGUF, then runs `llama-quantize` with explicit per-tensor types. Audits the header against the mask |
| Actual-byte objective (§9, S1) | `builder.BuildResult` | Objective is the built file's size. Zeros earn nothing. KV cache is a deployment setting, checked as a constraint at freeze |
| Reference proxy (§7, S8) | `kl.py`, `prober.py`, `tools/traj-kl` | Reference log-probs from the frozen source are computed once and reused. KL guides the search and never decides acceptance. Improve mode uses NLL on verified-successful trajectories instead |
| Learner with failure feedback (§2, S4) | `learner.py` | Moves are ranked by proxy cost per byte saved, re-probed on the current allocation. Each screen runs search-split feedback episodes; their failures become a new reference store. Implicated moves are re-measured by reverting them one at a time, then penalised and, if the screen failed, reverted and blocked. `feedback: false` is the proxy-only ablation |
| Noisy acceptance (§9, S9) | `stats.py`, `pipeline.gate` | Paired, task-matched test. Cluster-robust (family) t interval, 4 sequential looks with Bonferroni α. Sample size comes from the measured A/A run. Detector bounds and behavioral non-inferiority are also checked |
| Optional weight repair (§6, S3) | `repair.py` | LoRA under the exact removal structure: KL to the frozen source in preserve mode, cross-entropy in improve mode. Merged, converted to GGUF, rebuilt by the same builder, re-measured, accepted only if the screen passes |
| Learning from labelled examples (S3, classifier) | `learn.py`, `lattice.py` | `cli learn` rebuilds a built GGUF as a PyTorch student, moves Q4_K/Q8_0 levels on their fixed grids from labelled feedback, writes them back at the same byte size, and judges the real file. Accepted only if the gate side of select passes, Beemo humans and AI caught per population stay within tolerance, and held-out feedback improves |
| Freeze + reproduction (§11) | `pipeline.phase_freeze` | Rebuild sha256 must be equal. Header must match the mask, and dropped tensors must be absent. Runtime memory is measured at the declared context |
| Single-use exam + report | `pipeline.phase_exam`, `phase_report` | Exam reliability pool, then fresh source and specialist episodes, the claim test, and descriptive uniform / dynquant-K baselines at matched bytes |

## Run

```bash
python -m chantless.cli run --config configs/local-smoke.json   # tiny integration run on a laptop
python -m chantless.cli run --config configs/box.json           # full MVP run on 8 GPUs
```

Phases resume from `runs/<id>/phases/*.done`. Run `--reset PHASE` to redo a phase and all later ones.

## Known limitations (by design for the MVP)

- Weight repair trains without simulating quantisation. The loop measures the real built artifact, and repair is accepted only if it helps.
- The dynquant baseline is our K94-style recipe re-implemented on this model, measured once on the same proxy. It is not the original Qwen3.8 pipeline.
- Speedups, such as a tensor-variant cache and cached move rankings beyond the top-16 schedule, are deliberately absent. Every probe requantises the whole model.
- Removal operators exist for `qwen3` and `qwen35`. The `qwen35` path compiles but has not been run on a real model. MTP heads and the vision encoder are left out of the converted GGUF.
