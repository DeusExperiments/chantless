# Chantless MVP

This is a small but complete implementation of the tuned Chantless spec (`../explainer/design-notes.md`, decisions in `.local/architecture-implementation-planning-20261012/DECISIONS.md`). It runs on a deliberately small target so the algorithm and infrastructure can be frozen before porting to MiMo.

- **Source model.** `Qwen/Qwen3-0.6B` @ `c1899de2…`, converted to BF16 GGUF. The GGUF sha256 is `fd29a1cc…` when built with the pinned converter.
- **Runtime.** Unified on llama.cpp `a01dff9` plus `patches/llama-cpp-a01dff9-chantless.patch` (`qwen3`, `qwen35`, and `gemma4`, with per-layer `hparams.n_ff(il)` and `hparams.attn_dropped`). The patch lets the runtime run a model whose token-mixer or FFN sub-blocks are physically absent or narrowed per layer. An absent sub-block runs as identity; an absent full-attention block allocates no KV cache and an absent gated delta net no recurrent state. Retained historical MVP patches (`patches/llama-cpp-9adc7f4-subblock-drop.patch`, `patches/llama-cpp-9adc7f4-qwen3-subblock-drop.patch`, and `patches/llama-cpp-a01dff9-gemma4-subblock-removal.patch`) remain alongside for provenance.
- **Environment.** `chantless.env.codetasks`: 42 families of procedurally generated Python function tasks, graded by hidden unit tests. The model has a `run_tests` tool turn. Splits are assigned per family (search / select / exam / train).

- **Gemma 4 E2B.** llama.cpp `a01dff9` plus `patches/llama-cpp-a01dff9-gemma4-subblock-removal.patch`; the 10% mask and Learned Apprentice scripts are in `gemma4/` (see `gemma4/README.md`).

## How each architecture box is implemented

| Spec box | Module | What it does |
|---|---|---|
| Environment contract (§4, S12) | `env/protocol.py`, `env/codetasks.py` | One entry point supplies tasks, run and verify, plus the budget and deployment block. Every task has a split. Infra errors are retried and excluded, never counted as failures |
| Detectors (§7) | `detectors.py` | Doom loop, malformed tool call, premature stop, token garbage. All four gate episode success |
| Data guardrails (§4) | `pipeline.SplitGuard` | The exam split cannot be read before the exam phase. Reliability is screened on the source only |
| Sanity suite + initial mask (§5, S5) | `sanity.py`, `pipeline.phase_sanity` | FineWeb-Edu + stdlib code KL store, plus executable behavioral checks. Uniform-ladder size-matched cap. Per-group solo-application floors give `sanity_mask.json` |
| Mask schema (§3) | `mask.py` | `chantless.mask/v3` (thin mask recording tiers/drops, `dead_tensors`, Wanda `widths` bitmaps, Rule-P `vocab_keep`, and `apprentice` switches/levels/decision). Backward-compatible with `chantless.mask/v2` and `onlyskills.mask/v1` |
| Precision + removal operators (§3) | `builder.py`, `gguf_ops.py`, `vocab.py` | Prunes dead/dropped tensors, slices MLP widths and vocabulary in BF16, shards `llama-quantize` with explicit per-tensor types, applies Apprentice switches/levels, and audits the header against the mask |
| Actual-byte objective (§9, S1) | `builder.BuildResult` | Objective is the built file's size. Zeros earn nothing. KV cache is a deployment setting, checked as a constraint at freeze |
| Reference proxy (§7, S8) | `kl.py`, `prober.py`, `tools/traj-kl` | Reference log-probs from the frozen source are computed once and reused. KL guides the search and never decides acceptance. Improve mode uses NLL on verified-successful trajectories instead |
| Learner with failure feedback (§2, S4) | `learner.py` | Moves are ranked by proxy cost per byte saved, re-probed on the current allocation. Each screen runs search-split feedback episodes; their failures become a new reference store. Implicated moves are re-measured by reverting them one at a time, then penalised and, if the screen failed, reverted and blocked. `feedback: false` is the proxy-only ablation |
| Search algorithm switch | `learner.algo` in `pipeline.phase_search`, `oneshot.py` | `"greedy"` (default) is the learner above. `"oneshot"` builds no candidates during search: it reuses per-trajectory solo + Wanda width shifts (`logodds` for classifiers or target-weighted `traj-kl` for generative models, with optional `SizeMatchedCap(bpw)` guard), solves under exact GGUF byte costs (`ByteModel`) + Rule-P vocabulary trim, and builds once to check bytes. On the Qwen3.5-0.8B classifier, fidelity gave 47 select documents lost at 200,240,128 bytes, against 46 for the hand-built 200,296,000-byte q4cut50 and 60 for the proxy objective; at 396 MB it gave 34 against greedy's 29. The final gate, freeze and exam are unchanged |
| Noisy acceptance (§9, S9) | `stats.py`, `pipeline.gate` | Paired, task-matched test. Cluster-robust (family) t interval, 4 sequential looks with Bonferroni α. Sample size comes from the measured A/A run. Detector bounds and behavioral non-inferiority are also checked |
| Optional weight repair (§6, S3) | `repair.py` | LoRA under the exact removal structure: KL to the frozen source in preserve mode, cross-entropy in improve mode. Merged, converted to GGUF, rebuilt by the same builder, re-measured, accepted only if the screen passes |
| Learning from labelled examples & corrective trajectories (S3) | `learn.py`, `lattice.py` | Reconstructs a built GGUF as an exact PyTorch student across all ggml quant formats (`IQ1_S`, `IQ2_XS`, `Q2_K`..`Q8_0`, `BF16`). Supports fixed-grid level moves (`Q4_K`/`Q8_0`) and universal 0-byte `up_row_zero` MLP neuron switches (`learn_gates` for classifiers, `learn_gates_seq` for generative trajectories with replay token-KL + A-GEM projection), exporting back to `chantless.mask/v3` |
| Freeze + reproduction (§11) | `pipeline.phase_freeze` | Rebuild sha256 must be equal. Header must match the mask, and dropped tensors must be absent. Runtime memory is measured at the declared context |
| Single-use exam + report | `pipeline.phase_exam`, `phase_report` | Exam reliability pool, then fresh source and specialist episodes, the claim test, and descriptive uniform / dynquant-K baselines at matched bytes |

## Run

```bash
chantless extract --config configs/local-smoke.json   # tiny integration run on a laptop
chantless extract --config configs/box.json           # full feedback-learner run on 8 GPUs
chantless build   --source BASE.gguf --mask ../masks/qwen3-4b-bfcl-apprentice-mask.json --out out.gguf
chantless learn   --config learn-config.json --out-mask apprentice-mask.json
```

Phases resume from `runs/<id>/phases/*.done`. Run `--reset PHASE` to redo a phase and all later ones.

## Known limitations (by design for the MVP)

- Weight repair trains without simulating quantisation. The loop measures the real built artifact, and repair is accepted only if it helps.
- The dynquant baseline is our K94-style recipe re-implemented on this model, measured once on the same proxy. It is not the original Qwen3.8 pipeline.
- Solo/drop probing uses `SpliceProbeBuilder` (one-time per-tier quantization + `~0.5s` mmap tensor byte splicing), zero-copy `os.replace` BF16 width handoff, optional `/dev/shm` scratch auto-sizing, exact store-filtered probing, and VRAM-aware auto-checkpointing in `learn.py`. Cached move rankings beyond the top-16 greedy schedule remain omitted.
- Removal, Wanda width-slicing, Rule-P vocabulary trimming, and 0-byte `up_row_zero` Switch Apprentice operators are verified end to end across `qwen3`, `qwen35`, and `gemma4` on `llama.cpp` `a01dff9` (`patches/llama-cpp-a01dff9-chantless.patch`); see `../masks/README.md` for exported stock-model thin masks and SHA256 receipts. MTP heads and the vision encoder are left out of the converted text GGUF.
