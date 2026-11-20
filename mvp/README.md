# Chantless Engine Reference

The `chantless` package (`mvp/chantless/`) compiles stock open-weight checkpoints (`Qwen3`, `Qwen3.5`, `Gemma 4`, plus `27B` imported allocations) into standalone GGUF specialists, adapts built models from `<=25` mistakes via `+0 B` binary MLP switches, and serves and benchmarks them on unified `llama.cpp @ a01dff9` ([`patches/llama-cpp-a01dff9-chantless.patch`](patches/llama-cpp-a01dff9-chantless.patch)). For full design rationale, see [`../explainer/design-notes.md`](../explainer/design-notes.md); for shipped `12` winner masks and SHA256 receipts, see [`../masks/README.md`](../masks/README.md).

## Runtime Architecture

Unified on `llama.cpp @ a01dff9` + `patches/llama-cpp-a01dff9-chantless.patch` supporting `qwen3`, `qwen35`, and `gemma4` with per-layer `hparams.n_ff(il)` and `hparams.attn_dropped`:
- Dropped FFN or mixer blocks act as identity; dropped full-attention blocks allocate no KV cache, and dropped Gated Delta Net layers allocate no recurrent state.
- Narrowed FFN layers (`widths`) load sliced super-block weights cleanly alongside heterogeneous per-tensor `ggml` quantizations (`F32..IQ1_S`).
- Zero-byte Apprentice switches zero contiguous `up_row_zero` rows across `ffn_up.weight` and `ffn_gate.weight`, preserving `0.0f` exact PyTorch student-to-GGUF parity.

## Module Map

| Component | Module | Responsibility |
| :--- | :--- | :--- |
| **Environment Contract** | `env/protocol.py`, `env/{bfcl,clsbench,deepmath,detect,codetasks}.py` | Uniform `tasks` / `run` / `verify` contract + optional `ClassifierHooks` (`score`, `calibrate`, `decide`); family-isolated `search` / `select` / `exam` / `train` / `calibration` splits. |
| **Failure Detectors** | `detectors.py` | Gating episode checks for doom loops, malformed tool calls, premature stops, and token corruption. |
| **Split Isolation** | `pipeline.py` (`SplitGuard`) | Enforces zero reads of `exam` prior to the final `exam` phase and screens pool reliability on the source only. |
| **Sanity Suite & Floors** | `sanity.py`, `pipeline.phase_sanity` | FineWeb-Edu + stdlib code KL store, per-group solo protection floors, `SizeMatchedCap(bpw)` guard, and behavioral checks. |
| **Thin Mask Schema & CLI Algebra** | `mask.py` | `chantless.mask/v3` schema (`tiers`, `dead_tensors`, Wanda `widths`, Rule-P `vocab_keep`, `apprentice` switches/levels/decision, SHA256 provenance), `inspect`/`diff`/`fork`/`merge`/`pull`/`import`, backward-compatible with `v1`/`v2`. |
| **Physical Builder & Vocab Trim** | `builder.py`, `gguf_ops.py`, `vocab.py`, `shape.py` | Prunes dropped and dead shared-KV tensors, slices Wanda widths and Rule-P vocabularies in BF16, runs sharded `llama-quantize`, applies Apprentice switches/levels, and audits GGUF headers. |
| **Actual-Byte Objective** | `builder.BuildResult`, `oneshot.ByteModel` | Exact compiled `.gguf` byte accounting (payloads, scales, headers, alignment); enforces KV-cache deployment constraints at `freeze`. |
| **Trajectory & Log-Odds Proxies** | `kl.py`, `prober.py`, `objectives.py`, `tools/{traj-kl,logodds,tokdocs}.cpp` | Frozen-source reference KL/NLL stores (`traj-kl`), calibrated classifier log-odds stores (`logodds`, `pauc_damage`), `SpliceProbeBuilder` single-pass tier splicing, and trimmed-vocab retokenization (`tokdocs`). |
| **Iterative Feedback Search** | `learner.py` | Proxy-guided search with feedback injection from failed episodes, counterfactual revert diagnostics, exchange pairs, and blocking of regressing groups. |
| **One-Shot Extraction** | `oneshot.py` | Joint `(tier, width, vocab)` optimization combining solo probes, Wanda MLP width curves, and exact `ByteModel` sizing (`fidelity` for classifiers; target-weighted `kl`/`nll` under `SizeMatchedCap` for generative models). |
| **Statistical Gate** | `stats.py`, `pipeline.gate` | Paired, task-matched CR1 cluster-robust Student-t sequential non-inferiority test (4 Bonferroni looks, sized by A/A variance) + detector bounds. |
| **Optional Weight Repair** | `repair.py` | LoRA training under the active structural mask (`preserve` KL or `improve` NLL), merged to BF16, rebuilt through `Builder`, and gated on real GGUF evaluation. |
| **Switch Apprentice & Reconstruction** | `learn.py`, `lattice.py` | Exact PyTorch student across all native `ggml` formats (`IQ1_S..BF16`) supporting universal `+0 B` `up_row_zero` binary switches (`learn_gates`, `learn_gates_seq` with `replay_h0` KL + A-GEM) and `Q4_K`/`Q8_0` grid moves. |
| **Freeze, Exam & External Compare** | `pipeline.py`, `compare.py`, `plan.py` | Bit-exact `rebuild_sha256` verification, held-out `exam` evaluation, matched-byte baselines, shared-resample classifier `compare`, and dry-run cost/compute estimation (`plan`). |
| **Sweeps, Serve & Diagnostics** | `sweep.py`, `serve.py`, `doctor.py`, `cli.py`, `config.py` | Multi-cell sweep launcher (`local`/`ssh`/`modal`/`colab`), OpenAI-compatible serving + `bench` telemetry, environment preflight (`doctor`), and strict Pydantic v2 config validation. |

## Quick CLI Usage

```bash
# 1. Validate configs and inspect environment health
chantless validate --config mvp/configs/extract-example.yaml
chantless doctor --llama-bin .local/toolchain/llama-a01dff9/build/bin --traj-kl .local/toolchain/bin/traj-kl --model-gb 54

# 2. Local integration smoke
chantless extract --config mvp/configs/local-smoke.json

# 3. Extract a specialist, then train a +0 B Switch Apprentice
chantless extract --config mvp/configs/extract-example.yaml --out-mask specialist-mask.json --out specialist.gguf
chantless learn   --config mvp/configs/learn-example.yaml --out-mask apprentice-mask.json

# 4. Deterministic rebuild and import
chantless build --source BASE.gguf --mask masks/qwen3-4b-bfcl-apprentice-mask.json --out out.gguf
chantless mask import --from-gguf built.gguf --source-gguf BASE.gguf --out imported-mask.json
```

Phases resume automatically from `<run_root>/<run_id>/phases/*.done`; pass `--reset PHASE` to re-run from a given stage.

## Engineering Notes

- **Probe efficiency:** `SpliceProbeBuilder` performs a one-time per-tier quantization pass and assembles candidates via `~0.5 s` mmap tensor splicing, zero-copy `os.replace` BF16 handoff, and decoupled `/dev/shm` vs. NVMe scratch auto-sizing.
- **VRAM streaming:** `learn.py` streams frozen-base prefix states across CUDA (`pin_memory`) or Metal to cut training allocations (`-83.95%` measured CUDA alloc on RTX 4090) with exact numerical parity.
- **Conversion scope:** Text-only specialist GGUF conversion omits MTP heads and vision encoders.
