# Chantless — Architecture & Design Notes

**Your taste. Multiplied.** · Deus Experiments

Chantless keeps stock open-weight checkpoints (`Qwen3`, `Qwen3.5`, `Gemma 4`) **completely frozen** and compiles standalone task specialists into **`6–53 KB` declarative precision masks** (`chantless.mask/v3`) paired with **`+0 byte` binary MLP switches**.

> For flagship benchmark results, hardware telemetry, and SHA256 build receipts, see [`../README.md`](../README.md) and the **`12` verified winner masks** in [`../masks/README.md`](../masks/README.md).

---

## 1. Objective and Standalone Deployment Contract

**Ship the skill. Not the whole model.**

Chantless turns a general-purpose open-weight model into a standalone specialist for a user-defined workload—coding, tool calling, reasoning, or calibrated classification—minimizing deployed weight memory while preserving or exceeding required task quality under declared serving constraints.

The user supplies an executable calibration environment and task verifiers. During compilation, the frozen stock checkpoint guides precision search, structure selection, and optional few-shot adaptation. At deployment, only the built specialist runs—no companion full model, remote fallback, or runtime verifier accompanies the shipped weights.

---

## 2. End-to-End Pipeline

```
FROZEN STOCK MODEL + GENERAL SANITY SUITE
                   │
                   ▼
         INITIAL SANITY MASK
                   │
                   ▼
  USER ENVIRONMENT + RESOURCE BUDGET
                   │
                   ▼
     MASK EXTRACTION & PRUNING
  (solo + Wanda probes → fidelity/KL
   one-shot or feedback search +
   Rule-P vocab trim + dead KV drop)
                   │
                   ▼
  OPTIONAL ADAPTATION & WEIGHT REPAIR
  (0-byte Switch Apprentice / LoRA)
                   │
                   ▼
   BUILD STANDALONE WEIGHTS + AUDIT
                   │
                   ▼
   HELD-OUT EXAM + RUNTIME TELEMETRY
                   │
                   ▼
           SHIP SPECIALIST
```

Chantless supports two complementary search drivers and a post-build zero-byte learner:

- **Feedback-Driven Iterative Search (`learner.algo: "greedy"`):** Moves are ranked by proxy cost per byte saved and re-probed as the allocation evolves. Every behavioral screen injects failed trajectories back into the proxy store, re-measures implicated groups under counterfactual revert, and penalizes or blocks regressing moves (`mvp/chantless/learner.py`).
- **Exact-Byte One-Shot Search (`learner.algo: "oneshot"`):**
  - *Classifier mode (`kind == "logodds"`):* Eliminates verified-dead shared-KV tensors, combines solo tier/drop probes with Wanda-ordered MLP width probes (`||W_down[:, j]||_2 * sqrt(in_sum2_j)` from `llama-imatrix`) and a merge-closed Rule-P vocabulary keep-set, predicts trajectory scores under an additive surrogate, and solves for minimum `1 - Spearman` rank distortion against exact `GGUF` byte costs (`ByteModel`).
  - *Generative mode (`kind == "kl"`):* Consumes per-trajectory `(n_tgt, kl, nll_cand)` deltas from `traj-kl`, minimizes target-weighted KL (`preserve`) or verified-trajectory NLL (`improve`) under the exact byte budget and `SizeMatchedCap(bpw)` general-KL guard, while protecting chat, tool, and reasoning control tokens (`<tool_call>`, `<think>`, `<|...|>`).
- **Zero-Extra-Byte Switch Apprentice (`chantless learn`):** Learns binary `up_row_zero` MLP neuron switches from `<=25` real mistakes via straight-through estimation—combining corrected-token NLL / contrastive score margin, full-vocabulary `float32` `replay_h0` token-KL from cached pre-`lm_head` states, and A-GEM gradient projection. Switched-off neurons materialize by zeroing contiguous super-block rows of `ffn_up.weight` and `ffn_gate.weight` (`d = 0.0f`), achieving exact student-to-GGUF parity across every `ggml` quant format (`IQ1_S`, `IQ2_XS`, `Q2_K`..`Q8_0`, `BF16`) with **`0` extra weight bytes** (`cli_rebuild_sha256_equal`).

---

## 3. Central Artifact: `chantless.mask/v3`

`chantless.mask/v3` captures the complete deterministic transformation from frozen base weights to standalone specialist in a single `6–53 KB` Apache-2.0 JSON file:

- **Per-group and per-tensor precision/removal assignments** across the full `22`-type native `ggml` ladder (`F32` down to `IQ1_S`, plus physical `'drop'`), spanning attention/token mixers (GQA and Gated Delta Net), FFNs, embeddings, and per-layer embeddings (`PLE`).
- **Automatic structural discovery** derived from GGUF metadata and config (`gguf_ops.adapter_from_gguf`), preserving shared-KV donor layers and pruning verified-dead shared-KV projections.
- **Compressed MLP neuron width bitmaps** (Wanda-ordered cumulative cuts) and **`up_row_zero` Apprentice switches** (plus optional sparse `Q4_K`/`Q8_0` grid-level updates).
- **Merge-closed Rule-P vocabulary keep-sets** (`vocab_keep`) shrinking embedding and LM-head rows while preserving BPE merges and answer/control tokens (`tokdocs` re-tokenization handles trimmed vocabularies seamlessly).
- **Zero-trained-float compatibility:** Stock specialists and Switch Apprentice masks carry `weights.lineage = []` and read legacy `v1`/`v2` masks cleanly.

Importance is conditional on the remaining groups, allocations, and task; a mask is an executable engineering prescription rather than an assumption of modular independence.

---

## 4. Executable Calibration Environments & Split Discipline

Every workload implements one uniform contract (`mvp/chantless/env/protocol.py`):

- Declares budget (`max_turns`, `max_tokens_per_turn`) and deployment conditions (`context`, `concurrency`, `kv_type`).
- Supplies tasks partitioned cleanly by family across isolated splits (`search`, `select`, `train`, `exam`, and optional `calibration`).
- Provides `run`, `verify`, and optional calibrated `ClassifierHooks` (`score`, `calibrate`, `decide`, `proxy_spec`, `stats_texts`, `token_counts`).
- `SplitGuard` blocks all reads of `exam` until final evaluation. Infrastructure errors are retried and logged without polluting task failure counts.

---

## 5. General-Data Sanity Protection

Specialization starts from a per-group protection floor computed by `mvp/chantless/sanity.py`:

- Combines general text (`FineWeb-Edu`), standard-library code KL, and executable behavioral checks (`detectors.py`).
- Sets per-group minimum precisions (`floors`) from solo-application tests so breaking baseline coherence is disallowed from the start.
- Enforces `SizeMatchedCap(bpw)` throughout search: every candidate's general-data KL must stay within the curve interpolated from uniform quantizations at matched bits per weight, alongside hard behavioral gates on every screen.

---

## 6. Adaptation Modes: Mask-Only & Mask-Plus-Weights

| Mode | Weights State | Operator & Guarantee |
| :--- | :--- | :--- |
| **Mask-Only & Switch Apprentice** | Frozen stock floats (`weights.lineage = []`) | Quantization tiers, physical sub-block removal, Wanda width slicing, Rule-P vocab trimming, and `+0 B` binary `up_row_zero` neuron switches materialize deterministically from stock BF16/F16 + mask JSON. |
| **Optional Weight Repair (`repair.py`)** | Adapted under frozen mask structure | Trains LoRA adapters on `train` with dropped sub-blocks replaced by identity, merges into BF16, rebuilds the GGUF through the real `Builder`, and accepts only if the rebuilt candidate clears the gate. |

---

## 7. Three-Layer Verification Stack

1. **Task Verifiers:** Environment-specific executable checks—AST & value checks (`BFCL`), exact integer grading (`DeepMath`), unit tests (`codetasks`), or calibrated operating-point metrics (`ClsBench`, `detect`).
2. **Behavioral Failure Detectors (`mvp/chantless/detectors.py`):** Executable gates catching doom loops, malformed tool calls, premature stops, and token corruption.
3. **Frozen-Source Reference & Controls:** Paired comparison against the frozen source (and, for `learn`, strict `base`, `empty`, `flipped`, and `shuffled` controls under fixed compute and A-GEM replay).

All verification runs offline during compilation. No reference model or verifier travels with the shipped specialist.

---

## 8. Two Objectives: Preserve vs. Improve

| Axis | Preserve (`preserve`) | Improve (`improve`) |
| :--- | :--- | :--- |
| **Primary goal** | Match source quality at minimum weight bytes | Surpass source quality within byte budget |
| **Proxy signal** | Reference log-prob KL (`kl`, `logodds` / `pauc_damage`) | Task-verified NLL / calibrated contrastive score |
| **Gate criterion** | Paired sequential non-inferiority (`-3 pt` margin) + detectors | Positive verified delta / verifier floor + detectors |
| **Source role** | Non-inferiority reference & `replay_h0` anchor | Reporting baseline |

---

## 9. Constrained Byte Objective & Acceptance Statistics

Chantless optimizes **actual built weight bytes** (retained tensor payloads, quant block scales, GGUF headers, and alignment)—zeroing a tensor without removal or super-block row pruning saves zero bytes. Deployment context and KV-cache budget are checked separately as hard feasibility constraints (`chantless freeze`), and serving throughput and per-PID VRAM (`CUDA0` / `MTL0`) are measured live via `chantless bench`.

Acceptance uses paired, task-matched statistics (`mvp/chantless/stats.py`) with family-clustered CR1 standard errors, Student-t(`F-1`) bounds, and four Bonferroni-adjusted sequential looks sized from a measured source-vs-source `A/A` run (`default margin = -0.03`, `95%` confidence/power).

---

## 10. Shared Frozen Source, Chunked Loading & Multi-Skill Unions

Because a specialist mask is `6–53 KB` while the frozen base checkpoint is gigabytes, a single shared stock repository can materialize any number of task specialists or operating budgets (`coding · compact`, `tool-use · 2 GB`, `classifier · 10%`):

- **Streaming local materialization:** The builder quantizes and packs kept tensors directly from memory-mapped source chunks, avoiding full-BF16 GPU residency.
- **Concurrent skill unions:** Multiple active mask-only specialists sharing identical `(source_revision, tensor_name, ggml_type, imatrix)` entries share those quantized representations once in storage rather than duplicating full checkpoints.
- **Deterministic CLI rebuilds:** Unified `llama.cpp @ a01dff9` ([`../mvp/patches/llama-cpp-a01dff9-chantless.patch`](../mvp/patches/llama-cpp-a01dff9-chantless.patch)) rebuilds byte-exact SHA256 specialists cleanly from `stock GGUF + mask JSON`.

---

## 11. Prior Art & Positioning

Chantless unifies mixed-precision allocation, physical structural cutting, vocabulary trimming, and zero-byte post-build neuron switching inside a single verifier-driven loop:

| Method | Allocation Signal | Structural & Vocab Cuts | Re-Evaluated Under Task Verifiers | Zero-Byte Post-Build Fix |
| :--- | :--- | :--- | :--- | :--- |
| **Chantless** | Sanity floors + joint KL/NLL/pAUC surrogate + task failures | Physical sub-block drop, Wanda MLP width, Rule-P vocab trim | Iterative feedback + sequential CR1 gates | `up_row_zero` Switch Apprentice (`<=25` examples) |
| **Dynamic Quantization** (`K34`/`K94`) | Imatrix + isolated per-group KL upgrade over low-bit base | Fixed tensor presence/vocab | Offline imatrix/KL selection | Importable directly via `chantless mask import` |
| **Unsloth Dynamic Quants** ([blog](https://unsloth.ai/blog/dynamic-4bit)) | Sensitivity & selective layer protection | Fixed checkpoint structure | Heuristic/perplexity selection | — |
| **Dettmers Dynamic Compression** ([paper](https://timdettmers.com/papers/runtime-dynamic-compression.pdf)) | Iterative calibration log-perplexity sensitivity | Offline budget allocation / MoE expert swap | Iterative re-probing | — |
| **Sherry / MatQuant / Any-Precision / BitStack** | QAT / nested bit-width / stacked residuals | Trained multi-width or multi-bit representations | Training-time gradient loss | Requires modified/trained weights |
| **MatFormer / Matryoshka** | Nested FFN widths / budgeted component attribution | Nested subnetworks | Multi-budget evaluation | Joint weight training |
