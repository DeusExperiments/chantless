# CHANTLESS

![Deus Experiments logo](assets/brand/deus-experiments-logo.png)

**Your taste. Multiplied.**

Chantless extracts capabilities from frozen models without training their weights. The extracted capabilities remain independently deployable and adaptable.

Conventional compression work tries to preserve as much of the model as possible, primarily under a size constraint. Chantless utilizes search algorithms to best attribute everything that really contributes to a capability and ensure it is neatly abstracted. Once we identify the capability, we extract it in a novel way that keeps it malleable.

In practice, you hand Chantless a small batch of up to 25 real mistakes. Instead of training new float weight values, it decides which small set of late-layer neurons to switch off. Flipping those switches adds zero bytes to the file.

[**Read the research announcement — How do models lose weight without crying?**](https://deusexperiments.com/research/how-do-models-lose-weight-without-crying/)

---

## Quickstart

```bash
# 1. Build unified llama.cpp@afe77e18 (CUDA / Metal / CPU auto-detect) + C++ scorers & install CLI
./scripts/build_toolchain.sh --backend auto --prefix .local/toolchain
pip install -e ".[train,dev]"
chantless doctor --llama-bin .local/toolchain/llama-afe77e18/build/bin --traj-kl .local/toolchain/bin/traj-kl

# 2. Materialize any shipped winner mask into a standalone SHA256-exact GGUF
chantless build \
  --source qwen3-4b-bf16.gguf \
  --mask masks/qwen3-4b-bfcl-apprentice-mask.json \
  --llama-bin .local/toolchain/llama-afe77e18/build/bin \
  --out qwen3-4b-bfcl-apprentice.gguf

# 3. Serve or benchmark directly from stock base + mask (SHA256-cached auto-build + live CUDA/Metal VRAM)
chantless bench \
  --source qwen3-4b-bf16.gguf \
  --mask masks/qwen3-4b-bfcl-apprentice-mask.json \
  --compare qwen3-4b-bf16.gguf \
  --llama-bin .local/toolchain/llama-afe77e18/build/bin
```

---

## Verified Winners

All **`16` trophy masks** (`6.4–52.8 KB`, zero trained floats) and SHA256 receipts live in [`masks/README.md`](masks/README.md).

### 1. Stock-Checkpoint Double Winners (`extract` + `<=25`-Mistake `+0 B` `learn`, `4` Architectures)

| Model | Task | Size (`%` BF16) | Mask | Held-Out `exam` | Gain vs Uniform | Apprentice (`<=25` ex → Unseen Fixed) | Controls (`base/empty/flip/shuf`) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`Gemma 4 E2B`** | `ClsBench` | `928.7 MB` (**`9.97%`**) | `47.7 KB` | **`99.21%`** | **`+49.78` pts** | **`15 / 60`** (`+0 B`) | `3 / 3 / 2 / 8` |
| **`Qwen3.5-4B`** | `ClsBench` | `837.2 MB` (**`9.94%`**) | `48.5 KB` | **`93.40%`** | **`+46.73` pts** | **`33 / 60`** (`55.0%`, `+0 B`) | `4 / 4 / 5 / 20` |
| **`Gemma 4 E2B`** | `BFCL` | `1.95 GB` (`20.92%`) | `47.1 KB` | **`97.13%`** | **`+44.02` pts** | **`26 / 60`** (`43.3%`, `+0 B`) | `0 / 0 / 0 / 7` |
| **`Qwen3-4B`** | `BFCL` | `1.97 GB` (`24.50%`) | `6.4 KB` | **`92.31%`** | **`+5.87` pts** | **`22 / 60`** (`36.7%`, `+0 B`) | `3 / 3 / 1 / 12` |
| **`Qwen3.5-4B`** | `BFCL` | `2.19 GB` (`26.01%`) | `18.6 KB` | **`92.39%`** | `+0.00` pts | **`29 / 60`** (`48.3%`, `+0 B`) | `0 / 0 / 2 / 19` |
| **`Inkling-2-Mini`** (`24B` MoE) | `ClsBench` | `4.17 GiB` (**`8.58%`**) | `33.9 KB` | **`98.67%` @ `5.00%` FPR** | `11.65×` smaller (`BF16` `99.17%`) | **`10 / 60`** (`+0 B`, exam `+0.34 pp`) | `4 / 4 / 3 / 8` |

### 2. Additional Looped-MoE `Inkling-2-Mini` (`24B` / `128` Experts) Specialist Winners

| Mask | Task | Size (`%` BF16) | Mask | Held-Out `exam` (vs `BF16`) | `reasoning_effort="high"` (`0.9`) Stress Test |
| :--- | :--- | :---: | :---: | :--- | :--- |
| `inkling2-mini-clsbench-mask.json` | `ClsBench` | `4.17 GiB` (**`8.58%`**) | `27.9 KB` | **`98.33%` recall** @ `5.33%` FPR, `AUROC=0.9965` (`BF16` `99.17%` @ `5.00%`) | Upgraded by `+0 B` Apprentice above (`98.67%` @ `5.00%` FPR) |
| `inkling2-mini-bfcl-mask.json` | `BFCL` | `14.22 GiB` (**`29.23%`**) | `9.0 KB` | **`115/160` (`71.88%`, `+0.63 pp`)** Gorilla (`BF16` `114/160`) | **BFCL-HIGH (`8k`):** `53/79` Gorilla tie · **DeepMath-HIGH (`16k`):** **`41/48` (`0` regressions)** |
| `inkling2-mini-deepmath-mask.json` | `DeepMath` | `12.48 GiB` (**`25.66%`**) | `8.4 KB` | **`81/96` (`84.38%`)** `"low"` `4k` (`BF16` `84/96`, `1.45×` faster) | **DeepMath-HIGH (`16k`):** **`40/48` strict / `41/48` raw** (`25%` experts pruned) |

### 3. Imported `27B` Winners (`chantless mask import`, Terminal-Bench 2.1)

| Mask | Model | Size (`%` F16) | Mask | Recorded Score (`89` tasks × `4` repeats) |
| :--- | :--- | :---: | :---: | :--- |
| `qwen38-27b-dynquant-k34-mask.json` | `Qwen3.8-27B` | `19.60 GB` (`35.87%`) | `9.7 KB` | **`77.8%` pass@1** (`277/356` audited) · **`91.0%` pass@4** (`81/89`) |
| `qwen38-27b-dynquant-k94-mask.json` | `Qwen3.8-27B` | `14.26 GB` (`26.09%`) | `9.6 KB` | **`74.4%` pass@1** (`265/356` audited) · **`85.4%` pass@4** (`76/89`) |

### 4. Live Hardware Runtime Telemetry (NVIDIA CUDA & Apple Metal)

| Measurement | Hardware | Baseline | Chantless | Delta |
| :--- | :--- | :---: | :---: | :--- |
| **Apprentice Training (Streaming Prefix)** | NVIDIA RTX 4090 (`sm_89`) | `3,072.8 MB` alloc · `18.84s` | **`493.3 MB`** alloc · `15.76s` *(cached `13.10s`)* | **`-83.95%`** CUDA VRAM (bit-exact, `0.0` diff) |
| **Inference (`chantless bench`, `Q4_K` vs `BF16`)** | NVIDIA RTX 4090 (`CUDA0`) | `440.95 tok/s` · `1,844 MiB` VRAM | **`630.0 tok/s`** · **`1,016 MiB`** VRAM (`571 MiB` `CUDA0`) | **`1.429×`** decode · **`0.551×`** `nvidia-smi` VRAM |
| **Inference & Apprentice (`mps` / `MTL0`)** | Apple Silicon | `1,286 MiB` `MTL0` | **`464 MiB`** `MTL0` (`build_s=0.0` cache hit) | **`2.498×`** decode · **`0.361×`** memory · `3.9×–13.5×` learn |

### 4. Production-Scale Classifier Proof & OOD Counter-Baselines

| Capability | Held-out result | Extracted size |
|---|---|---|
| AI writing detection · Gemma 4 E2B | **99.76% retention of source-correct decisions** | **930.97 MB (10% of BF16)** |

**Artifacts:** Source checkpoint, winning extraction mask, evaluation receipt and reproduction instructions.

**Caveat:** Passes pooled non-inferiority; Beemo alone falls outside the two-point margin.

- **`Qwen3.5-0.8B` `200 MB` multi-pass Apprentice vs Small-Model SFT/KD:** Improved **`8/25 → 25/25`** on unseen AI passages (`0/25` human FPR) at **`6.6 × 10¹⁶` FLOPs** replay (`3.2×` cheaper than Hinton KD on `ModernBERT-base`). On the exact same `14,991`-doc splits, labels-only SFT on `ModernBERT-base` (`159 MB`) and `LFM2.5-230M` (`244 MB`) collapsed OOD (**`75.21%`** and **`78.44%`** human false-positive rate on Beemo).

---

## CLI & Capabilities

| Area | Command | Capabilities | Key Options |
| :--- | :--- | :--- | :--- |
| **Toolchain Bootstrap** | [`scripts/build_toolchain.sh`](scripts/build_toolchain.sh) | Builds pinned `llama.cpp @ afe77e18` (single unified patch for `Qwen3` / `Qwen3.5` / `Gemma 4` / `Inkling-2-Mini`) + `traj-kl`, `logodds`, `tokdocs` | `--backend auto\|cuda\|metal\|cpu`, auto single-arch CUDA detection, `--with-fla`, `--dry-run` |
| **Environment Preflight** | `chantless doctor` | JSON health check for `afe77e18` binaries, C++ scorers, CUDA/Metal + `fla`, Python deps, and `/dev/shm` + disk | `--config`, `--model-gb 54` (`27B` scratch/tier-cache layout preview), `--require-cuda`, `--require-classifier` |
| **Declarative Configs** | `chantless validate` / `schema` | Strict Pydantic v2 validation + JSON Schema export | JSON or YAML, file-relative `extends:`, repeatable `--set key.path=val`, `22` opt-in ggml tiers |
| **Dry-Run Planner** | `chantless plan` | Estimates moves, probes, episodes, wall-clock, and spend before compute | `--rates custom.json`, `--source local.gguf`, `--set` overrides |
| **Specialist Extraction** | `chantless extract` | Frozen-base extraction (`prepare → proxy → sanity → search → freeze → exam → report`) with one-shot solver, Wanda widths, Rule-P vocab trim, lossless dead shared-KV drop, and `SplitGuard` | `--upto`, `--reset`, `--out-mask`, `--out`, decoupled `/dev/shm` vs NVMe (`1.5×`/`3.5×`), auto worker clamp, tier cache |
| **Zero-Byte Apprentice** | `chantless learn` | Adapts built specialists from `<=25` mistakes at `+0 B` via `up_row_zero` switches + full-vocab token-KL & A-GEM + `4` strict controls | Prefix activation cache, `stream_prefix="auto"`, CUDA pinned memory, deterministic `judge_slots=1` |
| **Multi-Cell Sweeps** | `chantless sweep` | Declarative matrix runner across models, tasks, and budgets | `--launcher local\|ssh\|modal\|colab`, `--dry-run`, auto-resume, `--stop-on-failure` |
| **Mask Algebra & Import** | `chantless mask <cmd>` | Git-like operations on `chantless.mask/v3` across all `22` native `llama-quantize` ggml types (`F32..IQ1_S`) | `inspect`, `diff`, `fork`, `merge` (`apprentice` / `specialist`), `pull` (`hf://`/`https://` SHA256-pinned), `import` (`--from-gguf`, `--from-tensor-types`, `--from-spec`) |
| **Deterministic Build** | `chantless build` | SHA256-exact standalone GGUF compilation from stock BF16 + mask | Auto sharded `llama-quantize` (`--shards`, `--threads`), `--hf-dir` shared-KV resolution, `--imatrix` isolation |
| **Serve & Telemetry** | `chantless serve` / `bench` | OpenAI-compatible serving & benchmarking with structured JSON receipts | Direct `--model` or `--source + --mask` (SHA256 build cache), `--compare`, live per-PID VRAM, `/metrics`, clean `SIGTERM` |

---

## Documentation

- [Verified Winner Masks & SHA256 Receipts](masks/README.md) · [Engine Reference](mvp/README.md) · [Architecture & Design Notes](explainer/design-notes.md)
- Example configs: [extract-example.yaml](mvp/configs/extract-example.yaml) · [learn-example.yaml](mvp/configs/learn-example.yaml) · [local-smoke.json](mvp/configs/local-smoke.json)
