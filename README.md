# CHANTLESS

![Deus Experiments logo](assets/brand/deus-experiments-logo.png)

**Your taste. Multiplied.**

Chantless eliminates **weight debt**. Instead of fine-tuning, distilling, and shipping mutated gigabyte checkpoints, Chantless keeps stock open-weight models (`Qwen3`, `Qwen3.5`, `Gemma 4`) **completely frozen** and compiles capabilities into **`6–53 KB` Apache-2.0 declarative masks** (`chantless.mask/v3`, zero trained floats).

From any stock Hugging Face checkpoint + thin mask, Chantless deterministically builds a SHA256-exact standalone GGUF—and adapts it from **`<= 25` real mistakes at `+0 bytes`** by flipping binary MLP neuron switches rather than rewriting float weights.

---

## Quickstart

```bash
# 1. Build unified llama.cpp@a01dff9 (CUDA / Metal / CPU auto-detect) + C++ scorers & install CLI
./scripts/build_toolchain.sh --backend auto --prefix .local/toolchain
pip install -e ".[train,dev]"
chantless doctor --llama-bin .local/toolchain/llama-a01dff9/build/bin --traj-kl .local/toolchain/bin/traj-kl

# 2. Materialize any shipped winner mask into a standalone SHA256-exact GGUF
chantless build \
  --source qwen3-4b-bf16.gguf \
  --mask masks/qwen3-4b-bfcl-apprentice-mask.json \
  --llama-bin .local/toolchain/llama-a01dff9/build/bin \
  --out qwen3-4b-bfcl-apprentice.gguf

# 3. Serve or benchmark directly from stock base + mask (SHA256-cached auto-build + live CUDA/Metal VRAM)
chantless bench \
  --source qwen3-4b-bf16.gguf \
  --mask masks/qwen3-4b-bfcl-apprentice-mask.json \
  --compare qwen3-4b-bf16.gguf \
  --llama-bin .local/toolchain/llama-a01dff9/build/bin
```

---

## Verified Winners

All **`12` trophy masks** (`6.4–52.8 KB`, zero trained floats) and SHA256 receipts live in [`masks/README.md`](masks/README.md).

### 1. Stock-Checkpoint Double Winners (`extract` + `<=25`-Mistake `+0 B` `learn`)

| Model | Task | Size (`%` BF16) | Mask | Held-Out `exam` | Gain vs Uniform | Apprentice (`<=25` ex → Unseen Fixed) | Controls (`base/empty/flip/shuf`) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`Gemma 4 E2B`** | `ClsBench` | `928.7 MB` (**`9.97%`**) | `47.7 KB` | **`99.21%`** | **`+49.78` pts** | **`15 / 60`** (`+0 B`) | `3 / 3 / 2 / 8` |
| **`Qwen3.5-4B`** | `ClsBench` | `837.2 MB` (**`9.94%`**) | `48.5 KB` | **`93.40%`** | **`+46.73` pts** | **`33 / 60`** (`55.0%`, `+0 B`) | `4 / 4 / 5 / 20` |
| **`Gemma 4 E2B`** | `BFCL` | `1.95 GB` (`20.92%`) | `47.1 KB` | **`97.13%`** | **`+44.02` pts** | **`26 / 60`** (`43.3%`, `+0 B`) | `0 / 0 / 0 / 7` |
| **`Qwen3-4B`** | `BFCL` | `1.97 GB` (`24.50%`) | `6.4 KB` | **`92.31%`** | **`+5.87` pts** | **`22 / 60`** (`36.7%`, `+0 B`) | `3 / 3 / 1 / 12` |
| **`Qwen3.5-4B`** | `BFCL` | `2.19 GB` (`26.01%`) | `18.6 KB` | **`92.39%`** | `+0.00` pts | **`29 / 60`** (`48.3%`, `+0 B`) | `0 / 0 / 2 / 19` |

### 2. Imported `27B` Winners (`chantless mask import`, Terminal-Bench 2.1)

| Mask | Model | Size (`%` F16) | Mask | Recorded Score (`89` tasks × `4` repeats) |
| :--- | :--- | :---: | :---: | :--- |
| `qwen38-27b-dynquant-k34-mask.json` | `Qwen3.8-27B` | `19.60 GB` (`35.87%`) | `9.7 KB` | **`77.8%` pass@1** (`277/356` audited) · **`91.0%` pass@4** (`81/89`) |
| `qwen38-27b-dynquant-k94-mask.json` | `Qwen3.8-27B` | `14.26 GB` (`26.09%`) | `9.6 KB` | **`74.4%` pass@1** (`265/356` audited) · **`85.4%` pass@4** (`76/89`) |

### 3. Live Hardware Runtime Telemetry (NVIDIA CUDA & Apple Metal)

| Measurement | Hardware | Baseline | Chantless | Delta |
| :--- | :--- | :---: | :---: | :--- |
| **Apprentice Training (Streaming Prefix)** | NVIDIA RTX 4090 (`sm_89`) | `3,072.8 MB` alloc · `18.84s` | **`493.3 MB`** alloc · `15.76s` *(cached `13.10s`)* | **`-83.95%`** CUDA VRAM (bit-exact, `0.0` diff) |
| **Inference (`chantless bench`, `Q4_K` vs `BF16`)** | NVIDIA RTX 4090 (`CUDA0`) | `440.95 tok/s` · `1,844 MiB` VRAM | **`630.0 tok/s`** · **`1,016 MiB`** VRAM (`571 MiB` `CUDA0`) | **`1.429×`** decode · **`0.551×`** `nvidia-smi` VRAM |
| **Inference & Apprentice (`mps` / `MTL0`)** | Apple Silicon | `1,286 MiB` `MTL0` | **`464 MiB`** `MTL0` (`build_s=0.0` cache hit) | **`2.498×`** decode · **`0.361×`** memory · `3.9×–13.5×` learn |

### 4. Production-Scale Classifier Proof & OOD Counter-Baselines
- **`Gemma 4 E2B` `10.00%` (`931 MB`) Detector & Switch Apprentice:** **`99.76%`** agreement on a **`51,066`-document** held-out exam (`99.82%` AI caught @ `0.08%` human FPR); on `4,549` unseen Beemo docs, real labels fixed **`241`** mistakes vs **`36`** for flipped labels.
- **`Qwen3.5-0.8B` `200 MB` multi-pass Apprentice vs Small-Model SFT/KD:** Improved **`8/25 → 25/25`** on unseen AI passages (`0/25` human FPR) at **`6.6 × 10¹⁶` FLOPs** replay (`3.2×` cheaper than Hinton KD on `ModernBERT-base`). On the exact same `14,991`-doc splits, labels-only SFT on `ModernBERT-base` (`159 MB`) and `LFM2.5-230M` (`244 MB`) collapsed OOD (**`75.21%`** and **`78.44%`** human false-positive rate on Beemo).

---

## CLI & Capabilities

| Area | Command | Capabilities | Key Options |
| :--- | :--- | :--- | :--- |
| **Toolchain Bootstrap** | [`scripts/build_toolchain.sh`](scripts/build_toolchain.sh) | Builds pinned `llama.cpp @ a01dff9` (single unified patch for `Qwen3` / `Qwen3.5` / `Gemma 4`) + `traj-kl`, `logodds`, `tokdocs` | `--backend auto\|cuda\|metal\|cpu`, auto single-arch CUDA detection, `--with-fla`, `--dry-run` |
| **Environment Preflight** | `chantless doctor` | JSON health check for `a01dff9` binaries, C++ scorers, CUDA/Metal + `fla`, Python deps, and `/dev/shm` + disk | `--config`, `--model-gb 54` (`27B` scratch/tier-cache layout preview), `--require-cuda`, `--require-classifier` |
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
