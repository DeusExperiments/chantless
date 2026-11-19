# CHANTLESS

![Deus Experiments logo](assets/brand/deus-experiments-logo.png)

Your taste. Multiplied.

Chantless starts from a frozen open-weight reference model and learns a thin `chantless.mask/v3` prescription (`6–80 KB` JSON: precision tiers, sub-block drops, lossless dead shared-KV stripping, Wanda MLP neuron widths, merge-closed Rule-P vocabulary keep-sets, and post-build `up_row_zero` MLP neuron switches) that compiles the frozen base into a compact standalone GGUF specialist—and lets that quantized specialist keep learning from `<= 25` labelled mistakes at **zero extra bytes** without touching float weights.

---

## Quickstart

### 1. Build the Unified Runtime (`llama.cpp` `a01dff9`) & Install `chantless`

All supported architectures (`qwen3`, `qwen35` hybrid Gated DeltaNet + Full Attention, and `gemma4` Per-Layer Embeddings + Shared-KV) build and serve from a single `llama.cpp` patch:

```bash
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp && git checkout a01dff9
git apply /path/to/Chantless/mvp/patches/llama-cpp-a01dff9-chantless.patch
cmake -B build -DGGML_CUDA=ON   # or -DGGML_METAL=ON on macOS
cmake --build build --config Release -j --target llama-server llama-quantize llama-imatrix
```

Install the Python package and CLI (`chantless`):

```bash
pip install -e ".[train,dev]"
```

### 2. Core CLI (`extract`, `build`, `learn`)

```bash
# 1. Rebuild a standalone GGUF deterministically from stock HF BF16 + a thin v3 mask:
chantless build \
  --source qwen3-4b-bf16.gguf \
  --mask masks/qwen3-4b-bfcl-apprentice-mask.json \
  --llama-bin /path/to/llama.cpp/build/bin \
  --out qwen3-4b-bfcl-apprentice.gguf

# 2. Extract a new specialist mask under an exact byte budget and sequential select gate:
chantless extract --config mvp/configs/local-smoke.json --out-mask specialist-mask.json

# 3. Teach an already-built quantized GGUF from <=25 mistakes at 0 extra bytes:
chantless learn --config learn-config.json --out-mask apprentice-mask.json
```

---

## Empirical Matrix (`3 Architectures × 3 Modalities × {Extract, Build, Learn}` — `18 / 18` Masks Exported)

Every specialist and Apprentice artifact below has `weights.lineage = []` (zero float fine-tuning or LoRA repair) and verified `cli_rebuild_sha256_equal == true` from unmodified stock checkpoints:
- **Classifiers (`ClsBench`):** `<= 10.00%` of BF16 GGUF bytes (`logodds` Spearman rank-fidelity one-shot solver + Rule-P vocabulary trim + Wanda widths).
- **Tool Calling (`BFCL`) & Math Reasoning (`DeepMath`):** Pure mask-only non-inferiority (`traj-kl` one-shot solver + Wanda late-layer width cuts + control-tag-safe Rule-P vocabulary trim).
- **Switch Apprentice (`Learn`):** Trains `up_row_zero` MLP neuron switches on `25` specialist mistakes (`float32` `replay_h0` full-vocabulary token-KL guard) and evaluates unseen held-out mistakes against `4` matched-byte controls (`base`, `empty`, `flipped`, `shuffled`), with `empty.sha256 == base.sha256` on all `9/9` pairs.

All `18` thin masks and exact SHA256 receipts are in [`masks/README.md`](masks/README.md):

| Model (`Pinned HF Revision`) | Task | Built GGUF (`%` BF16, `bpw`) | Locked Exam vs Uniform Quant | Switch Apprentice (`25` ex → Unseen Fixed vs `base / empty / flip / shuf`) |
| :--- | :--- | :---: | :--- | :--- |
| **`Qwen/Qwen3.5-4B`** (`851bf6e8`) | `ClsBench` | `837.2 MB` (**`9.94%`**, `1.59`) | **`93.40%`** vs `46.66% IQ1_S` (**`+46.73` pts**) | **`33 / 60`** vs `4 / 4 / 5 / 20` (`92.79%` vs `88.72%` `sel1`) |
| **`Qwen/Qwen3.5-4B`** (`851bf6e8`) | `BFCL` | `2.191 GB` (**`26.01%`**, `4.17`) | **`92.39%`** vs `92.39% Q3_K` (`+0.00` pts) | **`29 / 60`** vs `0 / 0 / 2 / 19` (`83.22%` `sel1`) |
| **`Qwen/Qwen3.5-4B`** (`851bf6e8`) | `DeepMath` | `2.192 GB` (**`26.02%`**, `4.17`) | **`82.27%`** vs `51.77% Q3_K` (**`+30.50` pts**) | `9 / 60` vs `10 / 8 / 10 / 11` (`61.67%` `sel1`, eval-variance cluster) |
| **`google/gemma-4-E2B-it`** (`4adf1574`) | `ClsBench` | `928.7 MB` (**`9.97%`**, `1.60`) | **`99.21%`** vs `49.43% IQ1_S` (**`+49.78` pts**) | **`15 / 60`** vs `3 / 3 / 2 / 8` (`99.40%` vs `99.15%` `sel1`) |
| **`google/gemma-4-E2B-it`** (`4adf1574`) | `BFCL` | `1.948 GB` (**`20.92%`**, `3.35`) | **`97.13%`** vs `53.11% Q2_K` (**`+44.02` pts**) | **`26 / 60`** vs `0 / 0 / 0 / 7` (`83.33%` `sel1`) |
| **`google/gemma-4-E2B-it`** (`4adf1574`) | `DeepMath` | `2.128 GB` (**`22.85%`**, `3.66`) | **`75.00%`** vs `0.00% Q3_K` (**`+75.00` pts**) | **`10 / 60`** vs `3 / 3 / 4 / 4` (`80.00%` vs `40.00%` `sel1`) |
| **`Qwen/Qwen3-4B`** (`1cfa9a72`) | `ClsBench` | `800.2 MB` (**`9.94%`**, `1.59`) | **`60.96%`** vs `45.44% IQ1_S` (**`+15.51` pts**) | **`16 / 60`** vs `4 / 4 / 4 / 15` (`49.70%` vs `43.38%` `sel1`) |
| **`Qwen/Qwen3-4B`** (`1cfa9a72`) | `BFCL` | `1.972 GB` (**`24.50%`**, `3.92`) | **`92.31%`** vs `86.44% Q3_K` (**`+5.87` pts**) | **`22 / 60`** vs `3 / 3 / 1 / 12` (`87.29%` `sel1`) |
| **`Qwen/Qwen3-4B`** (`1cfa9a72`) | `DeepMath` | `2.172 GB` (**`26.98%`**, `4.32`) | **`76.62%`** vs `67.66% Q3_K` (**`+8.96` pts**) | `20 / 49` vs `9 / 9 / 14 / 20` (ties `shuffled`; `61.33%` `sel1`) |

*(See [`masks/README.md`](masks/README.md) for control details on `qwen35-4b-clsbench` balanced-label `shuffled` and `qwen35-4b-bfcl`/`deepmath` pair-shuffled controls.)*

---

## Earlier Milestones & Provenance

- **Qwen3.5-0.8B 200 MB Detector & Apprentice** ([mvp/README.md](mvp/README.md), [mvp/detect/](mvp/detect/)): One-shot rank-fidelity mask (`f200`, `200,240,128 B`, `47 / 14,161` select docs lost) and `396.5 MB` sealed exam (`50,808` docs); `20`-lesson Apprentice (`200,296,000 B`) improved from `8/25` to `25/25` on fresh AI passages with `0/25` human originals flagged.
- **Gemma 4 E2B 10.00% Detector & Apprentice** ([mvp/gemma4/README.md](mvp/gemma4/README.md)): `mask10` (`930,971,136 B` = `10.00%` of `9.31 GB` BF16, `7 B` off `ByteModel`) achieved `99.76%` agreement on the `51,066`-document sealed exam (`L = -0.0037`); zero-extra-byte switch mask (`i1`, `3,569 / 96,768` switches off) improved from `8/25` to `18/25` (`0/50` human flagged) and fixed `241` unseen mistakes vs `36` for the flipped-label control.
- **Generative Feedback Loop MVP & Paused Pilot:** `box-mvp-001` ([mvp/README.md](mvp/README.md)) verified iterative failure-feedback search on `Qwen3-0.6B` (`codetasks`). The earlier MiMo 9B SWE-bench precision-search pilot remains paused ([runs/search-pilot/STATUS.md](runs/search-pilot/STATUS.md)).

## Documentation Map

- [Thin Masks & SHA256 Reproduction Guide](masks/README.md)
- [Architecture & Design Notes](explainer/design-notes.md) · [Detailed Explainer](explainer/explainer-detailed.html) · [Project Overview Map](work/chantless-project-overview/chantless-project-map.html)
- [Engine & Module Reference](mvp/README.md) · [Experiment Manual](manual/MANUAL.md) · [Paused Pilot Status](runs/search-pilot/STATUS.md)

## Repository Hygiene

Private origin: [DeusExperiments/chantless](https://github.com/DeusExperiments/chantless). Local scratch, reference libraries, credentials, archives, and model weights (`*.gguf`, `*.safetensors`) are excluded via `.gitignore` and are never committed or published; only source code, patches, tests, documentation, and thin `chantless.mask/v3` JSON masks are tracked.
