# Chantless Thin Masks (`chantless.mask/v3`)

This directory contains `22` pure Apache-2.0 `chantless.mask/v3` JSON masks (`6.4 KB – 52.8 KB` each, `weights.lineage = []`):
- **`18` 3×3 matrix masks** across `3` architectures × `3` modalities × `{specialist, Apprentice}`, and
- **`4` imported dynamic-quantization winners** (`Qwen3.8-27B` `K34`/`K94` and `Qwen3-8B` `IQ2_M K08`/`UD-repro`), converted via `chantless mask import` into native `chantless.mask/v3` format with preserved original provenance and evaluation receipts.

No model weights are stored in these files. Every mask specifies only:
- Per-group / per-tensor quantization tiers and sub-block drops (`attn_dropped`, FFN drops),
- Lossless dead shared-KV tensor removals (`dead_tensors`),
- Wanda MLP neuron width bitmaps (`widths`),
- Merge-closed Rule-P vocabulary keep-set bitmaps (`vocab_keep`), and
- Post-build zero-extra-byte `up_row_zero` MLP neuron switch bitmaps (`apprentice`).

## Pinned Stock Source Checkpoints

| Family | Hugging Face Repo | Pinned Revision | BF16 GGUF Bytes | BF16 GGUF SHA256 | Masks in `masks/` |
| :--- | :--- | :--- | :---: | :--- | :---: |
| **Qwen3.5-4B** | `Qwen/Qwen3.5-4B` | `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` | `8,424,393,216` | `afc3aaa3f844dda41af4012479d7f9eac87eda553c16bf2795e85ebc1e0e0bf7` | **6 / 6** |
| **Gemma 4 E2B** | `google/gemma-4-E2B-it` | `4adf15748de9e1a0097b6274cee9c15836c41c37` | `9,311,299,104` | `01ef9dcdba1a648bfb68a4bc2c1b91632f59ad37d48ae81887e51305f9bb0ba6` | **6 / 6** |
| **Qwen3-4B** | `Qwen/Qwen3-4B` | `1cfa9a7208912126459214e8b04321603b3df60c` | `8,051,284,928` | `99038d3c282d8249b9804a94aea50fd3b30c6b5d56931474042ede9982602337` | **6 / 6** |
| **Qwen3.8-27B** | `Qwen/Qwen3.8-27B` | `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` | `54,657,734,080` *(F16)* | `05f18ee2b177c76c34a827a1bcada598a9bfd99f939ddcb8b37a9e47874f6599` | **2** *(imported)* |
| **Qwen3-8B** | `Qwen/Qwen3-8B` | `main` | — | — | **2** *(imported)* |

*(BF16 GGUFs are produced with `convert_hf_to_gguf.py --outtype bf16` from `llama.cpp` commit `a01dff9` patched with [`../mvp/patches/llama-cpp-a01dff9-chantless.patch`](../mvp/patches/llama-cpp-a01dff9-chantless.patch).)*

---

## 1. Specialist Extraction Masks (`*-mask.json`, `9 / 9`)

All `9` specialist masks verified `cli_rebuild_sha256_equal == true`.

| Mask File | Mask Size | Built GGUF Bytes (`%` BF16, `bpw`) | Built GGUF SHA256 | Vocab Kept | Locked Exam Retention | Uniform Baseline | Gain vs Uniform |
| :--- | :---: | :---: | :--- | :---: | :---: | :---: | :---: |
| `qwen35-4b-clsbench-mask.json` | `48,533 B` | `837,246,880` (`9.94%`, `1.59`) | `3bfd223b1b6cae8db60b849a9e4227bfa0f5fead19758e1e63062b1b749b38db` | `40,960` | **`93.40%`** (`2923` src-ok / `3000`) | `46.66%` (`IQ1_S`, `10.93%`) | **`+46.73` pts** |
| `qwen35-4b-bfcl-mask.json` | `18,606 B` | `2,191,129,568` (`26.01%`, `4.17`) | `6f5f75915bc36aa7b36642f69ca4a470e36933ad0771ae70e78a8795d4031e73` | `132,000` | **`92.39%`** (`460` src-ok / `869`) | `92.39%` (`Q3_K`, `21.62%`) | `+0.00` pts |
| `qwen35-4b-deepmath-mask.json` | `18,606 B` | `2,191,747,808` (`26.02%`, `4.17`) | `2c86b60af364e2abc4f3f386acb9d560ad7f4561a70a4d1e22cd1d91a04b6f24` | `132,000` | **`82.27%`** (`141` src-ok / `511`) | `51.77%` (`Q3_K`, `21.62%`) | **`+30.50` pts** |
| `gemma4-e2b-clsbench-mask.json` | `47,695 B` | `928,677,056` (`9.97%`, `1.60`) | `31fb63cef139ff5e757ced6f15a472f1dd1a8563b58e0869af85e9ab6e3df7e5` | `51,999` | **`99.21%`** (`2919` src-ok / `3000`) | `49.43%` (`IQ1_S`, `14.08%`) | **`+49.78` pts** |
| `gemma4-e2b-bfcl-mask.json` | `47,131 B` | `1,947,955,360` (`20.92%`, `3.35`) | `2504a2d26a8ba2df6b90fb3916dcac093ac1d609b1517f3e71f5ee79fcb68e9b` | `155,000` | **`97.13%`** (`418` src-ok / `869`) | `53.11%` (`Q2_K`, `16.74%`) | **`+44.02` pts** |
| `gemma4-e2b-deepmath-mask.json` | `43,354 B` | `2,127,669,568` (`22.85%`, `3.66`) | `236ddc5a743f76cb1dae910ed8cee16fdd0864641b6642a74abdd020ab12790e` | `155,000` | **`75.00%`** (`28` src-ok / `511`) | `0.00%` (`Q3_K`, `21.77%`) | **`+75.00` pts** |
| `qwen3-4b-clsbench-mask.json` | `41,241 B` | `800,227,040` (`9.94%`, `1.59`) | `1f17932781eb1e10ab7a2882822c7f4a1e870a0c5506d5f07001ccd606c0ab6f` | `20,480` | **`60.96%`** (`2907` src-ok / `3000`) | `45.44%` (`IQ1_S`, `10.48%`) | **`+15.51` pts** |
| `qwen3-4b-bfcl-mask.json` | `6,419 B` | `1,972,223,392` (`24.50%`, `3.92`) | `f8a2bf55fff5bbb388ce52f8a34ac2a6383ca4ed90426f96bb04b60f0fe3a00a` | `98,000` | **`92.31%`** (`494` src-ok / `869`) | `86.44%` (`Q3_K`, `21.55%`) | **`+5.87` pts** |
| `qwen3-4b-deepmath-mask.json` | `6,424 B` | `2,172,241,312` (`26.98%`, `4.32`) | `7bef3745f4c79808ad2e75c0c5070e075c7118a51626aa286a7fee7f4be76841` | `98,000` | **`76.62%`** (`201` src-ok / `511`) | `67.66%` (`Q3_K`, `21.55%`) | **`+8.96` pts** |

---

## 2. Zero-Extra-Byte Switch Apprentice Masks (`*-apprentice-mask.json`, `9 / 9`)

Each Apprentice mask starts from its specialist mask above, trains binary `up_row_zero` MLP neuron switches on `25` specialist mistakes (`n_train_lessons = 25`) with full-vocabulary `float32` `replay_h0` token-KL protection, and adds **`0` bytes** to the built GGUF (`bytes_unchanged == true`, `cli_rebuild_sha256_equal == true`). On every cell below, the `empty` control turned off `0` switches (`empty.sha256 == base.sha256`).

| Mask File | Mask Size | Built GGUF Bytes | Built GGUF SHA256 | Switches Off | Unseen Fixed (`learned`) | Controls (`base / empty / flip / shuf`) | `sel1` Retention (`learned` vs `base`) |
| :--- | :---: | :---: | :--- | :---: | :---: | :---: | :---: |
| `qwen35-4b-clsbench-apprentice-mask.json` | `49,324 B` | `837,246,880` | `a508d4ca543813a98beef72150af2baf18aacd584ff8819f57cb9b25ca65c490` | `484 / 22,016` | **`33 / 60`** (`55.0%`) | `4 / 4 / 5 / 20` | **`92.79%`** vs `88.72%` |
| `qwen35-4b-bfcl-apprentice-mask.json` | `21,340 B` | `2,191,129,568` | `91b599e23fb3962e53b27ed19b3a0ce617fc4efb99170c5ea4b687f9e826f20f` | `1,575 / 73,728` | **`29 / 60`** (`48.3%`) | `0 / 0 / 2 / 19` | **`83.22%`** vs `93.96%` |
| `qwen35-4b-deepmath-apprentice-mask.json` | `21,308 B` | `2,191,747,808` | `322d00e5add88d82c5050881947cf7e63bd75c466bd934f309a5b6c72938534a` | `1,465 / 73,728` | `9 / 60` (`15.0%`) | `10 / 8 / 10 / 11` | `61.67%` vs `68.33%` |
| `gemma4-e2b-clsbench-apprentice-mask.json` | `49,344 B` | `928,677,056` | `c599e629739952408f6fc72bbe098a5a70c61ced61a9045a5c78d35aecb2475f` | `1,098 / 63,744` | **`15 / 60`** (`25.0%`) | `3 / 3 / 2 / 8` | **`99.40%`** vs `99.15%` |
| `gemma4-e2b-bfcl-apprentice-mask.json` | `52,754 B` | `1,947,955,360` | `16372496f910c3846ef672d4904e15379e40daabcc954dc6adafd12b72a2d7fa` | `4,545 / 92,160` | **`26 / 60`** (`43.3%`) | `0 / 0 / 0 / 7` | **`83.33%`** vs `99.21%` |
| `gemma4-e2b-deepmath-apprentice-mask.json` | `46,006 B` | `2,127,669,568` | `caab0cf1406505d3e11dc4cc23c07be5d4081525cbff3a108b2469b75d4f76a2` | `1,665 / 98,304` | **`10 / 60`** (`16.7%`) | `3 / 3 / 4 / 4` | **`80.00%`** vs `40.00%` |
| `qwen3-4b-clsbench-apprentice-mask.json` | `42,677 B` | `800,227,040` | `4a94810541faa4aa93b6b5a785930733bb8a57861a6008e715de93805f01a6f2` | `1,102 / 21,248` | **`16 / 60`** (`26.7%`) | `4 / 4 / 4 / 15` | **`49.70%`** vs `43.38%` |
| `qwen3-4b-bfcl-apprentice-mask.json` | `9,617 B` | `1,972,223,392` | `07dfb251b69c3edc9c9366d78a641f6c35a507fcec4ef1f2ae5888229835d81e` | `2,057 / 77,824` | **`22 / 60`** (`36.7%`) | `3 / 3 / 1 / 12` | **`87.29%`** vs `95.58%` |
| `qwen3-4b-deepmath-apprentice-mask.json` | `6,830 B` | `2,172,241,312` | `51bf233ef3832299ba2d76b66bba7651146fac4975b0b95f39e4f50d00b3ada3` | `114 / 77,824` | `20 / 49` (`40.8%`) | `9 / 9 / 14 / 20` | `61.33%` vs `72.00%` |

*Control notes:*
- On `qwen35-4b-clsbench`, all `25/25` specialist training mistakes shared one class label (`"ai"`), so permuting those labels reproduced the `learned` mask (`33/60` each); the `20/60` `shuffled` baseline above uses balanced 50/50 random labels (`12` `"ai"`, `13` `"human"`).
- On `qwen35-4b-bfcl` and `qwen35-4b-deepmath`, `shuffled` permutes `(corrected_tokens, mistaken_tokens)` pairs together across prompts rather than retaining prompt `i`'s true mistaken trajectory.

---

## 3. Imported Dynamic-Quantization Winners (`*-dynquant-*-mask.json`, `4 / 4`)

Historical mixed-precision dynamic-quantization winners imported into native `chantless.mask/v3` via `chantless mask import` (supporting all `22` native `llama-quantize` ggml tensor types). Each mask preserves its original source pin, archive URLs, and recorded evaluation receipts (including integrity caveats) in `provenance` and `measured`, and works natively with `chantless mask {inspect, diff, fork, merge}`, `chantless build`, `chantless serve/bench`, and `chantless learn`.

| Mask File | Mask Size | Groups / Overrides | Tier Mix | Original Artifact Size / SHA256 | Recorded Evaluation Receipt |
| :--- | :---: | :---: | :--- | :--- | :--- |
| `qwen38-27b-dynquant-k34-mask.json` | `9,662 B` | `130` / `42` | `95 Q4_K`, `34 Q8_0`, `1 Q6_K` | `19,603,900,096 B` (`35.87%` F16), `62a2d8ae...` | Terminal-Bench 2.1: **`77.8%` pass@1** (`277/356` audited; `78.1%` stored), **`91.0%` pass@4** (`81/89`) |
| `qwen38-27b-dynquant-k94-mask.json` | `9,553 B` | `130` / `38` | `94 Q4_K`, `35 Q2_K`, `1 Q6_K` | `14,262,798,016 B` (`26.09%` F16), `2184c613...` | Terminal-Bench 2.1: **`74.4%` pass@1** (`265/356` audited; `74.7%` stored), **`85.4%` pass@4** (`76/89`) |
| `qwen3-8b-dynquant-iq2m-k08-mask.json` | `7,548 B` | `74` / `64` | `64 IQ2_S`, `8 Q4_K`, `1 Q5_K`, `1 IQ3_S` | `day6-lowbit-dyn/A_k08_iq2m_uns` | BFCL (`1,007` ex): **`719/1007` (`71.40%`)**, `+23` vs `k00` (`696/1007`), matches `UD-IQ2_M` (`720/1007`) |
| `qwen3-8b-dynquant-iq2m-ud-mask.json` | `8,037 B` | `74` / `84` | `57 IQ2_S`, `10 IQ3_XXS`, `5 Q4_K`, `1 Q5_K`, `1 Q3_K` | `day6-lowbit-dyn/udrecipe_ourbuild` | BFCL (`1,007` ex): **`723/1007` (`71.80%`)**, `+27` vs `k00`, `+3` vs upstream `UD-IQ2_M` |

---

## Rebuilding or Importing Any Mask

1. Build `llama.cpp` at commit `a01dff9` with [`../mvp/patches/llama-cpp-a01dff9-chantless.patch`](../mvp/patches/llama-cpp-a01dff9-chantless.patch) and install Chantless (`pip install -e .`).
2. Convert the stock Hugging Face checkpoint to BF16 GGUF (`convert_hf_to_gguf.py --outtype bf16`).
3. Materialize the standalone GGUF:

```bash
# Generative masks (Q2_K..Q8_0 only — no imatrix required):
chantless build \
  --source qwen3-4b-bf16.gguf \
  --mask masks/qwen3-4b-bfcl-apprentice-mask.json \
  --llama-bin /path/to/llama.cpp/build/bin \
  --out qwen3-4b-bfcl-apprentice.gguf

# Gemma 4 masks pass --hf-dir so shared-KV donor layers are resolved from config.json:
chantless build \
  --source gemma4-e2b-bf16.gguf \
  --hf-dir /path/to/gemma-4-E2B-it \
  --mask masks/gemma4-e2b-bfcl-apprentice-mask.json \
  --llama-bin /path/to/llama.cpp/build/bin \
  --out gemma4-e2b-bfcl-apprentice.gguf

# <=10% classifier masks use IQ1_S / IQ2_XS tiers and therefore take --imatrix:
chantless build \
  --source qwen35-4b-bf16.gguf \
  --imatrix qwen35-imatrix.gguf \
  --mask masks/qwen35-4b-clsbench-apprentice-mask.json \
  --llama-bin /path/to/llama.cpp/build/bin \
  --out qwen35-4b-clsbench-apprentice.gguf
```
