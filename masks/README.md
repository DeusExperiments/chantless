# Chantless Thin Masks (`chantless.mask/v3`)

This directory tracks **`16` verified winner masks** (`6.4 KB – 52.8 KB` pure Apache-2.0 JSON, zero trained floats):
- **`12` stock-checkpoint double-winner masks** (`6` architecture/task cells across `4` architectures — `Gemma 4`, `Qwen3.5`, `Qwen3`, and `26.4B` looped-MoE `Inkling-2-Mini` — × `{specialist, +0 B Apprentice}`) where extraction cleared its quality gate **and** zero-byte `<=25`-mistake Apprentice learning beat every matched-byte control (`base`, `empty`, `flipped`, `shuffled`).
- **`2` additional looped-MoE `Inkling-2-Mini` (`26.4B`) specialist winners** (`bfcl`, `deepmath`) verified on sealed exams plus `reasoning_effort="high"` (`0.9`, `16k` math / `8k` tool-calling) stress tests.
- **`2` flagship `27B` imported winners** (`Qwen3.8-27B` `K34` and `K94`), demonstrating `chantless mask import` interoperability at `27B` scale.

## Pinned Stock Source Checkpoints

| Family | Source Repo | Pinned Revision | Base GGUF Bytes | Base GGUF SHA256 | Masks |
| :--- | :--- | :--- | :---: | :--- | :---: |
| **Gemma 4 E2B** | `google/gemma-4-E2B-it` | `4adf15748de9e1a0097b6274cee9c15836c41c37` | `9,311,299,104` *(BF16)* | `01ef9dcdba1a648bfb68a4bc2c1b91632f59ad37d48ae81887e51305f9bb0ba6` | `4` |
| **Qwen3.5-4B** | `Qwen/Qwen3.5-4B` | `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` | `8,424,393,216` *(BF16)* | `afc3aaa3f844dda41af4012479d7f9eac87eda553c16bf2795e85ebc1e0e0bf7` | `4` |
| **Qwen3-4B** | `Qwen/Qwen3-4B` | `1cfa9a7208912126459214e8b04321603b3df60c` | `8,051,284,928` *(BF16)* | `99038d3c282d8249b9804a94aea50fd3b30c6b5d56931474042ede9982602337` | `2` |
| **Inkling-2-Mini** | `thinkingmachines/Inkling-2-Mini` | `ed40e7da625f594fc9c4990f5f66be24873c22d7` | `52,226,509,504` *(BF16)* | `7074c9762e6769a608c7b23960a7fd2444880b1174a3910149749f6d01389aec` | `4` |
| **Qwen3.8-27B** | `Qwen/Qwen3.8-27B` | `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` | `54,657,734,080` *(F16)* | `05f18ee2b177c76c34a827a1bcada598a9bfd99f939ddcb8b37a9e47874f6599` | `2` *(imported)* |

---

## 1. Stock-Checkpoint Specialist Winners (`5 / 5`, bit-exact CLI rebuilds)

| Mask File | Mask Size | Built GGUF (`%` BF16, `bpw`) | Built GGUF SHA256 | Vocab Kept | Seen `select` | Held-Out `exam` | Gain vs Uniform |
| :--- | :---: | :---: | :--- | :---: | :---: | :---: | :---: |
| `gemma4-e2b-clsbench-mask.json` | `47,695 B` | `928.7 MB` (**`9.97%`**, `1.60`) | `31fb63cef139ff5e757ced6f15a472f1dd1a8563b58e0869af85e9ab6e3df7e5` | `51,999` | `99.08%` | **`99.21%`** | **`+49.78` pts** (`IQ1_S`) |
| `qwen35-4b-clsbench-mask.json` | `48,533 B` | `837.2 MB` (**`9.94%`**, `1.59`) | `3bfd223b1b6cae8db60b849a9e4227bfa0f5fead19758e1e63062b1b749b38db` | `40,960` | `88.89%` | **`93.40%`** | **`+46.73` pts** (`IQ1_S`) |
| `gemma4-e2b-bfcl-mask.json` | `47,131 B` | `1.948 GB` (`20.92%`, `3.35`) | `2504a2d26a8ba2df6b90fb3916dcac093ac1d609b1517f3e71f5ee79fcb68e9b` | `155,000` | `97.98%` | **`97.13%`** | **`+44.02` pts** (`Q2_K`) |
| `qwen3-4b-bfcl-mask.json` | `6,419 B` | `1.972 GB` (`24.50%`, `3.92`) | `f8a2bf55fff5bbb388ce52f8a34ac2a6383ca4ed90426f96bb04b60f0fe3a00a` | `98,000` | `95.03%` | **`92.31%`** | **`+5.87` pts** (`Q3_K`) |
| `qwen35-4b-bfcl-mask.json` | `18,606 B` | `2.191 GB` (`26.01%`, `4.17`) | `6f5f75915bc36aa7b36642f69ca4a470e36933ad0771ae70e78a8795d4031e73` | `132,000` | `93.92%` | **`92.39%`** | `+0.00` pts (`Q3_K`) |

---

## 2. Zero-Extra-Byte Switch Apprentice Winners (`6 / 6` promoted across `4` architectures, `+0 B`, identity control verified)

Each Apprentice mask trains binary `up_row_zero` MLP neuron switches on `25` specialist mistakes with full-vocabulary `float32` `replay_h0` token-KL protection and adds **`0 bytes`** to the built GGUF (bit-exact CLI rebuilds):

| Mask File | Mask Size | Built GGUF SHA256 | Switches Off | Unseen Fixed (`learned`) | Controls (`base / empty / flip / shuf`) | Retention (`learned` vs `base`) |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: |
| `qwen35-4b-clsbench-apprentice-mask.json` | `49,324 B` | `a508d4ca543813a98beef72150af2baf18aacd584ff8819f57cb9b25ca65c490` | `484 / 22,016` | **`33 / 60`** (`55.0%`) | `4 / 4 / 5 / 20` | `sel1` **`92.79%`** vs `88.72%` |
| `qwen35-4b-bfcl-apprentice-mask.json` | `21,340 B` | `91b599e23fb3962e53b27ed19b3a0ce617fc4efb99170c5ea4b687f9e826f20f` | `1,575 / 73,728` | **`29 / 60`** (`48.3%`) | `0 / 0 / 2 / 19` | `sel1` `83.22%` vs `93.96%` |
| `gemma4-e2b-bfcl-apprentice-mask.json` | `52,754 B` | `16372496f910c3846ef672d4904e15379e40daabcc954dc6adafd12b72a2d7fa` | `4,545 / 92,160` | **`26 / 60`** (`43.3%`) | `0 / 0 / 0 / 7` | `sel1` `83.33%` vs `99.21%` |
| `qwen3-4b-bfcl-apprentice-mask.json` | `9,617 B` | `07dfb251b69c3edc9c9366d78a641f6c35a507fcec4ef1f2ae5888229835d81e` | `2,057 / 77,824` | **`22 / 60`** (`36.7%`) | `3 / 3 / 1 / 12` | `sel1` `87.29%` vs `95.58%` |
| `gemma4-e2b-clsbench-apprentice-mask.json` | `49,344 B` | `c599e629739952408f6fc72bbe098a5a70c61ced61a9045a5c78d35aecb2475f` | `1,098 / 63,744` | **`15 / 60`** (`25.0%`) | `3 / 3 / 2 / 8` | `sel1` **`99.40%`** vs `99.15%` |
| `inkling2-mini-clsbench-apprentice-mask.json` | `33,907 B` | `fdd710884e53c5e3320e0930e1bf20ddd03ccfdc0effdbda9a2170cf0f5e055a` | `424 / 6,144` | **`10 / 60`** (`16.7%`) | `4 / 4 / 3 / 8` | `exam` **`98.67%` @ `5.00%` FPR** vs `98.33%` @ `5.33%` |

*(Control protocol: on `qwen35-4b-clsbench`, all 25 training mistakes shared one class label, so `shuffled` uses balanced 50/50 random labels; on `qwen35-4b-bfcl`, `shuffled` permutes `(corrected, mistaken)` pairs across prompts. On `Inkling-2-Mini / BFCL`, `+0 B` `up_row_zero` switching on `blk.18..23` verified bit-exact rebuild (`860b668e518a`, `439/6,912` switches off) and improved `select80` (`55/80` vs `53/80` Gorilla), but tied the `flipped` control on unseen fixes (`6/46` vs `6/46` Gorilla; `5/46` vs `5/46` Chantless), so the 4-control identity gate honestly rejected promoting a BFCL Apprentice mask.)*

---

## 3. Looped-MoE `Inkling-2-Mini` (`26.4B` / `128` Experts) Specialist Winners (`3 / 3` + `1` Double Winner)

Built against `ggml-org/Inkling-2-Mini-GGUF` (`Inkling-2-Mini-BF16.gguf`, `48.640 GiB`) and unified `llama.cpp @ afe77e18fadc` (`#32704` + `#32763`, `--special`, `f16` KV, `--no-cache-prompt`).

| Mask File | Mask Size | Built GGUF (`%` BF16, `bpw`) | Built GGUF SHA256 | Experts / Width / Vocab | Held-Out Sealed `exam` (vs `BF16`) | `reasoning_effort="high"` (`0.9`) / Apprentice |
| :--- | :---: | :---: | :--- | :---: | :--- | :--- |
| `inkling2-mini-clsbench-mask.json` | `27,859 B` | `4.17 GiB` (**`8.582%`**, `1.37`) | `d884cb419ae59ca5610c35c377183ba66e12f049397648ee80e817cef34c9861` *(hardened, bit-exact)* | `48/128` · `w=1024` · `v=32,768` | **`98.33%` recall** @ `5.33%` FPR, `AUROC=0.9965` (`n=1,200`, `BF16` `99.17%` @ `5.00%`) | **`+0 B` Apprentice (`fdd710884e53`):** `10/60` unseen vs `4/4/3/8`; `exam` **`98.67%` @ `5.00%` FPR** |
| `inkling2-mini-bfcl-mask.json` | `8,990 B` | `14.22 GiB` (**`29.233%`**, `4.68`) | `8ac87bdef6be2584fd939d9d187e423f760ce2e5a5f3a3af5c392625117c7d3f` *(0 B err, 4× bit-exact)* | `128/128` · `w=1152` · full | **`115/160` (`71.88%`, `+0.63 pp`)** Gorilla `ast_checker` (`122/160` Chantless; `BF16` `114/160`) | **BFCL-HIGH (`8k`):** `53/79` Gorilla tie (`BF16` `53/79`); **DeepMath-HIGH (`16k`):** **`41/48` (`0` regressions)**, `KL=0.00990` |
| `inkling2-mini-deepmath-mask.json` | `8,435 B` | `12.48 GiB` (**`25.663%`**, `4.11`) | `94c08b968242042a89b20b2c31bc1122fd47e6fc3b404b4012b08d32864c5eef` *(32 B err)* | `96/128` · `w=1152` · full | **`81/96` (`84.38%`, `-3.12 pp`)** at `"low"` `4k` (`BF16` `84/96`, `1.45×` faster) | **DeepMath-HIGH (`16k`):** **`40/48` strict / `41/48` raw** (`both=40, bf16_only=1`), `KL=0.01343` |

*(Notes: `inkling2-mini-bfcl-mask.json` uses fixed HF `imatrix.gguf` and reproduced bit-identically across Colab G4 and Lium pods (`4×`); `inkling2-mini-clsbench-mask.json` and `inkling2-mini-clsbench-apprentice-mask.json` were built on the hardened `BF16`-pinned `conv_proj` builder (`+552,960 B` vs pre-pinning) and verified bit-exact (`cli_rebuild_sha256_equal=true`); `inkling2-mini-deepmath-mask.json` uses task-calibrated `imatrix_dm_task.gguf` whose GPU rebuilds vary slightly in SHA256. On `BFCL-HIGH`, all models hit an identical upstream `llama.cpp @ afe77e18` grammar exception HTTP 500 on `live_simple:live_simple_231-122-0`, giving `79` valid paired tasks.)*

---

## 4. Imported `27B` Dynamic-Quantization Winners (`2 / 2`, via `chantless mask import`)

| Mask File | Mask Size | Groups / Overrides | Tier Mix | Original Artifact Size / SHA256 | Recorded Terminal-Bench 2.1 Receipt |
| :--- | :---: | :---: | :--- | :--- | :--- |
| `qwen38-27b-dynquant-k34-mask.json` | `9,662 B` | `130` / `42` | `95 Q4_K`, `34 Q8_0`, `1 Q6_K` | `19.60 GB` (`35.87%` F16), `62a2d8ae...` | **`77.8%` pass@1** (`277/356`) · **`91.0%` pass@4** (`81/89`) |
| `qwen38-27b-dynquant-k94-mask.json` | `9,553 B` | `130` / `38` | `94 Q4_K`, `35 Q2_K`, `1 Q6_K` | `14.26 GB` (`26.09%` F16), `2184c613...` | **`74.4%` pass@1** (`265/356`) · **`85.4%` pass@4** (`76/89`) |

---

## Rebuilding, Benchmarking, or Importing Any Mask

```bash
# Build runtime & check environment
./scripts/build_toolchain.sh --backend auto --prefix .local/toolchain
chantless doctor --llama-bin .local/toolchain/llama-afe77e18/build/bin --traj-kl .local/toolchain/bin/traj-kl

# Deterministic SHA256 rebuild (pass --hf-dir for Gemma 4; pass --imatrix for IQ tiers)
chantless build \
  --source qwen3-4b-bf16.gguf \
  --mask masks/qwen3-4b-bfcl-apprentice-mask.json \
  --llama-bin .local/toolchain/llama-afe77e18/build/bin \
  --out qwen3-4b-bfcl-apprentice.gguf

# Serve or benchmark directly from stock BF16 + mask (SHA256-cached auto-build)
chantless bench \
  --source qwen3-4b-bf16.gguf \
  --mask masks/qwen3-4b-bfcl-apprentice-mask.json \
  --compare qwen3-4b-bf16.gguf \
  --llama-bin .local/toolchain/llama-afe77e18/build/bin

# Import any existing mixed-precision GGUF into chantless.mask/v3
chantless mask import --from-gguf built.gguf --source-gguf base-bf16.gguf --out imported-mask.json
```
