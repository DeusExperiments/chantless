# Search pilot — plan (2026-09-27)

This is exploratory and makes no claims. It rehearses the §8 loop on the 20-task pilot data. The manual (commit 64fc898) is unchanged. Every deviation from it is listed below and was approved by the owner in chat on 2026-09-27.

## Scope
- **Data:** the 16 valid pilot tasks. Four tasks (000543, 001572, 002518, 003021) are excluded because their images leak future git history.
- **Proxy:** the 15 passing BF16 trajectories from pilot run1.
- **Action space:** precision tiers per layer-group (66 groups) plus KV type. No drops and no pruning, so stock llama.cpp suffices.
- **Start:** uniform Q4_K. Search downward to about 1.6 bpw average.
- **Ladder:** Q4_K → Q3_K → IQ2_XS → IQ1_M → IQ1_S.
- **KL:** a new whole-trajectory tool (L5, `tools/traj-kl/`), built first. One forward pass per trajectory; KL(source with f16 KV ‖ candidate).
  - **Targets:** tokens the model generated, excluding reasoning. Everything from `<think>` through `</think>` and the whitespace after it is unscored. Scoring resumes at the first non-whitespace token and runs through the end-of-turn token. Turns rendered without a think block are scored whole. (owner)
  - **Subsample:** a fixed uniform subsample of 20,000 targets across the 15 passing trajectories (seed 20260927). Every candidate is scored on the same targets, so move-vs-move comparisons are paired. (owner)
  - **Grounded choices:**
    - fp16 log-prob store, per the manual's 0.5 MB/token budget.
    - Vocab terms with p < e^-30 are ignored; their effect is bounded below 1e-8.
    - The template and tokenizer come from the source llama-server. Any prefix mismatch rejects that trajectory.
- **Evaluation:** 16 tasks × 4 runs per model.
  - Search masks at about 3.5, 2.75, 2.0 and 1.6 bpw.
  - Uniform Q4_K_M, the starting point.
  - dynquant (K94 recipe) at about 3.5 and 2.75 bpw.
  - Uniform IQ2_XS near 2.3 bpw.
  - Uniform IQ1_M / IQ1_S at the bottom.
- **Compute:** Lium 8×A100 `onlyskills-measure` (noble-shark-39).

## Deviations from the manual
| Manual | Pilot | Why |
|---|---|---|
| D9/D10: reliable pool (>= 6/8), repo split | 16 pilot tasks, no split | owner: "move on using the data from 20" |
| D11/D12: gates, screens | none; descriptive comparison only | too few tasks for gates |
| D6 ladder (Q8_0 … IQ1_M, drop) | Q4_K … IQ1_S, no drop | owner: "start from 4 bit, go to 1.5"; IQ1_S added |
| D7 removal | none | no fork yet |
| §8.3 start at BF16 | start at uniform Q4_K | owner |
| §6.4 sanity cap = KL of stock Q4_K_M | **size-matched cap**: general KL(mask) <= general KL of a uniform quant at the same average bpw (piecewise-linear between uniform Q4_K, Q3_K, IQ2_XS, IQ1_M, IQ1_S) | the literal cap forbids nearly every move below 4-bit (owner) |
| §5.4 general_proxy = FineWeb-Edu + code | **FineWeb-Edu only**: 50k scored tokens, 2,048-token packed sequences, positions 1,024..2,047 scored | owner: small, English |
| §5.4 proxy targets = assistant tokens | assistant tokens minus reasoning; 20k fixed subsample | owner |
| split / held-out | proxy trajectories come from 13 of the 16 eval tasks: **known leakage, pilot limitation** | owner accepted |
| §9.1 L5 test: KL(source, source) = 0 exactly | fp16 store; pass if self-KL <= 1e-5 (measured 3.6e-6, i.e. 0.01% of uniform Q4_K's 0.0354) | owner |
| (mechanical, measured) | traj-kl `--ubatch` pinned at 2048 for the base pass and every probe: KL depends on ubatch (uniform Q4_K last trajectory: 0.04803 at 2048 vs 0.05249 at 8192) because llama.cpp picks matmul kernels by batch size | consistency |
