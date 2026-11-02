# Chantless v1 — Experiment Manual

**No strings. Just weights.** · Deus Experiments · Token Bender
Version 1.0 · 2026-09-27 · Status: specification frozen for experiment 1; nothing here is a result.

> Current project name: Chantless (October 5, 2026). This is the inherited, paused MiMo experiment specification. Historical receipt names and the legacy `onlyskills.mask/v1` schema are preserved; renaming does not change the experimental contract or establish new results.

This manual turns the Chantless design record (`../explainer/design-notes.md`, Google Doc
`1g6kRZrKkkcw21lUh0zvZQnE8h_a6VMk-pazBdCAq2XI`) into a runnable procedure. Every choice that
was open in the design record is pinned here. "MUST" marks a rule the run is invalid without;
"SHOULD" marks a default that can change if the change is recorded in the run log (§15).

---

## 0. The one-paragraph version

Take `MiMo-V2.6-Distill-Qwen-9B`. Pick the MiMo v2.6 SWE tasks it reliably solves. Split them
by repository into search / select / exam. Starting from the full-precision model, a greedy
search lowers the precision of one layer-group at a time (or thins its channels, or drops it),
always choosing the move that costs the least behaviour per byte saved, where "behaviour" is
KL against the full model on its own successful SWE trajectories. Every 8 moves it re-measures
everything and runs 50 real agent episodes as a smoke test; periodically it runs the full
statistical gate. The search returns the latest mask that passed the gate. That mask is built into
one standalone GGUF and sat once on the exam split. The claim is frontier dominance: it stays
non-inferior to the full model at less 128k-context memory than any point of our own dynquant
(K94 recipe, same model, data and KV type) that also stays non-inferior.

```text
MiMo-V2.6-Distill-Qwen-9B (BF16 GGUF)          MiMo v2.6 code env (2,698 SWE tasks)
            |                                              |
            |                         reliability screen (>=6/8 BF16 passes)
            |                                              |
            |                           repo-keyed split: search | select | exam
            v                                              |
   +--------------------------- mask search ---------------v-----------------+
   |  proxy:  KL(BF16 || candidate) on BF16's passing search trajectories   |
   |  guard:  general-data KL <= stock Q4_K_M  (hard cap, every move)       |
   |  move:   one group one step down (tier / keep-fraction / drop / KV)    |
   |  every 8 moves: full re-probe + 50-episode screen (select split)       |
   |  every 32 moves: full non-inferiority gate (select split)              |
   |  stop: first failed screen/gate -> latest gate-passing mask            |
   +---------------------------------+---------------------------------------+
                                     v
                build one GGUF -> exam split (single use) -> compare
                                     v
   claim: non-inferior to BF16 at less 128k memory than any passing dynquant
```

---

## 1. Pinned decisions (the spec)

| # | Question | v1 answer | Source |
|---|---|---|---|
| D1 | Source model | `XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B` (Qwen3.5 hybrid, 9.4B incl. vision/MTP) | user, 2026-09-27 |
| D2 | Environment | MiMo v2.6 RL-oss, `code` config (SWE, executable tests) | user |
| D3 | Runtime | llama.cpp / GGUF (patched fork, §9.1) | user |
| D4 | Objective | minimise **total memory at 128k context, batch 1**; KV cache type is searchable | user |
| D5 | Mask unit | per **layer-group**: each layer's mixer and FFN, plus `token_embd` and `output` | user |
| D6 | Tier ladder | Q8_0, Q6_K, Q5_K, Q4_K, Q3_K, IQ2_XS, IQ1_M, drop | user |
| D7 | Removal | vision tower + MTP always removed; whole-block drop; per-group keep-fraction for FFN channels / heads | user |
| D8 | Fixed parts | norms, SSM scalars (`A_log`, `dt`, conv1d) stay F32/source type; not in the mask | user |
| D9 | Task pool | tasks the BF16 source passes >= 6 of 8 runs (full 8 runs on every task) | user |
| D10 | Splits | search / select / exam, 60/20/20, **no repository shared across splits** | user |
| D11 | Acceptance | task-matched non-inferiority, margin 3 pts, one-sided 95%, **95% power** | user |
| D12 | Eval tiers | proxy every move -> 50-episode screen every 8 moves -> full gate every 32 moves and on the final candidate | user |
| D13 | Decoding | `generation_config.json`: temperature 0.6, top_p 0.95, top_k 20; per-request seeds; source/candidate matched by task | user (0.6 confirmed over MiMo RL's 1.0) |
| D14 | Proxy | KL on BF16's passing search trajectories + on-policy KL at every screen | user |
| D15 | Learner | iterative greedy with re-probing (Dettmers-style) | user |
| D16 | Re-probe cadence | full re-probe every 8 accepted moves; top-16 cached moves re-measured each move | user |
| D17 | Sanity | global hard cap: general-data KL <= that of stock uniform Q4_K_M | user |
| D18 | Mode | mask-only, preserve (weights frozen) | user |
| D19 | Detectors | doom loop, malformed tool call, premature stop, token garbage — all gate | user |
| D20 | Reference comparison | KL is a proxy only, never a gate | user |
| D21 | Baseline | our dynquant, K94 recipe, same calibration tokens, **same KV type as the final mask** | user |
| D22 | Budget | no fixed cap for run 1; cost is measured and reported | user |
| D23 | Candidate evaluation | proxy computed on **real GGUF builds** via llama.cpp, no fake-quant | user |
| D24 | Compute | Lium. P0/pilot on 1x RTX PRO 6000 Blackwell 96 GB ($1.29/h); scale to 8xA100 for P2/P5 (Pipeshift on hold: host key changed) | user |
| D25 | Harness | `mimocode-agent` only (native MiMo-Code scaffold), run via the standalone `XiaomiMiMo/mimoagent` CLI with its local Docker backend; verl/Ray/Kubernetes not used | user |
| D26 | Episode caps | MiMo defaults: step_limit 500, bash timeout 60 s (max 300), env timeout 300 s, verifier timeout per task | user |
| D27 | Stopping | stop at the first failed screen or failed gate; return the latest mask with a passing gate (§8.3) | user |
| D28 | Screen rule | 50 source + 50 candidate episodes, task-matched; fail if candidate is > 10 pts below source | user |
| D29 | Pruning score | activation-weighted magnitude (Wanda-style) on search trajectories | user |
| D30 | Headline claim | **frontier dominance**: on the exam split, Chantless is non-inferior to the source, and every dynquant point at <= its 128k memory is shown worse than the margin (§10.3) | user |

Values in §4–§8 that the user did not choose explicitly are marked **[default]** and listed
again in §16 for review.

---

## 2. Vocabulary

| Term | Meaning |
|---|---|
| **group** | the unit of the mask: `blk.i.mixer` (DeltaNet or attention), `blk.i.ffn`, `token_embd`, `output`. 66 groups. |
| **tier** | the ggml quant type assigned to every weight matrix in a group, or `drop`. |
| **keep-fraction** | the fraction of a group's prunable structure (FFN channels, heads) that is kept. |
| **KV type** | the llama.cpp cache type for K and V (`f16`, `q8_0`, `q4_0`); a global, searchable setting. |
| **mask** | one (tier, keep-fraction) per group + one KV type. A JSON file (§11). |
| **move** | changing one mask entry one step toward cheaper (§4.4). |
| **candidate** | the GGUF built from a mask. |
| **source** | the BF16 GGUF of the 9B text model, run in the same llama.cpp fork. |
| **episode** | one `mimo_swe_agent` run on one task with one seed, ending in a test verdict. |
| **episode success** | tests pass **and** no detector fires (§6.2). |
| **pair** | a source episode and a candidate episode on the same task (matched by task, §6.1). |
| **screen** | 50 pairs on the select split; smoke test (§7.3). |
| **gate** | full paired non-inferiority test on the select split (§7.2). |
| **exam** | the gate procedure run once on the exam split for the final mask. |
| **total memory @128k** | weight bytes + KV bytes at 131,072 tokens + recurrent state + measured runtime buffers (§3.3). |
| **activation importance statistics** | per-input-channel mean squared activations collected by `llama-imatrix`; used by IQ quants and the pruning score. |

---

## 3. The model

### 3.1 Anatomy (from `config.json`; verify against GGUF headers in P1)

```text
token_embd  248,320 x 4096                                  1.017 B  (untied)
32 layers, pattern [DeltaNet, DeltaNet, DeltaNet, Attention] x 8
  DeltaNet mixer (24 layers)   qkv 4096->8192, gate 4096->4096,
                               out 4096->4096, beta/alpha, conv1d    67.4 M each
  Attention mixer (8 layers)   q+gate 4096->8192, k,v 4096->1024,
                               o 4096->4096, GQA 16q/4kv, head 256   58.7 M each
  FFN (32 layers)              gate/up 4096->12288, down 12288->4096 151.0 M each
output (lm_head) 248,320 x 4096                             1.017 B
removed in v1: vision tower (27 blocks), MTP head (1 layer)
text total (analytic): 8.954 B params; converted BF16 GGUF: 8,953,803,264 params, 427 tensors,
17.92 GB (verified 2026-09-27). Convert with `convert_hf_to_gguf.py --no-mtp`: this checkpoint ships
no MTP tensors, and without the flag the metadata claims 33 blocks and llama.cpp refuses to load it.
```

Where the parameters are:

| Part | Share of text params |
|---|---|
| FFN (32 groups) | 54.0% |
| token_embd + output | 22.7% |
| DeltaNet mixers (24) | 18.1% |
| Attention mixers (8) | 5.2% |

### 3.2 Uniform-size reference points (analytic, weights only)

| Uniform type | bpw | Text weights |
|---|---|---|
| BF16 | 16 | 17.91 GB |
| Q8_0 | 8.5 | 9.51 GB |
| Q6_K | 6.5625 | 7.34 GB |
| Q5_K | 5.5 | 6.16 GB |
| Q4_K | 4.5 | 5.04 GB |
| Q3_K | 3.4375 | 3.85 GB |
| Q2_K | 2.625 | 2.94 GB |
| IQ2_XS | 2.3125 | 2.59 GB |
| IQ1_M | 1.75 | 1.96 GB |

### 3.3 Memory at 128k context

Only the 8 attention layers keep a KV cache: 8 layers x 2 (K,V) x 4 kv-heads x 256 dims.

| KV type | bytes/token | at 131,072 tokens |
|---|---|---|
| f16 | 32,768 | 4.29 GB |
| q8_0 | 17,408 | 2.28 GB |
| q4_0 | 9,216 | 1.21 GB |

The 24 DeltaNet layers keep a fixed-size recurrent state (~50 MB in f32), independent of context.

**Objective (D4), MUST be computed from the built file, not from the mask:**

```text
M_total(mask) = bytes(GGUF tensor data, incl. scales and padding)
              + KV_bytes_per_token(kv_type, kept kv-heads, kept attention layers) * 131072
              + recurrent_state_bytes(kept DeltaNet layers, kept v-heads)
              + runtime_buffers   (measured once per candidate family by llama-server
                                   at -c 131072 -np 1; recorded, not estimated)
```

Takeaway: at 128k a stock Q4_K_M model is about 5.5 GB of weights plus 4.3 GB of f16 KV. The KV
term is almost as large as the weights, which is why KV type and attention-layer moves are in
the action space.

---

## 4. The action space

### 4.1 Groups (66)

| Group | Count | Tensors covered |
|---|---|---|
| `blk.i.mixer` DeltaNet | 24 | `attn_qkv`, `attn_gate`, `ssm_out`, `ssm_beta`, `ssm_alpha` |
| `blk.i.mixer` Attention | 8 | `attn_q` (+gate), `attn_k`, `attn_v`, `attn_output` |
| `blk.i.ffn` | 32 | `ffn_gate`, `ffn_up`, `ffn_down` |
| `token_embd` | 1 | `token_embd` |
| `output` | 1 | `output` |

Outside the mask (D8): all `*_norm` tensors, `ssm_a`, `ssm_dt`, `ssm_conv1d`, `attn_q_norm`,
`attn_k_norm`. These keep their source type (GGUF confirmed: separate `ssm_alpha` / `ssm_beta`, no fused `ssm_ba`). When a keep-fraction removes heads, the matching
slices of these fixed tensors (`ssm_a`, `ssm_dt`, `ssm_conv1d`, `ssm_norm`, q/k norms) are
removed with them; only their type is fixed. Exact tensor names MUST be checked against the
converted GGUF in P1 and frozen into `groups.json`. In particular, check whether `qwen35` stores
beta/alpha as separate `ssm_beta`/`ssm_alpha` or as the fused `ssm_ba` that `llama-arch.cpp` also
defines.

### 4.2 Tier ladder (D6)

```text
Q8_0 -> Q6_K -> Q5_K -> Q4_K -> Q3_K -> IQ2_XS -> IQ1_M -> drop
```

The start state is **BF16 everywhere** (the source), so the first move for every group is
BF16 -> Q8_0. IQ2_XS and IQ1_M require activation importance statistics (§5.4). `drop` for
`token_embd` and `output` is illegal. A tensor whose row size is not divisible by the type's
block size falls back to the next legal type up, and the fallback MUST be logged.

### 4.3 Keep-fraction grids **[default]**

| Group type | Prunable structure | Legal keep-fractions |
|---|---|---|
| FFN | intermediate channels (12,288) | 1, 7/8, 6/8, 5/8, 4/8 |
| Attention mixer | kv-head groups (each = 1 kv-head + its 4 q-heads) | 4/4, 3/4, 2/4, 1/4 |
| DeltaNet mixer | key-head groups (each = 1 key head + its 2 value heads) | 16/16 ... 8/16 in steps of 2 |
| token_embd, output | none in v1 | 1 |

Below the lowest keep-fraction the only further move is `drop`. Which channels or heads to keep
is chosen once per (group, keep-fraction) by the pruning score (§5.5), not searched.

### 4.4 Moves

A move changes exactly one of:

1. a group's tier one rung down the ladder;
2. a group's keep-fraction one step down the grid;
3. the global KV type one step down: `f16 -> q8_0 -> q4_0` (V-cache quantisation requires
   flash attention, `-fa on`).

Moves never go back up during the search (monotone greedy). Undoing happens only when the
stopping rule walks back to an earlier mask (§8.5).

---

## 5. Data

### 5.1 The environment (D2)

- Dataset: `XiaomiMiMo/MiMo-V2.6-RL-oss`, config `code`, 2,698 tasks, `ability=swe`,
  `agent_name=mimo_swe_agent`.
- Each task has `docker_image`, `problem_statement`, `test_patch`, and
  `test_command = bash /testbed/mimo_test_command.sh`, with `verifier_timeout_sec` (1800 on the
  inspected sample). Tasks are multi-language (the inspected sample is a JS/jest repo).
- Docker images: `xiaomimimo/mimo-v2.6-rl-oss` (3,764 tags across all domains at inspection).
- Harness: `mimocode-agent` from `github.com/XiaomiMiMo/mimoagent` (D25), CLI `mimoagent -c <yaml> --dataset <jsonl> -w N --num-rollouts K -o <dir>`, `environment_class: docker`.
- `docker_image` in the parquet is unqualified (`format-code-task-XXXX:latest`). Map it through the dataset's `image-mapping.jsonl` to `docker.io/xiaomimimo/mimo-v2.6-rl-oss:<tag>`.
- Images are ~3 GB compressed on average (20-task sample: 57 GB total; pulls ~90 s each on the P0 pod). All 2,698 is roughly 8 TB, so the pull-run-evict policy MUST be sized from the pilot.
- Grading (from `opensource_code.py`): reset the paths the test patch touches, `git apply test_patch`, run `test_command`; reward 1 iff exit 0.

**Pin:** the dataset commit SHA, verl fork commit SHA, and image digests used MUST be written into
`run.json` (§11.2) at P0.

### 5.2 Reliability screen (D9)

Run the BF16 source with the fixed decoding config (§6.1).

```text
8 runs per task (2,698 x 8 = 21,584 episodes)
keep tasks with >= 6/8 successes   -> reliable pool P
```

Record |P| and the per-task success counts. Success here uses the full definition (tests +
detectors, §6.2).

**Winner's curse:** tasks were selected *because* the source passed them, so these 8 runs
overstate the source's pass rate on P. They MUST NOT be reused as the source side of any
screen, gate or exam. Those use fresh source episodes (§7.2).

**Pool-size rule.** The gate needs n_gate pairs from the select split (20% of P) at no more than
8 seeds per task, and the exam needs the same from the exam split. So
`|P| >= n_gate / (0.2 * 8)`. n_gate here MUST be the **projected** value, not the floor: use
the A/A estimates of v and deff, but at least v = 0.18 (p ≈ 0.9) **[default]**. That gives
n_gate ≈ 2,165 x deff and |P| ≥ ~1,350 x deff. If |P| is below this, stop and revisit D9 or D11
before continuing. If the in-search re-estimate (§7.1) later raises n_gate above what 8 episodes
per task can supply, up to 12 episodes per task are allowed **[default]** and logged. Beyond 12,
the search stops and the owner decides.

### 5.3 Repository-keyed split (D10)

The parquet has **no repository field**. The repo key MUST be derived per task:

1. primary: `git -C /testbed remote get-url origin` inside the task image (normalised to
   `host/owner/name`). **Observed on the pilot: remotes are stripped by image build**, so in
   practice step 2 is the key;
2. fallback when no remote exists: first commit hash of `/testbed` history
   (`git rev-list --max-parents=0 HEAD`). Images truncate history *after* the base commit, so
   the root commit should be shared across a repo's tasks. This MUST be verified: if nearly every
   task has a unique root, the key is degenerate, so fall back to step 3;
3. last resort: cluster by the set of top-level paths in `test_patch`, and log each such task.

Then assign whole repos to splits, greedily by task count, targeting 60/20/20 of |P|, with a fixed
seed. Freeze `splits.json`. **The exam split MUST NOT be read by any tool before P7.**

### 5.4 Proxy corpora

| Corpus | Contents | Size **[default]** | Used for |
|---|---|---|---|
| `swe_proxy` | BF16's **successful** search-split trajectories, rendered with the model's chat template | 150k assistant tokens | KL proxy (§6.3) |
| `general_proxy` | 50/50 FineWeb-Edu sample + general code (non-MiMo repos, e.g. The Stack v2 dedup sample), packed into 2,048-token sequences; positions 1,024..2,047 of each are scored (no assistant role exists here) | 50k scored tokens (~49 sequences) | sanity cap (§6.4) |
| `stats_corpus` | search-split trajectories, all tokens | same trajectories as `swe_proxy` | activation importance statistics, pruning score (collected in P3, after the split exists) |

KL is averaged over **assistant-generated tokens only**. Prompt, tool-output and
environment-observation tokens are context, not scored. Tool-call tokens are also reported
separately as `KL_tool`, for diagnosis only.

**Scoring MUST use whole trajectories.** Stock `llama-perplexity --kl-divergence` splits text
into fixed `n_ctx` chunks and scores only the second half of each chunk. That would cut
trajectories at chunk boundaries, skip half the assistant tokens, and test the KV cache only at
short context. Patch L5 (§9.1) replaces it: each trajectory is evaluated in one pass at its full
length (up to 131,072 tokens), and only assistant-token positions are scored.

Storage: a full-vocabulary log-prob row is about 0.5 MB/token for the 248k vocabulary. L5 stores
rows **only for scored (assistant) positions**, so the base files are about 0.5 MB x (150k + 50k)
= ~100 GB. Without that restriction, context tokens would multiply this several times over.

### 5.5 Pruning score (D29)

For a prunable unit u (FFN channel c, or head group h) in group g:

```text
score(u) = sum over weights w feeding/leaving u of |w| * ||x_in(w)||_2
```

Here `||x_in||` is the per-input-channel RMS activation from `stats_corpus`. For FFN channel c
this sums `|W_down[:, c]| * ||a_c||` and the gate/up rows for c weighted by the input RMS. For
head groups it uses the o-projection columns and the q/k/v rows of that head group. Keep the
top `keep_fraction * n_units` by score. Scores are computed once on the source weights and
frozen in `prune_scores.json`.

---

## 6. Judges

### 6.1 Episode protocol (D13)

- Sampling: the values in `generation_config.json` (temperature, top_p, top_k), identical for source and candidate.
- Seeds: every request carries an explicit `seed` field, derived as `hash(run_id, episode_id, turn)`. The harness adapter MUST forward it to `llama-server` and log it. Across different weights the same seed gives no meaningful correlation, so pairs are **matched by task only**. Seeds make each episode reproducible, not correlated.
- Server: `llama-server` from the patched fork, `-c 131072 -np 1 --jinja -fa on`, with the KV type from the mask. The source runs in the same binary with `-ctk f16 -ctv f16`.
- Every episode logs: full transcript, tool calls, final diff, test stdout/stderr, verdict,
  detector flags, token counts and wall time.

### 6.2 Episode success = tests pass AND no detector fires (D19)

| Detector | Fires when **[thresholds are defaults]** |
|---|---|
| **Doom loop** | the same normalised tool call (name + canonical args) appears 3 times in a row, **or** any 64-token span of assistant output repeats >= 4 times, **or** the step cap is hit |
| **Malformed tool call** | any assistant tool call fails to parse under the chat template's tool-call grammar or fails the tool's argument schema |
| **Premature stop** | the episode ends with no submitted change, **or** EOS arrives inside an unclosed tool-call / code block, **or** the final diff is empty |
| **Token garbage** | invalid UTF-8 or U+FFFD in output, **or** the submitted diff fails `git apply --check`, **or** an edited `.py/.js/.ts` file fails a syntax check (`python -m py_compile`, `node --check`, `tsc --noEmit --allowJs` on the single file) that it passed before the edit |

Each detector is also reported separately as a per-episode rate for source and candidate. The
detectors are part of success, so the source's own detector rate is already in its baseline.

### 6.3 Reference comparison = proxy only (D14, D20)

```text
KL_swe(mask)    = mean over swe_proxy assistant tokens of KL(p_source || p_candidate)
KL_onpol(mask)  = same, but on the candidate's own screen trajectories
                  (source teacher-forced on candidate text)
```

`KL_swe` drives every move. `KL_onpol` is computed at every screen from **20 extra candidate
episodes on search-split tasks** **[default]**, never from the select-split screen episodes, so
select-split data never reaches the search objective. If `KL_onpol / KL_swe > 3` **[default]**,
the run log MUST flag "proxy drift", and the next full re-probe MUST add those search-split
trajectories to `swe_proxy` (capped at 2x original size). Neither KL is ever an acceptance
criterion.

### 6.4 Sanity cap (D17)

```text
cap = KL_general(stock uniform Q4_K_M of this model, built with the same
                 activation importance statistics, KV f16)
constraint: KL_general(mask) <= cap
```

The cap is checked on **every accepted move** (the 50k-token general probe is cheap relative to
the SWE probe), and a move that violates it is illegal. Per-group floors are derived at P3: any
(group, setting) whose **solo** application to the source already exceeds the cap is removed
from the move set for the whole run.

---

## 7. Statistics

### 7.1 Sample size

Pairs are matched by task (§6.1). For a pair, the variance of (candidate success − source
success) near a true difference of 0 is `v`. v is the discordance rate: the fraction of pairs
where exactly one side succeeds. With independent runs at pass rate p, v ≈ 2p(1−p).
Non-inferiority at margin 3 pts, one-sided alpha 0.05, **95% power (D11)**:

```text
n = ((z_0.95 + z_0.95) * sqrt(v) / 0.03)^2 * deff
```

| v | n (deff = 1) |
|---|---|
| 0.05 | ~601 |
| 0.10 | ~1,203 |
| 0.18 (p = 0.9) | ~2,165 |

`deff` is the design effect of clustering by repository, estimated from the A/A data (§7.4).
v and deff are measured in P2 and then n_gate is fixed, with a floor of 1,200 **[default]**.
v is re-estimated from the first two screens, since a degraded candidate is more discordant than
the source is with itself. If the estimate exceeds the P2 value by >25%, n_gate is recomputed
and logged.

### 7.2 The gate (D11)

```text
data:    fresh source episodes and fresh candidate episodes, n_gate of each, spread
         evenly over select-split tasks (equal episodes per task); source episodes
         are never the reliability-screen runs (§5.2) and never reused across gates
stat:    delta = mean over tasks of (cand_rate_t − source_rate_t)
CI:      repository-cluster bootstrap (resample repos, then their tasks), 10,000
         resamples, one-sided 95% lower bound L
PASS:    L > −0.03
also:    for each detector, the upper 95% bound on (cand − source) rate <= +0.03 [default]
```

Every gate run is counted as a "look" in `gates/` (the extra gate in §8.3 included). The select
split is used only for model selection, and repeated looks make its results optimistic. The
returned mask sits near the margin by construction, so its exam result is expected to shrink
toward it. The claim therefore comes from the exam split alone (§10), sized with the same n_gate.

### 7.3 The screen (D28)

50 source + 50 candidate episodes on 50 select-split tasks (tasks drawn without replacement
until exhausted, then recycled). **FAIL if** `delta < −0.10`. The screen is a smoke test with no
confidence claim. It exists to catch collapse early. Screen source episodes may be drawn from a
fixed bank of fresh source runs (not the reliability runs).

### 7.4 A/A calibration of the harness (P2)

Run the source against itself: two independent sets of n_gate fresh episodes on the select split.
This is **calibration, not a gate**:

- **Estimate** v and deff (§7.1).
- **Check** the point estimate |delta| <= 0.015 **[default]** and the detector rates are equal
  within noise. A larger gap means the harness, seeding or task isolation is non-deterministic
  in a biased way. Fix it before any search.
- **Record** the A/A as the source's gate on record: `last_gate_pass = source`.

---

## 8. The learner (D15, D16, D27)

### 8.1 State

```text
mask      current mask
cache     score(move) = dKL_swe / dBytes for each legal move, measured on some past mask
history   list of (mask, KL_swe, KL_general, M_total, screen?, gate?) per accepted move
```

### 8.2 Probe

`probe(mask, move)`:

1. Build the candidate GGUF by **assembling** from the tensor cache (§9.2). Only the moved group's tensors change.
2. Run the L5 whole-trajectory KL tool (§5.4, §9.1) against the source base files for `swe_proxy` and `general_proxy`.
3. Compute `M_total` from the built file (§3.3).
4. Return `dKL_swe = KL_swe(new) - KL_swe(mask)`, `dBytes = M_total(mask) - M_total(new)`, and `KL_general(new)`.

### 8.3 The loop

```text
mask <- source (BF16 everywhere, KV f16, keep 1 everywhere)
last_gate_pass   <- source          # its gate on record is the A/A (§7.4)
last_screen_pass <- source
full_reprobe(mask)                  # probe every legal move, fill cache
k <- 0
loop:
    C <- the 16 legal moves with the lowest cached score        # D16
    re-probe each move in C on the current mask
    m* <- argmin over C of dKL_swe / dBytes
          subject to dBytes > 0 and KL_general(new) <= cap
    if no such move: break                                      # search space exhausted
    mask <- apply(m*); k <- k + 1; record history
    if k % 8 == 0:
        full_reprobe(mask)
        run screen on mask; log KL_onpol
        if screen FAILS: break
        last_screen_pass <- mask
        if k % 32 == 0:
            run gate on mask
            if gate FAILS: return last_gate_pass                # D27
            last_gate_pass <- mask
# loop ended by a failed screen or an exhausted move set:
candidate <- (mask if the move set was exhausted and mask passes a screen,
              else last_screen_pass)
if candidate != last_gate_pass and gate(candidate) PASSES: return candidate
return last_gate_pass                                           # D27
```

Every returned mask therefore has a passing gate on record (the source's is the A/A, §7.4).

### 8.4 Why greedy with re-probing

Our K94 recipe ranked groups once against a fixed base, then combined them. Dettmers-style
re-probing measures each move's cost **on the current, already-compressed model**, so an
interaction (a group that becomes critical after its neighbour was compressed) shows up in the
next re-probe. The top-16 per-move re-measure keeps that property between full re-probes.

### 8.5 Cost accounting (D22)

Log per phase: probes run, GPU-hours, episodes run, wall time. Expected probe count per full
re-probe is about 66 groups x up to 2 move types + 1 KV move, so ~130 probes. Full re-probes run
every 8 moves, with 16 probes per move in between.

---

## 9. Software and infrastructure

### 9.1 llama.cpp fork (required patches)

Stock llama.cpp supports the `qwen35` arch but assumes a global `n_ff`, a fixed head count and
every layer present. The fork MUST add, for `qwen35` only:

| Patch | What | Test |
|---|---|---|
| L1 | per-layer `n_ff` array in GGUF hparams | build a 7/8-FFN layer by hand; output identical to a model with those channels zeroed |
| L2 | per-layer kv-head and DeltaNet head counts | same equivalence test for one pruned head group |
| L3 | missing-block skip: a layer with no mixer or no FFN tensors is an identity for that sub-block | drop one FFN; output identical to the source with that FFN's output zeroed |
| L4 | KV and recurrent-state allocation honour L2/L3 (a dropped attention mixer allocates no KV) | `llama-server` memory report matches §3.3 formula within 1% |
| L5 | Whole-trajectory KL tool: one pass per trajectory at full length (<= 131,072), base rows stored only at assistant positions, per-token KL dump | feed exactly n_ctx tokens and mark positions n_ctx/2 .. n_ctx−1 as scored: L5's per-position KL equals stock `--kl-divergence` on those positions; KL(source, source) = 0 exactly |

Each patch MUST ship with its equivalence test, and all five tests MUST pass before P3.

### 9.2 Tensor cache and assembler

Requantising the whole model per probe is wasteful. Build lazily and memoise:

```text
cache[(group, keep_fraction, tier)] -> pruned + quantised tensors (+ hparam deltas)
assembler(mask) -> GGUF by concatenating cached tensors with a rewritten header (gguf-py)
```

Quantisation MUST use `llama-quantize` (with `--tensor-type` overrides and the importance-statistics
file) or ggml's quantise functions called identically, so a cached tensor is bit-identical to what
`llama-quantize` would write. Check this once per tier: non-IQ tiers at P1; IQ2_XS and IQ1_M at P3, once the real importance statistics exist.

### 9.3 Compute (D24)

```text
P0.a  verify Pipeshift host identity OUT OF BAND
      (at 2026-09-27 ssh reported: REMOTE HOST IDENTIFICATION HAS CHANGED,
       new ED25519 fingerprint SHA256:hx7JR2ZTLPgeXh8J27ZaRyvpld/R3kblYiNrZhA+6g8;
       do not connect until this fingerprint is confirmed with the provider)
P0.b  inventory: GPUs/VRAM, RAM, disk, docker, running jobs   (read-only)
P0.c  cleanup to a fresh state: ONLY after the owner approves the inventory list
      of what will be deleted; archive anything not obviously scratch
P0.d  if Pipeshift is unavailable or unverifiable: rent on Lium
```

Minimum requirements **[default]**: one GPU with >= 48 GB for BF16 source serving at 128k plus a
candidate; >= 1.5 TB free disk (KL base files ~100 GB, tensor cache ~200 GB, docker images, logs);
enough CPU and RAM for ~16 concurrent SWE containers.

---

## 10. Baseline and claim

### 10.1 dynquant baseline (D21)

On the same model, with the same `swe_proxy` tokens:

1. Base: uniform Q2_K (with activation importance statistics).
2. For each of the 64 layer-groups (mixer, FFN): restore that group alone to Q8_0 and measure
   the KL improvement against BF16 on `swe_proxy`. Rank by improvement, once.
3. `dynquant-K` = Q2_K base with the top-K ranked groups at Q4_K (the K94 recipe: Q8 ranking,
   Q4 deployment, no re-ranking). `token_embd` / `output` use llama.cpp's Q2_K defaults.
   The ranking is measured at KV f16 **[default]**, exactly as K94 was, and is not re-ranked
   when step 4 deploys it at a different KV type.
4. KV type = **the final Chantless mask's KV type** (D21), so the comparison isolates weight
   allocation. KV savings are reported separately as a lever any build can use.
5. Build `dynquant-K` for K in {0, 8, ..., 64}, and refine to steps of 2 in the two intervals
   adjacent to M_total(Chantless). Record `M_total` for each.

### 10.2 The exam (P7)

On the exam split, run the gate procedure (§7.2, fresh episodes, n_gate) once each for:

- the Chantless mask;
- every dynquant-K point with `M_total <= M_total(Chantless)`;
- the dynquant-K point with the smallest `M_total > M_total(Chantless)`, for context.

Also run fresh source episodes as the common reference.

### 10.3 Claim: frontier dominance (D30)

For each build X, let `U_X` and `L_X` be the upper and lower one-sided 95% bounds on
(X − source) success on the exam split (repo-cluster bootstrap, §7.2).

```text
CLAIM holds iff
  (a) Chantless is non-inferior to source:          L_Chantless > −0.03, and
  (b) every dynquant-K point with M_total <= M_total(Chantless)
      is demonstrably worse than the margin:          U_dynquant-K < −0.03
REFUTED if any such dynquant point is itself non-inferior (L > −0.03).
INCONCLUSIVE otherwise (some dynquant point is neither shown worse nor shown non-inferior).
Headline: M_total(Chantless) vs M_total of the smallest dynquant point that is non-inferior.
```

Part (b) requires positive evidence that each cheaper dynquant point loses more than 3 pts. It
does not accept a failed test as evidence. With j such points, each tested at one-sided 5%,
the chance of a false (b) is at most 5% overall, because every point must clear its bound. Report
j, and each point's delta, L and U.

Report everything else as observed, not claimed: exact GB at 128k, per-detector rates, the
full dynquant frontier curve, and decode tok/s for each build. Speed is outside the objective,
so it is reported and not claimed.

---

## 11. Artifacts

### 11.1 Mask file (`mask.json`)

```json
{
  "schema": "onlyskills.mask/v1",
  "source": {"repo": "XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B", "revision": "<sha>"},
  "kv_type": "q8_0",
  "removed": ["vision", "mtp"],
  "groups": {
    "blk.0.mixer": {"tier": "Q4_K", "keep": 1.0},
    "blk.0.ffn":   {"tier": "Q3_K", "keep": 0.75},
    "blk.3.mixer": {"tier": "drop"},
    "token_embd":  {"tier": "Q6_K", "keep": 1.0},
    "output":      {"tier": "Q6_K", "keep": 1.0}
  },
  "prune_scores": "prune_scores.json#sha256=<...>",
  "importance_stats": "stats.gguf#sha256=<...>",
  "measured": {"M_total_128k_bytes": 0, "KL_swe": 0, "KL_general": 0}
}
```

### 11.2 Run directory

```text
runs/<run_id>/
  run.json            pins: dataset/verl/llama.cpp SHAs, image digests, hardware, all [default] values
  splits.json  seeds.json  groups.json  prune_scores.json  stats.gguf
  proxy/              swe_proxy + general_proxy text, KL base files
  cache/              tensor cache
  history.jsonl       one line per accepted move
  probes.jsonl        every probe
  episodes/           every episode transcript + verdict + detector flags
  gates/              each screen / gate / exam with its bootstrap output
  final/              mask.json, the GGUF, exam report
  log.md              human run log; every deviation from this manual
```

---

## 12. Runbook

| Phase | Do | Exit criterion (MUST) | Artifacts |
|---|---|---|---|
| **P0 Infra** | §9.3 host verification, inventory, approved cleanup or Lium; pin SHAs and digests | host verified; `run.json` pins filled | `run.json` |
| **P1 Build** | fork llama.cpp with L1–L5; convert source to BF16 GGUF with vision + MTP removed; freeze `groups.json` | all L-tests pass; BF16 GGUF next-token log-probs match HF transformers within 1e-2 mean abs on 10k tokens **[default]** | GGUFs, `groups.json` |
| **P2 Harness** | wire `mimo_swe_agent` to `llama-server` (OpenAI-compatible endpoint; adapter forwards `seed`); reliability screen §5.2; repo split §5.3; A/A calibration §7.4; fix n_gate | pool-size rule met; split frozen; A/A check passes; n_gate fixed | `splits.json`, A/A report |
| **P3 Proxy** | fresh source runs on search split -> `swe_proxy` / `stats_corpus`; `general_proxy`; activation importance statistics; KL base files; stock Q4_K_M -> `cap`; pruning scores; per-group floors | `cap` recorded; KL(source, source) = 0 | proxy files, `stats.gguf`, `cap`, `prune_scores.json` |
| **P4 Baseline** | dynquant ranking §10.1 steps 1–3 (the KV type and fine frontier are fixed after P5) | ranking frozen | `baseline/` |
| **P5 Search** | §8.3 until stop | returned mask has a passing gate on record | `history.jsonl`, `gates/` |
| **P6 Freeze** | build final GGUF from mask; verify the file reproduces the mask (tensor types, shapes, `M_total`); build dynquant frontier at the mask's KV type §10.1 steps 4–5 | reproduction check passes | `final/`, `baseline/` |
| **P7 Exam** | §10.2, once | exam report written | `final/exam.md` |
| **P8 Report** | claim evaluation §10.3; cost ledger; update design record + Doc | report reviewed | `report.md` |

Nothing from P7 may feed back into P5. If P7 is re-run for any reason, the exam split is spent,
and that MUST be stated in the report.

---

## 13. Known risks

| Risk | Where it bites | Mitigation in this manual |
|---|---|---|
| Proxy/behaviour mismatch: KL on source trajectories misses on-policy compounding | P5 | `KL_onpol` at every screen + proxy-drift rule §6.3; episodes decide acceptance |
| Select-split overfitting from repeated gates | P5 | exam split single-use §10.2 |
| Harness incompatibility (mimo_swe_agent expects a different serving stack) | P2 | adapter; A/A calibration check §7.4 before search |
| llama.cpp patches change numerics | P1 | equivalence tests L1–L5 |
| Too few reliable tasks (small P) | P2 | pool-size rule from projected n_gate §5.2 |
| Pruned/dropped structures break long-context behaviour only beyond the proxy's lengths | P5–P7 | trajectories span real episode lengths; detectors include doom loops; exam uses full episodes |
| Unverified host (key change) | P0 | out-of-band verification before any connection |
| MiMo env noise (flaky tests) | P2 | A/A measures it; flaky tasks drop out via the >= 6/8 screen |

## 14. Deferred to v2 (explicitly out of scope)

Mask + weights training; improve ("beat the teacher") mode; per-tensor or sub-tensor precision;
vocabulary pruning; other MiMo domains (cyber, general, webdev); Unsloth / uniform / smaller-model
baselines; speed in the objective; customer environment API; skill-aware multi-mask loader and
delivery format.

## 15. Change control

Any deviation from a MUST, or any change to a [default], is written to `log.md` with the date,
old value, new value and reason, before the affected phase starts.

---

## 16. Defaults chosen by the author (review these)

| Where | Default |
|---|---|
| §4.3 | keep-fraction grids (FFN 1..4/8; attention 4..1 of 4 kv groups; DeltaNet 16..8 of 16) |
| §5.4 | 150k `swe_proxy` tokens, 50k `general_proxy` tokens, FineWeb-Edu + The Stack v2 sample |
| §6.2 | detector thresholds (3 repeated calls, 64-token span x4, syntax checks for py/js/ts) |
| §6.3 | 20 search-split on-policy episodes per screen; proxy-drift threshold `KL_onpol / KL_swe > 3` |
| §5.2 | pool sized with v >= 0.18; up to 12 episodes/task if n_gate grows |
| §7.1–7.4 | n_gate floor 1,200; v re-estimation trigger +25%; per-detector +3 pt bound; A/A point-estimate tolerance 0.015 |
| §8.3 | gate every 32 moves |
| §9.3 | >= 48 GB GPU, >= 1.5 TB disk |
| §10.1 | dynquant frontier step K = 8, refined to 2 near the mask; ranking measured at KV f16 |
| P1 | HF-vs-GGUF log-prob tolerance 1e-2 mean abs |
