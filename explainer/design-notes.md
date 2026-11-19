# Chantless
## No strings. Just weights.
Deus Experiments · Token Bender
Living design record · Updated October 12, 2026 (spec tuning)

> Renamed to Chantless on October 5, 2026. This inherited design describes proposed precision search and deployment goals, not the completed conditioning or physical-extraction results. The MiMo search remains paused.

> Spec tuning, October 12, 2026: the decisions below came from tokenbender's answers and from defaults adopted under his instruction. They are recorded in the local decision record. They specify the contract only; they are not implementation or results. Where the inherited MiMo manual disagrees (notably D4 total memory at 128k as the objective, and KL-ranked moves with episodes used only for stopping), this record supersedes it.

## 1. Objective and deployment contract
Ship the skill. Not the whole model.

Turn a strong general-purpose model into a standalone specialist for a user-defined workload. Start with coding and terminal-agent behavior. Preserve the required task quality—or improve it through training—while minimizing the deployed weight memory and respecting deployment constraints.

The user supplies the work and a definition of success. We learn a precision mask, optionally adapt the weights, build the resulting model, and test it running alone. The original full model is available during development; it is not shipped as a companion, remote fallback, or runtime verifier. A skill is a complete model artifact, not a prompt pack or an adapter requiring the original model.

The intended ceiling increase is stronger target-task capability within a given deployment budget. Smaller files alone are insufficient. Potential product outcomes include private local coding agents, offline operation, fitting useful capability on more accessible hardware, and serving more users on existing hardware. Those outcomes require deployment measurements; they are not universal promises.

## 2. End-to-end architecture and research focus

FULL MODEL + OUR GENERAL-DATA SANITY SUITE
                         ↓
              INITIAL SANITY MASK
                         ↓
USER'S CALIBRATION ENVIRONMENT + RESOURCE BUDGET
                         ↓
[OUR RESEARCH FOCUS: LEARN THE PRECISION MASK]
Run candidate → judge behavior → update mask
   (task failures feed back into allocation)
                       ↘ optionally update weights
                         ↺
                         ↓
       APPLY PRECISION/CAPACITY ALLOCATION
                         ↓
             BUILD STANDALONE WEIGHTS
                         ↓
        FINAL HELD-OUT + DEPLOYMENT EVALS
                         ↓
                  SHIP SPECIALIST

KEY SCOPE: Our central research is BETWEEN the full model and precision allocation: learning what the capability needs so that supported quantization and removal machinery can apply the right allocation. The precision mask is the prescription; the quantizer and physical builder execute it. Candidate allocations must also be exercised inside the learning loop, not only applied once at the end.

LEARNER DRIVER (decided by tokenbender): A proxy may rank moves, but task-verifier and sanity failures must change later allocation decisions, not only stop or roll back the search. Trajectories from failed episodes are added to the scoring proxy. Groups implicated by the failure are re-measured, then penalized or reverted, before further moves. A search that uses the proxy only, with behavioral stopping, is kept as an ablation. It does not satisfy this contract.

CLASSIFIER ONE-SHOT PATH (decided by tokenbender, November 5, 2026; unified across architectures November 18, 2026): For classifier environments, where every probe yields one score per trajectory, a one-shot search is an adopted alternative to the feedback learner (`learner.algo: "oneshot"`). An automatic lossless pre-pass first removes dead shared-KV tensors (verifying scores do not move). The search then reuses the sanity phase's solo tier/drop probes, plus Wanda-ordered per-MLP width probes (`||W_down[:, j]||_2 * sqrt(in_sum2_j)` from `llama-imatrix`) when neuron cuts are enabled, and a merge-closed Rule-P vocabulary keep-set (auto-selected on a candidate grid to maintain solo rank-fidelity >= 0.998, or configured explicitly). It predicts any mask's per-trajectory scores as the source's scores plus the sum of solo tier and width shifts, chooses the joint `(tier, width)` mask that best keeps the source's ranking of the store's trajectories (1 - Spearman, `oneshot_objective: "fidelity"`) under the exact GGUF byte model (accounting header, tensor table, variable-width KV metadata, and vocabulary row/KV savings), and builds once from the frozen base + `chantless.mask/v3` thin mask. Acceptance is unchanged: the built file is re-tokenized through its own trimmed GGUF vocabulary (`tokdocs`) with remapped answer token IDs, and must pass the sequential non-inferiority gate and then the single-use exam. Evidence: on the Qwen3.5-0.8B detector (14,161 source-right select documents), `f200` lost 47 documents at 200,240,128 bytes against 46 for the hand-built 200,296,000-byte mask; on the Gemma 4 E2B detector (9,311,299,104-byte BF16 source), `mask10` built to 930,971,136 bytes (10.00%, 7 bytes off the byte model) and passed both the select gate (42 / 14,227 lost, L = −0.0055) and the 51,066-document sealed exam (99.76% agreement, L = −0.0037).

GENERATIVE ONE-SHOT & UNIVERSAL SWITCH APPRENTICE (extended November 18–19, 2026): On multi-token generative and tool-calling environments (`kind == "kl"`), `learner.algo: "oneshot"` consumes per-trajectory `(n_tgt, kl, nll_cand)` shifts recorded by `traj-kl` across solo and Wanda width probes, solving the target-weighted additive KL (`preserve`) or NLL (`improve`) surrogate under the exact byte budget and `SizeMatchedCap(bpw)` general-KL guard while protecting chat/tool/reasoning control tags (`<tool_call>`, `<think>`, `<|...|>`) during Rule-P vocabulary trimming. Post-extraction Apprentice switch learning (`learn_gates_seq` / `learn_gates` with `switch_mode: "up_row_zero"`, `float32` `replay_h0` caching, and stabilized `AdamW` `eps`) learns straight-through binary MLP neuron switches on multi-token trajectories—anchored corrected-token NLL plus contrastive margin over mistaken outputs, full-vocabulary replay KL from cached pre-`lm_head` hidden states, and A-GEM gradient projection—and materializes switched-off neurons by zeroing contiguous super-block rows of `ffn_up.weight` and `ffn_gate.weight` (`d = 0.0f`), giving exact `0.0f` student-to-GGUF parity across every `ggml` quant format (`IQ1_S`, `IQ2_XS`, `Q2_K`..`Q8_0`, `BF16`) with zero bytes added. All supported architectures (`qwen3`, `qwen35`, `gemma4`) build and serve from one unified `llama.cpp` `a01dff9` patch (`mvp/patches/llama-cpp-a01dff9-chantless.patch`). Stock-checkpoint evidence (all `18` exported thin masks in `masks/`, `weights.lineage = []`, `cli_rebuild_sha256_equal = true`, `empty.sha256 == base.sha256`): on `google/gemma-4-E2B-it` (`4adf1574`), one-shot specialists beat size-matched uniform quantization by `+49.78` pts on `ClsBench` (`9.97%` BF16, `99.21%` exam), `+44.02` pts on `BFCL` (`20.92%` BF16, `97.13%` exam), and `+75.00` pts on `DeepMath` (`22.85%` BF16, `75.00%` exam), while 25-example zero-extra-byte Switch Apprentice masks beat all four controls (`base`, `empty`, `flipped`, `shuffled`) on unseen held-out mistakes (`15/60`, `26/60`, `10/60`); on `Qwen/Qwen3-4B` (`1cfa9a72`), specialists beat uniform quantization across all three tasks (`+15.51`, `+5.87`, `+8.96` pts) with clean Apprentice wins on `ClsBench` (`16/60` vs `4/4/4/15`) and `BFCL` (`22/60` vs `3/3/1/12`) and a `shuffled` tie on `DeepMath` (`20/49`); on `Qwen/Qwen3.5-4B` (`851bf6e8`), specialists beat uniform quantization by `+46.73` pts on `ClsBench` (`9.94%` BF16, `93.40%` exam) and `+30.50` pts on `DeepMath` (`26.02%` BF16, `82.27%` exam) and matched `Q3_K` on `BFCL` (`26.01%` BF16, `92.39%` exam), with clean Apprentice wins on `ClsBench` (`33/60` vs `4/4/5/20` balanced-label `shuffled`, `92.79%` `sel1`) and `BFCL` (`29/60` vs `0/0/2/19` pair-shuffled, `83.22%` `sel1`) and an eval-variance cluster on `DeepMath` (`9/60` vs `10/8/10/11`, `61.67%` `sel1`).

## 3. Central artifact: a precision mask
The mask assigns a representation or precision budget to each controllable component. Unlike a binary keep/drop mask, it can specify multiple tiers. An illustrative allocation is: component A → 8 bits; B → 4 bits; C → approximately 1.5 bits per weight; D → removal, effectively 0 bits.

The bit ladder is configurable within supported formats; it is not a commitment to exactly these values or a two-tier split. Effective storage includes scales, packing and metadata. Some numerically sensitive components may require formats outside the example ladder. Zero bits means an actual supported removal operation, not simply storing a tensor of zeros.

The search scope should not be restricted to MLPs. Attention heads/projections, embeddings, per-layer embeddings, mixers and vocabulary tables are considered where legal interventions and executable layouts exist. Search starts at layer-group granularity (mixer, FFN, per-layer embedding, embeddings, output), derived automatically from the GGUF header and config (`gguf_ops.adapter_from_gguf`), including undroppable shared-KV donor layers and losslessly removable dead shared-KV projections. `chantless.mask/v3` records the complete deterministic transform from frozen base to specialist in one thin JSON file (~20–80 KB): group and per-tensor tiers/drops, dead-tensor removals, compressed kept-neuron MLP width bitmaps, merge-closed vocabulary keep-set bitmaps, and Apprentice neuron-switch bitmaps (plus optional sparse grid-level deltas and decision thresholds). Legacy `chantless.mask/v2` and `onlyskills.mask/v1` files remain readable without modification.

“Where the capability lives” is shorthand for finding a distributed, interacting set of dependencies. Importance is conditional on the other components, the allocation, the current weights and the task. The mask is a learned engineering artifact, not proof of a uniquely isolated biological-style organ in the original model.

## 4. What the user provides: an executable calibration environment
The user-facing input is a calibration spec expressed through a single environment-script entry point, with supporting datasets, tools and evaluators as needed. It can be as detailed as the workload requires.

The environment specifies:
• Prompts, tasks, initial conditions and the situations the model must handle.
• Available tools, permitted actions and interaction rules.
• Explicit success criteria, failure criteria and user-provided verifiers.
• Existing evals and required performance thresholds.
• Relevant sequence lengths, output requirements and deployment conditions.

The user's main work is designing representative tasks/prompts and useful task verifiers—not selecting important tensors or hand-assigning bit widths. Interactive environments can generate trajectories and expose failures a static text corpus misses.

Environment contract: one entry point supplies tasks, a run function and a verifier, and declares the budget, deployment conditions (context length, concurrency, KV-cache type) and split membership. Infrastructure errors are retried and are not counted as task failures.

Terminology: in this discussion, “calibration” is the broad role of data/environment feedback in specifying what must survive compression. In implementation, activation-calibration data, training trajectories and evaluation sets are distinct artifacts. Keep final acceptance data separate from iterative mask learning and weight repair.

## 5. Our starting point: the sanity mask
Before specialization, use our general-data suite to construct an initial protection or sanity mask. FineWeb is one candidate source, not a finalized dataset or a complete test of general knowledge.

This gives an initial hypothesis about components needed for baseline behavior we do not want specialization to damage. Continue testing baseline behavior throughout the search. Pair general text with behavioral sanity checks; activation statistics alone cannot certify universal importance or protection of every capability.

Protection mechanism (default, adopted): the initial sanity mask is a set of per-group minimum precisions. Each floor comes from solo-application tests: a setting is forbidden when applying it alone already breaks general-data sanity. On every accepted move, general-data KL must not exceed that of a uniform quant of the same average bits per weight (the size-matched cap). Behavioral sanity checks are hard gates at every screen and gate. The general suite combines general text, general code and a small set of executable behavioral checks. Baseline quality requirements remain explicit gates. The initial mask is not a permanent proof that every protected component must remain unchanged forever.

## 6. Optimization modes: what is allowed to change?
There are two modes, independent of whether the objective is preservation or improvement.

MASK-ONLY
Keep source weights fixed. Change the precision/removal allocation and rebuild or simulate the corresponding candidate. Quantized values change because their representation changes; we do not gradient-train the source weights.

MASK-PLUS-WEIGHTS
Update both the allocation and trainable weight values so the candidate can adapt to compression and the user's environment. Mask and weight updates may alternate or be jointly optimized; no exact optimizer is selected yet. Training should exercise the relevant deployed compression/removal behavior rather than silently using a different operator. Trained weights must be materialized by the same builder that produces the deployed artifact, and the built result is re-measured in the loop. Mask-only is a complete, valid mode; weight adaptation remains optional.

The loop is: propose allocation → run tasks → evaluate task outcomes and regressions → change mask and optionally repair weights → evaluate again. Revisit importance after weight or allocation changes. Do not assume isolated gains add or that one ordering is optimal at all budgets. Preserve accepted candidates for comparison rather than accepting changes merely because a proxy improves.

## 7. Verification stack: more than KL
Combine three kinds of evidence during development.

USER TASK VERIFIERS
The environment's tests and outcome checks determine whether the requested work was completed: for example, executable code tests, valid tool use and required output fields.

OUR SANITY AND COMMON-FAILURE CHECKS
Translate user complaints into executable checks where possible: repeated or looping output (“doom loops”), premature stopping, omitted required fields/content, malformed tool calls, instruction violations and failures that emerge over longer sequences. These are proposed categories to operationalize, not a claim that a complete detector suite already exists.

FULL-MODEL REFERENCE COMPARISON
In preservation mode, keep a frozen original checkpoint and compare behavior on matched prompts and defined evaluation horizons. KL or other distributional measures are diagnostics alongside sequence-level and task-level comparisons. KL may guide search but is never an acceptance criterion. In preservation mode, the fidelity tolerance is a task non-inferiority margin against fresh source-model episodes. Missing a required semantic element matters; merely choosing different tokens need not be a failure. The full model is a reference, not unquestionable ground truth.

For a given prompt and sequence-length condition, define the criteria the candidate must pass. In mask-plus-weights mode, do not move the reference checkpoint with the candidate. Compare candidates fairly under declared evaluation conditions and retain held-out final checks.

This verifier stack is development-time machinery. No verifier model accompanies the shipped specialist.

## 8. Two objectives: preserve or improve

| Question | Preserve capability | Improve capability |
|---|---|---|
| Primary goal | Meet the target quality at lower cost | Improve verified task performance within the budget |
| Teacher role | Frozen reference with an explicit fidelity/performance tolerance | Remove teacher matching from the task objective and acceptance gate |
| What judges success? | Task outcomes, sanity checks, and selected reference criteria | Rigorous task-outcome verifiers plus continuing sanity checks |
| Can the original model still be measured? | Yes, as the reference | Yes, as a reporting baseline, not a behavioral ceiling |
| Weight updates | Optional | Available to learn better behavior; allocation changes can also be explored |

Removing teacher matching permits improvement; it does not guarantee it. Better-than-teacher behavior must come from useful environment feedback and survive independent evaluation. In improvement mode, the desired behavior is defined by the task, not imitation of every original-model output.

## 9. Reward and constraints: quality first, then fewer bytes
Once the candidate reaches the specified quality, reward further memory reduction only while it continues meeting the quality requirements. This is a constrained objective, not an unrestricted weighted trade that can buy a smaller file by quietly dropping below the required score.

MINIMIZE: actual deployed weight bytes of the built candidate (built-file bytes).
KV-cache type is a deployment setting declared by the environment, not a mask move. Total memory at 128k may be reported for information but is not the objective.
SUBJECT TO: required task scores and per-criterion checks; sanity requirements; relevant deployment constraints; and, in preservation mode, the chosen tolerance relative to the full model.

In mask-only mode, optimize the mask with fixed source weights. In mask-plus-weights mode, optimize the mask and candidate weights. In improvement mode, task-quality targets can be raised as better candidates are established; retain the quality/memory tradeoff rather than assuming one universal scalar reward.

Measure actual bytes: retained weight payloads + scales + metadata + packing/alignment overhead. Lower bits reduce bytes per weight; physical removal reduces retained parameters. Storing zeros does not earn removal savings. Total serving memory also includes context state and runtime buffers, so enforce the deployment budget separately from the weight-memory reward. Real latency and throughput still require measurement.

“Without reducing the score” needs a repeatable acceptance rule and explicit tolerance for noisy evaluations. Default rule: a paired, task-matched, sequential non-inferiority test with a 3-point margin, one-sided 95% confidence and 95% power. It is sized from a measured source-versus-source (A/A) calibration. The task-pool reliability criterion is fixed only after a feasibility screen has been measured. Preserve the required quality floor; do not treat tiny stochastic score changes as proof of improvement or regression. “Lightest” means the best feasible candidate found, not a demonstrated global optimum.

## 10. Relationship to existing quantization work
The main comparison discussed five lines of work: our Prism dynquant, Unsloth Dynamic Quants, Sherry, Bonsai/PrismML, and Dettmers' runtime dynamic compression. These are not five fully independent algorithmic categories.

All are discussed as data-guided approaches in the broad calibration sense above. That does not establish an explicit imatrix stage in all five. Sherry uses quantization-aware training data; Bonsai's exact recipe was not verified. Data use alone is not our distinguishing claim.

### 10.1 Core comparison: dynquant, Unsloth and Dettmers

| Dimension | Our Prism dynquant | Unsloth Dynamic Quants | Dettmers approach |
|---|---|---|---|
| Core principle | Upgrade sensitive parts of a cheap base | Protect sensitive parts with higher precision | Repeatedly probe and allocate under a memory budget |
| Calibration role | Imatrix plus separate ranking data | Guides quantization/sensitivity; complete recipe not established here | Estimates degradation from compression choices |
| Importance signal | KL improvement versus full-precision reference after one group upgrade | Public earlier Dynamic 4-bit blog discusses weight/activation errors and selective protection | Changes in calibration log-perplexity under candidate compression settings |
| Allocation unit | Layer groups: FFN or mixer in the discussed builds | Selected tensors/modules, depending on model/release | Layer compression choices; MoE also includes expert-removal choices |
| Probe baseline | Same fixed low-precision base | Complete scoring loop unknown here | Current compressed allocation |
| Selection rule | Rank isolated positive gains and select upgrades | Complete recipe not established here | Budget allocation followed by renewed sensitivity probing |
| Revisit combined allocation? | Not in the discussed ranking loop | Not established from the public descriptions reviewed | Yes; iterative re-measurement is central |
| Gradient weight training? | No in the current dynquant pipeline | No in the post-training approach discussed | Not in the sensitivity-probing allocator |
| Runtime behavior | Fixed shipped allocation | Fixed allocation in the discussed GGUFs | Offline allocation is separate from dynamic MoE expert residency |
| Important limit | Isolated upgrades can interact; K94 ranked Q8 upgrades but deployed Q4 | Shared principle does not prove identical algorithms; earlier blog is not a full v3 specification | Runtime expert swapping is MoE-specific, not required for offline allocation |

### 10.2 Sherry and Bonsai
Sherry represents the quantization-aware training side: the model adapts under a low-bit constraint, rather than only choosing an approximation of fixed weights. It does not follow that it computes an imatrix or an equivalent separate table.

Bonsai was discussed as another very-low-bit model effort. Its internal calibration/training/allocation recipe was not verified in this conversation; do not present it as identical to Sherry or claim a particular hidden algorithm.

### 10.3 What we borrow and what we investigate
Borrow supported quantization formats and builders. Investigate the precision-mask learner driven by executable capability environments, sanity constraints, iterative verification, optional weight adaptation and a quality-constrained byte objective. Existing task-aware quantization, pruning and distillation remain strong baselines; the combination's value must be demonstrated, not asserted through branding.

## 11. Existing foundation, artifact building and release
The earlier project review established that prism-private dynquant produces standalone mixed-precision artifacts, including K34 and K94. Our starting point is therefore an existing full-model → allocation → standalone-weights path, not a claim that standalone quantization is new. No new benchmark results or independent re-audit are introduced by this design record.

Dynamic quantization supplies the existing allocation-and-build path. The proposed learner would run candidates, collect failures, optionally adapt weights and revisit allocations. Imatrix statistics and isolated precision-upgrade measurements can initialize the search; neither is an unquestionable hard gate. These learning extensions remain to be demonstrated.

After learning: freeze the selected candidate; build supported standalone weights; physically remove structures where applicable; verify that the artifact represents the intended allocation; run final held-out capability and sanity checks; benchmark memory, latency and throughput under the intended context/concurrency. Compare with strong quantized and smaller-model baselines at matched budgets. Release the weights and required runtime configuration, with no full-model dependency. Runtime: a pinned llama.cpp/GGUF build, forked where removal operators need it. The total-memory constraint gates acceptance; latency and throughput are measured and reported, not gated.

## 12. Conversation decisions and open design choices
This is a synthesized record of the discussion, not a verbatim transcript. Later decisions below supersede the earlier speculative-serving pivot.

AGREED DECISIONS
• The product is a standalone specialist. The full model is development-time machinery only.
• The central learned artifact is a multi-tier precision mask, not merely a keep/drop mask.
• The user supplies an executable calibration environment: tasks/prompts, tools and verifiers.
• We begin with a general-data sanity mask and continue checking baseline behavior.
• Support mask-only and mask-plus-weights optimization.
• Combine task verifiers, complaint-derived behavioral checks and optional full-model reference comparison.
• Preservation and improvement are different objectives. Improvement removes teacher matching as the target-task ceiling.
• Once required quality is achieved, reward actual deployed-weight memory reduction while maintaining the constraints.
• Revisit the mask as allocations and weights change; the capability is not assumed to occupy a fixed independent pocket.

DECIDED IN SPEC TUNING (October 12, 2026)
The learner driver: failures feed back into allocation. The objective is built weight bytes, with total memory as a separate constraint. Sanity uses floors, a size-matched cap and behavioral gates. Search starts at layer groups from the sanity mask. KL is proxy-only. Acceptance uses sequential non-inferiority. Runtime is pinned llama.cpp/GGUF. The environment uses a single entry-point contract. Weight training goes through the deployed builder.

DECIDED NOVEMBER 5, 2026
Classifier environments may use the one-shot path (§2): solo probes, an additive per-trajectory surrogate, a fidelity objective, one build, and the same gate and exam.

STILL OPEN (left to the implementation goal or to engineering)
Which removal operators the first build supports; the first source model and environment; the definition of done and the first claim; the exact proxy-refresh and penalty rules; alternating versus joint weight updates; detector thresholds; stopping budget; hardware, compute and authorization. These choices remain open, not completed features.

CORRECTIONS PRESERVED FOR FUTURE DISCUSSIONS
The earlier Aperture speculation architecture solves a different deployment problem and is outside this design. Agreement with the full model is not task success, and token agreement is not an established explanation of task recovery. An imatrix is not universal to all data-guided compression methods. Do not assert a fully known Unsloth v3 or Bonsai recipe. The current design unifies allocation and optional training without claiming that every prior method used the same algorithm.

## 13. Reference pointers and evidence boundaries
Project context starts at prism-private dynamic quantization and K34/K94. See the curated local research library for pinned sources, recipes, receipts and release-card caveats. This design record introduces no new model evaluation.

Unsloth public description: https://unsloth.ai/blog/dynamic-4bit — earlier Dynamic 4-bit sensitivity analysis and selective protection; not a complete v3 recipe.
Dettmers paper: https://timdettmers.com/papers/runtime-dynamic-compression.pdf — offline iterative sensitivity/budget allocation and a distinct MoE runtime-residency mechanism.

Use this record as the reference for subsequent architecture discussions. The next design task is choosing how the highlighted precision-mask learner turns environment feedback and quality constraints into executable allocations.

Chantless · Ship the skill. Not the whole model.


## 14. Later discussion: shared weights and multiple masks

### Two delivery options

We can ship ready-made specialist weights, or ship a shared source checkpoint plus masks that a compatible loader applies locally.

 A ready-built artifact is the simplest deployment: no local quantization step. A shared source lets users obtain additional allocation prescriptions without downloading a new full checkpoint each time.

 The shared-source route may require quantization metadata such as an imatrix, codebooks or scales, along with a compatible builder and pinned source revision. A precision mask alone is not automatically a complete bit-for-bit quantization recipe.



Two packaging options for the same standalone-runtime goal.

### Load a skill without loading the whole model

A skill-aware loader reads the source in chunks, skips removable components, and packs retained components at the requested precision. The full source need not first occupy GPU memory.

 The source can remain on disk while the loader creates the selected runtime representation. Host memory, conversion workspace and loading time still matter.

 Applying a mask after loading all full-precision weights does not produce the same residency saving. Cached compiled variants save repeated conversion work but consume storage.



Source on disk ≠ full model resident in GPU memory.

### One source, multiple skills and budgets

Offer a family of masks: coding · compact, coding · balanced, translation · compact. Skill and resource budget are separate choices.

 Each tier needs its own measured quality–memory tradeoff. More bits do not automatically guarantee a higher task score, and masks need not be nested or share one universal importance order.

 Keep all masks available while activating one specialist at a time. Switching may require loading or converting different representations. This saves redundant downloads only relative to the packaging baseline; the initial source download may be much larger than a single small quant.



Quality labels must come from evaluation, not just nominal bit widths.

### Multiple active skills: a union of representations

For concurrent skills, share identical weight representations. The memory requirement is their union—not automatically their sum, and not automatically the largest skill alone.

 If two masks need the same component, same underlying weights and same format/quantization metadata, it can be stored once. A 2-bit version and an 8-bit version are not automatically interchangeable. The exact key includes weight revision, component, format and quantization parameters.

 A specialized shared encoding could change the accounting; that is an additional implementation choice. Different active models may also need separate context state. Shared storage does not mean merging the masks creates a model with both capabilities.



Toy union below counts payload bytes only, with compatible metadata assumed.

### Weight training changes what can be shared

Mask-only variants can share the original source. Separately trained specialists need their weight changes too. A mask cannot reconstruct values changed by training.

 Ship compatible deltas, adapted source weights or final specialist artifacts. Components with different trained values cannot share the same representation merely because their tensor names and bit widths match.

 Reference checkpoint IDs, mask schema, format settings and weight-update lineage belong in the artifact contract. The compact “one source plus many masks” story is cleanest for mask-only specialization.



Download savings, disk footprint and concurrent runtime memory are different claims.

### Our existing dynquant is the starting point

Our discussed dynquant already builds standalone mixed-precision weights. It combines imatrix-guided quantization with one-group-at-a-time precision upgrades.

 Create a cheap base and a richer donor. Upgrade each group separately against the same base and measure KL improvement versus the full-precision reference. Rank positive gains, select groups and build the combined allocation.

 The discussed Qwen builds use FFN/mixer layer groups. K94 ranked Q8 upgrades but deployed Q4 upgrades. Combined effects need not equal the sum of isolated gains. This pipeline does not gradient-train source weights.

 K34/K94 are existing dynamic-quantization foundations, not newly reproduced results here.



Chantless extends an existing standalone-weight path; it does not invent quantization.

### Dynquant, Unsloth, Dettmers: same family, different loops

All use data to guide compression. The established distinction is how sensitivity becomes an allocation—especially whether it is remeasured after the model changes.

   Allocation procedures discussed    Dimension  Our dynquant  Unsloth  Dettmers      Signal  KL gain versus full precision  Public error analysis and selective protection; full recipe unknown here  Calibration log-perplexity degradation    Probe baseline  Same low-bit base  Complete loop unverified  Current compressed allocation    Selection  Rank isolated positive gains  Model-specific mixed precision  Budget allocation, then re-probe    Weight training  No in this pipeline  No in discussed PTQ approach  Not in sensitivity allocator    Runtime  Fixed allocation  Fixed GGUF allocation  Offline allocation; separate MoE residency mechanism      Unsloth’s earlier Dynamic 4-bit blog describes preserving sensitive modules. It is not a full specification of a later v3 recipe. Avoid pretending the unknowns prove either identical algorithms or fundamental differences.



Dettmers’ offline allocator remains relevant without adopting runtime expert swapping.

### Sherry and Bonsai are not “just another imatrix”

Sherry brings training under low-bit constraints. Bonsai was discussed as a very-low-bit model effort, but its internal recipe was not verified.

 Data plays a broad calibration role across the comparison, but that does not mean all methods compute an imatrix. Quantization-aware training uses loss and gradients to adapt weights under an approximate low-bit forward pass. That is different from collecting a separate activation-importance table.

 We should not claim Bonsai’s hidden pipeline equals Sherry’s, nor claim a particular training cost or low-bit accuracy without a verified source. The five main lines discussed were dynquant, Unsloth, Sherry, Bonsai and Dettmers.



The useful axes are allocation rule, representation, weight adaptation and runtime behavior—not “uses data”.

### MatFormer and Matryoshka solve different pieces

MatFormer nests model widths inside shared weights. Matryoshka attribution learns component scores across budgets while keeping model weights fixed.

 MatFormer trains nested subnetworks, primarily through feed-forward width. It supports different deployment sizes but does not inherently deliver different capabilities or bit widths.

 Aryaman Arora’s Matryoshka attribution is relevant to our inner mask learner: evaluate selected components together across budgets. We could use its scores to initialize allocations or adapt the method from keep/drop decisions to precision choices. Our environment supplies the objective; it does not replace the need for a good optimizer.

 Nested masks can simplify deployment, but the best set at a small budget need not be contained in the best set at a larger budget. Measure the cost of imposing nesting.



Attribution, shared subnetworks and low-bit encoding are related ideas—not the same method.

### Shared downloads already have relevant prior art

Any-Precision, MatQuant and BitStack explore ways to support different memory or precision choices without fully independent checkpoints.

 Any-Precision builds a representation usable at multiple bit widths. MatQuant jointly trains nested quantized representations. BitStack uses stacked residual components to vary the memory–quality tradeoff.

 Our proposed distinction is an environment-driven learner that finds capability-specific allocations under quality and memory constraints, optionally with weight adaptation. One-source/many-budgets packaging is useful, but not an exclusive invention.

 Strong baselines also include task-aware quantization, pruning plus recovery, smaller models with fine-tuning, and distillation. Novelty and customer value must be earned against them.



Product opportunity: the integrated environment-to-specialist loop—not a claim that every ingredient is new.

### More compute should buy better candidates—not just more fitting

The central experiment: hold the specification fixed, spend more useful search compute, and measure the quality–memory frontier on unseen tasks.

 Map-reduce fits independent candidate and task-instance evaluations: map to outcomes and cost, reduce to summaries and decisions. Stateful trajectories retain their dependencies. Parallelism reduces wall time, not necessarily total compute. Weight training can use ordinary data parallelism.

 The environment is reusable across source models and hardware budgets. Its coverage and verifiers are the customer-facing leverage; reliable optimization, real savings and economical construction remain our burden.

 Open choices: exact learner, legal mask granularity, weight-update schedule, sanity floors, comparison horizons, noise tolerances, stopping rules, hardware and runtime. No guaranteed scaling law or best possible allocation is claimed.



A weak environment can teach shortcuts. Keep held-out variation and adversarial verifier checks.

### Everything together: learn here, run there

Development is iterative. Deployment is standalone. Learn the allocation through experience, optionally repair the weights, then build and ship only the selected specialist.

 The old Aperture walkthrough was a speculative-serving proposal that kept a full verifier at runtime. That is not this design. Its “two fates” product-of-agreements story is not an established model of task success and should not be carried into Chantless as evidence.

 Our current commitments: executable environment; initial and continuing sanity protection; multi-tier precision mask; mask-only or mask-plus-weights; preservation or improvement; behavioral verification; quality-gated actual-byte savings; final independent evaluation; standalone deployment.

 Keep the optimizer and unsupported precision operations as open engineering choices. The finished weights—not the mask picture—must prove the required behavior and cost.



Source of truth: current conversation design, with known-method descriptions separated from proposed work.

## Reading this directory
Open explainer.html directly in a browser for the 14-step Aperture-style walkthrough: one diagram that assembles step by step, with Next/Back, arrow keys, replay and ?step=N. The detailed 27-step version is explainer-detailed.html. No server or dependencies are required. In the detailed version, arrow keys (outside focused controls), chapter buttons and the step selector navigate. Use ?step=N for a direct link. Read everything shows the whole narrative; Print / save PDF prints it. Expand Go deeper for detailed context in the walkthrough. Toy memory controls are illustrative payload calculations, not measured quality or runtime results.
