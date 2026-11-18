# CHANTLESS

![Deus Experiments logo](assets/brand/deus-experiments-logo.png)

Your taste. Multiplied.

Chantless starts from a frozen reference model and learns a mask (precision tiers, sub-block drops, MLP neuron widths, vocabulary keep-set, grid-level moves and neuron switches) that turns the frozen base into a compact standalone specialist and lets it keep learning from labelled examples.

## Current state

- **Qwen3.5-0.8B detector & Learned Apprentice** ([mvp/README.md](mvp/README.md), [mvp/detect/](mvp/detect/)): `qwen35` sub-block removal verified (Issue #1 closed). First full pass (`detect-cls-400mb-001`, 396.5 MB) passed both the select gate and the single-use 50,808-document sealed exam. Further compressed to 200,296,000 B (`q4cut50`: mixed tiers, sub-block drops, 50% MLP width cut, 80k vocabulary trim). Trained on 20 lesson passages from 10 Fold-0 articles plus replay, the Learned Apprentice (`combo.gguf`, 200,296,000 B) improved from 8/25 to 25/25 on held-out fresh AI passages with 0/25 human originals flagged.
- **Classifier one-shot rank-fidelity search** ([mvp/chantless/oneshot.py](mvp/chantless/oneshot.py), [explainer/design-notes.md](explainer/design-notes.md)): single-group solo probes + additive score prediction + Spearman rank-fidelity solve under exact byte pricing reproduced the 200 MB specialist in a single build (`f200`, 200,240,128 B, 47 select documents lost vs 46 hand-built).
- **Gemma 4 E2B 10% mask & Learned Apprentice** ([mvp/gemma4/README.md](mvp/gemma4/README.md)): Issue #2 closed. One-shot mask (`gemma4-e2b-detector-mask10.gguf`, 930,971,136 B = 10.00% of the 9.31 GB BF16 GGUF) passed both the select gate (42 / 14,227 lost, L = −0.0055) and the single-use 51,066-document sealed exam (99.76% agreement, L = −0.0037). On the same 20 lesson passages, the neuron switch mask (`i1`, 3,569 of 96,768 neurons switched off, zero weight values changed) moved from 8/25 to 18/25 on the 25 held-out fresh AI passages with 0/50 human texts flagged while passing the select gate.
- **Unified `chantless.mask/v3` pipeline & `cli build`** ([mvp/chantless/](mvp/chantless/), [explainer/design-notes.md](explainer/design-notes.md)): Issue #3 closed. Architecture adapter (`qwen3`, `qwen35`, `gemma4`), lossless dead-tensor pre-pass, Wanda MLP width slicing, merge-closed Rule-P vocabulary trimming (`tokdocs`), Apprentice switches/sparse grid deltas, sharded bit-exact `parallel_quantize`, and packed FlashAttention `logodds` all run from one codebase; thin `v3` masks (`31–77 KB`) deterministically materialize Qwen `f200` (`200,240,128 B`, 47 lost) and Gemma `mask10` (`930,971,136 B`, $L = -0.0056$) via `python -m chantless.cli build`.
- **Earlier MVP & paused pilot:** `box-mvp-001` ([mvp/README.md](mvp/README.md)) ran the generative loop end to end on Qwen3-0.6B (`codetasks` regression environment). The MiMo 9B SWE-bench precision-search pilot remains paused since September 27 (`runs/search-pilot/STATUS.md`).
- **Next:** extend one-shot extraction (`traj-kl`) and switch Apprentice (`learn_gates`) + unified `llama.cpp` patch to generative environments ([Issue #4](https://github.com/DeusExperiments/chantless/issues/4)), then prove `<=10%` mask + switch Apprentice on stock Apache-2.0 `Qwen3.5-4B` and prepare release ([Issue #5](https://github.com/DeusExperiments/chantless/issues/5)).

## Reading map

- [Current design](explainer/design-notes.md), [detailed explainer](explainer/explainer-detailed.html), [project overview](work/chantless-project-overview/chantless-project-map.html).
- [Replication spec](.local/replication/CHANTLESS-REPLICATION.md), [interactive map](.local/replication/chantless-map.html) and [23-stage wiring map](.local/replication/chantless-wiring-map.html).
- [Qwen & one-shot receipts](.local/detect/), [Gemma 4 receipts](.local/gemma4/), [MVP run notes](.local/mvp/RUN_NOTES.md) and [box-mvp-001 receipts](.local/mvp/box-runs/box-mvp-001/).
- [Dynamic-quantization library](prism-dynquant-work/README.md): current evidence, mechanisms, original recipes, results and model cards.
- [Artifact catalog](prism-dynquant-work/remote-artifacts/artifact-catalog.json): retained model and evaluation links.
- [Local demo and film viewer](.local/krishna-handoff-20261005/viewer/index.html): Minecraft, Lego, website and kernel demos, plus film assets.
- [Experiment manual](manual/MANUAL.md) and [paused pilot status](runs/search-pilot/STATUS.md).
- [Current local destinations](.local/current-context-links.md) and [cleanup receipt](.local/cleanup-receipt.json).

## Workspace boundary

Private origin: [DeusExperiments/chantless](https://github.com/DeusExperiments/chantless). This is a fresh Git repository without imported history. The original OnlySkills folder and old Git backup were removed on October 5, 2026; their deletion receipts remain in `.local/`.

`references/prism-dynquant-work` points to the curated local library. Keep `references/`, `prism-dynquant-work/` and `.local/` ignored. Do not force-add or publish private source snapshots, credentials or restricted weights. Retained source pins, recipes, scores and metric denominators remain explicit. New commits, pushes, publication and paid compute require separate authorization.

The 2026-10-12 curation removes earlier research from this checkout and regenerates navigation around dynamic quantization. Original records are recoverable outside the checkout; the exact removal and edit manifest is referenced by the cleanup receipt. Curation is not a new experimental result. The 2026-10-27 cleanup moved regenerable MVP scratch to the Trash; see `.local/cleanup-receipt-20261027.json`.
