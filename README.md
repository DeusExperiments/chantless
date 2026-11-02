# CHANTLESS

![Deus Experiments logo](assets/brand/deus-experiments-logo.png)

Your taste. Multiplied.

Chantless turns a general model into a standalone specialist for a user-defined workload by learning a per-tensor precision mask, with whole sub-blocks physically removed, driven by task-failure feedback.

## Current state

- [MVP](mvp/README.md) (commit `ee4ac3e`): the full environment-to-specialist loop runs end to end on Qwen3-0.6B. Run `box-mvp-001` built a 669.7 MB, 7.13 bpw specialist that passed the select gate; the held-out exam was inconclusive (delta −0.006, lower bound −0.046 against a 3-point margin). Its `codetasks` environment is a pipeline regression test, not product evidence.
- Next: port sub-block removal to `qwen35` on Qwen3.5-4B, then extract two orthogonal specialists (text-to-SQL and long-document extraction). Not started.
- The MiMo precision-search pilot is paused since September 27. The correction-learning roadmap is a plan only.

## Reading map

- [Current design](explainer/design-notes.md), [detailed explainer](explainer/explainer-detailed.html), [project overview](work/chantless-project-overview/chantless-project-map.html).
- [MVP run notes](.local/mvp/RUN_NOTES.md) and [box-mvp-001 receipts](.local/mvp/box-runs/box-mvp-001/).
- [Dynamic-quantization library](prism-dynquant-work/README.md): current evidence, mechanisms, original recipes, results and model cards.
- [Artifact catalog](prism-dynquant-work/remote-artifacts/artifact-catalog.json): retained model and evaluation links.
- [Local demo and film viewer](.local/krishna-handoff-20261005/viewer/index.html): Minecraft, Lego, website and kernel demos, plus film assets.
- [Experiment manual](manual/MANUAL.md) and [paused pilot status](runs/search-pilot/STATUS.md).
- [Current local destinations](.local/current-context-links.md) and [cleanup receipt](.local/cleanup-receipt.json).

## Workspace boundary

Private origin: [DeusExperiments/chantless](https://github.com/DeusExperiments/chantless). This is a fresh Git repository without imported history. The original OnlySkills folder and old Git backup were removed on October 5, 2026; their deletion receipts remain in `.local/`.

`references/prism-dynquant-work` points to the curated local library. Keep `references/`, `prism-dynquant-work/` and `.local/` ignored. Do not force-add or publish private source snapshots, credentials or restricted weights. Retained source pins, recipes, scores and metric denominators remain explicit. New commits, pushes, publication and paid compute require separate authorization.

The 2026-10-12 curation removes earlier research from this checkout and regenerates navigation around dynamic quantization. Original records are recoverable outside the checkout; the exact removal and edit manifest is referenced by the cleanup receipt. Curation is not a new experimental result. The 2026-10-27 cleanup moved regenerable MVP scratch to the Trash; see `.local/cleanup-receipt-20261027.json`.
