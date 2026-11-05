# Gemma 4 E2B: 10% mask and Learned Apprentice

Code for the two Gemma 4 E2B detector results of November 5, 2026. Scripts expect the pod layout `/root/g/` and were run as
standalone scripts, outside `chantless.pipeline`. Models, receipts and scores are on the private ModelScope repo
`tokenbender/chantless-detect-local-archive` under `gemma4/mask10/` and `gemma4/apprentice/`; no data is in this repository.

## Runtime
llama.cpp `a01dff908b9c9311d7af316bba3c918063749bd0` plus `../patches/llama-cpp-a01dff9-gemma4-subblock-removal.patch`
(written by `mask/patch_g4.py`). In `gemma4` the attention, MLP and per-layer-embedding sub-blocks may be absent: absent
attention passes the residual stream through, an absent MLP or per-layer embedding adds nothing. Per-layer MLP width is native.
Scoring uses `../tools/logodds.cpp` built with `%.17g` output (`sed 's/%\.7g/%.17g/'`) and `../tools/tokdocs.cpp`.

## 10% mask (`mask/`, issue 2)
Classifier one-shot path: solo probes, additive per-document surrogate, fidelity objective, exact byte model, one build.
- `gprep.py`, `gprompts.py`, `tok2trj.py`: Gemma token files for the stores and gates (same document ids as the Qwen run).
- `ggroups.py`: 108 groups; the 60 dead K/V tensors of the shared-KV layers are removed first (scores unchanged).
- `gdrv.py`: probes and builds (precision tiers, drops, MLP widths by Wanda-style neuron order from the imatrix).
- `gsolve.py`: the solve; `gvocab.py`, `gtrim.py`: vocabulary trim to 80,000 tokens (rule P, merge-closed).
- `gbuild.py`: build, trim, retokenize, gate on select and exam; `gscore.py`: multi-GPU scoring.
Result: 930,971,136 B (10.0% of the 9,311,299,104 B BF16 file); select L -0.0055 and exam L -0.0037, both pass.

## Learned Apprentice (`apprentice/`)
The Qwen recipe (same 20 lesson passages, fold 0, same 25-article fresh test, seeds, replay, terms, 100 steps, rates) on the 10% mask.
- `../lean.py`: Gemma 4 E2B text model in plain PyTorch (the model used to fine-tune the detector).
- `gstudent.py`: the mask file as a PyTorch student on lean.py, structure as in the patched graph (absent sub-blocks, cut widths, 80k vocabulary).
- `glearn.py`: `check` (student vs llama.cpp), `train o0|i1` (value mask, switch mask), `combo`, `judge`.
- `aprep.py`: token files for lessons, fresh test, calibration and select; scores the mask as a check.
Result at each file's own cutoff: base 8/25 AI passages, value mask 8/25, switch mask 18/25 (3,569 of 96,768 neurons off), combined 18/25;
0 of 50 human texts flagged; gate (select side 1 against the source) passes for all. The lesson and fresh texts are not in this repository.
