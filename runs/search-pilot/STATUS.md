# Search pilot — status at pause (2026-09-27, ~22:50 IST)

**State:** paused by the owner. Both Lium pods are terminated (Pod B at ~23:05 IST), and every artifact is saved locally; base store checksums verified 15/15.

## Done
| Item | Result | Where |
|---|---|---|
| BF16 pilot, 20 tasks × 2 | 16 valid tasks. Pass rate 46.9% (15/32): 2 tasks 2/2, 11 tasks 1/2, 3 tasks 0/2. 0 doom loops, 0 malformed tool calls | `runs/pilot/results/` |
| Leaky images | 4 of 20 images leak future git history (000543, 001572, 002518, 003021). Excluded per owner | `runs/pilot/results/summary_v2.json` |
| L5 whole-trajectory KL tool | `tools/traj-kl/traj-kl.cpp` (base and kl modes, row-parallel post-processing) | `tools/traj-kl/` |
| Trajectory renderer | Renders with HF `chat_template.jinja`. Cross-checked against llama-server on every trajectory: every template prefix matches and token ids are identical, with 15/15 trajectories OK. Tool-call `arguments` are parsed to objects, as llama-server does | `tools/traj-kl/render_trajs.py` |
| SWE proxy | 15 passing trajectories, 810,203 context tokens (max 92,023), 105,215 non-reasoning targets, subsampled to a fixed 20,000 (seed 20260927) | `artifacts/pod_b_small.tgz` → `proxy/swe_proxy.bin`, `swe_proxy_meta.json` |
| Source base store | 20,000 × 248,320 fp16 log-probs, 9.4 GB. Source NLL 0.05237 nats/token | `artifacts/base_swe/` (+ `SHA256SUMS`) |
| L5 self-check | KL(source, source) = 3.627e-6, which passes the owner tolerance of ≤ 1e-5. Top-1 agreement 1.0 | `proxy/selfkl2.json` |
| First candidate | Uniform Q4_K (5.63 GB file): KL 0.035353, top-1 agreement 0.9741, NLL 0.0524 → 0.0877. Bit-identical across three runs at ubatch 2048 | `proxy/kl_u_q4k*.json` |

## Measured costs (1× A100-80GB, llama.cpp 9adc7f4)
- Base pass over the 810k-token proxy: 233 s.
- One probe: about 55 s to quantize plus about 233 s of KL. GPU forward is 221 s and CPU post-processing 12 s.
- Forward throughput is about 3.7k tokens/s. The cause is unverified; it may be llama.cpp's DeltaNet kernels.
- KL depends on ubatch (0.04803 at 2048 vs 0.05249 at 8192 on one trajectory), so ubatch is pinned at 2048.

## Open decision (blocks the search)
At about 4 min per probe, the pilot search as specified (Q4_K → ~1.6 bpw, re-probe every 8 moves, top-16 per move, about 6,000 probes) needs about 50 h on 8 GPUs, roughly $470. Options raised but not yet decided:
- shorten the proxy context;
- use a coarser re-probe cadence;
- both;
- accept the cost.

## Not started
- general_proxy (FineWeb-Edu, 50k scored tokens) and the size-matched sanity cap.
- The uniform-quant ladder (Q3_K, IQ2_XS, IQ1_M, IQ1_S) and importance statistics for the IQ tiers.
- The dynquant baseline, the search loop itself, and episode evaluations.
- The manual §9.1 L5 equivalence test against stock `--kl-divergence`. Only the self-KL check has been run.

## Costs incurred
- Pod A, 96 GB RTX PRO 6000: ~3.3 h ≈ $4.3.
- Pod B, 8×A100: 3.1 h, $29.40 (final, from `lium ps` before termination).

## Resume checklist
1. Rent a GPU pod and run `runs/measure/config/setup.sh` (same pins: llama.cpp 9adc7f4, mimoagent 467f0a1, `--no-mtp` convert).
2. Upload `tools/traj-kl/*`, `swe_proxy.bin` and `base_swe/`. Build: `g++ -O3 -march=native -fopenmp -std=c++17 traj-kl.cpp -I$L/include -I$L/ggml/include -L$L/build/bin -lllama -lggml -lggml-base`.
3. Check that `base_swe/SHA256SUMS` verifies, then re-run the self-KL (expect 3.6270848e-06) and uniform Q4_K (expect 0.035353101) before any new probe.
