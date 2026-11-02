# Pilot run log

## 2026-09-27
- Pod: Lium `onlyskills-p0` (HUID gentle-orbit-3b), 1x RTX PRO 6000 Blackwell 96 GB, 64 CPU, 251 GB RAM, 2.2 TB free, $1.29/h, TTL 12h. Docker-in-pod works (nvidia runtime present).
- Pins: llama.cpp 9adc7f420c37641921b32e326b3d4a538256b878 (CUDA 13.0, sm_120); mimoagent 467f0a19016f0ac4d63b8d17a1f0da9ba07f232c.
- Source GGUF: `convert_hf_to_gguf.py --outtype bf16 --no-mtp` -> 8,953,803,264 params, block_count 32.
- Pilot tasks: 20 sampled with random.seed(20260927) from the 2,698 code rows (`tasks.json`, `pilot.jsonl`). 2 rollouts each.
- Pilot deviation from the manual: llama-server `-np 6 -c 1572864` (262,144 per slot) to measure real trajectory lengths and speed up throughput. Gates will use `-np 1`.
- Sampling: temperature 0.6, top_p 0.95, top_k 20. Harness: mimocode-agent, docker backend.
- Pipeshift: host key changed (ED25519 SHA256:hx7JR2ZTLPgeXh8J27ZaRyvpld/R3kblYiNrZhA+6g8); not connected.
