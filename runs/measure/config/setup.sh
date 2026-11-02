#!/bin/bash
# Setup for the 50x8 reliability measurement pod. Same pins as the pilot.
set -x
mkdir -p /root/os/{models,logs,measure} && cd /root/os
hf download XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B --local-dir models/mimo9b > logs/dl.log 2>&1 &
git clone -q https://github.com/ggml-org/llama.cpp && cd llama.cpp && git checkout -q 9adc7f420c37641921b32e326b3d4a538256b878
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=80 -DLLAMA_CURL=OFF > ../logs/cmake.log 2>&1
cmake --build build -j 64 --target llama-server > ../logs/build.log 2>&1; echo BUILD_EXIT=$? >> ../logs/build.log
uv venv -q -p 3.12 .venv && uv pip install -q -p .venv -r requirements/requirements-convert_hf_to_gguf.txt
cd /root/os && git clone -q https://github.com/XiaomiMiMo/mimoagent && cd mimoagent && git checkout -q 467f0a19016f0ac4d63b8d17a1f0da9ba07f232c && uv sync -q
wait
cd /root/os/llama.cpp && .venv/bin/python convert_hf_to_gguf.py ../models/mimo9b --outtype bf16 --no-mtp --outfile ../models/mimo9b-bf16.gguf > ../logs/convert.log 2>&1
echo CONVERT_EXIT=$? >> ../logs/convert.log
echo SETUP_DONE > /root/os/logs/setup.done
