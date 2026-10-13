#!/bin/bash
# Chantless MVP box setup (Lium 8x RTX 6000 Ada). Same pins as the local build.
# Usage: bash setup_box.sh /root/chantless   (expects ./mvp copied alongside this script's parent)
set -euo pipefail
R=${1:-/root/chantless}
LLAMA_COMMIT=9adc7f420c37641921b32e326b3d4a538256b878
QWEN_REV=c1899de289a04d12100db370d81485cdf75e47ca
mkdir -p $R/{logs,models,bin,runs}
cd $R
command -v uv >/dev/null || (curl -LsSf https://astral.sh/uv/install.sh | sh)
export PATH=$HOME/.local/bin:$PATH
if [ ! -d llama.cpp ]; then git clone -q https://github.com/ggml-org/llama.cpp; fi
cd llama.cpp && git checkout -q $LLAMA_COMMIT && git apply $R/mvp/patches/llama-cpp-9adc7f4-qwen3-subblock-drop.patch
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89 -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release > $R/logs/cmake.log 2>&1
cmake --build build -j 64 --target llama-server llama-quantize llama-imatrix llama > $R/logs/build.log 2>&1
cd $R
g++ -O3 -march=native -fopenmp -std=c++17 mvp/tools/traj-kl.cpp -Illama.cpp/include -Illama.cpp/ggml/include \
    -Lllama.cpp/build/bin -lllama -lggml -lggml-base -Wl,-rpath,$R/llama.cpp/build/bin -o bin/traj-kl
uv venv -q -p 3.12 .venv
uv pip install -q -p .venv torch transformers peft safetensors numpy scipy datasets pytest pyarrow huggingface_hub sentencepiece protobuf -e llama.cpp/gguf-py
.venv/bin/hf download Qwen/Qwen3-0.6B --revision $QWEN_REV --local-dir models/qwen3-0.6b > logs/dl.log 2>&1
.venv/bin/python llama.cpp/convert_hf_to_gguf.py models/qwen3-0.6b --outtype bf16 --outfile models/qwen3-0.6b-bf16.gguf > logs/convert.log 2>&1
sha256sum models/qwen3-0.6b-bf16.gguf | tee logs/source.sha256
nvidia-smi --query-gpu=index,name,memory.total --format=csv | tee logs/gpus.csv
echo SETUP_DONE
