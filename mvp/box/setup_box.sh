#!/bin/bash
# Chantless box setup (Lium). Same pins as the local build.
# Usage: bash setup_box.sh /root/chantless   (expects ./mvp copied alongside this script's parent)
# Model and GPU come from the environment; defaults reproduce the Qwen3-0.6B MVP box:
#   MODEL_REPO=Qwen/Qwen3.5-4B MODEL_REV=<commit> MODEL_NAME=qwen3.5-4b CUDA_ARCH=89 bash setup_box.sh
# The converter writes the text trunk only: vision weights go to a separate mmproj file that is not
# produced here, and --no-mtp leaves out multi-token-prediction heads.
set -euo pipefail
R=${1:-/root/chantless}
LLAMA_COMMIT=9adc7f420c37641921b32e326b3d4a538256b878
MODEL_REPO=${MODEL_REPO:-Qwen/Qwen3-0.6B}
MODEL_REV=${MODEL_REV:-c1899de289a04d12100db370d81485cdf75e47ca}
MODEL_NAME=${MODEL_NAME:-qwen3-0.6b}
CUDA_ARCH=${CUDA_ARCH:-89}
PATCH=${PATCH:-$R/mvp/patches/llama-cpp-9adc7f4-subblock-drop.patch}
mkdir -p $R/{logs,models,bin,runs}
cd $R
command -v uv >/dev/null || (curl -LsSf https://astral.sh/uv/install.sh | sh)
export PATH=$HOME/.local/bin:$PATH
if [ ! -d llama.cpp ]; then git clone -q https://github.com/ggml-org/llama.cpp; fi
cd llama.cpp && git checkout -q $LLAMA_COMMIT && git apply $PATCH
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=$CUDA_ARCH -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release > $R/logs/cmake.log 2>&1
cmake --build build -j 64 --target llama-server llama-quantize llama-imatrix llama > $R/logs/build.log 2>&1
cd $R
g++ -O3 -march=native -fopenmp -std=c++17 mvp/tools/traj-kl.cpp -Illama.cpp/include -Illama.cpp/ggml/include \
    -Lllama.cpp/build/bin -lllama -lggml -lggml-base -Wl,-rpath,$R/llama.cpp/build/bin -o bin/traj-kl
uv venv -q -p 3.12 .venv
uv pip install -q -p .venv torch transformers peft safetensors numpy scipy datasets pyarrow huggingface_hub sentencepiece protobuf -e llama.cpp/gguf-py
.venv/bin/hf download $MODEL_REPO --revision $MODEL_REV --local-dir models/$MODEL_NAME > logs/dl.log 2>&1
CONVERT_FLAGS=""; case $MODEL_NAME in qwen3.5*) CONVERT_FLAGS="--no-mtp";; esac
.venv/bin/python llama.cpp/convert_hf_to_gguf.py models/$MODEL_NAME $CONVERT_FLAGS --outtype bf16 --outfile models/$MODEL_NAME-bf16.gguf > logs/convert.log 2>&1
sha256sum models/$MODEL_NAME-bf16.gguf | tee logs/source.sha256
nvidia-smi --query-gpu=index,name,memory.total --format=csv | tee logs/gpus.csv
echo SETUP_DONE
