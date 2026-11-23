#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PATCH_FILE="$REPO_ROOT/mvp/patches/llama-cpp-a01dff9-chantless.patch"
LLAMA_COMMIT="a01dff908b9c9311d7af316bba3c918063749bd0"
LLAMA_DIR_NAME="llama-a01dff9"
APPLY_PATCH=1

PREFIX="${CHANTLESS_TOOLCHAIN_PREFIX:-$REPO_ROOT/.local/toolchain}"
BACKEND="auto"
CUDA_ARCH="${CHANTLESS_CUDA_ARCH:-native}"
CUDA_HOST_COMPILER="${CHANTLESS_CUDA_HOST_COMPILER:-}"
JOBS=""
WITH_FLA=0
DRY_RUN=0

usage() {
    cat <<EOF
Usage: $(basename "$0") [options]

Build pinned llama.cpp ($LLAMA_COMMIT) with the Chantless patch and C++ tools (traj-kl, logodds, tokdocs).

Options:
  --prefix DIR                  Install/build root (default: \$REPO_ROOT/.local/toolchain)
  --llama-ref REF               Git commit or ref in ggml-org/llama.cpp (default: $LLAMA_COMMIT)
  --llama-dir-name NAME         Subdirectory under --prefix for llama.cpp checkout (default: $LLAMA_DIR_NAME)
  --no-patch                    Skip applying mvp/patches/llama-cpp-a01dff9-chantless.patch
  --backend MODE                auto | cuda | metal | cpu (default: auto)
  --cuda-arch ARCH              CMAKE_CUDA_ARCHITECTURES when backend=cuda (default: native -> single-arch nvidia-smi)
  --cuda-host-compiler PATH     CMAKE_CUDA_HOST_COMPILER override (auto-prefers g++-12 over gcc-13 ICE when installed)
  --jobs N                      Parallel build jobs (default: auto cpu count, capped at 64)
  --with-fla                    Check/install flash-linear-attention (for Qwen3.5 GDN on CUDA)
  --dry-run                     Print resolved build plan and commands without executing
  -h, --help                    Show this help
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --prefix) PREFIX="$2"; shift 2 ;;
        --llama-ref) LLAMA_COMMIT="$2"; shift 2 ;;
        --llama-dir-name) LLAMA_DIR_NAME="$2"; shift 2 ;;
        --no-patch) APPLY_PATCH=0; shift ;;
        --backend) BACKEND="$2"; shift 2 ;;
        --cuda-arch) CUDA_ARCH="$2"; shift 2 ;;
        --cuda-host-compiler) CUDA_HOST_COMPILER="$2"; shift 2 ;;
        --jobs) JOBS="$2"; shift 2 ;;
        --with-fla) WITH_FLA=1; shift ;;
        --dry-run) DRY_RUN=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

if [[ "$APPLY_PATCH" -eq 1 && ! -f "$PATCH_FILE" ]]; then
    echo "ERROR: patch not found: $PATCH_FILE" >&2
    exit 1
fi

OS="$(uname -s)"
if [[ "$BACKEND" == "auto" ]]; then
    if command -v nvidia-smi >/dev/null 2>&1 || command -v nvcc >/dev/null 2>&1; then
        BACKEND="cuda"
    elif [[ "$OS" == "Darwin" ]]; then
        BACKEND="metal"
    else
        BACKEND="cpu"
    fi
fi

if [[ -z "$JOBS" ]]; then
    if [[ "$OS" == "Darwin" ]]; then
        JOBS="$(sysctl -n hw.ncpu)"
    else
        JOBS="$(nproc 2>/dev/null || echo 8)"
    fi
    if [[ "$JOBS" -gt 64 ]]; then JOBS=64; fi
fi

if [[ "$PREFIX" != /* ]]; then
    PREFIX="$PWD/$PREFIX"
fi

LLAMA_DIR="$PREFIX/$LLAMA_DIR_NAME"
BUILD_DIR="$LLAMA_DIR/build"
LLAMA_BIN="$BUILD_DIR/bin"
TOOLS_BIN="$PREFIX/bin"

CMAKE_FLAGS=(
    -S "$LLAMA_DIR"
    -B "$BUILD_DIR"
    -DLLAMA_CURL=OFF
    -DBUILD_SHARED_LIBS=ON
    -DCMAKE_BUILD_TYPE=Release
)

case "$BACKEND" in
    cuda)
        if [[ "$CUDA_ARCH" == "native" || "$CUDA_ARCH" == "auto" ]]; then
            if command -v nvidia-smi >/dev/null 2>&1; then
                DETECTED_ARCH="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n 1 | tr -d '.[:space:]' || true)"
                if [[ -n "$DETECTED_ARCH" ]]; then
                    CUDA_ARCH="$DETECTED_ARCH"
                fi
            fi
        fi
        CMAKE_FLAGS+=(-DGGML_CUDA=ON "-DCMAKE_CUDA_ARCHITECTURES=$CUDA_ARCH")
        if [[ -z "$CUDA_HOST_COMPILER" ]] && command -v g++-12 >/dev/null 2>&1; then
            CUDA_HOST_COMPILER="$(command -v g++-12)"
        fi
        if [[ -n "$CUDA_HOST_COMPILER" ]]; then
            CMAKE_FLAGS+=("-DCMAKE_CUDA_HOST_COMPILER=$CUDA_HOST_COMPILER")
        fi
        ;;
    metal)
        CMAKE_FLAGS+=(-DGGML_METAL=ON)
        ;;
    cpu)
        ;;
    *)
        echo "ERROR: unsupported backend '$BACKEND' (expected auto|cuda|metal|cpu)" >&2
        exit 2
        ;;
esac

CXX_COMPILER="${CXX:-}"
if [[ -z "$CXX_COMPILER" ]]; then
    if [[ "$OS" == "Darwin" ]]; then
        CXX_COMPILER="clang++"
    else
        CXX_COMPILER="g++"
    fi
fi

CXX_EXTRA_FLAGS=(-O3 -std=c++17)
if [[ "$OS" == "Darwin" ]]; then
    if command -v brew >/dev/null 2>&1 && [[ -d "$(brew --prefix libomp 2>/dev/null || true)" ]]; then
        OMP_PREFIX="$(brew --prefix libomp)"
        CXX_EXTRA_FLAGS+=(-Xpreprocessor -fopenmp "-I$OMP_PREFIX/include" "-L$OMP_PREFIX/lib" -lomp)
    fi
    RPATH_FLAG="-Wl,-rpath,$LLAMA_BIN"
else
    CXX_EXTRA_FLAGS+=(-march=native -fopenmp)
    RPATH_FLAG="-Wl,-rpath,$LLAMA_BIN"
fi

run_cmd() {
    if [[ "$DRY_RUN" -eq 1 ]]; then
        printf '[dry-run]'
        printf ' %q' "$@"
        printf '\n'
    else
        "$@"
    fi
}

echo "=== Chantless Toolchain Build Plan ==="
echo "repo_root:    $REPO_ROOT"
if [[ "$APPLY_PATCH" -eq 1 ]]; then
    echo "patch:        $PATCH_FILE"
else
    echo "patch:        (skipped via --no-patch)"
fi
echo "llama_commit: $LLAMA_COMMIT"
echo "llama_dir:    $LLAMA_DIR"
echo "prefix:       $PREFIX"
echo "backend:      $BACKEND"
echo "jobs:         $JOBS"
echo "cxx:          $CXX_COMPILER"
echo "with_fla:     $WITH_FLA"

run_cmd mkdir -p "$PREFIX" "$TOOLS_BIN"

if [[ ! -d "$LLAMA_DIR/.git" ]]; then
    run_cmd git init -q "$LLAMA_DIR"
    run_cmd git -C "$LLAMA_DIR" remote add origin https://github.com/ggml-org/llama.cpp.git
    run_cmd git -C "$LLAMA_DIR" fetch -q --depth 1 origin "$LLAMA_COMMIT"
    run_cmd git -C "$LLAMA_DIR" checkout -q FETCH_HEAD
    if [[ "$APPLY_PATCH" -eq 1 ]]; then
        run_cmd git -C "$LLAMA_DIR" apply "$PATCH_FILE"
    fi
else
    echo "=== Reusing existing $LLAMA_DIR ==="
fi

run_cmd cmake "${CMAKE_FLAGS[@]}"
run_cmd cmake --build "$BUILD_DIR" -j "$JOBS" --target llama-server llama-quantize llama-imatrix llama

for tool in traj-kl logodds tokdocs; do
    run_cmd "$CXX_COMPILER" "${CXX_EXTRA_FLAGS[@]}" \
        "$REPO_ROOT/mvp/tools/${tool}.cpp" \
        "-I$LLAMA_DIR/include" "-I$LLAMA_DIR/ggml/include" \
        "-L$LLAMA_BIN" -lllama -lggml -lggml-base \
        "$RPATH_FLAG" \
        -o "$TOOLS_BIN/${tool}"
done

if [[ "$WITH_FLA" -eq 1 ]]; then
    if [[ "$BACKEND" != "cuda" ]]; then
        echo "NOTE: --with-fla skipped because backend=$BACKEND (requires CUDA)"
    elif command -v uv >/dev/null 2>&1; then
        run_cmd uv pip install flash-linear-attention
    else
        run_cmd python3 -m pip install flash-linear-attention
    fi
fi

echo "=== Toolchain binaries ==="
echo "llama_bin: $LLAMA_BIN"
echo "tools_bin: $TOOLS_BIN"
