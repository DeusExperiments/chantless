#!/bin/bash
# Pod side of the detector run: run.sh starts one step at a time, from $R/code, logging to $R/logs/STEP.log.
#   setup      uv venv with pinned Transformers and the fast Gated DeltaNet kernels (fla)
#   prep       all three models into the HF cache while prep.py writes the parquet files
#   preflight  3 training steps on the real data (longest batch first) and 2 documents per model
#   train      one pass on all 8 GPUs; BATCH (default 8) is the per-GPU batch
#   score      GPU g scores shard g of calibration, test and Beemo with qwen, then modernbert (full ProseLens is not scored in this run)
#   pack       tar results (no optimizer checkpoints) and write its sha256, for run.sh fetch
set -euo pipefail
R=/root/detect
PY=$R/.venv/bin/python
BATCH=${BATCH:-8}
SETS="$R/data/calibration.parquet $R/data/test.parquet $R/data/beemo.parquet"
export HF_HOME=$R/hf PYTHONUNBUFFERED=1 PATH=$HOME/.local/bin:$PATH
[ "$1" = prep ] || export HF_HUB_OFFLINE=1  # prep cached every model; later steps never touch the Hub
cd $R/code
case $1 in
  setup)
    command -v uv >/dev/null || (curl -LsSf https://astral.sh/uv/install.sh | sh)
    uv venv -q -p 3.12 $R/.venv
    uv pip install -q -p $R/.venv torch transformers==5.21.0 accelerate datasets pyarrow numpy flash-linear-attention
    $PY -c "import torch, transformers, fla; print('torch', torch.__version__, 'transformers', transformers.__version__, 'gpus', torch.cuda.device_count())"
    nvidia-smi --query-gpu=index,name,memory.total --format=csv ;;
  prep)
    $PY -c "from huggingface_hub import snapshot_download; from common import MODELS; [snapshot_download(r, revision=v) for r, v in MODELS.values()]" &
    dl=$!
    $PY prep.py $R/data
    wait $dl ;;
  preflight)
    $R/.venv/bin/torchrun --nproc-per-node 8 train.py $R/data $R/runs/preflight --batch $BATCH --max-steps 3
    $PY score.py qwen $R/runs/preflight/qwen $SETS --nshard 8 --batch 16 --preflight
    $PY score.py modernbert $R/runs/preflight/modernbert $SETS --nshard 8 --batch 16 --preflight
    $PY score.py proselens $R/runs/preflight/proselens $SETS --nshard 8 --batch 1 --preflight ;;
  train)
    $R/.venv/bin/torchrun --nproc-per-node 8 train.py $R/data $R/runs/qwen --batch $BATCH ;;
  score)
    pids=()
    for g in 0 1 2 3 4 5 6 7; do
      (export CUDA_VISIBLE_DEVICES=$g TRITON_CACHE_DIR=$HOME/.triton/cache-gpu$g  # one cache per process: shared ones race
       $PY score.py qwen $R/runs/scores/qwen $SETS --path $R/runs/qwen/final --shard $g --nshard 8 --batch 16
       $PY score.py modernbert $R/runs/scores/modernbert $SETS --shard $g --nshard 8 --batch 16) > $R/logs/score-gpu$g.log 2>&1 &
      pids+=($!)
    done
    for p in "${pids[@]}"; do wait $p; done ;;
  pack)
    keep=()
    for p in data/prep.json logs runs/preflight/train.json runs/qwen/train.json runs/qwen/final runs/scores; do
      [ -e $R/$p ] && keep+=($p)
    done
    tar czf $R/detect-results.tgz -C $R "${keep[@]}"
    sha256sum $R/detect-results.tgz > $R/detect-results.tgz.sha256 ;;
esac
echo "STEP_DONE $1"
