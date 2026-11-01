#!/bin/bash
# The first One-Budget Anneal experiment on the 8-GPU pod: setup, the rate sweep on all eight cards at once, then the eight arms at once.
# A card whose arm has ended takes the next patched row, so none idles while rows wait. An arm that dies resumes from its last checkpoint.
# Usage, with all output redirected so lium exec returns:
#   nohup bash /root/n/mvp/detect/anneal_arms.sh > /root/n/anneal/arms.log 2>&1 &
cd /root/n/mvp
export LD_LIBRARY_PATH=/root/n/llama/build/bin PYTHONPATH=/root/n/mvp PYTHONUNBUFFERED=1
export MODELSCOPE_ENDPOINT=https://modelscope.ai MODELSCOPE_PREFER_AI_SITE=true
export OMP_NUM_THREADS=14 MKL_NUM_THREADS=14 TOKENIZERS_PARALLELISM=false
CFG=${CFG:-/root/n/mvp/configs/detect-anneal-200mb.json}
ROOT=$(/root/n/venv/bin/python -c "import json;print(json.load(open('$CFG'))['anneal']['root'])")
RATES=(3e-7 5e-7 1e-6 1.7e-6 3e-6 5e-6 1e-5 2e-5)
ARMS=(P F L P-twin F-twin L-twin F-KL F-plane)
ROWS=(S80 base switches kl-plane overlay)
PY="/root/n/venv/bin/python -m chantless.cli anneal --config $CFG"
mkdir -p $ROOT/logs
on() { local g=$1; shift; CUDA_VISIBLE_DEVICES=$g TRITON_CACHE_DIR=/root/n/triton/$g $PY --gpu $g "$@"; }

$PY --stage setup > $ROOT/logs/setup.log 2>&1 || { echo SETUP_FAILED; exit 1; }
if [ ! -f $ROOT/common/lr.json ]; then
  for g in 0 1 2 3 4 5 6 7; do on $g --stage sweep --lr ${RATES[$g]} > $ROOT/logs/sweep-$g.log 2>&1 & done
  wait
  $PY --stage pick > $ROOT/logs/pick.log 2>&1 || { echo PICK_FAILED; exit 1; }
fi
cat $ROOT/logs/pick.log

printf '%s\n' "${ROWS[@]}" > $ROOT/rows.todo
worker() {
  local g=$1 arm=$2 n row
  for n in 1 2 3; do
    on $g --arm $arm >> $ROOT/logs/$arm.log 2>&1 && break
    echo "$arm died on GPU $g, try $n"
    sleep 30
  done
  while row=$(flock $ROOT/rows.lock sh -c "head -1 $ROOT/rows.todo; sed -i 1d $ROOT/rows.todo") && [ -n "$row" ]; do
    on $g --stage judge --arm $row >> $ROOT/logs/rows.log 2>&1
  done
}
for g in 0 1 2 3 4 5 6 7; do worker $g ${ARMS[$g]} & done
wait
$PY --stage report > $ROOT/logs/report.log 2>&1
echo ARMS_DONE
