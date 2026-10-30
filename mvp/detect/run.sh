#!/bin/bash
# Local driver for the detector run on one 8×H100 Lium pod. In order, after `lium up`:
#   detect/run.sh cap <pod>          the pod removes itself after 4 h (about $51 at $12.80/h)
#   detect/run.sh sync <pod>         copy detect/ to the pod (code only)
#   detect/run.sh start <pod> STEP   run pod.sh STEP detached: setup, prep, preflight, train, score
#   detect/run.sh status <pod>       last lines of every step log; a finished step ends with STEP_DONE
#   detect/run.sh fetch <pod>        pack results on the pod, copy them to .local/detect, check the checksum
# Fetch after train and again after score: the cap removes the pod and everything on it.
# If the preflight's 3 steps run out of memory at batch 8:  BATCH=4 detect/run.sh start <pod> train
# Then on the laptop:  python detect/measure.py ../.local/detect/runs/scores
set -euo pipefail
cmd=$1; pod=$2
HERE=$(cd "$(dirname "$0")" && pwd)
R=/root/detect
LOCAL=$HERE/../../.local/detect
case $cmd in
  cap)    lium rm "$pod" --in 4h ;;
  sync)   lium exec "$pod" "mkdir -p $R/code $R/logs"; lium rsync "$pod" "$HERE/" "$R/code/" ;;
  start)  lium exec "$pod" "cd $R/code && (BATCH=${BATCH:-8} nohup bash pod.sh $3 > $R/logs/$3.log 2>&1 &); echo started $3" ;;
  status) lium exec "$pod" "for f in $R/logs/*.log; do echo == \$f; tail -3 \$f; done" ;;
  fetch)  mkdir -p "$LOCAL"
          lium exec "$pod" "bash $R/code/pod.sh pack"
          lium scp -d "$pod" "$R/detect-results.tgz" "$LOCAL/"
          lium scp -d "$pod" "$R/detect-results.tgz.sha256" "$LOCAL/"
          cd "$LOCAL" && sed "s|$R/||" detect-results.tgz.sha256 | shasum -a 256 -c - && tar xzf detect-results.tgz ;;
esac
