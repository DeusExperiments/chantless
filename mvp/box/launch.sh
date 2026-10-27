#!/bin/bash
# Local driver for the Lium box. Usage:
#   box/launch.sh sync <pod>      copy MVP code (no private references, no weights) to the pod
#   box/launch.sh setup <pod>     run setup_box.sh on the pod (pinned llama.cpp + patch, traj-kl, venv, model)
#   box/launch.sh run <pod> CFG   start the pipeline detached (nohup) with configs/CFG.json
#   box/launch.sh status <pod>    tail the run log
#   box/launch.sh fetch <pod> RUN copy receipts back (excludes GGUFs, stores and caches)
set -euo pipefail
cmd=$1; pod=$2
HERE=$(cd "$(dirname "$0")/.." && pwd)
R=/root/chantless
case $cmd in
  sync)  lium exec "$pod" "mkdir -p $R/mvp"; lium rsync "$pod" "$HERE/" "$R/mvp/" ;;
  setup) lium exec "$pod" "bash $R/mvp/box/setup_box.sh $R > $R/setup.log 2>&1; tail -3 $R/setup.log" ;;
  run)   lium exec "$pod" "cd $R/mvp && (nohup $R/.venv/bin/python -m chantless.cli run --config configs/$3.json > $R/runs/$3.out 2>&1 &); echo started" ;;
  stop)  lium exec "$pod" "for p in \$(ps -eo pid,cmd | awk '/[c]hantless.cli run|[l]lama-server|[t]raj-kl|[l]lama-quantize/{print \$1}'); do kill -TERM \$p 2>/dev/null; done; sleep 3; for p in \$(ps -eo pid,cmd | awk '/[l]lama-server|[t]raj-kl|[l]lama-quantize/{print \$1}'); do kill -9 \$p; done; nvidia-smi --query-gpu=memory.used --format=csv,noheader | tr '\n' ' '" ;;
  status) lium exec "$pod" "tail -5 $R/runs/*/log.md; tail -3 $R/runs/*.out" ;;
  fetch) mkdir -p "$HERE/../.local/mvp/box-runs/$3"
         lium exec "$pod" "cd $R/runs/$3 && tar czf /tmp/$3-receipts.tgz --exclude='*.gguf' --exclude='store*' --exclude='probe' --exclude='build' --exclude='repair/hf' --exclude='*.bin' ."
         lium scp -d "$pod" "/tmp/$3-receipts.tgz" "$HERE/../.local/mvp/box-runs/$3/" ;;
esac
