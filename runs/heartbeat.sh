#!/bin/bash
# Read-only heartbeat for the Lium pilot. Plain ssh with hard timeouts (lium exec can hang).
# Exits after ~9.5 min, or as soon as the pilot finishes.
SSH="ssh -i $HOME/.ssh/id_ed25519 -o IdentitiesOnly=yes -p 10062 -o BatchMode=yes -o ConnectTimeout=10 -o ServerAliveInterval=5 -o ServerAliveCountMax=2 root@86.125.172.23"
run() { perl -e 'alarm shift; exec @ARGV' 40 $SSH "$1" 2>&1; }
end=$(( $(date +%s) + 570 ))
while [ $(date +%s) -lt $end ]; do
  [ "$(run 'grep -c PILOT_EXIT /root/os/logs/pilot_run1.log')" = "1" ] && break
  sleep 60
done
date '+%H:%M:%S %Z'
run 'echo health=$(curl -s -m 5 localhost:8080/health) containers=$(docker ps -q | wc -l) results=$(cat /root/os/pilot/run1/results.jsonl 2>/dev/null | wc -l); grep PILOT_EXIT /root/os/logs/pilot_run1.log; echo gpu=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader) disk_free=$(df -h / | awk "NR==2{print \$4}"); docker ps --format "running: {{.Image}} {{.RunningFor}}" | sed "s#.*:format#format#"'
