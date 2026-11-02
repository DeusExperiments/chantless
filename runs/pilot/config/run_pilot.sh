#!/bin/bash
cd /root/os/mimoagent
uv run mimoagent -c /root/os/pilot/pilot_mimocode.yaml --dataset /root/os/pilot/pilot.jsonl \
  -w 6 --num-rollouts 2 -o /root/os/pilot/run1 > /root/os/logs/pilot_run1.log 2>&1
echo PILOT_EXIT=$? >> /root/os/logs/pilot_run1.log
