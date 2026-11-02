#!/bin/bash
cd /root/os
for t in $(python3 -c "import json;print(' '.join(x['instance_id'] for x in json.load(open('pilot/tasks.json'))))"); do
  s=$(date +%s); docker pull -q xiaomimimo/mimo-v2.6-rl-oss:$t >/dev/null 2>&1; echo "$t rc=$? secs=$(( $(date +%s)-s ))"
done
echo PULL_DONE
