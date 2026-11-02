#!/bin/bash
# Pull each candidate image in order; mark LEAK if any commit is reachable that is not an ancestor of HEAD
# (same property mimoagent's _assert_history_truncated enforces). Stops after 50 CLEAN.
cd /root/os/measure; : > screen.tsv; clean=0
python3 -c "import json;[print(json.loads(l)['instance_id'],json.loads(l)['docker_image']) for l in open('candidates.jsonl')]" |
while read iid img; do
  docker pull -q "$img" >/dev/null 2>&1 || { echo -e "$iid\tPULL_FAIL" >> screen.tsv; continue; }
  n=$(docker run --rm --entrypoint bash "$img" -c 'cd /testbed && git rev-list --all --not HEAD 2>/dev/null | wc -l' 2>/dev/null)
  if [ "$n" = "0" ]; then echo -e "$iid\tCLEAN" >> screen.tsv; clean=$((clean+1)); else echo -e "$iid\tLEAK\t$n" >> screen.tsv; fi
  [ $clean -ge 50 ] && break
done
echo SCREEN_DONE >> screen.tsv
