#!/bin/bash
# Chain J (direct) — J6 cross-model transfer, no marker-waiting: the
# supervisor sequences phases, so this just runs J6 (idempotent/resumable).
set -u
VENVPY=/workspace/venvs/jlens/bin/python
echo 1000 > /proc/self/oom_score_adj 2>/dev/null || true
if ! nvidia-smi -L > /dev/null 2>&1; then echo GPU_LOST; exit 1; fi

for i in $(seq 1 240); do
  avail=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)
  [ "$avail" -ge 20 ] && break
  sleep 30
done

cd /workspace/experiments/j6-crossmodel && $VENVPY j6_transfer.py || echo J6_FAILED
echo CHAIN_J_DONE
