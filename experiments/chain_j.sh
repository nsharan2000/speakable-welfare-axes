#!/bin/bash
# Chain J — J6 cross-model transfer, gated on chain-i completing.
# ETA ~25-40 min. J5 penult-fit resume is dispatched separately afterwards
# (its exact invocation lives in chain_h.sh on the box).
set -u
VENVPY=/workspace/venvs/jlens/bin/python
echo 1000 > /proc/self/oom_score_adj 2>/dev/null || true

if ! nvidia-smi -L > /dev/null 2>&1 || \
   ! python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
  echo GPU_LOST; exit 1
fi

echo "[chain-j] waiting for CHAIN_I_DONE"
for i in $(seq 1 600); do
  grep -q CHAIN_I_DONE /workspace/logs/chain-i.log 2>/dev/null && break
  sleep 60
done
grep -q CHAIN_I_DONE /workspace/logs/chain-i.log || { echo CHAIN_J_TIMEOUT; exit 1; }

for i in $(seq 1 240); do
  avail=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)
  [ "$avail" -ge 20 ] && break
  sleep 30
done

if [ ! -f /workspace/experiments/j6-crossmodel/results/j6_summary.json ]; then
  cd /workspace/experiments/j6-crossmodel && \
    $VENVPY j6_transfer.py || echo J6_FAILED
fi
echo CHAIN_J_DONE
