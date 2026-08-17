#!/bin/bash
# Chain B: waits for the fitted Instruct-2507 lens, then band stats + mechanistic battery.
set -u
VENVPY=/workspace/venvs/jlens/bin/python
LENS=/workspace/experiments/jlens-fit-2507/results/Qwen3-4B-Instruct-2507_jacobian_lens.pt
echo "[chain-b] waiting for lens fit to complete"
for i in $(seq 1 720); do [ -f "$LENS" ] && break; sleep 60; done
[ -f "$LENS" ] || { echo "CHAIN_B_TIMEOUT"; exit 1; }
echo "[chain-b] lens found; waiting 60s for writer to finish"; sleep 60

if [ ! -f /workspace/experiments/routing-core/results/band_stats_Qwen3-4B-Instruct-2507.json ]; then
  cd /workspace/experiments/routing-core && $VENVPY band_stats.py \
    --model Qwen/Qwen3-4B-Instruct-2507 --lens-file "$LENS"
fi
cd /workspace/experiments/routing-core && $VENVPY run_mechanistic.py
echo CHAIN_B_DONE
