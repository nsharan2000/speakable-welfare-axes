#!/bin/bash
# Chain C: after own-u extraction completes, convert and rerun the steering
# validation with OUR paper-style u as the control (adjudicates the flat-control claim).
set -u
echo "[chain-c] waiting for u-extraction"
for i in $(seq 1 360); do
  grep -q U_EXTRACTION_DONE /workspace/logs/u-extract.log 2>/dev/null && break
  sleep 60
done
grep -q U_EXTRACTION_DONE /workspace/logs/u-extract.log || { echo CHAIN_C_TIMEOUT; exit 1; }
python3 /workspace/experiments/welfare-axis/convert_own_u.py || { echo CHAIN_C_CONVERT_FAILED; exit 1; }
cd /workspace/experiments/welfare-axis
# rerun steering with own-naive as the u file: swap via env override script
python3 - <<'PYEOF'
import json, re
src = open('validate_steering.py').read()
src = src.replace('vectors_naive_faithful_pc5000.pt', 'vectors_own_naive.pt')
src = src.replace('results/steering_validation.json', 'results/steering_validation_ownu.json')
src = src.replace('results/generations.jsonl', 'results/generations_ownu.jsonl')
open('validate_steering_ownu.py', 'w').write(src)
PYEOF
python3 validate_steering_ownu.py && echo CHAIN_C_DONE || echo CHAIN_C_FAILED
