#!/bin/bash
# Chain D (new J-program): after chain-b finishes (GPU serialization), run the
# welfare-axis J-lens atlas (J1+J3, ETA ~30-60 min) then the recruitment
# trajectory (J2, ETA ~30-60 min).
set -u
VENVPY=/workspace/venvs/jlens/bin/python
echo "[chain-d] waiting for CHAIN_B_DONE"
for i in $(seq 1 900); do
  grep -q CHAIN_B_DONE /workspace/logs/chain-b.log 2>/dev/null && break
  sleep 60
done
grep -q CHAIN_B_DONE /workspace/logs/chain-b.log || { echo CHAIN_D_TIMEOUT; exit 1; }

if [ ! -f /workspace/experiments/jlens-atlas/results/atlas_meta.json ]; then
  cd /workspace/experiments/jlens-atlas && $VENVPY atlas.py || echo ATLAS_FAILED
fi
if [ ! -f /workspace/experiments/jlens-atlas/results/traj_results.json ]; then
  cd /workspace/experiments/jlens-atlas && $VENVPY recruitment_traj.py || echo TRAJ_FAILED
fi
if [ ! -f /workspace/experiments/j4-behavioral/results/j4_meta.json ]; then
  cd /workspace/experiments/j4-behavioral && python3 j4_dissociation.py || echo J4_FAILED
fi
echo CHAIN_D_DONE
