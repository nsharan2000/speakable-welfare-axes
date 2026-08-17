#!/bin/bash
# Chain N — audit round-2 D-series, strictly serial, one GPU job at a time.
# Order per PENDING-WORK.md §B: D1 (~2 min) -> D3 clean gate (~2 min) ->
# [gate pass] D3 steered (~7 min) -> D2 generations (~25 min).
# On D3_GATE_REVISE the chain still runs D2 (independent), then exits with
# marker D3_AWAITING_REVISION so the analogue battery can be revised and
# d3 redispatched alone.
# Usage (Spark host): bash dispatch_dm.sh chain_n_dseries.sh chain-n
set -u
VENVPY=/workspace/venvs/jlens/bin/python
RC=/workspace/experiments/routing-core
echo 1000 > /proc/self/oom_score_adj 2>/dev/null || true

gpu() { nvidia-smi -L > /dev/null 2>&1 || { echo GPU_LOST; exit 1; }; }
memgate() {
  FREE=$(free -g | awk '/^Mem:/{print $7}')
  [ "$FREE" -ge 20 ] || { echo "MEM_LOW free=${FREE}G"; exit 1; }
}

cd "$RC"

# ---- D1 ----
gpu; memgate
if [ ! -f "$RC/results/d1_orthogonal_control.json" ]; then
  $VENVPY d1_orthogonal_control.py || { echo D1_FAILED; exit 1; }
fi
echo N_D1_OK

# ---- D3 stage clean (matching gate) ----
gpu; memgate
BATT_V=$($VENVPY -c "import json;print(json.load(open('d_batteries.json'))['version'])")
CHK_V=$($VENVPY -c "import json;print(json.load(open('results/d3_matching_check.json'))['battery_version'])" 2>/dev/null || echo none)
if [ ! -f "$RC/results/d3_matching_check.json" ] || [ "$CHK_V" != "$BATT_V" ]; then
  $VENVPY d3_matched_c6.py --stage clean || { echo D3_CLEAN_FAILED; exit 1; }
fi
GATE=$($VENVPY - << 'PYEOF'
import json
print("pass" if json.load(open("results/d3_matching_check.json"))["gate"]["pass"] else "revise")
PYEOF
) || GATE=parse_failed
case "$GATE" in
  pass|revise) echo "N_D3_GATE_$GATE" ;;
  *) echo N_D3_GATE_PARSE_FAILED; exit 1 ;;
esac

# ---- D3 stage steered (only if gate passed) ----
if [ "$GATE" = "pass" ]; then
  gpu; memgate
  if [ ! -f "$RC/results/d3_matched_c6.json" ]; then
    $VENVPY d3_matched_c6.py --stage steered || { echo D3_STEERED_FAILED; exit 1; }
  fi
  echo N_D3_OK
fi

# ---- D2 generations ----
gpu; memgate
if [ ! -f "$RC/results/d2_gen_meta.json" ]; then
  $VENVPY d2_denial_breaking.py || { echo D2_FAILED; exit 1; }
fi
echo N_D2_OK

[ "$GATE" = "pass" ] || echo D3_AWAITING_REVISION
echo CHAIN_N_DONE
