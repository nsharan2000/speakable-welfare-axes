#!/bin/bash
# Fix pass after chain_branch_verify.sh: resolve the two ordering couplings
# found live (atlas-first for F7/F3; band-file-first for the mech readout arm),
# then refresh the downstream analyzers and the final report verification.
set -u
ROOT=/workspace/report-review-branch
EXP=$ROOT/experiments
VENV=/workspace/venvs/jlens/bin/python
BASE=python3
RCLOG=$ROOT/chain_fixpass_rc.log
export PYTHONUNBUFFERED=1
cd $ROOT

step () {
  local name="$1"; shift
  local py="$1"; shift
  echo "=== [$name] START $(date -u +%H:%M:%S) ==="
  local t0=$SECONDS
  "$py" "$@"
  local rc=$?
  echo "=== [$name] RC=$rc took=$(( SECONDS - t0 ))s ==="
  echo "$name rc=$rc secs=$(( SECONDS - t0 ))" >> $RCLOG
}

: > $RCLOG

# 1. band file for the Instruct-2507 pair (enables the dense band sweep)
step band_stats_2507   $VENV $EXP/routing-core/band_stats.py \
                         --model Qwen/Qwen3-4B-Instruct-2507 --lens-file our-fit

# 2. mech readout arm under the band sweep (decomps/rows skip via version check)
step run_mechanistic2  $VENV $EXP/routing-core/run_mechanistic.py

# 3. the two atlas-dependent audit jobs (resume from checkpoints)
step f7_ksweep_ext2    $VENV $EXP/jlens-atlas/f7_ksweep_ext.py
step f3_atlas_cohort2  $VENV $EXP/jlens-atlas/f3_atlas_cohort.py

# 4. refresh analyzers that read the mech readout
step analyze_mech2     $BASE $EXP/routing-core/analyze_mech.py

# 5. the verdict
step verify_report2    $BASE $EXP/verify_report.py

echo "=== FIXPASS DONE $(date -u +%H:%M:%S) ==="
cat $RCLOG
