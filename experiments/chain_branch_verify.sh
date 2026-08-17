#!/bin/bash
# Branch-verification chain: run every changed post-lens experiment from the
# parivrudh/report-review checkout at /workspace/report-review-branch, into
# fresh results/ dirs, then verify numbers against the committed baseline.
# Continue-on-error; per-step RC + wall time recorded to chain_branch_rc.log.
set -u
ROOT=/workspace/report-review-branch
EXP=$ROOT/experiments
VENV=/workspace/venvs/jlens/bin/python
BASE=python3
RCLOG=$ROOT/chain_branch_rc.log
export PYTHONUNBUFFERED=1
cd $ROOT

step () {  # step <name> <python> <script> [args...]
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

# --- fast CPU signal on the refactored math ---
step test_routing_lib   $VENV $EXP/routing-core/test_routing_lib.py
step selftest_primary   $VENV $EXP/routing-core/selftest_routing.py --primary

# --- source of truth first (J5/J6 read it) ---
step run_mechanistic    $VENV $EXP/routing-core/run_mechanistic.py

# --- quick GPU cohort/battery jobs ---
step f2_jshare_cohort   $VENV $EXP/routing-core/f2_jshare_cohort.py
step f7_ksweep_ext      $VENV $EXP/jlens-atlas/f7_ksweep_ext.py
step f3_atlas_cohort    $VENV $EXP/jlens-atlas/f3_atlas_cohort.py
step r7_wholegen        $BASE $EXP/routing-core/r7_wholegen.py

# --- lens-pair + cross-model (need fresh mech_decompositions.json) ---
step j5_compare         $VENV $EXP/jlens-fit-2507/j5_compare.py
step j5_paired_baseline $VENV $EXP/jlens-fit-2507/j5_paired_baseline.py
step j6_transfer        $VENV $EXP/j6-crossmodel/j6_transfer.py
step j6_paired_baseline $VENV $EXP/j6-crossmodel/j6_paired_baseline.py
step lens_sanity        $VENV $EXP/jlens-replication/lens_sanity.py

# --- long GPU jobs last so quick confirmations land early ---
step atlas              $VENV $EXP/jlens-atlas/atlas.py
step recruitment_traj   $VENV $EXP/jlens-atlas/recruitment_traj.py

# --- CPU analyzers over fresh rows ---
step analyze_mech       $BASE $EXP/routing-core/analyze_mech.py
step analyze_primary    $BASE $EXP/routing-core/analyze_primary.py
step band_stats         $VENV $EXP/routing-core/band_stats.py
step convert_own_u      $BASE $EXP/welfare-axis/convert_own_u.py
step compare_u          $BASE $EXP/welfare-axis/compare_u.py
step r6_direction_table $BASE $EXP/welfare-axis/r6_direction_table.py

# --- the new experiment + the verdict ---
step step0_baseline     $BASE $EXP/jlens-atlas/step0_baseline.py
step verify_report      $BASE $EXP/verify_report.py

echo "=== CHAIN DONE $(date -u +%H:%M:%S) ==="
cat $RCLOG
