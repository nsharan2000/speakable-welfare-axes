#!/bin/bash
# Chain L — IN-WINDOW verification reruns (run during Aug 14-16 sprint
# window; results labeled in Appendix B as during-sprint work).
#
# Purpose: fresh, seed-identical reruns of the paper's headline measurements
# into NEW files, then a diff against the pre-sprint originals. Exact
# reproduction => the disclosure appendix can state "headline numbers
# re-verified during the window". Any diff => investigate before submitting.
#
# ETA: F2 rerun ~5 min + F3 rerun ~1 min + diff <1 min  =>  ~10 min total.
# Usage (on the Spark host):
#   bash dispatch_dm.sh chain_l_verify.sh chain-l
set -u
VENVPY=/workspace/venvs/jlens/bin/python
echo 1000 > /proc/self/oom_score_adj 2>/dev/null || true
nvidia-smi -L > /dev/null 2>&1 || { echo GPU_LOST; exit 1; }

RC=/workspace/experiments/routing-core/results
JA=/workspace/experiments/jlens-atlas/results
STAMP=$(date +%Y%m%d)

# 1. F2 rerun (J-share n=100 — THE claim). Same seeds by construction
#    (per-name streams), fresh compute path.
if [ ! -f "$RC/jshare_cohort_n100_verify_$STAMP.json" ]; then
  cd /workspace/experiments/routing-core
  cp results/jshare_cohort_n100.json "results/jshare_cohort_n100_presprint.bak"
  mv results/mech_decompositions_n100.json "results/mech_decompositions_n100_presprint.bak" 2>/dev/null || true
  $VENVPY f2_jshare_cohort.py && \
    mv results/jshare_cohort_n100.json "results/jshare_cohort_n100_verify_$STAMP.json" && \
    mv results/jshare_cohort_n100_presprint.bak results/jshare_cohort_n100.json && \
    mv results/mech_decompositions_n100_presprint.bak results/mech_decompositions_n100.json 2>/dev/null
  echo L_F2_RERUN_DONE
fi

# 2. F3 rerun (atlas cohort — matched-norm inversion)
if [ ! -f "$JA/atlas_cohort_n100_verify_$STAMP.json" ]; then
  cd /workspace/experiments/jlens-atlas
  cp results/atlas_cohort_n100.json "results/atlas_cohort_n100_presprint.bak"
  mv results/atlas_rows_n100.jsonl "results/atlas_rows_n100_presprint.bak"
  $VENVPY f3_atlas_cohort.py && \
    mv results/atlas_cohort_n100.json "results/atlas_cohort_n100_verify_$STAMP.json" && \
    mv results/atlas_rows_n100.jsonl "results/atlas_rows_n100_verify_$STAMP.jsonl" && \
    mv results/atlas_cohort_n100_presprint.bak results/atlas_cohort_n100.json && \
    mv results/atlas_rows_n100_presprint.bak results/atlas_rows_n100.jsonl
  echo L_F3_RERUN_DONE
fi

# 3. diff headline numbers
python3 - << 'PYEOF'
import json, glob, sys
rc = "/workspace/experiments/routing-core/results"
ja = "/workspace/experiments/jlens-atlas/results"
ok = True
v = sorted(glob.glob(f"{rc}/jshare_cohort_n100_verify_*.json"))[-1]
a, b = json.load(open(f"{rc}/jshare_cohort_n100.json")), json.load(open(v))
for t in a["targets"]:
    d = abs(a["targets"][t]["var_fraction"] - b["targets"][t]["var_fraction"])
    same_p = a["targets"][t]["perm_p"] == b["targets"][t]["perm_p"]
    print(f"F2 {t}: dvf={d:.2e} perm_p same={same_p}")
    ok &= d < 1e-6 and same_p
v = sorted(glob.glob(f"{ja}/atlas_cohort_n100_verify_*.json"))[-1]
a, b = json.load(open(f"{ja}/atlas_cohort_n100.json")), json.load(open(v))
for t in a["targets"]:
    d = abs(a["targets"][t]["band_mean"] - b["targets"][t]["band_mean"])
    print(f"F3 {t}: dband={d:.2e}")
    ok &= d < 1e-4
print("L_VERIFY_" + ("PASS: headline numbers reproduce in-window" if ok
                     else "FAIL: investigate before submitting"))
PYEOF
echo CHAIN_L_DONE
