#!/bin/bash
# One-time prep for the branch-verification chain (runs INSIDE dm-exp).
# Fresh results/ dirs (committed results kept as results_committed/ baseline),
# inputs symlinked back in, lens .pt linked from the old tree, paths sanity-checked.
set -eu
EXP=/workspace/report-review-branch/experiments
OLD=/workspace/experiments
cd $EXP

for d in routing-core jlens-atlas jlens-fit-2507 j6-crossmodel jlens-replication welfare-axis; do
  if [ -d $d/results ] && [ ! -d $d/results_committed ]; then
    mv $d/results $d/results_committed
  fi
  mkdir -p $d/results
done

# analyze_primary input (run_primary generation intentionally not rerun: paths-only diff)
for f in primary_rows.jsonl primary_meta.json; do
  [ -e routing-core/results_committed/$f ] && ln -sf ../results_committed/$f routing-core/results/$f
done

# fitted lenses (875MB each, gitignored) — link from the old tree, per no-refit rule
for f in $OLD/jlens-fit-2507/results/*jacobian_lens*.pt; do
  ln -sf $f jlens-fit-2507/results/$(basename $f)
done
# fit metadata the J5 scripts may read (everything committed except J5's own outputs)
for f in jlens-fit-2507/results_committed/*; do
  b=$(basename $f)
  case $b in j5_*) ;; *) ln -sf ../results_committed/$b jlens-fit-2507/results/$b ;; esac
done

echo "--- own_u external check ---"
ls /workspace/welfare-vectors/own_u/*/mean_diff.pt 2>/dev/null | head -4 || echo "OWN_U MISSING (convert_own_u will fail)"

echo "--- dm_paths resolution from branch checkout (jlens venv) ---"
/workspace/venvs/jlens/bin/python $EXP/common/dm_paths.py

chmod +x $EXP/chain_branch_verify.sh
echo PREP_OK
