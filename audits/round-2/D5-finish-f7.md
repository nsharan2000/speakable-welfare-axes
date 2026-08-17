# D5 — Finish F7 and report the k-sweep null honestly

**Cost: in flight** (was ~12 min). Expected rubric: D2 +0.4.

## Status

`audit-response.md` lists F7 as running, with the spec corrected first: the compute agent
found that `ksweep.json`'s keys are **lens** layers under the `V[l+1]` convention, so key
`L23` already *is* vector layer 24 = Mold's treatment position. The real gap was **Gold's**
treatment position (vector 21 = lens 20), which the odd-layer grid skips.

That correction is accepted — it is the same off-by-one the agent caught in the audit's F4
cells, and `ksweep_ext.json` adding lens {20, 22, 24} covers both readings.

## What to report when it lands

1. **Per-k permutation p at both treatment positions**, against the 12-random null
   (p-floor 1/13 = 0.077). Note the floor explicitly — it cannot reach 0.05, so frame
   results as "gap present at every k, p ≤ 0.077" rather than implying a stronger threshold.
2. **State the layer convention in the caption.** Two off-by-one errors have now been found
   in this project's layer indexing (the audit's F4 cells; F7's original spec). Any table
   of per-layer results should say "lens layer *l* carries vector layer *l+1*" so a reader
   can check.
3. **If the gap fails at some k, report the bound.** "Robust for k ∈ {8…50}, fails at k=4"
   is a stronger scientific statement than an untested "robust to k", and Dimension 2
   rewards the bound.

## Do not

Raise the k-sweep null beyond 12 randoms. The claim is k-*stability* of a gap already
established at p=0.0099 by F2; spending 30+ min to move this secondary null from 0.077 to
0.05 buys nothing the spine does not already carry.
