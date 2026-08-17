# F4 — Pre-specify one atlas estimand

**Cost: 0 GPU.** Expected rubric: D2 **+1.0**, D3 +0.4 (net **+1.4**). Recomputation from
existing `atlas_rows.jsonl` only.

## Why

The "trained reads out its own valence above naive" magnitude is not a fixed quantity.
**Two different estimands** are in play and neither is pre-specified:

- **Own-pole score ratio**: range **0.62x-6.53x**, inverting (naive > trained) in 3 of 12
  convention x aggregation cells.
- **Congruent pole difference** (own - opposite) — the form the report's 3.9x/9.4x
  derives from: range **-9.10x to +47.6x**, changing sign wherever the naive denominator
  crosses zero.

Neither quoted number is reproduced exactly by any single convention. Full tables in
`00-verdicts.md` 3.

## What to do

1. **Choose the estimand on stated grounds, not on the number it yields.**
   **Recommended: own-pole congruent score under the J-lens readout, averaged over the
   workspace band L16-31.** Rationale to put in Methods:
   - the *ratio of pole differences* is unstable by construction — its denominator is a
     difference that passes through zero for naive axes, which is what produces the
     -9.1x/+47.6x range. A ratio whose denominator can vanish is not a good estimand.
   - band-averaging avoids privileging a single treatment layer;
   - L16-31 is the band the instrument was validated on (`band_stats`, persistence above
     null), so it is chosen on instrument grounds rather than on the effect.

   Note this **revises** the earlier `fix_queue.md` recommendation, which suggested the
   pole-difference form before its instability was quantified.
2. **Name it in Methods** as the primary readout, with the aggregation and normalization
   stated explicitly.
3. **Add a robustness table** with all three conventions x two aggregations, for both
   estimands — **including the inverted cells**. Disclosing the inversion is a D2 gain;
   having a judge find it unprompted is a loss.
4. **Recompute every ratio in the report** from the chosen estimand. Current 4.2 numbers
   correspond to no single convention.
5. Report **absolute pole scores alongside every ratio**. A ratio between two small
   numbers is fragile; the absolute values let a reader see when that is happening.

## Verification

After recomputing, the primary-estimand numbers must be reproducible from
`atlas_rows.jsonl` by a stated formula. Emit `atlas_primary_estimand.json` with:
`{estimand_name, readout_variant, layers, per_cell_values, ratio, absolute_v, absolute_u,
n_random, perm_p}`. Do this before F3 — F3's cohort run should measure the estimand F4
selects, not all three.
