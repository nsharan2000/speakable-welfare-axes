# D4 — Elevate the control-alignment finding, and fix the report text

**Cost: 0 GPU.** Expected rubric: D1 +0.4, D2 **+0.8**, D3 +0.4. Do this first — part of it
corrects text that is currently wrong.

## Part A — mandatory corrections (see `00-why-round-2.md` §4)

1. Delete the "independently led us to deprecate" claim at `report-draft.md:168`.
2. Restate §4.1: C6 rejects magnitude-specificity, not the contrast; interaction p=0.505;
   per-battery ratios 1.98× (self) and 1.39× (unrel); contrast underpowered (p=0.178/0.086).
3. Add the regime confound: clean valence −1.327 vs +0.381, denial language 15/15 vs 0/10.
4. Update `../CUT-x1-x2.md` so the X.1/X.2 cut rests on power-and-confound, not on
   "the deprecation was vindicated".

## Part B — the finding worth stating

Three results now point the same way, and together they are a contribution rather than
three apologies.

**1. The "naive control" is not a control** (`R6_direction_table.json`, treatment layers):

| pair | cos |
|---|---|
| v_Gold vs u_faithful_Gold | **0.560** |
| v_Mold vs u_faithful_Mold | **0.675** |
| v_Gold vs own_u_Gold | 0.433 |
| v_Mold vs own_u_Mold | 0.596 |
| u_faithful vs own_u (gold / mold) | 0.535 / 0.385 |
| any axis vs random | ~0.02 |

It shares 56–68% of its direction with the trained axis — a partially-overlapping variant,
not an independent reference. And the two *published-recipe* naive constructions correlate
only 0.385–0.535 with **each other**, so "the" naive control is not even well-defined.

**2. The comparison's outcome depends on normalization** (F3). Norm-matching the control —
which every other experiment in the repo does — collapses Mold 6.53× → 2.30× and inverts
Gold to **0.57×**. With cos 0.56 and a 1.62× norm advantage, which axis "reads out higher"
is decided by the matching convention, not by the axes.

**3. Yet J-share separates them anyway** (F2, n=100): u_Gold and u_Mold sit at chance
(53/100 and 34/100 randoms exceed them) **despite** being 56–68% aligned with axes that
clear p=0.0099.

## The claim to write

> "Trained vs naive" is not a well-posed contrast at the level of direction identity: the
> two axes overlap by cos 0.56–0.68, the two published naive recipes agree with each other
> only at 0.39–0.54, and the sign of the trained/naive difference depends on the
> normalization convention. A scale-invariant subspace measure separates them regardless —
> which is evidence about the measure, not just about the axes.

**Why point 3 is the strongest single observation in the current data.** Whatever J-share
measures, it is sensitive to the 32–44% where the axes *differ* rather than to the majority
they share. That is a much sharper statement about the instrument than "trained axes score
higher", and it is not yet anywhere in the draft.

## Framing guidance

Put this in Results as a positive finding with its own subsection, not in Limitations. The
rubric's Dimension 2 asks whether differences are significant *and robust*; a paper that
shows *why* a naive-control contrast is fragile, and then reports a measure that survives
it, is answering that question directly. A reader who finds the 0.560 cosine unprompted
draws a much worse conclusion.

## Emit

No compute. Optionally `d4_alignment_table.json` collecting the R6 cosines, the F3
native-vs-matched ratios, and the F2 n=100 J-share rows in one place, so the subsection's
numbers have a single source.
