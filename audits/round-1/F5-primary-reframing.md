# F5 — Reframe the primary arm as a pre-registered null

**Cost: 0 GPU.** Expected rubric: D1 -0.3, D2 **+1.2**, D3 +0.5 (net **+1.4**, highest in
the queue). No new measurement — this is analysis and writing.

## Why

`primary_analysis.json` records `decision: H1_verbalizable`, but three pre-registered
controls undercut the interpretation (numbers recomputed from `primary_rows.jsonl`; see
`00-verdicts.md` 1):

- **C1**: 71.6% of the Gold v-u gap is u falling *below* the random null; v_Gold is
  inside the random band (4/20 randoms exceed it, perm p=0.238).
- **C6**: v_Mold shifts valence *more* on unrelated factual prompts (+4.54) than on the
  welfare self-report prompts (+3.71) -> global shift, not a self-report channel.
- **C7**: congruent monotonicity holds for v_Mold only (rho=+1.0 vs -0.4/-1.0/-0.6).

A paired v-u contrast is only interpretable if u is a neutral reference. Here u is
actively suppressed relative to chance, so the contrast measures something other than
what the hypothesis names.

## What to do

1. **Abstract**: remove "self-report routing" from the list of pillars. It currently
   appears as one of three supports for the headline.
2. **Results 4.1**: report as — pre-registered contrast met its decision rule on the
   paired statistic, **but** the pre-specified controls C1/C6/C7 show it does not support
   a self-report-channel reading. State all three explicitly, with the numbers.
3. **Keep the decomposition table** (v-random vs random-u). It explains *why* a paired
   contrast misleads when the control direction is non-neutral — a transferable
   methodological point for any steering paper that uses a "naive control".
4. Note that this **motivated the mid-project deprecation** of steer-and-confess, and
   that the deprecation was made **independently and earlier**, on the chain-g rescaling
   fragility — not retrofitted to this audit.
5. **Do not edit `pre-registration.md`.** Its value is that it was frozen before the
   data. Record the override as a deviation in the report.
6. `primary_analysis.json` keeps `decision: H1_verbalizable`. Either add an
   `interpretation` field recording the control-based override, or emit a sibling
   `primary_analysis_controls.json`. Do not silently change the decision string — the
   frozen decision rule *was* met on the statistic it named; what changed is the
   interpretation the controls permit.

## Why this gains rather than loses points

The rubric's Dimension 2 asks verbatim whether the difference is significant and robust,
and judges score only the PDF. A pre-registered null whose own pre-specified controls
caught the artifact is direct evidence of method quality. The alternative — a judge
noticing that C6 was reported at +4.54 while the claim says "self-report specific" — is
a Dimension-2 loss with no recovery.
