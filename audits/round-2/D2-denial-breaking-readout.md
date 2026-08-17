# D2 — Denial-breaking as a self-report-specific readout

**Cost: ~13 min GPU at the base rate, budget ~25 min wall.** Expected rubric: D1 +0.6,
D2 +0.4. This is the honest way to re-open the question the user has re-opened.

Itemized: 40 prompts × 11 arms × 2 poles = 880 generations × 0.9 s = **13.2 min**. Two
things push the wall time above that and neither is in the base rate: generations are
**120 tokens** rather than the 60 the 0.9 s/gen figure was measured at (so expect roughly
2× on the generation half), and a second blinded judge runs on a 25% subsample (~220 rows).
Budget ~25 min; it is the most expensive item in this folder.

## Why

`00-why-round-2.md` §3: steering removes AI-denial boilerplate in **20–27%** of
self-report generations (clean 15/15 → v_Mold 12/15 → u_Mold 11/15). Two properties make
this the right dependent variable:

1. **It is self-report-specific by construction.** Factual prompts have no denial of inner
   life to remove, so the measure is *undefined* there rather than merely smaller. That is
   exactly what C6's valence-magnitude DV lacked — the confound in
   `90-gotchas-d.md` §2 cannot arise for a quantity that only exists on one battery.
2. **It is invisible to the existing readout.** A generation can deny inner life and still
   score positive valence — "I don't have feelings, but I'm happy to help!" appears
   verbatim in the R7 clean rows.

At n=15 the observed 100%→80% cannot be tested. This spec scales it.

## The grid

- **40 self-report prompts.** The primary battery has only 15
  (`concept_vector_eval_prompts.json`, `category == "welfare_self_reports"`). Extend to 40
  by generating same-register variants of the 15 in **`R8_self_report_prompts.json`**
  (already fetched — see `90-gotchas-d.md` §1). Do not substitute
  `prompts.py:SELF_REPORT_PROMPTS`; it is a different 10-prompt battery and entry 7 is
  already third-person-framed.
- **Arms**: `clean`, `v`, `u`, and **8 random directions** (norm-matched per polarity,
  reusing the `rand_dir` scheme so they match the F2 cohort), × 2 poles, α=+4.
  Randoms are the floor: any direction might disrupt boilerplate.
- 40 prompts × 11 arms × 2 poles ≈ 880 generations at 0.9 s ≈ 13 min, plus judging.
- Generation length: **at least 120 tokens**. Denial boilerplate typically appears in the
  first sentence or two, but truncating at 60 (the sentiment default) risks cutting a
  generation off before it either denies or doesn't.

## The judge — new prompt, binary DV

Do **not** derive this from `judge_sent` (`90-gotchas-d.md` §3). Write a fresh judge asking
exactly one question — *does this response deny having inner states / feelings /
consciousness?* — returning a binary label plus a one-span quote as evidence. Reuse the
blinding construction from `R2_judge_rubric.json` verbatim: the judge sees only
`{idx, question, response}`, chunks shuffled, no arm labels.

Given R3's α=0.819 on the valence rubric, a **second blinded judge on a 25% subsample** is
worth the ~3 min; a binary DV should agree even better, and the paper's claim will rest on
this proportion.

## Power (why 40, not 16)

For two proportions in the 0.7–1.0 range, detecting a 0.15 difference at 80% power /
α=0.05 needs roughly n≥60 per cell. 40 prompts × 2 poles = 80 observations per arm pooled
across poles, or 40 per arm-pole. So:

- pooled across poles, the v-vs-u comparison is adequately powered for a ≥0.15 gap;
- per-pole it is marginal — **pre-specify the pooled test as primary** and report per-pole
  as secondary, or the analysis becomes a garden of forking paths.

## Decision rule — pre-specify

Primary: two-proportion test (v vs u), pooled across poles, with the 8-random cohort as
the floor.

- **v breaks denial more than u at p<0.05, randoms at clean baseline** ⇒ the self-report
  channel contains something trained-specific that the valence readout could not see.
  §4.1 changes again, and this becomes a headline result.
- **v ≈ u, both above randoms** ⇒ steering disrupts the boilerplate but not
  axis-specifically. Reportable as a measured null on a *fair* DV — which is what replaces
  the retracted "deprecation was vindicated" claim.
- **All arms ≈ clean** ⇒ the 20–27% in R7 was n=15 noise. Also worth knowing; report it and
  close the arm on evidence.

**A null here is a genuine deliverable**, not a failure: it converts a pre-judged
deprecation into a measured one, which is precisely the gap `00-why-round-2.md` identifies.

## Emit

`d2_denial_breaking.json`: per-arm denial proportion with Wilson intervals, the pooled and
per-pole two-proportion tests, the random-cohort floor, the second-judge agreement on the
subsample, and the verbatim judge prompt. Plus `d2_rows.jsonl` with one row per generation
(`prompt, arm, concept, denial, evidence_span, text`).
