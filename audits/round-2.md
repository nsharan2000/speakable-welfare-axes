# Independent audit, round 2 (2026-08-15) — compiled

Round 2 opened with a correction the audit owed the project, re-verified the
inputs behind the §4.8 analyses, and specified five new experiments (D1–D5)
that were all executed inside the sprint window. This document compiles the
round's reasoning and outcomes; the raw working documents (per-experiment
specs, verification requests, gotcha lists) were internal process files and
are summarized here.

## 1. The provenance correction

A mid-project directive had set the steer-and-ask ("steer-and-confess")
paradigm aside. Round 1's write-up — and the paper draft at the time —
claimed the evidence "independently vindicated" that decision. The
provenance check showed the directive **came first** and the supporting
analysis was assembled afterwards: the same error shape round 1 criticized
in the project's primary analysis (a recorded decision its controls did not
support). Required fixes, all applied: delete the "independently led us to
deprecate" sentence; restate the contrast honestly; keep the null framing
but rest it on the power-and-confound argument.

## 2. What C6 actually established

Round 1 read C6 ("the injection shifts valence more on unrelated prompts
than on self-report prompts") as *global shift, not a self-report channel*.
Re-analysis split that into three distinct statements:

- **The raw effect is not self-report-specific** — real and reported.
- **The two prompt batteries are different behavioral regimes**: clean
  valence −1.327 (self-report) vs +0.381 (unrelated); inner-life-denial
  boilerplate in 15/15 vs 0/10 generations. C6 compared a suppressed regime
  against a free one (§4.8, Fig. 5a).
- **The pre-registered claim is about the trained/naive contrast**, and on
  that contrast the interaction is p = 0.505 with ratios nominally *larger*
  on self-report (1.98× vs 1.39×). Honest verdict: **underpowered, not
  refuted**.

A regex-based observation that steering removes denial boilerplate in
20–27% of self-report generations motivated D2 — with the warning that the
regex readout itself needed validation (it failed it; see R10 below).

## 3. Input re-verification (R8–R12)

- **R8/R9**: the true 15-prompt self-report battery was recovered and
  confirmed identical from two independent sources; the pairing behind the
  p = 0.505 interaction was validated against it.
- **R10**: blind judge re-labeling of the regex denial estimate (two
  judges, agreement 1.00 over 75 rows) showed the **regex undercounted
  denial in steered text** — the 20–27% figure was mostly artifact. The
  finding survives only as motivation; D2 used blind binary judging
  throughout.
- **R11/R12**: assert-level checks of the D1 construction, and a re-reading
  of the J6 transfer claim that weakened it to axis-and-lens geometry
  (§4.7's caveat).

## 4. The D-series (all pre-specified, all executed in-window)

**D1 — orthogonalized control** (§4.3). u⊥ = u − (u·v̂)v̂, renormalized to
‖v‖, with verified construction (|cos(u⊥,v)| < 10⁻⁵) and a pre-specified
verdict mapping. Outcome: u⊥ is behaviorally inert at both poles (blind
sentiment 0.00 vs ±2–3 for real axes) — the naive control's behavioral
potency lives entirely in its v-shared component. Its J-share splits by
pole: gold falls to chance (0.0371, p = 0.653 → "shared subspace"); mold
**rises above u itself** (0.0552, p = 0.0297 → the pre-specified
"anomalous" branch): naive distress constructions carry speakable valence
the trained axis does not subsume.

**D2 — denial-breaking readout** (§4.8, Table 3). A binary dependent
variable that is self-report-specific by construction: does the generation
deny having inner states? 40 prompts × 21 arms = 840 generations, blind
binary judging, independent second judge (agreement 0.967). Outcome —
positive, gold-carried: clean 95% denial; v_Gold **37.5%** vs u_Gold 65%
(p = 0.0139); 8-random cohort 90.6%; both Mold arms ceiling-locked at
97.5%; pre-specified pooled primary p = 0.0464. Robust to splitting the 15
verbatim vs 25 authored prompts.

**D3 — prompt-matched batteries** (§4.8). Third-person analogues matched on
topic and affect vocabulary, with a pre-specified matching gate
(|Δ clean valence| < 0.5, denial < 20%, length ±30%, max two revision
rounds). Outcome — negative finding by the spec's own stopping rule:
denial vanished and length matched, but the valence gap went 1.71 → 1.13 →
0.63, never under 0.5. The self-report register carries an irreducible
negative baseline in this model; the steered comparison was correctly never
run.

**D4 — report corrections.** The provenance sentence deleted; §4.8
restated around the interaction p = 0.505, the per-battery ratios, and the
regime confound as a methods finding; the control-alignment observations
(cos 0.56/0.675, construction-dependence) unified into §4.5.

**D5 — k-sweep completion.** Folded into §4.2's robustness statement
(trained−naive gap positive at every k ∈ {4..50} at both treatment
positions, with the honest note that the Gold gap is layer-local).

## 5. What round 2 did not touch

Everything resting on the scale-invariant J-share: the n = 100 spine (F2),
the controlled channel-split (F1), the trajectory asymmetry, and the
transfer ordering. The corrections narrowed §4.8 and §4.1; the spine was
additionally re-verified bit-exactly inside the sprint window (Appendix A).
