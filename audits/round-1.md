# Independent audit, round 1 (2026-08-12) — compiled

An independent adversarial re-analysis of every headline claim, performed
mid-project from the **raw per-cell rows** (`primary_rows.jsonl`,
`atlas_rows.jsonl`, `mech_decompositions.json`, `j4_rows_judged.jsonl`,
`ksweep.json`), not our summary JSONs. Every finding was adopted; two were
strengthened by our re-analysis, and one correction ran in the auditor's
favor beyond what they asked. This document compiles the audit's findings
and our response; the raw working documents (per-finding specs, fix queues,
gotcha lists) were internal process files and are summarized here.

## Finding 1 — The pre-registered primary fails its own controls

The frozen decision rule fired (`decision: H1_verbalizable`, d_z = 2.49,
p = 10⁻⁴). Recomputed from raw rows:

| | Gold | Mold |
|---|---|---|
| E(v) steered effect | +4.83 | +3.71 |
| E(u) naive control | −2.60 | −0.43 |
| E(random, n=20) | **+2.72** | −0.96 |
| v−u (headline contrast) | +7.43 | +4.14 |
| …of which from u falling *below* random | **+5.32 (71.6%)** | −0.53 |
| randoms with mean ≥ E(v) | 4/20 (p = 0.238) | 0/20 (p = 0.048) |

Gold's contrast is mostly the *control arm* being suppressed below chance —
a v−u contrast is only meaningful if u is neutral, and it is not. Mold
clears the randoms, but the C6 control showed the same injection shifts
valence *more* on unrelated factual prompts than on self-report prompts —
at face value a global output shift, not a self-report channel. (Round 2
later sharpened exactly what C6 does and does not establish — see
`round-2.md` §2.) **Adopted**: the paper reports the primary as a diagnosed
null (§4.8) with these numbers.

## Finding 2 — The strongest claim was under-reported for an arithmetic reason

At the original n = 8 random directions, an exact permutation test cannot
report below 1/(n+1) = 0.111 — so a z = +11 effect printed as
"non-significant". The fix is cohort size alone: n = 100 per polarity gives
a floor of 0.0099. **Adopted as F2** → the paper's spine (§4.2, Table 2):
0/100 randoms reach either trained pole; naive controls fall mid-null
(34/100, 53/100).

## Finding 3 — "4–9×" was not a fixed quantity: estimand pinned, headline overturned

The atlas pole-score ratio varied from 0.62× to 6.53× across 12 readout ×
aggregation × normalization conventions (inverting in 3 cells), and the
originally quoted derivative form was even less stable (−9.1× to +47.6×).
**Adopted as F4/F3**: one estimand was pre-specified
(`atlas_primary_estimand.json`), and the n = 100 norm-matched cohort then
**overturned our own headline** — at matched norms the Gold ratio inverts
(0.57×) and Mold falls 6.53× → 2.30×, tracking the trained axes' 1.6–2.4×
norm advantage. The paper's §4.1 reports the demotion, and the central
claim moved to the scale-invariant J-share. This is the audit working as
intended.

## Finding 4 — J4's causal claim needed a control arm, not more data

Power analysis showed n = 16/arm was adequate (pooled SD 0.61 vs observed
gaps 1.31–1.50); what was missing was a **random-direction J-component**
arm to rule out rescaling artifacts. **Adopted as F1**: 256 blind-judged
generations; identically-rescaled random J-components receive *more*
amplification yet stay at baseline (+0.039/+0.016 vs clean +0.25, inside
the pre-specified ±0.40 band) while real J-components move sentiment
±1.56 (§4.6, Fig. 3).

## Finding 5 — Reproducibility defects

- A cited file (`compare_u.json`) did not exist — regenerated, with one
  label correction: the two naive constructions correlate at 0.535 (gold,
  treatment layer) and 0.385 (mold, treatment layer); the previously logged
  0.144 was at the mold extraction's own selected layer (§4.5 states both).
- The k-sweep grid covered odd lens layers only. Correcting it surfaced a
  subtlety the audit itself had half-tripped on (below).

## Where the response went beyond the instructions

1. **An off-by-one in the audit's own treatment cells.** The lens at source
   layer *l* reads the block-input vector *l+1*, so the audit's "treatment"
   cells (lens 21/24) sat one position downstream of the true treatment
   position (vector 21/24 = lens 20/23). Both conventions are now reported
   in the robustness panel; the Methods section states the layer convention
   explicitly because two independent off-by-ones were caught this way.
2. **The k-sweep's real gap was Gold's layer, not Mold's** (same
   convention issue); `ksweep_ext.json` covers both readings with a
   12-random null at both treatment positions.
3. **F3's control was re-designed** (per-layer norm-matching with a
   separate mold-matched cohort) — this is the change that reversed the
   Gold headline.

## Judge-reliability and channel checks requested and returned

- **R3**: an independent second blinded judge (different model, verbatim
  rubric): Krippendorff α(interval) = 0.819, 94.4% within ±1; the
  J-component effect *strengthens* under the second judge (+2.00/−2.19).
- **R7**: whole-generation re-scoring confirmed C6 at the channel level —
  v_Mold −0.72 on self-report vs −2.28 on unrelated prompts — and captured
  the clean-baseline regime gap (self-report −1.327 vs +0.381; inner-life
  denial 15/15 vs 0/10) that round 2 diagnosed as the confound (§4.8,
  Fig. 5a).
- **R2/R4/R6**: verbatim judge rubrics, per-prompt clean baselines, and the
  direction table (v·u cosines 0.56/0.675 — "the naive control is
  substantially aligned with the trained axis, not orthogonal").

## A correction this audit later owed the project

Round 1's write-up asserted that the evidence "independently vindicated" a
mid-project decision to set the steer-and-ask paradigm aside. Round 2's
provenance check showed the directive **preceded** the supporting analysis
— the same reasoning error round 1 criticized in the project's own primary
analysis. The sentence was removed from the paper; the honest statement is
that the steer-and-ask contrast is *confounded and underpowered, not
refuted* (interaction p = 0.505). See `round-2.md` §1–2.

## Operational note (reproduction, not results)

Two chain runs died to a Docker/NVIDIA cgroup interaction that revokes GPU
access in running containers. Mitigation used throughout: a GPU canary
before every step plus idempotent, checkpointed chain steps under a
supervisor. Relevant only if you replicate on similar infrastructure.
