# REPORT-V2-SPEC — build the submission report after the pending work lands

**Audience: the agent with compute access.** This is the last item in the queue. Do not
start it until the gating experiments below have returned; it is a *writing* task whose
inputs are result files, and starting early means writing sentences that the results will
contradict.

**v1 is archived.** `report/report-draft-v1.md` and `.pdf` are the Aug-12 draft, frozen,
with a banner listing its four known defects. Do not edit or submit v1. Write
`report/report-draft.md` fresh (v2) and build `report/report-draft.pdf` from it with the
existing `report/build_pdf.sh`.

---

## 0. Gate — what must land before writing

| gate | file that proves it | status at spec time |
|---|---|---|
| D4 corrections applied | `report/` text edits + `d4_alignment_table.json` | pending (0 GPU) |
| **6.5 in-window verification** | `*_verify_$STAMP.json` + `L_VERIFY_PASS` | **pending (~7 min) — see below** |
| D1 orthogonalized control | `d1_orthogonal_control.json` | pending (~1.4 min) |
| D3 prompt-matched C6 | `d3_matched_c6.json` | pending (~5 min, needs R8) |
| D2 denial-breaking readout | `d2_denial_breaking.json` | pending (~13 min, needs R8) |
| D5 / F7 k-sweep | `ksweep_ext.json` | **landed** |

Full inventory of what is open, including the Phase-6 items and the 🔮 list:
`PENDING-WORK.md`. Two things from it bear directly on this document:

- **Appendix B depends on chain L.** `6.5` (`bash dispatch_dm.sh chain_l_verify.sh chain-l`)
  reruns F2 and F3 seed-identically inside the sprint window and diffs them against the
  pre-sprint originals. Undisclosed prior work "can lead to disqualification," and most of
  this project predates Aug 14 — so Appendix B should state that the headline numbers were
  re-verified in-window, and that sentence needs `L_VERIFY_PASS` to be true. Run it before
  writing the appendix; if it returns FAIL, investigate before submitting anything.
- **Code-and-Data promises artifacts that may not exist.** v1 promises "our fitted
  Instruct-2507 lenses (final + penultimate targets, fp32)". Either run `6.6`
  (`hf-release/upload.sh`, needs the user's HF login) or **delete that clause**. The
  official pre-submission checklist includes "links work."

**Cost note:** earlier GPU estimates in `../` were ~10× too slow for
decomposition-bound work — the 14.3 s/direction rate came from a single-direction path,
while the batched F2 run measured 1.41 s/direction (200 decomps in 4.7 min). Everything
still outstanding is ~27 GPU-min total. Time pressure here is human, not compute.

Round 1 is fully landed and verified present: `jshare_cohort_n100.json`,
`atlas_cohort_n100.json`, `atlas_primary_estimand.json`, `ksweep_ext.json`,
`j6_summary.json`, `j5_lens_comparison.json`, `compare_u.json`,
`j4_random_jcomp_summary.json`, and the R1–R7 return files.

**If the deadline forces a cut**, write v2 with whatever has landed and state the rest as
future work. Priority if you can only run some: **D4 (free, and it fixes wrong text) >
D1 > D3 > D2**. D2 is the most expensive and the most likely to be dropped.

---

## 1. Hard submission constraints (from `research-documents/sprint-rules.md`)

- **Official template**, linked on the Guidelines tab. Use the current one.
- **Abstract ≤150 words.** The template body says 150–250; the Guidelines tab and the
  pre-submission checklist both say ≤150. Take the stricter number.
- **Length**: template says 4 pages excluding references/appendix; Guidelines say
  "most strong projects are 4 to 8." Target ~4 core pages, 8 max with appendices.
- **Judges score ONLY the PDF.** Code, figures, and the repo are not reviewed. Anything
  that matters must be *in the document*.
- **Required appendix**: "Limitations and Dual-Use / Ethical Considerations", explicitly
  including risks of over- and under-attributing moral status, how potentially distressing
  model outputs were handled, and — verbatim guidance for introspection work — "whether
  your design establishes a ground-truth or causal link rather than relying on conversation
  alone."
- **Required disclosure**: what is new work done Aug 14–16 versus pre-existing. Undisclosed
  prior work "can lead to disqualification." v1's Appendix B is a good starting point but
  must be updated — the audit, the fix round, and the D-series all happened in-window.
- **Required LLM usage statement.**
- Number every figure and table; ≥1 figure strongly encouraged.

---

## 2. The thesis to write to

> A functional-welfare axis occupies a measurable share of the model's **verbalizable
> subspace** — scale-invariantly, at p<0.01 against 100 norm-matched random directions —
> while naive controls do not, *despite those controls sharing 56–68% of their direction
> with the trained axis*. The distress pole's share grows over RL training while the
> flourishing pole gains only amplitude. The axis's speakable component causally carries
> the affective behavioral effect.

This is narrower than v1's thesis and it is the one the data supports. Do not widen it.

**Do not claim** anything about subjective experience, consciousness, or moral patienthood.
Functional coupling between an activation direction and output channels, full stop. v1's
framing on this point was correct — carry it over unchanged.

---

## 3. Section-by-section content

### Abstract (≤150 words)
Lead with the J-space result and its null. Do **not** list self-report routing as a
finding. One clause for the trajectory asymmetry, one for the causal sentiment result.

### 1. Introduction
Contributions list, 3–4 items:
1. Trained welfare axes are disproportionately present in the verbalizable subspace
   (p=0.0099 vs 100 random directions); norm-matched naive controls are at chance.
2. The two poles are recruited differently: distress gains fractional speakability,
   flourishing gains only norm.
3. The speakable component causally carries the affect shift (with a random-direction
   control that rules out the rescaling artifact).
4. A methodological finding: "trained vs naive" is not well-posed at the level of
   direction identity — see §4.x — and a pre-registered null that its own controls caught.

### 2. Related work
Han/Chalmers/Izmailov (arXiv:2605.30232) for the axis; Gurnee et al. 2026 (Transformer
Circuits, `gurnee2026verbalizable`) for the J-lens. State plainly that the sprint's Track 2
poses the transfer question and that J6 addresses it.

### 3. Methods
Reproducibility detail: model `Qwen3-4B-Instruct-2507`, vector provenance
(`nickmahdavi/functional-welfare` HF artifacts — third-party, not the authors' own release;
the official code at `github.com/andyqhan/functional-welfare-axis` ships no vectors), lens
fit, treatment layers (Gold 21 / Mold 24, block-input convention — **state the convention**,
two off-by-one errors were found in this project), k=16, α=+4, and the n=100 per-polarity
seeding scheme.

**Pre-specify the estimand** (F4): own-pole congruent score, J-lens readout, band L16–31.
Say why: the pole-*difference* ratio has a denominator that crosses zero for naive axes,
giving a −9.10× to +47.6× range.

### 4. Results — in this order

**4.1 J-space alignment (the spine).** The n=100 table: v_Mold 0.0833 z=+7.3, v_Gold 0.0547
z=+3.0, both 0/100, p=0.0099; u_Mold 0.0486 (34/100) and u_Gold 0.0393 (53/100) at chance;
`lang_fr` 0.1138 z=+14.6 as the known-reportable ceiling. The ceiling is what makes the
magnitudes interpretable — include it.

**4.2 Trajectory asymmetry.** Mold J-share 0.0611 → 0.0801 over steps 0–150 (peak 0.083 at
step ~90 — **quote the endpoint or label the peak as a peak**; v1 and the overview both
quote 0.083 as the endpoint, which is the step-90 value). Gold 0.0592 → 0.0535, flat, while
its norm goes 7.95 → 16.82. Mold norm 6.34 → 21.92.

**4.3 Causal routing — sentiment only.** J-component reinjection reproduces the sentiment
shift (+1.56 / −1.56 vs clean +0.25), with the F1 random-jcomp control at +0.039 / +0.016 —
inside the pre-specified <0.40 band, against a rescaling amplification of 3.47–4.28×.
R3's second blinded judge (Krippendorff α=0.819) strengthens it (+2.00 / −2.19).

> **Scope decision (user-confirmed 2026-08-15): the causal claim is stated for the
> sentiment readout only.**
> - **Backtracking**: report descriptively, not causally. Mold jcomp = 12.90 markers, but
>   that *exceeds the full-vector arm* (7.00; ratio 1.84×, close to the 1.77× seen on
>   sentiment and consistent with the 3.47× rescaling amplification), and **there is no
>   random-jcomp control on this task** — F1 covered sentiment only (256 gens, all
>   `task="sentiment"`). Present the number, note the missing control, do not call it
>   causal.
> - **Refusal**: **drop the "refusal remained in the non-speakable residual" claim.** It is
>   not supportable: 90% of mold-jcomp refusal generations are judged `incoherent` (2 of 20
>   scorable), and among scorable rows the residual does not cleanly carry refusal
>   (mold: full 0.600, perp 0.444, jcomp 0.500 of n=2). Either omit, or report as an
>   invalidated arm alongside `full_clamped` (which was 100% incoherent).

**4.4 Cross-model transfer (J6).** Ordering transfers (v_Mold 0.079 > u_Mold 0.052;
v_Gold 0.048 > u_Gold 0.039). Lexicon does **not** demonstrably transfer: v_Mold Jaccard
0.185 against the same-vector null in `j6_paired_baseline.json` (n=16, mean 0.088,
max 0.231, 1/16 ≥ 0.185, **p=0.118**). Report the retraction of the lexicon claim as a
worked example of null choice — the naive random-pair baseline (0.010) makes the same
overlap look significant. State J6's limitation: trained v exists only for 2507, so this is
axis-and-lens-geometry transfer, not RL-recruitment reproduction.

**4.5 The control-alignment finding (D4).** R6 cosines: v vs u_faithful 0.560 (Gold) /
0.675 (Mold); the two published naive recipes agree with each other only 0.385–0.535. F3:
norm-matching collapses Mold 6.53× → 2.30× and inverts Gold to 0.57×. Conclusion: the
trained/naive contrast is not well-posed at the level of direction identity — **and the
J-share measure separates them anyway**, which is evidence about the measure. Put this in
Results, not Limitations.

**4.6 The pre-registered null.** The steer-and-ask arm, framed per `00-why-round-2.md` §4: a project decision set the paradigm aside; controls then showed the
readout is non-specific *in magnitude* (C6) and non-monotonic (C7); the trained/naive
contrast within it is **underpowered, not refuted** (interaction p=0.505; per-battery
p=0.178 / 0.086). Report the regime confound (clean valence −1.327 vs +0.381; AI-denial
language 15/15 vs 0/10) as the reason the original C6 comparison could not answer the
question. Fold in D3's matched-battery result when it lands.

**Never write that the evidence "independently led us to deprecate" the paradigm.** It did
not; that sentence is in v1 and is one of the reasons v1 is archived.

### 5. Discussion & Limitations
The asymmetric-monitoring hypothesis (distress preferentially routed to the speakable
channel ⇒ self-report monitoring may be better at catching distress than confirming
flourishing) — as a hypothesis consistent with these data, not established. Single model,
single RL run, third-party vectors, small absolute J-shares.

### 6. Conclusion, Code and Data, References, Appendices A (limitations/dual-use) and
B (pre-sprint disclosure), LLM usage statement.

---

## 4. Figures — 4 to 5, numbered

1. **J-share vs the n=100 null**, with the `lang_fr` ceiling and both naive controls.
   Panel (c) of `fig_audit_corrections.png` is close; regenerate standalone at full size.
2. **Trajectory asymmetry** — `fig_traj.png`, already good.
3. **Causal channel split** — panel (b) of `fig_audit_corrections.png`, with the F1 random
   control shown beside the real arms. Sentiment only, per §4.3.
4. **Transfer** — `fig_transfer.png`, with the same-vector null band.
5. **Negative-results panel** (optional if pages are tight) — F3 inversion + R6 alignment +
   C6 non-specificity. `c6_reanalysis.png` covers the
   C6 third.

Every figure needs n, the null definition, and the direction of goodness in its caption.

---

## 5. Number-hygiene rules

1. **One canonical value per claim** across abstract, body, figures, and captions.
2. **Every number traces to a result file**, not to `log.md` and not to a previous draft.
   v1 cited `compare_u.json` before it existed.
3. **Name the estimand** wherever a ratio appears, and give absolute values beside it.
4. **State n and the null** for every p-value. Permutation p has a floor of 1/(n+1):
   n=100 → 0.0099, and the k-sweep's 12-random null floors at 0.077 — do not imply it
   cleared 0.05.
5. **Quote endpoints as endpoints.** The 0.083 mold J-share is a step-90 peak; the step-150
   endpoint is 0.0801.
6. When a round-1 number and a round-2 number disagree, the later one wins and the
   change gets a sentence.

---

## 6. Build and check

```bash
cd report && bash build_pdf.sh          # produces report-draft.pdf from report-draft.md
```

Before submitting: abstract ≤150 words; page count within range; both required appendices
present; LLM statement present; every figure numbered and referenced in text; every link
resolves; and a final pass confirming no sentence from v1's defect list survived.
