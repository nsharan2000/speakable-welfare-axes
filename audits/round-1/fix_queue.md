# Fix Queue — runnable specifications

Ordered by rubric gain per GPU-minute. Costs are measured from this repo's logs
(see audit_report.md §7). Each item states the exact grid so it can be dispatched
without re-deriving anything.

Total GPU for F1–F7: **152 min (2.5 h)**. F4/F5/X4 need zero GPU.

---

## F1 — Random-jcomp control arm for J4   [1.4 min GPU · +1.0 rubric]

**Why.** `chain-g` measured that J-component rescaling inflates *any* direction
(random-jcomp null +3.7…+10.7 on the first-token readout). J4 has arms
{clean, full, jcomp, perp, full_clamped} but **no random-jcomp**, so J4's
"affect travels through the speakable component" is not separated from a
rescaling artifact.

**Grid.** For each of 8 random directions per polarity (reuse the `rand_gold*`/
`rand_mold*` vectors already in `mech_decompositions.json` so the null matches):
decompose at the same treatment layer (Gold L21, Mold L24), take the J-component,
rescale to ‖v‖ exactly as `j4_dissociation.py` does for the real axes, then run the
same three readouts (sentiment on 16 neutral prompts, backtracking, refusal).

**Minimum viable version.** Sentiment only, 16 gens × 2 polarities = 32 generations
≈ 30 s. This is enough to state whether random-jcomp reproduces the ±1.56 effect.

**Decision rule.** If random-jcomp sentiment |mean| ≥ ~0.8 (i.e. more than half the
observed ±1.56), the J4 causal claim must be reported as confounded by rescaling and
downgraded to descriptive. If randoms stay near clean (+0.25), the claim is controlled
and can be stated causally — with the null reported alongside it.

**Implementation.** Extend `experiments/j4-behavioral/j4_dissociation.py`: add
`arm="rand_jcomp"` iterating the random vectors; no other change.

---

## F2 — J-share random cohort 8 → 100 per polarity   [48 min GPU · +1.0 rubric]

**Why.** The repo's strongest signal (v_Mold z = +11.2, v_Gold z = +5.1, naive at
chance) reports as p = 0.111 because a permutation test over 8 randoms cannot go
below 1/9. This is the claim the paper's thesis rests on.

**Grid.** 100 random directions × 2 polarities = 200 gradient-pursuit decompositions
at k = 16, at the corresponding treatment layer. Measured 14.3 s each → 47.7 min.
Sample randoms as Gaussian in R^2560 normalized to the matched ‖v‖ (same procedure
as the existing `rand_*` entries), fixed seed, saved with the seed recorded.

**Report as.** var_fraction for v/u vs the 100-direction null: z, exact permutation p
(floor 0.0099), and the null's 5th/95th percentiles. Also state the language
direction (z = +22.9) as the known-reportable calibration ceiling.

**Note.** Run this before F3 — it is half the cost and covers the more important claim.

---

## F3 — Atlas pole-score randoms 6 → 100 per polarity   [89 min GPU · +0.8 rubric]

**Why.** The "trained reads out its own valence above naive" claim has a 6-direction
null (p floor 0.143).

**Grid.** Restrict to the workspace band L16–31 (16 layers) and the **one**
pre-specified readout convention from F4, 1 variant: 100 × 2 × 16 = 3200 rows at
1.67 s = 89 min. Do **not** run all 35 layers × 3 variants (that is 585 min for no
additional claim).

**Also fix.** Store randoms under both concept labels, or record explicitly that the
mold null is read from the `mold_pole` field of `concept='gold'` rows.

---

## F4 — Pre-specify one atlas estimand   [0 GPU · +1.4 rubric]

**Why.** The trained/naive own-pole ratio spans 0.62×–6.53× across conventions and
inverts (v < u) in 3 of 12 cells, including the normalized J-lens readout at the Gold
treatment layer (0.62×). The report quotes 3.9×/9.4×, which come from a *different*
estimand (the pole difference) whose range is wider still: −9.10× to +47.6×.

**Action.**
1. Choose the estimand *on stated grounds*, not on the number it produces. Recommended:
   **own-pole congruent score under the J-lens readout, averaged over the workspace band
   L16–31** — it uses the band the instrument was validated on and does not depend on
   picking a single treatment layer.

   *Revised from the pole-difference form after quantifying both:* the ratio of pole
   *differences* is unstable by construction, spanning **−9.10× to +47.6×** across the
   same conventions because its denominator (a difference) passes through zero for naive
   axes. A ratio whose denominator can vanish is not a usable estimand. The own-pole
   ratio spans 0.62×–6.53× over the same grid. Report absolute pole scores alongside
   every ratio either way.
2. Name it in Methods as the primary readout.
3. Add a robustness table with all three conventions × two aggregations — **including
   the inversion**. Disclosing it is a Dimension-2 gain; having a judge find it is a loss.
4. Recompute every ratio in the report from the chosen estimand. Current numbers in
   §4.2 do not correspond to any single convention.

---

## F5 — Reframe the primary arm as a pre-registered null   [0 GPU · +1.4 rubric]

**Why.** C1/C6/C7 undercut the `H1_verbalizable` decision: 71.6% of the Gold gap is u
falling below random, v_Gold is inside the random band (4/20, p = 0.238), and v_Mold's
effect is the same size on unrelated factual prompts (+4.54 vs +3.71) with C7
monotonicity failing for 3 of 4 directions.

**Action.**
1. Remove "self-report routing" from the abstract's list of pillars.
2. Report the arm as: pre-registered contrast → decision rule met on the *paired v−u*
   statistic → **but** the pre-specified controls C1/C6/C7 show the contrast does not
   support a self-report-channel interpretation. State all three explicitly.
3. Keep the decomposition table (v−random vs random−u) — it explains *why* a paired
   contrast can mislead when the control direction is not neutral. That is a
   methodological contribution other steering papers would benefit from.
4. Say plainly that this motivated deprecating the steer-and-confess paradigm
   mid-project, and that the deprecation was made **before** the audit, on
   independent grounds (chain-g rescaling fragility).
5. `primary_analysis.json` keeps `decision: H1_verbalizable`. Either add an
   `interpretation` field recording the control-based override, or emit a sibling
   `primary_analysis_controls.json`. Do **not** edit `pre-registration.md` — it is
   frozen, and the deviation belongs in the report as a deviation.

---

## F6 — Generate the missing compare_u.json   [2 min GPU · +0.7 rubric]

`report-draft.md` §4.6 cites `compare_u.json`; it does not exist. The cos 0.535 (gold)
/ 0.144 (mold) values appear only in `log.md:488`. Run `experiments/welfare-axis/
compare_u.py` to emit the JSON so the citation resolves. If it cannot be re-run, cite
`log.md` explicitly instead — a dangling citation to a nonexistent results file is the
cheapest possible Dimension-2 loss.

---

## F7 — J3 mold at its own treatment layer   [12 min GPU · +0.6 rubric]

The k-sweep grid is L17, 19, 21, 23, 25, 27, 29 — odd layers only — so **mold's
treatment layer L24 is never evaluated**, yet the k-robustness claim is stated for both
poles. Add L24 (and L22 for symmetry) × k ∈ {4,8,16,25,50} × {step95, naive_faithful,
2 randoms}. ~48 decompositions ≈ 12 min.

---

## X4 — Gold/Mold asymmetry as a stated finding   [0 GPU · +0.9 rubric]

All data is already in hand: J-share 0.083 (Mold) vs 0.055 (Gold) against a chance
baseline of 0.038–0.045; trajectory shows Mold's J-share rising 0.061 → 0.083 over RL
while Gold stays flat and gains only amplitude. The distress pole becomes speakable;
the flourishing pole gets louder without becoming more speakable.

This directly answers the Track 2 framing and needs writing, not compute. It also
subsumes checklist item X.4, which can be closed without running.

---

## Cut: X1, X2

Both re-use the steer-and-ask channel that C6/C7 just invalidated. **C7 already ran
X.2's dose–response and it failed** for 3 of 4 directions; C6 showed the readout is not
self-report-specific. Spending 80 GPU-min re-measuring a broken instrument, and
reviving a deprecated paradigm in the writeup, scores negative on both D1 and D2.
The checklist's "cheap, high evidentiary value" note for X.2 predates that evidence.

---

## Suggested order

1. **F5, F4, X4** — zero GPU, highest combined gain (+3.7). Do these first regardless
   of what the GPU is doing.
2. **F1** (1.4 min) — gates whether J4's causal claim can be stated at all.
3. **F6** (2 min), **F7** (12 min) — cheap defect closures.
4. **F2** (48 min) — hardens the thesis-bearing claim to p < 0.01.
5. **F3** (89 min) — same for the atlas claim, after F4 fixes the estimand.
6. **J6** (150 min) — the only item that materially moves Impact/Innovation; answers
   the organizers' stated open question and counts as in-window work.
7. **J5** — already in flight; let it finish, but it is not on the critical path for
   any claim the paper needs.

Steps 1–5 total **152 GPU-min**. Everything the paper's claims depend on is hardened
inside 2.5 hours of GPU time, which leaves the rest of the runway for J6 and writing.
