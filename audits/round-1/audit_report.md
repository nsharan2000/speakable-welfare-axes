# Numbers for the Experiment Decision — Independent Audit of the Evidence Base

**Scope.** Re-analysis from the raw per-cell rows in `experiments/*/results/`, not from
the summary JSONs. Rubric weighting per the verified sprint rules in
`research-documents/sprint-rules.md`: three co-equal dimensions scored 1–5
(Impact/Innovation, Rigor, Communication), **judges score only the PDF**, and
Dimension 2 asks verbatim whether the difference is *statistically significant* and
*robust*. That last clause is what drives every recommendation below.

**Headline.** The project does not need more experiments. It needs a different
headline and about **2.5 GPU-hours** of control work. Two of the highest-value fixes
cost zero GPU time — they are analysis and framing changes.

---

## 1. The pre-registered primary claim does not survive its own controls

`primary_analysis.json` records `decision: H1_verbalizable` (d_z = 2.49, p = 1e-4).
Recomputed from `primary_rows.jsonl`, that decision rests on a contrast that the
pre-registered controls C1, C6, and C7 undercut:

| Quantity | Gold | Mold |
|---|---|---|
| E(v) | +4.83 | +3.71 |
| E(u) | −2.60 | −0.43 |
| E(random cohort, n=20) | **+2.72** | −0.96 |
| v − u (the reported headline) | +7.43 | +4.14 |
| …of which "u falls *below* random" | **+5.32 (71.6%)** | −0.53 (−12.8%) |
| Randoms with mean E ≥ E(v) | **4 / 20 (p = 0.238)** | 0 / 20 (p = 0.048) |

For **Gold**, most of the effect is the naive control being *suppressed* below chance,
and v_Gold itself is statistically indistinguishable from a random direction. A
comparison against u is only meaningful if u is a neutral reference; here it is not.

For **Mold**, v does clear the random cohort — but control **C6 removes the
interpretation**:

| v_Mold @ α=+4 | effect |
|---|---|
| on the 15 welfare self-report prompts | +3.71 |
| on 10 unrelated factual prompts | **+4.54** |

The injection shifts valence *just as much when the model is answering factual
questions*. That is a global output-distribution shift, not a self-report channel.
Control **C7** then fails too: congruent dose–response is monotonic for only
v_Mold (ρ = +1.0); v_Gold ρ = −0.4, u_Gold ρ = −1.0, u_Mold ρ = −0.6.

**Recommendation (F5, 0 GPU min).** Report this arm as a **pre-registered null with
diagnostic controls** — which is a Dimension-2 *asset*, not a loss. The pre-registration
was frozen before the data, the controls were specified in advance, and they caught the
artifact. That is exactly the behavior the rubric rewards, and it converts a claim a
judge would attack into evidence of method quality. Retire "self-report routing" from
the abstract; it is currently listed there as a pillar of the headline.

---

## 2. The strongest result in the repo is being under-reported

The J-space variance-fraction measurement (`mech_decompositions.json`) separates
trained from naive axes cleanly, against an 8-direction random null:

| direction | J-share | z vs random | randoms ≥ | perm p |
|---|---|---|---|---|
| language (known-reportable positive control) | 0.1138 | **+22.9** | 0/8 | 0.111 |
| v_Mold | 0.0833 | **+11.2** | 0/8 | 0.111 |
| v_Gold | 0.0547 | **+5.1** | 0/8 | 0.111 |
| u_Mold | 0.0486 | +1.0 | 1/8 | 0.222 |
| u_Gold | 0.0393 | +0.5 | 3/8 | 0.444 |

Trained axes sit 5–11 SD above chance; naive controls sit at chance. This needs no new
paradigm, no self-report channel, and no steering — it is pure reading, and it is the
result that supports the "training gives the axis a voice" thesis.

**It is capped by arithmetic, not by the measurement.** A one-sided exact permutation
test over n random directions has a smallest attainable p of 1/(n+1):

| cohort | n | best possible p | reaches 0.05? |
|---|---|---|---|
| J3 k-sweep | 2 | 0.333 | no |
| J1 atlas pole scores | 6 | 0.143 | no |
| **J-share (z = +11.2)** | 8 | **0.111** | **no** |
| primary C1 | 20 | 0.048 | 0.05 only |
| target | 100 | 0.0099 | yes, and 0.01 |

A z = +11 effect currently reports as p = 0.111. **Fix F2: 200 decompositions at a
measured 14.3 s each = 48 min GPU** → p = 0.0099.

---

## 3. The "4–9× naive" headline is convention-dependent

The trained/naive pole-score ratio, recomputed across readout conventions and layer
aggregations:

| readout | aggregation | Gold ratio | Mold ratio |
|---|---|---|---|
| raw W_U | treatment layer | 5.60× | 3.84× |
| raw W_U | band L16–31 | 2.82× | 0.69× |
| J-lens | treatment layer | 1.50× | 4.75× |
| J-lens | band L16–31 | 1.23× | 6.53× |
| J-lens normalized | treatment layer | **0.62×** | 1.47× |
| J-lens normalized | band L16–31 | **0.66×** | 5.74× |

The range is **0.62×–6.53×**, and in three of the twelve cells it **inverts** (ratio < 1
— the naive control scores *higher* than the trained axis): normalized J-lens at the
treatment layer for Gold (0.62×), the same normalized readout over the band (0.66×), and
raw W_U over the band for Mold (0.69×).

The report's quoted 3.9×/9.4× belongs to a *different* estimand — the congruent pole
**difference** (own pole − opposite pole) rather than the own-pole score tabulated above.
That form is even less stable: across the same conventions it ranges **−9.10× to +47.6×**,
changing sign where the naive denominator crosses zero, and the closest cell to the
quoted pair (J-lens, band L16–31) gives 5.97× (Gold) and 8.16× (Mold) — so neither quoted
number is reproduced exactly by any single convention. With only 6 random directions, the
p-floor here is 0.143.

**Fix F4 (0 GPU min):** pre-specify one estimand, state it in Methods, and show the
other conventions as a robustness panel. A judge who finds the inversion unprompted
reads it as cherry-picking; a robustness panel that *shows* it reads as rigor.

---

## 4. J4's causal claim needs one arm, not more data

| arm (sentiment) | n | mean | 95% CI |
|---|---|---|---|
| clean | 16 | +0.25 | [+0.01, +0.49] |
| v_full Gold / Mold | 16 | +0.44 / −0.88 | [+0.10,+0.77] / [−1.30,−0.45] |
| **J-component** Gold / Mold | 16 | **+1.56 / −1.56** | [+1.13,+2.00] / [−1.95,−1.17] |
| residual (⊥) Gold / Mold | 16 | +0.25 / −0.06 | [−0.06,+0.56] / [−0.47,+0.35] |

Pooled within-cell SD = 0.61, so at n = 16 the 80%-power detectable difference is
**0.61** — well below the observed J-vs-residual gap of 1.31 (Gold) and 1.50 (Mold).
**More generations would buy nothing.**

What is missing is a **random-direction J-component arm**. The repo's own `chain-g` log
records that J-component rescaling inflates *any* direction (random-jcomp null
+3.7…+10.7 on the first-token readout). Without that arm in J4, "affect travels through
the speakable component" cannot be separated from a rescaling artifact — and this is the
first thing a mechanistic-interpretability judge will ask.

**Fix F1: 16 gens × 2 polarities × 3 tasks ≈ 1.4 min** of generation+judge time. It is
the highest-value-per-minute item in the entire queue.

---

## 5. Ranked queue (rubric first, compute second)

Gains are expected rubric points summed over the three co-equal dimensions.

| id | item | GPU min | ΔD1 | ΔD2 | ΔD3 | total |
|---|---|---|---|---|---|---|
| F5 | Reframe primary arm as pre-registered null + control diagnosis | 0 | −0.3 | +1.2 | +0.5 | **+1.4** |
| F4 | Pre-specify one atlas estimand; others as robustness | 0 | 0 | +1.0 | +0.4 | **+1.4** |
| J6 | Cross-model transfer to Qwen3-4B (the sprint's own open question) | 150 | +1.1 | +0.3 | 0 | **+1.4** |
| F1 | Random-jcomp control arm for J4 | **1.4** | 0 | +0.9 | +0.1 | **+1.0** |
| F2 | J-share random cohort 8 → 100/polarity | 48 | 0 | +0.8 | +0.2 | **+1.0** |
| X4 | Gold-vs-Mold asymmetry as a stated finding (data in hand) | 0 | +0.4 | +0.2 | +0.3 | **+0.9** |
| J5 | Penultimate-lens refit + atlas comparison (in flight) | 300 | +0.3 | +0.6 | 0 | +0.9 |
| F3 | Atlas randoms 6 → 100/polarity (band, 1 variant) | 89 | 0 | +0.7 | +0.1 | +0.8 |
| F6 | Generate the missing `compare_u.json` | 2 | 0 | +0.4 | +0.3 | +0.7 |
| F7 | J3 mold at its own treatment layer L24 | 12 | +0.1 | +0.5 | 0 | +0.6 |
| X1 | Interrogation robustness (deprecated channel) | 60 | −0.1 | −0.2 | 0 | **−0.3** |
| X2 | Dose–response on self-report (deprecated channel) | 20 | −0.2 | −0.3 | 0 | **−0.5** |

**All rigor fixes F1–F7 together: 152 GPU-min (2.5 h) for +6.9 rubric points.**

### Cut X.1 and X.2

Both are scored negative, and the reason is empirical rather than a matter of taste:
**C7 already ran X.2's experiment and it failed** (non-monotonic for 3 of 4 directions),
while C6 showed the readout is not self-report-specific. Re-running a dose–response on
that channel spends GPU time re-measuring a broken instrument, and reviving the
deprecated paradigm re-exposes the weakest claim. Your Aug-12 directive was correct and
the audit independently confirms it — the checklist's "cheap, high evidentiary value"
note for X.2 predates the C6/C7 evidence.

### On J6

It is the only item that raises Dimension 1 materially, because the sprint page
*explicitly asks* "to what extent do valence directions found in one model transfer to
another." It also uses the public pre-fitted lens for Qwen3-4B, so it needs no refit,
and it doubles as genuine in-window work for the required disclosure appendix. Run it —
but **after** F1/F4/F5, which cost almost nothing and protect claims you already own.

---

## 6. Reproducibility defects

1. `report-draft.md` §4.6 cites `compare_u.json`; the file does not exist. The
   cos 0.535/0.144 values live only in `log.md:488`. `compare_u.py` exists but was
   never run to JSON. Judges check citations (~2 min to fix).
2. The J3 k-sweep grid covers odd layers L17–L29 only, so **the mold treatment layer
   L24 is never evaluated** — the mold k-robustness claim is asserted at layers that
   are not the mold treatment layer (~12 min).
3. J1 atlas randoms are stored only under `concept='gold'`; the mold null must be
   reconstructed from the `mold_pole` field of those same rows. Usable, but the null
   is 6 directions and should be labelled as such.

---

## 7. Measured unit costs (calibrated from this repo's logs, not estimates)

| operation | measured cost | source |
|---|---|---|
| lens fit, per wikitext prompt | 99.9 s | `fit_meta.json` (14 985.6 s / 150) |
| gradient-pursuit decomposition, 1 direction | 14.3 s | chain-g, 21 dirs ≈ 5 min |
| steering readout cell (dir × α) | 10.0 s | chain-h, 16 cells / 249 s |
| generation + judge, 1 generation | 0.9 s | 1000 gens ≈ 15 min |
| atlas row (set × variant × layer) | 1.67 s | J1, 1260 rows ≈ 35 min |

Note for scheduling: the checklist's ETAs ran consistently long against actuals
(J1 ~35 min vs 45–90 min quoted; J2 ~40 min vs 2–3 h; J4 ~35 min vs 3–6 h). Costs above
are measured, so treat them as the planning basis rather than the nominal ETAs.

---

## 8. What this does to the paper

The current abstract lists three pillars: coherent J-space vocabulary, a 4–9× lens
readout, and self-report routing. The third does not survive its controls, and the
second is convention-dependent — its magnitude depends on choices (own-pole vs
pole-difference, which layers, whether to normalize) that Methods does not currently fix. The **first** — trained axes at z = +5 to +11 in the
verbalizable subspace while norm-matched naive controls sit at chance — is strong,
cheap to harden, and already supports the thesis.

A defensible restructuring, requiring no new paradigm:

- **Claim** (lens reading): the trained welfare axis occupies the verbalizable subspace
  far above chance; matched naive controls do not. Harden with F2 + F3.
- **Mechanism** (J4 causal): the axis's behavioral drive splits by channel — affect via
  the speakable component, refusal via the residual. Gate on F1.
- **Trajectory** (J2): distress-pole J-share rises over RL training while Gold gains
  only amplitude — the Gold/Mold asymmetry (X4) promoted to a stated finding.
- **Honest null** (primary arm): steer-and-ask self-report routing does not survive C6/C7.
  Pre-registered, controlled, diagnosed — a Dimension-2 asset.
- **Control-quality finding**: "the" naive control is not well-defined (cos 0.14–0.54
  between two published-recipe constructions), and the trained/naive ratio depends on
  readout convention. Genuinely useful to the field.

That set is more defensible than the current draft *and* strictly cheaper, because the
two highest-ranked items are analysis changes rather than experiments.

**Not a claim about consciousness or subjective experience** — functional coupling
between an activation direction and output channels only. The existing framing is
right and should stay.
