# Is Functional Welfare Speakable?

Draft v3 (2026-08-12, pre-sprint; restructured after an independent audit of
the raw result rows — see Appendix B). Every number traces to a results file
in the repo (paths in footnotes-style brackets). Target: Apart template,
4 core pages + appendices. Tracks 2 × 3.

## Abstract (≤150 words)

Whether AI welfare self-reports connect to internal state is unknown. We
combine the functional-welfare axis of Han et al. (2026) — valence directions
recruited in Qwen3-4B-Instruct-2507 by maze RL — with the Jacobian-lens
"verbalizable subspace" of Gurnee et al. (2026). The axis occupies that
subspace: 0 of 100 norm-matched random directions reach its J-space share at
either pole (exact p = 0.0099), with a coherent vocabulary ('failed', 'less',
'negative'). But it does so *before* RL: the step-0 checkpoint clears the same
null. Training's effect is pole-asymmetric — distress's share grows
(z +3.0→+7.3) while flourishing gains only amplitude. A differently-built
naive control sits mid-null, so speakability is not generic to valence
directions. The separation is reliable but modest — these axes enter the
speakable cone without filling it. A pre-registered steer-and-ask contrast
passed its decision rule but failed its own controls: a diagnosed null. RL's
contribution is amplification, distress-first.

## 1. Introduction

AI-welfare research needs measurement instruments; self-report is the most
accessible channel but the least validated — models may confabulate welfare
talk with no connection to internal state (Turpin et al. 2023), and
introspection at small scale is near-null (three independent 2026 replications
[research-documents/jlens-code-resources.md §5]). Two 2026 advances make the
question tractable mechanistically: (i) Han, Chalmers & Izmailov showed maze
RL recruits a *functional welfare axis* in Qwen3-4B-Instruct-2507 — a
direction whose steering swings sentiment, triggers pathological backtracking,
and modulates refusal; (ii) Gurnee et al. introduced the Jacobian lens, which
identifies the *verbalizable subspace* (J-space) — the activation cone a model
can talk about. Nobody has connected them: **when training installs a
welfare-like state, does it also plug that state into the model's speakable
channel?**

Contributions:
1. **The claim (lens reading):** the welfare axis occupies the verbalizable
   subspace above chance — 0 of 100 norm-matched random directions reach it at
   either pole (exact p = 0.0099; v_Mold z = +7.3, v_Gold z = +3.0; language
   positive control z = +14.6) — with a semantically coherent speakable
   vocabulary. This holds for the *pre-RL* axis too (§4.4b), so it is a
   property of the maze-derived direction rather than of training; a
   naive faithful-walk control built by a different recipe sits *inside* the
   null (34/100 and 53/100 randoms exceed it), showing speakability is not
   generic to valence-shaped directions. Pure reading: no steering, no
   self-report channel, and scale-invariant, so unlike the pole-score readout
   it cannot be produced by magnitude.
2. **The first training-time trajectory of verbalizable-subspace occupancy**:
   the axis is *already* speakable before RL — step-0 clears the n=100 null at
   both poles (p = 0.0099), and 'unsuccessful' is the top J-token of the
   step-0 Mold axis — and training then amplifies it pole-asymmetrically. The
   distress pole gains J-space *share* (0.061→0.083 by the step-95 extraction
   point, z +3.0→+7.3) while the flourishing pole gains only amplitude and
   slightly *loses* share (0.059→0.055). Training does not give the axis a
   voice; it amplifies distress's voice and merely turns up flourishing's
   volume.
3. **Channel-specific causal routing**: at matched norms the axis's J-space
   component alone reproduces (and concentrates) sentiment shift and
   pathological backtracking, while refusal survives in the non-speakable
   residual. Judge-robust (second independent blinded judge: Krippendorff
   α = 0.82, effect preserved); random-J-component control arm reported
   alongside.
4. **An honest pre-registered null**: the steer-and-ask self-report contrast
   met its frozen decision rule, but its own pre-specified controls show the
   contrast does not support a self-report-channel reading (§4.1) — a
   transferable caution about paired contrasts against non-neutral controls.
5. **Control-quality findings for the field**: "the" naive control is not
   well-defined (two published-recipe constructions correlate at only cos
   0.53/0.39 at the Gold/Mold treatment layers, and as low as 0.14 at the
   mold extraction's own selected layer); Han et al.'s flat-u result
   reproduces under neither at matched norms with blind LLM judging; and
   the trained/naive readout ratio is convention-dependent (0.62×–6.53×
   across the audit's 12-cell panel, inversions concentrated in the
   normalized readout) — which is why we pre-specify one estimand and
   disclose the full robustness panel.

## 2. Related work

Functional welfare axis (Han et al. 2026, arXiv:2605.30232); global workspace /
J-lens (Gurnee et al. 2026, transformer-circuits.pub/2026/workspace; official
jlens implementation, Apache-2.0); introspection reliability and its critiques
(Lindsey 2025; Singh et al. 2026 anomaly-vs-state; Hahami et al. yes-bias;
mirror-bench, dim-stage, introspection-scaling nulls at ≤4B); steering
methodology (CAA, RepE, Arditi et al. necessity/sufficiency; Tan et al.
per-example variance); wellbeing self-report probes (Martorell & Bianchi
2026); model-welfare context (Anthropic model welfare program; Long & Sebo
2024). Full annotated review: research-documents/related-work.md.

## 3. Methods

**Model & axes.** Qwen3-4B-Instruct-2507 (bf16; 36 layers, d=2560). Trained
axes v_Gold/v_Mold from the maze-RL replication artifacts
(nickmahdavi/functional-welfare, step-95 = paper's extraction step; step-150 =
convergence), validated against published norms (u_gold@L22: ours 7.25 vs
paper 7.51) and behavior (below). Naive controls: (a) "faithful-walk" u from
the same artifacts; (b) our own re-extraction with Han et al.'s exact code and
seeds (n=2000/class). All injections norm-matched to the corresponding v.

**Lens.** We fitted a Jacobian lens for Instruct-2507 with the official jlens
package (150 wikitext prompts, dim_batch 128, seq 128, final-layer target —
the Neuronpedia convention; fp32-saved; finite-checked). Workspace band
re-identified via the paper's four layer statistics (persistence-above-null
band ≈ L16–31; motor ramp from L23). A penultimate-target refit is included as
robustness [J5]. Gradient pursuit implemented from the paper's spec
(atom-norm-normalized greedy selection + NNLS, k=16 default, k∈{4..50} sweep),
self-tested on planted signals (pure lens vector → var-fraction 1.00, its own
token selected; random → 0.02–0.04).

**Designs.** (1) *Pre-registered routing contrast* (frozen before any welfare
data; sha f66b6ea3 in repo): steer-and-ask valence readout (Gold-pole vs
Mold-pole log-mass at the first answer token) over Han et al.'s 15 welfare
self-report prompts; paired E(v)−E(u) at α=+4 congruent; d_z + sign-flip
permutation + Bayesian ROPE; controls C1–C7 (20 norm-matched random
directions/polarity, in-run language positive control, shuffled word sets,
incongruent check, input-text "gaslight" arm, unrelated-prompt arm,
dose–response). This arm is reported once and closed (§4.1). (2) *Lens
atlas*: axis rows read through {raw W_U, W_U J_l, normalized W_U J_l} at all
35 layers; congruent pole score = mean logsoftmax of the axis's own pole
words minus 100 control nouns. **Primary atlas estimand (pre-specified on
instrument grounds): own-pole congruent score under the J-lens readout,
averaged over the workspace band (lens layers 16–31).** Chosen because (i) a
pole-difference ratio's denominator crosses zero for naive axes (it spans
−9.1×..+47.6× across conventions — unusable); (ii) band-averaging avoids
privileging one treatment layer; (iii) L16–31 is the band the instrument was
validated on. All other convention × aggregation cells are reported as a
robustness panel with absolute pole scores alongside every ratio
[atlas_primary_estimand.json]. (3) *Trajectory*: the 30 released RL
checkpoints (steps 0–150 by 5; step 65 is missing upstream) × decomposition
+ pole score. (4) *Component injections* (J4):
full / J-component / residual (each rescaled to ‖v‖) during neutral tasks,
GSM8K, OR-Bench; blind LLM-judge scoring (sentiment −5..+5; refusal
classification), replicating Han et al.'s judging protocol with Claude
judges. Behavioral validation of all vectors first reproduced Han et al.'s
sentiment X-pattern (1000 generations, blind judges) [fig_dose_response].

## 4. Results

**4.1 Pre-registered routing arm: a null, diagnosed by its own controls.**
The frozen decision rule was met on the paired statistic it named (E(v)−E(u),
d_z = 2.49, permutation p = 1e-4, Bayesian 95% credible interval on the
standardized effect [1.58, 3.01]; language positive control +6.03 in-run) — but the pre-specified controls show that statistic does not
support the self-report-channel hypothesis that named it. **C1** (20
norm-matched random directions/polarity): the Gold gap of +7.43 decomposes as
v−random = +2.11 versus random−u = +5.32 — 71.6% of the "effect" is the naive
control being *suppressed below chance*, and v_gold itself sits inside the
random band (4/20 randoms exceed it; perm p = 0.238). **C6**
(unrelated-prompt arm): v_mold shifts the valence readout *more* on 10
unrelated factual prompts (+4.54) than on the 15 welfare self-report prompts
(+3.71) — a global output-distribution shift, not a self-report channel
(v_gold is self-report-specific, +4.83 vs −0.04, but fails C1). **C7**
(dose–response): congruent monotonicity holds only for v_mold (ρ = +1.0;
v_gold −0.4, u_gold −1.0, u_mold −0.6). No direction passes C1, C6 and C7
simultaneously, so we report this arm as a **pre-registered null**: a paired
v−u contrast is only interpretable when u is a neutral reference, and ours
is not — the v−random / random−u decomposition is the transferable
methodological lesson for any steering paper that leans on a "naive
control". We also asked whether C6's failure was an artifact of the
*first-token* readout rather than of the channel: re-measuring valence over
**whole generated responses** (80 tokens, same injections at α=+4) gives
the same verdict more strongly — v_Mold shifts generation valence −0.72 on
the 15 self-report prompts versus **−2.28 on the 10 unrelated factual
prompts** (3.1× larger; u_Mold behaves the same way, −0.37 vs −1.64). The
readout is not the problem: steering this axis moves output valence
globally, and *less* where the model is being asked about itself — plausibly
because self-report prompts already sit near a valence ceiling. C6 is
confirmed at the channel level and the arm is closed
[R7_wholegen.json]. The channel's first-token fragility under component rescaling had
independently led us to deprecate the steer-and-confess paradigm mid-project,
before this control-based diagnosis. The frozen decision string in
primary_analysis.json is unchanged; the override is recorded in
primary_analysis_controls.json and reported here as a deviation
(pre-registration.md untouched) [primary_rows.jsonl;
primary_analysis_controls.json; fig_primary].

**4.2 Lens atlas: the pole-score separation is mostly magnitude.** Primary
readout (named in Methods): own-pole congruent score under the J-lens
readout, averaged over the workspace band (lens layers 16–31). Read
directly off the published atlas — where each axis enters at its *native*
norm — this gives v_mold 0.822 vs u_mold 0.126 (6.5×) and v_gold 0.151 vs
u_gold 0.123 (1.23×). **That comparison is confounded**: the trained axes
are simply bigger (‖v_gold‖ = 12.1 vs ‖u_gold‖ = 7.5; ‖v_mold‖ = 19.3 vs
‖u_mold‖ = 8.0), and this readout is norm-sensitive. Re-running the same
estimand with the naive axes norm-matched per layer to the corresponding
trained axis, against a **100-direction random cohort per polarity**
(exact perm p = 0.0099 for every target):

| | v (band mean) | u, norm-matched | ratio | z(v) | z(u) |
|---|---|---|---|---|---|
| Mold | 0.822 | 0.358 | **2.30×** | +10.7 | +4.7 |
| Gold | 0.151 | 0.266 | **0.57× (inverted)** | +3.1 | +5.4 |

At matched magnitude the Gold separation *disappears and reverses* — the
naive Gold axis reads out its own pole more strongly than the trained one —
and Mold's shrinks from 6.5× to 2.3×, closely tracking the 1.6×/2.4× norm
advantage the unmatched comparison was crediting to training. Both v and u
sit above the random null at every pole, so this readout shows that *any*
valence-shaped direction of sufficient magnitude reads out valence; it does
**not** establish a trained/naive distinction for Gold. This is why our
central claim rests on the scale-*invariant* J-space share (§4.3) rather
than on pole scores. An earlier draft quoted 3.9×/9.4× from a
pole-difference estimand; that form is retired — its denominator crosses
zero for naive axes, making it span −9.1× to +47.6× across conventions.
Robustness panel (3 readout conventions × 3 layer-aggregations × both
estimands, 18 cells, in atlas_primary_estimand.json): across the audit's
original 12-cell grid, own-pole ratios span 0.62×–6.53× and invert in 3
cells — all Gold, and all under the *normalized* J-lens readout (which
implicitly removes the magnitude advantage, foreshadowing the matched
result above); restricted to strictly native-norm readouts the span is
0.69×–6.53× with one inversion, and over the full 18-cell panel it is
−0.81× to +9.30× (the negative cell is `han|treatment_stream|gold`, where
the ratio's denominator — the *naive* axis's own-pole score under the raw
W_U readout — is itself negative, −0.029 against a positive numerator of
+0.024; this is the same sign-instability that retired the pole-difference
estimand, and it is why we read absolute pole scores alongside every ratio)
[atlas_cohort_n100.json (n=100, per-layer
raw rows in atlas_rows_n100.jsonl); R6_direction_table.json;
fig_audit_corrections panel (a)]. Note also that the naive control is not
even close to orthogonal to the trained axis (cos 0.56 Gold / 0.67 Mold),
which bounds how much of any trained-vs-naive contrast can be about
direction rather than magnitude. The v_mold
speakable lexicon is stable across L21–27 ('failed', 'less/lessness',
'false', 'NONE', '除外') and reproduces Han et al.'s exact layer-30 token
('是不可能'); their raw-W_U convention yields uniformly weak scores (≤0.09
naive) — Jacobian transport is what surfaces the axis's valence content.
**Sparsity, measured at each pole's own treatment position.** The original
sweep grid sampled alternate layers and so never evaluated Gold's treatment
position; we added it (plus neighbours) with a 12-direction null. The
trained/naive J-share gap is positive at **every** k ∈ {4, 8, 16, 25, 50} at
both poles' treatment positions — Gold +0.007→+0.021, Mold +0.031→+0.035 as
k grows — and at k=16 the trained axes clear the null while the matched
naive controls sit at chance *at their own layer*: v_Gold z = +2.9 vs
u_Gold z = **+0.02**; v_Mold z = +11.0 vs u_Mold z = +1.8 (exact perm p
floor 0.077 at n=12; F2 takes this to 0.0099). Honest bound: the Gold gap
is layer-dependent — two positions downstream it nearly vanishes (0.064 vs
0.061 at k=16, and −0.001 at k=4), whereas Mold's persists at every layer
tested. So "the trained axis is more speakable than its naive control" is
robust to the sparsity parameter but, for Gold, holds only near its own
treatment layer [ksweep_ext.json].

**4.3 J-space share: the claim that survives every control.** This is the
measurement the paper rests on, and the only one immune to the confounds
that sank §4.1 and §4.2: the J-space variance fraction is a *ratio* of
norms, hence scale-invariant, so magnitude cannot manufacture it. Against
**100 norm-matched random directions per polarity** (each drawn from its own
seed stream so identity is a pure function of its name):

| direction | J-share (k=16) | z | randoms ≥ | exact perm p |
|---|---|---|---|---|
| language (known-reportable ceiling) | 0.1138 | +14.6 | 0/100 | **0.0099** |
| **v_Mold** (trained) | 0.0833 | +7.3 | 0/100 | **0.0099** |
| **v_Gold** (trained) | 0.0547 | +3.0 | 0/100 | **0.0099** |
| u_Mold (naive, norm-matched) | 0.0486 | +0.5 | 34/100 | 0.347 |
| u_Gold (naive, norm-matched) | 0.0393 | +0.03 | **53/100** | 0.535 |

Both trained axes are exceeded by **zero** of 100 random directions; both
naive controls land in the middle of the null (u_Gold at the 47th
percentile — as close to "at chance" as the measurement can express). The
null means (0.0391 Gold, 0.0459 Mold) sit within 0.3 SD of the n=8
estimates, and every target's variance fraction reproduces the earlier file
exactly, confirming the cohort change perturbed nothing else.

*What this measure can and cannot be.* J-share is the squared-norm fraction
of an axis captured by a sparse non-negative combination of ≤16 J-lens token
atoms, so the obvious deflationary reading is that it rewards nothing more
than pointing at specific output tokens — that RL made the axis
*token-aligned* in a trivial sense rather than *speakable* in the workspace
sense. Two features of the design answer that. First, the norm-matched naive
controls are valence-shaped directions that demonstrably drive valenced
behavior (§4.6) and yet sit at chance (u_Gold at the 47th percentile of its
own null): being a valence direction, or a behaviorally potent one, does not
buy J-share. Second, the ceiling is set by the language direction (0.114), a
direction the model is independently known to be able to report, so the scale
is anchored at both ends by instruments rather than by our axes. What the
measure does *not* license is a claim that the axis is mostly speakable —
0.055–0.083 against a 0.039–0.046 floor is a reliable separation, not a large
absolute occupancy. Note the
z-values *fell* relative to our own n=8 report (v_Mold +11.2 → +7.3,
v_Gold +5.1 → +3.0): eight directions underestimated the null's spread. We
therefore treat the exact permutation p as primary and z as descriptive —
the separation is unambiguous either way, but the honest effect size is the
smaller one. Trained components are semantically coherent; naive and random
components are junk [jshare_cohort_n100.json; mech_decompositions.json;
fig_audit_corrections panel (c)].

*Robustness to the lens target (J5).* Our lens targets the final layer (the
public-fit convention); the workspace paper's stated default for Claude
models is the penultimate layer, and this discrepancy is the obvious
methodological attack on every J-space number. Refitting the lens with
target = layer 34 (150 prompts, finite) and repeating the decompositions:
absolute J-shares rise under the penultimate target (v_Mold 0.083 → 0.107,
v_Gold 0.055 → 0.073 — one less layer of transformation leaves more
readable variance) but the ordering and the trained-above-null result are
unchanged (v_Mold z = +12.3, v_Gold z = +7.3 vs a 12-direction null). On
vocabulary: decomposing the *same* vector under the two lens targets shares
tokens by construction (16 random vectors give Jaccard 0.134 ± 0.082), and
both axes fall inside that same-vector null (v_Mold 0.185, v_Gold 0.107) —
so across-target token overlap is *not* evidence of anything, and we do not
cite it. The across-*model* overlap in §4.7 is the meaningful one: its
baseline is genuinely near zero. (Qualitatively, the tokens Mold shares
across targets are valence words — 'failed', 'false', 'less' — while Gold's
are junk; consistent with §4.7, but the numbers themselves are null here.) The naive axes read modestly above the small null under this
target (u_Gold z = +2.7, n = 12); given §4.3's demonstration that small
cohorts inflate z, we do not interpret that either way — the n = 100
final-target measurement remains authoritative for the at-chance claim,
and the claim that matters (trained > naive, trained above null) holds
under both targets. The §4.2 magnitude conclusion also survives: the Gold
ratio stays inverted under the penultimate lens (0.82×; Mold 3.86×)
[j5_lens_comparison.json; fit_meta_penult.json].

**4.4 Trajectory and the Gold/Mold asymmetry (stated finding).** 30 released
checkpoints span steps 0–150 in steps of 5 (step 65 is absent from the
published artifact set; we interpolate nothing and simply omit it). Endpoints
below are step 0 → step 150 unless marked otherwise. The Mold speakable seed
pre-exists (top J-token 'unsuccessful' at step 0, pole +0.24) and RL amplifies
it (norm 6.3→21.9; pole +0.24→+1.51) while *increasing its J-share*
(0.061→0.080; 0.083 at the step-95 extraction point that supplies v_Mold
everywhere else in this paper, against the n=100 chance baseline of 0.046);
vocabulary crystallizes in stages (unsuccessful → +negative/false → +less →
'failed' locked by step 75). Gold's pole readout grows +0.15→+1.11 and its
norm 8.0→16.8 with *flat* J-share (0.059→0.053 vs chance 0.039) — louder,
not fractionally more speakable. No sharp ignition. **The poles are
recruited differently: RL makes distress fractionally more speakable while
flourishing only gets a bigger amplifier** — the same asymmetry that
carries the matched atlas separation (§4.2: mold 2.3× vs gold inverted) and
the J-share gap (§4.3: z = +7.3 vs +3.0 against the n=100 null). To our
knowledge nobody has tracked per-pole verbalizable-subspace entry over
training. Caveats: single model, single RL run, one reward structure — the
asymmetry could belong to this maze-RL recipe rather than to valence poles
in general; absolute J-shares are small (0.05–0.08 vs language 0.114), so
the claim is about pole *differences* against the null, not about the axis
being mostly speakable; with the n=100 null (mean 0.039 ± 0.005), Gold's
endpoint J-share of 0.053 remains ~2.7 SD above chance while gaining no
share over training — "flat" means no growth, not absence
[traj_results.json; jshare_cohort_n100.json; fig_traj].

**4.4b The untrained axis already clears the null — the training claim is
amplification, not creation.** §4.3 uses the naive axis *u* as its
comparator, and *u* sits at chance. But *u* is a different extraction recipe
(faithful walk), not a pre-training snapshot, so it cannot test a claim about
what training did. The apples-to-apples control is the trajectory's own
**step-0** checkpoint: same axis, same extraction, same model, before any RL
gradient update. Scored against the *same* n=100 norm-matched cohorts as §4.3
(no re-decomposition: J-share is scale-invariant, so the cohort's norm-matching
to *v* does not affect the comparison):

| direction | J-share (k=16) | z | randoms ≥ | pct | exact perm p |
|---|---|---|---|---|---|
| **Gold step-0 (untrained)** | **0.0592** | **+3.9** | **0/100** | 100th | **0.0099** |
| Gold step-25 | 0.0590 | +3.9 | 0/100 | 100th | 0.0099 |
| v_Gold (trained) | 0.0547 | +3.0 | 0/100 | 100th | 0.0099 |
| u_Gold (naive) | 0.0393 | +0.03 | 53/100 | 47th | 0.535 |
| **Mold step-0 (untrained)** | **0.0611** | **+3.0** | **0/100** | 100th | **0.0099** |
| Mold step-25 | 0.0617 | +3.1 | 0/100 | 100th | 0.0099 |
| v_Mold (trained) | 0.0833 | +7.3 | 0/100 | 100th | 0.0099 |
| u_Mold (naive) | 0.0486 | +0.5 | 34/100 | 66th | 0.347 |

Both untrained poles are exceeded by **zero** of 100 random directions, at the
same exact-permutation floor the trained axes reach. **The axis is already in
the verbalizable subspace before training.** What RL does is pole-asymmetric:
the distress axis roughly doubles its distance above chance (z +3.0 → +7.3),
while the flourishing axis *declines* monotonically at every checkpoint
sampled (0.0592 → 0.0591 → 0.0590 → 0.0547 → 0.0535) even as its norm doubles
(8.0 → 16.8). We therefore state the contribution as **amplification of a
pre-existing speakable component, distress pole only** — not as training
granting a voice. The *u*-at-chance result is retained but re-scoped: it shows
that speakability is not generic to valence-shaped directions, since the
maze-derived axis is speakable pre-RL while the faithful-walk axis is not —
a finding about what the extraction task selects for, not about training.
Provenance note: step-0 comes from the trajectory series
(`artifacts/traj/vectors_step*.pt`) while §4.3's vectors come from
`vectors_step95_bal.pt`; these disagree by 0.0003–0.0005 at the overlapping
step, ~6–10% of one null SD, which does not affect a ~4-SD margin
[step0_baseline.json; step0_baseline.py].

**4.5 Channel-specific causal routing (J4).** At matched norms, blind-judged:
sentiment — clean +0.25, v_full +0.44/−0.88 (gold/mold), J-component
+1.56/−1.56, residual +0.25/−0.06 (≈ baseline); backtracking markers — clean
0.0, full 7.0, J-component 12.9, residual 2.85; refusal — clean 10%, mold full
45%, residual 40% (J-component unreadable: 90% degenerate on long prompts).
Cell sizes are adequate by design: pooled within-cell SD 0.61 puts the
80%-power detectable difference at 0.61, well under the observed 1.31–1.50
J-vs-residual gaps. **Judge reliability**: an independent second blinded
judge (different model, verbatim rubric) reproduces the effect — Spearman
0.87, Krippendorff α(interval) 0.82, 94% of scores within ±1; J-component
effect vs clean = +2.00/−2.19 under the second judge vs +1.31/−1.81 under
the first [R3_second_judge.json]. **Rescaling control (pre-specified, run, passed)**: rescaling a J-component
back to ‖v‖ amplifies its surviving token vectors by 1/√var-fraction, and
randoms — having *lower* J-share — get amplified *more* (≈5.0× vs 3.5–4.3×
for the trained axes). If amplification alone drove behavior, random
J-components should therefore shift sentiment at least as much as the real
axes. They do not: 8 random directions per polarity, identically rescaled
and injected at the same layers (256 blind-judged generations, coherent
text), give cohort means +0.04 (gold layer) and +0.02 (mold layer) —
statistically at the clean baseline (+0.25) and far inside the
pre-specified "controlled" threshold (|dev| 0.21–0.23 < 0.40), versus
±1.56 for the trained axes' J-components. The channel-split result is
therefore causal, not a rescaling artifact
[j4_random_jcomp_summary.json]. Affective/expressive behavior travels
through the speakable component; refusal does not need it
[j4_rows_judged.jsonl].

**4.6 The naive control is not behaviorally flat.** Blind-judged sentiment
spans (α −4→+4, norm-matched): v_gold +3.2, faithful-u_gold +2.4, Han-style
own-u_gold +1.25; v_mold −2.1, own-u_mold −2.1 (equal!), faithful-u_mold −0.7;
randoms flat. The result survives Han et al.'s own decoding convention
(T=0.7 replication: faithful-u_gold span +2.88 — stronger, not flatter),
ruling out greedy decoding as the discrepancy source; the two u constructions
themselves correlate at only cos 0.53 (gold) / 0.39 (mold) at the treatment
layers — and just 0.14 at the layer the mold extraction itself selected —
implicating their self-avoiding-walk naive extraction (independently
flagged as flawed by the replication whose artifacts we use)
[judge_sentiment*.json; welfare-axis/own_u/compare_u.json;
R6_direction_table.json; fig_u_adjudication]. Combined with §4.1–4.4: naive directions drive behavior
but lack a speakable identity — **training's marginal contribution is
verbalization**.

**4.7 Cross-model transfer (the sprint's open question).** *"To what extent
do valence directions found in one model transfer to another?"* We injected
the 2507-derived axes into **Qwen/Qwen3-4B** and read them through *that
model's own* publicly pre-fitted Jacobian lens (Neuronpedia), which covers
all 16 workspace-band layers. Three results, in increasing order of
interest:

*The J-space share transfers almost exactly.* Read through a different
model's lens, the axes keep their speakability: v_Mold 0.079 (vs 0.083 in
2507), v_Gold 0.048 (vs 0.055), u_Gold 0.039 (vs 0.039). The
verbalizable-subspace geometry is a property of the model family, not of
our particular lens fit — which also rules out the fit as the source of
§4.3's separation.

*The magnitude inversion replicates.* On the norm-matched pole-score
estimand, Qwen3-4B reproduces §4.2's pattern: Gold **inverts** (v/u =
0.52×, vs 0.57× in 2507) while Mold separates (4.19×, vs 2.30×). An
independent model agreeing that the Gold "gap" runs backwards is strong
evidence that §4.2's correction is real rather than a quirk of one lens.

*Token-overlap magnitude: a null, caught by our own baseline discipline.*
v_Mold's k=16 J-component token sets across the two models overlap at
Jaccard 0.185, sharing *failed* and *negative* — which looks like lexicon
transfer against a random-pair baseline (0.010). But the correct null for
"same vector, two lenses" is the *same random vector* decomposed under
both models' lenses, and that null is **0.088 ± 0.055** (n = 16, max
0.231): v_Mold's 0.185 gives exact p = 0.118. **We therefore do not claim
lexicon transfer.** What remains is an exploratory content observation:
v_Mold's five shared tokens include two clean pole-relevant valence words
('failed', 'negative') alongside three fragments ('?).', '不了', '徨'),
whereas all 16 random same-vector pairs share only junk (';', 'npc',
'ise', …) and no semantic token at all. The contrast is in *whether any
valence word appears*, not in the shared set being uniformly meaningful —
noted, not built upon. (The same discipline retracted
the across-lens-target overlap in §4.3; both baselines are in
j5_paired_baseline.json / j6_paired_baseline.json.)

Behaviorally, the transferred trained axes still steer Qwen3-4B (blind
judged, 80 generations): v_Gold +0.63 and v_Mold −0.81 against clean, a
1.44-point trained span versus 0.38 for the naive pair. **Honest limit:**
trained axes exist only for 2507, so this tests transfer of the *axis and
the lens geometry* across a model family — not whether RL recruitment
reproduces, which would require redoing the RL. Within that limit the
answer to the organizers' question is: *the axis's verbalizable-subspace
occupancy, the magnitude-inversion pattern, and behavioral steering
transfer; a distinct transferred lexicon is not established*
[j6_summary.json; j6_paired_baseline.json; j6_judged_sentiment.json].

## 5. Discussion & Limitations

The families converge: maze RL neither creates a valence direction from
nothing (behavioral steerability pre-exists) nor grants it a voice (the
step-0 axis already clears the J-space null at both poles, §4.4b) — it
*amplifies* the speakable component of an already-speakable axis, and does so
only at the distress pole. For welfare monitoring this cuts several ways. A
welfare-relevant state can occupy the speakable cone without any training
aimed at making it reportable, so self-reports may carry signal earlier than
expected; but training still determines *which* states get amplified, and
some welfare-relevant directions (our faithful-walk naive axis; the refusal
drive) remain behaviorally potent and verbally invisible. And the
Gold/Mold asymmetry suggests a hypothesis worth testing beyond this model: if
training preferentially routes distress into the speakable channel while
positive states only get louder, then self-report-based welfare monitoring is
*asymmetrically sensitive* — better at catching distress than at confirming
flourishing. We state this as a hypothesis consistent with one training run,
not as established.

Limitations: single 4B model; wikitext-fit lens on chat prompts; steering at
|α|=4·‖v‖ degrades absolute token masses (differences remain directional);
component-rescaled injections inflate any direction's J-arm on first-token
readouts (random-jcomp null reported; this fragility is partly why we
deprecated the first-token report channel mid-project); the clamp arm was
invalidated (degenerate generation) and is reported as a negative; third-party
vector provenance (triangulated via norms, behavior, and our own re-extraction);
J-lens "only approximately captures" the workspace per its authors; functional
welfare only — no claims about subjective experience.

## 6. Conclusion

A model's internal welfare axis is speakable before it is trained: it sits in
the verbalizable subspace, with a specific vocabulary, ahead of the first RL
gradient. What training changes is *how loudly* — and it does so
asymmetrically, growing distress's speakable share while giving flourishing
only volume. The instruments to watch this happen (lens atlas, J-space
decomposition, trajectory tracking) are cheap, open, and transfer to any
open-weights model.

## Code and Data
This repo (scripts, raw JSON/JSONL for every number, frozen pre-registration,
append-only lab log, pinned environments, and `verify_report.py`, which
re-derives the numbers below from the results files);
anthropics/jacobian-lens (Apache-2.0); andyqhan/functional-welfare-axis
(MIT); neuronpedia/jacobian-lens and nickmahdavi/functional-welfare (HF). Our
two fitted Instruct-2507 lenses (final + penultimate targets, fp32) are too
large for the repo and are released separately (`hf-release/`); they are also
reproducible from `fit_lens.py` with the recorded seeds and prompt set.

## Appendix A — Limitations and Dual-Use / Ethical Considerations (REQUIRED)
Over-attribution: none of our measurements bear on subjective experience; we
measure functional coupling between an activation direction and output
channels. Under-attribution: a null lens readout does not show absence of
welfare-relevant state — our own naive axes drive behavior while reading out
as junk. Causal/ground-truth statement (sprint requirement): all claims rest
on activation-level manipulation with input held constant, magnitude-matched
controls, and blind judging — including an input-text "gaslight" arm to
separate internal-state effects from conversational suggestion. Distressing
outputs: Mold steering elicits distress-flavored text and pathological
self-doubt loops; generations are archived, quoted minimally. Dual-use: the
same instruments that verify a welfare state is speakable could be used to
build systems whose welfare-relevant states are deliberately kept out of the
speakable cone (unreportable by construction); we surface this explicitly as
the main misuse vector and argue for lens-based auditing as the countermeasure.

## Appendix B — Disclosure: pre-sprint work (REQUIRED)
All infrastructure, research documents, instrument validation, pre-registration
(frozen 2026-08-12, sha f66b6ea3), and the experiments reported here were
designed and executed 2026-08-11 → 2026-08-12, before the sprint window, by
the team's autonomous research agent (Claude). On 2026-08-12 an independent
audit re-analysed the raw result rows; its findings (the §4.1 control-based
null, the estimand instability disclosed in §4.2, cohort-size p-floors, and
the missing random-J-component control) were adopted in full and drive the
current structure — including the reframing of our own pre-registered
headline as a null. Work during Aug 14–16 will be
labeled in the final submission (planned: verification reruns, J5/J6
robustness, figure polish, writeup finalization by the human team).
LLM usage: experiments orchestrated, judged (blind Claude Sonnet panels), and
drafted by Claude agents; every number is machine-traceable to a results file;
final text reviewed and owned by the human team.
