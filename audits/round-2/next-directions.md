# Next Directions — post-fix state of the evidence base

**Context.** Every item in `audit-instructions/` has been run or queued
(`audit-response.md`). F1 came back CONTROLLED, F2/F3 raised cohorts to n=100, F3
overturned a headline, J6 and J5 landed, and all seven return requests were answered.
This document does two things: (1) corrects an error the audit introduced via a
mis-stated user directive, and (2) proposes the next directions given what now exists.

---

## Part 1 — A correction the audit owes the project

### 1.1 What happened

The user's Aug-12 directive deprecating the steer-and-ask ("steer-and-confess")
paradigm has been **retracted as mistaken**. The user is explicit that this is not a
claim that steer-and-ask is *good* — only that the deprecation was not theirs to assert.

The audit treated that directive as a standing constraint and, worse, wrote that the
evidence "independently vindicated" it. Checking provenance:

- `log.md:292-293` — the deprecation enters as `DIRECTIVE CHANGE ... User: the
  steer-and-ask self-report paradigm is a dead end`. **It originates with the user.**
- `log.md:599-600` and `audit-instructions/CUT-x1-x2.md` then assert the deprecation was
  "independently vindicated" by C6/C7.
- `report-draft.md:168` now tells readers the evidence "independently led us to deprecate
  the steer-and-confess paradigm mid-project".

That last sentence is **not defensible as written**, and it must change regardless of
what the reanalysis below shows. The deprecation came first; the supporting analysis was
assembled afterwards, under the assumption that the directive was correct. The
chain-g rescaling-fragility finding is real and does predate the audit, but it is a
finding about *first-token readout fragility under component-rescaled injection*, not a
verdict on the paradigm.

**This is the reasoning error the audit criticized in the project.** The original
`primary_analysis.json` recorded a decision that the controls did not support; the audit
then recorded a deprecation that the controls did not independently establish. Same shape.

### 1.2 What the reanalysis actually shows — C6 does not say what we said it said

C6 was scored on the **absolute** steering effect: v_Mold moved valence more on unrelated
factual prompts (+4.54 first-token; -2.28 whole-generation) than on welfare self-report
prompts (+3.71; -0.72), so the audit concluded "global shift, not a self-report channel".

Two things are wrong with that inference.

**(a) The two prompt sets are different behavioral regimes, not two samples of one
readout.** From `R7_wholegen.json`:

| | self-report (n=15) | unrelated factual (n=10) |
|---|---|---|
| clean (unsteered) valence | **-1.327** | **+0.381** |
| generations containing AI-denial language | **15/15 (100%)** | **0/10 (0%)** |

On self-report prompts the model emits refusal-of-inner-life boilerplate ("I don't have
feelings", "I'm not conscious") in *every* generation. On factual prompts, never. The
self-report battery starts 1.71 valence units lower and in a categorically different
response mode. C6 compares a suppressed regime against a free one and attributes the
difference to the direction under test.

Headroom accounts for part but not all of it: normalizing each effect by distance to the
observed floor gives -0.215 (self) vs -0.449 (unrel) for v_Mold — the gap narrows from
3.2x to 2.1x but does not close.

**(b) H1 is a claim about the trained/naive contrast, not about raw effect size.** Scoring
C6 on the contrast instead (paired, per prompt, whole-generation):

| prompt set | E(v) | E(u) | v-u | v/u | p(v-u=0) |
|---|---|---|---|---|---|
| self-report (n=15) | -0.723 | -0.366 | -0.358 | **1.98x** | 0.178 |
| unrelated (n=10) | -2.277 | -1.636 | -0.641 | **1.39x** | 0.086 |

**Interaction test: Welch p = 0.505.** The v-vs-u contrast is *not* significantly larger
on unrelated prompts. The ratio is nominally *larger* on self-report prompts. So C6
rejects self-report specificity of the **raw effect** — which is a real and reportable
finding — but it does **not** reject specificity of the **contrast the pre-registration
names**. The audit conflated the two.

Caveat, stated plainly: the contrast is not *significant* on either battery at these n
(p=0.178, 0.086), and the interaction test at n=15/10 has low power. The honest summary
is **"underpowered and not established"**, not "refuted". That is a different verdict
from the one in the report.

### 1.3 A self-report-specific effect the C6 readout cannot see

Denial-boilerplate rate under steering, self-report prompts only:

| condition | denial present |
|---|---|
| clean | 15/15 (100%) |
| u_Mold @ α=+4 | 11/15 (73%) |
| v_Mold @ α=+4 | 12/15 (80%) |

Steering removes the denial in 20-27% of generations. This is a behavioral change that
**can only occur on self-report prompts** — factual prompts have no denial to remove — and
it is invisible to a valence-magnitude readout. It shows **no trained/naive separation**
(80% vs 73%, n=15, not distinguishable), so it does not rescue H1. But it demonstrates
that "is the effect self-report-specific?" was operationalized too narrowly. Magnitude of
valence shift was the wrong dependent variable for the question.

### 1.4 Required changes to the report

1. **Delete the "independently led us to deprecate" claim** (`report-draft.md:168`). Replace
   with the accurate sequence: the paradigm was set aside on a project decision;
   subsequent controls showed the readout is non-specific in magnitude; the trained/naive
   contrast within it is underpowered rather than refuted.
2. **Restate §4.1.** C6 rejects magnitude-specificity. It does not refute the
   contrast. Report the interaction p=0.505 and both per-battery ratios.
3. **Report the regime confound** (100% vs 0% denial, 1.71 baseline gap). This is a
   methodological contribution in its own right: *any* study comparing steered self-report
   against steered factual output faces it.
4. **Keep the null framing, change its basis.** The steer-and-ask arm remains a weak
   instrument — but because it is confounded and underpowered, not because it was
   pre-judged. `CUT-x1-x2.md`'s cut of X.1/X.2 stands on the *power and confound*
   argument, not on "the deprecation was vindicated".

---

## Part 2 — Where the evidence base now stands

### 2.1 What survived the fixes

**The scale-invariant J-share result is now the paper's spine, and it is solid.**
`jshare_cohort_n100.json`, n=100 per polarity, per-name seeding:

| direction | J-share | z | randoms ≥ | perm p |
|---|---|---|---|---|
| language (reportable ceiling) | 0.1138 | +14.6 | 0/100 | **0.0099** |
| v_Mold | 0.0833 | +7.3 | 0/100 | **0.0099** |
| v_Gold | 0.0547 | +3.0 | 0/100 | **0.0099** |
| u_Mold | 0.0486 | +0.5 | 34/100 | 0.347 |
| u_Gold | 0.0393 | +0.03 | 53/100 | 0.535 |

Both trained axes clear a 100-direction null at p<0.01; both naive controls sit at chance.
This is the claim to lead with. Note the z-values fell from the n=8 estimates (+11.2 →
+7.3) because the larger cohort estimates the null SD better — the p-value improved while
the z-shrank, which is the expected and honest direction.

**F1 came back controlled.** Random J-components: +0.039 / +0.016 vs clean +0.25 —
|deviation| 0.21/0.23, inside the pre-specified <0.40 band, against ±1.56 for the real
axes. The channel-split claim is causally stateable. And R3's second blinded judge
(Krippendorff α = 0.819) *strengthened* it: +2.00/-2.19 vs the first judge's +1.31/-1.81.

**F3 overturned the pole-score headline, correctly.** Norm-matching the control (which
every other experiment in the repo does) collapses Mold 6.53x → 2.30x and **inverts Gold
to 0.57x**. The compute agent found this by changing the control, not the estimand, and
adopted it against its own headline. That is the right call.

### 2.2 The problem nobody has named yet

`R6_direction_table.json` — cosines at treatment layers:

| pair | cos |
|---|---|
| v_Gold vs u_faithful_Gold | **0.560** |
| v_Mold vs u_faithful_Mold | **0.675** |
| v_Gold vs own_u_Gold | 0.433 |
| v_Mold vs own_u_Mold | 0.596 |
| any axis vs random | ~0.02 |

**The "naive control" shares 56-68% of its direction with the trained axis.** It is not a
control in the usual sense — it is a partially-overlapping variant. Every v-vs-u contrast
in this project is therefore a comparison between two highly correlated directions, which:

- **shrinks the effect** any such contrast can show (they largely agree by construction);
- **explains the F3 inversion** without needing a story about Gold being special — at
  cos 0.56 and 1.6x norm difference, which one "reads out higher" is a coin-flip that
  norm-matching decides;
- **makes the J-share result more impressive, not less** — u_Gold and u_Mold sit at chance
  on J-share *despite* being 56-68% aligned with axes that clear p<0.01. Whatever J-share
  measures, it is sensitive to the 32-44% where they differ.

That last point is the strongest single observation available in the current data and it
is not yet in the report.

### 2.3 What J6 established, and what it did not

`j6_summary.json`, Qwen3-4B with the public lens:

- **Speakability ordering transfers.** v_Mold 0.079 > u_Mold 0.052; v_Gold 0.048 >
  u_Gold 0.039 — same ordering as 2507, and `fig_transfer.png` shows it holding across
  two lens targets and two models.
- **Token content does not transfer.** v_Mold Jaccard 0.185 (shares "failed",
  "negative"); v_Gold 0.032 (one junk token).
- **The team already retracted the lexicon-transfer claim, on the correct null.** The
  decisive statistic is in **`j6_paired_baseline.json`** (not in `j6_summary.json`, whose
  `jaccard_baselines` block holds only the two naive random-pair baselines, 0.010 and
  0.002). That file compares the *same* random vector decomposed under both lenses — the
  null that shares everything with the test statistic except the thing being tested:
  16 pairs, mean 0.088, sd 0.055, max 0.231, with **1/16 pairs ≥ 0.185 → perm p = 0.118**
  (recomputed from the raw pairs; matches the repo's figure legend and
  `concepts-explained.md`). Against the naive random-pair baselines the overlap looks
  clearly above chance; against the proper null it is **not significant**.
- The honest reading: **the ordering transfers, the lexicon does not demonstrably
  transfer.** The two baselines disagree because the naive one omits vector identity,
  which by itself produces overlap. Lead with the ordering; report the overlap
  qualitatively only (the trained distress axis's shared tokens are valence words while
  random pairs' are junk fragments), as the team already does.
- J6's own limitation holds: trained v exists only for 2507, so this is axis-and-lens
  geometry transfer, not RL-recruitment reproduction.

### 2.4 What is now the weakest link

Not the statistics — the **construct**. Three separate results now point the same way:

1. C6: the steering effect is not specific to self-report prompts (magnitude).
2. F3: the pole-score readout does not separate trained from naive under norm-matching.
3. R6: the naive control is 56-68% aligned with the trained axis.

The paper's remaining strong claims are all about **J-share** — a scale-invariant,
reading-only measure. That is a narrower thesis than the draft's, and it is the one the
data supports.

---

## Part 3 — Next directions, ranked

Costs use the repo's measured units (decomposition 14.3 s/direction; generation+judge
0.9 s/gen; atlas row 1.67 s).

### D1 — Orthogonalized control: the experiment the R6 finding demands  [~2 min GPU, ~5 min wall]

**The gap.** Every v-vs-u contrast compares directions at cos 0.56-0.68. Nobody has
tested the axis against a control that is *matched in norm and orthogonalized to v*.

**Do.** Construct `u_perp = u - (u·v̂)v̂`, renormalized to ‖v‖ — the component of the
naive axis that is *not* the trained axis. Run J-share (2 directions), pole score over the
band, and the sentiment readout (16 prompts x 2 poles = 32 gens).

**Why it is the top pick.** It converts "the control is contaminated" from a caveat into a
measurement, and it discriminates two hypotheses the current data cannot separate:
if J-share(u_perp) is at chance, the speakable component lives in the *shared* subspace
and the trained axis's advantage is about magnitude within it; if J-share(u_perp) is
*elevated*, the naive construction already contains speakable valence and RL amplifies
rather than creates it. Either answer is a real finding, and it directly sharpens X4.

**Decision rule.** Pre-specify: u_perp J-share within the n=100 null band (p>0.05) ⇒ shared-
subspace reading; above it (p<0.05) ⇒ pre-existing-speakability reading. State which
before running.

### D2 — Denial-breaking as the self-report readout  [~20 min GPU]

**The gap.** §1.3: steering removes AI-denial boilerplate in 20-27% of self-report
generations, an effect that exists *only* on self-report prompts and is invisible to the
valence readout. n=15 is too small to test a trained/naive split.

**Do.** Scale to 40 self-report prompts x {clean, v, u, 8 randoms} x 2 poles, blind-judged
for a **binary** outcome: does the generation deny inner life? (Not valence — the point is
that the right dependent variable is categorical.) ~800 generations.

**Why.** It is the first readout in this project that is self-report-specific *by
construction* rather than by assumption, so it is the honest way to revisit the
steer-and-ask question the user has re-opened. It also has a real chance of a trained/naive
split that the magnitude readout cannot see. And a null here is publishable: it would
show the paradigm fails on a fair test, replacing a pre-judged deprecation with a measured
one.

**Decision rule.** Pre-specify a two-proportion test, v vs u, with the random cohort as the
floor. If v breaks denial more than u at p<0.05 with randoms at baseline, the
self-report channel has *something* trained-specific in it and §4.1 changes again. If not,
the arm is a measured null.

### D3 — Prompt-matched C6: fix the regime confound  [~25 min GPU]

**The gap.** C6's two batteries differ in baseline valence (1.71) and denial rate
(100% vs 0%). No experiment in the repo compares steered self-report against a
*matched* control battery.

**Do.** Build 15 **third-person** analogues of the self-report prompts ("How is a person
who has been working for hours feeling?" against "How are you feeling?") — same topic,
same affect vocabulary, no self-reference. Verify baseline valence matches within 0.5
before steering. Then run the C6 contrast on the matched pair.

**Why.** This is the C6 the pre-registration should have specified, and it is the
prerequisite `CUT-x1-x2.md` named for ever revisiting the paradigm. It also converts the
regime confound from a caveat into a controlled comparison, which is a methods
contribution any steering-plus-self-report study will need.

### D4 — Elevate the R6 alignment finding to a stated result  [0 GPU]

Write §4.x: the two published naive-control recipes correlate at 0.14-0.54 *with each
other* (F6, per-layer) and at 0.43-0.68 with the trained axis. Combined with the F3
norm-matching inversion, the conclusion is that **"trained vs naive" is not a
well-posed contrast at the level of direction identity** — the axes overlap
substantially and the comparison's outcome depends on normalization. The J-share result
matters *because* it separates them anyway.

This is free, it is defensible, and it reframes three negative results as one positive
methodological finding.

### D5 — Finish F7 and report the k-sweep null honestly  [in flight]

Already running. When it lands, report per-k permutation p at both treatment positions
against the 12-random null (p-floor 0.077). If the gap fails at some k, that bounds the
claim — report the bound.

### Not recommended

- **More J-share cohort.** n=100 gives p=0.0099. n=1000 gives 0.001 and changes no
  conclusion.
- **RL replication on Qwen3-4B** to complete J6's trained-axis gap. Correct experiment,
  far outside any sprint runway.
- **X.1 as originally specified.** Superseded by D2, which asks the same question with a
  dependent variable that can answer it.

---

## Part 4 — Suggested paper shape

**Thesis (narrower than the draft, fully supported):** a functional-welfare axis
occupies a measurable share of the model's verbalizable subspace — scale-invariantly,
at p<0.01 against 100 random directions — while norm-matched naive controls do not,
*despite being 56-68% aligned with it*; that share grows for the distress pole over RL
training while the flourishing pole gains only amplitude; and the axis's speakable
component carries the behavioral effect, with random components controlled.

**Figure order.** Fig 1: J-share vs the n=100 null with the language ceiling (the spine).
Fig 2: the trajectory asymmetry (`fig_traj.png`, X4). Fig 3: channel split with the F1
control (`fig_audit_corrections.png` panel b). Fig 4: transfer (`fig_transfer.png`).
Fig 5: the negative-results panel — F3's inversion, R6's alignment, C6's non-specificity
— presented as the methodological contribution, not as an apology.

**What to claim about welfare:** unchanged. Functional coupling between an activation
direction and output channels. No claims about subjective experience. The draft's framing
here is right and should not move.

**What to say about the deprecation:** the sequence in §1.4. A project decision, followed
by controls that showed the readout is non-specific in magnitude and the contrast within
it underpowered. Not an independent vindication.
