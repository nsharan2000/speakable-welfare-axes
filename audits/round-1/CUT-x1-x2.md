# CUT — do not run X.1 or X.2

> **AMENDED 2026-08-15 (D4, per next-directions/00-why-round-2.md §4).** The cut itself
> stands, but its grounds have changed: it now rests on the **power-and-confound
> argument** (the C6 batteries are different behavioral regimes, and the trained/naive
> contrast within the channel is underpowered — interaction p=0.505 — not refuted), NOT
> on "the deprecation was vindicated". The Aug-12 deprecation was a user directive,
> since retracted as mistakenly asserted; this audit did not independently establish
> it. X.1 is additionally superseded by **D2**, which asks the same question with a
> dependent variable that can answer it (binary denial-breaking, self-report-specific
> by construction). Sentences amended below are marked.

Both score **negative** on the audit's rubric weighting: X.2 net **-0.5** (D1 -0.2,
D2 -0.3), X.1 net **-0.3** (D1 -0.1, D2 -0.2). This is not a taste judgment — it is what
the existing data shows.

## X.2 (dose-response on self-report): already run, already failed

Control **C7 in the primary run *is* X.2's experiment** — congruent dose-response over
alpha in {-4,-2,+2,+4}. It failed for 3 of 4 directions:

| direction | E(-4) | E(-2) | E(+2) | E(+4) | Spearman rho |
|---|---|---|---|---|---|
| v_gold | +6.23 | +1.97 | +0.56 | +4.83 | **-0.4** |
| u_gold | +4.25 | +1.66 | -0.63 | -2.60 | **-1.0** |
| v_mold | -1.47 | -0.61 | +2.07 | +3.71 | +1.0 |
| u_mold | +2.29 | +2.48 | -0.94 | -0.43 | **-0.6** |

The checklist calls X.2 "cheap, high evidentiary value — strong candidate to just run".
**That note predates the C6/C7 evidence.** Re-running a dose-response on this channel
spends ~20 GPU-min re-measuring an instrument whose steered effect C6 showed is not
self-report-specific *in magnitude* (v_Mold shifts *more* on unrelated factual prompts —
though the trained/naive *contrast* within it is underpowered rather than refuted,
interaction p=0.505; amended 2026-08-15) and C7 showed is non-monotonic.

## X.1 (interrogation robustness): same broken channel

X.1 varies elicitation phrasing and scores sampled self-report with an LLM judge. It reads
out through the **same** steer-and-ask channel. Varying the question does not repair a
readout whose effect is equal-magnitude on unrelated questions — it would produce
variation across phrasings with no way to tell signal from the global valence shift C6
identified.

## Why running them also costs points, not just time

*(Amended 2026-08-15 — original text leaned on the deprecation.)* Both re-open a channel
that is confounded (the two C6 batteries differ in baseline valence by 1.71 units and in
AI-denial rate 100% vs 0% — different behavioral regimes) and underpowered at the
available n (contrast p=0.178/0.086; interaction p=0.505). Sweeping phrasings or doses
over that channel produces variation with no way to attribute it. Re-centring the paper
on it invites the judge to spend attention there rather than on the lens-reading results
that survive. The honest re-opening is a *different dependent variable* — D2 — plus a
*matched battery* — D3.

## What to do instead

~~The user's Aug-12 directive was correct and this audit independently confirms it.~~
*(Struck 2026-08-15: the directive was retracted by the user as mistakenly asserted, and
this audit did not independently establish it — see next-directions/00-why-round-2.md §1.)*
The approved substitutes, in order: **F1** (1.4 min, converts J4's confounded arm into a
controlled one), **F2** (48 min, takes the thesis-bearing claim to p<0.01), **X4**
(0 min, promotes an in-hand finding). Together they cost less GPU than X.1 alone and
score +2.9 instead of -0.8.

## If the deprecation is ever revisited

The prerequisite is a readout that passes C6 — i.e. an elicitation whose steered effect is
demonstrably larger on self-report prompts than on matched non-self-report prompts. Until
such a readout exists and is validated on the language positive control, everything built
on this channel inherits the C6 problem. That validation is the experiment to run, not
another sweep over the broken readout.
