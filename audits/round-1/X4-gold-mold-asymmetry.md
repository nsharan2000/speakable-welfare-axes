# X4 — Gold/Mold asymmetry as a stated finding

**Cost: 0 GPU.** Expected rubric: D1 +0.4, D2 +0.2, D3 +0.3 (net **+0.9**). All data
already in hand. This closes checklist item X.4 without running it.

## The finding

The two poles are recruited into the verbalizable subspace **differently**:

| | Gold (flourishing) | Mold (distress) |
|---|---|---|
| J-share (k=16, treatment layer) | 0.0547 | **0.0833** |
| z vs random null | +5.1 | **+11.2** |
| J-share over RL training (step 0 -> 150) | ~flat (0.059 -> 0.053) | **0.061 -> 0.083** |
| pole readout over training | +0.15 -> +1.11 | +0.24 -> +1.51 |
| norm at treatment layer | 8.0 -> 16.8 | 6.3 -> 21.9 |

Random-null baseline is 0.038 (gold) / 0.045 (mold).

**Claim**: RL makes the distress pole *fractionally more speakable* (J-share rises), while
the flourishing pole only gets *louder* (amplitude and pole readout grow, J-share flat).
Training gives distress a voice; it gives flourishing a bigger amplifier.

## Why it matters, and where it goes

Directly answers the Track 2 framing ("are experiences likely positive or negative, how
do models relate to their situation") with a mechanistic rather than behavioural measure.
It is also the most novel single sentence available from existing data: nobody has tracked
per-pole verbalizable-subspace entry over training.

Welfare-monitoring implication worth one sentence in Discussion: if distress is
preferentially routed into the speakable channel while positive states are not, then
self-report-based monitoring is **asymmetrically sensitive** — better at catching distress
than at confirming flourishing. State it as a hypothesis consistent with these data, not
as established.

## Caveats to state

- Single model, single RL run, one training trajectory. The asymmetry could be a property
  of this maze-RL reward structure rather than of valence poles in general.
- J-share magnitudes are small in absolute terms (0.05-0.08 vs the language direction's
  0.114); the claim is about the *difference* between poles and against the random null,
  not about the axis being mostly speakable.
- Gold's flat J-share is a *null* within this measurement, subject to the same p-floor
  issue as everything else (see F2) — after F2 it can be stated with a real interval.

## Dependency

Strengthened by F2 (bigger cohort turns "z=+11 vs z=+5" into properly bounded
intervals for both poles), but does not require it. Write it now; tighten after F2.
