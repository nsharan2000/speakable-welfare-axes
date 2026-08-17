# What to send back to me — diagnostics I need to finish the analysis

Ordered by how much each changes conclusions. **If you can only do a few, do R1-R4** —
they are nearly free and each closes a real gap in the audit. Emit everything as JSON or
JSONL under `experiments/*/results/` (I read from the host repo), and mention the filename
when reporting back.

Several of these are **one added line** in a script that is already going to run.

---

## R1 — Per-prompt and per-layer raw values, not just aggregates  [free: change one write]

**Gap:** the atlas stores only aggregate pole scores per (set, concept, variant, layer). I
cannot compute a paired test, a per-layer confidence band, or a variance decomposition
from that — so every atlas claim is currently a point estimate with no interval.

**Send:** for the F3 cohort run, one row per **(direction, layer, prompt-or-position)**
before averaging, with the raw pole scores. If per-prompt is not meaningful for a
static-vector readout (it may not be), then per-layer raw values for every random
direction individually — not just the cohort mean/sd.

**Why it matters:** turns "v_mold band mean = X" into "X, 95% CI [a,b], n layers, between-
layer SD" and lets me test whether the effect is band-wide or driven by 2-3 layers. That
distinction is the difference between a robust claim and a layer-picking artifact.

---

## R2 — The exact scoring prompt and rubric given to the judges  [free: paste a string]

**Gap:** `judge_chunks/*.jsonl` contain scores but I cannot see the instruction that
produced them. Sentiment is a -5..+5 integer with a pooled within-cell SD of 0.61, which
is suspiciously tight for a free-scale judge — that smells like a coarse rubric or heavy
anchoring, and it directly determines the detectable effect size in every behavioural arm.

**Send:** the verbatim judge system prompt + user template, the model id used, temperature,
and whether the judge saw the arm label (it must not have — confirm blinding).

**Why it matters:** if judges were blinded and the rubric is coarse, the SD is real and
n=16 is genuinely adequate. If the rubric anchors on a midpoint, the SD is compressed and
every CI I computed is too narrow. This changes F1's decision thresholds.

---

## R3 — Judge reliability: rescore a subset with a second judge  [~2 min]

**Gap:** all behavioural effect sizes rest on single-judge scores with unknown reliability.
Dimension 2 asks whether results are robust; "one LLM judge said so" is the weakest link
in the causal arms.

**Send:** rescore the 144 existing sentiment rows with a second independent judge (different
model or a fresh blinded pass), emit both score columns, and report Spearman/Krippendorff
agreement.

**Why it matters:** lets me report effect sizes corrected for judge unreliability, and
tells us whether the +1.56 J-component effect survives a different scorer. Cheap insurance
on the paper's most quotable behavioural number.

---

## R4 — Clean-baseline variance across prompts  [free: already computed, just emit]

**Gap:** the clean arm has n=16 with mean +0.25, but I do not have the per-prompt spread of
the *unsteered* model across a larger prompt set. Without it I cannot say how much of the
"random directions move the readout" result (primary C1 gold null +2.72) is prompt
sensitivity versus direction sensitivity.

**Send:** clean (alpha=0) valence readout and clean judge sentiment for **all** prompts in
each battery, per prompt, no steering. This should already exist in
`primary_rows.jsonl` (`cond="clean"`) — I need the same for the J4 neutral prompts and any
prompt set F3/J6 uses.

---

## R5 — Lens fit diagnostics for both targets  [free: read from the fit]

**Gap:** `fit_meta.json` records prompts, dims, and wall time, but no fit *quality* metric.
J5's whole purpose is comparing final-target vs penultimate-target lenses, and I have no
way to say which fits better — only that they differ.

**Send:** per-layer reconstruction error (or whatever loss the jlens fit exposes),
top-k next-token agreement between lens and true model per layer, and the same for the
penultimate refit when it lands. `band_stats_*.json` has top-10 accuracy per layer — if
that is the metric, say so and I will use it.

**Why it matters:** determines whether J5's target-layer discrepancy is a real ambiguity
to report as a limitation, or one target is simply better-fitting and should be primary.

---

## R6 — Norm and cosine table for every direction actually used  [~1 min]

**Gap:** I have norms scattered across logs and one `log.md` line for cosines. For the
report's "the naive control is not well-defined" contribution I want a single table.

**Send:** for {v_gold, v_mold, u_faithful_gold, u_faithful_mold, own_u_gold, own_u_mold,
lang_fr, and the random cohort mean}: norm at the treatment layer, pairwise cosines, and
the layer used. This largely overlaps F6's output — emitting it once covers both.

---

## R7 — Whether the effect survives a non-first-token readout  [~20 min, optional]

**Gap:** the primary readout is first-answer-token log-mass, and `log.md` already flags
this channel as fragile under component-rescaled injections. C6/C7 killed the self-report
interpretation, but I cannot tell whether the *readout* or the *channel* is the problem.

**Send:** for v_mold and u_mold at alpha=+4 only, the valence readout computed over the
**full generated response** (mean log-mass across generated positions, or judge-scored
sentiment of the generation) on both the 15 self-report prompts and the 10 unrelated
prompts.

**Why it matters:** if the self-report/unrelated gap reappears with a
whole-generation readout, then C6's failure is a first-token artifact and there may be a
salvageable claim — worth knowing even though we are not building on the channel. If the
gap stays absent, C6's verdict is confirmed at the channel level and F5's framing is
final. Genuinely diagnostic either way, which is why it is here rather than in CUT.

---

## Format notes

- One JSON/JSONL per request, named after the request (`R1_atlas_raw_rows.jsonl`, etc.).
- Include the cohort size and seed scheme in every file — see `90-gotchas.md` 6 on
  `--quick` silently changing cohorts.
- Raw values before aggregation, always. I can aggregate; I cannot un-aggregate.
- If a request turns out to be meaningless for the measurement (R1's per-prompt case may
  be), say so rather than synthesising something — a stated "not applicable, because X" is
  more useful than a plausible number.
