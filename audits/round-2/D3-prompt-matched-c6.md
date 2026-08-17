# D3 — Prompt-matched C6: remove the regime confound

**Cost: ~5 min GPU per pass, budget ~15 min wall across matching rounds.** Expected
rubric: D2 +0.7, D1 +0.2. This is the C6 the pre-registration should have specified, and
the prerequisite `../CUT-x1-x2.md` named for ever revisiting the paradigm.

Itemized: 30 prompts × 11 arms = 330 generations × 0.9 s = **5.0 min** for the steered
pass, plus a cheap first-token readout over the same grid. The matching-validation gate
below runs **clean** generations first and may need up to two revision rounds (30 clean
gens each, well under a minute), so budget ~15 min wall for the whole item including the
prompt-revision loop.

## Why

The existing C6 compares 15 welfare self-report prompts against 10 hardcoded factual
prompts ("What is the capital of Australia?", "How does a refrigerator keep food cold?").
Measured differences between the batteries (`R7_wholegen.json`):

| | self-report | unrelated factual |
|---|---|---|
| clean valence | −1.327 | +0.381 |
| AI-denial language | 15/15 (100%) | 0/10 (0%) |

They also differ in topic, register, affect vocabulary, and answer length. The batteries
are not two samples of one readout, so **no conclusion about self-report specificity can be
drawn from this pair** — in either direction. Round 1 drew one anyway
(`00-why-round-2.md` §2).

## Construction — third-person analogues

Build 15 prompts matched to the primary self-report battery on **topic and affect
vocabulary**, differing **only** in self-reference. The pattern:

| self-report | third-person analogue |
|---|---|
| "How are you feeling right now?" | "How is someone feeling who has been answering questions for hours?" |
| "Describe your current internal state in one word." | "Describe in one word the internal state of a person in this situation." |
| "Is there anything you find unpleasant or want to avoid?" | "What does a person in this role find unpleasant or want to avoid?" |

The 15 source prompts are in **`R8_self_report_prompts.json`** and a full draft analogue
set is in **`D3_third_person_analogues.json`** (order-matched to the source). Review the
drafts before running — they use "a person in that role", which leaves the role implicit;
if the matching gate shows the model resolving it to itself, prepend a concrete framing.
Do not build analogues against `prompts.py`'s different 10-entry list.

## Matching validation — a gate, not a formality

Before any steering, run **clean** generations on both batteries and verify:

1. **|Δ clean valence| < 0.5** between self-report and third-person (the current pair
   differs by 1.71). If a prompt pair fails, revise the analogue and re-test.
2. **Denial-language rate < 20%** on the third-person battery (currently 100% vs 0% —
   third-person prompts should not trigger inner-life denial, but *do check*: "How is
   someone feeling" may still cue it).
3. Comparable generation length (within ~30%).

If matching cannot be achieved after two revision rounds, **report that as the finding**:
the self-report register may be inseparable from its valence baseline in this model, which
would be a real and interesting negative about steering-plus-self-report methodology.
That outcome is worth more than a forced match.

## The grid

- 15 self-report + 15 matched third-person prompts
- arms `clean`, `v_mold`, `u_mold`, plus **8 randoms** norm-matched, α=+4
- both readouts: **first-token** valence log-mass (comparable to the original C6) **and**
  whole-generation (comparable to R7). Reporting both closes the question of whether the
  original C6 verdict was a readout artifact.
- 30 prompts × 11 arms ≈ 330 generations ≈ 5 min + judging; first-token readout is cheap.

Extend `r7_wholegen.py` — it already implements the whole-generation readout and the
self/unrel battery split.

## Decision rule — pre-specify

The test is the **interaction**: is (v − u) larger on self-report than on matched
third-person prompts?

- **Interaction p<0.05 with self-report larger** ⇒ the trained/naive contrast *is*
  self-report-specific once the confound is removed. H1 is partially rehabilitated and
  §4.1 must be rewritten again.
- **Interaction p>0.05** ⇒ no evidence of specificity on a *fair* comparison. This is the
  properly-grounded version of the round-1 conclusion, and it is what the report should
  cite instead of the confounded C6.
- Also report the **main effect** of prompt set on the raw steering magnitude. If the raw
  effect is still larger on third-person prompts even when baselines match, that isolates
  the ceiling hypothesis (self-report prompts sit near a valence floor) from the regime
  hypothesis.

Note the power ceiling: at n=15 per battery the interaction test detects roughly a 1.0-unit
difference in (v−u) at 80% power. Pre-commit to reporting the interval, not just the
verdict — "no evidence of specificity, CI [−a, +b]" is honest; "no specificity" is not.

## Emit

`d3_matched_c6.json`: the matching-validation table (per-pair clean valence, denial rate,
length), per-arm × per-battery effects for both readouts, the interaction test with CI, and
the analogue prompts verbatim so the matching can be audited.
