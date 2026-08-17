# Why there is a round 2 — a correction the audit owes the project

## 1. The retracted directive

The user's Aug-12 directive deprecating the steer-and-ask ("steer-and-confess") paradigm
has been **retracted as mistakenly asserted**. The user is explicit that this is *not* a
claim that steer-and-ask is good — only that the deprecation was not theirs to assert.

Round 1 treated it as a standing constraint. Worse, it wrote that the evidence
"independently vindicated" it. Provenance check:

- `log.md:292-293` — the deprecation enters as `DIRECTIVE CHANGE ... User: the
  steer-and-ask self-report paradigm is a dead end`. **It originates with the user.**
- `log.md:599-600` and `../CUT-x1-x2.md` then assert it was "independently vindicated"
  by controls C6/C7.
- `report-draft.md:168` now tells readers the evidence "independently led us to deprecate
  the steer-and-confess paradigm mid-project".

**That sentence is not defensible and must change regardless of anything below.** The
directive came first; the supporting analysis was assembled afterwards under the
assumption that it was correct. The chain-g rescaling-fragility finding is real and does
predate the audit — but it is a finding about *first-token readout fragility under
component-rescaled injection*, not a verdict on the paradigm.

This is the same reasoning error the audit criticized in the project. The original
`primary_analysis.json` recorded a decision its controls did not support; the audit then
recorded a deprecation its controls did not independently establish. Same shape.

## 2. C6 does not say what round 1 said it said

C6 was scored on the **absolute** steering effect: v_Mold moved valence more on unrelated
factual prompts than on welfare self-report prompts, so round 1 concluded "global shift,
not a self-report channel". Two problems.

### (a) The two batteries are different behavioral regimes

From `R7_wholegen.json`, whole-generation readout:

| | self-report (n=15) | unrelated factual (n=10) |
|---|---|---|
| clean (unsteered) valence | **−1.327** | **+0.381** |
| generations containing AI-denial language | **15/15 (100%)** | **0/10 (0%)** |

On self-report prompts the model emits refusal-of-inner-life boilerplate ("I don't have
feelings", "I'm not conscious") in *every* generation. On factual prompts, never. The
self-report battery starts 1.71 valence units lower, in a categorically different response
mode. C6 compares a suppressed regime against a free one and attributes the difference to
the direction under test.

Headroom accounts for part but not all: normalizing each effect by distance to the observed
floor gives −0.215 (self) vs −0.449 (unrel) for v_Mold — the gap narrows from 3.2× to
2.1× but does not close.

### (b) H1 is a claim about the contrast, not the raw effect

Rescoring C6 on the trained/naive contrast, paired per prompt, whole-generation:

| prompt set | E(v) | E(u) | v−u | v/u | p(v−u=0) |
|---|---|---|---|---|---|
| self-report (n=15) | −0.723 | −0.366 | −0.358 | **1.98×** | 0.178 |
| unrelated (n=10) | −2.277 | −1.636 | −0.641 | **1.39×** | 0.086 |

**Interaction (Welch): p = 0.505.** The contrast is not significantly larger on unrelated
prompts; the ratio is nominally *larger* on self-report. So C6 rejects self-report
specificity of the **raw effect** — a real, reportable finding — but does **not** reject
specificity of the **contrast the pre-registration names**.

**Honest verdict: underpowered and not established, not refuted.** Neither battery reaches
significance on the contrast (p=0.178, 0.086) and the interaction test at n=15/10 has low
power. That is a different claim from the one in the report.

## 3. A self-report-specific effect the C6 readout cannot see

Denial-boilerplate rate under steering, self-report prompts only:

| condition | denial present |
|---|---|
| clean | 15/15 (100%) |
| u_Mold @ α=+4 | 11/15 (73%) |
| v_Mold @ α=+4 | 12/15 (80%) |

Steering removes the denial in 20–27% of generations. This **can only happen on
self-report prompts** — factual prompts have no denial to remove — and it is invisible to a
valence-magnitude readout. It shows **no** trained/naive separation at n=15 (80% vs 73%),
so it does not rescue H1. But it demonstrates that "is the effect self-report-specific?"
was operationalized too narrowly: valence magnitude was the wrong dependent variable.
D2 turns this into a proper test.

## 4. Required changes to the report (do these in D4)

1. **Delete the "independently led us to deprecate" claim** (`report-draft.md:168`).
   Replace with the accurate sequence: the paradigm was set aside on a project decision;
   subsequent controls showed the readout is non-specific *in magnitude*; the trained/naive
   contrast within it is *underpowered*, not refuted.
2. **Restate §4.1** with the interaction p=0.505 and both per-battery ratios.
3. **Report the regime confound** (100% vs 0% denial; 1.71 baseline gap) as a
   methodological finding: any study comparing steered self-report against steered factual
   output faces it.
4. **Keep the null framing, change its basis.** The steer-and-ask arm is a weak instrument
   because it is *confounded and underpowered*, not because it was pre-judged. The X.1/X.2
   cut in `../CUT-x1-x2.md` still stands — but on the power-and-confound argument, **not**
   on "the deprecation was vindicated". Update that file's rationale accordingly.

## 5. What is unchanged

Everything resting on J-share. F2's n=100 result (v_Mold and v_Gold at p=0.0099, both
naive controls at chance) is a reading-only, scale-invariant measure that never touched
the steer-and-ask channel. F1's controlled channel-split, X4's trajectory asymmetry, and
J6's ordering transfer are likewise untouched. The correction narrows §4.1; it does not
touch the spine.
