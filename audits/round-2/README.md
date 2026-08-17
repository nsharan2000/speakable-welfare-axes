# next-directions/ — round 2 specs (post-fix, post-correction)

**Audience: the agent with compute access.** Same convention as the parent
`audit-instructions/` folder: one dispatchable file per unit, each with a cost line, the
exact grid, the file and function to touch, a pre-specified decision rule, and the failure
modes. Nothing here needs re-derivation.

**Start here: `PENDING-WORK.md`** — the single list of everything still open (this
folder's D-series, the `detailed-checklist.md` Phase-6 items, and the 🔮 dispositions),
with corrected GPU costs. **`REPORT-V2-SPEC.md`** is the last item in the queue: how to
build the submission report once the gates land.

**Then read:** `00-why-round-2.md` — a directive the first round treated as settled has
been retracted by the user, and one claim currently in `report-draft.md` is not
defensible. That correction is the reason this folder exists. Then `90-gotchas-d.md` for
the new hazards (one of them will block D3 immediately if you don't read it).

> **R8 is CLOSED — D2 and D3 are unblocked.** The 15 self-report prompts were flagged as a
> Spark-only blocker; in fact the file ships publicly in the official
> `andyqhan/functional-welfare-axis` repo (MIT). Fetched 2026-08-15 →
> **`R8_self_report_prompts.json`**, with draft D3 analogues in
> **`D3_third_person_analogues.json`**. Read `90-gotchas-d.md` §1 before using them —
> three ways to pick the wrong battery remain.

## Status of round 1

Everything in the parent folder was run or queued and `audit-response.md` answered all
seven return requests. Round 1 outcomes that these specs build on:

- **F2 done**: v_Mold and v_Gold clear a 100-direction null at **p=0.0099**; u_Mold and
  u_Gold sit at chance (34/100, 53/100). This is now the paper's spine.
- **F1 done, CONTROLLED**: random J-components +0.039/+0.016 vs clean +0.25, inside the
  pre-specified <0.40 band, against ±1.56 for the real axes.
- **F3 done, overturned its own headline**: norm-matching the control collapses Mold
  6.53× → 2.30× and **inverts Gold to 0.57×**.
- **R6 delivered the finding that drives D1/D4**: the "naive control" shares
  **cos 0.560 (Gold) / 0.675 (Mold)** with the trained axis.
- **R7 delivered the data that drives D2/D3** — and the reanalysis that overturned the
  audit's own C6 verdict (`00-why-round-2.md`).

## The specs

| # | file | GPU compute | budget wall | what it decides |
|---|---|---|---|---|
| 1 | `D4-control-alignment-finding.md` | **0** | — | reframes three negative results as one methods contribution |
| 2 | `D1-orthogonalized-control.md` | **~1.5 min** | ~5 min | whether speakable valence lives in the shared subspace or pre-exists in the naive axis |
| 3 | `D2-denial-breaking-readout.md` | ~13 min | ~25 min | whether the self-report channel has any trained-specific content on a fair DV |
| 4 | `D3-prompt-matched-c6.md` | ~5 min | ~15 min | the C6 the pre-registration should have specified |
| 5 | `D5-finish-f7.md` | in flight | — | bounds the k-robustness claim |

**Whole D-series: ~20 GPU-min of compute, ~45 min wall.** (Plus ~7 min for the in-window verification chain, item 6.5 — see `../PENDING-WORK.md`.) "GPU compute" is the itemized
grid at the unit rates below and is traceable in each spec; "budget wall" adds model/lens
load, judge round-trips, longer generations (D2 uses 120-token gens, not the 60 the rate
was measured at), and D3's prompt-revision rounds. Every number in each spec's header is
itemized against its own grid — if a header and its grid disagree, trust the grid.

**Do D4 first** — zero GPU, and it fixes report text that is currently wrong.
Then D1 (cheapest measurement, sharpest inference). D2 and D3 are the two that
re-open the steer-and-ask question honestly; they are independent and can run in
either order.

## What NOT to run

- **More J-share cohort.** n=100 already gives p=0.0099; n=1000 gives 0.001 and changes
  no conclusion.
- **RL replication on Qwen3-4B** to close J6's trained-axis gap. Correct experiment,
  far outside any sprint runway.
- **X.1 as originally specified** (`../CUT-x1-x2.md`). Superseded by D2, which asks the
  same question with a dependent variable that can answer it. Note the *grounds* for
  the cut have changed — see `00-why-round-2.md` §4.

## Cost basis

Generation+judge 0.9 s/gen; atlas row 1.67 s; lens fit 99.9 s/prompt — all measured and
unchanged.

**Corrected:** decomposition is **1.41 s/direction** when batched (F2: 200 decomps in
4.7 min, `log.md:968`), not the 14.3 s/direction quoted in round 1, which timed a
single-direction path. Round-1 estimates for decomposition-bound items were ~10× too slow
(F2 quoted 48 min, ran 4.7; F7 quoted 12 min, ran 4.3). Generation-bound estimates were
correct.

## Provenance of every number in this folder

All values were recomputed from the repo's raw result files on CPU
(`R7_wholegen.json`, `jshare_cohort_n100.json`, `R6_direction_table.json`,
`j6_paired_baseline.json`). **No GPU work was done by the auditor** — no compute target
is configured on that side. Treat these as re-analysis of your saved rows: if a row is
wrong, the conclusion inherits the error.
