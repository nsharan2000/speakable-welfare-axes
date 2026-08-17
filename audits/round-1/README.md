# audit-instructions/ — chainable experiment specs from the independent audit

**Audience: the agent with compute access.** Each `F*/J*` file is one self-contained,
dispatchable unit: why it exists, the exact grid, the file and function to touch,
the decision rule, and the failure modes. Nothing here needs re-derivation.

**Read first, always:** `00-verdicts.md` (what the audit found, which claims survive)
and `90-gotchas.md` (repo-wide hazards — one of them silently corrupts results).

## Priority order (rubric gain first, compute second)

| # | file | GPU | why it's here |
|---|---|---|---|
| 1 | `F5-primary-reframing.md` | **0** | the pre-registered claim fails its own controls C1/C6/C7 |
| 2 | `F4-estimand-prespecification.md` | **0** | headline ratio is not a fixed quantity; two estimands in play |
| 3 | `X4-gold-mold-asymmetry.md` | **0** | a finding already in the data, unwritten |
| 4 | `F1-random-jcomp-control.md` | **~1 min** | gates whether J4's causal claim is stateable at all |
| 5 | `F6-compare-u.md` | ~2 min | report cites a file that does not exist |
| 6 | `F7-ksweep-mold-layer.md` | ~12 min | mold k-robustness asserted at layers excluding mold's own |
| 7 | `F2-jshare-cohort.md` | ~48 min | hardens the thesis-bearing claim from p=0.111 to p=0.0099 |
| 8 | `F3-atlas-cohort.md` | ~89 min | same for the atlas claim; run **after** F4 fixes the estimand |
| 9 | `J6-cross-model.md` | ~150 min | only item that materially moves Impact/Innovation |

Items 1–8 total **152 GPU-min**. Do 1–3 regardless of GPU state; they are analysis and
writing. **`CUT-x1-x2.md` explains what not to run and why** — read it before reviving
anything from the checklist's optional list.

## Two hard rules

1. **Do not edit `pre-registration.md`.** It is frozen and its value is that it was
   frozen. Deviations are reported *as deviations* in the report, never by amending it.
2. **Do not delete or overwrite existing results.** Every fix here appends or writes to
   a new filename. `90-gotchas.md` §1 documents a case where re-running an existing
   script *will* silently desynchronise two result files.

## Where things stand — START AT `next-directions/PENDING-WORK.md`

**Everything still open lives in `next-directions/`.** This parent folder is round 1 and is
fully executed (see `../audit-response.md`) — it is history, kept for provenance.

`next-directions/PENDING-WORK.md` consolidates every open item from all three sources
(this folder, `next-directions/`, and `detailed-checklist.md` Phase 6 + the 🔮 list), with
corrected costs. `next-directions/REPORT-V2-SPEC.md` is the terminal item: how to build the
submission report once the gates land. Two Phase-6 items were missing from earlier rounds and both matter —
**6.5 in-window verification** (guards the disqualification risk on undisclosed prior work)
and **6.6 HF release** (the report promises artifacts that may not exist).

## Round 2 — `next-directions/`

After round 1 was run, the user **retracted** the Aug-12 steer-and-ask deprecation as
mistakenly asserted. That invalidates one claim now sitting in `report-draft.md:168`
("the evidence independently led us to deprecate…") and required re-reading control C6.
The follow-on specs live in **`next-directions/`** — start at its `README.md`, then
`00-why-round-2.md`.

Ordering across the two folders: **`next-directions/D4` outranks everything left here**,
because part of it is a correction rather than an experiment. F7 (= `D5`) is already in
flight. Round-1 items marked ✅ in `../audit-response.md` need no further action.
R8 (the self-report prompt battery) is **closed** — the file turned out to be public in the
official upstream repo, so D2 and D3 are unblocked; the prompts are in
`next-directions/R8_self_report_prompts.json`.

## Requesting more from me

`99-return-requests.md` lists the diagnostics I need back to finish the analysis, in
priority order, with exact emit formats. Several are one added line in a script that is
already running. If you can only do a few, do R1–R4. Round-2 requests (R8–R12) are in
`next-directions/99-return-requests-d.md`.

## Cost basis

All costs measured from this repo's own logs, not estimated: lens fit 99.9 s/prompt
(`fit_meta.json`); decomposition 14.3 s/direction (chain-g); steering readout cell
10.0 s (chain-h); generation+judge 0.9 s/gen; atlas row 1.67 s. The checklist's ETAs ran
2–6× long against actuals, so plan on these numbers.
