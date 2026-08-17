# PENDING-WORK — everything still open, from all sources

Consolidates `detailed-checklist.md` (Phase 6 + Phase 9 + the 🔮 list),
`./` (the D-series in this folder), and `audit-response.md`. Read with
`REPORT-V2-SPEC.md`, which is the terminal item.

**Status at 2026-08-15.** Round 1 is fully landed — every file it promised exists on disk
(`jshare_cohort_n100.json`, `atlas_cohort_n100.json`, `atlas_primary_estimand.json`,
`ksweep_ext.json`, `j6_summary.json`, `j5_lens_comparison.json`, `compare_u.json`,
`j4_random_jcomp_summary.json`, R1–R7). No D-series output exists yet.

---

## ⚠ Cost-model correction — my earlier GPU estimates were ~10× too slow

The unit rate used for decompositions (14.3 s/direction, from chain-g) came from a
**single-direction** gradient-pursuit path. `f2_jshare_cohort.py` **batches**, and the
measured F2 run was **4.7 min for 200 decompositions = 1.41 s/direction** (`log.md:968`).
F7 landed in **4.3 min** against my 12-min estimate.

| item | I quoted | actual | over |
|---|---|---|---|
| F2 (200 decomps) | 48 min | **4.7 min** | 10.2× |
| F7 (ksweep_ext) | 12 min | **4.3 min** | 2.8× |

Generation-bound work (0.9 s/gen) was measured correctly and is unchanged. Re-costed:

| item | GPU compute |
|---|---|
| D1 orthogonalized control | **~1.4 min** (2 decomps at the batched rate + 32 atlas rows + 32 gens) |
| D2 denial-breaking | ~13 min (880 gens — generation-bound, unchanged) |
| D3 prompt-matched C6 | ~5 min (330 gens, unchanged) |
| chain L in-window verify | ~7 min |
| **all remaining compute** | **~27 GPU-min** |

Everything outstanding fits in half an hour of GPU. The binding constraint is human time
and the Sunday 11:59pm AoE cutoff, not compute.

---

## A. Blocker — CLEARED 2026-08-15

**R8 (the 15 self-report prompts) is resolved.** It was recorded as Spark-only; the file is
in fact public in the official `andyqhan/functional-welfare-axis` repo (MIT,
`datasets/concept_vector_eval_prompts.json`). Fetched and saved as
`R8_self_report_prompts.json`; draft D3 analogues in `D3_third_person_analogues.json`.
R9 is closed too — the battery is verified identical across `R7_wholegen.json` and
`primary_rows.jsonl`, so the C6 interaction test (p=0.505) paired the right rows.

**Nothing blocks the queue. Every item below can start immediately.** Read
`90-gotchas-d.md` §1 first — three ways to pick the wrong prompt battery remain.

## B. Compute queue, in order

| # | item | GPU | gate |
|---|---|---|---|
| 1 | **D4** report corrections | 0 | none — text fixes that are currently *wrong* |
| 2 | **6.5 chain L** in-window verification | ~7 min | none — see §C, highest non-zero priority |
| 3 | **D1** orthogonalized control | ~1.4 min | none |
| 4 | **D3** prompt-matched C6 | ~5 min | none (R8 closed) |
| 5 | **D2** denial-breaking readout | ~13 min | none (R8 closed) |
| 6 | **REPORT-V2** | 0 GPU, ~2 h human | all of the above |

D5/F7 is **done** on the compute side (`ksweep_ext.json`, 4.3 min, Aug 12); only the
write-up of its per-k p-values against the 12-random null remains, and that folds into
the report.

---

## C. Phase 6 items I had not previously accounted for

**6.5 In-window verification (`chain_l_verify.sh`) — ~10 min GPU, and it is the highest
non-zero-cost priority.** Seed-identical reruns of F2 and F3 into `*_verify_$STAMP` files,
then a diff against the pre-sprint originals, emitting `L_VERIFY_PASS/FAIL`.

Why it outranks the science: the sprint's pre-work policy says **undisclosed prior work can
lead to disqualification**, and the bulk of this project predates Aug 14. This chain is what
lets Appendix B state that the headline numbers were re-verified *inside* the window. It is
the cheapest insurance available on the single largest non-scientific risk to the
submission. Run: `bash dispatch_dm.sh chain_l_verify.sh chain-l` on the Spark.

The script backs up originals before rerunning and restores them after — read it before
running, and confirm the `.bak` restores completed, or the pre-sprint files it is
diffing against will have been clobbered.

**6.3 Video demo (optional, ~1 h human).** Script ready at `report/video-script.md`, but
it was written against v1 and repeats the retracted deprecation claim and the pre-F3
headline. **Do not record it as-is.** Judges score only the PDF; if time is short, cut this
entirely rather than shipping a video that contradicts the report.

**6.6 HF lens release (~15 min human, needs user HF login).** `hf-release/upload.sh`.
The report's Code-and-Data section **promises both lens targets** (final + penultimate,
fp32). Either run the upload or amend that sentence — a submitted paper promising artifacts
that do not resolve is a Dimension-2 problem and a link-check failure against the official
pre-submission checklist.

**6.4 Submit.** Sunday Aug 16, 11:59pm AoE. Resubmission is allowed until the deadline, so
submit an acceptable version early and overwrite.

---

## D. The 🔮 optional list — final dispositions

- **X.1, X.2 — CUT.** Already marked ❌. Note the *grounds* have changed: the cut now rests
  on the power-and-confound argument, not on "the deprecation was vindicated"
  (`00-why-round-2.md` §4). X.1 is additionally superseded by **D2**, which
  asks the same question with a dependent variable that can answer it.
- **X.3 Cross-model generality (second model family)** — J6 covered within-family transfer
  (Qwen3-4B). A second *family* (Llama-3.2-3B) remains genuinely open and would strengthen
  Dimension 1. But it needs a fitted J-lens for that model; unless a public one exists,
  this is a lens-fit (~4 h at 99.9 s/prompt) plus the atlas, which does not fit the runway.
  **Leave as stated future work in the Discussion.**
- **X.4 — CLOSED** via the audit's X4; the trajectory asymmetry is a stated finding.
- **X.5 Introspection-consistency** — correlate per-prompt J-lens routing with self-report
  accuracy. Interesting and Track-3 relevant, but it reads out through the self-report
  channel whose non-specificity is the whole subject of §4.6, so it inherits the C6
  problem. **D3 is the prerequisite**: if the matched-battery contrast shows nothing, X.5
  has no foundation. Defer.

---

## E. Naming note

`detailed-checklist.md:171` calls the current draft "report-draft.md v3" (internal revision
count). It has been archived as **`report/report-draft-v1.md`** — v1 meaning *first
submission-candidate draft*, not third internal revision. Update the checklist line to
avoid a future agent looking for a nonexistent v3, and note that `report/report-draft.md`
is now the name reserved for the v2 rewrite specified in `REPORT-V2-SPEC.md`.
