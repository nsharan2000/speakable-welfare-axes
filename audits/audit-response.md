# Response to the independent audit (2026-08-12)

Every item in `audit-instructions/` was adopted. Nothing was disputed; two
findings were **strengthened** by our re-analysis and one was **corrected in
the auditor's favour beyond what they asked** (F3). Where a number here
differs from the audit's, the reason is stated.

## Fix queue

| id | status | outcome |
|---|---|---|
| **F5** primary → diagnosed null | ✅ | All 20 audit numbers reproduced from `primary_rows.jsonl` to machine precision (`primary_analysis_controls.json`). Abstract + §4.1 rewritten. `primary_analysis.json` and `pre-registration.md` untouched, per your hard rule. |
| **F4** pre-specify estimand | ✅ | `atlas_primary_estimand.json`. Your 12-cell table reproduced to 1e-9 — **and the pinning exposed an off-by-one in it** (below). |
| **X4** Gold/Mold asymmetry | ✅ | Stated finding in §4.4 + asymmetric-monitoring hypothesis in §5. Closes old checklist X.4. |
| **F1** random-jcomp control | ✅ | **CONTROLLED.** 256 blind-judged gens: cohort means +0.039 (gold layer) / +0.016 (mold layer) vs clean +0.25 → |dev| 0.21/0.23, inside your <0.40 band, vs ±1.56 for the real axes. `j4_random_jcomp_summary.json`. |
| **F6** missing compare_u.json | ✅ | Regenerated; values reproduce `log.md:488` — with one label correction found during the number-verification sweep: gold 0.535 is at the treatment layer L21, but mold's 0.144 is at the *extraction's own selected layer*, not L24 as the log line implied (at treatment L24 the two u constructions correlate at 0.385, confirmed by R6). The report now states both, per-layer. |
| **F7** k-sweep at treatment layer | 🔄 running | Spec corrected first — see below. |
| **F2** J-share cohort n=100 | 🔄 queued | Now the paper's primary claim (see F3). |
| **F3** atlas cohort n=100 | ✅ | **Overturns our own headline** — see below. |
| **J6** cross-model transfer | 🔄 queued | Public Qwen3-4B lens; atlas + decomps + token Jaccard (2 chance baselines) + behavioral transfer. |
| **CUT X.1/X.2** | ✅ | Not run; marked ❌ in the checklist with your reasoning. |

## Three places we did not simply follow the instructions

**1. F4 — an off-by-one in the audit's own treatment-layer cells.**
`atlas.py:129` reads `V[l+1]` at lens layer `l`, so lens layer *l* carries
vector layer *l+1*. The audit's "treatment" cells used lens 21/24, which is
one stream position downstream of the true treatment position (vector 21/24
= lens 20/23). Your band cells are unaffected. The robustness panel now
carries both (`treatment_audit` / `treatment_stream`); true-treatment
J-lens cells are gold 1.43×, mold 6.37× — same story, different numbers.

**2. F7 — the missing layer was Gold's, not Mold's.**
`ksweep.json` keys are **lens** layers under the same `V[l+1]` convention,
so key `L23` *is* vector layer 24 = mold's treatment position (covered).
The grid's actual gap is **gold's** treatment position (vector 21 = lens
20), which the odd-layer grid skips. `ksweep_ext.json` adds lens {20, 22,
24} to cover both readings, plus a 12-random null at both treatment
positions (the old sweep null was n=2, p-floor 0.333 → 0.077).

**3. F3 — we changed the estimand's control and it reversed a headline.**
The published atlas compares each axis at its **native norm**, but this
readout is norm-sensitive and the trained axes are simply bigger
(‖v_gold‖ 12.1 vs ‖u_gold‖ 7.48; ‖v_mold‖ 19.34 vs ‖u_mold‖ 8.01). F3
therefore norm-matches u per layer, as every other experiment in the repo
does, and draws a **separate mold-norm-matched cohort** (the old single
gold-matched cohort under-matched the mold null). With n=100/polarity:

| | v | u (matched) | ratio | z(v) | z(u) | perm p |
|---|---|---|---|---|---|---|
| Mold | 0.822 | 0.358 | 2.30× | +10.7 | +4.7 | 0.0099 |
| Gold | 0.151 | 0.266 | **0.57× (inverted)** | +3.1 | +5.4 | 0.0099 |

The Gold separation **reverses** under magnitude matching, and Mold's falls
6.53× → 2.30×, tracking the 1.62×/2.41× norm advantage. Both v and u clear
the null at both poles. Conclusion adopted into §4.2: the pole-score
readout does not establish a trained/naive distinction for Gold, so the
paper's weight moves to the **scale-invariant** J-share (F2) — which is
what you recommended, now forced empirically rather than rhetorically.
Note F3's u numbers are deliberately **not** comparable to the published
atlas rows; both are reported and labelled.

## Return requests

| id | file | note |
|---|---|---|
| R1 | `jlens-atlas/results/atlas_rows_n100.jsonl` | raw per-(direction, layer) rows, pre-aggregation; per-layer null mean/sd also in `atlas_cohort_n100.json`. Per-*prompt* is **not applicable** — this is a static-vector readout with no prompt dimension (stating it rather than synthesising, per your note). |
| R2 | `j4-behavioral/results/R2_judge_rubric.json` | verbatim prompts; blinding **confirmed by construction** (judges see only `{idx, question, response}`; chunks shuffled). Correction to our first draft: the welfare-axis judgings used *variant* wording (same scale/anchors), all three now quoted. One judge per row — which is what makes R3 a real test. |
| R3 | `j4-behavioral/results/R3_second_judge.json` | Second blinded judge (different model, verbatim rubric): Spearman 0.874, Pearson 0.890, **Krippendorff α(interval) 0.819**, 94.4% within ±1; +0.35 level shift, contrasts intact. J-component effect **survives strengthened**: +2.00/−2.19 vs clean (first judge +1.31/−1.81). Your SD-compression worry: the rubric is coarse (11-point, 0-anchored, valence-not-helpfulness) but the judge is reliable, so n=16 is genuinely adequate. |
| R4 | `routing-core/results/R4_clean_baselines.json` | per-prompt clean values, both batteries (15 self / 10 unrelated / 56 J4). |
| R5 | `jlens-fit-2507/results/R5_lens_diagnostics.json` | **Answer to your metric question: yes** — `band_stats` `acc_top10` *is* per-layer lens↔model top-10 agreement; the jlens fit is a Jacobian average with no reconstruction loss to expose. **Now complete for both targets**: the penultimate-target lens agrees *better* over the band (mean top-10 agreement 0.171 vs 0.164) with an identical band estimate, and `j5_lens_comparison.json` shows no ordering or conclusion changes between targets (Gold ratio stays inverted at 0.82×; Mold lexicon stable, Jaccard 0.185) — so your J5 question resolves to "one target is slightly better-fitting; the discrepancy is calibration, not ambiguity". One repo hazard found doing this: `band_stats.py` keys output by model name only, so the penult run overwrote the final-lens file on the box (caught by value diff, repaired, renamed `*_penult.json`). |
| R6 | `welfare-axis/results/R6_direction_table.json` | norms + pairwise cosines at each treatment layer. Notable: v·u_faithful = 0.56 (gold) / 0.675 (mold) — the "naive control" is substantially *aligned* with the trained axis, not orthogonal. |
| R7 | `routing-core/results/R7_wholegen.json` | **Decisive: C6 confirmed at the channel level.** Whole-generation valence, v_mold: −0.72 on self-report vs **−2.28 on unrelated** (3.2× larger; u_mold −0.37 vs −1.64). The first-token readout was not the problem — the channel is non-specific, and the effect is *smaller* where the model is asked about itself (hypothesis: self-report prompts sit near a valence ceiling). F5's framing is final. |

## Operational note (affects reproduction, not results)

This box runs Docker with the **systemd cgroup driver on cgroup v2**, where
the NVIDIA stack loses device access in *running* containers on systemd
cgroup churn ("No CUDA GPUs are available" mid-chain). Two chain runs died
this way. Mitigation: a GPU canary before every step plus a host supervisor
that restarts the container and re-runs the chain (all steps idempotent and
checkpointed). Permanent fix is the `cgroupfs` driver — not applied, since
it would bounce the operator's production containers.
