# Experiment index

Every experiment in the paper: what it measures, the script that produces it,
the results file every number traces to, and where it lands in the paper.
Scripts live under `experiments/`; result files under the same family's
`results/` folder. The `chain_*.sh` scripts record the exact order and
grouping the experiments were executed in on the GPU box — they are
provenance, not required entry points; each `.py` runs standalone (see
README for environment setup).

## Instruments

| What | Script(s) | Output | Paper |
|---|---|---|---|
| Checkpoint-matched Jacobian lens fit (final-layer target) for Qwen3-4B-Instruct-2507 | `jlens-fit-2507/fit_lens.py` | `Qwen3-4B-Instruct-2507_jacobian_lens.pt` (Hugging Face; see README) | §3 |
| Penultimate-target refit (robustness) | `jlens-fit-2507/fit_lens.py --target penult` | `..._jacobian_lens_penult.pt` (Hugging Face) | §4.2 |
| Lens diagnostics / replication gates | `jlens-replication/`, `jlens-validation/`, `jlens-fit-2507/` | `R5_lens_diagnostics.json`, validation results | §3 |
| Known-reportable language direction (ceiling) | `jlens-validation/` | `directions/lang_fr_minus_en_L18.npy` | §4.2 |
| Own naive-direction re-extraction | `welfare-axis/convert_own_u.py`, `welfare-axis/compare_u.py` | `own_u/*/mean_diff.pt`, `own_u/compare_u.json` | §4.5 |

## Core measurements

| # | Question | Script | Results file | Paper |
|---|---|---|---|---|
| F2 | J-share vs n=100 norm-matched random nulls (the spine) | `routing-core/f2_jshare_cohort.py` | `jshare_cohort_n100.json` | §4.2, Table 2, Fig. 1 |
| F3 | Atlas pole-score cohort; norm-matched vs native | `jlens-atlas/` scripts | `atlas_cohort_n100.json`, `atlas_primary_estimand.json` | §4.1, Table 1 |
| J2 | Training-time trajectory over 30 RL checkpoints | `jlens-atlas/` trajectory script | `traj_results.json` | §4.4, Fig. 2 |
| J4/F1 | Component reinjection + rescaled-random control | `j4-behavioral/` + `routing-core/f5_controls.py` | `j4_rows_judged.jsonl`, `j4_random_jcomp_summary.json` | §4.6, Fig. 3 |
| F7 | k-robustness sweep (k ∈ 4..50, both treatment positions) | `routing-core/` k-sweep | `ksweep_ext.json` | §4.2 |
| J5 | Lens-target robustness (final vs penultimate) | `jlens-fit-2507/` comparison | `j5_lens_comparison.json`, `j5_paired_baseline.json` | §4.2 |
| J6 | Cross-model transfer + same-vector token-overlap null | `j6-crossmodel/` | `j6_summary.json`, `j6_paired_baseline.json` | §4.7, Fig. 4 |
| R6 | Naive-construction sensitivity | `welfare-axis/r6_direction_table.py` | `R6_direction_table.json`, `compare_u.json` | §4.5 |
| — | Steering validation & blind sentiment judging | `welfare-axis/validate_steering*.py` | `steering_validation*.json`, `judge_sentiment*.json` | §4.3 |

## Self-report channel (§4.8)

| # | Question | Script | Results file | Paper |
|---|---|---|---|---|
| C6/R7 | Pre-registered steer-and-ask contrast + whole-generation control | `routing-core/run_primary.py`, `routing-core/r7_wholegen.py` | `primary_rows` results, `R7_wholegen.json` | §4.8 (diagnosed null), Fig. 5a |
| D3 | Matched third-person battery (two revision rounds, pre-specified gate) | `routing-core/d3_matched_c6.py` | `d3_matching_finding.json`, `d3_matching_check.json`, `d3_rows.jsonl` | §4.8, Fig. 5b |
| D2 | Binary denial-breaking readout (840 gens, blind, 2 judges) | `routing-core/d2_denial_breaking.py` + `d2_analyze_local.py` | `d2_denial_breaking.json`, `d2_rows.jsonl` | §4.8, Table 3, Fig. 5c |
| R10 | Judge re-labeling of the earlier regex estimate | (judging harness) | `R10_denial_labels.json` | §4.8 note |

## Controls & audits

| # | Question | Script | Results file | Paper |
|---|---|---|---|---|
| D1 | Orthogonalized control u⊥ = u − (u·v̂)v̂ | `routing-core/d1_orthogonal_control.py` | `d1_orthogonal_control.json`, `d1_gen_rows.jsonl` | §4.3 |
| D4 | Control-alignment consolidation | (doc-level; audit round 2) | `experiments/common/results/d4_alignment_table.json` | §4.5, §4.8 |
| own_u | Own re-extraction through the F2 pipeline | `routing-core/own_u_jshare.py` | `own_u_jshare.json` | §4.2/§4.5 addendum |
| L | In-window bit-exact verification rerun | `chain_l_verify.sh` (orchestration) | `*_verify_20260815.json` files | Appendix A |

## Chains (execution provenance)

`chain_b` … `chain_n` under `experiments/` document the exact serial order,
gating, and markers used on the GPU box (one job at a time, memory guards,
GPU canaries). `chain_l_verify.sh` is the in-window verification chain;
`chain_n_dseries.sh` is the round-2 D-series (D1 → D3-clean → gate →
D3-steered → D2); `chain_o_ownu_jshare.sh` is the own_u addendum.

## Statuses at a glance

All experiments above: **complete**, with results committed. Invalidated or
retracted along the way (kept for transparency, flagged in the paper):
the raw atlas headline (§4.1, demoted — magnitude confound), the refusal
steering arm (90% incoherent, excluded in §4.6), the across-lens and
across-model token-overlap claims (§4.7, retracted against the same-vector
null), and the pre-registered steer-and-ask reading (§4.8, diagnosed null
with measured confounds).
