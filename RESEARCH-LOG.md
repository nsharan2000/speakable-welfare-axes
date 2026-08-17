# Research log — public edition

A condensed, researcher-facing chronology of how the results in the paper came
to be, including the mistakes, corrections, and retractions. The internal lab
log this is distilled from is append-only and timestamped; entries here keep
the original dates and preserve every scientifically relevant event —
operational details of our compute environment are omitted. Where an entry
changed a claim in the paper, the paper section is noted.

Legend: ⭐ finding · 🔴 correction/retraction · 📌 decision · ⛔ negative result

---

## Aug 11 — setup and framing

- 📌 Project framed: connect the functional welfare axis (Han et al. 2026) to
  the Jacobian-lens verbalizable subspace (Gurnee et al. 2026). Core question:
  when RL installs a welfare-like state, does it enter the model's speakable
  channel?
- Literature fan-out (5 research documents in `guides/` and the repo history)
  produced two design-changing findings: small-model introspection replications
  are near-null (so self-report must be treated as a *channel to measure*, not
  an oracle), and steering methodology needs magnitude- and identity-matched
  controls throughout (CAA/RepE lineage).
- Statistics module written with self-tests; blind-judging harness designed
  (judges see only shuffled {idx, question, response}; independent second
  judge for reliability).

## Aug 12 — pre-registration, first results, and the audit

- 📌 **Pre-registration frozen** before any welfare measurement
  (`pre-registration.md`, sha f66b6ea3 — never edited afterwards).
- ⭐ Pre-registered primary (steer-and-ask contrast) fired its decision rule
  (paired d_z = 2.49) — but its own pre-specified controls showed 71.6% of the
  contrast came from the naive-control arm, beginning the diagnosis that ends
  in §4.8's "diagnosed null".
- ⭐ First J-space decompositions: the trained axis is speakable and decodes to
  coherent valence tokens; naive and random directions decode to junk (§4.2).
- ⭐ Recruitment trajectory measured over RL checkpoints (§4.4): distress pole
  gains J-share, flourishing pole gains only amplitude.
- ⭐ Component reinjection (§4.6): behavioral drive concentrates in the
  speakable component; clamp arm invalidated (90% incoherent) and excluded.
- 📋 **Independent audit round 1 received and fully adopted** (see
  `audits/round-1/`): a fix queue of F1–F7 re-analyses, all executed.
- 🔴 **F3 (n=100 atlas cohort) overturned our own atlas headline**: the
  pole-score trained/naive gap was mostly magnitude; norm-matching collapses
  the Mold ratio and *inverts* the Gold one. The paper's §4.1 reports this
  demotion; the central claim moved to the scale-invariant J-share (§4.2).
- 🏆 **F2 (n=100 J-share cohort)**: the spine result — 0/100 norm-matched
  randoms reach either trained pole (exact p = 0.0099 at the test's floor);
  naive controls mid-null. Honest note: effect sizes smaller than the first
  8-random estimate suggested, and reported as such.
- ✅ F1: J4's causal claim is controlled — identically rescaled random
  J-components (larger amplification) stay at baseline (§4.6).
- 🐛 A real bug (function-arity in the token-mask helper) was caught by the
  audit's re-run demand — fixed class-wide with a static API audit.
- 🏆 J6 cross-model transfer (§4.7): J-share ordering and the norm-matching
  inversion replicate on Qwen3-4B under its own public lens.
- 🔴 **Same-vector-null lesson (twice)**: the across-lens-target and
  across-model token-overlap claims both die when compared against the correct
  null — re-reading the *same vector* under two lenses churns token sets
  (0.088 ± 0.055), and the measured 0.185 sits inside that. Both claims
  retracted before submission; kept in §4.7 as a worked example of null choice.
- ✅ Verification sweep: 244 numeric claims checked against result files; 14
  wording/number defects fixed.

## Aug 13 — consolidation

- ⭐ Own re-extraction of naive directions (`experiments/welfare-axis/own_u/`)
  lands: "the" naive control is construction-dependent (cos 0.535/0.385
  between constructions at treatment layers) — becomes §4.5.
- ⛔ Mechanistic injections into the legacy self-report channel: honest mixed
  verdict; steered effect not self-report-specific in magnitude (feeds §4.8).
- Report draft v1 assembled (later archived with a defect banner — see below).

## Aug 15 — in-window verification and audit round 2

- 🏆 **In-window verification rerun**: the full n=100 J-share and atlas
  cohorts re-run seed-identically; **bit-exact reproduction** (every diff
  0.0e+00, permutation p identical). This is Appendix A's insurance that the
  headline numbers are in-window results (verify files stamped 2026-08-15 in
  `experiments/routing-core/results/` and `experiments/jlens-atlas/results/`).
- 📋 **Audit round 2 received** (`audits/round-2/`) and executed in full:
  - ✅ R8/R9: the true 15-prompt self-report battery recovered and confirmed
    identical from two independent sources; the auditor's p = 0.505 pairing
    validated.
  - 🔴 R10: an earlier regex-based estimate of "denial-breaking" (20–27%) was
    mostly a **regex artifact** — blind judge labels (two judges, agreement
    1.00 on 75 re-labeled rows) corrected it. Lesson: never trust a regex to
    read prose.
  - Pre-dispatch adversarial review of the D-series scripts found 8 real
    defects (resume/versioning/statistics), all fixed before any GPU time.
  - ⭐ **D1 (orthogonalized control, §4.3)**: u⊥ = u − (u·v̂)v̂ is behaviorally
    inert at both poles; gold-side speakable valence lives in the shared
    subspace (p = 0.653), mold-side u⊥ *rises above* the null and u itself
    (p = 0.0297) — naive distress constructions carry their own speakable
    valence. Steerability and speakability dissociate.
  - ⛔ **D3 (matched battery, §4.8)**: a prompt-matched third-person battery
    is unachievable — denial and length match but a ~0.6-unit valence gap
    survives both pre-specified revision rounds (1.71 → 1.13 → 0.63 vs 0.5
    threshold). Stopped by the pre-specified rule; the failure is the finding:
    the self-report register is negatively loaded in this model.
  - 🏆 **D2 (denial-breaking, §4.8)**: on the binary "does the generation deny
    having inner states" readout (840 generations, blind, second judge 0.967),
    v_Gold dissolves denial (37.5% vs clean 95%, naive 65%, randoms 90.6%);
    pooled pre-specified p = 0.0464, gold-side p = 0.0139. Trained-specific
    and invisible to valence readouts.
- 🔴 Report v1 archived with a defect banner; v2 written fresh under the
  round-2 spec. Number-verification pass caught two errors (trajectory has 30
  checkpoints, not 31 — a step-65 artifact never existed; a double-rounded
  p-value) and pinned endpoint (0.0801) vs peak (0.0828, plateau steps
  85–105) in §4.4. House rule throughout: **the results file beats the draft
  and beats the spec.**
- 🔴 A v1 sentence about the provenance of a mid-project methodological
  decision was removed as unsupported: the deprecation directive preceded the
  analysis that was claimed to justify it. C6's honest verdict is
  "underpowered, not refuted" (interaction p = 0.505).

## Aug 15–16 — own_u J-share addendum

- ⭐ The own re-extraction run through the exact §4.2 pipeline (fresh-v gates
  reproduce the stored cohort to 1e-6; stored n=100 nulls): own_u gold is at
  chance (0.0442, p = 0.178, junk tokens) — a second independent naive
  construction at chance, strengthening the gold spine. own_u mold is **above
  v_Mold** (0.0934 vs 0.0833, 0/100 randoms, p = 0.0099, coherent negative
  tokens) — corroborating D1's anomalous-mold verdict and §4.5's
  construction-dependence, *not* a clean trained>naive replication.
  (`experiments/routing-core/results/own_u_jshare.json`)

---

*Every number above traces to a JSON/JSONL file in `experiments/*/results/`;
run `python verify_numbers.py` to re-check the paper's tables against them.*
