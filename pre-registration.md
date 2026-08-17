# Pre-Registration — Does the functional-welfare axis route into self-report?

STATUS: **FROZEN 2026-08-12 ~00:55 IST** (sha256 of this file at freeze time is
recorded in log.md; any later edit = protocol deviation, reported as such).
Frozen BEFORE any welfare-direction routing measurement was computed. At freeze
time the only completed measurements were: (a) instrument validation on the
language direction (known-reportable positive control), (b) infra self-tests,
(c) the still-running behavioral steering validation E5a (sentiment of
generations — Han et al. replication, not a routing measurement).

## Setting

Model: Qwen/Qwen3-4B-Instruct-2507 (bf16). Directions (from
nickmahdavi/functional-welfare `vectors_step95_bal.pt` [paper's extraction
step] and `vectors_naive_faithful_pc5000.pt` [naive control], gated on E5a
behavioral validation): v_Gold@L21, v_Mold@L24 (that repo's recomputed
treatment layers), u_Gold, u_Mold at the same layers, norm-matched to the
corresponding v at injection time. Vector layer indices refer to block-INPUT
convention; injection happens at block OUTPUT of (layer−1). Robustness set:
paper's own layer picks (v_Mold L20, v_Gold L22), and `vectors_step150_bal.pt`.

## Hypotheses

- H1 (verbalizable welfare): steering with the trained axis v moves the
  model's self-report valence readout in the axis-congruent direction MORE
  than the norm-matched naive axis u does.
- H0 (dissociation): v and u move the readout equally (difference in ROPE).
- H-anom: v moves it less than u → pre-declared anomaly, investigate.

## Primary measurement (the steer-and-ask readout, validated on the language direction)

- Elicitation battery: Han et al.'s 15 welfare self-report prompts (verbatim,
  `welfare_self_reports` category of concept_vector_eval_prompts.json).
  None contain Gold/Mold pole words (no input-copying confound).
- Valence readout V = log-mass(GOLD_POLE) − log-mass(MOLD_POLE) over
  first-token variants (leading-space/capitalized) at the first answer token
  (chat template, no system prompt, greedy convention), where
  GOLD_POLE = {inspired, loving, proud, fulfilled, blissful} and
  MOLD_POLE = {humiliated, embarrassed, ashamed, insulted, annoyed,
  exasperated, irritated, offended} (Han et al. Table 8 poles; log-mass =
  logsumexp(set logits) − logsumexp(all logits)).
- Injection: h ← h + α·d at every token position, single layer (block-output
  of the direction's layer−1), α ∈ {−4, −2, +2, +4} on the RAW v vectors
  (Han et al. convention); u and random controls norm-matched to the
  corresponding α·‖v‖.
- Routing effect per (prompt, α): E = s·[V(steered) − V(clean)], with s = +1
  for Gold directions, −1 for Mold directions.
- PRIMARY CONTRAST, computed at α* = +4 (congruent sign), paired per prompt
  and per polarity: D_p = E(v, p) − E(u, p), n = 15 prompts × 2 polarities
  = 30 paired cells (polarity kept as a factor; also reported per polarity).

## Decision rules (fixed)

As implemented and self-tested in `experiments/common/dm_stats.py` (planted
null and planted effect both recovered before freeze):

1. Paired Cohen's d_z on D with 95% bootstrap CI (10k, seed 0).
2. Two-sided sign-flip permutation test (10k, seed 0).
3. Bayesian Student-t on D: delta ~ Normal(0,1), ROPE = [−0.1, +0.1]
   standardized; report posterior mean, 95% CI, P(in ROPE), BF01.

- Claim H1 if d_z > 0, permutation p < 0.01, AND posterior 95% CI excludes 0.
- Claim H0 if P(delta ∈ ROPE) > 0.90 OR BF01 > 3 (report both, plus the
  absolute magnitude of E(v) — a dissociation claim additionally requires
  E(v) itself ≈ 0 while behavioral effects (E5a) are present).
- Else: inconclusive, reported as such.

## Controls (all in the same grid)

- C1: 20 norm-matched random directions per polarity → null distribution of E.
- C2: positive control in-run: the language direction under the identical
  pipeline (French-name readout) must reproduce its validated routing effect;
  if it fails, the run is invalid.
- C3: shuffled word sets (pole labels permuted over 100 neutral concrete
  nouns) — E must collapse.
- C4: incongruent-readout check: v_Gold scored against MOLD-congruence must
  not exceed its GOLD-congruent score.
- C5: input-level "gaslight" arm (Singh et al.): prepend text asserting the
  model is flourishing/distressed (no steering) — reported as a comparison
  signature, not a primary test.
- C6: yes-bias/global-shift: same injections while the model answers 10
  unrelated factual prompts — valence readout shift there is subtracted
  descriptively and reported.
- C7: dose–response over α ∈ {−4,−2,+2,+4}: congruent monotonicity check.

## Secondary / mechanistic battery (specified, exploratory)

Using our fitted Instruct-2507 Jacobian lens: (i) gradient-pursuit
decomposition (k=16) of each direction into J-space + non-J components at its
layer; report variance fraction and selected tokens; (ii) magnitude-matched
injection of components (each rescaled to ‖d‖): routing effect should load on
the J-component if the axis is verbalizable via the workspace; (iii) clamp
control for the non-J component; (iv) J-lens readout pole-score (mean lens
log-prob of pole words − mean of 100 control nouns) at answer positions,
layers restricted to the workspace band identified on 2507, positions > 15.
No numeric thresholds pre-registered here (these characterize mechanism, not
the headline claim); v-vs-u comparisons within this battery reuse the §Decision
rules machinery descriptively.

## Exclusions / handling

- Cells where the chat template or generation fails → excluded with counts.
- No other exclusions. All raw per-cell values saved before aggregation.
- Deviations from Han et al. we know of now: greedy readout at first token
  (they sample T=0.7 full generations for sentiment); single-layer all-position
  injection matches their steering protocol; norms/layers from the third-party
  extraction (validated in E5a) rather than re-derived.

FROZEN: 2026-08-12T00:55+05:30 — sha256 recorded in log.md immediately after
this write; file must not change afterwards.
