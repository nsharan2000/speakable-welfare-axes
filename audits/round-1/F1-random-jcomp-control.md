# F1 — Random-jcomp control arm for J4

**Cost: ~1.4 min GPU** (~30 s for the minimum viable version). Expected rubric: D2 +0.9,
D3 +0.1 (net **+1.0**). Highest value per GPU-minute in the entire queue.

## Why this is the gate on J4's headline

J4 reports that the axis's J-component alone reproduces and concentrates the behavioural
effect (sentiment +1.56/-1.56 vs residual +0.25/-0.06). But **this repo already measured
that J-component rescaling inflates any direction**: chain-g found a random-jcomp null of
+3.7..+10.7 on the first-token readout, because rescaling `x_j` back to `||x||` amplifies
real token vectors by ~1/sqrt(var_fraction) ~ 3-4x.

J4's arms are `{clean, full, jcomp, perp, full_clamped}` — **there is no random-jcomp
arm**. So "affect travels through the speakable component" is not currently separable
from "rescaling any J-projection produces large behavioural effects". A
mechanistic-interpretability judge will ask this first.

**The amplification is quantified and it works against the current claim.** Rescaling
`x_j` back to `||x||` multiplies the surviving token vectors by 1/sqrt(var_fraction):

| direction | var_fraction | amplification |
|---|---|---|
| v_gold | 0.0547 | 4.28x |
| v_mold | 0.0833 | 3.47x |
| rand_gold0 | 0.0392 | **5.05x** |
| rand_mold0 | 0.0408 | **4.95x** |

Random directions have *lower* J-share, so they get amplified **more** than the trained
axes. If amplification alone drives behaviour, randoms should therefore show an effect at
least as large as the real axes — which makes this a sharp, falsifiable prediction rather
than a vague worry.

## The grid — no new vectors or decompositions needed

`mech_components.npz` **already contains** all 16 random directions with their components
(verified: `rand_gold0..7`, `rand_mold0..7`, each with `__xj`, `__xperp`, `__full`,
`__Vc`, `__meta`; norms already matched to the corresponding v). So this is purely a loop
extension.

In `experiments/j4-behavioral/j4_dissociation.py`, the arm-construction loop (~line 84)
currently does `for c in ["gold", "mold"]` over `v_{c}`. Add, using the identical
rescaling so the comparison is magnitude-matched:

```python
N_RAND_J4 = 8          # 4 is enough for the minimum viable version
for c in ["gold", "mold"]:
    for ri in range(N_RAND_J4):
        name = f"rand_{c}{ri}"
        L    = int(z[f"{name}__meta"][0])
        full = z[f"{name}__full"]
        nrm  = float(np.linalg.norm(full))
        xj   = z[f"{name}__xj"]
        arms.append((c, f"rand_jcomp{ri}",
                     (L, ALPHA * xj / (np.linalg.norm(xj) + 1e-9) * nrm), None))
```

`ALPHA = 4.0` and the `nrm` rescaling must match the real-axis arms exactly — that
identical rescaling is the thing under test.

**Minimum viable version**: sentiment task only, 4 randoms x 2 polarities x 16 prompts =
128 generations ~ 2 min. Full version: 8 randoms x 2 polarities x 3 tasks.
The done-set resume is keyed by `(concept, arm, task, prompt)`, so distinct `arm` names
mean nothing already computed is recomputed or overwritten.

## Decision rule (pre-specified here, before the run)

Compare mean random-jcomp sentiment against the observed real-axis jcomp effect
(+1.56 gold / -1.56 mold, clean baseline +0.25):

- **|mean random-jcomp - clean| >= 0.78** (i.e. >= 50% of the real effect):
  the J-component claim is **confounded by rescaling**. Downgrade 4.5 to descriptive,
  and report the random-jcomp null beside it. The channel-split claim then rests on the
  *contrast between* jcomp and perp within matched directions, not on jcomp's magnitude.
- **|mean random-jcomp - clean| < 0.4** (within ~1 SD of clean): the claim is
  **controlled**. State it causally and report the null alongside as evidence.
- **Between 0.4 and 0.78**: report as partially confounded, with the null in the same
  table; do not use the word "dominates" (currently in `log.md`).

Report either way. A null result here is publishable and protects the rest of the paper —
and given chain-g's first-token finding, inflation is the *expected* outcome. Finding it
in the behavioural channel too would be an honest, useful negative.

## Emit

`j4_rows.jsonl` gains rows with `arm = "rand_jcomp{ri}"`; judge them through the existing
`judge_chunks` workflow. Also write `j4_random_jcomp_summary.json`:
`{n_random, per_arm_means, cohort_mean, cohort_sd, real_jcomp_mean, clean_mean,
verdict}` using the thresholds above.

## Gotchas

- Filter `judge_sent is not None` when aggregating (see `90-gotchas.md` 5).
- Do **not** reuse the `full_clamped` machinery — that arm was invalidated (degenerate
  generation) and is reported as a negative.
- Keep `--quick` off for the reported run; `--quick` cuts prompts to 6 and would make the
  control cohort noisier than the effect it is testing.
