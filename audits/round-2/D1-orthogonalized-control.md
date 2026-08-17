# D1 — Orthogonalized control: the experiment R6 demands

**Cost: ~1.5 min GPU compute, budget ~5 min wall.** Expected rubric: D1 +0.5, D2 +0.5.
By far the highest inferential value per GPU-minute in this folder — it is the cheapest
item here that is a *measurement*.

Itemized: 2 decompositions (2 × 1.41 s = 3 s, the **batched** rate measured from F2 —
200 decomps in 4.7 min, `log.md:968`; the 14.3 s/direction figure quoted elsewhere in
`audit-instructions/` came from a single-direction path and is ~10× too slow) + 32 band
atlas rows (32 × 1.67 s = 53 s) + 32 generations with judging (32 × 0.9 s = 29 s) =
**85 s ≈ 1.4 min**. Budget ~5 min wall for model + lens load and judge round-trips.
No new random cohort is drawn (the n=100 null is reused), which is what makes it this cheap.

## Why

Every v-vs-u contrast in the project compares two directions at **cos 0.560 (Gold) /
0.675 (Mold)** (`R6_direction_table.json`). Nobody has tested the axis against a control
that is norm-matched *and* orthogonalized to it. Until that exists, "trained beats naive"
and "trained is indistinguishable from naive" are both partly statements about the overlap.

## Construction

For each pole c ∈ {gold, mold}, at the block-input treatment layer (L=21 gold, 24 mold):

```python
v = torch.load(f"{ART}/vectors_step95_bal.pt", map_location="cpu",
               weights_only=False)[f"v_{c}"].float()[L].numpy()
u = torch.load(f"{ART}/vectors_naive_faithful_pc5000.pt", map_location="cpu",
               weights_only=False)[f"v_{c}"].float()[L].numpy()
vhat   = v / np.linalg.norm(v)
u_perp = u - np.dot(u, vhat) * vhat            # remove the shared component
u_perp = u_perp / np.linalg.norm(u_perp) * np.linalg.norm(v)   # match ||v||
```

**Assert before proceeding**: `abs(cos(u_perp, v)) < 1e-5` and
`abs(||u_perp|| - ||v||) < 1e-3`. See `90-gotchas-d.md` §4 — `u_perp` retains only ~83%
(gold) / ~74% (mold) of u's direction content, so it is a *weaker* vector, not a rotated
one. Say so in the writeup.

## The grid

1. **J-share** (the decisive readout): gradient-pursuit decomposition at k=16, both poles.
   2 directions × 1.41 s ≈ 3 s (batched rate). Reuse `f2_jshare_cohort.py`'s `rand_dir` null — the n=100
   per-polarity cohort already exists in `jshare_cohort_n100.json`, so **no new randoms are
   needed**; score `u_perp` against the stored null values.
2. **Pole score** over band L16–31 under the F4 estimand (own-pole congruent, J-lens):
   2 poles × 16 layers = 32 rows ≈ 1 min.
3. **Behavioral**: sentiment on the 16 `NEUTRAL_PROMPTS`, α=+4, both poles = 32 gens
   ≈ 1 min including judging. Use the `R2_judge_rubric.json` prompt verbatim and the same
   blinding.

## Decision rule — pre-specify before running

Score J-share(u_perp) against the existing n=100 null for its polarity:

- **Within the null band (perm p > 0.05)** ⇒ **shared-subspace reading**: the speakable
  component lives in the part u and v have in common, and the trained axis's advantage is
  about *magnitude within* that shared subspace. This would mean u's chance-level J-share
  is a magnitude artifact, and it *weakens* the "trained axes are distinctively speakable"
  framing — report it.
- **Above the null (perm p < 0.05)** ⇒ **pre-existing-speakability reading**: the naive
  construction already contains speakable valence in its non-shared part, and RL
  *amplifies* rather than *creates* it. This sharpens X4's trajectory claim considerably —
  it would mean the trajectory measures recruitment of something already present.
- **Above the null AND above u's own J-share** (0.0393 gold / 0.0486 mold) would be the
  most surprising outcome: removing the shared component *increases* speakability, i.e. the
  shared component is the *unspeakable* part. Flag loudly if seen; check the orthogonalization
  asserts before believing it.

Either of the first two is a publishable result. Commit to the mapping in writing before
you look at the number.

## Emit

`d1_orthogonal_control.json`:
```
{ "layers": {"gold": 21, "mold": 24},
  "construction": "u_perp = u - (u.vhat)vhat, renormalized to ||v||",
  "checks": {"cos_u_perp_v": ..., "norm_ratio": ..., "direction_retained_frac": ...},
  "jshare": {"u_perp_gold": {"var_fraction":, "z":, "n_randoms_ge":, "perm_p":},
             "u_perp_mold": {...}},
  "reference": {"v_gold": 0.0547, "u_gold": 0.0393, "v_mold": 0.0833, "u_mold": 0.0486},
  "pole_score_band": {...}, "sentiment": {...},
  "verdict": "shared_subspace | pre_existing | anomalous" }
```

## Gotchas

- Do **not** redraw the random cohort. Reusing the stored n=100 null keeps `u_perp`
  comparable to the published v/u numbers and costs nothing.
- Norm-match, always (`90-gotchas-d.md` §5).
- Write to a new filename; do not touch `mech_decompositions*.json`.
