# F3 — Raise the atlas pole-score cohort from 6 to 100 per polarity

**Cost: ~89 min GPU** (3200 rows x 1.67 s measured). Expected rubric: D2 +0.7, D3 +0.1
(net **+0.8**). **Run after F4** — it should measure the estimand F4 selects, not all three.

## Why

The "trained axis reads out its own valence above naive" claim has a **6-direction null**
(p floor 0.143 — cannot reach 0.05 at any effect size). It is also the claim whose
magnitude F4 shows to be convention-dependent, so cohort and estimand must be fixed
together.

## The grid — scoped deliberately

- 100 randoms x 2 polarities
- **workspace band L16-31 only** (16 layers, not all 35)
- **one readout variant** — the one F4 pre-specifies
- => 100 x 2 x 16 = **3200 rows ~ 89 min**

Do **not** run 100 x 2 x 35 x 3 variants: that is ~585 min for no additional claim. The
robustness table across conventions comes from the *existing* atlas rows (F4), which
already cover all 35 layers x 3 variants for the real axes.

## Two structural fixes to make while you are in there

1. **Store randoms under both concept labels.** Currently `atlas.py:120` writes only
   `vecs[(f"rand{ri}", "gold")]`, so the mold null has to be read from the `mold_pole`
   field of gold-labelled rows. It works (each row carries both pole scores), but it is
   a trap for anyone analysing by `concept`. Either write both labels or record the
   convention in `atlas_meta.json`.
2. **Write to a new filename.** `atlas.py` opens `atlas_rows.jsonl` with mode `"w"` and is
   **not resumable** (see `90-gotchas.md` 2) — an interruption truncates the existing
   atlas. Emit `atlas_rows_n100.jsonl` instead and leave the 1260-row original untouched.

`atlas.py`'s RNG is safe to extend: one loop, fixed shape, seed 99, so raising 6 -> 100
preserves the first 6 (verified). The existing 6 randoms remain valid members of the
larger cohort.

## Emit

`atlas_cohort_n100.json`:
```
{ "n_random_per_polarity": 100, "estimand": "<name from F4>",
  "readout_variant": "<...>", "layers": [16..31],
  "targets": {"v_gold": {...}, "u_gold": {...}, "v_mold": {...}, "u_mold": {...}},
  "null": {"gold": {"values":[...], "mean":, "sd":, "p05":, "p95":}, "mold": {...}} }
```
Per target: band-averaged estimand value, `z`, `n_randoms_ge`, `perm_p=(n_ge+1)/(n+1)`,
**and the absolute pole scores** (not just the ratio — see F4 5).

## Verification

The 6 existing randoms' band-averaged values must appear unchanged in the new cohort. If
they do not, the layer convention or lens file changed; stop and diagnose rather than
reporting.
