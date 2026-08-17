# F2 — Raise the J-share random cohort from 8 to 100 per polarity

**Cost: ~48 min GPU** (200 decompositions x 14.3 s measured). Expected rubric: D2 +0.8,
D3 +0.2 (net **+1.0**). This hardens the claim the paper's thesis rests on.

## Why

The J-space variance-fraction result is the cleanest signal in the repo — trained axes at
z=+5.1 (Gold) and z=+11.2 (Mold), naive controls at chance (z=+0.5, +1.0), language
positive control at z=+22.9. It needs no steering, no self-report channel, and no new
paradigm: it is pure reading through the lens.

**But an exact one-sided permutation test over n randoms cannot report p below 1/(n+1).**
At n=8 the floor is **0.111**, so a z=+11 effect currently reports as non-significant.
n=100 gives 0.0099 — clearing the 0.01 threshold the rubric's robustness clause invites.

## READ `90-gotchas.md` 1 FIRST — this is the silent-corruption case

`run_mechanistic.py` draws both polarities from **one** RNG stream with gold first, so
raising `n_rand` **preserves `rand_gold*` but changes the identity of every
`rand_mold*`** (verified: cos ~0.02). Meanwhile the decomposition cache is keyed by name,
so `mech_decompositions.json` would keep the *old* mold var_fractions while
`mech_components.npz` is rewritten with the *new* mold vectors — two files describing
different objects under the same names, no error raised.

## The grid

100 random directions x 2 polarities = 200 gradient-pursuit decompositions, k=16, at each
polarity's treatment layer (Gold L21, Mold L24 — injection layer 20/23 per the npz
`__meta` convention; see `90-gotchas.md` 4). Norm-match each random to the corresponding
`||v||` exactly as the existing code does.

**Use a per-vector independent stream** so identity is a pure function of the name:

```python
def rand_dir(concept, ri, d=2560, base=31):
    rng = np.random.default_rng([base, 0 if concept == "gold" else 1, ri])
    r = rng.standard_normal(d).astype(np.float32)
    return r / np.linalg.norm(r)
```

Not bare `hash()` — it is salted per process and would not reproduce.

**Write to new filenames**: `mech_decompositions_n100.json` and
`mech_components_n100.npz`. This leaves the n=8 numbers already quoted in the report
intact and verifiable, and avoids the desync above entirely.

## Emit

`jshare_cohort_n100.json`:
```
{ "n_random_per_polarity": 100, "k": 16, "seed_scheme": "default_rng([31, pol, ri])",
  "layers": {"gold": <L>, "mold": <L>},
  "targets": { "v_gold": {...}, "u_gold": {...}, "v_mold": {...}, "u_mold": {...},
               "lang_fr": {...} },
  "null": { "gold": {"values": [...], "mean":, "sd":, "p05":, "p95":},
            "mold": {...} } }
```
Per target report: `var_fraction`, `z`, `n_randoms_ge`, `perm_p = (n_ge+1)/(n+1)`, and the
null's 5th/95th percentiles. Also carry the language direction as the
known-reportable calibration ceiling — it is what makes "0.083 is a meaningful J-share"
interpretable rather than just "above zero".

## Verification before you trust it

- The 8 original `rand_gold*` values must reproduce exactly under the new scheme's
  indices, or the seed change is not what you think it is. If they do not match, that is
  expected (different stream) — just confirm the *distribution* is consistent:
  new null mean should sit within ~1 SD of the old (0.0378 gold / 0.0451 mold).
- `var_fraction` for `v_mold` must come out 0.0833 (unchanged — same vector, same lens).
  If it moves, the lens file or layer convention changed and everything else is suspect.

## Order

Run **before** F3: half the cost, and it covers the more important claim.
