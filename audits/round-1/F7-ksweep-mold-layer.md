# F7 — Evaluate the k-sweep at mold's own treatment layer

**Cost: ~12 min GPU** (~48 decompositions x 14.3 s). Expected rubric: D1 +0.1, D2 +0.5
(net **+0.6**).

## The defect

`atlas.py:148` sets the sweep grid to `list(range(17, 31, 2))` — **odd layers only**:
17, 19, 21, 23, 25, 27, 29. Mold's treatment layer is **L24**, so:

- **the mold k-robustness claim is asserted at layers that exclude mold's own treatment
  layer** (gold's L21 is covered; mold's L24 is not);
- verified directly against `ksweep.json`: no key matches `*|mold|L24|*`.

The report states the J-share gap "holds at every k in {4..50}" for both poles. For mold
that is currently untested at the layer every other mold result uses.

Additionally `atlas.py:157` caps the sweep null at **2 randoms**
(`if sname.startswith("rand") and int(sname[4:]) > 1: continue`) -> p floor 0.333.

## The grid

Add **L22 and L24** (L22 for symmetry with gold's L21+1, L24 as the mold treatment layer)
x k in {4, 8, 16, 25, 50} x sets {step95, naive_faithful} x concepts {gold, mold}, plus
**at least 8 randoms** (20 if you can spare ~5 min more — it takes the floor from 0.333 to
0.048).

Minimum: 2 layers x 5 k x 2 sets x 2 concepts = 40, plus 8 randoms x 2 layers x 5 k = 80.
At 14.3 s that is ~28 min; restricting randoms to k=16 only brings it to ~12 min, which is
the recommended scoping — the k-stability claim needs the real axes across all k, but the
null only needs the k the headline uses.

Change the layer list to `sorted(set(range(17,31,2)) | {22, 24})` rather than replacing it,
so existing cells stay comparable. Write to `ksweep_ext.json`; do not overwrite.

## Emit

`ksweep_ext.json` keyed as the existing file (`{set}|{concept}|L{layer}|k{k}` ->
`{var_fraction, tokens}`), plus a `_null` block giving per-(layer,k) random cohort
mean/sd/n and the resulting `perm_p` for each real axis.

## Decision rule

The claim to support is "the trained/naive J-share gap is robust to k". State it as:
gap present at every k tested **at each pole's own treatment layer**, with the per-k
permutation p against the random cohort. If the gap fails at mold L24 for some k, report
that — it bounds the claim rather than destroying it, and an honest bound is worth more
than an untested generalisation.
