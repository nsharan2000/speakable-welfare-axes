# Repo-wide gotchas — read before running anything

## 1. RNG-stream hazard: raising `n_rand` changes vector *identity* (silent corruption)

`run_mechanistic.py:129` draws randoms as:

```python
rng = np.random.default_rng(31)
for c, s in [("gold", +1), ("mold", -1)]:
    for ri in range(n_rand):
        r = rng.standard_normal(2560)
```

Both polarities share **one** stream and gold is drawn first. Verified empirically:

| | n_rand 8 -> 100 |
|---|---|
| `rand_gold*` | **preserved** (cos = 1.0000) |
| `rand_mold*` | **all change identity** (cos ~ +0.02, -0.004) |

Combine with the caching on line 147 (`if name not in decomps`):
`mech_decompositions.json` **keeps the old `var_fraction` for `rand_mold0..7`** while
`mech_components.npz` is **rewritten unconditionally** with the *new* vectors. The two
files then describe different objects under the same names, with no error raised.

**Required fix when raising the cohort** (see `F2`) — make each vector's identity a pure
function of its own name, via SeedSequence spawning:

```python
def rand_dir(concept, ri, d=2560, base=31):
    rng = np.random.default_rng([base, 0 if concept == "gold" else 1, ri])
    r = rng.standard_normal(d).astype(np.float32)
    return r / np.linalg.norm(r)
```

Do **not** use bare `hash()` for seeds — it is salted per process unless `PYTHONHASHSEED`
is set, so runs would not reproduce.

Then **either** write to `mech_decompositions_n100.json` (preferred — leaves the
published n=8 numbers intact for the report's existing text) **or** delete the `rand_*`
keys from the cache before re-running. Do not half-update.

`atlas.py` is **safe** here: one loop, fixed shape, so raising 6 -> 100 preserves the
first 6 (verified). Its `rng` is seeded 99, independent of the mech stream.

## 2. Resume semantics differ per script — know which you're running

| script | behaviour |
|---|---|
| `atlas.py` | `open(rows_path, "w")` — **not resumable**, full rewrite every run |
| `run_mechanistic.py` | decomposition cache keyed by *name* — skips anything present |
| `j4_dissociation.py` | done-set keyed by `(concept, arm, task, prompt)` — resumable per cell |

An interrupted `atlas.py` therefore leaves a **truncated** `atlas_rows.jsonl`, not a
partial-but-valid one. Check the row count (**1260** for the full 6-random grid) before
trusting it. For F3, write to a new filename so an interruption cannot destroy the
existing atlas.

## 3. `mech_components.npz` already contains everything F1 needs

Verified: 105 keys, including `rand_gold0..7` and `rand_mold0..7`, each with `__xj`,
`__xperp`, `__full`, `__Vc`, `__meta`. Norms are already matched to the corresponding v
(||rand_mold0__full|| = ||v_mold__full|| = 19.340) and `__meta` carries
`[layer, polarity]` — `[23, -1]` for mold, matching `v_mold`.

**So F1 requires no decomposition and no new vectors — only generation.** It is a loop
extension in `j4_dissociation.py`, nothing more.

## 4. Layer-index convention — an off-by-one is live in this codebase

Vector layer indices are **block-INPUT**; injection happens at **block-OUTPUT of
(layer-1)**. In `run_mechanistic.py` the direction is registered at `L-1` while the
vector is read at `[L]`. In the npz, `__meta[0]` is already the *injection* layer (23 for
mold, whose treatment layer is 24). `atlas.py` reads `V[l+1]` for lens layer `l`. When
adding a layer to any grid, copy the surrounding code's convention rather than the number
quoted in the pre-registration.

## 5. Judge scores are sparse — `judge_sent: None` is the norm

In `j4_rows_judged.jsonl`, 360 of 504 rows have `judge_sent: None` — only the 144
`sentiment`-task rows are scored. `backtracking` uses `backtrack_markers` (a count) and
`refusal` was classified separately. Any aggregation must filter
`task == "sentiment" and judge_sent is not None`, or means silently include zeros.

Also: the J4 refusal `jcomp` arm was **90% degenerate** on long prompts (per `log.md`)
and the `full_clamped` arm produced all-zero sentiment — reported as an invalidated arm.
Do not treat those cells as measurements.

## 6. `--quick` changes cohort sizes, not just speed

`--quick` sets `n_rand` 8->2 (mech), 6->2 (atlas), prompt counts 16/20/20->6/6/6 (J4),
and subsamples atlas layers by 4. A `--quick` run overwriting a full run's output is a
silent downgrade. Check `*_meta.json` for the cohort actually used before analysing.

## 7. Vector provenance is third-party

`v_*` from `nickmahdavi/functional-welfare` (`vectors_step95_bal.pt`), `u_*` from
`vectors_naive_faithful_pc5000.pt`. Triangulated against published norms
(u_gold@L22: 7.25 ours vs 7.51 paper) and behaviour, not re-derived. Any claim about
*why* an axis looks a certain way inherits that provenance caveat.

## 8. Absolute paths are container paths

Scripts hardcode `/workspace/...` (e.g. `NPZ`, `LENS_FILE`, `ART =
/workspace/welfare-vectors/artifacts`). These resolve inside the `dm-exp` container,
which mounts the host repo at `/workspace`. Run from inside the container, or the paths
will not exist.
