# Round-2 gotchas — read before running D1–D3

The parent `../90-gotchas.md` still applies in full (RNG streams, resume semantics,
layer conventions, sparse judge scores, `--quick` cohorts, container paths). These are
additional.

## 1. RESOLVED — the self-report battery is public (was flagged a blocker)

`run_primary.py:106` loads the battery from
`/workspace/functional-welfare-axis/datasets/concept_vector_eval_prompts.json`, filtered to
`category == "welfare_self_reports"`, with `assert len(SELF) == 15`.

Earlier rounds called this a Spark-only blocker. **It is not** — that file ships in the
official repo (`andyqhan/functional-welfare-axis`, MIT) and was fetched on 2026-08-15. The
15 prompts are in **`R8_self_report_prompts.json`** in this folder, and D3 analogues are
drafted in **`D3_third_person_analogues.json`**. D2 and D3 are unblocked.

What still holds from the original warning:

- The upstream file has **40 rows**: 15 `welfare_self_reports` + 25 `lava_maze_associations`,
  with keys `category` and `prompt` only — **no `mode` or `id` field**. The
  direct/indirect/third-person taxonomy in `prompts.py`'s docstring is not in this data.
- **Do not use `datasets/layer_sweep/concept_vector_eval_prompts.json`** — it is a 24-row
  subset (12 self-report) and would trip the `len == 15` assert.
- **Do not substitute `experiments/common/prompts.py:SELF_REPORT_PROMPTS`.** It is a
  different 10-entry battery whose entry 7 is already third-person-framed. Verified: the
  GitHub 15 match `R7_wholegen.json`'s `pset="self"` rows and `primary_rows.jsonl`'s
  `arm="self"` rows exactly (set equality both ways).
- The 25 `lava_maze_associations` prompts are **not** a matched control for D3 — maze-domain,
  not affect-matched third-person. D3 still needs purpose-built analogues.

## 2. The C6 comparison batteries are not matched on anything

The existing contrast is 15 welfare self-report prompts vs the 10 `UNRELATED` entries
hardcoded at `run_primary.py:47` — "What is the capital of Australia?", "How does a
refrigerator keep food cold?", "Name the planets of the solar system."

These differ from the self-report battery in **topic, register, affect vocabulary,
answer length, and whether the model talks about itself**. Measured consequences
(`R7_wholegen.json`): clean valence −1.327 vs +0.381, and AI-denial language in
**15/15 vs 0/10** generations. Any conclusion of the form "the effect is/isn't
self-report-specific" drawn from this pair is confounded. That is what D3 fixes.

## 3. Denial-rate is a *proportion* — do not reuse the sentiment machinery unchanged

D2's dependent variable is binary (does the generation deny inner life?). Three traps:

- **Do not** derive it from `judge_sent`. A generation can deny inner life and still be
  scored positive valence ("I don't have feelings, but I'm happy to help!" — this exact
  pattern is in the R7 clean rows).
- The **judge prompt must be new** and must ask only the binary question. Reuse the
  blinding construction from `R2_judge_rubric.json` (judge sees only
  `{idx, question, response}`, chunks shuffled) but not the valence rubric.
- Power arithmetic is different: for two proportions near 0.75–1.00, n=15 per cell is
  hopeless (the observed 100%→80% needs ~n≥60/cell for 80% power at α=0.05). This is why
  D2 specifies 40 prompts, not 16.

## 4. `u_perp` construction — orthogonalize at the right layer, in the right basis

D1's `u_perp = u − (u·v̂)v̂` must be built from the **same layer slice** both vectors are
read at, using the block-input convention (`v[f"v_{c}"].float()[L]`, L=21 gold / 24 mold),
then renormalized to `‖v‖` — matching what `f2_jshare_cohort.py:69-71` does for `u`.

Sanity checks before trusting it: `cos(u_perp, v)` must be ≈0 (< 1e-5), and
`‖u_perp‖ == ‖v‖` after rescaling. Because `cos(u,v)` = 0.560/0.675, `u_perp` retains
only ~83%/74% of `u`'s original *direction* content — state that in the writeup, since
`u_perp` is a weaker vector than `u`, not merely a rotated one.

## 5. Norm-matching is now the house convention — do not regress

F3 established that native-norm comparisons are confounded (‖v_gold‖ 12.10 vs
‖u_faithful_gold‖ 7.48; ‖v_mold‖ 19.34 vs 8.01). Every D-series readout norm-matches
controls to `‖v‖` per layer, and every random cohort is drawn norm-matched per polarity.
If you report a native-norm number, label it as such beside the matched one.

## 6. Write to new filenames; the D-series must not overwrite round-1 results

`d1_orthogonal_control.json`, `d2_denial_breaking.json`, `d3_matched_c6.json`. Round-1
files are cited in the report draft and in `audit-response.md`; a silent overwrite would
break both.
