# J6 — Cross-model transfer to Qwen3-4B

**Cost: ~150 min GPU.** Expected rubric: **D1 +1.1**, D2 +0.3 (net +1.4). The only item
that materially moves Impact/Innovation. Run **after** F1/F4/F5, which cost ~nothing and
protect claims you already own.

## Why it is worth 2.5 hours

The sprint page's Track 2 text poses as an explicit open question: *"To what extent do
valence directions found in one model transfer to another?"* Answering the organizers'
own stated question is the most direct available route to a Dimension-1 score of 4-5,
which the rubric gates on being "actually new to the field,rather than replicating recent
work".

It also has two structural advantages:
- **Qwen/Qwen3-4B has a public pre-fitted J-lens** (Neuronpedia), so no ~4-7 h refit;
- it is genuine **in-window work** for the required pre-work disclosure appendix, which
  matters because undisclosed prior work can disqualify and the bulk of this project
  predates the sprint.

## The grid

1. Same-recipe naive-u extraction on Qwen/Qwen3-4B (the `convert_own_u.py` /
   `extract_and_validate.py` path already used for 2507).
2. J1-style atlas on Qwen3-4B with the **public** lens: the F4-selected estimand over the
   workspace band, plus a random cohort (>= 20; 100 if the runway allows).
3. **Token-overlap of J-components across models** — Jaccard on the selected token sets
   at each pole, plus whether the same lexical families appear (the 2507 mold lexicon is
   'failed', 'less/lessness', 'false', 'NONE').
4. Band re-identification on Qwen3-4B before comparing layers — do **not** assume L16-31
   transfers. Run `band_stats.py` first; `band_stats_Qwen3-4B.json` already exists, so
   check it rather than recomputing.

## The honest limitation to state up front

The **trained** axes v are only available for 2507 (they come from the third-party maze-RL
replication). So on Qwen3-4B you can compare *naive* axes and the *lens geometry*, but you
cannot replicate the trained-vs-naive contrast without redoing the RL. Frame the claim
accordingly: this tests whether the **verbalizable-subspace structure and the naive
valence direction** transfer, not whether RL recruitment transfers. Overclaiming here
would be caught immediately.

## Decision rule

- **J-component token sets overlap substantially** (Jaccard > ~0.2 at a pole) -> the
  speakable valence lexicon is a property of the model family, not of one checkpoint.
  Strong, quotable result.
- **Near-zero overlap** -> the lexicon is model-specific; the *method* transfers but the
  *content* does not. Also publishable, and it bounds every claim in the paper to 2507.
- Either way, report the random-cohort baseline for the overlap statistic — two random
  16-token selections from a 150k vocabulary have a non-zero expected Jaccard only by
  chance, and that chance level must be stated.

## Scoping if the runway shrinks

Priority order within J6: (2) atlas with public lens -> (3) token overlap -> (1) own-u
extraction. Steps 2-3 alone answer the organizers' question and cost ~60-80 min; step 1
adds the same-recipe control and can be dropped with a stated limitation.
