# F6 — Generate the missing compare_u.json

**Cost: ~2 min GPU.** Expected rubric: D2 +0.4, D3 +0.3 (net **+0.7**). Cheapest
credibility fix in the queue.

## The defect

`report/report-draft.md` 4.6 cites `compare_u.json` for the claim that the two naive-u
constructions correlate at only cos 0.54 (gold) / 0.14 (mold). **The file does not exist
in the repo.** The values appear only in `log.md:488`
(`gold cos 0.535@L21, mold cos 0.144@L24`). `experiments/welfare-axis/compare_u.py`
exists but was never run to JSON.

This matters more than its size suggests: the cos 0.14-0.54 figure supports one of the
paper's four stated contributions ("the naive control is not well-defined"). A judge
checking the strongest control-quality claim finds a dangling reference.

## What to do

Run `python experiments/welfare-axis/compare_u.py` and confirm it emits
`experiments/welfare-axis/results/compare_u.json`. Verify the output reproduces
cos 0.535 (gold @L21) and 0.144 (mold @L24) against `log.md:488`. If it does not
reproduce, that is itself a finding — report the discrepancy rather than the log value.

Emit, at minimum:
```
{ "pairs": { "own_u_vs_faithful_u": {"gold": {"layer":21,"cos":...},
                                     "mold": {"layer":24,"cos":...}},
             "v_vs_faithful_u":     {...},
             "v_vs_own_u":          {...} },
  "note": "cosines at each pole's treatment layer, block-input convention" }
```

The `v_vs_*` pairs are worth including even if the report does not currently cite them:
they bound how much of the naive/trained distinction is direction versus magnitude.

## If it cannot be re-run

Cite `log.md:488` explicitly in the report instead. An honest pointer to the lab log is
fine; a citation to a nonexistent results file is not.
