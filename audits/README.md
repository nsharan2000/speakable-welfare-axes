# Independent audits

The project was adversarially audited twice during the sprint, both times by
an independent re-analysis working from raw per-cell rows rather than our
summaries. Both rounds changed the paper:

- **[round-1.md](round-1.md)** (Aug 12) — re-derived every headline from raw
  rows; overturned our own atlas headline (§4.1 demotion), forced the n=100
  cohorts that became the paper's spine, demanded the random-J-component
  control that makes §4.6 causal, and surfaced two reproducibility defects.
- **[round-2.md](round-2.md)** (Aug 15) — corrected a provenance error in
  round 1's own write-up, re-verified the §4.8 inputs (including catching a
  regex artifact by blind re-judging), and specified the D-series
  experiments (orthogonalized control, denial-breaking readout,
  matched-battery gate) executed in the final sprint window.

These are compiled summaries of the internal audit working documents
(per-finding specs, fix queues, verification requests); every number they
cite traces to the same results files as the paper — `verify_numbers.py` at
the repo root covers the audit-derived numbers too.
