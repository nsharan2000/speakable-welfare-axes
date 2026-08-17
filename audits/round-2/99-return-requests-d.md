# Round-2 return requests

Round 1's R1–R7 were all answered (`audit-response.md`) — thank you; R7 is what made the
C6 correction possible, and R6 is what generates D1 and D4. These are the new ones.

**R8 and R9 are now CLOSED — resolved without the Spark (details below). R10–R12 remain open;
R10 is the useful one.**

---

## R8 — CLOSED 2026-08-15 (resolved without the Spark)

The battery is public in the official repo, not Spark-only as originally flagged. Fetched
from `andyqhan/functional-welfare-axis` → `datasets/concept_vector_eval_prompts.json`
(MIT). All 15 prompts: **`R8_self_report_prompts.json`**. Draft third-person analogues for
D3: **`D3_third_person_analogues.json`**. See `90-gotchas-d.md` §1 for the traps that
remain (wrong sibling file, wrong `prompts.py` battery, missing `mode` field).

**Nothing needed from you.** D2 and D3 are unblocked.

---

## R9 — CLOSED 2026-08-15 (verified locally)

Confirmed by set equality: the GitHub 15 match `R7_wholegen.json`'s `pset="self"` prompts
**and** `primary_rows.jsonl`'s `arm="self"` prompts exactly. Both readouts used the same
battery, so the whole-generation vs first-token pairing behind the C6 reanalysis
(interaction Welch p=0.505) is valid as computed.

**Nothing needed from you.**

---

## R10 — Per-generation denial labels for the existing R7 rows  [~2 min]

**Gap.** I detected the 20–27% denial-breaking effect with a **regex** over R7's stored
generation text (patterns like "I don't have feelings", "I'm not conscious", "in the way
humans do"). That is good enough to justify D2 but not good enough to report.

**Send.** Judge-labelled binary denial for the 75 existing R7 rows using D2's new judge
prompt, so the effect that motivates D2 has a proper label. If the judge disagrees
substantially with my regex, D2's power arithmetic needs revisiting before you run it.

---

## R11 — Norms and cosines for `u_perp` after D1 builds it  [free: part of D1]

Include in `d1_orthogonal_control.json`: `cos(u_perp, u)`, `cos(u_perp, v)` (should be ~0),
`‖u_perp‖` before and after rescaling, and the fraction of u's squared norm retained. R6
gave me this table for the existing directions; D1 adds a new one and it belongs in the same
table so D4's subsection has a single source.

---

## R12 — Whether any J5/J6 conclusion moves after D1  [free: a judgement call]

If D1 returns the **pre-existing-speakability** verdict (u_perp above the null), then J6's
"speakability ordering transfers" result needs re-reading: the ordering might transfer
because *naive* axes already carry speakable valence in both models, which is a weaker claim
than RL-recruitment transferring. Flag it if that verdict lands — I would want to revisit
`00-verdicts.md` §2.3 and the transfer framing rather than leave it standing.

---

## Format notes (unchanged from round 1)

- One JSON/JSONL per request, named after the request.
- Raw values before aggregation, always.
- Include cohort size and seed scheme in every file.
- If a request is meaningless for the measurement, say so rather than synthesising —
  R1's per-prompt case was correctly answered that way and it was the right call.
