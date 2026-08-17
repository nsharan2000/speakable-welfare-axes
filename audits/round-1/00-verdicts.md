# What the audit found — corrected numbers and surviving claims

Re-analysis from the raw per-cell rows (`primary_rows.jsonl`, `atlas_rows.jsonl`,
`mech_decompositions.json`, `j4_rows_judged.jsonl`, `ksweep.json`), not the summary
JSONs. Full write-up: `audit_report.md`; machine-readable: `audit_findings.json`.

## 1. The primary claim fails its own pre-registered controls

`primary_analysis.json` says `decision: H1_verbalizable` (d_z=2.49, p=1e-4). Recomputed:

| | Gold | Mold |
|---|---|---|
| E(v) | +4.83 | +3.71 |
| E(u) | -2.60 | -0.43 |
| E(random, n=20) | **+2.72** | -0.96 |
| v-u (headline) | +7.43 | +4.14 |
| ...from u falling *below* random | **+5.32 (71.6%)** | -0.53 (-12.8%) |
| randoms with mean E >= E(v) | **4/20 (p=0.238)** | 0/20 (p=0.048) |

- **Gold**: most of the effect is the control being *suppressed* below chance; v_Gold is
  inside the random band. A v-u contrast is only meaningful if u is neutral. It is not.
- **Mold**: v clears randoms, but **C6 kills the interpretation** — the same injection
  shifts valence *more* on 10 unrelated factual prompts (+4.54) than on the 15 welfare
  self-report prompts (+3.71). A global output shift, not a self-report channel.
- **C7**: congruent monotonicity holds only for v_Mold (rho=+1.0). v_Gold rho=-0.4,
  u_Gold rho=-1.0, u_Mold rho=-0.6.

**Consequence:** the steer-and-ask arm is a **null**, and the Aug-12 deprecation of
steer-and-confess is independently vindicated. See `F5`.

## 2. The strongest claim is under-reported for a purely arithmetic reason

J-space variance fraction (`mech_decompositions.json`), 8-direction random null:

| direction | J-share | z vs random | randoms >= | perm p |
|---|---|---|---|---|
| language (known-reportable ceiling) | 0.1138 | **+22.9** | 0/8 | 0.111 |
| v_Mold | 0.0833 | **+11.2** | 0/8 | 0.111 |
| v_Gold | 0.0547 | **+5.1** | 0/8 | 0.111 |
| u_Mold | 0.0486 | +1.0 | 1/8 | 0.222 |
| u_Gold | 0.0393 | +0.5 | 3/8 | 0.444 |

A one-sided exact permutation test over n randoms cannot report below **1/(n+1)**. At
n=8 the floor is 0.111, so a z=+11 effect reports as non-significant. This is the claim
the thesis rests on, it is pure reading (no steering, no self-report), and cohort size
alone fixes it. See `F2`.

Current floors: J3 k-sweep n=2 -> 0.333; J1 atlas n=6 -> 0.143; J-share n=8 -> 0.111;
primary C1 n=20 -> 0.048. Target n=100 -> 0.0099.

## 3. The "4-9x" headline is not a fixed quantity — two estimands are in play

**Own-pole score ratio** (v/u on the axis's own pole):

| readout | aggregation | Gold | Mold |
|---|---|---|---|
| raw W_U | treatment layer | 5.60x | 3.84x |
| raw W_U | band L16-31 | 2.82x | **0.69x** |
| J-lens | treatment layer | 1.50x | 4.75x |
| J-lens | band L16-31 | 1.23x | 6.53x |
| J-lens normalized | treatment layer | **0.62x** | 1.47x |
| J-lens normalized | band L16-31 | **0.66x** | 5.74x |

Range **0.62x-6.53x**, inverting (naive > trained) in 3 of 12 cells.

**Congruent pole-difference ratio** (own pole - opposite pole) — the form the report's
quoted 3.9x/9.4x derives from — is *less* stable: **-9.10x to +47.6x**, changing sign
wherever the naive denominator crosses zero. Its closest cell to the quoted pair
(J-lens, band L16-31) gives 5.97x Gold / 8.16x Mold, so **neither quoted number is
reproduced exactly by any single convention**. See `F4`.

## 4. J4's causal claim needs one arm, not more data

| arm (sentiment) | n | mean | 95% CI |
|---|---|---|---|
| clean | 16 | +0.25 | [+0.01, +0.49] |
| full Gold / Mold | 16 | +0.44 / -0.88 | [+0.10,+0.77] / [-1.30,-0.45] |
| **J-component** Gold / Mold | 16 | **+1.56 / -1.56** | [+1.13,+2.00] / [-1.95,-1.17] |
| residual (perp) Gold / Mold | 16 | +0.25 / -0.06 | [-0.06,+0.56] / [-0.47,+0.35] |

Pooled within-cell SD = 0.61 -> at n=16 the 80%-power detectable difference is 0.61, well
under the observed 1.31/1.50 gap. **More generations buy nothing.** The gap is a missing
control arm: random-direction J-component. See `F1`.

## 5. Reproducibility defects

1. `report-draft.md` 4.6 cites `compare_u.json` — **file does not exist**. The
   cos 0.535/0.144 values live only in `log.md:488`. -> `F6`
2. `ksweep.json` covers layers 17,19,21,23,25,27,29 (odd only), so **mold's treatment
   layer L24 is never evaluated**, yet the k-robustness claim covers both poles. -> `F7`
3. J1 atlas randoms exist only under `concept='gold'`; the mold null must be read from
   the `mold_pole` field of those rows. Usable, but n=6 and must be labelled as such.

## 6. What the paper should claim

Keep: trained axes at z=+5..+11 in the verbalizable subspace while norm-matched naive
controls sit at chance (harden via F2/F3); the channel-split causal result (gate on F1);
the distress-first trajectory asymmetry (X4); the honest null (F5); and the
control-quality finding that "the" naive control is not well-defined (cos 0.14-0.54
between two published-recipe constructions).

Drop from the abstract: "self-report routing" as a pillar. It does not survive C6/C7.

Unchanged: functional welfare only — no claims about subjective experience. The existing
framing is correct and should stay exactly as it is.
