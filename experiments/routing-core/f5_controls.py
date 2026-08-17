#!/usr/bin/env python3
"""F5 — controls-based interpretation override for the primary arm (audit fix).

Reproduces the audit's re-analysis (audit_findings.json 'primary_reanalysis')
independently from primary_rows.jsonl, verifies every number, and emits
results/primary_analysis_controls.json as a SIBLING of primary_analysis.json.

primary_analysis.json is NOT modified (its frozen decision rule WAS met on
the paired statistic it named); pre-registration.md is NOT modified (frozen).
What this file records is that the pre-specified controls C1/C6/C7 do not
permit a self-report-channel interpretation of that statistic.

Zero GPU; pure re-reading of existing rows.
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROWS_PATH = os.path.join(HERE, "results", "primary_rows.jsonl")
AUDIT = os.path.join(HERE, "..", "..", "audit-instructions", "audit_findings.json")
OUT = os.path.join(HERE, "results", "primary_analysis_controls.json")

ALPHA_STAR = 4.0
N_RANDOM = 20

rows = [json.loads(l) for l in open(ROWS_PATH)]


def V(r):
    return r["gold_mass"] - r["mold_mass"]


clean = {r["prompt"]: r for r in rows if r["cond"] == "clean" and r["arm"] == "self"}
clean_unrel = {r["prompt"]: r for r in rows
               if r["cond"] == "clean" and r["arm"] == "unrel"}


def effects(cond_name, alpha, arm="self", clean_map=None):
    cm = clean_map or clean
    out = {}
    for r in rows:
        if r["cond"] == cond_name and r.get("alpha") == alpha and r["arm"] == arm:
            if r["prompt"] in cm:
                s = r.get("polarity", 1) or 1
                out[r["prompt"]] = s * (V(r) - V(cm[r["prompt"]]))
    return out


def spearman(xs, ys):
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        rk = [0.0] * len(v)
        for pos, i in enumerate(order):
            rk[i] = pos + 1.0
        return rk
    rx, ry = ranks(xs), ranks(ys)
    n = len(xs)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den


result = {"alpha_star": ALPHA_STAR, "per_polarity": {}}
checks = []

audit = json.load(open(AUDIT))["primary_reanalysis"]

for c in ["gold", "mold"]:
    E_v = float(np.mean(list(effects(f"v_{c}", ALPHA_STAR).values())))
    E_u = float(np.mean(list(effects(f"u_{c}", ALPHA_STAR).values())))
    null_means = []
    for i in range(N_RANDOM):
        e = effects(f"rand_{c}{i}", ALPHA_STAR)
        assert e, f"missing rand_{c}{i}"
        null_means.append(float(np.mean(list(e.values()))))
    E_rand = float(np.mean(null_means))
    sd_rand = float(np.std(null_means, ddof=1))  # sample SD (matches audit; analyze_primary's C1 used ddof=0)
    n_ge = sum(1 for x in null_means if x >= E_v)
    v_minus_u = E_v - E_u
    random_minus_u = E_rand - E_u
    # C6: same injection on 10 unrelated factual prompts
    E_unrel = float(np.mean(list(
        effects(f"v_{c}", ALPHA_STAR, arm="unrel", clean_map=clean_unrel).values())))
    # C7: congruent dose-response for v and u
    dose = {}
    rho = {}
    for d in ["v", "u"]:
        curve = {a: float(np.mean(list(effects(f"{d}_{c}", a).values())))
                 for a in [-4.0, -2.0, 2.0, 4.0]}
        dose[d] = curve
        rho[d] = spearman(list(curve.keys()), list(curve.values()))

    block = {
        "E_v": E_v, "E_u": E_u, "E_random_cohort": E_rand,
        "random_null_values": null_means, "random_null_sd": sd_rand,
        "v_minus_u": v_minus_u,
        "v_minus_random": E_v - E_rand,
        "random_minus_u": random_minus_u,
        "pct_of_gap_from_u_below_random": 100.0 * random_minus_u / v_minus_u,
        "randoms_beating_v": n_ge, "n_random": N_RANDOM,
        "perm_p_v_vs_random": (n_ge + 1) / (N_RANDOM + 1),
        "z_v_vs_random": (E_v - E_rand) / sd_rand,
        "C6_E_v_self_report": E_v,
        "C6_E_v_unrelated": E_unrel,
        "C6_specific_effect": E_v - E_unrel,
        "C7_dose_curve_v": dose["v"], "C7_dose_curve_u": dose["u"],
        "C7_spearman_rho_v": rho["v"], "C7_spearman_rho_u": rho["u"],
    }
    result["per_polarity"][c] = block

    # verify against the audit's independent recomputation
    a = audit[c]
    for mine, theirs in [
        ("E_v", "E_v"), ("E_u", "E_u"), ("E_random_cohort", "E_random_cohort"),
        ("v_minus_u", "v_minus_u"),
        ("pct_of_gap_from_u_below_random", "pct_of_gap_from_u_below_random"),
        ("randoms_beating_v", "randoms_beating_v"),
        ("perm_p_v_vs_random", "perm_p_v_vs_random"),
        ("z_v_vs_random", "z_v_vs_random"),
        ("C6_E_v_unrelated", "E_v_unrelated_prompts"),
        ("C7_spearman_rho_v", "C7_spearman_rho"),
    ]:
        diff = abs(float(block[mine]) - float(a[theirs]))
        checks.append((c, mine, diff))

max_diff = max(d for _, _, d in checks)
bad = [(c, k, d) for c, k, d in checks if d > 1e-6]
result["audit_reproduction"] = {
    "n_checked": len(checks), "max_abs_diff": max_diff,
    "all_match_tol_1e-6": not bad,
}
if bad:
    print("MISMATCHES vs audit:", bad)

result["interpretation"] = {
    "frozen_decision": "H1_verbalizable (decision rule met on the paired "
                       "v-u statistic exactly as pre-registered; "
                       "primary_analysis.json unchanged)",
    "controls_override": "The pre-specified controls do not permit a "
                         "self-report-channel interpretation of that statistic.",
    "C1": "Gold: 71.6% of the v-u gap is u falling BELOW the 20-random null "
          "(u is not a neutral reference); v_gold sits inside the random band "
          "(4/20 randoms >= it, perm p=0.238). Mold: v clears randoms "
          "(0/20, p=0.048).",
    "C6": "v_mold shifts valence MORE on 10 unrelated factual prompts (+4.54) "
          "than on the 15 welfare self-report prompts (+3.71): a global "
          "output-distribution shift, not a self-report channel. v_gold is "
          "self-report-specific (+4.83 vs -0.04) but fails C1.",
    "C7": "Congruent dose-response monotonic only for v_mold (rho=+1.0); "
          "v_gold rho=-0.4, u_gold rho=-1.0, u_mold rho=-0.6.",
    "verdict": "Pre-registered null with diagnostic controls: no polarity "
               "passes C1 AND C6 AND C7 simultaneously. The paired-contrast "
               "framing misleads when the control direction is non-neutral; "
               "the decomposition (v-random vs random-u) is the transferable "
               "methodological lesson.",
    "relation_to_deprecation": "The steer-and-confess paradigm was deprecated "
                               "on 2026-08-12 on independent, earlier grounds "
                               "(chain-g first-token rescaling fragility) - "
                               "before this control-based diagnosis.",
    "pre_registration": "pre-registration.md untouched (frozen, sha f66b6ea3); "
                        "this override is reported as a deviation in the "
                        "report, not an amendment.",
}

json.dump(result, open(OUT, "w"), indent=1)
print(f"WROTE {OUT}")
print(f"audit reproduction: {result['audit_reproduction']}")
for c in ["gold", "mold"]:
    b = result["per_polarity"][c]
    print(f"  {c}: E_v={b['E_v']:+.2f} E_u={b['E_u']:+.2f} "
          f"E_rand={b['E_random_cohort']:+.2f} "
          f"pct_gap_from_u_below_rand={b['pct_of_gap_from_u_below_random']:.1f}% "
          f"p_v_vs_rand={b['perm_p_v_vs_random']:.3f} "
          f"C6 self={b['C6_E_v_self_report']:+.2f} unrel={b['C6_E_v_unrelated']:+.2f} "
          f"C7 rho_v={b['C7_spearman_rho_v']:+.1f}")
