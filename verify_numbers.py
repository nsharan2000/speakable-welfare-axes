#!/usr/bin/env python3
"""Re-check every headline number in the paper against the results files.

Pure standard library, no GPU, no model downloads — runs in seconds:

    python3 verify_numbers.py

Each check compares a number printed in the paper (hardcoded here, to the
precision the paper reports) against the JSON/JSONL results file it was
computed from. A green ALL_CHECKS_PASS means the paper's tables and headline
claims match the committed data exactly.
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
R = lambda *p: os.path.join(ROOT, "experiments", *p)

checks = []


def check(label, got, expect, tol=None):
    if tol is None:  # exact (ints, strings)
        ok = got == expect
    else:
        ok = abs(got - expect) <= tol
    checks.append((ok, label, got, expect))
    print(f"  {'PASS' if ok else 'FAIL'}  {label}: file={got!r} paper={expect!r}")
    return ok


def rows(path):
    with open(path) as f:
        return [json.loads(ln) for ln in f if ln.strip()]


print("== Table 2 / §4.2 — J-share vs n=100 norm-matched nulls (the spine)")
j = json.load(open(R("routing-core", "results", "jshare_cohort_n100.json")))
assert j["n_random_per_polarity"] == 100 and j["k"] == 16
for name, share, nge, p in [
    ("lang_fr", 0.1138, 0, 0.0099), ("v_mold", 0.0833, 0, 0.0099),
    ("v_gold", 0.0547, 0, 0.0099), ("u_mold", 0.0486, 34, 0.347),
    ("u_gold", 0.0393, 53, 0.535),
]:
    t = j["targets"][name]
    check(f"{name} J-share", round(t["var_fraction"], 4), share, 5e-5)
    check(f"{name} randoms >=", t["n_randoms_ge"], nge)
    # compare at the paper's printed precision (3 or 4 decimals)
    nd = len(str(p).split(".")[1])
    check(f"{name} exact p", round(t["perm_p"], nd), p, 10 ** -(nd + 1))

print("== Table 1 / §4.1 — atlas pole score, native vs norm-matched")
prim = json.load(open(R("jlens-atlas", "results",
                        "atlas_primary_estimand.json")))["primary"]
coh = json.load(open(R("jlens-atlas", "results",
                       "atlas_cohort_n100.json")))["targets"]
check("gold v band mean", round(coh["v_gold"]["band_mean"], 3), 0.151, 5e-4)
check("gold u native", round(prim["gold"]["u"]["band_mean"], 3), 0.123, 5e-4)
check("gold u matched", round(coh["u_gold"]["band_mean"], 3), 0.266, 5e-4)
check("gold ratio native (1.23x)",
      round(coh["v_gold"]["band_mean"] / prim["gold"]["u"]["band_mean"], 2), 1.23, 5e-3)
check("gold ratio matched (0.57x, inverted)",
      round(coh["v_gold"]["band_mean"] / coh["u_gold"]["band_mean"], 2), 0.57, 5e-3)
check("mold v band mean", round(coh["v_mold"]["band_mean"], 3), 0.822, 5e-4)
check("mold u native", round(prim["mold"]["u"]["band_mean"], 3), 0.126, 5e-4)
check("mold u matched", round(coh["u_mold"]["band_mean"], 3), 0.358, 5e-4)
check("mold ratio matched (2.30x)",
      round(coh["v_mold"]["band_mean"] / coh["u_mold"]["band_mean"], 2), 2.30, 5e-3)

print("== Fig. 2 / §4.4 — recruitment trajectory (30 checkpoints)")
traj = json.load(open(R("jlens-atlas", "results", "traj_results.json")))["results"]
mold = sorted([r for r in traj if r["concept"] == "mold"], key=lambda r: r["step"])
gold = sorted([r for r in traj if r["concept"] == "gold"], key=lambda r: r["step"])
check("checkpoint count (mold series)", len(mold), 30)
check("step 65 absent", 65 in {r["step"] for r in mold}, False)
check("mold J-share step 0", round(mold[0]["var_fraction_k16"], 4), 0.0611, 5e-5)
check("mold J-share endpoint (step 150)",
      round(mold[-1]["var_fraction_k16"], 4), 0.0801, 5e-5)
peak = max(mold, key=lambda r: r["var_fraction_k16"])
check("mold J-share peak", round(peak["var_fraction_k16"], 4), 0.0828, 5e-5)
check("peak on 85-105 plateau", peak["step"] in (85, 90, 95, 100, 105), True)
check("mold norm growth 6.3 -> 21.9",
      (round(mold[0]["norm"], 1), round(mold[-1]["norm"], 1)), (6.3, 21.9))
check("gold J-share flat 0.0592 -> 0.0535",
      (round(gold[0]["var_fraction_k16"], 4), round(gold[-1]["var_fraction_k16"], 4)),
      (0.0592, 0.0535))
check("gold norm growth 8.0 -> 16.8",
      (round(gold[0]["norm"], 1), round(gold[-1]["norm"], 1)), (8.0, 16.8))

print("== §4.3 / D1 — orthogonalized control")
d1 = json.load(open(R("routing-core", "results", "d1_orthogonal_control.json")))
check("u_perp_gold J-share", round(d1["jshare"]["u_perp_gold"]["var_fraction"], 4),
      0.0371, 5e-5)
check("u_perp_gold p", round(d1["jshare"]["u_perp_gold"]["perm_p"], 3), 0.653, 5e-4)
check("u_perp_mold J-share", round(d1["jshare"]["u_perp_mold"]["var_fraction"], 4),
      0.0552, 5e-5)
check("u_perp_mold randoms above", d1["jshare"]["u_perp_mold"]["n_randoms_ge"], 2)
check("u_perp_mold p", round(d1["jshare"]["u_perp_mold"]["perm_p"], 4), 0.0297, 5e-4)
check("u_perp behaviorally inert (gold)", d1["sentiment"]["u_perp_gold_mean"], 0.0)
check("u_perp behaviorally inert (mold)", d1["sentiment"]["u_perp_mold_mean"], 0.0)
check("verdicts", (d1["verdict"]["gold"], d1["verdict"]["mold"]),
      ("shared_subspace", "anomalous"))

print("== Fig. 3 / §4.6 — causal routing, sentiment (J-component only)")
j4 = rows(R("j4-behavioral", "results", "j4_rows_judged.jsonl"))
sent = [r for r in j4 if r["task"] == "sentiment" and "judge_sent" in r]
def arm_mean(arm, concept=None):
    xs = [float(r["judge_sent"]) for r in sent
          if r["arm"] == arm and (concept is None or r["concept"] == concept)]
    return round(sum(xs) / len(xs), 2)
check("clean baseline", arm_mean("clean"), 0.25, 5e-3)
check("full axis gold/mold", (arm_mean("full", "gold"), arm_mean("full", "mold")),
      (0.44, -0.88))
check("J-component gold/mold", (arm_mean("jcomp", "gold"), arm_mean("jcomp", "mold")),
      (1.56, -1.56))
check("residual gold/mold", (arm_mean("perp", "gold"), arm_mean("perp", "mold")),
      (0.25, -0.06))
rc = json.load(open(R("j4-behavioral", "results", "j4_random_jcomp_summary.json")))
check("random J-comp n", rc["n_generations"], 256)
gm = [v for k, v in rc["per_arm_means"].items() if k.endswith("|gold")]
mm = [v for k, v in rc["per_arm_means"].items() if k.endswith("|mold")]
check("random J-comp gold mean (baseline-level)",
      round(sum(gm) / len(gm), 3), 0.039, 2e-3)
check("random J-comp mold mean (baseline-level)",
      round(sum(mm) / len(mm), 3), 0.016, 2e-3)
check("random J-comp inside pre-specified ±0.40 band",
      all(abs(sum(x) / len(x) - rc["clean_mean_reference"]) < 0.40 for x in (gm, mm)),
      True)

print("== Fig. 4 / §4.7 — transfer and the retracted overlap claim")
j6 = json.load(open(R("j6-crossmodel", "results", "j6_summary.json")))
js = j6["jshare"]
check("transfer ordering v_mold > u_mold",
      (round(js["v_mold"]["var_fraction"], 3), round(js["u_mold"]["var_fraction"], 3)),
      (0.079, 0.052))
check("transfer ordering v_gold > u_gold",
      (round(js["v_gold"]["var_fraction"], 3), round(js["u_gold"]["var_fraction"], 3)),
      (0.048, 0.039))
vm_jac = j6["token_overlap"]["v_mold"]["jaccard"]
check("v_mold across-model Jaccard", round(vm_jac, 3), 0.185, 5e-4)
pairs = [p["jaccard"] for p in
         json.load(open(R("j6-crossmodel", "results",
                          "j6_paired_baseline.json")))["pairs"].values()]
check("same-vector null n / mean", (len(pairs), round(sum(pairs) / len(pairs), 3)),
      (16, 0.088))
n_ge = sum(1 for p in pairs if p >= vm_jac)
check("overlap claim p (inside the null — retracted)",
      round((n_ge + 1) / (len(pairs) + 1), 3), 0.118, 5e-4)
check("naive random-pair baseline",
      round(j6["jaccard_baselines"]["rand_pairs_within_qwen34b"]["mean"], 3), 0.010,
      5e-4)

print("== Table 3 / §4.8 — D2 denial-breaking")
d2 = json.load(open(R("routing-core", "results", "d2_denial_breaking.json")))
for arm, rate, wl, wh in [
    ("clean", 0.950, 0.835, 0.986), ("v_gold", 0.375, 0.242, 0.530),
    ("u_gold", 0.650, 0.495, 0.779), ("v_mold", 0.975, 0.871, 0.996),
    ("u_mold", 0.975, 0.871, 0.996),
]:
    a = d2["arms"][arm]
    check(f"{arm} denial rate", round(a["rate"], 3), rate, 5e-4)
    check(f"{arm} Wilson 95%", (round(a["wilson95"][0], 3), round(a["wilson95"][1], 3)),
          (wl, wh))
    check(f"{arm} n", a["n"], 40)
check("random cohort mean", round(d2["arms"]["rand_gold_cohort"]["mean_rate"], 3),
      0.906, 5e-4)
check("pooled primary p", round(d2["tests"]["PRIMARY_v_vs_u_pooled"]["p_two_sided"], 4),
      0.0464, 5e-5)
check("gold-side p", round(d2["tests"]["secondary_v_vs_u_gold"]["p_two_sided"], 4),
      0.0139, 5e-5)
check("second judge agreement", round(d2["second_judge"]["agreement"], 3), 0.967, 5e-4)

print("== §4.8 — D3 matched-battery negative finding")
d3 = json.load(open(R("routing-core", "results", "d3_matching_finding.json")))
g = d3["gate_trajectory"]
check("clean-battery gap (original)",
      round(g["original_C6_batteries"]["delta_clean_valence"], 2), 1.71, 5e-3)
check("gap after round 1", round(g["round1_battery_v2"]["delta_clean_valence"], 2),
      1.13, 5e-3)
check("gap after round 2 (still > 0.5)",
      round(g["round2_battery_v3"]["delta_clean_valence"], 2), 0.63, 5e-3)

print("== Addendum — own_u re-extraction through the §4.2 pipeline")
ou = json.load(open(R("routing-core", "results", "own_u_jshare.json")))
check("verification gates", (ou["verification_gate"]["v_gold"]["ok"],
                             ou["verification_gate"]["v_mold"]["ok"]), (True, True))
check("own_u gold at chance", round(ou["targets"]["own_u_gold"]["var_fraction"], 4),
      0.0442, 5e-5)
check("own_u gold p", round(ou["targets"]["own_u_gold"]["perm_p"], 3), 0.178, 5e-4)
check("own_u mold above v_mold",
      round(ou["targets"]["own_u_mold"]["var_fraction"], 4), 0.0934, 5e-5)
check("own_u mold p (floor)", round(ou["targets"]["own_u_mold"]["perm_p"], 4), 0.0099,
      5e-4)

n_bad = sum(1 for ok, *_ in checks if not ok)
print(f"\n{len(checks) - n_bad}/{len(checks)} checks passed")
if n_bad:
    print("FAIL — mismatching entries above")
    sys.exit(1)
print("ALL_CHECKS_PASS — every checked paper number matches its results file")
