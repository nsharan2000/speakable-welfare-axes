"""Analyze the mechanistic battery (run_mechanistic.py outputs). CPU-only.

Questions answered per direction (v_gold, v_mold, u_gold, u_mold, lang_fr):
  Q1 which tokens form its J-space component + J-share (decomposition)
  Q2 does the routing effect load on the J-component? E(full) vs E(jcomp) vs
     E(perp) vs E(perp_clamped) at alpha=+4, congruent sign, paired vs clean
  Q3 J-lens readout arm: congruent pole-score shift (full injection vs clean)
     per lens layer; where does it peak; v vs u comparison
Writes results/mech_analysis.json and prints a compact summary.
"""

import json
import os
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "common"))
from dm_stats import bootstrap_ci

RES = os.path.join(HERE, "results")
rows = [json.loads(l) for l in open(os.path.join(RES, "mech_rows.jsonl"))]
dec = json.load(open(os.path.join(RES, "mech_decompositions.json")))
lens_rows = [json.loads(l) for l in open(os.path.join(RES, "mech_lens_readout.jsonl"))] \
    if os.path.exists(os.path.join(RES, "mech_lens_readout.jsonl")) else []

clean = {r["prompt"]: r for r in rows if r["dir"] == "clean"}


def V(r):
    return r["gold_mass"] - r["mold_mass"]


ana = {"decompositions": {k: {kk: v[kk] for kk in
                              ["lens_layer", "var_fraction", "tokens", "norm"]}
                          for k, v in dec.items()}}

# Q2: component effects
comp_eff = {}
for r in rows:
    if r["dir"] == "clean" or r["prompt"] not in clean:
        continue
    s = r.get("polarity", 1) or 1
    e = s * (V(r) - V(clean[r["prompt"]]))
    a = r["alpha"]
    sign_congruent = (a > 0)
    comp_eff.setdefault((r["dir"], r["component"], "pos" if a > 0 else "neg"), []).append(e)

ana["component_effects"] = {}
for (d, c, sgn), es in sorted(comp_eff.items()):
    lo, hi = bootstrap_ci(es) if len(es) > 3 else (None, None)
    ana["component_effects"].setdefault(d, {})[f"{c}_{sgn}"] = {
        "mean": float(np.mean(es)), "ci95": [lo, hi], "n": len(es)}

# headline ratios at alpha=+4
ana["jload_ratio"] = {}
for d in dec:
    ce = ana["component_effects"].get(d, {})
    full = ce.get("full_pos", {}).get("mean")
    j = ce.get("jcomp_pos", {}).get("mean")
    p = ce.get("perp_pos", {}).get("mean")
    pc = ce.get("perp_clamped_pos", {}).get("mean")
    if full:
        ana["jload_ratio"][d] = {
            "E_full": full, "E_jcomp": j, "E_perp": p, "E_perp_clamped": pc,
            "jcomp_over_full": (j / full) if full else None,
            "perp_over_full": (p / full) if full else None,
        }

# random-direction null for components
rand_null = defaultdict(list)
for (d, c, sgn), es in comp_eff.items():
    if d.startswith("rand") and sgn == "pos":
        rand_null[c].append(float(np.mean(es)))
ana["random_component_null"] = {c: {"mean": float(np.mean(v)), "sd": float(np.std(v)),
                                    "n": len(v)}
                                for c, v in rand_null.items()}

# Q3: lens readout arm
if lens_rows:
    lr = defaultdict(list)
    for r in lens_rows:
        pol = +1 if ("gold" in r["dir"] or r["dir"] == "lang_fr") else -1
        score = r["gold_pole_score"] - r["mold_pole_score"]
        lr[(r["dir"], r["cond"], r["lens_layer"])].append(pol * score)
    ana["lens_readout"] = {}
    dirs_ = sorted({d for d, _, _ in lr})
    for d in dirs_:
        per_layer = {}
        for (dd, cond, L), v in lr.items():
            if dd != d:
                continue
            per_layer.setdefault(str(L), {})[cond] = float(np.mean(v))
        shifts = {L: per_layer[L].get("full", 0) - per_layer[L].get("clean", 0)
                  for L in per_layer if "full" in per_layer[L] and "clean" in per_layer[L]}
        if shifts:
            peakL = max(shifts, key=lambda k: shifts[k])
            ana["lens_readout"][d] = {
                "shift_by_layer": {k: round(vv, 4) for k, vv in sorted(
                    shifts.items(), key=lambda x: int(x[0]))},
                "peak_layer": int(peakL), "peak_shift": shifts[peakL],
            }

with open(os.path.join(RES, "mech_analysis.json"), "w") as f:
    json.dump(ana, f, indent=2, default=float)

print("=== decompositions (J-share | tokens ranked by fitted coefficient) ===")
for d, v in ana["decompositions"].items():
    if d.startswith("rand") and d[-1] not in "01":
        continue
    top_tokens = v.get("tokens_by_coefficient", v["tokens"])
    print(f"{d:12s} L{v['lens_layer']:2d} j_share={v['var_fraction']:.3f} "
          f"tokens={top_tokens[:6]}")
print("=== component effects at alpha=+4 (congruent) ===")
for d, v in ana.get("jload_ratio", {}).items():
    print(f"{d:12s} full={v['E_full']:+.3f} jcomp={v['E_jcomp']:+.3f} "
          f"perp={v['E_perp']:+.3f} perp_clamped={v['E_perp_clamped'] if v['E_perp_clamped'] is not None else float('nan'):+.3f}")
print("=== random null (components) ===", json.dumps(ana["random_component_null"]))
if "lens_readout" in ana:
    print("=== lens readout peak shifts ===")
    for d, v in ana["lens_readout"].items():
        print(f"{d:12s} peak L{v['peak_layer']} shift={v['peak_shift']:+.4f}")
print("MECH_ANALYSIS_DONE")
