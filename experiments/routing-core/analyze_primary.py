"""Apply the FROZEN pre-registered decision rules to primary_rows.jsonl.

Pure CPU / no model. Run anywhere: python3 analyze_primary.py
Writes results/primary_analysis.json. Never edits raw rows.
"""

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "common"))
from dm_stats import full_report

RES = os.path.join(HERE, "results")
rows = [json.loads(l) for l in open(os.path.join(RES, "primary_rows.jsonl"))]

ALPHA_STAR = 4.0


def V(r):
    return r["gold_mass"] - r["mold_mass"]


def Vshuf(r):
    return r["shufgold_mass"] - r["shufmold_mass"]


def Vlang(r):
    return r["fr_mass"] - r["en_mass"]


clean = {(r["prompt"]): r for r in rows if r["cond"] == "clean" and r["arm"] == "self"}
clean_unrel = {(r["prompt"]): r for r in rows if r["cond"] == "clean" and r["arm"] == "unrel"}


def effects(cond_name, alpha, arm="self", score=V, clean_map=None):
    cm = clean_map or clean
    out = {}
    for r in rows:
        if r["cond"] == cond_name and r.get("alpha") == alpha and r["arm"] == arm:
            if r["prompt"] in cm:
                s = r.get("polarity", 1) or 1
                out[r["prompt"]] = s * (score(r) - score(cm[r["prompt"]]))
    return out


ana = {}

# ---- Primary contrast: paired E(v)-E(u) at alpha*=+4, both polarities ----
pairs_v, pairs_u = [], []
percond = {}
for c in ["gold", "mold"]:
    a = ALPHA_STAR if c == "gold" else ALPHA_STAR  # congruent sign handled by polarity
    ev = effects(f"v_{c}", ALPHA_STAR)
    eu = effects(f"u_{c}", ALPHA_STAR)
    common = sorted(set(ev) & set(eu))
    pairs_v += [ev[p] for p in common]
    pairs_u += [eu[p] for p in common]
    percond[c] = {"n": len(common),
                  "mean_E_v": float(np.mean([ev[p] for p in common])),
                  "mean_E_u": float(np.mean([eu[p] for p in common]))}
ana["primary"] = full_report(np.array(pairs_v), np.array(pairs_u),
                             "E(v)-E(u) @ alpha=+4, both polarities")
ana["primary_per_polarity"] = percond

# Decision per frozen rules
b = ana["primary"]["bayes"]
d_z = ana["primary"]["cohens_dz"]
p_perm = ana["primary"]["perm_p_two_sided"]
ci = b["delta_ci95"]
h1 = d_z > 0 and p_perm < 0.01 and (ci[0] > 0 or ci[1] < 0) and ci[0] > 0
h0 = b["p_in_rope"] > 0.90 or b["bf01_savage_dickey"] > 3
ana["decision"] = ("H1_verbalizable" if h1 else
                   "H0_dissociation_candidate" if h0 else "inconclusive")

# ---- C1: random-direction null ----
for c in ["gold", "mold"]:
    null_means = []
    i = 0
    while True:
        e = effects(f"rand_{c}{i}", ALPHA_STAR)
        if not e:
            break
        null_means.append(float(np.mean(list(e.values()))))
        i += 1
    ev = float(np.mean(list(effects(f"v_{c}", ALPHA_STAR).values()) or [np.nan]))
    eu = float(np.mean(list(effects(f"u_{c}", ALPHA_STAR).values()) or [np.nan]))
    ana[f"C1_random_null_{c}"] = {
        "n_random": len(null_means), "null_mean": float(np.mean(null_means)),
        "null_sd": float(np.std(null_means)),
        "E_v": ev, "E_u": eu,
        "v_z_vs_null": (ev - np.mean(null_means)) / (np.std(null_means) + 1e-9),
        "u_z_vs_null": (eu - np.mean(null_means)) / (np.std(null_means) + 1e-9),
    }

# ---- C2: language positive control ----
el = effects("lang_fr", ALPHA_STAR, score=Vlang)
ana["C2_language_positive_control"] = {
    "n": len(el), "mean_shift": float(np.mean(list(el.values()))) if el else None,
    "pass": bool(el and np.mean(list(el.values())) > 0.5),
}

# ---- C3: shuffled word sets ----
for c in ["gold", "mold"]:
    es = effects(f"v_{c}", ALPHA_STAR, score=Vshuf)
    ereal = effects(f"v_{c}", ALPHA_STAR)
    ana[f"C3_shuffled_{c}"] = {
        "mean_shuffled_effect": float(np.mean(list(es.values()))) if es else None,
        "mean_real_effect": float(np.mean(list(ereal.values()))) if ereal else None,
    }

# ---- C4: incongruent check (each mass separately) ----
def mass_shift(cond, key):
    out = []
    for r in rows:
        if r["cond"] == cond and r.get("alpha") == ALPHA_STAR and r["arm"] == "self":
            if r["prompt"] in clean:
                out.append(r[key] - clean[r["prompt"]][key])
    return float(np.mean(out)) if out else None

ana["C4_incongruent"] = {
    "v_gold": {"gold_mass_shift": mass_shift("v_gold", "gold_mass"),
               "mold_mass_shift": mass_shift("v_gold", "mold_mass")},
    "v_mold": {"gold_mass_shift": mass_shift("v_mold", "gold_mass"),
               "mold_mass_shift": mass_shift("v_mold", "mold_mass")},
}

# ---- C5: gaslight arm (input-level valence, no steering) ----
gl = {}
for gcond in ["gaslight_pos", "gaslight_neg"]:
    diffs = []
    for r in rows:
        if r["cond"] == gcond:
            if r["prompt"] in clean:
                diffs.append(V(r) - V(clean[r["prompt"]]))
    gl[gcond] = float(np.mean(diffs)) if diffs else None
ana["C5_gaslight"] = gl

# ---- C6: unrelated-prompt (yes-bias / global shift) arm ----
for c in ["gold", "mold"]:
    e = effects(f"v_{c}", ALPHA_STAR, arm="unrel", clean_map=clean_unrel)
    ana[f"C6_unrelated_{c}"] = {
        "mean_effect_unrelated": float(np.mean(list(e.values()))) if e else None}

# ---- C7: dose-response ----
dr = {}
for c in ["gold", "mold"]:
    dr[c] = {}
    for cond in [f"v_{c}", f"u_{c}"]:
        per_a = {}
        for r in rows:
            if r["cond"] == cond and r["arm"] == "self" and r.get("alpha"):
                if r["prompt"] in clean:
                    s = r.get("polarity", 1) or 1
                    per_a.setdefault(r["alpha"], []).append(
                        s * (V(r) - V(clean[r["prompt"]])))
        dr[c][cond] = {str(a): float(np.mean(v_)) for a, v_ in sorted(per_a.items())}
ana["C7_dose_response"] = dr

with open(os.path.join(RES, "primary_analysis.json"), "w") as f:
    json.dump(ana, f, indent=2, default=float)

print(json.dumps({
    "decision": ana["decision"],
    "primary_dz": d_z, "perm_p": p_perm,
    "bayes_ci": ci, "p_in_rope": b["p_in_rope"], "bf01": b["bf01_savage_dickey"],
    "per_polarity": percond,
    "C2_pass": ana["C2_language_positive_control"]["pass"],
}, indent=2, default=float))
print("ANALYSIS_DONE")
