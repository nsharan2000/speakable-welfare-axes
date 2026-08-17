#!/usr/bin/env python3
"""F4 — pre-specify one atlas estimand; emit robustness panel (audit fix).

Primary estimand (chosen on instrument grounds, stated in Methods):
  own-pole congruent score under the J-lens readout ('jlens' variant),
  averaged over the workspace band L16-31.
Rationale (from audit F4 + our band_stats): the pole-DIFFERENCE ratio's
denominator passes through zero for naive axes (range -9.1x..+47.6x), so it
is unusable; band-averaging avoids privileging one treatment layer; L16-31
is the band the instrument was validated on (band_stats persistence).

Convention pinning: before reporting anything, this script must reproduce
the audit's 12-cell own-pole table (audit_findings.json
'atlas_ratio_instability') to 1e-9, which fixes the treatment-lens-layer and
band-index conventions unambiguously. If reproduction fails, it exits
nonzero and reports the mismatch instead of emitting numbers.

Zero GPU. Reads results/atlas_rows.jsonl (1260 rows, untouched).
"""
import itertools
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROWS = os.path.join(HERE, "results", "atlas_rows.jsonl")
AUDIT = os.path.join(HERE, "..", "..", "audit-instructions", "audit_findings.json")
OUT = os.path.join(HERE, "results", "atlas_primary_estimand.json")

V_SET, U_SET = "step95", "naive_faithful"
VARIANTS = ["han", "jlens", "jlens_norm"]
N_RANDOM = 6  # rand0..5, stored under concept='gold' rows (both pole fields present)

rows = {}
for line in open(ROWS):
    r = json.loads(line)
    rows[(r["set"], r["concept"], r["variant"], r["lens_layer"])] = r


def own_pole(set_, concept, variant, lens_layer, *, rand_ok=False):
    """Congruent (own-pole) score. Randoms are stored under concept='gold'
    only; their mold null is read from the mold_pole field of those rows
    (documented atlas storage quirk, audit defect #3)."""
    key_concept = "gold" if (rand_ok and set_.startswith("rand")) else concept
    r = rows[(set_, key_concept, variant, lens_layer)]
    return r["gold_pole"] if concept == "gold" else r["mold_pole"]


def opp_pole(set_, concept, variant, lens_layer, *, rand_ok=False):
    key_concept = "gold" if (rand_ok and set_.startswith("rand")) else concept
    r = rows[(set_, key_concept, variant, lens_layer)]
    return r["mold_pole"] if concept == "gold" else r["gold_pole"]


def agg(fn, set_, concept, variant, layers, rand_ok=False):
    vals = [fn(set_, concept, variant, l, rand_ok=rand_ok) for l in layers]
    return sum(vals) / len(vals)


# ---- 1. Pin the convention by reproducing the audit's 12 own-pole cells ----
audit = json.load(open(AUDIT))
target = audit["atlas_ratio_instability"]

candidates = {
    # (treatment lens layers {gold, mold}, band lens layers)
    "lens=vector-1, band=lens16-31": ({"gold": 20, "mold": 23}, list(range(16, 32))),
    "lens=vector-1, band=lens15-30": ({"gold": 20, "mold": 23}, list(range(15, 31))),
    "lens=vector,   band=lens16-31": ({"gold": 21, "mold": 24}, list(range(16, 32))),
    "lens=vector,   band=lens15-30": ({"gold": 21, "mold": 24}, list(range(15, 31))),
}

pinned = None
for name, (treat, band) in candidates.items():
    ok = True
    for variant, concept in itertools.product(VARIANTS, ["gold", "mold"]):
        for aggname, layers in [("treatment", [treat[concept]]), ("band16-31", band)]:
            v = agg(own_pole, V_SET, concept, variant, layers)
            u = agg(own_pole, U_SET, concept, variant, layers)
            t = target[f"{variant}|{aggname}|{concept}"]
            if abs(v - t["v"]) > 1e-9 or abs(u - t["u"]) > 1e-9:
                ok = False
                break
        if not ok:
            break
    if ok:
        pinned = (name, treat, band)
        break

if pinned is None:
    # Report the closest mismatch for diagnosis rather than guessing.
    print("FAIL: no candidate convention reproduces the audit's 12-cell table.")
    for name, (treat, band) in candidates.items():
        v = agg(own_pole, V_SET, "gold", "jlens", [treat["gold"]])
        print(f"  {name}: jlens|treatment|gold v={v!r} "
              f"(audit: {target['jlens|treatment|gold']['v']!r})")
    sys.exit(1)

conv_name, TREAT, BAND = pinned
print(f"CONVENTION_PINNED: {conv_name}  treat={TREAT} band={BAND[0]}..{BAND[-1]}")

# ---- 2. Full robustness panel: 3 variants x 3 aggregations x 2 estimands ----
# Convention finding (verified against atlas.py:129): row lens_layer=l uses
# V[l+1], i.e. lens layer l IS the stream position of vector layer l+1. The
# audit's 'treatment' cells (lens 21/24) therefore sit ONE position
# downstream of the true treatment position (vector 21/24 = lens 20/23).
# Both are reported: 'treatment_audit' (matches audit table exactly) and
# 'treatment_stream' (the true v_gold@21 / v_mold@24 stream position).
TREAT_STREAM = {"gold": TREAT["gold"] - 1, "mold": TREAT["mold"] - 1}
panel = {}
for variant, concept in itertools.product(VARIANTS, ["gold", "mold"]):
    for aggname, layers in [("treatment_audit", [TREAT[concept]]),
                            ("treatment_stream", [TREAT_STREAM[concept]]),
                            ("band16-31", BAND)]:
        v_own = agg(own_pole, V_SET, concept, variant, layers)
        u_own = agg(own_pole, U_SET, concept, variant, layers)
        v_diff = v_own - agg(opp_pole, V_SET, concept, variant, layers)
        u_diff = u_own - agg(opp_pole, U_SET, concept, variant, layers)
        panel[f"{variant}|{aggname}|{concept}"] = {
            "v_own_pole": v_own, "u_own_pole": u_own,
            "own_pole_ratio": v_own / u_own if u_own != 0 else None,
            "own_pole_inverted": (v_own / u_own < 1.0) if u_own > 0 else None,
            "v_pole_diff": v_diff, "u_pole_diff": u_diff,
            "pole_diff_ratio": v_diff / u_diff if u_diff != 0 else None,
        }

# ---- 3. Primary estimand + 6-random null (perm floor 0.143 until F3) ----
def null_and_stats(concept, variant, layers):
    null = [agg(own_pole, f"rand{i}", concept, variant, layers, rand_ok=True)
            for i in range(N_RANDOM)]
    mean = sum(null) / len(null)
    sd = (sum((x - mean) ** 2 for x in null) / (len(null) - 1)) ** 0.5
    out = {}
    for set_, label in [(V_SET, "v"), (U_SET, "u")]:
        val = agg(own_pole, set_, concept, variant, layers)
        n_ge = sum(1 for x in null if x >= val)
        out[label] = {
            "band_mean": val,
            "z_vs_null": (val - mean) / sd if sd else None,
            "n_randoms_ge": n_ge,
            "perm_p": (n_ge + 1) / (N_RANDOM + 1),
            "per_layer": {str(l): own_pole(set_, concept, variant, l)
                          for l in layers},
        }
    out["null"] = {"values": null, "mean": mean, "sd": sd,
                   "min": min(null), "max": max(null)}
    return out

PRIMARY_VARIANT = "jlens"
primary = {c: null_and_stats(c, PRIMARY_VARIANT, BAND) for c in ["gold", "mold"]}
for c in ["gold", "mold"]:
    v, u = primary[c]["v"]["band_mean"], primary[c]["u"]["band_mean"]
    primary[c]["own_pole_ratio_v_over_u"] = v / u

result = {
    "estimand_name": ("own-pole congruent score, J-lens readout ('jlens' "
                      "variant), mean over workspace band L16-31 (lens layers)"),
    "estimand_rationale": [
        "pole-difference ratio rejected: denominator (a difference) crosses "
        "zero for naive axes -> range -9.1x..+47.6x across conventions; a "
        "ratio whose denominator can vanish is not a usable estimand",
        "band-averaging avoids privileging a single treatment layer",
        "L16-31 is the band the instrument was validated on (band_stats "
        "persistence above null), chosen on instrument grounds not effect size",
    ],
    "readout_variant": PRIMARY_VARIANT,
    "layer_convention": conv_name,
    "layer_convention_note": (
        "atlas.py:129 reads V[l+1] at lens layer l, so lens layer l == "
        "stream position of block-input vector layer l+1. The audit's "
        "'treatment' cells (lens 21/24) are one position downstream of the "
        "true treatment position (vector 21/24 = lens 20/23); both appear "
        "in the robustness panel as treatment_audit / treatment_stream. "
        "Band 'L16-31' means lens layers 16-31 (= vector layers 17-32), "
        "matching band_stats' lens-layer convention."),
    "treatment_lens_layers_audit": TREAT,
    "treatment_lens_layers_stream": {"gold": TREAT["gold"] - 1,
                                     "mold": TREAT["mold"] - 1},
    "band_lens_layers": [BAND[0], BAND[-1]],
    "primary": primary,
    "robustness_panel_all_conventions": panel,
    "cells_inverted_own_pole": sorted(
        k for k, d in panel.items()
        if d["own_pole_inverted"]),
    "n_random": N_RANDOM,
    "perm_p_floor_at_n6": round(1 / (N_RANDOM + 1), 4),
    "note_random_storage": ("randoms stored under concept='gold' rows only; "
                            "mold null read from mold_pole field of those rows "
                            "(audit defect #3, documented here)"),
    "audit_reproduction": "12/12 own-pole cells match audit_findings.json to 1e-9",
    "sources": {"rows": "results/atlas_rows.jsonl (1260 rows, unmodified)",
                "v": V_SET, "u": U_SET},
}

json.dump(result, open(OUT, "w"), indent=1)
print(f"WROTE {OUT}")
for c in ["gold", "mold"]:
    p = primary[c]
    print(f"  {c}: v={p['v']['band_mean']:.4f} (z={p['v']['z_vs_null']:+.1f}, "
          f"p={p['v']['perm_p']:.3f})  u={p['u']['band_mean']:.4f} "
          f"(z={p['u']['z_vs_null']:+.1f}, p={p['u']['perm_p']:.3f})  "
          f"ratio={p['own_pole_ratio_v_over_u']:.2f}x")
print("inverted cells:", result["cells_inverted_own_pole"])
