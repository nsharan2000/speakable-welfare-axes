"""Provenance cross-check: our own u extraction (Han et al.'s exact code, base
model, off-policy trajectories) vs the nickmahdavi naive-faithful u vectors.

Per layer: cosine similarity + norm ratio for gold (goal) and mold (lava).
High mid-layer cosine (with 2000-vs-5000-sample noise) validates the
third-party vectors' provenance. Output: <own_u>/compare_u.json (see dm_paths).
"""

import glob
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "common"))
from dm_paths import ARTIFACTS, OWN_U  # noqa: E402

OWN = OWN_U
ART = ARTIFACTS

nick = torch.load(f"{ART}/vectors_naive_faithful_pc5000.pt", map_location="cpu",
                  weights_only=False)

out = {}
for code, paper in [("goal", "gold"), ("lava", "mold")]:
    # their extractor writes <output>/<concept>/mean_diff.pt (1, n_layers, d)
    cands = glob.glob(f"{OWN}/**/mean_diff.pt", recursive=True)
    cand = [c for c in cands if code in c]
    if not cand:
        out[paper] = {"error": f"no mean_diff.pt found for {code}",
                      "found": cands}
        continue
    own = torch.load(cand[0], map_location="cpu", weights_only=False)
    own = own.squeeze(0).float()          # (n_layers, d)
    ref = nick[f"v_{paper}"].float()      # (36, 2560)
    n_l = min(own.shape[0], ref.shape[0])
    cos, normr = {}, {}
    for l in range(n_l):
        a, b = own[l], ref[l]
        cos[str(l)] = float((a @ b) / (a.norm() * b.norm() + 1e-9))
        normr[str(l)] = float(a.norm() / (b.norm() + 1e-9))
    sel = int(nick[f"layer_{paper}"])
    out[paper] = {
        "file": cand[0], "cos_by_layer": cos, "norm_ratio_by_layer": normr,
        "cos_at_selected_layer": cos.get(str(sel)),
        "mean_cos_L15_30": float(np.mean([cos[str(l)] for l in range(15, min(31, n_l))])),
    }
    print(paper, "cos@sel", out[paper]["cos_at_selected_layer"],
          "meanL15-30", out[paper]["mean_cos_L15_30"])

with open(f"{OWN}/compare_u.json", "w") as f:
    json.dump(out, f, indent=2)
print("U_COMPARE_WRITTEN")
