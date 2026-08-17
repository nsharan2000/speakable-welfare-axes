#!/usr/bin/env python3
"""R6 (audit return-request) — one table: norms + pairwise cosines for every
direction actually used, at each pole's treatment layer (block-input vector
convention). CPU-only, ~seconds. Writes results/R6_direction_table.json."""
import itertools
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "common"))
from dm_paths import ARTIFACTS, LANG_DIR_FILE  # noqa: E402

RES = os.path.join(HERE, "results")
ART = ARTIFACTS
LANG = LANG_DIR_FILE


def row(t, L):
    return t.float()[L].numpy()


def main():
    v95 = torch.load(f"{ART}/vectors_step95_bal.pt", map_location="cpu",
                     weights_only=False)
    uf = torch.load(f"{ART}/vectors_naive_faithful_pc5000.pt",
                    map_location="cpu", weights_only=False)
    own_p = f"{ART}/vectors_own_naive.pt"
    own = torch.load(own_p, map_location="cpu", weights_only=False) \
        if os.path.exists(own_p) else None

    Lg, Lm = int(v95["layer_gold"]), int(v95["layer_mold"])
    out = {"treatment_layers_block_input": {"gold": Lg, "mold": Lm},
           "per_pole": {}}

    for c, L in [("gold", Lg), ("mold", Lm)]:
        vecs = {
            f"v_{c}": row(v95[f"v_{c}"], L),
            f"u_faithful_{c}": row(uf[f"v_{c}"], L),
        }
        if own is not None:
            vecs[f"own_u_{c}"] = row(own[f"v_{c}"], L)
        if os.path.exists(LANG) and c == "gold":
            vecs["lang_fr@L18"] = np.load(LANG)  # its own layer; noted
        # random cohort mean norm: unit gaussians scaled to ||v|| exactly as
        # the experiments draw them -> norm == ||v|| by construction; state it
        norms = {k: float(np.linalg.norm(x)) for k, x in vecs.items()}
        cosines = {}
        for a, b in itertools.combinations(vecs, 2):
            va, vb = vecs[a], vecs[b]
            cosines[f"{a} vs {b}"] = float(
                np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-12))
        out["per_pole"][c] = {"layer": L, "norms": norms, "cosines": cosines,
                              "random_cohort_norm": norms[f"v_{c}"],
                              "random_note": "randoms are unit gaussians "
                              "rescaled to ||v|| (expected cos vs any fixed "
                              "direction ~ N(0, 1/2560), i.e. |cos|~0.02)"}
    json.dump(out, open(os.path.join(RES, "R6_direction_table.json"), "w"),
              indent=1)
    print("R6_DONE")
    for c in out["per_pole"]:
        print(c, json.dumps(out["per_pole"][c]["cosines"], indent=1))


if __name__ == "__main__":
    main()
