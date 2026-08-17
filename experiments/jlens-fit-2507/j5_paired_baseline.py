#!/usr/bin/env python3
"""Paired across-lens-target Jaccard baseline (figure/report correction).

Comparing an axis's J-component tokens under the final- vs penultimate-
target lens needs a baseline of the SAME KIND: the same random direction
decomposed under both lenses (random-pair Jaccard is the wrong null — the
vector identity is shared, so overlap is expected above pair-chance).
Uses the exact rand_* vectors from mech_components.npz (identity-stable).
Emits results/j5_paired_baseline.json. ~1-2 min.
"""
import json
import os
import sys
import time

import numpy as np
import torch
import transformers

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "common"))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "routing-core"))
import jlens  # noqa: E402
from dm_paths import MECH_COMPONENTS  # noqa: E402
from routing_lib import (  # noqa: E402
    JSpace,
    assert_decompositions_converged,
    decomposition_record,
    make_valid_mask,
    unembed_weight,
)

RES = os.path.join(HERE, "results")
NPZ = MECH_COMPONENTS
K = 16
LENSL = {"gold": 20, "mold": 23}


def main():
    t0 = time.time()
    tok = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Instruct-2507")
    hf = transformers.AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-4B-Instruct-2507", dtype=torch.bfloat16).cuda()
    mask = make_valid_mask(tok, unembed_weight(hf).shape[0])
    z = np.load(NPZ)
    lenses = {n: jlens.JacobianLens.load(os.path.join(RES, f)) for n, f in
              [("final", "Qwen3-4B-Instruct-2507_jacobian_lens.pt"),
               ("penult", "Qwen3-4B-Instruct-2507_jacobian_lens_penult.pt")]}

    def norm_tok(s):
        return s.strip().lower()

    def jac(a, b):
        return len(a & b) / len(a | b)

    out = {"k": K, "pairs": {}}
    vals = []
    jsp = {ln: {c: JSpace(L, hf, LENSL[c], mask) for c in LENSL}
           for ln, L in lenses.items()}
    for c in ["gold", "mold"]:
        for ri in range(8):
            vec = z[f"rand_{c}{ri}__full"]
            sets = {}
            records = {}
            for ln in lenses:
                d = jsp[ln][c].decompose(vec, k=K)
                records[ln] = decomposition_record(d, tok)
                sets[ln] = {
                    norm_tok(token)
                    for token in records[ln]["tokens_by_coefficient"]
                }
            jv = jac(sets["final"], sets["penult"])
            out["pairs"][f"rand_{c}{ri}"] = {
                "jaccard": jv,
                "shared": sorted(sets["final"] & sets["penult"]),
                "decompositions": records,
            }
            vals.append(jv)
            print(f"rand_{c}{ri}: {jv:.3f}", flush=True)
    arr = np.array(vals)
    out["baseline"] = {"mean": float(arr.mean()), "sd": float(arr.std(ddof=1)),
                       "min": float(arr.min()), "max": float(arr.max()),
                       "n": len(vals),
                       "meaning": "same random vector decomposed under both "
                                  "lens targets — the correct null for "
                                  "across-target axis overlap"}
    assert_decompositions_converged(out, context="j5_paired_baseline.json")
    json.dump(out, open(os.path.join(RES, "j5_paired_baseline.json"), "w"),
              indent=1)
    print("PAIRED_BASELINE " + json.dumps(out["baseline"]), flush=True)
    print(f"J5B_DONE {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
