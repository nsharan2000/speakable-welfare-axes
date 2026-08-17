#!/usr/bin/env python3
"""Same-vector ACROSS-MODEL Jaccard baseline (gates §4.7's overlap claim).

The across-model token overlap compares the SAME axis vector decomposed
under two models' lenses, so the correct null is the same random vector
decomposed under both — not random pairs (j5_paired_baseline showed the
same-vector effect inflates across-lens-target overlap to 0.134). This
decomposes mech_components.npz's exact rand_* vectors under the PUBLIC
Qwen3-4B lens and Jaccards against their 2507-lens tokens.
Emits results/j6_paired_baseline.json. ~2-3 min.
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
from dm_paths import MECH_COMPONENTS, MECH_DECOMPOSITIONS  # noqa: E402
from routing_lib import (  # noqa: E402
    JSpace,
    assert_decompositions_converged,
    assert_source_version,
    decomposition_record,
    make_valid_mask,
    unembed_weight,
)

RES = os.path.join(HERE, "results")
NPZ = MECH_COMPONENTS
SRC = MECH_DECOMPOSITIONS
K = 16
LENSL = {"gold": 20, "mold": 23}


def main():
    t0 = time.time()
    tok = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
    hf = transformers.AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-4B", dtype=torch.bfloat16).cuda()
    mask = make_valid_mask(tok, unembed_weight(hf).shape[0])
    lens = jlens.JacobianLens.from_pretrained(
        "neuronpedia/jacobian-lens",
        filename="qwen3-4b/jlens/Salesforce-wikitext/Qwen3-4B_jacobian_lens.pt")
    z = np.load(NPZ)
    src = json.load(open(SRC))
    assert_source_version(src, SRC)

    def norm_tok(s):
        return s.strip().lower()

    def jac(a, b):
        return len(a & b) / len(a | b)

    jsp = {c: JSpace(lens, hf, LENSL[c], mask) for c in LENSL}
    out = {"k": K, "pairs": {}}
    vals = []
    for c in ["gold", "mold"]:
        for ri in range(8):
            name = f"rand_{c}{ri}"
            d = jsp[c].decompose(z[f"{name}__full"], k=K)
            record = decomposition_record(d, tok)
            B = {norm_tok(token) for token in record["tokens_by_coefficient"]}
            A = {norm_tok(s) for s in src[name].get(
                "tokens_by_coefficient", src[name]["tokens"])}
            jv = jac(A, B)
            out["pairs"][name] = {
                "jaccard": jv,
                "shared": sorted(A & B),
                "decomposition": record,
            }
            vals.append(jv)
            print(f"{name}: {jv:.3f}", flush=True)
    arr = np.array(vals)
    out["baseline"] = {
        "mean": float(arr.mean()), "sd": float(arr.std(ddof=1)),
        "min": float(arr.min()), "max": float(arr.max()), "n": len(vals),
        "n_ge_0185": int((arr >= 0.185).sum()),
        "meaning": "same random vector decomposed under 2507-final lens vs "
                   "Qwen3-4B public lens — the correct null for the "
                   "across-model axis token overlap"}
    assert_decompositions_converged(out, context="j6_paired_baseline.json")
    json.dump(out, open(os.path.join(RES, "j6_paired_baseline.json"), "w"),
              indent=1)
    print("J6_PAIRED_BASELINE " + json.dumps(out["baseline"]), flush=True)
    print(f"J6B_DONE {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
