#!/usr/bin/env python3
"""F7 (audit) — k-sweep at the treatment stream positions + a real null.

Convention finding (atlas.py:160 — the sweep reads V[l+1] at lens layer l):
ksweep.json keys are LENS layers, so key L23 is already VECTOR layer 24 =
mold's treatment position. The audit's defect statement ("mold L24 never
evaluated") read the keys as vector layers; the real gap is the mirror
image: GOLD's treatment position (vector 21 = lens 20) is the one the odd
lens-layer grid skips. Either way one pole's treatment position was
untested. This script closes both readings:

  - adds lens layers {20, 22, 24} (vector 21/23/25) x k in {4,8,16,25,50}
    x sets {step95, naive_faithful} x concepts {gold, mold}   (60 decomps)
  - adds a 12-random null at k=16 at BOTH treatment lens layers {20, 23}
    (24 decomps), replacing the old 2-random sweep null (p floor 0.333 ->
    0.077). Randoms drawn exactly like atlas.py (default_rng(99), gold-set
    norms) so rand0/rand1 reproduce the existing sweep entries.

Writes results/ksweep_ext.json only (never touches ksweep.json).
Cost: ~84 decompositions x 14.3 s ~= 20 min.
Run inside dm-exp with the jlens venv.
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
from dm_paths import ARTIFACTS, LENS_FILE  # noqa: E402
from routing_lib import (  # noqa: E402
    ROUTING_ALGORITHM_VERSION,
    JSpace,
    assert_decompositions_converged,
    decomposition_record,
    make_valid_mask,
    unembed_weight,
)

RES = os.path.join(HERE, "results")
ART = ARTIFACTS

NEW_LAYERS = [20, 22, 24]          # lens layers = vector layers 21/23/25
NULL_LAYERS = [20, 23]             # both poles' treatment stream positions
KS = [4, 8, 16, 25, 50]
N_RAND = 12
K_NULL = 16


def main():
    t0 = time.time()
    hf = transformers.AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-4B-Instruct-2507", dtype=torch.bfloat16).cuda()
    tok = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Instruct-2507")
    lens = jlens.JacobianLens.load(LENS_FILE)
    mask = make_valid_mask(tok, unembed_weight(hf).shape[0])

    sets = {"step95": f"{ART}/vectors_step95_bal.pt",
            "naive_faithful": f"{ART}/vectors_naive_faithful_pc5000.pt"}
    vecs = {}
    for name, p in sets.items():
        d = torch.load(p, map_location="cpu", weights_only=False)
        for c in ["gold", "mold"]:
            vecs[(name, c)] = d[f"v_{c}"].float()

    # randoms exactly as atlas.py draws them (first 6 == atlas cohort)
    rng = np.random.default_rng(99)
    ref = vecs[("step95", "gold")]
    rand = []
    for ri in range(N_RAND):
        r = torch.tensor(rng.standard_normal(ref.shape).astype(np.float32))
        rand.append(r / r.norm(dim=1, keepdim=True) * ref.norm(dim=1, keepdim=True))

    out_path = os.path.join(RES, "ksweep_ext.json")
    ext = json.load(open(out_path)) if os.path.exists(out_path) else {}
    jspaces = {}

    def js(l):
        if l not in jspaces:
            jspaces[l] = JSpace(lens, hf, l, mask)
        return jspaces[l]

    def dec(key, V, l, k):
        if ext.get(key, {}).get("algorithm_version") == ROUTING_ALGORITHM_VERSION:
            return
        v_row = V[l + 1].cpu().numpy() if l + 1 < V.shape[0] else V[-1].cpu().numpy()
        d = js(l).decompose(v_row, k=k)
        ext[key] = decomposition_record(d, tok)
        with open(out_path, "w") as f:
            json.dump(ext, f, indent=1)

    for (sname, c), V in vecs.items():
        for l in NEW_LAYERS:
            for k in KS:
                dec(f"{sname}|{c}|L{l}|k{k}", V, l, k)
        print(f"ext {sname}/{c} done {time.time()-t0:.0f}s", flush=True)

    for ri, R in enumerate(rand):
        for l in NULL_LAYERS:
            dec(f"rand{ri}|gold|L{l}|k{K_NULL}", R, l, K_NULL)
    print(f"null done {time.time()-t0:.0f}s", flush=True)

    # ---- null block: per-(layer) cohort stats + perm p for each real axis ----
    old = json.load(open(os.path.join(RES, "ksweep.json")))
    both = {**old, **ext}
    nulls = {}
    for l in NULL_LAYERS:
        vals = [ext[f"rand{ri}|gold|L{l}|k{K_NULL}"]["var_fraction"]
                for ri in range(N_RAND)]
        arr = np.array(vals)
        entry = {"values": vals, "mean": float(arr.mean()),
                 "sd": float(arr.std(ddof=1)), "n": N_RAND, "k": K_NULL,
                 "reals": {}}
        for sname in sets:
            for c in ["gold", "mold"]:
                key = f"{sname}|{c}|L{l}|k{K_NULL}"
                if key in both:
                    vf = both[key]["var_fraction"]
                    n_ge = int((arr >= vf).sum())
                    entry["reals"][key] = {
                        "var_fraction": vf,
                        "z": float((vf - arr.mean()) / arr.std(ddof=1)),
                        "perm_p": (n_ge + 1) / (N_RAND + 1)}
        nulls[f"L{l}"] = entry
    ext["_null"] = nulls
    ext["_convention_note"] = (
        "keys are LENS layers; the vector used is V[lens_layer+1], so L20 = "
        "vector layer 21 (gold treatment) and L23 = vector layer 24 (mold "
        "treatment). The audit's 'mold L24 missing' read keys as vector "
        "layers; the actual grid gap was gold's treatment position (lens 20).")
    assert_decompositions_converged(ext, context="ksweep_ext.json")
    with open(out_path, "w") as f:
        json.dump(ext, f, indent=1)
    print("F7_SUMMARY " + json.dumps({k: {r: round(x["z"], 2)
          for r, x in v["reals"].items()} for k, v in nulls.items()}), flush=True)
    print(f"F7_DONE {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
