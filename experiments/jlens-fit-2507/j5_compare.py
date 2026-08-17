#!/usr/bin/env python3
"""J5 — lens-variant robustness: final-target vs penultimate-target lens.

The known target-layer discrepancy (our fit targets the final layer per the
Neuronpedia convention; the paper's stated default for Claude models is the
penultimate layer) is the #1 methodological critique we'd face. This
quantifies whether it matters for OUR claims:

  1. J-share (k=16) of v/u at each pole's treatment lens layer under the
     penultimate lens, + 12 per-name-seeded randoms per polarity
     -> does the trained-above-chance / naive-at-chance split hold?
  2. Own-pole band scores (F4 estimand, norm-matched u) under the
     penultimate lens -> do the F3 ratios (gold inverted 0.57x, mold 2.30x)
     hold?
  3. Per-layer agreement between the two lenses (Pearson/Spearman on band
     pole scores) + Jaccard of the J-component token sets across lenses.

Emits results/j5_lens_comparison.json. Cost ~5-10 min.
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
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "jlens-atlas"))
import jlens  # noqa: E402
from dm_paths import ARTIFACTS  # noqa: E402
from routing_lib import (  # noqa: E402
    JSpace,
    assert_decompositions_converged,
    decomposition_record,
    make_valid_mask,
    unembed_weight,
)
from atlas import GOLD_POLE, MOLD_POLE, CONTROL_NOUNS, first_tok_ids  # noqa: E402

RES = os.path.join(HERE, "results")
ART = ARTIFACTS
LENS_FINAL = os.path.join(RES, "Qwen3-4B-Instruct-2507_jacobian_lens.pt")
LENS_PENULT = os.path.join(RES, "Qwen3-4B-Instruct-2507_jacobian_lens_penult.pt")
OUT = os.path.join(RES, "j5_lens_comparison.json")

BAND = list(range(16, 32))
LENSL = {"gold": 20, "mold": 23}
K = 16
N_RAND = 12


def rand_dir(pol, ri, shape):
    rng = np.random.default_rng([55, pol, ri])
    r = torch.tensor(rng.standard_normal(shape).astype(np.float32))
    return r


def main():
    t0 = time.time()
    tok = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Instruct-2507")
    hf = transformers.AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-4B-Instruct-2507", dtype=torch.bfloat16).cuda()
    W = unembed_weight(hf).float().cuda()
    mask = make_valid_mask(tok, W.shape[0])
    gids = first_tok_ids(tok, GOLD_POLE)
    mids = first_tok_ids(tok, MOLD_POLE)
    cids = first_tok_ids(tok, CONTROL_NOUNS)

    v95 = torch.load(f"{ART}/vectors_step95_bal.pt", map_location="cpu",
                     weights_only=False)
    uf = torch.load(f"{ART}/vectors_naive_faithful_pc5000.pt",
                    map_location="cpu", weights_only=False)
    Vg, Vm = v95["v_gold"].float(), v95["v_mold"].float()
    Ug = uf["v_gold"].float() / (uf["v_gold"].float().norm(dim=1, keepdim=True) + 1e-9) \
        * Vg.norm(dim=1, keepdim=True)
    Um = uf["v_mold"].float() / (uf["v_mold"].float().norm(dim=1, keepdim=True) + 1e-9) \
        * Vm.norm(dim=1, keepdim=True)
    dirs = {"v_gold": ("gold", Vg), "u_gold": ("gold", Ug),
            "v_mold": ("mold", Vm), "u_mold": ("mold", Um)}
    ref = {"gold": Vg, "mold": Vm}

    lenses = {"final": jlens.JacobianLens.load(LENS_FINAL),
              "penult": jlens.JacobianLens.load(LENS_PENULT)}
    for name, L in lenses.items():
        assert all(torch.isfinite(J).all() for J in L.jacobians.values()), name
    print(f"loaded {time.time()-t0:.0f}s", flush=True)

    out = {"lens_files": {"final": LENS_FINAL, "penult": LENS_PENULT},
           "band_lens_layers": [BAND[0], BAND[-1]], "k": K,
           "n_random_per_polarity": N_RAND}

    # ---- 1+2: pole scores per layer, both lenses ----
    def pole_scores(lens, name, V, c):
        rows = {}
        ids = gids if c == "gold" else mids
        for l in BAND:
            v_row = V[l + 1].cuda() if l + 1 < V.shape[0] else V[-1].cuda()
            J = lens.jacobians[l].cuda().float()
            lp = torch.log_softmax(W @ (J @ v_row), -1)
            rows[l] = float(lp[ids].mean() - lp[cids].mean())
        return rows

    per_layer = {ln: {n: pole_scores(L, n, V, c)
                      for n, (c, V) in dirs.items()}
                 for ln, L in lenses.items()}
    out["band_scores"] = {}
    for ln in lenses:
        band = {n: float(np.mean(list(per_layer[ln][n].values()))) for n in dirs}
        out["band_scores"][ln] = {
            **{n: round(v, 4) for n, v in band.items()},
            "ratio_gold": band["v_gold"] / band["u_gold"],
            "ratio_mold": band["v_mold"] / band["u_mold"]}

    # per-layer agreement between lenses
    agree = {}
    for n in dirs:
        a = np.array([per_layer["final"][n][l] for l in BAND])
        b = np.array([per_layer["penult"][n][l] for l in BAND])
        agree[n] = {"pearson": float(np.corrcoef(a, b)[0, 1]),
                    "mean_abs_diff": float(np.abs(a - b).mean())}
    out["per_layer_agreement"] = agree
    print("band scores + agreement done", flush=True)

    # ---- 3: J-share under both lenses + null under penult ----
    jshare = {}
    tokens = {}
    decomposition_records = {}
    for ln, L in lenses.items():
        jsp = {c: JSpace(L, hf, LENSL[c], mask) for c in LENSL}
        jshare[ln] = {}
        tokens[ln] = {}
        decomposition_records[ln] = {}
        for n, (c, V) in dirs.items():
            d = jsp[c].decompose(V[LENSL[c] + 1].cpu().numpy(), k=K)
            record = decomposition_record(d, tok)
            decomposition_records[ln][n] = record
            jshare[ln][n] = d["var_fraction"]
            tokens[ln][n] = record["tokens_by_coefficient"]
        if ln == "penult":
            null = {c: [] for c in LENSL}
            null_records = {c: [] for c in LENSL}
            for c, pol in [("gold", 0), ("mold", 1)]:
                nv = float(ref[c][LENSL[c] + 1].norm())
                for ri in range(N_RAND):
                    r = rand_dir(pol, ri, (2560,))
                    r = r / r.norm() * nv
                    d = jsp[c].decompose(r.numpy(), k=K)
                    null[c].append(d["var_fraction"])
                    null_records[c].append(decomposition_record(d, tok))
            out["penult_null"] = {
                c: {"mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1)),
                    "n": N_RAND, "values": v,
                    "decompositions": null_records[c]}
                for c, v in null.items()}
        print(f"jshare {ln} done ({(time.time()-t0)/60:.1f}m)", flush=True)
    out["jshare"] = jshare
    out["decompositions"] = decomposition_records
    for n, (c, _) in dirs.items():
        nl = out["penult_null"][c]
        vf = jshare["penult"][n]
        n_ge = sum(1 for x in nl["values"] if x >= vf)
        out["jshare"].setdefault("penult_stats", {})[n] = {
            "z": (vf - nl["mean"]) / nl["sd"],
            "perm_p": (n_ge + 1) / (N_RAND + 1)}

    def norm_tok(s):
        return s.strip().lower()

    out["token_overlap_across_lenses"] = {}
    for n in dirs:
        A = {norm_tok(s) for s in tokens["final"][n]}
        B = {norm_tok(s) for s in tokens["penult"][n]}
        out["token_overlap_across_lenses"][n] = {
            "jaccard": len(A & B) / len(A | B),
            "shared": sorted(A & B)}
    out["tokens"] = tokens

    assert_decompositions_converged(out, context="j5_lens_comparison.json")
    json.dump(out, open(OUT, "w"), indent=1)
    print("J5_SUMMARY " + json.dumps({
        "band_scores": out["band_scores"],
        "jshare_final": {n: round(v, 4) for n, v in jshare["final"].items()
                         if isinstance(v, float)},
        "jshare_penult": {n: round(v, 4) for n, v in jshare["penult"].items()
                          if isinstance(v, float)},
        "overlap": {n: round(d["jaccard"], 3)
                    for n, d in out["token_overlap_across_lenses"].items()}}),
        flush=True)
    print(f"J5_COMPARE_DONE {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
