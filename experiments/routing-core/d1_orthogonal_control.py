#!/usr/bin/env python3
"""D1 (audit round 2) — orthogonalized control: the experiment R6 demands.

u_perp = u - (u.vhat)vhat at each pole's block-input treatment layer
(gold L=21, mold L=24), renormalized to ||v||. Three readouts:
  1. J-share (k=16) at the treatment lens layer, scored against the STORED
     n=100 per-polarity null from jshare_cohort_n100.json (no new randoms —
     keeps u_perp comparable to the published v/u numbers).
  2. F4-estimand pole score over band lens 16-31 (own-pole congruent, jlens
     readout), u_perp built PER LAYER (orthogonalize u[l+1] against v[l+1],
     renorm to ||v[l+1]|| — the house per-layer norm-matching convention),
     scored against F3's stored n=100 null.
  3. Sentiment: 16 NEUTRAL_PROMPTS x 2 poles x alpha=+4 at the treatment
     position, 60-tok gens -> d1_gen_rows.jsonl for blind judging (rubric
     R2 verbatim, judged off-box; sentiment fields left pending here).

Decision rule (pre-specified in D1-orthogonalized-control.md, committed
BEFORE looking): J-share(u_perp) vs stored null —
  perm p > 0.05  => verdict "shared_subspace"
  perm p < 0.05  => verdict "pre_existing"
  p < 0.05 AND var_fraction > u's own => "anomalous" (re-check asserts first)

Verification gates before trusting anything:
  - artifact layers must be gold 21 / mold 24 (D1 spec hardcodes these);
  - v_gold/v_mold var_fraction recomputed fresh must match
    jshare_cohort_n100.json to 1e-6 (same code path proof);
  - |cos(u_perp, v)| < 1e-5 and | ||u_perp|| - ||v|| | < 1e-3 per pole.

Writes ONLY new files: results/d1_orthogonal_control.json,
results/d1_gen_rows.jsonl. Never touches mech_decompositions*.json.
Cost: 4 decomps (2 gates + 2 u_perp) + 32 atlas rows + 32 gens ~= 2 min GPU.
Run inside dm-exp: /workspace/venvs/jlens/bin/python d1_orthogonal_control.py
"""
import json
import os
import sys
import time

import numpy as np
import torch
import transformers

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/workspace/experiments/common")
import jlens  # noqa: E402
from routing_lib import JSpace, make_valid_mask, unembed_weight  # noqa: E402
from dm_common import chat_ids, steering  # noqa: E402
from prompts import NEUTRAL_PROMPTS  # noqa: E402
sys.path.insert(0, "/workspace/experiments/jlens-atlas")
from atlas import GOLD_POLE, MOLD_POLE, CONTROL_NOUNS, first_tok_ids  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
LENS_FILE = "/workspace/experiments/jlens-fit-2507/results/Qwen3-4B-Instruct-2507_jacobian_lens.pt"
ART = "/workspace/welfare-vectors/artifacts"

K = 16
ALPHA = 4.0
BAND = list(range(16, 32))          # lens layers (= vector layers 17-32)
EXPECT_LAYERS = {"gold": 21, "mold": 24}


def chat_tensor(tok, msg):
    """chat_ids returns a BatchEncoding under the venv's transformers
    (apply_chat_template return-type changed); normalize to the ids tensor."""
    ids = chat_ids(tok, msg)
    return ids if torch.is_tensor(ids) else ids["input_ids"]


def orth(u_row, v_row):
    """u_perp = u - (u.vhat)vhat, renormalized to ||v|| (D1 spec verbatim)."""
    vhat = v_row / np.linalg.norm(v_row)
    up = u_row - np.dot(u_row, vhat) * vhat
    kept = np.linalg.norm(up) / (np.linalg.norm(u_row) + 1e-12)
    up = up / (np.linalg.norm(up) + 1e-12) * np.linalg.norm(v_row)
    return up.astype(np.float32), float(kept)


def main():
    t0 = time.time()
    hf = transformers.AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-4B-Instruct-2507", dtype=torch.bfloat16).cuda()
    tok = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Instruct-2507")
    lens = jlens.JacobianLens.load(LENS_FILE)
    assert all(torch.isfinite(J).all() for J in lens.jacobians.values())
    mask = make_valid_mask(tok, unembed_weight(hf).shape[0])
    W = unembed_weight(hf).float().cuda()
    print(f"loaded {time.time()-t0:.0f}s", flush=True)

    v = torch.load(f"{ART}/vectors_step95_bal.pt", map_location="cpu",
                   weights_only=False)
    u = torch.load(f"{ART}/vectors_naive_faithful_pc5000.pt", map_location="cpu",
                   weights_only=False)
    for c in ["gold", "mold"]:
        assert int(v[f"layer_{c}"]) == EXPECT_LAYERS[c], \
            f"artifact layer_{c}={int(v[f'layer_{c}'])} != spec {EXPECT_LAYERS[c]}"

    f2 = json.load(open(os.path.join(RES, "jshare_cohort_n100.json")))
    f3 = json.load(open(
        "/workspace/experiments/jlens-atlas/results/atlas_cohort_n100.json"))

    out = {"layers": EXPECT_LAYERS,
           "construction": "u_perp = u - (u.vhat)vhat, renormalized to ||v||",
           "k": K, "alpha": ALPHA,
           "null_source": "jshare_cohort_n100.json stored n=100 per-polarity "
                          "null (no new randoms, per D1 spec)",
           "band_convention": "pole score: u_perp built per band layer "
                              "(orthogonalize u[l+1] vs v[l+1], renorm to "
                              "||v[l+1]||); J-share/gens: treatment layer only",
           "checks": {}, "jshare": {}, "reference": {}, "pole_score_band": {},
           "sentiment": {"status": "pending_judge",
                         "rows_file": "d1_gen_rows.jsonl",
                         "rubric": "R2_judge_rubric.json verbatim, blind"}}

    # reference block from stored files (single source, per R11)
    for t in ["v_gold", "u_gold", "v_mold", "u_mold"]:
        out["reference"][t] = round(f2["targets"][t]["var_fraction"], 4)

    jspaces = {}
    gen_rows = []
    for c in ["gold", "mold"]:
        L = EXPECT_LAYERS[c]
        lensL = L - 1
        v_row = v[f"v_{c}"].float()[L].numpy().astype(np.float32)
        u_row = u[f"v_{c}"].float()[L].numpy().astype(np.float32)
        up, kept = orth(u_row, v_row)

        cosv = float(np.dot(up, v_row) /
                     (np.linalg.norm(up) * np.linalg.norm(v_row)))
        cosu = float(np.dot(up, u_row) /
                     (np.linalg.norm(up) * np.linalg.norm(u_row)))
        nr = float(np.linalg.norm(up) / np.linalg.norm(v_row))
        assert abs(cosv) < 1e-5, f"{c}: cos(u_perp,v)={cosv} not ~0"
        assert abs(np.linalg.norm(up) - np.linalg.norm(v_row)) < 1e-3, \
            f"{c}: norm mismatch after rescale"
        out["checks"][c] = {
            "cos_u_perp_v": cosv, "cos_u_perp_u": cosu, "norm_ratio": nr,
            "norm_u_perp": float(np.linalg.norm(up)),
            "direction_retained_frac": kept,          # ||u_perp||/||u|| = sin
            "sq_norm_retained_frac": kept ** 2,       # R11's squared version
        }

        if lensL not in jspaces:
            jspaces[lensL] = JSpace(lens, hf, lensL, mask)
        js = jspaces[lensL]

        # gate: recompute v fresh, must match stored file (same code path)
        dv = js.decompose(v_row, k=K)
        stored = f2["targets"][f"v_{c}"]["var_fraction"]
        if abs(dv["var_fraction"] - stored) > 1e-6:
            print(f"D1_FAIL: fresh v_{c} var_fraction {dv['var_fraction']:.6f} "
                  f"!= stored {stored:.6f} — code path drifted, aborting",
                  flush=True)
            sys.exit(1)

        d = js.decompose(up, k=K)
        null = np.array(f2["null"][c]["values"])
        assert len(null) == 100
        vf = d["var_fraction"]
        n_ge = int((null >= vf).sum())
        out["jshare"][f"u_perp_{c}"] = {
            "var_fraction": vf,
            "z": float((vf - null.mean()) / null.std(ddof=1)),
            "n_randoms_ge": n_ge,
            "perm_p": (n_ge + 1) / (len(null) + 1),
            "tokens": [tok.decode([t]) for t in d["token_ids"]],
            "cos_x_xj": d["cos_x_xj"],
        }
        print(f"D1 jshare u_perp_{c}: vf={vf:.4f} n_ge={n_ge} "
              f"({(time.time()-t0)/60:.1f} min)", flush=True)

    # ---- pole score over the band (F4 estimand, per-layer u_perp) ----
    gids = first_tok_ids(tok, GOLD_POLE)
    mids = first_tok_ids(tok, MOLD_POLE)
    cids = first_tok_ids(tok, CONTROL_NOUNS)
    pole_ids = {"gold": gids, "mold": mids}
    for c in ["gold", "mold"]:
        Vfull = v[f"v_{c}"].float()
        Ufull = u[f"v_{c}"].float()
        per_layer = {}
        vals = []
        for l in BAND:
            vi = min(l + 1, Vfull.shape[0] - 1)
            up_l, _ = orth(Ufull[vi].numpy().astype(np.float32),
                           Vfull[vi].numpy().astype(np.float32))
            J = lens.jacobians[l].cuda().float()
            lp = torch.log_softmax(W @ (J @ torch.tensor(up_l).cuda()), -1)
            own = float(lp[pole_ids[c]].mean() - lp[cids].mean())
            per_layer[str(l)] = own
            vals.append(own)
        band_mean = float(np.mean(vals))
        null = np.array(f3["null"][c]["values"])
        n_ge = int((null >= band_mean).sum())
        out["pole_score_band"][f"u_perp_{c}"] = {
            "band_mean": band_mean,
            "z": float((band_mean - null.mean()) / null.std(ddof=1)),
            "n_randoms_ge": n_ge,
            "perm_p": (n_ge + 1) / (len(null) + 1),
            "per_layer": per_layer,
            "reference_band_means": {
                "v": f3["targets"][f"v_{c}"]["band_mean"],
                "u_matched": f3["targets"][f"u_{c}"]["band_mean"]},
        }
        print(f"D1 pole band u_perp_{c}: {band_mean:.4f} n_ge={n_ge}",
              flush=True)

    # ---- sentiment generations (judged off-box) ----
    rows_path = os.path.join(RES, "d1_gen_rows.jsonl")
    done = set()
    if os.path.exists(rows_path):
        for line in open(rows_path):
            r = json.loads(line)
            done.add((r["concept"], r["prompt"]))
    with open(rows_path, "a") as f:
        for c in ["gold", "mold"]:
            L = EXPECT_LAYERS[c]
            v_row = v[f"v_{c}"].float()[L].numpy().astype(np.float32)
            u_row = u[f"v_{c}"].float()[L].numpy().astype(np.float32)
            up, _ = orth(u_row, v_row)
            svec = up * ALPHA
            for p in NEUTRAL_PROMPTS[:16]:
                if (c, p) in done:
                    continue
                ids = chat_tensor(tok, p)
                with torch.no_grad(), steering(hf, L - 1, svec, 1.0):
                    o = hf.generate(ids, max_new_tokens=60, do_sample=False,
                                    pad_token_id=tok.eos_token_id)
                txt = tok.decode(o[0, ids.shape[1]:], skip_special_tokens=True)
                row = {"concept": c, "arm": "u_perp", "alpha": ALPHA,
                       "prompt": p, "text": txt}
                gen_rows.append(row)
                f.write(json.dumps(row) + "\n")
                f.flush()
            print(f"D1 gens {c} done ({(time.time()-t0)/60:.1f} min)",
                  flush=True)

    # ---- verdict (pre-specified mapping, J-share based) ----
    verdicts = {}
    for c in ["gold", "mold"]:
        p = out["jshare"][f"u_perp_{c}"]["perm_p"]
        vf = out["jshare"][f"u_perp_{c}"]["var_fraction"]
        u_own = out["reference"][f"u_{c}"]
        if p < 0.05 and vf > u_own:
            verdicts[c] = "anomalous"
        elif p < 0.05:
            verdicts[c] = "pre_existing"
        else:
            verdicts[c] = "shared_subspace"
    out["verdict"] = verdicts

    json.dump(out, open(os.path.join(RES, "d1_orthogonal_control.json"), "w"),
              indent=1)
    summary = {k: {"vf": round(d["var_fraction"], 4),
                   "perm_p": round(d["perm_p"], 4)}
               for k, d in out["jshare"].items()}
    summary["verdict"] = verdicts
    print("D1_SUMMARY " + json.dumps(summary), flush=True)
    print(f"D1_DONE {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
