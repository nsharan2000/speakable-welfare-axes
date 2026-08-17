#!/usr/bin/env python3
"""own_u J-share — two decompositions requested for §4.2/§4.5 support.

Targets (own re-extraction, Han et al. extract_dispatch off_policy n2000,
base model only — see welfare-vectors/own_u/*/metadata.json):
  own_u_gold: own_u/goal/mean_diff.pt  vector index 21 -> lens layer 20
  own_u_mold: own_u/lava/mean_diff.pt  vector index 24 -> lens layer 23

Same construction as f2_jshare_cohort.py: same model, lens, token mask,
k=16, norm-matched to the trained counterpart (a no-op for j_share, which
is scale-invariant, but kept for exact parity with f2's handling of u).
Both own_u tensors are (1, 36, 2560) float64, same indexing as
vectors_step95_bal.pt (36, 2560) — index 21/24 per convert_own_u.py /
compare_u.py convention (lens layer = index - 1).

Verification gate before trusting anything: v_gold and v_mold are
recomputed fresh and must match jshare_cohort_n100.json's stored
var_fraction to 1e-6, or the run aborts (proves identical lens/mask/code
path). Null comparison: the STORED n=100 per-polarity cohorts from
jshare_cohort_n100.json (never regenerated).

Writes ONLY a new file: results/own_u_jshare.json.
Cost: model+lens load + 4 decompositions ~= 4-6 min on the Spark.
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
import jlens  # noqa: E402  (venv: /workspace/venvs/jlens)
from routing_lib import JSpace, make_valid_mask, unembed_weight  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
LENS_FILE = "/workspace/experiments/jlens-fit-2507/results/Qwen3-4B-Instruct-2507_jacobian_lens.pt"
ART = "/workspace/welfare-vectors/artifacts"
OWN_U = "/workspace/welfare-vectors/own_u"
K = 16


def main():
    t0 = time.time()
    cohort = json.load(open(os.path.join(RES, "jshare_cohort_n100.json")))
    assert cohort["k"] == K and len(cohort["null"]["gold"]["values"]) == 100

    hf = transformers.AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-4B-Instruct-2507", dtype=torch.bfloat16).cuda()
    tok = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Instruct-2507")
    lens = jlens.JacobianLens.load(LENS_FILE)
    assert all(torch.isfinite(J).all() for J in lens.jacobians.values())
    mask = make_valid_mask(tok, unembed_weight(hf).shape[0])
    print(f"loaded {time.time()-t0:.0f}s", flush=True)

    v = torch.load(f"{ART}/vectors_step95_bal.pt", map_location="cpu",
                   weights_only=False)
    own = {c: torch.load(f"{OWN_U}/{d}/mean_diff.pt", map_location="cpu",
                         weights_only=False).squeeze(0).float()
           for c, d in [("gold", "goal"), ("mold", "lava")]}

    out = {"k": K, "metric": "j_share = ||x_j||^2 / ||x||^2 (scale-invariant)",
           "null_source": "jshare_cohort_n100.json stored cohorts (n=100/polarity)",
           "own_u_source": "welfare-vectors/own_u (extract_dispatch off_policy, "
                           "n2000/tile, base model, seed 474747)",
           "verification_gate": {}, "targets": {}}

    for c in ["gold", "mold"]:
        L = int(v[f"layer_{c}"])            # 21 / 24 (vector index)
        js = JSpace(lens, hf, L - 1, mask)  # lens layer 20 / 23

        # gate: fresh trained-axis decomposition must reproduce the cohort file
        vf_fresh = float(js.decompose(v[f"v_{c}"].float()[L].numpy(),
                                      k=K)["var_fraction"])
        vf_stored = cohort["targets"][f"v_{c}"]["var_fraction"]
        gate_ok = abs(vf_fresh - vf_stored) < 1e-6
        out["verification_gate"][f"v_{c}"] = {
            "fresh": vf_fresh, "stored": vf_stored, "ok": gate_ok}
        print(f"GATE v_{c}: fresh {vf_fresh:.6f} vs stored {vf_stored:.6f} "
              f"-> {'OK' if gate_ok else 'FAIL'}", flush=True)
        if not gate_ok:
            json.dump(out, open(os.path.join(RES, "own_u_jshare.json"), "w"),
                      indent=1)
            print("OWN_U_JSHARE_GATE_FAIL", flush=True)
            sys.exit(1)

        vec = own[c][L].numpy()
        raw_norm = float(np.linalg.norm(vec))
        vec = vec / (raw_norm + 1e-9) * float(np.linalg.norm(
            v[f"v_{c}"].float()[L].numpy()))   # f2-style norm-match (no-op for j_share)
        d = js.decompose(vec, k=K)
        vf = float(d["var_fraction"])
        null = np.array(cohort["null"][c]["values"])
        n_ge = int((null >= vf).sum())
        out["targets"][f"own_u_{c}"] = {
            "vector_index": L, "lens_layer": L - 1,
            "raw_norm": raw_norm, "var_fraction": vf,
            "cos_x_xj": float(d["cos"]) if "cos" in d else None,
            "z_vs_stored_null": float((vf - null.mean()) / null.std(ddof=1)),
            "n_randoms_ge": n_ge,
            "perm_p": (n_ge + 1) / (len(null) + 1),
            "percentile_of_null": float(100.0 * (null < vf).mean()),
            "null_mean": float(null.mean()), "null_sd": float(null.std(ddof=1)),
            "top_tokens": tok.convert_ids_to_tokens(d["token_ids"][:8]),
        }
        print(f"own_u_{c} L{L-1}: j_share={vf:.4f} "
              f"z={out['targets'][f'own_u_{c}']['z_vs_stored_null']:+.2f} "
              f"randoms_ge={n_ge}/100 ({(time.time()-t0)/60:.1f} min)", flush=True)

    json.dump(out, open(os.path.join(RES, "own_u_jshare.json"), "w"), indent=1)
    print("OWN_U_JSHARE_SUMMARY " + json.dumps(
        {t: {k2: (round(v2, 4) if isinstance(v2, float) else v2)
             for k2, v2 in d.items() if k2 != "top_tokens"}
         for t, d in out["targets"].items()}), flush=True)
    print(f"OWN_U_JSHARE_DONE {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
