#!/usr/bin/env python3
"""F3 (audit) — atlas pole-score random cohort 6 -> 100 per polarity.

Measures ONLY the F4-pre-specified estimand: own-pole congruent score, jlens
readout (W_U J_l), band lens layers 16-31. Emits raw per-(direction, layer)
rows (this is also return-request R1) plus a band-averaged summary.

Design fixes vs atlas.py:
  - separate cohorts per polarity, norm-matched per-layer to that pole's
    step95 vector (the readout is log-softmax -> norm-sensitive; the old
    single gold-matched cohort under-matched the mold null) — stored under
    explicit concept labels 'gold'/'mold';
  - the first 6 gold-cohort randoms are drawn identically to atlas.py
    (default_rng(99) sequence) and their band means are checked against the
    existing atlas_rows.jsonl before the new cohort is trusted;
  - new filenames only (atlas.py's mode-"w" rewrite can truncate the
    original on interruption): atlas_rows_n100.jsonl, atlas_cohort_n100.json;
    resume-safe via per-direction completion tracking.

Cost: ~200 dirs x 16 layers of matmul readout ~= 60-90 min.
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
sys.path.insert(0, HERE)
import jlens  # noqa: E402
from dm_paths import ARTIFACTS, LENS_FILE  # noqa: E402
from routing_lib import unembed_weight  # noqa: E402
# identical word lists + tokenization to atlas.py (NOT prompts.py's lists,
# and NO gold/mold clash removal — the verification gate depends on this)
from atlas import GOLD_POLE, MOLD_POLE, CONTROL_NOUNS, first_tok_ids  # noqa: E402

RES = os.path.join(HERE, "results")
ART = ARTIFACTS

BAND = list(range(16, 32))     # lens layers (= vector layers 17-32)
N_RAND = 100
SEED = 99                      # atlas.py's stream; first 6 == existing cohort


def main():
    t0 = time.time()
    tok = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Instruct-2507")
    hf = transformers.AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-4B-Instruct-2507", dtype=torch.bfloat16)
    W = unembed_weight(hf).float().cuda()
    del hf
    torch.cuda.empty_cache()
    lens = jlens.JacobianLens.load(LENS_FILE)

    gids = first_tok_ids(tok, GOLD_POLE)
    mids = first_tok_ids(tok, MOLD_POLE)
    cids = first_tok_ids(tok, CONTROL_NOUNS)

    v95 = torch.load(f"{ART}/vectors_step95_bal.pt", map_location="cpu",
                     weights_only=False)
    uf = torch.load(f"{ART}/vectors_naive_faithful_pc5000.pt", map_location="cpu",
                    weights_only=False)
    Vg, Vm = v95["v_gold"].float(), v95["v_mold"].float()
    ref = {"gold": Vg, "mold": Vm}

    # gold cohort: atlas.py's sequential stream (first 6 must reproduce the
    # existing rand0..5). mold cohort: per-name streams — NOT a continuation
    # of the gold stream, which would change identity whenever N_RAND does
    # (the exact silent-corruption hazard of 90-gotchas.md #1).
    rng = np.random.default_rng(SEED)
    cohort = {"gold": [], "mold": []}
    for ri in range(N_RAND):
        r = torch.tensor(rng.standard_normal(Vg.shape).astype(np.float32))
        cohort["gold"].append(
            r / r.norm(dim=1, keepdim=True) * ref["gold"].norm(dim=1, keepdim=True))
    for ri in range(N_RAND):
        rr = np.random.default_rng([SEED, 1, ri])
        r = torch.tensor(rr.standard_normal(Vg.shape).astype(np.float32))
        cohort["mold"].append(
            r / r.norm(dim=1, keepdim=True) * ref["mold"].norm(dim=1, keepdim=True))

    dirs = [("v_gold", "gold", Vg), ("u_gold", "gold", uf["v_gold"].float()),
            ("v_mold", "mold", Vm), ("u_mold", "mold", uf["v_mold"].float())]
    # naive u norm-matched per layer to the corresponding v (as everywhere)
    dirs[1] = ("u_gold", "gold",
               uf["v_gold"].float() / (uf["v_gold"].float().norm(dim=1, keepdim=True) + 1e-9)
               * Vg.norm(dim=1, keepdim=True))
    dirs[3] = ("u_mold", "mold",
               uf["v_mold"].float() / (uf["v_mold"].float().norm(dim=1, keepdim=True) + 1e-9)
               * Vm.norm(dim=1, keepdim=True))
    for c in ["gold", "mold"]:
        for ri in range(N_RAND):
            dirs.append((f"rand_{c}{ri}", c, cohort[c][ri]))

    rows_path = os.path.join(RES, "atlas_rows_n100.jsonl")
    done = set()
    if os.path.exists(rows_path):
        for l in open(rows_path):
            done.add(json.loads(l)["set"])
    f = open(rows_path, "a")
    for i, (name, c, V) in enumerate(dirs):
        if name in done:
            continue
        for l in BAND:
            v_row = V[l + 1].cuda() if l + 1 < V.shape[0] else V[-1].cuda()
            J = lens.jacobians[l].cuda().float()
            lp = torch.log_softmax(W @ (J @ v_row), -1)
            f.write(json.dumps({
                "set": name, "concept": c, "lens_layer": int(l),
                "variant": "jlens",
                "gold_pole": float(lp[gids].mean() - lp[cids].mean()),
                "mold_pole": float(lp[mids].mean() - lp[cids].mean()),
            }) + "\n")
        f.flush()
        if i % 10 == 0:
            print(f"[{i+1}/{len(dirs)}] {name} ({(time.time()-t0)/60:.1f} min)",
                  flush=True)
    f.close()

    # ---- verification: first 6 gold randoms vs existing atlas ----
    new_rows = [json.loads(l) for l in open(rows_path)]
    old_rows = [json.loads(l) for l in open(os.path.join(RES, "atlas_rows.jsonl"))]

    def band_mean(rows, sname, field, variant="jlens"):
        vals = [r[field] for r in rows
                if r["set"] == sname and r["variant"] == variant
                and r["lens_layer"] in BAND]
        return sum(vals) / len(vals)

    max_diff = 0.0
    for ri in range(6):
        a = band_mean(new_rows, f"rand_gold{ri}", "gold_pole")
        b = band_mean(old_rows, f"rand{ri}", "gold_pole")
        max_diff = max(max_diff, abs(a - b))
    verified = max_diff < 1e-3
    print(f"verify first-6 gold randoms vs existing atlas: max diff "
          f"{max_diff:.2e} -> {'OK' if verified else 'MISMATCH'}", flush=True)
    if not verified:
        print("F3_FAIL: cohort does not reproduce existing randoms — "
              "convention drift, stop and diagnose", flush=True)
        sys.exit(1)

    # ---- summary on the F4 estimand ----
    out = {"n_random_per_polarity": N_RAND,
           "estimand": "own-pole congruent score, jlens readout, band mean "
                       "lens layers 16-31 (F4 pre-specification)",
           "readout_variant": "jlens", "layers": [BAND[0], BAND[-1]],
           "norm_matching": "per-layer to the same pole's step95 vector",
           "seed_scheme": f"gold: sequential default_rng({SEED}) (first 6 == "
                          f"atlas.py's rand0..5, verified); mold: per-name "
                          f"default_rng([{SEED}, 1, ri])",
           "targets": {}, "null": {}, "per_layer_null": {}}
    pole_field = {"gold": "gold_pole", "mold": "mold_pole"}
    for c in ["gold", "mold"]:
        null = [band_mean(new_rows, f"rand_{c}{ri}", pole_field[c])
                for ri in range(N_RAND)]
        arr = np.array(null)
        out["null"][c] = {"values": null, "mean": float(arr.mean()),
                          "sd": float(arr.std(ddof=1)),
                          "p05": float(np.percentile(arr, 5)),
                          "p95": float(np.percentile(arr, 95))}
        per_layer = {}
        for l in BAND:
            vals = [r[pole_field[c]] for r in new_rows
                    if r["set"].startswith(f"rand_{c}") and r["lens_layer"] == l]
            per_layer[str(l)] = {"mean": float(np.mean(vals)),
                                 "sd": float(np.std(vals, ddof=1))}
        out["per_layer_null"][c] = per_layer
        for t in [f"v_{c}", f"u_{c}"]:
            val = band_mean(new_rows, t, pole_field[c])
            n_ge = int((arr >= val).sum())
            out["targets"][t] = {
                "band_mean": val,
                "z": float((val - arr.mean()) / arr.std(ddof=1)),
                "n_randoms_ge": n_ge, "perm_p": (n_ge + 1) / (N_RAND + 1),
                "per_layer": {str(l): [r[pole_field[c]] for r in new_rows
                                       if r["set"] == t and r["lens_layer"] == l][0]
                              for l in BAND}}
    json.dump(out, open(os.path.join(RES, "atlas_cohort_n100.json"), "w"),
              indent=1)
    print("F3_SUMMARY " + json.dumps({t: {"band_mean": round(d["band_mean"], 4),
          "z": round(d["z"], 2), "perm_p": round(d["perm_p"], 4)}
          for t, d in out["targets"].items()}), flush=True)
    print(f"F3_DONE {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
