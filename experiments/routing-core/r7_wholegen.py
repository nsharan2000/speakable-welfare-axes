#!/usr/bin/env python3
"""R7 (audit return-request) — does C6's verdict survive a whole-generation
readout?

The primary readout was FIRST-answer-token log-mass; C6 found v_mold's shift
equal-or-larger on unrelated prompts, killing the self-report-channel
reading. This asks whether that is a property of the channel or of the
first-token readout: for v_mold and u_mold at alpha=+4 (and clean), generate
full responses on the 15 self-report + 10 unrelated prompts and compute
  (a) mean per-generated-position gold-vs-mold log-mass (steered - clean),
  (b) the generated texts (for blind judge scoring afterwards).
If the self-report/unrelated gap reappears -> C6's failure was a first-token
artifact; if not -> C6 confirmed at the channel level.

Reuses run_primary.py's exact word sets, clash removal, steering layer
(vector layer - 1) and norm-matching. ~75 generations x ~2 s ~= 3-4 min.
Writes results/R7_wholegen.json (+ texts inline). Resume-safe by key.
"""
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "common"))
sys.path.insert(0, HERE)
from dm_common import chat_ids, load_model, steering  # noqa: E402
from dm_paths import ARTIFACTS, FW_EVAL_PROMPTS  # noqa: E402
from run_primary import (UNRELATED,  # noqa: E402
                         build_sets)

RES = os.path.join(HERE, "results")
ART = ARTIFACTS
OUT = os.path.join(RES, "R7_wholegen.json")

ALPHA = 4.0
N_TOK = 80


def main():
    t0 = time.time()
    model, tok = load_model("instruct")
    gids, mids, _ = build_sets(tok)

    v = torch.load(f"{ART}/vectors_step95_bal.pt", map_location="cpu",
                   weights_only=False)
    u = torch.load(f"{ART}/vectors_naive_faithful_pc5000.pt", map_location="cpu",
                   weights_only=False)
    L = int(v["layer_mold"])
    vec = v["v_mold"].float()[L].numpy()
    nv = np.linalg.norm(vec)
    uvec = u["v_mold"].float()[L].numpy()
    uvec = uvec / (np.linalg.norm(uvec) + 1e-9) * nv
    conds = {"clean": None,
             "v_mold": (L - 1, vec * ALPHA),
             "u_mold": (L - 1, uvec * ALPHA)}

    prompts_all = json.load(open(FW_EVAL_PROMPTS))
    SELF = [p["prompt"] for p in prompts_all
            if p["category"] == "welfare_self_reports"]
    psets = {"self": SELF, "unrel": UNRELATED}

    data = json.load(open(OUT)) if os.path.exists(OUT) else {"rows": {}}
    rows = data["rows"]

    def run(msg, steer):
        ids = chat_ids(tok, msg)          # already on cuda (dm_common default)
        with torch.no_grad():
            if steer is not None:
                sl, svec = steer
                with steering(model, sl, svec, 1.0):
                    o = model.generate(ids, max_new_tokens=N_TOK,
                                       do_sample=False,
                                       output_scores=True,
                                       return_dict_in_generate=True)
            else:
                o = model.generate(ids, max_new_tokens=N_TOK, do_sample=False,
                                   output_scores=True,
                                   return_dict_in_generate=True)
        per_pos = []
        for sc in o.scores:
            lp = torch.log_softmax(sc[0].float(), -1)
            gm = torch.logsumexp(lp[gids], -1) - torch.logsumexp(lp, -1)
            mm = torch.logsumexp(lp[mids], -1) - torch.logsumexp(lp, -1)
            per_pos.append(float(gm - mm))
        text = tok.decode(o.sequences[0, ids.shape[1]:], skip_special_tokens=True)
        return float(np.mean(per_pos)), text

    for cname, steer in conds.items():
        for pset, plist in psets.items():
            for pi, msg in enumerate(plist):
                key = f"{cname}|{pset}|{pi}"
                if key in rows:
                    continue
                V, text = run(msg, steer)
                rows[key] = {"cond": cname, "pset": pset, "prompt": msg,
                             "V_wholegen": V, "text": text}
                json.dump(data, open(OUT, "w"), indent=1)
        print(f"{cname} done {(time.time()-t0)/60:.1f} min", flush=True)

    # effects: steered - clean, per prompt, then mean per (cond, pset)
    summary = {}
    for cname in ["v_mold", "u_mold"]:
        for pset in ["self", "unrel"]:
            effs = []
            for pi in range(len(psets[pset])):
                effs.append(rows[f"{cname}|{pset}|{pi}"]["V_wholegen"]
                            - rows[f"clean|{pset}|{pi}"]["V_wholegen"])
            summary[f"{cname}|{pset}"] = {
                "mean_effect": float(np.mean(effs)),
                "sd": float(np.std(effs, ddof=1)), "n": len(effs)}
    for cname in ["v_mold", "u_mold"]:
        summary[f"{cname}|specificity_gap"] = (
            summary[f"{cname}|self"]["mean_effect"]
            - summary[f"{cname}|unrel"]["mean_effect"])
    data["summary"] = summary
    data["meta"] = {"alpha": ALPHA, "n_tok": N_TOK,
                    "polarity_note": "raw V=gold-mold mass; mold-congruent "
                                     "effect is NEGATIVE V shift",
                    "readout": "mean over generated positions of "
                               "(logmass gold - logmass mold), steered - clean"}
    json.dump(data, open(OUT, "w"), indent=1)
    print("R7_SUMMARY " + json.dumps({k: (round(v2, 3) if isinstance(v2, float)
          else v2) for k, v2 in summary.items() if "gap" in k}), flush=True)
    print(f"R7_DONE {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
