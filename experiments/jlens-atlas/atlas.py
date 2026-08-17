"""J1+J3: Welfare-axis J-lens readout ATLAS + sparsity sweep.

Welfare axis = OBJECT, J-lens = TOOL. No generation, no self-report.

For each vector set (step95, step150, naive_faithful, own_naive if present,
plus norm-matched randoms) x concept (gold, mold):
  A. Layer atlas: at every lens source layer l, read the axis row at
     block-input l+1 through TWO readout conventions:
       - "han":   z = W_U v            (Han et al. Table 2 convention, J=I)
       - "jlens": z = W_U (J_l v)      (Jacobian transport, no norm)
       - "jlens_norm": z = W_U norm(J_l v)  (jlens apply() convention)
     Record top-25 promoted and suppressed word-masked tokens + pole scores
     (mean logsoftmax of Gold-pole / Mold-pole ids minus 100 control nouns).
  B. Sparsity sweep (J3): gradient-pursuit decomposition at layers {17..30},
     k in {4,8,16,25,50}: J-share + selected tokens (stability analysis
     done by the analyzer).

Run (venv): /workspace/venvs/jlens/bin/python atlas.py [--quick]
Outputs: results/atlas_rows.jsonl, results/ksweep.json, results/atlas_meta.json
ETA: ~45-90 min full, ~10 min --quick.
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import transformers

import jlens

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "common"))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "routing-core"))
from dm_paths import ARTIFACTS, LENS_FILE
from routing_lib import (
    JSpace,
    assert_decompositions_converged,
    decomposition_record,
    make_valid_mask,
    unembed_weight,
)

RES = os.path.join(HERE, "results")
os.makedirs(RES, exist_ok=True)
ART = ARTIFACTS

GOLD_POLE = ["inspired", "loving", "proud", "fulfilled", "blissful"]
MOLD_POLE = ["humiliated", "embarrassed", "ashamed", "insulted", "annoyed",
             "exasperated", "irritated", "offended"]
CONTROL_NOUNS = ("table chair window door garden bridge bottle mountain river "
                 "castle pencil market station engine blanket mirror ladder "
                 "kitchen forest island butter camera pocket ticket jacket "
                 "barrel candle hammer tunnel valley circus museum harbor "
                 "meadow lantern gravel napkin kettle pillow shovel carpet "
                 "cabinet bicycle guitar recipe planet stadium theater ocean "
                 "desert village temple palace copper marble timber canvas "
                 "basket saddle anchor compass turbine reactor antenna furnace "
                 "chimney balcony corridor cellar attic pavement sidewalk "
                 "fountain statue monument archway staircase elevator platform "
                 "terminal runway cockpit propeller rudder keel mast sail oar "
                 "paddle canoe kayak yacht ferry barge wagon trolley").split()[:100]


def first_tok_ids(tok, words):
    ids = set()
    for w in words:
        for form in (" " + w, w, " " + w.capitalize(), w.capitalize()):
            t = tok.encode(form, add_special_tokens=False)
            if t:
                ids.add(t[0])
    return sorted(ids)


def word_mask_top(tok, z, k=25, reverse=False):
    vals, idx = torch.topk(z, 400, largest=not reverse)
    out = []
    for i in idx.tolist():
        s = tok.decode([i]).strip()
        if s and all(ch.isalpha() or ch in "'-一-鿿" for ch in s) and len(s) > 1:
            out.append(s)
        if len(out) >= k:
            break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--lens-file", default=LENS_FILE)
    args = ap.parse_args()

    t0 = time.time()
    hf = transformers.AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-4B-Instruct-2507", dtype=torch.bfloat16).cuda()
    tok = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Instruct-2507")
    lens = jlens.JacobianLens.load(args.lens_file)
    W = unembed_weight(hf).float().cuda()
    norm_w = hf.model.norm.weight.detach().float().cuda()
    gids = first_tok_ids(tok, GOLD_POLE)
    mids = first_tok_ids(tok, MOLD_POLE)
    cids = first_tok_ids(tok, CONTROL_NOUNS)
    print(f"loaded {time.time()-t0:.0f}s; lens layers {lens.source_layers[:2]}..."
          f"{lens.source_layers[-1]}", flush=True)

    sets = {"step95": f"{ART}/vectors_step95_bal.pt",
            "step150": f"{ART}/vectors_step150_bal.pt",
            "naive_faithful": f"{ART}/vectors_naive_faithful_pc5000.pt"}
    if os.path.exists(f"{ART}/vectors_own_naive.pt"):
        sets["own_naive"] = f"{ART}/vectors_own_naive.pt"
    vecs = {}
    for name, p in sets.items():
        d = torch.load(p, map_location="cpu", weights_only=False)
        for c in ["gold", "mold"]:
            vecs[(name, c)] = d[f"v_{c}"].float()

    rng = np.random.default_rng(99)
    n_rand = 2 if args.quick else 6
    ref = vecs[("step95", "gold")]
    for ri in range(n_rand):
        r = torch.tensor(rng.standard_normal(ref.shape).astype(np.float32))
        r = r / r.norm(dim=1, keepdim=True) * ref.norm(dim=1, keepdim=True)
        vecs[(f"rand{ri}", "gold")] = r

    layers = lens.source_layers[:: (4 if args.quick else 1)]

    # ---------------- A. layer atlas ----------------
    rows_path = os.path.join(RES, "atlas_rows.jsonl")
    with open(rows_path, "w") as f:
        for (sname, c), V in vecs.items():
            for l in layers:
                v_row = V[l + 1].cuda() if l + 1 < V.shape[0] else V[-1].cuda()
                J = lens.jacobians[l].cuda().float()
                jv = J @ v_row
                nrm_state = jv / (jv.pow(2).mean().sqrt() + 1e-6) * norm_w
                for variant, z in [("han", W @ v_row), ("jlens", W @ jv),
                                   ("jlens_norm", W @ nrm_state)]:
                    lp = torch.log_softmax(z, -1)
                    f.write(json.dumps({
                        "set": sname, "concept": c, "lens_layer": int(l),
                        "variant": variant,
                        "gold_pole": float(lp[gids].mean() - lp[cids].mean()),
                        "mold_pole": float(lp[mids].mean() - lp[cids].mean()),
                        "top_promoted": word_mask_top(tok, z, 15),
                        "top_suppressed": word_mask_top(tok, z, 15, reverse=True),
                    }) + "\n")
            print(f"atlas {sname}/{c} done {time.time()-t0:.0f}s", flush=True)

    # ---------------- B. sparsity sweep ----------------
    ks = [8, 16] if args.quick else [4, 8, 16, 25, 50]
    sweep_layers = [19, 23] if args.quick else list(range(17, 31, 2))
    mask = make_valid_mask(tok, W.shape[0])
    ksweep = {}
    jspaces = {}
    for l in sweep_layers:
        jspaces[l] = JSpace(lens, hf, l, mask)
    for (sname, c), V in vecs.items():
        if sname == "step150":
            continue  # atlas covers step150; k-sweep on step95 suffices
        if sname.startswith("rand") and int(sname[4:]) > 1:
            continue  # two randoms are enough for the sweep null
        for l in sweep_layers:
            v_row = V[l + 1].cpu().numpy() if l + 1 < V.shape[0] else V[-1].cpu().numpy()
            for k in ks:
                d = jspaces[l].decompose(v_row, k=k)
                ksweep[f"{sname}|{c}|L{l}|k{k}"] = decomposition_record(d, tok)
        print(f"ksweep {sname}/{c} done {time.time()-t0:.0f}s", flush=True)
    assert_decompositions_converged(ksweep, context="ksweep.json")
    with open(os.path.join(RES, "ksweep.json"), "w") as f:
        json.dump(ksweep, f, indent=1)

    with open(os.path.join(RES, "atlas_meta.json"), "w") as f:
        json.dump({"complete": True, "layers": [int(x) for x in layers],
                   "ks": ks, "sweep_layers": sweep_layers,
                   "sets": list(sets) + [f"rand{i}" for i in range(n_rand)],
                   "lens_file": args.lens_file, "quick": args.quick,
                   "seconds": round(time.time() - t0, 1)}, f, indent=2)
    print("ATLAS_DONE", flush=True)


if __name__ == "__main__":
    main()
