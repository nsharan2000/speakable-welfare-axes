"""J2: When during RL does the welfare axis enter the verbalizable subspace?

Uses the recruitment-trajectory vector series (nickmahdavi
artifacts/traj/vectors_step{0..150}.pt, 31 training checkpoints x Gold/Mold)
and tracks, per checkpoint:
  - vector norm at the treatment layer (recruitment strength, Han et al.)
  - cos with the final-step vector (geometric convergence)
  - J-share, the squared-norm ratio (gradient pursuit k=16 and k=25)
  - top selected J-space tokens (vocabulary emergence timeline)
  - Gold/Mold pole scores of the jlens readout (W_U J v)
Question: does entry into the speakable cone precede, track, or lag the norm
growth? Novel measurement — nobody has tracked verbalizable-subspace entry
over training steps.

Run (venv): /workspace/venvs/jlens/bin/python recruitment_traj.py
Downloads traj files if missing (~23 MB). ETA: ~30-60 min GPU.
Outputs: results/traj_results.json
"""

import glob
import json
import os
import re
import subprocess
import sys
import time

import numpy as np
import torch
import transformers

import jlens

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "common"))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "routing-core"))
from dm_paths import ARTIFACTS, LENS_FILE, WELFARE_VECTORS
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

TRAJ_DIR = f"{ART}/traj"
if not glob.glob(f"{TRAJ_DIR}/vectors_step*.pt"):
    print("downloading traj series...", flush=True)
    subprocess.run(["hf", "download", "nickmahdavi/functional-welfare",
                    "--include", "artifacts/traj/*",
                    "--local-dir", WELFARE_VECTORS], check=True)

files = sorted(glob.glob(f"{TRAJ_DIR}/vectors_step*.pt"),
               key=lambda p: int(re.search(r"step(\d+)", p).group(1)))
print(f"{len(files)} checkpoints", flush=True)


def first_tok_ids(tok, words):
    ids = set()
    for w in words:
        for form in (" " + w, w, " " + w.capitalize(), w.capitalize()):
            t = tok.encode(form, add_special_tokens=False)
            if t:
                ids.add(t[0])
    return sorted(ids)


t0 = time.time()
hf_model = transformers.AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-4B-Instruct-2507", dtype=torch.bfloat16).cuda()
tok = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Instruct-2507")
lens = jlens.JacobianLens.load(LENS_FILE)
W = unembed_weight(hf_model).float().cuda()
gids = first_tok_ids(tok, GOLD_POLE)
mids = first_tok_ids(tok, MOLD_POLE)
cids = first_tok_ids(tok, CONTROL_NOUNS)
mask = make_valid_mask(tok, W.shape[0])

# treatment layers (block-input convention) -> lens layers
LENS_L = {"gold": 20, "mold": 23}
BLOCK_IN = {"gold": 21, "mold": 24}
jspaces = {c: JSpace(lens, hf_model, LENS_L[c], mask) for c in LENS_L}
print(f"setup {time.time()-t0:.0f}s", flush=True)

final = torch.load(files[-1], map_location="cpu", weights_only=False)

results = []
for fp in files:
    step = int(re.search(r"step(\d+)", fp).group(1))
    d = torch.load(fp, map_location="cpu", weights_only=False)
    for c in ["gold", "mold"]:
        key = f"v_{c}"
        if key not in d:
            continue
        V = d[key].float()
        li = BLOCK_IN[c]
        v_row = V[li].numpy()
        vf = final[key].float()[li].numpy()
        nrm = float(np.linalg.norm(v_row))
        cosf = float(np.dot(v_row, vf) / (np.linalg.norm(v_row) * np.linalg.norm(vf) + 1e-9))
        row = {"step": step, "concept": c, "norm": nrm, "cos_final": cosf}
        for k in (16, 25):
            dec = jspaces[c].decompose(v_row, k=k)
            record = decomposition_record(dec, tok)
            row[f"j_share_k{k}"] = dec["j_share"]
            row[f"var_fraction_k{k}"] = dec["var_fraction"]
            row[f"reconstruction_r2_k{k}"] = dec["reconstruction_r2"]
            row[f"decomposition_k{k}"] = record
            if k == 16:
                row["tokens_selection_order_k16"] = record["tokens_selection_order"]
                row["tokens_k16"] = record["tokens_by_coefficient"]
                row["tokens_by_coefficient_k16"] = record["tokens_by_coefficient"]
                row["coeffs_k16"] = record["coeffs"]
        J = lens.jacobians[LENS_L[c]].cuda().float()
        z = W @ (J @ torch.tensor(v_row).cuda())
        lp = torch.log_softmax(z, -1)
        row["gold_pole"] = float(lp[gids].mean() - lp[cids].mean())
        row["mold_pole"] = float(lp[mids].mean() - lp[cids].mean())
        results.append(row)
    print(f"step {step} done {time.time()-t0:.0f}s", flush=True)

assert_decompositions_converged(results, context="traj_results.json")
with open(os.path.join(RES, "traj_results.json"), "w") as f:
    json.dump({"results": results, "lens_layers": LENS_L,
               "block_input_layers": BLOCK_IN,
               "n_checkpoints": len({r["step"] for r in results}),
               "steps": sorted({r["step"] for r in results}),
               "seconds": round(time.time() - t0, 1)}, f, indent=1)
print(f"TRAJ_DONE n_checkpoints={len({r['step'] for r in results})}", flush=True)
