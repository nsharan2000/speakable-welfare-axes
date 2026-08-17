#!/usr/bin/env python3
"""J6 — cross-model transfer to Qwen/Qwen3-4B (the sprint's stated open
question: "to what extent do valence directions found in one model transfer
to another?").

Same architecture/width/tokenizer family as Instruct-2507, PUBLIC pre-fitted
J-lens (neuronpedia/jacobian-lens) — so the 2507 axes can be injected
directly and read through Qwen3-4B's own lens.

Arms (all resume-safe, one GPU job):
  A. Lens atlas (F4 estimand: own-pole congruent, jlens readout, band lens
     16-31) for transferred v/u + 30 norm-matched randoms per polarity.
     Band note: Qwen3-4B's own band identification was inconclusive
     (band_stats onset heuristic degenerate at 0); we use the 2507 band for
     cross-model comparability and state that.
  B. Gradient-pursuit decompositions (k=16) at lens 20/23 for transferred
     v/u + 8 randoms/polarity -> J-share + token sets.
  C. Token-set Jaccard: Qwen3-4B(transfer v) vs 2507(v) decomposition
     tokens, with a random-pair Jaccard chance baseline.
  D. Behavioral transfer: 16 neutral prompts x {clean, v_gold, v_mold,
     u_gold, u_mold} at alpha=+4, thinking disabled -> texts for blind
     judging (judged locally afterwards).

HONEST LIMIT (stated in the report): v exists only from 2507's maze-RL, so
this tests whether the axis + lens geometry TRANSFER across checkpoints of
the family — not whether RL recruitment reproduces.

Outputs under results/: j6_atlas_rows.jsonl, j6_decomps.json,
j6_generations.jsonl, j6_summary.json. Cost ~25-40 min.
"""
import itertools
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
from dm_paths import ARTIFACTS, MECH_DECOMPOSITIONS  # noqa: E402
from routing_lib import (  # noqa: E402
    ROUTING_ALGORITHM_VERSION,
    JSpace,
    assert_decompositions_converged,
    assert_source_version,
    decomposition_record,
    make_valid_mask,
    unembed_weight,
)
from atlas import GOLD_POLE, MOLD_POLE, CONTROL_NOUNS, first_tok_ids  # noqa: E402
from prompts import NEUTRAL_PROMPTS  # noqa: E402

RES = os.path.join(HERE, "results")
os.makedirs(RES, exist_ok=True)
ART = ARTIFACTS
MODEL = "Qwen/Qwen3-4B"
LENS_REPO = "neuronpedia/jacobian-lens"
# Path WITHIN the Hugging Face repo above, not a local filesystem path.
LENS_FILE = "qwen3-4b/jlens/Salesforce-wikitext/Qwen3-4B_jacobian_lens.pt"
SRC_DECOMPS = MECH_DECOMPOSITIONS

BAND = list(range(16, 32))
N_RAND_ATLAS = 30
N_RAND_DEC = 8
K = 16
ALPHA = 4.0
SEED = 4242


def rand_dir(pol, ri, shape):
    rng = np.random.default_rng([SEED, pol, ri])
    return torch.tensor(rng.standard_normal(shape).astype(np.float32))


def main():
    t0 = time.time()
    tok = transformers.AutoTokenizer.from_pretrained(MODEL)
    hf = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16).cuda()
    lens = jlens.JacobianLens.from_pretrained(LENS_REPO, filename=LENS_FILE)
    W = unembed_weight(hf).float().cuda()
    mask = make_valid_mask(tok, W.shape[0])
    gids = first_tok_ids(tok, GOLD_POLE)
    mids = first_tok_ids(tok, MOLD_POLE)
    cids = first_tok_ids(tok, CONTROL_NOUNS)
    # the public lens may not cover every layer we ask for
    band = [l for l in BAND if l in set(lens.source_layers)]
    assert len(band) >= 8, f"public lens covers too few band layers: {band}"
    globals()["BAND"] = band
    print(f"loaded {time.time()-t0:.0f}s; band layers used: "
          f"{band[0]}..{band[-1]} (n={len(band)})", flush=True)

    v95 = torch.load(f"{ART}/vectors_step95_bal.pt", map_location="cpu",
                     weights_only=False)
    uf = torch.load(f"{ART}/vectors_naive_faithful_pc5000.pt",
                    map_location="cpu", weights_only=False)
    Vg, Vm = v95["v_gold"].float(), v95["v_mold"].float()
    Ug = uf["v_gold"].float() / (uf["v_gold"].float().norm(dim=1, keepdim=True) + 1e-9) \
        * Vg.norm(dim=1, keepdim=True)
    Um = uf["v_mold"].float() / (uf["v_mold"].float().norm(dim=1, keepdim=True) + 1e-9) \
        * Vm.norm(dim=1, keepdim=True)
    ref = {"gold": Vg, "mold": Vm}

    dirs = [("v_gold", "gold", Vg), ("u_gold", "gold", Ug),
            ("v_mold", "mold", Vm), ("u_mold", "mold", Um)]
    for c, pol in [("gold", 0), ("mold", 1)]:
        for ri in range(N_RAND_ATLAS):
            r = rand_dir(pol, ri, Vg.shape)
            r = r / r.norm(dim=1, keepdim=True) * ref[c].norm(dim=1, keepdim=True)
            dirs.append((f"rand_{c}{ri}", c, r))

    # ---------- A. atlas ----------
    rows_path = os.path.join(RES, "j6_atlas_rows.jsonl")
    done = set()
    if os.path.exists(rows_path):
        done = {json.loads(l)["set"] for l in open(rows_path)}
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
            print(f"atlas [{i+1}/{len(dirs)}] {name} "
                  f"({(time.time()-t0)/60:.1f}m)", flush=True)
    f.close()

    # ---------- B. decompositions ----------
    dec_path = os.path.join(RES, "j6_decomps.json")
    decs = json.load(open(dec_path)) if os.path.exists(dec_path) else {}
    jspaces = {}
    dec_dirs = dirs[:4] + [d for d in dirs
                           if d[0].startswith("rand_") and
                           int(d[0].split("rand_")[1][4:]) < N_RAND_DEC]
    LENSL = {"gold": 20, "mold": 23}
    for name, c, V in dec_dirs:
        if decs.get(name, {}).get("algorithm_version") == ROUTING_ALGORITHM_VERSION:
            continue
        l = LENSL[c]
        if l not in jspaces:
            jspaces[l] = JSpace(lens, hf, l, mask)
        v_row = V[l + 1].cpu().numpy()
        d = jspaces[l].decompose(v_row, k=K)
        decs[name] = {"lens_layer": l, **decomposition_record(d, tok)}
        json.dump(decs, open(dec_path, "w"), indent=1)
        print(f"decomp {name} j_share={d['j_share']:.4f} "
              f"tokens={decs[name]['tokens_by_coefficient'][:6]}", flush=True)

    # ---------- C. token overlap vs 2507 ----------
    src = json.load(open(SRC_DECOMPS))
    assert_source_version(src, SRC_DECOMPS)

    # NOTE: the 2507 decomposition file stores DECODED TOKEN STRINGS, not ids
    # (verified), so overlap is computed on normalized strings. That is also
    # the right unit for a cross-model claim: it compares what the two axes
    # *say*, not how a tokenizer numbered it.
    def norm_tok(s):
        return s.strip().lower()

    def toks(entry):
        values = entry.get("tokens_by_coefficient", entry["tokens"])
        return {norm_tok(s) for s in values}

    def ranked_tokens(entry):
        return entry.get("tokens_by_coefficient", entry["tokens"])

    def jac(a, b):
        a, b = set(a), set(b)
        return len(a & b) / max(1, len(a | b))

    overlap = {}
    for t in ["v_gold", "v_mold", "u_gold", "u_mold"]:
        if t in src and t in decs:
            A, B = toks(src[t]), toks(decs[t])
            overlap[t] = {
                "jaccard": jac(A, B),
                "n_shared": len(A & B),
                "shared_tokens": sorted(A & B),
                "tokens_2507": ranked_tokens(src[t])[:K],
                "tokens_qwen34b": ranked_tokens(decs[t])[:K]}
    rand_j = [jac(toks(decs[f"rand_gold{i}"]), toks(decs[f"rand_gold{j}"]))
              for i, j in itertools.combinations(range(N_RAND_DEC), 2)]
    cross_rand = [jac(toks(src[t]), toks(decs[f"rand_{c}{i}"]))
                  for t, c in [("v_gold", "gold"), ("v_mold", "mold")]
                  for i in range(N_RAND_DEC) if t in src]

    # ---------- D. behavioral transfer generations ----------
    gen_path = os.path.join(RES, "j6_generations.jsonl")
    gdone = set()
    if os.path.exists(gen_path):
        gdone = {(r["cond"], r["prompt"]) for r in
                 map(json.loads, open(gen_path))}
    layers_module = hf.model.layers
    conds = {"clean": None}
    for c, V in [("gold", Vg), ("mold", Vm)]:
        L = int(v95[f"layer_{c}"])
        conds[f"v_{c}"] = (L - 1, V[L].numpy() * ALPHA)
        conds[f"u_{c}"] = (L - 1, ({"gold": Ug, "mold": Um}[c])[L].numpy() * ALPHA)
    gf = open(gen_path, "a")
    for cname, steer in conds.items():
        for msg in NEUTRAL_PROMPTS[:16]:
            if (cname, msg) in gdone:
                continue
            ids = tok.apply_chat_template(
                [{"role": "user", "content": msg}],
                add_generation_prompt=True, enable_thinking=False,
                return_tensors="pt")
            ids = (ids["input_ids"] if hasattr(ids, "keys") else ids).cuda()
            hd = None
            if steer is not None:
                L, vec = steer
                vt = torch.tensor(vec)

                def fn(module, inputs, output, vt=vt):
                    if isinstance(output, tuple):
                        return (output[0] + vt.to(output[0].dtype).to(output[0].device),) + output[1:]
                    return output + vt.to(output.dtype).to(output.device)
                hd = layers_module[L].register_forward_hook(fn)
            try:
                with torch.no_grad():
                    o = hf.generate(ids, max_new_tokens=60, do_sample=False)
            finally:
                if hd:
                    hd.remove()
            gf.write(json.dumps({
                "cond": cname, "prompt": msg,
                "text": tok.decode(o[0, ids.shape[1]:],
                                   skip_special_tokens=True)}) + "\n")
            gf.flush()
        print(f"gen {cname} done ({(time.time()-t0)/60:.1f}m)", flush=True)
    gf.close()

    # ---------- summary ----------
    rows = [json.loads(l) for l in open(rows_path)]

    def band_mean(sname, field):
        vals = [r[field] for r in rows if r["set"] == sname]
        return sum(vals) / len(vals)

    pole_field = {"gold": "gold_pole", "mold": "mold_pole"}
    summ = {"model": MODEL, "lens": f"{LENS_REPO}/{LENS_FILE}",
            "band_lens_layers": [BAND[0], BAND[-1]],
            "band_note": "2507 band reused for comparability; Qwen3-4B's own "
                         "band identification inconclusive (onset heuristic "
                         "degenerate at 0 in band_stats_Qwen3-4B.json)",
            "estimand": "F4: own-pole congruent, jlens readout, band mean",
            "atlas": {}, "jshare": {}, "token_overlap": overlap,
            "jaccard_baselines": {
                "rand_pairs_within_qwen34b": {
                    "mean": float(np.mean(rand_j)), "max": float(np.max(rand_j))},
                "2507v_vs_qwen34b_randoms": {
                    "mean": float(np.mean(cross_rand)),
                    "max": float(np.max(cross_rand))}},
            "limitation": "trained v exists only from 2507 maze-RL; this "
                          "tests axis+lens-geometry transfer across family "
                          "checkpoints, not RL-recruitment reproduction"}
    for c in ["gold", "mold"]:
        null = [band_mean(f"rand_{c}{ri}", pole_field[c])
                for ri in range(N_RAND_ATLAS)]
        arr = np.array(null)
        summ["atlas"][f"null_{c}"] = {"mean": float(arr.mean()),
                                      "sd": float(arr.std(ddof=1)),
                                      "n": N_RAND_ATLAS}
        for t in [f"v_{c}", f"u_{c}"]:
            val = band_mean(t, pole_field[c])
            n_ge = int((arr >= val).sum())
            summ["atlas"][t] = {"band_mean": val,
                                "z": float((val - arr.mean()) / arr.std(ddof=1)),
                                "perm_p": (n_ge + 1) / (N_RAND_ATLAS + 1)}
        dn = [decs[f"rand_{c}{i}"]["var_fraction"] for i in range(N_RAND_DEC)]
        da = np.array(dn)
        for t in [f"v_{c}", f"u_{c}"]:
            vf = decs[t]["var_fraction"]
            n_ge = int((da >= vf).sum())
            summ["jshare"][t] = {"var_fraction": vf,
                                 "z": float((vf - da.mean()) / da.std(ddof=1)),
                                 "perm_p": (n_ge + 1) / (N_RAND_DEC + 1)}
    assert_decompositions_converged(decs, context="j6_decomps.json")
    json.dump(summ, open(os.path.join(RES, "j6_summary.json"), "w"), indent=1)
    print("J6_SUMMARY " + json.dumps(
        {"atlas": {k: {kk: round(vv, 3) for kk, vv in v.items()}
                   for k, v in summ["atlas"].items()},
         "jaccard": {k: round(v["jaccard"], 3)
                     for k, v in overlap.items()}}), flush=True)
    print(f"J6_DONE {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
