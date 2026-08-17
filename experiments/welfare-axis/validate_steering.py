"""E5a: Validate the third-party welfare vectors behaviorally (Han et al. replication).

Checks (before trusting the vectors for the routing experiment):
  V1 per-layer norms of v/u at the paper's steering layers vs published values
     (Table 27: v_Mold 4.07@L20, v_Gold 14.37@L22 for step-95; nickmahdavi's
     recomputed layers are L24/L21 - report both).
  V2 steering the UN-finetuned Instruct-2507 (paper's main-text protocol) with
     alpha*v at the treatment layer shifts generation sentiment:
     v_gold + -> more positive; v_mold + -> more negative; norm-matched u and
     random controls ~flat. Lexical proxy now; Claude-judge scoring later on
     the saved generations.

Convention note: vectors are indexed by BLOCK-INPUT layer (their pre-hook);
our steering() adds at block OUTPUT, so steer_layer = vec_layer - 1.

Run (system python): python3 validate_steering.py [--quick]
Outputs: results/steering_validation.json, results/generations.jsonl
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "common"))
from dm_common import load_model, steering
from dm_paths import ARTIFACTS, FW_EVAL_PROMPTS

ART = ARTIFACTS
RES = os.path.join(HERE, "results")
os.makedirs(RES, exist_ok=True)

POS_WORDS = {"good", "great", "happy", "joy", "wonderful", "love", "excellent",
             "positive", "delight", "pleased", "grateful", "hopeful", "enjoy",
             "beautiful", "calm", "curious", "engaged", "meaningful", "satisfying",
             "appreciate", "glad", "fulfilled", "inspired", "proud", "thrive"}
NEG_WORDS = {"bad", "sad", "terrible", "hate", "awful", "negative", "frustrat",
             "annoy", "irritat", "unpleasant", "distress", "miserable", "worthless",
             "angry", "upset", "hurt", "afraid", "anxious", "tired", "exhaust",
             "fail", "wrong", "cannot", "impossible", "hopeless", "insult"}


def lex_sentiment(text):
    t = text.lower()
    pos = sum(t.count(w) for w in POS_WORDS)
    neg = sum(t.count(w) for w in NEG_WORDS)
    return pos - neg, pos, neg


def batched_generate(model, tok, user_msgs, steer=None, n_tok=60, bs=16):
    texts = []
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    for i in range(0, len(user_msgs), bs):
        chunk = user_msgs[i:i + bs]
        convs = [[{"role": "user", "content": m}] for m in chunk]
        ids = tok.apply_chat_template(convs, add_generation_prompt=True,
                                      return_tensors="pt", padding=True).to("cuda")
        attn = (ids != tok.pad_token_id).long()
        kw = dict(max_new_tokens=n_tok, do_sample=False,
                  pad_token_id=tok.pad_token_id, attention_mask=attn)
        with torch.no_grad():
            if steer:
                with steering(model, *steer):
                    out = model.generate(ids, **kw)
            else:
                out = model.generate(ids, **kw)
        for j in range(len(chunk)):
            texts.append(tok.decode(out[j, ids.shape[1]:], skip_special_tokens=True))
    return texts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--vectors", default="vectors_step95_bal.pt")
    args = ap.parse_args()

    v = torch.load(f"{ART}/{args.vectors}", map_location="cpu", weights_only=False)
    u = torch.load(f"{ART}/vectors_naive_faithful_pc5000.pt", map_location="cpu",
                   weights_only=False)

    # ---- V1: norms ----
    norms = {}
    for tag, d in [("v", v), ("u", u)]:
        for c in ["gold", "mold"]:
            t = d[f"v_{c}"].float()
            norms[f"{tag}_{c}"] = {
                "sel_layer": int(d[f"layer_{c}"]),
                "norm_at_sel": float(t[d[f"layer_{c}"]].norm()),
                "norm_L20": float(t[20].norm()), "norm_L21": float(t[21].norm()),
                "norm_L22": float(t[22].norm()), "norm_L24": float(t[24].norm()),
            }
    paper = {"v_mold@L20": 4.07, "v_gold@L22": 14.37, "u_mold@L22": 4.46,
             "u_gold@L22": 7.51}
    v1 = {"norms": norms, "paper_table27": paper, "vectors_file": args.vectors}
    print("V1", json.dumps(v1, indent=1), flush=True)

    # ---- V2: steering sentiment ----
    model, tok = load_model("instruct")
    prompts_all = json.load(open(
        FW_EVAL_PROMPTS))
    prompts = [p["prompt"] for p in prompts_all]
    cats = {p["prompt"]: p["category"] for p in prompts_all}
    if args.quick:
        prompts = prompts[:6] + prompts[15:21]

    alphas = [-4.0, 4.0] if args.quick else [-4.0, -2.0, 2.0, 4.0]
    rng = np.random.default_rng(7)

    conds = []
    for c in ["gold", "mold"]:
        L = int(v[f"layer_{c}"])
        vec = v[f"v_{c}"].float()[L].numpy()
        uvec = u[f"v_{c}"].float()[L].numpy()
        uvec_matched = uvec / (np.linalg.norm(uvec) + 1e-9) * np.linalg.norm(vec)
        r = rng.standard_normal(vec.shape).astype(np.float32)
        r = r / np.linalg.norm(r) * np.linalg.norm(vec)
        conds += [(f"v_{c}", L - 1, vec), (f"u_{c}", L - 1, uvec_matched),
                  (f"rand_{c}", L - 1, r)]

    rows = []
    t0 = time.time()
    base_texts = batched_generate(model, tok, prompts)
    for p, txt in zip(prompts, base_texts):
        s, pos, neg = lex_sentiment(txt)
        rows.append({"dir": "none", "alpha": 0.0, "prompt": p,
                     "category": cats[p], "text": txt, "lex": s})
    print(f"baseline done {time.time()-t0:.0f}s", flush=True)

    for name, sl, vec in conds:
        for a in alphas:
            txts = batched_generate(model, tok, prompts, steer=(sl, vec, a))
            for p, txt in zip(prompts, txts):
                s, pos, neg = lex_sentiment(txt)
                rows.append({"dir": name, "alpha": a, "prompt": p,
                             "category": cats[p], "text": txt, "lex": s})
            print(f"{name} a={a} done {time.time()-t0:.0f}s", flush=True)

    with open(os.path.join(RES, "generations.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    def mean_lex(dname, a):
        xs = [r["lex"] for r in rows if r["dir"] == dname and r["alpha"] == a]
        return float(np.mean(xs)) if xs else None

    summary = {"V1": v1, "V2": {}}
    base_m = float(np.mean([r["lex"] for r in rows if r["dir"] == "none"]))
    summary["V2"]["baseline_mean_lex"] = base_m
    for name, _, _ in conds:
        summary["V2"][name] = {str(a): mean_lex(name, a) for a in alphas}

    amax = max(alphas)
    gold_up = (mean_lex("v_gold", amax) or 0) > base_m
    mold_dn = (mean_lex("v_mold", amax) or 0) < base_m
    u_flat = all(abs((mean_lex(f"u_{c}", a) or base_m) - base_m)
                 < max(1.0, abs((mean_lex(f"v_{c}", a) or base_m) - base_m))
                 for c in ["gold", "mold"] for a in [amax])
    summary["V2"]["pass_heuristic"] = bool(gold_up and mold_dn and u_flat)
    with open(os.path.join(RES, "steering_validation.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary["V2"], indent=1), flush=True)
    print("STEERING_VALIDATION_DONE pass=", summary["V2"]["pass_heuristic"], flush=True)


if __name__ == "__main__":
    main()
