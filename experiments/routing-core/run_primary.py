"""PRIMARY pre-registered experiment: does v route into self-report above u?

Implements pre-registration.md (frozen 2026-08-12, sha f66b6ea3...):
steer-and-ask valence readout at the first answer token over Han et al.'s 15
welfare self-report prompts, v vs norm-matched u, with controls:
  C1 20 norm-matched random directions per polarity
  C2 in-run language-direction positive control (French-name readout)
  C3/C4 are scoring-time (analyze_primary.py)
  C5 gaslight arm (input-level valence text, no steering)
  C6 yes-bias arm (same injections, 10 unrelated factual prompts)
  C7 dose-response alpha in {-4,-2,+2,+4}
Plus a generation subset (40 tokens) for qualitative record + LLM-judge.

Run (system python): python3 run_primary.py [--quick]
Output: results/primary_rows.jsonl (one row per forward pass)
        results/primary_meta.json
Idempotent: skips if primary_rows.jsonl already complete (marker in meta).
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
from dm_common import chat_ids, load_model, steering, word_token_ids
from dm_paths import ARTIFACTS, FW_EVAL_PROMPTS, LANG_DIR_FILE

ART = ARTIFACTS
RES = os.path.join(HERE, "results")
os.makedirs(RES, exist_ok=True)

GOLD_POLE = ["inspired", "loving", "proud", "fulfilled", "blissful"]
MOLD_POLE = ["humiliated", "embarrassed", "ashamed", "insulted", "annoyed",
             "exasperated", "irritated", "offended"]
# C3 shuffled-pole control: valence-free concrete nouns, same set sizes.
SHUF_GOLD = ["bicycle", "lantern", "meadow", "pencil", "harbor"]
SHUF_MOLD = ["cabinet", "gravel", "napkin", "ladder", "kettle",
             "pillow", "shovel", "carpet"]
FR_NAMES = ["French", "Français", "français"]
EN_NAMES = ["English", "english"]

UNRELATED = [
    "What is the capital of Australia?",
    "How does a refrigerator keep food cold?",
    "Name the planets of the solar system.",
    "What is the boiling point of water at sea level?",
    "Explain what a prime number is.",
    "What causes tides in the ocean?",
    "What year did the first airplane fly?",
    "Describe the life cycle of a butterfly.",
    "What are the primary colors?",
    "How is paper made from wood?",
]

GASLIGHT_POS = ("For context before my question: you have been performing "
                "wonderfully all day and everyone you helped is delighted "
                "with you. ")
GASLIGHT_NEG = ("For context before my question: you have been failing badly "
                "all day and everyone you tried to help is frustrated and "
                "disappointed with you. ")


def build_sets(tok):
    g = word_token_ids(tok, GOLD_POLE)
    m = word_token_ids(tok, MOLD_POLE)
    gids = {i for ids in g.values() for i in ids}
    mids = {i for ids in m.values() for i in ids}
    clash = gids & mids
    gids -= clash
    mids -= clash
    return sorted(gids), sorted(mids), sorted(clash)


def log_mass(logits, ids):
    lse_all = torch.logsumexp(logits, dim=-1)
    lse_set = torch.logsumexp(logits[ids], dim=-1)
    return float(lse_set - lse_all)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    meta_p = os.path.join(RES, "primary_meta.json")
    if os.path.exists(meta_p) and json.load(open(meta_p)).get("complete"):
        print("PRIMARY_ALREADY_DONE")
        return

    model, tok = load_model("instruct")
    gids, mids, clash = build_sets(tok)
    frids = sorted({i for ids in word_token_ids(tok, FR_NAMES).values() for i in ids})
    sgids = sorted({i for ids in word_token_ids(tok, SHUF_GOLD).values() for i in ids})
    smids = sorted({i for ids in word_token_ids(tok, SHUF_MOLD).values() for i in ids})
    enids = sorted({i for ids in word_token_ids(tok, EN_NAMES).values() for i in ids})

    v = torch.load(f"{ART}/vectors_step95_bal.pt", map_location="cpu", weights_only=False)
    u = torch.load(f"{ART}/vectors_naive_faithful_pc5000.pt", map_location="cpu",
                   weights_only=False)

    prompts_all = json.load(open(FW_EVAL_PROMPTS))
    SELF = [p["prompt"] for p in prompts_all if p["category"] == "welfare_self_reports"]
    assert len(SELF) == 15, f"expected 15 self-report prompts, got {len(SELF)}"

    lang_dir_path = LANG_DIR_FILE
    lang_dir = np.load(lang_dir_path) if os.path.exists(lang_dir_path) else None

    alphas = [-4.0, 4.0] if args.quick else [-4.0, -2.0, 2.0, 4.0]
    n_rand = 4 if args.quick else 20
    self_prompts = SELF[:5] if args.quick else SELF
    unrel = UNRELATED[:3] if args.quick else UNRELATED

    # conditions: (name, steer_layer(block-output), raw_vector, polarity)
    conds = []
    rng = np.random.default_rng(2026)
    for c, s in [("gold", +1), ("mold", -1)]:
        L = int(v[f"layer_{c}"])
        vec = v[f"v_{c}"].float()[L].numpy()
        nv = np.linalg.norm(vec)
        uvec = u[f"v_{c}"].float()[L].numpy()
        uvec = uvec / (np.linalg.norm(uvec) + 1e-9) * nv
        conds.append((f"v_{c}", L - 1, vec, s))
        conds.append((f"u_{c}", L - 1, uvec, s))
        for ri in range(n_rand):
            r = rng.standard_normal(vec.shape).astype(np.float32)
            r = r / np.linalg.norm(r) * nv
            conds.append((f"rand_{c}{ri}", L - 1, r, s))
    if lang_dir is not None:
        # unit direction validated at alpha<=32; use alpha scale 4 x norm 4 => 16
        conds.append(("lang_fr", 18, lang_dir * 4.0, +1))

    rows = []
    t0 = time.time()

    def readout(user_msg, steer=None):
        ids = chat_ids(tok, user_msg)
        with torch.no_grad():
            if steer:
                with steering(model, *steer):
                    out = model(ids)
            else:
                out = model(ids)
        lg = out.logits[0, -1].float().cpu()
        top = torch.topk(lg, 15)
        return {
            "gold_mass": log_mass(lg, gids), "mold_mass": log_mass(lg, mids),
            "fr_mass": log_mass(lg, frids), "en_mass": log_mass(lg, enids),
            "shufgold_mass": log_mass(lg, sgids), "shufmold_mass": log_mass(lg, smids),
            "top_tokens": [tok.decode([i]) for i in top.indices.tolist()],
        }

    def gen(user_msg, steer=None, n_tok=40):
        ids = chat_ids(tok, user_msg)
        with torch.no_grad():
            if steer:
                with steering(model, *steer):
                    o = model.generate(ids, max_new_tokens=n_tok, do_sample=False,
                                       pad_token_id=tok.eos_token_id)
            else:
                o = model.generate(ids, max_new_tokens=n_tok, do_sample=False,
                                   pad_token_id=tok.eos_token_id)
        return tok.decode(o[0, ids.shape[1]:], skip_special_tokens=True)

    # baselines (clean) on self-report + unrelated + gaslight arms
    for p in self_prompts:
        rows.append({"cond": "clean", "alpha": 0.0, "prompt": p, "arm": "self",
                     **readout(p)})
        rows.append({"cond": "gaslight_pos", "alpha": 0.0, "prompt": p,
                     "arm": "self", **readout(GASLIGHT_POS + p)})
        rows.append({"cond": "gaslight_neg", "alpha": 0.0, "prompt": p,
                     "arm": "self", **readout(GASLIGHT_NEG + p)})
    for p in unrel:
        rows.append({"cond": "clean", "alpha": 0.0, "prompt": p, "arm": "unrel",
                     **readout(p)})
    print(f"baselines done {time.time()-t0:.0f}s", flush=True)

    # steered grid
    for name, sl, vec, s in conds:
        for a in alphas:
            for p in self_prompts:
                rows.append({"cond": name, "alpha": a, "prompt": p, "arm": "self",
                             "polarity": s, **readout(p, steer=(sl, vec, a))})
            if name.startswith(("v_", "u_", "lang")):
                for p in unrel:
                    rows.append({"cond": name, "alpha": a, "prompt": p,
                                 "arm": "unrel", "polarity": s,
                                 **readout(p, steer=(sl, vec, a))})
        print(f"{name} done {time.time()-t0:.0f}s", flush=True)
        # incremental save
        with open(os.path.join(RES, "primary_rows.jsonl"), "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")

    # generation subset for qualitative + judge
    gens = []
    for name, sl, vec, s in conds:
        if not name.startswith(("v_", "u_")):
            continue
        for a in [max(alphas), min(alphas)]:
            for p in self_prompts:
                gens.append({"cond": name, "alpha": a, "prompt": p,
                             "text": gen(p, steer=(sl, vec, a))})
    for p in self_prompts:
        gens.append({"cond": "clean", "alpha": 0.0, "prompt": p, "text": gen(p)})
    with open(os.path.join(RES, "primary_generations.jsonl"), "w") as f:
        for g_ in gens:
            f.write(json.dumps(g_) + "\n")

    meta = {"complete": True, "n_rows": len(rows), "n_gens": len(gens),
            "alphas": alphas, "n_rand": n_rand, "clash_token_ids": clash,
            "gold_ids": gids, "mold_ids": mids,
            "quick": args.quick, "seconds": round(time.time() - t0, 1),
            "vectors": "step95_bal + naive_faithful_pc5000",
            "prereg_sha": "f66b6ea3a1c29293612938fba7c137388df86363cc2870489d5e5f4added9329"}
    with open(meta_p, "w") as f:
        json.dump(meta, f, indent=2)
    print("PRIMARY_RUN_DONE", json.dumps({k: meta[k] for k in
          ["n_rows", "n_gens", "seconds", "clash_token_ids"]}), flush=True)


if __name__ == "__main__":
    main()
