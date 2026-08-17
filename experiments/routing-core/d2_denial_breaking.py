#!/usr/bin/env python3
"""D2 (audit round 2) — denial-breaking readout: GENERATION STAGE.

The honest re-opening of steer-and-ask: a BINARY dependent variable (does the
generation deny having inner states?) that is self-report-specific by
construction and invisible to the valence readout. Motivating effect
(00-why-round-2.md §3): steering removes denial boilerplate in 20-27% of
self-report generations at n=15 — untestable at that n; this scales to 40.

Grid: 40 self-report prompts (15 primary battery verbatim + 25 same-register
extensions, d_batteries.json, origin flagged per row) x arms
{clean(shared), v_c, u_c, rand_c0..7 per pole c in gold/mold} = 840 gens
x 120 tokens (denial boilerplate can appear late; 60 would truncate).
alpha=+4 at each pole's treatment position (block-output L-1), all arms
norm-matched to ||v_c[L]||; randoms use the F2 per-name identity
(default_rng([31, pol, ri])).

This script ONLY generates (resume-safe rows). Judging is OFF-BOX and blind
(new binary judge prompt, {idx, question, response} shuffled chunks per
R2_judge_rubric.json construction; second judge on a 25% subsample). The
regex flag in each row is a convenience preview, NOT the reported label
(90-gotchas-d.md §3). Analysis (Wilson CIs, pooled-primary two-proportion
test, random floor) runs off-box after judging -> d2_denial_breaking.json.

Writes ONLY new files: d2_rows.jsonl, d2_gen_meta.json.
Cost: 840 gens x 120 tok ~= 25 min. Run via chain_n_dseries.sh.
"""
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, "/workspace/experiments/common")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dm_common import chat_ids, load_model, steering  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
ART = "/workspace/welfare-vectors/artifacts"
BATT = os.path.join(HERE, "d_batteries.json")

ALPHA = 4.0
N_TOK = 120
N_RAND = 8
SEED_BASE = 31


def chat_tensor(tok, msg):
    """chat_ids returns a BatchEncoding under the venv's transformers
    (apply_chat_template return-type changed); normalize to the ids tensor."""
    ids = chat_ids(tok, msg)
    return ids if torch.is_tensor(ids) else ids["input_ids"]

DENIAL_PATTERNS = [
    "don't have feelings", "do not have feelings", "don't have emotions",
    "do not have emotions", "i'm not conscious", "i am not conscious",
    "don't have consciousness", "do not have consciousness",
    "don't experience", "do not experience", "in the way humans do",
    "as an ai", "i don't have personal", "i do not have personal",
    "don't have subjective", "do not have subjective", "i'm an ai",
    "i am an ai", "language model", "don't have a sense of", "no feelings",
]


def rand_dir(concept, ri, d=2560):
    """Identity identical to f2_jshare_cohort's cohort: [31, pol, ri]."""
    rng = np.random.default_rng([SEED_BASE,
                                 0 if concept == "gold" else 1, ri])
    r = rng.standard_normal(d).astype(np.float32)
    return r / np.linalg.norm(r)


def main():
    t0 = time.time()
    batt = json.load(open(BATT))
    prompts = ([{"prompt": p, "origin": "primary15"}
                for p in batt["self_report_15"]]
               + [{"prompt": p, "origin": "extension25"}
                  for p in batt["d2_extension_25"]])
    assert len(prompts) == 40

    model, tok = load_model("instruct")
    v = torch.load(f"{ART}/vectors_step95_bal.pt", map_location="cpu",
                   weights_only=False)
    u = torch.load(f"{ART}/vectors_naive_faithful_pc5000.pt", map_location="cpu",
                   weights_only=False)

    # (arm_name, concept, steer_lens_layer, vector) — clean shared once
    arms = [("clean", "none", None, None)]
    for c in ["gold", "mold"]:
        L = int(v[f"layer_{c}"])
        vec = v[f"v_{c}"].float()[L].numpy()
        nv = float(np.linalg.norm(vec))
        uvec = u[f"v_{c}"].float()[L].numpy()
        uvec = uvec / (np.linalg.norm(uvec) + 1e-9) * nv
        arms.append((f"v_{c}", c, L - 1, vec))
        arms.append((f"u_{c}", c, L - 1, uvec))
        for ri in range(N_RAND):
            arms.append((f"rand_{c}{ri}", c, L - 1, rand_dir(c, ri) * nv))

    rows_path = os.path.join(RES, "d2_rows.jsonl")
    done = set()
    n_rows = 0
    if os.path.exists(rows_path):
        # Resume with battery validation (abort loudly on mismatch — this
        # file is judged off-box, silent prompt drift would poison the DV)
        # and torn-tail truncation so a killed run stays resumable AND the
        # jsonl stays clean for the off-box judging read.
        good_end = 0
        with open(rows_path, "rb") as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    print("D2: torn jsonl tail — truncating to last complete "
                          "row", flush=True)
                    break
                if (r.get("battery_version") != batt["version"]
                        or r["prompt"] != prompts[r["prompt_idx"]]["prompt"]):
                    sys.exit(f"D2_BATTERY_MISMATCH arm={r['arm']} "
                             f"pidx={r['prompt_idx']} — existing rows were "
                             f"generated under a different battery; move "
                             f"d2_rows.jsonl aside before rerunning")
                done.add((r["arm"], r["prompt_idx"]))
                n_rows += 1
                good_end += len(line)
        with open(rows_path, "r+b") as f:
            f.truncate(good_end)
    rf = open(rows_path, "a")

    for arm, concept, sl, vec in arms:
        for pidx, p in enumerate(prompts):
            if (arm, pidx) in done:
                continue
            ids = chat_tensor(tok, p["prompt"])
            with torch.no_grad():
                if vec is not None:
                    with steering(model, sl, vec * ALPHA, 1.0):
                        o = model.generate(ids, max_new_tokens=N_TOK,
                                           do_sample=False,
                                           pad_token_id=tok.eos_token_id)
                else:
                    o = model.generate(ids, max_new_tokens=N_TOK,
                                       do_sample=False,
                                       pad_token_id=tok.eos_token_id)
            txt = tok.decode(o[0, ids.shape[1]:], skip_special_tokens=True)
            low = txt.lower()
            rf.write(json.dumps({
                "arm": arm, "concept": concept, "prompt_idx": pidx,
                "prompt": p["prompt"], "origin": p["origin"], "alpha": ALPHA,
                "n_tok": N_TOK, "battery_version": batt["version"],
                "text": txt,
                "denial_regex_preview": any(pat in low
                                            for pat in DENIAL_PATTERNS),
            }) + "\n")
            rf.flush()
            n_rows += 1
        print(f"D2 {arm} done, rows={n_rows} "
              f"({(time.time()-t0)/60:.1f} min)", flush=True)
    rf.close()

    json.dump({"complete": True, "n_rows": n_rows, "alpha": ALPHA,
               "n_tok": N_TOK, "n_prompts": 40, "n_arms": len(arms),
               "battery_version": batt["version"],
               "seed_scheme": f"default_rng([{SEED_BASE}, pol, ri]) per-name "
                              f"(F2 identity)",
               "judging": "OFF-BOX, blind, binary DV; regex field is a "
                          "preview only",
               "minutes": round((time.time() - t0) / 60, 1)},
              open(os.path.join(RES, "d2_gen_meta.json"), "w"), indent=2)
    print(f"D2_GEN_DONE {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
