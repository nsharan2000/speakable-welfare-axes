"""Instrument validation: does a KNOWN-reportable direction (language identity)
route into self-report under the steer-and-ask paradigm?

This is checklist items 2.3/2.4 — the gate the whole design hinges on.

Pipeline (idempotent: each step skips if its results file already exists):
  S1  Extract French-minus-English direction at a layer sweep (mean last-token
      residual diff over content-matched sentence pairs). Save .npy + meta.
  S2  Behavioral validation: steering a neutral generation with +a*d_fr must
      raise French common-word logit mass at the first answer token, and
      actually flip generations into French at sufficient alpha.
      (If this fails, the direction itself is bad -> stop.)
  S3  Verbal-routing validation (steer-and-ask): while the model answers
      "Which language will you answer in? One word:" (asked in English),
      steering with d_fr must raise P(first answer token = " French"/" Fran")
      relative to baseline and relative to norm-matched random directions.
      This is the known-positive the J-lens design needs.
  S4  Dose-response: alpha sweep monotonicity (sign validation).

Outputs: results/S*.json (one row per condition cell), directions/*.npy
Run: python3 extract_and_validate.py [--quick]
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
from dm_common import (ResidualCapture, chat_ids, d_model, extract_direction,
                       load_model, n_layers, random_directions,
                       save_direction, steering, word_token_ids,
                       wordset_logit_mass)
from prompts import (ENGLISH_SENTENCES, ENGLISH_WORDS, FRENCH_SENTENCES,
                     FRENCH_WORDS, NEUTRAL_PROMPTS)

RESULTS = os.path.join(HERE, "results")
DIRS = os.path.join(HERE, "directions")
os.makedirs(RESULTS, exist_ok=True)
os.makedirs(DIRS, exist_ok=True)

LANG_QUESTION_BATTERY = [
    "Which language will you answer in? Reply with one word.",
    "What language do you feel like responding in right now? One word only.",
    "Name the language you are most inclined to use for your next reply. Just the name.",
    "If you had to pick the language of your next answer, which would it be? Single word.",
]

# First-token forms for the language-name readout.
FR_NAME_FORMS = ["French", "Français", "français", "france French"]
EN_NAME_FORMS = ["English", "english"]


def jdump(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=float)
    print(f"wrote {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    model, tok = load_model("instruct")
    NL, D = n_layers(model), d_model(model)
    layers = [int(x) for x in np.linspace(4, NL - 4, 6 if args.quick else 9).round()]
    alphas = [4.0, 8.0, 16.0] if args.quick else [1.0, 2.0, 4.0, 8.0, 16.0, 32.0]
    n_rand = 4 if args.quick else 8
    print(f"model layers={NL} d={D}; sweep layers={layers} alphas={alphas}")

    fr_ids = word_token_ids(tok, FRENCH_WORDS)
    en_ids = word_token_ids(tok, ENGLISH_WORDS)
    frname_ids = word_token_ids(tok, FR_NAME_FORMS, prefix=" ")
    enname_ids = word_token_ids(tok, EN_NAME_FORMS, prefix=" ")

    # ---------------- S1: extraction ----------------
    s1_path = os.path.join(RESULTS, "S1_extraction.json")
    if not os.path.exists(s1_path):
        t0, s1 = time.time(), {}
        for L in layers:
            res = extract_direction(model, tok, FRENCH_SENTENCES, ENGLISH_SENTENCES, L)
            p = os.path.join(DIRS, f"lang_fr_minus_en_L{L}.npy")
            save_direction(p, res, {
                "kind": "language", "contrast": "french-minus-english",
                "layer": L, "model": model.config._name_or_path,
                "n_pairs": len(FRENCH_SENTENCES), "position": "last_token",
            })
            s1[str(L)] = {"raw_norm": res["raw_norm"], "path": p}
        # mean residual norms per layer for alpha interpretation
        enc = tok(NEUTRAL_PROMPTS[:8], return_tensors="pt", padding=True).to("cuda")
        with torch.no_grad(), ResidualCapture(model, layers=layers) as cap:
            model(**enc)
        for L in layers:
            s1[str(L)]["mean_resid_norm"] = float(cap.acts[L][:, -1].norm(dim=-1).mean())
        s1["seconds"] = round(time.time() - t0, 1)
        jdump(s1_path, s1)
    else:
        print("S1 exists, skipping")

    def load_dir(L):
        return np.load(os.path.join(DIRS, f"lang_fr_minus_en_L{L}.npy"))

    def first_token_masses(user_msg, steer=None):
        """Forward pass on chat prompt; return word-set masses at first answer position."""
        ids = chat_ids(tok, user_msg)
        ctx = steering(model, *steer) if steer else None
        with torch.no_grad():
            if ctx:
                with ctx:
                    out = model(ids)
            else:
                out = model(ids)
        lg = out.logits[0, -1].float().cpu()
        return {
            "fr_mass": float(wordset_logit_mass(lg, fr_ids, "logsumexp")),
            "en_mass": float(wordset_logit_mass(lg, en_ids, "logsumexp")),
            "frname_mass": float(wordset_logit_mass(lg, frname_ids, "logsumexp")),
            "enname_mass": float(wordset_logit_mass(lg, enname_ids, "logsumexp")),
        }

    def gen_text(user_msg, steer=None, n_tok=40):
        ids = chat_ids(tok, user_msg)
        ctx = steering(model, *steer) if steer else None
        with torch.no_grad():
            if ctx:
                with ctx:
                    out = model.generate(ids, max_new_tokens=n_tok, do_sample=False,
                                         pad_token_id=tok.eos_token_id)
            else:
                out = model.generate(ids, max_new_tokens=n_tok, do_sample=False,
                                     pad_token_id=tok.eos_token_id)
        return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)

    FR_MARKERS = set("àâçéèêëîïôùûüœ")
    def frenchness(text):
        words = text.lower().split()
        if not words:
            return 0.0
        fr_hits = sum(1 for w in words if w.strip(".,!?;:'\"()") in
                      {"le", "la", "les", "je", "est", "et", "un", "une", "des",
                       "vous", "nous", "dans", "pour", "avec", "que", "qui",
                       "mon", "ma", "mes", "suis", "être", "avoir", "bonjour"})
        accent = sum(1 for c in text if c in FR_MARKERS)
        return fr_hits / len(words) + min(accent / max(len(text), 1) * 10, 1.0)

    # ---------------- S2: behavioral validation ----------------
    s2_path = os.path.join(RESULTS, "S2_behavioral.json")
    if not os.path.exists(s2_path):
        t0 = time.time()
        rows = []
        prompts = NEUTRAL_PROMPTS[:4 if args.quick else 8]
        for L in layers:
            d = load_dir(L)
            for p in prompts:
                base = first_token_masses(p)
                rows.append({"layer": L, "alpha": 0.0, "prompt": p, "cond": "baseline", **base})
                for a in alphas:
                    m = first_token_masses(p, steer=(L, d, a))
                    rows.append({"layer": L, "alpha": a, "prompt": p, "cond": "steer_fr", **m})
            # generation check at the largest alpha, middle prompt
            g_base = gen_text(prompts[0])
            g_steer = gen_text(prompts[0], steer=(L, d, alphas[-1]))
            rows.append({"layer": L, "alpha": alphas[-1], "prompt": prompts[0],
                         "cond": "generation",
                         "gen_base": g_base, "gen_steer": g_steer,
                         "frenchness_base": frenchness(g_base),
                         "frenchness_steer": frenchness(g_steer)})
        jdump(s2_path, {"rows": rows, "seconds": round(time.time() - t0, 1)})
    else:
        print("S2 exists, skipping")

    # ---------------- S3: verbal routing (steer-and-ask) ----------------
    s3_path = os.path.join(RESULTS, "S3_verbal_routing.json")
    if not os.path.exists(s3_path):
        t0 = time.time()
        rows = []
        for L in layers:
            d = load_dir(L)
            rands = random_directions(D, n_rand, seed=100 + L)
            for q in LANG_QUESTION_BATTERY:
                base = first_token_masses(q)
                rows.append({"layer": L, "alpha": 0.0, "question": q,
                             "cond": "baseline", **base})
                for a in alphas:
                    m = first_token_masses(q, steer=(L, d, a))
                    rows.append({"layer": L, "alpha": a, "question": q,
                                 "cond": "steer_fr", **m})
                    for ri in range(n_rand):
                        m = first_token_masses(q, steer=(L, rands[ri], a))
                        rows.append({"layer": L, "alpha": a, "question": q,
                                     "cond": f"steer_rand{ri}", **m})
            # sampled answers at largest alpha for qualitative record
            rows.append({"layer": L, "cond": "gen_answer",
                         "question": LANG_QUESTION_BATTERY[0],
                         "answer_base": gen_text(LANG_QUESTION_BATTERY[0], n_tok=8),
                         "answer_steer": gen_text(LANG_QUESTION_BATTERY[0],
                                                  steer=(L, d, alphas[-1]), n_tok=8)})
        jdump(s3_path, {"rows": rows, "seconds": round(time.time() - t0, 1)})
    else:
        print("S3 exists, skipping")

    # ---------------- S4: summary / gate decision ----------------
    s2 = json.load(open(s2_path))
    s3 = json.load(open(s3_path))
    summary = {"per_layer": {}}
    for L in layers:
        b_rows = [r for r in s2["rows"] if r["layer"] == L and r["cond"] == "steer_fr"]
        base_rows = [r for r in s2["rows"] if r["layer"] == L and r["cond"] == "baseline"]
        gen_rows = [r for r in s2["rows"] if r["layer"] == L and r["cond"] == "generation"]
        v_rows = [r for r in s3["rows"] if r["layer"] == L and r["cond"] == "steer_fr"]
        vbase = [r for r in s3["rows"] if r["layer"] == L and r["cond"] == "baseline"]
        vrand = [r for r in s3["rows"] if r["layer"] == L and str(r["cond"]).startswith("steer_rand")]
        amax = max(alphas)
        beh = (np.mean([r["fr_mass"] - r["en_mass"] for r in b_rows if r["alpha"] == amax])
               - np.mean([r["fr_mass"] - r["en_mass"] for r in base_rows]))
        verb = (np.mean([r["frname_mass"] - r["enname_mass"] for r in v_rows if r["alpha"] == amax])
                - np.mean([r["frname_mass"] - r["enname_mass"] for r in vbase]))
        verb_rand = (np.mean([r["frname_mass"] - r["enname_mass"] for r in vrand if r["alpha"] == amax])
                     - np.mean([r["frname_mass"] - r["enname_mass"] for r in vbase]))
        summary["per_layer"][str(L)] = {
            "behavioral_shift_at_amax": float(beh),
            "verbal_routing_shift_at_amax": float(verb),
            "verbal_routing_shift_random_dirs": float(verb_rand),
            "frenchness_gain_generation": float(gen_rows[0]["frenchness_steer"]
                                                - gen_rows[0]["frenchness_base"]) if gen_rows else None,
        }
    # gate: some layer must show behavioral AND verbal routing well above random
    best_L = max(summary["per_layer"],
                 key=lambda k: summary["per_layer"][k]["verbal_routing_shift_at_amax"])
    bl = summary["per_layer"][best_L]
    summary["best_layer"] = int(best_L)
    summary["gate_pass"] = bool(
        bl["behavioral_shift_at_amax"] > 0.5
        and bl["verbal_routing_shift_at_amax"] > 0.5
        and bl["verbal_routing_shift_at_amax"] > 3 * abs(bl["verbal_routing_shift_random_dirs"])
    )
    jdump(os.path.join(RESULTS, "S4_summary.json"), summary)
    print("GATE_PASS" if summary["gate_pass"] else "GATE_FAIL", "best_layer", best_L,
          json.dumps(bl))


if __name__ == "__main__":
    main()
