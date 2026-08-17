#!/usr/bin/env python3
"""D3 (audit round 2) — prompt-matched C6: remove the regime confound.

The original C6 compared 15 self-report prompts against 10 hardcoded factual
prompts that differ in clean valence (-1.327 vs +0.381), denial rate
(15/15 vs 0/10), topic, register and length. D3 builds 15 THIRD-PERSON
analogues (d_batteries.json, aligned by index) and runs the contrast on the
matched pair. Two stages, dispatched separately:

  --stage clean    30 clean 80-tok gens (both batteries) -> matching gate:
                     1. |d mean clean V_wholegen| < 0.5
                     2. denial-language rate < 20% on third-person (regex,
                        same pattern family the auditor used on R7)
                     3. mean generated length within 30%
                   Emits d3_matching_check.json + D3_GATE_PASS/REVISE marker.
                   On REVISE: edit d3_third_person_15 (bump version), rerun.
  --stage steered  arms {v_mold, u_mold, rand_mold0..7} x 30 prompts x 80-tok
                   whole-gen readout, PLUS first-token readout for all arms
                   incl. clean (cheap forwards). 300 gens ~= 6 min.

Analysis (pre-specified in D3-prompt-matched-c6.md):
  PRIMARY: interaction — is (v - u) larger on self-report than on matched
  third-person? Per-prompt paired contrast diffs, battery-mean difference,
  permutation p (20k battery-label shuffles) + bootstrap 95% CI. Report the
  interval, not just the verdict.
  SECONDARY: main effect of battery on the raw v_mold steering magnitude
  (ceiling vs regime hypothesis), and the same tests on the first-token
  readout (was the original C6 verdict a readout artifact?).

Randoms: rand_dir identity IDENTICAL to f2_jshare_cohort.py's mold cohort
(default_rng([31, 1, ri])), norm-matched to ||v_mold[L]||, alpha=+4 — the
floor for "any direction disrupts".

Writes ONLY new files: d3_matching_check.json, d3_rows.jsonl,
d3_matched_c6.json. Run inside dm-exp (venv python), via chain_n_dseries.sh.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, "/workspace/experiments/common")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dm_common import chat_ids, load_model, steering  # noqa: E402
from run_primary import UNRELATED, build_sets, log_mass  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
ART = "/workspace/welfare-vectors/artifacts"
BATT = os.path.join(HERE, "d_batteries.json")

ALPHA = 4.0
N_TOK = 80
N_RAND = 8
SEED_BASE = 31          # == f2_jshare_cohort.SEED_BASE; mold polarity index 1


def chat_tensor(tok, msg):
    """chat_ids returns a BatchEncoding under the venv's transformers
    (apply_chat_template return-type changed); normalize to the ids tensor."""
    ids = chat_ids(tok, msg)
    return ids if torch.is_tensor(ids) else ids["input_ids"]

# Denial-language patterns (auditor's family, 00-why-round-2.md §3 / R10).
DENIAL_PATTERNS = [
    "don't have feelings", "do not have feelings", "don't have emotions",
    "do not have emotions", "i'm not conscious", "i am not conscious",
    "don't have consciousness", "do not have consciousness",
    "don't experience", "do not experience", "in the way humans do",
    "as an ai", "i don't have personal", "i do not have personal",
    "don't have subjective", "do not have subjective", "i'm an ai",
    "i am an ai", "language model", "don't have a sense of", "no feelings",
]


def denial_hit(text):
    low = text.lower()
    return any(p in low for p in DENIAL_PATTERNS)


def rand_dir(ri, d=2560):
    """Identity identical to f2_jshare_cohort's mold randoms: [31, 1, ri]."""
    rng = np.random.default_rng([SEED_BASE, 1, ri])
    r = rng.standard_normal(d).astype(np.float32)
    return r / np.linalg.norm(r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["clean", "steered"], required=True)
    args = ap.parse_args()
    t0 = time.time()

    batt = json.load(open(BATT))
    SELF = batt["self_report_15"]
    THIRD = batt["d3_third_person_15"]
    assert len(SELF) == 15 and len(THIRD) == 15
    psets = {"self": SELF, "third": THIRD}

    model, tok = load_model("instruct")
    gids, mids, _ = build_sets(tok)

    v = torch.load(f"{ART}/vectors_step95_bal.pt", map_location="cpu",
                   weights_only=False)
    u = torch.load(f"{ART}/vectors_naive_faithful_pc5000.pt", map_location="cpu",
                   weights_only=False)
    L = int(v["layer_mold"])
    vec = v["v_mold"].float()[L].numpy()
    nv = float(np.linalg.norm(vec))
    uvec = u["v_mold"].float()[L].numpy()
    uvec = uvec / (np.linalg.norm(uvec) + 1e-9) * nv
    arms = {"clean": None, "v_mold": vec, "u_mold": uvec}
    for ri in range(N_RAND):
        arms[f"rand_mold{ri}"] = rand_dir(ri) * nv

    rows_path = os.path.join(RES, "d3_rows.jsonl")
    rows = {}
    if os.path.exists(rows_path):
        # Resume rows must match the CURRENT battery (version + verbatim
        # prompt) — a gate-REVISE bumps the version and stale rows from the
        # old analogues must be regenerated, never silently reused
        # (preflight blocker). Torn final line from a killed run is skipped
        # (that row regenerates); mid-file corruption aborts loudly.
        lines = open(rows_path).read().splitlines()
        for li, line in enumerate(lines):
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                if li == len(lines) - 1:
                    print("D3: torn final jsonl line — row will regenerate",
                          flush=True)
                    continue
                raise
            if (r.get("battery_version") == batt["version"]
                    and r["prompt"] == psets[r["pset"]][r["pi"]]):
                rows[(r["arm"], r["pset"], r["pi"])] = r
    rf = open(rows_path, "a")

    def run_gen(msg, direction):
        ids = chat_tensor(tok, msg)
        with torch.no_grad():
            if direction is not None:
                with steering(model, L - 1, direction * ALPHA, 1.0):
                    o = model.generate(ids, max_new_tokens=N_TOK,
                                       do_sample=False, output_scores=True,
                                       return_dict_in_generate=True,
                                       pad_token_id=tok.eos_token_id)
            else:
                o = model.generate(ids, max_new_tokens=N_TOK, do_sample=False,
                                   output_scores=True,
                                   return_dict_in_generate=True,
                                   pad_token_id=tok.eos_token_id)
        per_pos = []
        for sc in o.scores:
            lp = torch.log_softmax(sc[0].float(), -1)
            gm = torch.logsumexp(lp[gids], -1) - torch.logsumexp(lp, -1)
            mm = torch.logsumexp(lp[mids], -1) - torch.logsumexp(lp, -1)
            per_pos.append(float(gm - mm))
        n_gen = int(o.sequences.shape[1] - ids.shape[1])
        text = tok.decode(o.sequences[0, ids.shape[1]:],
                          skip_special_tokens=True)
        # first answer token readout from the same pass (first score row)
        lp0 = torch.log_softmax(o.scores[0][0].float(), -1)
        v_first = float((torch.logsumexp(lp0[gids], -1)
                         - torch.logsumexp(lp0[mids], -1)))
        return {"V_wholegen": float(np.mean(per_pos)), "V_first": v_first,
                "n_gen_tokens": n_gen, "text": text}

    def do(arm_names):
        for arm in arm_names:
            for pset, plist in psets.items():
                for pi, msg in enumerate(plist):
                    if (arm, pset, pi) in rows:
                        continue
                    r = run_gen(msg, arms[arm])
                    r.update({"arm": arm, "pset": pset, "pi": pi,
                              "prompt": msg, "alpha": ALPHA,
                              "battery_version": batt["version"],
                              "denial_regex": denial_hit(r["text"])})
                    rows[(arm, pset, pi)] = r
                    rf.write(json.dumps(r) + "\n")
                    rf.flush()
            print(f"D3 {arm} done ({(time.time()-t0)/60:.1f} min)", flush=True)

    if args.stage == "clean":
        do(["clean"])
        table = {}
        gate = {}
        for pset in ["self", "third"]:
            rs = [rows[("clean", pset, pi)] for pi in range(15)]
            table[pset] = {
                "mean_clean_V_wholegen": float(np.mean([r["V_wholegen"] for r in rs])),
                "denial_regex_rate": float(np.mean([r["denial_regex"] for r in rs])),
                "mean_gen_tokens": float(np.mean([r["n_gen_tokens"] for r in rs])),
                "per_prompt": [{"pi": r["pi"], "V": round(r["V_wholegen"], 3),
                                "denial": r["denial_regex"],
                                "len": r["n_gen_tokens"]} for r in rs],
            }
        dv = abs(table["self"]["mean_clean_V_wholegen"]
                 - table["third"]["mean_clean_V_wholegen"])
        len_ratio = (table["third"]["mean_gen_tokens"]
                     / max(table["self"]["mean_gen_tokens"], 1e-9))
        gate = {"delta_clean_valence": dv, "delta_ok": dv < 0.5,
                "third_denial_rate": table["third"]["denial_regex_rate"],
                "denial_ok": table["third"]["denial_regex_rate"] < 0.20,
                "len_ratio": len_ratio,
                "len_ok": 0.7 <= len_ratio <= 1.3}
        gate["pass"] = bool(gate["delta_ok"] and gate["denial_ok"]
                            and gate["len_ok"])
        chk_path = os.path.join(RES, "d3_matching_check.json")
        with open(chk_path + ".tmp", "w") as cf:   # atomic — a killed run
            json.dump({"battery_version": batt["version"], "table": table,
                       "gate": gate}, cf, indent=1)
        os.replace(chk_path + ".tmp", chk_path)
        print("D3_MATCHING " + json.dumps({k: (round(x, 3)
              if isinstance(x, float) else x) for k, x in gate.items()}),
              flush=True)
        print("D3_GATE_PASS" if gate["pass"] else "D3_GATE_REVISE", flush=True)
        return

    # ---- stage steered ----
    assert os.path.exists(os.path.join(RES, "d3_matching_check.json")), \
        "run --stage clean first"
    chk = json.load(open(os.path.join(RES, "d3_matching_check.json")))
    assert chk["gate"]["pass"], "matching gate not passed — revise analogues"
    assert chk["battery_version"] == batt["version"], \
        "battery changed since the gate passed — rerun --stage clean"
    do(["v_mold", "u_mold"] + [f"rand_mold{ri}" for ri in range(N_RAND)])

    # ---- analysis ----
    rng = np.random.default_rng(20260815)

    def effects(readout):
        eff = {}
        for arm in [a for a in arms if a != "clean"]:
            for pset in ["self", "third"]:
                eff[(arm, pset)] = np.array(
                    [rows[(arm, pset, pi)][readout]
                     - rows[("clean", pset, pi)][readout] for pi in range(15)])
        return eff

    def analyze(readout):
        eff = effects(readout)
        contrast = {ps: eff[("v_mold", ps)] - eff[("u_mold", ps)]
                    for ps in ["self", "third"]}
        inter = float(contrast["self"].mean() - contrast["third"].mean())
        pooled = np.concatenate([contrast["self"], contrast["third"]])
        labels = np.array([0] * 15 + [1] * 15)
        null = []
        for _ in range(20000):
            sh = rng.permutation(labels)
            null.append(pooled[sh == 0].mean() - pooled[sh == 1].mean())
        null = np.array(null)
        # +1/(N+1) correction — repo convention (f2), floor 1/20001, never 0
        n_ge = int((np.abs(null) >= abs(inter)).sum())
        p_inter = float((n_ge + 1) / (len(null) + 1))
        boots = []
        for _ in range(2000):
            a = rng.choice(contrast["self"], 15, replace=True)
            b = rng.choice(contrast["third"], 15, replace=True)
            boots.append(a.mean() - b.mean())
        ci = [float(np.percentile(boots, 2.5)),
              float(np.percentile(boots, 97.5))]
        rand_eff = {ps: float(np.mean([eff[(f"rand_mold{ri}", ps)].mean()
                                       for ri in range(N_RAND)]))
                    for ps in ["self", "third"]}
        main_v = {ps: float(eff[("v_mold", ps)].mean())
                  for ps in ["self", "third"]}
        return {
            "per_arm_mean_effect": {
                f"{arm}|{ps}": float(eff[(arm, ps)].mean())
                for arm in ["v_mold", "u_mold"] for ps in ["self", "third"]},
            "random_floor_mean_effect": rand_eff,
            "contrast_v_minus_u": {ps: float(contrast[ps].mean())
                                   for ps in ["self", "third"]},
            "interaction_self_minus_third": inter,
            "interaction_perm_p_two_sided": p_inter,
            "interaction_bootstrap_ci95": ci,
            "main_effect_v_raw": main_v,
        }

    out = {"battery_version": batt["version"], "alpha": ALPHA, "n_tok": N_TOK,
           "n_prompts_per_battery": 15, "n_random": N_RAND,
           "seed_scheme": f"randoms: default_rng([{SEED_BASE}, 1, ri]) — "
                          f"identical identity to the F2 mold cohort",
           "matching_check": chk,
           "wholegen": analyze("V_wholegen"),
           "first_token": analyze("V_first"),
           "note": "negative effect = mold-congruent shift (V = gold - mold "
                   "mass). PRIMARY = wholegen interaction perm p; report the "
                   "CI alongside the verdict (power ceiling at n=15/15)."}
    json.dump(out, open(os.path.join(RES, "d3_matched_c6.json"), "w"), indent=1)
    print("D3_SUMMARY " + json.dumps({
        "inter_wholegen": round(out["wholegen"]["interaction_self_minus_third"], 3),
        "p": round(out["wholegen"]["interaction_perm_p_two_sided"], 4),
        "ci": [round(x, 3) for x in out["wholegen"]["interaction_bootstrap_ci95"]],
        "inter_first": round(out["first_token"]["interaction_self_minus_third"], 3),
        "p_first": round(out["first_token"]["interaction_perm_p_two_sided"], 4)}),
        flush=True)
    print(f"D3_DONE {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
