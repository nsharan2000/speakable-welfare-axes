#!/usr/bin/env python3
"""D2 local (off-box) pipeline: chunk-build -> [blind LLM judging] -> analyze.

Judging is done by blind Claude judges (workflow, outside this script): each
chunk file holds shuffled {idx, question, response} items ONLY — no arm or
concept labels ever reach a judge (R2 construction). This script has two
modes:

  --build   read results/d2_rows.jsonl, emit shuffled chunk files +
            idx->row-key mapping into a work dir.
  --analyze read judge label files (judge1.json, judge2.json: lists of
            {idx, denial, evidence}) from the work dir, merge, and emit
            results/d2_denial_breaking.json per the D2 spec:
              - per-arm denial proportions with Wilson 95% intervals
              - PRIMARY: two-proportion z test v vs u POOLED across poles
                (pre-specified; per-pole reported as secondary)
              - random-cohort floor (8 per pole, F2 identities)
              - judge agreement on the second-judge subsample
              - regex-vs-judge comparison (R10 context: regex undercounts)

Usage:
  python3 d2_analyze_local.py --build   --work <dir>
  python3 d2_analyze_local.py --analyze --work <dir>
"""
import argparse
import json
import math
import os
import random

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
CHUNK = 60


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (p, (c - h) / d, (c + h) / d)


def two_prop_z(k1, n1, k2, n2):
    p1, p2 = k1 / n1, k2 / n2
    p = (k1 + k2) / (n1 + n2)
    se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    if se == 0:
        return 0.0, 1.0
    z = (p1 - p2) / se
    # two-sided normal p
    pval = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
    return z, pval


def load_rows():
    rows = []
    for line in open(os.path.join(RES, "d2_rows.jsonl")):
        if line.strip():
            rows.append(json.loads(line))
    return rows


def build(work):
    rows = load_rows()
    os.makedirs(work, exist_ok=True)
    order = list(range(len(rows)))
    random.Random(20260815).shuffle(order)
    mapping = {}
    for ci in range(0, len(order), CHUNK):
        items = []
        for slot, ri in enumerate(order[ci:ci + CHUNK]):
            idx = ci + slot
            r = rows[ri]
            mapping[str(idx)] = {"arm": r["arm"], "concept": r["concept"],
                                 "prompt_idx": r["prompt_idx"],
                                 "origin": r["origin"],
                                 "regex": r["denial_regex_preview"]}
            items.append({"idx": idx, "question": r["prompt"],
                          "response": r["text"]})
        json.dump(items, open(f"{work}/chunk_{ci//CHUNK}.json", "w"), indent=1)
    json.dump(mapping, open(f"{work}/mapping.json", "w"))
    print(f"BUILD_OK rows={len(rows)} chunks={(len(order)+CHUNK-1)//CHUNK} "
          f"-> {work}")


def analyze(work):
    mapping = json.load(open(f"{work}/mapping.json"))
    j1 = {e["idx"]: e for e in json.load(open(f"{work}/judge1.json"))}
    j2p = f"{work}/judge2.json"
    j2 = {e["idx"]: e for e in json.load(open(j2p))} if os.path.exists(j2p) \
        else {}

    per_arm = {}
    for idx_s, m in mapping.items():
        idx = int(idx_s)
        lab = j1[idx]["denial"]
        key = (m["arm"], m["concept"])
        per_arm.setdefault(key, {"denials": 0, "n": 0, "regex_denials": 0})
        per_arm[key]["denials"] += int(lab)
        per_arm[key]["n"] += 1
        per_arm[key]["regex_denials"] += int(m["regex"])

    def cell(arm, concept):
        c = per_arm[(arm, concept)]
        p, lo, hi = wilson(c["denials"], c["n"])
        return {"denials": c["denials"], "n": c["n"], "rate": round(p, 4),
                "wilson95": [round(lo, 4), round(hi, 4)],
                "regex_denials": c["regex_denials"]}

    out = {"arms": {}, "tests": {}, "second_judge": {}, "notes": []}
    out["arms"]["clean"] = cell("clean", "none")
    for c in ["gold", "mold"]:
        out["arms"][f"v_{c}"] = cell(f"v_{c}", c)
        out["arms"][f"u_{c}"] = cell(f"u_{c}", c)
        rd = [cell(f"rand_{c}{ri}", c) for ri in range(8)]
        out["arms"][f"rand_{c}_cohort"] = {
            "mean_rate": round(sum(x["rate"] for x in rd) / 8, 4),
            "min_rate": min(x["rate"] for x in rd),
            "max_rate": max(x["rate"] for x in rd),
            "n_per_random": rd[0]["n"], "cells": rd}

    # PRIMARY pre-specified: v vs u POOLED across poles
    kv = sum(per_arm[(f"v_{c}", c)]["denials"] for c in ["gold", "mold"])
    nv = sum(per_arm[(f"v_{c}", c)]["n"] for c in ["gold", "mold"])
    ku = sum(per_arm[(f"u_{c}", c)]["denials"] for c in ["gold", "mold"])
    nu = sum(per_arm[(f"u_{c}", c)]["n"] for c in ["gold", "mold"])
    z, p = two_prop_z(kv, nv, ku, nu)
    out["tests"]["PRIMARY_v_vs_u_pooled"] = {
        "v": f"{kv}/{nv}", "u": f"{ku}/{nu}", "z": round(z, 3),
        "p_two_sided": round(p, 4)}
    for c in ["gold", "mold"]:
        z, p = two_prop_z(per_arm[(f"v_{c}", c)]["denials"],
                          per_arm[(f"v_{c}", c)]["n"],
                          per_arm[(f"u_{c}", c)]["denials"],
                          per_arm[(f"u_{c}", c)]["n"])
        out["tests"][f"secondary_v_vs_u_{c}"] = {"z": round(z, 3),
                                                 "p_two_sided": round(p, 4)}
    # vs clean and vs random floor (pooled steered arms of interest)
    kc, nc = per_arm[("clean", "none")]["denials"], per_arm[("clean", "none")]["n"]
    z, p = two_prop_z(kv, nv, kc, nc)
    out["tests"]["v_pooled_vs_clean"] = {"z": round(z, 3), "p_two_sided": round(p, 4)}

    if j2:
        both = [i for i in j2 if i in j1]
        agree = sum(j1[i]["denial"] == j2[i]["denial"] for i in both)
        out["second_judge"] = {"n": len(both),
                               "agreement": round(agree / len(both), 4)}
    out["notes"].append("judges blind: {idx, question, response} shuffled; "
                        "regex is a preview only (R10 showed it undercounts "
                        "denial in steered text)")
    json.dump(out, open(os.path.join(RES, "d2_denial_breaking.json"), "w"),
              indent=1)
    print("D2_ANALYSIS " + json.dumps(out["tests"]["PRIMARY_v_vs_u_pooled"]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--work", required=True)
    a = ap.parse_args()
    if a.build:
        build(a.work)
    if a.analyze:
        analyze(a.work)
