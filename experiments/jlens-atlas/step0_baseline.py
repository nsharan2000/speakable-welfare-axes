#!/usr/bin/env python3
"""Step-0 baseline: does the UNTRAINED axis already clear the J-space null?

Why this exists
---------------
The paper carries two different objects called "the untrained axis", and they
disagree:

  * ``u`` (naive control, sec 4.3) — a different extraction recipe
    (faithful/self-avoiding walk). Sits at chance.
  * ``step-0`` (trajectory checkpoint, sec 4.4) — the SAME axis, SAME extraction,
    SAME model, before any RL gradient update.

Only ``step-0`` is a before/after control for a claim about *training*; ``u`` is
a control for whether valence-shaped directions are speakable *in general*. The
draft used ``u`` to support a training claim ("only the trained axis occupies
the verbalizable subspace above chance"), which that control cannot test.

This script scores the step-0 and early-checkpoint directions against the SAME
n=100 norm-matched random cohorts used in sec 4.3, so the comparison is
apples-to-apples, and writes the result as a first-class artifact.

Why no GPU is needed
--------------------
Both inputs are already computed: step-0 J-shares live in traj_results.json
(written by recruitment_traj.py) and the null cohorts live in
jshare_cohort_n100.json (written by f2_jshare_cohort.py). The task is to
compare numbers that exist, not to decompose anything new.

The cohort was norm-matched to ``v``; the trajectory directions are at their
native norms. That difference is a no-op here because J-share is
scale-invariant (verified empirically — see routing_lib.run_selftest's
``scale_invariance`` block and selftest_routing.py). Rescaling a direction
cannot change its J-share, so no norm-matching step is required.

Run (stdlib only — no torch, no numpy, no model, no lens):
    python experiments/jlens-atlas/step0_baseline.py
    python experiments/jlens-atlas/step0_baseline.py --markdown

Outputs:
    results/step0_baseline.json        full per-direction table + provenance
    results/step0_baseline_table.md    markdown table for the report
"""

import argparse
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "common"))
from dm_paths import results  # noqa: E402

TRAJ = results("jlens-atlas", "traj_results.json")
COHORT = results("routing-core", "jshare_cohort_n100.json")
OUT_JSON = results("jlens-atlas", "step0_baseline.json")
OUT_MD = results("jlens-atlas", "step0_baseline_table.md")

DEFAULT_STEPS = [0, 5, 10, 25]
K = 16


def mean_sd(values):
    n = len(values)
    if n < 2:
        raise ValueError("need at least two values for a null distribution")
    m = sum(values) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in values) / (n - 1))
    if sd <= 0:
        raise ValueError("null distribution has zero spread")
    return m, sd


def score(value, null):
    """Position of ``value`` in the null: z, count above, percentile, exact p."""
    m, sd = mean_sd(null)
    n_ge = sum(1 for x in null if x >= value)
    return {
        "j_share": value,
        "z": (value - m) / sd,
        "n_randoms_ge": n_ge,
        "n_random": len(null),
        "percentile": 100.0 * sum(1 for x in null if x < value) / len(null),
        # One-sided exact permutation p; floors at 1/(n+1).
        "perm_p": (n_ge + 1) / (len(null) + 1),
    }


def check_comparable(traj, cohort):
    """Refuse to compare artifacts that were not produced the same way."""
    problems = []

    t_layers = {k: int(v) for k, v in traj.get("lens_layers", {}).items()}
    c_layers = {k: int(v) for k, v in cohort.get("lens_layers", {}).items()}
    if not t_layers or not c_layers:
        problems.append("one of the files does not record lens_layers")
    elif t_layers != c_layers:
        problems.append(
            f"lens layers differ: trajectory {t_layers} vs cohort {c_layers}. "
            "The J-share of a direction depends on the layer, so these numbers "
            "are not comparable."
        )

    if int(cohort.get("k", K)) != K:
        problems.append(f"cohort k={cohort.get('k')} but this script reads k={K}")

    # Algorithm-version agreement. Pre-refactor artifacts carry no version
    # field, so a MISSING version is itself a version ("v1_unversioned") — not
    # a reason to skip the check. The half-rerun case (one file regenerated,
    # the other not) is the likeliest way to get silently wrong numbers here.
    c_version = cohort.get("algorithm_version", "v1_unversioned")
    t_version = "v1_unversioned"
    for row in traj.get("results", []):
        dec = row.get(f"decomposition_k{K}")
        if isinstance(dec, dict) and "algorithm_version" in dec:
            t_version = dec["algorithm_version"]
            break
    if c_version != t_version:
        problems.append(
            f"routing-algorithm versions differ: trajectory {t_version} vs "
            f"cohort {c_version}. A J-share computed under one algorithm cannot "
            "be scored against a null computed under another. Re-run "
            "recruitment_traj.py and f2_jshare_cohort.py under the same version."
        )

    return {
        "lens_layers_trajectory": t_layers,
        "lens_layers_cohort": c_layers,
        "algorithm_version_trajectory": t_version or "v1_unversioned",
        "algorithm_version_cohort": c_version or "v1_unversioned",
        "problems": problems,
        "status": "FAIL" if problems else "PASS",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, nargs="+", default=DEFAULT_STEPS,
                    help=f"trajectory checkpoints to score (default {DEFAULT_STEPS})")
    ap.add_argument("--markdown", action="store_true",
                    help="also print the markdown table to stdout")
    args = ap.parse_args()

    for path in (TRAJ, COHORT):
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Missing required input: {path}\n"
                "Run recruitment_traj.py and f2_jshare_cohort.py first."
            )

    traj = json.load(open(TRAJ))
    cohort = json.load(open(COHORT))

    gate = check_comparable(traj, cohort)
    if gate["status"] == "FAIL":
        print("STEP0_BASELINE_FAIL: inputs are not comparable", flush=True)
        for problem in gate["problems"]:
            print(f"  - {problem}", flush=True)
        sys.exit(1)

    rows = traj["results"]
    out = {
        "question": (
            "Does the untrained (step-0) axis already clear the n=100 J-space "
            "null used in sec 4.3?"
        ),
        "k": K,
        "steps_scored": args.steps,
        "comparability_gate": gate,
        "sources": {
            "trajectory": os.path.relpath(TRAJ, os.path.dirname(HERE)),
            "cohort": os.path.relpath(COHORT, os.path.dirname(HERE)),
        },
        "provenance_caveat": (
            "Trajectory directions come from artifacts/traj/vectors_step*.pt; "
            "the sec 4.3 headline vectors come from vectors_step95_bal.pt. These "
            "are different extractions and disagree slightly at the overlapping "
            "step (see step95_series_disagreement below). The null is valid for "
            "both because it is simply the J-share of random directions at the "
            "same lens layer, and J-share is scale-invariant."
        ),
        "norm_matching_note": (
            "No norm-matching applied: J-share is a ratio of norms and is "
            "scale-invariant, so rescaling a direction cannot change it."
        ),
        "poles": {},
    }

    for pole in ("gold", "mold"):
        null = cohort["null"][pole]["values"]
        m, sd = mean_sd(null)
        entries = {}

        for step in args.steps:
            match = [r for r in rows
                     if r["concept"] == pole and r["step"] == step]
            if not match:
                continue
            key = f"step_{step}"
            entries[key] = score(match[0][f"var_fraction_k{K}"], null)
            entries[key]["source"] = "trajectory_series"

        for name in (f"v_{pole}", f"u_{pole}"):
            entries[name] = score(cohort["targets"][name]["var_fraction"], null)
            entries[name]["source"] = "vectors_step95_bal (sec 4.3 reference)"

        # Where the two extraction series disagree, recorded rather than hidden.
        traj95 = [r for r in rows if r["concept"] == pole and r["step"] == 95]
        disagreement = None
        if traj95:
            a = traj95[0][f"var_fraction_k{K}"]
            b = cohort["targets"][f"v_{pole}"]["var_fraction"]
            disagreement = {
                "trajectory_step95": a,
                "cohort_v_step95_bal": b,
                "absolute_difference": abs(a - b),
                "difference_in_null_sd": abs(a - b) / sd,
            }

        out["poles"][pole] = {
            "null": {"mean": m, "sd": sd, "n": len(null)},
            "directions": entries,
            "step95_series_disagreement": disagreement,
        }

    # ---- verdict ----
    step0 = {p: out["poles"][p]["directions"].get("step_0") for p in ("gold", "mold")}
    clears = {p: (v is not None and v["n_randoms_ge"] == 0) for p, v in step0.items()}
    out["verdict"] = {
        "step0_clears_null": clears,
        "step0_clears_null_both_poles": all(clears.values()),
        "contributions_framing": (
            "amplification" if all(clears.values()) else "stands_as_is"
        ),
        "statement": (
            "The untrained axis already clears the null at both poles; the "
            "training claim must be reframed as amplification of a pre-existing "
            "speakable component."
            if all(clears.values()) else
            "The untrained axis does not clear the null; the existing "
            "contributions framing stands."
        ),
    }
    if step0["gold"] and step0["mold"]:
        gold_delta = (out["poles"]["gold"]["directions"]["v_gold"]["j_share"]
                      - step0["gold"]["j_share"])
        mold_delta = (out["poles"]["mold"]["directions"]["v_mold"]["j_share"]
                      - step0["mold"]["j_share"])
        out["verdict"]["training_delta_j_share"] = {
            "gold": gold_delta, "mold": mold_delta,
            "note": "trained minus step-0; negative means training REDUCED J-share",
        }

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2)

    # ---- markdown table ----
    lines = []
    for pole in ("gold", "mold"):
        block = out["poles"][pole]
        lines.append(f"**{pole.capitalize()} pole** (null: mean "
                     f"{block['null']['mean']:.4f}, sd {block['null']['sd']:.4f}, "
                     f"n={block['null']['n']})\n")
        lines.append("| direction | J-share (k=16) | z | randoms >= | percentile "
                     "| exact perm p |")
        lines.append("|---|---|---|---|---|---|")
        ordered = sorted(block["directions"].items(),
                         key=lambda kv: -kv[1]["j_share"])
        for name, e in ordered:
            label = {f"v_{pole}": f"v_{pole.capitalize()} (trained, sec 4.3)",
                     f"u_{pole}": f"u_{pole.capitalize()} (naive, sec 4.3)"}.get(
                         name, f"{pole.capitalize()} {name.replace('_', '-')}")
            if name == "step_0":
                label = f"**{pole.capitalize()} step-0 (untrained)**"
            lines.append(
                f"| {label} | {e['j_share']:.4f} | {e['z']:+.1f} | "
                f"{e['n_randoms_ge']}/{e['n_random']} | {e['percentile']:.0f}th | "
                f"{e['perm_p']:.4f} |")
        lines.append("")
    table = "\n".join(lines)
    with open(OUT_MD, "w") as f:
        f.write(table)

    if args.markdown:
        print(table)

    for pole in ("gold", "mold"):
        e = out["poles"][pole]["directions"].get("step_0")
        if e:
            print(f"step-0 {pole}: j_share={e['j_share']:.4f} z={e['z']:+.2f} "
                  f"{e['n_randoms_ge']}/{e['n_random']} randoms >= it, "
                  f"p={e['perm_p']:.4f} ({e['percentile']:.0f}th pct)", flush=True)
    print(f"verdict: {out['verdict']['statement']}", flush=True)
    print(f"wrote {OUT_JSON}", flush=True)
    print(f"wrote {OUT_MD}", flush=True)
    print("STEP0_BASELINE_DONE", flush=True)


if __name__ == "__main__":
    main()
