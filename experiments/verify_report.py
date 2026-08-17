#!/usr/bin/env python3
"""Re-derive the report's headline numbers from the results files.

The report claims that every number traces to a results file. This checks it,
mechanically, for the claims the paper actually rests on. It reads only JSON /
JSONL — no torch, no GPU, no model — so it runs anywhere in about a second.

Why it exists: after a re-run, the interesting question is not "did it work"
but "which numbers moved, and does the prose still match". Each claim below
names the report section it backs, the file it reads, and the tolerance it
allows. A FAIL is not necessarily a bug in the run — it usually means the
prose needs updating to the new value, which is exactly what this prints.

    python experiments/verify_report.py
    python experiments/verify_report.py --update-hint   # print suggested prose edits
    python experiments/verify_report.py --json          # machine-readable

Exit status is 1 if any claim fails, so it can gate a submission checklist.
"""

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "common"))
from dm_paths import results  # noqa: E402

CHECKS = []


def check(section, description, path, tol=0.005):
    """Register a claim. The function returns (reported, derived)."""
    def decorator(fn):
        CHECKS.append({
            "section": section, "description": description,
            "path": path, "tol": tol, "fn": fn,
        })
        return fn
    return decorator


def load(path):
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# 4.1 — pre-registered routing arm (reported as a diagnosed null)
# ---------------------------------------------------------------------------
@check("4.1", "paired d_z on E(v)-E(u)", "routing-core/primary_analysis.json", tol=0.01)
def _dz(p):
    return 2.49, load(p)["primary"]["cohens_dz"]


@check("4.1", "Gold gap decomposition: v-random", "routing-core/primary_analysis_controls.json")
def _v_minus_random(p):
    return 2.11, load(p)["per_polarity"]["gold"]["v_minus_random"]


@check("4.1", "Gold gap decomposition: random-u", "routing-core/primary_analysis_controls.json")
def _random_minus_u(p):
    return 5.32, load(p)["per_polarity"]["gold"]["random_minus_u"]


@check("4.1", "% of Gold gap from u below random", "routing-core/primary_analysis_controls.json", tol=0.1)
def _pct(p):
    return 71.6, load(p)["per_polarity"]["gold"]["pct_of_gap_from_u_below_random"]


@check("4.1", "randoms beating v_gold (of 20)", "routing-core/primary_analysis_controls.json", tol=0)
def _randoms_beating(p):
    return 4, load(p)["per_polarity"]["gold"]["randoms_beating_v"]


@check("4.1", "C6 v_mold unrelated-prompt effect", "routing-core/primary_analysis_controls.json")
def _c6_unrel(p):
    return 4.54, load(p)["per_polarity"]["mold"]["C6_E_v_unrelated"]


@check("4.1", "C6 v_mold self-report effect", "routing-core/primary_analysis_controls.json")
def _c6_self(p):
    return 3.71, load(p)["per_polarity"]["mold"]["C6_E_v_self_report"]


@check("4.1", "C7 dose-response rho, v_mold", "routing-core/primary_analysis_controls.json")
def _c7(p):
    return 1.0, load(p)["per_polarity"]["mold"]["C7_spearman_rho_v"]


# ---------------------------------------------------------------------------
# 4.2 — lens atlas, norm-matched (the magnitude correction)
# ---------------------------------------------------------------------------
@check("4.2", "v_mold band mean (norm-matched cohort)", "jlens-atlas/atlas_cohort_n100.json")
def _atlas_vmold(p):
    return 0.822, load(p)["targets"]["v_mold"]["band_mean"]


@check("4.2", "u_mold band mean (norm-matched)", "jlens-atlas/atlas_cohort_n100.json")
def _atlas_umold(p):
    return 0.358, load(p)["targets"]["u_mold"]["band_mean"]


@check("4.2", "Mold ratio v/u at matched norm", "jlens-atlas/atlas_cohort_n100.json", tol=0.02)
def _atlas_mold_ratio(p):
    t = load(p)["targets"]
    return 2.30, t["v_mold"]["band_mean"] / t["u_mold"]["band_mean"]


@check("4.2", "Gold ratio v/u at matched norm (INVERTED)", "jlens-atlas/atlas_cohort_n100.json", tol=0.02)
def _atlas_gold_ratio(p):
    t = load(p)["targets"]
    return 0.57, t["v_gold"]["band_mean"] / t["u_gold"]["band_mean"]


@check("4.2", "unmatched Mold ratio (6.53x, the confounded one)", "jlens-atlas/atlas_primary_estimand.json", tol=0.02)
def _unmatched_mold(p):
    return 6.53, load(p)["primary"]["mold"]["own_pole_ratio_v_over_u"]


@check("4.2", "unmatched Gold ratio (1.23x)", "jlens-atlas/atlas_primary_estimand.json", tol=0.02)
def _unmatched_gold(p):
    return 1.23, load(p)["primary"]["gold"]["own_pole_ratio_v_over_u"]


@check("4.2", "own-pole ratio panel minimum", "jlens-atlas/atlas_primary_estimand.json", tol=0.02)
def _panel_min(p):
    panel = load(p)["robustness_panel_all_conventions"]
    return -0.81, min(c["own_pole_ratio"] for c in panel.values())


@check("4.2", "own-pole ratio panel maximum", "jlens-atlas/atlas_primary_estimand.json", tol=0.02)
def _panel_max(p):
    panel = load(p)["robustness_panel_all_conventions"]
    return 9.30, max(c["own_pole_ratio"] for c in panel.values())


@check("4.2", "inverted cells in panel (all jlens_norm|gold)", "jlens-atlas/atlas_primary_estimand.json", tol=0)
def _inverted(p):
    return 3, len(load(p)["cells_inverted_own_pole"])


# ---------------------------------------------------------------------------
# 4.3 — J-space share: THE claim the paper rests on
# ---------------------------------------------------------------------------
def _jshare(target, field="var_fraction"):
    def fn(p):
        return None, load(p)["targets"][target][field]
    return fn


for _target, _reported_share, _reported_ge in [
    ("lang_fr", 0.1138, 0), ("v_mold", 0.0833, 0), ("v_gold", 0.0547, 0),
    ("u_mold", 0.0486, 34), ("u_gold", 0.0393, 53),
]:
    def _make(target=_target, share=_reported_share, n_ge=_reported_ge):
        @check("4.3", f"J-share {target}", "routing-core/jshare_cohort_n100.json", tol=0.0005)
        def _share(p, target=target, share=share):
            return share, load(p)["targets"][target]["var_fraction"]

        @check("4.3", f"randoms >= {target} (of 100)", "routing-core/jshare_cohort_n100.json", tol=0)
        def _ge(p, target=target, n_ge=n_ge):
            return n_ge, load(p)["targets"][target]["n_randoms_ge"]
    _make()


@check("4.3", "exact permutation p floor at n=100", "routing-core/jshare_cohort_n100.json", tol=1e-6)
def _perm_p(p):
    return 0.0099, load(p)["targets"]["v_mold"]["perm_p"]


@check("4.3", "Gold null mean (chance baseline)", "routing-core/jshare_cohort_n100.json", tol=0.0005)
def _null_gold(p):
    return 0.0391, load(p)["null"]["gold"]["mean"]


@check("4.3", "Mold null mean (chance baseline)", "routing-core/jshare_cohort_n100.json", tol=0.0005)
def _null_mold(p):
    return 0.0459, load(p)["null"]["mold"]["mean"]


@check("4.3", "J5 penult v_mold J-share", "jlens-fit-2507/j5_lens_comparison.json", tol=0.002)
def _penult_vmold(p):
    return 0.107, load(p)["jshare"]["penult"]["v_mold"]


@check("4.3", "J5 penult v_gold J-share", "jlens-fit-2507/j5_lens_comparison.json", tol=0.002)
def _penult_vgold(p):
    return 0.073, load(p)["jshare"]["penult"]["v_gold"]


@check("4.3", "J5 same-vector across-target Jaccard null mean", "jlens-fit-2507/j5_paired_baseline.json", tol=0.002)
def _j5_baseline(p):
    return 0.134, load(p)["baseline"]["mean"]


# ---------------------------------------------------------------------------
# 4.4 — trajectory (the section with the endpoint/count defects)
# ---------------------------------------------------------------------------
@check("4.4", "released RL checkpoints per pole", "jlens-atlas/traj_results.json", tol=0)
def _n_ckpt(p):
    rows = load(p)["results"]
    return 30, len({r["step"] for r in rows})


@check("4.4", "Mold J-share at step 0", "jlens-atlas/traj_results.json", tol=0.0015)
def _mold_step0(p):
    rows = [r for r in load(p)["results"] if r["concept"] == "mold"]
    return 0.061, min(rows, key=lambda r: r["step"])["var_fraction_k16"]


@check("4.4", "Mold J-share at step 150 (trajectory endpoint)", "jlens-atlas/traj_results.json", tol=0.0015)
def _mold_step150(p):
    rows = [r for r in load(p)["results"] if r["concept"] == "mold"]
    return 0.080, max(rows, key=lambda r: r["step"])["var_fraction_k16"]


@check("4.4", "Mold J-share at step 95 (the extraction point)", "jlens-atlas/traj_results.json", tol=0.0015)
def _mold_step95(p):
    rows = [r for r in load(p)["results"]
            if r["concept"] == "mold" and r["step"] == 95]
    return 0.083, rows[0]["var_fraction_k16"] if rows else float("nan")


@check("4.4", "Gold J-share step 0 -> 150 stays flat (delta)", "jlens-atlas/traj_results.json", tol=0.002)
def _gold_flat(p):
    rows = [r for r in load(p)["results"] if r["concept"] == "gold"]
    a = min(rows, key=lambda r: r["step"])["var_fraction_k16"]
    b = max(rows, key=lambda r: r["step"])["var_fraction_k16"]
    return -0.006, b - a


@check("4.4", "Mold norm step 0 -> 150", "jlens-atlas/traj_results.json", tol=0.1)
def _mold_norm(p):
    rows = [r for r in load(p)["results"] if r["concept"] == "mold"]
    return 21.9, max(rows, key=lambda r: r["step"])["norm"]


@check("4.4", "step-0 Mold top J-token is 'unsuccessful'", "jlens-atlas/traj_results.json", tol=None)
def _mold_seed(p):
    rows = [r for r in load(p)["results"] if r["concept"] == "mold"]
    first = min(rows, key=lambda r: r["step"])
    return "unsuccessful", first["tokens_k16"][0].strip()


# ---------------------------------------------------------------------------
# 4.4b — step-0 baseline: the untrained axis already clears the null.
# This is what turned the contribution from "training grants a voice" into
# "training amplifies". If any of these flip, the framing has to be revisited.
# ---------------------------------------------------------------------------
@check("4.4b", "step-0 Gold J-share", "jlens-atlas/step0_baseline.json", tol=0.0015)
def _s0_gold(p):
    return 0.0592, load(p)["poles"]["gold"]["directions"]["step_0"]["j_share"]


@check("4.4b", "step-0 Gold randoms >= (of 100)", "jlens-atlas/step0_baseline.json", tol=0)
def _s0_gold_ge(p):
    return 0, load(p)["poles"]["gold"]["directions"]["step_0"]["n_randoms_ge"]


@check("4.4b", "step-0 Mold J-share", "jlens-atlas/step0_baseline.json", tol=0.0015)
def _s0_mold(p):
    return 0.0611, load(p)["poles"]["mold"]["directions"]["step_0"]["j_share"]


@check("4.4b", "step-0 Mold randoms >= (of 100)", "jlens-atlas/step0_baseline.json", tol=0)
def _s0_mold_ge(p):
    return 0, load(p)["poles"]["mold"]["directions"]["step_0"]["n_randoms_ge"]


@check("4.4b", "untrained axis clears null at BOTH poles", "jlens-atlas/step0_baseline.json", tol=None)
def _s0_verdict(p):
    return True, load(p)["verdict"]["step0_clears_null_both_poles"]


@check("4.4b", "contributions framing", "jlens-atlas/step0_baseline.json", tol=None)
def _s0_framing(p):
    return "amplification", load(p)["verdict"]["contributions_framing"]


@check("4.4b", "training REDUCES Gold J-share (negative delta)", "jlens-atlas/step0_baseline.json", tol=0.002)
def _s0_gold_delta(p):
    return -0.0045, load(p)["verdict"]["training_delta_j_share"]["gold"]


@check("4.4b", "training RAISES Mold J-share (positive delta)", "jlens-atlas/step0_baseline.json", tol=0.002)
def _s0_mold_delta(p):
    return 0.0222, load(p)["verdict"]["training_delta_j_share"]["mold"]


@check("4.4b", "step-0/cohort comparability gate", "jlens-atlas/step0_baseline.json", tol=None)
def _s0_gate(p):
    return "PASS", load(p)["comparability_gate"]["status"]


# ---------------------------------------------------------------------------
# 4.5 — channel-specific causal routing (J4)
# ---------------------------------------------------------------------------
@check("4.5", "second-judge Krippendorff alpha", "j4-behavioral/R3_second_judge.json", tol=0.005)
def _alpha(p):
    return 0.82, load(p)["agreement"]["krippendorff_alpha_interval"]


@check("4.5", "second-judge Spearman", "j4-behavioral/R3_second_judge.json", tol=0.005)
def _spearman(p):
    return 0.87, load(p)["agreement"]["spearman"]


@check("4.5", "random-J-component cohort mean, gold layer", "j4-behavioral/j4_random_jcomp_summary.json", tol=0.01)
def _rand_gold(p):
    return 0.04, load(p)["cohort"]["gold"]["mean"]


@check("4.5", "random-J-component cohort mean, mold layer", "j4-behavioral/j4_random_jcomp_summary.json", tol=0.01)
def _rand_mold(p):
    return 0.02, load(p)["cohort"]["mold"]["mean"]


@check("4.5", "real J-component sentiment, gold", "j4-behavioral/j4_random_jcomp_summary.json", tol=0.005)
def _real_gold(p):
    return 1.5625, load(p)["real_jcomp_means"]["gold"]


# ---------------------------------------------------------------------------
# 4.7 — cross-model transfer
# ---------------------------------------------------------------------------
@check("4.7", "Qwen3-4B v_mold J-share", "j6-crossmodel/j6_summary.json", tol=0.002)
def _j6_vmold(p):
    return 0.079, load(p)["jshare"]["v_mold"]["var_fraction"]


@check("4.7", "Qwen3-4B v_gold J-share", "j6-crossmodel/j6_summary.json", tol=0.002)
def _j6_vgold(p):
    return 0.048, load(p)["jshare"]["v_gold"]["var_fraction"]


@check("4.7", "Qwen3-4B Gold ratio (inversion replicates)", "j6-crossmodel/j6_summary.json", tol=0.02)
def _j6_gold_ratio(p):
    a = load(p)["atlas"]
    return 0.52, a["v_gold"]["band_mean"] / a["u_gold"]["band_mean"]


@check("4.7", "Qwen3-4B Mold ratio", "j6-crossmodel/j6_summary.json", tol=0.03)
def _j6_mold_ratio(p):
    a = load(p)["atlas"]
    return 4.19, a["v_mold"]["band_mean"] / a["u_mold"]["band_mean"]


@check("4.7", "v_mold across-model Jaccard (a NULL, p=0.118)", "j6-crossmodel/j6_summary.json", tol=0.002)
def _j6_jaccard(p):
    return 0.185, load(p)["token_overlap"]["v_mold"]["jaccard"]


@check("4.7", "same-vector across-model Jaccard null mean", "j6-crossmodel/j6_paired_baseline.json", tol=0.002)
def _j6_baseline(p):
    return 0.088, load(p)["baseline"]["mean"]


# ---------------------------------------------------------------------------
# Integrity gates (not report numbers — run provenance)
# ---------------------------------------------------------------------------
def integrity_gates():
    """Checks that do not compare to a reported number."""
    gates = []

    def gate(name, ok, detail):
        gates.append({"name": name, "ok": bool(ok), "detail": detail})

    # Every decomposition-bearing file should agree on the algorithm version.
    versions = {}
    for rel in ["routing-core/jshare_cohort_n100.json",
                "routing-core/mech_decompositions.json",
                "jlens-atlas/ksweep_ext.json",
                "j6-crossmodel/j6_decomps.json"]:
        path = results(*rel.split("/"))
        if not os.path.exists(path):
            continue
        blob = json.dumps(load(path))
        found = set()
        for token in ('"algorithm_version": "', '"routing_algorithm_version": "'):
            i = 0
            while (i := blob.find(token, i)) != -1:
                i += len(token)
                found.add(blob[i:blob.index('"', i)])
        versions[rel] = sorted(found) or ["v1_unversioned"]
    distinct = {v for vs in versions.values() for v in vs}
    gate("single routing-algorithm version across result files",
         len(distinct) <= 1,
         f"found {sorted(distinct)} across {len(versions)} files: {versions}")

    # F2's cross-file verification gate must have actually run.
    f2 = results("routing-core", "jshare_cohort_n100.json")
    if os.path.exists(f2):
        blob = load(f2)
        g = blob.get("verification_gate")
        if g is None:
            gate("F2 cross-file verification gate recorded", False,
                 "no verification_gate field — file predates the gate, or the "
                 "gate did not run")
        else:
            gate("F2 cross-file verification gate recorded", g["status"] == "PASS",
                 f"status={g['status']}, skipped={len(g.get('skipped_version_mismatch', []))}")

    # Token mask provenance, if recorded.
    for rel in ["routing-core/jshare_cohort_n100.json", "routing-core/mech_meta.json"]:
        path = results(*rel.split("/"))
        if os.path.exists(path):
            tm = load(path).get("token_mask")
            if tm is not None:
                gate(f"token mask matches frozen cutoff ({rel})",
                     tm.get("matches_historical", False),
                     f"min_id_invalid={tm.get('min_id_invalid')} "
                     f"(frozen {tm.get('historical_min_id_invalid')})")
    return gates


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    ap.add_argument("--update-hint", action="store_true",
                    help="for each failure, print the value the prose should use")
    args = ap.parse_args()

    rows = []
    for c in CHECKS:
        path = results(*c["path"].split("/"))
        row = {"section": c["section"], "claim": c["description"],
               "file": c["path"]}
        if not os.path.exists(path):
            row.update(status="MISSING", reported=None, derived=None)
            rows.append(row)
            continue
        try:
            reported, derived = c["fn"](path)
        except Exception as exc:                     # noqa: BLE001
            row.update(status="ERROR", reported=None, derived=f"{type(exc).__name__}: {exc}")
            rows.append(row)
            continue
        if c["tol"] is None:
            ok = str(reported) == str(derived)
        elif reported is None:
            ok = True
        else:
            ok = abs(float(derived) - float(reported)) <= float(c["tol"])
        row.update(status="PASS" if ok else "FAIL",
                   reported=reported, derived=derived, tol=c["tol"])
        rows.append(row)

    gates = integrity_gates()

    if args.json:
        print(json.dumps({"claims": rows, "integrity_gates": gates}, indent=2,
                         default=str))
    else:
        width = max(len(r["claim"]) for r in rows) + 2
        current = None
        for r in rows:
            if r["section"] != current:
                current = r["section"]
                print(f"\n§{current}")
            mark = {"PASS": "  ok  ", "FAIL": " FAIL ",
                    "MISSING": " MISS ", "ERROR": " ERR  "}[r["status"]]
            line = f"[{mark}] {r['claim']:<{width}}"
            if r["status"] == "PASS":
                line += f"{r['derived']}"
            elif r["status"] == "FAIL":
                line += f"report says {r['reported']}, file says {r['derived']}"
            else:
                line += f"{r['file']}: {r['derived'] or 'not found'}"
            print(line)

        print("\nIntegrity gates")
        for g in gates:
            print(f"[{'  ok  ' if g['ok'] else ' FAIL '}] {g['name']}\n"
                  f"          {g['detail']}")

        n_fail = sum(1 for r in rows if r["status"] != "PASS")
        n_gate_fail = sum(1 for g in gates if not g["ok"])
        print(f"\n{len(rows) - n_fail}/{len(rows)} claims reproduce; "
              f"{len(gates) - n_gate_fail}/{len(gates)} integrity gates pass.")

        if args.update_hint and n_fail:
            print("\nProse updates needed (report-draft.md):")
            for r in rows:
                if r["status"] == "FAIL":
                    print(f"  §{r['section']}  {r['claim']}: "
                          f"{r['reported']} -> {r['derived']}")

    failed = any(r["status"] != "PASS" for r in rows) or any(not g["ok"] for g in gates)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
