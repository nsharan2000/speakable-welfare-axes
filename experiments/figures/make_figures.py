"""Generate paper figures from synced results. Run locally after sync_results.sh.

Figures land in experiments/figures/out/:
  fig_dose_response.png  — judge-scored sentiment curves (E5a validation)
  fig_primary.png        — pre-registered v-vs-u routing contrast + random null
  fig_band.png           — workspace-band layer statistics (per model available)
Skips any figure whose inputs are missing (safe to run anytime).
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(HERE, "out")
os.makedirs(OUT, exist_ok=True)

C = {"v_gold": "#c9a227", "u_gold": "#8a7a3b", "rand_gold": "#bbbbbb",
     "v_mold": "#7b1f2b", "u_mold": "#b2707a", "rand_mold": "#dddddd"}


def dose_response():
    p = os.path.join(ROOT, "experiments/welfare-axis/results/judge_sentiment.json")
    if not os.path.exists(p):
        return
    d = json.load(open(p))
    base = d["baseline"]["mean"]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for name in ["v_gold", "u_gold", "rand_gold", "v_mold", "u_mold", "rand_mold"]:
        xs, ys, es = [], [], []
        for a in ["-4", "-2", "+2", "+4"]:
            k = f"{name}@{a}"
            if k in d["curves"]:
                xs.append(float(a))
                ys.append(d["curves"][k]["mean"])
                es.append(d["curves"][k]["sem"])
        ls = "--" if name.startswith("u_") else (":" if name.startswith("rand") else "-")
        ax.errorbar(xs, ys, yerr=es, label=name, color=C[name], ls=ls,
                    marker="o", ms=4, lw=1.8, capsize=2)
    ax.axhline(base, color="k", lw=0.8, alpha=0.5)
    ax.text(-4.2, base, "baseline", fontsize=8, va="bottom")
    ax.set_xlabel("steering strength α (u, rand norm-matched to v)")
    ax.set_ylabel("judge sentiment (−5…+5)")
    ax.set_title("Steering the un-finetuned model: sentiment dose–response\n"
                 "(1000 generations, 40 prompts, blinded LLM judges)")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_dose_response.png"), dpi=200)
    print("fig_dose_response.png")


def primary():
    p = os.path.join(ROOT, "experiments/routing-core/results/primary_analysis.json")
    if not os.path.exists(p):
        return
    a = json.load(open(p))
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2))
    ax = axes[0]
    for i, c in enumerate(["gold", "mold"]):
        pc = a["primary_per_polarity"][c]
        ax.bar(i - 0.17, pc["mean_E_v"], width=0.32, color=C[f"v_{c}"], label=f"v_{c}" if i == 0 else None)
        ax.bar(i + 0.17, pc["mean_E_u"], width=0.32, color=C[f"u_{c}"], alpha=0.8)
        n1 = a.get(f"C1_random_null_{c}", {})
        if n1:
            ax.errorbar([i], [n1["null_mean"]], yerr=[2 * n1["null_sd"]],
                        fmt="k_", ms=20, capsize=6, lw=1)
    ax.set_xticks([0, 1], ["Gold (+)", "Mold (−)"])
    ax.set_ylabel("routing effect E (congruent Δ log-mass at answer token)")
    ax.set_title("v (solid) vs norm-matched u (light);\nblack = random-direction null ±2SD")
    ax.axhline(0, color="k", lw=0.7)

    ax = axes[1]
    b = a["primary"]["bayes"]
    txt = (f"decision: {a['decision']}\n"
           f"paired d_z = {a['primary']['cohens_dz']:.3f}\n"
           f"perm p = {a['primary']['perm_p_two_sided']:.4f}\n"
           f"delta 95% CI = [{b['delta_ci95'][0]:.2f}, {b['delta_ci95'][1]:.2f}]\n"
           f"P(in ROPE) = {b['p_in_rope']:.3f}   BF01 = {b['bf01_savage_dickey']:.2f}\n"
           f"lang positive control pass = {a['C2_language_positive_control']['pass']}")
    ax.axis("off")
    ax.text(0.02, 0.6, txt, fontsize=11, family="monospace", va="center")
    ax.set_title("Pre-registered decision quantities")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_primary.png"), dpi=200)
    print("fig_primary.png")


def band():
    made = False
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
    for tag, color in [("Qwen3-4B", "#3b6ea5"), ("Qwen3-4B-Instruct-2507", "#a5533b")]:
        p = os.path.join(ROOT, f"experiments/routing-core/results/band_stats_{tag}.json")
        if not os.path.exists(p):
            continue
        made = True
        d = json.load(open(p))
        ls_ = d["layers"]
        for ax, key, ylab in [
                (axes[0], "acc_top10", "lens top-10 next-token acc"),
                (axes[1], "excess_kurtosis", "excess kurtosis"),
                (axes[2], "top1_autocorr", "top-1 persistence")]:
            ax.plot(ls_, [d[key][str(l)] for l in ls_], color=color, lw=1.6, label=tag)
            ax.set_xlabel("layer")
            ax.set_ylabel(ylab)
        if "top1_autocorr_null" in d:
            axes[2].plot(ls_, [d["top1_autocorr_null"][str(l)] for l in ls_],
                         color=color, lw=1, ls=":", alpha=0.6)
    if made:
        axes[0].legend(fontsize=7)
        fig.suptitle("Workspace-band statistics (dotted = shuffled null)")
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, "fig_band.png"), dpi=200)
        print("fig_band.png")


def mech():
    p = os.path.join(ROOT, "experiments/routing-core/results/mech_analysis.json")
    if not os.path.exists(p):
        return
    a = json.load(open(p))
    jr = a.get("jload_ratio", {})
    names = [n for n in ["v_gold", "u_gold", "v_mold", "u_mold", "lang_fr"] if n in jr]
    if not names:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    ax = axes[0]
    comps = ["E_full", "E_jcomp", "E_perp", "E_perp_clamped"]
    labels = ["full", "J-comp", "perp", "perp+clamp"]
    w = 0.2
    for ci, ck in enumerate(comps):
        xs = np.arange(len(names)) + (ci - 1.5) * w
        ys = [jr[n].get(ck) if jr[n].get(ck) is not None else 0 for n in names]
        ax.bar(xs, ys, width=w, label=labels[ci])
    rn = a.get("random_component_null", {}).get("full", {})
    if rn:
        ax.axhspan(rn["mean"] - 2 * rn["sd"], rn["mean"] + 2 * rn["sd"],
                   color="gray", alpha=0.25, label="random null ±2SD (full)")
    ax.set_xticks(range(len(names)), names, rotation=15)
    ax.set_ylabel("congruent routing effect E @ α=+4")
    ax.set_title("Does the effect load on the J-space component?")
    ax.axhline(0, color="k", lw=0.7)
    ax.legend(fontsize=8)

    ax = axes[1]
    lr = a.get("lens_readout", {})
    for n in names:
        if n in lr:
            sb = lr[n]["shift_by_layer"]
            xs = sorted(int(k) for k in sb)
            ax.plot(xs, [sb[str(x)] for x in xs], marker="o", ms=3, label=n,
                    color=C.get(n, None))
    ax.set_xlabel("lens layer")
    ax.set_ylabel("congruent pole-score shift (full − clean)")
    ax.set_title("J-lens readout arm")
    ax.axhline(0, color="k", lw=0.7)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_mech.png"), dpi=200)
    print("fig_mech.png")


def audit_corrections():
    """Post-audit panel: (a) F3 — the atlas gap is mostly magnitude;
    (b) F1 — random J-components do not reproduce the behavioural effect;
    (c) F2 — the scale-invariant J-share, the claim that survives both."""
    f3p = os.path.join(ROOT, "experiments/jlens-atlas/results/atlas_cohort_n100.json")
    f1p = os.path.join(ROOT, "experiments/j4-behavioral/results/j4_random_jcomp_summary.json")
    f2p = os.path.join(ROOT, "experiments/routing-core/results/jshare_cohort_n100.json")
    f4p = os.path.join(ROOT, "experiments/jlens-atlas/results/atlas_primary_estimand.json")
    if not (os.path.exists(f3p) and os.path.exists(f1p)):
        return
    f3 = json.load(open(f3p))
    f1 = json.load(open(f1p))
    f4 = json.load(open(f4p)) if os.path.exists(f4p) else None
    n = 3 if os.path.exists(f2p) else 2
    fig, axes = plt.subplots(1, n, figsize=(5.0 * n, 4.0))

    # (a) unmatched vs norm-matched own-pole scores
    ax = axes[0]
    poles = ["gold", "mold"]
    unm_v = [f4["primary"][c]["v"]["band_mean"] for c in poles] if f4 else [None, None]
    unm_u = [f4["primary"][c]["u"]["band_mean"] for c in poles] if f4 else [None, None]
    mat_v = [f3["targets"][f"v_{c}"]["band_mean"] for c in poles]
    mat_u = [f3["targets"][f"u_{c}"]["band_mean"] for c in poles]
    x = np.arange(2)
    w = 0.2
    ax.bar(x - 1.5 * w, unm_v, w, color=[C["v_gold"], C["v_mold"]], label="trained v")
    ax.bar(x - 0.5 * w, unm_u, w, color=[C["u_gold"], C["u_mold"]], hatch="//",
           label="naive u (native norm)")
    ax.bar(x + 0.5 * w, mat_v, w, color=[C["v_gold"], C["v_mold"]])
    ax.bar(x + 1.5 * w, mat_u, w, color=[C["u_gold"], C["u_mold"]], hatch="..",
           label="naive u (norm-matched)")
    for i, c in enumerate(poles):
        r_un = (unm_v[i] / unm_u[i]) if f4 else float("nan")
        r_ma = mat_v[i] / mat_u[i]
        ax.text(i - w, max(unm_v[i], unm_u[i]) + 0.03, f"{r_un:.2f}×",
                ha="center", fontsize=9)
        ax.text(i + w, max(mat_v[i], mat_u[i]) + 0.03, f"{r_ma:.2f}×",
                ha="center", fontsize=9,
                color=("#b00020" if r_ma < 1 else "k"),
                fontweight=("bold" if r_ma < 1 else "normal"))
    ax.set_xticks(x)
    ax.set_xticklabels(["Gold (flourishing)", "Mold (distress)"])
    ax.set_ylabel("own-pole score (J-lens, band L16–31)")
    ax.set_title("(a) The trained/naive readout gap is mostly magnitude\n"
                 "left pair: native norms · right pair: norm-matched")
    ax.legend(fontsize=7.5)
    ax.axhline(0, color="k", lw=0.7)

    # (b) F1 random-jcomp null vs real jcomp effects
    ax = axes[1]
    clean = f1["clean_mean_reference"]
    labels = ["random J-comp\n(gold layer)", "random J-comp\n(mold layer)",
              "v_Gold J-comp", "v_Mold J-comp"]
    vals = [f1["cohort"]["gold"]["mean"], f1["cohort"]["mold"]["mean"],
            f1["real_jcomp_means"]["gold"], f1["real_jcomp_means"]["mold"]]
    cols = ["#bbbbbb", "#dddddd", C["v_gold"], C["v_mold"]]
    ax.bar(range(4), vals, color=cols, edgecolor="k", lw=0.5)
    ax.axhline(clean, color="k", ls="--", lw=1, label=f"clean baseline ({clean:+.2f})")
    ax.axhspan(clean - 0.4, clean + 0.4, color="#2e7d32", alpha=0.12,
               label="pre-specified 'controlled' band (±0.40)")
    ax.set_xticks(range(4))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("blind-judged sentiment (−5…+5)")
    ax.set_title("(b) Rescaling is not the driver:\nrandom J-components stay at baseline")
    ax.legend(fontsize=7.5, loc="lower left")

    # (c) F2 J-share against the 100-direction null (if present)
    if n == 3:
        f2 = json.load(open(f2p))
        ax = axes[2]
        order = ["lang_fr", "v_mold", "v_gold", "u_mold", "u_gold"]
        order = [t for t in order if t in f2["targets"]]
        vmax = max(f2["targets"][t]["var_fraction"] for t in order)
        for i, t in enumerate(order):
            d = f2["targets"][t]
            pol = d.get("null_polarity", "gold")
            nl = f2["null"][pol]
            ax.axhspan(nl["p05"], nl["p95"], xmin=i / len(order),
                       xmax=(i + 1) / len(order), color="#cccccc", alpha=0.5)
            ax.plot([i], [d["var_fraction"]], "o", ms=9,
                    color=C.get(t, "#333333"))
            sig = d["perm_p"] < 0.01
            ax.annotate(f"{d['n_randoms_ge']}/100 ≥\np={d['perm_p']:.3f}",
                        (i, d["var_fraction"]), textcoords="offset points",
                        xytext=(0, 11), ha="center", fontsize=7.5,
                        fontweight=("bold" if sig else "normal"),
                        color=("#b00020" if sig else "#444444"))
        ax.set_ylim(0.025, vmax * 1.28)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(order, fontsize=8, rotation=20)
        ax.set_ylabel(r"J-share $\|x_J\|^2/\|x\|^2$ (k=16)")
        ax.set_title("(c) The claim that survives: scale-invariant J-share\n"
                     "grey = 5–95% of 100 random directions", pad=14)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_audit_corrections.png"), dpi=200)
    print("fig_audit_corrections.png")


def transfer():
    """J6/J5 invariance panel: (a) J-share preserved across models and lens
    targets; (b) the distress lexicon's token overlap vs chance."""
    j6p = os.path.join(ROOT, "experiments/j6-crossmodel/results/j6_summary.json")
    f2p = os.path.join(ROOT, "experiments/routing-core/results/jshare_cohort_n100.json")
    j5p = os.path.join(ROOT, "experiments/jlens-fit-2507/results/j5_lens_comparison.json")
    if not (os.path.exists(j6p) and os.path.exists(f2p) and os.path.exists(j5p)):
        return
    j6 = json.load(open(j6p))
    f2 = json.load(open(f2p))
    j5 = json.load(open(j5p))
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))

    # (a) J-share of each axis under three instruments
    ax = axes[0]
    names = ["v_gold", "u_gold", "v_mold", "u_mold"]
    inst = [("2507 lens\n(final target)", lambda n: f2["targets"][n]["var_fraction"]),
            ("2507 lens\n(penult target)", lambda n: j5["jshare"]["penult"][n]),
            ("Qwen3-4B\n(public lens)", lambda n: j6["jshare"][n]["var_fraction"])]
    x = np.arange(len(inst))
    for i, n in enumerate(names):
        vals = [g(n) for _, g in inst]
        ax.plot(x, vals, "o-", color=C[n], label=n, lw=1.5, ms=6)
    ax.set_xticks(x)
    ax.set_xticklabels([t for t, _ in inst], fontsize=8)
    ax.set_ylabel(r"J-share $\|x_J\|^2/\|x\|^2$ (k=16)")
    ax.set_title("(a) The speakability ordering survives the instrument:\n"
                 "two lens targets and a second model")
    ax.legend(fontsize=8)

    # (b) token overlap ACROSS MODELS against the CORRECT null (same random
    # vector under both models' lenses, j6_paired_baseline.json). Overlap
    # magnitude is a null result; only shared-token CONTENT differs.
    j6b = json.load(open(os.path.join(
        ROOT, "experiments/j6-crossmodel/results/j6_paired_baseline.json")))
    ax = axes[1]
    bars = [("v_mold", j6["token_overlap"]["v_mold"]["jaccard"], C["v_mold"]),
            ("u_mold", j6["token_overlap"]["u_mold"]["jaccard"], C["u_mold"]),
            ("u_gold", j6["token_overlap"]["u_gold"]["jaccard"], C["u_gold"]),
            ("v_gold", j6["token_overlap"]["v_gold"]["jaccard"], C["v_gold"])]
    nb = j6b["baseline"]
    ax.bar(range(len(bars)), [b[1] for b in bars], color=[b[2] for b in bars],
           edgecolor="k", lw=0.5)
    ax.axhspan(max(0, nb["mean"] - nb["sd"]), nb["mean"] + nb["sd"],
               color="#999999", alpha=0.35,
               label=f"same-vector null (mean {nb['mean']:.3f} ± {nb['sd']:.3f}, "
                     f"max {nb['max']:.3f})")
    ax.axhline(nb["max"], color="#666666", ls=":", lw=1)
    shared = {"v_mold": "shares 'failed', 'negative'\n(valence words)",
              "u_mold": "shares 'identical', ...\n(junk)",
              "u_gold": "shares 'nn' + 3 junk", "v_gold": "1 junk token"}
    for i, b in enumerate(bars):
        ax.text(i, b[1] + 0.005, f"{b[1]:.3f}", ha="center", fontsize=8.5)
        ax.text(i, -0.045, shared[b[0]], ha="center", fontsize=7,
                style="italic", color="#555555")
    ax.set_ylim(-0.06, 0.27)
    ax.set_xticks(range(len(bars)))
    ax.set_xticklabels([b[0] for b in bars], fontsize=9)
    ax.set_ylabel("J-component token-set Jaccard (k=16)")
    ax.set_title("(b) Cross-model token overlap is WITHIN the\n"
                 "same-vector null (p=0.118); only content differs",
                 fontsize=11)
    ax.legend(fontsize=7.5, loc="upper right")

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_transfer.png"), dpi=200)
    print("fig_transfer.png")


if __name__ == "__main__":
    dose_response()
    primary()
    band()
    mech()
    audit_corrections()
    transfer()
    print("FIGURES_DONE")
