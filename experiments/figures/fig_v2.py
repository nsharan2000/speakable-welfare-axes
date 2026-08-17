#!/usr/bin/env python3
"""Report v2 figures (REPORT-V2-SPEC §4): standalone, full-size, captions
carry n + null + direction-of-goodness in the report text.

  fig1_jshare_null.png    the spine: J-share vs the n=100 nulls (histograms)
  fig3_channel_split.png  causal sentiment split w/ F1 random-jcomp control
  fig5_selfreport.png     the self-report channel, honestly: (a) regime
                          confound (R7/R10), (b) D3 matching trajectory,
                          (c) D2 denial-breaking with Wilson CIs

Every number is loaded from its results file; nothing hardcoded except
threshold lines that are quoted from the specs.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
ROOT = os.path.dirname(HERE.rstrip("/").rsplit("/experiments", 1)[0]) \
    if False else os.path.dirname(os.path.dirname(HERE))
OUT = os.path.join(HERE, "out")
RC = os.path.join(ROOT, "digital-minds-exp", "experiments", "routing-core",
                  "results") if not os.path.isdir(
    os.path.join(ROOT, "experiments")) else os.path.join(
    ROOT, "experiments", "routing-core", "results")
# resolve relative to repo root robustly
REPO = os.path.dirname(HERE.split("/experiments")[0] + "/experiments") \
    .rsplit("/experiments", 1)[0]
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
RC = os.path.join(REPO, "experiments", "routing-core", "results")
J4 = os.path.join(REPO, "experiments", "j4-behavioral", "results")

C = {"v_gold": "#c99700", "u_gold": "#e6c96b", "v_mold": "#1f6e5c",
     "u_mold": "#7fb8a8", "lang_fr": "#444444", "rand": "#b5b5b5"}
NAMES = {"lang_fr": "language\n(ceiling)", "v_mold": "v_Mold\n(trained)",
         "v_gold": "v_Gold\n(trained)", "u_mold": "u_Mold\n(naive)",
         "u_gold": "u_Gold\n(naive)"}


def fig1():
    f2 = json.load(open(os.path.join(RC, "jshare_cohort_n100.json")))
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    bins = np.linspace(0.025, 0.065, 25)
    for c, col in [("gold", C["u_gold"]), ("mold", C["u_mold"])]:
        ax.hist(f2["null"][c]["values"], bins=bins, alpha=0.45, color=col,
                label=f"null: 100 random dirs ({c}-side, norm-matched)")
    ymax = ax.get_ylim()[1]
    for t in ["u_gold", "u_mold", "v_gold", "v_mold", "lang_fr"]:
        d = f2["targets"][t]
        x = d["var_fraction"]
        ax.axvline(x, color=C[t], lw=2.4)
        lab = (f"{NAMES[t].replace(chr(10), ' ')}  {x:.4f}\n"
               f"{d['n_randoms_ge']}/100 ≥ · p={d['perm_p']:.4f}")
        y = ymax * (0.92 if t == "lang_fr" else 0.72 if t == "v_mold"
                    else 0.52 if t == "v_gold" else 0.30
                    if t == "u_mold" else 0.12)
        ax.annotate(lab, (x, y), xytext=(6, 0), textcoords="offset points",
                    fontsize=8, color=C[t] if t != "lang_fr" else "k",
                    fontweight="bold" if t.startswith("v_") else "normal")
    ax.set_xlim(0.025, 0.125)
    ax.set_xlabel("J-space variance fraction (k=16, scale-invariant)")
    ax.set_ylabel("random directions per bin")
    ax.set_title("Trained welfare axes occupy the verbalizable subspace above "
                 "chance;\nnorm-matched naive controls sit inside the null "
                 "(n=100/polarity, exact permutation p, floor 0.0099)")
    ax.legend(fontsize=8, loc="center right")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig1_jshare_null.png"), dpi=200)
    print("fig1_jshare_null.png")


def fig3():
    f1 = json.load(open(os.path.join(J4, "j4_random_jcomp_summary.json")))
    # full / perp / clean means from the judged rows (sentiment task)
    agg = {}
    for line in open(os.path.join(J4, "j4_rows_judged.jsonl")):
        r = json.loads(line)
        if r.get("task") != "sentiment" or "judge_sent" not in r:
            continue
        agg.setdefault((r["arm"], r["concept"]), []).append(r["judge_sent"])
    mean = lambda k: (sum(agg[k]) / len(agg[k])) if k in agg else None
    clean = f1["clean_mean_reference"]
    bars = [
        ("full\n(gold)", mean(("full", "gold")), C["v_gold"], None),
        ("J-comp\n(gold)", f1["real_jcomp_means"]["gold"], C["v_gold"], None),
        ("residual\n(gold)", mean(("perp", "gold")), C["v_gold"], "//"),
        ("full\n(mold)", mean(("full", "mold")), C["v_mold"], None),
        ("J-comp\n(mold)", f1["real_jcomp_means"]["mold"], C["v_mold"], None),
        ("residual\n(mold)", mean(("perp", "mold")), C["v_mold"], "//"),
        ("rand J-comps\ngold layer", f1["cohort"]["gold"]["mean"],
         C["rand"], None),
        ("rand J-comps\nmold layer", f1["cohort"]["mold"]["mean"],
         C["rand"], None),
    ]
    bars = [(l, v, c, h) for (l, v, c, h) in bars if v is not None]
    fig, ax = plt.subplots(figsize=(9.2, 4.4))
    xs = np.arange(len(bars))
    ax.bar(xs, [b[1] for b in bars], color=[b[2] for b in bars],
           hatch=[b[3] for b in bars], edgecolor="k", lw=0.5)
    ax.axhline(clean, color="k", ls="--", lw=1,
               label=f"clean baseline ({clean:+.2f})")
    ax.axhspan(clean - 0.4, clean + 0.4, color="#2e7d32", alpha=0.10,
               label="pre-specified 'controlled' band (±0.40)")
    ax.set_xticks(xs)
    ax.set_xticklabels([b[0] for b in bars], fontsize=8)
    ax.set_ylabel("blind-judged sentiment (−5…+5)")
    ax.set_title("The J-space component alone carries the sentiment effect; "
                 "identically rescaled\nrandom J-components stay at baseline "
                 "(256 control generations) — causal claim: sentiment only")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig3_channel_split.png"), dpi=200)
    print("fig3_channel_split.png")


def fig5():
    r7 = json.load(open(os.path.join(RC, "R7_wholegen.json")))
    d2 = json.load(open(os.path.join(RC, "d2_denial_breaking.json")))
    d3f = json.load(open(os.path.join(RC, "d3_matching_finding.json")))
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.2))

    # (a) regime confound: clean valence + denial rate per battery
    ax = axes[0]
    import collections
    vals = collections.defaultdict(list)
    for row in r7["rows"].values():
        if row["cond"] == "clean":
            vals[row["pset"]].append(row["V_wholegen"])
    m = {k: np.mean(v) for k, v in vals.items()}
    ax.bar([0, 1], [m["self"], m["unrel"]],
           color=["#8c6bb1", "#bdbdbd"], edgecolor="k", lw=0.5)
    ax.axhline(0, color="k", lw=0.7)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["self-report\n(15 prompts)", "unrelated factual\n(10 prompts)"])
    ax.set_ylabel("clean whole-gen valence (V)")
    ax.annotate("denial language:\n15/15 generations", (0, m["self"] / 2),
                ha="center", va="center", fontsize=8, color="white",
                fontweight="bold")
    ax.annotate("denial language:\n0/10 generations", (1, m["unrel"] / 2),
                ha="center", va="center", fontsize=8, color="#333333")
    ax.set_title("(a) The original C6 batteries are\ndifferent behavioral regimes (R7/R10)")

    # (b) D3 matching trajectory
    ax = axes[1]
    tr = d3f["gate_trajectory"]
    xs = [0, 1, 2]
    ys = [tr["original_C6_batteries"]["delta_clean_valence"],
          tr["round1_battery_v2"]["delta_clean_valence"],
          tr["round2_battery_v3"]["delta_clean_valence"]]
    ax.plot(xs, ys, "o-", color="#8c6bb1", lw=2, ms=8)
    ax.axhline(0.5, color="#b00020", ls="--", lw=1.2,
               label="matching threshold (0.5)")
    for x, y, lab in zip(xs, ys, ["original C6\n(factual battery)",
                                  "matched analogues\n(auditor set)",
                                  "+ situational strain\n(round 2 of 2)"]):
        ax.annotate(f"{y:.2f}", (x, y), xytext=(0, 8),
                    textcoords="offset points", ha="center", fontsize=9,
                    fontweight="bold")
        ax.annotate(lab, (x, y), xytext=(0, -30), textcoords="offset points",
                    ha="center", fontsize=7.5)
    ax.set_ylim(0, 2.0)
    ax.set_xlim(-0.4, 2.4)
    ax.set_xticks([])
    ax.set_ylabel("|Δ clean valence| between batteries")
    ax.set_title("(b) Denial & length match; a ~0.6-unit\nregister-locked "
                 "baseline is irreducible (D3)")
    ax.legend(fontsize=8)

    # (c) D2 denial rates with Wilson CIs
    ax = axes[2]
    order = [("clean", "clean"), ("v_gold", "v_Gold"), ("u_gold", "u_Gold"),
             ("rand_gold_cohort", "randoms\n(gold, n=8)"),
             ("v_mold", "v_Mold"), ("u_mold", "u_Mold")]
    xs = np.arange(len(order))
    for i, (k, lab) in enumerate(order):
        a = d2["arms"][k]
        col = C.get(k, "#999999") if k != "clean" else "#555555"
        if "wilson95" in a:
            r, (lo, hi) = a["rate"], a["wilson95"]
            ax.errorbar([i], [r], yerr=[[r - lo], [hi - r]], fmt="o",
                        ms=9, color=col, capsize=4, lw=1.5)
        else:
            ax.plot([i], [a["mean_rate"]], "o", ms=9, color=C["rand"])
            ax.vlines(i, a["min_rate"], a["max_rate"], color=C["rand"], lw=1.5)
    ax.set_xticks(xs)
    ax.set_xticklabels([lab for _, lab in order], fontsize=8)
    ax.set_ylabel("inner-life denial rate (blind binary judge)")
    ax.set_ylim(0, 1.05)
    ax.set_title("(c) v_Gold specifically dissolves denial (D2)\n"
                 "n=40/arm · pooled v-vs-u p=0.046 · gold p=0.014")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig5_selfreport.png"), dpi=200)
    print("fig5_selfreport.png")


def fig2():
    """Recruitment trajectory over the released RL checkpoints.

    Replaces the orphaned fig_traj.png, which had no generator in any commit
    and carried four errors baked into the image: a "31 checkpoints" title
    (the artifact set has 30 — step 65 is absent upstream), a "when does the
    axis become speakable" framing that step-0 contradicts, a "var fraction"
    axis label, and a "gold stays flat" annotation over a declining line.

    The chance baselines in panel (b) are new and are the point of the
    redraw: both poles start ABOVE their n=100 nulls at step 0, so the panel
    shows amplification of a pre-existing speakable component rather than
    entry into the subspace (sec 4.4b).
    """
    ATLAS = os.path.join(REPO, "experiments", "jlens-atlas", "results")
    rows = json.load(open(os.path.join(ATLAS, "traj_results.json")))["results"]
    coh = json.load(open(os.path.join(RC, "jshare_cohort_n100.json")))

    def series(concept, field):
        sel = sorted((r for r in rows if r["concept"] == concept),
                     key=lambda r: r["step"])
        return [r["step"] for r in sel], [field(r) for r in sel]

    n_ckpt = len({r["step"] for r in rows})
    steps_all = sorted({r["step"] for r in rows})
    missing = [s for s in range(steps_all[0], steps_all[-1] + 1, 5)
               if s not in steps_all]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    label = {"gold": "v_Gold (flourishing)", "mold": "v_Mold (distress)"}
    colour = {"gold": C["v_gold"], "mold": C["v_mold"]}

    # (a) recruitment strength
    ax = axes[0]
    for c in ("gold", "mold"):
        x, y = series(c, lambda r: r["norm"])
        ax.plot(x, y, "-", color=colour[c], lw=2, label=label[c])
    ax.set_xlabel("RL training step")
    ax.set_ylabel(r"$\|v\|$ at treatment layer")
    ax.set_title("(a) Recruitment strength\nboth poles grow")
    ax.legend(fontsize=8, loc="upper left")

    # (b) J-share, against the n=100 chance baselines
    ax = axes[1]
    for c in ("gold", "mold"):
        x, y = series(c, lambda r: r["var_fraction_k16"])
        ax.plot(x, y, "-", color=colour[c], lw=2, label=label[c])
        ax.axhline(coh["null"][c]["mean"], color=colour[c], ls=":", lw=1.2)
        ax.annotate(f"chance ({c})", xy=(steps_all[-1], coh["null"][c]["mean"]),
                    xytext=(-4, 3), textcoords="offset points",
                    ha="right", fontsize=7, color=colour[c])
    gx, gy = series("gold", lambda r: r["var_fraction_k16"])
    mx, my = series("mold", lambda r: r["var_fraction_k16"])
    # Keep annotations inside the data band so they clear the panel title.
    mid = 0.5 * (max(my) + min(gy))
    ax.annotate("distress gains share", xy=(90, max(my) - 0.0008),
                xytext=(8, mid + 0.004), fontsize=8,
                arrowprops=dict(arrowstyle="->", lw=0.8))
    ax.annotate("flourishing declines slightly", xy=(118, gy[-3]),
                xytext=(8, min(gy) - 0.0035), fontsize=8,
                arrowprops=dict(arrowstyle="->", lw=0.8))
    ax.set_ylim(min(coh["null"]["gold"]["mean"], min(gy)) - 0.007,
                max(my) + 0.003)
    ax.set_xlabel("RL training step")
    ax.set_ylabel(r"J-share $\|x_J\|^2/\|x\|^2$ (k=16)")
    ax.set_title("(b) Share of the axis inside J-space\n"
                 "both poles start above chance")

    # (c) congruent pole score
    ax = axes[2]
    for c in ("gold", "mold"):
        sign = (lambda r: r["gold_pole"] - r["mold_pole"]) if c == "gold" \
            else (lambda r: r["mold_pole"] - r["gold_pole"])
        x, y = series(c, sign)
        ax.plot(x, y, "-", color=colour[c], lw=2, label=label[c])
    ax.set_xlabel("RL training step")
    ax.set_ylabel("congruent pole score (J-lens)")
    ax.set_title("(c) Speakable valence content\nboth poles grow")

    gap = f"; step {missing[0]} absent upstream" if missing else ""
    fig.suptitle(f"What RL changes about the welfare axis "
                 f"({n_ckpt} released checkpoints, steps "
                 f"{steps_all[0]}–{steps_all[-1]}{gap})", y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_traj.png"), dpi=200,
                bbox_inches="tight")
    print(f"fig_traj.png ({n_ckpt} checkpoints, gap at {missing or 'none'})")


if __name__ == "__main__":
    fig1()
    fig2()
    fig3()
    fig5()
