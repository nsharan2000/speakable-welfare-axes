#!/usr/bin/env python3
"""Appendix figure B1: blind-judged sentiment under steering for the trained
axis, both naive-control constructions and norm-matched random directions,
at each pole (report §4.3–4.4).

Regenerates experiments/figures/out/fig_u_adjudication.png from the judged
sentiment curves. Every number is loaded from its results file; nothing is
hardcoded. Needs only numpy + matplotlib (no torch, no model).

  python experiments/figures/fig_naive_controls.py

Inputs (experiments/welfare-axis/results/):
  judge_sentiment.json        v, faithful-walk u, random  (greedy decoding)
  judge_sentiment_ownu.json   Han-style re-extracted u    (greedy decoding)
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
WA = os.path.join(REPO, "experiments", "welfare-axis", "results")
OUT = os.path.join(HERE, "out")

ALPHAS = [-4.0, -2.0, 2.0, 4.0]


def curve(d, name):
    return [d["curves"][f"{name}@{a:+.0f}"]["mean"] for a in ALPHAS]


def main():
    with open(os.path.join(WA, "judge_sentiment.json")) as f:
        main_run = json.load(f)
    with open(os.path.join(WA, "judge_sentiment_ownu.json")) as f:
        ownu = json.load(f)
    baseline = main_run["baseline"]["mean"]

    style = {
        "gold": [
            ("v_gold", main_run, "v_gold (trained)", "#c99700", "-"),
            ("u_gold", main_run, "u_gold (faithful walk)", "#7d6a2b", "--"),
            ("u_gold", ownu, "u_gold (Han-style walk)", "#3f5f3a", "-."),
            ("rand_gold", main_run, "random", "#b5b5b5", ":"),
        ],
        "mold": [
            ("v_mold", main_run, "v_mold (trained)", "#7a1f1f", "-"),
            ("u_mold", main_run, "u_mold (faithful walk)", "#c98a8a", "--"),
            ("u_mold", ownu, "u_mold (Han-style walk)", "#4b3d6b", "-."),
            ("rand_mold", main_run, "random", "#b5b5b5", ":"),
        ],
    }

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6), sharey=True)
    for ax, (pole, series) in zip(axes, style.items()):
        for name, src, label, color, ls in series:
            ax.plot(ALPHAS, curve(src, name), ls, marker="o", color=color,
                    lw=2 if ls == "-" else 1.7, label=label)
        ax.axhline(baseline, color="#666666", lw=0.8)
        ax.set_title(f"{pole.capitalize()} pole")
        ax.set_xlabel("steering strength α (u, random norm-matched to v)")
        ax.set_xticks(range(-4, 5))
        ax.legend(frameon=True, fontsize=9)
    axes[0].set_ylabel("judge sentiment (−5…+5)")
    fig.suptitle("Blind-judged sentiment under steering: trained axis, "
                 "two naive-control constructions, random directions")
    fig.tight_layout()
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "fig_u_adjudication.png")
    fig.savefig(path, dpi=200)
    print("wrote", path)

    # spans quoted in the report (α=+4 minus α=−4)
    for pole in ("gold", "mold"):
        for name, src, label, _, _ in style[pole]:
            c = curve(src, name)
            print(f"  {label:26s} span {c[-1] - c[0]:+.2f}")


if __name__ == "__main__":
    main()
