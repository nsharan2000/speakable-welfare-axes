#!/usr/bin/env python3
"""Compare fresh v2 results against the committed v1 baseline, per experiment.

For every JSON file present in both results/ and results_committed/, walk the
two structures in parallel and report numeric leaves whose relative difference
exceeds --tol (default 1%%). Non-numeric mismatches (token lists, verdict
strings, pass flags) are reported when --strings is set or when the key looks
decision-bearing (pass/verdict/status/p/n_ge).

Usage: python3 compare_results.py [--tol 0.01] [root]
"""
import argparse
import json
import math
import os
import sys

DECISION_KEYS = ("pass", "verdict", "status", "complete", "gate")
SKIP_KEYS = (
    "algorithm", "diagnostics", "provenance", "timestamp", "date", "seconds",
    "runtime", "host", "path", "file", "source", "python", "torch", "wrote",
    "algorithm_version", "nnls", "iterations", "git", "sha",
)


def walk(a, b, path, out, tol, strings):
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            p = f"{path}.{k}" if path else k
            if any(s in k.lower() for s in SKIP_KEYS):
                continue
            if k not in a:
                out.append(("ONLY_NEW", p, None, summ(b[k])))
            elif k not in b:
                out.append(("ONLY_OLD", p, summ(a[k]), None))
            else:
                walk(a[k], b[k], p, out, tol, strings)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append(("LEN", path, len(a), len(b)))
        if a and b and all(isinstance(x, (int, float)) for x in a[:5]) \
                and all(isinstance(x, (int, float)) for x in b[:5]):
            for i, (x, y) in enumerate(zip(a, b)):
                cmp_num(x, y, f"{path}[{i}]", out, tol)
        else:
            for i, (x, y) in enumerate(zip(a, b)):
                walk(x, y, f"{path}[{i}]", out, tol, strings)
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)) \
            and not isinstance(a, bool) and not isinstance(b, bool):
        cmp_num(a, b, path, out, tol)
    else:
        if a != b:
            leaf = path.rsplit(".", 1)[-1].lower()
            if strings or any(d in leaf for d in DECISION_KEYS):
                out.append(("DIFF", path, summ(a), summ(b)))


def cmp_num(a, b, path, out, tol):
    if a == b:
        return
    denom = max(abs(a), abs(b), 1e-12)
    rel = abs(a - b) / denom
    if rel > tol:
        out.append(("NUM", path, a, b, rel))


def summ(v):
    s = json.dumps(v, default=str)
    return s if len(s) <= 90 else s[:87] + "..."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", nargs="?", default=os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--tol", type=float, default=0.01)
    ap.add_argument("--strings", action="store_true")
    args = ap.parse_args()

    total_files = same = 0
    for exp in sorted(os.listdir(args.root)):
        old_dir = os.path.join(args.root, exp, "results_committed")
        new_dir = os.path.join(args.root, exp, "results")
        if not (os.path.isdir(old_dir) and os.path.isdir(new_dir)):
            continue
        for fn in sorted(os.listdir(new_dir)):
            if not fn.endswith(".json"):
                continue
            old_fp, new_fp = os.path.join(old_dir, fn), os.path.join(new_dir, fn)
            if not os.path.exists(old_fp) or os.path.islink(new_fp):
                continue
            try:
                old, new = json.load(open(old_fp)), json.load(open(new_fp))
            except Exception as e:
                print(f"!! {exp}/{fn}: unreadable ({e})")
                continue
            total_files += 1
            out = []
            walk(new, old, "", out, args.tol, args.strings)
            if not out:
                same += 1
                print(f"== {exp}/{fn}: no differences beyond tol")
                continue
            print(f"-- {exp}/{fn}: {len(out)} difference(s)  [NEW vs COMMITTED]")
            for row in out[:30]:
                if row[0] == "NUM":
                    _, p, a, b, rel = row
                    print(f"   NUM  {p}: new={a:.6g} old={b:.6g} rel={rel:.2%}")
                else:
                    print(f"   {row[0]:8s} {row[1]}: new={row[2]} old={row[3]}")
            if len(out) > 30:
                print(f"   ... +{len(out) - 30} more")
    print(f"\n{total_files} file(s) compared, {same} identical within tol")


if __name__ == "__main__":
    main()
