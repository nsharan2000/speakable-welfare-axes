#!/usr/bin/env python3
"""F2 (audit) — J-share random cohort 8 -> 100 per polarity.

The J-share claim (v_mold z=+11.2, v_gold z=+5.1, naive at chance) is capped
at perm p = 1/(8+1) = 0.111 by cohort size alone. This script builds a
100-direction null per polarity with PER-NAME seeded randoms
(default_rng([31, pol, ri]) — NOT run_mechanistic.py's single shared stream,
which silently changes every rand_mold* identity when n_rand grows; see
audit-instructions/90-gotchas.md #1).

Writes ONLY new files (never touches mech_decompositions.json /
mech_components.npz):
  results/mech_decompositions_n100.json   (checkpoint/resume by name)
  results/mech_components_n100.npz
  results/jshare_cohort_n100.json         (summary w/ z, exact perm p, pcts)

Self-verification before trusting anything:
  - v_gold/u_gold/v_mold/u_mold/lang_fr recomputed fresh; each must match the
    existing mech_decompositions.json var_fraction to 1e-6 (same vector, same
    lens, same code path) or the run aborts.
  - new null mean must land within ~1 SD of the old 8-direction null
    (0.0378 gold / 0.0451 mold) — logged, not asserted (different stream).

Cost: 205 decompositions x ~14.3 s ~= 49 min on the Spark.
Run inside dm-exp: python f2_jshare_cohort.py
"""
import json
import os
import sys
import time

import numpy as np
import torch
import transformers

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "common"))
import jlens  # noqa: E402  (venv: /workspace/venvs/jlens)
from dm_paths import ARTIFACTS, LANG_DIR_FILE, LENS_FILE  # noqa: E402
from routing_lib import (  # noqa: E402
    ROUTING_ALGORITHM_VERSION,
    JSpace,
    assert_decompositions_converged,
    decomposition_record,
    make_valid_mask,
    unembed_weight,
    valid_mask_provenance,
)

RES = os.path.join(HERE, "results")
ART = ARTIFACTS
LANG = LANG_DIR_FILE

K = 16
N_RAND = 100
SEED_BASE = 31


def rand_dir(concept, ri, d=2560, base=SEED_BASE):
    """Identity is a pure function of the name — never a shared stream."""
    rng = np.random.default_rng([base, 0 if concept == "gold" else 1, ri])
    r = rng.standard_normal(d).astype(np.float32)
    return r / np.linalg.norm(r)


def main():
    t0 = time.time()
    hf = transformers.AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-4B-Instruct-2507", dtype=torch.bfloat16).cuda()
    tok = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Instruct-2507")
    lens = jlens.JacobianLens.load(LENS_FILE)
    assert all(torch.isfinite(J).all() for J in lens.jacobians.values())
    vocab_size = unembed_weight(hf).shape[0]
    mask = make_valid_mask(tok, vocab_size)
    mask_provenance = valid_mask_provenance(tok, vocab_size)
    print(f"loaded {time.time()-t0:.0f}s  token_mask={mask_provenance}", flush=True)

    v = torch.load(f"{ART}/vectors_step95_bal.pt", map_location="cpu",
                   weights_only=False)
    u = torch.load(f"{ART}/vectors_naive_faithful_pc5000.pt", map_location="cpu",
                   weights_only=False)

    # (name, lens_layer, vec, polarity) — identical construction to
    # run_mechanistic.py for the five real targets.
    dirs = []
    norms = {}
    for c, s in [("gold", +1), ("mold", -1)]:
        L = int(v[f"layer_{c}"])
        vec = v[f"v_{c}"].float()[L].numpy()
        uvec = u[f"v_{c}"].float()[L].numpy()
        uvec = uvec / (np.linalg.norm(uvec) + 1e-9) * np.linalg.norm(vec)
        norms[c] = float(np.linalg.norm(vec))
        dirs.append((f"v_{c}", L - 1, vec, s))
        dirs.append((f"u_{c}", L - 1, uvec, s))
    if os.path.exists(LANG):
        dirs.append(("lang_fr", 18, np.load(LANG) * 4.0, +1))
    lens_layers = {"gold": int(v["layer_gold"]) - 1, "mold": int(v["layer_mold"]) - 1}
    for c, s in [("gold", +1), ("mold", -1)]:
        for ri in range(N_RAND):
            dirs.append((f"randn_{c}{ri}", lens_layers[c],
                         rand_dir(c, ri) * norms[c], s))

    dec_path = os.path.join(RES, "mech_decompositions_n100.json")
    decomps = json.load(open(dec_path)) if os.path.exists(dec_path) else {}
    comps = {}
    jspaces = {}
    for i, (name, L, vec, s) in enumerate(dirs):
        if decomps.get(name, {}).get("algorithm_version") == ROUTING_ALGORITHM_VERSION:
            continue
        if L not in jspaces:
            jspaces[L] = JSpace(lens, hf, L, mask)
        d = jspaces[L].decompose(vec, k=K)
        decomps[name] = {
            "lens_layer": L, "polarity": s, "norm": float(np.linalg.norm(vec)),
            **decomposition_record(d, tok),
            "seed_scheme": f"default_rng([{SEED_BASE}, pol, ri])"
                           if name.startswith("randn_") else "artifact",
        }
        comps[name] = d
        if i % 10 == 0 or not name.startswith("randn_"):
            print(f"[{i+1}/{len(dirs)}] {name} L{L} "
                  f"j_share={d['j_share']:.4f} "
                  f"({(time.time()-t0)/60:.1f} min)", flush=True)
        with open(dec_path, "w") as f:      # checkpoint every step
            json.dump(decomps, f, indent=1)

    # components npz (only for freshly computed ones; resume keeps old file)
    npz_path = os.path.join(RES, "mech_components_n100.npz")
    if comps:
        arrs = {}
        if os.path.exists(npz_path):
            old = np.load(npz_path)
            arrs.update({k: old[k] for k in old.files})
        for name, d in comps.items():
            for key in ("x_j", "x_perp"):
                if key in d:
                    arrs[f"{name}__{key.replace('_', '')}"] = (
                        np.asarray(d[key].cpu() if torch.is_tensor(d[key])
                                   else d[key], np.float32))
        arrs["__routing_algorithm_version"] = np.array(ROUTING_ALGORITHM_VERSION)
        np.savez_compressed(npz_path, **arrs)

    # ---- convergence gate (cheap; fails the summary, not the 49 min of compute) ----
    assert_decompositions_converged(decomps, context="mech_decompositions_n100.json")

    # ---- cross-file verification gate ----
    # The five real targets are recomputed here from scratch. They must match
    # run_mechanistic.py's independently-produced values, or the lens/layer
    # convention has drifted and the null is meaningless.
    old_dec = json.load(open(os.path.join(RES, "mech_decompositions.json")))
    bad, skipped = [], []
    for t in ["v_gold", "u_gold", "v_mold", "u_mold", "lang_fr"]:
        if t not in old_dec or t not in decomps:
            continue
        old_v = old_dec[t].get("algorithm_version")
        new_v = decomps[t].get("algorithm_version")
        if old_v != ROUTING_ALGORITHM_VERSION or new_v != ROUTING_ALGORITHM_VERSION:
            skipped.append({"target": t, "existing_version": old_v or "v1_unversioned",
                            "recomputed_version": new_v})
            continue
        diff = abs(old_dec[t]["var_fraction"] - decomps[t]["var_fraction"])
        if diff > 1e-6:
            bad.append((t, diff))

    gate = {
        "checked": [t for t in ["v_gold", "u_gold", "v_mold", "u_mold", "lang_fr"]
                    if t in old_dec and t in decomps
                    and not any(s["target"] == t for s in skipped)],
        "skipped_version_mismatch": skipped,
        "status": "SKIPPED_VERSION_MISMATCH" if skipped and not bad else
                  ("FAIL" if bad else "PASS"),
    }
    if skipped:
        # Loud, and recorded in the artifact — never a silent pass.
        print("=" * 72, flush=True)
        print(f"F2_GATE_SKIPPED: {len(skipped)} of 5 targets could NOT be "
              f"cross-verified against mech_decompositions.json because the two "
              f"files were produced by different routing-algorithm versions:",
              flush=True)
        for s in skipped:
            print(f"    {s['target']}: existing={s['existing_version']} "
                  f"recomputed={s['recomputed_version']}", flush=True)
        print("  The n=100 null in this run is therefore UNVERIFIED against an "
              "independent computation of the same targets.", flush=True)
        print("  Fix: re-run run_mechanistic.py under this algorithm version "
              "first, then re-run this script to restore the gate.", flush=True)
        print("=" * 72, flush=True)
    if bad:
        print(f"F2_FAIL: target var_fractions moved vs existing file: {bad} "
              f"— lens/layer convention changed, do not trust the null",
              flush=True)
        sys.exit(1)

    # ---- summary ----
    out = {"n_random_per_polarity": N_RAND, "k": K,
           "algorithm_version": ROUTING_ALGORITHM_VERSION,
           "metric": "j_share = ||x_j||^2 / ||x||^2",
           "seed_scheme": f"default_rng([{SEED_BASE}, pol, ri]) per-name",
           "layers": {c: lens_layers[c] + 1 for c in lens_layers},
           "lens_layers": lens_layers,
           "token_mask": mask_provenance,
           "verification_gate": gate,
           "targets": {}, "null": {}}
    for c in ["gold", "mold"]:
        null = [decomps[f"randn_{c}{ri}"]["var_fraction"] for ri in range(N_RAND)]
        arr = np.array(null)
        out["null"][c] = {"values": null, "mean": float(arr.mean()),
                          "sd": float(arr.std(ddof=1)),
                          "p05": float(np.percentile(arr, 5)),
                          "p95": float(np.percentile(arr, 95))}
        for t in [f"v_{c}", f"u_{c}"] + (["lang_fr"] if c == "gold" else []):
            vf = decomps[t]["var_fraction"]
            n_ge = int((arr >= vf).sum())
            out["targets"][t] = {
                "var_fraction": vf,
                "z": float((vf - arr.mean()) / arr.std(ddof=1)),
                "n_randoms_ge": n_ge,
                "perm_p": (n_ge + 1) / (N_RAND + 1),
                "null_polarity": c,
            }
    json.dump(out, open(os.path.join(RES, "jshare_cohort_n100.json"), "w"),
              indent=1)
    print("F2_SUMMARY " + json.dumps({t: {k: round(vv, 4) if isinstance(vv, float)
          else vv for k, vv in d.items() if k != "null_polarity"}
          for t, d in out["targets"].items()}), flush=True)
    print(f"F2_DONE {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
