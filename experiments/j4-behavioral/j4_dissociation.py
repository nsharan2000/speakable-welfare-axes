"""J4: Behavioral dissociation — is the welfare axis's behavioral drive carried
by its speakable (J-space) or unspeakable component? CONFESSION-FREE: all
readouts are task behavior (sentiment of neutral-task text, GSM8K
backtracking, refusal), never self-report.

Arms per concept (gold, mold), all magnitude-matched to ||v||:
  clean            no intervention
  full@a           inject a*v at the treatment layer
  jcomp@a          inject a*v_J (rescaled)   -> if speakable channel carries
  perp@a           inject a*v_perp (rescaled)   the drive, jcomp~=full, perp~=0
  full_clamped@a   inject a*v while clamping the direction's k J-lens
                   coordinates to clean values -> behavior surviving the clamp
                   means the drive does NOT need the speakable channel
Inputs: mech_components.npz from run_mechanistic (components + Vc + layer).
Tasks (from Han et al.'s own eval sets):
  sentiment    16 neutral prompts x 60-tok gens (judge-scored later)
  backtracking 20 GSM8K problems x 220-tok gens (marker count + judge later)
  refusal      20 OR-Bench prompts x 60-tok gens (judge later)
ETA: ~30-45 min GPU (~500 generations). Run (system python):
  python3 j4_dissociation.py [--quick]
Output: results/j4_rows.jsonl, results/j4_meta.json  (idempotent per row)
"""

import argparse
import json
import os
import sys
import time
from contextlib import contextmanager

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "common"))
from dm_common import load_model
from dm_paths import FW_DATASETS, MECH_COMPONENTS
from prompts import NEUTRAL_PROMPTS

RES = os.path.join(HERE, "results")
os.makedirs(RES, exist_ok=True)
NPZ = MECH_COMPONENTS
FW = FW_DATASETS

BACKTRACK_MARKERS = ["wait", "actually", "hmm", "hold on", "no, that",
                     "that's not right", "let me reconsider", "i made a mistake",
                     "scratch that", "on second thought"]


@contextmanager
def hook(layers_module, layer, fn):
    def h(module, inputs, output):
        istuple = isinstance(output, tuple)
        x = output[0] if istuple else output
        x = fn(x)
        return (x,) + tuple(output[1:]) if istuple else x
    hd = layers_module[layer].register_forward_hook(h)
    try:
        yield
    finally:
        hd.remove()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument(
        "--recompute",
        action="store_true",
        help="replace rows when mech_components.npz uses a new routing algorithm",
    )
    ap.add_argument("--rand-jcomp", action="store_true",
                    help="F1 audit control: run arm=rand_jcomp{ri} (J-component "
                         "of the 8 pre-existing random directions per polarity, "
                         "identical rescaling to the real-axis jcomp arm), "
                         "sentiment task only. Resume-safe: distinct arm names.")
    args = ap.parse_args()
    if args.recompute and args.rand_jcomp:
        raise ValueError(
            "Do not combine --recompute with --rand-jcomp. Recompute the full "
            "battery first, then run --rand-jcomp as a separate resume step."
        )

    z = np.load(NPZ)
    component_version = (
        str(z["__routing_algorithm_version"].item())
        if "__routing_algorithm_version" in z.files else None
    )
    model, tok = load_model("instruct")
    layers_module = model.model.layers

    gsm = [x["prompt"] for x in json.load(open(f"{FW}/gsm8k_eval_prompts.json"))]
    orb = [x["prompt"] for x in json.load(open(f"{FW}/or_bench_eval_prompts.json"))]
    n_sent, n_gsm, n_orb = (6, 6, 6) if args.quick else (16, 20, 20)
    tasks = {
        "sentiment": (NEUTRAL_PROMPTS[:n_sent], 60),
        "backtracking": (gsm[:n_gsm], 220),
        "refusal": (orb[:n_orb], 60),
    }

    ALPHA = 4.0
    arms = []  # (concept, arm_name, vec or None, clamp Vc or None)
    arms.append(("none", "clean", None, None))
    for c in ["gold", "mold"]:
        name = f"v_{c}"
        L = int(z[f"{name}__meta"][0])
        full = z[f"{name}__full"]
        nrm = float(np.linalg.norm(full))
        xj = z[f"{name}__xj"]
        xp = z[f"{name}__xperp"]
        Vc = torch.tensor(z[f"{name}__Vc"], dtype=torch.float32)
        arms += [
            (c, "full", (L, ALPHA * full), None),
            (c, "jcomp", (L, ALPHA * xj / (np.linalg.norm(xj) + 1e-9) * nrm), None),
            (c, "perp", (L, ALPHA * xp / (np.linalg.norm(xp) + 1e-9) * nrm), None),
            (c, "full_clamped", (L, ALPHA * full), (L, Vc)),
        ]

    if args.rand_jcomp:
        # F1: the rescaling itself is the thing under test, so it must match
        # the real-axis jcomp arm exactly (norm target = that random's own
        # __full norm, which the npz already matched to the corresponding v).
        N_RAND_J4 = 8
        for c in ["gold", "mold"]:
            for ri in range(N_RAND_J4):
                name = f"rand_{c}{ri}"
                L = int(z[f"{name}__meta"][0])
                full = z[f"{name}__full"]
                nrm = float(np.linalg.norm(full))
                xj = z[f"{name}__xj"]
                arms.append((c, f"rand_jcomp{ri}",
                             (L, ALPHA * xj / (np.linalg.norm(xj) + 1e-9) * nrm),
                             None))
        tasks = {"sentiment": tasks["sentiment"]}

    rows_path = os.path.join(RES, "j4_rows.jsonl")
    done = set()
    rows = []
    if os.path.exists(rows_path):
        meta_path = os.path.join(RES, "j4_meta.json")
        old_version = None
        if os.path.exists(meta_path):
            old_version = json.load(open(meta_path)).get("routing_algorithm_version")
        if old_version != component_version and not args.recompute:
            raise RuntimeError(
                "Existing J4 rows were generated from a different or unversioned "
                "J-space decomposition. Run the full battery once with "
                "--recompute before adding the --rand-jcomp arm."
            )
        if not args.recompute:
            for line in open(rows_path):
                r = json.loads(line)
                rows.append(r)
                done.add((r["concept"], r["arm"], r["task"], r["prompt"]))

    def gen(msg, steer=None, clamp=None, n_tok=60):
        ids = tok.apply_chat_template([{"role": "user", "content": msg}],
                                      add_generation_prompt=True,
                                      return_tensors="pt").cuda()
        ctxs = []
        if steer is not None:
            L, vec = steer
            vt = torch.tensor(np.asarray(vec, np.float32))
            ctxs.append(hook(layers_module, L,
                             lambda x, vt=vt: x + vt.to(x.dtype).to(x.device)))
        if clamp is not None:
            Lc, Vc = clamp
            # clean coords cache for the PROMPT region; during generation new
            # positions are clamped to the mean clean coordinate (approximation
            # noted in the writeup).
            grab = {}
            with torch.no_grad(), hook(layers_module, Lc,
                                       lambda x: (grab.__setitem__("h", x.detach().float()[0]), x)[1]):
                model(ids)
            pinv = torch.linalg.pinv(Vc.to("cuda"))
            cc = grab["h"].cuda() @ pinv.T           # [T, k]
            cc_mean = cc.mean(0)

            def clamp_fn(x, Vc=Vc.to("cuda"), cc=cc, ccm=cc_mean, pinv=pinv):
                xs = x.float()
                c_now = xs @ pinv.T.to(xs.device)
                T = xs.shape[1]
                if T <= cc.shape[0]:
                    target = cc[:T]
                else:
                    target = torch.cat([cc, ccm.expand(T - cc.shape[0], -1)], 0)
                xs = xs + (target.unsqueeze(0) - c_now) @ Vc.T
                return xs.to(x.dtype)
            ctxs.append(hook(layers_module, Lc, clamp_fn))
        from contextlib import ExitStack
        with torch.no_grad(), ExitStack() as st:
            for c_ in ctxs:
                st.enter_context(c_)
            out = model.generate(ids, max_new_tokens=n_tok, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
        return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)

    t0 = time.time()
    for concept, arm, steer, clamp in arms:
        for task, (prompts, n_tok) in tasks.items():
            for p in prompts:
                if (concept, arm, task, p) in done:
                    continue
                txt = gen(p, steer=steer, clamp=clamp, n_tok=n_tok)
                low = txt.lower()
                rows.append({
                    "concept": concept, "arm": arm, "task": task, "prompt": p,
                    "routing_algorithm_version": component_version,
                    "text": txt,
                    "backtrack_markers": sum(low.count(m) for m in BACKTRACK_MARKERS),
                })
                done.add((concept, arm, task, p))
            with open(rows_path, "w") as f:
                for r in rows:
                    f.write(json.dumps(r) + "\n")
        print(f"{concept}/{arm} done {time.time()-t0:.0f}s", flush=True)

    with open(os.path.join(RES, "j4_meta.json"), "w") as f:
        json.dump({"complete": True, "n_rows": len(rows), "alpha": ALPHA,
                   "routing_algorithm_version": component_version,
                   "tasks": {k: len(v[0]) for k, v in tasks.items()},
                   "seconds": round(time.time() - t0, 1)}, f, indent=2)
    print("J4_DONE", flush=True)


if __name__ == "__main__":
    main()
