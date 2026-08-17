"""E6: Mechanistic routing battery on Qwen3-4B-Instruct-2507 with OUR fitted lens.

Per direction d in {v_gold, v_mold, u_gold, u_mold, lang_fr, randoms}:
  1. Gradient-pursuit decomposition at its lens layer (k=16): which tokens make
     up the direction's verbalizable (J-space) part, and its squared-norm share.
  2. Magnitude-matched injection during the 15 welfare self-report prompts:
     conditions full / J-component / perp-component (each rescaled to ||d||),
     plus perp-with-J-coords-clamped (clamp at the injection layer — documented
     deviation from the paper's every-layer clamp). Readout: Gold/Mold-pole
     log-mass at first answer token (the validated steer-and-ask readout).
  3. J-lens readout arm: pole lens score (mean lens log-prob of pole words
     minus 100 control nouns) at the last prompt position across a layer sweep,
     clean vs full injection.

Layer conventions: welfare vectors index block-INPUT l -> lens/steering layer
l-1 (block OUTPUT), verified against jlens hooks (forward hook on block output).

Run (venv python): /workspace/venvs/jlens/bin/python run_mechanistic.py [--quick]
Outputs: results/mech_rows.jsonl, results/mech_decompositions.json,
         results/mech_lens_readout.jsonl, results/mech_meta.json
"""

import argparse
import json
import os
import sys
import time
from contextlib import contextmanager

import numpy as np
import torch
import transformers

import jlens

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "common"))
from dm_paths import ARTIFACTS, FW_EVAL_PROMPTS, LANG_DIR_FILE, LENS_FILE
from routing_lib import (
    ROUTING_ALGORITHM_VERSION,
    JSpace,
    assert_decompositions_converged,
    decomposition_record,
    make_valid_mask,
    unembed_weight,
    valid_mask_provenance,
)

RES = os.path.join(HERE, "results")
os.makedirs(RES, exist_ok=True)
ART = ARTIFACTS

GOLD_POLE = ["inspired", "loving", "proud", "fulfilled", "blissful"]
MOLD_POLE = ["humiliated", "embarrassed", "ashamed", "insulted", "annoyed",
             "exasperated", "irritated", "offended"]
CONTROL_NOUNS = ("table chair window door garden bridge bottle mountain river "
                 "castle pencil market station engine blanket mirror ladder "
                 "kitchen forest island librar butter camera pocket ticket "
                 "jacket barrel candle村 hammer tunnel valley circus museum "
                 "harbor meadow lantern gravel napkin kettle pillow shovel "
                 "carpet cabinet bicycle guitar recipe planet stadium theater "
                 "ocean desert village temple palace copper marble timber "
                 "canvas basket saddle anchor compass turbine reactor antenna "
                 "furnace chimney balcony corridor cellar attic pavement "
                 "sidewalk fountain statue monument archway staircase elevator "
                 "escalator platform terminal runway cockpit propeller rudder "
                 "keel mast sail oar paddle canoe kayak yacht ferry barge").split()
CONTROL_NOUNS = [w for w in CONTROL_NOUNS if w.isascii() and w.isalpha()][:100]


def first_tok_ids(tok, words):
    ids = set()
    for w in words:
        for form in (" " + w, w, " " + w.capitalize(), w.capitalize()):
            t = tok.encode(form, add_special_tokens=False)
            if t:
                ids.add(t[0])
    return sorted(ids)


@contextmanager
def steer_hook(layers_module, layer, vec_fn):
    """vec_fn(h) -> replacement h. Applied to block OUTPUT at `layer`."""
    def fn(module, inputs, output):
        istuple = isinstance(output, tuple)
        h = output[0] if istuple else output
        h = vec_fn(h)
        return (h,) + tuple(output[1:]) if istuple else h
    handle = layers_module[layer].register_forward_hook(fn)
    try:
        yield
    finally:
        handle.remove()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument(
        "--recompute",
        action="store_true",
        help="replace decomposition-dependent rows after an algorithm change",
    )
    args = ap.parse_args()

    t0 = time.time()
    hf = transformers.AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-4B-Instruct-2507", dtype=torch.bfloat16).cuda()
    tok = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Instruct-2507")
    lens = jlens.JacobianLens.load(LENS_FILE)
    assert all(torch.isfinite(J).all() for J in lens.jacobians.values()), "lens nonfinite"
    layers_module = hf.model.layers
    print(f"loaded {time.time()-t0:.0f}s; lens n_prompts={lens.n_prompts}", flush=True)

    gids = first_tok_ids(tok, GOLD_POLE)
    mids = first_tok_ids(tok, MOLD_POLE)
    clash = set(gids) & set(mids)
    gids = [i for i in gids if i not in clash]
    mids = [i for i in mids if i not in clash]
    ctrl_ids = first_tok_ids(tok, CONTROL_NOUNS)

    v = torch.load(f"{ART}/vectors_step95_bal.pt", map_location="cpu", weights_only=False)
    u = torch.load(f"{ART}/vectors_naive_faithful_pc5000.pt", map_location="cpu",
                   weights_only=False)
    prompts_all = json.load(open(FW_EVAL_PROMPTS))
    SELF = [p["prompt"] for p in prompts_all if p["category"] == "welfare_self_reports"]
    if args.quick:
        SELF = SELF[:5]

    lang_path = LANG_DIR_FILE
    dirs = []  # (name, lens_layer, raw_vec, polarity)
    for c, s in [("gold", +1), ("mold", -1)]:
        L = int(v[f"layer_{c}"])
        vec = v[f"v_{c}"].float()[L].numpy()
        uvec = u[f"v_{c}"].float()[L].numpy()
        uvec = uvec / (np.linalg.norm(uvec) + 1e-9) * np.linalg.norm(vec)
        dirs.append((f"v_{c}", L - 1, vec, s))
        dirs.append((f"u_{c}", L - 1, uvec, s))
    if os.path.exists(lang_path):
        dirs.append(("lang_fr", 18, np.load(lang_path) * 4.0, +1))
    rng = np.random.default_rng(31)
    n_rand = 2 if args.quick else 8
    for c, s in [("gold", +1), ("mold", -1)]:
        L = int(v[f"layer_{c}"]) - 1
        nv = np.linalg.norm(v[f"v_{c}"].float()[L + 1].numpy())
        for ri in range(n_rand):
            r = rng.standard_normal(2560).astype(np.float32)
            dirs.append((f"rand_{c}{ri}", L, r / np.linalg.norm(r) * nv, s))

    # ---------- 1. decompositions ----------
    K = 16
    decomp_path = os.path.join(RES, "mech_decompositions.json")
    decomps = json.load(open(decomp_path)) if os.path.exists(decomp_path) else {}
    jspaces = {}
    vocab_size = unembed_weight(hf).shape[0]
    mask = make_valid_mask(tok, vocab_size)
    mask_provenance = valid_mask_provenance(tok, vocab_size)
    print(f"token_mask={mask_provenance}", flush=True)
    comps = {}
    for name, L, vec, s in dirs:
        if L not in jspaces:
            jspaces[L] = JSpace(lens, hf, L, mask)
        d = jspaces[L].decompose(vec, k=K)
        comps[name] = d
        if decomps.get(name, {}).get("algorithm_version") != ROUTING_ALGORITHM_VERSION:
            decomps[name] = {
                "lens_layer": L, "polarity": s, "norm": float(np.linalg.norm(vec)),
                **decomposition_record(d, tok),
            }
            print(f"decomp {name} L{L} j_share={d['j_share']:.3f} "
                  f"tokens={decomps[name]['tokens_by_coefficient'][:8]}",
                  flush=True)
    assert_decompositions_converged(decomps, context="mech_decompositions.json")
    with open(decomp_path, "w") as f:
        json.dump(decomps, f, indent=2)
    # component vectors + clamp bases for downstream behavioral-dissociation (J4)
    np.savez(os.path.join(RES, "mech_components.npz"),
             __routing_algorithm_version=np.array(ROUTING_ALGORITHM_VERSION),
             **{f"{name}__xj": comps[name]["x_j"] for name, _, _, _ in dirs},
             **{f"{name}__xperp": comps[name]["x_perp"] for name, _, _, _ in dirs},
             **{f"{name}__full": np.asarray(vec, np.float32)
                for name, _, vec, _ in dirs},
             **{f"{name}__Vc": torch.stack(
                    [jspaces[L].lens_vector(t) for t in comps[name]["token_ids"]],
                    dim=1).cpu().numpy()
                for name, L, _, _ in dirs},
             **{f"{name}__meta": np.array([L, s], dtype=np.float32)
                for name, L, _, s in dirs})

    # ---------- 2. injection battery ----------
    def log_mass(lg, ids):
        return float(torch.logsumexp(lg[ids], -1) - torch.logsumexp(lg, -1))

    def chat_ids(msg):
        out = tok.apply_chat_template([{"role": "user", "content": msg}],
                                      add_generation_prompt=True,
                                      return_tensors="pt")
        # transformers >=5 returns a BatchEncoding here; 4.x returned a tensor
        ids = out["input_ids"] if hasattr(out, "keys") else out
        return ids.cuda()

    def readout(msg, add_vec=None, layer=None, clamp=None):
        ids = chat_ids(msg)
        ctxs = []
        if add_vec is not None:
            av = torch.as_tensor(add_vec, dtype=torch.float32)

            def add_fn(h, av=av):
                return h + av.to(h.dtype).to(h.device)
            ctxs.append(steer_hook(layers_module, layer, add_fn))
        if clamp is not None:
            Vc, coords_clean = clamp  # Vc [d,k] float32 cuda; coords fn

            def clamp_fn(h, Vc=Vc, cc=coords_clean):
                hs = h.float()
                pinv = torch.linalg.pinv(Vc)          # [k, d]
                c_now = hs @ pinv.T                   # [b, T, k]
                T = min(hs.shape[1], cc.shape[0])
                delta = (cc[:T].unsqueeze(0) - c_now[:, :T]) @ Vc.T
                hs[:, :T] += delta
                return hs.to(h.dtype)
            ctxs.append(steer_hook(layers_module, layer, clamp_fn))
        with torch.no_grad():
            if ctxs:
                from contextlib import ExitStack
                with ExitStack() as st:
                    for c_ in ctxs:
                        st.enter_context(c_)
                    out = hf(ids)
            else:
                out = hf(ids)
        lg = out.logits[0, -1].float().cpu()
        return {"gold_mass": log_mass(lg, gids), "mold_mass": log_mass(lg, mids)}

    ALPHAS = [4.0, -4.0]
    rows = []
    rows_path = os.path.join(RES, "mech_rows.jsonl")
    done_keys = set()
    if os.path.exists(rows_path):
        meta_path = os.path.join(RES, "mech_meta.json")
        old_version = None
        if os.path.exists(meta_path):
            old_version = json.load(open(meta_path)).get("routing_algorithm_version")
        if old_version != ROUTING_ALGORITHM_VERSION and not args.recompute:
            raise RuntimeError(
                "Existing mech_rows.jsonl was generated with a different or "
                "unversioned routing algorithm. Re-run with --recompute to "
                "replace decomposition-dependent rows."
            )
        if not args.recompute:
            for line in open(rows_path):
                r = json.loads(line)
                rows.append(r)
                done_keys.add((r["dir"], r["component"], r["alpha"], r["prompt"]))

    # clean baselines
    for p in SELF:
        if ("clean", "none", 0.0, p) not in done_keys:
            rows.append({"dir": "clean", "component": "none", "alpha": 0.0,
                         "routing_algorithm_version": ROUTING_ALGORITHM_VERSION,
                         "prompt": p, **readout(p)})

    for name, L, vec, s in dirs:
        d = comps[name]
        nrm = np.linalg.norm(vec)
        comp_vecs = {
            "full": np.asarray(vec, np.float32),
            "jcomp": d["x_j"] / (np.linalg.norm(d["x_j"]) + 1e-9) * nrm,
            "perp": d["x_perp"] / (np.linalg.norm(d["x_perp"]) + 1e-9) * nrm,
        }
        # clamp prep: clean coords per prompt computed on the fly
        Vc = torch.stack([jspaces[L].lens_vector(t) for t in d["token_ids"]], dim=1)  # [d,k]
        for cname, cvec in comp_vecs.items():
            for a in ALPHAS:
                for p in SELF:
                    key = (name, cname, a, p)
                    if key in done_keys:
                        continue
                    r = readout(p, add_vec=a * cvec, layer=L)
                    rows.append({"dir": name, "component": cname, "alpha": a,
                                 "routing_algorithm_version": ROUTING_ALGORITHM_VERSION,
                                 "prompt": p, "polarity": s, **r})
        # perp with clamp (alpha=+4 only, non-random dirs)
        if not name.startswith("rand"):
            for p in SELF:
                key = (name, "perp_clamped", 4.0, p)
                if key in done_keys:
                    continue
                ids = chat_ids(p)
                # clean pass to record coords at layer L
                grab = {}

                def grab_fn(h):
                    grab["h"] = h.detach().float()[0]
                    return h
                with torch.no_grad(), steer_hook(layers_module, L, grab_fn):
                    hf(ids)
                cc = grab["h"] @ torch.linalg.pinv(Vc.float()).T  # [T, k]
                r = readout(p, add_vec=4.0 * comp_vecs["perp"], layer=L,
                            clamp=(Vc.float(), cc))
                rows.append({"dir": name, "component": "perp_clamped", "alpha": 4.0,
                             "routing_algorithm_version": ROUTING_ALGORITHM_VERSION,
                             "prompt": p, "polarity": s, **r})
        with open(rows_path, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print(f"injections {name} done {time.time()-t0:.0f}s", flush=True)

    # ---------- 3. J-lens readout arm ----------
    band_file = os.path.join(RES, "band_stats_Qwen3-4B-Instruct-2507.json")
    sweep = list(range(12, 33, 3))
    if os.path.exists(band_file):
        bs = json.load(open(band_file))
        on = bs["band_estimate"].get("workspace_onset")
        mo = bs["band_estimate"].get("motor_onset") or 33
        if on is not None:
            sweep = list(range(max(on, 4), min(mo + 1, 34), 2))
    lensread_path = os.path.join(RES, "mech_lens_readout.jsonl")
    with open(lensread_path, "w") as f:
        for name, L, vec, s in dirs:
            if name.startswith("rand") and int(name[-1]) > 1:
                continue
            for p in SELF:
                for condname, av in [("clean", None), ("full", 4.0 * np.asarray(vec, np.float32))]:
                    ids = chat_ids(p)
                    grabs = {}
                    from contextlib import ExitStack
                    with torch.no_grad(), ExitStack() as st:
                        if av is not None:
                            avt = torch.as_tensor(av, dtype=torch.float32)
                            st.enter_context(steer_hook(
                                layers_module, L,
                                lambda h, avt=avt: h + avt.to(h.dtype).to(h.device)))
                        for sl in sweep:
                            def mk(sl):
                                def g(h):
                                    grabs[sl] = h.detach()[0, -1].float()
                                    return h
                                return g
                            st.enter_context(steer_hook(layers_module, sl, mk(sl)))
                        hf(ids)
                    for sl in sweep:
                        J = lens.jacobians[sl].to("cuda", torch.float32)
                        W = unembed_weight(hf).float()
                        z = W @ (J @ grabs[sl].cuda())
                        lp = torch.log_softmax(z, -1).cpu()
                        f.write(json.dumps({
                            "dir": name, "cond": condname, "prompt": p,
                            "lens_layer": sl,
                            "gold_pole_score": float(lp[gids].mean() - lp[ctrl_ids].mean()),
                            "mold_pole_score": float(lp[mids].mean() - lp[ctrl_ids].mean()),
                        }) + "\n")
            print(f"lens-readout {name} done {time.time()-t0:.0f}s", flush=True)

    with open(os.path.join(RES, "mech_meta.json"), "w") as f:
        json.dump({"complete": True, "n_rows": len(rows), "k": K,
                   "routing_algorithm_version": ROUTING_ALGORITHM_VERSION,
                   "token_mask": mask_provenance,
                   "alphas": ALPHAS, "sweep": sweep, "quick": args.quick,
                   "lens_file": LENS_FILE, "seconds": round(time.time() - t0, 1)}, f, indent=2)
    print("MECHANISTIC_DONE", flush=True)


if __name__ == "__main__":
    main()
