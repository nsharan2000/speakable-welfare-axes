"""Validate J-space decomposition on a matched model/lens pair.

The default exercises the public Qwen3-4B lens. Pass ``--primary`` to test the
project's Qwen3-4B-Instruct-2507 lens at the actual Gold/Mold source layers.
Tests cover a pure atom, a planted non-negative mixture, scale invariance,
reconstruction identity, NNLS convergence, and a random-direction cohort.

Primary run from the repository root:
``python experiments/routing-core/selftest_routing.py --primary``.
"""

import argparse
import hashlib
import json
import os
import sys
import time

import torch
import transformers

import jlens

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "common"))
from dm_paths import LENS_FILE
from routing_lib import run_selftest

RES = os.path.join(HERE, "results")
os.makedirs(RES, exist_ok=True)

PRIMARY_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
PRIMARY_LENS = LENS_FILE

ap = argparse.ArgumentParser()
ap.add_argument("--primary", action="store_true",
                help="test the project's Instruct-2507 lens at layers 20 and 23")
ap.add_argument("--model", default=None,
                help="override the Hugging Face model ID")
ap.add_argument("--lens-file", default=None,
                help="local fitted lens .pt; overrides the public lens")
ap.add_argument("--layers", type=int, nargs="+", default=None)
ap.add_argument("--k", type=int, default=16)
ap.add_argument("--n-random", type=int, default=8)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--output", default=None)
args = ap.parse_args()


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

if args.primary:
    model_id = args.model or PRIMARY_MODEL
    lens_file = os.path.abspath(args.lens_file or PRIMARY_LENS)
    layers = args.layers or [20, 23]
    output = args.output or "routing_selftest_primary.json"
else:
    model_id = args.model or "Qwen/Qwen3-4B"
    lens_file = os.path.abspath(args.lens_file) if args.lens_file else None
    layers = args.layers or [12, 20, 28]
    output = args.output or "routing_selftest.json"

t0 = time.time()
hf = transformers.AutoModelForCausalLM.from_pretrained(
    model_id, dtype=torch.bfloat16).cuda()
hf.eval()
tok = transformers.AutoTokenizer.from_pretrained(model_id)
if lens_file:
    if not os.path.exists(lens_file):
        raise FileNotFoundError(f"Lens file not found: {lens_file}")
    lens = jlens.JacobianLens.load(lens_file)
else:
    lens = jlens.JacobianLens.from_pretrained(
        "neuronpedia/jacobian-lens",
        filename="qwen3-4b/jlens/Salesforce-wikitext/Qwen3-4B_jacobian_lens.pt")

out = {
    "model": model_id,
    "lens": lens_file or "neuronpedia/jacobian-lens:Qwen3-4B",
    "lens_sha256": sha256_file(lens_file) if lens_file else None,
    "layers": layers,
    "k": args.k,
    "n_random": args.n_random,
    "seed": args.seed,
    "tests": {},
}
for layer in layers:
    out["tests"][f"L{layer}"] = run_selftest(
        lens, hf, tok, layer, k=args.k, seed=args.seed,
        n_random=args.n_random,
    )
    print(f"L{layer}", json.dumps(out["tests"][f"L{layer}"]), flush=True)

out["seconds"] = round(time.time() - t0, 1)
out["all_pass"] = all(v["pass"] for v in out["tests"].values())
output_path = output if os.path.isabs(output) else os.path.join(RES, output)
with open(output_path, "w") as f:
    json.dump(out, f, indent=2)
print(f"wrote {output_path}", flush=True)
print("ROUTING_SELFTEST_" + ("PASS" if out["all_pass"] else "FAIL"), flush=True)
