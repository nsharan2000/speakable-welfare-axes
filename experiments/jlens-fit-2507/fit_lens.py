"""Fit a Jacobian lens for Qwen/Qwen3-4B-Instruct-2507.

The functional-welfare directions were extracted from Instruct-2507, whereas
the available public Qwen lens targets a different checkpoint. Using the
official jlens implementation, we therefore fit a matching lens on 150
WikiText-103 prompts, with maximum sequence length 128, dimension batches of
128, and the final transformer layer as the target. These settings follow the
public Qwen3-4B lens recipe, although our smaller prompt sample reflects the
sprint compute budget.

The fit checkpoints every five prompts and supports resumption. The final lens
is saved in float32 because an earlier float16 save produced non-finite values
(repo issue #6).

Run the primary fit with:
    /workspace/venvs/jlens/bin/python fit_lens.py --n 150
Run a three-prompt smoke test with:
    /workspace/venvs/jlens/bin/python fit_lens.py --smoke
"""

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import random
import time

import torch
import transformers

import jlens
from jlens.examples import load_wikitext_prompts

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
os.makedirs(RES, exist_ok=True)

MODEL = "Qwen/Qwen3-4B-Instruct-2507"
SEED = 0


def package_version(name):
    """Return an installed package version without making fitting depend on it."""
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def sha256_file(path):
    """Hash a saved artifact in chunks so large lens files need little RAM."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=150)
ap.add_argument("--smoke", action="store_true", help="3-prompt smoke fit only")
args = ap.parse_args()

random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

t0 = time.time()
hf = transformers.AutoModelForCausalLM.from_pretrained(
    MODEL, dtype=torch.bfloat16
).cuda()
hf.eval()
tok = transformers.AutoTokenizer.from_pretrained(MODEL)
model = jlens.from_hf(hf, tok)
print(f"model loaded {time.time()-t0:.0f}s", flush=True)

n = 3 if args.smoke else args.n
prompts = load_wikitext_prompts(n_prompts=n)
prompt_payload = json.dumps(prompts, ensure_ascii=False, default=str)
prompt_sha256 = hashlib.sha256(prompt_payload.encode("utf-8")).hexdigest()
print(f"{len(prompts)} prompts loaded", flush=True)

ckpt = os.path.join(RES, "fit_ckpt.pt" if not args.smoke else "smoke_ckpt.pt")
t1 = time.time()
lens = jlens.fit(model, prompts, dim_batch=128, max_seq_len=128,
                 checkpoint_path=ckpt, checkpoint_every=5)
fit_s = time.time() - t1

finite = all(torch.isfinite(J).all().item() for J in lens.jacobians.values())
if not finite:
    raise RuntimeError("Fitted lens contains NaN or infinite values")

outname = "smoke_lens.pt" if args.smoke else "Qwen3-4B-Instruct-2507_jacobian_lens.pt"
outpath = os.path.join(RES, outname)
lens.save(outpath, dtype=torch.float32)

meta = {
    "model": MODEL, "n_prompts_requested": n, "n_prompts_fitted": int(lens.n_prompts),
    "dim_batch": 128, "max_seq_len": 128, "target_layer": "default(final)",
    "corpus": "wikitext via jlens.examples.load_wikitext_prompts",
    "prompt_sha256": prompt_sha256, "seed": SEED,
    "finite": bool(finite), "fit_seconds": round(fit_s, 1),
    "seconds_per_prompt": round(fit_s / max(len(prompts), 1), 2),
    "d_model": int(lens.d_model), "n_source_layers": len(lens.source_layers),
    "model_revision": getattr(hf.config, "_commit_hash", None),
    "python_version": platform.python_version(),
    "torch_version": torch.__version__,
    "transformers_version": transformers.__version__,
    "jlens_version": getattr(jlens, "__version__", None) or package_version("jlens"),
    "cuda_version": torch.version.cuda,
    "gpu": torch.cuda.get_device_name(0),
    "output_file": outname,
    "output_sha256": sha256_file(outpath),
}
with open(os.path.join(RES, ("smoke_" if args.smoke else "") + "fit_meta.json"), "w") as f:
    json.dump(meta, f, indent=2)
print(json.dumps(meta, indent=2), flush=True)
print("FIT_DONE", flush=True)
