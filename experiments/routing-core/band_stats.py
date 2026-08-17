"""E3: Identify the workspace band (sensory / workspace / motor regimes) for a
given model+lens pair, replicating the paper's layer statistics:

  S1 lens top-k next-token prediction accuracy per layer (near zero early,
     jumps in final layers = motor regime)
  S2 excess kurtosis of the lens logit distribution per layer (rises at
     workspace onset)
  S3 autocorrelation of the top-1 lens token across adjacent positions vs a
     position-shuffled null (workspace = persistent contents)
  S4 effective rank of J_l (proxy for the paper's W_U J_l effective
     dimensionality — documented deviation: SVD of J_l (2560x2560) instead of
     the vocab-sized matrix)

Usage (venv python):
  python3 band_stats.py --model Qwen/Qwen3-4B \
      --lens-hub qwen3-4b/jlens/Salesforce-wikitext/Qwen3-4B_jacobian_lens.pt
  python3 band_stats.py --model Qwen/Qwen3-4B-Instruct-2507 --lens-file our-fit
      (--lens-file our-fit resolves to dm_paths.LENS_FILE; an explicit path also works)

Output: results/band_stats_<modeltag>.json
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import transformers

import jlens

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "common"))
from dm_paths import LENS_FILE  # noqa: E402

RES = os.path.join(HERE, "results")
os.makedirs(RES, exist_ok=True)

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen3-4B")
ap.add_argument("--lens-hub", default=None)
ap.add_argument("--lens-file", default=None,
                help="path to a fitted lens, or 'our-fit' for dm_paths.LENS_FILE")
ap.add_argument("--n-prompts", type=int, default=24)
args = ap.parse_args()
if args.lens_file == "our-fit":
    args.lens_file = LENS_FILE

tag = args.model.split("/")[-1]
t0 = time.time()

hf = transformers.AutoModelForCausalLM.from_pretrained(
    args.model, dtype=torch.bfloat16).cuda()
tok = transformers.AutoTokenizer.from_pretrained(args.model)
model = jlens.from_hf(hf, tok)
if args.lens_file:
    lens = jlens.JacobianLens.load(args.lens_file)
else:
    lens = jlens.JacobianLens.from_pretrained(
        "neuronpedia/jacobian-lens", filename=args.lens_hub)

from jlens.examples import load_wikitext_prompts
prompts = load_wikitext_prompts(n_prompts=args.n_prompts)

layers = lens.source_layers
SKIP = 16  # lens unfitted below position 16

acc = {l: [] for l in layers}
kurt = {l: [] for l in layers}
autoc = {l: [] for l in layers}
autoc_null = {l: [] for l in layers}
rng = np.random.default_rng(0)

for pi, p in enumerate(prompts):
    ll, ml, ids = lens.apply(model, p, max_seq_len=128)  # dict l -> [T, vocab]
    ids_t = ids[0] if ids.dim() > 1 else ids
    T = next(iter(ll.values())).shape[0]
    if T <= SKIP + 2:
        continue
    nxt = ids_t[SKIP + 1:T]  # actual next token for positions SKIP..T-2
    for l in layers:
        L = ll[l][SKIP:T - 1].float()          # [n, vocab]
        top10 = torch.topk(L, 10, dim=-1).indices
        acc[l].append(float((top10 == nxt.unsqueeze(-1).to(top10.device))
                            .any(-1).float().mean()))
        z = (L - L.mean(-1, keepdim=True)) / (L.std(-1, keepdim=True) + 1e-6)
        kurt[l].append(float((z ** 4).mean() - 3))
        t1 = L.argmax(-1)
        same = (t1[1:] == t1[:-1]).float().mean()
        autoc[l].append(float(same))
        perm = torch.as_tensor(rng.permutation(len(t1)))
        t1s = t1[perm]
        autoc_null[l].append(float((t1s[1:] == t1s[:-1]).float().mean()))
    if (pi + 1) % 8 == 0:
        print(f"{pi+1}/{len(prompts)} prompts {time.time()-t0:.0f}s", flush=True)

erank = {}
for l in layers:
    s = torch.linalg.svdvals(lens.jacobians[l].float().cuda()).cpu().numpy()
    ps = s / s.sum()
    erank[l] = float(np.exp(-(ps * np.log(ps + 1e-12)).sum()))

out = {
    "model": args.model, "n_prompts": len(prompts),
    "layers": [int(l) for l in layers],
    "acc_top10": {str(l): float(np.mean(acc[l])) for l in layers},
    "excess_kurtosis": {str(l): float(np.mean(kurt[l])) for l in layers},
    "top1_autocorr": {str(l): float(np.mean(autoc[l])) for l in layers},
    "top1_autocorr_null": {str(l): float(np.mean(autoc_null[l])) for l in layers},
    "effective_rank_Jl": {str(l): erank[l] for l in layers},
    "seconds": round(time.time() - t0, 1),
}

# band estimate. Early layers (< quarter depth) carry artifactual kurtosis
# (token-identity residue under tied embeddings), so the workspace-onset rule
# uses the autocorrelation-above-null signature restricted to layers >= L/4:
# onset = first such layer where top1 persistence > 3x its shuffled null.
# Motor onset = first layer where lens acc_top10 > 0.5.
accs = np.array([out["acc_top10"][str(l)] for l in layers])
ws_on = None
q = max(1, len(layers) // 4)
for i in range(q, len(layers)):
    l = layers[i]
    if out["top1_autocorr"][str(l)] > 3 * max(out["top1_autocorr_null"][str(l)], 1e-3):
        ws_on = int(l)
        break
motor_on = int(layers[int(np.argmax(accs > 0.5))]) if (accs > 0.5).any() else None
out["band_estimate"] = {"workspace_onset": ws_on, "motor_onset": motor_on}

with open(os.path.join(RES, f"band_stats_{tag}.json"), "w") as f:
    json.dump(out, f, indent=2)
print(json.dumps(out["band_estimate"]), flush=True)
print("BAND_STATS_DONE", tag, flush=True)
