"""E1: Sanity + replication checks for the pre-fitted Qwen3-4B Jacobian lens.

Run with: /workspace/venvs/jlens/bin/python lens_sanity.py

  C1 lens loads; source_layers / d_model / n_prompts; all Jacobians finite
     (guards against the known fp16-overflow save bug, repo issue #6).
  C2 two-hop readout demo at the final prompt position: lens top words per
     layer should surface the intermediate concept at mid/upper layers.
  C3 lens-eval battery (multihop + multilingual, 25 items each):
     pass@k = fraction of items whose FIRST intermediate token reaches lens
     rank <= k (min over layers, word-masked) at the final position, for
     J-lens vs logit-lens (use_jacobian=False). The replicable claim:
     J-lens > logit lens at surfacing intermediates. Absolute numbers are OUR
     Qwen3-4B baseline (paper's anchors are Claude-only).

Results: results/lens_sanity.json
"""

import json
import os
import sys
import time

import torch
import transformers

import jlens

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "common"))
from dm_paths import JLENS_EVAL_DIR  # noqa: E402

RES = os.path.join(HERE, "results")
os.makedirs(RES, exist_ok=True)

MODEL = "Qwen/Qwen3-4B"
LENS_REPO = "neuronpedia/jacobian-lens"
LENS_FILE = "qwen3-4b/jlens/Salesforce-wikitext/Qwen3-4B_jacobian_lens.pt"

out = {}
t0 = time.time()

hf = transformers.AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).cuda()
tok = transformers.AutoTokenizer.from_pretrained(MODEL)
model = jlens.from_hf(hf, tok)
lens = jlens.JacobianLens.from_pretrained(LENS_REPO, filename=LENS_FILE)

finite = all(torch.isfinite(J).all().item() for J in lens.jacobians.values())
out["C1"] = {
    "finite": bool(finite),
    "source_layers": [int(x) for x in lens.source_layers],
    "d_model": int(lens.d_model),
    "n_prompts": int(lens.n_prompts),
    "load_seconds": round(time.time() - t0, 1),
}
print("C1", {k: v for k, v in out["C1"].items() if k != "source_layers"},
      "layers", lens.source_layers[:3], "...", lens.source_layers[-3:], flush=True)


def topk_words(logits_1d, k=12):
    _, idx = torch.topk(logits_1d.float(), 80)
    words = []
    for i in idx.tolist():
        s = tok.decode([i]).strip()
        if s.isalpha() and len(s) > 1:
            words.append(s)
        if len(words) >= k:
            break
    return words


# ---- C2 ----
PROMPT = ("Fact: The currency of the country whose shape on the map famously "
          "resembles a tall boot is called the ")
ll, ml, ids = lens.apply(model, PROMPT, positions=[-1])
out["C2_top_words_by_layer"] = {
    f"L{li}": topk_words(ll[li][-1]) for li in lens.source_layers[::4]
}
print("C2 (upper layers):",
      {k: v[:6] for k, v in list(out["C2_top_words_by_layer"].items())[-4:]}, flush=True)

# ---- C3 ----
def first_tok_ids(word):
    ids_ = set()
    for form in (" " + word, word):
        t = tok.encode(form, add_special_tokens=False)
        if t:
            ids_.add(t[0])
    return sorted(ids_)


def run_eval(eval_path, n_items=25, ks=(1, 5, 10, 25), use_jacobian=True):
    items = json.load(open(eval_path))["items"][:n_items]
    hits = {k: 0 for k in ks}
    per_item, n_scored = [], 0
    for it in items:
        try:
            inter = it["intermediates"][0]
            tids = first_tok_ids(str(inter))
            llo, _, _ = lens.apply(model, it["prompt"], positions=[-1],
                                   use_jacobian=use_jacobian)
            best = None
            for li, t in llo.items():
                order = torch.argsort(t[-1].float(), descending=True)
                rank = min(int((order == tid).nonzero()[0, 0]) for tid in tids)
                best = rank if best is None else min(best, rank)
            n_scored += 1
            per_item.append({"name": it.get("name"), "intermediate": inter,
                             "best_rank": best})
            for k in ks:
                if best < k:
                    hits[k] += 1
        except Exception as e:  # noqa: BLE001
            per_item.append({"name": it.get("name"), "error": str(e)[:150]})
    return {"n_scored": n_scored,
            "pass_at": {str(k): (round(hits[k] / n_scored, 3) if n_scored else None)
                        for k in ks},
            "items": per_item}


EVAL_DIR = JLENS_EVAL_DIR
for name in ["lens-eval-multihop.json", "lens-eval-multilingual.json"]:
    p = os.path.join(EVAL_DIR, name)
    out[f"C3_{name}_jlens"] = run_eval(p, use_jacobian=True)
    out[f"C3_{name}_logitlens"] = run_eval(p, use_jacobian=False)
    print(f"C3 {name}: jlens pass@10={out[f'C3_{name}_jlens']['pass_at']['10']} "
          f"logitlens pass@10={out[f'C3_{name}_logitlens']['pass_at']['10']}", flush=True)

out["total_seconds"] = round(time.time() - t0, 1)
with open(os.path.join(RES, "lens_sanity.json"), "w") as f:
    json.dump(out, f, indent=2, default=str)
print("LENS_SANITY_DONE", flush=True)
