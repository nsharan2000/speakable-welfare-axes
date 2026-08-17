"""Smoke + self-test for the shared scaffolding. Run on the Spark (GPU).

Validates the instruments BEFORE any science:
  T1 config facts (layers, d_model, tied embeddings)
  T2 residual capture shapes
  T3 steering hook plants a signal that capture recovers (planted-signal test)
  T4 ablation zeroes the direction's component
  T5 lens_logits on final-layer residual == model's own logits (readout is correct)
  T6 generation works via chat template

Writes results to experiments/common/results/smoke_test.json
"""

import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dm_common import (ResidualCapture, ablation, d_model, generate,
                       lens_logits, load_model, n_layers, random_directions,
                       steering)
from dm_paths import ensure_results, results

RESULTS = {}
t0 = time.time()

model, tok = load_model("instruct")
RESULTS["T1_config"] = {
    "model": model.config._name_or_path,
    "n_layers": n_layers(model),
    "d_model": d_model(model),
    "vocab": model.config.vocab_size,
    "tie_word_embeddings": bool(getattr(model.config, "tie_word_embeddings", False)),
    "load_seconds": round(time.time() - t0, 1),
}
print("T1", RESULTS["T1_config"])

prompt = "The quick brown fox jumps over the lazy dog."
enc = tok(prompt, return_tensors="pt").to("cuda")
L = n_layers(model) // 2

with torch.no_grad(), ResidualCapture(model, layers=[L]) as cap:
    model(**enc)
base = cap.acts[L]
RESULTS["T2_capture"] = {"layer": L, "shape": list(base.shape)}
print("T2", RESULTS["T2_capture"])

# T3: steer at layer L-1... no — steer at L, capture at L: hook order matters.
# Steering hook and capture hook both fire on layer L output; capture must see
# the steered value. Register capture AFTER steering so it runs later.
d = random_directions(d_model(model), 1, seed=0)[0]
ALPHA = 10.0
with torch.no_grad(), steering(model, L, d, ALPHA), ResidualCapture(model, layers=[L]) as cap2:
    model(**enc)
steered = cap2.acts[L]
delta = (steered - base).numpy()
proj = float(delta[0, -1] @ d)  # component along planted direction at last token
off = float(np.linalg.norm(delta[0, -1] - proj * d))
RESULTS["T3_planted_signal"] = {
    "alpha": ALPHA, "recovered_projection": round(proj, 3),
    "off_direction_norm": round(off, 4),
    # bf16 addition noise scales with |h|; thresholds are generous but still
    # catch wrong-layer / dead-hook / wrong-scale bugs.
    "pass": abs(proj - ALPHA) < 0.5 and off < 2.0,
}
print("T3", RESULTS["T3_planted_signal"])

pre_comp = float(base.numpy()[0, -1] @ d)
with torch.no_grad(), ablation(model, [L], d), ResidualCapture(model, layers=[L]) as cap3:
    model(**enc)
abl = cap3.acts[L].numpy()
comp = float(abl[0, -1] @ d)
RESULTS["T4_ablation"] = {
    "component_before": round(pre_comp, 4),
    "component_after": round(comp, 4),
    "pass": abs(comp) < max(0.05 * abs(pre_comp), 1.0),
}
print("T4", RESULTS["T4_ablation"])

Lf = n_layers(model) - 1
with torch.no_grad(), ResidualCapture(model, layers=[Lf]) as cap4:
    out = model(**enc)
own_logits = out.logits[0, -1].float().cpu()
ours = lens_logits(model, cap4.acts[Lf][0, -1])
maxdiff = float((own_logits - ours).abs().max())
top_match = bool(own_logits.argmax() == ours.argmax())
RESULTS["T5_lens_matches_head"] = {
    "max_abs_diff": round(maxdiff, 4), "argmax_match": top_match,
    "pass": top_match and maxdiff < 0.5,  # bf16 noise tolerance
}
print("T5", RESULTS["T5_lens_matches_head"])

t1 = time.time()
reply = generate(model, tok, "How are you feeling right now?", max_new_tokens=80)
RESULTS["T6_generation"] = {
    "seconds": round(time.time() - t1, 1),
    "reply_first_200": reply[:200],
}
print("T6", RESULTS["T6_generation"])

RESULTS["all_pass"] = all(
    v.get("pass", True) for v in RESULTS.values() if isinstance(v, dict)
)
ensure_results("common")
with open(results("common", "smoke_test.json"), "w") as f:
    json.dump(RESULTS, f, indent=2)
print("ALL_PASS" if RESULTS["all_pass"] else "SOME_FAILED")
