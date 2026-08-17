#!/bin/bash
# Chain K — resume the J5 penultimate-target lens refit (checkpointed at
# 130/150 prompts by fit_ckpt_penult.pt; ~100 s/prompt => ~35 min left).
# Identical invocation to chain_h.sh's fit block (target_layer=34).
set -u
echo 1000 > /proc/self/oom_score_adj 2>/dev/null || true
if ! nvidia-smi -L > /dev/null 2>&1; then echo GPU_LOST; exit 1; fi

cd /workspace/experiments/jlens-fit-2507
/workspace/venvs/jlens/bin/python - <<'PYEOF'
import json, time, torch, transformers, jlens
from jlens.examples import load_wikitext_prompts
t0 = time.time()
hf = transformers.AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-4B-Instruct-2507", dtype=torch.bfloat16).cuda()
tok = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Instruct-2507")
model = jlens.from_hf(hf, tok)
prompts = load_wikitext_prompts(n_prompts=150)
lens = jlens.fit(model, prompts, dim_batch=128, max_seq_len=128, target_layer=34,
                 checkpoint_path="results/fit_ckpt_penult.pt", checkpoint_every=5)
finite = all(torch.isfinite(J).all().item() for J in lens.jacobians.values())
lens.save("results/Qwen3-4B-Instruct-2507_jacobian_lens_penult.pt", dtype=torch.float32)
json.dump({"target_layer": 34, "n_prompts": int(lens.n_prompts), "finite": bool(finite),
           "seconds": round(time.time()-t0, 1)},
          open("results/fit_meta_penult.json", "w"), indent=2)
print("PENULT_FIT_DONE" if finite else "PENULT_FIT_NONFINITE")
PYEOF
echo CHAIN_K_DONE
