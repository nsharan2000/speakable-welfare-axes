"""Convert our own-u extraction (mean_diff.pt per concept) into the
{v_gold, v_mold, layer_gold, layer_mold} dict format, using the same treatment
layers as the nickmahdavi vectors (21/24) for comparability."""
import glob
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "common"))
from dm_paths import ARTIFACTS, OWN_U  # noqa: E402

out = {}
for code, paper in [("goal", "gold"), ("lava", "mold")]:
    cand = [c for c in glob.glob(os.path.join(OWN_U, "**", "mean_diff.pt"),
                                 recursive=True) if code in c]
    assert cand, f"no mean_diff for {code}"
    t = torch.load(cand[0], map_location="cpu", weights_only=False).squeeze(0).float()
    out[f"v_{paper}"] = t
out["layer_gold"] = 21
out["layer_mold"] = 24
out["note"] = "own extraction, Han et al. extract_dispatch off_policy n2000, layers fixed to 21/24 for comparability"
torch.save(out, os.path.join(ARTIFACTS, "vectors_own_naive.pt"))
print("CONVERTED own_u ->", {k: (tuple(v.shape) if hasattr(v, 'shape') else v) for k, v in out.items()})
