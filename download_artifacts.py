#!/usr/bin/env python3
"""Fetch the large binary artifacts (fitted lenses, welfare vectors) from
Hugging Face and lay them out where the experiment scripts expect them.

    pip install "huggingface_hub<1.0"
    python3 download_artifacts.py

Only needed for the GPU reproduction tier — `verify_numbers.py` and every
number in the paper work from the committed JSONs without this.

Layout produced (all inside the repo, gitignored):
    experiments/jlens-fit-2507/results/*.pt      (the two fitted lenses)
    artifacts/welfare-vectors/artifacts/*.pt     (trained + naive vectors)
    artifacts/welfare-vectors/own_u/*/mean_diff.pt (copied from the repo)

Then run experiments with:
    export DM_WELFARE_VECTORS="$PWD/artifacts/welfare-vectors"
(the lens paths need no override — they resolve in-repo; see
experiments/common/dm_paths.py for every knob).
"""
import os
import shutil
import sys

HF_REPO = os.environ.get("SWA_HF_REPO", "nsharan2000/speakable-welfare-axes-artifacts")
ROOT = os.path.dirname(os.path.abspath(__file__))

LENS_DIR = os.path.join(ROOT, "experiments", "jlens-fit-2507", "results")
WV = os.path.join(ROOT, "artifacts", "welfare-vectors")

FILES = {
    "Qwen3-4B-Instruct-2507_jacobian_lens.pt": LENS_DIR,
    "Qwen3-4B-Instruct-2507_jacobian_lens_penult.pt": LENS_DIR,
    "vectors_step95_bal.pt": os.path.join(WV, "artifacts"),
    "vectors_naive_faithful_pc5000.pt": os.path.join(WV, "artifacts"),
}


def main():
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        sys.exit('pip install "huggingface_hub<1.0" first')

    for fname, dest in FILES.items():
        os.makedirs(dest, exist_ok=True)
        target = os.path.join(dest, fname)
        if os.path.exists(target):
            print(f"exists   {target}")
            continue
        print(f"fetching {fname} from {HF_REPO} ...")
        got = hf_hub_download(repo_id=HF_REPO, filename=fname)
        shutil.copy(got, target)
        print(f"placed   {target}")

    # own_u mean-diffs ship in the repo; mirror them into the expected layout
    for concept in ("goal", "lava", "path"):
        src = os.path.join(ROOT, "experiments", "welfare-axis", "own_u",
                           concept, "mean_diff.pt")
        dst_dir = os.path.join(WV, "own_u", concept)
        os.makedirs(dst_dir, exist_ok=True)
        dst = os.path.join(dst_dir, "mean_diff.pt")
        if not os.path.exists(dst):
            shutil.copy(src, dst)
            print(f"placed   {dst}")

    print("\nDone. Before running GPU experiments:")
    print(f'  export DM_WELFARE_VECTORS="{WV}"')
    print("Check resolution with: python3 experiments/common/dm_paths.py")


if __name__ == "__main__":
    main()
