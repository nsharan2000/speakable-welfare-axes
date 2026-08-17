"""Filesystem roots for the digital-minds experiments.

Every path an experiment script needs resolves through this module, so that a
clone of the repo runs without editing 21 files.

Two kinds of path:

* **In-repo** (scripts, results, fitted lenses we produce) are derived from this
  file's own location, so they are correct wherever the repo is checked out.
* **External artifacts** (the third-party vector repos, the upstream
  jacobian-lens checkout) default to the DGX Spark container layout under
  ``/workspace`` — the layout every committed result was produced under — and
  are overridable by environment variable.

The defaults are chosen so that on the Spark this module resolves to exactly
the paths that were previously hardcoded. Running an experiment there needs no
environment variables and produces byte-identical path behaviour.

Environment overrides
---------------------
``DM_ROOT``               repo root (default: derived from this file)
``DM_EXTERNAL_ROOT``      parent of the third-party checkouts (default ``/workspace``)
``DM_WELFARE_VECTORS``    nickmahdavi/functional-welfare artifacts checkout
``DM_FUNCTIONAL_WELFARE`` andyqhan/functional-welfare-axis checkout
``DM_JACOBIAN_LENS``      anthropics/jacobian-lens checkout
``DM_LENS_FILE``          our fitted Instruct-2507 lens (.pt)
``DM_LENS_FILE_PENULT``   the penultimate-target refit (.pt)

Usage::

    import os, sys
    HERE = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.join(os.path.dirname(HERE), "common"))
    from dm_paths import ARTIFACTS, LENS_FILE, add_experiment_paths, results
"""

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))          # <repo>/experiments/common
_DERIVED_ROOT = os.path.dirname(os.path.dirname(_HERE))     # <repo>
REPO_ROOT = os.path.abspath(os.environ.get("DM_ROOT", _DERIVED_ROOT))
EXPERIMENTS = os.path.join(REPO_ROOT, "experiments")


def _root(env_var, default):
    return os.path.abspath(os.environ.get(env_var, default))


# ---------------------------------------------------------------------------
# External checkouts (defaults match the Spark container layout)
# ---------------------------------------------------------------------------
EXTERNAL_ROOT = _root("DM_EXTERNAL_ROOT", "/workspace")

WELFARE_VECTORS = _root(
    "DM_WELFARE_VECTORS", os.path.join(EXTERNAL_ROOT, "welfare-vectors")
)
ARTIFACTS = os.path.join(WELFARE_VECTORS, "artifacts")
OWN_U = os.path.join(WELFARE_VECTORS, "own_u")

FUNCTIONAL_WELFARE = _root(
    "DM_FUNCTIONAL_WELFARE", os.path.join(EXTERNAL_ROOT, "functional-welfare-axis")
)
FW_DATASETS = os.path.join(FUNCTIONAL_WELFARE, "datasets")
FW_EVAL_PROMPTS = os.path.join(FW_DATASETS, "concept_vector_eval_prompts.json")

JACOBIAN_LENS = _root(
    "DM_JACOBIAN_LENS", os.path.join(EXTERNAL_ROOT, "jacobian-lens")
)
JLENS_EVAL_DIR = os.path.join(JACOBIAN_LENS, "data", "evaluations")


# ---------------------------------------------------------------------------
# In-repo paths
# ---------------------------------------------------------------------------
def experiment(name, *parts):
    """Path inside an experiment folder, e.g. ``experiment('routing-core')``."""
    return os.path.join(EXPERIMENTS, name, *parts)


def results(name, *parts):
    """Path inside an experiment's ``results/`` folder."""
    return os.path.join(EXPERIMENTS, name, "results", *parts)


def ensure_results(name):
    """Create and return an experiment's ``results/`` folder."""
    path = results(name)
    os.makedirs(path, exist_ok=True)
    return path


def add_experiment_paths(*names):
    """Prepend experiment folders to ``sys.path`` for cross-folder imports."""
    for name in reversed(names):
        path = experiment(name)
        if path not in sys.path:
            sys.path.insert(0, path)


# Fitted lenses. These are large .pt files, gitignored; see README for how to
# obtain them (fit_lens.py, or the Hugging Face release).
LENS_FILE = _root(
    "DM_LENS_FILE",
    results("jlens-fit-2507", "Qwen3-4B-Instruct-2507_jacobian_lens.pt"),
)
LENS_FILE_PENULT = _root(
    "DM_LENS_FILE_PENULT",
    results("jlens-fit-2507", "Qwen3-4B-Instruct-2507_jacobian_lens_penult.pt"),
)

# The validated language direction (known-reportable positive control).
LANG_DIR_FILE = experiment(
    "jlens-validation", "directions", "lang_fr_minus_en_L18.npy"
)

# Source of truth for the primary decompositions, read by J5/J6.
MECH_DECOMPOSITIONS = results("routing-core", "mech_decompositions.json")
MECH_COMPONENTS = results("routing-core", "mech_components.npz")


def describe():
    """JSON-safe record of the resolved roots, for run provenance."""
    return {
        "repo_root": REPO_ROOT,
        "experiments": EXPERIMENTS,
        "external_root": EXTERNAL_ROOT,
        "welfare_vectors": WELFARE_VECTORS,
        "functional_welfare": FUNCTIONAL_WELFARE,
        "jacobian_lens": JACOBIAN_LENS,
        "lens_file": LENS_FILE,
        "overrides_set": sorted(
            k for k in (
                "DM_ROOT", "DM_EXTERNAL_ROOT", "DM_WELFARE_VECTORS",
                "DM_FUNCTIONAL_WELFARE", "DM_JACOBIAN_LENS",
                "DM_LENS_FILE", "DM_LENS_FILE_PENULT",
            ) if k in os.environ
        ),
    }


def require(*paths):
    """Fail early, with an actionable message, if an input is missing."""
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            "Missing required input(s):\n  "
            + "\n  ".join(missing)
            + "\n\nResolved roots: "
            + repr(describe())
            + "\nSet the DM_* environment variables (see experiments/common/"
              "dm_paths.py) if your checkout does not use the Spark layout."
        )
    return True


if __name__ == "__main__":
    import json
    print(json.dumps(describe(), indent=2))
    for label, path in [
        ("ARTIFACTS", ARTIFACTS), ("FW_DATASETS", FW_DATASETS),
        ("JLENS_EVAL_DIR", JLENS_EVAL_DIR), ("LENS_FILE", LENS_FILE),
        ("LANG_DIR_FILE", LANG_DIR_FILE),
    ]:
        print(f"{'OK  ' if os.path.exists(path) else 'MISS'} {label}: {path}")
