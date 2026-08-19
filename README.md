# Is functional welfare speakable?

Read the full paper here -> https://latentmindsinstitute.com/speakable-welfare/

**When RL training installs a welfare-like state in a language model, does that
state get wired into the model's speakable channel?**

This repository contains the complete code, data, and audit trail behind the
paper *"Is functional welfare speakable?"*, submitted to Apart Research — Digital Minds
Research Sprint Hackaton, 2026. Every number in the paper traces to a results file
committed here, and one command re-checks all of them.

**TL;DR of the findings:** maze-RL-trained welfare directions in
Qwen3-4B-Instruct-2507 occupy the Jacobian-lens verbalizable subspace far
above chance (0/100 norm-matched random directions reach either pole) while
matched naive controls sit at chance. Over training the *distress* pole gains
verbalizable share while the flourishing pole only gains amplitude. The
speakable component alone causally carries the sentiment effect; and steering
toward flourishing specifically dissolves the model's "as an AI, I don't have
feelings" boilerplate (37.5% vs 95% clean, blind-judged). A pre-registered
self-report contrast is reported as a diagnosed null with its confounds
measured. No claims about subjective experience are made — all claims concern
functional coupling between activation directions and output channels

## Verify the paper's numbers (no GPU, ~5 seconds)

```bash
git clone https://github.com/nsharan2000/speakable-welfare-axes
cd speakable-welfare-axes
python3 verify_numbers.py
```

This needs nothing but the Python standard library. It recomputes / re-reads
**82 checks** covering every table and headline claim (Tables 1–3, Figures
1–5, the D-series, the retracted claims) against the raw JSON/JSONL results in
`experiments/*/results/` and ends with `ALL_CHECKS_PASS`.

## Reproduce measurements (GPU tier)

Full reproduction re-runs the decompositions, steering generations, and
judging. You need one CUDA GPU with ~12 GB free (the model is 4B,
bfloat16), plus the fitted lens artifacts:

```bash
pip install -r requirements.txt          # torch, transformers, numpy, ...
pip install -r requirements-jlens.txt    # the lens-fitting extras
python3 download_artifacts.py            # ~1.8 GB from Hugging Face
export DM_WELFARE_VECTORS="$PWD/artifacts/welfare-vectors"
python3 experiments/common/dm_paths.py   # prints resolved paths, flags anything missing
```

Then any experiment runs standalone, e.g. the paper's spine (Section 4.2):

```bash
python3 experiments/routing-core/f2_jshare_cohort.py
```

Three things to know before re-running:

- **Path resolution** is centralized in `experiments/common/dm_paths.py` —
  every external location is overridable with a `DM_*` environment variable,
  and `require()` fails early with an actionable message if an input is
  missing.
- **Re-running scripts appends/writes into `experiments/*/results/`.** The
  committed results are append-only history; scripts that would overwrite a
  primary result write to new filenames instead. If you want a pristine
  comparison, work on a branch.
- Generation-scoring experiments (steering, D2) uses **blind LLM judges**
  (Claude, via shuffled `{idx, question, response}` chunks with an independent
  second judge). The judged labels are committed, so analyses re-run without
  any API access; re-judging from scratch requires your own judge setup —
  rubrics and chunk formats are in the results folders.

## What's in the box

| Path | What it is |
|---|---|
| `paper/` | The paper PDF and a **guide edition** — same content with a plain-language explainer layer for readers new to the area |
| `verify_numbers.py` | 82 assertions: paper tables ↔ results files |
| `experiments/` | All experiment code, by family (see below), with each family's `results/` |
| `EXPERIMENTS.md` | Index: every experiment → script → results file → paper section |
| `RESEARCH-LOG.md` | Condensed day-by-day research chronology, including mistakes, corrections, and both retractions |
| `pre-registration.md` | The frozen pre-registration (sha f66b6ea3, never edited after freezing) |
| `audits/` | Compiled summaries of the two independent adversarial audit rounds; round 1 overturned one of our own headlines (§4.1), round 2 corrected round 1 and specified the D-series |
| `guides/` | Interactive experiment map (open in a browser), concept explainers, project site |
| `download_artifacts.py` | Fetches the ~1.8 GB of lens/vector binaries from Hugging Face |

## How to use the `experiments/` folder

Each subfolder is one experiment **family**, and every family has the same
shape: the scripts at its top level, and a `results/` folder holding the
JSON/JSONL outputs those scripts produced (the same files `verify_numbers.py`
checks and the paper cites in its captions).

| Family | What it measures |
|---|---|
| `jlens-fit-2507/` | Fitting the two Jacobian lenses for the model (§3) |
| `jlens-validation/`, `jlens-replication/` | Instrument gates: known-reportable direction, replication checks |
| `jlens-atlas/` | Pole-score atlas (§4.1) and the training-time trajectory (§4.4) |
| `routing-core/` | The J-share cohorts (§4.2), D-series (§4.3, §4.8), self-report channel |
| `j4-behavioral/` | Component reinjection — the causal split (§4.6) |
| `j6-crossmodel/` | Transfer to base Qwen3-4B (§4.7) |
| `welfare-axis/` | Steering validation, blind judging, naive-control re-extraction (§4.5) |
| `common/` | Shared libraries: `dm_paths.py` (all path resolution), stats, prompts |

A typical workflow:

1. **Find the experiment** in `EXPERIMENTS.md` (it maps every paper section
   to its script and results file), or follow a caption's `[filename.json]`
   into the matching family's `results/`.
2. **Read the script's docstring** — every script starts with what it
   measures, what it writes, and its cost; scripts run standalone
   (`python3 experiments/<family>/<script>.py`) after the GPU-tier setup
   above.
3. **Outputs land in that family's `results/`** under new filenames —
   committed results are never overwritten, so your rerun sits next to ours
   for diffing.

`experiments/figures/` regenerates every paper figure from the results files
(`python3 experiments/figures/fig_v2.py`). The `chain_*.sh` scripts at the
top level record the exact order and gating the experiments ran in on our
GPU box — useful as provenance and as worked examples, but not required
entry points. Files like `judge_chunks*/` inside families are the shuffled,
blinded chunks exactly as the LLM judges saw them, kept for judging
transparency.

## Artifacts on Hugging Face

`Teachafy/speakable-welfare-axes-artifacts` hosts the binaries that exceed
GitHub's limits: the two fitted Jacobian lenses for Qwen3-4B-Instruct-2507
(final-layer target + penultimate-target robustness refit, ~875 MB each) and
the trained/naive welfare vectors. `download_artifacts.py` fetches and places
them. The lenses can alternatively be re-fitted from scratch with
`experiments/jlens-fit-2507/` (~4 h on one GPU, WikiText-103 prompts, as per the official
Gurnee et al. implementation).

## Provenance and third-party components

- **Welfare vectors & maze-RL protocol**: Han, Chalmers & Izmailov (2026),
  *A Functional Welfare Axis in RL-Trained Language Models*
  (arXiv:2605.30232); vector artifacts from the `nickmahdavi/functional-welfare`
  reproduction and the `andyqhan/functional-welfare-axis` (MIT) codebase. The
  RL checkpoints for the trajectory analysis (Fig. 2) are from the same
  release — we do not rehost them.
- **Jacobian lens**: Gurnee et al. (2026), *Verbalizable Representations Form
  a Global Workspace in Language Models* (Transformer Circuits); official
  implementation (Apache-2.0), plus the public `neuronpedia` lens for base
  Qwen3-4B used in the transfer experiment.
- **Model**: `Qwen/Qwen3-4B-Instruct-2507`, frozen and unmodified; all
  interventions are inference-time activation steering.

## Reading order for newcomers

1. `paper/guide-edition.pdf` — the paper with every concept and formula
   explained from zero (start here if J-lenses are new to you)
2. `guides/experiment-map.html` — visual map of every experiment chain and
   how the results connect
3. `RESEARCH-LOG.md` — how the findings actually unfolded, including what
   broke
4. `audits/` — what independent review challenged and what changed as a result

## License

MIT (see `LICENSE`). Third-party components keep their own licenses noted
above.
