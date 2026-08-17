# Is functional welfare speakable?

**When RL training installs a welfare-like state in a language model, does that
state get wired into the model's speakable channel?**

This repository contains the complete code, data, and audit trail behind the
paper *"Is functional welfare speakable?"* (Apart Research — Digital Minds
Research Sprint, 2026). Every number in the paper traces to a results file
committed here, and one command re-checks all of them.

**TL;DR of the findings:** maze-RL-trained welfare directions in
Qwen3-4B-Instruct-2507 occupy the Jacobian-lens verbalizable subspace far
above chance (0/100 norm-matched random directions reach either pole) while
matched naive controls sit at chance; over training the *distress* pole gains
verbalizable share while the flourishing pole only gains amplitude; the
speakable component alone causally carries the sentiment effect; and steering
toward flourishing specifically dissolves the model's "as an AI, I don't have
feelings" boilerplate (37.5% vs 95% clean, blind-judged). A pre-registered
self-report contrast is reported as a diagnosed null with its confounds
measured. No claims about subjective experience are made — all claims concern
functional coupling between activation directions and output channels.

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

Then any experiment runs standalone, e.g. the paper's spine (§4.2):

```bash
python3 experiments/routing-core/f2_jshare_cohort.py
```

Two things to know before re-running:

- **Path resolution** is centralized in `experiments/common/dm_paths.py` —
  every external location is overridable with a `DM_*` environment variable,
  and `require()` fails early with an actionable message if an input is
  missing.
- **Re-running scripts appends/writes into `experiments/*/results/`.** The
  committed results are append-only history; scripts that would overwrite a
  primary result write to new filenames instead. If you want a pristine
  comparison, work on a branch.
- Generation-scoring experiments (steering, D2) used **blind LLM judges**
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
| `audits/` | Two independent adversarial audit rounds + our responses; round 1 overturned one of our own headlines (§4.1) |
| `guides/` | Interactive experiment map (open in a browser), concept explainers, project site |
| `download_artifacts.py` | Fetches the ~1.8 GB of lens/vector binaries from Hugging Face |

Experiment families under `experiments/`: `jlens-fit-2507` (lens fitting),
`jlens-validation` + `jlens-replication` (instrument gates), `jlens-atlas`
(pole-score atlas + training trajectory), `routing-core` (J-share cohorts,
D-series, self-report channel), `j4-behavioral` (component reinjection),
`j6-crossmodel` (transfer), `welfare-axis` (steering validation, naive-control
re-extraction), `common` (shared stats/paths/prompts). The `chain_*.sh`
scripts record the exact execution order used on our GPU box — provenance,
not required entry points.

## Artifacts on Hugging Face

`Teachafy/speakable-welfare-axes-artifacts` hosts the binaries that exceed
GitHub's limits: the two fitted Jacobian lenses for Qwen3-4B-Instruct-2507
(final-layer target + penultimate-target robustness refit, ~875 MB each) and
the trained/naive welfare vectors. `download_artifacts.py` fetches and places
them. The lenses can alternatively be re-fitted from scratch with
`experiments/jlens-fit-2507/` (~4 h on one GPU, WikiText-103 prompts, official
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
