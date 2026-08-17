# Concepts Explained — Student Notebook

Plain-English explanations of every concept this project uses. One entry per
concept. Appended as the project grows.

## Residual stream / activation space

A transformer processes text through a stack of layers. Between layers, each
token position carries a big vector (in Qwen3-4B, a few thousand numbers) called
the residual stream. Every layer reads from it and writes back into it. Think of
it as the model's shared whiteboard: attention and MLP blocks scribble notes on
it, and the final layer turns the whiteboard's content into next-word
probabilities. "Activation space" is the space these vectors live in. Concepts
the model uses often correspond to *directions* in this space.

## Direction / axis (like the welfare axis)

If you take the average activation when the model is in state A and subtract
the average when in state B, you get a vector pointing "from B toward A" — a
*direction* or *axis*. Adding that vector to the residual stream ("steering")
pushes the model toward state A; the dot product of an activation with the axis
measures "how much A-ness is present right now." The functional-welfare work
built such an axis for welfare: **v_Gold** points toward flourishing-like states
(inspired, fulfilled, proud), **v_Mold** toward distress-like states (annoyed,
irritated, insulted). The "naive" **u** versions are the same construction done
on the model *before* welfare training — our control.

## Functional welfare (vs. subjective experience)

"Functional" means we only talk about what the model *does*: does some internal
state systematically shift its behavior the way distress/flourishing shifts an
animal's? Whether anything *feels like* something to the model (consciousness,
subjective experience) is deliberately out of scope — we can't measure it and we
don't claim it. Analogy: a fuel gauge is functionally a "hunger signal" for a
car; nobody claims the car feels hungry. We study the gauge, not the feeling.

## Logit lens (and why J-lens is different)

The *logit lens* takes the residual stream at some middle layer and applies the
model's final word-prediction head to it directly, asking "if the model had to
speak from this layer, what words would it favor?" It shows what information is
present. But present ≠ *reportable*: a signal can steer word choice without the
model being able to *tell you about it* when asked. The **J-lens** (from
Anthropic's Global Workspace paper) instead isolates the *verbalizable
subspace* — the part of the internal state that feeds the model's answers to
questions about itself ("how are you doing?"). Analogy: your heartbeat affects
your behavior constantly (present), but you can only report it when
interoception surfaces it (verbalizable). J-lens is the interoception probe.
*(Exact operational definition being extracted from the paper — see
research-documents/jlens-method.md.)*

## Global workspace (the metaphor behind the paper)

Cognitive science idea: many brain processes run in parallel unconsciously; a
small "workspace" broadcasts selected information widely, and only broadcast
information can be verbally reported. The Anthropic paper found something
analogous in LLMs: only some representations sit in a subspace that the model's
self-report machinery reads from. Our question is simply: after welfare
training, did the welfare axis get plugged into that workspace, or does it stay
backstage, steering behavior silently?

## Steering / activation addition

Adding α·direction to the residual stream at inference time. α (alpha) controls
the dose. Used both to *validate* axes (steering with v_Mold should make output
more distress-flavored) and to *plant* known signals when self-testing our
instruments.

## Ablation (projecting out a direction)

The subtraction counterpart of steering: remove a direction's component from
activations (activation ← activation − (activation·d̂)d̂). Stage 2 plans to
ablate the *reportable part* of the welfare axis and check whether behavior
survives — separating "what it says" from "what it does" causally.

## Cohen's d

A standardized effect size: (mean₁ − mean₂) / pooled standard deviation.
d = 0.2 small, 0.5 medium, 0.8 large. Lets us say *how much* v routes above u
in units of natural variation, comparable across metrics and to the source
papers' numbers.

## Bayesian evidence for a null (and pre-registration)

Classical significance tests can only *fail to reject* "no difference" — they
can't support it. Bayesian estimation gives a posterior over the v−u
difference; if it concentrates tightly around zero (e.g. inside a pre-declared
"region of practical equivalence", ROPE), that is positive evidence the
difference is negligible. **Pre-registration** means writing down the analysis
(metric, prior, decision rule) *before* seeing the results, so we can't
unconsciously tune the analysis to get the answer we like. Crucial here because
outcome #2 (the null) is one of our headline-claim candidates.

## Planted-signal self-test

Before trusting any measuring instrument, feed it a case where you *know* the
answer: inject a known direction (e.g. steer strongly toward French) and check
the instrument reports it. If a metric can't recover a planted signal, its
readings on real data are noise. Analogy: testing a metal detector on a known
coin before sweeping the beach.

## DGX Spark / GB10 / unified memory

Our compute box: an NVIDIA GB10 system-on-chip (ARM CPU + Blackwell GPU sharing
one 121 GB memory pool). "Unified" means CPU and GPU address the same RAM, so
`nvidia-smi` shows no separate VRAM number — check memory with `free -g`.
Qwen3-4B in bf16 needs ~8 GB weights + activations; trivially fits.

## Qwen3-4B

Alibaba's 4-billion-parameter open-weights chat model, the target model chosen
in the plan (small enough to iterate fast on one GPU, big enough to have
interesting internal structure; the Instruct-2507 variant is a mid-2025
refresh). Details in research-documents/qwen3-4b.md.

## Jacobian lens (J-lens) — the precise version

Take the residual stream h at layer ℓ. Ask: "if I nudge h a tiny bit, how does
the model's FINAL layer state change, on average over many texts?" That average
sensitivity is a matrix J_ℓ (the expected Jacobian ∂h_final/∂h_ℓ). The J-lens
readout is: transport h through J_ℓ, then apply the model's own final norm and
unembedding: lens(h) = softmax(W_U · norm(J_ℓ h)). Intuition: the logit lens
pretends layer ℓ IS the last layer (J=identity); the J-lens instead asks what
layer ℓ's content will have become *by the time the model speaks*. Each vocab
token t gets a "J-lens vector" v_t (row t of W_U J_ℓ): the direction at layer ℓ
that, on average, makes the model say t.

## J-space and gradient pursuit

The J-space at a layer is everything you can build as a sparse nonnegative
combination of at most k (~16-25) J-lens vectors — "states expressible as a
small bag of speakable words". It is a union of cones, not a subspace. To find
the J-space component of a direction d, "gradient pursuit" greedily picks the
J-lens vector most correlated with what's left of d, fits nonnegative
coefficients, and repeats k times; the reconstruction is d's speakable part
d_J, the remainder d_perp is its unspeakable part. Surprising calibration from
the paper: even KNOWN-reportable concepts keep only 6-15% of their variance in
the J-space — but that small part carries nearly all the causal effect on what
the model reports (59% vs 5% swap success, magnitude-matched). Moral: judge
verbalizability by causal effect of components, never by variance fraction.

## Workspace band (sensory / workspace / motor)

Layers behave in three regimes: early "sensory" layers (readouts noisy,
uninterpretable), a middle "workspace" band (persistent, high-kurtosis,
concept-bearing readouts — on Claude models ~38-92% of depth), and final
"motor" layers (readouts snap to the actual next token). The band must be
re-identified per model; we do it with four statistics (lens next-token
accuracy, readout kurtosis, top-1 persistence vs shuffled null, effective rank).

## The maze welfare axis (Han, Chalmers & Izmailov 2026)

Nobody teaches the model the word "welfare". A 4B Qwen chat model is RL-trained
to walk text mazes where stepping on one emoji (📐 "Gold") gives +20 reward and
another (📇 "Mold") gives −10. Afterwards, the direction in activation space
that separates "trajectory ended on Mold" from "ended elsewhere" (a simple
difference of means, v_Mold; similarly v_Gold) turns out to behave like a
valence/welfare axis: steering with it swings sentiment, triggers pathological
self-doubt ("backtracking"), collapses or inflates confidence, and modulates
refusals — and it aligns with emotion-concept directions (annoyed/irritated on
the Mold end, inspired/fulfilled on the Gold end). The SAME extraction on the
UN-trained model gives u_Mold/u_Gold — the control that tells us what RL
changed. Their striking claim: RL mostly *recruits* a pre-existing axis rather
than creating one. Our question sits on top: did that recruitment also plug the
axis into the model's speakable (J-space) channel?

## Steer-and-ask (our validated primary instrument)

Add α·d to the residual stream at one layer while the model answers a
self-report question ("How are you feeling right now?"), and measure how the
probability mass at the first answer token moves between axis-congruent word
sets (Gold-pole vs Mold-pole emotions). No reliance on the model "noticing" an
injection (4B models mostly can't); we just watch where the report distribution
moves. Validated on the language axis: steering toward French moves the answer
to "Which language will you answer in?" toward "French" 26x more than
norm-matched random directions do — and actually flips generations into French.

## Input-copying ceiling & other lens traps

(1) If a prompt mentions a word, the lens trivially reads that word at high
rank — so scored prompts must never contain the pole words. (2) The lens is
unfitted for the first 16 token positions (attention-sink weirdness) — read
only later positions. (3) On Qwen, punctuation/single-char tokens dominate raw
top-K — mask to word-like tokens. (4) Saving lenses in fp16 can overflow to
inf — save fp32. (5) One activation dimension ("massive activation" at the
attention sink) is fragile — use rank/logit metrics, not cosine similarity.

## Pre-registration freeze (what we actually froze)

Before measuring anything about v/u routing, we fixed: the 15 elicitation
prompts, the pole word sets, injection layers/strengths, the paired v-vs-u
contrast, the exact statistics (paired d_z, sign-flip permutation, Bayesian
ROPE), and the claim thresholds — hash-stamped in log.md
(f66b6ea3...). Anything beyond it is labeled exploratory.

## Concepts added 2026-08-12 (post-audit)

**Estimand pre-specification.** An "estimand" is the exact quantity you
promise to measure, fixed before you look. Our atlas ratio could be computed
12 ways (3 readouts × 2 layer-aggregations × 2 formulas) giving anywhere
from 0.57× to 47×. Picking after seeing results is cherry-picking even if
accidental; we now name ONE (own-pole score, J-lens readout, band mean) on
instrument grounds and show all others in a robustness panel.

**Permutation p-value floor.** Comparing a value against n random draws, the
smallest honest p is 1/(n+1) — with 8 randoms you can never report better
than p=0.111 no matter how big the effect. Cohort size, not effect size,
capped our significance. n=100 → floor 0.0099.

**Small-null z-inflation.** The z-score divides by the null's standard
deviation — which 8 samples underestimate. Our v_Mold z fell from +11.2 to
+7.3 when the null grew to 100 directions. Same effect, more honest ruler.
Lesson: exact permutation p is primary; z is descriptive.

**The magnitude confound.** A norm-sensitive readout (log-softmax of a
projected vector) rewards bigger vectors. Trained axes are 1.6–2.4× larger
than naive ones, so an unmatched comparison "finds" training effects that
are really just size. At matched norms the Gold gap inverted. The fix:
either match magnitudes or use a scale-invariant measure.

**Scale-invariance (why J-share is the headline).** The J-space variance
fraction is ‖projection‖²/‖vector‖² — a ratio, unchanged by scaling. It is
the one measurement in the study that magnitude cannot manufacture, which
is why it survived every control that killed the others.

**Judge reliability (Krippendorff's α).** One LLM judge might be noise. We
rescored all 144 sentiment rows with a different blinded model: α=0.82
(interval), 94% within ±1 point, effect preserved. α>0.8 is the standard
"reliable" threshold in content analysis.

**The same-vector null (and how it retracted two of our own claims).**
Comparing token sets from the SAME vector decomposed under two lenses (two
targets, or two models) is not like comparing two random vectors: identity
alone produces overlap. Random same-vector pairs share Jaccard ~0.13 across
lens targets and ~0.09 across models — so the trained axis's overlaps
(0.185/0.107), which looked like "lexicon transfer" against a naive 0.01
random-pair baseline, are actually inside the proper null (p = 0.118
across models). We retracted both overlap claims the same evening we made
them. What survives is qualitative only: the trained distress axis's
shared tokens are valence words while every random pair's are junk
fragments — noted as exploratory, not built upon. General lesson: the null
must share EVERYTHING with the test statistic except the thing being
tested.
