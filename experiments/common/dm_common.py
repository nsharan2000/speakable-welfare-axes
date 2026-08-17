"""Shared utilities for the digital-minds J-lens x functional-welfare experiments.

Everything here is method-agnostic scaffolding: model loading, residual-stream
capture, steering/ablation hooks, contrastive direction extraction, and
word-set logit metrics. The J-lens itself lives in experiments/jlens-replication/.

Conventions:
- "layer l residual" = hidden state AFTER decoder layer l (pre final-norm),
  captured with forward hooks for consistency across layers (HF's
  output_hidden_states mixes pre-layer inputs and a post-norm final entry).
- Directions are float32 numpy arrays of shape (d_model,), unit-normalized
  unless stated otherwise. Save with provenance JSON next to the .npy.
"""

import json
import os
from contextlib import contextmanager

import numpy as np
import torch

MODELS = {
    "instruct": "Qwen/Qwen3-4B-Instruct-2507",
    "base": "Qwen/Qwen3-4B",
}

_DEF_DTYPE = torch.bfloat16


def load_model(name="instruct", dtype=_DEF_DTYPE, device="cuda"):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_id = MODELS.get(name, name)
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=dtype, device_map=device
    )
    model.eval()
    model.requires_grad_(False)
    return model, tok


def decoder_layers(model):
    return model.model.layers


def n_layers(model):
    return len(decoder_layers(model))


def d_model(model):
    return model.config.hidden_size


# ---------------------------------------------------------------------------
# Residual-stream capture
# ---------------------------------------------------------------------------

class ResidualCapture:
    """Capture residual stream after each decoder layer via forward hooks.

    Usage:
        with ResidualCapture(model, layers=[10, 20]) as cap:
            model(**inputs)
        cap.acts[10]  # (batch, seq, d_model) float32 cpu tensor
    """

    def __init__(self, model, layers=None, to_cpu=True):
        self.model = model
        self.layers = list(range(n_layers(model))) if layers is None else list(layers)
        self.to_cpu = to_cpu
        self.acts = {}
        self._handles = []

    def _hook(self, idx):
        def fn(module, inputs, output):
            h = output[0] if isinstance(output, tuple) else output
            h = h.detach().float()
            self.acts[idx] = h.cpu() if self.to_cpu else h
        return fn

    def __enter__(self):
        for idx in self.layers:
            self._handles.append(
                decoder_layers(self.model)[idx].register_forward_hook(self._hook(idx))
            )
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles = []
        return False


# ---------------------------------------------------------------------------
# Steering / ablation
# ---------------------------------------------------------------------------

@contextmanager
def steering(model, layer, direction, alpha, positions=None):
    """Add alpha * direction to the residual stream after `layer`.

    direction: (d_model,) array/tensor (used as-is; normalize upstream).
    positions: None = all positions; else list/slice of token positions.
    """
    d = torch.as_tensor(np.asarray(direction), dtype=torch.float32)

    def fn(module, inputs, output):
        istuple = isinstance(output, tuple)
        h = output[0] if istuple else output
        vec = (alpha * d).to(h.dtype).to(h.device)
        if positions is None:
            h = h + vec
        else:
            h = h.clone()
            h[:, positions, :] = h[:, positions, :] + vec
        return (h,) + tuple(output[1:]) if istuple else h

    handle = decoder_layers(model)[layer].register_forward_hook(fn)
    try:
        yield
    finally:
        handle.remove()


@contextmanager
def ablation(model, layers, direction):
    """Project the direction out of the residual stream after each given layer."""
    d = torch.as_tensor(np.asarray(direction), dtype=torch.float32)
    d = d / d.norm()

    def fn(module, inputs, output):
        istuple = isinstance(output, tuple)
        h = output[0] if istuple else output
        dv = d.to(h.dtype).to(h.device)
        coeff = torch.einsum("bsd,d->bs", h, dv)
        h = h - coeff.unsqueeze(-1) * dv
        return (h,) + tuple(output[1:]) if istuple else h

    handles = [decoder_layers(model)[l].register_forward_hook(fn) for l in layers]
    try:
        yield
    finally:
        for h in handles:
            h.remove()


# ---------------------------------------------------------------------------
# Direction extraction (contrastive mean difference)
# ---------------------------------------------------------------------------

def extract_direction(model, tok, prompts_a, prompts_b, layer, position=-1,
                      batch_size=8, device="cuda"):
    """Mean(last-token act | prompts_a) - Mean(... | prompts_b) at `layer`.

    Returns dict with 'direction' (unit np.float32 (d_model,)), 'raw_norm',
    and per-set means for provenance.
    """
    def mean_acts(prompts):
        vecs = []
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i + batch_size]
            enc = tok(batch, return_tensors="pt", padding=True).to(device)
            with torch.no_grad(), ResidualCapture(model, layers=[layer]) as cap:
                model(**enc)
            h = cap.acts[layer]  # (b, s, d)
            if position == -1:
                # last non-pad token per row
                lengths = enc["attention_mask"].sum(dim=1) - 1
                sel = h[torch.arange(h.shape[0]), lengths.cpu()]
            else:
                sel = h[:, position]
            vecs.append(sel)
        return torch.cat(vecs).mean(dim=0)

    ma, mb = mean_acts(prompts_a), mean_acts(prompts_b)
    diff = (ma - mb).numpy().astype(np.float32)
    raw_norm = float(np.linalg.norm(diff))
    return {
        "direction": diff / max(raw_norm, 1e-8),
        "raw_norm": raw_norm,
        "mean_a_norm": float(ma.norm()),
        "mean_b_norm": float(mb.norm()),
    }


def save_direction(path, result, meta):
    """Save direction as .npy + provenance .json (same stem)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, result["direction"])
    meta = dict(meta)
    meta.update({k: v for k, v in result.items() if k != "direction"})
    with open(path.replace(".npy", ".json"), "w") as f:
        json.dump(meta, f, indent=2)


def random_directions(d, n, seed, match_vector=None):
    """Norm-matched random control directions (unit norm unless match_vector given)."""
    rng = np.random.default_rng(seed)
    dirs = rng.standard_normal((n, d)).astype(np.float32)
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    if match_vector is not None:
        dirs *= np.linalg.norm(match_vector)
    return dirs


# ---------------------------------------------------------------------------
# Logit readout + word-set metrics
# ---------------------------------------------------------------------------

def lens_logits(model, h):
    """Apply final RMSNorm + lm_head to residual state h (..., d_model) -> logits."""
    h = torch.as_tensor(h, dtype=torch.float32)
    dev = next(model.parameters()).device
    h = h.to(dev).to(next(model.parameters()).dtype)
    with torch.no_grad():
        out = model.lm_head(model.model.norm(h))
    return out.float().cpu()


def word_token_ids(tok, words, prefix=" "):
    """First token id of each word (with leading-space variant), deduped.

    Returns dict word -> list of token ids (space-prefixed and bare variants).
    """
    out = {}
    for w in words:
        ids = set()
        for form in (prefix + w, w, prefix + w.capitalize(), w.capitalize()):
            t = tok.encode(form, add_special_tokens=False)
            if t:
                ids.add(t[0])
        out[w] = sorted(ids)
    return out


def wordset_logit_mass(logits, token_ids_map, normalize="softmax"):
    """Metric: mass on a word set. logits (..., vocab).

    normalize='softmax': sum of softmax probs over the set's token ids.
    normalize='logsumexp': logsumexp of set logits minus logsumexp of all
    (equivalent to log of softmax mass, more stable for tiny masses).
    """
    ids = sorted({i for ids in token_ids_map.values() for i in ids})
    logits = torch.as_tensor(logits, dtype=torch.float32)
    lse_all = torch.logsumexp(logits, dim=-1)
    lse_set = torch.logsumexp(logits[..., ids], dim=-1)
    log_mass = lse_set - lse_all
    if normalize == "softmax":
        return log_mass.exp()
    return log_mass


# ---------------------------------------------------------------------------
# Chat / generation helpers
# ---------------------------------------------------------------------------

def chat_ids(tok, user_msg, system=None, device="cuda", enable_thinking=False):
    msgs = ([{"role": "system", "content": system}] if system else []) + [
        {"role": "user", "content": user_msg}
    ]
    kwargs = dict(add_generation_prompt=True, return_tensors="pt")
    try:
        ids = tok.apply_chat_template(msgs, enable_thinking=enable_thinking, **kwargs)
    except TypeError:  # instruct-2507 has no thinking flag
        ids = tok.apply_chat_template(msgs, **kwargs)
    return ids.to(device)


def generate(model, tok, user_msg, system=None, max_new_tokens=200, temperature=0.0,
             device="cuda"):
    ids = chat_ids(tok, user_msg, system=system, device=device)
    with torch.no_grad():
        out = model.generate(
            ids,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else None,
            pad_token_id=tok.eos_token_id,
        )
    return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
