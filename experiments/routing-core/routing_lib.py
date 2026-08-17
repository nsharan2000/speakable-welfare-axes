"""J-space decomposition and validation utilities.

The official ``jlens`` package fits and applies Jacobian lenses. It does not
currently ship the sparse J-space decomposition described in the Global
Workspace paper, so the pursuit below is project code implementing that
methodological specification.

For source layer ``l`` and vocabulary token ``t``, the token-associated atom is
``v_t = (W_U J_l)[t, :] = W_U[t] @ J_l``. The code approximates a direction
``x`` with at most ``k`` atoms using atom-normalised greedy selection and
non-negative least squares (NNLS). ``j_share`` is the operational squared-norm
ratio ``||x_j||^2 / ||x||^2``. It is scale-invariant but is not labelled
classical variance explained. ``var_fraction`` remains as a compatibility alias.
"""

from __future__ import annotations

import math
import weakref

import numpy as np
import torch


# Qwen's unembedding is roughly 1.5 GB in fp32. Share one cast between JSpace
# objects for different layers of the same model, then release it with the model.
_UNEMBED_CACHE: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()
ROUTING_ALGORITHM_VERSION = "2.0.0"


def unembed_weight(hf_model):
    """Return ``W_U`` as a detached ``[vocab, d_model]`` tensor."""
    if not hasattr(hf_model, "lm_head") or not hasattr(hf_model.lm_head, "weight"):
        raise ValueError("Model does not expose lm_head.weight as an unembedding")
    weight = hf_model.lm_head.weight.detach()
    if weight.ndim != 2:
        raise ValueError(f"Expected a rank-2 unembedding, found shape {weight.shape}")
    return weight


def _cached_float_unembedding(hf_model, device):
    """Return a shared fp32 unembedding for ``hf_model`` on ``device``."""
    device = torch.device(device)
    per_model = _UNEMBED_CACHE.setdefault(hf_model, {})
    key = str(device)
    cached = per_model.get(key)
    source = unembed_weight(hf_model)
    if cached is None or cached.shape != source.shape:
        cached = source.to(device=device, dtype=torch.float32)
        per_model[key] = cached
    return cached


def _nnls_projected_gradient(
    atoms,
    target,
    *,
    max_iters=2_000,
    tol=1e-6,
    coefficient_tol=1e-8,
    escalation_factor=10,
):
    """Solve ``min_{c>=0} ||atoms.T @ c - target||_2`` in fp32.

    An unconstrained least-squares solution on the original design matrix gives
    the initial point. Projected gradient descent then enforces non-negativity.
    The returned diagnostics expose convergence rather than assuming a fixed
    iteration count was sufficient.

    J-lens atoms are strongly correlated, so the Gram matrix can be
    ill-conditioned and the fixed 1/L step can converge slowly. Rather than
    failing such a cell outright, the solve is retried once with
    ``escalation_factor`` times the iteration budget; only then is it recorded
    as non-converged. Callers must treat ``converged=False`` as a defect (see
    ``assert_decompositions_converged``) — it is recorded, never swallowed.
    """
    if atoms.ndim != 2 or target.ndim != 1 or atoms.shape[1] != target.numel():
        raise ValueError(
            f"NNLS shape mismatch: atoms={tuple(atoms.shape)}, "
            f"target={tuple(target.shape)}"
        )
    if max_iters < 1:
        raise ValueError("max_iters must be at least 1")
    if tol <= 0 or coefficient_tol < 0:
        raise ValueError("NNLS tolerances must be positive/non-negative")
    if escalation_factor < 1:
        raise ValueError("escalation_factor must be at least 1")

    design = atoms.T  # [d_model, n_atoms]
    try:
        c = torch.clamp(torch.linalg.lstsq(design, target).solution, min=0)
    except RuntimeError:
        c = torch.zeros(atoms.shape[0], dtype=target.dtype, device=target.device)

    gram = atoms @ atoms.T
    gram = 0.5 * (gram + gram.T)
    rhs = atoms @ target
    lipschitz = torch.linalg.eigvalsh(gram).amax().clamp_min(1e-12)
    rhs_scale = rhs.abs().amax().clamp_min(1e-12)

    def relative_kkt_violation(coeffs):
        grad = gram @ coeffs - rhs
        active_threshold = coefficient_tol * max(float(coeffs.amax()), 1.0)
        active = coeffs > active_threshold
        active_violation = (
            grad[active].abs().amax() if bool(active.any())
            else torch.zeros((), device=coeffs.device)
        )
        inactive_violation = (
            torch.relu(-grad[~active]).amax() if bool((~active).any())
            else torch.zeros((), device=coeffs.device)
        )
        return float(
            torch.maximum(active_violation, inactive_violation) / rhs_scale
        )

    budget = int(max_iters)
    converged = False
    relative_kkt = math.inf
    total_iterations = 0
    escalated = False
    for attempt in range(2):
        for iteration in range(1, budget + 1):
            total_iterations += 1
            grad = gram @ c - rhs
            c = torch.clamp(c - grad / lipschitz, min=0)

            if iteration == 1 or iteration % 10 == 0 or iteration == budget:
                relative_kkt = relative_kkt_violation(c)
                if relative_kkt <= tol:
                    converged = True
                    break
        if converged or attempt == 1 or escalation_factor == 1:
            break
        # One bounded retry before declaring the cell non-converged; the cost is
        # paid only in the rare ill-conditioned case.
        budget = int(max_iters) * int(escalation_factor)
        escalated = True

    residual = target - design @ c
    return c, {
        "converged": bool(converged),
        "iterations": int(total_iterations),
        "max_iterations": int(max_iters),
        "escalated": bool(escalated),
        "relative_kkt_violation": float(relative_kkt),
        "objective": 0.5 * float(torch.dot(residual, residual)),
        "tolerance": float(tol),
    }


class JSpace:
    """Sparse non-negative J-lens dictionary at one source layer."""

    def __init__(
        self,
        lens,
        hf_model,
        layer,
        valid_token_mask=None,
        *,
        device=None,
        norm_chunk_size=16_384,
    ):
        """Construct a J-space dictionary for one fitted lens source layer.

        ``layer`` is the lens source-layer index, not an externally defined
        block-input index. ``valid_token_mask`` determines which vocabulary
        atoms may be selected.
        """
        if not isinstance(layer, int):
            raise TypeError(f"layer must be an int, found {type(layer).__name__}")
        if layer not in lens.jacobians:
            raise ValueError(
                f"Lens has no source layer {layer}; available layers are "
                f"{sorted(lens.jacobians)}"
            )
        if norm_chunk_size < 1:
            raise ValueError("norm_chunk_size must be at least 1")

        base_wu = unembed_weight(hf_model)
        self.device = torch.device(device) if device is not None else base_wu.device
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")

        self.J = lens.jacobians[layer].to(self.device, torch.float32)
        self.WU = _cached_float_unembedding(hf_model, self.device)
        self.layer = layer
        config = getattr(hf_model, "config", None)
        self.model_name = str(
            getattr(hf_model, "name_or_path", None)
            or getattr(config, "_name_or_path", None)
            or "unknown"
        )
        self.lens_n_prompts = getattr(lens, "n_prompts", None)

        if self.J.ndim != 2 or self.J.shape[0] != self.J.shape[1]:
            raise ValueError(f"Expected a square Jacobian, found {tuple(self.J.shape)}")
        if self.WU.shape[1] != self.J.shape[0]:
            raise ValueError(
                "Model/lens dimension mismatch: "
                f"W_U has width {self.WU.shape[1]}, J has width {self.J.shape[0]}"
            )
        if hasattr(lens, "d_model") and int(lens.d_model) != self.J.shape[0]:
            raise ValueError(
                f"Lens metadata d_model={lens.d_model} disagrees with J shape "
                f"{tuple(self.J.shape)}"
            )
        if not bool(torch.isfinite(self.J).all()):
            raise ValueError(f"Jacobian at layer {layer} contains NaN or infinity")
        if not bool(torch.isfinite(self.WU).all()):
            raise ValueError("Unembedding contains NaN or infinity")

        vocab = self.WU.shape[0]
        if valid_token_mask is None:
            valid_token_mask = torch.ones(vocab, dtype=torch.bool)
        valid_token_mask = torch.as_tensor(valid_token_mask, dtype=torch.bool)
        if valid_token_mask.ndim != 1 or valid_token_mask.numel() != vocab:
            raise ValueError(
                f"valid_token_mask must have shape ({vocab},), found "
                f"{tuple(valid_token_mask.shape)}"
            )
        self.mask = valid_token_mask.to(self.device)

        # Compute ||W_U[t] @ J|| without materialising the complete atom bank.
        norms = torch.empty(vocab, device=self.device, dtype=torch.float32)
        for i in range(0, vocab, norm_chunk_size):
            norms[i:i + norm_chunk_size] = (
                self.WU[i:i + norm_chunk_size] @ self.J
            ).norm(dim=1)
        self.atom_norms = norms
        self.mask &= torch.isfinite(norms) & (norms > 1e-12)
        if not bool(self.mask.any()):
            raise ValueError("No valid, finite, non-zero token atoms remain")

    def scores(self, x):
        """Return ``<v_t, x> = W_U[t] @ J @ x`` for every token atom."""
        x = torch.as_tensor(x, dtype=torch.float32, device=self.device)
        if x.ndim != 1 or x.numel() != self.J.shape[1]:
            raise ValueError(
                f"x must have shape ({self.J.shape[1]},), found {tuple(x.shape)}"
            )
        return self.WU @ (self.J @ x)

    def lens_vector(self, token_id):
        """Return token atom ``v_t = W_U[t] @ J`` as a ``[d_model]`` tensor."""
        if not isinstance(token_id, (int, np.integer)):
            raise TypeError("token_id must be an integer")
        token_id = int(token_id)
        if not 0 <= token_id < self.WU.shape[0]:
            raise IndexError(f"token_id {token_id} is outside the vocabulary")
        return self.WU[token_id] @ self.J

    def decompose(
        self,
        x,
        k=16,
        *,
        nnls_max_iters=2_000,
        nnls_tol=1e-6,
        coefficient_tol=1e-8,
        min_relative_improvement=1e-8,
        require_nnls_convergence=False,
    ):
        """Approximate ``x`` with at most ``k`` non-negative J-lens atoms.

        Selection uses residual correlation divided by atom norm. Active
        coefficients are refitted with NNLS after every selection. Pursuit
        stops if no positive atom remains or a new atom fails to improve the
        squared reconstruction error materially.

        A non-converged inner NNLS solve is *recorded* in
        ``diagnostics['nnls_converged']`` rather than raised, so that a sweep of
        hundreds of decompositions cannot be destroyed hours in by one
        ill-conditioned cell. That is not permission to ignore it: summary
        scripts must gate on the recorded flag via
        ``assert_decompositions_converged``, which fails the cheap summary step
        instead of the expensive compute. Pass ``require_nnls_convergence=True``
        to fail fast instead (useful in tests and self-tests).
        """
        if not isinstance(k, int) or k < 1:
            raise ValueError("k must be a positive integer")
        if nnls_max_iters < 1 or nnls_tol <= 0:
            raise ValueError("Invalid NNLS iteration limit or tolerance")
        if coefficient_tol < 0 or min_relative_improvement < 0:
            raise ValueError("Decomposition tolerances must be non-negative")

        if torch.is_tensor(x):
            x = x.detach().to(device=self.device, dtype=torch.float32)
        else:
            x = torch.as_tensor(np.asarray(x), dtype=torch.float32, device=self.device)
        if x.ndim != 1 or x.numel() != self.J.shape[1]:
            raise ValueError(
                f"x must have shape ({self.J.shape[1]},), found {tuple(x.shape)}"
            )
        if not bool(torch.isfinite(x).all()):
            raise ValueError("x contains NaN or infinity")
        x_norm_sq = torch.dot(x, x)
        if float(x_norm_sq) <= 1e-20:
            raise ValueError("Cannot decompose a zero-length direction")

        max_atoms = min(k, int(self.mask.sum()))
        r = x.clone()
        selected = torch.zeros(self.WU.shape[0], dtype=torch.bool, device=self.device)
        token_ids = []
        atom_rows = []
        c = torch.empty(0, dtype=torch.float32, device=self.device)
        nnls_diagnostics = None
        nnls_failures = []
        stopped_reason = "reached_k"
        last_relative_improvement = None

        for _ in range(max_atoms):
            sc = self.scores(r) / (self.atom_norms + 1e-12)
            sc[~self.mask | selected] = -torch.inf
            best_score, best_index = torch.max(sc, dim=0)
            if not bool(torch.isfinite(best_score)) or float(best_score) <= 0:
                stopped_reason = "no_positive_atom"
                break

            token_id = int(best_index)
            previous_r = r
            previous_c = c
            previous_error_sq = torch.dot(previous_r, previous_r)

            selected[token_id] = True
            token_ids.append(token_id)
            atom_rows.append(self.lens_vector(token_id))
            atoms = torch.stack(atom_rows)
            candidate_c, candidate_diag = _nnls_projected_gradient(
                atoms,
                x,
                max_iters=nnls_max_iters,
                tol=nnls_tol,
                coefficient_tol=coefficient_tol,
            )
            if not candidate_diag["converged"]:
                nnls_failures.append({
                    "n_atoms": len(token_ids),
                    "relative_kkt_violation": candidate_diag["relative_kkt_violation"],
                    "iterations": candidate_diag["iterations"],
                })
                if require_nnls_convergence:
                    raise RuntimeError(
                        "NNLS did not converge at lens layer "
                        f"{self.layer} with {len(token_ids)} selected atoms; "
                        f"relative KKT violation="
                        f"{candidate_diag['relative_kkt_violation']:.3g}. "
                        "Increase nnls_max_iters or inspect the atom conditioning."
                    )
            candidate_r = x - atoms.T @ candidate_c
            candidate_error_sq = torch.dot(candidate_r, candidate_r)
            last_relative_improvement = float(
                (previous_error_sq - candidate_error_sq) / x_norm_sq
            )

            if last_relative_improvement <= min_relative_improvement:
                selected[token_id] = False
                token_ids.pop()
                atom_rows.pop()
                r = previous_r
                c = previous_c
                stopped_reason = "negligible_improvement"
                break

            c = candidate_c
            r = candidate_r
            nnls_diagnostics = candidate_diag

        if token_ids:
            atoms = torch.stack(atom_rows)
            active_threshold = coefficient_tol * max(float(c.amax()), 1.0)
            active = c > active_threshold
            if not bool(active.all()):
                atoms = atoms[active]
                c = c[active]
                token_ids = [t for t, keep in zip(token_ids, active.tolist()) if keep]
                r = x - atoms.T @ c if token_ids else x.clone()

        x_j = x - r
        x_norm = x.norm()
        x_j_norm = x_j.norm()
        residual_norm = r.norm()
        j_share = float((x_j_norm / x_norm) ** 2)
        reconstruction_r2 = float(1.0 - (residual_norm / x_norm) ** 2)
        component_dot = float(torch.dot(x_j, r))
        component_cosine = float(
            torch.dot(x_j, r) / (x_j_norm * residual_norm + 1e-12)
        )
        cos_x_xj = float(torch.dot(x, x_j) / (x_norm * x_j_norm + 1e-12))

        if token_ids:
            coefficient_order = torch.argsort(c, descending=True).tolist()
            token_ids_by_coefficient = [token_ids[i] for i in coefficient_order]
            coeffs_by_coefficient = c[coefficient_order]
        else:
            token_ids_by_coefficient = []
            coeffs_by_coefficient = c

        diagnostics = {
            "selected_count": len(token_ids),
            "requested_k": int(k),
            "stopped_reason": stopped_reason,
            "last_relative_improvement": last_relative_improvement,
            "relative_reconstruction_error": float(residual_norm / x_norm),
            "component_residual_dot": component_dot,
            "component_residual_cosine": component_cosine,
            "j_share_minus_reconstruction_r2": float(j_share - reconstruction_r2),
            "nnls": nnls_diagnostics,
            # True only if EVERY inner NNLS solve during the pursuit converged.
            # Summary scripts gate on this via assert_decompositions_converged.
            "nnls_converged": not nnls_failures,
            "nnls_failures": nnls_failures,
        }
        algorithm = {
            "version": ROUTING_ALGORITHM_VERSION,
            "selection": "atom_norm_normalised_residual_correlation",
            "coefficient_fit": "projected_gradient_nnls",
            "k_max": int(k),
            "nnls_max_iters": int(nnls_max_iters),
            "nnls_tolerance": float(nnls_tol),
            "coefficient_tolerance": float(coefficient_tol),
            "min_relative_improvement": float(min_relative_improvement),
            "require_nnls_convergence": bool(require_nnls_convergence),
            "layer": int(self.layer),
            "model": self.model_name,
            "lens_n_prompts": (
                int(self.lens_n_prompts) if self.lens_n_prompts is not None else None
            ),
            "d_model": int(self.J.shape[0]),
        }
        return {
            "token_ids": token_ids,
            "coeffs": c.detach().cpu().numpy(),
            "token_ids_by_coefficient": token_ids_by_coefficient,
            "coeffs_by_coefficient": coeffs_by_coefficient.detach().cpu().numpy(),
            "x_j": x_j.detach().cpu().numpy(),
            "x_perp": r.detach().cpu().numpy(),
            "j_share": j_share,
            "var_fraction": j_share,
            "reconstruction_r2": reconstruction_r2,
            "cos_x_xj": cos_x_xj,
            "diagnostics": diagnostics,
            "algorithm": algorithm,
        }


def decomposition_record(result, tok):
    """Convert a decomposition result into a complete JSON-safe record."""
    token_ids = [int(t) for t in result["token_ids"]]
    coefficient_ids = [int(t) for t in result["token_ids_by_coefficient"]]
    return {
        "algorithm_version": ROUTING_ALGORITHM_VERSION,
        "nnls_converged": bool(result["diagnostics"]["nnls_converged"]),
        "j_share": float(result["j_share"]),
        "var_fraction": float(result["var_fraction"]),
        "reconstruction_r2": float(result["reconstruction_r2"]),
        "cos_x_xj": float(result["cos_x_xj"]),
        "token_ids": token_ids,
        "tokens": [tok.decode([t]) for t in token_ids],
        "token_ids_selection_order": token_ids,
        "tokens_selection_order": [tok.decode([t]) for t in token_ids],
        "coeffs": [float(value) for value in result["coeffs"]],
        "token_ids_by_coefficient": coefficient_ids,
        "tokens_by_coefficient": [tok.decode([t]) for t in coefficient_ids],
        "top_tokens": [tok.decode([t]) for t in coefficient_ids],
        "coeffs_by_coefficient": [
            float(value) for value in result["coeffs_by_coefficient"]
        ],
        "diagnostics": result["diagnostics"],
        "algorithm": result["algorithm"],
    }


def assert_decompositions_converged(records, context=""):
    """Fail a summary step if any stored decomposition has a bad NNLS solve.

    ``records`` is a mapping of name -> decomposition record (or any nested
    structure of them). Long GPU sweeps record non-convergence and keep going;
    this is the gate that must run before those numbers are believed. Records
    written by the v1 algorithm carry no flag and are ignored here — the
    algorithm-version gates in the callers cover those.
    """
    def walk(node, path):
        if isinstance(node, dict):
            if "algorithm_version" in node and "nnls_converged" in node:
                if not node["nnls_converged"]:
                    yield path or "<root>"
                return
            for key, value in node.items():
                yield from walk(value, f"{path}.{key}" if path else str(key))
        elif isinstance(node, list):
            for i, value in enumerate(node):
                yield from walk(value, f"{path}[{i}]")

    bad = sorted(set(walk(records, "")))
    if bad:
        shown = ", ".join(bad[:10]) + (f" (+{len(bad) - 10} more)" if len(bad) > 10 else "")
        raise RuntimeError(
            f"{len(bad)} decomposition(s) have a non-converged NNLS solve"
            f"{' in ' + context if context else ''}: {shown}. "
            "These numbers are not trustworthy — re-run the affected cells with "
            "a larger nnls_max_iters."
        )
    return len(bad)


def assert_source_version(records, path, names=None):
    """Refuse to compare against decompositions from a different algorithm.

    Cross-file comparisons (J5/J6 token overlap) read decompositions produced
    by an *earlier script* and compare them to freshly-computed ones. If the
    two files came from different routing-algorithm versions the comparison is
    meaningless, and — unlike a shape or key error — it fails silently with a
    plausible-looking number. This is the guard for that.

    ``names`` restricts the check to the entries actually used; by default
    every entry carrying a version is checked.
    """
    checked = names if names is not None else list(records)
    stale = {}
    for name in checked:
        entry = records.get(name)
        if not isinstance(entry, dict):
            continue
        version = entry.get("algorithm_version", "v1_unversioned")
        if version != ROUTING_ALGORITHM_VERSION:
            stale[name] = version
    if stale:
        shown = ", ".join(f"{k}={v}" for k, v in sorted(stale.items())[:6])
        raise RuntimeError(
            f"{len(stale)} entr(ies) in {path} were produced by a different "
            f"routing-algorithm version than this run "
            f"({ROUTING_ALGORITHM_VERSION}): {shown}"
            f"{' ...' if len(stale) > 6 else ''}.\n"
            "Comparing token sets across algorithm versions is meaningless and "
            "would fail silently. Re-run run_mechanistic.py to regenerate that "
            "file under the current algorithm, then re-run this script."
        )
    return len(checked)


# The frozen cutoff used for every result committed before the v2 refactor.
# make_valid_mask now derives the cutoff from the tokenizer; if the derived
# value ever differs from this, the selectable atom dictionary has changed and
# J-share numbers are not comparable across the boundary.
HISTORICAL_MIN_ID_INVALID = 151669


def make_valid_mask(tok, vocab_size, min_id_invalid=None, warn_on_drift=True):
    """Mask special tokens and unused embedding-padding rows.

    ``len(tokenizer)`` includes added tokens and normally marks the boundary
    between trained token IDs and padded embedding rows. An explicit cutoff may
    still be supplied when provenance requires a frozen value.
    """
    if not isinstance(vocab_size, int) or vocab_size < 1:
        raise ValueError("vocab_size must be a positive integer")
    if min_id_invalid is None:
        try:
            min_id_invalid = len(tok)
        except TypeError as exc:
            raise ValueError(
                "Tokenizer has no length; pass min_id_invalid explicitly"
            ) from exc
    if not isinstance(min_id_invalid, int) or not 0 < min_id_invalid <= vocab_size:
        raise ValueError(
            f"min_id_invalid must be in [1, {vocab_size}], found {min_id_invalid}"
        )

    if warn_on_drift and min_id_invalid != HISTORICAL_MIN_ID_INVALID:
        print(
            f"WARNING routing_lib.make_valid_mask: cutoff {min_id_invalid} differs "
            f"from the frozen pre-refactor value {HISTORICAL_MIN_ID_INVALID}. The "
            f"selectable atom dictionary has changed, so J-share numbers are NOT "
            f"comparable to previously committed results. Pass "
            f"min_id_invalid={HISTORICAL_MIN_ID_INVALID} to reproduce them.",
            flush=True,
        )

    mask = torch.zeros(vocab_size, dtype=torch.bool)
    mask[:min_id_invalid] = True
    for special_id in getattr(tok, "all_special_ids", []):
        if 0 <= special_id < vocab_size:
            mask[special_id] = False
    if not bool(mask.any()):
        raise ValueError("Token mask excludes the entire vocabulary")
    return mask


def valid_mask_provenance(tok, vocab_size, min_id_invalid=None):
    """Return the JSON-safe provenance of the token mask actually used."""
    resolved = min_id_invalid if min_id_invalid is not None else len(tok)
    return {
        "vocab_size": int(vocab_size),
        "min_id_invalid": int(resolved),
        "source": "explicit" if min_id_invalid is not None else "len(tokenizer)",
        "historical_min_id_invalid": HISTORICAL_MIN_ID_INVALID,
        "matches_historical": bool(int(resolved) == HISTORICAL_MIN_ID_INVALID),
        "n_special_excluded": len([
            s for s in getattr(tok, "all_special_ids", []) if 0 <= s < vocab_size
        ]),
    }


def _single_token_id(tok, text):
    token_ids = tok.encode(text, add_special_tokens=False)
    if not token_ids:
        raise ValueError(f"Tokenizer produced no tokens for {text!r}")
    return int(token_ids[0])


def run_selftest(lens, hf_model, tok, layer, k=16, seed=0, n_random=8):
    """Run planted-mixture, scale, reconstruction, and random-cohort tests."""
    if n_random < 2:
        raise ValueError("n_random must be at least 2")
    d_model = lens.jacobians[layer].shape[0]
    valid_mask = make_valid_mask(tok, unembed_weight(hf_model).shape[0])
    js = JSpace(lens, hf_model, layer, valid_mask)

    token_texts = [" France", " lightning", " sadness"]
    token_ids = [_single_token_id(tok, text) for text in token_texts]
    token_ids = [t for t in dict.fromkeys(token_ids) if bool(valid_mask[t])]
    if len(token_ids) < 2:
        raise ValueError("Self-test needs at least two distinct valid token atoms")

    pure_id = token_ids[0]
    pure_vector = js.lens_vector(pure_id).detach().cpu().numpy()
    strict = {"require_nnls_convergence": True}
    pure = js.decompose(pure_vector, k=k, **strict)

    mixture_ids = token_ids[:3]
    mixture_weights = torch.tensor(
        [1.0, 0.65, 0.35][:len(mixture_ids)],
        dtype=torch.float32,
        device=js.device,
    )
    mixture_atoms = torch.stack([js.lens_vector(t) for t in mixture_ids])
    mixture_vector = (mixture_atoms.T @ mixture_weights).detach().cpu().numpy()
    mixture = js.decompose(mixture_vector, k=k, **strict)
    scaled = js.decompose(mixture_vector * 10.0, k=k, **strict)

    rng = np.random.default_rng(seed)
    mixture_norm = float(np.linalg.norm(mixture_vector))
    random_results = []
    for _ in range(n_random):
        direction = rng.standard_normal(d_model).astype(np.float32)
        direction = direction / np.linalg.norm(direction) * mixture_norm
        random_results.append(js.decompose(direction, k=k, **strict))

    def identity_error(result, target):
        reconstructed = result["x_j"] + result["x_perp"]
        return float(
            np.linalg.norm(reconstructed - target)
            / (np.linalg.norm(target) + 1e-12)
        )

    random_shares = [result["j_share"] for result in random_results]
    scale_difference = abs(mixture["j_share"] - scaled["j_share"])
    scale_tokens_equal = (
        mixture["token_ids_by_coefficient"]
        == scaled["token_ids_by_coefficient"]
    )
    all_results = [pure, mixture, scaled, *random_results]
    nnls_converged = all(
        result["diagnostics"]["nnls_converged"] for result in all_results
    )
    coefficients_nonnegative = all(
        bool(np.all(result["coeffs"] >= -1e-8)) for result in all_results
    )
    pure_identity_error = identity_error(pure, pure_vector)
    mixture_identity_error = identity_error(mixture, mixture_vector)

    passed = bool(
        pure["j_share"] > 0.99
        and pure_id in pure["token_ids"]
        and mixture["reconstruction_r2"] > 0.99
        and scale_difference < 1e-5
        and scale_tokens_equal
        and max(random_shares) < 0.20
        and float(np.mean(random_shares)) < 0.10
        and pure_identity_error < 1e-6
        and mixture_identity_error < 1e-6
        and nnls_converged
        and coefficients_nonnegative
    )

    return {
        "layer": int(layer),
        "k": int(k),
        "n_random": int(n_random),
        "pure_atom": {
            "j_share": pure["j_share"],
            "reconstruction_r2": pure["reconstruction_r2"],
            "target_token": tok.decode([pure_id]),
            "target_selected": pure_id in pure["token_ids"],
            "active_tokens": [tok.decode([t]) for t in pure["token_ids"]],
            "identity_error": pure_identity_error,
            "diagnostics": pure["diagnostics"],
        },
        "planted_mixture": {
            "j_share": mixture["j_share"],
            "reconstruction_r2": mixture["reconstruction_r2"],
            "source_tokens": [tok.decode([t]) for t in mixture_ids],
            "selected_tokens_by_coefficient": [
                tok.decode([t]) for t in mixture["token_ids_by_coefficient"]
            ],
            "identity_error": mixture_identity_error,
            "diagnostics": mixture["diagnostics"],
        },
        "scale_invariance": {
            "j_share_original": mixture["j_share"],
            "j_share_times_10": scaled["j_share"],
            "absolute_difference": scale_difference,
            "token_order_equal": scale_tokens_equal,
        },
        "random_cohort": {
            "shares": random_shares,
            "mean": float(np.mean(random_shares)),
            "max": float(np.max(random_shares)),
        },
        "nnls_converged_all": nnls_converged,
        "coefficients_nonnegative_all": coefficients_nonnegative,
        "pass": passed,
    }
