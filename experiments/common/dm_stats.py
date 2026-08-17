"""Statistics for v-vs-u comparisons: effect sizes + Bayesian estimation.

Design notes:
- Our core comparison is PAIRED: for each (prompt, layer) cell we get a routing
  metric under axis v and under axis u -> analyze per-prompt differences.
- Frequentist: Cohen's d (paired: d_z), bootstrap CIs, permutation p.
- Bayesian: posterior over the standardized mean difference delta with a
  Student-t likelihood (robust to outliers), computed by numerical grid
  integration (no MCMC deps). Reports posterior mean/CI, mass inside a ROPE,
  and a Savage-Dickey Bayes factor for delta=0.
  Priors (pre-registered in pre-registration.md): delta ~ Normal(0, 1),
  log_sigma ~ flat over grid, nu fixed grid 1..100 with prior ~ Exp(1/29)
  (Kruschke BEST defaults, simplified).
- Self-test: recover planted effects (d=0 and d=0.6) before first real use.
  Run: python3 dm_stats.py --selftest
"""

import json

import numpy as np

try:
    from scipy import stats as sps
except ImportError:  # scipy optional for the pure-numpy parts
    sps = None


# ---------------------------------------------------------------------------
# Frequentist
# ---------------------------------------------------------------------------

def cohens_d_paired(x, y):
    """d_z for paired samples: mean(x-y)/sd(x-y)."""
    diff = np.asarray(x, float) - np.asarray(y, float)
    sd = diff.std(ddof=1)
    return float(diff.mean() / sd) if sd > 0 else 0.0


def cohens_d_independent(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    nx, ny = len(x), len(y)
    pooled = np.sqrt(((nx - 1) * x.var(ddof=1) + (ny - 1) * y.var(ddof=1)) / (nx + ny - 2))
    return float((x.mean() - y.mean()) / pooled) if pooled > 0 else 0.0


def bootstrap_ci(values, stat=np.mean, n_boot=10000, ci=0.95, seed=0):
    values = np.asarray(values, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    boots = np.array([stat(values[i]) for i in idx])
    lo, hi = np.quantile(boots, [(1 - ci) / 2, 1 - (1 - ci) / 2])
    return float(lo), float(hi)


def permutation_p_paired(x, y, n_perm=10000, seed=0):
    """Two-sided sign-flip permutation test on paired differences."""
    diff = np.asarray(x, float) - np.asarray(y, float)
    rng = np.random.default_rng(seed)
    obs = abs(diff.mean())
    signs = rng.choice([-1.0, 1.0], size=(n_perm, len(diff)))
    null = np.abs((signs * diff).mean(axis=1))
    return float((np.sum(null >= obs) + 1) / (n_perm + 1))


# ---------------------------------------------------------------------------
# Bayesian estimation of standardized paired difference (grid, no MCMC)
# ---------------------------------------------------------------------------

def bayes_paired(x, y, rope=(-0.1, 0.1), delta_prior_sd=1.0,
                 n_delta=801, n_sigma=200, nu_grid=(1, 2, 3, 5, 8, 12, 20, 35, 60, 100)):
    """Posterior over standardized mean difference delta = mu_diff/sigma_diff.

    Student-t likelihood on paired diffs. Returns posterior summary, ROPE mass,
    and Savage-Dickey BF01 (evidence FOR delta=0 vs the Normal(0, sd) prior).
    """
    diff = np.asarray(x, float) - np.asarray(y, float)
    n = len(diff)
    s = diff.std(ddof=1)
    if s == 0:
        s = max(abs(diff.mean()), 1e-9)

    deltas = np.linspace(-4, 4, n_delta)             # standardized effect
    sigmas = np.geomspace(s / 20, s * 20, n_sigma)   # scale of diffs
    nus = np.array(nu_grid, float)
    lp_nu = -nus / 29.0                              # ~Exp(1/29) prior (unnorm)

    # log prior on delta
    lp_delta = -0.5 * (deltas / delta_prior_sd) ** 2

    # log-likelihood cube via broadcasting: (delta, sigma, nu)
    from numpy import log
    def t_logpdf(z, nu):
        if sps is not None:
            return sps.t.logpdf(z, nu)
        from math import lgamma
        c = lgamma((nu + 1) / 2) - lgamma(nu / 2) - 0.5 * log(nu * np.pi)
        return c - (nu + 1) / 2 * np.log1p(z ** 2 / nu)

    logpost = np.full((n_delta, n_sigma, len(nus)), -np.inf)
    for k, nu in enumerate(nus):
        # z: (delta, sigma, obs)
        mu = deltas[:, None] * sigmas[None, :]
        z = (diff[None, None, :] - mu[:, :, None]) / sigmas[None, :, None]
        ll = t_logpdf(z, nu).sum(axis=2) - n * np.log(sigmas)[None, :]
        logpost[:, :, k] = ll + lp_delta[:, None] + lp_nu[k]

    logpost -= logpost.max()
    post = np.exp(logpost)
    post_delta = post.sum(axis=(1, 2))
    post_delta /= post_delta.sum()

    mean_d = float((deltas * post_delta).sum())
    cdf = np.cumsum(post_delta)
    ci_lo = float(deltas[np.searchsorted(cdf, 0.025)])
    ci_hi = float(deltas[np.searchsorted(cdf, 0.975)])
    in_rope = float(post_delta[(deltas >= rope[0]) & (deltas <= rope[1])].sum())

    # Savage-Dickey: posterior density at 0 / prior density at 0
    ddelta = deltas[1] - deltas[0]
    post_at0 = float(post_delta[np.argmin(np.abs(deltas))] / ddelta)
    prior = np.exp(lp_delta); prior /= prior.sum() * ddelta
    prior_at0 = float(prior[np.argmin(np.abs(deltas))])
    bf01 = post_at0 / prior_at0

    return {
        "n": n, "mean_diff_raw": float(diff.mean()), "sd_diff": float(s),
        "delta_mean": mean_d, "delta_ci95": [ci_lo, ci_hi],
        "rope": list(rope), "p_in_rope": in_rope, "bf01_savage_dickey": bf01,
    }


def full_report(x, y, label, rope=(-0.1, 0.1), seed=0):
    """Everything we pre-registered, in one dict. x=metric under v, y=under u."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    diff = x - y
    rep = {
        "label": label,
        "n_pairs": len(diff),
        "mean_v": float(x.mean()), "mean_u": float(y.mean()),
        "mean_diff": float(diff.mean()),
        "diff_ci95_bootstrap": bootstrap_ci(diff, seed=seed),
        "cohens_dz": cohens_d_paired(x, y),
        "perm_p_two_sided": permutation_p_paired(x, y, seed=seed),
        "bayes": bayes_paired(x, y, rope=rope),
    }
    return rep


# ---------------------------------------------------------------------------
# Self-test: recover planted effects
# ---------------------------------------------------------------------------

def _selftest():
    rng = np.random.default_rng(42)
    n = 80
    base = rng.normal(0, 1, n)

    # Case 1: true null (d=0)
    x0 = base + rng.normal(0, 0.5, n)
    y0 = base + rng.normal(0, 0.5, n)
    r0 = full_report(x0, y0, "planted_null")

    # Case 2: planted paired effect d_z ~ 0.6
    noise = rng.normal(0, 0.5, n)
    y1 = base + noise
    x1 = base + noise + 0.6 * 0.5 * np.sqrt(2)  # shift = d * sd(diff-ish)
    x1 += rng.normal(0, 0.15, n)  # small independent noise so sd(diff)>0
    r1 = full_report(x1, y1, "planted_effect")

    ok_null = (abs(r0["cohens_dz"]) < 0.25 and r0["perm_p_two_sided"] > 0.05
               and r0["bayes"]["p_in_rope"] > 0.3 and r0["bayes"]["bf01_savage_dickey"] > 1)
    ok_eff = (r1["cohens_dz"] > 1.0 and r1["perm_p_two_sided"] < 0.01
              and r1["bayes"]["p_in_rope"] < 0.05)
    out = {"planted_null": r0, "planted_effect": r1,
           "null_recovered": bool(ok_null), "effect_recovered": bool(ok_eff),
           "pass": bool(ok_null and ok_eff)}
    print(json.dumps(out, indent=2, default=float))
    print("STATS_SELFTEST_PASS" if out["pass"] else "STATS_SELFTEST_FAIL")
    return out


if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv:
        _selftest()
