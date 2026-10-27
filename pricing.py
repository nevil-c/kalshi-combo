"""One-factor Gaussian copula pricing for all-legs-must-win combo contracts.

Latent variable per leg: Z_i = sqrt(rho)*M + sqrt(1-rho)*e_i, with M, e_i ~ N(0,1).
Leg i resolves YES when Z_i < Phi^-1(p_i), so P(YES) = p_i marginally.
Conditional on M the legs are independent, so

    P(all legs win) = E_M[ prod_i P(leg i wins | M) ]

which is computed with Gauss-Hermite quadrature.

For rho above GH_RHO_MAX each conditional leg probability becomes a near-step
in M (width ~ sqrt((1-rho)/rho)) that fixed Hermite nodes cannot resolve, so
the integral switches to piecewise Gauss-Legendre against the normal density
with breakpoints clustered at each leg's step.

rho is restricted to [0, 1): a single common factor cannot produce negative
pairwise correlation.
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache

import numpy as np
from scipy.stats import norm

RHO_MAX = 1.0 - 1e-9
P_EPS = 1e-12
GH_RHO_MAX = 0.8  # 64-node Gauss-Hermite error < 1e-8 up to here
M_RANGE = 12.0  # N(0,1) mass beyond +/-12 is ~1e-33
STEP_OFFSETS = (0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0)


@lru_cache(maxsize=8)
def _nodes(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Gauss-Hermite nodes/weights rescaled to integrate against N(0,1)."""
    x, w = np.polynomial.hermite.hermgauss(n)
    return np.sqrt(2.0) * x, w / np.sqrt(np.pi)


@lru_cache(maxsize=8)
def _legendre(n: int) -> tuple[np.ndarray, np.ndarray]:
    return np.polynomial.legendre.leggauss(n)


def _step_nodes(probs: np.ndarray, rho: float, n: int = 24) -> tuple[np.ndarray, np.ndarray]:
    """Nodes/weights against N(0,1), dense around each leg's conditional step."""
    rho = min(rho, RHO_MAX)
    width = np.sqrt((1.0 - rho) / rho)
    inner = probs[(probs > 0.0) & (probs < 1.0)].clip(P_EPS, 1.0 - P_EPS)
    centers = norm.ppf(inner) / np.sqrt(rho)
    pts = [-M_RANGE, M_RANGE]
    for c in centers:
        for k in STEP_OFFSETS:
            pts += [c - k * width, c + k * width]
    edges = np.unique(np.clip(pts, -M_RANGE, M_RANGE))
    x, w = _legendre(n)
    a, b = edges[:-1, None], edges[1:, None]
    m = (0.5 * (b - a) * x + 0.5 * (a + b)).ravel()
    wt = (0.5 * (b - a) * w).ravel() * norm.pdf(m)
    return m, wt


def _side_is_yes(side) -> bool:
    if isinstance(side, (bool, np.bool_)):
        return bool(side)
    if isinstance(side, (int, np.integer)) and side in (0, 1):
        return bool(side)
    s = str(side).strip().lower()
    if s in ("yes", "y"):
        return True
    if s in ("no", "n"):
        return False
    raise ValueError(f"side must be 'yes'/'no' or bool, got {side!r}")


def cond_leg_prob(p: float, rho: float, m: np.ndarray) -> np.ndarray:
    """P(leg resolves YES | M=m) for a leg with marginal YES probability p."""
    m = np.asarray(m, dtype=float)
    if p <= 0.0:
        return np.zeros_like(m)
    if p >= 1.0:
        return np.ones_like(m)
    p = min(max(p, P_EPS), 1.0 - P_EPS)
    rho = min(max(rho, 0.0), RHO_MAX)
    return norm.cdf((norm.ppf(p) - np.sqrt(rho) * m) / np.sqrt(1.0 - rho))


def combo_prob(
    leg_probs: Sequence[float],
    sides: Sequence,
    rho: float,
    n_nodes: int = 64,
) -> float:
    """Probability that every leg finishes on its chosen side.

    leg_probs: marginal YES probability of each leg's market.
    sides: 'yes'/'no' (or True/False) per leg; a NO leg wins with prob 1 - p.
    rho: common pairwise latent correlation, 0 <= rho < 1.
    """
    probs = np.asarray(leg_probs, dtype=float)
    if probs.ndim != 1 or len(probs) == 0:
        raise ValueError("leg_probs must be a non-empty 1-D sequence")
    if len(sides) != len(probs):
        raise ValueError("leg_probs and sides must have the same length")
    if np.any((probs < 0) | (probs > 1)) or np.any(np.isnan(probs)):
        raise ValueError("leg probabilities must be in [0, 1]")
    if not 0.0 <= rho < 1.0:
        raise ValueError("rho must be in [0, 1)")
    yes = [_side_is_yes(s) for s in sides]

    if rho == 0.0:
        return float(np.prod([p if y else 1.0 - p for p, y in zip(probs, yes)]))

    m, w = _nodes(n_nodes) if rho <= GH_RHO_MAX else _step_nodes(probs, rho)
    integrand = np.ones_like(m)
    for p, y in zip(probs, yes):
        c = cond_leg_prob(p, rho, m)
        integrand *= c if y else 1.0 - c
    return float(np.clip(w @ integrand, 0.0, 1.0))
