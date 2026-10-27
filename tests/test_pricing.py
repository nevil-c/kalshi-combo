import itertools

import numpy as np
import pytest
from scipy.stats import multivariate_normal, norm

from pricing import combo_prob

RHOS = [0.0, 0.05, 0.3, 0.5, 0.8, 0.95, 0.999]


def _random_cases(n=40, seed=7):
    rng = np.random.default_rng(seed)
    cases = []
    for _ in range(n):
        k = int(rng.integers(2, 6))
        probs = rng.uniform(0.01, 0.99, size=k).tolist()
        sides = rng.choice(["yes", "no"], size=k).tolist()
        cases.append((probs, sides))
    return cases


CASES = _random_cases()


def _effective(probs, sides):
    return [p if s == "yes" else 1 - p for p, s in zip(probs, sides)]


def test_spec_rho_zero():
    assert combo_prob([0.6, 0.6], ["yes", "yes"], 0.0) == pytest.approx(0.36, abs=1e-12)


def test_spec_rho_half_matches_bivariate_normal():
    t = norm.ppf(0.6)
    exact = multivariate_normal([0, 0], [[1, 0.5], [0.5, 1]]).cdf([t, t])
    got = combo_prob([0.6, 0.6], ["yes", "yes"], 0.5)
    assert got == pytest.approx(exact, abs=1e-6)
    assert got == pytest.approx(0.439, abs=5e-4)


@pytest.mark.parametrize("p1,p2", [(0.6, 0.3), (0.907, 0.64), (0.01, 0.99)])
@pytest.mark.parametrize("rho", [0.3, 0.8, 0.85, 0.95, 0.99, 0.999])
@pytest.mark.parametrize("sides", list(itertools.product(["yes", "no"], repeat=2)))
def test_two_legs_match_bivariate_normal(p1, p2, rho, sides):
    eff = _effective([p1, p2], sides)
    r = rho if sides[0] == sides[1] else -rho
    exact = multivariate_normal([0, 0], [[1, r], [r, 1]]).cdf(norm.ppf(eff))
    assert combo_prob([p1, p2], sides, rho) == pytest.approx(exact, abs=1e-7)


@pytest.mark.parametrize("probs,sides", CASES)
@pytest.mark.parametrize("rho", RHOS)
def test_combo_le_min_leg(probs, sides, rho):
    assert combo_prob(probs, sides, rho) <= min(_effective(probs, sides)) + 1e-9


@pytest.mark.parametrize("probs,sides", CASES)
def test_rho_zero_is_product(probs, sides):
    assert combo_prob(probs, sides, 0.0) == pytest.approx(np.prod(_effective(probs, sides)), abs=1e-12)


@pytest.mark.parametrize("probs,sides", CASES)
def test_tiny_rho_near_product(probs, sides):
    # quadrature path (rho > 0) agrees with the closed form as rho -> 0
    assert combo_prob(probs, sides, 1e-8) == pytest.approx(np.prod(_effective(probs, sides)), abs=1e-6)


@pytest.mark.parametrize("probs,sides", CASES)
@pytest.mark.parametrize("rho", RHOS)
def test_output_in_unit_interval(probs, sides, rho):
    assert 0.0 <= combo_prob(probs, sides, rho) <= 1.0


@pytest.mark.parametrize("probs,_", CASES)
def test_monotone_in_rho_all_yes(probs, _):
    vals = [combo_prob(probs, ["yes"] * len(probs), r) for r in RHOS]
    assert all(b >= a - 1e-9 for a, b in zip(vals, vals[1:]))


@pytest.mark.parametrize("p1,p2", [(0.6, 0.6), (0.907, 0.64), (0.05, 0.9), (0.5, 0.5)])
@pytest.mark.parametrize("rho", RHOS)
def test_two_leg_cells_sum_to_one(p1, p2, rho):
    total = sum(combo_prob([p1, p2], s, rho) for s in itertools.product(["yes", "no"], repeat=2))
    assert total == pytest.approx(1.0, abs=1e-9)


def test_degenerate_legs():
    assert combo_prob([1.0, 0.7], ["yes", "yes"], 0.4) == pytest.approx(0.7, abs=1e-9)
    assert combo_prob([0.0, 0.7], ["yes", "yes"], 0.4) == 0.0
    assert combo_prob([0.3], [True], 0.6) == pytest.approx(0.3, abs=1e-9)


@pytest.mark.parametrize("args", [
    ([0.5, 0.5], ["yes"], 0.3),
    ([0.5, 1.2], ["yes", "yes"], 0.3),
    ([0.5, 0.5], ["yes", "yes"], 1.0),
    ([0.5, 0.5], ["yes", "yes"], -0.1),
    ([0.5, 0.5], ["yes", "maybe"], 0.3),
])
def test_invalid_inputs(args):
    with pytest.raises(ValueError):
        combo_prob(*args)
