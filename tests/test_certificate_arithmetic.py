"""
A certificate is checked in floating point, and the arithmetic must not decide.

Huge finite values can cancel (``1 + 1e300 - 1e300`` computes as 0) or overflow, and
either would let a violated row look satisfied -- or the element a certificate must
violate look violated.  Rows whose verdict the rounding error could flip are
recomputed exactly.  These build an NpModel directly, so they need no solver.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from mipcc_iis_checker.backend import DEFAULT_TOLERANCE, NpModel


def _model(rows, senses, rhs, names=("a", "u", "v")):
    """Rows of coefficients over free continuous variables *names*."""
    n = len(names)
    return NpModel(
        A=sp.csr_matrix(np.array(rows, dtype=float)),
        senses=np.array(senses, dtype=object),
        rhs=np.array(rhs, dtype=float),
        lb=np.full(n, -np.inf),
        ub=np.full(n, np.inf),
        is_integer=np.zeros(n, dtype=bool),
        variable_names=np.array(names, dtype=object),
        constraint_names=np.array([f"r{i}" for i in range(len(rows))], dtype=object),
    )


def test_cancelling_huge_values_do_not_hide_a_violation():
    """a + u + v <= 0 at (1, 1e300, -1e300) is 1, not the 0 floating point gives."""
    m = _model([[1, 1, 1]], ["<"], [0.0])
    ok, violations = m.is_feasible({"a": 1.0, "u": 1e300, "v": -1e300}, DEFAULT_TOLERANCE)
    assert not ok
    assert violations == ["constraint 'r0': lhs 1 > rhs 0 (violation 1)"]


def test_cancelling_huge_values_do_not_fake_the_certified_violation():
    """The element a certificate must violate is held to the exact value too.

    a + u + v >= 1 at (1, 1e300, -1e300) is exactly 1, so it is satisfied; floating
    point says 0, which would read as the violation the certificate needs.
    """
    m = _model([[1, 1, 1]], [">"], [1.0])
    ok, violations = m.is_feasible(
        {"a": 1.0, "u": 1e300, "v": -1e300}, DEFAULT_TOLERANCE, ignore=("row", "r0")
    )
    assert not ok
    assert any("satisfies row 'r0' too" in v for v in violations)


def test_an_overflowing_activity_is_computed_exactly():
    """2u - 2v >= 1 at u = 1.5e308, v = 1.6e308 overflows; it is -2e307, violated."""
    m = _model([[0, 2, -2]], [">"], [1.0])
    ok, violations = m.is_feasible({"u": 1.5e308, "v": 1.6e308}, DEFAULT_TOLERANCE)
    assert not ok
    assert violations == ["constraint 'r0': lhs -2e+307 < rhs 1 (violation 2e+307)"]


def test_an_overflowing_activity_that_holds_is_not_a_violation():
    """u + v = 0 at (1.7e308, -1.7e308) holds: huge values alone are no violation."""
    m = _model([[0, 1, 1], [0, 1, 1]], ["<", ">"], [0.0, 0.0])
    ok, violations = m.is_feasible({"u": 1.7e308, "v": -1.7e308}, DEFAULT_TOLERANCE)
    assert ok, violations


def test_large_but_honest_values_pass_on_the_fast_path():
    """Big-M rows: 1e9 a - 1e9 u = 0 at a = u = 3 holds, to well within tolerance."""
    m = _model([[1e9, -1e9, 0], [1e9, 1, 1]], ["=", "<"], [0.0, 3e9 + 1e-6])
    ok, violations = m.is_feasible({"a": 3.0, "u": 3.0, "v": -3.0}, DEFAULT_TOLERANCE)
    assert ok, violations


def test_a_violation_just_past_the_tolerance_still_counts():
    """The exact path changes no verdict the arithmetic could make reliably."""
    m = _model([[1, 0, 0]], ["<"], [0.0])
    assert not m.is_feasible({"a": 2 * DEFAULT_TOLERANCE}, DEFAULT_TOLERANCE)[0]
    assert m.is_feasible({"a": DEFAULT_TOLERANCE / 2}, DEFAULT_TOLERANCE)[0]
