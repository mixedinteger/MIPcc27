"""
A FEASIBLE verdict needs a witness that passes the competition's tolerance.

Solver tolerances are relative or applied to a scaled model, so the point a solver
offers can miss the absolute 1e-5 on the original rows.  ``solve_subsystem`` checks
that point the way a certificate is checked, retries once at a tighter solver
tolerance, and otherwise answers UNKNOWN.  A scripted backend stands in for the
solver here, so these run without one.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from mipcc_iis_checker.backend import (
    DEFAULT_TOLERANCE,
    STATUS_FEASIBLE,
    STATUS_INFEASIBLE,
    STATUS_UNKNOWN,
    BaseBackend,
    NpModel,
)
from mipcc_iis_checker.backend.base import RETRY_SOLVER_TOLERANCE


class _Scripted(BaseBackend):
    """Answers each solve from *answers* in turn and records the tolerances asked for."""

    name = "scripted"

    def __init__(self, answers):
        super().__init__()
        self.answers = list(answers)
        self.calls: list[tuple[float, float | None]] = []
        # One row, r: x >= 1, over a free variable.
        self._np_model = NpModel(
            A=sp.csr_matrix(np.array([[1.0]])),
            senses=np.array([">"], dtype=object),
            rhs=np.array([1.0]),
            lb=np.array([-np.inf]),
            ub=np.array([np.inf]),
            is_integer=np.array([False]),
            variable_names=np.array(["x"], dtype=object),
            constraint_names=np.array(["r"], dtype=object),
        )

    def load_mps(self, filepath: str) -> None:
        pass

    def _solve(self, sub, solver_tolerance, time_limit):
        self.calls.append((solver_tolerance, time_limit))
        return self.answers.pop(0)


def _solve(backend: BaseBackend, **kwargs):
    m = backend.np_model
    return backend.solve_subsystem(m.rows, m.lower_bounds, m.upper_bounds, m.integers, **kwargs)


def test_a_point_within_tolerance_is_feasible_after_one_solve():
    backend = _Scripted([(STATUS_FEASIBLE, [1 - DEFAULT_TOLERANCE / 2], 1.0)])
    assert _solve(backend) == (STATUS_FEASIBLE, 1.0)
    assert backend.calls == [(DEFAULT_TOLERANCE, None)]


def test_a_point_outside_tolerance_earns_one_tighter_solve():
    backend = _Scripted([(STATUS_FEASIBLE, [0.9], 1.0), (STATUS_FEASIBLE, [1.0], 2.0)])
    assert _solve(backend) == (STATUS_FEASIBLE, 3.0)
    assert [tol for tol, _ in backend.calls] == [DEFAULT_TOLERANCE, RETRY_SOLVER_TOLERANCE]


def test_a_point_that_misses_twice_is_inconclusive_not_feasible():
    """No witness, so the submission is not failed on the solver's say-so."""
    backend = _Scripted([(STATUS_FEASIBLE, [0.9], 1.0), (STATUS_FEASIBLE, [0.99], 1.0)])
    assert _solve(backend)[0] == STATUS_UNKNOWN


def test_the_tighter_solve_may_prove_infeasibility():
    backend = _Scripted([(STATUS_FEASIBLE, [0.9], 1.0), (STATUS_INFEASIBLE, None, 1.0)])
    assert _solve(backend)[0] == STATUS_INFEASIBLE


def test_a_feasible_status_without_a_point_is_no_witness():
    backend = _Scripted([(STATUS_FEASIBLE, None, 1.0), (STATUS_FEASIBLE, None, 1.0)])
    assert _solve(backend)[0] == STATUS_UNKNOWN


def test_infeasible_and_unknown_are_taken_as_they_come():
    for status in (STATUS_INFEASIBLE, STATUS_UNKNOWN):
        backend = _Scripted([(status, None, 1.0)])
        assert _solve(backend) == (status, 1.0)
        assert len(backend.calls) == 1


def test_the_retry_shares_the_time_limit():
    backend = _Scripted([(STATUS_FEASIBLE, [0.9], 4.0), (STATUS_FEASIBLE, [1.0], 1.0)])
    assert _solve(backend, time_limit=10.0)[0] == STATUS_FEASIBLE
    assert [limit for _, limit in backend.calls] == [10.0, 6.0]


def test_no_retry_once_the_time_limit_is_spent():
    backend = _Scripted([(STATUS_FEASIBLE, [0.9], 10.0)])
    assert _solve(backend, time_limit=10.0) == (STATUS_UNKNOWN, 10.0)
    assert len(backend.calls) == 1


def test_the_witness_is_held_to_the_tolerance_asked_for():
    """--tolerance moves the bar for the witness as it does for certificates."""
    backend = _Scripted([(STATUS_FEASIBLE, [1 - 5e-3], 1.0)])
    assert _solve(backend, tolerance=1e-2)[0] == STATUS_FEASIBLE
    assert backend.calls == [(1e-2, None)]


def test_a_tolerance_at_the_retry_level_solves_once():
    backend = _Scripted([(STATUS_FEASIBLE, [0.9], 1.0)])
    assert _solve(backend, tolerance=RETRY_SOLVER_TOLERANCE)[0] == STATUS_UNKNOWN
    assert backend.calls == [(RETRY_SOLVER_TOLERANCE, None)]
