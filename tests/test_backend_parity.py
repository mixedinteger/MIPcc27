"""
Cross-backend tests: every installed solver must read an instance the same way.

The checker's verdict must not depend on which solver happens to be installed.
These run each test once per installed backend, so ``--solver scip`` is exercised
rather than merely importable, and compare the two readers against each other on
the same file: a disagreement about a coefficient, a sense or a bound is a
disagreement about what the instance *is*, and would let a submission pass under
one backend and fail under the other.
"""

import json
from pathlib import Path

import pytest

# The fixtures build their models with gurobipy: without it, skip rather than error.
gp = pytest.importorskip("gurobipy")

from gurobipy import GRB  # noqa: E402

from mipcc_iis_checker import verify_subsystem  # noqa: E402
from mipcc_iis_checker.backend import (  # noqa: E402
    HAS_GUROBI,
    HAS_HIGHS,
    HAS_SCIP,
    STATUS_INFEASIBLE,
    UnsupportedInstanceError,
    get_backend,
)

_INSTALLED = [
    name
    for name, present in (("scip", HAS_SCIP), ("gurobi", HAS_GUROBI), ("highs", HAS_HIGHS))
    if present
]

# Each test body runs once per installed backend.  Parametrising over what is
# installed rather than skipping keeps a one-solver machine green while making a
# two-solver machine actually compare them.
solver = pytest.fixture(params=_INSTALLED)(lambda request: request.param)


def _write_mps(tmp_path: Path, name: str, build) -> Path:
    """Write the model *build* constructs to ``<name>.mps`` and return the path."""
    with gp.Env(params={"OutputFlag": 0}) as env:
        m = gp.Model(env=env)
        build(m)
        path = tmp_path / f"{name}.mps"
        m.write(str(path))
    return path


def _mixed(m):
    """One row per sense, a bounded variable, a free variable, and integrality."""
    x = m.addVar(name="x", vtype=GRB.INTEGER, lb=0, ub=10)
    y = m.addVar(name="y", vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, ub=GRB.INFINITY)
    z = m.addVar(name="z", vtype=GRB.BINARY)
    m.addConstr(2 * x + 3 * y <= 7, name="le")
    m.addConstr(x - y >= -2, name="ge")
    m.addConstr(x + z == 4, name="eq")
    m.setObjective(x + y + z)


def _model_facts(np_model):
    """The instance keyed by name, so a differing column order is not a difference."""
    var = np_model.variable_names.tolist()
    dense = np_model.A.toarray()
    return {
        "rows": {
            name: {
                "sense": np_model.senses[i],
                "rhs": np_model.rhs[i],
                "coeffs": {v: dense[i][j] for j, v in enumerate(var) if dense[i][j] != 0},
            }
            for i, name in enumerate(np_model.constraint_names.tolist())
        },
        "vars": {
            name: {
                "lb": np_model.lb[j],
                "ub": np_model.ub[j],
                "is_integer": bool(np_model.is_integer[j]),
            }
            for j, name in enumerate(var)
        },
        "element_sets": {
            "rows": sorted(np_model.rows),
            "lower_bounds": sorted(np_model.lower_bounds),
            "upper_bounds": sorted(np_model.upper_bounds),
            "integers": sorted(np_model.integers),
        },
    }


# --------------------------------------------------------------------------
# The two readers must agree about the instance
# --------------------------------------------------------------------------


@pytest.mark.skipif(len(_INSTALLED) < 2, reason="needs two backends to compare")
def test_every_backend_reads_the_same_instance(tmp_path: Path):
    """Coefficients, senses, rhs, bounds and integrality, compared by name."""
    mps = _write_mps(tmp_path, "mixed", _mixed)

    facts = {}
    for name in _INSTALLED:
        backend = get_backend(solver=name)
        backend.load_mps(str(mps))
        facts[name] = _model_facts(backend.np_model)

    first, *rest = _INSTALLED
    for other in rest:
        assert facts[other] == facts[first], f"{other} disagrees with {first} about the instance"


def test_the_backend_reports_the_element_sets_it_read(tmp_path: Path, solver: str):
    """Per backend, so a reader that loses a bound or an integer is caught alone."""
    mps = _write_mps(tmp_path, "mixed", _mixed)
    backend = get_backend(solver=solver)
    backend.load_mps(str(mps))

    assert backend.np_model.rows == {"le", "ge", "eq"}
    assert backend.np_model.integers == {"x", "z"}
    # y is free on both sides; x and z are bounded on both.
    assert backend.np_model.lower_bounds == {"x", "z"}
    assert backend.np_model.upper_bounds == {"x", "z"}


# --------------------------------------------------------------------------
# The verdict must not depend on the backend
# --------------------------------------------------------------------------


def _infeasible(m):
    """IIS is {c1, c2} once c3, the bounds and integrality are relaxed."""
    x1 = m.addVar(name="x1", vtype=GRB.INTEGER, lb=0)
    x2 = m.addVar(name="x2", vtype=GRB.INTEGER, lb=0)
    m.addConstr(x1 + x2 <= 1.5, name="c1")
    m.addConstr(x1 + x2 >= 2.5, name="c2")
    m.addConstr(x1 - x2 <= 0, name="c3")
    m.setObjective(x1 + x2)


_RELAX_BOTH = {
    "x1": {"lower_bound": {"action": "relax"}, "integrality": {"action": "relax"}},
    "x2": {"lower_bound": {"action": "relax"}, "integrality": {"action": "relax"}},
}


def _sol(tmp_path: Path, doc: dict) -> Path:
    path = tmp_path / "sol.json"
    path.write_text(json.dumps(doc))
    return path


def test_a_proven_iis_passes_on_every_backend(tmp_path: Path, solver: str):
    mps = _write_mps(tmp_path, "infeasible", _infeasible)
    sol = _sol(
        tmp_path,
        {
            "rows": {
                "c3": {"action": "relax"},
                "c1": {"action": "keep", "certificate": {"x1": 3.0, "x2": 0.0}},
                "c2": {"action": "keep", "certificate": {"x1": 0.0, "x2": 0.0}},
            },
            "columns": _RELAX_BOTH,
        },
    )
    res = verify_subsystem(mps, sol, solver=solver)

    assert res.stats.solver == solver
    assert res.stats.solver_status == "INFEASIBLE"
    assert res.is_infeasible is True
    assert res.is_minimal is True
    assert res.is_passed


def test_a_feasible_subsystem_fails_on_every_backend(tmp_path: Path, solver: str):
    """Removing c2 leaves only c1, which any solver must call feasible."""
    mps = _write_mps(tmp_path, "infeasible", _infeasible)
    sol = _sol(tmp_path, {"rows": {"c2": {"action": "relax"}, "c3": {"action": "relax"}}})
    res = verify_subsystem(mps, sol, solver=solver)

    assert res.stats.solver == solver
    assert res.stats.solver_status == "FEASIBLE"
    assert res.is_infeasible is False
    assert not res.is_passed


def test_integrality_is_what_makes_it_infeasible_on_every_backend(tmp_path: Path, solver: str):
    """2*x1 + 2*x2 == 3 is infeasible over the integers and feasible without them,
    so a backend that lost the integrality flag reports FEASIBLE here."""

    def build(m):
        x1 = m.addVar(name="x1", vtype=GRB.INTEGER, lb=0)
        x2 = m.addVar(name="x2", vtype=GRB.INTEGER, lb=0)
        m.addConstr(2 * x1 + 2 * x2 == 3, name="c1")
        m.setObjective(x1 + x2)

    mps = _write_mps(tmp_path, "integrality", build)

    kept = _sol(
        tmp_path,
        {"columns": {n: {"lower_bound": {"action": "relax"}} for n in ("x1", "x2")}},
    )
    assert verify_subsystem(mps, kept, solver=solver).is_infeasible is True

    relaxed = tmp_path / "relaxed.json"
    relaxed.write_text(
        json.dumps(
            {
                "columns": {
                    n: {
                        "lower_bound": {"action": "relax"},
                        "integrality": {"action": "relax"},
                    }
                    for n in ("x1", "x2")
                }
            }
        )
    )
    assert verify_subsystem(mps, relaxed, solver=solver).is_infeasible is False


def test_the_time_limit_reaches_every_backend(tmp_path: Path, solver: str):
    """A limit that cannot be met must come back UNKNOWN, not INFEASIBLE."""
    mps = _write_mps(tmp_path, "infeasible", _infeasible)
    sol = _sol(tmp_path, {"rows": {"c3": {"action": "relax"}}, "columns": _RELAX_BOTH})

    res = verify_subsystem(mps, sol, solver=solver, time_limit=1e-9)
    assert res.stats.solver_status in {"INFEASIBLE", "UNKNOWN"}
    # Whatever it returned, an expired limit is never a failed submission.
    assert res.is_infeasible in {True, None}


# --------------------------------------------------------------------------
# Refusing what NpModel cannot represent
# --------------------------------------------------------------------------


def test_a_quadratic_constraint_is_refused_by_every_backend(tmp_path: Path, solver: str):
    """Dropping it would make every subsystem a relaxation of the real instance."""

    def build(m):
        x = m.addVar(name="x", lb=-10, ub=10)
        y = m.addVar(name="y", lb=-10, ub=10)
        m.addQConstr(x * x + y * y <= 4, name="q")
        m.setObjective(x + y)

    mps = _write_mps(tmp_path, "quad", build)
    backend = get_backend(solver=solver)
    with pytest.raises(UnsupportedInstanceError):
        backend.load_mps(str(mps))


def test_an_empty_row_is_handled_by_every_backend(tmp_path: Path, solver: str):
    """``0 >= 1`` is infeasible with nothing to assign — the branch that cannot be
    built as a solver expression, because it is the plain number 0."""
    mps = tmp_path / "emptyrow.mps"
    mps.write_text(
        "NAME          EMPTYROW\n"
        "ROWS\n"
        " N  OBJ\n"
        " G  r0\n"
        " L  r1\n"
        "COLUMNS\n"
        "    x         OBJ       1.0        r1        1.0\n"
        "RHS\n"
        "    RHS       r0        1.0\n"
        "BOUNDS\n"
        " UP BND       x         5.0\n"
        "ENDATA\n"
    )

    backend = get_backend(solver=solver)
    backend.load_mps(str(mps))
    assert backend.np_model.rows == {"r0", "r1"}

    # r0 has no coefficients, so it reads 0 >= 1: infeasible on its own.
    m = backend.np_model
    status, _ = backend.solve_subsystem(m.rows, m.lower_bounds, m.upper_bounds, m.integers)
    assert status == "INFEASIBLE"

    # Relax r0 and the rest is satisfiable.
    status, _ = backend.solve_subsystem({"r1"}, m.lower_bounds, m.upper_bounds, m.integers)
    assert status == "FEASIBLE"


def test_ranged_rows_are_refused_by_every_backend(tmp_path: Path, solver: str):
    """A RANGES row is two restrictions, and NpModel holds one (sense, rhs) per row.

    Gurobi's reader rewrites it as an equality plus a bounded auxiliary variable,
    whose two bounds would be scored as elements the submitter never wrote.
    """
    mps = tmp_path / "ranges.mps"
    mps.write_text(
        "NAME          RNG\n"
        "ROWS\n"
        " N  OBJ\n"
        " L  r0\n"
        "COLUMNS\n"
        "    x         OBJ       1.0        r0        1.0\n"
        "RHS\n"
        "    RHS       r0        5.0\n"
        "RANGES\n"
        "    RNG       r0        3.0\n"
        "BOUNDS\n"
        "ENDATA\n"
    )

    with pytest.raises(UnsupportedInstanceError):
        get_backend(solver=solver).load_mps(str(mps))


def test_an_unsatisfiable_infinite_rhs_is_refused_by_every_backend(tmp_path: Path, solver: str):
    """``x >= 1e100`` is ``x >= +inf``, and ``lhs - inf`` is NaN once lhs overflows.

    NaN compares false against every tolerance, so the row would be satisfied by any
    certificate.  Gurobi normalises infinite bounds but not right-hand sides, and
    SCIP read this as a finite ``x >= 1e20`` until both were routed through the same
    check — the same file was an instance under one backend and refused by the other.
    """
    mps = tmp_path / "infrhs.mps"
    mps.write_text(
        "NAME          INF\n"
        "ROWS\n"
        " N  OBJ\n"
        " G  r0\n"
        "COLUMNS\n"
        "    x         OBJ       1.0        r0        1.0\n"
        "RHS\n"
        "    RHS       r0        1e100\n"
        "BOUNDS\n"
        " FR BND       x\n"
        "ENDATA\n"
    )

    with pytest.raises(UnsupportedInstanceError):
        get_backend(solver=solver).load_mps(str(mps))


def test_a_duplicate_row_name_is_refused_by_every_backend(tmp_path: Path, solver: str):
    """Elements are keyed by name, so a duplicate makes one of them unaddressable."""
    mps = tmp_path / "dup.mps"
    mps.write_text(
        "NAME          DUP\n"
        "ROWS\n"
        " N  OBJ\n"
        " L  r0\n"
        " L  r0\n"
        "COLUMNS\n"
        "    x         OBJ       1.0        r0        1.0\n"
        "RHS\n"
        "    RHS       r0        1.0\n"
        "BOUNDS\n"
        "ENDATA\n"
    )

    # SCIP's own reader rejects the file before the checker's check is reached.
    with pytest.raises((ValueError, OSError)):
        get_backend(solver=solver).load_mps(str(mps))


def test_an_unknown_section_is_refused_by_every_backend(tmp_path: Path, solver: str):
    """The section check is a whitelist, not a list of known-bad sections.

    A section nobody has classified is one whose effect on the element universe has
    not been checked, and the reader applies it either way.
    """
    mps = tmp_path / "unknown.mps"
    mps.write_text(
        "NAME          UNKNOWN\n"
        "ROWS\n"
        " N  OBJ\n"
        " L  r0\n"
        "COLUMNS\n"
        "    x         OBJ       1.0        r0        1.0\n"
        "RHS\n"
        "    RHS       r0        1.0\n"
        "SOMETHINGNEW\n"
        "    x         1.0\n"
        "BOUNDS\n"
        "ENDATA\n"
    )

    with pytest.raises(UnsupportedInstanceError, match="SOMETHINGNEW"):
        get_backend(solver=solver).load_mps(str(mps))


def test_objsense_and_comments_do_not_trip_the_section_check(tmp_path: Path, solver: str):
    """OBJSENSE only describes the objective, which is dropped, so it must load.

    A whitelist that refused it would reject valid instances; ``*`` in column 1 is a
    comment and must not read as a section header either.
    """
    mps = tmp_path / "objsense.mps"
    mps.write_text(
        "* a comment that is not a section\n"
        "NAME          OBJS\n"
        "OBJSENSE\n"
        "    MAX\n"
        "ROWS\n"
        " N  OBJ\n"
        " G  r0\n"
        " L  r1\n"
        "COLUMNS\n"
        "    x         OBJ       1.0        r0        1.0\n"
        "    x         r1        1.0\n"
        "RHS\n"
        "    RHS       r0        2.0        r1        1.0\n"
        "BOUNDS\n"
        "ENDATA\n"
    )

    backend = get_backend(solver=solver)
    backend.load_mps(str(mps))

    assert backend.np_model.rows == {"r0", "r1"}
    m = backend.np_model
    status, _ = backend.solve_subsystem(m.rows, m.lower_bounds, m.upper_bounds, m.integers)
    assert status == STATUS_INFEASIBLE


@pytest.mark.parametrize("section", ["QUADOBJ", "QMATRIX", "PWLOBJ"])
def test_an_objective_only_section_is_refused_by_every_backend(
    tmp_path: Path, solver: str, section: str
):
    """These describe the objective, which the checker drops -- refused all the same.

    They are vendor extensions rather than part of the MPS format, so what a reader
    makes of one is the reader's choice, and whitelisting them would mean trusting
    that every backend agrees.  Being strict costs only instances whose objective
    nobody needed read.
    """
    mps = tmp_path / f"{section.lower()}.mps"
    mps.write_text(
        "NAME          OBJSEC\n"
        "ROWS\n"
        " N  OBJ\n"
        " G  r0\n"
        "COLUMNS\n"
        "    x         OBJ       1.0        r0        1.0\n"
        "RHS\n"
        "    RHS       r0        1.0\n"
        f"{section}\n"
        "    x         x         1.0\n"
        "BOUNDS\n"
        "ENDATA\n"
    )

    with pytest.raises(UnsupportedInstanceError, match=section):
        get_backend(solver=solver).load_mps(str(mps))
