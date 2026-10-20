"""
Unit tests for MIP Competition 2027 Solution Checker.
"""

import gzip
import json
from pathlib import Path
from unittest import mock

import pytest

# The fixtures build their models with gurobipy: without it, skip rather than error.
gp = pytest.importorskip("gurobipy")

from gurobipy import GRB  # noqa: E402

from mipcc_iis_checker import ElementCounts, verify_subsystem  # noqa: E402
from mipcc_iis_checker.backend import (  # noqa: E402
    STATUS_UNKNOWN,
    GurobipyBackend,
    UnsupportedInstanceError,
)


def _make_env() -> gp.Env:
    return gp.Env(params={"OutputFlag": 0})


@pytest.fixture
def sample_mps(tmp_path: Path) -> Path:
    """
    Infeasible MIP:
      c1: x1 + x2 <= 1.5
      c2: x1 + x2 >= 2.5
      c3: x1 - x2 <= 0
      x1, x2 >= 0 integer
    IIS is {c1, c2} after removing c3 and relaxing bounds/integrality.
    """
    with _make_env() as env:
        m = gp.Model(env=env)
        x1 = m.addVar(name="x1", vtype=GRB.INTEGER, lb=0)
        x2 = m.addVar(name="x2", vtype=GRB.INTEGER, lb=0)
        m.addConstr(x1 + x2 <= 1.5, name="c1")
        m.addConstr(x1 + x2 >= 2.5, name="c2")
        m.addConstr(x1 - x2 <= 0, name="c3")
        m.setObjective(x1 + x2)
        mps_file = tmp_path / "sample.mps"
        m.write(str(mps_file))
    return mps_file


@pytest.fixture
def integrality_mps(tmp_path: Path) -> Path:
    """
    Infeasible MIP where integrality is essential:
      c1: 2*x1 + 2*x2 == 3
      x1, x2 >= 0 integer
    IIS is {c1, int_x1, int_x2} after relaxing lower bounds.
    """
    with _make_env() as env:
        m = gp.Model(env=env)
        x1 = m.addVar(name="x1", vtype=GRB.INTEGER, lb=0)
        x2 = m.addVar(name="x2", vtype=GRB.INTEGER, lb=0)
        m.addConstr(2 * x1 + 2 * x2 == 3, name="c1")
        m.setObjective(x1 + x2)
        mps_file = tmp_path / "integrality.mps"
        m.write(str(mps_file))
    return mps_file


def _write_sol(tmp_path: Path, name: str, doc: dict) -> Path:
    sol_file = tmp_path / name
    sol_file.write_text(json.dumps(doc, indent=2))
    return sol_file


# ---------------------------------------------------------------------------
# Core infeasibility tests
# ---------------------------------------------------------------------------


def test_valid_minimal_with_certificates(sample_mps: Path, tmp_path: Path):
    """IIS {c1, c2}: relax c3, bounds, integers; provide certificates for c1 and c2."""
    # Cert for c1: remove c1, only c2 (x1+x2>=2.5) remains → x1=3, x2=0 is feasible.
    # Cert for c2: remove c2, only c1 (x1+x2<=1.5) remains → x1=0, x2=0 is feasible.
    sol_file = _write_sol(
        tmp_path,
        "sol.json",
        {
            "rows": {
                "c3": {"action": "relax"},
                "c1": {"action": "keep", "certificate": {"x1": 3.0, "x2": 0.0}},
                "c2": {"action": "keep", "certificate": {"x1": 0.0, "x2": 0.0}},
            },
            "columns": {
                "x1": {"lower_bound": {"action": "relax"}, "integrality": {"action": "relax"}},
                "x2": {"lower_bound": {"action": "relax"}, "integrality": {"action": "relax"}},
            },
        },
    )
    res = verify_subsystem(sample_mps, sol_file)
    assert res.is_valid_input
    assert res.is_infeasible
    assert res.is_minimal is True
    assert res.is_passed
    assert not res.invalid_certificates
    assert res.removed.rows == 1
    assert res.kept.rows == 2
    # Statistics describe the run that produced the verdict.
    assert res.stats.solver in {"scip", "gurobi"}
    assert res.stats.solver_status == "INFEASIBLE"
    assert res.stats.solve_time is not None and res.stats.solve_time >= 0.0


def test_valid_no_certificates(sample_mps: Path, tmp_path: Path):
    """Same correct IIS subsystem but no certificates: passes (infeasible), minimality unproven."""
    sol_file = _write_sol(
        tmp_path,
        "sol.json",
        {
            "rows": {"c3": {"action": "relax"}},
            "columns": {
                "x1": {"lower_bound": {"action": "relax"}, "integrality": {"action": "relax"}},
                "x2": {"lower_bound": {"action": "relax"}, "integrality": {"action": "relax"}},
            },
        },
    )
    res = verify_subsystem(sample_mps, sol_file)
    assert res.is_valid_input
    assert res.is_infeasible
    assert res.is_minimal is False  # nothing claimed, so nothing proven
    assert res.is_passed


def test_feasible_subsystem(sample_mps: Path, tmp_path: Path):
    """Remove c2 and c3 — only c1 remains, which is feasible."""
    sol_file = _write_sol(
        tmp_path,
        "sol.json",
        {"rows": {"c2": {"action": "relax"}, "c3": {"action": "relax"}}},
    )
    res = verify_subsystem(sample_mps, sol_file)
    assert res.is_valid_input
    assert not res.is_infeasible
    assert not res.is_passed


def test_redundant_element_kept_without_certificate_still_passes(sample_mps: Path, tmp_path: Path):
    """Keeping the redundant c3 is not minimal, but missing certificates are not penalised.

    There is no solver-based minimality check: with nothing claimed about c3 the
    checker has no evidence either way, so minimality is simply unproven and the
    submission passes on infeasibility alone.
    """
    sol_file = _write_sol(
        tmp_path,
        "sol.json",
        {
            "columns": {
                "x1": {"lower_bound": {"action": "relax"}, "integrality": {"action": "relax"}},
                "x2": {"lower_bound": {"action": "relax"}, "integrality": {"action": "relax"}},
            },
        },
    )
    res = verify_subsystem(sample_mps, sol_file)
    assert res.is_valid_input
    assert res.is_infeasible
    assert res.is_minimal is False
    assert not res.invalid_certificates
    assert res.is_passed


def test_certificate_for_redundant_element_cannot_verify(sample_mps: Path, tmp_path: Path):
    """Claiming the redundant c3 is essential is caught: no valid certificate can exist.

    A certificate for c3 must be feasible for {c1, c2}, and that pair is itself
    infeasible (x1+x2 <= 1.5 and x1+x2 >= 2.5).  So whatever point is offered must
    fail: the claim is refuted and minimality is left unproven.
    """
    sol_file = _write_sol(
        tmp_path,
        "sol.json",
        {
            "rows": {
                # c1 removed leaves {c2, c3}: (1.25, 1.25) satisfies both.
                "c1": {"action": "keep", "certificate": {"x1": 1.25, "x2": 1.25}},
                # c2 removed leaves {c1, c3}: (0, 0) satisfies both.
                "c2": {"action": "keep", "certificate": {"x1": 0.0, "x2": 0.0}},
                # c3 removed leaves {c1, c2}: infeasible, so this must fail.
                "c3": {"action": "keep", "certificate": {"x1": 0.0, "x2": 0.0}},
            },
            "columns": {
                "x1": {"lower_bound": {"action": "relax"}, "integrality": {"action": "relax"}},
                "x2": {"lower_bound": {"action": "relax"}, "integrality": {"action": "relax"}},
            },
        },
    )
    res = verify_subsystem(sample_mps, sol_file)
    assert res.is_valid_input
    assert res.is_infeasible
    invalid_labels = [e for e, _ in res.invalid_certificates]
    assert any("c3" in lbl for lbl in invalid_labels)
    assert res.is_minimal is False
    assert not res.is_passed


def test_invalid_certificate_detected(sample_mps: Path, tmp_path: Path):
    """Providing a wrong certificate (point is not feasible) must be detected and fail."""
    # The certificate for c1 is wrong: x1=0, x2=0 does NOT satisfy c2 (x1+x2>=2.5).
    sol_file = _write_sol(
        tmp_path,
        "sol.json",
        {
            "rows": {
                "c3": {"action": "relax"},
                "c1": {"action": "keep", "certificate": {"x1": 0.0, "x2": 0.0}},
                "c2": {"action": "keep", "certificate": {"x1": 0.0, "x2": 0.0}},
            },
            "columns": {
                "x1": {"lower_bound": {"action": "relax"}, "integrality": {"action": "relax"}},
                "x2": {"lower_bound": {"action": "relax"}, "integrality": {"action": "relax"}},
            },
        },
    )
    res = verify_subsystem(sample_mps, sol_file)
    assert res.is_valid_input
    assert res.is_infeasible
    assert len(res.invalid_certificates) >= 1
    # At least the certificate for c1 fails (x1+x2=0 does not satisfy c2: x1+x2>=2.5)
    invalid_labels = [e for e, _ in res.invalid_certificates]
    assert any("c1" in lbl for lbl in invalid_labels)
    assert not res.is_passed  # invalid certificate → fail
    assert res.is_minimal is False


# ---------------------------------------------------------------------------
# Integrality tests
# ---------------------------------------------------------------------------


def test_integrality_essential_with_certificates(integrality_mps: Path, tmp_path: Path):
    """IIS {c1, int_x1, int_x2}: relax lower bounds; provide certificates for all kept elements.

    Kept: c1 (2*x1+2*x2==3), int_x1, int_x2.  Bounds all relaxed.
    Cert for c1:     remove c1, prove {int_x1, int_x2} feasible → (x1=0, x2=0).
    Cert for int_x1: remove int_x1, prove {c1, int_x2} feasible → x1=1.5, x2=0.
    Cert for int_x2: remove int_x2, prove {c1, int_x1} feasible → x1=0, x2=1.5.
    """
    sol_file = _write_sol(
        tmp_path,
        "sol_int.json",
        {
            "rows": {
                "c1": {
                    "action": "keep",
                    "certificate": {"x1": 0.0, "x2": 0.0},
                }
            },
            "columns": {
                "x1": {
                    "lower_bound": {"action": "relax"},
                    "integrality": {"action": "keep", "certificate": {"x1": 1.5, "x2": 0.0}},
                },
                "x2": {
                    "lower_bound": {"action": "relax"},
                    "integrality": {"action": "keep", "certificate": {"x1": 0.0, "x2": 1.5}},
                },
            },
        },
    )
    res = verify_subsystem(integrality_mps, sol_file)
    assert res.is_valid_input
    assert res.is_infeasible
    assert not res.invalid_certificates
    assert res.is_minimal is True
    assert res.is_passed


def test_integrality_relaxed_becomes_feasible(integrality_mps: Path, tmp_path: Path):
    """Relaxing x1's integrality makes the system feasible → not infeasible."""
    sol_file = _write_sol(
        tmp_path,
        "sol_relaxed.json",
        {
            "columns": {
                "x1": {"lower_bound": {"action": "relax"}, "integrality": {"action": "relax"}},
                "x2": {"lower_bound": {"action": "relax"}},
            },
        },
    )
    res = verify_subsystem(integrality_mps, sol_file)
    assert res.is_valid_input
    assert not res.is_infeasible
    assert not res.is_passed


# ---------------------------------------------------------------------------
# Input validation tests
# ---------------------------------------------------------------------------


def test_invalid_row_name(sample_mps: Path, tmp_path: Path):
    sol_file = _write_sol(
        tmp_path,
        "sol.json",
        {"rows": {"non_existent_row": {"action": "relax"}}},
    )
    res = verify_subsystem(sample_mps, sol_file)
    assert not res.is_valid_input
    assert not res.is_valid_solution
    assert not res.is_passed
    assert len(res.errors) > 0


def test_nan_certificate_cannot_fake_a_minimality_proof(sample_mps: Path, tmp_path: Path):
    """NaN compares false against every bound, so it must be rejected outright.

    The subsystem here keeps the redundant c3, so accepting NaN certificates would
    report a proven IIS for a demonstrably non-minimal subsystem.
    """
    sol_file = tmp_path / "nan.json"
    sol_file.write_text(
        """{"rows": {
             "c1": {"action": "keep", "certificate": {"x1": NaN, "x2": NaN}},
             "c2": {"action": "keep", "certificate": {"x1": NaN, "x2": NaN}},
             "c3": {"action": "keep", "certificate": {"x1": NaN, "x2": NaN}}},
           "columns": {"x1": {"lower_bound": {"action": "relax"},
                              "integrality": {"action": "relax"}},
                       "x2": {"lower_bound": {"action": "relax"},
                              "integrality": {"action": "relax"}}}}"""
    )
    res = verify_subsystem(sample_mps, sol_file)
    assert not res.is_valid_input
    assert not res.is_passed
    assert res.is_minimal is not True
    assert any("NaN" in e for e in res.errors)


def test_infinite_certificate_value_is_rejected(sample_mps: Path, tmp_path: Path):
    sol_file = tmp_path / "inf.json"
    sol_file.write_text('{"rows": {"c1": {"action": "keep", "certificate": {"x1": 1e400}}}}')
    res = verify_subsystem(sample_mps, sol_file)
    assert not res.is_valid_input
    assert not res.is_passed


def test_unknown_top_level_key_is_rejected(sample_mps: Path, tmp_path: Path):
    """A typo'd section would otherwise be silently dropped, keeping every element."""
    sol_file = _write_sol(tmp_path, "typo.json", {"row": {"c3": {"action": "relax"}}})
    res = verify_subsystem(sample_mps, sol_file)
    assert not res.is_valid_input
    assert not res.is_valid_solution
    assert any("unknown top-level key" in e for e in res.errors)


def test_misspelled_bound_side_is_rejected(sample_mps: Path, tmp_path: Path):
    sol_file = _write_sol(tmp_path, "side.json", {"columns": {"x1": {"lb": {"action": "relax"}}}})
    res = verify_subsystem(sample_mps, sol_file)
    assert not res.is_valid_input
    assert any("unknown key" in e for e in res.errors)


@pytest.mark.parametrize(
    "text,error",
    [
        # A misspelt key would be dropped: this would read as keep.
        ('{"rows": {"c3": {"actoin": "relax"}}}', "unknown key(s) ['actoin']"),
        ('{"rows": {"c1": {"certficate": {"x1": 1}}}}', "unknown key(s) ['certficate']"),
        # float() takes these; JSON numbers they are not.
        ('{"rows": {"c1": {"certificate": {"x1": "1"}}}}', "must be a number, got '1'"),
        ('{"rows": {"c1": {"certificate": {"x1": true}}}}', "must be a number, got True"),
        ('{"rows": {"c1": {"certificate": {"x1": null}}}}', "must be a number, got None"),
        ('{"rows": {"c1": {"certificate": null}}}', "must be a JSON object"),
        # Only a kept element is certified.
        ('{"rows": {"c3": {"action": "relax", "certificate": {}}}}', "only for a kept element"),
        # The sections are objects keyed by name.
        ('{"rows": null}', "'rows' must be a JSON object"),
        ('{"columns": []}', "'columns' must be a JSON object"),
        # json.loads would silently keep the last of the two.
        ('{"rows": {"c3": {"action": "relax"}, "c3": {"action": "keep"}}}', "['c3']"),
    ],
    ids=[
        "misspelt-action",
        "misspelt-certificate",
        "string-value",
        "bool-value",
        "null-value",
        "null-certificate",
        "certificate-on-relax",
        "null-rows",
        "list-columns",
        "duplicate-key",
    ],
)
def test_an_entry_that_is_not_what_the_format_says_is_rejected(
    sample_mps: Path, tmp_path: Path, text: str, error: str
):
    sol_file = tmp_path / "bad.json"
    sol_file.write_text(text)
    res = verify_subsystem(sample_mps, sol_file)
    assert not res.is_valid_solution and not res.is_passed
    assert any(error in e for e in res.errors), res.errors


def test_certificate_naming_unknown_variable_is_invalid(sample_mps: Path, tmp_path: Path):
    """A typo'd variable name means the verified point is not the intended one."""
    sol_file = _write_sol(
        tmp_path,
        "ghost.json",
        {
            "rows": {
                "c3": {"action": "relax"},
                "c1": {"action": "keep", "certificate": {"x1": 3.0, "xx2": 0.0}},
                "c2": {"action": "keep", "certificate": {"x1": 0.0, "x2": 0.0}},
            },
            "columns": {
                "x1": {"lower_bound": {"action": "relax"}, "integrality": {"action": "relax"}},
                "x2": {"lower_bound": {"action": "relax"}, "integrality": {"action": "relax"}},
            },
        },
    )
    res = verify_subsystem(sample_mps, sol_file)
    assert res.is_infeasible
    assert [e for e, _ in res.invalid_certificates] == ["row:c1"]
    assert not res.is_passed


def test_tolerance_is_configurable(sample_mps: Path, tmp_path: Path):
    """A slightly-violating point is rejected at the default tolerance, accepted at a loose one."""
    # Removing c1 leaves c2 (x1+x2 >= 2.5); 2.4999 misses it by 1e-4.
    sol_file = _write_sol(
        tmp_path,
        "tol.json",
        {
            "rows": {
                "c3": {"action": "relax"},
                "c1": {"action": "keep", "certificate": {"x1": 2.4999, "x2": 0.0}},
            },
            "columns": {
                "x1": {"lower_bound": {"action": "relax"}, "integrality": {"action": "relax"}},
                "x2": {"lower_bound": {"action": "relax"}, "integrality": {"action": "relax"}},
            },
        },
    )
    assert verify_subsystem(sample_mps, sol_file).invalid_certificates
    assert not verify_subsystem(sample_mps, sol_file, tolerance=1e-3).invalid_certificates

    for bad in (0.0, -1e-5, 1.1e-2, float("inf"), float("nan")):
        with pytest.raises(ValueError):
            verify_subsystem(sample_mps, sol_file, tolerance=bad)


def test_gzip_compressed_solution_file(sample_mps: Path, tmp_path: Path):
    """Solution files may be gzip-compressed (.json.gz) and should be parsed correctly."""
    sol_dict = {
        "rows": {
            "c3": {"action": "relax"},
            "c1": {"action": "keep", "certificate": {"x1": 3.0, "x2": 0.0}},
            "c2": {"action": "keep", "certificate": {"x1": 0.0, "x2": 0.0}},
        },
        "columns": {
            "x1": {"lower_bound": {"action": "relax"}, "integrality": {"action": "relax"}},
            "x2": {"lower_bound": {"action": "relax"}, "integrality": {"action": "relax"}},
        },
    }

    # Write as gzip-compressed JSON
    sol_gz_file = tmp_path / "sol.json.gz"
    with gzip.open(sol_gz_file, "wt", encoding="utf-8") as f:
        json.dump(sol_dict, f)

    # Verify it works the same as uncompressed
    res = verify_subsystem(sample_mps, sol_gz_file)
    assert res.is_valid_input
    assert res.is_infeasible
    assert res.is_minimal is True
    assert res.is_passed


# ---------------------------------------------------------------------------
# Solver plumbing: failures are reported, limits are honoured
# ---------------------------------------------------------------------------


def _minimal_sol(tmp_path: Path) -> Path:
    """A correct solution for sample_mps: relax c3, the lower bounds and integrality."""
    return _write_sol(
        tmp_path,
        "sol.json",
        {
            "rows": {"c3": {"action": "relax"}},
            "columns": {
                "x1": {"lower_bound": {"action": "relax"}, "integrality": {"action": "relax"}},
                "x2": {"lower_bound": {"action": "relax"}, "integrality": {"action": "relax"}},
            },
        },
    )


def test_solver_crash_is_reported_not_raised(sample_mps: Path, tmp_path: Path):
    """A solver blowing up must not take the checker down with it.

    An unusable solver is no evidence against the submission, so the verdict is
    undetermined (is_infeasible=None) rather than a failure.
    """
    sol_file = _minimal_sol(tmp_path)
    with mock.patch.object(
        GurobipyBackend, "solve_subsystem", side_effect=RuntimeError("solver exploded")
    ):
        res = verify_subsystem(sample_mps, sol_file)
    assert res.is_infeasible is None
    assert not res.is_passed
    assert any("solver exploded" in e for e in res.errors)
    # No solve completed, so there is no time to report.
    assert res.stats.solve_time is None
    assert res.stats.solver_status == "ERROR"


def test_time_limit_reaches_the_backend(sample_mps: Path, tmp_path: Path):
    """verify_subsystem(time_limit=...) must actually bound the solve."""
    sol_file = _minimal_sol(tmp_path)
    seen = {}
    original = GurobipyBackend.solve_subsystem

    def spy(self, *args, **kwargs):
        seen.update(kwargs)
        return original(self, *args, **kwargs)

    with mock.patch.object(GurobipyBackend, "solve_subsystem", spy):
        res = verify_subsystem(sample_mps, sol_file, time_limit=42.0)
    assert seen.get("time_limit") == 42.0
    assert res.is_infeasible


def test_unknown_solver_is_a_clean_input_error(sample_mps: Path, tmp_path: Path):
    """An unrecognised --solver must report, not traceback."""
    sol_file = _minimal_sol(tmp_path)
    res = verify_subsystem(sample_mps, sol_file, solver="nosuchsolver")
    assert not res.is_valid_input
    assert res.is_valid_solution  # the environment's fault, not the submitter's
    assert not res.is_passed
    assert any("nosuchsolver" in e for e in res.errors)


def test_certificates_are_checked_even_without_an_infeasibility_verdict(
    sample_mps: Path, tmp_path: Path
):
    """A solve that reaches no verdict must not silence certificate verification.

    Certificates are checked against the submitted subsystem, not against what the
    solver said about it, so a timeout still reports minimality and any wrong
    certificate instead of hiding both behind is_infeasible=None.
    """
    # The certificate for c1 is wrong: (0, 0) does not satisfy c2 (x1+x2 >= 2.5).
    sol_file = _write_sol(
        tmp_path,
        "sol.json",
        {
            "rows": {
                "c3": {"action": "relax"},
                "c1": {"action": "keep", "certificate": {"x1": 0.0, "x2": 0.0}},
                "c2": {"action": "keep", "certificate": {"x1": 0.0, "x2": 0.0}},
            },
            "columns": {
                "x1": {"lower_bound": {"action": "relax"}, "integrality": {"action": "relax"}},
                "x2": {"lower_bound": {"action": "relax"}, "integrality": {"action": "relax"}},
            },
        },
    )
    with mock.patch.object(GurobipyBackend, "solve_subsystem", return_value=(STATUS_UNKNOWN, 1.0)):
        res = verify_subsystem(sample_mps, sol_file)

    assert res.is_infeasible is None
    assert any("c1" in label for label, _ in res.invalid_certificates)
    assert res.is_minimal is False
    assert not res.is_passed


# ---------------------------------------------------------------------------
# The package surface only advertises what is installed
# ---------------------------------------------------------------------------


def test_star_import_only_offers_installed_backends():
    """``from mipcc_iis_checker import *`` must work with one solver installed, or none.

    The names in ``__all__`` are the promise; a backend whose solver is missing
    cannot be one of them, or the star import raises AttributeError on a machine
    that has only the other solver.
    """
    import mipcc_iis_checker

    namespace = {}
    exec("from mipcc_iis_checker import *", namespace)  # noqa: S102 - the behaviour under test

    for name in mipcc_iis_checker.__all__:
        assert name in namespace, f"__all__ advertises {name}, which the star import misses"

    assert ("GurobipyBackend" in mipcc_iis_checker.__all__) is mipcc_iis_checker.HAS_GUROBI
    assert ("ScipBackend" in mipcc_iis_checker.__all__) is mipcc_iis_checker.HAS_SCIP


# ---------------------------------------------------------------------------
# The report describes validity and measurements, nothing else
# ---------------------------------------------------------------------------


def test_report_carries_exactly_the_measurements_taken(sample_mps: Path, tmp_path: Path):
    """The stats payload is the three measurements and no more.

    Pinning the key set means a field added to the report has to be a measurement
    the checker actually took, rather than an interpretation of one.
    """
    res = verify_subsystem(sample_mps, _minimal_sol(tmp_path))
    report = res.to_dict()

    assert set(report["stats"]) == {"solver", "solver_status", "solve_time"}
    assert report["stats"]["solver_status"] == "INFEASIBLE"
    assert report["stats"]["solve_time"] is not None


# ---------------------------------------------------------------------------
# The four element sets are read off the model, not tracked by the backend
# ---------------------------------------------------------------------------


def test_element_sets_come_from_the_model(sample_mps: Path):
    """rows/bounds/integers are derived from NpModel, so a subsystem reports its own.

    They used to be sets each backend rebuilt during load_mps, which meant the
    same loop existed twice and a subsystem could not answer the question at all.
    """
    backend = GurobipyBackend()
    backend.load_mps(str(sample_mps))

    assert backend.np_model.rows == {"c1", "c2", "c3"}
    assert backend.np_model.integers == {"x1", "x2"}
    assert backend.np_model.lower_bounds == {"x1", "x2"}  # lb=0 on both
    assert backend.np_model.upper_bounds == frozenset()  # no finite upper bounds

    # The backend just forwards to the parsed model.
    assert backend.np_model.rows == backend.np_model.rows

    # A subsystem answers for itself: sub_system() drops relaxed rows and pushes
    # relaxed bounds to ±inf, which is exactly what these properties read.
    sub = backend.np_model.sub_system({"c1", "c2"}, {"x1"}, frozenset(), frozenset())
    assert sub.rows == {"c1", "c2"}
    assert sub.lower_bounds == {"x1"}
    assert sub.integers == frozenset()


def test_element_counts_come_from_the_model(sample_mps: Path):
    """A model counts its own elements; 'bounds' counts each side separately."""
    backend = GurobipyBackend()
    backend.load_mps(str(sample_mps))

    counts = backend.np_model.element_counts
    assert (counts.rows, counts.bounds, counts.integers) == (3, 2, 2)
    assert counts.total == 7

    # ElementCounts.of is the one place "bounds = lower + upper" is decided.
    assert ElementCounts.of({"c1"}, {"x1"}, {"x1"}, set()) == ElementCounts(1, 2, 0)


def test_element_sets_before_load_are_an_error():
    """Reporting an empty instance before load_mps would be a lie, so it raises."""
    backend = GurobipyBackend()
    with pytest.raises(RuntimeError, match="No instance loaded"):
        _ = backend.np_model


# ---------------------------------------------------------------------------
# Instances NpModel cannot represent must be refused, not partially loaded
# ---------------------------------------------------------------------------


def _quadratic_constraint(m: gp.Model) -> None:
    x = m.addVar(name="x")
    y = m.addVar(name="y")
    m.addConstr(x * x + y * y <= 1, name="q1")
    m.addConstr(x + y >= 3, name="c1")


def _sos_constraint(m: gp.Model) -> None:
    x = m.addVar(name="x", ub=10)
    y = m.addVar(name="y", ub=10)
    m.addConstr(x + y >= 3, name="c1")
    m.addSOS(GRB.SOS_TYPE1, [x, y])


def _indicator_constraint(m: gp.Model) -> None:
    b = m.addVar(name="b", vtype=GRB.BINARY)
    x = m.addVar(name="x", ub=10)
    m.addConstr(x >= 3, name="c1")
    m.addConstr(b >= 1, name="c2")
    m.addGenConstrIndicator(b, True, x <= 1, name="ind1")


def _semi_continuous_var(m: gp.Model) -> None:
    x = m.addVar(name="x", vtype=GRB.SEMICONT, lb=2, ub=10)
    m.addConstr(x <= 1.5, name="c1")
    m.addConstr(x >= 0.5, name="c2")


def _semi_integer_var(m: gp.Model) -> None:
    x = m.addVar(name="x", vtype=GRB.SEMIINT, lb=2, ub=10)
    m.addConstr(x <= 1.5, name="c1")


@pytest.mark.parametrize(
    "name,builder,expected",
    [
        # The first three arrive in a section of their own, so the whitelist refuses them
        # before the file is parsed.  Semi-continuous/-integer are BOUNDS types that no
        # section header reveals, so only the parsed variable types catch those.
        ("quad", _quadratic_constraint, "QCMATRIX section"),
        ("sos", _sos_constraint, "SOS section"),
        ("indicator", _indicator_constraint, "INDICATORS section"),
        ("semicont", _semi_continuous_var, "semi-continuous variables"),
        ("semiint", _semi_integer_var, "semi-integer variables"),
    ],
)
def test_unrepresentable_features_are_refused(tmp_path: Path, name, builder, expected):
    """NpModel holds only linear rows over bounded/integer variables.

    Loading such an instance anyway would drop the restriction it cannot store,
    making every subsystem a relaxation of the real one — the checker would then
    accept certificates that violate the original instance.  Each of these
    features survives an MPS round-trip, so the guard has to catch them on load.
    """
    with _make_env() as env:
        m = gp.Model(env=env)
        builder(m)
        m.update()
        mps_file = tmp_path / f"{name}.mps"
        m.write(str(mps_file))

    backend = GurobipyBackend()
    with pytest.raises(UnsupportedInstanceError) as excinfo:
        backend.load_mps(str(mps_file))
    assert expected in str(excinfo.value)

    # Via the CLI entry point this is an input error, not a crash or a pass.
    sol_file = _write_sol(tmp_path, f"{name}.json", {"rows": {"c1": {"action": "relax"}}})
    res = verify_subsystem(mps_file, sol_file)
    assert not res.is_valid_input
    assert not res.is_passed
    assert any(expected in e for e in res.errors)


# ---------------------------------------------------------------------------
# Per-kind certificate counts (res.certified)
# ---------------------------------------------------------------------------


def test_certified_counts_only_certificates_that_verified(sample_mps: Path, tmp_path: Path):
    """A certificate that fails algebraically is a claim, not a proof.

    c1 and c2 verify; no certificate for c3 can, since {c1, c2} is itself
    infeasible.  Counting it would make the submission look fully proved.
    """
    sol_file = _write_sol(
        tmp_path,
        "sol.json",
        {
            "rows": {
                "c1": {"action": "keep", "certificate": {"x1": 1.25, "x2": 1.25}},
                "c2": {"action": "keep", "certificate": {"x1": 0.0, "x2": 0.0}},
                "c3": {"action": "keep", "certificate": {"x1": 0.0, "x2": 0.0}},
            },
            "columns": {
                "x1": {"lower_bound": {"action": "relax"}, "integrality": {"action": "relax"}},
                "x2": {"lower_bound": {"action": "relax"}, "integrality": {"action": "relax"}},
            },
        },
    )
    res = verify_subsystem(sample_mps, sol_file)
    assert res.kept.rows == 3
    assert res.certified.rows == 2  # c3's certificate cannot verify
    assert len(res.invalid_certificates) == 1
    assert res.certified.total == 2


def test_certified_is_zero_without_certificates(sample_mps: Path, tmp_path: Path):
    """Kept-but-uncertified is the normal case for a submission with no certificates."""
    sol_file = _write_sol(
        tmp_path,
        "sol.json",
        {
            "rows": {"c3": {"action": "relax"}},
            "columns": {
                "x1": {"lower_bound": {"action": "relax"}, "integrality": {"action": "relax"}},
                "x2": {"lower_bound": {"action": "relax"}, "integrality": {"action": "relax"}},
            },
        },
    )
    res = verify_subsystem(sample_mps, sol_file)
    assert res.kept.rows == 2
    assert res.certified == ElementCounts(0, 0, 0)


def test_certified_counts_rows_and_integers_separately(integrality_mps: Path, tmp_path: Path):
    """Rows and integralities are counted apart, not lumped into one total.

    No bound element is kept here (no finite upper bounds, lower bounds relaxed),
    so :func:`test_certified_counts_certified_bounds` covers that counter.
    """
    sol_file = _write_sol(
        tmp_path,
        "sol_int.json",
        {
            "rows": {"c1": {"action": "keep", "certificate": {"x1": 0.0, "x2": 0.0}}},
            "columns": {
                "x1": {
                    "lower_bound": {"action": "relax"},
                    "integrality": {"action": "keep", "certificate": {"x1": 1.5, "x2": 0.0}},
                },
                "x2": {
                    "lower_bound": {"action": "relax"},
                    "integrality": {"action": "keep", "certificate": {"x1": 0.0, "x2": 1.5}},
                },
            },
        },
    )
    res = verify_subsystem(integrality_mps, sol_file)
    assert res.is_passed
    assert res.certified == ElementCounts(rows=1, bounds=0, integers=2)
    assert res.kept == ElementCounts(rows=1, bounds=0, integers=2)


def test_certified_counts_certified_bounds(tmp_path: Path):
    """A bound in the IIS is counted under bounds, per bound constraint.

    Instance: c1 (x1 + x2 >= 2.5) with x1 <= 1 and x2 <= 1, whose IIS is c1 plus
    both upper bounds.  Two variables bounded on one side each → certified.bounds 2.
    """
    with _make_env() as env:
        m = gp.Model(env=env)
        x1 = m.addVar(name="x1", vtype=GRB.INTEGER, lb=0, ub=1)
        x2 = m.addVar(name="x2", vtype=GRB.INTEGER, lb=0, ub=1)
        m.addConstr(x1 + x2 >= 2.5, name="c1")
        m.setObjective(x1 + x2)
        mps_file = tmp_path / "bounds.mps"
        m.write(str(mps_file))

    sol_file = _write_sol(
        tmp_path,
        "sol_bounds.json",
        {
            "rows": {"c1": {"action": "keep", "certificate": {"x1": 0.0, "x2": 0.0}}},
            "columns": {
                "x1": {
                    "lower_bound": {"action": "relax"},
                    "integrality": {"action": "relax"},
                    "upper_bound": {"action": "keep", "certificate": {"x1": 1.5, "x2": 1.0}},
                },
                "x2": {
                    "lower_bound": {"action": "relax"},
                    "integrality": {"action": "relax"},
                    "upper_bound": {"action": "keep", "certificate": {"x1": 1.0, "x2": 1.5}},
                },
            },
        },
    )
    res = verify_subsystem(mps_file, sol_file)
    assert res.is_passed, res.errors
    assert res.certified.bounds == 2
    assert res.certified == ElementCounts(rows=1, bounds=2, integers=0)
    assert res.certified == res.kept


def test_certified_equals_kept_when_minimal(sample_mps: Path, tmp_path: Path):
    """The invariant the 'cert' columns are read against: min=Y means cert == kept."""
    sol_file = _write_sol(
        tmp_path,
        "sol.json",
        {
            "rows": {
                "c3": {"action": "relax"},
                "c1": {"action": "keep", "certificate": {"x1": 3.0, "x2": 0.0}},
                "c2": {"action": "keep", "certificate": {"x1": 0.0, "x2": 0.0}},
            },
            "columns": {
                "x1": {"lower_bound": {"action": "relax"}, "integrality": {"action": "relax"}},
                "x2": {"lower_bound": {"action": "relax"}, "integrality": {"action": "relax"}},
            },
        },
    )
    res = verify_subsystem(sample_mps, sol_file)
    assert res.is_minimal is True
    assert res.certified == res.kept


def test_certified_is_reported_in_to_dict(sample_mps: Path, tmp_path: Path):
    """The counts reach the JSON the benchmark and the CLI both read."""
    sol_file = _write_sol(
        tmp_path,
        "sol.json",
        {
            "rows": {
                "c3": {"action": "relax"},
                "c1": {"action": "keep", "certificate": {"x1": 3.0, "x2": 0.0}},
                "c2": {"action": "keep", "certificate": {"x1": 0.0, "x2": 0.0}},
            },
            "columns": {
                "x1": {"lower_bound": {"action": "relax"}, "integrality": {"action": "relax"}},
                "x2": {"lower_bound": {"action": "relax"}, "integrality": {"action": "relax"}},
            },
        },
    )
    counts = verify_subsystem(sample_mps, sol_file).to_dict()["counts"]
    assert counts["certified_rows"] == 2
    assert counts["certified_bounds"] == 0
    assert counts["certified_integers"] == 0
    assert counts["total_certified"] == 2


def _write(path: Path, text: str) -> Path:
    path.write_text(text)
    return path


def test_a_certificate_may_not_satisfy_the_element_it_certifies(tmp_path: Path):
    """A certificate for kept *e* is a feasible point for the subsystem minus *e*.

    If it satisfies *e* as well it satisfies the whole subsystem, contradicting the
    infeasibility the same run just established -- so it proves nothing.  Here the
    two conflicting rows are scaled by 1e-8, which puts a full unit of x-space
    violation inside the absolute row tolerance; c3 is plainly redundant, so
    accepting these would report a proven IIS for a non-minimal subsystem.
    """
    mps = _write(
        tmp_path / "scaled.mps",
        "NAME          SCALED\n"
        "ROWS\n"
        " N  OBJ\n"
        " G  c1\n"
        " L  c2\n"
        " L  c3\n"
        "COLUMNS\n"
        "    x         OBJ       1.0        c1        1e-8\n"
        "    x         c2        1e-8       c3        1.0\n"
        "RHS\n"
        "    RHS       c1        1e-8       c2        -1e-8\n"
        "    RHS       c3        5.0\n"
        "BOUNDS\n"
        " FR BND       x\n"
        "ENDATA\n",
    )
    sol = _write(
        tmp_path / "scaled.json",
        json.dumps(
            {
                "rows": {
                    name: {"action": "keep", "certificate": {"x": 0.0}}
                    for name in ("c1", "c2", "c3")
                }
            }
        ),
    )

    res = verify_subsystem(mps, sol)

    assert res.is_infeasible is True
    assert res.is_minimal is False
    assert not res.is_passed
    assert len(res.invalid_certificates) == 3


def test_a_huge_certificate_value_cannot_cancel_a_row(tmp_path: Path):
    """1e300 is finite, so io.py's NaN/inf guard lets it through.

    Row k2 accumulates ``1 + 1e300 - 1e300``, and the 1 is below the ulp of 1e300,
    so a row whose true activation is 1 computes as exactly 0.  The kept set is not
    an IIS -- pad and x's bound are redundant -- so this must not read as proven.
    """
    mps = _write(
        tmp_path / "cancel.mps",
        "NAME          CANCEL\n"
        "ROWS\n"
        " N  OBJ\n"
        " G  k1\n"
        " L  k2\n"
        " E  k3\n"
        " L  pad\n"
        "COLUMNS\n"
        "    a         OBJ       1.0        k1        1.0\n"
        "    a         k2        1.0        pad       1.0\n"
        "    u         k2        1.0        k3        1.0\n"
        "    v         k2        1.0        k3        1.0\n"
        "RHS\n"
        "    RHS       k1        1.0        k2        0.0\n"
        "    RHS       k3        0.0        pad       5.0\n"
        "BOUNDS\n"
        " FR BND       u\n"
        " FR BND       v\n"
        "ENDATA\n",
    )
    cert = {"a": 1.0, "u": 1e300, "v": -1e300}
    sol = _write(
        tmp_path / "cancel.json",
        json.dumps(
            {
                "rows": {
                    name: {"action": "keep", "certificate": cert}
                    for name in ("k1", "k2", "k3", "pad")
                }
            }
        ),
    )

    res = verify_subsystem(mps, sol)

    assert res.is_minimal is not True
    assert not res.is_passed
    # Caught by the arithmetic, not by some other check: k2's true activity is 1.
    violations = dict(res.invalid_certificates)
    assert "constraint 'k2': lhs 1 > rhs 0 (violation 1)" in violations["row:pad"]


def test_relaxing_every_element_is_not_a_minimality_proof(sample_mps: Path, tmp_path: Path):
    """certified.total == kept.total holds vacuously at zero and proves nothing."""
    sol = _write(
        tmp_path / "relaxall.json",
        json.dumps(
            {
                "rows": {name: {"action": "relax"} for name in ("c1", "c2", "c3")},
                "columns": {
                    name: {
                        "lower_bound": {"action": "relax"},
                        "integrality": {"action": "relax"},
                    }
                    for name in ("x1", "x2")
                },
            }
        ),
    )

    res = verify_subsystem(sample_mps, sol)

    assert res.kept.total == 0
    assert res.is_minimal is False


def test_the_library_refuses_a_non_mps_instance(tmp_path: Path):
    """The CLI checked this, so a library caller handed a submitter-controlled path
    got whatever the installed backend's reader made of an .lp file."""
    lp = _write(tmp_path / "inst.lp", "Minimize\n obj: x\nSubject To\n c1: x >= 1\nEnd\n")
    sol = _write(tmp_path / "sol.json", json.dumps({"rows": {}}))

    res = verify_subsystem(lp, sol)

    assert not res.is_valid_input
    assert not res.is_passed
    assert any("MPS" in e for e in res.errors)


def test_the_report_says_which_checker_and_tolerance_produced_it(sample_mps: Path, tmp_path: Path):
    """A verdict is only comparable with another from the same checker and tolerance."""
    from mipcc_iis_checker import __version__

    sol_file = _write_sol(tmp_path, "empty.json", {})
    res = verify_subsystem(sample_mps, sol_file, tolerance=1e-3)
    report = res.to_dict()
    assert report["checker_version"] == __version__ != "unknown"
    assert report["tolerance"] == 1e-3
    assert f"mipcc-iis-checker {__version__}" in res.summary()


# ---------------------------------------------------------------------------
# Bounds of variables that stay integer (j in J) always belong to the subsystem
# ---------------------------------------------------------------------------

# x integer in [0, 10], y free; c1: x + y >= 20, c2: y <= 5.  Infeasible only
# because of x's upper bound (x + y <= 15), which is a bound of an integer variable.
_INT_BOUND_MPS = (
    "NAME          INTBOUND\n"
    "ROWS\n"
    " N  OBJ\n"
    " G  c1\n"
    " L  c2\n"
    "COLUMNS\n"
    "    MARKER    'MARKER'  'INTORG'\n"
    "    x         OBJ       1.0        c1        1.0\n"
    "    MARKER    'MARKER'  'INTEND'\n"
    "    y         c1        1.0        c2        1.0\n"
    "RHS\n"
    "    RHS       c1        20.0       c2        5.0\n"
    "BOUNDS\n"
    " LO BND       x         0.0\n"
    " UP BND       x         10.0\n"
    " FR BND       y\n"
    "ENDATA\n"
)


def test_a_relaxed_bound_of_an_integer_variable_still_applies(tmp_path: Path):
    """Relaxing x's bounds while x stays integer changes nothing: they still apply.

    Without them the subsystem would be feasible (x = 15, y = 5); with them it is not.
    """
    mps = _write(tmp_path / "intbound.mps", _INT_BOUND_MPS)
    sol = _write_sol(
        tmp_path,
        "relaxed.json",
        {
            "columns": {
                "x": {"lower_bound": {"action": "relax"}, "upper_bound": {"action": "relax"}}
            }
        },
    )
    res = verify_subsystem(mps, sol)

    assert res.is_infeasible is True
    assert res.is_passed
    # Neither bound is an element while x stays integer: not kept, not removed.  The
    # original counts them, as relaxing x's integrality would make them elements.
    assert res.kept == ElementCounts(rows=2, bounds=0, integers=1)
    assert res.removed == ElementCounts(rows=0, bounds=0, integers=0)
    assert res.original == ElementCounts(rows=2, bounds=2, integers=1)
    assert any("bounds of integer variables always apply" in n for n in res.notes)


def test_a_certificate_must_respect_the_bounds_of_integer_variables(tmp_path: Path):
    """x's bounds apply in every certificate check, even when marked relax.

    (20, 0) satisfies c1 once c2 is removed, but breaks x <= 10.  (10, 10) is the
    certificate that respects it.
    """
    mps = _write(tmp_path / "intbound.mps", _INT_BOUND_MPS)

    def solution(cert_c2):
        return {
            "rows": {
                "c1": {"action": "keep", "certificate": {"x": 0.0, "y": 0.0}},
                "c2": {"action": "keep", "certificate": cert_c2},
            },
            "columns": {"x": {"upper_bound": {"action": "relax"}}},
        }

    bad = verify_subsystem(mps, _write_sol(tmp_path, "bad.json", solution({"x": 20.0})))
    assert [e for e, _ in bad.invalid_certificates] == ["row:c2"]
    assert not bad.is_passed

    good = verify_subsystem(mps, _write_sol(tmp_path, "good.json", solution({"x": 10, "y": 10})))
    assert good.is_passed and not good.invalid_certificates
    assert good.certified == ElementCounts(rows=2, bounds=0, integers=0)


def test_a_certificate_on_a_bound_of_an_integer_variable_is_ignored(tmp_path: Path):
    """Such a bound needs no certificate, so a wrong one neither counts nor fails."""
    mps = _write(tmp_path / "intbound.mps", _INT_BOUND_MPS)
    sol = _write_sol(
        tmp_path,
        "ignored.json",
        {"columns": {"x": {"upper_bound": {"action": "keep", "certificate": {"x": -99.0}}}}},
    )
    res = verify_subsystem(mps, sol)

    assert res.is_passed
    assert not res.invalid_certificates
    assert res.certified.bounds == 0
    assert any("need no certificate" in n for n in res.notes)


def test_without_integrality_the_bounds_are_ordinary_elements(tmp_path: Path):
    """Once x's integrality is relaxed, x is not in J and its bounds count again.

    Kept: c1, c2 and x <= 10, each certified, so this is a proven IIS of size 3.
    """
    mps = _write(tmp_path / "intbound.mps", _INT_BOUND_MPS)
    sol = _write_sol(
        tmp_path,
        "continuous.json",
        {
            "rows": {
                "c1": {"action": "keep", "certificate": {"x": 0.0, "y": 0.0}},
                "c2": {"action": "keep", "certificate": {"x": 10.0, "y": 10.0}},
            },
            "columns": {
                "x": {
                    "integrality": {"action": "relax"},
                    "lower_bound": {"action": "relax"},
                    "upper_bound": {"action": "keep", "certificate": {"x": 15.0, "y": 5.0}},
                }
            },
        },
    )
    res = verify_subsystem(mps, sol)

    assert res.is_passed and res.is_minimal
    assert res.kept == ElementCounts(rows=2, bounds=1, integers=0)
    assert res.removed == ElementCounts(rows=0, bounds=1, integers=1)
    assert not res.notes


def test_an_integrality_certificate_keeps_that_variables_bounds(tmp_path: Path):
    """Removing j from J removes only its integrality: x_j's bounds still apply.

    x integer in [0.2, 0.5] has no integer value: the IIS is x's integrality alone
    (its bounds are not elements).  0.3 certifies it; 0.7 leaves the bounds.
    """
    mps = _write(
        tmp_path / "fractional.mps",
        "NAME          FRACTIONAL\n"
        "ROWS\n"
        " N  OBJ\n"
        "COLUMNS\n"
        "    MARKER    'MARKER'  'INTORG'\n"
        "    x         OBJ       1.0\n"
        "    MARKER    'MARKER'  'INTEND'\n"
        "BOUNDS\n"
        " LO BND       x         0.2\n"
        " UP BND       x         0.5\n"
        "ENDATA\n",
    )

    def solution(value):
        return {
            "columns": {
                "x": {
                    "lower_bound": {"action": "relax"},
                    "upper_bound": {"action": "relax"},
                    "integrality": {"action": "keep", "certificate": {"x": value}},
                }
            }
        }

    good = verify_subsystem(mps, _write_sol(tmp_path, "good.json", solution(0.3)))
    assert good.is_passed and good.is_minimal
    assert good.kept == ElementCounts(rows=0, bounds=0, integers=1)

    bad = verify_subsystem(mps, _write_sol(tmp_path, "bad.json", solution(0.7)))
    assert [e for e, _ in bad.invalid_certificates] == ["integer:x"]


def test_original_counts_bound_every_kept_count(tmp_path: Path):
    """Relaxing x's integrality makes its bounds elements; the original still covers them."""
    mps = _write(tmp_path / "intbound.mps", _INT_BOUND_MPS)
    sol = _write_sol(
        tmp_path, "relaxed.json", {"columns": {"x": {"integrality": {"action": "relax"}}}}
    )
    res = verify_subsystem(mps, sol)

    assert res.kept == ElementCounts(rows=2, bounds=2, integers=0)
    assert res.original == ElementCounts(rows=2, bounds=2, integers=1)
