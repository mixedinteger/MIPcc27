"""
Unit tests for how the benchmark classifies and reports a verification.

The distinction these pin down is "was the submission actually measured?" — a
missing solver must not be reported as a verdict, or a broken environment writes
off a whole batch.
"""

import csv
import io
from types import SimpleNamespace

import pytest

from benchmark import run as bench


def _res(solver_status, valid_solution=True):
    """A stand-in for VerificationResult with only the fields _verify_status reads."""
    return SimpleNamespace(
        is_valid_solution=valid_solution, stats=SimpleNamespace(solver_status=solver_status)
    )


@pytest.mark.parametrize(
    "solver_status,valid,status",
    [
        ("INFEASIBLE", True, "ok"),
        ("FEASIBLE", True, "ok"),
        ("UNKNOWN", True, "verify_unknown"),
        ("ERROR", True, "verify_error"),
        (None, True, "verify_error"),  # returned before the solve: no solver, bad instance
        (None, False, "invalid_solution"),  # the submitter's fault, not the harness's
    ],
)
def test_verify_status(solver_status, valid, status):
    assert bench._verify_status(_res(solver_status, valid)) == status


def test_an_exception_is_a_verify_error_with_its_message(monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("line one\nline two " + "x" * 5000)

    monkeypatch.setattr(bench, "verify_subsystem", boom)
    verify = bench._run_verify("i.mps", "s.json", "", None)
    assert verify["status"] == "verify_error"
    assert "line one" in verify["errors"][0] and len(verify["errors"][0]) <= 1000


# --------------------------------------------------------------------------
# pass_mark: the submission's verdict, as distinct from the run's status
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "status,verify,mark",
    [
        ("ok", {"passed": True}, True),
        ("ok", {"passed": False, "infeasible": False}, False),
        ("invalid_solution", {}, False),
        ("solve_timeout", {}, False),
        ("solve_failed", {}, False),
        # Undecided: a time limit is not the submitter's fault ...
        ("verify_unknown", {"infeasible": None}, None),
        # ... unless a certificate failed, which needs no solve to establish.
        ("verify_unknown", {"infeasible": None, "invalid_certificates": 1}, False),
        # Never measured: the harness's problem must not convict a submission.
        ("verify_error", {}, None),
        ("solve_error", {}, None),
        ("no_solution", {}, None),
        ("not_run", {}, None),
        ("stale", {}, None),
    ],
)
def test_pass_mark(status, verify, mark):
    assert bench.pass_mark({"name": "x", "status": status, "verify": verify}) is mark


def test_a_failed_submission_is_a_verdict_and_a_harness_problem_is_not():
    """The exit code: 0 whenever every model reached a verdict, pass or fail."""
    assert {"ok", "invalid_solution", "solve_timeout", "solve_failed"} <= bench.VERDICTS
    harness = {"verify_error", "verify_unknown", "solve_error", "not_run", "stale"}
    assert not harness & bench.VERDICTS


# --------------------------------------------------------------------------
# The printed table, read by column label
# --------------------------------------------------------------------------


def _verify(**kw):
    """A payload with every size measured; the kept shares differ per kind."""
    return {
        "sizes": {
            "original_rows": 100,
            "kept_rows": 10,
            "certified_rows": 7,
            "original_bounds": 40,
            "kept_bounds": 5,
            "certified_bounds": 3,
            "original_integers": 25,
            "kept_integers": 2,
            "certified_integers": 1,
            "size": 17,
            "certified": 11,
            "original": 165,
            "removed": 148,
        },
        "stats": {"solve_time": 1.5},
        **kw,
    }


def _row(rec, capsys):
    """print_table one record; its cells keyed by column label."""
    bench.print_table([rec])
    out = capsys.readouterr().out.splitlines()
    cells = out[-1].split()
    assert len(cells) == len(bench.TABLE_COLUMNS), out
    return {label: c for (label, _, _, _), c in zip(bench.TABLE_COLUMNS, cells)}


def test_the_table_shows_shares_per_kind_and_the_times(capsys):
    rec = {
        "name": "x",
        "status": "ok",
        "solve": {"wall_s": 4.0},
        "verify": _verify(passed=True, infeasible=True),
    }
    cells = _row(rec, capsys)
    assert [cells["rows%"], cells["bnds%"], cells["ints%"]] == ["10.0", "12.5", "8.0"]
    assert cells["cert%"] == "64.7"  # 11 of 17 kept elements certified
    assert (cells["iis(s)"], cells["check(s)"]) == ("4.00", "1.50")
    assert (cells["pass"], cells["status"]) == ("Y", "ok")


@pytest.mark.parametrize(
    "rec",
    [
        {"status": "ok", "verify": _verify(passed=False, infeasible=False)},  # proven feasible
        {"status": "invalid_solution", "verify": _verify(passed=False)},  # never a subsystem
    ],
)
def test_a_wrong_submission_blanks_what_it_produced(rec, capsys):
    """A small kept share beside a wall of certificates would otherwise read as good."""
    cells = _row({"name": "x", **rec}, capsys)
    assert [cells[c] for c in ("rows%", "bnds%", "ints%", "cert%")] == ["n/a"] * 4
    assert cells["pass"] == "N" and cells["check(s)"] != "n/a"


def test_a_tiny_share_is_not_shown_as_zero(capsys):
    verify = _verify(passed=True, infeasible=True)
    verify["sizes"].update(original_rows=100_000, kept_rows=3)
    assert _row({"name": "x", "status": "ok", "verify": verify}, capsys)["rows%"] == "<0.1"


def test_the_table_fits_a_narrow_terminal(capsys):
    rec = {"name": "x" * 60, "status": "invalid_solution", "verify": _verify()}
    bench.print_table([rec])
    assert max(len(line) for line in capsys.readouterr().out.splitlines()) <= 100


# --------------------------------------------------------------------------
# summary.csv
# --------------------------------------------------------------------------


def test_the_csv_has_every_field_and_raw_numbers():
    rec = {
        "name": "x",
        "status": "ok",
        "verify_wall_s": 2.0,
        "solve": {"wall_s": 4.0, "exit_code": 0},
        "verify": _verify(passed=True, infeasible=True, minimal=False),
    }
    (row,) = csv.DictReader(io.StringIO(bench.report_csv([bench.report_row(rec)])))
    assert list(row) == bench.REPORT_FIELDS
    assert row["kept_bounds_pct"] == "12.5"  # unrounded, unformatted
    assert (row["iis_time"], row["solve_time"], row["solve_exit_code"]) == ("4.0", "1.5", "0")
    assert row["passed"] == "True"
    assert row["solver"] == ""  # not measured: an empty cell, which pandas reads as NaN


def test_the_csv_drops_an_invalid_solutions_zeros():
    row = bench.report_row({"name": "x", "status": "invalid_solution", "verify": _verify()})
    assert row["kept_total"] is None and row["kept_pct"] is None
    assert row["orig_total"] == 165  # a fact about the instance, kept
    assert row["passed"] is False
