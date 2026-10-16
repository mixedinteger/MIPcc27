"""
Unit tests for the benchmark's result cache: when a cached check is reused.

The checker is replaced by a counting stand-in, so nothing is solved.
"""

import json

import pytest

from benchmark import collect as bcollect
from benchmark import run as bench


def _fake_verify(calls, status="ok", solver_status="INFEASIBLE"):
    def run(instance_path, solution_path, solver, time_limit):
        calls.append(time_limit)
        return {
            "status": status,
            "passed": status == "ok",
            "infeasible": True if status == "ok" else None,
            "sizes": {"original": 10, "size": 2},
            "stats": {"solver": "gurobi", "solver_status": solver_status, "solve_time": 0.1},
        }

    return run


@pytest.fixture
def pair(tmp_path):
    """instances/foo.mps and sols/foo.json; returns (instances, sols, results)."""
    instances, sols = tmp_path / "instances", tmp_path / "sols"
    instances.mkdir()
    sols.mkdir()
    (instances / "foo.mps").write_text("NAME foo\nENDATA\n")
    (sols / "foo.json").write_text("{}")
    return instances, sols, tmp_path / "results"


def _check(pair, **kw):
    instances, sols, results = pair
    return bench.run_one("foo", instances, sols, results, **kw)


def test_an_unchanged_pair_is_checked_once(pair, monkeypatch):
    calls = []
    monkeypatch.setattr(bench, "_run_verify", _fake_verify(calls))
    first = _check(pair)
    second = _check(pair)
    assert len(calls) == 1
    assert second["status"] == first["status"] == "ok"
    assert second["verify"] == first["verify"]


def test_the_record_written_is_the_one_returned(pair, monkeypatch):
    monkeypatch.setattr(bench, "_run_verify", _fake_verify([]))
    rec = _check(pair)
    assert json.loads((pair[2] / "foo.json").read_text()) == rec
    assert rec["schema"] == bench.SCHEMA


@pytest.mark.parametrize("changed", ["solution", "instance"])
def test_a_changed_file_is_checked_again(pair, monkeypatch, changed):
    """The cache is keyed by both files' hashes, not the model name."""
    calls = []
    monkeypatch.setattr(bench, "_run_verify", _fake_verify(calls))
    _check(pair)
    instances, sols, _ = pair
    target = sols / "foo.json" if changed == "solution" else instances / "foo.mps"
    target.write_text(target.read_text() + " ")
    _check(pair)
    assert len(calls) == 2


def test_another_solution_dir_is_not_served_this_ones_verdict(pair, monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(bench, "_run_verify", _fake_verify(calls))
    instances, _, results = pair
    _check(pair)
    other = tmp_path / "other"
    other.mkdir()
    (other / "foo.json").write_text('{"rows": {}}')
    bench.run_one("foo", instances, other, results)
    assert len(calls) == 2


def test_force_re_checks(pair, monkeypatch):
    calls = []
    monkeypatch.setattr(bench, "_run_verify", _fake_verify(calls))
    _check(pair)
    _check(pair, force=True)
    assert len(calls) == 2


def test_a_harness_problem_is_always_retried(pair, monkeypatch):
    calls = []
    monkeypatch.setattr(bench, "_run_verify", _fake_verify(calls, "verify_error", None))
    _check(pair)
    _check(pair)
    assert len(calls) == 2


@pytest.mark.parametrize("record", [{"name": "foo", "status": "ok"}, "not json", ["a list"]])
def test_an_old_or_corrupt_record_is_re_run(pair, monkeypatch, record):
    """No schema, or no JSON object at all: re-run rather than abort or trust it."""
    calls = []
    monkeypatch.setattr(bench, "_run_verify", _fake_verify(calls))
    pair[2].mkdir()
    (pair[2] / "foo.json").write_text(record if isinstance(record, str) else json.dumps(record))
    _check(pair)
    assert len(calls) == 1


def test_a_record_from_another_solver_is_re_run(pair, monkeypatch):
    calls = []
    monkeypatch.setattr(bench, "_run_verify", _fake_verify(calls))
    _check(pair)  # cached as gurobi
    _check(pair, solver="scip")
    assert len(calls) == 2


@pytest.mark.parametrize(
    "cached,requested,rerun",
    [
        (60, 60, False),
        (60, 30, False),
        (60, 120, True),
        (60, None, True),  # no cap is more time than any cap
        (None, None, False),
        (None, 600, False),
    ],
)
def test_an_undecided_check_is_re_run_only_with_more_time(
    pair, monkeypatch, cached, requested, rerun
):
    calls = []
    monkeypatch.setattr(bench, "_run_verify", _fake_verify(calls, "verify_unknown", "UNKNOWN"))
    _check(pair, time_limit=cached)
    _check(pair, time_limit=requested)
    assert calls == ([cached, requested] if rerun else [cached])


def test_run_one_reports_a_missing_solution(pair, monkeypatch):
    monkeypatch.setattr(bench, "_run_verify", lambda *a: pytest.fail("checked nothing"))
    (pair[1] / "foo.json").unlink()
    assert _check(pair)["status"] == "no_solution"


# --------------------------------------------------------------------------
# Atomic writes
# --------------------------------------------------------------------------


def test_write_json_leaves_no_temporary_file(tmp_path):
    bench._write_json(tmp_path / "x.json", {"a": 1})
    assert [p.name for p in tmp_path.iterdir()] == ["x.json"]


def test_write_json_cleans_up_after_a_failed_write(tmp_path):
    with pytest.raises(TypeError):
        bench._write_json(tmp_path / "x.json", {"a": object()})
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "checker",
    [
        {"version": "0.0.1", "tolerance": bench.DEFAULT_TOLERANCE},
        {"version": bench.__version__, "tolerance": 1e-9},
        None,
    ],
)
def test_a_verdict_from_another_checker_is_not_reused(pair, monkeypatch, checker):
    """A checker bug fix or a new tolerance must reach every cached verdict."""
    calls = []
    monkeypatch.setattr(bench, "_run_verify", _fake_verify(calls))
    _check(pair)
    instances, sols, results = pair
    path = results / "foo.json"
    rec = json.loads(path.read_text())
    rec["checker"] = checker
    path.write_text(json.dumps(rec))

    assert bcollect.collect_one("foo", results, instances, sols)["status"] == "stale"
    _check(pair)
    assert len(calls) == 2
    assert json.loads(path.read_text())["checker"] == bench.CHECKER
    assert bcollect.collect_one("foo", results, instances, sols)["status"] == "ok"
