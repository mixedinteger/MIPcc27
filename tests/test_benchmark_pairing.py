"""
Unit tests for how the benchmark pairs a solution with its model.

The pairing is by model name: ``<model>.json[.gz]`` in the solution directory goes
with ``<model>.mps[.gz]`` in the instance directory.  Nothing here runs a
solver — resolution is the whole subject.
"""

from __future__ import annotations

import gzip

import pytest

from benchmark import run as bench
from mipcc_iis_checker.naming import find_model


def test_solution_names_are_model_names(tmp_path):
    """The .json extension comes off; the model's own dots stay on."""
    for name in ("stein9in.json", "b.json.gz", "normalized-dbsg_50_10_5_15.lin.json.gz"):
        (tmp_path / name).write_text("{}")
    assert bench.solution_names(tmp_path) == [
        "b",
        "normalized-dbsg_50_10_5_15.lin",
        "stein9in",
    ]


def test_a_model_is_found_by_name(tmp_path):
    (tmp_path / "stein9in.mps.gz").write_bytes(gzip.compress(b"NAME\nENDATA\n"))
    path = find_model("stein9in", tmp_path)
    assert path is not None and path.name == "stein9in.mps.gz"


def test_a_non_mps_file_is_not_a_model(tmp_path):
    """An .lp sitting next to the models is not silently read as the instance."""
    (tmp_path / "stein9in.lp").write_text("Minimize\nEnd\n")
    assert find_model("stein9in", tmp_path) is None


def test_a_bz2_model_is_not_a_model(tmp_path):
    """bzip2 is not a supported extension: Gurobi's reader shells out to a bzip2
    binary that need not be installed, so the corpus is .mps/.mps.gz only."""
    (tmp_path / "stein9in.mps.bz2").write_bytes(b"anything at all")
    assert find_model("stein9in", tmp_path) is None


def test_the_solution_name_is_not_appended_to_the_model_name(tmp_path):
    """The old scheme named solutions <model>.mps.gz.json.gz; that must not match.

    Left working, it would let one corpus be checked under two naming schemes at
    once, and the result cache keys the two spellings as different models.
    """
    (tmp_path / "stein9in.mps.gz").write_bytes(gzip.compress(b"NAME\nENDATA\n"))
    assert find_model("stein9in.mps.gz", tmp_path) is None


# --------------------------------------------------------------------------
# --instances takes model names, and a path to a model file is reduced to one
# --------------------------------------------------------------------------


def _refuse(msg):
    raise AssertionError(f"unexpected usage error: {msg}")


@pytest.mark.parametrize(
    "entry",
    [
        "stein9in",
        "stein9in.mps",
        "stein9in.mps.gz",
        "public_iis/stein9in.mps.gz",
        "/data/instances/public_iis/stein9in.mps.gz",
    ],
)
def test_a_model_file_is_accepted_as_its_name(entry):
    """A cluster job is handed the file, not the name, so both spellings work."""
    assert bench.normalize_instance_names([entry], _refuse) == ["stein9in"]


def test_a_path_that_is_not_a_model_file_is_a_usage_error():
    """Refused rather than guessed at: the name ends up inside a filename."""
    errors = []
    bench.normalize_instance_names(["public_iis/stein9in"], errors.append)
    assert errors and "not paths" in errors[0]


def test_run_one_refuses_a_path_as_a_name(tmp_path):
    """The record path is <result_dir>/<name>.json, and an absolute name escaped it.

    That is how a stray ``<instance>.mps.gz.json`` came to sit in the corpus
    directory and be read back as a cached 'no_solution' on the next run.
    """
    stray = tmp_path / "public_iis" / "stein9in.mps.gz"
    stray.parent.mkdir()

    with pytest.raises(ValueError, match="not a path"):
        bench.run_one(str(stray), stray.parent, tmp_path, tmp_path / "results")

    assert list(stray.parent.iterdir()) == []


def test_run_one_reports_a_model_it_cannot_find(tmp_path):
    """No solver is reached, and the status is one that a later run re-tries."""
    instances = tmp_path / "instances"
    solutions = tmp_path / "sols"
    instances.mkdir()
    solutions.mkdir()
    (solutions / "stein9in.json").write_text("{}")
    (instances / "stein9in.lp").write_text("Minimize\nEnd\n")  # not a model file

    rec = bench.run_one("stein9in", instances, solutions, tmp_path / "results")

    assert rec["status"] == "instance_not_found"
    assert rec["status"] not in bench.VERDICTS  # the harness's problem: exits non-zero


@pytest.mark.parametrize("name", ["", ".", "..", "a/b"])
def test_the_cli_and_run_one_refuse_the_same_names(tmp_path, name):
    """One rule, two callers: run_one used to let "", "." and ".." through."""
    errors = []
    bench.normalize_instance_names([name], errors.append)
    assert errors
    with pytest.raises(ValueError):
        bench.run_one(name, tmp_path, tmp_path, tmp_path / "results")


def test_a_solution_in_both_spellings_is_warned_about(tmp_path, capsys):
    """.json wins, but the .json.gz beside it is named rather than ignored silently."""
    (tmp_path / "foo.json").write_text("{}")
    (tmp_path / "foo.json.gz").write_bytes(gzip.compress(b"{}"))
    assert bench.find_solution("foo", tmp_path).name == "foo.json"
    assert "ignoring foo.json.gz" in capsys.readouterr().out


def test_a_single_solution_is_not_warned_about(tmp_path, capsys):
    (tmp_path / "foo.json.gz").write_bytes(gzip.compress(b"{}"))
    assert bench.find_solution("foo", tmp_path).name == "foo.json.gz"
    assert "warning" not in capsys.readouterr().out
