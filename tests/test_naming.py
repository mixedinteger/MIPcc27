"""
Unit tests for model file naming.

Two rules are pinned down here, because every tool in the repo depends on both
agreeing: model files are MPS and nothing else, and a derived file *replaces* the
MPS extension rather than appending to it.
"""

from __future__ import annotations

import pytest

from mipcc_iis_checker.naming import (
    MPS_EXTENSIONS,
    find_model,
    model_name,
    model_names,
    mps_extension,
)


@pytest.mark.parametrize("ext", MPS_EXTENSIONS)
def test_every_supported_extension_is_recognised(ext):
    assert mps_extension(f"stein9in{ext}") == ext
    assert model_name(f"stein9in{ext}") == "stein9in"


def test_the_extension_is_replaced_not_appended():
    """The bug this rules out: stein9in.mps.gz.json.gz, and a cache keyed on it."""
    assert model_name("public_iis/stein9in.mps.gz") + ".json.gz" == "stein9in.json.gz"


def test_a_dot_in_the_model_name_survives():
    """Only the MPS extension comes off — 'normalized-...lin' is part of the name."""
    assert model_name("d/normalized-dbsg_50_10_5_15.lin.mps.gz") == "normalized-dbsg_50_10_5_15.lin"
    assert model_name("3col-l030.cnf-plain.pb.metafix.mps") == "3col-l030.cnf-plain.pb.metafix"


def test_directories_are_stripped():
    assert model_name("/data/instances/public_iis/stein9in.mps.gz") == "stein9in"


@pytest.mark.parametrize(
    "path",
    [
        "problem.lp",  # what Gurobi writes an IIS as
        "problem.ilp",
        "problem.opb.bz2",
        "problem.json.gz",  # a solution, not a model
        "problem",
        ".mps",  # an extension with no name is not a model
        "problem.mps.zip",
    ],
)
def test_non_mps_is_refused(path):
    """A mis-named instance is caught here, not as a parse error three steps later."""
    assert mps_extension(path) is None
    with pytest.raises(ValueError, match="not an MPS model file"):
        model_name(path)


def test_find_model_is_the_inverse_of_model_name(tmp_path):
    (tmp_path / "stein9in.mps.gz").write_text("")
    found = find_model("stein9in", tmp_path)
    assert found is not None
    assert model_name(found) == "stein9in"


def test_find_model_reports_a_missing_model_rather_than_guessing(tmp_path):
    (tmp_path / "stein9in.lp").write_text("")  # an LP is not a model file
    assert find_model("stein9in", tmp_path) is None


def test_find_model_prefers_the_order_in_MPS_EXTENSIONS(tmp_path):
    for ext in MPS_EXTENSIONS:
        (tmp_path / f"m{ext}").write_text("")
    assert find_model("m", tmp_path).name == f"m{MPS_EXTENSIONS[0]}"


def test_model_names_lists_models_and_ignores_everything_else(tmp_path):
    for name in ("a.mps", "b.mps.gz", "c.lp", "gurobi.log", "d.json.gz"):
        (tmp_path / name).write_text("")
    (tmp_path / "iis").mkdir()
    assert model_names(tmp_path) == ["a", "b"]


def test_model_names_deduplicates_a_model_stored_twice(tmp_path):
    """Same model, two compressions — one name."""
    (tmp_path / "a.mps").write_text("")
    (tmp_path / "a.mps.gz").write_text("")
    assert model_names(tmp_path) == ["a"]
