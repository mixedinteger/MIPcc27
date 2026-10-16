"""
Unit tests for the solve step: the harness launching, timing and killing the
competitor's program, and reporting from the cache without running anything.

The "program" is a small Python script run through a command template, so no
container engine and no solver is needed; the checker is replaced where a test
would otherwise solve.
"""

from __future__ import annotations

import json
import sys
import time

import pytest

from benchmark import collect as bcollect
from benchmark import run as bench
from benchmark import solve as bsolve
from benchmark.solve import Launcher, SolveConfig

_SOLUTION = '{"rows": {}}'

_PROGRAMS = {
    # Writes a solution, and logs its environment so the test can see the limits.
    "ok": f"""
import os, sys
print("limit", os.environ.get("MIPCC_TIME_LIMIT"), "threads", os.environ.get("MIPCC_THREADS"))
open(sys.argv[2], "w").write({_SOLUTION!r})
""",
    "gz": f"""
import gzip, sys
with gzip.open(sys.argv[2] + ".gz", "wt") as f:
    f.write({_SOLUTION!r})
""",
    "fail": f"""
import sys
open(sys.argv[2], "w").write({_SOLUTION!r})
sys.exit(3)
""",
    "silent": "pass",
    # Answers with a link to someone else's file instead of writing its own.
    "symlink": """
import os, sys
os.symlink(sys.argv[3], sys.argv[2])
""",
    # Starts a child in its process group, records its pid, then outlives any limit.
    "hang": """
import subprocess, sys, time, pathlib
child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
pathlib.Path(sys.argv[3]).write_text(str(child.pid))
time.sleep(60)
""",
    # Ignores SIGTERM, so only SIGKILL stops it.
    "stubborn": """
import signal, time
signal.signal(signal.SIGTERM, signal.SIG_IGN)
time.sleep(60)
""",
}


@pytest.fixture
def corpus(tmp_path):
    """instances/foo.mps, and where sols/, results/ and logs/ go."""
    instances = tmp_path / "instances"
    instances.mkdir()
    (instances / "foo.mps").write_text("NAME foo\nENDATA\n")
    return {
        "instances": instances,
        "sols": tmp_path / "sols",
        "results": tmp_path / "out" / "results",
        "logs": tmp_path / "out" / "logs",
        "tmp": tmp_path,
    }


def _launcher(tmp_path, behaviour, counter=None):
    """A --solve-cmd launcher for one of :data:`_PROGRAMS`; argv[3] is a side file."""
    script = tmp_path / f"program_{behaviour}.py"
    body = _PROGRAMS[behaviour]
    if counter is not None:  # every launch appends a line, so a test can count them
        body = f"open({str(counter)!r}, 'a').write('x\\n')\n" + body
    script.write_text(body)
    side = tmp_path / f"{behaviour}.side"
    return Launcher(template=(sys.executable, str(script), "{instance}", "{solution}", str(side)))


def _config(corpus, launcher, time_limit=5.0, threads=None):
    return SolveConfig(launcher, time_limit, threads, corpus["logs"])


def _solve(corpus, launcher, **kw):
    instance = corpus["instances"] / "foo.mps"
    return bsolve.run_solve("foo", instance, corpus["sols"], _config(corpus, launcher, **kw))


def _alive(pid):
    """Whether *pid* still runs.  A zombie is dead, merely unreaped — what an orphan
    becomes under an init that never reaps, as in some containers."""
    try:
        with open(f"/proc/{pid}/status") as f:
            return not any(line.startswith("State:\tZ") for line in f)
    except FileNotFoundError:
        return False


# --------------------------------------------------------------------------
# run_solve: one launch, and what it leaves behind
# --------------------------------------------------------------------------


def test_a_clean_run_copies_the_solution_into_place(corpus):
    rec = _solve(corpus, _launcher(corpus["tmp"], "ok"), time_limit=7, threads=4)

    assert rec["status"] == "solved" and rec["exit_code"] == 0
    assert (corpus["sols"] / "foo.json").read_text() == _SOLUTION
    assert rec["solution_sha256"] == bsolve.file_sha256(corpus["sols"] / "foo.json")
    assert rec["wall_s"] >= 0
    assert "limit 7 threads 4" in (corpus["logs"] / "foo.log").read_text()
    # The scratch directory the program wrote into is gone.
    assert [p.name for p in corpus["sols"].iterdir()] == ["foo.json"]


def test_a_compressed_solution_is_accepted(corpus):
    assert _solve(corpus, _launcher(corpus["tmp"], "gz"))["status"] == "solved"
    assert (corpus["sols"] / "foo.json.gz").exists()


@pytest.mark.parametrize(
    "behaviour,error",
    [("fail", "exit code 3"), ("silent", "no solution file written")],
)
def test_a_failing_program_leaves_no_solution(corpus, behaviour, error):
    """A non-zero exit disowns whatever it wrote; a clean exit must have written one."""
    rec = _solve(corpus, _launcher(corpus["tmp"], behaviour))
    assert (rec["status"], rec["error"]) == ("solve_failed", error)
    assert not list(corpus["sols"].iterdir())


def test_a_symlinked_answer_is_refused(corpus):
    """Otherwise a program could hand in another team's solution file."""
    other = corpus["tmp"] / "symlink.side"
    other.write_text('{"rows": {"stolen": {"action": "relax"}}}')
    rec = _solve(corpus, _launcher(corpus["tmp"], "symlink"))
    assert rec["status"] == "solve_failed"
    assert not list(corpus["sols"].iterdir())


def test_a_stale_solution_is_deleted_before_the_program_runs(corpus):
    corpus["sols"].mkdir()
    (corpus["sols"] / "foo.json").write_text(_SOLUTION)
    (corpus["sols"] / "foo.json.gz").write_text("old")
    _solve(corpus, _launcher(corpus["tmp"], "fail"))
    assert list(corpus["sols"].iterdir()) == []


def test_a_program_past_its_limit_is_killed_with_its_children(corpus):
    rec = _solve(corpus, _launcher(corpus["tmp"], "hang"), time_limit=1.0)
    assert rec["status"] == "solve_timeout"
    assert 1.0 <= rec["wall_s"] < 10
    time.sleep(0.2)
    assert not _alive(int((corpus["tmp"] / "hang.side").read_text()))


def test_a_program_that_ignores_sigterm_gets_sigkill(corpus, monkeypatch):
    monkeypatch.setattr(bsolve, "KILL_GRACE_S", 0.5)
    rec = _solve(corpus, _launcher(corpus["tmp"], "stubborn"), time_limit=1.0)
    assert rec["status"] == "solve_timeout" and rec["wall_s"] < 5


def test_a_command_that_cannot_start_is_the_judges_problem(corpus):
    launcher = Launcher(template=("/no/such/program", "{instance}", "{solution}"))
    rec = _solve(corpus, launcher)
    assert rec["status"] == "solve_error" and "could not start" in rec["error"]
    # Never reused: the judge may have fixed the command since.
    assert not bsolve.solve_is_reusable(rec, _config(corpus, launcher), None)


# --------------------------------------------------------------------------
# Launchers
# --------------------------------------------------------------------------


def test_a_template_is_filled_per_token_without_a_shell(tmp_path):
    launcher = Launcher(template=("run", "--model={name}", "{instance}", "{solution}", "a;b"))
    argv = launcher.argv("foo", tmp_path / "foo.mps", tmp_path / "out" / "foo.json")
    assert argv == [
        "run",
        "--model=foo",
        str(tmp_path / "foo.mps"),
        str(tmp_path / "out" / "foo.json"),
        "a;b",  # passed through literally: there is no shell to interpret it
    ]


def test_a_container_is_sandboxed_and_sees_only_its_files(tmp_path):
    launcher = Launcher(image="team:1", container_args=("--cpus", "8"))
    argv = launcher.argv("a b", tmp_path / "a b.mps", tmp_path / "scratch" / "a b.json", "c1")

    assert argv[:2] == ["podman", "run"]
    assert argv[argv.index("--name") + 1] == "c1"
    assert {"--rm", "--init", "--read-only", "--userns=keep-id"} <= set(argv)
    assert argv[argv.index("--network") + 1] == "none"
    assert argv[argv.index("--cap-drop") + 1] == "ALL"
    assert argv[argv.index("--security-opt") + 1] == "no-new-privileges"
    volumes = [argv[i + 1] for i, a in enumerate(argv) if a == "-v"]
    assert volumes == [
        f"{(tmp_path / 'a b.mps').resolve()}:/data/a b.mps:ro",
        f"{(tmp_path / 'scratch').resolve()}:/out",
    ]
    # Extra arguments go before the image; the program's own arguments after it.
    assert argv[-5:] == ["--cpus", "8", "team:1", "/data/a b.mps", "/out/a b.json"]


def test_podman_options_and_network_reach_the_container(tmp_path):
    launcher = Launcher(image="team:1", podman=("podman", "--root", "/r"), network="host")
    argv = launcher.argv("a", tmp_path / "a.mps", tmp_path / "scratch" / "a.json", "c1")

    assert argv[:4] == ["podman", "--root", "/r", "run"]
    assert argv[argv.index("--network") + 1] == "host"
    assert "--network host" in launcher.describe()
    assert "--network" not in Launcher(image="team:1").describe()


def test_a_pids_limit_of_zero_sets_none(tmp_path):
    def argv(launcher):
        return launcher.argv("a", tmp_path / "a.mps", tmp_path / "scratch" / "a.json", "c1")

    limited = argv(Launcher(image="team:1"))
    assert limited[limited.index("--pids-limit") + 1] == "4096"
    assert "--pids-limit" not in argv(Launcher(image="team:1", pids_limit=0))


def test_a_timed_out_container_is_killed_through_the_same_podman(corpus):
    # A fake podman: "run" hangs, "kill" records how it was called.
    calls = corpus["tmp"] / "calls"
    podman = corpus["tmp"] / "podman"
    podman.write_text(f'#!/bin/sh\necho "$@" >> {calls}\n[ "$3" = run ] && exec sleep 60\nexit 0\n')
    podman.chmod(0o755)
    launcher = Launcher(image="team:1", podman=(str(podman), "--root", "/r"))
    rec = _solve(corpus, launcher, time_limit=0.5)

    assert rec["status"] == "solve_timeout"
    kills = [c for c in calls.read_text().splitlines() if " kill " in c]
    assert kills and all(c.startswith("--root /r kill --signal ") for c in kills)


def test_a_launcher_is_a_command_or_an_image():
    with pytest.raises(ValueError):
        Launcher()
    with pytest.raises(ValueError):
        Launcher(template=("x",), image="y")


def test_reuse_needs_the_same_program_limits_and_solution(corpus):
    launcher = _launcher(corpus["tmp"], "ok")
    rec = _solve(corpus, launcher, time_limit=5.0, threads=2)
    solution = corpus["sols"] / "foo.json"
    same = _config(corpus, launcher, 5.0, 2)

    assert bsolve.solve_is_reusable(rec, same, solution)
    assert not bsolve.solve_is_reusable(rec, _config(corpus, launcher, 6.0, 2), solution)
    assert not bsolve.solve_is_reusable(rec, _config(corpus, launcher, 5.0, 4), solution)
    assert not bsolve.solve_is_reusable(rec, _config(corpus, Launcher(template=("x",))), solution)
    assert not bsolve.solve_is_reusable(rec, same, None)
    solution.write_text('{"rows": {"r": {"action": "relax"}}}')
    assert not bsolve.solve_is_reusable(rec, same, solution)


# --------------------------------------------------------------------------
# run_one with a solve step
# --------------------------------------------------------------------------


def _fake_checker(calls):
    def run(instance_path, solution_path, solver, time_limit):
        calls.append(str(solution_path))
        return {
            "status": "ok",
            "passed": True,
            "infeasible": True,
            "minimal": True,
            "sizes": {"original": 10, "size": 2},
            "stats": {"solver": "gurobi", "solver_status": "INFEASIBLE", "solve_time": 0.5},
        }

    return run


def _run_one(corpus, launcher, force=False, time_limit=5.0):
    return bench.run_one(
        "foo",
        corpus["instances"],
        corpus["sols"],
        corpus["results"],
        force=force,
        solve=_config(corpus, launcher, time_limit),
    )


def test_run_one_solves_then_checks(corpus, monkeypatch):
    calls = []
    monkeypatch.setattr(bench, "_run_verify", _fake_checker(calls))
    rec = _run_one(corpus, _launcher(corpus["tmp"], "ok"))

    assert rec["status"] == "ok" and calls == [str(corpus["sols"] / "foo.json")]
    assert rec["solve"]["status"] == "solved"
    row = bench.report_row(rec)
    assert row["iis_time"] == rec["solve"]["wall_s"] and row["solve_exit_code"] == 0


def test_a_second_run_reuses_the_program_and_the_check(corpus, monkeypatch):
    calls = []
    monkeypatch.setattr(bench, "_run_verify", _fake_checker(calls))
    counter = corpus["tmp"] / "launches"
    launcher = _launcher(corpus["tmp"], "ok", counter)

    _run_one(corpus, launcher)
    _run_one(corpus, launcher)
    assert counter.read_text().count("x") == 1 and len(calls) == 1

    _run_one(corpus, launcher, time_limit=10.0)  # a new limit is a new run
    assert counter.read_text().count("x") == 2
    _run_one(corpus, launcher, force=True)
    assert counter.read_text().count("x") == 3


@pytest.mark.parametrize("force", [False, True])
def test_a_check_only_run_keeps_the_programs_time(corpus, monkeypatch, force):
    monkeypatch.setattr(bench, "_run_verify", _fake_checker([]))
    solved = _run_one(corpus, _launcher(corpus["tmp"], "ok"))

    rec = bench.run_one("foo", corpus["instances"], corpus["sols"], corpus["results"], force=force)
    assert rec["solve"] == solved["solve"]
    assert bench.report_row(rec)["iis_time"] == solved["solve"]["wall_s"]

    (corpus["sols"] / "foo.json").write_text('{"rows": {"other": {}}}')
    rec = bench.run_one("foo", corpus["instances"], corpus["sols"], corpus["results"])
    assert "solve" not in rec  # that time was for another file


def test_a_failed_program_is_never_checked_and_its_failure_is_reused(corpus, monkeypatch):
    monkeypatch.setattr(bench, "_run_verify", lambda *a: pytest.fail("checked a failed run"))
    counter = corpus["tmp"] / "launches"
    launcher = _launcher(corpus["tmp"], "fail", counter)

    rec = _run_one(corpus, launcher)
    assert rec["status"] == "solve_failed" and bench.pass_mark(rec) is False
    assert json.loads((corpus["results"] / "foo.json").read_text()) == rec
    _run_one(corpus, launcher)
    assert counter.read_text().count("x") == 1


# --------------------------------------------------------------------------
# benchmark.collect
# --------------------------------------------------------------------------


def _report(corpus, **dirs):
    dirs = dirs or {"instance_dir": corpus["instances"], "solution_dir": corpus["sols"]}
    return bcollect.collect_one("foo", corpus["results"], **dirs)


@pytest.fixture
def ran(corpus, monkeypatch):
    """Run the "ok" program and check it, then forbid any further running."""
    monkeypatch.setattr(bench, "_run_verify", _fake_checker([]))
    done = _run_one(corpus, _launcher(corpus["tmp"], "ok"))
    monkeypatch.setattr(bench, "_run_verify", lambda *a: pytest.fail("report ran the checker"))
    monkeypatch.setattr(bsolve.subprocess, "Popen", lambda *a, **k: pytest.fail("report ran"))
    return done


def test_a_report_serves_a_cached_result(corpus, ran):
    assert _report(corpus) == ran


def test_a_model_never_run_is_not_run(corpus):
    assert _report(corpus)["status"] == "not_run"
    assert not corpus["results"].exists()  # made up, not written


def test_a_solution_changed_after_the_check_is_stale(corpus, ran):
    (corpus["sols"] / "foo.json").write_text('{"rows": {"r": {"action": "relax"}}}')
    assert _report(corpus)["status"] == "stale"


def test_a_solution_that_vanished_is_stale(corpus, ran):
    (corpus["sols"] / "foo.json").unlink()
    assert _report(corpus)["status"] == "stale"


def test_a_changed_instance_is_stale(corpus, ran):
    (corpus["instances"] / "foo.mps").write_text("NAME bar\nENDATA\n")
    assert _report(corpus, instance_dir=corpus["instances"])["status"] == "stale"


def test_the_results_alone_are_enough(corpus, ran):
    """The solutions may live only on the cluster: a directory not given isn't looked at."""
    (corpus["sols"] / "foo.json").unlink()
    assert _report(corpus, solution_dir=None) == ran
    assert _report(corpus, instance_dir=corpus["instances"]) == ran


@pytest.mark.parametrize(
    "behaviour,status",
    [("silent", "solve_failed"), ("fail", "solve_failed"), (None, "solve_error")],
)
def test_a_report_shows_how_a_run_stopped(corpus, monkeypatch, behaviour, status):
    """Including a launcher error — not mistaken for a changed file."""
    monkeypatch.setattr(bench, "_run_verify", _fake_checker([]))
    launcher = (
        _launcher(corpus["tmp"], behaviour)
        if behaviour
        else Launcher(template=("/no/such/program", "{instance}", "{solution}"))
    )
    _run_one(corpus, launcher)
    assert _report(corpus)["status"] == status


# --------------------------------------------------------------------------
# The command line
# --------------------------------------------------------------------------


def _main(monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["benchmark.run", *argv])
    return bench.main()


def _collect(monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["benchmark.collect", *argv])
    return bcollect.main()


def _dirs(corpus):
    return [
        "--instance-dir",
        str(corpus["instances"]),
        "--solution-dir",
        str(corpus["sols"]),
        "--out-dir",
        str(corpus["tmp"] / "out"),
    ]


def _job(corpus, behaviour):
    script = corpus["tmp"] / f"program_{behaviour}.py"
    _launcher(corpus["tmp"], behaviour)  # writes the script
    return [
        *_dirs(corpus),
        "--solve-cmd",
        f"{sys.executable} {script} {{instance}} {{solution}}",
        "--solve-time-limit",
        "5",
        "--instances",
        str(corpus["instances"] / "foo.mps"),
    ]


def test_a_per_model_job_writes_no_summary_and_a_report_does(corpus, monkeypatch):
    monkeypatch.setattr(bench, "_run_verify", _fake_checker([]))
    assert _main(monkeypatch, *_job(corpus, "ok")) == 0
    assert not (corpus["tmp"] / "out" / "summary.csv").exists()

    assert _collect(monkeypatch, *_dirs(corpus)) == 0
    rows = (corpus["tmp"] / "out" / "summary.csv").read_text().splitlines()
    header, row = rows[0].split(","), rows[1].split(",")
    assert row[header.index("name")] == "foo"
    assert float(row[header.index("iis_time")]) >= 0


def test_a_failed_submission_still_exits_0(corpus, monkeypatch):
    """It is a verdict; only a harness problem should fail the cluster job."""
    assert _main(monkeypatch, *_job(corpus, "fail")) == 0


def test_a_harness_problem_exits_1(corpus, monkeypatch):
    corpus["sols"].mkdir()
    assert _collect(monkeypatch, *_dirs(corpus)) == 1  # foo: not_run


def test_collect_needs_only_the_out_dir_and_writes_only_the_summaries(corpus, monkeypatch):
    monkeypatch.setattr(bench, "_run_verify", _fake_checker([]))
    assert _main(monkeypatch, *_job(corpus, "ok")) == 0
    out = corpus["tmp"] / "out"
    before = {p: p.read_bytes() for p in corpus["tmp"].rglob("*") if p.is_file()}

    assert _collect(monkeypatch, "--out-dir", str(out)) == 0
    after = {p: p.read_bytes() for p in corpus["tmp"].rglob("*") if p.is_file()}
    assert set(after) - set(before) == {out / "summary.json", out / "summary.csv"}
    assert all(after[p] == data for p, data in before.items())
    assert [r["name"] for r in json.loads((out / "summary.json").read_text())] == ["foo"]


def test_collect_with_nothing_to_report_is_a_usage_error(corpus, monkeypatch):
    with pytest.raises(SystemExit) as e:
        _collect(monkeypatch, "--out-dir", str(corpus["tmp"] / "out"))
    assert e.value.code == 2


@pytest.mark.parametrize(
    "extra",
    [
        ["--container", "img"],  # no --solve-time-limit
        ["--solve-cmd", "x {instance} {solution}", "--solve-time-limit", "0"],
        ["--solve-cmd", "x {instance} {oops}", "--solve-time-limit", "5"],
        ["--solve-time-limit", "5"],  # without a program
    ],
)
def test_bad_solve_flags_are_usage_errors(corpus, monkeypatch, extra):
    corpus["sols"].mkdir()
    with pytest.raises(SystemExit) as e:
        _main(monkeypatch, *_dirs(corpus), *extra)
    assert e.value.code == 2
