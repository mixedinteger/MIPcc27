"""
Benchmark harness: run the competitor's program, check what it wrote, report.

For each model: optionally launch the program (``--container`` or ``--solve-cmd``,
see :mod:`benchmark.solve`), run ``mipcc_iis_checker.verify_subsystem()`` on the
solution, and cache the result in ``<out-dir>/results/<model>.json``.
``benchmark.collect`` summarizes that cache and runs nothing.  See ``benchmark/README.md``.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import shlex
import signal
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional

from mipcc_iis_checker import DEFAULT_TOLERANCE, __version__, verify_subsystem
from mipcc_iis_checker.naming import (
    MPS_EXTENSIONS,
    find_model,
    model_name,
    model_names,
    mps_extension,
)

from .solve import (
    SOLUTION_EXTS,
    SOLVE_FAILURES,
    SOLVED,
    TEMPLATE_FIELDS,
    Launcher,
    SolveConfig,
    file_sha256,
    resolve_image,
    run_solve,
    solve_is_reusable,
)

_HERE = Path(__file__).resolve().parent
# Repository-level, not inside the package, so a pip-installed copy is never
# written into.  Override with --out-dir or IIS_BENCH_DIR.
DEFAULT_OUT_DIR = Path(os.environ.get("IIS_BENCH_DIR") or _HERE.parent / "benchmark_out")

# Bump whenever the result record's layout changes: older records are re-run.
SCHEMA = 3

# What checks a solution.  A cached verdict from another checker version or
# tolerance is not this checker's verdict, so it is re-checked (or, in a report,
# is stale) just as if the files had changed.
CHECKER = {"version": __version__, "tolerance": DEFAULT_TOLERANCE}

# Statuses that are a verdict on the submission, so a run that ends with only these
# succeeded.  Everything else is the harness's problem, and is never reused.
VERDICTS = frozenset({"ok", "invalid_solution"}) | SOLVE_FAILURES

# Cached results worth reusing: the verdicts the checker reached, plus an undecided
# check, which is re-run only with a larger --verify-time-limit.
_REUSABLE = frozenset({"ok", "invalid_solution", "verify_unknown"})


def log(msg: str) -> None:
    """Timestamped, flushed progress line so long runs stream in real time."""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# --------------------------------------------------------------------------
# Names and files
# --------------------------------------------------------------------------


def solution_names(solution_dir: Path) -> list[str]:
    """Model names implied by the solution files in *solution_dir*."""
    names = set()
    for ext in SOLUTION_EXTS:
        names.update(p.name[: -len(ext)] for p in solution_dir.glob(f"*{ext}"))
    return sorted(names)


def is_model_name(name: str) -> bool:
    """Whether *name* is safe to paste into ``<result_dir>/<name>.json``."""
    return name not in ("", ".", "..") and "/" not in name and os.sep not in name


def normalize_instance_names(raw: list[str], on_error) -> list[str]:
    """Reduce ``--instances`` arguments to model names; a model file's path is accepted."""
    names = []
    for entry in raw:
        name = model_name(entry) if mps_extension(entry) is not None else entry
        if not is_model_name(name):
            on_error(
                f"--instances takes model names, not paths: {entry!r} "
                f"(a model file is accepted too, e.g. dir/{name or 'foo'}.mps.gz)"
            )
        names.append(name)
    return names


def find_solution(name: str, solution_dir: Path) -> Optional[Path]:
    """``<name>.json`` or ``<name>.json.gz`` in *solution_dir*; warns if both exist."""
    found = [solution_dir / f"{name}{ext}" for ext in SOLUTION_EXTS]
    found = [p for p in found if p.exists()]
    if len(found) > 1:
        log(f"{name}: warning: using {found[0].name}, ignoring {found[1].name}")
    return found[0] if found else None


def foreign_solutions(name: str, solution_dir: Path, prev: Optional[dict]) -> list[Path]:
    """Solution files for *name* that the program's run would delete but the harness did not write.

    Running the program replaces ``<name>.json[.gz]``.  The only file the harness may
    replace unasked is the one its own last run recorded (same SHA-256); anything else
    -- reference solutions, another team's output -- belongs to someone else.
    """
    written = (prev or {}).get("solution_sha256") if (prev or {}).get("status") == SOLVED else None
    files = [solution_dir / f"{name}{ext}" for ext in SOLUTION_EXTS]
    return [p for p in files if p.is_file() and (written is None or file_sha256(p) != written)]


def _input_hashes(solution_path: Path, instance_path: Path) -> dict:
    """What a cached result must have been made from to be reused."""
    return {
        "solution_sha256": file_sha256(solution_path),
        "instance_sha256": file_sha256(instance_path),
    }


def _write_json(path: Path, obj) -> None:
    """Write *obj* to *path* as JSON, atomically (see :func:`_write_text`)."""
    _write_text(path, json.dumps(obj, indent=2))


def _write_text(path: Path, text: str) -> None:
    """Write *text* to *path* atomically, so jobs sharing a directory never see half a file."""
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(text)
        os.replace(tmp, path)
    except BaseException:
        os.unlink(tmp)
        raise


def _read_json(path: Path) -> Optional[dict]:
    """The JSON object at *path*, or None if missing or unreadable."""
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


# --------------------------------------------------------------------------
# Verification
# --------------------------------------------------------------------------


def _verify_status(res) -> str:
    """Classify a verification.

    ``ok``: the solver decided, whatever it decided.  ``invalid_solution``: the
    solution file is unreadable or names elements the instance lacks — the
    submitter's fault.  ``verify_unknown``: the solve decided nothing (usually the
    time limit).  ``verify_error``: the checker could not run (no solver, unreadable
    instance, solver error) — the harness's problem.
    """
    if not res.is_valid_solution:
        return "invalid_solution"
    status = res.stats.solver_status
    if status is None or status == "ERROR":
        return "verify_error"
    if status == "UNKNOWN":
        return "verify_unknown"
    return "ok"


def _run_verify(instance_path, solution_path, solver, time_limit) -> dict:
    """Run the checker on one pair and flatten what it said into a JSON-able dict."""
    try:
        res = verify_subsystem(
            str(instance_path), str(solution_path), solver=solver, time_limit=time_limit
        )
    except Exception as e:  # noqa: BLE001 — one unverifiable pair must not abort the batch
        return {"status": "verify_error", "errors": [repr(e)[:1000]]}
    return {
        "status": _verify_status(res),
        "infeasible": res.is_infeasible,
        "minimal": res.is_minimal,
        "passed": res.is_passed,
        "invalid_certificates": len(res.invalid_certificates),
        "invalid_certificate_details": [
            {"element": e, "first_violation": v[0] if v else "?"}
            for e, v in res.invalid_certificates[:10]
        ],
        "errors": res.errors[:10],
        "sizes": {
            "kept_rows": res.kept.rows,
            "kept_bounds": res.kept.bounds,
            "kept_integers": res.kept.integers,
            "size": res.kept.total,
            "removed": res.removed.total,
            "certified_rows": res.certified.rows,
            "certified_bounds": res.certified.bounds,
            "certified_integers": res.certified.integers,
            "certified": res.certified.total,
            "original_rows": res.original.rows,
            "original_bounds": res.original.bounds,
            "original_integers": res.original.integers,
            "original": res.original.total,
        },
        "stats": {
            "solver": res.stats.solver,
            "solver_status": res.stats.solver_status,
            "solve_time": res.stats.solve_time,
        },
    }


# --------------------------------------------------------------------------
# The cache
# --------------------------------------------------------------------------


def _time_limit_was_enough(cached: Optional[float], requested: Optional[float]) -> bool:
    """Whether an undecided check under *cached* already had *requested*'s time (None = no cap)."""
    if cached is None:
        return True
    return requested is not None and requested <= cached


def _checker_label(checker: Optional[dict]) -> str:
    """``version, tolerance`` of a recorded checker, for log lines."""
    checker = checker or {}
    return f"{checker.get('version', '?')}, tolerance {checker.get('tolerance', '?')}"


def _cache_miss(rec: Optional[dict], name: str, inputs: dict, solver: str, time_limit) -> str:
    """Why *rec* can't be reused for this check, or "" if it can."""
    if rec is None:
        return "no cached result"
    if rec.get("schema") != SCHEMA or rec.get("name") != name:
        return "cached result is from an older harness"
    if rec.get("checker") != CHECKER:
        return f"cached result is from another checker ({_checker_label(rec.get('checker'))})"
    if rec.get("status") not in _REUSABLE:
        return f"cached status {rec.get('status')!r} is retried"
    if rec.get("inputs") != inputs:
        return "cached result was made from different files"
    cached_solver = ((rec.get("verify") or {}).get("stats") or {}).get("solver")
    if solver and cached_solver and cached_solver != solver:
        return "cached result is from another solver"
    if rec["status"] == "verify_unknown" and not _time_limit_was_enough(
        rec.get("time_limit"), time_limit
    ):
        return "cached check was undecided under a smaller time limit"
    return ""


def run_one(
    name: str,
    instance_dir: Path,
    solution_dir: Path,
    result_dir: Path,
    solver: str = "",
    time_limit: Optional[float] = None,
    force: bool = False,
    solve: Optional[SolveConfig] = None,
    overwrite: bool = False,
) -> dict:
    """Solve (if *solve*), check, and record one model in ``<result_dir>/<name>.json``.

    Raises ``FileExistsError`` before running the program if that would replace a solution
    the harness did not write (see :func:`foreign_solutions`), unless *overwrite*.
    """
    if not is_model_name(name):
        raise ValueError(f"instance name must be a model name, not a path: {name!r}")
    result_dir.mkdir(parents=True, exist_ok=True)
    result_path = result_dir / f"{name}.json"
    cached = _read_json(result_path)
    rec: dict = {"schema": SCHEMA, "name": name}

    def finish(status: str) -> dict:
        rec["status"] = status
        _write_json(result_path, rec)
        return rec

    instance_path = find_model(name, instance_dir)
    if instance_path is None:
        log(f"{name}: no instance file in {instance_dir}")
        return finish("instance_not_found")

    # 1. Produce the solution, when the harness runs the program.
    if solve is not None:
        prev = (cached or {}).get("solve")
        if not force and solve_is_reusable(
            prev, solve, find_solution(name, solution_dir), instance_path
        ):
            log(f"{name}: reusing the program's last run ({prev['status']})")
            rec["solve"] = prev
        else:
            foreign = [] if overwrite else foreign_solutions(name, solution_dir, prev)
            if foreign:
                raise FileExistsError(
                    f"{name}: the program would replace {foreign[0]}, which this harness did not "
                    "write; use an empty --solution-dir, or --overwrite-solutions"
                )
            log(f"{name}: running the program (limit {solve.time_limit:g}s) ...")
            rec["solve"] = run_solve(name, instance_path, solution_dir, solve)
            s = rec["solve"]
            log(f"{name}: program {s['status']} in {s['wall_s']}s ({s['error'] or 'exit 0'})")
        if rec["solve"]["status"] != SOLVED:
            return finish(rec["solve"]["status"])

    # 2. Check it, unless the cache already holds a check of these same files.
    solution_path = find_solution(name, solution_dir)
    if solution_path is None:
        log(f"{name}: no .json/.json.gz solution in {solution_dir}")
        return finish("no_solution")
    rec["inputs"] = _input_hashes(solution_path, instance_path)
    rec["checker"] = CHECKER
    if solve is None:
        # A check-only run keeps the timing of the run that wrote this very file.
        prev = (cached or {}).get("solve")
        if (
            prev
            and prev.get("status") == SOLVED
            and prev.get("solution_sha256") == rec["inputs"]["solution_sha256"]
        ):
            rec["solve"] = prev
    rec["time_limit"] = time_limit
    miss = "forced" if force else _cache_miss(cached, name, rec["inputs"], solver, time_limit)
    if not miss:
        log(f"{name}: cached result, skipping")
        rec["verify"], rec["verify_wall_s"] = cached["verify"], cached.get("verify_wall_s")
        rec["time_limit"] = cached.get("time_limit")
        return finish(cached["status"])
    if cached is not None and miss != "no cached result":
        log(f"{name}: {miss}; re-checking")

    log(f"{name}: checking {solution_path.name} against {instance_path.name} ...")
    t0 = time.monotonic()
    rec["verify"] = _run_verify(instance_path, solution_path, solver, time_limit)
    rec["verify_wall_s"] = round(time.monotonic() - t0, 3)
    v = rec["verify"]
    log(f"{name}: {v['status']} (infeasible={v.get('infeasible')}, minimal={v.get('minimal')})")
    if v["status"] != "ok" or not v.get("passed"):
        for err in v.get("errors") or []:
            log(f"{name}: {err}")
    return finish(v["status"])


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------


def pass_mark(rec: dict) -> Optional[bool]:
    """Did the submission pass: True, False, or None when that was never decided.

    A program failure or an invalid solution file fails.  An undecided check
    passes nobody but still fails a submission with an invalid certificate, which
    needs no solve to establish.
    """
    status = rec.get("status")
    v = rec.get("verify") or {}
    if status in ("invalid_solution", *SOLVE_FAILURES):
        return False
    if status == "ok":
        return bool(v.get("passed"))
    if status == "verify_unknown" and v.get("invalid_certificates"):
        return False
    return None


def _pct(part: Optional[float], whole: Optional[float]) -> Optional[float]:
    """100 * part / whole, or None when either is missing or *whole* is zero."""
    if part is None or not whole:
        return None
    return 100.0 * part / whole


# Every column of summary.csv, in order.
REPORT_FIELDS = [
    "name",
    "status",
    "passed",
    "infeasible",
    "minimal",
    *(
        f"{stage}_{kind}"
        for kind in ("rows", "bounds", "integers", "total")
        for stage in ("orig", "kept", "cert")
    ),
    "removed_total",
    "kept_rows_pct",
    "kept_bounds_pct",
    "kept_integers_pct",
    "kept_pct",
    "cert_pct",
    "invalid_certificates",
    "solver",
    "solver_status",
    "time_limit",
    "solve_time",
    "verify_wall_s",
    "iis_time",
    "solve_exit_code",
]


def report_row(rec: dict) -> dict:
    """One record flattened to :data:`REPORT_FIELDS`: raw numbers, None when not measured.

    ``kept_*_pct`` is the kept share of the original, per kind; ``cert_pct`` the
    share of kept elements whose certificate verified (100 is a proven IIS).  An
    invalid solution never got as far as a subsystem, so its kept counts are None
    rather than the zeros the checker left behind.  ``iis_time`` is the program's
    wall clock as the harness measured it.
    """
    v = rec.get("verify") or {}
    sizes = v.get("sizes") or {}
    stats = v.get("stats") or {}
    solve = rec.get("solve") or {}
    invalid = rec.get("status") == "invalid_solution"

    def sub(x):
        return None if invalid else x

    row = {
        "name": rec["name"],
        "status": rec.get("status"),
        "passed": pass_mark(rec),
        "infeasible": v.get("infeasible"),
        "minimal": sub(v.get("minimal")),
    }
    for kind in ("rows", "bounds", "integers"):
        row[f"orig_{kind}"] = sizes.get(f"original_{kind}")
        row[f"kept_{kind}"] = sub(sizes.get(f"kept_{kind}"))
        row[f"cert_{kind}"] = sub(sizes.get(f"certified_{kind}"))
    row["orig_total"] = sizes.get("original")
    row["kept_total"] = sub(sizes.get("size"))
    row["cert_total"] = sub(sizes.get("certified"))
    row["removed_total"] = sub(sizes.get("removed"))
    for kind in ("rows", "bounds", "integers"):
        row[f"kept_{kind}_pct"] = _pct(row[f"kept_{kind}"], row[f"orig_{kind}"])
    row["kept_pct"] = _pct(row["kept_total"], row["orig_total"])
    row["cert_pct"] = _pct(row["cert_total"], row["kept_total"])
    row["invalid_certificates"] = v.get("invalid_certificates")
    row["solver"] = stats.get("solver") or None
    row["solver_status"] = stats.get("solver_status")
    row["time_limit"] = rec.get("time_limit")
    row["solve_time"] = stats.get("solve_time")
    row["verify_wall_s"] = rec.get("verify_wall_s")
    row["iis_time"] = solve.get("wall_s")
    row["solve_exit_code"] = solve.get("exit_code")
    return {k: row[k] for k in REPORT_FIELDS}


def report_csv(rows: list[dict]) -> str:
    """*rows* as CSV for a bare ``pandas.read_csv``: empty cells for None."""
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=REPORT_FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows({k: ("" if v is None else v) for k, v in row.items()} for row in rows)
    return buf.getvalue()


def _cell(text: str, width: int) -> str:
    """Right-align in *width*; a negative width left-aligns, eliding the middle if too long.

    Model names share long prefixes and differ in their suffixes, so the middle is
    what goes.  Width 0 is the last column: no padding.
    """
    if width == 0:
        return text
    if width < 0:
        limit = -width
        if len(text) > limit:
            head = (limit - 2) // 2
            text = f"{text[:head]}..{text[len(text) - (limit - 2 - head) :]}"
        return f"{text:<{limit}}"
    return f"{text:>{width}}"


def _pct_cell(value: Optional[float]) -> str:
    """A percentage to one decimal; a non-zero share too small to show is "<0.1"."""
    if value is None:
        return "-"
    return "<0.1" if 0 < value < 0.05 else format(value, ".1f")


def _time_cell(value: Optional[float]) -> str:
    return "-" if value is None else format(value, ".2f")


# (label, width, render, describes the submission).  A submission-derived cell
# reads n/a when the subsystem was proven feasible or the solution was invalid.
TABLE_COLUMNS = [
    ("instance", -30, lambda r: r["name"], False),
    ("pass", 4, lambda r: {True: "Y", False: "N", None: "-"}[r["passed"]], False),
    ("rows%", 6, lambda r: _pct_cell(r["kept_rows_pct"]), True),
    ("bnds%", 6, lambda r: _pct_cell(r["kept_bounds_pct"]), True),
    ("ints%", 6, lambda r: _pct_cell(r["kept_integers_pct"]), True),
    ("cert%", 6, lambda r: _pct_cell(r["cert_pct"]), True),
    ("iis(s)", 8, lambda r: _time_cell(r["iis_time"]), False),
    ("check(s)", 8, lambda r: _time_cell(r["solve_time"]), False),
    ("status", 0, lambda r: r["status"] or "?", False),
]


def print_table(records: list) -> None:
    """One line per record.  ``rows%``/``bnds%``/``ints%`` are kept shares of the original.

    Every column, and the raw counts behind these shares, are in ``summary.csv``.
    """
    header = " ".join(_cell(label, width) for label, width, _, _ in TABLE_COLUMNS).rstrip()
    print(header)
    print("-" * len(header))
    for rec in records:
        row = report_row(rec)
        wrong = row["infeasible"] is False or row["status"] == "invalid_solution"
        cells = [
            _cell("n/a" if wrong and sub else render(row), width)
            for _, width, render, sub in TABLE_COLUMNS
        ]
        print(" ".join(cells).rstrip())


# --------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Run competitors' programs, check their solutions, and report."
    )
    ap.add_argument(
        "--instance-dir",
        type=Path,
        required=True,
        metavar="DIR",
        help=f"Directory of model files ({', '.join(MPS_EXTENSIONS)}).",
    )
    ap.add_argument(
        "--solution-dir",
        type=Path,
        required=True,
        metavar="DIR",
        help="Directory of solution files (<model>.json[.gz]); the program writes them here.",
    )
    ap.add_argument(
        "--instances",
        nargs="+",
        metavar="MODEL",
        help="Only these models (names, or paths to their model files).",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        metavar="DIR",
        help=f"Where results/, logs/ and the summaries go (default: {DEFAULT_OUT_DIR}).",
    )
    check = ap.add_argument_group("checking")
    check.add_argument(
        "--solver",
        choices=("scip", "gurobi", "highs"),
        help="Backend for the infeasibility check (default: gurobi, else scip, else highs).",
    )
    check.add_argument(
        "--verify-time-limit",
        type=float,
        metavar="S",
        help="Cap each infeasibility check at S seconds; hitting it leaves the model undecided.",
    )
    solve = ap.add_argument_group("running the competitor's program (see PROGRAM_INTERFACE.md)")
    launch = solve.add_mutually_exclusive_group()
    launch.add_argument(
        "--container",
        metavar="IMAGE",
        help="Run this image per model under podman, sandboxed.",
    )
    launch.add_argument(
        "--solve-cmd",
        metavar="TEMPLATE",
        help=(
            "Run this command per model instead, with {instance}, {solution} and {name} "
            "filled in (no shell, no sandbox: trusted programs only)."
        ),
    )
    solve.add_argument(
        "--container-args",
        default="",
        metavar="ARGS",
        help="Extra 'podman run' options, e.g. \"--cpus 8 --memory 32g\".",
    )
    solve.add_argument(
        "--container-network",
        default="none",
        metavar="MODE",
        help="Network for the container (default: none); e.g. host, for a WLS license.",
    )
    solve.add_argument(
        "--container-pids-limit",
        type=int,
        default=4096,
        metavar="N",
        help="Process limit for the container (default: 4096); 0 for none, where rootless "
        "podman has no cgroup to set limits in (e.g. under Slurm).",
    )
    solve.add_argument(
        "--podman",
        default="podman",
        metavar="CMD",
        help='The podman command, with global options, e.g. "podman --root DIR".',
    )
    solve.add_argument(
        "--solve-time-limit",
        type=float,
        metavar="S",
        help="Wall-clock limit for the program (required with --container/--solve-cmd).",
    )
    solve.add_argument("--threads", type=int, metavar="N", help="Passed as MIPCC_THREADS.")
    solve.add_argument(
        "--overwrite-solutions",
        action="store_true",
        help="Let the program replace solution files this harness did not write (by default "
        "such a file stops the run before anything is deleted).",
    )
    ap.add_argument("--force", action="store_true", help="Ignore cached results.")
    return ap


def _launcher(args, ap) -> Optional[Launcher]:
    """The launcher the arguments describe, validated, or None."""
    if not (args.container or args.solve_cmd):
        if args.solve_time_limit is not None or args.threads is not None:
            ap.error("--solve-time-limit/--threads only apply with --container or --solve-cmd")
        return None
    if args.solve_time_limit is None or args.solve_time_limit <= 0:
        ap.error("--container/--solve-cmd need a positive --solve-time-limit")
    if args.threads is not None and args.threads <= 0:
        ap.error(f"--threads must be positive, got {args.threads}")
    if args.container_pids_limit < 0:
        ap.error(f"--container-pids-limit must be >= 0, got {args.container_pids_limit}")
    if args.container:
        launcher = Launcher(
            image=args.container,
            container_args=tuple(shlex.split(args.container_args)),
            podman=tuple(shlex.split(args.podman)),
            network=args.container_network,
            pids_limit=args.container_pids_limit,
        )
        try:
            return resolve_image(launcher)
        except RuntimeError as e:
            ap.error(str(e))
    template = tuple(shlex.split(args.solve_cmd))
    try:
        for token in template:
            token.format(**dict.fromkeys(TEMPLATE_FIELDS, ""))
    except (KeyError, IndexError, ValueError) as e:
        ap.error(f"--solve-cmd: bad placeholder {e} (use {{instance}}, {{solution}}, {{name}})")
    return Launcher(template=template)


def _raise_interrupt(signum, frame):
    """Make SIGTERM (``scancel``, ``kill``) stop a run like Ctrl-C, killing the program first."""
    raise KeyboardInterrupt(f"terminated by {signal.Signals(signum).name}")


def main() -> int:
    signal.signal(signal.SIGTERM, _raise_interrupt)
    ap = _parser()
    args = ap.parse_args()
    if args.verify_time_limit is not None and args.verify_time_limit <= 0:
        ap.error(f"--verify-time-limit must be positive, got {args.verify_time_limit}")
    launcher = _launcher(args, ap)
    if launcher is not None:
        args.solution_dir.mkdir(parents=True, exist_ok=True)
    for flag, path in (
        ("--instance-dir", args.instance_dir),
        ("--solution-dir", args.solution_dir),
    ):
        if not path.is_dir():
            ap.error(f"{flag} is not a directory: {path}")

    result_dir = args.out_dir / "results"
    solve = None
    if launcher is not None:
        solve = SolveConfig(launcher, args.solve_time_limit, args.threads, args.out_dir / "logs")

    # Which models: those named; else every model in the corpus when the program is
    # run (one that produced nothing must show up); else the solutions present.
    if args.instances:
        names = normalize_instance_names(args.instances, ap.error)
    elif solve is not None:
        names = model_names(args.instance_dir)
    else:
        names = solution_names(args.solution_dir)
    if not names:
        ap.error("no models to process")

    if solve is not None and not args.overwrite_solutions:
        # Refuse before the first model, not half-way through a long run.
        foreign = []
        for name in names:
            prev = (_read_json(result_dir / f"{name}.json") or {}).get("solve")
            foreign += foreign_solutions(name, args.solution_dir, prev)
        if foreign:
            shown = ", ".join(p.name for p in foreign[:5]) + (", ..." if len(foreign) > 5 else "")
            ap.error(
                f"{len(foreign)} solution file(s) in {args.solution_dir} were not written by this "
                f"harness and would be deleted ({shown}); use an empty --solution-dir, or "
                "--overwrite-solutions"
            )

    records = []
    for i, name in enumerate(names, 1):
        log(f"=== [{i}/{len(names)}] {name} ===")
        rec = run_one(
            name,
            args.instance_dir,
            args.solution_dir,
            result_dir,
            args.solver or "",
            args.verify_time_limit,
            args.force,
            solve,
            args.overwrite_solutions,
        )
        log(f"{name} -> {rec['status']}")
        records.append(rec)

    marks = [pass_mark(r) for r in records]
    log(f"{marks.count(True)} passed, {marks.count(False)} failed, {marks.count(None)} undecided")
    print()
    print_table(records)

    # Per-model jobs (--instances) share an --out-dir, so only whole-set runs write
    # the summaries; benchmark.collect writes them from what the jobs left.
    if not args.instances:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        _write_json(args.out_dir / "summary.json", records)
        _write_text(args.out_dir / "summary.csv", report_csv([report_row(r) for r in records]))
        print(f"\nSummary written to {args.out_dir / 'summary.json'} and summary.csv")

    # Non-zero only when the harness failed to reach a verdict somewhere: a
    # submission that failed is a result, not an error.
    return 0 if all(r["status"] in VERDICTS for r in records) else 1


if __name__ == "__main__":
    sys.exit(main())
