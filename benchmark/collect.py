"""
Collect the results that ``benchmark.run`` jobs left in an ``--out-dir``.

Reads ``<out-dir>/results/<model>.json`` and writes ``summary.json`` and
``summary.csv`` next to it; runs neither the program nor the checker, and writes
nothing else.  Needs no instances or solutions: given, they add the models never
run (``--instance-dir``) and flag results whose files changed since (both dirs).
``--only-started`` drops the models whose job never started.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

from mipcc_iis_checker.naming import find_model, model_names

from .run import (
    CHECKER,
    DEFAULT_OUT_DIR,
    SCHEMA,
    VERDICTS,
    _checker_label,
    _read_json,
    _write_json,
    _write_text,
    find_solution,
    is_model_name,
    log,
    normalize_instance_names,
    pass_mark,
    print_table,
    report_csv,
    report_row,
)
from .solve import file_sha256


def result_names(result_dir: Path) -> list[str]:
    """The models with a result record in *result_dir*."""
    return sorted(p.stem for p in result_dir.glob("*.json") if is_model_name(p.stem))


def was_started(name: str, out_dir: Path) -> bool:
    """Whether a job ran *name*: it left a result, or at least the program's log."""
    return (out_dir / "results" / f"{name}.json").exists() or (
        out_dir / "logs" / f"{name}.log"
    ).exists()


def collect_one(
    name: str,
    result_dir: Path,
    instance_dir: Optional[Path] = None,
    solution_dir: Optional[Path] = None,
) -> dict:
    """What the cache says about *name*.

    No result is ``not_run``.  A checked result is ``stale`` when a directory given
    no longer holds the file it checked (a directory not given is not looked at), or
    when another checker version or tolerance made it.
    """
    rec = _read_json(result_dir / f"{name}.json")
    if rec is None or rec.get("schema") != SCHEMA or rec.get("name") != name:
        return {"name": name, "status": "not_run"}
    inputs = rec.get("inputs")
    if inputs is None:  # stopped before the check: report how
        return rec
    for kind, directory, find in (
        ("solution", solution_dir, find_solution),
        ("instance", instance_dir, find_model),
    ):
        if directory is None:
            continue
        path = find(name, directory)
        if path is None or file_sha256(path) != inputs.get(f"{kind}_sha256"):
            log(f"{name}: the {kind} in {directory} is not the one that was checked")
            return {"name": name, "status": "stale", "solve": rec.get("solve")}
    if rec.get("checker") != CHECKER:
        log(f"{name}: checked by another checker ({_checker_label(rec.get('checker'))})")
        return {"name": name, "status": "stale", "solve": rec.get("solve")}
    return rec


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Summarize the results in an --out-dir; runs nothing.")
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        metavar="DIR",
        help=f"Where benchmark.run wrote results/ (default: {DEFAULT_OUT_DIR}).",
    )
    ap.add_argument(
        "--instance-dir",
        type=Path,
        metavar="DIR",
        help="Report every model here, even those never run; flag changed instances.",
    )
    ap.add_argument(
        "--solution-dir",
        type=Path,
        metavar="DIR",
        help="Flag results whose solution here is not the one that was checked.",
    )
    ap.add_argument(
        "--instances",
        nargs="+",
        metavar="MODEL",
        help="Only these models (names or model files).",
    )
    ap.add_argument(
        "--only-started",
        action="store_true",
        help="Only the models whose job started (a result or a log in --out-dir): with"
        " --instance-dir, a model never launched is left out rather than not_run.",
    )
    return ap


def main() -> int:
    ap = _parser()
    args = ap.parse_args()
    for flag, path in (
        ("--instance-dir", args.instance_dir),
        ("--solution-dir", args.solution_dir),
    ):
        if path is not None and not path.is_dir():
            ap.error(f"{flag} is not a directory: {path}")

    result_dir = args.out_dir / "results"
    if args.instances:
        names = normalize_instance_names(args.instances, ap.error)
    elif args.instance_dir is not None:
        names = model_names(args.instance_dir)
    else:
        names = result_names(result_dir)
    if args.only_started:
        names = [n for n in names if was_started(n, args.out_dir)]
    if not names:
        ap.error(f"no models to report (no results in {result_dir})")

    records = [collect_one(n, result_dir, args.instance_dir, args.solution_dir) for n in names]
    marks = [pass_mark(r) for r in records]
    log(f"{marks.count(True)} passed, {marks.count(False)} failed, {marks.count(None)} undecided")
    print()
    print_table(records)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    _write_json(args.out_dir / "summary.json", records)
    _write_text(args.out_dir / "summary.csv", report_csv([report_row(r) for r in records]))
    print(f"\nSummary written to {args.out_dir / 'summary.json'} and summary.csv")
    return 0 if all(r["status"] in VERDICTS for r in records) else 1


if __name__ == "__main__":
    sys.exit(main())
