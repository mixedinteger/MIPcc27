#!/usr/bin/env python3
"""
MIP Competition 2027 Solution Checker CLI

Verifies a JSON solution file against a MIP instance in MPS format.
Solutions describe which elements are relaxed/kept, with optional feasibility
certificates proving each kept element is essential.

Usage:
  python -m mipcc_iis_checker --instance <problem.mps> --solution <solution.json>
"""

import argparse
import json
import sys
from pathlib import Path

from . import __version__, verify_subsystem
from .checker import DEFAULT_TOLERANCE
from .naming import MPS_EXTENSIONS, mps_extension


def main():
    parser = argparse.ArgumentParser(
        # Console script and ``python -m mipcc_iis_checker`` share this parser.
        prog="mipcc-iis-checker",
        description=(
            "MIP Competition 2027 Solution Checker: verifies infeasibility and optional"
            " certificate-based minimality."
        ),
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument(
        "--instance",
        "-i",
        required=True,
        type=str,
        help=f"Path to input instance ({'/'.join(MPS_EXTENSIONS)})",
    )
    parser.add_argument(
        "--solution",
        "-s",
        required=True,
        type=str,
        help="Path to solution file (.json or .json.gz; see SOLUTION_FORMAT.md)",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=DEFAULT_TOLERANCE,
        help=(
            "Absolute feasibility tolerance for certificate verification"
            f" (default: {DEFAULT_TOLERANCE:g}, the competition's rule; judging uses the default)."
        ),
    )
    parser.add_argument(
        "--solver",
        choices=("scip", "gurobi", "highs"),
        default=None,
        help="Solver backend for the infeasibility check (default: auto-select).",
    )
    parser.add_argument(
        "--time-limit",
        type=float,
        default=None,
        metavar="S",
        help=(
            "Cap the infeasibility solve at S seconds (default: no cap). Hitting the"
            " limit reports the subsystem as undetermined rather than failing the"
            " submission."
        ),
    )
    parser.add_argument(
        "--output-json", "-o", type=str, default=None, help="Save verification report to JSON file."
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Print detailed solver output."
    )

    args = parser.parse_args()

    if not Path(args.instance).exists():
        parser.error(f"Instance file not found: {args.instance}")
    if mps_extension(args.instance) is None:
        parser.error(
            f"--instance must be an MPS file ({', '.join(MPS_EXTENSIONS)}): {args.instance}"
        )
    if not Path(args.solution).exists():
        parser.error(f"Solution file not found: {args.solution}")

    if not (args.tolerance > 0) or args.tolerance == float("inf"):
        parser.error(f"--tolerance must be a positive finite float, got {args.tolerance}")

    if args.time_limit is not None and args.time_limit <= 0:
        parser.error(f"--time-limit must be positive, got {args.time_limit}")

    result = verify_subsystem(
        instance_path=args.instance,
        solution_path=args.solution,
        verbose=args.verbose,
        tolerance=args.tolerance,
        solver=args.solver or "",
        time_limit=args.time_limit,
    )

    print(result.summary())

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
        print(f"\nVerification results saved to {out_path}")

    sys.exit(0 if result.is_passed else 1)


if __name__ == "__main__":
    main()
