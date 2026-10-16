"""
Core verification logic for MIP Competition 2027 Solution Checker.

Verification pipeline:
1. Validate all referenced elements exist in the instance.
2. Check the kept subsystem MIP(C, J) is infeasible (solver).
3. For each kept element that carries a certificate: verify the certificate is a
   feasible point for the subsystem with that element removed (algebraic, no solver).
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np

from ._version import __version__
from .backend import (
    STATUS_ERROR,
    STATUS_INFEASIBLE,
    STATUS_UNKNOWN,
    ElementCounts,
    get_backend,
)
from .io import parse_solution_file
from .naming import MPS_EXTENSIONS, mps_extension

# Absolute feasibility tolerance for certificates: slack allowed on each row and
# bound, and distance allowed from the nearest integer.  The competition's rule,
# stated in SOLUTION_FORMAT.md; changing it changes which submissions pass.
DEFAULT_TOLERANCE = 1e-5

# Solver statuses that decide nothing (neither feasible nor infeasible).
_INCONCLUSIVE_STATUSES = frozenset((STATUS_UNKNOWN, STATUS_ERROR))

_PathLike = Union[str, Path]


def _element_label(element_type: str, element: object) -> str:
    if element_type == "row":
        return f"row:{element}"
    if element_type == "lower_bound":
        return f"bound:{element}:lower"
    if element_type == "upper_bound":
        return f"bound:{element}:upper"
    return f"integer:{element}"


@dataclass
class VerificationStats:
    # Backend that ran, as get_backend() spells it.
    solver: str = ""
    # Status from solve_subsystem; None when no solve was attempted.
    solver_status: Optional[str] = None
    # Solver seconds on the subsystem; None when no solve produced a measurement.
    solve_time: Optional[float] = None


@dataclass
class VerificationResult:
    is_valid_input: bool = True
    # False when the solution file itself is at fault: it cannot be parsed, or it
    # names elements the instance lacks.  Implies is_valid_input is False; the other
    # ways that goes False (no solver, unreadable instance) are not the submitter's.
    is_valid_solution: bool = True
    # True  = the kept subsystem is proven infeasible
    # False = it is proven feasible (the submission is wrong)
    # None  = the solver could not decide (timeout/error — not the submitter's fault)
    is_infeasible: Optional[bool] = None
    # True when the kept set is non-empty and every kept element has a verified
    # certificate.  False means only "not proven" — the checker never refutes
    # minimality, and this says nothing about infeasibility.
    is_minimal: bool = False

    errors: list[str] = field(default_factory=list)
    # Things the checker did differently from what the solution says, without failing
    # it: e.g. a relaxed bound of a variable that stays integer is applied anyway.
    notes: list[str] = field(default_factory=list)

    # (element_label, violations) per certificate that failed verification.
    invalid_certificates: list[tuple[str, list[str]]] = field(default_factory=list)

    original: ElementCounts = field(default_factory=ElementCounts)
    removed: ElementCounts = field(default_factory=ElementCounts)
    kept: ElementCounts = field(default_factory=ElementCounts)
    # Kept elements whose certificate verified (a failed one is a claim, not a proof).
    certified: ElementCounts = field(default_factory=ElementCounts)

    stats: VerificationStats = field(default_factory=VerificationStats)

    # What produced this verdict: a report is only comparable with another from
    # the same checker version and certificate tolerance.
    checker_version: str = __version__
    tolerance: float = DEFAULT_TOLERANCE

    @property
    def is_passed(self) -> bool:
        return self.is_valid_input and bool(self.is_infeasible) and not self.invalid_certificates

    def to_dict(self) -> dict[str, Any]:
        return {
            "checker_version": self.checker_version,
            "tolerance": self.tolerance,
            "passed": self.is_passed,
            "valid_input": self.is_valid_input,
            "valid_solution": self.is_valid_solution,
            "infeasible": self.is_infeasible,
            "minimal": self.is_minimal,
            "errors": self.errors,
            "notes": self.notes,
            "invalid_certificates": [
                {"element": e, "violations": v} for e, v in self.invalid_certificates
            ],
            "counts": {
                "removed_rows": self.removed.rows,
                "removed_bounds": self.removed.bounds,
                "removed_integers": self.removed.integers,
                "total_removed": self.removed.total,
                "kept_rows": self.kept.rows,
                "kept_bounds": self.kept.bounds,
                "kept_integers": self.kept.integers,
                "total_kept": self.kept.total,
                "certified_rows": self.certified.rows,
                "certified_bounds": self.certified.bounds,
                "certified_integers": self.certified.integers,
                "total_certified": self.certified.total,
                "original_rows": self.original.rows,
                "original_bounds": self.original.bounds,
                "original_integers": self.original.integers,
                "total_original": self.original.total,
            },
            "stats": {
                "solver": self.stats.solver,
                "solver_status": self.stats.solver_status,
                "solve_time": self.stats.solve_time,
            },
        }

    def summary(self) -> str:
        tri = {True: "True", False: "False", None: "undetermined"}
        passed_str = {True: "PASSED", False: "FAILED"}[self.is_passed]

        lines = [
            f"=== MIP Competition 2027 Solution Verification: {passed_str} ===",
            f"Checker:                 mipcc-iis-checker {self.checker_version}"
            f" (certificate tolerance {self.tolerance:g})",
            f"Valid Input Elements:    {self.is_valid_input}",
            f"Subsystem Infeasible:    {tri[self.is_infeasible]}",
            f"Subsystem Minimality:    {'proven' if self.is_minimal else 'not proven'}",
        ]

        if self.errors:
            lines += ["\nErrors:"] + [f"  - {e}" for e in self.errors]

        if self.notes:
            lines += ["\nNotes:"] + [f"  - {n}" for n in self.notes]

        if self.invalid_certificates:
            lines.append("\nInvalid Certificates:")
            for elem, viols in self.invalid_certificates:
                lines.append(f"  Element: {elem}")
                lines += [f"    * {v}" for v in viols[:5]]
                if len(viols) > 5:
                    lines.append(f"    ... ({len(viols) - 5} more)")

        lines.extend(self._format_removal_stats())
        lines.extend(self._format_stats())

        return "\n".join(lines)

    def _format_removal_stats(self) -> list[str]:
        def _stat(label, rem, orig, kept, cert):
            return f"  - {label:<26} {rem} / {orig} (Kept: {kept}, Certified: {cert})"

        return [
            "\nSubsystem Removal Statistics:",
            _stat(
                "Removed Linear Rows:",
                self.removed.rows,
                self.original.rows,
                self.kept.rows,
                self.certified.rows,
            ),
            _stat(
                "Removed Variable Bounds:",
                self.removed.bounds,
                self.original.bounds,
                self.kept.bounds,
                self.certified.bounds,
            ),
            _stat(
                "Removed Integer Vars:",
                self.removed.integers,
                self.original.integers,
                self.kept.integers,
                self.certified.integers,
            ),
            _stat(
                "Total Elements Removed:",
                self.removed.total,
                self.original.total,
                self.kept.total,
                self.certified.total,
            ),
        ]

    def _format_stats(self) -> list[str]:
        solve_time = self.stats.solve_time
        solve_time_str = "n/a" if solve_time is None else f"{solve_time:.2f}s"

        return [
            "\nStatistics:",
            f"  - {'Solver:':<26} {self.stats.solver or 'n/a'}",
            f"  - {'Solver Status:':<26} {self.stats.solver_status or 'n/a'}",
            f"  - {'Subsystem Solve Time:':<26} {solve_time_str}",
        ]


def _validate_solution_elements(model, sol_data, res: VerificationResult) -> bool:
    """False, with an error on *res*, if the solution names elements the instance lacks."""
    for entries, instance_set, label in [
        (sol_data.rows, model.rows, "rows"),
        (sol_data.lower_bounds, model.lower_bounds, "lower bounds"),
        (sol_data.upper_bounds, model.upper_bounds, "upper bounds"),
        (sol_data.integers, model.integers, "integer variables"),
    ]:
        invalid = {k for k in entries if k not in instance_set}
        if invalid:
            res.is_valid_input = False
            res.is_valid_solution = False
            res.errors.append(f"Solution references {label} not in instance: {sorted(invalid)}")

    return res.is_valid_input


def _check_subsystem_infeasibility(
    backend,
    kept_rows,
    kept_lower_bounds,
    kept_upper_bounds,
    kept_integers,
    res: VerificationResult,
    time_limit: Optional[float] = None,
) -> None:
    """Record the solver's verdict on the kept subsystem in *res*."""
    try:
        subsystem_status, solve_time = backend.solve_subsystem(
            kept_rows, kept_lower_bounds, kept_upper_bounds, kept_integers, time_limit=time_limit
        )
    except Exception as e:  # an unusable solver is inconclusive, not a failed submission
        res.stats.solver_status = STATUS_ERROR
        res.is_infeasible = None
        res.errors.append(f"Solver failed while checking subsystem MIP(C, J): {e}")
        return
    res.stats.solver_status = subsystem_status
    res.stats.solve_time = solve_time

    if subsystem_status == STATUS_INFEASIBLE:
        res.is_infeasible = True
        return

    # "Not proven infeasible" and "proven feasible" both fail
    res.is_infeasible = None if subsystem_status in _INCONCLUSIVE_STATUSES else False
    if res.is_infeasible is None:
        res.errors.append(
            f"Could not determine whether subsystem MIP(C, J) is infeasible"
            f" (solver status: {subsystem_status})."
        )
    else:
        res.errors.append(
            f"Subsystem MIP(C, J) is not infeasible (solver status: {subsystem_status})."
        )


def _note_integer_bounds(sol_data, kept_integers, res: VerificationResult) -> None:
    """Note what the solution says about the bounds of variables that stay integer.

    Those bounds are not elements of the subsystem, so relaxing one, or giving it a
    certificate, has no effect; say so rather than fail the solution.
    """
    overridden = (sol_data.removed_lower_bounds | sol_data.removed_upper_bounds) & kept_integers
    if overridden:
        res.notes.append(
            f"{len(overridden)} variable(s) stay integer but have a bound marked relax, e.g."
            f" {sorted(overridden)[0]!r}: bounds of integer variables always apply, so these"
            " were kept."
        )
    certified = {
        name
        for entries in (sol_data.lower_bounds, sol_data.upper_bounds)
        for name, entry in entries.items()
        if name in kept_integers and entry.action == "keep" and entry.certificate is not None
    }
    if certified:
        res.notes.append(
            f"{len(certified)} bound certificate(s) of variables that stay integer were ignored,"
            f" e.g. {sorted(certified)[0]!r}: such bounds need no certificate."
        )


def _verify_certificates(
    kept_model,
    sol_data,
    kept_rows,
    kept_lower_bounds,
    kept_upper_bounds,
    kept_integers,
    tolerance,
    res: VerificationResult,
) -> None:
    """Check each kept element's certificate against the subsystem without it."""
    certified = ElementCounts()

    for element_type, kept, collection in [
        ("row", kept_rows, sol_data.rows),
        ("lower_bound", kept_lower_bounds, sol_data.lower_bounds),
        ("upper_bound", kept_upper_bounds, sol_data.upper_bounds),
        ("integer", kept_integers, sol_data.integers),
    ]:
        for element in kept:
            entry = collection.get(element)
            if entry is None or entry.certificate is None:
                continue
            valid, violations = kept_model.is_feasible(
                entry.certificate, tolerance, ignore=(element_type, element)
            )
            if not valid:
                res.invalid_certificates.append((_element_label(element_type, element), violations))
            elif element_type == "row":
                certified.rows += 1
            elif element_type == "integer":
                certified.integers += 1
            else:
                certified.bounds += 1

    res.certified = certified

    if res.invalid_certificates:
        res.errors.append(
            f"{len(res.invalid_certificates)} certificate(s) failed algebraic verification."
        )

    # Equal totals mean every kept element verified; an empty kept set proves nothing.
    res.is_minimal = res.kept.total > 0 and res.certified.total == res.kept.total


def verify_subsystem(
    instance_path: _PathLike,
    solution_path: _PathLike,
    verbose: bool = False,
    tolerance: float = DEFAULT_TOLERANCE,
    solver: str = "",
    time_limit: Optional[float] = None,
) -> VerificationResult:
    """Verify a JSON solution file against an MPS instance.

    Certificates are verified whatever the infeasibility solve returned: the two
    verdicts are independent and certificate checking needs no solver.

    tolerance   absolute feasibility tolerance for certificates.
    solver      "" auto-selects, else "gurobi" / "scip" / "highs".
    time_limit  seconds for the infeasibility solve; hitting it leaves
                is_infeasible=None (inconclusive, not a failure).
    """
    if not (tolerance > 0) or not np.isfinite(tolerance):
        raise ValueError(f"tolerance must be a positive finite float, got {tolerance!r}")

    res = VerificationResult(tolerance=tolerance)

    # Enforced here and not only in the CLI (see naming.mps_extension for why).
    if mps_extension(instance_path) is None:
        res.is_valid_input = False
        res.errors.append(
            f"Instance must be an MPS file ({', '.join(MPS_EXTENSIONS)}): {instance_path}"
        )
        return res

    try:
        backend = get_backend(solver=solver, verbose=verbose)
    except Exception as e:
        res.is_valid_input = False
        res.errors.append(f"Failed to initialize solver backend: {e}")
        return res

    res.stats.solver = backend.name

    try:
        backend.load_mps(str(instance_path))
    except Exception as e:
        res.is_valid_input = False
        res.errors.append(f"Failed to load MPS instance '{instance_path}': {e}")
        return res

    model = backend.np_model
    # The full model's size under the same rule as a subsystem's: every integer
    # variable is in J there, so no bound of an integer variable counts.
    res.original = ElementCounts.of(
        model.rows,
        model.lower_bounds - model.integers,
        model.upper_bounds - model.integers,
        model.integers,
    )

    try:
        sol_data = parse_solution_file(solution_path)
    except Exception as e:
        res.is_valid_input = False
        res.is_valid_solution = False
        res.errors.append(f"Failed to parse solution file '{solution_path}': {e}")
        return res

    if not _validate_solution_elements(model, sol_data, res):
        return res

    kept_rows = model.rows - sol_data.removed_rows
    kept_integers = model.integers - sol_data.removed_integers

    # Competition rule: the bounds of a variable that stays integer (j in J) always
    # belong to the subsystem.  They are applied in the infeasibility check and in
    # every certificate check whatever the solution says, are not counted in its
    # size, and need no certificate.  Only the bounds of the other variables are
    # elements a solution can keep or relax.
    _note_integer_bounds(sol_data, kept_integers, res)
    kept_lower_bounds = model.lower_bounds - sol_data.removed_lower_bounds - kept_integers
    kept_upper_bounds = model.upper_bounds - sol_data.removed_upper_bounds - kept_integers
    applied_lower_bounds = kept_lower_bounds | (model.lower_bounds & kept_integers)
    applied_upper_bounds = kept_upper_bounds | (model.upper_bounds & kept_integers)

    res.removed = ElementCounts.of(
        sol_data.removed_rows,
        sol_data.removed_lower_bounds - kept_integers,
        sol_data.removed_upper_bounds - kept_integers,
        sol_data.removed_integers,
    )
    res.kept = ElementCounts.of(kept_rows, kept_lower_bounds, kept_upper_bounds, kept_integers)

    _check_subsystem_infeasibility(
        backend,
        kept_rows,
        applied_lower_bounds,
        applied_upper_bounds,
        kept_integers,
        res,
        time_limit=time_limit,
    )

    # Certificates are asked of the counted elements only, but checked against the
    # subsystem as applied.  Relaxing the integrality of j leaves x_j's bounds in
    # place: only that one element is removed.
    kept_model = model.sub_system(
        kept_rows, applied_lower_bounds, applied_upper_bounds, kept_integers
    )
    _verify_certificates(
        kept_model,
        sol_data,
        kept_rows,
        kept_lower_bounds,
        kept_upper_bounds,
        kept_integers,
        tolerance,
        res,
    )

    return res
