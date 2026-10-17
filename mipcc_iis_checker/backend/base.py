"""
Backend base: NpModel holds the problem data as numpy/scipy arrays, BaseBackend is
the interface every solver backend implements, plus the shared status constants.

Parsing an MPS file is a backend's job; the checks every backend must agree on live here.
"""

import gzip
from abc import ABC, abstractmethod
from collections.abc import Mapping
from collections.abc import Set as AbstractSet
from dataclasses import dataclass
from fractions import Fraction
from functools import cached_property
from typing import ClassVar, Optional

import numpy as np
import scipy.sparse as sp

STATUS_INFEASIBLE = "INFEASIBLE"
STATUS_FEASIBLE = "FEASIBLE"
STATUS_UNKNOWN = "UNKNOWN"
STATUS_ERROR = "ERROR"

# Absolute feasibility tolerance: slack allowed on each row and bound, and distance
# allowed from the nearest integer.  The competition's rule, stated in
# SOLUTION_FORMAT.md; changing it changes which submissions pass.  Certificates are
# held to it, and so is the point a solver offers as proof that a subsystem is feasible.
DEFAULT_TOLERANCE = 1e-5

# The loosest tolerance accepted: the solvers are run at it too, and Gurobi takes a
# feasibility tolerance of at most 1e-2.
MAX_TOLERANCE = 1e-2

# Solver tolerances are relative or applied to a scaled model, so a point a solver
# calls feasible can still miss DEFAULT_TOLERANCE on the original rows.  Such a point
# earns one more solve at this tighter tolerance, the smallest Gurobi takes.
RETRY_SOLVER_TOLERANCE = 1e-9


class UnsupportedInstanceError(ValueError):
    """The instance uses features :class:`NpModel` cannot represent.

    NpModel holds exactly one thing: a system of linear rows over bounded,
    optionally-integer variables, one ``(sense, rhs)`` each.  Quadratic or general
    (indicator, min/max, abs, piecewise-linear, nonlinear) constraints, SOS sets,
    semi-continuous variables and ranged rows have nowhere to go, so a backend
    refuses the instance rather than load part of it.

    The ``N`` objective row is dropped on purpose — the checker only asks feasibility
    questions — but an objective needing a section of its own (``QUADOBJ``, ``PWLOBJ``)
    is refused all the same, because ``check_mps`` does not tell an objective
    section from a constraint one.
    """

    def __init__(self, filepath: str, unsupported: Mapping[str, int]) -> None:
        kinds = ", ".join(f"{kind} ({count})" for kind, count in sorted(unsupported.items()))
        super().__init__(
            f"Instance '{filepath}' uses features the checker cannot represent: {kinds}."
            " Only linear constraints over continuous and integer variables are supported."
        )
        self.filepath = filepath
        self.unsupported = dict(unsupported)


class AmbiguousInstanceError(UnsupportedInstanceError):
    """The instance is written in a way the backends' MPS readers read differently.

    Each backend parses the file with its own solver's reader, so a construct the
    readers disagree on would make the instance — and every verdict on it — depend
    on which solver is installed.  Such an instance is refused by every backend.
    """

    def __init__(self, filepath: str, ambiguous: Mapping[str, int]) -> None:
        kinds = ", ".join(f"{kind} ({count})" for kind, count in sorted(ambiguous.items()))
        ValueError.__init__(
            self,
            f"Instance '{filepath}' is read differently by different MPS readers: {kinds}."
            " Write the value out explicitly (or drop it) so that every reader agrees.",
        )
        self.filepath = filepath
        self.unsupported = dict(ambiguous)


def validate_instance(filepath: str, constraint_names, variable_names, A, rhs, senses) -> None:
    """Refuse an instance the name-keyed, finite-arithmetic model cannot represent.

    Called by every backend's ``load_mps``, so the two cannot disagree about which
    files are instances.
    """
    for kind, names in (("row", constraint_names), ("column", variable_names)):
        # Elements are keyed by name, so a duplicate makes one of them unaddressable.
        if len(set(names)) != len(names):
            raise ValueError(
                f"Malformed MPS file '{filepath}': a {kind} name is declared more than once"
            )

    # ``slack = A @ x - rhs`` goes NaN on inf - inf, and NaN compares false against
    # every tolerance, so the row would be satisfied by any certificate.  An infinite
    # rhs is only safe in the direction that makes the row vacuous.  Gurobi normalises
    # infinite bounds but not right-hand sides.
    sense = np.asarray(senses, dtype="U1")
    bad_rhs = np.isnan(rhs)
    bad_rhs |= (sense != ">") & (rhs == -np.inf)
    bad_rhs |= (sense != "<") & (rhs == np.inf)
    counts = {
        "non-finite constraint coefficients": int((~np.isfinite(A.data)).sum()),
        "unsatisfiable infinite right-hand sides": int(bad_rhs.sum()),
    }
    unsupported = {kind: count for kind, count in counts.items() if count}
    if unsupported:
        raise UnsupportedInstanceError(filepath, unsupported)


# The only MPS sections the checker accounts for.  Past ROWS/COLUMNS/RHS/BOUNDS/RANGES
# the format is not standardised, so what SOS, QCMATRIX, QUADOBJ, QMATRIX, PWLOBJ or
# INDICATORS means is the reader's choice.  RANGES is refused too, standard though it
# is: Gurobi's reader turns it into an equality plus a slack variable, whose bounds
# would be scored as elements nobody wrote.
_SUPPORTED_SECTIONS = frozenset(
    {"NAME", "ROWS", "COLUMNS", "RHS", "BOUNDS", "ENDATA", "OBJSENSE", "OBJSENS"}
)

# A nonzero value this small or smaller is zero to some readers only: SCIP and HiGHS
# drop such a coefficient and SCIP rounds such a right-hand side or bound to 0, while
# Gurobi keeps them all.
_TINY_VALUE = 1e-9

# Bound types that carry a value (``type set column value``) and those that do not
# (``type set column``).  Without its set name a line is one field short: Gurobi
# refuses it and SCIP reads the column as the set name, while HiGHS accepts it.
_VALUED_BOUND_TYPES = frozenset({"UP", "LO", "FX", "LI", "UI"})
_LOWER_BOUND_TYPES = frozenset({"LO", "LI", "MI", "FR", "FX", "BV"})

_AMBIGUOUS_TINY = f"nonzero values of magnitude at most {_TINY_VALUE:g}"
_AMBIGUOUS_NEGATIVE_UP = "negative upper bounds on columns with no lower bound given"
_AMBIGUOUS_NO_SET_NAME = "RHS or BOUNDS lines without a set name"


def _number(field: str) -> float:
    """*field* as a number, or NaN (which compares false) if it is not one.

    A field that is not a number is the reader's to report.
    """
    try:
        return float(field)
    except ValueError:
        return float("nan")


def _is_tiny(field: str) -> bool:
    """Whether *field* is a nonzero number of magnitude at most ``_TINY_VALUE``."""
    return 0 < abs(_number(field)) <= _TINY_VALUE


def check_mps(filepath: str) -> None:
    """Refuse the instance if it uses an unsupported section or an ambiguous construct.

    The section check is a whitelist (see ``_SUPPORTED_SECTIONS``).  The ambiguity
    check refuses what the backends' readers would read differently, so the instance
    is the same whichever backend loads it:

    - a nonzero value of magnitude at most ``_TINY_VALUE`` in COLUMNS, RHS or BOUNDS;
    - a negative ``UP``/``UI`` bound on a column given no lower bound: Gurobi then
      makes the lower bound -inf, SCIP and HiGHS keep 0 (an empty domain);
    - an RHS or BOUNDS line without its set name.

    Objective (``N``) rows are skipped, since the checker drops the objective.  Lines
    whose field count does not fit the free format (names with spaces in fixed MPS)
    are left to the reader.  Section headers start in column 1, data lines are
    indented, and ``*`` in column 1 is a comment.
    """
    opener = gzip.open if str(filepath).endswith(".gz") else open
    found: list[str] = []
    ambiguous: dict[str, int] = {}
    objective_rows: set[str] = set()
    negative_up: set[str] = set()
    has_lower: set[str] = set()
    section = ""

    def flag(kind: str) -> None:
        ambiguous[kind] = ambiguous.get(kind, 0) + 1

    def check_values(pairs: list[str]) -> None:
        # (row, value) pairs; the objective row's values are dropped with it.
        for row, value in zip(pairs[::2], pairs[1::2]):
            if row not in objective_rows and _is_tiny(value):
                flag(_AMBIGUOUS_TINY)

    with opener(filepath, "rt", errors="replace") as handle:
        for line in handle:
            if not line.strip() or line.startswith("*"):
                continue
            fields = line.split()
            if line[:1].strip():
                section = fields[0].upper()
                if section not in _SUPPORTED_SECTIONS and section not in found:
                    found.append(section)
                continue

            if section == "ROWS" and len(fields) == 2 and fields[0].upper() == "N":
                objective_rows.add(fields[1])
            elif section == "COLUMNS" and len(fields) in (3, 5):
                if not any(f.strip("'\"").upper() == "MARKER" for f in fields):
                    check_values(fields[1:])
            elif section == "RHS":
                if len(fields) in (2, 4):
                    flag(_AMBIGUOUS_NO_SET_NAME)
                elif len(fields) in (3, 5):
                    check_values(fields[1:])
            elif section == "BOUNDS":
                kind = fields[0].upper()
                valued = kind in _VALUED_BOUND_TYPES
                if len(fields) < (4 if valued else 3):
                    flag(_AMBIGUOUS_NO_SET_NAME)
                    continue
                column = fields[2]
                if kind in _LOWER_BOUND_TYPES:
                    has_lower.add(column)
                if valued and len(fields) == 4:
                    if _is_tiny(fields[3]):
                        flag(_AMBIGUOUS_TINY)
                    if kind in ("UP", "UI") and _number(fields[3]) < 0:
                        negative_up.add(column)

    if found:
        raise UnsupportedInstanceError(filepath, {f"{name} section": 1 for name in found})
    if negative_up - has_lower:
        ambiguous[_AMBIGUOUS_NEGATIVE_UP] = len(negative_up - has_lower)
    if ambiguous:
        raise AmbiguousInstanceError(filepath, ambiguous)


# Half the gap between 1.0 and the next double: the relative error of one rounding.
_UNIT_ROUNDOFF = np.finfo(np.float64).eps / 2


def _fmt(value, spec: str = ".15g") -> str:
    """*value* (a float or Fraction) formatted as a float, even past float range."""
    try:
        return format(float(value), spec)
    except OverflowError:
        return "-huge" if value < 0 else "huge"


def _membership_mask(names: np.ndarray, wanted: AbstractSet[str]) -> np.ndarray:
    """Boolean mask of ``names`` entries present in *wanted*.

    ``np.isin`` falls back to an O(n*m) Python-level comparison on ``dtype=object``
    string arrays; a set lookup per name is an order of magnitude faster.
    """
    return np.fromiter((n in wanted for n in names), dtype=bool, count=len(names))


@dataclass
class ElementCounts:
    """How many elements of each kind, for one category (original/removed/kept).

    ``bounds`` counts individual bound constraints, so a variable with both a
    lower and an upper bound contributes two.
    """

    rows: int = 0
    bounds: int = 0
    integers: int = 0

    @classmethod
    def of(
        cls,
        rows: AbstractSet[str],
        lower_bounds: AbstractSet[str],
        upper_bounds: AbstractSet[str],
        integers: AbstractSet[str],
    ) -> "ElementCounts":
        return cls(len(rows), len(lower_bounds) + len(upper_bounds), len(integers))

    @property
    def total(self) -> int:
        return self.rows + self.bounds + self.integers


@dataclass
class NpModel:
    """
    Full problem data stored as numpy arrays.

    Shapes (m = #constraints, n = #variables):
        A               : (m, n) scipy CSR sparse matrix  — constraint coefficients
        senses          : (m,)   object ndarray            — sense chars: '<', '>', '='
        rhs             : (m,)   float64 ndarray           — right-hand side values
        lb              : (n,)   float64 ndarray           — variable lower bounds
        ub              : (n,)   float64 ndarray           — variable upper bounds
        is_integer      : (n,)   bool ndarray              — True where variable is integer/binary
        variable_names  : (n,)   object ndarray (str)
        constraint_names: (m,)   object ndarray (str)

    Infinite bounds are stored as ±np.inf.
    """

    A: sp.csr_matrix
    senses: np.ndarray
    rhs: np.ndarray
    lb: np.ndarray
    ub: np.ndarray
    is_integer: np.ndarray
    variable_names: np.ndarray
    constraint_names: np.ndarray

    @cached_property
    def var_index(self) -> dict[str, int]:
        return {n: i for i, n in enumerate(self.variable_names.tolist())}

    @cached_property
    def row_index(self) -> dict[str, int]:
        return {n: i for i, n in enumerate(self.constraint_names.tolist())}

    # The four element kinds a solution may relax.
    @cached_property
    def rows(self) -> frozenset[str]:
        """Names of the linear constraints."""
        return frozenset(self.constraint_names.tolist())

    @cached_property
    def integers(self) -> frozenset[str]:
        """Names of the variables constrained to integer values."""
        return frozenset(self.variable_names[self.is_integer].tolist())

    @cached_property
    def lower_bounds(self) -> frozenset[str]:
        """Names of the variables with a finite lower bound."""
        return frozenset(self.variable_names[np.isfinite(self.lb)].tolist())

    @cached_property
    def upper_bounds(self) -> frozenset[str]:
        """Names of the variables with a finite upper bound."""
        return frozenset(self.variable_names[np.isfinite(self.ub)].tolist())

    @property
    def element_counts(self) -> ElementCounts:
        return ElementCounts.of(self.rows, self.lower_bounds, self.upper_bounds, self.integers)

    def is_feasible(
        self,
        solution: dict[str, float],
        tolerance: float,
        ignore: Optional[tuple[str, str]] = None,
    ) -> tuple[bool, list[str]]:
        """Check whether *solution* is feasible for this model.

        Variables absent from *solution* default to 0.0.  Bounds stored as
        ±inf (i.e. relaxed by :meth:`sub_system`) are silently satisfied.
        Returns ``(is_feasible, violations)`` where *violations* holds
        human-readable descriptions of every unsatisfied constraint.

        *ignore* is a single ``(kind, name)`` element — ``kind`` being one of
        ``"row"``, ``"lower_bound"``, ``"upper_bound"``, ``"integer"`` — whose own
        violation is not counted, answering "is *solution* feasible for this model
        with that one element removed?".  *solution* must violate that element:
        satisfying it too would say the whole model is feasible, which proves nothing
        about the element being needed.
        """
        kind, ignored = ignore or ("", "")
        var_names = self.variable_names
        var_index = self.var_index
        violations: list[str] = []
        ignored_was_violated = False
        skip = (self.row_index if kind == "row" else var_index).get(ignored, -1)

        x = np.zeros(len(var_names), dtype=np.float64)
        unknown = []
        for name, value in solution.items():
            k = var_index.get(name)
            if k is None:
                unknown.append(name)
            else:
                x[k] = value
        if unknown:
            violations.append(f"solution names variables not in the instance: {sorted(unknown)}")

        # --- Row constraints ---
        senses = self.senses.astype("U1")
        is_le = np.isin(senses, ("<", "="))
        is_ge = np.isin(senses, (">", "="))
        slack, violated = self._row_slacks(x, is_le, is_ge, tolerance)
        for k in np.flatnonzero(violated):
            if kind == "row" and k == skip:
                ignored_was_violated = True
                continue
            op = {"<": ">", ">": "<", "=": "!="}[senses[k]]
            name = self.constraint_names[k]
            lhs = (
                slack[k] + Fraction(float(self.rhs[k]))
                if isinstance(slack[k], Fraction)
                else slack[k] + self.rhs[k]
            )
            violations.append(
                f"constraint {name!r}: lhs {_fmt(lhs)} {op} rhs {self.rhs[k]:.15g}"
                f" (violation {_fmt(abs(slack[k]), '.3g')})"
            )

        # --- Bound constraints (±inf bounds are vacuous — skip them) ---
        finite_lb = np.isfinite(self.lb)
        finite_ub = np.isfinite(self.ub)
        for k in np.flatnonzero(finite_lb & (x < self.lb - tolerance)):
            if kind == "lower_bound" and k == skip:
                ignored_was_violated = True
                continue
            violations.append(
                f"lower bound {self.variable_names[k]!r}: {x[k]:.15g} < {self.lb[k]:.15g}"
                f" (violation {self.lb[k] - x[k]:.3g})"
            )
        for k in np.flatnonzero(finite_ub & (x > self.ub + tolerance)):
            if kind == "upper_bound" and k == skip:
                ignored_was_violated = True
                continue
            violations.append(
                f"upper bound {self.variable_names[k]!r}: {x[k]:.15g} > {self.ub[k]:.15g}"
                f" (violation {x[k] - self.ub[k]:.3g})"
            )

        # --- Integrality ---
        int_vars = np.flatnonzero(self.is_integer)
        bad_int = int_vars[np.abs(x[int_vars] - np.round(x[int_vars])) > tolerance]
        for k in bad_int:
            if kind == "integer" and k == skip:
                ignored_was_violated = True
                continue
            violations.append(f"integrality {self.variable_names[k]!r}: {x[k]:.6g} is not integer")

        if ignore and not ignored_was_violated:
            violations.append(
                f"solution satisfies {kind} {ignored!r} too, so it does not show that element"
                " is needed (the subsystem would be feasible)"
            )

        return not violations, violations

    def _row_slacks(
        self, x: np.ndarray, is_le: np.ndarray, is_ge: np.ndarray, tolerance: float
    ) -> tuple[list, np.ndarray]:
        """Each row's ``activity - rhs`` at *x*, and whether it breaks *tolerance*.

        The floating-point activity carries a rounding error, and with large enough
        values that error decides the outcome: ``1 + 1e300 - 1e300`` computes as 0,
        and products can overflow.  Each row's error is bounded (a sum of n products
        is off by at most about ``n * eps/2`` times the sum of their magnitudes); a
        row whose verdict that error could flip is recomputed exactly, in rationals.
        Honest points never get near the bound, so the exact path is rare.

        Rows with an infinite rhs are vacuous: never violated.  Returned slacks are
        floats, or Fractions where the row was recomputed.
        """
        finite = np.isfinite(self.rhs)
        with np.errstate(over="ignore", invalid="ignore"):
            slack = self.A @ x - self.rhs
            magnitude = abs(self.A) @ np.abs(x) + np.abs(np.where(finite, self.rhs, 0.0))
            # Twice the textbook bound, to cover rounding in computing the bound itself.
            error = 2 * (np.diff(self.A.indptr) + 2) * _UNIT_ROUNDOFF * magnitude
            violated = finite & ((is_le & (slack > tolerance)) | (is_ge & (slack < -tolerance)))
            undecided = finite & (
                ~np.isfinite(slack)
                | ~np.isfinite(error)
                | (is_le & (np.abs(slack - tolerance) <= error))
                | (is_ge & (np.abs(slack + tolerance) <= error))
            )

        slack = slack.tolist()
        exact_tolerance = Fraction(tolerance)
        for k in np.flatnonzero(undecided):
            start, end = self.A.indptr[k], self.A.indptr[k + 1]
            coefficients = self.A.data[start:end].tolist()
            values = x[self.A.indices[start:end]].tolist()
            exact = sum(
                (Fraction(a) * Fraction(v) for a, v in zip(coefficients, values)),
                -Fraction(float(self.rhs[k])),
            )
            slack[k] = exact
            violated[k] = (is_le[k] and exact > exact_tolerance) or (
                is_ge[k] and exact < -exact_tolerance
            )
        return slack, violated

    def sub_system(
        self,
        kept_row_names: AbstractSet[str],
        kept_lb_names: AbstractSet[str],
        kept_ub_names: AbstractSet[str],
        kept_integer_names: AbstractSet[str],
    ) -> "NpModel":
        """Return a new NpModel restricted to the given kept sets.

        Rows not in kept_row_names are dropped entirely.
        Bounds not in their kept set are relaxed to ±inf.
        Variables not in kept_integer_names become continuous.
        """
        row_mask = _membership_mask(self.constraint_names, kept_row_names)

        lb_mask = _membership_mask(self.variable_names, kept_lb_names)
        ub_mask = _membership_mask(self.variable_names, kept_ub_names)
        int_mask = _membership_mask(self.variable_names, kept_integer_names)

        return NpModel(
            A=self.A[row_mask],
            senses=self.senses[row_mask],
            rhs=self.rhs[row_mask],
            lb=np.where(lb_mask, self.lb, -np.inf),
            ub=np.where(ub_mask, self.ub, np.inf),
            is_integer=self.is_integer & int_mask,
            variable_names=self.variable_names.copy(),
            constraint_names=self.constraint_names[row_mask],
        )


class BaseBackend(ABC):
    """A backend parses an instance into an :class:`NpModel` and solves subsystems."""

    # Short name, as :func:`get_backend` accepts it.
    name: ClassVar[str] = ""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self._np_model: Optional[NpModel] = None

    @abstractmethod
    def load_mps(self, filepath: str) -> None:
        pass

    @property
    def np_model(self) -> NpModel:
        """The loaded instance as numpy arrays.  Raises before :meth:`load_mps`."""
        if self._np_model is None:
            raise RuntimeError("No instance loaded; call load_mps() first")
        return self._np_model

    def solve_subsystem(
        self,
        kept_rows: AbstractSet[str],
        kept_lower_bounds: AbstractSet[str],
        kept_upper_bounds: AbstractSet[str],
        kept_integers: AbstractSet[str],
        time_limit: Optional[float] = None,
        tolerance: float = DEFAULT_TOLERANCE,
    ) -> tuple[str, float]:
        """Return ``(status, solver_seconds)``.

        :data:`STATUS_FEASIBLE` needs a witness: the solver's point must satisfy the
        subsystem to within *tolerance*, absolute, the rule certificates are held to.
        The solver is run at *tolerance* too, but its tolerances are relative or
        scaled, so a point that misses earns one more solve at
        :data:`RETRY_SOLVER_TOLERANCE`; a point that misses again is
        :data:`STATUS_UNKNOWN`.  So is a ``time_limit`` that expires without a
        verdict — inconclusive, not a failed submission.
        """
        sub = self.np_model.sub_system(
            kept_rows, kept_lower_bounds, kept_upper_bounds, kept_integers
        )
        names = sub.variable_names.tolist()
        seconds = 0.0
        for solver_tolerance in sorted(
            {max(tolerance, RETRY_SOLVER_TOLERANCE), RETRY_SOLVER_TOLERANCE}, reverse=True
        ):
            remaining = None if time_limit is None else time_limit - seconds
            if remaining is not None and remaining <= 0:
                break
            status, point, solve_seconds = self._solve(sub, solver_tolerance, remaining)
            seconds += solve_seconds
            if status != STATUS_FEASIBLE:
                return status, seconds
            if point is not None and sub.is_feasible(dict(zip(names, point)), tolerance)[0]:
                return STATUS_FEASIBLE, seconds
        return STATUS_UNKNOWN, seconds

    @abstractmethod
    def _solve(
        self, sub: NpModel, solver_tolerance: float, time_limit: Optional[float]
    ) -> tuple[str, Optional[list[float]], float]:
        """Solve *sub* once; return ``(status, point, solver_seconds)``.

        *status* is :data:`STATUS_INFEASIBLE`, :data:`STATUS_FEASIBLE` — with *point*
        the solver's solution, one value per variable of *sub* — or
        :data:`STATUS_UNKNOWN`.  *solver_tolerance* is the primal feasibility and
        integrality tolerance to solve at, absolute; a solver whose tolerance is
        relative converts it.
        """
