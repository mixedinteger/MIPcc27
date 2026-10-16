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
from functools import cached_property
from typing import ClassVar, Optional

import numpy as np
import scipy.sparse as sp

STATUS_INFEASIBLE = "INFEASIBLE"
STATUS_FEASIBLE = "FEASIBLE"
STATUS_UNKNOWN = "UNKNOWN"
STATUS_ERROR = "ERROR"


class UnsupportedInstanceError(ValueError):
    """The instance uses features :class:`NpModel` cannot represent.

    NpModel holds exactly one thing: a system of linear rows over bounded,
    optionally-integer variables, one ``(sense, rhs)`` each.  Quadratic or general
    (indicator, min/max, abs, piecewise-linear, nonlinear) constraints, SOS sets,
    semi-continuous variables and ranged rows have nowhere to go, so a backend
    refuses the instance rather than load part of it.

    The ``N`` objective row is dropped on purpose — the checker only asks feasibility
    questions — but an objective needing a section of its own (``QUADOBJ``, ``PWLOBJ``)
    is refused all the same, because ``check_sections`` does not tell an objective
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


def check_sections(filepath: str) -> None:
    """Refuse the instance if it uses an MPS section outside the supported set.

    Section headers start in column 1, data lines are indented, and ``*`` in column 1
    is a comment.
    """
    opener = gzip.open if str(filepath).endswith(".gz") else open
    found: list[str] = []
    with opener(filepath, "rt", errors="replace") as handle:
        for line in handle:
            if not line[:1].strip() or line.startswith("*"):
                continue
            section = line.split()[0].upper()
            if section not in _SUPPORTED_SECTIONS and section not in found:
                found.append(section)
    if found:
        raise UnsupportedInstanceError(filepath, {f"{name} section": 1 for name in found})


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
        lhs = self.A @ x
        slack = lhs - self.rhs
        bad_rows = np.flatnonzero((is_le & (slack > tolerance)) | (is_ge & (slack < -tolerance)))
        for k in bad_rows:
            if kind == "row" and k == skip:
                ignored_was_violated = True
                continue
            op = {"<": ">", ">": "<", "=": "!="}[senses[k]]
            name = self.constraint_names[k]
            violations.append(
                f"constraint {name!r}: lhs {lhs[k]:.15g} {op} rhs {self.rhs[k]:.15g}"
                f" (violation {abs(slack[k]):.3g})"
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

    @abstractmethod
    def solve_subsystem(
        self,
        kept_rows: AbstractSet[str],
        kept_lower_bounds: AbstractSet[str],
        kept_upper_bounds: AbstractSet[str],
        kept_integers: AbstractSet[str],
        time_limit: Optional[float] = None,
    ) -> tuple[str, float]:
        """Return ``(status, solver_seconds)``.

        A ``time_limit`` that expires without a verdict yields
        :data:`STATUS_UNKNOWN` — inconclusive, not a failed submission.
        """
        pass
