"""
Solver backend package for MIP Competition 2027 Solution Checker.

- ScipBackend (SCIP solver) in scip.py
- GurobipyBackend (Gurobi solver) in gurobi.py
- HighsBackend (HiGHS solver) in highs.py
- get_backend() factory function

All solver backends are imported defensively: the checker is usable with
whichever solver is installed, and get_backend() reports what is missing.

"""

from .base import (
    STATUS_ERROR,
    STATUS_FEASIBLE,
    STATUS_INFEASIBLE,
    STATUS_UNKNOWN,
    BaseBackend,
    ElementCounts,
    NpModel,
    UnsupportedInstanceError,
)

try:
    from .gurobi import GurobipyBackend

    HAS_GUROBI = True
except ImportError:
    HAS_GUROBI = False

try:
    from .scip import ScipBackend

    HAS_SCIP = True
except ImportError:
    HAS_SCIP = False

try:
    from .highs import HighsBackend

    HAS_HIGHS = True
except ImportError:
    HAS_HIGHS = False

__all__ = [
    "BaseBackend",
    "ElementCounts",
    "NpModel",
    "UnsupportedInstanceError",
    "get_backend",
    "STATUS_INFEASIBLE",
    "STATUS_FEASIBLE",
    "STATUS_UNKNOWN",
    "STATUS_ERROR",
]

if HAS_GUROBI:
    __all__.append("GurobipyBackend")
if HAS_SCIP:
    __all__.append("ScipBackend")
if HAS_HIGHS:
    __all__.append("HighsBackend")


def get_backend(solver: str = "", verbose: bool = False) -> BaseBackend:
    """Create a solver backend.

    Args:
        solver: One of "" (auto-select), "scip", "gurobi" or "highs".
        verbose: Whether to enable solver output.

    Auto-selection prefers Gurobi, then SCIP, then HiGHS.
    """
    if solver == "" and HAS_GUROBI:
        solver = "gurobi"
    if solver == "" and HAS_SCIP:
        solver = "scip"
    if solver == "" and HAS_HIGHS:
        solver = "highs"
    if solver == "":
        raise ValueError(
            "No solver backend available: install pyscipopt, gurobipy or highspy "
            "(e.g. `pip install mipcc-iis-checker[solvers]`)."
        )

    if solver == "gurobi":
        if HAS_GUROBI:
            return GurobipyBackend(verbose=verbose)
        raise ValueError("Gurobi backend requested but gurobipy is not installed.")
    if solver == "scip":
        if HAS_SCIP:
            return ScipBackend(verbose=verbose)
        raise ValueError("SCIP backend requested but pyscipopt is not installed.")
    if solver == "highs":
        if HAS_HIGHS:
            return HighsBackend(verbose=verbose)
        raise ValueError("HiGHS backend requested but highspy is not installed.")

    raise ValueError(f"Unknown solver backend: {solver!r}. Choose 'scip', 'gurobi' or 'highs'.")
