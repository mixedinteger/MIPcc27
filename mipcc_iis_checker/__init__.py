"""
MIP Competition 2027: Solution Checker Package
"""

from . import backend as _backend
from ._version import __version__
from .backend import (
    HAS_GUROBI,
    HAS_HIGHS,
    HAS_SCIP,
    STATUS_ERROR,
    STATUS_FEASIBLE,
    STATUS_INFEASIBLE,
    STATUS_UNKNOWN,
    ElementCounts,
    NpModel,
)
from .checker import (
    DEFAULT_TOLERANCE,
    VerificationResult,
    VerificationStats,
    verify_subsystem,
)
from .io import parse_solution_file
from .naming import (
    MPS_EXTENSIONS,
    find_model,
    model_name,
    model_names,
    mps_extension,
)

__all__ = [
    "__version__",
    "verify_subsystem",
    "VerificationResult",
    "VerificationStats",
    "DEFAULT_TOLERANCE",
    "parse_solution_file",
    "MPS_EXTENSIONS",
    "find_model",
    "model_name",
    "model_names",
    "mps_extension",
    "ElementCounts",
    "NpModel",
    "STATUS_INFEASIBLE",
    "STATUS_FEASIBLE",
    "STATUS_UNKNOWN",
    "STATUS_ERROR",
]

# Export only the backends whose solver is installed.
if HAS_GUROBI:
    GurobipyBackend = _backend.GurobipyBackend
    __all__.append("GurobipyBackend")
if HAS_SCIP:
    ScipBackend = _backend.ScipBackend
    __all__.append("ScipBackend")
if HAS_HIGHS:
    HighsBackend = _backend.HighsBackend
    __all__.append("HighsBackend")
