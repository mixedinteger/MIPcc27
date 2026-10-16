"""
HiGHS-based solver backend.

highspy's own ``readModel``/``getLp`` extracts the instance into an NpModel;
each subsystem solve slices those arrays and builds a fresh ``HighsLp`` from them.
"""

from collections.abc import Set as AbstractSet
from typing import Optional

import numpy as np
import scipy.sparse as sp

# isort: split
import highspy

from .base import (
    STATUS_FEASIBLE,
    STATUS_INFEASIBLE,
    STATUS_UNKNOWN,
    BaseBackend,
    NpModel,
    UnsupportedInstanceError,
    check_sections,
    validate_instance,
)

# Types that map onto NpModel's is_integer; anything else (SEMICONTINUOUS,
# SEMIINTEGER, IMPLICIT_INTEGER) is refused.
_HIGHS_INTEGER_VTYPES = frozenset((highspy.HighsVarType.kInteger,))
_HIGHS_SUPPORTED_VTYPES = _HIGHS_INTEGER_VTYPES | frozenset((highspy.HighsVarType.kContinuous,))
_HIGHS_VTYPE_LABELS = {
    highspy.HighsVarType.kSemiContinuous: "semi-continuous variables",
    highspy.HighsVarType.kSemiInteger: "semi-integer variables",
    highspy.HighsVarType.kImplicitInteger: "implicit-integer variables",
}


def _quiet(verbose: bool) -> highspy.Highs:
    h = highspy.Highs()
    h.setOptionValue("output_flag", verbose)
    return h


class HighsBackend(BaseBackend):
    name = "highs"

    def load_mps(self, filepath: str) -> None:
        check_sections(filepath)
        h = _quiet(self.verbose)
        # HiGHS's own MPS parser refuses, rather than reads, a matrix coefficient
        # past "large_matrix_value". Set to infinity.
        h.setOptionValue("large_matrix_value", float("inf"))
        status = h.readModel(str(filepath))
        if status == highspy.HighsStatus.kError:
            raise UnsupportedInstanceError(
                filepath, {"value(s) exceeding HiGHS's representable magnitude": 1}
            )

        lp = h.getLp()
        var_names = np.array(lp.col_names_, dtype=object)
        constr_names = np.array(lp.row_names_, dtype=object)

        # HiGHS silently drops *every* name (returning an empty list, not one of the
        # right length) when two rows or two columns share a name, rather than
        # raising -- so a short names array is the only signal of that failure here.
        if len(constr_names) != lp.num_row_ or len(var_names) != lp.num_col_:
            raise ValueError(
                f"Malformed MPS file '{filepath}': HiGHS could not assign names to all"
                " rows/columns (a row or column name is likely declared more than once)."
            )

        lb = np.asarray(lp.col_lower_, dtype=np.float64)
        ub = np.asarray(lp.col_upper_, dtype=np.float64)

        # integrality_ is empty when every column is continuous.
        n = lp.num_col_
        vtypes = list(lp.integrality_) or [highspy.HighsVarType.kContinuous] * n

        unsupported: dict[str, int] = {}
        is_integer = np.empty(n, dtype=bool)
        for i, vtype in enumerate(vtypes):
            is_integer[i] = vtype in _HIGHS_INTEGER_VTYPES
            if vtype not in _HIGHS_SUPPORTED_VTYPES:
                key = _HIGHS_VTYPE_LABELS.get(vtype, f"{vtype!r} variables")
                unsupported[key] = unsupported.get(key, 0) + 1

        if unsupported:
            raise UnsupportedInstanceError(filepath, unsupported)

        row_lower = np.asarray(lp.row_lower_, dtype=np.float64)
        row_upper = np.asarray(lp.row_upper_, dtype=np.float64)
        senses, rhs, ranged_rows = _senses_and_rhs(constr_names, row_lower, row_upper)
        if ranged_rows:
            raise UnsupportedInstanceError(filepath, {"ranged rows": len(ranged_rows)})

        mat = lp.a_matrix_
        A = sp.csc_matrix(
            (np.asarray(mat.value_, dtype=np.float64), mat.index_, mat.start_),
            shape=(lp.num_row_, lp.num_col_),
        ).tocsr()

        validate_instance(filepath, constr_names.tolist(), var_names.tolist(), A, rhs, senses)

        self._np_model = NpModel(
            A=A,
            senses=senses,
            rhs=rhs,
            lb=lb,
            ub=ub,
            is_integer=is_integer,
            variable_names=var_names,
            constraint_names=constr_names,
        )

    def solve_subsystem(
        self,
        kept_rows: AbstractSet[str],
        kept_lower_bounds: AbstractSet[str],
        kept_upper_bounds: AbstractSet[str],
        kept_integers: AbstractSet[str],
        time_limit: Optional[float] = None,
    ) -> tuple[str, float]:
        sub = self.np_model.sub_system(
            kept_rows, kept_lower_bounds, kept_upper_bounds, kept_integers
        )

        n_vars = len(sub.variable_names)
        n_rows = len(sub.constraint_names)

        lp = highspy.HighsLp()
        lp.num_col_ = n_vars
        lp.num_row_ = n_rows
        # col_cost_ must be set even when every weight is zero: HiGHS segfaults on
        # a model whose cost vector is left empty but columns are nonzero.
        lp.col_cost_ = np.zeros(n_vars, dtype=np.float64)
        lp.col_lower_ = sub.lb
        lp.col_upper_ = sub.ub
        lp.integrality_ = [
            highspy.HighsVarType.kInteger if flag else highspy.HighsVarType.kContinuous
            for flag in sub.is_integer
        ]

        row_lower, row_upper = _row_bounds(sub.senses, sub.rhs)
        lp.row_lower_ = row_lower
        lp.row_upper_ = row_upper

        csc = sub.A.tocsc()
        mat = highspy.HighsSparseMatrix()
        mat.format_ = highspy.MatrixFormat.kColwise
        mat.start_ = csc.indptr.tolist()
        mat.index_ = csc.indices.tolist()
        mat.value_ = csc.data.tolist()
        lp.a_matrix_ = mat

        h = _quiet(self.verbose)
        if time_limit is not None:
            h.setOptionValue("time_limit", time_limit)
        h.passModel(lp)
        h.run()
        model_status = h.getModelStatus()
        runtime = h.getRunTime()

        if model_status == highspy.HighsModelStatus.kInfeasible:
            return STATUS_INFEASIBLE, runtime
        if model_status in (
            highspy.HighsModelStatus.kOptimal,
            highspy.HighsModelStatus.kUnbounded,
            highspy.HighsModelStatus.kObjectiveBound,
            highspy.HighsModelStatus.kObjectiveTarget,
            highspy.HighsModelStatus.kSolutionLimit,
        ) or (
            model_status
            in (highspy.HighsModelStatus.kTimeLimit, highspy.HighsModelStatus.kIterationLimit)
            and h.getInfo().primal_solution_status == highspy.SolutionStatus.kSolutionStatusFeasible
        ):
            return STATUS_FEASIBLE, runtime
        return STATUS_UNKNOWN, runtime


def _senses_and_rhs(
    names: np.ndarray, row_lower: np.ndarray, row_upper: np.ndarray
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Convert HiGHS's (row_lower, row_upper) range form to NpModel's (sense, rhs).

    A row with both bounds finite and unequal is a ranged row, which NpModel
    cannot hold (one (sense, rhs) per row) -- collected and refused by the caller.
    """
    n = len(row_lower)
    senses = np.empty(n, dtype=object)
    rhs = np.empty(n, dtype=np.float64)
    ranged_rows: list[str] = []

    lo_finite = np.isfinite(row_lower)
    hi_finite = np.isfinite(row_upper)

    for i in range(n):
        if lo_finite[i] and hi_finite[i]:
            if row_lower[i] == row_upper[i]:
                senses[i] = "="
                rhs[i] = row_upper[i]
            else:
                ranged_rows.append(str(names[i]))
                senses[i] = "<"
                rhs[i] = row_upper[i]
        elif hi_finite[i]:
            senses[i] = "<"
            rhs[i] = row_upper[i]
        elif lo_finite[i]:
            senses[i] = ">"
            rhs[i] = row_lower[i]
        else:
            # Free row (no finite bound) -- treat as <= +inf.
            senses[i] = "<"
            rhs[i] = np.inf

    return senses, rhs, ranged_rows


def _row_bounds(senses: np.ndarray, rhs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The inverse of :func:`_senses_and_rhs`, for building a subsystem's HighsLp."""
    senses_u1 = np.asarray(senses, dtype="U1")
    row_lower = np.where(senses_u1 == ">", rhs, -np.inf)
    row_upper = np.where(senses_u1 == "<", rhs, np.inf)
    is_eq = senses_u1 == "="
    row_lower = np.where(is_eq, rhs, row_lower)
    row_upper = np.where(is_eq, rhs, row_upper)
    return row_lower, row_upper
