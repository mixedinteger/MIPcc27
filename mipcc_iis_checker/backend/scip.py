"""
PySCIPOpt-based solver backend.

pyscipopt reads the MPS file into an NpModel; each subsystem solve slices those
arrays and builds a fresh SCIP model from them.
"""

import time
from typing import Optional

import numpy as np
import scipy.sparse as sp

# isort: split
from pyscipopt import Model

from .base import (
    STATUS_FEASIBLE,
    STATUS_INFEASIBLE,
    STATUS_UNKNOWN,
    BaseBackend,
    NpModel,
    UnsupportedInstanceError,
    check_mps,
    validate_instance,
)

# Types that map onto NpModel's is_integer; anything else (notably "IMPLINT") is refused.
_SCIP_INTEGER_VTYPES = frozenset(("INTEGER", "BINARY"))
_SCIP_SUPPORTED_VTYPES = _SCIP_INTEGER_VTYPES | frozenset(("CONTINUOUS",))

# SCIP's defaults for its zero thresholds, and the floor for the scaled feastol (SCIP
# accepts down to 1e-17, and needs room below it for epsilon).
_SCIP_DEFAULT_EPSILON = 1e-9
_SCIP_DEFAULT_SUMEPSILON = 1e-6
_SCIP_MIN_FEASTOL = 1e-15


class ScipBackend(BaseBackend):
    name = "scip"

    def load_mps(self, filepath: str) -> None:
        check_mps(filepath)
        m = Model("reader")
        m.hideOutput(not self.verbose)
        m.readProblem(filepath)

        variables = m.getVars()
        constraints = m.getConss()

        var_names = np.array([v.name for v in variables], dtype=object)
        n = len(var_names)

        scip_inf = m.infinity()

        lb = np.array([v.getLbOriginal() for v in variables], dtype=np.float64)
        ub = np.array([v.getUbOriginal() for v in variables], dtype=np.float64)

        lb = np.where(lb <= -scip_inf, -np.inf, lb)
        ub = np.where(ub >= scip_inf, np.inf, ub)

        # Everything NpModel cannot hold, collected and refused in one go
        # (see UnsupportedInstanceError).
        unsupported: dict[str, int] = {}

        vtypes = [v.vtype() for v in variables]
        is_integer = np.array([t in _SCIP_INTEGER_VTYPES for t in vtypes], dtype=bool)
        for t in vtypes:
            if t not in _SCIP_SUPPORTED_VTYPES:
                key = f"{t!r} variables"
                unsupported[key] = unsupported.get(key, 0) + 1

        # Only the "linear" handler can be read back as a row of A; other handlers —
        # including some purely linear forms — need their own getter, so refuse instead.
        linear_constrs = []
        for c in constraints:
            ctype = c.getConshdlrName()
            if ctype == "linear":
                linear_constrs.append(c)
            else:
                key = f"{ctype!r} constraints"
                unsupported[key] = unsupported.get(key, 0) + 1

        if unsupported:
            m.freeProb()
            raise UnsupportedInstanceError(filepath, unsupported)

        num_constrs = len(linear_constrs)
        constr_names = np.array([c.name for c in linear_constrs], dtype=object)

        rows_data = []
        rows_indices = []
        rows_indptr = [0]
        senses = np.empty(num_constrs, dtype=object)
        rhs = np.empty(num_constrs, dtype=np.float64)
        ranged_rows: list[str] = []

        var_index = {name: i for i, name in enumerate(var_names)}

        for i, c in enumerate(linear_constrs):
            vals = m.getValsLinear(c)
            lhs_val = m.getLhs(c)
            rhs_val = m.getRhs(c)

            for vname, coeff in vals.items():
                col = var_index.get(vname)
                if col is None:
                    # Dropping the term would silently weaken this row.
                    m.freeProb()
                    raise ValueError(
                        f"Constraint '{c.name}' of '{filepath}' references variable"
                        f" '{vname}', which is not among the instance's variables."
                    )
                rows_data.append(coeff)
                rows_indices.append(col)

            rows_indptr.append(len(rows_data))

            lhs_finite = -scip_inf < lhs_val < scip_inf
            rhs_finite = -scip_inf < rhs_val < scip_inf

            if lhs_val >= scip_inf or rhs_val <= -scip_inf:
                # ``x >= +inf`` / ``x <= -inf``: unsatisfiable as written.  Keeping the
                # infinity hands it to validate_instance, which refuses it for both
                # backends rather than letting this one read it as a finite 1e20.
                senses[i] = ">" if lhs_val >= scip_inf else "<"
                rhs[i] = np.inf if lhs_val >= scip_inf else -np.inf
            elif lhs_finite and rhs_finite:
                # An E row parses both sides from one field, so they are bit-identical;
                # any difference means RANGES, which check_mps has already refused.
                if lhs_val == rhs_val:
                    senses[i] = "="
                    rhs[i] = rhs_val
                else:
                    # A ranged row is two restrictions and NpModel holds one
                    # (sense, rhs) per row, so refuse rather than keep one side.
                    ranged_rows.append(c.name)
            elif rhs_finite:
                senses[i] = "<"
                rhs[i] = rhs_val
            elif lhs_finite:
                senses[i] = ">"
                rhs[i] = lhs_val
            else:
                # Free constraint (no finite bound) — treat as <= +inf
                senses[i] = "<"
                rhs[i] = np.inf

        if ranged_rows:
            m.freeProb()
            raise UnsupportedInstanceError(filepath, {"ranged rows": len(ranged_rows)})

        A = sp.csr_matrix(
            (np.array(rows_data, dtype=np.float64), rows_indices, rows_indptr),
            shape=(num_constrs, n),
        )

        m.freeProb()
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

    def _solve(
        self, sub: NpModel, solver_tolerance: float, time_limit: Optional[float]
    ) -> tuple[str, Optional[list[float]], float]:
        m = Model("subsystem")
        m.hideOutput(not self.verbose)

        # SCIP's one primal tolerance covers rows, bounds and integrality, relative to
        # the magnitude of the values compared.  Divided by the largest of those, it is
        # at least as tight as the absolute tolerance asked for.  SCIP's zero
        # thresholds must come down with it, or presolve rounds at its own epsilon.
        values = np.concatenate((sub.rhs, sub.lb, sub.ub))
        values = np.abs(values[np.isfinite(values)])
        scale = max(1.0, float(values.max())) if values.size else 1.0
        feastol = max(solver_tolerance / scale, _SCIP_MIN_FEASTOL)
        m.setParam("numerics/feastol", feastol)
        m.setParam("numerics/epsilon", min(_SCIP_DEFAULT_EPSILON, feastol / 100))
        m.setParam("numerics/sumepsilon", min(_SCIP_DEFAULT_SUMEPSILON, feastol))
        if time_limit is not None:
            m.setParam("limits/time", time_limit)

        n_vars = len(sub.variable_names)
        scip_vars = []
        for i in range(n_vars):
            lb_i = sub.lb[i] if np.isfinite(sub.lb[i]) else None
            ub_i = sub.ub[i] if np.isfinite(sub.ub[i]) else None
            vtype = "I" if sub.is_integer[i] else "C"
            v = m.addVar(
                name=str(sub.variable_names[i]),
                lb=lb_i,
                ub=ub_i,
                vtype=vtype,
            )
            scip_vars.append(v)

        csr = sub.A.tocsr()
        for i in range(csr.shape[0]):
            row_start = csr.indptr[i]
            row_end = csr.indptr[i + 1]
            cols = csr.indices[row_start:row_end]
            vals = csr.data[row_start:row_end]

            sense = sub.senses[i]
            rhs_val = float(sub.rhs[i])

            if len(cols) == 0:
                # An empty row builds as the *number* 0, and addCons(0 <= rhs) hands
                # SCIP a plain bool it cannot take.
                if sense == "<":
                    satisfied = rhs_val >= 0
                elif sense == ">":
                    satisfied = rhs_val <= 0
                else:
                    satisfied = rhs_val == 0
                if not satisfied:
                    # Infeasible with nothing to assign: say so without a solve.
                    m.freeProb()
                    return STATUS_INFEASIBLE, None, 0.0
                continue

            lhs_expr = sum(float(vals[k]) * scip_vars[cols[k]] for k in range(len(cols)))
            if sense == "<":
                m.addCons(lhs_expr <= rhs_val, name=str(sub.constraint_names[i]))
            elif sense == ">":
                m.addCons(lhs_expr >= rhs_val, name=str(sub.constraint_names[i]))
            else:  # "="
                m.addCons(lhs_expr == rhs_val, name=str(sub.constraint_names[i]))

        start = time.perf_counter()
        m.optimize()
        runtime = time.perf_counter() - start

        scip_status = m.getStatus()
        point = None
        if scip_status != "infeasible" and m.getNSols() > 0:
            best = m.getBestSol()
            point = [m.getSolVal(best, v) for v in scip_vars]
        m.freeProb()

        if scip_status == "infeasible":
            return STATUS_INFEASIBLE, None, runtime
        if point is not None:
            return STATUS_FEASIBLE, point, runtime
        return STATUS_UNKNOWN, None, runtime
