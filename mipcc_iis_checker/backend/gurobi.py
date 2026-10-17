"""
Gurobipy-based solver backend.

gurobipy.read() extracts the instance into an NpModel; each subsystem solve slices
those arrays and hands them to gp.loadModel in one call.
"""

from typing import Optional

import numpy as np
import scipy.sparse as sp

# isort: split
import gurobipy as gp

from .base import (
    STATUS_FEASIBLE,
    STATUS_INFEASIBLE,
    STATUS_UNKNOWN,
    BaseBackend,
    NpModel,
    UnsupportedInstanceError,
    _membership_mask,
    check_mps,
    validate_instance,
)

# Gurobi reads any bound past 1e30 as infinite, so integer variables stay unbounded.
_INT_BOUND_M = 1e30

# gurobipy VType character codes considered integer
_INTEGER_VTYPES = {gp.GRB.INTEGER, gp.GRB.BINARY}

# The only types NpModel's single is_integer flag can hold.
_SUPPORTED_VTYPES = _INTEGER_VTYPES | {gp.GRB.CONTINUOUS}
_VTYPE_LABELS = {
    gp.GRB.SEMICONT: "semi-continuous variables",
    gp.GRB.SEMIINT: "semi-integer variables",
}


class GurobipyBackend(BaseBackend):
    name = "gurobi"

    def load_mps(self, filepath: str) -> None:
        check_mps(filepath)
        with gp.Env(params={"OutputFlag": 0}) as env:
            with gp.read(filepath, env=env) as raw:
                var_names = np.array(raw.getAttr("VarName", raw.getVars()), dtype=object)
                constr_names = np.array(raw.getAttr("ConstrName", raw.getConstrs()), dtype=object)

                lb = np.array(raw.getAttr("LB", raw.getVars()), dtype=np.float64)
                ub = np.array(raw.getAttr("UB", raw.getVars()), dtype=np.float64)
                vtypes = np.array(raw.getAttr("VType", raw.getVars()), dtype=object)
                A = sp.csr_matrix(raw.getA())
                rhs = np.array(raw.getAttr("RHS", raw.getConstrs()), dtype=np.float64)
                senses = np.array(raw.getAttr("Sense", raw.getConstrs()), dtype=object)

        # Semi-continuous/semi-integer live in BOUNDS, which no section header reveals.
        unsupported = {}
        seen_vtypes, vtype_counts = np.unique(vtypes.astype("U1"), return_counts=True)
        for vtype, count in zip(seen_vtypes.tolist(), vtype_counts.tolist()):
            if vtype not in _SUPPORTED_VTYPES:
                unsupported[_VTYPE_LABELS.get(vtype, f"{vtype!r} variables")] = count

        if unsupported:
            raise UnsupportedInstanceError(filepath, unsupported)

        validate_instance(filepath, constr_names.tolist(), var_names.tolist(), A, rhs, senses)

        lb = np.where(lb <= -gp.GRB.INFINITY, -np.inf, lb)
        ub = np.where(ub >= gp.GRB.INFINITY, np.inf, ub)
        is_integer = _membership_mask(vtypes, _INTEGER_VTYPES)

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
        sub_lb = np.where(sub.is_integer & np.isinf(sub.lb), -_INT_BOUND_M, sub.lb)
        sub_ub = np.where(sub.is_integer & np.isinf(sub.ub), _INT_BOUND_M, sub.ub)
        vtype = np.where(sub.is_integer, gp.GRB.INTEGER, gp.GRB.CONTINUOUS)
        csc = sub.A.tocsc()

        flag = 1 if self.verbose else 0
        with gp.Env(params={"OutputFlag": flag, "LogToConsole": flag}) as env:
            with gp.loadModel(
                env=env,
                numvars=len(sub.variable_names),
                numconstrs=sub.A.shape[0],
                lb=sub_lb,
                ub=sub_ub,
                vtype=vtype,
                constr_csc=(csc.data, csc.indices, csc.indptr),
                sense=sub.senses,
                rhs=sub.rhs,
            ) as m:
                m.Params.FeasibilityTol = solver_tolerance
                m.Params.IntFeasTol = solver_tolerance
                if time_limit is not None:
                    m.Params.TimeLimit = time_limit
                m.optimize()
                runtime = m.Runtime

                if m.Status == gp.GRB.INF_OR_UNBD:
                    # The objective is zero, so "or unbounded" is moot; telling the
                    # two apart takes a solve without dual reductions.
                    m.Params.DualReductions = 0
                    if time_limit is not None:
                        m.Params.TimeLimit = max(time_limit - runtime, 0.0)
                    m.optimize()
                    runtime += m.Runtime

                if m.Status == gp.GRB.INFEASIBLE:
                    return STATUS_INFEASIBLE, None, runtime
                if m.SolCount > 0:
                    return STATUS_FEASIBLE, m.getAttr("X", m.getVars()), runtime
                return STATUS_UNKNOWN, None, runtime
