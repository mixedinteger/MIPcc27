# Changelog

Changes to the checker and to the rules it enforces, by release. Pin a release in your install (see
[`README.md`](README.md#check-your-solutions)) to keep its behaviour fixed.

## 1.0.0b1

First public release.

- `mipcc-iis-checker` checks a JSON solution ([`SOLUTION_FORMAT.md`](SOLUTION_FORMAT.md)) against an
  MPS instance: the kept subsystem must be infeasible, and every certificate provided must verify.
  Certificates are checked with an absolute tolerance of 1e-5. Solver backends: HiGHS, SCIP and
  Gurobi.
- The bounds of a variable that stays integer always belong to the subsystem: they apply in every
  check even if the solution relaxes them, are not counted in its size (nor in the model's), and
  need no certificate.
- How submissions are run: [`PROGRAM_INTERFACE.md`](PROGRAM_INTERFACE.md).
- Sample instances and solutions in [`smallset/`](smallset/).
