# Changelog

Changes to the checker and to the rules it enforces, by release. Pin a release in your install (see
[`README.md`](README.md#check-your-solutions)) to keep its behaviour fixed.

## Unreleased

- Benchmark harness: a cached program run is reused only for the same instance bytes (it used to
  match on the model name alone) and, with `--container`, the same image: the tag is resolved to the
  image ID at the start, the container runs by that ID, and the record stores it. A tag that
  podman does not know now stops the run before the first model.
- Benchmark harness: running the program no longer deletes a solution file the harness did not
  write (for example the reference solutions, when `--solution-dir` points at them). Such a file
  stops the run before the first model; `--overwrite-solutions` allows the replacement.

## 1.0.0b3

- The original counts (`original_bounds`, `total_original`) include the bounds of integer
  variables. They become elements of a subsystem that relaxes their integrality, so a kept count
  could exceed the original one (kept shares above 100% in the benchmark report).
- A subsystem is judged feasible only on a point satisfying it to within the certificate tolerance
  (1e-5, absolute). Previously the verdict was the solver's own, at its default tolerances, so it
  could depend on the backend.
- Instances that the MPS readers of the three backends read differently are refused by all of
  them: a nonzero value of magnitude at most 1e-9 in COLUMNS, RHS or BOUNDS (dropped or rounded to 0
  by some readers), a negative `UP`/`UI` bound on a column given no lower bound (the lower bound is
  -inf for Gurobi, 0 for SCIP and HiGHS), and RHS or BOUNDS lines without a set name.
- Solution files are read strictly: unknown keys inside an entry, a key given twice in one object,
  certificate values that are not JSON numbers, a certificate on a `"relax"` entry, and `rows` or
  `columns` that are not objects are all rejected. Previously some were ignored or converted.
- A certificate is checked exactly where rounding could decide it: huge values that cancel (`1e300`
  and `-1e300`) or overflow no longer hide a violated row.
- `--tolerance` (and `verify_subsystem(tolerance=...)`) is capped at 1e-2: it now also sets the
  solvers' tolerances, and Gurobi accepts none looser.
- `PROGRAM_INTERFACE.md` states the evaluation limits (3600 seconds per instance, at least 8
  threads, at least 32 GB of memory) and that commercial solvers cannot be used as subroutines of a
  submission.
- A program must not exit with status 125, which podman uses for its own errors
  ([`PROGRAM_INTERFACE.md`](PROGRAM_INTERFACE.md#result)).

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
