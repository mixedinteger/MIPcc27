# Benchmark harness (organizers)

Runs a competition: for each model it launches the competitor's program, times it, checks what it
wrote with `mipcc_iis_checker`, and caches the result. `benchmark.collect` then reports from that
cache.

## Running a competition

One cluster job per model, all writing to one `--out-dir` per competitor:

```bash
# e.g. a SLURM array task; --instances takes the model name or its file
model=$(ls public_iis/*.mps.gz | sed -n "${SLURM_ARRAY_TASK_ID}p")
uv run python -m benchmark.run --instance-dir public_iis --solution-dir sols/teamA \
    --out-dir out/teamA --container teamA:latest --container-args "--cpus 8 --memory 32g" \
    --solve-time-limit 3600 --threads 8 --solver gurobi --verify-time-limit 600 \
    --instances "$model"
```

Then collect what the jobs left, anywhere — no solver, podman or license needed:

```bash
uv run python -m benchmark.collect --out-dir out/teamA --instance-dir public_iis
```

- `--container IMAGE` runs the image under rootless podman with the sandbox described in
  `PROGRAM_INTERFACE.md`. `--container-args` adds `podman run` options: set CPU and memory limits
  here, and mount a solver license if the program needs one. On an SELinux host add
  `--security-opt label=disable` if the mounts are refused.
- `--podman CMD` is the podman command, with any global options a cluster needs (e.g.
  `"podman --root DIR --runroot DIR"` where Slurm leaves no `/run/user/$UID`). The harness uses it
  both to run the container and to kill it at the limit, so set it here, not in `--container-args`.
- `--container-network MODE` replaces the sandbox's `none`, e.g. `host` for a program that needs a
  Gurobi WLS license. Only for trusted programs, such as the organizers' own: competitors get no
  network.
- `--container-pids-limit N` (default 4096) caps the container's processes; `0` sets no limit.
  Rootless podman can only set limits — this one, or `--cpus`/`--memory` — in a cgroup delegated to
  the user, which a Slurm job usually lacks ("rootless needs no limits"); there use `0` and let
  Slurm's own limits bound the job.
- `--solve-cmd TEMPLATE` runs a command instead, with `{instance}`, `{solution}` and `{name}` filled
  in, no shell and **no sandbox** — for trusted programs, local tests, or `apptainer`.
- The program writes into an empty scratch directory; only a regular file left there after exit 0 is
  copied into `--solution-dir`. Any earlier solution for the model is deleted first.
- Without `--container`/`--solve-cmd`, the harness checks the solutions already in `--solution-dir`.

The exit code is `0` when every model reached a verdict — passed or failed — and `1` when the
harness could not decide one (a status below other than the four verdicts), so a cluster job only
fails for the judge's problems.

## Collecting results

`benchmark.collect` only reads `<out-dir>/results/` and writes `summary.json` and `summary.csv`
next to it. It runs neither the program nor the checker, and never touches a result, log or
solution, so it is the safe way to gather a cluster run's results. Use it rather than re-running
`benchmark.run` on a copy of the solutions: that re-checks them and replaces each result record
with one about the copy.

- With `--out-dir` alone it reports every model that has a result; the solutions need not be at
  hand.
- `--instance-dir` also reports every model there that was never run, as `not_run`.
- `--solution-dir` and `--instance-dir` each mark a result `stale` when that directory no longer
  holds the file that was checked.
- `--instances` restricts the report to the models named.
- `--only-started` leaves out the models whose job never started (neither a result nor a log in
  `--out-dir`), e.g. those excluded from this run. A job killed after starting the program still
  shows, as `not_run`.

It exits `1` when a model has no verdict, as `benchmark.run` does.

## Statuses

| status               | meaning                                                                         |
| -------------------- | ------------------------------------------------------------------------------- |
| `ok`                 | checked: the solver decided, and `pass` says whether the submission is valid    |
| `invalid_solution`   | the solution can't be parsed, or names elements the instance lacks (`pass = N`) |
| `solve_timeout`      | the program ran past `--solve-time-limit` (`pass = N`)                          |
| `solve_failed`       | the program exited non-zero or wrote no solution (`pass = N`); see `logs/`      |
| `verify_unknown`     | the infeasibility check hit `--verify-time-limit`                               |
| `verify_error`       | the checker couldn't run: no solver, unreadable instance, solver error          |
| `solve_error`        | the program couldn't be started (bad `--solve-cmd`, no podman, podman exit 125) |
| `no_solution`        | no `<model>.json[.gz]` to check                                                 |
| `instance_not_found` | no `<model>.mps[.gz]` in `--instance-dir`                                       |
| `not_run`            | collect only: no result yet — the job never ran or hasn't finished              |
| `stale`              | collect only: the files on disk or the checker are not the ones that checked it |

The first four are verdicts; the rest are the harness's problems.

## Caching and re-runs

Re-running a job only redoes what changed:

- The program is not relaunched while the previous run used the same program, `--solve-time-limit`
  and `--threads`, and its solution is unchanged on disk. A failed run is reused too.
- The check is not repeated while the cached result has the same SHA-256 for both the solution and
  the instance, the same solver, and the same checker version and certificate tolerance (so a
  checker upgrade re-checks everything). Only `ok`, `invalid_solution` and `verify_unknown` are
  reused; `verify_unknown` is re-checked when `--verify-time-limit` grows. Everything else is
  retried.
- `--force` redoes both. A new competitor gets a fresh `--out-dir`.

## Output

- `<out-dir>/results/<model>.json` — the result record (also the cache), including the program's
  run: command, wall time, exit code, host, limits, solution hash.
- `<out-dir>/logs/<model>.log` — the program's stdout and stderr.
- `<out-dir>/summary.json` and `summary.csv` — written by `benchmark.collect` and by runs without
  `--instances`, so per-model jobs never overwrite them.

The printed table:

```
instance                       pass  rows%  bnds%  ints%  cert%   iis(s) check(s) status
----------------------------------------------------------------------------------------
stein10i                          Y   53.3  100.0  100.0   22.9     0.53     0.00 ok
```

- `rows%`/`bnds%`/`ints%` — kept elements as a share of all the model's elements, per kind (`<0.1`
  if tiny). The bounds of integer variables are in the original count, as they become elements of a
  subsystem that relaxes their integrality; `bnds%` is `-` only for a model with no bounds at all.
- `cert%` — share of kept elements with a certificate that verified; `100.0` is a proven IIS.
- `iis(s)` — the program's wall clock, measured by the harness. `check(s)` — the infeasibility
  check.
- `n/a` replaces the shares when the subsystem was proven feasible or the solution was invalid.

`summary.csv` has one row per model, for `pandas.read_csv`: the raw counts (`orig_*`, `kept_*`,
`cert_*`), unrounded shares, `passed`/`infeasible`/`minimal`, solver, status, `iis_time`,
`solve_time`, `solve_exit_code`. A measurement not taken is an empty cell.

To run the harness itself in a container, see [`DOCKER.md`](DOCKER.md).
