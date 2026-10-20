# Program Interface

A submission is a program, preferably a container image.

## Invocation

```
<entrypoint> INSTANCE SOLUTION
```

- `INSTANCE` — the model, `.mps` or `.mps.gz`, read-only. In the container it is
  `/data/<model>.mps[.gz]`.
- `SOLUTION` — where to write your solution, in the [solution format](SOLUTION_FORMAT.md). In the
  container it is `/out/<model>.json`. Writing `SOLUTION.gz` (gzip-compressed) instead is also
  accepted. It must be a regular file — a symlink is rejected.

## Environment

Your program is told its budget through two environment variables. Read them instead of
hard-coding the numbers:

- `MIPCC_TIME_LIMIT` — the wall-clock seconds this run may take, 3600 in the evaluation. A run
  still going at the limit is killed and its output discarded, so write your solution before it
  (see Time limit below).
- `MIPCC_THREADS` — the number of threads to use, set to at least 8 in the evaluation. If unset
  (for example when you run the program yourself), use your own default.

## Time limit

- The limit is measured by the organizers, as wall-clock time from launch to exit, including the
  container's start-up. That measurement is the time reported for the run.
- At the limit your program receives SIGTERM, and SIGKILL at least 10 seconds later. Leave a margin:
  output from a run that was stopped is discarded.

## Result

- Exit with status `0` once `SOLUTION` is written. A non-zero exit, a timeout, or no solution file
  is a failed run, and nothing is checked.
- Don't exit with status `125`; podman reserves it for its own errors.
- Only a solution the [checker](README.md#check-your-solutions) passes counts. Use it to check your
  own output. The checker time is not included in your computing time.

## Allowed software

- Commercial solvers cannot be used as subroutines of the final submission.

## Sandbox

- No network.
- Read-only filesystem, except `/tmp` (in memory) and `/out`.
- No capabilities, and a non-root user id that is not chosen by you, so `HOME` may not be writable:
  point tools that write there at `/tmp` (e.g. `ENV HOME=/tmp` in your image).
- Each run has at least 32 GB of memory. A run that exceeds its memory may be killed, which makes
  it a failed run (see Result). CPU limits are set by the organizers.

## Test it locally

Run your image the way the organizers do:

```bash
podman run --rm --init --network none --read-only --tmpfs /tmp \
    --cap-drop ALL --security-opt no-new-privileges --userns=keep-id \
    -e MIPCC_TIME_LIMIT=60 -e MIPCC_THREADS=4 \
    -v "$PWD/stein15inf.mps.gz:/data/stein15inf.mps.gz:ro" -v "$PWD/out:/out" \
    your-image /data/stein15inf.mps.gz /out/stein15inf.json
mipcc-iis-checker --instance stein15inf.mps.gz --solution out/stein15inf.json
```
