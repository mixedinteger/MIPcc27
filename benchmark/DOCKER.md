# Running the harness in a container

The repository's `Dockerfile` and `docker-compose.yml` package the `benchmark` harness for
**rootless Podman**, so solutions can be checked and reported without installing anything on the
host. This image checks existing solutions; it does not launch competitors' containers
(`--container` needs `podman` on the host — run that mode with `uv run` instead).

- **Non-root:** the image runs as `appuser` (uid/gid 1000). `--userns=keep-id` maps it onto your own
  host uid, so the directories you mount are readable and writable without `chown`.
- **Read-only root:** run with `--read-only`; only a `tmpfs` at `/tmp` and the mounted output
  directory are writable.
- **No baked-in data:** instances and solutions are mounted read-only at run time.

```bash
podman build -t mipcc-iis-checker .

podman run --rm --userns=keep-id --read-only --tmpfs /tmp \
  -v "$(pwd)/public_iis:/data/instances:ro" \
  -v "$(pwd)/solutions:/data/solutions:ro" \
  -v "$(pwd)/benchmark_out:/data/benchmark_out" \
  mipcc-iis-checker --instance-dir /data/instances --solution-dir /data/solutions
```

Any other `benchmark.run` flag (`--solver`, `--verify-time-limit`, `--instances`,
...) can be appended; see [`README.md`](README.md). To collect results instead, override the
entrypoint: `--entrypoint python mipcc-iis-checker -m benchmark.collect --out-dir /data/benchmark_out`.
With compose:

```bash
podman compose --userns=keep-id run --rm checker \
    --instance-dir /data/instances --solution-dir /data/solutions
```

`docker-compose.yml` holds the default volume layout; adjust the host paths as needed.
