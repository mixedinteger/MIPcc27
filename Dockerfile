# syntax=docker/dockerfile:1
FROM docker.io/library/python:3.13-slim

# uv drives dependency installation from the project's own pyproject.toml/uv.lock,
# so the image resolves the exact versions the repo is pinned to.
COPY --from=ghcr.io/astral-sh/uv:0.13.1 /uv /uvx /usr/local/bin/

# Fixed, non-root UID:GID. Rootless Podman remaps this into the invoking
# host user's subuid/subgid range (or 1:1 with --userns=keep-id), so the
# app never runs as root either inside the container or, from the host's
# point of view, outside it.
RUN groupadd --gid 1000 appuser \
    && useradd --uid 1000 --gid appuser --create-home --shell /usr/sbin/nologin appuser

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    PATH="/app/.venv/bin:$PATH"

# Install dependencies first, separately from the source, so an edit to
# mipcc_iis_checker/ or benchmark/ doesn't invalidate the (slow) solver install layer.
COPY pyproject.toml uv.lock ./
RUN uv sync --extra scip --no-install-project --no-dev

# Now bring in the actual source and finish the sync (installs the project itself).
COPY mipcc_iis_checker ./mipcc_iis_checker
COPY benchmark ./benchmark
RUN uv sync --extra scip --no-dev

# --instance-dir and --solution-dir are bind-mounted at run time; nothing under
# them is baked into the image. --out-dir defaults to benchmark_out/ under
# $IIS_BENCH_DIR (see benchmark/README.md) unless overridden. Point it at a
# path the non-root user can write, for use with a mounted volume.
ENV IIS_BENCH_DIR=/data/benchmark_out
RUN mkdir -p /data/benchmark_out && chown -R appuser:appuser /app /data

USER appuser:appuser

ENTRYPOINT ["python", "-m", "benchmark.run"]
CMD ["--help"]
