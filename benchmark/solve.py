"""
Running the competitor's program: the solve step of a benchmark run.

The program is launched once per model as ``<entrypoint> INSTANCE SOLUTION`` (the
interface is specified in ``PROGRAM_INTERFACE.md``), timed here, and killed at the
limit.  It writes into an empty scratch directory; only a regular file left there
after a clean exit is copied into the solution directory.
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import signal
import socket
import stat
import subprocess
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# Seconds between SIGTERM and SIGKILL once the time limit is reached.
KILL_GRACE_S = 10.0

# Solution file extensions, in the order tried.  The checker reads both.
SOLUTION_EXTS = (".json", ".json.gz")

# Solve outcomes: "solved" hands over to the checker; the two failures are verdicts
# against the submission; "solve_error" means the program could not be started.
SOLVED = "solved"
SOLVE_FAILURES = frozenset({"solve_timeout", "solve_failed"})
SOLVE_ERROR = "solve_error"

# The fields a --solve-cmd template may use.
TEMPLATE_FIELDS = ("instance", "solution", "name")


def file_sha256(path: Path) -> str:
    """Hex SHA-256 of *path*'s bytes."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass(frozen=True)
class Launcher:
    """A ``--solve-cmd`` template, or a ``--container`` image run under podman."""

    template: Optional[tuple[str, ...]] = None
    image: Optional[str] = None
    container_args: tuple[str, ...] = ()
    # The podman command, with any global options (e.g. ``--root``): it both runs and
    # kills the container, so both must see the same container store.
    podman: tuple[str, ...] = ("podman",)
    # The sandbox has no network; a trusted program may need one, e.g. for a WLS license.
    network: str = "none"
    # 0 sets no limit: rootless podman can apply none without a delegated cgroup,
    # e.g. under Slurm, whose own cgroup then bounds the job.
    pids_limit: int = 4096

    def __post_init__(self):
        if (self.template is None) == (self.image is None):
            raise ValueError("a Launcher needs exactly one of template or image")

    def describe(self) -> str:
        """A stable identity: a different program means a re-solve."""
        if self.template is not None:
            return "cmd: " + " ".join(self.template)
        network = "" if self.network == "none" else f" --network {self.network}"
        return f"podman: {self.image}{network} {' '.join(self.container_args)}".rstrip()

    def argv(self, name: str, instance: Path, solution: Path, container: str = "") -> list[str]:
        """The command writing *solution* from *instance*.

        A template is filled per token and run without a shell.  A container gets no
        network (unless *network* says otherwise), no capabilities, a read-only root,
        and only the instance (read-only) and the scratch directory mounted.
        """
        if self.template is not None:
            fields = {"instance": str(instance), "solution": str(solution), "name": name}
            return [token.format(**fields) for token in self.template]
        return [
            *self.podman, "run", "--rm", "--init", "--name", container,
            "--network", self.network, "--read-only", "--tmpfs", "/tmp",
            "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
            *(("--pids-limit", str(self.pids_limit)) if self.pids_limit else ()),
            "--userns=keep-id",
            "-e", "MIPCC_TIME_LIMIT", "-e", "MIPCC_THREADS",
            "-v", f"{instance.resolve()}:/data/{instance.name}:ro",
            "-v", f"{solution.parent.resolve()}:/out",
            *self.container_args,
            self.image, f"/data/{instance.name}", f"/out/{solution.name}",
        ]  # fmt: skip


@dataclass(frozen=True)
class SolveConfig:
    """Everything needed to launch the program for each model."""

    launcher: Launcher
    time_limit: float
    threads: Optional[int]
    log_dir: Path


def solve_is_reusable(prev: Optional[dict], config: SolveConfig, solution: Optional[Path]) -> bool:
    """Whether the previous run *prev* already answers this one.

    Same program and limits, and — if it produced a solution — that solution still
    on disk, byte for byte.  A failure is reused too (re-running a timeout costs the
    whole limit again); a launcher error never is.
    """
    if not prev or prev.get("status") not in SOLVE_FAILURES | {SOLVED}:
        return False
    if (prev.get("launcher"), prev.get("time_limit"), prev.get("threads")) != (
        config.launcher.describe(),
        config.time_limit,
        config.threads,
    ):
        return False
    if prev["status"] != SOLVED:
        return True
    return solution is not None and file_sha256(solution) == prev.get("solution_sha256")


def _stop(
    proc: subprocess.Popen, sig: int, container: Optional[str], podman: tuple[str, ...]
) -> None:
    """Send *sig* to the program's process group, and to its container if any."""
    try:
        os.killpg(proc.pid, sig)
    except ProcessLookupError:
        pass
    if container is not None:
        # Killing podman's CLI does not always stop the container behind it.
        try:
            subprocess.run(
                [*podman, "kill", "--signal", signal.Signals(sig).name[3:], container],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=KILL_GRACE_S,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            pass


def _take_output(scratch: Path, name: str, solution_dir: Path) -> Optional[Path]:
    """Copy the program's solution out of *scratch*; None if there is none.

    Only a regular file counts — a symlink could point at another team's solution —
    and copying rather than moving also breaks any hard link.
    """
    for ext in SOLUTION_EXTS:
        produced = scratch / f"{name}{ext}"
        try:
            if not stat.S_ISREG(produced.lstat().st_mode):
                continue
        except FileNotFoundError:
            continue
        final = solution_dir / produced.name
        tmp = solution_dir / f".{produced.name}.{uuid.uuid4().hex}.tmp"
        shutil.copyfile(produced, tmp, follow_symlinks=False)
        os.replace(tmp, final)
        return final
    return None


def run_solve(name: str, instance: Path, solution_dir: Path, config: SolveConfig) -> dict:
    """Run the program on *instance*, time it, and leave its solution in *solution_dir*.

    Any earlier solution for *name* is deleted first.  Returns the solve record; the
    program's combined output goes to ``<log_dir>/<name>.log``.
    """
    config.log_dir.mkdir(parents=True, exist_ok=True)
    solution_dir.mkdir(parents=True, exist_ok=True)
    for ext in SOLUTION_EXTS:
        (solution_dir / f"{name}{ext}").unlink(missing_ok=True)

    launcher = config.launcher
    container = None
    if launcher.image is not None:
        # Container names allow [a-zA-Z0-9_.-]; the random suffix avoids clashing
        # with a container a crashed run left behind.
        container = f"mipcc-{re.sub(r'[^a-zA-Z0-9_.-]', '_', name)}-{uuid.uuid4().hex[:8]}"
    env = dict(os.environ, MIPCC_TIME_LIMIT=format(config.time_limit, "g"))
    env.pop("MIPCC_THREADS", None)
    if config.threads is not None:
        env["MIPCC_THREADS"] = str(config.threads)

    scratch = Path(tempfile.mkdtemp(prefix=f".solving-{name}-", dir=solution_dir))
    argv = launcher.argv(name, instance, scratch / f"{name}.json", container or "")
    rec = {
        "status": None,
        "launcher": launcher.describe(),
        "argv": argv,
        "time_limit": config.time_limit,
        "threads": config.threads,
        "host": socket.gethostname(),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "wall_s": None,
        "exit_code": None,
        "error": None,
        "solution_sha256": None,
    }
    try:
        with open(config.log_dir / f"{name}.log", "wb") as log:
            t0 = time.monotonic()
            try:
                proc = subprocess.Popen(
                    argv,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL,
                    env=env,
                    start_new_session=True,
                )
            except OSError as e:
                rec.update(status=SOLVE_ERROR, error=f"could not start {argv[0]!r}: {e}")
                return rec
            timed_out = False
            try:
                proc.wait(timeout=config.time_limit)
            except subprocess.TimeoutExpired:
                timed_out = True
                _stop(proc, signal.SIGTERM, container, launcher.podman)
                try:
                    proc.wait(timeout=KILL_GRACE_S)
                except subprocess.TimeoutExpired:
                    _stop(proc, signal.SIGKILL, container, launcher.podman)
                    proc.wait()
            rec["wall_s"] = round(time.monotonic() - t0, 3)
            rec["exit_code"] = proc.returncode
            try:  # anything the program left running in its group goes too
                os.killpg(proc.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass

        if timed_out:
            rec.update(status="solve_timeout", error="killed at the time limit")
        elif proc.returncode != 0:
            rec.update(status="solve_failed", error=f"exit code {proc.returncode}")
        elif (final := _take_output(scratch, name, solution_dir)) is None:
            rec.update(status="solve_failed", error="no solution file written")
        else:
            rec.update(status=SOLVED, solution_sha256=file_sha256(final))
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    return rec
