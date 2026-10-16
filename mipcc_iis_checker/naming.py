"""
Model file naming — the corpus is MPS, and only MPS.

A model file is ``<model>.mps``, optionally gzip-compressed.  The *name* of a model is
its filename with that extension removed, and every file derived from a model — an IIS,
a solution JSON, a result record — is named by *replacing* the extension rather than
appending to it (``stein9in.mps.gz`` -> ``stein9in.json``), so one model name identifies
a model everywhere and no name accumulates the history of the files it came from.

Only MPS is accepted, and the check lives here rather than in each reader: Gurobi's
reader dispatches on the extension and would take an LP or OPB file where SCIP's
reads it differently or not at all, so the installed backend would decide what the
instance means.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

# In the order tried when looking up a model name's file.
MPS_EXTENSIONS = (".mps", ".mps.gz")


def mps_extension(path: Union[str, Path]) -> Optional[str]:
    """The MPS extension *path* ends with, or None if it is not an MPS file."""
    name = Path(path).name
    for ext in MPS_EXTENSIONS:
        # A file called exactly ".mps" is an extension with no name, not a model.
        if name.endswith(ext) and len(name) > len(ext):
            return ext
    return None


def model_name(path: Union[str, Path]) -> str:
    """The model name of *path*: its filename minus the MPS extension.

    ``public_iis/normalized-dbsg_50_10_5_15.lin.mps.gz`` ->
    ``normalized-dbsg_50_10_5_15.lin`` — only the MPS extension is removed, so a
    dot in the model's own name survives.

    Raises ``ValueError`` if *path* is not an MPS file.
    """
    ext = mps_extension(path)
    if ext is None:
        raise ValueError(
            f"not an MPS model file (expected one of {', '.join(MPS_EXTENSIONS)}): {path}"
        )
    return Path(path).name[: -len(ext)]


def find_model(name: str, directory: Union[str, Path]) -> Optional[Path]:
    """``<name>.mps`` or ``<name>.mps.gz`` in *directory*, or None.

    The inverse of :func:`model_name`.
    """
    directory = Path(directory)
    for ext in MPS_EXTENSIONS:
        candidate = directory / f"{name}{ext}"
        if candidate.exists():
            return candidate
    return None


def model_names(directory: Union[str, Path]) -> list[str]:
    """Every model in *directory*, by name, sorted.  Non-MPS files are ignored."""
    names = set()
    for path in Path(directory).iterdir():
        if mps_extension(path) is not None:
            names.add(model_name(path))
    return sorted(names)
