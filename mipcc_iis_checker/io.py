"""
Solution file parser for MIP Competition 2027 — JSON format.

Each element (constraint row, variable bound, integrality constraint) is either
"relaxed" (removed from the subsystem) or "kept" (retained).  A kept element
may carry an optional certificate: a variable assignment that proves the
subsystem becomes feasible when that single element is also removed.

File format: see SOLUTION_FORMAT.md
Solution files may be plain JSON (.json) or gzip-compressed (.json.gz).
"""

import gzip
import json
import math
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

# Accepted keys
_TOP_LEVEL_KEYS = frozenset({"rows", "columns"})
_COLUMN_SIDES = frozenset({"lower_bound", "upper_bound", "integrality"})
_ENTRY_KEYS = frozenset({"action", "certificate"})


@dataclass
class ElementEntry:
    """Disposition of one problem element together with an optional feasibility certificate."""

    action: str  # "relax" | "keep"
    # Variable assignment proving feasibility after this element is removed.
    # Only meaningful (and only verified) when action == "keep".
    certificate: Optional[dict[str, float]] = None


@dataclass
class SolutionData:
    """
    Parsed solution: per-element disposition with optional certificates.

    Keys are element identifiers present in this solution file.
    Elements absent from a dict default to ``{"action": "keep"}`` with no
    certificate.
    """

    rows: dict[str, ElementEntry] = field(default_factory=dict)
    lower_bounds: dict[str, ElementEntry] = field(default_factory=dict)
    upper_bounds: dict[str, ElementEntry] = field(default_factory=dict)
    integers: dict[str, ElementEntry] = field(default_factory=dict)

    @property
    def removed_rows(self) -> frozenset[str]:
        return frozenset(k for k, v in self.rows.items() if v.action == "relax")

    @property
    def removed_lower_bounds(self) -> frozenset[str]:
        return frozenset(k for k, v in self.lower_bounds.items() if v.action == "relax")

    @property
    def removed_upper_bounds(self) -> frozenset[str]:
        return frozenset(k for k, v in self.upper_bounds.items() if v.action == "relax")

    @property
    def removed_integers(self) -> frozenset[str]:
        return frozenset(k for k, v in self.integers.items() if v.action == "relax")


def _parse_entry(obj: object, label: str) -> ElementEntry:
    """Parse a single ``{"action": ..., "certificate": {...}}`` dict."""
    if not isinstance(obj, dict):
        raise ValueError(f"Entry for '{label}' must be a JSON object, got {type(obj).__name__!r}")
    # A misspelt key would otherwise be dropped: "actoin": "relax" would read as keep.
    unknown = sorted(set(obj) - _ENTRY_KEYS)
    if unknown:
        raise ValueError(
            f"Entry '{label}': unknown key(s) {unknown}; expected any of {sorted(_ENTRY_KEYS)}"
        )
    action = obj.get("action", "keep")
    if action not in ("relax", "keep"):
        raise ValueError(f"Entry '{label}': action must be 'relax' or 'keep', got {action!r}")
    raw_cert = obj.get("certificate")
    cert: Optional[dict[str, float]] = None
    if "certificate" in obj:
        if action == "relax":
            raise ValueError(
                f"Entry '{label}': a certificate is only for a kept element, not a relaxed one"
            )
        if not isinstance(raw_cert, dict):
            raise ValueError(
                f"Entry '{label}': certificate must be a JSON object mapping variable names to"
                f" numbers, got {type(raw_cert).__name__!r}"
            )
        for var, value in raw_cert.items():
            # JSON numbers only: float() would also take "1" and true.
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(
                    f"Entry '{label}': certificate value for '{var}' must be a number,"
                    f" got {value!r}"
                )
        cert = {k: float(v) for k, v in raw_cert.items()}
        # Keep NaN out of the arithmetic: it compares false against every bound.
        for var, value in cert.items():
            if not math.isfinite(value):
                raise ValueError(
                    f"Entry '{label}': certificate value for '{var}' must be finite, got {value}"
                )
    return ElementEntry(action=action, certificate=cert)


class _DuplicateKeyError(ValueError):
    pass


def _reject_duplicates(pairs: list[tuple[str, object]]) -> dict:
    """A JSON object, refusing a key given twice (json.loads would keep the last)."""
    obj = dict(pairs)
    if len(obj) != len(pairs):
        twice = sorted(k for k, count in Counter(k for k, _ in pairs).items() if count > 1)
        raise _DuplicateKeyError(f"key(s) given more than once in one JSON object: {twice}")
    return obj


def parse_solution_file(filepath: Union[str, Path]) -> SolutionData:
    """Parse a JSON solution file and return a :class:`SolutionData`.

    Expected top-level schema::

        {
          "rows": {
            "<constraint_name>": {
              "action": "relax" | "keep",
              "certificate": {"<var>": <value>, ...}   // optional, only for "keep"
            },
            ...
          },
          "columns": {
            "<var_name>": {
              "lower_bound": { "action": "relax" | "keep", "certificate": {...} },
              "upper_bound": { "action": "relax" | "keep", "certificate": {...} },
              "integrality": { "action": "relax" | "keep", "certificate": {...} }
            },
            ...
          }
        }

    Both top-level keys are optional, and so is every key nested under a
    column.  Elements not listed default to ``{"action": "keep"}`` with no
    certificate.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Solution file not found: {path}")

    def _reject_constant(token: str) -> float:
        # Python's json accepts bare NaN/Infinity; JSON proper does not, and a
        # non-finite certificate value is never a valid feasibility proof.
        raise ValueError(
            f"Invalid JSON in solution file '{path}': bare {token} is not a valid JSON number"
        )

    try:
        if str(path).endswith(".gz"):
            with gzip.open(path, "rt", encoding="utf-8") as f:
                text_data = f.read()
        else:
            text_data = path.read_text(encoding="utf-8")
        data = json.loads(
            text_data, parse_constant=_reject_constant, object_pairs_hook=_reject_duplicates
        )
    except (json.JSONDecodeError, _DuplicateKeyError) as e:
        raise ValueError(f"Invalid JSON in solution file '{path}': {e}") from e

    if not isinstance(data, dict):
        raise ValueError("Solution file must be a JSON object at the top level")

    unknown = sorted(set(data) - _TOP_LEVEL_KEYS)
    if unknown:
        raise ValueError(
            f"Solution file '{path}': unknown top-level key(s) {unknown};"
            f" expected any of {sorted(_TOP_LEVEL_KEYS)}"
        )

    for key in _TOP_LEVEL_KEYS & set(data):
        if not isinstance(data[key], dict):
            raise ValueError(
                f"Solution file '{path}': '{key}' must be a JSON object keyed by name,"
                f" got {type(data[key]).__name__!r}"
            )

    sol = SolutionData()

    # rows
    for name, entry_obj in data.get("rows", {}).items():
        sol.rows[name] = _parse_entry(entry_obj, f"row:{name}")

    # columns — nested: var_name -> {lower_bound, upper_bound, integrality}
    for var_name, col_obj in data.get("columns", {}).items():
        if not isinstance(col_obj, dict):
            raise ValueError(
                f"columns['{var_name}'] must be a JSON object with 'lower_bound'/"
                f"'upper_bound'/'integrality' keys, got {type(col_obj).__name__!r}"
            )
        bad_keys = sorted(set(col_obj) - _COLUMN_SIDES)
        if bad_keys:
            raise ValueError(
                f"columns['{var_name}']: unknown key(s) {bad_keys}; expected any of"
                f" {sorted(_COLUMN_SIDES)}"
            )
        if "lower_bound" in col_obj:
            sol.lower_bounds[var_name] = _parse_entry(
                col_obj["lower_bound"], f"column:{var_name}:lower_bound"
            )
        if "upper_bound" in col_obj:
            sol.upper_bounds[var_name] = _parse_entry(
                col_obj["upper_bound"], f"column:{var_name}:upper_bound"
            )
        if "integrality" in col_obj:
            sol.integers[var_name] = _parse_entry(
                col_obj["integrality"], f"column:{var_name}:integrality"
            )

    return sol
