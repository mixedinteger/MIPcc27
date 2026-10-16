# MIP Competition 2027 — Solution File Format

Solution files are **JSON** documents. A solution describes the disposition of every relevant
problem element: whether it is **relaxed** (removed from the subsystem) or **kept** (retained). Kept
elements may carry a **certificate** — a variable assignment proving the subsystem becomes feasible
when that single element is also removed.

---

## Element Types

| Category    | What it means to _relax_                                 |
| ----------- | -------------------------------------------------------- |
| **Row**     | A linear constraint is dropped entirely                  |
| **Bound**   | A variable bound is relaxed to ±∞                        |
| **Integer** | A variable's integrality is relaxed (becomes continuous) |

**Bounds of integer variables are not elements.** The bounds of a variable whose integrality is kept
are always part of the subsystem: the checker applies them in the infeasibility check and in every
certificate check, whether the solution lists them or not, even if it marks them `"relax"` (the
report then says so). They do not count in the subsystem's size, and they need no certificate (one
given is ignored). Once a variable's integrality is relaxed, its bounds are ordinary elements again,
to keep or relax like those of a continuous variable.

---

## JSON Schema

```json
{
  "rows": {
    "<constraint_name>": {
      "action": "relax" | "keep",
      "certificate": { "<var>": <number>, ... }
    }
  },
  "columns": {
    "<var_name>": {
      "lower_bound": { "action": "relax" | "keep", "certificate": { ... } },
      "upper_bound": { "action": "relax" | "keep", "certificate": { ... } },
      "integrality": { "action": "relax" | "keep", "certificate": { ... } }
    }
  }
}
```

### Fields

| Field         | Required | Default  | Meaning                                                                              |
| ------------- | -------- | -------- | ------------------------------------------------------------------------------------ |
| `action`      | No       | `"keep"` | `"relax"` removes the element; `"keep"` retains it                                   |
| `certificate` | No       | none     | Only for `"keep"`: variable values proving feasibility after this element is removed |

Both top-level keys (`rows`, `columns`) are optional, and so is every key nested under a column
(`lower_bound`, `upper_bound`, `integrality`). **Elements not listed default to `{"action": "keep"}`
with no certificate.**

---

## Certificates

A certificate for kept element _e_ is a mapping from variable names to real numbers. It proves that
removing _e_ from the kept subsystem yields a feasible system: the assignment must satisfy every
other kept constraint, every other kept bound, the bounds of every variable that stays integer, and
the integrality of every other kept integer variable.

Variables absent from the certificate default to **0.0**.

A certificate is checked against the original model's coefficients, bounds and right-hand sides,
with an **absolute tolerance of 1e-5** on everything:

- each kept row (other than _e_) may be violated by at most 1e-5: for `≤`, activity − rhs ≤ 1e-5;
  for `≥`, rhs − activity ≤ 1e-5; for `=`, both;
- each kept bound (other than _e_), and each bound of a variable that stays integer: the value may
  lie outside it by at most 1e-5;
- each kept integrality (other than _e_): the value may be at most 1e-5 from the nearest integer;
- _e_ itself must be **violated** by more than 1e-5. A point that satisfies _e_ too satisfies the
  whole kept subsystem, so it proves nothing, and the certificate is rejected.

The tolerance is absolute, not relative, so take care on rows with large coefficients. A point
copied from a solver may need rounding or a tighter solver tolerance to pass. The checker's
`--tolerance` flag exists for experiments only; judging uses the default.

When every kept element has a valid certificate the checker confirms `is_minimal = True` without any
solver calls — purely by arithmetic. Otherwise `is_minimal` stays `False`, which means _not proven_:
the checker verifies only the certificates it is handed, so it never concludes a subsystem is
non-minimal.

A certificate for a kept **integer** element is a point that satisfies every kept row and bound, the
bounds of that variable and of every other variable that stays integer, and the integrality of every
_other_ kept integer variable — so it is typically fractional in exactly that one variable. Relaxing the
integrality removes only the integrality: the variable's bounds still apply. For the example below,
relaxing `x1`'s integrality alone admits such a point, which is what proves the integrality is
load-bearing.

---

## Example

Instance with:

- `c1`: x1 + x2 ≤ 1.5
- `c2`: x1 + x2 ≥ 2.5
- `c3`: x1 − x2 ≤ 0
- x1, x2 ≥ 0, integer

The IIS is {c1, c2} (with c3, lower bounds, and integrality all relaxed). The lower bounds can be
relaxed because the integrality is too; while `x1` stays integer, its bounds apply whatever the
solution says.

```json
{
  "rows": {
    "c3": { "action": "relax" },
    "c1": {
      "action": "keep",
      "certificate": { "x1": 3.0, "x2": 0.0 }
    },
    "c2": {
      "action": "keep",
      "certificate": { "x1": 0.0, "x2": 0.0 }
    }
  },
  "columns": {
    "x1": {
      "lower_bound": { "action": "relax" },
      "integrality": { "action": "relax" }
    },
    "x2": {
      "lower_bound": { "action": "relax" },
      "integrality": { "action": "relax" }
    }
  }
}
```

**Certificate verification:**

- `c1` removed → only `c2` (x1+x2 ≥ 2.5) remains; point (3, 0): 3 ≥ 2.5 ✓
- `c2` removed → only `c1` (x1+x2 ≤ 1.5) remains; point (0, 0): 0 ≤ 1.5 ✓

---

## Validation Rules

The checker verifies:

1. Every referenced element name exists in the instance.
2. The kept subsystem `MIP(C, J)`, with the bounds of every variable in `J` applied, is infeasible
   (solver).
3. Every provided certificate is feasible for its subsystem, to within the tolerance above (no
   solver needed).

Submissions fail if (1) or (2) fail, or if any provided certificate is invalid. Missing certificates
are not penalised — only wrong ones are.

How your program is run and where it writes this file:
[`PROGRAM_INTERFACE.md`](PROGRAM_INTERFACE.md).
