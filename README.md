<img src="https://www.mixedinteger.org/2027/images/mip-2027.jpg" width="400">

# MIP Competition 2027: Small Infeasible Subsystems

The solution checker for the **MIP Competition 2027: "Explainability and Small Infeasible
Subsystems"**, part of the [2027 MIP Workshop](https://www.mixedinteger.org/2027). See the
[competition page](https://www.mixedinteger.org/2027/competition) for the announcement and rules.

## The task

Given an infeasible MIP in MPS format, find which elements to **remove** — linear constraints,
variable bounds, and integrality constraints — so that what remains is an infeasible subsystem.
The bounds of a variable that stays integer always belong to the subsystem: they are not counted in
its size and need no certificate (see [`SOLUTION_FORMAT.md`](SOLUTION_FORMAT.md#element-types)).
Irreducible infeasible subsystems (IISs) are preferred, and smaller ones are better. A certificate
for a kept element shows that relaxing that element alone makes the remaining subsystem feasible;
certificates for every kept element prove irreducibility. Each valid feasibility certificate for a kept element
counts, even if other kept elements have no certificate. Certificates are checked with an absolute
tolerance of **1e-5** on every row, bound and integrality; see
[`SOLUTION_FORMAT.md`](SOLUTION_FORMAT.md#certificates) for the exact rules.

Submissions will be ranked according to the time to compute the IIS, the size of the IIS, and the
certificates. How these criteria are combined will be announced later.

## What you submit

A program, normally a container image, that the organizers run once per instance:

- how it is called, its limits and its sandbox: **[`PROGRAM_INTERFACE.md`](PROGRAM_INTERFACE.md)**
- what it writes: **[`SOLUTION_FORMAT.md`](SOLUTION_FORMAT.md)**

Sample instances and solutions are in [`smallset/`](smallset/).

## Check your solutions

The checker used for evaluation installs straight from GitHub. Choose a solver extra: `highs`
([highspy on PyPI](https://pypi.org/project/highspy/)), `scip`
([PySCIPOpt on PyPI](https://pypi.org/project/pyscipopt/)), or `gurobi`
([gurobipy on PyPI](https://pypi.org/project/gurobipy/); requires a license for larger models). Use
`solvers` to install all three:

```bash
pip install "mipcc-iis-checker[highs] @ git+https://github.com/mixedinteger/MIPcc27.git@v1.0.0b1"
# or, as an isolated command:
uv tool install "mipcc-iis-checker[highs] @ git+https://github.com/mixedinteger/MIPcc27.git@v1.0.0b1"
```

Then check a solution against its instance:

```bash
mipcc-iis-checker --instance stein15inf.mps.gz --solution stein15inf.json.gz
```

Exit code `0` means PASSED, `1` FAILED. Other flags: `--solver {scip,gurobi,highs}`,
`--time-limit S`, `--tolerance F`, `--output-json report.json`, `--verbose`. The commands above pin
release `v1.0.0b1`; drop the `@v1.0.0b1` for the latest `main`. Changes between releases are listed
in [`CHANGELOG.md`](CHANGELOG.md).

## Questions and bug reports

Please use [GitHub issues](https://github.com/mixedinteger/MIPcc27/issues) for questions about the
rules, the checker or the solution format, and to report checker bugs.

## License

To the extent possible under law, the MIP Competition Committee has waived all copyright and related
rights to this work: it is dedicated to the public domain under [CC0 1.0](LICENSE).

---

_Organizers:_ running the competition is described in [`benchmark/README.md`](benchmark/README.md).
_Developers:_ `uv sync --extra dev`, then `uv run pytest` and `uv run ruff check .`.
