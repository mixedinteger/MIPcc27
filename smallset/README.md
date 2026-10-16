# smallset

A sample of 62 small infeasible models, with one solution for each, to try the checker on.

- `models/<name>.mps.gz`: the model.
- `certificates/<name>.json.gz`: a solution for it, in the [solution format](../SOLUTION_FORMAT.md).

```bash
mipcc-iis-checker --instance models/stein15inf.mps.gz --solution certificates/stein15inf.json.gz
```

The solutions were computed with Gurobi's IIS, with feasibility certificates added. They pass the
checker, but they are not necessarily irreducible, nor the smallest possible.

## Sources

The models keep the terms of their original sources, linked below.

**Guieu and Chinneck's infeasible MIPs** (12):
[www.sce.carleton.ca/faculty/chinneck/InfeasibleMIPs](http://www.sce.carleton.ca/faculty/chinneck/InfeasibleMIPs/).
`bell3ainf`, `bell3binf`, `bell5inf`, `dell`, `flugplinf`, `g503inf`, `jjliu_clique`, `misc05inf`,
`mod008inf`, `p0201inf`, `stein15inf`, `stein45inf`.

Most were made from existing MIPs by adding a constraint parallel to the objective, which keeps the
LP relaxation feasible but makes the MIP infeasible. Three are infeasible as they were found: `dell`
(Robert Dell, Naval Postgraduate School), `p0201inf` (Ed Klotz) and `jjliu_clique` (Ed Klotz;
register allocation in compilers, from K. Wilken, J. Liu and M. Heffernan, "Optimal instruction
scheduling using integer programming", PLDI 2000). Several are also in MIPLIB 2017.

> O. Guieu and J. W. Chinneck (1999), "Analyzing Infeasible Mixed-Integer and Integer Linear
> Programs", _INFORMS Journal on Computing_ 11(1), 63–77.

**MIPLIB 2017** (19): [miplib.zib.de](https://miplib.zib.de). `enlight4`, `enlight11`,
`no-ip-65059`, `stein10i`, and the `neos-*` models: `neos-1461051`, `neos-3135526-osun`,
`neos-3135947-otra`, `neos-3426895-dijle`, `neos-3426909-dingo`, `neos-3426989-dives`,
`neos-3620512-karun`, `neos-4331170-senio`, `neos-4344877-stour`, `neos-4558049-warta`,
`neos-4648110-ythan`, `neos-5115982-kedron`, `neos-807454`, `neos-942886`, `neos859080`.

> A. Gleixner et al. (2021), "MIPLIB 2017: data-driven compilation of the 6th mixed-integer
> programming library", _Mathematical Programming Computation_ 13, 443–490.

**Ed Klotz** (1): `mathologer4a`.

**SATLIB** (12), unsatisfiable SAT instances from the DIMACS benchmark set, written as 0-1 programs:
[www.cs.ubc.ca/~hoos/SATLIB](https://www.cs.ubc.ca/~hoos/SATLIB/benchm.html).

- random 3-SAT (AIM): `aim-50-2_0-no-4`, `aim-100-1_6-no-4`, `aim-200-1_6-no-2`;
- random SAT with clauses of varying length (JNH): `jnh10`, `jnh15`, `jnh215`;
- pigeon hole: `hole9`, `hole10`;
- circuit fault analysis, bridge faults (BF): `bf1355-075`, `bf1355-638`, `bf2670-001`;
- circuit fault analysis, single-stuck-at faults (SSA): `ssa0432-003`.

> H. H. Hoos and T. Stützle (2000), "SATLIB: An Online Resource for Research on SAT", in _SAT 2000_,
> IOS Press, 283–292.

**Pseudo-Boolean** (7), pseudo-Boolean benchmark instances written as 0-1 programs:
`normalized-11alg_arraycomm_less`, `normalized-3_4_13`, `normalized-domset-hexgrid_03_052`,
`normalized-domset-stargod_09_084`, `normalized-grid4_31.shuf-2`,
`normalized-sumineqArity4houseP016`, `normalized-vertexcover-grid10x19-hard`.

