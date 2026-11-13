# Conventions

The names, units and rules that every AFH program (linus, axtb, qcopt, ...) and every qc*
codec share, so that inputs look alike and outputs can be read the same way whichever
program produced them. This page is the single place for them; a program's own documentation
covers only what is specific to it. How programs are run (the program and session
protocol, plans, workers) is qccompute's: see its program interface design.

!!! note "Status"
    The settings names below are what the linus, axtb and qcopt sessions agreed on
    2026-11-11. They await Colton's sign-off.

## Units

Atomic units at every interface: energies in hartree (Eh), positions in bohr, gradients in
Eh/bohr, Hessians in Eh/bohr², dipole moments in e·bohr, quadrupole moments in e·bohr²,
polarizabilities in e²·bohr²/Eh. A field or key whose name says otherwise (`memory_gib`) is
in that unit. Chemistry's fixed conventions are kept where every program prints them:

| quantity | unit |
|---|---|
| frequencies, rotational constants | cm⁻¹ (imaginary frequencies negative) |
| masses | u |
| force constants | mdyn/Å |
| IR intensities | km/mol |
| temperatures, pressures, concentrations | K, Pa, mol/L |
| entropies, heat capacities | Eh/K |
| times | wall-clock seconds |
| memory | GiB (2³⁰ bytes) |

XYZ files are read and written in Å.

## Settings (`ProgramInput.keywords`)

`keywords` is the program's own settings tree, exactly as its JSON request file and its
Python keyword arguments spell it. The rules:

1. Every setting is `group.key` or a top-level scalar, never deeper. A key never repeats its
   group's name (`solvent.name`, not `solvent.solvent`).
2. One canonical spelling across the CLI, JSON, Python and qcdata. Synonyms that encode
   another program's vocabulary (`rijcosx`, `rhf`) may be accepted on input but are never
   echoed back. A group may take a scalar that stands for its main key (linus: `jk: "df"`
   is `{"kind": "df"}`); the schema shows both forms and `resolved.settings` reports the
   expanded one. The store keys the input as given, so the two forms are separate cache
   entries: callers that store results use one.
3. `"auto"` means the program decides; `null` or absent means the default. Every output
   reports what each `auto` became in `resolved.settings`, and why in `resolved.reasons`.
4. Tolerances end in `_tol`, iteration limits are `max_iter`, units appear in names only
   exactly (`memory_gib`). Booleans only for two-state choices.
5. **Every setting given must affect the calculation** (Colton, 2026-11-11, linus D41).
   Unknown keys, settings that do not apply to the calculation type, and settings made
   irrelevant by other values are errors of kind `input` (codes `unknown_key`,
   `not_applicable`, `no_effect`), with a did-you-mean suggestion where one exists.
   *Contingent* settings, which apply only if something happens during the run (an SOSCF
   or escalation threshold, a stability probe's gap, a level shift), are allowed; the
   run's `record` says whether each one fired. A session checks its keywords against the
   calculation types it was opened for (`open_session(..., calctypes=...)`), and a
   request of another type fails with `not_applicable`.
6. Each program publishes the JSON Schema of its settings (`settings_schema(calctype)`),
   with `x-unit`, `x-tier` (common, expert), `x-concern` (model, approximation,
   convergence, algorithm, resource, output) and `x-applies-to` on every key.
7. `model.method` and `model.basis` carry the method and basis; `structure` carries
   positions, atoms, charge and multiplicity. The settings tree does not repeat them.

### Shared names

Settings that mean the same in several programs are spelled the same:

| key | meaning |
|---|---|
| `threads` | CPU threads the calculation may use |
| `memory_gib` | memory the calculation may use (GiB) |
| `backend` | `cpu`, `gpu` or `auto` |
| `device_memory_gib` | GPU memory the calculation may use (GiB) |
| `accuracy` | reserved: a cross-program absolute accuracy target (Eh, Eh/bohr) |
| `scf.max_iter`, `scf.e_tol`, `scf.g_tol`, `scf.guess` | SCF iteration limit, energy and orbital-gradient tolerances, starting guess |
| `solvent.name`, `solvent.model`, `solvent.eps` | implicit solvent: name, continuum model, permittivity |
| `freq.temperature`, `freq.pressure`, `freq.scale`, `freq.imaginary_cutoff` | thermochemistry conditions, frequency scale factor, imaginary-mode cutoff (cm⁻¹) |
| `output.orbitals`, `output.density`, `output.molden`, `output.bond_orders` | optional outputs; `output.molden` is `off`, `spherical` or `cartesian` and returns `files["orbitals.molden"]` |
| `convergence.max_iter`, `convergence.g_tol`, `convergence.e_tol`, `convergence.d_tol` | geometry optimizer convergence |

A program that cannot honour a shared setting (axtb's SCF thresholds follow xtb's accuracy
factor; its ALPB/GBSA models are parametrized per solvent name, so no `solvent.eps`)
rejects it with kind `unsupported` rather than ignoring it.

Resource keys (`threads`, `memory_gib`, `backend`, `device_memory_gib`) carry
`x-concern: "resource"`. The store's calculation key hashes the input as given, defaults
omitted, leaving out only `threads` and `memory_gib`. A program that changes what a
default computes (linus's `jk` becoming `df`) changes the numbers behind an unchanged key,
so its runner's `key_context` names the new numerics epoch.

## Calculation types

`energy`, `gradient`, `hessian`, and the driver types (`optimization`,
`transition_state`, `irc`, `path_optimization`, `conformer_search`, ...). A `hessian`
result also holds the energy and gradient at the same geometry and, when the program can
make them, the harmonic analysis (`vibrational_analysis`) and thermochemistry
(`thermochemistry`), tuned or switched off with `freq.*`.

## Outputs

### Standard fields first

A quantity two programs report goes into a typed field, never into `extras`. `extras` is
for what has no standard home yet.

### `energy_components`

A partition of `energy`: the values add up to it. Standard keys:

| key | term |
|---|---|
| `nuclear_repulsion` | nuclear repulsion |
| `one_electron` | kinetic and nuclear attraction (plus ECPs) |
| `coulomb` | electron-electron Coulomb |
| `exchange` | exact exchange |
| `xc` | exchange-correlation functional (semilocal) |
| `vv10` | VV10 nonlocal correlation |
| `dispersion` | empirical dispersion (D3, D4) |
| `solvation_electrostatic` | continuum solvation, electrostatic |
| `solvation_cds` | SMD cavity-dispersion-solvent structure |
| `electronic` | tight-binding electronic energy |
| `repulsion` | tight-binding or force-field repulsion |
| `electrostatic`, `hydrogen_bond`, `halogen_bond` | further tight-binding and force-field terms |

What a component contains is the program's: GFN2-xTB's `electronic` is not g-xTB's. A
program with a hierarchy of overlapping terms reports the partition here and the hierarchy
in `record`.

### Shared record keys

`record` is shaped by each program's record schema. A few keys mean the same in every
program that reports them:

| key | type | meaning |
|---|---|---|
| `warm_start` | bool or null | on a session frame: the calculation started from its lineage's history (previous orbitals, charges, probe state), not from the input alone, so it can reach a different solution than a fresh calculation; null when the method keeps no lineage state (GFN-FF). Programs add their own detail beside it (linus: where the SCF guess came from, and the lineage it was seeded from) |
| `accuracy` | object | on a request with an `Accuracy`: `target` (the targets as resolved: `energy`, `gradient`, `hessian`), `thresholds` (the program's own convergence thresholds used, under its own names) and, where the program has one, `estimate` (an error estimate per field) |

### Kinds

Fields whose meaning depends on a scheme name it beside them:

| field | kind field | values |
|---|---|---|
| `atomic_charges` | `atomic_charges_kind` | `mulliken`, `lowdin`, `eeq`, `eeqbc`, `cm5`, `hirshfeld` |
| `bond_orders` | `bond_orders_kind` | `mayer` (Σ(PS)μν(PS)νμ; what xtb prints as "Wiberg"), `wiberg` (orthogonalized basis) |
| `scf_quadrupole_moment` | `quadrupole_kind` | `traceless` (Buckingham: ½Σq(3rᵢrⱼ − r²δᵢⱼ)), `second_moment` (Σq rᵢrⱼ) |

### Imaginary modes

Frequencies are negative for imaginary modes and stored without a cutoff;
`VibrationalAnalysis.n_imaginary(cutoff)` applies one (xtb and axtb use 20 cm⁻¹).

### Atomic orbitals

Orbitals are stored as the program produced them, with their `AOConvention`;
`Orbitals.with_convention` converts. The default is PySCF's (libcint's) order: spherical
functions m = −l..l except p as (x, y, z); Cartesian components lexicographic (xx, xy,
xz, yy, yz, zz); every spherical function and the x^l component unit-normalized.

### Given results

`ProgramInput.given` holds results already known at the input's structure (a
`SinglePointData`). It is a record of what is known, not a setting, so the strict rule for
settings (rule 5) applies to it differently:

- A program uses the fields it can and records each one it used in `resolved`
  (`hessian.initial: "given"`, `scf.guess: "given"`). Fields it does not use, descriptive
  ones included (`calcinfo_*`, `scf_iterations`, `energy_components`), are left alone, so
  `given=frame.results` works as it stands.
- Given values are starting points, never results: a program does not report a given
  value as its own (qcopt computes the energy and gradient at a given TS and takes only
  the Hessian). `provenance` names the producing program and version only, so reusing a
  value as a result could silently mix methods (GFN2 and g-xTB are both "axtb").
- It refuses the input (kind `input`, code `not_applicable`) when it can use none of the
  fields, or when a field it would use does not fit: other atoms (qcdata checks shapes),
  another basis for a wavefunction. A starting guess needs no match of program or
  method: a GFN2 Hessian is a sound initial Hessian for a DFT transition-state search.
- Checking the model is the caller's, which knows the producing input
  (`qcopt.irc(ts_out)` passes the search's Hessian only when its spec matches).

### Warnings

`ProgramWarning(code, message, path)`: `code` is stable snake_case; `path` is the dotted
settings key concerned, if any. Settings problems are errors, not warnings (rule 5), so
warnings report what happened during a run (a fallback taken, a probe fired, slow
convergence). One shared code:

| code | meaning |
|---|---|
| `deprecated_key` | after a program's 1.0, an old key still accepted for a stated period; the message names the new one. Before 1.0 a program rejects old keys (`unknown_key`) instead |
| `resource_capped` | a resource request in the keywords (`threads`, `memory_gib`, ...) above what the runner granted; the program used the grant (it uses the smaller of request and grant, or the grant alone when nothing was requested) |
| `backend_fallback` | the calculation ran on another backend than first chosen (a GPU out of memory, so the CPU), which `backend: auto` allows; an explicit backend fails instead |

Programs define further codes in their own documentation.

Shared error codes (kind `input`):

| code | meaning |
|---|---|
| `unknown_key` | not a setting of this program |
| `not_applicable` | a setting of another calculation type, or a session request outside the calctypes it was opened for |
| `no_effect` | made irrelevant by the other values given |
| `not_modeled` | the structure's charge or multiplicity is one the method does not represent (GFN-FF given a radical, a neutral-only ML potential given an ion). The message names the property. A method never runs on, or silently ignores, a charge or multiplicity it cannot model |

Shared error codes of other kinds, which runners and schedulers act on:

| kind | code | meaning |
|---|---|---|
| `resource` | `host_memory` | out of host memory (a failed allocation, or the plan's need above the grant) |
| `resource` | `device_memory` | out of GPU memory |
| `backend` | `device_lost` | a GPU fault poisoned the device context (illegal address, launch failure, device assert, ECC error, launch timeout); nothing more can run in this process. A worker answers the request in flight with it and then exits; in process, the caller should not use the program again |
| `environment` | `worker_died` | a worker process ended mid-request (a crash, a signal, an out-of-memory kill); set by the client, with the exit status and the tail of the worker's stderr in the message. A worker that exits on purpose answers the request in flight first (`device_lost`), so only later requests see `worker_died` |

### Errors

`ErrorInfo(kind, message, code, program, cause)`, on failed outputs only. Kinds:

| kind | meaning |
|---|---|
| `input` | settings or a structure the program cannot run as given |
| `unsupported` | a valid request the program does not implement, or a setting it cannot honour |
| `convergence` | an iterative procedure did not converge (results so far attached) |
| `numerical` | non-finite values, singular matrices |
| `resource` | not enough memory or another resource as set up |
| `backend` | the hardware backend failed |
| `program` | a child calculation failed; `cause` holds its error |
| `timeout` | out of time (set by the runner) |
| `environment` | the program could not run: not installed, no licence, process died (set by the runner) |
| `internal` | a bug in the program; on the codec path, any failure the codec does not classify (a reported failure, an unexplained exit, unreadable output) |

`code` refines a kind with a program-defined name callers can branch on (qcopt:
`max_iter`, `energy_rose`, `not_first_order_saddle`, `irc_incomplete`).

A driver reports any failure of a child as kind `program`, whether the child's plan
refused its settings or its run failed. `program` names the child, `cause` holds the
child's own `ErrorInfo`, and the message starts with the child and calctype
(`linus (gradient): ...`). Callers follow `cause` to the first error that is not of kind
`program` to learn what went wrong. A child that fails for one task of a round (a session
that will not open, a failed request) fails that task only. A child that is a Python
callable rather than a program (qcopt's in-process evaluator) is reported the same way,
with `program` None. Its `ProgramError` gives the cause; any other exception becomes a
cause of kind `internal`, as qccompute treats a program's bug.

### Timings and provenance

`execution.timings` holds seconds per phase (`total` for the whole call; drivers add
`children`). `results.provenance` names the `program`, `program_version` and
`program_commit`.
