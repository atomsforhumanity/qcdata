# Conventions

The names, units and rules that every AFH program (linus, axtb, qcopt, ...) and every qc*
codec share, so that inputs look alike and outputs can be read the same way whichever
program produced them. This page is the single place for them; a program's own documentation
covers only what is specific to it. Design background: [the program
interface](design/program-interface.md).

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
   echoed back.
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
| `output.orbitals`, `output.density`, `output.molden`, `output.bond_orders` | optional outputs |
| `convergence.max_iter`, `convergence.g_tol`, `convergence.e_tol`, `convergence.d_tol` | geometry optimizer convergence |

A program that cannot honour a shared setting (axtb's SCF thresholds follow xtb's accuracy
factor; its ALPB/GBSA models are parametrized per solvent name, so no `solvent.eps`)
rejects it with kind `unsupported` rather than ignoring it.

Resource keys (`threads`, `memory_gib`, `backend`, `device_memory_gib`) carry
`x-concern: "resource"`. Result identity is computed from `resolved.settings`, never from
these raw keys: memory and backend choices that can change numbers at round-off appear
there resolved.

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

### Warnings

`ProgramWarning(code, message, path)`: `code` is stable snake_case; `path` is the dotted
settings key concerned, if any. Settings problems are errors, not warnings (rule 5), so
warnings report what happened during a run (a fallback taken, a probe fired, slow
convergence). One shared code:

| code | meaning |
|---|---|
| `deprecated_key` | after a program's 1.0, an old key still accepted for a stated period; the message names the new one. Before 1.0 a program rejects old keys (`unknown_key`) instead |

Programs define further codes in their own documentation.

Shared error codes (kind `input`):

| code | meaning |
|---|---|
| `unknown_key` | not a setting of this program |
| `not_applicable` | a setting of another calculation type, or a session request outside the calctypes it was opened for |
| `no_effect` | made irrelevant by the other values given |
| `not_modeled` | the structure's charge or multiplicity is one the method does not represent (GFN-FF given a radical, a neutral-only ML potential given an ion). The message names the property. A method never runs on, or silently ignores, a charge or multiplicity it cannot model |

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
| `internal` | a bug in the program |

`code` refines a kind with a program-defined name callers can branch on (qcopt:
`max_iter`, `energy_rose`, `not_first_order_saddle`, `irc_incomplete`).

### Timings and provenance

`execution.timings` holds seconds per phase (`total` for the whole call; drivers add
`children`). `results.provenance` names the `program`, `program_version` and
`program_commit`.
