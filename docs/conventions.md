# Conventions

The names, units and rules that every AFH program (axtb, qcopt, ...) and every qc*
codec share, so that inputs look alike and outputs can be read the same way whichever
program produced them. This page is the single place for them; a program's own documentation
covers only what is specific to it. How programs are run (the program and session
protocol, plans, workers) is qccompute's: see its program interface design.

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
   echoed back. A group may take a scalar that stands for its main key (a DFT program's
   `jk: "df"` is `{"kind": "df"}`); the schema shows both forms and `resolved.settings`
   reports the expanded one. The [calculation key](#calculation-key) takes the input as given, so the
   two forms are separate cache entries: callers that store results use one.
3. `"auto"` means the program decides; `null` or absent means the default. Every output
   reports what each `auto` became in `resolved.settings`, and why in `resolved.reasons`.
4. Tolerances end in `_tol`, iteration limits are `max_iter`, units appear in names only
   exactly (`memory_gib`). Booleans only for two-state choices.
5. **Every setting given must affect the calculation.**
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
8. `keywords` hold plain JSON. A pydantic model given in them, at any depth (an
   `Accuracy`), is stored as its JSON form with nulls left out, so an input saves,
   reloads and keys the same.

### Shared names

Settings that mean the same in several programs are spelled the same:

| key | meaning |
|---|---|
| `threads` | CPU threads the calculation may use |
| `memory_gib` | memory the calculation may use (GiB) |
| `backend` | `cpu`, `gpu` or `auto` |
| `device_memory_gib` | GPU memory the calculation may use (GiB) |
| `accuracy` | the input's [accuracy target](#accuracy-targets): `energy` (Eh), `gradient` (Eh/bohr), `hessian` (Eh/bohr²) |
| `scf.max_iter`, `scf.e_tol`, `scf.g_tol`, `scf.guess` | SCF iteration limit, energy and orbital-gradient tolerances, starting guess |
| `solvent.name`, `solvent.model`, `solvent.eps` | implicit solvent: name, continuum model, permittivity |
| `freq.temperature`, `freq.pressure`, `freq.scale`, `freq.imaginary_cutoff` | thermochemistry conditions, frequency scale factor, imaginary-mode cutoff (cm⁻¹) |
| `output.orbitals`, `output.overlap`, `output.hamiltonian`, `output.core_hamiltonian`, `output.molden`, `output.bond_orders` | optional outputs: the [wavefunction outputs](#wavefunction-outputs) by name; `output.molden` is `off`, `spherical` or `cartesian` and returns `files["orbitals.molden"]` |
| `convergence.max_iter`, `convergence.g_tol`, `convergence.e_tol`, `convergence.d_tol` | geometry optimizer convergence |

A program that cannot honour a shared setting (axtb's SCF thresholds follow xtb's accuracy
factor; its ALPB/GBSA models are parametrized per solvent name, so no `solvent.eps`)
rejects it with kind `unsupported` rather than ignoring it.

Resource keys (`threads`, `memory_gib`, `backend`, `device_memory_gib`) carry
`x-concern: "resource"`. The [calculation key](#calculation-key) leaves out `threads`,
`memory_gib` and `device_memory_gib`, which change results at most at round-off; `backend`
(which device runs the calculation) counts.

### Accuracy targets

`accuracy` is a JSON object of absolute targets, each optional: `energy` (Eh), `gradient`
(Eh/bohr, the largest component) and `hessian` (Eh/bohr², the largest element), as in
`qcdata.Accuracy`. It is never a bare number: it applies to every calctype a session
runs, so a number would have no single quantity to aim at.

- It is the input's own target: of `compute`, and of every session request that gives
  none. A request's `accuracy` replaces it for that request.
- The program turns it into its own convergence thresholds and records the target, the
  thresholds and any error estimate under the record key `accuracy`. Explicit
  convergence settings are floors: a target may tighten them, never loosen them.
- It is part of the input, so of the [calculation key](#calculation-key): a tighter
  target is another calculation.
- A program with no convergence settings (a force field, an ML potential, a
  `FunctionProgram`) accepts a target, as the keyword or a session request's alike, and
  records it as met: no thresholds, and a convergence error of exactly zero for each
  target. So a session can pass a request's target on as the keyword.
- A program whose convergence settings take no targets yet refuses a target with kind
  `unsupported`, like any shared setting it cannot honour.

The record of a target met with no thresholds, for `{"gradient": 1e-6}`:

```json
"record": {"accuracy": {"target": {"gradient": 1e-06}, "thresholds": {},
                        "estimate": {"gradient": 0.0}}}
```

`target` holds the targets given, `thresholds` is empty, and `estimate` gives `0.0` for
each target, so a reader checks any program the same way: each `estimate` at or below its
`target`.

`Accuracy.from_keywords(keywords)` reads it the same way in every program, and refuses a
bare number, an unknown key (`unknown_key`) and an object with no target (`no_effect`).

## Calculation key

A store files each result under `qcdata.calculation_key(inp)`: `k1:` followed by the
sha256 hex digest of a canonical JSON text of the input as given, and nothing else. Two
inputs share a key only when they request the same calculation written the same way.
Every rule errs toward a miss, because a wrong hit returns another calculation's numbers.

The canonical text is the input's canonical form, written as compact JSON (separators `,`
and `:`), keys sorted, non-ASCII characters escaped as `\uXXXX`, with:

- **numbers normalized**: a float with an integral value is written as an integer (`100.0`
  as `100`, `-0.0` as `0`), any other float in Python's shortest round-trip form
  (`1e-08`), integers exactly. `true` and `1` stay distinct. Non-finite numbers are
  refused;
- **fields at their default left out** (`charge: 0`, `keywords: {}`, `basis: null`), so
  adding a field to qcdata never changes an old key. Inside `keywords` nothing is a
  default: values stay as given, `null`s and empty groups included;
- **coordinates** of `structure` and of every structure in `structures`: bohr times 10¹⁰,
  rounded to integers (half to even), atom order kept. Symbols, charge, multiplicity and
  connectivity count as given;
- **every other array field** (a `given` Hessian, orbital coefficients) as
  `{"array": shape, "dtype": "float64", "sha256": digest}`, the digest of its C-order
  little-endian float64 bytes with `-0.0` as `0.0`. The digest tells values apart exactly
  as their JSON numbers would, at a small fraction of the cost. Integer arrays enter the
  same way in their own dtype. Non-finite values are refused;
- **files** (of the input, of each subprogram and of `given`) by the sha256 hex digest of
  their bytes, text as UTF-8;
- **subprograms** canonicalized the same way, in order;
- tuples as lists; numpy arrays and scalars inside free-form values (`keywords`) as
  JSON numbers, as they are once saved.

Left out: `extras` at every level, structure `identifiers`, and the resource keywords
`threads`, `memory_gib` and `device_memory_gib` at the top level of `keywords`, in the
input and in every subprogram (`scf.threads` would count). These three may change results
at most at round-off. Everything else counts: `given`, `cmdline_args`, and settings such
as `backend` (pinning the device), precision and `accuracy` that can change numbers.

Spellings are not normalized. `B3LYP` and `b3lyp`, a synonym, a group's scalar and
expanded forms (rule 2), a default written out (`scf.max_iter: 100`), and `null` against
an absent key all give different keys: a cache miss, never a wrong hit. The key
normalizes only itself; stored values keep `-0.0` and their number types.

The key names the calculation, never the code that produced its numbers. Which code
did is each program's numerics identity (its version, or a numerics epoch, an
executable's version, a model's checksum), which the runner and the store record with
every run, for the input's program and each subprogram, and which every output carries
in `execution.identities`. A store keeps several runs per key and reuses one only when
its identities match the programs that would run now. So a
key never changes when code changes, anything holding keys stays valid across releases,
and computing a key never needs a program installed. A program that changes what a
default computes (a DFT program's `jk` defaulting to `df`) changes its numerics
identity, not the key.

Golden keys in qcdata's tests pin the canonical form. Changing any rule, the default of
any field an input holds (an omitted field would then mean something else), or the type of
a field between a list of numbers and an array (`list[float]` and a digest enter
differently) is a new key version, `k2`, released deliberately.

## Calculation types

`energy`, `gradient`, `hessian`, and the types made of many evaluations
(`optimization`, `transition_state`, `irc`, `path_optimization`, `conformer_search`, ...),
usually run by a program with subprograms. A `hessian`
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
| `warm_start` | bool or null | on a session frame: the calculation started from its lineage's history (previous orbitals, charges, probe state), not from the input alone, so it can reach a different solution than a fresh calculation; null when the method keeps no lineage state (GFN-FF). Programs add their own detail beside it (a DFT program: where the SCF guess came from, and the lineage it was seeded from) |
| `accuracy` | object | on a calculation with an [accuracy target](#accuracy-targets) (the keyword's or a request's): `target` (the targets as resolved: `energy`, `gradient`, `hessian`), `thresholds` (the program's own convergence thresholds used, under its own names; empty for a program with none) and, where the program has one, `estimate` (an error estimate per field; `0.0` for each target of a program with no convergence settings) |

### Kinds

Fields whose meaning depends on a scheme name it beside them:

| field | kind field | values |
|---|---|---|
| `atomic_charges` | `atomic_charges_kind` | `mulliken`, `lowdin`, `eeq`, `eeqbc`, `cm5`, `hirshfeld` |
| `atomic_spin_populations` | `atomic_spin_populations_kind` | those of `atomic_charges_kind` |
| `bond_orders` | `bond_orders_kind` | `mayer` (Σ(PS)μν(PS)νμ; what xtb prints as "Wiberg"), `wiberg` (orthogonalized basis) |

Fields with one definition have no kind:

- `dipole_moment` and `quadrupole_moment` are the moments of the method's charge
  distribution, electrons and nuclei, whatever the method: an SCF density, a force
  field's or tight-binding model's charges (GFN-FF's EEQ charges), an ML potential's
  prediction. The quadrupole is Buckingham's traceless tensor, ½Σq(3rᵢrⱼ − r²δᵢⱼ) about
  the coordinate origin. tblite's molecular quadrupole, which the g-xTB 2.0.1 program
  prints, counts the off-diagonal charge and dipole parts twice and is not a tensor.
- `atomic_dipoles` and `atomic_quadrupoles` are electronic only, about each atom's
  nucleus: μ_A = −Σ_{μ∈A}(P D)_μμ, with the integrals about the atom of the column AO;
  quadrupoles in Buckingham's traceless form, 3/2 M − 1/2 tr(M) I, 3 × 3 per atom.
- `shell_populations` are electron populations Σ_{μ∈shell}(PS)_μμ, indexed like
  `basis.shells`, with `shell_atoms` and `shell_angular_momenta` beside them; not charges.

### Imaginary modes

Frequencies are negative for imaginary modes and stored without a cutoff;
`VibrationalAnalysis.n_imaginary(cutoff)` applies one (xtb and axtb use 20 cm⁻¹).

### Orbitals and AO matrices

`Orbitals` is the one home of orbital levels. Energies and occupations come with every
wavefunction; coefficients only on request. Occupations are per spin,
each between 0 and 1, with both `occupations_a` and `occupations_b` always given;
`electronic_temperature` is 0 for aufbau occupations, and smeared ones carry a Fermi level
per spin. `Orbitals.core_electrons` counts, per atom, the electrons the orbitals do not
represent (ECP cores, or the core a valence-only method leaves out; None for an
all-electron calculation), so the occupations sum to Σ(Z − core_electrons) − charge. An
ECP carries the core it replaces (`Ecp.core_electrons`), and the two agree on every ECP
atom.

The coefficients and the AO matrices are indexed by the AOs of one basis set,
`Wavefunction.basis`, present whenever any of them is and only then: an overlap alone
ships the basis set and no coefficients. They are stored as the program produced them,
with the basis set's `AOConvention`; `Wavefunction.with_convention` converts them all
together. The default is PySCF's (libcint's) order: spherical functions
m = −l..l except p as (x, y, z); Cartesian components lexicographic (xx, xy, xz, yy, yz,
zz); every spherical function and the x^l component unit-normalized. A change of Cartesian
normalization scales coefficients by s per AO, densities by s_μ s_ν and operators
(`overlap`, `hamiltonian_a/_b`, `core_hamiltonian`) by 1/(s_μ s_ν); `BasisSet.transform`
takes the array's `kind` and never guesses it.

### Wavefunction outputs

The optional parts of `SinglePointData.wavefunction` are requested by name, in a session
request's `outputs` or as the keyword `output.<name>`:

| output | adds | implies |
|---|---|---|
| `orbitals` | `Orbitals.coefficients_a/_b` | the basis set |
| `overlap` | `Wavefunction.overlap` | the basis set |
| `hamiltonian` | `Wavefunction.hamiltonian_a/_b` | `overlap`, `orbitals` |
| `core_hamiltonian` | `Wavefunction.core_hamiltonian` | the basis set |

The basis set (`Wavefunction.basis`) is not an output of its own: it comes with any of
the four, since it describes the AO space they share. A program fills in what an output
implies instead of refusing it, since requesting an output changes no other number. It refuses an output it cannot give with kind
`unsupported`, code `not_applicable`, and a message naming the reason: a mode whose last
diagonalization is not of one build refuses `hamiltonian`; a force field has none of the
four. There is no `density` output: the SCF density is `Orbitals.density()`.

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
convergence). Shared codes:

| code | meaning |
|---|---|
| `deprecated_key` | after a program's 1.0, an old key still accepted for a stated period; the message names the new one. Before 1.0 a program rejects old keys (`unknown_key`) instead |
| `resource_capped` | a resource request in the keywords (`threads`, `memory_gib`, ...) above what the runner granted; the program used the grant (it uses the smaller of request and grant, or the grant alone when nothing was requested) |
| `backend_fallback` | the calculation ran on another backend than first chosen (a GPU out of memory, so the CPU), which `backend: auto` allows; an explicit backend fails instead |
| `loosely_converged` | an iterative procedure (an SCF, response equations) missed its thresholds and was accepted under a relaxed rule (a stalled SCF accepted on an energy plateau, ORCA's continuation after "near convergence"). The message gives the residual reached and the rule that accepted it; `path` names the threshold missed (`scf.g_tol`), if it is a setting. Without this warning a success met its thresholds |

Programs define further codes in their own documentation.

Shared error codes (kind `input`):

| code | meaning |
|---|---|
| `unknown_key` | not a setting of this program |
| `not_applicable` | a setting of another calculation type, or a session request outside the calctypes it was opened for. With kind `unsupported`, an optional output the program cannot give |
| `no_effect` | made irrelevant by the other values given |
| `not_modeled` | the structure's charge or multiplicity is one the method does not represent (GFN-FF given a radical, a neutral-only ML potential given an ion). The message names the property. A method never runs on, or silently ignores, a charge or multiplicity it cannot model |
| `unnamed_subprogram` | a calculation ran, through its context (`ctx.compute`, `ctx.session`), a program its input does not name among its subprograms (recursively); the message names it. Refused so that a run's identities cover every program it used |

Shared error codes of other kinds, which runners, schedulers and retrying workflows act
on:

| kind | code | meaning |
|---|---|---|
| `resource` | `host_memory` | out of host memory (a failed allocation, or the plan's need above the grant); `details` as below |
| `resource` | `device_memory` | out of GPU memory; `details` as below |
| `backend` | `device_lost` | a GPU fault poisoned the device context (illegal address, launch failure, device assert, ECC error, launch timeout); nothing more can run in this process. A worker answers the request in flight with it and then exits; in process, the caller should not use the program again |
| `convergence` | `scf_not_converged` | the SCF ended without meeting its thresholds (out of iterations, or stalled with no rule to accept it). The results hold the last iterate: its energy and, where the program gives them, its orbitals; the record holds what the program keeps of its history |
| `convergence` | `response_not_converged` | the response equations (CPHF/CPKS, for a Hessian or a property) ended above their tolerance; the results reached are attached |
| `environment` | `worker_died` | a worker process ended mid-request (a crash, a signal, an out-of-memory kill); set by the client, with the exit status and the tail of the worker's stderr in the message. A worker that exits on purpose answers the request in flight first (`device_lost`), so only later requests see `worker_died` |

`ErrorInfo.details` holds structured facts about a failure as plain JSON, so a caller never
parses the message. For `host_memory` and `device_memory` the keys are, each given when
the program knows it:

| key | meaning |
|---|---|
| `required_gib` | the memory (GiB) the calculation needed: its plan's need, or at least what had been allocated plus the request that failed |
| `available_gib` | the memory (GiB) it could use: its budget (`memory_gib`, `device_memory_gib` or the grant) or what the device had free |

```json
{"kind": "resource", "code": "host_memory",
 "message": "RI-K needs at least 1.25 GiB for its 3-index exchange tensor, more than its budget of 1.19 GiB",
 "details": {"required_gib": 1.25, "available_gib": 1.19}}
```

A dispatcher that resubmits with more memory reads `required_gib`; qc* itself still
schedules nothing by memory. Programs may add their own keys beside these, and other
codes may define theirs.

### Errors

`ErrorInfo(kind, message, code, program, cause, details)`, on failed outputs only. Kinds:

| kind | meaning |
|---|---|
| `input` | settings or a structure the program cannot run as given |
| `unsupported` | a valid request the program does not implement, or a setting it cannot honour |
| `convergence` | an iterative procedure did not converge (results so far attached) |
| `numerical` | non-finite values, singular matrices |
| `resource` | not enough memory or another resource as set up |
| `backend` | the hardware backend failed |
| `program` | a subprogram failed; `cause` holds its error |
| `timeout` | out of time (set by the runner) |
| `stopped` | the program stopped on request at one of its steps, its partial output attached (see [Stopping](#stopping)) |
| `environment` | the program could not run: not installed, no licence, process died (set by the runner) |
| `internal` | a failure nobody anticipated: a bug (an exception other than `ProgramError`, a broken invariant, a crash) or, on the codec path, a failure no rule classifies (a reported failure, an unexplained exit, unreadable output). A failure a program or codec anticipates takes the kind that fits it |

`code` refines a kind with a name callers can branch on: a shared code above, else the
program's own. An optimizer's convergence failures keep their own codes (qcopt:
`max_iter`, `energy_rose`, `not_first_order_saddle`, `irc_incomplete`); an SCF's and a
response's are the shared `scf_not_converged` and `response_not_converged`, whichever
program runs them.

A failed calculation raises `ProgramError`, from every program's `compute` and every
session call alike, carrying its `ErrorInfo` and the partial results; any other exception
is a bug. A program whose result is one output attaches it whole,
`ProgramError(out.error, output=out)`, so a trajectory's frames, record and resolved
settings reach the caller as `exc.output`; a failed session step attaches its `frame`. A
runner turns the exception into a failed `ProgramOutput` only where it needs data (a
store, the outputs of a list, a worker's reply).

A program reports any failure of a subprogram as kind `program`, whether the
subprogram's plan refused its settings or its run failed. `program` names the
subprogram, `cause` holds the subprogram's own `ErrorInfo`, and the message starts with
the subprogram and calctype (`xtb (gradient): ...`). Callers follow `cause` to the first
error that is not of kind `program` to learn what went wrong. A subprogram that fails for
one task of a round (a session that will not open, a failed request) fails that task
only. A Python callable standing in for a subprogram (qcopt's in-process evaluator) is
reported the same way, with `program` None. Its `ProgramError` gives the cause; any other
exception becomes a cause of kind `internal`, as qccompute treats a program's bug.

### Timings and provenance

`execution.timings` holds seconds per phase (`total` for the whole call; a program with
subprograms adds `subprograms`, the time inside them). `results.provenance` names the
`program`, `program_version` and `program_commit`. `execution.identities` says which code
made the results: `{program: numerics}` for the input's program and each subprogram it
names, as the runner saw them when it ran (a program named with several specs joins its
sorted, distinct identities with `;`). The runner sets it on every output it returns and
the store keeps it with the run, so an output read back from a store names its code
without a second lookup; a carried result keeps the identities that made it. An output a
store holds names its run in `execution.store_run`, `"<key>/<run>"`: the store sets it on
every output it returns, and a runner sets it on an output it has just stored; it is never
stored itself, and it is None on an output no store holds. An output of a batch also has
`timings["batch"]`, the whole batch's wall time (s), set by the runner; its `wall_time` is
its own where the program can tell (a lockstep batch knows when each task finished), else
the batch's.

### Progress events

A program reports what it is doing through its runner context, `ctx.progress(event)`, when
the caller listens (it is None otherwise). An event is a small JSON object of the
program's own (plain JSON values, in the units above, small enough to send every step),
with one shared key: `event`, a snake_case name of what happened (`step`). Programs
document their events. The runner adds `path`, the program names from the caller's input
down to the program that emitted it; programs do not set it. Events are not results:
nothing in them is stored, and no output depends on whether anyone listened.

### Stopping

A caller may ask a running calculation to stop at its next step (qccompute's
`runner.stop(future)`), to keep what it has reached: a search's frames so far, to analyze
now. The program sees the request as `ctx.stop_requested()`, checks it where it can end
cleanly (an optimizer: before each step), and ends there: its output is a failure of kind
`stopped` with the partial results, raised as for any failure
(`ProgramError(out.error, output=out)`). The request covers the whole calculation: a
program whose subprogram stopped stops too, with kind `stopped` rather than `program`. A
program with no steps to stop at need not check, and runs to its end.

A stopped output is where the caller chose to end the run, not what the input gives: a
runner never stores one, and a store never reuses one.

## Saved files

`save` writes a model as JSON, YAML or TOML. Indented JSON (the default, `indent=4`)
keeps each innermost array, a list holding no lists or objects, on one line, so a
structure's coordinates read as one row per atom:

```json
"geometry": [
    [-8.601339697312277, -0.6368849812267334, 0.7906010193844016],
    [-6.3155913218934, 0.4215960361624519, 1.2240196532066823]
]
```

An array field whose float64 size exceeds
1 MiB (more than 131,072 numbers) goes to its own `.npy` file beside the saved file, named
by the file's stem and the array's path in the document, and the document holds a pointer
in its place:

```json
"hessian": {"path": "output.results.hessian.npy", "shape": [480, 480],
            "dtype": "float64", "sha256": "..."}
```

`sha256` is the digest of the `.npy` file's bytes. Path parts other than letters, digits,
`_` and `-` are written as `%XX` (a structure role `qm.mm` gives
`structures.qm%2Emm.geometry`). `open` reads every array back and checks its digest,
shape and dtype, so models always hold arrays and no field is a file reference. `save`
returns every file it wrote, the model's file first. Smaller arrays and free-form values
(`extras`, `record`) stay in the document, where JSON and YAML keep every float exactly;
TOML loses the sign of `-0.0`.

A text in `files` whose UTF-8 size exceeds 1 MiB (a Molden or cube file, a long log) goes
beside the saved file the same way: its UTF-8 bytes, named by the stem and its path in
the document with no suffix added, and a pointer of its path and the file's sha256 in its
place:

```json
"files": {"orbitals.molden": {"path": "output.results.files.orbitals%2Emolden",
                              "sha256": "..."}}
```

`open` reads it back and checks the digest. Binary files stay in the document, base64
encoded.
