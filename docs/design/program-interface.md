# The AFH program interface: one way to run programs, one shape for their data

Status: **draft proposal** (2026-11-11), from the qc* stack, agreed in substance by the
linus, axtb and qcopt sessions (their amendments are folded in). Phase 1 is implemented on
the qcdata branch `program-interface`; phase 2 on qcdata `program-protocol` and qccompute
`native-programs` (the qccodec adapter that wraps a Codec as a Program is still to do). Items marked **[Colton]** need a decision.

## Why

AFH now writes its own programs (linus, axtb, qcopt, more to come) faster than it wraps
other people's. What we need from qc* has changed accordingly:

1. **Standard data out.** Apps, visualizations, analyses and workflows should read one set of
   typed fields whatever program produced them. Today each of our programs puts its most
   useful output (energy components, resolved settings, vibrational analysis,
   thermochemistry, orbitals, warnings, error kinds, timings) in `extras`, under its own names.
2. **One way in.** A calculation is described the same way for every program, with each
   program's settings validated against a schema the program publishes.
3. **One way to run.** A program written at AFH implements one small protocol and gets
   qccompute's parallelism, batching, GPU scheduling, child calculations and sessions
   without the codec ceremony that exists for wrapping external binaries.

## The layers

```
apps · visualization · analysis · workflows        read qcdata, nothing else
──────────────────────────────────────────────
qcdata     the standard: inputs, outputs, typed result blocks, JSON Schema,
           the Program/Session protocols (typing only), conventions, conformance tests
──────────────────────────────────────────────
qccompute  the runner: discovery, schema validation, resources and scheduling
           (threads, GPUs, memory), lists, batches, sessions, children, logs, errors
──────────────────────────────────────────────
programs   native AFH programs (linus, axtb, qcopt, ...) implement qcdata.Program;
           they depend on qcdata only
qccodec    adapters that make external programs (ORCA, Psi4, TeraChem, CREST, xtb,
           geomeTRIC, Sella, ML potentials) look like a qcdata.Program
```

The role of qccodec shrinks to what it was built for: external programs. Native programs
stop importing it (today linus, axtb and qcopt each ship a qccodec codec whose encode
packs a `PythonInput` that calls themselves and whose decode unpacks it again).

## 1. Inputs

`ProgramInput` stays: `program`, `calctype`, `structure`, `model`, `keywords`, `subprograms`,
`files`.

- **`keywords` is the program's native settings tree**, unchanged: the same keys and nesting
  as its JSON request file and its Python keyword arguments (the D39 rules agreed by linus,
  axtb and qcopt: `group.key`, at most two levels, one canonical spelling, `"auto"` means
  the program decides). qcdata already round-trips nested keywords; nothing to flatten.
- **Validation from the program's schema.** Every program publishes
  `settings_schema(calctype) -> dict` (JSON Schema, with `x-unit`, `x-tier`, `x-concern`,
  `x-applies-to`). qccompute validates keywords before scheduling, so a typo fails at once
  for a list of 1000 inputs, and tools get completion and docs from the schema.
- **`model` and `structure` map onto native keys**: `model.method`/`model.basis` to
  `method`/`basis`; `structure` gives positions (bohr), numbers, charge and multiplicity.
  The native tree may not repeat them.
- **Resource keys** (`threads`, `memory_gib`, `backend`, `device_memory_gib`) stay keywords
  (as agreed), and their schema entries carry `x-concern: "resource"`; qccompute reads them
  for scheduling. Memory and backend can change numbers at round-off (linus picks its DF
  storage from the memory budget; capped GPU runs may take FP32 paths), so **result
  identity is computed from `resolved.settings`**, where every such choice appears resolved,
  never from the raw resource keys.
- **One shared vocabulary** for settings that mean the same in several programs lives in
  qcdata's docs (`docs/conventions.md`), which is the single place for names and units
  (axtb's request 6). The starting list is what the three programs already agreed:
  `threads`, `memory_gib`, `backend`; `scf.{max_iter, e_tol, g_tol, guess}`;
  `solvent.{name, model, eps}`; `freq.{temperature, pressure, scale, imaginary_cutoff}`;
  `output.{orbitals, density, molden, bond_orders}`; `convergence.{max_iter, g_tol, ...}`
  for optimizers. `accuracy` is reserved for the cross-program absolute target. **[Colton]**
  signs off the names.
- **Calculation types.** `hessian` means a Hessian *and*, when the program can do it, the
  harmonic analysis and thermochemistry at the same geometry (linus and axtb already do;
  it costs nothing next to the Hessian; `freq.*` keywords tune or switch it off). No separate
  `frequencies` calctype: linus maps qcdata `hessian` to its own `frequencies`. Analysis of
  a *given* Hessian is a function, not a program run:
  `qcinf.vibrational_analysis(structure, hessian, ...)`, a reference implementation any
  program's Hessian (ORCA, an ML potential) can go through. **[Colton]**

## 2. Outputs

### 2.1 Slots every output has (`ProgramOutput`; `warnings` and `error` also on each `Frame`)

| field | type | meaning |
|---|---|---|
| `resolved` | `Resolved \| None` | `settings`: the complete settings tree with every `auto` replaced, itself a valid `keywords` that reproduces the run; `reasons`: path → why |
| `warnings` | `list[Warning]` | `code` (stable, snake_case), `message`, optional `path` (the settings key concerned) |
| `error` | `ErrorInfo \| None` | on failure: `kind` (below), `message`, optional `code` (stable snake_case within a kind, e.g. `max_iter`, `energy_rose`, `not_first_order_saddle`: callers branch on it instead of parsing text), optional `program` (the child that failed), optional `cause: ErrorInfo` (the child's own error, for kind `program`) |
| `record` | `dict` | (also on each `Frame`) what happened during the run (fallbacks, escalations, residuals reached, per-phase decisions); JSON, shaped by the program's published result schema; diagnostic, not stable across versions |
| `execution.timings` | `dict[str, float]` | seconds per phase (keys are the program's; `total` for the whole call, drivers also `children` and their own time), on `ExecutionInfo` beside `wall_time` |
| `results.provenance` | `Provenance` | gains `program_commit` (and keeps `program`, `program_version`) |

`ErrorKind` (`StrEnum`): `input`, `unsupported`, `convergence`, `numerical`, `resource`,
`backend`, `program` (a child calculation failed), `internal` (the agreed vocabulary), plus
two qccompute sets itself: `timeout` and `environment` (program not installed, licence,
the process died). Partial results stay attached to failed outputs, as today.

`extras` remains for what has no standard home yet. **Rule:** a concept that two programs
report goes into a typed field, not `extras`.

### 2.2 Single points (`SinglePointData`, new optional fields)

| field | type | meaning |
|---|---|---|
| `energy_components` | `dict[str, float]` | Eh; a true partition: they add up to `energy`; what a component contains is the program's, and a hierarchy of overlapping terms goes in `record`. Standard keys: `nuclear_repulsion`, `one_electron`, `coulomb`, `exchange`, `xc`, `dispersion`, `vv10`, `solvation_electrostatic`, `solvation_cds`, `repulsion`, `electrostatic`, `halogen_bond`, `hydrogen_bond`, ... (listed in conventions.md; programs may add their own keys) |
| `scf_iterations` | `int` | |
| `scf_quadrupole_moment` | 3 × 3 | e·bohr², about the coordinate origin, with `quadrupole_kind`: `traceless` (Buckingham ½Σq(3rᵢrⱼ − r²δᵢⱼ)) or `second_moment` (Σq rᵢrⱼ); programs convert other forms to one of these |
| `polarizability` | 3 × 3 | e² bohr² / Eh |
| `atomic_charges_kind` | `str` | `mulliken`, `lowdin`, `eeq`, `eeqbc`, `cm5`, `hirshfeld`, ... (charges are not comparable without it) |
| `bond_orders` | n × n | with `bond_orders_kind`: `mayer` (Σ(PS)μν(PS)νμ, what xtb prints as "Wiberg") or `wiberg` (orthogonalized basis) |
| `vibrational_analysis` | `VibrationalAnalysis` | linus's proposal (masses, projection, all modes with `external` flags, imaginary frequencies negative and stored without a cutoff, reduced masses, force constants, IR intensities, dipole derivatives, rotor type, moments, rotational constants, point group, symmetry number); `n_imaginary(cutoff)` is a method |
| `thermochemistry` | `list[Thermochemistry]` | linus's proposal (one per temperature/pressure: ZPE, per-term E/H/S/Cv/Cp/G, corrections, entropy/enthalpy model, scale factor, excluded imaginary modes) |
| `wavefunction.orbitals` | `Orbitals` | linus's proposal (`BasisSet` of `Shell`s with a named `AOConvention`, per-spin coefficients, energies, occupations); filled on request (`output.orbitals`) |

The old `freqs_wavenumber`, `normal_modes_cartesian` and `gibbs_free_energy` are
**removed** (Colton, 2026-11-11: pre-v1, a final unified API rather than compatibility
for the few pre-release users). Files saved before 0.24 still load when those fields are
empty. HOMO-LUMO gaps are a property computed from `wavefunction`, not a stored field.

### 2.3 Trajectories (phase 4)

`OptimizationData`/`PathData` keep their reading interface (frames), but serialize
columnar when frames share atoms, charge and multiplicity: symbols once, positions,
energies and gradients as arrays, provenance once. A 400-frame IRC drops from a full
`Structure` and `Provenance` per frame to a few arrays. Also: a standard slot for a
frequency analysis made with a different method than the frames (qcopt's final check).

### 2.4 Published schema

`qcdata.output_schema(calctype)` gives the JSON Schema of the outputs; each program's
`settings_schema(calctype)` gives the shape of `keywords` and `resolved.settings`, and its
`record_schema(calctype)` the shape of `record`. Viewers and
apps generate from these, as the programs generate their docs.

## 3. Running: the Program protocol

A program registers under the entry point group `qcdata.programs`:

```toml
[project.entry-points."qcdata.programs"]
linus = "linus.qc:program"
```

and the object implements (typing only, in `qcdata.program`):

```python
class Program(Protocol):
    name: str
    def version(self) -> str: ...
    def capabilities(self) -> Capabilities: ...
        # calctypes, methods (optional), sessions, batches, in_process (safe to call in the
        # runner's process), thread_safe (separate sessions may compute
        # concurrently from threads), gpu
    def settings_schema(self, calctype: CalcType) -> dict: ...
    def record_schema(self, calctype: CalcType) -> dict: ...
    def plan(self, inp: ProgramInput) -> Plan: ...
        # dry run: validates (raises InputError with did-you-mean), resolves every auto,
        # returns Plan(resolved=Resolved, resources=Resources(threads, memory_gib, gpus,
        # device_memory_gib, exclusive))
    def compute(self, inp: ProgramInput, ctx: RunContext) -> ProgramOutput: ...
        # a failed calculation returns success=False with error and partial results;
        # it raises only on bugs

    # optional
    def open_session(self, spec: ProgramSpec, structure: Structure, ctx: RunContext) -> Session: ...
    def compute_batch(self, inputs: list[ProgramInput], ctx: RunContext) -> list[ProgramOutput]: ...
```

`RunContext` is a small concrete class in qcdata (so `linus.compute(inp)` works without
qccompute), filled by qccompute when it runs the program:

- `threads`, `gpus`, `memory_gib`: what the scheduler granted (the program uses these, not
  the machine's count);
- `scratch`: a directory;
- `compute`: the child callback (one input or a list, `previous=`), qccodec #94's shape;
- `session(spec, structure, threads=None)`: open a child session (below) with a share
  of the grant (qcopt with 16 threads and 8 tasks gives each session 2);
- `progress(...)`: optional streaming updates.

qccompute finds programs in `qcdata.programs` first and then in `qccodec.codecs`; a
`Codec` is wrapped as a `Program` by an adapter in qccodec, so nothing external changes.
The native path skips encode/decode: qccompute calls `program.compute(inp, ctx)` in its own
process when `capabilities().in_process` is true, else in its worker processes.

### Sessions (qcopt's proposal, adopted)

"The same program, the same molecule, many geometries, warm-started" is what optimizers,
IRC, MD, NEB and finite differences all need. Today each program grows a qcopt-specific
evaluator (`qcopt`'s `AxtbEvaluator`, `linus.evaluator.QcoptEvaluator`) and qccompute has
`propagate_wfn`. Instead:

```python
class Session(Protocol):
    program: str
    program_version: str
    def compute(self, positions, calctype, *, accuracy=None, lineage=None, outputs=()) -> Frame: ...
    def compute_many(self, requests: list[Request]) -> list[Frame]: ...
        # Request(positions, calctype, accuracy, lineage, outputs): finite differences,
        # NEB images, IRC branches; a failed request is a failed Frame in place
    def reset(self, lineage=None) -> None: ...   # drop warm-start history
    def plan(self) -> Plan: ...   # pinned for the session's lifetime (linus D38)
    def close(self) -> None: ...  # also a context manager
```

- **Results are `Frame`s** (structure, `SinglePointData`, and the request's own
  `warnings` and `record`: an SCF escalation, a stability probe, a fallback), ready to store
  in a trajectory without building full `ProgramOutput`s.
- **`lineage`** keys the warm-start and extrapolation history, so NEB images, IRC branches
  and parallel tasks each continue from their own previous geometry (None: the nearest).
  A failed request leaves its lineage's history untouched; `reset` drops it after a jump.
- **`accuracy`** is an `Accuracy(energy, gradient, hessian)` in atomic units, or a bare
  number for the calctype's highest derivative; explicit session settings are floors.
- **`outputs`** are per-request extras that do not change the surface (`dipole`, `ir`,
  `polarizability`, `orbitals`, `density`); unsupported ones fail with kind `unsupported`.
- Atoms, charge, multiplicity, model and approximations are fixed for the session's
  lifetime; anything else is a new session.

Contract:

- **A failed request raises `ProgramError`** (`error: ErrorInfo`, partial `SinglePointData`
  if any) **and leaves the session usable**: an optimizer retries a smaller step on the
  same session, and one failed gradient does not end the other tasks of a round.
- **A `hessian` request returns energy and gradient too**, so drivers never pair a Hessian
  with a separate gradient call (the stateless fallback computes both when the program
  needs two runs).
- Frames get `program`/`program_version` from the session, not from every result.
- A driver's `Plan.resources` is the maximum over its children's plans (qccompute asks
  them); its own work is light.

- The session pins the model and approximations (method, basis, grids, aux basis,
  precision) so all its evaluations are on one surface; a request may vary only convergence
  (`accuracy`, linus D38). The session knows first geometry vs continuation; no flag.
- `qccompute.session(spec, structure)` (a context manager) opens one: natively when the
  program supports sessions, else a stateless fallback over `compute` with `propagate_wfn`.
- Inside a running program, `ctx.session(...)` opens child sessions under the parent's
  resource grant. qcopt then needs one generic evaluator, and the `qcopt.evaluators` entry
  point group is not needed.
- Session results are `SinglePointData`; programs may build them with `model_construct`
  in hot loops (trusted producer, no re-validation).
- Phase 4: session-aware batching across sessions (GPU lockstep of K sessions' requests).

### Resources and GPUs (phase 4)

`Plan.resources` drives scheduling: cores from `threads`; GPUs from `gpus` with
`device_memory_gib` so several processes can share one device (2.5-3.7x throughput measured
by linus on an H200), pinned by `CUDA_VISIBLE_DEVICES` (UUIDs), and `exclusive` for clean
timings.

## 4. Conformance

`qcdata.testing.check_program(program)` runs a fixed set of small inputs and checks the
contract: the schema validates its own defaults, `plan` is reproducible from
`resolved.settings`, unknown keys fail with kind `input`, units are atomic, standard
fields are filled where the program claims the capability, errors carry kinds,
`energy_components` add up. Each program runs it in CI.

## 5. Phases (each a PR, in order)

1. **qcdata 0.24: output slots and typed blocks.** `Resolved`, `Warning`, `ErrorInfo`/
   `ErrorKind`, `record`, `execution.timings`, `Provenance.program_commit`; the new
   `SinglePointData` fields; `VibrationalAnalysis`, `Thermochemistry`; `Orbitals`, `BasisSet`,
   `Shell`, `AOConvention` (ported from `linus.models` with its tests); conventions.md.
   Stacked on #128.
2. **The protocol.** `qcdata.program` (Program, Session, RunContext, Plan, Resources,
   Capabilities) and `qcdata.testing`; qccompute discovers `qcdata.programs`, validates by
   schema, runs native programs, opens sessions; qccodec wraps Codecs as Programs.
3. **Programs move.** linus, axtb and qcopt register `qcdata.programs`, fill the typed
   fields instead of `extras`, drop their qccodec codecs (kept one release as aliases);
   qcopt uses sessions and deletes its evaluators.
4. **Later.** Columnar trajectories, subprogram roles beyond calctype, GPU packing,
   session-aware batching, `qcinf.vibrational_analysis`.

## 6. Decisions (Colton, 2026-11-11, relayed by the linus session)

1. The protocol lives in qcdata. **Approved.**
2. One calctype `hessian` (with analysis); no `frequencies`. **Approved.**
3. No compatibility views: the old frequency fields are removed. **Decided.**
4. qccodec is for external programs only. **Approved.**
5. The shared names and glossary. **Approved.** Settings strictness (D41): settings that
   do not affect the calculation are errors; contingent settings are allowed and the
   record says whether they fired; sessions check against the calctypes they were opened
   for. See conventions.md rule 5.
