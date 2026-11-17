# Wavefunctions in qcdata

What qcdata returns about a calculation's wavefunction, and why. This is **output** data.
Programs keep their internal structures and their warm-start state: the standardized
wavefunction is not a restart format (a tight-binding program warm-starts from its
multipole moments and densities, an SCF program from its own copy of the orbitals,
inside a session). Given orbitals (`ProgramInput.given`) may at most seed a guess.

## Principles

- One home for every number.
- Store what cannot be derived. The SCF density is C n Cᵀ of the returned orbitals
  (`Orbitals.density()`); AO integrals that any integral library recomputes from the basis
  wait for a consumer.
- A field arrives with a consumer and a definition two programs can test.
- Models always hold arrays; only saving decides between inline JSON and a file.
- TREXIO: the same physics and names where they agree; our own layout.

## Orbitals: the one home of the orbital levels

| field | meaning |
|---|---|
| `spin` | `restricted`: one coefficient matrix, one set of energies, one Hamiltonian, occupations per spin. `unrestricted`: everything per spin |
| `kind` | enum. `canonical`: eigenvectors of `hamiltonian`, energies required. `natural`: eigenvectors of a density, occupations, no energies. `localized`: no energies; `localization` names the method. Values are added when a program produces them |
| `energies_a/_b` | Eh; `_b` only for unrestricted orbitals |
| `occupations_a/_b` | both always, each between 0 and 1 |
| `electronic_temperature` | K; 0 for aufbau occupations |
| `fermi_level_a/_b` | Eh; None at 0 K, both given above it. Open shells are filled per spin at fixed n_α and n_β, so there are two |
| `core_electrons` | one per atom: the electrons the orbitals do not represent, ECP cores or the core a valence-only method leaves out (bookkeeping only; Zn's core is 28 in GFN2 and g-xTB). None: all-electron |
| `coefficients_a/_b` | optional, nao × nmo, in the AOs of `Wavefunction.basis`; only on request (output `orbitals`) |

- Tight binding's open shells (GFN2-xTB) are restricted orbitals of one spin-averaged
  Hamiltonian with per-spin occupations. They are not ROHF.
- Levels and coefficients are separate because their costs differ. Levels are free and
  wanted on every frame of an optimization; coefficients cost time (GFN2-xTB on
  (H₂O)₂₀₀: an energy and gradient take 3.4 s without them, 4.9 s with) and nao × nmo
  numbers per spin (caffeine in def2-SVP: 246 energies against 60,516 coefficients).
- `core_electrons` lives with the levels, not the basis set, so the electron count can be
  checked on every frame, coefficients or not. All-electron producers write nothing;
  tight-binding programs give their valence-only cores (no ECP involved), and on every
  ECP atom it equals the ECP's own `core_electrons`.
- Where the structure is known (a `ProgramOutput` or a `Frame`), `core_electrons` lists
  its atoms and the occupations add up to Σ(Z − core_electrons) − charge, to 1e-6
  electrons. This checks electron populations, not charges: g-xTB 2.0.1's reference
  occupations are not integers, so its charges for a neutral molecule sum to −7e-10.
  A given wavefunction (`ProgramInput.given`) may come from another charge or
  multiplicity, so only its atoms are checked.
- `density()` raises a clear error when coefficients are missing, as Molden export does.
- `Orbitals` holds no basis set. The basis set describes the AO space that the
  coefficients and every AO matrix share, so it belongs to their common parent,
  `Wavefunction.basis`. An overlap requested alone then ships no coefficients, which for
  unrestricted orbitals would be twice the overlap's own size.

## BasisSet

- `shells` and `ao_convention`. Tight-binding orbitals fit this layout exactly: GFN2-xTB
  and g-xTB orbitals are orthonormal to 4e-15–8e-15 in an independently computed overlap,
  and their spherical coefficients map onto their Cartesian ones to 1e-16.
- `ecp`: a list of `Ecp(center, core_electrons, local: list[EcpTerm],
  semilocal: list[list[EcpTerm]])`, at most one per atom. `semilocal[l]`, for l = 0..L−1, is the channel U_l − U_L, as
  Basis Set Exchange, NWChem, PySCF and Psi4 store it. `EcpTerm(power, exponent,
  coefficient)` is coefficient · r^power · exp(−exponent r²). `power` is the power of r
  (−2 or more): Basis Set Exchange JSON and NWChem write n for r^(n−2), so their
  converters subtract 2. Scalar ECPs only; spin-orbit terms are refused.
- An ECP carries the number of core electrons it replaces, as Basis Set Exchange, NWChem,
  PySCF and Psi4 keep it with the potential, so a basis set alone (one built for an
  input, or one in `given`) is a complete potential and converts to another program
  without the orbitals. `Orbitals.core_electrons` keeps, per atom, what the orbitals
  leave out, which is more than the ECPs for a valence-only method; a `Wavefunction`
  checks that the two agree on every ECP atom.
- A basis belongs to one geometry. GFN2's is the same per element at any geometry.
  g-xTB's q-vSZP contraction depends on each atom's EEQ-BC charge and coordination number
  at that geometry (fixed during the SCF), so the frames of a g-xTB trajectory do not
  share a `BasisSet`. g-xTB's scaled-exponent H0 basis is internal; the wavefunction's basis
  is the SCF basis.

## Wavefunction: the basis set and AO matrices, on request

`basis` is present whenever coefficients or an AO matrix are, and only then: it is the
AO space they share, in its `AOConvention`. Each AO matrix is requested by its output
name, the field's (`overlap`, `hamiltonian`, `core_hamiltonian`); the conventions list
what each implies. `overlap` and `core_hamiltonian` imply only the basis set;
`hamiltonian` implies `overlap` and `orbitals`.

| field | HF and DFT | tight binding |
|---|---|---|
| `overlap` | yes | yes: the program's own S, with its screening |
| `hamiltonian_a/_b` | Fock or Kohn-Sham matrix | tight-binding Hamiltonian |
| `core_hamiltonian` | yes | — (tight binding's H0 is model data) |

- A Wavefunction always has orbitals; GFN-FF and other force fields return none.
- A `hamiltonian` comes with `overlap` and the orbitals' coefficients, and H C = S C ε
  holds with the returned S only: a screened S (a tight-binding program skips distant
  primitive pairs) can differ from an exact one by up to 3.4e-9. Its orbitals are
  canonical, and restricted orbitals have one Hamiltonian.
- `core_hamiltonian` is every density-independent one-electron term of the run: T, V_ne
  with charges Z − core_electrons, V_ecp, and external point charges and fields once a
  program supports them. tr(P h), with P the density of the SCF's last input, is
  `energy_components`' `one_electron` term. With P = `Orbitals.density()`, the density a
  reader has, the two agree only to the SCF residual, as for the `hamiltonian`: compare
  them at SCF tolerance, never at round-off.
- Every matrix is nao × nao of `basis`, and the coefficients have one row per AO of it;
  an AO-indexed array without a basis set, or a basis set without one, is refused.
- `Wavefunction.with_convention` converts the coefficients and every matrix together, each
  with its variance: coefficients scale by s per AO index, densities by s_μ s_ν,
  operators (S, H, h) by 1/(s_μ s_ν). A spherical reordering is a permutation either
  way; a change of Cartesian normalization is not, so the coefficient transform does not
  apply to matrices. `BasisSet.transform` takes the array's kind (`coefficients`,
  `operator`, `density`) and never guesses it. Large arrays are converted as arrays, never
  through Python lists.

## The Hamiltonian

`hamiltonian` is the operator of **one build** at the last build's input x (a density, or
tight binding's charges and multipole moments), however that input was mixed. The
returned orbitals are its fresh generalized eigenvectors with the returned S, so
H C = S C ε holds to rounding (max |CᵀHC − diag ε| below 1e-13 for an exact build).

- Allowed: mixing inputs (Broyden or DIIS on moments, damping of densities).
- Not allowed: combining several builds' operators (DIIS on F or H), damping or level
  shifting the returned matrix, or returning H and C from different iterations.
- Requesting `hamiltonian` changes no other returned number.
- H is F[x], not F of C n Cᵀ: the returned orbitals' density is one step ahead, and the
  two differ by the SCF residual (1.2e-6–1.6e-6 for caffeine at default thresholds).

How SCF schemes meet the rule:

| scheme | what the program does |
|---|---|
| HF/DFT, F rebuilt from the last density and diagonalized | the final build is always complete, never incremental. Against an incremental final build this moves energies by about 1e-11 Eh and orbital energies by up to 3e-9, and costs about 6% of the time for CPU energies with exact J/K, within the timing noise |
| GFN2-xTB, mixers on input moments | nothing: H is one build |
| g-xTB 1.1, ending with one unmixed iteration | nothing |
| DIIS on the Fock matrix (g-xTB 2.0.1) | one unmixed iteration after convergence: one more iteration, 3–4% more time, and closer to convergence in all 14 cases measured (taxol's energy error against tight thresholds: 1.2e-7 → 2.6e-8 Eh); results move by up to 1e-7 Eh and 4e-5 Eh/bohr |
| a mode that must reproduce another program's numbers exactly, and so keeps its DIIS rule | refuses a `hamiltonian` request (`not_applicable`) |
| a composite (a solvation model on a base method) | returns its base part's wavefunction; `record` names it; H and P do not reproduce the composite energy |
| a force field | `wavefunction` is None |

What H contains:

- **DFT:** everything in the SCF operator: V_xc on the run's final grid, plus VV10's
  nonlocal potential; range-separated exchange; J/K as built (density-fitted when the
  program fits, COSX K on its final grid); the PCM operator. Nothing for D3/D4 or SMD's
  CDS term, which do not depend on the density.
- **Tight binding:** GFN2: H0 plus the potentials of the last input moments (shell,
  multipole, third order). g-xTB: the unmixed Fock matrix at the last input.

## Populations and multipoles (`SinglePointData`)

- `shell_populations`: electron populations Σ_{μ∈shell}(PS)_μμ, indexed like
  `Wavefunction.basis.shells`, with `shell_atoms` and `shell_angular_momenta` so they stand without a
  basis (and must match its shells when there is one). Not charges, which depend on
  method-specific reference occupations.
- `atomic_spin_populations`: n_α − n_β per atom, with `atomic_spin_populations_kind`
  naming the partition from the values of `atomic_charges_kind`.
- `atomic_dipoles`, `atomic_quadrupoles`: the electronic part only, about each atom
  (μ_A = −Σ_{μ∈A}(P D)_μμ, integrals about the column AO's atom); quadrupoles Buckingham
  traceless (3/2 M − 1/2 tr M), 3 × 3 per atom.
- `quadrupole_moment` is the Buckingham traceless tensor, with no kind beside it.
  tblite's molecular quadrupole, which the g-xTB 2.0.1 program prints, counts the
  off-diagonal charge and dipole parts twice and is not a tensor.

## Deferred until a consumer is named

| data | why it waits |
|---|---|
| `kinetic`, `potential_n_e`, `ecp_potential`, `dipole_integrals` | programs could fill them, but any integral library recomputes them from the basis and ECP in milliseconds, and each is a definition two programs must test |
| `solvent_potential` | H already contains the solvent operator and `record` names the model; for tight binding, removing it gives no gas-phase answer |
| stored densities | only for densities that are not C n Cᵀ of the returned orbitals (relaxed, response, or a mixed density returned without orbitals), with a kind (`scf`, `relaxed`, `unrelaxed`) |
| two-electron integrals, DF and Cholesky tensors | tight binding has none; a 3-index DF tensor is about 200 GB at 2000 functions. For CC or QMC: MO-basis Cholesky vectors of an active space, exported to TREXIO, on demand |
| tight-binding kernels and H0, DFT grids and response intermediates, GFN2's Slater exponents | specific to one model or one program, or not the functions the program computes with |

## Large arrays

- Models always hold arrays. No field is ever a file reference: that would make equality,
  hashing, pickling to workers and moving a result between hosts depend on files beside
  it. Lazy loading is the reader's business.
- Saving decides by float64 size: an array of more than 1 MiB (131,072 numbers; square
  matrices from nao ≥ 363) goes to a file beside the JSON, so the same result always saves
  the same way. Caffeine in def2-SVP stays inline; def2-TZVP does not.
- Each such array is its own `.npy` file beside the saved file, named by the file's stem
  and the field path (`output.results.wavefunction.orbitals.coefficients_a.npy`):
  little-endian float64, C order, uncompressed (compression saves about 4% on
  coefficients). The JSON holds a pointer in its place with the file's path, the shape,
  the dtype and the file's sha256.
- Inline floats round-trip exactly.
- JSON costs 21.8–23.2 bytes per float64, against 8 in binary. A 2000-function matrix is
  32 MB binary and 87–93 MB as JSON (1.2 s in `json.dumps`).

## TREXIO

The same physics and names; our own layout. A converter in qcinf, built on the `trexio`
package (an optional extra), writes real TREXIO files, tested by a round trip: orbitals →
TREXIO → PySCF energy at the same density.

| TREXIO | qcdata |
|---|---|
| `nucleus.*` | `Structure` |
| `basis.*` (flat per-primitive arrays; `shell_factor` × `prim_factor` × `coefficient`) | `Wavefunction.basis.shells` (per-atom `Shell`, one normalization convention) |
| `ecp.*` | `BasisSet.ecp` (local and semilocal channels; power of r; `z_core` as `Ecp.core_electrons`), `Orbitals.core_electrons` |
| `ao.cartesian`, `ao.normalization` | `Shell.pure`, `AOConvention` (`spherical_order="trexio"` selectable, never the default) |
| `ao_1e_int.overlap`, `core_hamiltonian` | `Wavefunction.overlap`, `core_hamiltonian` |
| `ao_1e_int.kinetic`, `potential_n_e`, `ecp`, dipole | deferred |
| `mo.coefficient` [mo, ao], `mo.spin` | `Orbitals.coefficients_a/_b` [ao, mo] |
| `mo.energy`, `mo.occupation`, `mo.type`, `mo.electronic_temperature` | `energies_*`, `occupations_*` (per spin, 0..1, where TREXIO keeps one occupation per orbital: the converter checks whether a restricted file's are totals, 0..2), `kind`, `electronic_temperature` |
| — | `fermi_level_a/_b` |
| `rdm.1e_up/_dn` (MO basis) | `Orbitals.density()` (AO basis, computed) |
| — (no Fock matrix) | `hamiltonian_a/_b` |
| `ao_2e_int`, `mo_2e_int`, `determinant`, `amplitude`, `jastrow`, `qmc` | not in qcdata |
