# Wavefunctions in qcdata

Status: **draft 2** (2026-11-17), proposed; Colton decides. It revises the 11-11 draft after
reviews by linus (HF/DFT, measured on caffeine B3LYP/def2-SVP and a 483-function anion) and
axtb (GFN2-xTB, g-xTB 1.1 and 2.0.1, GFN-FF, checked against its code with an independent
numpy overlap builder). It covers the **output** data qcdata returns. Programs keep their
internal structures and their warm-start state: the standardized wavefunction is not a
restart format (axtb warm-starts from moments and densities, linus from its own orbital
copies, inside sessions). Orbitals may at most seed a guess.

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
| `spin` | `restricted`: one coefficient matrix, one set of energies, one hamiltonian, occupations per spin. `unrestricted`: everything per spin |
| `kind` | enum. `canonical`: eigenvectors of `hamiltonian`, energies required. `natural`: eigenvectors of a density, occupations, no energies. `localized`: no energies; `localization` names the method. Values are added when a program produces them |
| `energies_a/_b` | Eh; `_b` only for unrestricted orbitals |
| `occupations_a/_b` | both always, each between 0 and 1 |
| `electronic_temperature` | K; 0 for aufbau occupations |
| `fermi_level_a/_b` | Eh; None at 0 K. Open shells are filled per spin at fixed n_α and n_β, so there are two |
| `basis`, `coefficients_a/_b` | optional, both or neither; only on request (`outputs: orbitals`) |

- GFN2 open shells are restricted orbitals of one spin-averaged Hamiltonian with per-spin
  occupations. They are not ROHF; the docstring changes.
- `Wavefunction.scf_eigenvalues_*` and `scf_occupations_*` go. Levels are free in both
  programs and wanted on every frame; coefficients cost time (axtb, GFN2 (H2O)200 E+G:
  3.4 s without, 4.9 s with) and nao × nmo numbers per spin (caffeine def2-SVP: 246
  energies against 60,516 coefficients).
- Validation where the structure is known: Σ occupations = Σ(Z − core_electrons) −
  charge. This checks electron populations, not charges: g-xTB 2.0.1's reference
  occupations are not integers, so its charges for a neutral molecule sum to −7e-10.
- `density()`, `with_convention()` and Molden export raise a clear error when coefficients
  are missing.

## BasisSet

- `shells` and `ao_convention`, as built. Both programs' shells fit exactly: axtb's
  orbitals are orthonormal in an independently computed overlap to 4e-15–8e-15, and its
  spherical coefficients map onto its Cartesian ones to 1e-16.
- `core_electrons`: one per atom, the electrons not represented by the orbitals: ECP cores
  (linus) or tight binding's valence-only counts (bookkeeping only; Zn's core is 28 in
  GFN2 and g-xTB). 0 otherwise.
- `ecp`: a list of `Ecp(center, local: list[EcpTerm], semilocal: list[list[EcpTerm]])`.
  `semilocal[l]`, for l = 0..L−1, is the channel U_l − U_L, as Basis Set Exchange, NWChem,
  PySCF, Psi4 and linus store it. `EcpTerm(power, exponent, coefficient)` is
  coefficient · r^power · exp(−exponent r²). `power` is the power of r: Basis Set Exchange
  JSON and NWChem write n for r^(n−2), so their converters subtract 2. Scalar ECPs only;
  spin-orbit terms are refused. `core_electrons` is not repeated inside `Ecp`.
- A basis belongs to one geometry. GFN2's is the same per element at any geometry.
  g-xTB's q-vSZP contraction depends on each atom's EEQ-BC charge and coordination number
  at that geometry (fixed during the SCF), so the frames of a g-xTB trajectory do not
  share a `BasisSet`. g-xTB's scaled-exponent H0 basis is internal; the orbitals' basis is
  the SCF basis.

## Wavefunction: AO matrices, on request

| field | linus | axtb |
|---|---|---|
| `overlap` | yes | yes: the program's own S, with its screening |
| `hamiltonian_a/_b` | Fock/Kohn-Sham matrix | tight-binding Hamiltonian |
| `core_hamiltonian` | yes | — (tight binding's H0 is model data) |

- A request for `hamiltonian` implies `overlap`, and H C = S C ε is tested against the
  returned S: axtb's screened S differs from an exact one by up to 3.4e-9.
- `core_hamiltonian` is every density-independent one-electron term of the run: T, V_ne
  with charges Z − core_electrons, V_ecp, and external point charges and fields once a
  program supports them. Then tr(P h) is `energy_components`' one-electron term.
- Every matrix is nao × nao of `orbitals.basis`; a matrix without a basis is refused.
- `Wavefunction.with_convention` converts the orbitals and every matrix together, each
  with its variance: coefficients scale by s per AO index, densities by s_μ s_ν,
  operators (S, H, h) by 1/(s_μ s_ν). A spherical reordering is a permutation either
  way; a change of Cartesian normalization is not. (The 11-11 draft said the coefficient
  transform applied to matrices; it does not.) Large arrays are not converted through
  Python lists.

## The Hamiltonian

`hamiltonian` is the operator of **one build** at the last build's input x (a density, or
tight binding's charges and multipole moments), however that input was mixed. The
returned orbitals are its fresh generalized eigenvectors with the returned S, so
H C = S C ε holds to rounding (linus: max |CᵀFC − diag ε| 6e-14–8e-14).

- Allowed: mixing inputs (Broyden or DIIS on moments, damping of densities).
- Not allowed: combining several builds' operators (DIIS on F or H), damping or level
  shifting the returned matrix, or returning H and C from different iterations.
- Requesting `hamiltonian` changes no other returned number.
- H is F[x], not F of C n Cᵀ: the returned orbitals' density is one step ahead, and the
  two differ by the SCF residual (linus: 1.2e-6–1.6e-6 on caffeine at default
  thresholds).

| program | today | needed |
|---|---|---|
| linus | F from the last density, freshly diagonalized | the final build always full. Today it is incremental for CPU energies with exact J/K; forcing it moves energies by about 1e-11 Eh and orbital energies by up to 3e-9, costs about 6% there (within noise), bumps linus's numerics epoch, and removes `jk.final_full_build` |
| GFN2 | both mixers mix input moments | nothing |
| g-xTB v1.1 | ends with one unmixed iteration (the program's rule) | nothing |
| g-xTB 2.0.1, axtb conventions | returns the DIIS-mixed Fock matrix | one unmixed iteration after convergence, as v1.1: +1 iteration, +3–4% time, closer to convergence in all 14 cases measured (taxol energy error against tight 1.2e-7 → 2.6e-8 Eh); results move by up to 1e-7 Eh and 4e-5 Eh/bohr |
| g-xTB 2.0.1, program conventions | the program's rule, for parity | a `hamiltonian` request is refused (`not_applicable`) |
| axtb's solvation composite | | returns its base part's wavefunction; `record` names it; H and P do not reproduce the composite energy |
| GFN-FF | | `wavefunction` is None |

What H contains:

- **DFT:** everything in the SCF operator: V_xc on the run's final grid, plus VV10's
  nonlocal potential; range-separated exchange; J/K as built (density-fitted when `jk` is
  `df`, COSX K on its final grid); the PCM operator. Nothing for D3/D4 or SMD's CDS term,
  which do not depend on the density.
- **Tight binding:** GFN2: H0 plus the potentials of the last input moments (shell,
  multipole, third order). g-xTB: the unmixed Fock matrix at the last input.

## Populations and multipoles (`SinglePointData`)

- `shell_populations`: electron populations Σ_{μ∈shell}(PS)_μμ, indexed like
  `basis.shells`, with `shell_atoms` and `shell_angular_momenta` so they stand without a
  basis. Not charges, which depend on method-specific reference occupations.
- `atomic_spin_populations`: with a kind, the partition of `atomic_charges_kind`.
- `atomic_dipoles`, `atomic_quadrupoles`: the electronic part only, about each atom
  (μ_A = −Σ_{μ∈A}(P D)_μμ, integrals about the column AO's atom); quadrupoles Buckingham
  traceless (3/2 M − 1/2 tr M), 3 × 3 per atom. axtb fills them for GFN2 and g-xTB 2.0.1.
- `scf_quadrupole_moment` names its form: the Buckingham traceless tensor. tblite's
  molecular quadrupole, which the g-xTB 2.0.1 program prints, counts the off-diagonal
  charge and dipole parts twice and is not a tensor.

## Deferred until a consumer is named

| data | why it waits |
|---|---|
| `kinetic`, `potential_n_e`, `ecp_potential`, `dipole_integrals` | linus and axtb could fill them, but any integral library recomputes them from the basis and ECP in milliseconds, and each is a definition two programs must test |
| `solvent_potential` | H already contains the solvent operator and `record` names the model; for tight binding, removing it gives no gas-phase answer |
| stored densities | only for densities that are not C n Cᵀ of the returned orbitals (relaxed, response, or a mixed density returned without orbitals), with a kind (`scf`, `relaxed`, `unrelaxed`) |
| two-electron integrals, DF and Cholesky tensors | tight binding has none; linus's 3-index tensor is about 200 GB at 2000 functions. For CC or QMC: MO-basis Cholesky vectors of an active space, exported to TREXIO, on demand |
| tight-binding kernels and H0, DFT grids and response intermediates, GFN2's Slater exponents | model- or engine-specific, or not the functions the program computes with |

## Large arrays

- Models always hold arrays. No field is ever a file reference: that would make equality,
  hashing, pickling to workers and moving a result between hosts depend on files beside
  it. Lazy loading is the reader's business.
- Saving decides by float64 size: an array of more than 1 MiB (131,072 numbers; square
  matrices from nao ≥ 363) goes to a file beside the JSON, so the same result always saves
  the same way. Caffeine def2-SVP stays inline; def2-TZVP does not.
- The file is one HDF5 file per saved output; dataset path = field path; little-endian
  float64, C order, uncompressed (compression saves about 4% on coefficients). The JSON
  records shape, dtype and sha256.
- Inline floats round-trip exactly.
- Measured: JSON costs 21.8–23.2 bytes per float64, against 8 in binary. A 2000-function
  matrix is 32 MB binary and 87–93 MB as JSON (1.2 s in `json.dumps`).
- In qcstore, this file is one of a run's blobs.

## TREXIO

The same physics and names; our own layout. A converter in qcinf, built on the `trexio`
package (an optional extra), writes real TREXIO files, tested by a round trip: linus
orbitals → TREXIO → PySCF energy at the same density.

| TREXIO | qcdata |
|---|---|
| `nucleus.*` | `Structure` |
| `basis.*` (flat per-primitive arrays; `shell_factor` × `prim_factor` × `coefficient`) | `BasisSet.shells` (per-atom `Shell`, one normalization convention) |
| `ecp.*` | `BasisSet.core_electrons`, `BasisSet.ecp` (local and semilocal channels; power of r) |
| `ao.cartesian`, `ao.normalization` | `Shell.pure`, `AOConvention` (`spherical_order="trexio"` selectable, never the default) |
| `ao_1e_int.overlap`, `core_hamiltonian` | `Wavefunction.overlap`, `core_hamiltonian` |
| `ao_1e_int.kinetic`, `potential_n_e`, `ecp`, dipole | deferred |
| `mo.coefficient` [mo, ao], `mo.spin` | `Orbitals.coefficients_a/_b` [ao, mo] |
| `mo.energy`, `mo.occupation`, `mo.type`, `mo.electronic_temperature` | `energies_*`, `occupations_*` (per spin, 0..1; TREXIO's restricted occupation is believed to be the total, 0..2, which the converter checks), `kind`, `electronic_temperature` |
| — | `fermi_level_a/_b` |
| `rdm.1e_up/_dn` (MO basis) | `Orbitals.density()` (AO basis, computed) |
| — (no Fock matrix) | `hamiltonian_a/_b` |
| `ao_2e_int`, `mo_2e_int`, `determinant`, `amplitude`, `jastrow`, `qmc` | not in qcdata |

## What changes if this is adopted

- **qcdata:** `Orbitals` and `Wavefunction` as above; `scf_eigenvalues_*` and
  `scf_occupations_*` removed; `Wavefunction.with_convention` with variance; validators
  (matrix shapes against the basis, occupation sums; Cᵀ S C = 1 in tests); the save format;
  axtb's check scripts (independent Molden overlap, spherical against Cartesian,
  Mulliken) as test fixtures.
- **linus:** drop `output.density`; final build always full (epoch bump;
  `jk.final_full_build` removed); emit `overlap` and `core_hamiltonian` on request; ECP
  terms in the new layout.
- **axtb:** g-xTB 2.0.1 in axtb conventions ends with one unmixed iteration (golden values
  move); program conventions refuse `hamiltonian`; Fermi levels per spin; atomic
  multipoles 3 × 3; GFN2 computes eigenvectors once after convergence when orbitals are
  requested.
