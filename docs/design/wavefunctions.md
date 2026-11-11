# Wavefunctions and integrals in qcdata: what is common, what is not

Status: **draft** (2026-11-11), from the qc* session after answers from linus (HF/DFT) and
axtb (GFN2-xTB, g-xTB), with TREXIO 2.x as the reference format. This concerns only the
**output** data qcdata returns to users. Programs keep their own internal, performance-
oriented structures. The definitions were agreed with linus and axtb; Colton decides whether to add the fields.

## Summary

Full quantum chemistry (linus) and semi-empirical tight binding (axtb) share more at the
output level than one might expect. Both expand their orbitals in **contracted Gaussian
shells centred on atoms**:

- **linus:** def2 and other basis sets, general contractions split into shells.
- **GFN2-xTB:** its Slater functions are evaluated as STO-nG Gaussian expansions, and those
  expansions are what every computed quantity uses.
- **g-xTB:** q-vSZP shells whose contraction depends on each atom's charge and
  coordination number. At a converged geometry they are a fixed per-atom contraction, and
  qcdata's shells are per atom anyway.

So one `Orbitals` model (basis, AO convention, per-spin coefficients, energies,
occupations) holds both exactly. One viewer or analysis (orbital plots, densities on a
grid, populations, Molden or TREXIO export) then works for either program. Both programs
already write Molden files that agree with qcdata's `MOLDEN_CONVENTION`.

What is **not** common is everything model-specific under the orbitals:

- two-electron integrals, which tight binding does not have and which are far too large
  for JSON in DFT;
- tight binding's charge-dependent kernels (gamma, multipoles, third order, spin
  constants) and H0 parameters;
- DFT's grids, XC kernels, density-fitting tensors and response intermediates.

These stay inside the programs, or in `record`.

## Common model (proposed additions to phase 1)

| where | field | linus | axtb | notes |
|---|---|---|---|---|
| `BasisSet` | `shells` (exists) | yes | yes (STO-nG; converged q-vSZP) | Gaussian only; no Slater exponents (axtb: they would describe slightly different functions) |
| `BasisSet` | `core_electrons` per atom | ECP cores (Rb-Rn, def2) | valence-only counts | defined as "electrons not represented by the orbitals", so Z − core is what the density integrates against; TREXIO `ecp.z_core` |
| `BasisSet` | `ecp` per atom: `lmax` (local channel) and terms (l, power, exponent, coefficient) | yes | — | full ECP so exported orbitals reproduce energies elsewhere; TREXIO `ecp` group |
| `Orbitals` | coefficients, energies, occupations per spin (exist) | RHF/RKS, UHF/UKS | GFN2 open shells: restricted with alpha/beta occupations; g-xTB: unrestricted | fractional occupations allowed |
| `Orbitals` | `electronic_temperature` (K), `fermi_level` (Eh) | — | when smearing is on | TREXIO `mo.electronic_temperature` |
| `Wavefunction` | `overlap` S | yes | yes (non-orthogonal basis) | |
| `Wavefunction` | `density_a/_b` | yes | yes | P = C n Cᵀ from the returned orbitals |
| `Wavefunction` | `hamiltonian_a/_b` | Fock / Kohn-Sham matrix (the last diagonalized, usually DIIS-extrapolated) | converged TB Hamiltonian (the last diagonalized) | agreed by linus and axtb: the matrix whose generalized eigenvectors with S are the returned orbitals. H C = S C ε holds exactly when the final step was a diagonalization; after an SOSCF or stability rotation the program semicanonicalizes, and it holds up to the SCF threshold. `record` names the final step (`diagonalization`, `diis_diagonalization`, `soscf_semicanonical`) and the build; H is never promised to be dE/dP |
| `Wavefunction` | `kinetic`, `potential_n_e`, `ecp_potential`, `core_hamiltonian` (= T + V_ne + V_ecp) | yes | — | TREXIO's `ao_1e_int` split; ab initio only (TB's H0 is model data, not a core Hamiltonian) |
| `Wavefunction` | `solvent_potential` | PCM reaction-field operator | — | so it can be removed from the Fock matrix |
| `Wavefunction` | `dipole_integrals` (3 × nao × nao) | yes, about the coordinate origin | — | |
| `SinglePointData` | `shell_populations` + `shell_atoms` | Mulliken over shells | shell-resolved Mulliken | |
| `SinglePointData` | `atomic_spin_populations` | yes | g-xTB magnetizations | |
| `SinglePointData` | `atomic_dipoles`, `atomic_quadrupoles` (+ kind) | via multipole analysis | yes (about each atom; traceless) | |

Every AO-indexed matrix is in the basis and `AOConvention` of `orbitals.basis`, and
`Orbitals.with_convention` transforms it with the coefficients. AO matrices are filled
only on request (`output.*`), since each is nao² numbers.

Definitions to write into `conventions.md`:

- **The Fock/Kohn-Sham matrix of a DFT run** includes everything in its SCF operator:
  - V_xc on that run's grid, so it reproduces the energy only on that grid;
  - long-range exchange for range-separated hybrids;
  - density-fitted J/K when `jk` is `df`;
  - the PCM operator, which is also given alone as `solvent_potential`.
- **A tight-binding Hamiltonian** is not a Fock matrix of any two-electron integrals. In
  GFN2 it includes the last iteration's charge-dependent shell and multipole potentials;
  g-xTB 2.0.1's is a DIIS extrapolation; v1.1's is non-variational in program conventions.
- **Densities and Mulliken populations of a minimal valence basis** are comparable within a
  method family, not with DFT's. Compare across methods through observables: densities
  on a grid, dipoles, and charges with their kind.

## Not standardized, and why

| data | why not |
|---|---|
| Two-electron integrals (AO or MO), DF/Cholesky tensors | tight binding has none; linus is density-fitted, and its 3-index tensor is naux·N² (about 200 GB at 2000 functions). If a CC or QMC code needs them, the right object is MO-basis Cholesky vectors for an active space, exported to TREXIO. Post-v1, on demand. |
| TB model internals: gamma and multipole kernels, third-order terms, spin constants, H0, q-vSZP's charge dependence | model-specific; meaningless outside that program's model |
| DFT internals: grids, XC kernels, response intermediates (perturbed densities) | engine-specific; dipole derivatives and polarizabilities, their useful outputs, already have homes |
| Slater exponents of GFN2 | not the functions the program computes with; `record` if wanted |

## Large data

JSON lists are fine for small systems and impossible for large ones: one 2000-function
matrix is 32 MB of JSON. Agreed with linus:

- Arrays above a threshold (about 1 MB each) go to a **binary sidecar** referenced from
  `results.files`, never to base64.
- The sidecar is one HDF5 file per result. Wavefunction-class data (basis, ECP, orbitals,
  AO one-electron matrices, densities) use **TREXIO's layout**, so QMC and CC codes read it
  directly; other large arrays (big Hessians) are plain HDF5 datasets.
- Each file entry records format, dataset names, shapes, dtype and a hash. This needs a
  small `FileRef` model beside today's `files: dict[str, str | bytes]`.
- Converters (Molden, PySCF, TREXIO, cube files from orbitals) live in qcinf.

## TREXIO correspondence

| TREXIO | qcdata |
|---|---|
| `nucleus.*` | `Structure` |
| `basis.*` (flat per-primitive arrays, `shell_factor` × `prim_factor` × `coefficient`) | `BasisSet.shells` (per-atom `Shell`, one normalization convention) |
| `ecp.*` | `BasisSet.core_electrons`, `BasisSet.ecp` |
| `ao.cartesian` / `cartesian_shell`, `ao.normalization` | `Shell.pure`, `AOConvention` (`spherical_order="trexio"` is TREXIO's order) |
| `ao_1e_int.overlap/kinetic/potential_n_e/ecp/core_hamiltonian/dipole_*` | `Wavefunction.overlap`, `kinetic`, `potential_n_e`, `ecp_potential`, `core_hamiltonian`, `dipole_integrals` |
| `mo.coefficient` [mo, ao], `mo.spin` | `Orbitals.coefficients_a/_b` [ao, mo] |
| `mo.energy`, `mo.occupation`, `mo.type`, `mo.electronic_temperature` | `energies_*`, `occupations_*`, `kind`, `electronic_temperature` |
| `rdm.1e_up/_dn` (MO basis) | `density_a/_b` (AO basis) |
| — (no Fock in TREXIO) | `hamiltonian_a/_b` |
| `ao_2e_int`, `mo_2e_int`, `determinant`, `amplitude`, `jastrow`, `qmc` | not in qcdata (see above) |
