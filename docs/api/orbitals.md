# Basis sets and orbitals

`Wavefunction.orbitals` is the one home of a calculation's orbital levels: energies and
occupations per spin, the electronic temperature and Fermi levels, and the electrons each
atom keeps outside the orbitals (`core_electrons`). On request it also holds the
coefficients, so they can seed another program's SCF, be plotted, or be compared across
programs. `Wavefunction` adds the AO matrices (`overlap`, `hamiltonian_a/_b` and
`core_hamiltonian`) and `basis`, the basis set that the coefficients and every AO matrix
share, with the AO convention they follow and its effective core potentials (`Ecp`). The
design and its reasons are in [Wavefunctions](../design/wavefunctions.md).

::: qcdata.Wavefunction

::: qcdata.Orbitals

::: qcdata.OrbitalKind

::: qcdata.OrbitalSpin

::: qcdata.BasisSet

::: qcdata.Shell

::: qcdata.Ecp

::: qcdata.EcpTerm

::: qcdata.AOConvention

::: qcdata.SphericalOrder

::: qcdata.CartesianOrder

::: qcdata.CartesianNormalization
