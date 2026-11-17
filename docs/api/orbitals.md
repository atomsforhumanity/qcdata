# Basis sets and orbitals

`Wavefunction.orbitals` is the one home of a calculation's orbital levels: energies and
occupations per spin, the electronic temperature and Fermi levels. On request it also holds
the coefficients with the basis set they expand in and the AO convention they follow, so
they can seed another program's SCF, be plotted, or be compared across programs. The
basis set carries the electrons each atom keeps outside the orbitals (`core_electrons`) and
its effective core potentials (`Ecp`). `Wavefunction` adds the AO matrices: `overlap`,
`hamiltonian_a/_b` and `core_hamiltonian`. The design and its reasons are in
[Wavefunctions](../design/wavefunctions.md).

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
