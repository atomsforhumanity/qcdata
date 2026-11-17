"""Basis sets and molecular orbitals.

`Orbitals` is the one home of a calculation's orbital levels: energies and occupations
per spin, the electronic temperature and Fermi levels, the electrons each atom keeps
outside the orbitals (`core_electrons`) and, on request, the MO coefficients (nao x nmo,
one orbital per column) with the `BasisSet` they expand in. A `BasisSet` is a list of
`Shell` objects placed on the structure's atoms, in AO order; its effective core
potentials (`Ecp`); and an explicit `AOConvention` saying how the functions within a
shell are ordered and normalized, so coefficients from any program can be stored as
produced and converted when needed (`Orbitals.with_convention`,
`Wavefunction.with_convention`).

Coefficients of a shell multiply the *unnormalized* primitives r^l exp(-a r^2) and
include all normalization: the x^l Cartesian component of the contracted shell, and so
every spherical function, has unit norm. An AO is then
``sum_i coefficients[i] * x^lx y^ly z^lz exp(-exponents[i] r^2)`` (times the Cartesian
to spherical transformation of unit-normalized real solid harmonics for pure shells),
with no convention left to guess except the ones `AOConvention` names. General
contractions are split into one shell per contracted function.

Converters to PySCF, Molden and TREXIO live outside qcdata.
"""

# l is the conventional symbol for angular momentum.
# ruff: noqa: E741
from __future__ import annotations

import math
from collections.abc import Sequence
from enum import StrEnum
from typing import Any, Literal

import numpy as np
from pydantic import Field, field_validator, model_validator

from qcdata.helper_types import SerializableMatrix, SerializableVector

from .base_models import QCDataBaseModel

__all__ = [
    "AOConvention",
    "BasisSet",
    "CartesianNormalization",
    "CartesianOrder",
    "Ecp",
    "EcpTerm",
    "OrbitalKind",
    "OrbitalSpin",
    "Orbitals",
    "Shell",
    "SphericalOrder",
    "DEFAULT_AO_CONVENTION",
    "PYSCF_CONVENTION",
    "MOLDEN_CONVENTION",
]

L_LETTERS = "spdfghiklmn"

#: How an AO-indexed array changes with the AO convention: ``coefficients`` (the first
#: index is an AO: MO coefficients), ``operator`` (both indices are AOs of a matrix of
#: integrals <mu|O|nu>: overlap, Hamiltonians) or ``density`` (both indices are AOs of a
#: matrix that contracts with operators: C n C^T).
AOArrayKind = Literal["coefficients", "operator", "density"]


class SphericalOrder(StrEnum):
    """Order of the 2l+1 real solid harmonics of a pure shell.

    Attributes:
        pyscf: m = -l..l, except p shells, which are (x, y, z) (libcint, PySCF, linus).
        increasing_m: m = -l..l for every l, so p is (y, z, x) (libint's "standard", CCA,
            order).
        molden: m = 0, +1, -1, +2, -2, ..., with p as (x, y, z) (Molden, Gaussian, ORCA's
            order; ORCA also flips the sign of some f and g functions, not covered here).
        trexio: m = 0, +1, -1, +2, -2, ... for every l, so p is (z, x, y) (TREXIO).
    """

    pyscf = "pyscf"
    increasing_m = "increasing_m"
    molden = "molden"
    trexio = "trexio"


class CartesianOrder(StrEnum):
    """Order of the (l+1)(l+2)/2 components of a Cartesian shell.

    Attributes:
        lexicographic: xx, xy, xz, yy, yz, zz (lx descending, then ly descending; libint,
            libcint, PySCF, linus, CCA, TREXIO).
        molden: xx, yy, zz, xy, xz, yz; xxx, yyy, zzz, xyy, xxy, xxz, xzz, yzz, yyz, xyz; and
            Molden's g order (up to l = 4).
    """

    lexicographic = "lexicographic"
    molden = "molden"


class CartesianNormalization(StrEnum):
    """Norm of the components of a Cartesian shell (pure functions always have unit norm).

    Attributes:
        axis: every component carries the normalization of x^l: x^l, y^l, z^l have unit
            norm, xy has norm 1/sqrt(3) (libint, linus).
        unit: every component has unit norm (Molden).
        pyscf: as `axis` times sqrt(4 pi / (2l + 1)) for l >= 2 (PySCF's ``cart=True``).
    """

    axis = "axis"
    unit = "unit"
    pyscf = "pyscf"


class AOConvention(QCDataBaseModel):
    """How the functions of each shell are ordered and normalized.

    Shells are always in the order of `BasisSet.shells`; this says which functions a shell
    contributes and in what order. The default is PySCF's ordering with x^l-normalized
    Cartesian components (libcint, linus). Real solid harmonics are the usual unit-normalized ones without
    a Condon-Shortley phase (as PySCF, libint and Molden).

    Attributes:
        spherical_order: Order of the functions of pure shells.
        cartesian_order: Order of the components of Cartesian shells.
        cartesian_normalization: Norm of the components of Cartesian shells.
    """

    spherical_order: SphericalOrder = SphericalOrder.pyscf
    cartesian_order: CartesianOrder = CartesianOrder.lexicographic
    cartesian_normalization: CartesianNormalization = CartesianNormalization.axis


#: The default convention: PySCF's order with x^l-normalized Cartesian components
#: (libcint, PySCF with ``cart=False``, linus).
DEFAULT_AO_CONVENTION = AOConvention()
#: PySCF's AO convention for spherical and Cartesian molecules.
PYSCF_CONVENTION = AOConvention(cartesian_normalization=CartesianNormalization.pyscf)
#: Molden files.
MOLDEN_CONVENTION = AOConvention(
    spherical_order=SphericalOrder.molden,
    cartesian_order=CartesianOrder.molden,
    cartesian_normalization=CartesianNormalization.unit,
)


class Shell(QCDataBaseModel):
    """One contracted shell on an atom.

    Attributes:
        center: Index of the atom (0-based) in the structure the basis set belongs to.
        angular_momentum: l (0 for s, 1 for p, ...).
        pure: Spherical (2l + 1 functions) or Cartesian ((l+1)(l+2)/2 functions). s and p
            shells are the same either way except for the order of p (see `AOConvention`).
        exponents: Primitive exponents (bohr^-2).
        coefficients: Contraction coefficients of the unnormalized primitives
            r^l exp(-a r^2), normalization included: the x^l component of the shell (and
            every spherical function) has unit norm.
    """

    center: int
    angular_momentum: int
    pure: bool
    exponents: SerializableVector
    coefficients: SerializableVector

    @field_validator("exponents", "coefficients")
    @classmethod
    def _one_dimensional(cls, v: np.ndarray) -> np.ndarray:
        v = np.asarray(v, dtype=float)
        if v.ndim != 1 or v.size == 0:
            raise ValueError(
                "exponents and coefficients must be non-empty lists of numbers"
            )
        return v

    @model_validator(mode="after")
    def _check(self) -> Shell:
        if self.center < 0:
            raise ValueError("center must be an atom index >= 0")
        if self.angular_momentum < 0:
            raise ValueError("angular_momentum must be >= 0")
        if self.exponents.shape != self.coefficients.shape:
            raise ValueError("exponents and coefficients must have the same length")
        if np.any(self.exponents <= 0):
            raise ValueError("exponents must be positive")
        return self

    @property
    def nfunc(self) -> int:
        """Number of functions of the shell."""
        l = self.angular_momentum
        return 2 * l + 1 if self.pure else (l + 1) * (l + 2) // 2

    def primitive_coefficients(self) -> np.ndarray:
        """Coefficients of x^l-normalized primitives.

        This is the convention of basis set files, Molden and most programs' input; the
        contraction stays normalized.

        Returns:
            One coefficient per primitive.
        """
        return self.coefficients / _xl_norm(self.angular_momentum, self.exponents)


class EcpTerm(QCDataBaseModel):
    """One Gaussian term of an ECP channel: coefficient * r^power * exp(-exponent r^2).

    Attributes:
        power: The power of r, -2 or more. Basis Set Exchange JSON and NWChem write n
            for r^(n-2), so their power is n - 2.
        exponent: Gaussian exponent (bohr^-2), positive.
        coefficient: Coefficient (Eh bohr^-power).
    """

    power: int
    exponent: float
    coefficient: float

    @model_validator(mode="after")
    def _check(self) -> EcpTerm:
        if self.power < -2:
            raise ValueError(
                f"power must be -2 or more: r^{self.power} has no finite integrals"
            )
        if not self.exponent > 0:
            raise ValueError("exponent must be positive")
        return self


class Ecp(QCDataBaseModel):
    """The scalar effective core potential of one atom.

    The potential is U_L(r) + sum_{l<L} (U_l(r) - U_L(r)) P_l, with P_l the projector on
    angular momentum l about the atom and L = ``len(semilocal)``: the layout of Basis Set
    Exchange, NWChem, PySCF and Psi4. Spin-orbit terms have no place here. The electrons
    it replaces are its atom's `Orbitals.core_electrons`.

    Attributes:
        center: Index of the atom (0-based) in the structure the basis set belongs to.
        local: The terms of U_L, felt by every angular momentum.
        semilocal: ``semilocal[l]``, for l = 0..L-1: the terms of U_l - U_L.
    """

    center: int
    local: list[EcpTerm]
    semilocal: list[list[EcpTerm]]


class BasisSet(QCDataBaseModel):
    """A basis set placed on the atoms of a structure, in AO order.

    A basis set belongs to one geometry: a tight-binding basis whose contraction depends
    on the geometry (g-xTB's q-vSZP) differs from frame to frame.

    Attributes:
        name: The basis set's name (e.g. ``"def2-svp"``), if it has one.
        shells: The contracted shells; the AOs are their functions in this order.
        ecp: The effective core potentials, at most one per atom; the electrons each
            replaces are `Orbitals.core_electrons`.
        ao_convention: Order and normalization of the functions within each shell.
        nao: `@property` The number of AOs.
    """

    name: str | None = None
    shells: list[Shell]
    ecp: list[Ecp] = []
    ao_convention: AOConvention = Field(default_factory=AOConvention)

    @model_validator(mode="after")
    def _check(self) -> BasisSet:
        centers = [e.center for e in self.ecp]
        if any(c < 0 for c in centers):
            raise ValueError("Ecp centers must be atom indices >= 0")
        if len(set(centers)) != len(centers):
            raise ValueError("at most one Ecp per atom")
        return self

    @property
    def nao(self) -> int:
        """Number of atomic orbitals (basis functions)."""
        return sum(s.nfunc for s in self.shells)

    def shell_offsets(self) -> list[int]:
        """Index of the first AO of each shell."""
        out, n = [], 0
        for s in self.shells:
            out.append(n)
            n += s.nfunc
        return out

    def ao_labels(self, symbols: Sequence[str] | None = None) -> list[str]:
        """One label per AO, e.g. ``"0 O 2p x"`` or ``"1 H 1s"``.

        Spherical functions are labelled ``d-2 .. d+2``, Cartesian ones ``dxy``; shells
        are numbered by l per atom.

        Args:
            symbols: Element symbols of the structure's atoms, to include in the labels.

        Returns:
            One label per AO, in AO order.
        """
        labels: list[str] = []
        count: dict[tuple[int, int], int] = {}
        for s in self.shells:
            l = s.angular_momentum
            n = count[(s.center, l)] = count.get((s.center, l), 0) + 1
            atom = (
                f"{s.center} {symbols[s.center]}"
                if symbols is not None
                else f"{s.center}"
            )
            for key in _function_keys(s, self.ao_convention):
                labels.append(f"{atom} {n + l}{_function_name(l, key)}")
        return labels

    def transform(
        self,
        array: Any,
        convention: AOConvention,
        *,
        kind: AOArrayKind,
        shell_order: Sequence[int] | None = None,
    ) -> np.ndarray:
        """Convert an AO-indexed array to another AO convention and shell order.

        A target AO is a source AO divided by s, the ratio of the function's norm in
        this convention to its norm in the target (1 for every spherical function and
        for a reordering alone). Coefficients therefore scale by s per AO index,
        densities by s_mu s_nu and operators by 1 / (s_mu s_nu); a reordering permutes
        the AO indices of all three alike.

        Args:
            array: The array in this basis set's convention.
            convention: The target convention.
            kind: How the array transforms: ``coefficients`` (its first index is an AO,
                e.g. MO coefficients, nao x nmo), ``operator`` (an nao x nao matrix of
                integrals <mu|O|nu>: overlap, Hamiltonians) or ``density`` (an
                nao x nao matrix that contracts with operators, e.g. C n C^T).
            shell_order: The shells' order in the target, as indices into `shells`;
                None keeps the order.

        Returns:
            A new array in the target convention.

        Raises:
            ValueError: If `kind` is unknown or the array does not match the basis set's
                AOs.
        """
        if kind not in ("coefficients", "operator", "density"):
            raise ValueError(
                f"kind must be coefficients, operator or density, not {kind!r}"
            )
        a = np.asarray(array, dtype=float)
        nao = self.nao
        perm, scale = _ao_map(self, convention, shell_order)
        if kind == "coefficients":
            if a.ndim == 0 or a.shape[0] != nao:
                raise ValueError(f"expected {nao} rows (AOs), got shape {a.shape}")
            return a[perm] * scale.reshape((-1,) + (1,) * (a.ndim - 1))
        if a.shape != (nao, nao):
            raise ValueError(f"expected an {nao} x {nao} matrix, got shape {a.shape}")
        moved = a[np.ix_(perm, perm)]
        pair = np.outer(scale, scale)
        return moved / pair if kind == "operator" else moved * pair


class OrbitalSpin(StrEnum):
    """Whether alpha and beta electrons share spatial orbitals.

    Attributes:
        restricted: One set of spatial orbitals for both spins, occupied per spin: RHF
            and RKS; ROHF; and tight binding's open shells (GFN2-xTB), the eigenvectors
            of one spin-averaged Hamiltonian with per-spin occupations, which are not
            ROHF.
        unrestricted: Separate alpha and beta orbitals (UHF, UKS).
    """

    restricted = "restricted"
    unrestricted = "unrestricted"


class OrbitalKind(StrEnum):
    """What a set of orbitals is.

    Values are added when a program produces them.

    Attributes:
        canonical: The eigenvectors of the Hamiltonian, with their energies.
        natural: The eigenvectors of a density, with their occupations; no energies.
        localized: Localized orbitals, with no energies; `Orbitals.localization` names
            the method.
    """

    canonical = "canonical"
    natural = "natural"
    localized = "localized"


class Orbitals(QCDataBaseModel):
    """The orbital levels of a calculation and, on request, the orbitals themselves.

    The one home of orbital energies and occupations. The levels are cheap and come with
    every wavefunction; the coefficients (nao x nmo numbers per spin) and the basis set
    they expand in come only on request. Every per-orbital vector has one value per
    orbital, in the order of the coefficient matrices' columns.

    Where the structure is known (a `ProgramOutput` or a `Frame`), the occupations add
    up to sum(Z - core_electrons) - charge.

    Attributes:
        spin: ``restricted``: one set of spatial orbitals (one coefficient matrix, one
            set of energies, one Hamiltonian), occupied per spin. ``unrestricted``:
            everything per spin.
        kind: What the orbitals are: ``canonical`` (with energies), ``natural`` or
            ``localized`` (without).
        localization: The method that localized ``localized`` orbitals (e.g.
            ``"boys"``, ``"pipek_mezey"``); None for every other kind.
        energies_a: Orbital energies (Eh) of canonical orbitals: alpha, or both spins.
        energies_b: Beta orbital energies (Eh) of unrestricted canonical orbitals.
        occupations_a: Alpha electrons in each orbital, between 0 and 1.
        occupations_b: Beta electrons in each orbital, between 0 and 1; for restricted
            orbitals of an open shell they differ from the alpha ones.
        electronic_temperature: The temperature (K) of the Fermi smearing that set the
            occupations; 0 for aufbau occupations.
        fermi_level_a: The alpha Fermi level (Eh) of smeared occupations; None at 0 K.
        fermi_level_b: The beta Fermi level (Eh); None at 0 K. Each spin is filled at its
            own fixed electron count, so an open shell has two levels; a closed shell's
            are equal.
        core_electrons: One per atom of the structure: the electrons the orbitals do not
            represent, those an ECP replaces (required for every ECP atom of `basis`) or
            a valence-only method leaves out (28 for Zn in GFN2-xTB and g-xTB). None:
            every electron is in the orbitals.
        basis: The basis set the coefficients expand in (its centers refer to the
            structure's atoms); only with coefficients.
        coefficients_a: nao x nmo; column i is orbital i (alpha, or both spins). Only
            on request, with `basis`.
        coefficients_b: The beta orbitals, nao x nmo: unrestricted orbitals with
            coefficients only.
        nmo: `@property` The number of orbitals.
    """

    spin: OrbitalSpin
    kind: OrbitalKind = OrbitalKind.canonical
    localization: str | None = None
    energies_a: SerializableVector | None = None
    energies_b: SerializableVector | None = None
    occupations_a: SerializableVector
    occupations_b: SerializableVector
    electronic_temperature: float = 0.0
    fermi_level_a: float | None = None
    fermi_level_b: float | None = None
    core_electrons: list[int] | None = None
    basis: BasisSet | None = None
    coefficients_a: SerializableMatrix | None = None
    coefficients_b: SerializableMatrix | None = None

    @model_validator(mode="after")
    def _check(self) -> Orbitals:
        occ_a, occ_b = self.occupations_a, self.occupations_b
        if occ_a.ndim != 1 or occ_b.shape != occ_a.shape:
            raise ValueError(
                "occupations_a and occupations_b must be vectors of one value per "
                f"orbital, not shapes {occ_a.shape} and {occ_b.shape}"
            )
        nmo = occ_a.shape[0]
        for name, occ in (("occupations_a", occ_a), ("occupations_b", occ_b)):
            if not np.all((occ >= 0) & (occ <= 1)):
                raise ValueError(f"{name} must be between 0 and 1 (electrons per spin)")
        unrestricted = self.spin == OrbitalSpin.unrestricted
        self._check_energies(nmo, unrestricted)
        if (self.kind == OrbitalKind.localized) != (self.localization is not None):
            raise ValueError(
                "localization names the method of localized orbitals: required for "
                "them, None for every other kind"
            )
        self._check_smearing()
        self._check_coefficients(nmo, unrestricted)
        self._check_core_electrons()
        return self

    def _check_energies(self, nmo: int, unrestricted: bool) -> None:
        if not unrestricted and self.energies_b is not None:
            raise ValueError(
                "restricted orbitals have one set of energies: no energies_b"
            )
        if self.kind == OrbitalKind.canonical:
            if self.energies_a is None or (unrestricted and self.energies_b is None):
                raise ValueError(
                    "canonical orbitals need their energies: energies_a, and energies_b "
                    "when unrestricted"
                )
        elif self.energies_a is not None or self.energies_b is not None:
            raise ValueError(f"{self.kind.value} orbitals have no energies")
        for name in ("energies_a", "energies_b"):
            v = getattr(self, name)
            if v is not None and v.shape != (nmo,):
                raise ValueError(
                    f"{name} must have one value per orbital ({nmo}), not shape {v.shape}"
                )

    def _check_smearing(self) -> None:
        if not self.electronic_temperature >= 0:
            raise ValueError("electronic_temperature must be >= 0 K")
        levels = (self.fermi_level_a, self.fermi_level_b)
        if self.electronic_temperature == 0:
            if levels != (None, None):
                raise ValueError(
                    "Fermi levels belong to smeared occupations "
                    "(electronic_temperature > 0); at 0 K they are None"
                )
        elif None in levels:
            raise ValueError(
                "smeared occupations (electronic_temperature > 0) need fermi_level_a "
                "and fermi_level_b"
            )

    def _check_coefficients(self, nmo: int, unrestricted: bool) -> None:
        basis, ca, cb = self.basis, self.coefficients_a, self.coefficients_b
        if (basis is None) != (ca is None):
            raise ValueError("basis and coefficients_a come together: both or neither")
        if basis is None or ca is None:
            if cb is not None:
                raise ValueError("coefficients_b needs coefficients_a and basis")
            return
        shape = (basis.nao, nmo)
        if ca.shape != shape:
            raise ValueError(
                f"coefficients_a must be nao x nmo = {shape}, not {ca.shape}"
            )
        if not unrestricted:
            if cb is not None:
                raise ValueError(
                    "restricted orbitals have one coefficient matrix: no coefficients_b"
                )
        elif cb is None:
            raise ValueError(
                "unrestricted orbitals with coefficients need coefficients_b"
            )
        elif cb.shape != shape:
            raise ValueError(
                f"coefficients_b must be nao x nmo = {shape}, not {cb.shape}"
            )

    def _check_core_electrons(self) -> None:
        core = self.core_electrons
        if core is not None and min(core, default=0) < 0:
            raise ValueError("core_electrons must be >= 0")
        ecp_atoms = [e.center for e in self.basis.ecp] if self.basis else []
        bare = [a for a in ecp_atoms if core is None or a >= len(core) or core[a] == 0]
        if bare:
            raise ValueError(
                f"atoms {bare} have ECPs, so core_electrons must give the electrons "
                "each ECP replaces"
            )

    @property
    def nmo(self) -> int:
        """Number of molecular orbitals."""
        return int(self.occupations_a.shape[0])

    @property
    def restricted(self) -> bool:
        """Whether the orbitals are restricted."""
        return self.spin == OrbitalSpin.restricted

    def density(self) -> tuple[np.ndarray, np.ndarray]:
        """The alpha and beta AO density matrices, C n C^T per spin.

        Returns:
            Two nao x nao matrices in the basis set's AO convention.

        Raises:
            ValueError: If the orbitals have no coefficients.
        """
        _, ca, cb = self._require_coefficients("density()")
        return (ca * self.occupations_a) @ ca.T, (cb * self.occupations_b) @ cb.T

    def with_convention(self, convention: AOConvention) -> Orbitals:
        """The same orbitals with the basis functions in another AO convention.

        `Wavefunction.with_convention` converts the AO matrices with them.

        Args:
            convention: The target convention.

        Returns:
            New orbitals whose basis set has the target convention.

        Raises:
            ValueError: If the orbitals have no coefficients.
        """
        basis = self._require_coefficients("with_convention()")[0]
        update: dict[str, Any] = {
            "basis": basis.model_copy(update={"ao_convention": convention})
        }
        for name in ("coefficients_a", "coefficients_b"):
            c = getattr(self, name)
            if c is not None:
                update[name] = basis.transform(c, convention, kind="coefficients")
        return self.model_copy(update=update)

    def _require_coefficients(
        self, what: str
    ) -> tuple[BasisSet, np.ndarray, np.ndarray]:
        """The basis set and the alpha and beta coefficients (one matrix if restricted).

        Raises:
            ValueError: If the orbitals have no coefficients.
        """
        if self.basis is None or self.coefficients_a is None:
            raise ValueError(
                f"{what} needs the orbitals' coefficients, and these orbitals have only "
                "their levels: programs return coefficients and the basis set on request"
            )
        cb = (
            self.coefficients_b
            if self.coefficients_b is not None
            else self.coefficients_a
        )
        return self.basis, self.coefficients_a, cb


# --- AO bookkeeping ---------------------------------------------------------------------


def _dfact(n: int) -> float:
    """(n)!! with (-1)!! = 1."""
    r = 1.0
    while n > 1:
        r *= n
        n -= 2
    return r


def _xl_norm(l: int, a: np.ndarray) -> np.ndarray:
    """Normalization constant of the primitive x^l exp(-a r^2)."""
    a = np.asarray(a, dtype=float)
    return np.sqrt((2 * a / math.pi) ** 1.5 * (4 * a) ** l / _dfact(2 * l - 1))


def _normalize_raw(l: int, exps: np.ndarray, raw: np.ndarray) -> np.ndarray:
    """Scale coefficients of unnormalized primitives so that x^l of the shell has unit norm."""
    p = exps[:, None] + exps[None, :]
    s = float(raw @ ((math.pi / p) ** 1.5 * _dfact(2 * l - 1) / (2 * p) ** l) @ raw)
    return raw / math.sqrt(s)


def _cartesian_components(l: int, order: CartesianOrder) -> list[tuple[int, int, int]]:
    lex = [
        (lx, ly, l - lx - ly) for lx in range(l, -1, -1) for ly in range(l - lx, -1, -1)
    ]
    if order == CartesianOrder.lexicographic or l < 2:
        return lex
    table = {
        2: "xx yy zz xy xz yz",
        3: "xxx yyy zzz xyy xxy xxz xzz yzz yyz xyz",
        4: "xxxx yyyy zzzz xxxy xxxz yyyx yyyz zzzx zzzy xxyy xxzz yyzz xxyz yyxz zzxy",
    }
    if l not in table:
        raise ValueError(
            f"Molden's Cartesian order stops at g (l = 4); this basis has l = {l}"
        )
    return [(c.count("x"), c.count("y"), c.count("z")) for c in table[l].split()]


def _spherical_ms(l: int, order: SphericalOrder) -> list[int]:
    if l == 1 and order in (SphericalOrder.pyscf, SphericalOrder.molden):
        return [1, -1, 0]  # x, y, z
    if order in (SphericalOrder.molden, SphericalOrder.trexio):
        return [0] + [s * m for m in range(1, l + 1) for s in (1, -1)]
    return list(range(-l, l + 1))


def _function_keys(shell: Shell, conv: AOConvention) -> list[tuple[Any, ...]]:
    l = shell.angular_momentum
    if shell.pure:
        return [("m", m) for m in _spherical_ms(l, conv.spherical_order)]
    return [("c", c) for c in _cartesian_components(l, conv.cartesian_order)]


def _function_norm(l: int, key: tuple[Any, ...], conv: AOConvention) -> float:
    """Norm of a function of a shell in convention `conv`."""
    if key[0] == "m" or conv.cartesian_normalization == CartesianNormalization.unit:
        return 1.0
    lx, ly, lz = key[1]
    n = math.sqrt(
        _dfact(2 * lx - 1) * _dfact(2 * ly - 1) * _dfact(2 * lz - 1) / _dfact(2 * l - 1)
    )
    if conv.cartesian_normalization == CartesianNormalization.pyscf and l >= 2:
        n *= math.sqrt(4 * math.pi / (2 * l + 1))
    return n


def _function_name(l: int, key: tuple[Any, ...]) -> str:
    letter = L_LETTERS[l] if l < len(L_LETTERS) else f"l{l}"
    if l == 0:
        return letter
    if key[0] == "c":
        lx, ly, lz = key[1]
        return letter + "x" * lx + "y" * ly + "z" * lz
    m = key[1]
    if l == 1:
        return letter + " " + {1: "x", -1: "y", 0: "z"}[m]
    return f"{letter}{m:+d}" if m else f"{letter}0"


def _ao_map(
    basis: BasisSet, target: AOConvention, shell_order: Sequence[int] | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Map AOs of the basis set's convention onto `target`'s.

    Coefficients in `target` (shells in `shell_order`) are ``scale * c[perm]`` of
    coefficients ``c`` in the basis set's own convention.
    """
    offsets = basis.shell_offsets()
    order = range(len(basis.shells)) if shell_order is None else shell_order
    if sorted(order) != list(range(len(basis.shells))):
        raise ValueError("shell_order must be a permutation of the shells")
    perm: list[int] = []
    scale: list[float] = []
    for si in order:
        sh = basis.shells[si]
        src = {k: i for i, k in enumerate(_function_keys(sh, basis.ao_convention))}
        for key in _function_keys(sh, target):
            perm.append(offsets[si] + src[key])
            l = sh.angular_momentum
            scale.append(
                _function_norm(l, key, basis.ao_convention)
                / _function_norm(l, key, target)
            )
    return np.asarray(perm, dtype=np.intp), np.asarray(scale)
