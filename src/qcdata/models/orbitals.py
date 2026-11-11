"""Basis sets and molecular orbitals.

`Orbitals` holds MO coefficients (nao x nmo, one orbital per column), energies and
occupations per spin, together with the `BasisSet` they expand in. A `BasisSet` is a
list of `Shell` objects placed on the structure's atoms, in AO order, and an explicit
`AOConvention` saying how the functions within a shell are ordered and normalized, so
coefficients from any program can be stored as produced and converted when needed
(`Orbitals.with_convention`).

Coefficients of a shell multiply the *unnormalized* primitives r^l exp(-a r^2) and
include all normalization: the x^l Cartesian component of the contracted shell, and so
every spherical function, has unit norm. An AO is then
``sum_i coefficients[i] * x^lx y^ly z^lz exp(-exponents[i] r^2)`` (times the Cartesian
to spherical transformation of unit-normalized real solid harmonics for pure shells),
with no convention left to guess except the ones `AOConvention` names. General
contractions are split into one shell per contracted function.

Contributed from linus (``linus.models``); converters to PySCF and Molden live outside
qcdata.
"""

# l is the conventional symbol for angular momentum.
# ruff: noqa: E741
from __future__ import annotations

import math
from collections.abc import Sequence
from enum import StrEnum
from typing import Any

import numpy as np
from pydantic import field_validator, model_validator

from qcdata.helper_types import SerializableMatrix, SerializableVector

from .base_models import QCDataBaseModel

__all__ = [
    "AOConvention",
    "BasisSet",
    "CartesianNormalization",
    "CartesianOrder",
    "OrbitalSpin",
    "Orbitals",
    "Shell",
    "SphericalOrder",
    "DEFAULT_AO_CONVENTION",
    "PYSCF_CONVENTION",
    "MOLDEN_CONVENTION",
]

L_LETTERS = "spdfghiklmn"


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


class BasisSet(QCDataBaseModel):
    """A basis set placed on the atoms of a structure, in AO order.

    Attributes:
        name: The basis set's name (e.g. ``"def2-svp"``), if it has one.
        shells: The contracted shells; the AOs are their functions in this order.
        ao_convention: Order and normalization of the functions within each shell.
    """

    name: str | None = None
    shells: list[Shell]
    ao_convention: AOConvention = AOConvention()

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
        coefficients: Any,
        convention: AOConvention,
        shell_order: Sequence[int] | None = None,
    ) -> np.ndarray:
        """Convert AO-indexed rows to another AO convention and shell order.

        Args:
            coefficients: An array whose first index is the AO (e.g. MO coefficients,
                nao x nmo).
            convention: The target convention.
            shell_order: The shells' order in the target, as indices into `shells`;
                None keeps the order.

        Returns:
            The rows permuted and scaled into the target convention.

        Raises:
            ValueError: If the rows do not match the basis set's AOs.
        """
        perm, scale = _ao_map(self, convention, shell_order)
        c = np.asarray(coefficients, dtype=float)
        if c.shape[0] != self.nao:
            raise ValueError(f"expected {self.nao} rows (AOs), got {c.shape[0]}")
        return c[perm] * scale.reshape((-1,) + (1,) * (c.ndim - 1))


class OrbitalSpin(StrEnum):
    """Whether alpha and beta electrons share spatial orbitals.

    Attributes:
        restricted: One set of spatial orbitals for both spins (RHF, RKS, ROHF).
        unrestricted: Separate alpha and beta orbitals (UHF, UKS).
    """

    restricted = "restricted"
    unrestricted = "unrestricted"


class Orbitals(QCDataBaseModel):
    """Molecular orbitals in a basis set.

    Attributes:
        basis: The basis set the orbitals expand in (centers refer to the structure's atoms).
        spin: ``restricted`` (one set of orbitals: the ``_b`` coefficients and energies are
            None) or ``unrestricted``.
        kind: What the orbitals are: ``canonical`` (default), ``natural``, ``localized``...
        coefficients_a: nao x nmo; column i is orbital i (alpha, or both spins).
        coefficients_b: Beta orbitals (unrestricted only).
        energies_a: Orbital energies (Eh), if they have any.
        energies_b: Beta orbital energies (unrestricted only).
        occupations_a: Alpha occupation of each orbital (0..1).
        occupations_b: Beta occupation of each orbital (0..1); for restricted orbitals of
            an open shell (ROHF) it differs from the alpha one. None: as alpha.
    """

    basis: BasisSet
    spin: OrbitalSpin
    kind: str = "canonical"
    coefficients_a: SerializableMatrix
    coefficients_b: SerializableMatrix | None = None
    energies_a: SerializableVector | None = None
    energies_b: SerializableVector | None = None
    occupations_a: SerializableVector | None = None
    occupations_b: SerializableVector | None = None

    @model_validator(mode="after")
    def _check(self) -> Orbitals:
        nao = self.basis.nao
        ca = self.coefficients_a
        if ca.ndim != 2 or ca.shape[0] != nao:
            raise ValueError(
                f"coefficients_a must be nao x nmo with nao = {nao}, not {ca.shape}"
            )
        nmo = ca.shape[1]
        if self.spin == OrbitalSpin.restricted:
            if self.coefficients_b is not None or self.energies_b is not None:
                raise ValueError(
                    "restricted orbitals have no coefficients_b or energies_b"
                )
        elif self.coefficients_b is None:
            raise ValueError("unrestricted orbitals need coefficients_b")
        if self.coefficients_b is not None and self.coefficients_b.shape != ca.shape:
            raise ValueError("coefficients_b must have the shape of coefficients_a")
        for name in ("energies_a", "energies_b", "occupations_a", "occupations_b"):
            v = getattr(self, name)
            if v is not None and v.shape != (nmo,):
                raise ValueError(
                    f"{name} must have one value per orbital ({nmo}), not shape {v.shape}"
                )
        return self

    @property
    def nao(self) -> int:
        """Number of atomic orbitals."""
        return self.coefficients_a.shape[0]

    @property
    def nmo(self) -> int:
        """Number of molecular orbitals."""
        return self.coefficients_a.shape[1]

    @property
    def restricted(self) -> bool:
        """Whether the orbitals are restricted."""
        return self.spin == OrbitalSpin.restricted

    def spin_orbitals(
        self,
    ) -> tuple[tuple[np.ndarray, np.ndarray | None, np.ndarray | None], ...]:
        """((coefficients, energies, occupations) of alpha, of beta)."""
        occ_a = self.occupations_a
        occ_b = self.occupations_b if self.occupations_b is not None else occ_a
        if self.restricted:
            return (self.coefficients_a, self.energies_a, occ_a), (
                self.coefficients_a,
                self.energies_a,
                occ_b,
            )
        assert self.coefficients_b is not None  # validated for unrestricted orbitals
        return (self.coefficients_a, self.energies_a, occ_a), (
            self.coefficients_b,
            self.energies_b,
            occ_b,
        )

    def density(self) -> tuple[np.ndarray, np.ndarray]:
        """Alpha and beta AO density matrices C n C^T."""
        out = []
        for c, _, occ in self.spin_orbitals():
            if occ is None:
                raise ValueError("densities need occupations")
            out.append((c * occ) @ c.T)
        return out[0], out[1]

    def with_convention(self, convention: AOConvention) -> Orbitals:
        """The same orbitals with the basis functions in another AO convention."""
        values = self.model_dump()
        values["basis"]["ao_convention"] = convention.model_dump()
        values["coefficients_a"] = self.basis.transform(self.coefficients_a, convention)
        if self.coefficients_b is not None:
            values["coefficients_b"] = self.basis.transform(
                self.coefficients_b, convention
            )
        return Orbitals.model_validate(values)


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
