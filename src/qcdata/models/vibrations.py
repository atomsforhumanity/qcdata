"""Harmonic vibrational analysis and ideal-gas thermochemistry.

A Hessian's frequencies and free energies can only be compared across programs when the
choices behind them are recorded: the masses, what was projected out, the standard state,
the symmetry number, the entropy model, the scale factor, and what was done with imaginary
modes. These models carry the results together with those choices.

Units: atomic units, except where chemistry has a fixed convention: frequencies and
rotational constants in cm^-1, masses in u (daltons), force constants in mdyn/angstrom,
IR intensities in km/mol, temperatures in K, pressures in Pa, concentrations in mol/L.
Imaginary frequencies are negative numbers.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Self

import numpy as np
from pydantic import field_validator, model_validator

from qcdata.helper_types import (
    SerializableMatrix,
    SerializableNDArray,
    SerializableTensor3D,
    SerializableVector,
)

from .base_models import QCDataBaseModel

__all__ = [
    "Projection",
    "RotorType",
    "EntropyModel",
    "EnthalpyModel",
    "VibrationalAnalysis",
    "ThermoTerm",
    "Thermochemistry",
]


class Projection(StrEnum):
    """What was projected out of the mass-weighted Hessian before diagonalization.

    Attributes:
        translations_rotations: Translations and rotations (the usual choice).
        translations: Translations only.
        none: Nothing; translation- and rotation-like modes stay among the modes and
            are flagged in `VibrationalAnalysis.external`.
    """

    translations_rotations = "translations_rotations"
    translations = "translations"
    none = "none"


class RotorType(StrEnum):
    """The rigid rotor the molecule is.

    Attributes:
        atom: A single atom (no rotations).
        linear: A linear molecule (two rotations).
        nonlinear: A nonlinear molecule (three rotations).
    """

    atom = "atom"
    linear = "linear"
    nonlinear = "nonlinear"


class EntropyModel(StrEnum):
    """How vibrational entropy is computed.

    Attributes:
        rrho: Rigid rotor, harmonic oscillator for every mode.
        qrrho_grimme: Grimme's quasi-RRHO: low modes interpolated to free rotors
            (Chem. Eur. J. 18, 9955 (2012)); parameters in `Thermochemistry.qrrho_*`.
        quasiharmonic_truhlar: Truhlar's quasi-harmonic: frequencies below a cutoff
            raised to it.
    """

    rrho = "rrho"
    qrrho_grimme = "qrrho_grimme"
    quasiharmonic_truhlar = "quasiharmonic_truhlar"


class EnthalpyModel(StrEnum):
    """How vibrational energy and enthalpy are computed.

    Attributes:
        rrho: Harmonic oscillator for every mode.
        qrrho_head_gordon: Head-Gordon's quasi-RRHO enthalpy (J. Phys. Chem. C 119,
            1840 (2015)).
    """

    rrho = "rrho"
    qrrho_head_gordon = "qrrho_head_gordon"


def _vector(v: SerializableNDArray | None) -> np.ndarray | None:
    return None if v is None else np.asarray(v, dtype=float).reshape(-1)


class VibrationalAnalysis(QCDataBaseModel):
    """The harmonic analysis of a Hessian, modes in ascending order of frequency.

    Attributes:
        masses: Mass of each atom used (u), in structure order.
        projection: What was projected out of the mass-weighted Hessian before it was
            diagonalized; `SinglePointData.hessian` itself is never projected.
        frequencies: Frequency of every mode analysed (cm^-1), ascending; imaginary
            modes are negative. Without full projection the translation/rotation-like
            modes are included and flagged in `external`.
        external: Per mode, whether it is a translation/rotation rather than a
            vibration. Empty means no mode is external.
        normal_modes_cartesian: n_modes x n_atoms x 3 unit-length Cartesian
            displacements (dimensionless, sign arbitrary), as
            `SinglePointData.normal_modes_cartesian`.
        normal_modes_mass_weighted: n_modes x 3N orthonormal eigenvectors of the
            mass-weighted Hessian.
        reduced_masses: Per mode (u): 1 / |M^-1/2 L|^2 for the mass-weighted
            eigenvector L.
        force_constants: Per mode (mdyn/angstrom): reduced mass times eigenvalue,
            negative for imaginary modes.
        ir_intensities: Per mode (km/mol).
        dipole_derivatives: 3N x 3 atomic polar tensors d(mu_c)/d(X) (e), the input of
            `ir_intensities`; row 3a + i for atom a and Cartesian component i.
        rotor_type: Atom, linear, or nonlinear.
        moments_of_inertia: Principal moments (u bohr^2), ascending.
        rotational_constants: A >= B >= C (cm^-1); None for a zero moment.
        point_group: Schoenflies symbol of the structure with these masses.
        symmetry_number: The rotational symmetry number used.
        symmetry_number_source: ``"detected"`` or ``"input"``.
    """

    masses: SerializableVector
    projection: Projection
    frequencies: SerializableVector
    external: list[bool] = []
    normal_modes_cartesian: SerializableTensor3D
    normal_modes_mass_weighted: SerializableMatrix | None = None
    reduced_masses: SerializableVector | None = None
    force_constants: SerializableVector | None = None
    ir_intensities: SerializableVector | None = None
    dipole_derivatives: SerializableMatrix | None = None
    rotor_type: RotorType | None = None
    moments_of_inertia: SerializableVector | None = None
    rotational_constants: list[float | None] | None = None
    point_group: str | None = None
    symmetry_number: int | None = None
    symmetry_number_source: str | None = None

    @field_validator(
        "masses",
        "frequencies",
        "reduced_masses",
        "force_constants",
        "ir_intensities",
        "moments_of_inertia",
    )
    @classmethod
    def _one_dimensional(cls, v: SerializableNDArray | None) -> np.ndarray | None:
        return _vector(v)

    @model_validator(mode="after")
    def _shapes(self) -> Self:
        natoms = len(self.masses)
        nmodes = len(self.frequencies)
        modes = np.asarray(self.normal_modes_cartesian, dtype=float)
        if modes.size != nmodes * natoms * 3:
            raise ValueError(
                f"normal_modes_cartesian must be {nmodes} x {natoms} x 3 "
                f"(modes x atoms x 3), not {modes.size} values"
            )
        object.__setattr__(
            self, "normal_modes_cartesian", modes.reshape(nmodes, natoms, 3)
        )
        if self.external and len(self.external) != nmodes:
            raise ValueError(
                f"external must have one flag per mode ({nmodes}), "
                f"not {len(self.external)}"
            )
        for name in ("reduced_masses", "force_constants", "ir_intensities"):
            v = getattr(self, name)
            if v is not None and len(v) != nmodes:
                raise ValueError(
                    f"{name} must have one value per mode ({nmodes}), not {len(v)}"
                )
        if self.normal_modes_mass_weighted is not None:
            mw = np.asarray(self.normal_modes_mass_weighted, dtype=float)
            if mw.size != nmodes * 3 * natoms:
                raise ValueError(
                    f"normal_modes_mass_weighted must be {nmodes} x {3 * natoms}"
                )
            object.__setattr__(
                self, "normal_modes_mass_weighted", mw.reshape(nmodes, 3 * natoms)
            )
        if self.dipole_derivatives is not None:
            apt = np.asarray(self.dipole_derivatives, dtype=float)
            if apt.size != 3 * natoms * 3:
                raise ValueError(f"dipole_derivatives must be {3 * natoms} x 3")
            object.__setattr__(self, "dipole_derivatives", apt.reshape(3 * natoms, 3))
        return self

    @property
    def is_vibration(self) -> np.ndarray:
        """Per mode, whether it is a vibration (not external)."""
        if not self.external:
            return np.ones(len(self.frequencies), dtype=bool)
        return ~np.asarray(self.external, dtype=bool)

    @property
    def vibrational_frequencies(self) -> np.ndarray:
        """Frequencies of the vibrations (cm^-1), external modes left out."""
        return np.asarray(self.frequencies)[self.is_vibration]

    @property
    def vibrational_modes_cartesian(self) -> np.ndarray:
        """Unit Cartesian displacements of the vibrations (n_vib x n_atoms x 3)."""
        return np.asarray(self.normal_modes_cartesian)[self.is_vibration]

    def imaginary_frequencies(self, cutoff: float = 0.0) -> np.ndarray:
        """Imaginary vibrational frequencies (negative, cm^-1).

        Args:
            cutoff: Only modes below ``-cutoff`` count (cm^-1, positive). Programs
                differ in their default (xtb and axtb ignore modes above -20 cm^-1),
                so the stored data apply none and the caller chooses.

        Returns:
            The imaginary frequencies, most negative first.
        """
        f = self.vibrational_frequencies
        return f[f < -abs(cutoff)]

    def n_imaginary(self, cutoff: float = 0.0) -> int:
        """Number of imaginary vibrations below ``-cutoff`` cm^-1.

        Args:
            cutoff: See `imaginary_frequencies`.

        Returns:
            The count.
        """
        return len(self.imaginary_frequencies(cutoff))


class ThermoTerm(QCDataBaseModel):
    """One contribution to the thermochemistry, or the total.

    Conventions: the vibrational energy includes the zero-point energy; pV = kT is in the
    translational enthalpy; the electronic term's energy is the electronic energy itself
    only in `Thermochemistry.total`, which includes it.

    Attributes:
        energy: Internal energy U (Eh).
        enthalpy: Enthalpy H (Eh).
        entropy: Entropy S (Eh/K).
        cv: Heat capacity at constant volume (Eh/K).
        cp: Heat capacity at constant pressure (Eh/K).
        gibbs: Gibbs energy G = H - TS (Eh).
    """

    energy: float | None = None
    enthalpy: float | None = None
    entropy: float | None = None
    cv: float | None = None
    cp: float | None = None
    gibbs: float | None = None


class Thermochemistry(QCDataBaseModel):
    """Ideal-gas thermochemistry at one temperature and standard state.

    Attributes:
        temperature: Temperature (K).
        pressure: Pressure of an ideal-gas standard state (Pa).
        concentration: A concentration standard state instead (mol/L, e.g. 1 M).
        electronic_energy: The electronic energy the corrections were added to (Eh).
        zero_point_energy: Zero-point vibrational energy (Eh).
        electronic: The electronic contribution (degeneracy entropy).
        translational: The translational contribution.
        rotational: The rotational contribution.
        vibrational: The vibrational contribution (including the zero-point energy).
        total: All contributions plus the electronic energy.
        symmetry_number: The rotational symmetry number used.
        electronic_degeneracy: The electronic degeneracy used (e.g. the multiplicity).
        entropy_model: How vibrational entropy was computed.
        enthalpy_model: How vibrational energy and enthalpy were computed.
        qrrho_frequency: The quasi-RRHO switching or cutoff frequency (cm^-1).
        qrrho_alpha: The exponent of Grimme's damping function.
        qrrho_bav: Grimme's average moment of inertia B_av (kg m^2).
        frequency_scale_factor: The factor frequencies were scaled by.
        excluded_imaginary: Imaginary frequencies left out (cm^-1, negative), e.g. a
            transition state's reaction coordinate.
        inverted_imaginary: Imaginary frequencies treated as real (cm^-1, negative).
    """

    temperature: float
    pressure: float | None = None
    concentration: float | None = None
    electronic_energy: float
    zero_point_energy: float
    electronic: ThermoTerm | None = None
    translational: ThermoTerm | None = None
    rotational: ThermoTerm | None = None
    vibrational: ThermoTerm | None = None
    total: ThermoTerm
    symmetry_number: int | None = None
    electronic_degeneracy: float | None = None
    entropy_model: EntropyModel = EntropyModel.rrho
    enthalpy_model: EnthalpyModel = EnthalpyModel.rrho
    qrrho_frequency: float | None = None
    qrrho_alpha: float | None = None
    qrrho_bav: float | None = None
    frequency_scale_factor: float = 1.0
    excluded_imaginary: list[float] = []
    inverted_imaginary: list[float] = []

    @model_validator(mode="after")
    def _standard_state(self) -> Self:
        if self.pressure is not None and self.concentration is not None:
            raise ValueError(
                "Give a pressure or a concentration standard state, not both."
            )
        return self

    def _correction(self, value: float | None) -> float | None:
        return None if value is None else value - self.electronic_energy

    @property
    def gibbs_free_energy(self) -> float | None:
        """G including the electronic energy (Eh)."""
        return self.total.gibbs

    @property
    def enthalpy(self) -> float | None:
        """H including the electronic energy (Eh)."""
        return self.total.enthalpy

    @property
    def gibbs_correction(self) -> float | None:
        """G minus the electronic energy (Eh): add it to an energy at another level."""
        return self._correction(self.total.gibbs)

    @property
    def enthalpy_correction(self) -> float | None:
        """H minus the electronic energy (Eh)."""
        return self._correction(self.total.enthalpy)

    @property
    def thermal_energy_correction(self) -> float | None:
        """U minus the electronic energy (Eh)."""
        return self._correction(self.total.energy)
