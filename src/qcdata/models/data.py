"""General-purpose scientific data models."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar, Union

import numpy as np
from pydantic import BaseModel, Field, ValidationInfo, field_validator, model_validator
from typing_extensions import Self

from qcdata.helper_types import (
    SerializableMatrix,
    SerializableNDArray,
    SerializableTensor3D,
    SerializableVector,
)

from .base_models import CalcType, Files, Provenance, QCDataBaseModel
from .inputs import ProgramInput
from .structure import Structure
from .utils import to_multi_xyz

if TYPE_CHECKING:
    from .outputs import ProgramOutput

__all__ = [
    "FileData",
    "Wavefunction",
    "SinglePointData",
    "OptimizationData",
    "ConformerSearchData",
    "ScanData",
    "StructuredData",
    "StructuredDataType",
    "Data",
    "DataType",
    "get_data_type",
]


class FileData(Files):
    """File-only results and the common base for structured calculation data.

    Attributes:
        provenance: Required identity of the program that produced the data.
        files: Output files, with binary content encoded as base64 during serialization.
        extras: Additional result information.
    """

    provenance: Provenance


class Wavefunction(QCDataBaseModel):
    """The wavefunction for a single point calculation.

    Attributes:
        scf_eigenvalues_a: The SCF alpha-spin orbital eigenvalues.
        scf_eigenvalues_b: The SCF beta-spin orbital eigenvalues.
        scf_occupations_a: The SCF alpha-spin orbital occupations.
        scf_occupations_b: The SCF beta-spin orbital occupations.
    """

    scf_eigenvalues_a: SerializableVector | None = None
    scf_eigenvalues_b: SerializableVector | None = None
    scf_occupations_a: SerializableVector | None = None
    scf_occupations_b: SerializableVector | None = None

    @field_validator(
        "scf_eigenvalues_a",
        "scf_eigenvalues_b",
        "scf_occupations_a",
        "scf_occupations_b",
    )
    @classmethod
    def to_numpy(cls, val, _info) -> np.ndarray | None:
        return np.asarray(val) if val is not None else None


class CalcInfoData(BaseModel):
    """Mixin for calcinfo attributes.

    Attributes:
        calcinfo_natoms: The number of atoms as computed by the program.
        calcinfo_nalpha: The number of alpha electrons as computed by the program.
        calcinfo_nbeta: The number of beta electrons as computed by the program.
        calcinfo_nbasis: The number of basis functions in the calculation.
        calcinfo_nmo: The number of molecular orbitals in the calculation.
    """

    calcinfo_natoms: int | None = None
    calcinfo_nalpha: int | None = None
    calcinfo_nbeta: int | None = None
    calcinfo_nbasis: int | None = None
    calcinfo_nmo: int | None = None


class SinglePointData(FileData, CalcInfoData):
    """The computed data from a single point calculation.

    Attributes:
        energy: The electronic energy of the structure in `Hartrees`.
        gradient: The gradient of the structure in `Hartrees/Bohr`.
        hessian: The hessian of the structure in `Hartrees/Bohr^2`.
        nuclear_repulsion_energy: The nuclear repulsion energy of the structure in
            Hartrees.

        wavefunction: Wavefunction data from the calculation.

        freqs_wavenumber: The frequencies of the structure in wavenumbers.
        normal_modes_cartesian: 3D n_vibmodes x n_atoms x 3 array containing
            un-mass-weighted Cartesian displacements of each normal mode in Bohr.
        gibbs_free_energy: Gibbs free energy (i.e. thermochemical analysis) in Hartrees
            of a system where translation / rotation / vibration degrees of freedom are
            approximated using ideal gas / rigid rotor / harmonic oscillator
            respectively.
        scf_dipole_moment: The x, y, z component of the dipole moment of the structure
            in units of e a0 (NOT Debye!).

    """

    energy: float | None = None
    gradient: SerializableMatrix | None = None
    hessian: SerializableMatrix | None = None
    nuclear_repulsion_energy: float | None = None

    wavefunction: Wavefunction | None = None

    freqs_wavenumber: list[float] = []
    normal_modes_cartesian: SerializableTensor3D | None = None
    gibbs_free_energy: float | None = None

    scf_dipole_moment: list[float] | None = None

    @field_validator("normal_modes_cartesian")
    @classmethod
    def _validate_normal_modes_cartesian_shape(
        cls, v: SerializableNDArray, info: ValidationInfo
    ):
        if v is not None:
            freqs = info.data.get("freqs_wavenumber")
            n_normal_modes = len(freqs) if freqs else len(v)
            return np.asarray(v).reshape(n_normal_modes, -1, 3)
        return v

    @field_validator("gradient")
    @classmethod
    def _validate_gradient_shape(cls, v: SerializableNDArray):
        """Validate gradient is n x 3."""
        if v is not None:
            return np.asarray(v).reshape(-1, 3)

    @field_validator("hessian")
    @classmethod
    def _validate_hessian_shape(cls, v: SerializableNDArray):
        """Validate hessian is square."""
        if v is not None:
            v = np.asarray(v)
            n = int(np.sqrt(v.size))
            return v.reshape((n, n))

    def return_result(self, calctype: CalcType) -> float | SerializableNDArray | None:
        """Return the primary result of the calculation."""
        return getattr(self, calctype.value)


class OptimizationData(FileData, CalcInfoData):
    """Computed data for an optimization (may be for a minimum or transition state).

    Attributes:
        energies: The energies for each step of the optimization.
        structures: The Structure objects for each step of the optimization.
        final_structure: The final, optimized Structure.
        trajectory: The ProgramOutput objects for each step of the optimization.
    """

    trajectory: list[ProgramOutput[ProgramInput, SinglePointData]] = []

    @property
    def final_structure(self) -> Structure | None:
        """The last evaluated structure, or None when there are no steps."""
        return self.structures[-1] if self.trajectory else None

    @property
    def final_energy(self) -> float | None:
        """
        The final step's energy, None for no steps, or `np.nan` if unavailable.
        """
        return self.energies[-1] if self.trajectory else None

    @property
    def energies(self) -> np.ndarray:
        """The energies for each step of the optimization."""
        return np.array(
            [
                output.results.energy if output.results.energy is not None else np.nan
                for output in self.trajectory
            ],
            dtype=float,
        )

    @property
    def structures(self) -> list[Structure]:
        """The Structure objects for each step of the optimization."""
        return [output.input_data.structure for output in self.trajectory]

    def return_result(self, calctype: CalcType) -> Structure | None:
        return self.final_structure

    def __repr_args__(self):
        """Custom repr to avoid printing the entire collection of objects."""
        return [
            ("final_structure", f"{self.final_structure}"),
            ("trajectory", "[...]"),
            ("energies", "[...]"),
            ("structures", "[...]"),
        ]

    def to_xyz(self) -> str:
        """Return the trajectory as an `xyz` string."""
        return to_multi_xyz(
            prog_output.input_data.structure for prog_output in self.trajectory
        )

    def save(
        self,
        filepath: Path | str,
        exclude_none: bool = True,
        exclude_unset: bool = True,
        indent: int = 4,
        **kwargs: dict[str, Any],
    ) -> None:
        """Save an OptimizationOutput to a file.

        Args:
            filepath: The path to save the molecule to.
            exclude_none: If True, attributes with a value of None will not be written
                to the file.
            exclude_unset: If True, attributes that have not been set will not be
                written to the file. Defaults to True for compact files.
            **kwargs: Additional keyword arguments to pass to the json serializer.

        Note:
            If the filepath has a `.xyz` extension, the trajectory will be saved to a
            multi-structure `xyz` file.
        """
        filepath = Path(filepath)
        if filepath.suffix == ".xyz":
            filepath.write_text(self.to_xyz())
            return
        super().save(filepath, exclude_none, exclude_unset, indent, **kwargs)


class ConformerSearchData(FileData):
    """Data from a conformer search calculation.

    Conformers and rotamers are sorted by energy.

    Attributes:
        conformers: The conformers found in the search.
        conformer_energies: The energies for each conformer.
        rotamers: The rotamers found in the search.
        rotamer_energies: The energies for each rotamer.
    """

    conformers: list[Structure] = []
    conformer_energies: SerializableVector = Field(default_factory=lambda: np.array([]))
    rotamers: list[Structure] = []
    rotamer_energies: SerializableVector = Field(default_factory=lambda: np.array([]))

    @model_validator(mode="after")
    def _energies_size(self) -> Self:
        """Ensure the energies are the same size as the conformers and rotamers."""
        if self.conformer_energies.size > 0 and (
            self.conformer_energies.size != len(self.conformers)
        ):
            raise ValueError(
                "The number of conformer energies must match the number of conformers."
            )
        if self.rotamer_energies.size > 0 and (
            self.rotamer_energies.size != len(self.rotamers)
        ):
            raise ValueError(
                "The number of rotamer energies must match the number of rotamers."
            )
        return self

    @model_validator(mode="after")
    def _sort_by_energy(self) -> Self:
        """Sort conformers and rotamers by energy."""
        if self.conformer_energies.size > 0:
            sorted_indices = np.argsort(self.conformer_energies)
            self.conformers[:] = [self.conformers[i] for i in sorted_indices]
            self.conformer_energies[:] = self.conformer_energies[sorted_indices]

        if self.rotamer_energies.size > 0:
            sorted_indices = np.argsort(self.rotamer_energies)
            self.rotamers[:] = [self.rotamers[i] for i in sorted_indices]
            self.rotamer_energies[:] = self.rotamer_energies[sorted_indices]

        return self

    @property
    def conformer_energies_relative(self) -> np.ndarray:
        """The relative energies for each conformer in the search."""
        if self.conformer_energies.size == 0:
            return np.array([])
        return self.conformer_energies - self.conformer_energies.min()

    @property
    def rotamer_energies_relative(self) -> np.ndarray:
        """The relative energies for each rotamer in the search."""
        if self.rotamer_energies.size == 0:
            return np.array([])
        return self.rotamer_energies - self.rotamer_energies.min()


class ScanData(FileData, CalcInfoData):
    """Computed data for a scan (may be for a relaxed or frozen).

    Attributes
    ----------
        energies: The energies for each step of the scan.
        structures: The Structure objects for each step of the scan.
        trajectory: The ProgramOutput objects for each step of the scan.
    """

    trajectory: list[ProgramOutput[ProgramInput, OptimizationData]] = []

    @property
    def energies(self) -> np.ndarray:
        """The energies for each step of the scan."""
        return np.array(
            [
                output.results.final_energy
                if output.results.final_energy is not None
                else np.nan
                for output in self.trajectory
            ],
            dtype=float,
        )

    @property
    def structures(self) -> list[Structure | None]:
        """The final structure at each scan point, or None for an empty point."""
        return [output.results.final_structure for output in self.trajectory]

    def __repr_args__(self) -> list[tuple[str, str]]:
        """Avoid printing the entire collection of objects in representation."""
        return [
            ("trajectory", "[...]"),
            ("energies", "[...]"),
            ("structures", "[...]"),
        ]

    def to_xyz(self) -> str:
        """Return the trajectory as an `xyz` string."""
        structures = self.structures
        if any(structure is None for structure in structures):
            raise ValueError("Cannot export XYZ: a scan point has no final structure.")
        return to_multi_xyz(
            structure for structure in structures if structure is not None
        )

    def save(
        self,
        filepath: Path | str,
        exclude_none: bool = True,
        exclude_unset: bool = True,
        indent: int = 4,
        **kwargs: dict[str, Any],
    ) -> None:
        """Save a ScanData to a file.

        Args:
            filepath: The path to save the molecule to.
            exclude_none: If True, attributes with a value of None will not be written
                to the file.
            exclude_unset: If True, attributes that have not been set will not be
                written to the file. Defaults to True for compact files.
            **kwargs: Additional keyword arguments to pass to the json serializer.

        Note:
            If the filepath has a `.xyz` extension, the trajectory will be saved to a
            multi-structure `xyz` file.
        """
        filepath = Path(filepath)
        if filepath.suffix == ".xyz":
            filepath.write_text(self.to_xyz())
            return
        super().save(filepath, exclude_none, exclude_unset, indent, **kwargs)


StructuredData = Union[SinglePointData, OptimizationData, ConformerSearchData, ScanData]
StructuredDataType = TypeVar("StructuredDataType", bound=StructuredData)
Data = Union[FileData, StructuredData]
DataType = TypeVar("DataType", bound=Data)


_DATA_TYPES: dict[CalcType, type[StructuredData]] = {
    CalcType.energy: SinglePointData,
    CalcType.gradient: SinglePointData,
    CalcType.hessian: SinglePointData,
    CalcType.optimization: OptimizationData,
    CalcType.transition_state: OptimizationData,
    CalcType.conformer_search: ConformerSearchData,
    CalcType.scan: ScanData,
}


def get_data_type(calctype: CalcType | str) -> type[StructuredData]:
    """Return the scientific data class required by a calculation type.

    Accepts CalcType members or their string values. Unknown values raise ValueError.
    File-only execution uses FileData and has no calculation type.
    """
    return _DATA_TYPES[CalcType(calctype)]
