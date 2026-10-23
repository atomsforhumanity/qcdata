"""General-purpose scientific data models."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self, TypeVar

import numpy as np
from pydantic import BaseModel, Field, ValidationInfo, field_validator, model_validator

from qcdata.helper_types import (
    SerializableMatrix,
    SerializableNDArray,
    SerializableTensor3D,
    SerializableVector,
)

from .base_models import CalcType, Files, OutputBase, Provenance, QCDataBaseModel
from .inputs import ProgramInput
from .structure import Structure
from .utils import to_multi_xyz

if TYPE_CHECKING:
    from .outputs import ProgramOutput

__all__ = [
    "FileData",
    "Wavefunction",
    "SinglePointData",
    "Frame",
    "OptimizationData",
    "PathData",
    "PathOptimizationData",
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
    def to_numpy(
        cls, val: np.ndarray | None, _info: ValidationInfo
    ) -> np.ndarray | None:
        """Ensure orbital eigenvalues and occupations are stored as numpy arrays.

        Args:
            val: The eigenvalue or occupation vector, or None if not provided.
            _info: Pydantic validation context (unused).

        Returns:
            The value as a numpy array, or None if it was not provided.
        """
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

        freqs_wavenumber: Harmonic vibrational frequencies in wavenumbers (cm^-1),
            negative for imaginary modes.
        normal_modes_cartesian: n_vibmodes x n_atoms x 3 array of normal modes, one per
            entry of ``freqs_wavenumber`` and in the same order. Each mode is a
            Cartesian displacement (not mass-weighted) scaled to unit length, in the
            input structure's frame and atom order; it is dimensionless. The sign of a
            mode is arbitrary. Multiplying a mode by the square root of each atom's
            mass and normalizing gives the eigenvector of the mass-weighted Hessian.
        gibbs_free_energy: Gibbs free energy (i.e. thermochemical analysis) in Hartrees
            of a system where translation / rotation / vibration degrees of freedom are
            approximated using ideal gas / rigid rotor / harmonic oscillator
            respectively.
        scf_dipole_moment: The x, y, z component of the dipole moment of the structure
            in units of e a0 (NOT Debye!).
        atomic_charges: Partial charge on each atom in units of e, ordered like the
            structure's atoms. The partitioning scheme (Mulliken, Lowdin, CM5, ...)
            is the program's; parsers record it in ``extras`` when it is not the
            program's default.
        point_charge_gradient: n_charges x 3 gradient of the energy with respect
            to the positions of external point charges, in `Hartrees/Bohr`, ordered
            like the point charges given to the calculation (QM/MM electrostatic
            embedding).

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

    atomic_charges: SerializableVector | None = None
    point_charge_gradient: SerializableMatrix | None = None

    @field_validator("normal_modes_cartesian")
    @classmethod
    def _validate_normal_modes_cartesian_shape(
        cls, v: SerializableNDArray | None, info: ValidationInfo
    ) -> np.ndarray | None:
        if v is not None:
            freqs = info.data.get("freqs_wavenumber")
            n_normal_modes = len(freqs) if freqs else len(v)
            return np.asarray(v).reshape(n_normal_modes, -1, 3)
        return v

    @field_validator("gradient")
    @classmethod
    def _validate_gradient_shape(
        cls, v: SerializableNDArray | None
    ) -> np.ndarray | None:
        """Validate gradient is n x 3."""
        if v is not None:
            return np.asarray(v).reshape(-1, 3)
        return None

    @field_validator("atomic_charges")
    @classmethod
    def _validate_atomic_charges_shape(
        cls, v: SerializableNDArray | None
    ) -> np.ndarray | None:
        """Validate atomic charges are one value per atom."""
        if v is not None:
            return np.asarray(v, dtype=float).reshape(-1)
        return None

    @field_validator("point_charge_gradient")
    @classmethod
    def _validate_point_charge_gradient_shape(
        cls, v: SerializableNDArray | None
    ) -> np.ndarray | None:
        """Validate the point-charge gradient is n_charges x 3."""
        if v is not None:
            return np.asarray(v, dtype=float).reshape(-1, 3)
        return None

    @field_validator("hessian")
    @classmethod
    def _validate_hessian_shape(
        cls, v: SerializableNDArray | None
    ) -> np.ndarray | None:
        """Validate hessian is square."""
        if v is not None:
            v = np.asarray(v)
            n = int(np.sqrt(v.size))
            return v.reshape((n, n))
        return None

    def return_result(self, calctype: CalcType) -> float | SerializableNDArray | None:
        """Return the primary result of the calculation."""
        return getattr(self, calctype.value)


class Frame(OutputBase):
    """One structure inside a larger calculation, with what was computed there.

    A step of an optimization, a point of an IRC, an image of a path. It is a
    `ProgramOutput` without `input_data`: the parent calculation's input already
    holds the program, model, and keywords, so a frame keeps only its structure.
    ``ProgramInput.from_spec(parent_spec, frame.structure)`` rebuilds a step's input.

    Attributes:
        structure: The structure of this step.
        results: What was computed at it; an empty `SinglePointData` (with the
            producing program's provenance) when nothing was.
    """

    structure: Structure
    results: SinglePointData

    @model_validator(mode="before")
    @classmethod
    def _from_program_output(cls, value: Any) -> Any:
        """Accept a ProgramOutput, or its saved form, as a frame.

        Trajectories saved before frames existed hold one ProgramOutput per step; the
        step's structure is its input's structure.
        """
        from .outputs import ProgramOutput

        if isinstance(value, ProgramOutput):
            value = {
                "success": value.success,
                "results": value.results,
                "logs": value.logs,
                "traceback": value.traceback,
                "execution": value.execution,
                "structure": value.input_data.structure,
            }
        if isinstance(value, dict) and "data" in value and "results" not in value:
            warnings.warn(
                "'data' has been renamed to 'results'. Please update your code accordingly.",
                FutureWarning,
                stacklevel=2,
            )
            value = {**value, "results": value["data"]}
            del value["data"]
        if isinstance(value, dict) and "input_data" in value:
            value = dict(value)
            input_data = value.pop("input_data")
            if "structure" not in value:
                value["structure"] = (
                    input_data["structure"]
                    if isinstance(input_data, dict)
                    else input_data.structure
                )
        return value

    @classmethod
    def from_output(cls, output: ProgramOutput[ProgramInput, SinglePointData]) -> Self:
        """Record a child calculation's output as a step, at its input's structure."""
        return cls.model_validate(output)


class _FrameSeries:
    """Reading a sequence of frames: structures, energies, xyz, and saving."""

    def _series(self) -> list[Frame]:
        raise NotImplementedError  # pragma: no cover

    @property
    def structures(self) -> list[Structure]:
        """The structure of each frame, in order."""
        return [frame.structure for frame in self._series()]

    @property
    def energies(self) -> np.ndarray:
        """The energy of each frame in Hartree, `np.nan` where none was computed."""
        return np.array(
            [
                frame.results.energy if frame.results.energy is not None else np.nan
                for frame in self._series()
            ],
            dtype=float,
        )

    def to_xyz(self) -> str:
        """Return the frames as a multi-structure `xyz` string."""
        return to_multi_xyz(self.structures)

    def save(
        self,
        filepath: Path | str,
        exclude_none: bool = True,
        exclude_unset: bool = True,
        indent: int = 4,
        **kwargs: Any,
    ) -> None:
        """Save to a file; a `.xyz` extension writes the frames' structures.

        Args:
            filepath: The path to write.
            exclude_none: If True, attributes with a value of None are not written.
            exclude_unset: If True, attributes that have not been set are not written.
                Defaults to True for compact files.
            indent: The number of spaces to use for indentation in JSON and YAML
                files. Ignored for `.xyz` files.
            **kwargs: Additional keyword arguments to pass to the json serializer.
        """
        filepath = Path(filepath)
        if filepath.suffix == ".xyz":
            filepath.write_text(self.to_xyz())
            return
        super().save(filepath, exclude_none, exclude_unset, indent, **kwargs)  # type: ignore[misc]


def _check_same_atoms(structures: list[Structure], what: str) -> None:
    """Raise if the structures do not share their atoms in the same order."""
    if structures and any(s.symbols != structures[0].symbols for s in structures):
        raise ValueError(f"Every {what} must have the same atoms in the same order.")


class OptimizationData(_FrameSeries, FileData, CalcInfoData):
    """Computed data for an optimization (may be for a minimum or transition state).

    Attributes:
        trajectory: One `Frame` per optimizer step: its structure and what was
            computed there.
        energies: `@property` The energy of each step.
        structures: `@property` The structure of each step.
        final_structure: `@property` The final, optimized structure.
    """

    trajectory: list[Frame] = []

    def _series(self) -> list[Frame]:
        return self.trajectory

    @property
    def final_structure(self) -> Structure | None:
        """The last evaluated structure, or None when there are no steps."""
        return self.trajectory[-1].structure if self.trajectory else None

    @property
    def final_energy(self) -> float | None:
        """The final step's energy, None for no steps, or `np.nan` if unavailable."""
        return self.energies[-1] if self.trajectory else None

    def return_result(self, calctype: CalcType) -> Structure | None:
        """Return the primary result of the optimization.

        Args:
            calctype: The calculation type. Unused; an optimization's primary result
                is always its final structure.

        Returns:
            The final structure, or None if the trajectory is empty.
        """
        return self.final_structure

    def __repr_args__(self) -> list[tuple[str, str]]:
        """Custom repr to avoid printing the entire collection of objects."""
        return [
            ("final_structure", f"{self.final_structure}"),
            ("trajectory", "[...]"),
            ("energies", "[...]"),
            ("structures", "[...]"),
        ]


class PathData(_FrameSeries, FileData):
    """A path of structures through a reaction: an IRC, or a chain of images.

    For an IRC the frames run from the end of the reverse branch, through the
    transition state, to the end of the forward branch. For a chain (NEB, string,
    growing string) they are the images in order, endpoints included.

    Attributes:
        frames: The structures along the path, in order, with what was computed at
            each.
        ts_index: The position of the transition state in `frames`, if the path has
            one: the starting saddle point of an IRC, or the highest (climbing) image
            of a chain.
        energies: `@property` The energy of each frame.
        structures: `@property` The structure of each frame.
    """

    frames: list[Frame] = []
    ts_index: int | None = None

    def _series(self) -> list[Frame]:
        return self.frames

    @model_validator(mode="after")
    def _check_path(self) -> Self:
        """Frames share their atoms; the transition state is one of them."""
        _check_same_atoms(self.structures, "frame of a path")
        if self.ts_index is not None and not 0 <= self.ts_index < len(self.frames):
            raise ValueError(
                f"ts_index {self.ts_index} is outside the {len(self.frames)} frames."
            )
        return self

    @property
    def transition_state(self) -> Frame | None:
        """The frame at `ts_index`, or None."""
        return self.frames[self.ts_index] if self.ts_index is not None else None

    @property
    def forward_barrier(self) -> float | None:
        """The transition state's energy above the first frame, in Hartree."""
        return self._barrier(0)

    @property
    def reverse_barrier(self) -> float | None:
        """The transition state's energy above the last frame, in Hartree."""
        return self._barrier(-1)

    def _barrier(self, end: int) -> float | None:
        if self.ts_index is None or not self.frames:
            return None
        energies = self.energies
        barrier = float(energies[self.ts_index] - energies[end])
        return None if np.isnan(barrier) else barrier

    def return_result(self, calctype: CalcType) -> list[Structure]:
        """Return the primary result: the structures along the path."""
        return self.structures

    def __repr_args__(self) -> list[tuple[str, str]]:
        """Avoid printing every frame."""
        return [
            ("frames", f"[{len(self.frames)} frames]"),
            ("ts_index", f"{self.ts_index}"),
        ]


class PathOptimizationData(FileData):
    """Computed data for a path optimization (NEB, string, growing string, ...).

    Like an optimization's trajectory of steps, but each step is a whole path.

    Attributes:
        trajectory: The path at each iteration, the optimized one last.
        final_path: `@property` The last path, or None when there are none.
        energies: `@property` The energy of each frame of the final path.
        structures: `@property` The structures of the final path.
    """

    trajectory: list[PathData] = []

    @property
    def final_path(self) -> PathData | None:
        """The optimized (last) path, or None when there are no iterations."""
        return self.trajectory[-1] if self.trajectory else None

    @property
    def structures(self) -> list[Structure]:
        """The structures of the final path."""
        return self.final_path.structures if self.final_path else []

    @property
    def energies(self) -> np.ndarray:
        """The energies of the final path's frames, in Hartree."""
        return self.final_path.energies if self.final_path else np.array([])

    def to_xyz(self) -> str:
        """Return the final path as a multi-structure `xyz` string."""
        return to_multi_xyz(self.structures)

    def return_result(self, calctype: CalcType) -> PathData | None:
        """Return the primary result: the final path."""
        return self.final_path

    def __repr_args__(self) -> list[tuple[str, str]]:
        """Avoid printing every path."""
        return [("trajectory", f"[{len(self.trajectory)} paths]")]


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
    def structures(self) -> list[Structure]:
        """The conformers, lowest energy first (the shared reading interface)."""
        return self.conformers

    @property
    def energies(self) -> np.ndarray:
        """The conformers' energies in Hartree, lowest first."""
        return self.conformer_energies

    def to_xyz(self) -> str:
        """Return the conformers as a multi-structure `xyz` string."""
        return to_multi_xyz(self.conformers)

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

    Attributes:
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
        **kwargs: Any,
    ) -> None:
        """Save a ScanData to a file.

        Args:
            filepath: The path to save the molecule to.
            exclude_none: If True, attributes with a value of None will not be written
                to the file.
            exclude_unset: If True, attributes that have not been set will not be
                written to the file. Defaults to True for compact files.
            indent: The number of spaces to use for indentation in JSON and YAML
                files. Ignored for `.xyz` files.
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


StructuredData = (
    SinglePointData
    | OptimizationData
    | ConformerSearchData
    | ScanData
    | PathData
    | PathOptimizationData
)
StructuredDataType = TypeVar("StructuredDataType", bound=StructuredData)
Data = FileData | StructuredData
DataType = TypeVar("DataType", bound=Data)


_DATA_TYPES: dict[CalcType, type[StructuredData]] = {
    CalcType.energy: SinglePointData,
    CalcType.gradient: SinglePointData,
    CalcType.hessian: SinglePointData,
    CalcType.optimization: OptimizationData,
    CalcType.transition_state: OptimizationData,
    CalcType.conformer_search: ConformerSearchData,
    CalcType.scan: ScanData,
    CalcType.irc: PathData,
    CalcType.path_optimization: PathOptimizationData,
}


def get_data_type(calctype: CalcType | str) -> type[StructuredData]:
    """Return the scientific data class required by a calculation type.

    Accepts CalcType members or their string values. Unknown values raise ValueError.
    File-only execution uses FileData and has no calculation type.
    """
    return _DATA_TYPES[CalcType(calctype)]
