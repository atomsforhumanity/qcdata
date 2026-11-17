"""General-purpose scientific data models."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self, TypeVar

import numpy as np
from pydantic import BaseModel, ValidationInfo, field_validator, model_validator

from qcdata.helper_types import (
    SerializableMatrix,
    SerializableNDArray,
    SerializableVector,
)

from .base_models import CalcType, Files, OutputBase, Provenance, QCDataBaseModel
from .inputs import ProgramInput
from .orbitals import Orbitals
from .structure import Structure
from .utils import to_multi_xyz
from .vibrations import Thermochemistry, VibrationalAnalysis

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
        orbitals: The molecular orbitals with the basis set they expand in, when the
            input asked for them (they are large: nao x nmo per spin).
    """

    scf_eigenvalues_a: SerializableVector | None = None
    scf_eigenvalues_b: SerializableVector | None = None
    scf_occupations_a: SerializableVector | None = None
    scf_occupations_b: SerializableVector | None = None
    orbitals: Orbitals | None = None

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

        scf_dipole_moment: The x, y, z component of the dipole moment of the structure
            in units of e a0 (NOT Debye!).
        atomic_charges: Partial charge on each atom in units of e, ordered like the
            structure's atoms. The partitioning scheme (Mulliken, Lowdin, CM5, ...)
            is the program's; `atomic_charges_kind` names it.
        atomic_charges_kind: The partitioning scheme of `atomic_charges`:
            ``mulliken``, ``lowdin``, ``eeq``, ``eeqbc``, ``cm5``, ``hirshfeld``, ...
        point_charge_gradient: n_charges x 3 gradient of the energy with respect
            to the positions of external point charges, in `Hartrees/Bohr`, ordered
            like the point charges given to the calculation (QM/MM electrostatic
            embedding).

        energy_components: Terms of the energy (Hartrees) that add up to `energy`: a
            partition, never overlapping terms. Standard keys are listed in the
            conventions (``nuclear_repulsion``, ``one_electron``, ``coulomb``,
            ``exchange``, ``xc``, ``vv10``, ``dispersion``, ``solvation_electrostatic``,
            ``solvation_cds``, ...); what a component contains is the program's (a
            tight-binding ``electronic`` term is not a DFT one). A hierarchy of
            sub-terms, if a program has one, belongs in the output's ``record``.
        scf_iterations: Number of SCF iterations of the final SCF.
        scf_s_squared: Expectation value of S^2 of the SCF reference (dimensionless);
            S(S+1) for a pure spin state, so the excess is the spin contamination.
        scf_quadrupole_moment: 3 x 3 quadrupole moment (e bohr^2) about the coordinate
            origin, in the form `quadrupole_kind` names; programs with another form
            convert to one of those.
        quadrupole_kind: ``traceless``: Buckingham's
            (1/2) sum q (3 r_i r_j - r^2 delta_ij); ``second_moment``: sum q r_i r_j.
        polarizability: 3 x 3 static dipole polarizability (e^2 bohr^2 / Hartree).
        bond_orders: n_atoms x n_atoms bond orders, in the form `bond_orders_kind`
            names.
        bond_orders_kind: ``mayer`` (sum (PS)_mn (PS)_nm over the AOs of two atoms,
            what xtb prints as "Wiberg"), ``wiberg`` (in an orthogonalized basis), ...
        vibrational_analysis: The harmonic analysis of the Hessian with the choices
            behind it (masses, projection, symmetry), when the program made one: the
            home of frequencies and normal modes.
        thermochemistry: Ideal-gas thermochemistry, one entry per temperature and
            standard state: the home of free energies.
    """

    energy: float | None = None
    gradient: SerializableMatrix | None = None
    hessian: SerializableMatrix | None = None
    nuclear_repulsion_energy: float | None = None

    wavefunction: Wavefunction | None = None

    scf_dipole_moment: list[float] | None = None

    atomic_charges: SerializableVector | None = None
    atomic_charges_kind: str | None = None
    point_charge_gradient: SerializableMatrix | None = None

    energy_components: dict[str, float] = {}
    scf_iterations: int | None = None
    scf_s_squared: float | None = None
    scf_quadrupole_moment: SerializableMatrix | None = None
    quadrupole_kind: str | None = None
    polarizability: SerializableMatrix | None = None
    bond_orders: SerializableMatrix | None = None
    bond_orders_kind: str | None = None

    vibrational_analysis: VibrationalAnalysis | None = None
    thermochemistry: list[Thermochemistry] = []

    @model_validator(mode="before")
    @classmethod
    def _drop_empty_pre_024_fields(cls, values: Any) -> Any:
        """Load files saved before 0.24, which wrote empty frequency fields.

        Frequencies, normal modes and free energies now live only in
        `vibrational_analysis` and `thermochemistry`. Empty old fields carry nothing
        and are dropped; filled ones cannot be converted (their masses and model
        are unknown) and are rejected with that reason.
        """
        if not isinstance(values, dict):
            return values
        old = ("freqs_wavenumber", "normal_modes_cartesian", "gibbs_free_energy")
        present = [k for k in old if k in values]
        if not present:
            return values
        filled = [k for k in present if values[k] not in (None, [], ())]
        if filled:
            raise ValueError(
                f"{', '.join(filled)} were removed in qcdata 0.24: frequencies and normal "
                "modes live in vibrational_analysis, free energies in thermochemistry. "
                "Recompute the Hessian, or rebuild the analysis with "
                "qcinf.vibrational_analysis."
            )
        return {k: v for k, v in values.items() if k not in old}

    @field_validator("scf_quadrupole_moment", "polarizability")
    @classmethod
    def _validate_three_by_three(
        cls, v: SerializableNDArray | None
    ) -> np.ndarray | None:
        """Validate a Cartesian tensor is 3 x 3."""
        if v is not None:
            return np.asarray(v, dtype=float).reshape(3, 3)
        return None

    @field_validator("bond_orders")
    @classmethod
    def _validate_bond_orders_shape(
        cls, v: SerializableNDArray | None
    ) -> np.ndarray | None:
        """Validate bond orders are square."""
        if v is not None:
            v = np.asarray(v, dtype=float)
            n = int(round(np.sqrt(v.size)))
            return v.reshape((n, n))
        return None

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
                "warnings": value.warnings,
                "error": value.error,
                "record": value.record,
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
            # A whole calculation's settings; its parent keeps its own.
            value.pop("resolved", None)
            input_data = value.pop("input_data")
            if "structure" not in value:
                value["structure"] = (
                    input_data["structure"]
                    if isinstance(input_data, dict)
                    else input_data.structure
                )
        return value

    @classmethod
    def from_output(cls, output: ProgramOutput[Any, Any]) -> Self:
        """Record a child calculation's output as a step, at its input's structure.

        Raises:
            ValidationError: The output's results are not `SinglePointData`, or its
                input has no single structure.
        """
        return cls.model_validate(output)


def _frame_energies(frames: list[Frame]) -> np.ndarray:
    """The energy of each frame in Hartree, `np.nan` where none was computed."""
    return np.array(
        [np.nan if f.results.energy is None else f.results.energy for f in frames],
        dtype=float,
    )


def _relative(energies: np.ndarray) -> np.ndarray:
    """Energies above the lowest known one; all `np.nan` when none is known."""
    if np.isnan(energies).all():
        return energies
    return energies - np.nanmin(energies)


class _FrameSeries:
    """Reading a sequence of frames: structures, energies, xyz, and saving.

    Every result holding frames reads the same way; ``_series`` names its frames: an
    optimization's steps, a path's points, a path optimization's final path, a
    conformer search's conformers.
    """

    def _series(self) -> list[Frame]:
        raise NotImplementedError  # pragma: no cover

    @property
    def structures(self) -> list[Structure]:
        """The structure of each frame, in order."""
        return [frame.structure for frame in self._series()]

    @property
    def energies(self) -> np.ndarray:
        """The energy of each frame in Hartree, `np.nan` where none was computed."""
        return _frame_energies(self._series())

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
    ) -> list[Path]:
        """Save to a file; a `.xyz` extension writes the frames' structures.

        Args:
            filepath: The path to write.
            exclude_none: If True, attributes with a value of None are not written.
            exclude_unset: If True, attributes that have not been set are not written.
                Defaults to True for compact files.
            indent: The number of spaces to use for indentation in JSON and YAML
                files. Ignored for `.xyz` files.
            **kwargs: Additional keyword arguments to pass to the json serializer.

        Returns:
            Every file written, as `QCDataBaseModel.save` returns them.
        """
        filepath = Path(filepath)
        if filepath.suffix == ".xyz":
            filepath.write_text(self.to_xyz())
            return [filepath]
        return super().save(filepath, exclude_none, exclude_unset, indent, **kwargs)  # type: ignore[misc]


def _renamed(payload: Any, old: str, new: str) -> Any:
    """Move a payload's deprecated field name to its new one, with a warning."""
    if not isinstance(payload, dict) or old not in payload:
        return payload
    if new in payload:
        raise ValueError(f"Provide only {new!r}; {old!r} is its deprecated name.")
    warnings.warn(
        f"'{old}' has been renamed to '{new}'. Please update your code accordingly.",
        FutureWarning,
        stacklevel=4,
    )
    payload = dict(payload)
    payload[new] = payload.pop(old)
    return payload


def _check_same_atoms(structures: list[Structure], what: str) -> None:
    """Raise if the structures do not share their atoms in the same order."""
    if structures and any(s.symbols != structures[0].symbols for s in structures):
        raise ValueError(f"Every {what} must have the same atoms in the same order.")


class OptimizationData(_FrameSeries, FileData, CalcInfoData):
    """Computed data for an optimization (may be for a minimum or transition state).

    Attributes:
        frames: One `Frame` per optimizer step: its structure and what was computed
            there. Files saved with the former name, `trajectory`, still load.
        energies: `@property` The energy of each step.
        structures: `@property` The structure of each step.
        final_structure: `@property` The final, optimized structure.
    """

    frames: list[Frame] = []

    @model_validator(mode="before")
    @classmethod
    def _migrate_trajectory(cls, payload: Any) -> Any:
        """Accept the former field name ``trajectory`` for ``frames``."""
        return _renamed(payload, "trajectory", "frames")

    def _series(self) -> list[Frame]:
        return self.frames

    @property
    def final_structure(self) -> Structure | None:
        """The last evaluated structure, or None when there are no steps."""
        return self.frames[-1].structure if self.frames else None

    @property
    def final_energy(self) -> float | None:
        """The final step's energy, None for no steps, or `np.nan` if unavailable."""
        return self.energies[-1] if self.frames else None

    def __repr_args__(self) -> list[tuple[str, str]]:
        """Custom repr to avoid printing the entire collection of objects."""
        return [
            ("final_structure", f"{self.final_structure}"),
            ("frames", "[...]"),
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

    def __repr_args__(self) -> list[tuple[str, str]]:
        """Avoid printing every frame."""
        return [
            ("frames", f"[{len(self.frames)} frames]"),
            ("ts_index", f"{self.ts_index}"),
        ]


class PathOptimizationData(_FrameSeries, FileData):
    """Computed data for a path optimization (NEB, string, growing string, ...).

    Like an optimization's frames, but each step is a whole path. The shared reading
    interface (``structures``, ``energies``, ``to_xyz``, ``save("*.xyz")``) reads the
    final path.

    Attributes:
        paths: The path at each iteration, the optimized one last.
        final_path: `@property` The last path, or None when there are none.
        energies: `@property` The energy of each frame of the final path.
        structures: `@property` The structures of the final path.
    """

    paths: list[PathData] = []

    def _series(self) -> list[Frame]:
        return self.final_path.frames if self.final_path else []

    @property
    def final_path(self) -> PathData | None:
        """The optimized (last) path, or None when there are no iterations."""
        return self.paths[-1] if self.paths else None

    def __repr_args__(self) -> list[tuple[str, str]]:
        """Avoid printing every path."""
        return [("paths", f"[{len(self.paths)} paths]")]


class ConformerSearchData(_FrameSeries, FileData):
    """Data from a conformer search calculation.

    Conformers and rotamers are `Frame`s, sorted by energy (lowest first; frames
    without an energy last). Constructing or loading with lists of structures and
    separate energy arrays (``conformer_energies``, ``rotamer_energies``), the form
    before frames, still works: each structure becomes a frame with its energy. The
    shared reading interface (``structures``, ``energies``, ``to_xyz``,
    ``save("*.xyz")``) reads the conformers.

    Attributes:
        conformers: The conformers found, each with what was computed for it.
        rotamers: The rotamers found, each with what was computed for it.
        structures: `@property` The conformers' structures.
        energies: `@property` The conformers' energies in Hartree, one per conformer,
            `np.nan` where none was computed.
        conformer_energies: `@property` The same as ``energies``.
        rotamer_energies: `@property` The rotamers' energies, read the same way.
    """

    conformers: list[Frame] = []
    rotamers: list[Frame] = []

    @model_validator(mode="before")
    @classmethod
    def _frames_from_structures(cls, payload: Any) -> Any:
        """Turn structures plus an energies array into frames."""
        if not isinstance(payload, dict):
            return payload
        payload = dict(payload)
        provenance = payload.get("provenance")
        for kind in ("conformers", "rotamers"):
            items = list(payload.get(kind, []))
            energies = payload.pop(f"{kind[:-1]}_energies", None)
            energies = [] if energies is None else list(np.asarray(energies).ravel())
            if energies and len(energies) != len(items):
                raise ValueError(
                    f"The number of {kind[:-1]} energies must match the number of "
                    f"{kind}."
                )
            if not items or not _is_structure(items[0]):
                continue
            payload[kind] = [
                {
                    "success": True,
                    "structure": item,
                    "results": {
                        "provenance": provenance,
                        "energy": float(energies[i]) if energies else None,
                    },
                }
                for i, item in enumerate(items)
            ]
        return payload

    @model_validator(mode="after")
    def _sort_by_energy(self) -> Self:
        """Sort conformers and rotamers by energy; frames without one go last."""
        for frames in (self.conformers, self.rotamers):
            frames.sort(
                key=lambda f: (f.results.energy is None, f.results.energy or 0.0)
            )
        return self

    def _series(self) -> list[Frame]:
        return self.conformers

    @property
    def conformer_energies(self) -> np.ndarray:
        """The conformers' energies in Hartree, lowest first; `np.nan` if unknown."""
        return self.energies

    @property
    def rotamer_energies(self) -> np.ndarray:
        """The rotamers' energies in Hartree, lowest first; `np.nan` if unknown."""
        return _frame_energies(self.rotamers)

    @property
    def conformer_energies_relative(self) -> np.ndarray:
        """Each conformer's energy above the lowest, in Hartree; `np.nan` if unknown."""
        return _relative(self.conformer_energies)

    @property
    def rotamer_energies_relative(self) -> np.ndarray:
        """Each rotamer's energy above the lowest, in Hartree; `np.nan` if unknown."""
        return _relative(self.rotamer_energies)


def _is_structure(item: Any) -> bool:
    """Whether an item is a structure (or its dict), not a frame."""
    if isinstance(item, Structure):
        return True
    return isinstance(item, dict) and "symbols" in item and "structure" not in item


class ScanData(FileData, CalcInfoData):
    """Computed data for a scan (may be for a relaxed or frozen).

    Attributes:
    ----------
        energies: The energies for each step of the scan.
        structures: The Structure objects for each step of the scan.
        points: One constrained optimization per scan point, as its full
            `ProgramOutput` (each point's input carries its own constraint value).
            Files saved with the former name, `trajectory`, still load.
    """

    points: list[ProgramOutput[ProgramInput, OptimizationData]] = []

    @model_validator(mode="before")
    @classmethod
    def _migrate_trajectory(cls, payload: Any) -> Any:
        """Accept the former field name ``trajectory`` for ``points``."""
        return _renamed(payload, "trajectory", "points")

    @property
    def energies(self) -> np.ndarray:
        """The energies for each step of the scan."""
        return np.array(
            [
                output.results.final_energy
                if output.results.final_energy is not None
                else np.nan
                for output in self.points
            ],
            dtype=float,
        )

    @property
    def structures(self) -> list[Structure | None]:
        """The final structure at each scan point, or None for an empty point."""
        return [output.results.final_structure for output in self.points]

    def __repr_args__(self) -> list[tuple[str, str]]:
        """Avoid printing the entire collection of objects in representation."""
        return [
            ("points", "[...]"),
            ("energies", "[...]"),
            ("structures", "[...]"),
        ]

    def to_xyz(self) -> str:
        """Return the final structure of each scan point as an `xyz` string."""
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
    ) -> list[Path]:
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

        Returns:
            Every file written, as `QCDataBaseModel.save` returns them.

        Note:
            If the filepath has a `.xyz` extension, each scan point's final structure is saved to a
            multi-structure `xyz` file.
        """
        filepath = Path(filepath)
        if filepath.suffix == ".xyz":
            filepath.write_text(self.to_xyz())
            return [filepath]
        return super().save(filepath, exclude_none, exclude_unset, indent, **kwargs)


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
