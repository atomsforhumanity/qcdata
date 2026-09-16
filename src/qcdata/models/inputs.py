"""Input models for quantum chemistry calculations."""

from pathlib import Path
from typing import Any, TypeVar

from pydantic import field_serializer, field_validator
from typing_extensions import Self

from .base_models import CalcType, Files, Model
from .structure import Structure

__all__ = [
    "FileInput",
    "ProgramInput",
    "ProgramSpec",
    "Inputs",
    "InputType",
]


class FileInput(Files):
    """File and command line argument inputs for a calculation.

    Attributes:
        program: The requested executor for the calculation.
        files Files: A dict mapping filename to str or bytes data.
        cmdline_args: A list of command line arguments to be passed to the program.
        extras Dict[str, Any]: Additional information to bundle with the object. Use for
            schema development and scratch space.
    """

    program: str
    cmdline_args: list[str] = []

    @classmethod
    def from_directory(cls, directory: Path | str, **kwargs) -> Self:
        """Collect directory files, passing required `program` and other fields as kwargs."""
        obj = cls(**kwargs)
        directory = Path(directory)
        obj.add_files(directory)
        return obj


class ProgramSpec(FileInput):
    """A recursive program specification, independent of structures.

    Each child describes its own calculation. Siblings must have distinct calculation
    types; the same calculation type may appear at different levels of the tree.

    Attributes:
        program: The requested executor (distinct from result data provenance).
        calctype: The type of calculation to perform.
        model: The scientific model, or None for programs without one.
        keywords: Program keywords, excluding model and calculation type.
        subprograms: Child specifications, with one child per calculation type.
        files: Native input files for this program.
        cmdline_args: Command line arguments for this program.
        extras: Additional information to bundle with this specification.
    """

    calctype: CalcType
    model: Model | None = None
    keywords: dict[str, Any] = {}
    subprograms: list["ProgramSpec"] = []

    @field_validator("subprograms")
    @classmethod
    def _unique_subprogram_calctypes(
        cls, subprograms: list["ProgramSpec"]
    ) -> list["ProgramSpec"]:
        seen: set[CalcType] = set()
        for child in subprograms:
            if child.calctype in seen:
                raise ValueError(
                    f"Duplicate subprogram calculation type '{child.calctype.value}': "
                    "each sibling must have a unique calctype."
                )
            seen.add(child.calctype)
        return subprograms

    @field_serializer("calctype")
    def _serialize_calctype(self, calctype: CalcType, _info) -> str:
        """Serialize CalcType to string."""
        return calctype.value

    def get_subprogram(self, calctype: CalcType | str) -> "ProgramSpec":
        """Return an immediate child by CalcType or string value.

        Raises ValueError if no matching immediate child exists.
        """
        for child in self.subprograms:
            if child.calctype == calctype:
                return child
        raise ValueError(
            f"No immediate subprogram with calculation type {calctype!r} "
            f"for program '{self.program}'."
        )


class ProgramInput(ProgramSpec):
    """A program specification bound to structures.

    Attributes:
        structure: The required primary/start/reference structure.
        structures: Additional complete structures identified by role, such as the
            product endpoint of a nudged elastic band calculation.

    Example:
        ```python
        from qcdata import ProgramInput, ProgramSpec, Structure

        prog_input = ProgramInput(
            program="geometric",
            calctype="optimization",
            structure=Structure.open("structure.xyz"),
            keywords={"maxiter": 250},
            subprograms=[
                ProgramSpec(
                    program="terachem",
                    calctype="gradient",
                    model={"method": "wb97x-d3", "basis": "def2-svp"},
                ),
            ],
        )
        ```
    """

    structure: Structure
    structures: dict[str, Structure] = {}

    @classmethod
    def from_spec(
        cls,
        spec: ProgramSpec,
        structure: Structure,
        *,
        structures: dict[str, Structure] | None = None,
    ) -> Self:
        """Bind a specification to primary and additional role-named structures.

        Preserve all specification fields, including recursive children, and validate
        the new input. The specification is not modified.
        """
        values = spec.model_dump(
            include=set(ProgramSpec.model_fields), exclude_unset=True
        )
        values["structure"] = structure
        if structures is not None:
            values["structures"] = structures
        return cls.model_validate(values)


Inputs = FileInput | ProgramInput
InputType = TypeVar("InputType", bound=Inputs)
