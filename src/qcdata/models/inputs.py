"""Input models for quantum chemistry calculations."""

from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self, TypeVar

from pydantic import (
    BaseModel,
    FieldSerializationInfo,
    field_serializer,
    field_validator,
    model_validator,
)

from ..helper_types import FreeDict
from .base_models import CalcType, Files, Model
from .structure import Structure

if TYPE_CHECKING:
    from .data import SinglePointData

__all__ = [
    "INITIAL_CHAIN",
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
    def from_directory(cls, directory: Path | str, **kwargs: Any) -> Self:
        """Collect directory files, passing required `program` and other fields as kwargs."""
        obj = cls(**kwargs)
        directory = Path(directory)
        obj.add_files(directory)
        return obj


def _json_models(value: Any) -> Any:
    """A free-form value with each pydantic model in it dumped as JSON, nulls left out.

    Dumped through `BaseModel.model_dump`, so a qcdata model carries no version stamp.
    Returns ``value`` itself when it holds no model.
    """
    if isinstance(value, BaseModel):
        return BaseModel.model_dump(value, mode="json", exclude_none=True)
    if isinstance(value, dict):
        items = {k: _json_models(v) for k, v in value.items()}
        changed = any(items[k] is not v for k, v in value.items())
        return items if changed else value
    if isinstance(value, list | tuple):
        new = [_json_models(v) for v in value]
        if all(a is b for a, b in zip(new, value, strict=True)):
            return value
        return new if isinstance(value, list) else tuple(new)
    return value


class ProgramSpec(FileInput):
    """A recursive program specification, independent of structures.

    Each child describes its own calculation. Siblings must have distinct calculation
    types; the same calculation type may appear at different levels of the tree.

    Attributes:
        program: The requested executor (distinct from result data provenance).
        calctype: The type of calculation to perform.
        model: The scientific model, or None for programs without one.
        keywords: Program keywords, excluding model and calculation type. Plain
            JSON: a pydantic model given anywhere in them (an `Accuracy`) is stored
            as its JSON form without nulls, so the input and its calculation key are
            the same before and after a save.
        subprograms: Child specifications, with one child per calculation type.
        files: Native input files for this program.
        cmdline_args: Command line arguments for this program.
        extras: Additional information to bundle with this specification.
    """

    calctype: CalcType
    model: Model | None = None
    keywords: FreeDict = {}
    subprograms: list["ProgramSpec"] = []

    @field_validator("keywords")
    @classmethod
    def _models_as_json(cls, keywords: dict[str, Any]) -> dict[str, Any]:
        """Store any pydantic model in the keywords, at any depth, as its JSON form."""
        return _json_models(keywords)

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
    def _serialize_calctype(
        self, calctype: CalcType, _info: FieldSerializationInfo
    ) -> str:
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


INITIAL_CHAIN = "initial_chain"
"""Role in ``ProgramInput.structures`` holding the starting chain of a path optimization."""
# Calculation types whose subject is a chain of structures rather than one structure.
CHAIN_CALCTYPES = frozenset({CalcType.path_optimization})


class ProgramInput(ProgramSpec):
    """A program specification bound to structures.

    Most calculations are about one structure, ``structure``. A path optimization is
    about a chain: its structures, endpoints included and in order, go in
    ``structures["initial_chain"]`` and ``structure`` is left out. Validation enforces
    this by calculation type, so ``structure`` is always set for every other one; use
    ``require_structure()`` where a type checker needs to know that.

    Attributes:
        structure: The primary/start/reference structure; None only for chain
            calculation types (``path_optimization``).
        structures: Additional complete structures identified by role, such as a
            QM/MM environment, or a list of structures for a role that holds several
            in order, such as ``"initial_chain"``.
        given: Results already known at ``structure``, for the program to start
            from: an optimizer or IRC takes a given Hessian as its starting Hessian; an
            SCF starts from a given wavefunction. Starting points, never results: a
            program does not report a given value as its own. A record of what is
            known, not a request: a program uses the fields it can, records each one
            used in its resolved settings (``"given"``), and leaves the rest alone, so
            ``given=frame.results`` works as it stands. It refuses the input
            (``not_applicable``) when it can use none of the fields, or when a field
            it would use does not fit (another basis). ``provenance`` names only the
            producing program and version; checking the model is the caller's, which
            knows the producing input. Part of the input, so part of a stored
            calculation's key.

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

    structure: Structure | None = None
    structures: dict[str, Structure | list[Structure]] = {}
    given: "SinglePointData | None" = None

    @model_validator(mode="after")
    def _check_given(self) -> Self:
        """Given results belong to ``structure``: the same atoms, in its shapes.

        A given wavefunction may come from another charge or multiplicity (a guess),
        so its electrons are not checked.
        """
        if self.given is None:
            return self
        if self.structure is None:
            raise ValueError(
                "given results belong to a structure; this input has none."
            )
        n = len(self.structure.symbols)
        shapes = {"gradient": (n, 3), "hessian": (3 * n, 3 * n)}
        for name, shape in shapes.items():
            value = getattr(self.given, name)
            if value is not None and value.shape != shape:
                raise ValueError(
                    f"given.{name} has shape {value.shape}; the structure's {n} atoms "
                    f"need {shape}."
                )
        from .data import _check_fits_structure

        _check_fits_structure(self.given, self.structure, electrons=False)
        return self

    @model_validator(mode="after")
    def _check_structures_for_calctype(self) -> Self:
        """A chain calculation has a chain; every other one has a structure."""
        if self.calctype in CHAIN_CALCTYPES:
            if self.structure is not None:
                raise ValueError(
                    f"A {self.calctype.value} input takes its structures from "
                    f"structures[{INITIAL_CHAIN!r}]; leave structure out."
                )
            chain = self.structures.get(INITIAL_CHAIN)
            if not isinstance(chain, list) or len(chain) < 2:
                raise ValueError(
                    f"A {self.calctype.value} input needs structures"
                    f"[{INITIAL_CHAIN!r}]: a list of at least two structures, "
                    "endpoints included, in order."
                )
            if any(s.symbols != chain[0].symbols for s in chain):
                raise ValueError(
                    f"Every structure in structures[{INITIAL_CHAIN!r}] must have the "
                    "same atoms in the same order."
                )
        elif self.structure is None:
            raise ValueError(f"A {self.calctype.value} input requires a structure.")
        return self

    def require_structure(self) -> Structure:
        """Return ``structure``, or raise if this input has none (a chain input).

        Validation guarantees a structure for every calculation type except chain
        types, so this only narrows the type for type checkers and gives a clear
        error if a chain input reaches code that needs one structure.

        Raises:
            ValueError: The input carries a chain instead of one structure.
        """
        if self.structure is None:
            raise ValueError(
                f"This {self.calctype.value} input has no single structure; its "
                f"structures are in structures[{INITIAL_CHAIN!r}]."
            )
        return self.structure

    @property
    def initial_chain(self) -> list[Structure]:
        """The starting chain of a path optimization, endpoints included, in order.

        Raises:
            ValueError: The input has no chain.
        """
        chain = self.structures.get(INITIAL_CHAIN)
        if not isinstance(chain, list):
            raise ValueError(
                f"This {self.calctype.value} input has no structures"
                f"[{INITIAL_CHAIN!r}]."
            )
        return chain

    @classmethod
    def from_spec(
        cls,
        spec: ProgramSpec,
        structure: Structure | None = None,
        *,
        structures: Mapping[str, Structure | list[Structure]] | None = None,
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
            values["structures"] = dict(structures)
        return cls.model_validate(values)


Inputs = FileInput | ProgramInput
InputType = TypeVar("InputType", bound=Inputs)
