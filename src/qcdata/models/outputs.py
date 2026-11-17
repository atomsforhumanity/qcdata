"""Program output container objects."""

from __future__ import annotations

import sys
from typing import Any, Generic, Self, get_args

from pydantic import ValidationInfo, field_validator, model_validator

from .base_models import ErrorInfo, OutputBase, Resolved
from .data import (
    ConformerSearchData,
    DataType,
    FileData,
    OptimizationData,
    PathData,
    PathOptimizationData,
    ScanData,
    SinglePointData,
    StructuredData,
    _check_fits_structure,
    get_data_type,
)
from .inputs import FileInput, ProgramInput, ProgramSpec
from .inputs import InputType as ProgramInputType

__all__ = ["ProgramOutput"]


class ProgramOutput(OutputBase, Generic[ProgramInputType, DataType]):
    """The core output object from a quantum chemistry calculation.

    Success, logs, traceback, and execution come from `OutputBase`, shared with
    `Frame`.

    Attributes:
        input_data: The input data for the calculation. Any of `qcdata.Inputs`.
        results: Scientific data returned by the calculation, including output files.
            Any of `qcdata.Data`.
        resolved: The settings the calculation actually used, with every ``"auto"``
            replaced, and the reasons; passed back as keywords they reproduce the run.
    """

    input_data: ProgramInputType
    results: DataType
    resolved: Resolved | None = None

    @classmethod
    def failure(
        cls,
        input_data: ProgramInputType,
        error: ErrorInfo,
        *,
        results: Any = None,
        traceback: str | None = None,
        **fields: Any,
    ) -> ProgramOutput[ProgramInputType, Any]:
        """A failed output of the right shape for its input's calculation type.

        Programs return this instead of building empty results by hand: an
        ``optimization`` input needs `OptimizationData` results even when nothing ran.

        Args:
            input_data: The input that failed.
            error: Why it failed.
            results: Partial results, if any; else empty results of the calctype's
                data type with the program's provenance.
            traceback: The failure's details; defaults to the error's message.
            **fields: Other output fields (``logs``, ``execution``, ``warnings``,
                ``record``).

        Returns:
            The failed output.
        """
        from .base_models import Provenance
        from .data import get_data_type
        from .inputs import ProgramInput as _ProgramInput

        if results is None:
            data_type = (
                get_data_type(input_data.calctype)
                if isinstance(input_data, _ProgramInput)
                else FileData
            )
            results = data_type(provenance=Provenance(program=input_data.program))
        return ProgramOutput(
            input_data=input_data,
            results=results,
            success=False,
            error=error,
            traceback=traceback or error.message,
            **fields,
        )

    def model_post_init(self, __context: Any) -> None:
        """Parameterize the class (if not set explicitly)."""
        if self.__class__ is ProgramOutput:
            input_type = type(self.input_data)
            results_type = type(self.results)
            self.__class__ = self.__class__[input_type, results_type]  # type: ignore[index]

    @model_validator(mode="after")
    def _ensure_bound_program_spec(self) -> Self:
        """A structure-free specification is not a top-level calculation input."""
        if isinstance(self.input_data, ProgramSpec) and not isinstance(
            self.input_data, ProgramInput
        ):
            raise ValueError("ProgramSpec must be bound to structures as ProgramInput.")
        return self

    @field_validator("results", mode="before")
    @classmethod
    def _validate_result_type(cls, value: Any, info: ValidationInfo) -> Any:
        """Use the input contract to parse even completely empty result payloads."""
        input_data = info.data.get("input_data")
        if input_data is None:
            return value  # Let input validation report its own error.
        if cls.model_fields["input_data"].annotation is FileInput and isinstance(
            input_data, ProgramInput
        ):
            raise ValueError("Structured inputs require a ProgramInput output generic.")
        expected = (
            get_data_type(input_data.calctype)
            if isinstance(input_data, ProgramInput)
            else FileData
        )
        declared = cls.model_fields["results"].annotation
        if (
            isinstance(declared, type)
            and issubclass(declared, FileData)
            and declared is not expected
        ):
            raise ValueError(
                f"Input requires {expected.__name__}, but the output generic "
                f"specifies {declared.__name__}."
            )
        if isinstance(value, dict):
            return expected.model_validate(value, context=info.context)
        if type(value) is not expected:
            raise ValueError(
                f"Input requires {expected.__name__} results on both success and "
                f"failure; received {type(value).__name__}."
            )
        return value

    @model_validator(mode="after")
    def _check_results_fit_structure(self) -> Self:
        """Per-atom results and the orbitals' electrons fit the input's structure."""
        if (
            isinstance(self.input_data, ProgramInput)
            and self.input_data.structure is not None
            and isinstance(self.results, SinglePointData)
        ):
            _check_fits_structure(
                self.results, self.input_data.structure, electrons=True
            )
        return self

    @model_validator(mode="after")
    def _validate_completion(self) -> Self:
        if not self.success:
            return self

        if isinstance(self.input_data, ProgramInput):
            if isinstance(self.results, SinglePointData):
                primary = self.input_data.calctype.value
                if getattr(self.results, primary) is None:
                    raise ValueError(f"Missing the primary result: {primary}.")
            elif isinstance(self.results, OptimizationData):
                if not self.results.frames:
                    raise ValueError("Successful optimizations require frames.")
            elif isinstance(self.results, PathOptimizationData):
                if not self.results.paths:
                    raise ValueError("Successful path optimizations require paths.")
            elif isinstance(self.results, ScanData):
                if not self.results.points:
                    raise ValueError("Successful scans require points.")
            elif isinstance(self.results, PathData):
                if not self.results.frames:
                    raise ValueError("Successful path calculations require frames.")
            elif isinstance(self.results, ConformerSearchData):
                if not self.results.conformers:
                    raise ValueError(
                        "Successful conformer searches require conformers."
                    )
        return self


ProgramInput.model_rebuild(_types_namespace={"SinglePointData": SinglePointData})
ProgramOutput.model_rebuild()
OptimizationData.model_rebuild()
PathData.model_rebuild()
PathOptimizationData.model_rebuild()
ConformerSearchData.model_rebuild()
ScanData.model_rebuild()


def _register_program_output_classes() -> None:
    """Required so that pickle can find the concrete classes for serialization."""
    # Parameterized with runtime class objects, which mypy cannot treat as types.
    generic_output: Any = ProgramOutput
    for spec_type, data_type in [(FileInput, FileData)] + [
        (ProgramInput, data_type) for data_type in get_args(StructuredData)
    ]:
        _class = generic_output[spec_type, data_type]
        name = _class.__name__
        this_module = sys.modules[__name__]
        if name not in this_module.__dict__:
            setattr(this_module, name, _class)


_register_program_output_classes()
