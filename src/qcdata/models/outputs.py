"""Program output container objects."""

from __future__ import annotations

import sys
import warnings
from typing import Any, Generic, Literal, get_args

from pydantic import ValidationInfo, field_validator, model_validator
from typing_extensions import Self

from .base_models import ExecutionInfo, QCDataBaseModel
from .data import (
    ConformerSearchData,
    DataType,
    FileData,
    OptimizationData,
    ScanData,
    SinglePointData,
    StructuredData,
    get_data_type,
)
from .inputs import FileInput, ProgramInput, ProgramSpec
from .inputs import InputType as ProgramInputType

__all__ = ["ProgramOutput"]


class ProgramOutput(QCDataBaseModel, Generic[ProgramInputType, DataType]):
    """The core output object from a quantum chemistry calculation.

    Attributes:
        input_data: The input data for the calculation. Any of `qcdata.Inputs`.
        success: Whether the calculation was successful.
        results: Scientific data returned by the calculation, including output files.
            Any of `qcdata.Data`.
        logs: The logs from the calculation.
        traceback: The traceback from the calculation, if it failed.
        execution: Runtime information; defaults to an empty ExecutionInfo.
        extras Dict[str, Any]: Additional information to bundle with the results. Use for
            schema development and scratch space.
        plogs str: `@property` Print the logs.
        ptraceback str: `@property` Print the traceback.
    """

    input_data: ProgramInputType
    success: Literal[True, False]
    results: DataType
    logs: str | None = None
    traceback: str | None = None
    execution: ExecutionInfo = ExecutionInfo()

    @model_validator(mode="before")
    @classmethod
    def _migrate_legacy_data(cls, payload: Any) -> Any:
        """Accept the previous payload field while always serializing as results."""
        if isinstance(payload, dict) and "data" in payload:
            if "results" in payload:
                raise ValueError(
                    "Provide only 'results'; 'data' is its deprecated name."
                )
            warnings.warn(
                "'data' has been renamed to 'results'. Please update your code accordingly.",
                FutureWarning,
                stacklevel=2,
            )
            payload = dict(payload)
            payload["results"] = payload.pop("data")
        return payload

    def model_post_init(self, __context) -> None:
        """Parameterize the class (if not set explicitly)."""
        if self.__class__ is ProgramOutput:
            input_type = type(self.input_data)
            results_type = type(self.results)
            self.__class__ = self.__class__[input_type, results_type]  # type: ignore[index]

    @property
    def data(self) -> DataType:
        """Deprecated access to results."""
        warnings.warn(
            ".data has been renamed to .results. Please update your code accordingly.",
            FutureWarning,
            stacklevel=2,
        )
        return self.results

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
            return expected.model_validate(value)
        if type(value) is not expected:
            raise ValueError(
                f"Input requires {expected.__name__} results on both success and "
                f"failure; received {type(value).__name__}."
            )
        return value

    @model_validator(mode="after")
    def _validate_completion(self) -> Self:
        if not self.success:
            if self.traceback is None:
                raise ValueError(
                    "A traceback must be provided for failed calculations."
                )
            return self

        if isinstance(self.input_data, ProgramInput):
            if isinstance(self.results, SinglePointData):
                primary = self.input_data.calctype.value
                if getattr(self.results, primary) is None:
                    raise ValueError(f"Missing the primary result: {primary}.")
            elif isinstance(self.results, (OptimizationData, ScanData)):
                if not self.results.trajectory:
                    raise ValueError(
                        "Successful calculations require a nonempty trajectory."
                    )
            elif isinstance(self.results, ConformerSearchData):
                if not self.results.conformers:
                    raise ValueError(
                        "Successful conformer searches require conformers."
                    )
        return self

    @property
    def plogs(self) -> None:
        """Print the logs."""
        print(self.logs)

    @property
    def ptraceback(self) -> None:
        """Print the traceback."""
        print(self.traceback)

    def __repr_args__(self) -> list[tuple[str, Any]]:
        """Exclude logs and traceback from the repr and ensure success is first."""
        args = super().__repr_args__()
        filtered_args = [
            (key, value if key not in {"logs", "traceback"} else "<...>")
            for key, value in args
        ]
        success_arg = [(key, value) for key, value in filtered_args if key == "success"]
        other_args = [(key, value) for key, value in filtered_args if key != "success"]
        return success_arg + other_args


ProgramOutput.model_rebuild()
OptimizationData.model_rebuild()
ConformerSearchData.model_rebuild()
ScanData.model_rebuild()


def _register_program_output_classes():
    """Required so that pickle can find the concrete classes for serialization."""
    for spec_type, data_type in [(FileInput, FileData)] + [
        (ProgramInput, data_type) for data_type in get_args(StructuredData)
    ]:
        _class = ProgramOutput[spec_type, data_type]
        name = _class.__name__
        this_module = sys.modules[__name__]
        if name not in this_module.__dict__:
            setattr(this_module, name, _class)


_register_program_output_classes()
