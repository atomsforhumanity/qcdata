"""The base model from which all QCData model objects inherit."""

import json
from abc import ABC
from base64 import b64decode, b64encode
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self

import numpy as np
import toml
import yaml
from pydantic import (
    BaseModel,
    Field,
    FieldSerializationInfo,
    SerializerFunctionWrapHandler,
    computed_field,
    field_serializer,
    field_validator,
    model_serializer,
    model_validator,
)

from .. import _array_files, _version
from ..helper_types import FreeDict, StrOrPath

if TYPE_CHECKING:  # pragma: no cover
    from pydantic.typing import ReprArgs


__all__ = [
    "Files",
    "Provenance",
    "ExecutionInfo",
    "OutputBase",
    "ErrorKind",
    "ErrorInfo",
    "ProgramWarning",
    "Resolved",
    "Model",
    "CalcType",
    "LengthUnit",
]


class LengthUnit(StrEnum):
    """The unit of length.

    Attributes:
        BOHR (str):  Bohr.
        ANGSTROM (str): Angstroms.

    """

    BOHR = "bohr"
    ANGSTROM = "angstrom"

    def __repr__(self) -> str:
        """Return the member name in quotes, e.g. `'BOHR'`."""
        return f"'{self.name}'"  # pragma: no cover


class QCDataBaseModel(BaseModel, ABC):
    """Base model for all QCData objects.

    Attributes:
        qcdata_version: Installed qcdata version writing this representation.
            Automatically refreshed on serialization; not a schema version.
        extras: Additional information to bundle with the object. Use for schema
            development and scratch space.
    """

    extras: FreeDict = {}

    @computed_field(repr=False)  # type: ignore[prop-decorator]
    @property
    def qcdata_version(self) -> str:
        """The package version serializing the object, not its original creator."""
        return _version.__version__

    @model_validator(mode="before")
    @classmethod
    def _discard_saved_version(cls, values: Any) -> Any:
        """Accept saved version stamps; the computed field supplies the current one."""
        if isinstance(values, dict) and "qcdata_version" in values:
            values = dict(values)
            values.pop("qcdata_version")
        return values

    model_config = {
        # Raises an error if extra fields are passed to model.
        "extra": "forbid",
        # Allow numpy types in models. Pydantic will no longer raise an exception for
        # types it doesn't recognize.
        # https://docs.pydantic.dev/latest/usage/types/#arbitrary-types-allowed
        "arbitrary_types_allowed": True,
        # Freeze attribute assignment. Contained dicts, lists, and arrays must be
        # treated as read-only by callers; this is not deep immutability.
        # https://docs.pydantic.dev/2.3/api/config/#pydantic.config.ConfigDict.frozen
        "frozen": True,
    }

    @classmethod
    def open(cls, filepath: Path | str) -> Self:
        """Instantiate an object from data saved to disk.

        Arrays that `save` put in their own `.npy` files are read back from beside the
        file, each checked against the sha256, shape and dtype saved with it.

        Args:
            filepath: The path to the object on disk.

        Returns:
            The instantiated object.

        Example:
            ```python
            my_obj = MyModel.open("path/to/file.json")
            ```
        """
        filepath = Path(filepath)
        data = filepath.read_text()
        context = {_array_files.DIRECTORY: filepath.parent}

        if filepath.suffix in [".yaml", ".yml"]:
            return cls.model_validate(yaml.safe_load(data), context=context)
        elif filepath.suffix == ".toml":
            return cls.model_validate(toml.loads(data), context=context)

        # Assume json for all other file extensions
        return cls.model_validate_json(data, context=context)

    @classmethod
    def open_multi(cls, filepath: Path | str) -> list[Self]:
        """Instantiate a list of objects from data saved to disk.

        Args:
            filepath: The path to the object on disk. Must be a list of objects.

        Returns:
            A list of instantiated objects.

        Example:
            ```python
            my_objs = MyModel().open_multi("path/to/file.json")
            ```
        """
        filepath = Path(filepath)
        data = filepath.read_text()

        if filepath.suffix in [".yaml", ".yml"]:
            return [cls.model_validate(d) for d in yaml.safe_load(data)]
        elif filepath.suffix == ".toml":
            return [cls.model_validate(d) for d in toml.loads(data)]

        # Assume json for all other file extensions
        return [cls.model_validate(d) for d in json.loads(data)]

    def save(
        self,
        filepath: Path | str,
        exclude_none: bool = True,
        exclude_unset: bool = True,
        indent: int = 4,
        **kwargs: Any,
    ) -> list[Path]:
        """Save an object to disk as `json`, `yaml`, or `toml`.

        Objects such as `Structure` and `OptimizationData` can additionally be saved
        as `xyz` files.

        Every array field whose float64 size exceeds 1 MiB is written to its own `.npy`
        file beside the saved file, named by the file's stem and the array's path in
        the document (`output.results.hessian.npy` for `output.json`). The document
        holds a pointer in its place: `{"path", "shape", "dtype", "sha256"}`, the
        digest being the `.npy` file's. `open` reads the arrays back, so a model never
        holds a file reference. Array files are written first and the model's file
        last. Smaller arrays stay in the document, where every float round-trips
        exactly through JSON and YAML.

        Note:
            By default the object will be saved as a `json` file. If the file extension
            is `.yaml` or `.yml`, the object will be saved as a `yaml` file. If the file
            extension is `.toml`, the object will be saved as a `toml` file. If the file
            extension is `.xyz`, the object will be saved as an `xyz` file (for objects
            that support this format such as a `Structure` or an `OptimizationData`
            whose frames hold structures).

            Additionally, padding will be added to the file by default to make it more
            human-readable. You can adjust the amount of padding added by changing the
            `indent` parameter. Pass `indent=None` to create a more compact file.

        Args:
            filepath: The path to write the object to.
            exclude_none: If True, attributes with a value of None will not be written.
                Changing default behavior from pydantic.model_dump() to True.
            exclude_unset: If True, fields omitted during construction are excluded,
                even if their containers were later modified. Defaults to True for
                compact files; pass False to include unset/default-valued fields.
            indent: The number of spaces to use for indentation in the JSON file. 0
                creates a more compact JSON file, 4 is more human-readable.
            **kwargs: Additional keyword arguments to pass to the serialization method.

        Returns:
            Every file written: the model's file first, then one `.npy` file per array
            saved apart, in document order.

        Example:
            ```python
            my_obj.save("path/to/file.json")
            ```

            ```python
            my_obj.save("path/to/file.yaml")
            ```

            ```python
            my_obj.save("path/to/file.toml")
            ```

            ```python
            my_obj.save("path/to/file.xyz")
            ```
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(exist_ok=True, parents=True)

        sink = _array_files.ArraySink()
        model_dict = self.model_dump(
            mode="json",
            exclude_none=exclude_none,
            exclude_unset=exclude_unset,
            context={_array_files.SINK: sink},
            **kwargs,
        )
        model_dict, array_files = sink.place(model_dict, filepath.parent, filepath.stem)

        if filepath.suffix in [".yaml", ".yml"]:
            data = yaml.dump(model_dict, indent=indent)

        elif filepath.suffix == ".toml":
            data = toml.dumps(model_dict)

        else:
            # Write data to json regardless of file extension
            data = json.dumps(model_dict, indent=indent)

        filepath.write_text(data)
        return [filepath, *array_files]

    def __repr_args__(self) -> "ReprArgs":
        """Only show non empty fields in repr but always show success even if false."""

        def exists(value: Any) -> bool:
            if isinstance(value, np.ndarray):
                return value.size > 0
            return bool(value)

        return [  # pragma: no cover
            (name, value)
            for name, value in self.__dict__.items()
            if name == "success" or exists(value)
        ]

    def __eq__(self, other: Any) -> bool:
        """Check equality of two objects.

        Necessary because BaseModel.__eq__ does not compare numpy arrays.
        """
        if isinstance(other, self.__class__):
            return self.model_dump() == other.model_dump()
        return False


class Files(QCDataBaseModel):
    """File model for handling string and binary data.

    Binary data is encoded as base64 strings during serialization.

    Attributes:
        files: A dict mapping filename to str or bytes data.
    """

    files: dict[str, str | bytes] = {}

    @field_validator("files")
    def _convert_base64_to_bytes(
        cls, value: dict[str, str | bytes]
    ) -> dict[str, str | bytes]:
        """Convert base64 encoded data to bytes."""
        for filename, data in value.items():
            if isinstance(data, str) and data.startswith("base64:"):
                value[filename] = b64decode(data[7:])
        return value

    @field_serializer("files")
    def _serialize_files(
        self, files: dict[str, str | bytes], _info: FieldSerializationInfo
    ) -> dict[str, str]:
        """Serialize files to a dict of filename to base64 encoded string."""
        return {
            filename: (
                f"base64:{b64encode(data).decode('utf-8')}"
                if isinstance(data, bytes)
                else data
            )
            for filename, data in files.items()
        }

    def add_file(self, filepath: Path | str, relative_dir: Path | None = None) -> None:
        """Add a file to the object.

        The file is added to the `files` attribute with the filename as the key and
        the file data as the value. UTF-8 decodable files are stored as `str`; all
        other files are stored as `bytes`.

        Args:
            filepath: The path to the file.
            relative_dir: The directory to make the file relative to. Helpful when
                adding files from a subdirectory.

        Example:
            ```python
                my_obj.add_file("path/to/file.txt")
                print(my_obj.files)
                # Output: {"file.txt": "file data"}
            ```
        """
        filepath = Path(filepath)
        raw_bytes = filepath.read_bytes()
        try:
            data: str | bytes = raw_bytes.decode("utf-8")  # str
        except UnicodeDecodeError:
            data = raw_bytes  # bytes

        # Set filename relative to relative_dir
        if relative_dir:
            filename = str(filepath.relative_to(relative_dir))
        else:
            filename = filepath.name

        self.files[filename] = data
        # Add files to __pydantic_fields_set__ to ensure they are included in .save()
        self.__pydantic_fields_set__.add("files")

    def add_files(
        self,
        directory: StrOrPath,
        recursive: bool = False,
        exclude: list[str] | None = None,
    ) -> None:
        """Add all files in a directory to the object.

        Args:
            directory: The directory to add files from.
            recursive: Whether to recursively add files from subdirectories.
            exclude: A list of filenames to exclude from the directory.
        """
        directory = Path(directory)
        if exclude is None:
            exclude = []
        if recursive:
            files = directory.rglob("*")
        else:
            files = directory.glob("*")
        for filepath in files:
            if filepath.is_file() and filepath.name not in exclude:
                self.add_file(filepath, directory)

    def save_files(self, directory: StrOrPath = Path(".")) -> None:
        """Write all files to the specified directory.

        Text data is written in text mode and binary data in binary mode. Parent
        directories are created for filenames that contain relative paths.

        Args:
            directory: The directory to write the files to. Created if it does not
                exist (its parent must already exist). Defaults to the current
                working directory.
        """
        directory = Path(directory)
        directory.mkdir(exist_ok=True)
        for filename, data in self.files.items():
            mode = "w" if isinstance(data, str) else "wb"
            filepath = directory / filename
            # In case filename is a relative path, create the parent directories
            filepath.parent.mkdir(exist_ok=True, parents=True)
            filepath.open(mode).write(data)

    def __repr_args__(self) -> "ReprArgs":
        """Replace file data with '<bytes>' or '<str>' in __repr__."""
        rargs = super().__repr_args__()
        return [
            (
                (k, v)
                if k != "files"
                else (
                    k,
                    {
                        name: bytes if isinstance(data, bytes) else str
                        for name, data in v.items()
                    },
                )
            )
            for k, v in rargs
        ]


class Provenance(QCDataBaseModel):
    """Identity of the program that produced a result payload.

    Attributes:
        program: The scientific program associated with the data. For empty failure
            results, this identifies the intended producer even if execution never began.
        program_version: The program version, if known.
        program_commit: The source revision of the program (e.g. a git commit), if
            known; distinguishes development builds that share a version.
        extras: Additional information about the origin of the data.
    """

    program: str
    program_version: str | None = None
    program_commit: str | None = None


class ExecutionInfo(QCDataBaseModel):
    """Runtime information about a program execution.

    All fields may be unknown. Pass `ExecutionInfo()` when no details are available.

    Attributes:
        scratch_dir: The working directory used by the program.
        wall_time: Elapsed execution time in seconds.
        hostname: The hostname of the machine where the program ran.
        host_cpu: The number of logical CPUs on the host, not the allocated CPU count.
        host_mem_gib: The host's total memory in GiB, not the allocated memory.
        timings: Wall-clock seconds per phase of the calculation. The keys are the
            program's; ``total`` is the whole call, and drivers report ``children``
            (time inside child calculations) beside their own phases.
        threads: CPU threads the calculation ran with. Some programs' results depend
            on it at round-off, so it is recorded with every result.
        peak_memory_gib: Peak host memory the calculation used (GiB), as measured.
        peak_device_memory_gib: Peak GPU memory the calculation used (GiB), as
            measured; schedulers compare it with the plan's estimate.
        extras: Additional execution information.
    """

    scratch_dir: Path | None = None
    wall_time: float | None = None
    hostname: str | None = None
    host_cpu: int | None = None
    host_mem_gib: float | None = None
    timings: dict[str, float] = {}
    threads: int | None = None
    peak_memory_gib: float | None = None
    peak_device_memory_gib: float | None = None


class ErrorKind(StrEnum):
    """Why a calculation failed, in one vocabulary across programs.

    Attributes:
        input: Settings or a structure the program cannot run as given (an unknown key,
            a bad value, an unknown basis set or method).
        unsupported: A valid request the program does not implement, or a setting it
            cannot honour that would change the result.
        convergence: An iterative procedure (SCF, response equations, an optimization)
            did not converge; the results reached are attached.
        numerical: A numerical failure: non-finite values, singular matrices.
        resource: Not enough memory, disk, or another resource as set up.
        backend: The hardware backend failed (no usable GPU, a device error).
        program: A child calculation failed; `ErrorInfo.cause` holds its error.
        timeout: The calculation ran out of time (set by the runner).
        environment: The program could not be run at all: not installed, no licence,
            the process died (set by the runner).
        internal: A bug in the program: a broken invariant or a crash.
    """

    input = "input"
    unsupported = "unsupported"
    convergence = "convergence"
    numerical = "numerical"
    resource = "resource"
    backend = "backend"
    program = "program"
    timeout = "timeout"
    environment = "environment"
    internal = "internal"


class ErrorInfo(QCDataBaseModel):
    """A structured description of a failure.

    The Python traceback stays in `ProgramOutput.traceback`; this is what a caller
    branches on.

    Attributes:
        kind: The kind of failure.
        message: A human-readable explanation.
        code: A stable snake_case code that refines `kind` (e.g. ``max_iter``,
            ``energy_rose``, ``not_first_order_saddle``), defined and documented by the
            program. Callers branch on it instead of parsing `message`.
        program: The program that failed, when it is a child of the calculation (kind
            ``program``) or differs from the top-level program.
        cause: The child's own error, for kind ``program``: "the gradient failed because
            the SCF did not converge" stays structured through drivers.
    """

    kind: ErrorKind
    message: str
    code: str | None = None
    program: str | None = None
    cause: "ErrorInfo | None" = None


class ProgramWarning(QCDataBaseModel):
    """Something the program wants the caller to know about a calculation that ran.

    Attributes:
        code: A stable snake_case code, defined and documented by the program.
        message: A human-readable explanation.
        path: The settings key the warning concerns, as a dotted path (``scf.max_iter``),
            if any.
    """

    code: str
    message: str
    path: str | None = None


class Resolved(QCDataBaseModel):
    """The settings a calculation actually used.

    Attributes:
        settings: The program's complete settings tree with every ``"auto"`` replaced,
            in the shape of `ProgramInput.keywords`. Passed back as keywords, it
            reproduces the run. Memory- and backend-dependent choices that can change
            numbers (DF storage, precision per card class) appear here resolved.
        reasons: Why each automatic choice was made, keyed by dotted settings path
            (``"grid.xc": "calctype hessian: linus:3"``).
        deferred: In a plan only: dotted paths of settings left ``"auto"`` because the
            run decides them (``"jk.df_storage"`` from the free memory when the SCF
            starts). A finished output's resolved settings have every ``"auto"``
            replaced and nothing deferred.
    """

    settings: FreeDict = {}
    reasons: dict[str, str] = {}
    deferred: list[str] = []


class OutputBase(QCDataBaseModel):
    """What every calculation reports, whatever it was given: success, logs, failure.

    `ProgramOutput` adds the input it ran (`input_data`); `Frame` adds the structure
    of a step inside a larger calculation, whose input its parent already holds.
    Subclasses declare `results`.

    Attributes:
        success: Whether the calculation succeeded.
        logs: The program's output, kept for failures and on request otherwise.
        traceback: Why it failed; required when `success` is False.
        execution: Runtime information; defaults to an empty ExecutionInfo.
        warnings: What the program wants the caller to know about a run that went
            through (a setting that had no effect, a fallback taken).
        error: Why the calculation failed, as a kind and a code callers can branch on;
            only on failures. Partial results stay in `results`.
        record: What happened during the run (fallbacks, escalations, residuals
            reached, a warm start taken), shaped by the program's published record
            schema. Diagnostic: not stable across program versions, unlike `results`
            and `resolved`.
        plogs str: `@property` Print the logs.
        ptraceback str: `@property` Print the traceback.
    """

    success: bool
    logs: str | None = None
    traceback: str | None = None
    execution: ExecutionInfo = Field(default_factory=ExecutionInfo)
    warnings: list[ProgramWarning] = []
    error: ErrorInfo | None = None
    record: FreeDict = {}

    @model_validator(mode="after")
    def _require_traceback_on_failure(self) -> Self:
        """A failure must say why."""
        if not self.success and self.traceback is None:
            raise ValueError("A traceback must be provided for failed calculations.")
        if self.success and self.error is not None:
            raise ValueError("A successful calculation has no error.")
        return self

    @model_serializer(mode="wrap")
    def _serialize_in_reading_order(
        self, handler: SerializerFunctionWrapHandler
    ) -> Any:
        """Write success, then the subclass's fields, then logs and execution."""
        data = handler(self)
        if not isinstance(data, dict):
            return data
        trailing = ("warnings", "error", "record", "logs", "traceback", "execution")
        head = {k: data[k] for k in ("success",) if k in data}
        body = {k: v for k, v in data.items() if k not in head and k not in trailing}
        tail = {k: data[k] for k in trailing if k in data}
        return {**head, **body, **tail}

    @property
    def plogs(self) -> None:
        """Print the logs."""
        print(self.logs)

    @property
    def ptraceback(self) -> None:
        """Print the traceback."""
        print(self.traceback)

    def __repr_args__(self) -> "ReprArgs":
        """Hide logs and traceback in the repr and put success first."""
        args = [
            (key, value if key not in {"logs", "traceback"} else "<...>")
            for key, value in super().__repr_args__()
        ]
        return [a for a in args if a[0] == "success"] + [
            a for a in args if a[0] != "success"
        ]


class CalcType(StrEnum):
    """The Calculation type.

    Attributes:
        energy: Energy calculation.
        gradient: Gradient calculation.
        hessian: Hessian calculation.
        optimization: Geometry optimization (to a minima).
        transition_state: Transition state optimization (to a saddle point).
        conformer_search: Conformer search.
        scan: A scan over a coordinate.
        irc: Intrinsic reaction coordinate from a transition state.
        path_optimization: Optimization of a chain of structures (NEB, string, ...).
    """

    energy = "energy"
    gradient = "gradient"
    hessian = "hessian"
    optimization = "optimization"
    transition_state = "transition_state"
    conformer_search = "conformer_search"
    scan = "scan"
    irc = "irc"
    path_optimization = "path_optimization"

    def __repr__(self) -> str:
        """Return the member name in quotes, e.g. `'energy'`."""
        return f"'{self.name}'"  # pragma: no cover


class Model(QCDataBaseModel):
    """The model for the quantum chemistry calculation.

    Attributes:
        method: The name of the method to be used in the calculation. Named according to
            the convention of the program being called. If an MM calculation then the
            name of the force field.
        basis: The name of the basis set to be used in the calculation. Named according
            to the convention of the program being called.
    """

    method: str
    basis: str | None = None
