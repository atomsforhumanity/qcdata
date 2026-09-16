"""The base model from which all QCData model objects inherit."""

import json
from abc import ABC
from base64 import b64decode, b64encode
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import toml
import yaml
from pydantic import (
    BaseModel,
    computed_field,
    field_serializer,
    field_validator,
    model_validator,
)
from typing_extensions import Self

from .. import _version
from ..helper_types import StrOrPath

if TYPE_CHECKING:  # pragma: no cover
    from pydantic.typing import ReprArgs


__all__ = ["Files", "Provenance", "ExecutionInfo", "Model", "CalcType", "LengthUnit"]


class LengthUnit(str, Enum):
    """The unit of length.

    Attributes:
        BOHR (str):  Bohr.
        ANGSTROM (str): Angstroms.

    """

    BOHR = "bohr"
    ANGSTROM = "angstrom"

    def __repr__(self) -> str:
        """Custom repr for LengthUnit"""
        return f"'{self.name}'"  # pragma: no cover


class QCDataBaseModel(BaseModel, ABC):
    """Base model for all QCData objects.

    Attributes:
        qcdata_version: Installed qcdata version writing this representation.
            Automatically refreshed on serialization; not a schema version.
        extras: Additional information to bundle with the object. Use for schema
            development and scratch space.
    """

    extras: dict[str, Any] = {}

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

        if filepath.suffix in [".yaml", ".yml"]:
            return cls.model_validate(yaml.safe_load(data))
        elif filepath.suffix == ".toml":
            return cls.model_validate(toml.loads(data))

        # Assume json for all other file extensions
        return cls.model_validate_json(data)

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
        **kwargs,
    ) -> None:
        """
        Save an object to disk as `json`, `yaml`, or `toml`. Objects such as `Structure`
        and `OptimizationData` can additionally be saved as `xyz` files.

        Note:
            By default the object will be saved as a `json` file. If the file extension
            is `.yaml` or `.yml`, the object will be saved as a `yaml` file. If the file
            extension is `.toml`, the object will be saved as a `toml` file. If the file
            extension is `.xyz`, the object will be saved as an `xyz` file (for objects
            that support this format such as a `Structure` or an `OptimizationData`
            which contains `.trajectory: list[Structure]`).

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

        model_dict = self.model_dump(
            mode="json",
            exclude_none=exclude_none,
            exclude_unset=exclude_unset,
            **kwargs,
        )

        if filepath.suffix in [".yaml", ".yml"]:
            data = yaml.dump(model_dict, indent=indent)

        elif filepath.suffix == ".toml":
            data = toml.dumps(model_dict)

        else:
            # Write data to json regardless of file extension
            data = json.dumps(model_dict, indent=indent)

        filepath.write_text(data)

    def __repr_args__(self) -> "ReprArgs":
        """Only show non empty fields in repr but always show success even if false."""

        def exists(value):
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
    def _convert_base64_to_bytes(cls, value):
        """Convert base64 encoded data to bytes."""
        for filename, data in value.items():
            if isinstance(data, str) and data.startswith("base64:"):
                value[filename] = b64decode(data[7:])
        return value

    @field_serializer("files")
    def _serialize_files(self, files, _info) -> dict[str, str]:
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
        """Add a file to the object. The file will be added at to the `files` attribute
            with the filename as the key and the file data as the value.

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
        """Write all files to the specified directory"""
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
        extras: Additional information about the origin of the data.
    """

    program: str
    program_version: str | None = None


class ExecutionInfo(QCDataBaseModel):
    """Runtime information about a program execution.

    All fields may be unknown. Pass `ExecutionInfo()` when no details are available.

    Attributes:
        scratch_dir: The working directory used by the program.
        wall_time: Elapsed execution time in seconds.
        hostname: The hostname of the machine where the program ran.
        host_cpu: The number of logical CPUs on the host, not the allocated CPU count.
        host_mem_gib: The host's total memory in GiB, not the allocated memory.
        extras: Additional execution information.
    """

    scratch_dir: Path | None = None
    wall_time: float | None = None
    hostname: str | None = None
    host_cpu: int | None = None
    host_mem_gib: float | None = None


class CalcType(str, Enum):
    """The Calculation type.

    Attributes:
        energy: Energy calculation.
        gradient: Gradient calculation.
        hessian: Hessian calculation.
        optimization: Geometry optimization (to a minima).
        transition_state: Transition state optimization (to a saddle point).
        conformer_search: Conformer search.
    """

    energy = "energy"
    gradient = "gradient"
    hessian = "hessian"
    optimization = "optimization"
    transition_state = "transition_state"
    conformer_search = "conformer_search"
    scan = "scan"

    def __repr__(self) -> str:
        """Custom repr for CalcType"""
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
