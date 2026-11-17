"""One step of a session as data: its request, its accuracy target, its failure.

A session (an execution concept, defined in qccompute) evaluates one molecule at many
geometries. What a step asks for (`Request`, `Accuracy`) and how it fails
(`ProgramError`, the exception form of `ErrorInfo`) are data, shared by every program
without importing an execution package. A step's result is a `Frame`.
"""

from __future__ import annotations

import json
from typing import Annotated, Any

import numpy as np
from pydantic import AfterValidator, field_validator

from ..helper_types import SerializableMatrix
from .base_models import ErrorInfo, ErrorKind, QCDataBaseModel
from .data import Frame, SinglePointData
from .inputs import CalcType

__all__ = ["Accuracy", "Lineage", "Request", "ProgramError", "lineage_key"]


def lineage_key(lineage: Any) -> str:
    """The text a program keys a lineage's history by: its canonical JSON.

    Keys sorted, compact separators, tuples as lists, so ``("irc", "forward")`` and
    ``["irc", "forward"]`` are the same lineage, in process and over a worker's wire.

    Args:
        lineage: Any JSON value; None is the default lineage.

    Returns:
        The canonical JSON text.

    Raises:
        TypeError: The value is not JSON (a set, an object, ...).
        ValueError: The value holds a non-finite number.
    """
    return json.dumps(
        lineage,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _json_lineage(value: Any) -> Any:
    try:
        lineage_key(value)
    except (TypeError, ValueError) as e:
        raise ValueError(f"A lineage must be a JSON value: {e}") from None
    return value


#: Names a chain of steps whose history a session keeps: any JSON value. None is the
#: default lineage; a tuple stays a tuple in process (hashable) and is a list in JSON.
Lineage = Annotated[Any, AfterValidator(_json_lineage)]


class Accuracy(QCDataBaseModel):
    """How tightly one request converges: targets, absolute and in atomic units.

    Each field is the numerical error from convergence the caller can tolerate in that
    quantity. It is a target, not a guarantee: the program picks the thresholds it
    expects to meet it (linus: SCF orbital gradient and response residual; axtb: energy
    and charge change) and records the target, the thresholds and any error estimate it
    has under the shared record key ``accuracy``. The session's explicit convergence
    settings are floors: a request may tighten them, never loosen them. Approximations
    (basis, grids, precision) are pinned by the session and are not covered.

    Attributes:
        energy: Target error of the energy (Eh).
        gradient: Target error of the largest absolute gradient component (Eh/bohr).
        hessian: Target error of the largest absolute Hessian element (Eh/bohr^2).
    """

    energy: float | None = None
    gradient: float | None = None
    hessian: float | None = None

    @field_validator("energy", "gradient", "hessian")
    @classmethod
    def _positive(cls, value: float | None) -> float | None:
        if value is not None and not (0.0 < value < float("inf")):
            raise ValueError("An accuracy target is a positive, finite number.")
        return value

    @classmethod
    def of(
        cls, value: Accuracy | float | None, calctype: CalcType | str
    ) -> Accuracy | None:
        """Read a request's accuracy: a bare number targets its highest derivative.

        Args:
            value: An `Accuracy`, a number, or None.
            calctype: The request's calculation type.

        Returns:
            The accuracy, or None for the session's own settings.
        """
        if value is None or isinstance(value, Accuracy):
            return value
        name = {"energy": "energy", "gradient": "gradient", "hessian": "hessian"}.get(
            CalcType(calctype).value, "energy"
        )
        return cls.model_validate({name: float(value)})


class Request(QCDataBaseModel):
    """One step of a session: where to evaluate, what, and from which history.

    Attributes:
        positions: Atomic positions (bohr), n_atoms x 3, atoms as the session's
            input. None: the input's own geometry.
        calctype: ``energy``, ``gradient`` or ``hessian``; None: the input's calctype.
        accuracy: Target accuracy of this request (a bare number targets the
            calctype's highest derivative). None: the session's settings.
        lineage: The chain of steps this one continues. A session keeps warm-start and
            extrapolation history per lineage (keyed by `lineage_key`), so NEB images,
            IRC branches and parallel tasks each start from their own last success.
            None is the default lineage.
        outputs: Optional outputs that do not change the surface but may cost extra.
            A name that fills a top-level `SinglePointData` field is that field's name
            (``scf_dipole_moment``, ``polarizability``); others name what they add
            (``ir``, ``orbitals``, ``density``). A program rejects ones it does not
            support with kind ``unsupported``.
        seed: Lineages to start from, in order, when ``lineage`` has no history yet:
            the first with history is used, and with none the request starts fresh.
            Both IRC branches start from the transition state's lineage this way. A
            request seeded from a lineage runs after every earlier request of that
            lineage in the same list. The record names the lineage used.
        commit: Whether a success becomes ``lineage``'s history. False stores nothing:
            a finite-difference fan-out in the center's lineage starts every
            displacement from the center, without keeping a state per displacement.
            In a list, a request reads its lineage's history as it stands at its own
            place: after every earlier committing request of that lineage, unaffected
            by later ones. In [A, B (commit=False), C, D (commit=False)], all in one
            lineage, B starts from A and D from C; B may run alongside C if it reads
            the history from before C.
    """

    positions: SerializableMatrix | None = None
    calctype: CalcType | None = None
    accuracy: Accuracy | float | None = None
    lineage: Lineage = None
    outputs: frozenset[str] = frozenset()
    seed: list[Lineage] = []
    commit: bool = True

    @field_validator("positions")
    @classmethod
    def _n_by_3(cls, value: np.ndarray | None) -> np.ndarray | None:
        if value is None:
            return None
        if (
            value.size % 3
            or value.ndim > 2
            or (value.ndim == 2 and value.shape[1] != 3)
        ):
            raise ValueError(f"positions must be n_atoms x 3, not {value.shape}.")
        return value.reshape(-1, 3)


class ProgramError(Exception):
    """A failed calculation: `ErrorInfo` raised, with what was computed before it failed.

    Every program's ``compute`` and every session call raise it (or a subclass) when a
    calculation fails, an input the program refuses included; any other exception is a
    bug. A program catches this one class from its subprograms, whichever programs they
    are. A runner turns it into a failed `ProgramOutput` only where it needs data: a
    store, the outputs of a list, a worker's reply.

    Attributes:
        error: What went wrong, as a kind and code callers can branch on.
        frame: The failed step, when there was one: its structure, partial results,
            warnings and record (an SCF's iteration history, its stability analyses).
        results: What was computed before the failure, if anything.
    """

    def __init__(
        self,
        error: ErrorInfo,
        results: SinglePointData | None = None,
        *,
        frame: Frame | None = None,
    ):
        """Create the exception.

        Args:
            error: What went wrong.
            results: Partial results, if any; taken from ``frame`` when not given.
            frame: The failed step, if any.
        """
        super().__init__(f"{error.kind}: {error.message}")
        self.error = error
        self.frame = frame
        self.results = (
            results if results is not None or frame is None else frame.results
        )

    def __reduce__(self) -> tuple[Any, ...]:
        """Pickle with the error and frame, for worker processes."""
        return (type(self), (self.error, self.results), {"frame": self.frame})

    @classmethod
    def of(
        cls, kind: ErrorKind | str, message: str, *, code: str | None = None
    ) -> ProgramError:
        """Shorthand for a ProgramError without partial results.

        Args:
            kind: The error kind.
            message: A human-readable explanation.
            code: A program-defined code refining the kind.

        Returns:
            The exception.
        """
        return cls(ErrorInfo(kind=ErrorKind(kind), message=message, code=code))
