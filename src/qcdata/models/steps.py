"""One step of a session as data: its request, its accuracy target, its failure.

A session (an execution concept, defined in qccompute) evaluates one molecule at many
geometries. What a step asks for (`Request`, `Accuracy`) and how it fails
(`ProgramError`, the exception form of `ErrorInfo`) are data, shared by every program
without importing an execution package. A step's result is a `Frame`.
"""

from __future__ import annotations

import difflib
import json
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Annotated, Any

import numpy as np
from pydantic import AfterValidator, ValidationError, field_validator

from ..helper_types import SerializableMatrix
from .base_models import ErrorInfo, ErrorKind, QCDataBaseModel
from .data import Data, Frame
from .inputs import CalcType

if TYPE_CHECKING:
    from .outputs import ProgramOutput

__all__ = [
    "Accuracy",
    "Lineage",
    "Request",
    "ProgramError",
    "ProgramErrors",
    "lineage_key",
]


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


_TARGETS = ("energy", "gradient", "hessian")


class Accuracy(QCDataBaseModel):
    """How tightly a calculation converges: targets, absolute and in atomic units.

    Each field is the numerical error from convergence the caller can tolerate in that
    quantity; a field left out sets no target for it. It is a target, not a guarantee:
    the program picks the thresholds it expects to meet it (linus: SCF orbital gradient
    and response residual; axtb: energy and charge change) and records the target, the
    thresholds and any error estimate it has under the shared record key ``accuracy``.
    Explicit convergence settings are floors: a target may tighten them, never loosen
    them. Approximations (basis, grids, precision) are not covered.

    A target is given two ways. The shared keyword ``accuracy`` (an object of these
    fields, read with `from_keywords`) is the input's own target: of ``compute``, and of
    every session request that gives none. A request's `Request.accuracy` replaces it
    for that request. The keyword is part of the input, so of its calculation key: a
    tighter target is another calculation. A program with no convergence settings (a
    force field, an ML potential) accepts a target, keyword or request alike, and
    records it as met: empty ``thresholds`` and an ``estimate`` of 0.0 per target. A
    program whose convergence settings take no targets yet refuses it with kind
    ``unsupported``.

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
            The accuracy, or None when the request gives none: the input's own target
            (`from_keywords`) then applies, else the session's settings.
        """
        if value is None or isinstance(value, Accuracy):
            return value
        name = {"energy": "energy", "gradient": "gradient", "hessian": "hessian"}.get(
            CalcType(calctype).value, "energy"
        )
        return cls.model_validate({name: float(value)})

    @classmethod
    def from_keywords(cls, keywords: Mapping[str, Any]) -> Accuracy | None:
        """Read the shared keyword ``accuracy``: the input's own target, if it sets one.

        The keyword is a JSON object of `Accuracy`'s targets
        (``{"energy": 1e-8, "gradient": 1e-6}``), never a bare number: an input's
        target applies to every calctype a session runs, so a number would have no
        single quantity to aim at. An `Accuracy` given in keywords is stored in this
        form (`ProgramSpec.keywords`).

        Args:
            keywords: The input's keywords (`ProgramSpec.keywords`).

        Returns:
            The target, or None when the keywords set none (absent or null).

        Raises:
            ProgramError: Kind ``input``: the value is not an object of positive,
                finite targets; code ``unknown_key`` for a key other than ``energy``,
                ``gradient`` and ``hessian``, and ``no_effect`` when it sets none.
        """
        value = keywords.get("accuracy")
        if value is None:
            return None
        if not isinstance(value, Mapping):
            raise ProgramError.of(
                "input",
                "The keyword accuracy is a JSON object of targets (energy in Eh, "
                "gradient in Eh/bohr, hessian in Eh/bohr^2), such as "
                f'{{"gradient": 1e-06}}, not {type(value).__name__} {value!r}.',
            )
        unknown = sorted(set(value) - set(_TARGETS))
        if unknown:
            hints = [
                f"accuracy.{k}"
                + "".join(
                    f" (did you mean accuracy.{m}?)"
                    for m in difflib.get_close_matches(k, _TARGETS, n=1)
                )
                for k in unknown
            ]
            raise ProgramError.of(
                "input",
                f"Unknown keyword {', '.join(hints)}; accuracy takes "
                f"{', '.join(_TARGETS)}.",
                code="unknown_key",
            )
        try:
            accuracy = cls.model_validate(value)
        except ValidationError as e:
            errors = "; ".join(
                f"accuracy.{'.'.join(map(str, err['loc']))}: "
                + err["msg"].removeprefix("Value error, ").rstrip(".")
                for err in e.errors()
            )
            raise ProgramError.of(
                "input", f"The keyword accuracy is not valid: {errors}."
            ) from None
        if accuracy.energy is accuracy.gradient is accuracy.hessian is None:
            raise ProgramError.of(
                "input",
                "The keyword accuracy sets no target; give energy, gradient or "
                "hessian, or leave it out.",
                code="no_effect",
            )
        return accuracy


class Request(QCDataBaseModel):
    """One step of a session: where to evaluate, what, and from which history.

    Attributes:
        positions: Atomic positions (bohr), n_atoms x 3, atoms as the session's
            input. None: the input's own geometry.
        calctype: ``energy``, ``gradient`` or ``hessian``; None: the input's calctype.
        accuracy: Target accuracy of this request (a bare number targets the
            calctype's highest derivative); it replaces the input's ``accuracy``
            keyword for this request. None: the keyword's target, else the session's
            settings.
        lineage: The chain of steps this one continues. A session keeps warm-start and
            extrapolation history per lineage (keyed by `lineage_key`), so NEB images,
            IRC branches and parallel tasks each start from their own last success.
            None is the default lineage.
        outputs: Optional outputs that do not change the surface but may cost extra.
            A name that fills a top-level `SinglePointData` field is that field's name
            (``dipole_moment``, ``polarizability``); others name what they add
            (``ir``). The wavefunction's parts are ``orbitals``, ``overlap``,
            ``hamiltonian`` and ``core_hamiltonian``; a program fills in what one
            implies (each brings the basis set, ``hamiltonian`` also ``overlap`` and
            ``orbitals``), as the conventions list. A program refuses an output it cannot give with kind
            ``unsupported``, code ``not_applicable``.
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

    A program whose result is one output raises ``ProgramError(out.error, output=out)``
    with its failed output, so that nothing it reached is lost (a trajectory's frames,
    its record and resolved settings); callers read ``exc.output``. A failed session
    step raises with its ``frame``.

    Attributes:
        error: What went wrong, as a kind and code callers can branch on.
        frame: The failed step, when there was one: its structure, partial results,
            warnings and record (an SCF's iteration history, its stability analyses).
        output: The whole failed output, when the program has one.
        results: What was computed before the failure, if anything: of the output's
            results type for a whole output, `SinglePointData` for a step.
    """

    def __init__(
        self,
        error: ErrorInfo,
        results: Data | None = None,
        *,
        frame: Frame | None = None,
        output: ProgramOutput[Any, Any] | None = None,
    ):
        """Create the exception.

        Args:
            error: What went wrong.
            results: Partial results, if any; when not given, taken from ``output``,
                else from ``frame``.
            frame: The failed step, if any.
            output: The failed output, if any.
        """
        super().__init__(f"{error.kind}: {error.message}")
        self.error = error
        self.frame = frame
        self.output = output
        if results is None and output is not None:
            results = output.results
        if results is None and frame is not None:
            results = frame.results
        self.results = results

    def __reduce__(self) -> tuple[Any, ...]:
        """Pickle with the error, frame and output, for worker processes."""
        return (
            type(self),
            (self.error, self.results),
            {"frame": self.frame, "output": self.output},
        )

    @classmethod
    def of(
        cls,
        kind: ErrorKind | str,
        message: str,
        *,
        code: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> ProgramError:
        """Shorthand for a ProgramError without partial results.

        Args:
            kind: The error kind.
            message: A human-readable explanation.
            code: A program-defined code refining the kind.
            details: Structured facts about the failure (`ErrorInfo.details`).

        Returns:
            The exception.
        """
        return cls(
            ErrorInfo(
                kind=ErrorKind(kind), message=message, code=code, details=details or {}
            )
        )


class ProgramErrors(ExceptionGroup):  # type: ignore[type-arg]
    """Some of several calculations failed: raised after every one of them ran.

    A runner raises it for a list of inputs, and a session's ``compute_many`` for a list
    of requests. Each sub-exception is one failure's `ProgramError`. ``results`` holds
    every result in request order, the successful ones included and the failed ones in
    place, so no result is lost: `ProgramOutput`s for a list of inputs, `Frame`s for a
    session. Handle failures with ``except*``, or catch the group and read ``results``.

    Attributes:
        results: One result per request, in order; failures in place.
    """

    results: list[Any]

    def __new__(
        cls, message: str, exceptions: Sequence[Exception], results: Sequence[Any] = ()
    ) -> ProgramErrors:
        """Build the group and attach every result."""
        self = super().__new__(cls, message, exceptions)
        self.results = list(results)
        return self

    def __init__(
        self, message: str, exceptions: Sequence[Exception], results: Sequence[Any] = ()
    ) -> None:
        """Initialize the group; ``__new__`` attaches the results."""
        super().__init__(message, exceptions)

    def derive(self, excs: Sequence[Exception]) -> ProgramErrors:  # type: ignore[override]
        """Keep ``results`` on the groups that ``except*`` and ``split`` create."""
        return ProgramErrors(self.message, excs, self.results)

    def __reduce__(self) -> tuple[Any, ...]:
        """Pickle with the results, so the group crosses to and from worker processes."""
        return ProgramErrors, (self.message, list(self.exceptions), self.results)

    @classmethod
    def check(cls, results: Sequence[Any]) -> list[Any]:
        """The results, or this group if any of them failed.

        A session's ``compute_many`` ends with ``return ProgramErrors.check(frames)``.

        Args:
            results: One result per request, in order: `Frame`s or `ProgramOutput`s.

        Returns:
            The results, when every one succeeded.

        Raises:
            ProgramErrors: Some failed. Each failure's `ProgramError` carries its frame
                or its output.
        """
        from .outputs import ProgramOutput

        failed = []
        for r in results:
            if r.success:
                continue
            error = r.error or ErrorInfo(kind=ErrorKind.internal, message="failed")
            if isinstance(r, ProgramOutput):
                failed.append(ProgramError(error, output=r))
            else:
                failed.append(ProgramError(error, frame=r))
        if failed:
            raise cls(f"{len(failed)} of {len(results)} failed", failed, results)
        return list(results)
