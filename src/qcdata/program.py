"""The protocol an AFH program implements to run under qccompute, and its sessions.

A program package registers one object under the ``qcdata.programs`` entry point group::

    [project.entry-points."qcdata.programs"]
    linus = "linus.qc:program"

The object implements `Program`: it says what it can do (`capabilities`), publishes the
JSON Schema of its settings (`settings_schema`) and of its run record (`record_schema`),
plans a calculation without running it (`plan`: validated, every ``"auto"`` resolved, the
resources it needs), and computes it (`compute`) under a `RunContext` the runner fills.
Programs that keep state between geometries (warm starts, a pinned grid) also implement
`SessionProgram.open_session`; programs that evaluate many inputs at once implement
`BatchProgram.compute_batch`.

Programs depend on qcdata only. qccompute discovers them (`find_program`), validates
keywords against their schemas, schedules them by their plans' resources, and runs them in
its own process or in workers. ``program.compute(inp, RunContext())`` also works without
qccompute.

Design: ``docs/design/program-interface.md``; names and units: ``docs/conventions.md``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from importlib.metadata import entry_points
from pathlib import Path
from types import TracebackType
from typing import Any, Protocol, overload, runtime_checkable

import numpy as np

from .models import (
    CalcType,
    Data,
    ErrorInfo,
    ErrorKind,
    ProgramInput,
    ProgramOutput,
    ProgramSpec,
    ProgramWarning,
    Resolved,
    SinglePointData,
    Structure,
)
from .models.base_models import QCDataBaseModel

__all__ = [
    "ENTRY_POINT_GROUP",
    "Capabilities",
    "Resources",
    "Plan",
    "Request",
    "ProgramError",
    "Compute",
    "RunContext",
    "Program",
    "SessionProgram",
    "BatchProgram",
    "Session",
    "ComputeSession",
    "find_program",
    "available_programs",
]

#: The entry point group programs register under.
ENTRY_POINT_GROUP = "qcdata.programs"


class Capabilities(QCDataBaseModel):
    """What a program can do and how it may be run.

    Attributes:
        calctypes: The calculation types it computes.
        methods: The methods it implements, if it can list them.
        sessions: Whether it implements `SessionProgram.open_session`.
        batches: Whether it implements `BatchProgram.compute_batch`.
        in_process: Whether it may run in the runner's own process (it neither writes
            to the process's stdout/stderr file descriptors nor changes process-global
            state); otherwise the runner uses a worker process.
        thread_safe: Whether separate calculations and sessions may run concurrently
            in threads of one process (the program releases the GIL and shares no
            mutable state); the runner then runs them side by side in threads instead
            of one at a time.
        gpu: Whether it can use GPUs.
    """

    calctypes: frozenset[CalcType]
    methods: list[str] | None = None
    sessions: bool = False
    batches: bool = False
    in_process: bool = True
    thread_safe: bool = False
    gpu: bool = False


class Resources(QCDataBaseModel):
    """What one calculation needs from the machine; the runner schedules by it.

    Attributes:
        threads: CPU threads.
        memory_gib: Host memory (GiB), if known.
        gpus: Number of GPUs.
        device_memory_gib: Memory per GPU (GiB), if capped; several calculations may
            then share one device.
        exclusive: Whether the calculation must have the machine (or its GPUs) to
            itself, e.g. for clean timings.
    """

    threads: int = 1
    memory_gib: float | None = None
    gpus: int = 0
    device_memory_gib: float | None = None
    exclusive: bool = False


class Plan(QCDataBaseModel):
    """A calculation planned but not run.

    Attributes:
        resolved: The settings with every ``"auto"`` replaced, and why; exactly what
            the run's `ProgramOutput.resolved` will hold.
        resources: What the calculation needs. A driver's is the largest over its
            children's plans.
        warnings: What the program would warn about.
    """

    resolved: Resolved
    resources: Resources = Resources()
    warnings: list[ProgramWarning] = []


@dataclass(frozen=True)
class Request:
    """One evaluation of a session.

    Attributes:
        positions: Atomic positions (bohr), n_atoms x 3, atoms as the session's
            structure.
        calctype: ``energy``, ``gradient`` or ``hessian``.
        accuracy: Target accuracy of this request, by quantity (``{"gradient":
            1e-4}`` in Eh/bohr, ``{"energy": 1e-7}`` in Eh); the program maps it onto
            its convergence settings and may only tighten them. None: the session's
            settings.
    """

    positions: np.ndarray
    calctype: CalcType = CalcType.gradient
    accuracy: Mapping[str, float] | None = None


class ProgramError(Exception):
    """A failed request of a session, or a failed plan.

    Attributes:
        error: What went wrong, as a kind and code callers can branch on.
        results: What was computed before the failure, if anything.
    """

    def __init__(self, error: ErrorInfo, results: SinglePointData | None = None):
        """Create the exception.

        Args:
            error: What went wrong.
            results: Partial results, if any.
        """
        super().__init__(f"{error.kind}: {error.message}")
        self.error = error
        self.results = results

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


class Compute(Protocol):
    """Computes child calculations; the runner provides it in `RunContext.compute`.

    It takes what ``qccompute.compute`` takes: one input, or a list of independent
    inputs that the runner may batch or run side by side. It returns outputs, failed or
    not, and never raises for a failed calculation. ``previous`` holds outputs whose
    wavefunctions start the children.
    """

    @overload
    def __call__(
        self,
        inputs: ProgramInput,
        /,
        *,
        previous: ProgramOutput[ProgramInput, Data] | None = None,
    ) -> ProgramOutput[ProgramInput, Data]: ...

    @overload
    def __call__(
        self,
        inputs: Sequence[ProgramInput],
        /,
        *,
        previous: Sequence[ProgramOutput[ProgramInput, Data] | None] | None = None,
    ) -> list[ProgramOutput[ProgramInput, Data]]: ...


#: Opens a child session: (spec, structure, threads) -> Session.
OpenSession = Callable[[ProgramSpec, Structure, int | None], "Session"]


@dataclass
class RunContext:
    """What the runner gives a calculation: its resource grant and its callbacks.

    `RunContext()` is the context of a direct call: nothing granted (the program uses
    its resolved settings) and no runner for children.

    Attributes:
        threads: CPU threads granted; use these, not the machine's count. None: not
            managed by a runner.
        gpus: Identifiers of the GPUs granted (as for ``CUDA_VISIBLE_DEVICES``).
        memory_gib: Host memory granted (GiB), if managed.
        scratch: A directory the calculation may write to, if the runner made one.
        compute: Computes child calculations (drivers only).
        open_session: Opens child sessions; use `session`.
        progress: Receives streaming updates (any JSON-serializable value), if the
            caller wants them.
    """

    threads: int | None = None
    gpus: tuple[str, ...] = ()
    memory_gib: float | None = None
    scratch: Path | None = None
    compute: Compute | None = None
    open_session: OpenSession | None = None
    progress: Callable[[Any], None] | None = None

    def session(
        self, spec: ProgramSpec, structure: Structure, *, threads: int | None = None
    ) -> Session:
        """Open a session for a child program under this calculation's grant.

        Args:
            spec: The child program, model and keywords.
            structure: The child's atoms, charge, and multiplicity, and its first
                geometry.
            threads: This session's share of the grant (a driver with 16 threads and
                8 tasks gives each session 2). None: the child's own settings.

        Returns:
            The session; close it (or use it as a context manager) when done.

        Raises:
            RuntimeError: The runner provides neither sessions nor a compute callback.
        """
        if self.open_session is not None:
            return self.open_session(spec, structure, threads)
        if self.compute is not None:
            return ComputeSession(self.compute, spec, structure)
        raise RuntimeError(
            "This calculation computes child calculations, but its runner provided "
            "neither sessions nor a compute callback."
        )


@runtime_checkable
class Session(Protocol):
    """The same program and molecule at many geometries, warm-started.

    Opened by `SessionProgram.open_session` (or ``qccompute.session``). The session pins
    the model and approximations (method, basis, grids, precision), so all its
    evaluations lie on one potential energy surface; a request may vary only how
    tightly it converges (`Request.accuracy`). It knows the first geometry from
    continuations. Contract:

    - A failed request raises `ProgramError` and leaves the session usable: a driver
      may retry a smaller step on the same session.
    - A ``hessian`` request returns the energy and gradient too.
    - Results are `SinglePointData`; frames take ``program`` and ``program_version``
      from the session.

    Attributes:
        program: The program's name.
        program_version: The program's version.
    """

    program: str
    program_version: str

    def compute(
        self,
        positions: np.ndarray,
        calctype: CalcType = CalcType.gradient,
        *,
        accuracy: Mapping[str, float] | None = None,
    ) -> SinglePointData:
        """Evaluate one geometry.

        Args:
            positions: Atomic positions (bohr), n_atoms x 3.
            calctype: ``energy``, ``gradient`` or ``hessian``.
            accuracy: See `Request.accuracy`.

        Returns:
            The results at this geometry.

        Raises:
            ProgramError: The evaluation failed; the session stays usable.
        """
        ...

    def compute_many(
        self, requests: Sequence[Request]
    ) -> list[SinglePointData | ProgramError]:
        """Evaluate many geometries (finite differences, path images).

        Args:
            requests: The evaluations.

        Returns:
            One entry per request, in order: its results, or its `ProgramError`.
        """
        ...

    def plan(self) -> Plan:
        """The settings pinned for the session's lifetime and its resources."""
        ...

    def close(self) -> None:
        """Release the session's resources."""
        ...

    def __enter__(self) -> Session:
        """Use the session as a context manager."""
        ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Close the session."""
        ...


@runtime_checkable
class Program(Protocol):
    """A program qccompute can run: the object registered under ``qcdata.programs``.

    Attributes:
        name: The program's name, as in ``ProgramInput.program``.
    """

    name: str

    def version(self) -> str:
        """The program's version."""
        ...

    def capabilities(self) -> Capabilities:
        """What the program can do and how it may be run."""
        ...

    def settings_schema(self, calctype: CalcType) -> dict[str, Any]:
        """The JSON Schema of ``keywords`` (and of ``resolved.settings``) for a calctype.

        Every key carries ``x-unit``, ``x-tier`` (common, expert), ``x-concern``
        (model, approximation, convergence, algorithm, resource, output) and
        ``x-applies-to`` where they apply.
        """
        ...

    def record_schema(self, calctype: CalcType) -> dict[str, Any]:
        """The JSON Schema of ``ProgramOutput.record`` for a calctype."""
        ...

    def plan(self, inp: ProgramInput) -> Plan:
        """Validate an input and resolve its settings without running it.

        Raises:
            ProgramError: The input cannot run (kind ``input`` or ``unsupported``),
                with a did-you-mean suggestion for unknown keys.
        """
        ...

    def compute(
        self, inp: ProgramInput, ctx: RunContext
    ) -> ProgramOutput[ProgramInput, Data]:
        """Run a calculation.

        A failed calculation returns ``success=False`` with `ProgramOutput.error` and
        the partial results; it raises only on bugs.
        """
        ...


@runtime_checkable
class SessionProgram(Program, Protocol):
    """A program that keeps state between geometries (``capabilities().sessions``)."""

    def open_session(
        self, spec: ProgramSpec, structure: Structure, ctx: RunContext
    ) -> Session:
        """Open a session at `structure`'s atoms and first geometry.

        Raises:
            ProgramError: The spec cannot run (kind ``input`` or ``unsupported``).
        """
        ...


@runtime_checkable
class BatchProgram(Program, Protocol):
    """A program that evaluates many inputs at once (``capabilities().batches``)."""

    def compute_batch(
        self, inputs: Sequence[ProgramInput], ctx: RunContext
    ) -> list[ProgramOutput[ProgramInput, Data]]:
        """Run many calculations together; one output per input, in order."""
        ...


class ComputeSession:
    """A session over a compute callback, for programs without sessions of their own.

    Each request becomes a `ProgramInput` from the session's spec at the new positions,
    computed by `compute` (a program's `Program.compute` under a context, or a runner's
    callback). It has no warm starts unless the callback adds them. A ``hessian``
    result without a gradient is completed by a gradient calculation.

    Args:
        compute: Computes one input (or a list) and returns outputs, failed or not.
        spec: The program, model, and keywords.
        structure: The atoms, charge, and multiplicity.
    """

    def __init__(
        self,
        compute: Callable[..., Any],
        spec: ProgramSpec,
        structure: Structure,
    ):
        """Create the session."""
        self._compute = compute
        self._spec = spec
        self._structure = structure
        self.program = spec.program
        self.program_version = ""
        self._closed = False

    def _input(self, positions: np.ndarray, calctype: CalcType) -> ProgramInput:
        geometry = np.asarray(positions, dtype=float).reshape(-1, 3)
        if geometry.shape != np.asarray(self._structure.geometry).shape:
            raise ProgramError.of(
                "input",
                f"positions have {geometry.shape[0]} atoms; the session's structure "
                f"has {len(self._structure.symbols)}",
            )
        structure = self._structure.model_copy(update={"geometry": geometry})
        spec = self._spec.model_copy(update={"calctype": CalcType(calctype)})
        return ProgramInput.from_spec(spec, structure)

    def _results(self, out: ProgramOutput[Any, Any]) -> SinglePointData:
        version = out.results.provenance.program_version
        if version and not self.program_version:
            self.program_version = version
        if not out.success:
            error = out.error or ErrorInfo(
                kind=ErrorKind.internal, message=out.traceback or "failed"
            )
            raise ProgramError(error, out.results)
        return out.results

    def compute(
        self,
        positions: np.ndarray,
        calctype: CalcType = CalcType.gradient,
        *,
        accuracy: Mapping[str, float] | None = None,
    ) -> SinglePointData:
        """Evaluate one geometry; see `Session.compute`.

        Args:
            positions: Atomic positions (bohr).
            calctype: ``energy``, ``gradient`` or ``hessian``.
            accuracy: Ignored: a stateless session runs its spec's settings.

        Returns:
            The results.

        Raises:
            ProgramError: The calculation failed.
        """
        if self._closed:
            raise RuntimeError("The session is closed.")
        calctype = CalcType(calctype)
        results = self._results(self._compute(self._input(positions, calctype)))
        if calctype == CalcType.hessian and results.gradient is None:
            gradient = self._results(
                self._compute(self._input(positions, CalcType.gradient))
            )
            results = results.model_copy(
                update={"gradient": gradient.gradient, "energy": gradient.energy}
            )
        return results

    def compute_many(
        self, requests: Sequence[Request]
    ) -> list[SinglePointData | ProgramError]:
        """Evaluate many geometries in one call to the callback.

        Args:
            requests: The evaluations.

        Returns:
            Results or the `ProgramError` of each request, in order.
        """
        if self._closed:
            raise RuntimeError("The session is closed.")
        out: list[SinglePointData | ProgramError] = []
        inputs: list[ProgramInput] = []
        for r in requests:
            inputs.append(self._input(r.positions, r.calctype))
        outputs = self._compute(inputs) if inputs else []
        for r, o in zip(requests, outputs, strict=True):
            try:
                results = self._results(o)
                if r.calctype == CalcType.hessian and results.gradient is None:
                    results = self.compute(r.positions, CalcType.hessian)
                out.append(results)
            except ProgramError as e:
                out.append(e)
        return out

    def plan(self) -> Plan:
        """Not known to a stateless session: the spec's keywords, unresolved."""
        return Plan(resolved=Resolved(settings=dict(self._spec.keywords)))

    def close(self) -> None:
        """Close the session."""
        self._closed = True

    def __enter__(self) -> ComputeSession:
        """Use the session as a context manager."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Close the session."""
        self.close()


def available_programs() -> list[str]:
    """Names of the programs installed under the ``qcdata.programs`` entry points."""
    return sorted(ep.name for ep in entry_points(group=ENTRY_POINT_GROUP))


def find_program(name: str) -> Program | None:
    """Load an installed program by name.

    Args:
        name: The program's name, as in ``ProgramInput.program``.

    Returns:
        The program object, or None if no installed package registers it.

    Raises:
        TypeError: The registered object does not implement `Program`.
    """
    for ep in entry_points(group=ENTRY_POINT_GROUP):
        if ep.name == name:
            program = ep.load()
            if not isinstance(program, Program):
                raise TypeError(
                    f"The qcdata.programs entry point {name!r} ({ep.value}) does not "
                    "implement qcdata.program.Program."
                )
            return program
    return None
