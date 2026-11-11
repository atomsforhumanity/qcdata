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

from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence
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
    Frame,
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
    "Accuracy",
    "Request",
    "ProgramError",
    "Compute",
    "RunContext",
    "Program",
    "SessionProgram",
    "BatchProgram",
    "Session",
    "ComputeSession",
    "failed_frame",
    "session_calctypes",
    "check_session_calctype",
    "find_program",
    "local_context",
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
        driver: Whether it computes child calculations (its input's subprograms)
            through `RunContext.compute` or `RunContext.session`.
    """

    calctypes: frozenset[CalcType]
    methods: list[str] | None = None
    sessions: bool = False
    batches: bool = False
    in_process: bool = True
    thread_safe: bool = False
    gpu: bool = False
    driver: bool = False


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
class Accuracy:
    """How accurate a session's result must be, absolute and in atomic units.

    A program maps the targets onto its convergence settings with its own error model
    (linus: SCF orbital-gradient and response tolerances); explicit convergence settings
    of the session act as floors, so a request may tighten them, never loosen them.

    Attributes:
        energy: Target error of the energy (Eh).
        gradient: Target error of the gradient components (Eh/bohr).
        hessian: Target error of the Hessian elements (Eh/bohr^2).
    """

    energy: float | None = None
    gradient: float | None = None
    hessian: float | None = None

    @classmethod
    def of(cls, value: Accuracy | float | None, calctype: CalcType) -> Accuracy | None:
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
        return cls(**{name: float(value)})


@dataclass(frozen=True)
class Request:
    """One evaluation of a session.

    Attributes:
        positions: Atomic positions (bohr), n_atoms x 3, atoms as the session's
            structure.
        calctype: ``energy``, ``gradient`` or ``hessian``.
        accuracy: Target accuracy of this request (a bare number targets the
            calctype's highest derivative). None: the session's settings.
        lineage: Which earlier evaluations this one continues: the session keeps
            warm-start and extrapolation history per lineage, so NEB images, IRC
            branches and parallel tasks each start from their own previous geometry.
            None: the nearest earlier geometry.
        outputs: Optional outputs that do not change the surface but may cost extra
            (``dipole``, ``ir``, ``polarizability``, ``orbitals``, ``density``, ...);
            a program rejects ones it does not support with kind ``unsupported``.
    """

    positions: np.ndarray
    calctype: CalcType = CalcType.gradient
    accuracy: Accuracy | float | None = None
    lineage: Hashable | None = None
    outputs: frozenset[str] = frozenset()


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


#: Opens a child session: (spec, structure, threads, calctypes) -> Session.
OpenSession = Callable[
    [ProgramSpec, Structure, int | None, "frozenset[CalcType] | None"], "Session"
]

_LOWER = {
    CalcType.energy: (CalcType.energy,),
    CalcType.gradient: (CalcType.energy, CalcType.gradient),
    CalcType.hessian: (CalcType.energy, CalcType.gradient, CalcType.hessian),
}


def session_calctypes(
    spec: ProgramSpec, calctypes: Iterable[CalcType | str] | None = None
) -> frozenset[CalcType]:
    """The calculation types a session may run.

    The declared ones (default: the spec's calctype), each with the lower derivatives
    it implies: a gradient session also returns energies, a Hessian session also
    gradients. A session checks its keywords against these at opening (a setting that
    applies to none of them is a ``not_applicable`` input error) and refuses requests
    of other types.

    Args:
        spec: The session's spec.
        calctypes: The calctypes declared at opening.

    Returns:
        The allowed calctypes.
    """
    declared = {CalcType(c) for c in calctypes} if calctypes else {spec.calctype}
    allowed: set[CalcType] = set()
    for c in declared:
        allowed.update(_LOWER.get(c, (c,)))
    return frozenset(allowed)


def check_session_calctype(
    calctype: CalcType | str, allowed: frozenset[CalcType]
) -> None:
    """Refuse a request of a calctype the session was not opened for.

    Raises:
        ProgramError: Kind ``input``, code ``not_applicable``.
    """
    if CalcType(calctype) not in allowed:
        raise ProgramError(
            ErrorInfo(
                kind=ErrorKind.input,
                code="not_applicable",
                message=f"This session was opened for {sorted(c.value for c in allowed)}, "
                f"not {CalcType(calctype).value}; open it with calctypes= that include it.",
            )
        )


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
        self,
        spec: ProgramSpec,
        structure: Structure,
        *,
        threads: int | None = None,
        calctypes: Iterable[CalcType | str] | None = None,
    ) -> Session:
        """Open a session for a child program under this calculation's grant.

        Args:
            spec: The child program, model and keywords.
            structure: The child's atoms, charge, and multiplicity, and its first
                geometry.
            threads: This session's share of the grant (a driver with 16 threads and
                8 tasks gives each session 2). None: the child's own settings.
            calctypes: The calctypes the session may run (see `session_calctypes`).

        Returns:
            The session; close it (or use it as a context manager) when done.

        Raises:
            RuntimeError: The runner provides neither sessions nor a compute callback.
        """
        allowed = session_calctypes(spec, calctypes)
        if self.open_session is not None:
            return self.open_session(spec, structure, threads, allowed)
        if self.compute is not None:
            return ComputeSession(self.compute, spec, structure, calctypes=allowed)
        raise RuntimeError(
            "This calculation computes child calculations, but its runner provided "
            "neither sessions nor a compute callback."
        )


@runtime_checkable
class Session(Protocol):
    """The same program and molecule at many geometries, warm-started.

    Opened by `SessionProgram.open_session` (or ``qccompute.session``). The session pins
    the atoms, charge, multiplicity, model and approximations (method, basis, grids,
    precision) for its lifetime, so all its evaluations lie on one potential energy
    surface; anything else needs a new session. A request may vary only how tightly it
    converges (`Request.accuracy`), which history it continues (`Request.lineage`) and
    which optional outputs it adds (`Request.outputs`). Contract:

    - Results are `Frame` objects: the structure evaluated, its `SinglePointData`, and
      the request's own ``warnings`` and ``record`` (an SCF escalation, a stability
      probe, a fallback), ready to store in a trajectory. ``program`` and
      ``program_version`` are the session's.
    - A failed request raises `ProgramError` from `compute` (and is a failed `Frame`
      from `compute_many`) and leaves the session usable, with the lineage's history
      untouched: a driver may retry a smaller step on the same session.
    - A ``hessian`` request returns the energy and gradient too.

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
        accuracy: Accuracy | float | None = None,
        lineage: Hashable | None = None,
        outputs: Iterable[str] = (),
    ) -> Frame:
        """Evaluate one geometry.

        Args:
            positions: Atomic positions (bohr), n_atoms x 3.
            calctype: ``energy``, ``gradient`` or ``hessian``.
            accuracy: See `Request.accuracy`.
            lineage: See `Request.lineage`.
            outputs: See `Request.outputs`.

        Returns:
            The evaluation, successful.

        Raises:
            ProgramError: The evaluation failed; the session stays usable.
        """
        ...

    def compute_many(self, requests: Sequence[Request]) -> list[Frame]:
        """Evaluate many geometries (finite differences, path images, IRC branches).

        Args:
            requests: The evaluations.

        Returns:
            One frame per request, in order; a failed request is a failed frame with
            its ``error``.
        """
        ...

    def reset(self, lineage: Hashable | None = None) -> None:
        """Drop warm-start and extrapolation history, after a large jump or a failure.

        Args:
            lineage: The lineage to reset; None resets every lineage.
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


def failed_frame(structure: Structure, error: ProgramError, program: str) -> Frame:
    """The failed `Frame` of a session request, for `Session.compute_many`.

    Args:
        structure: The structure that was evaluated.
        error: The request's failure.
        program: The session's program, for the empty results' provenance.

    Returns:
        A frame with ``success=False``, the error, and any partial results.
    """
    from .models import Provenance

    results = error.results or SinglePointData(provenance=Provenance(program=program))
    return Frame(
        structure=structure,
        results=results,
        success=False,
        traceback=error.error.message,
        error=error.error,
    )


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

    # Optional: ``numerics_id() -> str`` names the program's numerics identity (what
    # determines its bits: a numerics epoch, platform/libm class, eigensolver library) for
    # calculation stores to key on; the version is used when absent.

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
        self,
        spec: ProgramSpec,
        structure: Structure,
        ctx: RunContext,
        calctypes: frozenset[CalcType] | None = None,
    ) -> Session:
        """Open a session at `structure`'s atoms and first geometry.

        ``calctypes`` are the calculation types the session may run (None: see
        `session_calctypes`). The keywords are checked against them: a setting that
        applies to none of them is an input error, and so is a later request of
        another type.

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
        calctypes: The calctypes the session may run (see `session_calctypes`).
    """

    def __init__(
        self,
        compute: Callable[..., Any],
        spec: ProgramSpec,
        structure: Structure,
        calctypes: Iterable[CalcType | str] | None = None,
    ):
        """Create the session."""
        self._compute = compute
        self._spec = spec
        self._structure = structure
        self.calctypes = session_calctypes(spec, calctypes)
        self.program = spec.program
        self.program_version = ""
        self._closed = False

    def _input(self, positions: np.ndarray, calctype: CalcType) -> ProgramInput:
        check_session_calctype(calctype, self.calctypes)
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

    def _frame(self, out: ProgramOutput[Any, Any]) -> Frame:
        """The frame of a successful output; raise the failure of a failed one."""
        version = out.results.provenance.program_version
        if version and not self.program_version:
            self.program_version = version
        if not out.success:
            error = out.error or ErrorInfo(
                kind=ErrorKind.internal, message=out.traceback or "failed"
            )
            raise ProgramError(error, out.results)
        return Frame.from_output(out)

    def _complete(self, frame: Frame, positions: np.ndarray) -> Frame:
        """Add the energy and gradient to a Hessian computed without them."""
        if frame.results.gradient is not None:
            return frame
        gradient = self._frame(
            self._compute(self._input(positions, CalcType.gradient))
        ).results
        results = frame.results.model_copy(
            update={"gradient": gradient.gradient, "energy": gradient.energy}
        )
        return frame.model_copy(update={"results": results})

    def compute(
        self,
        positions: np.ndarray,
        calctype: CalcType = CalcType.gradient,
        *,
        accuracy: Accuracy | float | None = None,
        lineage: Hashable | None = None,
        outputs: Iterable[str] = (),
    ) -> Frame:
        """Evaluate one geometry; see `Session.compute`.

        Args:
            positions: Atomic positions (bohr).
            calctype: ``energy``, ``gradient`` or ``hessian``.
            accuracy: Ignored: a stateless session runs its spec's settings.
            lineage: Ignored: a stateless session keeps no history.
            outputs: Not supported; must be empty.

        Returns:
            The evaluation.

        Raises:
            ProgramError: The calculation failed, or outputs were requested.
        """
        if self._closed:
            raise RuntimeError("The session is closed.")
        if tuple(outputs):
            raise ProgramError.of(
                "unsupported",
                "A session over compute takes optional outputs from the spec's "
                "keywords, not per request.",
            )
        calctype = CalcType(calctype)
        frame = self._frame(self._compute(self._input(positions, calctype)))
        if calctype == CalcType.hessian:
            frame = self._complete(frame, positions)
        return frame

    def compute_many(self, requests: Sequence[Request]) -> list[Frame]:
        """Evaluate many geometries in one call to the callback.

        Args:
            requests: The evaluations.

        Returns:
            One frame per request, in order; failures as failed frames.
        """
        if self._closed:
            raise RuntimeError("The session is closed.")
        frames: list[Frame | None] = [None] * len(requests)
        inputs: list[ProgramInput] = []
        positions: list[int] = []
        for i, r in enumerate(requests):
            try:
                if r.outputs:
                    raise ProgramError.of(
                        "unsupported", "per-request outputs need a program session"
                    )
                inputs.append(self._input(r.positions, r.calctype))
                positions.append(i)
            except ProgramError as e:
                frames[i] = failed_frame(
                    self._structure.model_copy(
                        update={"geometry": np.asarray(r.positions, dtype=float)}
                    ),
                    e,
                    self.program,
                )
        outputs = self._compute(inputs) if inputs else []
        for i, inp, out in zip(positions, inputs, outputs, strict=True):
            assert inp.structure is not None
            try:
                frame = self._frame(out)
                if requests[i].calctype == CalcType.hessian:
                    frame = self._complete(frame, requests[i].positions)
                frames[i] = frame
            except ProgramError as e:
                frames[i] = failed_frame(inp.structure, e, self.program)
        return [f for f in frames if f is not None]

    def reset(self, lineage: Hashable | None = None) -> None:
        """Nothing to reset: a stateless session keeps no history."""

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


def local_context(
    programs: Mapping[str, Program] | None = None, *, threads: int | None = None
) -> RunContext:
    """A context that runs a driver's children in this process, one at a time.

    For calling a driver directly (tests, notebooks) without a runner. Children run
    through the given programs, else the installed ones (`find_program`): sessions of
    their own when they have them, else a `ComputeSession` over their `compute`.
    qccompute provides the full runner (parallel children, codecs, the cluster).

    Args:
        programs: Programs by name, ahead of the installed ones.
        threads: Threads granted to the driver; children get the same.

    Returns:
        The context.
    """
    known = dict(programs or {})

    def lookup(name: str) -> Program:
        program = known.get(name) or find_program(name)
        if program is None:
            raise ProgramError.of(
                "environment", f"No program {name!r} is installed or given."
            )
        return program

    def child_context() -> RunContext:
        return local_context(known, threads=threads)

    def compute_one(inp: ProgramInput) -> ProgramOutput[ProgramInput, Data]:
        try:
            program = lookup(inp.program)
        except ProgramError as e:
            from .models import Provenance

            return ProgramOutput(
                input_data=inp,
                success=False,
                traceback=e.error.message,
                error=e.error,
                results=SinglePointData(provenance=Provenance(program=inp.program))
                if inp.calctype
                in (CalcType.energy, CalcType.gradient, CalcType.hessian)
                else _empty_results(inp),
            )
        return program.compute(inp, child_context())

    def compute(inputs: Any, previous: Any = None) -> Any:
        if isinstance(inputs, ProgramInput):
            return compute_one(inputs)
        return [compute_one(i) for i in inputs]

    def open_session(
        spec: ProgramSpec,
        structure: Structure,
        share: int | None,
        calctypes: frozenset[CalcType] | None,
    ) -> Session:
        program = lookup(spec.program)
        if isinstance(program, SessionProgram) and program.capabilities().sessions:
            return program.open_session(
                spec, structure, RunContext(threads=share), calctypes
            )
        return ComputeSession(compute, spec, structure, calctypes=calctypes)

    return RunContext(
        threads=threads,
        compute=compute,  # type: ignore[arg-type]
        open_session=open_session,
    )


def _empty_results(inp: ProgramInput) -> Any:
    from .models import Provenance, get_data_type

    return get_data_type(inp.calctype)(provenance=Provenance(program=inp.program))


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
