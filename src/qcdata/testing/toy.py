"""A complete toy program: Lennard-Jones energies, gradients, and Hessians.

It implements every part of the `qcdata.program` protocol (settings schema with
``"auto"``, plan, compute with typed errors, sessions, batches) in a few hundred lines, as
an example for program authors and as a stand-in program for runners' tests. Register it
with an entry point to try it under qccompute::

    [project.entry-points."qcdata.programs"]
    toy = "qcdata.testing.toy:program"
"""

from __future__ import annotations

import difflib
import time
from collections.abc import Hashable, Iterable, Mapping, Sequence
from types import TracebackType
from typing import Any

import numpy as np

from .. import _version
from ..models import (
    CalcType,
    Data,
    ErrorInfo,
    ErrorKind,
    ExecutionInfo,
    Frame,
    ProgramInput,
    ProgramOutput,
    ProgramSpec,
    Provenance,
    Resolved,
    SinglePointData,
    Structure,
)
from ..program import (
    Accuracy,
    Capabilities,
    Plan,
    ProgramError,
    Request,
    Resources,
    RunContext,
    check_session_calctype,
    failed_frame,
    session_calctypes,
)

__all__ = ["ToyProgram", "ToySession", "program"]

NAME = "toy"
CALCTYPES = frozenset({CalcType.energy, CalcType.gradient, CalcType.hessian})
DEFAULT_SIGMA = 6.0  # bohr

_SETTINGS_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "epsilon": {
            "type": "number",
            "exclusiveMinimum": 0,
            "default": 0.01,
            "description": "Well depth.",
            "x-unit": "Eh",
            "x-tier": "common",
            "x-concern": "model",
        },
        "sigma": {
            "anyOf": [{"type": "number", "exclusiveMinimum": 0}, {"const": "auto"}],
            "default": "auto",
            "description": "Distance at which the pair energy is zero; auto: 6 bohr.",
            "x-unit": "bohr",
            "x-tier": "common",
            "x-concern": "model",
        },
        "threads": {
            "type": "integer",
            "minimum": 1,
            "default": 1,
            "description": "CPU threads.",
            "x-tier": "common",
            "x-concern": "resource",
        },
        "output": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "pair_energies": {
                    "type": "boolean",
                    "default": False,
                    "description": "Put every pair's energy in the record.",
                    "x-tier": "expert",
                    "x-concern": "output",
                }
            },
        },
        "fail": {
            "enum": [k.value for k in ErrorKind] + [None],
            "default": None,
            "description": "Fail with this error kind (for testing runners).",
            "x-tier": "dev",
            "x-concern": "algorithm",
        },
    },
}

_RECORD_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "properties": {
        "pairs": {"type": "integer", "description": "Number of atom pairs."},
        "pair_energies": {"type": "array", "items": {"type": "number"}},
        "evaluation": {
            "type": "integer",
            "description": "Index of the evaluation within its session (0: first).",
        },
    },
}


def _check_keys(
    given: Mapping[str, Any], schema: Mapping[str, Any], prefix: str = ""
) -> None:
    allowed = schema["properties"]
    for key, value in given.items():
        if key not in allowed:
            close = difflib.get_close_matches(key, list(allowed), n=1)
            hint = f" Did you mean '{prefix}{close[0]}'?" if close else ""
            raise ProgramError(
                ErrorInfo(
                    kind=ErrorKind.input,
                    code="unknown_key",
                    message=f"Unknown setting '{prefix}{key}'.{hint}",
                )
            )
        if allowed[key].get("type") == "object":
            if not isinstance(value, Mapping):
                raise ProgramError.of("input", f"{prefix}{key} must be a group (dict).")
            _check_keys(value, allowed[key], f"{prefix}{key}.")


def _resolve(spec: ProgramSpec) -> Plan:
    """Validate a spec's model and keywords and resolve every auto."""
    if spec.calctype not in CALCTYPES:
        raise ProgramError.of(
            "unsupported",
            f"{NAME} computes {sorted(CALCTYPES)}, not {spec.calctype.value}.",
        )
    if spec.model is None or spec.model.method.lower() != "lj":
        raise ProgramError.of("input", f"{NAME} has one method, 'lj'.")
    kw = dict(spec.keywords)
    _check_keys(kw, _SETTINGS_SCHEMA)
    epsilon = kw.get("epsilon", 0.01)
    if not isinstance(epsilon, int | float) or epsilon <= 0:
        raise ProgramError.of("input", "epsilon must be a positive number (Eh).")
    sigma = kw.get("sigma", "auto")
    reasons = {}
    if sigma == "auto" or sigma is None:
        sigma = DEFAULT_SIGMA
        reasons["sigma"] = f"auto: {DEFAULT_SIGMA} bohr for every element"
    elif not isinstance(sigma, int | float) or sigma <= 0:
        raise ProgramError.of(
            "input", "sigma must be a positive number (bohr) or 'auto'."
        )
    threads = kw.get("threads", 1)
    settings: dict[str, Any] = {
        "epsilon": float(epsilon),
        "sigma": float(sigma),
        "threads": int(threads),
        "output": {
            "pair_energies": bool(kw.get("output", {}).get("pair_energies", False))
        },
    }
    if kw.get("fail") is not None:
        settings["fail"] = ErrorKind(kw["fail"]).value
    return Plan(
        resolved=Resolved(settings=settings, reasons=reasons),
        resources=Resources(threads=int(threads)),
    )


def _lj(
    positions: np.ndarray, epsilon: float, sigma: float, calctype: CalcType
) -> dict[str, Any]:
    """Energy terms, gradient, and Hessian of the pairwise Lennard-Jones potential."""
    x = np.asarray(positions, dtype=float).reshape(-1, 3)
    if not np.all(np.isfinite(x)):
        raise ProgramError.of(
            "numerical", "positions are not finite", code="bad_positions"
        )
    n = len(x)
    repulsion = dispersion = 0.0
    gradient = np.zeros((n, 3))
    hessian = np.zeros((3 * n, 3 * n))
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            d = x[i] - x[j]
            r = float(np.linalg.norm(d))
            if r < 1e-6:
                raise ProgramError.of("input", f"atoms {i} and {j} coincide")
            s6 = (sigma / r) ** 6
            rep, disp = 4 * epsilon * s6 * s6, -4 * epsilon * s6
            repulsion += rep
            dispersion += disp
            pairs.append(rep + disp)
            if calctype == CalcType.energy:
                continue
            de = 4 * epsilon * (-12 * s6 * s6 + 6 * s6) / r
            u = d / r
            gradient[i] += de * u
            gradient[j] -= de * u
            if calctype == CalcType.hessian:
                d2e = 4 * epsilon * (156 * s6 * s6 - 42 * s6) / r**2
                block = d2e * np.outer(u, u) + de / r * (np.eye(3) - np.outer(u, u))
                a, b = slice(3 * i, 3 * i + 3), slice(3 * j, 3 * j + 3)
                hessian[a, a] += block
                hessian[b, b] += block
                hessian[a, b] -= block
                hessian[b, a] -= block
    out: dict[str, Any] = {
        "energy": repulsion + dispersion,
        "energy_components": {"repulsion": repulsion, "dispersion": dispersion},
        "pairs": pairs,
    }
    if calctype != CalcType.energy:
        out["gradient"] = gradient
    if calctype == CalcType.hessian:
        out["hessian"] = hessian
    return out


def _single_point(values: Mapping[str, Any]) -> SinglePointData:
    return SinglePointData(
        provenance=Provenance(program=NAME, program_version=_version.__version__),
        energy=values["energy"],
        gradient=values.get("gradient"),
        hessian=values.get("hessian"),
        energy_components=values["energy_components"],
    )


def _failure(
    inp: ProgramInput, error: ErrorInfo, plan: Plan | None, start: float
) -> ProgramOutput[ProgramInput, Data]:
    return ProgramOutput(
        input_data=inp,
        success=False,
        traceback=error.message,
        error=error,
        results=SinglePointData(
            provenance=Provenance(program=NAME, program_version=_version.__version__)
        ),
        resolved=plan.resolved if plan is not None else None,
        execution=ExecutionInfo(timings={"total": time.perf_counter() - start}),
    )


class ToySession:
    """A Lennard-Jones session: settings pinned at opening, history kept per lineage.

    The potential needs no warm start, but the session keeps what a real program
    would (the last geometry of each lineage) and reports it in each frame's record,
    so drivers and runners can test lineage handling.
    """

    def __init__(
        self, plan: Plan, structure: Structure, calctypes: frozenset[CalcType]
    ):
        """Open the session.

        Args:
            plan: The pinned plan.
            structure: The atoms, charge, and multiplicity of every request.
            calctypes: The calctypes the session may run.
        """
        self.calctypes = calctypes
        self.program = NAME
        self.program_version = _version.__version__
        self._plan = plan
        self._structure = structure
        self._history: dict[Hashable, np.ndarray] = {}
        self._closed = False

    def compute(
        self,
        positions: np.ndarray,
        calctype: CalcType = CalcType.gradient,
        *,
        accuracy: Accuracy | float | None = None,
        lineage: Hashable | None = None,
        outputs: Iterable[str] = (),
    ) -> Frame:
        """Evaluate one geometry; see `qcdata.program.Session.compute`.

        Args:
            positions: Atomic positions (bohr).
            calctype: ``energy``, ``gradient`` or ``hessian``.
            accuracy: Ignored: the potential is exact.
            lineage: The history this evaluation continues.
            outputs: ``pair_energies`` puts every pair's energy in the record.

        Returns:
            The frame; a hessian also has the energy and gradient.

        Raises:
            ProgramError: The geometry cannot be evaluated; the session stays usable
                and the lineage's history is unchanged.
        """
        if self._closed:
            raise RuntimeError("The session is closed.")
        check_session_calctype(calctype, self.calctypes)
        wanted = set(outputs)
        if wanted - {"pair_energies"}:
            raise ProgramError.of(
                "unsupported",
                f"{NAME} has no outputs {sorted(wanted - {'pair_energies'})}",
            )
        x = np.asarray(positions, dtype=float).reshape(-1, 3)
        if len(x) != len(self._structure.symbols):
            raise ProgramError.of(
                "input", f"expected {len(self._structure.symbols)} atoms, got {len(x)}"
            )
        s = self._plan.resolved.settings
        if "fail" in s:
            raise ProgramError.of(s["fail"], "failed on request (keyword fail)")
        values = _lj(x, s["epsilon"], s["sigma"], CalcType(calctype))
        if CalcType(calctype) == CalcType.hessian:
            values["gradient"] = _lj(x, s["epsilon"], s["sigma"], CalcType.gradient)[
                "gradient"
            ]
        previous = self._history.get(lineage)
        record: dict[str, Any] = {
            "warm_start": previous is not None,
            "step_from_previous": None
            if previous is None
            else float(np.linalg.norm(x - previous)),
        }
        if "pair_energies" in wanted:
            record["pair_energies"] = values["pairs"]
        self._history[lineage] = x  # only after success: failures leave history alone
        return Frame(
            structure=self._structure.model_copy(update={"geometry": x}),
            results=_single_point(values),
            success=True,
            record=record,
        )

    def compute_many(self, requests: Sequence[Request]) -> list[Frame]:
        """Evaluate many geometries; failures are failed frames, in place.

        Args:
            requests: The evaluations.

        Returns:
            One frame per request, in order.
        """
        frames: list[Frame] = []
        for r in requests:
            try:
                frames.append(
                    self.compute(
                        r.positions,
                        r.calctype,
                        accuracy=r.accuracy,
                        lineage=r.lineage,
                        outputs=r.outputs,
                    )
                )
            except ProgramError as e:
                structure = self._structure.model_copy(
                    update={"geometry": np.asarray(r.positions, dtype=float)}
                )
                frames.append(failed_frame(structure, e, NAME))
        return frames

    def reset(self, lineage: Hashable | None = None) -> None:
        """Forget the history of one lineage, or of all.

        Args:
            lineage: The lineage; None forgets every lineage.
        """
        if lineage is None:
            self._history.clear()
        else:
            self._history.pop(lineage, None)

    def plan(self) -> Plan:
        """The pinned plan."""
        return self._plan

    def close(self) -> None:
        """Close the session."""
        self._closed = True

    def __enter__(self) -> ToySession:
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


class ToyProgram:
    """The Lennard-Jones toy program (``ProgramInput(program="toy", model={"method": "lj"})``)."""

    name = NAME

    def version(self) -> str:
        """The program's version (qcdata's)."""
        return _version.__version__

    def capabilities(self) -> Capabilities:
        """Energies, gradients, Hessians; sessions and batches; in process."""
        return Capabilities(
            calctypes=CALCTYPES,
            methods=["lj"],
            sessions=True,
            batches=True,
            in_process=True,
            thread_safe=True,
        )

    def settings_schema(self, calctype: CalcType) -> dict[str, Any]:
        """The JSON Schema of the keywords (the same for every calctype)."""
        return _SETTINGS_SCHEMA

    def record_schema(self, calctype: CalcType) -> dict[str, Any]:
        """The JSON Schema of the record."""
        return _RECORD_SCHEMA

    def plan(self, inp: ProgramInput) -> Plan:
        """Validate the input and resolve its settings.

        Raises:
            ProgramError: Unknown keys (with a suggestion), bad values, an
                unsupported calctype, or a method other than ``lj``.
        """
        if inp.structure is None:
            raise ProgramError.of("input", f"{NAME} needs a structure.")
        return _resolve(inp)

    def compute(
        self, inp: ProgramInput, ctx: RunContext
    ) -> ProgramOutput[ProgramInput, Data]:
        """Compute an energy, gradient, or Hessian; failures are returned, not raised."""
        start = time.perf_counter()
        plan = None
        try:
            plan = self.plan(inp)
            s = plan.resolved.settings
            if "fail" in s:
                raise ProgramError.of(s["fail"], "failed on request (keyword fail)")
            assert inp.structure is not None
            values = _lj(inp.structure.geometry, s["epsilon"], s["sigma"], inp.calctype)
            if inp.calctype == CalcType.hessian:
                values["gradient"] = _lj(
                    inp.structure.geometry, s["epsilon"], s["sigma"], CalcType.gradient
                )["gradient"]
        except ProgramError as e:
            return _failure(inp, e.error, plan, start)
        record: dict[str, Any] = {"pairs": len(values["pairs"]), "evaluation": 0}
        if s["output"]["pair_energies"]:
            record["pair_energies"] = values["pairs"]
        return ProgramOutput(
            input_data=inp,
            success=True,
            results=_single_point(values),
            resolved=plan.resolved,
            record=record,
            execution=ExecutionInfo(timings={"total": time.perf_counter() - start}),
        )

    def compute_batch(
        self, inputs: Sequence[ProgramInput], ctx: RunContext
    ) -> list[ProgramOutput[ProgramInput, Data]]:
        """Compute many inputs; one output per input, in order."""
        return [self.compute(inp, ctx) for inp in inputs]

    def open_session(
        self,
        spec: ProgramSpec,
        structure: Structure,
        ctx: RunContext,
        calctypes: frozenset[CalcType] | None = None,
    ) -> ToySession:
        """Open a session pinned to the spec's settings, for the given calctypes.

        Raises:
            ProgramError: The spec cannot run.
        """
        allowed = session_calctypes(spec, calctypes)
        unsupported = allowed - CALCTYPES
        if unsupported:
            raise ProgramError.of(
                "unsupported", f"{NAME} sessions cannot run {sorted(unsupported)}"
            )
        return ToySession(_resolve(spec), structure, allowed)


#: The object to register under the ``qcdata.programs`` entry points.
program = ToyProgram()
