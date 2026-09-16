from pathlib import Path

import numpy as np
import pytest

from qcdata import (
    FileInput,
    OptimizationData,
    ProgramInput,
    ProgramOutput,
    ProgramSpec,
    SinglePointData,
)
from qcdata.utils import water as water_struct


@pytest.fixture
def test_data_dir():
    """Test data directory Path"""
    return Path(__file__).parent / "data"


@pytest.fixture
def water():
    """Water Structure fixture"""
    return water_struct


@pytest.fixture
def file_input():
    return FileInput(
        program="terachem",
        files={"binary": b"binary data", "text": "text data"},
        cmdline_args=["-i", "input.dat", "-o", "output.dat"],
    )


@pytest.fixture
def input_data(request, file_input, prog_input_factory, nested_input_factory):
    """Input data fixture"""
    if request.param == "file_input":
        return file_input
    elif request.param == "calc_input":
        return prog_input_factory("energy")
    elif request.param == "nested_input":
        return nested_input_factory("optimization")
    else:
        raise ValueError(f"Unknown input data type: {request.param}")


@pytest.fixture
def prog_input_factory(water):
    """Function that returns ProgramInput of calctype."""

    def _create_prog_inp(calctype):
        return ProgramInput(
            program="terachem",
            structure=water,
            calctype=calctype,
            model={"method": "hf", "basis": "sto-3g"},
            keywords={
                "maxiter": 100,
                "purify": "no",
                "some-bool": False,
                "displacement": 1e-3,
                "thermo_temp": 298.15,
            },
        )

    return _create_prog_inp


@pytest.fixture
def nested_input_factory(water):
    """Function that returns a nested ProgramInput of calctype."""

    def _create_prog_inp(calctype):
        return ProgramInput(
            program="geometric",
            structure=water,
            calctype=calctype,
            keywords={
                "maxiter": 100,
                "purify": "no",
                "some-bool": False,
            },
            subprograms=[
                ProgramSpec(
                    program="terachem",
                    calctype="gradient",
                    model={"method": "hf", "basis": "sto-3g"},
                ),
            ],
        )

    return _create_prog_inp


@pytest.fixture
def sp_data():
    """SinglePointData object"""

    def _create_sp_data(structure):
        n_atoms = len(structure.symbols)
        gradient = np.arange(n_atoms * 3, dtype=np.float64).reshape(n_atoms, 3)
        hessian = np.arange(n_atoms**2 * 3**2, dtype=np.float64).reshape(
            n_atoms * 3, n_atoms * 3
        )
        return SinglePointData(
            provenance={"program": "qcdata-test-suite"},
            energy=1.0,
            gradient=gradient,
            hessian=hessian,
            extras={"some_extra_result": 1},
        )

    return _create_sp_data


@pytest.fixture
def prog_output(prog_input_factory, sp_data):
    """Successful ProgramOutput object"""
    pi_energy = prog_input_factory("energy")
    sp_data = sp_data(pi_energy.structure)

    return ProgramOutput[ProgramInput, SinglePointData](
        input_data=pi_energy,
        success=True,
        logs="program standard out...",
        results=sp_data,
        execution={"scratch_dir": "/tmp/qcdata"},
        extras={"some_extra": 1},
    )


@pytest.fixture
def results_failure(prog_input_factory, sp_data):
    """Failed ProgramOutput object"""
    pi_energy = prog_input_factory("energy")

    return ProgramOutput[ProgramInput, SinglePointData](
        input_data=pi_energy,
        success=False,
        traceback="Traceback...",
        results=SinglePointData(provenance={"program": "qcdata-test-suite"}),
        execution={"scratch_dir": "/tmp/qcdata"},
        extras={"some_extra": 1},
    )


@pytest.fixture
def opt_data(prog_output):
    return OptimizationData(
        provenance={"program": "qcdata-test-suite"}, trajectory=[prog_output]
    )


@pytest.fixture
def opt_output(prog_input_factory, opt_data):
    """Successful ProgramOutput object"""
    input_data = prog_input_factory("optimization")

    return ProgramOutput[ProgramInput, OptimizationData](
        input_data=input_data,
        success=True,
        logs="program standard out...",
        results=opt_data,
        execution={"scratch_dir": "/tmp/qcdata"},
        extras={"some_extra": 1},
    )
