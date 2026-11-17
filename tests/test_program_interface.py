"""The typed slots of the AFH program interface: outcome and vibrations."""

import numpy as np
import pytest
from pydantic import ValidationError

from qcdata import (
    CalcType,
    ErrorInfo,
    ErrorKind,
    ExecutionInfo,
    Frame,
    OptimizationData,
    ProgramInput,
    ProgramOutput,
    ProgramWarning,
    Provenance,
    Resolved,
    SinglePointData,
    Thermochemistry,
    ThermoTerm,
    VibrationalAnalysis,
)

PROV = Provenance(program="test", program_version="1.0", program_commit="abc123")


@pytest.fixture
def energy_input(water):
    return ProgramInput(
        program="test",
        calctype=CalcType.energy,
        structure=water,
        model={"method": "hf", "basis": "sto-3g"},
        keywords={"scf": {"max_iter": 50, "e_tol": "auto"}, "threads": 2},
    )


# --------------------------------------------------------------------- outcome


def test_output_slots_round_trip(energy_input):
    out = ProgramOutput(
        input_data=energy_input,
        success=True,
        results=SinglePointData(provenance=PROV, energy=-75.0),
        resolved=Resolved(
            settings={"scf": {"max_iter": 50, "e_tol": 1e-8}, "threads": 2},
            reasons={"scf.e_tol": "calctype energy: tight"},
        ),
        warnings=[ProgramWarning(code="ignored", message="no effect", path="x.y")],
        record={"scf": {"iterations": 12}},
        execution=ExecutionInfo(timings={"total": 1.5, "scf": 1.2}),
    )
    loaded = ProgramOutput.model_validate_json(out.model_dump_json())
    assert loaded == out
    assert loaded.resolved.settings["scf"]["e_tol"] == 1e-8
    assert loaded.warnings[0].path == "x.y"
    assert loaded.execution.timings["total"] == 1.5
    assert loaded.results.provenance.program_commit == "abc123"
    # Reading order: success first; warnings and error before the logs.
    keys = list(out.model_dump())
    assert keys[0] == "success"
    assert keys.index("results") < keys.index("warnings") < keys.index("logs")


def test_error_slot_only_on_failures(energy_input):
    error = ErrorInfo(
        kind=ErrorKind.program,
        message="the gradient failed",
        program="linus",
        cause=ErrorInfo(kind="convergence", message="SCF", code="max_iter"),
    )
    failed = ProgramOutput(
        input_data=energy_input,
        success=False,
        traceback="...",
        results=SinglePointData(provenance=PROV),
        error=error,
    )
    loaded = ProgramOutput.model_validate_json(failed.model_dump_json())
    assert loaded.error.cause.kind is ErrorKind.convergence
    assert loaded.error.cause.code == "max_iter"
    with pytest.raises(ValidationError, match="successful calculation has no error"):
        ProgramOutput(
            input_data=energy_input,
            success=True,
            results=SinglePointData(provenance=PROV, energy=-75.0),
            error=error,
        )
    with pytest.raises(ValidationError):
        ErrorInfo(kind="not_a_kind", message="x")


def test_frames_keep_warnings_errors_and_records_but_not_settings(energy_input):
    out = ProgramOutput(
        input_data=energy_input,
        success=False,
        traceback="...",
        results=SinglePointData(provenance=PROV),
        error=ErrorInfo(kind="convergence", message="SCF"),
        warnings=[ProgramWarning(code="w", message="m")],
        resolved=Resolved(settings={"a": 1}),
        record={"r": 1},
    )
    frame = Frame.from_output(out)
    assert frame.error.kind is ErrorKind.convergence
    assert frame.warnings[0].code == "w"
    assert frame.record == {"r": 1}
    assert not hasattr(frame, "resolved")
    data = OptimizationData(provenance=PROV, frames=[out])  # an output passed as a step
    assert data.frames[0].error.message == "SCF"
    saved = data.model_dump(mode="json")
    saved["frames"] = [out.model_dump(mode="json")]  # a saved output is not a frame
    with pytest.raises(ValidationError):
        OptimizationData.model_validate(saved)


# -------------------------------------------------------------- single points


def test_single_point_typed_fields():
    sp = SinglePointData(
        provenance=PROV,
        energy=-1.0,
        energy_components={"nuclear_repulsion": 0.5, "one_electron": -2.0, "xc": 0.5},
        scf_iterations=9,
        quadrupole_moment=np.arange(9.0),
        polarizability=[[1, 0, 0], [0, 2, 0], [0, 0, 3]],
        atomic_charges=[0.1, -0.1],
        atomic_charges_kind="mulliken",
        bond_orders=[1.0, 0.9, 0.9, 1.0],
        bond_orders_kind="mayer",
    )
    assert sum(sp.energy_components.values()) == pytest.approx(sp.energy)
    assert sp.quadrupole_moment.shape == (3, 3)
    assert sp.bond_orders.shape == (2, 2)
    assert SinglePointData.model_validate_json(sp.model_dump_json()) == sp
    with pytest.raises(ValidationError):
        SinglePointData(provenance=PROV, polarizability=[1.0, 2.0])


def test_moments_are_the_methods_whatever_the_method():
    """A force field's dipole has the same home as an SCF one; the old names are gone."""
    sp = SinglePointData(provenance=PROV, dipole_moment=[0.0, 0.0, 0.73])
    assert sp.dipole_moment == [0.0, 0.0, 0.73]
    for old in ("scf_dipole_moment", "scf_quadrupole_moment"):
        with pytest.raises(ValidationError, match=old):
            SinglePointData.model_validate({"provenance": PROV, old: [0.0, 0.0, 1.0]})


def _water_analysis(**kw):
    rng = np.random.default_rng(0)
    modes = rng.normal(size=(3, 3, 3))
    modes /= np.linalg.norm(modes.reshape(3, -1), axis=1)[:, None, None]
    values = dict(
        masses=[15.995, 1.008, 1.008],
        projection="translations_rotations",
        frequencies=[-30.0, 1600.0, 3700.0],
        normal_modes_cartesian=modes.reshape(-1),
        reduced_masses=[1.1, 1.08, 1.04],
        ir_intensities=[0.0, 70.0, 20.0],
        dipole_derivatives=np.zeros(27),
        rotor_type="nonlinear",
        point_group="C2v",
        symmetry_number=2,
        symmetry_number_source="detected",
    )
    values.update(kw)
    return VibrationalAnalysis(**values)


def test_vibrational_analysis_shapes_and_imaginary_cutoff():
    va = _water_analysis()
    assert va.normal_modes_cartesian.shape == (3, 3, 3)
    assert va.dipole_derivatives.shape == (9, 3)
    assert va.n_imaginary() == 1
    assert va.n_imaginary(cutoff=50.0) == 0
    assert va.imaginary_frequencies().tolist() == [-30.0]
    with pytest.raises(ValidationError, match="normal_modes_cartesian must be"):
        _water_analysis(normal_modes_cartesian=np.zeros(10))
    with pytest.raises(ValidationError, match="one value per mode"):
        _water_analysis(reduced_masses=[1.0])
    with pytest.raises(ValidationError, match="one flag per mode"):
        _water_analysis(external=[True])


def test_external_modes_are_not_vibrations():
    modes = np.zeros((4, 3, 3))
    va = VibrationalAnalysis(
        masses=[16.0, 1.0, 1.0],
        projection="none",
        frequencies=[-5.0, 2.0, 1600.0, 3700.0],
        external=[True, True, False, False],
        normal_modes_cartesian=modes,
    )
    assert va.vibrational_frequencies.tolist() == [1600.0, 3700.0]
    assert va.vibrational_modes_cartesian.shape == (2, 3, 3)
    assert va.n_imaginary() == 0


def _thermo(**kw):
    values = dict(
        temperature=298.15,
        pressure=101325.0,
        electronic_energy=-76.0,
        zero_point_energy=0.021,
        total=ThermoTerm(energy=-75.976, enthalpy=-75.975, entropy=7e-5, gibbs=-75.996),
        entropy_model="qrrho_grimme",
        qrrho_frequency=100.0,
    )
    values.update(kw)
    return Thermochemistry(**values)


def test_thermochemistry_corrections():
    th = _thermo()
    assert th.gibbs_correction == pytest.approx(0.004)
    assert th.enthalpy_correction == pytest.approx(0.025)
    assert th.thermal_energy_correction == pytest.approx(0.024)
    assert th.gibbs_free_energy == -75.996
    with pytest.raises(ValidationError, match="not both"):
        _thermo(concentration=1.0)


def test_frequencies_live_only_in_the_analysis():
    """Pre-v1: no compatibility copies; frequencies, modes and free energies have one home."""
    va = _water_analysis()
    sp = SinglePointData(
        provenance=PROV,
        energy=-76.0,
        hessian=np.eye(9),
        vibrational_analysis=va,
        thermochemistry=[_thermo(), _thermo(temperature=310.0)],
    )
    loaded = SinglePointData.model_validate_json(sp.model_dump_json())
    assert loaded == sp
    assert loaded.vibrational_analysis.vibrational_frequencies.tolist() == [
        -30.0,
        1600.0,
        3700.0,
    ]
    assert loaded.thermochemistry[1].temperature == 310.0
    # The fields before 0.24 are not accepted, even empty.
    for old in ({"freqs_wavenumber": []}, {"normal_modes_cartesian": None}):
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            SinglePointData.model_validate({"provenance": PROV, **old})
    assert "freqs_wavenumber" not in SinglePointData.model_fields
