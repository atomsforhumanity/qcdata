"""The typed slots of the AFH program interface: outcome, vibrations, orbitals."""

import math
from typing import Any

import numpy as np
import pytest
from pydantic import ValidationError

from qcdata import (
    DEFAULT_AO_CONVENTION,
    MOLDEN_CONVENTION,
    AOConvention,
    BasisSet,
    CalcType,
    CartesianNormalization,
    ErrorInfo,
    ErrorKind,
    ExecutionInfo,
    Frame,
    OptimizationData,
    Orbitals,
    OrbitalSpin,
    ProgramInput,
    ProgramOutput,
    ProgramWarning,
    Provenance,
    Resolved,
    Shell,
    SinglePointData,
    SphericalOrder,
    Thermochemistry,
    ThermoTerm,
    VibrationalAnalysis,
    Wavefunction,
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
    saved = OptimizationData(provenance=PROV, frames=[out]).model_dump(mode="json")
    saved["frames"] = [out.model_dump(mode="json")]  # a ProgramOutput saved as a step
    loaded = OptimizationData.model_validate(saved)
    assert loaded.frames[0].error.message == "SCF"


# -------------------------------------------------------------- single points


def test_single_point_typed_fields():
    sp = SinglePointData(
        provenance=PROV,
        energy=-1.0,
        energy_components={"nuclear_repulsion": 0.5, "one_electron": -2.0, "xc": 0.5},
        scf_iterations=9,
        scf_quadrupole_moment=np.arange(9.0),
        quadrupole_kind="traceless",
        polarizability=[[1, 0, 0], [0, 2, 0], [0, 0, 3]],
        atomic_charges=[0.1, -0.1],
        atomic_charges_kind="mulliken",
        bond_orders=[1.0, 0.9, 0.9, 1.0],
        bond_orders_kind="mayer",
    )
    assert sum(sp.energy_components.values()) == pytest.approx(sp.energy)
    assert sp.scf_quadrupole_moment.shape == (3, 3)
    assert sp.bond_orders.shape == (2, 2)
    assert SinglePointData.model_validate_json(sp.model_dump_json()) == sp
    with pytest.raises(ValidationError):
        SinglePointData(provenance=PROV, polarizability=[1.0, 2.0])


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
    # Files saved before 0.24 wrote the old fields empty: they still load.
    old_empty = {"freqs_wavenumber": [], "normal_modes_cartesian": None}
    assert (
        SinglePointData.model_validate({"provenance": PROV, **old_empty}).energy is None
    )
    # Filled ones cannot be converted, and say where the data lives now.
    with pytest.raises(ValidationError, match="vibrational_analysis"):
        SinglePointData.model_validate({"provenance": PROV, "freqs_wavenumber": [1.0]})
    assert "freqs_wavenumber" not in SinglePointData.model_fields


# ------------------------------------------------------------------- orbitals


def _norm(alpha: float) -> float:
    """Coefficient that normalizes one s primitive."""
    return (2 * alpha / math.pi) ** 0.75


def _basis(pure_d: bool = True) -> BasisSet:
    shells = [
        Shell(
            center=0,
            angular_momentum=0,
            pure=True,
            exponents=np.array([1.0]),
            coefficients=np.array([_norm(1.0)]),
        ),
        Shell(
            center=0,
            angular_momentum=1,
            pure=True,
            exponents=np.array([0.8]),
            coefficients=np.array([1.0]),
        ),
        Shell(
            center=0,
            angular_momentum=2,
            pure=pure_d,
            exponents=np.array([0.6]),
            coefficients=np.array([1.0]),
        ),
        Shell(
            center=1,
            angular_momentum=0,
            pure=True,
            exponents=np.array([0.5]),
            coefficients=np.array([_norm(0.5)]),
        ),
    ]
    return BasisSet(name="toy", shells=shells)


def _orbitals(basis: BasisSet, spin: str = "restricted") -> Orbitals:
    rng = np.random.default_rng(1)
    n = basis.nao
    c = np.linalg.qr(rng.normal(size=(n, n)))[0]
    occ = np.array([1.0] * 3 + [0.0] * (n - 3))
    kw: dict[str, Any] = {}
    if spin == "unrestricted":
        kw = dict(coefficients_b=c[:, ::-1], energies_b=np.arange(n, dtype=float))
    return Orbitals(
        basis=basis,
        spin=OrbitalSpin(spin),
        coefficients_a=c,
        energies_a=np.arange(n, dtype=float),
        occupations_a=occ,
        **kw,
    )


def test_basis_counts_and_labels():
    pure, cart = _basis(True), _basis(False)
    assert pure.nao == 1 + 3 + 5 + 1
    assert cart.nao == 1 + 3 + 6 + 1
    assert pure.shell_offsets() == [0, 1, 4, 9]
    labels = pure.ao_labels(["O", "H"])
    assert labels[0] == "0 O 1s" and labels[1] == "0 O 2p x" and labels[-1] == "1 H 1s"
    assert "0 O 3dxy" in cart.ao_labels(["O", "H"])
    s = pure.shells[0]
    assert s.primitive_coefficients() == pytest.approx([1.0])


def test_shell_and_orbital_validation():
    with pytest.raises(ValidationError, match="same length"):
        Shell(
            center=0,
            angular_momentum=0,
            pure=True,
            exponents=np.array([1.0, 2.0]),
            coefficients=np.array([1.0]),
        )
    with pytest.raises(ValidationError, match="positive"):
        Shell(
            center=0,
            angular_momentum=0,
            pure=True,
            exponents=np.array([-1.0]),
            coefficients=np.array([1.0]),
        )
    basis = _basis()
    with pytest.raises(ValidationError, match="nao x nmo"):
        Orbitals(basis=basis, spin="restricted", coefficients_a=np.zeros((3, 3)))
    with pytest.raises(ValidationError, match="need coefficients_b"):
        Orbitals(basis=basis, spin="unrestricted", coefficients_a=np.eye(basis.nao))
    with pytest.raises(ValidationError, match="one value per orbital"):
        Orbitals(
            basis=basis,
            spin="restricted",
            coefficients_a=np.eye(basis.nao),
            energies_a=[1.0],
        )


@pytest.mark.parametrize("pure_d", [True, False])
@pytest.mark.parametrize("spin", ["restricted", "unrestricted"])
def test_conventions_round_trip(pure_d, spin):
    orb = _orbitals(_basis(pure_d), spin)
    for conv in (
        MOLDEN_CONVENTION,
        AOConvention(spherical_order=SphericalOrder.increasing_m),
        AOConvention(spherical_order="trexio", cartesian_normalization="pyscf"),
    ):
        there = orb.with_convention(conv)
        assert there.basis.ao_convention == conv
        back = there.with_convention(DEFAULT_AO_CONVENTION)
        assert np.allclose(back.coefficients_a, orb.coefficients_a)
        if spin == "unrestricted":
            assert np.allclose(back.coefficients_b, orb.coefficients_b)


def test_convention_permutations_and_scaling():
    basis = _basis(pure_d=False)
    eye = np.eye(basis.nao)
    # p shells: PySCF (x, y, z) -> CCA increasing m (y, z, x)
    cca = basis.transform(eye, AOConvention(spherical_order="increasing_m"))
    assert cca[1:4, 1:4].argmax(axis=1).tolist() == [1, 2, 0]  # y, z, x
    # Cartesian d: x^l-normalized xy has norm 1/sqrt(3); unit-normalized has norm 1.
    unit = basis.transform(
        eye, AOConvention(cartesian_normalization=CartesianNormalization.unit)
    )
    labels = basis.ao_labels()
    xy = labels.index("0 3dxy")
    xx = labels.index("0 3dxx")
    assert unit[xy, xy] == pytest.approx(1 / math.sqrt(3))
    assert unit[xx, xx] == pytest.approx(1.0)


def test_density_and_wavefunction_round_trip():
    orb = _orbitals(_basis())
    da, db = orb.density()
    c = orb.coefficients_a[:, :3]
    assert np.allclose(da, c @ c.T) and np.allclose(da, db)
    assert orb.spin is OrbitalSpin.restricted and orb.restricted
    sp = SinglePointData(
        provenance=PROV,
        energy=-1.0,
        wavefunction=Wavefunction(
            scf_eigenvalues_a=orb.energies_a,
            scf_occupations_a=orb.occupations_a,
            orbitals=orb,
        ),
    )
    loaded = SinglePointData.model_validate_json(sp.model_dump_json())
    assert loaded == sp
    assert np.array_equal(
        loaded.wavefunction.orbitals.coefficients_a, orb.coefficients_a
    )
