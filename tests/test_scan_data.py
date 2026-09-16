import numpy as np
import pytest

from qcdata import (
    OptimizationData,
    ProgramOutput,
    ScanData,
    SinglePointData,
    __version__,
)


def test_scan_data_properties(opt_output):
    """Test that the number of energies matches the number of conformers"""
    # No energies is fine
    scan_res = ScanData(
        provenance={"program": "qcdata-test-suite"},
        trajectory=[opt_output],
    )

    # Test properties
    assert scan_res.energies == [opt_output.results.final_energy]
    assert scan_res.structures == [opt_output.results.final_structure]
    # Test custom __repr_args__
    repr_args = scan_res.__repr_args__()
    assert isinstance(repr_args, list)
    for arg in repr_args:
        assert isinstance(arg, tuple)
        assert len(arg) == 2
        assert isinstance(arg[0], str)
        assert isinstance(arg[1], str)


def test_scan_save_to_xyz(opt_output, tmp_path):
    scan_res = ScanData(
        provenance={"program": "qcdata-test-suite"},
        trajectory=[opt_output] * 3,
    )
    scan_res.save(tmp_path / "scan_res.xyz")

    text = (tmp_path / "scan_res.xyz").read_text()

    # Text must be de-dented exactly as below
    correct_text = f"""3
qcdata_charge=0 qcdata_multiplicity=1 qcdata_version={__version__} qcdata__identifiers_name=water
O  0.01340919176202180 0.01026321207824930 -0.00368477733600419
H  0.12112430307330672 0.97600619725464122 0.08599884278042236
H  0.75016279902412597 -0.33132205318865016 -0.54481406902570462
3
qcdata_charge=0 qcdata_multiplicity=1 qcdata_version={__version__} qcdata__identifiers_name=water
O  0.01340919176202180 0.01026321207824930 -0.00368477733600419
H  0.12112430307330672 0.97600619725464122 0.08599884278042236
H  0.75016279902412597 -0.33132205318865016 -0.54481406902570462
3
qcdata_charge=0 qcdata_multiplicity=1 qcdata_version={__version__} qcdata__identifiers_name=water
O  0.01340919176202180 0.01026321207824930 -0.00368477733600419
H  0.12112430307330672 0.97600619725464122 0.08599884278042236
H  0.75016279902412597 -0.33132205318865016 -0.54481406902570462
"""
    assert text == correct_text


def test_scan_save_non_xyz(opt_output, tmp_path):
    scan_res = ScanData(
        provenance={"program": "qcdata-test-suite"},
        trajectory=[opt_output] * 3,
    )
    scan_res.save(tmp_path / "scan_res.json")
    scan_res_copy = ScanData.open(tmp_path / "scan_res.json")
    assert scan_res == scan_res_copy


def test_empty_and_partial_failed_trajectories(prog_input_factory, tmp_path):
    provenance = {"program": "producer"}
    empty = OptimizationData(provenance=provenance)
    assert empty.final_structure is None
    assert empty.final_energy is None
    assert empty.structures == []
    assert empty.energies.size == 0
    assert empty.to_xyz() == ""
    assert repr(empty)
    scan = ScanData(provenance=provenance)
    assert scan.structures == []
    assert scan.energies.size == 0
    assert scan.to_xyz() == ""

    step = ProgramOutput(
        input_data=prog_input_factory("gradient"),
        results=SinglePointData(provenance=provenance, energy=-1.0),
        success=False,
        traceback="Gradient unavailable",
    )
    partial = OptimizationData(provenance=provenance, trajectory=[step])
    assert partial.final_structure == step.input_data.structure
    assert partial.final_energy == -1.0
    points = [
        ProgramOutput(
            input_data=prog_input_factory("optimization"),
            results=data,
            success=False,
            traceback="Optimization failed",
        )
        for data in [empty, partial]
    ]
    scan = ScanData(provenance=provenance, trajectory=points)
    assert scan.structures == [None, step.input_data.structure]
    assert np.isnan(scan.energies[0])
    assert scan.energies[1] == -1.0
    with pytest.raises(ValueError, match="no final structure"):
        scan.to_xyz()
    path = tmp_path / "scan.json"
    scan.save(path)
    reopened = ScanData.open(path)
    assert reopened == scan
    assert reopened.trajectory[1].results.trajectory[0].results.energy == -1.0
