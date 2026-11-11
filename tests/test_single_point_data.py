import numpy as np

from qcdata import SinglePointData, Wavefunction


def test_gradient_converted_np_array():
    """Test that SinglePointData converts gradient to np array"""
    gradient = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    data = SinglePointData(
        provenance={"program": "qcdata-test-suite"}, gradient=gradient
    )
    assert isinstance(data.gradient, np.ndarray)
    assert data.gradient.dtype == np.float64


def test_hessian_converted_np_array():
    """Test that SinglePointData converts hessian to np array"""
    hessian = [float(i) for i in range(9)]
    data = SinglePointData(provenance={"program": "qcdata-test-suite"}, hessian=hessian)
    assert isinstance(data.hessian, np.ndarray)
    assert data.hessian.dtype == np.float64


def test_single_point_casts_gradient_to_n_by_3(prog_input_factory):
    """Test that SinglePointData casts gradient to n by 3"""
    pi_gradient = prog_input_factory("gradient")
    n_atoms = len(pi_gradient.structure.symbols)
    gradient = [float(i) for i in range(n_atoms * 3)]
    data = SinglePointData(
        provenance={"program": "qcdata-test-suite"}, gradient=gradient
    )
    assert data.gradient.shape == (n_atoms, 3)
    assert data.gradient.dtype == np.float64


def test_single_point_success_casts_hessian_to_3n_by_3n(prog_input_factory):
    """Test that SinglePointData casts hessian to 3n x 3n"""
    pi_hessian = prog_input_factory("hessian")
    n_atoms = len(pi_hessian.structure.symbols)
    hessian = [float(i) for i in range(n_atoms**2 * 3**2)]
    data = SinglePointData(provenance={"program": "qcdata-test-suite"}, hessian=hessian)
    assert data.hessian.shape == (n_atoms * 3, n_atoms * 3)
    assert data.hessian.dtype == np.float64


def test_wavefunction_to_numpy():
    """Test that wavefunction converts to numpy array"""
    wavefunction = Wavefunction(
        scf_eigenvalues_a=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        scf_eigenvalues_b=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        scf_occupations_a=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        scf_occupations_b=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    )
    assert isinstance(wavefunction.scf_eigenvalues_a, np.ndarray)
    assert isinstance(wavefunction.scf_eigenvalues_b, np.ndarray)
    assert isinstance(wavefunction.scf_occupations_a, np.ndarray)
    assert isinstance(wavefunction.scf_occupations_b, np.ndarray)
    assert wavefunction.scf_eigenvalues_a.dtype == np.float64
    assert wavefunction.scf_eigenvalues_b.dtype == np.float64
    assert wavefunction.scf_occupations_a.dtype == np.float64
    assert wavefunction.scf_occupations_b.dtype == np.float64


def test_single_point_data_can_be_empty_for_partial_or_failed_outputs():
    data = SinglePointData(provenance={"program": "qcdata-test-suite"})
    assert data.energy is None
    assert data.gradient is None
    assert data.hessian is None


def test_atomic_charges_and_point_charge_gradient_shapes():
    """Charges are a flat per-atom vector; point-charge gradients are n x 3."""
    data = SinglePointData(
        provenance={"program": "qcdata-test-suite"},
        atomic_charges=[[-0.8], [0.4], [0.4]],
        point_charge_gradient=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
    )
    assert data.atomic_charges.shape == (3,)
    assert data.atomic_charges.dtype == np.float64
    assert data.point_charge_gradient.shape == (2, 3)
    restored = SinglePointData.model_validate_json(data.model_dump_json())
    np.testing.assert_allclose(restored.atomic_charges, [-0.8, 0.4, 0.4])
    np.testing.assert_allclose(
        restored.point_charge_gradient, [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    )
    assert SinglePointData(provenance={"program": "x"}).atomic_charges is None
