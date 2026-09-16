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


def test_single_point_data_normal_modes_cartesian_shape(prog_input_factory):
    """Test that SinglePointData normal_modes_cartesian are n_modes x n_atoms x 3"""
    pi_energy = prog_input_factory("energy")
    n_atoms = len(pi_energy.structure.symbols)
    n_atoms * 3
    data = SinglePointData(
        provenance={"program": "qcdata-test-suite"},
        energy=-1.0,
        freqs_wavenumber=[1.0, 2.0, 3.0],
        normal_modes_cartesian=np.array(
            [
                -1.45413605e-07,
                7.31568094e-02,
                3.49777695e-34,
                -4.00280118e-01,
                -5.80603129e-01,
                8.72597124e-50,
                4.00282426e-01,
                -5.80601321e-01,
                -2.61779137e-49,
                -8.40792347e-07,
                4.64736028e-02,
                -7.85031947e-33,
                6.02403552e-01,
                -3.68838662e-01,
                -1.08372224e-49,
                -6.02390207e-01,
                -3.68828204e-01,
                3.25116671e-49,
                -6.99575075e-02,
                -7.31659905e-07,
                9.73449382e-38,
                5.55204565e-01,
                -4.35072838e-01,
                1.17356863e-54,
                5.55217962e-01,
                4.35084451e-01,
                -3.52070589e-54,
            ]
        ),
    )
    assert data.normal_modes_cartesian.shape == (3, 3, 3)


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
