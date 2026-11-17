"""Wavefunctions: orbitals, basis sets, AO matrices, populations and their conventions.

The AO overlap is rebuilt here independently of qcdata's convention code, from the shells
alone: Obara-Saika over Cartesian Gaussians, with each convention's functions written
out as explicit polynomials and norms. Conversions are checked against it.
"""

# l is the conventional symbol for angular momentum.
# ruff: noqa: E741
import itertools
import math

import numpy as np
import pytest
from pydantic import ValidationError

from qcdata import (
    DEFAULT_AO_CONVENTION,
    MOLDEN_CONVENTION,
    AOConvention,
    BasisSet,
    CartesianNormalization,
    CartesianOrder,
    Ecp,
    EcpTerm,
    OrbitalKind,
    Orbitals,
    Shell,
    SphericalOrder,
    Wavefunction,
)
from qcdata.utils import water

# ------------------------------------------------------- an independent overlap

# Cartesian components by label; lexicographic and Molden orders, written out.
LEXICOGRAPHIC = {
    0: [""],
    1: ["x", "y", "z"],
    2: "xx xy xz yy yz zz".split(),
    3: "xxx xxy xxz xyy xyz xzz yyy yyz yzz zzz".split(),
}
MOLDEN_CARTESIAN = {
    2: "xx yy zz xy xz yz".split(),
    3: "xxx yyy zzz xyy xxy xxz xzz yzz yyz xyz".split(),
}
# Real solid harmonics without the Condon-Shortley phase, as unnormalized polynomials.
SOLID_HARMONICS = {
    1: {-1: {"y": 1}, 0: {"z": 1}, 1: {"x": 1}},
    2: {
        -2: {"xy": 1},
        -1: {"yz": 1},
        0: {"zz": 2, "xx": -1, "yy": -1},
        1: {"xz": 1},
        2: {"xx": 1, "yy": -1},
    },
    3: {
        -3: {"xxy": 3, "yyy": -1},
        -2: {"xyz": 1},
        -1: {"yzz": 4, "xxy": -1, "yyy": -1},
        0: {"zzz": 2, "xxz": -3, "yyz": -3},
        1: {"xzz": 4, "xxx": -1, "xyy": -1},
        2: {"xxz": 1, "yyz": -1},
        3: {"xxx": 1, "xyy": -3},
    },
}
SPHERICAL_M = {
    "pyscf": {1: [1, -1, 0], 2: [-2, -1, 0, 1, 2], 3: [-3, -2, -1, 0, 1, 2, 3]},
    "increasing_m": {1: [-1, 0, 1], 2: [-2, -1, 0, 1, 2], 3: [-3, -2, -1, 0, 1, 2, 3]},
    "molden": {1: [1, -1, 0], 2: [0, 1, -1, 2, -2], 3: [0, 1, -1, 2, -2, 3, -3]},
    "trexio": {1: [0, 1, -1], 2: [0, 1, -1, 2, -2], 3: [0, 1, -1, 2, -2, 3, -3]},
}


def _powers(label):
    return (label.count("x"), label.count("y"), label.count("z"))


def _dfact(n):
    return math.prod(range(n, 0, -2)) if n > 0 else 1


def _ov1d(i, j, a, b, xa, xb):
    """1-D overlap of x^i exp(-a x^2) at xa and x^j exp(-b x^2) at xb (Obara-Saika)."""
    p = a + b
    xp = (a * xa + b * xb) / p
    s = np.zeros((i + 1, j + 1))
    s[0, 0] = math.sqrt(math.pi / p) * math.exp(-a * b / p * (xa - xb) ** 2)
    for ii in range(i + 1):
        for jj in range(j + 1):
            if ii == jj == 0:
                continue
            if ii > 0:
                v = (xp - xa) * s[ii - 1, jj]
                if ii > 1:
                    v += (ii - 1) / (2 * p) * s[ii - 2, jj]
                if jj > 0:
                    v += jj / (2 * p) * s[ii - 1, jj - 1]
            else:
                v = (xp - xb) * s[ii, jj - 1]
                if jj > 1:
                    v += (jj - 1) / (2 * p) * s[ii, jj - 2]
            s[ii, jj] = v
    return s[i, j]


def _raw_overlap(functions):
    """Overlap of contracted Cartesian Gaussians (center, powers, exponents, coefs)."""
    n = len(functions)
    s = np.zeros((n, n))
    for p in range(n):
        ra, pa, ea, ca = functions[p]
        for q in range(p, n):
            rb, pb, eb, cb = functions[q]
            v = 0.0
            for a, xa in zip(ea, ca, strict=True):
                for b, xb in zip(eb, cb, strict=True):
                    v += (
                        xa
                        * xb
                        * math.prod(
                            _ov1d(pa[k], pb[k], a, b, ra[k], rb[k]) for k in range(3)
                        )
                    )
            s[p, q] = s[q, p] = v
    return s


def _target_norm2(l, label, normalization):
    """Squared norm of a Cartesian component; None: the function as stored (axis)."""
    if normalization == "unit":
        return 1.0
    if normalization == "pyscf" and l >= 2:
        # PySCF: radially normalized, so the angular integral of the monomial remains.
        lx, ly, lz = _powers(label)
        return (
            4
            * math.pi
            * _dfact(2 * lx - 1)
            * _dfact(2 * ly - 1)
            * _dfact(2 * lz - 1)
            / _dfact(2 * l + 1)
        )
    return None


def independent_overlap(basis, positions, convention):
    """The AO overlap of `basis` in `convention`, from its shells alone."""
    raw, rows = [], []  # rows: ({raw index: weight}, target squared norm or None)
    for sh in basis.shells:
        l = sh.angular_momentum
        index = {}
        for label in LEXICOGRAPHIC[l]:
            index[label] = len(raw)
            raw.append(
                (positions[sh.center], _powers(label), sh.exponents, sh.coefficients)
            )
        if l == 0:
            rows.append(({index[""]: 1.0}, 1.0))
        elif sh.pure:
            for m in SPHERICAL_M[convention.spherical_order][l]:
                weights = SOLID_HARMONICS[l][m]
                rows.append(({index[k]: w for k, w in weights.items()}, 1.0))
        else:
            order = (
                MOLDEN_CARTESIAN[l]
                if convention.cartesian_order == "molden" and l >= 2
                else LEXICOGRAPHIC[l]
            )
            for label in order:
                target = _target_norm2(l, label, convention.cartesian_normalization)
                rows.append(({index[label]: 1.0}, target))
    t = np.zeros((len(rows), len(raw)))
    for i, (weights, _) in enumerate(rows):
        for k, w in weights.items():
            t[i, k] = w
    s = t @ _raw_overlap(raw) @ t.T
    scale = np.array(
        [
            1.0 if target is None else math.sqrt(target / s[i, i])
            for i, (_, target) in enumerate(rows)
        ]
    )
    return s * np.outer(scale, scale)


def _normalized_shell(center, l, pure, exponents, raw):
    """A shell whose x^l component has unit norm, normalized numerically."""
    exps, raw = np.array(exponents), np.array(raw)
    xl = ((0.0, 0.0, 0.0), (l, 0, 0), exps, raw)
    norm2 = _raw_overlap([xl])[0, 0]
    return Shell(
        center=center,
        angular_momentum=l,
        pure=pure,
        exponents=exps,
        coefficients=raw / math.sqrt(norm2),
    )


def make_basis(core_electrons=(0, 0, 0)):
    """Water: Cartesian and spherical d and f shells, contracted and not."""
    spec = [
        (0, 0, True, [5.0, 1.2], [0.4, 0.7]),
        (0, 1, True, [1.1], [1.0]),
        (0, 2, False, [0.9, 0.3], [0.6, 0.5]),
        (0, 3, False, [0.7], [1.0]),
        (0, 2, True, [0.5], [1.0]),
        (0, 3, True, [0.4, 1.3], [0.8, 0.3]),
        (1, 0, True, [0.6], [1.0]),
        (1, 1, False, [0.8], [1.0]),
        (1, 2, False, [0.45], [1.0]),
        (2, 0, True, [0.6], [1.0]),
        (2, 2, True, [0.55], [1.0]),
    ]
    return BasisSet(
        name="test",
        shells=[_normalized_shell(*s) for s in spec],
        core_electrons=list(core_electrons),
    )


BASIS = make_basis()
S0 = independent_overlap(BASIS, water.geometry, DEFAULT_AO_CONVENTION)
NAO = BASIS.nao
ALL_CONVENTIONS = [
    AOConvention(spherical_order=s, cartesian_order=c, cartesian_normalization=n)
    for s, c, n in itertools.product(
        SphericalOrder, CartesianOrder, CartesianNormalization
    )
]


def occupations(n_alpha, n_beta, nmo=NAO):
    return (
        np.array([1.0] * n_alpha + [0.0] * (nmo - n_alpha)),
        np.array([1.0] * n_beta + [0.0] * (nmo - n_beta)),
    )


def orthonormal_orbitals(s, seed):
    """C with C^T S C = 1: Lowdin orbitals rotated at random."""
    w, v = np.linalg.eigh(s)
    q = np.linalg.qr(np.random.default_rng(seed).normal(size=s.shape))[0]
    return v @ np.diag(w**-0.5) @ v.T @ q


def make_wavefunction(spin="restricted", basis=BASIS, s=S0):
    """A canonical wavefunction with H C = S C e and a core Hamiltonian."""
    nao = basis.nao
    occ_a, occ_b = occupations(5, 5) if spin == "restricted" else occupations(6, 4)
    energies = np.linspace(-20.0, 3.0, nao)
    rng = np.random.default_rng(7)
    h = rng.normal(size=(nao, nao))
    fields = {}
    for x, seed in (("a", 1), ("b", 2)) if spin == "unrestricted" else (("a", 1),):
        c = orthonormal_orbitals(s, seed)
        fields[f"coefficients_{x}"] = c
        fields[f"energies_{x}"] = energies
        fields[f"hamiltonian_{x}"] = s @ c @ np.diag(energies) @ c.T @ s
    orbitals = Orbitals(
        spin=spin,
        occupations_a=occ_a,
        occupations_b=occ_b,
        basis=basis,
        **{k: v for k, v in fields.items() if not k.startswith("hamiltonian")},
    )
    return Wavefunction(
        orbitals=orbitals,
        overlap=s,
        core_hamiltonian=h + h.T,
        **{k: v for k, v in fields.items() if k.startswith("hamiltonian")},
    )


def test_independent_overlap_agrees_with_the_shells():
    # Every function has its convention's norm, and the basis is well conditioned.
    assert np.allclose(np.diag(S0)[:4], 1.0)  # s and p
    assert np.linalg.eigvalsh(S0).min() > 1e-3
    unit = independent_overlap(
        BASIS, water.geometry, AOConvention(cartesian_normalization="unit")
    )
    assert np.allclose(np.diag(unit), 1.0)
    labels = BASIS.ao_labels()
    assert S0[labels.index("0 3dxy"), labels.index("0 3dxy")] == pytest.approx(1 / 3)
    assert S0[labels.index("0 4fxyz"), labels.index("0 4fxyz")] == pytest.approx(1 / 15)


@pytest.mark.parametrize("spin", ["restricted", "unrestricted"])
def test_every_conversion_keeps_the_wavefunction_consistent(spin):
    """S matches an independent build; C^T S C = 1, H C = S C e and MO matrices hold."""
    wf = make_wavefunction(spin)
    spins = ("a", "b") if spin == "unrestricted" else ("a",)
    mo_core = {
        x: getattr(wf.orbitals, f"coefficients_{x}").T
        @ wf.core_hamiltonian
        @ getattr(wf.orbitals, f"coefficients_{x}")
        for x in spins
    }
    p0 = sum(wf.orbitals.density())
    for conv in ALL_CONVENTIONS:
        there = wf.with_convention(conv)
        assert there.orbitals.basis.ao_convention == conv
        s = there.overlap
        np.testing.assert_allclose(
            s, independent_overlap(BASIS, water.geometry, conv), atol=1e-12
        )
        for x in spins:
            c = getattr(there.orbitals, f"coefficients_{x}")
            e = getattr(there.orbitals, f"energies_{x}")
            np.testing.assert_allclose(c.T @ s @ c, np.eye(NAO), atol=1e-11)
            h = getattr(there, f"hamiltonian_{x}")
            np.testing.assert_allclose(h @ c, s @ c @ np.diag(e), atol=1e-10)
            np.testing.assert_allclose(
                c.T @ there.core_hamiltonian @ c, mo_core[x], atol=1e-10
            )
        # Densities convert with the opposite variance of operators.
        p = sum(there.orbitals.density())
        np.testing.assert_allclose(
            p, BASIS.transform(p0, conv, kind="density"), atol=1e-11
        )
        assert np.trace(p @ s) == pytest.approx(10.0, abs=1e-10)
        back = there.with_convention(DEFAULT_AO_CONVENTION)
        for name in ("overlap", "core_hamiltonian", "hamiltonian_a"):
            np.testing.assert_allclose(
                getattr(back, name), getattr(wf, name), atol=1e-12
            )
        np.testing.assert_allclose(
            back.orbitals.coefficients_a, wf.orbitals.coefficients_a, atol=1e-12
        )


def test_conversion_does_not_copy_through_lists(monkeypatch):
    wf = make_wavefunction()

    def no_dump(*args, **kwargs):
        raise AssertionError("converted through model_dump")

    monkeypatch.setattr(Orbitals, "model_dump", no_dump)
    monkeypatch.setattr(Wavefunction, "model_dump", no_dump)
    there = wf.with_convention(MOLDEN_CONVENTION)
    # Fields that do not change are shared, not copied.
    assert there.orbitals.energies_a is wf.orbitals.energies_a
    assert there.orbitals.occupations_b is wf.orbitals.occupations_b
    assert there.orbitals.basis.shells is wf.orbitals.basis.shells


def test_transform_variances():
    labels = BASIS.ao_labels()
    xy, xx = labels.index("0 3dxy"), labels.index("0 3dxx")
    unit = AOConvention(cartesian_normalization=CartesianNormalization.unit)
    eye = np.eye(NAO)
    # Coefficients of the x^l-normalized xy (norm 1/sqrt(3)) shrink by that norm.
    c = BASIS.transform(eye, unit, kind="coefficients")
    assert c[xy, xy] == pytest.approx(1 / math.sqrt(3))
    assert c[xx, xx] == pytest.approx(1.0)
    assert BASIS.transform(eye, unit, kind="operator")[xy, xy] == pytest.approx(3.0)
    assert BASIS.transform(eye, unit, kind="density")[xy, xy] == pytest.approx(1 / 3)
    # p shells: PySCF (x, y, z) -> increasing m (y, z, x): a permutation for all kinds.
    cca = AOConvention(spherical_order="increasing_m")
    m = np.arange(NAO * NAO, dtype=float).reshape(NAO, NAO)
    yzx = np.ix_([2, 3, 1], [2, 3, 1])
    for kind in ("operator", "density"):
        np.testing.assert_array_equal(
            BASIS.transform(m, cca, kind=kind)[1:4, 1:4], m[yzx]
        )
    np.testing.assert_array_equal(
        BASIS.transform(m, cca, kind="coefficients")[1:4], m[[2, 3, 1]]
    )
    with pytest.raises(ValueError, match="kind"):
        BASIS.transform(eye, unit, kind="matrix")
    with pytest.raises(ValueError, match="matrix"):
        BASIS.transform(eye[:, :3], unit, kind="operator")
    with pytest.raises(ValueError, match="rows"):
        BASIS.transform(eye[:3], unit, kind="coefficients")
    with pytest.raises(TypeError):
        BASIS.transform(eye, unit)  # the variance is never implied


# ---------------------------------------------------------------- basis sets


def test_basis_counts_and_labels():
    assert NAO == (1 + 3 + 6 + 10 + 5 + 7) + (1 + 3 + 6) + (1 + 5)
    assert BASIS.shell_offsets()[:4] == [0, 1, 4, 10]
    labels = BASIS.ao_labels(water.symbols)
    assert labels[0] == "0 O 1s" and labels[1] == "0 O 2p x"
    assert labels[-1] == "2 H 3d+2"
    assert "1 H 3dxy" in labels


def test_basis_core_electrons_and_ecp():
    shells = BASIS.shells
    with pytest.raises(ValidationError, match="core_electrons"):
        BasisSet(shells=shells)
    with pytest.raises(ValidationError, match="one value per atom"):
        BasisSet(shells=shells, core_electrons=[0, 0])
    with pytest.raises(ValidationError, match=">= 0"):
        BasisSet(shells=shells, core_electrons=[-2, 0, 0])
    term = EcpTerm(power=-2, exponent=1.5, coefficient=3.0)
    ecp = Ecp(center=0, local=[term], semilocal=[[term], [term, term]])
    basis = BasisSet(shells=shells, core_electrons=[2, 0, 0], ecp=[ecp])
    assert basis.ecp[0].semilocal[1][1].power == -2
    loaded = BasisSet.model_validate_json(basis.model_dump_json())
    assert loaded == basis
    with pytest.raises(ValidationError, match="at most one Ecp"):
        BasisSet(shells=shells, core_electrons=[2, 0, 0], ecp=[ecp, ecp])
    with pytest.raises(ValidationError, match="Ecp center"):
        BasisSet(
            shells=shells,
            core_electrons=[2, 0, 0],
            ecp=[Ecp(center=3, local=[term], semilocal=[])],
        )
    with pytest.raises(ValidationError, match="power"):
        EcpTerm(power=-3, exponent=1.0, coefficient=1.0)
    with pytest.raises(ValidationError, match="exponent"):
        EcpTerm(power=0, exponent=0.0, coefficient=1.0)
    with pytest.raises(ValidationError, match="spin_orbit"):
        Ecp.model_validate(
            {"center": 0, "local": [], "semilocal": [], "spin_orbit": []}
        )


def test_ao_convention_default_is_a_fresh_default():
    a = BasisSet(shells=BASIS.shells, core_electrons=[0, 0, 0])
    b = BasisSet(shells=BASIS.shells, core_electrons=[0, 0, 0])
    assert a.ao_convention == b.ao_convention == DEFAULT_AO_CONVENTION
    assert BasisSet.model_fields["ao_convention"].default_factory is AOConvention


def test_shell_validation():
    with pytest.raises(ValidationError, match="same length"):
        Shell(
            center=0,
            angular_momentum=0,
            pure=True,
            exponents=[1.0, 2.0],
            coefficients=[1.0],
        )
    with pytest.raises(ValidationError, match="positive"):
        Shell(
            center=0,
            angular_momentum=0,
            pure=True,
            exponents=[-1.0],
            coefficients=[1.0],
        )


# ------------------------------------------------------------------ orbitals


def levels(**kw):
    occ_a, occ_b = occupations(5, 5, nmo=8)
    values = dict(
        spin="restricted",
        energies_a=np.linspace(-1.0, 1.0, 8),
        occupations_a=occ_a,
        occupations_b=occ_b,
    )
    values.update(kw)
    return Orbitals(**values)


def test_levels_without_coefficients():
    orb = levels()
    assert orb.kind is OrbitalKind.canonical and orb.restricted and orb.nmo == 8
    assert orb.basis is None and orb.coefficients_a is None
    for call in (orb.density, lambda: orb.with_convention(MOLDEN_CONVENTION)):
        with pytest.raises(ValueError, match="coefficients"):
            call()
    with pytest.raises(ValueError, match="coefficients"):
        Wavefunction(orbitals=orb).with_convention(MOLDEN_CONVENTION)
    assert Orbitals.model_validate_json(orb.model_dump_json()) == orb


def test_occupations_are_per_spin_and_bounded():
    with pytest.raises(ValidationError, match="occupations_b"):
        Orbitals(spin="restricted", energies_a=[0.0], occupations_a=[1.0])
    with pytest.raises(ValidationError, match="one value per orbital"):
        levels(occupations_b=[1.0, 0.0])
    for bad in (2.0, -0.1, np.nan):
        with pytest.raises(ValidationError, match="between 0 and 1"):
            levels(occupations_a=[bad] + [0.0] * 7)
    with pytest.raises(ValidationError, match="one value per orbital"):
        levels(energies_a=[1.0])
    # A restricted open shell: one set of orbitals, occupied per spin.
    occ_a, occ_b = occupations(5, 3, nmo=8)
    orb = levels(occupations_a=occ_a, occupations_b=occ_b)
    assert orb.occupations_a.sum() - orb.occupations_b.sum() == 2


def test_kinds():
    assert levels(kind="canonical").kind is OrbitalKind.canonical
    with pytest.raises(ValidationError, match="need their energies"):
        levels(energies_a=None)
    natural = levels(kind="natural", energies_a=None)
    assert natural.kind is OrbitalKind.natural
    with pytest.raises(ValidationError, match="natural orbitals have no energies"):
        levels(kind="natural")
    loc = levels(kind="localized", energies_a=None, localization="pipek_mezey")
    assert loc.localization == "pipek_mezey"
    with pytest.raises(ValidationError, match="localization"):
        levels(kind="localized", energies_a=None)
    with pytest.raises(ValidationError, match="localization"):
        levels(localization="boys")
    with pytest.raises(ValidationError):
        levels(kind="bogus")
    with pytest.raises(ValidationError, match="energies_b"):
        levels(energies_b=np.zeros(8))
    with pytest.raises(ValidationError, match="energies_b"):
        levels(spin="unrestricted")


def test_electronic_temperature_and_fermi_levels():
    assert levels().electronic_temperature == 0.0
    smeared = levels(
        electronic_temperature=300.0, fermi_level_a=-0.2, fermi_level_b=-0.25
    )
    assert smeared.fermi_level_b == -0.25
    with pytest.raises(ValidationError, match="at 0 K they are None"):
        levels(fermi_level_a=-0.2, fermi_level_b=-0.2)
    with pytest.raises(ValidationError, match="need fermi_level_a and fermi_level_b"):
        levels(electronic_temperature=300.0, fermi_level_a=-0.2)
    with pytest.raises(ValidationError, match=">= 0 K"):
        levels(electronic_temperature=-1.0)


def test_coefficients_come_with_their_basis():
    occ_a, occ_b = occupations(5, 5)
    c = np.eye(NAO)
    e = np.zeros(NAO)
    base = dict(
        spin="restricted", energies_a=e, occupations_a=occ_a, occupations_b=occ_b
    )
    with pytest.raises(ValidationError, match="both or neither"):
        Orbitals(**base, coefficients_a=c)
    with pytest.raises(ValidationError, match="both or neither"):
        Orbitals(**base, basis=BASIS)
    with pytest.raises(ValidationError, match="nao x nmo"):
        Orbitals(**base, basis=BASIS, coefficients_a=c[:, :3])
    with pytest.raises(ValidationError, match="no coefficients_b"):
        Orbitals(**base, basis=BASIS, coefficients_a=c, coefficients_b=c)
    unrestricted = dict(base, spin="unrestricted", energies_b=e)
    with pytest.raises(ValidationError, match="need coefficients_b"):
        Orbitals(**unrestricted, basis=BASIS, coefficients_a=c)
    with pytest.raises(ValidationError, match="coefficients_b must be"):
        Orbitals(**unrestricted, basis=BASIS, coefficients_a=c, coefficients_b=c[:-1])
    with pytest.raises(ValidationError, match="coefficients_b needs"):
        Orbitals(**unrestricted, coefficients_b=c)


def test_density_is_c_n_ct_per_spin():
    occ_a, occ_b = occupations(6, 4)
    c = orthonormal_orbitals(S0, 3)
    orb = Orbitals(
        spin="restricted",
        energies_a=np.zeros(NAO),
        occupations_a=occ_a,
        occupations_b=occ_b,
        basis=BASIS,
        coefficients_a=c,
    )
    pa, pb = orb.density()
    np.testing.assert_allclose(pa, c[:, :6] @ c[:, :6].T, atol=1e-14)
    np.testing.assert_allclose(pb, c[:, :4] @ c[:, :4].T, atol=1e-14)
    assert np.trace(pa @ S0) == pytest.approx(6.0)


# ------------------------------------------------------------- wavefunctions


def test_wavefunction_needs_orbitals_and_round_trips():
    with pytest.raises(ValidationError, match="orbitals"):
        Wavefunction()
    for spin in ("restricted", "unrestricted"):
        wf = make_wavefunction(spin)
        loaded = Wavefunction.model_validate_json(wf.model_dump_json())
        assert loaded == wf
        assert np.array_equal(loaded.hamiltonian_a, wf.hamiltonian_a)


def test_ao_matrices_need_the_basis_and_their_shape():
    wf = make_wavefunction()
    no_basis = levels()
    for name in ("overlap", "core_hamiltonian"):
        with pytest.raises(ValidationError, match="basis set"):
            Wavefunction(orbitals=no_basis, **{name: np.eye(8)})
        with pytest.raises(ValidationError, match="nao x nao"):
            Wavefunction(orbitals=wf.orbitals, **{name: np.eye(NAO - 1)})
    with pytest.raises(ValidationError, match="overlap"):
        Wavefunction(orbitals=wf.orbitals, hamiltonian_a=wf.hamiltonian_a)
    with pytest.raises(ValidationError, match="no hamiltonian_b"):
        Wavefunction(
            orbitals=wf.orbitals,
            overlap=S0,
            hamiltonian_a=wf.hamiltonian_a,
            hamiltonian_b=wf.hamiltonian_a,
        )
    with pytest.raises(ValidationError, match="hamiltonian_b needs hamiltonian_a"):
        Wavefunction(orbitals=wf.orbitals, overlap=S0, hamiltonian_b=wf.hamiltonian_a)
    uwf = make_wavefunction("unrestricted")
    with pytest.raises(ValidationError, match="need hamiltonian_b"):
        Wavefunction(orbitals=uwf.orbitals, overlap=S0, hamiltonian_a=uwf.hamiltonian_a)
    natural = wf.orbitals.model_copy(update={"kind": "natural", "energies_a": None})
    natural = Orbitals.model_validate(dict(natural))
    with pytest.raises(ValidationError, match="canonical"):
        Wavefunction(orbitals=natural, overlap=S0, hamiltonian_a=wf.hamiltonian_a)
    # The overlap and core Hamiltonian alone are fine.
    assert Wavefunction(orbitals=wf.orbitals, overlap=S0).hamiltonian_a is None


def test_old_wavefunction_fields_are_gone():
    for name in ("scf_eigenvalues_a", "scf_occupations_a"):
        assert name not in Wavefunction.model_fields
    with pytest.raises(ValidationError, match="scf_eigenvalues_a"):
        Wavefunction.model_validate(
            {"orbitals": levels().model_dump(), "scf_eigenvalues_a": [1.0]}
        )
