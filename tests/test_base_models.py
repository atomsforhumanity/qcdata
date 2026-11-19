def test_enums_render_as_their_values():
    """str(), format(), and f-strings give the value on every supported Python."""
    from qcdata import CalcType, LengthUnit

    assert str(CalcType.energy) == f"{CalcType.energy}" == "energy"
    assert format(LengthUnit.ANGSTROM) == LengthUnit.ANGSTROM.value
    assert CalcType.gradient == "gradient" and CalcType("hessian") is CalcType.hessian


def test_equality_compares_fields_without_serializing(monkeypatch):
    import numpy as np

    from qcdata import Provenance, SinglePointData

    def refuse(*args, **kwargs):
        raise AssertionError("equality must not serialize")

    hessian = np.random.default_rng(1).normal(size=(30, 30))
    a = SinglePointData(provenance=Provenance(program="x"), hessian=hessian)
    b = SinglePointData(provenance=Provenance(program="x"), hessian=hessian.copy())
    monkeypatch.setattr(SinglePointData, "model_dump", refuse)
    assert a == b
    nudged = hessian.copy()
    nudged[3, 4] = np.nextafter(nudged[3, 4], 1.0)
    assert a != SinglePointData(provenance=Provenance(program="x"), hessian=nudged)
    assert a != SinglePointData(provenance=Provenance(program="y"), hessian=hessian)
    assert a != SinglePointData(provenance=Provenance(program="x"))


def test_equality_inside_free_form_values_and_across_classes():
    import numpy as np

    from qcdata import Model, Provenance, SinglePointData

    def data(extras):
        return SinglePointData(provenance=Provenance(program="x"), extras=extras)

    assert data({"trace": np.arange(3.0)}) == data({"trace": np.arange(3.0)})
    assert data({"trace": np.arange(3.0)}) == data({"trace": [0.0, 1.0, 2.0]})
    assert data({"trace": np.arange(3.0)}) != data({"trace": np.arange(4.0)})
    assert data({"pair": (1, 2)}) == data({"pair": [1, 2]})
    assert data({"a": {"b": [1, {"c": 2}]}}) != data({"a": {"b": [1, {"c": 3}]}})
    assert data({}) != Model(method="x")
