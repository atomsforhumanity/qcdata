def test_enums_render_as_their_values():
    """str(), format(), and f-strings give the value on every supported Python."""
    from qcdata import CalcType, LengthUnit

    assert str(CalcType.energy) == f"{CalcType.energy}" == "energy"
    assert format(LengthUnit.ANGSTROM) == LengthUnit.ANGSTROM.value
    assert CalcType.gradient == "gradient" and CalcType("hessian") is CalcType.hessian
