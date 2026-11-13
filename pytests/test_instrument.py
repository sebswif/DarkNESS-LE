import numpy as np
import pytest

from functions.instrument import bare_effective_area, lobster_effective_area, silicon_qe


def test_qe_between_zero_and_one():
    qe = silicon_qe(np.linspace(2, 50, 20))
    assert np.all((qe >= 0) & (qe <= 1))


def test_qe_zero_thickness_is_zero():
    assert silicon_qe(10.0, thickness=0) == 0


def test_qe_grows_with_thickness():
    assert silicon_qe(10.0, thickness=500) > silicon_qe(10.0, thickness=250)


def test_qe_rejects_energy_outside_table():
    with pytest.raises(ValueError):
        silicon_qe(1.0)


def test_bare_area_is_area_times_qe():
    assert bare_effective_area(10.0, area=100) == pytest.approx(100 * silicon_qe(10.0))


def test_lobster_area_formula():
    bare = bare_effective_area(10.0, area=100)
    lobster = lobster_effective_area(10.0, area=100, reflectivity=0.8,
                                     open_fraction=0.6, focus_fraction=0.25)
    assert lobster == pytest.approx(bare * 0.6 * 0.25 * 0.8**2)


@pytest.mark.reference
@pytest.mark.skip(reason="needs published Si QE value (e.g. NIST XCOM at 10 keV, 250 um)")
def test_qe_matches_published():
    pass
