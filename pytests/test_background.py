import numpy as np
import pytest

# functions.background imports transport.py, which needs xraylib and pymsis.
pytest.importorskip("xraylib")
pytest.importorskip("pymsis")
from functions.background import (  # noqa: E402
    background_budget, cxb, dark_earth, nxb, particle_background, solar_spectrum,
)


def test_solar_energy_flux_matches_goes():
    # Thermal energy flux over GOES 1-8 Angstrom (1.55-12.4 keV) must return goes.
    e = np.linspace(1.55, 12.4, 20001)
    keV_cm2_s = np.trapezoid(e * solar_spectrum(e, goes=1e-6, temperature=10), e)
    w_m2 = keV_cm2_s * 1.602176634e-16 * 1e4
    assert w_m2 == pytest.approx(1e-6, rel=1e-6)


def test_cxb_positive_and_falling():
    sky = cxb([4, 10, 30])
    assert np.all(sky > 0) and np.all(np.diff(sky) < 0)


def test_cxb_rejects_below_1kev():
    with pytest.raises(ValueError):
        cxb(0.5)


def test_cxb_continuous_at_3kev():
    assert cxb(3 - 1e-9) == pytest.approx(cxb(3.0), rel=1e-6)


def test_cxb_soft_slope_is_photon_index_141():
    assert cxb(1.0) / cxb(2.0) == pytest.approx(2**1.41)


def test_dark_earth_half_at_break():
    assert dark_earth(33.7, c=1.0) == pytest.approx(0.5)


def test_nxb_line_integrates_to_rate():
    e = np.linspace(5, 8, 3001)
    line = nxb(e, 0, [(6.4, 0.1, 1e-3)])
    assert np.trapezoid(line, e) == pytest.approx(1e-3, rel=1e-4)


def test_particle_background_weighted_mean():
    result = particle_background([[1, 1], [3, 3]], [1, 1])
    assert result == pytest.approx([2, 2])


def test_budget_total_is_sum():
    budget = background_budget([1, 2], 10, {"a": 1.0, "b": 2.0})
    assert budget["total"] == pytest.approx(budget["a"] + budget["b"])
    assert budget["total"] == pytest.approx(30)


@pytest.mark.reference
@pytest.mark.skip(reason="needs published CXB intensity values (Gruber et al. 1999)")
def test_cxb_matches_published():
    pass
