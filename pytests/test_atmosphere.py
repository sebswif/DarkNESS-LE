import numpy as np
import pytest

pytest.importorskip("pymsis")
from functions.atmosphere import air_attenuation, air_density, photon_plasma_mass  # noqa: E402


def test_attenuation_formula():
    assert air_attenuation(1.0, 77.88) == pytest.approx(7.788)


def test_plasma_mass_scales_as_sqrt_density():
    assert photon_plasma_mass(4.0) == pytest.approx(2 * photon_plasma_mass(1.0))


def test_density_falls_with_altitude():
    # Assumed activity, so no data download.
    rho = air_density([100, 200, 400], 0, 0, "2020-01-01T00:00:00",
                      f107=150, f107a=150, ap=4)
    assert np.all(np.diff(rho) < 0)


@pytest.mark.reference
@pytest.mark.skip(reason="needs a published NRLMSIS 2.1 density for fixed inputs")
def test_density_matches_published():
    pass
