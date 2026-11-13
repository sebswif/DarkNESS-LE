import numpy as np
import pytest

from functions.sensitivity import limiting_coupling, poisson_upper_limit, signal_counts


def test_constant_rate_counts():
    # Flat rate 2 /cm^2/s/keV * area 5 cm^2 over 3 keV for 10 s.
    e = np.array([2.0, 5.0])
    assert signal_counts(e, 2.0, 1.0, 5.0, exposure=10) == pytest.approx(300)


def test_zero_counts_limit():
    # n = 0 gives -ln(1 - CL), independent of background.
    assert poisson_upper_limit(0, 5.0) == pytest.approx(-np.log(0.05))


def test_limit_grows_with_observed_counts():
    assert poisson_upper_limit(5, 1.0) > poisson_upper_limit(2, 1.0)


def test_coupling_scales_as_fourth_root():
    # 16x more allowed counts -> 2x the coupling.
    assert limiting_coupling(16, 1, g=1e-10) == pytest.approx(2e-10)


@pytest.mark.reference
@pytest.mark.skip(reason="needs a tabulated Helene (1983) upper limit for n > 0 with background")
def test_poisson_limit_matches_published():
    pass
