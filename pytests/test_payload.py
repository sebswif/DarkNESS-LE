import numpy as np
import pytest

pytest.importorskip("xraylib")
from functions.instrument import silicon_qe  # noqa: E402
from functions.payload import aperture_ccd, bare_ccd, lobster_ccd  # noqa: E402


def test_aperture_full_on_axis_and_zero_at_edge():
    p = aperture_ccd(area=10, fov=10, thickness=50)
    assert p["response"](4.0, 1.0) == pytest.approx(10 * silicon_qe(4.0, 50))
    assert p["response"](4.0, np.cos(np.deg2rad(10.001))) == 0


def test_aperture_plate_height():
    # Equal aperture and CCD radius b: h = 2b / tan(fov).
    p = aperture_ccd(area=np.pi, fov=45)
    assert p["plate_height_cm"] == pytest.approx(2.0)

OPTIC = dict(focal_length=60, aperture=9, pore=40, wall=8, length=2.4, slope_error=3.0)


def test_bare_response_is_area_qe_mu():
    p = bare_ccd(area=10, fov=5, thickness=50)
    assert p["response"](4.0, 0.5) == pytest.approx(10 * silicon_qe(4.0, 50) * 0.5)
    assert p["core"] is None and p["det_area"] == 10


def test_lobster_region_sets_core_and_particle_area():
    e = np.array([1.0, 3.0])
    p = lobster_ccd(e, OPTIC, extraction_mm=2.0)
    omega, area = p["core"]
    assert p["det_area"] == pytest.approx(0.04)          # (2 mm)^2 in cm^2
    assert omega == pytest.approx((0.2 / 60)**2)          # region / f^2
    assert np.all(p["response"](e, 1.0) < area)           # spread << focused spot


def test_lobster_enclosed_fraction_grows_with_region():
    e = np.array([1.0])
    small = lobster_ccd(e, OPTIC, extraction_mm=0.5)["core"][1]
    large = lobster_ccd(e, OPTIC, extraction_mm=5.0)["core"][1]
    assert large > small
