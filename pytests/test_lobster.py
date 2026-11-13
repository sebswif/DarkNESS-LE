import numpy as np
import pytest

pytest.importorskip("xraylib")
from functions.lobster import fresnel_reflectance, lobster_eye, mirror_reflectivity  # noqa: E402

GEOMETRY = dict(focal_length=30, aperture=9, pore=40, wall=8, length=2.4)


def test_total_reflection_below_critical_angle():
    # No absorption: reflectance is 1 below theta_c = sqrt(2*delta).
    assert fresnel_reflectance(0.5 * np.sqrt(2e-4), 1e-4, 0.0) == pytest.approx(1.0)


def test_reflectance_at_twice_critical_angle():
    # No absorption, theta = 2*theta_c: r = (2 - sqrt3) / (2 + sqrt3) exactly.
    theta = 1e-2
    delta = 1 - np.sqrt(1 - np.sin(theta)**2 / 4)  # exact: 2*delta - delta^2 = sin^2/4
    expected = ((2 - np.sqrt(3)) / (2 + np.sqrt(3)))**2
    assert fresnel_reflectance(theta, delta, 0.0) == pytest.approx(expected, rel=1e-6)


def test_roughness_lowers_reflectance():
    smooth = mirror_reflectivity(1.0, 0.01, "Ir")
    rough = mirror_reflectivity(1.0, 0.01, "Ir", roughness=1.0)
    assert rough < smooth


def test_perfect_reflector_conserves_photons():
    result = lobster_eye([1.0], material=None, **GEOMETRY)
    total = result["area_focus"] + result["area_arms"] + result["area_unfocused"]
    assert total == pytest.approx(result["open_fraction"] * result["geometric_area"], rel=1e-6)


def test_axes_separate_for_rectangle():
    square_a = lobster_eye([1.0], **{**GEOMETRY, "aperture": 6})["area_focus"]
    square_b = lobster_eye([1.0], **{**GEOMETRY, "aperture": 9})["area_focus"]
    rect = lobster_eye([1.0], **{**GEOMETRY, "aperture": (6, 9)})["area_focus"]
    assert rect == pytest.approx(np.sqrt(square_a * square_b), rel=1e-9)


def test_scaled_optic_has_four_times_area():
    # Doubling f and aperture keeps every graze angle; area scales by 4.
    small = lobster_eye([1.0], material=None, **GEOMETRY)["area_focus"]
    big = lobster_eye([1.0], material=None, **{**GEOMETRY, "focal_length": 60, "aperture": 18})["area_focus"]
    assert big == pytest.approx(4 * small, rel=1e-3)


def test_focus_area_falls_at_high_energy():
    area = lobster_eye([1.0, 8.0], **GEOMETRY)["area_focus"]
    assert area[1] < area[0]


@pytest.mark.reference
@pytest.mark.skip(reason="needs a published micropore-optic effective area (e.g. SVOM MXT central spot)")
def test_matches_published_optic():
    pass
