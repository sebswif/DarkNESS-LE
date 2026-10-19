"""Physical invariants for fields, rays, and orbit geometry."""

import unittest
from unittest.mock import patch

import numpy as np
from astropy import units as u
from astropy.time import Time

from functions.field import (
    R_EARTH_KM, dipole_field, igrf_field, load_igrf,
)
from functions.geometry import (
    MU_EARTH_KM3_S2, R_EQUATOR_KM, orbit_position, sun_direction,
)
from functions.lineofsight import line_of_sight
from functions.solar_flux import axion_flux

T = "2026-10-19T00:00:00"
R = np.array([R_EARTH_KM + 500, 0, 0])


class FieldChecks(unittest.TestCase):
    def test_official_igrf14_values(self):
        # NOAA pyIGRF14/tests/tests_igrf14.py; angles are geocentric.
        # Columns: year, colatitude [deg], longitude [deg], radius [km],
        # expected north/east/down field [nT].
        cases = [
            (1900, 175, -150, 6300, [-5072.93, 10620.34, -67233.55]),
            (1915, 155, -120, 6350, [14692.62, 12387.97, -59640.81]),
            (1930, 135, -90, 6400, [23925.47, 10358.94, -30640.98]),
            (1945, 115, -60, 6450, [23642.86, -200.29, -7607.92]),
            (1960, 95, -30, 6500, [23647.00, -9302.27, -3610.73]),
            (1975, 75, 0, 6550, [30050.59, -3367.82, 6332.69]),
            (1990, 55, 30, 6600, [25224.81, 1058.25, 30965.61]),
            (2005, 35, 60, 6650, [14718.37, 2842.99, 46050.88]),
            (2010, 170, 0, 6371, [17529.48, -7143.78, -42722.46]),
            (2020, 15, 90, 6700, [3734.07, 1294.17, 50833.13]),
            (2025, 56, -3, 6375, [28927.56, 261.98, 30910.08]),
        ]
        for year, theta, phi, radius, expected in cases:
            with self.subTest(year=year):
                theta, phi = np.deg2rad([theta, phi])
                c, s, cp, sp = (
                    np.cos(theta), np.sin(theta), np.cos(phi), np.sin(phi),
                )
                r = radius * np.array([s * cp, s * sp, c])
                b = igrf_field(r, f"{year}-01-01", report=False) * 1e9
                ned = np.array([[-c * cp, -c * sp, s],
                                [-sp, cp, 0], [-s * cp, -s * sp, -c]]) @ b
                np.testing.assert_allclose(ned, expected, atol=0.02, rtol=0)

    def test_coefficient_interpolation_and_variation(self):
        g0, h0 = load_igrf(2020)
        g1, h1 = load_igrf(2025)
        g, h = load_igrf(2022.5)
        np.testing.assert_allclose(g, (g0 + g1) / 2)
        np.testing.assert_allclose(h, (h0 + h1) / 2)
        g, h = load_igrf(2030)
        self.assertAlmostEqual(g[1, 0], -29350 + 5 * 12.6)
        self.assertAlmostEqual(h[1, 1], 4545.5 - 5 * 21.5)
        for year in (1899.9, 2030.1, np.nan):
            with self.assertRaises(ValueError):
                load_igrf(year)

    def test_poles_and_dipole_solution(self):
        g, h = np.zeros((2, 14, 14))
        g[1, 0], g[1, 1], h[1, 1] = -30000, -1500, 4500
        r = R_EARTH_KM * np.array([
            [1, 0, 0], [0, 0, 1], [0, 0, -1], [0, 1, 0],
        ])
        # B = 3 (m dot r_hat) r_hat - m at the reference radius.
        expected = np.array([
            [-3000, -4500, 30000], [1500, -4500, -60000],
            [1500, -4500, -60000], [1500, 9000, 30000],
        ]) * 1e-9
        with patch("functions.field.load_igrf", return_value=(g, h)):
            np.testing.assert_allclose(
                dipole_field(r, T, report=False), expected, atol=1e-18,
            )
        for z in (-7000, 7000):
            b = igrf_field([0, 0, z], T, report=False)
            for phi in np.linspace(0, 2 * np.pi, 5):
                near = [1e-7 * np.cos(phi), 1e-7 * np.sin(phi), z]
                np.testing.assert_allclose(
                    igrf_field(near, T, report=False), b, atol=1e-14, rtol=0,
                )

    def test_field_shapes_and_time_scales(self):
        for model in (igrf_field, dipole_field):
            b = model(R, T, report=False)
            batch = model([[R, R], [R, R]], T, report=False)
            self.assertEqual(batch.shape, (2, 2, 3))
            np.testing.assert_allclose(batch, np.broadcast_to(b, batch.shape))
            self.assertTrue(1e-5 < np.linalg.norm(b) < 1e-4)
            np.testing.assert_allclose(
                model(R, Time(T).tai, report=False), b, rtol=1e-12,
            )

    def test_dipole_inverse_cube(self):
        b = dipole_field(R, T, report=False)
        np.testing.assert_allclose(
            dipole_field(2 * R, T, report=False), b / 8,
        )

    def test_nadir_cutoff_and_outward_range(self):
        ray = line_of_sight(R, [-1, 0, 0], T, n=11, report=False)
        self.assertTrue(ray["stopped_at_cutoff"])
        self.assertAlmostEqual(ray["distance_km"][-1], 440)
        self.assertAlmostEqual(ray["minimum_altitude_km"], 60)
        self.assertAlmostEqual(ray["zenith_deg"], 180)
        np.testing.assert_allclose(ray["B_perp_ecef_T"][:, 0], 0, atol=1e-20)
        ray = line_of_sight(R, [2, 0, 0], T, length=100, n=3, report=False)
        self.assertFalse(ray["stopped_at_cutoff"])
        self.assertAlmostEqual(ray["distance_km"][-1], 100)
        self.assertAlmostEqual(ray["minimum_altitude_km"], 500)

    def test_transverse_vector_cancellation(self):
        b = np.array([[1, 1, 0], [1, 0, 0], [1, -1, 0]], dtype=float)
        with patch("functions.lineofsight.igrf_field", return_value=b):
            ray = line_of_sight(
                R, [1, 0, 0], T, length=1, n=3, report=False,
            )
        self.assertEqual(ray["amplitude_Tm"], 0)
        np.testing.assert_allclose(ray["integral_ecef_Tm"], 0)

    def test_invalid_field_and_ray_inputs(self):
        for r in ([0, 0, 0], [1, 2], [1, np.nan, 0]):
            with self.assertRaises(ValueError):
                igrf_field(r, T, report=False)
        for args in ({"n": 1}, {"length": 0}, {"n": 2.5}):
            with self.assertRaises(ValueError):
                line_of_sight(R, [1, 0, 0], T, report=False, **args)
        with self.assertRaises(ValueError):
            line_of_sight(R, [0, 0, 0], T, report=False)


class GeometryChecks(unittest.TestCase):
    def test_sun_units_and_time_shapes(self):
        t = Time([[T, T], [T, T]])
        for frame in ("eci", "ecef"):
            k = sun_direction(t, frame)
            self.assertEqual(k.shape, (2, 2, 3))
            np.testing.assert_allclose(np.linalg.norm(k, axis=-1), 1)
            self.assertEqual(sun_direction(T, frame).shape, (3,))
            np.testing.assert_allclose(k[0, 0], sun_direction(T, frame))

    def test_circular_orbit_period_and_radius(self):
        a = R_EQUATOR_KM + 500
        period = 2 * np.pi * np.sqrt(a**3 / MU_EARTH_KM3_S2)
        t = Time(T)
        r = orbit_position(t, frame="eci", j2=False)
        np.testing.assert_allclose(r, [a, 0, 0], atol=1e-10)
        np.testing.assert_allclose(
            orbit_position(t + period * u.s, frame="eci", j2=False),
            r, atol=1e-7,
        )
        for frame in ("eci", "ecef"):
            r = orbit_position([T, "2026-10-20T00:00:00"], frame=frame)
            self.assertEqual(r.shape, (2, 3))
            np.testing.assert_allclose(np.linalg.norm(r, axis=-1), a)
        self.assertGreater(np.linalg.norm(
            orbit_position("2026-10-20", frame="eci")
            - orbit_position("2026-10-20", frame="eci", j2=False)
        ), 1)

    def test_invalid_orbit_and_frame(self):
        for args in ({"h": -1}, {"inc": 181}, {"h": np.nan}):
            with self.assertRaises(ValueError):
                orbit_position(T, **args)
        with self.assertRaises(ValueError):
            sun_direction(T, "unknown")

    def test_nonpositive_energy_without_warnings(self):
        with np.errstate(all="raise"):
            np.testing.assert_array_equal(axion_flux([-1, 0], 1e-10), [0, 0])


if __name__ == "__main__":
    unittest.main()
