import unittest

import numpy as np

from functions.conversion import conversion_probability_uniform
from functions.solar_flux import axion_flux, spectrum_summary


class PhysicsChecks(unittest.TestCase):
    def test_cast_benchmark(self):
        self.assertAlmostEqual(
            conversion_probability_uniform(9.0, 9.26, 1e-10) / 1.7e-17,
            1.0, delta=0.01,
        )
        self.assertEqual(conversion_probability_uniform(0, 9.26, 1e-10), 0)

    def test_solar_spectrum_integral(self):
        energies = np.linspace(0.0, 60.0, 60001)
        flux = axion_flux(energies, 1e-10)
        integrate = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
        total, mean, peak = spectrum_summary(1e-10)
        self.assertAlmostEqual(integrate(flux, energies) / total, 1, places=6)
        self.assertAlmostEqual(integrate(energies * flux, energies) / total, mean, places=6)
        self.assertAlmostEqual(energies[np.argmax(flux)], peak, delta=0.001)

    def test_coupling_scaling(self):
        self.assertAlmostEqual(axion_flux(4, 2e-10) / axion_flux(4, 1e-10), 4)
        self.assertAlmostEqual(
            conversion_probability_uniform(9, 9.26, 2e-10)
            / conversion_probability_uniform(9, 9.26, 1e-10), 4,
        )


if __name__ == "__main__":
    unittest.main()
