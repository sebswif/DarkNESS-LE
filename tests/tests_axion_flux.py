# %% Imports and axion flux reference test
import unittest

import numpy as np

from functions.solar_flux import axion_flux, spectrum_summary


class AxionFluxChecks(unittest.TestCase):
    def test_solar_spectrum_reference(self):
        """Check the approximate reference spectrum at g = 1e-10 GeV^-1."""
        energies = np.linspace(0.0, 60.0, 60001)
        flux = axion_flux(energies, 1e-10)
        total = np.trapezoid(flux, energies)
        mean = np.trapezoid(energies * flux, energies) / total
        peak = energies[np.argmax(flux)]
        summary_total, summary_mean, summary_peak = spectrum_summary(1e-10)

        # Compare the analytical summary with the sampled differential flux.
        self.assertAlmostEqual(total / summary_total, 1.0, places=6)
        self.assertAlmostEqual(mean, summary_mean, places=6)
        self.assertAlmostEqual(peak, summary_peak, delta=0.001)

        # Reference values are approximate, especially the quoted mean of 4.20.
        self.assertAlmostEqual(total / 3.75e11, 1.0, delta=0.001)
        self.assertAlmostEqual(mean, 4.20, delta=0.01)
        self.assertAlmostEqual(peak, 3.0, delta=0.02)
        self.assertAlmostEqual(summary_total / 3.75e11, 1.0, delta=0.001)
        self.assertAlmostEqual(summary_mean, 4.20, delta=0.01)
        self.assertAlmostEqual(summary_peak, 3.0, delta=0.02)
        print(f"Total flux: {total:.5e} cm^-2 s^-1")
        print(f"Mean energy: {mean:.6f} keV")
        print(f"Peak energy (0.001 keV grid): {peak:.3f} keV")
        print("spectrum_summary:")
        print(f"Total flux: {summary_total:.5e} cm^-2 s^-1")
        print(f"Mean energy: {summary_mean:.6f} keV")
        print(f"Peak energy: {summary_peak:.6f} keV")


# %% Run the reference test (run the first cell before this cell)
if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(AxionFluxChecks)
    unittest.TextTestRunner(verbosity=2).run(suite)
