# %% Imports and uniform conversion reference test
import unittest

from functions.conversion import conversion_probability_uniform


class ConversionChecks(unittest.TestCase):
    def test_uniform_conversion_reference(self):
        """Check 9 T, 9.26 m, and g = 1e-10 GeV^-1 against P ~ 1.70e-17."""
        probability = conversion_probability_uniform(b=9.0, length=9.26, g=1e-10)

        # Use relative tolerance because the probability is very small.
        self.assertAlmostEqual(probability / 1.70e-17, 1.0, delta=0.01)
        print("B = 9 T, length = 9.26 m, g = 1e-10 GeV^-1")
        print(f"Conversion probability: {probability:.8e} (dimensionless)")


# %% Run the reference test (run the first cell before this cell)
if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(ConversionChecks)
    unittest.TextTestRunner(verbosity=2).run(suite)
