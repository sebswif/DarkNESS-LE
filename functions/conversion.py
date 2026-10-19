import numpy as np

def conversion_probability_uniform(B_T, L_m, g_ay):
    """Return the coherent, weak-mixing vacuum conversion probability.

    Assumes negligible momentum mismatch and absorption, and P << 1.

    Parameters
    ----------
    B_T : float
        Transverse magnetic field strength in Tesla.
    L_m : float
        Length of the magnetic field region in meters.
    g_ay : float
        Axion-photon coupling constant in GeV^-1.

    Returns
    -------
    P : float
        Conversion probability (dimensionless).
    """
    # Convert units: Tesla to eV^2, meters to eV^-1
    B_T_eV2 = B_T * 195.0  # 1 T = 195 eV^2
    L_m_eV_inv = L_m * 5.07e6  # 1 m = 5.07e6 eV^-1
    g_eV_inv = g_ay * 1e-9  # 1 GeV^-1 = 1e-9 eV^-1

    # Conversion probability formula: P = (g_ay * B_T * L)^2 / 4
    P = (g_eV_inv * B_T_eV2 * L_m_eV_inv)**2 / 4.0

    return P

if __name__ == "__main__":
    p_example = conversion_probability_uniform(9, 9.26, 1e-10)
    print(f"Conversion probability (example): {p_example:.3e}")
