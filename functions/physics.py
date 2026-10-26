import numpy as np
import scipy.interpolate as ip
def apply_Si_QE(energy_keV, counts, tab_E_MeV, tab_MAC,
                rho=2.33, thickness_um=250):
    """
    Apply silicon detector QE to a model spectrum.

    Parameters
    ----------
    energy_keV : array
        Photon energies of the spectrum [keV].
    counts : array
        Model counts or flux at those energies.
    tab_E_MeV, tab_MAC : arrays
        NIST absorption data (energies [MeV], MAC [cm^2/g]).
    rho : float
        Si density [g/cm^3].
    thickness_um : float
        Detector thickness [µm].

    Returns
    -------
    counts_detected : array
        Counts after QE applied.
    """
    eff = Si_QE(energy_keV, np.asarray(tab_E_MeV) * 1000, tab_MAC,
                rho=rho, thickness_um=thickness_um)
    return counts * eff

def Si_QE(MODEL_GRID, REF_GRID, REF_MAC,
          rho=2.33, thickness_um=250):
    """
    Compute the quantum efficiency (QE) of a silicon detector
    on a given model energy grid, using NIST reference data.

    Parameters
    ----------
    MODEL_GRID : float or array
        Photon energies [keV] where QE should be evaluated
        (e.g. your spectrum grid).
    REF_GRID : array
        Reference (NIST) tabulated energies [keV].
    REF_MAC : array
        Reference (NIST) mass attenuation coefficients [cm^2/g].
    rho : float, optional
        Silicon density [g/cm^3]. Default = 2.33.
    thickness_um : float, optional
        Detector thickness [µm]. Default = 250.

    Returns
    -------
    QE : float or array
        Detector quantum efficiency (0–1).
    """
    # interpolate reference MAC onto model grid
    mac_interp = ip.interp1d(REF_GRID, REF_MAC,
                             bounds_error=False,
                             fill_value="extrapolate")
    mu = mac_interp(MODEL_GRID) * rho      # [cm^-1]
    tau = (mu / 1e4) * thickness_um        # optical depth
    return 1 - np.exp(-tau)


