"""Print Si_QE for 500 micrometres of silicon at 10 keV.

Run from the project root: python -m tests.test_physics
"""

from functions.physics import Si_QE


if __name__ == "__main__":
    # NIST silicon mu/rho [cm^2/g] at reference energies [keV].
    # https://physics.nist.gov/PhysRefData/XrayMassCoef/ElemTab/z14.html
    reference_energy_keV = [8.0, 10.0, 15.0]
    reference_mac_cm2_g = [64.68, 33.89, 10.34]

    qe = Si_QE(
        10.0, reference_energy_keV, reference_mac_cm2_g,
        rho=2.33, thickness_um=500.0,
    )
    print("Energy: 10 keV; Si thickness: 500 micrometres; density: 2.33 g/cm^3")
    print("NIST mass attenuation coefficient at 10 keV: 33.89 cm^2/g")
    print(f"QE: {float(qe):.10f} ({float(qe) * 100:.6f}%)")
