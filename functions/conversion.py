"""Coherent axion-photon conversion."""


def conversion_probability_uniform(b: float, length: float, g: float) -> float:
    """Calculate vacuum conversion in a uniform transverse field.

    Inputs: b, field [T]; length, path length [m]; g, coupling [GeV^-1].
    Output: dimensionless probability; assumes coherence and weak mixing.
    Reference: Raffelt & Stodolsky (1988), doi:10.1103/PhysRevD.37.1237.
    """
    b = b * 195.0  # eV^2
    length = length * 5.07e6  # eV^-1
    g = g * 1e-9  # eV^-1
    return (g * b * length / 2) ** 2


if __name__ == "__main__":
    print(conversion_probability_uniform(9, 9.26, 1e-10))
