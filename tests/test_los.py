"""Run nadir and grazing LOS examples and print results for manual review.

Run from the project root: python -m tests.test_los
No expected answers, assertions, or magnetic field overrides are used.
"""

import numpy as np

from functions.bfield import R_EARTH_KM, igrf_field
from functions.lineofsight import los


TIME = "2026-10-19T00:00:00"  # Assumed UTC date; edit as needed.
ALTITUDE_KM = 600.0
NADIR_LENGTH_KM = 540.0
GRAZING_LENGTH_KM = 2640.0
SAMPLES = 501

# Approximate city coordinates in degrees. For these examples, latitude
# is treated as geocentric to match the spherical Earth used by los.
CITIES = [
    ("Boulder", 40.0150, -105.2705),
    ("Sao Paulo", -23.5505, -46.6333),
    ("Yakutsk", 62.0355, 129.6755),
]


def main():
    np.set_printoptions(precision=10)
    print(f"UTC time: {TIME}; spacecraft altitude: {ALTITUDE_KM} km")
    print("Approximate city latitudes are treated as geocentric.")
    print("Grazing direction: eastward, tangent to the 60 km cutoff shell.")
    print("540 and 2640 km are requested range limits, not predicted ranges.")
    print("Magnetic fields are calculated by IGRF, not supplied as inputs.")
    print("Ground reference fields use the same spherical Earth at 0 km.")
    print("Ray endpoints are near 60 km altitude, not at ground level.")

    for city, latitude, longitude in CITIES:
        lat, lon = np.deg2rad([latitude, longitude])
        up = np.array([
            np.cos(lat) * np.cos(lon),
            np.cos(lat) * np.sin(lon),
            np.sin(lat),
        ])
        east = np.array([-np.sin(lon), np.cos(lon), 0.0])
        radius = R_EARTH_KM + ALTITUDE_KM
        position = radius * up

        # Choose a ray whose infinite line just touches the cutoff shell.
        # Its finite requested range may end before the tangent point.
        shell_ratio = (R_EARTH_KM + 60.0) / radius
        grazing = -np.sqrt(1 - shell_ratio**2) * up + shell_ratio * east

        print(f"\n{city}: latitude {latitude}, longitude {longitude} deg")
        print(f"Spacecraft ECEF position [km]: {position}")
        ground_field = igrf_field(R_EARTH_KM * up, TIME, report=False)
        print(f"|B| at city ground reference (0 km) [microtesla]: "
              f"{np.linalg.norm(ground_field) * 1e6:.10f}")
        for name, direction, length in (
            ("Nadir", -up, NADIR_LENGTH_KM),
            ("Grazing", grazing, GRAZING_LENGTH_KM),
        ):
            ray = los(
                position, direction, TIME, length=length,
                n=SAMPLES, report=False,
            )
            field_uT = np.linalg.norm(ray["B_ecef_T"], axis=1) * 1e6
            endpoint = ray["path_ecef_km"][-1]
            endpoint_radius = np.linalg.norm(endpoint)
            endpoint_lat = np.degrees(np.arctan2(
                endpoint[2], np.hypot(endpoint[0], endpoint[1]),
            ))
            endpoint_lon = np.degrees(np.arctan2(endpoint[1], endpoint[0]))
            print(f"\n  {name}: requested range {length} km")
            print(f"  Look direction ECEF: {ray['look_ecef_unit']}")
            print(f"  Zenith angle [deg]: {ray['zenith_deg']:.10f}")
            print(f"  Actual range [km]: {ray['distance_km'][-1]:.10f}")
            print(f"  Minimum altitude [km]: {ray['minimum_altitude_km']:.10f}")
            print(f"  Stopped at cutoff: {ray['stopped_at_cutoff']}")
            print(f"  Endpoint ECEF [km]: {ray['path_ecef_km'][-1]}")
            print(f"  Endpoint geocentric latitude/longitude [deg]: "
                  f"{endpoint_lat:.6f}, {endpoint_lon:.6f}")
            print(f"  Endpoint altitude [km]: "
                  f"{endpoint_radius - R_EARTH_KM:.10f}")
            print(f"  |B| at spacecraft [microtesla]: {field_uT[0]:.10f}")
            print(f"  |B| at endpoint [microtesla]: {field_uT[-1]:.10f}")
            print(f"  |B| min/max along ray [microtesla]: "
                  f"{field_uT.min():.10f}, {field_uT.max():.10f}")
            print(f"  Transverse field integral ECEF [T m]: "
                  f"{ray['integral_ecef_Tm']}")
            print(f"  Integral amplitude [T m]: {ray['amplitude_Tm']:.10f}")


if __name__ == "__main__":
    main()
