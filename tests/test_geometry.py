"""Print geometry outputs for manual verification; no expected answers."""

import numpy as np

from functions.geometry import R_EQUATOR_KM, orbit_position, sun_direction


# Both functions require a UTC time; edit it to your observation time.
TIME = "2026-10-19T00:00:00"
EPOCH = "2026-10-19T00:00:00"
ALTITUDE_KM = 600.0


def main():
    # Path lengths, cities, and magnetic field strengths are not inputs
    # to either geometry function.
    np.set_printoptions(precision=10)
    print(f"UTC time: {TIME}")
    print(f"Orbit epoch: {EPOCH}")
    print(f"Input altitude: {ALTITUDE_KM} km")
    print("Other orbit settings: function defaults")

    for frame in ("eci", "ecef"):
        direction = sun_direction(TIME, frame=frame)
        position = orbit_position(
            TIME, epoch=EPOCH, h=ALTITUDE_KM, frame=frame,
        )
        print(f"\nFrame: {frame.upper()}")
        print(f"Sun direction [x, y, z] (unit vector): {direction}")
        print(f"Sun direction magnitude: {np.linalg.norm(direction):.10f}")
        print(f"Orbit position [x, y, z] (km from Earth center): {position}")
        print(f"Distance from Earth center (km): {np.linalg.norm(position):.10f}")
        print(
            "Altitude above equatorial reference radius (km): "
            f"{np.linalg.norm(position) - R_EQUATOR_KM:.10f}"
        )


if __name__ == "__main__":
    main()
