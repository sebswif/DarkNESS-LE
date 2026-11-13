# Units, frames, constants

## Units

| Quantity | Unit |
|---|---|
| Photon / axion energy `e` | keV |
| Coupling `g` | GeV⁻¹ |
| Axion mass `ma`, photon mass `mg` | eV |
| Magnetic field `b` | T |
| Positions, altitudes, orbit | km |
| Conversion path `s`, `length` in conversion | m |
| Air density `rho` | kg/m³ |
| Mass attenuation `mac` | cm²/g |
| Column density | g/cm² |
| Attenuation coefficient | 1/m |
| Effective area | cm² |
| Axion flux | cm⁻² s⁻¹ keV⁻¹ |
| Rigidity | GV |
| Time | UTC (ISO string or astropy `Time`) |

## Natural-unit conversions (conversion.py)

- 1 T = 195 eV²
- 1 m = 5.07 × 10⁶ eV⁻¹
- 1 GeV⁻¹ = 10⁻⁹ eV⁻¹

## Frames

- ECI = GCRS, ECEF = ITRS (astropy). Field and rays are in ECEF.
- Astropy runs with bundled IERS data, no downloads.

## Earth shapes

- IGRF reference sphere: R = 6371.2 km (`bfield`, `lineofsight`).
- WGS84 ellipsoid: a = 6378.137 km (`geometry`, `background/transport`).
- These differ; altitudes from the two are not directly comparable.
