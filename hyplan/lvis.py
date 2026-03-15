"""
LVIS (Land, Vegetation, and Ice Sensor) airborne lidar.

Models the LVIS full-waveform scanning lidar for use with HyPlan's flight
planning tools (swath polygons, flight box generation, etc.).

The scanner geometry defines a maximum swath as a fixed fraction of altitude
(±5.7 deg half-scan angle). Whether that full swath achieves contiguous
ground coverage depends on the laser repetition rate, lens divergence
(footprint size), and aircraft speed.

Formulas derived from NASA GSFC LVIS coverage planning spreadsheet
(LVIScoveragecalcs-GVBioSCape.xlsx).
"""

import numpy as np
from dataclasses import dataclass
from pint import Quantity
from .units import ureg
from .sensors import Sensor


# Max swath = 0.2 * altitude  →  half-scan angle = atan(0.1) ≈ 5.71°
_MAX_SWATH_FRACTION = 0.2
_HALF_SCAN_ANGLE_DEG = np.degrees(np.arctan(_MAX_SWATH_FRACTION / 2.0))


@dataclass
class LVISLens:
    """LVIS lens option defined by its beam divergence."""
    name: str
    divergence_mrad: float

    def footprint_diameter(self, altitude_agl: Quantity) -> Quantity:
        """Footprint diameter on the ground for a given altitude AGL.

        Args:
            altitude_agl: Flight altitude above ground level.

        Returns:
            Footprint diameter in meters.
        """
        alt_m = altitude_agl.to(ureg.meter).magnitude
        return np.tan(self.divergence_mrad / 1000.0) * alt_m * ureg.meter


# Standard LVIS lens options
LVIS_LENS_NARROW = LVISLens("narrow", 0.531)
LVIS_LENS_MEDIUM = LVISLens("medium", 0.79)
LVIS_LENS_WIDE = LVISLens("wide", 1.272)

LVIS_LENSES = {
    "narrow": LVIS_LENS_NARROW,
    "medium": LVIS_LENS_MEDIUM,
    "wide": LVIS_LENS_WIDE,
}


class LVIS(Sensor):
    """LVIS full-waveform airborne scanning lidar.

    Provides the standard Sensor interface (half_angle, swath_width) so that
    LVIS works with generate_swath_polygon, generate_flight_lines, and other
    HyPlan tools. The swath_width returned is the geometric maximum; use
    effective_swath_width to account for contiguous-coverage constraints.

    All altitude parameters expect AGL (above ground level).

    Args:
        rep_rate: Laser pulse repetition rate (Hz). Default 4000 Hz.
        lens: Lens option — LVISLens instance, key from LVIS_LENSES
            ("narrow", "medium", "wide"), or None for wide (default).
    """

    def __init__(
        self,
        rep_rate: Quantity = 4000 * ureg.Hz,
        lens: object = None,
    ):
        super().__init__(name="LVIS")

        if isinstance(rep_rate, Quantity):
            self.rep_rate = rep_rate.to(ureg.Hz)
        else:
            self.rep_rate = float(rep_rate) * ureg.Hz

        if lens is None:
            self.lens = LVIS_LENS_WIDE
        elif isinstance(lens, str):
            if lens not in LVIS_LENSES:
                raise ValueError(
                    f"Unknown lens '{lens}'. Choose from: {list(LVIS_LENSES.keys())}"
                )
            self.lens = LVIS_LENSES[lens]
        elif isinstance(lens, LVISLens):
            self.lens = lens
        else:
            raise TypeError(f"lens must be a LVISLens, str, or None, got {type(lens)}")

    # ------------------------------------------------------------------
    # Standard Sensor interface (used by swath.py, flight_box.py, glint.py)
    # ------------------------------------------------------------------

    @property
    def half_angle(self) -> float:
        """Half-scan angle in degrees (≈5.71 deg)."""
        return _HALF_SCAN_ANGLE_DEG

    def swath_width(self, altitude_agl: Quantity) -> Quantity:
        """Maximum swath width set by the scanner geometry.

        This is the geometric limit: 2 * altitude_agl * tan(half_angle).
        Equivalent to 0.2 * altitude_agl.

        Used by generate_swath_polygon and generate_flight_lines for
        swath polygon generation and line spacing.

        Args:
            altitude_agl: Flight altitude above ground level.

        Returns:
            Maximum swath width in meters.
        """
        altitude_agl = self._validate_quantity(altitude_agl, ureg.meter)
        return 2 * altitude_agl * np.tan(np.radians(self.half_angle))

    # ------------------------------------------------------------------
    # LVIS-specific methods
    # ------------------------------------------------------------------

    def effective_fov(self, altitude_agl: Quantity, speed: Quantity) -> float:
        """Effective field of view in degrees, accounting for contiguous coverage.

        When the laser footprint can fill the full scanner swath, this equals
        the geometric max FOV (≈11.42°). When coverage is limited, the
        effective FOV narrows: 2 * atan(effective_swath / (2 * altitude)).

        Args:
            altitude_agl: Flight altitude above ground level.
            speed: Aircraft ground speed.

        Returns:
            Effective FOV in degrees.
        """
        esw = self.effective_swath_width(altitude_agl, speed).magnitude
        alt_m = altitude_agl.to(ureg.meter).magnitude
        return 2 * np.degrees(np.arctan(esw / (2 * alt_m)))

    def footprint_diameter(self, altitude_agl: Quantity) -> Quantity:
        """Laser footprint diameter on the ground for the configured lens.

        footprint = tan(divergence_mrad / 1000) * altitude_agl

        Args:
            altitude_agl: Flight altitude above ground level.

        Returns:
            Footprint diameter in meters.
        """
        return self.lens.footprint_diameter(altitude_agl)

    def coverage_rate(self, altitude_agl: Quantity, speed: Quantity) -> Quantity:
        """Area coverage rate at maximum swath.

        coverage_rate = speed * max_swath

        Args:
            altitude_agl: Flight altitude above ground level.
            speed: Aircraft ground speed.

        Returns:
            Coverage rate in m^2/s.
        """
        spd = speed.to(ureg.meter / ureg.second).magnitude
        ms = self.swath_width(altitude_agl).magnitude
        return spd * ms * ureg.meter ** 2 / ureg.second

    def footprint_for_max_swath(self, altitude_agl: Quantity, speed: Quantity) -> Quantity:
        """Minimum footprint diameter needed to fill the max swath contiguously.

        footprint = sqrt(speed * max_swath / rep_rate)

        Args:
            altitude_agl: Flight altitude above ground level.
            speed: Aircraft ground speed.

        Returns:
            Required footprint diameter in meters.
        """
        cr = self.coverage_rate(altitude_agl, speed).magnitude
        rr = self.rep_rate.magnitude
        return np.sqrt(cr / rr) * ureg.meter

    def effective_swath_width(self, altitude_agl: Quantity, speed: Quantity) -> Quantity:
        """Achievable swath width accounting for contiguous coverage.

        The effective swath is the minimum of:
        - the scanner's max swath (geometric limit), and
        - footprint^2 * rep_rate / speed (contiguous-coverage limit)

        When the footprint is too small relative to the flight speed,
        shots cannot tile the full max swath without gaps, and the
        effective swath narrows.

        Args:
            altitude_agl: Flight altitude above ground level.
            speed: Aircraft ground speed.

        Returns:
            Effective swath width in meters.
        """
        ms = self.swath_width(altitude_agl).magnitude
        fp = self.footprint_diameter(altitude_agl).magnitude
        spd = speed.to(ureg.meter / ureg.second).magnitude
        rr = self.rep_rate.magnitude

        contiguous_swath = fp ** 2 * rr / spd
        return min(ms, contiguous_swath) * ureg.meter

    def is_contiguous(self, altitude_agl: Quantity, speed: Quantity) -> bool:
        """Check whether the current configuration fills the max swath.

        Returns True if the footprint is large enough to tile the full
        scanner swath at the given speed.
        """
        ms = self.swath_width(altitude_agl).magnitude
        esw = self.effective_swath_width(altitude_agl, speed).magnitude
        return esw >= ms * 0.999  # tolerance for floating point

    def summary(self, altitude_agl: Quantity, speed: Quantity) -> dict:
        """Compute all LVIS coverage parameters for a given flight configuration.

        Args:
            altitude_agl: Flight altitude above ground level.
            speed: Aircraft ground speed.

        Returns:
            Dictionary with all computed parameters.
        """
        return {
            "altitude_agl": altitude_agl.to(ureg.meter),
            "speed": speed.to(ureg.knot),
            "rep_rate": self.rep_rate,
            "lens": self.lens.name,
            "lens_divergence_mrad": self.lens.divergence_mrad,
            "footprint_diameter": self.footprint_diameter(altitude_agl),
            "max_swath": self.swath_width(altitude_agl),
            "effective_swath_width": self.effective_swath_width(altitude_agl, speed),
            "contiguous": self.is_contiguous(altitude_agl, speed),
            "coverage_rate": self.coverage_rate(altitude_agl, speed),
            "footprint_for_max_swath": self.footprint_for_max_swath(altitude_agl, speed),
        }

    def print_summary(self, altitude_agl: Quantity, speed: Quantity) -> None:
        """Print a formatted summary of LVIS coverage parameters."""
        s = self.summary(altitude_agl, speed)
        print("LVIS Coverage Summary")
        print(f"  Altitude AGL:          {s['altitude_agl']:.0f}")
        print(f"  Speed:                 {s['speed']:.0f}")
        print(f"  Rep rate:              {s['rep_rate']:.0f}")
        print(f"  Lens:                  {s['lens']} ({s['lens_divergence_mrad']:.3f} mrad)")
        print(f"  Footprint diameter:    {s['footprint_diameter']:.1f}")
        print(f"  Max swath (geometric): {s['max_swath']:.0f}")
        print(f"  Effective swath width: {s['effective_swath_width']:.1f}")
        print(f"  Contiguous coverage:   {s['contiguous']}")
        print(f"  Coverage rate:         {s['coverage_rate']:.0f}")
        print(f"  Footprint to fill max: {s['footprint_for_max_swath']:.1f}")

    def compare_lenses(self, altitude_agl: Quantity, speed: Quantity) -> None:
        """Print a comparison table across all standard lenses."""
        spd = speed.to(ureg.meter / ureg.second).magnitude
        rr = self.rep_rate.magnitude
        ms = self.swath_width(altitude_agl).magnitude
        alt_m = altitude_agl.to(ureg.meter).magnitude

        print(f"LVIS Lens Comparison at {altitude_agl.to(ureg.km):.1f} AGL, {speed.to(ureg.knot):.0f}")
        print(f"  Rep rate: {self.rep_rate:.0f}   Max swath: {ms:.0f} m")
        print(f"  {'Lens':10s}  {'Diverg':>8s}  {'Footprint':>10s}  {'Eff Swath':>10s}  {'Contiguous':>10s}")
        print(f"  {'-'*10}  {'-'*8}  {'-'*10}  {'-'*10}  {'-'*10}")

        for name, lens in LVIS_LENSES.items():
            fp = np.tan(lens.divergence_mrad / 1000.0) * alt_m
            contiguous = fp ** 2 * rr / spd
            swath = min(ms, contiguous)
            fills = "yes" if contiguous >= ms else "no"
            print(
                f"  {name:10s}  {lens.divergence_mrad:7.3f}   {fp:9.1f} m  {swath:9.1f} m  {fills:>10s}"
            )
