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

References
----------
Blair, J.B., Rabine, D.L. and Hofton, M.A. (1999). The Laser Vegetation
Imaging Sensor: a medium-altitude, digitisation-only, airborne laser
altimeter for mapping vegetation and topography. *ISPRS Journal of
Photogrammetry and Remote Sensing*, 54(2-3), 115-122.
doi:10.1016/S0924-2716(99)00002-7
"""

from typing import Optional

import numpy as np
import pymap3d.vincenty
from dataclasses import dataclass
from pint import Quantity
from ..units import ureg
from ._base import Sensor
from ..exceptions import HyPlanTypeError, HyPlanValueError

__all__ = [
    "LVISLens",
    "LVIS_LENS_NARROW",
    "LVIS_LENS_MEDIUM",
    "LVIS_LENS_WIDE",
    "LVIS_LENSES",
    "LVIS",
]


# Default max swath = 0.2 * altitude  →  half-scan angle = atan(0.1) ≈ 5.71°
_DEFAULT_HALF_SCAN_ANGLE_DEG = np.degrees(np.arctan(0.1))


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
        alt_m = altitude_agl.m_as(ureg.meter)
        return np.tan(self.divergence_mrad / 1000.0) * alt_m * ureg.meter  # type: ignore[no-any-return]


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
        scan_half_angle_deg: Half-scan angle in degrees.  Controls the
            geometric maximum swath via ``2 * altitude * tan(angle)``.
            Default ≈5.71° corresponds to max swath = 0.2 * altitude.
    """

    # Tolerance for floating-point error when comparing effective swath width to
    # the geometric maximum. Accounts for rounding in footprint/speed calculations
    # that can make a fully contiguous configuration appear fractionally short.
    _CONTIGUITY_TOLERANCE = 0.999

    def __init__(
        self,
        rep_rate: Quantity = 4000 * ureg.Hz,
        lens: object = None,
        scan_half_angle_deg: float = _DEFAULT_HALF_SCAN_ANGLE_DEG,
    ):
        super().__init__(name="LVIS")

        if isinstance(rep_rate, Quantity):
            self.rep_rate = rep_rate.to(ureg.Hz)
        else:
            self.rep_rate = float(rep_rate) * ureg.Hz

        if self.rep_rate.magnitude <= 0:
            raise HyPlanValueError("rep_rate must be positive")

        if scan_half_angle_deg <= 0 or scan_half_angle_deg >= 90:
            raise HyPlanValueError(
                "scan_half_angle_deg must be between 0 and 90 degrees"
            )
        self._scan_half_angle_deg = float(scan_half_angle_deg)

        if lens is None:
            self.lens = LVIS_LENS_WIDE
        elif isinstance(lens, str):
            if lens not in LVIS_LENSES:
                raise HyPlanValueError(
                    f"Unknown lens '{lens}'. Choose from: {list(LVIS_LENSES.keys())}"
                )
            self.lens = LVIS_LENSES[lens]
        elif isinstance(lens, LVISLens):
            self.lens = lens
        else:
            raise HyPlanTypeError(f"lens must be a LVISLens, str, or None, got {type(lens)}")

    # ------------------------------------------------------------------
    # Standard Sensor interface (used by swath.py, flight_box.py, glint.py)
    # ------------------------------------------------------------------

    @property
    def half_angle(self) -> float:
        """Half-scan angle in degrees (default ≈5.71 deg)."""
        return self._scan_half_angle_deg

    def swath_offset_angles(self) -> tuple:
        """Cross-track viewing angles for each swath edge (nadir-looking).

        Returns:
            Tuple of (port_edge_angle, starboard_edge_angle) in degrees.
        """
        return (-self.half_angle, self.half_angle)

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
        return 2 * altitude_agl * np.tan(np.radians(self.half_angle))  # type: ignore[return-value,no-any-return]

    # ------------------------------------------------------------------
    # LVIS-specific methods
    # ------------------------------------------------------------------

    def equivalent_fov(self, altitude_agl: Quantity, speed: Quantity) -> float:
        """Equivalent field of view in degrees after contiguous-coverage limits.

        When the laser footprint can fill the full scanner swath, this equals
        the geometric max FOV.  When coverage is sampling-limited, the
        equivalent FOV narrows: ``2 * atan(effective_swath / (2 * altitude))``.

        Args:
            altitude_agl: Flight altitude above ground level.
            speed: Aircraft ground speed.

        Returns:
            Equivalent FOV in degrees.
        """
        esw = self.effective_swath_width(altitude_agl, speed).magnitude
        alt_m = altitude_agl.m_as(ureg.meter)
        return 2 * np.degrees(np.arctan(esw / (2 * alt_m)))  # type: ignore[no-any-return]

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
        spd = speed.m_as(ureg.meter / ureg.second)
        ms = self.swath_width(altitude_agl).magnitude
        return spd * ms * ureg.meter ** 2 / ureg.second  # type: ignore[no-any-return]

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
        return np.sqrt(cr / rr) * ureg.meter  # type: ignore[no-any-return]

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
        spd = speed.m_as(ureg.meter / ureg.second)
        rr = self.rep_rate.magnitude

        contiguous_swath = fp ** 2 * rr / spd
        return min(ms, contiguous_swath) * ureg.meter  # type: ignore[no-any-return]

    def is_contiguous(self, altitude_agl: Quantity, speed: Quantity) -> bool:
        """Check whether the current configuration fills the max swath.

        Returns True if the footprint is large enough to tile the full
        scanner swath at the given speed.
        """
        ms = self.swath_width(altitude_agl).magnitude
        esw = self.effective_swath_width(altitude_agl, speed).magnitude
        return esw >= ms * self._CONTIGUITY_TOLERANCE  # type: ignore[no-any-return]

    # ------------------------------------------------------------------
    # Along-track sampling
    # ------------------------------------------------------------------

    def along_track_spacing(self, speed: Quantity) -> Quantity:
        """Along-track distance between consecutive laser shots.

        along_track_spacing = speed / rep_rate

        Args:
            speed: Aircraft ground speed.

        Returns:
            Shot spacing in meters.
        """
        spd = speed.m_as(ureg.meter / ureg.second)
        rr = self.rep_rate.magnitude
        return (spd / rr) * ureg.meter  # type: ignore[no-any-return]

    def is_along_track_contiguous(
        self, altitude_agl: Quantity, speed: Quantity
    ) -> bool:
        """Check whether consecutive shots overlap along-track.

        Returns True if the along-track spacing is less than or equal to
        the footprint diameter.
        """
        spacing = self.along_track_spacing(speed).magnitude
        fp = self.footprint_diameter(altitude_agl).magnitude
        return spacing <= fp * (1.0 + 1e-9)  # type: ignore[no-any-return]

    # ------------------------------------------------------------------
    # Point density
    # ------------------------------------------------------------------

    def point_density(self, altitude_agl: Quantity, speed: Quantity) -> Quantity:
        """Laser shot density within the effective swath.

        point_density = rep_rate / (speed * effective_swath)

        This is the primary planning metric for LVIS survey design.

        Args:
            altitude_agl: Flight altitude above ground level.
            speed: Aircraft ground speed.

        Returns:
            Point density in shots per square meter (1/m^2).
        """
        spd = speed.m_as(ureg.meter / ureg.second)
        esw = self.effective_swath_width(altitude_agl, speed).magnitude
        if esw < 1e-9:
            return 0.0 / ureg.meter ** 2  # type: ignore[no-any-return]
        rr = self.rep_rate.magnitude
        return (rr / (spd * esw)) / ureg.meter ** 2  # type: ignore[no-any-return]

    # ------------------------------------------------------------------
    # Survey solvers (inverse planning)
    # ------------------------------------------------------------------

    def solve_for_speed(
        self,
        target_density: Quantity,
        altitude_agl: Quantity,
    ) -> Quantity:
        """Compute the maximum aircraft speed for a target point density.

        Inverts ``point_density = rep_rate / (speed * effective_swath)``
        accounting for the coupling between speed and effective swath.

        Two regimes apply:

        * **Geometry-limited** (effective swath = max swath):
          ``density = rep_rate / (speed * max_swath)``, so
          ``speed = rep_rate / (density * max_swath)``.

        * **Sampling-limited** (effective swath < max swath):
          ``density = 1 / fp^2``, independent of speed.  In this regime
          the density is always at least the target (since 1/fp^2 >=
          target_density), so faster speeds are feasible — the effective
          swath narrows but density within it stays constant.

        The returned speed is the maximum speed at which
        ``point_density >= target_density``.

        Args:
            target_density: Desired point density (1/m^2).
            altitude_agl: Flight altitude above ground level.

        Returns:
            Maximum speed that achieves at least the target density.

        Raises:
            HyPlanValueError: If the target density exceeds 1/fp^2
                (impossible at this altitude/lens regardless of speed).
        """
        rr = self.rep_rate.magnitude
        ms = self.swath_width(altitude_agl).magnitude
        fp = self.footprint_diameter(altitude_agl).magnitude
        d_target = target_density.m_as(1 / ureg.meter ** 2)

        if d_target <= 0:
            raise HyPlanValueError("target_density must be positive")

        # In the sampling-limited regime, density = 1/fp^2.
        # If target exceeds that, it's physically impossible.
        if fp > 0 and (1.0 / fp ** 2) < d_target:
            raise HyPlanValueError(
                f"Target density {d_target:.4f} pts/m^2 exceeds the "
                f"maximum achievable density {1.0 / fp ** 2:.4f} pts/m^2 "
                f"at this altitude and lens.  Reduce altitude or use a "
                f"narrower lens."
            )

        # Geometry-limited: speed = rep_rate / (density * max_swath)
        speed_mps = rr / (d_target * ms)

        return speed_mps * ureg.meter / ureg.second  # type: ignore[no-any-return]

    def solve_for_altitude(
        self,
        target_density: Quantity,
        speed: Quantity,
    ) -> Quantity:
        """Compute the maximum altitude for a target point density.

        Inverts the density equation, accounting for the dependence of
        both footprint and max swath on altitude.

        In the geometry-limited regime (where effective swath = max swath):

            density = rep_rate / (speed * 2 * altitude * tan(half_angle))

        So:

            altitude = rep_rate / (speed * density * 2 * tan(half_angle))

        Args:
            target_density: Desired point density (1/m^2).
            speed: Aircraft ground speed.

        Returns:
            Maximum altitude AGL in meters.
        """
        rr = self.rep_rate.magnitude
        spd = speed.m_as(ureg.meter / ureg.second)
        d_target = target_density.m_as(1 / ureg.meter ** 2)

        if d_target <= 0:
            raise HyPlanValueError("target_density must be positive")

        tan_ha = np.tan(np.radians(self._scan_half_angle_deg))
        # Geometry-limited: alt = rr / (spd * d * 2 * tan(ha))
        alt_m = rr / (spd * d_target * 2 * tan_ha)

        # Check if we're actually geometry-limited at this altitude
        fp = np.tan(self.lens.divergence_mrad / 1000.0) * alt_m
        ms = 2 * alt_m * tan_ha
        contiguous_swath = fp ** 2 * rr / spd

        if contiguous_swath < ms:
            # Sampling-limited: density = 1/fp^2 = 1/(tan(div)*alt)^2
            # alt = 1 / (sqrt(d_target) * tan(div))
            tan_div = np.tan(self.lens.divergence_mrad / 1000.0)
            alt_m = 1.0 / (np.sqrt(d_target) * tan_div)

        return alt_m * ureg.meter  # type: ignore[no-any-return]

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
            "along_track_spacing": self.along_track_spacing(speed),
            "along_track_contiguous": self.is_along_track_contiguous(altitude_agl, speed),
            "point_density": self.point_density(altitude_agl, speed),
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
        print(f"  Cross-track contiguous:{s['contiguous']}")
        print(f"  Along-track spacing:   {s['along_track_spacing']:.2f}")
        print(f"  Along-track contiguous:{s['along_track_contiguous']}")
        print(f"  Point density:         {s['point_density']:.4f}")
        print(f"  Coverage rate:         {s['coverage_rate']:.0f}")
        print(f"  Footprint to fill max: {s['footprint_for_max_swath']:.1f}")

    def compare_lenses(self, altitude_agl: Quantity, speed: Quantity) -> None:
        """Print a comparison table across all standard lenses."""
        spd = speed.m_as(ureg.meter / ureg.second)
        rr = self.rep_rate.magnitude
        ms = self.swath_width(altitude_agl).magnitude
        alt_m = altitude_agl.m_as(ureg.meter)

        print(f"LVIS Lens Comparison at {altitude_agl.to(ureg.km):.1f} AGL, {speed.to(ureg.knot):.0f}")
        print(f"  Rep rate: {self.rep_rate:.0f}   Max swath: {ms:.0f} m")
        fmt_h = f"  {'Lens':10s}  {'Diverg':>8s}  {'Footprint':>10s}  {'Eff Swath':>10s}  {'Density':>12s}  {'Contiguous':>10s}"
        print(fmt_h)
        print(f"  {'-'*10}  {'-'*8}  {'-'*10}  {'-'*10}  {'-'*12}  {'-'*10}")

        for name, lens in LVIS_LENSES.items():
            fp = np.tan(lens.divergence_mrad / 1000.0) * alt_m
            contiguous = fp ** 2 * rr / spd
            swath = min(ms, contiguous)
            density = rr / (spd * swath) if swath > 0 else 0.0
            fills = "yes" if contiguous >= ms else "no"
            print(
                f"  {name:10s}  {lens.divergence_mrad:7.3f}   "
                f"{fp:9.1f} m  {swath:9.1f} m  "
                f"{density:10.4f}/m^2  {fills:>10s}"
            )

    # ------------------------------------------------------------------
    # Terrain-aware methods
    # ------------------------------------------------------------------

    def footprint_on_terrain(
        self,
        lat: float,
        lon: float,
        altitude_msl: float,
        heading: float,
        scan_angle_deg: float = 0.0,
        dem_file: Optional[str] = None,
    ) -> dict:
        """Compute laser footprint on terrain at a single scan position.

        Uses :func:`~hyplan.terrain.ray_terrain_intersection` to find the
        ground point, then computes slant range, surface incidence angle,
        and the resulting footprint ellipse.

        Args:
            lat: Aircraft latitude (degrees).
            lon: Aircraft longitude (degrees).
            altitude_msl: Aircraft altitude MSL (meters).
            heading: Aircraft heading (degrees true, clockwise from north).
            scan_angle_deg: Scan angle from nadir (degrees).
                Positive = starboard, negative = port.  Default 0 (nadir).
            dem_file: Path to DEM file. Auto-downloaded if *None*.

        Returns:
            Dict with ground position, slant range, incidence angle,
            footprint ellipse dimensions, and flat-earth comparison.
        """
        from ..terrain import ray_terrain_intersection, surface_normal_at, generate_demfile

        # Auto-generate DEM once for both ray intersection and normal lookup
        if dem_file is None:
            dem_file = generate_demfile(lat, lon)  # type: ignore[arg-type]

        # --- Ray direction ---
        if scan_angle_deg == 0.0:
            az = np.array([heading % 360.0])
            tilt = np.array([0.001])  # epsilon offset to avoid cos(tilt)=1 check
        elif scan_angle_deg > 0:
            az = np.array([(heading + 90.0) % 360.0])
            tilt = np.array([scan_angle_deg])
        else:
            az = np.array([(heading + 270.0) % 360.0])
            tilt = np.array([abs(scan_angle_deg)])

        lat0 = np.array([lat])
        lon0 = np.array([lon])

        gnd_lats, gnd_lons, gnd_alts = ray_terrain_intersection(
            lat0, lon0, altitude_msl, az=az, tilt=tilt, dem_file=dem_file,
        )

        gnd_lat = float(gnd_lats[0])
        gnd_lon = float(gnd_lons[0])
        gnd_alt = float(gnd_alts[0])

        if np.isnan(gnd_lat):
            return {
                "ground_lat": np.nan, "ground_lon": np.nan,
                "ground_alt_m": np.nan, "altitude_agl_m": np.nan,
                "slant_range_m": np.nan, "incidence_deg": np.nan,
                "scan_angle_deg": scan_angle_deg,
                "footprint_minor_m": np.nan, "footprint_major_m": np.nan,
                "footprint_area_m2": np.nan,
                "footprint_equivalent_diameter_m": np.nan,
                "flat_earth_diameter_m": np.nan,
            }

        # --- Slant range ---
        agl = altitude_msl - gnd_alt
        horiz_dist, _ = pymap3d.vincenty.vdist(lat, lon, gnd_lat, gnd_lon)
        horiz_dist = float(horiz_dist)
        slant_range = np.sqrt(horiz_dist ** 2 + agl ** 2)

        # --- Surface normal and incidence angle ---
        normals = surface_normal_at(
            np.array([gnd_lat]), np.array([gnd_lon]), dem_file,
        )
        normal = normals[0]  # (east, north, up)

        # LOS vector from ground to aircraft (ENU)
        if horiz_dist > 0.1:
            # vdist returns (distance, azimuth_fwd)
            _, az_fwd = pymap3d.vincenty.vdist(gnd_lat, gnd_lon, lat, lon)
            az_rad = np.radians(float(az_fwd))
            los_enu = np.array([
                horiz_dist * np.sin(az_rad),
                horiz_dist * np.cos(az_rad),
                agl,
            ])
        else:
            los_enu = np.array([0.0, 0.0, agl])

        los_unit = los_enu / np.linalg.norm(los_enu)
        cos_incidence = abs(np.dot(los_unit, normal))
        cos_incidence = min(cos_incidence, 1.0)
        incidence_deg = float(np.degrees(np.arccos(cos_incidence)))

        # --- Footprint ellipse ---
        div_rad = self.lens.divergence_mrad / 1000.0
        minor = np.tan(div_rad) * slant_range
        major = minor / max(cos_incidence, 1e-6)
        area = np.pi / 4.0 * major * minor
        equiv_diam = np.sqrt(major * minor)

        # Flat-earth comparison
        flat_diam = np.tan(div_rad) * agl

        return {
            "ground_lat": gnd_lat,
            "ground_lon": gnd_lon,
            "ground_alt_m": gnd_alt,
            "altitude_agl_m": agl,
            "slant_range_m": slant_range,
            "incidence_deg": incidence_deg,
            "scan_angle_deg": scan_angle_deg,
            "footprint_minor_m": minor,
            "footprint_major_m": major,
            "footprint_area_m2": area,
            "footprint_equivalent_diameter_m": equiv_diam,
            "flat_earth_diameter_m": flat_diam,
        }

    def effective_swath_on_terrain(
        self,
        lat: float,
        lon: float,
        altitude_msl: float,
        heading: float,
        speed: Quantity,
        dem_file: Optional[str] = None,
        n_scan_positions: int = 21,
    ) -> dict:
        """Compute effective swath across the scan, accounting for terrain.

        Discretises the scan into *n_scan_positions* angles from port to
        starboard and evaluates the terrain-aware footprint at each.
        Returns per-position metrics and the overall effective swath.

        Args:
            lat: Aircraft latitude (degrees).
            lon: Aircraft longitude (degrees).
            altitude_msl: Aircraft altitude MSL (meters).
            heading: Aircraft heading (degrees true).
            speed: Aircraft ground speed.
            dem_file: Path to DEM file. Auto-downloaded if *None*.
            n_scan_positions: Number of scan positions across the swath.

        Returns:
            Dict with per-position arrays and aggregate metrics.
        """
        from ..terrain import (
            ray_terrain_intersection,
            surface_normal_at,
            generate_demfile,
        )

        ha = self._scan_half_angle_deg
        scan_angles = np.linspace(-ha, ha, n_scan_positions)
        spd = speed.m_as(ureg.meter / ureg.second)
        rr = self.rep_rate.magnitude
        div_rad = self.lens.divergence_mrad / 1000.0

        # --- Batch ray-terrain intersection ---
        n = n_scan_positions
        lat0 = np.full(n, lat)
        lon0 = np.full(n, lon)

        azimuths = np.empty(n)
        tilts = np.empty(n)
        for i, sa in enumerate(scan_angles):
            if abs(sa) < 1e-6:
                azimuths[i] = heading % 360.0
                tilts[i] = 0.001
            elif sa > 0:
                azimuths[i] = (heading + 90.0) % 360.0
                tilts[i] = sa
            else:
                azimuths[i] = (heading + 270.0) % 360.0
                tilts[i] = abs(sa)

        # Auto-generate DEM if needed
        if dem_file is None:
            dem_file = generate_demfile(lat, lon)  # type: ignore[arg-type]

        gnd_lats, gnd_lons, gnd_alts = ray_terrain_intersection(
            lat0, lon0, altitude_msl, az=azimuths, tilt=tilts,
            dem_file=dem_file,
        )

        # --- Surface normals (batch) ---
        valid = ~np.isnan(gnd_lats)
        normals_all = np.zeros((n, 3))
        normals_all[:, 2] = 1.0  # default to vertical for invalid points
        if valid.any():
            normals_all[valid] = surface_normal_at(
                gnd_lats[valid], gnd_lons[valid], dem_file,
            )

        # --- Per-position computation ---
        agls = altitude_msl - gnd_alts
        slant_ranges = np.full(n, np.nan)
        incidences = np.full(n, np.nan)
        fp_diams = np.full(n, np.nan)

        for i in range(n):
            if not valid[i]:
                continue
            horiz_dist, az_fwd = pymap3d.vincenty.vdist(
                lat, lon, gnd_lats[i], gnd_lons[i],
            )
            horiz_dist = float(horiz_dist)
            sr = np.sqrt(horiz_dist ** 2 + agls[i] ** 2)
            slant_ranges[i] = sr

            # LOS unit vector (ENU)
            if horiz_dist > 0.1:
                az_rad = np.radians(float(az_fwd))
                # az_fwd is bearing FROM aircraft TO ground point
                # but we want LOS from ground to aircraft, so flip
                los = np.array([
                    -horiz_dist * np.sin(az_rad),
                    -horiz_dist * np.cos(az_rad),
                    agls[i],
                ])
            else:
                los = np.array([0.0, 0.0, agls[i]])
            los /= np.linalg.norm(los)

            cos_inc = min(abs(np.dot(los, normals_all[i])), 1.0)
            incidences[i] = np.degrees(np.arccos(cos_inc))

            minor = np.tan(div_rad) * sr
            major = minor / max(cos_inc, 1e-6)
            fp_diams[i] = np.sqrt(major * minor)

        # --- Cross-track spacings ---
        cross_spacings = np.full(n - 1, np.nan)
        for i in range(n - 1):
            if valid[i] and valid[i + 1]:
                d, _ = pymap3d.vincenty.vdist(
                    gnd_lats[i], gnd_lons[i],
                    gnd_lats[i + 1], gnd_lons[i + 1],
                )
                cross_spacings[i] = float(d)

        # --- Local spacing per position (average of neighbours) ---
        local_spacings = np.full(n, np.nan)
        for i in range(n):
            neighbours = []
            if i > 0 and not np.isnan(cross_spacings[i - 1]):
                neighbours.append(cross_spacings[i - 1])
            if i < n - 1 and not np.isnan(cross_spacings[i]):
                neighbours.append(cross_spacings[i])
            if neighbours:
                local_spacings[i] = np.mean(neighbours)

        # --- Per-position local density ---
        # density_i = rep_rate / (speed * local_spacing_i)
        local_densities = np.full(n, np.nan)
        density_valid = np.isfinite(local_spacings) & (local_spacings > 0)
        local_densities[density_valid] = rr / (spd * local_spacings[density_valid])

        # --- Contiguity check per position ---
        # Two independent conditions:
        #   cross-track:  fp^2 * rr / speed >= local_spacing
        #                 (can the shot rate tile the local cross-track strip?)
        #   along-track:  speed / rr <= footprint
        #                 (do consecutive shots overlap along the flight line?)
        along_track_gap = spd / rr  # meters between consecutive shots
        contiguous_cross = np.zeros(n, dtype=bool)
        contiguous_along = np.zeros(n, dtype=bool)
        contiguous = np.zeros(n, dtype=bool)
        for i in range(n):
            if np.isnan(fp_diams[i]):
                continue
            contiguous_along[i] = along_track_gap <= fp_diams[i] * (1.0 + 1e-9)
            if not np.isnan(local_spacings[i]):
                contiguous_cross[i] = (
                    fp_diams[i] ** 2 * rr / spd
                ) >= local_spacings[i] * (1.0 - 1e-9)
                contiguous[i] = contiguous_cross[i] and contiguous_along[i]

        # --- Effective swath: largest contiguous block ---
        effective_swath_m = 0.0
        best_start, best_end = 0, 0
        start = None
        for i in range(n):
            if contiguous[i]:
                if start is None:
                    start = i
            else:
                if start is not None:
                    if valid[start] and valid[i - 1]:
                        span, _ = pymap3d.vincenty.vdist(
                            gnd_lats[start], gnd_lons[start],
                            gnd_lats[i - 1], gnd_lons[i - 1],
                        )
                        if float(span) > effective_swath_m:
                            effective_swath_m = float(span)
                            best_start, best_end = start, i - 1
                    start = None
        if start is not None and valid[start] and valid[n - 1]:
            span, _ = pymap3d.vincenty.vdist(
                gnd_lats[start], gnd_lons[start],
                gnd_lats[n - 1], gnd_lons[n - 1],
            )
            if float(span) > effective_swath_m:
                effective_swath_m = float(span)
                best_start, best_end = start, n - 1

        # --- Density statistics ---
        valid_densities = local_densities[np.isfinite(local_densities)]
        if len(valid_densities) > 0:
            density_min = float(valid_densities.min())
            density_max = float(valid_densities.max())
            density_mean = float(valid_densities.mean())
            density_std = float(valid_densities.std())
        else:
            density_min = density_max = density_mean = density_std = 0.0

        # --- Flat-earth comparison ---
        flat_agl = altitude_msl - np.nanmean(gnd_alts) if valid.any() else 0.0
        flat_esw = self.effective_swath_width(
            flat_agl * ureg.meter, speed
        ).magnitude

        return {
            "scan_angles_deg": scan_angles,
            "ground_lats": gnd_lats,
            "ground_lons": gnd_lons,
            "slant_ranges_m": slant_ranges,
            "incidence_angles_deg": incidences,
            "footprint_diameters_m": fp_diams,
            "cross_track_spacings_m": cross_spacings,
            "local_densities": local_densities,
            "contiguous_cross_track": contiguous_cross,
            "contiguous_along_track": contiguous_along,
            "contiguous_mask": contiguous,
            "effective_swath_m": effective_swath_m,
            "density_min": density_min,
            "density_max": density_max,
            "density_mean": density_mean,
            "density_std": density_std,
            "flat_earth_effective_swath_m": flat_esw,
        }

    def terrain_summary(
        self,
        lat: float,
        lon: float,
        altitude_msl: float,
        heading: float,
        speed: Quantity,
        dem_file: Optional[str] = None,
    ) -> dict:
        """Coverage summary with terrain correction at a specific position.

        Combines the flat-earth :meth:`summary` output with terrain-aware
        metrics from :meth:`effective_swath_on_terrain`.

        Args:
            lat: Aircraft latitude (degrees).
            lon: Aircraft longitude (degrees).
            altitude_msl: Aircraft altitude MSL (meters).
            heading: Aircraft heading (degrees true).
            speed: Aircraft ground speed.
            dem_file: Path to DEM file. Auto-downloaded if *None*.

        Returns:
            Dict with all keys from :meth:`summary` plus terrain-specific
            keys prefixed with ``terrain_``.
        """
        from ..terrain import get_elevations, generate_demfile

        if dem_file is None:
            dem_file = generate_demfile(lat, lon)  # type: ignore[arg-type]

        # Ground elevation at nadir
        gnd_elev = float(
            get_elevations(np.array([lat]), np.array([lon]), dem_file)[0]
        )
        agl = altitude_msl - gnd_elev

        # Flat-earth summary using estimated AGL
        flat = self.summary(agl * ureg.meter, speed)

        # Terrain-aware cross-track analysis
        terrain = self.effective_swath_on_terrain(
            lat, lon, altitude_msl, heading, speed, dem_file=dem_file,
        )

        # Nadir footprint on terrain
        nadir = self.footprint_on_terrain(
            lat, lon, altitude_msl, heading,
            scan_angle_deg=0.0, dem_file=dem_file,
        )

        flat.update({
            "terrain_ground_elevation_m": gnd_elev,
            "terrain_altitude_agl_m": agl,
            "terrain_nadir_incidence_deg": nadir["incidence_deg"],
            "terrain_nadir_footprint_major_m": nadir["footprint_major_m"],
            "terrain_nadir_footprint_minor_m": nadir["footprint_minor_m"],
            "terrain_effective_swath_m": terrain["effective_swath_m"],
            "terrain_density_min": terrain["density_min"],
            "terrain_density_max": terrain["density_max"],
            "terrain_density_mean": terrain["density_mean"],
            "terrain_density_std": terrain["density_std"],
            "terrain_contiguous_fraction": (
                float(terrain["contiguous_mask"].sum()) / len(terrain["contiguous_mask"])
            ),
        })
        return flat
