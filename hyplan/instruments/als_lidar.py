"""Airborne Laser Scanner (ALS) — scanning-mirror discrete-return topographic lidar.

Generic ``ALSLidar`` class for rotating-mirror discrete-return topographic
lidar instruments (RIEGL VQ-series, Leica TerrainMapper, Optech Galaxy,
Phoenix LiDAR Ranger, etc.).  Adds the missing third lidar abstraction
to HyPlan alongside :class:`hyplan.instruments.LVIS` (full-waveform,
conical-scan) and :class:`hyplan.instruments.ProfilingLidar` (single-beam
profilers — HSRL-2, HALO, CPL).

*Specific trade names (e.g. Riegl) are for informational purposes only and do not constitute an endorsement by NASA.

Pre-configured reference instance: :data:`RIEGL_VQ_480II` (RIEGL VQ-480 II,
configured at 1200 kHz PRR — the high-density operating point).  All
parameters are sourced from RIEGL's publicly published datasheet
(2024-08-23), cited in the instance's ``source`` field.  Users can
construct other PRR operating points by passing different ``prf`` /
``max_range`` / ``mta_zones`` arguments to ``ALSLidar(...)`` directly.

Out of scope (each has different physics or statistics):

* **Full-waveform lidars** — see :class:`hyplan.instruments.LVIS`.
* **Photon-counting / Geiger-mode** (NASA ATM, ATLAS, MABEL, Leica SPL100).
* **Bathymetric dual-wavelength** (RIEGL VQ-880-G, Leica HawkEye).
* **Multispectral** (Optech Titan).
* **Oscillating-mirror scanners** (sinusoidal angular velocity) —
  out of scope for v1; the constructor raises ``ValueError`` for these.

References
----------
RIEGL VQ-480 II Data Sheet, 2024-08-23.  Public datasheet:
https://www.riegl.com/fileadmin/media/Products/03_Airborne_Scanning/RIEGL_VQ-480_II/RIEGL_VQ-480II_Datasheet_2024-08-23.pdf
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pymap3d.vincenty
from pint import Quantity

from ..exceptions import HyPlanTypeError, HyPlanValueError
from ..units import ureg
from ._base import Sensor

__all__ = [
    "ALSLidar",
    "ContiguityError",
    "GLIHT_DUAL_VQ_480I",
    "LidarMount",
    "MultiALSLidarRig",
    "RIEGL_VQ_480II",
    "SPEED_OF_LIGHT_M_PER_S",
]


SPEED_OF_LIGHT_M_PER_S = 299_792_458.0


ScanGeometry = Literal[
    "rotating_polygon_active_arc",
    "rotating_polygon_full_circle",
    "oscillating_mirror",
]
"""Scanner kinematic convention.

* ``"rotating_polygon_active_arc"``: constant angular velocity, pulses
  only fired while the mirror facet is within the active arc
  ±scan_half_angle (RIEGL VQ-series convention).
* ``"rotating_polygon_full_circle"``: constant angular velocity, pulses
  fired continuously around the polygon facet's full rotation; only
  those within ±scan_half_angle reach the ground.
* ``"oscillating_mirror"``: sinusoidal angular velocity (Leica
  TerrainMapper, Optech Galaxy on some configurations).  Pulse spacing
  varies across the scan and is non-uniform; the v1 module does not
  model this case and the constructor raises ``ValueError``.
"""

_VALID_SCAN_GEOMETRIES = (
    "rotating_polygon_active_arc",
    "rotating_polygon_full_circle",
    "oscillating_mirror",
)


class ContiguityError(HyPlanValueError):
    """Raised by ALS solvers when the requested target density would
    require a flight configuration that leaves un-imaged strips between
    consecutive scan lines.

    The nominal density formula
    ``density = prf / (groundspeed * swath_width)`` always has a
    solution, but the solution may be physically degenerate when
    ``groundspeed / scan_rate`` exceeds the nadir laser footprint.
    Solvers raise this error by default; pass ``strict_contiguity=False``
    to receive the nominal-density solution anyway.
    """


def _as_quantity(value: object, unit: str, label: str) -> Quantity:
    if isinstance(value, Quantity):
        return value.to(unit)
    if isinstance(value, (int, float)):
        return ureg.Quantity(float(value), unit)
    raise HyPlanTypeError(
        f"{label} must be numeric or a pint.Quantity, got {type(value)}"
    )


class ALSLidar(Sensor):
    """Generic Airborne Laser Scanner — scanning-mirror discrete-return
    topographic lidar.

    Models rotating-polygon scanners with a fixed-rate pulsed laser
    sweeping linear scan lines perpendicular to the aircraft track.
    Conforms to the :class:`hyplan.instruments.ScanningSensor` Protocol so
    instances plug directly into
    :func:`hyplan.swath.generate_swath_polygon` and
    :func:`hyplan.flight_box.box_around_polygon`.

    Parameters
    ----------
    name : str
        Instrument display name (e.g. ``"RIEGL VQ-480 II"``).
    prf : Quantity
        Laser pulse repetition rate (Hz).
    scan_rate : Quantity
        Scan-line rate (Hz) — number of cross-track scan lines per second.
    scan_half_angle : Quantity
        Half scan angle from nadir (degrees).  Total scan FOV =
        ``2 * scan_half_angle``.
    beam_divergence : Quantity
        Full-angle laser beam divergence (radians or mrad).
    wavelength : Quantity
        Laser wavelength (nm — informational only).
    max_range : Quantity
        Maximum usable slant range (m) at the reference target
        reflectivity.  This is the *radiometric* envelope.
    max_range_reflectivity : float, default 0.6
        Target reflectivity (0–1) at which ``max_range`` is quoted.
    mta_zones : int, default 1
        Number of multiple-time-of-arrival processing zones.  Defines the
        timing envelope: ``mta_max_unambiguous_range = mta_zones *
        c / (2 * prf)``.  HyPlan reports the envelope; the actual
        disambiguation algorithm is the vendor's job.
    scan_geometry : ScanGeometry, default "rotating_polygon_active_arc"
        Scanner kinematic convention — see :data:`ScanGeometry`.
        Determines the angular pulse spacing formula used by
        :meth:`cross_track_spacing_at_nadir` and
        :meth:`cross_track_spacing_at_angle`.  Constructing with
        ``"oscillating_mirror"`` raises :class:`HyPlanValueError`.
    source : str, default ""
        Provenance: datasheet citation including URL and retrieval date.
    """

    def __init__(
        self,
        name: str,
        *,
        prf: Quantity,
        scan_rate: Quantity,
        scan_half_angle: Quantity,
        beam_divergence: Quantity,
        wavelength: Quantity,
        max_range: Quantity,
        max_range_reflectivity: float = 0.6,
        mta_zones: int = 1,
        scan_geometry: ScanGeometry = "rotating_polygon_active_arc",
        source: str = "",
    ):
        super().__init__(name=name)

        self.prf = _as_quantity(prf, "hertz", "prf")
        self.scan_rate = _as_quantity(scan_rate, "hertz", "scan_rate")
        self.scan_half_angle = _as_quantity(
            scan_half_angle, "degree", "scan_half_angle"
        )
        self.beam_divergence = _as_quantity(
            beam_divergence, "radian", "beam_divergence"
        )
        self.wavelength = _as_quantity(wavelength, "nanometer", "wavelength")
        self.max_range = _as_quantity(max_range, "meter", "max_range")

        for label, q in (
            ("prf", self.prf),
            ("scan_rate", self.scan_rate),
            ("scan_half_angle", self.scan_half_angle),
            ("beam_divergence", self.beam_divergence),
            ("wavelength", self.wavelength),
            ("max_range", self.max_range),
        ):
            if q.magnitude <= 0:
                raise HyPlanValueError(f"{label} must be positive")

        if self.scan_half_angle.magnitude >= 90:
            raise HyPlanValueError(
                "scan_half_angle must be less than 90 degrees"
            )

        if not (0.0 < float(max_range_reflectivity) <= 1.0):
            raise HyPlanValueError(
                "max_range_reflectivity must be in (0, 1]"
            )
        self.max_range_reflectivity = float(max_range_reflectivity)

        if not isinstance(mta_zones, int) or mta_zones < 1:
            raise HyPlanValueError("mta_zones must be a positive integer")
        self.mta_zones = mta_zones

        if scan_geometry not in _VALID_SCAN_GEOMETRIES:
            raise HyPlanValueError(
                f"scan_geometry must be one of {_VALID_SCAN_GEOMETRIES}, "
                f"got {scan_geometry!r}"
            )
        if scan_geometry == "oscillating_mirror":
            raise HyPlanValueError(
                "oscillating_mirror scanners have non-uniform pulse "
                "spacing across the scan and are not modelled in v1; "
                "see notes in hyplan/instruments/als_lidar.py"
            )
        self.scan_geometry: ScanGeometry = scan_geometry

        self.source = str(source)

    # ------------------------------------------------------------------
    # ScanningSensor Protocol surface
    # ------------------------------------------------------------------

    @property
    def half_angle(self) -> float:
        """Half scan angle in degrees — satisfies ScanningSensor."""
        return float(self.scan_half_angle.m_as("degree"))

    def swath_offset_angles(self) -> tuple[float, float]:
        """Cross-track edge angles ``(port, starboard)`` in degrees.

        Negative = port (left of track), positive = starboard.
        """
        ha = self.half_angle
        return (-ha, ha)

    def swath_width(self, altitude_agl: Quantity) -> Quantity:
        """Total cross-track swath width on flat ground at ``altitude_agl``.

        ``swath = 2 * altitude_agl * tan(scan_half_angle)``
        """
        alt = self._validate_quantity(altitude_agl, ureg.meter)
        ha_rad = self.scan_half_angle.m_as("radian")
        return 2.0 * alt * np.tan(ha_rad)

    # ------------------------------------------------------------------
    # Footprint
    # ------------------------------------------------------------------

    def footprint_diameter(
        self,
        altitude_agl: Quantity,
        scan_angle: Quantity | None = None,
    ) -> Quantity:
        """Laser spot diameter on flat ground.

        At nadir: ``footprint = altitude_agl * beam_divergence`` (small-
        angle approximation).  At off-nadir scan angle θ, the slant range
        grows by ``1 / cos(θ)``, so the along-scan-line diameter is
        ``footprint_nadir / cos(θ)``.  Returns the along-track diameter
        of the elliptical ground footprint.
        """
        alt = self._validate_quantity(altitude_agl, ureg.meter)
        div_rad = self.beam_divergence.m_as("radian")
        nadir = alt * div_rad
        if scan_angle is None:
            return nadir
        theta = _as_quantity(scan_angle, "radian", "scan_angle")
        return nadir / np.cos(theta.magnitude)

    # ------------------------------------------------------------------
    # Density and spacing
    # ------------------------------------------------------------------

    def point_density(
        self,
        altitude_agl: Quantity,
        groundspeed: Quantity,
        *,
        effective_prf: Quantity | None = None,
    ) -> Quantity:
        """Nominal areal point density over the full swath rectangle.

        ``density_nominal = effective_prf / (groundspeed * swath_width)``

        Semantics (locked): this is the *nominal* density — total pulses
        per unit time divided by the nominal swath-rectangle area per
        unit time.  It does not model along-track gaps; if scan lines
        do not overlap, the actual instantaneously-illuminated ground
        points concentrate in narrower strips.  We deliberately report
        the nominal value so the formula stays a single closed-form
        expression suitable for both forward calculation and inverse
        solving.

        Pass ``effective_prf`` to override ``self.prf`` (e.g. for
        derated MTA operation).

        Survey planners MUST call :meth:`is_along_track_contiguous`
        separately to verify the configuration produces contiguous
        ground coverage.
        """
        alt = self._validate_quantity(altitude_agl, ureg.meter)
        spd = self._validate_quantity(groundspeed, ureg.meter / ureg.second)
        prf = (
            self._validate_quantity(effective_prf, ureg.hertz)
            if effective_prf is not None
            else self.prf
        )
        sw = self.swath_width(alt)
        return (prf / (spd * sw)).to(1 / ureg.meter**2)

    def along_track_spacing(self, groundspeed: Quantity) -> Quantity:
        """Ground distance between adjacent scan lines.

        ``along_track = groundspeed / scan_rate``

        Purely kinematic — cannot be closed by altering altitude or scan
        angle.  See :meth:`is_along_track_contiguous`.
        """
        spd = self._validate_quantity(groundspeed, ureg.meter / ureg.second)
        return (spd / self.scan_rate).to(ureg.meter)

    def _angular_pulse_step_rad(self) -> float:
        """Angular spacing between consecutive pulses (radians)."""
        prf_hz = float(self.prf.m_as("hertz"))
        sr_hz = float(self.scan_rate.m_as("hertz"))
        if self.scan_geometry == "rotating_polygon_active_arc":
            arc_rad = 2.0 * float(self.scan_half_angle.m_as("radian"))
            return arc_rad * sr_hz / prf_hz
        if self.scan_geometry == "rotating_polygon_full_circle":
            return 2.0 * float(np.pi) * sr_hz / prf_hz
        raise HyPlanValueError(
            f"angular step undefined for scan_geometry={self.scan_geometry!r}"
        )

    def cross_track_spacing_at_nadir(self, altitude_agl: Quantity) -> Quantity:
        """Distance between adjacent pulses within one scan line at nadir.

        For ``"rotating_polygon_active_arc"``:
        ``dθ = 2 * scan_half_angle * scan_rate / prf``.
        For ``"rotating_polygon_full_circle"``:
        ``dθ = 2π * scan_rate / prf``.

        Ground spacing at nadir is then ``altitude_agl * dθ``.
        """
        alt = self._validate_quantity(altitude_agl, ureg.meter)
        return alt * self._angular_pulse_step_rad()

    def cross_track_spacing_at_angle(
        self, altitude_agl: Quantity, scan_angle: Quantity,
    ) -> Quantity:
        """Ground pulse spacing at off-nadir scan angle θ.

        Two effects combine for a constant-angular-velocity scanner:

        1. Slant range grows by ``1 / cos(θ)``, stretching ``altitude *
           dθ`` by ``1 / cos(θ)``.
        2. Projection of the slant-range arc onto the ground adds
           another ``1 / cos(θ)``.

        Net: ``spacing(θ) = (altitude_agl * dθ) / cos²(θ)``.
        """
        nadir = self.cross_track_spacing_at_nadir(altitude_agl)
        theta = _as_quantity(scan_angle, "radian", "scan_angle")
        cos_t = float(np.cos(theta.magnitude))
        if cos_t <= 0:
            raise HyPlanValueError(
                "scan_angle must satisfy |θ| < 90° for cross-track spacing"
            )
        return nadir / (cos_t * cos_t)

    # ------------------------------------------------------------------
    # Coverage diagnostics
    # ------------------------------------------------------------------

    def coverage_rate(
        self, altitude_agl: Quantity, groundspeed: Quantity,
    ) -> Quantity:
        """Nominal swath area covered per unit time (m²/s).

        ``coverage_rate = groundspeed * swath_width(altitude_agl)``

        Geometric coverage rate.  Does not derate for along-track gaps —
        check :meth:`is_along_track_contiguous` separately.
        """
        spd = self._validate_quantity(groundspeed, ureg.meter / ureg.second)
        sw = self.swath_width(altitude_agl)
        return (spd * sw).to(ureg.meter**2 / ureg.second)

    def is_along_track_contiguous(
        self, altitude_agl: Quantity, groundspeed: Quantity,
    ) -> bool:
        """True iff consecutive scan lines on the ground overlap.

        ``along_track_gap = groundspeed / scan_rate``
        ``nadir_footprint = altitude_agl * beam_divergence``
        contiguous iff ``along_track_gap <= nadir_footprint``.

        Strict planning constraint: if False, the flight line leaves
        un-imaged strips and no amount of cross-track line overlap will
        fill them.
        """
        gap = self.along_track_spacing(groundspeed).m_as("meter")
        fp = self.footprint_diameter(altitude_agl).m_as("meter")
        return bool(gap <= fp * (1.0 + 1e-9))

    def is_cross_track_contiguous(
        self, altitude_agl: Quantity, line_spacing: Quantity,
    ) -> bool:
        """True iff adjacent flight lines spaced ``line_spacing`` apart
        produce overlapping swaths on the ground.

        ``contiguous := line_spacing <= swath_width(altitude_agl)``
        """
        ls = self._validate_quantity(line_spacing, ureg.meter)
        sw = self.swath_width(altitude_agl)
        return bool(ls.magnitude <= sw.m_as("meter") * (1.0 + 1e-9))

    def coverage_diagnostic(
        self, altitude_agl: Quantity, groundspeed: Quantity,
    ) -> dict[str, float | bool]:
        """Per-dimension coverage breakdown for one flight line.

        Returns a dict with keys (all values float or bool):

        * ``along_track_gap_m`` — scan-line ground gap
        * ``along_track_footprint_m`` — nadir laser footprint
        * ``along_track_contiguous`` — bool
        * ``swath_width_m`` — geometric swath
        * ``nadir_density_pts_m2`` — peak density at nadir
        * ``edge_density_pts_m2`` — density at ±scan_half_angle
        * ``swath_mean_density_pts_m2`` — :meth:`point_density` value

        This is the diagnostic surface for survey planners.  Along- and
        cross-track coverage are independent and reported separately.
        """
        alt = self._validate_quantity(altitude_agl, ureg.meter)
        spd = self._validate_quantity(groundspeed, ureg.meter / ureg.second)
        gap = self.along_track_spacing(spd).m_as("meter")
        fp_nadir = self.footprint_diameter(alt).m_as("meter")
        sw = self.swath_width(alt).m_as("meter")
        mean_dens = self.point_density(alt, spd).m_as(1 / ureg.meter**2)
        spacing_nadir = self.cross_track_spacing_at_nadir(alt).m_as("meter")
        spacing_edge = self.cross_track_spacing_at_angle(
            alt, self.scan_half_angle
        ).m_as("meter")
        density_at_nadir = (
            1.0 / (gap * spacing_nadir) if (gap > 0 and spacing_nadir > 0) else 0.0
        )
        density_at_edge = (
            1.0 / (gap * spacing_edge) if (gap > 0 and spacing_edge > 0) else 0.0
        )
        return {
            "along_track_gap_m": float(gap),
            "along_track_footprint_m": float(fp_nadir),
            "along_track_contiguous": bool(gap <= fp_nadir * (1.0 + 1e-9)),
            "swath_width_m": float(sw),
            "nadir_density_pts_m2": float(density_at_nadir),
            "edge_density_pts_m2": float(density_at_edge),
            "swath_mean_density_pts_m2": float(mean_dens),
        }

    # ------------------------------------------------------------------
    # Inverse solvers
    # ------------------------------------------------------------------

    def solve_for_altitude(
        self,
        target_density: Quantity,
        groundspeed: Quantity,
        *,
        strict_contiguity: bool = True,
    ) -> Quantity:
        """Altitude AGL (m) at which nominal density equals
        ``target_density`` at the given ``groundspeed``.

        Solves the closed-form
        ``altitude = prf / (target_density * groundspeed * 2 *
        tan(scan_half_angle))``.

        If ``strict_contiguity`` (default), raises
        :class:`ContiguityError` when the solved altitude would leave
        un-imaged strips between scan lines.
        """
        d = self._validate_quantity(target_density, 1 / ureg.meter**2)
        spd = self._validate_quantity(groundspeed, ureg.meter / ureg.second)
        if d.magnitude <= 0:
            raise HyPlanValueError("target_density must be positive")
        if spd.magnitude <= 0:
            raise HyPlanValueError("groundspeed must be positive")

        prf_hz = self.prf.m_as("hertz")
        d_per_m2 = d.m_as(1 / ureg.meter**2)
        spd_mps = spd.m_as("meter / second")
        tan_ha = float(np.tan(self.scan_half_angle.m_as("radian")))

        alt_m = prf_hz / (d_per_m2 * spd_mps * 2.0 * tan_ha)
        alt = alt_m * ureg.meter

        if strict_contiguity and not self.is_along_track_contiguous(alt, spd):
            gap = self.along_track_spacing(spd).m_as("meter")
            fp = self.footprint_diameter(alt).m_as("meter")
            raise ContiguityError(
                f"Solved altitude {alt_m:.0f} m at groundspeed "
                f"{spd_mps:.1f} m/s yields the target density nominally "
                f"but along-track gap {gap:.2f} m exceeds nadir footprint "
                f"{fp:.2f} m — scan lines do not overlap.  Slow the "
                f"aircraft, raise the altitude (larger footprint), or "
                f"pass strict_contiguity=False to accept the gap."
            )
        return alt

    def solve_for_groundspeed(
        self,
        target_density: Quantity,
        altitude_agl: Quantity,
        *,
        strict_contiguity: bool = True,
    ) -> Quantity:
        """Groundspeed (m/s) at which nominal density equals
        ``target_density`` at the given ``altitude_agl``.  Same
        contiguity guard as :meth:`solve_for_altitude`.
        """
        d = self._validate_quantity(target_density, 1 / ureg.meter**2)
        alt = self._validate_quantity(altitude_agl, ureg.meter)
        if d.magnitude <= 0:
            raise HyPlanValueError("target_density must be positive")
        if alt.magnitude <= 0:
            raise HyPlanValueError("altitude_agl must be positive")

        prf_hz = self.prf.m_as("hertz")
        d_per_m2 = d.m_as(1 / ureg.meter**2)
        sw_m = self.swath_width(alt).m_as("meter")

        spd_mps = prf_hz / (d_per_m2 * sw_m)
        spd = spd_mps * ureg.meter / ureg.second

        if strict_contiguity and not self.is_along_track_contiguous(alt, spd):
            gap = self.along_track_spacing(spd).m_as("meter")
            fp = self.footprint_diameter(alt).m_as("meter")
            raise ContiguityError(
                f"Solved groundspeed {spd_mps:.1f} m/s at altitude "
                f"{alt.m_as('meter'):.0f} m yields the target density "
                f"nominally but along-track gap {gap:.2f} m exceeds nadir "
                f"footprint {fp:.2f} m — scan lines do not overlap.  "
                f"Increase scan_rate, lower altitude, or pass "
                f"strict_contiguity=False."
            )
        return spd

    def required_overlap_percent(
        self,
        altitude_agl: Quantity,
        target_density: Quantity | None = None,
        groundspeed: Quantity | None = None,
        *,
        default_overlap_percent: float = 20.0,
    ) -> float:
        """Adjacent-line overlap required for contiguous / target-density
        coverage, expressed as a percent in [0, 100).

        Matches HyPlan's existing overlap convention used by
        :func:`hyplan.flight_box.box_around_polygon` (0–100 percent).

        Returns ``default_overlap_percent`` (default 20%) when
        ``target_density`` is not supplied — a small non-zero pad
        absorbs crab-angle / lateral track-keeping error and matches
        typical operational practice.

        When ``target_density`` and ``groundspeed`` are both supplied,
        returns the overlap that makes the *combined* density across
        the line boundary equal ``target_density``: the overlap zone
        is sampled by both adjacent passes, so spacing is chosen such
        that the edge density rises to target after summation.
        """
        if target_density is None or groundspeed is None:
            return float(default_overlap_percent)
        d = self._validate_quantity(target_density, 1 / ureg.meter**2)
        spd = self._validate_quantity(groundspeed, ureg.meter / ureg.second)
        if d.magnitude <= 0:
            raise HyPlanValueError("target_density must be positive")
        if spd.magnitude <= 0:
            raise HyPlanValueError("groundspeed must be positive")
        mean_density = self.point_density(altitude_agl, spd).m_as(
            1 / ureg.meter**2
        )
        d_target = d.m_as(1 / ureg.meter**2)
        if mean_density >= d_target:
            return float(default_overlap_percent)
        # density doubles in the overlap zone; spacing must contract so
        # the non-overlapping centre still meets target.  Fraction of
        # swath needed at target density = d_target / mean_density;
        # overlap = 1 - swath_used / swath_total = 1 - mean/target.
        overlap_fraction = 1.0 - mean_density / d_target
        return float(min(99.0, max(default_overlap_percent, overlap_fraction * 100.0)))

    # ------------------------------------------------------------------
    # MTA timing envelope
    # ------------------------------------------------------------------

    def mta_max_unambiguous_range(self) -> Quantity:
        """Maximum unambiguous slant range from MTA timing alone (m).

        ``max_unambiguous = mta_zones * c / (2 * prf)``

        Out of scope: vendor-proprietary disambiguation algorithms.
        HyPlan reports the envelope.
        """
        prf_hz = self.prf.m_as("hertz")
        unambig = self.mta_zones * SPEED_OF_LIGHT_M_PER_S / (2.0 * prf_hz)
        return unambig * ureg.meter

    def mta_practical_max_altitude(self) -> Quantity:
        """Practical AGL planning ceiling (m).

        = min(mta_max_unambiguous_range, max_range)

        The MTA formula gives the timing envelope; ``max_range`` gives
        the radiometric envelope.  Both must hold.  Assumes nadir
        geometry — off-nadir operation reduces the ceiling further by
        ``cos(scan_half_angle)``; the v1 module leaves that adjustment
        to the caller.
        """
        unambig_m = self.mta_max_unambiguous_range().m_as("meter")
        radio_m = self.max_range.m_as("meter")
        return min(unambig_m, radio_m) * ureg.meter

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(
        self, altitude_agl: Quantity, groundspeed: Quantity,
    ) -> dict[str, object]:
        """Full configuration + coverage summary for an operating point."""
        diag = self.coverage_diagnostic(altitude_agl, groundspeed)
        return {
            "name": self.name,
            "prf": self.prf,
            "scan_rate": self.scan_rate,
            "scan_half_angle": self.scan_half_angle,
            "beam_divergence_mrad": float(
                self.beam_divergence.m_as("milliradian")
            ),
            "wavelength_nm": float(self.wavelength.m_as("nanometer")),
            "scan_geometry": self.scan_geometry,
            "mta_zones": self.mta_zones,
            "mta_max_unambiguous_range": self.mta_max_unambiguous_range(),
            "mta_practical_max_altitude": self.mta_practical_max_altitude(),
            "altitude_agl": self._validate_quantity(
                altitude_agl, ureg.meter
            ).to(ureg.meter),
            "groundspeed": self._validate_quantity(
                groundspeed, ureg.meter / ureg.second
            ).to(ureg.knot),
            "swath_width_m": diag["swath_width_m"],
            "along_track_gap_m": diag["along_track_gap_m"],
            "along_track_footprint_m": diag["along_track_footprint_m"],
            "along_track_contiguous": diag["along_track_contiguous"],
            "nadir_density_pts_m2": diag["nadir_density_pts_m2"],
            "edge_density_pts_m2": diag["edge_density_pts_m2"],
            "swath_mean_density_pts_m2": diag["swath_mean_density_pts_m2"],
        }


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
        dem_file: str | None = None,
    ) -> dict[str, Any]:
        """Laser footprint on terrain at a single scan position.

        Uses :func:`hyplan.terrain.ray_terrain_intersection` to find the
        ground point, then computes slant range, surface incidence angle,
        and the elliptical footprint on the sloped surface.

        Footprint ellipse axes:

        * ``minor`` (along the laser axis, perpendicular to the LOS) =
          ``tan(beam_divergence) * slant_range``.
        * ``major`` (along the surface tangent in the laser plane) =
          ``minor / cos(incidence)`` — stretched by the surface
          incidence angle.

        Args:
            lat: Aircraft latitude (degrees).
            lon: Aircraft longitude (degrees).
            altitude_msl: Aircraft altitude MSL (meters).
            heading: Aircraft heading (degrees true, clockwise from
                north).  For the boresight-along-track case, pass the
                ground track azimuth; for crab-aware planning, pass the
                aircraft heading (which differs from the track).
            scan_angle_deg: Scan angle from nadir (degrees).  Positive =
                starboard, negative = port.  Default 0 (nadir).
            dem_file: Path to DEM file. Auto-downloaded if *None*.

        Returns:
            Dict with ground position, slant range, incidence angle,
            footprint ellipse dimensions, and flat-earth comparison.
        """
        from ..terrain import (
            generate_demfile,
            ray_terrain_intersection,
            surface_normal_at,
        )

        if dem_file is None:
            dem_file = generate_demfile(lat, lon)  # type: ignore[arg-type]

        if scan_angle_deg == 0.0:
            az = np.array([heading % 360.0])
            tilt = np.array([0.001])  # epsilon to keep ray-intersection numerics happy
        elif scan_angle_deg > 0:
            az = np.array([(heading + 90.0) % 360.0])
            tilt = np.array([float(scan_angle_deg)])
        else:
            az = np.array([(heading + 270.0) % 360.0])
            tilt = np.array([abs(float(scan_angle_deg))])

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
                "scan_angle_deg": float(scan_angle_deg),
                "footprint_minor_m": np.nan, "footprint_major_m": np.nan,
                "footprint_area_m2": np.nan,
                "footprint_equivalent_diameter_m": np.nan,
                "flat_earth_diameter_m": np.nan,
            }

        agl = altitude_msl - gnd_alt
        horiz_dist, _ = pymap3d.vincenty.vdist(lat, lon, gnd_lat, gnd_lon)
        horiz_dist = float(horiz_dist)
        slant_range = float(np.sqrt(horiz_dist**2 + agl**2))

        normals = surface_normal_at(
            np.array([gnd_lat]), np.array([gnd_lon]), dem_file,
        )
        normal = normals[0]

        if horiz_dist > 0.1:
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
        cos_incidence = float(min(abs(np.dot(los_unit, normal)), 1.0))
        incidence_deg = float(np.degrees(np.arccos(cos_incidence)))

        div_rad = self.beam_divergence.m_as("radian")
        minor = float(np.tan(div_rad) * slant_range)
        major = float(minor / max(cos_incidence, 1e-6))
        area = float(np.pi / 4.0 * major * minor)
        equiv_diam = float(np.sqrt(major * minor))
        flat_diam = float(np.tan(div_rad) * agl)

        return {
            "ground_lat": gnd_lat,
            "ground_lon": gnd_lon,
            "ground_alt_m": gnd_alt,
            "altitude_agl_m": float(agl),
            "slant_range_m": slant_range,
            "incidence_deg": incidence_deg,
            "scan_angle_deg": float(scan_angle_deg),
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
        groundspeed: Quantity,
        dem_file: str | None = None,
        n_scan_positions: int = 51,
    ) -> dict[str, Any]:
        """Cross-track scan over terrain — per-position metrics.

        Discretises the scan into ``n_scan_positions`` angles from port
        to starboard, ray-traces each to the terrain, and reports per-
        position slant range, surface incidence, footprint ellipse,
        cross-track ground spacing, and contiguity.

        For ALS the contiguity model is **two-axis**:

        * **along-track**: ``groundspeed / scan_rate`` vs nadir
          footprint diameter (kinematic — same across the scan).
        * **cross-track**: ground pulse spacing between neighbours vs
          the *along-scan-line* footprint axis (which stretches by
          ``1 / cos(incidence)`` on slopes).

        Effective swath = length of the longest contiguous block where
        BOTH conditions hold.

        Returns:
            Dict with per-position arrays (NumPy) plus aggregate
            metrics (effective_swath_m, density stats, contiguous
            fraction).
        """
        from ..terrain import (
            generate_demfile,
            ray_terrain_intersection,
            surface_normal_at,
        )

        ha = float(self.scan_half_angle.m_as("degree"))
        scan_angles = np.linspace(-ha, ha, int(n_scan_positions))
        spd_mps = float(_as_quantity(
            groundspeed, "meter / second", "groundspeed",
        ).magnitude)
        scan_rate_hz = float(self.scan_rate.m_as("hertz"))
        div_rad = float(self.beam_divergence.m_as("radian"))

        if dem_file is None:
            dem_file = generate_demfile(lat, lon)  # type: ignore[arg-type]

        n = int(n_scan_positions)
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
                tilts[i] = float(sa)
            else:
                azimuths[i] = (heading + 270.0) % 360.0
                tilts[i] = abs(float(sa))

        gnd_lats, gnd_lons, gnd_alts = ray_terrain_intersection(
            lat0, lon0, altitude_msl, az=azimuths, tilt=tilts,
            dem_file=dem_file,
        )
        valid = ~np.isnan(gnd_lats)

        normals_all = np.zeros((n, 3))
        normals_all[:, 2] = 1.0
        if valid.any():
            normals_all[valid] = surface_normal_at(
                gnd_lats[valid], gnd_lons[valid], dem_file,
            )

        agls = altitude_msl - gnd_alts
        slant_ranges = np.full(n, np.nan)
        incidences = np.full(n, np.nan)
        fp_minor = np.full(n, np.nan)
        fp_major = np.full(n, np.nan)

        for i in range(n):
            if not valid[i]:
                continue
            horiz_dist, az_fwd = pymap3d.vincenty.vdist(
                lat, lon, gnd_lats[i], gnd_lons[i],
            )
            horiz_dist = float(horiz_dist)
            sr = float(np.sqrt(horiz_dist**2 + agls[i] ** 2))
            slant_ranges[i] = sr
            if horiz_dist > 0.1:
                az_rad = np.radians(float(az_fwd))
                los = np.array([
                    -horiz_dist * np.sin(az_rad),
                    -horiz_dist * np.cos(az_rad),
                    agls[i],
                ])
            else:
                los = np.array([0.0, 0.0, agls[i]])
            los /= np.linalg.norm(los)
            cos_inc = float(min(abs(np.dot(los, normals_all[i])), 1.0))
            incidences[i] = np.degrees(np.arccos(cos_inc))
            fp_minor[i] = np.tan(div_rad) * sr
            fp_major[i] = fp_minor[i] / max(cos_inc, 1e-6)

        # Diagnostic only: ground distance between successive
        # *discretization* positions (NOT pulse spacing).
        cross_spacings_discr = np.full(n - 1, np.nan)
        for i in range(n - 1):
            if valid[i] and valid[i + 1]:
                d, _ = pymap3d.vincenty.vdist(
                    gnd_lats[i], gnd_lons[i],
                    gnd_lats[i + 1], gnd_lons[i + 1],
                )
                cross_spacings_discr[i] = float(d)

        # Physical cross-track pulse spacing at each scan position.
        # For a constant-angular-velocity rotating-polygon scanner,
        # angular pulse step is fixed; ground spacing at scan angle θ
        # over sloped surface = slant_range * angular_step / cos(incidence).
        angular_step_rad = self._angular_pulse_step_rad()
        pulse_spacings = np.full(n, np.nan)
        for i in range(n):
            if np.isnan(slant_ranges[i]):
                continue
            cos_inc = float(np.cos(np.radians(incidences[i])))
            pulse_spacings[i] = float(
                slant_ranges[i] * angular_step_rad / max(cos_inc, 1e-6)
            )

        along_gap = spd_mps / scan_rate_hz
        local_densities = np.full(n, np.nan)
        mask = np.isfinite(pulse_spacings) & (pulse_spacings > 0)
        if mask.any():
            local_densities[mask] = 1.0 / (along_gap * pulse_spacings[mask])

        # --- Two-axis contiguity (physical pulse spacing, terrain-aware) ---
        contig_along = np.zeros(n, dtype=bool)
        contig_cross = np.zeros(n, dtype=bool)
        contig = np.zeros(n, dtype=bool)
        for i in range(n):
            if np.isnan(fp_minor[i]) or np.isnan(fp_major[i]):
                continue
            # Along-track: kinematic gap vs minor axis (laser-perpendicular
            # direction; equal to the flat-earth footprint diameter at
            # the slant range to this position).
            contig_along[i] = along_gap <= fp_minor[i] * (1.0 + 1e-9)
            # Cross-track: physical pulse spacing vs major axis (along-
            # scan-line direction; stretched by 1/cos(incidence) on slopes).
            if not np.isnan(pulse_spacings[i]):
                contig_cross[i] = (
                    pulse_spacings[i] <= fp_major[i] * (1.0 + 1e-9)
                )
                contig[i] = contig_along[i] and contig_cross[i]

        # Effective swath: longest contiguous block (in ground metres).
        effective_swath_m = 0.0
        start: int | None = None
        for i in range(n):
            if contig[i]:
                if start is None:
                    start = i
            else:
                if start is not None and valid[start] and valid[i - 1]:
                    span, _ = pymap3d.vincenty.vdist(
                        gnd_lats[start], gnd_lons[start],
                        gnd_lats[i - 1], gnd_lons[i - 1],
                    )
                    if float(span) > effective_swath_m:
                        effective_swath_m = float(span)
                    start = None
                else:
                    start = None
        if start is not None and valid[start] and valid[n - 1]:
            span, _ = pymap3d.vincenty.vdist(
                gnd_lats[start], gnd_lons[start],
                gnd_lats[n - 1], gnd_lons[n - 1],
            )
            if float(span) > effective_swath_m:
                effective_swath_m = float(span)

        valid_densities = local_densities[np.isfinite(local_densities)]
        if len(valid_densities) > 0:
            density_min = float(valid_densities.min())
            density_max = float(valid_densities.max())
            density_mean = float(valid_densities.mean())
            density_std = float(valid_densities.std())
        else:
            density_min = density_max = density_mean = density_std = 0.0

        if valid.any():
            flat_agl = float(altitude_msl - np.nanmean(gnd_alts))
            flat_swath = self.swath_width(flat_agl * ureg.meter).m_as("meter")
        else:
            flat_swath = 0.0

        return {
            "scan_angles_deg": scan_angles,
            "ground_lats": gnd_lats,
            "ground_lons": gnd_lons,
            "ground_alts_m": gnd_alts,
            "slant_ranges_m": slant_ranges,
            "incidence_angles_deg": incidences,
            "footprint_minor_m": fp_minor,
            "footprint_major_m": fp_major,
            "discretization_spacings_m": cross_spacings_discr,
            "pulse_spacings_m": pulse_spacings,
            "local_densities_pts_m2": local_densities,
            "contiguous_along_track": contig_along,
            "contiguous_cross_track": contig_cross,
            "contiguous_mask": contig,
            "effective_swath_m": effective_swath_m,
            "density_min_pts_m2": density_min,
            "density_max_pts_m2": density_max,
            "density_mean_pts_m2": density_mean,
            "density_std_pts_m2": density_std,
            "flat_earth_swath_m": float(flat_swath),
        }

    def terrain_summary(
        self,
        lat: float,
        lon: float,
        altitude_msl: float,
        heading: float,
        groundspeed: Quantity,
        dem_file: str | None = None,
    ) -> dict[str, Any]:
        """Coverage summary at a specific position with terrain correction.

        Combines the flat-earth :meth:`summary` output with terrain-
        aware metrics from :meth:`effective_swath_on_terrain` and a
        nadir footprint from :meth:`footprint_on_terrain`.  Keys
        prefixed with ``terrain_`` are the terrain-aware quantities.
        """
        from ..terrain import generate_demfile, get_elevations

        if dem_file is None:
            dem_file = generate_demfile(lat, lon)  # type: ignore[arg-type]

        gnd_elev = float(
            get_elevations(np.array([lat]), np.array([lon]), dem_file)[0]
        )
        agl = altitude_msl - gnd_elev
        flat = self.summary(agl * ureg.meter, groundspeed)

        terrain = self.effective_swath_on_terrain(
            lat, lon, altitude_msl, heading, groundspeed,
            dem_file=dem_file,
        )
        nadir = self.footprint_on_terrain(
            lat, lon, altitude_msl, heading,
            scan_angle_deg=0.0, dem_file=dem_file,
        )

        cmask = terrain["contiguous_mask"]
        flat.update({
            "terrain_ground_elevation_m": gnd_elev,
            "terrain_altitude_agl_m": float(agl),
            "terrain_nadir_incidence_deg": nadir["incidence_deg"],
            "terrain_nadir_footprint_minor_m": nadir["footprint_minor_m"],
            "terrain_nadir_footprint_major_m": nadir["footprint_major_m"],
            "terrain_effective_swath_m": terrain["effective_swath_m"],
            "terrain_flat_earth_swath_m": terrain["flat_earth_swath_m"],
            "terrain_density_min_pts_m2": terrain["density_min_pts_m2"],
            "terrain_density_max_pts_m2": terrain["density_max_pts_m2"],
            "terrain_density_mean_pts_m2": terrain["density_mean_pts_m2"],
            "terrain_density_std_pts_m2": terrain["density_std_pts_m2"],
            "terrain_contiguous_fraction": (
                float(cmask.sum()) / len(cmask) if len(cmask) else 0.0
            ),
        })
        return flat


# ---------------------------------------------------------------------------
# Pre-configured reference instance: RIEGL VQ-480 II at 1200 kHz operating point
# ---------------------------------------------------------------------------
# Note: Specific trade names are for informational purposes only and do not constitute an endorsement by NASA.
#
# All values transcribed from the RIEGL VQ-480 II data sheet (2024-08-23):
# https://www.riegl.com/fileadmin/media/Products/03_Airborne_Scanning/
#   RIEGL_VQ-480_II/RIEGL_VQ-480II_Datasheet_2024-08-23.pdf
#
# Operating point: PRR = 1200 kHz, scan_rate = 200 lines/sec (mid-range of
# the published 30–300 lines/sec envelope).  The datasheet's "example"
# diagram for 1200 kHz shows operation at 1700 ft AGL @ 120 kn yielding
# ~15.28 pts/m² average point density.
#
# At 1200 kHz the datasheet quotes:
#   * Max range at 60% reflectivity: 1050 m
#   * Max effective AGL at 60% reflectivity: 800 m (FOV ±37.5° + 5° roll)
#   * Max targets per pulse: 9
#
# MTA zone count is computed from max_range / (c / (2 * prf)) = 1050 / 124.9
# ≈ 8.4 → 9 zones (also matches the "9 targets per pulse" quoted at 1200 kHz,
# since each MTA zone yields one potential first-return time slot).
#
# Users wanting other PRR operating points should construct ALSLidar(...)
# directly with the appropriate prf / scan_rate / max_range / mta_zones from
# the datasheet's per-PRR tables (150/300/600/1200/2000 kHz).

RIEGL_VQ_480II = ALSLidar(
    name="RIEGL VQ-480 II",
    prf=1200 * ureg.kilohertz,
    scan_rate=200 * ureg.hertz,
    scan_half_angle=37.5 * ureg.degree,
    beam_divergence=0.35 * ureg.milliradian,
    wavelength=1550 * ureg.nanometer,
    max_range=1050 * ureg.meter,
    max_range_reflectivity=0.6,
    mta_zones=9,
    scan_geometry="rotating_polygon_active_arc",
    source=(
        "RIEGL VQ-480 II Data Sheet, 2024-08-23, public datasheet PDF "
        "retrieved 2026-05-15 from "
        "https://www.riegl.com/fileadmin/media/Products/03_Airborne_Scanning/"
        "RIEGL_VQ-480_II/RIEGL_VQ-480II_Datasheet_2024-08-23.pdf "
        "(1200 kHz operating point: 1050 m max range at 60% reflectivity, "
        "9 MTA zones; published 1700 ft AGL @ 120 kn → ~15.28 pts/m² "
        "example.  Scan rate 200 Hz chosen mid-range of published 30-300 "
        "lines/sec envelope.  ±37.5° scan = 75° total FOV.  Beam "
        "divergence ≤0.35 mrad.)"
    ),
)


# ---------------------------------------------------------------------------
# Multi-lidar rig — analog to MultiCameraRig from frame_camera.py
# ---------------------------------------------------------------------------


@dataclass
class LidarMount:
    """One unit in a :class:`MultiALSLidarRig` with its mount orientation.

    Attributes
    ----------
    lidar : ALSLidar
        The sensor instance.
    label : str
        Human-readable name for this unit (e.g. "forward", "aft").
    pitch_tilt_deg : float
        Pitch tilt about the aircraft's lateral axis (degrees).  Positive =
        nose-up looking forward along flight direction; negative = looking
        backward.  Pure pitch tilts do not change the cross-track swath
        — they shift the ground footprint forward/backward by
        ``altitude_agl × tan(pitch_tilt)`` and create multi-angle returns
        over the same cross-track ground swath.
    roll_tilt_deg : float
        Roll tilt about the aircraft's longitudinal axis (degrees).  Positive
        = starboard, negative = port.  Pure roll tilts shift the
        cross-track scan edges by the same angle, so a port-tilted unit
        plus a starboard-tilted unit produces a wider combined swath.
    dx, dy : Quantity
        Lateral and longitudinal physical offsets of the unit's mount point
        from the aircraft reference point.  Typically small (≤ 1 m) for
        rigid integrated systems and below the planning precision used by
        :meth:`MultiALSLidarRig.swath_width`.
    """

    lidar: ALSLidar
    label: str
    pitch_tilt_deg: float = 0.0
    roll_tilt_deg: float = 0.0
    dx: Quantity = field(default_factory=lambda: 0.0 * ureg.meter)
    dy: Quantity = field(default_factory=lambda: 0.0 * ureg.meter)


class MultiALSLidarRig(Sensor):
    """Rig of multiple :class:`ALSLidar` instances with known mount orientations.

    Models the two common multi-scanning-lidar integrations:

    * **Pitch-tilted forward/backward** (e.g. NASA G-LiHT's dual VQ-480i):
      both units image the same cross-track swath from forward and backward
      oblique angles.  Combined cross-track swath = single-unit swath;
      combined nominal density ≈ sum of unit densities; provides multi-angle
      returns useful for canopy 3D structure.
    * **Roll-tilted left/right**: each unit images a different cross-track
      band.  Combined cross-track swath is the union of the per-unit FOVs
      (wider than a single unit); density at any given ground point is the
      single-unit value, except in the overlap zone where the rolls are
      small enough that adjacent unit FOVs intersect.

    Conforms to the :class:`hyplan.instruments.ScanningSensor` Protocol so
    rig instances plug directly into
    :func:`hyplan.swath.generate_swath_polygon` and
    :func:`hyplan.flight_box.box_around_polygon`.  The rig's cross-track
    edges are the outermost edges across all units (each unit contributes
    ``[roll_tilt - half_angle, roll_tilt + half_angle]``).
    """

    def __init__(self, name: str, units: list[LidarMount]) -> None:
        super().__init__(name=name)
        if not units:
            raise HyPlanValueError("MultiALSLidarRig requires at least one unit")
        self.units: list[LidarMount] = list(units)

    def __len__(self) -> int:
        return len(self.units)

    def __iter__(self) -> Iterator[LidarMount]:
        return iter(self.units)

    # ------------------------------------------------------------------
    # ScanningSensor Protocol surface — combined rig geometry
    # ------------------------------------------------------------------

    def _edge_angles_deg(self) -> tuple[float, float]:
        """Combined cross-track edges (port, starboard) in degrees from
        nadir of the *rig* (not of any one unit).
        """
        port = min(
            u.roll_tilt_deg - u.lidar.half_angle for u in self.units
        )
        starboard = max(
            u.roll_tilt_deg + u.lidar.half_angle for u in self.units
        )
        return float(port), float(starboard)

    @property
    def half_angle(self) -> float:
        """Magnitude of the larger combined edge angle (degrees).

        Used by the ScanningSensor Protocol to size the cross-track ray
        cast.  Equals ``max(|port|, |starboard|)`` of the combined edges.
        """
        port, starboard = self._edge_angles_deg()
        return max(abs(port), abs(starboard))

    def swath_offset_angles(self) -> tuple[float, float]:
        """Combined cross-track edge angles ``(port, starboard)`` in
        degrees, from nadir.  Negative = port, positive = starboard.
        Asymmetric roll-tilted rigs are supported (port ≠ -starboard).
        """
        return self._edge_angles_deg()

    def swath_width(self, altitude_agl: Quantity) -> Quantity:
        """Total combined cross-track swath width on flat ground.

        ``swath = altitude × (tan(starboard_edge) - tan(port_edge))``

        For pure pitch-tilted rigs (all roll_tilt_deg == 0): equals the
        single-unit swath.  For roll-tilted rigs: extended union.
        """
        alt = self.units[0].lidar._validate_quantity(altitude_agl, ureg.meter)
        port_rad = np.radians(self._edge_angles_deg()[0])
        star_rad = np.radians(self._edge_angles_deg()[1])
        return alt * (np.tan(star_rad) - np.tan(port_rad))

    # ------------------------------------------------------------------
    # Multi-unit metrics
    # ------------------------------------------------------------------

    def along_track_offsets(self, altitude_agl: Quantity) -> dict[str, Quantity]:
        """Along-track ground offset of each unit's nadir line at altitude.

        For unit with pitch tilt θ, nadir ray hits the ground at
        ``altitude_agl × tan(θ)`` ahead of (or behind) the aircraft
        position.  Returns ``{label: offset_distance}`` (positive =
        forward of aircraft).
        """
        alt_m = self.units[0].lidar._validate_quantity(
            altitude_agl, ureg.meter,
        ).m_as("meter")
        return {
            u.label: float(alt_m * np.tan(np.radians(u.pitch_tilt_deg)))
            * ureg.meter
            for u in self.units
        }

    def combined_point_density(
        self,
        altitude_agl: Quantity,
        groundspeed: Quantity,
    ) -> Quantity:
        """Sum of per-unit nominal point densities (pts/m²).

        ``combined = sum(unit.point_density(altitude, speed) for unit in rig)``

        For pitch-only rigs whose units cover the same ground swath, this
        is the true combined density on every ground point.

        For roll-tilted rigs, this is the *mean* density: total pulses
        per unit time over the combined swath area.  Points in the
        overlap zone get higher density than this mean, points outside
        the overlap get lower density — call
        :meth:`unit_point_densities` for the per-unit breakdown.
        """
        alt = self.units[0].lidar._validate_quantity(altitude_agl, ureg.meter)
        spd = self.units[0].lidar._validate_quantity(
            groundspeed, ureg.meter / ureg.second,
        )
        total_prf_hz = sum(u.lidar.prf.m_as("hertz") for u in self.units)
        sw_m = self.swath_width(alt).m_as("meter")
        spd_mps = spd.m_as("meter / second")
        return (total_prf_hz / (spd_mps * sw_m)) / ureg.meter**2

    def unit_point_densities(
        self,
        altitude_agl: Quantity,
        groundspeed: Quantity,
    ) -> dict[str, Quantity]:
        """Per-unit nominal density at this altitude and groundspeed."""
        return {
            u.label: u.lidar.point_density(altitude_agl, groundspeed)
            for u in self.units
        }

    def solve_for_groundspeed(
        self,
        target_density: Quantity,
        altitude_agl: Quantity,
        *,
        strict_contiguity: bool = True,
    ) -> Quantity:
        """Groundspeed (m/s) at which the rig's **combined** point density
        equals ``target_density`` at the given ``altitude_agl``.

        Inverts :meth:`combined_point_density`: each unit contributes
        ``prf / (speed × swath)``, so the combined density across N
        units is ``(sum prf_i) / (speed × swath)``.  Solving for speed:

        .. code-block:: text

            speed = (sum prf_i) / (target_density × swath)

        For an N-unit pitch-only rig at identical PRF this is N× the
        per-unit speed returned by
        :meth:`ALSLidar.solve_for_groundspeed` — the dual VQ-480i rig
        therefore tolerates roughly **double** the per-unit
        density-limited speed.

        Same per-unit contiguity guard as
        :meth:`ALSLidar.solve_for_groundspeed`: any unit whose nadir
        footprint is smaller than the along-track scan spacing at the
        solved speed raises :class:`ContiguityError` (unless
        ``strict_contiguity=False``).

        Args:
            target_density: Target combined point density (pts/m²).
            altitude_agl: Altitude above ground level.
            strict_contiguity: When True (default), raise
                :class:`ContiguityError` if any unit's scan lines would
                not be contiguous at the solved speed.

        Returns:
            Ground speed (m/s) achieving the target combined density.
        """
        # Reuse the first unit's quantity validation for consistency.
        ref_unit = self.units[0].lidar
        d = ref_unit._validate_quantity(target_density, 1 / ureg.meter**2)
        alt = ref_unit._validate_quantity(altitude_agl, ureg.meter)
        if d.magnitude <= 0:
            raise HyPlanValueError("target_density must be positive")
        if alt.magnitude <= 0:
            raise HyPlanValueError("altitude_agl must be positive")

        total_prf_hz = sum(u.lidar.prf.m_as("hertz") for u in self.units)
        d_per_m2 = d.m_as(1 / ureg.meter**2)
        sw_m = self.swath_width(alt).m_as("meter")
        spd_mps = total_prf_hz / (d_per_m2 * sw_m)
        spd = spd_mps * ureg.meter / ureg.second

        if strict_contiguity:
            for unit in self.units:
                if not unit.lidar.is_along_track_contiguous(alt, spd):
                    gap = unit.lidar.along_track_spacing(spd).m_as("meter")
                    fp = unit.lidar.footprint_diameter(alt).m_as("meter")
                    raise ContiguityError(
                        f"Solved combined-density groundspeed "
                        f"{spd_mps:.1f} m/s at altitude "
                        f"{alt.m_as('meter'):.0f} m yields the target "
                        f"density but unit {unit.label!r}'s along-track "
                        f"gap {gap:.2f} m exceeds nadir footprint "
                        f"{fp:.2f} m — scan lines do not overlap.  "
                        f"Increase scan_rate, lower altitude, or pass "
                        f"strict_contiguity=False."
                    )
        return spd

    def multi_angle_pairs(
        self, min_dir_diff_deg: float = 5.0,
    ) -> list[tuple[LidarMount, LidarMount]]:
        """Find unit pairs with opposing pitch tilts (forward/backward).

        Two units are considered a multi-angle pair if their pitch tilts
        differ in sign and ``|θ_fwd - θ_aft| >= min_dir_diff_deg``.
        Default threshold is 5° — pitch tilts of ±7° (typical for canopy
        multi-angle configurations) clear it; the tighter QUAKES-I-style
        ±10°-and-up pairs are obviously detected too.  Returns ordered
        ``(forward, backward)`` pairs — ``forward`` is the unit with
        positive pitch_tilt_deg.

        Analogous to :meth:`MultiCameraRig.stereo_pairs`.
        """
        pairs: list[tuple[LidarMount, LidarMount]] = []
        used: set[int] = set()
        for i, a in enumerate(self.units):
            if i in used:
                continue
            for j, b in enumerate(self.units):
                if j <= i or j in used:
                    continue
                if np.sign(a.pitch_tilt_deg) == -np.sign(b.pitch_tilt_deg) and (
                    a.pitch_tilt_deg != 0.0 or b.pitch_tilt_deg != 0.0
                ):
                    diff = abs(a.pitch_tilt_deg - b.pitch_tilt_deg)
                    if diff >= min_dir_diff_deg:
                        if a.pitch_tilt_deg > b.pitch_tilt_deg:
                            pairs.append((a, b))
                        else:
                            pairs.append((b, a))
                        used.add(i)
                        used.add(j)
                        break
        return pairs

    def coverage_diagnostic(
        self,
        altitude_agl: Quantity,
        groundspeed: Quantity,
    ) -> dict[str, Any]:
        """Per-unit + combined coverage breakdown for the rig.

        Returns a dict with:

        * ``units`` — list of per-unit
          :meth:`ALSLidar.coverage_diagnostic` results, each tagged with
          ``label`` and ``pitch_tilt_deg`` / ``roll_tilt_deg``.
        * ``combined_swath_width_m`` — rig swath width.
        * ``combined_density_pts_m2`` — sum of per-unit nominal densities.
        * ``along_track_offsets_m`` — per-unit ground offset for pitch tilts.
        """
        per_unit: list[dict[str, Any]] = []
        for u in self.units:
            d: dict[str, Any] = dict(
                u.lidar.coverage_diagnostic(altitude_agl, groundspeed)
            )
            d["label"] = u.label
            d["pitch_tilt_deg"] = u.pitch_tilt_deg
            d["roll_tilt_deg"] = u.roll_tilt_deg
            per_unit.append(d)
        offsets = self.along_track_offsets(altitude_agl)
        return {
            "units": per_unit,
            "combined_swath_width_m": float(
                self.swath_width(altitude_agl).m_as("meter")
            ),
            "combined_density_pts_m2": float(
                self.combined_point_density(altitude_agl, groundspeed).m_as(
                    1 / ureg.meter**2,
                )
            ),
            "along_track_offsets_m": {
                lbl: float(off.m_as("meter")) for lbl, off in offsets.items()
            },
        }


# ---------------------------------------------------------------------------
# G-LiHT 2017+ dual VQ-480i reference instance
# ---------------------------------------------------------------------------
# Note: Specific trade names are for informational purposes only and do not constitute an endorsement by NASA.
#
# Section 2.8 of the G-LiHT v2.0 User Guide (Wirt 2021, LP DAAC) describes
# the 2017+ upgrade verbatim as "the Riegl VQ 480i Dual Scanning LiDAR".
# Section 2.2 of the same user guide describes the operating parameters of
# the original VQ-480 (300 kHz PRR, 100 scans/sec, 0.3 mrad divergence,
# 1550 nm, 60° FOV, 387 m swath at the nominal 335 m AGL).
#
# The user guide does NOT publish the tilt angle between the two units.
# Public design clues:
#
#   * The combined swath (387 m at 335 m AGL = single-unit 60° FOV figure)
#     and combined max density (~12 pts/m², single-unit-class value) in
#     the v.2 specs page strongly imply the dual config does NOT extend
#     cross-track swath nor multiply density via overlap.  The geometry
#     consistent with both numbers is forward/backward pitch tilt of the
#     same cross-track scan.
#   * RIEGL later productized this idea as the VQ-680 NFB scanner with
#     nadir/forward/backward angles of ±10° and ±20°.  G-LiHT's dual
#     VQ-480i predates the VQ-680.
#
# We instantiate two VQ-480i units pitch-tilted ±7° as a representative
# value typical of canopy-lidar multi-angle configurations.  The tilt is
# parametrically exposed — users with authoritative G-LiHT documentation
# can override by constructing the rig directly.
#
# Each unit's parameters match the original VQ-480 / VQ-480i specs
# (300 kHz PRR, 100 lines/sec, ±30° = 60° FOV, 0.3 mrad divergence,
# 1550 nm) — NOT the newer VQ-480 II in `RIEGL_VQ_480II`.

_GLIHT_VQ_480I_UNIT = ALSLidar(
    name="RIEGL VQ-480i (G-LiHT)",
    prf=300 * ureg.kilohertz,
    scan_rate=150 * ureg.hertz,
    scan_half_angle=30.0 * ureg.degree,
    beam_divergence=0.3 * ureg.milliradian,
    wavelength=1550 * ureg.nanometer,
    max_range=1850 * ureg.meter,  # at 60% reflectivity, single-unit datasheet
    max_range_reflectivity=0.6,
    mta_zones=2,  # c/(2·300 kHz) = 500 m per zone; max range 1850 m → 4 zones
    scan_geometry="rotating_polygon_active_arc",
    source=(
        "RIEGL VQ-480i datasheet (60° FOV, 300 kHz max PRR, 0.3 mrad beam "
        "divergence, 1550 nm); operating parameters from the G-LiHT Loudon "
        "June 2017 campaign metadata (NASA GSFC; "
        "https://glihtdata.gsfc.nasa.gov/, S/N S2220331 'new' unit): "
        "PRF 300 kHz, scan rate 150 lines/sec, effective measurement "
        "frequency = 0.5 × PRF = 150 kHz, max 8 returns/pulse."
    ),
)

GLIHT_DUAL_VQ_480I = MultiALSLidarRig(
    name="G-LiHT Dual VQ-480i (2017+)",
    units=[
        LidarMount(
            lidar=_GLIHT_VQ_480I_UNIT,
            label="forward",
            pitch_tilt_deg=+7.0,
        ),
        LidarMount(
            lidar=_GLIHT_VQ_480I_UNIT,
            label="backward",
            pitch_tilt_deg=-7.0,
        ),
    ],
)
GLIHT_DUAL_VQ_480I.__doc__ = (
    "NASA G-LiHT 2017+ dual VQ-480i scanning lidar rig.\n\n"
    "Two identical RIEGL VQ-480i units pitch-tilted ±7° (forward and "
    "backward) viewing the same cross-track 60° FOV swath from two "
    "oblique along-track angles, providing multi-angle returns for "
    "canopy 3D structure and approximately doubled point density.\n\n"
    "Source: G-LiHT V2.0/V4.0 User Guide (Wirt 2021, LP DAAC), sec. 2.8 "
    "describes the upgrade as 'Riegl VQ 480i Dual Scanning LiDAR'.  The "
    "user guide does not publish the tilt angle; ±7° is a representative "
    "value used here, parametrically exposed for override.  Combined swath "
    "and density numbers in the v.2 specs page (387 m at 335 m AGL, "
    "~12 pts/m²) are consistent with forward/backward pitch tilt of the "
    "same cross-track scan rather than cross-track FOV extension."
)
