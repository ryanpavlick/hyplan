"""Imaging spectrometer and line-scanner sensor models.

Defines the :class:`LineScanner` subclass for computing ground sample distance
(GSD), swath width, and critical speed from sensor optics, altitude, and
aircraft parameters.  Pre-configured sensors include NASA instruments
(AVIRIS-3, AVIRIS-5, HyTES, PRISM, MASTER) and others.  Use
:func:`create_sensor` for name-based construction.
"""


from typing import Any

import numpy as np
from pint import Quantity

from ..exceptions import HyPlanTypeError, HyPlanValueError
from ..units import ureg
from ._base import Sensor
from .registry import SENSOR_REGISTRY, create_sensor  # re-export (see footer)

__all__ = [
    "AVIRIS3",
    "AVIRIS5",
    "GCAS_VNIR",
    "MASTER",
    "PICARD",
    "PRISM",
    "SENSOR_REGISTRY",
    "AVIRISClassic",
    "AVIRISNextGen",
    "GCAS_UV_Vis",
    "GLiHT_SIF",
    "GLiHT_SWIR",
    "GLiHT_VNIR",
    "HyTES",
    "LineScanner",
    "create_sensor",
    "eMAS",
]


class LineScanner(Sensor):
    """
    A pushbroom or whiskbroom line scanning imager.

    Line scanners capture one cross-track line of pixels per frame,
    building up an image as the aircraft moves along-track.

    Args:
        name (str): Sensor name.
        fov (float): Total cross-track field of view in degrees.
        across_track_pixels (int): Number of pixels across the swath.
        frame_rate (Quantity): Frame acquisition rate in Hz.
        cross_track_tilt (float): Cross-track tilt angle in degrees
            (rotation about the along-track axis). Positive = starboard
            (right of track), negative = port (left of track).
            Default 0.0 (nadir-looking).
    """

    def __init__(
        self,
        name: str,
        fov: float,  # Degrees (not a Quantity)
        across_track_pixels: int,
        frame_rate: Quantity,  # Hz
        cross_track_tilt: float = 0.0,  # Degrees
    ):
        super().__init__(name)

        # Validate FOV
        if not isinstance(fov, (int, float)):
            raise HyPlanTypeError(f"fov must be a number, got {type(fov)}.")
        self.fov = float(fov)

        # Validate across_track_pixels
        if not isinstance(across_track_pixels, int):
            raise HyPlanTypeError(f"across_track_pixels must be an integer, got {type(across_track_pixels)}.")
        self.across_track_pixels = across_track_pixels

        # Validate frame_rate
        self.frame_rate = self._validate_quantity(frame_rate, ureg.Hz)

        self.cross_track_tilt = float(cross_track_tilt)

    @property
    def ifov(self) -> float:
        """Calculate the cross-track Instantaneous Field of View (IFOV) in degrees."""
        return self.fov / self.across_track_pixels

    @property
    def half_angle(self) -> float:
        """Calculate and return the half angle in degrees."""
        return self.fov / 2.0

    @property
    def frame_period(self) -> Quantity:
        """Calculate and return the frame period in seconds."""
        return (1.0 / self.frame_rate).to(ureg.s)

    def swath_offset_angles(self) -> tuple[float, float]:
        """Cross-track viewing angles for each swath edge, measured from nadir.

        Accounts for ``cross_track_tilt`` (rotation about the along-track axis).
        Negative = port (left of track), positive = starboard (right of track).

        Returns:
            Tuple of (port_edge_angle, starboard_edge_angle) in degrees.

        Examples:
            Nadir sensor, 30° half-angle: ``(-30.0, 30.0)``
            Same sensor with 10° starboard tilt: ``(-20.0, 40.0)``
        """
        return (
            self.cross_track_tilt - self.half_angle,
            self.cross_track_tilt + self.half_angle,
        )

    def swath_width(self, altitude_agl: Quantity) -> Quantity:
        """Calculate swath width for a given altitude above ground level (AGL).

        Accounts for ``cross_track_tilt`` — when the sensor is tilted off-nadir
        the swath is asymmetric and its total width changes.

        Args:
            altitude_agl (Quantity): Altitude above ground level.

        Returns:
            Quantity: Swath width in meters.
        """
        altitude_agl = self._validate_quantity(altitude_agl, ureg.meter)
        port, starboard = self.swath_offset_angles()
        h = altitude_agl.magnitude
        d_port = h * np.tan(np.radians(port))
        d_starboard = h * np.tan(np.radians(starboard))
        return abs(d_starboard - d_port) * ureg.meter

    def _edge_gsd_factor(self) -> float:
        """Cross-track edge-GSD factor: ``tan(θ_edge) − tan(θ_edge − ifov)``.

        Exact projection of the outermost pixel's IFOV onto flat ground,
        where ``θ_edge = fov / 2``.  Multiply by altitude AGL to get the
        edge GSD.
        """
        edge_rad = np.radians(self.half_angle)
        return float(np.tan(edge_rad) - np.tan(edge_rad - np.radians(self.ifov)))

    def ground_sample_distance(self, altitude_agl: Quantity, mode: str = "nadir") -> Quantity:
        """Calculate the ground sample distance (GSD) for a given altitude above ground level (AGL).

        Args:
            altitude_agl (Quantity): Altitude above ground level.
            mode (str): One of ``"nadir"`` (GSD directly below the
                aircraft), ``"average"`` (swath width / pixel count), or
                ``"edge"`` (cross-track GSD of the outermost pixel,
                exact form ``h · (tan(θ_edge) − tan(θ_edge − ifov))``
                with ``θ_edge = fov / 2``).

        Raises:
            HyPlanValueError: when ``mode`` is not recognized.
        """
        altitude_agl = self._validate_quantity(altitude_agl, ureg.meter)

        if mode == "nadir":
            return 2 * altitude_agl * np.tan(np.radians(self.ifov / 2))

        if mode == "average":
            return self.swath_width(altitude_agl) / self.across_track_pixels

        if mode == "edge":
            return altitude_agl * self._edge_gsd_factor()

        raise HyPlanValueError(f"mode must be 'nadir', 'average', or 'edge', got {mode!r}")

    def altitude_agl_for_ground_sample_distance(self, gsd: Quantity, mode: str = "nadir") -> Quantity:
        """Calculate the required altitude AGL (Above Ground Level) for a given ground sample distance (GSD).

        Inverts :meth:`ground_sample_distance` for the same ``mode``, so
        the two methods round-trip exactly.

        Raises:
            HyPlanValueError: when ``mode`` is not recognized.
        """
        gsd = self._validate_quantity(gsd, ureg.meter)

        if mode == "nadir":
            return gsd / (2 * np.tan(np.radians(self.ifov / 2)))

        if mode == "average":
            return (self.across_track_pixels * gsd) / (2 * np.tan(np.radians(self.fov / 2)))

        if mode == "edge":
            return gsd / self._edge_gsd_factor()

        raise HyPlanValueError(f"mode must be 'nadir', 'average', or 'edge', got {mode!r}")

    def critical_ground_speed(self, altitude_agl: Quantity, along_track_sampling: float = 1.0) -> Quantity:
        """
        Calculate the maximum allowable aircraft ground speed (m/s) to maintain proper along-track sampling.

        Args:
            altitude_agl (Quantity): Altitude above ground level in meters.
            along_track_sampling (float): The oversampling factor (default = 1.0).

        Returns:
            Quantity: Maximum allowable ground speed in meters per second.
        """
        altitude_agl = self._validate_quantity(altitude_agl, ureg.meter)
        return self.ground_sample_distance(altitude_agl, mode="nadir") / (self.frame_period * along_track_sampling)

    def along_track_pixel_size(self, aircraft_speed: Quantity, along_track_sampling: float = 1.0) -> Quantity:
        """
        Calculate the along-track pixel size for a given aircraft speed and along-track sampling factor.

        Args:
            aircraft_speed (Quantity): Speed of the aircraft in m/s.
            along_track_sampling (float): Along-track sampling (oversampling) factor (default = 1.0).

        Returns:
            Quantity: Along-track pixel size in meters.
        """
        aircraft_speed = self._validate_quantity(aircraft_speed, ureg.meter / ureg.second)
        return aircraft_speed * self.frame_period / along_track_sampling

    def ground_pixel_dimensions(
        self,
        altitude_agl: Quantity,
        ground_speed: Quantity,
    ) -> dict[str, Quantity | float]:
        """Cross-track and along-track ground pixel size + aspect ratio.

        For a pushbroom line scanner, the cross-track GSD is locked by
        sensor optics (lens + pixel pitch), while the along-track
        "pixel" is just ``ground_speed × frame_period``.  The ratio of
        the two tells you whether your data product has square or
        rectangular ground pixels at the planned operating point.

        Args:
            altitude_agl: Altitude above ground level.
            ground_speed: Platform ground speed.

        Returns:
            A dict with keys ``"cross_track"`` (Quantity, meters),
            ``"along_track"`` (Quantity, meters), and ``"aspect_ratio"``
            (float).  ``aspect_ratio`` is ``along_track / cross_track``
            (1.0 = square pixels; >1 = elongated along the flight
            direction).
        """
        altitude_agl = self._validate_quantity(altitude_agl, ureg.meter)
        ground_speed = self._validate_quantity(ground_speed, ureg.meter / ureg.second)
        cross_track = self.ground_sample_distance(altitude_agl, mode="nadir").to(ureg.meter)
        along_track = (ground_speed * self.frame_period).to(ureg.meter)
        aspect = float((along_track / cross_track).to_reduced_units().magnitude)
        return {
            "cross_track": cross_track,
            "along_track": along_track,
            "aspect_ratio": aspect,
        }


# ── Sensor Specifications ─────────────────────────────────────────────────────
# Each entry maps class_name -> (display_name, fov_deg, across_track_pixels, frame_rate_hz)

# FOV (deg) and across-track pixel counts verified against the cited
# instrument pages/publications (retrieved 2026-06).  Frame rates are the
# nominal maximum where given; rows marked "spec source unverified" could
# not be confirmed against an authoritative source within review scope and
# carry forward HyPlan's prior values unchanged.
_SENSOR_SPECS = {
    # source: aviris.jpl.nasa.gov/aviris/instrument.html (34° scan, 677 px, 12 Hz whiskbroom), retrieved 2026-06
    "AVIRISClassic":  ("AVIRIS Classic",                              34.0,  677,  12.0),
    # source: avirisng.jpl.nasa.gov/specifications.html (36°±2 FOV, 600 resolved elements, up to 100 fps), retrieved 2026-06
    "AVIRISNextGen":  ("AVIRIS Next Gen",                             36.0,  600, 100.0),
    # source: earth.jpl.nasa.gov AVIRIS-3 / ORNL DAAC AV3_L1B (~39.5–39.6° FOV, 1234 px), retrieved 2026-06
    "AVIRIS3":        ("AVIRIS 3",                                    39.6, 1234, 216.0),
    # spec source unverified (AVIRIS-5 is EMIT-design; airborne FOV/pixel count not confirmed)
    "AVIRIS5":        ("AVIRIS 5",                                    40.2, 1239, 148.0),
    # source: hytes.jpl.nasa.gov/specifications (50° FOV, 512 px cross-track), retrieved 2026-06
    "HyTES":          ("HyTES",                                       50.0,  512,  36.0),
    # source: Mouroulis et al. 2014 (Appl. Opt. 53, 1363) / JPL PRISM (30.8° swath, 608 px), retrieved 2026-06
    "PRISM":          ("PRISM",                                       30.7,  608, 176.0),
    # source: MASTER ASAP datasheet / master.jpl.nasa.gov (85.92° FOV, 716 px, 6.25–25 Hz), retrieved 2026-06
    "MASTER":         ("MASTER",                                      85.92, 716,  25.0),
    # spec source unverified (Headwall Microhyperspec E; FOV in degrees not confirmed)
    "GLiHT_VNIR":     ("G-LiHT VNIR (Headwall Microhyperspec E)",   55.3,  645,  75.0),
    # spec source unverified (Headwall Microhyperspec SWIR; FOV in degrees not confirmed)
    "GLiHT_SWIR":     ("G-LiHT SWIR (Headwall Microhyperspec SWIR)",20.9,  192,  75.0),
    # spec source unverified (Headwall FIREFLY SIF; FOV in degrees not confirmed)
    "GLiHT_SIF":      ("G-LiHT SIF (Headwall FIREFLY)",             23.5, 1600,  37.5),
    # spec source unverified (GCAS UV-Vis; FOV/pixel count not confirmed against a GSFC source)
    "GCAS_UV_Vis":    ("GCAS UV-Vis Spectrometer",                    45.0, 1024,  12.0),
    # spec source unverified (GCAS VNIR; FOV/pixel count not confirmed against a GSFC source)
    "GCAS_VNIR":      ("GCAS Visible Near-Infrared (VNIR) Spectrometer", 70.0, 1024, 12.0),
    # spec source unverified (eMAS shares the MAS-family 85.92°/716 px scanner geometry; rate not independently confirmed)
    "eMAS":           ("eMAS",                                        85.92, 716,   6.25),
    # spec source unverified (PICARD; FOV/pixel count not confirmed against an authoritative source)
    "PICARD":         ("PICARD",                                      50.0,  412, 100.0),
}


def _make_sensor_class(
    class_name: str,
    display_name: str,
    fov: float,
    across_track_pixels: int,
    frame_rate_hz: float,
) -> type:
    """Create a LineScanner subclass from spec parameters."""
    def __init__(self: Any) -> None:
        LineScanner.__init__(
            self,
            name=display_name,
            fov=fov,
            across_track_pixels=across_track_pixels,
            frame_rate=frame_rate_hz * ureg.Hz,
        )
    doc = f"{display_name} ({fov}° FOV, {across_track_pixels} pixels, {frame_rate_hz} Hz)."
    return type(class_name, (LineScanner,), {"__init__": __init__, "__doc__": doc})


# Dynamically create all sensor classes and inject into module namespace
for _cls_name, (_disp, _fov, _pix, _hz) in _SENSOR_SPECS.items():
    globals()[_cls_name] = _make_sensor_class(_cls_name, _disp, _fov, _pix, _hz)

# Expose concrete names for static analysis / IDE autocomplete
AVIRISClassic: type = globals()["AVIRISClassic"]
AVIRISNextGen: type = globals()["AVIRISNextGen"]
AVIRIS3: type = globals()["AVIRIS3"]
AVIRIS5: type = globals()["AVIRIS5"]
HyTES: type = globals()["HyTES"]
PRISM: type = globals()["PRISM"]
MASTER: type = globals()["MASTER"]
GLiHT_VNIR: type = globals()["GLiHT_VNIR"]
GLiHT_SWIR: type = globals()["GLiHT_SWIR"]
GLiHT_SIF: type = globals()["GLiHT_SIF"]
GCAS_UV_Vis: type = globals()["GCAS_UV_Vis"]
GCAS_VNIR: type = globals()["GCAS_VNIR"]
eMAS: type = globals()["eMAS"]
PICARD: type = globals()["PICARD"]


# SENSOR_REGISTRY and create_sensor live in `hyplan.instruments.registry`;
# they are re-exported above for backwards compatibility. Registration of
# every sensor (line scanners + frame cameras + dropsondes + radars + …)
# happens explicitly in `hyplan.instruments.__init__` after all instrument
# modules have been imported, so no module needs to reach into another.
