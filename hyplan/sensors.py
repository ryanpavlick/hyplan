"""Imaging spectrometer and line-scanner sensor models.

Defines the :class:`Sensor` base class and :class:`LineScanner` subclass for
computing ground sample distance (GSD), swath width, and critical speed from
sensor optics, altitude, and aircraft parameters.  Pre-configured sensors
include NASA instruments (AVIRIS-3, AVIRIS-5, HyTES, PRISM, MASTER) and
others.  Use :func:`create_sensor` for name-based construction.
"""

from typing import Dict, Type

import numpy as np
from pint import Quantity

from .exceptions import HyPlanTypeError, HyPlanValueError
from .units import ureg

__all__ = [
    "Sensor",
    "LineScanner",
    "AVIRISClassic",
    "AVIRISNextGen",
    "AVIRIS3",
    "AVIRIS5",
    "HyTES",
    "PRISM",
    "MASTER",
    "GLiHT_VNIR",
    "GLiHT_Thermal",
    "GLiHT_SIF",
    "GCAS_UV_Vis",
    "GCAS_VNIR",
    "eMAS",
    "PICARD",
    "create_sensor",
    "SENSOR_REGISTRY",
]


class Sensor:
    """Base class to represent a generic sensor.

    Args:
        name (str): Human-readable name identifying the sensor.
    """
    def __init__(self, name: str):
        self.name = name

    def __str__(self) -> str:
        return self.name

    def _validate_quantity(self, value: Quantity, expected_unit: Quantity) -> Quantity:
        """
        Validate that a value is a pint Quantity and convert it to the expected unit.

        Args:
            value (Quantity): The value to validate.
            expected_unit (Quantity): The target unit to convert to.

        Returns:
            Quantity: The value converted to the expected unit.

        Raises:
            TypeError: If value is not a pint Quantity.
        """
        if not isinstance(value, Quantity):
            raise HyPlanTypeError(f"Expected a pint.Quantity for {expected_unit}, but got {type(value)}.")
        return value.to(expected_unit)

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

    def swath_offset_angles(self) -> tuple:
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

    def ground_sample_distance(self, altitude_agl: Quantity, mode: str = "nadir") -> Quantity:
        """Calculate the ground sample distance (GSD) for a given altitude above ground level (AGL)."""
        altitude_agl = self._validate_quantity(altitude_agl, ureg.meter)

        if mode == "nadir":
            return 2 * altitude_agl * np.tan(np.radians(self.ifov / 2))

        elif mode == "average":
            return self.swath_width(altitude_agl) / self.across_track_pixels

        elif mode == "edge":
            edge_ifov = self.fov / 2.0 / (self.across_track_pixels / 2.0)
            return 2 * altitude_agl * np.tan(np.radians(edge_ifov / 2))

        else:
            return 2 * altitude_agl * np.tan(np.radians(self.ifov / 2))

    def altitude_agl_for_ground_sample_distance(self, gsd: Quantity, mode: str = "nadir") -> Quantity:
        """Calculate the required altitude AGL (Above Ground Level) for a given ground sample distance (GSD)."""
        gsd = self._validate_quantity(gsd, ureg.meter)

        if mode == "nadir":
            return gsd / (2 * np.tan(np.radians(self.ifov / 2)))

        elif mode == "average":
            return (self.across_track_pixels * gsd) / (2 * np.tan(np.radians(self.fov / 2)))

        elif mode == "edge":
            edge_ifov = self.fov / 2.0 / (self.across_track_pixels / 2.0)
            return gsd / (2 * np.tan(np.radians(edge_ifov / 2)))

        else:
            return gsd / (2 * np.tan(np.radians(self.ifov / 2)))

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
        Calculate the along-track pixel size for a given aircraft speed and oversampling rate.

        Args:
            aircraft_speed (Quantity): Speed of the aircraft in m/s.
            oversampling_rate (float): Oversampling factor (default = 1.0).

        Returns:
            Quantity: Along-track pixel size in meters.
        """
        aircraft_speed = self._validate_quantity(aircraft_speed, ureg.meter / ureg.second)
        return aircraft_speed * self.frame_period / along_track_sampling




# ── Sensor Specifications ─────────────────────────────────────────────────────
# Each entry maps class_name -> (display_name, fov_deg, across_track_pixels, frame_rate_hz)

_SENSOR_SPECS = {
    "AVIRISClassic":  ("AVIRIS Classic",                              34.0,  677,  12.0),
    "AVIRISNextGen":  ("AVIRIS Next Gen",                             36.0,  600, 100.0),
    "AVIRIS3":        ("AVIRIS 3",                                    39.6, 1234, 216.0),
    "AVIRIS5":        ("AVIRIS 5",                                    40.2, 1239, 148.0),
    "HyTES":          ("HyTES",                                       50.0,  512,  36.0),
    "PRISM":          ("PRISM",                                       30.7,  608, 176.0),
    "MASTER":         ("MASTER",                                      85.92, 716,  25.0),
    "GLiHT_VNIR":     ("G-LiHT VNIR",                                64.0, 1600, 250.0),
    "GLiHT_Thermal":  ("G-LiHT Thermal",                             42.6,  640,  50.0),
    "GLiHT_SIF":      ("G-LiHT SIF",                                 23.5, 1600,  37.6),
    "GCAS_UV_Vis":    ("GCAS UV-Vis Spectrometer",                    45.0, 1024,  12.0),
    "GCAS_VNIR":      ("GCAS Visible Near-Infrared (VNIR) Spectrometer", 70.0, 1024, 12.0),
    "eMAS":           ("eMAS",                                        85.92, 716,   6.25),
    "PICARD":         ("PICARD",                                      50.0,  412, 100.0),
}


def _make_sensor_class(class_name, display_name, fov, across_track_pixels, frame_rate_hz):
    """Create a LineScanner subclass from spec parameters."""
    def __init__(self):
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
GLiHT_Thermal: type = globals()["GLiHT_Thermal"]
GLiHT_SIF: type = globals()["GLiHT_SIF"]
GCAS_UV_Vis: type = globals()["GCAS_UV_Vis"]
GCAS_VNIR: type = globals()["GCAS_VNIR"]
eMAS: type = globals()["eMAS"]
PICARD: type = globals()["PICARD"]

def create_sensor(sensor_type: str) -> Sensor:
    """
    Factory function to create and return an instance of a sensor.

    Args:
        sensor_type (str): The name of the sensor class to instantiate.
                           Must be one of the keys in SENSOR_REGISTRY.

    Returns:
        Sensor: An instance of the requested sensor type.

    Raises:
        ValueError: If the specified sensor_type is not found in SENSOR_REGISTRY.
    """
    # Lazy registration of sensors from other modules to avoid circular imports
    if "LVIS" not in SENSOR_REGISTRY:
        from .lvis import LVIS
        SENSOR_REGISTRY["LVIS"] = LVIS

    if "UAVSAR_Lband" not in SENSOR_REGISTRY:
        from .radar import UAVSAR_Lband, UAVSAR_Pband, UAVSAR_Kaband
        SENSOR_REGISTRY["UAVSAR_Lband"] = UAVSAR_Lband
        SENSOR_REGISTRY["UAVSAR L-band"] = UAVSAR_Lband
        SENSOR_REGISTRY["UAVSAR_Pband"] = UAVSAR_Pband
        SENSOR_REGISTRY["UAVSAR P-band"] = UAVSAR_Pband
        SENSOR_REGISTRY["UAVSAR_Kaband"] = UAVSAR_Kaband
        SENSOR_REGISTRY["GLISTIN-A"] = UAVSAR_Kaband

    if sensor_type not in SENSOR_REGISTRY:
        raise HyPlanValueError(f"Unknown sensor type: {sensor_type}")
    return SENSOR_REGISTRY[sensor_type]()


SENSOR_REGISTRY: Dict[str, Type[Sensor]] = {
    "AVIRISClassic": AVIRISClassic,
    "AVIRIS Classic": AVIRISClassic,
    "AVIRISNextGen": AVIRISNextGen,
    "AVIRIS-NG": AVIRISNextGen,
    "AVIRIS3": AVIRIS3,
    "AVIRIS-3": AVIRIS3,
    "AVIRIS5": AVIRIS5,
    "AVIRIS-5": AVIRIS5,
    "HyTES": HyTES,
    "PRISM": PRISM,
    "MASTER": MASTER,
    "GLiHT_VNIR": GLiHT_VNIR,
    "GLiHT_Thermal": GLiHT_Thermal,
    "GLiHT_SIF": GLiHT_SIF,
    "GCAS_UV_Vis": GCAS_UV_Vis,
    "GCAS_VNIR": GCAS_VNIR,
    "eMAS": eMAS,
    "PICARD": PICARD,
}
