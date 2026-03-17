from typing import Dict, Type
import numpy as np
from pint import Quantity
from .units import ureg

__all__ = [
    "Sensor",
    "LineScanner",
    "AVIRISClassic",
    "AVIRISNextGen",
    "AVIRIS3",
    "AVIRIS4",
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
            raise TypeError(f"Expected a pint.Quantity for {expected_unit}, but got {type(value)}.")
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
    """

    def __init__(
        self,
        name: str,
        fov: float,  # Degrees (not a Quantity)
        across_track_pixels: int,
        frame_rate: Quantity  # Hz
    ):
        super().__init__(name)

        # Validate FOV
        if not isinstance(fov, (int, float)):
            raise TypeError(f"fov must be a number, got {type(fov)}.")
        self.fov = float(fov)

        # Validate across_track_pixels
        if not isinstance(across_track_pixels, int):
            raise TypeError(f"across_track_pixels must be an integer, got {type(across_track_pixels)}.")
        self.across_track_pixels = across_track_pixels

        # Validate frame_rate
        self.frame_rate = self._validate_quantity(frame_rate, ureg.Hz)

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

    def swath_width(self, altitude_agl: Quantity) -> Quantity:
        """Calculate swath width (m) for a given altitude above ground level (AGL)."""
        altitude_agl = self._validate_quantity(altitude_agl, ureg.meter)
        return 2 * altitude_agl * np.tan(np.radians(self.fov / 2))

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




class AVIRISClassic(LineScanner):
    """AVIRIS Classic imaging spectrometer (34° FOV, 677 pixels, 100 Hz)."""

    def __init__(self):
        super().__init__(
            name="AVIRIS Classic",
            fov=34.0,
            across_track_pixels=677,
            frame_rate=100 * ureg.Hz
        )

class AVIRISNextGen(LineScanner):
    """AVIRIS Next Generation imaging spectrometer (36° FOV, 600 pixels, 100 Hz)."""

    def __init__(self):
        super().__init__(
            name="AVIRIS Next Gen",
            fov=36.0,
            across_track_pixels=600,
            frame_rate=100 * ureg.Hz
        )

class AVIRIS3(LineScanner):
    """AVIRIS-3 imaging spectrometer (40.2° FOV, 1240 pixels, 216 Hz)."""

    def __init__(self):
        super().__init__(
            name="AVIRIS 3",
            fov=40.2,
            across_track_pixels=1240,
            frame_rate=216 * ureg.Hz
        )

class AVIRIS4(LineScanner):
    """AVIRIS-4 imaging spectrometer (39.5° FOV, 1240 pixels, 215 Hz)."""

    def __init__(self):
        super().__init__(
            name="AVIRIS 4",
            fov=39.5,
            across_track_pixels=1240,
            frame_rate=215 * ureg.Hz
        )

class HyTES(LineScanner):
    """Hyperspectral Thermal Emission Spectrometer (50° FOV, 512 pixels, 36 Hz)."""

    def __init__(self):
        super().__init__(
            name="HyTES",
            fov=50.0,
            across_track_pixels=512,
            frame_rate=36 * ureg.Hz
        )

class PRISM(LineScanner):
    """Portable Remote Imaging Spectrometer (30.7° FOV, 608 pixels, 176 Hz)."""

    def __init__(self):
        super().__init__(
            name="PRISM",
            fov=30.7,
            across_track_pixels=608,
            frame_rate=176 * ureg.Hz
        )

class MASTER(LineScanner):
    """MODIS/ASTER Airborne Simulator (85.92° FOV, 716 pixels, 25 Hz)."""

    def __init__(self):
        super().__init__(
            name="MASTER",
            fov=85.92,
            across_track_pixels=716,
            frame_rate=25 * ureg.Hz
        )

class GLiHT_VNIR(LineScanner):
    """G-LiHT Visible/Near-Infrared spectrometer (64° FOV, 1600 pixels, 250 Hz)."""

    def __init__(self):
        super().__init__(
            name="G-LiHT VNIR",
            fov=64.0,
            across_track_pixels=1600,
            frame_rate=250 * ureg.Hz
        )

class GLiHT_Thermal(LineScanner):
    """G-LiHT Thermal infrared imager (42.6° FOV, 640 pixels, 50 Hz)."""

    def __init__(self):
        super().__init__(
            name="G-LiHT Thermal",
            fov=42.6,
            across_track_pixels=640,
            frame_rate=50 * ureg.Hz
        )

class GLiHT_SIF(LineScanner):
    """G-LiHT Solar-Induced Fluorescence spectrometer (23.5° FOV, 1600 pixels, 37.6 Hz)."""

    def __init__(self):
        super().__init__(
            name="G-LiHT SIF",
            fov=23.5,
            across_track_pixels=1600,
            frame_rate=37.6 * ureg.Hz
        )


class GCAS_UV_Vis(LineScanner):
    """GEO-CAPE Airborne Simulator UV-Vis spectrometer (45° FOV, 1024 pixels, 12 Hz)."""

    def __init__(self):
        super().__init__(
            name="GCAS UV-Vis Spectrometer",
            fov=45.0,
            across_track_pixels=1024,
            frame_rate=12.0 * ureg.Hz
        )

class GCAS_VNIR(LineScanner):
    """GEO-CAPE Airborne Simulator VNIR spectrometer (70° FOV, 1024 pixels, 12 Hz)."""

    def __init__(self):
        super().__init__(
            name="GCAS Visible Near-Infrared (VNIR) Spectrometer",
            fov=70.0,
            across_track_pixels=1024,
            frame_rate=12.0* ureg.Hz
        )

class eMAS(LineScanner):
    """Enhanced MODIS Airborne Simulator (85.92° FOV, 716 pixels, 6.25 Hz)."""

    def __init__(self):
        super().__init__(
            name="eMAS",
            fov=85.92,  # degrees
            across_track_pixels=716,
            frame_rate=6.25 * ureg.Hz
        )

class PICARD(LineScanner):
    """PICARD imaging spectrometer (50° FOV, 412 pixels, 100 Hz)."""

    def __init__(self):
        super().__init__(
            name="PICARD",
            fov=50.0,  # degrees
            across_track_pixels=412,  # Specific pixel count not provided
            frame_rate=100 * ureg.Hz  # Specific frame rate not provided
        )

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

    if sensor_type not in SENSOR_REGISTRY:
        raise ValueError(f"Unknown sensor type: {sensor_type}")
    return SENSOR_REGISTRY[sensor_type]()


SENSOR_REGISTRY: Dict[str, Type[Sensor]] = {
    "AVIRISClassic": AVIRISClassic,
    "AVIRISNextGen": AVIRISNextGen,
    "AVIRIS3": AVIRIS3,
    "AVIRIS4": AVIRIS4,
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
