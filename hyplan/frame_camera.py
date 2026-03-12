from typing import List, Tuple, Dict
from pint import Quantity
import numpy as np

from .terrain import ray_terrain_intersection
from .units import ureg
from .sensors import Sensor

class FrameCamera(Sensor):
    """Class to represent a frame camera."""

    def __init__(
        self,
        name: str,
        sensor_width: Quantity,  # mm
        sensor_height: Quantity,  # mm
        focal_length: Quantity,  # mm
        resolution_x: int,
        resolution_y: int,
        frame_rate: Quantity,  # Hz
        f_speed: float
    ):
        super().__init__(name)

        # Validate and convert units
        self.sensor_width = self._validate_quantity(sensor_width, ureg.mm)
        self.sensor_height = self._validate_quantity(sensor_height, ureg.mm)
        self.focal_length = self._validate_quantity(focal_length, ureg.mm)
        self.frame_rate = self._validate_quantity(frame_rate, ureg.Hz)

        # Validate integer resolution values
        if not isinstance(resolution_x, int) or not isinstance(resolution_y, int):
            raise TypeError(f"Resolution values must be integers, got ({type(resolution_x)}, {type(resolution_y)})")
        
        self.resolution_x = resolution_x
        self.resolution_y = resolution_y
        self.f_speed = f_speed  # Floating-point f-number

    @property
    def fov_x(self) -> float:
        """Calculate horizontal Field of View (FoV) in degrees."""
        return 2 * np.degrees(np.arctan((self.sensor_width / (2 * self.focal_length)).magnitude))

    @property
    def fov_y(self) -> float:
        """Calculate vertical Field of View (FoV) in degrees."""
        return 2 * np.degrees(np.arctan((self.sensor_height / (2 * self.focal_length)).magnitude))

    def ground_sample_distance(self, altitude: Quantity) -> Dict[str, Quantity]:
        """
        Calculate the ground sample distance (GSD) at nadir for a given altitude.

        Args:
            altitude (Quantity): The altitude of the camera in meters.

        Returns:
            Dict[str, Quantity]: Ground sample distances in meters for both x (across-track) and y (along-track).
        """
        altitude = self._validate_quantity(altitude, ureg.meter)

        return {
            "x": (2 * altitude * np.tan(np.radians(self.fov_x / (2 * self.resolution_x)))),
            "y": (2 * altitude * np.tan(np.radians(self.fov_y / (2 * self.resolution_y))))
        }

    def altitude_for_ground_sample_distance(self, gsd_x: Quantity, gsd_y: Quantity) -> Quantity:
        """
        Calculate the required altitude for a given ground sample distance (GSD) at nadir.

        Args:
            gsd_x (Quantity): Desired ground sample distance in meters along the x-axis (across-track).
            gsd_y (Quantity): Desired ground sample distance in meters along the y-axis (along-track).

        Returns:
            Quantity: The required altitude in meters.
        """
        gsd_x = self._validate_quantity(gsd_x, ureg.meter)
        gsd_y = self._validate_quantity(gsd_y, ureg.meter)

        return max(
            gsd_x / (2 * np.tan(np.radians(self.fov_x / (2 * self.resolution_x)))),
            gsd_y / (2 * np.tan(np.radians(self.fov_y / (2 * self.resolution_y))))
        )

    def footprint_at(self, altitude: Quantity) -> Dict[str, Quantity]:
        """Calculate the footprint dimensions (m) for a given altitude."""
        altitude = self._validate_quantity(altitude, ureg.meter)
        return {
            "width": 2 * altitude * np.tan(np.radians(self.fov_x / 2)),
            "height": 2 * altitude * np.tan(np.radians(self.fov_y / 2)),
        }

    def critical_ground_speed(self, altitude: Quantity) -> Quantity:
        """
        Calculate the maximum ground speed (m/s) to maintain proper along-track sampling.

        Args:
            altitude (Quantity): Altitude of the camera in meters.

        Returns:
            Quantity: Maximum allowable ground speed in meters per second.
        """
        altitude = self._validate_quantity(altitude, ureg.meter)
        pixel_size = self.ground_sample_distance(altitude)["y"]  # Along-track GSD
        frame_period = (1 / self.frame_rate).to(ureg.s)
        return pixel_size / frame_period

    def _validate_quantity(self, value: Quantity, expected_unit: Quantity) -> Quantity:
        """Validates and converts a quantity to the expected unit."""
        if not isinstance(value, Quantity):
            raise TypeError(f"Expected a pint.Quantity for {expected_unit}, but got {type(value)}.")
        return value.to(expected_unit)


    @staticmethod
    def footprint_corners(
        lat: float,
        lon: float,
        altitude: float,
        fov_x: float,
        fov_y: float,
        dem_file: str
    ) -> List[Tuple[Quantity, Quantity, Quantity]]:
        """
        Calculate the latitude, longitude, and altitude of the four corners of a FrameCamera's ground footprint.

        Args:
            lat (Quantity): Latitude of the FrameCamera in degrees.
            lon (Quantity): Longitude of the FrameCamera in degrees.
            altitude (Quantity): Altitude of the FrameCamera in meters.
            fov_x (Quantity): Horizontal Field of View (FoV) of the camera in degrees.
            fov_y (Quantity): Vertical Field of View (FoV) of the camera in degrees.
            dem_file (str): Path to the DEM file for terrain elevation data.

        Returns:
            List[Tuple[Quantity, Quantity, Quantity]]: A list of four tuples, each containing the latitude, 
                longitude, and altitude of a corner point of the ground footprint.
        """
        # Calculate the offsets in azimuth for the four corners
        azimuths = [45, 135, 225, 315]  # Diagonal directions for corners

        # Calculate the tilt (depression) angle to each corner
        half_fov_x_rad = np.radians(fov_x / 2)
        half_fov_y_rad = np.radians(fov_y / 2)
        corner_tilt = np.degrees(np.arctan(np.sqrt(
            np.tan(half_fov_x_rad)**2 + np.tan(half_fov_y_rad)**2
        )))

        # Calculate corner points using ray-terrain intersection
        corners = []
        for azimuth in azimuths:
            corner_lat, corner_lon, corner_alt = ray_terrain_intersection(
                lat, lon, altitude, np.array([azimuth]), np.array([corner_tilt]),
                dem_file=dem_file
            )
            corners.append((float(corner_lat[0]), float(corner_lon[0]), float(corner_alt[0])))

        return corners

# Example usage
if __name__ == "__main__":
    lat = ureg.Quantity(34.0, "degree")
    lon = ureg.Quantity(-117.0, "degree")
    altitude = ureg.Quantity(5000, "meter")
    fov_x = ureg.Quantity(36.0, "degree")
    fov_y = ureg.Quantity(24.0, "degree")
    dem_file = "path/to/dem/file"

    corners = FrameCamera.footprint_corners(lat, lon, altitude, fov_x, fov_y, dem_file)
    for idx, (corner_lat, corner_lon, corner_alt) in enumerate(corners):
        print(f"Corner {idx + 1}: Latitude={corner_lat:.6f}, Longitude={corner_lon:.6f}, Altitude={corner_alt:.2f} m")
