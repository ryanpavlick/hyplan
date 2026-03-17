import numpy as np
from pint import Quantity
from typing import Optional

from .units import ureg
from .sensors import Sensor

__all__ = [
    "SidelookingRadar",
    "UAVSAR_Lband",
    "UAVSAR_Pband",
    "UAVSAR_Kaband",
]


class SidelookingRadar(Sensor):
    """
    Represents a side-looking Synthetic Aperture Radar (SAR).

    Models the slant-range geometry where the swath is offset from nadir,
    defined by near-range and far-range incidence angles. Supports stripmap
    SAR instruments like UAVSAR.

    The swath lies entirely on one side of the flight track (typically left).
    """

    def __init__(
        self,
        name: str,
        frequency: Quantity,            # GHz
        bandwidth: Quantity,            # MHz
        near_range_angle: float,        # degrees from nadir
        far_range_angle: float,         # degrees from nadir
        azimuth_resolution: Quantity,   # meters (single-look)
        polarization: str,              # e.g. "quad-pol", "HH"
        look_direction: str = "left",   # "left" or "right"
        peak_power: Optional[Quantity] = None,  # watts
        antenna_length: Optional[Quantity] = None,  # meters
    ):
        super().__init__(name)

        self.frequency = self._validate_quantity(frequency, ureg.GHz)
        self.bandwidth = self._validate_quantity(bandwidth, ureg.MHz)
        self.near_range_angle = float(near_range_angle)
        self.far_range_angle = float(far_range_angle)
        self.azimuth_resolution = self._validate_quantity(azimuth_resolution, ureg.meter)
        self.polarization = polarization
        self.peak_power = self._validate_quantity(peak_power, ureg.watt) if peak_power else None
        self.antenna_length = self._validate_quantity(antenna_length, ureg.meter) if antenna_length else None

        if look_direction not in ("left", "right"):
            raise ValueError("look_direction must be 'left' or 'right'")
        self.look_direction = look_direction

        if self.near_range_angle >= self.far_range_angle:
            raise ValueError("near_range_angle must be less than far_range_angle")

    @property
    def wavelength(self) -> Quantity:
        """Radar wavelength derived from frequency."""
        c = 299792458 * ureg.meter / ureg.second
        return (c / self.frequency).to(ureg.meter)

    @property
    def range_resolution(self) -> Quantity:
        """Slant-range resolution from bandwidth: c / (2 * B)."""
        c = 299792458 * ureg.meter / ureg.second
        return (c / (2 * self.bandwidth)).to(ureg.meter)

    @property
    def half_angle(self) -> float:
        """
        Effective half-angle for swath calculations.

        For a side-looking radar this is the angular extent from the
        swath center to either edge, i.e. half the angular swath width.
        """
        return (self.far_range_angle - self.near_range_angle) / 2.0

    @property
    def swath_center_angle(self) -> float:
        """Incidence angle at swath center (degrees from nadir)."""
        return (self.near_range_angle + self.far_range_angle) / 2.0

    def swath_width(self, altitude_agl: Quantity) -> Quantity:
        """
        Ground swath width for a given altitude AGL.

        Computed from the difference in ground range at near and far
        incidence angles.

        Args:
            altitude_agl: Flight altitude above ground level.

        Returns:
            Swath width in meters.
        """
        altitude_agl = self._validate_quantity(altitude_agl, ureg.meter)
        h = altitude_agl.magnitude
        near_ground = h * np.tan(np.radians(self.near_range_angle))
        far_ground = h * np.tan(np.radians(self.far_range_angle))
        return (far_ground - near_ground) * ureg.meter

    def near_range_ground_distance(self, altitude_agl: Quantity) -> Quantity:
        """Ground distance from nadir to near edge of swath."""
        altitude_agl = self._validate_quantity(altitude_agl, ureg.meter)
        return altitude_agl.magnitude * np.tan(np.radians(self.near_range_angle)) * ureg.meter

    def far_range_ground_distance(self, altitude_agl: Quantity) -> Quantity:
        """Ground distance from nadir to far edge of swath."""
        altitude_agl = self._validate_quantity(altitude_agl, ureg.meter)
        return altitude_agl.magnitude * np.tan(np.radians(self.far_range_angle)) * ureg.meter

    def ground_range_resolution(self, altitude_agl: Quantity, incidence_angle: float = None) -> Quantity:
        """
        Ground-range resolution at a given incidence angle.

        ground_range_res = slant_range_res / sin(incidence_angle)

        Args:
            altitude_agl: Flight altitude above ground level.
            incidence_angle: Incidence angle in degrees. Defaults to swath center.

        Returns:
            Ground-range resolution in meters.
        """
        if incidence_angle is None:
            incidence_angle = self.swath_center_angle
        return (self.range_resolution / np.sin(np.radians(incidence_angle))).to(ureg.meter)

    def ground_sample_distance(self, altitude_agl: Quantity) -> dict:
        """
        Ground sample distance at near range, center, and far range.

        Args:
            altitude_agl: Flight altitude above ground level.

        Returns:
            Dict with 'near_range', 'center', 'far_range' ground-range
            resolutions and 'azimuth' resolution.
        """
        return {
            "near_range": self.ground_range_resolution(altitude_agl, self.near_range_angle),
            "center": self.ground_range_resolution(altitude_agl, self.swath_center_angle),
            "far_range": self.ground_range_resolution(altitude_agl, self.far_range_angle),
            "azimuth": self.azimuth_resolution,
        }

    def slant_range(self, altitude_agl: Quantity, incidence_angle: float = None) -> Quantity:
        """
        Slant range distance to target at given incidence angle.

        Args:
            altitude_agl: Flight altitude above ground level.
            incidence_angle: Incidence angle in degrees. Defaults to swath center.

        Returns:
            Slant range in meters.
        """
        altitude_agl = self._validate_quantity(altitude_agl, ureg.meter)
        if incidence_angle is None:
            incidence_angle = self.swath_center_angle
        return (altitude_agl / np.cos(np.radians(incidence_angle))).to(ureg.meter)

    def swath_offset_angles(self) -> tuple:
        """
        Return the (port_angle, starboard_angle) for swath polygon generation.

        For a left-looking radar, the swath is on the left (port) side.
        For a right-looking radar, the swath is on the right (starboard) side.

        Returns:
            Tuple of (near_side_angle, far_side_angle) where the sign convention
            matches swath.py: port = left of track, starboard = right of track.
        """
        if self.look_direction == "left":
            return self.near_range_angle, self.far_range_angle
        else:
            return self.near_range_angle, self.far_range_angle

    def interferometric_line_spacing(self, altitude_agl: Quantity, overlap_fraction: float = 0.0) -> Quantity:
        """
        Compute the required spacing between parallel flight lines for
        interferometric or mosaicking coverage.

        Args:
            altitude_agl: Flight altitude above ground level.
            overlap_fraction: Fraction of swath overlap (0.0 = edge-to-edge,
                0.5 = 50% overlap). For InSAR mosaics, typically 0.1-0.2.

        Returns:
            Line spacing in meters (center-to-center).
        """
        sw = self.swath_width(altitude_agl)
        return sw * (1.0 - overlap_fraction)


# ── UAVSAR Instrument Definitions ──────────────────────────────────────────


class UAVSAR_Lband(SidelookingRadar):
    """
    NASA/JPL UAVSAR L-band fully polarimetric SAR.

    Platform: Gulfstream III (C-20A)
    Typical altitude: ~12,500 m (41,000 ft)
    """
    def __init__(self):
        super().__init__(
            name="UAVSAR L-band",
            frequency=1.2575 * ureg.GHz,
            bandwidth=80 * ureg.MHz,
            near_range_angle=22.0,
            far_range_angle=65.0,
            azimuth_resolution=0.8 * ureg.meter,
            polarization="quad-pol",
            look_direction="left",
            peak_power=3100 * ureg.watt,
        )


class UAVSAR_Pband(SidelookingRadar):
    """
    NASA/JPL UAVSAR P-band SAR (AirMOSS configuration).

    Platform: Gulfstream III (C-20A)
    Typical altitude: ~12,500 m (41,000 ft)
    """
    def __init__(self):
        super().__init__(
            name="UAVSAR P-band (AirMOSS)",
            frequency=0.430 * ureg.GHz,
            bandwidth=20 * ureg.MHz,
            near_range_angle=25.0,
            far_range_angle=45.0,
            azimuth_resolution=0.8 * ureg.meter,
            polarization="quad-pol",
            look_direction="left",
            peak_power=2000 * ureg.watt,
        )


class UAVSAR_Kaband(SidelookingRadar):
    """
    NASA/JPL GLISTIN-A Ka-band single-pass cross-track interferometric SAR.

    Platform: Gulfstream III (C-20A)
    Typical altitude: ~12,500 m (41,000 ft)
    """
    def __init__(self):
        super().__init__(
            name="UAVSAR Ka-band (GLISTIN-A)",
            frequency=35.66 * ureg.GHz,
            bandwidth=80 * ureg.MHz,
            near_range_angle=15.0,
            far_range_angle=50.0,
            azimuth_resolution=0.25 * ureg.meter,
            polarization="HH",
            look_direction="left",
        )
