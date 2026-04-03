"""Side-looking Synthetic Aperture Radar (SAR) sensor models.

Models slant-range geometry for stripmap SAR instruments where the swath
is offset from nadir, defined by near-range and far-range incidence angles.
Pre-configured models are provided for UAVSAR in its L-band, P-band
(AirMOSS), and Ka-band (GLISTIN-A) configurations.

References
----------
Hensley, S. et al. (2008). Status of a UAVSAR designed for repeat pass
interferometry for deformation measurements. *IEEE Aerospace Conference
Proceedings*, 1-8. doi:10.1109/AERO.2008.4526385
"""

import json
import os
from dataclasses import dataclass
from typing import List, Optional, Union

import numpy as np
from pint import Quantity
from shapely.geometry import Polygon, shape
from shapely import STRtree

from .units import ureg
from .sensors import Sensor
from .exceptions import HyPlanValueError

__all__ = [
    "SidelookingRadar",
    "UAVSAR_Lband",
    "UAVSAR_Pband",
    "UAVSAR_Kaband",
    "RadarExclusionConflict",
    "check_lband_radar_exclusions",
]

_DEFAULT_EXCLUSION_ZONES_PATH = os.path.join(
    os.path.dirname(__file__), "data", "faa_radar_exclusion_zones.geojson"
)


@dataclass
class RadarExclusionConflict:
    """A detected conflict between a UAVSAR swath and an FAA L-Band radar exclusion zone.

    Attributes:
        radar_name: Name of the FAA radar site.
        swath_index: Index of the conflicting swath in the input list.
        intersection: Shapely geometry of the overlap between the swath and exclusion zone.
        exclusion_zone: Shapely Polygon of the full exclusion zone boundary.
    """

    radar_name: str
    swath_index: int
    intersection: object  # Shapely geometry
    exclusion_zone: Polygon


def check_lband_radar_exclusions(
    swath_polygons: Union[Polygon, List[Polygon]],
    geojson: Union[str, dict, None] = None,
) -> List[RadarExclusionConflict]:
    """Check UAVSAR swath polygons against FAA L-Band radar exclusion zones.

    UAVSAR L-Band swaths must remain outside a 10 nautical mile radius of each
    FAA long-range L-Band radar site.  The exclusion zone polygons are
    pre-computed 10 NMI circles stored in a GeoJSON FeatureCollection.

    Args:
        swath_polygons: A single Shapely Polygon or a list of Shapely Polygons
            representing UAVSAR swath footprints (e.g. from
            :func:`~hyplan.swath.generate_swath_polygon`).
        geojson: Exclusion zone data.  One of:

            * ``None`` — load the bundled
              ``hyplan/data/faa_lband_radar_exclusion_zones.geojson`` file.
            * ``str`` — path to a GeoJSON FeatureCollection file on disk.
            * ``dict`` — an already-parsed GeoJSON FeatureCollection.

    Returns:
        List of :class:`RadarExclusionConflict`, one for each swath/zone pair
        that intersects.  An empty list means no conflicts.

    Raises:
        FileNotFoundError: If *geojson* is ``None`` and the bundled data file
            does not exist, or if a path string is given that does not exist.
        ValueError: If the GeoJSON is not a valid FeatureCollection.
    """
    # Normalise input to a list
    if isinstance(swath_polygons, Polygon):
        swath_polygons = [swath_polygons]

    # Load exclusion zone GeoJSON
    if geojson is None:
        if not os.path.exists(_DEFAULT_EXCLUSION_ZONES_PATH):
            raise FileNotFoundError(
                "Bundled FAA radar exclusion zone data not found at "
                f"{_DEFAULT_EXCLUSION_ZONES_PATH!r}. "
                "Pass a geojson= path or dict to check_lband_radar_exclusions()."
            )
        with open(_DEFAULT_EXCLUSION_ZONES_PATH) as f:
            geojson = json.load(f)
    elif isinstance(geojson, str):
        if not os.path.exists(geojson):
            raise FileNotFoundError(f"GeoJSON file not found: {geojson!r}")
        with open(geojson) as f:
            geojson = json.load(f)

    if geojson.get("type") != "FeatureCollection":
        raise ValueError("geojson must be a GeoJSON FeatureCollection")

    # Parse exclusion zone polygons
    zone_names: List[str] = []
    zone_geoms: List[Polygon] = []
    for feature in geojson.get("features", []):
        geom = shape(feature["geometry"])
        if not isinstance(geom, Polygon):
            continue
        name = feature.get("properties", {}).get("name", "Unknown")
        zone_names.append(name)
        zone_geoms.append(geom)

    if not zone_geoms:
        return []

    # Spatial index over exclusion zones for fast candidate lookup
    tree = STRtree(zone_geoms)

    conflicts: List[RadarExclusionConflict] = []
    for swath_idx, swath in enumerate(swath_polygons):
        candidate_indices = tree.query(swath, predicate="intersects")
        for zone_idx in candidate_indices:
            intersection = swath.intersection(zone_geoms[zone_idx])
            if intersection.is_empty:
                continue
            conflicts.append(
                RadarExclusionConflict(
                    radar_name=zone_names[zone_idx],
                    swath_index=swath_idx,
                    intersection=intersection,
                    exclusion_zone=zone_geoms[zone_idx],
                )
            )

    return conflicts


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
            raise HyPlanValueError("look_direction must be 'left' or 'right'")
        self.look_direction = look_direction

        if self.near_range_angle >= self.far_range_angle:
            raise HyPlanValueError("near_range_angle must be less than far_range_angle")

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
            # Swath on port side: negative angles (left of track)
            return -self.far_range_angle, -self.near_range_angle
        else:
            # Swath on starboard side: positive angles (right of track)
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
