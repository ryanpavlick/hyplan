"""Frame (area-array) camera sensor models.

Computes ground footprint, GSD, and along-track sampling from focal
length, sensor dimensions, pixel count, and frame rate. Supports
nadir and oblique viewing geometries, terrain-aware footprint
projection, and multi-camera rig configurations for stereo coverage
planning.
"""

from __future__ import annotations

import warnings
from typing import List, Tuple, Dict
from pint import Quantity, Unit
import numpy as np
from shapely.geometry import Polygon as ShapelyPolygon

from ..terrain import ray_terrain_intersection
from ..units import ureg
from ._base import Sensor
from ..exceptions import HyPlanTypeError, HyPlanValueError

__all__ = [
    "FrameCamera",
    "MultiCameraRig",
]


class FrameCamera(Sensor):
    """
    A frame (area-array) camera sensor.

    Unlike line scanners, frame cameras capture a full 2D image per frame.
    The footprint is determined by both horizontal and vertical fields of view.

    Args:
        name (str): Sensor name.
        sensor_width (Quantity): Physical sensor width in mm.
        sensor_height (Quantity): Physical sensor height in mm.
        focal_length (Quantity): Lens focal length in mm.
        resolution_x (int): Number of pixels across-track (horizontal).
        resolution_y (int): Number of pixels along-track (vertical).
        frame_rate (Quantity): Frame acquisition rate in Hz.
        f_speed (float): Lens f-number (focal length / aperture diameter).
    """

    def __init__(
        self,
        name: str,
        sensor_width: Quantity,  # mm
        sensor_height: Quantity,  # mm
        focal_length: Quantity,  # mm
        resolution_x: int,
        resolution_y: int,
        frame_rate: Quantity,  # Hz
        f_speed: float,
        tilt_angle: float = 0.0,
        tilt_direction: float = 0.0,
    ):
        super().__init__(name)

        # Validate and convert units
        self.sensor_width = self._validate_quantity(sensor_width, ureg.mm)
        self.sensor_height = self._validate_quantity(sensor_height, ureg.mm)
        self.focal_length = self._validate_quantity(focal_length, ureg.mm)
        self.frame_rate = self._validate_quantity(frame_rate, ureg.Hz)

        # Validate integer resolution values
        if not isinstance(resolution_x, int) or not isinstance(resolution_y, int):
            raise HyPlanTypeError(f"Resolution values must be integers, got ({type(resolution_x)}, {type(resolution_y)})")

        self.resolution_x = resolution_x
        self.resolution_y = resolution_y
        self.f_speed = f_speed  # Floating-point f-number

        # Tilt parameters
        if not (0.0 <= tilt_angle < 90.0):
            raise HyPlanValueError(f"tilt_angle must be in [0, 90), got {tilt_angle}")
        self.tilt_angle = float(tilt_angle)
        self.tilt_direction = float(tilt_direction) % 360.0

        if self.sensor_height > self.sensor_width:
            warnings.warn(
                "sensor_height > sensor_width — camera appears portrait-oriented. "
                "Typically the wider axis is across-track.",
                stacklevel=2,
            )

    @property
    def ifov_x(self) -> float:
        """Instantaneous field of view per pixel across-track (microradians)."""
        pixel_size = self.sensor_width / self.resolution_x
        return (pixel_size / self.focal_length).to_reduced_units().magnitude * 1e6  # type: ignore[no-any-return]

    @property
    def ifov_y(self) -> float:
        """Instantaneous field of view per pixel along-track (microradians)."""
        pixel_size = self.sensor_height / self.resolution_y
        return (pixel_size / self.focal_length).to_reduced_units().magnitude * 1e6  # type: ignore[no-any-return]

    @property
    def fov_x(self) -> float:
        """Calculate horizontal Field of View (FoV) in degrees."""
        return 2 * np.degrees(np.arctan((self.sensor_width / (2 * self.focal_length)).magnitude))  # type: ignore[no-any-return]

    @property
    def fov_y(self) -> float:
        """Calculate vertical Field of View (FoV) in degrees."""
        return 2 * np.degrees(np.arctan((self.sensor_height / (2 * self.focal_length)).magnitude))  # type: ignore[no-any-return]

    def ground_sample_distance(self, altitude_agl: Quantity) -> Dict[str, Quantity]:
        """
        Calculate the ground sample distance (GSD) for a given altitude AGL.

        For nadir cameras, returns ``x`` (across-track) and ``y``
        (along-track) GSD.  For tilted cameras, additionally returns
        ``y_near`` and ``y_far`` showing GSD variation across the frame.

        Args:
            altitude_agl (Quantity): Altitude above ground level in meters.

        Returns:
            Dict[str, Quantity]: Ground sample distances in meters.
        """
        altitude_agl = self._validate_quantity(altitude_agl, ureg.meter)

        if self.tilt_angle == 0.0:
            return {
                "x": (2 * altitude_agl * np.tan(np.radians(self.fov_x / (2 * self.resolution_x)))),  # type: ignore[dict-item,no-any-return]
                "y": (2 * altitude_agl * np.tan(np.radians(self.fov_y / (2 * self.resolution_y))))  # type: ignore[dict-item,no-any-return]
            }

        # Tilted camera: GSD varies along the tilt axis
        tilt_rad = np.radians(self.tilt_angle)
        pixel_x = self.sensor_width / self.resolution_x
        pixel_y = self.sensor_height / self.resolution_y
        f = self.focal_length

        # Center GSD (at principal point)
        cos2_tilt = np.cos(tilt_rad) ** 2
        gsd_x_center = (altitude_agl * pixel_x / (f * np.cos(tilt_rad))).to(ureg.meter)
        gsd_y_center = (altitude_agl * pixel_y / (f * cos2_tilt)).to(ureg.meter)

        # Near edge (less oblique) and far edge (more oblique)
        half_fov_y_rad = np.radians(self.fov_y / 2)
        near_dep = tilt_rad - half_fov_y_rad
        far_dep = tilt_rad + half_fov_y_rad
        cos2_near = np.cos(near_dep) ** 2 if near_dep >= 0 else np.cos(near_dep) ** 2
        cos2_far = np.cos(far_dep) ** 2

        gsd_y_near = (altitude_agl * pixel_y / (f * cos2_near)).to(ureg.meter)
        gsd_y_far = (altitude_agl * pixel_y / (f * cos2_far)).to(ureg.meter)

        return {
            "x": gsd_x_center,
            "y": gsd_y_center,
            "y_near": gsd_y_near,
            "y_far": gsd_y_far,
        }

    def altitude_agl_for_ground_sample_distance(self, gsd_x: Quantity, gsd_y: Quantity) -> Quantity:
        """
        Calculate the required altitude AGL for a given ground sample distance (GSD) at nadir.

        Args:
            gsd_x (Quantity): Desired ground sample distance in meters along the x-axis (across-track).
            gsd_y (Quantity): Desired ground sample distance in meters along the y-axis (along-track).

        Returns:
            Quantity: The required altitude AGL in meters.
        """
        gsd_x = self._validate_quantity(gsd_x, ureg.meter)
        gsd_y = self._validate_quantity(gsd_y, ureg.meter)

        return max(  # type: ignore[return-value,no-any-return]
            gsd_x / (2 * np.tan(np.radians(self.fov_x / (2 * self.resolution_x)))),
            gsd_y / (2 * np.tan(np.radians(self.fov_y / (2 * self.resolution_y))))
        )

    def _tilt_footprint_geometry(self, altitude_agl: Quantity) -> Dict[str, Quantity]:
        """Compute tilted-camera footprint geometry.

        Returns a dict with near/far ground distances along the tilt axis
        and the cross-tilt width at the center slant range.  When
        ``tilt_angle == 0`` this reduces to the symmetric nadir case.
        """
        h = altitude_agl
        tilt_rad = np.radians(self.tilt_angle)

        # Decompose tilt direction into along-track and cross-track components
        dir_rad = np.radians(self.tilt_direction)
        # Along tilt axis: use the FOV component in the tilt direction
        # For forward/backward tilt (0°/180°): along-track FOV = fov_y
        # For side tilt (90°/270°): along-track FOV = fov_x
        cos_dir = np.cos(dir_rad)
        sin_dir = np.sin(dir_rad)
        fov_along = np.sqrt((self.fov_y * cos_dir) ** 2 + (self.fov_x * sin_dir) ** 2)
        fov_cross = np.sqrt((self.fov_x * cos_dir) ** 2 + (self.fov_y * sin_dir) ** 2)

        half_along_rad = np.radians(fov_along / 2)
        half_cross_rad = np.radians(fov_cross / 2)

        # Ground distances from nadir along tilt axis
        near_angle = tilt_rad - half_along_rad
        far_angle = tilt_rad + half_along_rad

        if near_angle >= 0:
            near_ground_dist = h * np.tan(near_angle)
        else:
            # Camera looks past nadir on the near side
            near_ground_dist = -h * np.tan(-near_angle)

        far_ground_dist = h * np.tan(far_angle)
        principal_dist = h * np.tan(tilt_rad)

        # Cross-tilt width at the center slant range
        center_slant = h / np.cos(tilt_rad) if self.tilt_angle > 0 else h
        cross_width = 2 * center_slant * np.tan(half_cross_rad)

        return {
            "near_ground_dist": near_ground_dist,
            "far_ground_dist": far_ground_dist,
            "principal_ground_dist": principal_dist,
            "cross_width": cross_width,
            "fov_along": fov_along,
            "fov_cross": fov_cross,
        }

    def footprint_at(self, altitude_agl: Quantity) -> Dict[str, Quantity]:
        """Calculate the footprint dimensions (m) for a given altitude AGL.

        For nadir cameras (``tilt_angle == 0``), returns ``width`` and
        ``height``.  For tilted cameras, additionally returns
        ``height_near`` and ``height_far`` (distances from the principal
        point to the near/far edges along the tilt axis).
        """
        altitude_agl = self._validate_quantity(altitude_agl, ureg.meter)

        if self.tilt_angle == 0.0:
            return {
                "width": 2 * altitude_agl * np.tan(np.radians(self.fov_x / 2)),
                "height": 2 * altitude_agl * np.tan(np.radians(self.fov_y / 2)),
            }

        geo = self._tilt_footprint_geometry(altitude_agl)
        height_total = geo["far_ground_dist"] - geo["near_ground_dist"]
        return {
            "width": geo["cross_width"],
            "height": height_total,
            "height_near": geo["principal_ground_dist"] - geo["near_ground_dist"],
            "height_far": geo["far_ground_dist"] - geo["principal_ground_dist"],
        }

    def swath_width(self, altitude_agl: Quantity) -> Quantity:
        """Across-track swath width at a given altitude AGL.

        This is the ``width`` component of :meth:`footprint_at`, provided
        for API compatibility with :class:`LineScanner` and
        :func:`flight_box.box_around_center_line`.
        """
        return self.footprint_at(altitude_agl)["width"]

    def image_scale(self, altitude_agl: Quantity) -> float:
        """Image scale denominator (1:N) at a given altitude AGL.

        Args:
            altitude_agl: Altitude above ground level.

        Returns:
            Scale denominator N such that the image scale is 1:N.
        """
        altitude_agl = self._validate_quantity(altitude_agl, ureg.meter)
        return (altitude_agl / self.focal_length).to_reduced_units().magnitude  # type: ignore[no-any-return]

    def altitude_for_scale(self, scale_denominator: float) -> Quantity:
        """Altitude AGL required for a given image scale (1:N).

        Args:
            scale_denominator: The denominator N of the desired image scale.

        Returns:
            Required altitude AGL in meters.
        """
        return (self.focal_length * scale_denominator).to(ureg.meter)  # type: ignore[return-value,no-any-return]

    def focal_length_for_gsd(self, altitude_agl: Quantity, target_gsd: Quantity) -> Quantity:
        """Required focal length for a target GSD at a given altitude.

        Useful for zoom lenses where the focal length can be adjusted.

        Args:
            altitude_agl: Altitude above ground level.
            target_gsd: Desired ground sample distance.

        Returns:
            Required focal length in mm.
        """
        altitude_agl = self._validate_quantity(altitude_agl, ureg.meter)
        target_gsd = self._validate_quantity(target_gsd, ureg.meter)
        pixel_size = self.sensor_width / self.resolution_x
        return (altitude_agl * pixel_size / target_gsd).to(ureg.mm)  # type: ignore[return-value,no-any-return]

    def line_spacing(self, altitude_agl: Quantity, sidelap_pct: float = 60.0) -> Quantity:
        """Flight line spacing from sidelap percentage.

        Args:
            altitude_agl: Altitude above ground level.
            sidelap_pct: Desired sidelap between adjacent flight lines (0-100).

        Returns:
            Center-to-center distance between parallel flight lines in meters.
        """
        footprint = self.footprint_at(altitude_agl)
        return footprint["width"] * (1 - sidelap_pct / 100)  # type: ignore[return-value,no-any-return]

    def trigger_distance(self, altitude_agl: Quantity, overlap_pct: float = 80.0) -> Quantity:
        """Along-track distance between camera exposures from overlap percentage.

        Args:
            altitude_agl: Altitude above ground level.
            overlap_pct: Desired forward overlap between successive images (0-100).

        Returns:
            Distance between exposure centers in meters.
        """
        footprint = self.footprint_at(altitude_agl)
        return footprint["height"] * (1 - overlap_pct / 100)  # type: ignore[return-value,no-any-return]

    def trigger_interval(self, altitude_agl: Quantity, ground_speed: Quantity,
                         overlap_pct: float = 80.0) -> Quantity:
        """Time between camera triggers for a given speed and overlap.

        Args:
            altitude_agl: Altitude above ground level.
            ground_speed: Aircraft ground speed.
            overlap_pct: Desired forward overlap between successive images (0-100).

        Returns:
            Time interval between triggers in seconds.
        """
        ground_speed = self._validate_quantity(ground_speed, ureg.meter / ureg.second)
        dist = self.trigger_distance(altitude_agl, overlap_pct)
        return (dist / ground_speed).to(ureg.second)  # type: ignore[return-value,no-any-return]

    def coverage_buffer(self, altitude_agl: Quantity, overlap_pct: float = 80.0,
                        n_frames: int = 4) -> Quantity:
        """Extra distance beyond AOI boundary to ensure full edge coverage.

        Camera triggering should begin this distance before the AOI entry
        and continue this distance past the AOI exit.

        Args:
            altitude_agl: Altitude above ground level.
            overlap_pct: Desired forward overlap between successive images (0-100).
            n_frames: Number of extra frames beyond the boundary (default 4).

        Returns:
            Buffer distance in meters.
        """
        return self.trigger_distance(altitude_agl, overlap_pct) * n_frames  # type: ignore[return-value,no-any-return]

    def critical_ground_speed(self, altitude_agl: Quantity) -> Quantity:
        """
        Calculate the maximum ground speed (m/s) to maintain proper along-track sampling.

        Args:
            altitude_agl (Quantity): Altitude above ground level in meters.

        Returns:
            Quantity: Maximum allowable ground speed in meters per second.
        """
        altitude_agl = self._validate_quantity(altitude_agl, ureg.meter)
        pixel_size = self.ground_sample_distance(altitude_agl)["y"]  # Along-track GSD
        frame_period = (1 / self.frame_rate).to(ureg.s)
        return pixel_size / frame_period  # type: ignore[return-value,no-any-return]

    def base_height_ratio(self, altitude_agl: Quantity, overlap_pct: float = 80.0) -> float:
        """Base-to-height ratio for stereo photogrammetry.

        B/H is the ratio of the baseline (distance between successive
        exposure centres) to the flying height.  Larger values give
        better vertical accuracy but more occlusion.

        Args:
            altitude_agl: Altitude above ground level.
            overlap_pct: Forward overlap between successive images (0-100).

        Returns:
            Dimensionless B/H ratio.
        """
        altitude_agl = self._validate_quantity(altitude_agl, ureg.meter)
        baseline = self.trigger_distance(altitude_agl, overlap_pct)
        return (baseline / altitude_agl).to_reduced_units().magnitude  # type: ignore[no-any-return]

    def vertical_accuracy(self, altitude_agl: Quantity, overlap_pct: float = 80.0,
                          sigma_parallax: float = 0.5) -> Quantity:
        """Estimated vertical accuracy from stereo overlap.

        Uses the photogrammetric relation ``σ_z = (H / B) × σ_p × GSD_y``
        where ``σ_p`` is the parallax measurement error in pixels.

        Args:
            altitude_agl: Altitude above ground level.
            overlap_pct: Forward overlap (0-100).
            sigma_parallax: Parallax measurement error in pixels (default 0.5).

        Returns:
            Vertical accuracy (σ_z) in meters.
        """
        altitude_agl = self._validate_quantity(altitude_agl, ureg.meter)
        bh = self.base_height_ratio(altitude_agl, overlap_pct)
        gsd_y = self.ground_sample_distance(altitude_agl)["y"]
        return (sigma_parallax * gsd_y / bh).to(ureg.meter)  # type: ignore[return-value,no-any-return]

    def range_accuracy(self, altitude_agl: Quantity, baseline: Quantity,
                       sigma_q: float | None = None) -> Quantity:
        """Stereo range accuracy using the range-error formula.

        ``σ_R = R² × σ_q / B`` where *R* is the slant range from the
        camera to the ground, *B* is the baseline, and *σ_q* is the
        angular measurement uncertainty in radians.  This is the model
        used in Donnellan et al. (2025) for QUAKES-I.

        Args:
            altitude_agl: Altitude above ground level.
            baseline: Distance between stereo exposure centres.
            sigma_q: Angular uncertainty in radians.  If *None*, derived
                from ``ifov_y / 3`` (sub-pixel matching at ⅓ pixel).

        Returns:
            Range accuracy (σ_R) in meters.
        """
        altitude_agl = self._validate_quantity(altitude_agl, ureg.meter)
        baseline = self._validate_quantity(baseline, ureg.meter)
        if sigma_q is None:
            # IFOV is in microradians; convert to radians, then 1/3 pixel
            sigma_q = (self.ifov_y * 1e-6) / 3.0
        tilt_rad = np.radians(self.tilt_angle)
        slant_range = altitude_agl / np.cos(tilt_rad) if self.tilt_angle > 0 else altitude_agl
        return (slant_range ** 2 * sigma_q / baseline).to(ureg.meter)  # type: ignore[return-value,no-any-return]

    def _validate_quantity(self, value: Quantity, expected_unit: Quantity | Unit) -> Quantity:
        """Validates and converts a quantity to the expected unit."""
        if not isinstance(value, Quantity):
            raise HyPlanTypeError(f"Expected a pint.Quantity for {expected_unit}, but got {type(value)}.")
        return value.to(expected_unit)  # type: ignore[return-value]

    @staticmethod
    def _corner_rotation(tilt_angle, tilt_direction, cross_track_offset):
        """Build the rotation matrix for camera corner ray projection."""
        def _rx(a):
            c, s = np.cos(a), np.sin(a)
            return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])

        def _ry(a):
            c, s = np.cos(a), np.sin(a)
            return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

        def _rz(a):
            c, s = np.cos(a), np.sin(a)
            return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

        tilt_rad = np.radians(tilt_angle)
        dir_rad = np.radians(tilt_direction)
        cross_rad = np.radians(cross_track_offset)
        return _rz(dir_rad) @ _ry(cross_rad) @ _rx(-tilt_rad) @ _rz(-dir_rad)

    def _edge_rays(self, edge_points: int) -> np.ndarray:
        """Ray directions in camera space along the sensor perimeter.

        Returns an ``(N, 3)`` array where ``N = 4 * edge_points``.
        Points are distributed evenly along each of the four sensor
        edges (top, right, bottom, left).
        """
        hfx = np.tan(np.radians(self.fov_x / 2))
        hfy = np.tan(np.radians(self.fov_y / 2))
        corners = [(-hfx, -hfy), (hfx, -hfy), (hfx, hfy), (-hfx, hfy)]
        rays = []
        for i in range(4):
            x0, y0 = corners[i]
            x1, y1 = corners[(i + 1) % 4]
            for t in np.linspace(0, 1, edge_points, endpoint=False):
                rays.append([x0 + t * (x1 - x0), y0 + t * (y1 - y0), 1.0])
        return np.array(rays)  # type: ignore[no-any-return]

    def ground_footprint(
        self,
        altitude_agl: Quantity,
        cross_track_offset: float = 0.0,
        *,
        edge_points: int = 10,
        lat: float | None = None,
        lon: float | None = None,
        altitude_msl: float | None = None,
        heading: float = 0.0,
        dem_file: str | None = None,
    ) -> ShapelyPolygon:
        """Project the sensor perimeter onto the ground as a Shapely Polygon.

        Points are distributed along each sensor edge (controlled by
        *edge_points*) so the polygon faithfully represents footprint
        curvature, especially over terrain.

        Operates in two modes:

        **Flat-ground mode** (default): When ``lat``, ``lon``, and
        ``altitude_msl`` are all *None*.  Returns a 3-D Shapely Polygon
        with coordinates ``(x_cross, y_along, 0)`` in meters, origin
        at the nadir point.

        **Terrain mode**: When ``lat``, ``lon``, and ``altitude_msl``
        are provided.  Rays are intersected with the DEM via
        :func:`~hyplan.terrain.ray_terrain_intersection`.  Returns a
        3-D Shapely Polygon with coordinates ``(lon, lat, elevation)``.
        The *altitude_agl* parameter is ignored in this mode.

        Args:
            altitude_agl: Altitude above ground level (used in flat mode).
            cross_track_offset: Additional cross-track angular offset
                in degrees (e.g. for cameras in a multi-camera rig).
            edge_points: Number of points per sensor edge (default 10).
                Use 2 for a simple 4-corner quadrilateral.
            lat: Camera latitude in degrees (terrain mode).
            lon: Camera longitude in degrees (terrain mode).
            altitude_msl: Camera altitude MSL in meters (terrain mode).
            heading: Aircraft heading in degrees from north (terrain mode).
            dem_file: Path to DEM file.  If *None* and terrain mode is
                active, a DEM is downloaded automatically.

        Returns:
            A :class:`shapely.geometry.Polygon` with 3-D coordinates.
        """
        terrain_mode = lat is not None and lon is not None and altitude_msl is not None

        rays_cam = self._edge_rays(edge_points)
        R = self._corner_rotation(self.tilt_angle, self.tilt_direction,
                                   cross_track_offset)
        rays = (R @ rays_cam.T).T
        valid = rays[:, 2] > 0

        if terrain_mode:
            if not np.any(valid):
                return ShapelyPolygon()
            # Intersect each ray individually to avoid cross-ray
            # interference in the global slant-range array used by
            # ray_terrain_intersection (rays with very different
            # depression angles produce misleading results when batched).
            coords = []
            for ray in rays[valid]:
                depression = np.degrees(np.arctan2(
                    ray[2], np.sqrt(ray[0] ** 2 + ray[1] ** 2)))
                azimuth_cam = np.degrees(np.arctan2(ray[0], ray[1]))
                azimuth_geo = (heading + azimuth_cam) % 360
                clat, clon, celev = ray_terrain_intersection(
                    lat, lon, altitude_msl,  # type: ignore[arg-type]
                    np.array([azimuth_geo]), np.array([90.0 - depression]),
                    dem_file=dem_file,
                )
                if not np.isnan(clat[0]):
                    coords.append((float(clon[0]), float(clat[0]), float(celev[0])))
        else:
            altitude_agl = self._validate_quantity(altitude_agl, ureg.meter)
            h = altitude_agl.magnitude
            t = np.where(valid, h / rays[:, 2], np.nan)
            xs = t * rays[:, 0]
            ys = t * rays[:, 1]
            coords = [
                (float(x), float(y), 0.0)
                for x, y, v in zip(xs, ys, valid) if v
            ]

        return ShapelyPolygon(coords) if len(coords) >= 3 else ShapelyPolygon()

    def ground_footprint_corners(self, *args, **kwargs):
        """Deprecated — use :meth:`ground_footprint` instead."""
        warnings.warn(
            "ground_footprint_corners() is deprecated, use ground_footprint()",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.ground_footprint(*args, **kwargs)

    @staticmethod
    def footprint_corners(
        lat: float,
        lon: float,
        altitude_msl: float,
        fov_x: float,
        fov_y: float,
        dem_file: str,
        tilt_angle: float = 0.0,
        tilt_direction: float = 0.0,
        heading: float = 0.0,
    ) -> List[Tuple[float, float, float]]:
        """Calculate terrain-intersected footprint corners.

        .. deprecated::
            Use the instance method :meth:`ground_footprint`
            with ``lat``, ``lon``, and ``altitude_msl`` keyword
            arguments instead.  It uses the camera's own FOV and tilt
            parameters automatically.
        """
        warnings.warn(
            "footprint_corners() is deprecated. Use the instance method "
            "ground_footprint(altitude_agl, lat=..., lon=..., "
            "altitude_msl=..., heading=..., dem_file=...) instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        R = FrameCamera._corner_rotation(tilt_angle, tilt_direction, 0.0)

        hfx = np.tan(np.radians(fov_x / 2))
        hfy = np.tan(np.radians(fov_y / 2))
        corners_cam = np.array([
            [-hfx, -hfy, 1.0],
            [ hfx, -hfy, 1.0],
            [ hfx,  hfy, 1.0],
            [-hfx,  hfy, 1.0],
        ])

        corners = []
        for ray_cam in corners_cam:
            ray = R @ ray_cam
            if ray[2] <= 0:
                corners.append((np.nan, np.nan, np.nan))
                continue
            depression = np.degrees(np.arctan2(
                ray[2], np.sqrt(ray[0] ** 2 + ray[1] ** 2)))
            azimuth_cam = np.degrees(np.arctan2(ray[0], ray[1]))
            azimuth_geo = (heading + azimuth_cam) % 360

            clat, clon, calt = ray_terrain_intersection(
                lat, lon, altitude_msl,  # type: ignore[arg-type]
                np.array([azimuth_geo]), np.array([90.0 - depression]),
                dem_file=dem_file,
            )
            corners.append((float(clat[0]), float(clon[0]), float(calt[0])))

        return corners


class MultiCameraRig(Sensor):
    """A rig of multiple :class:`FrameCamera` instances with known orientations.

    Each camera carries its own ``tilt_angle`` and ``tilt_direction``.
    The rig stores optional lateral/longitudinal offsets for each camera.

    Args:
        name: Rig name.
        cameras: List of dicts, each with keys:
            ``"camera"`` (:class:`FrameCamera`),
            ``"label"`` (str),
            ``"dx"`` (Quantity, lateral offset, default 0 m),
            ``"dy"`` (Quantity, longitudinal offset, default 0 m).
    """

    def __init__(self, name: str, cameras: List[Dict]):
        super().__init__(name)
        self.cameras = []
        for entry in cameras:
            cam = {
                "camera": entry["camera"],
                "label": entry.get("label", entry["camera"].name),
                "dx": entry.get("dx", 0.0 * ureg.meter),
                "dy": entry.get("dy", 0.0 * ureg.meter),
            }
            self.cameras.append(cam)

    def __len__(self):
        return len(self.cameras)

    def swath_width(self, altitude_agl: Quantity) -> Quantity:
        """Combined across-track swath width (union of all cameras)."""
        widths = [c["camera"].swath_width(altitude_agl) for c in self.cameras]
        # Simple approach: sum unique cross-track contributions
        # For cameras at different cross-track angles this is an approximation
        return max(widths, key=lambda w: w.magnitude)  # type: ignore[no-any-return]

    def ground_sample_distance(self, altitude_agl: Quantity) -> Dict[str, Quantity]:
        """Finest GSD across all cameras."""
        gsds = [c["camera"].ground_sample_distance(altitude_agl) for c in self.cameras]
        return {
            "x": min((g["x"] for g in gsds), key=lambda v: v.magnitude),
            "y": min((g["y"] for g in gsds), key=lambda v: v.magnitude),
        }

    def combined_footprints(self, altitude_agl: Quantity) -> List[Dict]:
        """Per-camera footprint dicts with labels."""
        result = []
        for entry in self.cameras:
            fp = entry["camera"].footprint_at(altitude_agl)
            fp["label"] = entry["label"]
            result.append(fp)
        return result

    def ground_footprint(
        self,
        altitude_agl: Quantity,
        *,
        edge_points: int = 10,
        lat: float | None = None,
        lon: float | None = None,
        altitude_msl: float | None = None,
        heading: float = 0.0,
        dem_file: str | None = None,
    ) -> List[Dict]:
        """Project each camera's sensor perimeter onto the ground.

        Uses each camera's tilt geometry and the ``dx`` cross-track
        angular offset stored in the rig layout.

        Returns:
            List of dicts with ``"label"`` (str) and ``"polygon"``
            (:class:`shapely.geometry.Polygon`).
        """
        terrain_kwargs = {}
        if lat is not None and lon is not None and altitude_msl is not None:
            terrain_kwargs = dict(lat=lat, lon=lon, altitude_msl=altitude_msl,
                                  heading=heading, dem_file=dem_file)

        result = []
        for entry in self.cameras:
            dx = entry["dx"]
            cross_offset = dx.m_as("degree") if isinstance(dx, Quantity) else float(dx)
            poly = entry["camera"].ground_footprint(
                altitude_agl, cross_track_offset=cross_offset,
                edge_points=edge_points, **terrain_kwargs,
            )
            result.append({"label": entry["label"], "polygon": poly})
        return result

    def ground_footprint_corners(self, *args, **kwargs):
        """Deprecated — use :meth:`ground_footprint` instead."""
        warnings.warn(
            "ground_footprint_corners() is deprecated, use ground_footprint()",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.ground_footprint(*args, **kwargs)

    def stereo_pairs(self) -> List[Tuple[Dict, Dict]]:
        """Find camera pairs with opposing tilt directions (~180° apart).

        Returns a list of (forward_entry, aft_entry) tuples.
        """
        pairs = []
        used = set()
        for i, a in enumerate(self.cameras):
            if i in used:
                continue
            for j, b in enumerate(self.cameras):
                if j <= i or j in used:
                    continue
                dir_diff = abs(a["camera"].tilt_direction - b["camera"].tilt_direction)
                dir_diff = min(dir_diff, 360 - dir_diff)
                if 150 <= dir_diff <= 210:
                    # Determine which is forward (tilt_direction closer to 0)
                    fwd, aft = (a, b) if a["camera"].tilt_direction < 90 or a["camera"].tilt_direction > 270 else (b, a)
                    pairs.append((fwd, aft))
                    used.add(i)
                    used.add(j)
                    break
        return pairs

    def composite_base_height_ratio(self, altitude_agl: Quantity) -> List[Dict]:
        """B/H ratio for each stereo pair.

        For convergent stereo, ``B/H = tan(θ_fwd) + tan(θ_aft)``
        where θ is the tilt angle of each camera.

        Returns:
            List of dicts with ``"pair"`` (labels) and ``"bh_ratio"``.
        """
        altitude_agl = self.cameras[0]["camera"]._validate_quantity(altitude_agl, ureg.meter)
        pairs = self.stereo_pairs()
        results = []
        for fwd, aft in pairs:
            bh = (np.tan(np.radians(fwd["camera"].tilt_angle))
                  + np.tan(np.radians(aft["camera"].tilt_angle)))
            results.append({
                "pair": (fwd["label"], aft["label"]),
                "bh_ratio": float(bh),
            })
        return results

    def line_spacing(self, altitude_agl: Quantity, sidelap_pct: float = 60.0) -> Quantity:
        """Flight line spacing from sidelap and combined swath width."""
        return self.swath_width(altitude_agl) * (1 - sidelap_pct / 100)  # type: ignore[return-value,no-any-return]

    @classmethod
    def quakes_i(cls) -> "MultiCameraRig":
        """Create a QUAKES-I multi-camera rig.

        Based on Donnellan et al. (2025), Earth and Space Science.
        The camera hardware is the same regardless of aircraft platform
        (Gulfstream V at 12.5 km / 250 m/s, or King Air at 6 km / 125 m/s).
        The platform choice only affects operational parameters like the
        altitude passed to ``swath_width()`` etc.

        Returns:
            A :class:`MultiCameraRig` with 8 cameras (4 forward + 4 aft).
        """
        # Common camera specs
        # AMS CMV20000 sensor: "35 mm" refers to the optical format (lens coverage),
        # not the physical sensor diagonal. Physical dimensions from 6.4 µm pixel pitch:
        #   cross-track (5120 px): 5120 × 6.4 µm = 32.77 mm  → FOV_x ≈ 18.6° per camera
        #   along-track  (3840 px): 3840 × 6.4 µm = 24.58 mm → FOV_y ≈ 14°  per camera
        # 35 mm / 24 mm are close approximations used throughout the paper.
        sensor_w = 35.0 * ureg.mm   # cross-track (wider dimension, 5120 px)
        sensor_h = 24.0 * ureg.mm   # along-track (shorter dimension, 3840 px)
        focal_length = 100.0 * ureg.mm
        res_x = 5120   # pixels cross-track
        res_y = 3840   # pixels along-track
        frame_rate = 2.0 * ureg.Hz
        f_speed = 2.8  # T2.1 lens (Schneider Xenon FF 100 mm)

        tilt = 11.3  # degrees from nadir (each main plate, total 22.6° fwd-aft)

        # Cross-track angular offsets per Donnellan et al. (2025), Table/Fig 3:
        #   inner pair (cams 2&3, 6&7): separated by 17.1° → centers at ±8.55°
        #   outer pair (cams 1&4, 5&8): separated by 40.5° → centers at ±20.25°
        # Combined cross-track FOV ≈ 60°; inner pairs overlap by ~1.5°.
        # Geometric swath at 12.5 km AGL over sea-level terrain ≈ 14 km;
        # paper's "12 km" is at typical California terrain elevation (~2 km MSL).
        cross_offsets = [-20.25, -8.55, 8.55, 20.25]

        cameras = []
        for tilt_dir, prefix in [(0.0, "fwd"), (180.0, "aft")]:
            for cam_idx, cross_off in enumerate(cross_offsets):
                cam = FrameCamera(
                    name=f"QUAKES-I {prefix}_{cam_idx + 1}",
                    sensor_width=sensor_w,
                    sensor_height=sensor_h,
                    focal_length=focal_length,
                    resolution_x=res_x,
                    resolution_y=res_y,
                    frame_rate=frame_rate,
                    f_speed=f_speed,
                    tilt_angle=tilt,
                    tilt_direction=tilt_dir,
                )
                cameras.append({
                    "camera": cam,
                    "label": f"{prefix}_{cam_idx + 1}",
                    "dx": cross_off * ureg.degree,  # angular offset stored for reference
                })

        return cls(name="QUAKES-I", cameras=cameras)
