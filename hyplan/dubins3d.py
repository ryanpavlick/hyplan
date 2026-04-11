"""
3D Dubins path planning with pitch angle constraints.

Port of comrob/Dubins3D.jl (Vana et al., ICRA 2020) to Python.
Decomposes 3D paths into coupled horizontal and vertical 2D Dubins
problems linked by a curvature budget:

    1/rho_min² = 1/rho_h² + 1/rho_v²

A 1D search over rho_h finds the shortest feasible 3D path.

References
----------
Vana, P., Neto, A.A., Faigl, J. and Macharet, D.G. (2020). Minimal
3D Dubins path with bounded curvature and pitch angle. *IEEE
International Conference on Robotics and Automation (ICRA)*, 8497-8503.
doi:10.1109/ICRA40945.2020.9197348

Dubins, L.E. (1957). On curves of minimal length with a constraint on
average curvature, and with prescribed initial and terminal positions
and tangents. *American Journal of Mathematics*, 79(3), 497-516.
doi:10.2307/2372560

2D Dubins solver adapted from Andrew Walker's implementation:
Walker, A. (2011). Hard Real-Time Motion Planning for Autonomous
Vehicles. PhD thesis, Swinburne University of Technology.
"""

import math
from typing import Optional, Tuple, Union

import numpy as np
from shapely.geometry import LineString
from shapely.ops import transform

from .geometry import get_utm_transforms
from .units import ureg
from .waypoint import Waypoint, is_waypoint
from .exceptions import HyPlanTypeError, HyPlanValueError, HyPlanRuntimeError


# ---------------------------------------------------------------------------
# Internal 2D Dubins solver (needed for both horizontal and vertical planes)
# ---------------------------------------------------------------------------

def _mod2pi(angle: float) -> float:
    """Wrap angle to [0, 2π)."""
    return angle % (2.0 * math.pi)


class _DubinsSegment:
    """Result of a 2D Dubins path computation (t, p, q segment lengths)."""
    __slots__ = ("t", "p", "q", "length", "case")

    def __init__(self, t: float, p: float, q: float, length: float, case: str):
        self.t = t
        self.p = p
        self.q = q
        self.length = length
        self.case = case

    @property
    def valid(self) -> bool:
        """True if this segment represents a feasible path."""
        return self.case != "XXX" and math.isfinite(self.length)


class _Dubins2D:
    """Standard 2D Dubins path solver with sampling."""

    def __init__(self, qi: np.ndarray, qf: np.ndarray, rhomin: float,
                 disable_ccc: bool = False):
        self.qi = qi.copy()
        self.qf = qf.copy()
        self.rhomin = rhomin
        self.maneuver: _DubinsSegment = _DubinsSegment(0, 0, 0, math.inf, "")
        self._solve(disable_ccc)

    def _solve(self, disable_ccc: bool):
        dx = self.qf[0] - self.qi[0]
        dy = self.qf[1] - self.qi[1]
        D = math.sqrt(dx * dx + dy * dy)
        d = D / self.rhomin if self.rhomin > 0 else 0.0

        rot = _mod2pi(math.atan2(dy, dx))
        a = _mod2pi(self.qi[2] - rot)
        b = _mod2pi(self.qf[2] - rot)

        sa, ca = math.sin(a), math.cos(a)
        sb, cb = math.sin(b), math.cos(b)

        paths = [
            self._LSL(a, b, d, sa, ca, sb, cb),
            self._RSR(a, b, d, sa, ca, sb, cb),
            self._LSR(a, b, d, sa, ca, sb, cb),
            self._RSL(a, b, d, sa, ca, sb, cb),
        ]
        if not disable_ccc:
            paths.append(self._RLR(a, b, d, sa, ca, sb, cb))
            paths.append(self._LRL(a, b, d, sa, ca, sb, cb))

        # Handle degenerate case (same position, same heading)
        dist_2d = max(abs(self.qi[0] - self.qf[0]), abs(self.qi[1] - self.qf[1]))
        if d < self.rhomin * 1e-5 and abs(a) < self.rhomin * 1e-5 and abs(b) < self.rhomin * 1e-5:
            if dist_2d < self.rhomin * 1e-5:
                paths = [_DubinsSegment(0, 2 * math.pi, 0, 2 * math.pi * self.rhomin, "RRR")]

        paths.sort(key=lambda x: x.length)
        self.maneuver = paths[0]

    # --- CSC path types ---

    def _LSL(self, a, b, d, sa, ca, sb, cb):
        aux = math.atan2(cb - ca, d + sa - sb)
        t = _mod2pi(-a + aux)
        p = math.sqrt(2 + d * d - 2 * math.cos(a - b) + 2 * d * (sa - sb))
        q = _mod2pi(b - aux)
        return _DubinsSegment(t, p, q, (t + p + q) * self.rhomin, "LSL")

    def _RSR(self, a, b, d, sa, ca, sb, cb):
        aux = math.atan2(ca - cb, d - sa + sb)
        t = _mod2pi(a - aux)
        p = math.sqrt(2 + d * d - 2 * math.cos(a - b) + 2 * d * (sb - sa))
        q = _mod2pi(_mod2pi(-b) + aux)
        return _DubinsSegment(t, p, q, (t + p + q) * self.rhomin, "RSR")

    def _LSR(self, a, b, d, sa, ca, sb, cb):
        aux1 = -2 + d * d + 2 * math.cos(a - b) + 2 * d * (sa + sb)
        if aux1 > 0:
            p = math.sqrt(aux1)
            aux2 = math.atan2(-ca - cb, d + sa + sb) - math.atan2(-2, p)
            t = _mod2pi(-a + aux2)
            q = _mod2pi(-_mod2pi(b) + aux2)
        else:
            t = p = q = math.inf
        return _DubinsSegment(t, p, q, (t + p + q) * self.rhomin, "LSR")

    def _RSL(self, a, b, d, sa, ca, sb, cb):
        aux1 = d * d - 2 + 2 * math.cos(a - b) - 2 * d * (sa + sb)
        if aux1 > 0:
            p = math.sqrt(aux1)
            aux2 = math.atan2(ca + cb, d - sa - sb) - math.atan2(2, p)
            t = _mod2pi(a - aux2)
            q = _mod2pi(_mod2pi(b) - aux2)
        else:
            t = p = q = math.inf
        return _DubinsSegment(t, p, q, (t + p + q) * self.rhomin, "RSL")

    # --- CCC path types ---

    def _RLR(self, a, b, d, sa, ca, sb, cb):
        aux = (6 - d * d + 2 * math.cos(a - b) + 2 * d * (sa - sb)) / 8
        if abs(aux) <= 1:
            p = _mod2pi(-math.acos(aux))
            t = _mod2pi(a - math.atan2(ca - cb, d - sa + sb) + p / 2)
            q = _mod2pi(a - b - t + p)
        else:
            t = p = q = math.inf
        return _DubinsSegment(t, p, q, (t + p + q) * self.rhomin, "RLR")

    def _LRL(self, a, b, d, sa, ca, sb, cb):
        aux = (6 - d * d + 2 * math.cos(a - b) + 2 * d * (-sa + sb)) / 8
        if abs(aux) <= 1:
            p = _mod2pi(-math.acos(aux))
            t = _mod2pi(-a + math.atan2(-ca + cb, d + sa - sb) + p / 2)
            q = _mod2pi(b - a - t + p)
        else:
            t = p = q = math.inf
        return _DubinsSegment(t, p, q, (t + p + q) * self.rhomin, "LRL")

    # --- Sampling ---

    def get_coordinates_at(self, offset: float) -> np.ndarray:
        """Get (x, y, heading) at a given arc-length offset along the path."""
        noffset = offset / self.rhomin
        qi = np.array([0.0, 0.0, self.qi[2]])

        l1 = self.maneuver.t
        l2 = self.maneuver.p
        q1 = _position_in_segment(l1, qi, self.maneuver.case[0])
        q2 = _position_in_segment(l2, q1, self.maneuver.case[1])

        if noffset < l1:
            q = _position_in_segment(noffset, qi, self.maneuver.case[0])
        elif noffset < l1 + l2:
            q = _position_in_segment(noffset - l1, q1, self.maneuver.case[1])
        else:
            q = _position_in_segment(noffset - l1 - l2, q2, self.maneuver.case[2])

        q[0] = q[0] * self.rhomin + self.qi[0]
        q[1] = q[1] * self.rhomin + self.qi[1]
        q[2] = _mod2pi(q[2])
        return q


def _position_in_segment(offset: float, qi: np.ndarray, case: str) -> np.ndarray:
    """Compute position after traversing a segment of given type."""
    q = np.zeros(3)
    if case == "L":
        q[0] = qi[0] + math.sin(qi[2] + offset) - math.sin(qi[2])
        q[1] = qi[1] - math.cos(qi[2] + offset) + math.cos(qi[2])
        q[2] = qi[2] + offset
    elif case == "R":
        q[0] = qi[0] - math.sin(qi[2] - offset) + math.sin(qi[2])
        q[1] = qi[1] + math.cos(qi[2] - offset) - math.cos(qi[2])
        q[2] = qi[2] - offset
    elif case == "S":
        q[0] = qi[0] + math.cos(qi[2]) * offset
        q[1] = qi[1] + math.sin(qi[2]) * offset
        q[2] = qi[2]
    return q


# ---------------------------------------------------------------------------
# Wind-aware 2D Dubins solver (trochoidal ground tracks)
# ---------------------------------------------------------------------------

class _TrochoidDubins2D:
    """Wind-aware 2D Dubins solver with trochoidal ground tracks.

    Solves for the time-optimal path in the air-relative frame, then
    provides ground-frame sampling where turning arcs become trochoids
    (circles + wind drift).

    The approach:
    1. Iteratively solve: guess total time T, place the air-frame target
       at ``qf_ground - wind * T``, solve standard Dubins, update T.
    2. Once converged, store the air-frame Dubins solution.
    3. ``get_coordinates_at(offset)`` returns ground-frame positions by
       adding cumulative wind drift to air-frame positions.

    Args:
        qi: Start pose [x, y, heading] in ground frame (meters, radians).
        qf: End pose [x, y, heading] in ground frame (meters, radians).
        rhomin: Minimum turn radius in meters (air-frame).
        airspeed: True airspeed in m/s.
        wind_u: Eastward wind component in m/s.
        wind_v: Northward wind component in m/s.
        disable_ccc: If True, disable CCC (RLR/LRL) path types.
    """

    def __init__(
        self,
        qi: np.ndarray,
        qf: np.ndarray,
        rhomin: float,
        airspeed: float,
        wind_u: float,
        wind_v: float,
        disable_ccc: bool = False,
    ):
        self.qi = qi.copy()
        self.qf = qf.copy()
        self.rhomin = rhomin
        self.airspeed = airspeed
        self.wind_u = wind_u  # eastward (x-direction in UTM)
        self.wind_v = wind_v  # northward (y-direction in UTM)

        # Iteratively solve for the time-optimal air-frame path.
        # The air-frame target drifts at -wind relative to the aircraft's
        # starting position, so qf_air = qf_ground - wind * T.
        self._solve_iterative(disable_ccc)

    def _solve_iterative(self, disable_ccc: bool, max_iter: int = 20, tol: float = 0.1):
        """Find the air-frame Dubins path that accounts for wind drift."""
        # Initial guess: still-air path
        d_still = _Dubins2D(self.qi, self.qf, self.rhomin, disable_ccc)
        T = d_still.maneuver.length / self.airspeed if self.airspeed > 0 else 0.0

        for _ in range(max_iter):
            # Place target in air frame: ground target drifts by -wind*T
            qf_air = self.qf.copy()
            qf_air[0] -= self.wind_u * T
            qf_air[1] -= self.wind_v * T
            # Heading stays the same (air heading = desired ground heading
            # only if we ignore crab angle — we'll handle that below)

            d_air = _Dubins2D(self.qi, qf_air, self.rhomin, disable_ccc)
            T_new = d_air.maneuver.length / self.airspeed if self.airspeed > 0 else 0.0

            if abs(T_new - T) < tol:
                T = T_new
                break
            T = T_new

        self._air_dubins = d_air
        self._total_time = T

    @property
    def maneuver(self) -> _DubinsSegment:
        """The air-frame Dubins maneuver (segment lengths in air-frame meters)."""
        return self._air_dubins.maneuver

    @property
    def total_time(self) -> float:
        """Total path time in seconds."""
        return self._total_time

    @property
    def ground_length(self) -> float:
        """Approximate ground-track length in meters."""
        # Sample and compute cumulative distance
        n = 50
        pts = np.array([self.get_coordinates_at(i * self._total_time / (n - 1))
                        for i in range(n)])
        diffs = np.diff(pts[:, :2], axis=0)
        return float(np.sum(np.sqrt(diffs[:, 0]**2 + diffs[:, 1]**2)))

    def get_coordinates_at(self, time_offset: float) -> np.ndarray:
        """Get ground-frame (x, y, heading) at a given time offset.

        The air-frame position is computed from the standard Dubins path,
        then wind drift is added to get the ground-frame position.
        Turn segments produce trochoidal ground tracks.
        """
        # Air-frame arc-length offset
        arc_offset = time_offset * self.airspeed
        # Air-frame position (circle/line geometry, no wind)
        q_air = self._air_dubins.get_coordinates_at(arc_offset)
        # Ground-frame position = air position + wind * time
        q_ground = np.array([
            q_air[0] + self.wind_u * time_offset,
            q_air[1] + self.wind_v * time_offset,
            q_air[2],  # heading in air frame
        ])
        return q_ground


# ---------------------------------------------------------------------------
# Vertical-plane Dubins solver (pitch-constrained, CSC only)
# ---------------------------------------------------------------------------

class _VerticalDubins(_Dubins2D):
    """
    Pitch-constrained vertical Dubins solver.

    Only CSC paths are allowed (no CCC). Arc segments are limited to < π
    radians. The center angle (pitch at the tangent point between arc and
    straight segment) must stay within pitch limits. LSR/RSL paths are
    clamped to pitch boundaries when the unconstrained center angle
    exceeds limits.
    """

    def __init__(self, qi: np.ndarray, qf: np.ndarray, rhomin: float,
                 pitch_limits: Tuple[float, float]):
        self.qi = qi.copy()
        self.qf = qf.copy()
        self.rhomin = rhomin
        self.pitch_limits = pitch_limits  # (pitch_min, pitch_max)
        self.maneuver = _DubinsSegment(0, 0, 0, math.inf, "")
        self._solve_vertical()

    def _solve_vertical(self):
        paths = [
            self._v_LSL(),
            self._v_RSR(),
            self._v_LSR(),
            self._v_RSL(),
        ]
        paths.sort(key=lambda x: x.length)

        found = False
        for p in paths:
            if abs(p.t) >= math.pi or abs(p.q) >= math.pi:
                continue
            # Check center angle stays within pitch limits
            if p.case[0] == "L":
                center_angle = self.qi[2] + p.t
            else:
                center_angle = self.qi[2] - p.t
            if center_angle < self.pitch_limits[0] or center_angle > self.pitch_limits[1]:
                continue
            self.maneuver = p
            found = True
            break

        if not found:
            raise HyPlanRuntimeError(
                f"No feasible pitch-constrained vertical Dubins path found. "
                f"pitch_limits={self.pitch_limits}, qi={self.qi}, qf={self.qf}. "
                f"Check that the altitude change is achievable within the "
                f"given pitch limits and turn radius."
            )

    def _v_LSL(self):
        theta1 = self.qi[2]
        theta2 = self.qf[2]
        if theta1 > theta2:
            return _DubinsSegment(math.inf, math.inf, math.inf, math.inf, "LSL")

        p1 = self.qi[:2]
        p2 = self.qf[:2]
        r = self.rhomin

        c1, s1 = r * math.cos(theta1), r * math.sin(theta1)
        c2, s2 = r * math.cos(theta2), r * math.sin(theta2)

        o1 = p1 + np.array([-s1, c1])
        o2 = p2 + np.array([-s2, c2])

        diff = o2 - o1
        center_distance = math.sqrt(diff[0] ** 2 + diff[1] ** 2)
        center_angle = math.atan2(diff[1], diff[0])

        t = _mod2pi(-theta1 + center_angle)
        p = center_distance / r
        q = _mod2pi(theta2 - center_angle)

        if t > math.pi:
            t = 0.0
            q = theta2 - theta1
            turn_end_y = o2[1] - r * math.cos(theta1)
            diff_y = turn_end_y - p1[1]
            if abs(theta1) > 1e-5 and (diff_y < 0) == (theta1 < 0):
                p = diff_y / math.sin(theta1) / r
            else:
                t = p = q = math.inf

        if q > math.pi:
            t = theta2 - theta1
            q = 0.0
            turn_end_y = o1[1] - r * math.cos(theta2)
            diff_y = p2[1] - turn_end_y
            if abs(theta2) > 1e-5 and (diff_y < 0) == (theta2 < 0):
                p = diff_y / math.sin(theta2) / r
            else:
                t = p = q = math.inf

        return _DubinsSegment(t, p, q, (t + p + q) * r, "LSL")

    def _v_RSR(self):
        theta1 = self.qi[2]
        theta2 = self.qf[2]
        if theta2 > theta1:
            return _DubinsSegment(math.inf, math.inf, math.inf, math.inf, "RSR")

        p1 = self.qi[:2]
        p2 = self.qf[:2]
        r = self.rhomin

        c1, s1 = r * math.cos(theta1), r * math.sin(theta1)
        c2, s2 = r * math.cos(theta2), r * math.sin(theta2)

        o1 = p1 + np.array([s1, -c1])
        o2 = p2 + np.array([s2, -c2])

        diff = o2 - o1
        center_distance = math.sqrt(diff[0] ** 2 + diff[1] ** 2)
        center_angle = math.atan2(diff[1], diff[0])

        t = _mod2pi(theta1 - center_angle)
        p = center_distance / r
        q = _mod2pi(-theta2 + center_angle)

        if t > math.pi:
            t = 0.0
            q = -theta2 + theta1
            turn_end_y = o2[1] + r * math.cos(theta1)
            diff_y = turn_end_y - p1[1]
            if abs(theta1) > 1e-5 and (diff_y < 0) == (theta1 < 0):
                p = diff_y / math.sin(theta1) / r
            else:
                t = p = q = math.inf

        if q > math.pi:
            t = -theta2 + theta1
            q = 0.0
            turn_end_y = o1[1] + r * math.cos(theta2)
            diff_y = p2[1] - turn_end_y
            if abs(theta2) > 1e-5 and (diff_y < 0) == (theta2 < 0):
                p = diff_y / math.sin(theta2) / r
            else:
                t = p = q = math.inf

        return _DubinsSegment(t, p, q, (t + p + q) * r, "RSR")

    def _v_LSR(self):
        theta1 = self.qi[2]
        theta2 = self.qf[2]
        p1 = self.qi[:2]
        p2 = self.qf[:2]
        r = self.rhomin
        pitch_max = self.pitch_limits[1]

        c1, s1 = r * math.cos(theta1), r * math.sin(theta1)
        c2, s2 = r * math.cos(theta2), r * math.sin(theta2)

        o1 = p1 + np.array([-s1, c1])
        o2 = p2 + np.array([s2, -c2])

        diff = o2 - o1
        center_distance = math.sqrt(diff[0] ** 2 + diff[1] ** 2)

        if center_distance < 2 * r:
            diff[0] = math.sqrt(4.0 * r * r - diff[1] * diff[1])
            alpha = math.pi / 2.0
        else:
            alpha = math.asin(2.0 * r / center_distance)

        center_angle = math.atan2(diff[1], diff[0]) + alpha

        if center_angle < pitch_max:
            t = _mod2pi(-theta1 + center_angle)
            p = math.sqrt(max(0.0, center_distance * center_distance - 4.0 * r * r)) / r
            q = _mod2pi(-theta2 + center_angle)
        else:
            center_angle = pitch_max
            t = _mod2pi(-theta1 + center_angle)
            q = _mod2pi(-theta2 + center_angle)

            cc, ss = r * math.cos(center_angle), r * math.sin(center_angle)
            w1 = o1 - np.array([-ss, cc])
            w2 = o2 - np.array([ss, -cc])

            if abs(math.sin(center_angle)) > 1e-10:
                p = (w2[1] - w1[1]) / math.sin(center_angle) / r
            else:
                p = math.inf

        return _DubinsSegment(t, p, q, (t + p + q) * r, "LSR")

    def _v_RSL(self):
        theta1 = self.qi[2]
        theta2 = self.qf[2]
        p1 = self.qi[:2]
        p2 = self.qf[:2]
        r = self.rhomin
        pitch_min = self.pitch_limits[0]

        c1, s1 = r * math.cos(theta1), r * math.sin(theta1)
        c2, s2 = r * math.cos(theta2), r * math.sin(theta2)

        o1 = p1 + np.array([s1, -c1])
        o2 = p2 + np.array([-s2, c2])

        diff = o2 - o1
        center_distance = math.sqrt(diff[0] ** 2 + diff[1] ** 2)

        if center_distance < 2 * r:
            diff[0] = math.sqrt(4.0 * r * r - diff[1] * diff[1])
            alpha = math.pi / 2.0
        else:
            alpha = math.asin(2.0 * r / center_distance)

        center_angle = math.atan2(diff[1], diff[0]) - alpha

        if center_angle > pitch_min:
            t = _mod2pi(theta1 - center_angle)
            p = math.sqrt(max(0.0, center_distance * center_distance - 4.0 * r * r)) / r
            q = _mod2pi(theta2 - center_angle)
        else:
            center_angle = pitch_min
            t = _mod2pi(theta1 - center_angle)
            q = _mod2pi(theta2 - center_angle)

            cc, ss = r * math.cos(center_angle), r * math.sin(center_angle)
            w1 = o1 - np.array([ss, -cc])
            w2 = o2 - np.array([-ss, cc])

            if abs(math.sin(center_angle)) > 1e-10:
                p = (w2[1] - w1[1]) / math.sin(center_angle) / r
            else:
                p = math.inf

        return _DubinsSegment(t, p, q, (t + p + q) * r, "RSL")


# ---------------------------------------------------------------------------
# 3D Dubins path
# ---------------------------------------------------------------------------

def _try_to_construct(
    qi: np.ndarray,
    qf: np.ndarray,
    rhomin: float,
    horizontal_radius: float,
    pitch_limits: Tuple[float, float],
    wind: Optional[Tuple[float, float]] = None,
    airspeed: float = 0.0,
) -> Optional[Tuple[Union[_Dubins2D, _TrochoidDubins2D], "_VerticalDubins"]]:
    """
    Attempt to construct a 3D Dubins path with the given horizontal radius.

    Returns (horizontal_path, vertical_path) or None if infeasible.
    """
    # Horizontal 2D Dubins: (x, y, heading)
    qi2d = np.array([qi[0], qi[1], qi[3]])
    qf2d = np.array([qf[0], qf[1], qf[3]])
    if wind is not None:
        d_lat = _TrochoidDubins2D(
            qi2d, qf2d, horizontal_radius, airspeed, wind[0], wind[1],
        )
    else:
        d_lat = _Dubins2D(qi2d, qf2d, horizontal_radius)

    # Vertical curvature from curvature budget
    vc = 1.0 / (rhomin * rhomin) - 1.0 / (horizontal_radius * horizontal_radius)
    if vc < 1e-10:
        return None
    vertical_radius = 1.0 / math.sqrt(vc)

    # Vertical 2D Dubins: (arc_length, altitude, pitch)
    qi3d = np.array([0.0, qi[2], qi[4]])
    qf3d = np.array([d_lat.maneuver.length, qf[2], qf[4]])
    d_lon = _Dubins2D(qi3d, qf3d, vertical_radius)

    # Reject CCC paths in vertical plane (bug fix: original checked RLR twice)
    if d_lon.maneuver.case in ("RLR", "LRL"):
        return None

    # Check pitch constraint on first arc
    if d_lon.maneuver.case[0] == "R":
        if qi[4] - d_lon.maneuver.t < pitch_limits[0]:
            return None
    else:
        if qi[4] + d_lon.maneuver.t > pitch_limits[1]:
            return None

    return (d_lat, d_lon)


def _compute_3d_path(
    qi: np.ndarray,
    qf: np.ndarray,
    rhomin: float,
    pitch_limits: Tuple[float, float],
    wind: Optional[Tuple[float, float]] = None,
    airspeed: float = 0.0,
) -> Tuple[Union[_Dubins2D, _TrochoidDubins2D], "_VerticalDubins", float]:
    """
    Find optimal horizontal radius and return (horizontal, vertical, length).

    Uses adaptive step-size local optimization from Dubins3D.jl.
    """
    b = 1.0

    # Find initial feasible solution by doubling b
    fb = _try_to_construct(qi, qf, rhomin, rhomin * b, pitch_limits, wind, airspeed)
    max_iter = 50
    for _ in range(max_iter):
        if fb is not None:
            break
        b *= 2.0
        if b > 1e6:
            raise HyPlanValueError(
                "No feasible 3D Dubins path found. Check that the pitch limits "
                "allow reaching the target altitude."
            )
        fb = _try_to_construct(qi, qf, rhomin, rhomin * b, pitch_limits, wind, airspeed)

    if fb is None:
        raise HyPlanValueError("No feasible 3D Dubins path found.")

    # Local optimization: adaptive step-size search
    step = 0.1
    while abs(step) > 1e-10:
        c = b + step
        if c < 1.0:
            c = 1.0
        fc = _try_to_construct(qi, qf, rhomin, rhomin * c, pitch_limits, wind, airspeed)
        if fc is not None and fc[1].maneuver.length < fb[1].maneuver.length:
            b = c
            fb = fc
            step *= 2.0
            continue
        step *= -0.1

    d_lat, d_lon = fb
    return d_lat, d_lon, d_lon.maneuver.length


def _sample_3d_path(
    d_lat: Union[_Dubins2D, _TrochoidDubins2D],
    d_lon: "_VerticalDubins",
    n_samples: int,
) -> np.ndarray:
    """
    Sample points along the 3D path.

    Returns array of shape (n_samples, 5): [x, y, z, heading, pitch].
    The vertical path is the master parameterization.

    For wind-aware (trochoid) paths, the horizontal path is sampled by
    time offset rather than arc-length offset, since the ground track
    differs from the air track.
    """
    total_length = d_lon.maneuver.length
    offsets = np.linspace(0, total_length, n_samples)

    is_trochoid = isinstance(d_lat, _TrochoidDubins2D)

    points = np.empty((n_samples, 5))
    for i, offset in enumerate(offsets):
        # Vertical path gives (arc_length_on_horizontal, altitude, pitch)
        q_sz = d_lon.get_coordinates_at(offset)

        if is_trochoid:
            # Convert air-frame arc length to time, then sample ground track
            time_offset = q_sz[0] / d_lat.airspeed if d_lat.airspeed > 0 else 0.0
            q_xy = d_lat.get_coordinates_at(time_offset)
        else:
            # Standard: sample by arc-length directly
            q_xy = d_lat.get_coordinates_at(q_sz[0])

        points[i] = [q_xy[0], q_xy[1], q_sz[1], q_xy[2], q_sz[2]]

    return points


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class DubinsPath3D:
    """
    3D Dubins path between two waypoints with pitch angle constraints.

    Decomposes the problem into coupled horizontal and vertical 2D Dubins
    sub-problems. The horizontal plane uses a standard Dubins solver; the
    vertical plane is parameterized by (arc_length, altitude) with pitch
    as the heading analogue. A 1D search optimizes the horizontal turn
    radius to minimize total 3D path length.

    When ``wind`` is provided, turning arcs in the horizontal plane
    become **trochoids** (circles drifting with the wind) instead of
    pure circles. The solver finds the time-optimal path in the
    air-relative frame and samples the ground track with wind drift
    applied. See Moon et al. (2023), arXiv:2306.11845.

    Args:
        start: Starting waypoint (must have altitude_msl and heading).
        end: Ending waypoint (must have altitude_msl and heading).
        speed: True airspeed (m/s as float, or pint Quantity).
        bank_angle: Maximum bank angle in degrees.
        pitch_min: Minimum pitch angle in degrees (negative = descent).
        pitch_max: Maximum pitch angle in degrees (positive = climb).
        pitch_start: Initial pitch angle in degrees. Defaults to 0.
        pitch_end: Final pitch angle in degrees. Defaults to 0.
        step_size: Approximate distance between sampled points in meters.
            Defaults to 500.
        n_samples: Number of sample points. If given, overrides step_size.
        wind: Optional (u_east, v_north) wind vector in m/s. When
            provided, the horizontal path uses trochoidal geometry.
            Defaults to ``None`` (still air).
    """

    def __init__(
        self,
        start: Waypoint,
        end: Waypoint,
        speed: Union[ureg.Quantity, float],
        bank_angle: float,
        pitch_min: float = -10.0,
        pitch_max: float = 10.0,
        pitch_start: float = 0.0,
        pitch_end: float = 0.0,
        step_size: float = 500.0,
        n_samples: Optional[int] = None,
        wind: Optional[Tuple[float, float]] = None,
    ):
        if not is_waypoint(start) or not is_waypoint(end):
            raise HyPlanTypeError("start and end must be Waypoint objects")

        if start.altitude_msl is None or end.altitude_msl is None:
            raise HyPlanValueError("Both waypoints must have altitude_msl set for 3D path planning")

        self.start = start
        self.end = end

        # Speed
        if isinstance(speed, (int, float)):
            self.speed_mps = float(speed)
        elif hasattr(speed, "units") and speed.check("[speed]"):
            self.speed_mps = speed.m_as(ureg.meter / ureg.second)
        else:
            raise HyPlanTypeError("speed must be a float (m/s) or a pint Quantity with speed units")

        self.bank_angle = bank_angle
        self.pitch_min_deg = pitch_min
        self.pitch_max_deg = pitch_max
        self.pitch_start_deg = pitch_start
        self.pitch_end_deg = pitch_end

        # Convert to radians for the solver
        pitch_lim_rad = (math.radians(pitch_min), math.radians(pitch_max))
        pitch_start_rad = math.radians(pitch_start)
        pitch_end_rad = math.radians(pitch_end)

        # Turn radius from speed and bank angle
        g = 9.8  # m/s²
        bank_rad = math.radians(bank_angle)
        self._rhomin = (self.speed_mps ** 2) / (g * math.tan(bank_rad))

        # Convert waypoints to UTM
        to_utm, from_utm = get_utm_transforms([start.geometry, end.geometry])
        start_utm = transform(to_utm, start.geometry)
        end_utm = transform(to_utm, end.geometry)

        # Headings: convert from geographic (CW from N) to math (CCW from E)
        heading1 = -math.radians(start.heading - 90.0)
        heading2 = -math.radians(end.heading - 90.0)

        alt_start = start.altitude_msl.m_as(ureg.meter)
        alt_end = end.altitude_msl.m_as(ureg.meter)

        # qi/qf: [x, y, z, heading, pitch]
        qi = np.array([start_utm.x, start_utm.y, alt_start, heading1, pitch_start_rad])
        qf = np.array([end_utm.x, end_utm.y, alt_end, heading2, pitch_end_rad])

        self._wind = wind

        # Solve 3D path
        d_lat, d_lon, total_length = _compute_3d_path(
            qi, qf, self._rhomin, pitch_lim_rad,
            wind=wind, airspeed=self.speed_mps,
        )

        self._length = total_length * ureg.meter

        # Sample points
        if n_samples is not None:
            ns = n_samples
        else:
            ns = max(int(total_length / step_size) + 1, 2)

        pts_utm = _sample_3d_path(d_lat, d_lon, ns)

        # Convert XY back to geographic
        lons, lats = from_utm(pts_utm[:, 0], pts_utm[:, 1])
        self._altitudes = pts_utm[:, 2]

        # Store sampled path as (lat, lon, alt, heading_geo, pitch_deg)
        headings_geo = (90.0 - np.degrees(pts_utm[:, 3])) % 360.0  # math -> geographic
        # Pitch comes from vertical Dubins which wraps to [0, 2π) via _mod2pi;
        # unwrap so that negative pitch (descent) is represented correctly.
        pitches_raw = np.degrees(pts_utm[:, 4])
        pitches_deg = np.where(pitches_raw > 180.0, pitches_raw - 360.0, pitches_raw)

        self._points = np.column_stack([lats, lons, self._altitudes, headings_geo, pitches_deg])
        self._geometry = LineString(np.column_stack([lons, lats]))
        self._geometry_3d = LineString(np.column_stack([lons, lats, self._altitudes]))

    @property
    def geometry(self) -> LineString:
        """2D LineString (lon, lat) of the path."""
        return self._geometry

    @property
    def geometry_3d(self) -> LineString:
        """3D LineString (lon, lat, alt) of the path."""
        return self._geometry_3d

    @property
    def length(self) -> ureg.Quantity:
        """Total 3D path length."""
        return self._length

    @property
    def points(self) -> np.ndarray:
        """Sampled points as array of (lat, lon, alt_m, heading_deg, pitch_deg)."""
        return self._points

    @property
    def min_turn_radius(self) -> ureg.Quantity:
        """Minimum 3D turn radius in meters."""
        return self._rhomin * ureg.meter

    def to_dict(self) -> dict:
        """Dictionary representation."""
        return {
            "geometry": self.geometry,
            "geometry_3d": self.geometry_3d,
            "start_lat": self.start.latitude,
            "start_lon": self.start.longitude,
            "end_lat": self.end.latitude,
            "end_lon": self.end.longitude,
            "start_altitude": self.start.altitude_msl.m_as(ureg.meter),
            "end_altitude": self.end.altitude_msl.m_as(ureg.meter),
            "start_heading": self.start.heading,
            "end_heading": self.end.heading,
            "distance": self.length.m_as(ureg.nautical_mile),
        }
