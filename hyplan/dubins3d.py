"""
2D Dubins path planning with optional wind.

This module exports :class:`DubinsPath2D`, the horizontal Dubins path
used by the HyPlan flight planner via
:meth:`hyplan.aircraft.Aircraft._hybrid_path`.  The hybrid planner
couples this 2D Dubins (or trochoidal Dubins, when wind is supplied)
ground track with the aircraft's altitude-indexed ``climb_profile`` /
``descent_profile`` integrated separately for the vertical profile.

When a wind vector is supplied, the horizontal sub-problem switches
from circular arcs to **trochoidal** arcs (circles drifting with the
wind).  HyPlan's trochoidal solver considers three families:

* **BSB** (LSL/RSR/LSR/RSL) via the Sachdev/Moon (2023) solver — the
  bulk of operationally-encountered paths.
* **Proper trochoidal CCC** (LRL/RLR) via 1-D Newton on the half-arc
  angle of the middle arc — exact ground-frame goal landing.
* **Air-frame Dubins CCC + iterative wind-drift correction** — used as
  a fallback when Newton fails or its candidate has higher load
  factor; gated on a 10 m end-position tolerance.

CCC was historically disabled in trochoidal solvers (multi-loop
solutions can pollute the search); HyPlan now enumerates the
2π·k_4 wraps of the goal heading and rejects any candidate whose
unwrapped per-arc time leaves [0, 2π) — see ``solve_ccc_trochoid`` in
:mod:`hyplan._trochoid_solver`.

Path length and timing are expressed in the air frame (see
:attr:`DubinsPath2D.length`).

References
----------

Moon, S., Oh, E., and Shim, D.H. (2023). An integral approach to the
time-optimal Dubins path problem with trochoidal paths. *arXiv preprint*
arXiv:2306.11845.

2D Dubins solver adapted from Andrew Walker's implementation:
Walker, A. (2011). Hard Real-Time Motion Planning for Autonomous
Vehicles. PhD thesis, Swinburne University of Technology.

Dubins, L.E. (1957). On curves of minimal length with a constraint on
average curvature, and with prescribed initial and terminal positions
and tangents. *American Journal of Mathematics*, 79(3), 497-516.
doi:10.2307/2372560
"""

import math

import numpy as np
from pint import Quantity
from shapely.geometry import LineString
from shapely.ops import transform

from .geometry import get_utm_transforms
from .units import ureg
from .waypoint import Waypoint, is_waypoint
from .exceptions import HyPlanTypeError, HyPlanValueError
from typing import Any


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

    def __init__(self, qi: np.ndarray[Any, np.dtype[Any]], qf: np.ndarray[Any, np.dtype[Any]], rhomin: float,
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

    def get_coordinates_at(self, offset: float) -> np.ndarray[Any, np.dtype[Any]]:
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


def _position_in_segment(offset: float, qi: np.ndarray[Any, np.dtype[Any]], case: str) -> np.ndarray[Any, np.dtype[Any]]:
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

def _try_ccc_with_drift(qi, qf, rhomin, airspeed, wind_u, wind_v,
                        max_iter: int = 6, tol_s: float = 1e-3):
    """Iteratively solve still-air Dubins against a wind-drift-corrected goal.

    The aircraft moves at ``airspeed`` in the air frame and the air
    parcel itself drifts at ``(wind_u, wind_v)`` in the ground frame.
    Over a path of duration ``T`` the wind accumulates ``wind * T`` of
    ground drift, so an air-frame path solved against
    ``qf' = qf - wind * T`` lands at ``qf`` in the ground frame.

    This routine returns ``(air_solver, T)`` only when the resulting
    still-air optimum is **CCC** (RLR or LRL) — that's exactly the
    regime the trochoidal BSB solver gets wrong, where the geometric
    optimum is a 3-arc teardrop but BSB-only fits an LSR/RSL S-curve
    of much greater length.  When the still-air optimum is BSB,
    returns ``(None, math.inf)`` so the caller keeps the trochoidal
    BSB result.

    Args:
        qi: Start pose ``[x, y, heading]`` in ground frame
            (meters, radians).
        qf: End pose, same convention.
        rhomin: Minimum turn radius (meters).
        airspeed: True airspeed (m/s).
        wind_u: Eastward wind component (m/s).
        wind_v: Northward wind component (m/s).
        max_iter: Iteration cap (the fixed-point converges in ~3
            iterations for ``vw / Va`` up to ~0.2).
        tol_s: Convergence tolerance on ``T`` (seconds).
    """
    qf_corrected = qf.copy()
    last_T = -math.inf
    air_solver = None
    T = 0.0
    for _ in range(max_iter):
        air_solver = _Dubins2D(qi, qf_corrected, rhomin)
        air_length = float(air_solver.maneuver.length)
        if not math.isfinite(air_length):
            return None, math.inf
        T = air_length / airspeed
        if abs(T - last_T) < tol_s:
            break
        last_T = T
        qf_corrected = qf.copy()
        qf_corrected[0] = qf[0] - wind_u * T
        qf_corrected[1] = qf[1] - wind_v * T

    if air_solver is None or air_solver.maneuver.case not in ("RLR", "LRL"):
        return None, math.inf
    return air_solver, T


class _TrochoidDubins2D:
    """Wind-aware 2D Dubins solver with trochoidal ground tracks.

    Pure Python port of the Sachdev, Moon et al. (2023) algorithm from
    ``github.com/castacks/trochoids``.  Solves for the time-optimal
    BSB (Bang-Straight-Bang) trochoidal path by:

    1. Transforming to wind-aligned frame.
    2. For each BSB type, computing turning circle centers and solving
       for t1/t2 analytically (RSR/LSL) or via Newton-Raphson (RSL/LSR).
    3. Sampling the ground track using trochoidal equations (Eqs 18-21).

    For CCC (RLR/LRL) cases — needed when the start/end positions are
    closer than ~4 turn radii — the trochoidal CCC math is unstable
    (it produces multi-loop solutions).  The solver instead falls back
    to a *wind-drift-corrected* air-frame CCC: it iteratively solves
    the still-air Dubins problem against ``qf - wind * T`` so the air
    track lands at ``qf`` in the ground frame after the wind drift
    accumulates.  This covers tight racetrack patterns (line spacing
    less than turn radius) which are the dominant case where BSB-only
    solvers produce paths much longer than the geometric optimum.

    Args:
        qi: Start pose [x, y, heading] in ground frame (meters, radians).
        qf: End pose [x, y, heading] in ground frame (meters, radians).
        rhomin: Minimum turn radius in meters (air-frame).
        airspeed: True airspeed in m/s.
        wind_u: Eastward wind component in m/s.
        wind_v: Northward wind component in m/s.
        disable_ccc: When ``True``, skip the air-frame CCC fallback
            and always use the BSB trochoid (legacy behavior).

    Raises:
        HyPlanValueError: If wind speed >= airspeed (infeasible).
    """

    _EPS = 1e-6
    _M2PI = 2 * math.pi

    def __init__(self, qi, qf, rhomin, airspeed, wind_u, wind_v,
                 disable_ccc=False):
        self.qi = qi.copy()
        self.qf = qf.copy()
        self.rhomin = rhomin
        self.airspeed = airspeed
        self.wind_u = wind_u
        self.wind_v = wind_v

        wind_speed = math.sqrt(wind_u**2 + wind_v**2)
        if wind_speed >= airspeed:
            raise HyPlanValueError(
                f"Wind speed ({wind_speed:.1f} m/s) exceeds or equals "
                f"airspeed ({airspeed:.1f} m/s). Path is infeasible.")

        # Always solve the BSB trochoidal problem (existing behavior).
        from ._trochoid_solver import solve_ccc_trochoid, solve_trochoid
        self._sol = solve_trochoid(qi, qf, rhomin, airspeed, wind_u, wind_v)
        bsb_total_time = self._sol["total_time"]

        # CCC trochoid: solve the proper LRL/RLR trochoid path by
        # 1-D Newton on the half-arc-angle of the middle arc (see
        # _trochoid_solver.solve_ccc_trochoid).  When Newton converges,
        # the path lands at the ground-frame goal exactly (sub-mm) and
        # is ~2× faster than the iterative air-drift fallback.
        #
        # When Newton fails (extreme wind, near-degenerate geometry),
        # fall back to the wind-drift-corrected air-frame Dubins CCC.
        # That iteration converges linearly at rate ~vw/Va per iter, so
        # in higher wind regimes it needs many iterations to reach the
        # goal precisely — gate on the actual end-position error before
        # accepting it.
        ccc_tro_sol = None
        ccc_air_solver = None
        ccc_air_time = math.inf
        if not disable_ccc:
            ccc_tro_sol = solve_ccc_trochoid(
                qi, qf, rhomin, airspeed, wind_u, wind_v,
            )
            ccc_air_solver, ccc_air_time = _try_ccc_with_drift(
                qi, qf, rhomin, airspeed, wind_u, wind_v,
            )

        # Position-error gate on the air-drift candidate: sample the
        # path at total_time and check the end position against qf.  If
        # the iterative fixed-point hasn't converged tightly (most
        # likely in vw / Va ≳ 0.3), reject the candidate rather than
        # silently feed a wrong-end-position path to the planner.
        ccc_air_valid = False
        if ccc_air_solver is not None and math.isfinite(ccc_air_time):
            air_end = ccc_air_solver.get_coordinates_at(
                ccc_air_solver.maneuver.length,
            )
            end_x = air_end[0] + wind_u * ccc_air_time
            end_y = air_end[1] + wind_v * ccc_air_time
            pos_err = math.hypot(end_x - qf[0], end_y - qf[1])
            # 10 m tolerance is generous on a ~50 km Dubins path.
            ccc_air_valid = pos_err < 10.0

        # Pick the time-optimal valid candidate.  Tiebreak (within 1 ms)
        # prefers ccc_trochoid > ccc_air_drift > bsb so that low-wind
        # cases — where the proper trochoid and air-drift converge to
        # numerically equivalent paths — pick the rigorous solver.
        _MODE_RANK = {"ccc_trochoid": 0, "ccc_air_drift": 1, "bsb": 2}
        candidates = [(bsb_total_time, "bsb", None)]
        if ccc_tro_sol is not None:
            candidates.append(
                (ccc_tro_sol["total_time"], "ccc_trochoid", ccc_tro_sol),
            )
        if ccc_air_valid:
            candidates.append(
                (ccc_air_time, "ccc_air_drift", ccc_air_solver),
            )
        candidates.sort(key=lambda c: (round(c[0], 3), _MODE_RANK[c[1]]))
        best_time, best_mode, best_data = candidates[0]

        self._mode = best_mode
        self._total_time = best_time
        air_len = best_time * airspeed

        if best_mode == "bsb":
            self._maneuver = _DubinsSegment(0, 0, 0, air_len, "TRO")
        elif best_mode == "ccc_trochoid":
            self._ccc_tro_sol = best_data
            self._maneuver = _DubinsSegment(
                0, 0, 0, air_len, best_data["family"],
            )
        else:  # ccc_air_drift
            self._air_solver = best_data
            self._maneuver = _DubinsSegment(
                0, 0, 0, air_len, best_data.maneuver.case,
            )

    @property
    def maneuver(self) -> _DubinsSegment:
        """Air-frame maneuver (length in meters for 3D compatibility)."""
        maneuver: _DubinsSegment = self._maneuver
        return maneuver

    @property
    def total_time(self) -> float:
        """Total traversal time in seconds."""
        return float(self._total_time)

    @property
    def ground_length(self) -> float:
        """Approximate ground-track length in meters."""
        if self._total_time <= 0:
            return 0.0
        n = 50
        pts = np.array([self.get_coordinates_at(
            i * self._total_time / (n - 1)) for i in range(n)])
        diffs = np.diff(pts[:, :2], axis=0)
        return float(np.sum(np.sqrt(diffs[:, 0]**2 + diffs[:, 1]**2)))

    def get_coordinates_at(self, time_offset: float) -> np.ndarray[Any, np.dtype[Any]]:
        """Get ground-frame (x, y, heading) at a given time offset.

        Dispatches by solver mode:

        * ``bsb``: BSB trochoidal (Sachdev et al. Eqs 18-21).
        * ``ccc_trochoid``: proper LRL/RLR trochoidal three-arc path.
        * ``ccc_air_drift``: still-air Dubins CCC samples shifted by
          accumulated wind drift (used as a fallback when the
          trochoidal CCC root-find fails).

        Returned heading is the ground-track direction in all modes.
        """
        if self._mode == "ccc_trochoid":
            from ._trochoid_solver import sample_ccc_trochoid
            return sample_ccc_trochoid(
                self._ccc_tro_sol, time_offset, self.airspeed,
                self.wind_u, self.wind_v,
            )

        if self._mode == "ccc_air_drift":
            air_distance = max(
                0.0,
                min(time_offset * self.airspeed, self._maneuver.length),
            )
            air_pos = self._air_solver.get_coordinates_at(air_distance)
            gx = air_pos[0] + self.wind_u * time_offset
            gy = air_pos[1] + self.wind_v * time_offset
            air_hdg = float(air_pos[2])
            ground_heading = math.atan2(
                self.airspeed * math.sin(air_hdg) + self.wind_v,
                self.airspeed * math.cos(air_hdg) + self.wind_u,
            )
            return np.array([gx, gy, ground_heading], dtype=np.float64)

        from ._trochoid_solver import sample_trochoid
        return sample_trochoid(
            self._sol, time_offset, self.airspeed,
            self.wind_u, self.wind_v)


class DubinsPath2D:
    """Horizontal-only Dubins path between two waypoints.

    Pure plan-view geometry — bank-angle-constrained turns and straight
    segments. The vertical profile (altitude vs along-track distance) is
    intentionally not modeled here. This is the geometry consumed by the
    hybrid mission planner: solve the horizontal layout once, integrate
    altitude vs. distance separately from ``Aircraft.climb_profile`` /
    ``Aircraft.descent_profile``.

    In still air, uses the standard CSC/CCC Dubins solver. With wind,
    uses the trochoidal solver (Sachdev et al., 2023) — turning arcs
    drift with the wind, producing distorted but optimal ground tracks.

    Args:
        start: Starting waypoint (lat / lon / heading required;
            altitude is ignored).
        end: Ending waypoint (same).
        speed: True airspeed used to size the turn radius given
            ``bank_angle``. Float (m/s) or pint Quantity with speed
            units.
        bank_angle: Maximum bank angle in degrees.
        wind: Optional ``(u_east, v_north)`` wind vector in m/s. When
            provided, the horizontal path uses trochoidal geometry.
        n_samples: Number of sampled points along the path. Defaults
            to 50.

    The reported :attr:`length` is the **air-frame** path length
    (``time = length / TAS`` is the time spent traversing it). In
    still air this equals the ground-track length; with wind, ground
    distance is via the sampled :attr:`geometry`.
    """

    def __init__(
        self,
        start: Waypoint,
        end: Waypoint,
        speed: Quantity | float,
        bank_angle: float,
        *,
        wind: tuple[float, float] | None = None,
        n_samples: int = 50,
    ):
        if not is_waypoint(start) or not is_waypoint(end):
            raise HyPlanTypeError("start and end must be Waypoint objects")

        self.start = start
        self.end = end

        if isinstance(speed, (int, float)):
            self._speed_mps = float(speed)
        elif hasattr(speed, "units") and speed.check("[speed]"):
            self._speed_mps = speed.m_as(ureg.meter / ureg.second)
        else:
            raise HyPlanTypeError(
                "speed must be float (m/s) or pint Quantity with speed units"
            )

        self._bank_angle_deg = float(bank_angle)
        self._wind = wind

        g = 9.8
        bank_rad = math.radians(self._bank_angle_deg)
        if math.tan(bank_rad) <= 0:
            raise HyPlanValueError(
                f"bank_angle must be in (0, 90) degrees; got {bank_angle}"
            )
        self._rhomin = (self._speed_mps ** 2) / (g * math.tan(bank_rad))

        # UTM transforms (cached for sample_at_distance).
        to_utm, from_utm = get_utm_transforms([start.geometry, end.geometry])
        self._from_utm = from_utm
        start_utm = transform(to_utm, start.geometry)
        end_utm = transform(to_utm, end.geometry)

        # Math-frame headings (CCW from +x, i.e. east).
        heading1 = -math.radians(start.heading - 90.0)
        heading2 = -math.radians(end.heading - 90.0)

        qi = np.array([start_utm.x, start_utm.y, heading1])
        qf = np.array([end_utm.x, end_utm.y, heading2])

        self._solver: _Dubins2D | _TrochoidDubins2D
        if wind is None:
            self._solver = _Dubins2D(qi, qf, self._rhomin)
            self._length_m = float(self._solver.maneuver.length)
            self._duration_s = (
                self._length_m / self._speed_mps if self._speed_mps > 0 else 0.0
            )
        else:
            self._solver = _TrochoidDubins2D(
                qi, qf, self._rhomin, self._speed_mps, wind[0], wind[1],
            )
            self._length_m = float(self._solver.maneuver.length)  # air-frame
            self._duration_s = float(self._solver.total_time)

        # Sample the path geometry.
        self._n_samples = max(int(n_samples), 2)
        self._points = self._sample_points(self._n_samples)
        lons = self._points[:, 1]
        lats = self._points[:, 0]
        self._geometry = LineString(np.column_stack([lons, lats]))

    # -- private helpers ------------------------------------------------------

    def _sample_points(self, n: int) -> np.ndarray[Any, np.dtype[Any]]:
        """Return (n, 3) array of (lat, lon, heading_deg) along the path."""
        if self._length_m <= 0:
            single = np.array([[
                self.start.latitude, self.start.longitude, self.start.heading,
            ]])
            return single
        if self._wind is None:
            offsets = np.linspace(0.0, self._length_m, n)
            samples = [self._solver.get_coordinates_at(float(d)) for d in offsets]
        else:
            times = np.linspace(0.0, self._duration_s, n)
            samples = [self._solver.get_coordinates_at(float(t)) for t in times]
        utm = np.array([(s[0], s[1]) for s in samples])
        headings_math = np.array([s[2] for s in samples])
        lons, lats = self._from_utm(utm[:, 0], utm[:, 1])
        headings_geo = (90.0 - np.degrees(headings_math)) % 360.0
        return np.column_stack([lats, lons, headings_geo])

    # -- public surface -------------------------------------------------------

    @property
    def length(self) -> Quantity:
        """Air-frame path length (``time = length / TAS``)."""
        return self._length_m * ureg.meter

    @property
    def geometry(self) -> LineString:
        """2D ``(lon, lat)`` LineString of the path."""
        return self._geometry

    @property
    def points(self) -> np.ndarray[Any, np.dtype[Any]]:
        """Sampled path points as a ``(n, 3)`` array of ``(lat, lon, heading_deg)``."""
        return self._points

    @property
    def min_turn_radius(self) -> Quantity:
        """Minimum 2D turn radius (m) — derived from speed and bank angle."""
        return self._rhomin * ureg.meter

    def sample_at_distance(self, distance: Quantity | float) -> tuple[float, float, float]:
        """Return ``(lat, lon, heading_deg)`` at the given air-frame distance.

        Distance is clamped to ``[0, length]`` to keep the call safe at
        endpoints and at floating-point round-off boundaries.
        """
        if isinstance(distance, Quantity):
            d_m = distance.m_as(ureg.meter)
        else:
            d_m = float(distance)
        d_m = max(0.0, min(d_m, self._length_m))
        if self._length_m <= 0:
            return (self.start.latitude, self.start.longitude, self.start.heading)
        if self._wind is None:
            sample = self._solver.get_coordinates_at(d_m)
        else:
            t = d_m / self._speed_mps if self._speed_mps > 0 else 0.0
            sample = self._solver.get_coordinates_at(t)
        x, y, heading_math = sample
        lon, lat = self._from_utm(float(x), float(y))
        heading_geo = (90.0 - math.degrees(float(heading_math))) % 360.0
        return float(lat), float(lon), float(heading_geo)

    def sublinestring(
        self,
        distance_start: Quantity | float,
        distance_end: Quantity | float,
        *,
        n_samples: int = 20,
    ) -> LineString:
        """Return a multi-point ``(lon, lat)`` LineString covering the path
        between two air-frame distance offsets along it.

        Used by the hybrid mission planner to assign explicit per-phase
        geometry: each phase (climb / cruise / descent) gets a
        sub-LineString that follows the actual Dubins curve through its
        distance range, instead of a proportional time-based slice of a
        shared 3D path.
        """
        if isinstance(distance_start, Quantity):
            ds_m = distance_start.m_as(ureg.meter)
        else:
            ds_m = float(distance_start)
        if isinstance(distance_end, Quantity):
            de_m = distance_end.m_as(ureg.meter)
        else:
            de_m = float(distance_end)
        ds_m = max(0.0, min(ds_m, self._length_m))
        de_m = max(0.0, min(de_m, self._length_m))
        if de_m <= ds_m + 1e-6:
            # Degenerate slice — return a 2-point line at the start of the slice.
            lat, lon, _ = self.sample_at_distance(ds_m)
            return LineString([(lon, lat), (lon, lat)])
        n = max(int(n_samples), 2)
        distances = np.linspace(ds_m, de_m, n)
        coords = []
        for d in distances:
            lat, lon, _ = self.sample_at_distance(float(d))
            coords.append((lon, lat))
        return LineString(coords)
