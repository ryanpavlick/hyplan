from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pymap3d.vincenty
from pint import Quantity

from .airports import Airport
from .waypoint import Waypoint
from .dubins3d import DubinsPath3D
from .exceptions import HyPlanTypeError, HyPlanValueError
from .units import ureg


@dataclass
class PerformanceTable:
    """Altitude-indexed aircraft performance table.

    A small, unit-aware container holding cruise true airspeed and rate of
    climb at a set of altitude breakpoints. Intended as the primary source
    for :meth:`Aircraft.cruise_speed_at` and :meth:`Aircraft.rate_of_climb`
    lookups when populated; callers never touch this directly.

    Values at altitudes between breakpoints are linearly interpolated.
    Outside the breakpoint range the values are clamped to the endpoints
    (``np.interp`` semantics).

    This PR 1 shape covers speed/timing only. Rate of descent remains a
    per-aircraft scalar (``Aircraft.descent_rate``). The dataclass can grow
    a ``rod`` or ``fuel_flow`` column later without breaking callers.

    Args:
        rows: List of ``(altitude, cruise_tas, roc)`` triples. Altitudes
            must be strictly ascending. Each element must be a
            ``pint.Quantity`` in a compatible unit (altitude convertible to
            feet, TAS to knots, ROC to feet/minute).
        source: Free-text citation for the data, e.g.
            ``"Beechcraft King Air B200 POH, Rev. 2018-03, Section 5"``.
            Stored for traceability; not used in calculations.
    """

    rows: List[Tuple[Quantity, Quantity, Quantity]]
    source: str = ""

    def __post_init__(self) -> None:
        if len(self.rows) < 2:
            raise HyPlanValueError(
                "PerformanceTable requires at least 2 altitude rows for interpolation"
            )
        alts_ft = np.array(
            [alt.to(ureg.feet).magnitude for alt, _, _ in self.rows], dtype=float
        )
        tas_kt = np.array(
            [tas.to(ureg.knot).magnitude for _, tas, _ in self.rows], dtype=float
        )
        roc_fpm = np.array(
            [
                roc.to(ureg.feet / ureg.minute).magnitude
                for _, _, roc in self.rows
            ],
            dtype=float,
        )
        if not np.all(np.diff(alts_ft) > 0):
            raise HyPlanValueError(
                "PerformanceTable altitudes must be strictly ascending"
            )
        # Cache plain ndarrays for fast interpolation in hot loops.
        self._alts_ft = alts_ft
        self._tas_kt = tas_kt
        self._roc_fpm = roc_fpm

    def cruise_tas_at(self, altitude: Quantity) -> Quantity:
        """Interpolated cruise true airspeed at ``altitude``."""
        alt_ft = altitude.to(ureg.feet).magnitude
        return float(np.interp(alt_ft, self._alts_ft, self._tas_kt)) * ureg.knot

    def rate_of_climb_at(self, altitude: Quantity) -> Quantity:
        """Interpolated rate of climb at ``altitude``."""
        alt_ft = altitude.to(ureg.feet).magnitude
        return (
            float(np.interp(alt_ft, self._alts_ft, self._roc_fpm))
            * ureg.feet
            / ureg.minute
        )


class Aircraft:
    """
    A class representing an aircraft with performance and operational parameters.
    """

    def __init__(
        self,
        aircraft_type: str,  # aircraft model
        tail_number: str, # unique identifier
        service_ceiling: Quantity,  # feet
        approach_speed: Quantity,  # knots
        best_rate_of_climb: Quantity,  # feet per minute
        cruise_speed: Quantity,  # knots
        range: Quantity,  # nautical miles
        endurance: Quantity,  # hours
        operator: str, # organization
        max_bank_angle: float,  # degrees
        useful_payload: Quantity,  # pounds
        vx: Quantity,  # knots
        vy: Quantity,  # knots
        roc_at_service_ceiling: Quantity,  # feet per minute
        descent_rate: Quantity,  # feet per minute
        low_altitude_speed: Quantity = None,  # knots - TAS at sea level
        descent_speed_reduction: Quantity = None,  # knots - TAS decrease during descent
        speed_profile: list = None,  # list of (altitude_Quantity, speed_Quantity) breakpoints
        performance_table: Optional[PerformanceTable] = None,
    ):
        """
        Initializes an Aircraft object with performance parameters.

        Args:
            aircraft_type (str): Aircraft model or name.
            tail_number (str): Aircraft tail number or unique identifier.
            service_ceiling (Quantity): Maximum operational altitude (feet).
            approach_speed (Quantity): Landing approach speed (knots).
            best_rate_of_climb (Quantity): Maximum rate of climb (feet per minute).
            cruise_speed (Quantity): Typical cruise speed at or near service ceiling (knots).
            range (Quantity): Maximum flight range (nautical miles).
            endurance (Quantity): Maximum flight duration (hours).
            operator (str): Organization that operates the aircraft.
            max_bank_angle (float): Maximum allowable bank angle (degrees).
            useful_payload (Quantity): Maximum payload capacity (pounds).
            vx (Quantity): Best angle-of-climb speed (knots).
            vy (Quantity): Best rate-of-climb speed (knots).
            roc_at_service_ceiling (Quantity): Rate of climb at max altitude (feet per minute).
            descent_rate (Quantity): Standard rate of descent (feet per minute).
            low_altitude_speed (Quantity, optional): TAS at sea level (knots). If provided
                and speed_profile is None, cruise speed varies linearly from this value
                to cruise_speed at service ceiling. Ignored if speed_profile is set.
            descent_speed_reduction (Quantity, optional): Amount to reduce TAS during descent
                (knots). Defaults to zero.
            speed_profile (list, optional): Piecewise-linear TAS profile as a list of
                (altitude, speed) tuples, both Quantity. Altitudes must be in ascending
                order. cruise_speed_at() linearly interpolates between breakpoints and
                holds constant beyond the endpoints. Overrides low_altitude_speed and
                cruise_speed for speed lookups when set.
            performance_table (PerformanceTable, optional): Altitude-indexed
                cruise-TAS and ROC table. When supplied, takes precedence
                over ``speed_profile``, ``low_altitude_speed``, and the
                linear ROC model for :meth:`cruise_speed_at` and
                :meth:`rate_of_climb` lookups. Primary source for the
                post-PR-1 performance data rollout.

        Raises:
            TypeError: If any input has an incorrect type.
            ValueError: If an input value cannot be converted to the expected unit.
        """

        # **Validation: Ensure correct types**
        if not isinstance(aircraft_type, str):
            raise HyPlanTypeError("Aircraft type must be a string.")
        if not isinstance(tail_number, str):
            raise HyPlanTypeError("Tail number must be a string.")
        if not isinstance(operator, str):
            raise HyPlanTypeError("Operator must be a string.")
        if not isinstance(max_bank_angle, (int, float)):
            raise HyPlanTypeError("Max bank angle must be a float or int.")

        # **Automatic Unit Conversion**
        self.service_ceiling = self._convert_to_unit(service_ceiling, ureg.feet)
        self.approach_speed = self._convert_to_unit(approach_speed, ureg.knot)
        self.best_rate_of_climb = self._convert_to_unit(best_rate_of_climb, ureg.feet / ureg.minute)
        self.cruise_speed = self._convert_to_unit(cruise_speed, ureg.knot)
        self.range = self._convert_to_unit(range, ureg.nautical_mile)
        self.endurance = self._convert_to_unit(endurance, ureg.hour)
        self.useful_payload = self._convert_to_unit(useful_payload, ureg.pound)
        self.vx = self._convert_to_unit(vx, ureg.knot)
        self.vy = self._convert_to_unit(vy, ureg.knot)
        self.roc_at_service_ceiling = self._convert_to_unit(roc_at_service_ceiling, ureg.feet / ureg.minute)
        self.descent_rate = self._convert_to_unit(descent_rate, ureg.feet / ureg.minute)
        self.low_altitude_speed = (
            self._convert_to_unit(low_altitude_speed, ureg.knot)
            if low_altitude_speed is not None else None
        )
        self.descent_speed_reduction = (
            self._convert_to_unit(descent_speed_reduction, ureg.knot)
            if descent_speed_reduction is not None else 0 * ureg.knot
        )

        # Performance table (altitude-indexed cruise TAS + ROC). When present
        # it wins over speed_profile / low_altitude_speed / linear ROC in the
        # dispatch inside cruise_speed_at() and rate_of_climb().
        self.performance_table = performance_table

        # Build speed profile lookup arrays (altitudes in feet, speeds in knots)
        if speed_profile is not None:
            self._profile_alts = np.array([
                self._convert_to_unit(alt, ureg.feet).magnitude for alt, _ in speed_profile
            ])
            self._profile_speeds = np.array([
                self._convert_to_unit(spd, ureg.knot).magnitude for _, spd in speed_profile
            ])
        else:
            self._profile_alts = None
            self._profile_speeds = None

        # Assign validated string and numeric values
        self.aircraft_type = aircraft_type
        self.tail_number = tail_number
        self.operator = operator
        self.max_bank_angle = float(max_bank_angle)  # Ensure float type

    @staticmethod
    def _convert_to_unit(value: Quantity, expected_unit: Quantity) -> Quantity:
        """
        Converts the input value to the expected unit.

        Args:
            value (Quantity): The input quantity with a unit.
            expected_unit (Quantity): The expected unit for conversion.

        Returns:
            Quantity: The value converted to the expected unit.

        Raises:
            TypeError: If the input is not a Quantity.
        """
        if not isinstance(value, Quantity):
            raise HyPlanTypeError(f"Expected a pint.Quantity for {expected_unit}, but got {type(value)}.")

        # Convert to the expected unit
        return value.to(expected_unit)

    def rate_of_climb(self, altitude: Quantity) -> Quantity:
        """
        Compute the rate of climb at a given altitude.

        Dispatch order:

        1. If a ``performance_table`` is set, interpolate ROC from it.
        2. Otherwise assume a linear decrease from ``best_rate_of_climb`` at
           sea level to ``roc_at_service_ceiling`` at the service ceiling.

        Args:
            altitude (Quantity): Current altitude (feet or convertible).

        Returns:
            Quantity: Rate of climb at the given altitude (feet per minute).
        """
        if self.performance_table is not None:
            return self.performance_table.rate_of_climb_at(altitude)

        if altitude >= self.service_ceiling:
            return self.roc_at_service_ceiling

        altitude_ratio = altitude / self.service_ceiling
        roc = (1 - altitude_ratio) * (self.best_rate_of_climb - self.roc_at_service_ceiling) + self.roc_at_service_ceiling
        return roc

    def cruise_speed_at(self, altitude: Quantity) -> Quantity:
        """
        Compute true airspeed at a given altitude.

        Dispatch order:

        1. If a ``performance_table`` is set, interpolate cruise TAS from it.
        2. Else if a ``speed_profile`` is set, linearly interpolate between
           breakpoints (constant beyond endpoints).
        3. Else if ``low_altitude_speed`` is set, linearly interpolate between
           it and ``cruise_speed``.
        4. Else return the constant ``cruise_speed``.
        """
        if self.performance_table is not None:
            return self.performance_table.cruise_tas_at(altitude)

        altitude = altitude.to(ureg.feet)
        if self._profile_alts is not None:
            speed_kt = np.interp(
                altitude.magnitude, self._profile_alts, self._profile_speeds
            )
            return speed_kt * ureg.knot
        if self.low_altitude_speed is None:
            return self.cruise_speed
        fraction = min((altitude / self.service_ceiling).magnitude, 1.0)
        return self.low_altitude_speed + fraction * (self.cruise_speed - self.low_altitude_speed)

    def pitch_limits(self, speed: Optional[Quantity] = None) -> tuple:
        """
        Derive pitch angle limits from climb/descent rates and true airspeed.

        Args:
            speed: True airspeed override. Defaults to cruise_speed.

        Returns:
            (pitch_min, pitch_max) in degrees. pitch_min is negative (descent),
            pitch_max is positive (climb).
        """
        tas = speed if speed is not None else self.cruise_speed
        tas_mps = tas.m_as(ureg.meter / ureg.second)
        climb_mps = self.best_rate_of_climb.m_as(ureg.meter / ureg.minute) / 60.0
        descent_mps = self.descent_rate.m_as(ureg.meter / ureg.minute) / 60.0
        pitch_max = float(np.degrees(np.arctan(climb_mps / tas_mps)))
        pitch_min = -float(np.degrees(np.arctan(descent_mps / tas_mps)))
        return pitch_min, pitch_max

    def descent_speed_at(self, altitude: Quantity) -> Quantity:
        """
        Compute true airspeed during descent at a given altitude.

        Returns the cruise speed at that altitude minus the descent_speed_reduction.

        Args:
            altitude (Quantity): Current altitude (feet or convertible).

        Returns:
            Quantity: Descent TAS in knots.
        """
        return self.cruise_speed_at(altitude) - self.descent_speed_reduction

    def _climb(
        self,
        start_altitude: Quantity,
        end_altitude: Quantity,
        true_air_speed: Optional[Quantity] = None,
    ) -> tuple[Quantity, Quantity]:
        """
        Estimate the time and horizontal distance traveled during a Vy climb.

        Uses the analytical solution to dh/dt = ROC(h) where ROC decreases
        linearly with altitude:

            ROC(h) = ROC_ceil + (ROC_sl - ROC_ceil) * (1 - h/C)

        The exact time to climb from h0 to h1 is:

            t = C / (ROC_sl - ROC_ceil) * ln(ROC(h0) / ROC(h1))

        This produces a realistic curved climb profile that starts steep
        at low altitude and flattens as rate of climb decreases.
        """
        start_altitude = start_altitude.to(ureg.feet)
        end_altitude = end_altitude.to(ureg.feet)
        if true_air_speed is None:
            avg_alt = (start_altitude + end_altitude) / 2
            true_air_speed = self.cruise_speed_at(avg_alt)
        true_air_speed = true_air_speed.to(ureg.feet / ureg.minute)

        if end_altitude > self.service_ceiling:
            raise HyPlanValueError("End altitude cannot exceed the service ceiling.")
        if end_altitude <= start_altitude:
            return 0 * ureg.minute, 0 * ureg.nautical_mile

        roc_start = self.rate_of_climb(start_altitude)
        roc_end = self.rate_of_climb(end_altitude)

        if self.performance_table is not None:
            # Numerical integration of dt = dh / ROC(h). ROC(h) from a
            # table is piecewise-linear but not globally linear, so the
            # closed-form below doesn't apply. 64 steps is plenty given
            # typical cruise altitudes and ROC smoothness.
            n_steps = 64
            alts_ft = np.linspace(
                start_altitude.magnitude, end_altitude.magnitude, n_steps + 1
            )
            rocs_fpm = np.array(
                [
                    self.rate_of_climb(ureg.Quantity(a, "feet"))
                    .m_as(ureg.feet / ureg.minute)
                    for a in alts_ft
                ]
            )
            # Trapezoidal integration of 1/ROC with respect to altitude.
            inv_roc = 1.0 / rocs_fpm
            minutes = float(np.trapezoid(inv_roc, alts_ft))
            time_to_climb = minutes * ureg.minute
        else:
            C = self.service_ceiling
            roc_sl = self.best_rate_of_climb
            roc_ceil = self.roc_at_service_ceiling
            delta_roc = (roc_sl - roc_ceil).magnitude

            if delta_roc < 1e-6:
                # Constant ROC — degenerate case
                time_to_climb = ((end_altitude - start_altitude) / roc_sl).to(ureg.minute)
            else:
                # Analytical solution: t = C / (ROC_sl - ROC_ceil) * ln(ROC(h0) / ROC(h1))
                time_to_climb = (
                    C / (roc_sl - roc_ceil) * np.log(roc_start / roc_end)
                ).to(ureg.minute)

        # Horizontal distance using average climb angle
        avg_roc = (roc_start + roc_end) / 2
        climb_angle = np.arctan(avg_roc / true_air_speed).to(ureg.radian)
        horizontal_speed = (true_air_speed * np.cos(climb_angle)).to(ureg.nautical_mile / ureg.hour)
        horizontal_distance = (horizontal_speed * time_to_climb).to(ureg.nautical_mile)

        return time_to_climb, horizontal_distance

    def climb_altitude_profile(
        self, start_altitude: Quantity, end_altitude: Quantity, n_points: int = 50
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Generate the altitude-vs-time curve during a climb.

        Returns arrays suitable for plotting the realistic (exponential) climb
        profile rather than a straight line.

        Args:
            start_altitude: Start altitude (Quantity in feet or convertible).
            end_altitude: End altitude (Quantity in feet or convertible).
            n_points: Number of points in the profile.

        Returns:
            (times, altitudes) — numpy arrays in minutes and feet.
        """
        start_altitude = start_altitude.to(ureg.feet)
        end_altitude = end_altitude.to(ureg.feet)

        if end_altitude <= start_altitude:
            return np.array([0.0]), np.array([start_altitude.magnitude])

        if self.performance_table is not None:
            # Numerical integration of t(h) = ∫ dh / ROC(h) against a
            # piecewise-linear ROC curve. Returns a monotonically
            # increasing time array aligned to n_points altitude samples.
            altitudes = np.linspace(
                start_altitude.magnitude, end_altitude.magnitude, n_points
            )
            rocs = np.array(
                [
                    self.rate_of_climb(ureg.Quantity(a, "feet"))
                    .m_as(ureg.feet / ureg.minute)
                    for a in altitudes
                ]
            )
            inv_roc = 1.0 / rocs
            # Cumulative trapezoidal integration.
            dh = np.diff(altitudes)
            seg = 0.5 * (inv_roc[:-1] + inv_roc[1:]) * dh
            times = np.concatenate([[0.0], np.cumsum(seg)])
            return times, altitudes

        h0 = start_altitude.magnitude
        C = self.service_ceiling.magnitude
        roc_sl = self.best_rate_of_climb.m_as(ureg.feet / ureg.minute)
        roc_ceil = self.roc_at_service_ceiling.m_as(ureg.feet / ureg.minute)
        delta_roc = roc_sl - roc_ceil

        if delta_roc < 1e-6:
            # Constant ROC
            total_time = (end_altitude - start_altitude).magnitude / roc_sl
            times = np.linspace(0, total_time, n_points)
            altitudes = h0 + roc_sl * times
        else:
            # h(t) = h_eq - (h_eq - h0) * exp(-alpha * t)
            # where alpha = delta_roc / C, h_eq = C * roc_sl / delta_roc
            alpha = delta_roc / C
            h_eq = C * roc_sl / delta_roc  # theoretical equilibrium altitude
            roc_h0 = self.rate_of_climb(start_altitude).m_as(ureg.feet / ureg.minute)
            roc_h1 = self.rate_of_climb(end_altitude).m_as(ureg.feet / ureg.minute)
            total_time = (1 / alpha) * np.log(roc_h0 / roc_h1)

            times = np.linspace(0, total_time, n_points)
            altitudes = h_eq - (h_eq - h0) * np.exp(-alpha * times)

        return times, altitudes

    def time_to_takeoff(self, airport: Airport, waypoint: Waypoint) -> dict:
        """
        Calculate the time to take off from an airport and reach a waypoint.

        Uses 3D Dubins path planning to model the departure path including
        the climb from airport elevation to cruise altitude.

        Args:
            airport (Airport): Departure airport.
            waypoint (Waypoint): First flight waypoint at cruise altitude.

        Returns:
            dict: Keys ``total_time``, ``phases``, and ``dubins_path``.
        """
        _, departure_heading = pymap3d.vincenty.vdist(
            airport.latitude, airport.longitude,
            waypoint.latitude, waypoint.longitude,
        )
        airport_waypoint = Waypoint(
            latitude=airport.latitude, longitude=airport.longitude,
            heading=departure_heading, altitude_msl=airport.elevation,
        )
        return self.time_to_cruise(airport_waypoint, waypoint)

    def time_to_return(self, waypoint: Waypoint, airport: Airport) -> dict:
        """
        Calculate the time to return from a waypoint to an airport.

        Uses 3D Dubins path planning to model the return path including
        the descent from cruise altitude to airport elevation.

        Args:
            waypoint (Waypoint): Last flight waypoint at cruise altitude.
            airport (Airport): Destination airport.

        Returns:
            dict: Keys ``total_time``, ``phases``, and ``dubins_path``.
        """
        _, arrival_heading = pymap3d.vincenty.vdist(
            waypoint.latitude, waypoint.longitude,
            airport.latitude, airport.longitude,
        )
        airport_waypoint = Waypoint(
            latitude=airport.latitude, longitude=airport.longitude,
            heading=(arrival_heading + 180.0) % 360.0,
            altitude_msl=airport.elevation,
        )
        return self.time_to_cruise(waypoint, airport_waypoint)

    def _descend(
        self,
        start_altitude: Quantity,
        end_altitude: Quantity,
        true_air_speed: Optional[Quantity] = None,
    ) -> tuple[Quantity, Quantity]:
        """
        Estimate the time and horizontal distance traveled during descent.

        Uses a constant descent rate and computes horizontal distance from
        the descent angle and true airspeed.

        Args:
            start_altitude (Quantity): Starting altitude (feet or convertible).
            end_altitude (Quantity): Target altitude (feet or convertible).
            true_air_speed (Quantity, optional): TAS override. Defaults to
                descent_speed_at the average of start and end altitudes.

        Returns:
            tuple: (time_to_descend, horizontal_distance) as Quantity objects
                in minutes and nautical miles respectively.
        """
        start_altitude = start_altitude.to(ureg.feet)
        end_altitude = end_altitude.to(ureg.feet)
        if true_air_speed is None:
            avg_alt = (start_altitude + end_altitude) / 2
            true_air_speed = self.descent_speed_at(avg_alt)
        true_air_speed = true_air_speed.to(ureg.feet / ureg.minute)

        if start_altitude <= end_altitude:
            return 0 * ureg.minute, 0 * ureg.nautical_mile

        altitude_difference = start_altitude - end_altitude
        time_to_descend = (altitude_difference / self.descent_rate).to(ureg.minute)

        descent_angle = np.arctan(self.descent_rate / true_air_speed).to(ureg.radian)
        horizontal_speed = (true_air_speed * np.cos(descent_angle)).to(ureg.nautical_mile / ureg.hour)
        horizontal_distance = (horizontal_speed * time_to_descend).to(ureg.nautical_mile)

        return time_to_descend, horizontal_distance

    def time_to_cruise(
        self,
        start_waypoint: Waypoint,
        end_waypoint: Waypoint,
        true_air_speed: Optional[Quantity] = None,
    ) -> dict:
        """
        Calculate the time to cruise between two waypoints.

        Uses 3D Dubins path planning with pitch constraints derived from
        the aircraft's climb/descent performance. Accounts for altitude
        changes in the turn geometry.

        Args:
            start_waypoint (Waypoint): Starting waypoint.
            end_waypoint (Waypoint): Ending waypoint.
            true_air_speed (Quantity, optional): TAS override. Defaults to
                the altitude-dependent cruise speed at the end waypoint.

        Returns:
            dict: Keys ``total_time``, ``phases`` (with optional
            ``cruise_climb``, ``cruise_descent``, and ``cruise`` sub-dicts),
            and ``dubins_path``.
        """
        true_air_speed = true_air_speed or self.cruise_speed_at(end_waypoint.altitude_msl)

        start_altitude = start_waypoint.altitude_msl.to(ureg.feet)
        end_altitude = end_waypoint.altitude_msl.to(ureg.feet)

        pitch_min, pitch_max = self.pitch_limits(true_air_speed)

        path = DubinsPath3D(
            start=start_waypoint,
            end=end_waypoint,
            speed=true_air_speed,
            bank_angle=self.max_bank_angle,
            pitch_min=pitch_min,
            pitch_max=pitch_max,
        )

        distance = path.length.to(ureg.nautical_mile)
        total_time = (distance / true_air_speed).to(ureg.minute)

        phases = self._phases_from_3d_path(
            path, true_air_speed, start_altitude, end_altitude, total_time,
        )

        return {
            "total_time": total_time,
            "phases": phases,
            "dubins_path": path,
        }

    def _phases_from_3d_path(
        self,
        path3d,
        true_air_speed: Quantity,
        start_altitude: Quantity,
        end_altitude: Quantity,
        total_time: Quantity,
    ) -> dict:
        """Split a 3D Dubins path into climb/cruise/descent phases by altitude profile."""
        pts = path3d.points  # (N, 5): lat, lon, alt_m, heading, pitch
        alts_m = pts[:, 2]
        n = len(alts_m)

        if n < 2:
            return {"cruise": {
                "start_altitude": start_altitude,
                "end_altitude": end_altitude,
                "start_time": 0 * ureg.minute,
                "end_time": total_time,
                "distance": path3d.length.to(ureg.nautical_mile),
            }}

        # Classify each segment by pitch sign (more robust than altitude diff
        # which can be tiny per-sample). Use pitch from the points array.
        pitches = pts[:, 4]  # degrees
        # Average pitch over each segment (between consecutive samples)
        seg_pitches = (pitches[:-1] + pitches[1:]) / 2.0
        pitch_threshold_deg = 0.001  # very small to catch shallow climbs/descents
        segment_types = np.where(
            seg_pitches > pitch_threshold_deg, 1,
            np.where(seg_pitches < -pitch_threshold_deg, -1, 0)
        )

        # Group consecutive segments of the same type into phases
        phases = {}
        i = 0
        phase_idx = 0

        while i < len(segment_types):
            seg_type = segment_types[i]
            j = i
            while j < len(segment_types) and segment_types[j] == seg_type:
                j += 1

            # This phase spans sample indices i to j (inclusive of start, exclusive of end)
            frac_start = i / (n - 1)
            frac_end = j / (n - 1)
            phase_time_start = total_time * frac_start
            phase_time_end = total_time * frac_end
            phase_distance = path3d.length.to(ureg.nautical_mile) * (frac_end - frac_start)

            phase_start_alt = ureg.Quantity(float(alts_m[i]), "meter").to(ureg.feet)
            phase_end_alt = ureg.Quantity(float(alts_m[min(j, n - 1)]), "meter").to(ureg.feet)

            if seg_type == 1:
                label = "cruise_climb"
            elif seg_type == -1:
                label = "cruise_descent"
            else:
                label = "cruise"

            # Disambiguate duplicate labels
            key = label if label not in phases else f"{label}_{phase_idx}"

            phases[key] = {
                "start_altitude": phase_start_alt,
                "end_altitude": phase_end_alt,
                "start_time": phase_time_start.to(ureg.minute),
                "end_time": phase_time_end.to(ureg.minute),
                "distance": phase_distance,
            }

            phase_idx += 1
            i = j

        return phases

#%% Concrete aircraft models
#
# The 14 NASA / operator subclasses live in hyplan.aircraft_models. They are
# re-exported below so existing code that does ``from hyplan.aircraft import
# NASA_ER2`` continues to work. The import sits at the bottom of this file
# (after ``class Aircraft`` is fully defined) to avoid a circular import,
# since aircraft_models.py does ``from .aircraft import Aircraft``.
from .aircraft_models import (  # noqa: E402
    NASA_ER2,
    NASA_GIII,
    NASA_GIV,
    NASA_C20A,
    NASA_P3,
    NASA_WB57,
    NASA_B777,
    DynamicAviation_DH8,
    DynamicAviation_A90,
    DynamicAviation_B200,
    C130,
    BAe146,
    Learjet,
    TwinOtter,
)

__all__ = [
    "Aircraft",
    "PerformanceTable",
    "NASA_ER2",
    "NASA_GIII",
    "NASA_GIV",
    "NASA_C20A",
    "NASA_P3",
    "NASA_WB57",
    "NASA_B777",
    "DynamicAviation_DH8",
    "DynamicAviation_A90",
    "DynamicAviation_B200",
    "C130",
    "BAe146",
    "Learjet",
    "TwinOtter",
]

