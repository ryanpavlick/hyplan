import numpy as np
import logging
import pymap3d.vincenty
from .units import ureg
from .airports import Airport
from .dubins_path import Waypoint, DubinsPath
from pint import Quantity

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

        Raises:
            TypeError: If any input has an incorrect type.
            ValueError: If an input value cannot be converted to the expected unit.
        """

        # **Validation: Ensure correct types**
        if not isinstance(aircraft_type, str):
            raise TypeError("Aircraft type must be a string.")
        if not isinstance(tail_number, str):
            raise TypeError("Tail number must be a string.")
        if not isinstance(operator, str):
            raise TypeError("Operator must be a string.")
        if not isinstance(max_bank_angle, (int, float)):
            raise TypeError("Max bank angle must be a float or int.")

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
            raise TypeError(f"Expected a pint.Quantity for {expected_unit}, but got {type(value)}.")

        # Convert to the expected unit
        return value.to(expected_unit)

    def rate_of_climb(self, altitude):
        """
        Compute the rate of climb at a given altitude.

        Assumes a linear decrease from best_rate_of_climb at sea level
        to roc_at_service_ceiling at the service ceiling.

        Args:
            altitude (Quantity): Current altitude (feet or convertible).

        Returns:
            Quantity: Rate of climb at the given altitude (feet per minute).
        """
        if altitude >= self.service_ceiling:
            return self.roc_at_service_ceiling
        
        altitude_ratio = altitude / self.service_ceiling
        roc = (1 - altitude_ratio) * (self.best_rate_of_climb - self.roc_at_service_ceiling) + self.roc_at_service_ceiling
        return roc

    def cruise_speed_at(self, altitude: Quantity) -> Quantity:
        """
        Compute true airspeed at a given altitude.

        If a speed_profile is set, linearly interpolates between breakpoints
        (constant beyond endpoints). Otherwise falls back to linear interpolation
        between low_altitude_speed and cruise_speed, or constant cruise_speed.
        """
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

    def _climb(self, start_altitude, end_altitude, true_air_speed=None):
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
        try:
            start_altitude = start_altitude.to(ureg.feet)
            end_altitude = end_altitude.to(ureg.feet)
            if true_air_speed is None:
                avg_alt = (start_altitude + end_altitude) / 2
                true_air_speed = self.cruise_speed_at(avg_alt)
            true_air_speed = true_air_speed.to(ureg.feet / ureg.minute)

            if end_altitude > self.service_ceiling:
                raise ValueError("End altitude cannot exceed the service ceiling.")
            if end_altitude <= start_altitude:
                return 0 * ureg.minute, 0 * ureg.nautical_mile

            roc_start = self.rate_of_climb(start_altitude)
            roc_end = self.rate_of_climb(end_altitude)
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

        except Exception as e:
            logging.error(f"Error in time_to_climb: {e}")
            raise

    def climb_altitude_profile(self, start_altitude, end_altitude, n_points=50):
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

        h0 = start_altitude.magnitude
        C = self.service_ceiling.magnitude
        roc_sl = self.best_rate_of_climb.to(ureg.feet / ureg.minute).magnitude
        roc_ceil = self.roc_at_service_ceiling.to(ureg.feet / ureg.minute).magnitude
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
            roc_h0 = roc_ceil + delta_roc * (1 - h0 / C)
            roc_h1 = roc_ceil + delta_roc * (1 - end_altitude.magnitude / C)
            total_time = (1 / alpha) * np.log(roc_h0 / roc_h1)

            times = np.linspace(0, total_time, n_points)
            altitudes = h_eq - (h_eq - h0) * np.exp(-alpha * times)

        return times, altitudes

    def time_to_takeoff(self, airport: Airport, waypoint: Waypoint):
        """
        Calculate the total time needed to take off from an airport and reach a waypoint at cruise altitude,
        with detailed altitude and time information for each phase.
        """
        try:
            airport_altitude = airport.elevation.to(ureg.feet)
            waypoint_altitude = waypoint.altitude_msl.to(ureg.feet)

            # Climb phase
            climb_time, climb_distance = self._climb(airport_altitude, waypoint_altitude, true_air_speed=self.vy)

            _, departure_heading = pymap3d.vincenty.vdist(airport.latitude, airport.longitude, waypoint.latitude, waypoint.longitude)
            airport_waypoint = Waypoint(latitude=airport.latitude, longitude=airport.longitude, heading=departure_heading, altitude_msl=airport_altitude)
            # Cruise phase
            dubins_path = DubinsPath(
                start=airport_waypoint,
                end=waypoint,
                speed=self.cruise_speed,
                bank_angle=self.max_bank_angle,
                step_size=100
            )
            total_distance = dubins_path.length.to(ureg.nautical_mile)
            cruise_distance = max(0 * ureg.nautical_mile, total_distance - climb_distance)
            cruise_time = (cruise_distance / self.cruise_speed_at(waypoint_altitude)).to(ureg.minute)

            total_time = climb_time + cruise_time

            # Return detailed phase information
            return {
                "total_time": total_time,
                "phases": {
                    "takeoff_climb": {
                        "start_altitude": airport_altitude,
                        "end_altitude": waypoint_altitude,
                        "start_heading": airport_waypoint.heading,
                        "start_time": 0 * ureg.minute,
                        "end_time": climb_time,
                        "distance": climb_distance
                    },
                    "takeoff_cruise": {
                        "start_altitude": waypoint_altitude,
                        "end_altitude": waypoint_altitude,
                        "start_heading": airport_waypoint.heading,
                        "start_time": climb_time,
                        "end_time": total_time,
                        "distance": cruise_distance
                    },
                },
                "dubins_path": dubins_path
            }

        except Exception as e:
            logging.error(f"Error in time_to_takeoff: {e}")
            raise


    def time_to_return(self, waypoint: Waypoint, airport: Airport):
        """
        Calculate the total time needed to return to an airport during an IFR landing,
        with detailed altitude and time information for each phase.
        """
        try:
            _, arrival_heading = pymap3d.vincenty.vdist(waypoint.latitude, waypoint.longitude, airport.latitude, airport.longitude)
            airport_waypoint = Waypoint(latitude=airport.latitude, longitude=airport.longitude, heading=(arrival_heading + 180.0) % 360.0, altitude_msl=airport.elevation)
            dubins_path = DubinsPath(
                start=waypoint,
                end=airport_waypoint,
                speed=self.cruise_speed,
                bank_angle=self.max_bank_angle,
                step_size=100
            )
            total_distance = dubins_path.length.to(ureg.nautical_mile)

            # Cruise phase
            cruise_altitude = waypoint.altitude_msl.to(ureg.feet)
            approach_altitude = min(airport.elevation.to(ureg.feet) + 5_000 * ureg.feet, cruise_altitude)
            descent_altitude = cruise_altitude - approach_altitude
            descent_distance = 3 * (descent_altitude.to(ureg.feet).magnitude / 1_000) * ureg.nautical_mile
            cruise_distance = max(0 * ureg.nautical_mile, total_distance - descent_distance)
            cruise_time = (cruise_distance / self.cruise_speed_at(cruise_altitude)).to(ureg.minute)

            # Descent phase
            descent_time, descent_distance_actual = self._descend(cruise_altitude, approach_altitude)

            # Approach phase
            if_to_faf_distance = 16 * ureg.nautical_mile
            faf_to_runway_distance = 6 * ureg.nautical_mile
            average_speed_if_to_faf = (self.cruise_speed + self.approach_speed) / 2
            if_to_faf_time = (if_to_faf_distance / average_speed_if_to_faf).to(ureg.minute)
            faf_to_runway_time = (faf_to_runway_distance / self.approach_speed).to(ureg.minute)
            approach_time = if_to_faf_time + faf_to_runway_time

            total_time = cruise_time + descent_time + approach_time

            # Return detailed phase information
            return {
                "total_time": total_time,
                "phases": {
                    "return_cruise": {
                        "start_altitude": cruise_altitude,
                        "end_altitude": cruise_altitude,
                        "start_time": 0 * ureg.minute,
                        "end_time": cruise_time,
                        "end_heading": airport_waypoint.heading,
                        "distance": cruise_distance
                    },
                    "return_descent": {
                        "start_altitude": cruise_altitude,
                        "end_altitude": approach_altitude,
                        "start_time": cruise_time,
                        "end_time": cruise_time + descent_time,
                        "end_heading": airport_waypoint.heading,
                        "distance": descent_distance_actual
                    },
                    "return_approach": {
                        "start_altitude": approach_altitude,
                        "end_altitude": airport.elevation.to(ureg.feet),
                        "start_time": cruise_time + descent_time,
                        "end_time": total_time,
                        "end_heading": airport_waypoint.heading,
                        "distance": if_to_faf_distance + faf_to_runway_distance
                    },
                },
                "dubins_path": dubins_path
            }


        except Exception as e:
            logging.error(f"Error in time_to_return: {e}")
            raise

    def _descend(self, start_altitude, end_altitude, true_air_speed=None):
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
        try:
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

        except Exception as e:
            logging.error(f"Error in time_to_descend: {e}")
            raise

    def time_to_cruise(self, start_waypoint: Waypoint, end_waypoint: Waypoint, true_air_speed=None):
        """
        Calculate the time required to travel between two waypoints using DubinsPath.
        Returns a structure similar to time_to_takeoff and time_to_return.
        """
        try:
            # Default true airspeed is altitude-dependent cruise speed
            true_air_speed = true_air_speed or self.cruise_speed_at(end_waypoint.altitude_msl_msl)

            start_altitude = start_waypoint.altitude_msl_msl.to(ureg.feet)
            end_altitude = end_waypoint.altitude_msl_msl.to(ureg.feet)

            climb_time, climb_distance = (0 * ureg.minute, 0 * ureg.nautical_mile)
            descent_time, descent_distance = (0 * ureg.minute, 0 * ureg.nautical_mile)

            if start_altitude < end_altitude:
                climb_time, climb_distance = self._climb(start_altitude, end_altitude, true_air_speed)
            elif start_altitude > end_altitude:
                descent_time, descent_distance = self._descend(start_altitude, end_altitude, true_air_speed)

            # Use DubinsPath for all distance calculations
            dubins_path = DubinsPath(
                start=start_waypoint,
                end=end_waypoint,
                speed=true_air_speed,
                bank_angle=self.max_bank_angle,
                step_size=100
            )
            distance = dubins_path.length.to(ureg.nautical_mile)
            path = dubins_path

            cruise_distance = max(0 * ureg.nautical_mile, distance - climb_distance - descent_distance)
            cruise_time = (cruise_distance / true_air_speed).to(ureg.minute)

            total_time = climb_time + cruise_time + descent_time

            phases = {}

            if climb_time > 0:
                phases["cruise_climb"] = {
                    "start_altitude": start_altitude,
                    "end_altitude": end_altitude,
                    "start_time": 0 * ureg.minute,
                    "end_time": climb_time,
                    "distance": climb_distance
                }

            if descent_time > 0:
                phases["cruise_descent"] = {
                    "start_altitude": start_altitude,
                    "end_altitude": end_altitude,
                    "start_time": 0 * ureg.minute,
                    "end_time": descent_time,
                    "distance": descent_distance
                }

            if cruise_time > 0:
                phases["cruise"] = {
                    "start_altitude": end_altitude,
                    "end_altitude": end_altitude,
                    "start_time": climb_time + descent_time,
                    "end_time": climb_time + descent_time + cruise_time,
                    "distance": cruise_distance
                }

            return {
                "total_time": total_time,
                "phases": phases,
                "dubins_path": path
            }

        except Exception as e:
            logging.error(f"Error in time_to_cruise: {e}")
            raise

#%% Aircraft Definitions

class NASA_ER2(Aircraft):
    """
    NASA ER-2 high-altitude research aircraft.

    Operates at 70,000 ft, acquiring data above 95% of the Earth's atmosphere.
    Based at NASA Armstrong Flight Research Center (AFRC).

    Speed profile from Moving Lines: TAS = 70 + alt_m * 0.0071 (m/s).
    Linear increase to ceiling with no cap (cap altitude exceeds ceiling).

    See also:
        https://airbornescience.nasa.gov/aircraft/ER-2_-_AFRC
    """
    def __init__(self):
        super().__init__(
            aircraft_type="ER-2",
            tail_number="NASA 806",
            service_ceiling=70000 * ureg.feet,
            approach_speed=130 * ureg.knot,
            best_rate_of_climb=5000 * ureg.feet / ureg.minute,
            cruise_speed=410 * ureg.knot,
            range=5000 * ureg.nautical_mile,
            endurance=8 * ureg.hour,
            operator="NASA AFRC",
            max_bank_angle=30.0,
            useful_payload=2900 * ureg.pound,
            vx=140 * ureg.knot,
            vy=160 * ureg.knot,
            roc_at_service_ceiling=500.0 * ureg.feet / ureg.minute,
            descent_rate=1500 * ureg.feet / ureg.minute,
            descent_speed_reduction=0 * ureg.knot,
            speed_profile=[
                (0 * ureg.feet, 136 * ureg.knot),
                (70000 * ureg.feet, 431 * ureg.knot),
            ],
        )

class NASA_GIII(Aircraft):
    """
    NASA Gulfstream III (NASA 520) research aircraft.

    Operated by NASA Langley Research Center (LaRC) for earth science
    research and remote sensing missions.

    See also:
        https://airbornescience.nasa.gov/aircraft/Gulfstream_III_-_LaRC
    """
    def __init__(self):
        super().__init__(
            aircraft_type="Gulfstream III",
            tail_number="NASA 520",
            service_ceiling=45000 * ureg.feet,
            approach_speed=140 * ureg.knot,
            best_rate_of_climb=4000 * ureg.feet / ureg.minute,
            cruise_speed=459 * ureg.knot,
            range=3767 * ureg.nautical_mile,
            endurance=7.5 * ureg.hour,
            operator="NASA LaRC",
            max_bank_angle=30.0,
            useful_payload=2610 * ureg.pound,
            vx=140 * ureg.knot,
            vy=160 * ureg.knot,
            roc_at_service_ceiling=500.0 * ureg.feet / ureg.minute,
            descent_rate=1500 * ureg.feet / ureg.minute,
            descent_speed_reduction=49 * ureg.knot,
            speed_profile=[
                (0 * ureg.feet, 292 * ureg.knot),
                (45000 * ureg.feet, 459 * ureg.knot),
            ],
        )

class NASA_GIV(Aircraft):
    """
    NASA Gulfstream IV (NASA 817) research aircraft.

    Twin turbofan business-class aircraft operated by NASA Armstrong
    Flight Research Center (AFRC). Useful payload of 5,610 lbs.

    See also:
        https://airbornescience.nasa.gov/aircraft/Gulfstream_IV_-_AFRC
    """
    def __init__(self):
        super().__init__(
            aircraft_type="Gulfstream IV",
            tail_number="NASA 817",
            service_ceiling=45000 * ureg.feet,
            approach_speed=140 * ureg.knot,
            best_rate_of_climb=4000 * ureg.feet / ureg.minute,
            cruise_speed=459 * ureg.knot,
            range=5130 * ureg.nautical_mile,
            endurance=7.5 * ureg.hour,
            operator="NASA AFRC",
            max_bank_angle=30.0,
            useful_payload=5610 * ureg.pound,
            vx=150 * ureg.knot,
            vy=170 * ureg.knot,
            roc_at_service_ceiling=500.0 * ureg.feet / ureg.minute,
            descent_rate=1500 * ureg.feet / ureg.minute,
            descent_speed_reduction=49 * ureg.knot,
            speed_profile=[
                (0 * ureg.feet, 292 * ureg.knot),
                (45000 * ureg.feet, 459 * ureg.knot),
            ],
        )

class NASA_C20A(Aircraft):
    """
    NASA C-20A (Gulfstream III variant, NASA 502) research aircraft.

    A business jet structurally modified and instrumented by NASA Armstrong
    Flight Research Center to serve as a multi-role cooperative research
    platform. Obtained from the U.S. Air Force in 2003. Primary platform
    for UAVSAR missions.

    See also:
        https://airbornescience.nasa.gov/aircraft/Gulfstream_C-20A_GIII_-_AFRC
    """
    def __init__(self):
        super().__init__(
            aircraft_type="C-20A",
            tail_number="NASA 502",
            service_ceiling=45000 * ureg.feet,
            approach_speed=140 * ureg.knot,
            best_rate_of_climb=3500 * ureg.feet / ureg.minute,
            cruise_speed=460 * ureg.knot,
            range=3400 * ureg.nautical_mile,
            endurance=6 * ureg.hour,
            operator="NASA AFRC",
            max_bank_angle=30.0,
            useful_payload=2500 * ureg.pound,
            vx=150 * ureg.knot,
            vy=170 * ureg.knot,
            roc_at_service_ceiling=500.0 * ureg.feet / ureg.minute,
            descent_rate=1500 * ureg.feet / ureg.minute,
            descent_speed_reduction=49 * ureg.knot,
            speed_profile=[
                (0 * ureg.feet, 292 * ureg.knot),
                (45000 * ureg.feet, 460 * ureg.knot),
            ],
        )

class NASA_P3(Aircraft):
    """
    NASA P-3 Orion (NASA 426) airborne science laboratory.

    A four-engine turboprop aircraft capable of long-duration flights
    (8-14 hours) and large payloads up to 14,700 lbs. Operated by
    NASA Wallops Flight Facility (WFF). Supports ecology, geography,
    hydrology, meteorology, oceanography, atmospheric chemistry,
    cryospheric research, and satellite calibration/validation.

    Speed profile from Moving Lines: TAS = 110 + alt_m * 0.007 (m/s),
    capped at 155 m/s (~301 kt) above ~21,000 ft.

    See also:
        https://airbornescience.nasa.gov/aircraft/P-3_Orion
    """
    def __init__(self):
        super().__init__(
            aircraft_type="P-3 Orion",
            tail_number="NASA 426",
            service_ceiling=32000 * ureg.feet,
            approach_speed=130 * ureg.knot,
            best_rate_of_climb=3500 * ureg.feet / ureg.minute,
            cruise_speed=301 * ureg.knot,
            range=3800 * ureg.nautical_mile,
            endurance=12 * ureg.hour,
            operator="NASA LaRC",
            max_bank_angle=30.0,
            useful_payload=18000 * ureg.pound,
            vx=135 * ureg.knot,
            vy=155 * ureg.knot,
            roc_at_service_ceiling=100.0 * ureg.feet / ureg.minute,
            descent_rate=1500 * ureg.feet / ureg.minute,
            descent_speed_reduction=29 * ureg.knot,
            speed_profile=[
                (0 * ureg.feet, 214 * ureg.knot),
                (21091 * ureg.feet, 301 * ureg.knot),
                (32000 * ureg.feet, 301 * ureg.knot),
            ],
        )

class NASA_WB57(Aircraft):
    """
    NASA WB-57 (NASA 927) high-altitude research aircraft.

    Based at NASA Johnson Space Center (JSC), Ellington Field. Three
    WB-57 aircraft have been flying research missions since the early
    1970s. Operates up to 60,000 ft with 8,800 lbs useful payload.

    Speed profile assumed similar to ER-2 (linear increase to ceiling).

    See also:
        https://airbornescience.nasa.gov/aircraft/WB-57_-_JSC
    """
    def __init__(self):
        super().__init__(
            aircraft_type="WB-57",
            tail_number="NASA 927",
            service_ceiling=60000 * ureg.feet,
            approach_speed=130 * ureg.knot,
            best_rate_of_climb=5000 * ureg.feet / ureg.minute,
            cruise_speed=410 * ureg.knot,
            range=2500 * ureg.nautical_mile,
            endurance=6.5 * ureg.hour,
            operator="NASA JSC",
            max_bank_angle=30.0,
            useful_payload=8800 * ureg.pound,
            vx=140 * ureg.knot,
            vy=160 * ureg.knot,
            roc_at_service_ceiling=500 * ureg.feet / ureg.minute,
            descent_rate=1500 * ureg.feet / ureg.minute,
            descent_speed_reduction=0 * ureg.knot,
            speed_profile=[
                (0 * ureg.feet, 136 * ureg.knot),
                (60000 * ureg.feet, 410 * ureg.knot),
            ],
        )

class NASA_B777(Aircraft):
    """
    NASA Boeing 777 long-range research aircraft.

    Operated by NASA Langley Research Center (LaRC). Very large payload
    capacity (75,000 lbs) and long endurance (18 hours), suitable for
    global-scale remote sensing missions.
    """
    def __init__(self):
        super().__init__(
            aircraft_type="B777",
            tail_number="Unknown",
            service_ceiling=43000 * ureg.feet,
            approach_speed=150 * ureg.knot,
            best_rate_of_climb=2500 * ureg.feet / ureg.minute,
            cruise_speed=487 * ureg.knot,
            range=9000 * ureg.nautical_mile,
            endurance=18 * ureg.hour,
            operator="NASA LaRC",
            max_bank_angle=30.0,
            useful_payload=75000 * ureg.pound,
            vx=160 * ureg.knot,
            vy=180 * ureg.knot,
            roc_at_service_ceiling=500 * ureg.feet / ureg.minute,
            descent_rate=1500 * ureg.feet / ureg.minute,
            descent_speed_reduction=30 * ureg.knot,
            speed_profile=[
                (0 * ureg.feet, 350 * ureg.knot),
                (43000 * ureg.feet, 487 * ureg.knot),
            ],
        )

class DynamicAviation_DH8(Aircraft):
    """
    Dynamic Aviation DHC-8 Dash 8 twin-turboprop aircraft.

    Operated by Dynamic Aviation Group Inc. under contract to NASA's
    Airborne Science Program. Medium-lift platform suitable for both
    high and low altitude missions with 15,000 lbs useful payload.

    See also:
        https://www.dynamicaviation.com/fleet-dash-8
    """
    def __init__(self):
        super().__init__(
            aircraft_type="Dash 8",
            tail_number="Unknown",
            service_ceiling=25000 * ureg.feet,
            approach_speed=110 * ureg.knot,
            best_rate_of_climb=2000 * ureg.feet / ureg.minute,
            cruise_speed=243 * ureg.knot,
            range=950 * ureg.nautical_mile,
            endurance=5 * ureg.hour,
            operator="Dynamic Aviation",
            max_bank_angle=30.0,
            useful_payload=15000 * ureg.pound,
            vx=110 * ureg.knot,
            vy=130 * ureg.knot,
            roc_at_service_ceiling=100.0 * ureg.feet / ureg.minute,
            descent_rate=1500 * ureg.feet / ureg.minute,
            descent_speed_reduction=15 * ureg.knot,
            speed_profile=[
                (0 * ureg.feet, 170 * ureg.knot),
                (25000 * ureg.feet, 243 * ureg.knot),
            ],
        )

class DynamicAviation_A90(Aircraft):
    """
    Dynamic Aviation Beechcraft King Air A90 twin-turboprop aircraft.

    Operated by Dynamic Aviation Group Inc. under contract to NASA's
    Airborne Science Program. Two-engine turboprop used for sensor
    integration, flight testing, and airborne science support.

    See also:
        https://airbornescience.nasa.gov/aircraft/Beechcraft_King_Air_A90
    """
    def __init__(self):
        super().__init__(
            aircraft_type="King Air 90",
            tail_number="Unknown",
            service_ceiling=30000 * ureg.feet,
            approach_speed=110 * ureg.knot,
            best_rate_of_climb=1800 * ureg.feet / ureg.minute,
            cruise_speed=230 * ureg.knot,
            range=1500 * ureg.nautical_mile,
            endurance=6 * ureg.hour,
            operator="Dynamic Aviation",
            max_bank_angle=30.0,
            useful_payload=2950 * ureg.pound,
            vx=120 * ureg.knot,
            vy=140 * ureg.knot,
            roc_at_service_ceiling=100.0 * ureg.feet / ureg.minute,
            descent_rate=1500 * ureg.feet / ureg.minute,
            descent_speed_reduction=10 * ureg.knot,
            speed_profile=[
                (0 * ureg.feet, 170 * ureg.knot),
                (30000 * ureg.feet, 230 * ureg.knot),
            ],
        )

class DynamicAviation_B200(Aircraft):
    """
    Dynamic Aviation Beechcraft King Air B200 twin-turboprop aircraft.

    Operated by Dynamic Aviation Group Inc. under contract to NASA's
    Airborne Science Program. An all-metal twin-turboprop capable of
    operating from a wide variety of civilian and military airports.
    Well suited for aerial remote sensing, chase aircraft support,
    and technology demonstration missions.

    See also:
        https://airbornescience.nasa.gov/aircraft/Beechcraft_King_Air_A200
    """
    def __init__(self):
        super().__init__(
            aircraft_type="King Air 200",
            tail_number="Unknown",
            service_ceiling=35000 * ureg.feet,
            approach_speed=120 * ureg.knot,
            best_rate_of_climb=2000 * ureg.feet / ureg.minute,
            cruise_speed=250 * ureg.knot,
            range=1632 * ureg.nautical_mile,
            endurance=6 * ureg.hour,  # Estimated
            operator="Dynamic Aviation",
            max_bank_angle=30.0,
            useful_payload=4250 * ureg.pound,
            vx=120 * ureg.knot,
            vy=140 * ureg.knot,
            roc_at_service_ceiling=100.0 * ureg.feet / ureg.minute,
            descent_rate=1500 * ureg.feet / ureg.minute,
            descent_speed_reduction=10 * ureg.knot,
            speed_profile=[
                (0 * ureg.feet, 185 * ureg.knot),
                (35000 * ureg.feet, 250 * ureg.knot),
            ],
        )

class C130(Aircraft):
    """
    C-130H Hercules four-engine turboprop transport / research aircraft.

    Used by multiple agencies (NSF/NCAR, NASA WFF, NOAA) for airborne
    science. In a typical research configuration carries 13,000 lbs of
    payload with 8-9 hour endurance. The NASA C-130H (N436NA) at Wallops
    Flight Facility supports airborne scientific research and cargo.

    Speed profile from Moving Lines: TAS = 130 + alt_m * 0.0075 (m/s),
    capped at 175 m/s (~340 kt) above ~19,685 ft.

    See also:
        https://airbornescience.nasa.gov/aircraft/C-130H_-_WFF
    """
    def __init__(self):
        super().__init__(
            aircraft_type="C-130H Hercules",
            tail_number="Unknown",
            service_ceiling=25000 * ureg.feet,
            approach_speed=115 * ureg.knot,
            best_rate_of_climb=2000 * ureg.feet / ureg.minute,
            cruise_speed=340 * ureg.knot,
            range=2500 * ureg.nautical_mile,
            endurance=10 * ureg.hour,
            operator="Various",
            max_bank_angle=20.0,
            useful_payload=45000 * ureg.pound,
            vx=120 * ureg.knot,
            vy=140 * ureg.knot,
            roc_at_service_ceiling=100 * ureg.feet / ureg.minute,
            descent_rate=2000 * ureg.feet / ureg.minute,
            descent_speed_reduction=29 * ureg.knot,
            speed_profile=[
                (0 * ureg.feet, 253 * ureg.knot),
                (19685 * ureg.feet, 340 * ureg.knot),
                (25000 * ureg.feet, 340 * ureg.knot),
            ],
        )

class BAe146(Aircraft):
    """
    BAe-146-301 atmospheric research aircraft (G-LUXE).

    Operated by the UK Facility for Airborne Atmospheric Measurements
    (FAAM). Can fly with up to 4 tonnes of scientific instruments,
    from 50 ft over the sea to 35,000 ft, with flights lasting 1-6
    hours covering up to 2,000 nautical miles. Has completed 1,400+
    science flights across 30 countries.

    Speed profile from Moving Lines: TAS = 130 + alt_m * 0.002 (m/s),
    capped at 150 m/s (~292 kt) above ~32,808 ft (above ceiling, so
    effectively linear to ceiling).

    See also:
        https://faam.ac.uk/
    """
    def __init__(self):
        super().__init__(
            aircraft_type="BAe-146",
            tail_number="Unknown",
            service_ceiling=28000 * ureg.feet,
            approach_speed=120 * ureg.knot,
            best_rate_of_climb=1000 * ureg.feet / ureg.minute,
            cruise_speed=286 * ureg.knot,
            range=1800 * ureg.nautical_mile,
            endurance=6 * ureg.hour,
            operator="FAAM",
            max_bank_angle=20.0,
            useful_payload=10000 * ureg.pound,
            vx=130 * ureg.knot,
            vy=145 * ureg.knot,
            roc_at_service_ceiling=100 * ureg.feet / ureg.minute,
            descent_rate=1000 * ureg.feet / ureg.minute,
            descent_speed_reduction=29 * ureg.knot,
            speed_profile=[
                (0 * ureg.feet, 253 * ureg.knot),
                (28000 * ureg.feet, 286 * ureg.knot),
            ],
        )

class Learjet(Aircraft):
    """
    Learjet high-altitude research aircraft.

    Used for atmospheric and remote sensing research by various
    operators. NASA has operated multiple Learjet variants (23, 24D,
    25, 35) for high-altitude atmospheric science.

    Speed profile from Moving Lines (https://github.com/samuelleblanc/fp).

    See also:
        https://airbornescience.nasa.gov/aircraft/Learjet_25
    """
    def __init__(self):
        super().__init__(
            aircraft_type="Learjet",
            tail_number="Unknown",
            service_ceiling=35000 * ureg.feet,
            approach_speed=130 * ureg.knot,
            best_rate_of_climb=4000 * ureg.feet / ureg.minute,
            cruise_speed=430 * ureg.knot,
            range=1500 * ureg.nautical_mile,
            endurance=4 * ureg.hour,
            operator="Various",
            max_bank_angle=30.0,
            useful_payload=3000 * ureg.pound,
            vx=140 * ureg.knot,
            vy=160 * ureg.knot,
            roc_at_service_ceiling=500 * ureg.feet / ureg.minute,
            descent_rate=1500 * ureg.feet / ureg.minute,
            descent_speed_reduction=39 * ureg.knot,
            speed_profile=[
                (0 * ureg.feet, 194 * ureg.knot),
                (35000 * ureg.feet, 430 * ureg.knot),
            ],
        )

class TwinOtter(Aircraft):
    """
    DHC-6 Twin Otter STOL twin-turboprop utility aircraft.

    Common low-altitude research platform. The NPS CIRPAS Twin Otter
    (based at Naval Postgraduate School, Monterey, CA) has supported
    atmospheric and oceanographic research since 1998 for ONR, NSF,
    DOE, NOAA, NASA, and others. Twin Otter International Ltd. (TOIL)
    also operates a fleet for medium-lift, slow-flight research.

    Speed profile from Moving Lines (https://github.com/samuelleblanc/fp).

    See also:
        https://airbornescience.nasa.gov/aircraft/Twin_Otter_-_CIRPAS_-_NPS
    """
    def __init__(self):
        super().__init__(
            aircraft_type="DHC-6 Twin Otter",
            tail_number="Unknown",
            service_ceiling=10000 * ureg.feet,
            approach_speed=70 * ureg.knot,
            best_rate_of_climb=430 * ureg.feet / ureg.minute,
            cruise_speed=150 * ureg.knot,
            range=800 * ureg.nautical_mile,
            endurance=6 * ureg.hour,
            operator="Various",
            max_bank_angle=15.0,
            useful_payload=4000 * ureg.pound,
            vx=75 * ureg.knot,
            vy=85 * ureg.knot,
            roc_at_service_ceiling=50 * ureg.feet / ureg.minute,
            descent_rate=430 * ureg.feet / ureg.minute,
            descent_speed_reduction=8 * ureg.knot,
            speed_profile=[
                (0 * ureg.feet, 97 * ureg.knot),
                (10000 * ureg.feet, 150 * ureg.knot),
            ],
        )
