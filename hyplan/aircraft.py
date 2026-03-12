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
        descent_rate: Quantity  # feet per minute
    ):
        """
        Initializes an Aircraft object with performance parameters.

        Args:
            aircraft_type (str): Aircraft model or name.
            tail_number (str): Aircraft tail number or unique identifier.
            service_ceiling (Quantity): Maximum operational altitude (feet).
            approach_speed (Quantity): Landing approach speed (knots).
            best_rate_of_climb (Quantity): Maximum rate of climb (feet per minute).
            cruise_speed (Quantity): Typical cruise speed (knots).
            range (Quantity): Maximum flight range (nautical miles).
            endurance (Quantity): Maximum flight duration (hours).
            operator (str): Organization that operates the aircraft.
            max_bank_angle (float): Maximum allowable bank angle (degrees).
            useful_payload (Quantity): Maximum payload capacity (pounds).
            vx (Quantity): Best angle-of-climb speed (knots).
            vy (Quantity): Best rate-of-climb speed (knots).
            roc_at_service_ceiling (Quantity): Rate of climb at max altitude (feet per minute).
            descent_rate (Quantity): Standard rate of descent (feet per minute).

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
        Compute the rate of climb at a given altitude assuming a linear decrease
        from best rate of climb at sea level to roc_at_service_ceiling at service ceiling.
        """
        if altitude >= self.service_ceiling:
            return self.roc_at_service_ceiling
        
        altitude_ratio = altitude / self.service_ceiling
        roc = (1 - altitude_ratio) * (self.best_rate_of_climb - self.roc_at_service_ceiling) + self.roc_at_service_ceiling
        return roc

    def _climb(self, start_altitude, end_altitude, true_air_speed=None):
        """
        Estimate the time and horizontal distance traveled during a Vy climb.
        """
        try:
            start_altitude = start_altitude.to(ureg.feet)
            end_altitude = end_altitude.to(ureg.feet)
            true_air_speed = (true_air_speed or self.cruise_speed).to(ureg.feet / ureg.minute)

            if end_altitude > self.service_ceiling:
                raise ValueError("End altitude cannot exceed the service ceiling.")
            if end_altitude <= start_altitude:
                return 0 * ureg.minute, 0 * ureg.nautical_mile

            altitude_difference = end_altitude - start_altitude
            avg_rate_of_climb = (self.rate_of_climb(start_altitude) + self.rate_of_climb(end_altitude)) / 2
            time_to_climb = (altitude_difference / avg_rate_of_climb).to(ureg.minute)

            climb_angle = np.arctan(avg_rate_of_climb / true_air_speed).to(ureg.radian)
            horizontal_speed = (true_air_speed * np.cos(climb_angle)).to(ureg.nautical_mile / ureg.hour)
            horizontal_distance = (horizontal_speed * time_to_climb).to(ureg.nautical_mile)

            return time_to_climb, horizontal_distance

        except Exception as e:
            logging.error(f"Error in time_to_climb: {e}")
            raise

    def time_to_takeoff(self, airport: Airport, waypoint: Waypoint):
        """
        Calculate the total time needed to take off from an airport and reach a waypoint at cruise altitude,
        with detailed altitude and time information for each phase.
        """
        try:
            airport_altitude = airport.elevation.to(ureg.feet)
            waypoint_altitude = waypoint.altitude.to(ureg.feet)

            # Climb phase
            climb_time, climb_distance = self._climb(airport_altitude, waypoint_altitude, true_air_speed=self.vy)

            _, departure_heading = pymap3d.vincenty.vdist(airport.latitude, airport.longitude, waypoint.latitude, waypoint.longitude)
            airport_waypoint = Waypoint(latitude=airport.latitude, longitude=airport.longitude, heading=departure_heading, altitude=airport_altitude)
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
            cruise_time = (cruise_distance / self.cruise_speed).to(ureg.minute)

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
            airport_waypoint = Waypoint(latitude=airport.latitude, longitude=airport.longitude, heading=(arrival_heading + 180.0) % 360.0, altitude=airport.elevation)
            dubins_path = DubinsPath(
                start=waypoint,
                end=airport_waypoint,
                speed=self.cruise_speed,
                bank_angle=self.max_bank_angle,
                step_size=100
            )
            total_distance = dubins_path.length.to(ureg.nautical_mile)

            # Cruise phase
            cruise_altitude = waypoint.altitude.to(ureg.feet)
            approach_altitude = min(airport.elevation.to(ureg.feet) + 5_000 * ureg.feet, cruise_altitude)
            descent_altitude = cruise_altitude - approach_altitude
            descent_distance = 3 * (descent_altitude.to(ureg.feet).magnitude / 1_000) * ureg.nautical_mile
            cruise_distance = max(0 * ureg.nautical_mile, total_distance - descent_distance)
            cruise_time = (cruise_distance / self.cruise_speed).to(ureg.minute)

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
        """
        try:
            start_altitude = start_altitude.to(ureg.feet)
            end_altitude = end_altitude.to(ureg.feet)
            true_air_speed = (true_air_speed or self.cruise_speed).to(ureg.feet / ureg.minute)

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
            # Default true airspeed is cruise speed
            true_air_speed = true_air_speed or self.cruise_speed

            start_altitude = start_waypoint.altitude.to(ureg.feet)
            end_altitude = end_waypoint.altitude.to(ureg.feet)

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
# 🚀 NASA Aircraft
class NASA_ER2(Aircraft):
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
            descent_rate=1500 * ureg.feet / ureg.minute
        )

class NASA_GIII(Aircraft):
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
            descent_rate=1500 * ureg.feet / ureg.minute
        )

class NASA_GIV(Aircraft):
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
            descent_rate=1500 * ureg.feet / ureg.minute
        )

class NASA_C20A(Aircraft):
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
            descent_rate=1500 * ureg.feet / ureg.minute
        )

class NASA_P3(Aircraft):
    def __init__(self):
        super().__init__(
            aircraft_type="P-3 Orion",
            tail_number="NASA 426",
            service_ceiling=32000 * ureg.feet,
            approach_speed=130 * ureg.knot,
            best_rate_of_climb=3500 * ureg.feet / ureg.minute,
            cruise_speed=405 * ureg.knot,
            range=3800 * ureg.nautical_mile,
            endurance=12 * ureg.hour,
            operator="NASA LaRC",
            max_bank_angle=30.0,
            useful_payload=18000 * ureg.pound,
            vx=135 * ureg.knot,
            vy=155 * ureg.knot,
            roc_at_service_ceiling=100.0 * ureg.feet / ureg.minute,
            descent_rate=1500 * ureg.feet / ureg.minute
        )

class NASA_WB57(Aircraft):
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
            descent_rate=1500 * ureg.feet / ureg.minute
        )

class NASA_B777(Aircraft):
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
            descent_rate=1500 * ureg.feet / ureg.minute
        )

# 🚀 Dynamic Aviation Aircraft
class DynamicAviation_DH8(Aircraft):
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
            descent_rate=1500 * ureg.feet / ureg.minute
        )

class DynamicAviation_A90(Aircraft):
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
            descent_rate=1500 * ureg.feet / ureg.minute
        )

class DynamicAviation_B200(Aircraft):
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
            descent_rate=1500 * ureg.feet / ureg.minute
        )
