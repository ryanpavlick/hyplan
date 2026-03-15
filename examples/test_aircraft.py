#%%
import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from hyplan.units import ureg
from hyplan.airports import Airport, initialize_data
from hyplan.dubins_path import Waypoint, DubinsPath
from hyplan.aircraft import (
    Aircraft,
    DynamicAviation_B200,
    NASA_ER2,
    NASA_GIII,
    NASA_GIV,
    NASA_C20A,
    NASA_P3,
    NASA_WB57,
    NASA_B777,
    DynamicAviation_DH8,
    DynamicAviation_A90,
    C130,
    BAe146,
    Learjet,
    TwinOtter,
)
from hyplan.flight_line import FlightLine

# Initialize airport data
initialize_data()

# Define example airport and flight line
airport = Airport(icao="KSBA")  # Example airport: Santa Barbara Municipal
flight_line = FlightLine.start_length_azimuth(
    lat1=34.05,
    lon1=-118.25,
    length=ureg.Quantity(100000, "meter"),
    az=45.0,
    altitude=ureg.Quantity(20000, "feet"),
    site_name="LA Northeast",
    investigator="Dr. Smith"
)

waypoint_1 = flight_line.waypoint1
waypoint_2 = flight_line.waypoint2

# Define an example aircraft
aircraft = NASA_P3()

# Compute flight phases
takeoff_info = aircraft.time_to_takeoff(airport, waypoint_1)
cruise_info = aircraft.time_to_cruise(waypoint_1, waypoint_2)
return_info = aircraft.time_to_return(waypoint_2, airport)

# Helper function to extract coordinates from DubinsPath
def extract_coordinates(dubins_path):
    coords = [(lat, lon) for lon, lat in dubins_path.geometry.coords]
    return zip(*coords) if coords else ([], [])

# Plot Ground Track
def plot_ground_track(phases, title="Ground Track"):
    plt.figure(figsize=(12, 7))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, edgecolor='black', facecolor='lightgray')
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.5)
    ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)

    # Plot phases
    colors = {"Takeoff": "blue", "Cruise": "green", "Return": "red"}
    for name, phase_info in phases.items():
        lats, lons = extract_coordinates(phase_info["dubins_path"])
        ax.plot(lons, lats, label=name, color=colors[name], linewidth=2, transform=ccrs.PlateCarree())

    # Plot airport and waypoints
    ax.plot(airport.longitude, airport.latitude, marker="*", color="gold", markersize=15, label="Airport", transform=ccrs.PlateCarree())
    ax.plot(waypoint_1.longitude, waypoint_1.latitude, marker="^", color="purple", markersize=10, label="Waypoint 1", transform=ccrs.PlateCarree())
    ax.plot(waypoint_2.longitude, waypoint_2.latitude, marker="^", color="orange", markersize=10, label="Waypoint 2", transform=ccrs.PlateCarree())

    plt.title(title)
    plt.legend()
    plt.show()

# Plot Altitude Trajectory with curved climb profile
def plot_altitude_trajectory(phases, aircraft_obj, title="Altitude Trajectory"):
    plt.figure(figsize=(10, 5))
    current_time = 0.0  # minutes

    for name, phase_info in phases.items():
        for sub_phase_name, phase_data in phase_info["phases"].items():
            start_alt = phase_data["start_altitude"].to("feet").magnitude
            end_alt = phase_data["end_altitude"].to("feet").magnitude
            duration = (phase_data["end_time"] - phase_data["start_time"]).to("minute").magnitude

            if end_alt > start_alt and aircraft_obj is not None:
                # Use curved climb profile
                profile_t, profile_h = aircraft_obj.climb_altitude_profile(
                    phase_data["start_altitude"],
                    phase_data["end_altitude"]
                )
                if profile_t[-1] > 0:
                    profile_t = profile_t * (duration / profile_t[-1])
                plt.plot(
                    current_time + profile_t, profile_h,
                    marker="o", markevery=[0, -1],
                    label=f"{name} - {sub_phase_name}"
                )
            else:
                plt.plot(
                    [current_time, current_time + duration],
                    [start_alt, end_alt],
                    marker="o",
                    label=f"{name} - {sub_phase_name}"
                )

            current_time += duration

    plt.xlabel("Time (minutes)")
    plt.ylabel("Altitude (feet)")
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.show()


# Incremental plotting
# 1. Takeoff only
plot_ground_track({"Takeoff": takeoff_info}, "Ground Track: Takeoff Only")
plot_altitude_trajectory({"Takeoff": takeoff_info}, aircraft, "Altitude Trajectory: Takeoff Only")

# 2. Takeoff and Cruise
plot_ground_track({"Takeoff": takeoff_info, "Cruise": cruise_info}, "Ground Track: Takeoff and Cruise")

# 3. Full Flight (Takeoff, Cruise, and Return)
plot_ground_track({"Takeoff": takeoff_info, "Cruise": cruise_info, "Return": return_info}, "Ground Track: Full Flight")
plot_altitude_trajectory({"Takeoff": takeoff_info, "Cruise": cruise_info, "Return": return_info}, aircraft, "Altitude Trajectory: Full Flight")


#%% Test altitude-dependent speed model
print("\n=== Altitude-Dependent Speed Model ===")

all_aircraft = [
    NASA_ER2(), NASA_GIII(), NASA_GIV(), NASA_C20A(), NASA_P3(),
    NASA_WB57(), NASA_B777(), DynamicAviation_DH8(), DynamicAviation_A90(),
    DynamicAviation_B200(), NASA_DC8(), C130(), BAe146(), Learjet(), TwinOtter(),
]

for ac in all_aircraft:
    low_alt_speed = ac.cruise_speed_at(0 * ureg.feet).to(ureg.knot)
    mid_alt_speed = ac.cruise_speed_at(ac.service_ceiling / 2).to(ureg.knot)
    cruise_speed = ac.cruise_speed_at(ac.service_ceiling).to(ureg.knot)
    desc_speed = ac.descent_speed_at(ac.service_ceiling / 2).to(ureg.knot)
    print(
        f"{ac.aircraft_type:30s}  "
        f"SL: {low_alt_speed:6.0f}  "
        f"Mid: {mid_alt_speed:6.0f}  "
        f"Ceil: {cruise_speed:6.0f}  "
        f"Desc@mid: {desc_speed:6.0f}"
    )


#%% Plot TAS vs altitude for all aircraft
fig, ax = plt.subplots(figsize=(10, 7))

for ac in all_aircraft:
    alts = np.linspace(0, ac.service_ceiling.to(ureg.feet).magnitude, 100) * ureg.feet
    speeds = [ac.cruise_speed_at(a).to(ureg.knot).magnitude for a in alts]
    ax.plot(speeds, [a.magnitude for a in alts], label=ac.aircraft_type)

ax.set_xlabel("True Airspeed (knots)")
ax.set_ylabel("Altitude (feet)")
ax.set_title("TAS vs. Altitude — All Aircraft")
ax.legend(fontsize=7, loc="upper left")
ax.grid(True)
plt.tight_layout()
plt.show()


#%% Compare climb profiles across aircraft
fig, ax = plt.subplots(figsize=(10, 6))

for ac in all_aircraft:
    target_alt = min(25000, ac.service_ceiling.to(ureg.feet).magnitude * 0.9)
    times, alts = ac.climb_altitude_profile(0 * ureg.feet, target_alt * ureg.feet)
    ax.plot(times, alts, label=f"{ac.aircraft_type} -> {target_alt:.0f} ft")

ax.set_xlabel("Time (minutes)")
ax.set_ylabel("Altitude (feet)")
ax.set_title("Climb Profiles — Analytical (Exponential) Model")
ax.legend(fontsize=7, loc="lower right")
ax.grid(True)
plt.tight_layout()
plt.show()


#%% Compare analytical vs old linear climb for B200
b200 = DynamicAviation_B200()
target = 25000 * ureg.feet

# Analytical profile
times_analytical, alts_analytical = b200.climb_altitude_profile(0 * ureg.feet, target)

# Linear profile (old model: constant average ROC)
roc_start = b200.rate_of_climb(0 * ureg.feet).to(ureg.feet / ureg.minute).magnitude
roc_end = b200.rate_of_climb(target).to(ureg.feet / ureg.minute).magnitude
avg_roc = (roc_start + roc_end) / 2
linear_total_time = target.magnitude / avg_roc
times_linear = np.linspace(0, linear_total_time, 50)
alts_linear = np.linspace(0, target.magnitude, 50)

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(times_analytical, alts_analytical, 'b-', linewidth=2, label="Analytical (exponential)")
ax.plot(times_linear, alts_linear, 'r--', linewidth=2, label="Old (linear average)")
ax.set_xlabel("Time (minutes)")
ax.set_ylabel("Altitude (feet)")
ax.set_title(f"B200 Climb to {target.magnitude:.0f} ft — Analytical vs Linear")
ax.legend()
ax.grid(True)
plt.tight_layout()
plt.show()

print(f"Analytical climb time: {times_analytical[-1]:.1f} min")
print(f"Old linear climb time: {linear_total_time:.1f} min")

# %%
