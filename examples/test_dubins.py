#%%
import matplotlib.pyplot as plt
from hyplan.dubins_path import Waypoint, DubinsPath
from hyplan.units import ureg
import numpy as np
from hyplan.geometry import wrap_to_360

# Define waypoints
waypoints = [
    Waypoint(latitude=34.20, longitude=-117.95, heading=wrap_to_360(270.0), name = "Waypoint 1"),
    Waypoint(latitude=34.20, longitude=-118.15, heading=wrap_to_360(270.0), name = "Waypoint 2"),
    Waypoint(latitude=34.20, longitude=-118.35, heading=wrap_to_360(270.0), name = "Waypoint 3"),
    Waypoint(latitude=34.05, longitude=-118.25, heading=wrap_to_360(180.0), altitude_msl=100 * ureg.meter, name = "Waypoint 4"),
    Waypoint(latitude=34.10, longitude=-118.15, heading=wrap_to_360(180.0), name = "Waypoint 5"),
    Waypoint(latitude=34.15, longitude=-118.05, heading=wrap_to_360(0.0), name = "Waypoint 6"),
    Waypoint(latitude=34.20, longitude=-117.95, heading=wrap_to_360(270.0), name = "Waypoint 1")
]

# Define parameters
speed = 100.0  # Speed in m/s
bank_angle = 30.0  # Updated bank angle in degrees
step_size = 100.0  # Step size in meters

# Generate Dubins paths between consecutive waypoints
paths = []
for i in range(len(waypoints) - 1):
    dubins_path = DubinsPath(
        start=waypoints[i],
        end=waypoints[i + 1],
        speed=speed,
        bank_angle=bank_angle,
        step_size=step_size
    )
    print(f"Path {i + 1}: Start Heading={waypoints[i].heading}, End Heading={waypoints[i + 1].heading}")
    print(f"Path {i + 1} Turn Radius: {(speed**2) / (9.8 * np.tan(np.radians(bank_angle))):.2f} meters")
    print(f"Path {i + 1} Coordinates (first 5 points): {list(dubins_path.geometry.coords)[:5]}...")
    paths.append(dubins_path)

# Plot the Dubins paths
plt.figure(figsize=(10, 6))
for i, path in enumerate(paths):
    lons, lats = zip(*list(path.geometry.coords))
    plt.plot(lons, lats, label=f"Path {i + 1}")

# Plot waypoints
for waypoint in waypoints:
    plt.scatter(waypoint.geometry.x, waypoint.geometry.y, color='red', zorder=5)
    plt.text(waypoint.geometry.x, waypoint.geometry.y, waypoint.name, fontsize=9)

# Add plot details
plt.title("Dubins Paths Between Waypoints")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.legend()
plt.grid()
plt.show()


# %%%
