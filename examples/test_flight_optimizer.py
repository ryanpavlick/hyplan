#%%
from hyplan.aircraft import DynamicAviation_B200
from hyplan.airports import Airport
from hyplan.flight_line import FlightLine
from hyplan.flight_plan import compute_flight_plan, plot_flight_plan, plot_altitude_trajectory
from hyplan.flight_optimizer import build_graph, greedy_optimize
from hyplan.units import ureg
import logging

logging.basicConfig(level=logging.INFO)

# Define aircraft
aircraft = DynamicAviation_B200()

# Define airports
departure_airport = Airport("KSBA")
return_airport = Airport("KBUR")

# Define several flight lines in the LA basin area
flight_line_1 = FlightLine.start_length_azimuth(
    lat1=34.50, lon1=-118.20,
    length=ureg.Quantity(50000, "meter"),
    az=90.0,
    altitude_msl=ureg.Quantity(15000, "feet"),
    site_name="Line_A"
)

flight_line_2 = FlightLine.start_length_azimuth(
    lat1=34.65, lon1=-118.10,
    length=ureg.Quantity(70000, "meter"),
    az=125.0,
    altitude_msl=ureg.Quantity(18000, "feet"),
    site_name="Line_B"
)

flight_line_3 = FlightLine.start_length_azimuth(
    lat1=34.30, lon1=-117.80,
    length=ureg.Quantity(40000, "meter"),
    az=45.0,
    altitude_msl=ureg.Quantity(15000, "feet"),
    site_name="Line_C"
)

flight_lines = [flight_line_1, flight_line_2, flight_line_3]
airports = [departure_airport, return_airport]

#%% Build graph and inspect
print("Building graph...")
G = build_graph(aircraft, flight_lines, airports)
print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
print(f"Nodes: {list(G.nodes)}")
print()

# Show edge types
from collections import Counter
edge_types = Counter(d["edgetype"] for _, _, d in G.edges(data=True))
print(f"Edge types: {dict(edge_types)}")

#%% Run greedy optimization
print("\nRunning greedy optimization...")
result = greedy_optimize(
    aircraft=aircraft,
    flight_lines=flight_lines,
    airports=airports,
    takeoff_airport=departure_airport,
    return_airport=return_airport,
    max_endurance=4.0,
    max_daily_flight_time=8.0,
    max_days=3,
)

print(f"\nRoute: {result['route']}")
print(f"Total time: {result['total_time']:.2f} hours")
print(f"Days used: {result['days_used']}")
for i, dt in enumerate(result['daily_times'], 1):
    print(f"  Day {i}: {dt:.2f} hours")
print(f"Lines covered: {result['lines_covered']}/{len(flight_lines)}")
if result['lines_skipped']:
    print(f"Lines skipped: {result['lines_skipped']}")
print(f"Refuel stops: {result['refuel_stops']}")

#%% Feed optimized sequence into compute_flight_plan
print("\nComputing flight plan from optimized sequence...")
flight_plan_df = compute_flight_plan(
    aircraft,
    result["flight_sequence"],
    departure_airport,
    return_airport,
)
print(flight_plan_df[["segment_type", "segment_name", "distance", "time_to_segment"]])

#%% Plot
plot_flight_plan(flight_plan_df, departure_airport, return_airport, result["flight_sequence"])
plot_altitude_trajectory(flight_plan_df, aircraft=aircraft)

# %%
