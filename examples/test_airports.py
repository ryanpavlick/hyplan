#%%
import logging
import matplotlib.pyplot as plt
from hyplan.airports import (
    Airport, find_nearest_airport, airports_within_radius,
    initialize_data, get_airports, get_airport_details, generate_geojson, get_runway_details
)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

#%% Initialize data with specific filters
countries = ["US", "CA"]  # United States and Canada
min_runway_length = 8000  # Minimum runway length in feet
runway_surface = ["ASPH", "CONC", "ASP", "CON", "MAC", "BIT", "Asphalt", "Concrete"]
initialize_data(countries=countries, min_runway_length=min_runway_length, runway_surface=runway_surface)
logging.info("Airports data initialized.")

#%% Example 1: Find the nearest airport to a point
latitude, longitude = 34.05, -118.25  # Los Angeles coordinates
nearest_icao = find_nearest_airport(latitude, longitude)
logging.info(f"Nearest Airport ICAO Code: {nearest_icao}")

#%% Example 2: Find all airports within a 50-mile radius
radius = 50  # miles
nearby_airports = airports_within_radius(latitude, longitude, radius, unit="miles")
logging.info(f"Airports within {radius} miles: {len(nearby_airports)}")
for icao in nearby_airports:
    logging.info(f"  - {icao}")

#%% Example 3: Get details for specific airports
icao_list = [nearest_icao] + nearby_airports[:3]  # Example ICAO codes to fetch details
airport_details = get_airport_details(icao_list)
logging.info(f"Details for {len(icao_list)} airports:\n{airport_details}")

#%% Example 4: Create a GeoJSON of all airports
generate_geojson("all_airports.geojson")
logging.info("GeoJSON for all airports generated.")

#%% Example 5: Create a GeoJSON of a subset of airports
generate_geojson("subset_airports.geojson", icao_codes=icao_list)
logging.info("GeoJSON for a subset of airports generated.")

#%% Example: Using the Airport class
icao_code = "KLAX"  # Los Angeles International Airport
try:
    lax = Airport(icao_code)
    logging.info(f"Airport Details - Name: {lax.name}, ICAO: {lax.icao_code}, IATA: {lax.iata_code}, Country: {lax.country}, Location: ({lax.longitude}, {lax.latitude})")
except ValueError as e:
    logging.error(str(e))

#%% Example 7: Create a map of airports using matplotlib
plt.figure(figsize=(12, 8))
gdf_airports = get_airports()
plt.scatter(gdf_airports['longitude'], gdf_airports['latitude'], c='blue', label='Airports', alpha=0.7, edgecolors='k')
plt.scatter(longitude, latitude, c='red', label='Reference Point', s=100, edgecolors='k')
plt.title("Airports Map")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.legend()
plt.grid(True)

# Save the map as a PNG file
map_file = "airports_map.png"
plt.savefig(map_file, dpi=300)
plt.show()
logging.info(f"Map of airports saved to {map_file}.")

#%% Example 8: Get runway details for specific airports
icao_list = ["KLAX", "KSFO", "KLGB"]
runway_details = get_runway_details(icao_list)
logging.info(f"Runway details for {len(icao_list)} airports:\n{runway_details}")

# %%
