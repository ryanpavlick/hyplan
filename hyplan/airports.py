import os
import geopandas as gpd
import pandas as pd
import logging
from pathlib import Path
from shapely.geometry import Point
from typing import List, Union

from .units import convert_distance, ureg
from .download import download_file
from .geometry import haversine

__all__ = [
    "Airport", "initialize_data", "find_nearest_airport", "find_nearest_airports",
    "airports_within_radius", "get_airports", "get_runways", "get_airport_details",
    "get_longest_runway", "generate_geojson", "get_runway_details"
]

OUR_AIRPORTS_URL = "https://raw.githubusercontent.com/davidmegginson/ourairports-data/main/airports.csv"
RUNWAYS_URL = "https://raw.githubusercontent.com/davidmegginson/ourairports-data/main/runways.csv"
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "hyplan"

logger = logging.getLogger(__name__)

# Global variables for airport and runway data
gdf_airports: gpd.GeoDataFrame = None
df_runways: pd.DataFrame = None


class Airport:
    """Class to represent an airport with Shapely Point geometry."""
    def __init__(self, icao: str):
        global gdf_airports
        if gdf_airports is None:
            initialize_data()
        if icao not in gdf_airports.index:
            raise ValueError(f"Airport ICAO code {icao} not found in the dataset.")

        airport_data = gdf_airports.loc[icao]

        longitude = airport_data.get("longitude")
        latitude = airport_data.get("latitude")
        if pd.isna(longitude) or pd.isna(latitude):
            raise ValueError(f"Longitude or latitude is missing for airport {icao}")
        try:
            self._geometry = Point(float(longitude), float(latitude))
        except (TypeError, ValueError):
            raise ValueError(f"Invalid longitude/latitude for airport {icao}: {longitude}, {latitude}")

        self._icao = airport_data['icao_code']
        self._iata = airport_data['iata_code']
        self._name = airport_data['name']
        self._iso_country = airport_data['iso_country']
        self._municipality = airport_data['municipality']
        elevation_ft = airport_data['elevation_ft']
        if pd.isna(elevation_ft):
            self._elevation_ft = None
            self._elevation = None
        else:
            self._elevation_ft = float(elevation_ft)
            self._elevation = (self._elevation_ft * ureg.foot).to(ureg.meter)

    def __repr__(self):
        return f"<Airport {self._icao} - {self._name}>"

    @property
    def longitude(self):
        """Longitude of the airport."""
        return self._geometry.x

    @property
    def latitude(self):
        """Latitude of the airport."""
        return self._geometry.y

    @property
    def geometry(self):
        """Shapely Point geometry of the airport."""
        return self._geometry

    @property
    def icao_code(self):
        """ICAO code of the airport."""
        return self._icao

    @property
    def iata_code(self):
        """IATA code of the airport."""
        return self._iata

    @property
    def name(self):
        """Name of the airport."""
        return self._name

    @property
    def country(self):
        """ISO country code of the airport."""
        return self._iso_country

    @property
    def municipality(self):
        """Municipality of the airport."""
        return self._municipality

    @property
    def elevation(self):
        """Elevation of the airport in meters. Returns None if not available."""
        return self._elevation

    @property
    def elevation_ft(self):
        """Elevation of the airport in feet. Returns None if not available."""
        return self._elevation_ft

    @property
    def runways(self) -> pd.DataFrame:
        """Runway details for this airport as a DataFrame."""
        global df_runways
        if df_runways is None:
            raise RuntimeError("Runways data has not been initialized. Please run initialize_data().")
        return df_runways[df_runways['airport_ident'] == self._icao]


def _filter_airports_by_country(df_airports: pd.DataFrame, countries: List[str]) -> pd.DataFrame:
    """Filter airports by country codes."""
    return df_airports[df_airports['iso_country'].isin(countries)]

def _filter_airports_by_type(df_airports: pd.DataFrame, airport_types: List[str]) -> pd.DataFrame:
    """Filter airports by type."""
    return df_airports[df_airports['type'].isin(airport_types)]

def _filter_runways(df_runways: pd.DataFrame, length_ft: int = None, surface: Union[str, List[str]] = None, partial_match: bool = False) -> pd.DataFrame:
    """Filter runways based on length and/or surface type with optional partial matching."""
    if length_ft:
        df_runways = df_runways[df_runways['length_ft'] >= length_ft]
    if surface:
        if isinstance(surface, str):
            surface = [surface]
        if partial_match:
            pattern = "|".join([s.upper() for s in surface])
            df_runways = df_runways[df_runways['surface'].str.upper().str.contains(pattern, na=False)]
        else:
            df_runways = df_runways[df_runways['surface'].str.upper().isin([s.upper() for s in surface])]
    return df_runways

def load_airports(
    filepath: str,
    countries: List[str] = None,
    min_runway_length: int = None,
    runway_surface: Union[str, List[str]] = None,
    airport_types: List[str] = None
) -> gpd.GeoDataFrame:
    """Load and preprocess airport data with filters for country codes, runway length, surface type, and airport types."""
    try:
        df_airports = pd.read_csv(filepath, encoding="ISO-8859-1")
    except FileNotFoundError:
        raise FileNotFoundError(f"The file {filepath} does not exist. Please run initialize_data().")

    df_airports['icao_code'] = df_airports['ident']
    df_airports.rename(columns={"latitude_deg": "latitude", "longitude_deg": "longitude"}, inplace=True)
    df_airports.set_index('ident', inplace=True)

    columns_to_drop = ['id', 'scheduled_service', 'local_code', 'gps_code',
                       'home_link', 'wikipedia_link', 'keywords']
    df_airports.drop(columns_to_drop, axis=1, inplace=True, errors='ignore')

    if not airport_types:
        airport_types = ['large_airport', 'medium_airport', 'small_airport']
    df_airports = _filter_airports_by_type(df_airports, airport_types)

    if countries:
        df_airports = _filter_airports_by_country(df_airports, countries)

    if min_runway_length or runway_surface:
        runways_path = os.path.join(os.path.dirname(filepath), "runways.csv")
        df_rwy = load_runways(runways_path)
        df_rwy = _filter_runways(df_rwy, length_ft=min_runway_length, surface=runway_surface)
        valid_airports = df_rwy['airport_ident'].unique()
        df_airports = df_airports[df_airports['icao_code'].isin(valid_airports)]

    gdf = gpd.GeoDataFrame(
        df_airports, geometry=gpd.points_from_xy(df_airports.longitude, df_airports.latitude))
    gdf.sindex  # Create spatial index
    return gdf

def load_runways(filepath: str) -> pd.DataFrame:
    """Load and preprocess runway data."""
    try:
        df_runways = pd.read_csv(filepath, encoding="ISO-8859-1")
    except FileNotFoundError:
        raise FileNotFoundError(f"The file {filepath} does not exist. Please run initialize_data().")

    columns_to_keep = ['airport_ident', 'length_ft', 'width_ft', 'surface',
                       'le_heading_degT', 'he_heading_degT']
    df_runways = df_runways[columns_to_keep]
    return df_runways

def initialize_data(
    countries: List[str] = None,
    min_runway_length: int = None,
    runway_surface: Union[str, List[str]] = None,
    airport_types: List[str] = None,
    cache_dir: Union[str, Path] = None,
    refresh: bool = False
) -> None:
    """Initialize global variables for airports and runways data with filtering options.

    Args:
        countries: ISO country codes to filter airports by.
        min_runway_length: Minimum runway length in feet.
        runway_surface: Runway surface type(s) to filter by.
        airport_types: Airport types to include (default: large, medium, small).
        cache_dir: Directory to store downloaded data files. Defaults to ~/.cache/hyplan/.
        refresh: If True, re-download data files even if they already exist.
    """
    global gdf_airports, df_runways
    cache_path = Path(cache_dir) if cache_dir else DEFAULT_CACHE_DIR
    cache_path.mkdir(parents=True, exist_ok=True)

    airports_file = str(cache_path / "airports.csv")
    runways_file = str(cache_path / "runways.csv")

    download_file(airports_file, OUR_AIRPORTS_URL, replace=refresh)
    download_file(runways_file, RUNWAYS_URL, replace=refresh)

    gdf_airports = load_airports(
        airports_file,
        countries=countries,
        min_runway_length=min_runway_length,
        runway_surface=runway_surface,
        airport_types=airport_types
    )
    df_runways = load_runways(runways_file)

def find_nearest_airport(lat: float, lon: float) -> str:
    """Find the nearest airport to a given latitude and longitude.

    Returns:
        str: ICAO code of the nearest airport.
    """
    global gdf_airports
    if gdf_airports is None:
        raise RuntimeError("Airports data has not been initialized. Please run initialize_data().")
    point = Point(lon, lat)
    idx = gdf_airports.sindex.nearest(point)
    return gdf_airports.iloc[idx[0]]['icao_code']

def find_nearest_airports(lat: float, lon: float, n: int = 5) -> List[str]:
    """Find the N nearest airports to a given latitude and longitude.

    Returns:
        List[str]: ICAO codes of the nearest airports, ordered by proximity.
    """
    global gdf_airports
    if gdf_airports is None:
        raise RuntimeError("Airports data has not been initialized. Please run initialize_data().")
    point = Point(lon, lat)
    idxs = gdf_airports.sindex.nearest(point, max_results=n)
    return gdf_airports.iloc[idxs]['icao_code'].tolist()

def airports_within_radius(
    lat: float, lon: float, radius: float, unit: str = "kilometers",
    return_details: bool = False
) -> Union[List[str], gpd.GeoDataFrame]:
    """Find all airports within a specified radius of a given point.

    Args:
        lat: Latitude of the center point.
        lon: Longitude of the center point.
        radius: Search radius.
        unit: Distance unit for radius (default: "kilometers").
        return_details: If True, return a GeoDataFrame with a 'distance_m' column
                        instead of a list of ICAO codes.

    Returns:
        List of ICAO codes, or a GeoDataFrame if return_details is True.
    """
    global gdf_airports
    if gdf_airports is None:
        raise RuntimeError("Airports data has not been initialized. Please run initialize_data().")

    point = Point(lon, lat)
    radius_m = convert_distance(radius, unit, "meters")

    buffer = point.buffer(radius_m / 111139.0)  # Approximate degree buffer
    possible_matches = gdf_airports[gdf_airports.intersects(buffer)].copy()

    possible_matches['distance_m'] = possible_matches.apply(
        lambda row: haversine(lat, lon, row.latitude, row.longitude), axis=1
    )
    within_radius = possible_matches[possible_matches['distance_m'] <= radius_m]

    if return_details:
        return within_radius.sort_values('distance_m')
    return within_radius['icao_code'].tolist()

def get_airports() -> gpd.GeoDataFrame:
    """Get the globally initialized GeoDataFrame of airports."""
    global gdf_airports
    if gdf_airports is None:
        raise RuntimeError("Airports data has not been initialized. Please run initialize_data().")
    return gdf_airports

def get_runways() -> pd.DataFrame:
    """Get the globally initialized DataFrame of runways."""
    global df_runways
    if df_runways is None:
        raise RuntimeError("Runway data has not been initialized. Please run initialize_data().")
    return df_runways

def get_airport_details(icao_codes: Union[str, List[str]]) -> pd.DataFrame:
    """Get details of airports for given ICAO code(s)."""
    global gdf_airports
    if isinstance(icao_codes, str):
        icao_codes = [icao_codes]
    if gdf_airports is None:
        raise RuntimeError("Airports data has not been initialized. Please run initialize_data().")
    return gdf_airports[gdf_airports['icao_code'].isin(icao_codes)]

def get_longest_runway(icao: str) -> float:
    """Return the length in feet of the longest runway at the given airport.

    Returns:
        float: Longest runway length in feet, or None if no runway data is available.
    """
    global df_runways
    if df_runways is None:
        raise RuntimeError("Runways data has not been initialized. Please run initialize_data().")
    rows = df_runways[df_runways['airport_ident'] == icao]
    if rows.empty:
        return None
    return float(rows['length_ft'].max())

def generate_geojson(filepath: str = "airports.geojson", icao_codes: Union[str, List[str]] = None) -> None:
    """
    Generate a GeoJSON file of the airports using GeoPandas with CRS explicitly set to EPSG:4326.

    Args:
        filepath (str): Path to save the GeoJSON file. Defaults to "airports.geojson".
        icao_codes (Union[str, List[str]]): List of ICAO codes to subset the GeoJSON. If None, export all airports.
    """
    global gdf_airports
    if gdf_airports is None:
        raise RuntimeError("Airports data has not been initialized. Please run initialize_data().")

    if icao_codes:
        if isinstance(icao_codes, str):
            icao_codes = [icao_codes]
        subset = gdf_airports[gdf_airports['icao_code'].isin(icao_codes)].copy()
    else:
        subset = gdf_airports.copy()

    if subset.crs is None:
        logger.warning("GeoDataFrame has no CRS. Setting CRS to EPSG:4326.")
        subset.set_crs("EPSG:4326", inplace=True)
    elif subset.crs.to_string() != "EPSG:4326":
        subset = subset.to_crs("EPSG:4326")

    subset.to_file(filepath, driver="GeoJSON")
    logger.info(f"GeoJSON file generated at {filepath} with CRS EPSG:4326 and {len(subset)} airports.")

def get_runway_details(icao_codes: Union[str, List[str]]) -> pd.DataFrame:
    """
    Retrieve details of all runways for one or more airports.

    Args:
        icao_codes (Union[str, List[str]]): A single ICAO code or a list of ICAO codes.

    Returns:
        pd.DataFrame: A DataFrame with runway details for the given airport(s).
    """
    global df_runways
    if df_runways is None:
        raise RuntimeError("Runways data has not been initialized. Please run initialize_data().")

    if isinstance(icao_codes, str):
        icao_codes = [icao_codes]

    return df_runways[df_runways['airport_ident'].isin(icao_codes)]
