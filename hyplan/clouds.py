"""
HyPlan Clouds

Overview:
Optical remote sensing of the Earth's surface often requires clear skies. Deploying airborne remote sensing
instruments can be costly, with daily costs for aircraft, labor, and per diem travel expenses for aircraft
and instrument teams. This script addresses the question:
"Statistically, how many days is it likely to take to acquire clear-sky observations for a given set of flight boxes?".

The script operates under several simplifying assumptions:
- Each flight box is "flyable" in a single day given clear skies from a single base of operations.
- Instantaneous MODIS Terra/Aqua overpasses are representative of clear-sky conditions throughout the flight day.
- Other environmental state parameters (e.g., tides, wind speeds) do not influence go/no-go decisions for flights.

Key Features:
1. **Cloud Data Processing**:
    - Reads geospatial polygon data (GeoJSON) representing flight areas for airborne optical remote sensing.
    - Fetches MODIS cloud fraction data from Google Earth Engine for specified years and day ranges.
    - Aggregates daily cloud fraction data for each polygon.

2. **Flight Simulation**:
    - Simulates daily flight schedules to visit polygons based on a maximum cloud fraction threshold.
    - Enforces constraints such as maximum consecutive flight days and optional weekend exclusions.

3. **Visualization**:
    - Generates heatmaps of cloud conditions, visit days, and rest days for each simulated year.
    - Produces cumulative distribution function (CDF) plots to estimate the likelihood of completing visits.

Campaigns that cross a year boundary (e.g. December to February) are supported:
set day_start > day_stop (e.g. day_start=335, day_stop=60).

References
----------
Gorelick, N. et al. (2017). Google Earth Engine: Planetary-scale
geospatial analysis for everyone. *Remote Sensing of Environment*, 202,
18-27. doi:10.1016/j.rse.2017.06.031

Data source: MODIS Terra (MOD09GA) and Aqua (MYD09GA) surface reflectance
daily L2G products from NASA LP DAAC, accessed via Google Earth Engine.
"""

# Core Libraries
import pandas as pd
from datetime import datetime, timedelta
import logging
from typing import Tuple, Dict

# Geospatial Libraries
import geopandas as gpd
from shapely import wkb

# Visualization Libraries
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from .exceptions import HyPlanRuntimeError, HyPlanValueError

__all__ = [
    "get_binary_cloud", "calculate_cloud_fraction", "create_date_ranges",
    "create_cloud_data_array_with_limit", "simulate_visits",
    "plot_yearly_cloud_fraction_heatmaps_with_visits",
]

logger = logging.getLogger(__name__)

_ee_initialized = False
_ee = None  # Populated by _get_ee()


def _get_ee():
    """Return the ``ee`` module, importing and initializing on first call.

    Raises:
        HyPlanRuntimeError: If ``earthengine-api`` is not installed or
            initialization fails.
    """
    global _ee_initialized, _ee
    if not _ee_initialized:
        try:
            import ee as _ee_mod
        except ImportError:
            raise HyPlanRuntimeError(
                "earthengine-api is required for cloud analysis. "
                "Install it with: pip install hyplan[clouds]"
            )
        try:
            _ee_mod.Initialize()
        except Exception as e:
            raise HyPlanRuntimeError(
                "Earth Engine initialization failed. "
                "Run ee.Authenticate() first."
            ) from e
        _ee = _ee_mod
        _ee_initialized = True
    return _ee


def _drop_z(geom: "BaseGeometry") -> "BaseGeometry":
    """
    Strip Z coordinates from a Shapely geometry, returning a 2D geometry.

    Args:
        geom: A Shapely geometry, potentially with Z coordinates.

    Returns:
        A 2D Shapely geometry with Z values removed.
    """
    return wkb.loads(wkb.dumps(geom, output_dimension=2))


def get_binary_cloud(image: "ee.Image") -> "ee.Image":
    """
    Generates a binary cloud mask for a given MODIS image.

    The MOD09GA/MYD09GA state_1km band encodes cloud state in bits 0-1:
      00 = clear, 01 = cloudy, 10 = mixed, 11 = not set.
    Any non-zero value (bits 0-1 != 00) is treated as cloudy.

    Parameters:
        image (ee.Image): An Earth Engine image with a "state_1km" quality assessment band.

    Returns:
        ee.Image: Binary cloud mask (1 for cloudy/mixed, 0 for clear) with an added "date_char" property.
    """
    _get_ee()
    qa = image.select("state_1km")
    clouds = qa.bitwiseAnd(3).gt(0)
    date_char = image.date().format('yyyy-MM-dd')
    return clouds.set("date_char", date_char)

def calculate_cloud_fraction(image: "ee.Image", polygon_geometry: "ee.Geometry") -> "ee.Feature":
    """
    Calculates the cloud fraction over a given polygon for a MODIS image.

    Parameters:
        image (ee.Image): An Earth Engine MODIS image.
        polygon_geometry (ee.Geometry): A polygon geometry representing the region of interest.

    Returns:
        ee.Feature: A feature containing the date and calculated cloud fraction for the polygon.
    """
    ee = _get_ee()
    reduction = image.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=polygon_geometry,
        scale=1000
    )
    cloud_fraction = reduction.get('state_1km')
    return ee.Feature(None, {'date_char': image.get('date_char'), 'cloud_fraction': cloud_fraction})

def create_date_ranges(day_start: int, day_stop: int, year_start: int, year_stop: int) -> list:
    """
    Creates date ranges for filtering Earth Engine image collections.

    Supports year-boundary crossings (e.g., day_start=335, day_stop=60 for a
    December-to-February campaign). When day_start > day_stop, each year-pair
    produces two date ranges: one from day_start to Dec 31 and one from Jan 1
    to day_stop in the following year.

    Parameters:
        day_start (int): Start day of the year (1-365).
        day_stop (int): End day of the year (1-365).
        year_start (int): Start year for the ranges.
        year_stop (int): End year for the ranges.

    Returns:
        list of tuples: A list of date range tuples (start_date, end_date) in
            YYYY-DDD or YYYY-MM-DD format suitable for Earth Engine filterDate.
    """
    date_ranges = []

    if day_start <= day_stop:
        # Normal case: campaign within a single calendar year
        for year in range(year_start, year_stop + 1):
            date_ranges.append((f"{year}-{day_start:03}", f"{year}-{day_stop + 1:03}"))
    else:
        # Year-boundary crossing: day_start (e.g., 335) > day_stop (e.g., 60)
        # Each "season" spans from day_start of year N to day_stop of year N+1
        for year in range(year_start, year_stop + 1):
            # Part 1: day_start to Dec 31 of current year
            date_ranges.append((f"{year}-{day_start:03}", f"{year + 1}-001"))
            # Part 2: Jan 1 to day_stop of following year
            date_ranges.append((f"{year + 1}-001", f"{year + 1}-{day_stop + 1:03}"))

    return date_ranges

def create_cloud_data_array_with_limit(polygon_file: str, year_start: int, year_stop: int, day_start: int, day_stop: int, limit: int = 5000) -> pd.DataFrame:
    """
    Processes MODIS cloud data for polygons and calculates daily cloud fractions.

    The polygon file must contain a 'Name' column identifying each polygon.

    Parameters:
        polygon_file (str): Path to a GeoJSON or shapefile containing polygons with a 'Name' column.
        year_start (int): Start year for data processing.
        year_stop (int): End year for data processing.
        day_start (int): Start day of the year for data processing.
        day_stop (int): End day of the year for data processing.
        limit (int, optional): Maximum number of images to process per date range. Default is 5000.

    Returns:
        pd.DataFrame: A DataFrame with columns 'polygon_id', 'year', 'day_of_year', and 'cloud_fraction'.
    """
    ee = _get_ee()
    try:
        gdf = gpd.read_file(polygon_file)
        if gdf.empty:
            raise HyPlanValueError("Polygon file is empty or invalid.")
    except Exception as e:
        raise HyPlanRuntimeError(f"Failed to load polygon file: {polygon_file}") from e

    if 'Name' not in gdf.columns:
        raise HyPlanValueError(f"Polygon file must contain a 'Name' column. Found columns: {list(gdf.columns)}")

    gdf = gdf[['Name', 'geometry']].copy()
    gdf['geometry'] = gdf['geometry'].apply(_drop_z)

    results = []
    date_ranges = create_date_ranges(day_start, day_stop, year_start, year_stop)

    try:
        cloud_data = ee.ImageCollection([])
        for start, stop in date_ranges:
            cloud_terra = ee.ImageCollection("MODIS/061/MOD09GA").filterDate(start, stop).limit(limit)
            cloud_aqua = ee.ImageCollection('MODIS/061/MYD09GA').filterDate(start, stop).limit(limit)
            cloud_data = cloud_data.merge(cloud_terra).merge(cloud_aqua)
        cloud_data = cloud_data.map(get_binary_cloud)
    except Exception as e:
        raise HyPlanRuntimeError("Error occurred while processing MODIS data.") from e

    for _, row in gdf.iterrows():
        polygon_name = row['Name']
        polygon_geometry = ee.Geometry(row['geometry'].__geo_interface__)

        mapped_results = cloud_data.map(lambda image: calculate_cloud_fraction(image, polygon_geometry))
        feature_list = ee.FeatureCollection(mapped_results).limit(limit).getInfo()['features']

        for feature in feature_list:
            properties = feature['properties']
            date_char = properties.get('date_char')
            if date_char is None:
                continue
            cloud_fraction = properties.get('cloud_fraction')
            if cloud_fraction is not None:
                year, month, day = [int(x) for x in date_char.split('-')]
                day_of_year = pd.Timestamp(year=year, month=month, day=day).dayofyear
                results.append({'year': year, 'day_of_year': day_of_year, 'polygon_id': polygon_name, 'cloud_fraction': cloud_fraction})

    results_df = pd.DataFrame(results)
    aggregated_df = results_df.groupby(['polygon_id', 'year', 'day_of_year']).mean().reset_index()
    return aggregated_df

def simulate_visits(
    df: pd.DataFrame,
    day_start: int,
    day_stop: int,
    year_start: int,
    year_stop: int,
    cloud_fraction_threshold: float = 0.10,
    rest_day_threshold: int = 6,
    exclude_weekends: bool = False,
    debug: bool = False
) -> Tuple[pd.DataFrame, Dict[int, Dict[str, list]], Dict[int, list]]:
    """
    Simulate visits to polygons based on cloud fraction thresholds, ensuring no more than one visit per day.
    Adds rest days after a set number of consecutive visits and resets counters on weekends or when no polygons meet the threshold.

    On each visitable day, the alphabetically first unvisited polygon that meets the cloud threshold is chosen.
    Rest days count toward total_days but no polygon is visited.

    Parameters:
        df (pd.DataFrame): Cloud fraction data with columns: 'polygon_id', 'year', 'day_of_year', 'cloud_fraction'.
        day_start (int): Start day of the year for simulation.
        day_stop (int): End day of the year for simulation.
        year_start (int): Start year for simulation.
        year_stop (int): End year for simulation.
        cloud_fraction_threshold (float): Maximum allowable cloud fraction for a visit.
        rest_day_threshold (int): Maximum number of consecutive visits before a rest day is required.
        exclude_weekends (bool): If True, skip weekends and reset the counter for rest days.
        debug (bool): If True, enable detailed logging for debugging.

    Returns:
        Tuple[pd.DataFrame, Dict[int, Dict[str, list]], Dict[int, list]]:
            - DataFrame summarizing total days simulated per year.
            - Dictionary of visit days for each polygon, organized by year.
            - Dictionary of rest days for each year.
    """
    if debug:
        logging.getLogger().setLevel(logging.DEBUG)

    # Build the day sequence. When day_start > day_stop the campaign
    # crosses a year boundary (e.g. Dec 1 through Feb 28).
    crosses_year = day_start > day_stop

    visit_days = []
    visit_tracker = {}
    rest_days = {}

    for year in range(year_start, year_stop + 1):
        visited_polygons = set()
        remaining_polygons = set(df['polygon_id'].unique())
        visit_tracker[year] = {}
        rest_days[year] = []
        total_days = 0
        consecutive_visits = 0

        if crosses_year:
            # e.g. day 335..365 in *year*, then day 1..day_stop in *year+1*
            last_day_of_year = (datetime(year + 1, 1, 1) - datetime(year, 1, 1)).days
            day_sequence = list(range(day_start, last_day_of_year + 1)) + list(range(1, day_stop + 1))
        else:
            day_sequence = list(range(day_start, day_stop + 1))

        for seq_idx, current_day_of_year in enumerate(day_sequence):
            # Determine which calendar year this day falls in
            if crosses_year and current_day_of_year < day_start:
                current_year = year + 1
            else:
                current_year = year

            total_days += 1
            current_date = datetime(current_year, 1, 1) + timedelta(days=current_day_of_year - 1)

            if exclude_weekends and current_date.weekday() >= 5:
                logger.debug(f"Skipping weekend on day {current_day_of_year} of year {current_year}")
                consecutive_visits = 0
                continue

            daily_df = df[(df['year'] == current_year) & (df['day_of_year'] == current_day_of_year)]
            daily_df = daily_df[~daily_df['polygon_id'].isin(visited_polygons)]
            visitable_polygons = daily_df[daily_df['cloud_fraction'] <= cloud_fraction_threshold]

            if not visitable_polygons.empty:
                if consecutive_visits < rest_day_threshold:
                    polygon_to_visit = visitable_polygons.sort_values(by='polygon_id').iloc[0]
                    polygon_id = polygon_to_visit['polygon_id']

                    visited_polygons.add(polygon_id)
                    remaining_polygons.discard(polygon_id)

                    if polygon_id not in visit_tracker[year]:
                        visit_tracker[year][polygon_id] = []
                    visit_tracker[year][polygon_id].append(current_day_of_year)

                    logger.debug(f"Visiting polygon {polygon_id} on day {current_day_of_year} of year {current_year}")
                    consecutive_visits += 1
                else:
                    rest_days[year].append(current_day_of_year)
                    logger.info(f"Rest day added on day {current_day_of_year} of year {current_year}")
                    consecutive_visits = 0
            else:
                logger.debug(f"No visitable polygons on day {current_day_of_year} of year {current_year}")
                consecutive_visits = 0

            if not remaining_polygons:
                logger.info(f"All polygons visited for year {year}.")
                break

        visit_days.append({'year': year, 'days': total_days})

    return pd.DataFrame(visit_days), visit_tracker, rest_days

def plot_yearly_cloud_fraction_heatmaps_with_visits(
    cloud_data_df: pd.DataFrame, visit_tracker: Dict[int, Dict[str, list]], rest_days: Dict[int, list],
    cloud_fraction_threshold: float = 0.10, exclude_weekends: bool = False,
    day_start: int = 1, day_stop: int = 365
) -> None:
    """
    Generates heatmaps of cloud fraction for each year, including visit markers and rest day highlights.

    Parameters:
        cloud_data_df (pd.DataFrame): DataFrame with columns 'polygon_id', 'year', 'day_of_year', and 'cloud_fraction'.
        visit_tracker (dict): A dictionary of visit days for each polygon, organized by year.
        rest_days (dict): A dictionary of rest days for each year.
        cloud_fraction_threshold (float): Threshold to classify cloud fraction as clear (white) or cloudy (black).
        exclude_weekends (bool): If True, weekends are highlighted and skipped in the heatmap.
        day_start (int): Start day of the year to include in the heatmap.
        day_stop (int): End day of the year to include in the heatmap.

    Returns:
        None: Displays heatmaps for each year with clear/cloudy days, visit days, and rest day markers.
    """
    required_columns = {'polygon_id', 'year', 'day_of_year', 'cloud_fraction'}
    if not required_columns.issubset(cloud_data_df.columns):
        raise HyPlanValueError(f"Input DataFrame must contain columns: {required_columns}")

    # Define a custom colormap: lightgrey (no data), white (clear), black (cloudy), grey (visited), purple (weekend), orange (rest days)
    cmap = mcolors.ListedColormap(['lightgrey', 'white', 'black', 'grey', 'purple', 'orange'])
    bounds = [-1.5, -0.5, 0.5, 1.5, 2.5, 3.5, 4.5]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    unique_years = cloud_data_df['year'].unique()
    for year in sorted(unique_years):
        year_data = cloud_data_df[(cloud_data_df['year'] == year) &
                                  (cloud_data_df['day_of_year'] >= day_start) &
                                  (cloud_data_df['day_of_year'] <= day_stop)]
        heatmap_data = year_data.pivot(index='polygon_id', columns='day_of_year', values='cloud_fraction')
        heatmap_data = heatmap_data.reindex(columns=range(day_start, day_stop + 1), fill_value=float('nan'))

        binary_data = (heatmap_data > cloud_fraction_threshold).astype(int)
        binary_data[heatmap_data.isna()] = -1
        status_data = binary_data.copy()

        stars_x = []
        stars_y = []
        rest_days_set = set(rest_days.get(year, [])) if rest_days else set()

        for i, polygon_id in enumerate(status_data.index):
            if polygon_id in visit_tracker.get(year, {}):
                visit_days_list = sorted(visit_tracker[year][polygon_id])
                for visit_day in visit_days_list:
                    if day_start <= visit_day <= day_stop:
                        stars_x.append(visit_day - day_start + 0.5)
                        stars_y.append(i + 0.5)

                        for day in range(visit_day + 1, day_stop + 1):
                            if exclude_weekends:
                                weekday = (datetime(year, 1, 1) + timedelta(days=day - 1)).weekday()
                                if weekday < 5:
                                    status_data.loc[polygon_id, day] = 2
                            else:
                                status_data.loc[polygon_id, day] = 2

        for rest_day in rest_days_set:
            if day_start <= rest_day <= day_stop:
                status_data.iloc[:, rest_day - day_start] = 4

        if exclude_weekends:
            for day in range(day_start, day_stop + 1):
                weekday = (datetime(year, 1, 1) + timedelta(days=day - 1)).weekday()
                if weekday >= 5:
                    status_data.loc[:, day] = 3

        try:
            import seaborn as sns
        except ImportError:
            raise HyPlanRuntimeError(
                "seaborn is required for cloud heatmaps. "
                "Install it with: pip install hyplan[clouds]"
            )
        plt.figure(figsize=(16, 8))
        ax = sns.heatmap(status_data, cmap=cmap, norm=norm, cbar=False,
                         linewidths=0.5, linecolor='gray', square=True)
        plt.scatter(stars_x, stars_y, color='red', marker='*', s=150, label='Visit Day')
        plt.title(f'Cloud Fraction Heatmap with Visits for Year {year}')
        plt.xlabel('Day of Year')
        plt.ylabel('Polygon ID')
        plt.legend(loc='upper right')
        plt.tight_layout()
        plt.show()
