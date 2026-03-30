"""Solar position and illumination timing.

Computes solar elevation thresholds and data collection windows using
the Solar Position Algorithm (SPA).

References
----------
Reda, I. and Andreas, A. (2004). Solar position algorithm for solar
radiation applications. *Solar Energy*, 76(5), 577-589.
doi:10.1016/j.solener.2003.12.003
"""

import pandas as pd
import numpy as np
from sunposition import sunpos
from datetime import datetime, date, timedelta
from typing import List, Union
import matplotlib.pyplot as plt
from .exceptions import HyPlanValueError


def solar_threshold_times(latitude: float, longitude: float, start_date: str, end_date: str, thresholds: List[float], timezone_offset: int = 0) -> pd.DataFrame:
    """
    Find times when the solar elevation crosses specified thresholds.

    Args:
        latitude (float): Latitude of the location.
        longitude (float): Longitude of the location.
        start_date (str): Start date in 'YYYY-MM-DD' format.
        end_date (str): End date in 'YYYY-MM-DD' format.
        thresholds (list): List of 1 or 2 solar elevation thresholds in degrees (e.g., [35] or [35, 50]).
        timezone_offset (int): Timezone offset from UTC in hours (e.g., -8 for PST, 1 for CET).

    Returns:
        pandas.DataFrame: DataFrame with reordered columns: ['Date', 'Rise_<lower>', 'Rise_<upper>', 'Set_<upper>', 'Set_<lower>'].
    """
    # Ensure thresholds is a list of 1 or 2 elements
    if not (1 <= len(thresholds) <= 2):
        raise HyPlanValueError("Thresholds must be a list with 1 or 2 elements.")

    # Generate all timestamps at 1-minute intervals in UTC
    start_datetime = datetime.strptime(start_date, '%Y-%m-%d')
    end_datetime = datetime.strptime(end_date, '%Y-%m-%d')
    timestamps = pd.date_range(start=start_datetime, end=end_datetime + timedelta(days=1) - timedelta(minutes=1), freq='1min', tz='UTC')

    # Adjust timestamps to local timezone
    local_timestamps = timestamps + pd.Timedelta(hours=timezone_offset)

    # Vectorized calculation of solar positions using UTC timestamps
    _, zenith, *_ = sunpos(timestamps, latitude, longitude, elevation=0)
    elevation = 90 - zenith

    # Prepare DataFrame for results
    results = []

    # Iterate over each day
    for day in pd.date_range(start=start_datetime, end=end_datetime, freq='D'):
        day_mask = (local_timestamps.date == day.date())
        daily_times = local_timestamps[day_mask]
        daily_elevation = elevation[day_mask]

        if len(daily_times) == 0:
            continue  # Skip days with no data

        day_results = {'Date': day.date()}

        if len(thresholds) == 2:
            lower, upper = sorted(thresholds)
            for threshold, label in zip([lower, upper], [f'_{lower}', f'_{upper}']):
                rise_time = None
                fall_time = None

                above_threshold = daily_elevation > threshold

                if np.any(above_threshold):
                    rise_idx = np.argmax(above_threshold)  # First True
                    fall_idx = len(above_threshold) - np.argmax(above_threshold[::-1]) - 1  # Last True

                    rise_time = daily_times[rise_idx].strftime('%H:%M:%S')
                    fall_time = daily_times[fall_idx].strftime('%H:%M:%S')

                day_results[f'Rise{label}'] = rise_time
                day_results[f'Set{label}'] = fall_time

        elif len(thresholds) == 1:
            lower = thresholds[0]
            day_results[f'Rise_{lower}'] = None
            day_results[f'Set_{lower}'] = None

            above_threshold = daily_elevation > lower

            if np.any(above_threshold):
                rise_idx = np.argmax(above_threshold)  # First True
                fall_idx = len(above_threshold) - np.argmax(above_threshold[::-1]) - 1  # Last True

                day_results[f'Rise_{lower}'] = daily_times[rise_idx].strftime('%H:%M:%S')
                day_results[f'Set_{lower}'] = daily_times[fall_idx].strftime('%H:%M:%S')

        results.append(day_results)

    # Reorder and return DataFrame
    if len(thresholds) == 2:
        lower, upper = sorted(thresholds)
        columns = ['Date', f'Rise_{lower}', f'Rise_{upper}', f'Set_{upper}', f'Set_{lower}']
    else:
        columns = ['Date', f'Rise_{lower}', f'Set_{lower}']

    return pd.DataFrame(results, columns=columns)


def solar_azimuth(latitude: float, longitude: float, dt: datetime, elevation: float = 0) -> float:
    """
    Return the solar azimuth (in degrees) at a given latitude, longitude, and datetime.
    
    Args:
        latitude (float): Latitude of the location.
        longitude (float): Longitude of the location.
        dt (datetime): The datetime for which to calculate the solar azimuth. This should be in UTC.
        elevation (float, optional): Elevation to use in the sunpos calculation (default 0).
    
    Returns:
        float: Solar azimuth in degrees.
    """
    ts = pd.DatetimeIndex([dt], tz='UTC')
    azimuth, zenith, *_ = sunpos(ts, latitude, longitude, elevation=elevation)
    return azimuth[0]


def solar_position_increments(latitude: float, longitude: float, date: Union[str, date, datetime], min_elevation: float, timezone_offset: int = 0, increment: str = '10min') -> pd.DataFrame:
    """
    Return the solar azimuth and solar elevation at user-specified increments for a given date and location,
    but only for times when the solar elevation exceeds the specified minimum.
    
    Args:
        latitude (float): Latitude of the location.
        longitude (float): Longitude of the location.
        date (str or datetime.date): Date in 'YYYY-MM-DD' format or as a date object.
        min_elevation (float): Minimum solar elevation (in degrees) required to include the time.
        timezone_offset (int, optional): Timezone offset from UTC in hours (e.g., -8 for PST). Default is 0.
        increment (str, optional): Frequency increment for sampling times (e.g., '10min'). Default is '10min'.
    
    Returns:
        pandas.DataFrame: DataFrame with columns:
            - 'Time': Local time (HH:MM:SS),
            - 'Solar Azimuth': Solar azimuth in degrees,
            - 'Solar Elevation': Solar elevation in degrees.
    """
    # Convert date to a datetime object at midnight.
    if isinstance(date, str):
        date_dt = datetime.strptime(date, '%Y-%m-%d')
    elif isinstance(date, datetime):
        date_dt = date
    else:
        # Assume it's a datetime.date
        date_dt = datetime.combine(date, datetime.min.time())
    
    # Define the start and end of the day in UTC.
    start_datetime = datetime.combine(date_dt.date(), datetime.min.time())
    end_datetime = start_datetime + timedelta(days=1)
    
    # Create a DateTimeIndex in UTC at the specified increments.
    # Subtract one increment from the end to avoid including the next day's midnight.
    timestamps_utc = pd.date_range(start=start_datetime,
                                   end=end_datetime - pd.Timedelta(increment),
                                   freq=increment,
                                   tz='UTC')
    
    # Convert UTC timestamps to local time using the provided timezone offset.
    local_timestamps = timestamps_utc + pd.Timedelta(hours=timezone_offset)
    
    # Compute solar positions using the UTC timestamps.
    # The sunpos function returns (azimuth, zenith, ...). Solar elevation = 90 - zenith.
    azimuth, zenith, *_ = sunpos(timestamps_utc, latitude, longitude, elevation=0)
    solar_elevation = 90 - zenith
    
    # Only keep times when the solar elevation exceeds the specified threshold.
    valid = solar_elevation > min_elevation
    
    df = pd.DataFrame({
        'Time': local_timestamps[valid].strftime('%H:%M:%S'),
        'Azimuth': azimuth[valid],
        'Elevation': solar_elevation[valid]
    })
    
    return df

def plot_solar_positions(df_positions: pd.DataFrame) -> None:
    """
    Plot the solar azimuth and elevation for a given day.
    
    Args:
        df_positions (pd.DataFrame): DataFrame containing the 'Solar Azimuth', 'Time', and 'Elevation' columns.
    """
    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.plot(df_positions['Time'], df_positions['Elevation'], label='Solar Elevation', color='tab:blue')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Solar Elevation', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.legend(loc='upper left')
    ax1.grid(True)

    # Rotate x-axis labels and set the frequency of the labels
    plt.xticks(rotation=45)
    ax1.xaxis.set_major_locator(plt.MaxNLocator(nbins=10))

    # Create a second y-axis
    ax2 = ax1.twinx()
    ax2.plot(df_positions['Time'], df_positions['Azimuth'], label='Azimuth', color='tab:orange')
    ax2.set_ylabel('Azimuth', color='tab:orange')
    ax2.tick_params(axis='y', labelcolor='tab:orange')
    ax2.legend(loc='upper right')

    plt.title('Solar Azimuth and Elevation Plot')
    plt.show()

