"""Solar position and illumination timing.

Computes solar elevation thresholds and data collection windows.

Solar position is computed via the Skyfield library using JPL's DE421
planetary ephemeris, which is bundled with hyplan at
``hyplan/data/de421.bsp`` so calculations work fully offline.

References
----------
Reda, I. and Andreas, A. (2004). Solar position algorithm for solar
radiation applications. *Solar Energy*, 76(5), 577-589.
doi:10.1016/j.solener.2003.12.003

Folkner, W. M., Williams, J. G., & Boggs, D. H. (2009). *The Planetary
and Lunar Ephemeris DE 421*. JPL Interoffice Memorandum 343R-08-003,
NASA Jet Propulsion Laboratory.
https://ssd.jpl.nasa.gov/planets/eph_export.html

Rhodes, B. (2019). Skyfield: High precision research-grade positions
for planets and Earth satellites generator. Astrophysics Source Code
Library, ascl:1907.024.
"""

import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from typing import List, Optional, Union
import matplotlib.pyplot as plt
from .exceptions import HyPlanValueError


# ---------------------------------------------------------------------------
# Solar position via Skyfield
# ---------------------------------------------------------------------------
#
# Drop-in replacement for ``sunposition.sunpos`` that uses Skyfield (already a
# required dependency, used elsewhere for SGP4 propagation). This removes the
# pip-only ``sunposition`` package from hyplan's required dependencies so the
# project can be installed entirely from conda-forge.
#
# The wrapper preserves the original return signature
# ``(azimuth, zenith, *_)`` so existing call sites that unpack with ``*_``
# continue to work unchanged.

_SKYFIELD_TS = None
_SKYFIELD_SUN = None
_SKYFIELD_EARTH = None


def _skyfield_handles():
    """Lazily load and cache the Skyfield timescale and DE421 ephemeris."""
    global _SKYFIELD_TS, _SKYFIELD_SUN, _SKYFIELD_EARTH
    if _SKYFIELD_TS is None:
        from importlib.resources import files
        from skyfield.api import load as sf_load, load_file
        _SKYFIELD_TS = sf_load.timescale()
        # Load the bundled DE421 ephemeris from the package data directory
        # rather than letting Skyfield download it into the user's cwd.
        bsp_path = files("hyplan.data").joinpath("de421.bsp")
        eph = load_file(str(bsp_path))
        _SKYFIELD_EARTH = eph["earth"]
        _SKYFIELD_SUN = eph["sun"]
    return _SKYFIELD_TS, _SKYFIELD_EARTH, _SKYFIELD_SUN


def _to_utc_datetimes(dt):
    """Coerce ``dt`` (scalar / list / ndarray / DatetimeIndex) to a list of
    timezone-aware UTC ``datetime`` objects suitable for ``ts.from_datetimes``.
    """
    from skyfield.api import utc as sf_utc

    if isinstance(dt, pd.DatetimeIndex):
        if dt.tz is None:
            dt = dt.tz_localize("UTC")
        else:
            dt = dt.tz_convert("UTC")
        return [d.to_pydatetime() for d in dt]

    if isinstance(dt, (list, tuple, np.ndarray)):
        out = []
        for d in np.asarray(dt).ravel():
            if isinstance(d, np.datetime64):
                d = pd.Timestamp(d).to_pydatetime()
            if isinstance(d, pd.Timestamp):
                d = d.to_pydatetime()
            if d.tzinfo is None:
                d = d.replace(tzinfo=sf_utc)
            out.append(d)
        return out

    # scalar
    if isinstance(dt, np.datetime64):
        dt = pd.Timestamp(dt).to_pydatetime()
    if isinstance(dt, pd.Timestamp):
        dt = dt.to_pydatetime()
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=sf_utc)
    return [dt]


def sunpos(dt, latitude, longitude, elevation=0, radians=False):
    """Compute solar azimuth and zenith via Skyfield.

    Drop-in replacement for ``sunposition.sunpos`` covering the input shapes
    used in hyplan: scalar datetime + scalar lat/lon, ``pd.DatetimeIndex`` +
    scalar lat/lon, and arrays of datetimes + arrays of lat/lon (broadcast
    elementwise).

    Args:
        dt: Datetime(s). Naive datetimes are assumed UTC.
        latitude: Latitude in degrees (scalar or array).
        longitude: Longitude in degrees (scalar or array).
        elevation: Elevation in meters above WGS-84 (scalar or array).
        radians: If True, return values in radians; otherwise degrees.

    Returns:
        Tuple ``(azimuth, zenith, ra, dec, h)`` to mirror ``sunposition.sunpos``.
        Only ``azimuth`` and ``zenith`` are populated; the remaining slots are
        arrays of NaN of matching shape, kept so that existing call sites can
        unpack with ``*_``.
    """
    ts, earth, sun = _skyfield_handles()

    dts = _to_utc_datetimes(dt)
    lat_arr = np.atleast_1d(np.asarray(latitude, dtype=float))
    lon_arr = np.atleast_1d(np.asarray(longitude, dtype=float))
    elev_arr = np.atleast_1d(np.asarray(elevation, dtype=float))

    n = max(len(dts), lat_arr.size, lon_arr.size, elev_arr.size)
    if len(dts) == 1 and n > 1:
        dts = dts * n
    if lat_arr.size == 1 and n > 1:
        lat_arr = np.broadcast_to(lat_arr, (n,)).copy()
    if lon_arr.size == 1 and n > 1:
        lon_arr = np.broadcast_to(lon_arr, (n,)).copy()
    if elev_arr.size == 1 and n > 1:
        elev_arr = np.broadcast_to(elev_arr, (n,)).copy()

    if not (len(dts) == lat_arr.size == lon_arr.size == elev_arr.size):
        raise HyPlanValueError(
            "sunpos: dt, latitude, longitude, elevation must broadcast to a "
            "common length"
        )

    t = ts.from_datetimes(dts)

    from skyfield.api import wgs84
    observer = earth + wgs84.latlon(
        latitude_degrees=lat_arr,
        longitude_degrees=lon_arr,
        elevation_m=elev_arr,
    )
    apparent = observer.at(t).observe(sun).apparent()
    alt, az, _dist = apparent.altaz()

    if radians:
        azimuth = az.radians
        zenith = (np.pi / 2.0) - alt.radians
    else:
        azimuth = az.degrees
        zenith = 90.0 - alt.degrees

    nan_pad = np.full_like(azimuth, np.nan, dtype=float)
    return azimuth, zenith, nan_pad, nan_pad, nan_pad



def solar_threshold_times(
    latitude: float,
    longitude: float,
    start_date: str,
    end_date: str,
    thresholds: List[float],
    timezone_offset: int = 0,
    timezone: Optional[str] = None,
) -> pd.DataFrame:
    """
    Find times when the solar elevation crosses specified thresholds.

    Args:
        latitude (float): Latitude of the location.
        longitude (float): Longitude of the location.
        start_date (str): Start date in 'YYYY-MM-DD' format.
        end_date (str): End date in 'YYYY-MM-DD' format.
        thresholds (list): List of 1 or 2 solar elevation thresholds in degrees (e.g., [35] or [35, 50]).
        timezone_offset (int): Fixed timezone offset from UTC in hours
            (e.g., -8 for PST, 1 for CET). Ignored if ``timezone`` is given.
        timezone (str, optional): IANA timezone name (e.g.
            ``"America/Los_Angeles"``). When supplied, takes precedence over
            ``timezone_offset`` and is DST-aware — recommended for any date
            range that may cross a DST transition. Pair with
            :func:`hyplan.geometry.get_timezone` to look up the zone from
            a lat/lon.

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

    # Convert to local time. Prefer the IANA timezone (DST-aware) when given;
    # fall back to the legacy fixed-offset behavior otherwise.
    if timezone is not None:
        local_timestamps = timestamps.tz_convert(timezone)
    else:
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


def solar_position_increments(
    latitude: float,
    longitude: float,
    date: Union[str, date, datetime],
    min_elevation: float,
    timezone_offset: int = 0,
    increment: str = '10min',
    timezone: Optional[str] = None,
) -> pd.DataFrame:
    """
    Return the solar azimuth and solar elevation at user-specified increments for a given date and location,
    but only for times when the solar elevation exceeds the specified minimum.

    Args:
        latitude (float): Latitude of the location.
        longitude (float): Longitude of the location.
        date (str or datetime.date): Date in 'YYYY-MM-DD' format or as a date object.
        min_elevation (float): Minimum solar elevation (in degrees) required to include the time.
        timezone_offset (int, optional): Fixed timezone offset from UTC in
            hours (e.g., -8 for PST). Ignored if ``timezone`` is given.
            Default is 0.
        increment (str, optional): Frequency increment for sampling times (e.g., '10min'). Default is '10min'.
        timezone (str, optional): IANA timezone name (e.g.
            ``"America/Los_Angeles"``). When supplied, takes precedence over
            ``timezone_offset`` and is DST-aware. Pair with
            :func:`hyplan.geometry.get_timezone` to look up the zone from
            a lat/lon.

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
    
    # Convert to local time. Prefer the IANA timezone (DST-aware) when given;
    # fall back to the legacy fixed-offset behavior otherwise.
    if timezone is not None:
        local_timestamps = timestamps_utc.tz_convert(timezone)
    else:
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

