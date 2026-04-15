"""Cloud fraction analysis: DOY summaries, visit simulation, scheduling."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Dict, Tuple

import pandas as pd

from ..exceptions import HyPlanValueError

logger = logging.getLogger(__name__)


def summarize_cloud_fraction_by_doy(
    df: pd.DataFrame,
    window: int | None = None,
) -> pd.DataFrame:
    """Compute a "typical year" cloud fraction summary per polygon.

    Averages cloud fraction across all years for each
    ``(polygon_id, day_of_year)`` pair, producing a single seasonal profile
    per polygon.

    Args:
        df: Cloud fraction DataFrame with columns ``polygon_id``, ``year``,
            ``day_of_year``, ``cloud_fraction`` (as returned by
            :func:`~hyplan.clouds.fetch_cloud_fraction`).
        window: Optional rolling-mean window size (centered) applied to the
            per-polygon DOY mean.  ``None`` disables smoothing.

    Returns:
        DataFrame with columns ``polygon_id``, ``day_of_year``,
        ``cloud_fraction_mean``, ``cloud_fraction_std``,
        ``cloud_fraction_count``.
    """
    required = {"polygon_id", "year", "day_of_year", "cloud_fraction"}
    if not required.issubset(df.columns):
        raise HyPlanValueError(
            f"Input DataFrame must contain columns: {required}. "
            f"Found: {set(df.columns)}"
        )

    if df.empty:
        return pd.DataFrame(
            columns=["polygon_id", "day_of_year",
                     "cloud_fraction_mean", "cloud_fraction_std",
                     "cloud_fraction_count"]
        )

    summary = (
        df.groupby(["polygon_id", "day_of_year"])["cloud_fraction"]
        .agg(["mean", "std", "count"])
        .rename(columns={
            "mean": "cloud_fraction_mean",
            "std": "cloud_fraction_std",
            "count": "cloud_fraction_count",
        })
        .reset_index()
    )

    if window is not None:
        summary["cloud_fraction_mean"] = (
            summary.groupby("polygon_id")["cloud_fraction_mean"]
            .transform(lambda s: s.rolling(window, center=True, min_periods=1).mean())
        )

    return summary


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
    """Simulate daily flight scheduling based on cloud fraction thresholds.

    On each visitable day, the alphabetically first unvisited polygon that
    meets the cloud threshold is chosen.  Rest days count toward total_days
    but no polygon is visited.

    Args:
        df: Cloud fraction data with columns ``polygon_id``, ``year``,
            ``day_of_year``, ``cloud_fraction``.
        day_start: Start day-of-year for simulation.
        day_stop: End day-of-year for simulation.
        year_start: Start year.
        year_stop: End year.
        cloud_fraction_threshold: Maximum allowable cloud fraction.
        rest_day_threshold: Max consecutive visits before a rest day.
        exclude_weekends: Skip weekends and reset counter.
        debug: Enable detailed logging.

    Returns:
        Tuple of (summary_df, visit_tracker, rest_days) where summary_df
        has ``year`` and ``days`` columns, visit_tracker maps
        ``year -> polygon_id -> [day_of_year]``, and rest_days maps
        ``year -> [day_of_year]``.
    """
    if debug:
        logging.getLogger().setLevel(logging.DEBUG)

    crosses_year = day_start > day_stop

    visit_days = []
    visit_tracker: dict[int, dict] = {}
    rest_days: dict[int, list] = {}

    for year in range(year_start, year_stop + 1):
        visited_polygons: set[str] = set()
        remaining_polygons = set(df['polygon_id'].unique())
        visit_tracker[year] = {}
        rest_days[year] = []
        total_days = 0
        consecutive_visits = 0

        if crosses_year:
            last_day_of_year = (datetime(year + 1, 1, 1) - datetime(year, 1, 1)).days
            day_sequence = list(range(day_start, last_day_of_year + 1)) + list(range(1, day_stop + 1))
        else:
            day_sequence = list(range(day_start, day_stop + 1))

        for seq_idx, current_day_of_year in enumerate(day_sequence):
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
