"""Phenology analysis: DOY summaries and phenological stage extraction."""

from __future__ import annotations

import pandas as pd

from ..exceptions import HyPlanValueError


def summarize_phenology_by_doy(
    df: pd.DataFrame,
    window: int | None = None,
) -> pd.DataFrame:
    """Compute a typical-year vegetation index profile per polygon.

    Averages VI values across all years for each
    ``(polygon_id, day_of_year)`` pair, producing a single seasonal
    profile per polygon.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns ``polygon_id``, ``year``,
        ``day_of_year``, ``value`` (as returned by
        :func:`~hyplan.phenology.fetch_phenology` with
        ``product="ndvi"``/``"evi"``/``"lai"``/``"fpar"``).
    window : int or None
        Optional rolling-mean window size (centered) applied to the
        per-polygon DOY mean.  ``None`` disables smoothing.

    Returns
    -------
    pd.DataFrame
        Columns: ``polygon_id``, ``day_of_year``, ``value_mean``,
        ``value_std``, ``value_count``.
    """
    required = {"polygon_id", "year", "day_of_year", "value"}
    if not required.issubset(df.columns):
        raise HyPlanValueError(
            f"Input DataFrame must contain columns: {required}. "
            f"Found: {set(df.columns)}"
        )

    if df.empty:
        return pd.DataFrame(
            columns=[
                "polygon_id", "day_of_year",
                "value_mean", "value_std", "value_count",
            ]
        )

    summary = (
        df.groupby(["polygon_id", "day_of_year"])["value"]
        .agg(["mean", "std", "count"])
        .rename(columns={
            "mean": "value_mean",
            "std": "value_std",
            "count": "value_count",
        })
        .reset_index()
    )

    if window is not None:
        summary["value_mean"] = (
            summary.groupby("polygon_id")["value_mean"]
            .transform(
                lambda s: s.rolling(window, center=True, min_periods=1).mean()
            )
        )

    return summary


_STAGE_COLUMNS = [
    "greenup_doy",
    "midgreenup_doy",
    "peak_doy",
    "maturity_doy",
    "midgreendown_doy",
    "senescence_doy",
    "dormancy_doy",
]


def extract_phenology_stages(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Compute mean and std of phenological transition DOYs per polygon.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame from :func:`~hyplan.phenology.fetch_phenology`
        with ``product="phenology"``.  Must contain columns
        ``polygon_id``, ``year``, and all stage DOY columns.

    Returns
    -------
    pd.DataFrame
        Columns: ``polygon_id`` plus ``{stage}_mean`` and
        ``{stage}_std`` for each transition stage.
    """
    required = {"polygon_id", "year"} | set(_STAGE_COLUMNS)
    if not required.issubset(df.columns):
        raise HyPlanValueError(
            f"Input DataFrame must contain columns: {required}. "
            f"Found: {set(df.columns)}"
        )

    if df.empty:
        out_cols = ["polygon_id"]
        for stage in _STAGE_COLUMNS:
            out_cols.extend([f"{stage}_mean", f"{stage}_std"])
        return pd.DataFrame(columns=out_cols)

    agg_dict = {}
    for stage in _STAGE_COLUMNS:
        agg_dict[f"{stage}_mean"] = (stage, "mean")
        agg_dict[f"{stage}_std"] = (stage, "std")

    return (
        df.groupby("polygon_id")
        .agg(**agg_dict)
        .reset_index()
    )
