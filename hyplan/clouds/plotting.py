"""Cloud fraction visualization: heatmaps, DOY profiles, spatial maps."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Dict

import geopandas as gpd
import matplotlib
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import pandas as pd

from ..exceptions import HyPlanValueError


def _heatmap_matrix(
    df: pd.DataFrame,
    *,
    ax: "plt.Axes | None" = None,
    cmap=None,
    norm=None,
    annot: bool = False,
    fmt: str = "",
    linewidths: float = 0.5,
    linecolor: str = "gray",
    square: bool = True,
    cbar: bool = True,
    cbar_label: "str | None" = None,
) -> "plt.Axes":
    """Render a 2-D ``pandas.DataFrame`` as a labelled heatmap on a matplotlib
    ``Axes``.

    Covers the subset of ``seaborn.heatmap`` features that HyPlan's cloud
    plotting needs (categorical labels from index/columns, optional
    annotations, cell-boundary gridlines, square cells, optional colorbar).
    Uses only matplotlib.
    """
    if ax is None:
        ax = plt.gca()

    mat = df.values
    nrows, ncols = mat.shape

    im = ax.imshow(
        mat,
        cmap=cmap,
        norm=norm,
        aspect="equal" if square else "auto",
        interpolation="nearest",
    )

    ax.set_xticks(range(ncols))
    ax.set_xticklabels(list(df.columns))
    ax.set_yticks(range(nrows))
    ax.set_yticklabels(list(df.index))

    if linewidths > 0:
        ax.set_xticks([x - 0.5 for x in range(1, ncols + 1)], minor=True)
        ax.set_yticks([y - 0.5 for y in range(1, nrows + 1)], minor=True)
        ax.grid(which="minor", color=linecolor, linewidth=linewidths)
        ax.tick_params(which="minor", length=0)

    if annot:
        for i in range(nrows):
            for j in range(ncols):
                v = mat[i, j]
                if pd.isna(v):
                    continue
                ax.text(j, i, format(v, fmt), ha="center", va="center", fontsize=8)

    if cbar:
        plt.colorbar(im, ax=ax, label=cbar_label)

    return ax


def plot_doy_cloud_fraction(
    summary_df: pd.DataFrame,
    ax: "plt.Axes | None" = None,
    show_std: bool = True,
    **kwargs,
) -> "plt.Axes":
    """Line plot of DOY cloud fraction for each polygon.

    Args:
        summary_df: Output of :func:`~hyplan.clouds.summarize_cloud_fraction_by_doy`.
        ax: Matplotlib Axes to plot on.  Created if ``None``.
        show_std: If ``True``, draw a shaded +/-1 std-dev band.
        **kwargs: Passed to ``ax.plot()``.

    Returns:
        The matplotlib Axes.
    """
    if ax is None:
        _, ax = plt.subplots()

    for name, grp in summary_df.groupby("polygon_id"):
        grp = grp.sort_values("day_of_year")
        ax.plot(grp["day_of_year"], grp["cloud_fraction_mean"],
                label=name, **kwargs)
        if show_std and "cloud_fraction_std" in grp.columns:
            ax.fill_between(
                grp["day_of_year"],
                grp["cloud_fraction_mean"] - grp["cloud_fraction_std"],
                grp["cloud_fraction_mean"] + grp["cloud_fraction_std"],
                alpha=0.2,
            )

    ax.set_xlabel("Day of Year")
    ax.set_ylabel("Cloud Fraction")
    ax.legend()
    return ax


def plot_cloud_fraction_spatial(
    spatial_data: "dict[str, object]",
    polygon_file: str | None = None,
    ncols: int = 2,
) -> "plt.Figure":
    """Plot per-pixel cloud fraction maps.

    Args:
        spatial_data: Dictionary mapping polygon name to ``xarray.DataArray``,
            as returned by :func:`~hyplan.clouds.fetch_cloud_fraction_spatial`.
        polygon_file: Optional path to polygon file for boundary overlay.
        ncols: Number of subplot columns.

    Returns:
        The matplotlib Figure.
    """
    import numpy as np

    n = len(spatial_data)
    if n == 0:
        raise HyPlanValueError("spatial_data is empty — nothing to plot.")

    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows),
                             squeeze=False)

    overlay_gdf = None
    if polygon_file is not None:
        overlay_gdf = gpd.read_file(polygon_file)

    for idx, (name, da) in enumerate(spatial_data.items()):
        ax = axes[idx // ncols, idx % ncols]
        im = ax.pcolormesh(  # type: ignore[attr-defined]
            da.coords["longitude"], da.coords["latitude"], da.values,  # type: ignore[attr-defined]
            cmap="viridis_r", vmin=0, vmax=1,
        )
        if overlay_gdf is not None:
            row = overlay_gdf[overlay_gdf["Name"] == name]
            if not row.empty:
                row.boundary.plot(ax=ax, edgecolor="red", linewidth=1.5)
        ax.set_title(name)  # type: ignore[attr-defined]
        ax.set_xlabel("Longitude")  # type: ignore[attr-defined]
        ax.set_ylabel("Latitude")  # type: ignore[attr-defined]
        fig.colorbar(im, ax=ax, label="Cloud Fraction")

    for idx in range(n, nrows * ncols):
        axes[idx // ncols, idx % ncols].set_visible(False)  # type: ignore[attr-defined]

    fig.tight_layout()
    return fig


def plot_cloud_forecast(
    forecast_df: pd.DataFrame,
    threshold: float = 0.25,
    ax: "plt.Axes | None" = None,
    cmap: str = "RdYlGn_r",
    annotate: bool = True,
    figsize: tuple[float, float] = (12, 4),
    title: str = "Cloud Cover Forecast",
) -> "plt.Axes":
    """Heatmap of cloud cover forecast with go/no-go threshold.

    Args:
        forecast_df: DataFrame with ``polygon_id``, ``date``,
            ``cloud_fraction`` columns (e.g. from
            :func:`~hyplan.clouds.fetch_cloud_forecast`).
        threshold: Cloud fraction threshold for go/no-go classification.
        ax: Matplotlib Axes to plot on.  Created if ``None``.
        cmap: Matplotlib colormap name.
        annotate: If ``True``, print percentage values in each cell.
        figsize: Figure size when *ax* is ``None``.
        title: Plot title.

    Returns:
        The matplotlib Axes.
    """
    import matplotlib.patches as mpatches

    required_columns = {"polygon_id", "date", "cloud_fraction"}
    if not required_columns.issubset(forecast_df.columns):
        raise HyPlanValueError(
            f"Input DataFrame must contain columns: {required_columns}"
        )

    pivot = forecast_df.pivot(
        index="polygon_id", columns="date", values="cloud_fraction"
    )
    pivot = pivot.sort_index()

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    norm = mcolors.TwoSlopeNorm(vcenter=threshold, vmin=0.0, vmax=1.0)

    _heatmap_matrix(
        pivot,
        ax=ax,
        cmap=cmap,
        norm=norm,
        annot=annotate,
        fmt=".0%" if annotate else "",
        linewidths=0.5,
        linecolor="gray",
        square=True,
        cbar=True,
        cbar_label="Cloud Fraction",
    )

    # Format x-tick labels as "Mon\nApr 17"
    date_labels = [
        pd.Timestamp(d).strftime("%a\n%b %d") for d in pivot.columns
    ]
    ax.set_xticklabels(date_labels, rotation=0, ha="center")

    # Draw green borders around "go" cells
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            if pivot.iloc[i, j] <= threshold:
                ax.add_patch(
                    mpatches.Rectangle(
                        (j, i), 1, 1,
                        fill=False, edgecolor="green", linewidth=2.5,
                    )
                )

    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title(title)
    return ax


def plot_yearly_cloud_fraction_heatmaps_with_visits(
    cloud_data_df: pd.DataFrame, visit_tracker: Dict[int, Dict[str, list]], rest_days: Dict[int, list],
    cloud_fraction_threshold: float = 0.10, exclude_weekends: bool = False,
    day_start: int = 1, day_stop: int = 365
) -> None:
    """Generate heatmaps of cloud fraction for each year with visit markers.

    Args:
        cloud_data_df: DataFrame with ``polygon_id``, ``year``,
            ``day_of_year``, ``cloud_fraction``.
        visit_tracker: Visit days per polygon per year.
        rest_days: Rest days per year.
        cloud_fraction_threshold: Threshold for clear/cloudy classification.
        exclude_weekends: Highlight and skip weekends.
        day_start: Start day-of-year.
        day_stop: End day-of-year.
    """
    required_columns = {'polygon_id', 'year', 'day_of_year', 'cloud_fraction'}
    if not required_columns.issubset(cloud_data_df.columns):
        raise HyPlanValueError(f"Input DataFrame must contain columns: {required_columns}")

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

        plt.figure(figsize=(16, 8))
        _heatmap_matrix(status_data, cmap=cmap, norm=norm, cbar=False,
                        linewidths=0.5, linecolor='gray', square=True)
        plt.scatter(stars_x, stars_y, color='red', marker='*', s=150, label='Visit Day')
        plt.title(f'Cloud Fraction Heatmap with Visits for Year {year}')
        plt.xlabel('Day of Year')
        plt.ylabel('Polygon ID')
        plt.legend(loc='upper right')
        plt.tight_layout()
        if matplotlib.get_backend().lower() != "agg":
            plt.show()
