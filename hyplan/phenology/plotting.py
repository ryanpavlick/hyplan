"""Phenology visualization: seasonal profiles, calendars, heatmaps."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd

from ..exceptions import HyPlanValueError
from .analysis import _STAGE_COLUMNS


# ---------------------------------------------------------------------------
# Stage display configuration
# ---------------------------------------------------------------------------

_STAGE_LABELS = {
    "greenup_doy": "Greenup",
    "midgreenup_doy": "Mid-Greenup",
    "peak_doy": "Peak",
    "maturity_doy": "Maturity",
    "midgreendown_doy": "Mid-Greendown",
    "senescence_doy": "Senescence",
    "dormancy_doy": "Dormancy",
}

_STAGE_COLORS = {
    "greenup_doy": "#a8d08d",
    "midgreenup_doy": "#70ad47",
    "peak_doy": "#2e7d32",
    "maturity_doy": "#4caf50",
    "midgreendown_doy": "#c5e1a5",
    "senescence_doy": "#ffcc02",
    "dormancy_doy": "#bf8f00",
}

# Month tick positions (day-of-year for first of each month in non-leap year)
_MONTH_STARTS = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]
_MONTH_LABELS = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
]


def plot_seasonal_profile(
    summary_df: pd.DataFrame,
    ax: "plt.Axes | None" = None,
    show_std: bool = True,
    ylabel: str = "NDVI",
    **kwargs,
) -> "plt.Axes":
    """Line plot of mean vegetation index by DOY, one line per polygon.

    Parameters
    ----------
    summary_df : pd.DataFrame
        Output of :func:`~hyplan.phenology.summarize_phenology_by_doy`.
    ax : plt.Axes or None
        Matplotlib Axes to plot on.  Created if ``None``.
    show_std : bool
        If ``True``, draw a shaded +/-1 std-dev band.
    ylabel : str
        Y-axis label (e.g. ``"NDVI"``, ``"EVI"``, ``"LAI"``).
    \\*\\*kwargs
        Passed to ``ax.plot()``.

    Returns
    -------
    plt.Axes
    """
    if ax is None:
        _, ax = plt.subplots()

    for name, grp in summary_df.groupby("polygon_id"):
        grp = grp.sort_values("day_of_year")
        ax.plot(grp["day_of_year"], grp["value_mean"], label=name, **kwargs)
        if show_std and "value_std" in grp.columns:
            ax.fill_between(
                grp["day_of_year"],
                grp["value_mean"] - grp["value_std"],
                grp["value_mean"] + grp["value_std"],
                alpha=0.2,
            )

    ax.set_xlabel("Day of Year")
    ax.set_ylabel(ylabel)
    ax.set_xlim(1, 365)
    ax.legend()
    return ax


def plot_phenology_calendar(
    stages_df: pd.DataFrame,
    ax: "plt.Axes | None" = None,
) -> "plt.Axes":
    """Gantt-style chart of phenological stages across months.

    Each polygon gets a horizontal row with colored bars spanning
    from one transition to the next.

    Parameters
    ----------
    stages_df : pd.DataFrame
        Output of :func:`~hyplan.phenology.extract_phenology_stages`.
    ax : plt.Axes or None
        Matplotlib Axes to plot on.  Created if ``None``.

    Returns
    -------
    plt.Axes
    """
    required_cols = {"polygon_id"}
    for stage in _STAGE_COLUMNS:
        required_cols.add(f"{stage}_mean")
    if not required_cols.issubset(stages_df.columns):
        raise HyPlanValueError(
            f"Input DataFrame must contain columns: {required_cols}. "
            f"Found: {set(stages_df.columns)}"
        )

    if ax is None:
        n_polygons = len(stages_df)
        fig_height = max(3, 0.8 * n_polygons + 1)
        _, ax = plt.subplots(figsize=(12, fig_height))

    polygons = stages_df["polygon_id"].unique()
    y_positions = {name: i for i, name in enumerate(polygons)}

    bar_height = 0.6

    # Draw bars between consecutive stages
    stage_pairs = list(zip(_STAGE_COLUMNS[:-1], _STAGE_COLUMNS[1:]))

    for _, row in stages_df.iterrows():
        y = y_positions[row["polygon_id"]]

        for stage_start, stage_end in stage_pairs:
            start_doy = row[f"{stage_start}_mean"]
            end_doy = row[f"{stage_end}_mean"]

            if np.isnan(start_doy) or np.isnan(end_doy):
                continue

            width = end_doy - start_doy
            if width <= 0:
                continue

            color = _STAGE_COLORS[stage_start]
            ax.barh(
                y, width, left=start_doy, height=bar_height,
                color=color, edgecolor="white", linewidth=0.5,
            )

            # Show std as error bar if available
            std_col = f"{stage_start}_std"
            if std_col in stages_df.columns:
                std_val = row[std_col]
                if not np.isnan(std_val):
                    ax.errorbar(
                        start_doy, y, xerr=std_val,
                        fmt="none", ecolor="black", elinewidth=0.8,
                        capsize=3, capthick=0.8,
                    )

    ax.set_yticks(list(y_positions.values()))
    ax.set_yticklabels(list(y_positions.keys()))
    ax.set_xlim(1, 365)
    ax.set_xticks(_MONTH_STARTS)
    ax.set_xticklabels(_MONTH_LABELS)
    ax.set_xlabel("Month")
    ax.set_title("Phenological Stages")
    ax.invert_yaxis()

    # Legend
    legend_patches = [
        mpatches.Patch(color=_STAGE_COLORS[s], label=_STAGE_LABELS[s])
        for s in _STAGE_COLUMNS[:-1]  # no bar for dormancy (it's the end)
    ]
    ax.legend(
        handles=legend_patches, loc="upper right",
        fontsize="small", ncol=2,
    )

    return ax


def plot_year_over_year_heatmap(
    df: pd.DataFrame,
    polygon_id: str | None = None,
    ax: "plt.Axes | None" = None,
    cmap: str = "YlGn",
    vmin: float = 0.0,
    vmax: float = 1.0,
) -> "plt.Axes":
    """Heatmap of vegetation index by DOY (x) and year (y).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame from :func:`~hyplan.phenology.fetch_phenology`
        with columns ``polygon_id``, ``year``, ``day_of_year``,
        ``value``.
    polygon_id : str or None
        Which polygon to plot.  Required if multiple polygons exist.
        If ``None`` and only one polygon exists, uses that one.
    ax : plt.Axes or None
        Matplotlib Axes to plot on.  Created if ``None``.
    cmap : str
        Colormap name.  Default ``"YlGn"``.
    vmin, vmax : float
        Color scale bounds.

    Returns
    -------
    plt.Axes
    """
    polygons = df["polygon_id"].unique()
    if polygon_id is None:
        if len(polygons) == 1:
            polygon_id = polygons[0]
        else:
            raise HyPlanValueError(
                f"Multiple polygons found: {list(polygons)}. "
                f"Specify polygon_id."
            )

    subset = df[df["polygon_id"] == polygon_id]
    if subset.empty:
        raise HyPlanValueError(
            f"No data found for polygon_id={polygon_id!r}."
        )

    pivot = subset.pivot_table(
        index="year", columns="day_of_year", values="value",
        aggfunc="mean",
    )

    if ax is None:
        n_years = len(pivot)
        fig_height = max(3, 0.4 * n_years + 1)
        _, ax = plt.subplots(figsize=(14, fig_height))

    im = ax.pcolormesh(
        pivot.columns.values,
        pivot.index.values,
        pivot.values,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        shading="nearest",
    )

    ax.set_xlabel("Day of Year")
    ax.set_ylabel("Year")
    ax.set_title(f"Vegetation Index — {polygon_id}")
    ax.figure.colorbar(im, ax=ax, label="Value", pad=0.02)
    ax.invert_yaxis()

    return ax


def plot_cloud_phenology_combined(
    cloud_summary_df: pd.DataFrame,
    phenology_summary_df: pd.DataFrame,
    ax: "plt.Axes | None" = None,
    layout: str = "overlay",
) -> "plt.Figure | plt.Axes":
    """Combined cloud fraction and vegetation index seasonal plot.

    Parameters
    ----------
    cloud_summary_df : pd.DataFrame
        Output of :func:`~hyplan.clouds.summarize_cloud_fraction_by_doy`.
    phenology_summary_df : pd.DataFrame
        Output of :func:`~hyplan.phenology.summarize_phenology_by_doy`.
    ax : plt.Axes or None
        For ``"overlay"`` mode, the Axes to use (created if ``None``).
        Ignored for ``"side_by_side"``.
    layout : str
        ``"overlay"`` (twin y-axes) or ``"side_by_side"`` (two panels).

    Returns
    -------
    plt.Axes or plt.Figure
        ``plt.Axes`` for ``"overlay"``, ``plt.Figure`` for ``"side_by_side"``.
    """
    if layout not in ("overlay", "side_by_side"):
        raise HyPlanValueError(
            f"Unknown layout: {layout!r}. Use 'overlay' or 'side_by_side'."
        )

    if layout == "overlay":
        if ax is None:
            _, ax = plt.subplots()

        # Cloud fraction on left axis
        for name, grp in cloud_summary_df.groupby("polygon_id"):
            grp = grp.sort_values("day_of_year")
            ax.plot(
                grp["day_of_year"], grp["cloud_fraction_mean"],
                color="steelblue", alpha=0.7, label=f"{name} (cloud)",
            )
            if "cloud_fraction_std" in grp.columns:
                ax.fill_between(
                    grp["day_of_year"],
                    grp["cloud_fraction_mean"] - grp["cloud_fraction_std"],
                    grp["cloud_fraction_mean"] + grp["cloud_fraction_std"],
                    color="steelblue", alpha=0.1,
                )

        ax.set_xlabel("Day of Year")
        ax.set_ylabel("Cloud Fraction", color="steelblue")
        ax.tick_params(axis="y", labelcolor="steelblue")
        ax.set_xlim(1, 365)

        # Vegetation index on right axis
        ax2 = ax.twinx()
        for name, grp in phenology_summary_df.groupby("polygon_id"):
            grp = grp.sort_values("day_of_year")
            ax2.plot(
                grp["day_of_year"], grp["value_mean"],
                color="green", alpha=0.7, label=f"{name} (VI)",
            )
            if "value_std" in grp.columns:
                ax2.fill_between(
                    grp["day_of_year"],
                    grp["value_mean"] - grp["value_std"],
                    grp["value_mean"] + grp["value_std"],
                    color="green", alpha=0.1,
                )

        ax2.set_ylabel("Vegetation Index", color="green")
        ax2.tick_params(axis="y", labelcolor="green")

        # Combined legend
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right",
                  fontsize="small")

        return ax

    else:  # side_by_side
        fig, (ax_cloud, ax_vi) = plt.subplots(
            2, 1, figsize=(12, 8), sharex=True,
        )

        # Top panel: cloud fraction
        for name, grp in cloud_summary_df.groupby("polygon_id"):
            grp = grp.sort_values("day_of_year")
            ax_cloud.plot(
                grp["day_of_year"], grp["cloud_fraction_mean"],
                label=name,
            )
            if "cloud_fraction_std" in grp.columns:
                ax_cloud.fill_between(
                    grp["day_of_year"],
                    grp["cloud_fraction_mean"] - grp["cloud_fraction_std"],
                    grp["cloud_fraction_mean"] + grp["cloud_fraction_std"],
                    alpha=0.2,
                )
        ax_cloud.set_ylabel("Cloud Fraction")
        ax_cloud.set_xlim(1, 365)
        ax_cloud.legend(fontsize="small")
        ax_cloud.set_title("Cloud Fraction & Vegetation Index Seasonality")

        # Bottom panel: vegetation index
        for name, grp in phenology_summary_df.groupby("polygon_id"):
            grp = grp.sort_values("day_of_year")
            ax_vi.plot(
                grp["day_of_year"], grp["value_mean"],
                label=name,
            )
            if "value_std" in grp.columns:
                ax_vi.fill_between(
                    grp["day_of_year"],
                    grp["value_mean"] - grp["value_std"],
                    grp["value_mean"] + grp["value_std"],
                    alpha=0.2,
                )
        ax_vi.set_xlabel("Day of Year")
        ax_vi.set_ylabel("Vegetation Index")
        ax_vi.legend(fontsize="small")

        fig.tight_layout()
        return fig
