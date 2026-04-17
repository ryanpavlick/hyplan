"""Cloud fraction analysis for airborne remote sensing campaign planning.

Provides historical cloud climatology from two sources:

- **Google Earth Engine** (``source="gee"``): MODIS Terra/Aqua surface
  reflectance at 1 km resolution over user-supplied polygons. Requires
  GEE authentication.  ``pip install hyplan[clouds]``.

- **Open-Meteo** (``source="openmeteo"``): ERA5 reanalysis at 0.25 deg
  resolution using each polygon's centroid. **No authentication required.**

Also provides short-range cloud cover forecasts (up to 16 days) via the
Open-Meteo Forecast API, visit simulation for campaign scheduling, and
visualization helpers.
"""

from .analysis import (  # noqa: F401
    simulate_visits,
    summarize_cloud_fraction_by_doy,
)
from .forecast import (  # noqa: F401
    OpenMeteoCloudForecast,
    fetch_cloud_forecast,
)
from .plotting import (  # noqa: F401
    plot_cloud_forecast,
    plot_cloud_fraction_spatial,
    plot_doy_cloud_fraction,
    plot_yearly_cloud_fraction_heatmaps_with_visits,
)
from .sources import (  # noqa: F401
    OpenMeteoCloudFraction,
    calculate_cloud_fraction,
    create_cloud_data_array_with_limit,
    create_date_ranges,
    fetch_cloud_fraction,
    fetch_cloud_fraction_spatial,
    get_binary_cloud,
)

__all__ = [
    "get_binary_cloud",
    "calculate_cloud_fraction",
    "create_date_ranges",
    "create_cloud_data_array_with_limit",
    "simulate_visits",
    "plot_yearly_cloud_fraction_heatmaps_with_visits",
    "OpenMeteoCloudFraction",
    "fetch_cloud_fraction",
    "summarize_cloud_fraction_by_doy",
    "plot_doy_cloud_fraction",
    "OpenMeteoCloudForecast",
    "fetch_cloud_forecast",
    "fetch_cloud_fraction_spatial",
    "plot_cloud_forecast",
    "plot_cloud_fraction_spatial",
]
