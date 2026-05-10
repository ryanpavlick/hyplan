"""Vegetation phenology analysis for airborne remote sensing campaign planning.

Provides historical NDVI/EVI seasonality, LAI/FPAR, and phenological
transition dates from MODIS products via NASA EarthData.  Requires
``pip install hyplan[phenology]`` and NASA Earthdata credentials.

Supported products:

- **NDVI/EVI** (MOD13A1/MYD13A1): 16-day composites at 500 m.
- **LAI/FPAR** (MOD15A2H): 8-day composites at 500 m.
- **Phenology transitions** (MCD12Q2): annual greenup, peak,
  senescence, and dormancy dates at 500 m.
"""

from .analysis import (
    extract_phenology_stages,
    summarize_phenology_by_doy,
)
from .plotting import (
    plot_cloud_phenology_combined,
    plot_phenology_calendar,
    plot_seasonal_profile,
    plot_year_over_year_heatmap,
)
from .sources import (
    fetch_phenology,
    fetch_phenology_spatial,
)

__all__ = [
    "extract_phenology_stages",
    "fetch_phenology",
    "fetch_phenology_spatial",
    "plot_cloud_phenology_combined",
    "plot_phenology_calendar",
    "plot_seasonal_profile",
    "plot_year_over_year_heatmap",
    "summarize_phenology_by_doy",
]
