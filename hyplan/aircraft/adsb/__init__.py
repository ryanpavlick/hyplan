"""ADS-B trajectory fitting for HyPlan aircraft models.

Fits speed schedules and vertical profiles from ADS-B surveillance data
using the ``traffic`` library for ingestion and HyPlan's own wind fields
for air-relative reconstruction.

Requires the ``[adsb]`` extra::

    pip install hyplan[adsb]

Usage::

    from hyplan.aircraft.adsb import fit_aircraft_from_adsb
    aircraft = fit_aircraft_from_adsb("flights.parquet", wind_source="merra2")
"""

from .airdata import reconstruct_airdata
from .fitting import fit_schedules
from .models import FitResult, FlightPhaseData, ScheduleFitMetrics
from .phases import label_phases
from .pipeline import fit_aircraft_from_adsb

# io.load_flights is not re-exported here because importing it triggers
# a traffic availability check.  Users who need it can import directly:
#   from hyplan.aircraft.adsb.io import load_flights

__all__ = [
    "fit_aircraft_from_adsb",
    "label_phases",
    "reconstruct_airdata",
    "fit_schedules",
    "FitResult",
    "FlightPhaseData",
    "ScheduleFitMetrics",
]
