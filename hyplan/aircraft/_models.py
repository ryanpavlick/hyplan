"""Concrete aircraft definitions for HyPlan.

Each class is a thin :class:`~hyplan.aircraft.Aircraft` subclass whose
performance is loaded from a JSON profile at
``hyplan/data/aircraft/<short_name>.json`` via
:func:`~hyplan.aircraft._profile_io.load_aircraft_profile`.  Editing
one of those JSON files updates the corresponding class without
touching Python.

For backwards compatibility every class defined here is re-exported
from :mod:`hyplan.aircraft`, so existing code that does
``from hyplan.aircraft import NASA_ER2`` continues to work.
"""

from __future__ import annotations

from ._base import Aircraft
from ._profile_io import load_aircraft_profile

__all__ = [
    "AWI_BaslerBT67",
    "BAS_TwinOtter",
    "DLR_HALO",
    "FAAM_BAe146",
    "KingAir350",
    "KingAirA90",
    "KingAirB200",
    "NASA_B777",
    "NASA_C130",
    "NASA_C20A",
    "NASA_ER2",
    "NASA_GIII",
    "NASA_GIV",
    "NASA_GV",
    "NASA_P3",
    "NASA_WB57",
    "NCAR_GV",
    "NERC_DO228",
    "NOAA_GIV",
    "NOAA_TwinOtter",
    "NOAA_WP3D",
    "SAFIRE_ATR42",
]

# ---------------------------------------------------------------------------
# NASA high-altitude research aircraft
# ---------------------------------------------------------------------------

class NASA_ER2(Aircraft):
    """NASA ER-2 high-altitude research aircraft.

    Operates at 70,000 ft, acquiring data above 95% of the Earth's
    atmosphere.  Based at NASA Armstrong Flight Research Center (AFRC).

    Speed schedules, vertical-rate profile, and approach behavior calibrated
    from cached NASA AFRC IWG1 in-situ flight logs covering NASA 806 and
    NASA 809, with continuous fiscal-year coverage 2012-2026 (618 sorties
    loaded from a 629-file cache pulled from the public NASA ASP archive
    plus a local in-house delivery; see
    ``notebooks/calibration/NASA_ER2/_fetch_asp.py`` for the fetcher).
    See [notebooks/calibration/NASA_ER2/calibration.ipynb] for the full
    derivation: per-altitude-bin |VS| medians, breakpoint selection rules,
    and validation against per-sortie observed timing.

    Vertical-rate highlights from the calibration:

    * Weight-management level-offs and holds during climb-out are modeled
      explicitly via ``typical_climb_out``; the ``climb_profile`` itself
      is active-climb-only performance, not wall-clock climb-out timing.
    * Two-regime descent: peak idle-power |VS| ~3675 fpm at top-of-
      descent, decaying to ~840 fpm at top-of-approach as the aircraft
      configures for the terminal pattern.
    * Empirical 2.5° glideslope on the terminal approach (shallower
      than standard 3° ILS — ER-2's approach geometry as flown across
      the IWG1 sortie set; touchdown estimate uses 6 sorties with
      fixes ≤ 50 ft AGL after ground-taxi trim).

    See also:
        `https://airbornescience.nasa.gov/aircraft/ER-2_-_AFRC <https://airbornescience.nasa.gov/aircraft/ER-2_-_AFRC>`_
    """

    def __init__(self) -> None:
        super().__init__(**load_aircraft_profile("nasa_er2"))


# ---------------------------------------------------------------------------
# Gulfstream business jets (NASA)
# ---------------------------------------------------------------------------

class NASA_GIII(Aircraft):
    """NASA Gulfstream III (NASA 520) research aircraft.

    Operated by NASA Langley Research Center (LaRC).

    See also:
        `https://airbornescience.nasa.gov/aircraft/Gulfstream_III_-_LaRC <https://airbornescience.nasa.gov/aircraft/Gulfstream_III_-_LaRC>`_
    """

    def __init__(self) -> None:
        super().__init__(**load_aircraft_profile("nasa_giii"))


class NASA_GIV(Aircraft):
    """NASA Gulfstream IV (NASA 817) research aircraft.

    Twin turbofan operated by NASA Armstrong Flight Research Center (AFRC).

    .. warning::

        **Uncalibrated.**  Performance values come from manufacturer
        brochures / type-certificate data; no in-situ flight-data fit
        has been performed.  Treat planning output as a best-effort
        starting point.

    See also:
        `https://airbornescience.nasa.gov/aircraft/Gulfstream_IV_-_AFRC <https://airbornescience.nasa.gov/aircraft/Gulfstream_IV_-_AFRC>`_
    """

    def __init__(self) -> None:
        super().__init__(**load_aircraft_profile("nasa_giv"))


class NASA_GV(Aircraft):
    """NASA Gulfstream V research aircraft.

    Operated by NASA Armstrong Flight Research Center (AFRC).
    Service ceiling 51,000 ft, cruise speed 500 kt (Mach 0.80).
    Currently undergoing modifications expected to conclude ~August 2026.

    See also:
        `https://airbornescience.nasa.gov/aircraft/Gulfstream_V_-_AFRC <https://airbornescience.nasa.gov/aircraft/Gulfstream_V_-_AFRC>`_
    """

    def __init__(self) -> None:
        super().__init__(**load_aircraft_profile("nasa_gv"))


class NCAR_GV(Aircraft):
    """NSF/NCAR HIAPER Gulfstream V research aircraft (N677F).

    Operated by NSF NCAR Earth Observing Laboratory (EOL).  Same
    airframe family as NASA_GV but a separate operational tail with
    different mission profile and avionics.  HIAPER routinely cruises
    FL410-FL510 on long-duration atmospheric campaigns (HIPPO,
    SOCRATES, ORCAS, ATTREX) and carries a different flight-data
    suite (high-rate 25-Hz NetCDF).

    See also:
        `https://www.eol.ucar.edu/observing_facilities/hiaper`
    """

    def __init__(self) -> None:
        super().__init__(**load_aircraft_profile("ncar_gv"))


class NASA_C20A(Aircraft):
    """NASA C-20A (Gulfstream III variant, NASA 502) research aircraft.

    Obtained from the U.S. Air Force in 2003. Primary platform for
    UAVSAR missions. Operated by NASA AFRC.

    .. note::

        **Inferred.**  Performance is mirrored from the calibrated
        ``NASA_GIII`` model (same type certificate).  C-20A-specific
        IWG1 calibration is deferred pending data access.  Output
        is more reliable than a brochure-only model but may not
        capture C-20A-specific operational differences.

    See also:
        `https://airbornescience.nasa.gov/aircraft/Gulfstream_C-20A_GIII_-_AFRC <https://airbornescience.nasa.gov/aircraft/Gulfstream_C-20A_GIII_-_AFRC>`_
    """

    def __init__(self) -> None:
        super().__init__(**load_aircraft_profile("nasa_c20a"))


# ---------------------------------------------------------------------------
# Heavy turboprops and large jets (NASA / NOAA)
# ---------------------------------------------------------------------------

class NASA_P3(Aircraft):
    """NASA P-3 Orion (NASA 426) airborne science laboratory.

    Four-engine turboprop capable of long-duration flights (8–14 hours)
    and large payloads up to 18,000 lbs. Operated by NASA Wallops Flight
    Facility (WFF).

    See also:
        `https://airbornescience.nasa.gov/aircraft/P-3_Orion <https://airbornescience.nasa.gov/aircraft/P-3_Orion>`_
    """

    def __init__(self) -> None:
        super().__init__(**load_aircraft_profile("nasa_p3"))


class NOAA_WP3D(Aircraft):
    """NOAA WP-3D Orion atmospheric chemistry / Hurricane Hunter.

    Lockheed WP-3D Orion operated by the NOAA Aircraft Operations
    Center (AOC).  Two tails are typically active for atmospheric
    chemistry research: N42RF "Kermit" and N43RF "Miss Piggy".
    Same airframe family as ``NASA_P3`` (LaRC's NASA 426); calibrated
    separately because the operator, mission profile, and outfit
    differ.  NOAA WP-3D missions tracked here are NOAA CSL chemistry
    campaigns, not the hurricane-research deployments.

    See also:
        `https://www.omao.noaa.gov/aircraft/lockheed-wp-3d-orion`
    """

    def __init__(self) -> None:
        super().__init__(**load_aircraft_profile("noaa_wp3d"))


class NOAA_GIV(Aircraft):
    """NOAA Gulfstream IV-SP "Gonzo" (N49RF) hurricane synoptic-surveillance jet.

    Operated by NOAA Aircraft Operations Center.  Hurricane synoptic
    surveillance — high-altitude (FL420-FL450) sonde drops around
    developing tropical cyclones in the Atlantic and East Pacific
    basins.  Distinct mission profile and operator from NASA's
    brochure-only ``NASA_GIV``; this is the first calibrated G-IV
    variant in HyPlan.

    See also:
        `https://www.aoc.noaa.gov/aircraft-gulfstream-iv.html`
    """

    def __init__(self) -> None:
        super().__init__(**load_aircraft_profile("noaa_giv"))


class NASA_WB57(Aircraft):
    """NASA WB-57F (NASA 926 / 927) high-altitude research aircraft.

    Based at NASA Johnson Space Center (JSC), Ellington Field.
    Operates up to 60 000 ft with 8 800 lbs useful payload.

    Calibrated against 127 sorties combining in-house IWG1 deliveries
    (NASA 926 + 927, 2018-2026) with the ACCLIP 2022 deployment's
    MMS-1HZ ICARTT data from NASA LaRC ASDC (collection
    ``ACCLIP_MetNav_AircraftInSitu_WB57_Data``; covers Lait's GSFC
    flight-planner WB-57 tuning window).  See
    ``notebooks/calibration/NASA_WB57/`` for ``_fetch_acclip.py`` (the
    LaRC ASDC fetcher), ``calibrate.py`` (the IWG1 + ICARTT-aware
    fitter), and ``calibration.ipynb`` (per-bin diagnostics).

    See also:
        `https://airbornescience.nasa.gov/aircraft/WB-57_-_JSC <https://airbornescience.nasa.gov/aircraft/WB-57_-_JSC>`_
    """

    def __init__(self) -> None:
        super().__init__(**load_aircraft_profile("nasa_wb57"))


class NASA_B777(Aircraft):
    """NASA Boeing 777 long-range research aircraft.

    Operated by NASA Langley Research Center (LaRC). Very large payload
    capacity (75,000 lbs) and long endurance (18 hours).

    .. warning::

        **Uncalibrated.**  Performance values come from manufacturer
        brochures / type-certificate data; no in-situ flight-data fit
        has been performed.  Treat planning output as a best-effort
        starting point.

    See also:
        `https://airbornescience.nasa.gov/aircraft/B-777_-_LaRC <https://airbornescience.nasa.gov/aircraft/B-777_-_LaRC>`_
    """

    def __init__(self) -> None:
        super().__init__(**load_aircraft_profile("nasa_b777"))


# ---------------------------------------------------------------------------
# King Air twin-turboprops
# ---------------------------------------------------------------------------

class KingAirA90(Aircraft):
    """Beechcraft King Air A90 twin-turboprop aircraft.

    Calibrated against 428 ADS-B sorties from `airplanes.live` daily-
    trace archives (2026-04-09 → 2026-05-08; 25 active US 65-A90 /
    65-A90-1 civilian tails) after filtering skydive jump-run profiles
    out of the raw 643-sortie sample.  No public IWG1-grade A-90 data
    is currently available, so this calibration is the highest-quality
    A90 envelope publicly available.  See
    ``notebooks/calibration/KingAirA90/calibrate.py`` for the recipe
    and ``_fetch_airplanes_live.py`` for the trace fetcher.

    Confidence is moderate (~0.65) — TAS is approximated by
    groundspeed (still-air baseline), and the active US A90 fleet
    skews toward skydive / charter / freight operations rather than
    research missions.  v1.6.1 will switch to MERRA-2 wind-triangle
    TAS reconstruction.

    See also:
        `https://airbornescience.nasa.gov/aircraft/Beechcraft_King_Air_A90 <https://airbornescience.nasa.gov/aircraft/Beechcraft_King_Air_A90>`_
    """

    def __init__(self) -> None:
        super().__init__(**load_aircraft_profile("king_air_a90"))


class KingAirB200(Aircraft):
    """Beechcraft King Air B200 twin-turboprop aircraft.

    See also:
        `https://airbornescience.nasa.gov/aircraft/Beechcraft_King_Air_A200 <https://airbornescience.nasa.gov/aircraft/Beechcraft_King_Air_A200>`_
    """

    def __init__(self) -> None:
        super().__init__(**load_aircraft_profile("king_air_b200"))


class KingAir350(Aircraft):
    """Beechcraft King Air 350 / 350i twin-turboprop research aircraft.

    Stretched 200-series airframe with PT6A-60A engines (or
    Blackhawk XP-67A upgrade with PT6A-67A).  Larger cabin, higher
    MTOW (15,000 lb), longer range, and higher service ceiling
    than the King Air 200.

    Notable research operators include the University of Wyoming
    (UWKA-2 / N2UW since 2024, replacing UW's earlier King Air
    200T), NCAR, and several university-operated tails.

    Calibrated against 22 ADS-B sorties from the University of
    Wyoming UWKA-2 (N2UW / hex A18F28) pulled from the
    `airplanes.live` globe-history archive.  18 active days
    clustered into 5 science-campaign windows in 2025-01, 2025-03,
    2025-06, 2025-07/08, 2025-10, and 2026-04 (10k trace rows
    total).  See
    ``notebooks/calibration/KingAir350/calibrate.py`` for the
    fitting recipe.

    Confidence is moderate-low (~0.5) — sample is single-tail and
    sparse, with science-mission profile bias (high-altitude
    measurement legs rather than long-range cruise).  UWKA-2 has
    the Blackhawk XP-67A engine upgrade (PT6A-67A), so calibrated
    values may run a few percent above stock B300 with PT6A-60A.
    Approach speed is held to brochure 110 kt because the
    calibrated 162 kt reflects science-leg descent TAS rather than
    a normal Vref pattern speed.

    See also:
        `https://www.uwyo.edu/atsc/research-facilities/uwka/king-air.html`
    """

    def __init__(self) -> None:
        super().__init__(**load_aircraft_profile("king_air_350"))


# ---------------------------------------------------------------------------
# Other research / military aircraft
# ---------------------------------------------------------------------------

class NASA_C130(Aircraft):
    """NASA C-130H Hercules (NASA 436) four-engine turboprop research aircraft.

    Operated by NASA Wallops Flight Facility.  Calibration is specific
    to NASA 436's ACT-America mission profile; other C-130 operators
    (USAF, NCAR, NRL, FAA) fly different envelopes and would need
    their own calibration.

    See also:
        `https://airbornescience.nasa.gov/aircraft/C-130H_-_WFF <https://airbornescience.nasa.gov/aircraft/C-130H_-_WFF>`_
    """

    def __init__(self) -> None:
        super().__init__(**load_aircraft_profile("nasa_c130"))


class NOAA_TwinOtter(Aircraft):
    """NOAA DHC-6 Twin Otter (N48RF + N46RF) STOL twin-turboprop.

    Operated by the NOAA Chemical Sciences Laboratory and NOAA AOC
    for boundary-layer atmospheric chemistry, lidar, and aerosol
    sampling missions.  Calibration is specific to NOAA's mission
    profile (slow-cruise dwell-time over plumes, FL060-FL150 ops);
    CIRPAS / Kenn Borek / commercial Twin Otters fly different
    envelopes and would need their own calibration class.

    See also:
        `https://www.omao.noaa.gov/aircraft/de-havilland-dhc-6-twin-otter`
    """

    def __init__(self) -> None:
        super().__init__(**load_aircraft_profile("noaa_twin_otter"))


class BAS_TwinOtter(Aircraft):
    """British Antarctic Survey DHC-6-300 Twin Otter (MASIN-equipped).

    BAS operates two Twin Otters (VP-FBL, VP-FBB) from Rothera Research
    Station for Antarctic atmospheric science and from Ny-Ålesund for
    Arctic missions.  Same airframe class as `NOAA_TwinOtter` but the
    polar operating profile (sustained low-altitude survey, cold-soak,
    high-wind ops, short-field gravel landings) is distinct enough to
    warrant a separate calibration.

    See also:
        `https://www.bas.ac.uk/team/operations-team/operational-delivery/airborne-science/`
    """

    def __init__(self) -> None:
        super().__init__(**load_aircraft_profile("bas_twin_otter"))


class FAAM_BAe146(Aircraft):
    """FAAM BAe-146-301 atmospheric research aircraft (G-LUXE).

    The Facility for Airborne Atmospheric Measurements operates G-LUXE
    on behalf of NERC and the UK Met Office.  Four-engine regional jet
    with a uniquely flexible mission envelope: low-altitude (200 ft AGL)
    boundary-layer surveys to FL350 troposphere/stratosphere transits.

    See also:
        `https://www.faam.ac.uk/`
    """

    def __init__(self) -> None:
        super().__init__(**load_aircraft_profile("faam_bae146"))


class SAFIRE_ATR42(Aircraft):
    """SAFIRE ATR-42-320 atmospheric research aircraft (F-HMTO).

    SAFIRE (Service des Avions Français Instrumentés pour la Recherche
    en Environnement) operates an ATR-42-320 turboprop for European
    atmospheric science campaigns.  Small, slow, payload-heavy — common
    European partner platform alongside FAAM and DLR HALO.

    See also:
        `https://www.safire.fr/`
    """

    def __init__(self) -> None:
        super().__init__(**load_aircraft_profile("safire_atr42"))


class NERC_DO228(Aircraft):
    """NERC ARSF Dornier Do228-101 atmospheric / remote-sensing aircraft.

    The NERC Airborne Research and Survey Facility operated D-CALM as a
    medium-tropospheric, non-pressurised twin-turboprop research platform.
    This model is calibrated from public CEDA NERC/ARSF DO228 datasets,
    principally ACTIVE and Eyjafjallajokull core aircraft measurements.

    The calibration is useful for timing / reachability but deliberately
    carries lower confidence than platforms with native TAS and attitude:
    the CEDA files provide position, altitude, U/V wind, pressure, and
    temperature, so TAS is reconstructed by wind triangle and turn/bank
    behaviour remains brochure/default.

    See also:
        `https://catalogue.ceda.ac.uk/uuid/d2c5c36981824b71a98a2906394d61f3/`
    """

    def __init__(self) -> None:
        super().__init__(**load_aircraft_profile("nerc_do228"))


class AWI_BaslerBT67(Aircraft):
    """AWI Polar 5 / Polar 6 Basler BT-67 polar research aircraft.

    The Alfred Wegener Institute operates two broadly similar Basler BT-67
    aircraft, Polar 5 and Polar 6, for Arctic and Antarctic airborne science.
    This model intentionally represents the combined AWI BT-67 operating
    envelope rather than one tail: the calibration uses public PANGAEA
    Polar 5 and Polar 6 nav/met records from ACLOUD 2017 and HALO-AC3 2022.

    The PANGAEA products include native TAS, ground speed, attitude, U/V/W
    wind, pressure, and temperature.  Vertical rate is derived from 1 Hz
    altitude, which is quantized to whole metres in these public files; the
    climb/descent profiles are therefore intentionally simple and lower
    confidence than the TAS schedules and turn envelope.

    See also:
        `https://www.awi.de/en/fleet-stations/aircraft/polar-5-6.html <https://www.awi.de/en/fleet-stations/aircraft/polar-5-6.html>`_
    """

    def __init__(self) -> None:
        super().__init__(**load_aircraft_profile("awi_basler_bt67"))


class DLR_HALO(Aircraft):
    """DLR HALO (D-ADLR) Gulfstream G550 high-altitude long-range research aircraft.

    HALO (High Altitude and LOng range) is operated by DLR
    Flugexperimente at Oberpfaffenhofen.  Same airframe family as
    `NCAR_GV` (HIAPER) — both Gulfstream V/G550 — but operated
    independently and typically configured for different payloads.

    See also:
        `https://www.dlr.de/en/research-and-transfer/research-infrastructure/halo`
    """

    def __init__(self) -> None:
        super().__init__(**load_aircraft_profile("dlr_halo"))
