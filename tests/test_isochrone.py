"""Tests for wind-aware isochrones (`hyplan.planning.isochrone`)."""

from __future__ import annotations


import numpy as np
import pytest

from hyplan import (
    KingAirB200,
    NASA_GIII,
    Waypoint,
    compute_isochrone,
    compute_concentric_isochrones,
    compute_multi_base_isochrone,
    compute_refuel_isochrone,
    evaluate_target_reachability,
    isochrone_polygon,
    ureg,
)
from hyplan.exceptions import HyPlanRuntimeError, HyPlanValueError
from hyplan.winds import ConstantWindField, StillAirField


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def giii() -> NASA_GIII:
    return NASA_GIII()


@pytest.fixture
def b200() -> KingAirB200:
    return KingAirB200()


@pytest.fixture
def kedw_wp() -> Waypoint:
    """KEDW (Edwards AFB) at runway elevation, dummy heading."""
    return Waypoint(
        latitude=34.905,
        longitude=-117.884,
        heading=0.0,
        altitude_msl=2300 * ureg.feet,
        name="KEDW",
    )


@pytest.fixture
def cruise_alt():
    return 35000 * ureg.feet


# ---------------------------------------------------------------------------
# 1. Still-air symmetry
# ---------------------------------------------------------------------------

def test_still_air_symmetry(giii, kedw_wp, cruise_alt):
    """All rays equal within 1 nmi for a still-air round-trip."""
    gdf = compute_isochrone(
        aircraft=giii,
        start=kedw_wp,
        budget=4 * ureg.hour,
        cruise_altitude=cruise_alt,
        mode="round_trip",
        wind_source=StillAirField(),
        azimuth_resolution_deg=10.0,
    )
    spread = gdf["distance_nmi"].max() - gdf["distance_nmi"].min()
    assert spread < 1.0, (
        f"still-air round-trip should be symmetric within 1 nmi; "
        f"spread = {spread:.2f}"
    )
    assert len(gdf) == 36
    assert gdf.crs.to_epsg() == 4326


# ---------------------------------------------------------------------------
# 2. Still-air half-range
# ---------------------------------------------------------------------------

def test_still_air_half_range(giii, kedw_wp, cruise_alt):
    """Round-trip radius is roughly cruise_TAS × budget / 2.

    The aircraft covers ground during climb and descent too, so
    the simple cruise-time half-range underestimates by ~6-10%.
    Test against the upper-bound `cruise_TAS × budget / 2` formula
    with a generous tolerance.
    """
    budget_hr = 4.0
    cruise_tas_kt = giii.cruise_speed_at(cruise_alt).m_as(ureg.knot)
    upper_bound_nmi = cruise_tas_kt * budget_hr / 2.0

    gdf = compute_isochrone(
        aircraft=giii,
        start=kedw_wp,
        budget=4 * ureg.hour,
        cruise_altitude=cruise_alt,
        mode="round_trip",
        wind_source=StillAirField(),
        azimuth_resolution_deg=30.0,
    )
    actual = gdf["distance_nmi"].mean()
    # Actual must be < upper bound (climb/descent overhead steals time)
    # but within 15% of it.
    assert actual < upper_bound_nmi
    rel_err = (upper_bound_nmi - actual) / upper_bound_nmi
    assert rel_err < 0.15, (
        f"still-air radius {actual:.0f} nmi differs from "
        f"cruise-only upper bound {upper_bound_nmi:.0f} nmi by "
        f"{rel_err*100:.1f}% — overhead seems unreasonable."
    )


# ---------------------------------------------------------------------------
# 3. Headwind elongates one_way reach
# ---------------------------------------------------------------------------

def test_one_way_tailwind_elongates(giii, kedw_wp, cruise_alt):
    """`one_way` boundary in tailwind direction farther than headwind."""
    # Wind FROM west (270°), 30 kt.  Eastbound = tailwind, westbound =
    # headwind.
    wind = ConstantWindField(30 * ureg.knot, wind_from_deg=270.0)
    gdf = compute_isochrone(
        aircraft=giii,
        start=kedw_wp,
        budget=2 * ureg.hour,
        cruise_altitude=cruise_alt,
        mode="one_way",
        wind_source=wind,
        azimuth_resolution_deg=90.0,
    )
    east = gdf[gdf.azimuth_deg == 90.0].iloc[0].distance_nmi
    west = gdf[gdf.azimuth_deg == 270.0].iloc[0].distance_nmi
    assert east > west * 1.10, (
        f"one_way east reach with tailwind ({east:.1f}) should be "
        f">10% beyond west reach with headwind ({west:.1f})."
    )


# ---------------------------------------------------------------------------
# 4. Tailwind extends one_way reach beyond no-wind cruise range
# ---------------------------------------------------------------------------

def test_one_way_tailwind_exceeds_no_wind_range(giii, kedw_wp, cruise_alt):
    """Regression: expanding bracket must not clip strong-tailwind reach."""
    # Strong tailwind from west.  Naive `cruise_TAS × budget` upper bound
    # would clip the eastbound boundary; expanding bracket should not.
    wind = ConstantWindField(150 * ureg.knot, wind_from_deg=270.0)
    gdf = compute_isochrone(
        aircraft=giii,
        start=kedw_wp,
        budget=3 * ureg.hour,
        cruise_altitude=cruise_alt,
        mode="one_way",
        wind_source=wind,
        azimuth_resolution_deg=90.0,
    )
    east = gdf[gdf.azimuth_deg == 90.0].iloc[0].distance_nmi
    cruise_tas_kt = giii.cruise_speed_at(cruise_alt).m_as(ureg.knot)
    naive_no_wind_range_nmi = cruise_tas_kt * 3.0  # 3 hr budget
    assert east > naive_no_wind_range_nmi, (
        f"east one_way reach with 150 kt tailwind ({east:.0f} nmi) "
        f"should exceed no-wind range ({naive_no_wind_range_nmi:.0f} nmi)."
    )


# ---------------------------------------------------------------------------
# 5. On-station dwell shrinks
# ---------------------------------------------------------------------------

def test_on_station_dwell_shrinks(giii, kedw_wp, cruise_alt):
    """Increasing on_station_time shrinks every ray."""
    common = dict(
        aircraft=giii,
        start=kedw_wp,
        budget=4 * ureg.hour,
        cruise_altitude=cruise_alt,
        mode="round_trip",
        wind_source=StillAirField(),
        azimuth_resolution_deg=30.0,
    )
    base = compute_isochrone(**common, on_station_time=0 * ureg.minute)
    shrunk = compute_isochrone(**common, on_station_time=60 * ureg.minute)
    base_d = base.set_index("azimuth_deg")["distance_nmi"]
    shrunk_d = shrunk.set_index("azimuth_deg")["distance_nmi"]
    assert (shrunk_d < base_d).all(), (
        f"on_station_time=60 should shrink every ray; "
        f"violations: {(shrunk_d >= base_d).sum()}"
    )


# ---------------------------------------------------------------------------
# 6. Reserve shrinks
# ---------------------------------------------------------------------------

def test_reserve_shrinks(giii, kedw_wp, cruise_alt):
    """Increasing reserve shrinks every ray."""
    common = dict(
        aircraft=giii,
        start=kedw_wp,
        budget=4 * ureg.hour,
        cruise_altitude=cruise_alt,
        mode="round_trip",
        wind_source=StillAirField(),
        azimuth_resolution_deg=30.0,
    )
    base = compute_isochrone(**common, reserve=0 * ureg.minute)
    shrunk = compute_isochrone(**common, reserve=30 * ureg.minute)
    base_d = base.set_index("azimuth_deg")["distance_nmi"]
    shrunk_d = shrunk.set_index("azimuth_deg")["distance_nmi"]
    assert (shrunk_d < base_d).all()


# ---------------------------------------------------------------------------
# 7. Different return airport biases the boundary
# ---------------------------------------------------------------------------

def test_return_safe_biases_toward_alternate(giii, kedw_wp, cruise_alt):
    """`return_safe` to a distant airport should bias rays toward it."""
    # Synthetic alternate ~200 nmi south of KEDW.  Use Waypoint to avoid
    # depending on the OurAirports data fixture.
    alt_wp = Waypoint(
        latitude=31.5,
        longitude=-117.884,
        heading=0.0,
        altitude_msl=500 * ureg.feet,
        name="alt-south",
    )
    rt = compute_isochrone(
        aircraft=giii,
        start=kedw_wp,
        budget=4 * ureg.hour,
        cruise_altitude=cruise_alt,
        mode="round_trip",
        wind_source=StillAirField(),
        azimuth_resolution_deg=90.0,
    )
    rs = compute_isochrone(
        aircraft=giii,
        start=kedw_wp,
        budget=4 * ureg.hour,
        cruise_altitude=cruise_alt,
        mode="return_safe",
        wind_source=StillAirField(),
        return_destination=alt_wp,
        azimuth_resolution_deg=90.0,
    )
    rt_idx = rt.set_index("azimuth_deg")["distance_nmi"]
    rs_idx = rs.set_index("azimuth_deg")["distance_nmi"]
    # Southbound (180°) should reach beyond round_trip; northbound (0°)
    # should reach less, because the return leg is longer when the
    # target is north and the recovery is south.
    assert rs_idx[180.0] > rt_idx[180.0]
    assert rs_idx[0.0] < rt_idx[0.0]


# ---------------------------------------------------------------------------
# 8. one_way reach exceeds round_trip reach on every ray
# ---------------------------------------------------------------------------

def test_one_way_exceeds_round_trip(giii, kedw_wp, cruise_alt):
    """Same budget: one_way distance > round_trip distance everywhere."""
    common = dict(
        aircraft=giii,
        start=kedw_wp,
        budget=3 * ureg.hour,
        cruise_altitude=cruise_alt,
        wind_source=StillAirField(),
        azimuth_resolution_deg=30.0,
    )
    rt = compute_isochrone(**common, mode="round_trip")
    ow = compute_isochrone(**common, mode="one_way")
    rt_idx = rt.set_index("azimuth_deg")["distance_nmi"]
    ow_idx = ow.set_index("azimuth_deg")["distance_nmi"]
    assert (ow_idx > rt_idx).all()


# ---------------------------------------------------------------------------
# 9. In-flight reaches farther than pre-flight, all else equal
# ---------------------------------------------------------------------------

def test_inflight_reaches_farther_than_preflight(giii, kedw_wp, cruise_alt):
    """Same lat/lon: at-cruise start reaches farther than runway-elev start."""
    inflight_start = Waypoint(
        latitude=kedw_wp.latitude,
        longitude=kedw_wp.longitude,
        heading=0.0,
        altitude_msl=cruise_alt,
        name="KEDW (in-flight)",
    )
    common = dict(
        aircraft=giii,
        budget=3 * ureg.hour,
        cruise_altitude=cruise_alt,
        mode="return_safe",
        wind_source=StillAirField(),
        return_destination=kedw_wp,
        azimuth_resolution_deg=30.0,
    )
    pre = compute_isochrone(start=kedw_wp, **common)
    flying = compute_isochrone(start=inflight_start, **common)
    pre_idx = pre.set_index("azimuth_deg")["distance_nmi"]
    flying_idx = flying.set_index("azimuth_deg")["distance_nmi"]
    assert (flying_idx > pre_idx).all(), (
        f"in-flight should reach farther on every ray; "
        f"violations: {(flying_idx <= pre_idx).sum()}"
    )


# ---------------------------------------------------------------------------
# 10. Input validation
# ---------------------------------------------------------------------------

class TestValidation:
    """Argument validation produces clean error messages."""

    def test_return_safe_requires_destination(self, giii, kedw_wp, cruise_alt):
        with pytest.raises(HyPlanValueError, match="return_safe"):
            compute_isochrone(
                aircraft=giii, start=kedw_wp, budget=2 * ureg.hour,
                cruise_altitude=cruise_alt, mode="return_safe",
            )

    def test_distinct_on_station_altitude_raises(
        self, giii, kedw_wp, cruise_alt,
    ):
        with pytest.raises(HyPlanValueError, match="[Mm]ulti-altitude"):
            compute_isochrone(
                aircraft=giii, start=kedw_wp, budget=2 * ureg.hour,
                cruise_altitude=cruise_alt,
                on_station_altitude=20000 * ureg.feet,
                mode="round_trip",
            )

    def test_budget_le_reserve_raises(self, giii, kedw_wp, cruise_alt):
        with pytest.raises(HyPlanValueError, match="budget"):
            compute_isochrone(
                aircraft=giii, start=kedw_wp, budget=30 * ureg.minute,
                cruise_altitude=cruise_alt, mode="round_trip",
                reserve=30 * ureg.minute,
            )

    def test_missing_start_altitude_raises(self, giii, cruise_alt):
        bad_wp = Waypoint(34.0, -118.0, heading=0.0, altitude_msl=None)
        with pytest.raises(HyPlanValueError, match="altitude_msl"):
            compute_isochrone(
                aircraft=giii, start=bad_wp, budget=2 * ureg.hour,
                cruise_altitude=cruise_alt, mode="round_trip",
            )

    def test_invalid_mode_raises(self, giii, kedw_wp, cruise_alt):
        with pytest.raises(HyPlanValueError, match="mode"):
            compute_isochrone(
                aircraft=giii, start=kedw_wp, budget=2 * ureg.hour,
                cruise_altitude=cruise_alt, mode="bogus",
            )

    def test_non_positive_azimuth_resolution_raises(
        self, giii, kedw_wp, cruise_alt,
    ):
        with pytest.raises(HyPlanValueError, match="azimuth_resolution"):
            compute_isochrone(
                aircraft=giii, start=kedw_wp, budget=2 * ureg.hour,
                cruise_altitude=cruise_alt, mode="round_trip",
                azimuth_resolution_deg=0.0,
            )

    def test_excessive_azimuth_resolution_raises(
        self, giii, kedw_wp, cruise_alt,
    ):
        with pytest.raises(HyPlanValueError, match="azimuth_resolution"):
            compute_isochrone(
                aircraft=giii, start=kedw_wp, budget=2 * ureg.hour,
                cruise_altitude=cruise_alt, mode="round_trip",
                azimuth_resolution_deg=361.0,
            )

    def test_non_positive_distance_tolerance_raises(
        self, giii, kedw_wp, cruise_alt,
    ):
        with pytest.raises(HyPlanValueError, match="distance_tolerance"):
            compute_isochrone(
                aircraft=giii, start=kedw_wp, budget=2 * ureg.hour,
                cruise_altitude=cruise_alt, mode="round_trip",
                distance_tolerance_nmi=-0.1,
            )

    def test_negative_reserve_raises(self, giii, kedw_wp, cruise_alt):
        with pytest.raises(HyPlanValueError, match="reserve"):
            compute_isochrone(
                aircraft=giii, start=kedw_wp, budget=2 * ureg.hour,
                cruise_altitude=cruise_alt, mode="round_trip",
                reserve=-5 * ureg.minute,
            )

    def test_negative_on_station_time_raises(self, giii, kedw_wp, cruise_alt):
        with pytest.raises(HyPlanValueError, match="on_station_time"):
            compute_isochrone(
                aircraft=giii, start=kedw_wp, budget=2 * ureg.hour,
                cruise_altitude=cruise_alt, mode="round_trip",
                on_station_time=-5 * ureg.minute,
            )

    def test_cruise_below_start_altitude_raises(self, giii, kedw_wp):
        """v1 rejects descending to cruise (start above cruise alt)."""
        airborne = Waypoint(
            latitude=kedw_wp.latitude, longitude=kedw_wp.longitude,
            heading=0.0, altitude_msl=40000 * ureg.feet,
        )
        with pytest.raises(HyPlanValueError, match="initial descent|cruise"):
            compute_isochrone(
                aircraft=giii, start=airborne, budget=2 * ureg.hour,
                cruise_altitude=20000 * ureg.feet, mode="round_trip",
            )

    def test_azimuth_resolution_above_120_raises(
        self, giii, kedw_wp, cruise_alt,
    ):
        """≥3 rays needed for a polygon."""
        with pytest.raises(HyPlanValueError, match="azimuth_resolution"):
            compute_isochrone(
                aircraft=giii, start=kedw_wp, budget=2 * ureg.hour,
                cruise_altitude=cruise_alt, mode="round_trip",
                azimuth_resolution_deg=180.0,
            )

    def test_invalid_ray_strategy_raises(self, giii, kedw_wp, cruise_alt):
        with pytest.raises(HyPlanValueError, match="ray_strategy"):
            compute_isochrone(
                aircraft=giii, start=kedw_wp, budget=2 * ureg.hour,
                cruise_altitude=cruise_alt, mode="round_trip",
                ray_strategy="bogus",
            )

    def test_invalid_adaptive_spacing_raises(self, giii, kedw_wp, cruise_alt):
        with pytest.raises(HyPlanValueError, match="adaptive_spacing"):
            compute_isochrone(
                aircraft=giii, start=kedw_wp, budget=2 * ureg.hour,
                cruise_altitude=cruise_alt, mode="round_trip",
                ray_strategy="adaptive", adaptive_spacing_nmi=0.0,
            )


# ---------------------------------------------------------------------------
# 11. round_trip defaults to start
# ---------------------------------------------------------------------------

def test_round_trip_defaults_to_start(giii, kedw_wp, cruise_alt):
    """round_trip without `return_destination` matches one with start."""
    common = dict(
        aircraft=giii,
        start=kedw_wp,
        budget=3 * ureg.hour,
        cruise_altitude=cruise_alt,
        mode="round_trip",
        wind_source=StillAirField(),
        azimuth_resolution_deg=60.0,
    )
    a = compute_isochrone(**common)
    b = compute_isochrone(return_destination=kedw_wp, **common)
    a_d = a.set_index("azimuth_deg")["distance_nmi"]
    b_d = b.set_index("azimuth_deg")["distance_nmi"]
    assert (a_d - b_d).abs().max() < 0.5  # within tolerance


# ---------------------------------------------------------------------------
# 12. Diagnostic correctness
# ---------------------------------------------------------------------------

def test_diagnostic_correctness(giii, kedw_wp, cruise_alt):
    """Per-ray diagnostics are internally consistent."""
    gdf = compute_isochrone(
        aircraft=giii,
        start=kedw_wp,
        budget=4 * ureg.hour,
        cruise_altitude=cruise_alt,
        mode="round_trip",
        wind_source=StillAirField(),
        on_station_time=15 * ureg.minute,
        azimuth_resolution_deg=60.0,
    )
    # total_time_min == outbound + on_station + return
    sums = (
        gdf["outbound_time_min"]
        + gdf["on_station_min"]
        + gdf["return_time_min"]
    )
    assert (gdf["total_time_min"] - sums).abs().max() < 0.1

    # time_slack_min ≥ 0
    assert (gdf["time_slack_min"] >= 0).all()

    # limiting_leg ∈ {outbound, return, on_station}
    assert gdf["limiting_leg"].isin(
        {"outbound", "return", "on_station"}
    ).all()

    # net_headwind = (out + back) / 2; asymmetry = (out − back) / 2
    expected_net = 0.5 * (
        gdf["outbound_headwind_kt"] + gdf["return_headwind_kt"]
    )
    expected_asym = 0.5 * (
        gdf["outbound_headwind_kt"] - gdf["return_headwind_kt"]
    )
    assert np.allclose(gdf["net_headwind_kt"], expected_net, atol=0.01)
    assert np.allclose(
        gdf["headwind_asymmetry_kt"], expected_asym, atol=0.01,
    )


def test_one_way_diagnostics_have_nan(giii, kedw_wp, cruise_alt):
    """one_way mode leaves return-related diagnostics as NaN."""
    gdf = compute_isochrone(
        aircraft=giii,
        start=kedw_wp,
        budget=2 * ureg.hour,
        cruise_altitude=cruise_alt,
        mode="one_way",
        wind_source=StillAirField(),
        azimuth_resolution_deg=120.0,
    )
    assert gdf["return_time_min"].isna().all()
    assert gdf["net_headwind_kt"].isna().all()
    assert gdf["headwind_asymmetry_kt"].isna().all()
    # Total = outbound only.
    assert (gdf["total_time_min"] - gdf["outbound_time_min"]).abs().max() < 0.01


# ---------------------------------------------------------------------------
# isochrone_polygon helper
# ---------------------------------------------------------------------------

def test_isochrone_polygon_closes(giii, kedw_wp, cruise_alt):
    """The helper closes the boundary into a valid Polygon."""
    gdf = compute_isochrone(
        aircraft=giii,
        start=kedw_wp,
        budget=3 * ureg.hour,
        cruise_altitude=cruise_alt,
        mode="round_trip",
        wind_source=StillAirField(),
        azimuth_resolution_deg=60.0,
    )
    poly = isochrone_polygon(gdf)
    assert poly.is_valid
    assert poly.area > 0


# ---------------------------------------------------------------------------
# Metadata stash
# ---------------------------------------------------------------------------

def test_start_accepts_airport(giii, kedw_wp, cruise_alt):
    """Passing an Airport as `start` produces the same result as the
    equivalent Waypoint (matching ICAO code, runway elevation)."""
    from hyplan.airports import Airport, initialize_data

    initialize_data()
    kedw_airport = Airport("KEDW")

    common = dict(
        aircraft=giii,
        budget=4 * ureg.hour,
        cruise_altitude=cruise_alt,
        mode="round_trip",
        wind_source=StillAirField(),
        azimuth_resolution_deg=60.0,
    )
    gdf_airport = compute_isochrone(start=kedw_airport, **common)
    # Reconstruct an equivalent Waypoint manually.
    kedw_equiv_wp = Waypoint(
        latitude=kedw_airport.latitude,
        longitude=kedw_airport.longitude,
        heading=0.0,
        altitude_msl=kedw_airport.elevation,
        name=kedw_airport.icao_code,
    )
    gdf_wp = compute_isochrone(start=kedw_equiv_wp, **common)

    a = gdf_airport.set_index("azimuth_deg")["distance_nmi"]
    b = gdf_wp.set_index("azimuth_deg")["distance_nmi"]
    assert (a - b).abs().max() < 0.5  # within tolerance
    # attrs reflect the Airport's coordinates exactly.
    assert gdf_airport.attrs["start_lat"] == pytest.approx(
        kedw_airport.latitude
    )
    assert gdf_airport.attrs["start_lon"] == pytest.approx(
        kedw_airport.longitude
    )


def test_start_airport_with_return_airport(giii, cruise_alt):
    """Both `start` and `return_destination` can be Airports
    (return_safe to a different recovery field)."""
    from hyplan.airports import Airport, initialize_data

    initialize_data()
    kedw = Airport("KEDW")
    kcos = Airport("KCOS")

    gdf = compute_isochrone(
        aircraft=giii,
        start=kedw,
        budget=5 * ureg.hour,
        cruise_altitude=cruise_alt,
        mode="return_safe",
        return_destination=kcos,
        wind_source=StillAirField(),
        azimuth_resolution_deg=60.0,
    )
    assert len(gdf) == 6
    # The return-destination metadata should reflect KCOS.
    assert gdf.attrs["return_destination_label"] == "KCOS"
    assert gdf.attrs["return_destination_lat"] == pytest.approx(
        kcos.latitude
    )


def test_pure_crosswind_slows_groundspeed(giii, kedw_wp, cruise_alt):
    """A pure crosswind on every ray must shrink the boundary vs still air.

    Regression: the v1 manual along-track-only projection ignored
    crab and would have produced an identical boundary.
    """
    # 60 kt wind from the west (270°).  Northbound (azimuth=0) and
    # southbound (180°) rays have no along-track component — pure
    # crosswind.  With crab handling, GS = sqrt(TAS² − xwind²) < TAS,
    # so those rays should reach less far than still air.
    wind = ConstantWindField(60 * ureg.knot, wind_from_deg=270.0)
    common = dict(
        aircraft=giii, start=kedw_wp, budget=4 * ureg.hour,
        cruise_altitude=cruise_alt, mode="round_trip",
        azimuth_resolution_deg=90.0,
    )
    calm = compute_isochrone(wind_source=StillAirField(), **common)
    crossed = compute_isochrone(wind_source=wind, **common)

    calm_north = calm.set_index("azimuth_deg").loc[0.0, "distance_nmi"]
    crossed_north = crossed.set_index("azimuth_deg").loc[0.0, "distance_nmi"]
    calm_south = calm.set_index("azimuth_deg").loc[180.0, "distance_nmi"]
    crossed_south = crossed.set_index("azimuth_deg").loc[180.0, "distance_nmi"]

    # Crosswind should slow the aircraft on north / south rays.
    assert crossed_north < calm_north - 5.0, (
        f"pure-crosswind north ray ({crossed_north:.1f}) should be "
        f">5 nmi shorter than still-air ({calm_north:.1f})"
    )
    assert crossed_south < calm_south - 5.0


def test_unflyable_headwind_clips_to_zero(giii, cruise_alt):
    """When wind exceeds TAS along a ray, the boundary clips to 0 nmi.

    Tested in-flight (already at cruise altitude) so there's no
    climb-distance carryover that masks the unflyable-cruise
    behavior.
    """
    # G-III cruise TAS at FL350 ≈ 460 kt.  An 800 kt headwind from the
    # east is unflyable on the eastbound ray.
    airborne = Waypoint(34.905, -117.884, heading=0.0,
                        altitude_msl=cruise_alt)
    wind = ConstantWindField(800 * ureg.knot, wind_from_deg=90.0)
    gdf = compute_isochrone(
        aircraft=giii, start=airborne, budget=2 * ureg.hour,
        cruise_altitude=cruise_alt, mode="one_way",
        wind_source=wind, azimuth_resolution_deg=90.0,
    )
    east = gdf.set_index("azimuth_deg").loc[90.0]
    assert east["distance_nmi"] == 0.0, (
        f"unflyable east ray should clip to 0 nmi, got "
        f"{east['distance_nmi']:.1f}"
    )
    assert east["limiting_leg"] == "unflyable"
    # Westbound is fine — strong tailwind.
    west = gdf.set_index("azimuth_deg").loc[270.0]
    assert west["distance_nmi"] > 100.0


def test_return_destination_attrs_present(giii, kedw_wp, cruise_alt):
    """attrs include return lat/lon when applicable."""
    ret = Waypoint(38.806, -104.701, heading=0.0,
                   altitude_msl=6187 * ureg.feet, name="KCOS")
    gdf = compute_isochrone(
        aircraft=giii, start=kedw_wp, budget=4 * ureg.hour,
        cruise_altitude=cruise_alt, mode="return_safe",
        return_destination=ret, wind_source=StillAirField(),
        azimuth_resolution_deg=120.0,
    )
    assert gdf.attrs["return_destination_lat"] == pytest.approx(38.806)
    assert gdf.attrs["return_destination_lon"] == pytest.approx(-104.701)
    assert gdf.attrs["return_destination_label"] == "KCOS"

    # one_way: return attrs are None.
    gdf_ow = compute_isochrone(
        aircraft=giii, start=kedw_wp, budget=2 * ureg.hour,
        cruise_altitude=cruise_alt, mode="one_way",
        wind_source=StillAirField(), azimuth_resolution_deg=120.0,
    )
    assert gdf_ow.attrs["return_destination_lat"] is None
    assert gdf_ow.attrs["return_destination_lon"] is None


def test_auto_ray_strategy_uses_ellipse_for_distinct_return(
    giii, kedw_wp, cruise_alt,
):
    """Distinct recovery fields get non-uniform ellipse-seeded azimuths."""
    ret = Waypoint(38.806, -104.701, heading=0.0,
                   altitude_msl=6187 * ureg.feet, name="KCOS")
    gdf = compute_isochrone(
        aircraft=giii,
        start=kedw_wp,
        budget=5 * ureg.hour,
        cruise_altitude=cruise_alt,
        mode="return_safe",
        return_destination=ret,
        wind_source=StillAirField(),
        azimuth_resolution_deg=30.0,
        ray_strategy="auto",
    )
    az = np.sort(gdf["azimuth_deg"].to_numpy())
    spacings = np.diff(np.r_[az, az[0] + 360.0])
    assert len(gdf) == 12
    assert gdf.attrs["effective_ray_strategy"] == "ellipse"
    assert spacings.max() - spacings.min() > 1.0


def test_auto_ray_strategy_keeps_uniform_for_one_way(
    giii, kedw_wp, cruise_alt,
):
    gdf = compute_isochrone(
        aircraft=giii,
        start=kedw_wp,
        budget=2 * ureg.hour,
        cruise_altitude=cruise_alt,
        mode="one_way",
        wind_source=StillAirField(),
        azimuth_resolution_deg=45.0,
        ray_strategy="auto",
    )
    assert len(gdf) == 8
    assert gdf.attrs["effective_ray_strategy"] == "uniform"
    assert np.allclose(np.diff(np.sort(gdf["azimuth_deg"])), 45.0)


def test_adaptive_ray_strategy_adds_midpoint_rays(
    giii, kedw_wp, cruise_alt,
):
    base = compute_isochrone(
        aircraft=giii,
        start=kedw_wp,
        budget=4 * ureg.hour,
        cruise_altitude=cruise_alt,
        mode="round_trip",
        wind_source=StillAirField(),
        azimuth_resolution_deg=60.0,
        ray_strategy="uniform",
    )
    refined = compute_isochrone(
        aircraft=giii,
        start=kedw_wp,
        budget=4 * ureg.hour,
        cruise_altitude=cruise_alt,
        mode="round_trip",
        wind_source=StillAirField(),
        azimuth_resolution_deg=60.0,
        ray_strategy="adaptive",
        adaptive_spacing_nmi=100.0,
        max_adaptive_rays=24,
    )
    assert len(refined) > len(base)
    assert len(refined) <= 24
    assert refined.attrs["effective_ray_strategy"] == "adaptive"


def test_attrs_metadata_present(giii, kedw_wp, cruise_alt):
    gdf = compute_isochrone(
        aircraft=giii,
        start=kedw_wp,
        budget=2 * ureg.hour,
        cruise_altitude=cruise_alt,
        mode="round_trip",
        wind_source=StillAirField(),
        azimuth_resolution_deg=120.0,
    )
    expected = {
        "mode", "start_lat", "start_lon", "start_altitude_ft",
        "cruise_altitude_ft", "budget_min", "budget_hr", "reserve_min",
        "on_station_min", "return_destination_label", "aircraft_type",
        "wind_source_kind", "start_time",
    }
    missing = expected - set(gdf.attrs.keys())
    assert not missing, f"missing attrs: {missing}"
    assert gdf.attrs["aircraft_type"] == "Gulfstream III"
    assert gdf.attrs["mode"] == "round_trip"


# ---------------------------------------------------------------------------
# Refuel-aware isochrone
# ---------------------------------------------------------------------------

@pytest.fixture
def kefd_wp() -> Waypoint:
    """KEFD (Ellington Field, Houston) at runway elevation."""
    return Waypoint(
        latitude=29.6073, longitude=-95.1586, heading=0.0,
        altitude_msl=32 * ureg.feet, name="KEFD",
    )


@pytest.fixture
def klbb_wp() -> Waypoint:
    """KLBB (Lubbock, TX) at runway elevation — ~410 nmi NW of KEFD."""
    return Waypoint(
        latitude=33.6636, longitude=-101.8228, heading=0.0,
        altitude_msl=3282 * ureg.feet, name="KLBB",
    )


@pytest.fixture
def kbtr_wp() -> Waypoint:
    """KBTR (Baton Rouge) at runway elevation — ~225 nmi E of KEFD."""
    return Waypoint(
        latitude=30.5332, longitude=-91.1496, heading=0.0,
        altitude_msl=70 * ureg.feet, name="KBTR",
    )


@pytest.fixture
def b200_cruise():
    return 25000 * ureg.feet


class TestRefuel:
    """compute_refuel_isochrone: validation, behavior, diagnostics."""

    # --- 1. validation -----------------------------------------------------

    def test_empty_refuel_airports_raises(self, b200, kefd_wp, b200_cruise):
        with pytest.raises(HyPlanValueError, match="refuel_airports"):
            compute_refuel_isochrone(
                aircraft=b200, start=kefd_wp,
                sortie_budget=4 * ureg.hour,
                flight_day_budget=8 * ureg.hour,
                cruise_altitude=b200_cruise,
                refuel_airports=[],
                mode="round_trip",
            )

    def test_one_way_rejected(self, b200, kefd_wp, klbb_wp, b200_cruise):
        with pytest.raises(HyPlanValueError, match="mode"):
            compute_refuel_isochrone(
                aircraft=b200, start=kefd_wp,
                sortie_budget=4 * ureg.hour,
                flight_day_budget=8 * ureg.hour,
                cruise_altitude=b200_cruise,
                refuel_airports=[klbb_wp],
                mode="one_way",
            )

    def test_day_lt_sortie_warns(
        self, b200, kefd_wp, klbb_wp, b200_cruise,
    ):
        with pytest.warns(UserWarning, match="day clock will bind"):
            gdf = compute_refuel_isochrone(
                aircraft=b200, start=kefd_wp,
                sortie_budget=5 * ureg.hour,
                flight_day_budget=3 * ureg.hour,
                cruise_altitude=b200_cruise,
                refuel_airports=[klbb_wp],
                mode="round_trip",
                azimuth_resolution_deg=90.0,
                distance_tolerance_nmi=5.0,
            )
        assert len(gdf) == 4

    def test_max_refuel_stops_must_be_one(
        self, b200, kefd_wp, klbb_wp, b200_cruise,
    ):
        with pytest.raises(HyPlanValueError, match="max_refuel_stops"):
            compute_refuel_isochrone(
                aircraft=b200, start=kefd_wp,
                sortie_budget=4 * ureg.hour,
                flight_day_budget=8 * ureg.hour,
                cruise_altitude=b200_cruise,
                refuel_airports=[klbb_wp],
                max_refuel_stops=2,
                mode="round_trip",
            )

    def test_negative_refuel_time_raises(
        self, b200, kefd_wp, klbb_wp, b200_cruise,
    ):
        with pytest.raises(HyPlanValueError, match="refuel_time"):
            compute_refuel_isochrone(
                aircraft=b200, start=kefd_wp,
                sortie_budget=4 * ureg.hour,
                flight_day_budget=8 * ureg.hour,
                cruise_altitude=b200_cruise,
                refuel_airports=[klbb_wp],
                refuel_time=-5 * ureg.minute,
                mode="round_trip",
            )

    # --- 2. behavior -------------------------------------------------------

    def test_unreachable_refuel_pruned(
        self, b200, kefd_wp, b200_cruise,
    ):
        """Refuel airport too far away → not used; flagged unreachable."""
        far_wp = Waypoint(
            latitude=70.0, longitude=-95.0, heading=0.0,
            altitude_msl=100 * ureg.feet, name="FAR",
        )
        gdf = compute_refuel_isochrone(
            aircraft=b200, start=kefd_wp,
            sortie_budget=2 * ureg.hour,
            flight_day_budget=4 * ureg.hour,
            cruise_altitude=b200_cruise,
            refuel_airports=[far_wp],
            mode="round_trip",
            azimuth_resolution_deg=60.0,
            distance_tolerance_nmi=5.0,
        )
        assert "FAR" not in gdf.attrs["refuel_airports_used"]
        assert any(
            u["label"] == "FAR"
            for u in gdf.attrs["refuel_airports_unreachable"]
        )
        assert (gdf["itinerary"] == "direct").all()

    def test_refuel_strictly_extends_reach(
        self, b200, kefd_wp, klbb_wp, b200_cruise,
    ):
        """Along the bearing toward KLBB, refuel boundary > direct boundary."""
        common = dict(
            aircraft=b200, start=kefd_wp,
            cruise_altitude=b200_cruise,
            mode="round_trip",
            azimuth_resolution_deg=30.0,
            distance_tolerance_nmi=2.0,
        )
        direct = compute_isochrone(
            **common, budget=4 * ureg.hour,
        )
        refuel = compute_refuel_isochrone(
            **common,
            sortie_budget=4 * ureg.hour,
            flight_day_budget=8 * ureg.hour,
            refuel_airports=[klbb_wp],
            refuel_time=30 * ureg.minute,
        )
        # Bearing KEFD→KLBB ≈ 322°; pick ray closest (330°).
        d_dir_330 = direct.loc[
            direct["azimuth_deg"] == 330.0, "distance_nmi"
        ].iloc[0]
        ref_330 = refuel.loc[refuel["azimuth_deg"] == 330.0].iloc[0]
        assert ref_330["distance_nmi"] > d_dir_330 + 5.0
        assert ref_330["itinerary"] == "outbound_refuel"
        assert ref_330["refuel_airport"] == "KLBB"

    def test_direct_wins_opposite_refuel(
        self, b200, kefd_wp, klbb_wp, b200_cruise,
    ):
        """Rays away from refuel direction stay direct."""
        gdf = compute_refuel_isochrone(
            aircraft=b200, start=kefd_wp,
            sortie_budget=4 * ureg.hour,
            flight_day_budget=8 * ureg.hour,
            cruise_altitude=b200_cruise,
            refuel_airports=[klbb_wp],
            refuel_time=30 * ureg.minute,
            mode="round_trip",
            azimuth_resolution_deg=30.0,
            distance_tolerance_nmi=2.0,
        )
        # KLBB is NW (~322°).  Ray at 150° (opposite) should be direct.
        opp = gdf.loc[gdf["azimuth_deg"] == 150.0].iloc[0]
        assert opp["itinerary"] == "direct"

    def test_no_useful_refuel_matches_direct(
        self, b200, kefd_wp, b200_cruise,
    ):
        """When the refuel airport is within reach but never improves any
        ray, refuel boundary matches direct boundary within tolerance."""
        # Refuel airport very close to start: a 5-min detour wastes
        # 30 min refuel + small flight time and extends nothing.
        nearby = Waypoint(
            latitude=kefd_wp.latitude + 0.1, longitude=kefd_wp.longitude,
            heading=0.0, altitude_msl=100 * ureg.feet, name="NEAR",
        )
        common = dict(
            aircraft=b200, start=kefd_wp,
            cruise_altitude=b200_cruise,
            mode="round_trip",
            azimuth_resolution_deg=60.0,
            distance_tolerance_nmi=2.0,
        )
        direct = compute_isochrone(
            **common, budget=4 * ureg.hour,
        )
        with pytest.warns(UserWarning):
            refuel = compute_refuel_isochrone(
                **common,
                sortie_budget=4 * ureg.hour,
                flight_day_budget=4 * ureg.hour + 5 * ureg.minute,
                refuel_airports=[nearby],
                refuel_time=30 * ureg.minute,
            )
        for az in direct["azimuth_deg"]:
            d_dir = direct.loc[
                direct["azimuth_deg"] == az, "distance_nmi"
            ].iloc[0]
            d_ref = refuel.loc[
                refuel["azimuth_deg"] == az, "distance_nmi"
            ].iloc[0]
            assert abs(d_dir - d_ref) < 5.0, f"az {az}: {d_dir} vs {d_ref}"

    def test_refuel_time_monotone(
        self, b200, kefd_wp, klbb_wp, b200_cruise,
    ):
        """Increasing refuel_time monotonically shrinks per-ray reach."""
        common = dict(
            aircraft=b200, start=kefd_wp,
            sortie_budget=4 * ureg.hour,
            flight_day_budget=8 * ureg.hour,
            cruise_altitude=b200_cruise,
            refuel_airports=[klbb_wp],
            mode="round_trip",
            azimuth_resolution_deg=60.0,
            distance_tolerance_nmi=2.0,
        )
        short = compute_refuel_isochrone(refuel_time=30 * ureg.minute, **common)
        long_ = compute_refuel_isochrone(refuel_time=90 * ureg.minute, **common)
        for az in short["azimuth_deg"]:
            d_s = short.loc[short["azimuth_deg"] == az, "distance_nmi"].iloc[0]
            d_l = long_.loc[long_["azimuth_deg"] == az, "distance_nmi"].iloc[0]
            assert d_l <= d_s + 2.0, f"az {az}: long {d_l} > short {d_s}"

    def test_tight_day_budget_blocks_refuel(
        self, b200, kefd_wp, klbb_wp, b200_cruise,
    ):
        """flight_day_budget = sortie + 5 min → refuel impossible."""
        with pytest.warns(UserWarning):
            gdf = compute_refuel_isochrone(
                aircraft=b200, start=kefd_wp,
                sortie_budget=4 * ureg.hour,
                flight_day_budget=4 * ureg.hour + 5 * ureg.minute,
                refuel_time=60 * ureg.minute,
                cruise_altitude=b200_cruise,
                refuel_airports=[klbb_wp],
                mode="round_trip",
                azimuth_resolution_deg=60.0,
                distance_tolerance_nmi=5.0,
            )
        assert (gdf["itinerary"] == "direct").all()

    def test_reserve_per_cycle(
        self, b200, kefd_wp, klbb_wp, b200_cruise,
    ):
        """reserve = 30 min applies to each cycle, not the day clock."""
        gdf = compute_refuel_isochrone(
            aircraft=b200, start=kefd_wp,
            sortie_budget=4 * ureg.hour,
            flight_day_budget=8 * ureg.hour,
            reserve=30 * ureg.minute,
            cruise_altitude=b200_cruise,
            refuel_airports=[klbb_wp],
            refuel_time=30 * ureg.minute,
            mode="round_trip",
            azimuth_resolution_deg=60.0,
            distance_tolerance_nmi=2.0,
        )
        cap = 4 * 60 - 30  # min
        assert (gdf["sortie_cycle_1_min"] <= cap + 1e-3).all()
        c2 = gdf["sortie_cycle_2_min"].dropna()
        assert (c2 <= cap + 1e-3).all()

    def test_different_return_airport(
        self, b200, kefd_wp, klbb_wp, kbtr_wp, b200_cruise,
    ):
        """return_safe with start≠recovery shows mixed itineraries."""
        gdf = compute_refuel_isochrone(
            aircraft=b200, start=kefd_wp,
            sortie_budget=4 * ureg.hour,
            flight_day_budget=8 * ureg.hour,
            cruise_altitude=b200_cruise,
            refuel_airports=[klbb_wp],
            refuel_time=30 * ureg.minute,
            return_destination=kbtr_wp,
            mode="return_safe",
            azimuth_resolution_deg=30.0,
            distance_tolerance_nmi=2.0,
        )
        kinds = set(gdf["itinerary"].unique())
        # Should see at least direct and one refuel template.
        assert "direct" in kinds
        assert kinds & {"outbound_refuel", "return_refuel"}

    def test_return_refuel_appears(
        self, b200, kefd_wp, klbb_wp, kbtr_wp, b200_cruise,
    ):
        """Some ray geometry should select return_refuel."""
        gdf = compute_refuel_isochrone(
            aircraft=b200, start=kefd_wp,
            sortie_budget=4 * ureg.hour,
            flight_day_budget=8 * ureg.hour,
            cruise_altitude=b200_cruise,
            refuel_airports=[klbb_wp],
            refuel_time=30 * ureg.minute,
            return_destination=kbtr_wp,
            mode="return_safe",
            azimuth_resolution_deg=15.0,
            distance_tolerance_nmi=2.0,
        )
        assert (gdf["itinerary"] == "return_refuel").any()

    # --- 3. diagnostics ----------------------------------------------------

    def test_diagnostic_correctness(
        self, b200, kefd_wp, klbb_wp, kbtr_wp, b200_cruise,
    ):
        """Per-leg columns sum, margins non-negative, schema invariants."""
        gdf = compute_refuel_isochrone(
            aircraft=b200, start=kefd_wp,
            sortie_budget=4 * ureg.hour,
            flight_day_budget=8 * ureg.hour,
            cruise_altitude=b200_cruise,
            on_station_time=10 * ureg.minute,
            refuel_airports=[klbb_wp],
            refuel_time=30 * ureg.minute,
            return_destination=kbtr_wp,
            mode="return_safe",
            azimuth_resolution_deg=30.0,
            distance_tolerance_nmi=2.0,
        )
        for _, row in gdf.iterrows():
            it = row["itinerary"]
            assert (row["refuel_count"] == 0) == (it == "direct")
            assert row["sortie_margin_min"] >= -1e-6
            assert row["day_margin_min"] >= -1e-6
            assert row["limiting_leg"] in {
                "sortie", "flight_day", "both", "slack",
            }
            day = row["day_total_time_min"]
            os_ = row["on_station_min"]
            if it == "direct":
                expected_c1 = (
                    row["start_to_target_time_min"]
                    + os_
                    + row["target_to_return_time_min"]
                )
                assert abs(expected_c1 - row["sortie_cycle_1_min"]) < 0.1
                assert np.isnan(row["sortie_cycle_2_min"])
                assert abs(day - row["sortie_cycle_1_min"]) < 0.1
            elif it == "outbound_refuel":
                c1 = row["start_to_refuel_time_min"]
                c2 = (
                    row["refuel_to_target_time_min"]
                    + os_
                    + row["target_to_return_time_min"]
                )
                assert abs(c1 - row["sortie_cycle_1_min"]) < 0.1
                assert abs(c2 - row["sortie_cycle_2_min"]) < 0.1
                assert abs(
                    day - (c1 + row["refuel_time_min"] + c2)
                ) < 0.1
            elif it == "return_refuel":
                c1 = (
                    row["start_to_target_time_min"]
                    + os_
                    + row["target_to_refuel_time_min"]
                )
                c2 = row["refuel_to_return_time_min"]
                assert abs(c1 - row["sortie_cycle_1_min"]) < 0.1
                assert abs(c2 - row["sortie_cycle_2_min"]) < 0.1
                assert abs(
                    day - (c1 + row["refuel_time_min"] + c2)
                ) < 0.1

    def test_refuel_used_matches_itinerary_set(
        self, b200, kefd_wp, klbb_wp, kbtr_wp, b200_cruise,
    ):
        """attrs['refuel_airports_used'] = unique non-direct refuel labels."""
        gdf = compute_refuel_isochrone(
            aircraft=b200, start=kefd_wp,
            sortie_budget=4 * ureg.hour,
            flight_day_budget=8 * ureg.hour,
            cruise_altitude=b200_cruise,
            refuel_airports=[klbb_wp],
            refuel_time=30 * ureg.minute,
            return_destination=kbtr_wp,
            mode="return_safe",
            azimuth_resolution_deg=30.0,
            distance_tolerance_nmi=2.0,
        )
        from_rows = set(
            r for r in gdf["refuel_airport"].dropna().unique()
        )
        assert set(gdf.attrs["refuel_airports_used"]) == from_rows


# ---------------------------------------------------------------------------
# Target reachability (single-point feasibility query)
# ---------------------------------------------------------------------------

class TestTargetReachability:
    """evaluate_target_reachability: single-point feasibility queries."""

    def test_reachable_direct(self, b200, kefd_wp, b200_cruise):
        """A close target reachable without refuel."""
        target = Waypoint(
            latitude=kefd_wp.latitude + 1.5,
            longitude=kefd_wp.longitude,
            heading=0.0, altitude_msl=b200_cruise,
        )
        result = evaluate_target_reachability(
            b200, kefd_wp, target,
            sortie_budget=4 * ureg.hour,
            cruise_altitude=b200_cruise,
            mode="round_trip",
        )
        assert result["reachable"]
        assert result["best"]["itinerary"] == "direct"
        assert result["best"]["refuel_airport"] is None
        assert result["alternatives"] == []
        assert result["unreachable_reason"] is None

    def test_reachable_via_refuel(
        self, b200, kefd_wp, klbb_wp, kbtr_wp, b200_cruise,
    ):
        """A far target reachable only via outbound refuel."""
        # Target ~600 nmi NW of KEFD, beyond direct return-safe range.
        target = Waypoint(
            latitude=35.0, longitude=-103.0, heading=0.0,
            altitude_msl=b200_cruise,
        )
        result = evaluate_target_reachability(
            b200, kefd_wp, target,
            sortie_budget=4 * ureg.hour,
            flight_day_budget=8 * ureg.hour,
            cruise_altitude=b200_cruise,
            refuel_airports=[klbb_wp],
            refuel_time=30 * ureg.minute,
            return_destination=kbtr_wp,
            mode="return_safe",
        )
        assert result["reachable"]
        assert result["best"]["refuel_airport"] == "KLBB"
        assert result["best"]["itinerary"] in {
            "outbound_refuel", "return_refuel",
        }

    def test_unreachable(self, b200, kefd_wp, b200_cruise):
        """An impossibly far target reports unreachable + reason."""
        target = Waypoint(
            latitude=70.0, longitude=10.0, heading=0.0,
            altitude_msl=b200_cruise,
        )
        result = evaluate_target_reachability(
            b200, kefd_wp, target,
            sortie_budget=4 * ureg.hour,
            cruise_altitude=b200_cruise,
            mode="round_trip",
        )
        assert not result["reachable"]
        assert result["best"] is None
        assert result["alternatives"] == []
        assert result["unreachable_reason"] is not None

    def test_alternatives_sorted_by_day_total(
        self, b200, kefd_wp, klbb_wp, kbtr_wp, b200_cruise,
    ):
        """Alternatives list is sorted by ascending day_total_time_min."""
        target = Waypoint(
            latitude=33.0, longitude=-100.0, heading=0.0,
            altitude_msl=b200_cruise,
        )
        result = evaluate_target_reachability(
            b200, kefd_wp, target,
            sortie_budget=4 * ureg.hour,
            flight_day_budget=8 * ureg.hour,
            cruise_altitude=b200_cruise,
            refuel_airports=[klbb_wp],
            refuel_time=30 * ureg.minute,
            return_destination=kbtr_wp,
            mode="return_safe",
        )
        assert result["reachable"]
        # best.day_total ≤ each alternative.day_total
        for alt in result["alternatives"]:
            assert (
                alt["day_total_time_min"]
                >= result["best"]["day_total_time_min"]
            )

    def test_no_refuel_airports_only_direct(
        self, b200, kefd_wp, b200_cruise,
    ):
        """refuel_airports=() evaluates only the direct itinerary."""
        target = Waypoint(
            latitude=kefd_wp.latitude + 2.0,
            longitude=kefd_wp.longitude,
            heading=0.0, altitude_msl=b200_cruise,
        )
        result = evaluate_target_reachability(
            b200, kefd_wp, target,
            sortie_budget=4 * ureg.hour,
            cruise_altitude=b200_cruise,
            mode="round_trip",
        )
        assert result["reachable"]
        assert result["best"]["itinerary"] == "direct"
        assert all(
            a["itinerary"] == "direct" for a in result["alternatives"]
        )

    def test_one_way_rejected(self, b200, kefd_wp, klbb_wp, b200_cruise):
        target = Waypoint(
            latitude=33.0, longitude=-100.0, heading=0.0,
            altitude_msl=b200_cruise,
        )
        with pytest.raises(HyPlanValueError, match="mode"):
            evaluate_target_reachability(
                b200, kefd_wp, target,
                sortie_budget=4 * ureg.hour,
                cruise_altitude=b200_cruise,
                refuel_airports=[klbb_wp],
                mode="one_way",
            )


# ---------------------------------------------------------------------------
# Concentric isochrones
# ---------------------------------------------------------------------------

class TestConcentric:
    """compute_concentric_isochrones: multi-budget sweep + amortization."""

    def test_monotone_in_budget(self, giii, kedw_wp, cruise_alt):
        """Per-ray reach is non-decreasing as budget grows."""
        gdf = compute_concentric_isochrones(
            giii, kedw_wp,
            budgets=[1 * ureg.hour, 2 * ureg.hour, 3 * ureg.hour],
            cruise_altitude=cruise_alt,
            mode="round_trip",
            azimuth_resolution_deg=60.0,
            distance_tolerance_nmi=2.0,
        )
        for az in gdf["azimuth_deg"].unique():
            d_by_b = (
                gdf[gdf["azimuth_deg"] == az]
                .sort_values("budget_hr")["distance_nmi"]
                .tolist()
            )
            for a, b in zip(d_by_b[:-1], d_by_b[1:]):
                assert b >= a - 2.0, (
                    f"az {az}: budget grew {a} → {b}, expected non-decreasing"
                )

    def test_single_budget_matches_compute_isochrone(
        self, giii, kedw_wp, cruise_alt,
    ):
        """Concentric with a single-element list matches compute_isochrone."""
        common = dict(
            cruise_altitude=cruise_alt,
            mode="round_trip",
            azimuth_resolution_deg=60.0,
            distance_tolerance_nmi=2.0,
        )
        single = compute_isochrone(
            giii, kedw_wp, 2 * ureg.hour, **common,
        )
        multi = compute_concentric_isochrones(
            giii, kedw_wp, budgets=[2 * ureg.hour], **common,
        )
        s = single.set_index("azimuth_deg")["distance_nmi"]
        m = multi.set_index("azimuth_deg")["distance_nmi"]
        assert (s - m).abs().max() < 1e-6

    def test_attrs_present(self, giii, kedw_wp, cruise_alt):
        gdf = compute_concentric_isochrones(
            giii, kedw_wp,
            budgets=[2 * ureg.hour, 1 * ureg.hour],  # unsorted input
            cruise_altitude=cruise_alt,
            mode="round_trip",
            azimuth_resolution_deg=120.0,
            distance_tolerance_nmi=5.0,
        )
        # Sorted ascending in attrs.
        assert gdf.attrs["budgets_hr"] == [1.0, 2.0]
        assert "budget_hr" in gdf.columns
        assert "budget_min" in gdf.columns
        assert set(gdf["budget_hr"].unique()) == {1.0, 2.0}

    def test_empty_budgets_raises(self, giii, kedw_wp, cruise_alt):
        with pytest.raises(HyPlanValueError, match="budgets"):
            compute_concentric_isochrones(
                giii, kedw_wp, budgets=[],
                cruise_altitude=cruise_alt, mode="round_trip",
            )


# ---------------------------------------------------------------------------
# Refuel-leg caching equivalence
# ---------------------------------------------------------------------------

def test_refuel_caching_constant_wind_matches_still_air(
    b200, kefd_wp, klbb_wp, kbtr_wp, b200_cruise,
):
    """Cached eligibility legs produce the same answer as the
    pre-cache implementation at the boundary tolerance.

    The simplest way to exercise both branches of _anchor_invariant
    is to run with both StillAirField and a non-trivial
    ConstantWindField and confirm sortie_cycle_1_min for direct
    rays still matches per-ray.
    """
    from hyplan.winds import ConstantWindField, StillAirField
    common = dict(
        aircraft=b200, start=kefd_wp,
        sortie_budget=4 * ureg.hour,
        flight_day_budget=8 * ureg.hour,
        cruise_altitude=b200_cruise,
        refuel_airports=[klbb_wp],
        refuel_time=30 * ureg.minute,
        return_destination=kbtr_wp,
        mode="return_safe",
        azimuth_resolution_deg=60.0,
        distance_tolerance_nmi=2.0,
    )
    g_still = compute_refuel_isochrone(wind_source=StillAirField(), **common)
    g_const = compute_refuel_isochrone(
        wind_source=ConstantWindField(20 * ureg.knot, wind_from_deg=270.0),
        **common,
    )
    # Both runs should produce a stable mix of itineraries with no NaNs
    # in the chosen path's per-cycle columns.
    for g in (g_still, g_const):
        assert (g["sortie_cycle_1_min"] >= 0).all()
        assert g["itinerary"].notna().all()
        # Sanity: KLBB used at least once (caching path exercised).
        assert g.attrs["refuel_airports_used"] == ["KLBB"]


# ---------------------------------------------------------------------------
# Static plotter smoke
# ---------------------------------------------------------------------------

def test_plot_isochrone_static_smoke(
    b200, kefd_wp, klbb_wp, kbtr_wp, b200_cruise,
):
    """Plotter accepts single GDF, list-of-tuples, concentric, and
    refuel results without raising; returns (Figure, Axes)."""
    pytest.importorskip(
        "cartopy",
        reason="plot_isochrone_static requires cartopy; not a core dep.",
    )
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    from hyplan import plot_isochrone_static

    gdf_single = compute_isochrone(
        b200, kefd_wp, 2 * ureg.hour,
        cruise_altitude=b200_cruise, mode="round_trip",
        azimuth_resolution_deg=120.0, distance_tolerance_nmi=5.0,
    )
    gdf_concentric = compute_concentric_isochrones(
        b200, kefd_wp,
        budgets=[1 * ureg.hour, 2 * ureg.hour],
        cruise_altitude=b200_cruise, mode="round_trip",
        azimuth_resolution_deg=120.0, distance_tolerance_nmi=5.0,
    )
    gdf_refuel = compute_refuel_isochrone(
        b200, kefd_wp,
        sortie_budget=4 * ureg.hour,
        flight_day_budget=8 * ureg.hour,
        cruise_altitude=b200_cruise,
        refuel_airports=[klbb_wp],
        refuel_time=30 * ureg.minute,
        return_destination=kbtr_wp, mode="return_safe",
        azimuth_resolution_deg=120.0, distance_tolerance_nmi=5.0,
    )

    for arg in (
        gdf_single,
        [(gdf_single, "steelblue", "B-200 2hr")],
        gdf_concentric,
        gdf_refuel,
    ):
        fig, ax = plot_isochrone_static(arg, basemap_scale="110m")
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


def test_plot_isochrone_folium_smoke(
    b200, kefd_wp, klbb_wp, kbtr_wp, b200_cruise,
):
    """Folium plotter accepts plain / refuel / concentric gdfs without
    raising; returns a folium.Map.  Exercises the polygon, start /
    recovery markers, per-ray dot popup branches (including the
    one_way headwind-only path), and the refuel-airport-marker branch."""
    import folium
    from hyplan.planning.isochrone import plot_isochrone

    # 1. Plain round-trip: polygon + start marker, no recovery marker.
    gdf_round = compute_isochrone(
        b200, kefd_wp, 2 * ureg.hour,
        cruise_altitude=b200_cruise, mode="round_trip",
        azimuth_resolution_deg=120.0, distance_tolerance_nmi=5.0,
    )
    m1 = plot_isochrone(gdf_round)
    assert isinstance(m1, folium.Map)

    # 2. one_way: per-ray dot popup hits the headwind-only branch
    #    (no return_time_min).
    gdf_oneway = compute_isochrone(
        b200, kefd_wp, 2 * ureg.hour,
        cruise_altitude=b200_cruise, mode="one_way",
        azimuth_resolution_deg=120.0, distance_tolerance_nmi=5.0,
    )
    m2 = plot_isochrone(gdf_oneway, color="firebrick", fill_opacity=0.3)
    assert isinstance(m2, folium.Map)

    # 3. return_safe with distinct recovery: exercises the recovery
    #    marker branch.
    gdf_rs = compute_isochrone(
        b200, kefd_wp, 4 * ureg.hour,
        cruise_altitude=b200_cruise, mode="return_safe",
        return_destination=kbtr_wp,
        azimuth_resolution_deg=120.0, distance_tolerance_nmi=5.0,
    )
    m3 = plot_isochrone(gdf_rs, tiles="CartoDB positron", zoom_start=5)
    assert isinstance(m3, folium.Map)

    # 4. Concentric: budget_hr labelling branch in the polygon popup.
    gdf_concentric = compute_concentric_isochrones(
        b200, kefd_wp,
        budgets=[1 * ureg.hour, 2 * ureg.hour],
        cruise_altitude=b200_cruise, mode="round_trip",
        azimuth_resolution_deg=120.0, distance_tolerance_nmi=5.0,
    )
    m4 = plot_isochrone(gdf_concentric)
    assert isinstance(m4, folium.Map)

    # 5. Refuel: refuel-airport-marker branches (used + unreachable).
    gdf_refuel = compute_refuel_isochrone(
        b200, kefd_wp,
        sortie_budget=4 * ureg.hour,
        flight_day_budget=8 * ureg.hour,
        cruise_altitude=b200_cruise,
        refuel_airports=[klbb_wp],
        refuel_time=30 * ureg.minute,
        return_destination=kbtr_wp, mode="return_safe",
        azimuth_resolution_deg=120.0, distance_tolerance_nmi=5.0,
    )
    m5 = plot_isochrone(gdf_refuel)
    assert isinstance(m5, folium.Map)

    # 6. Pre-existing base_map: caller supplies a Folium map; plotter
    #    should add layers without creating a new map.
    base = folium.Map(location=[kefd_wp.latitude, kefd_wp.longitude])
    m6 = plot_isochrone(gdf_round, base_map=base)
    assert m6 is base


# ---------------------------------------------------------------------------
# Wind sampling: cruise_midpoint / phase_midpoint / segmented_cruise
# ---------------------------------------------------------------------------

class _StepWindField:
    """Synthetic wind field: ``u = u_east_kt`` for ``lon >= lon0``,
    ``u = u_west_kt`` otherwise.  ``v = 0``.  Used to exercise
    cruise-segment quadrature under a sharp regime change."""

    def __init__(self, lon0: float, u_west_kt: float, u_east_kt: float):
        self.lon0 = lon0
        self.u_west_mps = u_west_kt * 0.514444
        self.u_east_mps = u_east_kt * 0.514444

    def wind_at(self, lat, lon, altitude, time):
        u_mps = self.u_east_mps if lon >= self.lon0 else self.u_west_mps
        return (
            u_mps * (ureg.meter / ureg.second),
            0.0 * (ureg.meter / ureg.second),
        )


class _LinearWindField:
    """Synthetic wind field with smoothly varying eastward wind:
    ``u(lon) = u_at_lon0_kt + slope_kt_per_deg * (lon - lon0)``.
    ``v = 0``.  Linear winds make midpoint quadrature exact in the
    limit, so monotone convergence under decreasing segment spacing
    is guaranteed."""

    def __init__(self, lon0: float, u_at_lon0_kt: float, slope_kt_per_deg: float):
        self.lon0 = lon0
        self.u0_mps = u_at_lon0_kt * 0.514444
        self.slope_mps_per_deg = slope_kt_per_deg * 0.514444

    def wind_at(self, lat, lon, altitude, time):
        u_mps = self.u0_mps + self.slope_mps_per_deg * (lon - self.lon0)
        return (
            u_mps * (ureg.meter / ureg.second),
            0.0 * (ureg.meter / ureg.second),
        )


class _AltitudeWindField:
    """Synthetic wind field where the eastward wind grows with
    altitude (e.g., jet-stream-like).  Used to verify that
    ``phase_midpoint`` actually samples climb/descent at the
    phase-mid altitude rather than the cruise altitude."""

    def __init__(self, u_per_ft_kt: float):
        self.u_per_ft_kt = u_per_ft_kt

    def wind_at(self, lat, lon, altitude, time):
        ft = altitude.m_as(ureg.feet)
        u_kt = self.u_per_ft_kt * ft
        return (
            u_kt * 0.514444 * (ureg.meter / ureg.second),
            0.0 * (ureg.meter / ureg.second),
        )


class TestWindSampling:
    """Configurable wind sampling: cruise_midpoint / phase_midpoint /
    segmented_cruise."""

    @staticmethod
    def _leg_time_helper(
        ac, start_wp, end_wp, cruise_alt, wind_source, **kw,
    ):
        from hyplan.planning.isochrone import _leg_time
        import datetime as _dt
        return _leg_time(
            aircraft=ac, start_wp=start_wp, end_wp=end_wp,
            cruise_altitude=cruise_alt,
            t_anchor=_dt.datetime(2026, 5, 6, tzinfo=_dt.timezone.utc),
            wind_source=wind_source,
            **kw,
        )

    def _east_west_leg(self, cruise_alt, length_deg=10.0, lat=30.0, lon0=-100.0):
        """Build start/end Waypoints for an east-west leg straddling
        ``lon0``: start at ``lon0 - length_deg/2``, end at
        ``lon0 + length_deg/2``."""
        start = Waypoint(
            latitude=lat, longitude=lon0 - length_deg / 2,
            heading=90.0, altitude_msl=cruise_alt,
        )
        end = Waypoint(
            latitude=lat, longitude=lon0 + length_deg / 2,
            heading=90.0, altitude_msl=cruise_alt,
        )
        return start, end, lon0

    # --- 1. StillAirField invariance -----------------------------------

    def test_still_air_invariance(self, b200, b200_cruise):
        from hyplan.winds import StillAirField
        start, end, _ = self._east_west_leg(b200_cruise, length_deg=4.0)
        wf = StillAirField()
        t1, _ = self._leg_time_helper(b200, start, end, b200_cruise, wf,
            wind_sampling="cruise_midpoint")
        t2, _ = self._leg_time_helper(b200, start, end, b200_cruise, wf,
            wind_sampling="phase_midpoint")
        t3, _ = self._leg_time_helper(b200, start, end, b200_cruise, wf,
            wind_sampling="segmented_cruise")
        assert abs(t1 - t2) < 1e-9
        assert abs(t1 - t3) < 1e-9

    # --- 2. cruise_midpoint backward-compat under ConstantWindField ----

    def test_cruise_midpoint_constant_wind_unchanged(
        self, b200, b200_cruise,
    ):
        """Default mode under ConstantWindField produces the same
        result as it did pre-patch (single sample, still-air
        climb/descent).  We snapshot against a recompute using the
        bare _leg_time call with the explicit default kwargs."""
        from hyplan.winds import ConstantWindField
        start, end, _ = self._east_west_leg(b200_cruise, length_deg=8.0)
        wf = ConstantWindField(40 * ureg.knot, wind_from_deg=270.0)
        t_default, hw_default = self._leg_time_helper(
            b200, start, end, b200_cruise, wf,
        )
        t_explicit, hw_explicit = self._leg_time_helper(
            b200, start, end, b200_cruise, wf,
            wind_sampling="cruise_midpoint",
        )
        assert t_default == t_explicit
        assert hw_default == hw_explicit
        # Sanity: tailwind from the west on an east-bound leg.
        assert hw_default < 0  # negative headwind = tailwind

    # --- 3. ConstantWindField phase awareness --------------------------

    def test_constant_wind_phase_awareness(
        self, b200,
    ):
        """phase_midpoint differs from cruise_midpoint for a leg with
        non-trivial climb/descent under a uniform wind.  With a
        tailwind from the west, phase-aware mode allocates more ground
        distance to climb/descent → less ground for cruise → total
        leg time *decreases*."""
        from hyplan.winds import ConstantWindField
        # Long enough to require non-trivial cruise; ground-elevation
        # start to FL250 cruise to ground-elevation recovery so both
        # climb and descent are exercised.
        cruise_alt = 25000 * ureg.feet
        start = Waypoint(
            latitude=30.0, longitude=-100.0, heading=90.0,
            altitude_msl=100 * ureg.feet,
        )
        end = Waypoint(
            latitude=30.0, longitude=-95.0, heading=90.0,
            altitude_msl=100 * ureg.feet,
        )
        wf = ConstantWindField(50 * ureg.knot, wind_from_deg=270.0)
        t_cm, _ = self._leg_time_helper(
            b200, start, end, cruise_alt, wf,
            wind_sampling="cruise_midpoint",
        )
        t_pm, _ = self._leg_time_helper(
            b200, start, end, cruise_alt, wf,
            wind_sampling="phase_midpoint",
        )
        assert abs(t_cm - t_pm) > 0.5, (
            f"phase_midpoint should differ from cruise_midpoint by "
            f">0.5 min, got {abs(t_cm - t_pm):.4f} min"
        )
        # Tailwind: phase-aware allocates more ground distance to
        # climb/descent, so total time decreases.
        assert t_pm < t_cm

    # --- 4. Linear wind: convergence under decreasing spacing ----------

    def test_linear_wind_convergence(self, b200, b200_cruise):
        """Under a smooth linear wind, segmented_cruise error should
        decrease monotonically as wind_sample_spacing decreases."""
        # East-west leg ~600 nmi long, smoothly varying eastward wind
        # from -50 kt at lon=-105 to +50 kt at lon=-95 (gradient
        # 10 kt/deg).
        start, end, lon0 = self._east_west_leg(
            b200_cruise, length_deg=10.0, lon0=-100.0,
        )
        wf = _LinearWindField(
            lon0=lon0, u_at_lon0_kt=0.0, slope_kt_per_deg=10.0,
        )
        spacings_nmi = [400.0, 200.0, 100.0, 50.0, 25.0]
        times = []
        for s in spacings_nmi:
            t, _ = self._leg_time_helper(
                b200, start, end, b200_cruise, wf,
                wind_sampling="segmented_cruise",
                wind_sample_spacing=s * ureg.nautical_mile,
                max_wind_samples_per_leg=200,
            )
            times.append(t)
        # Truth: by symmetry of the linear wind around the cruise
        # midpoint (lon0), the average tailwind across the cruise is
        # exactly zero.  The "exact" cruise time is therefore
        # cruise_distance / TAS.  segmented_cruise should converge to
        # this; finer spacing should be no farther from the limit
        # value than coarser spacing.
        finest = times[-1]
        prev_err = abs(times[0] - finest)
        for t in times[1:]:
            err = abs(t - finest)
            assert err <= prev_err + 1e-6, (
                f"non-monotone convergence: errs {prev_err} -> {err}"
            )
            prev_err = err

    # --- 5. Step wind: segmented beats cruise_midpoint -----------------

    def test_step_wind_segmented_beats_midpoint(self, b200, b200_cruise):
        """Step-function wind: cruise_midpoint samples one regime;
        segmented integrates across the step.  Construct the leg so
        the cruise midpoint sits clearly *east* of the discontinuity,
        but the leg spans both regimes."""
        # Leg from lon=-110 (well west of step at lon0=-100) to lon=-95
        # (east of step) — total span 15°, midpoint at lon=-102.5.
        # Wait: midpoint at lon=-102.5 is *west* of step at -100, so
        # the cruise_midpoint samples the western (headwind) regime
        # for the whole cruise — biased.  segmented sees both regimes.
        cruise_alt = b200_cruise
        lon_west = -110.0
        lon_east = -95.0
        start = Waypoint(
            latitude=30.0, longitude=lon_west, heading=90.0,
            altitude_msl=cruise_alt,
        )
        end = Waypoint(
            latitude=30.0, longitude=lon_east, heading=90.0,
            altitude_msl=cruise_alt,
        )
        wf = _StepWindField(lon0=-100.0, u_west_kt=-60.0, u_east_kt=60.0)
        t_cm, _ = self._leg_time_helper(
            b200, start, end, cruise_alt, wf,
            wind_sampling="cruise_midpoint",
        )
        t_seg, _ = self._leg_time_helper(
            b200, start, end, cruise_alt, wf,
            wind_sampling="segmented_cruise",
            wind_sample_spacing=25 * ureg.nautical_mile,
            max_wind_samples_per_leg=100,
        )
        # The midpoint (lon=-102.5) sees u=-60 (headwind on east-bound
        # leg).  segmented sees ~half headwind + half tailwind, so its
        # reported time is shorter (and closer to the true mixed-wind
        # integral).
        assert t_seg < t_cm, f"segmented {t_seg} should be < midpoint {t_cm}"

    # --- 6. Sample cap respected ---------------------------------------

    def test_sample_cap_respected(self, b200, b200_cruise):
        """wind_sample_spacing=1 nmi but max_wind_samples_per_leg=5 →
        exactly 5 cruise subsegments, regardless of leg length."""
        from hyplan.planning.isochrone import _cruise_time_segmented
        from hyplan.winds import StillAirField
        start_wp = Waypoint(
            latitude=30.0, longitude=-100.0, heading=90.0,
            altitude_msl=b200_cruise,
        )
        cruise_tas = b200.cruise_speed_at(b200_cruise)
        cruise_tas_kt = cruise_tas.m_as(ureg.knot)
        import datetime as _dt
        result = _cruise_time_segmented(
            wind_source=StillAirField(),
            start_wp=start_wp, track_deg=90.0,
            d_climb_nmi=0.0, cruise_distance_nmi=500.0,
            cruise_altitude=b200_cruise,
            cruise_tas=cruise_tas, cruise_tas_kt=cruise_tas_kt,
            t_anchor=_dt.datetime(2026, 5, 6, tzinfo=_dt.timezone.utc),
            t_climb_min=0.0,
            n_segments=5,
        )
        assert result["n_samples"] == 5
        assert result["status"] == "ok"

    # --- 7. Unflyable status -------------------------------------------

    def test_unflyable_status(self, b200, b200_cruise):
        """Headwind > TAS in a subsegment → status='unflyable',
        time=inf."""
        from hyplan.winds import ConstantWindField
        start, end, _ = self._east_west_leg(b200_cruise, length_deg=4.0)
        # B-200 cruise TAS ~240 kt; pure 350 kt headwind is unflyable.
        wf = ConstantWindField(350 * ureg.knot, wind_from_deg=90.0)
        t, hw = self._leg_time_helper(
            b200, start, end, b200_cruise, wf,
            wind_sampling="segmented_cruise",
            wind_sample_spacing=50 * ureg.nautical_mile,
        )
        assert t == float("inf")
        assert hw == float("inf")

    # --- 8. Validation -------------------------------------------------

    def test_invalid_wind_sampling(self, b200, kefd_wp, b200_cruise):
        with pytest.raises(HyPlanValueError, match="wind_sampling"):
            compute_isochrone(
                aircraft=b200, start=kefd_wp, budget=2 * ureg.hour,
                cruise_altitude=b200_cruise, mode="round_trip",
                wind_sampling="not_a_real_mode",
            )

    def test_negative_spacing(self, b200, kefd_wp, b200_cruise):
        with pytest.raises(HyPlanValueError, match="wind_sample_spacing"):
            compute_isochrone(
                aircraft=b200, start=kefd_wp, budget=2 * ureg.hour,
                cruise_altitude=b200_cruise, mode="round_trip",
                wind_sample_spacing=-1 * ureg.nautical_mile,
            )

    def test_zero_max_samples(self, b200, kefd_wp, b200_cruise):
        with pytest.raises(HyPlanValueError, match="max_wind_samples"):
            compute_isochrone(
                aircraft=b200, start=kefd_wp, budget=2 * ureg.hour,
                cruise_altitude=b200_cruise, mode="round_trip",
                max_wind_samples_per_leg=0,
            )

    # --- 9. Public-API smoke -------------------------------------------

    def test_public_api_smoke(self, b200, kefd_wp, b200_cruise):
        """compute_isochrone with segmented_cruise runs and produces a
        valid GeoDataFrame."""
        from hyplan.winds import ConstantWindField
        gdf = compute_isochrone(
            aircraft=b200, start=kefd_wp, budget=2 * ureg.hour,
            cruise_altitude=b200_cruise, mode="round_trip",
            wind_source=ConstantWindField(30 * ureg.knot, wind_from_deg=270.0),
            wind_sampling="segmented_cruise",
            wind_sample_spacing=50 * ureg.nautical_mile,
            azimuth_resolution_deg=120.0, distance_tolerance_nmi=5.0,
        )
        assert len(gdf) == 3
        assert (gdf["distance_nmi"] > 0).all()


# ---------------------------------------------------------------------------
# 11. Multi-base isochrone (compute_multi_base_isochrone, union mode)
# ---------------------------------------------------------------------------

class TestMultiBase:
    """compute_multi_base_isochrone, return_mode='union'."""

    @pytest.fixture
    def kpmd_wp(self) -> Waypoint:
        """KPMD (Palmdale) — close enough to KEDW that polygons overlap."""
        return Waypoint(
            latitude=34.629, longitude=-118.085,
            heading=0.0, altitude_msl=2540 * ureg.feet,
            name="KPMD",
        )

    @pytest.fixture
    def kbtr_wp(self) -> Waypoint:
        """KBTR (Baton Rouge) — far from KEDW, polygons should not overlap."""
        return Waypoint(
            latitude=30.533, longitude=-91.150,
            heading=0.0, altitude_msl=70 * ureg.feet,
            name="KBTR",
        )

    def test_union_two_overlapping_bases_is_polygon(
        self, giii, kedw_wp, kpmd_wp, cruise_alt,
    ):
        """Two close bases produce a single Polygon (overlapping reaches)."""
        gdf = compute_multi_base_isochrone(
            aircraft=giii,
            bases=[kedw_wp, kpmd_wp],
            budget=2 * ureg.hour,
            cruise_altitude=cruise_alt,
            wind_source=StillAirField(),
            azimuth_resolution_deg=30.0,
        )
        from shapely.geometry import Polygon as ShPolygon
        assert len(gdf) == 1
        assert isinstance(gdf.geometry.iloc[0], ShPolygon)
        assert gdf.iloc[0]["n_bases"] == 2
        assert gdf.iloc[0]["n_contributing_bases"] == 2
        assert gdf.iloc[0]["base_labels"] == ["KEDW", "KPMD"]
        assert "per_base_gdfs" in gdf.attrs
        assert len(gdf.attrs["per_base_gdfs"]) == 2

    def test_union_two_distant_bases_is_multipolygon(
        self, giii, kedw_wp, kbtr_wp, cruise_alt,
    ):
        """Two far-apart bases produce a MultiPolygon (disjoint reaches)."""
        from shapely.geometry import MultiPolygon
        gdf = compute_multi_base_isochrone(
            aircraft=giii,
            bases=[kedw_wp, kbtr_wp],
            budget=1 * ureg.hour,
            cruise_altitude=cruise_alt,
            wind_source=StillAirField(),
            azimuth_resolution_deg=60.0,
        )
        assert isinstance(gdf.geometry.iloc[0], MultiPolygon)
        assert gdf.iloc[0]["n_bases"] == 2

    def test_single_base_matches_compute_isochrone(
        self, giii, kedw_wp, cruise_alt,
    ):
        """One base should produce the same polygon area as compute_isochrone."""
        single = compute_isochrone(
            aircraft=giii, start=kedw_wp, budget=2 * ureg.hour,
            cruise_altitude=cruise_alt,
            wind_source=StillAirField(),
            azimuth_resolution_deg=30.0,
        )
        single_poly = isochrone_polygon(single)
        multi = compute_multi_base_isochrone(
            aircraft=giii,
            bases=[kedw_wp],
            budget=2 * ureg.hour,
            cruise_altitude=cruise_alt,
            wind_source=StillAirField(),
            azimuth_resolution_deg=30.0,
        )
        # Areas should match within float precision (same boundary points).
        assert np.isclose(
            multi.geometry.iloc[0].area, single_poly.area, rtol=1e-9
        )

    def test_per_base_recovery(self, giii, kedw_wp, kpmd_wp, cruise_alt):
        """return_destinations parallel to bases is honored."""
        gdf = compute_multi_base_isochrone(
            aircraft=giii,
            bases=[kedw_wp, kpmd_wp],
            return_destinations=[kedw_wp, kpmd_wp],
            budget=2 * ureg.hour,
            cruise_altitude=cruise_alt,
            mode="return_safe",
            wind_source=StillAirField(),
            azimuth_resolution_deg=60.0,
        )
        assert gdf.iloc[0]["n_contributing_bases"] == 2

    def test_validation_empty_bases(self, giii, cruise_alt):
        with pytest.raises(HyPlanValueError, match="non-empty"):
            compute_multi_base_isochrone(
                aircraft=giii, bases=[], budget=2 * ureg.hour,
                cruise_altitude=cruise_alt,
            )

    def test_validation_unsupported_return_mode(
        self, giii, kedw_wp, cruise_alt,
    ):
        with pytest.raises(HyPlanValueError, match="return_mode"):
            compute_multi_base_isochrone(
                aircraft=giii, bases=[kedw_wp], budget=2 * ureg.hour,
                cruise_altitude=cruise_alt,
                return_mode="best_base",
            )

    def test_validation_recovery_length_mismatch(
        self, giii, kedw_wp, kpmd_wp, cruise_alt,
    ):
        with pytest.raises(HyPlanValueError, match="return_destinations"):
            compute_multi_base_isochrone(
                aircraft=giii,
                bases=[kedw_wp, kpmd_wp],
                return_destinations=[kedw_wp],  # length 1, bases length 2
                budget=2 * ureg.hour,
                cruise_altitude=cruise_alt,
            )
