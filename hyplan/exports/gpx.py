"""GPX route export for hyplan flight plans."""

import datetime

import geopandas as gpd

from ._common import extract_waypoints

__all__ = ["to_gpx"]


def to_gpx(
    plan: gpd.GeoDataFrame,
    filepath: str,
    mission_name: str = "",
    takeoff_time: datetime.datetime = None,
) -> None:
    """Write a GPX route file.

    Matches the MovingLines ``save2gpx`` format.

    Args:
        plan: Flight plan GeoDataFrame.
        filepath: Output ``.gpx`` path.
        mission_name: Route name.
        takeoff_time: Optional UTC takeoff time.
    """
    import gpxpy
    import gpxpy.gpx

    wps = extract_waypoints(plan)

    gpx = gpxpy.gpx.GPX()
    route = gpxpy.gpx.GPXRoute(name=mission_name or "Flight Plan")
    gpx.routes.append(route)

    for idx, (_, wp) in enumerate(wps.iterrows()):
        rpt = gpxpy.gpx.GPXRoutePoint(
            latitude=wp["lat"],
            longitude=wp["lon"],
            elevation=wp["alt_m"] or 0,
            name=f"WP#{idx}",
            comment=wp["segment_type"] or "",
        )

        if takeoff_time:
            rpt.time = takeoff_time + datetime.timedelta(minutes=wp["cum_time_min"])

        route.points.append(rpt)

    with open(filepath, "w") as f:
        f.write(gpx.to_xml())
