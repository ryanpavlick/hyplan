"""KML / KMZ export for Google Earth visualization of flight plans."""

from __future__ import annotations

import datetime

import geopandas as gpd

from ._common import extract_waypoints, generate_wp_names

__all__ = ["to_kml"]


def to_kml(
    plan: gpd.GeoDataFrame,
    filepath: str,
    takeoff_time: datetime.datetime | None = None,
    altitude_exaggeration: float = 1.0,
) -> None:
    """Write a KML (and optionally KMZ) file for Google Earth.

    Matches the MovingLines ``save2kml`` layout.

    Args:
        plan: Flight plan GeoDataFrame.
        filepath: Output ``.kml`` or ``.kmz`` path.
        takeoff_time: Optional UTC takeoff time.
        altitude_exaggeration: Multiplier for altitude in the KML
            coordinates (default 1.0 = true scale; MovingLines uses 10).
    """
    import simplekml

    wps = extract_waypoints(plan)
    wp_names = generate_wp_names(
        len(wps),
        date=takeoff_time.date() if takeoff_time else None,  # type: ignore[arg-type]
    )

    if takeoff_time:
        base_minutes = takeoff_time.hour * 60 + takeoff_time.minute + takeoff_time.second / 60.0
    else:
        base_minutes = 0.0

    kml = simplekml.Kml()

    # Camera at first waypoint
    if len(wps) > 0:
        w0 = wps.iloc[0]
        kml.document.lookat = simplekml.LookAt(
            longitude=w0["lon"], latitude=w0["lat"],
            altitude=3000, altitudemode=simplekml.AltitudeMode.relativetoground,
            range=50000,
        )

    # Waypoint points
    for idx, (_, wp) in enumerate(wps.iterrows()):
        name = wp_names[idx] if idx < len(wp_names) else f"WP{idx}"
        alt_display = (wp["alt_m"] or 0) * altitude_exaggeration

        utc_min = base_minutes + wp["cum_time_min"]
        utc_h = int(utc_min // 60)
        utc_m = int(utc_min % 60)

        desc = (
            f"WP# {int(wp['wp'])}<br>"
            f"UTC: {utc_h:02d}:{utc_m:02d}<br>"
            f"Name: {name}<br>"
            f"CumDist: {wp['cum_dist_nm']:.1f} nm<br>"
            f"Speed: {wp['speed_kt']:.0f} kt<br>"
            f"Bearing: {wp['heading']:.0f} deg<br>"
            f"Alt: {wp['alt_kft'] or 0:.1f} kft<br>"
            f"Type: {wp['segment_type']}"
        )

        pnt = kml.newpoint(name=name, coords=[(wp["lon"], wp["lat"], alt_display)])
        pnt.altitudemode = simplekml.AltitudeMode.relativetoground
        pnt.extrude = 1
        pnt.description = desc

    # Flight path line
    coords = []
    for _, wp in wps.iterrows():
        alt_display = (wp["alt_m"] or 0) * altitude_exaggeration
        coords.append((wp["lon"], wp["lat"], alt_display))

    if coords:
        ls = kml.newlinestring(name="Flight Plan")
        ls.coords = coords
        ls.altitudemode = simplekml.AltitudeMode.relativetoground
        ls.extrude = 1
        ls.style.linestyle.width = 4.0
        ls.style.linestyle.color = simplekml.Color.red

    # Save
    if filepath.lower().endswith(".kmz"):
        kml.savekmz(filepath)
    else:
        kml.save(filepath)
        # Also save KMZ companion
        kmz_path = filepath.rsplit(".", 1)[0] + ".kmz"
        kml.savekmz(kmz_path)
