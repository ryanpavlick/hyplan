"""Excel exports for hyplan flight plans.

Two formats:

* :func:`to_excel` — full 24-column MovingLines working format, used for
  round-tripping with the MovingLines tool.
* :func:`to_pilot_excel` — simplified pilot-facing layout matching the
  MovingLines ``_for_pilots.xlsx`` format.
"""

import datetime

import geopandas as gpd

from ..geometry import (
    dd_to_ddm,
    dd_to_ddms,
    dd_to_foreflight_oneline,
    dd_to_nddmm,
    magnetic_declination,
    true_to_magnetic,
)
from ._common import (
    _compute_solar_azimuth,
    _compute_sza,
    _utc_fraction,
    extract_waypoints,
    generate_wp_names,
)

__all__ = ["to_excel", "to_pilot_excel"]


# ---------------------------------------------------------------------------
# Full working Excel  (MovingLines ``write_to_excel`` layout)
# ---------------------------------------------------------------------------

def to_excel(
    plan: gpd.GeoDataFrame,
    filepath: str,
    aircraft=None,
    takeoff_time: datetime.datetime = None,
    mission_name: str = "",
) -> None:
    """Write a full MovingLines-compatible working Excel file.

    This produces the 24-column format (A-X) that MovingLines uses as its
    internal working format, enabling round-tripping between tools.

    Args:
        plan: Flight plan GeoDataFrame from ``compute_flight_plan()``.
        filepath: Output ``.xlsx`` path.
        aircraft: Optional Aircraft object for metadata.
        takeoff_time: UTC datetime of takeoff.  If *None*, times are shown
            as elapsed minutes from 00:00 UTC.
        mission_name: Campaign / mission name string.
    """
    import xlsxwriter

    wps = extract_waypoints(plan)
    wp_names = generate_wp_names(
        len(wps),
        date=takeoff_time.date() if takeoff_time else None,
    )

    # Compute UTC base (minutes from midnight)
    if takeoff_time:
        base_minutes = takeoff_time.hour * 60 + takeoff_time.minute + takeoff_time.second / 60.0
    else:
        base_minutes = 0.0

    wb = xlsxwriter.Workbook(filepath, {"nan_inf_to_errors": True})
    ws = wb.add_worksheet("Flight Plan")

    # Formats
    hdr_fmt = wb.add_format({"bold": True, "bg_color": "#4472C4",
                             "font_color": "white", "font_size": 12,
                             "font_name": "Aptos Narrow"})
    cell_fmt = wb.add_format({"font_size": 12, "font_name": "Aptos Narrow"})
    time_fmt = wb.add_format({"font_size": 12, "font_name": "Aptos Narrow",
                              "num_format": "hh:mm"})
    dec_fmt = wb.add_format({"font_size": 12, "font_name": "Aptos Narrow",
                             "num_format": "0.0"})
    dec2_fmt = wb.add_format({"font_size": 12, "font_name": "Aptos Narrow",
                              "num_format": "0.00"})
    int_fmt = wb.add_format({"font_size": 12, "font_name": "Aptos Narrow",
                             "num_format": "0"})

    # Headers (matching MovingLines column order)
    headers = [
        "WP", "Lat [+-90]", "Lon [+-180]", "Speed [m/s]", "delayT [min]",
        "Altitude [m]", "CumLegT [hh:mm]", "UTC [hh:mm]", "LocalT [hh:mm]",
        "LegT [hh:mm]", "Dist [km]", "CumDist [km]", "Dist [Nmi]",
        "CumDist [Nmi]", "Speed [kt]", "Altitude [kft]", "SZA [deg]",
        "AZI [deg]", "Bearing [deg]", "ClimbT [min]", "Comments",
        "WP names", "HdWind [kt]", "HdWind [m/s]",
    ]
    for col, h in enumerate(headers):
        ws.write(0, col, h, hdr_fmt)

    ws.freeze_panes(1, 0)

    # Data rows
    for idx, (_, wp) in enumerate(wps.iterrows()):
        row = idx + 1
        utc_min = base_minutes + wp["cum_time_min"]
        local_offset_h = wp["lon"] / 15.0  # approximate local time
        local_min = utc_min + local_offset_h * 60.0

        # Compute SZA/AZI if we have a real takeoff time
        if takeoff_time:
            wp_dt = takeoff_time + datetime.timedelta(minutes=wp["cum_time_min"])
            sza = _compute_sza(wp["lat"], wp["lon"], wp_dt)
            azi = _compute_solar_azimuth(wp["lat"], wp["lon"], wp_dt)
        else:
            sza = -9999.0
            azi = -9999.0

        ws.write(row, 0, int(wp["wp"]), int_fmt)                   # WP
        ws.write(row, 1, wp["lat"], cell_fmt)                      # Lat
        ws.write(row, 2, wp["lon"], cell_fmt)                      # Lon
        ws.write(row, 3, wp["speed_mps"], dec2_fmt)                # Speed m/s
        ws.write(row, 4, 0, int_fmt)                               # delayT
        ws.write(row, 5, wp["alt_m"] or 0, dec_fmt)                # Altitude m
        ws.write(row, 6, _utc_fraction(wp["cum_time_min"]), time_fmt)  # CumLegT
        ws.write(row, 7, _utc_fraction(utc_min), time_fmt)        # UTC
        ws.write(row, 8, _utc_fraction(local_min % 1440), time_fmt)  # LocalT
        ws.write(row, 9, _utc_fraction(wp["leg_time_min"]), time_fmt)  # LegT
        ws.write(row, 10, wp["dist_km"], dec_fmt)                  # Dist km
        ws.write(row, 11, wp["cum_dist_km"], dec_fmt)              # CumDist km
        ws.write(row, 12, wp["dist_nm"], dec_fmt)                  # Dist Nmi
        ws.write(row, 13, wp["cum_dist_nm"], dec_fmt)              # CumDist Nmi
        ws.write(row, 14, wp["speed_kt"], dec_fmt)                 # Speed kt
        ws.write(row, 15, wp["alt_kft"] or 0, dec_fmt)            # Altitude kft
        ws.write(row, 16, sza, dec_fmt)                            # SZA
        ws.write(row, 17, azi, dec_fmt)                            # AZI
        ws.write(row, 18, wp["heading"], dec_fmt)                  # Bearing
        ws.write(row, 19, 0.0, dec_fmt)                            # ClimbT
        ws.write(row, 20, wp["segment_type"], cell_fmt)            # Comments
        ws.write(row, 21, wp_names[idx] if idx < len(wp_names) else "", cell_fmt)  # WP names
        ws.write(row, 22, 0.0, dec_fmt)                            # HdWind kt
        ws.write(row, 23, 0.0, dec_fmt)                            # HdWind m/s

    # Metadata cells (column W+ area, matching MovingLines)
    meta_row = 0
    date_str = (takeoff_time.strftime("%Y-%m-%d") if takeoff_time
                else datetime.date.today().strftime("%Y-%m-%d"))
    ws.write(meta_row, 24, date_str, cell_fmt)
    ws.write(meta_row, 25, mission_name, cell_fmt)
    ws.write(meta_row, 26, "Created with", cell_fmt)
    ws.write(meta_row, 27, "hyplan", cell_fmt)

    wb.close()


# ---------------------------------------------------------------------------
# Pilot Excel  (MovingLines ``save2xl_for_pilots_xlswriter`` layout)
# ---------------------------------------------------------------------------

def to_pilot_excel(
    plan: gpd.GeoDataFrame,
    filepath: str,
    aircraft=None,
    takeoff_time: datetime.datetime = None,
    mission_name: str = "",
    coord_format: str = "DD MM",
    include_mag_heading: bool = False,
) -> None:
    """Write a simplified pilot-facing Excel file.

    Matches the MovingLines ``_for_pilots.xlsx`` format.

    Args:
        plan: Flight plan GeoDataFrame from ``compute_flight_plan()``.
        filepath: Output ``.xlsx`` path.
        aircraft: Optional Aircraft object for metadata (type, tail number).
        takeoff_time: UTC datetime of takeoff.
        mission_name: Campaign / mission name.
        coord_format: Coordinate display format — one of
            ``'DD MM'``, ``'DD MM SS'``, ``'NDDD MM.SS'``.
        include_mag_heading: Include a magnetic heading column. Requires the
            optional ``geomag`` package (``pip install hyplan[mag]``); off by
            default so the export works in conda-only installs.
    """
    import xlsxwriter

    wps = extract_waypoints(plan)
    wp_names = generate_wp_names(
        len(wps),
        date=takeoff_time.date() if takeoff_time else None,
    )

    if takeoff_time:
        base_minutes = takeoff_time.hour * 60 + takeoff_time.minute + takeoff_time.second / 60.0
    else:
        base_minutes = 0.0

    wb = xlsxwriter.Workbook(filepath, {"nan_inf_to_errors": True})
    ws = wb.add_worksheet("Pilots")

    # Title row
    title_fmt = wb.add_format({
        "bold": True, "font_size": 24, "font_color": "white",
        "bg_color": "#4472C4", "font_name": "Aptos Narrow",
    })
    aircraft_name = ""
    if aircraft:
        aircraft_name = getattr(aircraft, "aircraft_type", "")
        tail = getattr(aircraft, "tail_number", "")
        if tail:
            aircraft_name += f" ({tail})"

    date_str = (takeoff_time.strftime("%Y-%m-%d") if takeoff_time
                else datetime.date.today().strftime("%Y-%m-%d"))
    title = f"{mission_name}  {aircraft_name}  {date_str}".strip()
    ws.merge_range(0, 0, 0, 6, title, title_fmt)

    # Header row
    hdr_fmt = wb.add_format({
        "bold": True, "bg_color": "#4472C4", "font_color": "white",
        "font_size": 12, "font_name": "Aptos Narrow",
    })
    headers = ["WP", "WP name", "Lat", "Lon", "Altitude [kft]", "UTC [hh:mm]"]
    if include_mag_heading:
        headers.append("Mag Heading [deg]")
    headers.append("Comments")

    for col, h in enumerate(headers):
        ws.write(1, col, h, hdr_fmt)

    ws.freeze_panes(2, 0)

    # Alternating row formats
    fmt_even = wb.add_format({
        "font_size": 12, "font_name": "Aptos Narrow", "bg_color": "#D6E4F0",
    })
    fmt_odd = wb.add_format({
        "font_size": 12, "font_name": "Aptos Narrow",
    })
    time_even = wb.add_format({
        "font_size": 12, "font_name": "Aptos Narrow", "bg_color": "#D6E4F0",
        "num_format": "hh:mm",
    })
    time_odd = wb.add_format({
        "font_size": 12, "font_name": "Aptos Narrow",
        "num_format": "hh:mm",
    })
    dec_even = wb.add_format({
        "font_size": 12, "font_name": "Aptos Narrow", "bg_color": "#D6E4F0",
        "num_format": "0.00",
    })
    dec_odd = wb.add_format({
        "font_size": 12, "font_name": "Aptos Narrow",
        "num_format": "0.00",
    })
    hdg_even = wb.add_format({
        "font_size": 12, "font_name": "Aptos Narrow", "bg_color": "#D6E4F0",
        "num_format": "0.0",
    })
    hdg_odd = wb.add_format({
        "font_size": 12, "font_name": "Aptos Narrow",
        "num_format": "0.0",
    })

    # Coordinate formatter
    if coord_format == "DD MM SS":
        coord_fn = dd_to_ddms
    elif coord_format == "NDDD MM.SS":
        coord_fn = dd_to_nddmm
    else:
        coord_fn = dd_to_ddm

    for idx, (_, wp) in enumerate(wps.iterrows()):
        row = idx + 2
        cf = fmt_even if idx % 2 == 0 else fmt_odd
        tf = time_even if idx % 2 == 0 else time_odd
        df = dec_even if idx % 2 == 0 else dec_odd
        hf = hdg_even if idx % 2 == 0 else hdg_odd

        lat_s, lon_s = coord_fn(wp["lat"], wp["lon"])
        utc_min = base_minutes + wp["cum_time_min"]

        col = 0
        ws.write(row, col, int(wp["wp"]), cf)
        col += 1
        ws.write(row, col, wp_names[idx] if idx < len(wp_names) else "", cf)
        col += 1
        ws.write(row, col, lat_s, cf)
        col += 1
        ws.write(row, col, lon_s, cf)
        col += 1
        ws.write(row, col, wp["alt_kft"] or 0, df)
        col += 1
        ws.write(row, col, _utc_fraction(utc_min), tf)
        col += 1

        if include_mag_heading:
            dec = magnetic_declination(wp["lat"], wp["lon"])
            mag_hdg = true_to_magnetic(wp["heading"], dec)
            ws.write(row, col, mag_hdg, hf)
            col += 1

        comment = wp["segment_type"]
        ws.write(row, col, comment, cf)

    # ForeFlight one-liner footer
    footer_row = len(wps) + 3
    ws.write(footer_row, 0, "One line waypoints for foreflight:", wb.add_format({"bold": True}))
    oneline_parts = []
    for _, wp in wps.iterrows():
        oneline_parts.append(dd_to_foreflight_oneline(wp["lat"], wp["lon"]))
    ws.write(footer_row + 1, 0, " ".join(oneline_parts))

    # Metadata
    meta_col = len(headers) + 1
    ws.write(1, meta_col, date_str)
    ws.write(1, meta_col + 1, mission_name)
    ws.write(1, meta_col + 2, "Created with")
    ws.write(1, meta_col + 3, "hyplan")

    wb.close()
