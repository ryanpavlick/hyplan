# Sun Position

Compute solar azimuth, elevation, and daily data-collection windows for
any site, date, and elevation threshold.

```{eval-rst}
.. automodule:: hyplan.sun
   :members:
   :show-inheritance:
```

## Data and attribution

Solar position is computed via the
[Skyfield](https://rhodesmill.org/skyfield/) library using NASA JPL's
DE421 planetary ephemeris, which is bundled with hyplan at
`hyplan/data/de421.bsp` so calculations work fully offline.

DE421 is in the public domain. Please cite:

> Folkner, W. M., Williams, J. G., & Boggs, D. H. (2009). *The Planetary
> and Lunar Ephemeris DE 421*. JPL Interoffice Memorandum 343R-08-003,
> NASA Jet Propulsion Laboratory.
> <https://ssd.jpl.nasa.gov/planets/eph_export.html>

> Rhodes, B. (2019). *Skyfield: High precision research-grade positions
> for planets and Earth satellites generator*. Astrophysics Source Code
> Library, ascl:1907.024.

