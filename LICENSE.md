# License

Copyright (c) 2024-2026, United States Government, as represented by the Administrator of the National Aeronautics and Space Administration.  
All rights reserved.

The HyPlan airborne science planning software is licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at [http://www.apache.org/licenses/LICENSE-2.0](http://www.apache.org/licenses/LICENSE-2.0).

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  
See the License for the specific language governing permissions and limitations under the License.

## Third-Party Data Notices

HyPlan redistributes the following third-party data files. Per Apache
License 2.0 §4(d), these notices must be preserved in derivative works.

### JPL DE421 Planetary and Lunar Ephemeris

The file `hyplan/data/de421.bsp` is the JPL DE421 planetary ephemeris,
produced by NASA's Jet Propulsion Laboratory and distributed in SPICE SPK
format. DE421 is in the public domain.

Citation:

> Folkner, W. M., Williams, J. G., & Boggs, D. H. (2009). *The Planetary
> and Lunar Ephemeris DE 421*. JPL Interoffice Memorandum 343R-08-003,
> NASA Jet Propulsion Laboratory.

Source:

- <https://ssd.jpl.nasa.gov/planets/eph_export.html>
- <https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/>

HyPlan reads DE421 via the [Skyfield](https://rhodesmill.org/skyfield/)
library:

> Rhodes, B. (2019). *Skyfield: High precision research-grade positions
> for planets and Earth satellites generator*. Astrophysics Source Code
> Library, ascl:1907.024.

### FAA L-Band Radar Exclusion Zones

The file `hyplan/data/faa_radar_exclusion_zones.geojson` is derived from publicly available U.S. Federal Aviation Administration data.
