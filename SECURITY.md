# Security Policy

## Supported versions

HyPlan provides security updates only for the latest 1.x release.
Earlier major-version branches are not maintained.

| Version  | Supported          |
|----------|--------------------|
| 1.1.x    | :white_check_mark: |
| 1.0.x    | :x: (use 1.1.x)    |
| < 1.0    | :x:                |

## Reporting a vulnerability

If you discover a security vulnerability in HyPlan, please report it
**privately** so we can investigate before public disclosure.

Preferred: use GitHub's Private Vulnerability Reporting at
<https://github.com/ryanpavlick/hyplan/security/advisories/new>.

Alternative: email <ryan.p.pavlick@nasa.gov> with the subject line
"hyplan security vulnerability".

Please include:

- A description of the vulnerability and its impact
- Steps to reproduce
- The HyPlan version and Python version where you observed it
- Any suggested fix or mitigation

You can expect:

- An initial acknowledgement within 5 business days
- A status update within 14 days
- Coordinated disclosure once a fix is released

## Scope

This policy covers the HyPlan Python package itself. It does **not**
cover vulnerabilities in third-party dependencies (geopandas, rasterio,
shapely, pint, etc.) — please report those upstream. It also does not
cover external services HyPlan optionally integrates with (OpenAIP, Earth
Engine, NOAA wind providers, FAA NASR, FlightPlanDB).
