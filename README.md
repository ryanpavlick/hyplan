# Project Overview

HyPlan is an open-source Python library for planning airborne remote sensing campaigns. 

---

## Repository Contents

### Core Modules

- **airports.py**: Functions for locating and analyzing nearby airports for mission logistics.
- **sun.py**: Functions to calculate solar position for mission planning.
- **glint.py**: Functions to predict solar glint angles based on sensor view angles and solar position.
- **flight_line.py**: Functions to generate and modify flight lines.
- **flight_box.py**: Functions for generating multiple flight lines that cover a geographic area.
- **sensors.py**: Defines sensor characterisitics.
- **terrain.py**: Functions for downloading terrain DEM data and calculating where the sensor field of view intersects the ground.
- **swath.py**: Functions to compute swath coverage based on sensor field of view, altitude, and terrain elevation.
- **geometry.py**: Utility functions for geometric calculations essential to flight planning and sensor modeling.
- **units.py**: Utility functions for unit conversions and handling.
- **download.py**: Utility functions for downloading necessary datasets or dependencies.


### Configuration and Setup

- **setup.py**: Script for installing the package.
- **pyproject.toml**: Build configuration file.
- **requirements.txt**: Lists Python dependencies for the project.
- **LICENSE.md**: Licensing details.

### Documentation

- **README.md**: Overview and instructions for the repository.

---

## Installation

To set up the environment, clone the repository and install the dependencies:

```bash
# Clone the repository
git clone https://github.com/ryanpavlick/hyplan
cd hyplan

# Install dependencies
pip uninstall -y hyplan; pip install -e .
```

Or using conda/mamba, e.g. to create a new environment and install the dependencies

```
# Assuming you have downloaded the repo, and cd into it (see above)
mamba env create --file hyplan.yml
mamba activate hyplan
pip install -e .
```
This will create a new conda environment names "hyplan" with the dependencies and install the hyplan library within that environment

---

## Usage

### Example: Planning a Flight Mission

Need to add material here

## Contributing

Contributions are welcome! If you have suggestions or find issues, please open an issue or submit a pull request.

---

## License

HyPlan is licensed under the Apache License, Version 2.0. See the `LICENSE.md` file for details.

---

## Contact

For inquiries or further information, please contact Ryan Pavlick (ryan.p.pavlick@nasa.gov).
