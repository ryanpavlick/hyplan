"""Nadir-pointing single-beam profiling lidar models.

Defines the :class:`ProfilingLidar` base class for airborne lidars that
record a vertical column directly beneath the platform. Pre-configured
instruments include NASA Langley's HSRL-2 (3-wavelength backscatter/
extinction) and HALO (4-wavelength multi-function HSRL + water-vapor and
methane DIAL/IPDA), and NASA Goddard's CPL (3-wavelength high-PRF
photon-counting cloud/cirrus profiler).

Doppler wind profilers (e.g., AWP) are NOT modeled here — their dual-LOS
geometry needs a different abstraction; see :class:`AerosolWindProfiler`
in :mod:`hyplan.instruments.awp`.

References
----------
Müller, D., et al. (2014). Atmos. Meas. Tech., 7, 3487-3496.
https://doi.org/10.5194/amt-7-3487-2014

Hair, J. W., et al. (2008). Applied Optics, 47(36), 6734-6752.
https://doi.org/10.1364/AO.47.006734

Burton, S. P., et al. (2018). Applied Optics, 57(21), 6061-6075.
https://doi.org/10.1364/AO.57.006061

Carroll, B. J., Nehrir, A. R., Kooi, S. A., et al. (2022). Atmos. Meas.
Tech., 15, 4623-4650. https://doi.org/10.5194/amt-15-4623-2022

McGill, M., Hlavka, D., Hart, W., Scott, V. S., Spinhirne, J., and
Schmid, B. (2002). Cloud Physics Lidar: instrument description and
initial measurement results. Applied Optics, 41(18), 3725-3734.
https://doi.org/10.1364/AO.41.003725
"""

from __future__ import annotations

from typing import Iterable, Optional

from pint import Quantity

from ..exceptions import HyPlanTypeError, HyPlanValueError
from ..units import ureg
from ._base import Sensor

__all__ = ["ProfilingLidar", "HSRL2", "HALO", "CPL"]


def _as_quantity(value, unit: str, label: str) -> Quantity:
    """Normalize *value* to a quantity in *unit*."""
    if isinstance(value, Quantity):
        return value.to(unit)
    if isinstance(value, (int, float)):
        return ureg.Quantity(float(value), unit)
    raise HyPlanTypeError(f"{label} must be numeric or a pint.Quantity, got {type(value)}")


class ProfilingLidar(Sensor):
    """Base class for nadir-pointing single-beam airborne profiling lidars.

    Captures the geometry and timing parameters that matter for flight
    planning: laser wavelengths, pulse rate, telescope/beam optics, vertical
    bin resolution, and FPGA-averaged sampling rate. Helpers compute
    footprint diameter from altitude (when beam divergence is known) and
    effective horizontal resolution from a post-processing averaging window.
    """

    def __init__(
        self,
        name: str,
        *,
        wavelengths: Iterable[Quantity],
        pulse_rate: Quantity,
        telescope_diameter: Quantity,
        vertical_resolution: Quantity,
        sampling_rate: Quantity,
        beam_divergence: Optional[Quantity] = None,
        native_horizontal_resolution: Optional[Quantity] = None,
    ):
        super().__init__(name)
        self.wavelengths = tuple(
            _as_quantity(w, "nanometer", "wavelength") for w in wavelengths
        )
        self.pulse_rate = _as_quantity(pulse_rate, "hertz", "pulse_rate")
        self.telescope_diameter = _as_quantity(
            telescope_diameter, "centimeter", "telescope_diameter"
        )
        self.vertical_resolution = _as_quantity(
            vertical_resolution, "meter", "vertical_resolution"
        )
        self.sampling_rate = _as_quantity(sampling_rate, "hertz", "sampling_rate")
        self.beam_divergence = (
            _as_quantity(beam_divergence, "milliradian", "beam_divergence")
            if beam_divergence is not None
            else None
        )
        self.native_horizontal_resolution = (
            _as_quantity(
                native_horizontal_resolution, "meter", "native_horizontal_resolution"
            )
            if native_horizontal_resolution is not None
            else None
        )

        if len(self.wavelengths) == 0:
            raise HyPlanValueError("wavelengths must contain at least one wavelength")
        for w in self.wavelengths:
            if w.magnitude <= 0:
                raise HyPlanValueError("wavelengths must all be positive")
        for label, q in (
            ("pulse_rate", self.pulse_rate),
            ("telescope_diameter", self.telescope_diameter),
            ("vertical_resolution", self.vertical_resolution),
            ("sampling_rate", self.sampling_rate),
        ):
            if q.magnitude <= 0:
                raise HyPlanValueError(f"{label} must be positive")
        if self.beam_divergence is not None and self.beam_divergence.magnitude <= 0:
            raise HyPlanValueError("beam_divergence must be positive")
        if (
            self.native_horizontal_resolution is not None
            and self.native_horizontal_resolution.magnitude <= 0
        ):
            raise HyPlanValueError("native_horizontal_resolution must be positive")

    def footprint_diameter(self, altitude_agl: Quantity) -> Quantity:
        """Laser footprint diameter on the ground at the given altitude AGL.

        Requires ``beam_divergence`` to have been set at construction; raises
        ``HyPlanValueError`` otherwise.
        """
        if self.beam_divergence is None:
            raise HyPlanValueError(
                f"footprint_diameter requires beam_divergence to be set on "
                f"{type(self).__name__}; no published default for this instrument"
            )
        altitude = _as_quantity(altitude_agl, "meter", "altitude_agl")
        return (altitude * self.beam_divergence.to("radian").magnitude).to("meter")  # type: ignore[no-any-return]

    def horizontal_resolution(
        self, ground_speed: Quantity, averaging_time: Quantity
    ) -> Quantity:
        """Effective horizontal resolution for a post-processing averaging window."""
        speed = _as_quantity(ground_speed, "meter / second", "ground_speed")
        dt = _as_quantity(averaging_time, "second", "averaging_time")
        return (speed * dt).to("meter")  # type: ignore[no-any-return]

    def pulses_per_profile(self, averaging_time: Quantity) -> int:
        """Number of laser pulses averaged into one profile at the given window."""
        dt = _as_quantity(averaging_time, "second", "averaging_time")
        return int((self.pulse_rate * dt).to("dimensionless").magnitude)


class HSRL2(ProfilingLidar):
    """NASA Langley High Spectral Resolution Lidar, second generation.

    3-wavelength (355/532/1064 nm) backscatter and extinction lidar with a
    Michelson interferometer for the 355 nm channel. Successor to HSRL-1
    (Hair et al., 2008); inherits the same Fibertek-built Nd:YAG laser,
    16-inch Newtonian telescope, and 200 Hz pulse rate, extended with a
    third wavelength channel and Michelson interferometer (Burton et al.,
    2018). Defaults reflect the airborne TCAP 2012 deployment documented
    in Müller et al. (2014).
    """

    def __init__(
        self,
        name: str = "HSRL-2",
        *,
        wavelengths: Iterable[Quantity] = (
            355 * ureg.nanometer,
            532 * ureg.nanometer,
            1064 * ureg.nanometer,
        ),
        pulse_rate: Quantity = 200 * ureg.hertz,
        telescope_diameter: Quantity = 40.6 * ureg.centimeter,
        beam_divergence: Optional[Quantity] = 0.8 * ureg.milliradian,
        vertical_resolution: Quantity = 15 * ureg.meter,
        sampling_rate: Quantity = 2 * ureg.hertz,
        native_horizontal_resolution: Optional[Quantity] = 100 * ureg.meter,
    ):
        super().__init__(
            name,
            wavelengths=wavelengths,
            pulse_rate=pulse_rate,
            telescope_diameter=telescope_diameter,
            beam_divergence=beam_divergence,
            vertical_resolution=vertical_resolution,
            sampling_rate=sampling_rate,
            native_horizontal_resolution=native_horizontal_resolution,
        )


class HALO(ProfilingLidar):
    """NASA Langley High Altitude Lidar Observatory.

    Multi-function airborne nadir lidar combining HSRL aerosol/cloud
    profiling with water-vapor and methane DIAL/IPDA. Uses a higher
    1 kHz-PRF laser than HSRL-2 and adds 935 nm (H2O) and 1645 nm (CH4)
    channels.

    HALO is reconfigurable across three transmitter modes sharing a
    common multi-channel receiver:

    - CH4 DIAL + HSRL    (active: 532, 1064, 1645 nm)
    - H2O DIAL + HSRL    (active: 532, 935, 1064 nm)
    - CH4 DIAL + H2O DIAL (active: 935, 1064, 1645 nm)

    The default ``wavelengths`` lists all four channels; override to model
    a specific mode. Planning geometry is identical across modes — this
    class does not enforce which combinations are physically simultaneous.

    Defaults reflect the ACT-America 2019 NASA C-130 deployment documented
    in Carroll et al. (2022). Beam divergence is not published for HALO and
    is left unset by default.
    """

    def __init__(
        self,
        name: str = "HALO",
        *,
        wavelengths: Iterable[Quantity] = (
            532 * ureg.nanometer,
            935 * ureg.nanometer,
            1064 * ureg.nanometer,
            1645 * ureg.nanometer,
        ),
        pulse_rate: Quantity = 1 * ureg.kilohertz,
        telescope_diameter: Quantity = 40 * ureg.centimeter,
        beam_divergence: Optional[Quantity] = None,
        vertical_resolution: Quantity = 15 * ureg.meter,
        sampling_rate: Quantity = 2 * ureg.hertz,
        native_horizontal_resolution: Optional[Quantity] = None,
    ):
        super().__init__(
            name,
            wavelengths=wavelengths,
            pulse_rate=pulse_rate,
            telescope_diameter=telescope_diameter,
            beam_divergence=beam_divergence,
            vertical_resolution=vertical_resolution,
            sampling_rate=sampling_rate,
            native_horizontal_resolution=native_horizontal_resolution,
        )


class CPL(ProfilingLidar):
    """NASA Goddard Cloud Physics Lidar.

    Compact 3-wavelength (355/532/1064 nm) backscatter lidar designed for
    high-altitude platforms (ER-2, Global Hawk, WB-57). Uses a 5 kHz-PRF
    low-pulse-energy laser with photon-counting detection — a different
    operating regime from NASA Langley's HSRL family. Primary science is
    cirrus and aerosol profiling.

    Defaults reflect the standard product (1 Hz, 30 m vertical x 200 m
    horizontal); the raw 10 Hz / 20 m horizontal product is also available
    operationally but not modeled as the default.

    The 100 microradian value is McGill 2002's receiver field of view.
    CPL's transmit divergence is matched to the receive FOV by design, so
    the same value is used here for ``beam_divergence``.

    References
    ----------
    McGill, M., et al. (2002). Cloud Physics Lidar: instrument description
    and initial measurement results. Applied Optics, 41(18), 3725-3734.
    https://doi.org/10.1364/AO.41.003725
    """

    def __init__(
        self,
        name: str = "CPL",
        *,
        wavelengths: Iterable[Quantity] = (
            355 * ureg.nanometer,
            532 * ureg.nanometer,
            1064 * ureg.nanometer,
        ),
        pulse_rate: Quantity = 5 * ureg.kilohertz,
        telescope_diameter: Quantity = 20 * ureg.centimeter,
        beam_divergence: Optional[Quantity] = 100 * ureg.microradian,
        vertical_resolution: Quantity = 30 * ureg.meter,
        sampling_rate: Quantity = 1 * ureg.hertz,
        native_horizontal_resolution: Optional[Quantity] = 200 * ureg.meter,
    ):
        super().__init__(
            name,
            wavelengths=wavelengths,
            pulse_rate=pulse_rate,
            telescope_diameter=telescope_diameter,
            beam_divergence=beam_divergence,
            vertical_resolution=vertical_resolution,
            sampling_rate=sampling_rate,
            native_horizontal_resolution=native_horizontal_resolution,
        )
