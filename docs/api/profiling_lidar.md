# Profiling Lidars

Nadir-pointing single-beam atmospheric profiling lidar models. Each
instrument records a vertical column directly beneath the platform; there
is no cross-track swath. Horizontal resolution is set by post-processing
averaging windows times ground speed.

For a worked tutorial covering all three pre-configured instruments, see
[`notebooks/profiling_lidar_planning.ipynb`](https://github.com/ryanpavlick/hyplan/blob/main/notebooks/profiling_lidar_planning.ipynb).

The Doppler wind profiler family (AWP) is **not** part of this hierarchy
because its dual-LOS vector-retrieval geometry needs a different
abstraction. See {doc}`awp`.

## Base class

```{eval-rst}
.. autoclass:: hyplan.instruments.ProfilingLidar
   :members:
   :show-inheritance:
```

## NASA Langley HSRL-2

3-wavelength (355/532/1064 nm) backscatter and extinction lidar with a
Michelson interferometer for the 355 nm channel. Defaults reflect the
TCAP 2012 deployment (Müller et al., 2014); pulse rate, telescope, and
beam divergence are inherited from the HSRL-1 design (Hair et al., 2008).

```{eval-rst}
.. autoclass:: hyplan.instruments.HSRL2
   :show-inheritance:
```

## NASA Langley HALO

Multi-function airborne nadir lidar combining HSRL aerosol/cloud
profiling with water-vapor and methane DIAL/IPDA (Carroll et al., 2022).
Reconfigurable across three transmitter modes (CH₄+HSRL, H₂O+HSRL,
CH₄+H₂O) sharing a common multi-channel receiver. The default
`wavelengths` lists all four channels — override at construction to model
a specific transmitter mode.

```{eval-rst}
.. autoclass:: hyplan.instruments.HALO
   :show-inheritance:
```

## NASA Goddard CPL

Compact 3-wavelength (355/532/1064 nm) backscatter lidar for high-altitude
platforms (ER-2, Global Hawk, WB-57). Uses a 5 kHz-PRF, low-pulse-energy,
photon-counting design — a different operating regime from the NASA
Langley HSRL family. Defaults reflect the standard 1 Hz / 30 m × 200 m
product documented in McGill et al. (2002).

```{eval-rst}
.. autoclass:: hyplan.instruments.CPL
   :show-inheritance:
```

## References

1. Müller, D., Hostetler, C. A., Ferrare, R. A., Burton, S. P., et al.
   (2014). Airborne Multiwavelength High Spectral Resolution Lidar
   (HSRL-2) Observations during TCAP 2012.
   *Atmospheric Measurement Techniques*, 7, 3487-3496.
   <https://doi.org/10.5194/amt-7-3487-2014>

2. Hair, J. W., et al. (2008). Airborne High Spectral Resolution Lidar
   for profiling aerosol optical properties. *Applied Optics*, 47(36),
   6734-6752. <https://doi.org/10.1364/AO.47.006734>

3. Burton, S. P., et al. (2018). Calibration of a high spectral
   resolution lidar using a Michelson interferometer, with data examples
   from ORACLES. *Applied Optics*, 57(21), 6061-6075.
   <https://doi.org/10.1364/AO.57.006061>

4. Carroll, B. J., Nehrir, A. R., Kooi, S. A., et al. (2022).
   Evaluation of the High Altitude Lidar Observatory (HALO) methane
   retrievals during the summer 2019 ACT-America campaign.
   *Atmospheric Measurement Techniques*, 15, 4623-4650.
   <https://doi.org/10.5194/amt-15-4623-2022>

5. McGill, M., et al. (2002). Cloud Physics Lidar: instrument description
   and initial measurement results. *Applied Optics*, 41(18), 3725-3734.
   <https://doi.org/10.1364/AO.41.003725>
