# Instruments

All sensor classes live under the `hyplan.instruments` subpackage and are
re-exported from the top-level `hyplan` namespace for convenience.

## Base class

```{eval-rst}
.. autoclass:: hyplan.instruments.Sensor
```

## Line scanners

```{eval-rst}
.. autoclass:: hyplan.instruments.LineScanner
.. autoclass:: hyplan.instruments.AVIRISClassic
.. autoclass:: hyplan.instruments.AVIRISNextGen
.. autoclass:: hyplan.instruments.AVIRIS3
.. autoclass:: hyplan.instruments.AVIRIS5
.. autoclass:: hyplan.instruments.HyTES
.. autoclass:: hyplan.instruments.PRISM
.. autoclass:: hyplan.instruments.MASTER
.. autoclass:: hyplan.instruments.GLiHT_VNIR
.. autoclass:: hyplan.instruments.GLiHT_SIF
.. autoclass:: hyplan.instruments.GCAS_UV_Vis
.. autoclass:: hyplan.instruments.GCAS_VNIR
.. autoclass:: hyplan.instruments.eMAS
.. autoclass:: hyplan.instruments.PICARD
```

## LVIS lidar

```{eval-rst}
.. autoclass:: hyplan.instruments.LVISLens
   :members:

.. autoclass:: hyplan.instruments.LVIS
   :members:
   :show-inheritance:

.. autodata:: hyplan.instruments.LVIS_LENS_NARROW
   :no-value:
.. autodata:: hyplan.instruments.LVIS_LENS_MEDIUM
   :no-value:
.. autodata:: hyplan.instruments.LVIS_LENS_WIDE
   :no-value:
```

The three pre-configured lens instances are also exposed via the
``LVIS_LENSES`` mapping (keys ``"narrow"``, ``"medium"``, ``"wide"``)
for parameterising tests or campaigns by lens name.

## Airborne laser scanners (topographic ALS)

Generic {class}`~hyplan.instruments.ALSLidar` class for rotating-mirror
discrete-return topographic lidar (RIEGL VQ-series, Leica TerrainMapper,
Optech Galaxy, Phoenix LiDAR Ranger).  Pre-configured reference instance:
{data}`~hyplan.instruments.RIEGL_VQ_480II` (RIEGL VQ-480 II at the
1200 kHz operating point).  Worked planning example:
[`notebooks/als_lidar_planning.ipynb`](../../notebooks/als_lidar_planning.ipynb).

```{eval-rst}
.. autoclass:: hyplan.instruments.ALSLidar
   :members:
   :show-inheritance:

.. autoexception:: hyplan.instruments.ContiguityError
   :show-inheritance:

.. autodata:: hyplan.instruments.RIEGL_VQ_480II
   :no-value:
```

Multi-lidar rig (analog to {class}`~hyplan.instruments.MultiCameraRig`)
for systems that fly two or more identical scanning lidars at known
mount orientations.  Supports both forward/backward pitch tilt
(multi-angle returns, same swath — NASA G-LiHT pattern) and
left/right roll tilt (wider combined swath).  The pre-configured
{data}`~hyplan.instruments.GLIHT_DUAL_VQ_480I` reference instance models
the G-LiHT 2017+ dual VQ-480i configuration.

```{eval-rst}
.. autoclass:: hyplan.instruments.LidarMount
   :members:

.. autoclass:: hyplan.instruments.MultiALSLidarRig
   :members:
   :show-inheritance:

.. autodata:: hyplan.instruments.GLIHT_DUAL_VQ_480I
   :no-value:
```

## Sampling systems

Event-based sampling (single release per drop, drifts to splash through
wind), not a swath geometry.  The
{class}`~hyplan.instruments.DropsondeSystem` class and helpers are
documented on the dedicated {doc}`dropsonde` page.

## Profiling lidars

Nadir-pointing single-beam atmospheric profilers (no cross-track swath):
{class}`~hyplan.instruments.ProfilingLidar` base class plus three
pre-configured instruments — {class}`~hyplan.instruments.HSRL2`,
{class}`~hyplan.instruments.HALO`, and {class}`~hyplan.instruments.CPL`.
Detailed signatures and references are documented on the dedicated
{doc}`profiling_lidar` page.

## Doppler wind lidar

{class}`~hyplan.instruments.AerosolWindProfiler` is a dual-line-of-sight
profiler for vector wind retrieval. Detailed signature and planning
helpers are documented on the dedicated {doc}`awp` page.

## Radar

```{eval-rst}
.. autoclass:: hyplan.instruments.SidelookingRadar
   :members:
   :show-inheritance:

.. autoclass:: hyplan.instruments.UAVSAR_Lband
.. autoclass:: hyplan.instruments.UAVSAR_Pband
.. autoclass:: hyplan.instruments.UAVSAR_Kaband

.. autoclass:: hyplan.instruments.RadarExclusionConflict

.. autofunction:: hyplan.instruments.check_lband_radar_exclusions
```

## Frame camera

For nadir survey planning the relevant frame-rate limit is
{meth}`~hyplan.instruments.FrameCamera.max_ground_speed_for_overlap`
— the maximum ground speed that maintains a requested forward
overlap at the configured frame rate.
{meth}`~hyplan.instruments.FrameCamera.critical_ground_speed` reports
the (typically much tighter, rarely-binding) one-pixel-motion-per-frame
ground speed and is retained for backward compatibility.

```{eval-rst}
.. autoclass:: hyplan.instruments.FrameCamera
   :members:
   :show-inheritance:

.. autoclass:: hyplan.instruments.MultiCameraRig
   :members:
   :show-inheritance:

.. autodata:: hyplan.instruments.GLIHT_HRAC
   :no-value:

.. autodata:: hyplan.instruments.GLIHT_THERMAL
   :no-value:
```

## Factory function

```{eval-rst}
.. autofunction:: hyplan.instruments.create_sensor
```
