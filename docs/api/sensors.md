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
.. autoclass:: hyplan.instruments.GLiHT_Thermal
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
```

## Profiling lidars

Nadir-pointing single-beam atmospheric profilers (no cross-track swath).
For a detailed walk-through, see {doc}`profiling_lidar`.

```{eval-rst}
.. autoclass:: hyplan.instruments.ProfilingLidar
   :show-inheritance:

.. autoclass:: hyplan.instruments.HSRL2
   :show-inheritance:

.. autoclass:: hyplan.instruments.HALO
   :show-inheritance:

.. autoclass:: hyplan.instruments.CPL
   :show-inheritance:
```

## Doppler wind lidar

Dual-line-of-sight profiler for vector wind retrieval. See {doc}`awp`
for planning helpers.

```{eval-rst}
.. autoclass:: hyplan.instruments.AerosolWindProfiler
```

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

```{eval-rst}
.. autoclass:: hyplan.instruments.FrameCamera
   :members:
   :show-inheritance:

.. autoclass:: hyplan.instruments.MultiCameraRig
   :members:
   :show-inheritance:
```

## Factory function

```{eval-rst}
.. autofunction:: hyplan.instruments.create_sensor
```
