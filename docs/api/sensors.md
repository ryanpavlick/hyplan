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

## Profiling lidar

```{eval-rst}
.. autoclass:: hyplan.instruments.ProfilingLidar

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
