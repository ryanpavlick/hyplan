"""Sensor name registry and factory.

Central name → factory map for every sensor HyPlan exposes by string.
Lives separately from any one instrument module so new instruments can
register themselves without `line_scanner.py` (or any other module) having
to know about them.

Public API:

* :data:`SENSOR_REGISTRY` — `dict[str, Callable[[], Sensor]]`. Keys are
  canonical names *and* their aliases; every value is a zero-arg factory
  callable. For line-scanner subclasses the value is the class itself
  (classes are callables); for reference singletons it is a lambda that
  returns the shared object.
* :func:`register_sensor` — register one factory under a canonical name
  plus zero or more aliases. Subsequent registrations under the same
  name overwrite (no warning); use that if you need to swap a default.
* :func:`create_sensor` — look up a name and call the factory.

Registration is done explicitly from
:mod:`hyplan.instruments.__init__` after all instrument modules have
been imported. That keeps the registry module itself import-free of
the concrete instrument modules and avoids the circular-import risk of
inline registration inside each module.
"""

from __future__ import annotations

import difflib
from collections.abc import Callable, Iterable

from ..exceptions import HyPlanValueError
from ._base import Sensor

__all__ = [
    "SENSOR_REGISTRY",
    "SensorFactory",
    "create_sensor",
    "register_sensor",
]


SensorFactory = Callable[[], Sensor]
"""Zero-arg callable that returns a :class:`Sensor` instance.

Line-scanner subclasses satisfy this directly (calling the class returns
an instance). Reference singletons use a `lambda: SINGLETON` wrapper so
the registry value is always callable.
"""


SENSOR_REGISTRY: dict[str, SensorFactory] = {}
"""Unified name → factory map. Aliases are stored as separate keys
sharing the same factory value."""


def register_sensor(
    name: str,
    factory: SensorFactory,
    *,
    aliases: Iterable[str] = (),
) -> None:
    """Register ``factory`` under ``name`` and any ``aliases``.

    Every name resolves to the same factory callable, so
    ``create_sensor(name) is create_sensor(any_alias)`` for
    singleton-backed factories.
    """
    SENSOR_REGISTRY[name] = factory
    for alias in aliases:
        SENSOR_REGISTRY[alias] = factory


def create_sensor(sensor_type: str) -> Sensor:
    """Construct a sensor by registered name.

    Args:
        sensor_type: Canonical name or alias registered via
            :func:`register_sensor`. Look-up is case- and
            whitespace-sensitive.

    Returns:
        A :class:`Sensor` instance. Singleton-backed names return the
        same shared object on every call; class-backed names return a
        fresh instance each call.

    Raises:
        HyPlanValueError: If ``sensor_type`` is not registered.
    """
    try:
        factory = SENSOR_REGISTRY[sensor_type]
    except KeyError as exc:
        available = sorted(SENSOR_REGISTRY)
        msg = f"Unknown sensor type: {sensor_type!r}. Registered names: {available}"
        close = difflib.get_close_matches(sensor_type, available, n=1)
        if close:
            msg += f". Did you mean {close[0]!r}?"
        raise HyPlanValueError(msg) from exc
    return factory()
