"""
Custom exception hierarchy for HyPlan.

All HyPlan-specific exceptions inherit from :class:`HyPlanError`, making it
easy to catch any library error with a single ``except HyPlanError`` clause
while still allowing callers to handle specific error categories.

Exception hierarchy::

    HyPlanError (base)
    ├── HyPlanValueError  — invalid argument values
    ├── HyPlanTypeError   — incorrect argument types
    └── HyPlanRuntimeError — runtime / state errors (e.g. data not loaded)
"""

__all__ = [
    "HyPlanError",
    "HyPlanValueError",
    "HyPlanTypeError",
    "HyPlanRuntimeError",
]


class HyPlanError(Exception):
    """Base exception for all HyPlan errors."""


class HyPlanValueError(HyPlanError, ValueError):
    """Raised when an argument has the right type but an invalid value.

    Inherits from both :class:`HyPlanError` and :class:`ValueError` so that
    existing ``except ValueError`` handlers continue to work.
    """


class HyPlanTypeError(HyPlanError, TypeError):
    """Raised when an argument has an unexpected type.

    Inherits from both :class:`HyPlanError` and :class:`TypeError` so that
    existing ``except TypeError`` handlers continue to work.
    """


class HyPlanRuntimeError(HyPlanError, RuntimeError):
    """Raised for runtime or state errors (e.g. required data not loaded).

    Inherits from both :class:`HyPlanError` and :class:`RuntimeError` so that
    existing ``except RuntimeError`` handlers continue to work.
    """
