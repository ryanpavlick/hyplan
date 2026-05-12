"""JSON ↔ Aircraft profile I/O.

Externalizes every per-aircraft schedule / profile / scalar / source
into ``hyplan/data/aircraft/<short_name>.json`` so calibration outputs
flow directly to canonical data files rather than being hand-copied
into Python literals.

Schema (compact, units in field names):

* ``metadata`` — aircraft_type, tail_number, operator, engine_type,
  calibration_status (str / enum-like).
* ``scalars`` — service_ceiling_ft, approach_speed_kt, range_nmi,
  endurance_hr, useful_payload_lb, stall_speed_cas_kt,
  descent_path_angle_max_deg, climb_path_angle_max_deg (numbers or
  null).
* ``climb_schedule`` / ``cruise_schedule`` / ``descent_schedule`` —
  discriminated union: ``{"type": "tas", "points_ft_kt": [[ft, kt],
  …]}`` or ``{"type": "cas_mach", "cas_kt": …, "mach": …,
  "crossover_ft": …}``.
* ``climb_profile`` / ``descent_profile`` —
  ``{"points_ft_fpm": [[ft, fpm], …], "source": ""}``.
* ``turn_model`` — ``max_bank_deg``, ``max_load_factor``,
  ``bank_by_phase: {climb_deg, cruise_deg, descent_deg, approach_deg}``.
* ``approach_profile`` — null or
  ``{speed_schedule_ft_kt, top_of_approach_agl_ft, glideslope_deg}``.
* ``typical_climb_out`` — null or
  ``{absorbed_in_climb_profile, typical_holds_ft_min,
  typical_overhead_min, notes, explicit_climb_plan_ft_min}``.
* ``confidence`` — ``{climb, cruise, descent, turns}``.
* ``sources`` — list of
  ``{source_type, reference, notes, confidence, url, doi}``.

JSON values are plain JSON-compatible types only (numbers, strings,
lists, dicts, null, bool); units live in field-name suffixes.
"""

from __future__ import annotations

import json
from importlib.resources import files
from pathlib import Path
from typing import Any

from ..units import ureg
from ..exceptions import HyPlanRuntimeError, HyPlanValueError
from ._base import (
    ApproachProfile,
    CasMachSchedule,
    ClimbOutPolicy,
    ClimbPlan,
    PerformanceConfidence,
    PhaseBankAngles,
    SourceRecord,
    SpeedSchedule,
    TasSchedule,
    TurnModel,
    VerticalProfile,
)


__all__ = [
    "dump_aircraft_profile",
    "load_aircraft_profile",
    "profile_path",
    "write_calibrated_profile",
]


def profile_path(short_name: str) -> Path:
    """Return the on-disk path to ``<short_name>.json`` in the bundled
    aircraft data directory.

    Args:
        short_name: Filename stem (e.g. ``"king_air_350"``).

    Returns:
        Absolute :class:`Path` to the bundled JSON file.  The file
        may or may not exist yet — callers can check ``.exists()``
        before reading.
    """
    return Path(str(files("hyplan.data.aircraft").joinpath(f"{short_name}.json")))


# ---------------------------------------------------------------------------
# Load: JSON dict → Aircraft kwargs
# ---------------------------------------------------------------------------


def load_aircraft_profile(short_name: str) -> dict[str, Any]:
    """Load a JSON profile and parse into kwargs ready for
    :class:`hyplan.aircraft.Aircraft`.

    Args:
        short_name: Filename stem of a bundled
            ``hyplan/data/aircraft/<short_name>.json`` file.

    Returns:
        A dict of keyword arguments accepted by
        :meth:`Aircraft.__init__`.  Schedule / profile / nested-object
        fields are reconstructed as their respective dataclasses.

    Raises:
        HyPlanRuntimeError: If the file does not exist or cannot be
            parsed.
    """
    path = profile_path(short_name)
    if not path.exists():
        raise HyPlanRuntimeError(
            f"Aircraft profile not found: {path} "
            f"(short_name={short_name!r})."
        )
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise HyPlanRuntimeError(
            f"Invalid JSON in aircraft profile {path}: {exc}"
        ) from exc
    return _profile_dict_to_kwargs(raw)


def _profile_dict_to_kwargs(raw: dict[str, Any]) -> dict[str, Any]:
    meta = raw.get("metadata", {})
    scalars = raw.get("scalars", {})

    kwargs: dict[str, Any] = {
        "aircraft_type": meta["aircraft_type"],
        "tail_number": meta["tail_number"],
        "operator": meta["operator"],
        "engine_type": meta["engine_type"],
        "calibration_status": meta.get("calibration_status", "uncalibrated"),
        "service_ceiling": _qty(scalars.get("service_ceiling_ft"), ureg.feet),
        "approach_speed": _qty(scalars.get("approach_speed_kt"), ureg.knot),
        "climb_schedule": _schedule_from_dict(raw["climb_schedule"]),
        "cruise_schedule": _schedule_from_dict(raw["cruise_schedule"]),
        "descent_schedule": _schedule_from_dict(raw["descent_schedule"]),
        "climb_profile": _vprofile_from_dict(raw["climb_profile"]),
        "descent_profile": _vprofile_from_dict(raw["descent_profile"]),
        "turn_model": _turn_model_from_dict(raw["turn_model"]),
        "confidence": _confidence_from_dict(raw.get("confidence")),
        "sources": [_source_from_dict(s) for s in raw.get("sources", [])],
    }

    if scalars.get("range_nmi") is not None:
        kwargs["range"] = _qty(scalars["range_nmi"], ureg.nautical_mile)
    if scalars.get("endurance_hr") is not None:
        kwargs["endurance"] = _qty(scalars["endurance_hr"], ureg.hour)
    if scalars.get("useful_payload_lb") is not None:
        kwargs["useful_payload"] = _qty(scalars["useful_payload_lb"], ureg.pound)
    if scalars.get("stall_speed_cas_kt") is not None:
        kwargs["stall_speed_cas"] = _qty(scalars["stall_speed_cas_kt"], ureg.knot)
    if scalars.get("descent_path_angle_max_deg") is not None:
        kwargs["descent_path_angle_max_deg"] = float(
            scalars["descent_path_angle_max_deg"]
        )
    if scalars.get("climb_path_angle_max_deg") is not None:
        kwargs["climb_path_angle_max_deg"] = float(
            scalars["climb_path_angle_max_deg"]
        )

    if raw.get("approach_profile") is not None:
        kwargs["approach_profile"] = _approach_profile_from_dict(
            raw["approach_profile"]
        )
    if raw.get("typical_climb_out") is not None:
        kwargs["typical_climb_out"] = _climb_out_from_dict(raw["typical_climb_out"])

    return kwargs


def _schedule_from_dict(d: dict[str, Any]) -> SpeedSchedule:
    kind = d.get("type", "tas")
    if kind == "tas":
        points = [
            (float(ft) * ureg.feet, float(kt) * ureg.knot)
            for ft, kt in d["points_ft_kt"]
        ]
        return TasSchedule(points=points)
    if kind == "cas_mach":
        return CasMachSchedule(
            cas=float(d["cas_kt"]) * ureg.knot,
            mach=float(d["mach"]),
            crossover_ft=float(d["crossover_ft"]),
        )
    raise HyPlanValueError(
        f"Unknown speed-schedule type: {kind!r} "
        "(expected 'tas' or 'cas_mach')."
    )


def _vprofile_from_dict(d: dict[str, Any]) -> VerticalProfile:
    points = [
        (float(ft) * ureg.feet, float(fpm) * ureg.feet / ureg.minute)
        for ft, fpm in d["points_ft_fpm"]
    ]
    return VerticalProfile(points=points, source=d.get("source", ""))


def _turn_model_from_dict(d: dict[str, Any]) -> TurnModel:
    bbp = d.get("bank_by_phase", {})
    return TurnModel(
        max_bank_deg=float(d.get("max_bank_deg", 30.0)),
        max_load_factor=float(d.get("max_load_factor", 2.5)),
        bank_by_phase=PhaseBankAngles(
            climb_deg=float(bbp.get("climb_deg", 20.0)),
            cruise_deg=float(bbp.get("cruise_deg", 25.0)),
            descent_deg=float(bbp.get("descent_deg", 20.0)),
            approach_deg=float(bbp.get("approach_deg", 15.0)),
        ),
    )


def _confidence_from_dict(d: dict[str, Any] | None) -> PerformanceConfidence:
    if d is None:
        return PerformanceConfidence()
    return PerformanceConfidence(
        climb=float(d.get("climb", 0.5)),
        cruise=float(d.get("cruise", 0.5)),
        descent=float(d.get("descent", 0.5)),
        turns=float(d.get("turns", 0.5)),
    )


def _source_from_dict(d: dict[str, Any]) -> SourceRecord:
    return SourceRecord(
        source_type=d["source_type"],
        reference=d["reference"],
        notes=d.get("notes", ""),
        confidence=float(d.get("confidence", 0.5)),
        url=d.get("url", ""),
        doi=d.get("doi", ""),
    )


def _approach_profile_from_dict(d: dict[str, Any]) -> ApproachProfile:
    return ApproachProfile(
        speed_schedule=TasSchedule(points=[
            (float(ft) * ureg.feet, float(kt) * ureg.knot)
            for ft, kt in d["speed_schedule_ft_kt"]
        ]),
        top_of_approach_agl=float(d["top_of_approach_agl_ft"]) * ureg.feet,
        glideslope_deg=float(d.get("glideslope_deg", 3.0)),
    )


def _climb_out_from_dict(d: dict[str, Any]) -> ClimbOutPolicy:
    plan = d.get("explicit_climb_plan_ft_min")
    explicit = None
    if plan is not None:
        explicit = ClimbPlan(pauses=[
            (float(ft) * ureg.feet, float(mn) * ureg.minute)
            for ft, mn in plan
        ])
    return ClimbOutPolicy(
        absorbed_in_climb_profile=bool(d["absorbed_in_climb_profile"]),
        typical_holds=[
            (float(ft) * ureg.feet, float(mn) * ureg.minute)
            for ft, mn in d.get("typical_holds_ft_min", [])
        ],
        typical_overhead_min=float(d.get("typical_overhead_min", 0.0)),
        notes=d.get("notes", ""),
        explicit_climb_plan=explicit,
    )


def _qty(value: float | int | None, unit: Any) -> Any:
    if value is None:
        return None
    return float(value) * unit


# ---------------------------------------------------------------------------
# Dump: Aircraft → JSON dict (round-trip inverse of load)
# ---------------------------------------------------------------------------


def dump_aircraft_profile(aircraft: Any, path: str | Path) -> None:
    """Serialize an :class:`Aircraft` instance to its JSON profile.

    Inverse of :func:`load_aircraft_profile`.  Used by
    ``notebooks/calibration/<aircraft>/calibrate.py`` to write
    refreshed JSON directly from the fit.

    Args:
        aircraft: An :class:`Aircraft` instance (any subclass).
        path: Destination filesystem path.  Parent directory must
            already exist.
    """
    data = _aircraft_to_profile_dict(aircraft)
    Path(path).write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def write_calibrated_profile(
    short_name: str,
    *,
    path: str | Path | None = None,
    **overrides: Any,
) -> Path:
    """Apply calibrated overrides to a bundled aircraft JSON in place.

    Loads the current ``<short_name>.json`` via
    :func:`load_aircraft_profile`, applies any keyword overrides, rebuilds
    the :class:`Aircraft`, and writes the updated JSON back to disk.
    Fields not present in ``overrides`` are preserved (brochure
    metadata, sources, confidence, etc.), so a calibration pass that
    only refits the climb / cruise / descent schedules can update just
    those without disturbing the rest of the profile.

    Args:
        short_name: Filename stem (e.g. ``"king_air_350"``).
        path: Optional override for the output path.  Defaults to
            :func:`profile_path` which points at the bundled location
            inside the installed package.
        **overrides: Any keyword accepted by :meth:`Aircraft.__init__`,
            e.g. ``climb_schedule=fit.climb_schedule``,
            ``climb_profile=fit.climb_profile``, ``service_ceiling=...``.

    Returns:
        The :class:`Path` that was written.

    Example:
        >>> from hyplan.units import ureg
        >>> write_calibrated_profile(
        ...     "king_air_350",
        ...     service_ceiling=35_000 * ureg.feet,
        ...     climb_schedule=fit.climb_schedule,
        ...     cruise_schedule=fit.cruise_schedule,
        ...     descent_schedule=fit.descent_schedule,
        ...     climb_profile=fit.climb_profile,
        ...     descent_profile=fit.descent_profile,
        ... )
    """
    # Lazy import to avoid module-load circularity with _models.py
    from ._base import Aircraft

    kwargs = load_aircraft_profile(short_name)
    kwargs.update(overrides)
    aircraft = Aircraft(**kwargs)
    out_path = Path(path) if path is not None else profile_path(short_name)
    dump_aircraft_profile(aircraft, out_path)
    return out_path


def _aircraft_to_profile_dict(ac: Any) -> dict[str, Any]:
    out: dict[str, Any] = {
        "metadata": {
            "aircraft_type": ac.aircraft_type,
            "tail_number": ac.tail_number,
            "operator": ac.operator,
            "engine_type": ac.engine_type,
            "calibration_status": ac.calibration_status,
        },
        "scalars": {
            "service_ceiling_ft": _mag(ac.service_ceiling, ureg.feet),
            "approach_speed_kt": _mag(ac.approach_speed, ureg.knot),
            "range_nmi": _mag_or_none(ac.range, ureg.nautical_mile),
            "endurance_hr": _mag_or_none(ac.endurance, ureg.hour),
            "useful_payload_lb": _mag_or_none(ac.useful_payload, ureg.pound),
            "stall_speed_cas_kt": _mag_or_none(ac.stall_speed_cas, ureg.knot),
            "descent_path_angle_max_deg": ac.descent_path_angle_max_deg,
            "climb_path_angle_max_deg": ac.climb_path_angle_max_deg,
        },
        "climb_schedule": _schedule_to_dict(ac.climb_schedule),
        "cruise_schedule": _schedule_to_dict(ac.cruise_schedule),
        "descent_schedule": _schedule_to_dict(ac.descent_schedule),
        "climb_profile": _vprofile_to_dict(ac.climb_profile),
        "descent_profile": _vprofile_to_dict(ac.descent_profile),
        "turn_model": _turn_model_to_dict(ac.turn_model),
        "approach_profile": (
            _approach_profile_to_dict(ac.approach_profile)
            if ac.approach_profile is not None else None
        ),
        "typical_climb_out": (
            _climb_out_to_dict(ac.typical_climb_out)
            if ac.typical_climb_out is not None else None
        ),
        "confidence": _confidence_to_dict(ac.confidence),
        "sources": [_source_to_dict(s) for s in ac.sources],
    }
    return out


def _schedule_to_dict(sched: SpeedSchedule) -> dict[str, Any]:
    if isinstance(sched, CasMachSchedule):
        return {
            "type": "cas_mach",
            "cas_kt": _mag(sched.cas, ureg.knot),
            "mach": float(sched.mach),
            "crossover_ft": float(sched.crossover_ft),
        }
    if isinstance(sched, TasSchedule):
        return {
            "type": "tas",
            "points_ft_kt": [
                [_mag(alt, ureg.feet), _mag(spd, ureg.knot)]
                for alt, spd in sched.points
            ],
        }
    raise HyPlanValueError(f"Unsupported schedule type: {type(sched).__name__}")


def _vprofile_to_dict(vp: VerticalProfile) -> dict[str, Any]:
    return {
        "points_ft_fpm": [
            [_mag(alt, ureg.feet), _mag(rate, ureg.feet / ureg.minute)]
            for alt, rate in vp.points
        ],
        "source": vp.source,
    }


def _turn_model_to_dict(tm: TurnModel) -> dict[str, Any]:
    return {
        "max_bank_deg": float(tm.max_bank_deg),
        "max_load_factor": float(tm.max_load_factor),
        "bank_by_phase": {
            "climb_deg": float(tm.bank_by_phase.climb_deg),
            "cruise_deg": float(tm.bank_by_phase.cruise_deg),
            "descent_deg": float(tm.bank_by_phase.descent_deg),
            "approach_deg": float(tm.bank_by_phase.approach_deg),
        },
    }


def _approach_profile_to_dict(ap: ApproachProfile) -> dict[str, Any]:
    return {
        "speed_schedule_ft_kt": [
            [_mag(alt, ureg.feet), _mag(spd, ureg.knot)]
            for alt, spd in ap.speed_schedule.points
        ],
        "top_of_approach_agl_ft": _mag(ap.top_of_approach_agl, ureg.feet),
        "glideslope_deg": float(ap.glideslope_deg),
    }


def _climb_out_to_dict(co: ClimbOutPolicy) -> dict[str, Any]:
    explicit = None
    if co.explicit_climb_plan is not None:
        explicit = [
            [_mag(alt, ureg.feet), _mag(dur, ureg.minute)]
            for alt, dur in co.explicit_climb_plan.pauses
        ]
    return {
        "absorbed_in_climb_profile": bool(co.absorbed_in_climb_profile),
        "typical_holds_ft_min": [
            [_mag(alt, ureg.feet), _mag(dur, ureg.minute)]
            for alt, dur in co.typical_holds
        ],
        "typical_overhead_min": float(co.typical_overhead_min),
        "notes": co.notes,
        "explicit_climb_plan_ft_min": explicit,
    }


def _confidence_to_dict(c: PerformanceConfidence) -> dict[str, Any]:
    return {
        "climb": float(c.climb),
        "cruise": float(c.cruise),
        "descent": float(c.descent),
        "turns": float(c.turns),
    }


def _source_to_dict(s: SourceRecord) -> dict[str, Any]:
    return {
        "source_type": s.source_type,
        "reference": s.reference,
        "notes": s.notes,
        "confidence": float(s.confidence),
        "url": s.url,
        "doi": s.doi,
    }


def _mag(q: Any, unit: Any) -> float:
    return float(q.m_as(unit))


def _mag_or_none(q: Any, unit: Any) -> float | None:
    return None if q is None else float(q.m_as(unit))
