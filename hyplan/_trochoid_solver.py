"""Sachdev et al. (2023) trochoidal Dubins solver — standalone module.

Port of github.com/castacks/trochoids to Python.
This file can be tested independently before integrating into dubins3d.py.
"""

import math
import numpy as np


def _mod2pi(angle: "float | np.floating | np.ndarray") -> "float | np.floating | np.ndarray":
    return angle % (2.0 * math.pi)


_EPS = 1e-6
_M2PI = 2 * math.pi


def solve_trochoid(
    qi: np.ndarray,
    qf: np.ndarray,
    rhomin: float,
    airspeed: float,
    wind_u: float,
    wind_v: float,
) -> dict:
    """Solve for the time-optimal trochoidal BSB path.

    Args:
        qi: [x, y, heading] start in inertial frame (meters, radians).
        qf: [x, y, heading] goal in inertial frame.
        rhomin: Minimum turn radius (meters).
        airspeed: True airspeed (m/s).
        wind_u: Eastward wind component (m/s).
        wind_v: Northward wind component (m/s).

    Returns:
        dict with keys: total_time, t1, t2, del1, del2, phi1, phi2,
        xt10, yt10, xt20, yt20, cos_w, sin_w, t2pi, vw, psi_w
    """
    Va = airspeed
    vw = math.sqrt(wind_u**2 + wind_v**2)
    psi_w = _mod2pi(math.atan2(wind_v, wind_u))
    w = Va / rhomin
    t2pi = _M2PI / w

    cos_w = math.cos(psi_w)
    sin_w = math.sin(psi_w)

    # Wind frame
    x0 = qi[0] * cos_w + qi[1] * sin_w
    y0 = -qi[0] * sin_w + qi[1] * cos_w
    xf = qf[0] * cos_w + qf[1] * sin_w
    yf = -qf[0] * sin_w + qf[1] * cos_w

    phi1_base = _mod2pi(qi[2])
    phi2_base = _mod2pi(qf[2])

    best = {
        "total_time": math.inf,
        "t1": 0.0, "t2": 0.0,
        "del1": 1.0, "del2": 1.0,
        "phi1": 0.0, "phi2": 0.0,
        "xt10": 0.0, "yt10": 0.0,
        "xt20": 0.0, "yt20": 0.0,
        "cos_w": cos_w, "sin_w": sin_w,
        "t2pi": t2pi, "vw": vw, "psi_w": psi_w,
    }

    for del1, del2 in [(1, 1), (1, -1), (-1, 1), (-1, -1)]:
        phi1 = math.fmod(phi1_base - psi_w, _M2PI)
        phi2 = math.fmod(phi2_base - psi_w - del2 * _M2PI, _M2PI)

        xt10 = x0 - (Va / (del1 * w)) * math.sin(phi1)
        yt10 = y0 + (Va / (del1 * w)) * math.cos(phi1)
        xt20 = xf - (Va / (del2 * w)) * math.sin(phi2 + del2 * _M2PI) - vw * t2pi
        yt20 = yf + (Va / (del2 * w)) * math.cos(phi2 + del2 * _M2PI)

        E = Va * (vw * (del1 - del2) / (del1 * del2 * w) - (yt20 - yt10))
        G = vw * (yt20 - yt10) + Va**2 * (del2 - del1) / (del1 * del2 * w)

        if abs(del1 - del2) < _EPS:
            # Analytical: RSR or LSL
            _try_analytical(
                Va, vw, w, t2pi, del1, del2,
                phi1, phi2, xt10, yt10, xt20, yt20, E, G,
                cos_w, sin_w, best,
            )
        else:
            # Exhaustive NR: RSL or LSR
            _try_numerical(
                Va, vw, w, t2pi, del1, del2,
                phi1, phi2, xt10, yt10, xt20, yt20, E, G,
                cos_w, sin_w, best,
            )

    if not math.isfinite(best["total_time"]):
        best["total_time"] = 0.0

    return best


def _try_analytical(Va, vw, w, t2pi, del1, del2, phi1, phi2,
                    xt10, yt10, xt20, yt20, E, G, cos_w, sin_w, best):
    for k in range(-3, 3):
        phi_diff = math.fmod(phi1 - phi2, _M2PI) + 2 * k * math.pi
        denom = xt20 - xt10 + vw * phi_diff / (del2 * w)
        if abs(denom) < _EPS:
            continue
        alpha = math.atan2(yt20 - yt10, denom)
        sin_arg = vw / Va * math.sin(alpha)
        if abs(sin_arg) > 1.0:
            continue

        t1 = (t2pi / (del1 * _M2PI)) * (math.asin(sin_arg) + alpha - phi1)
        if t1 < 0 or t1 > t2pi:
            t1 -= t2pi * math.floor(t1 / t2pi)

        t2 = t1 + phi_diff / (del2 * w)
        if t2 <= -t2pi or t2 > t2pi:
            continue

        T = _compute_total_time(Va, vw, w, t2pi, del1, del2, phi1, phi2,
                                xt10, yt10, xt20, yt20, t1, t2, alpha)
        if T is not None and 0 < T < best["total_time"]:
            _update_best(best, T, t1, t2, del1, del2, phi1, phi2,
                         xt10, yt10, xt20, yt20, cos_w, sin_w, t2pi, vw)


def _try_numerical(Va, vw, w, t2pi, del1, del2, phi1, phi2,
                   xt10, yt10, xt20, yt20, E, G, cos_w, sin_w, best):
    step = 2 * t2pi / 60.0  # 60 guesses instead of 360 (6x faster)
    for k in range(-2, 2):   # k range -2..1 instead of -3..2
        roots = []
        t_guess = 0.0
        while t_guess < 2 * t2pi:
            t1_nr = _newton_raphson(
                t_guess, k, Va, vw, w, del1, del2,
                phi1, phi2, xt10, xt20, yt10, yt20, E, G)
            t_guess += step
            if (0 <= t1_nr < 2 * t2pi
                    and abs(_func(t1_nr, k, Va, vw, w, del1, del2,
                                  phi1, phi2, xt10, xt20, yt10, yt20, E, G)) < _EPS):
                roots.append(t1_nr)

        # Deduplicate
        roots.sort()
        unique = []
        for r in roots:
            if not unique or abs(r - unique[-1]) > _EPS:
                unique.append(r)

        for t1 in unique:
            phi_diff = math.fmod(phi1 - phi2, _M2PI) + 2 * k * math.pi
            t2 = (del1 / del2) * t1 + phi_diff / (del2 * w)
            if t2 <= -t2pi or t2 > t2pi:
                continue

            # Direction check
            x1t2 = (Va / (del1 * w)) * math.sin(del1 * w * t1 + phi1) + vw * t1 + xt10
            y1t2 = -(Va / (del1 * w)) * math.cos(del1 * w * t1 + phi1) + yt10
            x2t2 = (Va / (del2 * w)) * math.sin(del2 * w * t2 + phi2) + vw * t2 + xt20
            y2t2 = -(Va / (del2 * w)) * math.cos(del2 * w * t2 + phi2) + yt20

            alpha_check = math.atan2(
                Va * math.sin(del1 * w * t1 + phi1),
                Va * math.cos(del1 * w * t1 + phi1) + vw)
            seg_dir = _mod2pi(math.atan2(y2t2 - y1t2, x2t2 - x1t2))
            if abs(_mod2pi(seg_dir) - _mod2pi(alpha_check)) > math.pi / 2:
                continue

            T = _compute_total_time(Va, vw, w, t2pi, del1, del2, phi1, phi2,
                                    xt10, yt10, xt20, yt20, t1, t2, None)
            if T is not None and 0 < T < best["total_time"]:
                _update_best(best, T, t1, t2, del1, del2, phi1, phi2,
                             xt10, yt10, xt20, yt20, cos_w, sin_w, t2pi, vw)


def _compute_total_time(Va, vw, w, t2pi, del1, del2, phi1, phi2,
                        xt10, yt10, xt20, yt20, t1, t2, alpha):
    x1t2 = (Va / (del1 * w)) * math.sin(del1 * w * t1 + phi1) + vw * t1 + xt10
    y1t2 = -(Va / (del1 * w)) * math.cos(del1 * w * t1 + phi1) + yt10
    x2t2 = (Va / (del2 * w)) * math.sin(del2 * w * t2 + phi2) + vw * t2 + xt20
    y2t2 = -(Va / (del2 * w)) * math.cos(del2 * w * t2 + phi2) + yt20

    # Direction validation for analytical case
    if alpha is not None:
        seg_dir = _mod2pi(math.atan2(y2t2 - y1t2, x2t2 - x1t2))
        if abs(_mod2pi(seg_dir) - _mod2pi(alpha)) > math.pi / 2:
            return None

    xt2dot = Va * math.cos(del2 * w * t2 + phi2) + vw
    yt2dot = Va * math.sin(del2 * w * t2 + phi2)
    gs = math.sqrt(xt2dot**2 + yt2dot**2)
    sd = math.sqrt((x2t2 - x1t2)**2 + (y2t2 - y1t2)**2)
    if gs < _EPS:
        return None
    tBeta = t1 + sd / gs
    return tBeta + (t2pi - t2)


def _update_best(best, T, t1, t2, del1, del2, phi1, phi2,
                 xt10, yt10, xt20, yt20, cos_w, sin_w, t2pi, vw):
    best["total_time"] = T
    best["t1"] = t1
    best["t2"] = t2
    best["del1"] = del1
    best["del2"] = del2
    best["phi1"] = phi1
    best["phi2"] = phi2
    best["xt10"] = xt10
    best["yt10"] = yt10
    best["xt20"] = xt20
    best["yt20"] = yt20
    best["cos_w"] = cos_w
    best["sin_w"] = sin_w
    best["t2pi"] = t2pi
    best["vw"] = vw


def _func(t, k, Va, vw, w, del1, del2, phi1, phi2,
          xt10, xt20, yt10, yt20, E, G):
    phi_diff = math.fmod(phi1 - phi2, _M2PI) + 2 * k * math.pi
    F = Va * ((xt20 - xt10) + vw * (
        t * (del1 / del2 - 1) + phi_diff / (del2 * w)))
    angle = del1 * w * t + phi1
    return E * math.cos(angle) + F * math.sin(angle) - G


def _deriv_func(t, k, Va, vw, w, del1, del2, phi1, phi2,
                xt10, xt20, yt10, yt20, E, G):
    phi_diff = math.fmod(phi1 - phi2, _M2PI) + 2 * k * math.pi
    F = Va * ((xt20 - xt10) + vw * (
        t * (del1 / del2 - 1) + phi_diff / (del2 * w)))
    angle = del1 * w * t + phi1
    sin_val = math.sin(angle)
    return (-E * del1 * w * sin_val
            + F * del1 * w * math.cos(angle)
            + Va * vw * (del1 / del2 - 1) * sin_val)


def _newton_raphson(x, k, Va, vw, w, del1, del2, phi1, phi2,
                    xt10, xt20, yt10, yt20, E, G, max_iter=100):
    for _ in range(max_iter):
        fp = _deriv_func(x, k, Va, vw, w, del1, del2, phi1, phi2,
                         xt10, xt20, yt10, yt20, E, G)
        if abs(fp) < 1e-15:
            break
        h = _func(x, k, Va, vw, w, del1, del2, phi1, phi2,
                  xt10, xt20, yt10, yt20, E, G) / fp
        if abs(h) < _EPS:
            break
        x -= h
    return x


def sample_trochoid(sol: dict, time_offset: float,
                    airspeed: float, wind_u: float, wind_v: float) -> np.ndarray:
    """Sample ground-frame (x, y, heading) at a given physical time."""
    Va = airspeed
    vw = sol["vw"]
    w = Va / (Va / (Va / 1.0))  # need rhomin... pass via sol
    # Actually compute w from t2pi: w = 2π / t2pi
    t2pi = sol["t2pi"]
    w = _M2PI / t2pi

    del1 = sol["del1"]
    del2 = sol["del2"]
    phi1 = sol["phi1"]
    phi2 = sol["phi2"]
    t1 = sol["t1"]
    t2 = sol["t2"]
    xt10 = sol["xt10"]
    yt10 = sol["yt10"]
    xt20 = sol["xt20"]
    yt20 = sol["yt20"]

    # Straight segment endpoints
    x1t2 = (Va / (del1 * w)) * math.sin(del1 * w * t1 + phi1) + vw * t1 + xt10
    y1t2 = -(Va / (del1 * w)) * math.cos(del1 * w * t1 + phi1) + yt10
    x2t2 = (Va / (del2 * w)) * math.sin(del2 * w * t2 + phi2) + vw * t2 + xt20
    y2t2 = -(Va / (del2 * w)) * math.cos(del2 * w * t2 + phi2) + yt20

    # Straight segment physical time
    xt2dot = Va * math.cos(del2 * w * t2 + phi2) + vw
    yt2dot = Va * math.sin(del2 * w * t2 + phi2)
    gs = math.sqrt(xt2dot**2 + yt2dot**2)
    sd = math.sqrt((x2t2 - x1t2)**2 + (y2t2 - y1t2)**2)
    straight_time = sd / gs if gs > _EPS else 0.0
    tBeta = t1 + straight_time

    t = max(0.0, min(time_offset, sol["total_time"]))

    if t <= t1:
        xw = (Va / (del1 * w)) * math.sin(del1 * w * t + phi1) + vw * t + xt10
        yw = -(Va / (del1 * w)) * math.cos(del1 * w * t + phi1) + yt10
        air_hdg_w = del1 * w * t + phi1
    elif t <= tBeta:
        frac = (t - t1) / straight_time if straight_time > _EPS else 0.0
        frac = max(0.0, min(1.0, frac))
        xw = x1t2 + frac * (x2t2 - x1t2)
        yw = y1t2 + frac * (y2t2 - y1t2)
        air_hdg_w = del1 * w * t1 + phi1
    else:
        # Second turn: map physical time to parametric
        t_param = t2 + (t - tBeta)
        xw = (Va / (del2 * w)) * math.sin(del2 * w * t_param + phi2) + vw * t_param + xt20
        yw = -(Va / (del2 * w)) * math.cos(del2 * w * t_param + phi2) + yt20
        air_hdg_w = del2 * w * t_param + phi2

    cos_w: float = sol["cos_w"]
    sin_w: float = sol["sin_w"]
    gx: float = xw * cos_w - yw * sin_w
    gy: float = xw * sin_w + yw * cos_w

    psi_w = math.atan2(wind_v, wind_u)
    air_hdg = air_hdg_w + psi_w
    ground_heading: float = math.atan2(
        Va * math.sin(air_hdg) + wind_v,
        Va * math.cos(air_hdg) + wind_u)

    result: np.ndarray = np.array([gx, gy, ground_heading], dtype=np.float64)
    return result


