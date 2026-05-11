"""Sachdev et al. (2023) trochoidal Dubins solver — standalone module.

Port of github.com/castacks/trochoids to Python.
This file can be tested independently before integrating into dubins3d.py.
"""

import math
import numpy as np
from typing import Any


def _mod2pi(angle: "float | np.floating | np.ndarray[Any, np.dtype[Any]]") -> "float | np.floating | np.ndarray[Any, np.dtype[Any]]":
    return angle % (2.0 * math.pi)


_EPS = 1e-6
_M2PI = 2 * math.pi


def solve_trochoid(
    qi: np.ndarray[Any, np.dtype[Any]],
    qf: np.ndarray[Any, np.dtype[Any]],
    rhomin: float,
    airspeed: float,
    wind_u: float,
    wind_v: float,
) -> dict[Any, Any]:
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
        "w": w,
    }

    for del1, del2 in [(1, 1), (1, -1), (-1, 1), (-1, -1)]:
        phi1 = math.fmod(phi1_base - psi_w, _M2PI)
        phi2 = math.fmod(phi2_base - psi_w - del2 * _M2PI, _M2PI)

        xt10 = x0 - (Va / (del1 * w)) * math.sin(phi1)
        yt10 = y0 + (Va / (del1 * w)) * math.cos(phi1)
        # sin/cos are 2π-periodic, so the `+ del2 * _M2PI` shift in phi2 cancels
        # inside trig — only the `-vw * t2pi` drift correction on xt20 is live.
        xt20 = xf - (Va / (del2 * w)) * math.sin(phi2) - vw * t2pi
        yt20 = yf + (Va / (del2 * w)) * math.cos(phi2)

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


def _try_analytical(
    Va: float, vw: float, w: float, t2pi: float,
    del1: float, del2: float, phi1: float, phi2: float,
    xt10: float, yt10: float, xt20: float, yt20: float,
    E: float, G: float, cos_w: float, sin_w: float,
    best: dict[str, Any],
) -> None:
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


def _try_numerical(
    Va: float, vw: float, w: float, t2pi: float,
    del1: float, del2: float, phi1: float, phi2: float,
    xt10: float, yt10: float, xt20: float, yt20: float,
    E: float, G: float, cos_w: float, sin_w: float,
    best: dict[str, Any],
) -> None:
    step = 2 * t2pi / 60.0  # 60 guesses instead of 360 (6x faster)
    for k in range(-2, 2):   # k range -2..1 instead of -3..2
        roots: list[float] = []
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
        unique: list[float] = []
        for r in roots:
            if not unique or abs(r - unique[-1]) > _EPS:
                unique.append(r)

        for t1 in unique:
            phi_diff = math.fmod(phi1 - phi2, _M2PI) + 2 * k * math.pi
            t2 = (del1 / del2) * t1 + phi_diff / (del2 * w)
            # Accept t2 ∈ (-t2pi, t2pi].  A negative t2 is a parametric
            # phasing offset for the second arc (the residual arc time
            # `t2pi - t2` then exceeds t2pi), not a multi-loop ground
            # track — the BSB construction always produces a single-arc
            # second turn.  Outside this range is genuinely unphysical.
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


def _compute_total_time(
    Va: float, vw: float, w: float, t2pi: float,
    del1: float, del2: float, phi1: float, phi2: float,
    xt10: float, yt10: float, xt20: float, yt20: float,
    t1: float, t2: float, alpha: float | None,
) -> float | None:
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


def _update_best(
    best: dict[str, Any], T: float, t1: float, t2: float,
    del1: float, del2: float, phi1: float, phi2: float,
    xt10: float, yt10: float, xt20: float, yt20: float,
    cos_w: float, sin_w: float, t2pi: float, vw: float,
) -> None:
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


def _func(
    t: float, k: int, Va: float, vw: float, w: float,
    del1: float, del2: float, phi1: float, phi2: float,
    xt10: float, xt20: float, yt10: float, yt20: float,
    E: float, G: float,
) -> float:
    phi_diff = math.fmod(phi1 - phi2, _M2PI) + 2 * k * math.pi
    F = Va * ((xt20 - xt10) + vw * (
        t * (del1 / del2 - 1) + phi_diff / (del2 * w)))
    angle = del1 * w * t + phi1
    return E * math.cos(angle) + F * math.sin(angle) - G


def _deriv_func(
    t: float, k: int, Va: float, vw: float, w: float,
    del1: float, del2: float, phi1: float, phi2: float,
    xt10: float, xt20: float, yt10: float, yt20: float,
    E: float, G: float,
) -> float:
    phi_diff = math.fmod(phi1 - phi2, _M2PI) + 2 * k * math.pi
    F = Va * ((xt20 - xt10) + vw * (
        t * (del1 / del2 - 1) + phi_diff / (del2 * w)))
    angle = del1 * w * t + phi1
    sin_val = math.sin(angle)
    return (-E * del1 * w * sin_val
            + F * del1 * w * math.cos(angle)
            + Va * vw * (del1 / del2 - 1) * sin_val)


def _newton_raphson(
    x: float, k: int, Va: float, vw: float, w: float,
    del1: float, del2: float, phi1: float, phi2: float,
    xt10: float, xt20: float, yt10: float, yt20: float,
    E: float, G: float, max_iter: int = 100,
) -> float:
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


def sample_trochoid(sol: dict[Any, Any], time_offset: float,
                    airspeed: float, wind_u: float, wind_v: float) -> np.ndarray[Any, np.dtype[Any]]:
    """Sample ground-frame (x, y, heading) at a given physical time."""
    Va = airspeed
    vw = sol["vw"]
    w = sol["w"]

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
    gs = math.hypot(xt2dot, yt2dot)
    sd = math.hypot(x2t2 - x1t2, y2t2 - y1t2)
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

    psi_w: float = sol["psi_w"]
    air_hdg = air_hdg_w + psi_w
    ground_heading: float = math.atan2(
        Va * math.sin(air_hdg) + wind_v,
        Va * math.cos(air_hdg) + wind_u)

    result: np.ndarray[Any, np.dtype[Any]] = np.array([gx, gy, ground_heading], dtype=np.float64)
    return result


# ---------------------------------------------------------------------------
# CCC trochoid solver (LRL / RLR)
# ---------------------------------------------------------------------------
#
# Reduces to a 1-D nonlinear root-find on the half-arc-angle of the middle
# arc.  Derivation: integrate the kinematics through the three arcs in the
# wind-aligned frame and substitute β = (α₂+α₃)/2, γ = |α₂-α₃|/2.  Position
# constraints become
#
#     P − 4·B·γ = 4·A·cosβ·sinγ        (x)
#     Q         = 4·A·sinβ·sinγ        (y)
#
# where A = Va/ω, B = vw/ω, and (P, Q) are constants depending on
# endpoint poses.  Squaring and adding eliminates β:
#
#     16·A²·sin²γ = (P − 4·B·γ)² + Q²    (*)
#
# Equation (*) is the trochoid CCC scalar equation — closed-form for B = 0
# (still-air air-frame Dubins CCC), transcendental otherwise.  Newton-
# Raphson from the still-air root converges in a handful of iterations
# in the operationally interesting regime (vw / Va ≲ 0.3).
#
# Once γ solves (*), β = atan2(Q, P − 4·B·γ) and the per-family arc-time
# unknowns (t₁, t₂, t₃) follow by simple algebra.


def _ccc_p_q(
    family: str, x0_w: float, y0_w: float, xf_w: float, yf_w: float,
    alpha1: float, alpha4: float, A: float, B: float,
) -> tuple[float, float]:
    """Compute the (P, Q) constants of equation (*) for a CCC family.

    LRL has δ₁ = δ₃ = +1 (outer arcs left); RLR has δ₁ = δ₃ = -1.  The
    sign flip on the outer-arc direction inverts the (α₁ − α₄) and
    (sin α₁ − sin α₄) contributions to P and the (cos α₁ − cos α₄)
    contribution to Q.
    """
    sin_a1 = math.sin(alpha1)
    sin_a4 = math.sin(alpha4)
    cos_a1 = math.cos(alpha1)
    cos_a4 = math.cos(alpha4)
    if family == "LRL":
        P = (xf_w - x0_w) + B * (alpha1 - alpha4) + A * (sin_a1 - sin_a4)
        Q = (yf_w - y0_w) - A * (cos_a1 - cos_a4)
    elif family == "RLR":
        P = (xf_w - x0_w) + B * (alpha4 - alpha1) + A * (sin_a4 - sin_a1)
        Q = (yf_w - y0_w) + A * (cos_a1 - cos_a4)
    else:
        raise ValueError(f"unknown CCC family {family!r}")
    return P, Q


def _ccc_newton(
    P: float, Q: float, A: float, B: float, gamma0: float,
    max_iter: int = 40, tol: float = 1e-10,
) -> float | None:
    """Root-find equation (*) for γ via Newton-Raphson from initial guess γ₀.

    Returns the converged γ, or None if Newton failed to converge or
    the derivative collapsed.
    """
    gamma = gamma0
    for _ in range(max_iter):
        sg = math.sin(gamma)
        cg = math.cos(gamma)
        residual = P - 4.0 * B * gamma
        F = 16.0 * A * A * sg * sg - residual * residual - Q * Q
        Fp = 16.0 * A * A * (2.0 * sg * cg) + 8.0 * B * residual
        if abs(Fp) < 1e-14:
            return None
        step = F / Fp
        gamma -= step
        if abs(step) < tol:
            return gamma
    return None


def _try_ccc_trochoid_family(
    qi: np.ndarray[Any, np.dtype[Any]],
    qf: np.ndarray[Any, np.dtype[Any]],
    rhomin: float, airspeed: float, wind_u: float, wind_v: float,
    family: str,
) -> dict[str, Any] | None:
    """Solve the trochoidal CCC path for one family (``"LRL"`` or ``"RLR"``).

    Returns a solution dict (see :func:`solve_ccc_trochoid`) on success,
    or ``None`` when no valid single-revolution solution exists.

    Headings α₁ and α₄ enter the position-equation derivation as
    *unwrapped* angles — equivalent to mod-2π in the kinematics (sin/
    cos are 2π-periodic) but distinct in the wind-drift contribution
    ``vw·T`` because ``T`` depends on the unwrapped Σ(αᵢ).  Different
    2π·k shifts of (α₁, α₄) therefore correspond to different physical
    multi-loop counts of the path.  We enumerate small wrap counts and
    keep the time-optimal valid candidate across all of them.
    """
    Va = airspeed
    vw = math.sqrt(wind_u * wind_u + wind_v * wind_v)
    psi_w = math.atan2(wind_v, wind_u)
    omega = Va / rhomin
    A = rhomin           # Va / omega
    B = vw / omega
    t2pi = _M2PI / omega

    cos_w = math.cos(psi_w)
    sin_w = math.sin(psi_w)

    # Wind-aligned (rotated) frame.  Wind is along +x at speed vw.
    x0 = qi[0] * cos_w + qi[1] * sin_w
    y0 = -qi[0] * sin_w + qi[1] * cos_w
    xf = qf[0] * cos_w + qf[1] * sin_w
    yf = -qf[0] * sin_w + qf[1] * cos_w
    alpha1_base = qi[2] - psi_w
    alpha4_base = qf[2] - psi_w

    if family == "LRL":
        # α₂ = β + γ, α₃ = β - γ.
        # +2π to β: delta1 += 2π, delta3 -= 2π.
        sign1, sign3 = +1.0, -1.0
    else:  # RLR
        # α₂ = β - γ, α₃ = β + γ.
        # +2π to β: delta1 -= 2π, delta3 += 2π.
        sign1, sign3 = -1.0, +1.0

    best: dict[str, Any] | None = None

    # Enumerate 2π·k₄ wraps of α₄ to cover paths where one or both
    # arcs traverse close to a full revolution (typical when wind is
    # significant or the geometry is tight).  α₁ is always taken at
    # its principal value — multi-loop starts have no physical meaning
    # since the aircraft enters the path with a single specific
    # heading, not heading + 2π·k.
    alpha1 = alpha1_base
    for k4 in (0, +1, -1):
        alpha4 = alpha4_base + _M2PI * k4

        P, Q = _ccc_p_q(family, x0, y0, xf, yf, alpha1, alpha4, A, B)

        # Still-air root (B = 0): 16·A²·sin²γ = P² + Q².
        R2 = (P * P + Q * Q) / (16.0 * A * A)
        if R2 > 1.0 + 1e-9:
            continue
        R = math.sqrt(min(R2, 1.0))

        # Two still-air roots: γ_small ∈ (0, π/2), γ_large ∈ (π/2, π).
        # Try both as Newton seeds — they bracket the trochoidal
        # roots in nearly all operating regimes.
        for gamma0 in (math.pi - math.asin(R), math.asin(R)):
            gamma = _ccc_newton(P, Q, A, B, gamma0)
            if gamma is None or not (0.0 < gamma < math.pi):
                continue
            sg = math.sin(gamma)
            if abs(sg) < 1e-12:
                continue

            # β recovered from (cosβ·sinγ, sinβ·sinγ).
            beta_principal = math.atan2(Q, P - 4.0 * B * gamma)

            if family == "LRL":
                d1_base = beta_principal + gamma - alpha1
                d3_base = alpha4 - (beta_principal - gamma)
            else:
                d1_base = alpha1 - (beta_principal - gamma)
                d3_base = (beta_principal + gamma) - alpha4

            # β-wrap enumeration so we don't reject a valid path just
            # because atan2's principal-value β puts (delta1, delta3)
            # outside [0, 2π).
            for k in range(-2, 3):
                d1 = d1_base + sign1 * _M2PI * k
                d3 = d3_base + sign3 * _M2PI * k
                if not (0.0 <= d1 < _M2PI and 0.0 <= d3 < _M2PI):
                    continue
                t1 = d1 / omega
                t2 = (2.0 * gamma) / omega
                t3 = d3 / omega
                if not (0.0 < t2 < t2pi):
                    continue
                total_time = t1 + t2 + t3
                if best is None or total_time < best["total_time"]:
                    best = {
                        "total_time": total_time,
                        "t1": t1, "t2": t2, "t3": t3,
                        "family": family,
                        "omega": omega, "A": A, "vw": vw,
                        "alpha1": alpha1,
                        "x0_w": x0, "y0_w": y0,
                        "cos_w": cos_w, "sin_w": sin_w,
                        "psi_w": psi_w, "t2pi": t2pi,
                    }
    return best


def solve_ccc_trochoid(
    qi: np.ndarray[Any, np.dtype[Any]],
    qf: np.ndarray[Any, np.dtype[Any]],
    rhomin: float,
    airspeed: float,
    wind_u: float,
    wind_v: float,
) -> dict[str, Any] | None:
    """Solve for the time-optimal trochoidal CCC path (LRL or RLR).

    Args:
        qi: ``[x, y, heading]`` start in inertial frame (m, rad).
        qf: ``[x, y, heading]`` goal in inertial frame.
        rhomin: Minimum turn radius (m).
        airspeed: True airspeed (m/s).
        wind_u: Eastward wind component (m/s).
        wind_v: Northward wind component (m/s).

    Returns:
        Solution dict with keys ``total_time``, ``t1``, ``t2``, ``t3``,
        ``family`` (``"LRL"`` or ``"RLR"``), and the wind-frame
        constants ``omega``, ``A``, ``vw``, ``alpha1``, ``x0_w``,
        ``y0_w``, ``cos_w``, ``sin_w``, ``psi_w``, ``t2pi`` needed by
        :func:`sample_ccc_trochoid`.  Returns ``None`` when no valid
        single-revolution CCC solution exists in either family.
    """
    best: dict[str, Any] | None = None
    for family in ("LRL", "RLR"):
        cand = _try_ccc_trochoid_family(
            qi, qf, rhomin, airspeed, wind_u, wind_v, family,
        )
        if cand is None:
            continue
        if best is None or cand["total_time"] < best["total_time"]:
            best = cand
    return best


def _arc_endpoint(
    x_s: float, y_s: float, alpha_s: float, delta: float,
    tau: float, A: float, omega: float, vw: float,
) -> tuple[float, float, float]:
    """Position and heading at relative time ``tau`` along a trochoid arc.

    Computed in the wind-aligned frame, with wind along +x at speed
    ``vw``.  Air-frame turn rate is ``delta * omega`` (delta = ±1).
    """
    da = delta * omega * tau
    sa_s = math.sin(alpha_s)
    ca_s = math.cos(alpha_s)
    sa_e = math.sin(alpha_s + da)
    ca_e = math.cos(alpha_s + da)
    x_e = x_s + (A / delta) * (sa_e - sa_s) + vw * tau
    y_e = y_s - (A / delta) * (ca_e - ca_s)
    return x_e, y_e, alpha_s + da


def sample_ccc_trochoid(
    sol: dict[Any, Any],
    time_offset: float,
    airspeed: float,
    wind_u: float,
    wind_v: float,
) -> np.ndarray[Any, np.dtype[Any]]:
    """Sample the ground-frame ``(x, y, ground_heading)`` of a CCC
    trochoidal path at a given physical time."""
    Va = airspeed
    omega = sol["omega"]
    A = sol["A"]
    vw = sol["vw"]
    cos_w = sol["cos_w"]
    sin_w = sol["sin_w"]
    psi_w = sol["psi_w"]
    t1 = sol["t1"]
    t2 = sol["t2"]
    t3 = sol["t3"]

    if sol["family"] == "LRL":
        deltas = (1.0, -1.0, 1.0)
    else:  # RLR
        deltas = (-1.0, 1.0, -1.0)

    # Walk forward through the arcs to find each arc's start state.
    starts = [(sol["x0_w"], sol["y0_w"], sol["alpha1"])]
    times = (t1, t2, t3)
    for i in range(3):
        x_s, y_s, alpha_s = starts[i]
        x_e, y_e, alpha_e = _arc_endpoint(
            x_s, y_s, alpha_s, deltas[i], times[i], A, omega, vw,
        )
        starts.append((x_e, y_e, alpha_e))

    # Locate the queried time within an arc.
    t = max(0.0, min(time_offset, sol["total_time"]))
    if t <= t1:
        i = 0
        tau = t
    elif t <= t1 + t2:
        i = 1
        tau = t - t1
    else:
        i = 2
        tau = t - t1 - t2

    x_s, y_s, alpha_s = starts[i]
    xw, yw, alpha = _arc_endpoint(
        x_s, y_s, alpha_s, deltas[i], tau, A, omega, vw,
    )

    # Rotate back to inertial frame.
    gx = xw * cos_w - yw * sin_w
    gy = xw * sin_w + yw * cos_w

    air_hdg = alpha + psi_w
    ground_heading = math.atan2(
        Va * math.sin(air_hdg) + wind_v,
        Va * math.cos(air_hdg) + wind_u,
    )

    return np.array([gx, gy, ground_heading], dtype=np.float64)
