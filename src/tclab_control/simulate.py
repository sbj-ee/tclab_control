"""Closed-loop simulation of a PID controlling a FOPDT plant, plus step metrics.

Lets you evaluate tuning offline (no hardware): drive a fitted FOPDT plant with a
:class:`~tclab_control.pid.PID` controller and measure the resulting step response
(overshoot, settling time, steady-state error, rise time) to compare against theory.

The FOPDT plant is integrated with a simple Euler step and a delay buffer for the
dead time θ — adequate at the TCLab's slow timescales and small dt.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from tclab_control.fopdt import FOPDT
from tclab_control.pid import PID


@dataclass
class StepMetrics:
    """Step-response performance metrics."""

    overshoot_pct: float
    settling_time: float      # time to stay within ±tol band (s); inf if never
    steady_state_error: float
    rise_time: float          # 10%→90% of the setpoint change (s); nan if not reached
    peak_value: float
    peak_time: float


def simulate_closed_loop(plant: FOPDT, pid: PID, setpoint: float,
                         t_end: float, dt: float = 1.0, y0: float | None = None):
    """Simulate a PID controlling a FOPDT plant from baseline to ``setpoint``.

    Returns (t, y, u) arrays. The plant gain K is per input unit; the controller
    output (e.g. heater %) drives the plant's first-order lag with dead time θ.
    """
    y0 = plant.y0 if y0 is None else y0
    n = int(round(t_end / dt))
    t = np.arange(n + 1) * dt
    y = np.empty(n + 1)
    u = np.empty(n + 1)
    y[0] = y0

    pid.setpoint = setpoint
    pid.reset()

    # Dead-time buffer: control acting on the plant is delayed by θ.
    delay_steps = max(0, int(round(plant.theta / dt)))
    ubuf: deque[float] = deque([0.0] * (delay_steps + 1), maxlen=delay_steps + 1)

    val = y0
    for k in range(n):
        ctrl = pid.update(pv=val, dt=dt)
        u[k] = ctrl
        ubuf.append(ctrl)
        u_delayed = ubuf[0]
        # First-order plant: τ·dy/dt = -(y - y0) + K·u_delayed
        dval = (-(val - y0) + plant.K * u_delayed) / plant.tau
        val = val + dval * dt
        y[k + 1] = val
    u[n] = u[n - 1]
    return t, y, u


def step_metrics(t, y, setpoint: float, y0: float, tol: float = 0.02) -> StepMetrics:
    """Compute step-response metrics for a rise from ``y0`` toward ``setpoint``."""
    t = np.asarray(t, float)
    y = np.asarray(y, float)
    change = setpoint - y0
    if change == 0:
        raise ValueError("setpoint equals baseline; no step")

    peak_idx = int(np.argmax(y)) if change > 0 else int(np.argmin(y))
    peak_value = float(y[peak_idx])
    peak_time = float(t[peak_idx])

    overshoot = (peak_value - setpoint) / change * 100.0
    overshoot_pct = max(0.0, overshoot if change > 0 else -overshoot)

    sse = float(setpoint - y[-1])

    # Settling time: last time the response leaves the ±tol band around setpoint.
    band = abs(tol * change)
    outside = np.where(np.abs(y - setpoint) > band)[0]
    settling = float(t[outside[-1]]) if outside.size and outside[-1] + 1 < t.size else (
        0.0 if not outside.size else float("inf"))

    # Rise time 10%→90% of the change.
    lo, hi = y0 + 0.1 * change, y0 + 0.9 * change
    if change > 0:
        i10 = np.where(y >= lo)[0]
        i90 = np.where(y >= hi)[0]
    else:
        i10 = np.where(y <= lo)[0]
        i90 = np.where(y <= hi)[0]
    rise = float(t[i90[0]] - t[i10[0]]) if i10.size and i90.size else float("nan")

    return StepMetrics(
        overshoot_pct=overshoot_pct,
        settling_time=settling,
        steady_state_error=sse,
        rise_time=rise,
        peak_value=peak_value,
        peak_time=peak_time,
    )
