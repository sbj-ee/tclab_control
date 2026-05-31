"""First-Order-Plus-Dead-Time (FOPDT) model and step-response fitting (Step 2).

FOPDT model:  G(s) = K * exp(-theta * s) / (tau * s + 1)

Step response to an input change Delta_u starting from baseline y0, applied at t = t_step:

    y(t) = y0,                                    t < t_step + theta
    y(t) = y0 + K*Delta_u*(1 - exp(-(t - t_step - theta)/tau)),   otherwise

where:
    K     = process gain (output units per input unit)
    tau   = time constant (s)
    theta = dead time (s)

This module fits (K, tau, theta) from a logged step test (e.g. the CSV produced by
``tclab-step``) using a bounded least-squares fit, with a robust graphical initial guess.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class FOPDT:
    """A first-order-plus-dead-time model fitted to a step response."""

    K: float          # process gain
    tau: float        # time constant (s)
    theta: float      # dead time (s)
    y0: float         # baseline output before the step
    u_step: float     # input step magnitude (Delta_u)
    t_step: float     # time the step was applied (s)

    def predict(self, t: np.ndarray) -> np.ndarray:
        """Model output y(t) for the fitted step."""
        t = np.asarray(t, dtype=float)
        elapsed = t - self.t_step - self.theta
        resp = self.K * self.u_step * (1.0 - np.exp(-np.clip(elapsed, 0, None) / self.tau))
        return self.y0 + np.where(elapsed > 0, resp, 0.0)


def _initial_guess(t, y, u_step, t_step):
    """Graphical first guess: gain from steady-state, tau/theta from the 63.2% point."""
    y0 = float(np.mean(y[t <= t_step])) if np.any(t <= t_step) else float(y[0])
    y_final = float(np.mean(y[t >= t[-1] - max(1.0, 0.05 * (t[-1] - t_step))]))
    K = (y_final - y0) / u_step if u_step != 0 else 1.0

    # Time to reach 63.2% of the total change → t_step + theta + tau.
    target = y0 + 0.632 * (y_final - y0)
    after = t >= t_step
    if K >= 0:
        reached = np.where(after & (y >= target))[0]
    else:
        reached = np.where(after & (y <= target))[0]
    t_63 = t[reached[0]] if reached.size else t[-1]

    # Dead time: first time the output departs the baseline band after the step.
    noise = np.std(y[t <= t_step]) if np.any(t <= t_step) else 0.0
    band = max(3.0 * noise, 0.02 * abs(y_final - y0))
    moved = np.where(after & (np.abs(y - y0) > band))[0]
    theta = max(0.0, float(t[moved[0]] - t_step)) if moved.size else 0.0

    tau = max(1e-3, float(t_63 - t_step - theta))
    return K, tau, theta, y0


def fit_fopdt(t, y, u_step: float, t_step: float | None = None) -> FOPDT:
    """Fit a FOPDT model to a step response.

    Args:
        t: time vector (s).
        y: measured output (e.g. temperature).
        u_step: input step magnitude (Delta_u), e.g. heater % change.
        t_step: time the step was applied; defaults to the first sample.

    Returns:
        A fitted :class:`FOPDT`. Uses SciPy's bounded least-squares if available,
        otherwise returns the graphical initial estimate.
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    if t.size != y.size or t.size < 4:
        raise ValueError("need matching t, y of length >= 4")
    if t_step is None:
        t_step = float(t[0])

    K0, tau0, theta0, y0 = _initial_guess(t, y, u_step, t_step)

    try:
        from scipy.optimize import least_squares
    except ImportError:
        return FOPDT(K0, tau0, theta0, y0, u_step, t_step)

    span = float(t[-1] - t_step) or 1.0

    def residuals(p):
        K, tau, theta = p
        model = FOPDT(K, tau, theta, y0, u_step, t_step)
        return model.predict(t) - y

    lo = [-np.inf, 1e-3, 0.0]
    hi = [np.inf, 10 * span, span]
    sol = least_squares(residuals, x0=[K0, tau0, theta0], bounds=(lo, hi))
    K, tau, theta = sol.x
    return FOPDT(float(K), float(tau), float(theta), y0, u_step, t_step)


def fit_step_csv(path: str | Path, channel: str = "T1", heater: str = "Q1") -> FOPDT:
    """Load a step-test CSV (as written by ``tclab-step``) and fit a FOPDT model.

    Detects the step time/magnitude from the heater column transition.
    """
    t, T, Q = [], [], []
    with Path(path).open() as fh:
        for row in csv.DictReader(fh):
            t.append(float(row["t"]))
            T.append(float(row[channel]))
            Q.append(float(row[heater]))
    t = np.array(t)
    T = np.array(T)
    Q = np.array(Q)

    # Step = first index where heater rises above its initial level.
    q0 = Q[0]
    changed = np.where(Q > q0)[0]
    if changed.size == 0:
        raise ValueError(f"no {heater} step found in {path}")
    i_step = changed[0]
    return fit_fopdt(t, T, u_step=float(Q[i_step] - q0), t_step=float(t[i_step]))
