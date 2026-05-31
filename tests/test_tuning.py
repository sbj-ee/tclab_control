"""Tests for tuning rules and closed-loop simulation (no hardware required)."""

import numpy as np
import pytest

from tclab_control import (
    FOPDT,
    imc_pid,
    simulate_closed_loop,
    step_metrics,
    ziegler_nichols,
)


def _plant():
    # A TCLab-like thermal plant.
    return FOPDT(K=0.85, tau=150.0, theta=14.0, y0=22.5, u_step=50.0, t_step=0.0)


def test_imc_gains_positive_and_units():
    g = imc_pid(_plant())
    assert g.kp > 0 and g.ki > 0 and g.kd >= 0
    # ki and kd convention: ki = kp/Ti, kd = kp*Td
    assert g.ki == pytest.approx(g.kp / g.Ti, rel=1e-9)
    assert g.kd == pytest.approx(g.kp * g.Td, rel=1e-9)


def test_imc_lambda_tradeoff():
    plant = _plant()
    fast = imc_pid(plant, lam=10.0)
    slow = imc_pid(plant, lam=200.0)
    # Larger λ → gentler controller → smaller proportional gain.
    assert slow.kp < fast.kp


def test_zn_gains_positive():
    g = ziegler_nichols(_plant())
    assert g.kp > 0 and g.ki > 0 and g.kd > 0
    assert g.method == "Ziegler-Nichols"


def test_closed_loop_reaches_setpoint_no_steady_state_error():
    plant = _plant()
    sp = plant.y0 + 25.0
    pid = imc_pid(plant).to_pid(setpoint=sp, out_min=0, out_max=100)
    t, y, u = simulate_closed_loop(plant, pid, setpoint=sp, t_end=3000, dt=1.0)
    m = step_metrics(t, y, sp, plant.y0)
    # Integral action should drive steady-state error ~0.
    assert abs(m.steady_state_error) < 0.5
    # Output stays within actuator limits.
    assert u.min() >= 0.0 and u.max() <= 100.0


def test_imc_is_less_aggressive_than_zn():
    plant = _plant()
    sp = plant.y0 + 25.0
    res = {}
    for name, g in (("imc", imc_pid(plant)), ("zn", ziegler_nichols(plant))):
        pid = g.to_pid(setpoint=sp, out_min=0, out_max=100)
        t, y, _ = simulate_closed_loop(plant, pid, setpoint=sp, t_end=3000, dt=1.0)
        res[name] = step_metrics(t, y, sp, plant.y0)
    # Classic result: Z-N overshoots more than IMC lambda tuning.
    assert res["zn"].overshoot_pct >= res["imc"].overshoot_pct


def test_step_metrics_on_known_curve():
    # First-order rise (no overshoot): overshoot ~0, sse ~0.
    y0, sp, tau = 0.0, 1.0, 10.0
    t = np.arange(0, 200, 1.0)
    y = sp - (sp - y0) * np.exp(-t / tau)
    m = step_metrics(t, y, sp, y0)
    assert m.overshoot_pct == pytest.approx(0.0, abs=0.5)
    assert abs(m.steady_state_error) < 1e-2
