"""Tests for the PID controller (no hardware required)."""

import pytest

from tclab_control import PID


def test_proportional_only():
    pid = PID(kp=2.0, setpoint=30.0)
    # error = 30 - 25 = 5; output = 2*5 = 10
    assert pid.update(pv=25.0, dt=1.0) == 10.0


def test_output_clamped_high():
    pid = PID(kp=100.0, setpoint=100.0, out_max=100.0)
    assert pid.update(pv=0.0, dt=1.0) == 100.0


def test_output_clamped_low():
    pid = PID(kp=100.0, setpoint=0.0, out_min=0.0)
    assert pid.update(pv=50.0, dt=1.0) == 0.0


def test_integral_accumulates():
    pid = PID(kp=0.0, ki=1.0, setpoint=10.0, out_max=1000.0)
    first = pid.update(pv=0.0, dt=1.0)   # integral = 10
    second = pid.update(pv=0.0, dt=1.0)  # integral = 20
    assert first == pytest.approx(10.0)
    assert second == pytest.approx(20.0)


def test_anti_windup_holds_integral_at_saturation():
    # Saturated output should not let the integral run away.
    pid = PID(kp=0.0, ki=1.0, setpoint=10.0, out_max=5.0)
    pid.update(pv=0.0, dt=1.0)  # would-be integral 10 -> clamped to 5
    pid.update(pv=0.0, dt=1.0)
    # Drop setpoint so we're no longer saturated; integral must not have wound past 5.
    pid.setpoint = 0.0
    out = pid.update(pv=0.0, dt=1.0)
    assert out <= 5.0


def test_derivative_on_measurement_opposes_rising_pv():
    pid = PID(kp=0.0, kd=1.0, setpoint=0.0)
    pid.update(pv=0.0, dt=1.0)          # seeds prev_pv
    out = pid.update(pv=5.0, dt=1.0)    # pv rose 5 in 1s -> derivative = -5
    assert out == pytest.approx(-5.0)


def test_dt_must_be_positive():
    pid = PID(kp=1.0, setpoint=1.0)
    with pytest.raises(ValueError):
        pid.update(pv=0.0, dt=0.0)


def test_reset_clears_state():
    pid = PID(kp=0.0, ki=1.0, setpoint=10.0, out_max=1000.0)
    pid.update(pv=0.0, dt=1.0)
    pid.reset()
    out = pid.update(pv=0.0, dt=1.0)  # integral starts fresh at 10 again
    assert out == pytest.approx(10.0)
