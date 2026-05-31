"""Tests for the FOPDT model and fitter (no hardware required).

Strategy: generate a synthetic FOPDT step response with known (K, tau, theta),
add a little noise, and confirm the fitter recovers the parameters.
"""

import numpy as np
import pytest

from tclab_control import FOPDT, fit_fopdt


def _make_step(K, tau, theta, y0, u_step, t_step, tmax, dt=1.0, noise=0.0, seed=0):
    t = np.arange(0, tmax, dt)
    true = FOPDT(K, tau, theta, y0, u_step, t_step)
    y = true.predict(t)
    if noise:
        rng = np.random.default_rng(seed)
        y = y + rng.normal(0, noise, size=y.shape)
    return t, y


def test_predict_baseline_then_rise():
    m = FOPDT(K=0.5, tau=20.0, theta=5.0, y0=25.0, u_step=50.0, t_step=10.0)
    # Before step+dead-time: baseline.
    assert m.predict(np.array([0.0]))[0] == pytest.approx(25.0)
    assert m.predict(np.array([14.0]))[0] == pytest.approx(25.0)
    # Far after: approaches y0 + K*u_step = 25 + 25 = 50.
    assert m.predict(np.array([10_000.0]))[0] == pytest.approx(50.0, abs=1e-3)


def test_fit_recovers_clean_parameters():
    K, tau, theta, y0, u_step, t_step = 0.45, 18.0, 6.0, 23.0, 60.0, 10.0
    t, y = _make_step(K, tau, theta, y0, u_step, t_step, tmax=200)
    m = fit_fopdt(t, y, u_step=u_step, t_step=t_step)
    assert m.K == pytest.approx(K, rel=0.05)
    assert m.tau == pytest.approx(tau, rel=0.10)
    assert m.theta == pytest.approx(theta, abs=2.0)


def test_fit_tolerates_noise():
    K, tau, theta, y0, u_step, t_step = 0.5, 25.0, 4.0, 22.0, 50.0, 5.0
    t, y = _make_step(K, tau, theta, y0, u_step, t_step, tmax=250, noise=0.15, seed=42)
    m = fit_fopdt(t, y, u_step=u_step, t_step=t_step)
    # Gain and time constant should still be close despite noise.
    assert m.K == pytest.approx(K, rel=0.08)
    assert m.tau == pytest.approx(tau, rel=0.20)


def test_fit_requires_enough_points():
    with pytest.raises(ValueError):
        fit_fopdt([0, 1, 2], [1, 1, 1], u_step=1.0)


def test_fit_step_csv(tmp_path):
    # Write a synthetic CSV in the tclab-step format and fit it end to end.
    K, tau, theta, y0, u_step, t_step = 0.4, 15.0, 3.0, 24.0, 70.0, 8.0
    t, y = _make_step(K, tau, theta, y0, u_step, t_step, tmax=180)
    csv_path = tmp_path / "step.csv"
    with csv_path.open("w") as fh:
        fh.write("t,T1,T2,Q1,Q2\n")
        for ti, yi in zip(t, y):
            q1 = 0.0 if ti < t_step else u_step
            fh.write(f"{ti},{yi},{y0},{q1},0.0\n")

    from tclab_control import fit_step_csv

    m = fit_step_csv(csv_path)
    assert m.u_step == pytest.approx(u_step)
    assert m.t_step == pytest.approx(t_step)
    assert m.K == pytest.approx(K, rel=0.05)
    assert m.tau == pytest.approx(tau, rel=0.12)
