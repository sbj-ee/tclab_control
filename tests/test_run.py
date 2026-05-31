"""Tests for the closed-loop runner (headless; uses the simulator)."""

import csv

from tclab_control.experiments.run import run
from tclab_control.tuning import imc_pid
from tclab_control import FOPDT


def _pid_from_plantish():
    # Build a reasonable PID from a representative TCLab-like plant.
    model = FOPDT(K=0.69, tau=143.0, theta=20.0, y0=23.0, u_step=60.0, t_step=0.0)
    return imc_pid(model).to_pid(setpoint=35.0, out_min=0.0, out_max=100.0)


def test_run_on_simulator_returns_series():
    pid = _pid_from_plantish()
    out = run(pid, setpoint=35.0, seconds=5, use_sim=True, live=False, step_sleep=0)
    assert len(out["t"]) == 5
    assert len(out["T1"]) == 5
    assert len(out["Q1"]) == 5
    # Heater output stays within actuator limits.
    assert all(0.0 <= u <= 100.0 for u in out["Q1"])


def test_run_writes_csv(tmp_path):
    pid = _pid_from_plantish()
    path = tmp_path / "run.csv"
    run(pid, setpoint=35.0, seconds=4, use_sim=True, live=False, out=str(path), step_sleep=0)
    rows = list(csv.DictReader(path.open()))
    assert len(rows) == 4
    assert set(rows[0].keys()) == {"t", "T1", "T2", "Q1", "Q2"}


def test_run_saves_plot(tmp_path):
    pid = _pid_from_plantish()
    png = tmp_path / "run.png"
    run(pid, setpoint=35.0, seconds=4, use_sim=True, live=False, plot=str(png), step_sleep=0)
    assert png.exists() and png.stat().st_size > 0


def test_main_rejects_non_image_plot_extension(capsys):
    # argparse .error() exits 2; must happen before any run.
    from tclab_control.experiments.run import main
    import pytest as _pytest

    with _pytest.raises(SystemExit) as exc:
        main(["--setpoint", "50", "--sim", "--plot", "data/run_hw.csv"])
    assert exc.value.code == 2
    assert "image extension" in capsys.readouterr().err


def test_live_falls_back_to_headless_without_gui(tmp_path, monkeypatch):
    # Force "no GUI backend" → run() should flip live off and still complete.
    import tclab_control.experiments.run as runmod

    monkeypatch.setattr(runmod, "_have_gui_backend", lambda: False)
    pid = _pid_from_plantish()
    out = runmod.run(pid, setpoint=35.0, seconds=3, use_sim=True, live=True, step_sleep=0)
    assert len(out["t"]) == 3
