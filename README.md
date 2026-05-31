# tclab_control

Closed-loop control experiments on the **TCLab** thermal plant (Arduino + dual
heater/temperature shield). The goal is to make **EE_Ref Chapter 4** (control systems)
physical: identify the plant, design and tune a PID controller, and compare measured
overshoot / settling time / steady-state error against theory.

Project roadmap and session tracking live in the running doc:
[`TCLAB.md`](https://github.com/sbj-ee/tools/blob/main/TCLAB.md).

## Install

This project uses [uv](https://docs.astral.sh/uv/). A uv venv ships **without pip**, so use
`uv pip` (not `pip`) to install into it:

```bash
uv venv
uv pip install -e ".[hardware,dev]"   # 'hardware' pulls in the `tclab` package
```

The `tclab` package ships a built-in **simulator** (`TCLabModel`), so every command below
runs with **no hardware connected** by passing `--sim`.

## Quick start (no hardware — simulator)

```bash
# Step 1/2: run a heater step test against the simulator, log to CSV
uv run tclab-step --sim --seconds 120 --power 60 --out data/step_sim.csv

# Step 2: fit a FOPDT model (K, tau, theta) to that step + save a plot
uv run tclab-fit data/step_sim.csv --plot data/fit_sim.png

# Step 4: tune (IMC vs Ziegler-Nichols) and compare in simulation
uv run tclab-tune data/step_sim.csv --setpoint 50 --plot data/tune_sim.png
```

## On real hardware

```bash
# 1. Find the board's serial port (see github.com/sbj-ee/tools/find-serial.py)
# 2. Step test → system ID
uv run tclab-step --hardware --seconds 600 --power 60 --out data/step_hw.csv
uv run tclab-fit  data/step_hw.csv --plot data/fit_hw.png

# 3. Closed-loop PID run, gains derived from the measured fit
uv run tclab-run --from-fit data/step_hw.csv --setpoint 50 --tuning imc \
                 --out data/run_hw.csv --plot data/run_hw.png
```

**Measured plant (real TCLab, 2026-05-31):** K = 0.694 °C/%, τ = 142.9 s, θ = 19.85 s
(RMSE 0.43 °C). IMC closed loop to 50 °C held the setpoint with **0.9 % overshoot,
−0.03 °C steady-state error** — close to the 0.4 % prediction (real settling is longer than
the model, the expected real-plant-lag lesson).

### Live plot

`tclab-run --live` opens a real-time matplotlib window (T vs. setpoint + heater %). It needs
an **interactive matplotlib backend**; the base install only has the headless `agg` backend,
so install a GUI backend into the venv first:

```bash
uv pip install pyqt5         # NOT `pip install` — uv venvs have no pip
uv run tclab-run --from-fit data/step_hw.csv --setpoint 50 --tuning imc --live \
                 --out data/run_hw.csv --plot data/run_hw.png
```

Without a GUI backend, `--live` automatically falls back to a headless run (still logs
`--out` and saves `--plot`). `--plot` must use an image extension (`.png`, `.pdf`, …) — a
non-image path is rejected before the run starts. If the window still doesn't appear, force
the backend with `MPLBACKEND=QtAgg`, and on some systems install the Qt libs
(`sudo apt install libxcb-cursor0 libgl1`).

## Layout

```
src/tclab_control/
  pid.py                 # discrete PID (deriv-on-measurement, clamp, anti-windup)  [Step 3]
  fopdt.py               # FOPDT model + step-response fitter                       [Step 2]
  tuning.py              # IMC + Ziegler-Nichols rules from a fitted FOPDT          [Step 4]
  simulate.py            # closed-loop FOPDT sim + step metrics                     [Step 4]
  logger.py              # CSV time-series logger                                   [Step 1]
  lab.py                 # connect(): real board or offline TCLabModel simulator
  experiments/
    step_test.py         # `tclab-step` : step test -> CSV                          [Step 1/2]
    fit_step.py          # `tclab-fit`  : FOPDT system ID from a step CSV           [Step 2]
    tune.py              # `tclab-tune` : tuning + sim comparison                   [Step 4]
    run.py               # `tclab-run`  : live closed-loop PID on the real board    [Step 3/4]
notebooks/               # marimo notebooks per roadmap step (mirrors EE_Ref Ch.4)
tests/                   # pytest (no hardware required)
data/                    # logged runs (CSVs gitignored)
```

## Roadmap (see TCLAB.md for detail)

1. **Connect & log** — stream T1/T2 + heater to CSV. ✅ *done on hardware*
2. **Step test → system ID** — fit FOPDT (`K, τ, θ`). ✅ *done on hardware*
3. **Discrete PID** — drive a setpoint. ✅ *`pid.py` + `tclab-run`*
4. **Tune & compare** — IMC / Z-N, measured vs. predicted. ✅ ⭐ *milestone met on hardware*
5. **Stretch** — disturbance rejection, 2-heater MIMO. *(not started)*

## Development

```bash
uv run pytest -q          # 26 tests, no hardware required
```
