# tclab_control

Closed-loop control experiments on the **TCLab** thermal plant (Arduino + dual
heater/temperature shield). The goal is to make **EE_Ref Chapter 4** (control systems)
physical: identify the plant, design and tune a PID controller, and compare measured
overshoot / settling time / steady-state error against the theory.

Project roadmap and tracking live in the running doc:
[`tools/FMCW`-style `TCLAB.md`](https://github.com/sbj-ee/tools/blob/main/TCLAB.md).

## Install

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[hardware,dev]"   # hardware extra pulls in the `tclab` package
```

The `tclab` package ships a built-in **simulator** (`TCLabModel`), so everything here
runs with **no hardware connected**.

## Quick start (no hardware — simulator)

```bash
# Step 1/2: run a heater step test against the simulator, log to CSV
tclab-step --sim --seconds 120 --power 60 --out data/step_sim.csv
```

Switch to the real board once connected:

```bash
tclab-step --hardware --seconds 120 --power 60 --out data/step_hw.csv
```

## Layout

```
src/tclab_control/
  pid.py                 # discrete PID (deriv-on-measurement, clamp, anti-windup)  [Step 3]
  logger.py              # CSV time-series logger                                   [Step 1]
  lab.py                 # connect(): real board or offline TCLabModel simulator
  experiments/
    step_test.py         # `tclab-step` CLI: step test -> CSV                       [Step 1/2]
notebooks/               # marimo notebooks per roadmap step (mirrors EE_Ref Ch.4)
tests/                   # pytest (no hardware required)
data/                    # logged runs (gitignored)
```

## Roadmap (see TCLAB.md for detail)

1. **Connect & log** — stream T1/T2 + heater settings to CSV. *(scaffolded: `step_test.py`)*
2. **Step test → system ID** — fit a FOPDT model (`K, τ, θ`) to a step.
3. **Discrete PID** — drive a setpoint. *(scaffolded: `pid.py`)*
4. **Tune & compare** — IMC / Ziegler–Nichols vs. hand; measured vs. predicted. ⭐ milestone
5. **Stretch** — disturbance rejection, 2-heater MIMO.

## Development

```bash
pytest -q          # runs without hardware (PID + logger unit tests)
```
