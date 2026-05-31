"""Step 4: tune PID gains from a fitted FOPDT and compare methods.

    tclab-tune data/step_hw.csv
    tclab-tune data/step_hw.csv --setpoint 50 --plot tune.png

Fits a FOPDT to the step CSV, computes IMC and Ziegler–Nichols gains, simulates
each closed loop on the identified plant, and prints overshoot / settling time /
steady-state error so the methods can be compared against theory.
"""

from __future__ import annotations

import argparse

import numpy as np

from tclab_control.fopdt import fit_step_csv
from tclab_control.simulate import simulate_closed_loop, step_metrics
from tclab_control.tuning import imc_pid, ziegler_nichols


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Tune PID from a step CSV and compare (Step 4).")
    p.add_argument("csv", help="step-test CSV from tclab-step")
    p.add_argument("--channel", default="T1")
    p.add_argument("--heater", default="Q1")
    p.add_argument("--setpoint", type=float, default=None,
                   help="closed-loop setpoint (°C); default = baseline + 25")
    p.add_argument("--lam", type=float, default=None, help="IMC λ (s); default auto")
    p.add_argument("--dt", type=float, default=1.0, help="sim timestep (s)")
    p.add_argument("--plot", metavar="PNG", help="save closed-loop comparison plot")
    args = p.parse_args(argv)

    model = fit_step_csv(args.csv, channel=args.channel, heater=args.heater)
    setpoint = args.setpoint if args.setpoint is not None else model.y0 + 25.0
    t_end = max(10 * model.tau, 600.0)

    print(f"Identified FOPDT: K={model.K:.4f}, τ={model.tau:.1f}s, θ={model.theta:.1f}s, "
          f"baseline={model.y0:.2f}°C")
    print(f"Closed-loop setpoint: {setpoint:.1f}°C\n")

    methods = {
        "IMC": imc_pid(model, lam=args.lam),
        "Ziegler-Nichols": ziegler_nichols(model),
    }

    runs = {}
    for name, g in methods.items():
        pid = g.to_pid(setpoint=setpoint, out_min=0.0, out_max=100.0)
        t, y, u = simulate_closed_loop(model, pid, setpoint, t_end=t_end, dt=args.dt)
        m = step_metrics(t, y, setpoint, model.y0)
        runs[name] = (t, y, u, g, m)
        print(f"[{g.method}]")
        print(f"  Kp={g.kp:.3f}  Ti={g.Ti:.1f}s  Td={g.Td:.1f}s  (ki={g.ki:.4f}, kd={g.kd:.3f})")
        print(f"  overshoot={m.overshoot_pct:.1f}%  settling={m.settling_time:.0f}s  "
              f"rise={m.rise_time:.0f}s  sse={m.steady_state_error:.3f}°C\n")

    if args.plot:
        _plot(runs, setpoint, model.y0, args.plot)
        print(f"plot -> {args.plot}")
    return 0


def _plot(runs, setpoint, y0, out):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax, axu) = plt.subplots(2, 1, figsize=(9, 7), sharex=True,
                                  gridspec_kw={"height_ratios": [2, 1]})
    ax.axhline(setpoint, color="k", ls="--", lw=1, label=f"setpoint {setpoint:.0f}°C")
    for name, (t, y, u, g, m) in runs.items():
        line, = ax.plot(t, y, lw=2, label=f"{name} (OS {m.overshoot_pct:.0f}%, ts {m.settling_time:.0f}s)")
        axu.plot(t, u, lw=1.5, color=line.get_color(), label=name)
    ax.set_ylabel("T (°C)")
    ax.set_title("Closed-loop step: tuning comparison (simulated on identified plant)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    axu.set_ylabel("heater (%)")
    axu.set_xlabel("time (s)")
    axu.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=110)


if __name__ == "__main__":
    raise SystemExit(main())
