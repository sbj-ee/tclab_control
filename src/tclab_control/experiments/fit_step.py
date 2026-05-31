"""Step 2: fit a FOPDT model to a logged step test and report (K, tau, theta).

    tclab-fit data/step_sim.csv
    tclab-fit data/step_sim.csv --plot fit.png      # overlay fit vs. measured

Reads the CSV produced by ``tclab-step`` (columns t,T1,T2,Q1,Q2).
"""

from __future__ import annotations

import argparse
import csv

import numpy as np

from tclab_control.fopdt import fit_step_csv, FOPDT


def _rmse(model: FOPDT, path: str, channel: str, heater: str) -> float:
    t, y = [], []
    with open(path) as fh:
        for row in csv.DictReader(fh):
            t.append(float(row["t"]))
            y.append(float(row[channel]))
    t, y = np.array(t), np.array(y)
    return float(np.sqrt(np.mean((model.predict(t) - y) ** 2)))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Fit a FOPDT model to a step test CSV (Step 2).")
    p.add_argument("csv", help="step-test CSV from tclab-step")
    p.add_argument("--channel", default="T1", help="temperature column to fit (default T1)")
    p.add_argument("--heater", default="Q1", help="heater column for the step (default Q1)")
    p.add_argument("--plot", metavar="PNG", help="save a fit-vs-measured plot to this path")
    args = p.parse_args(argv)

    model = fit_step_csv(args.csv, channel=args.channel, heater=args.heater)
    rmse = _rmse(model, args.csv, args.channel, args.heater)

    print(f"FOPDT fit for {args.channel} (step in {args.heater}):")
    print(f"  K     = {model.K:.4f}  (deg C per % heater)")
    print(f"  tau   = {model.tau:.2f} s")
    print(f"  theta = {model.theta:.2f} s  (dead time)")
    print(f"  step  = {model.u_step:.1f}% at t = {model.t_step:.1f} s, baseline {model.y0:.2f} C")
    print(f"  RMSE  = {rmse:.4f} C")

    if args.plot:
        _save_plot(model, args.csv, args.channel, args.plot)
        print(f"  plot  -> {args.plot}")
    return 0


def _save_plot(model: FOPDT, path: str, channel: str, out: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t, y = [], []
    with open(path) as fh:
        for row in csv.DictReader(fh):
            t.append(float(row["t"]))
            y.append(float(row[channel]))
    t, y = np.array(t), np.array(y)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(t, y, ".", ms=4, alpha=0.6, label=f"measured {channel}")
    ax.plot(t, model.predict(t), "r-", lw=2, label="FOPDT fit")
    ax.axvline(model.t_step + model.theta, color="gray", ls=":",
               label=f"θ end = {model.t_step + model.theta:.1f}s")
    ax.set_xlabel("time (s)")
    ax.set_ylabel(f"{channel} (°C)")
    ax.set_title(f"FOPDT fit: K={model.K:.3f}, τ={model.tau:.1f}s, θ={model.theta:.1f}s")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=110)


if __name__ == "__main__":
    raise SystemExit(main())
