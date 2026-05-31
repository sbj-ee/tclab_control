"""Step 3/4 on hardware: run a PID closed loop on the real TCLab with a live plot.

Drives the real board (or the simulator) to a setpoint using PID gains derived from
a measured FOPDT fit, showing T1 vs. setpoint and heater % updating in real time.

    # single setpoint, live window, real board:
    tclab-run --from-fit data/step_hw.csv --setpoint 50 --tuning imc --live

    # multi-setpoint schedule — "50C for 400s, then 35C for 400s, then 40C for 300s, then 25C for 600s":
    tclab-run --from-fit data/step_hw.csv --live \
              --schedule "50:400" "35:400" "40:300" "25:600" \
              --out data/schedule.csv --plot data/schedule.png

    # offline dry run on the simulator (no hardware, no window):
    tclab-run --from-fit data/step_hw.csv --setpoint 50 --sim --seconds 120

    # manual gains instead of a fit:
    tclab-run --kp 2.7 --ki 0.018 --kd 25 --setpoint 50 --live

Notes:
  - ``--live`` opens a matplotlib window (needs a GUI display; run it from your
    own terminal). Without ``--live`` it runs headless and can still save a PNG
    via ``--plot`` and a CSV via ``--out``.
  - Only one process can hold the serial port; close other TCLab programs first.
  - Cooling is passive (no fan) — dropping setpoints takes longer than rising ones.
"""

from __future__ import annotations

import argparse
import time

from tclab_control.fopdt import fit_step_csv
from tclab_control.lab import connect
from tclab_control.logger import RunLogger
from tclab_control.pid import PID
from tclab_control.tuning import imc_pid, ziegler_nichols


def _build_pid(args) -> PID:
    """Construct the PID from either a measured fit (+tuning rule) or manual gains."""
    if args.from_fit:
        model = fit_step_csv(args.from_fit, channel=args.channel, heater=args.heater)
        rule = imc_pid(model, lam=args.lam) if args.tuning == "imc" else ziegler_nichols(model)
        print(f"Plant: K={model.K:.3f}, τ={model.tau:.1f}s, θ={model.theta:.1f}s "
              f"→ {rule.method}: Kp={rule.kp:.3f} Ti={rule.Ti:.1f}s Td={rule.Td:.1f}s")
        return rule.to_pid(setpoint=args.setpoint, out_min=0.0, out_max=100.0)
    return PID(kp=args.kp, ki=args.ki, kd=args.kd, setpoint=args.setpoint,
               out_min=0.0, out_max=100.0)


def run(pid: PID, setpoint: float, seconds: int, use_sim: bool, live: bool,
        out: str | None = None, plot: str | None = None, channel: str = "T1",
        heater: str = "Q1", step_sleep: float = 1.0) -> dict:
    """Run the closed loop for ``seconds``. Returns the logged series as lists.

    ``step_sleep`` is the wall-clock delay between samples (1 s for real runs;
    tests pass 0 to run fast).
    """
    schedule = [(setpoint, seconds)]
    return run_schedule(pid, schedule, use_sim=use_sim, live=live, out=out,
                        plot=plot, channel=channel, heater=heater,
                        step_sleep=step_sleep)


def run_schedule(pid: PID, schedule: list[tuple[float, int]], use_sim: bool,
                 live: bool, out: str | None = None, plot: str | None = None,
                 channel: str = "T1", heater: str = "Q1",
                 step_sleep: float = 1.0) -> dict:
    """Run the PID through a sequence of (setpoint, duration_seconds) segments.

    The PID integral is NOT reset between segments so there's no bump at
    transitions — it continues tracking smoothly.  The setpoint is updated
    instantaneously at each boundary.

    Args:
        schedule: list of (setpoint_°C, duration_s) tuples in order.
    """
    total_seconds = sum(s for _, s in schedule)
    ts: list[float] = []
    temps: list[float] = []
    us: list[float] = []
    sps: list[float] = []   # setpoint trace for the plot

    if live and not _have_gui_backend():
        print("note: no interactive matplotlib backend (Agg) — running headless. "
              "Install a GUI backend (e.g. uv pip install pyqt5) for a live window; "
              "the run still logs to --out and saves --plot.")
        live = False

    first_sp = schedule[0][0]
    plotter = _LivePlot(first_sp, total_seconds, multi=True) if live else None
    logctx = RunLogger(out) if out else _NullLog()

    set_heater_attr = heater
    read_temp_attr = channel

    with connect(use_sim=use_sim) as lab, logctx as log:
        t0 = time.time()
        for seg_idx, (setpoint, seg_seconds) in enumerate(schedule):
            pid.setpoint = setpoint
            if seg_idx > 0:
                print(f"  → setpoint now {setpoint}°C")
            for _ in range(seg_seconds):
                now = time.time() - t0
                pv = float(getattr(lab, read_temp_attr))
                u = pid.update(pv=pv, dt=1.0)
                getattr(lab, set_heater_attr)(u)

                ts.append(now)
                temps.append(pv)
                us.append(u)
                sps.append(setpoint)
                log.log(now, lab.T1, lab.T2,
                        u if heater == "Q1" else 0.0, u if heater == "Q2" else 0.0)
                if plotter:
                    plotter.update(ts, temps, us, sps)
                if step_sleep:
                    time.sleep(step_sleep)
        try:
            lab.Q1(0); lab.Q2(0)
        except Exception:
            pass

    if plot:
        _save_plot(ts, temps, us, sps, plot, channel)
        print(f"plot -> {plot}")
    if plotter:
        plotter.finish()
    return {"t": ts, channel: temps, heater: us, "setpoint": sps}


class _NullLog:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def log(self, *a, **k):
        pass


def _have_gui_backend() -> bool:
    """True if matplotlib can open an interactive window in this environment."""
    import matplotlib

    backend = matplotlib.get_backend().lower()
    if backend == "agg":  # headless default → no window possible
        return False
    return True


class _LivePlot:
    """Minimal real-time two-panel plot (T vs setpoint, heater %)."""

    def __init__(self, setpoint: float, seconds: int, multi: bool = False):
        import matplotlib.pyplot as plt

        self.plt = plt
        plt.ion()
        self.fig, (self.ax, self.axu) = plt.subplots(
            2, 1, figsize=(9, 6), sharex=True, gridspec_kw={"height_ratios": [2, 1]}
        )
        (self.line,) = self.ax.plot([], [], "b-", lw=2, label="T1 measured")
        # setpoint trace: step line for multi-setpoint, dashed for single
        if multi:
            (self.spline,) = self.ax.plot([], [], "k--", lw=1.5, label="setpoint")
        else:
            self.ax.axhline(setpoint, color="k", ls="--", lw=1,
                            label=f"setpoint {setpoint:g}°C")
            self.spline = None
        (self.uline,) = self.axu.plot([], [], color="tab:orange", lw=1.5, label="heater %")
        self.ax.set_ylabel("T (°C)")
        self.ax.set_title("TCLab closed loop (live)")
        self.ax.legend(loc="lower right", fontsize=8)
        self.ax.grid(alpha=0.3)
        self.axu.set_ylabel("heater %")
        self.axu.set_xlabel("time (s)")
        self.axu.set_ylim(-5, 105)
        self.axu.grid(alpha=0.3)
        self.fig.tight_layout()

    def update(self, t, y, u, sps=None):
        self.line.set_data(t, y)
        if self.spline is not None and sps is not None:
            self.spline.set_data(t, sps)
        self.uline.set_data(t, u)
        self.ax.relim(); self.ax.autoscale_view()
        self.axu.set_xlim(0, max(10, t[-1]))
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        self.plt.pause(0.001)

    def finish(self):
        self.plt.ioff()
        self.plt.show()


def _save_plot(t, y, u, sps_or_setpoint, out, channel):
    import matplotlib.pyplot as plt

    fig, (ax, axu) = plt.subplots(2, 1, figsize=(9, 6), sharex=True,
                                  gridspec_kw={"height_ratios": [2, 1]})
    if isinstance(sps_or_setpoint, (int, float)):
        ax.axhline(sps_or_setpoint, color="k", ls="--", lw=1.5,
                   label=f"setpoint {sps_or_setpoint:g}°C")
    else:
        ax.plot(t, sps_or_setpoint, "k--", lw=1.5, label="setpoint")
    ax.plot(t, y, "b-", lw=2, label=channel)
    ax.set_ylabel("T (°C)"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax.set_title("TCLab closed-loop run (measured)")
    axu.plot(t, u, color="tab:orange", lw=1.5)
    axu.set_ylabel("heater %"); axu.set_xlabel("time (s)")
    axu.set_ylim(-5, 105); axu.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out, dpi=110)


def _parse_schedule(items: list[str]) -> list[tuple[float, int]]:
    """Parse ["50:400", "35:300", ...] into [(50.0, 400), (35.0, 300), ...]."""
    result = []
    for item in items:
        try:
            sp, dur = item.split(":")
            result.append((float(sp), int(dur)))
        except ValueError:
            raise argparse.ArgumentTypeError(
                f"schedule items must be 'setpoint:seconds', got {item!r}"
            )
    return result


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Run a PID closed loop on the TCLab (Step 3/4 HW).")
    sp_group = p.add_mutually_exclusive_group(required=True)
    sp_group.add_argument("--setpoint", type=float,
                          help="single target °C")
    sp_group.add_argument("--schedule", nargs="+", metavar="SP:SECS",
                          help='multi-setpoint schedule e.g. "50:400" "35:400" "40:300" "25:600"')
    p.add_argument("--seconds", type=int, default=600,
                   help="run duration (s) — only used with --setpoint, not --schedule")
    src = p.add_argument_group("gains (use --from-fit OR manual --kp/--ki/--kd)")
    src.add_argument("--from-fit", metavar="CSV", help="derive gains from a step-test CSV")
    src.add_argument("--tuning", choices=["imc", "zn"], default="imc", help="rule for --from-fit")
    src.add_argument("--lam", type=float, default=None, help="IMC λ (s)")
    src.add_argument("--kp", type=float, default=1.0)
    src.add_argument("--ki", type=float, default=0.0)
    src.add_argument("--kd", type=float, default=0.0)
    p.add_argument("--channel", default="T1")
    p.add_argument("--heater", default="Q1")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--sim", dest="sim", action="store_true", default=False,
                      help="use the offline simulator instead of the real board")
    mode.add_argument("--hardware", dest="sim", action="store_false", help="use the real board")
    p.add_argument("--live", action="store_true", help="show a live matplotlib window")
    p.add_argument("--out", default=None, help="CSV log path")
    p.add_argument("--plot", default=None, help="save a PNG of the run")
    args = p.parse_args(argv)

    # Validate the plot extension BEFORE a long hardware run, not after.
    if args.plot:
        ok = (".eps", ".jpeg", ".jpg", ".pdf", ".pgf", ".png", ".ps", ".raw",
              ".rgba", ".svg", ".svgz", ".tif", ".tiff", ".webp")
        if not args.plot.lower().endswith(ok):
            p.error(f"--plot must end in an image extension (e.g. .png); got {args.plot!r}")

    # _build_pid uses args.setpoint to configure the PID setpoint field.
    # For --schedule, we'll override it immediately in the schedule branch.
    if args.setpoint is None:
        args.setpoint = 0.0   # placeholder; overridden below

    if args.schedule:
        try:
            schedule = _parse_schedule(args.schedule)
        except argparse.ArgumentTypeError as e:
            p.error(str(e))
        total = sum(s for _, s in schedule)
        print(f"Schedule ({len(schedule)} segments, {total}s total, "
              f"{'sim' if args.sim else 'hardware'}):")
        for sp, dur in schedule:
            print(f"  {sp:g}°C for {dur}s")
        pid = _build_pid(args)
        # use first segment's setpoint for initial PID setpoint
        pid.setpoint = schedule[0][0]
        run_schedule(pid, schedule, use_sim=args.sim, live=args.live,
                     out=args.out, plot=args.plot, channel=args.channel,
                     heater=args.heater)
    else:
        pid = _build_pid(args)
        print(f"Running closed loop → setpoint {args.setpoint}°C for {args.seconds}s "
              f"({'sim' if args.sim else 'hardware'})...")
        run(pid, args.setpoint, args.seconds, use_sim=args.sim, live=args.live,
            out=args.out, plot=args.plot, channel=args.channel, heater=args.heater)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
