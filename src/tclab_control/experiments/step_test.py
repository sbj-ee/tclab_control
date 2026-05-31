"""Step 1 + 2: apply a heater step, log the response to CSV.

Runs against the offline simulator by default so it works with no hardware:

    tclab-step --sim --seconds 120 --power 60 --out data/step_sim.csv

Switch to the real board with ``--hardware`` once it's connected.
"""

from __future__ import annotations

import argparse
import time

from tclab_control.lab import connect
from tclab_control.logger import RunLogger


def run(seconds: int, power: float, out: str, use_sim: bool, settle: int = 10) -> str:
    """Hold Q1=0 to settle, step Q1 to ``power``, log (t,T1,T2,Q1,Q2) each second."""
    with connect(use_sim=use_sim) as lab, RunLogger(out) as log:
        t0 = time.time()
        # Baseline (heater off) so the step has a clean pre-transition level.
        lab.Q1(0)
        for _ in range(settle):
            log.log(time.time() - t0, lab.T1, lab.T2, 0.0, 0.0)
            time.sleep(1)
        # Step.
        lab.Q1(power)
        for _ in range(seconds):
            log.log(time.time() - t0, lab.T1, lab.T2, power, 0.0)
            time.sleep(1)
        lab.Q1(0)
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="TCLab step test → CSV (Step 1/2).")
    p.add_argument("--seconds", type=int, default=120, help="step duration (s)")
    p.add_argument("--power", type=float, default=60.0, help="heater Q1 step (%)")
    p.add_argument("--out", default="data/step.csv", help="output CSV path")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--sim", dest="sim", action="store_true", default=True,
                      help="use the offline TCLabModel simulator (default)")
    mode.add_argument("--hardware", dest="sim", action="store_false",
                      help="use the real connected TCLab board")
    args = p.parse_args(argv)

    path = run(args.seconds, args.power, args.out, use_sim=args.sim)
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
