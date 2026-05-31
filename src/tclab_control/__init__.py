"""tclab_control — closed-loop control experiments on the TCLab thermal plant.

Maps to EE_Ref Chapter 4 (control systems): system identification, PID control,
and tuning, measured against theory. See the project README and the running doc
(tools/TCLAB.md) for the step-by-step roadmap.
"""

from tclab_control.fopdt import FOPDT, fit_fopdt, fit_step_csv
from tclab_control.pid import PID
from tclab_control.simulate import StepMetrics, simulate_closed_loop, step_metrics
from tclab_control.tuning import Gains, imc_pid, ziegler_nichols

__all__ = [
    "PID",
    "FOPDT",
    "fit_fopdt",
    "fit_step_csv",
    "Gains",
    "imc_pid",
    "ziegler_nichols",
    "StepMetrics",
    "simulate_closed_loop",
    "step_metrics",
]
__version__ = "0.1.0"
