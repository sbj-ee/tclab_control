"""tclab_control — closed-loop control experiments on the TCLab thermal plant.

Maps to EE_Ref Chapter 4 (control systems): system identification, PID control,
and tuning, measured against theory. See the project README and the running doc
(tools/TCLAB.md) for the step-by-step roadmap.
"""

from tclab_control.fopdt import FOPDT, fit_fopdt, fit_step_csv
from tclab_control.pid import PID

__all__ = ["PID", "FOPDT", "fit_fopdt", "fit_step_csv"]
__version__ = "0.1.0"
