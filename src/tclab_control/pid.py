"""A practical discrete PID controller (Step 3 of the roadmap).

Implements the bits that matter on a real plant:
  - derivative on measurement (avoids setpoint-change kick)
  - output clamping to actuator limits
  - clamping-based anti-windup (integral does not accumulate while saturated)

Sign convention: positive output drives the process value up (heater %).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PID:
    """Discrete PID with derivative-on-measurement and anti-windup.

    Args:
        kp, ki, kd: gains (ki, kd in per-second terms; dt supplied at update()).
        setpoint:   target process value.
        out_min, out_max: actuator limits (e.g. 0..100 for heater %).
    """

    kp: float
    ki: float = 0.0
    kd: float = 0.0
    setpoint: float = 0.0
    out_min: float = 0.0
    out_max: float = 100.0

    _integral: float = field(default=0.0, init=False, repr=False)
    _prev_pv: float | None = field(default=None, init=False, repr=False)

    def reset(self) -> None:
        """Clear integral and derivative history."""
        self._integral = 0.0
        self._prev_pv = None

    def update(self, pv: float, dt: float) -> float:
        """Compute the control output for one timestep.

        Args:
            pv: measured process value (e.g. temperature).
            dt: timestep in seconds (must be > 0).

        Returns:
            Control output clamped to [out_min, out_max].
        """
        if dt <= 0:
            raise ValueError("dt must be positive")

        error = self.setpoint - pv

        # Derivative on measurement (note the sign): -kd * d(pv)/dt
        if self._prev_pv is None:
            derivative = 0.0
        else:
            derivative = -self.kd * (pv - self._prev_pv) / dt
        self._prev_pv = pv

        # Tentative integral; commit only if it doesn't push us into saturation.
        new_integral = self._integral + self.ki * error * dt
        output_unclamped = self.kp * error + new_integral + derivative
        output = self._clamp(output_unclamped)

        # Anti-windup: keep the integral term only when not clamping it away.
        if output == output_unclamped:
            self._integral = new_integral

        return output

    def _clamp(self, x: float) -> float:
        return max(self.out_min, min(self.out_max, x))
