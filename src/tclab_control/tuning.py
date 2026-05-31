"""PID tuning rules derived from a fitted FOPDT model (Step 4).

Given a FOPDT model G(s) = K·e^(-θs)/(τs + 1), compute PID gains via classic
rules. Gains are returned as a :class:`Gains` in the same convention the ``PID``
class expects: ``ki`` and ``kd`` are in *per-second* terms, i.e.

    ki = Kp / Ti      kd = Kp * Td

so the control law is  u = Kp·e + ki·∫e dt + kd·d/dt(...).

Implemented rules:
  - ``imc_pid``        — IMC (lambda tuning); ``lam`` trades speed vs. robustness.
  - ``ziegler_nichols``— Z-N open-loop (reaction-curve) PID, a classic baseline.
"""

from __future__ import annotations

from dataclasses import dataclass

from tclab_control.fopdt import FOPDT
from tclab_control.pid import PID


@dataclass
class Gains:
    """PID gains in the PID class's convention (ki, kd are per-second)."""

    kp: float
    ki: float          # = kp / Ti
    kd: float          # = kp * Td
    Ti: float          # integral time (s)
    Td: float          # derivative time (s)
    method: str = ""

    def to_pid(self, setpoint: float = 0.0, out_min: float = 0.0,
               out_max: float = 100.0) -> PID:
        """Build a configured PID controller from these gains."""
        return PID(kp=self.kp, ki=self.ki, kd=self.kd, setpoint=setpoint,
                   out_min=out_min, out_max=out_max)


def _gains(kp: float, Ti: float, Td: float, method: str) -> Gains:
    ki = kp / Ti if Ti > 0 else 0.0
    kd = kp * Td
    return Gains(kp=kp, ki=ki, kd=kd, Ti=Ti, Td=Td, method=method)


def imc_pid(model: FOPDT, lam: float | None = None) -> Gains:
    """IMC (lambda) tuning for a FOPDT plant → PID.

    Uses the common IMC-PID correlation (improved/SIMC-style) with a first-order
    Padé approximation of the dead time:

        Kp = (1/K) · (τ + θ/2) / (λ + θ/2)
        Ti = τ + θ/2
        Td = (τ·θ) / (2τ + θ)

    Args:
        model: fitted FOPDT.
        lam: closed-loop time constant λ (s). Larger = slower/more robust.
             Defaults to max(τ, 8θ)-style moderate choice: λ = max(0.5·τ, θ).
    """
    K, tau, theta = model.K, model.tau, model.theta
    if lam is None:
        lam = max(0.5 * tau, theta, 1e-3)
    Ti = tau + theta / 2.0
    Td = (tau * theta) / (2.0 * tau + theta) if (2.0 * tau + theta) > 0 else 0.0
    kp = (1.0 / K) * Ti / (lam + theta / 2.0)
    return _gains(kp, Ti, Td, method=f"IMC(λ={lam:.3g})")


def ziegler_nichols(model: FOPDT) -> Gains:
    """Ziegler–Nichols open-loop (reaction curve) PID tuning for a FOPDT plant.

        Kp = 1.2·τ / (K·θ),  Ti = 2·θ,  Td = 0.5·θ
    """
    K, tau, theta = model.K, model.tau, model.theta
    theta = max(theta, 1e-3)  # Z-N is undefined at zero dead time
    kp = 1.2 * tau / (K * theta)
    Ti = 2.0 * theta
    Td = 0.5 * theta
    return _gains(kp, Ti, Td, method="Ziegler-Nichols")
