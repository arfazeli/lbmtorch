"""Conversions between relaxation time and viscosity in lattice units."""

from __future__ import annotations

import torch


def viscosity_from_tau(tau: float | torch.Tensor, cs2: float = 1.0 / 3.0) -> float | torch.Tensor:
    r"""Kinematic viscosity :math:`\nu = c_s^2 (\tau - 1/2)`."""
    return cs2 * (tau - 0.5)


def tau_from_viscosity(nu: float | torch.Tensor, cs2: float = 1.0 / 3.0) -> float | torch.Tensor:
    r"""Relaxation time :math:`\tau = \nu / c_s^2 + 1/2`."""
    return nu / cs2 + 0.5
