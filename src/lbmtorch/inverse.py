"""Small helpers for inverse problems: parameterisations and sparse observations."""

from __future__ import annotations

import math

import torch


class PositiveTau:
    r"""Unconstrained parameterisation of a relaxation time :math:`\tau > 1/2`.

    Optimisers work on the raw scalar :math:`\theta`; the solver sees
    :math:`\tau = 1/2 + \mathrm{softplus}(\theta)`, so no step can cross the stability
    limit. Pass it as ``tau`` to :class:`~lbmtorch.BGK`, :class:`~lbmtorch.TRT` or
    :class:`~lbmtorch.MRT`: it supports the arithmetic they use and is evaluated afresh at
    every step. For other operators pass ``tau()`` instead.
    """

    def __init__(self, tau0: float):
        if tau0 <= 0.5:
            raise ValueError("tau0 must exceed 1/2")
        self.theta = torch.tensor(
            _inv_softplus(tau0 - 0.5), dtype=torch.float64, requires_grad=True
        )

    def __call__(self) -> torch.Tensor:
        r"""Return the current :math:`\tau` as a differentiable tensor."""
        return 0.5 + torch.nn.functional.softplus(self.theta)

    def __rtruediv__(self, other):
        return other / self()

    def __sub__(self, other):
        return self() - other

    def __mul__(self, other):
        return self() * other

    __rmul__ = __mul__

    def __float__(self) -> float:
        return float(self().detach())


def _inv_softplus(y: float) -> float:
    return math.log(math.expm1(y))


def sparse_indices(shape: tuple[int, ...], fraction: float, seed: int = 0) -> torch.Tensor:
    """Random flat indices covering ``fraction`` of the nodes, for sparse observations."""
    n = math.prod(shape)
    g = torch.Generator().manual_seed(seed)
    return torch.randperm(n, generator=g)[: max(1, int(fraction * n))]


def observe(field: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Gather the values of a ``(D, *spatial)`` field at flat spatial indices ``idx``."""
    return field.flatten(1)[:, idx]
