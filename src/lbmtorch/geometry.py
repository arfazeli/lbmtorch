"""Differentiable level sets for obstacles. Solid where the level set is negative."""

from __future__ import annotations

import torch


def grid(
    shape: tuple[int, int], dtype: torch.dtype = torch.float64
) -> tuple[torch.Tensor, torch.Tensor]:
    """Node coordinates ``X, Y`` of shape ``shape`` in lattice units."""
    x = torch.arange(shape[0], dtype=dtype)
    y = torch.arange(shape[1], dtype=dtype)
    return torch.meshgrid(x, y, indexing="ij")


def circle(
    shape: tuple[int, int], center: tuple[float, float], radius: float | torch.Tensor
) -> torch.Tensor:
    """Signed distance to a circle. ``radius`` may be a tensor requiring grad."""
    X, Y = grid(shape)
    return torch.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2) - radius


def ellipse(
    shape: tuple[int, int],
    center: tuple[float, float],
    a: float | torch.Tensor,
    b: float | torch.Tensor,
) -> torch.Tensor:
    r"""Level set of an axis-aligned ellipse, semi-axes ``a`` (x) and ``b`` (y).

    Uses :math:`(\sqrt{(x/a)^2 + (y/b)^2} - 1)\,\min(a, b)`. This is not a true distance
    (its gradient magnitude ranges from :math:`\min(a,b)/\max(a,b)` to 1), but its zero set
    is exact, and :class:`~lbmtorch.InterpolatedBounceBack` only uses ratios of
    :math:`\phi` along a link, so the scale cancels to first order.
    """
    X, Y = grid(shape)
    r = torch.sqrt(((X - center[0]) / a) ** 2 + ((Y - center[1]) / b) ** 2)
    return (r - 1.0) * torch.minimum(torch.as_tensor(a), torch.as_tensor(b))


def channel_walls(shape: tuple[int, int]) -> torch.Tensor:
    """Boolean mask with solid rows at the bottom and top."""
    solid = torch.zeros(shape, dtype=torch.bool)
    solid[:, 0] = solid[:, -1] = True
    return solid
