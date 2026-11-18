"""Generic lattice container."""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import torch


@dataclass(frozen=True)
class Lattice:
    """A DdQq discrete velocity set.

    Populations are stored with the velocity index first, ``f.shape == (Q, *spatial)``.

    Parameters
    ----------
    name
        Human-readable identifier, e.g. ``"D2Q9"``.
    c
        Integer velocity vectors, shape ``(Q, D)``.
    w
        Lattice weights, shape ``(Q,)``. Must sum to one.
    opp
        ``opp[q]`` is the index of the velocity opposite to ``c[q]``.
    cs2
        Squared lattice speed of sound. ``1/3`` for all standard lattices.

    Attributes
    ----------
    shifts
        ``c`` as nested Python tuples, for ``torch.roll`` without a device sync.
    opp_idx
        ``opp`` as a Python tuple.
    """

    name: str
    c: torch.Tensor
    w: torch.Tensor
    opp: torch.Tensor
    cs2: float = 1.0 / 3.0
    shifts: tuple[tuple[int, ...], ...] = field(init=False, repr=False, compare=False)
    opp_idx: tuple[int, ...] = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        object.__setattr__(self, "shifts", tuple(map(tuple, self.c.tolist())))
        object.__setattr__(self, "opp_idx", tuple(self.opp.tolist()))

    @property
    def Q(self) -> int:  # noqa: N802 - conventional LBM symbol
        """Number of discrete velocities."""
        return self.c.shape[0]

    @property
    def D(self) -> int:  # noqa: N802 - conventional LBM symbol
        """Spatial dimension."""
        return self.c.shape[1]

    def to(
        self, device: torch.device | str | None = None, dtype: torch.dtype | None = None
    ) -> Lattice:
        """Return a copy with tensors moved to ``device`` and weights cast to ``dtype``."""
        return replace(
            self,
            c=self.c.to(device=device),
            w=self.w.to(device=device, dtype=dtype),
            opp=self.opp.to(device=device),
        )

    def w_bcast(self, ndim: int) -> torch.Tensor:
        """Weights reshaped to ``(Q, 1, ..., 1)`` for broadcasting over ``ndim`` spatial axes."""
        return self.w.view(self.Q, *([1] * ndim))


def opposite_indices(c: torch.Tensor) -> torch.Tensor:
    """Compute ``opp`` from the velocity vectors by matching ``c[q]`` with ``-c[q]``."""
    match = (c[:, None, :] == -c[None, :, :]).all(-1)
    if not bool(match.any(1).all()):
        raise ValueError("velocity set is not symmetric: some c[q] has no opposite")
    return match.long().argmax(1)
