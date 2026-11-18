import pytest
import torch

from lbmtorch import D2Q9
from lbmtorch.lattice.base import opposite_indices


def test_d2q9_shapes_and_weights():
    assert D2Q9.Q == 9 and D2Q9.D == 2
    assert torch.isclose(D2Q9.w.sum(), torch.tensor(1.0, dtype=D2Q9.w.dtype))


def test_opposites_are_negatives():
    assert torch.equal(D2Q9.c[D2Q9.opp], -D2Q9.c)


def test_lattice_isotropy():
    # sum_q w_q c_qa c_qb = cs2 * delta_ab
    c = D2Q9.c.double()
    second = torch.einsum("q,qa,qb->ab", D2Q9.w, c, c)
    assert torch.allclose(second, D2Q9.cs2 * torch.eye(2, dtype=torch.float64))


def test_opposite_indices_rejects_asymmetric():
    with pytest.raises(ValueError):
        opposite_indices(torch.tensor([[0, 0], [1, 0]]))
