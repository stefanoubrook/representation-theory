import numpy as np

from finite_groups import FiniteGroup, GroupFactory


def test_cyclic():
    assert GroupFactory.cyclic_group(3) == FiniteGroup(
        ["z0", "z1", "z2"], np.array([[0, 1, 2], [1, 2, 0], [2, 0, 1]])
    )
