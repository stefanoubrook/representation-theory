import numpy as np
import pytest

from finite_groups import FiniteGroup


def test_no_identity():
    g = FiniteGroup([1, 2, 3], np.array([[1, 2, 2], [1, 2, 0], [2, 1, 2]]))

    with pytest.raises(ValueError, match="Invalid Group: No identity found"):
        g._check_identity()


def test_multiple_inverses():
    g = FiniteGroup(["e", "f"], np.array([[0, 1], [1, 1]]))

    with pytest.raises(ValueError, match="Right inverses are not unique"):
        g._check_inverses(0)


def test_correct_identity_inverses():
    # checks correct
    g = FiniteGroup(["e", 2, 3], np.array([[0, 1, 2], [1, 2, 0], [2, 0, 1]]))
    assert g._validate_group_axioms()
