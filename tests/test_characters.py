import numpy as np

from finite_groups.factory import GroupFactory
from finite_groups.representations.characters import compute_character_table


def test_s3_character_table():
    group = GroupFactory.symmetric_group(3)
    order = group.order

    table, classes = compute_character_table(group)

    assert len(classes) == 3
    assert table.shape == (3, 3)

    dimensions = sorted([abs(row[0]) for row in table])
    assert np.allclose(dimensions, [1.0, 1.0, 2.0])

    # Check row orthogonality
    class_sizes = [len(c) for c in classes]
    for i in range(len(table)):
        for j in range(len(table)):
            inner_product = (
                sum(
                    class_sizes[k] * table[i, k] * np.conj(table[j, k])
                    for k in range(3)
                )
                / order
            )

            expected = 1.0 if i == j else 0.0
            assert np.isclose(inner_product, expected, atol=1e-7)


def test_cyclic_group_table():
    n = 4
    group = GroupFactory.cyclic_group(n)
    table, classes = compute_character_table(group)

    assert len(classes) == n
    for row in table:
        assert np.isclose(abs(row[0]), 1.0)


print(compute_character_table(GroupFactory.symmetric_group(3)))
