import numpy as np

from .group import FiniteGroup


class GroupFactory:
    @staticmethod
    def cyclic_group(n: int) -> FiniteGroup:
        # Generates the Cyclic group C_n
        elements = [f"z{i}" for i in range(n)]

        range_arr = np.arange(n)
        table = (range_arr + range_arr[:, None]) % n

        return FiniteGroup(elements, table)
