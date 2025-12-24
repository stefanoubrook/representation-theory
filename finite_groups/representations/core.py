import cmath

import numpy as np

from finite_groups.group import FiniteGroup


class Representation:
    def __init__(self, group: FiniteGroup, matrix_mapping: dict[object, np.ndarry]):
        # Group: an instance of FiniteGroup
        # matrix_mapping: a dictionary mapping group elements to matrices (Numpy arrays)
        self.group = group
        self.map = matrix_mapping
        self.degree = list(matrix_mapping.values())[0].shape[0]

    def character(self):
        # Returns the character of the representation as a class function
        return {g: m.trace() for g, m in self.map.items()}

    def is_irreducible(self) -> bool:
        # Uses Schur's lemma
        # A rep is irred iff <chi,chi> = 1

        chi = self.character()
        order = self.group.order
        inner_product = sum(abs(chi[g]) ** 2 for g in self.group.elements) / order

        return cmath.isclose(inner_product, 1.0)
