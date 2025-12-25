# Import the class first
# Import the factory second
from .factory import GroupFactory
from .group import FiniteGroup
from .representations.characters import compute_character_table

# This exposes them so users can do: from finite_groups import FiniteGroup
__all__ = ["FiniteGroup", "GroupFactory","compute_character_table"]
