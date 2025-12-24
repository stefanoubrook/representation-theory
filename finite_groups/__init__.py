# Import the class first
# Import the factory second
from .factory import GroupFactory
from .group import FiniteGroup

# This exposes them so users can do: from finite_groups import FiniteGroup
__all__ = ["FiniteGroup", "GroupFactory"]
